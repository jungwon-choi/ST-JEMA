"""Dataset loading for brain fMRI timeseries with k-fold cross-validation."""

import os
import pickle
import logging
from random import shuffle, randrange
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split


class StandardScaler:
    """Standard scaler for regression targets."""
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = np.mean(X)
        self.std_ = np.std(X)

    def transform(self, X):
        return list((np.array(X) - self.mean_) / self.std_)

    def inverse_transform(self, X):
        if isinstance(X, np.ndarray):
            return np.array(X) * self.std_ + self.mean_
        elif isinstance(X, torch.Tensor):
            return X * self.std_ + self.mean_
        return list(np.array(X) * self.std_ + self.mean_)


class DatasetRest(Dataset):
    """Brain fMRI resting-state dataset with cross-validation support.

    Expected data format:
        {sourcedir}/{dataset}/{dataset}_roi-{roi}.pkl  - dict: subject_id -> ndarray[T, N]
        {sourcedir}/{dataset}/{dataset}.csv             - CSV with subject_id and target columns
    """
    def __init__(
        self,
        sourcedir,
        dataset,
        roi,
        target_feature,
        k_fold=None,
        regression: bool = False,
        dynamic_length: Optional[int] = None,
        ssl: bool = False,
        same_data_ssl_setting: bool = False,
        val_ratio: float = 0.1,
        ssl_test_ratio: float = 0.1,
        num_finetune_subsample: Optional[int] = None,
        eval_full_length: bool = False,
    ):
        super().__init__()

        sourcedir = os.path.join(sourcedir, dataset)

        self.filename = dataset
        self.timeseries_filename = dataset + f'_roi-{roi}'
        assert os.path.isfile(os.path.join(sourcedir, f'{self.timeseries_filename}.pkl')), \
            f'Timeseries file not found: {self.timeseries_filename}.pkl'

        # Load timeseries
        with open(os.path.join(sourcedir, f'{self.timeseries_filename}.pkl'), 'rb') as f:
            self.timeseries_dict = pickle.load(f)

        # Load behavioral data
        behavioral_df = pd.read_csv(os.path.join(sourcedir, f'{self.filename}.csv'))
        behavioral_df = behavioral_df.set_index('subject_id')
        behavioral_df.index = behavioral_df.index.map(str)
        behavioral_df.dropna(subset=target_feature, inplace=True)

        # Dataset-specific filtering (for reproducibility)
        if dataset == 'adhd200-rest' and target_feature == 'DX':
            behavioral_df = behavioral_df[behavioral_df[target_feature].isin(['0', '1', '2', '3'])].astype({target_feature: int})
            behavioral_df[target_feature] = behavioral_df[target_feature].map(lambda x: 0 if x == 0 else 1)
        elif dataset == 'abcd-rest' and target_feature == 'sex':
            behavioral_df = behavioral_df[behavioral_df[target_feature].isin(['M', 'F'])]
        elif dataset == 'hcpa-rest' and target_feature == 'interview_age':
            behavioral_df = behavioral_df[behavioral_df[target_feature] < 1200]

        # Filter to subjects with timeseries data
        self.timeseries_dict = {k: v for k, v in self.timeseries_dict.items() if k in behavioral_df.index}

        # Dataset info
        self.num_nodes = list(self.timeseries_dict.values())[0].shape[1]
        timeseries_lengths = [ts.shape[0] for ts in self.timeseries_dict.values()]
        self.min_timepoints = min(timeseries_lengths)
        logging.info(f'Timeseries: min {min(timeseries_lengths)}, max {max(timeseries_lengths)}, avg {np.mean(timeseries_lengths):.0f}')

        if dynamic_length is not None:
            assert dynamic_length <= self.min_timepoints, \
                f'dynamic_length {dynamic_length} > min_timepoints {self.min_timepoints}'

        self.dynamic_length = dynamic_length
        # If True, validation/test samples use the full time series instead of a random crop
        self.eval_full_length = eval_full_length
        self.train = True
        self.full_subject_list = list(self.timeseries_dict.keys())

        # Label info
        if regression and target_feature == 'interview_age' and dataset in ['hcpa-rest', 'hcpd-rest', 'hcpep-rest']:
            behavioral_df[target_feature] = behavioral_df[target_feature] / 12.0  # months -> years
            logging.info(f'Converting months to years for {dataset}')

        self.behavioral_dict = behavioral_df[target_feature].to_dict()
        self.full_label_list = np.array([self.behavioral_dict[s] for s in self.full_subject_list])
        self.num_classes = 1 if regression else len(behavioral_df[target_feature].unique())
        logging.info(f'Subjects: {len(self.full_subject_list)}, classes: {self.num_classes}')

        # K-fold setup
        if isinstance(k_fold, int):
            self.folds = list(range(k_fold))
            if k_fold > 1:
                self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0) if self.num_classes > 1 \
                    else KFold(k_fold, shuffle=True, random_state=0)
            else:
                self.k_fold = None
                self.subject_list = self.full_subject_list
        elif isinstance(k_fold, str):
            self.folds = list(behavioral_df[k_fold].unique())
            self.k_fold = {}
            for fold in self.folds:
                self.k_fold[fold] = [
                    [behavioral_df.index.to_list().index(i) for i in behavioral_df.loc[behavioral_df[k_fold] != fold].index],
                    [behavioral_df.index.to_list().index(i) for i in behavioral_df.loc[behavioral_df[k_fold] == fold].index],
                ]

        self.k = None

        # Label encoder
        if self.num_classes > 1:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.full_label_list)
        else:
            self.label_encoder = StandardScaler()
            self.label_encoder.fit(self.full_label_list)

        self.val_ratio = val_ratio
        self.ssl = ssl
        self.ssl_test_ratio = ssl_test_ratio
        self.same_data_ssl_setting = same_data_ssl_setting
        self.num_finetune_subsample = num_finetune_subsample

    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)

    def set_fold(self, fold, mode='train', verbose=False):
        if not self.k_fold:
            return

        self.k = fold

        if not self.same_data_ssl_setting:
            # Standard k-fold split
            if isinstance(fold, int):
                if self.num_classes > 1:
                    train_idx, test_idx = list(self.k_fold.split(self.full_subject_list, self.label_encoder.transform(self.full_label_list)))[fold]
                else:
                    train_idx, test_idx = list(self.k_fold.split(self.full_subject_list))[fold]
            elif isinstance(fold, str):
                train_idx, test_idx = self.k_fold[fold]

            # Train/val split
            if self.val_ratio > 0:
                num_val = round(len(self.full_subject_list) * self.val_ratio)
                if self.num_classes > 1:
                    train_idx, val_idx = train_test_split(
                        train_idx, test_size=num_val, shuffle=True, random_state=0,
                        stratify=self.label_encoder.transform(self.full_label_list)[train_idx],
                    )
                else:
                    train_idx, val_idx = train_test_split(train_idx, test_size=num_val, shuffle=True, random_state=0)
            else:
                val_idx = []

        else:
            # UKB SSL setting: SSL uses fold-0 train split, finetune uses fold-0 test split
            if isinstance(fold, int):
                if self.num_classes > 1:
                    train_idx, test_idx = list(self.k_fold.split(self.full_subject_list, self.label_encoder.transform(self.full_label_list)))[0]
                else:
                    train_idx, test_idx = list(self.k_fold.split(self.full_subject_list))[0]
            elif isinstance(fold, str):
                train_idx, test_idx = self.k_fold[fold]

            if self.num_finetune_subsample is not None:
                rs = np.random.RandomState(0)
                test_subsample_idx = rs.choice(test_idx, self.num_finetune_subsample, replace=False)
                train_idx = np.concatenate((train_idx, [idx for idx in test_idx if idx not in test_subsample_idx]))
                test_idx = test_subsample_idx

            if self.ssl:
                ssl_test_ratio = int(self.ssl_test_ratio) if self.ssl_test_ratio == 0 else self.ssl_test_ratio
                ssl_train_idx, ssl_test_idx = train_test_split(train_idx, test_size=ssl_test_ratio, shuffle=True, random_state=0)
                num_val = round(len(train_idx) * self.val_ratio)
                ssl_train_idx, ssl_val_idx = train_test_split(ssl_train_idx, test_size=num_val, shuffle=True, random_state=0)
                train_idx, val_idx, test_idx = ssl_train_idx, ssl_val_idx, ssl_test_idx
            else:
                test_idx_subjects = [self.full_subject_list[i] for i in test_idx]
                test_idx_labels = [self.full_label_list[i] for i in test_idx]
                num_val = round(len(test_idx_subjects) * self.val_ratio)
                if self.num_classes > 1:
                    ft_train_idx, ft_test_idx = list(self.k_fold.split(test_idx_subjects, self.label_encoder.transform(test_idx_labels)))[fold]
                    ft_train_idx, ft_val_idx = train_test_split(
                        ft_train_idx, test_size=num_val, shuffle=True, random_state=0,
                        stratify=self.label_encoder.transform(test_idx_labels)[ft_train_idx],
                    )
                else:
                    ft_train_idx, ft_test_idx = list(self.k_fold.split(test_idx_subjects))[fold]
                    ft_train_idx, ft_val_idx = train_test_split(ft_train_idx, test_size=num_val, shuffle=True, random_state=0)
                train_idx, val_idx, test_idx = ft_train_idx, ft_val_idx, ft_test_idx

        if verbose:
            logging.info(f'Fold {fold}: train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)}')

        # Regression targets are standardized with statistics of the training split only
        if self.num_classes == 1:
            self.label_encoder.fit(self.full_label_list[np.asarray(train_idx, dtype=int)])

        if mode == 'train':
            shuffle(train_idx)
            self.train = True
            self.subject_list = [self.full_subject_list[i] for i in train_idx]
        elif mode == 'val':
            self.train = False
            self.subject_list = [self.full_subject_list[i] for i in val_idx]
        elif mode == 'test':
            self.train = False
            self.subject_list = [self.full_subject_list[i] for i in test_idx]
        else:
            raise ValueError('mode should be train/val/test')

    def __getitem__(self, idx):
        subject = self.subject_list[idx]

        timeseries = self.timeseries_dict[subject]
        # Z-score normalization per ROI
        timeseries_mean = np.mean(timeseries, axis=0, keepdims=True)
        timeseries_std = np.std(timeseries, axis=0, keepdims=True)
        timeseries = (timeseries - timeseries_mean) / (timeseries_std + 1e-9)

        # Random temporal crop (evaluation uses the full time series when eval_full_length is set)
        if self.dynamic_length and (self.train or not self.eval_full_length):
            sampling_init = randrange(0, len(timeseries) - self.dynamic_length + 1)
            timeseries = timeseries[sampling_init:sampling_init + self.dynamic_length]

        label = self.behavioral_dict[subject]
        label = self.label_encoder.transform([label])
        return {
            'id': subject,
            'timeseries': torch.as_tensor(timeseries, dtype=torch.float32),
            'label': torch.as_tensor(label, dtype=(torch.float32 if self.num_classes == 1 else torch.int64)).squeeze(),
        }


def build_dataloaders(cfg, root="./data"):
    """Build dataset and dataloaders from config."""
    dataset = DatasetRest(
        sourcedir=root,
        dataset=cfg.dataset,
        roi=cfg.roi,
        target_feature=cfg.target_feature,
        k_fold=cfg.k_fold,
        regression=cfg.regression,
        dynamic_length=cfg.dynamic_length,
        ssl=cfg.ssl,
        same_data_ssl_setting=cfg.get('same_data_ssl_setting', False),
        val_ratio=cfg.val_ratio,
        ssl_test_ratio=cfg.get('ssl_test_ratio', 0.1),
        num_finetune_subsample=cfg.get('num_finetune_subsample', None),
        eval_full_length=cfg.get('eval_full_length', False),
    )

    dataset.set_fold(cfg.train_k, mode='train', verbose=True)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.minibatch_size,
        shuffle=True,
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True,
        drop_last=cfg.get('drop_last', True),
    )
    eval_dataloader = DataLoader(
        dataset,
        batch_size=cfg.get('eval_minibatch_size', 1),
        shuffle=False,
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True,
    )

    return dataset, dataloader, eval_dataloader
