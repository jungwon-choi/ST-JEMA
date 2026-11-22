import os
import csv
import numpy as np
from sklearn import metrics
from scipy import stats


class LoggerSTAGIN:
    """Metrics logger for classification and regression tasks with k-fold support."""

    def __init__(self, k_fold=None, num_classes=None):
        super().__init__()
        self.k_fold = k_fold
        self.num_classes = num_classes
        self.initialize(k=None)

    def __call__(self, **kwargs):
        if len(kwargs) == 0:
            self.get()
        else:
            self.add(**kwargs)

    def _initialize_metric_dict(self):
        return {'pred': [], 'true': [], 'prob': []}

    def initialize(self, k=None):
        if self.k_fold is None:
            self.samples = self._initialize_metric_dict()
        else:
            if k is None:
                self.samples = {}
                for _k in range(self.k_fold):
                    self.samples[_k] = self._initialize_metric_dict()
            else:
                self.samples[k] = self._initialize_metric_dict()

    def add(self, k=None, **kwargs):
        if self.k_fold is None:
            for sample, value in kwargs.items():
                self.samples[sample].append(value)
        else:
            assert k in list(range(self.k_fold))
            for sample, value in kwargs.items():
                self.samples[k][sample].append(value)

    def get(self, k=None, initialize=False):
        if self.k_fold is None:
            true = np.concatenate(self.samples['true'])
            pred = np.concatenate(self.samples['pred'])
            prob = np.concatenate(self.samples['prob'])
        else:
            if k is None:
                true, pred, prob = {}, {}, {}
                for k in range(self.k_fold):
                    true[k] = np.concatenate(self.samples[k]['true'])
                    pred[k] = np.concatenate(self.samples[k]['pred'])
                    prob[k] = np.concatenate(self.samples[k]['prob'])
            else:
                true = np.concatenate(self.samples[k]['true'])
                pred = np.concatenate(self.samples[k]['pred'])
                prob = np.concatenate(self.samples[k]['prob'])

        if initialize:
            self.initialize(k)

        return dict(true=true, pred=pred, prob=prob)

    def evaluate(self, k=None, initialize=False, option='mean', prefix=None):
        samples = self.get(k)

        if self.num_classes == 1:
            # Regression
            if not self.k_fold is None and k is None:
                aggregate = np.mean if option == 'mean' else np.std
                for _k in range(self.k_fold):
                    samples['true'][_k] = np.asarray(samples['true'][_k]).ravel()
                    samples['pred'][_k] = np.asarray(samples['pred'][_k]).ravel()

                r2 = aggregate([metrics.r2_score(samples['true'][k], samples['pred'][k]) for k in range(self.k_fold)])
                mse = aggregate([metrics.mean_squared_error(samples['true'][k], samples['pred'][k]) for k in range(self.k_fold)])
                mae = aggregate([metrics.mean_absolute_error(samples['true'][k], samples['pred'][k]) for k in range(self.k_fold)])
                rmse = aggregate([np.sqrt(metrics.mean_squared_error(samples['true'][k], samples['pred'][k])) for k in range(self.k_fold)])
                pearson = aggregate([stats.pearsonr(samples['true'][k], samples['pred'][k])[0] for k in range(self.k_fold)])
                spearman = aggregate([stats.spearmanr(samples['true'][k], samples['pred'][k])[0] for k in range(self.k_fold)])
            else:
                samples['true'] = np.asarray(samples['true']).ravel()
                samples['pred'] = np.asarray(samples['pred']).ravel()
                r2 = metrics.r2_score(samples['true'], samples['pred'])
                mse = metrics.mean_squared_error(samples['true'], samples['pred'])
                mae = metrics.mean_absolute_error(samples['true'], samples['pred'])
                rmse = np.sqrt(mse)
                pearson = stats.pearsonr(samples['true'], samples['pred'])[0]
                spearman = stats.spearmanr(samples['true'], samples['pred'])[0]

            if initialize:
                self.initialize(k)

            result = dict(r2=r2, mse=mse, mae=mae, rmse=rmse, pearson=pearson, spearman=spearman)
            if prefix is not None:
                return {f'{prefix}_{k}': v for k, v in result.items()}
            return result

        elif self.num_classes > 1:
            # Classification
            if not self.k_fold is None and k is None:
                aggregate = np.mean if option == 'mean' else np.std

                accuracy = aggregate([metrics.accuracy_score(samples['true'][k], samples['pred'][k]) for k in range(self.k_fold)])
                balanced_accuracy = aggregate([metrics.balanced_accuracy_score(samples['true'][k], samples['pred'][k]) for k in range(self.k_fold)])
                precision = aggregate([metrics.precision_score(samples['true'][k], samples['pred'][k], average='binary' if self.num_classes == 2 else 'micro') for k in range(self.k_fold)])
                recall = aggregate([metrics.recall_score(samples['true'][k], samples['pred'][k], average='binary' if self.num_classes == 2 else 'micro') for k in range(self.k_fold)])
                f1 = aggregate([metrics.f1_score(samples['true'][k], samples['pred'][k], average='binary' if self.num_classes == 2 else 'macro') for k in range(self.k_fold)])
                if self.num_classes == 2:
                    roc_auc = aggregate([metrics.roc_auc_score(samples['true'][k], samples['prob'][k][:, 1]) for k in range(self.k_fold)])
                    specificity = aggregate([metrics.recall_score(samples['true'][k], samples['pred'][k], pos_label=0) for k in range(self.k_fold)])
                    auprc = aggregate([metrics.average_precision_score(samples['true'][k], samples['prob'][k][:, 1]) for k in range(self.k_fold)])
                else:
                    roc_auc = aggregate([metrics.roc_auc_score(samples['true'][k], samples['prob'][k], average='macro', multi_class='ovr') for k in range(self.k_fold)])
                    specificity = None
                    auprc = None
            else:
                accuracy = metrics.accuracy_score(samples['true'], samples['pred'])
                balanced_accuracy = metrics.balanced_accuracy_score(samples['true'], samples['pred'])
                precision = metrics.precision_score(samples['true'], samples['pred'], average='binary' if self.num_classes == 2 else 'micro')
                recall = metrics.recall_score(samples['true'], samples['pred'], average='binary' if self.num_classes == 2 else 'micro')
                f1 = metrics.f1_score(samples['true'], samples['pred'], average='binary' if self.num_classes == 2 else 'macro')
                if self.num_classes == 2:
                    roc_auc = metrics.roc_auc_score(samples['true'], samples['prob'][:, 1])
                    specificity = metrics.recall_score(samples['true'], samples['pred'], pos_label=0)
                    auprc = metrics.average_precision_score(samples['true'], samples['prob'][:, 1])
                else:
                    roc_auc = metrics.roc_auc_score(samples['true'], samples['prob'], average='macro', multi_class='ovr')
                    specificity = None
                    auprc = None

            if initialize:
                self.initialize(k)

            result = dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1, roc_auc=roc_auc, balanced_accuracy=balanced_accuracy)
            if self.num_classes == 2:
                result['specificity'] = specificity
                result['auprc'] = auprc

            if prefix is not None:
                return {f'{prefix}_{k}': v for k, v in result.items()}
            return result

        else:
            raise ValueError(f"number of classes must be positive integer, but got {self.num_classes}")

    def save_predictions(self, targetdir, k=None, mode='test', epoch=None):
        """Save predictions, labels, and probabilities."""
        samples = self.get(k)
        suffix = '' if epoch is None else f'_ep{epoch}'
        np.save(os.path.join(targetdir, f'{mode}_pred{suffix}.npy'), samples['pred'])
        np.save(os.path.join(targetdir, f'{mode}_label{suffix}.npy'), samples['true'])
        np.save(os.path.join(targetdir, f'{mode}_prob{suffix}.npy'), samples['prob'])

    def to_csv(self, targetdir, k=None, initialize=False, epoch=None):
        metric_dict = self.evaluate(k, initialize)
        file = 'metrics.csv' if epoch is None else f'metrics_{epoch}ep.csv'
        append = os.path.isfile(os.path.join(targetdir, file))

        with open(os.path.join(targetdir, file), 'a', newline='') as f:
            writer = csv.writer(f)
            if not append:
                writer.writerow(['fold'] + [str(key) for key in metric_dict.keys()])
            writer.writerow([str(k)] + [str(value) for value in metric_dict.values()])
            if k is None:
                writer.writerow([str(k)] + list(self.evaluate(k, initialize, 'std').values()))
