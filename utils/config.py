import yaml


class ConfigDict(dict):
    """Dictionary subclass that allows attribute-style access (cfg.key)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")

    def update(self, other=None, **kwargs):
        if other:
            for k, v in (other.items() if hasattr(other, 'items') else other):
                if v is not None:
                    self[k] = v
        for k, v in kwargs.items():
            if v is not None:
                self[k] = v


def load_config(path):
    """Load YAML config file and return ConfigDict."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return ConfigDict(cfg)
