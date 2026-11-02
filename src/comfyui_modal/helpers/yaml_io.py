"""YAML reads whose errors never echo file contents."""

from pathlib import Path

import yaml


def read_yaml(path, error, *, missing=None):
    """Parse a YAML file, raising RuntimeError(error) on failure.

    Returns ``missing`` for an absent file when it is given.
    """
    try:
        return yaml.safe_load(Path(path).read_text())
    except FileNotFoundError:
        if missing is not None:
            return missing
        raise RuntimeError(error) from None
    except (OSError, UnicodeError, yaml.YAMLError):
        # YAML exceptions include source lines, which could contain a key.
        raise RuntimeError(error) from None
