"""YAML-based configuration utilities.

The resolution order is:
    1. ``cfg/default.yaml``  (shipped defaults)
    2. Task-specific YAML    (e.g. ``cfg/car_quality.yaml``)
    3. CLI / API overrides   (plain ``key=value`` pairs or a dict)

Usage::

    from classify.utils.config import get_cfg

    # Load defaults only
    cfg = get_cfg()

    # Override with a task YAML
    cfg = get_cfg("cfg/car_quality.yaml")

    # Override with a dict (e.g. from parsed CLI args)
    cfg = get_cfg(overrides={"max_epochs": 100, "data_dir": "/data/cars"})
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Path to the shipped default config — lives two levels above this file.
_DEFAULT_CFG_PATH: Path = Path(__file__).parents[2] / "cfg" / "default.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary of parsed YAML contents.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_default_cfg() -> dict[str, Any]:
    """Return a fresh copy of the default configuration.

    Returns:
        Default configuration as a plain dict.
    """
    return load_yaml(_DEFAULT_CFG_PATH)


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge an arbitrary number of config dicts left-to-right.

    Later dicts take precedence over earlier ones; ``None`` values in an
    override are *ignored* so they do not clobber existing defaults.

    Args:
        *configs: Config dicts to merge in order.

    Returns:
        Merged configuration dict.
    """
    result: dict[str, Any] = {}
    for cfg in configs:
        result.update({k: v for k, v in copy.deepcopy(cfg).items() if v is not None})
    return result


def get_cfg(
    overrides: dict[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Resolve a full configuration by layering overrides on top of defaults.

    Resolution order:

    1. ``cfg/default.yaml``  (shipped defaults)
    2. Task YAML specified by a path argument **or** the ``cfg`` key inside a
       dict override (e.g. ``{"cfg": "cfg/car_quality.yaml", "num_classes": 4}``)
    3. Remaining key/value overrides from the dict

    Args:
        overrides: One of:

            * ``None``                — return defaults unchanged.
            * ``str`` / ``Path``      — path to a task-specific YAML.
            * ``dict``                — key/value overrides.  If the dict
              contains a ``"cfg"`` key whose value is a YAML path, that file
              is loaded as layer 2 before the remaining keys are applied.

    Returns:
        Fully resolved configuration dict.

    Raises:
        FileNotFoundError: If a YAML path is given but does not exist.
        TypeError: If *overrides* is an unsupported type.
    """
    cfg = get_default_cfg()

    if overrides is None:
        return cfg

    if isinstance(overrides, (str, Path)):
        task_cfg = load_yaml(overrides)
        return merge_configs(cfg, task_cfg)

    if not isinstance(overrides, dict):
        raise TypeError(
            f"'overrides' must be a dict, str, or Path, got {type(overrides).__name__}."
        )

    # Support {'cfg': 'path/to/task.yaml', 'key': value, ...}
    # The 'cfg' key is a pointer to a task YAML, not a config value itself.
    task_yaml_path = overrides.get("cfg")
    rest = {k: v for k, v in overrides.items() if k != "cfg"}
    if task_yaml_path:
        task_cfg = load_yaml(task_yaml_path)
        cfg = merge_configs(cfg, task_cfg)

    return merge_configs(cfg, rest)
