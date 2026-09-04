"""YAML config loading with lightweight inheritance."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    inherit_from = data.pop("inherit_from", None)
    if inherit_from is None:
        return data

    parent_path = (config_path.parent / inherit_from).resolve()
    parent_cfg = load_config(parent_path)
    return _deep_update(parent_cfg, data)


def dump_yaml(path: str | Path, data: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def clone_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return deepcopy(cfg)

