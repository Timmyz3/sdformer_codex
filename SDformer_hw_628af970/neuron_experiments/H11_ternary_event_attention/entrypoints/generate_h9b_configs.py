"""Generate H9b stage/block search configs from the H9a template."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


VARIANTS: dict[str, list[str]] = {
    "h9b_attn_stage0_120": ["0:0", "0:1"],
    "h9b_attn_stage1_120": ["1:0", "1:1"],
    "h9b_attn_stage2_120": ["2:0", "2:1", "2:2", "2:3", "2:4", "2:5"],
    "h9b_attn_stage3_120": ["3:0", "3:1"],
    "h9b_attn_stage23_120": ["2:0", "2:1", "2:2", "2:3", "2:4", "2:5", "3:0", "3:1"],
    "h9b_attn_h8_goodmix_120": ["1:0", "2:3", "3:0"],
}


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def write_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def make_short_config(base: dict, name: str, target_blocks: list[str]) -> dict:
    config = deepcopy(base)
    config["experiment"] = name
    config.setdefault("runtime", {})["max_train_steps"] = 120
    config.setdefault("runtime", {})["skip_state_save"] = True
    config.setdefault("loader", {})["n_epochs"] = 1
    config["loader"]["batch_size"] = 8
    config.setdefault("test", {})["sample"] = 10
    config["bsa_attention"] = dict(config.get("bsa_attention", {}))
    config["bsa_attention"]["enabled"] = True
    config["bsa_attention"].pop("stage_selection", None)
    config["bsa_attention"]["target_blocks"] = target_blocks
    config["note"] = f"H9b attention subset search: Shiftmax target_blocks={target_blocks}."
    return config


def make_full_config(short_config: dict, name: str) -> dict:
    config = deepcopy(short_config)
    config["experiment"] = name
    config.setdefault("runtime", {})["max_train_steps"] = 0
    config.setdefault("runtime", {})["skip_state_save"] = True
    config.setdefault("loader", {})["n_epochs"] = 30
    config.setdefault("optimizer", {})["milestones"] = [20, 25]
    config.setdefault("test", {})["sample"] = 10
    config["note"] = f"Promoted full run from {short_config['experiment']}."
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--full-from", type=Path)
    parser.add_argument("--full-name")
    args = parser.parse_args()

    if args.full_from:
        if not args.full_name:
            raise SystemExit("--full-name is required with --full-from")
        short_config = load_config(args.full_from)
        write_config(args.out_dir / f"{args.full_name}.yml", make_full_config(short_config, args.full_name))
        return

    base = load_config(args.base)
    for name, target_blocks in VARIANTS.items():
        write_config(args.out_dir / f"{name}.yml", make_short_config(base, name, target_blocks))


if __name__ == "__main__":
    main()
