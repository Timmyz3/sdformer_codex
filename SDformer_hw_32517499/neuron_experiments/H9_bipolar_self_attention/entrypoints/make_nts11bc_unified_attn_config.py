"""NTS-11bc: unified h60 attention on all 12 Swin blocks + strict two-neuron scope.

Deployment story:
  - Attention: h60 Shiftmax on every encoder attention block (S0–S3, 12 blocks)
  - Neurons: ternary Q/K + binary official ATLIF everywhere else (sn2_q explicit)
  - Scope: downsample_ternary (same as 11aq winner)
  - Recipe: warm720 + freeze1224 + fastlr (11aq)
  - Train: NB0 ep59 -> full30
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_nts11_phase4_scope_configs import apply_scope_policy, build_path_sets, read_yaml, write_yaml
from make_nts11_phase5_configs import RECIPES, apply_recipe

EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"
AA_FULL = EXP_ROOT / "configs/nts11aa_hw_h60_s23_scope_downsample_ternary_scope_full30_20260612_065413.yml"
NB0 = Path(__file__).resolve().parents[3] / "experiments/baseline_stride_upstream/checkpoint_epoch59.pth"

ALL12_BLOCKS = (
    "0:0",
    "0:1",
    "1:0",
    "1:1",
    "2:0",
    "2:1",
    "2:2",
    "2:3",
    "2:4",
    "2:5",
    "3:0",
    "3:1",
)


def _apply_11bc_core(cfg: dict[str, Any], paths: dict[str, list[str]]) -> None:
    apply_scope_policy(cfg, "downsample_ternary", paths)
    apply_recipe(cfg, RECIPES["w720_fastlr"])

    atlif = cfg.setdefault("atlif_ternary_psn", {})
    atlif["enabled"] = True
    atlif["target"] = "qk"
    atlif["stage_selection"] = "all"

    attn = cfg.setdefault("bsa_attention", {})
    attn["enabled"] = True
    attn["mode"] = "h60"
    attn.pop("stage_selection", None)
    attn["target_blocks"] = list(ALL12_BLOCKS)

    cfg.setdefault("metrics", {})["name"] = ["AEE", "AAE"]
    cfg.setdefault("test", {})["sample"] = 10


def make_short_config() -> tuple[Path, dict[str, Any]]:
    base = read_yaml(AA_FULL)
    paths = build_path_sets()
    cfg = deepcopy(base)
    name = "nts11bc_hw_h60_all12_ds_w720_fastlr_s1224"
    cfg["experiment"] = name
    cfg["note"] = (
        "NTS-11bc short: all 12 blocks h60 + two-neuron downsample scope; "
        "1224-step screen from NB0."
    )
    _apply_11bc_core(cfg, paths)

    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = 1
    loader["batch_size"] = 8

    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 1224
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = [0]
    runtime["use_mlflow_model_logging"] = False

    out = GENERATED / f"{name}.yml"
    write_yaml(out, cfg)
    meta = {
        "name": name,
        "config": str(out),
        "resume": str(NB0),
        "attention_blocks": len(ALL12_BLOCKS),
        "steps": 1224,
    }
    return out, meta


def make_full30_config() -> tuple[Path, dict[str, Any]]:
    base = read_yaml(AA_FULL)
    paths = build_path_sets()
    cfg = deepcopy(base)
    name = "nts11bc_hw_h60_all12_ds_w720_fastlr_full30"
    cfg["experiment"] = name
    cfg["note"] = (
        "NTS-11bc: all 12 encoder blocks use h60 Shiftmax (unified attention). "
        "Two-neuron deployment: Q/K ternary + sn2_q binary + all_non_qk binary; "
        "downsample ternary scope; warm720/fastlr/freeze1224; NB0->full30."
    )
    _apply_11bc_core(cfg, paths)

    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = 30
    loader["batch_size"] = 8
    loader["n_workers"] = 8
    loader["persistent_workers"] = True
    loader["prefetch_factor"] = 4

    optimizer = cfg.setdefault("optimizer", {})
    optimizer["use_amp"] = True
    optimizer["milestones"] = [20, 25]

    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = False
    runtime["force_save_epochs"] = [0, 4, 9, 14, 19, 24, 28, 29]
    runtime["use_mlflow_model_logging"] = False

    out = GENERATED / f"{name}.yml"
    write_yaml(out, cfg)
    meta = {
        "name": name,
        "config": str(out),
        "resume": str(NB0),
        "attention_blocks": len(ALL12_BLOCKS),
        "scope": "downsample_ternary",
        "recipe": "w720_fastlr",
    }
    return out, meta


def main() -> int:
    if not NB0.is_file():
        raise FileNotFoundError(f"missing NB0 checkpoint: {NB0}")
    short_out, short_meta = make_short_config()
    full_out, full_meta = make_full30_config()
    print(short_out)
    print(full_out)
    print(f"# resume: {short_meta['resume']}")
    print(f"# h60 blocks: {short_meta['attention_blocks']}")
    print(f"# short steps: {short_meta['steps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())