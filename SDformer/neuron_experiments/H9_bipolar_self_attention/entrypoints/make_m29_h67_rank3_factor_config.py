#!/usr/bin/env python3
"""Generate the fail-closed M29 H67 rank-3 factorization fine-tune config."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = (
    EXP_ROOT
    / "configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml"
)
DEFAULT_CHECKPOINT = (
    EXP_ROOT
    / "results/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth"
)
DEFAULT_OUTPUT = (
    EXP_ROOT
    / "configs/generated/dsec_fullres_w15_H67_ep35_M29_rank3_factor_atlif_ft5_20260822.yml"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_config(base: dict, source_checkpoint: Path) -> dict:
    cfg = deepcopy(base)
    cfg["experiment"] = "dsec_fullres_w15_H67_ep35_M29_rank3_factor_atlif_ft5_20260822"
    atlif = cfg.setdefault("atlif_ternary_psn", {})
    atlif.update({
        "temporal_factor_rank": 3,
        "temporal_factor_init": "balanced_svd",
        "trainable": "temporal_factor_atlif",
    })
    cfg.setdefault("loader", {}).update({
        "n_epochs": 5,
        "batch_size": 2,
        "batch_multiplyer": 1,
        "n_workers": 8,
        "persistent_workers": True,
        "pin_memory": False,
        "prefetch_factor": 2,
        "non_blocking": True,
    })
    cfg.setdefault("optimizer", {}).update({
        "lr": 5.0e-5,
        "milestones": [],
        "use_amp": True,
    })
    param_groups = cfg["optimizer"].setdefault("param_groups", {})
    param_groups.update({
        "enabled": True,
        "neuron_lr": 5.0e-5,
        "threshold_lr": 5.0e-6,
        "threshold_wd": 0.0,
    })
    cfg.setdefault("runtime", {}).update({
        "max_train_steps": 0,
        "skip_state_save": False,
        "force_save_epochs": [0, 1, 2, 3, 4],
        "state_save_epochs": [0, 1, 2, 3, 4],
        "save_only_force_epochs": True,
        "epoch_offset": 36,
        "seed": 0,
        "m29_scope": "floating_factor_valid40_internal_screen_amp_before_int8_qat",
        "m29_source_checkpoint": str(source_checkpoint.resolve()),
        "m29_source_checkpoint_sha256": sha256(source_checkpoint),
        "m29_requested_rank": 3,
        "m29_expected_t10_factorized_modules": 45,
        "m29_expected_t2_dense_fallback_modules": 60,
    })
    cfg["note"] = (
        "M29 H67 ep35 rank-3 temporal-factor feasibility: balanced-SVD migration, "
        "only profitable T10 ATLIF matrices use factors, all T2 matrices remain dense, "
        "and only factorized ATLIF factors/bias/threshold are trainable. This is a "
        "floating-factor AMP valid40 internal accuracy feasibility screen before "
        "INT8 QAT; it is not valid825 admission and is not a hardware speedup result."
    )
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    base = args.base.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if not base.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("M29 base config/checkpoint is missing")
    if output.exists() and not args.force:
        raise ValueError("refusing to overwrite M29 config: {}".format(output))
    cfg = build_config(
        yaml.safe_load(base.read_text(encoding="utf-8")) or {}, checkpoint
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    receipt = {
        "schema": "m29_h67_rank3_factor_config_receipt_v1",
        "status": "READY_FLOATING_FACTOR_AMP_ACCURACY_SCREEN_NOT_INT8_NOT_SPEEDUP",
        "base": str(base),
        "base_sha256": sha256(base),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "output": str(output),
        "output_sha256": sha256(output),
        "requested_rank": 3,
        "expected_t10_factorized_modules": 45,
        "expected_t2_dense_fallback_modules": 60,
        "headline_admitted": False,
    }
    receipt_path = output.with_suffix(".receipt.json")
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
