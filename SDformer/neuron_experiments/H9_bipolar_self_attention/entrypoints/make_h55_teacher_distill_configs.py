"""Generate H55 teacher-distillation configs from the H54b full config."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = EXP_ROOT / "configs"
BASE_CONFIG = CONFIG_DIR / "generated/h54b_three_mu05_l08_g20_fast_full30.yml"
TEACHER_CHECKPOINT = "experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def _base_full(base: dict[str, Any], name: str, note: str) -> dict[str, Any]:
    cfg = deepcopy(base)
    cfg["experiment"] = name
    cfg["note"] = note
    runtime = cfg.setdefault("runtime", {})
    runtime.pop("max_train_steps", None)
    runtime["force_save_epochs"] = list(range(30))
    runtime["skip_state_save"] = True
    cfg.setdefault("loader", {})["n_epochs"] = 30
    return cfg


def _set_teacher(cfg: dict[str, Any], *, lambda_epe: float, lambda_dir: float) -> None:
    cfg["teacher_distill"] = {
        "enabled": True,
        "checkpoint": TEACHER_CHECKPOINT,
        "lambda_epe": lambda_epe,
        "lambda_dir": lambda_dir,
        "min_gt_mag": 0.25,
        "full_weight_gt_mag": 2.0,
        "teacher_confidence_max_epe": 4.0,
        "use_all_predictions": False,
        "epsilon": 1.0e-6,
    }


def _slow_ffn_thresholds(cfg: dict[str, Any]) -> None:
    atlif = cfg.setdefault("atlif_ternary_psn", {})
    for group in atlif.get("target_groups", []) or []:
        if group.get("name") == "s0_ffn":
            group["threshold_eta"] = 1.5e-5
            group["threshold_lr_scale"] = 1800
            group["activity_eta"] = 0.8
        if group.get("name") == "s2_half":
            group["threshold_eta"] = 1.2e-5
            group["threshold_lr_scale"] = 1500
            group["activity_eta"] = 0.65


def main() -> int:
    base = load_yaml(BASE_CONFIG)
    variants = []

    h55a = _base_full(
        base,
        "h55a_h54b_teacher_epe_full30",
        "H55a: H54b + baseline PSN teacher EPE distillation. Keep attention and ATLIF unchanged.",
    )
    _set_teacher(h55a, lambda_epe=0.05, lambda_dir=0.0)
    variants.append(h55a)

    h55b = _base_full(
        base,
        "h55b_h54b_teacher_epe_dir_full30",
        "H55b: H54b + teacher EPE + flow-magnitude weighted teacher direction distillation.",
    )
    _set_teacher(h55b, lambda_epe=0.04, lambda_dir=0.03)
    variants.append(h55b)

    h55c = _base_full(
        base,
        "h55c_h54b_teacher_epe_slowffn_full30",
        "H55c: H54b + teacher EPE distillation + slower official ATLIF threshold growth in FFN.",
    )
    _set_teacher(h55c, lambda_epe=0.05, lambda_dir=0.0)
    _slow_ffn_thresholds(h55c)
    variants.append(h55c)

    for cfg in variants:
        path = CONFIG_DIR / "generated" / f"{cfg['experiment']}.yml"
        dump_yaml(path, cfg)
        print(path.relative_to(CONFIG_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
