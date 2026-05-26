"""Generate H50/H51/H52 configs for the post-H49 short screening pass."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = EXP_ROOT / "configs" / "generated"
BASE = GENERATED_DIR / "h49_txsel_s02_tr07_softffn.yml"


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def set_common_runtime(cfg: dict[str, Any]) -> None:
    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = 30
    loader["batch_size"] = 8
    loader["n_workers"] = 8
    loader["pin_memory"] = True
    loader["persistent_workers"] = True
    loader["prefetch_factor"] = 4
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = list(range(30))
    runtime["use_mlflow_model_logging"] = False
    cfg.setdefault("optimizer", {})["use_amp"] = True


def set_lr_slowbb(cfg: dict[str, Any], *, threshold_lr: float = 3.0e-6) -> None:
    optimizer = cfg.setdefault("optimizer", {})
    groups = optimizer.setdefault("param_groups", {})
    groups["enabled"] = True
    groups["backbone_lr"] = 2.0e-7
    groups["norm_lr"] = 2.0e-7
    groups["neuron_lr"] = 1.2e-5
    groups["threshold_lr"] = threshold_lr
    cfg.setdefault("atlif_ternary_psn", {})["threshold_base_lr"] = threshold_lr


def soften_group(group: dict[str, Any], *, threshold_eta: float, lr_scale: float, activity_eta: float) -> dict[str, Any]:
    out = deepcopy(group)
    out["threshold_eta"] = threshold_eta
    out["threshold_lr_scale"] = lr_scale
    out["activity_eta"] = activity_eta
    return out


def set_layered_atlif(
    cfg: dict[str, Any],
    *,
    target_rate: float,
    stage_target_rate: dict[int, float],
    stage_threshold_eta: dict[int, float],
    stage_threshold_lr_scale: dict[int, float],
    stage_activity_eta: dict[int, float],
    target_rate_eta: float,
    groups: list[dict[str, Any]],
) -> None:
    atlif = cfg.setdefault("atlif_ternary_psn", {})
    atlif["target_rate"] = target_rate
    atlif["target_rate_eta"] = target_rate_eta
    atlif["target_rate_mode"] = "upper_bound"
    atlif["stage_target_rate"] = {str(k): float(v) for k, v in stage_target_rate.items()}
    atlif["stage_threshold_eta"] = {str(k): float(v) for k, v in stage_threshold_eta.items()}
    atlif["stage_threshold_lr_scale"] = {str(k): float(v) for k, v in stage_threshold_lr_scale.items()}
    atlif["stage_activity_eta"] = {str(k): float(v) for k, v in stage_activity_eta.items()}
    atlif["target_groups"] = groups


def make_config(base: dict[str, Any], spec: dict[str, Any]) -> Path:
    cfg = deepcopy(base)
    name = spec["name"]
    cfg["experiment"] = name
    set_common_runtime(cfg)
    set_lr_slowbb(cfg, threshold_lr=spec.get("threshold_lr", 3.0e-6))
    set_layered_atlif(
        cfg,
        target_rate=spec["target_rate"],
        stage_target_rate=spec["stage_target_rate"],
        stage_threshold_eta=spec["stage_threshold_eta"],
        stage_threshold_lr_scale=spec["stage_threshold_lr_scale"],
        stage_activity_eta=spec["stage_activity_eta"],
        target_rate_eta=spec["target_rate_eta"],
        groups=spec["groups"],
    )

    attn = cfg.setdefault("bsa_attention", {})
    attn["enabled"] = True
    attn["mode"] = spec["mode"]
    attn["score_scale"] = spec.get("score_scale", 1.0)
    attn["preserve_mean"] = spec.get("preserve_mean", True)
    attn["center_scores"] = True
    attn["alpha0"] = spec.get("alpha0", 0.02)
    attn["mismatch_penalty"] = spec.get("mismatch_penalty", 0.25)
    attn["single_active_penalty"] = spec.get("single_active_penalty", 0.0)
    attn["single_active_penalty_grad"] = spec.get("single_active_penalty_grad", "ste")
    attn["consensus_score_norm"] = spec.get("consensus_score_norm", "head_dim")
    attn["value_mode"] = spec.get("value_mode", "threshold")
    attn["relu_k_floor"] = spec.get("relu_k_floor", 0.0)

    cfg["note"] = spec["note"]
    out = GENERATED_DIR / f"{name}.yml"
    write_yaml(out, cfg)
    return out


def main() -> int:
    base = read_yaml(BASE)
    h49_groups = deepcopy(base["atlif_ternary_psn"].get("target_groups", []))
    s0_group = next(group for group in h49_groups if group["name"] == "s0_ffn")
    s2_group = next(group for group in h49_groups if group["name"] == "s2_half")

    precision_groups = [
        soften_group(s0_group, threshold_eta=2.5e-5, lr_scale=2400, activity_eta=1.0),
        soften_group(s2_group, threshold_eta=2.0e-5, lr_scale=2200, activity_eta=0.8),
    ]
    balanced_groups = [
        soften_group(s0_group, threshold_eta=3.5e-5, lr_scale=3000, activity_eta=1.3),
        soften_group(s2_group, threshold_eta=2.5e-5, lr_scale=2600, activity_eta=1.0),
    ]
    sparse_groups = [
        soften_group(s0_group, threshold_eta=4.5e-5, lr_scale=3400, activity_eta=1.5),
        soften_group(s2_group, threshold_eta=3.0e-5, lr_scale=3000, activity_eta=1.2),
    ]

    h50a = {
        "name": "h50a_h49_layered_precision",
        "mode": "ternary_alpha_xnor_qkselector_shiftmax",
        "target_rate": 0.085,
        "target_rate_eta": 0.045,
        "stage_target_rate": {0: 0.075, 1: 0.090, 2: 0.085, 3: 0.105},
        "stage_threshold_eta": {0: 5.0e-4, 1: 4.0e-4, 2: 3.0e-4, 3: 2.0e-4},
        "stage_threshold_lr_scale": {0: 42000, 1: 34000, 2: 26000, 3: 18000},
        "stage_activity_eta": {0: 1.2, 1: 1.0, 2: 0.8, 3: 0.5},
        "groups": precision_groups,
        "note": "H50a：H49 selector + 分层 target_rate/阈值增长，保精度优先，SOPs 目标约 3.1-3.25G。",
    }
    h50b = {
        "name": "h50b_h49_layered_balanced",
        "mode": "ternary_alpha_xnor_qkselector_shiftmax",
        "target_rate": 0.075,
        "target_rate_eta": 0.06,
        "stage_target_rate": {0: 0.060, 1: 0.080, 2: 0.072, 3: 0.095},
        "stage_threshold_eta": {0: 6.5e-4, 1: 4.8e-4, 2: 3.8e-4, 3: 2.5e-4},
        "stage_threshold_lr_scale": {0: 50000, 1: 38000, 2: 30000, 3: 22000},
        "stage_activity_eta": {0: 1.5, 1: 1.15, 2: 1.0, 3: 0.65},
        "groups": balanced_groups,
        "note": "H50b：H49 selector + 分层稀疏均衡版，目标 SOPs 约 3.0-3.15G。",
    }
    h50c = {
        "name": "h50c_h49_layered_sparse",
        "mode": "ternary_alpha_xnor_qkselector_shiftmax",
        "target_rate": 0.065,
        "target_rate_eta": 0.075,
        "stage_target_rate": {0: 0.052, 1: 0.072, 2: 0.062, 3: 0.085},
        "stage_threshold_eta": {0: 8.0e-4, 1: 5.5e-4, 2: 4.5e-4, 3: 3.0e-4},
        "stage_threshold_lr_scale": {0: 56000, 1: 42000, 2: 34000, 3: 25000},
        "stage_activity_eta": {0: 1.7, 1: 1.25, 2: 1.15, 3: 0.8},
        "groups": sparse_groups,
        "note": "H50c：H49 selector + 更强分层稀疏，验证是否可回到 3G 以下但控制 AAE。",
    }

    h51 = {
        **h50b,
        "name": "h51a_dual_channel_balanced",
        "mode": "dual_channel_qkselector_shiftmax",
        "alpha0": 0.02,
        "mismatch_penalty": 0.35,
        "single_active_penalty": 0.10,
        "note": "H51a：双通道兴奋/抑制 selector，沿用 H50b 分层稀疏，测试负脉冲抑噪能否压 AAE。",
    }
    h51b = {
        **h50a,
        "name": "h51b_dual_channel_precision",
        "mode": "dual_channel_qkselector_shiftmax",
        "alpha0": 0.02,
        "mismatch_penalty": 0.25,
        "single_active_penalty": 0.05,
        "note": "H51b：双通道兴奋/抑制 selector 的保精度版本。",
    }
    h52 = {
        **h50a,
        "name": "h52a_kasv_a2os2a_shiftmax",
        "mode": "a2os2a_kasv_shiftmax",
        "score_scale": 0.75,
        "value_mode": "threshold",
        "relu_k_floor": 0.0,
        "note": "H52a：K-as-Proxy V 的 A2OS2A 改编，Q 二值选择、K 非负打分、V 复用 K，不引入独立 V。",
    }

    for spec in (h50a, h50b, h50c, h51, h51b, h52):
        print(make_config(base, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
