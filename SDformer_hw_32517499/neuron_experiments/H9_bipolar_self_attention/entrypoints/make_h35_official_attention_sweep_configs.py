"""生成 H35：官方 ATLIF 范式下重做注意力方案短测配置。

H35 的原则：
- Q/K 仍保留三值 PSN-ATLIF，因为 BSA/alpha-XNOR/TSN 类注意力需要 signed
  ternary event 作为输入；
- FFN/downsample 等高 SOP 二值稀疏层统一使用 Activity-Pruning-SNN 的
  official ATLIF 更新范式；
- 每个注意力方案只改 `bsa_attention` 配置，方便和 H34 神经元范围短测对照。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = EXP_ROOT / "configs"
BASE_CONFIG = CONFIG_DIR / "h34_hybrid_h9_highsop_s150k_act2p0.yml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def set_common_runtime(cfg: dict[str, Any], name: str, note: str) -> None:
    cfg["experiment"] = name
    cfg["note"] = note
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = [0]
    runtime["use_mlflow_model_logging"] = False
    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = 1
    loader["batch_size"] = 8
    loader["n_workers"] = 8
    loader["persistent_workers"] = True
    loader["prefetch_factor"] = 4
    loader["pin_memory"] = True
    optimizer = cfg.setdefault("optimizer", {})
    optimizer["use_amp"] = True
    optimizer["lr"] = 1.0e-6
    optimizer["param_groups"] = {
        "enabled": True,
        "backbone_lr": 1.0e-6,
        "neuron_lr": 3.0e-5,
        "threshold_lr": 1.0e-5,
        "norm_lr": 1.0e-6,
        "norm_wd": 0.0,
        "threshold_wd": 0.0,
    }
    cfg.setdefault("test", {})["sample"] = 10
    cfg.setdefault("metrics", {})["name"] = ["AEE", "AAE"]


def set_official_highsop_strength(cfg: dict[str, Any], *, scale: float, activity_eta: float) -> None:
    """只调高 SOP official ATLIF 组，不改变 Q/K 三值范式。"""

    atlif = cfg["atlif_ternary_psn"]
    atlif["threshold_base_lr"] = 1.0e-5
    for group in atlif.get("target_groups", []) or []:
        if group.get("threshold_mode") == "official_atlif":
            group["threshold_lr_scale"] = scale
            group["activity_eta"] = activity_eta
            group["threshold_init"] = 0.1
            group["threshold_eta"] = 0.001
            group["target_rate"] = None
            group["target_rate_eta"] = 0.0
            group["min_threshold"] = None
            group["max_threshold"] = None


def set_attention(cfg: dict[str, Any], **kwargs: Any) -> None:
    base = {
        "enabled": True,
        "stage_selection": "all",
        "score_scale": 1.0,
        "center_scores": True,
        "preserve_mean": False,
        "eps": 1.0e-6,
        "consensus_score_norm": "head_dim",
        "consensus_bias": 0.02,
        "value_mode": "threshold",
        "alpha0": 0.02,
        "mismatch_penalty": 0.25,
        "relu_k_floor": 0.0,
    }
    base.update(kwargs)
    cfg["bsa_attention"] = base


def main() -> int:
    base = load_yaml(BASE_CONFIG)
    specs: list[tuple[str, dict[str, Any], str]] = []

    attention_variants: list[tuple[str, dict[str, Any], str]] = [
        (
            "compat_qk_shiftmax",
            {"mode": "compat_qk_product", "score_scale": 1.0, "center_scores": True},
            "历史 H9a 兼容门控：保留 QKFormer carrier，用 Q/K 相似性 Shiftmax 只做辅助 gate。",
        ),
        (
            "alpha_xnor_shiftmax_a002_p025",
            {
                "mode": "alpha_xnor_matrix_shiftmax",
                "alpha0": 0.02,
                "mismatch_penalty": 0.25,
                "score_scale": 1.0,
                "center_scores": True,
                "value_mode": "threshold",
            },
            "直接 alpha-XNOR token-token 矩阵 + Shiftmax，当前 H28/H34 基线式注意力。",
        ),
        (
            "alpha_xnor_shiftmax_a005_p05",
            {
                "mode": "alpha_xnor_matrix_shiftmax",
                "alpha0": 0.05,
                "mismatch_penalty": 0.5,
                "score_scale": 1.0,
                "center_scores": True,
                "value_mode": "threshold",
            },
            "alpha-XNOR 更强调静默奖励和反极性惩罚，检查三值负发放兼容性。",
        ),
        (
            "alpha_xnor_l1_a002_p025",
            {
                "mode": "alpha_xnor_matrix_l1",
                "alpha0": 0.02,
                "mismatch_penalty": 0.25,
                "consensus_bias": 0.02,
                "center_scores": False,
                "value_mode": "threshold",
            },
            "alpha-XNOR 矩阵 + L1 归一，无 Shiftmax，用来区分归一化和打分本身的影响。",
        ),
        (
            "strict_bsa_thresholdv_head",
            {
                "mode": "strict_bsa_shiftmax",
                "value_mode": "threshold",
                "consensus_score_norm": "head_dim",
                "score_scale": 1.0,
                "center_scores": True,
            },
            "标准 BSA 范式：sign(Q) @ sign(K)^T -> Shiftmax -> threshold-K value。",
        ),
        (
            "strict_bsa_signv_head",
            {
                "mode": "strict_bsa_shiftmax",
                "value_mode": "sign",
                "consensus_score_norm": "head_dim",
                "score_scale": 1.0,
                "center_scores": True,
            },
            "标准 BSA 范式的纯三值 value，对硬件更友好但表达更弱。",
        ),
        (
            "strict_bsa_thresholdv_sqrt",
            {
                "mode": "strict_bsa_shiftmax",
                "value_mode": "threshold",
                "consensus_score_norm": "sqrt_head_dim",
                "score_scale": 1.0,
                "center_scores": True,
            },
            "标准 BSA，sqrt 归一让注意力更尖锐，测试 AAE 是否受益。",
        ),
        (
            "signed_consensus_shiftmax",
            {
                "mode": "signed_consensus_shiftmax",
                "consensus_score_norm": "head_dim",
                "score_scale": 1.0,
                "center_scores": True,
                "preserve_mean": True,
            },
            "signed popcount token gate + Shiftmax，仍保留 QKFormer carrier。",
        ),
        (
            "signed_consensus_shiftnorm",
            {
                "mode": "signed_consensus_shiftnorm",
                "consensus_score_norm": "head_dim",
                "consensus_bias": 1.0,
                "center_scores": False,
                "preserve_mean": True,
            },
            "signed popcount + power-of-two shiftnorm，无 2^score 指数。",
        ),
        (
            "signed_consensus_l1",
            {
                "mode": "signed_consensus_popcount_l1",
                "consensus_score_norm": "head_dim",
                "consensus_bias": 1.0,
                "center_scores": False,
                "preserve_mean": True,
            },
            "signed popcount + L1，无 Shiftmax，硬件友好对照。",
        ),
        (
            "a2os2a_direct_l1",
            {
                "mode": "a2os2a_direct",
                "consensus_score_norm": "head_dim",
                "consensus_bias": 1.0e-6,
                "center_scores": False,
                "preserve_mean": True,
                "value_mode": "threshold",
            },
            "A2OS2A 启发的直接矩阵替换：binary Q、非负 K、L1 归一。",
        ),
        (
            "hamming_ternary_active",
            {
                "mode": "hamming_ternary_active_direct",
                "value_mode": "threshold",
                "center_scores": False,
                "preserve_mean": False,
            },
            "SpikeVideoFormer Hamming 线性注意力的三值 active 版本，静默不当作强负号。",
        ),
    ]

    sparsity_variants = [
        ("s150k_act2", 150000.0, 2.0),
        ("s300k_act4", 300000.0, 4.0),
    ]

    for attn_name, attn_cfg, attn_note in attention_variants:
        for sparse_name, scale, activity_eta in sparsity_variants:
            cfg = deepcopy(base)
            name = f"h35_{attn_name}_{sparse_name}"
            set_official_highsop_strength(cfg, scale=scale, activity_eta=activity_eta)
            set_attention(cfg, **attn_cfg)
            set_common_runtime(
                cfg,
                name,
                f"H35 注意力重测：{attn_note} 高 SOP 二值层使用 official ATLIF，scale={scale}, activity_eta={activity_eta}。",
            )
            specs.append((name, cfg, attn_note))

    for name, cfg, _ in specs:
        path = CONFIG_DIR / f"{name}.yml"
        dump_yaml(path, cfg)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
