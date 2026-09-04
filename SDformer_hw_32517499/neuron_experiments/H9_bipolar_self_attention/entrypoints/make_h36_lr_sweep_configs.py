"""生成 H36：对 H34/H35 优先候选做续训学习率策略短测。

H36 不改变神经元/注意力范式，只改变 fine-tuning 的学习率分组和
threshold_update 的 base lr。用于回答：同一个方案到底是模块不行，还是续训
LR 不合适。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = EXP_ROOT / "configs"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def set_runtime(cfg: dict[str, Any], name: str, note: str) -> None:
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
    cfg.setdefault("test", {})["sample"] = 10
    cfg.setdefault("metrics", {})["name"] = ["AEE", "AAE"]


def set_lr_strategy(cfg: dict[str, Any], strategy: dict[str, Any]) -> None:
    opt = cfg.setdefault("optimizer", {})
    opt["use_amp"] = True
    opt["lr"] = float(strategy["backbone_lr"])
    opt["param_groups"] = {
        "enabled": True,
        "backbone_lr": float(strategy["backbone_lr"]),
        "neuron_lr": float(strategy["neuron_lr"]),
        "threshold_lr": float(strategy["threshold_lr"]),
        "norm_lr": float(strategy["norm_lr"]),
        "norm_wd": 0.0,
        "threshold_wd": 0.0,
    }
    cfg.setdefault("atlif_ternary_psn", {})["threshold_base_lr"] = float(strategy["threshold_base_lr"])


def main() -> int:
    candidates = [
        (
            "stage02_highsop",
            CONFIG_DIR / "h34_hybrid_h9_stage02_highsop_s150k_act2p0.yml",
            "H34 当前最均衡神经元范围：Q/K 三值 PSN+ATLIF，高 SOP stage0/2 二值 official PSN+ATLIF。",
        ),
        (
            "highsop_sparse",
            CONFIG_DIR / "h34_hybrid_h9_highsop_s300k_act4p0.yml",
            "H34 稀疏优先方案：全高 SOP 集合 official ATLIF，强阈值增长。",
        ),
        (
            "strict_bsa_signv",
            CONFIG_DIR / "h35_strict_bsa_signv_head_s150k_act2.yml",
            "H35 标准 BSA sign-V 方案：sign(Q)@sign(K)^T -> Shiftmax -> sign/ternary value。",
        ),
        (
            "signed_consensus_shiftmax",
            CONFIG_DIR / "h35_signed_consensus_shiftmax_s150k_act2.yml",
            "H35 signed-popcount + Shiftmax token gate 方案。",
        ),
    ]
    strategies = [
        {
            "name": "cur",
            "backbone_lr": 1.0e-6,
            "norm_lr": 1.0e-6,
            "neuron_lr": 3.0e-5,
            "threshold_lr": 1.0e-5,
            "threshold_base_lr": 1.0e-5,
            "note": "当前 diff-LR 基线。",
        },
        {
            "name": "conservative",
            "backbone_lr": 5.0e-7,
            "norm_lr": 5.0e-7,
            "neuron_lr": 1.5e-5,
            "threshold_lr": 5.0e-6,
            "threshold_base_lr": 5.0e-6,
            "note": "保守续训，减少 baseline 权重漂移和阈值过快上升。",
        },
        {
            "name": "neuronfast",
            "backbone_lr": 5.0e-7,
            "norm_lr": 5.0e-7,
            "neuron_lr": 5.0e-5,
            "threshold_lr": 1.0e-5,
            "threshold_base_lr": 1.0e-5,
            "note": "新 ATLIF/PSN 参数快速适配，backbone 慢速续训。",
        },
        {
            "name": "threshfast",
            "backbone_lr": 5.0e-7,
            "norm_lr": 5.0e-7,
            "neuron_lr": 3.0e-5,
            "threshold_lr": 1.5e-5,
            "threshold_base_lr": 2.0e-5,
            "note": "阈值更新更强，优先压 SOPs/firing。",
        },
        {
            "name": "backbone2x",
            "backbone_lr": 2.0e-6,
            "norm_lr": 1.0e-6,
            "neuron_lr": 3.0e-5,
            "threshold_lr": 1.0e-5,
            "threshold_base_lr": 1.0e-5,
            "note": "backbone 略快，测试精度是否受 backbone 适配不足限制。",
        },
    ]

    written: list[Path] = []
    for candidate_name, path, candidate_note in candidates:
        base = load_yaml(path)
        for strategy in strategies:
            cfg = deepcopy(base)
            name = f"h36_{candidate_name}_{strategy['name']}"
            set_lr_strategy(cfg, strategy)
            set_runtime(
                cfg,
                name,
                f"H36 学习率策略短测。候选：{candidate_note} 学习率：{strategy['note']}",
            )
            out = CONFIG_DIR / f"{name}.yml"
            dump_yaml(out, cfg)
            written.append(out)
            print(out)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
