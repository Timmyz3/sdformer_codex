"""Generate H49 QKFormer-native TX selector sweep configs.

H49 differs from H45/H47 by avoiding an N x N attention matrix. It keeps the
QKFormer K carrier and replaces the Q-only token selector with a same-token
ternary Q/K consistency selector.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = EXP_ROOT / "configs"
GENERATED_DIR = CONFIG_DIR / "generated"
BASE = GENERATED_DIR / "h41_txs02c_slowbb_full30_20260523_032052.yml"


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def clone_sn2_only_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cloned: list[dict[str, Any]] = []
    for group in groups:
        item = deepcopy(group)
        item["name"] = f"{item.get('name', 'group')}_sn2only"
        item["paths"] = [path for path in item.get("paths", []) if str(path).endswith(".mlp.sn2")]
        if item["paths"]:
            cloned.append(item)
    return cloned


def soften_ffn_groups(groups: list[dict[str, Any]], eta: float, lr_scale: float) -> list[dict[str, Any]]:
    cloned = deepcopy(groups)
    for group in cloned:
        group["threshold_eta"] = eta
        group["threshold_lr_scale"] = lr_scale
    return cloned


def apply_lr_strategy(cfg: dict[str, Any], strategy: str) -> None:
    opt = cfg.setdefault("optimizer", {})
    groups = opt.setdefault("param_groups", {})
    groups["enabled"] = True
    if strategy == "slowbb":
        groups.update(
            {
                "backbone_lr": 2.0e-7,
                "norm_lr": 2.0e-7,
                "neuron_lr": 1.2e-5,
                "threshold_lr": 3.0e-6,
            }
        )
        cfg.setdefault("atlif_ternary_psn", {})["threshold_base_lr"] = 3.0e-6
    elif strategy == "warm_slowbb":
        groups.update(
            {
                "backbone_lr": 2.0e-7,
                "norm_lr": 2.0e-7,
                "neuron_lr": 1.2e-5,
                "threshold_lr": 3.0e-6,
            }
        )
        opt["lr_warmup"] = {"enabled": True, "steps": 100, "start_factor": 0.2}
        cfg.setdefault("atlif_ternary_psn", {})["threshold_base_lr"] = 3.0e-6
    elif strategy == "midbb":
        groups.update(
            {
                "backbone_lr": 5.0e-7,
                "norm_lr": 5.0e-7,
                "neuron_lr": 1.2e-5,
                "threshold_lr": 2.0e-6,
            }
        )
        cfg.setdefault("atlif_ternary_psn", {})["threshold_base_lr"] = 2.0e-6
    else:
        raise ValueError(f"unknown lr strategy: {strategy}")


def main() -> int:
    base = read_yaml(BASE)
    base_groups = deepcopy(base["atlif_ternary_psn"].get("target_groups", []))
    variants = [
        {
            "name": "h49_txsel_s02_tr05_slowbb",
            "target_rate": 0.05,
            "threshold_eta": 0.001,
            "activity_eta": 2.0,
            "score_scale": 1.0,
            "preserve_mean": True,
            "groups": base_groups,
            "lr": "slowbb",
            "lambda_ang": 0.0,
        },
        {
            "name": "h49_txsel_s02_tr07_slowbb",
            "target_rate": 0.07,
            "threshold_eta": 0.0007,
            "activity_eta": 1.5,
            "score_scale": 1.0,
            "preserve_mean": True,
            "groups": base_groups,
            "lr": "slowbb",
            "lambda_ang": 0.0,
        },
        {
            "name": "h49_txsel_s02_tr09_slowbb",
            "target_rate": 0.09,
            "threshold_eta": 0.0005,
            "activity_eta": 1.2,
            "score_scale": 1.0,
            "preserve_mean": True,
            "groups": base_groups,
            "lr": "slowbb",
            "lambda_ang": 0.0,
        },
        {
            "name": "h49_txsel_s02_tr07_warm",
            "target_rate": 0.07,
            "threshold_eta": 0.0007,
            "activity_eta": 1.5,
            "score_scale": 1.0,
            "preserve_mean": True,
            "groups": base_groups,
            "lr": "warm_slowbb",
            "lambda_ang": 0.0,
        },
        {
            "name": "h49_txsel_s02_tr07_softffn",
            "target_rate": 0.07,
            "threshold_eta": 0.0007,
            "activity_eta": 1.5,
            "score_scale": 1.0,
            "preserve_mean": True,
            "groups": soften_ffn_groups(base_groups, eta=3.0e-5, lr_scale=3000),
            "lr": "slowbb",
            "lambda_ang": 0.0,
        },
        {
            "name": "h49_txsel_sn2only_tr07",
            "target_rate": 0.07,
            "threshold_eta": 0.0007,
            "activity_eta": 1.5,
            "score_scale": 1.0,
            "preserve_mean": True,
            "groups": clone_sn2_only_groups(base_groups),
            "lr": "slowbb",
            "lambda_ang": 0.0,
        },
        {
            "name": "h49_txsel_s02_tr07_score075",
            "target_rate": 0.07,
            "threshold_eta": 0.0007,
            "activity_eta": 1.5,
            "score_scale": 0.75,
            "preserve_mean": True,
            "groups": base_groups,
            "lr": "slowbb",
            "lambda_ang": 0.0,
        },
        {
            "name": "h49_txsel_s02_tr07_score125",
            "target_rate": 0.07,
            "threshold_eta": 0.0007,
            "activity_eta": 1.5,
            "score_scale": 1.25,
            "preserve_mean": True,
            "groups": base_groups,
            "lr": "slowbb",
            "lambda_ang": 0.0,
        },
        {
            "name": "h49_txsel_s02_tr07_ang002",
            "target_rate": 0.07,
            "threshold_eta": 0.0007,
            "activity_eta": 1.5,
            "score_scale": 1.0,
            "preserve_mean": True,
            "groups": base_groups,
            "lr": "slowbb",
            "lambda_ang": 0.02,
        },
        {
            "name": "h49_txsel_s02_tr07_midbb",
            "target_rate": 0.07,
            "threshold_eta": 0.0007,
            "activity_eta": 1.5,
            "score_scale": 1.0,
            "preserve_mean": True,
            "groups": base_groups,
            "lr": "midbb",
            "lambda_ang": 0.0,
        },
        {
            "name": "h49_txsel_s02_tr07_nopreserve",
            "target_rate": 0.07,
            "threshold_eta": 0.0007,
            "activity_eta": 1.5,
            "score_scale": 1.0,
            "preserve_mean": False,
            "groups": base_groups,
            "lr": "slowbb",
            "lambda_ang": 0.0,
        },
    ]

    generated: list[str] = []
    for variant in variants:
        cfg = deepcopy(base)
        name = str(variant["name"])
        cfg["experiment"] = name
        atlif = cfg.setdefault("atlif_ternary_psn", {})
        atlif["target_rate"] = float(variant["target_rate"])
        atlif["threshold_eta"] = float(variant["threshold_eta"])
        atlif["activity_eta"] = float(variant["activity_eta"])
        atlif["target_groups"] = deepcopy(variant["groups"])
        for group in atlif.get("target_groups", []):
            group["activity_eta"] = float(variant["activity_eta"])

        attn = cfg.setdefault("bsa_attention", {})
        attn["mode"] = "ternary_alpha_xnor_qkselector_shiftmax"
        attn["score_scale"] = float(variant["score_scale"])
        attn["preserve_mean"] = bool(variant["preserve_mean"])
        attn["center_scores"] = True
        attn["value_mode"] = "threshold"

        loss = cfg.setdefault("loss", {})
        lambda_ang = float(variant["lambda_ang"])
        loss["lambda_ang"] = lambda_ang
        loss["use_angular_loss"] = lambda_ang > 0.0

        loader = cfg.setdefault("loader", {})
        loader["n_epochs"] = 30
        loader["batch_size"] = 8
        loader["n_workers"] = 8
        loader["pin_memory"] = True
        loader["persistent_workers"] = True
        loader["prefetch_factor"] = 4

        apply_lr_strategy(cfg, str(variant["lr"]))
        cfg.setdefault("optimizer", {})["use_amp"] = True
        cfg["note"] = (
            "H49 QKFormer-native TX selector sweep. This replaces the Q-only "
            "QKFormer token selector with same-token ternary Q/K consistency "
            "Shiftmax, avoiding H45/H47's N x N gate@K/V mixing. "
            f"variant={name}; target_rate={variant['target_rate']}; "
            f"lr={variant['lr']}; score_scale={variant['score_scale']}; "
            f"preserve_mean={variant['preserve_mean']}."
        )
        out = GENERATED_DIR / f"{name}.yml"
        write_yaml(out, cfg)
        generated.append(f"generated/{out.name}")

    print("\n".join(generated))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
