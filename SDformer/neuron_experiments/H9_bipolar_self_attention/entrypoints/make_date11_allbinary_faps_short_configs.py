"""Generate DATE11 all-binary FAPS short-sweep configs.

The sweep starts from the all-binary TX ep19 checkpoint and changes only the
FAPS attention scope / sparse K-magnitude lane. LR is kept at the DATE11 FT5
standard setting in this first pass so structure and LR are not confounded.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GEN = EXP_ROOT / "configs/generated"
MANIFEST = GEN / "date11_allbinary_faps_short_manifest.json"
BASE = GEN / "date11full_all_binary_atlif_tx_stdlr_ft_ep19_ft5.yml"

TX_EP19_RESUME = Path(
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "date11full_all_binary_atlif_tx_w720_fastlr_full30_bs8_20260617_024526_setsid/"
    "checkpoint_epoch19.pth"
)

ALL12_BLOCKS = ["0:0", "0:1", "1:0", "1:1", "2:0", "2:1", "2:2", "2:3", "2:4", "2:5", "3:0", "3:1"]
S2_BLOCKS = ["2:0", "2:1", "2:2", "2:3", "2:4", "2:5"]


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def apply_short_runtime(cfg: dict[str, Any]) -> None:
    cfg.setdefault("loader", {})["n_epochs"] = 1
    cfg.setdefault("loader", {})["batch_size"] = 8
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 360
    runtime["skip_state_save"] = False
    runtime["force_save_epochs"] = [0]
    runtime["use_mlflow_model_logging"] = False


def apply_lr(cfg: dict[str, Any], *, lr_strategy: str) -> None:
    optimizer = cfg.setdefault("optimizer", {})
    groups = optimizer.setdefault("param_groups", {})
    groups["enabled"] = True
    if lr_strategy == "fastlr_ft":
        optimizer["lr"] = 3.0e-5
        groups["backbone_lr"] = 2.0e-6
        groups["norm_lr"] = 2.0e-6
        groups["neuron_lr"] = 5.0e-5
        groups["threshold_lr"] = 5.0e-6
    elif lr_strategy == "slowlr_ft":
        optimizer["lr"] = 1.0e-5
        groups["backbone_lr"] = 5.0e-7
        groups["norm_lr"] = 5.0e-7
        groups["neuron_lr"] = 1.5e-5
        groups["threshold_lr"] = 2.5e-6
    elif lr_strategy != "stdlr_ft":
        raise ValueError(f"unknown lr_strategy: {lr_strategy}")


def apply_faps(cfg: dict[str, Any], *, target_blocks: list[str], k_magnitude_alpha: float) -> None:
    attn = cfg.setdefault("bsa_attention", {})
    attn.update(
        {
            "enabled": True,
            "mode": "faps",
            "center_scores": True,
            "preserve_mean": True,
            "alpha0": 0.02,
            "mismatch_penalty": 0.25,
            "single_active_penalty": 0.05,
            "single_active_penalty_grad": "ste",
            "score_scale": 1.0,
            "consensus_bias": 0.02,
            "consensus_score_norm": "head_dim",
            "eps": 1.0e-6,
            "relu_k_floor": 0.0,
            "value_mode": "threshold",
            "target_rate": None,
            "sc_mu_schedule_enabled": False,
            "bipolar_mu": 0.0,
            "k_magnitude_alpha": float(k_magnitude_alpha),
            "directional_channels_enabled": True,
            "directional_merge_mode": "mean",
            "flow_disagreement_gamma": 0.0,
            "confidence_min_active": 8 if k_magnitude_alpha else 0,
            "kmag_quantize_bits": 2,
            "target_blocks": target_blocks,
        }
    )


def make_one(
    base: dict[str, Any],
    *,
    name: str,
    target_blocks: list[str],
    k_magnitude_alpha: float,
    lr_strategy: str = "stdlr_ft",
) -> tuple[Path, dict[str, Any]]:
    cfg = deepcopy(base)
    cfg["experiment"] = name
    scope = "all12" if len(target_blocks) == 12 else "s2only"
    kmag = "kmag032" if k_magnitude_alpha else "nokmag"
    cfg["note"] = (
        f"DATE11 all-binary FAPS short sweep: {scope}, {kmag}, "
        f"360 train steps from all-binary TX ep19, {lr_strategy} learning rates."
    )
    apply_short_runtime(cfg)
    apply_lr(cfg, lr_strategy=lr_strategy)
    apply_faps(cfg, target_blocks=target_blocks, k_magnitude_alpha=k_magnitude_alpha)
    out = GEN / f"{name}.yml"
    write_yaml(out, cfg)
    return out, {
        "name": name,
        "config": str(out),
        "resume": str(TX_EP19_RESUME),
        "target_blocks": scope,
        "k_magnitude_alpha": k_magnitude_alpha,
        "steps": 360,
        "batch_size": 8,
        "lr_strategy": lr_strategy,
        "status": "config_generated_not_run",
    }


def main() -> int:
    if not BASE.is_file():
        raise FileNotFoundError(f"missing base config: {BASE}")
    base = read_yaml(BASE)
    specs = [
        ("date11allbin_faps_all12_nokmag_stdlr_s360", ALL12_BLOCKS, 0.0),
        ("date11allbin_faps_all12_kmag032_stdlr_s360", ALL12_BLOCKS, 0.03125),
        ("date11allbin_faps_s2only_nokmag_stdlr_s360", S2_BLOCKS, 0.0),
        ("date11allbin_faps_s2only_kmag032_stdlr_s360", S2_BLOCKS, 0.03125),
        ("date11allbin_faps_s2only_nokmag_fastlr_s360", S2_BLOCKS, 0.0, "fastlr_ft"),
        ("date11allbin_faps_s2only_nokmag_slowlr_s360", S2_BLOCKS, 0.0, "slowlr_ft"),
    ]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        name, blocks, k_alpha, *rest = spec
        lr_strategy = rest[0] if rest else "stdlr_ft"
        path, row = make_one(
            base,
            name=name,
            target_blocks=blocks,
            k_magnitude_alpha=k_alpha,
            lr_strategy=lr_strategy,
        )
        rows.append(row)
        print(path)
    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
