"""Generate standard full30 configs for DE9, MC49, and AX17 Match-Code."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
GEN = EXP / "configs/generated"
SOURCE = GEN / "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8.yml"
MANIFEST = GEN / "h73_h74_match_code_full30_manifest.json"
SAVE_EPOCHS = [0, 4, 9, 14, 19, 24, 28, 29]
CANDIDATES = (
    ("h73_allbinary_all12_de9_match_code_w720_fastlr_full30", "binary_de9_match_code", 18),
    ("h74_allbinary_all12_mc49_match_code_w720_fastlr_full30", "binary_mc49_match_code", 49),
    ("h75_allbinary_all12_ax17_match_code_w720_fastlr_full30", "binary_ax17_match_code", 17),
)


def make_config(base: dict, name: str, mode: str, descriptor_dim: int) -> dict:
    config = deepcopy(base)
    config["experiment"] = name
    config["loader"].update({
        "n_epochs": 30,
        "batch_size": 8,
        "n_workers": 8,
        "persistent_workers": True,
        "pin_memory": False,
        "prefetch_factor": 4,
        "non_blocking": True,
    })
    config["optimizer"].update({
        "lr": 2.0e-5,
        "milestones": [20, 25],
        "use_amp": True,
    })
    config["optimizer"]["param_groups"].update({
        "backbone_lr": 2.0e-6,
        "neuron_lr": 5.0e-5,
        "new_module_lr": 5.0e-5,
        "norm_lr": 1.0e-6,
        "threshold_lr": 5.0e-6,
    })
    config["optimizer"]["lr_warmup"].update({
        "enabled": True,
        "steps": 720,
        "start_factor": 0.05,
    })
    config["bsa_attention"].update({
        "mode": mode,
        "center_scores": False,
        "preserve_mean": False,
        "alpha0": 1.0 / 64.0,
        "binary_motion_xor_alpha": 0.0,
        "castling_matrix_aux_weight": 0.0,
        "castling_matrix_aux_end_step": 0,
        "event_temperature_enabled": False,
        "context_broadcast_enabled": False,
        "hardware_quant_enabled": False,
        "match_code_seed": 6701,
        "match_code_weight_quant_enabled": False,
        "match_code_weight_step": 1.0 / 128.0,
        "match_code_weight_min": -1.0,
        "match_code_weight_max": 127.0 / 128.0,
    })
    config.setdefault("runtime", {}).update({
        "max_train_steps": 0,
        "skip_state_save": False,
        "save_only_force_epochs": True,
        "state_save_epochs": [19, 24, 29],
        "force_save_epochs": list(SAVE_EPOCHS),
        "use_mlflow_model_logging": False,
    })
    config["metrics"]["name"] = ["AEE", "AAE"]
    config["test"].update({"sample": 10, "n_valid": 1})
    config["note"] = (
        f"Standard full30 Match-Code candidate from frozen TTX epoch2: mode={mode}, "
        f"descriptor_dim={descriptor_dim}, one-sided binary ATLIF105, one uniform all12 "
        "attention formula, no native K carrier, and a learned static per-head codebook."
    )
    return config


def validate(config: dict, mode: str) -> None:
    attention = config["bsa_attention"]
    assert attention["mode"] == mode
    assert len(attention["target_blocks"]) == 12
    assert config["atlif_ternary_psn"]["output_mode"] == "binary"
    assert config["atlif_ternary_psn"]["threshold_mode"] == "official_atlif"
    assert config["loader"]["n_epochs"] == 30
    assert config["runtime"]["max_train_steps"] == 0
    assert config["runtime"]["save_only_force_epochs"] is True


def main() -> int:
    base = yaml.safe_load(SOURCE.read_text(encoding="utf-8")) or {}
    rows = []
    for order, (name, mode, descriptor_dim) in enumerate(CANDIDATES, start=1):
        config = make_config(base, name, mode, descriptor_dim)
        validate(config, mode)
        path = GEN / f"{name}.yml"
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        rows.append({
            "order": order,
            "name": name,
            "config": str(path),
            "mode": mode,
            "descriptor_dim": descriptor_dim,
            "epochs": 30,
            "start_checkpoint": "TTX epoch2",
            "status": "generated_not_run",
        })
        print(path)
    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
