"""Generate the pre-registered H79-H80 Round4 assignment full30 configs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
GEN = EXP / "configs/generated"
SOURCE = GEN / "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8.yml"
TTX_EP2 = EXP / (
    "results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_"
    "20260629_154937_setsid/checkpoint_epoch2.pth"
)
MANIFEST = GEN / "h79_h80_round4_assignment_full30_manifest.json"
SAVE_EPOCHS = [0, 4, 9, 14, 19, 24, 28, 29]
CANDIDATES = (
    {
        "id": "H79",
        "name": "h79_allbinary_all12_cf10_match_code_w720_fastlr_full30",
        "mode": "binary_cf10_match_code",
        "descriptor_dim": 10,
        "stored_codebook_rows": 9,
        "expected_new_keys": 24,
        "formula": "Omega9 row Shiftmax plus top2/activity null and fixed-zero null codeword",
    },
    {
        "id": "H80",
        "name": "h80_allbinary_all12_dn9_match_code_w720_fastlr_full30",
        "mode": "binary_dn9_match_code",
        "descriptor_dim": 9,
        "stored_codebook_rows": 9,
        "expected_new_keys": 12,
        "formula": "Omega9 row/destination dual Shiftmax with unsigned Q1.7 gate product",
    },
)


def make_config(base: dict, candidate: dict) -> dict:
    config = deepcopy(base)
    config["experiment"] = candidate["name"]
    config["loader"].update({
        "n_epochs": 30,
        "batch_size": 8,
        "n_workers": 8,
        "persistent_workers": True,
        "pin_memory": False,
        "prefetch_factor": 4,
        "non_blocking": True,
    })
    config["optimizer"].update({"lr": 2.0e-5, "milestones": [20, 25], "use_amp": True})
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
        "mode": candidate["mode"],
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
        "cf10_beta_step": 1.0 / 64.0,
        "cf10_beta_min": -1.0,
        "cf10_beta_max": 1.0,
    })
    config.setdefault("runtime", {}).update({
        "allow_tf32": True,
        "cudnn_benchmark": True,
        "snn_backend": "cupy",
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
        f"{candidate['id']} Round4 full30 independently warm-started from frozen TTX epoch2; "
        f"{candidate['formula']}; one-sided binary ATLIF105, one uniform all12 formula, "
        "static per-head codebook output, and no native K/V carrier."
    )
    return config


def validate(config: dict, candidate: dict) -> None:
    attention = config["bsa_attention"]
    assert attention["mode"] == candidate["mode"]
    assert len(attention["target_blocks"]) == 12
    assert config["atlif_ternary_psn"]["output_mode"] == "binary"
    assert config["atlif_ternary_psn"]["threshold_mode"] == "official_atlif"
    assert config["loader"]["n_epochs"] == 30
    assert config["loader"]["batch_size"] == 8
    assert config["loader"]["n_workers"] == 8
    assert config["optimizer"]["use_amp"] is True
    assert config["optimizer"]["lr_warmup"]["steps"] == 720
    assert config["optimizer"]["milestones"] == [20, 25]
    assert config["runtime"]["snn_backend"] == "cupy"
    assert config["runtime"]["max_train_steps"] == 0
    assert config["runtime"]["force_save_epochs"] == SAVE_EPOCHS
    assert config["runtime"]["save_only_force_epochs"] is True
    assert attention["cf10_beta_step"] == 1.0 / 64.0


def main() -> int:
    base = yaml.safe_load(SOURCE.read_text(encoding="utf-8")) or {}
    rows = []
    for order, candidate in enumerate(CANDIDATES, start=1):
        config = make_config(base, candidate)
        validate(config, candidate)
        path = GEN / f"{candidate['name']}.yml"
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        rows.append({
            "order": order,
            **candidate,
            "config": str(path),
            "epochs": 30,
            "start_checkpoint": str(TTX_EP2),
            "save_epochs": list(SAVE_EPOCHS),
            "expected_atlif_modules": 105,
            "expected_attention_modules": 12,
            "expected_candidate_modules": 12,
            "expected_checkpoint_overlay_keys": 210,
            "status": "generated_not_run",
        })
        print(path)
    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
