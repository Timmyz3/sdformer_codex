"""Generate H64 centered-symmetric ternary reference and direct-STC configs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GEN = EXP_ROOT / "configs/generated"
BASE = GEN / "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8.yml"
CHECKPOINT = EXP_ROOT / "results/h63_checkpoints/ttxep2_centered_symmetric_budget_calibrated.pth"
MANIFEST = GEN / "h64_centered_symmetric_manifest.json"


def candidate(base: dict, name: str, mode: str) -> tuple[Path, dict]:
    cfg = deepcopy(base)
    cfg["experiment"] = name
    neuron = cfg["atlif_ternary_psn"]
    neuron.update(
        {
            "output_mode": "ternary",
            "threshold_mode": "symmetric_bsa_tsn",
            "center_mode": "calibrated",
            "negative_threshold_scale": 1.0,
            "target_rate": None,
            "target_rate_eta": 0.0,
            "activity_eta": 0.0,
            "threshold_eta": 0.0,
        }
    )
    for group in neuron.get("target_groups", []):
        group.update(
            {
                "output_mode": "ternary",
                "threshold_mode": "symmetric_bsa_tsn",
                "center_mode": "calibrated",
                "negative_threshold_scale": 1.0,
                "target_rate": None,
                "target_rate_eta": 0.0,
                "activity_eta": 0.0,
                "threshold_eta": 0.0,
            }
        )

    attention = cfg["bsa_attention"]
    attention.update(
        {
            "mode": mode,
            "alpha0": 1.0 / 64.0,
            "bipolar_mu": 0.0,
            "k_magnitude_alpha": 0.0,
            "sc_mu_schedule_enabled": False,
            "target_rate": None,
            "direct_shiftmax_signed_events": True,
            "direct_shiftmax_center_output": False,
            "direct_shiftmax_groups": 1,
        }
    )
    cfg["loader"]["n_epochs"] = 1
    cfg["runtime"]["max_train_steps"] = 20
    cfg["runtime"]["skip_state_save"] = False
    cfg["runtime"]["force_save_epochs"] = [0]
    cfg["runtime"]["use_mlflow_model_logging"] = False
    cfg["note"] = (
        "H64 all105 offline-centered symmetric ternary ATLIF. "
        + (
            "H60/TX gate-K diagnostic reference only."
            if mode == "h60"
            else "All12 raw token-channel direct Shiftmax; no gate*K/value carrier."
        )
    )
    path = GEN / f"{name}.yml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path, {
        "name": name,
        "config": str(path),
        "checkpoint": str(CHECKPOINT),
        "atlif_expected": 105,
        "attention_expected": 12,
        "center_mode": "calibrated",
        "attention_mode": mode,
        "uses_gate_k": mode == "h60",
        "status": "stopped_activity_gate_zero_shot",
    }


def main() -> int:
    if not BASE.is_file():
        raise FileNotFoundError(BASE)
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    rows = []
    for name, mode in (
        ("h64_centered_symtern_h60_ref", "h60"),
        ("h64_centered_symtern_stc_raw", "tx_direct_token_channel_shiftmax"),
    ):
        path, row = candidate(base, name, mode)
        rows.append(row)
        print(path)
    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
