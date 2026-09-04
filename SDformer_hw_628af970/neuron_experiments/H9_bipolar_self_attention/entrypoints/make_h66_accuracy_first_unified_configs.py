"""Generate accuracy-first, full-network, unified no-carrier attention configs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GEN = EXP_ROOT / "configs/generated"
BASE = GEN / "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8.yml"
MANIFEST = GEN / "h66_accuracy_first_unified_manifest.json"
TTX_CHECKPOINT = (
    EXP_ROOT
    / "results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid"
    / "checkpoint_epoch2.pth"
)
ALL12_BLOCKS = [
    "0:0", "0:1", "1:0", "1:1", "2:0", "2:1",
    "2:2", "2:3", "2:4", "2:5", "3:0", "3:1",
]


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def make_candidate(
    base: dict[str, Any], *, name: str, mode: str, preserve_mean: bool,
    self_bias: float = 0.0,
) -> dict[str, Any]:
    cfg = deepcopy(base)
    cfg["experiment"] = name
    neuron = cfg["atlif_ternary_psn"]
    neuron.update({
        "output_mode": "binary", "threshold_mode": "official_atlif", "center_mode": "zero",
        "target_rate": None, "target_rate_eta": 0.0, "activity_eta": 0.0, "threshold_eta": 0.0,
        "target": "qk", "stage_selection": "all",
    })
    for group in neuron.get("target_groups", []):
        group.update({
            "output_mode": "binary", "threshold_mode": "official_atlif", "center_mode": "zero",
            "target_rate": None, "target_rate_eta": 0.0, "activity_eta": 0.0, "threshold_eta": 0.0,
        })
    cfg["bsa_attention"].update({
        "enabled": True, "mode": mode, "target_blocks": list(ALL12_BLOCKS),
        "center_scores": True, "preserve_mean": preserve_mean, "alpha0": 1.0 / 64.0,
        "mismatch_penalty": 0.0, "single_active_penalty": 0.0, "bipolar_mu": 0.0,
        "k_magnitude_alpha": 0.0, "sc_mu_schedule_enabled": False, "target_rate": None,
        "hardware_quant_enabled": False, "value_mode": "threshold", "value_branch": "reuse_k",
        "matrix_diag_bias": float(self_bias),
    })
    cfg.setdefault("runtime", {}).update({
        "max_train_steps": 120, "skip_state_save": True, "force_save_epochs": [0],
        "use_mlflow_model_logging": False,
    })
    cfg["loader"].update({"n_epochs": 1, "batch_size": 8, "n_workers": 8})
    cfg["metrics"]["name"] = ["AEE", "AAE"]
    cfg["test"].update({"sample": 10, "n_valid": 1})
    cfg["note"] = (
        "H66 accuracy-first DSEC screen: full 105 one-sided binary ATLIF wrappers and "
        f"all12 unified {mode}. No native QKFormer carrier, TX/SC mixing, Kmag, target-rate, "
        "or partial replacement. K is reused only as the value stream. Warm-start from TTX epoch2."
    )
    path = GEN / f"{name}.yml"
    write_yaml(path, cfg)
    return {
        "name": name, "config": str(path), "resume": str(TTX_CHECKPOINT),
        "neuron": "105 x one-sided binary official ATLIF", "attention": f"12 x {mode}",
        "native_qkformer_carrier": False, "value_path": "attention output uses reused K",
        "self_bias": float(self_bias),
        "status": "generated_not_run",
    }


def main() -> int:
    if not BASE.is_file():
        raise FileNotFoundError(BASE)
    if not TTX_CHECKPOINT.is_file():
        raise FileNotFoundError(TTX_CHECKPOINT)
    base = load_yaml(BASE)
    rows = [
        make_candidate(base, name="h66a_allbinary_all12_axnor_matrix_shiftmax_s120",
                       mode="binary_alpha_xnor_matrix_shiftmax", preserve_mean=False),
        make_candidate(base, name="h66b_allbinary_all12_hamming_linear_s120",
                       mode="hamming_binary_direct", preserve_mean=False),
        make_candidate(base, name="h66c_allbinary_all12_tp_ttx_s120",
                       mode="binary_axnor_temporal_pair_shiftmax", preserve_mean=False),
        make_candidate(base, name="h66d_allbinary_all12_lr_ttx_s120",
                       mode="binary_axnor_local5_shiftmax", preserve_mean=False),
        make_candidate(base, name="h66e_allbinary_all12_tp_ttx_selfbias1_s120",
                       mode="binary_axnor_temporal_pair_shiftmax", preserve_mean=False,
                       self_bias=1.0),
    ]
    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(row["config"])
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
