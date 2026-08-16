"""Generate H71 window-context TTX health-check and full30 configs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
GEN = EXP / "configs/generated"
BASE = GEN / "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8.yml"
SMOKE = GEN / "h71_allbinary_all12_window_context_ttx_s360.yml"
FULL = GEN / "h71_allbinary_all12_window_context_ttx_w720_fastlr_full30.yml"
MANIFEST = GEN / "h71_window_context_ttx_manifest.json"
SAVE_EPOCHS = [0, 4, 9, 14, 19, 24, 28, 29]


def dump(path: Path, config: dict) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> int:
    base = yaml.safe_load(BASE.read_text(encoding="utf-8")) or {}
    common = deepcopy(base)
    common["bsa_attention"].update({
        "score_scale": 1.0,
        "binary_motion_xor_alpha": 0.0,
        "castling_matrix_aux_weight": 0.0,
        "castling_matrix_aux_end_step": 0,
        "event_temperature_enabled": False,
        "context_broadcast_enabled": True,
    })
    common["optimizer"].update({"lr": 2.0e-5, "use_amp": True})
    common["optimizer"]["param_groups"].update({
        "backbone_lr": 2.0e-6,
        "neuron_lr": 5.0e-5,
        "norm_lr": 1.0e-6,
        "threshold_lr": 5.0e-6,
    })
    common["optimizer"]["lr_warmup"].update({"enabled": True, "steps": 720, "start_factor": 0.05})

    smoke = deepcopy(common)
    smoke["experiment"] = SMOKE.stem
    smoke["loader"].update({"n_epochs": 1, "batch_size": 8, "n_workers": 8})
    smoke.setdefault("runtime", {}).update({
        "max_train_steps": 360,
        "skip_state_save": True,
        "force_save_epochs": [0],
        "use_mlflow_model_logging": False,
    })
    smoke["note"] = "H71 implementation-health check only; not valid evidence to reject CB-TTX."
    dump(SMOKE, smoke)

    full = deepcopy(common)
    full["experiment"] = FULL.stem
    full["loader"].update({
        "n_epochs": 30,
        "batch_size": 8,
        "n_workers": 8,
        "persistent_workers": True,
        "pin_memory": False,
        "prefetch_factor": 4,
        "non_blocking": True,
    })
    full["optimizer"]["milestones"] = [20, 25]
    full.setdefault("runtime", {}).update({
        "max_train_steps": 0,
        "skip_state_save": False,
        "save_only_force_epochs": True,
        "state_save_epochs": [19, 24, 29],
        "force_save_epochs": list(SAVE_EPOCHS),
        "use_mlflow_model_logging": False,
    })
    full["note"] = (
        "H71 full30 from TTX epoch2. Each all12 H60 block applies parameter-free "
        "window context broadcasting: (gate*K + mean_token(gate*K))/2."
    )
    dump(FULL, full)

    MANIFEST.write_text(json.dumps({
        "name": "H71 Window-Context TTX",
        "smoke_config": str(SMOKE),
        "full_config": str(FULL),
        "smoke_is_rejection_evidence": False,
        "full_epochs": 30,
        "status": "generated_not_run",
    }, indent=2) + "\n", encoding="utf-8")
    for path in (SMOKE, FULL, MANIFEST):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
