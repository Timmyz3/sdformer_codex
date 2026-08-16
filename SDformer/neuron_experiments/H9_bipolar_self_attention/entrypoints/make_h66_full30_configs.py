"""Promote every structurally eligible H66 candidate to standard full30."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
GEN = EXP / "configs/generated"
MANIFEST = GEN / "h66_all_candidates_full30_manifest.json"
SAVE_EPOCHS = [0, 4, 9, 14, 19, 24, 28, 29]
SHORT_NAMES = [
    "h66a_allbinary_all12_axnor_matrix_shiftmax_s120",
    "h66b_allbinary_all12_hamming_linear_s120",
    "h66c_allbinary_all12_tp_ttx_s120",
    "h66d_allbinary_all12_lr_ttx_s120",
    "h66e_allbinary_all12_tp_ttx_selfbias1_s120",
]


def main() -> int:
    rows = []
    for order, short_name in enumerate(SHORT_NAMES, start=1):
        source = GEN / f"{short_name}.yml"
        config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        name = short_name.removesuffix("_s120") + "_w720_fastlr_full30"
        config = deepcopy(config)
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
            "norm_lr": 1.0e-6,
            "threshold_lr": 5.0e-6,
        })
        config["optimizer"]["lr_warmup"].update({
            "enabled": True,
            "steps": 720,
            "start_factor": 0.05,
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
            f"Standard full30 promotion of {short_name}. The 120/360-step result is "
            "implementation evidence only and is not used to reject this unified all12 candidate."
        )
        path = GEN / f"{name}.yml"
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        rows.append({
            "order": order,
            "name": name,
            "config": str(path),
            "source_short_config": str(source),
            "mode": config["bsa_attention"]["mode"],
            "epochs": 30,
            "status": "generated_not_run",
        })
        print(path)
    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
