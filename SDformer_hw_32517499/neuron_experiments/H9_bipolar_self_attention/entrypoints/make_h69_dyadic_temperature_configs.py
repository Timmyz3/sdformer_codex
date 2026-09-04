"""Generate H69 dyadic-temperature TTX screening configs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GEN = EXP_ROOT / "configs/generated"
BASE = GEN / "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8.yml"
MANIFEST = GEN / "h69_dyadic_temperature_ttx_manifest.json"
SCALES = (4, 8, 16)


def dump(path: Path, config: dict) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> int:
    base = yaml.safe_load(BASE.read_text(encoding="utf-8")) or {}
    rows = []
    for scale in SCALES:
        name = f"h69_allbinary_all12_dyadic_temperature_ttx_x{scale}"
        path = GEN / f"{name}.yml"
        config = deepcopy(base)
        config["experiment"] = name
        config["bsa_attention"].update({
            "score_scale": float(scale),
            "binary_motion_xor_alpha": 0.0,
            "castling_matrix_aux_weight": 0.0,
            "castling_matrix_aux_end_step": 0,
        })
        config["optimizer"].update({"lr": 2.0e-5, "use_amp": True})
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
        config["note"] = (
            f"H69 all12 H60 TTX with fixed score_scale={scale}. The scale is a "
            "power of two, so deployment replaces a generic multiplier with a left shift."
        )
        dump(path, config)
        rows.append({
            "name": name,
            "config": str(path),
            "score_scale": scale,
            "screen_steps": 360,
            "status": "generated_not_run",
        })
        print(path)
    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
