#!/usr/bin/env python3
"""Generate the strict same-parent MVSEC C00 continuation config."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
GENERATED = EXP / "configs/generated"
SOURCE = GENERATED / "mvsec_cicc_h67_motion_w8_seed0.yml"
OUTPUT = GENERATED / "mvsec_cicc_strict_c00_psn_original_w8_seed0_full30_20260830.yml"
MANIFEST = GENERATED / "mvsec_strict_same_parent_c00_20260830.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new_or_identical(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"refusing to overwrite changed artifact: {path}")
        return
    path.write_text(text, encoding="utf-8")


def main() -> int:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    config = deepcopy(yaml.safe_load(SOURCE.read_text(encoding="utf-8")))
    config["experiment"] = "mvsec_cicc_strict_c00_psn_original_w8_seed0_full30_20260830"
    config["atlif_ternary_psn"] = {"enabled": False}
    config["bsa_attention"] = {"enabled": False}
    config["experimental_neuron"] = {"enabled": False}
    runtime = config.setdefault("runtime", {})
    runtime["initialization"] = "same_mvsec_nb0_ep11_checkpoint_fresh_optimizer"
    runtime["ablation_role"] = "strict_c00_original_psn_original_attention"
    runtime["strict_same_parent"] = True
    config["note"] = (
        "DATE strict MVSEC C00: original PSN and original SDSA, initialized from the same "
        "NB0 ep11 parent as C10/C12, then trained with the same fresh optimizer, seed0, "
        "day2-only split, crop256, window2x8x8, and full30 budget."
    )
    write_new_or_identical(OUTPUT, yaml.safe_dump(config, sort_keys=False, width=120))
    payload = {
        "schema": "mvsec_strict_same_parent_c00_manifest_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_config": str(SOURCE.resolve()),
        "source_config_sha256": sha256(SOURCE),
        "config": str(OUTPUT.resolve()),
        "config_sha256": sha256(OUTPUT),
        "parent_checkpoint": str(
            (EXP / "results/mvsec_cicc_nb0_w8_seed0_v4_20260811/checkpoint_epoch11.pth").resolve()
        ),
        "protocol": {
            "data": "outdoor_day2 dt1 train-only manifest with held-out day2 validation",
            "seed": 0,
            "crop": [256, 256],
            "window": [2, 8, 8],
            "epochs": 30,
            "optimizer": "fresh AdamW; same C10/C12 groups and milestones",
            "selection": "minimum held-out day2 validation loss, then one fixed800 and full-sequence test",
            "claim_boundary": "strict same-parent MVSEC internal ablation, not MDR cross-dataset protocol",
        },
    }
    write_new_or_identical(MANIFEST, json.dumps(payload, indent=2) + "\n")
    print(MANIFEST.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
