#!/usr/bin/env python3
"""Generate frozen-ep29 dyadic-alpha sensitivity configs for DATE C12."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
GENERATED = EXP / "configs/generated"
SOURCE = GENERATED / "dsec_fullres_w15_two_contrib_c12_binary_motion_ttx_nb0ep29_ft30_20260826.yml"
MANIFEST = GENERATED / "dsec_c12_ep29_alpha_sensitivity_20260830.json"


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
    source = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    variants = []
    for variant_id, alpha in (("alpha0125", 0.125), ("alpha0500", 0.5)):
        config = deepcopy(source)
        config["experiment"] = f"dsec_c12_ep29_{variant_id}_sensitivity_20260830"
        config["bsa_attention"]["binary_motion_xor_alpha"] = alpha
        runtime = config.setdefault("runtime", {})
        runtime["frozen_checkpoint_sensitivity"] = True
        runtime["sensitivity_parent_epoch"] = 29
        runtime["sensitivity_field"] = "binary_motion_xor_alpha"
        runtime["sensitivity_value"] = alpha
        config["note"] = (
            "DATE DSEC frozen C12 ep29 dyadic-alpha sensitivity. This changes only "
            "the shift/add deployment constant and performs no training."
        )
        path = GENERATED / f"dsec_c12_ep29_{variant_id}_sensitivity_20260830.yml"
        write_new_or_identical(path, yaml.safe_dump(config, sort_keys=False, width=120))
        variants.append(
            {
                "id": variant_id,
                "alpha": alpha,
                "config": str(path.resolve()),
                "config_sha256": sha256(path),
            }
        )

    payload = {
        "schema": "dsec_c12_ep29_alpha_sensitivity_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_config": str(SOURCE.resolve()),
        "source_config_sha256": sha256(SOURCE),
        "source_alpha": 0.25,
        "variants": variants,
        "protocol": {
            "checkpoint": "C12 same-parent full30 ep29",
            "training": False,
            "evaluation": "standard local DSEC valid825",
            "values": [0.125, 0.25, 0.5],
            "promotion_gate": (
                "candidate AEE < C10 ep29 AEE, candidate AE-3D no worse than C12 alpha=0.25 "
                "by more than 0.2%, and spikes increase <=1%"
            ),
            "claim_boundary": "frozen-checkpoint sensitivity, not a trained causal ablation",
        },
    }
    write_new_or_identical(MANIFEST, json.dumps(payload, indent=2) + "\n")
    print(MANIFEST.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
