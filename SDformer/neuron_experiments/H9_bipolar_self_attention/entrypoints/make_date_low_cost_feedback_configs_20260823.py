#!/usr/bin/env python3
"""Generate inference-only Motion/Local5 constant-sensitivity configs."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
GENERATED = EXP / "configs/generated"
MOTION_SOURCE = GENERATED / "dsec_fullres_w15_factorial_c12_binary_motion_ttx_nb0ep29_ft10_20260821.yml"
LOCAL_SOURCE = GENERATED / "dsec_fullres_w15_factorial_c20_binary_local5_nb0ep29_ft10_20260821.yml"
MANIFEST = GENERATED / "date_low_cost_feedback_sensitivity_20260823.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new_or_identical(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"refusing to overwrite changed generated config: {path}")
        return
    path.write_text(text, encoding="utf-8")


def make_variant(source: dict, *, experiment: str, field: str, value: float) -> dict:
    config = deepcopy(source)
    config["experiment"] = experiment
    config.setdefault("bsa_attention", {})[field] = float(value)
    runtime = config.setdefault("runtime", {})
    runtime["feedback_sensitivity_only"] = True
    runtime["feedback_source_field"] = field
    runtime["feedback_source_value"] = float(value)
    return config


def main() -> int:
    for source in (MOTION_SOURCE, LOCAL_SOURCE):
        if not source.is_file():
            raise FileNotFoundError(source)
    motion = yaml.safe_load(MOTION_SOURCE.read_text(encoding="utf-8"))
    local = yaml.safe_load(LOCAL_SOURCE.read_text(encoding="utf-8"))

    specs = [
        ("motion_a0000", motion, "binary_motion_xor_alpha", 0.0),
        ("motion_a0125", motion, "binary_motion_xor_alpha", 0.125),
        ("motion_a0500", motion, "binary_motion_xor_alpha", 0.5),
        ("local_bm0015625", local, "matrix_diag_bias", -1.0 / 64.0),
        ("local_bp0015625", local, "matrix_diag_bias", 1.0 / 64.0),
        ("local_bp0031250", local, "matrix_diag_bias", 1.0 / 32.0),
    ]
    variants = []
    for variant_id, source, field, value in specs:
        experiment = f"date_feedback_{variant_id}_20260823"
        config = make_variant(source, experiment=experiment, field=field, value=value)
        path = GENERATED / f"{experiment}.yml"
        text = yaml.safe_dump(config, sort_keys=False, width=120)
        write_new_or_identical(path, text)
        variants.append(
            {
                "id": variant_id,
                "family": "motion" if variant_id.startswith("motion_") else "local5",
                "field": field,
                "value": value,
                "config": str(path.resolve()),
                "config_sha256": sha256(path),
            }
        )

    manifest = {
        "schema": "date_low_cost_feedback_sensitivity_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_configs": {
            "motion": {
                "path": str(MOTION_SOURCE.resolve()),
                "sha256": sha256(MOTION_SOURCE),
                "baseline_value": 0.25,
            },
            "local5": {
                "path": str(LOCAL_SOURCE.resolve()),
                "sha256": sha256(LOCAL_SOURCE),
                "baseline_value": 0.0,
            },
        },
        "variants": variants,
        "protocol": {
            "checkpoint_reuse": True,
            "training": False,
            "evaluation": "standard local DSEC valid825",
            "motion_values": [0.0, 0.125, 0.25, 0.5],
            "local5_self_bias_values": [-1.0 / 64.0, 0.0, 1.0 / 64.0, 1.0 / 32.0],
            "motion_promotion_gate": "AEE improves >=0.3% or AAE-2D improves >=0.5%, with spikes increase <=2%",
            "local5_promotion_gate": "AEE improves >=0.3%, with spikes increase <=2%",
            "claim_boundary": "inference constant sensitivity on frozen checkpoints; not a trained causal ablation",
        },
    }
    write_new_or_identical(MANIFEST, json.dumps(manifest, indent=2) + "\n")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
