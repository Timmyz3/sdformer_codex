#!/usr/bin/env python3
"""Generate validation-gated MVSEC Complete-TTX dyadic-alpha candidates."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
GENERATED = EXP / "configs/generated"
SOURCE = GENERATED / "mvsec_cicc_h67_motion_w8_seed0.yml"
PARENT = EXP / "results/mvsec_cicc_nb0_w8_seed0_v4_20260811/checkpoint_epoch11.pth"
STAMP = "20260826"
CANDIDATES = (("alpha050", 0.5), ("alpha0125", 0.125))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    source = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    rows = []
    for name, alpha in CANDIDATES:
        cfg = copy.deepcopy(source)
        experiment = f"mvsec_cicc_complete_ttx_{name}_w8_seed0_full30_{STAMP}"
        cfg["experiment"] = experiment
        cfg["bsa_attention"]["binary_motion_xor_alpha"] = alpha
        cfg["runtime"]["alpha_screen_protocol"] = (
            "DSEC-prior dyadic alpha; train and checkpoint selection use only "
            "MVSEC outdoor_day2 train/held-out-validation"
        )
        cfg["runtime"]["alpha_screen_parent_checkpoint"] = str(PARENT.resolve())
        cfg["runtime"]["alpha_screen_parent_checkpoint_sha256"] = sha256(PARENT)
        cfg["note"] = (
            "DATE MVSEC Complete-TTX dyadic-alpha screen. Same NB0 ep11 parent, seed0, "
            "full30 recipe, and train-only manifest as the alpha=0.25 H67 route. "
            "Selection is by held-out outdoor_day2 validation loss before any OD1/IF test."
        )
        path = GENERATED / f"{experiment}.yml"
        path.write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        rows.append(
            {
                "name": name,
                "alpha": alpha,
                "config": str(path.resolve()),
                "config_sha256": sha256(path),
            }
        )
    manifest = {
        "schema": "mvsec_ttx_alpha_screen_manifest_v1",
        "source_config": str(SOURCE.resolve()),
        "source_config_sha256": sha256(SOURCE),
        "parent_checkpoint": str(PARENT.resolve()),
        "parent_checkpoint_sha256": sha256(PARENT),
        "baseline_h67_alpha": 0.25,
        "baseline_h67_best_validation_loss": 8.0027529,
        "promotion_relative_margin": 0.0025,
        "selection_rule": (
            "Run alpha=0.5 first. If it does not improve held-out validation loss by "
            "at least 0.25% over alpha=0.25, run alpha=0.125. Evaluate OD1/IF1/IF2/IF3 "
            "only for a new candidate that passes the validation margin."
        ),
        "candidates": rows,
        "claim_boundary": (
            "MVSEC day2 train/held-out-validation tuning; test sequences are never used "
            "for checkpoint or alpha selection"
        ),
    }
    output = GENERATED / f"mvsec_ttx_alpha_screen_{STAMP}.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
