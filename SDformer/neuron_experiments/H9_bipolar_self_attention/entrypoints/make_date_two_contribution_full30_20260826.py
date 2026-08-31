#!/usr/bin/env python3
"""Generate clean same-parent full30 DATE two-contribution controls."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
GENERATED = EXP / "configs/generated"
SHORT_MANIFEST = GENERATED / "date_fullres_factorial_controls_20260821.json"
STAMP = "20260826"
KEEP_CELLS = {
    "c00_psn_original",
    "c10_binary_original",
    "c12_binary_motion_ttx",
}
EVAL_EPOCHS = [9, 14, 19, 24, 29]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(SHORT_MANIFEST.read_text(encoding="utf-8"))
    generated = []
    for row in manifest["controls"]:
        if row["cell"] not in KEEP_CELLS:
            continue
        source = Path(row["config"])
        cfg = copy.deepcopy(yaml.safe_load(source.read_text(encoding="utf-8")))
        experiment = (
            f"dsec_fullres_w15_two_contrib_{row['cell']}_"
            f"nb0ep29_ft30_{STAMP}"
        )
        cfg["experiment"] = experiment
        cfg["loader"]["n_epochs"] = 30
        cfg["optimizer"]["milestones"] = [20, 25]
        cfg["runtime"]["force_save_epochs"] = EVAL_EPOCHS
        cfg["runtime"]["state_save_epochs"] = [29]
        cfg["runtime"]["save_only_force_epochs"] = True
        cfg["runtime"]["full_resolution_protocol"] = (
            "date_same_parent_nb0ep29_fresh_optimizer_480x640_"
            "window2x15x15_full30"
        )
        cfg["runtime"]["two_contribution_cell"] = row["cell"]
        cfg["note"] = (
            "DATE clean same-parent two-contribution control. Fresh optimizer, seed0, "
            "30 full-resolution epochs from frozen NB0 ep29; evaluate only "
            f"predeclared epochs {EVAL_EPOCHS}. Local valid825 is not official DSEC test."
        )
        path = GENERATED / f"{experiment}.yml"
        path.write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        generated.append(
            {
                **{key: value for key, value in row.items() if key != "config_sha256"},
                "source_short_config": str(source.resolve()),
                "source_short_config_sha256": sha256(source),
                "config": str(path.resolve()),
                "config_sha256": sha256(path),
            }
        )

    if {row["cell"] for row in generated} != KEEP_CELLS:
        raise RuntimeError("full30 control set is incomplete")
    output = {
        "schema": "date_two_contribution_full30_manifest_v1",
        "parent_checkpoint": manifest["parent_checkpoint"],
        "parent_checkpoint_sha256": manifest["parent_checkpoint_sha256"],
        "source_short_manifest": str(SHORT_MANIFEST.resolve()),
        "source_short_manifest_sha256": sha256(SHORT_MANIFEST),
        "shared_protocol": {
            **manifest["shared_protocol"],
            "epochs": 30,
            "optimizer_milestones": [20, 25],
            "evaluation_epochs": EVAL_EPOCHS,
            "selection_metric": "local_valid825_AEE",
        },
        "controls": generated,
        "claim_boundary": (
            "same-parent fresh-optimizer full30 local-valid825 causal ablation; "
            "not official DSEC hidden-test"
        ),
    }
    output_path = GENERATED / f"date_two_contribution_full30_{STAMP}.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
