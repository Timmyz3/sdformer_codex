#!/usr/bin/env python3
"""Build an isolated manifest-present fixture for the formal preflight runner."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import local5_erep_formal_preflight_v4 as preflight


OLD_MANIFEST = (
    ROOT
    / "results/local5_fullres_bb1e4_postg0_profile100_20260805/ordered_term_manifest.json"
)


def build_fixture(output: Path) -> None:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"fixture output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    selection_path = output / preflight.SELECTION_PLAN.name
    projection_path = output / preflight.PROJECTION_CONTRACT.name
    projection_payload = output / "checkpoint_projection_contract.npz"
    shutil.copyfile(preflight.SELECTION_PLAN, selection_path)
    shutil.copyfile(preflight.PROJECTION_CONTRACT, projection_path)
    shutil.copyfile(
        preflight.DEFAULT_PROFILE_DIR / "checkpoint_projection_contract.npz",
        projection_payload,
    )

    ordered_payload = output / "ordered_term_items.npz"
    np.savez_compressed(ordered_payload, fixture=np.asarray([1], dtype=np.uint8))
    cohort = output / "ordered_cohort.json"
    cohort.write_text('{"schema":"synthetic_positive_fixture_v1"}\n', encoding="utf-8")

    plan = json.loads(selection_path.read_text(encoding="utf-8"))
    windows = preflight.validate_selection_plan(plan)
    groups = []
    tag = 0
    for row in windows:
        stage = row["stage"]
        heads = preflight.STAGE_HEADS[stage]
        for head in range(heads):
            groups.append(
                {
                    "tag": tag,
                    "empty": False,
                    "sample": row["sample"],
                    "stage": stage,
                    "block": row["block"],
                    "window": row["window"],
                    "head": head,
                    "flat_group": row["window"] * heads + head,
                    "batch_windows": preflight.STAGE_WINDOWS[stage],
                    "heads": heads,
                    "lanes": 32,
                    "tokens": 450,
                    "time_planes": 2,
                    "plane_tokens": 225,
                    "spatial_side": 15,
                    "plane_execution": "plane_serial_drain",
                    "module": (
                        "sttmultires_unet.encoders.swin3d.layers."
                        f"{stage}.swin_blocks.{row['block']}.attn"
                    ),
                    "selection": "uniform_plan_window_all_heads_v1",
                    "ordered_item_sha256": f"{tag:064x}",
                }
            )
            tag += 1
    if tag != 13_800:
        raise ValueError("positive fixture did not expand to 13800 groups")

    manifest = copy.deepcopy(json.loads(OLD_MANIFEST.read_text(encoding="utf-8")))
    manifest.update(
        {
            "evidence_level": "post_g0",
            "payload_file": ordered_payload.name,
            "payload_sha256": preflight.sha256_file(ordered_payload),
            "cohort_file": cohort.name,
            "cohort_file_sha256": preflight.sha256_file(cohort),
            "cohort_sha256": plan["cohort_sha256"],
            "qualification": {"qualified": True},
            "sampling": {
                "method": "uniform_plan_window_all_heads_v1",
                "seed": 20260809,
                "selection_plan_sha256": preflight.SELECTION_PLAN_SHA256,
            },
            "groups": groups,
            "projection_contract_file": projection_path.name,
            "projection_contract_file_sha256": preflight.sha256_file(
                projection_path
            ),
            "projection_contract_payload": projection_payload.name,
            "projection_contract_payload_sha256": preflight.sha256_file(
                projection_payload
            ),
        }
    )
    (output / "ordered_term_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_fixture(args.output)
    print(f"PASS positive fixture={args.output.resolve()}")


if __name__ == "__main__":
    main()
