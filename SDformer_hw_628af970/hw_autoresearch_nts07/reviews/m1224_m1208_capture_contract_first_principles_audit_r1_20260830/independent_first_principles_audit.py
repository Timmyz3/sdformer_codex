#!/usr/bin/env python3
"""Read-only local consistency check for the sealed M1224 audit."""
import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    review = json.loads((HERE / "review.json").read_text())
    obs = json.loads((HERE / "remote_read_only_observation.json").read_text())
    policy = json.loads((ROOT / "hw_autoresearch_nts07/contracts/m1177_motion_checkpoint_parametric_unified_capture_source_contract_r2_20260830.json").read_text())
    atlif_path = ROOT / "hw_autoresearch_nts07/results/h67_ep35_full_network_ordered_trace_s10_20260821/atlif_activity.csv"
    with atlif_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    dead = [
        f"sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.{block}.attn.sn_v.spiking_neuron"
        for stage, blocks in enumerate((2, 2, 6, 2)) for block in range(blocks)
    ]
    static_counts = {key: int(value["modules"]) for key, value in policy["expected_inventory"].items()}
    live_counts = review["live_hook_inventory"]
    checks = {
        "docs359": sha(ROOT / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md") == DOCS359_SHA,
        "static259": sum(static_counts.values()) == 259 and static_counts["atlif"] == 105,
        "runtime93": len(rows) == 93 and all(int(row["calls"]) == 10 for row in rows),
        "dead12": len(dead) == 12 and all(name not in {row["name"] for row in rows} for name in dead),
        "live247": int(live_counts["sum"]) == 247 and sum(int(value) for key, value in live_counts.items() if key != "sum") == 247,
        "ordered9880": review["root_cause"]["arithmetic"]["expected_ordered_records_for_40_samples"] == 9880,
        "attention480": obs["attention_qk"]["records"] == 480 and obs["attention_qk"]["cartesian_40_by_12"],
        "payload640": obs["payloads"]["files"] == 640 and obs["payloads"]["stems"] == 320,
        "missing8": len(obs["staging_population"]["missing_release_members"]) == 8,
        "not_canonical": not obs["claim_boundary"]["canonical_capture"],
    }
    print(json.dumps({"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
