#!/usr/bin/env python3
"""Independent entry-point replay for the high-risk M45-r2 samples.

This intentionally bypasses the M45-r2 result builder.  It loads the frozen
M40 masks and M43 record identities, invokes only the pinned r1 transaction
scheduler for K2/C8, and compares the returned record ledger with the sealed
M45-r2 canonical result.  It never writes the canonical result.
"""

from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import types


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
R1_ANALYZER = HW_ROOT / (
    "system_simulator/scripts/analyze_m45_dual_destination_bank_fused_integrated_schedule.py")
CANONICAL = HW_ROOT / (
    "results/m45_dual_destination_bank_fused_integrated_schedule_r2_20260823/"
    "m45_r2_context8_primary_schedule.json")
EXPECTED_R1_SHA256 = (
    "c1e3610ce59753f786498db46cde7b330155fa2e3c836198be165aad3eb3f38f")
EXPECTED_CANONICAL_SHA256 = (
    "0f16e75601fdb18f31f9bc36f6aae8a17a9e62a20f5c07e18226562e9ba0d37c")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_r1():
    require(sha256(R1_ANALYZER) == EXPECTED_R1_SHA256,
            "pinned M45-r1 analyzer drift")
    source = R1_ANALYZER.read_text(encoding="utf-8")
    old_measure = "min(16, len(ready)))"
    old_assert = ('counts["maximum_metadata_occupancy"] <= '
                  'METADATA_FIFO_ENTRIES,')
    require(source.count(old_measure) == 1 and source.count(old_assert) == 1,
            "cannot apply isolated raw-ready-depth instrumentation")
    source = source.replace(old_measure, "len(ready))")
    source = source.replace(old_assert, "True,")
    module = types.ModuleType("m45_r2_independent_replay_r1")
    module.__file__ = str(R1_ANALYZER)
    exec(compile(source, str(R1_ANALYZER), "exec"), module.__dict__)
    module.INDEPENDENT_INSTRUMENTED_SOURCE_SHA256 = hashlib.sha256(
        source.encode("utf-8")).hexdigest()
    return module


def build(sample_ids):
    require(sha256(CANONICAL) == EXPECTED_CANONICAL_SHA256,
            "canonical M45-r2 result drift")
    r1 = load_r1()
    r1.validate_contract()
    manifest = r1.read_json(r1.MANIFEST)
    m43_result = r1.read_json(r1.M43_RESULT)
    m43_records = dict(((row["sample_id"], row["operator"]), row)
                       for row in m43_result["records"])
    canonical = read_json(CANONICAL)
    primary = next(item for item in canonical["configurations"]
                   if item["name"] == "K2_CTX8_PRIMARY")
    expected = dict(((row["sample_id"], row["operator"]), row)
                    for row in primary["records"])
    m43 = r1.load_m43_module()
    replayed = []
    for record in manifest["records"]:
        if record["sample_id"] not in sample_ids:
            continue
        key = (record["sample_id"], record["operator"])
        masks = m43.unpack_record_masks(r1.MANIFEST.parent, record)
        actual = r1.analyze_record(m43, masks, m43_records[key], 2, 8)
        actual["sample_id"] = key[0]
        actual["operator"] = key[1]
        raw_ready_depth = actual["maximum_metadata_occupancy"]
        comparable = dict(actual)
        comparable["maximum_metadata_occupancy"] = expected[key][
            "maximum_metadata_occupancy"]
        require(comparable == expected[key],
                "targeted K2/C8 replay mismatch outside raw-ready metric: {}".format(key))
        replayed.append({
            "sample_id": key[0],
            "operator": key[1],
            "record_exact_match": True,
            "source_only_cycles": actual["source_only_cycles"],
            "integrated_cycles": actual["integrated_cycles"],
            "maximum_complete_occupancy": actual["maximum_complete_occupancy"],
            "reported_clamped_metadata_occupancy": expected[key]["maximum_metadata_occupancy"],
            "raw_spatial_dag_ready_depth": raw_ready_depth,
            "maximum_resident_occupancy": actual["maximum_resident_occupancy"],
        })
        print("[independent M45-r2 replay] sample={} operator={} PASS".format(
            key[0], key[1]))
    require(len(replayed) == len(sample_ids) * 4,
            "targeted replay population drift")
    return {
        "schema": "m45_r2_independent_targeted_replay_receipt_v1",
        "status": "PASS_EXACT_RECORD_REPLAY_K2_CTX8_SAMPLES_3_AND_7",
        "identity": {
            "canonical_result_sha256": sha256(CANONICAL),
            "r1_analyzer_sha256": sha256(R1_ANALYZER),
            "replay_script_sha256": sha256(Path(__file__).resolve()),
            "instrumented_in_memory_source_sha256":
                r1.INDEPENDENT_INSTRUMENTED_SOURCE_SHA256,
        },
        "scope": {
            "configuration": "K2_CTX8_PRIMARY",
            "sample_ids": sorted(sample_ids),
            "records": len(replayed),
            "qualification": "same pinned transaction scheduler through an independent entry point, with only the reporting clamp and its assertion removed in memory to expose raw DAG-ready depth; not an independent scheduling algorithm or RTL replay",
        },
        "records": replayed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite replay receipt")
    payload = build(set((3, 7)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
