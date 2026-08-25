#!/usr/bin/env python3
"""Non-citable M89 probe for screening selective K4 fusion-hold policies.

This deliberately reuses the exact M53 frozen cohort and only replaces the
canonical two-cycle K4 wait block with a disabled guard.  It is a screening
probe, not a production result or an admitted speedup claim.
"""

from __future__ import print_function

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
M53_PATH = HW_ROOT / (
    "system_simulator/scripts/"
    "analyze_m53_adaptive_temporal_parent_k4_ctx16_dse.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


POLICY_CONDITION = {
    "nohold": "False and fanout_k == 4 and len(prepared) < fanout_k",
    "only_one": "fanout_k == 4 and len(prepared) == 1",
    "only_two": "fanout_k == 4 and len(prepared) == 2",
    "only_three": "fanout_k == 4 and len(prepared) == 3",
    "up_to_two": "fanout_k == 4 and len(prepared) <= 2",
    "two_or_three": "fanout_k == 4 and 2 <= len(prepared) < fanout_k",
}


def build(policy):
    m53 = load_module(M53_PATH, "m89_probe_m53")
    m53.validate_contract()
    canonical, transformed, edits = m53.transformed_m45_source(True)
    hold_from = '''        if fanout_k == 4 and len(prepared) < fanout_k:
            future_resident = [item[0] for item in resident.values()
                               if item[0] > now]
            if future_resident:
                next_cycle = min(future_resident)
                if next_cycle - now <= 2:
                    counts["fusion_hold_wait_cycles"] += next_cycle - now
                    now = next_cycle
                    continue
'''
    require(policy in POLICY_CONDITION, "M89 unknown hold policy")
    hold_to = hold_from.replace(
        "fanout_k == 4 and len(prepared) < fanout_k",
        POLICY_CONDITION[policy])
    require(canonical.count(hold_from) == 1,
            "M89 canonical M45 fusion-hold identity drift")
    require(transformed.count(hold_from) == 1,
            "M89 transformed M53 fusion-hold identity drift")
    transformed = transformed.replace(hold_from, hold_to)
    edits = list(edits) + [{
        "name": "selective_k4_two_cycle_fusion_hold_{}".format(policy),
        "occurrences": 1,
        "qualification": "M89_NON_CITABLE_SCREENING_PROBE",
    }]

    namespace = {
        "__file__": str(M53_PATH),
        "__name__": "m89_k4_nohold_transformed_m45",
    }
    exec(compile(transformed, str(M53_PATH) + "#M89_NOHOLD", "exec"),
         namespace)
    m43 = namespace["load_m43_module"]()
    require(bool(m43.ALLOW_TEMPORAL_PARENT),
            "M89 temporal parent was not enabled")

    manifest = namespace["read_json"](namespace["MANIFEST"])
    reference_path = HW_ROOT / (
        "results/m43_tile_resident_parent_delta_schedule_r1_20260823/"
        "m43_spatiotemporal_parent_delta_ablation.json")
    reference = m53.read_json(reference_path)
    reference_records = dict(
        ((row["sample_id"], row["operator"]), row)
        for row in reference["records"])
    require(len(manifest["records"]) == 40 and len(reference_records) == 40,
            "M89 frozen cohort drift")
    cached = []
    for record in manifest["records"]:
        key = (record["sample_id"], record["operator"])
        require(key in reference_records,
                "M89 reference record identity mismatch")
        masks = m43.unpack_record_masks(namespace["MANIFEST"].parent, record)
        cached.append((record, masks, reference_records[key]))

    result = m53.analyze_configuration(
        namespace, m43, cached,
        "K4_CTX16_TEMPORAL_{}_PROBE".format(policy.upper()),
        4, 16, True, "M89_NON_CITABLE_SCREENING_PROBE")
    result["m89_probe"] = {
        "status": "NON_CITABLE_SCREENING_ONLY",
        "canonical_m45_sha256": m53.sha256_bytes(canonical.encode("utf-8")),
        "transformed_source_sha256":
            m53.sha256_bytes(transformed.encode("utf-8")),
        "edits": edits,
        "policy": policy,
        "paper_ppa_ready": False,
        "rtl_cycle_speedup": False,
        "system_speedup": False,
        "headline": False,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--policy", choices=sorted(POLICY_CONDITION),
                        default="nohold")
    args = parser.parse_args()
    result = build(args.policy)
    compact = {
        "source": result["aggregate_source_only_cycles"],
        "integrated": result["aggregate_integrated_cycles"],
        "p95": result["integrated_cycle_distribution"]["p95_nearest_rank"],
        "fusion_hold": sum(row["fusion_hold_wait_cycles"]
                           for row in result["per_sample"]),
        "max_metadata": max(row["maximum_metadata_occupancy"]
                            for row in result["per_sample"]),
        "max_complete": max(row["maximum_complete_occupancy"]
                            for row in result["per_sample"]),
        "policy": args.policy,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    print("M89_K4_NOHOLD_PROBE=" + json.dumps(compact, sort_keys=True))


if __name__ == "__main__":
    main()
