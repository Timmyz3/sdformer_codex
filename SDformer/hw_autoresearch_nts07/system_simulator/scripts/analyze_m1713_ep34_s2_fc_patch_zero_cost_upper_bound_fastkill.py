#!/usr/bin/env python3
"""M1713: fail-closed zero-cost S2 upper bound on ep34 FC/patch work.

This audit deliberately gives S2 an impossible advantage: for each candidate
object it deletes *all* product service that remains after exact C2 zero-source
suppression, while charging no metadata, bound calculation, AEE, port, bank,
burst, terminal, commit, or pipeline cost.  Therefore a ratio below 1.15x is a
mathematical NO-GO for that object under this capture/model boundary.

The capture has no retained FC/patch value payload.  Consequently this script
does not estimate an epsilon-dependent drop rate and does not claim cycles,
traffic, accuracy, energy, or performance.  It only computes an Amdahl-style
upper bound from sealed aggregate nonzero-source statistics.

Compatible with CPython 3.6.
"""
from __future__ import print_function

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
CAPTURE = ROOT / (
    "hw_autoresearch_nts07/results/"
    "m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_"
    "20260831")
TRACE = CAPTURE / "execution_trace.json"
ORDERED = CAPTURE / "unified_ordered_records.jsonl"
MANIFEST = CAPTURE / "manifest.json"
SUMS = CAPTURE / "SHA256SUMS"
OUTER = CAPTURE / "SHA256SUMS.seal.sha256"
DOCS359 = ROOT / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "trace": "55759fb2e723b4d1a5902a84b95682245b8fde70b21187f1fe1ad9fa08c4ffaa",
    "ordered": "5956085b196979848c3d283744396ea3b0a38a268fb21af0eaecb53e87fc6c9c",
    "manifest": "3ab8431e3d7d17d6933c0b87da4a3405e87c97ccc302a27c78491b0a02491d6d",
    "sums": "f7f7a08696611875837196b990575453141b5e8edbf6d4aae61f7db1ed238b8e",
    "outer": "7cf434b834d30c003153eef8e83e70d574b1c5a7d20ca4c2208902c6e0c76eed",
    "checkpoint": "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
LANES = 96
SOURCES_PER_CYCLE = 8
ISSUE_PRODUCTS = LANES * SOURCES_PER_CYCLE
SPEEDUP_GATE = 1.15
EXPECTED_RECORDS = {"patch_embed": 320, "fc1": 480, "fc2": 480}
EXPECTED_MODULES = {"patch_embed": 8, "fc1": 12, "fc2": 12}
EXPECTED_SEQUENCE_COUNTS = {
    "interlaken_01_a": 10,
    "thun_01_b": 10,
    "zurich_city_09_a": 10,
    "zurich_city_12_a": 10,
}


class AuditError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise AuditError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           AuditError("nonfinite JSON: " + token)))
    return value


def verify_capture():
    for label, path in (("trace", TRACE), ("ordered", ORDERED),
                        ("manifest", MANIFEST), ("sums", SUMS),
                        ("outer", OUTER)):
        require(path.is_file() and not path.is_symlink(),
                label + " must be a regular non-symlink")
        require(sha256(path) == EXPECTED[label], label + " SHA drift")
    manifest = strict_json(MANIFEST)
    require(DOCS359.is_file() and not DOCS359.is_symlink() and
            sha256(DOCS359) == EXPECTED["docs359"],
            "docs/359 identity drift")
    selected = manifest.get("identity", {}).get("selection", {}).get(
        "selected", {})
    checkpoint = selected.get("checkpoint", {})
    require(checkpoint.get("sha256") == EXPECTED["checkpoint"],
            "checkpoint identity drift")
    require(manifest.get("cohort", {}).get("population") == 40 and
            len(manifest.get("cohort", {}).get("samples", [])) == 40,
            "capture sample count drift")
    return manifest


def category(row):
    name = row.get("name", "")
    operator = row.get("operator")
    if operator == "Linear" and ".mlp.fc1" in name:
        return "fc1"
    if operator == "Linear" and ".mlp.fc2" in name:
        return "fc2"
    if operator == "Conv2d" and "patch_embed" in name:
        return "patch_embed"
    return None


def ordered_crosscheck():
    rows = {}
    category_counts = dict((key, 0) for key in EXPECTED_RECORDS)
    retained_counts = dict((key, 0) for key in EXPECTED_RECORDS)
    with ORDERED.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            cat = row.get("category")
            if cat not in EXPECTED_RECORDS:
                continue
            key = (row.get("global_sample_id"), row.get("name"))
            require(key not in rows, "duplicate ordered target record")
            inp = row.get("input", {})
            active = inp.get("active")
            elements = inp.get("elements")
            require(type(active) is int and type(elements) is int and
                    0 <= active <= elements and inp.get("nonfinite") == 0,
                    "invalid ordered activity at line " + str(line_no))
            payload = row.get("payload", {})
            retained = payload.get("retained") is True
            rows[key] = (cat, active, elements, retained)
            category_counts[cat] += 1
            retained_counts[cat] += int(retained)
    require(category_counts == EXPECTED_RECORDS,
            "ordered FC/patch population drift")
    require(all(value == 0 for value in retained_counts.values()),
            "FC/patch retained-payload boundary drift")
    return rows, retained_counts


def upper_bound(total_cycles, removable_cycles):
    candidate = total_cycles - removable_cycles
    require(candidate > 0, "upper-bound candidate must retain unaffected work")
    return float(total_cycles) / float(candidate)


def minimum_drop_fraction(target_share):
    needed_removed_share = 1.0 - 1.0 / SPEEDUP_GATE
    return needed_removed_share / target_share


def analyze():
    manifest = verify_capture()
    ordered, retained_counts = ordered_crosscheck()
    trace = strict_json(TRACE)
    require(type(trace) is list and len(trace) == 7360,
            "execution trace population drift")

    family_work = dict((key, 0) for key in EXPECTED_RECORDS)
    module_work = {}
    record_counts = dict((key, 0) for key in EXPECTED_RECORDS)
    module_samples = {}
    sequence_samples = {}
    matched = set()
    for row in trace:
        cat = category(row)
        if cat is None:
            continue
        name = row.get("name")
        sample_id = row.get("sample_id")
        sequence = row.get("sequence_key")
        sample_key = row.get("sample_key")
        require(type(sample_id) is int and 0 <= sample_id < 40 and
                type(sequence) is str and type(sample_key) is str,
                "invalid sample identity")
        key = (sample_id, name)
        require(key in ordered and ordered[key][0] == cat,
                "execution/ordered target join failure")
        active = row.get("input_active")
        elements = row.get("input_elements")
        dense_macs = row.get("dense_macs")
        require(type(active) is int and type(elements) is int and
                type(dense_macs) is int and 0 <= active <= elements and
                elements > 0 and dense_macs > 0 and
                dense_macs % elements == 0,
                "invalid exact zero-source work fields")
        require((active, elements) == ordered[key][1:3],
                "execution/ordered activity mismatch")
        fanout = dense_macs // elements
        work = active * fanout
        family_work[cat] += work
        module_work[name] = module_work.get(name, 0) + work
        record_counts[cat] += 1
        module_samples.setdefault(name, set()).add(sample_id)
        sequence_samples.setdefault(sequence, set()).add(sample_id)
        require(key not in matched, "duplicate execution target record")
        matched.add(key)

    require(record_counts == EXPECTED_RECORDS,
            "execution FC/patch population drift")
    actual_modules = dict((cat, len([name for name in module_work
                                    if category({"name": name,
                                                 "operator": "Linear" if
                                                 ".mlp.fc" in name else
                                                 "Conv2d"}) == cat]))
                          for cat in EXPECTED_MODULES)
    require(actual_modules == EXPECTED_MODULES, "module inventory drift")
    require(all(len(samples) == 40 for samples in module_samples.values()),
            "each target module must cover all forty samples")
    require(dict((key, len(value)) for key, value in sequence_samples.items()) ==
            EXPECTED_SEQUENCE_COUNTS, "sequence cohort drift")

    total_work = sum(family_work.values())
    total_cycles = int(math.ceil(float(total_work) / ISSUE_PRODUCTS))
    family_rows = []
    for name in ("patch_embed", "fc1", "fc2"):
        work = family_work[name]
        share = float(work) / float(total_work)
        removable_cycles = int(math.floor(float(work) / ISSUE_PRODUCTS))
        ratio = upper_bound(total_cycles, removable_cycles)
        required = minimum_drop_fraction(share)
        family_rows.append({
            "object": name,
            "exact_c2_zero_source_product_work": work,
            "target_work_share": share,
            "impossible_zero_cost_removed_issue_cycles": removable_cycles,
            "zero_cost_complete_elimination_upper_bound": ratio,
            "minimum_fraction_of_family_remaining_work_to_drop_for_1p15x":
                required,
            "direct_no_go_below_1p15": ratio < SPEEDUP_GATE,
            "decision": ("NO_GO_AS_S2_TARGET" if ratio < SPEEDUP_GATE else
                         "RETAIN_ONLY_FOR_VALUE_PAYLOAD_PAIRED_GATE"),
        })

    module_rows = []
    for name, work in sorted(module_work.items(),
                             key=lambda item: (-item[1], item[0])):
        share = float(work) / float(total_work)
        removable_cycles = int(math.floor(float(work) / ISSUE_PRODUCTS))
        ratio = upper_bound(total_cycles, removable_cycles)
        module_rows.append({
            "module": name,
            "exact_c2_zero_source_product_work": work,
            "target_work_share": share,
            "zero_cost_complete_elimination_upper_bound": ratio,
            "minimum_fraction_of_module_remaining_work_to_drop_for_1p15x":
                minimum_drop_fraction(share),
            "direct_no_go_below_1p15": ratio < SPEEDUP_GATE,
        })

    require(all(row["direct_no_go_below_1p15"] for row in module_rows),
            "expected standalone module fast-kill changed")
    fc2 = [row for row in family_rows if row["object"] == "fc2"][0]
    require(fc2["direct_no_go_below_1p15"],
            "expected FC2 family fast-kill changed")

    return {
        "schema": "m1713_ep34_s2_fc_patch_zero_cost_upper_bound_fastkill_r1_v1",
        "status": "PASS_ZERO_COST_UPPER_BOUND__FC2_AND_ALL_STANDALONE_MODULES_NO_GO__FC1_PATCH_CONDITIONAL__NO_PERFORMANCE",
        "date_cst": "2026-09-01",
        "identity": {
            "analyzer_sha256": sha256(Path(__file__)),
            "capture_manifest_sha256": EXPECTED["manifest"],
            "capture_execution_trace_sha256": EXPECTED["trace"],
            "capture_ordered_sha256": EXPECTED["ordered"],
            "capture_manifest_file_sha256": EXPECTED["sums"],
            "capture_outer_seal_file_sha256": EXPECTED["outer"],
            "checkpoint_sha256": EXPECTED["checkpoint"],
            "docs359_sha256": EXPECTED["docs359"],
            "samples": 40,
            "sequence_sample_counts": EXPECTED_SEQUENCE_COUNTS,
            "capture_status": manifest.get("status"),
        },
        "method": {
            "baseline": "exact C2 zero-source suppression",
            "work": "sum(input_active * dense_macs / input_elements)",
            "issue_width": {"lanes": LANES,
                            "signed_sources_per_cycle": SOURCES_PER_CYCLE,
                            "products_per_ideal_issue_cycle": ISSUE_PRODUCTS},
            "denominator": "FC1 + FC2 + patch Conv2d target work only; all other network work excluded to make the upper bound more optimistic",
            "candidate": "delete 100% of the named object's remaining nonzero-source product work",
            "free_costs": ["CCBS metadata", "bound calculation", "AEE",
                           "ports", "banks", "bursts", "pipeline",
                           "terminal", "commit", "all object work"],
            "gate": SPEEDUP_GATE,
        },
        "population": {
            "records": record_counts,
            "modules": actual_modules,
            "fc_patch_retained_value_payloads": retained_counts,
            "exact_c2_zero_source_product_work": total_work,
            "ideal_k8x96_issue_cycles": total_cycles,
        },
        "family_upper_bounds": family_rows,
        "module_upper_bounds": module_rows,
        "decision": {
            "fc2": "DIRECT_NO_GO_EVEN_IF_ALL_REMAINING_FC2_WORK_IS_FREE",
            "standalone_modules": "DIRECT_NO_GO_32_OF_32",
            "fc1": "CONDITIONAL_ONLY_AS_CROSS_LAYER_FAMILY; NEED RETAINED VALUES, PAIRED AEE, AND SAME_RESOURCE REPLAY",
            "patch_embed": "CONDITIONAL_ONLY_AS_CROSS_LAYER_FAMILY; NEED RETAINED VALUES, PAIRED AEE, AND SAME_RESOURCE REPLAY",
            "rtl_authorized": False,
        },
        "limitations": [
            "No FC/patch values are retained in M1458, so no epsilon-dependent S2 drop rate can be measured.",
            "Complete object deletion is intentionally impossible and only forms an upper bound.",
            "This is a source-product issue proxy, not address-timed or cycle simulation.",
            "No metadata, AEE, port, bank, burst, terminal, commit, energy, RTL, VCS, synthesis, or system cost is present.",
        ],
        "claim_boundary": {
            "read_only": True,
            "zero_cost_upper_bound": True,
            "paired_aee": False,
            "cycles": False,
            "traffic": False,
            "energy": False,
            "speedup": False,
            "system_speedup": False,
            "rtl": False,
            "vcs": False,
            "eda": False,
            "paper_result": False,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = analyze()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        for row in result["family_upper_bounds"]:
            print("%s %.6fx required_drop=%.6f %s" % (
                row["object"],
                row["zero_cost_complete_elimination_upper_bound"],
                row["minimum_fraction_of_family_remaining_work_to_drop_for_1p15x"],
                row["decision"]))
        print(result["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
