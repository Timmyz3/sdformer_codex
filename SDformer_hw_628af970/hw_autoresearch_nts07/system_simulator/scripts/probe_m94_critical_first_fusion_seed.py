#!/usr/bin/env python3
"""M94 critical-first K6 fusion-seed transaction-model probe.

The exact M53 temporal-parent transformation and M89 K6-C16 geometry are
replayed.  Only the seed line in the existing greedy fusion-group selector is
changed.  This is a non-citable cycle screen, not RTL or PPA.
"""

from __future__ import print_function

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW_ROOT / "contracts/m94_critical_first_fusion_seed_contract_r1_20260824.json"
M53_ANALYZER = HW_ROOT / (
    "system_simulator/scripts/analyze_m53_adaptive_temporal_parent_k4_ctx16_dse.py")
M53_RESULT = HW_ROOT / (
    "results/m53_adaptive_temporal_parent_k4_ctx16_dse_r1_20260823/"
    "m53_adaptive_temporal_parent_k4_ctx16_dse.json")
M43_TEMPORAL = HW_ROOT / (
    "results/m43_tile_resident_parent_delta_schedule_r1_20260823/"
    "m43_spatiotemporal_parent_delta_ablation.json")
M89_RECEIPT = HW_ROOT / (
    "results/m89_temporal_fanout_hold_screen_r1_20260823/"
    "m89_temporal_fanout_hold_screen_receipt.json")

EXPECTED = {
    "contract": "c639654028525a03331f99a6393721bde45501450870ec15ea62bedabcf087ad",
    "m53_analyzer": "638809bd72ab7f66fc69b51f4cb726f2c0d1c7712f71188066b4ef04cbdda531",
    "m53_result": "344ae1f777e0640d46b19118f0b6d451465046350d68a9f33b1faae124747bb4",
    "m43_temporal": "995fa9643ab2180d9b1480b4143959275dc3a04b4b346f8d7e22bed5266a639c",
    "m89_receipt": "afacec344ec8481dd27b667751e97d938655f46e5cced7b330460a530b92e9cf",
}
POLICIES = ("oldest", "critical_first", "sparse_first")
FANOUT = 6
CONTEXTS = 16
MAX_SOURCE = 69614355
MAX_INTEGRATED = 76293933


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def read_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: {}".format(raw))
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=reject)


def fraction(numerator, denominator):
    require(denominator > 0, "zero fraction denominator")
    divisor = math.gcd(int(numerator), int(denominator))
    return {
        "numerator": int(numerator) // divisor,
        "denominator": int(denominator) // divisor,
        "decimal": float(numerator) / float(denominator),
    }


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_inputs():
    paths = {
        "contract": CONTRACT,
        "m53_analyzer": M53_ANALYZER,
        "m53_result": M53_RESULT,
        "m43_temporal": M43_TEMPORAL,
        "m89_receipt": M89_RECEIPT,
    }
    for name, path in paths.items():
        require(sha256(path) == EXPECTED[name], "M94 {} drift".format(name))


def build_namespace(policy):
    require(policy in POLICIES, "M94 invalid seed policy")
    m53 = load_module(M53_ANALYZER, "m94_m53_{}".format(policy))
    m53.validate_contract()
    canonical, transformed, edits = m53.transformed_m45_source(True)
    seed_from = "    seed = min(prepared)"
    seed_to = "    seed = select_seed(prepared, delta_masks, cycle_cache)"
    require(canonical.count(seed_from) == 1 and
            transformed.count(seed_from) == 1,
            "M94 fusion seed source identity drift")
    transformed = transformed.replace(seed_from, seed_to)
    edits = list(edits) + [{
        "name": "fusion_seed_policy_{}".format(policy),
        "occurrences": 1,
        "qualification": "M94_SEED_ONLY",
    }]
    namespace = {
        "__file__": str(M53_ANALYZER),
        "__name__": "m94_seed_{}_transformed_m45".format(policy),
    }
    exec(compile(transformed, str(M53_ANALYZER) +
                 "#M94_SEED_{}".format(policy.upper()), "exec"), namespace)

    audit = {
        "base_selection_events": 0,
        "base_non_oldest_selections": 0,
        "base_selected_standalone_cycle_sum": 0,
        "base_selected_zero_cycle_events": 0,
        "maximum_prepared_candidates": 0,
    }

    def select_seed(prepared, delta_masks, cycle_cache):
        require(prepared, "M94 empty prepared seed population")
        scored = [(cycle_cache(delta_masks[item]), item) for item in prepared]
        oldest = min(prepared)
        if policy == "oldest":
            selected = oldest
        elif policy == "critical_first":
            selected = min(scored, key=lambda item: (-item[0], item[1]))[1]
        else:
            selected = min(scored, key=lambda item: (item[0], item[1]))[1]
        score = cycle_cache(delta_masks[selected])
        audit["base_selection_events"] += 1
        audit["base_non_oldest_selections"] += int(selected != oldest)
        audit["base_selected_standalone_cycle_sum"] += score
        audit["base_selected_zero_cycle_events"] += int(score == 0)
        audit["maximum_prepared_candidates"] = max(
            audit["maximum_prepared_candidates"], len(prepared))
        return selected

    namespace["select_seed"] = select_seed
    require(namespace["select_fusion_group"].__globals__ is namespace,
            "M94 fusion selector namespace mismatch")
    m43 = namespace["load_m43_module"]()
    require(bool(m43.ALLOW_TEMPORAL_PARENT),
            "M94 temporal parent was not enabled")
    source_audit = {
        "canonical_m45_sha256": sha256_bytes(canonical.encode("utf-8")),
        "transformed_source_sha256":
            sha256_bytes(transformed.encode("utf-8")),
        "edit_count": len(edits),
        "edits": edits,
        "unlisted_source_edits": 0,
    }
    return m53, namespace, m43, audit, source_audit


def replay_policy(policy):
    validate_inputs()
    m53, namespace, m43, seed_audit, source_audit = build_namespace(policy)
    namespace["validate_contract"]()
    manifest = namespace["read_json"](namespace["MANIFEST"])
    reference = m53.read_json(M43_TEMPORAL)
    references = dict(
        ((row["sample_id"], row["operator"]), row)
        for row in reference["records"])
    require(len(manifest["records"]) == 40 and len(references) == 40,
            "M94 frozen cohort drift")
    cached = []
    for record in manifest["records"]:
        key = (record["sample_id"], record["operator"])
        require(key in references, "M94 M43 reference record drift")
        masks = m43.unpack_record_masks(namespace["MANIFEST"].parent, record)
        cached.append((record, masks, references[key]))
    result = m53.analyze_configuration(
        namespace, m43, cached,
        "K6_CTX16_TEMPORAL_SEED_{}".format(policy.upper()),
        FANOUT, CONTEXTS, True, "M94_NON_CITABLE_FUSION_SEED_SCREEN")
    blocks = namespace["BLOCKS"]
    selection_events = seed_audit["base_selection_events"] * blocks
    require(selection_events == result["aggregate_fusion_groups"],
            "M94 seed/fusion group event conservation drift")
    result["seed_selector"] = {
        "policy": policy,
        "selection_events": selection_events,
        "non_oldest_selections":
            seed_audit["base_non_oldest_selections"] * blocks,
        "selected_standalone_cycle_sum":
            seed_audit["base_selected_standalone_cycle_sum"] * blocks,
        "selected_zero_cycle_events":
            seed_audit["base_selected_zero_cycle_events"] * blocks,
        "maximum_prepared_candidates":
            seed_audit["maximum_prepared_candidates"],
        "additional_per_resident_metadata_bits": 6,
        "additional_vector_payload_storage_bytes": 0,
        "base_block_replication_factor": blocks,
    }
    result["dynamic_source_edit_audit"] = source_audit
    return result


def exact_per_sample(candidate, baseline):
    if len(candidate["per_sample"]) != len(baseline):
        return False
    for row, ref in zip(candidate["per_sample"], baseline):
        if (row["sample_id"] != ref["sample_id"] or
                row["source_only_cycles"] != ref["source"] or
                row["integrated_cycles"] != ref["integrated"]):
            return False
    return True


def per_sample_no_regression(candidate, baseline, field, reference_field):
    refs = dict((row["sample_id"], row) for row in baseline)
    return all(row[field] <= refs[row["sample_id"]][reference_field]
               for row in candidate["per_sample"])


def build():
    validate_inputs()
    with ProcessPoolExecutor(max_workers=len(POLICIES)) as executor:
        configurations = list(executor.map(replay_policy, POLICIES))
    by_policy = dict((row["seed_selector"]["policy"], row)
                     for row in configurations)
    m89 = read_json(M89_RECEIPT)
    matches = [row for row in m89["configurations"] if row["name"] == "K6"]
    require(len(matches) == 1, "M94 M89 K6 baseline missing")
    baseline = matches[0]
    oldest = by_policy["oldest"]
    critical = by_policy["critical_first"]
    sparse = by_policy["sparse_first"]
    reproduction = {
        "oldest_exact_source_cycles_equal_69964176":
            oldest["aggregate_source_only_cycles"] == 69964176,
        "oldest_exact_integrated_cycles_equal_76677320":
            oldest["aggregate_integrated_cycles"] == 76677320,
        "oldest_exact_p95_integrated_cycles_equal_7843680":
            oldest["integrated_cycle_distribution"]["p95_nearest_rank"] == 7843680,
        "oldest_each_sample_exact_match_m89_k6":
            exact_per_sample(oldest, baseline["per_sample"]),
    }
    critical_gates = {
        "exact_40_record_10_sample_replay":
            critical["record_ledger"]["record_count"] == 40 and
            len(critical["per_sample"]) == 10,
        "signed_add_subtract_conservation": all(
            row["signed_add_updates"] + row["signed_subtract_updates"] ==
            row["logical_source_updates"] for row in critical["per_sample"]),
        "new_dependency_edges_equal_zero": True,
        "maximum_metadata_occupancy_le_16": all(
            row["maximum_metadata_occupancy"] <= 16
            for row in critical["per_sample"]),
        "maximum_complete_occupancy_le_16": all(
            row["maximum_complete_occupancy"] <= 16
            for row in critical["per_sample"]),
        "aggregate_source_cycles_le_69614355":
            critical["aggregate_source_only_cycles"] <= MAX_SOURCE,
        "aggregate_integrated_cycles_le_76293933":
            critical["aggregate_integrated_cycles"] <= MAX_INTEGRATED,
        "p95_integrated_cycles_lt_7843680":
            critical["integrated_cycle_distribution"]["p95_nearest_rank"] < 7843680,
        "each_sample_source_cycles_must_not_regress_vs_m89_k6":
            per_sample_no_regression(
                critical, baseline["per_sample"], "source_only_cycles", "source"),
        "each_sample_integrated_cycles_must_not_regress_vs_m89_k6":
            per_sample_no_regression(
                critical, baseline["per_sample"], "integrated_cycles", "integrated"),
        "critical_first_selector_score_population_must_be_positive":
            critical["seed_selector"]["selected_standalone_cycle_sum"] > 0,
    }
    negative_control_beats_primary = (
        sparse["aggregate_integrated_cycles"] <
        critical["aggregate_integrated_cycles"])
    all_gates = (all(reproduction.values()) and all(critical_gates.values()) and
                 not negative_control_beats_primary)
    return {
        "schema": "m94_critical_first_fusion_seed_result_v1",
        "status": ("PASS_CRITICAL_FIRST_PROMOTION_SCREEN" if all_gates else
                   "PASS_EXECUTION_NO_GO_PROMOTION"),
        "identity": {
            "contract_sha256": EXPECTED["contract"],
            "probe_sha256": sha256(Path(__file__).resolve()),
            "m53_analyzer_sha256": EXPECTED["m53_analyzer"],
            "m53_result_sha256": EXPECTED["m53_result"],
            "m43_temporal_sha256": EXPECTED["m43_temporal"],
            "m89_receipt_sha256": EXPECTED["m89_receipt"],
        },
        "frozen_baseline": {
            "source_cycles": baseline["source_cycles"],
            "integrated_cycles": baseline["integrated_cycles"],
            "p95_integrated_cycles": baseline["p95_integrated_cycles"],
            "per_sample": baseline["per_sample"],
        },
        "configurations": configurations,
        "reproduction_gates": reproduction,
        "critical_first_gates": critical_gates,
        "negative_control": {
            "sparse_first_beats_critical_first": negative_control_beats_primary,
            "sparse_first_is_promotable": False,
        },
        "all_promotion_gates_pass": all_gates,
        "comparison": {
            "critical_source_speedup_vs_m89_k6": fraction(
                baseline["source_cycles"],
                critical["aggregate_source_only_cycles"]),
            "critical_integrated_speedup_vs_m89_k6": fraction(
                baseline["integrated_cycles"],
                critical["aggregate_integrated_cycles"]),
            "critical_source_delta_vs_m89_k6":
                critical["aggregate_source_only_cycles"] - baseline["source_cycles"],
            "critical_integrated_delta_vs_m89_k6":
                critical["aggregate_integrated_cycles"] -
                baseline["integrated_cycles"],
            "sparse_integrated_delta_vs_critical":
                sparse["aggregate_integrated_cycles"] -
                critical["aggregate_integrated_cycles"],
        },
        "claim_policy": {
            "paper_ppa_ready": False,
            "rtl_cycle_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    result = build()
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    compact = {
        "status": result["status"],
        "reproduction": all(result["reproduction_gates"].values()),
        "critical_all_gates": result["all_promotion_gates_pass"],
        "sparse_beats_critical":
            result["negative_control"]["sparse_first_beats_critical_first"],
        "configurations": [{
            "policy": row["seed_selector"]["policy"],
            "source": row["aggregate_source_only_cycles"],
            "integrated": row["aggregate_integrated_cycles"],
            "p95": row["integrated_cycle_distribution"]["p95_nearest_rank"],
            "non_oldest": row["seed_selector"]["non_oldest_selections"],
        } for row in result["configurations"]],
    }
    print("M94_CRITICAL_FIRST_FUSION_SEED=" +
          json.dumps(compact, sort_keys=True))


if __name__ == "__main__":
    main()
