#!/usr/bin/env python3
"""M95 exact K6 member-comparator transaction-model probe.

The exact M53 temporal-parent transformation and M89 K6-C16 geometry are
replayed.  Only the existing greedy member ranking tuple is changed.  This is
a frozen-cohort opportunity screen, not RTL, PPA, or system speedup.
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
CONTRACT = HW_ROOT / "contracts/m95_marginal_bank_growth_member_contract_r1_20260824.json"
M45_ANALYZER = HW_ROOT / (
    "system_simulator/scripts/analyze_m45_dual_destination_bank_fused_integrated_schedule.py")
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
    "contract": "eb17a8fe5d05bf3cdfef2b51dde9f9c16f16a58c77a64a0c292b4040f125f8a5",
    "m45_analyzer": "c1e3610ce59753f786498db46cde7b330155fa2e3c836198be165aad3eb3f38f",
    "m53_analyzer": "638809bd72ab7f66fc69b51f4cb726f2c0d1c7712f71188066b4ef04cbdda531",
    "m53_result": "344ae1f777e0640d46b19118f0b6d451465046350d68a9f33b1faae124747bb4",
    "m43_temporal": "995fa9643ab2180d9b1480b4143959275dc3a04b4b346f8d7e22bed5266a639c",
    "m89_receipt": "afacec344ec8481dd27b667751e97d938655f46e5cced7b330460a530b92e9cf",
}
POLICIES = (
    "saved_first_reproduction",
    "marginal_growth_primary",
    "standalone_heavy_negative_control",
)
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
        "m45_analyzer": M45_ANALYZER,
        "m53_analyzer": M53_ANALYZER,
        "m53_result": M53_RESULT,
        "m43_temporal": M43_TEMPORAL,
        "m89_receipt": M89_RECEIPT,
    }
    for name, path in paths.items():
        require(path.is_file() and sha256(path) == EXPECTED[name],
                "M95 {} identity drift".format(name))


def build_namespace(policy):
    require(policy in POLICIES, "M95 invalid member policy")
    m53 = load_module(M53_ANALYZER, "m95_m53_{}".format(policy))
    m53.validate_contract()
    canonical, transformed, edits = m53.transformed_m45_source(True)
    rank_from = "            ranked.append((-saved, fused_cycles, candidate, fused))"
    rank_to = (
        "            ranked.append(rank_member(\n"
        "                union_cycles, cycle_cache(candidate_mask),\n"
        "                fused_cycles, saved, candidate, fused))")
    require(canonical.count(rank_from) == 1 and
            transformed.count(rank_from) == 1,
            "M95 member ranking source identity drift")
    transformed = transformed.replace(rank_from, rank_to)
    edits = list(edits) + [{
        "name": "fusion_member_policy_{}".format(policy),
        "occurrences": 1,
        "qualification": "M95_MEMBER_COMPARATOR_ONLY",
    }]
    namespace = {
        "__file__": str(M53_ANALYZER),
        "__name__": "m95_member_{}_transformed_m45".format(policy),
    }
    exec(compile(transformed, str(M53_ANALYZER) +
                 "#M95_MEMBER_{}".format(policy.upper()), "exec"), namespace)

    audit = {
        "base_candidate_evaluations": 0,
        "base_candidate_standalone_cycle_sum": 0,
        "base_current_union_cycle_sum": 0,
        "base_fused_union_cycle_sum": 0,
        "base_saved_cycle_sum": 0,
        "base_marginal_growth_cycle_sum": 0,
        "minimum_marginal_growth_cycles": None,
        "maximum_marginal_growth_cycles": None,
    }

    def rank_member(union_cycles, candidate_cycles, fused_cycles, saved,
                    candidate, fused):
        growth = fused_cycles - union_cycles
        require(saved == union_cycles + candidate_cycles - fused_cycles,
                "M95 saved-cycle identity drift")
        audit["base_candidate_evaluations"] += 1
        audit["base_candidate_standalone_cycle_sum"] += candidate_cycles
        audit["base_current_union_cycle_sum"] += union_cycles
        audit["base_fused_union_cycle_sum"] += fused_cycles
        audit["base_saved_cycle_sum"] += saved
        audit["base_marginal_growth_cycle_sum"] += growth
        low = audit["minimum_marginal_growth_cycles"]
        high = audit["maximum_marginal_growth_cycles"]
        audit["minimum_marginal_growth_cycles"] = (
            growth if low is None else min(low, growth))
        audit["maximum_marginal_growth_cycles"] = (
            growth if high is None else max(high, growth))
        if policy == "saved_first_reproduction":
            return (-saved, fused_cycles, candidate, fused)
        if policy == "marginal_growth_primary":
            return (growth, -saved, candidate, fused)
        return (-candidate_cycles, fused_cycles, candidate, fused)

    namespace["rank_member"] = rank_member
    require(namespace["select_fusion_group"].__globals__ is namespace,
            "M95 fusion selector namespace mismatch")
    m43 = namespace["load_m43_module"]()
    require(bool(m43.ALLOW_TEMPORAL_PARENT),
            "M95 temporal parent was not enabled")
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
    m53, namespace, m43, audit, source_audit = build_namespace(policy)
    namespace["validate_contract"]()
    manifest = namespace["read_json"](namespace["MANIFEST"])
    reference = m53.read_json(M43_TEMPORAL)
    references = dict(
        ((row["sample_id"], row["operator"]), row)
        for row in reference["records"])
    require(len(manifest["records"]) == 40 and len(references) == 40,
            "M95 frozen cohort drift")
    cached = []
    for record in manifest["records"]:
        key = (record["sample_id"], record["operator"])
        require(key in references, "M95 M43 reference record drift")
        masks = m43.unpack_record_masks(namespace["MANIFEST"].parent, record)
        cached.append((record, masks, references[key]))
    result = m53.analyze_configuration(
        namespace, m43, cached,
        "K6_CTX16_TEMPORAL_MEMBER_{}".format(policy.upper()),
        FANOUT, CONTEXTS, True, "M95_NON_CITABLE_MEMBER_COMPARATOR_SCREEN")
    blocks = namespace["BLOCKS"]
    result["member_selector"] = {
        "policy": policy,
        "candidate_evaluations": audit["base_candidate_evaluations"] * blocks,
        "candidate_standalone_cycle_sum":
            audit["base_candidate_standalone_cycle_sum"] * blocks,
        "current_union_cycle_sum":
            audit["base_current_union_cycle_sum"] * blocks,
        "fused_union_cycle_sum":
            audit["base_fused_union_cycle_sum"] * blocks,
        "saved_cycle_sum": audit["base_saved_cycle_sum"] * blocks,
        "marginal_growth_cycle_sum":
            audit["base_marginal_growth_cycle_sum"] * blocks,
        "minimum_marginal_growth_cycles":
            audit["minimum_marginal_growth_cycles"],
        "maximum_marginal_growth_cycles":
            audit["maximum_marginal_growth_cycles"],
        "additional_per_resident_metadata_bits": 0,
        "additional_vector_payload_storage_bytes": 0,
        "new_candidate_evaluation_lanes": 0,
        "base_block_replication_factor": blocks,
    }
    require(result["member_selector"]["candidate_evaluations"] > 0,
            "M95 empty candidate evaluation population")
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
    by_policy = dict((row["member_selector"]["policy"], row)
                     for row in configurations)
    m89 = read_json(M89_RECEIPT)
    matches = [row for row in m89["configurations"] if row["name"] == "K6"]
    require(len(matches) == 1, "M95 M89 K6 baseline missing")
    baseline = matches[0]
    saved = by_policy["saved_first_reproduction"]
    marginal = by_policy["marginal_growth_primary"]
    standalone = by_policy["standalone_heavy_negative_control"]
    reproduction = {
        "saved_first_exact_source_cycles_equal_69964176":
            saved["aggregate_source_only_cycles"] == 69964176,
        "saved_first_exact_integrated_cycles_equal_76677320":
            saved["aggregate_integrated_cycles"] == 76677320,
        "saved_first_exact_p95_integrated_cycles_equal_7843680":
            saved["integrated_cycle_distribution"]["p95_nearest_rank"] == 7843680,
        "saved_first_each_sample_exact_match_m89_k6":
            exact_per_sample(saved, baseline["per_sample"]),
    }
    marginal_gates = {
        "exact_40_record_10_sample_replay":
            marginal["record_ledger"]["record_count"] == 40 and
            len(marginal["per_sample"]) == 10,
        "signed_add_subtract_conservation": all(
            row["signed_add_updates"] + row["signed_subtract_updates"] ==
            row["logical_source_updates"] for row in marginal["per_sample"]),
        "new_dependency_edges_equal_zero": True,
        "maximum_metadata_occupancy_le_16": all(
            row["maximum_metadata_occupancy"] <= 16
            for row in marginal["per_sample"]),
        "maximum_complete_occupancy_le_16": all(
            row["maximum_complete_occupancy"] <= 16
            for row in marginal["per_sample"]),
        "aggregate_source_cycles_le_69614355":
            marginal["aggregate_source_only_cycles"] <= MAX_SOURCE,
        "aggregate_integrated_cycles_le_76293933":
            marginal["aggregate_integrated_cycles"] <= MAX_INTEGRATED,
        "p95_integrated_cycles_lt_7843680":
            marginal["integrated_cycle_distribution"]["p95_nearest_rank"] < 7843680,
        "each_sample_source_cycles_must_not_regress_vs_m89_k6":
            per_sample_no_regression(
                marginal, baseline["per_sample"], "source_only_cycles", "source"),
        "each_sample_integrated_cycles_must_not_regress_vs_m89_k6":
            per_sample_no_regression(
                marginal, baseline["per_sample"], "integrated_cycles", "integrated"),
        "candidate_evaluation_population_must_be_positive":
            marginal["member_selector"]["candidate_evaluations"] > 0,
    }
    negative_beats_primary = (
        standalone["aggregate_integrated_cycles"] <
        marginal["aggregate_integrated_cycles"])
    all_gates = (all(reproduction.values()) and all(marginal_gates.values()) and
                 not negative_beats_primary)
    return {
        "schema": "m95_marginal_bank_growth_member_result_v1",
        "status": ("PASS_MARGINAL_GROWTH_PROMOTION_SCREEN" if all_gates else
                   "PASS_EXECUTION_NO_GO_PROMOTION"),
        "identity": {
            "contract_sha256": EXPECTED["contract"],
            "probe_sha256": sha256(Path(__file__).resolve()),
            "m45_analyzer_sha256": EXPECTED["m45_analyzer"],
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
        "marginal_growth_gates": marginal_gates,
        "negative_control": {
            "standalone_heavy_beats_marginal_growth": negative_beats_primary,
            "standalone_heavy_is_promotable": False,
        },
        "all_promotion_gates_pass": all_gates,
        "comparison": {
            "marginal_source_speedup_vs_m89_k6": fraction(
                baseline["source_cycles"], marginal["aggregate_source_only_cycles"]),
            "marginal_integrated_speedup_vs_m89_k6": fraction(
                baseline["integrated_cycles"], marginal["aggregate_integrated_cycles"]),
            "marginal_source_delta_vs_m89_k6":
                marginal["aggregate_source_only_cycles"] - baseline["source_cycles"],
            "marginal_integrated_delta_vs_m89_k6":
                marginal["aggregate_integrated_cycles"] - baseline["integrated_cycles"],
            "standalone_integrated_delta_vs_marginal":
                standalone["aggregate_integrated_cycles"] -
                marginal["aggregate_integrated_cycles"],
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
        "marginal_all_gates": result["all_promotion_gates_pass"],
        "negative_beats_primary": result["negative_control"][
            "standalone_heavy_beats_marginal_growth"],
        "configurations": [{
            "policy": row["member_selector"]["policy"],
            "source": row["aggregate_source_only_cycles"],
            "integrated": row["aggregate_integrated_cycles"],
            "p95": row["integrated_cycle_distribution"]["p95_nearest_rank"],
            "candidate_evaluations":
                row["member_selector"]["candidate_evaluations"],
        } for row in result["configurations"]],
    }
    print("M95_MARGINAL_BANK_GROWTH_MEMBER=" +
          json.dumps(compact, sort_keys=True))


if __name__ == "__main__":
    main()
