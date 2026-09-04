#!/usr/bin/env python3
"""Independent exact-rational and tamper validator for producer-owned M65."""

from __future__ import print_function

import argparse
import copy
import datetime
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_REVIEW_SHA256 = \
    "e4c6593a3508d59b1dfa1df92360b651eb1c2ad5b987a1c942cbb1db1ff1bdcf"
CONTRACT_REL = Path("contracts/m65_m53_m63_nonoverlap_joint_contract_r1_20260823.json")
ANALYZER_REL = Path("system_simulator/scripts/analyze_m65_m53_m63_nonoverlap_joint.py")
PRODUCER_VALIDATOR_REL = Path("system_simulator/scripts/validate_m65_m53_m63_nonoverlap_joint.py")
RESULT_REL = Path("results/m65_m53_m63_nonoverlap_joint_r1_20260823/m65_m53_m63_nonoverlap_joint_result_r1.json")
M25_REL = Path("results/m25_resource_bounded_tiled_cycles_r5_rowclosed_20260822/m25_resource_bounded_tiled_cycles.json")
M39_REL = Path("results/m39_remaining_bottleneck_r3_20260822/m39_remaining_bottleneck.json")
M53_REL = Path("results/m53_adaptive_temporal_parent_k4_ctx16_dse_r1_20260823/m53_adaptive_temporal_parent_k4_ctx16_dse.json")
M63_REL = Path("results/m63_linear_k4_spatiotemporal_full_network_opportunity_r1_20260823/m63_linear_k4_spatiotemporal_full_network_opportunity_result_r2.json")
M4_STATEFUL = Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs/m4_stateful_wall_cycles_full_s10_20260821/m4_stateful_wall_cycles.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant {}".format(raw))

    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key {}".format(key))
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def ceil_fraction(value):
    return -(-value.numerator // value.denominator)


def floor_fraction(value):
    return value.numerator // value.denominator


def resolve_binding(name):
    path = Path(name)
    return path if path.is_absolute() else ROOT / path


def validate_review(review, review_path):
    require(sha256_path(review_path) == EXPECTED_REVIEW_SHA256,
            "independent review SHA")
    require(review["schema"] == "m65_independent_hammer_review_v1" and
            review["status"] ==
            "ARITHMETIC_PASS_SPATIAL_AND_TEMPORAL_NO_GO_PRODUCER_SEAL_NEEDS_HARDENING",
            "review schema/status")
    require(review["reviewer_is_producer"] is False and
            review["producer_evidence_modified"] is False and
            review["headline"] is False and
            review["system_speedup"] is False, "review boundary")
    score = review["date_oriented_score"]
    require(score["score_0_to_100"] == 60 and
            sum(score["subscores"].values()) == 60, "DATE score")
    require(review["issues"]["P0"] == [] and
            len(review["issues"]["P1"]) == 5 and
            len(review["issues"]["P2"]) == 3, "issue counts")
    observed = {}
    for name, expected in review["exact_sha_bindings"].items():
        path = resolve_binding(name)
        require(path.is_file() and sha256_path(path) == expected,
                "review binding {}".format(name))
        observed[name] = expected
    return observed


def expected_input_ledger(contract):
    return dict((row["path"], row["sha256"]) for row in contract["inputs"])


def validate_claim_boundary(contract, result):
    policy = result["claim_boundary"]
    require(policy == contract["claim_policy"], "contract/result claim policy")
    require(policy["headline"] is False and
            policy["system_speedup_admitted"] is False and
            policy["paper_ppa_ready"] is False and
            policy["power_or_energy_admitted"] is False,
            "claim Boolean promotion")
    forbidden = " ".join(policy["forbidden"]).lower()
    for phrase in ("system speedup", "exact per-operator",
                   "temporal k4 across all 24", "date headline"):
        require(phrase in forbidden, "claim forbidden text {}".format(phrase))


def exact_lineage(m25, m4):
    local_m4 = m4["variants"]["local"]["per_identity"]["H67"]
    speed_numerator = int(local_m4["p1_sparse_wall_cycles"])
    speed_denominator = int(local_m4["stateful_wall_cycles"])
    require(speed_numerator == 5158877 and speed_denominator == 860504,
            "exact M4 speed lineage")
    local = m25["compute_envelopes"]["local"]["10"]
    require(local["m4_profiled_eligible_cycles"] == 327131854 and
            local["accelerated_m4_cycles"] == 54565804 and
            local["effective_m4_speed"] ==
            float(Fraction(speed_numerator, speed_denominator)),
            "M25 local lineage")
    total_raw = 327131854
    captured_raw = 154631318
    complement_raw = total_raw - captured_raw
    total_ideal = Fraction(total_raw * speed_denominator, speed_numerator)
    captured_ideal = Fraction(captured_raw * speed_denominator, speed_numerator)
    complement_ideal = Fraction(complement_raw * speed_denominator,
                                speed_numerator)
    global_ceil = ceil_fraction(total_ideal)
    captured_floor = floor_fraction(captured_ideal)
    captured_ceil = ceil_fraction(captured_ideal)
    complement_floor = floor_fraction(complement_ideal)
    complement_ceil = ceil_fraction(complement_ideal)
    require(global_ceil == 54565804 and
            (captured_floor, captured_ceil) == (25792603, 25792604) and
            (complement_floor, complement_ceil) == (28773200, 28773201) and
            captured_floor + complement_ceil == global_ceil and
            captured_ceil + complement_floor == global_ceil,
            "one-token exact global-ceil apportionment")
    return {
        "speed_numerator": speed_numerator,
        "speed_denominator": speed_denominator,
        "total_ideal": total_ideal,
        "captured_ideal": captured_ideal,
        "complement_ideal": complement_ideal,
        "global_ceil": global_ceil,
        "captured_floor": captured_floor,
        "captured_ceil": captured_ceil,
        "complement_floor": complement_floor,
        "complement_ceil": complement_ceil,
    }


def validate_semantics(contract, result, m25, m39, m53, m63, m4):
    require(contract["schema"] ==
            "m65_m53_m63_nonoverlap_joint_contract_v1" and
            result["schema"] == "m65_m53_m63_nonoverlap_joint_result_v1" and
            result["status"] ==
            "PASS_ONE_CYCLE_TIGHT_NONOVERLAP_SPATIAL_K4_NO_GO",
            "schema/status")
    require(result["identity"]["contract_sha256"] == sha256_path(ROOT / CONTRACT_REL) and
            result["identity"]["analyzer_sha256"] == sha256_path(ROOT / ANALYZER_REL),
            "producer code identity")
    ledger = expected_input_ledger(contract)
    require(result["identity"]["inputs_sha256"] == ledger,
            "result input SHA ledger")
    for name, expected in ledger.items():
        require(sha256_path(ROOT / name) == expected,
                "contract input SHA {}".format(name))
    validate_claim_boundary(contract, result)
    exact = exact_lineage(m25, m4)
    captured = result["captured_linear_decomposition"]
    require(captured["modules"] == 24 and
            captured["raw_eligible_cycles"] == 154631318 and
            captured["complement_raw_eligible_cycles"] == 172500536 and
            captured["raw_partition_conserved"] is True and
            abs(captured["ideal_inherited_cycles_before_global_ceil"] -
                float(exact["captured_ideal"])) < 1e-8 and
            captured["inherited_integral_interval"] == {
                "minimum": 25792603, "maximum": 25792604,
                "width_cycles": 1}, "captured interval")

    local_rows = [row for row in m39["conditional_dse"][
        "four_bottleneck_rows"] if row["line"] == "Local"]
    require(len(local_rows) == 2 and
            all(row["m38_model_substituted_ideal_before_scope_cycles"] ==
                268455448 and row["before_cycles"] == 79630957
                for row in local_rows), "M39 Local rows")
    outside = 268455448 - 79630957
    require(outside == 188824491, "M39 outside reconstruction")
    m53_model = m53["conditional_frozen_compute_model"]
    require(m53_model["fixed_compute_reference_cycles"] == 620868243 and
            m53_model["pair_p95_nearest_rank_cycles"] == 9798504 and
            outside + 2636515 + 9798504 ==
            m53_model["conditional_total_cycles"] == 201259510,
            "M53 denominator reconstruction")
    reconstruction = result["reconstruction"]
    require(reconstruction["outside_four_bottleneck_cycles"] == outside and
            reconstruction["m53_conditional_cycles"] == 201259510 and
            abs(reconstruction["m53_conditional_ratio_not_system_speedup"] -
                620868243.0 / 201259510.0) < 1e-15,
            "result reconstruction")

    categories = m63["m39_category_ledger"]
    require(sum(row["captured_modules"] for row in categories.values()) == 24 and
            sum(row["captured_m39_activity_cycles"]
                for row in categories.values()) == 154631318 and
            all(row["captured_m39_activity_cycles"] ==
                row["dual_line_category_eligible_cycles"]
                for row in categories.values()), "M63 M25-eligible capture")
    m39_names = set(row["name"] for row in m39[
        "four_bottleneck_event_late_scale_model"]["operator_census"])
    m63_names = set(row["module_name"] for row in m63["per_module"])
    require(len(m39_names) == 4 and len(m63_names) == 24 and
            not (m39_names & m63_names), "M53/M63 operator non-overlap")
    replacement = m63["aggregate_configurations"]["spatial_K4"][
        "serialized_integrated_cycle_distribution"]["p95_nearest_rank"]
    require(replacement == 28535569 and
            m63["aggregate_configurations"]["spatial_K4"][
                "capacity_feasible_modules"] == 24 and
            m63["aggregate_configurations"]["temporal_K4"][
                "capacity_infeasible_modules"] == 11,
            "M63 replacement/capacity")

    joint = result["spatial_k4_nonoverlap_joint"]
    require(joint["replacement_p95_cycles"] == replacement and
            joint["decision"] == "NO_GO_AS_ADDITIVE_M53_ACCELERATOR",
            "joint decision")
    endpoint_by_inherited = dict((row["inherited_linear_cycles"], row)
                                 for row in joint["endpoints"])
    require(set(endpoint_by_inherited) == {25792603, 25792604},
            "joint endpoint inherited set")
    expected_cycles = []
    expected_ratios = []
    expected_deltas = []
    for inherited in (25792603, 25792604):
        denominator = outside - inherited + 2636515 + 9798504 + replacement
        ratio = Fraction(620868243, denominator)
        delta = replacement - inherited
        row = endpoint_by_inherited[inherited]
        require(row["joint_conditional_cycles"] == denominator and
                abs(row["conditional_ratio_not_system_speedup"] -
                    float(ratio)) < 1e-15 and
                row["replacement_minus_inherited_cycles"] == delta,
                "joint endpoint {}".format(inherited))
        expected_cycles.append(denominator)
        expected_ratios.append(float(ratio))
        expected_deltas.append(delta)
    require(joint["joint_conditional_cycle_interval"] == {
                "minimum": min(expected_cycles), "maximum": max(expected_cycles)} and
            joint["conditional_ratio_interval_not_system_speedup"] == {
                "minimum": min(expected_ratios), "maximum": max(expected_ratios)} and
            joint["replacement_regression_cycles_interval"] == {
                "minimum": min(expected_deltas), "maximum": max(expected_deltas)},
            "joint intervals")
    require((min(expected_cycles), max(expected_cycles)) ==
            (204002475, 204002476) and
            (min(expected_deltas), max(expected_deltas)) ==
            (2742965, 2742966), "joint integer anchors")
    temporal = result["temporal_k4"]
    require(temporal == {
        "all24_joint_admitted": False,
        "capacity_infeasible_modules": 11,
        "decision": "KILLED_BY_11_OF_24_LOCAL_CAPACITY_FAILURES"},
        "temporal 11/24 kill")
    validate_target_gates(result, exact["captured_ceil"])
    return {"exact": exact, "outside": outside,
            "joint_cycles": [min(expected_cycles), max(expected_cycles)],
            "joint_ratios": [min(expected_ratios), max(expected_ratios)],
            "regression": [min(expected_deltas), max(expected_deltas)]}


def validate_target_gates(result, inherited_ceil):
    gates = result["conditional_target_gates"]
    numerator = Fraction(620868243, 1)
    base_minus_inherited = Fraction(201259510 - inherited_ceil, 1)
    targets = {
        "3p1": Fraction(31, 10),
        "3p2": Fraction(16, 5),
        "3p3": Fraction(33, 10),
        "3p45": Fraction(69, 20),
        "preserve_m53": Fraction(620868243, 201259510),
    }
    for name, target in targets.items():
        maximum = numerator / target - base_minus_inherited
        reduction = max(Fraction(0, 1), Fraction(28535569, 1) - maximum)
        row = gates[name]
        require(abs(row["target_ratio"] - float(target)) < 1e-15 and
                abs(row["maximum_spatial_k4_replacement_cycles_not_system"] -
                    float(maximum)) < 1e-7 and
                row["current_replacement_cycles"] == 28535569 and
                abs(row["additional_reduction_required_cycles"] -
                    float(reduction)) < 1e-7,
                "target gate {}".format(name))


def write_fixture(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def run_producer_validator(contract, result, analyzer_bytes):
    with tempfile.TemporaryDirectory(prefix="m65_independent_attack_") as raw:
        directory = Path(raw)
        contract_path = directory / "contract.json"
        result_path = directory / "result.json"
        analyzer_path = directory / "analyzer.py"
        write_fixture(contract_path, contract)
        analyzer_path.write_bytes(analyzer_bytes)
        result["identity"]["contract_sha256"] = sha256_path(contract_path)
        result["identity"]["analyzer_sha256"] = sha256_path(analyzer_path)
        write_fixture(result_path, result)
        completed = subprocess.run([
            sys.executable, str(ROOT / PRODUCER_VALIDATOR_REL),
            "--contract", str(contract_path), "--analyzer", str(analyzer_path),
            "--result", str(result_path)], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, universal_newlines=True)
        return {"returncode": completed.returncode,
                "accepted": completed.returncode == 0,
                "diagnostic": (completed.stdout + completed.stderr).strip()[-400:]}


def producer_attack_suite(contract, result):
    analyzer = (ROOT / ANALYZER_REL).read_bytes()
    attacks = []

    def execute(name, expected_accept, mutate):
        c = copy.deepcopy(contract)
        r = copy.deepcopy(result)
        a = analyzer
        c, r, a = mutate(c, r, a)
        outcome = run_producer_validator(c, r, a)
        require(outcome["accepted"] == expected_accept,
                "unexpected producer attack outcome {}".format(name))
        outcome.update({"name": name,
                        "classification": ("SURVIVED_PRODUCER_VALIDATOR"
                                           if outcome["accepted"] else
                                           "REJECTED_BY_PRODUCER_VALIDATOR")})
        attacks.append(outcome)

    def ratio(c, r, a):
        r["spatial_k4_nonoverlap_joint"][
            "conditional_ratio_interval_not_system_speedup"] = {
                "minimum": 99.0, "maximum": 100.0}
        return c, r, a

    def input_ledger(c, r, a):
        first = sorted(r["identity"]["inputs_sha256"])[0]
        r["identity"]["inputs_sha256"][first] = "0" * 64
        return c, r, a

    def analyzer_drift(c, r, a):
        return c, r, a + b"\n# independent drift attack\n"

    def forbidden(c, r, a):
        c["claim_policy"]["forbidden"] = []
        r["claim_boundary"] = copy.deepcopy(c["claim_policy"])
        return c, r, a

    def system(c, r, a):
        c["claim_policy"]["system_speedup_admitted"] = True
        r["claim_boundary"] = copy.deepcopy(c["claim_policy"])
        return c, r, a

    def temporal(c, r, a):
        r["temporal_k4"]["all24_joint_admitted"] = True
        return c, r, a

    execute("ratio_interval_forgery", True, ratio)
    execute("result_input_sha_ledger_forgery", True, input_ledger)
    execute("analyzer_drift_rebound_in_result", True, analyzer_drift)
    execute("contract_and_result_forbidden_text_removal", True, forbidden)
    execute("system_speedup_boolean_promotion", False, system)
    execute("temporal_all24_kill_removal", False, temporal)
    return attacks


def independent_guard(contract, result):
    errors = []
    if result.get("identity", {}).get("analyzer_sha256") != \
            "97699b76706b27841c11b1d60971195bdd0a7fb1df8d169f65a3f4b59de35a31":
        errors.append("analyzer exact SHA")
    if result.get("identity", {}).get("inputs_sha256") != expected_input_ledger(contract):
        errors.append("input SHA ledger")
    policy = result.get("claim_boundary", {})
    forbidden = " ".join(policy.get("forbidden", [])).lower()
    if policy != contract.get("claim_policy") or "system speedup" not in forbidden:
        errors.append("claim boundary")
    if policy.get("system_speedup_admitted") is not False:
        errors.append("system speedup")
    joint = result.get("spatial_k4_nonoverlap_joint", {})
    if joint.get("conditional_ratio_interval_not_system_speedup") != {
            "minimum": 3.0434348404673286,
            "maximum": 3.0434348553859456}:
        errors.append("ratio interval")
    if joint.get("joint_conditional_cycle_interval") != {
            "minimum": 204002475, "maximum": 204002476}:
        errors.append("cycle interval")
    if joint.get("decision") != "NO_GO_AS_ADDITIVE_M53_ACCELERATOR":
        errors.append("NO-GO decision")
    if result.get("temporal_k4") != {
            "all24_joint_admitted": False,
            "capacity_infeasible_modules": 11,
            "decision": "KILLED_BY_11_OF_24_LOCAL_CAPACITY_FAILURES"}:
        errors.append("temporal kill")
    return errors


def independent_tamper_suite(contract, result):
    mutations = []
    item = copy.deepcopy(result); item["spatial_k4_nonoverlap_joint"][
        "conditional_ratio_interval_not_system_speedup"]["minimum"] = 99.0
    mutations.append(("ratio_interval_forgery", contract, item))
    item = copy.deepcopy(result); item["identity"]["inputs_sha256"][
        sorted(item["identity"]["inputs_sha256"])[0]] = "0" * 64
    mutations.append(("input_SHA_ledger_forgery", contract, item))
    item = copy.deepcopy(result); item["identity"]["analyzer_sha256"] = "0" * 64
    mutations.append(("analyzer_SHA_rebind", contract, item))
    con = copy.deepcopy(contract); con["claim_policy"]["forbidden"] = []
    item = copy.deepcopy(result); item["claim_boundary"] = con["claim_policy"]
    mutations.append(("forbidden_text_removal", con, item))
    con = copy.deepcopy(contract); con["claim_policy"][
        "system_speedup_admitted"] = True
    item = copy.deepcopy(result); item["claim_boundary"] = con["claim_policy"]
    mutations.append(("system_speedup_promotion", con, item))
    item = copy.deepcopy(result); item["spatial_k4_nonoverlap_joint"][
        "joint_conditional_cycle_interval"]["minimum"] -= 1
    mutations.append(("cycle_endpoint_forgery", contract, item))
    item = copy.deepcopy(result); item["spatial_k4_nonoverlap_joint"][
        "decision"] = "GO"
    mutations.append(("NO_GO_removal", contract, item))
    item = copy.deepcopy(result); item["temporal_k4"][
        "all24_joint_admitted"] = True
    mutations.append(("temporal_kill_removal", contract, item))
    receipts = []
    for name, con, res in mutations:
        errors = independent_guard(con, res)
        require(errors, "independent tamper survived {}".format(name))
        receipts.append({"name": name, "result": "REJECTED",
                         "diagnostic": errors[0]})
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    arguments = parser.parse_args()
    require(not arguments.receipt.exists(), "refusing existing independent receipt")
    review = strict_json(arguments.review)
    bindings = validate_review(review, arguments.review)
    contract = strict_json(ROOT / CONTRACT_REL)
    result = strict_json(ROOT / RESULT_REL)
    m25 = strict_json(ROOT / M25_REL)
    m39 = strict_json(ROOT / M39_REL)
    m53 = strict_json(ROOT / M53_REL)
    m63 = strict_json(ROOT / M63_REL)
    m4 = strict_json(M4_STATEFUL)
    summary = validate_semantics(contract, result, m25, m39, m53, m63, m4)
    require(not independent_guard(contract, result), "untampered guard failure")
    producer_attacks = producer_attack_suite(contract, result)
    independent_attacks = independent_tamper_suite(contract, result)
    exact = summary["exact"]
    receipt = {
        "schema": "m65_independent_hammer_validation_receipt_v1",
        "status": "PASS_EXACT_RATIONAL_NO_GO_WITH_PRODUCER_SEAL_P1",
        "generated_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "review": {"path": str(arguments.review.resolve()),
                   "sha256": sha256_path(arguments.review)},
        "validator": {"path": str(Path(__file__).resolve()),
                      "sha256": sha256_path(Path(__file__).resolve())},
        "producer_bindings_sha256": bindings,
        "exact_speed": {"numerator": exact["speed_numerator"],
                        "denominator": exact["speed_denominator"]},
        "global_ceil_recomputed": exact["global_ceil"],
        "L24_interval_recomputed": [exact["captured_floor"],
                                    exact["captured_ceil"]],
        "complement_interval_recomputed": [exact["complement_floor"],
                                            exact["complement_ceil"]],
        "joint_cycle_interval_recomputed": summary["joint_cycles"],
        "joint_ratio_interval_not_system_speedup": summary["joint_ratios"],
        "replacement_regression_interval_recomputed": summary["regression"],
        "temporal_capacity_infeasible_modules": 11,
        "producer_validator_attack_audit": producer_attacks,
        "independent_tamper_attacks": independent_attacks,
        "date_oriented_score_0_to_100": 60,
        "issues": {"P0": 0, "P1": 5, "P2": 3},
        "headline": False,
        "system_speedup": False,
        "claim_boundary": review["admission"],
    }
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    with arguments.receipt.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "receipt_sha256": sha256_path(arguments.receipt),
        "score": 60,
        "status": receipt["status"],
        "headline": False,
        "system_speedup": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
