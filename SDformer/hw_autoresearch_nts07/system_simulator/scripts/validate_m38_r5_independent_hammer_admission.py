#!/usr/bin/env python3
"""Fail-closed independent hammer and admission validator for M38-r5."""

import argparse
import copy
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW_ROOT / "contracts/m38_rst_math_input_contract_r5_20260822.json"
ANALYZER = HW_ROOT / (
    "system_simulator/scripts/analyze_m38_rst_math_protocol_reachable_r5.py")
REGRESSION = HW_ROOT / (
    "system_simulator/tests/test_m38_rst_math_protocol_reachable_r5.py")
RESULT = HW_ROOT / (
    "results/m38_rst_math_protocol_reachable_r5_20260822/"
    "m38_rst_math_protocol_reachable_state.json")
SPECIFICATION = HW_ROOT / "rtl_m38/M38_RST_TYPE_STRICT_REFERENCE_R5.md"
REVIEW = HW_ROOT / (
    "results/m38_rst_math_protocol_reachable_r5_20260822/"
    "m38_r5_independent_hammer_go_review.json")
ADMISSION = HW_ROOT / (
    "contracts/m38_r5_independent_model_only_admission_r1_20260822.json")

EXPECTED_ANCHORS = {
    "contract": [
        "hw_autoresearch_nts07/contracts/"
        "m38_rst_math_input_contract_r5_20260822.json",
        "5ec623bba1023035dad68d695774168783efa45e1d7caafe63a16f7f16d32f6e"],
    "analyzer": [
        "hw_autoresearch_nts07/system_simulator/scripts/"
        "analyze_m38_rst_math_protocol_reachable_r5.py",
        "e88a1016c9e258f26c45ea2ea11e86c20afaeb78d1c5b5fea27f6928cb6f2748"],
    "regression": [
        "hw_autoresearch_nts07/system_simulator/tests/"
        "test_m38_rst_math_protocol_reachable_r5.py",
        "54b2afa852ead7952491e1454e8b0407d2bab3f31e33258c5955c11b5990aba9"],
    "result": [
        "hw_autoresearch_nts07/results/"
        "m38_rst_math_protocol_reachable_r5_20260822/"
        "m38_rst_math_protocol_reachable_state.json",
        "fd4e4769fe39ce0eadb3b7f9c7df5cdae7088933b564d22e58a0fa03867570de"],
    "specification": [
        "hw_autoresearch_nts07/rtl_m38/M38_RST_TYPE_STRICT_REFERENCE_R5.md",
        "4a305b3c9e4f6e4c7a83d7ce57aa672d6a2bfa3895fde7754e1db32b65baf8f7"],
    "r4_nogo_review": [
        "hw_autoresearch_nts07/results/"
        "m38_rst_math_protocol_reachable_r4_20260822/"
        "m38_r4_independent_hammer_nogo_review.json",
        "d45406ce03b486d98a33e0a8fdf486dc3c1e1bde662392d38aec82865857f14a"],
    "r3_nogo_review": [
        "hw_autoresearch_nts07/results/"
        "m38_rst_math_protocol_reachable_r3_20260822/"
        "m38_r3_independent_hammer_nogo_review.json",
        "d93335610d5d01d02a33507014188e9348f8a33e3c16035c7b51c640747ff9d6"],
}

EXPECTED_STATUS = (
    "PASS_M38_R5_TYPE_STRICT_MATH_PROTOCOL_COMPLETE_REACHABLE_STATE_ONLY")
FORBIDDEN_ADMISSION_KEYS = (
    "integrated_rtl", "integrated_rtl_vcs", "dc_sta_formality", "ppa",
    "power_energy", "memory_timing", "trained_coverage",
    "local_motion_system_cycles", "system_speedup", "headline")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_constant(raw):
    raise ValueError("non-standard JSON numeric constant: {}".format(raw))


def read_json(path):
    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: {}".format(key))
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook,
                      parse_constant=reject_constant)


def type_strict_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return (set(actual) == set(expected)
                and all(type_strict_equal(actual[key], expected[key])
                        for key in actual))
    if isinstance(actual, list):
        return (len(actual) == len(expected)
                and all(type_strict_equal(left, right)
                        for left, right in zip(actual, expected)))
    return actual == expected


def resolve(raw):
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_analyzer():
    spec = importlib.util.spec_from_file_location("m38_r5_hammer_target", str(ANALYZER))
    require(spec is not None and spec.loader is not None,
            "M38-r5 analyzer import failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_anchors():
    for name, pair in EXPECTED_ANCHORS.items():
        path = resolve(pair[0])
        require(path.is_file(), "missing {} anchor".format(name))
        require(sha256(path) == pair[1], "{} anchor drift".format(name))


def write_contract(root, payload, name):
    path = Path(root) / name
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def expect_reject(module, root, payload, name):
    path = write_contract(root, payload, name)
    try:
        module.build(path)
    except Exception:
        return
    raise ValueError("attack unexpectedly passed: {}".format(name))


def run_regression():
    process = subprocess.run(
        [sys.executable, "-m", "unittest", "-v", str(REGRESSION)],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True)
    require(process.returncode == 0, "M38-r5 regression failed")
    require("Ran 17 tests" in process.stdout and process.stdout.rstrip().endswith("OK"),
            "M38-r5 regression population/status drift")
    return {"passed": 17, "failed": 0, "errors": 0}


def run_repeat_build(module):
    result_a = module.build(CONTRACT)
    result_b = module.build(CONTRACT)
    bytes_a = (json.dumps(result_a, indent=2, sort_keys=True) + "\n").encode("utf-8")
    bytes_b = (json.dumps(result_b, indent=2, sort_keys=True) + "\n").encode("utf-8")
    frozen = RESULT.read_bytes()
    require(bytes_a == bytes_b == frozen, "M38-r5 repeat build/frozen result drift")
    require(result_a["status"] == EXPECTED_STATUS, "M38-r5 status drift")
    for key in (
            "integrated_rtl_admitted", "integrated_rtl_vcs_admitted",
            "dc_sta_formality_admitted", "area_power_energy_admitted",
            "memory_and_system_cycles_admitted", "system_speedup_admitted",
            "headline_admitted"):
        require(result_a["admission"].get(key) is False,
                "forbidden result admission opened: {}".format(key))
    return {
        "runs": 2, "a_equals_b": True, "both_equal_frozen_result": True,
        "sha256": hashlib.sha256(bytes_a).hexdigest()}


def run_attacks(module):
    contract = read_json(CONTRACT)
    counts = {}
    with tempfile.TemporaryDirectory(prefix="m38_r5_hammer_") as directory:
        root = Path(directory)

        semantic = []
        for key in (
                "state_fields", "context_modes", "stage1_phase_domain",
                "reconstruction_phase_domain", "reservation_relation",
                "writer_rule", "overflow_rule", "liveness_scope"):
            forged = copy.deepcopy(contract)
            forged["reachable_state_model"][key] = "FORGED"
            semantic.append(forged)
        forged = copy.deepcopy(contract)
        forged["offer_schemas"]["t10_offer"]["exact_keys"].append("forged")
        semantic.append(forged)
        forged = copy.deepcopy(contract)
        forged["offer_schemas"]["t10_offer"]["ranges"]["tag"] = [0, 99]
        semantic.append(forged)
        forged = copy.deepcopy(contract)
        forged["offer_schemas"]["other_writer_offer"]["enums"]["mode"].append(
            "FORGED")
        semantic.append(forged)
        forged = copy.deepcopy(contract)
        forged["canonical_configuration_frame"]["field_order"].reverse()
        semantic.append(forged)
        forged = copy.deepcopy(contract)
        forged["canonical_configuration_frame"]["field_bit_order"] = "FORGED"
        semantic.append(forged)
        forged = copy.deepcopy(contract)
        forged["canonical_configuration_frame"]["crc"][
            "reflected_recurrence_polynomial"] = "0xFORGED"
        semantic.append(forged)
        require(len(semantic) == 14, "semantic attack population drift")
        for index, payload in enumerate(semantic):
            expect_reject(module, root, payload, "old_semantic_{}.json".format(index))
        counts["r3_semantic_omissions"] = {"tested": 14, "rejected": 14}

        m31_spec = contract["independent_review_admissions"]["m31_r4"]
        canonical_m31 = read_json(resolve(m31_spec["path"]))
        for index in range(5):
            forged_contract = copy.deepcopy(contract)
            review = copy.deepcopy(canonical_m31)
            if index == 0:
                review["claim_boundary"]["forbidden"] = ""
            elif index == 1:
                review["log_audit"]["warning_count"] = 999
            elif index == 2:
                review["observed"]["conditional_t10_no_stall_accept_ii"] = 999
            elif index == 3:
                review["source_audit"]["dynamic_phase_indexed_t10_arrays"] = 999
            else:
                review["forged_headline_admitted"] = True
            review_path = write_contract(
                root, review, "old_m31_review_{}.json".format(index))
            forged_spec = forged_contract["independent_review_admissions"]["m31_r4"]
            forged_spec["path"] = str(review_path)
            forged_spec["sha256"] = sha256(review_path)
            expect_reject(module, root, forged_contract,
                          "old_m31_contract_{}.json".format(index))
        counts["r3_m31_review_omissions"] = {"tested": 5, "rejected": 5}

        bool_int = []
        forged = copy.deepcopy(contract)
        forged["frozen_architecture"]["intermediate_elastic_slots_target"] = True
        bool_int.append(forged)
        forged = copy.deepcopy(contract)
        forged["theory_rules"]["configuration_load_cycles_included"] = 0
        bool_int.append(forged)
        forged = copy.deepcopy(contract)
        forged["offer_schemas"]["t10_offer"]["ranges"]["tag"][0] = False
        bool_int.append(forged)
        forged = copy.deepcopy(contract)
        forged["reachable_state_model"]["reserved_domain"][0] = False
        bool_int.append(forged)
        for index, payload in enumerate(bool_int):
            expect_reject(module, root, payload, "r4_bool_int_{}.json".format(index))
        counts["r4_bool_integer_counterexamples"] = {"tested": 4, "rejected": 4}

        nested_bool_int = []
        for keys, value in (
                (("frozen_architecture", "ternary_codes", "0"), False),
                (("canonical_configuration_frame", "crc",
                  "extra_output_reflection_after_recurrence"), 0),
                (("theory_rules", "result_backpressure_included_in_theory"), 0),
                (("theory_rules", "energy_admitted"), 0)):
            forged = copy.deepcopy(contract)
            cursor = forged
            for key in keys[:-1]:
                cursor = cursor[key]
            cursor[keys[-1]] = value
            nested_bool_int.append(forged)
        for index, payload in enumerate(nested_bool_int):
            expect_reject(module, root, payload,
                          "nested_bool_int_{}.json".format(index))
        counts["additional_nested_bool_integer"] = {"tested": 4, "rejected": 4}

        float_int = []
        for keys in (
                ("frozen_architecture", "temporal_rows"),
                ("theory_rules", "conditional_t10_steady_throughput_limit",
                 "numerator"),
                ("canonical_configuration_frame", "fragment_ranges", "index", 0),
                ("reachable_state_model", "reserved_domain", 1)):
            forged = copy.deepcopy(contract)
            cursor = forged
            for key in keys[:-1]:
                cursor = cursor[key]
            cursor[keys[-1]] = float(cursor[keys[-1]])
            float_int.append(forged)
        for index, payload in enumerate(float_int):
            expect_reject(module, root, payload, "float_int_{}.json".format(index))
        counts["float_integer_equal_value"] = {"tested": 4, "rejected": 4}

        canonical_text = CONTRACT.read_text(encoding="utf-8")
        raw_cases = {
            "NaN": canonical_text.replace(
                '"intermediate_elastic_slots_target": 1',
                '"intermediate_elastic_slots_target": NaN', 1),
            "Infinity": canonical_text.replace(
                '"intermediate_elastic_slots_target": 1',
                '"intermediate_elastic_slots_target": Infinity', 1),
            "-Infinity": canonical_text.replace(
                '"intermediate_elastic_slots_target": 1',
                '"intermediate_elastic_slots_target": -Infinity', 1),
        }
        for name, raw in raw_cases.items():
            path = root / "nonstandard_{}.json".format(name.replace("-", "minus"))
            path.write_text(raw, encoding="utf-8")
            try:
                module.build(path)
            except Exception:
                continue
            raise ValueError("non-standard numeric attack passed: {}".format(name))
        counts["nonstandard_json_numeric_constants"] = {
            "tested": 3, "rejected": 3,
            "tokens": ["NaN", "Infinity", "-Infinity"]}

        duplicate_raw = []
        compact = json.dumps(contract, separators=(",", ":"))
        duplicate_raw.append('{"schema":"FORGED",' + compact[1:])
        duplicate_raw.append('{"claim_boundary":"FORGED",' + compact[1:])
        duplicate_raw.append(canonical_text.replace(
            '"temporal_rows": 10',
            '"temporal_rows": 10, "temporal_rows": 10', 1))
        duplicate_raw.append(canonical_text.replace(
            '"configuration_load_cycles_included": false',
            '"configuration_load_cycles_included": false, '
            '"configuration_load_cycles_included": false', 1))
        for index, raw in enumerate(duplicate_raw):
            path = root / "duplicate_contract_{}.json".format(index)
            path.write_text(raw, encoding="utf-8")
            try:
                module.build(path)
            except Exception:
                continue
            raise ValueError("duplicate contract attack passed: {}".format(index))
        for review_name in ("m31_r4", "m37_r8"):
            forged_contract = copy.deepcopy(contract)
            spec = forged_contract["independent_review_admissions"][review_name]
            source = resolve(spec["path"]).read_text(encoding="utf-8")
            forged_review = root / "duplicate_{}_review.json".format(review_name)
            forged_review.write_text(
                '{"schema":"FORGED",' + source.lstrip()[1:], encoding="utf-8")
            spec["path"] = str(forged_review)
            spec["sha256"] = sha256(forged_review)
            expect_reject(module, root, forged_contract,
                          "duplicate_{}_contract.json".format(review_name))
        counts["duplicate_json_keys"] = {"tested": 6, "rejected": 6}

        dependency_cases = []
        forged = copy.deepcopy(contract)
        forged["inputs"]["m31_vcs_receipt"]["sha256"] = "0" * 64
        dependency_cases.append(forged)
        forged = copy.deepcopy(contract)
        same_m31 = root / "m31_receipt_same_bytes.json"
        same_m31.write_bytes(resolve(
            contract["inputs"]["m31_vcs_receipt"]["path"]).read_bytes())
        forged["inputs"]["m31_vcs_receipt"]["path"] = str(same_m31)
        dependency_cases.append(forged)
        forged = copy.deepcopy(contract)
        forged["inputs"]["m31_review_validator"]["path"] = str(ANALYZER)
        forged["inputs"]["m31_review_validator"]["sha256"] = sha256(ANALYZER)
        dependency_cases.append(forged)
        forged = copy.deepcopy(contract)
        forged["inputs"]["m37_vcs_receipt"]["sha256"] = "f" * 64
        dependency_cases.append(forged)
        forged = copy.deepcopy(contract)
        same_m37 = root / "m37_rtl_same_bytes.sv"
        same_m37.write_bytes(resolve(
            contract["inputs"]["m37_r8_frozen_rtl"]["path"]).read_bytes())
        forged["inputs"]["m37_r8_frozen_rtl"]["path"] = str(same_m37)
        dependency_cases.append(forged)
        forged = copy.deepcopy(contract)
        forged["inputs"]["m37_review_validator"]["path"] = str(ANALYZER)
        forged["inputs"]["m37_review_validator"]["sha256"] = sha256(ANALYZER)
        dependency_cases.append(forged)
        for index, payload in enumerate(dependency_cases):
            expect_reject(module, root, payload,
                          "dependency_rebind_{}.json".format(index))
        counts["m31_m37_dependency_drift_or_rebind"] = {
            "tested": 6, "rejected": 6}

        frame = module.pack_configuration_frame(module.GOLDEN_CONFIG)
        fragments = module.make_fragments(frame)
        loader = module.StrictFragmentLoader()
        loader.accept(fragments[0], datapath_drained=True)
        try:
            loader.accept(fragments[1], datapath_drained=1)
        except Exception:
            pass
        else:
            raise ValueError("midframe boolean/integer drain attack passed")
        require(loader.failed is True and loader.next_index == 0
                and bytes(loader.shadow) == b"" and loader.active_config is None,
                "midframe failure did not clear shadow state")
        try:
            loader.accept(fragments[1], datapath_drained=True)
        except Exception:
            pass
        else:
            raise ValueError("midframe continuation bypassed fragment-zero restart")
        for fragment in fragments:
            activated = loader.accept(fragment, datapath_drained=True)
        require(activated and loader.active_config == module.GOLDEN_CONFIG,
                "midframe failure fragment-zero recovery failed")
        counts["midframe_shadow_reset_restart"] = {
            "tested": 1, "rejected": 1, "shadow_cleared": True,
            "next_index_reset": True, "fragment_zero_restart_required": True,
            "active_context_unchanged": True, "full_recovery": True}

    total = sum(item["tested"] for item in counts.values())
    rejected = sum(item["rejected"] for item in counts.values())
    require(total == 47 and rejected == 47, "hammer attack population drift")
    return {"total_tested": total, "total_rejected": rejected,
            "categories": counts}


def expected_review():
    verify_anchors()
    module = load_analyzer()
    regression = run_regression()
    repeat = run_repeat_build(module)
    attacks = run_attacks(module)
    result = read_json(RESULT)
    return {
        "schema": "m38_r5_independent_hammer_review_v1",
        "status": "GO_M38_R5_TYPE_STRICT_REFERENCE_MODEL_ONLY",
        "date": "2026-08-22",
        "review": {
            "independent_of_m38_r5_implementation": True,
            "score_0_to_100": 96, "p0": 0, "p1": 0, "p2": 2,
            "decision": "GO_EXECUTABLE_REFERENCE_MODEL_ONLY",
            "pass_admission_generated": True},
        "exact_anchors": EXPECTED_ANCHORS,
        "validator": [
            "hw_autoresearch_nts07/system_simulator/scripts/"
            "validate_m38_r5_independent_hammer_admission.py",
            sha256(Path(__file__).resolve())],
        "mandatory_rereview_passes": {
            "python": "Python {}".format(platform.python_version()),
            "upstream_tests": regression,
            "repeat_build": repeat,
            "adversarial_matrix": attacks,
            "unchanged_model_evidence": {
                "reachable_states": result["finite_reachable_state_audit"][
                    "reachable_states"],
                "transitions_checked": result["finite_reachable_state_audit"][
                    "transitions_checked"],
                "maximum_directed_drain_steps": result[
                    "finite_reachable_state_audit"]["maximum_directed_drain_steps"],
                "system_speedup_admitted": False}},
        "findings": {
            "p0": [], "p1": [],
            "p2": [
                {"id": "P2_DIRECTED_DRAIN_IS_NOT_GENERAL_HARDWARE_LIVENESS",
                 "disposition": "scope_guard_preserved_nonblocking"},
                {"id": "P2_ASYMPTOTIC_TWO_X_IS_CONDITIONAL_KERNEL_ONLY",
                 "disposition": "scope_guard_preserved_nonblocking"}]},
        "admitted": {
            "python36_executable_reference_model": True,
            "recursive_type_strict_semantic_binding": True,
            "duplicate_and_nonstandard_json_rejection": True,
            "m31_r4_m37_r8_hash_bound_vcs_only_recursive_identity": True,
            "integrated_rtl": False, "integrated_rtl_vcs": False,
            "dc_sta_formality": False, "ppa": False,
            "power_energy": False, "memory_timing": False,
            "trained_coverage": False, "local_motion_system_cycles": False,
            "system_speedup": False, "headline": False},
        "claim_boundary": (
            "GO is limited to the exact SHA-bound M38-r5 Python3.6 executable "
            "reference model and its recursive VCS-only M31-r4/M37-r8 identity. "
            "It is not integrated RTL, RTL VCS, DC/STA/Formality, PPA, power, "
            "energy, memory timing, trained coverage, Local/Motion system cycles, "
            "system speedup, accelerator comparison, DATE headline, or best-paper "
            "evidence."),
        "next_gate": (
            "A separately reviewed integrated-RTL milestone followed by fresh "
            "Synopsys VCS/DC/STA/Formality and system trace evidence is required."),
    }


def validate_review(path=REVIEW):
    actual = read_json(path)
    expected = expected_review()
    require(type_strict_equal(actual, expected),
            "M38-r5 independent review payload drift")
    require((Path(path).stat().st_mode & 0o777) == 0o444,
            "M38-r5 independent review must be read-only")
    return actual


def validate_admission(path=ADMISSION, review_path=REVIEW):
    review = validate_review(review_path)
    admission = read_json(path)
    require(set(admission) == {
        "schema", "status", "date", "review", "validator", "admitted",
        "forbidden", "claim_boundary", "review_required_for_scope_extension"},
        "M38-r5 admission top population drift")
    require(admission["schema"] ==
            "m38_r5_independent_model_only_admission_v1",
            "M38-r5 admission schema drift")
    require(admission["status"] ==
            "PASS_EXACT_M38_R5_PYTHON36_REFERENCE_MODEL_ONLY",
            "M38-r5 admission status drift")
    require(admission["review"] == [
        "hw_autoresearch_nts07/results/"
        "m38_rst_math_protocol_reachable_r5_20260822/"
        "m38_r5_independent_hammer_go_review.json", sha256(review_path)],
        "M38-r5 admission review identity drift")
    require(admission["validator"] == [
        "hw_autoresearch_nts07/system_simulator/scripts/"
        "validate_m38_r5_independent_hammer_admission.py",
        sha256(Path(__file__).resolve())], "M38-r5 admission validator drift")
    require(admission["admitted"] == {
        "exact_m38_r5_python36_reference_model": True,
        "exact_recursive_m31_r4_m37_r8_vcs_only_identity": True},
        "M38-r5 admitted scope drift")
    require(set(admission["forbidden"]) == set(FORBIDDEN_ADMISSION_KEYS)
            and all(admission["forbidden"][key] is False
                    for key in FORBIDDEN_ADMISSION_KEYS),
            "M38-r5 forbidden scope opened")
    require(admission["claim_boundary"] == review["claim_boundary"],
            "M38-r5 admission claim boundary drift")
    require(admission["review_required_for_scope_extension"] is True,
            "M38-r5 scope extension gate opened")
    require((Path(path).stat().st_mode & 0o777) == 0o444,
            "M38-r5 admission must be read-only")
    return admission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=REVIEW)
    parser.add_argument("--admission", type=Path, default=ADMISSION)
    parser.add_argument("--emit-review", action="store_true")
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()
    if args.emit_review:
        print(json.dumps(expected_review(), indent=2, sort_keys=True))
        return
    validate_review(args.review)
    if not args.review_only:
        validate_admission(args.admission, args.review)
    print("M38_R5_INDEPENDENT_MODEL_ONLY_ADMISSION_VALID=1")


if __name__ == "__main__":
    main()
