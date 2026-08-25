#!/usr/bin/env python3
"""Positive and adversarial regression for the M38-r5 type-strict model."""

import copy
import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
R4_TEST = HW_ROOT / (
    "system_simulator/tests/test_m38_rst_math_protocol_reachable_r4.py")
R5_SCRIPT = HW_ROOT / (
    "system_simulator/scripts/analyze_m38_rst_math_protocol_reachable_r5.py")
R5_CONTRACT = HW_ROOT / "contracts/m38_rst_math_input_contract_r5_20260822.json"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R4TEST = load_module(R4_TEST, "m38_r4_regression_base")
M38 = load_module(R5_SCRIPT, "m38_r5_under_test")

# The inherited tests resolve their module globals at execution time. Point
# those globals at r5 while keeping the frozen r4 regression source unchanged.
R4TEST.M38 = M38
R4TEST.CONTRACT = R5_CONTRACT


class M38R5Test(R4TEST.M38R4Test):
    """Run every r4 check and replace gates whose diagnostics changed in r5."""

    def test_status_and_both_independent_reviews_are_bound(self):
        self.assertEqual(
            self.result["status"],
            "PASS_M38_R5_TYPE_STRICT_MATH_PROTOCOL_COMPLETE_REACHABLE_STATE_ONLY")
        reviews = self.result["independent_review_admission_audit"]
        self.assertTrue(reviews["m31_r4"]["admitted"])
        self.assertTrue(reviews["m37_r8"]["admitted"])
        self.assertTrue(self.result["admission"][
            "recursive_type_strict_semantic_binding_admitted"])
        self.assertFalse(self.result["admission"]["system_speedup_admitted"])

    def test_contract_population_identity_claim_and_protocol_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = []
            top = copy.deepcopy(self.contract)
            top["extra"] = 1
            cases.append((top, "contract population drift"))
            identity = copy.deepcopy(self.contract)
            identity["identity"] += "_bad"
            cases.append((identity, "identity drift"))
            claim = copy.deepcopy(self.contract)
            claim["claim_boundary"] += " bad"
            cases.append((claim, "claim boundary drift"))
            fragment = copy.deepcopy(self.contract)
            fragment["canonical_configuration_frame"][
                "fragment_exact_keys"].append("bad")
            cases.append((fragment, "type-strict drift"))
            reachable = copy.deepcopy(self.contract)
            reachable["reachable_state_model"]["reserved_domain"] = [0, 16]
            cases.append((reachable, "type-strict drift"))
            for index, (payload, message) in enumerate(cases):
                path = root / "contract_{}.json".format(index)
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    M38.build(path)

            semantic_forgeries = []
            for key in (
                    "state_fields", "context_modes", "stage1_phase_domain",
                    "reconstruction_phase_domain", "reservation_relation",
                    "writer_rule", "overflow_rule", "liveness_scope"):
                payload = copy.deepcopy(self.contract)
                payload["reachable_state_model"][key] = "FORGED"
                semantic_forgeries.append(payload)
            payload = copy.deepcopy(self.contract)
            payload["offer_schemas"]["t10_offer"]["exact_keys"].append("forged")
            semantic_forgeries.append(payload)
            payload = copy.deepcopy(self.contract)
            payload["offer_schemas"]["t10_offer"]["ranges"]["tag"] = [0, 99]
            semantic_forgeries.append(payload)
            payload = copy.deepcopy(self.contract)
            payload["offer_schemas"]["other_writer_offer"]["enums"][
                "mode"].append("FORGED")
            semantic_forgeries.append(payload)
            payload = copy.deepcopy(self.contract)
            payload["canonical_configuration_frame"]["field_order"].reverse()
            semantic_forgeries.append(payload)
            payload = copy.deepcopy(self.contract)
            payload["canonical_configuration_frame"]["field_bit_order"] = "FORGED"
            semantic_forgeries.append(payload)
            payload = copy.deepcopy(self.contract)
            payload["canonical_configuration_frame"]["crc"][
                "reflected_recurrence_polynomial"] = "0xFORGED"
            semantic_forgeries.append(payload)
            self.assertEqual(len(semantic_forgeries), 14)
            original_sha = R4TEST.digest(R5_CONTRACT)
            for index, payload in enumerate(semantic_forgeries):
                path = root / "semantic_forgery_{}.json".format(index)
                path.write_text(json.dumps(payload), encoding="utf-8")
                self.assertNotEqual(R4TEST.digest(path), original_sha)
                with self.assertRaisesRegex(ValueError, "type-strict drift"):
                    M38.build(path)

            # These four were accepted by r4 because Python considers bool and
            # int equal. R5 must reject each at the recursive type gate.
            type_forgeries = []
            payload = copy.deepcopy(self.contract)
            payload["frozen_architecture"]["intermediate_elastic_slots_target"] = True
            type_forgeries.append(payload)
            payload = copy.deepcopy(self.contract)
            payload["theory_rules"]["configuration_load_cycles_included"] = 0
            type_forgeries.append(payload)
            payload = copy.deepcopy(self.contract)
            payload["offer_schemas"]["t10_offer"]["ranges"]["tag"][0] = False
            type_forgeries.append(payload)
            payload = copy.deepcopy(self.contract)
            payload["reachable_state_model"]["reserved_domain"][0] = False
            type_forgeries.append(payload)
            self.assertEqual(len(type_forgeries), 4)
            for index, payload in enumerate(type_forgeries):
                path = root / "bool_int_forgery_{}.json".format(index)
                path.write_text(json.dumps(payload), encoding="utf-8")
                self.assertNotEqual(R4TEST.digest(path), original_sha)
                with self.assertRaisesRegex(ValueError, "type-strict drift"):
                    M38.build(path)

            canonical = json.dumps(self.contract, separators=(",", ":"))
            for index, prefix in enumerate((
                    '{"schema":"FORGED",',
                    '{"claim_boundary":"FORGED",')):
                path = root / "duplicate_contract_{}.json".format(index)
                path.write_text(prefix + canonical[1:], encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                    M38.build(path)

            raw = R5_CONTRACT.read_text(encoding="utf-8").replace(
                '"intermediate_elastic_slots_target": 1',
                '"intermediate_elastic_slots_target": NaN', 1)
            nonstandard = root / "nonstandard_numeric.json"
            nonstandard.write_text(raw, encoding="utf-8")
            with self.assertRaisesRegex(
                    ValueError, "non-standard JSON numeric constant"):
                M38.build(nonstandard)

    def test_forged_review_admission_boundaries_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m31_spec = self.contract["independent_review_admissions"]["m31_r4"]
            original = M38.read_json_no_duplicates(M38.resolve(m31_spec["path"]))
            for index in range(5):
                contract = copy.deepcopy(self.contract)
                spec = contract["independent_review_admissions"]["m31_r4"]
                admission = copy.deepcopy(original)
                if index == 0:
                    admission["claim_boundary"]["forbidden"] = ""
                elif index == 1:
                    admission["log_audit"]["warning_count"] = 999
                elif index == 2:
                    admission["observed"][
                        "conditional_t10_no_stall_accept_ii"] = 999
                elif index == 3:
                    admission["source_audit"][
                        "dynamic_phase_indexed_t10_arrays"] = 999
                else:
                    admission["forged_headline_admitted"] = True
                forged = root / "forged_m31_review_{}.json".format(index)
                forged.write_text(json.dumps(admission), encoding="utf-8")
                spec["path"] = str(forged)
                spec["sha256"] = R4TEST.digest(forged)
                with self.assertRaisesRegex(
                        ValueError, "review payload type-strict drift"):
                    M38.build(self.write_contract(root, contract))

            contract = copy.deepcopy(self.contract)
            spec = contract["independent_review_admissions"]["m37_r8"]
            admission = M38.read_json_no_duplicates(M38.resolve(spec["path"]))
            admission["admitted"]["system"] = True
            forged = root / "forged_m37_review.json"
            forged.write_text(json.dumps(admission), encoding="utf-8")
            spec["path"] = str(forged)
            spec["sha256"] = R4TEST.digest(forged)
            with self.assertRaisesRegex(ValueError, "review payload type-strict drift"):
                M38.build(self.write_contract(root, contract))

            for review_name in ("m31_r4", "m37_r8"):
                contract = copy.deepcopy(self.contract)
                spec = contract["independent_review_admissions"][review_name]
                source = M38.resolve(spec["path"]).read_text(encoding="utf-8")
                forged = root / "duplicate_{}_review.json".format(review_name)
                forged.write_text(
                    '{"schema":"FORGED",' + source.lstrip()[1:],
                    encoding="utf-8")
                spec["path"] = str(forged)
                spec["sha256"] = R4TEST.digest(forged)
                with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                    M38.build(self.write_contract(root, contract))

    def test_claims_remain_model_only(self):
        super().test_claims_remain_model_only()
        admission = self.result["admission"]
        for key in (
                "recursive_type_strict_semantic_binding_admitted",
                "boolean_integer_interchange_rejected",
                "nonstandard_json_numeric_constants_rejected",
                "both_review_payloads_type_strict_canonical_match",
                "m31_independent_admission_type_strict_rebuild_match"):
            self.assertTrue(admission[key])


if __name__ == "__main__":
    R4TEST.unittest.main()
