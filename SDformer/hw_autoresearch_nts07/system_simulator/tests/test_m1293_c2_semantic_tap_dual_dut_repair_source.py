from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = (Path(__file__).resolve().parents[1] / "scripts" /
    "check_m1293_c2_semantic_tap_dual_dut_repair_source.py")
SPEC = importlib.util.spec_from_file_location("m1293_checker", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M1293SourceRepairTests(unittest.TestCase):
    def test_01_baseline_source_and_closed_contract(self):
        result = M.run_checks()
        self.assertEqual(result["status"],
            "PASS_M1293_SOURCE_REPAIR__NO_EXECUTION_AUTHORIZED")
        self.assertEqual(result["real_tool_calls"], 0)
        self.assertEqual(result["tap_exact_rhs"]["tap_count"], 13)
        self.assertEqual(result["dual_dut_tb"]["transaction_classes"], 3)

    def test_02_endpoint_normalized_block_ignores_comments(self):
        text = M.ENDPOINT.read_text(encoding="utf-8")
        changed = text.replace("always_comb begin : valid_qualified_guard",
            "/* harmless formatting/comment */\nalways_comb begin : valid_qualified_guard", 1)
        self.assertEqual(M.check_endpoint_text(changed)["guard_token_sha256"],
                         M.ENDPOINT_GUARD_TOKEN_SHA)

    def test_03_unconditional_valid_gate_attack_is_rejected(self):
        text = M.ENDPOINT.read_text(encoding="utf-8")
        attacked = text.replace("if (mem_req_valid === 1'b1) begin",
            "if (1'b1) begin // mem_req_valid === 1'b1", 1)
        with self.assertRaisesRegex(M.Failure, "normalized guard"):
            M.check_endpoint_text(attacked)

    def test_04_unconditional_payload_gate_attack_is_rejected(self):
        text = M.ENDPOINT.read_text(encoding="utf-8")
        attacked = text.replace("if (request_payload_known) begin",
            "if (1'b1) begin // request_payload_known", 1)
        with self.assertRaisesRegex(M.Failure, "normalized guard"):
            M.check_endpoint_text(attacked)

    def test_05_endpoint_unreached_pass_attack_is_rejected(self):
        text = M.TB.read_text(encoding="utf-8")
        attacked = text.replace("request_count_original <= 0",
                                "request_count_original < 0", 1)
        with self.assertRaisesRegex(M.Failure, "atomic reachability"):
            M.check_tb_text(attacked)

    def test_06_transaction_class_compare_mutation_is_rejected(self):
        text = M.TB.read_text(encoding="utf-8")
        attacked = text.replace("req_accept_original !== req_accept_qualified",
                                "req_accept_original === req_accept_qualified", 1)
        with self.assertRaisesRegex(M.Failure, "transaction-class compare"):
            M.check_tb_text(attacked)

    def test_07_exact_tap_rhs_and_x_coercion_attack(self):
        text = M.TOP.read_text(encoding="utf-8")
        result = M.check_tap_exact_rhs_text(text)
        self.assertEqual(result["direct_exact_rhs"], 8)
        self.assertEqual(result["hierarchical_exact_rhs"], 5)
        attacked = text.replace(
            "assign tap_core_protocol_error = core_protocol_error;",
            "assign tap_core_protocol_error = $isunknown(core_protocol_error) ? 1'b0 : core_protocol_error;",
            1)
        with self.assertRaisesRegex(M.Failure, "exact RHS|occurrence"):
            M.check_tap_exact_rhs_text(attacked)

    def test_08_contract_rejects_all_claim_promotions(self):
        base = M.strict_json(M.CONTRACT)
        attacks = (
            "k8_present", "equal_bandwidth_k1x8_present",
            "single_k1_power_admitted", "fair_energy_comparison_admitted",
            "performance_admitted", "mapped_functionality",
            "system_speedup", "paper_ppa_ready", "paper_headline",
        )
        for key in attacks:
            value = copy.deepcopy(base)
            value["claim_boundary"][key] = True
            with self.subTest(key=key):
                with self.assertRaisesRegex(M.Failure, "exact bool"):
                    M.check_contract_data(value, validate_source_hashes=False)

    def test_09_contract_rejects_open_world_keys_and_bool_int(self):
        base = M.strict_json(M.CONTRACT)
        added = copy.deepcopy(base)
        added["claim_boundary"]["future_power_escape"] = False
        with self.assertRaisesRegex(M.Failure, "keyset"):
            M.check_contract_data(added, validate_source_hashes=False)
        top = copy.deepcopy(base); top["K8"] = False
        with self.assertRaisesRegex(M.Failure, "top keyset"):
            M.check_contract_data(top, validate_source_hashes=False)
        int_zero = copy.deepcopy(base)
        int_zero["claim_boundary"]["system_speedup"] = 0
        with self.assertRaisesRegex(M.Failure, "exact bool"):
            M.check_contract_data(int_zero, validate_source_hashes=False)

    def test_10_duplicate_json_and_source_row_key_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m1293_test.") as name:
            duplicate = Path(name) / "duplicate.json"
            duplicate.write_text('{"schema":1,"schema":2}', encoding="utf-8")
            with self.assertRaisesRegex(M.Failure, "duplicate"):
                M.strict_json(duplicate)
        base = M.strict_json(M.CONTRACT)
        base["sources"][0]["claim"] = "performance"
        with self.assertRaisesRegex(M.Failure, "source row schema"):
            M.check_contract_data(base, validate_source_hashes=False)

    def test_11_filelist_and_frozen_m1279_identity(self):
        self.assertEqual(M.check_filelist()["members"], 11)
        for path in M.FROZEN_RTL:
            self.assertTrue((M.HW / path).is_file())
        for path, digest in M.UPSTREAM.items():
            self.assertEqual(M.sha256(M.HW / path), digest)

    def test_12_pass_tokens_are_inside_guarded_atomic_block_only(self):
        result = M.check_tb_text(M.TB.read_text(encoding="utf-8"))
        self.assertFalse(result["endpoint_can_be_unreached_and_pass"])
        self.assertTrue(result["request_reachability_required"])
        self.assertTrue(result["result_reachability_required"])
        self.assertTrue(result["token_done_reachability_required"])


if __name__ == "__main__":
    unittest.main()
