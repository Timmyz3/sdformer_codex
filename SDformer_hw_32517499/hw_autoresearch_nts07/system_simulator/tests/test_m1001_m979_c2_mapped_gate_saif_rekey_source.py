#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
CHECKER = HERE.parent / "scripts/check_m1001_m979_c2_mapped_gate_saif_rekey_source.py"
SPEC = importlib.util.spec_from_file_location("m1001_checker", CHECKER)
M = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def fake_saif(duration):
    return f'''(SAIFILE (DURATION {duration})
      (INSTANCE tb_m979_c2_three_axis_mapped_gate_case_saif (INSTANCE dut
       (PORT (clk_core (T0 1) (T1 1) (TX 0) (TC 20))
        (rst_core (T0 1) (T1 0) (TX 0) (TC 0))
        (header_accept (T0 1) (T1 1) (TX 0) (TC 2))
        (raw_accept (T0 1) (T1 1) (TX 0) (TC 3))
        (mem_req_accept[0] (T0 1) (T1 1) (TX 0) (TC 4))
        (mem_rsp_accept[0] (T0 1) (T1 1) (TX 0) (TC 4))
        (result_accumulator[0] (T0 1) (T1 1) (TX 0) (TC 6))
        (result_accept (T0 1) (T1 1) (TX 0) (TC 4))
        (token_done_accept (T0 1) (T1 1) (TX 0) (TC 2))))))'''


class M1001Test(unittest.TestCase):
    def test_static_rekey(self):
        value = M.validate_static()
        self.assertEqual(value["status"], "PASS_M1001_STATIC_REKEY__NO_EDA")
        self.assertFalse(value["m979_semantics_modified"])

    def test_all_frozen_m979_hashes(self):
        for path, expected in M.FROZEN.items():
            self.assertEqual(M.sha(path), expected)

    def test_new_chain_has_no_conflicting_m993(self):
        text = M.RUNNER.read_text()
        for token in ("M1002", "M1003", "M1004", "M1005"):
            self.assertIn(token, text)
        self.assertNotIn("M993_", text)

    def test_canonical_attempt_and_result_are_rekeyed(self):
        data = __import__("json").loads(M.CONTRACT.read_text())
        self.assertIn("m1005_m1001", data["canonical"]["result"])
        self.assertIn(".m1005_m1001", data["canonical"]["attempt"])

    def test_frozen_saif_validator_passes_same_anchor(self):
        with tempfile.NamedTemporaryFile("w", suffix=".saif", delete=False) as f:
            f.write(fake_saif(153)); path = Path(f.name)
        try:
            value = M.M979.validate_saif(path, "k8", 0, 51)
            self.assertEqual(value["status"], "PASS_M979_PER_CASE_MAPPED_GATE_SAIF")
        finally:
            path.unlink()

    def test_frozen_cycle_anchor_still_fails_closed(self):
        with tempfile.NamedTemporaryFile("w", suffix=".saif", delete=False) as f:
            f.write(fake_saif(156)); path = Path(f.name)
        try:
            with self.assertRaisesRegex(RuntimeError, "cycle anchor"):
                M.M979.validate_saif(path, "k8", 0, 52)
        finally:
            path.unlink()

    def test_no_execution_claim(self):
        data = __import__("json").loads(M.CONTRACT.read_text())
        self.assertFalse(data["claim_boundary"]["vcs_executed"])
        self.assertFalse(data["claim_boundary"]["saif_created"])


if __name__ == "__main__":
    unittest.main()
