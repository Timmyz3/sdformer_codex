#!/usr/bin/env python3
"""Mutation tests for the zero-execution M1768 source checker."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest


HW = Path(__file__).resolve().parents[2]
CHECKER = HW / "system_simulator/scripts/check_m1768_m1753_c2_python312_wrapper_source.py"
SPEC = importlib.util.spec_from_file_location("m1768_check", str(CHECKER))
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class M1768SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = CHECK.WRAPPER.read_text()
        cls.contract = json.loads(CHECK.CONTRACT.read_text())

    def assertRejected(self, function, value):
        with self.assertRaises(RuntimeError):
            function(value)

    def test_01_live_source_text(self):
        CHECK.validate_source_text(self.text)

    def test_02_wrong_shebang_rejected(self):
        self.assertRejected(CHECK.validate_source_text,
                            self.text.replace("#!/usr/bin/python3.12", "#!/usr/bin/python3"))

    def test_03_second_execve_rejected(self):
        self.assertRejected(CHECK.validate_source_text,
                            self.text + "\nos.execve('/bin/false', [], {})\n")

    def test_04_borrowed_interpreter_rejected(self):
        self.assertRejected(CHECK.validate_source_text,
                            self.text.replace("/usr/bin/python3.12", "/opt/anaconda3/bin/python3.12"))

    def test_05_live_contract(self):
        CHECK.validate_contract(self.contract)

    def test_06_interpreter_drift_rejected(self):
        value = copy.deepcopy(self.contract)
        value["interpreter_identity"]["version"] = "3.12.12"
        self.assertRejected(CHECK.validate_contract, value)

    def test_07_failure_authority_drift_rejected(self):
        value = copy.deepcopy(self.contract)
        value["bound_authority"]["m1767_receipt_sha256"] = "0" * 64
        self.assertRejected(CHECK.validate_contract, value)

    def test_08_budget_drift_rejected(self):
        value = copy.deepcopy(self.contract)
        value["future_budget"]["m1768_wrapper_attempts"] = 2
        self.assertRejected(CHECK.validate_contract, value)

    def test_09_claim_promotion_rejected(self):
        value = copy.deepcopy(self.contract)
        value["claim_boundary"]["mapped_vcs"] = True
        self.assertRejected(CHECK.validate_contract, value)


if __name__ == "__main__":
    unittest.main()
