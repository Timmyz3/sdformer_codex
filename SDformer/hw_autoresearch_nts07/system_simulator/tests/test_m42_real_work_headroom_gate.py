from __future__ import print_function

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCRIPT = os.path.join(ROOT, "hw_autoresearch_nts07", "system_simulator", "scripts",
                      "analyze_m42_real_work_headroom_gate.py")
CONTRACT = os.path.join(ROOT, "hw_autoresearch_nts07", "contracts",
                        "m42_real_work_headroom_gate_contract_r1_20260823.json")


def strict_load(path):
    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(value)

    with open(path, "r") as handle:
        return json.load(handle, object_pairs_hook=pairs_hook, parse_constant=reject)


class M42RealWorkHeadroomGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("m42_analyzer", SCRIPT)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        cls.contract = strict_load(CONTRACT)
        cls.result = cls.module.build_result(cls.contract)

    def test_exact_diagnostic_envelopes(self):
        diagnostic = self.result["non_executable_diagnostic_envelopes"]
        self.assertEqual(diagnostic["local_uncoalesced_mean_total"],
                         {"numerator": 1327866918, "denominator": 5})
        self.assertEqual(diagnostic["local_uncoalesced_mean_compute_speedup"],
                         {"numerator": 1034780405, "denominator": 442622306})
        self.assertEqual(diagnostic["local_uncoalesced_p95_total"],
                         {"numerator": 266456878, "denominator": 1})
        self.assertEqual(diagnostic["local_uncoalesced_p95_compute_speedup"],
                         {"numerator": 620868243, "denominator": 266456878})

    def test_2p7_and_3p0_required_issue_widths_are_exact(self):
        gates = self.result["target_gates"]
        self.assertEqual(gates[1]["target_compute_speedup"],
                         {"numerator": 27, "denominator": 10})
        self.assertEqual(
            gates[1]["required_effective_source_issue_width_from_local_mean"],
            {"numerator": 833764248, "denominator": 433014695})
        self.assertEqual(gates[2]["target_compute_speedup"],
                         {"numerator": 3, "denominator": 1})
        self.assertEqual(
            gates[2]["required_effective_source_issue_width_from_local_mean"],
            {"numerator": 370561888, "denominator": 77475375})
        self.assertEqual(gates[2]["issue_width_peak"], 8)
        self.assertGreater(
            gates[2]["peak_issue_width_margin_from_local_mean"]["numerator"],
            gates[2]["peak_issue_width_margin_from_local_mean"]["denominator"])

    def test_motion_is_strictly_worse_and_not_admitted(self):
        real = self.result["independently_reviewed_real_work"]
        self.assertTrue(real["pure_motion_is_worse_on_this_cohort"])
        self.assertEqual(real["motion_over_local_mean"],
                         {"numerator": 34348361, "denominator": 23160118})
        self.assertFalse(self.result["admission"]["system_speedup_admitted"])
        for gate in self.result["target_gates"]:
            self.assertFalse(gate["real_executable_schedule_admitted"])
            self.assertFalse(gate["target_crossing_admitted"])

    def test_identity_sha_mutation_rejected(self):
        forged = json.loads(json.dumps(self.contract))
        forged["identity"]["m40_result"]["sha256"] = "0" * 64
        with self.assertRaises(self.module.AuditError):
            self.module.build_result(forged)

    def test_bool_as_integer_rejected(self):
        forged = json.loads(json.dumps(self.contract))
        forged["frozen_model"]["fixed_compute_reference_cycles"] = True
        with self.assertRaises(self.module.AuditError):
            self.module.build_result(forged)

    def test_nonstandard_and_duplicate_json_rejected(self):
        with tempfile.TemporaryDirectory() as tempdir:
            duplicate = os.path.join(tempdir, "duplicate.json")
            with open(duplicate, "w") as handle:
                handle.write('{"schema":"a","schema":"b"}')
            with self.assertRaises(self.module.AuditError):
                self.module.load_json(duplicate)
            nan_file = os.path.join(tempdir, "nan.json")
            with open(nan_file, "w") as handle:
                handle.write('{"value":NaN}')
            with self.assertRaises(self.module.AuditError):
                self.module.load_json(nan_file)

    def test_cli_refuses_output_overwrite(self):
        with tempfile.TemporaryDirectory() as tempdir:
            output = os.path.join(tempdir, "result.json")
            first = subprocess.run(["/usr/bin/python3.6", SCRIPT, "--output", output],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(first.returncode, 0)
            with open(output, "rb") as handle:
                before = hashlib.sha256(handle.read()).hexdigest()
            second = subprocess.run(["/usr/bin/python3.6", SCRIPT, "--output", output],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(second.returncode, 2)
            with open(output, "rb") as handle:
                after = hashlib.sha256(handle.read()).hexdigest()
            self.assertEqual(before, after)

    def test_result_rebuild_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as tempdir:
            first = os.path.join(tempdir, "first.json")
            second = os.path.join(tempdir, "second.json")
            for output in [first, second]:
                subprocess.check_call(["/usr/bin/python3.6", SCRIPT,
                                       "--output", output])
            with open(first, "rb") as left, open(second, "rb") as right:
                self.assertEqual(left.read(), right.read())


if __name__ == "__main__":
    unittest.main()
