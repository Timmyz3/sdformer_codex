#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Source-only mutations for M1354; never invokes VCS, simv, or EDA."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
CHECKER = HERE / "check_m1354_c1_r16_vcs_release_source.py"
SPEC = importlib.util.spec_from_file_location("m1354_source_gate", CHECKER)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class Tests(unittest.TestCase):
    def test_01_canonical_exact_bytes(self):
        for path in M.EXPECTED:
            self.assertTrue(M.exact_byte_gate(path, path.read_bytes()), path)

    def test_02_runner_mutations_rejected(self):
        source = M.RUNNER.read_bytes()
        for mutant in (source + b"\n", source.replace(b"at most one", b"at most two", 1),
                       source.replace(b"./simv -no_save", b"./simv", 1),
                       source.replace(b"M1354_EXPECTED_RELEASE_SHA256",
                                      b"M1354_RELEASE_SHA256", 1)):
            self.assertFalse(M.exact_byte_gate(M.RUNNER, mutant))

    def test_03_witness_and_tb_mutations_rejected(self):
        for path in (M.WITNESS, M.TB):
            source = path.read_bytes()
            self.assertFalse(M.exact_byte_gate(path, source + b"\n"))
            self.assertFalse(M.exact_byte_gate(path,
                source.replace(b"functional_vcs=false", b"functional_vcs=true", 1)))

    def test_04_filelist_mutations_rejected(self):
        source = M.FILELIST.read_bytes()
        self.assertFalse(M.exact_byte_gate(M.FILELIST, source + b"\n"))
        self.assertFalse(M.exact_byte_gate(M.FILELIST,
            source.replace(b"m1337r15", b"m1337r14", 1)))

    def test_05_r16_authority_mutations_rejected(self):
        for path in (M.R16_CHECKER, M.R16_TESTS, M.R16_CONTRACT):
            self.assertFalse(M.exact_byte_gate(path, path.read_bytes() + b"X"))

    def test_06_corpus_mutations_rejected(self):
        for path in (M.PARENT, M.M935, M.WRAPPER, M.SVA, M.FOUNDRY,
                     M.VCS, M.PYTHON, M.DOCS359):
            self.assertFalse(M.exact_byte_gate(path, path.read_bytes() + b"X"))

    def test_07_external_release_env_fail_closed(self):
        names = (
            "M1354_EXPECTED_RELEASE_SHA256",
            "M1354_EXPECTED_HAMMER_REVIEW_SHA256",
            "M1354_EXPECTED_HAMMER_MANIFEST_SHA256",
            "M1354_EXPECTED_HAMMER_OUTER_SEAL_FILE_SHA256",
        )
        good = {name: "a" * 64 for name in names}
        self.assertTrue(M.env_gate(good))
        for name in names:
            bad = dict(good); bad.pop(name)
            self.assertFalse(M.env_gate(bad))
            bad = dict(good); bad[name] = "B" * 64
            self.assertFalse(M.env_gate(bad))

    def test_08_contract_claim_injection_rejected(self):
        contract = __import__("json").loads(M.CONTRACT.read_text())
        for key in ("functional_vcs", "timing_verified", "cycles_measured",
                    "speedup", "ppa", "power", "energy", "system_speedup",
                    "headline"):
            mutant = __import__("copy").deepcopy(contract)
            mutant["claim_boundary"][key] = True
            with self.assertRaises(AssertionError):
                M.check_contract_dict(mutant)

    def test_09_future_release_injection_rejected(self):
        contract = __import__("json").loads(M.CONTRACT.read_text())
        for key, value in (("launch_authorized", True), ("vcs_compiles_now", 1),
                           ("simv_runs_now", 1), ("automatic_retry", True)):
            mutant = __import__("copy").deepcopy(contract)
            mutant["future_release"][key] = value
            with self.assertRaises(AssertionError):
                M.check_contract_dict(mutant)


if __name__ == "__main__":
    unittest.main(verbosity=2)
