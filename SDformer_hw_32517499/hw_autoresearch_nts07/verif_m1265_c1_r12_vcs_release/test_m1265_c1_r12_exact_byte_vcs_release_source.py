#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Exact-byte release mutations only; never invokes VCS/simv/EDA."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
CHECKER = HERE / "static_check_m1265_c1_r12_exact_byte_vcs_release_source.py"
SPEC = importlib.util.spec_from_file_location("m1265_exact_byte", CHECKER)
M = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = M; SPEC.loader.exec_module(M)


class Tests(unittest.TestCase):
    def test_01_canonical_exact_bytes(self):
        self.assertTrue(M.exact_byte_gate(M.RUNNER, M.RUNNER.read_text()))
        self.assertTrue(M.exact_byte_gate(M.FILELIST, M.FILELIST.read_text()))
        self.assertTrue(M.exact_byte_gate(M.TB, M.TB.read_text()))

    def test_02_any_runner_byte_rejected(self):
        source = M.RUNNER.read_text()
        for mutant in (source+"\n", source.replace("one attempt", "two attempts",1),
                       source.replace("./simv -no_save", "./simv",1)):
            self.assertFalse(M.exact_byte_gate(M.RUNNER, mutant))

    def test_03_any_tb_byte_rejected(self):
        source = M.TB.read_text()
        for mutant in (source+"\n", source.replace("test_index < 24", "test_index < 0",1),
                       source.replace("normal_m935_completion();", "if (1'b0) normal_m935_completion();",1)):
            self.assertFalse(M.exact_byte_gate(M.TB, mutant))

    def test_04_any_filelist_byte_rejected(self):
        source = M.FILELIST.read_text()
        self.assertFalse(M.exact_byte_gate(M.FILELIST, source+"\n"))
        self.assertFalse(M.exact_byte_gate(M.FILELIST, source.replace("m1258r12", "m1232r11",1)))

    def test_05_all_corpus_identity_mutations_rejected(self):
        for path in (M.PARENT,M.M935,M.WRAPPER,M.SVA,M.FOUNDRY,M.VCS,M.PYTHON,M.DOCS359):
            self.assertFalse(M.exact_byte_gate(path, path.read_text(errors="replace")+"X"))

    def test_06_external_release_env_fail_closed(self):
        names=("M1265_EXPECTED_RELEASE_SHA256","M1265_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256",
               "M1265_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256","M1265_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256")
        good={name:"a"*64 for name in names}; self.assertTrue(M.env_gate(good))
        for name in names:
            bad=dict(good); bad.pop(name); self.assertFalse(M.env_gate(bad))
            bad=dict(good); bad[name]="B"*64; self.assertFalse(M.env_gate(bad))


if __name__ == "__main__": unittest.main(verbosity=2)
