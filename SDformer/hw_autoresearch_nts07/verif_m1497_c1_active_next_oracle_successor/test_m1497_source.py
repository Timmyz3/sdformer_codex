#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Author-only static and small-model tests; never invokes VCS/simv/EDA."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
CHECKER = HERE / "check_m1497_source.py"
RUNNER = HERE.parent / "dc_handoff/scripts/run_m1497_m1459_c1_active_next_oracle_clean_result_successor_one_shot.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load " + str(path))
    module = importlib.util.module_from_spec(spec)
    saved = list(sys.argv)
    try:
        sys.argv = [str(path)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


C = load("m1497_checker", CHECKER)
R = load("m1497_runner", RUNNER)


def model(**changes):
    values = dict(active=1, issue_valid=1, public_first=0,
                  public_source=1, latched_first=0, latched_source=1,
                  weight_accepted=0, psum_accepted=1, served_source=0)
    values.update(changes)
    return C.active_next_oracle(**values)


class M1497SourceTests(unittest.TestCase):
    def test_01_first_to_nonfirst(self):
        self.assertTrue(model())

    def test_02_first_to_first(self):
        self.assertTrue(model(public_first=1, latched_first=1,
                              psum_accepted=0))

    def test_03_idle_retired(self):
        self.assertTrue(model(active=0, issue_valid=None,
                              public_first=None, public_source=None,
                              latched_first=None, latched_source=None,
                              weight_accepted=None, psum_accepted=None,
                              served_source=None))

    def test_04_each_unknown_fails_closed(self):
        for field in ("active", "issue_valid", "public_first",
                      "public_source", "latched_first", "latched_source",
                      "weight_accepted", "psum_accepted", "served_source"):
            with self.subTest(field=field):
                self.assertFalse(model(**{field: None}))

    def test_05_wrong_weight_accepted_fails(self):
        self.assertFalse(model(weight_accepted=1))

    def test_06_wrong_nonfirst_psum_accepted_fails(self):
        self.assertFalse(model(psum_accepted=0))

    def test_07_wrong_first_psum_accepted_fails(self):
        self.assertFalse(model(public_first=1, latched_first=1,
                               psum_accepted=1))

    def test_08_latched_public_first_mismatch_fails(self):
        self.assertFalse(model(public_first=1))

    def test_09_latched_public_source_mismatch_fails(self):
        self.assertFalse(model(public_source=2))

    def test_10_stale_source_fails(self):
        self.assertFalse(model(served_source=1))

    def test_11_exact_single_tb_delta(self):
        old = C.TB_R13.read_text()
        self.assertEqual(C.TB.read_text(), old.replace(C.OLD, C.NEW))

    def test_12_frozen_r13(self):
        self.assertEqual(C.sha(C.TB_R13), C.R13_SHA)

    def test_13_clean_result_membership(self):
        self.assertEqual(R.CLEAN_PAYLOAD, {
            "compile.log", "sim.log",
            "m1497_c1_active_next_oracle_unit_delay_vcs_receipt_r1.json",
            "m1497_c1_active_next_oracle_identity_r1.json",
        })

    def test_14_symlink_still_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "result"
            root.mkdir()
            target = root / "compile.log"
            target.write_text("regular\n")
            (root / "alias").symlink_to(target)
            with self.assertRaises(RuntimeError):
                R.P.seal_dir_generic(root)

    def test_15_source_gate(self):
        result = C.check_source(require_runtime_authority=False)
        self.assertEqual(result["status"],
                         "PASS_M1497_C1_ACTIVE_NEXT_ORACLE_CLEAN_RESULT_SOURCE__NO_VCS_NO_EDA")

    def test_16_no_c2_namespace_collision(self):
        text = RUNNER.read_text() + C.TB.read_text() + C.FILELIST.read_text()
        self.assertNotIn("m1494", text.lower())


if __name__ == "__main__":
    unittest.main()
