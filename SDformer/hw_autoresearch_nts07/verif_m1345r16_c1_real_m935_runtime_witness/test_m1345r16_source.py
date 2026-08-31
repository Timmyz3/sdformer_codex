#!/opt/anaconda3/envs/pytorch310/bin/python3.10
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
HW = HERE.parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("import spec failed: " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load("m1345r16_check", HERE / "check_m1345r16_source.py")
R15_TEST = load("m1345_bound_r15_tests",
                HW / "verif_m1337r15_c1_real_m935_runtime_witness/test_m1337r15_source.py")


class InheritedR15Tests(R15_TEST.Tests):
    """The complete 20-test R15 suite remains live in R16."""


def stage_mutation(source: str, stage: str, next_stage: str,
                   old: str, new: str = "") -> str:
    begin = source.index(stage + ": begin")
    end = source.index(next_stage + ": begin", begin)
    body = source[begin:end]
    if body.count(old) != 1:
        raise AssertionError("stage mutation anchor drift: " + stage)
    return source[:begin] + body.replace(old, new, 1) + source[end:]


class R16ClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = M.WITNESS.read_text()

    def reject(self, mutant: str) -> None:
        self.assertNotEqual(mutant, self.source)
        with self.assertRaises(AssertionError):
            M.check_witness_text(mutant)

    def test_01_first_accept_second_request_guard(self):
        self.reject(stage_mutation(
            self.source, "W_FIRST_REQUEST", "W_FIRST_ACCEPT",
            "                                    && (weight_request_fire === 1'b0)\n"))

    def test_02_second_accept_commit_guard(self):
        self.reject(stage_mutation(
            self.source, "W_SECOND_REQUEST", "W_SECOND_ACCEPT",
            "                                    && (psum_commit_fire === 1'b0)\n"))

    def test_03_commit_row_guard(self):
        self.reject(stage_mutation(
            self.source, "W_SECOND_ACCEPT", "W_PSUM_COMMIT",
            "                                    && (row_complete_fire === 1'b0)\n"))

    def test_04_row_task_guard(self):
        self.reject(stage_mutation(
            self.source, "W_PSUM_COMMIT", "W_ROW_DONE",
            "                                    && (task_done_fire === 1'b0)\n"))

    def mutate_control(self, control: str) -> str:
        begin = self.source.index("control_unknown = $isunknown({")
        end = self.source.index("});", begin) + 3
        block = self.source[begin:end]
        self.assertEqual(block.count(control), 1)
        return self.source[:begin] + block.replace(control, "1'b0", 1) + self.source[end:]

    def test_05_unknown_weight_control(self):
        self.reject(self.mutate_control("weight_request_fire"))

    def test_06_unknown_psum_control(self):
        self.reject(self.mutate_control("psum_request_fire"))

    def test_07_unknown_response_control(self):
        self.reject(self.mutate_control("response_accept"))

    def test_08_unknown_core_control(self):
        self.reject(self.mutate_control("core_accept"))

    def test_09_unknown_commit_control(self):
        self.reject(self.mutate_control("psum_commit_fire"))

    def test_10_unknown_row_control(self):
        self.reject(self.mutate_control("row_complete_fire"))

    def test_11_unknown_task_control(self):
        self.reject(self.mutate_control("task_done_fire"))

    def oracle_delete(self, term: str) -> str:
        self.assertEqual(self.source.count(term), 1)
        return self.source.replace(term, "", 1)

    def test_12_design_issue_oracle_conjunct(self):
        self.reject(self.oracle_delete(
            "            && (design_issue_accepts === 64'd2)\n"))

    def test_13_design_commit_oracle_conjunct(self):
        self.reject(self.oracle_delete(
            "            && (design_psum_commits === 64'd1)\n"))

    def test_14_design_row_oracle_conjunct(self):
        self.reject(self.oracle_delete(
            "            && (design_row_completions === 64'd1)\n"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
