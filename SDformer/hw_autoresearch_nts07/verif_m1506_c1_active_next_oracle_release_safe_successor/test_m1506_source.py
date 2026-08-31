#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Author-only static/mocked tests for M1506; never invokes EDA."""
from __future__ import annotations

import copy
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
CHECKER = HERE / "check_m1506_source.py"
RUNNER = HERE.parent / "dc_handoff/scripts/run_m1506_m1497_c1_active_next_oracle_release_safe_successor_one_shot.py"


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


C = load("m1506_checker_test", CHECKER)
R = load("m1506_runner_test", RUNNER)


def positive_log() -> str:
    lines = [
        R.R13_ENTER,
        R.R13_COMPLETE,
        R.WITNESS_OPERANDS,
        "COVERAGE_M1270R13_REAL_M935 first_beats=1 nonfirst_beats=1 "
        "join_hold_cycles=2 issue_accepts=2 psum_reads=1 row_completions=1 "
        "task_completions=1 response_cycle_gap=2 oracle_records=80 "
        "parent_issue_override=0 child_issue_override=0",
        '"sva.sv", 133: tb.u_protocol_sva.cp_nonfirst, 80 attempts, 1 match',
        '"sva.sv", 142: tb.u_protocol_sva.cp_ii2, 80 attempts, 1 match',
        R.BASE.R13_PASS,
        R.BASE.R15_PASS,
    ]
    lines.extend("ORACLE_M1270R13 site=x pass=1 index=%d" % index
                 for index in range(80))
    return "\n".join(lines) + "\n"


def walk_dicts(value, path=()):
    if isinstance(value, dict):
        yield path, value
        for key, item in value.items():
            yield from walk_dicts(item, path + (key,))


def walk_leaves(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from walk_leaves(item, path + (key,))
    else:
        yield path, value


def parent_at(value, path):
    for key in path:
        value = value[key]
    return value


def mutated_leaf(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return value + "__MUTATION"
    if isinstance(value, list):
        return value + ["__MUTATION"]
    raise AssertionError("unsupported contract leaf " + type(value).__name__)


class M1506SourceTests(unittest.TestCase):
    def test_01_source_gate(self):
        result = C.check_source(False)
        self.assertEqual(result["status"], C.AUTHOR_STATUS)

    def test_02_m1497_tb_oracle_preserved(self):
        self.assertEqual(C.sha(C.M1497_TB), C.M1497_PINS["testbench_sha256"])
        old = C.R13.read_text()
        m1497 = load("m1506_bound_m1497_checker", C.M1497_CHECKER)
        self.assertEqual(old.count(m1497.OLD), 1)
        self.assertEqual(C.M1497_TB.read_text(), old.replace(m1497.OLD, m1497.NEW))

    def test_03_contract_every_leaf_mutation_rejected(self):
        canonical = C.expected_contract()
        count = 0
        for path, value in walk_leaves(canonical):
            candidate = copy.deepcopy(canonical)
            parent_at(candidate, path[:-1])[path[-1]] = mutated_leaf(value)
            with self.subTest(path=".".join(path)):
                with self.assertRaises(RuntimeError):
                    C.validate_contract(candidate, canonical)
            count += 1
        self.assertGreaterEqual(count, 70)

    def test_04_contract_every_key_deletion_rejected(self):
        canonical = C.expected_contract()
        count = 0
        for path, mapping in list(walk_dicts(canonical)):
            for key in tuple(mapping):
                candidate = copy.deepcopy(canonical)
                del parent_at(candidate, path)[key]
                with self.subTest(path=".".join(path + (key,))):
                    with self.assertRaises(RuntimeError):
                        C.validate_contract(candidate, canonical)
                count += 1
        self.assertGreaterEqual(count, 80)

    def test_05_contract_every_object_extra_rejected(self):
        canonical = C.expected_contract()
        count = 0
        for path, _mapping in list(walk_dicts(canonical)):
            candidate = copy.deepcopy(canonical)
            parent_at(candidate, path)["__M1506_EXTRA__"] = True
            with self.subTest(path=".".join(path)):
                with self.assertRaises(RuntimeError):
                    C.validate_contract(candidate, canonical)
            count += 1
        self.assertGreaterEqual(count, 10)

    def test_06_contract_duplicate_key_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "duplicate.json"
            path.write_text('{"schema":1,"schema":2}\n')
            with self.assertRaises(RuntimeError):
                C.strict_json(path)

    def test_07_runtime_exact_reads_required_corpus(self):
        contract = C.expected_contract()
        called = set()
        real_exact = R.exact
        def spy(path, digest):
            called.add(Path(path)); return real_exact(path, digest)
        with mock.patch.object(R, "exact", side_effect=spy):
            R.validate_frozen_inputs(contract)
        required = {R.BASE.PARENT, R.BASE.M935, R.BASE.WRAPPER, R.BASE.SVA,
                    R.BASE.WITNESS, R.BASE.FOUNDRY, R.BASE.VCS,
                    R.BASE.DOCS359, R.CHECKER, R.TESTS, R.P.TB,
                    R.P.FILELIST, R.P.TB_R13}
        self.assertTrue(required <= called, sorted(str(path) for path in required - called))

    def test_08_positive_log_exact_metrics(self):
        audit = R.validate_sim_log(positive_log())
        self.assertEqual(audit["weight_requests"], 2)
        self.assertEqual(audit["psum_requests"], 1)
        self.assertEqual(audit["responses"], 2)
        self.assertEqual(audit["core_accepts"], 2)
        self.assertEqual(audit["psum_commits"], 1)
        self.assertEqual(audit["row_completions"], 1)
        self.assertEqual(audit["task_completions"], 1)
        self.assertEqual(audit["assertion_failures"], 0)

    def test_09_each_witness_metric_mutation_rejected(self):
        for old, new in (("weight_req=2", "weight_req=1"),
                         ("psum_req=1", "psum_req=0"),
                         ("responses=2", "responses=1"),
                         ("core_accepts=2", "core_accepts=1"),
                         ("psum_commits=1", "psum_commits=0"),
                         ("rows=1", "rows=0"),
                         ("tasks=1", "tasks=0"),
                         ("design_issue=2", "design_issue=1"),
                         ("design_commit=1", "design_commit=0"),
                         ("design_rows=1", "design_rows=0"),
                         ("masks=0", "masks=1"),
                         ("faults=0", "faults=1")):
            with self.subTest(field=old):
                with self.assertRaises(RuntimeError):
                    R.validate_sim_log(positive_log().replace(old, new, 1))

    def test_10_cover_and_oracle_mutations_rejected(self):
        mutations = (
            lambda text: text.replace("cp_nonfirst, 80 attempts, 1 match",
                                      "cp_nonfirst, 80 attempts, 0 match"),
            lambda text: text.replace("cp_ii2, 80 attempts, 1 match",
                                      "cp_ii2, 80 attempts, 0 match"),
            lambda text: text.replace("response_cycle_gap=2", "response_cycle_gap=1"),
            lambda text: text.replace("oracle_records=80", "oracle_records=79"),
            lambda text: text.replace(" pass=1 index=0", " pass=0 index=0"),
            lambda text: text.replace(R.R13_ENTER + "\n", ""),
            lambda text: text + R.BASE.R13_PASS + "\n",
        )
        for ordinal, mutate in enumerate(mutations):
            with self.subTest(ordinal=ordinal):
                with self.assertRaises(RuntimeError):
                    R.validate_sim_log(mutate(positive_log()))

    def test_11_error_fatal_assertion_failure_rejected(self):
        for line in ("Error: injected", "Fatal: injected", "$error injected",
                     "$fatal injected", "Assertion failure injected",
                     "assertion produced an error"):
            with self.subTest(line=line):
                with self.assertRaises(RuntimeError):
                    R.validate_sim_log(positive_log() + line + "\n")

    def test_12_unknown_or_nonzero_fault_rejected(self):
        for line in ("boundary_fault=1", "core_fault=x", "m935_fault=z",
                     "faults=2"):
            with self.subTest(line=line):
                with self.assertRaises(RuntimeError):
                    R.validate_sim_log(positive_log() + line + "\n")

    def test_13_post_attempt_raw_failure_is_quarantined(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            raw = temp / "raw"; raw.mkdir()
            paths = {
                "ATTEMPT": temp / "attempt", "RESULT": temp / "result",
                "QUARANTINE": temp / "quarantine", "RAW_BUILD": raw,
                "CLEAN_RESULT_STAGE": temp / "clean",
                "ATTEMPT_STAGE": temp / "attempt_stage",
                "FAILURE_STAGE": temp / "failure_stage",
            }
            completed = SimpleNamespace(returncode=0, stderr="", stdout="")
            with ExitStack() as stack:
                for name, value in paths.items():
                    stack.enter_context(mock.patch.object(R, name, value))
                stack.enter_context(mock.patch.object(R, "validate_authority", return_value=None))
                stack.enter_context(mock.patch.object(R.subprocess, "run", return_value=completed))
                stack.enter_context(mock.patch.object(R, "namespace_gate", return_value=None))
                stack.enter_context(mock.patch.object(R.BASE, "collision_gate", return_value=None))
                stack.enter_context(mock.patch.object(R.BASE, "resource_gate", return_value=None))
                stack.enter_context(mock.patch.object(
                    R, "publish_no_replace", side_effect=lambda source, destination:
                    os.rename(source, destination)))
                with self.assertRaises(FileExistsError):
                    R.main()
            self.assertTrue(paths["ATTEMPT"].is_dir())
            self.assertTrue(paths["QUARANTINE"].is_dir())
            R.P.P.verify_recursive_seal_generic(paths["QUARANTINE"])
            receipt = json.loads((paths["QUARANTINE"] /
                "m1506_c1_active_next_oracle_unit_delay_vcs_receipt_r1.json").read_text())
            self.assertEqual(receipt["status"], "FAILED_OR_INCOMPLETE")
            self.assertFalse(receipt["claim_boundary"]["functional_vcs"])

    def test_14_clean_stage_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name) / "clean"; root.mkdir()
            for name in R.CLEAN_PAYLOAD:
                (root / name).write_text("regular\n")
            target = root / "compile.log"; target.unlink()
            target.symlink_to(root / "sim.log")
            with self.assertRaises(RuntimeError):
                R.seal_clean_result(root)

    def test_15_fresh_future_namespaces_only(self):
        text = RUNNER.read_text() + CHECKER.read_text()
        for number in ("M1507", "M1508", "M1509"):
            self.assertIn(number, text)
        self.assertNotIn("M1501_", text)
        self.assertNotIn("M1502_", text)
        self.assertNotIn("M1503_", text)
        self.assertNotIn("M1504_", text)
        self.assertNotIn("M1505_", text)

    def test_16_failure_nonregular_raw_log_still_sealed(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            raw = temp / "raw"; raw.mkdir()
            target = temp / "target"; target.write_text("must not be followed\n")
            (raw / "compile.log").symlink_to(target)
            (raw / "sim.log").write_text("regular failure log\n")
            stage = temp / "failure_stage"
            with mock.patch.object(R, "RAW_BUILD", raw):
                R.make_clean_evidence(stage, "COMPILE", "RuntimeError: injected",
                                      1, 0, None)
            R.P.P.verify_recursive_seal_generic(stage)
            self.assertIn("nonregular", (stage / "compile.log").read_text())
            receipt = json.loads((stage /
                "m1506_c1_active_next_oracle_unit_delay_vcs_receipt_r1.json").read_text())
            self.assertEqual(receipt["status"], "FAILED_OR_INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
