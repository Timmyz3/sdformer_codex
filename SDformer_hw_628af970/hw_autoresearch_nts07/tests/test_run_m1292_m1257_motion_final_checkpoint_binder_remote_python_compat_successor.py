from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/run_m1292_m1257_motion_final_checkpoint_binder_remote_python_compat_successor.py"
M1257_TEST = ROOT / "tests/test_run_m1257_m1253_motion_cross_run_final_checkpoint_binder_one_shot_release_successor.py"
CONTRACT = ROOT / "contracts/m1292_m1257_motion_final_checkpoint_binder_remote_python_compat_successor_source_contract_r1_20260830.json"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


S = load("m1292_source_under_test", SOURCE)
T = load("m1292_frozen_m1257_fixture", M1257_TEST)


class M1292CompatibilitySuccessorTest(unittest.TestCase):
    def setUp(self):
        self.old = T.M1257ReleaseSuccessorTest(
            methodName="test_01_import_is_inert_and_predecessor_is_exact")
        self.old.setUp()
        # The inherited fixture uses arbitrary marker bytes for the three sealed
        # execution sources.  M1292 intentionally compiles those exact bytes, so
        # replace only the temporary fixture payloads with minimal valid Python
        # and rebind their fixture-only SHA pins.
        for index, relative in enumerate(self.old.policy.base.execution_pins):
            path = self.old.fx.repo / relative
            path.write_text("MARK_{} = {!r}\n".format(index, str(relative)),
                            encoding="utf-8")
            self.old.policy.base.execution_pins[relative] = sha(path)
        self.policy = S.rebind_interpreter(self.old.policy)
        self.good_probe = {key: True for key in S.RUNTIME_KEYS}
        self.good_probe.update(interpreter=str(S.TARGET_INTERPRETER),
                               version=S.TARGET_PYTHON_VERSION)

    def tearDown(self):
        self.old.tearDown()

    def probe(self, *_):
        return copy.deepcopy(self.good_probe)

    def prepare(self):
        return S.prepare(self.policy, S.TARGET_INTERPRETER,
                         S.TARGET_PYTHON_VERSION, self.old.fx.repo,
                         probe=self.probe)

    def test_01_import_is_inert_and_m1257_is_frozen(self):
        code = ("import importlib.util,sys;"
                "s=importlib.util.spec_from_file_location('isolated_m1292',{!r});"
                "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;"
                "s.loader.exec_module(m);print('PASS')").format(str(SOURCE))
        self.assertEqual(subprocess.check_output(
            ["/usr/bin/python3.12", "-c", code]).decode().strip(), "PASS")
        self.assertEqual(sha(S.M1257_SOURCE), S.M1257_SOURCE_SHA256)
        self.assertEqual(sha(S.M1257_TEST), S.M1257_TEST_SHA256)
        self.assertEqual(sha(S.M1257_CONTRACT), S.M1257_CONTRACT_SHA256)
        self.assertFalse(self.old.fx.attempt.exists())
        self.assertFalse(self.old.fx.log.exists())
        self.assertFalse(self.old.output.exists())

    def test_02_only_interpreter_and_version_change(self):
        old, new = self.old.policy, self.policy
        self.assertEqual(new.base.interpreter, Path("/usr/bin/python3"))
        self.assertEqual(new.base.python_version, "3.12.3")
        for field in S.M.B.Policy.__dataclass_fields__:
            if field not in ("interpreter", "python_version"):
                self.assertEqual(getattr(new.base, field), getattr(old.base, field))
        for field in S.M.Policy.__dataclass_fields__:
            if field != "base":
                self.assertEqual(getattr(new, field), getattr(old, field))

    def test_03_remote_repo_candidate_config_checkpoint_profile_identity_unchanged(self):
        self.assertEqual(S.PRODUCTION_POLICY.base.repo, S.TARGET_REPO)
        self.assertEqual(S.PRODUCTION_POLICY.base.repo, S.M.PRODUCTION_POLICY.base.repo)
        self.assertEqual(S.PRODUCTION_POLICY.base.candidates,
                         S.M.PRODUCTION_POLICY.base.candidates)
        self.assertEqual(S.M.B.artifact_map(S.PRODUCTION_POLICY.base),
                         S.M.B.artifact_map(S.M.PRODUCTION_POLICY.base))
        self.assertEqual(S.PRODUCTION_POLICY.base.execution_pins,
                         S.M.PRODUCTION_POLICY.base.execution_pins)

    def test_04_actual_python310_without_memfd_or_seals_fails_closed(self):
        code = ("import importlib.util,sys;from pathlib import Path;"
                "s=importlib.util.spec_from_file_location('m1292_py310',{!r});"
                "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;"
                "s.loader.exec_module(m);"
                "p=m.probe_current_runtime(m.TARGET_INTERPRETER,m.TARGET_PYTHON_VERSION);"
                "print(int(p.os_memfd_create),int(p.fcntl_add_seals));"
                "\ntry:m.validate_runtime_probe(p)\n"
                "except Exception:print('FAIL_CLOSED')\n"
                "else:raise SystemExit(9)").format(str(SOURCE))
        output = subprocess.check_output([
            "/opt/anaconda3/envs/pytorch310/bin/python", "-c", code]).decode().splitlines()
        self.assertEqual(output, ["0 0", "FAIL_CLOSED"])

    def test_05_python312_memfd_seals_and_stdlib_positive_control(self):
        code = ("import importlib.util,sys;"
                "s=importlib.util.spec_from_file_location('m1292_py312',{!r});"
                "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;"
                "s.loader.exec_module(m);"
                "p=m.probe_current_runtime(m.TARGET_INTERPRETER,m.TARGET_PYTHON_VERSION);"
                "m.validate_runtime_probe(p);"
                "print('PASS',len(m.CHILD_STDLIB_MODULES))").format(str(SOURCE))
        self.assertEqual(subprocess.check_output(
            ["/usr/bin/python3.12", "-c", code]).decode().strip(), "PASS 15")

    def test_06_prepare_has_eleven_snapshots_three_sealed_children_and_target_command(self):
        prepared = self.prepare()
        try:
            self.assertEqual(len(prepared.snapshots), 11)
            self.assertEqual(len(prepared.source_fds), 3)
            self.assertEqual(prepared.command[0], "/usr/bin/python3")
            self.assertEqual(tuple(map(int, prepared.command[-4:-1])),
                             prepared.source_fds)
            for descriptor in prepared.source_fds:
                self.assertGreater(os.fstat(descriptor).st_size, 0)
        finally:
            prepared.close()

    def test_07_interpreter_version_and_repository_drift_rejected_before_base_prepare(self):
        attacks = (
            (Path("/usr/bin/python3.12"), "3.12.3", self.old.fx.repo),
            (S.TARGET_INTERPRETER, "3.12.4", self.old.fx.repo),
            (S.TARGET_INTERPRETER, True, self.old.fx.repo),
            (S.TARGET_INTERPRETER, "3.12.3", self.old.fx.repo / "wrong"),
        )
        for executable, version, cwd in attacks:
            with self.subTest(executable=executable, version=version, cwd=cwd):
                with mock.patch.object(S.M, "prepare") as inherited:
                    with self.assertRaises(S.M.B.ReleaseError):
                        S.prepare(self.policy, executable, version, cwd,
                                  probe=self.probe)
                    inherited.assert_not_called()

    def test_08_runtime_probe_missing_extra_bool_and_string_attacks_rejected(self):
        attacks = []
        value = copy.deepcopy(self.good_probe); value.pop("fcntl_seal_write"); attacks.append(value)
        value = copy.deepcopy(self.good_probe); value["extra"] = False; attacks.append(value)
        value = copy.deepcopy(self.good_probe); value["os_memfd_create"] = 1; attacks.append(value)
        value = copy.deepcopy(self.good_probe); value["fcntl_add_seals"] = "true"; attacks.append(value)
        value = copy.deepcopy(self.good_probe); value["sealed_launcher_compiles"] = 1; attacks.append(value)
        for value in attacks:
            with self.subTest(value=value):
                with self.assertRaises(S.M.B.ReleaseError):
                    S.validate_runtime_probe(value)

    def test_09_result_claim_integer_zero_is_rejected_after_valid_reseal(self):
        prepared = self.prepare()
        try:
            self.old.publish(prepared)
            path = self.old.output / "final_checkpoint_selection.json"
            value = json.loads(path.read_text())
            false_key = next(key for key, expected in S.M.B.EXACT_CLAIM_BOUNDARY.items()
                             if expected is False)
            value["claim_boundary"][false_key] = 0
            path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
            self.old.reseal()
            with self.assertRaisesRegex(S.M.B.ReleaseError, "exact boolean"):
                S.verify_receipt(self.old.output, prepared)
        finally:
            prepared.close()

    def test_10_positive_and_extra_claims_are_rejected(self):
        for attack in ("positive", "extra_false"):
            with self.subTest(attack=attack):
                prepared = self.prepare()
                try:
                    self.old.publish(prepared)
                    path = self.old.output / "final_checkpoint_selection.json"
                    value = json.loads(path.read_text())
                    if attack == "positive":
                        false_key = next(key for key, expected in S.M.B.EXACT_CLAIM_BOUNDARY.items()
                                         if expected is False)
                        value["claim_boundary"][false_key] = True
                    else:
                        value["claim_boundary"]["paper_metric"] = False
                    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
                    self.old.reseal()
                    with self.assertRaises(S.M.B.ReleaseError):
                        S.verify_receipt(self.old.output, prepared)
                finally:
                    prepared.close()
                for child in self.old.output.iterdir(): child.unlink()
                self.old.output.rmdir()

    def test_11_contract_claim_boundary_is_closed_and_exact_bool(self):
        S.validate_contract_claim_boundary(dict(S.CONTRACT_CLAIM_BOUNDARY))
        attacks = []
        value = dict(S.CONTRACT_CLAIM_BOUNDARY); value["paper_metric"] = 0; attacks.append(value)
        value = dict(S.CONTRACT_CLAIM_BOUNDARY); value["extra_false"] = False; attacks.append(value)
        value = dict(S.CONTRACT_CLAIM_BOUNDARY); value["hardware_speedup"] = True; attacks.append(value)
        for value in attacks:
            with self.assertRaises(S.M.B.ReleaseError):
                S.validate_contract_claim_boundary(value)

    def test_12_o_excl_and_no_retry_are_preserved(self):
        calls = []
        def failed(command, cwd, pass_fds):
            calls.append(tuple(command))
            return subprocess.CompletedProcess(command, 19, stdout="", stderr="failed")
        with self.assertRaisesRegex(S.M.B.ReleaseError, "no retry"):
            S.execute_once(self.policy, S.TARGET_INTERPRETER,
                           S.TARGET_PYTHON_VERSION, self.old.fx.repo,
                           runner=failed, probe=self.probe)
        self.assertEqual(len(calls), 1)
        self.assertTrue(self.old.fx.attempt.exists())
        self.assertTrue(self.old.fx.log.exists())
        with self.assertRaises(S.M.B.ReleaseError):
            S.execute_once(self.policy, S.TARGET_INTERPRETER,
                           S.TARGET_PYTHON_VERSION, self.old.fx.repo,
                           runner=failed, probe=self.probe)
        self.assertEqual(len(calls), 1)

    def test_13_source_has_no_remote_gpu_eda_or_checkpoint_selection_action(self):
        text = SOURCE.read_text(encoding="utf-8")
        for forbidden in ("paramiko", "ssh ", "scp ", "rsync ", "import torch",
                          "nvidia-smi", "dc_shell", "vcs -full64"):
            self.assertNotIn(forbidden, text)
        self.assertFalse(self.old.fx.attempt.exists())
        self.assertFalse(self.old.output.exists())

    def test_14_contract_identity_when_present_and_docs359_frozen(self):
        docs = ROOT / "docs/359_DATE终局冻结_20260813.md"
        self.assertEqual(sha(docs), S.M.B.DOCS359_SHA256)
        if CONTRACT.exists():
            value = json.loads(CONTRACT.read_text(encoding="utf-8"))
            self.assertEqual(value["source"]["sha256"], sha(SOURCE))
            self.assertEqual(value["test"]["sha256"], sha(Path(__file__).resolve()))
            S.validate_contract_claim_boundary(value["claim_boundary"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
