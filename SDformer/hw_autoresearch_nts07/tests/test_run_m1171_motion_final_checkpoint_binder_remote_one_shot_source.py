from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/run_m1171_motion_final_checkpoint_binder_remote_one_shot_source.py"
SPEC = importlib.util.spec_from_file_location("m1171_launcher", SOURCE)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class M1171RemoteLauncherMockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="m1171_mock.")
        root = Path(self.temp.name)
        repo = root / "repo"
        scripts = repo / "hw_autoresearch_nts07/scripts"
        scripts.mkdir(parents=True)
        sources = {}
        for relative, body in (
            (M.R1_REL, b"r1\n"), (M.R2_REL, b"r2\n"), (M.R3_REL, b"r3\n")
        ):
            path = repo / relative
            path.write_bytes(body)
            sources[relative] = digest(path)
        docs = repo / M.DOCS359_REL
        docs.parent.mkdir(parents=True)
        docs.write_bytes(b"protected\n")
        interpreter = root / "python"
        interpreter.write_bytes(b"mock interpreter\n")
        interpreter.chmod(0o700)
        run = repo / "run"
        standard = run / "standard_valid825"
        standard.mkdir(parents=True)
        for epoch in M.EPOCHS:
            epoch_dir = standard / f"epoch{epoch}"
            epoch_dir.mkdir()
            (epoch_dir / "spike_profile.json").write_text("{}\n", encoding="utf-8")
        ranking = run / "profile_ranking_valid825.md"
        ranking.write_text("Ranking mode: `aee`.\n", encoding="utf-8")
        config = repo / "config.yml"
        config.write_bytes(b"config\n")
        results = repo / "hw_autoresearch_nts07/results"
        results.mkdir()
        self.policy = M.Policy(
            repo=repo, interpreter=interpreter, python_version="3.10.20",
            source_sha256=sources, docs_rel=M.DOCS359_REL,
            docs_sha256=digest(docs), run_dir=run, config=config,
            config_sha256=digest(config), ranking=ranking, epochs=M.EPOCHS,
            output=results / "m1171-result", attempt=results / ".m1171-attempt",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def publish_mock_result(self) -> None:
        self.policy.output.mkdir()
        payloads = {
            "RUN_COMPLETE.txt":
                "PASS_M1167_FINAL_CHECKPOINT_SELECTED_R3_CANONICAL_EPOCH_NAMES__"
                "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY\n",
            "e0_e8_rebind_targets.json": "{}\n",
            "final_checkpoint_selection.json": "{}\n",
            "five_checkpoint_metrics.csv": "epoch,AEE\n",
        }
        for name, value in payloads.items():
            (self.policy.output / name).write_text(value, encoding="utf-8")
        manifest = self.policy.output / M.MANIFEST
        manifest.write_text("".join(
            f"{digest(self.policy.output / name)}  {name}\n" for name in sorted(payloads)
        ), encoding="utf-8")
        (self.policy.output / M.OUTER).write_text(
            f"{digest(manifest)}  {M.MANIFEST}\n", encoding="utf-8")

    def good_runner(self, command, cwd):
        self.assertEqual(cwd, self.policy.repo)
        self.assertEqual(command[0], str(self.policy.interpreter))
        self.assertEqual(command[1], str(self.policy.repo / M.R3_REL))
        self.assertEqual(command[-2:], ["--output-dir", str(self.policy.output)])
        self.publish_mock_result()
        return subprocess.CompletedProcess(command, 0,
            stdout=("PASS_M1167_FINAL_CHECKPOINT_SELECTED_R3_CANONICAL_EPOCH_NAMES__"
                    "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY\n"),
            stderr="")

    def test_success_consumes_once_and_verifies_double_seal(self):
        calls = []
        def runner(command, cwd):
            calls.append(tuple(command))
            return self.good_runner(command, cwd)
        M.execute_once(self.policy, self.policy.interpreter, "3.10.20",
                       self.policy.repo, runner)
        self.assertEqual(len(calls), 1)
        self.assertTrue(self.policy.attempt.is_file())
        self.assertIn("automatic_retry=false", self.policy.attempt.read_text())
        M.verify_sealed_output(self.policy.output)

    def test_wrong_interpreter_rejected_before_attempt(self):
        with self.assertRaisesRegex(M.LaunchError, "interpreter path mismatch"):
            M.execute_once(self.policy, Path("/wrong/python"), "3.10.20",
                           self.policy.repo, self.good_runner)
        self.assertFalse(self.policy.attempt.exists())

    def test_wrong_version_rejected_before_attempt(self):
        with self.assertRaisesRegex(M.LaunchError, "version mismatch"):
            M.execute_once(self.policy, self.policy.interpreter, "3.10.19",
                           self.policy.repo, self.good_runner)
        self.assertFalse(self.policy.attempt.exists())

    def test_conda_style_interpreter_symlink_is_allowed(self):
        target = self.policy.interpreter.with_name("python3.10")
        self.policy.interpreter.rename(target)
        self.policy.interpreter.symlink_to(target.name)
        command = M.preflight(self.policy, self.policy.interpreter, "3.10.20",
                              self.policy.repo)
        self.assertEqual(command[0], str(self.policy.interpreter))

    def test_source_drift_rejected_before_attempt(self):
        (self.policy.repo / M.R2_REL).write_text("drift\n", encoding="utf-8")
        with self.assertRaisesRegex(M.LaunchError, "sealed source SHA drift"):
            M.execute_once(self.policy, self.policy.interpreter, "3.10.20",
                           self.policy.repo, self.good_runner)
        self.assertFalse(self.policy.attempt.exists())

    def test_profile_alias_rejected_before_attempt(self):
        alias = self.policy.run_dir / "standard_valid825/epoch09"
        alias.mkdir()
        with self.assertRaisesRegex(M.LaunchError, "canonical epoch population"):
            M.execute_once(self.policy, self.policy.interpreter, "3.10.20",
                           self.policy.repo, self.good_runner)
        self.assertFalse(self.policy.attempt.exists())

    def test_child_failure_consumes_attempt_and_never_retries(self):
        calls = []
        def runner(command, cwd):
            calls.append(tuple(command))
            return subprocess.CompletedProcess(command, 7, stdout="", stderr="failed")
        with self.assertRaisesRegex(M.LaunchError, "no retry authorized"):
            M.execute_once(self.policy, self.policy.interpreter, "3.10.20",
                           self.policy.repo, runner)
        self.assertEqual(len(calls), 1)
        self.assertTrue(self.policy.attempt.exists())
        with self.assertRaisesRegex(M.LaunchError, "attempt already consumed"):
            M.execute_once(self.policy, self.policy.interpreter, "3.10.20",
                           self.policy.repo, runner)
        self.assertEqual(len(calls), 1)

    def test_unsealed_output_fails_after_attempt_consumption(self):
        def runner(command, cwd):
            self.policy.output.mkdir()
            return subprocess.CompletedProcess(command, 0,
                stdout=("PASS_M1167_FINAL_CHECKPOINT_SELECTED_R3_CANONICAL_EPOCH_NAMES__"
                        "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY\n"),
                stderr="")
        with self.assertRaises(M.LaunchError):
            M.execute_once(self.policy, self.policy.interpreter, "3.10.20",
                           self.policy.repo, runner)
        self.assertTrue(self.policy.attempt.exists())

    def test_preexisting_output_rejected_without_consuming_attempt(self):
        self.policy.output.mkdir()
        with self.assertRaisesRegex(M.LaunchError, "fresh output namespace"):
            M.execute_once(self.policy, self.policy.interpreter, "3.10.20",
                           self.policy.repo, self.good_runner)
        self.assertFalse(self.policy.attempt.exists())


if __name__ == "__main__":
    unittest.main()
