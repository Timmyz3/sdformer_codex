#!/usr/bin/env python3
"""Independent temporary-fixture hammer for the M1171 remote launcher source.

This program never calls the production ``main`` function, never accesses the
remote host, and never reads a checkpoint.  It attacks the injectable launcher
surface and independently checks the sealed source/contract/author identities.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


HW = Path(__file__).resolve().parents[2]
SOURCE = HW / "scripts/run_m1171_motion_final_checkpoint_binder_remote_one_shot_source.py"
TEST = HW / "tests/test_run_m1171_motion_final_checkpoint_binder_remote_one_shot_source.py"
CONTRACT = HW / "contracts/m1171_motion_final_checkpoint_binder_remote_one_shot_launcher_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1171_motion_final_checkpoint_binder_remote_launcher_author_receipt_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

SPEC = importlib.util.spec_from_file_location("m1171_independent_target", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def exact_manifest(directory: Path) -> None:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    assert outer.read_text(encoding="utf-8").split() == [digest(manifest), "SHA256SUMS"]
    rows = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        assert len(fields) == 2
        name = fields[1].lstrip("*")
        assert name not in rows and Path(name).name == name
        rows[name] = fields[0]
    assert set(rows) == {p.name for p in directory.iterdir()} - {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    for name, expected in rows.items():
        assert digest(directory / name) == expected


class Fixture:
    def __init__(self, root: Path):
        self.root = root
        self.repo = root / "repo"
        self.repo.mkdir()
        source_hashes = {}
        for relative, body in ((M.R1_REL, b"r1\n"), (M.R2_REL, b"r2\n"), (M.R3_REL, b"r3\n")):
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            source_hashes[relative] = digest(path)
        docs = self.repo / M.DOCS359_REL
        docs.parent.mkdir(parents=True, exist_ok=True)
        docs.write_bytes(b"protected\n")
        self.target = root / "python3.10-real"
        self.target.write_bytes(b"python target\n")
        self.target.chmod(0o700)
        self.interpreter = root / "python"
        self.interpreter.symlink_to(self.target.name)
        self.run_dir = self.repo / "run"
        standard = self.run_dir / "standard_valid825"
        standard.mkdir(parents=True)
        for epoch in M.EPOCHS:
            epoch_dir = standard / f"epoch{epoch}"
            epoch_dir.mkdir()
            (epoch_dir / "spike_profile.json").write_text("{}\n", encoding="utf-8")
        self.ranking = self.run_dir / "profile_ranking_valid825.md"
        self.ranking.write_text("Ranking mode: `aee`.\n", encoding="utf-8")
        self.config = self.repo / "config.yml"
        self.config.write_bytes(b"config\n")
        results = self.repo / "hw_autoresearch_nts07/results"
        results.mkdir()
        self.policy = M.Policy(
            repo=self.repo,
            interpreter=self.interpreter,
            python_version="3.10.20",
            source_sha256=source_hashes,
            docs_rel=M.DOCS359_REL,
            docs_sha256=digest(docs),
            run_dir=self.run_dir,
            config=self.config,
            config_sha256=digest(self.config),
            ranking=self.ranking,
            epochs=M.EPOCHS,
            output=results / "m1171-result",
            attempt=results / ".m1171-attempt",
        )
        self.calls = []

    def publish(self, *, token: str | None = None, extra: str | None = None) -> None:
        self.policy.output.mkdir()
        payloads = {
            "RUN_COMPLETE.txt": token or (
                "PASS_M1167_FINAL_CHECKPOINT_SELECTED_R3_CANONICAL_EPOCH_NAMES__"
                "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY\n"
            ),
            "e0_e8_rebind_targets.json": "{}\n",
            "final_checkpoint_selection.json": "{}\n",
            "five_checkpoint_metrics.csv": "epoch,AEE\n",
        }
        for name, value in payloads.items():
            (self.policy.output / name).write_text(value, encoding="utf-8")
        if extra is not None:
            (self.policy.output / extra).write_text("attack\n", encoding="utf-8")
        manifest = self.policy.output / M.MANIFEST
        manifest.write_text("".join(
            f"{digest(self.policy.output / name)}  {name}\n" for name in sorted(payloads)
        ), encoding="utf-8")
        (self.policy.output / M.OUTER).write_text(
            f"{digest(manifest)}  {M.MANIFEST}\n", encoding="utf-8"
        )

    def good_runner(self, command, cwd):
        self.calls.append(tuple(command))
        assert cwd == self.repo
        self.publish()
        return subprocess.CompletedProcess(command, 0, stdout=(
            "PASS_M1167_FINAL_CHECKPOINT_SELECTED_R3_CANONICAL_EPOCH_NAMES__"
            "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY\n"
        ), stderr="")

    def run(self, runner=None, *, executable=None, version="3.10.20", cwd=None):
        return M.execute_once(
            self.policy,
            executable or self.interpreter,
            version,
            cwd or self.repo,
            runner or self.good_runner,
        )


class IndependentHammer(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="m1173_m1171_hammer.")
        self.f = Fixture(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def assert_preflight_reject(self, pattern: str):
        with self.assertRaisesRegex(M.LaunchError, pattern):
            self.f.run()
        self.assertFalse(self.f.policy.attempt.exists())
        self.assertEqual(self.f.calls, [])

    def test_00_exact_production_constants_and_identities(self):
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(digest(SOURCE), "ec3483ec484e3e61c7bb27530682b597837e375c2649403f9b27617b4b54c695")
        self.assertEqual(digest(TEST), "012d308a324dd9368c9c48ffe0e0936ab184c1193208c12c60158a43505eb52b")
        self.assertEqual(digest(CONTRACT), "e524452deedeaa4231323bce5ca600b9c431bc98a0f02886b4583b2aa7b9f376")
        self.assertEqual(M.INTERPRETER, Path("/opt/conda/envs/sdformerflow/bin/python"))
        self.assertEqual(M.PYTHON_VERSION, "3.10.20")
        self.assertEqual(tuple(M.EPOCHS), (9, 14, 19, 24, 29))
        self.assertEqual(M.DOCS359_SHA256, digest(DOCS359))
        self.assertEqual(contract["one_shot_execution"]["exact_child_argv"], M.preflight.__globals__["PRODUCTION_POLICY"] and [
            str(M.INTERPRETER), str(M.REPO / M.R3_REL), "--run-dir", str(M.RUN_DIR),
            "--config", str(M.CONFIG), "--ranking", str(M.RANKING), "--ranking-mode", "aee",
            "--output-dir", str(M.OUTPUT),
        ])
        for rel, expected in M.SOURCE_SHA256.items():
            self.assertEqual(digest(HW.parent / rel), expected)

    def test_01_author_and_contract_seals(self):
        exact_manifest(AUTHOR)
        sidecar = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256")
        self.assertEqual(sidecar.read_text().split(), [digest(CONTRACT), CONTRACT.name])
        outer = Path(str(sidecar) + ".seal.sha256")
        self.assertEqual(outer.read_text().split(), [digest(sidecar), sidecar.name])

    def test_02_success_exact_argv_attempt_and_one_child(self):
        completed = self.f.run()
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(len(self.f.calls), 1)
        expected = [str(self.f.interpreter), str(self.f.repo / M.R3_REL),
                    "--run-dir", str(self.f.run_dir), "--config", str(self.f.config),
                    "--ranking", str(self.f.ranking), "--ranking-mode", "aee",
                    "--output-dir", str(self.f.policy.output)]
        self.assertEqual(list(self.f.calls[0]), expected)
        attempt = self.f.policy.attempt.read_text(encoding="utf-8")
        self.assertIn("ATTEMPT_CONSUMED_BEFORE_CHILD", attempt)
        self.assertIn("automatic_retry=false", attempt)

    def test_03_wrong_interpreter(self):
        with self.assertRaisesRegex(M.LaunchError, "interpreter path mismatch"):
            self.f.run(executable=Path("/wrong/python"))
        self.assertFalse(self.f.policy.attempt.exists())

    def test_04_wrong_version(self):
        with self.assertRaisesRegex(M.LaunchError, "version mismatch"):
            self.f.run(version="3.10.19")
        self.assertFalse(self.f.policy.attempt.exists())

    def test_05_wrong_cwd(self):
        with self.assertRaisesRegex(M.LaunchError, "repository cwd mismatch"):
            self.f.run(cwd=self.f.root)
        self.assertFalse(self.f.policy.attempt.exists())

    def test_06_broken_interpreter_symlink(self):
        self.f.interpreter.unlink(); self.f.interpreter.symlink_to("missing")
        self.assert_preflight_reject("broken or cyclic")

    def test_07_nonexecuting_symlink_target(self):
        self.f.target.chmod(0o600)
        self.assert_preflight_reject("executable regular file")

    def test_08_directory_symlink_target(self):
        self.f.interpreter.unlink(); self.f.interpreter.symlink_to(self.f.repo)
        self.assert_preflight_reject("executable regular file")

    def test_09_each_source_drift(self):
        for relative in M.SOURCE_SHA256:
            with self.subTest(relative=str(relative)):
                original = (self.f.repo / relative).read_bytes()
                (self.f.repo / relative).write_bytes(original + b"drift")
                with self.assertRaisesRegex(M.LaunchError, "sealed source SHA drift"):
                    self.f.run()
                self.assertFalse(self.f.policy.attempt.exists())
                (self.f.repo / relative).write_bytes(original)

    def test_10_source_symlink(self):
        path = self.f.repo / M.R3_REL
        copy = path.with_name("r3-copy")
        path.rename(copy); path.symlink_to(copy.name)
        self.assert_preflight_reject("non-symlink regular file")

    def test_11_docs_drift(self):
        (self.f.repo / M.DOCS359_REL).write_text("changed\n")
        self.assert_preflight_reject("docs/359 SHA drift")

    def test_12_config_drift(self):
        self.f.config.write_text("changed\n")
        self.assert_preflight_reject("configuration SHA drift")

    def test_13_config_symlink(self):
        copy = self.f.config.with_name("config-copy")
        self.f.config.rename(copy); self.f.config.symlink_to(copy.name)
        self.assert_preflight_reject("non-symlink regular file")

    def test_14_ranking_symlink(self):
        copy = self.f.ranking.with_name("ranking-copy")
        self.f.ranking.rename(copy); self.f.ranking.symlink_to(copy.name)
        self.assert_preflight_reject("non-symlink regular file")

    def test_15_epoch_alias_and_extra_entry(self):
        (self.f.run_dir / "standard_valid825/epoch09").mkdir()
        self.assert_preflight_reject("canonical epoch population")

    def test_16_missing_epoch(self):
        shutil.rmtree(self.f.run_dir / "standard_valid825/epoch29")
        self.assert_preflight_reject("canonical epoch population")

    def test_17_epoch_directory_symlink(self):
        epoch = self.f.run_dir / "standard_valid825/epoch29"
        moved = self.f.run_dir / "epoch29-real"
        epoch.rename(moved); epoch.symlink_to(moved, target_is_directory=True)
        self.assert_preflight_reject("non-symlink directory")

    def test_18_profile_symlink(self):
        profile = self.f.run_dir / "standard_valid825/epoch29/spike_profile.json"
        moved = profile.with_name("profile-real.json")
        profile.rename(moved); profile.symlink_to(moved.name)
        self.assert_preflight_reject("non-symlink regular file")

    def test_19_preexisting_output_directory(self):
        self.f.policy.output.mkdir()
        self.assert_preflight_reject("fresh output namespace")

    def test_20_preexisting_output_symlink(self):
        self.f.policy.output.symlink_to(self.f.repo, target_is_directory=True)
        self.assert_preflight_reject("fresh output namespace")

    def test_21_preexisting_attempt(self):
        self.f.policy.attempt.write_text("used\n")
        with self.assertRaisesRegex(M.LaunchError, "attempt already consumed"):
            self.f.run()
        self.assertEqual(self.f.policy.attempt.read_text(), "used\n")
        self.assertEqual(self.f.calls, [])

    def test_22_child_failure_consumes_and_never_retries(self):
        calls = []
        def fail(command, cwd):
            calls.append(tuple(command)); return subprocess.CompletedProcess(command, 7, stdout="", stderr="x")
        with self.assertRaisesRegex(M.LaunchError, "no retry authorized"):
            self.f.run(runner=fail)
        self.assertEqual(len(calls), 1)
        self.assertTrue(self.f.policy.attempt.is_file())
        with self.assertRaisesRegex(M.LaunchError, "attempt already consumed"):
            self.f.run(runner=fail)
        self.assertEqual(len(calls), 1)

    def test_23_unsealed_output_consumes_attempt(self):
        def unsealed(command, cwd):
            self.f.policy.output.mkdir()
            return subprocess.CompletedProcess(command, 0, stdout=(
                "PASS_M1167_FINAL_CHECKPOINT_SELECTED_R3_CANONICAL_EPOCH_NAMES__"
                "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY\n"), stderr="")
        with self.assertRaises(M.LaunchError): self.f.run(runner=unsealed)
        self.assertTrue(self.f.policy.attempt.exists())

    def test_24_extra_output_member_rejected(self):
        def extra(command, cwd):
            self.f.publish(extra="injected")
            return subprocess.CompletedProcess(command, 0, stdout=(
                "PASS_M1167_FINAL_CHECKPOINT_SELECTED_R3_CANONICAL_EPOCH_NAMES__"
                "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY\n"), stderr="")
        with self.assertRaisesRegex(M.LaunchError, "member set mismatch"): self.f.run(runner=extra)

    def test_25_stdout_duplicate_rejected(self):
        def duplicate(command, cwd):
            self.f.publish()
            token = ("PASS_M1167_FINAL_CHECKPOINT_SELECTED_R3_CANONICAL_EPOCH_NAMES__"
                     "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY\n")
            return subprocess.CompletedProcess(command, 0, stdout=token + token, stderr="")
        with self.assertRaisesRegex(M.LaunchError, "stdout mismatch"): self.f.run(runner=duplicate)

    def test_26_wrong_terminal_token_rejected(self):
        def wrong(command, cwd):
            self.f.publish(token="WRONG\n")
            return subprocess.CompletedProcess(command, 0, stdout=(
                "PASS_M1167_FINAL_CHECKPOINT_SELECTED_R3_CANONICAL_EPOCH_NAMES__"
                "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY\n"), stderr="")
        with self.assertRaisesRegex(M.LaunchError, "terminal token mismatch"): self.f.run(runner=wrong)

    def test_27_default_runner_clean_environment(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn('env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}', source)
        self.assertIn("stdout=subprocess.PIPE", source)
        self.assertIn("stderr=subprocess.PIPE", source)
        self.assertNotIn("shell=True", source)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(IndependentHammer)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = {
        "testsRun": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "successful": result.wasSuccessful(),
    }
    print("HAMMER_JSON=" + json.dumps(summary, sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
