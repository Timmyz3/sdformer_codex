from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SOURCE = (Path(__file__).resolve().parents[1] / "scripts" /
          "run_m1248_m1241_motion_cross_run_final_checkpoint_binder_one_shot_release_source.py")
CONTRACT = (Path(__file__).resolve().parents[1] / "contracts" /
            "m1248_m1241_motion_cross_run_final_checkpoint_binder_one_shot_release_source_contract_r1_20260830.json")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


M = load("m1248_release_under_test", SOURCE)


class M1248ReleaseSourceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="m1248_source.")
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        self.interpreter = Path(self.tmp.name) / "python"
        self.interpreter.write_text("mock python\n", encoding="utf-8")
        self.interpreter.chmod(0o700)

        pins = {}
        for relative, body in (
            (M.M1241_REL, "m1241 source\n"),
            (M.M1241_TEST_REL, "m1241 tests\n"),
            (M.M1241_CONTRACT_REL, "{}\n"),
        ):
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
            pins[relative] = sha(path)

        docs = self.repo / M.DOCS359_REL
        docs.parent.mkdir(parents=True, exist_ok=True)
        docs.write_text("frozen\n", encoding="utf-8")

        self.review_rel = Path("hw_autoresearch_nts07/reviews/m1245_fixture")
        review_root = self.repo / self.review_rel
        review_root.mkdir(parents=True)
        review = {
            "schema": M.M1245_REVIEW_SCHEMA,
            "status": M.M1245_REVIEW_STATUS,
            "authority": {
                "release_authoring_allowed": True,
                "production_binder_execution_allowed_by_this_review": False,
                "hardware_rebind_authorized": False,
                "result_hammer_still_required": True,
            },
        }
        (review_root / "review.json").write_text(
            json.dumps(review, sort_keys=True) + "\n", encoding="utf-8")
        manifest = review_root / M.MANIFEST
        manifest.write_text("{}  review.json\n".format(
            sha(review_root / "review.json")), encoding="utf-8")
        outer = review_root / M.OUTER
        outer.write_text("{}  {}\n".format(sha(manifest), M.MANIFEST), encoding="utf-8")

        old_run_rel = Path("runs/old")
        new_run_rel = Path("runs/new")
        old_config_rel = Path("configs/old.yml")
        new_config_rel = Path("configs/new.yml")
        manifest_rel = Path("configs/new.json")
        for relative, body in (
            (old_config_rel, b"old config\n"),
            (new_config_rel, b"new config\n"),
            (manifest_rel, b'{"evaluation_epochs":[30,32,34]}\n'),
        ):
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        candidates = (
            M.CandidateInput("legacy_ep29", old_run_rel, old_config_rel, 29),
            M.CandidateInput("resume_ep30", new_run_rel, new_config_rel, 30),
            M.CandidateInput("resume_ep32", new_run_rel, new_config_rel, 32),
            M.CandidateInput("resume_ep34", new_run_rel, new_config_rel, 34),
        )
        for row in candidates:
            run = self.repo / row.run_rel
            run.mkdir(parents=True, exist_ok=True)
            (run / "checkpoint_epoch{}.pth".format(row.epoch)).write_bytes(
                ("checkpoint-{}\n".format(row.epoch)).encode("ascii"))
            profile = run / "standard_valid825" / "epoch{}".format(row.epoch) / "spike_profile.json"
            profile.parent.mkdir(parents=True, exist_ok=True)
            profile.write_text("{}\n", encoding="utf-8")

        results = self.repo / "hw_autoresearch_nts07/results"
        results.mkdir(parents=True)
        self.policy = M.Policy(
            repo=self.repo,
            interpreter=self.interpreter,
            python_version="3.10.20",
            m1241_pins=pins,
            m1245_rel=self.review_rel,
            m1245_manifest_sha256=sha(manifest),
            m1245_outer_sha256=sha(outer),
            docs_rel=M.DOCS359_REL,
            docs_sha256=sha(docs),
            candidates=candidates,
            new_manifest_rel=manifest_rel,
            output_rel=Path("hw_autoresearch_nts07/results/m1248-result"),
            attempt_rel=Path("hw_autoresearch_nts07/results/.m1248-attempt"),
            log_rel=Path("hw_autoresearch_nts07/results/m1248.launch.log"),
        )

    def tearDown(self):
        self.tmp.cleanup()

    @property
    def output(self):
        return self.repo / self.policy.output_rel

    @property
    def attempt(self):
        return self.repo / self.policy.attempt_rel

    @property
    def log(self):
        return self.repo / self.policy.log_rel

    def publish_result(self, *, boundary_override=None):
        self.output.mkdir()
        boundary = {
            "fresh_result_hammer_required": True,
            "hardware_rebind_authorized": False,
            "hardware_speedup": False,
            "system_speedup": False,
        }
        if boundary_override:
            boundary.update(boundary_override)
        result = {
            "schema": M.RESULT_SCHEMA,
            "status": M.RESULT_STATUS,
            "selected": {"candidate_id": "resume_ep32", "epoch": 32},
            "claim_boundary": boundary,
        }
        payloads = {
            "RUN_COMPLETE.txt": M.RUN_COMPLETE,
            "e0_e8_activation_rebind_targets.json": "[]\n",
            "final_checkpoint_selection.json": json.dumps(result) + "\n",
            "four_checkpoint_metrics.csv": "candidate_id,epoch,AEE\n",
            "selected_checkpoint_and_config.json": "{}\n",
        }
        for name, value in payloads.items():
            (self.output / name).write_text(value, encoding="utf-8")
        manifest = self.output / M.MANIFEST
        manifest.write_text("".join(
            "{}  {}\n".format(sha(self.output / name), name)
            for name in sorted(payloads)), encoding="utf-8")
        (self.output / M.OUTER).write_text(
            "{}  {}\n".format(sha(manifest), M.MANIFEST), encoding="utf-8")

    def good_runner(self, command, cwd):
        self.assertEqual(cwd, self.repo)
        self.assertEqual(command, [
            str(self.interpreter), str(self.repo / M.M1241_REL),
            "--ranking-mode", "aee", "--output-dir", str(self.output)])
        self.publish_result()
        return subprocess.CompletedProcess(
            command, 0,
            stdout=M.CHILD_TOKEN + "\nselected_candidate=resume_ep32\nselected_epoch=32\n",
            stderr="")

    def test_01_import_is_inert(self):
        code = (
            "import importlib.util,sys;"
            "s=importlib.util.spec_from_file_location('isolated_m1248',{!r});"
            "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;"
            "s.loader.exec_module(m);print('PASS')").format(str(SOURCE))
        self.assertEqual(subprocess.check_output(
            ["/usr/bin/python3.12", "-c", code]).decode().strip(), "PASS")
        self.assertFalse(self.attempt.exists())

    def test_02_success_consumes_once_logs_and_verifies_double_seal(self):
        M.execute_once(self.policy, self.interpreter, "3.10.20", self.repo,
                       self.good_runner)
        self.assertTrue(self.attempt.is_file())
        self.assertTrue(self.log.is_file())
        self.assertIn("automatic_retry=false", self.attempt.read_text())
        self.assertIn("returncode=0", self.log.read_text())
        result = M.verify_selection_receipt(self.output)
        self.assertEqual(result["selected"]["epoch"], 32)

    def test_03_every_profile_checkpoint_config_and_manifest_preflight_before_attempt(self):
        for path in M.artifact_files(self.policy):
            with self.subTest(path=str(path)):
                content = path.read_bytes()
                path.unlink()
                with self.assertRaises(M.ReleaseError):
                    M.preflight(self.policy, self.interpreter, "3.10.20", self.repo)
                self.assertFalse(self.attempt.exists())
                path.write_bytes(content)

    def test_04_m1241_and_m1245_hammer_drift_rejected_before_attempt(self):
        m1241 = self.repo / M.M1241_REL
        canonical = m1241.read_bytes()
        m1241.write_bytes(canonical + b"drift")
        with self.assertRaisesRegex(M.ReleaseError, "M1241 input SHA drift"):
            M.preflight(self.policy, self.interpreter, "3.10.20", self.repo)
        self.assertFalse(self.attempt.exists())
        m1241.write_bytes(canonical)

        review = self.repo / self.review_rel / "review.json"
        review.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(M.ReleaseError, "M1245 member SHA drift"):
            M.preflight(self.policy, self.interpreter, "3.10.20", self.repo)
        self.assertFalse(self.attempt.exists())

    def test_05_output_attempt_and_log_namespace_collisions_fail_pre_attempt(self):
        self.output.mkdir()
        with self.assertRaisesRegex(M.ReleaseError, "fresh output namespace"):
            M.preflight(self.policy, self.interpreter, "3.10.20", self.repo)
        self.assertFalse(self.attempt.exists())
        self.output.rmdir()

        self.log.write_text("collision\n", encoding="utf-8")
        with self.assertRaisesRegex(M.ReleaseError, "fresh log namespace"):
            M.preflight(self.policy, self.interpreter, "3.10.20", self.repo)
        self.assertFalse(self.attempt.exists())
        self.log.unlink()

        self.attempt.write_text("consumed\n", encoding="utf-8")
        with self.assertRaisesRegex(M.ReleaseError, "fresh attempt namespace"):
            M.preflight(self.policy, self.interpreter, "3.10.20", self.repo)

    def test_06_child_failure_consumes_attempt_and_never_retries(self):
        calls = []
        def failed(command, cwd):
            calls.append(tuple(command))
            return subprocess.CompletedProcess(command, 7, stdout="", stderr="failed")
        with self.assertRaisesRegex(M.ReleaseError, "no retry authorized"):
            M.execute_once(self.policy, self.interpreter, "3.10.20", self.repo, failed)
        self.assertEqual(len(calls), 1)
        self.assertTrue(self.attempt.exists())
        self.assertTrue(self.log.exists())
        with self.assertRaises(M.ReleaseError):
            M.execute_once(self.policy, self.interpreter, "3.10.20", self.repo, failed)
        self.assertEqual(len(calls), 1)

    def test_07_unsealed_or_overauthorized_result_fails_after_consumption(self):
        def unsealed(command, cwd):
            self.output.mkdir()
            return subprocess.CompletedProcess(command, 0, stdout=M.CHILD_TOKEN + "\n", stderr="")
        with self.assertRaises(M.ReleaseError):
            M.execute_once(self.policy, self.interpreter, "3.10.20", self.repo, unsealed)
        self.assertTrue(self.attempt.exists())
        self.assertTrue(self.log.exists())

    def test_08_result_claim_boundary_is_fail_closed(self):
        self.publish_result(boundary_override={"hardware_rebind_authorized": True})
        with self.assertRaisesRegex(M.ReleaseError, "claim boundary"):
            M.verify_selection_receipt(self.output)

    def test_09_interpreter_version_cwd_and_docs_drift_precede_attempt(self):
        attacks = (
            (Path("/wrong/python"), "3.10.20", self.repo),
            (self.interpreter, "3.10.19", self.repo),
            (self.interpreter, "3.10.20", self.repo.parent),
        )
        for executable, version, cwd in attacks:
            with self.subTest(executable=str(executable), version=version, cwd=str(cwd)):
                with self.assertRaises(M.ReleaseError):
                    M.preflight(self.policy, executable, version, cwd)
                self.assertFalse(self.attempt.exists())
        docs = self.repo / self.policy.docs_rel
        docs.write_text("drift\n", encoding="utf-8")
        with self.assertRaisesRegex(M.ReleaseError, "docs/359 SHA drift"):
            M.preflight(self.policy, self.interpreter, "3.10.20", self.repo)
        self.assertFalse(self.attempt.exists())

    def test_10_source_only_and_docs359_remains_frozen(self):
        text = SOURCE.read_text(encoding="utf-8")
        for forbidden in ("import torch", "import cupy", "paramiko", "dc_shell",
                          "vcs -full64", "nvidia-smi"):
            self.assertNotIn(forbidden, text)
        docs = Path(__file__).resolve().parents[1] / "docs/359_DATE终局冻结_20260813.md"
        self.assertEqual(sha(docs),
                         "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(contract["source"]["sha256"], sha(SOURCE))
        self.assertEqual(contract["test"]["sha256"], sha(Path(__file__).resolve()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
