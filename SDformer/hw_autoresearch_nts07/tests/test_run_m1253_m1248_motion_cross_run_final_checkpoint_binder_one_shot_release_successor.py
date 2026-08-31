from __future__ import annotations

import csv
import fcntl
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
          "run_m1253_m1248_motion_cross_run_final_checkpoint_binder_one_shot_release_successor.py")
CONTRACT = (Path(__file__).resolve().parents[1] / "contracts" /
            "m1253_m1248_motion_cross_run_final_checkpoint_binder_one_shot_release_successor_source_contract_r1_20260830.json")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


M = load("m1253_release_under_test", SOURCE)


class M1253ReleaseSuccessorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="m1253_source.")
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        self.interpreter = Path(self.tmp.name) / "python"
        self.interpreter.write_text("mock python\n", encoding="utf-8")
        self.interpreter.chmod(0o700)

        authority = {}
        for index, relative in enumerate(M.M1248_PINS):
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("authority {}\n".format(index), encoding="utf-8")
            authority[relative] = sha(path)
        auxiliary = {}
        for index, relative in enumerate(M.M1241_AUX_PINS):
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("auxiliary {}\n".format(index), encoding="utf-8")
            auxiliary[relative] = sha(path)
        execution = {}
        self.execution_payloads = {}
        for index, relative in enumerate((M.M1241_SOURCE_REL, M.M1234_SOURCE_REL,
                                          M.M1228_SOURCE_REL)):
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = ("sealed execution source {}\n".format(index)).encode()
            path.write_bytes(payload)
            execution[relative] = sha(path)
            self.execution_payloads[relative] = payload

        self.review_rel = Path("hw_autoresearch_nts07/reviews/m1251_fixture")
        review_root = self.repo / self.review_rel
        review_root.mkdir(parents=True)
        review = {
            "schema": M.M1251_SCHEMA, "status": M.M1251_STATUS,
            "authority": {
                "production_execution_authorized_now": False,
                "future_execution_authorized_by_M1251": False,
                "release_successor_authoring_required": True,
                "fresh_different_author_successor_hammer_required": True,
            },
        }
        (review_root / "review.json").write_text(
            json.dumps(review, sort_keys=True) + "\n", encoding="utf-8")
        manifest = review_root / M.MANIFEST
        manifest.write_text("{}  review.json\n".format(sha(review_root / "review.json")),
                            encoding="utf-8")
        outer = review_root / M.OUTER
        outer.write_text("{}  {}\n".format(sha(manifest), M.MANIFEST), encoding="utf-8")

        docs = self.repo / M.DOCS359_REL
        docs.parent.mkdir(parents=True, exist_ok=True)
        docs.write_text("frozen\n", encoding="utf-8")

        old_run_rel, new_run_rel = Path("runs/old"), Path("runs/new")
        old_config_rel, new_config_rel = Path("configs/old.yml"), Path("configs/new.yml")
        manifest_rel = Path("configs/new.json")
        for relative, payload in (
            (old_config_rel, b"old config\n"), (new_config_rel, b"new config\n"),
            (manifest_rel, b'{"evaluation_epochs":[30,32,34]}\n')):
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        candidates = (
            M.CandidateInput("legacy_ep29", old_run_rel, "old", old_config_rel, 29),
            M.CandidateInput("resume_ep30", new_run_rel, "new", new_config_rel, 30),
            M.CandidateInput("resume_ep32", new_run_rel, "new", new_config_rel, 32),
            M.CandidateInput("resume_ep34", new_run_rel, "new", new_config_rel, 34),
        )
        for row in candidates:
            run = self.repo / row.run_rel
            run.mkdir(parents=True, exist_ok=True)
            (run / "checkpoint_epoch{}.pth".format(row.epoch)).write_bytes(
                ("checkpoint {}\n".format(row.epoch)).encode())
            profile = run / "standard_valid825" / "epoch{}".format(row.epoch) / "spike_profile.json"
            profile.parent.mkdir(parents=True, exist_ok=True)
            profile.write_text("{}\n", encoding="utf-8")
        (self.repo / "hw_autoresearch_nts07/results").mkdir(parents=True)
        self.policy = M.Policy(
            self.repo, self.interpreter, "3.10.20", authority, execution, auxiliary,
            self.review_rel, sha(manifest), sha(outer), M.DOCS359_REL, sha(docs),
            candidates, manifest_rel,
            Path("hw_autoresearch_nts07/results/m1253-result"),
            Path("hw_autoresearch_nts07/results/.m1253-attempt"),
            Path("hw_autoresearch_nts07/results/m1253.launch.log"),
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

    def identity(self, snapshot, profile=False):
        value = snapshot.receipt_identity()
        if profile:
            value.update({
                "immutable_single_read": True, "hash_and_parse_same_bytes": True,
                "post_parse_path_identity_frozen": True,
                "descriptor_rooted_no_symlink_components": True,
                "samples": 825, "artifact_identity_exact": True,
                "load_audit_exact_zero": True,
                "module_counts": {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12},
            })
        return value

    def result(self, prepared, identity_override=None, claim_override=None,
               pair_override=None, selected_override=None):
        aees = ("1.2", "1.0", "0.9", "1.1")
        rows = []
        for index, candidate in enumerate(self.policy.candidates):
            checkpoint = prepared.snapshots[candidate.candidate_id + ":checkpoint"]
            config = prepared.snapshots["config:" + candidate.config_key]
            profile = prepared.snapshots[candidate.candidate_id + ":profile"]
            if identity_override and candidate.candidate_id in identity_override:
                checkpoint = identity_override[candidate.candidate_id]
            candidate_id, epoch = candidate.candidate_id, candidate.epoch
            if pair_override and index == 0:
                candidate_id, epoch = pair_override
            metrics = {key: (aees[index] if key == "AEE" else "0.1")
                       for key in M.ERROR_METRIC_KEYS}
            rows.append({
                "candidate_id": candidate_id, "epoch": epoch,
                "run_directory": str(self.repo / candidate.run_rel),
                "checkpoint": self.identity(checkpoint),
                "configuration": self.identity(config),
                "profile": self.identity(profile, profile=True),
                "accuracy_metrics": metrics,
                "activity": {"total_spikes": 1, "global_firing_rate": 0.1,
                             "dense_flops": 10.0, "effective_flops": 1.0,
                             "effective_sparsity": 0.9,
                             "spike_energy_proxy_uj": 1.0,
                             "energy_scope": "spike_activity_proxy_not_hardware_energy"},
            })
        winner = rows[2]
        selected = {key: winner[key] for key in (
            "candidate_id", "epoch", "run_directory", "checkpoint", "configuration",
            "profile", "accuracy_metrics", "activity")}
        if selected_override:
            selected.update(selected_override)
        claim = dict(M.EXACT_CLAIM_BOUNDARY)
        if claim_override:
            claim.update(claim_override)
        return {
            "schema": M.RESULT_SCHEMA, "status": M.RESULT_STATUS,
            "new_run_manifest": self.identity(prepared.snapshots["manifest"]),
            "candidate_population": rows,
            "selection_rule": {
                "candidate_ids": list(M.EXACT_PAIRS), "epochs": list(M.EXACT_PAIRS.values()),
                "primary": "minimum finite nonnegative standard-valid825 AEE",
                "tie_break": "lowest epoch", "all_four_candidates_required": True,
                "cross_run": True, "cross_config": True,
                "profile_hash_and_parse_same_immutable_bytes": True,
            },
            "selected": selected,
            "e0_e8_activation_dependent_invalidation_and_rebind_targets": ["E0"],
            "claim_boundary": claim,
        }

    def publish(self, prepared, **kwargs):
        result = self.result(prepared, **kwargs)
        self.output.mkdir()
        payloads = {}
        payloads["final_checkpoint_selection.json"] = json.dumps(
            result, sort_keys=True) + "\n"
        payloads["selected_checkpoint_and_config.json"] = json.dumps({
            "schema": "m1234_selected_checkpoint_and_config_r1_v1",
            **{key: result["selected"][key] for key in (
                "candidate_id", "epoch", "run_directory", "checkpoint",
                "configuration", "profile")}}, sort_keys=True) + "\n"
        payloads["e0_e8_activation_rebind_targets.json"] = json.dumps(
            result["e0_e8_activation_dependent_invalidation_and_rebind_targets"]) + "\n"
        payloads["RUN_COMPLETE.txt"] = M.RUN_COMPLETE
        csv_path = self.output / "four_checkpoint_metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["candidate_id", "epoch", "config_sha256", "checkpoint_sha256",
                             "profile_sha256", "samples", *M.ERROR_METRIC_KEYS])
            for row in result["candidate_population"]:
                writer.writerow([row["candidate_id"], row["epoch"],
                                 row["configuration"]["sha256"],
                                 row["checkpoint"]["sha256"], row["profile"]["sha256"], 825,
                                 *[row["accuracy_metrics"][key] for key in M.ERROR_METRIC_KEYS]])
        for name, value in payloads.items():
            (self.output / name).write_text(value, encoding="utf-8")
        manifest = self.output / M.MANIFEST
        manifest.write_text("".join(
            "{}  {}\n".format(sha(self.output / name), name)
            for name in sorted(M.RESULT_PAYLOADS)), encoding="utf-8")
        (self.output / M.OUTER).write_text(
            "{}  {}\n".format(sha(manifest), M.MANIFEST), encoding="utf-8")
        return result

    def good_runner(self, command, cwd, pass_fds):
        self.assertEqual(cwd, self.repo)
        self.assertEqual(tuple(int(value) for value in command[-4:-1]), pass_fds)
        for descriptor, relative in zip(pass_fds, (M.M1241_SOURCE_REL, M.M1234_SOURCE_REL,
                                                   M.M1228_SOURCE_REL)):
            seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
            required = (fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW |
                        fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
            self.assertEqual(seals & required, required)
            os.lseek(descriptor, 0, os.SEEK_SET)
            self.assertEqual(os.read(descriptor, 1 << 20), self.execution_payloads[relative])
        prepared = self.current_prepared
        self.publish(prepared)
        return subprocess.CompletedProcess(command, 0, stdout=M.CHILD_TOKEN + "\n", stderr="")

    def test_01_import_is_inert(self):
        code = ("import importlib.util,sys;"
                "s=importlib.util.spec_from_file_location('isolated_m1253',{!r});"
                "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;"
                "s.loader.exec_module(m);print('PASS')").format(str(SOURCE))
        self.assertEqual(subprocess.check_output(
            ["/usr/bin/python3.12", "-c", code]).decode().strip(), "PASS")
        self.assertFalse(self.attempt.exists())

    def test_02_exact_eleven_snapshot_and_sealed_execution_sources(self):
        prepared = M.prepare(self.policy, self.interpreter, "3.10.20", self.repo)
        try:
            self.assertEqual(len(prepared.snapshots), 11)
            self.assertFalse(self.attempt.exists())
            for descriptor in prepared.source_fds:
                with self.assertRaises(PermissionError):
                    os.write(descriptor, b"drift")
        finally:
            prepared.close()

    def test_03_every_candidate_artifact_missing_fails_before_attempt(self):
        for path in M.artifact_map(self.policy).values():
            with self.subTest(path=str(path)):
                body = path.read_bytes()
                path.unlink()
                with self.assertRaises(M.ReleaseError):
                    M.prepare(self.policy, self.interpreter, "3.10.20", self.repo)
                self.assertFalse(self.attempt.exists())
                path.write_bytes(body)

    def test_04_source_path_drift_after_prepare_cannot_change_sealed_child_bytes(self):
        prepared = M.prepare(self.policy, self.interpreter, "3.10.20", self.repo)
        self.current_prepared = prepared
        source = self.repo / M.M1241_SOURCE_REL
        source.write_text("malicious replacement\n", encoding="utf-8")
        try:
            M.consume_attempt(prepared)
            completed = self.good_runner(prepared.command, self.repo, prepared.source_fds)
            M.publish_log(prepared, completed)
            M.verify_receipt(self.output, prepared)
        finally:
            prepared.close()
        self.assertTrue(self.attempt.exists())

    def test_05_candidate_drift_after_prepare_is_rejected_against_snapshot(self):
        prepared = M.prepare(self.policy, self.interpreter, "3.10.20", self.repo)
        try:
            target = self.repo / self.policy.candidates[0].run_rel / "checkpoint_epoch29.pth"
            target.write_bytes(b"replacement\n")
            replacement, _ = M.snapshot_file(target, "replacement")
            self.publish(prepared, identity_override={"legacy_ep29": replacement})
            with self.assertRaisesRegex(M.ReleaseError, "checkpoint sha256 mismatch"):
                M.verify_receipt(self.output, prepared)
        finally:
            prepared.close()

    def test_06_extra_or_positive_claim_boundary_is_rejected(self):
        attacks = ({"paper_metric": True}, {"paper_metric": False},
                   {"power_or_energy": True}, {"hardware_replay_complete": True})
        for attack in attacks:
            with self.subTest(attack=attack):
                prepared = M.prepare(self.policy, self.interpreter, "3.10.20", self.repo)
                try:
                    self.publish(prepared, claim_override=attack)
                    with self.assertRaisesRegex(M.ReleaseError, "exact claim boundary"):
                        M.verify_receipt(self.output, prepared)
                finally:
                    prepared.close()
                for child in self.output.iterdir():
                    child.unlink()
                self.output.rmdir()

    def test_07_candidate_epoch_pair_and_nonminimum_selection_are_rejected(self):
        prepared = M.prepare(self.policy, self.interpreter, "3.10.20", self.repo)
        try:
            self.publish(prepared, pair_override=("legacy_ep29", 34))
            with self.assertRaisesRegex(M.ReleaseError, "candidate pair mismatch"):
                M.verify_receipt(self.output, prepared)
        finally:
            prepared.close()
        for child in self.output.iterdir():
            child.unlink()
        self.output.rmdir()
        prepared = M.prepare(self.policy, self.interpreter, "3.10.20", self.repo)
        try:
            self.publish(prepared, selected_override={"candidate_id": "resume_ep30", "epoch": 30})
            with self.assertRaisesRegex(M.ReleaseError, "exact minimum-AEE"):
                M.verify_receipt(self.output, prepared)
        finally:
            prepared.close()

    def test_08_success_consumes_once_and_full_receipt_rebinds(self):
        original_prepare = M.prepare
        def capture(*args, **kwargs):
            result = original_prepare(*args, **kwargs)
            self.current_prepared = result
            return result
        M.prepare = capture
        try:
            completed = M.execute_once(self.policy, self.interpreter, "3.10.20", self.repo,
                                       self.good_runner)
        finally:
            M.prepare = original_prepare
        self.assertEqual(completed.returncode, 0)
        self.assertTrue(self.attempt.is_file())
        self.assertTrue(self.log.is_file())
        self.assertIn("automatic_retry=false", self.attempt.read_text())

    def test_09_namespace_race_and_child_failure_preserve_attempt(self):
        prepared = M.prepare(self.policy, self.interpreter, "3.10.20", self.repo)
        prepared.close()
        self.attempt.write_text("racer\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            M.consume_attempt(prepared)
        self.assertEqual(self.attempt.read_text(), "racer\n")

    def test_10_child_failure_consumes_attempt_and_forbids_retry(self):
        calls = []
        def failed(command, cwd, pass_fds):
            calls.append(tuple(command))
            return subprocess.CompletedProcess(command, 9, stdout="", stderr="failed")
        with self.assertRaisesRegex(M.ReleaseError, "no retry authorized"):
            M.execute_once(self.policy, self.interpreter, "3.10.20", self.repo, failed)
        self.assertEqual(len(calls), 1)
        self.assertTrue(self.attempt.exists())
        self.assertTrue(self.log.exists())
        with self.assertRaises(M.ReleaseError):
            M.execute_once(self.policy, self.interpreter, "3.10.20", self.repo, failed)
        self.assertEqual(len(calls), 1)

    def test_11_seal_schema_status_and_selected_sidecar_are_fail_closed(self):
        prepared = M.prepare(self.policy, self.interpreter, "3.10.20", self.repo)
        try:
            self.publish(prepared)
            sidecar = self.output / "selected_checkpoint_and_config.json"
            sidecar.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(M.ReleaseError):
                M.verify_receipt(self.output, prepared)
        finally:
            prepared.close()

    def test_12_sealed_launcher_executes_only_passed_dependency_fds(self):
        m1241 = b'''from dataclasses import dataclass\nfrom pathlib import Path\n@dataclass(frozen=True)\nclass Frozen:\n    public_identity: dict\n    physical_identity: tuple\ndef freeze_file(*args, **kwargs):\n    return Frozen({}, (11, 22, 0))\ndef load_predecessor():\n    raise RuntimeError("unpatched")\ndef build(policy):\n    assert load_predecessor().load_predecessor().MARK == "sealed-m1228"\n    frozen = freeze_file()\n    assert frozen.public_identity == {"device": 11, "inode": 22}\n    return {"selected": {"candidate_id": "resume_ep32", "epoch": 32}}\ndef write_receipt(path, result):\n    path.mkdir()\n    (path / "proof.txt").write_text("sealed launcher\\n")\n'''
        m1234 = b'''PRODUCTION_POLICY = object()\ndef load_predecessor():\n    raise RuntimeError("unpatched")\n'''
        m1228 = b'''MARK = "sealed-m1228"\n'''
        descriptors = tuple(M.make_sealed_memfd(name, payload) for name, payload in (
            ("dummy-m1241", m1241), ("dummy-m1234", m1234), ("dummy-m1228", m1228)))
        output = Path(self.tmp.name) / "sealed-launcher-output"
        command = ["/usr/bin/python3.12", "-I", "-B", "-c", M.SEALED_LAUNCHER,
                   *(str(value) for value in descriptors), str(output)]
        try:
            completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, pass_fds=descriptors, check=False)
        finally:
            for descriptor in descriptors:
                os.close(descriptor)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.count(M.CHILD_TOKEN), 1)
        self.assertEqual((output / "proof.txt").read_text(), "sealed launcher\n")

    def test_13_source_only_contract_and_docs359_frozen(self):
        text = SOURCE.read_text(encoding="utf-8")
        for forbidden in ("import torch", "import cupy", "paramiko", "dc_shell",
                          "vcs -full64", "nvidia-smi"):
            self.assertNotIn(forbidden, text)
        docs = Path(__file__).resolve().parents[1] / "docs/359_DATE终局冻结_20260813.md"
        self.assertEqual(sha(docs), M.DOCS359_SHA256)
        if CONTRACT.exists():
            contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
            self.assertEqual(contract["source"]["sha256"], sha(SOURCE))
            self.assertEqual(contract["test"]["sha256"], sha(Path(__file__).resolve()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
