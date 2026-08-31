from __future__ import annotations

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
          "run_m1257_m1253_motion_cross_run_final_checkpoint_binder_one_shot_release_successor.py")
CONTRACT = (Path(__file__).resolve().parents[1] / "contracts" /
            "m1257_m1253_motion_cross_run_final_checkpoint_binder_one_shot_release_successor_source_contract_r1_20260830.json")
M1253_TEST = (Path(__file__).resolve().parent /
              "test_run_m1253_m1248_motion_cross_run_final_checkpoint_binder_one_shot_release_successor.py")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


S = load("m1257_source_under_test", SOURCE)
T = load("m1257_m1253_fixture", M1253_TEST)


class M1257ReleaseSuccessorTest(unittest.TestCase):
    def setUp(self):
        self.fx = T.M1253ReleaseSuccessorTest(methodName="test_01_import_is_inert")
        self.fx.setUp()
        self.fx.identity = self.identity
        self.review_rel = Path("hw_autoresearch_nts07/reviews/m1255_fixture")
        root = self.fx.repo / self.review_rel
        root.mkdir(parents=True)
        review = {
            "schema": S.M1255_SCHEMA, "status": S.M1255_STATUS,
            "authority": {
                "production_execution_authorized_now": False,
                "future_execution_authorized_by_M1255": False,
                "release_successor_authoring_required": True,
                "fresh_different_author_successor_hammer_required": True,
            },
        }
        (root / "review.json").write_text(json.dumps(review) + "\n", encoding="utf-8")
        manifest = root / S.B.MANIFEST
        manifest.write_text("{}  review.json\n".format(sha(root / "review.json")),
                            encoding="utf-8")
        outer = root / S.B.OUTER
        outer.write_text("{}  {}\n".format(sha(manifest), S.B.MANIFEST), encoding="utf-8")
        self.policy = S.Policy(self.fx.policy, self.review_rel, sha(manifest), sha(outer))

    def tearDown(self):
        self.fx.tearDown()

    def identity(self, snapshot, profile=False):
        value = S._snapshot_identity(snapshot)
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

    @property
    def output(self):
        return self.fx.output

    def reseal(self):
        manifest = self.output / S.B.MANIFEST
        manifest.write_text("".join(
            "{}  {}\n".format(sha(self.output / name), name)
            for name in sorted(S.B.RESULT_PAYLOADS)), encoding="utf-8")
        (self.output / S.B.OUTER).write_text(
            "{}  {}\n".format(sha(manifest), S.B.MANIFEST), encoding="utf-8")

    def publish(self, prepared, **kwargs):
        result = self.fx.publish(prepared, **kwargs)
        result["e0_e8_activation_dependent_invalidation_and_rebind_targets"] = (
            S.exact_rebind_targets())
        (self.output / "final_checkpoint_selection.json").write_text(
            json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
        (self.output / "e0_e8_activation_rebind_targets.json").write_text(
            json.dumps(S.exact_rebind_targets(), sort_keys=True) + "\n", encoding="utf-8")
        self.reseal()
        return result

    def mutate_result(self, callback):
        path = self.output / "final_checkpoint_selection.json"
        value = json.loads(path.read_text())
        callback(value)
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        self.reseal()

    def good_runner(self, command, cwd, pass_fds):
        self.assertEqual(cwd, self.fx.repo)
        self.assertEqual(tuple(int(value) for value in command[-4:-1]), pass_fds)
        required = (fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW |
                    fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
        for descriptor in pass_fds:
            self.assertEqual(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & required, required)
        self.publish(self.current_prepared)
        return subprocess.CompletedProcess(command, 0, stdout=S.CHILD_TOKEN + "\n", stderr="")

    def test_01_import_is_inert_and_predecessor_is_exact(self):
        code = ("import importlib.util,sys;"
                "s=importlib.util.spec_from_file_location('isolated_m1257',{!r});"
                "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;"
                "s.loader.exec_module(m);print('PASS')").format(str(SOURCE))
        self.assertEqual(subprocess.check_output(
            ["/usr/bin/python3.12", "-c", code]).decode().strip(), "PASS")
        self.assertEqual(sha(S.BASE_SOURCE), S.BASE_SOURCE_SHA256)
        self.assertFalse(self.fx.attempt.exists())

    def test_02_prepare_keeps_eleven_inputs_three_sealed_memfds_and_exact_pass_fds(self):
        prepared = S.prepare(self.policy, self.fx.interpreter, "3.10.20", self.fx.repo)
        try:
            self.assertEqual(len(prepared.snapshots), 11)
            self.assertEqual(len(prepared.source_fds), 3)
            self.assertEqual(tuple(int(value) for value in prepared.command[-4:-1]),
                             prepared.source_fds)
            required = (fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW |
                        fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
            for descriptor in prepared.source_fds:
                self.assertEqual(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & required,
                                 required)
        finally:
            prepared.close()

    def test_03_success_rebinds_mode_everywhere_and_full_receipt(self):
        original = S.prepare
        def capture(*args, **kwargs):
            prepared = original(*args, **kwargs)
            self.current_prepared = prepared
            return prepared
        S.prepare = capture
        try:
            completed = S.execute_once(self.policy, self.fx.interpreter, "3.10.20",
                                       self.fx.repo, self.good_runner)
        finally:
            S.prepare = original
        self.assertEqual(completed.returncode, 0)
        result = json.loads((self.output / "final_checkpoint_selection.json").read_text())
        for row in result["candidate_population"]:
            for key in ("checkpoint", "configuration", "profile"):
                self.assertIn("mode", row[key])
        self.assertIn("mode", result["selected"]["checkpoint"])
        self.assertIn("mode", result["new_run_manifest"])

    def test_04_post_prepare_mode_drift_in_child_receipt_is_rejected(self):
        prepared = S.prepare(self.policy, self.fx.interpreter, "3.10.20", self.fx.repo)
        try:
            target = Path(prepared.snapshots["legacy_ep29:checkpoint"].absolute_path)
            os.chmod(target, 0o600)
            replacement, _ = S.B.snapshot_file(target, "replacement")
            self.publish(prepared, identity_override={"legacy_ep29": replacement})
            with self.assertRaisesRegex(S.B.ReleaseError, "exact identity mismatch"):
                S.verify_receipt(self.output, prepared)
        finally:
            prepared.close()

    def test_05_missing_mode_and_nested_extra_false_claim_are_rejected(self):
        for attack in ("missing_mode", "extra_false"):
            with self.subTest(attack=attack):
                prepared = S.prepare(self.policy, self.fx.interpreter, "3.10.20", self.fx.repo)
                try:
                    self.publish(prepared)
                    def mutation(result):
                        identity = result["candidate_population"][0]["checkpoint"]
                        if attack == "missing_mode":
                            identity.pop("mode")
                        else:
                            identity["paper_metric"] = False
                    self.mutate_result(mutation)
                    with self.assertRaisesRegex(S.B.ReleaseError, "exact identity key"):
                        S.verify_receipt(self.output, prepared)
                finally:
                    prepared.close()
                for child in self.output.iterdir():
                    child.unlink()
                self.output.rmdir()

    def test_06_result_root_extra_false_and_positive_claims_are_rejected(self):
        prepared = S.prepare(self.policy, self.fx.interpreter, "3.10.20", self.fx.repo)
        try:
            self.publish(prepared)
            self.mutate_result(lambda result: result.update(
                paper_metric=False, hardware_speedup=True))
            with self.assertRaisesRegex(S.B.ReleaseError, "exact root key"):
                S.verify_receipt(self.output, prepared)
        finally:
            prepared.close()

    def test_07_spliced_result_and_sidecar_e0_e8_are_rejected_against_policy(self):
        prepared = S.prepare(self.policy, self.fx.interpreter, "3.10.20", self.fx.repo)
        try:
            self.publish(prepared)
            spliced = [{"id": "E0", "target": "spliced", "state_after_selection": "DONE"}]
            self.mutate_result(lambda result: result.update(
                e0_e8_activation_dependent_invalidation_and_rebind_targets=spliced))
            (self.output / "e0_e8_activation_rebind_targets.json").write_text(
                json.dumps(spliced) + "\n", encoding="utf-8")
            self.reseal()
            with self.assertRaisesRegex(S.B.ReleaseError, "E0-E8 exact map"):
                S.verify_receipt(self.output, prepared)
        finally:
            prepared.close()

    def test_08_profile_and_activity_extra_keys_and_types_are_rejected(self):
        for attack in ("profile_extra", "activity_extra", "activity_bool"):
            with self.subTest(attack=attack):
                prepared = S.prepare(self.policy, self.fx.interpreter, "3.10.20", self.fx.repo)
                try:
                    self.publish(prepared)
                    def mutation(result):
                        row = result["candidate_population"][0]
                        if attack == "profile_extra":
                            row["profile"]["extra"] = False
                        elif attack == "activity_extra":
                            row["activity"]["hardware_energy"] = False
                        else:
                            row["activity"]["global_firing_rate"] = True
                    self.mutate_result(mutation)
                    with self.assertRaises(S.B.ReleaseError):
                        S.verify_receipt(self.output, prepared)
                finally:
                    prepared.close()
                for child in self.output.iterdir():
                    child.unlink()
                self.output.rmdir()

    def test_09_candidate_pair_and_nonminimum_projection_are_rejected(self):
        prepared = S.prepare(self.policy, self.fx.interpreter, "3.10.20", self.fx.repo)
        try:
            self.publish(prepared, pair_override=("legacy_ep29", 34))
            with self.assertRaises(S.B.ReleaseError):
                S.verify_receipt(self.output, prepared)
        finally:
            prepared.close()

    def test_09b_nonminimum_projection_and_wrong_tie_break_are_rejected(self):
        for attack in ("nonminimum", "wrong_tie"):
            with self.subTest(attack=attack):
                prepared = S.prepare(self.policy, self.fx.interpreter, "3.10.20", self.fx.repo)
                try:
                    if attack == "nonminimum":
                        self.publish(prepared, selected_override={
                            "candidate_id": "resume_ep30", "epoch": 30})
                    else:
                        self.publish(prepared)
                        def tie(result):
                            result["candidate_population"][1]["accuracy_metrics"]["AEE"] = "0.9"
                        self.mutate_result(tie)
                    with self.assertRaisesRegex(S.B.ReleaseError, "selected projection"):
                        S.verify_receipt(self.output, prepared)
                finally:
                    prepared.close()
                for child in self.output.iterdir():
                    child.unlink()
                self.output.rmdir()

    def test_10_selected_sidecar_and_csv_splices_are_rejected(self):
        prepared = S.prepare(self.policy, self.fx.interpreter, "3.10.20", self.fx.repo)
        try:
            self.publish(prepared)
            sidecar = self.output / "selected_checkpoint_and_config.json"
            value = json.loads(sidecar.read_text())
            value["paper_metric"] = False
            sidecar.write_text(json.dumps(value) + "\n", encoding="utf-8")
            self.reseal()
            with self.assertRaisesRegex(S.B.ReleaseError, "selected sidecar"):
                S.verify_receipt(self.output, prepared)
        finally:
            prepared.close()

    def test_11_sealed_launcher_publishes_full_mode(self):
        m1241 = b'''from dataclasses import dataclass\nfrom pathlib import Path\n@dataclass(frozen=True)\nclass Frozen:\n    public_identity: dict\n    pathname_identity: tuple\ndef freeze_file(*args, **kwargs):\n    return Frozen({}, (11,22,33188,7,8))\ndef load_predecessor(): raise RuntimeError("unpatched")\ndef build(policy):\n    frozen=freeze_file()\n    assert frozen.public_identity=={"device":11,"inode":22,"mode":33188}\n    return {"selected":{"candidate_id":"resume_ep32","epoch":32}}\ndef write_receipt(path,result):\n    path.mkdir();(path/"proof.txt").write_text("mode sealed\\n")\n'''
        m1234 = b'''PRODUCTION_POLICY=object()\ndef load_predecessor(): raise RuntimeError("unpatched")\n'''
        m1228 = b'''MARK="sealed"\n'''
        fds = tuple(S.B.make_sealed_memfd(name, payload) for name, payload in (
            ("m1241", m1241), ("m1234", m1234), ("m1228", m1228)))
        output = Path(self.fx.tmp.name) / "launcher"
        command = ["/usr/bin/python3.12", "-I", "-B", "-c", S.SEALED_LAUNCHER,
                   *(str(fd) for fd in fds), str(output)]
        try:
            completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, pass_fds=fds)
        finally:
            for fd in fds:
                os.close(fd)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.count(S.CHILD_TOKEN), 1)

    def test_12_o_excl_and_no_retry_remain_fail_closed(self):
        calls = []
        def failed(command, cwd, pass_fds):
            calls.append(tuple(command))
            return subprocess.CompletedProcess(command, 9, stdout="", stderr="failed")
        with self.assertRaisesRegex(S.B.ReleaseError, "no retry"):
            S.execute_once(self.policy, self.fx.interpreter, "3.10.20", self.fx.repo, failed)
        self.assertEqual(len(calls), 1)
        self.assertTrue(self.fx.attempt.exists())
        self.assertTrue(self.fx.log.exists())
        with self.assertRaises(S.B.ReleaseError):
            S.execute_once(self.policy, self.fx.interpreter, "3.10.20", self.fx.repo, failed)
        self.assertEqual(len(calls), 1)

    def test_13_source_only_contract_and_docs359_frozen(self):
        text = SOURCE.read_text()
        for forbidden in ("import torch", "import cupy", "paramiko", "dc_shell",
                          "vcs -full64", "nvidia-smi"):
            self.assertNotIn(forbidden, text)
        docs = Path(__file__).resolve().parents[1] / "docs/359_DATE终局冻结_20260813.md"
        self.assertEqual(sha(docs), S.B.DOCS359_SHA256)
        if CONTRACT.exists():
            contract = json.loads(CONTRACT.read_text())
            self.assertEqual(contract["source"]["sha256"], sha(SOURCE))
            self.assertEqual(contract["test"]["sha256"], sha(Path(__file__).resolve()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
