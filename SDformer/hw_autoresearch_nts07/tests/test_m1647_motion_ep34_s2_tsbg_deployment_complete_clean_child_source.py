#!/usr/bin/env python3
"""Source-only tests for the M1647 deployment-complete successor."""
from __future__ import print_function

import copy
from io import BytesIO
import importlib.util
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[2] / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1647_motion_ep34_s2_tsbg_deployment_complete_clean_child_"
    "successor_r1.py")
SPEC = importlib.util.spec_from_file_location("m1647_source", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M1647SourceTest(unittest.TestCase):
    def test_01_actual_deployment_closure_passes_before_runtime(self):
        row = M.verify_deployment_completeness()
        self.assertEqual(row["runtime_predecessor_seals"], 16)
        self.assertEqual(row["sealed_members"], 116)
        self.assertEqual(row["current_archive_missing_members"], 0)
        self.assertEqual(row["parent_processes"], 0)
        self.assertEqual(row["gpu_runs"], 0)

    def test_02_m1314_author_log_is_exact_and_in_current_git_archive(self):
        self.assertEqual(M.sha256(M.M1314_AUTHOR_TEST),
                         M.M1314_AUTHOR_TEST_SHA256)
        relative = str(M.M1314_AUTHOR_TEST.relative_to(M.ROOT))
        archive = subprocess.check_output(
            ["git", "archive", "--format=tar", "HEAD", relative],
            cwd=str(M.ROOT))
        with tarfile.open(fileobj=BytesIO(archive), mode="r:") as stream:
            names = set(stream.getnames())
        self.assertIn(relative, names)
        self.assertTrue(M.validate_archive_member_inventory(names)[
            "m1314_author_test_present"])

    def test_03_previous_failure_and_current_inventory_are_distinct(self):
        value = M.strict_json(M.DEPLOYMENT_MANIFEST)
        previous = value[
            "failed_pre_attempt_archive_missing_runtime_required_members"]
        self.assertEqual(len(previous), 1)
        self.assertEqual(previous[0]["sha256"], M.M1314_AUTHOR_TEST_SHA256)
        self.assertEqual(value[
            "current_git_archive_missing_runtime_required_members"], [])

    def test_04_archive_inventory_rejects_old_missing_state(self):
        with self.assertRaises(M.M1647Error):
            M.validate_archive_member_inventory(set())

    def test_05_all_sealed_members_are_actually_rehashed(self):
        original = M.regular_exact
        labels = []
        def counted(path, digest, label):
            labels.append(label)
            return original(path, digest, label)
        M.regular_exact = counted
        try:
            M.verify_deployment_completeness()
        finally:
            M.regular_exact = original
        self.assertGreaterEqual(len(labels), 150)
        self.assertEqual(sum(" member " in label for label in labels), 116)

    def test_06_regular_exact_rejects_missing_mismatch_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "good"
            good.write_bytes(b"good")
            with self.assertRaises(M.M1647Error):
                M.regular_exact(root / "missing", M.sha256(good), "missing")
            with self.assertRaises(M.M1647Error):
                M.regular_exact(good, "0" * 64, "mismatch")
            link = root / "link"
            link.symlink_to(good)
            with self.assertRaises(M.M1647Error):
                M.regular_exact(link, M.sha256(good), "symlink")

    def test_07_parent_preflight_precedes_exact_chain_and_subprocess_budget(self):
        events = []
        originals = (M.verify_deployment_completeness,
                     M.verify_exact_m1624_m1640, M.P.launch_parent)
        M.verify_deployment_completeness = lambda: events.append("deploy")
        M.verify_exact_m1624_m1640 = lambda: events.append("exact")
        M.P.launch_parent = lambda: events.append("parent") or 0
        try:
            self.assertEqual(M.launch_parent(), 0)
        finally:
            (M.verify_deployment_completeness,
             M.verify_exact_m1624_m1640, M.P.launch_parent) = originals
        self.assertEqual(events, ["deploy", "exact", "parent"])

    def test_08_child_preflight_precedes_exact_chain_and_child_budget(self):
        events = []
        originals = (M.verify_deployment_completeness,
                     M.verify_exact_m1624_m1640, M.P.fixed_clean_child)
        M.verify_deployment_completeness = lambda: events.append("deploy")
        M.verify_exact_m1624_m1640 = lambda: events.append("exact")
        M.P.fixed_clean_child = lambda: events.append("child") or 0
        try:
            self.assertEqual(M.fixed_clean_child(), 0)
        finally:
            (M.verify_deployment_completeness,
             M.verify_exact_m1624_m1640, M.P.fixed_clean_child) = originals
        self.assertEqual(events, ["deploy", "exact", "child"])

    def test_09_failed_parent_preflight_cannot_reach_subprocess_budget(self):
        reached = []
        originals = M.verify_deployment_completeness, M.P.launch_parent
        def fail():
            raise M.M1647Error("deployment incomplete")
        M.verify_deployment_completeness = fail
        M.P.launch_parent = lambda: reached.append(True)
        try:
            with self.assertRaises(M.M1647Error):
                M.launch_parent()
        finally:
            M.verify_deployment_completeness, M.P.launch_parent = originals
        self.assertEqual(reached, [])

    def test_10_exact_m1624_m1640_chain_passes(self):
        review = M.verify_exact_m1624_m1640()
        self.assertEqual(review["p0_count"], 0)
        self.assertEqual(review["p1_count"], 0)
        self.assertFalse(review["authorization"]["automatic_retry"])

    def test_11_binding_changes_only_named_surface_and_restores(self):
        names = ("SOURCE", "TEST", "SOURCE_CONTRACT", "FUTURE_REVIEW",
                 "FUTURE_RELEASE", "RESULT", "ATTEMPT", "WORK", "FAILURE",
                 "validate_source_contract", "validate_future_authorities",
                 "require_fresh_namespaces", "write_child_receipt")
        before = dict((name, getattr(M.P, name)) for name in names)
        with M._bound_exact_m1624():
            self.assertEqual(M.P.RESULT, M.RESULT)
            self.assertEqual(M.P.ATTEMPT, M.ATTEMPT)
            self.assertIs(M.P.validate_future_authorities,
                          M.validate_future_authorities)
        self.assertTrue(all(getattr(M.P, name) is before[name]
                            if callable(before[name]) else
                            getattr(M.P, name) == before[name]
                            for name in names))

    def test_12_new_namespaces_are_fresh_distinct_and_not_m1624(self):
        M.require_fresh_namespaces()
        paths = (M.RESULT, M.ATTEMPT, M.WORK, M.FAILURE)
        self.assertEqual(len(set(paths)), 4)
        self.assertTrue(all("m1647_" in path.name for path in paths))
        self.assertTrue(all(path not in (M.P.RESULT, M.P.ATTEMPT,
                                        M.P.WORK, M.P.FAILURE)
                            for path in paths))

    def test_13_source_self_check_is_inert_and_does_not_call_m1624_runtime(self):
        original = M.P.verify_fixed_metadata
        M.P.verify_fixed_metadata = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("M1624 runtime metadata must not run in source stage"))
        try:
            row = M.source_self_check()
        finally:
            M.P.verify_fixed_metadata = original
        self.assertEqual(row["child_processes"], 0)
        self.assertEqual(row["attempt_writes"], 0)
        self.assertFalse(row["payload_opened"])
        self.assertFalse(row["gpu_runs"])

    def test_14_future_review_release_and_all_namespaces_are_absent(self):
        self.assertFalse(M.FUTURE_REVIEW.exists())
        self.assertFalse(M.FUTURE_RELEASE.exists())
        self.assertTrue(all(not path.exists() for path in
                            (M.RESULT, M.ATTEMPT, M.WORK, M.FAILURE)))

    def test_15_contract_keeps_one_shot_and_all_claims_closed(self):
        value = M.strict_json(M.SOURCE_CONTRACT)
        future = value["future_release_shape"]
        self.assertEqual(future["parent_calls"], 1)
        self.assertEqual(future["clean_child_processes"], 1)
        self.assertEqual(future["gpu_runs"], 1)
        self.assertEqual(future["production_captures"], 1)
        self.assertFalse(future["automatic_retry"])
        claims = value["claim_boundary"]
        self.assertTrue(claims["source_only"])
        self.assertTrue(all(claims[key] is False for key in claims
                            if key != "source_only"))

    def test_16_source_contains_no_remote_client_or_payload_read(self):
        text = SOURCE.read_text(encoding="utf-8")
        for token in ("paramiko", "ssh ", "scp ", "rsync ", "np.load(",
                      "torch.load(", "requests."):
            self.assertNotIn(token, text)
        self.assertIn("verify_deployment_completeness()", text)


if __name__ == "__main__":
    unittest.main()
