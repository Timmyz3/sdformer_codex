#!/usr/bin/env python3
"""Source-only tests for the M1668 runtime/entity-closed successor."""
from __future__ import print_function

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[2] / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1668_motion_ep34_s2_tsbg_runtime_closed_entity_rebind_"
    "successor_r1.py")
SPEC = importlib.util.spec_from_file_location("m1668_source", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M1668SourceTest(unittest.TestCase):
    def test_01_new_selection_identity_is_content_identical_entity_rebind(self):
        value = M.selection_identity()
        old = value["configuration_frozen_selection_entity"]
        new = value["configuration_current_capture_entity"]
        self.assertEqual(old["absolute_path"], new["absolute_path"])
        self.assertEqual(old["sha256"], new["sha256"])
        self.assertEqual(old["size_bytes"], new["size_bytes"])
        self.assertNotEqual((old["inode"], old["mode"], old["mtime_ns"]),
                            (new["inode"], new["mode"], new["mtime_ns"]))
        self.assertFalse(value["selection_semantics"]["checkpoint_reselected"])

    def test_02_runtime_handoff_closes_exact_m1257_payload(self):
        row = M.verify_runtime_handoff_source()
        self.assertEqual(row["archive_files"], 9)
        self.assertEqual(row["canonical_files"], 7)
        self.assertEqual(row["attempt"], 1)
        self.assertEqual(row["launch_log"], 1)

    def test_03_runtime_handoff_missing_tar_fails_closed(self):
        original = M.RUNTIME_TAR
        M.RUNTIME_TAR = original.with_name("absent_m1668_runtime.tar")
        try:
            with self.assertRaises(M.M1668Error):
                M.verify_runtime_handoff_source()
        finally:
            M.RUNTIME_TAR = original

    def test_04_verify_entity_accepts_exact_and_rejects_stat_or_sha_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "entity"
            path.write_bytes(b"m1668")
            st = path.lstat()
            exact = {"absolute_path": str(path), "device": st.st_dev,
                     "inode": st.st_ino, "mode": st.st_mode,
                     "mtime_ns": st.st_mtime_ns, "sha256": M.sha256(path),
                     "size_bytes": st.st_size}
            self.assertEqual(M.verify_entity(path, exact, "temporary"), exact)
            bad = dict(exact, inode=st.st_ino + 1)
            with self.assertRaises(M.M1668Error):
                M.verify_entity(path, bad, "temporary")
            bad = dict(exact, sha256="0" * 64)
            with self.assertRaises(M.M1668Error):
                M.verify_entity(path, bad, "temporary")

    def test_05_dual_identity_rebind_is_narrow_and_restored(self):
        def extended(value, label):
            return value
        def legacy(value, label):
            return value
        frozen = SimpleNamespace(exact_identity=legacy)
        m1319 = SimpleNamespace(exact_extended_identity=extended,
                                FROZEN_M1233=frozen)
        m1434 = SimpleNamespace(M1349=SimpleNamespace(
            M1327=SimpleNamespace(M1319=m1319)))
        old_extended = m1319.exact_extended_identity
        old_frozen = frozen.exact_identity
        identity = M.selection_identity()
        calls = []
        original_verify = M.verify_entity
        M.verify_entity = lambda path, expected, label: calls.append(label) or expected
        try:
            with M.current_configuration_entity_rebind(m1434, identity):
                self.assertIsNot(m1319.exact_extended_identity, old_extended)
                self.assertIsNot(frozen.exact_identity, old_frozen)
                selected = identity["configuration_frozen_selection_entity"]
                self.assertEqual(
                    m1319.exact_extended_identity(selected,
                                                   "selected configuration"),
                    selected)
                self.assertEqual(
                    frozen.exact_identity(selected, "selected configuration"),
                    selected)
        finally:
            M.verify_entity = original_verify
        self.assertIs(m1319.exact_extended_identity, old_extended)
        self.assertIs(frozen.exact_identity, old_frozen)
        self.assertEqual(calls, ["current selected configuration",
                                 "current nested selected configuration"])

    def test_06_nonconfiguration_identity_still_uses_original_verifier(self):
        calls = []
        def extended(value, label):
            calls.append(label)
            return value
        frozen = SimpleNamespace(exact_identity=lambda value, label: value)
        m1319 = SimpleNamespace(exact_extended_identity=extended,
                                FROZEN_M1233=frozen)
        m1434 = SimpleNamespace(M1349=SimpleNamespace(
            M1327=SimpleNamespace(M1319=m1319)))
        identity = M.selection_identity()
        original = m1319.exact_extended_identity
        original_verify = M.verify_entity
        M.verify_entity = lambda path, expected, label: expected
        try:
            with M.current_configuration_entity_rebind(m1434, identity):
                value = {"sentinel": 1}
                self.assertEqual(m1319.exact_extended_identity(value,
                                                                "selected checkpoint"),
                                 value)
        finally:
            M.verify_entity = original_verify
        self.assertIs(m1319.exact_extended_identity, original)
        self.assertEqual(calls, ["selected checkpoint"])

    def test_07_parent_build_runtime_preflight_precedes_subprocess_budget(self):
        events = []
        originals = (M.verify_predecessors, M.preflight_runtime_binding,
                     M.P.launch_parent)
        M.verify_predecessors = lambda: events.append("predecessors")
        M.preflight_runtime_binding = lambda: events.append("build_runtime")
        M.P.launch_parent = lambda: events.append("parent") or 0
        try:
            self.assertEqual(M.launch_parent(), 0)
        finally:
            (M.verify_predecessors, M.preflight_runtime_binding,
             M.P.launch_parent) = originals
        self.assertEqual(events, ["predecessors", "build_runtime", "parent"])

    def test_08_child_build_runtime_preflight_precedes_gpu_attempt_delegate(self):
        events = []
        originals = (M.verify_predecessors, M.preflight_runtime_binding,
                     M.P.fixed_clean_child)
        M.verify_predecessors = lambda: events.append("predecessors")
        M.preflight_runtime_binding = lambda: events.append("build_runtime")
        M.P.fixed_clean_child = lambda: events.append("child_delegate") or 0
        try:
            self.assertEqual(M.fixed_clean_child(), 0)
        finally:
            (M.verify_predecessors, M.preflight_runtime_binding,
             M.P.fixed_clean_child) = originals
        self.assertEqual(events,
                         ["predecessors", "build_runtime", "child_delegate"])

    def test_09_failed_build_runtime_blocks_parent_and_child_budget(self):
        reached = []
        originals = (M.verify_predecessors, M.preflight_runtime_binding,
                     M.P.launch_parent)
        M.verify_predecessors = lambda: None
        def fail():
            raise M.M1668Error("runtime unavailable")
        M.preflight_runtime_binding = fail
        M.P.launch_parent = lambda: reached.append(True)
        try:
            with self.assertRaises(M.M1668Error):
                M.launch_parent()
        finally:
            (M.verify_predecessors, M.preflight_runtime_binding,
             M.P.launch_parent) = originals
        self.assertEqual(reached, [])

    def test_10_bound_m1647_changes_only_successor_surface_and_restores(self):
        names = ("SOURCE", "TEST", "SOURCE_CONTRACT", "FUTURE_REVIEW",
                 "FUTURE_RELEASE", "RESULT", "ATTEMPT", "WORK", "FAILURE",
                 "validate_source_contract", "validate_future_authorities",
                 "require_fresh_namespaces", "write_child_receipt")
        before = dict((name, getattr(M.P, name)) for name in names)
        old_loader = M.P.P.load_m1434
        with M._bound_exact_m1647():
            self.assertEqual(M.P.RESULT, M.RESULT)
            self.assertIs(M.P.validate_future_authorities,
                          M.validate_future_authorities)
            self.assertIs(M.P.P.load_m1434, M.load_m1434_rebound)
        self.assertTrue(all(getattr(M.P, name) is before[name]
                            if callable(before[name]) else
                            getattr(M.P, name) == before[name]
                            for name in names))
        self.assertIs(M.P.P.load_m1434, old_loader)

    def test_11_new_namespaces_are_fresh_distinct_and_not_m1647(self):
        M.require_fresh_namespaces()
        paths = (M.RESULT, M.ATTEMPT, M.WORK, M.FAILURE)
        self.assertEqual(len(set(paths)), 4)
        self.assertTrue(all("m1668_" in path.name for path in paths))
        self.assertTrue(all(path not in (M.P.RESULT, M.P.ATTEMPT,
                                        M.P.WORK, M.P.FAILURE)
                            for path in paths))

    def test_12_source_self_check_is_inert_and_runtime_handoff_complete(self):
        row = M.source_self_check()
        self.assertEqual(row["runtime_handoff_files"], 9)
        self.assertEqual(row["runtime_canonical_files"], 7)
        self.assertTrue(row["build_runtime_before_parent_subprocess"])
        self.assertTrue(row["build_runtime_before_child_gpu_attempt"])
        self.assertEqual(row["gpu_runs"], 0)
        self.assertEqual(row["attempt_writes"], 0)

    def test_13_future_review_release_and_all_namespaces_are_absent(self):
        self.assertFalse(M.FUTURE_REVIEW.exists())
        self.assertFalse(M.FUTURE_RELEASE.exists())
        self.assertTrue(all(not path.exists() for path in
                            (M.RESULT, M.ATTEMPT, M.WORK, M.FAILURE)))

    def test_14_contract_keeps_one_shot_and_all_claims_closed(self):
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

    def test_15_contract_requires_runtime_and_build_preflight_before_budget(self):
        value = M.strict_json(M.SOURCE_CONTRACT)
        order = value["mandatory_pre_budget_order"]
        self.assertLess(order.index("verify M1257 canonical runtime result"),
                        order.index("launch the only clean child subprocess"))
        self.assertLess(order.index("execute M1434 build_runtime"),
                        order.index("launch the only clean child subprocess"))
        self.assertLess(order.index("repeat M1434 build_runtime in child"),
                        order.index("acquire GPU lease and consume attempt"))

    def test_16_source_contains_no_remote_client_or_direct_payload_read(self):
        text = SOURCE.read_text(encoding="utf-8")
        for token in ("paramiko", "ssh ", "scp ", "rsync ", "np.load(",
                      "torch.load(", "requests."):
            self.assertNotIn(token, text)
        self.assertIn("preflight_runtime_binding()", text)

    def test_17_predecessors_and_protected_doc_are_exact(self):
        M.verify_predecessors()
        self.assertEqual(M.sha256(M.DOCS359), M.DOCS359_SHA256)

    def test_18_selection_observation_is_not_launch_authority(self):
        value = M.selection_identity()
        observed = value["remote_read_only_observation"]
        self.assertFalse(observed["observation_is_launch_authority"])
        self.assertTrue(observed["future_launch_must_repeat_gpu_and_entity_preflight"])
        self.assertFalse(value["claim_boundary"]["paper_result"])


if __name__ == "__main__":
    unittest.main()
