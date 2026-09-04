from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "scripts/run_m1186_motion_ep29_e8_handoff_release_source.py"
CONTRACT = HW / "contracts/m1186_motion_ep29_e8_handoff_release_source_contract_r1_20260830.json"
CANONICAL = HW / "contracts/m1177r2_motion_ep29_e8_canonical_40_source_manifest_r1_20260830.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source():
    spec = importlib.util.spec_from_file_location("m1186_under_test", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class M1186Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_source()
        cls.contract = cls.m.strict_json(CONTRACT)

    def test_exact_source_contract_test_bindings(self):
        self.assertEqual(self.contract["source"], {
            "path": str(SOURCE.relative_to(ROOT)), "sha256": sha(SOURCE)})
        self.assertEqual(self.contract["tests"], {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha(Path(__file__).resolve())})

    def test_actual_contract_validates_without_hashing_large(self):
        result = self.m.validate_contract(self.contract, root=ROOT,
                                          hash_small=True, hash_large=False)
        self.assertEqual(result["status"],
                         "INERT_SOURCE_ONLY__M1185_P0_REPAIR__FRESH_M1187_HAMMER_REQUIRED")

    def test_exact40_matches_canonical_manifest_order_and_identity(self):
        canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
        rows = self.contract["preexisting_large"]
        self.assertEqual(len(rows), 40)
        self.assertEqual([row["local_path"] for row in rows],
                         [row["path"] for row in canonical["rows"]])
        self.assertEqual([(row["size_bytes"], row["sha256"]) for row in rows],
                         [(row["bytes"], row["sha256"]) for row in canonical["rows"]])

    def test_small_and_large_classes_are_disjoint(self):
        small = self.contract["small_transfer"]
        large = self.contract["preexisting_large"]
        self.assertTrue(all(row["class"] == "TRANSFER_SMALL" for row in small))
        self.assertTrue(all(row["class"] == "REMOTE_PREEXISTING_HASH_ONLY" for row in large))
        self.assertFalse({row["local_path"] for row in small} &
                         {row["local_path"] for row in large})

    def test_transfer_argv_is_literal_no_shell_interpolation(self):
        argv = self.m.transfer_argv(self.contract)
        self.assertEqual(argv[0], "/usr/bin/rsync")
        self.assertIn("--relative", argv)
        self.assertEqual(argv[-1], self.m.REMOTE_HOST + ":" + str(self.m.REMOTE_REPO) + "/")
        self.assertFalse(any("$" in item or "`" in item or "$(`" in item for item in argv))
        source = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("shell=True", source)

    def test_missing_required_runtime_dependency_rejected(self):
        broken = deepcopy(self.contract)
        target = str(self.m.RUNTIME_REL)
        broken["small_transfer"] = [row for row in broken["small_transfer"]
                                    if row["local_path"] != target]
        with self.assertRaisesRegex(self.m.HandoffError, "transfer populations|minimum runtime"):
            self.m.validate_contract(broken, root=ROOT, hash_small=False, hash_large=False)

    def test_remote_path_redirection_rejected(self):
        broken = deepcopy(self.contract)
        broken["small_transfer"][0]["remote_path"] = "/tmp/substitution"
        with self.assertRaisesRegex(self.m.HandoffError, "remote mapping"):
            self.m.validate_contract(broken, root=ROOT, hash_small=False, hash_large=False)

    def test_exact_namespace_and_zero_retry_policy(self):
        policy = self.contract["remote_policy"]
        namespaces = self.contract["runtime_namespaces"]
        self.assertIs(policy["automatic_retry"], False)
        self.assertNotEqual(namespaces["output"], namespaces["attempt_marker"])
        self.assertEqual(policy["canonical_lease"],
                         "hw_autoresearch_nts07/results/gpu_profile_lease.lock")

    def test_legacy_sigstop_process_is_visible_from_cmdline(self):
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            row = proc / "123"
            row.mkdir()
            (row / "cmdline").write_bytes(b"python\0m511_capture_watcher\0")
            self.assertEqual(self.m.running_legacy_watchers(proc), [123])

    def test_future_hammer_is_not_current_contract_authority(self):
        future = self.contract["future_hammer"]
        self.assertFalse(future["production_authorized_by_contract"])
        self.assertEqual(future["canonical_review_path"], str(self.m.FUTURE_HAMMER_REL))

    def test_docs359_unchanged(self):
        self.assertEqual(sha(ROOT / self.m.DOCS359_REL), self.m.DOCS359_SHA256)


if __name__ == "__main__":
    unittest.main()
