#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""No-EDA directed tests for M1344 source_absent/runtime_present release gates."""
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


HERE = Path(__file__).resolve().parent
CHECKER = HERE / "static_check_m1344_c2_activity_vcs_release_source.py"
SPEC = importlib.util.spec_from_file_location("m1344_checker_test", CHECKER)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def seal_dir(root: Path) -> None:
    for path in (root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"):
        if path.exists() or path.is_symlink():
            path.unlink()
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            rows.append((path.relative_to(root).as_posix(), M.sha(path)))
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join("{}  {}\n".format(digest, name)
                                for name, digest in rows))
    (root / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(M.sha(manifest)))


def sidecar(path: Path) -> None:
    sums = Path(str(path) + ".sha256")
    sums.write_text("{}  {}\n".format(M.sha(path), path.name))
    Path(str(path) + ".sha256.seal.sha256").write_text(
        "{}  {}\n".format(M.sha(sums), sums.name))


class RuntimeFixture:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory(prefix="m1344_runtime_chain_")
        self.root = Path(self.temp.name)
        self.paths = M.future_paths(self.root)
        source = self.paths["source_hammer"]
        release = self.paths["launch_release"]
        final = self.paths["final_hammer"]
        source.mkdir(parents=True)
        final.mkdir(parents=True)
        release.parent.mkdir(parents=True, exist_ok=True)
        claims = {key: False for key in M.CLAIMS}
        source_review = {
            "status": "PASS_M1345_M1344_C2_ACTIVITY_RELEASE_SOURCE__LAUNCH_RELEASE_MAY_BE_AUTHORED",
            "bindings": {"runner_sha256": M.sha(M.RUNNER),
                         "source_contract_sha256": M.sha(M.CONTRACT)},
            "claim_boundary": claims}
        (source / "review.json").write_text(json.dumps(source_review, sort_keys=True))
        seal_dir(source)
        authorization = {"vcs_compiles": 2, "simv_runs": 10,
                         "all_other_eda_runs": 0, "automatic_retry": False}
        release_json = {
            "status": "AUTHORIZE_ONE_M1344_C2_MAPPED_PRODUCTION_ACTIVITY_VCS_ATTEMPT",
            "launch_now": True,
            "identity": {"runner_sha256": M.sha(M.RUNNER),
                         "source_contract_sha256": M.sha(M.CONTRACT),
                         "source_hammer_review_sha256": M.sha(source / "review.json")},
            "authorization": authorization, "claim_boundary": claims}
        release.write_text(json.dumps(release_json, sort_keys=True))
        sidecar(release)
        final_review = {
            "status": "PASS_M1347_AUTHORIZE_ONE_M1344_C2_MAPPED_PRODUCTION_ACTIVITY_VCS_LAUNCH",
            "bindings": {"runner_sha256": M.sha(M.RUNNER),
                         "source_contract_sha256": M.sha(M.CONTRACT),
                         "source_hammer_review_sha256": M.sha(source / "review.json"),
                         "launch_release_sha256": M.sha(release)},
            "authorization": authorization, "claim_boundary": claims}
        (final / "review.json").write_text(json.dumps(final_review, sort_keys=True))
        seal_dir(final)
        self.expected = {
            "M1344_EXPECTED_RUNNER_SHA256": M.sha(M.RUNNER),
            "M1344_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256": M.sha(source / "review.json"),
            "M1344_EXPECTED_SOURCE_HAMMER_MANIFEST_SHA256": M.sha(source / "SHA256SUMS"),
            "M1344_EXPECTED_SOURCE_HAMMER_OUTER_FILE_SHA256": M.sha(source / "SHA256SUMS.seal.sha256"),
            "M1344_EXPECTED_LAUNCH_RELEASE_SHA256": M.sha(release),
            "M1344_EXPECTED_FINAL_HAMMER_REVIEW_SHA256": M.sha(final / "review.json"),
            "M1344_EXPECTED_FINAL_HAMMER_MANIFEST_SHA256": M.sha(final / "SHA256SUMS"),
            "M1344_EXPECTED_FINAL_HAMMER_OUTER_FILE_SHA256": M.sha(final / "SHA256SUMS.seal.sha256"),
        }

    def close(self):
        self.temp.cleanup()


class Tests(unittest.TestCase):
    def test_01_source_absent_positive(self):
        with tempfile.TemporaryDirectory(prefix="m1344_absent_") as td:
            out = M.validate_future("source_absent", M.future_paths(Path(td)))
            self.assertTrue(out["future_absent"])

    def test_02_runtime_present_legal_chain_positive(self):
        fixture = RuntimeFixture()
        try:
            out = M.validate_future("runtime_present", fixture.paths, fixture.expected)
            self.assertTrue(out["future_present"])
        finally:
            fixture.close()

    def test_03_old_future_absence_contradiction_regression(self):
        fixture = RuntimeFixture()
        try:
            with self.assertRaisesRegex(AssertionError, "future authority residue"):
                M.validate_future("source_absent", fixture.paths)
            M.validate_future("runtime_present", fixture.paths, fixture.expected)
            self.assertIn("assert all(not os.path.lexists(path) for path in future)",
                          M.OLD_CHECKER.read_text())
        finally:
            fixture.close()

    def test_04_each_missing_runtime_authority_rejected(self):
        for key in ("source_hammer", "launch_release", "final_hammer"):
            fixture = RuntimeFixture()
            try:
                path = fixture.paths[key]
                if path.is_dir(): shutil.rmtree(path)
                else: path.unlink()
                with self.assertRaises((AssertionError, FileNotFoundError)):
                    M.validate_future("runtime_present", fixture.paths, fixture.expected)
            finally:
                fixture.close()

    def test_05_each_external_sha_mismatch_rejected(self):
        fixture = RuntimeFixture()
        try:
            for key in M.ENV_NAMES:
                bad = dict(fixture.expected); bad[key] = "0" * 64
                with self.assertRaises(AssertionError, msg=key):
                    M.validate_future("runtime_present", fixture.paths, bad)
        finally:
            fixture.close()

    def test_06_symlinked_runtime_authority_rejected(self):
        fixture = RuntimeFixture()
        try:
            source = fixture.paths["source_hammer"]
            real = source.with_name(source.name + ".real")
            source.rename(real); os.symlink(real, source)
            with self.assertRaises(AssertionError):
                M.validate_future("runtime_present", fixture.paths, fixture.expected)
        finally:
            fixture.close()

    def test_07_runtime_claim_or_cardinality_lift_rejected(self):
        for mutation in ("claim", "cardinality"):
            fixture = RuntimeFixture()
            try:
                release = fixture.paths["launch_release"]
                value = json.loads(release.read_text())
                if mutation == "claim": value["claim_boundary"]["performance"] = True
                else: value["authorization"]["simv_runs"] = 11
                release.write_text(json.dumps(value, sort_keys=True)); sidecar(release)
                fixture.expected["M1344_EXPECTED_LAUNCH_RELEASE_SHA256"] = M.sha(release)
                with self.assertRaises(AssertionError):
                    M.validate_future("runtime_present", fixture.paths, fixture.expected)
            finally:
                fixture.close()

    def test_08_runner_is_inert_without_external_sha(self):
        before = [os.path.lexists(str(path)) for path in M.namespaces()]
        run = subprocess.run(["/usr/bin/bash", str(M.RUNNER)],
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        self.assertEqual(run.returncode, 2)
        self.assertIn("M1344_EXPECTED_RUNNER_SHA256 absent/invalid", run.stderr)
        self.assertEqual(before, [os.path.lexists(str(path)) for path in M.namespaces()])

    def test_09_full_chain_identity_is_in_all_receipt_paths(self):
        runner = M.RUNNER.read_text()
        for key in ("source_hammer_review_sha256", "source_hammer_manifest_sha256",
                    "source_hammer_outer_file_sha256", "final_hammer_review_sha256",
                    "final_hammer_manifest_sha256", "final_hammer_outer_file_sha256"):
            self.assertGreaterEqual(runner.count(key), 3, key)

    def test_10_exact_workloads_and_two_by_five_cardinality(self):
        contract = json.loads(M.CONTRACT.read_text())
        self.assertEqual(contract["workloads"], {"events": M.EVENTS,
            "k8_cycles": M.CYCLES["k8"], "k1x8_cycles": M.CYCLES["k1x8"]})
        self.assertEqual(contract["future_execution"]["vcs_compiles"], 2)
        self.assertEqual(contract["future_execution"]["simv_runs"], 10)

    def test_11_runtime_mode_is_the_only_runner_checker_mode(self):
        runner = M.RUNNER.read_text()
        self.assertIn('"${RELEASE_CHECKER}" --mode runtime_present', runner)
        self.assertNotIn('"${RELEASE_CHECKER}" --mode source_absent', runner)

    def test_12_source_common_static_passes_without_author(self):
        self.assertGreaterEqual(M.validate_common(skip_author=True), 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
