from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REVIEW_ROOT = ROOT / "reviews/m1312_m1309_canonical_path_compatibility_successor_r1_20260831"
REVIEW = REVIEW_ROOT / "review.json"
OBSERVATION = REVIEW_ROOT / "remote_canonical_readonly_observation.json"
M1309 = ROOT / "reviews/m1309_m1306_remote_final_selection_result_independent_hammer_r1_20260831"
ARCHIVE = ROOT / "system_handoff/incoming/m1306_remote_selection_result_20260830/m1306_remote_selection_result_20260830.tar"
STAGED_RESULTS = ROOT / "system_handoff/incoming/m1306_remote_selection_result_20260830/hw_autoresearch_nts07/results"
STAGED_OUT = STAGED_RESULTS / "m1257_motion_cross_run_final_checkpoint_selection_r5_20260830"
CONTRACT = ROOT / "contracts/m1312_m1309_canonical_path_compatibility_successor_contract_r1_20260831.json"

AUTHORITY_KEYS = {
    "result_path", "selection_member", "selection_sha256",
    "selection_manifest_sha256", "selection_outer_file_sha256",
    "selection_schema", "selection_status", "selected_candidate_id",
    "selected_epoch", "selected_profile_sha256", "selected_checkpoint_sha256",
    "selected_config_sha256",
}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class M1312CanonicalPathCompatibilityTest(unittest.TestCase):
    def test_01_frozen_m1309_double_seal_review_and_archive(self):
        self.assertEqual(sha(M1309 / "SHA256SUMS"),
                         "39a55a80ee2b7a0f9d5a1e4aad6df52b2bebfdccc856d9cd3d6532b6fa37861c")
        self.assertEqual(sha(M1309 / "SHA256SUMS.seal.sha256"),
                         "275b8f18538c72819879f921da0d6d56acf356f8f7df3e2ba60e7eb09acdac02")
        self.assertEqual(sha(M1309 / "review.json"),
                         "b0373ecfc5344c5f377d994d746222b8ad220d0ecc0070287ababb356f87603a")
        self.assertEqual(sha(ARCHIVE),
                         "0524a94ccb36adc7ebc17603dedc322810141d8b14dc743923c5b942a5c6c36f")

    def test_02_remote_observation_exact_result_population_matches_staged(self):
        observed = json.loads(OBSERVATION.read_text())
        rows = observed["exact_file_population"]
        self.assertEqual(set(rows), {path.name for path in STAGED_OUT.iterdir()})
        for name, expected in rows.items():
            self.assertEqual(sha(STAGED_OUT / name), expected)
        self.assertEqual(observed["comparison_to_staged"]["verdict"],
                         "PASS_CANONICAL_BYTES_EQUAL_STAGED")

    def test_03_remote_attempt_log_match_staged_and_are_nonproduction_observation(self):
        observed = json.loads(OBSERVATION.read_text())
        attempt = STAGED_RESULTS / Path(observed["attempt"]["path"]).name
        log = STAGED_RESULTS / Path(observed["log"]["path"]).name
        self.assertEqual(sha(attempt), observed["attempt"]["sha256"])
        self.assertEqual(sha(log), observed["log"]["sha256"])
        self.assertEqual(observed["attempt"]["mode"], "0400")
        self.assertEqual(observed["log"]["mode"], "0400")
        self.assertFalse(observed["transport"]["remote_python_or_production_invoked"])
        self.assertFalse(observed["scope"]["remote_mutation"])

    def test_04_review_is_exact_M1237_compatible_shape(self):
        value = json.loads(REVIEW.read_text())
        self.assertEqual(value["schema"],
                         "m1237_m1234_motion_cross_run_final_checkpoint_binder_result_hammer_r1_v1")
        self.assertEqual(value["status"],
                         "PASS_M1237_M1234_FINAL_SELECTION__HARDWARE_REBIND_RELEASE_AUTHORING_ALLOWED")
        self.assertEqual(set(value["selection_authority"]), AUTHORITY_KEYS)
        self.assertEqual(value["independence"], {"different_author": True})
        self.assertEqual(value["authorization"], {
            "hardware_rebind_release_authoring": True,
            "production_capture": False})

    def test_05_only_result_path_differs_from_M1309(self):
        old = json.loads((M1309 / "review.json").read_text())["selection_authority"]
        new = json.loads(REVIEW.read_text())["selection_authority"]
        changed = {key for key in AUTHORITY_KEYS if old[key] != new[key]}
        self.assertEqual(changed, {"result_path"})
        self.assertEqual(new["result_path"],
                         "hw_autoresearch_nts07/results/m1257_motion_cross_run_final_checkpoint_selection_r5_20260830")

    def test_06_selection_member_seals_and_selected_tuple_are_unchanged(self):
        authority = json.loads(REVIEW.read_text())["selection_authority"]
        self.assertEqual(authority["selection_sha256"], sha(STAGED_OUT / "final_checkpoint_selection.json"))
        self.assertEqual(authority["selection_manifest_sha256"], sha(STAGED_OUT / "SHA256SUMS"))
        self.assertEqual(authority["selection_outer_file_sha256"],
                         sha(STAGED_OUT / "SHA256SUMS.seal.sha256"))
        self.assertEqual((authority["selected_candidate_id"], authority["selected_epoch"]),
                         ("resume_ep34", 34))

    def test_07_authorization_does_not_promote_capture_speedup_or_energy(self):
        value = json.loads(REVIEW.read_text())
        boundary = value["m1312_audit"]["boundary"]
        self.assertTrue(boundary["hardware_rebind_release_authoring"])
        for key in ("hardware_rebind_execution", "production_capture",
                    "hardware_speedup", "system_speedup", "hardware_energy"):
            self.assertFalse(boundary[key])
        self.assertTrue(boundary["E2_E8_replay_or_recapture_required"])

    def test_08_contract_identity_when_present(self):
        if CONTRACT.exists():
            value = json.loads(CONTRACT.read_text())
            self.assertEqual(value["test"]["sha256"], sha(Path(__file__).resolve()))
            self.assertEqual(value["compatible_review"]["sha256"], sha(REVIEW))


if __name__ == "__main__": unittest.main(verbosity=2)
