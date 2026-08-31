#!/usr/bin/env python3
"""Adversarial future final-hammer authorization tests for M826/C2 R20.

Source-only: creates only temporary synthetic sealed chains and never invokes
VCS, simv, a license query, or EDA.
"""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
HW_ROOT = HERE.parent
GUARD_PATH = HERE / "m826_c2_r20_atomic_guard.py"
RUNNER = HW_ROOT / "dc_handoff/scripts/run_vcs_m826_c2_r20_atomic_exact_sha.sh"
CONTRACT = HW_ROOT / "contracts/m826_c2_r20_atomic_source_only_contract_r1_20260829.json"
CANDIDATE = HW_ROOT / "contracts/m826_c2_r20_vcs_launch_candidate_source_only_r1_20260829.json"


def load_guard():
    spec = importlib.util.spec_from_file_location("m826_final_auth_guard",
                                                  GUARD_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load M826 guard")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M826 = load_guard()


def seal_file(path):
    path = Path(path)
    inner = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    inner.write_text(M826.sha256(path) + "  " + path.name + "\n",
                     encoding="utf-8")
    outer.write_text(M826.sha256(inner) + "  " + inner.name + "\n",
                     encoding="utf-8")
    M826.verify_double_sealed_file(path)


class SyntheticChain(object):
    def __init__(self, root, authorization):
        self.root = Path(root)
        self.source = M826.validate_source(HW_ROOT, CONTRACT, CANDIDATE,
                                           RUNNER)
        self.source_hammer = self.root / "source_hammer"
        self.source_hammer.mkdir()
        M826.write_json(self.source_hammer / "review.json", {
            "status": M826.SOURCE_HAMMER_STATUS,
            "score_out_of_100": 100,
            "p0_count": 0, "p1_count": 0, "p2_count": 0,
            "review_target": {
                "runner_sha256": self.source["runner_sha256"],
                "contract_sha256": self.source["contract_sha256"],
                "candidate_sha256": self.source["candidate_sha256"],
            },
        })
        source_hammer_identity = M826.seal_directory(self.source_hammer)

        self.release = self.root / "release.json"
        M826.write_json(self.release, {
            "schema": "m826_c2_r20_atomic_vcs_launch_admission_v1",
            "status": M826.RELEASE_STATUS,
            "authorization": {
                "launch_now": True, "run_vcs": True,
                "run_simv": True, "query_license": True,
                "run_eda": False, "max_attempts": 1,
            },
            "source_binding": {
                "runner_sha256": self.source["runner_sha256"],
                "contract_sha256": self.source["contract_sha256"],
                "candidate_sha256": self.source["candidate_sha256"],
                "source_hammer_outer_seal_file_sha256":
                    source_hammer_identity["outer_seal_file_sha256"],
            },
        })
        seal_file(self.release)

        self.final_hammer = self.root / "final_hammer"
        self.final_hammer.mkdir()
        M826.write_json(self.final_hammer / "review.json", {
            "status": M826.FINAL_HAMMER_STATUS,
            "score_out_of_100": 100,
            "p0_count": 0, "p1_count": 0, "p2_count": 0,
            "authorization": authorization,
            "review_target": {
                "release_sha256": M826.sha256(self.release),
                "runner_sha256": self.source["runner_sha256"],
                "contract_sha256": self.source["contract_sha256"],
                "candidate_sha256": self.source["candidate_sha256"],
            },
        })
        self.final_identity = M826.seal_directory(self.final_hammer)

    def validate(self):
        return M826.validate_launch_chain(
            HW_ROOT, CONTRACT, CANDIDATE, RUNNER, self.source_hammer,
            self.release, self.final_hammer,
            self.final_identity["outer_seal_file_sha256"])


class M826FinalAuthorizationTests(unittest.TestCase):
    def validate_authorization(self, authorization):
        with tempfile.TemporaryDirectory(prefix="m826_final_auth.") as raw:
            return SyntheticChain(raw, authorization).validate()

    def assert_rejected(self, authorization):
        with self.assertRaisesRegex(
                M826.Failure, "final hammer authorization is not the exact"):
            self.validate_authorization(authorization)

    def test_exact_closed_authorization_passes(self):
        result = self.validate_authorization(
            copy.deepcopy(M826.FINAL_HAMMER_AUTHORIZATION))
        self.assertEqual(result["status"],
                         "PASS_M826_R20_EXACT_LAUNCH_CHAIN")

    def test_run_vcs_false_is_rejected(self):
        value = copy.deepcopy(M826.FINAL_HAMMER_AUTHORIZATION)
        value["run_vcs"] = False
        self.assert_rejected(value)

    def test_run_simv_false_is_rejected(self):
        value = copy.deepcopy(M826.FINAL_HAMMER_AUTHORIZATION)
        value["run_simv"] = False
        self.assert_rejected(value)

    def test_query_license_false_is_rejected(self):
        value = copy.deepcopy(M826.FINAL_HAMMER_AUTHORIZATION)
        value["query_license"] = False
        self.assert_rejected(value)

    def test_max_attempts_zero_is_rejected(self):
        value = copy.deepcopy(M826.FINAL_HAMMER_AUTHORIZATION)
        value["max_attempts"] = 0
        self.assert_rejected(value)

    def test_extra_key_is_rejected(self):
        value = copy.deepcopy(M826.FINAL_HAMMER_AUTHORIZATION)
        value["unexpected_key"] = "deny-list bypass"
        self.assert_rejected(value)

    def test_every_missing_key_is_rejected(self):
        for key in sorted(M826.FINAL_HAMMER_AUTHORIZATION):
            with self.subTest(key=key):
                value = copy.deepcopy(M826.FINAL_HAMMER_AUTHORIZATION)
                del value[key]
                self.assert_rejected(value)

    def test_boolean_integer_type_confusion_is_rejected(self):
        samples = (
            ("launch_now", 1), ("run_vcs", 1), ("run_simv", 1),
            ("query_license", 1), ("max_attempts", True),
            ("run_dc", 0), ("network_or_remote_jobs", False),
        )
        for key, injected in samples:
            with self.subTest(key=key, injected=repr(injected)):
                value = copy.deepcopy(M826.FINAL_HAMMER_AUTHORIZATION)
                value[key] = injected
                self.assert_rejected(value)


if __name__ == "__main__":
    unittest.main()
