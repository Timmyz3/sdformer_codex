#!/usr/bin/env python3
"""Source-only R22 identity closure and synthetic launch-chain tests."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
HW = HERE.parent
GUARD_PATH = HERE / "m837_c2_r22_identity_compat_guard.py"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m837_c2_r22_identity_compat_exact_sha.sh"
CONTRACT = HW / "contracts/m837_c2_r22_identity_compat_source_only_contract_r1_20260829.json"
CANDIDATE = HW / "contracts/m837_c2_r22_identity_compat_vcs_launch_candidate_source_only_r1_20260829.json"


spec = importlib.util.spec_from_file_location("m837_guard", str(GUARD_PATH))
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
B = G.base


def file_seal(path):
    path = Path(path)
    inner = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    inner.write_text(B.sha256(path) + "  " + path.name + "\n",
                     encoding="utf-8")
    outer.write_text(B.sha256(inner) + "  " + inner.name + "\n",
                     encoding="utf-8")


def sealed_review(path, value):
    path.mkdir()
    B.write_json(path / "review.json", value)
    B.seal_directory(path)
    return B.verify_sealed_directory(path)


class M837R22IdentityCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = G.validate_source(HW, CONTRACT, CANDIDATE, RUNNER)

    def m834(self):
        return B.strict_json(HW / G.M834_DIR / "review.json")

    def assert_m834_rejected(self, mutate, pattern):
        value = copy.deepcopy(self.m834())
        mutate(value)
        with self.assertRaisesRegex(B.Failure, pattern):
            G.validate_m834_review_object(value)

    def test_m834_exact_r21_status_and_four_key_target_pass(self):
        G.validate_m834_review_object(self.m834())

    def test_old_m826_status_is_rejected(self):
        self.assert_m834_rejected(
            lambda x: x.__setitem__("status", B.SOURCE_HAMMER_STATUS),
            "status drift")

    def test_wrong_m833_source_status_is_rejected(self):
        self.assert_m834_rejected(
            lambda x: x.__setitem__("status",
                "PASS100_M833_R21_WRONG_COMPATIBILITY_STATUS"),
            "status drift")

    def test_three_key_target_is_rejected(self):
        self.assert_m834_rejected(
            lambda x: x["review_target"].pop("author_handoff_sha256"),
            "key set drift")

    def test_extra_target_key_is_rejected(self):
        self.assert_m834_rejected(
            lambda x: x["review_target"].__setitem__("extra", "0" * 64),
            "key set drift")

    def test_missing_each_target_key_is_rejected(self):
        for key in sorted(G.expected_m834_target()):
            with self.subTest(key=key):
                self.assert_m834_rejected(
                    lambda x, k=key: x["review_target"].pop(k),
                    "key set drift")

    def test_spent_m826_release_is_bound_and_not_reusable(self):
        contract = B.strict_json(CONTRACT)
        G.verify_predecessor_authority(HW, contract)
        spent = contract["m832_spent_release_authority"]
        self.assertIs(spent["m826_release_reusable"], False)
        self.assertIs(spent["m826_attempt_consumed"], False)

    def make_chain(self, root, source_status=None, target=None):
        source_status = source_status or G.SOURCE_HAMMER_STATUS
        target = target or G.expected_r22_source_target(self.source)
        hammer = root / "source_hammer"
        source_identity = sealed_review(hammer, {
            "status": source_status, "score_out_of_100": 100,
            "p0_count": 0, "p1_count": 0, "p2_count": 0,
            "review_target": target,
        })
        release = root / "release.json"
        B.write_json(release, {
            "schema": "m837_c2_r22_vcs_launch_admission_v1",
            "status": G.RELEASE_STATUS,
            "authorization": {
                "launch_now": True, "run_vcs": True, "run_simv": True,
                "query_license": True, "run_eda": False,
                "max_attempts": 1,
            },
            "source_binding": {
                "runner_sha256": self.source["runner_sha256"],
                "contract_sha256": self.source["contract_sha256"],
                "candidate_sha256": self.source["candidate_sha256"],
                "source_hammer_outer_seal_file_sha256":
                    source_identity["outer_seal_file_sha256"],
                "m834_r21_outer_seal_file_sha256":
                    self.source["m834_r21_outer_seal_file_sha256"],
            },
        })
        file_seal(release)
        release_sha = B.sha256(release)
        final = root / "final_hammer"
        final_identity = sealed_review(final, {
            "status": G.FINAL_HAMMER_STATUS, "score_out_of_100": 100,
            "p0_count": 0, "p1_count": 0, "p2_count": 0,
            "review_target": {
                "release_sha256": release_sha,
                "runner_sha256": self.source["runner_sha256"],
                "contract_sha256": self.source["contract_sha256"],
                "candidate_sha256": self.source["candidate_sha256"],
            },
            "authorization": dict(G.FINAL_HAMMER_AUTHORIZATION),
        })
        return hammer, release, final, final_identity

    def test_positive_synthetic_r22_chain(self):
        with tempfile.TemporaryDirectory(prefix="m837_r22_positive.") as raw:
            paths = self.make_chain(Path(raw))
            result = G.validate_launch_chain(
                HW, CONTRACT, CANDIDATE, RUNNER, paths[0], paths[1], paths[2],
                paths[3]["outer_seal_file_sha256"])
            self.assertEqual(result["status"],
                             "PASS_M837_R22_EXACT_LAUNCH_CHAIN")

    def test_old_source_status_rejected_before_release(self):
        for status in (B.SOURCE_HAMMER_STATUS,
                       G.M834_STATUS,
                       "PASS100_M833_R21_WRONG_STATUS"):
            with self.subTest(status=status):
                with tempfile.TemporaryDirectory(prefix="m837_old_status.") as raw:
                    paths = self.make_chain(Path(raw), source_status=status)
                    with self.assertRaisesRegex(B.Failure,
                                                "PASS100 semantics drift"):
                        G.validate_launch_chain(
                            HW, CONTRACT, CANDIDATE, RUNNER,
                            paths[0], paths[1], paths[2],
                            paths[3]["outer_seal_file_sha256"])

    def test_r22_source_target_three_extra_and_missing_rejected(self):
        exact = G.expected_r22_source_target(self.source)
        mutations = []
        three = dict(exact)
        three.pop("m834_r21_outer_seal_file_sha256")
        mutations.append(three)
        extra = dict(exact)
        extra["extra"] = "0" * 64
        mutations.append(extra)
        for key in exact:
            missing = dict(exact)
            missing.pop(key)
            mutations.append(missing)
        for index, target in enumerate(mutations):
            with self.subTest(index=index):
                with tempfile.TemporaryDirectory(prefix="m837_bad_target.") as raw:
                    paths = self.make_chain(Path(raw), target=target)
                    with self.assertRaisesRegex(B.Failure,
                                                "key set drift"):
                        G.validate_launch_chain(
                            HW, CONTRACT, CANDIDATE, RUNNER,
                            paths[0], paths[1], paths[2],
                            paths[3]["outer_seal_file_sha256"])

    def test_wrong_release_and_final_statuses_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m837_bad_chain.") as raw:
            root = Path(raw)
            paths = self.make_chain(root)
            release = paths[1]
            value = B.strict_json(release)
            value["status"] = B.RELEASE_STATUS
            for suffix in (".sha256", ".sha256.seal.sha256"):
                Path(str(release) + suffix).unlink()
            B.write_json(release, value)
            file_seal(release)
            with self.assertRaisesRegex(B.Failure, "release status/schema drift"):
                G.validate_launch_chain(HW, CONTRACT, CANDIDATE, RUNNER,
                    paths[0], release, paths[2],
                    paths[3]["outer_seal_file_sha256"])


if __name__ == "__main__":
    unittest.main()
