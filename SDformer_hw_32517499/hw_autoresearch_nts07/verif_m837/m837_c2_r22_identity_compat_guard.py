#!/usr/bin/env python3
"""M837/C2 R22 additive identity-compatibility guard.

All atomic publication, receipt, sealing and strict-JSON mechanics remain in
the frozen M826 guard.  This wrapper only closes the R22 source/release/final
identity vocabulary and the M834 R21 predecessor authority.  It never invokes
VCS, simv, lmutil or any EDA program.
"""

import argparse
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent / "verif_m826"
sys.path.insert(0, str(BASE_DIR))
import m826_c2_r20_atomic_guard as base  # noqa: E402


M834_DIR = "reviews/m834_m833_c2_r21_unicode_source_fresh_hammer_r1_20260829"
M834_STATUS = (
    "PASS100_M833_R21_UNICODE_SOURCE_FRESH_HAMMER__"
    "ONE_FRESH_TRUE_RELEASE_AUTHOR_ONLY__NO_LIVE_VCS"
)
M834_REVIEW_SHA256 = (
    "00b4289f5389f1d0b00e96bf80af6ed660e91d8349f947c94bb33b6d5eba8f0e"
)
M834_MANIFEST_SHA256 = (
    "e019d14190093c7cec3099279f8d7e7805de7092082677932b3b379a56ec5922"
)
M834_OUTER_SEAL_FILE_SHA256 = (
    "f43e74a6f0e7fc3c056568f3b9bb2870d2d60c2af1b512a52c2523b02908ed9e"
)
M833_RUNNER_SHA256 = (
    "a7f7494f65ac1f80fd8dbae3cc1065f3c7130f926369c82837ac299027d7389f"
)
M833_CONTRACT_SHA256 = (
    "f8ef374c1325ffb029f61d29afd74a8ff8a4c55cec7c658db381ee806780591c"
)
M833_CANDIDATE_SHA256 = (
    "95b667d96e5b1c503c45bc865806b177c79bb2a7eda3f21391f1d4a56c387939"
)
M833_AUTHOR_HANDOFF_SHA256 = (
    "9ca7f478c30304b9217c04314e54d31bf3c12292301b21f344f5dfdd8ffc038a"
)
M832_DIR = "reviews/m832_m826_c2_r20_unicode_preattempt_failure_hammer_r1_20260829"
M832_REVIEW_SHA256 = (
    "a0099bbd4ec42679a31c7cdaf44964427c16bf3d971f4c5673955c0db9f06de2"
)
M832_MANIFEST_SHA256 = (
    "93372a7ab3a14e9932f5d37af6464274d4b67825a35b8b6dd355be8e122cd5bc"
)
M832_OUTER_SEAL_FILE_SHA256 = (
    "1426d81e47c027e19d0ed9b38f60a0a7339127de205c8b98a18a457a7c06f6cd"
)

SOURCE_HAMMER_STATUS = (
    "PASS100_M837_R22_IDENTITY_COMPAT_SOURCE__"
    "AUTHORIZE_ONE_FRESH_RELEASE_ONLY"
)
RELEASE_STATUS = "AUTHORIZED_ONE_M837_R22_CHANNEL_SPLIT_VCS_ATTEMPT"
FINAL_HAMMER_STATUS = (
    "PASS100_M837_R22_FINAL_LAUNCH__ONE_VCS_ATTEMPT_AUTHORIZED"
)
FINAL_HAMMER_AUTHORIZATION = dict(base.FINAL_HAMMER_AUTHORIZATION)


def require_exact_mapping(actual, expected, label):
    base.require_exact_typed_mapping(actual, expected, label)


def expected_m834_target():
    return {
        "runner_sha256": M833_RUNNER_SHA256,
        "contract_sha256": M833_CONTRACT_SHA256,
        "candidate_sha256": M833_CANDIDATE_SHA256,
        "author_handoff_sha256": M833_AUTHOR_HANDOFF_SHA256,
    }


def validate_m834_review_object(review):
    base.require(review.get("status") == M834_STATUS and
                 review.get("score_out_of_100") == 100 and
                 (review.get("p0_count"), review.get("p1_count"),
                  review.get("p2_count")) == (0, 0, 0),
                 "M834 R21 PASS100 status drift")
    require_exact_mapping(review.get("review_target"),
                          expected_m834_target(),
                          "M834 R21 four-key target drift")


def verify_predecessor_authority(hw_root, contract):
    hw_root = Path(hw_root).resolve()
    m834_dir = hw_root / M834_DIR
    identity = base.verify_sealed_directory(m834_dir)
    base.regular_exact(m834_dir / "review.json", M834_REVIEW_SHA256,
                       "M834 R21 review")
    base.require(identity["manifest_sha256"] == M834_MANIFEST_SHA256 and
                 identity["outer_seal_file_sha256"] ==
                 M834_OUTER_SEAL_FILE_SHA256,
                 "M834 R21 double seal drift")
    review = base.strict_json(m834_dir / "review.json")
    validate_m834_review_object(review)
    expected_binding = {
        "directory": M834_DIR,
        "review_sha256": M834_REVIEW_SHA256,
        "manifest_sha256": M834_MANIFEST_SHA256,
        "outer_seal_file_sha256": M834_OUTER_SEAL_FILE_SHA256,
        "status": M834_STATUS,
        "review_target": expected_m834_target(),
    }
    require_exact_mapping(contract.get("m834_r21_source_authority"),
                          expected_binding,
                          "contract M834 R21 authority drift")

    m832_dir = hw_root / M832_DIR
    m832_identity = base.verify_sealed_directory(m832_dir)
    base.regular_exact(m832_dir / "review.json", M832_REVIEW_SHA256,
                       "M832 spent-release audit")
    base.require(m832_identity["manifest_sha256"] == M832_MANIFEST_SHA256 and
                 m832_identity["outer_seal_file_sha256"] ==
                 M832_OUTER_SEAL_FILE_SHA256,
                 "M832 spent-release audit seal drift")
    m832 = base.strict_json(m832_dir / "review.json")
    base.require(m832.get("claim_boundary", {}).get(
        "m826_release_reusable") is False and
        m832.get("claim_boundary", {}).get("m826_attempt_consumed") is False,
        "M826 spent-release boundary drift")
    expected_spent = {
        "directory": M832_DIR,
        "review_sha256": M832_REVIEW_SHA256,
        "manifest_sha256": M832_MANIFEST_SHA256,
        "outer_seal_file_sha256": M832_OUTER_SEAL_FILE_SHA256,
        "m826_release_reusable": False,
        "m826_attempt_consumed": False,
    }
    require_exact_mapping(contract.get("m832_spent_release_authority"),
                          expected_spent,
                          "contract M832 spent-release authority drift")
    return identity


def validate_source(hw_root, contract_path, candidate_path, runner_path):
    source = base.validate_source(hw_root, contract_path, candidate_path,
                                  runner_path)
    contract = base.strict_json(contract_path)
    predecessor = verify_predecessor_authority(hw_root, contract)
    source["m834_r21_outer_seal_file_sha256"] = predecessor[
        "outer_seal_file_sha256"]
    source["status"] = "PASS_M837_R22_SOURCE_IDENTITY__NO_VCS_OR_EDA"
    return source


def expected_r22_source_target(source):
    return {
        "runner_sha256": source["runner_sha256"],
        "contract_sha256": source["contract_sha256"],
        "candidate_sha256": source["candidate_sha256"],
        "m834_r21_outer_seal_file_sha256":
            source["m834_r21_outer_seal_file_sha256"],
    }


def validate_launch_chain(hw_root, contract_path, candidate_path, runner_path,
                          source_hammer_dir, release_path, final_hammer_dir,
                          expected_final_outer):
    source = validate_source(hw_root, contract_path, candidate_path,
                             runner_path)
    source_hammer_dir = Path(source_hammer_dir).resolve()
    release_path = Path(release_path).resolve()
    final_hammer_dir = Path(final_hammer_dir).resolve()

    source_identity = base.verify_sealed_directory(source_hammer_dir)
    source_review = base.strict_json(source_hammer_dir / "review.json")
    base.require(source_review.get("status") == SOURCE_HAMMER_STATUS and
                 source_review.get("score_out_of_100") == 100 and
                 (source_review.get("p0_count"), source_review.get("p1_count"),
                  source_review.get("p2_count")) == (0, 0, 0),
                 "R22 source hammer PASS100 semantics drift")
    require_exact_mapping(source_review.get("review_target"),
                          expected_r22_source_target(source),
                          "R22 source hammer exact four-key target drift")

    base.verify_double_sealed_file(release_path)
    release = base.strict_json(release_path)
    base.require(release.get("schema") ==
                 "m837_c2_r22_vcs_launch_admission_v1" and
                 release.get("status") == RELEASE_STATUS,
                 "R22 true release status/schema drift")
    require_exact_mapping(release.get("authorization"), {
        "launch_now": True, "run_vcs": True, "run_simv": True,
        "query_license": True, "run_eda": False, "max_attempts": 1,
    }, "R22 true release authorization drift")
    require_exact_mapping(release.get("source_binding"), {
        "runner_sha256": source["runner_sha256"],
        "contract_sha256": source["contract_sha256"],
        "candidate_sha256": source["candidate_sha256"],
        "source_hammer_outer_seal_file_sha256":
            source_identity["outer_seal_file_sha256"],
        "m834_r21_outer_seal_file_sha256":
            source["m834_r21_outer_seal_file_sha256"],
    }, "R22 true release source binding drift")

    final_identity = base.verify_sealed_directory(final_hammer_dir)
    base.require(final_identity["outer_seal_file_sha256"] ==
                 expected_final_outer,
                 "caller final-hammer outer seal pin drift")
    final_review = base.strict_json(final_hammer_dir / "review.json")
    base.require(final_review.get("status") == FINAL_HAMMER_STATUS and
                 final_review.get("score_out_of_100") == 100 and
                 (final_review.get("p0_count"), final_review.get("p1_count"),
                  final_review.get("p2_count")) == (0, 0, 0),
                 "R22 final hammer PASS100 semantics drift")
    require_exact_mapping(final_review.get("authorization"),
                          FINAL_HAMMER_AUTHORIZATION,
                          "R22 final hammer exact authorization drift")
    require_exact_mapping(final_review.get("review_target"), {
        "release_sha256": base.sha256(release_path),
        "runner_sha256": source["runner_sha256"],
        "contract_sha256": source["contract_sha256"],
        "candidate_sha256": source["candidate_sha256"],
    }, "R22 final hammer exact target drift")
    return {
        "status": "PASS_M837_R22_EXACT_LAUNCH_CHAIN",
        "release_sha256": base.sha256(release_path),
        "final_hammer_outer_seal_file_sha256": expected_final_outer,
        "source": source,
    }


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command not in ("validate-source", "validate-launch-chain"):
        return base.main()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    source = sub.add_parser("validate-source")
    source.add_argument("--hw-root", required=True)
    source.add_argument("--contract", required=True)
    source.add_argument("--candidate", required=True)
    source.add_argument("--runner", required=True)
    launch = sub.add_parser("validate-launch-chain")
    launch.add_argument("--hw-root", required=True)
    launch.add_argument("--contract", required=True)
    launch.add_argument("--candidate", required=True)
    launch.add_argument("--runner", required=True)
    launch.add_argument("--source-hammer", required=True)
    launch.add_argument("--release", required=True)
    launch.add_argument("--final-hammer", required=True)
    launch.add_argument("--expected-final-outer", required=True)
    args = parser.parse_args()
    if args.command == "validate-source":
        value = validate_source(args.hw_root, args.contract, args.candidate,
                                args.runner)
    else:
        value = validate_launch_chain(
            args.hw_root, args.contract, args.candidate, args.runner,
            args.source_hammer, args.release, args.final_hammer,
            args.expected_final_outer)
    print(base.json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
