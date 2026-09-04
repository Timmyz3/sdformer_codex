#!/usr/bin/env python3
from __future__ import print_function

import glob
import hashlib
import json
import os
import sys


ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
RELEASE_REL = "contracts/m882_m880_m803_c2_r16_channel_split_three_axis_dc_launch_admission_r1_20260829.json"
CANDIDATE_REL = "contracts/m880_m803_c2_r16_channel_split_three_axis_dc_launch_candidate_source_only_r1_20260829.json"
CONTRACT_REL = "contracts/m880_m803_c2_r16_channel_split_three_axis_dc_source_only_contract_r1_20260829.json"
RUNNER_REL = "dc_handoff/scripts/run_dc_m880_m803_c2_r16_channel_split_three_axis_exact_sha_r1.sh"
M881_REL = "reviews/m881_m880_m803_c2_r16_terminology_repair_source_fresh_hammer_r1_20260829/review.json"
M881_REQUEST_REL = "reviews/m881_m880_m803_c2_r16_terminology_repair_source_fresh_hammer_REQUEST_r1_20260829/request.json"
DOCS359_REL = "docs/359_DATE终局冻结_20260813.md"
FINAL_HAMMER_REL = "reviews/m886_m882_m803_c2_r16_final_release_hammer_r1_20260829"

EXPECTED = {
    RUNNER_REL: "3f5553cac5ccd61e87fe7e76bb5febc988c429ee5f36be7f23953879e402212e",
    CONTRACT_REL: "70c65ee56e8147de242081376e3da3cd73ac7b39ee0520aaaa7a8942808f6ee4",
    CANDIDATE_REL: "941f38419acb013ea2804dc88a25e607b35846a4855a7cf2cac950a1f7fafec2",
    M881_REL: "1c0ba000f182fe5184a870c11ceccb48b76723314d85263c65452948e62a548d",
    M881_REQUEST_REL: "cb0e5f9a1bcde1463c8e5babf59abef16ddd362b59e08c89a45a4bdc7260fb20",
    DOCS359_REL: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
EXPECTED_M881_MANIFEST_FILE_SHA = "d18d1b08a860a2d034a66a0012590e9a936a13c4b3bab203f9cd158520b8982d"
EXPECTED_M881_OUTER_FILE_SHA = "1b1d9bb85e99a48b5ed0944b8098ffe20a2ae4c6eeaa87d8e867f5c8a324d1b7"
EXPECTED_M880_HANDOFF_OUTER_FILE_SHA = "7108d4b9c540e7c0b2451609e75996a1e1f4b823c57eee60d2c95b2280ce5843"
EXPECTED_M881_REQUEST_OUTER_FILE_SHA = "9d3e5fd82927495b2580bf274a24a473ae4be886d00f60ff12bc04036bf95004"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def reject_nonfinite(value):
    raise ValueError("non-finite JSON constant: %s" % value)


def strict_load_bytes(payload):
    return json.loads(payload.decode("utf-8"), object_pairs_hook=unique_object,
                      parse_constant=reject_nonfinite)


def strict_load(path):
    with open(path, "rb") as handle:
        return strict_load_bytes(handle.read())


def verify_file_double_seal(path):
    sidecar = path + ".sha256"
    outer = sidecar + ".seal.sha256"
    require(os.path.isfile(path) and not os.path.islink(path), "payload missing/symlink")
    require(os.path.isfile(sidecar) and not os.path.islink(sidecar), "sidecar missing/symlink")
    require(os.path.isfile(outer) and not os.path.islink(outer), "outer missing/symlink")
    expected_payload, payload_name = open(sidecar, "r").read().strip().split(None, 1)
    expected_sidecar, sidecar_name = open(outer, "r").read().strip().split(None, 1)
    require(payload_name.lstrip("*") == os.path.basename(path), "sidecar name")
    require(sidecar_name.lstrip("*") == os.path.basename(sidecar), "outer name")
    require(expected_payload == sha(path), "payload SHA")
    require(expected_sidecar == sha(sidecar), "sidecar SHA")


def expect_type(value, expected, label):
    require(type(value) is expected, "%s type" % label)


def main():
    for rel, expected in EXPECTED.items():
        path = os.path.join(ROOT, rel)
        require(os.path.isfile(path) and not os.path.islink(path), "frozen file %s" % rel)
        require(sha(path) == expected, "frozen SHA %s" % rel)

    release_path = os.path.join(ROOT, RELEASE_REL)
    candidate_path = os.path.join(ROOT, CANDIDATE_REL)
    release = strict_load(release_path)
    candidate = strict_load(candidate_path)
    contract = strict_load(os.path.join(ROOT, CONTRACT_REL))
    m881 = strict_load(os.path.join(ROOT, M881_REL))
    strict_load(os.path.join(ROOT, M881_REQUEST_REL))
    verify_file_double_seal(release_path)
    verify_file_double_seal(candidate_path)
    verify_file_double_seal(os.path.join(ROOT, CONTRACT_REL))

    expect_type(release, dict, "release")
    expect_type(release["launch_now"], bool, "launch_now")
    expect_type(release["authorization"], dict, "authorization")
    expect_type(release["authorization"]["max_attempts"], int, "max_attempts")
    expect_type(release["authorization"]["run_dc"], bool, "run_dc")
    expect_type(release["source_static_hammer_binding"]["score_out_of_100"], int, "score")
    expect_type(release["source_static_hammer_binding"]["p0_p1_p2"], list, "p0_p1_p2")

    require(release["status"] == "AUTHORIZED_ONE_M880_M803_C2_R16_CHANNEL_SPLIT_THREE_AXIS_LOGIC_ONLY_DC_ATTEMPT_R1", "release status")
    require(release["launch_now"] is True, "release launch")
    require(release["authorization"] == {
        "max_attempts": 1, "run_dc": True, "run_formality": False,
        "run_pt": False, "run_ptpx": False, "run_remote": False,
        "run_vcs": False}, "closed authorization")
    require("source_only_authorization" not in release, "stale source-only authorization")
    require(candidate["launch_now"] is False, "candidate launch")
    require(candidate["status"] == "READY_FOR_FRESH_M880_M803_C2_R16_TERMINOLOGY_REPAIR_THREE_AXIS_DC_SOURCE_HAMMER__NO_EDA_AUTHORIZED", "candidate status")

    binding = release["candidate_binding"]
    require(binding["candidate_sha256"] == EXPECTED[CANDIDATE_REL], "candidate pin")
    require(binding["candidate_launch_now"] is False, "candidate binding launch")
    for key in binding["preserved_semantic_sections"]:
        require(release[key] == candidate[key], "semantic section %s" % key)

    source = release["source_static_hammer_binding"]
    require(source["review_sha256"] == EXPECTED[M881_REL], "M881 review pin")
    require(source["manifest_file_sha256"] == EXPECTED_M881_MANIFEST_FILE_SHA, "M881 manifest pin")
    require(source["outer_seal_file_sha256"] == EXPECTED_M881_OUTER_FILE_SHA, "M881 outer pin")
    require(source["candidate_author_handoff_outer_seal_file_sha256"] == EXPECTED_M880_HANDOFF_OUTER_FILE_SHA, "M880 handoff pin")
    require(source["source_hammer_request_sha256"] == EXPECTED[M881_REQUEST_REL], "M881 request pin")
    require(source["source_hammer_request_outer_seal_file_sha256"] == EXPECTED_M881_REQUEST_OUTER_FILE_SHA, "M881 request outer pin")
    require(source["exact_status"] == m881["status"], "M881 status")
    require(source["score_out_of_100"] == 100 and source["p0_p1_p2"] == [0, 0, 0], "M881 score")

    plan = release["three_axis_pre_attempt_plan"]
    require(plan["point_order"] == ["k1", "k8", "k1x8"], "axis order")
    require(plan["k1_binding"] == "frozen M519 ARCH_MODE=0 fairness baseline", "K1 ARCH0")
    require(plan["k8_binding"] == "M803 channel-split ARCH_MODE=1 candidate", "K8 ARCH1")
    require(plan["k1x8_binding"] == "frozen M519 ARCH_MODE=2 equal-bandwidth fairness baseline", "K1x8 ARCH2")
    require(plan["tim209_required_each_axis"] == 0, "TIM-209")
    require(plan["opt150_required_each_axis"] == 0, "OPT-150")
    require(plan["all_three_axes_same_attempt_required"] is True, "same attempt")
    require(release["identity"]["dc_runner_sha256"] == EXPECTED[RUNNER_REL], "runner identity")
    require(release["identity"]["recovery_contract_sha256"] == EXPECTED[CONTRACT_REL], "contract identity")
    require(release["release_authorization"]["inert_until_fresh_final_hammer_pass100"] is True, "inert final gate")
    require(release["claim_boundary"]["final_release_independently_hammered"] is False, "final hammer false")
    for key in ("dc_completed", "area", "setup_timing", "power", "energy", "throughput_per_area", "paper_ppa_ready", "complete_fc2", "system_speedup", "headline", "ppa", "system"):
        require(release["claim_boundary"][key] is False, "claim %s" % key)

    canonical = os.path.join(ROOT, "dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829")
    attempt = os.path.join(ROOT, "dc_handoff/runs/.m872_m803_c2_r16_channel_split_three_axis_dc_attempt_consumed")
    require(not os.path.lexists(canonical), "canonical absent")
    require(not os.path.lexists(attempt), "attempt absent")
    require(glob.glob(os.path.join(ROOT, "dc_handoff/runs/.m872_m803_c2_r16_channel_split_three_axis_dc_work.*")) == [], "work absent")
    require(glob.glob(canonical + ".failed_or_incomplete.*.quarantine") == [], "quarantine absent")
    require(not os.path.lexists(os.path.join(ROOT, FINAL_HAMMER_REL)), "final hammer output absent")

    duplicate_negatives = [b'{"x":1,"x":2}', b'{"launch_now":true,"launch_now":false}', b'{"authorization":{},"authorization":{}}']
    nonfinite_negatives = [b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}']
    for payload in duplicate_negatives + nonfinite_negatives:
        try:
            strict_load_bytes(payload)
        except ValueError:
            pass
        else:
            raise AssertionError("strict JSON negative accepted")

    require(contract["three_axis_pre_attempt_plan"] == release["three_axis_pre_attempt_plan"], "contract plan equality")
    print("PASS_M885_M882_INERT_RELEASE_AUTHOR_NO_EDA")
    print("python=%s" % sys.version.split()[0])
    print("strict_json_positive_documents=5")
    print("duplicate_key_negatives=3")
    print("nonfinite_negatives=3")
    print("typed_json_checks=6")
    print("preserved_semantic_sections=%d" % len(binding["preserved_semantic_sections"]))
    print("canonical_attempt_work_quarantine_absent=true")
    print("final_hammer_output_absent=true")
    print("eda_runs=0")
    print("license_queries=0")
    print("remote_runs=0")


if __name__ == "__main__":
    main()
