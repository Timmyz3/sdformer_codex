#!/usr/bin/env python3
from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import sys


EXPECTED = {
    "checkpoint": "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48",
    "capture_manifest": "3ab8431e3d7d17d6933c0b87da4a3405e87c97ccc302a27c78491b0a02491d6d",
    "capture_ordered": "5956085b196979848c3d283744396ea3b0a38a268fb21af0eaecb53e87fc6c9c",
    "capture_sha256s": "f7f7a08696611875837196b990575453141b5e8edbf6d4aae61f7db1ed238b8e",
    "capture_outer_seal_file": "7cf434b834d30c003153eef8e83e70d574b1c5a7d20ca4c2208902c6e0c76eed",
    "m1575_result": "b8baf32b89579ef4f7065973d93364ec633d5e8965def97bf9cf5f2972a3fdad",
    "m1575_sha256s": "485f17c224d33d7a9be2cee3a66acc4b248b8c0d45ba762ee26dfb17d9fbb8f2",
    "m1575_outer_seal_file": "3f243add74ee9934e35f963fbc98ff12d8e0ff8623882ff66f3e94f3896228e8",
    "m1542_review": "b85014ca32604b7b2659a7ba962bfb873bdb4c330dc011ff94d263ee6898c970",
    "m1562_review": "40d7784854f25ef7e547087dc35ce496b22030e34b3a20886430597b44dac6ee",
    "m1572_review": "34e109794409ad0c1af56101862cd9ce2c21a3ae327a94e3044cf5cfc9b3f9d1"
}


def reject_duplicate_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key: " + key)
        out[key] = value
    return out


def reject_constant(value):
    raise ValueError("non-finite JSON constant: " + value)


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle, object_pairs_hook=reject_duplicate_pairs,
                         parse_constant=reject_constant)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(a, b, tol=1e-12):
    return math.fabs(float(a) - float(b)) <= tol


def parse_sha256s(path):
    rows = []
    with open(path, "r") as handle:
        for number, raw in enumerate(handle, 1):
            line = raw.rstrip("\n")
            if not line:
                continue
            require(len(line) >= 67 and line[64:66] == "  ",
                    "malformed SHA256SUMS line %d" % number)
            rows.append((line[:64], line[66:]))
    return rows


def verify_small_seal(directory, expected_members, expected_manifest,
                      expected_outer_file):
    manifest = os.path.join(directory, "SHA256SUMS")
    outer = os.path.join(directory, "SHA256SUMS.seal.sha256")
    require(sha256(manifest) == expected_manifest, "manifest hash mismatch")
    require(sha256(outer) == expected_outer_file, "outer seal file hash mismatch")
    with open(outer, "r") as handle:
        outer_line = handle.read().strip()
    require(outer_line == expected_manifest + "  SHA256SUMS",
            "outer seal content mismatch")
    rows = parse_sha256s(manifest)
    require([name for _, name in rows] == expected_members,
            "sealed member list/order mismatch")
    for expected_hash, name in rows:
        require(sha256(os.path.join(directory, name)) == expected_hash,
                "sealed member hash mismatch: " + name)
    return len(rows)


def file_inventory(paths):
    present = []
    missing = []
    for path in paths:
        if os.path.isfile(path):
            present.append({
                "name": os.path.basename(path),
                "bytes": os.path.getsize(path),
                "sha256": sha256(path)
            })
        else:
            missing.append(os.path.basename(path))
    return present, missing


def build_audit(project_root):
    hw = os.path.join(project_root, "hw_autoresearch_nts07")
    m1575_dir = os.path.join(
        hw, "reviews",
        "m1575_m1458_ep34_live93_s2_ccbs16_activity_relative_fastkill_r1_20260901")
    capture_dir = os.path.join(
        hw, "results",
        "m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831")
    checkpoint = os.path.join(
        hw, "system_handoff", "incoming",
        "motion_c12_ep34_live93_checkpoint_epoch34.pth")
    contract_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "next_experiment_contract.json")

    sealed_count = verify_small_seal(
        m1575_dir,
        ["analyze.py", "result.json", "review.json", "review.md", "RUN_COMPLETE.txt"],
        EXPECTED["m1575_sha256s"], EXPECTED["m1575_outer_seal_file"])
    result_path = os.path.join(m1575_dir, "result.json")
    require(sha256(result_path) == EXPECTED["m1575_result"],
            "M1575 result identity mismatch")

    result = load_json(result_path)
    review = load_json(os.path.join(m1575_dir, "review.json"))
    require(result["claim_boundary"]["decoder_only"] is True,
            "M1575 is not decoder-only")
    for key in ["cycles", "traffic", "energy", "speedup", "system_speedup",
                "paired_aee", "rtl", "eda"]:
        require(result["claim_boundary"][key] is False,
                "M1575 unexpectedly authorizes " + key)
    require(review["headline_legal"] is False, "M1575 unexpectedly headline legal")
    require(result["population"]["samples_per_sequence"] == 10,
            "M1575 samples per sequence mismatch")
    require(result["population"]["sequences"] ==
            ["interlaken_01_a", "thun_01_b", "zurich_city_12_a"],
            "M1575 decoder sequence identity mismatch")

    rows = dict((float(row["epsilon"]), row)
                for row in result["global_epsilon_rows"])
    require(sorted(rows.keys()) == [0.0, 0.01, 0.02, 0.05, 0.1],
            "M1575 epsilon grid mismatch")
    expected_numbers = {
        0.0: (0.2217227564102564, 0.21999321857857373),
        0.02: (0.25165214342948716, 0.24888587632435194),
        0.1: (0.3016820245726496, 0.29930345474346565)
    }
    for epsilon, expected_pair in expected_numbers.items():
        require(close(rows[epsilon]["drop_fraction"], expected_pair[0]),
                "drop fraction mismatch at epsilon %s" % epsilon)
        require(close(rows[epsilon]["potential_weight_byte_suppression_fraction"],
                      expected_pair[1]),
                "weight eligibility mismatch at epsilon %s" % epsilon)
        require(rows[epsilon]["cycles"] is False and
                rows[epsilon]["traffic"] is False and
                rows[epsilon]["paired_aee"] is False,
                "proxy row acquired a measured claim")

    manifest_path = os.path.join(capture_dir, "manifest.json")
    ordered_path = os.path.join(capture_dir, "unified_ordered_records.jsonl")
    capture_sha256s = os.path.join(capture_dir, "SHA256SUMS")
    capture_outer = os.path.join(capture_dir, "SHA256SUMS.seal.sha256")
    require(sha256(manifest_path) == EXPECTED["capture_manifest"],
            "capture manifest mismatch")
    require(sha256(ordered_path) == EXPECTED["capture_ordered"],
            "capture ordered trace mismatch")
    require(sha256(capture_sha256s) == EXPECTED["capture_sha256s"],
            "capture SHA256SUMS mismatch")
    require(sha256(capture_outer) == EXPECTED["capture_outer_seal_file"],
            "capture outer seal file mismatch")
    with open(capture_outer, "r") as handle:
        require(handle.read().strip() == EXPECTED["capture_sha256s"] + "  SHA256SUMS",
                "capture outer seal content mismatch")
    require(sha256(checkpoint) == EXPECTED["checkpoint"],
            "checkpoint mismatch")

    capture = load_json(manifest_path)
    samples = capture["cohort"]["samples"]
    require(len(samples) == 40, "capture population is not 40")
    require([row["global_sample_id"] for row in samples] == list(range(40)),
            "capture sample order/IDs are not canonical 0..39")
    keys = [row["sample_key"] for row in samples]
    require(len(set(keys)) == 40, "duplicate sample key")
    counts = {}
    for row in samples:
        counts[row["sequence"]] = counts.get(row["sequence"], 0) + 1
    expected_counts = {
        "zurich_city_09_a": 10,
        "interlaken_01_a": 10,
        "thun_01_b": 10,
        "zurich_city_12_a": 10
    }
    require(counts == expected_counts, "capture sequence coverage mismatch")

    event_missing = []
    event_bad_hash = []
    event_bytes = 0
    for row in samples:
        path = os.path.join(project_root, row["path"])
        if not os.path.isfile(path):
            event_missing.append(row["sample_key"])
            continue
        event_bytes += os.path.getsize(path)
        if sha256(path) != row["sha256"]:
            event_bad_hash.append(row["sample_key"])
    require(not event_missing, "event tensors missing")
    require(not event_bad_hash, "event tensor hash mismatch")

    data_root = os.path.join(project_root, "data", "Datasets", "DSEC",
                             "saved_flow_data")
    gt_paths = [os.path.join(data_root, "gt_tensors", key) for key in keys]
    mask_paths = [os.path.join(data_root, "mask_tensors", key) for key in keys]
    gt_present, gt_missing = file_inventory(gt_paths)
    mask_present, mask_missing = file_inventory(mask_paths)
    require(len(gt_present) == 10 and len(gt_missing) == 30,
            "GT inventory changed; re-audit required")
    require(len(mask_present) == 10 and len(mask_missing) == 30,
            "mask inventory changed; re-audit required")
    require(gt_missing == mask_missing, "GT/mask missing populations differ")
    require(all(name.startswith(("interlaken_01_a_", "thun_01_b_",
                                 "zurich_city_12_a_")) for name in gt_missing),
            "unexpected GT/mask missing cohort")

    for name, expected in [("m1542", EXPECTED["m1542_review"]),
                           ("m1562", EXPECTED["m1562_review"]),
                           ("m1572", EXPECTED["m1572_review"])]:
        matches = [entry for entry in os.listdir(os.path.join(hw, "reviews"))
                   if entry.startswith(name + "_")]
        require(len(matches) == 1, name + " review path ambiguity")
        path = os.path.join(hw, "reviews", matches[0], "review.json")
        require(sha256(path) == expected, name + " review identity mismatch")

    contract = load_json(contract_path)
    require(contract["status"] == "DESIGN_ONLY__NO_GO_EXECUTION__NO_GO_RTL",
            "contract status mismatch")
    require(contract["scope_authority"]["operator_scope"] ==
            "decoder ConvTranspose D0-D3 only", "scope drift")
    require([arm["epsilon"] for arm in contract["policy_arms"]] ==
            [None, 0.0, 0.02, 0.1], "policy arm drift")
    require(contract["joint_admission_gate"]["otherwise"].startswith("NO_GO_RTL"),
            "fail-closed gate missing")
    require(close(contract["paired_aee_protocol"]["lossy_gate"]
                  ["overall_valid_pixel_weighted_delta_aee_le"], 0.02),
            "overall AEE gate drift")

    return {
        "schema": "m1604_s2_next_experiment_first_principles_independent_audit_r1_v1",
        "status": "PASS_READONLY_IDENTITY_AND_DATA_GAP_AUDIT__NO_GO_EXECUTION__NO_GO_RTL",
        "date": "2026-09-01",
        "verification": {
            "m1575_exact_member_seal": "PASS_%d_OF_%d" % (sealed_count, sealed_count),
            "final_capture_manifest_order_outer_seal_and_checkpoint": "PASS",
            "capture_population": 40,
            "sequence_counts": counts,
            "event_tensors_present_and_hash_matched": 40,
            "event_tensor_bytes": event_bytes,
            "ground_truth_present": len(gt_present),
            "ground_truth_missing": len(gt_missing),
            "valid_masks_present": len(mask_present),
            "valid_masks_missing": len(mask_missing),
            "m1542_m1562_m1572_authorities": "PASS_EXACT_SHA"
        },
        "eligibility_audit": {
            "epsilon_0_block_eligibility_fraction": rows[0.0]["drop_fraction"],
            "epsilon_0_weight_eligibility_fraction":
                rows[0.0]["potential_weight_byte_suppression_fraction"],
            "epsilon_0p02_block_eligibility_fraction": rows[0.02]["drop_fraction"],
            "epsilon_0p02_weight_eligibility_fraction":
                rows[0.02]["potential_weight_byte_suppression_fraction"],
            "epsilon_0p1_block_eligibility_fraction": rows[0.1]["drop_fraction"],
            "epsilon_0p1_weight_eligibility_fraction":
                rows[0.1]["potential_weight_byte_suppression_fraction"],
            "epsilon_0p1_incremental_weight_eligibility_over_epsilon_0_fraction":
                rows[0.1]["potential_weight_byte_suppression_fraction"] -
                rows[0.0]["potential_weight_byte_suppression_fraction"],
            "measured_cycles": False,
            "measured_speedup": False,
            "paired_aee": False,
            "interpretation": "30.168202% is decoder block eligibility, not speedup; epsilon=0.1 adds only 7.931024 percentage points of weight eligibility over the exact epsilon=0 arm."
        },
        "scope_audit": {
            "m1575_actual_scope": "decoder ConvTranspose D0-D3",
            "fc_or_patch_supported_by_m1575": False,
            "c1_bottleneck_conv_overlap": False,
            "physical_shared_fabric_cost_must_be_charged": True,
            "local_ratio_multiplication_allowed": False
        },
        "blocking_data_gaps": {
            "missing_gt_sample_keys": gt_missing,
            "missing_mask_sample_keys": mask_missing,
            "unpruned_and_epsilon_predictions_present": False,
            "decoder_retained_payload_for_all_40_present": False,
            "missing_decoder_payload_population": "ten zurich_city_09_a samples",
            "frozen_s2_same_resource_address_timed_result_present": False,
            "o16_physical_bank_burst_mapping_frozen": False,
            "fixed_point_quantization_bridge_frozen": False
        },
        "decision": {
            "execution_authorized": False,
            "rtl_authorized": False,
            "why": "The 40-sample paired AEE population is incomplete locally and no same-resource address-timed S2 transaction result or fixed-point metadata bridge exists.",
            "next_experiment_contract": "next_experiment_contract.json",
            "go_rtl_gate": "same precommitted arm: epsilon0 exact, overall DeltaAEE<=0.02, every sequence DeltaAEE<=0.03, local decoder cycle speedup>=1.15x, all 40 samples, charged bank/burst/metadata/debt and frozen quantization"
        },
        "claim_boundary": {
            "read_only_audit": True,
            "new_performance": False,
            "new_aee": False,
            "gpu": False,
            "eda": False,
            "rtl": False,
            "author_evidence_modified": False,
            "paper_headline": False
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-frozen", default=None)
    args = parser.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    result = build_audit(project_root)
    if args.check_frozen:
        frozen = load_json(args.check_frozen)
        require(frozen == result, "frozen audit does not match recomputation")
        print("PASS_M1604_FROZEN_AUDIT_MATCH")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
