#!/usr/bin/env python3
"""Fail-closed validator for the independent M51-r1 hammer review."""

from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
REVIEW = (HW_ROOT / "results/m51_r1_independent_hammer_20260823"
          / "m51_r1_independent_hammer_review.json")
EXPECTED_REVIEW_SHA256 = "6e0fba14200d5d1be730f351b4b8e34f8327cf8b4030207aba9dbd876707c0d7"
EXPECTED_RECONSTRUCTION_SHA256 = "96bc764041ad8afa0d9ad3fd10174bcd9ee5e973cb20054c73cb554edfbe6188"
EXPECTED_ATTACK_SHA256 = "acf10f237d54575ded0a49aaef0004bd42d63810757b9eddf87dafb4cdf835e6"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON: {}".format(raw))

    def pairs(raw_pairs):
        value = {}
        for key, item in raw_pairs:
            require(key not in value, "duplicate JSON key: {}".format(key))
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def resolve_hw(item):
    return (HW_ROOT / item["path"]).resolve()


def check_call(command, label):
    result = subprocess.run(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)
    require(result.returncode == 0,
            "{} failed rc={} stdout={} stderr={}".format(
                label, result.returncode, result.stdout[-1000:], result.stderr[-1000:]))
    return result


def validate(rerun):
    require(sha256(REVIEW) == EXPECTED_REVIEW_SHA256, "review SHA drift")
    review = strict_json(REVIEW)
    require(review["status"] == "NO_GO_GPU_READY_RUNNER_HAS_P1_REPAIRS",
            "review status drift")
    require(review["verdict"] ==
            "NO_GO_M51_R1_GPU_READY_PREFLIGHT_PENDING_RUNNER_R2",
            "review verdict drift")
    require(review["score_0_to_100"] == 82, "score drift")
    require(review["severity_counts"] == {"P0": 0, "P1": 2, "P2": 5},
            "severity drift")
    require(len(review["findings"]) == 7 and
            [item["severity"] for item in review["findings"]] ==
            ["P1", "P1", "P2", "P2", "P2", "P2", "P2"],
            "finding population drift")
    require([item["id"] for item in review["findings"][:2]] ==
            ["M51-P1-01", "M51-P1-02"], "P1 identity drift")

    for group in ("candidate_anchors", "frozen_trace_anchors"):
        for name, item in review[group].items():
            path = resolve_hw(item)
            require(path.is_file() and sha256(path) == item["sha256"],
                    "{} anchor drift: {}".format(group, name))

    launch = strict_json(resolve_hw(review["candidate_anchors"]["launch_manifest"]))
    require(launch["status"] == "GPU_READY_NOT_EXECUTED" and
            launch["claim_boundary"] ==
            "NO_GPU_PAYLOAD_NO_OUTPUT_NO_CYCLE_NO_SPEEDUP_NO_PPA_NO_ENERGY_CLAIM",
            "producer launch boundary drift")
    for name, item in launch["sources"].items():
        path = (ROOT / item["path"]).resolve()
        require(path.is_file() and sha256(path) == item["sha256"],
                "launch source drift: {}".format(name))

    receipt = strict_json(resolve_hw(review["candidate_anchors"]["preflight_receipt"]))
    require(receipt["status"] ==
            "PASS_GPU_READY_STATIC_AND_PURE_CPU_ONLY_NOT_EXECUTED",
            "producer receipt status")
    require(receipt["population"] == {
        "dual_line_rows": 3100,
        "hook_calls": 310,
        "input_elements": 10506240000,
        "modules": 31,
        "operator_types": {"Conv2d": 7, "Linear": 24},
        "packed_bytes": 1313280000,
        "packed_gib": 1.2230873107910156,
        "samples": 10,
    }, "producer receipt population")
    require(receipt["execution"] == {
        "activation_payload_present": False,
        "checkpoint_opened": False,
        "gpu_run_performed": False,
        "remote_contacted": False,
    }, "producer execution boundary")
    require(all(value == "PASS" for value in receipt["checks"].values()) and
            len(receipt["checks"]) == 6, "producer checks")

    evidence = review["independent_evidence"]
    for name in ("reconstruction_checker", "reconstruction_result",
                 "writer_attack_test", "preflight_attack_runner",
                 "preflight_attack_result"):
        item = evidence[name]
        path = resolve_hw(item)
        require(path.is_file() and sha256(path) == item["sha256"],
                "independent evidence drift: {}".format(name))
    reconstruction_path = resolve_hw(evidence["reconstruction_result"])
    require(sha256(reconstruction_path) == EXPECTED_RECONSTRUCTION_SHA256,
            "reconstruction identity")
    reconstructed = strict_json(reconstruction_path)
    require(reconstructed["status"] ==
            "PASS_FROZEN_TRACE_RECONSTRUCTION_WITH_GPU_RUNNER_GAPS",
            "reconstruction status")
    summary = reconstructed["reconstruction"]
    require(summary == {
        "all_call_elements_byte_aligned": True,
        "all_input_temporal_axis0_size10": True,
        "dual_line_rows": 3100,
        "hook_calls": 310,
        "input_elements_bits": 10506240000,
        "max_call_elements": 147456000,
        "max_call_float32_bytes": 589824000,
        "modules": 31,
        "operator_types": {"Conv2d": 7, "Linear": 24},
        "packed_bytes": 1313280000,
        "reference_execution_call_indices": [
            2, 4, 6, 8, 10, 13, 23, 25, 35, 37, 39, 49, 51, 61,
            63, 65, 75, 77, 87, 89, 99, 101, 111, 113, 123, 125,
            135, 137, 151, 163, 183],
        "samples": 10,
    }, "reconstruction metric drift")
    require(len(reconstructed["negative_mutations_rejected"]) == 10 and
            reconstructed["producer_modules_imported"] == [],
            "independent mutation/isolation drift")
    runner_static = reconstructed["runner_static"]
    require(runner_static["checks"]["explicit_cuda_synchronize"] is False and
            runner_static["checks"]["full_tensor_contiguous_before_chunk"] is True and
            runner_static["final_explicit_cuda_sync_before_manifest"] is False and
            runner_static["possible_full_contiguous_copy_bytes_float32"] == 589824000,
            "P1 runner evidence drift")

    attack_path = resolve_hw(evidence["preflight_attack_result"])
    require(sha256(attack_path) == EXPECTED_ATTACK_SHA256,
            "attack result identity")
    attack = strict_json(attack_path)
    require(attack["status"] == "PASS_BASELINE_AND_FAIL_CLOSED_ATTACKS" and
            attack["baseline_return_code"] == 0 and attack["attack_count"] == 5 and
            attack["baseline_cpu_tests_pass_marker_count"] == 1 and
            attack["attacks_rejected"] == [
                "mutated_plan_sha", "mutated_launch_status",
                "mutated_launch_source_sha", "occupied_receipt_no_overwrite",
                "occupied_plan_no_overwrite"], "attack result drift")

    require(review["independent_frozen_reconstruction"]["mismatch_count"] == 0 and
            review["independent_frozen_reconstruction"]["max_call_float32_mib"] == 562.5,
            "review reconstruction claim drift")
    require(review["writer_and_runner_audit"]["gpu_capture_performed"] is False,
            "review falsely admits GPU execution")
    require(len(review["admitted_claims"]) == 3 and
            len(review["not_admitted"]) == 3,
            "claim population drift")
    forbidden = " ".join(review["not_admitted"])
    for token in ("GPU-ready", "captured", "cycle", "speedup", "PPA", "DATE",
                  "best-paper"):
        require(token in forbidden, "missing forbidden token: {}".format(token))

    if rerun:
        cpu_test = resolve_hw(review["candidate_anchors"]["cpu_tests"])
        cpu = check_call(["/usr/bin/python3.6", str(cpu_test), "-v"],
                         "producer CPU tests")
        require("Ran 6 tests" in cpu.stderr and "OK" in cpu.stderr,
                "producer CPU test marker")
        preflight = resolve_hw(review["candidate_anchors"]["preflight_validator"])
        preflight_run = check_call(["/usr/bin/python3.6", str(preflight)],
                                   "producer preflight")
        require('"status": "PASS_GPU_READY_STATIC_AND_PURE_CPU_ONLY_NOT_EXECUTED"'
                in preflight_run.stdout, "producer preflight marker")
        writer_attack = resolve_hw(evidence["writer_attack_test"])
        writer_test = check_call(["/usr/bin/python3.6", str(writer_attack), "-v"],
                                 "independent writer attacks")
        require("Ran 5 tests" in writer_test.stderr and "OK" in writer_test.stderr,
                "writer attack marker")
        with tempfile.TemporaryDirectory(prefix="m51_review_rerun_") as temp:
            temp_path = Path(temp)
            rebuilt = temp_path / "reconstruction.json"
            checker = resolve_hw(evidence["reconstruction_checker"])
            command = [
                "/usr/bin/python3.6", str(checker),
                "--execution", str(resolve_hw(review["frozen_trace_anchors"]["execution_trace"])),
                "--dual", str(resolve_hw(review["frozen_trace_anchors"]["dual_line_operator_trace"])),
                "--runtime", str(resolve_hw(review["frozen_trace_anchors"]["operator_runtime"])),
                "--profile", str(resolve_hw(review["frozen_trace_anchors"]["profile"])),
                "--plan", str(resolve_hw(review["candidate_anchors"]["target_plan"])),
                "--runner", str(resolve_hw(review["candidate_anchors"]["gpu_runner"])),
                "--writer", str(resolve_hw(review["candidate_anchors"]["streaming_writer"])),
                "--output", str(rebuilt)]
            check_call(command, "independent reconstruction")
            require(sha256(rebuilt) == EXPECTED_RECONSTRUCTION_SHA256,
                    "independent reconstruction rerun drift")
            attacks = temp_path / "attacks.json"
            attack_runner = resolve_hw(evidence["preflight_attack_runner"])
            check_call(["/usr/bin/python3.6", str(attack_runner),
                        "--root", str(ROOT), "--output", str(attacks)],
                       "independent preflight attacks")
            require(sha256(attacks) == EXPECTED_ATTACK_SHA256,
                    "preflight attack rerun drift")

    return {
        "schema": "m51_r1_independent_hammer_review_validator_result_v1",
        "status": "PASS_M51_R1_REVIEW_NO_GO_GPU_READY_PENDING_R2",
        "review_sha256": sha256(REVIEW),
        "score_0_to_100": 82,
        "severity_counts": {"P0": 0, "P1": 2, "P2": 5},
        "verdict": review["verdict"],
        "rerun": bool(rerun),
        "frozen_reconstruction_mismatch_count": 0,
        "gpu_capture_performed": False,
        "performance_claim_admitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.rerun)
    if args.output is not None:
        require(not args.output.exists(), "refusing validator output overwrite")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("PASS M51-r1 review score=82 P0=0 P1=2 P2=5 NO-GO GPU-ready pending runner-r2")


if __name__ == "__main__":
    main()
