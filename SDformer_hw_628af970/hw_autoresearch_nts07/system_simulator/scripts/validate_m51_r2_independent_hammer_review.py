#!/usr/bin/env python3
"""Fail-closed validator for the independent M51 runner-r2 hammer review."""

from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
REVIEW = (HW_ROOT / "results/m51_r2_independent_hammer_20260823"
          / "m51_r2_independent_hammer_review.json")
EXPECTED_REVIEW_SHA256 = (
    "adea96e27256a84e1084d2122594fc279716bfab9e0da059efbf8d994f6cf537")
EXPECTED_RECONSTRUCTION_SHA256 = (
    "96bc764041ad8afa0d9ad3fd10174bcd9ee5e973cb20054c73cb554edfbe6188")
EXPECTED_PREFLIGHT_SHA256 = (
    "8afd9058c599eae187ea299dd9c6f457abfd854f4e1c205743ac977a2df849ce")
EXPECTED_TAMPER_SHA256 = (
    "2508f5bc4d48ae5e96c0d9289f9af0c83ffaa8d945c437c0ad468526d3568182")
PYTHON = "/usr/bin/python3.6"


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


def resolve_hw(row):
    return (HW_ROOT / row["path"]).resolve()


def check_call(command, label):
    result = subprocess.run(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)
    require(result.returncode == 0,
            "{} failed rc={} stdout={} stderr={}".format(
                label, result.returncode, result.stdout[-1500:],
                result.stderr[-1500:]))
    return result


def validate_static_runner(review):
    runner_path = resolve_hw(review["candidate_anchors"]["runner_r2"])
    writer_path = resolve_hw(review["candidate_anchors"]["writer_r2"])
    runner = runner_path.read_text(encoding="utf-8")
    writer = writer_path.read_text(encoding="utf-8")
    stream_start = runner.index("def stream_torch_binary_r2")
    stream_end = runner.index("\ndef validate_frozen_protocol", stream_start)
    stream = runner[stream_start:stream_end]
    require("require_c_order_contiguous(tensor)" in stream and
            "tensor.detach().view(-1)" in stream and
            ".contiguous(" not in stream and ".reshape(" not in stream,
            "whole-call layout/copy repair drift")
    model = runner.index("                model(x)\n")
    sample_sync = runner.index(
        "                torch.cuda.synchronize(device)\n", model)
    sample_end = runner.index("                writer.end_sample()\n",
                              sample_sync)
    final_sync = runner.index("        torch.cuda.synchronize(device)\n",
                              sample_end)
    memory = runner.index("        memory_after = cuda_memory_snapshot",
                          final_sync)
    close = runner.index("        manifest = writer.close()\n", memory)
    abort = runner.index("    except BaseException as error:", close)
    require(model < sample_sync < sample_end < final_sync < memory < close < abort,
            "runner lifecycle order drift")
    require(runner.count("torch.cuda.synchronize(device)") == 3 and
            "failure_memory_snapshot(device)" in runner and
            "for handle in handles:" in runner and "handle.remove()" in runner,
            "runner synchronization/cleanup source drift")
    require("per_sample_post_forward\": 10" in writer and
            "final_pre_manifest\": 1" in writer and
            writer.count("rglob(\"*.partial\")") == 2 and
            "refusing FAILED alongside a published PASS manifest" in writer,
            "writer close/abort admission drift")


def validate_review(rerun):
    require(REVIEW.is_file() and sha256(REVIEW) == EXPECTED_REVIEW_SHA256,
            "review SHA drift")
    review = strict_json(REVIEW)
    require(review["schema"] == "m51_r2_independent_hammer_review_v1" and
            review["status"] ==
            "GO_GPU_CAPTURE_P1_REPAIRS_CLOSED_STATIC_CPU_MOCK_CUDA" and
            review["verdict"] ==
            "GO_M51_R2_REAL_GPU_CAPTURE_WITH_P2_CAVEATS",
            "review identity/verdict drift")
    require(review["score_0_to_100"] == 94 and
            review["severity_counts"] == {"P0": 0, "P1": 0, "P2": 4} and
            len(review["findings"]) == 4 and
            all(row["severity"] == "P2" for row in review["findings"]),
            "review score/severity drift")
    require(set(review["p1_closure"]) == {"M51-P1-01", "M51-P1-02"} and
            all(row["status"] == "CLOSED_STATIC_CPU_MOCK_CUDA"
                for row in review["p1_closure"].values()),
            "P1 closure drift")
    require(len(review["admitted_claims"]) == 3 and
            len(review["not_admitted"]) == 3,
            "review claim population drift")
    forbidden = " ".join(review["not_admitted"])
    for token in ("GPU", "activation payload", "cycle", "speedup", "PPA",
                  "power", "energy", "DATE", "best-paper"):
        require(token in forbidden, "missing claim-boundary token: {}".format(token))

    for group in ("candidate_anchors", "independent_evidence"):
        for name, row in review[group].items():
            path = resolve_hw(row)
            require(path.is_file() and sha256(path) == row["sha256"],
                    "{} drift: {}".format(group, name))

    launch = strict_json(resolve_hw(
        review["candidate_anchors"]["launch_manifest_r2"]))
    require(launch["status"] ==
            "RUNNER_R2_READY_FOR_INDEPENDENT_REVIEW_GPU_NOT_EXECUTED" and
            launch["claim_boundary"] ==
            "NO_GPU_PAYLOAD_NO_CYCLE_NO_SPEEDUP_NO_PPA_NO_ENERGY_CLAIM",
            "launch execution boundary drift")
    for name, row in launch["sources"].items():
        path = Path(row["path"])
        path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
        require(path.is_file() and sha256(path) == row["sha256"],
                "launch source drift: {}".format(name))

    preflight = strict_json(resolve_hw(
        review["independent_evidence"]["producer_preflight_rerun"]))
    require(sha256(resolve_hw(review["independent_evidence"][
                "producer_preflight_rerun"])) == EXPECTED_PREFLIGHT_SHA256 and
            preflight["status"] ==
            "PASS_RUNNER_R2_STATIC_CPU_READY_FOR_INDEPENDENT_REVIEW_GPU_NOT_RUN" and
            preflight["execution"] == {
                "activation_payload_present": False,
                "checkpoint_opened": False,
                "cuda_memory_values_available": False,
                "gpu_run_performed": False,
                "remote_contacted": False,
            }, "preflight result/boundary drift")
    tamper = strict_json(resolve_hw(
        review["independent_evidence"]["producer_tamper_rerun"]))
    require(sha256(resolve_hw(review["independent_evidence"][
                "producer_tamper_rerun"])) == EXPECTED_TAMPER_SHA256 and
            tamper["status"] ==
            "PASS_BASELINE_AND_FAIL_CLOSED_TAMPER_ATTACKS_GPU_NOT_RUN" and
            tamper["attack_count"] == 8 and
            all(row["rejected"] for row in tamper["attacks"]),
            "tamper result drift")

    reconstruction_path = resolve_hw(
        review["independent_evidence"]["r1_plan_identity_reconstruction"])
    require(sha256(reconstruction_path) == EXPECTED_RECONSTRUCTION_SHA256,
            "r1 plan reconstruction identity drift")
    reconstruction = strict_json(reconstruction_path)
    summary = reconstruction["reconstruction"]
    require((summary["modules"], summary["operator_types"],
             summary["hook_calls"], summary["dual_line_rows"],
             summary["input_elements_bits"], summary["packed_bytes"]) ==
            (31, {"Conv2d": 7, "Linear": 24}, 310, 3100,
             10506240000, 1313280000) and
            len(reconstruction["negative_mutations_rejected"]) == 10 and
            reconstruction["producer_modules_imported"] == [],
            "r1 plan reconstruction population/isolation drift")
    stderr = resolve_hw(review["independent_evidence"][
        "mock_cuda_lifecycle_stderr"]).read_text(encoding="utf-8")
    require("Ran 6 tests" in stderr and "OK" in stderr,
            "mock CUDA saved result marker drift")
    mock = review["mock_cuda_results"]
    require((mock["pre_capture_sync"],
             mock["per_sample_post_forward_sync"],
             mock["final_pre_manifest_sync"],
             mock["total_sync_calls_success"], mock["tests_passed"]) ==
            (1, 10, 1, 12, 6) and
            mock["noncontiguous_detach_calls_before_rejection"] == 0 and
            mock["contiguous_empty_view_copy_calls"] == 0 and
            len(mock["baseexception_injections"]) == 3,
            "mock CUDA evidence summary drift")
    validate_static_runner(review)

    if rerun:
        r2_test = resolve_hw(review["candidate_anchors"]["unit_tests_r2"])
        producer_test = check_call([PYTHON, str(r2_test), "-v"],
                                   "producer r2 tests")
        require("Ran 6 tests" in producer_test.stderr and
                "OK" in producer_test.stderr,
                "producer r2 test marker")
        preflight_path = resolve_hw(
            review["candidate_anchors"]["preflight_validator_r2"])
        preflight_run = check_call([PYTHON, str(preflight_path)],
                                   "producer r2 preflight")
        require("PASS_RUNNER_R2_STATIC_CPU_READY_FOR_INDEPENDENT_REVIEW_GPU_NOT_RUN"
                in preflight_run.stdout, "producer preflight marker")
        mock_path = resolve_hw(
            review["independent_evidence"]["mock_cuda_lifecycle_test"])
        mock_run = check_call([PYTHON, str(mock_path), "-v"],
                              "independent mock CUDA lifecycle")
        require("Ran 6 tests" in mock_run.stderr and "OK" in mock_run.stderr,
                "mock CUDA lifecycle marker")
        with tempfile.TemporaryDirectory(prefix="m51_r2_review_rerun_") as temp:
            temp_path = Path(temp)
            tamper_path = temp_path / "tamper.json"
            tamper_runner = resolve_hw(
                review["candidate_anchors"]["tamper_runner_r2"])
            check_call([PYTHON, str(tamper_runner), "--output",
                        str(tamper_path)], "producer r2 tamper")
            require(sha256(tamper_path) == EXPECTED_TAMPER_SHA256,
                    "tamper rerun drift")
            rebuilt = temp_path / "reconstruction.json"
            checker = resolve_hw(
                review["independent_evidence"]["reconstruction_checker"])
            command = [
                PYTHON, str(checker),
                "--execution", str(HW_ROOT / (
                    "results/h67_ep35_full_network_ordered_trace_s10_20260821/"
                    "execution_trace.csv")),
                "--dual", str(HW_ROOT / (
                    "results/h67_ep35_full_network_ordered_trace_s10_20260821/"
                    "dual_line_operator_trace.csv")),
                "--runtime", str(HW_ROOT / (
                    "results/h67_ep35_full_network_ordered_trace_s10_20260821/"
                    "operator_runtime.csv")),
                "--profile", str(HW_ROOT / (
                    "results/h67_ep35_full_network_ordered_trace_s10_20260821/"
                    "nts11_hardware_p0_profile.json")),
                "--plan", str(resolve_hw(
                    review["candidate_anchors"]["target_plan"])),
                "--runner", str(ROOT / (
                    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
                    "capture_h67_full_network_binary_inputs.py")),
                "--writer", str(ROOT / (
                    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
                    "h67_binary_input_trace.py")),
                "--output", str(rebuilt),
            ]
            check_call(command, "independent r1 plan reconstruction")
            require(sha256(rebuilt) == EXPECTED_RECONSTRUCTION_SHA256,
                    "reconstruction rerun drift")

    return {
        "schema": "m51_r2_independent_hammer_review_validator_result_v1",
        "status": "PASS_M51_R2_GO_GPU_CAPTURE_P1_CLOSED",
        "review_sha256": sha256(REVIEW),
        "score_0_to_100": 94,
        "severity_counts": {"P0": 0, "P1": 0, "P2": 4},
        "verdict": review["verdict"],
        "rerun": bool(rerun),
        "mock_cuda_sync_counts": {
            "before_capture": 1,
            "per_sample_post_forward": 10,
            "final_pre_manifest": 1,
        },
        "r1_plan_mismatch_count": 0,
        "gpu_capture_performed": False,
        "performance_claim_admitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_review(args.rerun)
    if args.output is not None:
        require(not args.output.exists(), "refusing validator output overwrite")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    print("PASS M51-r2 review score=94 P0=0 P1=0 P2=4 GO GPU-capture")


if __name__ == "__main__":
    main()
