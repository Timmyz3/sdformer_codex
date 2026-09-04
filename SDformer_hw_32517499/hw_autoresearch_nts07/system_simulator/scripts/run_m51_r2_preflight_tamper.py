#!/usr/bin/env python3
"""Mutation attacks for the M51 runner-r2 producer preflight boundary."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
VALIDATOR = ROOT / (
    "hw_autoresearch_nts07/system_simulator/scripts/"
    "validate_m51_h67_binary_input_trace_preflight_r2.py")
LAUNCH = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m51_h67_ep35_binary_input_trace_launch_manifest_r2_20260823.json")
CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m51_h67_ep35_binary_input_trace_runner_r2_contract_r1_20260823.json")
PLAN = ROOT / (
    "hw_autoresearch_nts07/results/"
    "m51_h67_ep35_full_network_binary_input_trace_plan_r1_20260823/"
    "m51_h67_ep35_binary_input_trace_target_plan.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command):
    return subprocess.run(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True)


def validator_command(launch, contract=CONTRACT, plan=PLAN, receipt=None):
    command = [
        sys.executable, str(VALIDATOR),
        "--contract", str(contract),
        "--launch-manifest", str(launch),
        "--plan", str(plan),
        "--skip-tests",
    ]
    if receipt is not None:
        command.extend(["--receipt", str(receipt)])
    return command


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def expect_rejected(name, command, results):
    result = run(command)
    require(result.returncode != 0, "tamper admitted: {}".format(name))
    results.append({
        "name": name,
        "return_code": result.returncode,
        "rejected": True,
        "stdout_tail": result.stdout[-300:],
        "stderr_tail": result.stderr[-500:],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), "refusing to overwrite tamper receipt")
    launch = json.loads(LAUNCH.read_text(encoding="utf-8"))
    results = []
    with tempfile.TemporaryDirectory(prefix="m51_r2_tamper_") as directory:
        temp = Path(directory)
        baseline = run(validator_command(LAUNCH))
        require(baseline.returncode == 0 and
                "PASS_RUNNER_R2_STATIC_CPU_READY_FOR_INDEPENDENT_REVIEW_GPU_NOT_RUN"
                in baseline.stdout, "baseline preflight failed")

        mutated = json.loads(json.dumps(launch))
        mutated["status"] = "GPU_EXECUTED_FALSE_CLAIM"
        path = temp / "launch_bad_status.json"
        write_json(path, mutated)
        expect_rejected("mutated_launch_status", validator_command(path), results)

        mutated = json.loads(json.dumps(launch))
        mutated["sources"]["r2_gpu_runner"]["sha256"] = "0" * 64
        path = temp / "launch_bad_runner_sha.json"
        write_json(path, mutated)
        expect_rejected("mutated_runner_source_sha", validator_command(path),
                        results)

        mutated = json.loads(json.dumps(launch))
        mutated["expected_output"]["activation_payload_bytes"] += 1
        path = temp / "launch_bad_population.json"
        write_json(path, mutated)
        expect_rejected("mutated_payload_population", validator_command(path),
                        results)

        runner_path = ROOT / launch["sources"]["r2_gpu_runner"]["path"]
        runner_text = runner_path.read_text(encoding="utf-8")
        require("flat = tensor.detach().view(-1)" in runner_text,
                "runner mutation target missing")
        bad_runner = temp / "capture_r2_bad_contiguous.py"
        bad_runner.write_text(runner_text.replace(
            "flat = tensor.detach().view(-1)",
            "flat = tensor.detach().contiguous().view(-1)", 1),
            encoding="utf-8")
        mutated = json.loads(json.dumps(launch))
        mutated["sources"]["r2_gpu_runner"] = {
            "path": str(bad_runner), "sha256": sha256_path(bad_runner)}
        path = temp / "launch_bad_contiguous.json"
        write_json(path, mutated)
        expect_rejected("whole_call_contiguous_copy", validator_command(path),
                        results)

        writer_path = ROOT / launch["sources"]["r2_streaming_writer"]["path"]
        writer_text = writer_path.read_text(encoding="utf-8")
        require("rglob(\"*.partial\")" in writer_text,
                "writer mutation target missing")
        bad_writer = temp / "writer_r2_no_partial_cleanup.py"
        bad_writer.write_text(writer_text.replace(
            "rglob(\"*.partial\")", "rglob(\"*.never\")", 1),
            encoding="utf-8")
        mutated = json.loads(json.dumps(launch))
        mutated["sources"]["r2_streaming_writer"] = {
            "path": str(bad_writer), "sha256": sha256_path(bad_writer)}
        path = temp / "launch_bad_cleanup.json"
        write_json(path, mutated)
        expect_rejected("removed_partial_cleanup", validator_command(path),
                        results)

        bad_contract = temp / "contract_bad_sync.json"
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        contract["required_repairs"]["M51-P1-01"][
            "final_pre_manifest_cuda_synchronize"] = False
        write_json(bad_contract, contract)
        mutated = json.loads(json.dumps(launch))
        mutated["sources"]["r2_contract"] = {
            "path": str(bad_contract), "sha256": sha256_path(bad_contract)}
        path = temp / "launch_bad_contract.json"
        write_json(path, mutated)
        expect_rejected("removed_final_sync_contract",
                        validator_command(path, contract=bad_contract), results)

        bad_plan = temp / "plan_bad_population.json"
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        plan["population"]["hook_calls"] = 309
        write_json(bad_plan, plan)
        mutated = json.loads(json.dumps(launch))
        mutated["sources"]["target_plan"] = {
            "path": str(bad_plan), "sha256": sha256_path(bad_plan)}
        path = temp / "launch_bad_plan.json"
        write_json(path, mutated)
        expect_rejected("mutated_frozen_target_plan",
                        validator_command(path, plan=bad_plan), results)

        occupied = temp / "occupied_receipt.json"
        occupied.write_bytes(b"do-not-overwrite")
        expect_rejected("occupied_receipt_no_overwrite",
                        validator_command(LAUNCH, receipt=occupied), results)
        require(occupied.read_bytes() == b"do-not-overwrite",
                "occupied receipt was modified")

    receipt = {
        "schema": "m51_runner_r2_preflight_tamper_receipt_v1",
        "status": "PASS_BASELINE_AND_FAIL_CLOSED_TAMPER_ATTACKS_GPU_NOT_RUN",
        "baseline_return_code": baseline.returncode,
        "attack_count": len(results),
        "attacks": results,
        "identity": {
            "launch_manifest_sha256": sha256_path(LAUNCH),
            "contract_sha256": sha256_path(CONTRACT),
            "target_plan_sha256": sha256_path(PLAN),
            "validator_sha256": sha256_path(VALIDATOR),
            "tamper_runner_sha256": sha256_path(Path(__file__).resolve()),
        },
        "execution": {
            "gpu_run_performed": False,
            "remote_contacted": False,
            "checkpoint_opened": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.{}".format(os.getpid()))
    write_json(temporary, receipt)
    os.link(str(temporary), str(output))
    temporary.unlink()
    print("PASS M51-r2 tamper attacks={}".format(len(results)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
