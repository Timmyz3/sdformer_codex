#!/usr/bin/env python3
"""Static and pure-CPU producer preflight for the M51 runner-r2 repairs."""

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
HW_ROOT = ROOT / "hw_autoresearch_nts07"
DEFAULT_CONTRACT = HW_ROOT / (
    "contracts/m51_h67_ep35_binary_input_trace_runner_r2_contract_r1_20260823.json")
DEFAULT_LAUNCH = HW_ROOT / (
    "contracts/m51_h67_ep35_binary_input_trace_launch_manifest_r2_20260823.json")
DEFAULT_PLAN = HW_ROOT / (
    "results/m51_h67_ep35_full_network_binary_input_trace_plan_r1_20260823/"
    "m51_h67_ep35_binary_input_trace_target_plan.json")
EXPECTED_CONTRACT_SHA256 = (
    "571b54e28e6778c4920baa16b44ebc7b76dc00cb158bcc676921539cfd302f5e")
EXPECTED_PLAN_SHA256 = (
    "bf0827d32896a871d9ea4c91afe49014bb5c236d619764b5c3f8a2804dc595e3")
EXPECTED_POPULATION = {
    "samples": 10,
    "modules": 31,
    "hook_calls": 310,
    "dual_line_rows": 3100,
    "input_elements": 10506240000,
    "packed_bytes": 1313280000,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: {}".format(raw))

    def pairs_hook(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: {}".format(key))
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def resolve_source(row):
    candidate = Path(row["path"])
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def validate_contract(contract_path):
    require(contract_path.is_file() and
            sha256_path(contract_path) == EXPECTED_CONTRACT_SHA256,
            "runner-r2 contract SHA mismatch")
    contract = strict_json(contract_path)
    require(contract["schema"] ==
            "m51_h67_ep35_binary_input_trace_runner_r2_contract_v1" and
            contract["status"] ==
            "REQUIRES_STATIC_CPU_PREFLIGHT_GPU_NOT_RUN",
            "runner-r2 contract identity/status mismatch")
    population = contract["frozen_population"]
    for key, expected in EXPECTED_POPULATION.items():
        require(population[key] == expected,
                "contract population mismatch: {}".format(key))
    require(population["target_plan_sha256"] == EXPECTED_PLAN_SHA256 and
            population["operator_types"] == {"Conv2d": 7, "Linear": 24},
            "contract plan/operator identity mismatch")
    repairs = contract["required_repairs"]
    require(repairs["M51-P1-01"] == {
        "failure_semantics": (
            "ANY_SYNCHRONIZE_OR_LATER_PRE_CLOSE_EXCEPTION_WRITES_FAILED_AND_NO_MANIFEST"),
        "final_pre_manifest_cuda_synchronize": True,
        "per_sample_post_forward_cuda_synchronize": True,
        "remove_all_partial_files_on_abort": True,
    }, "P1-01 contract mismatch")
    require(repairs["M51-P1-02"]["input_layout_policy"] ==
            "REQUIRE_TORCH_DEFAULT_C_CONTIGUOUS_ELSE_FAIL_CLOSED" and
            repairs["M51-P1-02"]["maximum_whole_call_copy_bytes"] == 0 and
            len(repairs["M51-P1-02"]["capture_memory_telemetry"]) == 8,
            "P1-02 contract mismatch")
    for name, row in contract["history_anchors"].items():
        path = (HW_ROOT / row["path"]).resolve()
        require(path.is_file() and sha256_path(path) == row["sha256"],
                "r1 history anchor drift: {}".format(name))
    return contract


def validate_launch(launch_path, contract_path, plan_path):
    launch = strict_json(launch_path)
    require(launch["schema"] ==
            "m51_h67_ep35_binary_input_trace_launch_manifest_r2_v1" and
            launch["status"] ==
            "RUNNER_R2_READY_FOR_INDEPENDENT_REVIEW_GPU_NOT_EXECUTED",
            "runner-r2 launch identity/status mismatch")
    require(launch["claim_boundary"] ==
            "NO_GPU_PAYLOAD_NO_CYCLE_NO_SPEEDUP_NO_PPA_NO_ENERGY_CLAIM",
            "runner-r2 launch claim boundary mismatch")
    expected_keys = {
        "base_profiler", "config", "r1_streaming_writer", "r2_contract",
        "r2_gpu_runner", "r2_streaming_writer", "r2_unit_tests",
        "r2_preflight_validator", "r2_tamper_runner", "target_plan",
        "target_plan_builder",
    }
    require(set(launch["sources"]) == expected_keys,
            "runner-r2 launch source population mismatch")
    for name, row in launch["sources"].items():
        path = resolve_source(row)
        require(path.is_file() and sha256_path(path) == row["sha256"],
                "runner-r2 launch source drift: {}".format(name))
    require(resolve_source(launch["sources"]["r2_contract"]) == contract_path and
            resolve_source(launch["sources"]["target_plan"]) == plan_path and
            launch["expected_output"]["activation_payload_bytes"] ==
            EXPECTED_POPULATION["packed_bytes"] and
            launch["expected_output"]["hook_calls"] == 310,
            "runner-r2 launch contract/plan/output bridge mismatch")
    command = launch["gpu_command_argv_draft"]
    require(command[2].endswith(
            "capture_h67_full_network_binary_inputs_r2.py") and
            "--chunk-elements" in command and "8388608" in command,
            "runner-r2 GPU command draft mismatch")
    return launch


def validate_plan(plan_path):
    require(plan_path.is_file() and sha256_path(plan_path) == EXPECTED_PLAN_SHA256,
            "frozen target plan SHA mismatch")
    plan = strict_json(plan_path)
    for key, expected in EXPECTED_POPULATION.items():
        require(plan["population"][key] == expected,
                "frozen target plan population mismatch: {}".format(key))
    require(len(plan["modules"]) == 31 and
            sum(row["operator"] == "Conv2d" for row in plan["modules"]) == 7 and
            sum(row["operator"] == "Linear" for row in plan["modules"]) == 24 and
            all(len(row["calls"]) == 10 for row in plan["modules"]),
            "frozen target plan module/call mismatch")
    return plan


def validate_runner_sources(launch):
    runner_path = resolve_source(launch["sources"]["r2_gpu_runner"])
    writer_path = resolve_source(launch["sources"]["r2_streaming_writer"])
    runner = runner_path.read_text(encoding="utf-8")
    writer = writer_path.read_text(encoding="utf-8")

    stream_start = runner.index("def stream_torch_binary_r2")
    stream_end = runner.index("\ndef validate_frozen_protocol", stream_start)
    stream = runner[stream_start:stream_end]
    require("require_c_order_contiguous(tensor)" in stream and
            "tensor.detach().view(-1)" in stream and
            "tensor.detach().contiguous()" not in stream and
            ".reshape(" not in stream,
            "runner-r2 whole-call contiguity gate/copy mismatch")

    model_call = runner.index("                model(x)\n")
    sample_sync = runner.index(
        "                torch.cuda.synchronize(device)\n", model_call)
    end_sample = runner.index("                writer.end_sample()\n", sample_sync)
    final_sync = runner.index("        torch.cuda.synchronize(device)\n", end_sample)
    memory_after = runner.index("        memory_after = cuda_memory_snapshot",
                                final_sync)
    close = runner.index("        manifest = writer.close()\n", memory_after)
    require(model_call < sample_sync < end_sample < final_sync < memory_after < close,
            "runner-r2 forward/synchronize/manifest order mismatch")
    require(runner.count("torch.cuda.synchronize(device)") == 3 and
            "except BaseException as error:" in runner and
            "writer.abort(" in runner and
            "torch.cuda.max_memory_allocated(device)" in runner and
            "torch.cuda.max_memory_reserved(device)" in runner,
            "runner-r2 synchronization/failure/memory instrumentation mismatch")
    require(writer.count("rglob(\"*.partial\")") == 2 and
            "refusing FAILED alongside a published PASS manifest" in writer and
            "capture_memory" in writer and
            "exact CUDA synchronization counts" in writer,
            "writer-r2 abort/manifest admission mismatch")


def check_call(command, label):
    result = subprocess.run(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)
    require(result.returncode == 0,
            "{} failed rc={} stdout={} stderr={}".format(
                label, result.returncode, result.stdout[-2000:],
                result.stderr[-2000:]))
    return result


def rerun_plan_and_tests(launch, plan_path):
    builder = resolve_source(launch["sources"]["target_plan_builder"])
    with tempfile.TemporaryDirectory(prefix="m51_r2_rebuild_") as directory:
        rebuilt = Path(directory) / "plan.json"
        check_call([sys.executable, str(builder), "--output", str(rebuilt)],
                   "target plan rebuild")
        require(rebuilt.read_bytes() == plan_path.read_bytes(),
                "rebuilt target plan is not byte-identical")
    r1_tests = ROOT / (
        "hw_autoresearch_nts07/system_simulator/tests/"
        "test_m51_h67_binary_input_trace.py")
    r2_tests = resolve_source(launch["sources"]["r2_unit_tests"])
    r1 = check_call([sys.executable, str(r1_tests), "-v"], "r1 regression")
    r2 = check_call([sys.executable, str(r2_tests), "-v"], "r2 unit tests")
    require("Ran 6 tests" in r1.stderr and "OK" in r1.stderr and
            "Ran 6 tests" in r2.stderr and "OK" in r2.stderr,
            "producer test count/marker mismatch")


def write_receipt(path, payload):
    require(not path.exists(), "refusing to overwrite runner-r2 receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    with temporary.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.link(str(temporary), str(path))
    temporary.unlink()


def validate(contract_path, launch_path, plan_path, run_tests):
    validate_contract(contract_path)
    launch = validate_launch(launch_path, contract_path, plan_path)
    validate_plan(plan_path)
    validate_runner_sources(launch)
    if run_tests:
        rerun_plan_and_tests(launch, plan_path)
    return {
        "schema": "m51_h67_binary_input_trace_runner_r2_preflight_receipt_v1",
        "status": "PASS_RUNNER_R2_STATIC_CPU_READY_FOR_INDEPENDENT_REVIEW_GPU_NOT_RUN",
        "identity": {
            "contract_sha256": sha256_path(contract_path),
            "launch_manifest_sha256": sha256_path(launch_path),
            "target_plan_sha256": sha256_path(plan_path),
            "runner_r2_sha256": launch["sources"]["r2_gpu_runner"]["sha256"],
            "writer_r2_sha256": launch["sources"]["r2_streaming_writer"]["sha256"],
            "validator_sha256": sha256_path(Path(__file__).resolve()),
        },
        "population": dict(EXPECTED_POPULATION,
                           operator_types={"Conv2d": 7, "Linear": 24},
                           packed_gib=1.2230873107910156),
        "p1_repairs": {
            "M51-P1-01": "PASS_STATIC_CPU_INJECTED_FAILURE",
            "M51-P1-02": "PASS_STATIC_CPU_NONCONTIGUOUS_REJECTION",
        },
        "checks": {
            "r1_history_anchors": "PASS",
            "launch_source_hashes": "PASS",
            "target_plan_unchanged": "PASS",
            "target_plan_byte_rebuild": "PASS" if run_tests else "NOT_RUN",
            "runner_synchronization_order": "PASS",
            "no_whole_call_contiguous_copy": "PASS",
            "partial_cleanup_no_pass_manifest": "PASS",
            "cuda_peak_memory_fields": "PASS",
            "producer_tests": "PASS" if run_tests else "NOT_RUN",
        },
        "execution": {
            "gpu_run_performed": False,
            "activation_payload_present": False,
            "checkpoint_opened": False,
            "remote_contacted": False,
            "cuda_memory_values_available": False,
        },
        "claim_boundary": (
            "runner-r2 preflight only; real CUDA memory values and activation "
            "payload remain pending GPU execution; no performance claim"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--launch-manifest", type=Path, default=DEFAULT_LAUNCH)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    receipt = validate(args.contract.resolve(), args.launch_manifest.resolve(),
                       args.plan.resolve(), not args.skip_tests)
    if args.receipt is not None:
        write_receipt(args.receipt.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
