#!/usr/bin/env python3
"""Fail-closed static/CPU preflight for the M51 GPU capture handoff."""

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
DEFAULT_PLAN = ROOT / (
    "hw_autoresearch_nts07/results/"
    "m51_h67_ep35_full_network_binary_input_trace_plan_r1_20260823/"
    "m51_h67_ep35_binary_input_trace_target_plan.json")
DEFAULT_LAUNCH = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m51_h67_ep35_binary_input_trace_launch_manifest_r1_20260823.json")
EXPECTED_PLAN_SHA256 = (
    "bf0827d32896a871d9ea4c91afe49014bb5c236d619764b5c3f8a2804dc595e3")
EXPECTED_CONTRACT_SHA256 = (
    "60dd1d2da9bbfe348965922b50de227764f3f9714a0651b56c0a8f4167db3411")
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


def product(shape):
    require(isinstance(shape, list) and shape and
            all(isinstance(item, int) and not isinstance(item, bool) and item > 0
                for item in shape), "invalid positive tensor shape")
    result = 1
    for item in shape:
        result *= item
    return result


def validate_launch(launch_path):
    launch = strict_json(launch_path)
    require(launch["schema"] ==
            "m51_h67_ep35_binary_input_trace_launch_manifest_v1",
            "launch manifest schema mismatch")
    require(launch["status"] == "GPU_READY_NOT_EXECUTED",
            "launch status must remain unexecuted")
    for name, row in launch["sources"].items():
        path = (ROOT / row["path"]).resolve()
        require(path.is_file(), "missing launch source: {}".format(name))
        require(sha256_path(path) == row["sha256"],
                "launch source SHA mismatch: {}".format(name))
    require(launch["claim_boundary"] ==
            "NO_GPU_PAYLOAD_NO_OUTPUT_NO_CYCLE_NO_SPEEDUP_NO_PPA_NO_ENERGY_CLAIM",
            "launch claim boundary drift")
    return launch


def validate_plan(plan_path):
    require(plan_path.is_file() and sha256_path(plan_path) == EXPECTED_PLAN_SHA256,
            "target plan SHA mismatch")
    plan = strict_json(plan_path)
    require(plan["schema"] ==
            "m51_h67_ep35_binary_input_trace_target_plan_v1",
            "target plan schema mismatch")
    require(plan["identity"]["contract_sha256"] == EXPECTED_CONTRACT_SHA256,
            "contract SHA bridge mismatch")
    require(plan["status"] ==
            "READY_FOR_GPU_CAPTURE_NO_ACTIVATION_PAYLOAD_ON_THIS_HOST",
            "target plan status drift")
    for key, expected in EXPECTED_POPULATION.items():
        require(plan["population"][key] == expected,
                "population mismatch: {}".format(key))
    require(plan["packing"] == {
        "layout": "C_ORDER_FLAT",
        "bit_order": "LITTLE_WITHIN_BYTE",
        "file_granularity": "ONE_RAW_FILE_PER_HOOK_CALL",
        "tail_padding_high_bits_zero": True,
        "float_payload_retained": False,
        "delta_payload_retained": False,
    }, "packing contract mismatch")
    samples = plan["samples"]
    require(len(samples) == 10 and
            [row["sample_id"] for row in samples] == list(range(10)),
            "sample population/order mismatch")
    sample_by_id = dict((row["sample_id"], row) for row in samples)
    modules = plan["modules"]
    require(len(modules) == 31 and
            [row["module_index"] for row in modules] == list(range(31)) and
            len(set(row["name"] for row in modules)) == 31,
            "module population/index/name mismatch")
    operator_count = {"Conv2d": 0, "Linear": 0}
    flattened = []
    paths = set()
    total_elements = total_bytes = 0
    sample_execution_indices = dict((sample_id, []) for sample_id in range(10))
    for module in modules:
        operator = module["operator"]
        require(operator in operator_count, "unsupported target operator")
        operator_count[operator] += 1
        require(module["expected_hook_calls"] == 10 and
                module["expected_weight_elements"] > 0 and
                module["runtime_weight_and_bias_content_sha256_required"] is True,
                "module runtime identity contract mismatch")
        calls = module["calls"]
        require(len(calls) == 10 and
                [row["sample_id"] for row in calls] == list(range(10)),
                "module call population/order mismatch")
        for call in calls:
            sample_id = call["sample_id"]
            expected_sample = sample_by_id[sample_id]
            require(call["sample_key"] == expected_sample["sample_key"] and
                    call["sequence_key"] == expected_sample["sequence_key"],
                    "call/sample identity mismatch")
            require(call["target_order_index"] == module["module_index"] and
                    call["dual_line_operator_call_index"] == sample_id and
                    call["dual_line_temporal_steps"] == list(range(10)),
                    "call order/timestep identity mismatch")
            input_elements = product(call["input_shape"])
            require(input_elements == call["input_elements"] and
                    product(call["output_shape"]) == call["output_elements"] and
                    call["packed_bytes"] == (input_elements + 7) // 8,
                    "call tensor size/packing mismatch")
            expected_path = (
                "calls/s{:02d}_m{:02d}_c{:03d}.activation.le.bitpack".format(
                    sample_id, module["module_index"],
                    call["frozen_execution_call_index"]))
            require(call["relative_output_path"] == expected_path and
                    expected_path not in paths,
                    "unsafe/duplicate call output path")
            paths.add(expected_path)
            sample_execution_indices[sample_id].append(
                call["frozen_execution_call_index"])
            total_elements += input_elements
            total_bytes += call["packed_bytes"]
            flattened.append(dict(call, name=module["name"], operator=operator))
    require(operator_count == {"Conv2d": 7, "Linear": 24},
            "Conv2d/Linear population mismatch")
    require(total_elements == EXPECTED_POPULATION["input_elements"] and
            total_bytes == EXPECTED_POPULATION["packed_bytes"] and
            len(paths) == EXPECTED_POPULATION["hook_calls"],
            "aggregate target payload mismatch")
    require(all(indices == sorted(indices) and len(set(indices)) == 31
                for indices in sample_execution_indices.values()),
            "per-sample frozen execution order mismatch")
    require(plan["expected_call_records"] == flattened,
            "flattened expected call record mismatch")
    return plan, operator_count


def rebuild_and_compare(plan_path, launch):
    builder_row = launch["sources"]["target_plan_builder"]
    builder = (ROOT / builder_row["path"]).resolve()
    with tempfile.TemporaryDirectory(prefix="m51_plan_rebuild_") as directory:
        rebuilt = Path(directory) / "plan.json"
        subprocess.check_call([
            sys.executable, str(builder), "--output", str(rebuilt)])
        require(rebuilt.read_bytes() == plan_path.read_bytes(),
                "rebuilt plan is not byte-identical")
        return sha256_path(rebuilt)


def run_cpu_tests(launch):
    test_path = (ROOT / launch["sources"]["cpu_unit_tests"]["path"]).resolve()
    subprocess.check_call([sys.executable, str(test_path), "-v"])


def write_receipt(path, payload):
    require(not path.exists(), "refusing to overwrite preflight receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    with temporary.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--launch-manifest", type=Path, default=DEFAULT_LAUNCH)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    launch_path = args.launch_manifest.resolve()
    launch = validate_launch(launch_path)
    plan, operator_count = validate_plan(plan_path)
    rebuilt_sha = rebuild_and_compare(plan_path, launch)
    run_cpu_tests(launch)
    receipt = {
        "schema": "m51_h67_ep35_binary_input_trace_preflight_receipt_v1",
        "status": "PASS_GPU_READY_STATIC_AND_PURE_CPU_ONLY_NOT_EXECUTED",
        "identity": {
            "launch_manifest_path": str(launch_path),
            "launch_manifest_sha256": sha256_path(launch_path),
            "target_plan_path": str(plan_path),
            "target_plan_sha256": sha256_path(plan_path),
            "rebuilt_target_plan_sha256": rebuilt_sha,
            "validator_sha256": sha256_path(Path(__file__).resolve()),
        },
        "population": dict(EXPECTED_POPULATION,
                           operator_types=operator_count,
                           packed_gib=float(EXPECTED_POPULATION["packed_bytes"])
                           / (1 << 30)),
        "checks": {
            "launch_source_hashes": "PASS",
            "target_plan_byte_rebuild": "PASS",
            "shape_call_sample_order": "PASS",
            "dual_line_timestep_population": "PASS",
            "packing_contract": "PASS",
            "pure_cpu_unit_tests": "PASS",
        },
        "execution": {
            "gpu_run_performed": False,
            "activation_payload_present": False,
            "checkpoint_opened": False,
            "remote_contacted": False,
        },
        "claim_boundary": (
            "preflight only; no captured activation, output, cycle, speedup, "
            "RTL, PPA, energy or full-system claim"),
    }
    if args.receipt is not None:
        write_receipt(args.receipt.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
