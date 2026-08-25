#!/usr/bin/env python3
"""Build the exact 31-module M51 H67 binary-input capture plan."""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW_ROOT / (
    "contracts/m51_h67_ep35_full_network_binary_input_trace_contract_r1_20260823.json")
EXPECTED_CONTRACT_SHA256 = (
    "60dd1d2da9bbfe348965922b50de227764f3f9714a0651b56c0a8f4167db3411")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
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


def read_csv(path):
    with Path(path).open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames and len(reader.fieldnames) ==
                len(set(reader.fieldnames)), "duplicate/missing CSV header")
        return list(reader)


def parse_shape(raw, label):
    value = json.loads(raw)
    require(isinstance(value, list) and value and
            all(isinstance(item, int) and not isinstance(item, bool) and item > 0
                for item in value), "invalid {} shape".format(label))
    return value


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


def build():
    require(sha256(CONTRACT) == EXPECTED_CONTRACT_SHA256,
            "M51 contract SHA drift")
    contract = read_json(CONTRACT)
    require(contract["schema"] ==
            "m51_h67_ep35_full_network_binary_input_trace_contract_v1",
            "M51 contract schema drift")
    paths = {}
    for name, item in contract["inputs"].items():
        path = (HW_ROOT / item["path"]).resolve()
        require(path.is_file() and sha256(path) == item["sha256"],
                "M51 input identity drift: {}".format(name))
        paths[name] = path
    m22 = read_json(paths["m22_ordered_trace_identity_manifest"])
    profile = read_json(paths["h67_profile"])
    execution = read_csv(paths["h67_execution_trace"])
    dual = read_csv(paths["h67_dual_line_operator_trace"])
    runtime = read_csv(paths["h67_operator_runtime"])
    h67_identity = m22["identities"]["h67_ep35"]
    require(h67_identity["files_sha256"]["execution_trace.csv"] ==
            contract["inputs"]["h67_execution_trace"]["sha256"] and
            h67_identity["files_sha256"]["dual_line_operator_trace.csv"] ==
            contract["inputs"]["h67_dual_line_operator_trace"]["sha256"] and
            h67_identity["files_sha256"]["operator_runtime.csv"] ==
            contract["inputs"]["h67_operator_runtime"]["sha256"] and
            h67_identity["files_sha256"]["nts11_hardware_p0_profile.json"] ==
            contract["inputs"]["h67_profile"]["sha256"],
            "M51 M22/file identity bridge drift")
    require(profile["ordered_trace"] is True and
            profile["dual_line_trace"] is True and profile["samples"] == 10,
            "M51 profile scope drift")
    require(profile["artifact_identity"]["config_sha256"] ==
            contract["inputs"]["h67_config"]["sha256"] and
            profile["artifact_identity"]["checkpoint_sha256"] ==
            contract["checkpoint_identity"]["expected_sha256"] and
            profile["artifact_identity"]["checkpoint_size"] ==
            contract["checkpoint_identity"]["expected_size_bytes"],
            "M51 config/checkpoint bridge drift")

    exact_dual = [row for row in dual
                  if row["status"] == "PASS_EXACT_SOURCE_WORK"]
    target_names = set(row["name"] for row in exact_dual)
    require(len(exact_dual) == 3100 and len(target_names) == 31,
            "M51 exact dual population drift")
    require(all(row["operator"] in ("Conv2d", "Linear")
                for row in exact_dual), "M51 unexpected target operator")
    require(not [row for row in dual
                 if row["name"] in target_names and
                 row["status"] != "PASS_EXACT_SOURCE_WORK"],
            "M51 target has mixed dual-line qualification")
    execution_targets = [row for row in execution
                         if row["kind"] == "operator" and
                         row["name"] in target_names]
    require(len(execution_targets) == 310,
            "M51 target execution population drift")
    runtime_by_name = {}
    for row in runtime:
        require(row["name"] not in runtime_by_name,
                "duplicate operator runtime name")
        runtime_by_name[row["name"]] = row

    sample_records = profile["summary"]["sample_records"]
    require([row["sample_id"] for row in sample_records] == list(range(10)),
            "M51 profile sample order drift")
    sample_plan = []
    for row in sample_records:
        sample_id = int(row["sample_id"])
        ex_keys = set((item["sample_key"], item["sequence_key"])
                      for item in execution_targets
                      if int(item["sample_id"]) == sample_id)
        dual_keys = set((item["sample_key"], item["sequence_key"])
                        for item in exact_dual
                        if int(item["sample_id"]) == sample_id)
        expected_key = (row["sample_key"], row["sequence_key"])
        require(ex_keys == {expected_key} and dual_keys == {expected_key},
                "M51 sample identity mismatch")
        sample_plan.append({
            "sample_id": sample_id,
            "sample_key": row["sample_key"],
            "sequence_key": row["sequence_key"],
        })

    first_sample = sorted(
        [row for row in execution_targets if row["sample_id"] == "0"],
        key=lambda row: int(row["call_index"]))
    require(len(first_sample) == 31 and
            len(set(row["name"] for row in first_sample)) == 31,
            "M51 first-sample target order drift")
    module_order = [row["name"] for row in first_sample]
    modules = []
    total_elements = total_packed = 0
    expected_call_records = []
    for module_index, name in enumerate(module_order):
        ex_rows = sorted([row for row in execution_targets if row["name"] == name],
                         key=lambda row: int(row["sample_id"]))
        dual_rows = [row for row in exact_dual if row["name"] == name]
        require(len(ex_rows) == 10 and len(dual_rows) == 100,
                "M51 per-module population drift")
        operator = ex_rows[0]["operator"]
        require(operator in ("Conv2d", "Linear") and
                all(row["operator"] == operator for row in ex_rows + dual_rows),
                "M51 per-module operator drift")
        runtime_row = runtime_by_name.get(name)
        require(runtime_row is not None and
                runtime_row["operator"] == operator and
                int(runtime_row["calls"]) == 10,
                "M51 runtime module identity drift")
        calls = []
        for sample_id, row in enumerate(ex_rows):
            require(int(row["sample_id"]) == sample_id,
                    "M51 execution sample order drift")
            input_shape = parse_shape(row["input_shape"], "input")
            output_shape = parse_shape(row["output_shape"], "output")
            input_elements = int(row["input_elements"])
            output_elements = int(row["output_elements"])
            require(product(input_shape) == input_elements and
                    product(output_shape) == output_elements,
                    "M51 execution shape/element drift")
            sample_dual = [item for item in dual_rows
                           if int(item["sample_id"]) == sample_id]
            require(len(sample_dual) == 10 and
                    sorted(int(item["temporal_step"]) for item in sample_dual) ==
                    list(range(10)) and
                    set(int(item["operator_call_index"]) for item in sample_dual) ==
                    {sample_id},
                    "M51 dual timestep/call population drift")
            nonempty_shapes = set(item["input_shape"] for item in sample_dual
                                  if item["input_shape"])
            require(not nonempty_shapes or nonempty_shapes == {row["input_shape"]},
                    "M51 dual/execution input shape drift")
            packed_bytes = (input_elements + 7) // 8
            call = {
                "sample_id": sample_id,
                "sample_key": row["sample_key"],
                "sequence_key": row["sequence_key"],
                "frozen_execution_call_index": int(row["call_index"]),
                "target_order_index": module_index,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "input_elements": input_elements,
                "output_elements": output_elements,
                "packed_bytes": packed_bytes,
                "dual_line_temporal_steps": list(range(10)),
                "dual_line_operator_call_index": sample_id,
                "relative_output_path": (
                    "calls/s{:02d}_m{:02d}_c{:03d}.activation.le.bitpack".format(
                        sample_id, module_index, int(row["call_index"]))),
            }
            calls.append(call)
            expected_call_records.append(dict(call, name=name, operator=operator))
            total_elements += input_elements
            total_packed += packed_bytes
        require(parse_shape(runtime_row["input_shape_first"], "runtime input") ==
                calls[0]["input_shape"],
                "M51 runtime input shape drift")
        require(parse_shape(runtime_row["output_shape_first"], "runtime output") ==
                calls[0]["output_shape"] and
                int(runtime_row["input_elements"]) ==
                sum(call["input_elements"] for call in calls),
                "M51 runtime aggregate shape/element drift")
        modules.append({
            "module_index": module_index,
            "name": name,
            "operator": operator,
            "expected_hook_calls": 10,
            "expected_weight_elements": int(runtime_row["weight_elements"]),
            "runtime_weight_and_bias_content_sha256_required": True,
            "calls": calls,
        })
    require(total_elements ==
            contract["target_derivation"]["expected_input_elements"] and
            total_packed ==
            contract["target_derivation"]["expected_packed_bytes"],
            "M51 exact size drift")
    for sample_id in range(10):
        observed = [row["name"] for row in sorted(
            [item for item in execution_targets
             if int(item["sample_id"]) == sample_id],
            key=lambda item: int(item["call_index"]))]
        require(observed == module_order, "M51 cross-sample target order drift")
    return {
        "schema": "m51_h67_ep35_binary_input_trace_target_plan_v1",
        "status": "READY_FOR_GPU_CAPTURE_NO_ACTIVATION_PAYLOAD_ON_THIS_HOST",
        "identity": {
            "contract_sha256": sha256(CONTRACT),
            "builder_sha256": sha256(Path(__file__).resolve()),
            "inputs_sha256": dict((name, item["sha256"])
                                   for name, item in contract["inputs"].items()),
        },
        "checkpoint_identity": contract["checkpoint_identity"],
        "frozen_run_identity": h67_identity["profile_identity"],
        "samples": sample_plan,
        "modules": modules,
        "population": {
            "samples": 10,
            "modules": 31,
            "hook_calls": 310,
            "dual_line_rows": 3100,
            "input_elements": total_elements,
            "packed_bytes": total_packed,
            "packed_gib": float(total_packed) / (1 << 30),
        },
        "packing": {
            "layout": "C_ORDER_FLAT",
            "bit_order": "LITTLE_WITHIN_BYTE",
            "file_granularity": "ONE_RAW_FILE_PER_HOOK_CALL",
            "tail_padding_high_bits_zero": True,
            "float_payload_retained": False,
            "delta_payload_retained": False,
        },
        "runtime_requirements": {
            "exact_binary_input_only": True,
            "one_hook_call_per_module_per_sample": True,
            "strict_shape_call_and_order_match": True,
            "module_weight_bias_content_identity": True,
            "overwrite_refusal": True,
            "bounded_streaming_per_call": True,
        },
        "expected_call_records": expected_call_records,
        "claim_policy": contract["claim_policy"],
    }


def write(path, payload, force=False):
    path = Path(path)
    require(force or not path.exists(), "refusing to overwrite M51 target plan")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    with temporary.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    write(args.output, build(), force=args.force)
    print(args.output)


if __name__ == "__main__":
    main()
