#!/usr/bin/env python3
"""Independently hash and validate the real M51 H67 GPU payload."""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


EXPECTED_MANIFEST_SHA256 = (
    "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e")
EXPECTED_PLAN_SHA256 = (
    "bf0827d32896a871d9ea4c91afe49014bb5c236d619764b5c3f8a2804dc595e3")
EXPECTED_POPULATION = {
    "samples": 10,
    "modules": 31,
    "hook_calls": 310,
    "input_elements": 10506240000,
    "packed_bytes": 1313280000,
}
EXPECTED_SOURCES = {
    "base_profiler": "04f692c5bda6d1f88cdc932ce48f012767f22a2bb1ca161378971232f99c0684",
    "runner_r2": "79ab16c00d2dc90308dca52e995047c951e6721f7d934b69e59bf3d63b99c7cf",
    "writer_r1_base": "6422572a91604b252e421b4343dc05ab76c911403f04632a8398423ddc9bc4eb",
    "writer_r2": "da9c9df4c4c0bacdcf91e9beb35403c506ec6a2432bcbb2eef1baaecec79927b",
}
EXPECTED_CHECKPOINT = {
    "sha256": "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
    "size_bytes": 591167876,
}
EXPECTED_CONFIG_SHA256 = (
    "8be3f7bbffd75c4356d3abf5935679d80e15c1caefd307c19a727729659e6c49")
POPCOUNT_TABLE = bytes(bytearray(bin(value).count("1") for value in range(256)))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise RuntimeError("non-standard JSON constant: {}".format(raw))

    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: {}".format(key))
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def product(values):
    result = 1
    for value in values:
        require(isinstance(value, int) and not isinstance(value, bool) and
                value >= 0, "invalid shape dimension")
        result *= value
    return result


def hash_and_popcount(path):
    digest = hashlib.sha256()
    active = size = 0
    last = None
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
            size += len(block)
            active += sum(block.translate(POPCOUNT_TABLE))
            if block:
                last = block[-1]
    return digest.hexdigest(), size, active, last


def validate_memory(snapshot, phase):
    require(snapshot["phase"] == phase and
            snapshot["cuda_available"] is True and
            snapshot["capture_device_type"] == "cuda",
            "CUDA memory snapshot phase/device mismatch")
    for key in ("memory_allocated_bytes", "memory_reserved_bytes",
                "max_memory_allocated_bytes", "max_memory_reserved_bytes"):
        require(isinstance(snapshot[key], int) and
                not isinstance(snapshot[key], bool) and snapshot[key] >= 0,
                "invalid CUDA memory metric: {}".format(key))
    require(snapshot["max_memory_allocated_bytes"] >=
            snapshot["memory_allocated_bytes"] and
            snapshot["max_memory_reserved_bytes"] >=
            snapshot["memory_reserved_bytes"],
            "CUDA peak/current memory mismatch")


def validate(manifest_path, plan_path):
    manifest_path = manifest_path.resolve()
    plan_path = plan_path.resolve()
    require(sha256_path(manifest_path) == EXPECTED_MANIFEST_SHA256,
            "real GPU manifest SHA mismatch")
    require(sha256_path(plan_path) == EXPECTED_PLAN_SHA256,
            "frozen target plan SHA mismatch")
    manifest = strict_json(manifest_path)
    plan = strict_json(plan_path)
    output_root = manifest_path.parent

    require(manifest["schema"] ==
            "m51_h67_ep35_binary_input_trace_manifest_v1" and
            manifest["status"] ==
            "PASS_EXACT_BINARY_INPUT_TRACE_NO_OUTPUT_OR_PERFORMANCE_CLAIM",
            "manifest schema/status mismatch")
    require(manifest["population"] == dict(
        EXPECTED_POPULATION, active_elements=712894209),
        "manifest population mismatch")
    require(plan["population"] == dict(
        EXPECTED_POPULATION, dual_line_rows=3100,
        packed_gib=1.2230873107910156),
        "target plan population mismatch")
    require(manifest["identity"]["target_plan_sha256"] ==
            EXPECTED_PLAN_SHA256 and
            manifest["packing"] == {
                "bit_order": "LITTLE_WITHIN_BYTE",
                "delta_payload_retained": False,
                "file_granularity": "ONE_RAW_FILE_PER_HOOK_CALL",
                "float_payload_retained": False,
                "layout": "C_ORDER_FLAT",
                "tail_padding_high_bits_zero": True,
            }, "manifest plan/packing identity mismatch")

    context = manifest["run_context"]
    require(context["source_sha256"] == EXPECTED_SOURCES and
            context["runner_contract_sha256"] ==
                "571b54e28e6778c4920baa16b44ebc7b76dc00cb158bcc676921539cfd302f5e" and
            context["config_sha256"] == EXPECTED_CONFIG_SHA256 and
            context["checkpoint_sha256"] == EXPECTED_CHECKPOINT["sha256"] and
            context["checkpoint_size_bytes"] == EXPECTED_CHECKPOINT["size_bytes"] and
            context["checkpoint_load_audit"]["missing_count"] == 0 and
            context["checkpoint_load_audit"]["unexpected_count"] == 0 and
            context["cuda_synchronization"] == {
                "before_capture": 1,
                "per_sample_post_forward": 10,
                "final_pre_manifest": 1,
            } and context["input_layout_policy"] ==
                "REQUIRE_DEFAULT_C_CONTIGUOUS_NO_WHOLE_CALL_COPY",
            "run/checkpoint/source identity mismatch")
    memory = context["capture_memory"]
    require(memory["peak_stats_reset_before_capture"] is True,
            "CUDA peak reset evidence missing")
    validate_memory(memory["before"], "BEFORE_CAPTURE")
    validate_memory(memory["after"], "AFTER_FINAL_SYNCHRONIZE")
    require(memory["after"]["max_memory_allocated_bytes"] >=
            memory["before"]["max_memory_allocated_bytes"] and
            memory["after"]["max_memory_reserved_bytes"] >=
            memory["before"]["max_memory_reserved_bytes"],
            "capture peak did not dominate baseline")

    plan_modules = plan["modules"]
    require(len(plan_modules) == 31 and
            [row["module_index"] for row in plan_modules] ==
                list(range(31)), "target module order mismatch")
    plan_by_name = {}
    calls_by_key = {}
    for module in plan_modules:
        name = module["name"]
        require(name not in plan_by_name and len(module["calls"]) == 10,
                "target module duplicate/call-count mismatch")
        plan_by_name[name] = module
        for call in module["calls"]:
            key = (call["sample_id"], name)
            require(key not in calls_by_key,
                    "target call duplicate: {}".format(key))
            calls_by_key[key] = call
    require(len(calls_by_key) == 310, "target call population mismatch")

    identities = manifest["module_identities"]
    require(set(identities) == set(plan_by_name),
            "module identity name set mismatch")
    for name, identity in identities.items():
        target = plan_by_name[name]
        require(identity["operator"] == target["operator"],
                "module operator mismatch: {}".format(name))
        weight = identity["weight"]
        require(product(weight["shape"]) == target["expected_weight_elements"] and
                weight["layout"] == "C_ORDER_CONTIGUOUS" and
                weight["byte_order"] == "little" and
                len(weight["content_sha256"]) == 64 and
                weight["content_bytes"] > 0,
                "module weight identity mismatch: {}".format(name))

    records = manifest["records"]
    require(len(records) == 310, "manifest record population mismatch")
    seen_keys = set()
    seen_paths = set()
    payload_collection = hashlib.sha256()
    sample_totals = dict((sample, {"records": 0, "elements": 0,
                                   "packed_bytes": 0, "active": 0})
                         for sample in range(10))
    total_elements = total_packed = total_active = 0
    for index, record in enumerate(records):
        key = (record["sample_id"], record["name"])
        require(key in calls_by_key and key not in seen_keys,
                "record target identity mismatch: {}".format(key))
        target_call = calls_by_key[key]
        target_module = plan_by_name[record["name"]]
        require(record["operator"] == target_module["operator"] and
                record["module_index"] == target_module["module_index"] and
                record["target_order_index"] == target_module["module_index"] and
                record["frozen_execution_call_index"] ==
                    target_call["frozen_execution_call_index"] and
                record["input_shape"] == target_call["input_shape"] and
                record["output_shape"] == target_call["output_shape"] and
                record["sample_key"] == target_call["sample_key"] and
                record["sequence_key"] == target_call["sequence_key"],
                "record-plan field mismatch at index {}".format(index))
        elements = product(record["input_shape"])
        packed = (elements + 7) // 8
        tail = elements % 8 or 8
        require(elements == record["input_elements"] ==
                    target_call["input_elements"] and
                packed == record["packed_bytes"] ==
                    target_call["packed_bytes"] and
                tail == record["tail_used_bits"] and
                0 <= record["active_elements"] <= elements,
                "record element/packing arithmetic mismatch")

        relative = Path(record["relative_path"])
        require(not relative.is_absolute() and ".." not in relative.parts and
                relative.as_posix() == target_call["relative_output_path"] and
                relative.as_posix() not in seen_paths,
                "record relative path mismatch")
        payload_path = (output_root / relative).resolve()
        require(output_root in payload_path.parents and payload_path.is_file(),
                "payload missing or escaped output root")
        digest, size, active, last = hash_and_popcount(payload_path)
        require(digest == record["file_sha256"] and size == packed and
                active == record["active_elements"],
                "payload hash/size/popcount mismatch: {}".format(relative))
        if tail < 8:
            require(last is not None and (last & ~((1 << tail) - 1)) == 0,
                    "nonzero packed tail bits: {}".format(relative))

        seen_keys.add(key)
        seen_paths.add(relative.as_posix())
        payload_collection.update((relative.as_posix() + "\0" + digest +
                                   "\0{}\0{}\n".format(size, active)).encode("utf-8"))
        totals = sample_totals[record["sample_id"]]
        totals["records"] += 1
        totals["elements"] += elements
        totals["packed_bytes"] += packed
        totals["active"] += active
        total_elements += elements
        total_packed += packed
        total_active += active

    actual_payloads = set(
        path.relative_to(output_root).as_posix()
        for path in (output_root / "calls").rglob("*") if path.is_file())
    require(actual_payloads == seen_paths and
            len(seen_keys) == len(calls_by_key) == 310,
            "payload file set/target coverage mismatch")
    require(not list(output_root.rglob("*.partial")) and
            not (output_root / "FAILED.json").exists(),
            "partial or FAILED artifact exists beside PASS")
    require(total_elements == EXPECTED_POPULATION["input_elements"] and
            total_packed == EXPECTED_POPULATION["packed_bytes"] and
            total_active == manifest["population"]["active_elements"],
            "aggregate payload conservation mismatch")
    require(all(row["records"] == 31 for row in sample_totals.values()),
            "per-sample module coverage mismatch")

    return {
        "schema": "m51_h67_ep35_binary_input_trace_gpu_payload_validation_receipt_v1",
        "status": "PASS_REAL_GPU_ALL310_PAYLOAD_SHA_SIZE_POPCOUNT_PLAN_IDENTITY",
        "identity": {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "target_plan_sha256": EXPECTED_PLAN_SHA256,
            "validator_sha256": sha256_path(Path(__file__).resolve()),
            "payload_collection_sha256": payload_collection.hexdigest(),
        },
        "population": dict(manifest["population"], dual_line_rows=3100,
                           operator_types={"Conv2d": 7, "Linear": 24}),
        "per_sample": [dict({"sample_id": sample}, **sample_totals[sample])
                       for sample in range(10)],
        "cuda_memory": memory,
        "checks": {
            "all_310_payload_sha256_recomputed": True,
            "all_310_payload_sizes_recomputed": True,
            "all_310_payload_popcounts_recomputed": True,
            "all_tail_padding_high_bits_zero": True,
            "plan_module_call_shape_path_identity": True,
            "checkpoint_config_runner_writer_identity": True,
            "synchronization_1_plus_10_plus_1": True,
            "partial_or_failed_artifacts_absent": True,
        },
        "claim_boundary": (
            "exact raw binary input payload only; no operator output, cycle, "
            "speedup, RTL, PPA, power, energy, system or DATE headline claim"),
    }


def write_receipt(path, payload):
    require(not path.exists(), "refusing existing GPU payload receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    with temporary.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.link(str(temporary), str(path))
    temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = validate(args.manifest, args.target_plan)
    write_receipt(args.receipt.resolve(), receipt)
    print("PASS M51 real GPU payload files=310 bytes=1313280000 active=712894209")
    print(args.receipt.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
