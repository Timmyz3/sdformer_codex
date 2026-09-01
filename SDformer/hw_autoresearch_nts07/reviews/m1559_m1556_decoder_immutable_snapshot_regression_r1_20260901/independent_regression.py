#!/usr/bin/env python3
"""Independent, non-executing consistency regression for the M1556 source.

This check never calls ``stream_actual_call``.  It verifies the clean-import
closure and the immutable-byte input discipline with a synthetic 576000-byte
file, then reruns the author's test, synthetic self-test and preflight under
the invoking CPython runtime.

Compatible with CPython 3.6.
"""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/build_m1543_ep34_decoder_nonproduct_streaming_single_call_pilot_source.py"
TEST = HW / "system_simulator/tests/test_m1543_ep34_decoder_nonproduct_streaming_single_call_pilot_source.py"
CONTRACT = HW / "contracts/m1556_ep34_decoder_immutable_snapshot_streaming_source_contract_r1_20260901.json"
RECEIPT = HW / "results/m1556_ep34_decoder_immutable_snapshot_streaming_source_r1_20260901/receipt.json"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

PINNED = {
    "source": "a2fd0e3b1d5fbadcb18ccbadd7b4f709114abb22a19b6c92eec940afab5f9dfa",
    "test": "6a21bdd540f9d648853ffd616516a041b720dc6e24f29cb5f20f7ff044689934",
    "contract": "b50f24624877b9ec5f15c2848f16c9d1e73d6fcd635a0f0c5f608d4b8a90e593",
    "receipt": "4ca384c512c9a50a7a8c5e0ff74875f383b997832b1b4ec517456eedda761b54",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
EXPECTED_MEMBER = "payloads/c000_s10_d0.positive.le.bitpack"
EXPECTED_PAYLOAD_SHA256 = "37208563da5f5b218f3aff5b292f05e10a5db16b078672762b2cb9ed60678a1c"
EXPECTED_SHAPE = (10, 1, 1536, 15, 20)
EXPECTED_BYTES = 576000
CONFIGS = (
    "DENSE_TYPED_K8", "BIT_EQUAL_SERVICE_K1X8", "BIT_TYPED_K8")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_json(arguments):
    output = subprocess.check_output(
        [sys.executable] + [str(value) for value in arguments],
        stderr=subprocess.STDOUT).decode("utf-8")
    return json.loads(output), output


def load_source():
    spec = importlib.util.spec_from_file_location("m1559_bound_m1556", SOURCE)
    require(spec is not None and spec.loader is not None,
            "cannot import pinned M1556 source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    actual_inputs = {
        "source": sha256(SOURCE), "test": sha256(TEST),
        "contract": sha256(CONTRACT), "receipt": sha256(RECEIPT),
        "docs359": sha256(DOC359),
    }
    require(actual_inputs == PINNED, "pinned input SHA drift")

    module = load_source()
    require(list(inspect.signature(module.stream_actual_call).parameters) ==
            ["config"], "executable entry signature drift")
    closure = dict(zip(
        module.stream_actual_call.__code__.co_freevars,
        [cell.cell_contents for cell in module.stream_actual_call.__closure__]))
    required_closure = {"canonical_path", "payload_sha256", "payload_shape",
                        "plane_type", "schedule_verified", "bound_m"}
    require(required_closure.issubset(set(closure)),
            "clean-import closure is incomplete")
    require("selected_pilot_record" not in closure and
            not any(isinstance(value, (dict, list, bytearray))
                    for value in closure.values()),
            "clean-import closure retained mutable selector metadata")
    schedule = closure["schedule_verified"]
    schedule_closure = dict(zip(
        schedule.__code__.co_freevars,
        [cell.cell_contents for cell in schedule.__closure__]))
    require(schedule_closure.get("call_ordinal") == 0 and
            schedule_closure.get("module") == 0 and
            schedule_closure.get("canonical_path") ==
            closure["canonical_path"] and
            schedule_closure.get("payload_sha256") ==
            closure["payload_sha256"],
            "schedule closure call-0 identity drift")
    canonical_root = module.M.M1521_ROOT.resolve()
    member = str(closure["canonical_path"].relative_to(canonical_root))
    require(member == EXPECTED_MEMBER and
            closure["payload_sha256"] == EXPECTED_PAYLOAD_SHA256 and
            tuple(closure["payload_shape"]) == EXPECTED_SHAPE,
            "call-0 clean-import metadata drift")

    # Ordinary input-consistency check only.  The synthetic file has the exact
    # canonical byte count but contains no model payload and is never scheduled.
    with tempfile.TemporaryDirectory(prefix="m1559_snapshot_consistency.") as directory:
        payload_path = Path(directory) / "synthetic_576000.bitpack"
        payload_path.write_bytes(bytes(EXPECTED_BYTES))
        original_sha = sha256(payload_path)
        plane = module.ImmutableLittleBitPlane(
            payload_path, EXPECTED_SHAPE, original_sha)
        require(plane.bytes == EXPECTED_BYTES and
                plane.opened_size == EXPECTED_BYTES and
                plane.opened_sha256 == original_sha and
                plane._stream is None and type(plane._snapshot) is bytes and
                len(plane._snapshot) == EXPECTED_BYTES and
                plane.bit(0, 0, 0, 0) == 0,
                "immutable snapshot creation/close invariant failed")
        snapshot_sha_before = hashlib.sha256(plane._snapshot).hexdigest()
        with payload_path.open("r+b") as stream:
            stream.seek(0)
            stream.write(b"\xff")
            stream.flush()
        modified_source_sha = sha256(payload_path)
        require(modified_source_sha != original_sha and
                plane.bit(0, 0, 0, 0) == 0 and
                hashlib.sha256(plane._snapshot).hexdigest() ==
                snapshot_sha_before == original_sha and
                plane._stream is None,
                "copied bytes changed after later source-file modification")
        plane.close()
        require(plane._stream is None and plane._snapshot is None,
                "snapshot close invariant failed")

    author_output = subprocess.check_output(
        [sys.executable, str(TEST)], stderr=subprocess.STDOUT).decode("utf-8")
    expected_author = (
        "PASS M1556 source tests attacks=16 configs=3 immutable_snapshot=1 "
        "pilot=0 production=0 product=0")
    require(expected_author in author_output, "author test did not pass")

    synthetic, synthetic_output = run_json([SOURCE, "--synthetic-self-test"])
    require(synthetic.get("status") ==
            "PASS_M1556_IMMUTABLE_SNAPSHOT_STREAMING_SOURCE_SYNTHETIC_TEST__NO_PILOT_NO_PRODUCTION" and
            synthetic.get("pilot_execution") is False and
            synthetic.get("production") is False and
            synthetic.get("product_capture") is False,
            "author synthetic self-test boundary drift")
    preflight, preflight_output = run_json([SOURCE, "--preflight"])
    require(preflight.get("status") ==
            "PASS_M1556_IMMUTABLE_SNAPSHOT_STREAMING_SOURCE_PREFLIGHT__NO_PILOT_NO_PRODUCTION" and
            preflight.get("pilot_execution") is False and
            preflight.get("production") is False,
            "author preflight boundary drift")

    description = module.describe()
    require(description.get("schema") ==
            "m1556_ep34_decoder_nonproduct_streaming_single_call_pilot_immutable_snapshot_source_r4_v1" and
            description.get("configurations") == list(CONFIGS) and
            description.get("forbidden_configuration") ==
            "PRODUCT_CAPTURE_TYPED_K8" and
            description.get("pilot") == {"call_ordinal": 0,
                "sample_id": 10, "module_ordinal": 0,
                "timesteps": 10, "execution": False} and
            description["streaming"]["immutable_compact_plane_snapshot"] is True and
            description["source_capabilities"]["closure_row_snapshot"] is True and
            description["source_capabilities"]["mutable_file_backing_during_schedule"] is False and
            description["source_capabilities"]["pilot_cli"] is False and
            description["source_capabilities"]["production_cli"] is False,
            "describe claim boundary drift")

    result = {
        "schema": "m1559_m1556_decoder_immutable_snapshot_regression_r1_v1",
        "status": "PASS_M1559_M1556_INPUT_FIXEDNESS_AND_DUAL_RUNTIME_REGRESSION__ONE_DIAGNOSTIC_PILOT_PRECONDITION_MET",
        "runtime": {"executable": sys.executable,
                    "version": sys.version.split()[0]},
        "pinned_inputs": actual_inputs,
        "clean_import_metadata": {
            "fixed_at_module_load": True,
            "member": member,
            "sha256": closure["payload_sha256"],
            "shape": list(closure["payload_shape"]),
            "sample_id": 10,
            "module_ordinal": schedule_closure["module"],
            "call_ordinal": schedule_closure["call_ordinal"],
            "selector_function_in_entry_closure": False,
        },
        "entry": {"parameters": ["config"],
                  "configurations": list(CONFIGS)},
        "immutable_snapshot": {
            "synthetic_bytes": EXPECTED_BYTES,
            "copied_type_exact_bytes": True,
            "file_closed_before_consumer_access": True,
            "source_sha_changed_after_copy": True,
            "snapshot_sha_unchanged_after_source_modification": True,
            "snapshot_cleared_on_close": True,
        },
        "author_regression": {
            "test_pass": True,
            "test_output": author_output.strip(),
            "synthetic_status": synthetic["status"],
            "preflight_status": preflight["status"],
        },
        "execution": {
            "actual_pilot": False, "request_zero": False,
            "production": False, "product_capture": False,
            "gpu": False, "ssh": False, "eda": False,
        },
        "authorization": {
            "one_d0_call0_three_nonproduct_diagnostic_release_authoring": True,
            "actual_run_authorized_by_this_review": False,
            "separately_sealed_one_shot_release_required": True,
            "production_execution": False,
            "product_configuration": False,
            "automatic_retry": False,
        },
        "claim_boundary": {
            "ordinary_consistency_regression_only": True,
            "security_or_attack_testing": False,
            "pilot_executed": False, "production_executed": False,
            "transactions": False, "cycles": False, "traffic": False,
            "speedup": False, "system_speedup": False,
            "energy": False, "rtl": False, "eda": False,
            "ppa": False, "table_a": False, "paper_result": False,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
