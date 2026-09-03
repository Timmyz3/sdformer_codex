#!/usr/bin/env python3
"""Ordinary data-consistency regression for the exact M1556 source.

This test performs no pilot, production, request scheduling, security bypass,
GPU, SSH, RTL, or EDA operation.  It is CPython 3.6 compatible.
"""
from __future__ import print_function

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
AUTHOR = HW / "results/m1556_ep34_decoder_immutable_snapshot_streaming_source_r1_20260901/receipt.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    SOURCE: "a2fd0e3b1d5fbadcb18ccbadd7b4f709114abb22a19b6c92eec940afab5f9dfa",
    TEST: "6a21bdd540f9d648853ffd616516a041b720dc6e24f29cb5f20f7ff044689934",
    CONTRACT: "b50f24624877b9ec5f15c2848f16c9d1e73d6fcd635a0f0c5f608d4b8a90e593",
    AUTHOR: "4ca384c512c9a50a7a8c5e0ff74875f383b997832b1b4ec517456eedda761b54",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


for _path, _digest in EXPECTED.items():
    assert _path.is_file() and sha256(_path) == _digest, (
        "identity drift: " + str(_path))

SPEC = importlib.util.spec_from_file_location("m1557_bound_m1556", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def must_reject(function):
    try:
        function()
    except Exception:
        return True
    raise AssertionError("blocked entry unexpectedly returned")


def main():
    description = M.describe()
    assert description["schema"] == (
        "m1556_ep34_decoder_nonproduct_streaming_single_call_pilot_immutable_snapshot_source_r4_v1")
    assert list(inspect.signature(M.stream_actual_call).parameters) == ["config"]
    assert not hasattr(M, "stream_tensor")
    assert description["pilot"] == {"call_ordinal": 0, "sample_id": 10,
        "module_ordinal": 0, "timesteps": 10, "execution": False}
    assert description["streaming"]["immutable_compact_plane_snapshot"] is True
    assert description["source_capabilities"]["closure_row_snapshot"] is True
    assert description["source_capabilities"]["mutable_file_backing_during_schedule"] is False

    # Read-only closure inspection: call-0 metadata is represented as scalar
    # values captured at module load, not as a selector or mutable row object.
    closure = dict(zip(M.stream_actual_call.__code__.co_freevars,
                       [cell.cell_contents for cell in
                        M.stream_actual_call.__closure__]))
    assert "canonical_path" in closure and "payload_sha256" in closure
    assert "payload_shape" in closure and "selected_pilot_record" not in closure
    assert closure["canonical_path"].relative_to(M.M.M1521_ROOT).as_posix() == (
        "payloads/c000_s10_d0.positive.le.bitpack")
    assert closure["payload_sha256"] == (
        "37208563da5f5b218f3aff5b292f05e10a5db16b078672762b2cb9ed60678a1c")
    assert closure["payload_shape"] == (10, 1, 1536, 15, 20)
    row0 = M.selected_pilot_record()
    assert row0["global_call_ordinal"] == 0 and row0["global_sample_id"] == 10
    assert row0["module_ordinal"] == 0
    assert row0["positive_output"] == (
        "payloads/c000_s10_d0.positive.le.bitpack")
    assert row0["positive_output_sha256"] == closure["payload_sha256"]

    # Exactly the D0 compact payload size.  After construction, the source
    # descriptor is closed and subsequent file changes do not affect bit().
    shape = (10, 1, 1536, 15, 20)
    compact_bytes = 576000
    with tempfile.TemporaryDirectory(prefix="m1557_snapshot.",
                                     dir=str(HERE)) as directory:
        path = Path(directory) / "d0.bitpack"
        path.write_bytes(bytes(compact_bytes))
        expected = sha256(path)
        with M.ImmutableLittleBitPlane(path, shape, expected) as plane:
            assert plane.bytes == compact_bytes
            assert plane.opened_size == compact_bytes
            assert plane.opened_sha256 == expected
            assert plane._stream is None
            assert type(plane._snapshot) is bytes
            assert len(plane._snapshot) == compact_bytes
            snapshot_sha = hashlib.sha256(plane._snapshot).hexdigest()
            assert snapshot_sha == expected and plane.bit(0, 0, 0, 0) == 0
            with path.open("r+b", buffering=0) as mutable:
                mutable.write(b"\xff")
                mutable.flush()
            assert sha256(path) != expected
            assert plane.bit(0, 0, 0, 0) == 0
            assert hashlib.sha256(plane._snapshot).hexdigest() == snapshot_sha
        assert plane._stream is None and plane._snapshot is None

    # These are ordinary release-boundary regressions, not launch attempts.
    assert must_reject(lambda: M.M.validate_config(M.FORBIDDEN_CONFIG))
    assert must_reject(M.pilot_release)
    assert must_reject(M.production_release)

    synthetic = json.loads(subprocess.check_output(
        [sys.executable, str(SOURCE), "--synthetic-self-test"],
        stderr=subprocess.STDOUT).decode("utf-8"))
    assert synthetic["status"] == (
        "PASS_M1556_IMMUTABLE_SNAPSHOT_STREAMING_SOURCE_SYNTHETIC_TEST__NO_PILOT_NO_PRODUCTION")
    assert synthetic["pilot_execution"] is False
    assert synthetic["production"] is False
    assert synthetic["product_capture"] is False

    preflight = json.loads(subprocess.check_output(
        [sys.executable, str(SOURCE), "--preflight"],
        stderr=subprocess.STDOUT).decode("utf-8"))
    assert preflight["status"] == (
        "PASS_M1556_IMMUTABLE_SNAPSHOT_STREAMING_SOURCE_PREFLIGHT__NO_PILOT_NO_PRODUCTION")
    assert preflight["pilot_execution"] is False
    assert preflight["production"] is False
    assert preflight["authorities"]["pilot_call_ordinal"] == 0

    author_test = subprocess.check_output([sys.executable, str(TEST)],
                                          stderr=subprocess.STDOUT).decode("utf-8")
    assert "PASS M1556 source tests attacks=16 configs=3 immutable_snapshot=1 pilot=0 production=0 product=0" in author_test

    result = {
        "schema": "m1557_m1556_decoder_immutable_snapshot_regression_output_r1_v1",
        "status": "PASS_M1557_M1556_INPUT_FIXEDNESS_AND_DUAL_RUNTIME_REGRESSION__ONE_DIAGNOSTIC_PILOT_PRECONDITION_MET",
        "python": sys.version.split()[0],
        "bindings": dict((path.name, digest) for path, digest in EXPECTED.items()),
        "verified": {
            "entry_parameters": ["config"],
            "call_ordinal": 0,
            "sample_id": 10,
            "module_ordinal": 0,
            "payload_member": row0["positive_output"],
            "payload_sha256": closure["payload_sha256"],
            "payload_shape": list(closure["payload_shape"]),
            "closure_contains_selector_or_row_object": False,
            "compact_snapshot_bytes": compact_bytes,
            "fd_closed_before_snapshot_consumption": True,
            "snapshot_is_bytes": True,
            "post_snapshot_source_change_isolated": True,
            "synthetic_pass": True,
            "preflight_pass": True,
            "author_test_pass": True,
            "product_configuration_blocked": True,
            "pilot_release_blocked": True,
            "production_release_blocked": True,
        },
        "execution": {
            "actual_pilot": False,
            "request_zero": False,
            "production": False,
            "product_configuration": False,
            "gpu": False,
            "ssh": False,
            "rtl_eda": False,
        },
        "authorization": {
            "one_d0_call0_three_nonproduct_diagnostic_pilot_next": True,
            "production": False,
            "product_configuration": False,
            "automatic_retry": False,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
