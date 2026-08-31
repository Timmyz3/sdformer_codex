#!/usr/bin/env python3
"""Independent fail-closed hammer for the final M1543 source bytes.

This test never opens the selected canonical bit plane and never launches a
pilot, production population, or product configuration.  The sole execution
witness is a tiny synthetic non-product tensor used to test the importable
scope boundary.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/build_m1543_ep34_decoder_nonproduct_streaming_single_call_pilot_source.py"
TEST = HW / "system_simulator/tests/test_m1543_ep34_decoder_nonproduct_streaming_single_call_pilot_source.py"
CONTRACT = HW / "contracts/m1543_ep34_decoder_nonproduct_streaming_single_call_pilot_source_contract_r1_20260831.json"
EXPECTED = {
    SOURCE: "cf9a0938e1d67b58d5fd3db906c0c52527f5de3b8c3e0ff637c66a0b7c436e81",
    TEST: "488d4ec961e349688d1b026363a2970af30e4965c2de14a2f765933facb2d2f2",
    CONTRACT: "d442e5df0b2c74a5025d0a98098088178f3e09b3ff594efbd5971959a2ea4051",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


for path, expected in EXPECTED.items():
    assert sha256(path) == expected, "final author SHA drift: " + str(path)

spec = importlib.util.spec_from_file_location("m1546_bound_m1543", SOURCE)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def rejected(name, function, attacks):
    try:
        function()
    except Exception:
        attacks.append(name)
        return
    raise AssertionError("attack accepted: " + name)


class TinyPlane(object):
    timesteps = 10
    channels = 8
    height = 1
    width = 1

    @staticmethod
    def bit(_timestep, _channel, _y, _x):
        return 0


def main():
    attacks = []
    assert M.describe()["pilot"] == {
        "call_ordinal": 0, "sample_id": 10, "module_ordinal": 0,
        "timesteps": 10, "execution": False}
    assert tuple(M.M.CONFIGS) == (
        "DENSE_TYPED_K8", "BIT_EQUAL_SERVICE_K1X8", "BIT_TYPED_K8")

    rejected("product_configuration", lambda: M.M.validate_config(
        "PRODUCT_CAPTURE_TYPED_K8"), attacks)
    rejected("foreign_configuration", lambda: M.M.validate_config(
        "OFFICIAL_M700"), attacks)
    rejected("pilot_release", M.pilot_release, attacks)
    rejected("production_release", M.production_release, attacks)
    rejected("pilot_cli", lambda: subprocess.check_output(
        [str(os.environ.get("PYTHON", "python3")), str(SOURCE), "--pilot"],
        stderr=subprocess.STDOUT), attacks)
    rejected("production_cli", lambda: subprocess.check_output(
        [str(os.environ.get("PYTHON", "python3")), str(SOURCE), "--production"],
        stderr=subprocess.STDOUT), attacks)

    row = M.selected_pilot_record()
    manifest = M.M.strict_json(M.M.M1521_MANIFEST)
    bad = dict(row); bad["global_call_ordinal"] = 1
    manifest["records"][0] = bad
    rejected("manifest_call_drift", lambda: M.M.validate_population_manifest(
        manifest), attacks)

    with tempfile.TemporaryDirectory(prefix="m1546_attack.") as directory:
        root = Path(directory)
        short = root / "short.bitpack"; short.write_bytes(b"\x00")
        rejected("payload_size", lambda: M.MmapLittleBitPlane(
            short, (10, 1, 8, 2, 2)), attacks)
        rejected("payload_shape", lambda: M.MmapLittleBitPlane(
            short, (10, 2, 8, 2, 2)), attacks)
        plane_path = root / "plane.bitpack"
        plane_path.write_bytes(bytes(40))
        rejected("payload_sha", lambda: M.MmapLittleBitPlane(
            plane_path, (10, 1, 8, 2, 2), "0" * 64), attacks)
        link = root / "plane.link"; link.symlink_to(plane_path)
        rejected("payload_symlink", lambda: M.MmapLittleBitPlane(
            link, (10, 1, 8, 2, 2)), attacks)
        with M.MmapLittleBitPlane(plane_path, (10, 1, 8, 2, 2)) as plane:
            rejected("timestep_oob", lambda: plane.bit(10, 0, 0, 0), attacks)
            rejected("channel_oob", lambda: plane.bit(0, 8, 0, 0), attacks)
            rejected("coordinate_oob", lambda: plane.bit(0, 0, 2, 0), attacks)

        # A 64 MiB sparse mmap, touching only two pages, must not materialize
        # the bit plane.  This is diagnostic; the hard source cap remains 8 GiB.
        large = root / "large.bitpack"
        with large.open("wb") as stream:
            stream.truncate(64 * 1024 * 1024)
        before = M.peak_rss_kib()
        with M.MmapLittleBitPlane(
                large, (1, 1, 1, 1, 64 * 1024 * 1024 * 8)) as plane:
            assert plane.bit(0, 0, 0, 0) == 0
            assert plane.bit(0, 0, 0, plane.width - 1) == 0
        mmap_rss_delta_kib = max(0, M.peak_rss_kib() - before)
        assert mmap_rss_delta_kib < 16 * 1024

        # Copying and corrupting the sealed authority must fail closed.
        seal_copy = root / "seal"
        shutil.copytree(str(M.M1542), str(seal_copy))
        with (seal_copy / "review.json").open("ab") as stream:
            stream.write(b"\n")
        rejected("authority_seal_tamper", lambda: M.verify_flat_seal(
            seal_copy, M.M1542_REVIEW_SHA256, M.M1542_OUTER_FILE_SHA256),
            attacks)

    scheduler = M.StreamingCallScheduler("BIT_TYPED_K8")
    rejected("unresolved_dependency", lambda: scheduler.one(M.M.request(
        "dep", "BIT_TYPED_K8", "compute", [0], [0], 288,
        ["missing"], "done")), attacks)
    rejected("unknown_retained_token", lambda: scheduler.retire_destination(
        ("missing",)), attacks)
    rejected("bank_oob", lambda: scheduler.one(M.M.request(
        "bank", "BIT_TYPED_K8", "weight_read", [0], [8], 96)), attacks)
    rejected("psum_capacity", lambda: scheduler.one(M.M.request(
        "psum", "BIT_TYPED_K8", "psum_read", [221184], [0], 48)), attacks)
    rejected("weight_capacity", lambda: scheduler.one(M.M.request(
        "weight", "BIT_TYPED_K8", "weight_read", [1728], [0], 96)), attacks)
    rejected("cross_config", lambda: scheduler.one(M.M.request(
        "cross", "DENSE_TYPED_K8", "compute", [0], [0], 288)), attacks)
    scheduler.one(M.M.request("token_a", "BIT_TYPED_K8", "compute",
                              [0], [0], 288, produces="a"))
    rejected("duplicate_token", lambda: scheduler.one(M.M.request(
        "token_a2", "BIT_TYPED_K8", "compute", [0], [0], 288,
        produces="a")), attacks)

    original_peak = M.peak_rss_kib
    try:
        M.peak_rss_kib = lambda: M.PEAK_RSS_LIMIT_KIB
        rejected("rss_cap", M.memory_gate, attacks)
    finally:
        M.peak_rss_kib = original_peak

    # Destination retirement must shrink tokens without erasing the physical
    # scheduler calendar, outstanding returns, counters, or digests.
    state = M.StreamingCallScheduler("BIT_TYPED_K8")
    state.one(M.M.request("a", "BIT_TYPED_K8", "compute", [0], [0], 288,
                          produces="a"))
    state.one(M.M.request("b", "BIT_TYPED_K8", "compute", [0], [0], 288,
                          ["a"], "b"))
    physical = (dict(state.scheduler.next_port),
                dict((k, list(v)) for k, v in state.scheduler.outstanding.items()),
                state.scheduler.requests, state.scheduler.last_cycle,
                state.scheduler.address_digest.hexdigest(),
                state.scheduler.commit_digest.hexdigest())
    state.retire_destination(("b",))
    assert set(state.scheduler.tokens) == {"b"}
    assert physical == (dict(state.scheduler.next_port),
                        dict((k, list(v)) for k, v in state.scheduler.outstanding.items()),
                        state.scheduler.requests, state.scheduler.last_cycle,
                        state.scheduler.address_digest.hexdigest(),
                        state.scheduler.commit_digest.hexdigest())

    # Fail witness: the generic importable engine admits a non-canonical
    # module and call ordinal, and it does not bind the plane path or digest.
    old_geometry = M.M.GEOMETRY
    try:
        geometry = list(old_geometry)
        geometry[2] = (8, 96, 1, 1, 2, 2)
        M.M.GEOMETRY = tuple(geometry)
        bypass = M.stream_tensor("BIT_TYPED_K8", TinyPlane(), 2, 7)
    finally:
        M.M.GEOMETRY = old_geometry
    assert bypass["module_ordinal"] == 2
    assert bypass["pilot_call_ordinal"] == 7
    assert bypass["request_count"] > 0
    assert bypass["product_capture"] is False and bypass["production"] is False

    assert len(attacks) == 23
    result = {
        "schema": "m1546_m1543_streaming_source_independent_hammer_r1_v1",
        "status": "NO_GO_SCOPE_BYPASS__STREAM_TENSOR_NOT_BOUND_TO_CALL0_D0_OR_CANONICAL_PLANE",
        "attacks_rejected": attacks,
        "attack_count": len(attacks),
        "full_identity_preflight_separately_required": True,
        "mmap_rss_delta_kib": mmap_rss_delta_kib,
        "scope_bypass_witness": {
            "configuration": bypass["configuration"],
            "module_ordinal": bypass["module_ordinal"],
            "call_ordinal": bypass["pilot_call_ordinal"],
            "timesteps": bypass["timesteps"],
            "request_count": bypass["request_count"],
            "transaction_address_sha256": bypass["transaction_address_sha256"],
            "commit_sequence_sha256": bypass["commit_sequence_sha256"],
            "synthetic_only": True,
        },
        "pilot_executed": False,
        "production_executed": False,
        "product_executed": False,
        "authorization": "NO_GO",
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
