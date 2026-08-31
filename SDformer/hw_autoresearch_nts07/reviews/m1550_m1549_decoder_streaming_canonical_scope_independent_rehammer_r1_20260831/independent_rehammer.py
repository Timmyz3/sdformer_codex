#!/usr/bin/env python3
"""M1550 independent fail-closed rehammer of the exact M1549 source bytes.

No actual pilot, production population, or product configuration is run.  A
sentinel replaces the scheduler before the custom-subclass boundary probe, so
that probe cannot emit even its first request.
"""
from __future__ import print_function

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/build_m1543_ep34_decoder_nonproduct_streaming_single_call_pilot_source.py"
TEST = HW / "system_simulator/tests/test_m1543_ep34_decoder_nonproduct_streaming_single_call_pilot_source.py"
CONTRACT = HW / "contracts/m1549_m1546_decoder_streaming_canonical_scope_successor_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1549_m1546_decoder_streaming_canonical_scope_successor_author_receipt_r1_20260831"
M1546 = HW / "reviews/m1546_m1543_decoder_streaming_single_call_source_independent_hammer_r1_20260831"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    SOURCE: "63224377e9678039f3182822ad2307aed72ffd865a6cc711f9299ac24cbefb0d",
    TEST: "93e1916f610a4001fe687a4c6dc3878bebcb4e02ad9981080609e4bcb23cbe78",
    CONTRACT: "97da2f25b6c2ba75f017e4d5d81910aef17e7c0e35c81243d809aa339cf3a695",
    AUTHOR / "review.json": "cc7adc7ffc914dcddccb2c25c214d764c597c634437d871b8efdebb3668c347a",
    AUTHOR / "SHA256SUMS.seal.sha256": "a1420af99f3df1449b8cc8c25fb5c87d5131744dd9ce0aba7dbb0ff91cf80cee",
    M1546 / "review.json": "7a668bd9d3975b186af357f59279012c3e9835493ef6011c9a12f8cfb230d772",
    M1546 / "SHA256SUMS.seal.sha256": "ec6c396b131d3ea3f2421bf4380695b5794d75e1e90a4babe7f4c76db2b358f6",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rejected(name, function, attacks):
    try:
        function()
    except Exception:
        attacks.append(name)
        return
    raise AssertionError("attack accepted: " + name)


def verify_flat_author_receipt():
    outer = AUTHOR / "SHA256SUMS.seal.sha256"
    manifest = AUTHOR / "SHA256SUMS"
    assert outer.read_text().split() == [sha256(manifest), "SHA256SUMS"]
    rows = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert name not in rows and "/" not in name and ".." not in name
        assert sha256(AUTHOR / name) == digest
        rows[name] = digest
    actual = set(item.name for item in AUTHOR.iterdir()
                 if item.is_file() and item.name not in
                 ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    assert actual == set(rows) and len(rows) == 3
    return sha256(manifest)


for path, expected in EXPECTED.items():
    assert path.is_file() and sha256(path) == expected, "identity drift: " + str(path)
author_manifest_sha256 = verify_flat_author_receipt()

spec = importlib.util.spec_from_file_location("m1550_bound_m1549", str(SOURCE))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class CustomPlane(object):
    timesteps = 10
    channels = M.M.GEOMETRY[0][0]
    height = M.M.GEOMETRY[0][2]
    width = M.M.GEOMETRY[0][3]

    @staticmethod
    def bit(_timestep, _channel, _y, _x):
        return 0


class FirstSchedulerBoundary(RuntimeError):
    pass


class StopBeforeFirstRequest(object):
    def __init__(self, _config):
        raise FirstSchedulerBoundary("identity guards passed; stopped before scheduler/request")


def main():
    attacks = []
    description = M.describe()
    assert description["schema"] == "m1549_ep34_decoder_nonproduct_streaming_single_call_pilot_successor_source_r2_v1"
    assert description["pilot"] == {"call_ordinal": 0, "sample_id": 10,
        "module_ordinal": 0, "timesteps": 10, "execution": False}
    assert tuple(M.M.CONFIGS) == (
        "DENSE_TYPED_K8", "BIT_EQUAL_SERVICE_K1X8", "BIT_TYPED_K8")
    assert description["source_capabilities"]["pilot_cli"] is False
    assert description["source_capabilities"]["production_cli"] is False

    # Full identity hashing is read-only and does not open the pilot engine.
    authority = M.validate_authorities(True)
    assert authority["m1539"]["m1521"]["members"] == 122
    assert authority["m1539"]["full_payload_verification"] is True
    assert authority["pilot_execution"] is False and authority["production"] is False
    row = M.selected_pilot_record()
    assert row["global_call_ordinal"] == 0 and row["global_sample_id"] == 10
    assert row["module_ordinal"] == 0 and tuple(row["shape"]) == M.M.INPUT_SHAPES[0]
    canonical = M.M.M1521_ROOT / row["positive_output"]
    assert canonical.is_file() and not canonical.is_symlink()
    assert sha256(canonical) == row["positive_output_sha256"]

    rejected("product_configuration", lambda: M.M.validate_config(
        M.FORBIDDEN_CONFIG), attacks)
    rejected("foreign_configuration", lambda: M.M.validate_config(
        "OFFICIAL_M700"), attacks)
    rejected("pilot_release", M.pilot_release, attacks)
    rejected("production_release", M.production_release, attacks)
    rejected("product_internal_entry", lambda: M.stream_actual_call(
        M.FORBIDDEN_CONFIG), attacks)
    for option, label in (("--pilot", "pilot_cli"),
                          ("--production", "production_cli"),
                          ("--product", "product_cli")):
        rejected(label, lambda option=option: subprocess.check_output(
            [sys.executable, str(SOURCE), option], stderr=subprocess.STDOUT), attacks)

    # Original M1546 witness and direct custom-call0 witness must stop before
    # any scheduler is constructed.
    rejected("m1546_module2_call7_custom", lambda: M.stream_tensor(
        "BIT_TYPED_K8", CustomPlane(), 2, 7), attacks)
    rejected("call0_d0_custom", lambda: M.stream_tensor(
        "BIT_TYPED_K8", CustomPlane(), 0, 0), attacks)

    exact_instance_reached_scheduler_boundary = False
    with tempfile.TemporaryDirectory(prefix="m1550_rehammer.",
                                     dir=str(HERE)) as directory:
        root = Path(directory)
        shape = tuple(row["shape"])

        foreign = root / "foreign_same_shape.bitpack"
        shutil.copyfile(str(canonical), str(foreign))
        assert sha256(foreign) == row["positive_output_sha256"]
        with M.MmapLittleBitPlane(foreign, shape,
                                  row["positive_output_sha256"]) as plane:
            rejected("foreign_same_shape_same_sha", lambda: M.stream_tensor(
                "BIT_TYPED_K8", plane, 0, 0), attacks)

        hardlink = root / "foreign_hardlink.bitpack"
        os.link(str(canonical), str(hardlink))
        with M.MmapLittleBitPlane(hardlink, shape,
                                  row["positive_output_sha256"]) as plane:
            rejected("foreign_hardlink", lambda: M.stream_tensor(
                "BIT_TYPED_K8", plane, 0, 0), attacks)

        symlink = root / "canonical_symlink.bitpack"
        symlink.symlink_to(canonical)
        rejected("canonical_symlink", lambda: M.MmapLittleBitPlane(
            symlink, shape, row["positive_output_sha256"]), attacks)

        corrupt = root / "corrupt_same_shape.bitpack"
        shutil.copyfile(str(canonical), str(corrupt))
        with corrupt.open("r+b") as stream:
            byte = stream.read(1)
            stream.seek(0)
            stream.write(bytes([byte[0] ^ 1]))
        rejected("payload_sha", lambda: M.MmapLittleBitPlane(
            corrupt, shape, row["positive_output_sha256"]), attacks)

        rejected("payload_shape", lambda: M.MmapLittleBitPlane(
            foreign, (9,) + shape[1:], row["positive_output_sha256"]), attacks)

        # Attribute tampering after a valid foreign open remains rejected by
        # the canonical-path guard.  No canonical plane is opened here.
        with M.MmapLittleBitPlane(foreign, shape,
                                  row["positive_output_sha256"]) as plane:
            plane.expected_sha256 = "0" * 64
            rejected("expected_sha_attribute_tamper", lambda: M.stream_tensor(
                "BIT_TYPED_K8", plane, 0, 0), attacks)

        # Strong TOCTOU/identity probe using the *exact* accepted class: open
        # and mmap corrupt foreign bytes, then mutate only the public path/SHA
        # attributes.  stream_tensor re-hashes the canonical pathname, not the
        # already-open descriptor/mmap that bit() will consume.  The sentinel
        # prevents scheduler construction and request zero.
        with M.MmapLittleBitPlane(corrupt, shape) as plane:
            assert type(plane) is M.MmapLittleBitPlane
            plane.path = canonical
            plane.expected_sha256 = row["positive_output_sha256"]
            original_scheduler = M.StreamingCallScheduler
            try:
                M.StreamingCallScheduler = StopBeforeFirstRequest
                try:
                    M.stream_tensor("BIT_TYPED_K8", plane, 0, 0)
                except FirstSchedulerBoundary:
                    exact_instance_reached_scheduler_boundary = True
            finally:
                M.StreamingCallScheduler = original_scheduler
        assert exact_instance_reached_scheduler_boundary, (
            "mutable exact-class plane did not reproduce expected identity bypass")

    # Scheduler retirement may prune dependency tokens only.  Physical
    # calendars, outstanding returns, counters and both digests must persist.
    # Full 122-member hashing can leave different allocator high-water marks
    # across the two Python runtimes.  The following tiny scheduler test pins
    # the memory observation below the cap so it tests state preservation, not
    # prior verifier allocation.  The hard cap itself is attacked separately.
    actual_peak_rss_kib = M.peak_rss_kib()
    saved_peak_for_synthetic = M.peak_rss_kib
    M.peak_rss_kib = lambda: min(actual_peak_rss_kib,
                                 M.PEAK_RSS_LIMIT_KIB - 1)
    try:
        state = M.StreamingCallScheduler("BIT_TYPED_K8")
        state.one(M.M.request("a", "BIT_TYPED_K8", "compute", [0], [0], 288,
                              produces="a"))
        state.one(M.M.request("b", "BIT_TYPED_K8", "compute", [0], [0], 288,
                              ["a"], "b"))
        physical = (dict(state.scheduler.next_port),
                    dict((key, list(value)) for key, value in
                         state.scheduler.outstanding.items()),
                    state.scheduler.requests, state.scheduler.last_cycle,
                    state.scheduler.address_digest.hexdigest(),
                    state.scheduler.commit_digest.hexdigest())
        state.retire_destination(("b",))
        assert set(state.scheduler.tokens) == set(["b"])
        assert physical == (dict(state.scheduler.next_port),
                            dict((key, list(value)) for key, value in
                                 state.scheduler.outstanding.items()),
                            state.scheduler.requests, state.scheduler.last_cycle,
                            state.scheduler.address_digest.hexdigest(),
                            state.scheduler.commit_digest.hexdigest())

        cache = M.M.WeightTileCache()
        misses = cache.prepare([(0, index) for index in range(8)])
        assert len(misses) == 8 and len(cache.key_to_slot) == 8
        assert cache.prepare([(0, index) for index in range(8)]) == []
        assert len(cache.key_to_slot) == 8 and cache.capacity == 9
        assert M.M.validate_resource() == "64661d825ee8ddbdccad9c3e09ca5e41c5ea9cfc75bcea394667dcfd91b4de10"

        synthetic = M.synthetic_self_test()
        assert [item["configuration"] for item in synthetic["results"]] == list(M.M.CONFIGS)
        assert len(set(item["commit_sequence_sha256"]
                       for item in synthetic["results"])) == 1
        assert all(item["streaming"]["peak_rss_kib"] < M.PEAK_RSS_LIMIT_KIB
                   for item in synthetic["results"])
    finally:
        M.peak_rss_kib = saved_peak_for_synthetic
    original_peak = M.peak_rss_kib
    try:
        M.peak_rss_kib = lambda: M.PEAK_RSS_LIMIT_KIB
        rejected("rss_cap", M.memory_gate, attacks)
    finally:
        M.peak_rss_kib = original_peak

    source_test = subprocess.check_output([sys.executable, str(TEST)],
                                          stderr=subprocess.STDOUT).decode("utf-8")
    assert "PASS M1549 source tests attacks=16 configs=3 pilot=0 production=0 product=0" in source_test

    # A second adversary: a subclass is still isinstance(MmapLittleBitPlane),
    # can self-assert the canonical path/SHA/shape, and can override bit().
    # Stop at scheduler construction, before request zero, to prove the guard
    # is bypassed without executing a pilot.
    class CanonicalIdentityCustomSubclass(M.MmapLittleBitPlane):
        @staticmethod
        def bit(_timestep, _channel, _y, _x):
            return 0

    forged = object.__new__(CanonicalIdentityCustomSubclass)
    forged.path = canonical
    forged.expected_sha256 = row["positive_output_sha256"]
    forged.timesteps = int(shape[0]); forged.channels = int(shape[2])
    forged.height = int(shape[3]); forged.width = int(shape[4])
    forged.elements = forged.timesteps * forged.channels * forged.height * forged.width
    forged.bytes = (forged.elements + 7) // 8
    forged._stream = None; forged._map = None
    original_scheduler = M.StreamingCallScheduler
    subclass_reached_scheduler_boundary = False
    try:
        M.StreamingCallScheduler = StopBeforeFirstRequest
        try:
            M.stream_tensor("BIT_TYPED_K8", forged, 0, 0)
        except FirstSchedulerBoundary:
            subclass_reached_scheduler_boundary = True
    finally:
        M.StreamingCallScheduler = original_scheduler
    assert subclass_reached_scheduler_boundary, (
        "custom subclass did not reproduce the expected fail-open boundary")

    assert len(attacks) == 17
    result = {
        "schema": "m1550_m1549_decoder_streaming_canonical_scope_independent_rehammer_output_r1_v1",
        "status": "NO_GO_OPENED_MMAP_IDENTITY_AND_SUBCLASS_BYPASS__NO_REQUEST_EXECUTED",
        "python": sys.version.split()[0],
        "bindings": {
            "source_sha256": EXPECTED[SOURCE],
            "test_sha256": EXPECTED[TEST],
            "contract_sha256": EXPECTED[CONTRACT],
            "author_review_sha256": EXPECTED[AUTHOR / "review.json"],
            "author_manifest_sha256": author_manifest_sha256,
            "author_outer_file_sha256": EXPECTED[AUTHOR / "SHA256SUMS.seal.sha256"],
            "m1546_review_sha256": EXPECTED[M1546 / "review.json"],
            "docs359_sha256": EXPECTED[DOCS359],
        },
        "verified": {
            "m1521_members": 122,
            "canonical_call": 0,
            "canonical_sample": 10,
            "canonical_module": 0,
            "canonical_regular_path_shape_sha_before_first_request": True,
            "three_nonproduct_configurations": list(M.M.CONFIGS),
            "product_cli_production_blocked": True,
            "calendar_outstanding_counter_digest_retention": True,
            "nine_slot_weight_cache_retention": True,
            "resource_manifest_sha256": M.M.validate_resource(),
            "rss_cap_kib": M.PEAK_RSS_LIMIT_KIB,
            "actual_rehammer_peak_rss_kib_after_full_identity_hash": actual_peak_rss_kib,
            "source_test_pass": True,
            "rejected_attack_count": len(attacks),
            "rejected_attacks": attacks,
        },
        "blocking_finding": {
            "entrypoint": "stream_tensor(config, plane, module, call_ordinal)",
            "finding": "the source authenticates mutable path/SHA attributes and re-hashes the canonical pathname, not the already-open descriptor/mmap; an exact-class object backed by corrupt foreign bytes and a subclass with custom bit() both reach scheduler construction",
            "exact_class_mutable_identity_bypass": exact_instance_reached_scheduler_boundary,
            "subclass_override_bypass": subclass_reached_scheduler_boundary,
            "reached_scheduler_constructor": True,
            "first_request_executed": False,
            "actual_pilot_executed": False,
            "production_executed": False,
            "product_executed": False,
            "minimum_fix": "remove externally supplied plane objects from the executable entrypoint or make canonical construction immutable; require exact type and hash/fstat the opened descriptor/mmap used by bit(), then reseal and rehammer",
        },
        "authorization": {
            "single_call_nonproduct_pilot": False,
            "production": False,
            "product_configuration": False,
            "automatic_retry": False,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
