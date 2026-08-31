#!/usr/bin/env python3
"""Independent fail-closed hammer for the exact M1553 source bytes.

This harness never schedules request zero.  Entry-point probes replace the
captured scheduler's ``one`` method with a sentinel before invoking the
canonical entry.  It never runs a pilot, production population, product
configuration, GPU, RTL, EDA, or remote command.

The syntax is compatible with CPython 3.6.
"""
from __future__ import print_function

import hashlib
import importlib.util
import inspect
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
CONTRACT = HW / "contracts/m1553_ep34_decoder_nonproduct_fd_bound_streaming_source_contract_r1_20260831.json"
AUTHOR = HW / "results/m1553_ep34_decoder_nonproduct_fd_bound_streaming_source_r1_20260831/receipt.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    SOURCE: "177410fd4a70f401c734ab09b99bc64f9a14d443b614592d6c76eae76b5e7f00",
    TEST: "8a2a32446ea11d4708f7e82dc5b9eac139106ee1eada5e0ccd70e082de35549e",
    CONTRACT: "b1ec28932f14464da0ee16a032cf2d23f338c60d30ecb689d2ff00e8f22238d8",
    AUTHOR: "72cc59aacefe770e9e047a0657e6c8d835cb6bb6e7450673881c0de799f5272d",
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


for _path, _expected in EXPECTED.items():
    assert _path.is_file() and sha256(_path) == _expected, (
        "identity drift: " + str(_path))

SPEC = importlib.util.spec_from_file_location("m1554_bound_m1553", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class FirstRequestBoundary(RuntimeError):
    pass


def stop_before_first_request(self, _row):
    raise FirstRequestBoundary("identity boundary passed; request zero not scheduled")


def reaches_first_request_boundary(function):
    """Run an entry probe while making request zero impossible."""
    original = M.StreamingCallScheduler.one
    M.StreamingCallScheduler.one = stop_before_first_request
    try:
        try:
            function()
        except FirstRequestBoundary:
            return True
    finally:
        M.StreamingCallScheduler.one = original
    return False


def main():
    attacks = []
    verified = {}
    description = M.describe()
    assert description["schema"] == (
        "m1553_ep34_decoder_nonproduct_streaming_single_call_pilot_fd_bound_source_r3_v1")
    assert list(inspect.signature(M.stream_actual_call).parameters) == ["config"]
    assert not hasattr(M, "stream_tensor")
    assert description["source_capabilities"]["external_plane_parameter"] is False
    assert description["source_capabilities"]["pilot_cli"] is False
    assert description["source_capabilities"]["production_cli"] is False

    authority = M.validate_authorities(False)
    assert authority["pilot_execution"] is False and authority["production"] is False
    row0 = M.selected_pilot_record()
    assert row0["global_call_ordinal"] == 0 and row0["global_sample_id"] == 10
    assert row0["module_ordinal"] == 0 and tuple(row0["shape"]) == M.M.INPUT_SHAPES[0]
    canonical0 = M.M.M1521_ROOT / row0["positive_output"]
    assert canonical0.is_file() and not canonical0.is_symlink()
    assert sha256(canonical0) == row0["positive_output_sha256"]

    rejected("product_internal_entry", lambda: M.stream_actual_call(
        M.FORBIDDEN_CONFIG), attacks)
    rejected("pilot_release", M.pilot_release, attacks)
    rejected("production_release", M.production_release, attacks)
    for option, label in (("--pilot", "pilot_cli"),
                          ("--production", "production_cli"),
                          ("--product", "product_cli")):
        rejected(label, lambda option=option: subprocess.check_output(
            [sys.executable, str(SOURCE), option], stderr=subprocess.STDOUT), attacks)

    # The ordinary module names are captured at construction: replacing them
    # after import is ignored.  The sentinel proves only that the clean,
    # internally opened call reaches the boundary before request zero.
    original_plane_name = M.MmapLittleBitPlane
    original_selector_name = M.selected_pilot_record
    original_upstream_root = M.M.M1521_ROOT
    try:
        class RejectPlane(object):
            def __init__(self, *_args, **_kwargs):
                raise AssertionError("replacement plane name was consulted")
        M.MmapLittleBitPlane = RejectPlane
        assert reaches_first_request_boundary(lambda: M.stream_actual_call(
            "BIT_TYPED_K8"))
        verified["module_plane_name_replacement_ignored"] = True
    finally:
        M.MmapLittleBitPlane = original_plane_name
    try:
        M.selected_pilot_record = lambda: (_ for _ in ()).throw(
            AssertionError("replacement selector name was consulted"))
        assert reaches_first_request_boundary(lambda: M.stream_actual_call(
            "BIT_TYPED_K8"))
        verified["module_selector_name_replacement_ignored"] = True
    finally:
        M.selected_pilot_record = original_selector_name
    with tempfile.TemporaryDirectory(prefix="m1554_root.", dir=str(HERE)) as directory:
        try:
            M.M.M1521_ROOT = Path(directory)
            assert reaches_first_request_boundary(lambda: M.stream_actual_call(
                "BIT_TYPED_K8"))
            verified["upstream_root_name_replacement_ignored"] = True
        finally:
            M.M.M1521_ROOT = original_upstream_root

    # Subclasses/custom planes cannot be passed through the only public entry.
    class SubclassPlane(original_plane_name):
        @staticmethod
        def bit(_timestep, _channel, _y, _x):
            return 0
    try:
        M.stream_actual_call("BIT_TYPED_K8", SubclassPlane)
    except TypeError:
        attacks.append("subclass_argument_injection")
    else:
        raise AssertionError("subclass argument injection accepted")
    verified["exact_type_is_constructed_internally"] = True

    shape = (10, 1, 8, 2, 2)
    byte_count = (10 * 8 * 2 * 2 + 7) // 8
    inplace_mutation_visible = False
    with tempfile.TemporaryDirectory(prefix="m1554_fd.", dir=str(HERE)) as directory:
        root = Path(directory)
        original = bytes(byte_count)
        replacement = bytes([0xff]) * byte_count
        path = root / "plane.bitpack"
        path.write_bytes(original)
        expected = sha256(path)

        # Symlinks are rejected, including a link to byte-identical content.
        symlink = root / "plane.symlink"
        symlink.symlink_to(path)
        rejected("symlink_payload", lambda: M.MmapLittleBitPlane(
            symlink, shape, expected), attacks)

        # The low-level reader can open a regular hardlink, but no caller can
        # inject that path into canonical_entry and the root/path are captured.
        hardlink = root / "plane.hardlink"
        os.link(str(path), str(hardlink))
        with M.MmapLittleBitPlane(hardlink, shape, expected) as plane:
            assert plane.opened_inode == path.stat().st_ino
            assert plane.bit(0, 0, 0, 0) == 0
        verified["hardlink_has_no_public_entry_injection_seam"] = True

        # A pathname replacement after open does not alter the mapped fd.
        moved = root / "plane.opened"
        with M.MmapLittleBitPlane(path, shape, expected) as plane:
            path.rename(moved)
            path.write_bytes(replacement)
            assert plane.bit(0, 0, 0, 0) == 0
            assert plane.opened_sha256 == expected
            verified["rename_replace_fd_binding"] = True

        # lstat/open TOCTOU is rejected by the dev/inode/size equality check.
        race = root / "race.bitpack"
        race_old = root / "race.old"
        race.write_bytes(original)
        original_open = M.os.open
        swapped = [False]
        def racing_open(name, flags):
            if Path(name) == race and not swapped[0]:
                swapped[0] = True
                race.rename(race_old)
                race.write_bytes(replacement)
            return original_open(name, flags)
        try:
            M.os.open = racing_open
            rejected("lstat_open_inode_swap", lambda: M.MmapLittleBitPlane(
                race, shape, expected), attacks)
        finally:
            M.os.open = original_open

        # Blocking finding: mmap.ACCESS_READ remains backed by the mutable
        # inode.  An in-place write after the one-time fd hash changes bit()
        # while opened_sha256 still asserts the pre-write identity.
        mutable = root / "mutable.bitpack"
        mutable.write_bytes(original)
        mutable_sha = sha256(mutable)
        with M.MmapLittleBitPlane(mutable, shape, mutable_sha) as plane:
            assert type(plane) is original_plane_name
            assert plane.bit(0, 0, 0, 0) == 0
            with mutable.open("r+b", buffering=0) as stream:
                stream.write(b"\x01")
                stream.flush()
                os.fsync(stream.fileno())
            inplace_mutation_visible = (
                plane.bit(0, 0, 0, 0) == 1 and
                plane.opened_sha256 == mutable_sha and
                sha256(mutable) != mutable_sha)
        assert inplace_mutation_visible, "in-place mmap mutation witness disappeared"

    # A second blocking finding: the captured selector is a function whose
    # global M remains replaceable.  Substitute another *sealed population*
    # row (D0 sample 11/call 4), rewrite only its logical identity fields, and
    # the entry opens it as call 0.  Stop before request zero.
    upstream_original = M.M
    manifest = upstream_original.strict_json(upstream_original.M1521_MANIFEST)
    row4 = dict(manifest["records"][4])
    assert row4["module_ordinal"] == 0 and row4["global_sample_id"] == 11
    forged = dict(row4)
    forged["global_call_ordinal"] = 0
    forged["global_sample_id"] = 10
    class ForgedSelectorUpstream(object):
        INPUT_SHAPES = upstream_original.INPUT_SHAPES
        M1521_MANIFEST = upstream_original.M1521_MANIFEST
        def __getattr__(self, name):
            return getattr(upstream_original, name)
        @staticmethod
        def strict_json(_path):
            return {"records": [forged]}
        @staticmethod
        def validate_population_manifest(_manifest):
            return {"calls": 120}
    wrong_sample_reached_boundary = False
    try:
        M.M = ForgedSelectorUpstream()
        wrong_sample_reached_boundary = reaches_first_request_boundary(
            lambda: M.stream_actual_call("BIT_TYPED_K8"))
    finally:
        M.M = upstream_original
    assert wrong_sample_reached_boundary, (
        "mutable selector-global wrong-sample witness did not reach request boundary")

    # Author tests remain reproducible in both requested runtimes; they do not
    # cover either blocking witness above.
    source_test = subprocess.check_output([sys.executable, str(TEST)],
                                          stderr=subprocess.STDOUT).decode("utf-8")
    assert "PASS M1553 source tests attacks=15 configs=3 fd_bound=1 pilot=0 production=0 product=0" in source_test

    result = {
        "schema": "m1554_m1553_decoder_fd_bound_streaming_independent_hammer_output_r1_v1",
        "status": "NO_GO_M1553_SINGLE_CALL_PILOT__MUTABLE_MMAP_AND_SELECTOR_GLOBAL_BYPASSES__NO_REQUEST_EXECUTED",
        "python": sys.version.split()[0],
        "bindings": dict((path.name, digest) for path, digest in EXPECTED.items()),
        "verified_passes": {
            "entry_signature_only_config": True,
            "external_plane_module_call_parameters_absent": True,
            "module_plane_name_replacement_ignored": True,
            "module_selector_name_replacement_ignored": True,
            "upstream_root_name_replacement_ignored": True,
            "subclass_argument_injection_rejected": True,
            "symlink_rejected": True,
            "hardlink_has_no_public_entry_injection_seam": True,
            "lstat_open_inode_swap_rejected": True,
            "rename_replace_preserves_open_fd_bytes": True,
            "product_pilot_production_cli_blocked": True,
            "author_source_test_pass": True,
            "ordinary_rejected_attack_count": len(attacks),
            "ordinary_rejected_attacks": attacks,
        },
        "blocking_findings": [
            {
                "severity": "P0_FAIL_CLOSED",
                "name": "post_hash_inplace_inode_mutation_changes_mmap_bytes",
                "finding": "The one-time opened-fd SHA authenticates pre-mmap bytes, but mmap.ACCESS_READ remains backed by the mutable inode; an in-place write changes bit() while opened_sha256 remains the admitted digest.",
                "witness_reproduced": inplace_mutation_visible,
                "reached_scheduler_constructor": False,
                "first_request_executed": False,
            },
            {
                "severity": "P0_FAIL_CLOSED",
                "name": "captured_selector_function_retains_mutable_module_global",
                "finding": "The closure captures selected_pilot_record as a function, but that function resolves global M at call time. Replacing that global admits another canonical D0 payload (sample 11/call 4) after rewriting only logical row fields, and reaches the boundary before request zero.",
                "witness_reproduced": wrong_sample_reached_boundary,
                "reached_scheduler_constructor": True,
                "first_request_executed": False,
            },
        ],
        "minimum_fix": "Snapshot and pin the exact call-0 row/path/SHA inside the canonical closure at clean import, without a function that resolves mutable globals; copy the verified compact bitplane into immutable anonymous storage (or equivalently guarantee and verify immutable bytes for the full scheduling interval), then reseal and independently rehammer in a fresh process.",
        "execution": {
            "actual_pilot": False,
            "first_request": False,
            "production": False,
            "product_configuration": False,
            "gpu": False,
            "ssh": False,
            "rtl_eda": False,
        },
        "authorization": {
            "single_call_d0_three_nonproduct_pilot": False,
            "production": False,
            "product_configuration": False,
            "successor_hardening_authoring": True,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
