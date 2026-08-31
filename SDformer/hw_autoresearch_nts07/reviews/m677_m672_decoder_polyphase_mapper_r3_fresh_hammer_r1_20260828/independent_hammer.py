#!/usr/bin/env python3
"""Independent CPU-only path-identity hammer for frozen M672 mapper r3."""

from __future__ import print_function

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / ("hw_autoresearch_nts07/system_simulator/scripts/"
                 "map_m672_decoder_convtranspose_polyphase_workload_r3.py")
spec = importlib.util.spec_from_file_location("m677_frozen_target", str(TARGET))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def require(value, message):
    if not value:
        raise AssertionError(message)


def expect_reject(action, label):
    try:
        action()
    except (RuntimeError, ValueError, TypeError, OSError):
        return
    raise AssertionError(label + " was accepted")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_bits(path, values):
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    Path(path).write_bytes(np.packbits(flat, bitorder="little").tobytes())
    return Path(path)


def make_manifest(package):
    shape = [1, 1, 1, 1, 8]
    binary = write_bits(package / "binary.bitpack", [0] * 8)
    scaled = write_bits(package / "scaled.bitpack", [0] * 8)
    binary_rows = []
    d1_rows = []
    for sample_id in range(10):
        for module_index in (0, 2, 3):
            binary_rows.append({
                "sample_id": sample_id,
                "module_index": module_index,
                "name": M.R2.M660_MODULE_NAMES[module_index],
                "route": "EXACT_BINARY_BITPACK",
                "relative_path": "binary.bitpack",
                "input_shape": shape,
                "input": {"packed_bytes": 1,
                          "packed_sha256": digest(binary)},
            })
        d1_rows.append({
            "sample_id": sample_id,
            "module_index": 1,
            "name": M.R2.M660_MODULE_NAMES[1],
            "route": "EXACT_SCALED_BINARY_BITPACK",
            "relative_path": "scaled.bitpack",
            "input_shape": shape,
            "theta_binary_candidate": {
                "packed_bytes": 1, "packed_sha256": digest(scaled)},
        })
    return {
        "schema": "m660_h67_ep35_layer_static_decoder_payload_v1",
        "packing": {"values": [0, 1], "bit_order": "little",
                    "order": "C_ORDER_FLAT",
                    "whole_call_contiguous_copy_allowed": False},
        "d0_d2_d3_binary_records": binary_rows,
        "d1_records": d1_rows,
    }


def direct_oracle(activation, weight):
    t_count, _batch, channels, height, width = activation.shape
    output = np.zeros((t_count, weight.shape[1], 2 * height, 2 * width),
                      dtype=np.int64)
    for t_index in range(t_count):
        for channel in range(channels):
            for source_y in range(height):
                for source_x in range(width):
                    source = int(activation[t_index, 0, channel,
                                            source_y, source_x])
                    for kernel_y, kernel_x in M.M514_SLOT_ORDER:
                        destination_y = 2 * source_y - 1 + kernel_y
                        destination_x = 2 * source_x - 1 + kernel_x
                        if (0 <= destination_y < 2 * height and
                                0 <= destination_x < 2 * width):
                            output[t_index, :, destination_y,
                                   destination_x] += (
                                source * weight[channel, :,
                                                kernel_y, kernel_x])
    return output


checks = []
details = {}

require(digest(TARGET) ==
        "989094c739ac12c448faf1e1388374bdabdb3bd5e4ebab6dd17aadf16ecf8254",
        "frozen M672 mapper SHA drift")
require(digest(Path(M.R2.__file__)) ==
        "875b31ed1994729cc29321af0053fcea5586077aa468398d31eb4fe0fdb1596b",
        "frozen M670-r2 mapper SHA drift")
checks.append("frozen_mapper_hashes")


# Rebuild the arithmetic on fresh non-square inputs, rather than treating the
# author suite or r2 wrapper as a numerical oracle.
numeric = []
with tempfile.TemporaryDirectory(prefix="m677_numeric_") as temporary:
    package = Path(temporary)
    for seed, shape, cout in ((677011, (2, 1, 2, 2, 5), 3),
                              (677037, (1, 1, 3, 4, 2), 2),
                              (677099, (3, 1, 1, 3, 4), 4)):
        rng = np.random.RandomState(seed)
        activation = rng.randint(0, 2, size=shape).astype(np.uint8)
        weight = rng.randint(-5, 6,
                             size=(shape[2], cout, 3, 3)).astype(np.int16)
        payload = write_bits(package / ("x{}.bitpack".format(seed)),
                             activation)
        observed = M.reconstruct_convtranspose(
            payload, shape, weight, tile_m=3, trusted_root=package)
        expected = direct_oracle(activation, weight)
        require(np.array_equal(observed, expected),
                "fresh independent ConvTranspose oracle mismatch")
        account = M.workload_accounting(
            payload, shape, output_channels=cout, tile_m=2,
            trusted_root=package)
        require(account["active_products"] ==
                account["active_tap_events"] * cout,
                "active-product conservation mismatch")
        numeric.append({"seed": seed, "shape": list(shape),
                        "active_taps": account["active_tap_events"]})
details["fresh_numeric"] = numeric
checks.append("fresh_non_square_oracle_and_accounting")


# Reproduce M671's dual-file exploit against every public consumer.  The
# trusted package contains zero; the current directory contains the same
# relative name with all ones.  Every result must come from package/zero.
with tempfile.TemporaryDirectory(prefix="m677_dual_file_") as temporary:
    base = Path(temporary)
    package = base / "package"
    work = base / "work"
    package.mkdir()
    work.mkdir()
    payload = write_bits(package / "same.bitpack", [0] * 8)
    write_bits(work / "same.bitpack", [1] * 8)
    shape = (1, 1, 1, 1, 8)
    weight = np.ones((1, 2, 3, 3), dtype=np.int16)
    expected_path = payload.resolve()
    captured = {}

    # Runtime argument probes establish the exact path handed across each r3
    # to r2 data-consuming API boundary.
    originals = {}
    for name in ("iter_polyphase_tiles", "materialize_polyphase",
                 "reconstruct_convtranspose", "workload_accounting"):
        originals[name] = getattr(M.R2, name)

    def wrap(name):
        original = originals[name]
        def probe(*args, **kwargs):
            captured.setdefault(name, []).append(str(args[0]))
            return original(*args, **kwargs)
        return probe

    old_cwd = os.getcwd()
    os.chdir(str(work))
    try:
        identity = M.validate_bitpack(
            "same.bitpack", shape, trusted_root=package)

        M.R2.iter_polyphase_tiles = wrap("iter_polyphase_tiles")
        tiles = list(M.iter_polyphase_tiles(
            "same.bitpack", shape, tile_m=2, trusted_root=package))
        M.R2.iter_polyphase_tiles = originals["iter_polyphase_tiles"]

        M.R2.materialize_polyphase = wrap("materialize_polyphase")
        matrices, _metadata = M.materialize_polyphase(
            "same.bitpack", shape, tile_m=2, trusted_root=package)
        M.R2.materialize_polyphase = originals["materialize_polyphase"]

        M.R2.reconstruct_convtranspose = wrap("reconstruct_convtranspose")
        reconstructed = M.reconstruct_convtranspose(
            "same.bitpack", shape, weight, tile_m=2,
            trusted_root=package)
        M.R2.reconstruct_convtranspose = originals[
            "reconstruct_convtranspose"]

        M.R2.workload_accounting = wrap("workload_accounting")
        accounting = M.workload_accounting(
            "same.bitpack", shape, output_channels=2, tile_m=2,
            trusted_root=package)
        M.R2.workload_accounting = originals["workload_accounting"]
    finally:
        for name, original in originals.items():
            setattr(M.R2, name, original)
        os.chdir(old_cwd)

    require(Path(identity["path"]) == expected_path and
            identity["sha256"] == digest(payload),
            "validate_bitpack did not bind package payload")
    require(sum(int(tile["values"].sum()) for tile in tiles) == 0,
            "iterator consumed CWD collision")
    require(sum(int(value.sum()) for value in matrices.values()) == 0,
            "materializer consumed CWD collision")
    require(int(reconstructed.sum()) == 0,
            "reconstructor consumed CWD collision")
    require(accounting["source_popcount"] == 0 and
            accounting["active_tap_events"] == 0,
            "accounting consumed CWD collision")
    for name in originals:
        require(captured.get(name), name + " boundary was not observed")
        require(all(Path(path).is_absolute() and Path(path) == expected_path
                    for path in captured[name]),
                name + " did not receive exact validated absolute path")
    details["dual_file"] = {
        "validated_path": str(expected_path),
        "iterator_sum": 0,
        "materialized_sum": 0,
        "reconstructed_sum": 0,
        "source_popcount": accounting["source_popcount"],
        "all_r2_boundaries_absolute_and_equal": True,
    }
checks.append("all_public_bitpack_consumers_close_dual_file_identity_split")


# Manifest and payload names receive an independent dual-file attack.  A
# malformed CWD manifest and all-one CWD payloads must not affect the 40
# records validated under the package root.
with tempfile.TemporaryDirectory(prefix="m677_manifest_dual_") as temporary:
    base = Path(temporary)
    package = base / "package"
    work = base / "work"
    package.mkdir()
    work.mkdir()
    manifest = make_manifest(package)
    manifest_path = package / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (work / "manifest.json").write_text("{}", encoding="utf-8")
    write_bits(work / "binary.bitpack", [1] * 8)
    write_bits(work / "scaled.bitpack", [1] * 8)
    old_cwd = os.getcwd()
    os.chdir(str(work))
    try:
        records = M.m660_bitpack_records(
            "manifest.json", trusted_root=package)
    finally:
        os.chdir(old_cwd)
    require(len(records) == 40, "package manifest was not consumed")
    allowed = set((package / name).resolve()
                  for name in ("binary.bitpack", "scaled.bitpack"))
    require(all(Path(record["path"]).is_absolute() and
                Path(record["path"]) in allowed for record in records),
            "M660 record contains a non-package or non-absolute payload")
    require(all(record["packed_sha256"] == digest(Path(record["path"]))
                for record in records),
            "M660 returned payload identity drift")
    details["m660_dual_file"] = {
        "records": len(records),
        "all_payloads_absolute_under_package": True,
        "cwd_manifest_malformed": True,
        "cwd_payloads_opposite": True,
    }
checks.append("m660_manifest_and_payload_dual_file_identity_closed")


# A symlink in any trusted-root ancestor must be rejected before any r2
# validator/consumer is called.  Exercise every public data entry point.
with tempfile.TemporaryDirectory(prefix="m677_linked_ancestor_") as temporary:
    base = Path(temporary)
    real_parent = base / "real_parent"
    package = real_parent / "package"
    package.mkdir(parents=True)
    payload = write_bits(package / "x.bitpack", [0] * 8)
    manifest = make_manifest(package)
    (package / "manifest.json").write_text(json.dumps(manifest),
                                            encoding="utf-8")
    alias = base / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    linked_root = alias / "package"
    linked_payload = linked_root / payload.name
    shape = (1, 1, 1, 1, 8)
    weight = np.ones((1, 1, 3, 3), dtype=np.int16)

    expect_reject(lambda: M.validate_bitpack(
        linked_payload, shape, trusted_root=linked_root), "linked validate")
    expect_reject(lambda: list(M.iter_polyphase_tiles(
        linked_payload, shape, trusted_root=linked_root)), "linked iterator")
    expect_reject(lambda: M.materialize_polyphase(
        linked_payload, shape, trusted_root=linked_root),
        "linked materializer")
    expect_reject(lambda: M.reconstruct_convtranspose(
        linked_payload, shape, weight, trusted_root=linked_root),
        "linked reconstructor")
    expect_reject(lambda: M.workload_accounting(
        linked_payload, shape, trusted_root=linked_root),
        "linked accounting")
    expect_reject(lambda: M.trusted_regular_file(
        linked_root, linked_payload, "linked direct"),
        "linked trusted_regular_file")
    expect_reject(lambda: M.m660_bitpack_records(
        linked_root / "manifest.json", trusted_root=linked_root),
        "linked M660 manifest")

    # Prove the rejection occurs in r3's root walk, before delegated reads.
    original_validate = M.R2.validate_bitpack
    original_records = M.R2.m660_bitpack_records
    delegated = {"validate": 0, "records": 0}
    def forbidden_validate(*args, **kwargs):
        delegated["validate"] += 1
        raise AssertionError("delegated validate reached")
    def forbidden_records(*args, **kwargs):
        delegated["records"] += 1
        raise AssertionError("delegated records reached")
    M.R2.validate_bitpack = forbidden_validate
    M.R2.m660_bitpack_records = forbidden_records
    try:
        expect_reject(lambda: M.validate_bitpack(
            linked_payload, shape, trusted_root=linked_root),
            "pre-delegation linked validate")
        expect_reject(lambda: M.m660_bitpack_records(
            linked_root / "manifest.json", trusted_root=linked_root),
            "pre-delegation linked manifest")
    finally:
        M.R2.validate_bitpack = original_validate
        M.R2.m660_bitpack_records = original_records
    require(delegated == {"validate": 0, "records": 0},
            "linked root reached delegated data code")
    details["linked_ancestor"] = {
        "public_entry_points_rejected": 7,
        "delegated_data_calls_before_rejection": 0,
    }
checks.append("all_public_data_entries_reject_linked_root_ancestor")


# Lexically ambiguous roots remain fail closed.  This is separate from the
# linked-ancestor attack and guards accidental normalization regressions.
for root in ("relative", "/", "/tmp//x", "/tmp/./x", "/tmp/../x"):
    expect_reject(lambda value=root: M.trusted_root_all_components(value),
                  "ambiguous root " + root)
checks.append("ambiguous_trusted_roots_rejected")


print(json.dumps({
    "schema": "m677_m672_decoder_polyphase_mapper_r3_independent_hammer_v1",
    "status": "PASS_P0_0_P1_0__GO_M660_PAYLOAD_INTEGRATION_ONLY",
    "python": sys.executable,
    "python_version": sys.version,
    "numpy_version": np.__version__,
    "checks": checks,
    "details": details,
    "severity": {"p0": 0, "p1": 0, "p2": 0},
    "scope_boundary": {
        "trusted_package_immutable_during_one_evaluation": True,
        "hostile_concurrent_filesystem_mutation_protected": False,
        "gpu": False,
        "eda": False,
        "performance": False,
        "production_mapping": False,
    },
}, indent=2, sort_keys=True))
