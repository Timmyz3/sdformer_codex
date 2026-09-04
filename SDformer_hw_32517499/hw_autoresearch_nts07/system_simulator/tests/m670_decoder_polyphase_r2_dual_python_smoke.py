#!/usr/bin/env python3
"""Torch-free dual-Python smoke for the M670/M665-r2 mapper boundary."""

from __future__ import print_function

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / (
    "hw_autoresearch_nts07/system_simulator/scripts/"
    "map_m670_decoder_convtranspose_polyphase_workload_r2.py")


def load_target():
    spec = importlib.util.spec_from_file_location("m670_dual", str(TARGET))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M670 = load_target()


def expect_reject(action, label):
    try:
        action()
    except RuntimeError:
        return
    raise AssertionError(label + " was accepted")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_bitpack(path, values):
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    path.write_bytes(np.packbits(flat, bitorder="little").tobytes())
    return path


def make_manifest(package):
    shape = [1, 1, 1, 1, 8]
    binary = write_bitpack(
        package / "binary.bitpack", [0, 1, 0, 1, 1, 0, 0, 0])
    scaled = write_bitpack(
        package / "scaled.bitpack", [1, 0, 1, 0, 0, 1, 0, 0])
    rows = []
    d1_rows = []
    for sample_id in range(10):
        for module_index in M670.M660_BINARY_MODULES:
            rows.append({
                "sample_id": sample_id,
                "module_index": module_index,
                "name": M670.M660_MODULE_NAMES[module_index],
                "route": "EXACT_BINARY_BITPACK",
                "relative_path": "binary.bitpack",
                "input_shape": shape,
                "input": {"packed_bytes": 1,
                          "packed_sha256": digest(binary)},
            })
        d1_rows.append({
            "sample_id": sample_id,
            "module_index": 1,
            "name": M670.M660_MODULE_NAMES[1],
            "route": "EXACT_SCALED_BINARY_BITPACK",
            "relative_path": "scaled.bitpack",
            "input_shape": shape,
            "input": {"packed_bytes": 1, "packed_sha256": "0" * 64},
            "theta_binary_candidate": {
                "packed_bytes": 1, "packed_sha256": digest(scaled)},
        })
    return {
        "schema": "m660_h67_ep35_layer_static_decoder_payload_v1",
        "packing": {"values": [0, 1], "bit_order": "little",
                    "order": "C_ORDER_FLAT",
                    "whole_call_contiguous_copy_allowed": False},
        "d0_d2_d3_binary_records": rows,
        "d1_records": d1_rows,
    }


def write_manifest(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def main():
    checks = []
    assert M670.expected_packed_bytes((1, 1, 1, 1, 9)) == 2
    for shape in ((2 ** 32, 1, 2 ** 32, 1, 1),
                  (2 ** 20, 1, 2 ** 20, 1, 1),
                  (0, 1, 1, 1, 1), (True, 1, 1, 1, 1),
                  (np.bool_(True), 1, 1, 1, 1)):
        expect_reject(lambda value=shape: M670.expected_packed_bytes(value),
                      "bounded shape")
    checks.append("bounded_python_integer_shape_product")

    expect_reject(lambda: M670.phase_bank(True, 0), "bool phase")
    expect_reject(lambda: M670.build_phase_plan(True, 1, 1, 3),
                  "bool channel")
    expect_reject(lambda: M670.validate_convtranspose_spec(groups=True),
                  "bool groups")
    checks.append("boolean_integer_aliases")

    with tempfile.TemporaryDirectory(prefix="m670_dual_") as temporary:
        base = Path(temporary)
        package = base / "package"
        package.mkdir()
        payload = write_bitpack(
            package / "x.bitpack", [1, 0, 0, 0, 0, 0, 0, 1, 1])
        identity = M670.validate_bitpack(
            payload, (1, 1, 1, 1, 9), trusted_root=package)
        assert identity["elements"] == 9 and identity["packed_bytes"] == 2
        mapped, _metadata = M670.materialize_polyphase(
            payload, (1, 1, 1, 1, 9), tile_m=3, trusted_root=package)
        assert list(mapped) == [3, 2, 1, 0]
        checks.append("little_bit_and_polyphase_core")

        outside = base / "outside"
        outside.mkdir()
        escaped = write_bitpack(outside / "escaped.bitpack", [0] * 8)
        alias = package / "alias"
        alias.symlink_to(outside, target_is_directory=True)
        expect_reject(lambda: M670.validate_bitpack(
            alias / escaped.name, (1, 1, 1, 1, 8), trusted_root=package),
            "parent symlink")
        expect_reject(lambda: M670.validate_bitpack(
            "../outside/escaped.bitpack", (1, 1, 1, 1, 8),
            trusted_root=package), "traversal")
        checks.append("trusted_root_parent_symlink_and_traversal")

        manifest = make_manifest(package)
        manifest_path = write_manifest(package / "manifest.json", manifest)
        records = M670.m660_bitpack_records(
            manifest_path, trusted_root=package)
        assert len(records) == 40
        assert sum(row["module_index"] == 1 for row in records) == 10
        checks.append("complete_route_module_name_lattice")

        crossed = copy.deepcopy(manifest)
        crossed["d0_d2_d3_binary_records"][0]["route"] = \
            "EXACT_SCALED_BINARY_BITPACK"
        write_manifest(manifest_path, crossed)
        expect_reject(lambda: M670.m660_bitpack_records(
            manifest_path, trusted_root=package), "cross route")
        missing = copy.deepcopy(manifest)
        missing["d1_records"].pop()
        write_manifest(manifest_path, missing)
        expect_reject(lambda: M670.m660_bitpack_records(
            manifest_path, trusted_root=package), "missing lattice")
        boolean = copy.deepcopy(manifest)
        boolean["d0_d2_d3_binary_records"][0]["sample_id"] = True
        write_manifest(manifest_path, boolean)
        expect_reject(lambda: M670.m660_bitpack_records(
            manifest_path, trusted_root=package), "bool record integer")
        checks.append("cross_route_missing_and_bool_record_attacks")

    print(json.dumps({
        "schema": "m670_decoder_polyphase_r2_dual_python_smoke_v1",
        "status": "PASS_M670_R2_DUAL_PYTHON_STATIC_SMOKE",
        "python": sys.executable,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "checks": checks,
        "claim_boundary": {"gpu": False, "eda": False,
                           "cycles": False, "speedup": False},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
