#!/usr/bin/env python3
"""Independent CPU-only hammer for frozen M665; never imports author tests."""

from __future__ import print_function

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

try:
    import torch
    import torch.nn.functional as torch_functional
except ImportError:
    torch = None
    torch_functional = None


ROOT = Path(__file__).resolve().parents[3]
MAPPER = ROOT / (
    "hw_autoresearch_nts07/system_simulator/scripts/"
    "map_m665_decoder_convtranspose_polyphase_workload.py")
RTL = ROOT / (
    "hw_autoresearch_nts07/rtl_m514/"
    "m514_c2_convtranspose_k3s2_polyphase_address_mapper.sv")
VCS_CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/m514_c2d_directed_vcs_contract_r1_20260827.json")
EXPECTED_MAPPER_SHA = (
    "07dd6474764993add120091514334deb02c5a71caa0c9955b85d8f577634abd4")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


require(digest(MAPPER) == EXPECTED_MAPPER_SHA, "frozen mapper SHA drift")
spec = importlib.util.spec_from_file_location("m667_frozen_m665", str(MAPPER))
M665 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M665)


checks = []
findings = []


def check(name, function):
    try:
        detail = function()
        checks.append({"name": name, "pass": True, "detail": detail})
    except Exception as error:
        checks.append({"name": name, "pass": False,
                       "detail": "{}: {}".format(
                           type(error).__name__, error)})


def expect_reject(function, contains=None):
    try:
        function()
    except Exception as error:
        if contains is not None:
            require(contains in str(error),
                    "wrong rejection: {}".format(error))
        return type(error).__name__ + ": " + str(error)
    raise AssertionError("attack was accepted")


def write_pack(path, values, bitorder="little"):
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    Path(path).write_bytes(np.packbits(flat, bitorder=bitorder).tobytes())
    return Path(path)


def direct_oracle(activation, weight):
    activation = np.asarray(activation, dtype=np.int64)
    weight = np.asarray(weight, dtype=np.int64)
    time_steps, batch, channels, height, width = activation.shape
    require(batch == 1 and weight.shape[0] == channels,
            "oracle shape mismatch")
    output = np.zeros((time_steps, weight.shape[1], 2 * height, 2 * width),
                      dtype=np.int64)
    for timestep in range(time_steps):
        for source_channel in range(channels):
            for source_y in range(height):
                for source_x in range(width):
                    value = int(activation[timestep, 0, source_channel,
                                           source_y, source_x])
                    for kernel_y in range(3):
                        for kernel_x in range(3):
                            destination_y = 2 * source_y - 1 + kernel_y
                            destination_x = 2 * source_x - 1 + kernel_x
                            if (0 <= destination_y < 2 * height and
                                    0 <= destination_x < 2 * width):
                                output[timestep, :, destination_y,
                                       destination_x] += (
                                    value * weight[source_channel, :,
                                                   kernel_y, kernel_x])
    return output


def test_rtl_derived_phase_taps():
    text = RTL.read_text(encoding="utf-8")
    parsed = {}
    for slot, kernel_y, kernel_x in re.findall(
            r"(\d): begin selected_ky = (\d); selected_kx = (\d); end",
            text):
        parsed[int(slot)] = (int(kernel_y), int(kernel_x))
    require(len(parsed) == 8 and
            "default: begin selected_ky = 1; selected_kx = 1; end" in text,
            "cannot derive all RTL slots")
    parsed[8] = (1, 1)
    slot_order = tuple(parsed[index] for index in range(9))
    require(slot_order == M665.M514_SLOT_ORDER, "RTL slot order mismatch")
    grouped = {3: [], 2: [], 1: [], 0: []}
    for kernel_y, kernel_x in slot_order:
        bank = (((kernel_y - 1) & 1) << 1) | ((kernel_x - 1) & 1)
        grouped[bank].append((kernel_y, kernel_x))
    grouped = {bank: tuple(value) for bank, value in grouped.items()}
    require(grouped == M665.M514_PHASE_TAPS, "RTL phase/tap mismatch")
    contract = json.loads(VCS_CONTRACT.read_text(encoding="utf-8"))
    require(contract["functional_contract"]["coordinate_formula"] ==
            "destination = 2 * source - 1 + kernel_index",
            "VCS coordinate contract mismatch")
    require(contract["functional_contract"]["boundary_fanout"] ==
            [4, 6, 6, 9], "VCS boundary fanout mismatch")
    return {"slot_order": slot_order,
            "phase_order": M665.M514_PHASE_ORDER,
            "phase_taps": grouped}


def test_fresh_oracles():
    cases = [
        (667003, (1, 1, 1, 1, 5), 2),
        (667021, (2, 1, 2, 2, 5), 3),
        (667089, (3, 1, 3, 4, 2), 4),
    ]
    results = []
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        for seed, shape, output_channels in cases:
            rng = np.random.default_rng(seed)
            activation = rng.integers(0, 2, size=shape, dtype=np.uint8)
            weight = rng.integers(-5, 6,
                                  size=(shape[2], output_channels, 3, 3),
                                  dtype=np.int16)
            path = write_pack(directory / ("{}.bitpack".format(seed)),
                              activation)
            observed = M665.reconstruct_convtranspose(
                path, shape, weight, tile_m=3)
            independent = direct_oracle(activation, weight)
            require(np.array_equal(observed, independent),
                    "independent nested-loop mismatch seed {}".format(seed))
            torch_equal = None
            if torch is not None:
                expected = torch_functional.conv_transpose2d(
                    torch.from_numpy(activation[:, 0].astype(np.float64)),
                    torch.from_numpy(weight.astype(np.float64)), bias=None,
                    stride=(2, 2), padding=(1, 1), output_padding=(1, 1),
                    groups=1, dilation=(1, 1)).numpy()
                torch_equal = bool(np.array_equal(observed, expected))
                require(torch_equal, "fresh PyTorch mismatch seed {}".format(
                    seed))
            results.append({"seed": seed, "shape": shape,
                            "output_channels": output_channels,
                            "torch_equal": torch_equal})
    return results


def test_plan_every_coordinate_and_k_slot():
    channels, height, width = 3, 3, 5
    for bank, taps in M665.M514_PHASE_TAPS.items():
        plan = M665.build_phase_plan(channels, height, width, bank)
        require(plan["m"] == height * width and
                plan["k"] == len(taps) * channels,
                "plan dimensions mismatch")
        seen_destinations = set()
        for m_index in range(plan["m"]):
            destination_y = int(plan["destination_y"][m_index])
            destination_x = int(plan["destination_x"][m_index])
            require(0 <= destination_y < 2 * height and
                    0 <= destination_x < 2 * width,
                    "destination OOB")
            require(((destination_y & 1) << 1 | (destination_x & 1)) == bank,
                    "destination bank mismatch")
            require((destination_y, destination_x) not in seen_destinations,
                    "duplicate destination")
            seen_destinations.add((destination_y, destination_x))
            for k_index in range(plan["k"]):
                tap_index = k_index // channels
                channel = k_index % channels
                kernel_y, kernel_x = taps[tap_index]
                numerator_y = destination_y + 1 - kernel_y
                numerator_x = destination_x + 1 - kernel_x
                source_y, source_x = numerator_y // 2, numerator_x // 2
                expected = -1
                if (0 <= source_y < height and 0 <= source_x < width):
                    expected = (channel * height * width +
                                source_y * width + source_x)
                require(int(plan["source_flat_index"][m_index, k_index]) ==
                        expected, "source/K order mismatch")
    return "all phases, destinations, taps and channels match independent inverse"


def test_tile_iteration_bounds():
    shape = (2, 1, 2, 3, 5)
    values = (np.arange(np.prod(shape), dtype=np.uint64) * 13 + 7
              ).reshape(shape).astype(np.uint8) & 1
    with tempfile.TemporaryDirectory() as directory:
        path = write_pack(Path(directory) / "tiles.bitpack", values)
        reference, _ = M665.materialize_polyphase(path, shape, tile_m=99)
        for tile_m in (1, 2, 14, 15, 16, 31):
            by_bank = {}
            for tile in M665.iter_polyphase_tiles(path, shape, tile_m=tile_m):
                bank = tile["phase_bank"]
                by_bank.setdefault(bank, []).append(tile)
                require(0 <= tile["m_start"] < tile["m_stop"] <= 15,
                        "tile M bounds violation")
                require(np.all(tile["destination_y"] < 6) and
                        np.all(tile["destination_x"] < 10),
                        "tile destination bounds violation")
            for bank in (3, 2, 1, 0):
                tiles = by_bank[bank]
                require(tiles[0]["m_start"] == 0 and
                        tiles[-1]["m_stop"] == 15,
                        "tile endpoints incomplete")
                for left, right in zip(tiles, tiles[1:]):
                    require(left["m_stop"] == right["m_start"],
                            "tile gap/overlap")
                joined = np.concatenate([row["values"] for row in tiles],
                                        axis=1)
                require(np.array_equal(joined, reference[bank]),
                        "tile concat mismatch")
    return "tile_m 1/2/M-1/M/M+1/>M cover without gap, overlap or OOB"


def test_boundaries_and_conservation():
    shape = (2, 1, 2, 3, 4)
    values = np.zeros(shape, dtype=np.uint8)
    events = [(0, 0, 0, 0), (0, 0, 0, 3),
              (0, 0, 2, 0), (0, 0, 2, 3),
              (0, 1, 1, 2), (1, 0, 1, 1)]
    for timestep, channel, y, x in events:
        values[timestep, 0, channel, y, x] = 1
    expected_active_taps = 4 + 6 + 6 + 9 + 9 + 9
    with tempfile.TemporaryDirectory() as directory:
        path = write_pack(Path(directory) / "boundary.bitpack", values)
        account = M665.workload_accounting(
            path, shape, output_channels=7, tile_m=2)
        require(account["source_popcount"] == len(events),
                "source popcount mismatch")
        require(account["active_tap_events"] == expected_active_taps,
                "active tap boundary mismatch")
        require(account["active_products"] == expected_active_taps * 7,
                "product conservation mismatch")
        independent_valid_per_time = 0
        for channel in range(shape[2]):
            for y in range(shape[3]):
                for x in range(shape[4]):
                    independent_valid_per_time += sum(
                        0 <= 2 * y - 1 + ky < 2 * shape[3] and
                        0 <= 2 * x - 1 + kx < 2 * shape[4]
                        for ky in range(3) for kx in range(3))
        require(account["valid_tap_slots_per_time"] ==
                independent_valid_per_time, "valid tap conservation mismatch")
        require(account["dense_valid_products"] ==
                independent_valid_per_time * shape[0] * 7,
                "dense product conservation mismatch")
    return account


def test_bit_tail_and_parameter_attacks():
    rejected = []
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        shape = (1, 1, 1, 1, 10)
        pattern = np.asarray([1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                             dtype=np.uint8).reshape(shape)
        good = write_pack(directory / "good", pattern)
        require(good.read_bytes() == bytes([0x81, 0x01]),
                "little-bit fixture drift")
        unpacked = M665._unpack_input(good, shape)
        require(np.array_equal(unpacked, pattern), "little-bit decode mismatch")
        rejected.append(expect_reject(
            lambda: M665.validate_bitpack(good, shape, bit_order="big"),
            "non-little"))
        rejected.append(expect_reject(
            lambda: M665.validate_bitpack(good, shape, flat_order="F_ORDER"),
            "non-C-order"))
        rejected.append(expect_reject(
            lambda: M665.validate_bitpack(good, shape,
                                           k_order="CHANNEL_THEN_TAP"),
            "K-order"))
        bad_tail = directory / "bad_tail"
        bad_tail.write_bytes(bytes([0x81, 0xfd]))
        rejected.append(expect_reject(
            lambda: M665.validate_bitpack(bad_tail, shape), "tail padding"))
        short = directory / "short"
        short.write_bytes(b"\x00")
        rejected.append(expect_reject(
            lambda: M665.validate_bitpack(short, shape), "byte length"))
        rejected.append(expect_reject(
            lambda: M665.validate_input_shape((1, 2, 1, 1, 10)), "batch"))
        rejected.append(expect_reject(
            lambda: M665.validate_input_shape((1, 1, 0, 1, 10)), "positive"))
        rejected.append(expect_reject(
            lambda: M665.validate_input_shape((True, 1, 1, 1, 10)),
            "integers"))
        for field, value in (
                ("kernel_size", (2, 3)), ("stride", (1, 2)),
                ("padding", (0, 1)), ("output_padding", (0, 1)),
                ("dilation", (2, 1)), ("groups", 2)):
            arguments = dict(M665.EXPECTED_SPEC)
            arguments[field] = value
            rejected.append(expect_reject(
                lambda arguments=arguments:
                    M665.validate_convtranspose_spec(**arguments),
                "only exact"))
        weight = np.zeros((1, 2, 3, 3), dtype=np.int8)
        rejected.append(expect_reject(
            lambda: M665.phase_weight_matrix(
                weight, 3, weight_layout="COUT_CIN_KY_KX"),
            "weight-layout"))
        rejected.append(expect_reject(
            lambda: list(M665.iter_polyphase_tiles(
                good, shape, phases=(3, 3))), "unique"))
    return {"rejections": len(rejected)}


def make_manifest(path, rows, d1_rows=None):
    manifest = {
        "schema": "m660_h67_ep35_layer_static_decoder_payload_v1",
        "packing": {"values": [0, 1], "bit_order": "little",
                    "order": "C_ORDER_FLAT",
                    "whole_call_contiguous_copy_allowed": False},
        "d0_d2_d3_binary_records": rows,
        "d1_records": [] if d1_rows is None else d1_rows,
    }
    Path(path).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def standard_row(relative, payload_sha, route="EXACT_BINARY_BITPACK",
                 sample_id=0, module_index=0, shape=None):
    if shape is None:
        shape = [1, 1, 1, 1, 8]
    row = {"sample_id": sample_id, "module_index": module_index,
           "route": route, "relative_path": relative,
           "input_shape": shape,
           "input": {"packed_bytes": 1, "packed_sha256": payload_sha}}
    if route == "EXACT_SCALED_BINARY_BITPACK":
        row["theta_binary_candidate"] = {
            "packed_bytes": 1, "packed_sha256": payload_sha}
    return row


def test_manifest_positive_and_rejections():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        payload = write_pack(directory / "x", np.asarray(
            [1, 0, 1, 0, 0, 1, 0, 0], dtype=np.uint8))
        payload_sha = digest(payload)
        row = standard_row("x", payload_sha)
        d1 = standard_row("x", payload_sha,
                          route="EXACT_SCALED_BINARY_BITPACK",
                          sample_id=1, module_index=1)
        fallback = {"sample_id": 2, "module_index": 1,
                    "route": "COMMON_FP32_DENSE_FALLBACK"}
        manifest_path = directory / "manifest.json"
        manifest = make_manifest(manifest_path, [row], [d1, fallback])
        result = M665.m660_bitpack_records(manifest_path)
        require([(item["sample_id"], item["module_index"], item["route"])
                 for item in result] == [
                    (0, 0, "EXACT_BINARY_BITPACK"),
                    (1, 1, "EXACT_SCALED_BINARY_BITPACK")],
                "route-specific positive adapter mismatch")

        duplicate = dict(manifest)
        duplicate["d0_d2_d3_binary_records"] = [row, dict(row)]
        manifest_path.write_text(json.dumps(duplicate), encoding="utf-8")
        expect_reject(lambda: M665.m660_bitpack_records(manifest_path),
                      "duplicate")
        bad_schema = dict(manifest)
        bad_schema["schema"] = "wrong"
        manifest_path.write_text(json.dumps(bad_schema), encoding="utf-8")
        expect_reject(lambda: M665.m660_bitpack_records(manifest_path),
                      "schema")
        bad_pack = dict(manifest)
        bad_pack["packing"] = dict(manifest["packing"])
        bad_pack["packing"]["bit_order"] = "big"
        manifest_path.write_text(json.dumps(bad_pack), encoding="utf-8")
        expect_reject(lambda: M665.m660_bitpack_records(manifest_path),
                      "packing")
        bad_route = dict(manifest)
        bad_route["d0_d2_d3_binary_records"] = [dict(row)]
        bad_route["d0_d2_d3_binary_records"][0]["route"] = "OTHER"
        manifest_path.write_text(json.dumps(bad_route), encoding="utf-8")
        expect_reject(lambda: M665.m660_bitpack_records(manifest_path),
                      "non-bitpack")
        traversal = dict(manifest)
        traversal["d0_d2_d3_binary_records"] = [dict(row)]
        traversal["d0_d2_d3_binary_records"][0]["relative_path"] = "../x"
        manifest_path.write_text(json.dumps(traversal), encoding="utf-8")
        expect_reject(lambda: M665.m660_bitpack_records(manifest_path),
                      "unsafe")
        absolute = dict(manifest)
        absolute["d0_d2_d3_binary_records"] = [dict(row)]
        absolute["d0_d2_d3_binary_records"][0]["relative_path"] = str(payload)
        manifest_path.write_text(json.dumps(absolute), encoding="utf-8")
        expect_reject(lambda: M665.m660_bitpack_records(manifest_path),
                      "unsafe")
        duplicate_json = directory / "duplicate.json"
        duplicate_json.write_text(
            '{"schema":"x","schema":"y"}', encoding="utf-8")
        expect_reject(lambda: M665.strict_json(duplicate_json), "duplicate")
    return "positive route fields and schema/packing/route/traversal/duplicate attacks"


def find_shape_overflow_acceptance():
    with tempfile.TemporaryDirectory() as directory:
        zero = Path(directory) / "zero"
        zero.write_bytes(b"")
        shape = (2 ** 32, 1, 2 ** 32, 1, 1)
        identity = M665.validate_bitpack(zero, shape)
        require(identity["elements"] == 0 and identity["packed_bytes"] == 0,
                "overflow attack did not wrap to zero")
    findings.append({
        "id": "P1_SHAPE_PRODUCT_INT64_OVERFLOW_ACCEPTED",
        "severity": "P1",
        "evidence": "shape [2^32,1,2^32,1,1] wraps np.int64 product to zero and validates an empty file",
    })
    return identity


def find_parent_symlink_escape():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        package = root / "package"
        outside = root / "outside"
        package.mkdir()
        outside.mkdir()
        payload = write_pack(outside / "x", np.asarray(
            [1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8))
        (package / "alias").symlink_to(outside, target_is_directory=True)
        row = standard_row("alias/x", digest(payload))
        manifest = package / "manifest.json"
        make_manifest(manifest, [row])
        result = M665.m660_bitpack_records(manifest)
        require(len(result) == 1 and Path(result[0]["path"]).read_bytes() ==
                payload.read_bytes(), "parent symlink escape did not reproduce")
        leaf = package / "leaf"
        leaf.symlink_to(payload)
        expect_reject(lambda: M665.validate_bitpack(
            leaf, (1, 1, 1, 1, 8)), "non-symlink")
    findings.append({
        "id": "P1_PARENT_DIRECTORY_SYMLINK_ESCAPE_ACCEPTED",
        "severity": "P1",
        "evidence": "lexically safe alias/x escapes the manifest package through symlink parent; only leaf symlinks are rejected",
    })
    return "parent escape accepted while leaf alias rejected"


def find_route_module_misbinding():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        payload = write_pack(directory / "x", np.asarray(
            [1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8))
        # An EXACT_SCALED row is incorrectly placed in the D0/D2/D3 list and
        # claims module 3.  The adapter accepts it and selects D1's theta field.
        wrong = standard_row(
            "x", digest(payload), route="EXACT_SCALED_BINARY_BITPACK",
            sample_id=0, module_index=3)
        manifest = directory / "manifest.json"
        make_manifest(manifest, [wrong])
        result = M665.m660_bitpack_records(manifest)
        require(result[0]["module_index"] == 3 and
                result[0]["route"] == "EXACT_SCALED_BINARY_BITPACK",
                "route/module misbinding did not reproduce")
    findings.append({
        "id": "P1_ROUTE_CONTAINER_MODULE_SEMANTICS_NOT_BOUND",
        "severity": "P1",
        "evidence": "d0_d2_d3 list accepts EXACT_SCALED_BINARY_BITPACK/module3 and consumes theta_binary_candidate; module/list/route relation is unchecked",
    })
    return result[0]


def find_loose_boolean_parameters():
    with tempfile.TemporaryDirectory() as directory:
        path = write_pack(Path(directory) / "x", np.zeros(
            (1, 1, 1, 1, 8), dtype=np.uint8))
        tiles = list(M665.iter_polyphase_tiles(
            path, (1, 1, 1, 1, 8), tile_m=True, phases=(True,)))
        account = M665.workload_accounting(
            path, (1, 1, 1, 1, 8), output_channels=True)
        require(tiles and account["output_channels"] is True,
                "boolean type laxity did not reproduce")
    findings.append({
        "id": "P2_BOOLEAN_TILE_PHASE_OUTPUT_CHANNEL_ACCEPTED",
        "severity": "P2",
        "evidence": "tile_m=True, phases=(True,) and output_channels=True pass integer gates",
    })
    return "boolean integer aliases accepted"


check("RTL_AND_VCS_INDEPENDENT_PHASE_TAP_DERIVATION", test_rtl_derived_phase_taps)
check("FRESH_NUMPY_AND_PYTORCH_CONVTRANSPOSE_ORACLES", test_fresh_oracles)
check("EVERY_PHASE_DESTINATION_TAP_CHANNEL_K_SLOT", test_plan_every_coordinate_and_k_slot)
check("TILE_ITERATION_GAP_OVERLAP_OOB", test_tile_iteration_bounds)
check("BOUNDARY_POPCOUNT_TAP_PRODUCT_CONSERVATION", test_boundaries_and_conservation)
check("BIT_TAIL_SHAPE_SPEC_PARAMETER_ATTACKS", test_bit_tail_and_parameter_attacks)
check("MANIFEST_ROUTE_AND_STRUCTURAL_REJECTIONS", test_manifest_positive_and_rejections)
check("ATTACK_SHAPE_PRODUCT_OVERFLOW", find_shape_overflow_acceptance)
check("ATTACK_PARENT_SYMLINK_ESCAPE", find_parent_symlink_escape)
check("ATTACK_ROUTE_CONTAINER_MODULE_MISBINDING", find_route_module_misbinding)
check("ATTACK_BOOLEAN_INTEGER_ALIASES", find_loose_boolean_parameters)

result = {
    "schema": "m667_m665_independent_hammer_validation_v1",
    "python": sys.executable,
    "python_version": sys.version,
    "numpy_version": np.__version__,
    "torch_available": torch is not None,
    "torch_version": None if torch is None else torch.__version__,
    "mapper_sha256": digest(MAPPER),
    "checks": checks,
    "findings": findings,
    "summary": {"checks": len(checks),
                "checks_passed": sum(1 for item in checks if item["pass"]),
                "checks_failed": sum(1 for item in checks if not item["pass"]),
                "p0": sum(1 for item in findings if item["severity"] == "P0"),
                "p1": sum(1 for item in findings if item["severity"] == "P1"),
                "p2": sum(1 for item in findings if item["severity"] == "P2")},
    "claim_boundary": {"gpu_run": False, "eda_run": False,
                       "production_m660_mapping": False,
                       "cycles": False, "speedup": False},
}
print(json.dumps(result, indent=2, sort_keys=True))
if result["summary"]["checks_failed"]:
    raise SystemExit(2)
