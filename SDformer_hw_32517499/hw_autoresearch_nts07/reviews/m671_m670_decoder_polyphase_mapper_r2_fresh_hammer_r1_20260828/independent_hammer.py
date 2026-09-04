#!/usr/bin/env python3
"""Fresh CPU-only hammer for frozen M670 decoder polyphase mapper r2."""

from __future__ import print_function

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as torch_f
except ImportError:
    torch = None
    torch_f = None


ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / ("hw_autoresearch_nts07/system_simulator/scripts/"
                 "map_m670_decoder_convtranspose_polyphase_workload_r2.py")
spec = importlib.util.spec_from_file_location("m671_frozen_target", str(TARGET))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

RTL_SLOT_ORDER = ((0, 0), (0, 2), (2, 0), (2, 2),
                  (0, 1), (2, 1), (1, 0), (1, 2), (1, 1))
RTL_PHASE_ORDER = (3, 2, 1, 0)
RTL_PHASE_TAPS = {
    bank: tuple(tap for tap in RTL_SLOT_ORDER
                if (((tap[0] + 1) & 1) << 1 | ((tap[1] + 1) & 1)) == bank)
    for bank in RTL_PHASE_ORDER
}


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


def direct_oracle(activation, weight):
    """Independent source-event ConvTranspose oracle from M514 coordinates."""
    t_count, _batch, channels, height, width = activation.shape
    output = np.zeros((t_count, weight.shape[1], 2 * height, 2 * width),
                      dtype=np.int64)
    for t in range(t_count):
        for channel in range(channels):
            for sy in range(height):
                for sx in range(width):
                    value = int(activation[t, 0, channel, sy, sx])
                    for ky, kx in RTL_SLOT_ORDER:
                        dy, dx = 2 * sy - 1 + ky, 2 * sx - 1 + kx
                        if 0 <= dy < 2 * height and 0 <= dx < 2 * width:
                            output[t, :, dy, dx] += (
                                value * weight[channel, :, ky, kx])
    return output


def make_manifest(package, d1_scaled=True):
    shape = [1, 1, 1, 2, 4]
    binary = write_bits(package / "binary.bitpack", [0, 1, 0, 1, 1, 0, 0, 0])
    scaled = write_bits(package / "scaled.bitpack", [1, 0, 1, 0, 0, 1, 0, 0])
    binary_rows, d1_rows = [], []
    for sample_id in range(10):
        for module_index in (0, 2, 3):
            binary_rows.append({
                "sample_id": sample_id, "module_index": module_index,
                "name": M.M660_MODULE_NAMES[module_index],
                "route": "EXACT_BINARY_BITPACK",
                "relative_path": "binary.bitpack", "input_shape": shape,
                "input": {"packed_bytes": 1,
                          "packed_sha256": digest(binary)},
                "theta_binary_candidate": {"packed_bytes": 1,
                          "packed_sha256": "0" * 64},
            })
        row = {
            "sample_id": sample_id, "module_index": 1,
            "name": M.M660_MODULE_NAMES[1],
            "route": ("EXACT_SCALED_BINARY_BITPACK" if d1_scaled else
                      "COMMON_FP32_DENSE_FALLBACK"),
            "input_shape": shape,
            "input": {"packed_bytes": 1, "packed_sha256": "f" * 64},
        }
        if d1_scaled:
            row.update({"relative_path": "scaled.bitpack",
                        "theta_binary_candidate": {
                            "packed_bytes": 1, "packed_sha256": digest(scaled)}})
        d1_rows.append(row)
    return {
        "schema": "m660_h67_ep35_layer_static_decoder_payload_v1",
        "packing": {"values": [0, 1], "bit_order": "little",
                    "order": "C_ORDER_FLAT",
                    "whole_call_contiguous_copy_allowed": False},
        "d0_d2_d3_binary_records": binary_rows,
        "d1_records": d1_rows,
    }


def write_manifest(path, manifest):
    Path(path).write_text(json.dumps(manifest), encoding="utf-8")
    return Path(path)


checks = []
details = {}

# Rebuild every phase/tap/channel slot from the M514 coordinate rule.
require(M.M514_PHASE_ORDER == RTL_PHASE_ORDER, "phase order drift")
require(M.M514_SLOT_ORDER == RTL_SLOT_ORDER, "slot order drift")
require(M.M514_PHASE_TAPS == RTL_PHASE_TAPS, "phase tap derivation drift")
for channels, height, width in ((1, 1, 5), (2, 3, 4), (3, 4, 2)):
    for bank in RTL_PHASE_ORDER:
        plan = M.build_phase_plan(channels, height, width, bank)
        taps = RTL_PHASE_TAPS[bank]
        for row in range(height * width):
            dy, dx = int(plan["destination_y"][row]), int(plan["destination_x"][row])
            require(((dy & 1) << 1 | (dx & 1)) == bank, "bank drift")
            for tap_index, (ky, kx) in enumerate(taps):
                ny, nx = dy + 1 - ky, dx + 1 - kx
                sy, sx = ny // 2, nx // 2
                for channel in range(channels):
                    observed = int(plan["source_flat_index"][
                        row, tap_index * channels + channel])
                    expected = (-1 if not (0 <= sy < height and 0 <= sx < width)
                                else channel * height * width + sy * width + sx)
                    require(observed == expected, "phase/tap/K slot mismatch")
checks.append("rtl_rederived_all_phase_tap_channel_slots")

# Fresh non-square NumPy and optional PyTorch reconstructions, including tiles.
numeric = []
with tempfile.TemporaryDirectory(prefix="m671_numeric_") as temporary:
    package = Path(temporary)
    for seed, shape, cout in ((671011, (2, 1, 2, 2, 5), 3),
                              (671037, (1, 1, 3, 4, 2), 4),
                              (671099, (3, 1, 1, 3, 4), 2)):
        rng = np.random.default_rng(seed)
        activation = rng.integers(0, 2, size=shape, dtype=np.uint8)
        weight = rng.integers(-5, 6, size=(shape[2], cout, 3, 3), dtype=np.int8)
        payload = write_bits(package / ("x{}.bitpack".format(seed)), activation)
        observed = M.reconstruct_convtranspose(
            payload, shape, weight, tile_m=3, trusted_root=package)
        expected = direct_oracle(activation, weight)
        require(np.array_equal(observed, expected), "NumPy oracle mismatch")
        torch_equal = None
        if torch is not None:
            torch_expected = torch_f.conv_transpose2d(
                torch.from_numpy(activation[:, 0].astype(np.float64)),
                torch.from_numpy(weight.astype(np.float64)), bias=None,
                stride=(2, 2), padding=(1, 1), output_padding=(1, 1),
                dilation=(1, 1), groups=1).numpy()
            torch_equal = bool(np.array_equal(observed, torch_expected))
            require(torch_equal, "PyTorch CPU mismatch")
        full, _ = M.materialize_polyphase(
            payload, shape, tile_m=999, trusted_root=package)
        tiled, _ = M.materialize_polyphase(
            payload, shape, tile_m=3, trusted_root=package)
        require(all(np.array_equal(full[b], tiled[b]) for b in RTL_PHASE_ORDER),
                "tile boundary mismatch")
        account = M.workload_accounting(
            payload, shape, output_channels=cout, tile_m=2,
            trusted_root=package)
        require(account["active_products"] ==
                account["active_tap_events"] * cout, "product conservation")
        numeric.append({"seed": seed, "shape": list(shape),
                        "torch_equal": torch_equal,
                        "active_taps": account["active_tap_events"]})
details["numeric"] = numeric
checks.append("fresh_numpy_torch_tile_boundary_popcount_product")

# Python-integer bounds, zero/negative and boolean aliases.
with tempfile.TemporaryDirectory(prefix="m671_integer_") as temporary:
    root = Path(temporary)
    empty = root / "empty.bitpack"
    empty.write_bytes(b"")
    bad_shapes = [(2 ** 32, 1, 2 ** 32, 1, 1),
                  (2 ** 20, 1, 2 ** 15, 1, 1),
                  (0, 1, 1, 1, 1), (-1, 1, 1, 1, 1)]
    for position in range(5):
        for boolean in (True, np.bool_(True)):
            row = [1, 1, 1, 1, 1]
            row[position] = boolean
            bad_shapes.append(tuple(row))
    for shape in bad_shapes:
        expect_reject(lambda value=shape: M.validate_bitpack(
            empty, value, trusted_root=root), "unsafe shape")
    source = TARGET.read_text(encoding="utf-8")
    require("np.prod" not in source, "np.prod returned")
    real = write_bits(root / "real.bitpack", np.zeros((1, 1, 1, 1, 8), np.uint8))
    for boolean in (True, np.bool_(True)):
        for pair_name in ("kernel_size", "stride", "padding",
                          "output_padding", "dilation"):
            for position in (0, 1):
                pair = [3, 3] if pair_name == "kernel_size" else [1, 1]
                pair[position] = boolean
                kwargs = dict(M.EXPECTED_SPEC)
                kwargs[pair_name] = pair
                expect_reject(lambda value=kwargs: M.validate_convtranspose_spec(
                    **value), "bool pair")
        for attack in (
                lambda b=boolean: M.phase_bank(b, 0),
                lambda b=boolean: M.phase_bank(0, b),
                lambda b=boolean: M.build_phase_plan(b, 1, 1, 3),
                lambda b=boolean: M.build_phase_plan(1, b, 1, 3),
                lambda b=boolean: M.build_phase_plan(1, 1, b, 3),
                lambda b=boolean: M.build_phase_plan(1, 1, 1, b),
                lambda b=boolean: list(M.iter_polyphase_tiles(
                    real, (1, 1, 1, 1, 8), phases=(b,), trusted_root=root)),
                lambda b=boolean: list(M.iter_polyphase_tiles(
                    real, (1, 1, 1, 1, 8), tile_m=b, trusted_root=root)),
                lambda b=boolean: M.workload_accounting(
                    real, (1, 1, 1, 1, 8), output_channels=b,
                    trusted_root=root),
                lambda b=boolean: M.validate_convtranspose_spec(groups=b)):
            expect_reject(attack, "boolean integer alias")
checks.append("bounded_python_products_and_all_boolean_integer_gates")

# Trusted-root, leaf/parent symlink, traversal, escape and leaf-type attacks.
path_findings = {}
with tempfile.TemporaryDirectory(prefix="m671_path_") as temporary:
    base = Path(temporary)
    package = base / "package"
    package.mkdir()
    nested = package / "nested"
    nested.mkdir()
    outside = base / "outside"
    outside.mkdir()
    outside_payload = write_bits(outside / "x.bitpack", [0] * 8)
    good = write_bits(package / "good.bitpack", [0] * 8)
    for root in (None, Path("relative-root")):
        expect_reject(lambda value=root: M.validate_bitpack(
            good, (1, 1, 1, 1, 8), trusted_root=value), "unsafe root")
    root_link = base / "root_link"
    root_link.symlink_to(package, target_is_directory=True)
    expect_reject(lambda: M.validate_bitpack(
        root_link / "good.bitpack", (1, 1, 1, 1, 8),
        trusted_root=root_link), "root leaf symlink")
    leaf_link = package / "leaf.bitpack"
    leaf_link.symlink_to(outside_payload)
    parent_link = package / "parent_link"
    parent_link.symlink_to(outside, target_is_directory=True)
    deep = package / "deep"
    deep.mkdir()
    deep_link = deep / "parent_link"
    deep_link.symlink_to(outside, target_is_directory=True)
    for candidate in (leaf_link, parent_link / "x.bitpack",
                      deep_link / "x.bitpack", "../outside/x.bitpack",
                      "nested/../../outside/x.bitpack", outside_payload,
                      package, package / "missing.bitpack"):
        expect_reject(lambda value=candidate: M.validate_bitpack(
            value, (1, 1, 1, 1, 8), trusted_root=package),
            "unsafe candidate")
    manifest = make_manifest(package)
    direct_manifest = write_manifest(package / "manifest.json", manifest)
    shutil.copyfile(str(direct_manifest), str(nested / "manifest.json"))
    expect_reject(lambda: M.m660_bitpack_records(
        nested / "manifest.json", trusted_root=package), "non-direct manifest")

    # Remaining P1-A: a symlink in an ancestor of trusted_root is not checked.
    real_parent = base / "real_parent"
    real_parent.mkdir()
    root_under_real = real_parent / "pkg"
    root_under_real.mkdir()
    alias_parent = base / "alias_parent"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    alias_file = write_bits(root_under_real / "x.bitpack", [0] * 8)
    accepted = M.validate_bitpack(
        alias_parent / "pkg" / "x.bitpack", (1, 1, 1, 1, 8),
        trusted_root=alias_parent / "pkg")
    path_findings["trusted_root_ancestor_symlink_accepted"] = accepted["path"]

    # Remaining P1-B: validation uses package/x, consumption reopens CWD/x.
    cwd = base / "cwd"
    cwd.mkdir()
    write_bits(package / "same.bitpack", [0] * 8)
    write_bits(cwd / "same.bitpack", [1] * 8)
    old_cwd = os.getcwd()
    os.chdir(str(cwd))
    try:
        mapped, _ = M.materialize_polyphase(
            "same.bitpack", (1, 1, 1, 1, 8), trusted_root=package)
        mapped_sum = sum(int(value.sum()) for value in mapped.values())
        account = M.workload_accounting(
            "same.bitpack", (1, 1, 1, 1, 8), output_channels=1,
            trusted_root=package)
    finally:
        os.chdir(old_cwd)
    require(mapped_sum > 0 and account["source_popcount"] == 8,
            "relative reopen attack unexpectedly failed")
    path_findings["validated_zero_consumed_one_mapped_sum"] = mapped_sum
    path_findings["validated_zero_consumed_one_source_popcount"] = account[
        "source_popcount"]

    # Same identity failure under a post-validation same-size replacement.
    race = write_bits(package / "race.bitpack", [0] * 8)
    replacement = write_bits(package / "replacement.tmp", [1] * 8)
    original_validate = M.validate_bitpack
    observed_identity = {}
    def swap_after_validate(*args, **kwargs):
        identity = original_validate(*args, **kwargs)
        observed_identity.update(identity)
        os.replace(str(replacement), str(race))
        return identity
    M.validate_bitpack = swap_after_validate
    try:
        raced, _ = M.materialize_polyphase(
            race, (1, 1, 1, 1, 8), trusted_root=package)
    finally:
        M.validate_bitpack = original_validate
    race_sum = sum(int(value.sum()) for value in raced.values())
    require(observed_identity["sha256"] != digest(race) and race_sum > 0,
            "post-validation replacement attack unexpectedly failed")
    path_findings["post_validation_hash_changed"] = True
details["path_attacks"] = path_findings
checks.append("path_guards_and_two_remaining_path_identity_attacks")

# Complete S10x4 route/container/module/name/identity lattice attacks.
with tempfile.TemporaryDirectory(prefix="m671_manifest_") as temporary:
    package = Path(temporary)
    baseline = make_manifest(package, d1_scaled=True)
    manifest_path = package / "manifest.json"
    write_manifest(manifest_path, baseline)
    records = M.m660_bitpack_records(manifest_path, trusted_root=package)
    require(len(records) == 40, "valid S10x4 rejected")
    rejected = 0
    mutations = []
    for module_index in (0, 2, 3):
        index = next(i for i, row in enumerate(
            baseline["d0_d2_d3_binary_records"])
            if row["module_index"] == module_index)
        for route in ("EXACT_SCALED_BINARY_BITPACK",
                      "COMMON_FP32_DENSE_FALLBACK"):
            mutations.append(("binary_route_{}_{}".format(module_index, route),
                              lambda x, i=index, r=route: x[
                                  "d0_d2_d3_binary_records"][i].update(route=r)))
        mutations.append(("binary_wrong_name_{}".format(module_index),
                          lambda x, i=index: x[
                              "d0_d2_d3_binary_records"][i].update(name="wrong")))
    for wrong_module in (0, 2, 3):
        mutations.append(("d1_wrong_module_{}".format(wrong_module),
                          lambda x, m=wrong_module: x["d1_records"][0].update(
                              module_index=m, name=M.M660_MODULE_NAMES[m])))
    mutations.extend([
        ("d1_binary_route", lambda x: x["d1_records"][0].update(
            route="EXACT_BINARY_BITPACK")),
        ("d1_wrong_name", lambda x: x["d1_records"][0].update(name="wrong")),
        ("binary_wrong_input_identity", lambda x: x[
            "d0_d2_d3_binary_records"][0]["input"].update(packed_sha256="f" * 64)),
        ("d1_wrong_theta_identity", lambda x: x["d1_records"][0][
            "theta_binary_candidate"].update(packed_sha256="f" * 64)),
        ("missing_cell", lambda x: x["d0_d2_d3_binary_records"].pop()),
        ("duplicate_cell", lambda x: x["d0_d2_d3_binary_records"].__setitem__(
            -1, copy.deepcopy(x["d0_d2_d3_binary_records"][0]))),
        ("out_of_s10", lambda x: x["d0_d2_d3_binary_records"][0].update(
            sample_id=10)),
        ("malformed_hash", lambda x: x["d0_d2_d3_binary_records"][0][
            "input"].update(packed_sha256="A" * 64)),
        ("malformed_bytes", lambda x: x["d0_d2_d3_binary_records"][0][
            "input"].update(packed_bytes=2)),
        ("missing_container", lambda x: x.pop("d1_records")),
        ("container_not_list", lambda x: x.update(d1_records={})),
    ])
    # JSON can encode only the built-in boolean.  The NumPy boolean aliases
    # are attacked directly at the same internal integer gates below.
    for boolean in (True,):
        mutations.extend([
            ("bool_sample_{}".format(type(boolean).__name__),
             lambda x, b=boolean: x["d0_d2_d3_binary_records"][0].update(
                 sample_id=b)),
            ("bool_module_{}".format(type(boolean).__name__),
             lambda x, b=boolean: x["d0_d2_d3_binary_records"][0].update(
                 module_index=b)),
            ("bool_bytes_{}".format(type(boolean).__name__),
             lambda x, b=boolean: x["d0_d2_d3_binary_records"][0][
                 "input"].update(packed_bytes=b)),
        ])
    for label, mutate in mutations:
        attacked = copy.deepcopy(baseline)
        mutate(attacked)
        write_manifest(manifest_path, attacked)
        expect_reject(lambda: M.m660_bitpack_records(
            manifest_path, trusted_root=package), label)
        rejected += 1
    expect_reject(lambda: M._record_integer(
        {"sample_id": np.bool_(True)}, "sample_id"), "np.bool sample")
    expect_reject(lambda: M._record_integer(
        {"module_index": np.bool_(True)}, "module_index"), "np.bool module")
    expect_reject(lambda: M._packed_identity(
        {"input": {"packed_bytes": np.bool_(True),
                   "packed_sha256": "0" * 64}}, "input"),
        "np.bool packed bytes")
    # Prove route-specific fields, not the opposite field, are authoritative.
    opposite = copy.deepcopy(baseline)
    for row in opposite["d0_d2_d3_binary_records"]:
        row["theta_binary_candidate"] = {"packed_bytes": True,
                                          "packed_sha256": "bad"}
    for row in opposite["d1_records"]:
        row["input"] = {"packed_bytes": True, "packed_sha256": "bad"}
    write_manifest(manifest_path, opposite)
    require(len(M.m660_bitpack_records(manifest_path,
                                       trusted_root=package)) == 40,
            "route-specific identity selection drift")
    manifest_path.write_text('{"schema":"a","schema":"b"}',
                             encoding="utf-8")
    expect_reject(lambda: M.m660_bitpack_records(
        manifest_path, trusted_root=package), "duplicate JSON key")
details["manifest_mutations_rejected"] = rejected + 1
checks.append("complete_s10x4_route_container_name_identity_lattice")

print(json.dumps({
    "schema": "m671_m670_decoder_polyphase_mapper_r2_independent_hammer_v1",
    "status": "PASS_HAMMER_WITH_TWO_P1_FINDINGS",
    "python": sys.executable,
    "python_version": sys.version,
    "numpy_version": np.__version__,
    "torch_available": torch is not None,
    "torch_version": None if torch is None else torch.__version__,
    "checks": checks,
    "details": details,
    "p1_findings": [
        "P1_VALIDATED_PATH_NOT_USED_FOR_CONSUMPTION_AND_TOCTOU_IDENTITY_SPLIT",
        "P1_TRUSTED_ROOT_ANCESTOR_SYMLINK_NOT_REJECTED",
    ],
    "claim_boundary": {"gpu": False, "eda": False,
                       "production_mapping": False, "performance": False},
}, indent=2, sort_keys=True))
