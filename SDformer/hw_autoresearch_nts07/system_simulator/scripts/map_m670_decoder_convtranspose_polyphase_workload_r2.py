#!/usr/bin/env python3
"""Exact CPU workload mapper r2 for the M514 K3/S2/P1/OP1 decoder path.

The mapper consumes a little-bit-first, C-order bitpack whose logical shape is
``[T, 1, C, H, W]``.  It never inserts a zero-expanded image.  Instead, it
builds four destination-parity matrices with logical shape ``[T, M, K]``:

* M is the row-major list of output sites in one parity bank (H * W sites),
* K is ordered by the M514 phase-major tap order, then source channel, and
* out-of-range source coordinates are represented by exact structural zeros.

This file is an input mapper, not a cycle simulator.  It reports no speedup and
does not bind an unfinished M660 payload identity.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath

import numpy as np


M514_SLOT_ORDER = (
    (0, 0), (0, 2), (2, 0), (2, 2),
    (0, 1), (2, 1),
    (1, 0), (1, 2),
    (1, 1),
)

# M514 emits destination banks in this phase-major order.  A bank value is
# (destination_y_lsb << 1) | destination_x_lsb.
M514_PHASE_ORDER = (3, 2, 1, 0)
M514_PHASE_TAPS = {
    3: ((0, 0), (0, 2), (2, 0), (2, 2)),
    2: ((0, 1), (2, 1)),
    1: ((1, 0), (1, 2)),
    0: ((1, 1),),
}

EXPECTED_BIT_ORDER = "little"
EXPECTED_FLAT_ORDER = "C_ORDER_FLAT"
EXPECTED_K_ORDER = "M514_PHASE_TAP_THEN_SOURCE_CHANNEL"
EXPECTED_SPEC = {
    "kernel_size": (3, 3),
    "stride": (2, 2),
    "padding": (1, 1),
    "output_padding": (1, 1),
    "dilation": (1, 1),
    "groups": 1,
}

# These bounds are deliberately much larger than every frozen H67 decoder
# activation (the largest spatial dimension is 80 and the largest channel
# count is 1536), while keeping all downstream NumPy indexing and allocation
# arithmetic inside an explicitly supported domain.
MAX_SUPPORTED_DIMENSION = 1 << 20
MAX_SUPPORTED_INPUT_ELEMENTS = 1 << 34
MAX_SUPPORTED_PLAN_SITES = 1 << 26
MAX_SUPPORTED_K = 1 << 24

M660_MODULE_NAMES = {
    0: "sttmultires_unet.decoders.0.deconv.0",
    1: "sttmultires_unet.decoders.1.deconv.0",
    2: "sttmultires_unet.decoders.2.deconv.0",
    3: "sttmultires_unet.decoders.3.deconv.0",
}
M660_SAMPLE_IDS = tuple(range(10))
M660_BINARY_MODULES = (0, 2, 3)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def strict_integer(value, name, minimum=None, maximum=None):
    require(isinstance(value, (int, np.integer)) and
            not isinstance(value, (bool, np.bool_)),
            name + " must be a non-boolean integer")
    result = int(value)
    if minimum is not None:
        require(result >= minimum, name + " is below supported minimum")
    if maximum is not None:
        require(result <= maximum, name + " exceeds supported maximum")
    return result


def checked_positive_product(values, limit, name):
    """Multiply with Python integers and fail before crossing ``limit``."""
    product = 1
    for index, value in enumerate(values):
        item = strict_integer(value, "{}[{}]".format(name, index), 1,
                              MAX_SUPPORTED_DIMENSION)
        require(product <= limit // item,
                name + " product exceeds supported maximum")
        product *= item
    require(product > 0, name + " product must be positive")
    return product


def _trusted_root(path):
    root = Path(path)
    require(root.is_absolute() and ".." not in root.parts,
            "trusted root must be an absolute lexical path")
    try:
        root_stat = os.lstat(str(root))
    except OSError as error:
        raise RuntimeError("trusted root lstat failed: " + str(error))
    require(stat.S_ISDIR(root_stat.st_mode) and
            not stat.S_ISLNK(root_stat.st_mode),
            "trusted root must be a real non-symlink directory")
    return root


def trusted_regular_file(trusted_root, path, label):
    """Bind a regular file below a trusted root without following symlinks."""
    require(trusted_root is not None,
            label + " requires an explicit trusted root")
    root = _trusted_root(trusted_root)
    require(isinstance(path, (str, os.PathLike, PurePosixPath)),
            label + " path must be string-like")
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            raise RuntimeError(label + " escapes trusted root")
    else:
        relative = candidate
    member = safe_member(PurePosixPath(relative.as_posix()))
    current = root
    for index, part in enumerate(member.parts):
        current = current / part
        try:
            observed = os.lstat(str(current))
        except OSError as error:
            raise RuntimeError(label + " component lstat failed: " +
                               str(error))
        require(not stat.S_ISLNK(observed.st_mode),
                label + " path contains a symlink component")
        if index + 1 == len(member.parts):
            require(stat.S_ISREG(observed.st_mode),
                    label + " leaf must be a regular file")
        else:
            require(stat.S_ISDIR(observed.st_mode),
                    label + " parent component must be a directory")
    root_resolved = root.resolve(strict=True)
    candidate_resolved = current.resolve(strict=True)
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError:
        raise RuntimeError(label + " resolves outside trusted root")
    return current


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(token):
        raise RuntimeError("non-standard JSON token: " + token)

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def safe_member(name):
    require(isinstance(name, (str, os.PathLike, PurePosixPath)),
            "payload member must be string-like")
    member = PurePosixPath(name)
    require(not member.is_absolute() and member.parts and
            ".." not in member.parts and member.parts[0] not in ("", "."),
            "unsafe payload member: " + str(name))
    return member


def _pair(value, name):
    require(isinstance(value, (tuple, list)) and len(value) == 2 and
            all(isinstance(item, (int, np.integer)) and
                not isinstance(item, (bool, np.bool_)) for item in value),
            name + " must be an integer pair")
    return tuple(int(item) for item in value)


def validate_convtranspose_spec(kernel_size=(3, 3), stride=(2, 2),
                                padding=(1, 1), output_padding=(1, 1),
                                dilation=(1, 1), groups=1):
    observed = {
        "kernel_size": _pair(kernel_size, "kernel_size"),
        "stride": _pair(stride, "stride"),
        "padding": _pair(padding, "padding"),
        "output_padding": _pair(output_padding, "output_padding"),
        "dilation": _pair(dilation, "dilation"),
        "groups": groups,
    }
    require(isinstance(groups, (int, np.integer)) and
            not isinstance(groups, (bool, np.bool_)),
            "groups must be an integer")
    observed["groups"] = int(groups)
    require(observed == EXPECTED_SPEC,
            "only exact K3/S2/P1/OP1/dilation1/groups1 is admitted")
    return observed


def validate_input_shape(shape):
    require(isinstance(shape, (tuple, list)) and len(shape) == 5,
            "input shape must be [T,1,C,H,W]")
    result = tuple(strict_integer(
        item, "input_shape[{}]".format(index), 1,
        MAX_SUPPORTED_DIMENSION) for index, item in enumerate(shape))
    require(result[1] == 1, "M665 admits batch dimension exactly one")
    checked_positive_product(result, MAX_SUPPORTED_INPUT_ELEMENTS,
                             "input_shape")
    checked_positive_product(result[3:5], MAX_SUPPORTED_PLAN_SITES,
                             "spatial_shape")
    require(result[2] <= MAX_SUPPORTED_K,
            "channel count exceeds supported K domain")
    return result


def expected_packed_bytes(shape):
    shape = validate_input_shape(shape)
    elements = checked_positive_product(
        shape, MAX_SUPPORTED_INPUT_ELEMENTS, "input_shape")
    return (elements + 7) // 8


def validate_bitpack(path, shape, bit_order=EXPECTED_BIT_ORDER,
                     flat_order=EXPECTED_FLAT_ORDER,
                     k_order=EXPECTED_K_ORDER, trusted_root=None):
    shape = validate_input_shape(shape)
    require(bit_order == EXPECTED_BIT_ORDER,
            "M665 rejects non-little bit order")
    require(flat_order == EXPECTED_FLAT_ORDER,
            "M665 rejects non-C-order-flat payloads")
    require(k_order == EXPECTED_K_ORDER,
            "M665 rejects K-order drift")
    path = trusted_regular_file(trusted_root, path, "bitpack")
    expected = expected_packed_bytes(shape)
    require(path.stat().st_size == expected,
            "bitpack byte length does not match shape")
    elements = checked_positive_product(
        shape, MAX_SUPPORTED_INPUT_ELEMENTS, "input_shape")
    tail = elements & 7
    if tail:
        with path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            last = handle.read(1)[0]
        require((last & (~((1 << tail) - 1) & 0xff)) == 0,
                "nonzero tail padding bits")
    return {"path": str(path), "shape": list(shape),
            "elements": elements, "packed_bytes": expected,
            "sha256": sha256(path), "bit_order": bit_order,
            "flat_order": flat_order, "k_order": k_order}


def phase_bank(destination_y_parity, destination_x_parity):
    destination_y_parity = strict_integer(
        destination_y_parity, "destination_y_parity", 0, 1)
    destination_x_parity = strict_integer(
        destination_x_parity, "destination_x_parity", 0, 1)
    return (destination_y_parity << 1) | destination_x_parity


def build_phase_plan(channels, height, width, bank):
    channels = strict_integer(channels, "channels", 1,
                              MAX_SUPPORTED_DIMENSION)
    height = strict_integer(height, "height", 1, MAX_SUPPORTED_DIMENSION)
    width = strict_integer(width, "width", 1, MAX_SUPPORTED_DIMENSION)
    bank = strict_integer(bank, "phase bank", 0, 3)
    require(bank in M514_PHASE_TAPS, "unknown M514 phase bank")
    parity_y, parity_x = (bank >> 1) & 1, bank & 1
    taps = M514_PHASE_TAPS[bank]
    m_count = checked_positive_product(
        (height, width), MAX_SUPPORTED_PLAN_SITES, "phase_plan_spatial")
    k_count = checked_positive_product(
        (len(taps), channels), MAX_SUPPORTED_K, "phase_plan_k")
    source_flat = np.full((m_count, k_count), -1, dtype=np.int64)
    destination_y = np.empty(m_count, dtype=np.int64)
    destination_x = np.empty(m_count, dtype=np.int64)

    m_index = 0
    for phase_y in range(height):
        dy = 2 * phase_y + parity_y
        for phase_x in range(width):
            dx = 2 * phase_x + parity_x
            destination_y[m_index] = dy
            destination_x[m_index] = dx
            for tap_index, (ky, kx) in enumerate(taps):
                numerator_y = dy + 1 - ky
                numerator_x = dx + 1 - kx
                require((numerator_y & 1) == 0 and
                        (numerator_x & 1) == 0,
                        "phase/tap parity inconsistency")
                sy, sx = numerator_y // 2, numerator_x // 2
                if 0 <= sy < height and 0 <= sx < width:
                    k_start = tap_index * channels
                    source_flat[m_index, k_start:k_start + channels] = (
                        np.arange(channels, dtype=np.int64) * height * width +
                        sy * width + sx)
            m_index += 1
    require(m_index == m_count, "phase destination population drift")
    return {
        "bank": bank,
        "destination_parity_yx": [parity_y, parity_x],
        "taps": [list(tap) for tap in taps],
        "m": m_count,
        "k": k_count,
        "k_order": EXPECTED_K_ORDER,
        "destination_y": destination_y,
        "destination_x": destination_x,
        "source_flat_index": source_flat,
        "valid": source_flat >= 0,
    }


def _read_packed_indices(packed, flat_indices):
    byte_indices = np.right_shift(flat_indices, 3)
    bit_indices = np.bitwise_and(flat_indices, 7).astype(np.uint8)
    return np.bitwise_and(
        np.right_shift(packed[byte_indices], bit_indices), 1).astype(np.uint8)


def iter_polyphase_tiles(bitpack_path, shape, tile_m=256,
                         phases=M514_PHASE_ORDER,
                         bit_order=EXPECTED_BIT_ORDER,
                         flat_order=EXPECTED_FLAT_ORDER,
                         k_order=EXPECTED_K_ORDER,
                         kernel_size=(3, 3), stride=(2, 2),
                         padding=(1, 1), output_padding=(1, 1),
                         dilation=(1, 1), groups=1, trusted_root=None):
    """Yield exact phase tiles; each ``values`` array is [T, tile_M, K]."""
    validate_convtranspose_spec(kernel_size, stride, padding, output_padding,
                                dilation, groups)
    identity = validate_bitpack(bitpack_path, shape, bit_order, flat_order,
                                k_order, trusted_root=trusted_root)
    tile_m = strict_integer(tile_m, "tile_m", 1, MAX_SUPPORTED_PLAN_SITES)
    require(isinstance(phases, (tuple, list)) and phases and
            len(phases) == len(set(phases)) and
            all(isinstance(bank, (int, np.integer)) and
                not isinstance(bank, (bool, np.bool_)) and
                int(bank) in M514_PHASE_TAPS for bank in phases),
            "phases must be unique M514 banks")
    phases = tuple(int(bank) for bank in phases)
    time_steps, _batch, channels, height, width = tuple(identity["shape"])
    plane = channels * height * width
    packed = np.memmap(bitpack_path, dtype=np.uint8, mode="r")
    time_offsets = np.arange(time_steps, dtype=np.int64) * plane

    for bank in phases:
        plan = build_phase_plan(channels, height, width, bank)
        for m_start in range(0, plan["m"], tile_m):
            m_stop = min(plan["m"], m_start + tile_m)
            source = plan["source_flat_index"][m_start:m_stop]
            valid = source >= 0
            safe_source = np.maximum(source, 0)
            flat = time_offsets[:, None, None] + safe_source[None, :, :]
            values = _read_packed_indices(packed, flat)
            values = np.bitwise_and(values, valid[None, :, :])
            yield {
                "phase_bank": bank,
                "destination_parity_yx": plan["destination_parity_yx"],
                "taps": plan["taps"],
                "k_order": plan["k_order"],
                "m_start": m_start,
                "m_stop": m_stop,
                "destination_y": plan["destination_y"][m_start:m_stop],
                "destination_x": plan["destination_x"][m_start:m_stop],
                "source_flat_index": source,
                "valid": valid,
                "values": values,
            }


def materialize_polyphase(bitpack_path, shape, tile_m=256, **kwargs):
    result = {}
    metadata = {}
    for tile in iter_polyphase_tiles(bitpack_path, shape, tile_m=tile_m,
                                     **kwargs):
        bank = tile["phase_bank"]
        result.setdefault(bank, []).append(tile["values"])
        metadata.setdefault(bank, {
            "destination_y": [], "destination_x": [],
            "taps": tile["taps"], "k_order": tile["k_order"],
            "destination_parity_yx": tile["destination_parity_yx"],
        })
        metadata[bank]["destination_y"].append(tile["destination_y"])
        metadata[bank]["destination_x"].append(tile["destination_x"])
    for bank in result:
        result[bank] = np.concatenate(result[bank], axis=1)
        metadata[bank]["destination_y"] = np.concatenate(
            metadata[bank]["destination_y"])
        metadata[bank]["destination_x"] = np.concatenate(
            metadata[bank]["destination_x"])
    return result, metadata


def phase_weight_matrix(weight, bank,
                        weight_layout="CIN_COUT_KY_KX"):
    require(weight_layout == "CIN_COUT_KY_KX",
            "M665 rejects weight-layout drift")
    weight = np.asarray(weight)
    require(weight.ndim == 4 and tuple(weight.shape[2:]) == (3, 3) and
            weight.shape[0] > 0 and weight.shape[1] > 0,
            "weight must be [Cin,Cout,3,3]")
    bank = strict_integer(bank, "phase bank", 0, 3)
    require(bank in M514_PHASE_TAPS, "unknown M514 phase bank")
    # tap-major, then source-channel.  This is the same K order used by the
    # activation matrix and is intentionally not channel-major.
    return np.concatenate(
        [weight[:, :, ky, kx] for ky, kx in M514_PHASE_TAPS[bank]], axis=0)


def reconstruct_convtranspose(bitpack_path, shape, weight, tile_m=256,
                              **kwargs):
    shape = validate_input_shape(shape)
    weight = np.asarray(weight)
    require(weight.ndim == 4 and weight.shape[0] == shape[2],
            "weight Cin does not match bitpack C")
    if np.issubdtype(weight.dtype, np.integer):
        compute_weight = weight.astype(np.int64, copy=False)
        output = np.zeros((shape[0], weight.shape[1], 2 * shape[3],
                           2 * shape[4]), dtype=np.int64)
    else:
        compute_weight = weight.astype(np.float64, copy=False)
        output = np.zeros((shape[0], weight.shape[1], 2 * shape[3],
                           2 * shape[4]), dtype=np.float64)
    for tile in iter_polyphase_tiles(bitpack_path, shape, tile_m=tile_m,
                                     **kwargs):
        matrix = phase_weight_matrix(compute_weight, tile["phase_bank"])
        products = np.matmul(tile["values"].astype(output.dtype), matrix)
        for local_m, (dy, dx) in enumerate(zip(
                tile["destination_y"], tile["destination_x"])):
            output[:, :, int(dy), int(dx)] = products[:, local_m, :]
    return output


def _unpack_input(bitpack_path, shape, trusted_root):
    identity = validate_bitpack(bitpack_path, shape,
                                trusted_root=trusted_root)
    packed = np.fromfile(bitpack_path, dtype=np.uint8)
    bits = np.unpackbits(packed, bitorder="little")[:identity["elements"]]
    return bits.reshape(tuple(identity["shape"]))


def workload_accounting(bitpack_path, shape, output_channels=1,
                        tile_m=256, trusted_root=None):
    shape = validate_input_shape(shape)
    output_channels = strict_integer(
        output_channels, "output_channels", 1, MAX_SUPPORTED_DIMENSION)
    values = _unpack_input(bitpack_path, shape, trusted_root)[:, 0]
    source_popcount = int(values.sum(dtype=np.int64))
    direct_active_taps = 0
    direct_structural_taps_per_time = 0
    channels, height, width = shape[2], shape[3], shape[4]
    for channel in range(channels):
        for sy in range(height):
            for sx in range(width):
                fanout = 0
                for ky, kx in M514_SLOT_ORDER:
                    dy, dx = 2 * sy - 1 + ky, 2 * sx - 1 + kx
                    if 0 <= dy < 2 * height and 0 <= dx < 2 * width:
                        fanout += 1
                direct_structural_taps_per_time += fanout
                direct_active_taps += fanout * int(
                    values[:, channel, sy, sx].sum(dtype=np.int64))

    mapped_active_taps = 0
    mapped_structural_taps_per_time = 0
    materialized_entries_per_time = 0
    for bank in M514_PHASE_ORDER:
        plan = build_phase_plan(channels, height, width, bank)
        mapped_structural_taps_per_time += int(
            plan["valid"].sum(dtype=np.int64))
        materialized_entries_per_time += int(plan["m"] * plan["k"])
    for tile in iter_polyphase_tiles(
            bitpack_path, shape, tile_m=tile_m, trusted_root=trusted_root):
        mapped_active_taps += int(tile["values"].sum(dtype=np.int64))

    require(mapped_structural_taps_per_time ==
            direct_structural_taps_per_time,
            "valid-tap conservation failed")
    require(mapped_active_taps == direct_active_taps,
            "active-tap/popcount conservation failed")
    return {
        "source_popcount": source_popcount,
        "valid_tap_slots_per_time": mapped_structural_taps_per_time,
        "valid_tap_slots_all_time": (
            mapped_structural_taps_per_time * shape[0]),
        "materialized_entries_per_time": materialized_entries_per_time,
        "structural_padding_zero_entries_all_time": (
            (materialized_entries_per_time -
             mapped_structural_taps_per_time) * shape[0]),
        "active_tap_events": mapped_active_taps,
        "active_products": mapped_active_taps * output_channels,
        "dense_valid_products": (mapped_structural_taps_per_time *
                                 shape[0] * output_channels),
        "output_channels": output_channels,
    }


def _record_integer(row, field):
    require(isinstance(row, dict), "M660 record must be an object")
    return strict_integer(row.get(field), "M660 " + field, 0,
                          MAX_SUPPORTED_DIMENSION)


def _packed_identity(row, field):
    identity = row.get(field)
    require(isinstance(identity, dict),
            "M660 route-specific packed identity missing")
    packed_bytes = strict_integer(identity.get("packed_bytes"),
                                  "M660 packed_bytes", 1,
                                  (MAX_SUPPORTED_INPUT_ELEMENTS + 7) // 8)
    digest = identity.get("packed_sha256")
    require(isinstance(digest, str) and len(digest) == 64 and
            all(character in "0123456789abcdef" for character in digest),
            "M660 packed SHA256 is malformed")
    return {"packed_bytes": packed_bytes, "packed_sha256": digest}


def _validate_record_semantics(row, container):
    sample_id = _record_integer(row, "sample_id")
    module_index = _record_integer(row, "module_index")
    require(sample_id in M660_SAMPLE_IDS, "M660 sample_id is outside S10")
    require(module_index in M660_MODULE_NAMES,
            "M660 decoder module_index is unknown")
    require(row.get("name") == M660_MODULE_NAMES[module_index],
            "M660 decoder module name/index mismatch")
    if container == "d0_d2_d3_binary_records":
        require(module_index in M660_BINARY_MODULES and
                row.get("route") == "EXACT_BINARY_BITPACK",
                "M660 D0/D2/D3 container/module/route mismatch")
        identity_field = "input"
    else:
        require(container == "d1_records" and module_index == 1,
                "M660 D1 container/module mismatch")
        require(row.get("route") in (
            "EXACT_SCALED_BINARY_BITPACK",
            "COMMON_FP32_DENSE_FALLBACK"),
            "M660 D1 route mismatch")
        identity_field = ("theta_binary_candidate" if row.get("route") ==
                          "EXACT_SCALED_BINARY_BITPACK" else None)
    return sample_id, module_index, identity_field


def m660_bitpack_records(manifest_path, trusted_root):
    """Return admitted M660 bitpack records without pinning a result SHA."""
    package = _trusted_root(trusted_root)
    manifest_path = trusted_regular_file(
        package, manifest_path, "M660 manifest")
    require(manifest_path.parent == package,
            "M660 manifest must be directly under trusted package root")
    manifest = strict_json(manifest_path)
    require(manifest.get("schema") ==
            "m660_h67_ep35_layer_static_decoder_payload_v1",
            "unexpected M660 manifest schema")
    packing = manifest.get("packing", {})
    require(packing == {"values": [0, 1], "bit_order": "little",
                        "order": "C_ORDER_FLAT",
                        "whole_call_contiguous_copy_allowed": False},
            "M660 packing contract drift")
    require("d0_d2_d3_binary_records" in manifest and
            isinstance(manifest["d0_d2_d3_binary_records"], list) and
            "d1_records" in manifest and
            isinstance(manifest["d1_records"], list),
            "M660 decoder record containers missing or malformed")
    binary_rows = manifest["d0_d2_d3_binary_records"]
    d1_rows = manifest["d1_records"]
    require(len(binary_rows) == 30 and len(d1_rows) == 10,
            "M660 decoder record population is incomplete")
    typed_rows = []
    for row in binary_rows:
        typed_rows.append((row, "d0_d2_d3_binary_records"))
    for row in d1_rows:
        typed_rows.append((row, "d1_records"))
    observed_lattice = []
    result = []
    for row, container in typed_rows:
        sample_id, module_index, identity_field = (
            _validate_record_semantics(row, container))
        observed_lattice.append((sample_id, module_index))
        if identity_field is None:
            continue
        member = safe_member(row.get("relative_path"))
        path = trusted_regular_file(package, member, "M660 payload")
        identity = validate_bitpack(
            path, row.get("input_shape"), trusted_root=package)
        packed_identity = _packed_identity(row, identity_field)
        require(identity["packed_bytes"] ==
                packed_identity["packed_bytes"] and
                identity["sha256"] == packed_identity["packed_sha256"],
                "M660 record bitpack identity mismatch")
        result.append({
            "sample_id": sample_id,
            "module_index": module_index,
            "name": row["name"],
            "route": row["route"],
            "path": str(path),
            "shape": identity["shape"],
            "packed_sha256": identity["sha256"],
        })
    expected_lattice = set((sample_id, module_index)
                           for sample_id in M660_SAMPLE_IDS
                           for module_index in range(4))
    require(len(observed_lattice) == len(set(observed_lattice)) and
            set(observed_lattice) == expected_lattice,
            "duplicate or missing M660 sample/module record")
    result.sort(key=lambda item: (item["sample_id"], item["module_index"]))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitpack", required=True)
    parser.add_argument("--shape", required=True,
                        help="comma-separated T,1,C,H,W")
    parser.add_argument("--output-channels", type=int, required=True)
    parser.add_argument("--tile-m", type=int, default=256)
    parser.add_argument("--trusted-root", required=True)
    args = parser.parse_args(argv)
    shape = tuple(int(item) for item in args.shape.split(","))
    result = {
        "schema": "m665_decoder_polyphase_workload_summary_v1",
        "status": "PASS_EXACT_CPU_MAPPING_INPUT_ONLY",
        "input": validate_bitpack(
            args.bitpack, shape, trusted_root=args.trusted_root),
        "mapping": {
            "phase_order": list(M514_PHASE_ORDER),
            "phase_taps": {str(bank): [list(tap) for tap in
                                        M514_PHASE_TAPS[bank]]
                           for bank in M514_PHASE_ORDER},
            "k_order": EXPECTED_K_ORDER,
            "convtranspose_spec": {key: (list(value)
                if isinstance(value, tuple) else value)
                for key, value in EXPECTED_SPEC.items()},
        },
        "accounting": workload_accounting(
            args.bitpack, shape, args.output_channels, args.tile_m,
            trusted_root=args.trusted_root),
        "claim_boundary": {"exact_input_mapping": True, "cycles": False,
                           "speedup": False, "rtl": False, "eda": False,
                           "paper_headline": False},
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
