#!/usr/bin/env python3
"""M1558 reduced binary source for a future ep34 S2/TSBG capture.

PATCH hooks only accumulate vectorized S1 histogram/debt statistics.  FC1/FC2
hooks emit independently compressed binary frames; no token is represented by
a Python JSON object.  Construction requires a one-shot pre-load permit bound
to the exact output path, M1458 layer inventory and a first-principles size
estimate.

This module is source-only.  It has no checkpoint/model loader, CUDA path,
capture CLI, remote integration or release capability.  Captured int8 values
are diagnostic coordinates only and are not a hardware quantization authority.
Production and synthetic pre-load permits have distinct exact types.  Only the
production issuer may create production provenance, and it always obtains free
space from ``shutil.disk_usage`` rather than a caller-supplied value.
"""

from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import stat
import struct
import sys
import zlib


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = Path(__file__).resolve()
M1552_SOURCE = SOURCE.with_name(
    "capture_m1552_motion_ep34_s2_tsbg_incremental_source_r1.py")
M1555_PRODUCER = HW / (
    "reviews/m1555_m1552_ep34_sparse_compact_producer_source_independent_hammer_r1_20260901")
M1555_DESTINATION = HW / (
    "reviews/m1555_m1554_s2_destination_debt_independent_hammer_r1_20260901")

EXPECTED = {
    M1552_SOURCE: "245d65c98f893811a48c36050d764f5269b45a0657f6118eb33a1e16f2192c94",
    M1555_PRODUCER / "review.json":
        "2723c1ed6531fca46225138f29f870a4cfd58123f8e942191b5e58ef89c53c45",
    M1555_PRODUCER / "SHA256SUMS.seal.sha256":
        "8fb87a157aa539a5ee01b30581c06223f90eb8c7b225f1e21f90c66854b88f92",
    M1555_DESTINATION / "review.json":
        "e9c2313bfb0f9d68e98e3bbb0a72d358991f43b1fd93eb1704153f24f03fc7c4",
    M1555_DESTINATION / "SHA256SUMS.seal.sha256":
        "c6009c0bf81a1a195c1e6218d56e5b616aa38c44313df613053b63aaea0f4b69",
}

SOURCE_SCHEMA = "m1558_motion_ep34_s2_tsbg_reduced_binary_source_r1_v1"
SOURCE_STATUS = "SOURCE_ONLY__REDUCED_BINARY_PRODUCER__NO_GPU_NO_CAPTURE_NO_RELEASE"
PERMIT_SCHEMA = "m1582_m1576_minted_instance_registry_successor_r1_v1"
PRODUCTION_PROVENANCE = "PRODUCTION_REAL_DISK"
SYNTHETIC_PROVENANCE = "SYNTHETIC_CALLER_BUDGET"
TARGET_COUNTS = {"FC1": 12, "FC2": 12, "PATCH": 8}
GROUP_WIDTH = 16
OUTPUT_TILE_WIDTH = 96
FRAME_TOKENS = 4096
MAX_RUNTIME_BYTES = 12 * 1024 * 1024 * 1024
MIN_FREE_AFTER_BYTES = 16 * 1024 * 1024 * 1024
AUXILIARY_UPPER_BYTES = 64 * 1024 * 1024
MAGNITUDE_EDGES = (0, 1, 2, 4, 8, 16, 32, 64, 129)
FRAME_MAGIC = b"M1558F01"
FRAME_VERSION = 1
FRAME_HEADER = struct.Struct("<8sHH11I")
BASE_INVENTORY_SHA256 = "5e04692dfe9b671ef73d4c26497edc8fa83042ab84fdf573ab6b2764a0b21f2e"


class M1558Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1558Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise M1558Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA mismatch")


def strict_json(path, root_type=dict):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           M1558Error("nonfinite JSON: " + token)))
    require(type(value) is root_type, "JSON root type mismatch")
    return value


def canonical_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_m1552():
    regular_exact(M1552_SOURCE, EXPECTED[M1552_SOURCE], "sealed M1552 source")
    spec = importlib.util.spec_from_file_location("m1558_bound_m1552", str(M1552_SOURCE))
    require(spec is not None and spec.loader is not None, "cannot import sealed M1552")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M1552 = load_m1552()


def verify_authorities():
    for path, expected in EXPECTED.items():
        regular_exact(path, expected, str(path.relative_to(ROOT)))
    samples = M1552.verify_bindings()
    producer = strict_json(M1555_PRODUCER / "review.json")
    destination = strict_json(M1555_DESTINATION / "review.json")
    require(producer.get("status") ==
            "NO_GO_M1552_AS_IS_REMOTE_INTEGRATION__SUCCESSOR_REDESIGN_AUTHORING_ONLY__CAPTURE_FORBIDDEN" and
            producer.get("authorization", {}).get(
                "successor_reduced_population_or_binary_rle_integration_source_authoring") is True and
            producer.get("authorization", {}).get("capture") is False,
            "M1555 producer-redesign authority drift")
    require(destination.get("verdict", {}).get("incremental_fc_patch_capture") ==
            "CONDITIONALLY_ALLOWED" and
            destination.get("verdict", {}).get("s2_mechanism_admitted") is False and
            destination.get("verdict", {}).get(
                "cycles_traffic_energy_speedup_authorized") is False and
            destination.get("verdict", {}).get("rtl_vcs_eda_authorized") is False,
            "M1555 destination-screen boundary drift")
    return samples


def product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def frozen_layer_specs():
    """Bind exact M1458 names, shapes, axes, order and S40 populations."""
    verify_authorities()
    base = M1552.frozen_layer_specs()
    require(M1552.canonical_sha(base) == BASE_INVENTORY_SHA256,
            "M1552 exact 32-layer inventory drift")
    runtime = strict_json(M1552.M1458_OPERATORS, list)
    by_name = dict((row["name"], row) for row in runtime)
    require(len(by_name) == len(runtime), "M1458 operator name duplication")
    result = []
    for original in base:
        row = dict(original)
        authority = by_name[row["module_name"]]
        require(int(authority["calls"]) == 40 and
                tuple(authority["input_shape_first"]) == tuple(row["input_shape"]) and
                tuple(authority["output_shape_first"]) == tuple(row["output_shape"]),
                "M1458 layer population/shape drift")
        elements = int(authority["input_elements"])
        active = int(authority["input_active"])
        per_call_elements = product(row["input_shape"])
        tokens_per_call = per_call_elements // int(row["input_channels"])
        require(per_call_elements * 40 == elements and
                tokens_per_call * int(row["input_channels"]) * 40 == elements and
                0 <= active <= elements,
                "M1458 exact element/activity accounting drift")
        row["input_elements_s40"] = elements
        row["input_active_s40"] = active
        row["tokens_per_call"] = tokens_per_call
        row["tokens_s40"] = tokens_per_call * 40
        result.append(row)
    require(len(result) == 32 and
            {key: sum(row["target"] == key for row in result)
             for key in TARGET_COUNTS} == TARGET_COUNTS and
            [row["layer_id"] for row in result] == list(range(32)) and
            [row["operator_order"] for row in result] == sorted(
                row["operator_order"] for row in result),
            "extended exact inventory drift")
    return result


def zlib_bound_total(raw_bytes, frame_count):
    """Conservative sum of zlib compressBound over independently framed data."""
    raw = int(raw_bytes)
    frames = int(frame_count)
    require(raw >= 0 and frames >= 0, "negative zlib bound input")
    return (raw + raw // 4096 + raw // 16384 + raw // 33554432 +
            16 * frames)


def estimate_from_specs(specs, sample_count=40):
    rows = []
    raw_payload = 0
    frame_count = 0
    fc_tokens = 0
    patch_tokens = 0
    for spec in specs:
        tokens = int(spec["tokens_s40"])
        if spec["target"] == "PATCH":
            patch_tokens += tokens
            continue
        require(spec["target"] in ("FC1", "FC2"), "unexpected binary target")
        channels = int(spec["input_channels"])
        bitrow = (channels + 7) // 8
        bitmatrix_bytes = 3 * bitrow * tokens
        nnz_bytes = 2 * tokens
        code_bytes = int(spec["input_active_s40"])
        layer_raw = bitmatrix_bytes + nnz_bytes + code_bytes
        frames_per_call = (int(spec["tokens_per_call"]) + FRAME_TOKENS - 1) // FRAME_TOKENS
        layer_frames = frames_per_call * int(sample_count)
        raw_payload += layer_raw
        frame_count += layer_frames
        fc_tokens += tokens
        rows.append({"layer_id": int(spec["layer_id"]),
                     "target": spec["target"], "tokens": tokens,
                     "channels": channels, "bitmatrix_bytes": bitmatrix_bytes,
                     "uint16_nnz_bytes": nnz_bytes,
                     "nonzero_code_upper_bytes": code_bytes,
                     "raw_payload_upper_bytes": layer_raw,
                     "frame_upper_count": layer_frames})
    compressed_upper = zlib_bound_total(raw_payload, frame_count)
    container_upper = (compressed_upper + frame_count * FRAME_HEADER.size +
                       AUXILIARY_UPPER_BYTES)
    estimate = {
        "schema": "m1558_preload_estimate_r1_v1",
        "inventory_sha256": canonical_sha(specs),
        "sample_count": int(sample_count),
        "fc_tokens": fc_tokens,
        "patch_tokens_histogram_only": patch_tokens,
        "binary_frame_upper_count": frame_count,
        "raw_fc_payload_upper_bytes": raw_payload,
        "zlib_payload_upper_bytes": compressed_upper,
        "frame_header_upper_bytes": frame_count * FRAME_HEADER.size,
        "auxiliary_upper_bytes": AUXILIARY_UPPER_BYTES,
        "result_upper_bytes": container_upper,
        "layers": rows,
        "source_of_population": "M1458 exact input_elements/input_active",
    }
    require(raw_payload > 0 and raw_payload < MAX_RUNTIME_BYTES and
            container_upper < MAX_RUNTIME_BYTES,
            "first-principles result upper bound exceeds strict 12 GiB")
    return estimate


def _permit_authority():
    production_secret = object()
    synthetic_secret = object()
    # These closure-owned dictionaries are the capability authority.  The
    # permit objects deliberately remain opaque handles; their writable slots
    # are never trusted by consume().  Production and synthetic namespaces are
    # distinct so membership in one can never authorize the other provenance.
    production_minted = {}
    synthetic_minted = {}

    class ProductionPreloadPermit(object):
        __slots__ = ("__output", "__inventory", "__estimate", "__free", "__consumed")

        def __init__(self, output, inventory, estimate, free_bytes, token):
            require(token is production_secret,
                    "production permit constructor is private")
            self.__output = str(output)
            self.__inventory = str(inventory)
            self.__estimate = dict(estimate)
            self.__free = int(free_bytes)
            self.__consumed = False

        def consume(self, output, inventory):
            require(type(self) is ProductionPreloadPermit,
                    "production permit exact type invalid")
            try:
                bound_output, bound_inventory, bound_estimate, bound_free = (
                    production_minted.pop(self))
            except KeyError:
                raise M1558Error(
                    "production permit was not minted or was already consumed")
            self.__consumed = True
            require(str(Path(output).resolve()) == bound_output and
                    str(inventory) == bound_inventory,
                    "permit output/inventory mismatch")
            require(not os.path.lexists(bound_output),
                    "permit output namespace no longer fresh")
            return {"schema": PERMIT_SCHEMA,
                    "provenance": PRODUCTION_PROVENANCE,
                    "output": bound_output,
                    "inventory_sha256": bound_inventory,
                    "estimate": dict(bound_estimate),
                    "free_bytes_before": bound_free,
                    "free_bytes_after_upper":
                        bound_free - int(bound_estimate["result_upper_bytes"]),
                    "consumed": True, "checkpoint_loaded": False}

    class SyntheticPreloadPermit(object):
        __slots__ = ("__output", "__inventory", "__estimate", "__free", "__consumed")

        def __init__(self, output, inventory, estimate, free_bytes, token):
            require(token is synthetic_secret,
                    "synthetic permit constructor is private")
            self.__output = str(output)
            self.__inventory = str(inventory)
            self.__estimate = dict(estimate)
            self.__free = int(free_bytes)
            self.__consumed = False

        def consume(self, output, inventory):
            require(type(self) is SyntheticPreloadPermit,
                    "synthetic permit exact type invalid")
            try:
                bound_output, bound_inventory, bound_estimate, bound_free = (
                    synthetic_minted.pop(self))
            except KeyError:
                raise M1558Error(
                    "synthetic permit was not minted or was already consumed")
            self.__consumed = True
            require(str(Path(output).resolve()) == bound_output and
                    str(inventory) == bound_inventory,
                    "permit output/inventory mismatch")
            require(not os.path.lexists(bound_output),
                    "permit output namespace no longer fresh")
            return {"schema": PERMIT_SCHEMA,
                    "provenance": SYNTHETIC_PROVENANCE,
                    "output": bound_output,
                    "inventory_sha256": bound_inventory,
                    "estimate": dict(bound_estimate),
                    "free_bytes_before": bound_free,
                    "free_bytes_after_upper":
                        bound_free - int(bound_estimate["result_upper_bytes"]),
                    "consumed": True, "checkpoint_loaded": False}

    def checked_common(output, specs, sample_count, available, permit_type,
                       permit_secret, minted_registry):
        output = Path(output).resolve()
        require(not os.path.lexists(str(output)),
                "fresh output namespace required")
        parent = output.parent
        require(parent.is_dir() and not parent.is_symlink(),
                "output parent invalid")
        estimate = estimate_from_specs(specs, sample_count)
        available = int(available)
        require(available - int(estimate["result_upper_bytes"]) >
                MIN_FREE_AFTER_BYTES,
                "capture would not leave strictly more than 16 GiB free")
        inventory = canonical_sha(specs)
        permit = permit_type(output, inventory, estimate, available,
                             permit_secret)
        require(permit not in minted_registry, "permit registry collision")
        minted_registry[permit] = (str(output), inventory, dict(estimate),
                                   available)
        return permit

    def issue_production(output):
        # Deliberately no caller-controlled inventory, sample count or free-space
        # argument exists on this production authority.
        specs = frozen_layer_specs()
        resolved = Path(output).resolve()
        require(not os.path.lexists(str(resolved)),
                "fresh output namespace required")
        parent = resolved.parent
        require(parent.is_dir() and not parent.is_symlink(),
                "output parent invalid")
        available = shutil.disk_usage(str(parent)).free
        return checked_common(resolved, specs, 40, available,
                              ProductionPreloadPermit, production_secret,
                              production_minted)

    def issue_synthetic(output, specs, sample_count, free_bytes):
        require(int(sample_count) > 0 and int(sample_count) <= 40,
                "synthetic sample count invalid")
        return checked_common(output, specs, int(sample_count), int(free_bytes),
                              SyntheticPreloadPermit, synthetic_secret,
                              synthetic_minted)

    return (ProductionPreloadPermit, SyntheticPreloadPermit,
            issue_production, issue_synthetic)


(_ProductionPreloadPermit, _SyntheticPreloadPermit,
 _issue_production_permit, _issue_synthetic_permit) = _permit_authority()
del _permit_authority


def issue_preload_permit(output):
    """The only future production permit factory; no model/torch is loaded."""
    return _issue_production_permit(output)


def issue_synthetic_permit(output, specs, sample_count, free_bytes):
    """Test-only issuer; it can never create production provenance."""
    return _issue_synthetic_permit(output, specs, sample_count, free_bytes)


class RuntimeBudget(object):
    def __init__(self, maximum=MAX_RUNTIME_BYTES):
        self.maximum = int(maximum)
        require(self.maximum > 0, "runtime cap invalid")
        self.raw_bytes = 0
        self.disk_bytes = 0

    def charge(self, raw_bytes, disk_bytes):
        raw = int(raw_bytes); disk = int(disk_bytes)
        require(raw >= 0 and disk >= 0, "negative runtime charge")
        require(self.raw_bytes + raw < self.maximum and
                self.disk_bytes + disk < self.maximum,
                "runtime 12 GiB hard cap exceeded")
        self.raw_bytes += raw
        self.disk_bytes += disk


def diagnostic_codebook():
    return {"width_bits": 8, "signed": True, "zero_point": 0,
            "unit_code": 1, "scale_numerator": 1, "scale_denominator": 1,
            "rounding": "nearest_even", "saturation": "signed_clamp",
            "authority": "diagnostic_fixed_point_codeword",
            "diagnostic_capture_only": True,
            "hardware_quant_authority": False, "model_bit_exact": False,
            "tsbg_exact_scope": "captured_codeword_and_contributor_only"}


class TorchBinaryAdapter(object):
    """Vectorized chunk adapter; never creates Python rows or a full CPU tensor."""
    def __init__(self, torch_module):
        self.torch = torch_module

    def shape(self, tensor):
        return tuple(int(value) for value in tensor.shape)

    def chunks(self, tensor, spec, wanted=FRAME_TOKENS):
        value = tensor.detach()
        axis = int(spec["channel_axis"])
        if axis != value.dim() - 1:
            value = value.movedim(axis, -1)
        value = value.reshape(-1, int(spec["input_channels"]))
        for begin in range(0, int(value.shape[0]), int(wanted)):
            chunk = self.torch.clamp(
                self.torch.round(value[begin:begin + int(wanted)]), -128, 127)
            chunk = chunk.to(device="cpu", dtype=self.torch.int8).contiguous()
            yield chunk.numpy()


class SyntheticBinaryAdapter(object):
    def shape(self, tensor):
        return tuple(tensor.shape)

    def chunks(self, tensor, spec, wanted=FRAME_TOKENS):
        import numpy as np
        value = np.asarray(tensor.rows, dtype=np.int16)
        require(value.ndim == 2 and value.shape[1] == int(spec["input_channels"]),
                "synthetic tensor channel drift")
        require(bool(((value >= -128) & (value <= 127)).all()),
                "synthetic diagnostic code outside int8")
        value = value.astype(np.int8)
        for begin in range(0, int(value.shape[0]), int(wanted)):
            yield value[begin:begin + int(wanted)]


def encode_frame_payload(codes, expected_channels):
    import numpy as np
    value = np.asarray(codes)
    require(value.dtype == np.int8 and value.ndim == 2 and
            int(value.shape[1]) == int(expected_channels) and
            0 < int(value.shape[0]) <= FRAME_TOKENS,
            "binary frame code matrix drift")
    support = value != 0
    sign = value < 0
    nonunit = support & (np.abs(value.astype(np.int16)) != 1)
    support_bits = np.packbits(support, axis=1, bitorder="little")
    sign_bits = np.packbits(sign, axis=1, bitorder="little")
    nonunit_bits = np.packbits(nonunit, axis=1, bitorder="little")
    nnz = support.sum(axis=1)
    require(bool((nnz <= 65535).all()), "uint16 nnz overflow")
    nnz_le = nnz.astype("<u2", copy=False)
    nonzero_codes = value[support].astype(np.int8, copy=False)
    raw = (support_bits.tobytes(order="C") + sign_bits.tobytes(order="C") +
           nonunit_bits.tobytes(order="C") + nnz_le.tobytes(order="C") +
           nonzero_codes.tobytes(order="C"))
    return raw, int(nnz.sum()), (int(value.shape[1]) + 7) // 8


def decode_frame_payload(raw, token_count, channels, bitrow_bytes, nnz_total,
                         return_codes=False):
    import numpy as np
    tokens = int(token_count); channels = int(channels); rowsize = int(bitrow_bytes)
    require(tokens > 0 and channels > 0 and rowsize == (channels + 7) // 8,
            "frame payload dimensions invalid")
    matrix_bytes = tokens * rowsize
    fixed = 3 * matrix_bytes + 2 * tokens
    require(len(raw) == fixed + int(nnz_total), "frame raw extent mismatch")
    offset = 0
    matrices = []
    for _unused in range(3):
        packed = np.frombuffer(raw[offset:offset + matrix_bytes], dtype=np.uint8)
        packed = packed.reshape(tokens, rowsize)
        bits = np.unpackbits(packed, axis=1, bitorder="little")
        require(bool((bits[:, channels:] == 0).all()), "nonzero tail bit")
        matrices.append(bits[:, :channels].astype(bool))
        offset += matrix_bytes
    support, sign, nonunit = matrices
    require(bool((sign <= support).all()) and bool((nonunit <= support).all()),
            "sign/nonunit outside support")
    nnz = np.frombuffer(raw[offset:offset + 2 * tokens], dtype="<u2")
    offset += 2 * tokens
    require(bool((nnz == support.sum(axis=1)).all()) and
            int(nnz.sum()) == int(nnz_total), "uint16 nnz/support mismatch")
    codes = np.frombuffer(raw[offset:], dtype=np.int8)
    require(int(codes.size) == int(nnz_total) and bool((codes != 0).all()),
            "nonzero code stream mismatch")
    require(bool(((codes < 0) == sign[support]).all()), "sign/code mismatch")
    require(bool(((np.abs(codes.astype(np.int16)) != 1) == nonunit[support]).all()),
            "nonunit/code mismatch")
    result = {"tokens": tokens, "channels": channels,
              "zero_tokens": int((nnz == 0).sum()),
              "nonzero_codes": int(codes.size),
              "nonunit_codes": int(nonunit.sum())}
    if return_codes:
        dense = np.zeros((tokens, channels), dtype=np.int8)
        dense[support] = codes
        result["codes"] = dense
    return result


class BinaryFrameWriter(object):
    def __init__(self, path, budget):
        self.path = Path(path)
        self.stream = self.path.open("wb")
        self.budget = budget
        self.frames = 0
        self.tokens = 0
        self.nonzero_codes = 0

    def write(self, layer_id, sample_id, frame_index, token_start, codes):
        raw, nnz_total, bitrow = encode_frame_payload(codes, codes.shape[1])
        compressed = zlib.compress(raw, 9)
        header = FRAME_HEADER.pack(
            FRAME_MAGIC, FRAME_VERSION, FRAME_HEADER.size,
            int(layer_id), int(sample_id), int(frame_index), int(token_start),
            int(codes.shape[0]), int(codes.shape[1]), int(bitrow), int(nnz_total),
            len(raw), len(compressed), zlib.crc32(raw) & 0xffffffff)
        self.budget.charge(len(raw), len(header) + len(compressed))
        self.stream.write(header); self.stream.write(compressed)
        self.frames += 1; self.tokens += int(codes.shape[0])
        self.nonzero_codes += int(nnz_total)

    def close(self):
        if self.stream is not None:
            self.stream.flush(); self.stream.close(); self.stream = None


class CanonicalZlibJsonlWriter(object):
    def __init__(self, path, budget):
        self.path = Path(path); self.budget = budget
        self.stream = self.path.open("wb"); self.compressor = zlib.compressobj(9)
        self.rows = 0

    def write(self, value):
        raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        compressed = self.compressor.compress(raw)
        self.budget.charge(len(raw), len(compressed))
        if compressed:
            self.stream.write(compressed)
        self.rows += 1

    def close(self):
        if self.stream is not None:
            tail = self.compressor.flush(zlib.Z_FINISH)
            self.budget.charge(0, len(tail)); self.stream.write(tail)
            self.stream.close(); self.stream = None


def build_static_layers(model, specs):
    named = dict(model.named_modules())
    layers = []
    betas = {}
    next_address = 0
    for spec in specs:
        name = spec["module_name"]
        require(name in named, "frozen target missing from model: " + name)
        module = named[name]
        require(M1552.module_dimensions(module) ==
                (int(spec["input_channels"]), int(spec["output_channels"])),
                "model target dimensions drift: " + name)
        source_groups = (int(spec["input_channels"]) + GROUP_WIDTH - 1) // GROUP_WIDTH
        output_tiles = (int(spec["output_channels"]) + OUTPUT_TILE_WIDTH - 1) // OUTPUT_TILE_WIDTH
        row_bytes = GROUP_WIDTH * OUTPUT_TILE_WIDTH * 4
        base = next_address
        next_address += source_groups * output_tiles * row_bytes
        row = dict(spec)
        row["input_shape"] = list(spec["input_shape"])
        row["output_shape"] = list(spec["output_shape"])
        row["codebook"] = diagnostic_codebook()
        row["weight_layout"] = {
            "base_address": base, "source_group_count": source_groups,
            "output_tile_count": output_tiles, "row_bytes": row_bytes,
            "block_count": source_groups * output_tiles, "bank_count": 8,
            "address_formula":
                "base+(output_tile_id*source_group_count+source_group_id)*row_bytes",
            "bank_formula": "(address//row_bytes)%8",
            "identity": "layer_id,source_group_id,output_tile_id"}
        layers.append(row)
        betas[int(spec["layer_id"])] = M1552.weight_beta_by_tile(
            module, int(spec["output_channels"]))
    return layers, betas


class ReducedBinaryProducer(object):
    def __init__(self, model, adapter, root, specs, sample_order, permit,
                 production_inventory=False):
        self.production_inventory = bool(production_inventory)
        expected_type = (_ProductionPreloadPermit if self.production_inventory
                         else _SyntheticPreloadPermit)
        expected_provenance = (PRODUCTION_PROVENANCE if self.production_inventory
                               else SYNTHETIC_PROVENANCE)
        require(type(permit) is expected_type,
                "permit exact type/provenance does not match producer mode")
        self.root = Path(root).resolve()
        self.specs = [dict(row) for row in specs]
        self.inventory_sha256 = canonical_sha(self.specs)
        if self.production_inventory:
            require(len(self.specs) == 32 and self.inventory_sha256 ==
                    canonical_sha(frozen_layer_specs()),
                    "production inventory is not exact M1458")
        self.permit_receipt = permit.consume(self.root, self.inventory_sha256)
        require(self.permit_receipt.get("provenance") == expected_provenance,
                "consumed permit provenance drift")
        require(not os.path.lexists(str(self.root)), "producer output must be fresh")
        self.root.mkdir()
        self.samples = list(sample_order["samples"])
        expected_count = int(self.permit_receipt["estimate"]["sample_count"])
        require(len(self.samples) == expected_count and
                [int(row["global_sample_id"]) for row in self.samples] ==
                list(range(expected_count)), "sample identity/order drift")
        self.model = model; self.adapter = adapter
        self.layers, self.betas = build_static_layers(model, self.specs)
        self.budget = RuntimeBudget()
        self.binary = BinaryFrameWriter(self.root / "fc_frames.bin", self.budget)
        self.patch = CanonicalZlibJsonlWriter(
            self.root / "patch_s1_histogram_debt.jsonl.zlib", self.budget)
        self.active_sample = None; self.expected_hook = 0; self.handles = []
        self.patch_accum = {}; self.observed_tokens = {}; self.observed_active = {}
        self._write_static(sample_order)
        self._attach()

    def _write_static(self, sample_order):
        sample_raw = (json.dumps(sample_order, indent=2, sort_keys=True) + "\n").encode("utf-8")
        layers_value = {"schema": "m1558_static_layers_r1_v1",
                        "status": "EXACT_INVENTORY_OR_SYNTHETIC_AS_DECLARED",
                        "inventory_sha256": self.inventory_sha256,
                        "layers": self.layers}
        layers_raw = (json.dumps(layers_value, indent=2, sort_keys=True) + "\n").encode("utf-8")
        permit_raw = (json.dumps(self.permit_receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
        self.budget.charge(len(sample_raw) + len(layers_raw) + len(permit_raw),
                           len(sample_raw) + len(layers_raw) + len(permit_raw))
        (self.root / "sample_order.json").write_bytes(sample_raw)
        (self.root / "layers.json").write_bytes(layers_raw)
        (self.root / "preload_permit_receipt.json").write_bytes(permit_raw)

    def _attach(self):
        named = dict(self.model.named_modules())
        for spec in self.specs:
            module = named[spec["module_name"]]
            def hook(_module, inputs, _output, _spec=spec):
                self._on_input(_spec, inputs)
            self.handles.append(module.register_forward_hook(hook))

    def begin_sample(self, sample):
        require(self.active_sample is None, "nested producer sample")
        sample_id = int(sample["global_sample_id"])
        require(0 <= sample_id < len(self.samples) and
                M1552.project_m1434_sample(sample) == self.samples[sample_id],
                "producer sample identity/order drift")
        require(sample_id == getattr(self, "completed_samples", 0),
                "sample order drift")
        self.active_sample = self.samples[sample_id]
        self.expected_hook = 0; self.patch_accum = {}

    def _on_input(self, spec, inputs):
        require(self.active_sample is not None, "hook fired outside active sample")
        require(self.expected_hook < len(self.specs) and
                int(spec["layer_id"]) == int(self.specs[self.expected_hook]["layer_id"]),
                "hook order/duplicate drift")
        self.expected_hook += 1
        tensors = [value for value in inputs if hasattr(value, "shape")]
        require(len(tensors) == 1, "hook requires exactly one tensor input")
        tensor = tensors[0]
        require(self.adapter.shape(tensor) == tuple(spec["input_shape"]),
                "hook input shape drift")
        token_start = 0; frame_index = 0; active_total = 0
        patch_counts = [0] * (len(MAGNITUDE_EDGES) - 1)
        patch_magnitude = [0] * (len(MAGNITUDE_EDGES) - 1)
        for codes in self.adapter.chunks(tensor, spec, FRAME_TOKENS):
            require(hasattr(codes, "dtype") and str(codes.dtype) == "int8" and
                    len(codes.shape) == 2 and
                    int(codes.shape[1]) == int(spec["input_channels"]),
                    "adapter binary chunk drift")
            if spec["target"] == "PATCH":
                counts, magnitudes, active = vector_patch_histogram(codes)
                patch_counts = [a + b for a, b in zip(patch_counts, counts)]
                patch_magnitude = [a + b for a, b in zip(patch_magnitude, magnitudes)]
                active_total += active
            else:
                self.binary.write(spec["layer_id"],
                    self.active_sample["global_sample_id"], frame_index,
                    token_start, codes)
                active_total += int((codes != 0).sum())
                frame_index += 1
            token_start += int(codes.shape[0])
        require(token_start == int(spec["tokens_per_call"]), "token population drift")
        key = int(spec["layer_id"])
        self.observed_tokens[key] = self.observed_tokens.get(key, 0) + token_start
        self.observed_active[key] = self.observed_active.get(key, 0) + active_total
        if spec["target"] == "PATCH":
            self._write_patch_rows(spec, patch_counts, patch_magnitude, active_total)

    def _write_patch_rows(self, spec, counts, magnitude, active_total):
        sample_id = int(self.active_sample["global_sample_id"])
        betas = self.betas[int(spec["layer_id"])]
        for output_tile, beta in enumerate(betas):
            self.patch.write({
                "schema": "m1558_patch_s1_histogram_debt_r1_v1",
                "sample_global_id": sample_id,
                "layer_id": int(spec["layer_id"]),
                "output_tile_id": output_tile,
                "magnitude_bin_edges_abs_code": list(MAGNITUDE_EDGES),
                "count_by_magnitude_bin": list(counts),
                "beta_abs_code_debt_by_magnitude_bin":
                    [int(beta) * int(value) for value in magnitude],
                "nonzero_source_count": int(active_total),
                "beta_rounding": "ceil_upper_bound",
                "per_token_payload_emitted": False})

    def end_sample(self):
        require(self.active_sample is not None and self.expected_hook == len(self.specs),
                "sample hook population incomplete")
        self.completed_samples = getattr(self, "completed_samples", 0) + 1
        self.active_sample = None; self.patch_accum = {}

    def finalize_source_result(self):
        require(self.active_sample is None and
                getattr(self, "completed_samples", 0) == len(self.samples),
                "producer sample population incomplete")
        for spec in self.specs:
            layer = int(spec["layer_id"])
            require(self.observed_tokens.get(layer) == int(spec["tokens_s40"]) and
                    self.observed_active.get(layer, 0) <= int(spec["input_active_s40"]),
                    "observed token/activity exceeds M1458-bound population")
        while self.handles:
            self.handles.pop().remove()
        self.binary.close(); self.patch.close()
        manifest = {
            "schema": "m1558_reduced_binary_capture_manifest_r1_v1",
            "status": ("SYNTHETIC_SOURCE_RESULT__NO_CAPTURE_AUTHORITY" if
                       not self.production_inventory else
                       "PRODUCTION_PAYLOAD_REQUIRES_INDEPENDENT_RELEASE_AND_HAMMER"),
            "identity": {"inventory_sha256": self.inventory_sha256,
                         "checkpoint_sha256": M1552.CHECKPOINT_SHA256,
                         "m1458_manifest_sha256": M1552.EXPECTED[M1552.M1458_MANIFEST]},
            "population": {"samples": len(self.samples), "layers": len(self.layers),
                           "FC1": sum(row["target"] == "FC1" for row in self.layers),
                           "FC2": sum(row["target"] == "FC2" for row in self.layers),
                           "PATCH": sum(row["target"] == "PATCH" for row in self.layers),
                           "fc_frames": self.binary.frames,
                           "fc_tokens": self.binary.tokens,
                           "patch_histogram_rows": self.patch.rows},
            "encoding": {
                "fc_container": "independent_zlib_binary_frames",
                "fc_payload": "support_sign_nonunit_bitmatrices_then_uint16_nnz_then_row_major_nonzero_int8_codes",
                "zero_fc_tokens_retained": True,
                "patch_per_token_payload": False,
                "patch_payload": "vectorized_sample_layer_output_tile_S1_histogram_and_debt_only",
                "canonical_token_order": True,
                "full_tensor_saved": False},
            "runtime_budget": {"hard_cap_bytes": MAX_RUNTIME_BYTES,
                               "raw_bytes": self.budget.raw_bytes,
                               "disk_bytes": self.budget.disk_bytes},
            "claim_boundary": {
                "source_only": True, "diagnostic_capture_only": True,
                "hardware_quantization_authority": False, "model_bit_exact": False,
                "tsbg_exact_scope": "captured_codeword_and_contributor_only",
                "s2_mechanism_admitted": False, "aee": False, "cycles": False,
                "traffic": False, "energy": False, "speedup": False,
                "rtl": False, "paper_headline": False}}
        manifest_raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        self.budget.charge(len(manifest_raw), len(manifest_raw))
        (self.root / "capture_manifest.json").write_bytes(manifest_raw)
        complete = ("M1558_SYNTHETIC_SOURCE_COMPLETE__NO_GPU_NO_CAPTURE_NO_RELEASE\n"
                    if not self.production_inventory else
                    "M1558_PAYLOAD_COMPLETE__INDEPENDENT_RELEASE_HAMMER_REQUIRED\n")
        self.budget.charge(len(complete), len(complete))
        (self.root / "RUN_COMPLETE.txt").write_text(complete, encoding="ascii")
        members = sorted(["RUN_COMPLETE.txt", "capture_manifest.json", "fc_frames.bin",
                          "layers.json", "patch_s1_histogram_debt.jsonl.zlib",
                          "preload_permit_receipt.json", "sample_order.json"])
        sums = "".join("{}  {}\n".format(sha256(self.root / name), name)
                       for name in members)
        self.budget.charge(len(sums), len(sums))
        (self.root / "SHA256SUMS").write_text(sums, encoding="ascii")
        seal = "{}  SHA256SUMS\n".format(sha256(self.root / "SHA256SUMS"))
        self.budget.charge(len(seal), len(seal))
        (self.root / "SHA256SUMS.seal.sha256").write_text(seal, encoding="ascii")
        require(sum(path.stat().st_size for path in self.root.iterdir() if path.is_file()) <
                MAX_RUNTIME_BYTES, "final directory exceeds runtime hard cap")
        return self.root


def vector_patch_histogram(codes):
    import numpy as np
    value = np.asarray(codes)
    require(value.dtype == np.int8 and value.ndim == 2,
            "PATCH histogram requires int8 matrix")
    magnitude = np.abs(value.astype(np.int16)).reshape(-1)
    hist = np.bincount(magnitude, minlength=129)
    counts = []
    sums = []
    for low, high in zip(MAGNITUDE_EDGES[:-1], MAGNITUDE_EDGES[1:]):
        counts.append(int(hist[low:high].sum()))
        sums.append(int(sum(index * int(hist[index]) for index in range(low, high))))
    # Bin 0 is deliberately excluded from the S1 nonzero histogram.
    counts[0] -= int(hist[0])
    require(counts[0] >= 0 and sum(counts) == int((magnitude != 0).sum()),
            "PATCH histogram population drift")
    return counts, sums, int(sum(counts))


def _read_exact(stream, size):
    value = stream.read(int(size))
    require(len(value) == int(size), "truncated binary frame")
    return value


def validate_binary_result(root, specs, sample_order):
    """Incremental parser/validator; never decompresses the full file."""
    root = Path(root)
    require(sum(path.stat().st_size for path in root.iterdir() if path.is_file()) <
            MAX_RUNTIME_BYTES, "result directory exceeds runtime hard cap")
    sums = root / "SHA256SUMS"
    seal = root / "SHA256SUMS.seal.sha256"
    require(seal.read_text(encoding="ascii").split() == [sha256(sums), "SHA256SUMS"],
            "result outer seal drift")
    for line in sums.read_text(encoding="ascii").splitlines():
        digest, name = line.split(None, 1)
        member = root / name.strip()
        require(member.parent == root and member.is_file() and
                not member.is_symlink() and sha256(member) == digest,
                "result member seal drift: " + name)
    manifest = strict_json(root / "capture_manifest.json")
    layers = strict_json(root / "layers.json")
    static_fields = ("layer_id", "target", "module_name", "operator",
                     "operator_order", "channel_axis", "input_channels",
                     "output_channels", "input_elements_s40", "input_active_s40",
                     "tokens_per_call", "tokens_s40")
    require(layers["inventory_sha256"] == canonical_sha(specs) and
            len(layers["layers"]) == len(specs), "static inventory mismatch")
    for emitted, expected in zip(layers["layers"], specs):
        require(all(emitted[field] == expected[field] for field in static_fields) and
                emitted["input_shape"] == list(expected["input_shape"]) and
                emitted["output_shape"] == list(expected["output_shape"]),
                "static layer name/shape/axis/order drift")
    require(strict_json(root / "sample_order.json") == sample_order,
            "sample-order file drift")
    permit = strict_json(root / "preload_permit_receipt.json")
    expected_provenance = (PRODUCTION_PROVENANCE if
                           manifest["status"] ==
                           "PRODUCTION_PAYLOAD_REQUIRES_INDEPENDENT_RELEASE_AND_HAMMER"
                           else SYNTHETIC_PROVENANCE)
    require(permit["consumed"] is True and
            permit.get("schema") == PERMIT_SCHEMA and
            permit.get("provenance") == expected_provenance and
            permit["inventory_sha256"] == canonical_sha(specs) and
            permit["estimate"] == estimate_from_specs(specs, len(sample_order["samples"])),
            "preload permit receipt drift")
    samples = list(sample_order["samples"])
    expected_pairs = [(int(sample["global_sample_id"]), int(spec["layer_id"]))
                      for sample in samples for spec in specs
                      if spec["target"] in ("FC1", "FC2")]
    specs_by_id = dict((int(row["layer_id"]), row) for row in specs)
    pair_index = 0; pair_frame = 0; pair_token = 0
    frames = 0; tokens = 0; zero_tokens = 0; nonzero_codes = 0
    active_by_layer = {}
    with (root / "fc_frames.bin").open("rb") as stream:
        while True:
            prefix = stream.read(FRAME_HEADER.size)
            if not prefix:
                break
            require(len(prefix) == FRAME_HEADER.size, "truncated frame header")
            values = FRAME_HEADER.unpack(prefix)
            (magic, version, header_size, layer_id, sample_id, frame_index,
             token_start, token_count, channels, bitrow, nnz_total,
             raw_bytes, compressed_bytes, crc32) = values
            require(magic == FRAME_MAGIC and version == FRAME_VERSION and
                    header_size == FRAME_HEADER.size, "frame header identity drift")
            require(pair_index < len(expected_pairs) and
                    (sample_id, layer_id) == expected_pairs[pair_index],
                    "frame sample/layer canonical order drift")
            spec = specs_by_id[layer_id]
            require(frame_index == pair_frame and token_start == pair_token and
                    channels == int(spec["input_channels"]) and
                    0 < token_count <= FRAME_TOKENS,
                    "frame order/dimensions drift")
            maximum_raw = (3 * token_count * ((channels + 7) // 8) +
                           2 * token_count + token_count * channels)
            minimum_raw = 3 * token_count * ((channels + 7) // 8) + 2 * token_count
            require(0 <= nnz_total <= token_count * channels and
                    minimum_raw <= raw_bytes <= maximum_raw and
                    0 < compressed_bytes <= zlib_bound_total(raw_bytes, 1) and
                    compressed_bytes < MAX_RUNTIME_BYTES,
                    "frame size/count fields exceed mathematical bounds")
            compressed = _read_exact(stream, compressed_bytes)
            decoder = zlib.decompressobj()
            raw = decoder.decompress(compressed) + decoder.flush()
            require(decoder.eof and not decoder.unused_data and
                    not decoder.unconsumed_tail and len(raw) == raw_bytes and
                    (zlib.crc32(raw) & 0xffffffff) == crc32,
                    "frame zlib/CRC/extent drift")
            decoded = decode_frame_payload(raw, token_count, channels, bitrow,
                                           nnz_total, return_codes=False)
            frames += 1; tokens += decoded["tokens"]
            zero_tokens += decoded["zero_tokens"]
            nonzero_codes += decoded["nonzero_codes"]
            active_by_layer[layer_id] = active_by_layer.get(layer_id, 0) + decoded["nonzero_codes"]
            pair_frame += 1; pair_token += token_count
            expected_tokens = int(spec["tokens_per_call"])
            require(pair_token <= expected_tokens, "frame token overflow")
            if pair_token == expected_tokens:
                pair_index += 1; pair_frame = 0; pair_token = 0
    require(pair_index == len(expected_pairs) and pair_token == 0,
            "binary population incomplete")
    for spec in specs:
        if spec["target"] in ("FC1", "FC2"):
            require(active_by_layer.get(int(spec["layer_id"]), 0) <=
                    int(spec["input_active_s40"]),
                    "binary active population exceeds M1458 bound")

    patch_rows = 0
    decoder = zlib.decompressobj()
    expected_patch = [(int(sample["global_sample_id"]), int(spec["layer_id"]), tile)
                      for sample in samples for spec in specs if spec["target"] == "PATCH"
                      for tile in range((int(spec["output_channels"]) +
                                         OUTPUT_TILE_WIDTH - 1) // OUTPUT_TILE_WIDTH)]
    with (root / "patch_s1_histogram_debt.jsonl.zlib").open("rb") as stream:
        buffer = b""
        while True:
            chunk = stream.read(1 << 16)
            if not chunk:
                buffer += decoder.flush(); break
            buffer += decoder.decompress(chunk)
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line:
                    continue
                row = json.loads(line.decode("utf-8"))
                require(patch_rows < len(expected_patch) and
                        (int(row["sample_global_id"]), int(row["layer_id"]),
                         int(row["output_tile_id"])) == expected_patch[patch_rows] and
                        row["per_token_payload_emitted"] is False and
                        row["magnitude_bin_edges_abs_code"] == list(MAGNITUDE_EDGES) and
                        len(row["count_by_magnitude_bin"]) == len(MAGNITUDE_EDGES) - 1 and
                        len(row["beta_abs_code_debt_by_magnitude_bin"]) ==
                            len(MAGNITUDE_EDGES) - 1 and
                        sum(int(value) for value in row["count_by_magnitude_bin"]) ==
                            int(row["nonzero_source_count"]),
                        "PATCH histogram canonical order/schema drift")
                patch_rows += 1
    require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail and
            buffer == b"" and patch_rows == len(expected_patch),
            "PATCH histogram stream incomplete")
    require(manifest["population"]["fc_frames"] == frames and
            manifest["population"]["fc_tokens"] == tokens and
            manifest["population"]["patch_histogram_rows"] == patch_rows,
            "manifest population mismatch")
    require(manifest["claim_boundary"]["hardware_quantization_authority"] is False and
            manifest["claim_boundary"]["s2_mechanism_admitted"] is False and
            all(manifest["claim_boundary"][key] is False for key in
                ("aee", "cycles", "traffic", "energy", "speedup", "rtl",
                 "paper_headline")), "manifest claim boundary drift")
    return {"status": "PASS_M1558_INCREMENTAL_BINARY_VALIDATION",
            "frames": frames, "fc_tokens": tokens, "zero_fc_tokens": zero_tokens,
            "nonzero_codes": nonzero_codes, "patch_histogram_rows": patch_rows,
            "hardware_quantization_authority": False}


def production_release(_token=None):
    raise M1558Error("M1558 is source-only; remote integration, GPU, capture and release remain forbidden")


def describe():
    return {"schema": SOURCE_SCHEMA, "status": SOURCE_STATUS,
            "targets": TARGET_COUNTS, "samples": 40,
            "population": {"FC1_FC2_tokens": 44640000,
                           "PATCH_tokens_histogram_only": 430080000},
            "format": {"per_token_json": False,
                       "fc": "chunked independent zlib binary frames",
                       "patch": "vectorized histogram/debt only"},
            "preload": {"permit_required": True,
                        "production_provenance": PRODUCTION_PROVENANCE,
                        "synthetic_provenance": SYNTHETIC_PROVENANCE,
                        "production_free_space": "shutil.disk_usage_only",
                        "production_caller_free_override": False,
                        "raw_upper_source": "M1458 input_elements/input_active",
                        "raw_upper_bytes": 7528535874,
                        "strict_max_bytes": MAX_RUNTIME_BYTES,
                        "strict_free_after_bytes": MIN_FREE_AFTER_BYTES},
            "quantization": {"hardware_authority": False,
                             "exact_scope": "captured_codeword_and_contributor_only"},
            "execution": {"gpu": False, "ssh": False, "capture": False,
                          "release": False, "automatic_retry": False}}


def source_self_check():
    specs = frozen_layer_specs()
    estimate = estimate_from_specs(specs)
    require(estimate["fc_tokens"] == 44640000 and
            estimate["patch_tokens_histogram_only"] == 430080000 and
            estimate["raw_fc_payload_upper_bytes"] == 7528535874 and
            len(estimate["layers"]) == 24,
            "first-principles production estimate drift")
    return {"status": "PASS_M1558_SOURCE_SELF_CHECK__NO_GPU_NO_CAPTURE",
            "layers": len(specs), "FC_layers": len(estimate["layers"]),
            "fc_tokens": estimate["fc_tokens"],
            "patch_tokens_histogram_only": estimate["patch_tokens_histogram_only"],
            "raw_fc_payload_upper_bytes": estimate["raw_fc_payload_upper_bytes"],
            "result_upper_bytes": estimate["result_upper_bytes"],
            "hardware_quantization_authority": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--describe", action="store_true")
    modes.add_argument("--source-self-check", action="store_true")
    args = parser.parse_args(argv)
    value = describe() if args.describe else source_self_check()
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
