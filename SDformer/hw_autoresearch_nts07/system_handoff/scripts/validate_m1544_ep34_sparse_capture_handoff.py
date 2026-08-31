#!/usr/bin/env python3
"""Validate the compact M1544 ep34 S2/TSBG incremental capture.

This module is deliberately Python 3.6 compatible and uses only the standard
library.  It performs no model execution, GPU access, network access, or file
mutation.  The capture format stores one compressed record per logical token;
only non-empty source groups carry bitsets and non-zero fixed-point codes.
Static weight-block addresses are stored once in ``layers.json`` rather than
being repeated for every token.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import zlib


CHECKPOINT_SHA256 = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
M1458_MANIFEST_SHA256 = "3ab8431e3d7d17d6933c0b87da4a3405e87c97ccc302a27c78491b0a02491d6d"
M1458_INNER_SHA256 = "f7f7a08696611875837196b990575453141b5e8edbf6d4aae61f7db1ed238b8e"
M1458_OUTER_SHA256 = "7cf434b834d30c003153eef8e83e70d574b1c5a7d20ca4c2208902c6e0c76eed"
M1458_ORDER_SHA256 = "88db38f9cc3f3e0b89cf332ef84958ed87e7c84873075e4399a2a54d2ce64c47"
M1540_REVIEW_SHA256 = "218e3d23fae126ddc4a8655f8e9cd7cb762276ab87c7494b7ad05f6e469730bb"
M1541_REVIEW_SHA256 = "849fd69b735779057ea2d197985b1dc81183f62b6c49c569f490659cdef86365"

MANIFEST_SCHEMA = "m1544_ep34_sparse_incremental_capture_manifest_r1_v1"
MANIFEST_STATUS = "CAPTURE_COMPLETE__INDEPENDENT_HAMMER_REQUIRED__NO_PERFORMANCE_CLAIM"
LAYER_SCHEMA = "m1544_ep34_sparse_capture_layers_r1_v1"
TOKEN_SCHEMA = "m1544_ep34_sparse_token_source_groups_r1_v1"
S1_SCHEMA = "m1544_ep34_s1_histogram_debt_r1_v1"
SAMPLE_SCHEMA = "m1544_ep34_m1458_sample_order_r1_v1"
PASS_TOKEN = "PASS_M1544_EP34_SPARSE_CAPTURE_READ_ONLY_VALIDATE"

CAPTURE_MEMBERS = {
    "capture_manifest.json", "sample_order.json", "layers.json",
    "token_source_groups.jsonl.zlib", "s1_histogram_debt.jsonl.zlib",
    "SHA256SUMS", "SHA256SUMS.seal.sha256", "RUN_COMPLETE.txt",
}
TARGETS = {"FC1", "FC2", "PATCH"}
HEX = set("0123456789abcdef")


class M1544Error(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise M1544Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def strict_json_bytes(payload, label):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key in " + label + ": " + key)
            result[key] = value
        return result

    def reject(value):
        raise M1544Error("non-finite JSON constant in " + label + ": " + value)

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
                           parse_constant=reject)
    except (UnicodeError, ValueError) as exc:
        raise M1544Error("invalid JSON in " + label) from exc
    require(isinstance(value, dict), label + " root must be an object")
    return value


def strict_json(path):
    return strict_json_bytes(path.read_bytes(), str(path))


def regular(path, label):
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise M1544Error("missing " + label) from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be a regular non-symlink")


def safe_member(value):
    require(type(value) is str and value and not value.startswith("/") and
            "\\" not in value and "//" not in value, "unsafe member path")
    pure = PurePosixPath(value)
    require(str(pure) == value and all(part not in ("", ".", "..") for part in pure.parts),
            "non-canonical member path")
    return value


def canonical_json_sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_sha_manifest(root):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    regular(manifest, "capture SHA256SUMS")
    regular(outer, "capture outer seal")
    outer_rows = outer.read_text(encoding="ascii").splitlines()
    require(len(outer_rows) == 1, "outer seal must contain one row")
    outer_parts = outer_rows[0].split("  ", 1)
    require(len(outer_parts) == 2 and outer_parts[1] == "SHA256SUMS" and
            outer_parts[0] == sha256(manifest), "outer seal mismatch")
    rows = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        parts = line.split("  ", 1)
        require(len(parts) == 2 and len(parts[0]) == 64 and set(parts[0]) <= HEX,
                "malformed SHA256SUMS row")
        name = safe_member(parts[1])
        require("/" not in name and name not in rows and
                name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"),
                "invalid or duplicate SHA member")
        path = root / name
        regular(path, "capture member " + name)
        require(sha256(path) == parts[0], "capture member SHA mismatch: " + name)
        rows[name] = parts[0]
    require(set(rows) == CAPTURE_MEMBERS - {"SHA256SUMS", "SHA256SUMS.seal.sha256"},
            "sealed member population mismatch")
    actual = {path.name for path in root.iterdir()}
    require(actual == CAPTURE_MEMBERS, "capture directory population mismatch")
    return rows


def validate_sample_order(value):
    require(set(value) == {"schema", "identity", "samples"} and
            value["schema"] == SAMPLE_SCHEMA, "sample order schema/shape mismatch")
    identity = value["identity"]
    require(identity == {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "m1458_manifest_sha256": M1458_MANIFEST_SHA256,
        "m1458_inner_manifest_sha256": M1458_INNER_SHA256,
        "m1458_outer_file_sha256": M1458_OUTER_SHA256,
    }, "sample order identity mismatch")
    samples = value["samples"]
    require(isinstance(samples, list) and len(samples) == 40, "sample order must contain 40 rows")
    minimal = []
    sequence_ids = {}
    for expected, row in enumerate(samples):
        require(isinstance(row, dict) and set(row) == {
            "global_sample_id", "sequence", "sequence_sample_id", "sample_key", "sha256"
        }, "sample order row shape mismatch")
        require(row["global_sample_id"] == expected and
                type(row["sequence"]) is str and row["sequence"] and
                type(row["sequence_sample_id"]) is int and row["sequence_sample_id"] >= 0 and
                type(row["sample_key"]) is str and row["sample_key"] and
                type(row["sha256"]) is str and len(row["sha256"]) == 64 and
                set(row["sha256"]) <= HEX, "sample order row malformed")
        prior = sequence_ids.setdefault(row["sequence"], [])
        prior.append(row["sequence_sample_id"])
        minimal.append(row)
    require(canonical_json_sha(minimal) == M1458_ORDER_SHA256,
            "sample order differs from M1458 exact S40 order")
    for sequence, ids in sequence_ids.items():
        require(ids == list(range(len(ids))), "non-contiguous sequence order: " + sequence)
    return samples


def validate_codebook(value, label):
    required = {
        "width_bits", "signed", "zero_point", "unit_code", "scale_numerator",
        "scale_denominator", "rounding", "saturation", "authority",
        "diagnostic_capture_only", "hardware_quant_authority",
    }
    require(isinstance(value, dict) and set(value) == required, label + " shape mismatch")
    width = value["width_bits"]
    require(width in (8, 16) and value["signed"] is True and value["zero_point"] == 0 and
            type(value["unit_code"]) is int and value["unit_code"] > 0 and
            type(value["scale_numerator"]) is int and value["scale_numerator"] > 0 and
            type(value["scale_denominator"]) is int and value["scale_denominator"] > 0 and
            value["rounding"] in ("nearest_even", "nearest_away", "floor", "ceil") and
            value["saturation"] == "signed_clamp" and
            value["authority"] in
            ("captured_binary_codeword", "diagnostic_fixed_point_codeword") and
            value["diagnostic_capture_only"] is True and
            value["hardware_quant_authority"] is False,
            label + " malformed")
    require(value["unit_code"] < (1 << (width - 1)), label + " unit code out of range")


def validate_layers(value):
    require(set(value) == {"schema", "status", "layers"} and
            value["schema"] == LAYER_SCHEMA and
            value["status"] == "STATIC_WEIGHT_LAYOUT_COMPLETE__NO_CYCLE_OR_ENERGY_CLAIM",
            "layers schema/status/shape mismatch")
    layers = value["layers"]
    require(isinstance(layers, list) and layers, "layers must be non-empty")
    mapped = {}
    seen_targets = set()
    for expected_id, layer in enumerate(layers):
        required = {
            "layer_id", "target", "module_name", "operator_order", "input_channels",
            "output_channels", "group_width", "output_tile_width", "codebook",
            "weight_layout", "s1_eligible", "s1_magnitude_bin_edges_abs_code",
        }
        require(isinstance(layer, dict) and set(layer) == required,
                "layer row shape mismatch")
        require(layer["layer_id"] == expected_id and layer["target"] in TARGETS and
                type(layer["module_name"]) is str and layer["module_name"] and
                type(layer["operator_order"]) is int and layer["operator_order"] >= 0 and
                type(layer["input_channels"]) is int and layer["input_channels"] > 0 and
                type(layer["output_channels"]) is int and layer["output_channels"] > 0 and
                type(layer["group_width"]) is int and 1 <= layer["group_width"] <= 96 and
                type(layer["output_tile_width"]) is int and 1 <= layer["output_tile_width"] <= 96,
                "layer identity/dimensions malformed")
        validate_codebook(layer["codebook"], "layer codebook")
        edges = layer["s1_magnitude_bin_edges_abs_code"]
        if layer["s1_eligible"]:
            require(layer["target"] == "PATCH" and isinstance(edges, list) and
                    2 <= len(edges) <= 33 and edges[0] == 0 and
                    all(type(item) is int for item in edges) and
                    all(edges[index] < edges[index + 1] for index in range(len(edges) - 1)),
                    "S1 magnitude bins malformed")
        else:
            require(edges == [], "non-S1 layer must not carry S1 bins")
        layout = layer["weight_layout"]
        require(isinstance(layout, dict) and set(layout) == {
            "base_address", "bank_count", "row_bytes", "address_formula",
            "bank_formula", "row_buffer_baseline", "blocks"
        } and type(layout["base_address"]) is int and layout["base_address"] >= 0 and
                type(layout["bank_count"]) is int and layout["bank_count"] > 0 and
                type(layout["row_bytes"]) is int and layout["row_bytes"] > 0 and
                layout["address_formula"] ==
                "base_address+(output_tile_id*source_group_count+source_group_id)*row_bytes" and
                layout["bank_formula"] == "(address//row_bytes)%bank_count" and
                layout["row_buffer_baseline"] == "ordinary_same_capacity_LRU_weight_row_buffer",
                "weight layout header mismatch")
        source_groups = (layer["input_channels"] + layer["group_width"] - 1) // layer["group_width"]
        output_tiles = (layer["output_channels"] + layer["output_tile_width"] - 1) // layer["output_tile_width"]
        blocks = layout["blocks"]
        require(isinstance(blocks, list) and len(blocks) == source_groups * output_tiles,
                "weight block population mismatch")
        for index, block in enumerate(blocks):
            output_tile_id = index // source_groups
            source_group_id = index % source_groups
            address = layout["base_address"] + index * layout["row_bytes"]
            expected_key = "%d:%d:%d" % (expected_id, output_tile_id, source_group_id)
            require(block == {
                "source_group_id": source_group_id,
                "output_tile_id": output_tile_id,
                "address": address,
                "bank_key": (address // layout["row_bytes"]) % layout["bank_count"],
                "row_buffer_key": expected_key,
            }, "weight block mapping mismatch")
        require(layer["module_name"] not in mapped, "duplicate module name")
        mapped[layer["module_name"]] = layer
        seen_targets.add(layer["target"])
    require(seen_targets == TARGETS, "capture must contain FC1, FC2 and PATCH layers")
    return {layer["layer_id"]: layer for layer in layers}


def read_zlib_jsonl(path, label):
    regular(path, label)
    try:
        payload = zlib.decompress(path.read_bytes())
    except zlib.error as exc:
        raise M1544Error(label + " is not a canonical zlib stream") from exc
    require(zlib.compress(payload, 9) == path.read_bytes(), label + " is not level-9 canonical zlib")
    require(payload.endswith(b"\n"), label + " must end with newline")
    rows = []
    for index, line in enumerate(payload.splitlines()):
        require(line, label + " contains blank row")
        rows.append(strict_json_bytes(line, label + " row " + str(index)))
    return rows


def decode_codes(payload_hex, width_bits, expected):
    require(type(payload_hex) is str and len(payload_hex) % 2 == 0 and
            set(payload_hex) <= HEX, "code payload must be lower-case hex")
    payload = bytes.fromhex(payload_hex)
    width_bytes = width_bits // 8
    require(len(payload) == expected * width_bytes, "non-zero code count mismatch")
    codes = []
    for offset in range(0, len(payload), width_bytes):
        codes.append(int.from_bytes(payload[offset:offset + width_bytes], "little", signed=True))
    return codes


def bit_count(payload):
    return sum(bin(byte).count("1") for byte in payload)


def parse_bitset(value, valid_channels, label):
    require(type(value) is str and len(value) % 2 == 0 and set(value) <= HEX,
            label + " must be lower-case hex")
    payload = bytes.fromhex(value)
    require(len(payload) == (valid_channels + 7) // 8, label + " byte count mismatch")
    if valid_channels % 8:
        require(payload[-1] >> (valid_channels % 8) == 0, label + " has high padding bits")
    return payload


def validate_tokens(rows, layers, samples):
    require(rows, "token capture must be non-empty")
    sample_map = {row["global_sample_id"]: row for row in samples}
    expected_global = 0
    coverage = {}
    last_key = None
    for row in rows:
        required = {
            "schema", "global_order", "sample_global_id", "sequence",
            "sequence_sample_id", "sample_key", "operator_order", "layer_id",
            "token_order", "window_order", "spatial_y", "spatial_x", "groups",
        }
        require(isinstance(row, dict) and set(row) == required and row["schema"] == TOKEN_SCHEMA,
                "token row shape/schema mismatch")
        require(row["global_order"] == expected_global, "token global order gap/duplicate")
        expected_global += 1
        sample = sample_map.get(row["sample_global_id"])
        require(sample is not None and row["sequence"] == sample["sequence"] and
                row["sequence_sample_id"] == sample["sequence_sample_id"] and
                row["sample_key"] == sample["sample_key"], "token sample identity mismatch")
        layer = layers.get(row["layer_id"])
        require(layer is not None and row["operator_order"] == layer["operator_order"] and
                type(row["token_order"]) is int and row["token_order"] >= 0 and
                (row["window_order"] is None or
                 (type(row["window_order"]) is int and row["window_order"] >= 0)) and
                type(row["spatial_y"]) is int and row["spatial_y"] >= 0 and
                type(row["spatial_x"]) is int and row["spatial_x"] >= 0,
                "token layer/order/spatial identity mismatch")
        order_key = (row["sample_global_id"], row["operator_order"], row["token_order"])
        require(last_key is None or order_key > last_key, "token canonical order violation")
        last_key = order_key
        groups = row["groups"]
        require(isinstance(groups, list), "token groups must be a list")
        prior_group = -1
        group_count = (layer["input_channels"] + layer["group_width"] - 1) // layer["group_width"]
        for group in groups:
            require(isinstance(group, dict) and set(group) == {
                "source_group_id", "valid_channels", "support_hex", "sign_hex",
                "nonunit_hex", "nonzero_codes_le_hex"
            }, "source group shape mismatch")
            group_id = group["source_group_id"]
            require(type(group_id) is int and prior_group < group_id < group_count,
                    "source groups must be unique and increasing")
            prior_group = group_id
            expected_valid = min(layer["group_width"],
                                 layer["input_channels"] - group_id * layer["group_width"])
            require(group["valid_channels"] == expected_valid, "valid channel count mismatch")
            support = parse_bitset(group["support_hex"], expected_valid, "support")
            signs = parse_bitset(group["sign_hex"], expected_valid, "sign")
            nonunit = parse_bitset(group["nonunit_hex"], expected_valid, "non-unit")
            require(any(support), "zero source group must be omitted")
            require(all((signs[i] & ~support[i]) == 0 for i in range(len(support))),
                    "sign bit outside support")
            require(all((nonunit[i] & ~support[i]) == 0 for i in range(len(support))),
                    "non-unit bit outside support")
            count = bit_count(support)
            codes = decode_codes(group["nonzero_codes_le_hex"],
                                 layer["codebook"]["width_bits"], count)
            unit = layer["codebook"]["unit_code"]
            sign_vector = []
            nonunit_vector = []
            for code in codes:
                require(code != 0, "zero code stored in non-zero stream")
                sign_vector.append(code < 0)
                nonunit_vector.append(abs(code) != unit)
            support_positions = []
            for channel in range(expected_valid):
                if support[channel // 8] & (1 << (channel % 8)):
                    support_positions.append(channel)
            for index, channel in enumerate(support_positions):
                require(bool(signs[channel // 8] & (1 << (channel % 8))) == sign_vector[index],
                        "sign bit/code disagreement")
                require(bool(nonunit[channel // 8] & (1 << (channel % 8))) ==
                        nonunit_vector[index], "non-unit bit/code disagreement")
        key = (row["sample_global_id"], row["layer_id"])
        item = coverage.setdefault(key, {"tokens": 0, "nonempty_groups": 0})
        item["tokens"] += 1
        item["nonempty_groups"] += len(groups)
    for sample_id in sample_map:
        for layer_id in layers:
            require((sample_id, layer_id) in coverage,
                    "missing sample/layer token coverage")
    return coverage


def validate_s1(rows, layers, samples):
    eligible = {key: value for key, value in layers.items() if value["s1_eligible"]}
    require(eligible and rows, "S1 compact histogram/debt rows are required")
    sample_ids = {row["global_sample_id"] for row in samples}
    seen = set()
    last_key = None
    for row in rows:
        required = {
            "schema", "sample_global_id", "layer_id", "output_tile_id",
            "count_by_magnitude_bin", "beta_abs_code_debt_by_magnitude_bin",
            "nonzero_source_count", "beta_rounding",
        }
        require(isinstance(row, dict) and set(row) == required and row["schema"] == S1_SCHEMA,
                "S1 row shape/schema mismatch")
        layer = eligible.get(row["layer_id"])
        require(row["sample_global_id"] in sample_ids and layer is not None,
                "S1 row sample/layer mismatch")
        output_tiles = (layer["output_channels"] + layer["output_tile_width"] - 1) // layer["output_tile_width"]
        require(type(row["output_tile_id"]) is int and 0 <= row["output_tile_id"] < output_tiles,
                "S1 output tile out of range")
        bins = len(layer["s1_magnitude_bin_edges_abs_code"]) - 1
        counts = row["count_by_magnitude_bin"]
        debt = row["beta_abs_code_debt_by_magnitude_bin"]
        require(isinstance(counts, list) and len(counts) == bins and
                all(type(item) is int and item >= 0 for item in counts) and
                isinstance(debt, list) and len(debt) == bins and
                all(type(item) is int and item >= 0 for item in debt) and
                row["nonzero_source_count"] == sum(counts) and
                row["beta_rounding"] == "ceil_upper_bound",
                "S1 histogram/debt malformed")
        key = (row["sample_global_id"], row["layer_id"], row["output_tile_id"])
        require(key not in seen and (last_key is None or key > last_key),
                "S1 duplicate or canonical order violation")
        last_key = key
        seen.add(key)
    for sample_id in sample_ids:
        for layer_id, layer in eligible.items():
            tiles = (layer["output_channels"] + layer["output_tile_width"] - 1) // layer["output_tile_width"]
            for output_tile in range(tiles):
                require((sample_id, layer_id, output_tile) in seen,
                        "missing S1 sample/layer/output-tile row")
    return len(seen)


def validate_gates(value):
    expected = {
        "S1": {
            "metadata_plus_beta_over_saved_weight_bytes_veto": 0.25,
            "beta_port_cycle_regression_veto": 0.05,
            "mean_delta_aee_max": 0.02,
            "per_sequence_delta_aee_max": 0.03,
        },
        "S2": {
            "total_metadata_over_weight_bytes_max": 0.02,
            "metadata_reduction_vs_g11_min": 8.0,
            "dynamic_same_block_keep_drop_witness_required": True,
        },
        "TSBG": {
            "aggregate_fc1_fc2_cycle_speedup_min": 1.15,
            "every_sequence_cycle_speedup_min": 1.05,
            "energy_branch_cycle_regression_max": 0.05,
            "energy_branch_weight_byte_reduction_min": 0.30,
            "energy_branch_memory_energy_reduction_min": 0.20,
        },
    }
    require(value == expected, "M1541 P1 repaired gates mismatch")


def validate_manifest(value, rows, layers, s1_rows, samples, files):
    required = {
        "schema", "status", "identity", "population", "files", "encoding",
        "coverage", "admission_gates", "claim_boundary",
    }
    require(set(value) == required and value["schema"] == MANIFEST_SCHEMA and
            value["status"] == MANIFEST_STATUS, "capture manifest schema/status/shape mismatch")
    require(value["identity"] == {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "m1458_manifest_sha256": M1458_MANIFEST_SHA256,
        "m1458_inner_manifest_sha256": M1458_INNER_SHA256,
        "m1458_outer_file_sha256": M1458_OUTER_SHA256,
        "m1458_sample_order_sha256": M1458_ORDER_SHA256,
        "m1540_review_sha256": M1540_REVIEW_SHA256,
        "m1541_review_sha256": M1541_REVIEW_SHA256,
    }, "capture manifest identity mismatch")
    require(value["population"] == {
        "samples": len(samples), "layers": len(layers), "token_records": len(rows),
        "s1_histogram_rows": s1_rows,
    }, "capture population mismatch")
    expected_files = {
        "sample_order": "sample_order.json",
        "layers": "layers.json",
        "tokens": "token_source_groups.jsonl.zlib",
        "s1": "s1_histogram_debt.jsonl.zlib",
    }
    require(value["files"] == expected_files, "capture file mapping mismatch")
    require(value["encoding"] == {
        "token_container": "canonical_jsonl_zlib_level9",
        "zero_groups": "omitted_from_groups_but_token_record_retained",
        "support_sign_nonunit": "little_endian_channel_bitsets",
        "codes": "signed_little_endian_nonzero_only",
        "full_fp_tensor_saved": False,
        "static_weight_mapping_repeated_per_token": False,
    }, "capture encoding contract mismatch")
    coverage = value["coverage"]
    require(coverage.get("all_40_samples") is True and
            coverage.get("all_layers_each_sample") is True and
            coverage.get("targets") == ["FC1", "FC2", "PATCH"] and
            coverage.get("token_records") == len(rows), "capture coverage mismatch")
    validate_gates(value["admission_gates"])
    require(value["claim_boundary"] == {
        "capture_only": True, "static_opportunity": False, "cycles": False,
        "speedup": False, "traffic": False, "energy": False, "aee": False,
        "rtl": False, "paper_headline": False,
        "hardware_quantization_authority": False, "model_bit_exact": False,
        "tsbg_exact_scope": "captured_codeword_and_contributor_only",
        "formal_int8_bridge_required": True,
    }, "claim boundary mismatch")
    require(set(files) == CAPTURE_MEMBERS - {"SHA256SUMS", "SHA256SUMS.seal.sha256"},
            "sealed file ledger mismatch")


def validate_capture(root):
    root = Path(root).resolve()
    require(root.is_dir() and not root.is_symlink(), "capture root must be a directory")
    file_shas = parse_sha_manifest(root)
    require((root / "RUN_COMPLETE.txt").read_text(encoding="ascii") ==
            "M1544_EP34_SPARSE_CAPTURE_COMPLETE__NO_HARDWARE_CLAIM\n",
            "RUN_COMPLETE token mismatch")
    samples = validate_sample_order(strict_json(root / "sample_order.json"))
    layers = validate_layers(strict_json(root / "layers.json"))
    token_rows = read_zlib_jsonl(root / "token_source_groups.jsonl.zlib", "token records")
    coverage = validate_tokens(token_rows, layers, samples)
    s1_raw = read_zlib_jsonl(root / "s1_histogram_debt.jsonl.zlib", "S1 rows")
    s1_count = validate_s1(s1_raw, layers, samples)
    manifest = strict_json(root / "capture_manifest.json")
    validate_manifest(manifest, token_rows, layers, s1_count, samples, file_shas)
    return {
        "samples": len(samples), "layers": len(layers), "token_records": len(token_rows),
        "nonempty_source_groups": sum(item["nonempty_groups"] for item in coverage.values()),
        "s1_histogram_rows": s1_count, "checkpoint_sha256": CHECKPOINT_SHA256,
        "m1458_outer_file_sha256": M1458_OUTER_SHA256,
        "cycles_admitted": False, "energy_admitted": False, "aee_admitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate_capture(args.capture_dir)
    print(PASS_TOKEN + " " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
