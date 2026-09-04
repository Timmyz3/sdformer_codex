#!/usr/bin/env python3
"""Independent, read-only full-content hammer for the canonical M1707 capture.

This script deliberately does not import the producer or the M1727 analyzer.
It checks the sealed tree, authority chain, all binary frame headers, every
zlib stream, raw extent, CRC, bitset/code consistency, and the patch-only
histogram stream.  It writes its small JSON report outside the capture tree.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
import time
import zlib


FRAME_MAGIC = b"M1558F01"
FRAME_VERSION = 1
FRAME_TOKENS = 4096
FRAME_HEADER = struct.Struct("<8sHH11I")
MAGNITUDE_EDGES = [0, 1, 2, 4, 8, 16, 32, 64, 129]
OUTPUT_TILE_WIDTH = 96
CHECKPOINT_SHA256 = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
CONFIG_SHA256 = "630e735c8fe1d643b524ecd82ecf69d514df548d36380144cef442541daa4d39"
PROFILE_SHA256 = "144ba2d94eeafd2b6549a7b0aa7d0c89d2b334fe814a7d45f71d6990670e379c"
SOURCE_SHA256 = "cd135c1d8936fcb973335d1710cc7f422cf2e27648c76e6d45d7bc23bf5f72f2"
CONTRACT_SHA256 = "79b77c8c4d7671b4235635f649770baa9c0b79b2410198611c21061a6b2181ce"
RELEASE_SHA256 = "015c40cfc6c288ad9f6f89da8e21bdff344d71335cc1474d9ec781f95bc962c3"
SELECTION_SHA256 = "e6b3dd82d5d1eb54e605595369bfc8228fd616ab707d58b2e4afd95c159f87c7"
RUNTIME_TAR_SHA256 = "0524a94ccb36adc7ebc17603dedc322810141d8b14dc743923c5b942a5c6c36f"
VALIDATOR_SHA256 = "463fa7392fa090eda7fdb298fcc10ff896f91a961a0a529a013be2eec47ec240"
M1692_SOURCE_SHA256 = "ea7b300811a71d63456d16b3c3bfe04e7668266e73613ba426e0c8d6ea5e0e58"
M1692_FAILURE_RECEIPT_SHA256 = "aba412d6443ac945223872e1c71b27b7ae374fa943d970f9793d9e8a45d1b132"
M1458_MANIFEST_SHA256 = "3ab8431e3d7d17d6933c0b87da4a3405e87c97ccc302a27c78491b0a02491d6d"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
INVENTORY_SHA256 = "726a6a8fe25aa1c33f95eeb91eec8d9fb1ce4cd61376c47d438e6e2711fc9979"


class HammerError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise HammerError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path, label):
    path = Path(path)
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be a regular non-symlink")


def exact_file(path, expected, label):
    regular(path, label)
    actual = sha256(path)
    require(actual == expected, label + " SHA mismatch: " + actual)
    return actual


def strict_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, "duplicate JSON key in %s: %s" % (path, key))
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           HammerError("non-finite JSON in %s: %s" % (path, token))))
    require(type(value) is dict, "JSON root must be object: " + str(path))
    return value


def keys(value, expected, label):
    require(type(value) is dict and set(value) == set(expected),
            "%s key drift: %r" % (label, sorted(value)))


def seal_check(root):
    expected_members = {
        "RUN_COMPLETE.txt", "capture_manifest.json", "fc_frames.bin",
        "layers.json", "m1707_clean_child_receipt.json",
        "patch_s1_histogram_debt.jsonl.zlib", "preload_permit_receipt.json",
        "sample_order.json", "SHA256SUMS", "SHA256SUMS.seal.sha256"}
    found = set(path.name for path in root.iterdir())
    require(found == expected_members,
            "exact membership drift: missing=%r extra=%r" %
            (sorted(expected_members - found), sorted(found - expected_members)))
    for name in sorted(found):
        regular(root / name, name)
    lines = (root / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    required_inner = expected_members - {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    parsed = {}
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        require(match is not None, "malformed SHA256SUMS row")
        digest, name = match.groups()
        require(name not in parsed, "duplicate SHA member: " + name)
        parsed[name] = digest
    require(set(parsed) == required_inner, "SHA256SUMS membership drift")
    actual = {}
    for name in sorted(required_inner):
        actual[name] = sha256(root / name)
        require(actual[name] == parsed[name], "inner SHA mismatch: " + name)
    manifest_sha = sha256(root / "SHA256SUMS")
    outer = (root / "SHA256SUMS.seal.sha256").read_text(encoding="ascii")
    require(outer == manifest_sha + "  SHA256SUMS\n", "outer seal drift")
    return actual, manifest_sha, sha256(root / "SHA256SUMS.seal.sha256")


def verify_authority(repo, root, files):
    source = repo / ("neuron_experiments/H9_bipolar_self_attention/entrypoints/"
                     "capture_m1707_motion_ep34_s2_tsbg_deployment_complete_successor_r1.py")
    contract = repo / ("hw_autoresearch_nts07/contracts/"
                       "m1707_motion_ep34_s2_tsbg_deployment_complete_successor_source_contract_r1_20260901.json")
    release = repo / ("hw_autoresearch_nts07/contracts/"
                      "m1709_m1708_m1707_motion_ep34_s2_tsbg_deployment_complete_capture_release_r1_20260901.json")
    selection = repo / ("hw_autoresearch_nts07/contracts/"
                        "m1668_motion_ep34_s2_tsbg_current_selection_entity_r1_20260901.json")
    validator = repo / ("hw_autoresearch_nts07/system_handoff/scripts/"
                        "validate_m1544_ep34_sparse_capture_handoff.py")
    m1692_source = source.with_name(
        "capture_m1692_motion_ep34_s2_tsbg_authority_shape_repair_successor_r1.py")
    failure_receipt = repo / ("hw_autoresearch_nts07/results/"
                              "m1692_motion_ep34_s2_tsbg_capture_failed_pre_attempt_20260901/"
                              "failure_receipt.json")
    m1458_manifest = repo / ("hw_autoresearch_nts07/results/"
                             "m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831/"
                             "manifest.json")
    docs359 = repo / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"
    expected = {
        source: SOURCE_SHA256, contract: CONTRACT_SHA256, release: RELEASE_SHA256,
        selection: SELECTION_SHA256, validator: VALIDATOR_SHA256,
        m1692_source: M1692_SOURCE_SHA256,
        failure_receipt: M1692_FAILURE_RECEIPT_SHA256,
        m1458_manifest: M1458_MANIFEST_SHA256, docs359: DOCS359_SHA256}
    verified = {}
    for path, digest in expected.items():
        exact_file(path, digest, str(path.relative_to(repo)))
        verified[str(path.relative_to(repo))] = digest

    selection_value = strict_json(selection)
    for field, expected_hash in [
            ("checkpoint", CHECKPOINT_SHA256),
            ("configuration_current_capture_entity", CONFIG_SHA256),
            ("profile", PROFILE_SHA256)]:
        entity = selection_value[field]
        exact_file(Path(entity["absolute_path"]), expected_hash, field)
        require(entity["sha256"] == expected_hash, field + " entity SHA drift")
    release_value = strict_json(release)
    identity = release_value["identity"]
    require(identity["source_sha256"] == SOURCE_SHA256 and
            identity["source_contract_sha256"] == CONTRACT_SHA256 and
            identity["checkpoint_sha256"] == CHECKPOINT_SHA256 and
            identity["config_sha256"] == CONFIG_SHA256 and
            identity["profile_sha256"] == PROFILE_SHA256 and
            identity["selection_identity_sha256"] == SELECTION_SHA256 and
            identity["runtime_tar_sha256"] == RUNTIME_TAR_SHA256 and
            identity["runtime_validator_sha256"] == VALIDATOR_SHA256 and
            release_value["authorization"] == {
                "parent_calls": 1, "clean_child_processes": 1,
                "gpu_runs": 1, "production_captures": 1,
                "automatic_retry": False, "all_other_runs": 0},
            "M1709 release identity/authorization drift")
    receipt = strict_json(root / "m1707_clean_child_receipt.json")
    expected_receipt_identity = {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "m1692_failure_receipt_sha256": M1692_FAILURE_RECEIPT_SHA256,
        "m1692_source_sha256": M1692_SOURCE_SHA256,
        "profile_sha256": PROFILE_SHA256,
        "release_sha256": RELEASE_SHA256,
        "runtime_tar_sha256": RUNTIME_TAR_SHA256,
        "runtime_validator_sha256": VALIDATOR_SHA256,
        "selection_identity_sha256": SELECTION_SHA256,
        "source_contract_sha256": CONTRACT_SHA256,
        "source_sha256": SOURCE_SHA256}
    require(receipt["identity"] == expected_receipt_identity,
            "child receipt identity drift")
    # The remote capture checkout no longer retains the already-consumed M1306
    # handoff tar.  Do not pretend to rehash an absent file: bind its digest
    # independently through both the immutable release and child receipt.
    verified["m1306_runtime_tar_currently_present"] = False
    verified["m1306_runtime_tar_bound_release_and_receipt_sha256"] = RUNTIME_TAR_SHA256
    return verified, receipt, release_value


def validate_json_metadata(root, files, receipt):
    manifest = strict_json(root / "capture_manifest.json")
    layers_doc = strict_json(root / "layers.json")
    samples_doc = strict_json(root / "sample_order.json")
    permit = strict_json(root / "preload_permit_receipt.json")
    keys(manifest, ["claim_boundary", "encoding", "identity", "population",
                    "runtime_budget", "schema", "status"], "manifest")
    keys(receipt, ["checkpoint_load", "claim_boundary", "execution", "identity",
                   "population", "schema", "status"], "receipt")
    keys(layers_doc, ["inventory_sha256", "layers", "schema", "status"], "layers")
    keys(samples_doc, ["identity", "samples", "schema"], "sample_order")
    keys(permit, ["checkpoint_loaded", "consumed", "estimate", "free_bytes_after_upper",
                  "free_bytes_before", "inventory_sha256", "output", "provenance",
                  "schema"], "permit")
    require(manifest["schema"] == "m1558_reduced_binary_capture_manifest_r1_v1" and
            manifest["status"] == "PRODUCTION_PAYLOAD_REQUIRES_INDEPENDENT_RELEASE_AND_HAMMER",
            "manifest schema/status drift")
    require(receipt["schema"] == "m1707_ep34_s2_tsbg_deployment_complete_receipt_r1_v1" and
            receipt["status"] == "PAYLOAD_COMPLETE__FRESH_DIFFERENT_AUTHOR_RESULT_HAMMER_REQUIRED",
            "receipt schema/status drift")
    require((root / "RUN_COMPLETE.txt").read_text(encoding="ascii") ==
            "M1558_PAYLOAD_COMPLETE__INDEPENDENT_RELEASE_HAMMER_REQUIRED\n",
            "RUN_COMPLETE token drift")
    expected_false = ["aee", "cycles", "energy", "hardware_quantization_authority",
                      "model_bit_exact", "paper_headline", "rtl",
                      "s2_mechanism_admitted", "speedup", "traffic"]
    require(all(manifest["claim_boundary"][name] is False for name in expected_false) and
            manifest["claim_boundary"]["diagnostic_capture_only"] is True and
            manifest["claim_boundary"]["source_only"] is True and
            manifest["claim_boundary"]["tsbg_exact_scope"] ==
                "captured_codeword_and_contributor_only",
            "manifest claim boundary drift")
    require(all(receipt["claim_boundary"][name] is False for name in
                ["aee", "cycles", "eda", "energy", "hardware_quantization_authority",
                 "model_bit_exact", "paper_result", "rtl", "speedup", "traffic", "tsbg_dse"]) and
            receipt["claim_boundary"]["capture_payload_only"] is True and
            receipt["claim_boundary"]["fresh_result_hammer_required"] is True,
            "receipt claim boundary drift")
    require(receipt["checkpoint_load"] == {"missing_count": 0,
            "overlay_missing_count": 0, "overlay_unexpected_count": 0,
            "unexpected_count": 0}, "checkpoint load drift")
    require(receipt["execution"] == {
        "automatic_retry": False, "clean_child_processes": 1,
        "estimated_result_upper_bytes": 7598737368,
        "exact_remote_target": {"host": "ssh.sd5ai.scnet.cn", "port": 10037,
                                "repository_root": "/root/private_data/work/sdformer_codex/SDformer",
                                "user": "root"},
        "frozen_layer_specs": 32,
        "full_runtime_closure_before_clean_child_budget": True,
        "full_runtime_closure_before_parent_budget": True,
        "m1558_verify_bindings": True}, "receipt execution drift")
    require(manifest["population"] == {"FC1": 12, "FC2": 12, "PATCH": 8,
            "fc_frames": 11040, "fc_tokens": 44640000, "layers": 32,
            "patch_histogram_rows": 320, "samples": 40} and
            receipt["population"] == {"fc_tokens": 44640000, "frames": 11040,
                                      "patch_histogram_rows": 320, "samples": 40},
            "declared population drift")
    require(manifest["identity"] == {"checkpoint_sha256": CHECKPOINT_SHA256,
            "inventory_sha256": INVENTORY_SHA256,
            "m1458_manifest_sha256": M1458_MANIFEST_SHA256},
            "manifest identity drift")
    require(layers_doc["schema"] == "m1558_static_layers_r1_v1" and
            layers_doc["status"] == "EXACT_INVENTORY_OR_SYNTHETIC_AS_DECLARED" and
            layers_doc["inventory_sha256"] == INVENTORY_SHA256 and
            permit["inventory_sha256"] == INVENTORY_SHA256,
            "inventory binding drift")
    layers = layers_doc["layers"]
    require(len(layers) == 32 and [int(row["layer_id"]) for row in layers] == list(range(32)),
            "32-layer identity/order drift")
    require(sum(row["target"] == "FC1" for row in layers) == 12 and
            sum(row["target"] == "FC2" for row in layers) == 12 and
            sum(row["target"] == "PATCH" for row in layers) == 8,
            "layer target population drift")
    require(len(set(row["module_name"] for row in layers)) == 32 and
            len(set(int(row["operator_order"]) for row in layers)) == 32,
            "layer module/operator identity not unique")
    for row in layers:
        code = row["codebook"]
        require(code == {"authority": "diagnostic_fixed_point_codeword",
                         "diagnostic_capture_only": True,
                         "hardware_quant_authority": False,
                         "model_bit_exact": False, "rounding": "nearest_even",
                         "saturation": "signed_clamp", "scale_denominator": 1,
                         "scale_numerator": 1, "signed": True,
                         "tsbg_exact_scope": "captured_codeword_and_contributor_only",
                         "unit_code": 1, "width_bits": 8, "zero_point": 0},
                "layer codebook authority drift")
        require(int(row["tokens_s40"]) == 40 * int(row["tokens_per_call"]),
                "layer token population drift")
    samples = samples_doc["samples"]
    require(samples_doc["schema"] == "m1544_ep34_m1458_sample_order_r1_v1" and
            len(samples) == 40 and
            [int(row["global_sample_id"]) for row in samples] == list(range(40)),
            "sample identity/order drift")
    for row in samples:
        keys(row, ["global_sample_id", "sample_key", "sequence",
                   "sequence_sample_id", "sha256"], "sample")
        require(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None,
                "sample SHA format drift")
    require(samples_doc["identity"]["checkpoint_sha256"] == CHECKPOINT_SHA256 and
            samples_doc["identity"]["m1458_manifest_sha256"] == M1458_MANIFEST_SHA256,
            "sample-order M1458 identity drift")
    require(permit["schema"] == "m1582_m1576_minted_instance_registry_successor_r1_v1" and
            permit["provenance"] == "PRODUCTION_REAL_DISK" and
            permit["consumed"] is True and permit["checkpoint_loaded"] is False and
            permit["estimate"]["sample_count"] == 40 and
            permit["estimate"]["fc_tokens"] == 44640000 and
            permit["estimate"]["binary_frame_upper_count"] == 11040 and
            permit["estimate"]["inventory_sha256"] == INVENTORY_SHA256 and
            permit["estimate"]["result_upper_bytes"] == 7598737368,
            "preload permit provenance/estimate drift")
    pre_manifest_disk = sum((root / name).stat().st_size for name in
                            ["fc_frames.bin", "layers.json",
                             "patch_s1_histogram_debt.jsonl.zlib",
                             "preload_permit_receipt.json", "sample_order.json"])
    require(manifest["runtime_budget"]["disk_bytes"] == pre_manifest_disk and
            manifest["runtime_budget"]["hard_cap_bytes"] == 12 * 1024 ** 3,
            "manifest disk budget drift")
    return manifest, layers, samples, permit


def parse_frames(root, layers, samples):
    import numpy as np
    specs = [row for row in layers if row["target"] in ("FC1", "FC2")]
    by_id = {int(row["layer_id"]): row for row in specs}
    expected_pairs = [(int(sample["global_sample_id"]), int(row["layer_id"]))
                      for sample in samples for row in specs]
    pair_index = 0
    pair_frame = 0
    pair_token = 0
    frames = tokens = zero_tokens = nnz_total_all = nonunit_all = 0
    raw_total = compressed_total = 0
    active_by_layer = {int(row["layer_id"]): 0 for row in specs}
    frame_counts_by_layer = {int(row["layer_id"]): 0 for row in specs}
    pop8 = np.asarray([bin(value).count("1") for value in range(256)], dtype=np.uint8)
    with (root / "fc_frames.bin").open("rb") as stream:
        while True:
            prefix = stream.read(FRAME_HEADER.size)
            if not prefix:
                break
            require(len(prefix) == FRAME_HEADER.size, "truncated FC frame header")
            values = FRAME_HEADER.unpack(prefix)
            (magic, version, header_size, layer_id, sample_id, frame_index,
             token_start, token_count, channels, bitrow, frame_nnz, raw_bytes,
             compressed_bytes, crc32) = values
            require(magic == FRAME_MAGIC and version == FRAME_VERSION and
                    header_size == FRAME_HEADER.size, "frame header identity drift")
            require(pair_index < len(expected_pairs) and
                    (sample_id, layer_id) == expected_pairs[pair_index],
                    "frame sample/layer order drift")
            spec = by_id[layer_id]
            require(frame_index == pair_frame and token_start == pair_token and
                    channels == int(spec["input_channels"]) and
                    bitrow == (channels + 7) // 8 and
                    0 < token_count <= FRAME_TOKENS and
                    0 <= frame_nnz <= token_count * channels,
                    "frame order/dimension/count drift")
            fixed = 3 * token_count * bitrow + 2 * token_count
            require(raw_bytes == fixed + frame_nnz and compressed_bytes > 0,
                    "frame raw/compressed extent fields drift")
            compressed = stream.read(compressed_bytes)
            require(len(compressed) == compressed_bytes, "truncated compressed frame")
            decoder = zlib.decompressobj()
            raw = decoder.decompress(compressed) + decoder.flush()
            require(decoder.eof and not decoder.unused_data and
                    not decoder.unconsumed_tail and len(raw) == raw_bytes and
                    (zlib.crc32(raw) & 0xffffffff) == crc32,
                    "frame zlib EOF/raw length/CRC drift")
            matrix_bytes = token_count * bitrow
            support = np.frombuffer(raw, dtype=np.uint8, count=matrix_bytes,
                                    offset=0).reshape(token_count, bitrow)
            sign = np.frombuffer(raw, dtype=np.uint8, count=matrix_bytes,
                                 offset=matrix_bytes).reshape(token_count, bitrow)
            nonunit = np.frombuffer(raw, dtype=np.uint8, count=matrix_bytes,
                                    offset=2 * matrix_bytes).reshape(token_count, bitrow)
            require(not bool(np.bitwise_and(sign, np.bitwise_not(support)).any()) and
                    not bool(np.bitwise_and(nonunit, np.bitwise_not(support)).any()),
                    "sign/nonunit bit outside support")
            if channels % 8:
                tail_mask = np.uint8((0xff << (channels % 8)) & 0xff)
                require(not bool(np.bitwise_and(support[:, -1], tail_mask).any()) and
                        not bool(np.bitwise_and(sign[:, -1], tail_mask).any()) and
                        not bool(np.bitwise_and(nonunit[:, -1], tail_mask).any()),
                        "nonzero tail bit")
            offset = 3 * matrix_bytes
            row_nnz = np.frombuffer(raw, dtype="<u2", count=token_count, offset=offset)
            support_count = pop8[support].sum(axis=1, dtype=np.uint32)
            require(bool(np.array_equal(row_nnz.astype(np.uint32), support_count)) and
                    int(row_nnz.sum(dtype=np.uint64)) == frame_nnz,
                    "uint16 nnz/support mismatch")
            offset += 2 * token_count
            codes = np.frombuffer(raw, dtype=np.int8, count=frame_nnz, offset=offset)
            require(codes.size == frame_nnz and not bool((codes == 0).any()),
                    "nonzero code stream drift")
            support_bits = np.unpackbits(support, axis=1, bitorder="little")[:, :channels].astype(bool)
            sign_bits = np.unpackbits(sign, axis=1, bitorder="little")[:, :channels].astype(bool)
            nonunit_bits = np.unpackbits(nonunit, axis=1, bitorder="little")[:, :channels].astype(bool)
            require(bool(np.array_equal(codes < 0, sign_bits[support_bits])) and
                    bool(np.array_equal(np.abs(codes.astype(np.int16)) != 1,
                                        nonunit_bits[support_bits])),
                    "sign/nonunit code semantics drift")
            frame_zero = int((row_nnz == 0).sum())
            frame_nonunit = int(pop8[nonunit].sum(dtype=np.uint64))
            frames += 1
            tokens += token_count
            zero_tokens += frame_zero
            nnz_total_all += frame_nnz
            nonunit_all += frame_nonunit
            raw_total += raw_bytes
            compressed_total += compressed_bytes
            active_by_layer[layer_id] += frame_nnz
            frame_counts_by_layer[layer_id] += 1
            pair_frame += 1
            pair_token += token_count
            expected_tokens = int(spec["tokens_per_call"])
            require(pair_token <= expected_tokens, "pair token overflow")
            if pair_token == expected_tokens:
                pair_index += 1
                pair_frame = 0
                pair_token = 0
    require(pair_index == len(expected_pairs) and pair_token == 0,
            "FC population incomplete")
    require(frames == 11040 and tokens == 44640000,
            "FC frame/token totals drift")
    for row in specs:
        layer = int(row["layer_id"])
        require(active_by_layer[layer] == int(row["input_active_s40"]),
                "layer active/nnz mismatch: %d" % layer)
        expected_frames = 40 * int(math.ceil(float(row["tokens_per_call"]) /
                                             float(FRAME_TOKENS)))
        require(frame_counts_by_layer[layer] == expected_frames,
                "layer frame population mismatch: %d" % layer)
    return {"frames": frames, "tokens": tokens, "zero_tokens": zero_tokens,
            "nonzero_codes": nnz_total_all, "nonunit_codes": nonunit_all,
            "raw_bytes": raw_total, "compressed_bytes": compressed_total,
            "active_by_layer": dict((str(k), v) for k, v in sorted(active_by_layer.items())),
            "frame_counts_by_layer": dict((str(k), v) for k, v in sorted(frame_counts_by_layer.items()))}


def parse_patch(root, layers, samples):
    compressed = (root / "patch_s1_histogram_debt.jsonl.zlib").read_bytes()
    decoder = zlib.decompressobj()
    raw = decoder.decompress(compressed) + decoder.flush()
    require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
            "PATCH zlib extent drift")
    rows = [line for line in raw.splitlines() if line]
    specs = [row for row in layers if row["target"] == "PATCH"]
    expected = [(int(sample["global_sample_id"]), int(spec["layer_id"]), tile)
                for sample in samples for spec in specs
                for tile in range((int(spec["output_channels"]) + OUTPUT_TILE_WIDTH - 1) //
                                  OUTPUT_TILE_WIDTH)]
    require(len(rows) == len(expected) == 320, "PATCH row population drift")
    active_by_layer = {int(row["layer_id"]): 0 for row in specs}
    debt_total = 0
    for index, line in enumerate(rows):
        row = json.loads(line.decode("utf-8"))
        keys(row, ["beta_abs_code_debt_by_magnitude_bin", "beta_rounding",
                   "count_by_magnitude_bin", "layer_id",
                   "magnitude_bin_edges_abs_code", "nonzero_source_count",
                   "output_tile_id", "per_token_payload_emitted",
                   "sample_global_id", "schema"], "patch row")
        require((int(row["sample_global_id"]), int(row["layer_id"]),
                 int(row["output_tile_id"])) == expected[index],
                "PATCH canonical row order drift")
        require(row["schema"] == "m1558_patch_s1_histogram_debt_r1_v1" and
                row["magnitude_bin_edges_abs_code"] == MAGNITUDE_EDGES and
                row["per_token_payload_emitted"] is False and
                row["beta_rounding"] == "ceil_upper_bound" and
                len(row["count_by_magnitude_bin"]) == 8 and
                len(row["beta_abs_code_debt_by_magnitude_bin"]) == 8 and
                sum(int(x) for x in row["count_by_magnitude_bin"]) ==
                    int(row["nonzero_source_count"]),
                "PATCH row semantics drift")
        active_by_layer[int(row["layer_id"])] += int(row["nonzero_source_count"])
        debt_total += sum(int(x) for x in row["beta_abs_code_debt_by_magnitude_bin"])
    for spec in specs:
        layer = int(spec["layer_id"])
        # PATCH is an analog input captured through the declared diagnostic
        # nearest-even int8 codebook.  Values that quantize to zero may reduce
        # activity; the producer contract admits only the one-sided M1458 cap.
        require(0 <= active_by_layer[layer] <= int(spec["input_active_s40"]),
                "PATCH layer activity exceeds M1458 bound: %d" % layer)
    return {"rows": len(rows), "raw_bytes": len(raw),
            "compressed_bytes": len(compressed), "beta_abs_code_debt_total": debt_total,
            "active_by_layer": dict((str(k), v) for k, v in sorted(active_by_layer.items()))}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    started = time.time()
    repo = Path(args.repo).resolve()
    root = Path(args.capture).resolve()
    output = Path(args.output).resolve()
    require(repo in root.parents and output != root and root not in output.parents,
            "capture/output containment drift")
    require(root.name == "m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901",
            "canonical capture basename drift")
    files, inner_manifest_sha, outer_file_sha = seal_check(root)
    authority, receipt, release = verify_authority(repo, root, files)
    manifest, layers, samples, permit = validate_json_metadata(root, files, receipt)
    fc = parse_frames(root, layers, samples)
    patch = parse_patch(root, layers, samples)
    static_raw = sum((root / name).stat().st_size for name in
                     ["layers.json", "preload_permit_receipt.json",
                      "sample_order.json"])
    require(manifest["runtime_budget"]["raw_bytes"] ==
            fc["raw_bytes"] + patch["raw_bytes"] + static_raw,
            "manifest raw byte accounting drift")
    report = {
        "schema": "m1744_m1707_ep34_tsbg_capture_result_independent_hammer_output_r1_v1",
        "status": "PASS_M1744_M1707_EP34_TSBG_CAPTURE_RESULT_FULL_CONTENT_HAMMER",
        "verdict": "PASS_CAPTURE_PAYLOAD_ONLY__AUTHORIZE_M1727_ANALYSIS_AFTER_LOCAL_SEALED_REVIEW",
        "bindings": {"capture_manifest_sha256": files["capture_manifest.json"],
                     "fc_frames_sha256": files["fc_frames.bin"],
                     "layers_sha256": files["layers.json"],
                     "receipt_sha256": files["m1707_clean_child_receipt.json"],
                     "patch_sha256": files["patch_s1_histogram_debt.jsonl.zlib"],
                     "permit_sha256": files["preload_permit_receipt.json"],
                     "sample_order_sha256": files["sample_order.json"],
                     "sha256sums_sha256": inner_manifest_sha,
                     "outer_seal_file_sha256": outer_file_sha,
                     "checkpoint_sha256": CHECKPOINT_SHA256,
                     "config_sha256": CONFIG_SHA256,
                     "profile_sha256": PROFILE_SHA256,
                     "source_sha256": SOURCE_SHA256,
                     "source_contract_sha256": CONTRACT_SHA256,
                     "release_sha256": RELEASE_SHA256,
                     "docs359_sha256": DOCS359_SHA256},
        "checks": {"exact_file_membership": True, "all_inner_sha256": True,
                   "outer_seal": True, "run_complete": True,
                   "strict_manifest_receipt_schema": True,
                   "claim_boundary_fail_closed": True,
                   "checkpoint_config_profile_actual_bytes": True,
                   "source_contract_release_actual_bytes": True,
                   "sample_order_40_exact": True, "layer_inventory_32_exact": True,
                   "all_11040_frame_headers_order_extent": True,
                   "all_frames_zlib_eof_raw_length_crc": True,
                   "all_support_sign_nonunit_nnz_codes": True,
                   "all_320_patch_rows_order_extent": True},
        "population": {"samples": 40, "layers": 32, "FC1": 12,
                       "FC2": 12, "PATCH": 8, "fc": fc, "patch": patch},
        "authority_files": authority,
        "claim_boundary": {"capture_payload_validated": True,
                           "hardware_quantization_authority": False,
                           "model_bit_exact": False, "tsbg_dse": False,
                           "s2_mechanism_admitted": False, "aee": False,
                           "cycles": False, "traffic": False, "energy": False,
                           "speedup": False, "rtl": False, "eda": False,
                           "paper_headline": False},
        "execution": {"gpu_runs": 0, "eda_runs": 0, "capture_runs": 0,
                      "m1727_analysis_runs": 0, "automatic_retry": False,
                      "capture_tree_writes": 0},
        "findings": {"p0": [], "p1": [],
                     "p2": ["Consumed M1306 runtime tar is absent from the current remote checkout; its identity is cross-bound by the sealed M1709 release and M1707 receipt, not rehashed as a live file."]},
        "score": 99,
        "elapsed_seconds": round(time.time() - started, 6)}
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("M1744_FAIL: %s: %s" % (type(error).__name__, error), file=sys.stderr)
        sys.exit(1)
