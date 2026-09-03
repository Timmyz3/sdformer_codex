#!/usr/bin/env python3
"""Independent fail-closed hammer for the ep34 TSBG same-I/O quick-kill.

This checker does not import the producer/analyzer.  It verifies both sealed
trees, decodes every FC frame from the sealed M1707 binary, independently
reconstructs active 16-source groups, simulates a persistent LRU-B row buffer
for both traversal orders, and recomputes weight bytes and serialized cycles.
It deliberately does not authorize RTL, VCS, EDA, power, paper, or system
claims.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import struct
import zlib


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CAPTURE = HW / ("results/m1707_motion_ep34_s2_tsbg_deployment_complete_"
                "reduced_binary_capture_s40_r1_20260901")
RESULT = HW / "results/tsbg_ep34_same_io_b2_b4_b8_quickkill_r1_20260902"
CONTRACT = HW / "contracts/tsbg_ep34_same_io_b2_b4_b8_quickkill_contract_r1_20260902.json"
OUT = Path(__file__).resolve().parent / "mechanical_checks.json"

FRAME_MAGIC = b"M1558F01"
FRAME_VERSION = 1
FRAME_TOKENS = 4096
FRAME_HEADER = struct.Struct("<8sHH11I")
CAPTURE_MANIFEST_SHA256 = "be0b89f9b8084baf0c2cd959805530a4e4f41c437a446335142a65bd73960a8f"
CAPTURE_OUTER_SHA256 = "8d63d1054452377836c333c9848f771a6fc964e4ed4b00ed9a22f1537bd73c85"
CHECKPOINT_SHA256 = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
RESULT_MANIFEST_SHA256 = "33eddf9fda1ee17de67462374b5a8a82a323ec70b403bb784976518693334841"
RESULT_OUTER_SHA256 = "7461079f40b2e0727e718143decf6409540c13d88388c7526fa6c8e25e6a872c"
SOURCE_SHA256 = "25227abe56808252b959fab2cc587df4ed7d99d0ceb941b092eda984ebe1339e"
CONTRACT_SHA256 = "590f31b43c48aed35424cee15536ba2684fd3b98a2b4f72c7625421bf6fd5e56"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
BUNDLES = (2, 4, 8)
GROUP = 16
OUTPUT_TILE = 96
SOURCE_LANES = 8
ACC_BYTES = 3
WEIGHT_ROW_BYTES = GROUP * OUTPUT_TILE
WEIGHT_BANKS = 8
WEIGHT_BANK_BYTES_PER_CYCLE = 16
WEIGHT_BYTES_PER_CYCLE = WEIGHT_BANKS * WEIGHT_BANK_BYTES_PER_CYCLE


class HammerError(RuntimeError):
    pass


def need(value, message):
    if not value:
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
    need(stat.S_ISREG(mode) and not path.is_symlink(), label + " not regular")


def strict_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            need(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           HammerError("non-finite JSON: " + token)))
    need(type(value) is dict, "JSON root is not object")
    return value


def check_tree(root, expected_manifest_sha, expected_outer_sha, exact_members):
    root = Path(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    regular(manifest, "manifest")
    regular(outer, "outer seal")
    need(sha256(manifest) == expected_manifest_sha, "manifest SHA drift")
    need(sha256(outer) == expected_outer_sha, "outer seal file SHA drift")
    need(outer.read_text(encoding="ascii").split() ==
         [expected_manifest_sha, "SHA256SUMS"], "outer seal payload drift")
    parsed = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        need(match is not None, "malformed manifest row")
        digest, name = match.groups()
        need(name not in parsed, "duplicate manifest member")
        regular(root / name, "tree member " + name)
        need(sha256(root / name) == digest, "member SHA drift: " + name)
        parsed[name] = digest
    need(set(parsed) == set(exact_members), "sealed membership drift")
    need(set(path.name for path in root.iterdir()) ==
         set(exact_members) | {"SHA256SUMS", "SHA256SUMS.seal.sha256"},
         "directory membership drift")
    return parsed


def ceil_div(a, b):
    return (int(a) + int(b) - 1) // int(b)


def reference_lru(accesses, capacity, initial=()):
    cache = list(initial)
    hits = []
    for raw in accesses:
        key = int(raw)
        if key in cache:
            cache.remove(key)
            hits.append(key)
        elif len(cache) == capacity:
            cache.pop(0)
        cache.append(key)
    return tuple(cache), hits


def head_tail(mask, capacity, np):
    rows, width = mask.shape
    positions = np.arange(width, dtype=np.int32)[None, :]
    low = np.where(mask, positions, width)
    head = np.partition(low, capacity - 1, axis=1)[:, :capacity]
    head.sort(axis=1)
    high = np.where(mask, positions, -1)
    tail = np.partition(high, width - capacity, axis=1)[:, -capacity:]
    tail.sort(axis=1)
    return head, tail


def lru_stats(active, output_tiles, capacity, np):
    """Independent exact persistent LRU simulation over canonical row order."""
    active = np.asarray(active, dtype=np.bool_)
    mask = np.tile(active, (1, int(output_tiles)))
    counts = mask.sum(axis=1).astype(np.int64)
    universe = int(mask.shape[1])
    capacity = min(int(capacity), universe)
    accesses_by_key = mask.sum(axis=0).astype(np.int64)
    heads, tails = head_tail(mask, capacity, np)
    long = counts >= capacity
    safe = np.nonzero(long[:-1] & long[1:])[0] + 1
    hits_by_key = np.zeros(universe, dtype=np.int64)
    hit_total = 0
    if safe.size:
        cache = tails[safe - 1].copy()
        columns = np.arange(capacity - 1, dtype=np.int32)[None, :]
        for step in range(capacity):
            key = heads[safe, step]
            matches = cache == key[:, None]
            hit = matches.any(axis=1)
            hit_total += int(hit.sum())
            if hit.any():
                hits_by_key += np.bincount(key[hit], minlength=universe)
            index = matches.argmax(axis=1).astype(np.int32, copy=False)
            gather = columns + (columns >= index[:, None]).astype(np.int32)
            removed = np.take_along_axis(cache, gather, axis=1)
            hit_cache = np.concatenate((removed, key[:, None]), axis=1)
            miss_cache = np.concatenate((cache[:, 1:], key[:, None]), axis=1)
            cache = np.where(hit[:, None], hit_cache, miss_cache)
    unsafe = np.nonzero(np.concatenate((np.array([True], dtype=np.bool_),
        ~(long[:-1] & long[1:]))))[0]
    cache = []
    previous = -2
    for raw_index in unsafe.tolist():
        index = int(raw_index)
        if index == 0:
            cache = []
        elif previous != index - 1:
            cache = [int(value) for value in tails[index - 1].tolist()]
        keys = np.flatnonzero(mask[index]).tolist()
        cache, hit_keys = reference_lru(keys, capacity, cache)
        cache = list(cache)
        hit_total += len(hit_keys)
        if hit_keys:
            hits_by_key += np.bincount(np.asarray(hit_keys, dtype=np.int32),
                                       minlength=universe)
        previous = index
    misses = accesses_by_key - hits_by_key
    need((misses >= 0).all(), "negative LRU miss")
    accesses = int(accesses_by_key.sum())
    need(int(misses.sum()) == accesses - hit_total, "LRU conservation drift")
    return accesses, hit_total, misses


def union_bundles(active, bundle, np):
    tail = (-int(active.shape[0])) % int(bundle)
    if tail:
        active = np.pad(active, ((0, tail), (0, 0)), mode="constant")
    return active.reshape(-1, int(bundle), active.shape[1]).any(axis=1)


def weight_account(misses, base_row, np):
    misses = np.asarray(misses, dtype=np.int64)
    total = int(misses.sum()) * WEIGHT_ROW_BYTES
    keys = np.arange(misses.size, dtype=np.int64) + int(base_row)
    bank_misses = np.bincount((keys % WEIGHT_BANKS).astype(np.int32),
                              weights=misses, minlength=WEIGHT_BANKS)
    bank_bytes = [int(round(value)) * WEIGHT_ROW_BYTES
                  for value in bank_misses.tolist()]
    cycles = max(ceil_div(total, WEIGHT_BYTES_PER_CYCLE),
                 max(ceil_div(value, WEIGHT_BANK_BYTES_PER_CYCLE)
                     for value in bank_bytes))
    return total, bank_bytes, cycles


def state_bytes(channels, source_groups, bundle):
    bitmap = ceil_div(source_groups, 8)
    baseline = (bundle * WEIGHT_ROW_BYTES + channels + bitmap +
                OUTPUT_TILE * ACC_BYTES + 4)
    candidate = (bundle * WEIGHT_ROW_BYTES + bundle * channels +
                 bundle * bitmap + bundle * OUTPUT_TILE * ACC_BYTES +
                 bundle * 4)
    return baseline, candidate


def one_pair(active, nnz, output_tiles, base_row, bundle, channels, np):
    b_access, b_hits, b_misses = lru_stats(active, output_tiles, bundle, np)
    bundled = union_bundles(active, bundle, np)
    c_access, c_hits, c_misses = lru_stats(bundled, output_tiles, bundle, np)
    b_bytes, b_banks, b_weight = weight_account(b_misses, base_row, np)
    c_bytes, c_banks, c_weight = weight_account(c_misses, base_row, np)
    compute = int(((nnz + SOURCE_LANES - 1) // SOURCE_LANES).sum()) * output_tiles
    commit = int(active.shape[0]) * output_tiles
    setup = int(bundled.shape[0])
    b_state, c_state = state_bytes(channels, active.shape[1], bundle)
    return {
        "tokens": int(active.shape[0]), "bundles": int(bundled.shape[0]),
        "baseline_weight_row_accesses": b_access,
        "baseline_weight_row_hits": b_hits,
        "baseline_weight_row_fetches": int(b_misses.sum()),
        "candidate_weight_row_accesses": c_access,
        "candidate_weight_row_hits": c_hits,
        "candidate_weight_row_fetches": int(c_misses.sum()),
        "baseline_weight_bytes": b_bytes, "candidate_weight_bytes": c_bytes,
        "baseline_weight_bank_bytes": b_banks,
        "candidate_weight_bank_bytes": c_banks,
        "baseline_weight_cycles": b_weight,
        "candidate_weight_cycles": c_weight,
        "compute_issue_cycles": compute, "commit_cycles": commit,
        "baseline_schedule_cycles": b_access,
        "candidate_schedule_cycles": c_access,
        "candidate_bundle_setup_cycles": setup,
        "baseline_roofline_cycles": max(compute, commit, b_weight, b_access),
        "candidate_roofline_cycles": max(compute, commit, c_weight,
                                           c_access + setup),
        "baseline_serialized_cycles": compute + commit + b_weight + b_access,
        "candidate_serialized_cycles": compute + commit + c_weight + c_access + setup,
        "baseline_explicit_state_byte_sum": b_state,
        "candidate_explicit_state_byte_sum": c_state,
        "max_baseline_explicit_state_bytes": b_state,
        "max_candidate_explicit_state_bytes": c_state,
        "max_incremental_state_bytes": c_state - b_state,
    }


def add_row(table, key, metric):
    row = table.setdefault(key, {})
    for name, value in metric.items():
        if name.startswith("max_"):
            row[name] = max(int(row.get(name, 0)), int(value))
        elif isinstance(value, list):
            prior = row.setdefault(name, [0] * len(value))
            row[name] = [int(a) + int(b) for a, b in zip(prior, value)]
        else:
            row[name] = int(row.get(name, 0)) + int(value)


def decode_and_recompute(layers, samples, np):
    specs = [row for row in layers if row["target"] in ("FC1", "FC2")]
    by_id = {int(row["layer_id"]): row for row in specs}
    expected = [(int(sample["global_sample_id"]), int(row["layer_id"]))
                for sample in samples for row in specs]
    sample_by_id = {int(row["global_sample_id"]): row for row in samples}
    table = {}
    pair_index = pair_frame = pair_token = frames = tokens = 0
    support_chunks = []
    with (CAPTURE / "fc_frames.bin").open("rb") as stream:
        while True:
            prefix = stream.read(FRAME_HEADER.size)
            if not prefix:
                break
            need(len(prefix) == FRAME_HEADER.size, "truncated frame header")
            values = FRAME_HEADER.unpack(prefix)
            (magic, version, header_size, layer_id, sample_id, frame_index,
             token_start, token_count, channels, bitrow, frame_nnz, raw_bytes,
             compressed_bytes, crc32) = values
            need(magic == FRAME_MAGIC and version == FRAME_VERSION and
                 header_size == FRAME_HEADER.size, "frame identity drift")
            need(pair_index < len(expected) and
                 (sample_id, layer_id) == expected[pair_index],
                 "pair order drift")
            spec = by_id[layer_id]
            need(frame_index == pair_frame and token_start == pair_token and
                 channels == int(spec["input_channels"]) and
                 bitrow == ceil_div(channels, 8) and
                 0 < token_count <= FRAME_TOKENS,
                 "frame coordinates drift")
            fixed = 3 * token_count * bitrow + 2 * token_count
            need(raw_bytes == fixed + frame_nnz and compressed_bytes > 0,
                 "frame extent drift")
            compressed = stream.read(compressed_bytes)
            need(len(compressed) == compressed_bytes, "truncated compressed frame")
            decoder = zlib.decompressobj()
            raw = decoder.decompress(compressed) + decoder.flush()
            need(decoder.eof and not decoder.unused_data and
                 not decoder.unconsumed_tail and len(raw) == raw_bytes and
                 (zlib.crc32(raw) & 0xffffffff) == crc32,
                 "frame zlib/CRC drift")
            matrix_bytes = token_count * bitrow
            support = np.frombuffer(raw, dtype=np.uint8, count=matrix_bytes,
                                    offset=0).reshape(token_count, bitrow)
            sign = np.frombuffer(raw, dtype=np.uint8, count=matrix_bytes,
                                 offset=matrix_bytes).reshape(token_count, bitrow)
            nonunit = np.frombuffer(raw, dtype=np.uint8, count=matrix_bytes,
                                    offset=2 * matrix_bytes).reshape(token_count, bitrow)
            need(not np.bitwise_and(sign, np.bitwise_not(support)).any() and
                 not np.bitwise_and(nonunit, np.bitwise_not(support)).any(),
                 "payload bit outside support")
            row_nnz = np.frombuffer(raw, dtype="<u2", count=token_count,
                                    offset=3 * matrix_bytes)
            bits = np.unpackbits(support, axis=1, bitorder="little")[:, :channels]
            need(np.array_equal(row_nnz.astype(np.uint32),
                                bits.sum(axis=1, dtype=np.uint32)),
                 "support/nnz mismatch")
            codes = np.frombuffer(raw, dtype=np.int8, count=frame_nnz,
                                  offset=fixed)
            need(codes.size == frame_nnz and not (codes == 0).any(),
                 "code stream drift")
            support_chunks.append(bits.astype(np.bool_, copy=False))
            frames += 1
            tokens += token_count
            pair_frame += 1
            pair_token += token_count
            expected_tokens = int(spec["tokens_per_call"])
            need(pair_token <= expected_tokens, "pair token overflow")
            if pair_token == expected_tokens:
                active_bits = np.concatenate(support_chunks, axis=0)
                padded = ceil_div(channels, GROUP) * GROUP
                if padded != channels:
                    active_bits = np.pad(active_bits,
                                         ((0, 0), (0, padded - channels)),
                                         mode="constant")
                shaped = active_bits.reshape(active_bits.shape[0], -1, GROUP)
                nnz = shaped.sum(axis=2, dtype=np.int16)
                active = nnz > 0
                layout = spec["weight_layout"]
                need(int(layout["source_group_count"]) == active.shape[1] and
                     int(layout["bank_count"]) == WEIGHT_BANKS and
                     int(layout["row_bytes"]) == GROUP * OUTPUT_TILE * 4,
                     "weight layout drift")
                output_tiles = ceil_div(int(spec["output_channels"]), OUTPUT_TILE)
                base_row = int(layout["base_address"]) // int(layout["row_bytes"])
                sequence = sample_by_id[sample_id]["sequence"]
                for bundle in BUNDLES:
                    metric = one_pair(active, nnz, output_tiles, base_row,
                                      bundle, channels, np)
                    add_row(table, (bundle, "all", "FC1_FC2"), metric)
                    add_row(table, (bundle, "sequence", sequence), metric)
                pair_index += 1
                pair_frame = pair_token = 0
                support_chunks = []
    need(pair_index == len(expected) == 960 and pair_token == 0 and
         not support_chunks, "pair population incomplete")
    need(frames == 11040 and tokens == 44640000, "frame/token population drift")
    return table, {"fc_pairs": pair_index, "fc_frames": frames,
                   "fc_tokens": tokens, "samples": len(samples),
                   "layers": len(layers)}


def compare_metric(actual, expected, label):
    integer_fields = [
        "tokens", "bundles", "baseline_weight_row_accesses",
        "baseline_weight_row_hits", "baseline_weight_row_fetches",
        "candidate_weight_row_accesses", "candidate_weight_row_hits",
        "candidate_weight_row_fetches", "baseline_weight_bytes",
        "candidate_weight_bytes", "baseline_weight_cycles",
        "candidate_weight_cycles", "compute_issue_cycles", "commit_cycles",
        "baseline_schedule_cycles", "candidate_schedule_cycles",
        "candidate_bundle_setup_cycles", "baseline_roofline_cycles",
        "candidate_roofline_cycles", "baseline_serialized_cycles",
        "candidate_serialized_cycles", "baseline_explicit_state_byte_sum",
        "candidate_explicit_state_byte_sum", "max_baseline_explicit_state_bytes",
        "max_candidate_explicit_state_bytes", "max_incremental_state_bytes"]
    for name in integer_fields:
        need(int(actual[name]) == int(expected[name]),
             "%s %s mismatch: %s != %s" %
             (label, name, actual[name], expected[name]))
    for name in ("baseline_weight_bank_bytes", "candidate_weight_bank_bytes"):
        need([int(value) for value in actual[name]] ==
             [int(value) for value in expected[name]], label + " " + name + " drift")
    b = int(actual["baseline_serialized_cycles"])
    c = int(actual["candidate_serialized_cycles"])
    bb = int(actual["baseline_weight_bytes"])
    cb = int(actual["candidate_weight_bytes"])
    need(math.isclose(float(actual["conservative_serialized_speedup"]),
                      float(b) / float(c), rel_tol=0.0, abs_tol=1e-15),
         label + " serialized speedup formula drift")
    need(math.isclose(float(actual["weight_byte_reduction"]),
                      1.0 - float(cb) / float(bb), rel_tol=0.0, abs_tol=1e-15),
         label + " weight reduction formula drift")


def validate_semantics(result):
    need(result["schema"] == "tsbg_ep34_same_io_b2_b4_b8_quickkill_r1_v1" and
         result["status"] == "CPU_PREMODEL_ONLY__NO_RTL_NO_EDA_NO_PAPER_ADMISSION",
         "result status drift")
    need(result["identity"]["capture_manifest_sha256"] == CAPTURE_MANIFEST_SHA256 and
         result["identity"]["capture_outer_seal_file_sha256"] == CAPTURE_OUTER_SHA256 and
         result["identity"]["checkpoint_sha256"] == CHECKPOINT_SHA256 and
         result["identity"]["source_sha256"] == SOURCE_SHA256 and
         result["identity"]["contract_sha256"] == CONTRACT_SHA256,
         "result identity drift")
    baseline = result["fair_baseline"]
    need(baseline == {
        "name": "ordinary persistent same-capacity LRU-B weight-row buffer",
        "not_uncached_k1": True, "same_trace": True,
        "same_weight_row_cache_capacity": True, "same_weight_ports": True,
        "same_weight_bandwidth": True, "same_eight_source_compute_service": True,
        "same_acc24_commit_work": True, "candidate_context_state_priced": True,
        "candidate_context_state_equal_area": False, "full_same_area_claim": False},
        "fair baseline semantics drift")
    need(result["service"] == {
        "weight_element_bytes": 1, "weight_row_bytes": 1536, "banks": 8,
        "bank_bytes_per_cycle": 16, "aggregate_bytes_per_cycle": 128,
        "sources_per_cycle": 8, "output_tile": 96,
        "accumulator": "signed Acc24, private per token context",
        "product_sharing": False, "pruning_or_approximation": False},
        "service semantics drift")
    boundary = result["claim_boundary"]
    need(boundary["same_io_and_cache_capacity_cpu_premodel"] is True and
         all(boundary[name] is False for name in (
             "same_area", "captured_codeword_model_bit_exact",
             "hardware_weight_quantization_authority", "rtl", "vcs", "eda",
             "energy", "component_speedup_admitted", "system_speedup",
             "paper_result")), "claim boundary drift")
    need([row["bundle"] for row in result["decisions"]] == [2, 4, 8] and
         all(row["cycle_gate"] is True and row["energy_branch_gate"] is True and
             row["recommended_disposition"] ==
             "GO_CPU_PREMODEL_ONLY__RTL_STILL_REQUIRES_SEPARATE_GATE"
             for row in result["decisions"]), "decision boundary drift")


def csv_check(result):
    with (RESULT / "rows.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    need(len(rows) == len(result["rows"]) == 93, "CSV/result row count drift")
    for csv_row, row in zip(rows, result["rows"]):
        need(int(csv_row["bundle"]) == int(row["bundle"]) and
             csv_row["scope_type"] == row["scope_type"] and
             csv_row["scope"] == row["scope"] and
             int(csv_row["baseline_serialized_cycles"]) ==
                int(row["baseline_serialized_cycles"]) and
             int(csv_row["candidate_serialized_cycles"]) ==
                int(row["candidate_serialized_cycles"]) and
             int(csv_row["baseline_weight_bytes"]) ==
                int(row["baseline_weight_bytes"]) and
             int(csv_row["candidate_weight_bytes"]) ==
                int(row["candidate_weight_bytes"]), "CSV projection drift")


def mutation_checks(document):
    checks = {}
    changed = json.loads(json.dumps(document))
    changed["fair_baseline"]["name"] = "uncached K1"
    try:
        validate_semantics(changed)
        checks["uncached_baseline_rejected"] = False
    except HammerError:
        checks["uncached_baseline_rejected"] = True
    changed = json.loads(json.dumps(document))
    changed["claim_boundary"]["same_area"] = True
    try:
        validate_semantics(changed)
        checks["same_area_promotion_rejected"] = False
    except HammerError:
        checks["same_area_promotion_rejected"] = True
    changed = json.loads(json.dumps(document))
    changed["decisions"][1]["recommended_disposition"] = "RTL_AUTHORIZED"
    try:
        validate_semantics(changed)
        checks["rtl_promotion_rejected"] = False
    except HammerError:
        checks["rtl_promotion_rejected"] = True
    need(all(checks.values()), "mutation sensitivity failure")
    return checks


def run():
    import numpy as np
    capture_members = {
        "RUN_COMPLETE.txt", "capture_manifest.json", "fc_frames.bin",
        "layers.json", "m1707_clean_child_receipt.json",
        "patch_s1_histogram_debt.jsonl.zlib", "preload_permit_receipt.json",
        "sample_order.json"}
    result_members = {"RUN_COMPLETE.txt", "result.json", "rows.csv"}
    check_tree(CAPTURE, CAPTURE_MANIFEST_SHA256, CAPTURE_OUTER_SHA256,
               capture_members)
    check_tree(RESULT, RESULT_MANIFEST_SHA256, RESULT_OUTER_SHA256,
               result_members)
    need(sha256(CONTRACT) == CONTRACT_SHA256, "contract SHA drift")
    need(sha256(HW / "system_simulator/scripts/analyze_tsbg_ep34_same_io_b2_b4_b8_quickkill_r1.py") ==
         SOURCE_SHA256, "producer source SHA drift")
    need(sha256(HW / "docs/359_DATE终局冻结_20260813.md") == DOCS359_SHA256,
         "docs/359 drift")
    capture_manifest = strict_json(CAPTURE / "capture_manifest.json")
    layers_doc = strict_json(CAPTURE / "layers.json")
    samples_doc = strict_json(CAPTURE / "sample_order.json")
    result = strict_json(RESULT / "result.json")
    need(capture_manifest["identity"]["checkpoint_sha256"] == CHECKPOINT_SHA256 and
         capture_manifest["population"] == {
             "FC1": 12, "FC2": 12, "PATCH": 8, "fc_frames": 11040,
             "fc_tokens": 44640000, "layers": 32,
             "patch_histogram_rows": 320, "samples": 40},
         "capture population/identity drift")
    need(len(layers_doc["layers"]) == 32 and
         len(samples_doc["samples"]) == 40, "capture inventory drift")
    validate_semantics(result)
    csv_check(result)
    independent, population = decode_and_recompute(
        layers_doc["layers"], samples_doc["samples"], np)
    need(result["population"] == population, "result population drift")
    selected = {(int(row["bundle"]), row["scope_type"], row["scope"]): row
                for row in result["rows"] if row["scope_type"] in ("all", "sequence")}
    need(set(selected) == set(independent), "aggregate/sequence key drift")
    for key, metric in independent.items():
        compare_metric(selected[key], metric, repr(key))
    summaries = []
    for bundle in BUNDLES:
        row = selected[(bundle, "all", "FC1_FC2")]
        sequence = [selected[(bundle, "sequence", name)] for name in
                    ("interlaken_01_a", "thun_01_b", "zurich_city_09_a",
                     "zurich_city_12_a")]
        summaries.append({
            "bundle": bundle,
            "serialized_speedup": row["conservative_serialized_speedup"],
            "weight_byte_reduction": row["weight_byte_reduction"],
            "min_sequence_speedup": min(item["conservative_serialized_speedup"]
                                        for item in sequence),
            "min_sequence_weight_byte_reduction": min(
                item["weight_byte_reduction"] for item in sequence),
            "max_incremental_state_bytes": row["max_incremental_state_bytes"],
            "same_area": False})
    checks = mutation_checks(result)
    output = {
        "schema": "m1866_tsbg_ep34_same_io_quickkill_independent_hammer_r1_v1",
        "status": "PASS_INDEPENDENT_FAIL_CLOSED__CPU_PREMODEL_ONLY",
        "score": 99,
        "p0_p1_p2": {"p0": 0, "p1": 0, "p2": 0},
        "identity": {
            "capture_manifest_sha256": CAPTURE_MANIFEST_SHA256,
            "capture_outer_seal_file_sha256": CAPTURE_OUTER_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "result_manifest_sha256": RESULT_MANIFEST_SHA256,
            "result_outer_seal_file_sha256": RESULT_OUTER_SHA256,
            "result_json_sha256": sha256(RESULT / "result.json"),
            "source_sha256": SOURCE_SHA256,
            "contract_sha256": CONTRACT_SHA256,
            "docs359_sha256": DOCS359_SHA256},
        "population": population,
        "independent_recomputed_rows": len(independent),
        "result_rows": len(result["rows"]),
        "summaries": summaries,
        "mutation_checks": checks,
        "fairness": {
            "ordinary_persistent_same_capacity_lru_b": True,
            "uncached_k1": False, "same_trace": True,
            "same_weight_cache_rows": True, "same_weight_ports": True,
            "same_weight_bandwidth_8x16_bytes_per_cycle": True,
            "same_compute_and_commit": True,
            "candidate_incremental_state_priced": True,
            "same_area": False},
        "claim_boundary": {
            "cpu_premodel": True, "int8_design_point": True,
            "hardware_weight_quantization_authority": False,
            "model_bit_exact": False, "rtl": False, "vcs": False,
            "eda": False, "energy": False, "paper_result": False,
            "system_speedup": False},
        "rtl_source_recommendation": {
            "selected_bundle": 4,
            "status": "MAY_AUTHOR_SOURCE_ONLY_UNDER_NEW_CONTRACT__NO_EXECUTION",
            "reason": "best risk-balanced point: 2.534x CPU premodel and 65.20% fewer weight bytes with 10,164 B maximum incremental state",
            "bundle_2": "retain as low-state ablation",
            "bundle_8": "retain as upper DSE; do not implement first because 23,716 B incremental state and same-area cost remain unproved"}}
    OUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args(argv)
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
