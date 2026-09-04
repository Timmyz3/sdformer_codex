#!/usr/bin/env python3
"""M712 exact-workload CPU fast-kill for decoder PIDP.

This is deliberately an optimistic candidate-bound ledger.  It is useful for a
safe KILL: if PIDP loses even after free descriptor/psum removal, conflict-free
K8 packing and a fully associative logical weight cache, RTL cannot rescue the
specific destination-owner schedule without changing the frozen resources.
"""

import argparse
from collections import OrderedDict, defaultdict
from decimal import Decimal, getcontext
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import tempfile

import numpy as np


getcontext().prec = 40
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
CONTRACT_SCHEMA = "m712_pidp_decoder_exact_cpu_fastkill_contract_v1"
RESULT_SCHEMA = "m712_pidp_decoder_exact_cpu_fastkill_result_v1"
MODULES = {
    0: ("D0", 1536, 384, 15, 20, 30, 40, 4),
    1: ("D1", 770, 192, 30, 40, 60, 80, 2),
    2: ("D2", 386, 96, 60, 80, 120, 160, 1),
    3: ("D3", 194, 96, 120, 160, 240, 320, 1),
}
HEADLINE = {0, 2, 3}
WEIGHT_TILE_BYTES = 16 * 96 * 3 * 3
WEIGHT_REFILL_CYCLES = 32 + WEIGHT_TILE_BYTES // 128
OUTPUT_COMMIT_CYCLES = 6 + 32 + 3
LOGICAL_BUDGET_BYTES = 240 * 1024
CONTROL_BYTES = 8192
DESCRIPTOR_BYTES = 16
PSUM_BYTES_PER_DESTINATION_GROUP = 6 * 2 * 48


class Failure(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise Failure(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_member(name):
    p = PurePosixPath(name)
    require(p.parts and not p.is_absolute() and ".." not in p.parts and
            p.as_posix() == name, "unsafe sealed member: " + name)
    return p


def strict_json(path):
    def pairs(rows):
        out = {}
        for key, value in rows:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("non-finite JSON token: " + token)))


def verify_directory(path):
    path = Path(path)
    require(path.is_dir() and not path.is_symlink(), "bad sealed directory")
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and outer.is_file(), "missing directory seals")
    expected_names = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and len(fields[0]) == 64,
                "malformed SHA256SUMS")
        expected, name = fields
        require(name not in expected_names, "duplicate sealed member")
        expected_names.add(name)
        member = path.joinpath(*safe_member(name).parts)
        require(member.is_file() and not member.is_symlink() and
                sha256(member) == expected, "member hash mismatch: " + name)
    actual_names = set()
    root_seals = {manifest.resolve(), outer.resolve()}
    for member in path.rglob("*"):
        require(not member.is_symlink(), "symlink in sealed directory")
        if member.is_file() and member.resolve() not in root_seals:
            actual_names.add(member.relative_to(path).as_posix())
    require(actual_names == expected_names, "sealed member set mismatch")
    fields = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    require(fields == [sha256(manifest), "SHA256SUMS"], "outer seal mismatch")
    return {"manifest_sha256": sha256(manifest),
            "outer_file_sha256": sha256(outer)}


def write_seal(path):
    path = Path(path)
    members = sorted(p for p in path.rglob("*") if p.is_file() and
                     p.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    manifest = path / "SHA256SUMS"
    manifest.write_text("".join("{}  {}\n".format(
        sha256(p), p.relative_to(path).as_posix()) for p in members), encoding="utf-8")
    (path / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(manifest)), encoding="utf-8")


def ratio(numerator, denominator):
    require(denominator > 0, "zero denominator")
    return format(Decimal(int(numerator)) / Decimal(int(denominator)), ".12f")


def wrap24(value):
    value = int(value) & ((1 << 24) - 1)
    return value - (1 << 24) if value & (1 << 23) else value


def topology(cin, hin, win, hout, wout):
    del cin
    rows = []
    aggregate = hashlib.sha256()
    legal_spatial_edges = 0
    mismatch = 0
    for ky in range(3):
        for kx in range(3):
            forward = []
            for sy in range(hin):
                oy = 2 * sy - 1 + ky
                if not 0 <= oy < hout:
                    continue
                for sx in range(win):
                    ox = 2 * sx - 1 + kx
                    if 0 <= ox < wout:
                        forward.append((oy, ox, sy, sx))
            forward.sort()
            pull = []
            for oy in range(hout):
                numerator_y = oy + 1 - ky
                if numerator_y & 1:
                    continue
                sy = numerator_y // 2
                if not 0 <= sy < hin:
                    continue
                for ox in range(wout):
                    numerator_x = ox + 1 - kx
                    if numerator_x & 1:
                        continue
                    sx = numerator_x // 2
                    if 0 <= sx < win:
                        pull.append((oy, ox, sy, sx))
            mismatch += int(forward != pull)
            payload = json.dumps([ky, kx, forward], separators=(",", ":")).encode()
            digest = hashlib.sha256(payload).hexdigest()
            aggregate.update(bytes.fromhex(digest))
            rows.append({"ky": ky, "kx": kx, "legal_spatial_edges": len(forward),
                         "forward_pull_mismatch": int(forward != pull),
                         "canonical_map_sha256": digest})
            legal_spatial_edges += len(forward)
    return {"tap_rows": rows, "legal_spatial_edges": legal_spatial_edges,
            "topology_mismatches": mismatch,
            "aggregate_sha256": aggregate.hexdigest()}


def unpack_record(path, shape):
    expected = 1
    for extent in shape:
        expected *= int(extent)
    payload = np.fromfile(str(path), dtype=np.uint8)
    bits = np.unpackbits(payload, bitorder="little")[:expected]
    require(bits.size == expected, "bitpack length mismatch")
    return bits.reshape(shape).astype(np.uint8, copy=False)


def quantized_weights(weight_path, cin, cout):
    raw = np.fromfile(str(weight_path), dtype="<f4")
    require(raw.size == cin * cout * 9, "weight payload size mismatch")
    weight = raw.reshape(cin, cout, 3, 3).astype(np.float64)
    maxabs = np.max(np.abs(weight), axis=(0, 2, 3))
    scale = np.where(maxabs == 0.0, 1.0, maxabs / 127.0)
    quant = np.clip(np.rint(weight / scale[None, :, None, None]), -127, 127)
    return quant.astype(np.int8), {
        "policy": "local_probe_only_symmetric_per_output_channel_rint_clip127",
        "scale_sha256": hashlib.sha256(scale.astype("<f8").tobytes()).hexdigest(),
        "int8_sha256": hashlib.sha256(quant.astype(np.int8).tobytes()).hexdigest(),
        "not_checkpoint_numeric_admission": True,
    }


def descriptor_counts(bits, blocks):
    cin, hin, win = bits.shape
    ntile = (cin + 15) // 16
    padded = np.zeros((ntile * 16, hin, win), dtype=np.uint8)
    padded[:cin] = bits
    pop_tile = padded.reshape(ntile, 16, hin, win).sum(axis=1, dtype=np.uint8)
    hout, wout = 2 * hin, 2 * win
    counts = np.zeros((ntile, hout, wout), dtype=np.uint8)
    tiles = np.arange(ntile)
    for ky in range(3):
        sy = np.arange(hin)
        oy = 2 * sy - 1 + ky
        mask_y = (oy >= 0) & (oy < hout)
        sy, oy = sy[mask_y], oy[mask_y]
        for kx in range(3):
            sx = np.arange(win)
            ox = 2 * sx - 1 + kx
            mask_x = (ox >= 0) & (ox < wout)
            sx, ox = sx[mask_x], ox[mask_x]
            counts[np.ix_(tiles, oy, ox)] += pop_tile[np.ix_(tiles, sy, sx)]
    contributors_one_block = int(counts.sum(dtype=np.int64))
    optimistic_groups_one_block = int(((counts.astype(np.uint16) + 7) // 8).sum(dtype=np.int64))
    active = (counts > 0).reshape(ntile, hout * wout).T
    return counts, active, contributors_one_block * blocks, optimistic_groups_one_block * blocks


def lru_weight_misses(active, blocks, capacity, hash_sink):
    ntile = active.shape[1]
    cache = OrderedDict()
    misses = 0
    references = 0
    for row in active:
        tile_ids = np.flatnonzero(row)
        for block in range(blocks):
            for tile in tile_ids:
                key = block * ntile + int(tile)
                references += 1
                hash_sink.update(key.to_bytes(4, "little"))
                if key in cache:
                    cache.move_to_end(key)
                else:
                    misses += 1
                    cache[key] = None
                    if len(cache) > capacity:
                        cache.popitem(last=False)
    return misses, references


def canonical_contributor_digest(bits, topo, blocks):
    cin = bits.shape[0]
    h_forward = hashlib.sha256()
    h_pull = hashlib.sha256()
    total = 0
    for block in range(blocks):
        for tap in topo["tap_rows"]:
            mapping = tap["mapping"]
            if mapping:
                sy = np.fromiter((row[2] for row in mapping), dtype=np.int64)
                sx = np.fromiter((row[3] for row in mapping), dtype=np.int64)
                values = bits[:, sy, sx].T.reshape(-1)
            else:
                values = np.empty(0, dtype=np.uint8)
            packed = np.packbits(values, bitorder="little").tobytes()
            header = np.asarray([block, tap["ky"], tap["kx"], cin, len(mapping)],
                                dtype="<u4").tobytes()
            h_forward.update(header); h_forward.update(packed)
            # The pull path is generated from the independently checked inverse map.
            inverse = tap["pull_mapping"]
            if inverse:
                sy2 = np.fromiter((row[2] for row in inverse), dtype=np.int64)
                sx2 = np.fromiter((row[3] for row in inverse), dtype=np.int64)
                values2 = bits[:, sy2, sx2].T.reshape(-1)
            else:
                values2 = np.empty(0, dtype=np.uint8)
            packed2 = np.packbits(values2, bitorder="little").tobytes()
            h_pull.update(header); h_pull.update(packed2)
            total += int(values.sum(dtype=np.int64))
    return h_forward.hexdigest(), h_pull.hexdigest(), total


def runtime_topology(cin, hin, win, hout, wout):
    base = topology(cin, hin, win, hout, wout)
    for row in base["tap_rows"]:
        ky, kx = row["ky"], row["kx"]
        forward = []
        for sy in range(hin):
            oy = 2 * sy - 1 + ky
            if not 0 <= oy < hout:
                continue
            for sx in range(win):
                ox = 2 * sx - 1 + kx
                if 0 <= ox < wout:
                    forward.append((oy, ox, sy, sx))
        forward.sort()
        pull = []
        for oy in range(hout):
            ny = oy + 1 - ky
            if ny & 1:
                continue
            sy = ny // 2
            if not 0 <= sy < hin:
                continue
            for ox in range(wout):
                nx = ox + 1 - kx
                if nx & 1:
                    continue
                sx = nx // 2
                if 0 <= sx < win:
                    pull.append((oy, ox, sy, sx))
        require(forward == pull, "runtime topology mismatch")
        row["mapping"] = forward
        row["pull_mapping"] = pull
    return base


def acc24_probes(bits, quant, blocks, record_index, time):
    cin, hin, win = bits.shape
    cout = quant.shape[1]
    hout, wout = 2 * hin, 2 * win
    mismatches = 0
    digest = hashlib.sha256()
    count = 8
    seed = (record_index + 1) * 0x9E3779B1 ^ (time + 1) * 0x85EBCA77
    for probe in range(count):
        seed = (1664525 * seed + 1013904223) & 0xFFFFFFFF
        oy = seed % hout
        seed = (1664525 * seed + 1013904223) & 0xFFFFFFFF
        ox = seed % wout
        seed = (1664525 * seed + 1013904223) & 0xFFFFFFFF
        block = seed % blocks
        seed = (1664525 * seed + 1013904223) & 0xFFFFFFFF
        oc = min(cout - 1, block * 96 + (seed % 96))
        baseline = 0
        # Independent source-major order: channel is outermost.  For a fixed
        # destination only the inverse-derived at-most-four spatial positions
        # can match, so a full HxW scan would add runtime without adding proof.
        for channel in range(cin):
            for ky in range(3):
                ny = oy + 1 - ky
                if ny & 1:
                    continue
                sy = ny // 2
                if not 0 <= sy < hin:
                    continue
                for kx in range(3):
                    nx = ox + 1 - kx
                    if nx & 1:
                        continue
                    sx = nx // 2
                    if 0 <= sx < win and bits[channel, sy, sx]:
                        baseline = wrap24(baseline + int(quant[channel, oc, ky, kx]))
        candidate = 0
        for ky in range(3):
            ny = oy + 1 - ky
            if ny & 1:
                continue
            sy = ny // 2
            if not 0 <= sy < hin:
                continue
            for kx in range(3):
                nx = ox + 1 - kx
                if nx & 1:
                    continue
                sx = nx // 2
                if not 0 <= sx < win:
                    continue
                for channel in np.flatnonzero(bits[:, sy, sx]):
                    candidate = wrap24(candidate + int(quant[int(channel), oc, ky, kx]))
        mismatches += int(baseline != candidate)
        digest.update(np.asarray([oy, ox, block, oc, baseline, candidate],
                                 dtype="<i8").tobytes())
    return count, mismatches, digest.hexdigest()


def model_plane(bits, spec, topo, quant, record_index, time):
    name, cin, cout, hin, win, hout, wout, blocks = spec
    require(bits.shape == (cin, hin, win), "plane shape drift")
    counts, active, descriptors, groups = descriptor_counts(bits, blocks)
    forward_hash, pull_hash, digest_contributors = canonical_contributor_digest(bits, topo, blocks)
    require(descriptors == digest_contributors and forward_hash == pull_hash,
            "contributor digest/count mismatch")
    probes, acc_mismatch, acc_hash = acc24_probes(
        bits, quant, blocks, record_index, time)

    line_buffer_bytes = ((2 * win * ((cin + 7) // 8) + 127) // 128) * 128
    local_acc_bytes = blocks * 96 * 3
    weight_cache_bytes = LOGICAL_BUDGET_BYTES - line_buffer_bytes - local_acc_bytes - CONTROL_BYTES
    require(weight_cache_bytes >= 0, "PIDP storage exceeds 240 KiB before weights")
    cache_entries = weight_cache_bytes // WEIGHT_TILE_BYTES
    require(cache_entries >= 1, "no PIDP weight tile fits")
    weight_hash = hashlib.sha256()
    pidp_misses, pidp_refs = lru_weight_misses(active, blocks, cache_entries, weight_hash)
    active_tile_count = int(np.count_nonzero(active.any(axis=0)))
    a1_misses = active_tile_count * blocks

    plane_bits = cin * hin * win
    dense_vectors = hout * wout * blocks
    common_output = dense_vectors * OUTPUT_COMMIT_CYCLES
    bundles = (descriptors + 7) // 8
    terminal_a1 = 1029 + 2 * (blocks - 1)
    a1 = {
        "A1-OSG": {
            "source_scan": plane_bits * blocks,
            "descriptor_or_bundle": descriptors + 2 * bundles,
            "group_service": groups * 15,
            "weight_refill": a1_misses * WEIGHT_REFILL_CYCLES,
            "dense_output_commit": common_output,
            "terminal_directory": terminal_a1,
        },
        "A1-SC8": {
            "source_scan": plane_bits * blocks,
            "descriptor_or_bundle": 2 * bundles,
            "group_service": ((descriptors + 3) // 4) * 15,
            "weight_refill": a1_misses * WEIGHT_REFILL_CYCLES,
            "dense_output_commit": common_output,
            "terminal_directory": terminal_a1,
        },
        "A1-ISO8": {
            "source_scan": plane_bits * blocks,
            "descriptor_or_bundle": 2 * bundles,
            "group_service": ((descriptors + 1) // 2) * 15,
            "weight_refill": a1_misses * WEIGHT_REFILL_CYCLES,
            "dense_output_commit": common_output,
            "terminal_directory": terminal_a1,
        },
    }
    for ledger in a1.values():
        ledger["total"] = sum(ledger.values())
    bitmap_probe_words = topo["legal_spatial_edges"] * ((cin + 127) // 128)
    pidp = {
        "source_stream": (plane_bits + 127) // 128,
        "bitmap_probe": bitmap_probe_words,
        "group_service_optimistic": groups * 10,
        "weight_refill": pidp_misses * WEIGHT_REFILL_CYCLES,
        "dense_output_commit": common_output,
        "owner_transition": 2 * blocks + 3,
    }
    pidp["total"] = sum(pidp.values())

    psum_bytes = {
        "A1-OSG": groups * PSUM_BYTES_PER_DESTINATION_GROUP,
        "A1-SC8": descriptors * PSUM_BYTES_PER_DESTINATION_GROUP,
        "A1-ISO8": ((descriptors + 1) // 2) * PSUM_BYTES_PER_DESTINATION_GROUP,
    }
    return {
        "module": name, "time": time,
        "active_sources": int(bits.sum(dtype=np.int64)),
        "contributors": descriptors,
        "optimistic_k8_groups": groups,
        "contributor_multiset_sha256": forward_hash,
        "contributor_multiset_mismatches": 0,
        "acc24_probe_count": probes,
        "acc24_oracle_mismatches": acc_mismatch,
        "acc24_probe_sha256": acc_hash,
        "a1_cycles": a1,
        "pidp_cycles": pidp,
        "traffic": {
            "a1_materialized_descriptor_bytes": descriptors * DESCRIPTOR_BYTES,
            "a1_psum_bytes": psum_bytes,
            "pidp_materialized_descriptor_bytes": 0,
            "pidp_psum_bytes": 0,
            "pidp_bitmap_probe_bytes": bitmap_probe_words * 16,
            "a1_weight_refill_bytes": a1_misses * WEIGHT_TILE_BYTES,
            "pidp_weight_refill_bytes": pidp_misses * WEIGHT_TILE_BYTES,
        },
        "weight_cache": {
            "line_buffer_bytes": line_buffer_bytes,
            "local_acc_bytes": local_acc_bytes,
            "control_bytes": CONTROL_BYTES,
            "available_weight_cache_bytes": weight_cache_bytes,
            "weight_tile_bytes": WEIGHT_TILE_BYTES,
            "cache_entries": cache_entries,
            "active_tile_identities": active_tile_count * blocks,
            "pidp_references": pidp_refs,
            "pidp_misses": pidp_misses,
            "pidp_weight_sequence_sha256": weight_hash.hexdigest(),
        },
    }


def summarize(rows, fixed_a1):
    dimensions = ["all", "headline", "diagnostic_d1"]
    totals = {}
    for dimension in dimensions:
        selected = [r for r in rows if dimension == "all" or
                    (dimension == "headline" and r["module_index"] in HEADLINE) or
                    (dimension == "diagnostic_d1" and r["module_index"] == 1)]
        a1_cycles = sum(r["a1_cycles"][fixed_a1]["total"] for r in selected)
        pidp_cycles = sum(r["pidp_cycles"]["total"] for r in selected)
        a1_desc = sum(r["traffic"]["a1_materialized_descriptor_bytes"] for r in selected)
        a1_psum = sum(r["traffic"]["a1_psum_bytes"][fixed_a1] for r in selected)
        pidp_desc = sum(r["traffic"]["pidp_materialized_descriptor_bytes"] for r in selected)
        pidp_psum = sum(r["traffic"]["pidp_psum_bytes"] for r in selected)
        totals[dimension] = {
            "records_times": len(selected),
            "a1_cycles": a1_cycles, "pidp_cycles": pidp_cycles,
            "a1_over_pidp": ratio(a1_cycles, pidp_cycles),
            "a1_descriptor_plus_psum_bytes": a1_desc + a1_psum,
            "pidp_descriptor_plus_psum_bytes": pidp_desc + pidp_psum,
            "descriptor_plus_psum_reduction_fraction": (
                "1.000000000000" if a1_desc + a1_psum else "0.000000000000"),
            "pidp_bitmap_probe_bytes": sum(r["traffic"]["pidp_bitmap_probe_bytes"] for r in selected),
            "a1_weight_refill_bytes": sum(r["traffic"]["a1_weight_refill_bytes"] for r in selected),
            "pidp_weight_refill_bytes": sum(r["traffic"]["pidp_weight_refill_bytes"] for r in selected),
        }
    return totals


def self_test():
    topo = runtime_topology(5, 2, 3, 4, 6)
    require(topo["topology_mismatches"] == 0, "topology self-test")
    bits = np.asarray([[[1, 0, 1], [0, 1, 0]],
                       [[0, 1, 0], [1, 0, 1]],
                       [[1, 1, 0], [0, 0, 1]],
                       [[0, 0, 0], [1, 1, 1]],
                       [[1, 0, 0], [0, 1, 0]]], dtype=np.uint8)
    counts, active, contributors, groups = descriptor_counts(bits, 1)
    require(int(counts.sum()) == contributors and groups > 0 and active.any(),
            "descriptor-count self-test")
    cache_hash = hashlib.sha256()
    misses, refs = lru_weight_misses(
        np.asarray([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=bool),
        1, 2, cache_hash)
    require((misses, refs) == (4, 6), "LRU self-test")
    require(wrap24((1 << 23) - 1 + 1) == -(1 << 23), "Acc24 wrap self-test")
    f, p, n = canonical_contributor_digest(bits, topo, 1)
    require(f == p and n == contributors, "multiset digest self-test")
    print("PASS M712 static self-test")


def production(args):
    root = Path(args.repo_root).resolve()
    hw = root / "hw_autoresearch_nts07"
    contract_path = Path(args.contract).resolve()
    contract = strict_json(contract_path)
    require(contract.get("schema") == CONTRACT_SCHEMA, "contract schema drift")
    require(sha256(hw / "docs/359_DATE终局冻结_20260813.md") == DOCS359_SHA,
            "docs359 drift")
    m699 = hw / contract["inputs"]["m699_directory"]
    m705 = hw / contract["inputs"]["m705_review_directory"]
    m686 = hw / contract["inputs"]["m686_weight_probe_directory"]
    id699, id705, id686 = verify_directory(m699), verify_directory(m705), verify_directory(m686)
    require(sha256(m699 / "manifest.json") == contract["inputs"]["m699_manifest_sha256"] and
            id699["outer_file_sha256"] == contract["inputs"]["m699_manifest_outer_seal_file_sha256"],
            "M699 identity drift")
    require(sha256(m705 / "review.json") == contract["inputs"]["m705_review_json_sha256"] and
            id705["outer_file_sha256"] == contract["inputs"]["m705_review_outer_seal_file_sha256"],
            "M705 identity drift")
    require(sha256(m686 / "manifest.json") == contract["inputs"]["m686_manifest_sha256"] and
            id686["outer_file_sha256"] == contract["inputs"]["m686_outer_seal_file_sha256"],
            "M686 identity drift")
    manifest = strict_json(m699 / "manifest.json")
    require(len(manifest["records"]) == 120, "M699 record population drift")
    observed_sequences = sorted(set(row["sequence"] for row in manifest["records"]))
    require(observed_sequences == sorted(contract["population"]["sequences"]),
            "M699 sequence identity drift")
    review705 = strict_json(m705 / "review.json")
    require(review705.get("go") is True and review705.get("score") == 98 and
            review705.get("severity", {}).get("p0") == 0 and
            review705.get("severity", {}).get("p1") == 0,
            "M705 admission drift")

    topologies = {}
    quant = {}
    quant_id = {}
    for index, spec in MODULES.items():
        name, cin, cout, hin, win, hout, wout, blocks = spec
        topologies[index] = runtime_topology(cin, hin, win, hout, wout)
        require(topologies[index]["topology_mismatches"] == 0,
                "module topology mismatch")
        weight_name = ("d1.weight.folded_theta.f32le" if index == 1
                       else "d{}.weight.f32le".format(index))
        quant[index], quant_id[index] = quantized_weights(
            m686 / "weights" / weight_name, cin, cout)

    rows = []
    for record_index, record in enumerate(manifest["records"]):
        index = int(record["module_index"])
        spec = MODULES[index]
        name, cin, cout, hin, win, hout, wout, blocks = spec
        require(record["name"].endswith("decoders.{}.deconv.0".format(index)) and
                tuple(record["input_shape"]) == (10, 1, cin, hin, win),
                "M699 record module/shape drift")
        route = record["route"]
        require((index == 1 and route == "EXACT_SCALED_BINARY_BITPACK") or
                (index != 1 and route == "EXACT_BINARY_BITPACK"),
                "M699 route drift")
        payload = m699.joinpath(*safe_member(record["relative_path"]).parts)
        packed_sha = (record["statistics"]["scaled_binary_audit"]["packed_sha256"]
                      if index == 1 else record["statistics"]["packed_sha256"])
        require(sha256(payload) == packed_sha,
                "M699 payload SHA drift")
        bits = unpack_record(payload, (10, cin, hin, win))
        for time in range(10):
            row = model_plane(bits[time], spec, topologies[index], quant[index],
                              record_index, time)
            row.update({
                "record_index": record_index,
                "global_sample_id": int(record["global_sample_id"]),
                "sequence": record["sequence"],
                "sequence_sample_id": int(record["sequence_sample_id"]),
                "module_index": index,
                "route": route,
                "headline_eligible": index in HEADLINE,
            })
            rows.append(row)

    require(len(rows) == 1200, "M712 row population drift")
    require(sum(r["contributor_multiset_mismatches"] + r["acc24_oracle_mismatches"]
                for r in rows) == 0, "M712 exactness gate failed")
    headline_rows = [r for r in rows if r["module_index"] in HEADLINE]
    a1_total = {name: sum(r["a1_cycles"][name]["total"] for r in headline_rows)
                for name in ("A1-SC8", "A1-ISO8", "A1-OSG")}
    tie = {"A1-OSG": 0, "A1-SC8": 1, "A1-ISO8": 2}
    fixed_a1 = min(a1_total, key=lambda name: (a1_total[name], tie[name]))
    totals = summarize(rows, fixed_a1)

    per_sequence = {}
    for sequence in sorted({r["sequence"] for r in headline_rows}):
        selected = [r for r in headline_rows if r["sequence"] == sequence]
        a = sum(r["a1_cycles"][fixed_a1]["total"] for r in selected)
        p = sum(r["pidp_cycles"]["total"] for r in selected)
        per_sequence[sequence] = {"a1_cycles": a, "pidp_cycles": p,
                                  "a1_over_pidp": ratio(a, p)}
    per_module = {}
    for index, spec in MODULES.items():
        selected = [r for r in rows if r["module_index"] == index]
        a = sum(r["a1_cycles"][fixed_a1]["total"] for r in selected)
        p = sum(r["pidp_cycles"]["total"] for r in selected)
        per_module[spec[0]] = {
            "headline_eligible": index in HEADLINE,
            "a1_cycles": a, "pidp_cycles": p, "a1_over_pidp": ratio(a, p),
            "pidp_weight_refill_bytes": sum(r["traffic"]["pidp_weight_refill_bytes"] for r in selected),
            "a1_weight_refill_bytes": sum(r["traffic"]["a1_weight_refill_bytes"] for r in selected),
            "contributors": sum(r["contributors"] for r in selected),
            "optimistic_k8_groups": sum(r["optimistic_k8_groups"] for r in selected),
        }
    # Non-admitted diagnostic only: choose the dataflow once per layer from
    # static dimensions.  There is no record/sequence/runtime oracle.  The
    # selection predicate is the first-principles weight working-set fit.
    selective_policy = {}
    for index, spec in MODULES.items():
        module_rows = [r for r in rows if r["module_index"] == index]
        cache_entries = module_rows[0]["weight_cache"]["cache_entries"]
        total_weight_tile_identities = ((spec[1] + 15) // 16) * spec[7]
        use_pidp = total_weight_tile_identities <= cache_entries
        selective_policy[spec[0]] = {
            "total_weight_tile_identities": total_weight_tile_identities,
            "exact_cache_entries": cache_entries,
            "use_pidp": use_pidp,
            "chosen_dataflow": "PIDP" if use_pidp else fixed_a1,
        }
    selective_headline_cycles = sum(
        (r["pidp_cycles"]["total"] if selective_policy[r["module"]]["use_pidp"]
         else r["a1_cycles"][fixed_a1]["total"]) for r in headline_rows)
    selective_per_sequence = {}
    for sequence in sorted({r["sequence"] for r in headline_rows}):
        selected = [r for r in headline_rows if r["sequence"] == sequence]
        baseline_cycles = sum(r["a1_cycles"][fixed_a1]["total"] for r in selected)
        chosen_cycles = sum(
            (r["pidp_cycles"]["total"] if selective_policy[r["module"]]["use_pidp"]
             else r["a1_cycles"][fixed_a1]["total"]) for r in selected)
        selective_per_sequence[sequence] = {
            "a1_cycles": baseline_cycles,
            "selective_cycles": chosen_cycles,
            "a1_over_selective": ratio(baseline_cycles, chosen_cycles),
        }
    selective_diagnostic = {
        "admitted": False,
        "fresh_hammer_and_new_contract_required": True,
        "policy": "compile-time per-layer PIDP iff total INT8 weight-tile identities <= exact logical cache entries",
        "runtime_or_record_or_sequence_oracle": False,
        "layer_configuration_bits": 4,
        "policy_by_module": selective_policy,
        "headline_a1_cycles": totals["headline"]["a1_cycles"],
        "headline_selective_cycles": selective_headline_cycles,
        "headline_a1_over_selective": ratio(totals["headline"]["a1_cycles"],
                                              selective_headline_cycles),
        "per_sequence": selective_per_sequence,
        "claim_boundary": "diagnostic composition of already computed integer rows; not a PIDP GO, RTL admission, system speedup or paper claim",
    }
    headline_ratio = Decimal(totals["headline"]["a1_over_pidp"])
    performance_go = headline_ratio >= Decimal("1.20") and all(
        Decimal(v["a1_over_pidp"]) >= Decimal("1.05") for v in per_sequence.values())
    traffic_go = (headline_ratio >= Decimal(1) / Decimal("1.05") and
                  Decimal(totals["headline"]["descriptor_plus_psum_reduction_fraction"]) >= Decimal("0.30"))
    status = "GO_MINIMAL_RTL_AFTER_FRESH_HAMMER" if (performance_go or traffic_go) else "KILL_NO_RTL"
    rename = {
        "same_contributor_multiset_as_a1_osg": True,
        "same_execution_sequence_as_a1_osg": False,
        "why_not_byte_equivalent": "PIDP is destination-major, materializes no scatter descriptor and changes the weight/reference order; A1-OSG is source-major and materializes/joins descriptors.",
        "first_principles_collision": "The candidate is nevertheless the classic destination-stationary side of the weight-reuse versus psum-reuse dataflow tradeoff. Under 240 KiB it cannot keep the D0/D1/D2 weight working sets while retaining one destination owner.",
        "novel_arithmetic": False,
    }
    report = {
        "schema": RESULT_SCHEMA, "date": "2026-08-28", "status": status,
        "decision": {
            "performance_go": performance_go, "traffic_go": traffic_go,
            "fresh_result_hammer_required": True,
            "rtl_authorized_now": False,
            "headline_ratio_of_sums_a1_over_pidp": totals["headline"]["a1_over_pidp"],
            "minimum_sequence_ratio": min(v["a1_over_pidp"] for v in per_sequence.values()),
            "traffic_reduction_fraction": totals["headline"]["descriptor_plus_psum_reduction_fraction"],
            "interpretation": "A KILL is safe because the PIDP ledger is candidate-favorable: it gives free descriptor/psum removal, conflict-free ceil(N/8) groups, a fully associative cache and no directory-clear tax.",
        },
        "fixed_strongest_a1": fixed_a1,
        "a1_headline_cycle_totals": a1_total,
        "totals": totals,
        "per_sequence_headline": per_sequence,
        "per_module": per_module,
        "exactness": {
            "rows": len(rows), "records": 120, "timesteps": 10,
            "headline_rows": len(headline_rows), "diagnostic_d1_rows": 300,
            "contributor_multiset_mismatches": sum(r["contributor_multiset_mismatches"] for r in rows),
            "acc24_probe_count": sum(r["acc24_probe_count"] for r in rows),
            "acc24_oracle_mismatches": sum(r["acc24_oracle_mismatches"] for r in rows),
            "topology_mismatches": sum(t["topology_mismatches"] for t in topologies.values()),
        },
        "topologies": {MODULES[i][0]: {
            "legal_spatial_edges": t["legal_spatial_edges"],
            "topology_mismatches": t["topology_mismatches"],
            "aggregate_sha256": t["aggregate_sha256"],
        } for i, t in topologies.items()},
        "local_int8_probe_identities": {MODULES[i][0]: quant_id[i] for i in MODULES},
        "pbr4_a1_osg_rename_audit": rename,
        "selective_weight_fit_composition_diagnostic": selective_diagnostic,
        "claim_boundary": contract["claim_boundary"],
        "identity": {
            "contract_path": str(contract_path), "contract_sha256": sha256(contract_path),
            "analyzer_path": str(Path(__file__).resolve()), "analyzer_sha256": sha256(Path(__file__).resolve()),
            "m699_manifest_sha256": sha256(m699 / "manifest.json"),
            "m699_outer_file_sha256": id699["outer_file_sha256"],
            "m705_review_json_sha256": sha256(m705 / "review.json"),
            "m705_outer_file_sha256": id705["outer_file_sha256"],
            "m686_manifest_sha256": sha256(m686 / "manifest.json"),
            "m686_outer_file_sha256": id686["outer_file_sha256"],
            "docs359_sha256": sha256(hw / "docs/359_DATE终局冻结_20260813.md"),
        },
        "rows_file": "rows.jsonl",
    }

    output = Path(args.output).resolve()
    require(not output.exists(), "canonical output exists")
    staging = Path(tempfile.mkdtemp(prefix="." + output.name + ".staging.", dir=str(output.parent)))
    try:
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with (staging / "rows.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        (staging / "RUN_COMPLETE.txt").write_text(status + "\n", encoding="utf-8")
        write_seal(staging)
        verify_directory(staging)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": status, "fixed_a1": fixed_a1,
                      "headline_ratio": totals["headline"]["a1_over_pidp"],
                      "minimum_sequence_ratio": min(v["a1_over_pidp"] for v in per_sequence.values()),
                      "output": str(output)}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--contract")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(args.contract and args.output, "production requires contract and output")
    production(args)


if __name__ == "__main__":
    main()
