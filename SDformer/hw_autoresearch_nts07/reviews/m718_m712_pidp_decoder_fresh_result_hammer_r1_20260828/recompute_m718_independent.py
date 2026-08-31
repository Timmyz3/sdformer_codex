#!/usr/bin/env python3
"""Receipt-blind independent M712 result recomputation.

The author M712 analyzer is never imported or executed.  This program verifies
the canonical seals, independently unpacks all 120 M699 bitpacks, rebuilds all
1200 contributor/cycle/cache rows, checks author rows as outputs, and runs a
different small Acc24 scheduling oracle on sealed M686 weights.
"""

import argparse
import ctypes
from decimal import Decimal, getcontext
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import tempfile

import numpy as np


getcontext().prec = 40
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
PSUM_BYTES_PER_GROUP = 6 * 2 * 48


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(rows):
        output = {}
        for key, value in rows:
            require(key not in output, "duplicate JSON key in {}: {}".format(path, key))
            output[key] = value
        return output
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON token: " + token)))


def safe_member(name):
    member = PurePosixPath(name)
    require(member.parts and not member.is_absolute() and ".." not in member.parts and
            member.as_posix() == name, "unsafe member: " + name)
    return member


def verify_sealed_directory(path):
    path = Path(path)
    require(path.is_dir() and not path.is_symlink(), "bad sealed directory: " + str(path))
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and outer.is_file(), "missing seals: " + str(path))
    expected_names = set()
    member_count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and len(fields[0]) == 64, "bad manifest line")
        expected, name = fields
        require(name not in expected_names, "duplicate member: " + name)
        member = path.joinpath(*safe_member(name).parts)
        require(member.is_file() and not member.is_symlink(), "missing/symlink member: " + name)
        require(sha256(member) == expected, "member hash mismatch: " + name)
        expected_names.add(name)
        member_count += 1
    actual_names = set()
    root_seals = {manifest.resolve(), outer.resolve()}
    for member in path.rglob("*"):
        require(not member.is_symlink(), "symlink in sealed directory: " + str(member))
        if member.is_file() and member.resolve() not in root_seals:
            actual_names.add(member.relative_to(path).as_posix())
    require(actual_names == expected_names, "sealed member set mismatch: " + str(path))
    outer_fields = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    require(outer_fields == [sha256(manifest), "SHA256SUMS"],
            "outer seal mismatch: " + str(path))
    return {
        "member_count": member_count,
        "manifest_sha256": sha256(manifest),
        "outer_file_sha256": sha256(outer),
    }


def compile_lru_oracle(c_source):
    temporary = tempfile.TemporaryDirectory(prefix="m718_lru_oracle_")
    shared = Path(temporary.name) / "m718_lru_oracle.so"
    command = ["/usr/bin/gcc", "-std=c99", "-O3", "-fPIC", "-shared",
               str(c_source), "-o", str(shared)]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True)
    require(process.returncode == 0, "LRU oracle compile failed: " + process.stderr)
    library = ctypes.CDLL(str(shared))
    function = library.m718_lru_misses
    function.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                         ctypes.c_int, ctypes.c_int, ctypes.c_int,
                         ctypes.POINTER(ctypes.c_uint64)]
    function.restype = ctypes.c_uint64
    return temporary, function, command


def ratio(numerator, denominator):
    require(denominator > 0, "zero denominator")
    return format(Decimal(int(numerator)) / Decimal(int(denominator)), ".12f")


def wrap24(value):
    unsigned = int(value) & ((1 << 24) - 1)
    return unsigned - (1 << 24) if unsigned & (1 << 23) else unsigned


def independent_topology(hin, win, hout, wout):
    tap_rows = []
    total_edges = 0
    mismatch = 0
    aggregate = hashlib.sha256()
    for ky in range(3):
        for kx in range(3):
            forward = []
            for sy in range(hin):
                for sx in range(win):
                    oy = 2 * sy - 1 + ky
                    ox = 2 * sx - 1 + kx
                    if 0 <= oy < hout and 0 <= ox < wout:
                        forward.append((oy, ox, sy, sx))
            inverse = []
            for oy in range(hout):
                ny = oy + 1 - ky
                if ny % 2:
                    continue
                sy = ny // 2
                for ox in range(wout):
                    nx = ox + 1 - kx
                    if nx % 2:
                        continue
                    sx = nx // 2
                    if 0 <= sy < hin and 0 <= sx < win:
                        inverse.append((oy, ox, sy, sx))
            forward.sort()
            inverse.sort()
            mismatch += int(forward != inverse)
            canonical = json.dumps([ky, kx, forward], separators=(",", ":")).encode()
            digest = hashlib.sha256(canonical).digest()
            aggregate.update(digest)
            tap_rows.append({"ky": ky, "kx": kx, "mapping": forward,
                             "edge_count": len(forward),
                             "forward_inverse_mismatch": int(forward != inverse)})
            total_edges += len(forward)
    return {"tap_rows": tap_rows, "legal_spatial_edges": total_edges,
            "topology_mismatches": mismatch,
            "aggregate_sha256": aggregate.hexdigest()}


def unpack_bitpack(path, expected_bits):
    payload = np.fromfile(str(path), dtype=np.uint8)
    require(payload.size * 8 == expected_bits, "unexpected bitpack size: " + str(path))
    return np.unpackbits(payload, bitorder="little").astype(np.uint8, copy=False)


def contributor_plane(bits, spec):
    name, cin, cout, hin, win, hout, wout, blocks = spec
    del name, cout
    require(bits.shape == (cin, hin, win), "plane shape mismatch")
    input_tiles = (cin + 15) // 16
    padded = np.zeros((input_tiles * 16, hin, win), dtype=np.uint8)
    padded[:cin] = bits
    tile_pop = padded.reshape(input_tiles, 16, hin, win).sum(
        axis=1, dtype=np.uint16)
    counts = np.zeros((input_tiles, hout, wout), dtype=np.uint16)
    tile_index = np.arange(input_tiles)
    for ky in range(3):
        source_y = np.arange(hin)
        output_y = 2 * source_y - 1 + ky
        legal_y = (output_y >= 0) & (output_y < hout)
        source_y, output_y = source_y[legal_y], output_y[legal_y]
        for kx in range(3):
            source_x = np.arange(win)
            output_x = 2 * source_x - 1 + kx
            legal_x = (output_x >= 0) & (output_x < wout)
            source_x, output_x = source_x[legal_x], output_x[legal_x]
            counts[np.ix_(tile_index, output_y, output_x)] += \
                tile_pop[np.ix_(tile_index, source_y, source_x)]
    contributors = int(counts.sum(dtype=np.int64)) * blocks
    groups = int(((counts.astype(np.uint32) + 7) // 8).sum(dtype=np.int64)) * blocks
    active = np.ascontiguousarray((counts > 0).reshape(input_tiles, -1).T,
                                  dtype=np.uint8)
    return active, contributors, groups


def lru_counts(function, active, blocks, capacity):
    references = ctypes.c_uint64(0)
    pointer = active.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    misses = int(function(pointer, active.shape[0], active.shape[1], blocks,
                          capacity, ctypes.byref(references)))
    require(misses != (1 << 64) - 1, "LRU oracle rejected geometry")
    return misses, int(references.value)


def quantized_weights(path, cin, cout):
    raw = np.fromfile(str(path), dtype="<f4")
    require(raw.size == cin * cout * 9, "weight size mismatch: " + str(path))
    weight = raw.reshape(cin, cout, 3, 3).astype(np.float64)
    maximum = np.max(np.abs(weight), axis=(0, 2, 3))
    scale = np.where(maximum == 0.0, 1.0, maximum / 127.0)
    quant = np.clip(np.rint(weight / scale[None, :, None, None]), -127, 127)
    return quant.astype(np.int8), {
        "scale_sha256": hashlib.sha256(scale.astype("<f8").tobytes()).hexdigest(),
        "int8_sha256": hashlib.sha256(quant.astype(np.int8).tobytes()).hexdigest(),
    }


def independent_acc24_probes(bits, quant, spec, record_index, time_index):
    name, cin, cout, hin, win, hout, wout, blocks = spec
    del name, hin, win
    seed = hashlib.sha256("{}:{}".format(record_index, time_index).encode()).digest()
    positions = [
        (0, 0, 0),
        (hout - 1, wout - 1, cout - 1),
        (hout // 2, wout // 2, cout // 2),
        (int.from_bytes(seed[0:4], "little") % hout,
         int.from_bytes(seed[4:8], "little") % wout,
         int.from_bytes(seed[8:12], "little") % cout),
    ]
    mismatches = 0
    digest = hashlib.sha256()
    for oy, ox, oc in positions:
        require(oc // 96 < blocks, "probe block mismatch")
        source_major = 0
        for channel in range(cin):
            for ky in range(3):
                numerator_y = oy + 1 - ky
                if numerator_y % 2:
                    continue
                sy = numerator_y // 2
                for kx in range(3):
                    numerator_x = ox + 1 - kx
                    if numerator_x % 2:
                        continue
                    sx = numerator_x // 2
                    if 0 <= sy < bits.shape[1] and 0 <= sx < bits.shape[2] and \
                            bits[channel, sy, sx]:
                        source_major = wrap24(
                            source_major + int(quant[channel, oc, ky, kx]))
        destination_pull = 0
        for ky in range(3):
            numerator_y = oy + 1 - ky
            if numerator_y % 2:
                continue
            sy = numerator_y // 2
            for kx in range(3):
                numerator_x = ox + 1 - kx
                if numerator_x % 2:
                    continue
                sx = numerator_x // 2
                if not (0 <= sy < bits.shape[1] and 0 <= sx < bits.shape[2]):
                    continue
                for channel in np.flatnonzero(bits[:, sy, sx]):
                    destination_pull = wrap24(
                        destination_pull + int(quant[int(channel), oc, ky, kx]))
        mismatches += int(source_major != destination_pull)
        digest.update(np.asarray(
            [oy, ox, oc, source_major, destination_pull], dtype="<i8").tobytes())
    return len(positions), mismatches, digest.hexdigest()


def build_ledgers(bits, spec, topology, lru_function):
    name, cin, cout, hin, win, hout, wout, blocks = spec
    del cout
    active, contributors, groups = contributor_plane(bits, spec)
    line_buffer_bytes = int(math.ceil(
        (2 * win * int(math.ceil(cin / 8.0))) / 128.0)) * 128
    local_acc_bytes = blocks * 96 * 3
    cache_bytes = LOGICAL_BUDGET_BYTES - line_buffer_bytes - local_acc_bytes - CONTROL_BYTES
    cache_entries = cache_bytes // WEIGHT_TILE_BYTES
    require(cache_entries > 0, "no cache entry")
    pidp_misses, pidp_references = lru_counts(
        lru_function, active, blocks, cache_entries)
    cap12_misses = None
    if name == "D3":
        cap12_misses, cap12_references = lru_counts(lru_function, active, blocks, 12)
        require(cap12_references == pidp_references, "D3 cap12 reference drift")
    active_tile_count = int(np.count_nonzero(active.any(axis=0)))
    a1_misses = active_tile_count * blocks
    plane_bits = cin * hin * win
    dense_vectors = hout * wout * blocks
    common_output = dense_vectors * OUTPUT_COMMIT_CYCLES
    bundles = (contributors + 7) // 8
    terminal = 1029 + 2 * (blocks - 1)
    a1 = {
        "A1-OSG": {
            "source_scan": plane_bits * blocks,
            "descriptor_or_bundle": contributors + 2 * bundles,
            "group_service": groups * 15,
            "weight_refill": a1_misses * WEIGHT_REFILL_CYCLES,
            "dense_output_commit": common_output,
            "terminal_directory": terminal,
        },
        "A1-SC8": {
            "source_scan": plane_bits * blocks,
            "descriptor_or_bundle": 2 * bundles,
            "group_service": ((contributors + 3) // 4) * 15,
            "weight_refill": a1_misses * WEIGHT_REFILL_CYCLES,
            "dense_output_commit": common_output,
            "terminal_directory": terminal,
        },
        "A1-ISO8": {
            "source_scan": plane_bits * blocks,
            "descriptor_or_bundle": 2 * bundles,
            "group_service": ((contributors + 1) // 2) * 15,
            "weight_refill": a1_misses * WEIGHT_REFILL_CYCLES,
            "dense_output_commit": common_output,
            "terminal_directory": terminal,
        },
    }
    for ledger in a1.values():
        ledger["total"] = sum(ledger.values())
    bitmap_probe_words = topology["legal_spatial_edges"] * int(math.ceil(cin / 128.0))
    pidp = {
        "source_stream": int(math.ceil(plane_bits / 128.0)),
        "bitmap_probe": bitmap_probe_words,
        "group_service_optimistic": groups * 10,
        "weight_refill": pidp_misses * WEIGHT_REFILL_CYCLES,
        "dense_output_commit": common_output,
        "owner_transition": 2 * blocks + 3,
    }
    pidp["total"] = sum(pidp.values())
    cap12_total = None
    if cap12_misses is not None:
        cap12_total = pidp["total"] - pidp["weight_refill"] + \
            cap12_misses * WEIGHT_REFILL_CYCLES
    traffic = {
        "a1_materialized_descriptor_bytes": contributors * DESCRIPTOR_BYTES,
        "a1_psum_bytes": {
            "A1-OSG": groups * PSUM_BYTES_PER_GROUP,
            "A1-SC8": contributors * PSUM_BYTES_PER_GROUP,
            "A1-ISO8": ((contributors + 1) // 2) * PSUM_BYTES_PER_GROUP,
        },
        "pidp_materialized_descriptor_bytes": 0,
        "pidp_psum_bytes": 0,
        "pidp_bitmap_probe_bytes": bitmap_probe_words * 16,
        "a1_weight_refill_bytes": a1_misses * WEIGHT_TILE_BYTES,
        "pidp_weight_refill_bytes": pidp_misses * WEIGHT_TILE_BYTES,
    }
    cache = {
        "line_buffer_bytes": line_buffer_bytes,
        "local_acc_bytes": local_acc_bytes,
        "control_bytes": CONTROL_BYTES,
        "available_weight_cache_bytes": cache_bytes,
        "weight_tile_bytes": WEIGHT_TILE_BYTES,
        "cache_entries": cache_entries,
        "active_tile_identities": active_tile_count * blocks,
        "pidp_references": pidp_references,
        "pidp_misses": pidp_misses,
    }
    return {
        "active_sources": int(bits.sum(dtype=np.int64)),
        "contributors": contributors,
        "groups": groups,
        "a1": a1,
        "pidp": pidp,
        "traffic": traffic,
        "cache": cache,
        "d3_cap12_pidp_total": cap12_total,
        "d3_cap12_misses": cap12_misses,
    }


def compare_equal(observed, expected, category, mismatches, key):
    if observed != expected:
        mismatches[category] = mismatches.get(category, 0) + 1
        if len(mismatches.setdefault("samples", [])) < 20:
            mismatches["samples"].append({
                "category": category, "key": key,
                "observed": observed, "expected": expected})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    hw = root / "hw_autoresearch_nts07"
    review_dir = Path(__file__).resolve().parent
    result_dir = hw / "results/m712_pidp_decoder_exact_cpu_fastkill_r1_20260828"
    handoff_dir = hw / "reviews/m712_pidp_decoder_exact_cpu_fastkill_author_handoff_r1_20260828"
    m699_dir = hw / "system_handoff/outgoing/m699_h67_ep35_multisequence_decoder_payload_s3x10_r1_20260828"
    m705_dir = hw / "reviews/m705_m699_multisequence_decoder_payload_fresh_result_hammer_r1_20260828"
    m686_dir = hw / "system_handoff/outgoing/m686r6_h67_ep35_layer_static_decoder_payload_s10_r1_20260828"
    contract_path = hw / "contracts/m712_pidp_decoder_exact_cpu_fastkill_contract_r1_20260828.json"
    author_analyzer = hw / "system_simulator/scripts/analyze_m712_pidp_decoder_exact_cpu_fastkill.py"

    seals = {
        "m712_result": verify_sealed_directory(result_dir),
        "m712_author_handoff": verify_sealed_directory(handoff_dir),
        "m699_payload": verify_sealed_directory(m699_dir),
        "m705_review": verify_sealed_directory(m705_dir),
        "m686_payload": verify_sealed_directory(m686_dir),
    }
    contract = strict_json(contract_path)
    manifest699 = strict_json(m699_dir / "manifest.json")
    review705 = strict_json(m705_dir / "review.json")
    report = strict_json(result_dir / "report.json")
    handoff = strict_json(handoff_dir / "handoff.json")
    require(sha256(m699_dir / "manifest.json") == contract["inputs"]["m699_manifest_sha256"],
            "M699 contract identity mismatch")
    require(sha256(m705_dir / "review.json") == contract["inputs"]["m705_review_json_sha256"],
            "M705 contract identity mismatch")
    require(sha256(m686_dir / "manifest.json") == contract["inputs"]["m686_manifest_sha256"],
            "M686 contract identity mismatch")
    require(review705.get("go") is True and review705.get("score") == 98,
            "M705 admission mismatch")
    require(sha256(author_analyzer) == report["identity"]["analyzer_sha256"],
            "author analyzer identity mismatch")

    rows = []
    with (result_dir / "rows.jsonl").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            require(line.strip(), "blank author row")
            row = json.loads(line)
            rows.append(row)
    require(len(rows) == 1200, "author row count mismatch")
    author_rows = {}
    for row in rows:
        key = (int(row["record_index"]), int(row["time"]))
        require(key not in author_rows, "duplicate author row")
        author_rows[key] = row

    c_temporary, lru_function, compile_command = compile_lru_oracle(
        review_dir / "m718_lru_oracle.c")
    try:
        topologies = {}
        quantized = {}
        quant_identity = {}
        for index, spec in MODULES.items():
            name, cin, cout, hin, win, hout, wout, blocks = spec
            del blocks
            topologies[index] = independent_topology(hin, win, hout, wout)
            weight_name = "d1.weight.folded_theta.f32le" if index == 1 else \
                "d{}.weight.f32le".format(index)
            quantized[index], quant_identity[index] = quantized_weights(
                m686_dir / "weights" / weight_name, cin, cout)

        mismatch = {"samples": []}
        independent_rows = []
        acc_probe_count = 0
        acc_probe_mismatches = 0
        acc_probe_digest = hashlib.sha256()
        record_population = {"sequences": {}, "modules": {}}
        for record_index, record in enumerate(manifest699["records"]):
            index = int(record["module_index"])
            spec = MODULES[index]
            name, cin, cout, hin, win, hout, wout, blocks = spec
            del cout, hout, wout, blocks
            expected_shape = [10, 1, cin, hin, win]
            compare_equal(record["input_shape"], expected_shape, "manifest_shape",
                          mismatch, [record_index, -1])
            payload = m699_dir.joinpath(*safe_member(record["relative_path"]).parts)
            expected_sha = (record["statistics"]["scaled_binary_audit"]["packed_sha256"]
                            if index == 1 else record["statistics"]["packed_sha256"])
            compare_equal(sha256(payload), expected_sha, "payload_sha", mismatch,
                          [record_index, -1])
            flat = unpack_bitpack(payload, 10 * cin * hin * win)
            bits10 = flat.reshape(10, cin, hin, win)
            expected_ones = (record["statistics"]["scaled_binary_audit"]["theta_count"]
                             if index == 1 else record["statistics"]["one_count"])
            compare_equal(int(flat.sum(dtype=np.int64)), int(expected_ones),
                          "payload_popcount", mismatch, [record_index, -1])
            sequence = record["sequence"]
            record_population["sequences"][sequence] = \
                record_population["sequences"].get(sequence, 0) + 1
            record_population["modules"][name] = \
                record_population["modules"].get(name, 0) + 1

            for time_index in range(10):
                key = (record_index, time_index)
                observed = author_rows[key]
                ledgers = build_ledgers(bits10[time_index], spec,
                                        topologies[index], lru_function)
                compare_equal(observed["record_index"], record_index,
                              "row_identity", mismatch, key)
                compare_equal(observed["time"], time_index,
                              "row_identity", mismatch, key)
                compare_equal(observed["sequence"], sequence,
                              "row_identity", mismatch, key)
                compare_equal(observed["module_index"], index,
                              "row_identity", mismatch, key)
                compare_equal(observed["active_sources"], ledgers["active_sources"],
                              "active_sources", mismatch, key)
                compare_equal(observed["contributors"], ledgers["contributors"],
                              "contributors", mismatch, key)
                compare_equal(observed["optimistic_k8_groups"], ledgers["groups"],
                              "groups", mismatch, key)
                compare_equal(observed["a1_cycles"], ledgers["a1"],
                              "a1_ledgers", mismatch, key)
                compare_equal(observed["pidp_cycles"], ledgers["pidp"],
                              "pidp_ledgers", mismatch, key)
                compare_equal(observed["traffic"], ledgers["traffic"],
                              "traffic_ledgers", mismatch, key)
                observed_cache = dict(observed["weight_cache"])
                observed_cache.pop("pidp_weight_sequence_sha256", None)
                compare_equal(observed_cache, ledgers["cache"],
                              "cache_ledgers", mismatch, key)
                independent_rows.append({
                    "record_index": record_index,
                    "time": time_index,
                    "sequence": sequence,
                    "module_index": index,
                    "module": MODULES[index][0],
                    "a1": ledgers["a1"],
                    "pidp": ledgers["pidp"],
                    "traffic": ledgers["traffic"],
                    "cache": ledgers["cache"],
                    "d3_cap12_pidp_total": ledgers["d3_cap12_pidp_total"],
                    "d3_cap12_misses": ledgers["d3_cap12_misses"],
                })

                if int(record["sequence_sample_id"]) in (0, 9) and \
                        time_index in (0, 4, 9):
                    count, errors, digest = independent_acc24_probes(
                        bits10[time_index], quantized[index], spec,
                        record_index, time_index)
                    acc_probe_count += count
                    acc_probe_mismatches += errors
                    acc_probe_digest.update(bytes.fromhex(digest))

        numeric_mismatch_count = sum(
            value for key, value in mismatch.items() if key != "samples")
        require(numeric_mismatch_count == 0, "independent row mismatch")
        require(len(independent_rows) == 1200, "independent row population mismatch")

        headline_rows = [row for row in independent_rows
                         if row["module_index"] in HEADLINE]
        a1_totals = {point: sum(row["a1"][point]["total"] for row in headline_rows)
                     for point in ("A1-SC8", "A1-ISO8", "A1-OSG")}
        fixed_a1 = min(a1_totals,
                       key=lambda point: (a1_totals[point],
                                          {"A1-OSG": 0, "A1-SC8": 1,
                                           "A1-ISO8": 2}[point]))
        require(fixed_a1 == "A1-OSG", "strongest fixed A1 drift")
        a1_headline = sum(row["a1"][fixed_a1]["total"] for row in headline_rows)
        pidp_headline = sum(row["pidp"]["total"] for row in headline_rows)
        full_ratio = ratio(a1_headline, pidp_headline)

        per_module = {}
        cache_entries_by_module = {}
        for index, spec in MODULES.items():
            selected = [row for row in independent_rows if row["module_index"] == index]
            name = spec[0]
            cache_values = sorted(set(row["cache"]["cache_entries"] for row in selected))
            require(len(cache_values) == 1, "cache entries vary within module")
            cache_entries_by_module[name] = cache_values[0]
            per_module[name] = {
                "rows": len(selected),
                "a1_cycles": sum(row["a1"][fixed_a1]["total"] for row in selected),
                "pidp_cycles": sum(row["pidp"]["total"] for row in selected),
                "a1_over_pidp": ratio(
                    sum(row["a1"][fixed_a1]["total"] for row in selected),
                    sum(row["pidp"]["total"] for row in selected)),
                "pidp_weight_refill_bytes": sum(
                    row["traffic"]["pidp_weight_refill_bytes"] for row in selected),
                "pidp_misses": sum(row["cache"]["pidp_misses"] for row in selected),
                "pidp_references": sum(
                    row["cache"]["pidp_references"] for row in selected),
                "min_active_tile_identities": min(
                    row["cache"]["active_tile_identities"] for row in selected),
                "max_active_tile_identities": max(
                    row["cache"]["active_tile_identities"] for row in selected),
            }

        per_sequence = {}
        for sequence in sorted(set(row["sequence"] for row in headline_rows)):
            selected = [row for row in headline_rows if row["sequence"] == sequence]
            a1_value = sum(row["a1"][fixed_a1]["total"] for row in selected)
            pidp_value = sum(row["pidp"]["total"] for row in selected)
            per_sequence[sequence] = {
                "a1_cycles": a1_value,
                "pidp_cycles": pidp_value,
                "a1_over_pidp": ratio(a1_value, pidp_value),
            }

        policy = {}
        for index, spec in MODULES.items():
            total_tiles = int(math.ceil(spec[1] / 16.0)) * spec[7]
            entries = cache_entries_by_module[spec[0]]
            policy[spec[0]] = {
                "total_weight_tile_identities": total_tiles,
                "cache_entries": entries,
                "use_pidp": total_tiles <= entries,
                "chosen": "PIDP" if total_tiles <= entries else fixed_a1,
            }
        selective_cycles = sum(
            row["pidp"]["total"] if policy[row["module"]]["use_pidp"]
            else row["a1"][fixed_a1]["total"] for row in headline_rows)
        selective_per_sequence = {}
        for sequence in per_sequence:
            selected = [row for row in headline_rows if row["sequence"] == sequence]
            selected_cycles = sum(
                row["pidp"]["total"] if policy[row["module"]]["use_pidp"]
                else row["a1"][fixed_a1]["total"] for row in selected)
            selective_per_sequence[sequence] = {
                "selective_cycles": selected_cycles,
                "a1_over_selective": ratio(per_sequence[sequence]["a1_cycles"],
                                            selected_cycles),
            }

        # Two receipt-blind sensitivities.  They are not replacements for the
        # frozen ledger: one charges the A1 source ingress at the same 128-bit
        # word granularity, and one deliberately reduces D3 cache below its
        # thirteen-tile static working set.
        a1_word_headline = sum(
            row["a1"][fixed_a1]["total"] - row["a1"][fixed_a1]["source_scan"] +
            row["pidp"]["source_stream"] for row in headline_rows)
        selective_word_cycles = sum(
            row["pidp"]["total"] if policy[row["module"]]["use_pidp"] else
            row["a1"][fixed_a1]["total"] - row["a1"][fixed_a1]["source_scan"] +
            row["pidp"]["source_stream"] for row in headline_rows)
        d3_cap12_selective_cycles = sum(
            row["d3_cap12_pidp_total"] if row["module_index"] == 3 else
            row["a1"][fixed_a1]["total"] for row in headline_rows)
        d3_rows = [row for row in headline_rows if row["module_index"] == 3]
        d3_equal15_pidp_cycles = sum(
            row["pidp"]["total"] +
            (row["pidp"]["group_service_optimistic"] // 10) * 5
            for row in d3_rows)
        selective_equal15_cycles = sum(
            (row["pidp"]["total"] +
             (row["pidp"]["group_service_optimistic"] // 10) * 5)
            if row["module_index"] == 3 else row["a1"][fixed_a1]["total"]
            for row in headline_rows)
        selective_word_equal15_cycles = selective_word_cycles + (
            selective_equal15_cycles - selective_cycles)

        compare_equal(a1_totals, report["a1_headline_cycle_totals"],
                      "report_a1_totals", mismatch, "report")
        compare_equal(full_ratio,
                      report["decision"]["headline_ratio_of_sums_a1_over_pidp"],
                      "report_full_ratio", mismatch, "report")
        compare_equal(per_sequence, report["per_sequence_headline"],
                      "report_per_sequence", mismatch, "report")
        for name in per_module:
            for field in ("a1_cycles", "pidp_cycles", "a1_over_pidp",
                          "pidp_weight_refill_bytes"):
                compare_equal(per_module[name][field], report["per_module"][name][field],
                              "report_per_module", mismatch, [name, field])
        report_selective = report["selective_weight_fit_composition_diagnostic"]
        compare_equal(selective_cycles, report_selective["headline_selective_cycles"],
                      "report_selective_cycles", mismatch, "report")
        compare_equal(ratio(a1_headline, selective_cycles),
                      report_selective["headline_a1_over_selective"],
                      "report_selective_ratio", mismatch, "report")
        handoff_equal15 = handoff["selective_diagnostic"].get(
            "equal_15_cycle_group_service_sensitivity", {})
        compare_equal(ratio(per_module["D3"]["a1_cycles"],
                            d3_equal15_pidp_cycles),
                      handoff_equal15.get("d3_a1_over_pidp"),
                      "handoff_equal15_d3_ratio", mismatch, "handoff")
        compare_equal(ratio(a1_headline, selective_equal15_cycles),
                      handoff_equal15.get("headline_a1_over_selective"),
                      "handoff_equal15_selective_ratio", mismatch, "handoff")
        for sequence in selective_per_sequence:
            compare_equal(selective_per_sequence[sequence], {
                "selective_cycles": report_selective["per_sequence"][sequence]["selective_cycles"],
                "a1_over_selective": report_selective["per_sequence"][sequence]["a1_over_selective"],
            }, "report_selective_sequence", mismatch, sequence)
        final_mismatch_count = sum(
            value for key, value in mismatch.items() if key != "samples")
        require(final_mismatch_count == 0, "report aggregate mismatch")

        result = {
            "schema": "m718_m712_pidp_independent_recompute_v1",
            "status": "PASS_RECEIPT_BLIND_INDEPENDENT_RECOMPUTE",
            "method": {
                "author_analyzer_imported": False,
                "author_analyzer_executed": False,
                "author_rows_used_as_input": False,
                "author_rows_used_only_as_output_comparison": True,
                "m699_bitpacks_independently_unpacked": 120,
                "independent_rows": len(independent_rows),
                "gpu_used": False,
                "eda_used": False,
                "lru_oracle_source_sha256": sha256(review_dir / "m718_lru_oracle.c"),
                "lru_oracle_compile_command": [
                    "/usr/bin/gcc", "-std=c99", "-O3", "-fPIC", "-shared",
                    "m718_lru_oracle.c", "-o", "<TEMP>/m718_lru_oracle.so"],
            },
            "identity": {
                "m712_result_report_sha256": sha256(result_dir / "report.json"),
                "m712_result_rows_sha256": sha256(result_dir / "rows.jsonl"),
                "m712_contract_sha256": sha256(contract_path),
                "m712_author_analyzer_sha256": sha256(author_analyzer),
                "m699_manifest_sha256": sha256(m699_dir / "manifest.json"),
                "m705_review_sha256": sha256(m705_dir / "review.json"),
                "m686_manifest_sha256": sha256(m686_dir / "manifest.json"),
                "m712_handoff_json_sha256": sha256(handoff_dir / "handoff.json"),
                "docs359_sha256": sha256(hw / "docs/359_DATE终局冻结_20260813.md"),
            },
            "seals": seals,
            "population": {
                "m699_records": len(manifest699["records"]),
                "m712_author_rows": len(rows),
                "independent_rows": len(independent_rows),
                "headline_rows": len(headline_rows),
                "record_distribution": record_population,
            },
            "topology": {MODULES[index][0]: {
                "legal_spatial_edges": topology["legal_spatial_edges"],
                "topology_mismatches": topology["topology_mismatches"],
                "aggregate_sha256": topology["aggregate_sha256"],
            } for index, topology in topologies.items()},
            "row_comparison": {
                "numeric_mismatch_count": final_mismatch_count,
                "mismatch_categories": {key: value for key, value in mismatch.items()
                                        if key != "samples"},
                "mismatch_samples": mismatch["samples"],
            },
            "acc24_independent_small_oracle": {
                "probe_count": acc_probe_count,
                "mismatches": acc_probe_mismatches,
                "digest_sha256": acc_probe_digest.hexdigest(),
                "probe_selection": "sequence samples 0 and 9, timesteps 0/4/9, four edge/center/hash positions, all four modules",
                "local_quant_identity": quant_identity,
                "checkpoint_numeric_admission": False,
            },
            "fixed_baseline": {
                "a1_headline_cycle_totals": a1_totals,
                "fixed_strongest": fixed_a1,
            },
            "full_pidp": {
                "a1_cycles": a1_headline,
                "pidp_cycles": pidp_headline,
                "a1_over_pidp": full_ratio,
                "per_sequence": per_sequence,
                "minimum_sequence_ratio": min(
                    value["a1_over_pidp"] for value in per_sequence.values()),
                "decision": "KILL_NO_RTL",
            },
            "per_module": per_module,
            "selective_static_composition": {
                "policy": policy,
                "runtime_or_record_or_sequence_oracle": False,
                "a1_cycles": a1_headline,
                "selective_cycles": selective_cycles,
                "a1_over_selective": ratio(a1_headline, selective_cycles),
                "per_sequence": selective_per_sequence,
            },
            "sensitivities_not_admission": {
                "pidp_equal_15_cycle_group_service": {
                    "d3_pidp_cycles": d3_equal15_pidp_cycles,
                    "d3_a1_over_pidp": ratio(per_module["D3"]["a1_cycles"],
                                             d3_equal15_pidp_cycles),
                    "selective_cycles": selective_equal15_cycles,
                    "a1_over_selective": ratio(a1_headline,
                                                selective_equal15_cycles),
                },
                "joint_a1_128bit_ingress_and_pidp_equal15_group_service": {
                    "baseline_cycles": a1_word_headline,
                    "selective_cycles": selective_word_equal15_cycles,
                    "a1_over_selective": ratio(a1_word_headline,
                                                selective_word_equal15_cycles),
                },
                "a1_128bit_source_ingress": {
                    "baseline_cycles": a1_word_headline,
                    "selective_cycles": selective_word_cycles,
                    "a1_over_selective": ratio(a1_word_headline,
                                                selective_word_cycles),
                },
                "d3_cache_capacity_12_below_13_tile_working_set": {
                    "selective_cycles": d3_cap12_selective_cycles,
                    "a1_over_selective": ratio(a1_headline,
                                                d3_cap12_selective_cycles),
                    "d3_misses": sum(row["d3_cap12_misses"] or 0
                                     for row in independent_rows),
                },
            },
            "author_handoff_crosscheck": {
                "status": handoff["status"],
                "full_pidp_ratio_matches":
                    handoff["full_pidp_decision"]["headline_a1_over_pidp"] == full_ratio,
                "selective_ratio_matches":
                    handoff["selective_diagnostic"]["headline_a1_over_selective"] ==
                    ratio(a1_headline, selective_cycles),
            },
        }
        require(acc_probe_mismatches == 0, "independent Acc24 oracle mismatch")
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        c_temporary.cleanup()


if __name__ == "__main__":
    main()
