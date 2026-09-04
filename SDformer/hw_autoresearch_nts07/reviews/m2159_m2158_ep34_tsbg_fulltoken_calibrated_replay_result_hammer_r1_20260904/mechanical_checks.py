#!/opt/anaconda3/bin/python
"""Independent, read-only M2159 audit of the sealed M2158 replay.

This checker does not import or invoke the M2145 production analyzer.  It
reimplements the frozen event/cache timing recurrence, reconstructs both VCS
calibration populations, then decodes every frame in the frozen ep34 capture
and recomputes every aggregate and breakdown reported by M2158.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import stat
import struct
import zlib

import numpy as np
from numba import njit, prange, set_num_threads


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RESULT_DIR = HW / (
    "results/m2158_m2145_ep34_tsbg_fulltoken_calibrated_replay_"
    "r1_20260904")
CAPTURE = HW / (
    "results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_"
    "binary_capture_s40_r1_20260901")
LOW_META = HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.json"
LOW_MEMH = HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.memh"
LOW_RESULT = HW / (
    "results/m2057_m2053_ep34_tsbg_full40_missing3_vcs_r1_20260903/"
    "result.json")
HIGH_META = HW / (
    "tb_m2018/fixtures/m2067_ep34_fc2_exact_continuation_s960.json")
HIGH_MEMH = HW / (
    "tb_m2018/fixtures/m2067_ep34_fc2_exact_continuation_s960.memh")
HIGH_RESULT = HW / (
    "results/m2067_ep34_fc2_exact_continuation_vcs_r10_codex_batch_"
    "20260904/result.json")
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED_SHA = {
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    CAPTURE / "fc_frames.bin": "dceb6c0c80b9c5898d10b4ad813fbcd7683fa80191b54b78eadaadda04a818b1",
    CAPTURE / "layers.json": "bd40c213f075ea3198f7145d25e9c96988701f46d5572c1e40d36e008feab08a",
    CAPTURE / "sample_order.json": "d4f1f6e140b531b972d53b48aa64e5f0aa5497b79d460616a0b3f89139a4f773",
    LOW_META: "3ac7048f0a97aeea0ac91627d303f4eea06b8a48bab816468825acfee180ccc5",
    LOW_MEMH: "487ca0073526b973220abd77c91d12dbc2420901443541ec5a79e36a780e1bf0",
    LOW_RESULT: "b4ee4f9cf4d55a4f722f1487ba4bc23948bc3f6a096178fa835d9ed18b50fe2a",
    HIGH_META: "5b44aa6a248a8768d59a85270a50b3ba805467377365e1b6e4ad8e58eafc7b34",
    HIGH_MEMH: "c617c6311ce44f15fb820f5dba5460ebd127235a13acd56724b56ccbb10cd594",
    HIGH_RESULT: "d0707b65f1c453636e3d6d050b036789f2366bcaa564f6fc32423aae3a128756",
}

MAGIC = b"M1558F01"
HEADER = struct.Struct("<8sHH11I")
CONTEXTS = 4
SOURCES = 16
PHYSICAL_GROUPS = 48
SLICES = 6
CACHE_ROWS = 4
START = 383
LOAD_PER_CHUNK = 768
EXPECTED_QUARTETS = 11_160_000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key {key} in {path}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          RuntimeError(f"non-finite JSON {value} in {path}")))


def regular(path: Path) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            f"not a regular non-symlink: {path}")


def verify_result_seal() -> dict:
    expected_nodes = {
        "result.json", "summary.json", "SHA256SUMS",
        "SHA256SUMS.seal.sha256",
    }
    actual_nodes = set()
    for node in RESULT_DIR.rglob("*"):
        rel = node.relative_to(RESULT_DIR).as_posix()
        actual_nodes.add(rel)
        regular(node)
    require(actual_nodes == expected_nodes,
            f"unexpected result nodes: {sorted(actual_nodes ^ expected_nodes)}")
    manifest = RESULT_DIR / "SHA256SUMS"
    outer = RESULT_DIR / "SHA256SUMS.seal.sha256"
    require(outer.read_text(encoding="ascii").split() ==
            [sha256(manifest), "SHA256SUMS"], "outer seal mismatch")
    listed = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, name = line.split(None, 1)
        name = name.strip().lstrip("*")
        require(name not in listed and name in {"result.json", "summary.json"},
                f"bad manifest member {name}")
        listed[name] = digest
        require(sha256(RESULT_DIR / name) == digest,
                f"manifest digest mismatch {name}")
    require(set(listed) == {"result.json", "summary.json"},
            "non-exhaustive result manifest")
    return {"result_json_sha256": listed["result.json"],
            "summary_json_sha256": listed["summary.json"],
            "manifest_sha256": sha256(manifest),
            "outer_file_sha256": sha256(outer)}


def dense_from_memh(lines: list[str], slot: int, groups: int) -> np.ndarray:
    physical = 48 if groups <= 48 else 192
    begin = slot * CONTEXTS * physical
    values = np.zeros((CONTEXTS, groups, SOURCES), dtype=np.int8)
    for context in range(CONTEXTS):
        for group in range(groups):
            word = int(lines[begin + context * physical + group], 16)
            active = word & 0xffff
            sign = (word >> 16) & 0xffff
            require(sign & ~active == 0, "calibration sign outside support")
            for lane in range(SOURCES):
                if active & (1 << lane):
                    values[context, group, lane] = (
                        -1 if sign & (1 << lane) else 1)
    return values


def descriptor_sha(values: np.ndarray) -> str:
    return hashlib.sha256(
        values.astype(np.int8, copy=False).tobytes()).hexdigest()


def scalar_axis(values: np.ndarray, mode: int) -> tuple[int, dict]:
    """Independent literal state walk for calibration, not production code."""
    groups = values.shape[1]
    order = ((c, g) for c in range(4) for g in range(groups)) if mode == 0 else (
        (c, g) for g in range(groups) for c in range(4))
    valid = [False] * 4
    keys = [0] * 4
    ages = [0] * 4
    clock = 1
    cycle = START
    hits = misses = evictions = issues = live = 0
    for context, group in order:
        row = values[context, group]
        lower = bool(np.any(row[:8]))
        upper = bool(np.any(row[8:]))
        if not lower and not upper:
            continue
        live += 1
        hit = next((i for i in range(4) if valid[i] and keys[i] == group), -1)
        if hit < 0:
            misses += 1
            victim = next((i for i in range(4) if not valid[i]), -1)
            if victim < 0:
                victim = min(range(4), key=lambda i: (ages[i], i))
                evictions += 1
            valid[victim] = True
            keys[victim] = group
            ages[victim] = clock + 1
        else:
            hits += 1
            ages[hit] = clock
        clock += 1
        cycle += 1
        if hit < 0:
            for _ in range(12):
                completions = []
                for bank in range(8):
                    accepted = cycle
                    while (accepted + 1 + 2 * bank) % 7 == 0:
                        accepted += 1
                    completions.append(accepted + 9 - bank)
                cycle = max(completions) + 1
        count = 6 * (int(lower) + int(upper))
        issues += count
        for _ in range(count):
            while cycle % 11 == 3:
                cycle += 1
            cycle += 1
    cycle += 1
    for _ in range(24):
        while cycle % 13 == 5:
            cycle += 1
        cycle += 1
    return cycle - START, {
        "hits": hits, "misses": misses, "evictions": evictions,
        "live_rows": live, "issues": issues,
        "scalar_reads": misses * 96,
    }


@njit(cache=False, parallel=True)
def batch_axis(lower: np.ndarray, upper: np.ndarray, mode: int) -> np.ndarray:
    """Independent batch implementation used only by this review."""
    population = lower.shape[0]
    groups = lower.shape[2]
    answer = np.zeros((population, 7), dtype=np.int64)
    for item in prange(population):
        valid = np.zeros(4, dtype=np.uint8)
        keys = np.zeros(4, dtype=np.int64)
        ages = np.zeros(4, dtype=np.int64)
        clock = 1
        cycle = START
        hits = 0
        misses = 0
        evictions = 0
        live = 0
        issues = 0
        for ordinal in range(4 * groups):
            if mode == 0:
                context = ordinal // groups
                group = ordinal - context * groups
            else:
                group = ordinal // 4
                context = ordinal - group * 4
            lo = lower[item, context, group]
            hi = upper[item, context, group]
            if not lo and not hi:
                continue
            live += 1
            hit = -1
            for way in range(4):
                if valid[way] and keys[way] == group:
                    hit = way
                    break
            if hit < 0:
                misses += 1
                victim = -1
                for way in range(4):
                    if not valid[way]:
                        victim = way
                        break
                if victim < 0:
                    victim = 0
                    for way in range(1, 4):
                        if (ages[way] < ages[victim] or
                                (ages[way] == ages[victim] and way < victim)):
                            victim = way
                    evictions += 1
                valid[victim] = 1
                keys[victim] = group
                ages[victim] = clock + 1
            else:
                hits += 1
                ages[hit] = clock
            clock += 1
            cycle += 1
            if hit < 0:
                for _ in range(12):
                    last = 0
                    for bank in range(8):
                        accepted = cycle
                        while (accepted + 1 + 2 * bank) % 7 == 0:
                            accepted += 1
                        done = accepted + 9 - bank
                        if done > last:
                            last = done
                    cycle = last + 1
            issue_count = 6 * (int(lo) + int(hi))
            issues += issue_count
            for _ in range(issue_count):
                while cycle % 11 == 3:
                    cycle += 1
                cycle += 1
        cycle += 1
        for _ in range(24):
            while cycle % 13 == 5:
                cycle += 1
            cycle += 1
        answer[item, 0] = cycle - START
        answer[item, 1] = hits
        answer[item, 2] = misses
        answer[item, 3] = evictions
        answer[item, 4] = live
        answer[item, 5] = issues
        answer[item, 6] = misses * 96
    return answer


def high_backbone(values: np.ndarray, tiles: int, mode: int) -> int:
    total = 0
    for begin in range(0, values.shape[1], 48):
        cycles, _ = scalar_axis(values[:, begin:begin + 48], mode)
        total += LOAD_PER_CHUNK + cycles
    return total * tiles


def reconstruct_calibration() -> dict:
    low_meta = strict_json(LOW_META)
    low_result = strict_json(LOW_RESULT)
    low_lines = LOW_MEMH.read_text(encoding="ascii").splitlines()
    require(len(low_meta["rows"]) == len(low_result["rows"]) == 1920,
            "low calibration population")
    low_mismatches = 0
    for slot, observed in enumerate(low_result["rows"]):
        values = dense_from_memh(low_lines, slot, 48)
        for mode, field in ((0, "base_cycles"), (1, "tsbg_cycles")):
            predicted, _ = scalar_axis(values, mode)
            low_mismatches += int(predicted != int(observed[field]))
    require(low_mismatches == 0, "independent G<=48 recurrence mismatch")

    high_meta = strict_json(HIGH_META)
    high_result = strict_json(HIGH_RESULT)
    high_lines = HIGH_MEMH.read_text(encoding="ascii").splitlines()
    require(len(high_meta["rows"]) == len(high_result["rows"]) == 960,
            "high calibration population")
    registries = {0: {}, 1: {}}
    locations = defaultdict(list)
    residual_lists = defaultdict(list)
    duplicate_axis_rows = 0
    for slot, observed in enumerate(high_result["rows"]):
        meta = high_meta["rows"][slot]
        groups = int(meta["source_groups"])
        tiles = int(meta["output_tiles"])
        values = dense_from_memh(high_lines, slot, groups)
        digest = descriptor_sha(values)
        key = (groups, tiles, digest)
        row_residual = {}
        for mode, field in ((0, "base_cycles"), (1, "tsbg_cycles")):
            backbone = high_backbone(values, tiles, mode)
            residual = int(observed[field]) - backbone
            if key in registries[mode]:
                require(registries[mode][key] == residual,
                        "conflicting duplicate calibration residual")
                duplicate_axis_rows += 1
            registries[mode][key] = residual
            residual_lists[(groups, mode)].append(residual)
            row_residual[mode] = residual
            require(backbone + registries[mode][key] == int(observed[field]),
                    "high calibrated reconstruction mismatch")
        locations[(int(meta["sample_id"]), int(meta["layer_id"]))].append({
            "token_start": int(meta["token_start"]),
            "digest": digest,
            "ordinary_residual": row_residual[0],
            "tsbg_residual": row_residual[1],
        })
    profiles = {}
    for (groups, mode), values in sorted(residual_lists.items()):
        ordered = sorted(values)
        profiles[f"G{groups}_{'ordinary' if mode == 0 else 'tsbg'}"] = {
            "count": len(values), "minimum": min(values),
            "median": int(ordered[len(ordered) // 2]),
            "maximum": max(values),
            "histogram": {str(k): v for k, v in sorted(Counter(values).items())},
        }
    return {
        "low_rows": 1920, "low_cycle_fields": 3840,
        "low_mismatches": low_mismatches,
        "high_rows": 960, "high_cycle_fields": 1920,
        "duplicate_axis_rows": duplicate_axis_rows,
        "profiles": profiles, "registries": registries,
        "locations": dict(locations),
    }


class Totals:
    def __init__(self):
        self.rows = self.base = self.tsbg = 0
        self.base_low = self.tsbg_high = 0
        self.read_base = self.read_tsbg = 0
        self.hit_base = self.hit_tsbg = 0
        self.miss_base = self.miss_tsbg = 0
        self.evict_base = self.evict_tsbg = 0
        self.empty = self.slower = self.equal = 0
        self.exact_hits = self.median = 0

    def add(self, base, tsbg, base_low, tsbg_high, ledger_base,
            ledger_tsbg, empty, exact, median):
        self.rows += int(base.size)
        self.base += int(np.sum(base, dtype=np.int64))
        self.tsbg += int(np.sum(tsbg, dtype=np.int64))
        self.base_low += int(np.sum(base_low, dtype=np.int64))
        self.tsbg_high += int(np.sum(tsbg_high, dtype=np.int64))
        self.read_base += int(np.sum(ledger_base[:, 6], dtype=np.int64))
        self.read_tsbg += int(np.sum(ledger_tsbg[:, 6], dtype=np.int64))
        self.hit_base += int(np.sum(ledger_base[:, 1], dtype=np.int64))
        self.hit_tsbg += int(np.sum(ledger_tsbg[:, 1], dtype=np.int64))
        self.miss_base += int(np.sum(ledger_base[:, 2], dtype=np.int64))
        self.miss_tsbg += int(np.sum(ledger_tsbg[:, 2], dtype=np.int64))
        self.evict_base += int(np.sum(ledger_base[:, 3], dtype=np.int64))
        self.evict_tsbg += int(np.sum(ledger_tsbg[:, 3], dtype=np.int64))
        self.empty += int(np.count_nonzero(empty))
        self.slower += int(np.count_nonzero(tsbg > base))
        self.equal += int(np.count_nonzero(tsbg == base))
        self.exact_hits += int(np.count_nonzero(exact))
        self.median += int(np.count_nonzero(median))

    def report(self, ratios: np.ndarray) -> dict:
        require(ratios.size == self.rows and self.base > 0 and self.tsbg > 0,
                "statistics population mismatch")
        return {
            "aligned_b4_quartets": self.rows,
            "ordinary_cycles": self.base,
            "tsbg_cycles": self.tsbg,
            "ratio_of_sums": self.base / self.tsbg,
            "time_reduction_fraction": 1.0 - self.tsbg / self.base,
            "p10_workload_ratio": float(np.percentile(ratios, 10)),
            "p50_workload_ratio": float(np.percentile(ratios, 50)),
            "p90_workload_ratio": float(np.percentile(ratios, 90)),
            "worst_workload_ratio": float(np.min(ratios)),
            "slower_case_rate": self.slower / self.rows,
            "slower_cases": self.slower,
            "equal_cases": self.equal,
            "empty_quartets": self.empty,
            "ordinary_scalar_weight_reads": self.read_base,
            "tsbg_scalar_weight_reads": self.read_tsbg,
            "weight_read_reduction_fraction": 1.0 - self.read_tsbg / self.read_base,
            "ordinary_cache": {"hits": self.hit_base,
                               "misses": self.miss_base,
                               "evictions": self.evict_base},
            "tsbg_cache": {"hits": self.hit_tsbg,
                           "misses": self.miss_tsbg,
                           "evictions": self.evict_tsbg},
            "schedule_fallback": {
                "implemented": False, "fallbacks": 0,
                "reason": "M2018 SCHEDULE_MODE is elaboration-time; no dynamic fallback is claimed",
            },
            "continuation_exact_calibration_hits": self.exact_hits,
            "continuation_median_residual_extrapolations": self.median,
            "pessimistic_residual_envelope_ratio": self.base_low / self.tsbg_high,
        }


def compare_tree(actual, expected, path="root") -> None:
    require(type(actual) is type(expected), f"type mismatch at {path}")
    if isinstance(actual, dict):
        require(set(actual) == set(expected), f"key mismatch at {path}")
        for key in actual:
            compare_tree(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, list):
        require(len(actual) == len(expected), f"length mismatch at {path}")
        for index, value in enumerate(actual):
            compare_tree(value, expected[index], f"{path}[{index}]")
    elif isinstance(actual, float):
        require(np.isfinite(actual) and np.isfinite(expected) and
                abs(actual - expected) <= 1e-12 * max(1.0, abs(expected)),
                f"float mismatch at {path}: {actual} != {expected}")
    else:
        require(actual == expected, f"value mismatch at {path}: {actual} != {expected}")


def decode_frame_payload(raw: bytes, tokens: int, channels: int,
                         row_bytes: int, nnz_total: int):
    matrix_bytes = tokens * row_bytes
    require(len(raw) == 3 * matrix_bytes + 2 * tokens + nnz_total,
            "payload length mismatch")
    masks = []
    offset = 0
    for _ in range(3):
        packed = np.frombuffer(raw[offset:offset + matrix_bytes], dtype=np.uint8)
        bits = np.unpackbits(packed.reshape(tokens, row_bytes), axis=1,
                             bitorder="little")
        require(bool(np.all(bits[:, channels:] == 0)), "nonzero padding bits")
        masks.append(bits[:, :channels].astype(bool))
        offset += matrix_bytes
    support, sign, nonunit = masks
    counts = np.frombuffer(raw[offset:offset + 2 * tokens], dtype="<u2")
    offset += 2 * tokens
    codes = np.frombuffer(raw[offset:], dtype=np.int8)
    require(bool(np.all(sign <= support)) and
            bool(np.all(nonunit <= support)) and
            bool(np.all(counts == np.sum(support, axis=1))) and
            int(np.sum(counts)) == nnz_total and codes.size == nnz_total and
            bool(np.all(codes != 0)) and
            bool(np.all((codes < 0) == sign[support])) and
            not bool(np.any(nonunit)), "descriptor semantic mismatch")
    return support, sign, codes


def independently_replay(calibration: dict, observed: dict) -> dict:
    layers = [row for row in strict_json(CAPTURE / "layers.json")["layers"]
              if row["target"] in ("FC1", "FC2")]
    samples = strict_json(CAPTURE / "sample_order.json")["samples"]
    require(len(layers) == 24 and len(samples) == 40 and
            sum(x["target"] == "FC1" for x in layers) == 12 and
            sum(x["target"] == "FC2" for x in layers) == 12 and
            len({x["sequence"] for x in samples}) == 4,
            "capture inventory mismatch")
    layer_by_id = {int(row["layer_id"]): row for row in layers}
    sample_by_id = {int(row["global_sample_id"]): row for row in samples}
    expected_pairs = [(int(sample["global_sample_id"]), int(layer["layer_id"]))
                      for sample in samples for layer in layers]
    sequences = sorted({x["sequence"] for x in samples})
    sequence_code = {name: index for index, name in enumerate(sequences)}

    ratios = np.empty(EXPECTED_QUARTETS, dtype=np.float64)
    layer_labels = np.empty(EXPECTED_QUARTETS, dtype=np.uint8)
    sequence_labels = np.empty(EXPECTED_QUARTETS, dtype=np.uint8)
    target_labels = np.empty(EXPECTED_QUARTETS, dtype=np.uint8)
    aggregate = Totals()
    by_layer = {str(row["layer_id"]): Totals() for row in layers}
    by_sequence = {name: Totals() for name in sequences}
    by_target = {name: Totals() for name in ("FC1", "FC2")}

    pair_index = 0
    frame_index_expected = 0
    token_start_expected = 0
    frames = 0
    cursor = 0
    exact_locations_seen = set()
    with (CAPTURE / "fc_frames.bin").open("rb") as stream:
        while True:
            packed_header = stream.read(HEADER.size)
            if not packed_header:
                break
            require(len(packed_header) == HEADER.size, "truncated frame header")
            (magic, version, header_size, layer_id, sample_id, frame_index,
             token_start, token_count, channels, row_bytes, nnz_total,
             raw_bytes, compressed_bytes, crc32) = HEADER.unpack(packed_header)
            require(pair_index < len(expected_pairs) and
                    (sample_id, layer_id) == expected_pairs[pair_index] and
                    magic == MAGIC and version == 1 and
                    header_size == HEADER.size and
                    frame_index == frame_index_expected and
                    token_start == token_start_expected and token_count % 4 == 0,
                    "noncanonical frame identity/order")
            layer = layer_by_id[layer_id]
            groups = int(layer["weight_layout"]["source_group_count"])
            tiles = int(layer["weight_layout"]["output_tile_count"])
            require(channels == int(layer["input_channels"]) == groups * 16 and
                    row_bytes == (channels + 7) // 8,
                    "frame geometry mismatch")
            compressed = stream.read(compressed_bytes)
            require(len(compressed) == compressed_bytes,
                    "truncated compressed payload")
            decoder = zlib.decompressobj()
            raw = decoder.decompress(compressed) + decoder.flush()
            require(decoder.eof and not decoder.unused_data and
                    not decoder.unconsumed_tail and len(raw) == raw_bytes and
                    (zlib.crc32(raw) & 0xffffffff) == crc32,
                    "zlib/CRC mismatch")
            support, sign, codes = decode_frame_payload(
                raw, token_count, channels, row_bytes, nnz_total)
            quartets = support.reshape(token_count // 4, 4, groups, 16)
            lower = np.any(quartets[:, :, :, :8], axis=3)
            upper = np.any(quartets[:, :, :, 8:], axis=3)
            population = quartets.shape[0]

            ledgers = []
            cycles = []
            for mode in (0, 1):
                total_cycles = np.zeros(population, dtype=np.int64)
                total_ledger = np.zeros((population, 7), dtype=np.int64)
                for chunk_begin in range(0, groups, 48):
                    chunk_end = min(chunk_begin + 48, groups)
                    ledger = batch_axis(lower[:, :, chunk_begin:chunk_end],
                                        upper[:, :, chunk_begin:chunk_end], mode)
                    total_cycles += LOAD_PER_CHUNK + ledger[:, 0]
                    total_ledger += ledger
                cycles.append(total_cycles * tiles)
                ledgers.append(total_ledger * tiles)
            # Keep the uncorrected backbone arrays distinct: exact VCS-keyed
            # locations replace (rather than add to) the median correction.
            base = cycles[0].copy()
            tsbg = cycles[1].copy()
            base_low = base.copy()
            tsbg_high = tsbg.copy()
            exact = np.zeros(population, dtype=bool)
            median = np.zeros(population, dtype=bool)
            if groups > 48:
                ordinary_profile = calibration["profiles"][f"G{groups}_ordinary"]
                tsbg_profile = calibration["profiles"][f"G{groups}_tsbg"]
                base += ordinary_profile["median"]
                tsbg += tsbg_profile["median"]
                base_low += ordinary_profile["minimum"]
                tsbg_high += tsbg_profile["maximum"]
                median[:] = True
                frame_end = token_start + token_count
                for location in calibration["locations"].get(
                        (int(sample_id), int(layer_id)), []):
                    selected = location["token_start"]
                    if token_start <= selected < frame_end:
                        require((selected - token_start) % 4 == 0,
                                "unaligned calibration location")
                        index = (selected - token_start) // 4
                        dense = np.zeros((4, groups, 16), dtype=np.int8)
                        selected_support = quartets[index]
                        selected_sign = sign.reshape(
                            token_count // 4, 4, groups, 16)[index]
                        dense[selected_support] = 1
                        dense[selected_sign] = -1
                        require(descriptor_sha(dense) == location["digest"],
                                "capture/calibration descriptor mismatch")
                        base[index] = (cycles[0][index] +
                                       location["ordinary_residual"])
                        tsbg[index] = (cycles[1][index] +
                                       location["tsbg_residual"])
                        exact[index] = True
                        median[index] = False
                        exact_locations_seen.add(
                            (int(sample_id), int(layer_id), int(selected)))
            empty = ~np.any(lower | upper, axis=(1, 2))
            row_ratios = base / tsbg
            end = cursor + population
            ratios[cursor:end] = row_ratios
            layer_labels[cursor:end] = int(layer_id)
            sequence = sample_by_id[sample_id]["sequence"]
            sequence_labels[cursor:end] = sequence_code[sequence]
            target_labels[cursor:end] = 0 if layer["target"] == "FC1" else 1
            for collector in (aggregate, by_layer[str(layer_id)],
                              by_sequence[sequence], by_target[layer["target"]]):
                collector.add(base, tsbg, base_low, tsbg_high, ledgers[0],
                              ledgers[1], empty, exact, median)
            cursor = end
            frames += 1
            token_start_expected += token_count
            require(token_start_expected <= int(layer["tokens_per_call"]),
                    "pair token overflow")
            if token_start_expected == int(layer["tokens_per_call"]):
                pair_index += 1
                frame_index_expected = 0
                token_start_expected = 0
            else:
                frame_index_expected += 1
    require(pair_index == len(expected_pairs) and token_start_expected == 0 and
            cursor == EXPECTED_QUARTETS and aggregate.rows == EXPECTED_QUARTETS,
            "incomplete full-token capture")
    require(len(exact_locations_seen) == 960,
            "not all high-group calibration locations found")

    recomputed = aggregate.report(ratios)
    breakdown = {"layer_id": {}, "sequence": {}, "target": {}}
    for row in layers:
        key = str(row["layer_id"])
        breakdown["layer_id"][key] = by_layer[key].report(
            ratios[layer_labels == int(row["layer_id"])])
    for name, code in sequence_code.items():
        breakdown["sequence"][name] = by_sequence[name].report(
            ratios[sequence_labels == code])
    for name, code in (("FC1", 0), ("FC2", 1)):
        breakdown["target"][name] = by_target[name].report(
            ratios[target_labels == code])
    compare_tree(recomputed, observed["aggregate"], "aggregate")
    compare_tree(breakdown, observed["breakdown"], "breakdown")
    return {
        "frames": frames,
        "pairs": pair_index,
        "exact_locations_seen": len(exact_locations_seen),
        "aggregate": recomputed,
        "breakdown": breakdown,
    }


def main() -> int:
    set_num_threads(3)
    seals = verify_result_seal()
    for path, digest in EXPECTED_SHA.items():
        regular(path)
        require(sha256(path) == digest, f"identity drift {path}")
    result = strict_json(RESULT_DIR / "result.json")
    summary = strict_json(RESULT_DIR / "summary.json")
    require(summary == {key: result[key] for key in
                        ("status", "population", "aggregate", "claim_boundary")},
            "summary is not exact result projection")
    require(result["status"] ==
            "CPU_MODEL_PASS_PENDING_INDEPENDENT_RESULT_HAMMER_DO_NOT_CITE",
            "pre-review status drift")
    require(result["population"] == {
        "checkpoint": "motion_ep34_live93", "sequences": 4, "samples": 40,
        "layers": 24, "fc1_layers": 12, "fc2_layers": 12,
        "aligned_b4_quartets": EXPECTED_QUARTETS, "frames": 11040,
        "all_aligned_b4_tokens": True,
    }, "population declaration drift")
    required_false = {"hardware_weight_values", "rtl_execution", "same_area",
                      "power", "energy", "full_network", "system_speedup",
                      "fps", "paper_admitted", "headline"}
    require(all(result["claim_boundary"].get(key) is False
                for key in required_false), "unsafe claim flag")

    calibration = reconstruct_calibration()
    require(calibration["low_mismatches"] == 0 and
            calibration["low_cycle_fields"] +
            calibration["high_cycle_fields"] == 5760,
            "calibration exactness gate")
    require(calibration["profiles"] ==
            result["calibration"]["continuation_residual_profiles"],
            "residual profile mismatch")
    require(calibration["duplicate_axis_rows"] ==
            result["calibration"]["continuation_duplicate_descriptor_axis_rows"],
            "duplicate calibration count mismatch")

    replay = independently_replay(calibration, result)
    aggregate = replay["aggregate"]
    require(aggregate["continuation_exact_calibration_hits"] == 960 and
            aggregate["continuation_median_residual_extrapolations"] == 779040,
            "continuation hit/extrapolation census")
    require(aggregate["ordinary_scalar_weight_reads"] ==
            aggregate["ordinary_cache"]["misses"] * 96 and
            aggregate["tsbg_scalar_weight_reads"] ==
            aggregate["tsbg_cache"]["misses"] * 96,
            "read/miss accounting mismatch")
    sequence_ratios = [row["ratio_of_sums"] for row in
                       replay["breakdown"]["sequence"].values()]
    output = {
        "status": "PASS_M2159_INDEPENDENT_FULL_RECOMPUTE",
        "result_identity": seals,
        "population": {
            "frames": replay["frames"], "sample_layer_pairs": replay["pairs"],
            "aligned_b4_quartets": aggregate["aligned_b4_quartets"],
            "high_group_exact_locations": replay["exact_locations_seen"],
            "high_group_median_extrapolations":
                aggregate["continuation_median_residual_extrapolations"],
        },
        "calibration": {
            "g_le_48_rows": calibration["low_rows"],
            "g_le_48_axis_cycle_fields": calibration["low_cycle_fields"],
            "g_le_48_mismatches": calibration["low_mismatches"],
            "continuation_rows": calibration["high_rows"],
            "continuation_axis_cycle_fields": calibration["high_cycle_fields"],
            "continuation_duplicate_axis_rows":
                calibration["duplicate_axis_rows"],
            "residual_profiles": calibration["profiles"],
        },
        "independent_aggregate": aggregate,
        "diagnostics": {
            "sequence_ratio_min": min(sequence_ratios),
            "sequence_ratio_max": max(sequence_ratios),
            "high_group_quartets": 780000,
            "high_group_median_extrapolation_fraction_within_high_group":
                779040 / 780000,
            "high_group_median_extrapolation_fraction_of_all":
                779040 / EXPECTED_QUARTETS,
            "conservative_ratio_delta_fraction":
                1.0 - aggregate["pessimistic_residual_envelope_ratio"] /
                aggregate["ratio_of_sums"],
        },
        "comparison": {
            "aggregate_fields_match": True,
            "all_breakdown_fields_match": True,
            "summary_projection_matches": True,
            "all_result_nodes_regular_and_sealed": True,
        },
        "execution_boundary": {
            "production_analyzer_imported": False,
            "production_analyzer_invoked": False,
            "result_modified": False,
            "vcs_simv_eda_gpu_license_runs": 0,
            "review_cpu_workers": 3,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
