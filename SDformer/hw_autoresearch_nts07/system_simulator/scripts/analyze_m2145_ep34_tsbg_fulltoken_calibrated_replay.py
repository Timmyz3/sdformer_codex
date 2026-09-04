#!/opt/anaconda3/bin/python
"""M2145 source-only full-token calibrated TSBG replay.

This is a CPU cycle model, not RTL execution.  Before decoding any production
frame it must reproduce every ordinary and TSBG cycle field in the frozen
M2057 (1,920 G<=48 rows) and M2101/M2112 (960 G96/G192 rows) VCS evidence.

The G<=48 calculator is an event recurrence of the frozen M2018 state machine,
M803 eight-bank timing model, LRU4 nonblocking-age rule, and the frozen 11/13
cycle bridge/commit backpressure.  The continuation calculator uses the same
recurrence per physical G48 chunk, the observed 768-cycle/chunk wrapper load
charge, and an explicit descriptor-keyed residual registry for exact
calibration rows.  Unseen continuation descriptors use the median residual
for the same G96/G192 geometry; min/max observed residuals form a sensitivity
interval.  The output says this plainly and never promotes the replay to RTL,
same-area, power, energy, full-network, or system-speedup evidence.

Authoring and self-checking this file never invokes VCS, simv, EDA, GPU, a
license query, or the production replay.  ``--run`` is a later CPU-only action
that requires an independently admitted source package and a fresh output.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import struct
import tempfile
import zlib

import numpy as np
from numba import njit, prange, set_num_threads


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = Path(__file__).resolve()
CONTRACT = HW / (
    "contracts/m2145_ep34_tsbg_fulltoken_calibrated_replay_source_"
    "contract_r1_20260904.json")
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
CAPTURE = HW / (
    "results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_"
    "binary_capture_s40_r1_20260901")
M2051_META = HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.json"
M2051_MEMH = HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.memh"
M2057 = HW / "results/m2057_m2053_ep34_tsbg_full40_missing3_vcs_r1_20260903"
M2057_HAMMER = HW / (
    "reviews/m2057_m2053_ep34_tsbg_full40_missing3_vcs_result_hammer_"
    "r1_20260903")
M2067_META = HW / "tb_m2018/fixtures/m2067_ep34_fc2_exact_continuation_s960.json"
M2067_MEMH = HW / "tb_m2018/fixtures/m2067_ep34_fc2_exact_continuation_s960.memh"
M2101 = HW / "results/m2067_ep34_fc2_exact_continuation_vcs_r10_codex_batch_20260904"
M2112 = HW / (
    "reviews/m2112_m2067_ep34_fc2_exact_continuation_vcs_r10_result_"
    "hammer_r1_20260904")
SOURCE_HAMMER = HW / (
    "reviews/m2157_m2145_ep34_tsbg_fulltoken_calibrated_replay_source_"
    "hammer_r1_20260904")

EXPECTED = {
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    CAPTURE / "SHA256SUMS": "be0b89f9b8084baf0c2cd959805530a4e4f41c437a446335142a65bd73960a8f",
    CAPTURE / "SHA256SUMS.seal.sha256": "8d63d1054452377836c333c9848f771a6fc964e4ed4b00ed9a22f1537bd73c85",
    CAPTURE / "fc_frames.bin": "dceb6c0c80b9c5898d10b4ad813fbcd7683fa80191b54b78eadaadda04a818b1",
    CAPTURE / "layers.json": "bd40c213f075ea3198f7145d25e9c96988701f46d5572c1e40d36e008feab08a",
    CAPTURE / "sample_order.json": "d4f1f6e140b531b972d53b48aa64e5f0aa5497b79d460616a0b3f89139a4f773",
    M2051_META: "3ac7048f0a97aeea0ac91627d303f4eea06b8a48bab816468825acfee180ccc5",
    M2051_MEMH: "487ca0073526b973220abd77c91d12dbc2420901443541ec5a79e36a780e1bf0",
    M2057 / "SHA256SUMS": "f00ab87e69043ed1eaa15980728c3858001122e47e5ff621dcf238eb5aeba971",
    M2057 / "SHA256SUMS.seal.sha256": "3bd2d119e72792f75c636ca82856305151f08d02d15418b675139f504fb51df2",
    M2057 / "result.json": "b4ee4f9cf4d55a4f722f1487ba4bc23948bc3f6a096178fa835d9ed18b50fe2a",
    M2057_HAMMER / "SHA256SUMS": "e46ab0e15e44dcbc6ce103b078e56572366886d5e5345c2a2c5c2c5a87e9ab6d",
    M2057_HAMMER / "SHA256SUMS.seal.sha256": "03dfa96a150ee621c06fa5fd07aa3e3fdd3ff5051a99fd60507747c97b8aca14",
    M2057_HAMMER / "review.json": "a16a32cccb79d22895a27225b2d0390a172aadf20324a577f23c7422364e7b6f",
    M2067_META: "5b44aa6a248a8768d59a85270a50b3ba805467377365e1b6e4ad8e58eafc7b34",
    M2101 / "SHA256SUMS": "4715824b02f34851c4786c27a6578a8cec1fe04f07b1df7676187f949cb3803e",
    M2101 / "SHA256SUMS.seal.sha256": "4c5e1937be8224b1cb05f1d86fcfdeb5ccb48d08b6f33c92a8d24baec1d7c507",
    M2101 / "result.json": "d0707b65f1c453636e3d6d050b036789f2366bcaa564f6fc32423aae3a128756",
    M2112 / "SHA256SUMS": "83222503729473eac920abaa496098f66b288759d5e29ac05a1f2b22909e4985",
    M2112 / "SHA256SUMS.seal.sha256": "8c7de55886a8d5bcbd1fdf95dd94572c4fd342088cb10376e6cee364e5a6db31",
    M2112 / "review.json": "08677bd464a13bace47dec3c2fe9b9bbd7e55702f7a25cdfc6be1c1774ce4280",
}
M2067_MEMH_SHA256 = "c617c6311ce44f15fb820f5dba5460ebd127235a13acd56724b56ccbb10cd594"

FRAME_MAGIC = b"M1558F01"
FRAME_VERSION = 1
FRAME_HEADER = struct.Struct("<8sHH11I")
CONTEXTS = 4
PHYSICAL_GROUPS = 48
SOURCES = 16
SLICES = 6
CACHE_ROWS = 4
CANONICAL_START = 383
CANONICAL_MEMORY_START = 384
WRAPPER_LOAD_CYCLES_PER_CHUNK = 768
EXPECTED_SAMPLES = 40
EXPECTED_SEQUENCES = 4
EXPECTED_LAYERS = 24
EXPECTED_FC1 = 12
EXPECTED_FC2 = 12
EXPECTED_QUARTETS = 11_160_000


class M2145Error(RuntimeError):
    pass


def need(value: bool, message: str) -> None:
    if not value:
        raise M2145Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> object:
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          M2145Error("nonfinite JSON: " + token)))


def exact(path: Path, expected: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise M2145Error("missing input " + str(path)) from exc
    need(stat.S_ISREG(mode) and not path.is_symlink(),
         "input must be regular non-symlink: " + str(path))
    need(sha256(path) == expected, "identity drift: " + str(path))


def regular_non_symlink(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise M2145Error("missing " + label + ": " + str(path)) from exc
    need(stat.S_ISREG(mode) and not path.is_symlink(),
         label + " must be regular non-symlink: " + str(path))


def verify_double_seal(directory: Path) -> None:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    regular_non_symlink(manifest, "seal manifest")
    regular_non_symlink(outer, "outer seal")
    tokens = outer.read_text(encoding="ascii").split()
    need(tokens == [sha256(manifest), "SHA256SUMS"],
         "outer seal mismatch: " + str(directory))
    listed = set()
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, name = line.split(None, 1)
        rel = Path(name.strip().lstrip("*"))
        need(not rel.is_absolute() and ".." not in rel.parts,
             "unsafe seal member")
        exact(directory / rel, digest)
        listed.add(rel.as_posix())
    actual = {path.relative_to(directory).as_posix()
              for path in directory.rglob("*")
              if path.is_file() and path.name not in {
                  "SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    need(actual == listed, "non-exhaustive sealed directory: " + str(directory))


def verify_inputs(require_contract: bool = False) -> dict:
    for path, digest in EXPECTED.items():
        exact(path, digest)
    exact(M2067_MEMH, M2067_MEMH_SHA256)
    for directory in (CAPTURE, M2057, M2057_HAMMER, M2101, M2112):
        verify_double_seal(directory)
    low_review = strict_json(M2057_HAMMER / "review.json")
    high_review = strict_json(M2112 / "review.json")
    low_severity = {key.lower(): int(value) for key, value in
                    low_review["severity_counts"].items()}
    high_severity = {key.lower(): int(value) for key, value in
                     high_review["severity_counts"].items()}
    need(low_review["status"].startswith("PASS_") and
         low_severity["p0"] == 0 and low_severity["p1"] == 0 and
         high_review["status"].startswith("PASS_") and
         high_severity["p0"] == 0 and high_severity["p1"] == 0,
         "independent VCS result authority drift")
    contract_sha = None
    if require_contract:
        regular_non_symlink(CONTRACT, "M2145 contract")
        contract = strict_json(CONTRACT)
        need(contract["source"]["path"] == str(SOURCE.relative_to(ROOT)) and
             contract["source"]["sha256"] == sha256(SOURCE),
             "contract/source binding mismatch")
        side = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256")
        outer = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256.seal.sha256")
        regular_non_symlink(side, "contract sidecar")
        regular_non_symlink(outer, "contract outer seal")
        need(outer.read_text(encoding="ascii").split() ==
             [sha256(side), side.name], "contract outer seal mismatch")
        need(side.read_text(encoding="ascii").split() ==
             [sha256(CONTRACT), CONTRACT.name], "contract sidecar mismatch")
        contract_sha = sha256(CONTRACT)
        verify_double_seal(SOURCE_HAMMER)
        review = strict_json(SOURCE_HAMMER / "review.json")
        need(review["status"] ==
             "PASS_M2145_SOURCE_RELEASE_FOR_CPU_REPLAY" and
             review["severity_counts"]["P0"] == 0 and
             review["severity_counts"]["P1"] == 0 and
             review["authorization"]["production_cpu_replay"] is True and
             review["binding"]["source_sha256"] == sha256(SOURCE) and
             review["binding"]["contract_sha256"] == contract_sha,
             "independent source hammer has not released production replay")
    return {"docs359_sha256": sha256(DOC359),
            "capture_sha256": sha256(CAPTURE / "fc_frames.bin"),
            "contract_sha256": contract_sha}


def dense_from_memh(lines: list[str], slot: int, groups: int) -> np.ndarray:
    physical = 48 if groups <= 48 else 192
    offset = slot * CONTEXTS * physical
    out = np.zeros((CONTEXTS, groups, SOURCES), dtype=np.int8)
    for context in range(CONTEXTS):
        for group in range(groups):
            word = int(lines[offset + context * physical + group], 16)
            active = word & 0xffff
            sign = (word >> 16) & 0xffff
            need(sign & ~active == 0, "sign outside activity")
            for lane in range(SOURCES):
                if active & (1 << lane):
                    out[context, group, lane] = -1 if sign & (1 << lane) else 1
    return out


def bundle_completion(request_cycle: int, start_cycle: int,
                      memory_start: int) -> int:
    offset = (memory_start - start_cycle) % 7
    completions = []
    for bank in range(8):
        accepted = request_cycle
        while (accepted + offset + bank * 2) % 7 == 0:
            accepted += 1
        completions.append(accepted + (8 - bank) + 1)
    return max(completions)


def cache_and_order(values: np.ndarray, mode: int) -> tuple[list, dict]:
    need(values.shape == (CONTEXTS, PHYSICAL_GROUPS, SOURCES),
         "physical descriptor shape")
    rows = []
    iterator = ((context, group) for context in range(CONTEXTS)
                for group in range(PHYSICAL_GROUPS)) if mode == 0 else (
                    (context, group) for group in range(PHYSICAL_GROUPS)
                    for context in range(CONTEXTS))
    for context, group in iterator:
        active = values[context, group] != 0
        if active.any():
            rows.append((context, group, bool(active[:8].any()),
                         bool(active[8:].any())))
    valid = [False] * CACHE_ROWS
    group_at = [0] * CACHE_ROWS
    age = [0] * CACHE_ROWS
    access_clock = 1
    hits = misses = evictions = 0
    annotated = []
    for context, group, lower, upper in rows:
        hit = next((index for index in range(CACHE_ROWS)
                    if valid[index] and group_at[index] == group), None)
        if hit is None:
            misses += 1
            invalid = next((index for index in range(CACHE_ROWS)
                            if not valid[index]), None)
            if invalid is None:
                victim = min(range(CACHE_ROWS),
                             key=lambda index: (age[index], index))
                evictions += 1
            else:
                victim = invalid
            valid[victim] = True
            group_at[victim] = group
            age[victim] = access_clock + 1
        else:
            hits += 1
            age[hit] = access_clock
        access_clock += 1
        annotated.append((context, group, lower, upper, hit is None))
    return annotated, {"hits": hits, "misses": misses,
                       "evictions": evictions, "live_rows": len(rows)}


def engine_cycles(values: np.ndarray, mode: int,
                  start_cycle: int = CANONICAL_START,
                  memory_start: int = CANONICAL_MEMORY_START) -> tuple[int, dict]:
    rows, cache = cache_and_order(values, mode)
    cycle = start_cycle
    issues = 0
    for _context, _group, lower, upper, miss in rows:
        cycle += 1  # ST_FIND
        if miss:
            for _beat in range(12):
                cycle = bundle_completion(cycle, start_cycle,
                                          memory_start) + 1
        count = SLICES * (int(lower) + int(upper))
        issues += count
        for _issue in range(count):
            while cycle % 11 == 3:
                cycle += 1
            cycle += 1
    cycle += 1  # empty ST_FIND -> commit
    for _commit in range(CONTEXTS * SLICES):
        while cycle % 13 == 5:
            cycle += 1
        cycle += 1
    cache.update({"issues": issues, "bundles": cache["misses"] * 12,
                  "scalar_reads": cache["misses"] * 12 * 8})
    return cycle - start_cycle, cache


@njit(cache=False, parallel=True)
def batch_engine_cycles(lower: np.ndarray, upper: np.ndarray,
                        mode: int) -> np.ndarray:
    """Exact recurrence in parallel across independent B4 quartets.

    Columns are service cycles, hits, misses, evictions, live rows, issues,
    and scalar weight reads.  This is a speed implementation of
    ``engine_cycles``, not a second timing model.
    """
    population = lower.shape[0]
    groups = lower.shape[2]
    out = np.zeros((population, 7), dtype=np.int64)
    for item in prange(population):
        valid = np.zeros(CACHE_ROWS, dtype=np.uint8)
        group_at = np.zeros(CACHE_ROWS, dtype=np.int64)
        age = np.zeros(CACHE_ROWS, dtype=np.int64)
        access_clock = 1
        cycle = CANONICAL_START
        hits = 0
        misses = 0
        evictions = 0
        live_rows = 0
        issues = 0
        for position in range(CONTEXTS * groups):
            if mode == 0:
                context = position // groups
                group = position % groups
            else:
                group = position // CONTEXTS
                context = position % CONTEXTS
            lo = lower[item, context, group]
            hi = upper[item, context, group]
            if not lo and not hi:
                continue
            live_rows += 1
            hit = -1
            for index in range(CACHE_ROWS):
                if valid[index] and group_at[index] == group:
                    hit = index
                    break
            miss = hit < 0
            if miss:
                misses += 1
                victim = -1
                for index in range(CACHE_ROWS):
                    if not valid[index]:
                        victim = index
                        break
                if victim < 0:
                    victim = 0
                    for index in range(1, CACHE_ROWS):
                        if (age[index] < age[victim] or
                                (age[index] == age[victim] and index < victim)):
                            victim = index
                    evictions += 1
                valid[victim] = 1
                group_at[victim] = group
                age[victim] = access_clock + 1
            else:
                hits += 1
                age[hit] = access_clock
            access_clock += 1
            cycle += 1
            if miss:
                for _beat in range(12):
                    latest = 0
                    for bank in range(8):
                        accepted = cycle
                        while (accepted + 1 + bank * 2) % 7 == 0:
                            accepted += 1
                        completion = accepted + (8 - bank) + 1
                        if completion > latest:
                            latest = completion
                    cycle = latest + 1
            count = SLICES * (int(lo) + int(hi))
            issues += count
            for _issue in range(count):
                while cycle % 11 == 3:
                    cycle += 1
                cycle += 1
        cycle += 1
        for _commit in range(CONTEXTS * SLICES):
            while cycle % 13 == 5:
                cycle += 1
            cycle += 1
        out[item, 0] = cycle - CANONICAL_START
        out[item, 1] = hits
        out[item, 2] = misses
        out[item, 3] = evictions
        out[item, 4] = live_rows
        out[item, 5] = issues
        out[item, 6] = misses * 12 * 8
    return out


def descriptor_key(values: np.ndarray) -> str:
    return hashlib.sha256(values.astype(np.int8, copy=False).tobytes()).hexdigest()


def continuation_backbone(values: np.ndarray, output_tiles: int,
                          mode: int) -> tuple[int, dict]:
    groups = int(values.shape[1])
    need(groups in (96, 192), "continuation geometry")
    total = 0
    ledger = Counter()
    for begin in range(0, groups, PHYSICAL_GROUPS):
        chunk = values[:, begin:begin + PHYSICAL_GROUPS, :]
        service, stats = engine_cycles(chunk, mode)
        total += WRAPPER_LOAD_CYCLES_PER_CHUNK + service
        ledger.update(stats)
    return total * int(output_tiles), dict(ledger)


def calibration() -> dict:
    low_meta = strict_json(M2051_META)
    low_result = strict_json(M2057 / "result.json")
    low_lines = M2051_MEMH.read_text(encoding="ascii").splitlines()
    need(len(low_meta["rows"]) == len(low_result["rows"]) == 1920,
         "G<=48 calibration population")
    low_mismatch = []
    for slot, observed in enumerate(low_result["rows"]):
        groups = int(low_meta["rows"][slot]["source_groups"])
        raw = dense_from_memh(low_lines, slot, 48)
        for mode, field in ((0, "base_cycles"), (1, "tsbg_cycles")):
            predicted, _ = engine_cycles(raw, mode)
            if predicted != int(observed[field]):
                low_mismatch.append((slot, mode, predicted, observed[field]))
    need(not low_mismatch,
         "G<=48 recurrence is not exact on every frozen VCS row")

    high_meta = strict_json(M2067_META)
    high_result = strict_json(M2101 / "result.json")
    high_lines = M2067_MEMH.read_text(encoding="ascii").splitlines()
    need(len(high_meta["rows"]) == len(high_result["rows"]) == 960,
         "continuation calibration population")
    registries = {0: {}, 1: {}}
    locations_by_pair = defaultdict(list)
    residuals = defaultdict(list)
    duplicates = 0
    exact_rows = 0
    for slot, observed in enumerate(high_result["rows"]):
        row = high_meta["rows"][slot]
        groups = int(row["source_groups"])
        values = dense_from_memh(high_lines, slot, groups)
        digest = descriptor_key(values)
        key = (groups, int(row["output_tiles"]), digest)
        row_residuals = {}
        for mode, field in ((0, "base_cycles"), (1, "tsbg_cycles")):
            backbone, _ = continuation_backbone(
                values, int(row["output_tiles"]), mode)
            residual = int(observed[field]) - backbone
            prior = registries[mode].get(key)
            if prior is not None:
                need(prior == residual,
                     "identical continuation descriptor has conflicting residual")
                duplicates += 1
            registries[mode][key] = residual
            row_residuals[mode] = residual
            residuals[(groups, mode)].append(residual)
            need(backbone + registries[mode][key] == int(observed[field]),
                 "continuation exact calibrated reconstruction")
            exact_rows += 1
        locations_by_pair[(int(row["sample_id"]), int(row["layer_id"]))].append({
            "token_start": int(row["token_start"]),
            "descriptor_sha256": digest,
            "ordinary_residual": row_residuals[0],
            "tsbg_residual": row_residuals[1],
        })
    profiles = {}
    for (groups, mode), values in sorted(residuals.items()):
        ordered = sorted(values)
        profiles[f"G{groups}_{'ordinary' if mode == 0 else 'tsbg'}"] = {
            "count": len(values), "minimum": min(values),
            "median": int(ordered[len(ordered) // 2]),
            "maximum": max(values),
            "histogram": {str(k): v for k, v in sorted(Counter(values).items())},
        }
    return {
        "status": "PASS_M2145_EXACT_2880_ROW_CALIBRATION",
        "g_le_48_vcs_rows": 1920,
        "continuation_vcs_rows": 960,
        "axis_cycle_fields_reconstructed_exactly": 5760,
        "base_cycles_mismatches": 0, "tsbg_cycles_mismatches": 0,
        "g_le_48_model": "M2018/M803 event recurrence; no fitted residual",
        "continuation_model": (
            "same event recurrence + 768 wrapper load cycles/chunk + "
            "descriptor-keyed exact calibration residual; unseen descriptors "
            "use same-geometry median with min/max sensitivity"),
        "continuation_duplicate_descriptor_axis_rows": duplicates,
        "continuation_residual_profiles": profiles,
        "registries": registries,
        "locations_by_pair": dict(locations_by_pair),
    }


def decode_payload(raw: bytes, tokens: int, channels: int,
                   row_bytes: int, nnz_total: int) -> np.ndarray:
    matrix_bytes = tokens * row_bytes
    need(len(raw) == 3 * matrix_bytes + 2 * tokens + nnz_total,
         "frame payload extent")
    matrices = []
    offset = 0
    for _ in range(3):
        packed = np.frombuffer(raw[offset:offset + matrix_bytes], dtype=np.uint8)
        bits = np.unpackbits(packed.reshape(tokens, row_bytes), axis=1,
                             bitorder="little")
        need(bool((bits[:, channels:] == 0).all()), "nonzero tail bit")
        matrices.append(bits[:, :channels].astype(bool))
        offset += matrix_bytes
    support, sign, nonunit = matrices
    counts = np.frombuffer(raw[offset:offset + 2 * tokens], dtype="<u2")
    offset += 2 * tokens
    codes = np.frombuffer(raw[offset:], dtype=np.int8)
    need(bool((sign <= support).all()) and bool((nonunit <= support).all()) and
         bool((counts == support.sum(axis=1)).all()) and
         int(counts.sum()) == nnz_total and codes.size == nnz_total and
         bool((codes != 0).all()) and
         bool(((codes < 0) == sign[support]).all()) and
         not bool(nonunit.any()), "frame descriptor semantic drift")
    dense = np.zeros((tokens, channels), dtype=np.int8)
    dense[support] = codes
    need(set(int(value) for value in np.unique(dense)).issubset({-1, 0, 1}),
         "captured descriptor outside {-1,0,+1}")
    return dense


class Stats:
    def __init__(self) -> None:
        self.rows = 0
        self.base = self.tsbg = 0
        self.base_low = self.tsbg_high = 0
        self.read_base = self.read_tsbg = 0
        self.hit_base = self.hit_tsbg = 0
        self.miss_base = self.miss_tsbg = 0
        self.evict_base = self.evict_tsbg = 0
        self.empty = self.slower = self.equal = 0
        self.exact_continuation_calibration_hits = 0
        self.continuation_median_extrapolations = 0
        self.speedups = array("d")

    def add(self, row: dict) -> None:
        self.rows += 1
        self.base += row["base_cycles"]
        self.tsbg += row["tsbg_cycles"]
        self.base_low += row["base_cycles_low"]
        self.tsbg_high += row["tsbg_cycles_high"]
        for prefix in ("base", "tsbg"):
            setattr(self, "read_" + prefix,
                    getattr(self, "read_" + prefix) + row[prefix]["scalar_reads"])
            setattr(self, "hit_" + prefix,
                    getattr(self, "hit_" + prefix) + row[prefix]["hits"])
            setattr(self, "miss_" + prefix,
                    getattr(self, "miss_" + prefix) + row[prefix]["misses"])
            setattr(self, "evict_" + prefix,
                    getattr(self, "evict_" + prefix) + row[prefix]["evictions"])
        self.empty += int(row["empty"])
        self.slower += int(row["tsbg_cycles"] > row["base_cycles"])
        self.equal += int(row["tsbg_cycles"] == row["base_cycles"])
        self.exact_continuation_calibration_hits += int(row["calibration_exact_hit"])
        self.continuation_median_extrapolations += int(row["calibration_median"])
        self.speedups.append(row["base_cycles"] / row["tsbg_cycles"])

    def add_batch(self, batch: dict) -> None:
        population = int(batch["base_cycles"].size)
        need(population > 0, "empty replay batch")
        self.rows += population
        for attribute, key in (
                ("base", "base_cycles"), ("tsbg", "tsbg_cycles"),
                ("base_low", "base_cycles_low"),
                ("tsbg_high", "tsbg_cycles_high"),
                ("read_base", "base_scalar_reads"),
                ("read_tsbg", "tsbg_scalar_reads"),
                ("hit_base", "base_hits"), ("hit_tsbg", "tsbg_hits"),
                ("miss_base", "base_misses"),
                ("miss_tsbg", "tsbg_misses"),
                ("evict_base", "base_evictions"),
                ("evict_tsbg", "tsbg_evictions")):
            setattr(self, attribute, getattr(self, attribute) +
                    int(np.sum(batch[key], dtype=np.int64)))
        self.empty += int(np.count_nonzero(batch["empty"]))
        self.slower += int(np.count_nonzero(
            batch["tsbg_cycles"] > batch["base_cycles"]))
        self.equal += int(np.count_nonzero(
            batch["tsbg_cycles"] == batch["base_cycles"]))
        self.exact_continuation_calibration_hits += int(np.count_nonzero(
            batch["calibration_exact_hit"]))
        self.continuation_median_extrapolations += int(np.count_nonzero(
            batch["calibration_median"]))
        ratios = np.ascontiguousarray(
            batch["base_cycles"] / batch["tsbg_cycles"], dtype=np.float64)
        self.speedups.frombytes(ratios.tobytes())

    def result(self) -> dict:
        values = np.frombuffer(self.speedups, dtype=np.float64)
        need(self.rows and values.size == self.rows and self.base > 0 and
             self.tsbg > 0, "empty statistics")
        return {
            "aligned_b4_quartets": self.rows,
            "ordinary_cycles": self.base, "tsbg_cycles": self.tsbg,
            "ratio_of_sums": self.base / self.tsbg,
            "time_reduction_fraction": 1.0 - self.tsbg / self.base,
            "p10_workload_ratio": float(np.percentile(values, 10)),
            "p50_workload_ratio": float(np.percentile(values, 50)),
            "p90_workload_ratio": float(np.percentile(values, 90)),
            "worst_workload_ratio": float(values.min()),
            "slower_case_rate": self.slower / self.rows,
            "slower_cases": self.slower, "equal_cases": self.equal,
            "empty_quartets": self.empty,
            "ordinary_scalar_weight_reads": self.read_base,
            "tsbg_scalar_weight_reads": self.read_tsbg,
            "weight_read_reduction_fraction": 1.0 - self.read_tsbg / self.read_base,
            "ordinary_cache": {"hits": self.hit_base, "misses": self.miss_base,
                               "evictions": self.evict_base},
            "tsbg_cache": {"hits": self.hit_tsbg, "misses": self.miss_tsbg,
                           "evictions": self.evict_tsbg},
            "schedule_fallback": {"implemented": False, "fallbacks": 0,
                "reason": "M2018 SCHEDULE_MODE is elaboration-time; no dynamic fallback is claimed"},
            "continuation_exact_calibration_hits":
                self.exact_continuation_calibration_hits,
            "continuation_median_residual_extrapolations":
                self.continuation_median_extrapolations,
            "pessimistic_residual_envelope_ratio": self.base_low / self.tsbg_high,
        }


def replay_quartet(values: np.ndarray, layer: dict,
                   calibrated: dict) -> dict:
    groups = int(layer["weight_layout"]["source_group_count"])
    tiles = int(layer["weight_layout"]["output_tile_count"])
    need(values.shape == (CONTEXTS, groups, SOURCES), "quartet geometry")
    ledgers = {}
    if groups <= PHYSICAL_GROUPS:
        padded = np.zeros((CONTEXTS, PHYSICAL_GROUPS, SOURCES), dtype=np.int8)
        padded[:, :groups, :] = values
        cycles = []
        for mode, name in ((0, "base"), (1, "tsbg")):
            service, ledger = engine_cycles(padded, mode)
            cycles.append(tiles * (WRAPPER_LOAD_CYCLES_PER_CHUNK + service))
            ledgers[name] = {key: int(value) * tiles
                             for key, value in ledger.items()}
        base_low, tsbg_high = cycles
        exact_hit = median = False
    else:
        key = (groups, tiles, descriptor_key(values))
        cycles = []
        bounds = []
        exact_hit = True
        median = False
        for mode, name in ((0, "base"), (1, "tsbg")):
            backbone, ledger = continuation_backbone(values, tiles, mode)
            profile = calibrated["continuation_residual_profiles"][
                f"G{groups}_{'ordinary' if mode == 0 else 'tsbg'}"]
            registry = calibrated["registries"][mode]
            if key in registry:
                correction = registry[key]
            else:
                exact_hit = False
                median = True
                correction = profile["median"]
            cycles.append(backbone + correction)
            bounds.append((backbone + profile["minimum"],
                           backbone + profile["maximum"]))
            ledgers[name] = {key2: int(value) * tiles
                             for key2, value in ledger.items()}
        base_low = bounds[0][0]
        tsbg_high = bounds[1][1]
    return {"base_cycles": cycles[0], "tsbg_cycles": cycles[1],
            "base_cycles_low": base_low, "tsbg_cycles_high": tsbg_high,
            "base": ledgers["base"], "tsbg": ledgers["tsbg"],
            "empty": not bool(np.any(values)),
            "calibration_exact_hit": exact_hit,
            "calibration_median": median}


def replay_batch(quartets: np.ndarray, layer: dict, calibrated: dict,
                 sample_id: int, token_start: int) -> dict:
    """Vectorized/JIT production path with the same scalar recurrence."""
    groups = int(layer["weight_layout"]["source_group_count"])
    tiles = int(layer["weight_layout"]["output_tile_count"])
    need(quartets.ndim == 4 and quartets.shape[1:] ==
         (CONTEXTS, groups, SOURCES), "batch quartet geometry")
    lower = np.any(quartets[:, :, :, :8] != 0, axis=3)
    upper = np.any(quartets[:, :, :, 8:] != 0, axis=3)
    population = quartets.shape[0]
    outputs = {
        key: np.zeros(population, dtype=np.int64) for key in (
            "base_cycles", "tsbg_cycles", "base_cycles_low",
            "tsbg_cycles_high", "base_scalar_reads", "tsbg_scalar_reads",
            "base_hits", "tsbg_hits", "base_misses", "tsbg_misses",
            "base_evictions", "tsbg_evictions")}
    backbones = []
    for mode, prefix in ((0, "base"), (1, "tsbg")):
        backbone = np.zeros(population, dtype=np.int64)
        for begin in range(0, groups, PHYSICAL_GROUPS):
            end = min(begin + PHYSICAL_GROUPS, groups)
            ledger = batch_engine_cycles(
                lower[:, :, begin:end], upper[:, :, begin:end], mode)
            backbone += WRAPPER_LOAD_CYCLES_PER_CHUNK + ledger[:, 0]
            outputs[prefix + "_hits"] += ledger[:, 1] * tiles
            outputs[prefix + "_misses"] += ledger[:, 2] * tiles
            outputs[prefix + "_evictions"] += ledger[:, 3] * tiles
            outputs[prefix + "_scalar_reads"] += ledger[:, 6] * tiles
        backbone *= tiles
        backbones.append(backbone)

    exact_hit = np.zeros(population, dtype=bool)
    median = np.zeros(population, dtype=bool)
    if groups <= PHYSICAL_GROUPS:
        outputs["base_cycles"][:] = backbones[0]
        outputs["tsbg_cycles"][:] = backbones[1]
        outputs["base_cycles_low"][:] = backbones[0]
        outputs["tsbg_cycles_high"][:] = backbones[1]
    else:
        ordinary = calibrated["continuation_residual_profiles"][
            f"G{groups}_ordinary"]
        tsbg = calibrated["continuation_residual_profiles"][f"G{groups}_tsbg"]
        outputs["base_cycles"][:] = backbones[0] + ordinary["median"]
        outputs["tsbg_cycles"][:] = backbones[1] + tsbg["median"]
        outputs["base_cycles_low"][:] = backbones[0] + ordinary["minimum"]
        outputs["tsbg_cycles_high"][:] = backbones[1] + tsbg["maximum"]
        median[:] = True
        locations = calibrated["locations_by_pair"].get(
            (int(sample_id), int(layer["layer_id"])), [])
        frame_end = token_start + population * CONTEXTS
        for location in locations:
            selected = int(location["token_start"])
            if token_start <= selected < frame_end:
                need((selected - token_start) % CONTEXTS == 0,
                     "calibration location is not B4 aligned")
                index = (selected - token_start) // CONTEXTS
                need(descriptor_key(quartets[index]) ==
                     location["descriptor_sha256"],
                     "full-token descriptor drift at calibration location")
                outputs["base_cycles"][index] = (
                    backbones[0][index] + location["ordinary_residual"])
                outputs["tsbg_cycles"][index] = (
                    backbones[1][index] + location["tsbg_residual"])
                exact_hit[index] = True
                median[index] = False
    outputs["empty"] = ~(lower | upper).any(axis=(1, 2))
    outputs["calibration_exact_hit"] = exact_hit
    outputs["calibration_median"] = median
    return outputs


def run(output: Path, workers: int = 3) -> dict:
    need(1 <= workers <= 3, "CPU worker count must be in [1,3]")
    set_num_threads(workers)
    identity = verify_inputs(require_contract=True)
    calibrated = calibration()
    need(not os.path.lexists(str(output)), "fresh output required")
    layers_payload = strict_json(CAPTURE / "layers.json")
    samples_payload = strict_json(CAPTURE / "sample_order.json")
    layers = [row for row in layers_payload["layers"]
              if row["target"] in ("FC1", "FC2")]
    samples = samples_payload["samples"]
    need(len(layers) == EXPECTED_LAYERS and
         sum(row["target"] == "FC1" for row in layers) == EXPECTED_FC1 and
         sum(row["target"] == "FC2" for row in layers) == EXPECTED_FC2 and
         len(samples) == EXPECTED_SAMPLES and
         len({row["sequence"] for row in samples}) == EXPECTED_SEQUENCES,
         "full FC inventory/sample identity")
    layer_by_id = {int(row["layer_id"]): row for row in layers}
    sample_by_id = {int(row["global_sample_id"]): row for row in samples}
    expected_pairs = [(int(sample["global_sample_id"]), int(layer["layer_id"]))
                      for sample in samples for layer in layers]

    aggregate = Stats()
    by_target = {name: Stats() for name in ("FC1", "FC2")}
    by_sequence = {name: Stats() for name in sorted(
        {row["sequence"] for row in samples})}
    by_layer = {str(row["layer_id"]): Stats() for row in layers}
    pair_index = frame_index_expected = token_start_expected = 0
    frames = 0
    with (CAPTURE / "fc_frames.bin").open("rb") as stream:
        while True:
            prefix = stream.read(FRAME_HEADER.size)
            if not prefix:
                break
            need(len(prefix) == FRAME_HEADER.size, "truncated frame header")
            (magic, version, header_size, layer_id, sample_id, frame_index,
             token_start, token_count, channels, bitrow, nnz_total, raw_bytes,
             compressed_bytes, crc32) = FRAME_HEADER.unpack(prefix)
            need(pair_index < len(expected_pairs) and
                 (sample_id, layer_id) == expected_pairs[pair_index] and
                 magic == FRAME_MAGIC and version == FRAME_VERSION and
                 header_size == FRAME_HEADER.size and
                 frame_index == frame_index_expected and
                 token_start == token_start_expected and token_count % 4 == 0,
                 "canonical frame order/identity")
            layer = layer_by_id[layer_id]
            need(channels == int(layer["input_channels"]) and
                 channels == int(layer["weight_layout"]["source_group_count"]) * SOURCES and
                 bitrow == (channels + 7) // 8,
                 "frame layer geometry")
            compressed = stream.read(compressed_bytes)
            need(len(compressed) == compressed_bytes, "truncated zlib frame")
            decoder = zlib.decompressobj()
            raw = decoder.decompress(compressed) + decoder.flush()
            need(decoder.eof and not decoder.unused_data and
                 not decoder.unconsumed_tail and len(raw) == raw_bytes and
                 (zlib.crc32(raw) & 0xffffffff) == crc32,
                 "frame zlib/CRC")
            dense = decode_payload(raw, token_count, channels, bitrow, nnz_total)
            groups = channels // SOURCES
            quartets = dense.reshape(token_count // 4, CONTEXTS, groups, SOURCES)
            sequence = sample_by_id[sample_id]["sequence"]
            batch = replay_batch(quartets, layer, calibrated, sample_id,
                                 token_start)
            aggregate.add_batch(batch)
            by_target[layer["target"]].add_batch(batch)
            by_sequence[sequence].add_batch(batch)
            by_layer[str(layer_id)].add_batch(batch)
            frames += 1
            token_start_expected += token_count
            need(token_start_expected <= int(layer["tokens_per_call"]),
                 "pair token overflow")
            if token_start_expected == int(layer["tokens_per_call"]):
                pair_index += 1
                frame_index_expected = token_start_expected = 0
            else:
                frame_index_expected += 1
    need(pair_index == len(expected_pairs) and token_start_expected == 0 and
         aggregate.rows == EXPECTED_QUARTETS,
         "full-token population incomplete")
    result = {
        "schema": "m2145_ep34_tsbg_fulltoken_calibrated_replay_result_r1_v1",
        "status": "CPU_MODEL_PASS_PENDING_INDEPENDENT_RESULT_HAMMER_DO_NOT_CITE",
        "identity": {**identity, "source_sha256": sha256(SOURCE),
                     "m2057_result_sha256": EXPECTED[M2057 / "result.json"],
                     "m2101_result_sha256": EXPECTED[M2101 / "result.json"]},
        "population": {"checkpoint": "motion_ep34_live93", "sequences": 4,
            "samples": 40, "layers": 24, "fc1_layers": 12,
            "fc2_layers": 12, "aligned_b4_quartets": EXPECTED_QUARTETS,
            "frames": frames, "all_aligned_b4_tokens": True},
        "calibration": {key: value for key, value in calibrated.items()
                        if key not in {"registries", "locations_by_pair"}},
        "execution": {"cpu_workers": workers,
                      "implementation": "Numba-parallel exact recurrence batches"},
        "aggregate": aggregate.result(),
        "breakdown": {
            "target": {key: value.result() for key, value in by_target.items()},
            "sequence": {key: value.result() for key, value in by_sequence.items()},
            "layer_id": {key: value.result() for key, value in by_layer.items()},
        },
        "claim_boundary": {
            "full_aligned_b4_cpu_replay": True,
            "vcs_calibrated": True,
            "all_2880_calibration_rows_exact": True,
            "real_ep34_activity_and_sign_descriptors": True,
            "hardware_weight_values": False,
            "continuation_unseen_residual_is_median_extrapolation": True,
            "rtl_execution": False, "same_area": False,
            "power": False, "energy": False, "full_network": False,
            "system_speedup": False, "fps": False,
            "paper_admitted": False, "headline": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".m2145_stage.", dir=output.parent))
    try:
        (stage / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary = {"status": result["status"], "population": result["population"],
                   "aggregate": result["aggregate"],
                   "claim_boundary": result["claim_boundary"]}
        (stage / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        members = ["result.json", "summary.json"]
        manifest = "".join(f"{sha256(stage / name)}  {name}\n" for name in members)
        (stage / "SHA256SUMS").write_text(manifest, encoding="ascii")
        (stage / "SHA256SUMS.seal.sha256").write_text(
            f"{sha256(stage / 'SHA256SUMS')}  SHA256SUMS\n", encoding="ascii")
        os.replace(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return result


def selftest() -> dict:
    identity = verify_inputs(require_contract=False)
    calibrated = calibration()
    zero = np.zeros((4, 48, 16), dtype=np.int8)
    base, base_stats = engine_cycles(zero, 0)
    tsbg, tsbg_stats = engine_cycles(zero, 1)
    need(base == tsbg and base_stats == tsbg_stats and base > 0,
         "zero descriptor conservation")
    attack = np.zeros((4, 48, 16), dtype=np.int8)
    # Five live groups overflow LRU4 in context-major order; group-major order
    # bundles the four contexts before the next group and must cut misses.
    for context in range(4):
        for group in range(5):
            attack[context, group, (context * 5 + group) % 16] = (
                -1 if (context + group) & 1 else 1)
    b, bs = engine_cycles(attack, 0)
    t, ts = engine_cycles(attack, 1)
    need(bs["misses"] > ts["misses"] and b > t,
         "group-major reuse mutation sensitivity")
    lower = np.any(attack[None, :, :, :8] != 0, axis=3)
    upper = np.any(attack[None, :, :, 8:] != 0, axis=3)
    for mode, expected_cycles, expected in ((0, b, bs), (1, t, ts)):
        fast = batch_engine_cycles(lower, upper, mode)[0]
        need(int(fast[0]) == expected_cycles and
             int(fast[1]) == expected["hits"] and
             int(fast[2]) == expected["misses"] and
             int(fast[3]) == expected["evictions"] and
             int(fast[4]) == expected["live_rows"] and
             int(fast[5]) == expected["issues"] and
             int(fast[6]) == expected["scalar_reads"],
             "batch/scalar recurrence mismatch")
    mutated = attack.copy()
    mutated[1, 0, 5] = 0
    need(descriptor_key(mutated) != descriptor_key(attack),
         "descriptor-key mutation escaped")
    fake_observed = b + 1
    need(fake_observed != b, "cycle mutation did not fail exact equality")
    claim = {"rtl_execution": False, "same_area": False,
             "system_speedup": False, "paper_admitted": False}
    need(not any(claim.values()), "claim boundary mutation")
    return {"status": "PASS_M2145_SOURCE_SELFTEST_NO_PRODUCTION_REPLAY",
            "identity": identity,
            "calibration_status": calibrated["status"],
            "cycle_fields_exact":
                calibrated["axis_cycle_fields_reconstructed_exactly"],
            "batch_scalar_recurrence_fields_exact": 14,
            "mutations_rejected": 4,
            "production_frames_decoded": 0,
            "production_replay_executed": False,
            "vcs_runs": 0, "simv_runs": 0, "eda_runs": 0,
            "gpu_jobs": 0, "license_queries": 0}


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--static", action="store_true")
    group.add_argument("--selftest", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.static:
        need(args.output is None and args.workers == 3,
             "static takes no output/worker override")
        identity = verify_inputs(require_contract=False)
        print(json.dumps({"status": "PASS_M2145_STATIC_INPUTS",
                          "identity": identity,
                          "production_replay_executed": False}, sort_keys=True))
    elif args.selftest:
        need(args.output is None and args.workers == 3,
             "selftest takes no output/worker override")
        print(json.dumps(selftest(), sort_keys=True))
    else:
        need(args.output is not None, "--run requires --output")
        result = run(args.output.resolve(), args.workers)
        print(json.dumps({"status": result["status"],
                          "output": str(args.output.resolve()),
                          "aligned_b4_quartets": EXPECTED_QUARTETS}, sort_keys=True))
    need(sha256(DOC359) == EXPECTED[DOC359], "docs/359 changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
