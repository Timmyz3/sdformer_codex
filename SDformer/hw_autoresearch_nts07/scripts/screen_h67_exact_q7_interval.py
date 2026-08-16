#!/usr/bin/env python3
"""Screen exact Q7 certification after dominant Motion statistics are known."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from profile_h67_tare_zkqi_overlap import rne_div16_array, unpack_bits


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/h67_fullres_ep30_t450_all12_bit_trace_20260805/manifest.json"
OUT = ROOT / "results/h67_exact_q7_interval_screen_20260814"
CHUNKS = (4, 8, 16)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def certify_lanes(
    silence_bits: np.ndarray,
    base_raw: np.ndarray,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return earliest common certification point and the exact final scores."""

    if silence_bits.shape[-2:] != (2, 32):
        raise ValueError(f"illegal silence shape: {silence_bits.shape}")
    if base_raw.shape != silence_bits.shape[:-1]:
        raise ValueError("base/silence shape mismatch")
    full_silence = silence_bits.sum(axis=-1, dtype=np.int64)
    exact = rne_div16_array(base_raw + full_silence)
    stop = np.full(base_raw.shape[:-1], 32, dtype=np.int64)
    unresolved = np.ones(stop.shape, dtype=np.bool_)
    partial = np.zeros(base_raw.shape, dtype=np.int64)
    for processed in range(chunk, 33, chunk):
        begin = processed - chunk
        partial += silence_bits[..., begin:processed].sum(axis=-1, dtype=np.int64)
        remaining = 32 - processed
        lower = rne_div16_array(base_raw + partial)
        upper = rne_div16_array(base_raw + partial + remaining)
        certified = (lower == upper).all(axis=-1)
        newly = unresolved & certified
        stop[newly] = processed
        unresolved &= ~certified
    if np.any(unresolved):
        raise AssertionError("full 32-lane score failed to certify")
    return stop, exact


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    records = manifest.get("records", [])
    if len(records) != 12:
        raise ValueError("expected all 12 Motion blocks")

    stop_values = {chunk: [] for chunk in CHUNKS}
    exact_rows = 0
    active_pairs = 0
    stage_values = {
        stage: {chunk: [] for chunk in CHUNKS} for stage in range(4)
    }
    source_sha = {}

    for record in records:
        path = Path(record["file"])
        observed = sha256(path)
        if observed != record["sha256"]:
            raise ValueError(f"trace SHA mismatch: {path}")
        source_sha[str(path)] = observed
        with np.load(path) as payload:
            q = unpack_bits(payload, "q")
            k = unpack_bits(payload, "k")
        if q.shape != k.shape or q.shape[0] != 2 or q.shape[-1] != 32:
            raise ValueError(f"illegal Q/K shape: {q.shape}/{k.shape}")

        overlap = (q & k).sum(axis=-1, dtype=np.int64)
        motion = (k[0] ^ k[1]).sum(axis=-1, dtype=np.int64)
        base = 64 * overlap + 16 * motion[None, ...]
        silence = ~(q | k)
        active = k.any(axis=-1).any(axis=0)
        stage = int(record["name"][1])
        exact_rows += int(active.size)
        active_pairs += int(active.sum())

        for chunk in CHUNKS:
            # Reorder to [..., time, lane] so one certification decision covers
            # both temporal scores required by the RQTB packet.
            silence_pair = np.moveaxis(silence, 0, -2)
            base_pair = np.moveaxis(base, 0, -1)
            stop, exact = certify_lanes(silence_pair, base_pair, chunk)
            direct = rne_div16_array(base_pair + silence_pair.sum(axis=-1))
            if not np.array_equal(exact, direct):
                raise AssertionError("exact Q7 reconstruction mismatch")
            selected = stop[active]
            stop_values[chunk].append(selected)
            stage_values[stage][chunk].append(selected)

    results = {}
    for chunk in CHUNKS:
        values = np.concatenate(stop_values[chunk]).astype(np.int64)
        hist = {str(lanes): int(np.count_nonzero(values == lanes)) for lanes in range(chunk, 33, chunk)}
        silence_lane_work = int((2 * values).sum())
        baseline_silence_lane_work = int(2 * 32 * values.size)
        baseline_five_stat_work = int(5 * 32 * values.size)
        candidate_five_stat_work = int(3 * 32 * values.size + silence_lane_work)
        results[str(chunk)] = {
            "active_pairs": int(values.size),
            "stop_lane_histogram": hist,
            "early_certified_pairs": int(np.count_nonzero(values < 32)),
            "early_certified_ratio": float(np.mean(values < 32)),
            "mean_processed_lanes": float(values.mean()),
            "same_zero_lane_work_reduction": 1.0 - silence_lane_work / baseline_silence_lane_work,
            "five_stat_lane_work_reduction": 1.0 - candidate_five_stat_work / baseline_five_stat_work,
            "serial_mean_cycles": float((values / chunk).mean()),
            "pipeline_note": "one-pair/cycle requires all chunk stages; early certification then gates activity but does not remove the later stage area",
        }

    stage_report = {}
    for stage, chunk_map in stage_values.items():
        stage_report[str(stage)] = {}
        for chunk, arrays in chunk_map.items():
            values = np.concatenate(arrays).astype(np.int64)
            stage_report[str(stage)][str(chunk)] = {
                "active_pairs": int(values.size),
                "early_certified_ratio": float(np.mean(values < 32)),
                "mean_processed_lanes": float(values.mean()),
            }

    best = max(results.values(), key=lambda row: row["five_stat_lane_work_reduction"])
    status = (
        "CONDITIONAL_ACTIVITY_ONLY"
        if best["five_stat_lane_work_reduction"] >= 0.10
        and best["early_certified_ratio"] >= 0.50
        else "REJECT_AS_DATE_ARCHITECTURE"
    )
    report = {
        "schema": "h67_exact_q7_interval_screen_v1",
        "status": status,
        "evidence": "[prof] exact hardware-order bit trace; lane-work/activity upper bound, not RTL cycles or energy",
        "scope": "ep30 sample0, one T450 window per block, all 12 blocks/138 head rows; active pairs after exact K-zero exclusion",
        "contract": {
            "dominant_statistics": "overlap0, overlap1, and shared motion are fully known first",
            "certified_term": "same-zero is consumed in fixed lane chunks; stop only when RNE(lower)==RNE(upper) for both temporal scores",
            "exactness": "remaining same-zero contribution is bounded by [0, remaining_lanes]; no approximation or token pruning",
        },
        "population": {"row_pair_positions": exact_rows, "active_pairs": active_pairs},
        "chunks": results,
        "by_stage": stage_report,
        "source_sha256": source_sha,
        "claim_boundary": [
            "TARE W8/W16 remains rejected; this screen does not compact changed lanes",
            "a serial implementation loses throughput; a fully pipelined implementation retains later-stage area and only gates activity",
            "no SAIF, DC, SRAM, row-top, or full-encoder result",
            "does not modify frozen Motion cycles or docs/359",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Motion exact Q7 interval certification screen",
        "",
        f"- Verdict: `{status}`.",
        f"- Scope: {report['scope']}.",
        "- Exact contract: compute both overlap counts and shared motion first; consume same-zero bits by fixed chunks, and stop only when lower/upper bounds round to the same Q7 class for both temporal scores.",
        "",
        "| chunk | early pair | mean lanes | same-zero work reduction | five-stat work reduction | serial cycles |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for chunk in CHUNKS:
        row = results[str(chunk)]
        lines.append(
            f"| {chunk} | {row['early_certified_ratio']:.2%} | {row['mean_processed_lanes']:.3f} | "
            f"{row['same_zero_lane_work_reduction']:.2%} | {row['five_stat_lane_work_reduction']:.2%} | "
            f"{row['serial_mean_cycles']:.3f} |"
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        "This is an exact profile upper bound, not RTL performance or energy. A serial engine exposes the listed multi-cycle latency; a throughput-one pipeline still instantiates all chunk stages, so early certification only gates activity. TARE remains rejected and frozen Motion ledgers are unchanged.",
    ])
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
