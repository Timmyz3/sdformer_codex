#!/usr/bin/env python3
"""Profile exact narrow-Acc promotion on ordered Motion and Local5 updates.

This is an architecture screen, not a PPA estimator. A promoted entry retains a
full signed Acc32 value; no saturation, truncation, or changed summation order is
allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

try:
    from scripts.analyze_projection_accumulator_range import build_activations
except ModuleNotFoundError:
    from analyze_projection_accumulator_range import build_activations


WIDTHS = (12, 14, 16, 18)
LOCAL_HEIGHT = 15
LOCAL_WIDTH = 15
LOCAL_PLANES = 2
LOCAL_SOURCES = 450
LOCAL_HEAD_DIM = 32
LOCAL_OUT_DIM = 32
LOCAL_ROLE_DY = (0, 1, -1, 0, 0)
LOCAL_ROLE_DX = (0, 0, 0, 1, -1)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_signed_memh(path: Path, width: int) -> np.ndarray:
    unsigned_dtype = np.uint8 if width <= 8 else np.uint32
    signed_dtype = np.int8 if width <= 8 else np.int32
    raw = np.asarray(
        [int(line, 16) for line in path.read_text().splitlines() if line.strip()],
        dtype=unsigned_dtype,
    )
    return raw.view(signed_dtype).astype(np.int64)


def load_unsigned_memh(path: Path) -> np.ndarray:
    return np.asarray(
        [int(line, 16) for line in path.read_text().splitlines() if line.strip()],
        dtype=np.int64,
    )


class PromotionTracker:
    def __init__(self, width: int, entries: int):
        self.width = width
        self.limit = 1 << (width - 1)
        self.sticky = np.zeros(entries, dtype=bool)
        self.dynamic_count = 0
        self.dynamic_peak = 0
        self.dynamic_high_accesses = 0
        self.sticky_high_accesses = 0
        self.scalar_updates = 0

    def update(self, flat_indices: np.ndarray, old: np.ndarray, new: np.ndarray) -> None:
        before = (old < -self.limit) | (old >= self.limit)
        after = (new < -self.limit) | (new >= self.limit)
        self.dynamic_high_accesses += int(np.count_nonzero(before | after))
        self.dynamic_count += int(np.count_nonzero(after)) - int(np.count_nonzero(before))
        self.dynamic_peak = max(self.dynamic_peak, self.dynamic_count)

        sticky_before = self.sticky[flat_indices]
        self.sticky_high_accesses += int(np.count_nonzero(sticky_before | after))
        self.sticky[flat_indices] = sticky_before | after
        self.scalar_updates += int(old.size)

    def row(self, entries: int) -> dict:
        addr_bits = max(1, math.ceil(math.log2(entries)))
        entry_bits = 32 + addr_bits
        sticky_entries = int(np.count_nonzero(self.sticky))
        dynamic_state = entries * (self.width + 1) + self.dynamic_peak * entry_bits
        sticky_state = entries * (self.width + 1) + sticky_entries * entry_bits
        updates = self.scalar_updates
        dynamic_access_fraction = self.dynamic_high_accesses / updates if updates else 0.0
        sticky_access_fraction = self.sticky_high_accesses / updates if updates else 0.0
        return {
            "width": self.width,
            "entries": entries,
            "scalar_updates": updates,
            "dynamic_peak_entries": self.dynamic_peak,
            "dynamic_peak_fraction": self.dynamic_peak / entries,
            "dynamic_high_access_fraction": dynamic_access_fraction,
            "dynamic_state_bits": dynamic_state,
            "dynamic_state_reduction_vs_acc32": entries * 32 / dynamic_state,
            "dynamic_accessed_payload_ratio_vs_acc32": (
                (self.width + 1 + dynamic_access_fraction * 32) / 32
            ),
            "sticky_entries": sticky_entries,
            "sticky_fraction": sticky_entries / entries,
            "sticky_high_access_fraction": sticky_access_fraction,
            "sticky_state_bits": sticky_state,
            "sticky_state_reduction_vs_acc32": entries * 32 / sticky_state,
        }


def local5_profile(vector_dir: Path, ordered_payload: Path) -> dict:
    manifest_path = vector_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = manifest["selection"]["rows"]
    if len(rows) != 100 or manifest["shape"]["out_dim"] != LOCAL_OUT_DIM:
        raise ValueError("Local5 profile must be the locked 100-group OUT32 population")

    weights = load_signed_memh(vector_dir / "input_weights.memh", 8).reshape(
        100, LOCAL_HEAD_DIM, LOCAL_OUT_DIM
    )
    expected = load_signed_memh(vector_dir / "expected_acc.memh", 32).reshape(
        100, LOCAL_SOURCES, LOCAL_OUT_DIM
    )

    group_results = []
    with np.load(ordered_payload, allow_pickle=False) as payload:
        offsets = np.asarray(payload["descriptor_group_offsets"])
        planes = np.asarray(payload["descriptor_source_plane"])
        source_y = np.asarray(payload["descriptor_source_y"])
        source_x = np.asarray(payload["descriptor_source_x"])
        k_bitmap = np.asarray(payload["descriptor_k_bitmap"])
        gates_all = np.asarray(payload["descriptor_incoming_gates"])
        valid_masks = np.asarray(payload["descriptor_valid_mask"])

        entries = LOCAL_SOURCES * LOCAL_OUT_DIM
        out_indices = np.arange(LOCAL_OUT_DIM, dtype=np.int64)
        for output_group, metadata in enumerate(rows):
            input_group = int(metadata["input_group_index"])
            start = int(offsets[input_group])
            stop = int(offsets[input_group + 1])
            if stop - start != LOCAL_SOURCES:
                raise ValueError("Local5 descriptor group is not T450")

            acc = np.zeros((LOCAL_SOURCES, LOCAL_OUT_DIM), dtype=np.int64)
            trackers = {width: PromotionTracker(width, entries) for width in WIDTHS}
            for index in range(start, stop):
                plane = int(planes[index])
                sy = int(source_y[index])
                sx = int(source_x[index])
                k_value = int(k_bitmap[index])
                gates = gates_all[index]
                valid_mask = int(valid_masks[index])

                unique_gates = []
                for role in range(5):
                    gate = int(gates[role])
                    if (
                        ((valid_mask >> role) & 1)
                        and gate != 0
                        and gate not in unique_gates
                    ):
                        unique_gates.append(gate)

                for lane in range(LOCAL_HEAD_DIM):
                    if not ((k_value >> lane) & 1):
                        continue
                    for gate in unique_gates:
                        delta = gate * weights[output_group, lane]
                        for role in range(5):
                            if not (
                                ((valid_mask >> role) & 1)
                                and int(gates[role]) == gate
                            ):
                                continue
                            dy = sy + LOCAL_ROLE_DY[role]
                            dx = sx + LOCAL_ROLE_DX[role]
                            destination = (
                                plane * LOCAL_HEIGHT * LOCAL_WIDTH
                                + dy * LOCAL_WIDTH
                                + dx
                            )
                            old = acc[destination].copy()
                            new = old + delta
                            flat_indices = destination * LOCAL_OUT_DIM + out_indices
                            for tracker in trackers.values():
                                tracker.update(flat_indices, old, new)
                            acc[destination] = new

            mismatch = int(np.count_nonzero(acc != expected[output_group]))
            if mismatch:
                raise ValueError(
                    f"Local5 group {output_group} ordered Acc32 mismatch={mismatch}"
                )
            group_results.append(
                {
                    "group": output_group,
                    "sample": int(metadata["sample"]),
                    "stage": int(metadata["stage"]),
                    "block": int(metadata["block"]),
                    "window": int(metadata["window"]),
                    "head": int(metadata["head"]),
                    "empty": bool(metadata["empty"]),
                    "mismatch": mismatch,
                    "widths": {str(width): trackers[width].row(entries) for width in WIDTHS},
                }
            )

    return {
        "scope": "current Local5 fullres 100-group OUT32 ordered source-major updates",
        "evidence": "[prof]+[real-checkpoint-int8]+[software-ordered-miter]",
        "groups": len(group_results),
        "mismatch": 0,
        "results": group_results,
        "provenance": {
            "vector_manifest": str(manifest_path.resolve()),
            "vector_manifest_sha256": sha256(manifest_path),
            "ordered_payload": str(ordered_payload.resolve()),
            "ordered_payload_sha256": sha256(ordered_payload),
        },
    }


def motion_historical_profile(vector_root: Path, raw_records_path: Path) -> dict:
    raw_records = np.asarray(
        [int(line, 16) for line in raw_records_path.read_text().splitlines() if line.strip()],
        dtype=np.int64,
    )
    head_offset = 0
    stage_results = []
    for stage in range(4):
        heads = 3 << stage
        dim = heads * 32
        vector_dir = vector_root / f"real_sample0_s{stage}_b0_capacity"
        weights = load_signed_memh(
            vector_dir / "projection_weights_int8.memh", 8
        ).reshape(dim, dim)
        bias = load_signed_memh(vector_dir / "projection_bias_acc.memh", 32)
        expected = load_signed_memh(
            vector_dir / "expected_output_acc32.memh", 32
        ).reshape(162, dim)
        activations, _, _ = build_activations(
            raw_records, head_offset=head_offset, heads=heads
        )

        entries = 162 * dim
        acc = np.zeros((162, dim), dtype=np.int64)
        trackers = {width: PromotionTracker(width, entries) for width in WIDTHS}
        out_indices = np.arange(dim, dtype=np.int64)
        for token in range(162):
            for channel in range(dim):
                activation = int(activations[token, channel])
                if activation == 0:
                    continue
                old = acc[token].copy()
                new = old + activation * weights[:, channel]
                flat_indices = token * dim + out_indices
                for tracker in trackers.values():
                    tracker.update(flat_indices, old, new)
                acc[token] = new
        mismatch = int(np.count_nonzero(acc + bias[None, :] != expected))
        if mismatch:
            raise ValueError(f"historical Motion S{stage} Acc32 mismatch={mismatch}")
        stage_results.append(
            {
                "stage": stage,
                "heads": heads,
                "dim": dim,
                "mismatch": mismatch,
                "widths": {str(width): trackers[width].row(entries) for width in WIDTHS},
            }
        )
        head_offset += heads

    return {
        "scope": "historical H67 sample0/window0 T162 projection vectors, not current T450 paper population",
        "evidence": "[historical-rtl-vector]+[software-ordered-miter]",
        "mismatch": 0,
        "stages": stage_results,
        "provenance": {
            "raw_records": str(raw_records_path.resolve()),
            "raw_records_sha256": sha256(raw_records_path),
        },
    }


def motion_t450_profile(vector_manifest_path: Path) -> dict:
    manifest = json.loads(vector_manifest_path.read_text())
    records = manifest["records"]
    if len(records) != 12 or int(manifest["temporal_tokens"]) != 450:
        raise ValueError("Motion profile must be the locked ep35 all-12 T450 vector set")

    record_results = []
    for record in records:
        vector_dir = Path(record["vector_dir"])
        heads = int(record["heads"])
        dim = int(record["dim"])
        tokens_per_window = int(record["tokens"])
        if dim != heads * 32 or tokens_per_window != 450:
            raise ValueError(f"Motion vector shape mismatch: {record['name']}")

        head_offsets = load_unsigned_memh(vector_dir / "head_term_offsets.memh")
        term_offsets = load_unsigned_memh(vector_dir / "term_token_offsets.memh")
        gates = load_unsigned_memh(vector_dir / "term_gate_codes.memh")
        lanes = load_unsigned_memh(vector_dir / "term_lane_ids.memh")
        destinations = load_unsigned_memh(vector_dir / "term_tokens.memh")
        weights = load_signed_memh(
            vector_dir / "projection_weights_int8.memh", 8
        ).reshape(dim, dim)
        bias = load_signed_memh(vector_dir / "projection_bias_acc32.memh", 32)
        expected = load_signed_memh(
            vector_dir / "expected_output_acc32.memh", 32
        ).reshape(tokens_per_window, dim)

        total_terms = len(gates)
        if (
            len(head_offsets) != heads + 1
            or len(term_offsets) != total_terms + 1
            or len(lanes) != total_terms
            or int(head_offsets[-1]) != total_terms
            or int(term_offsets[-1]) != len(destinations)
        ):
            raise ValueError(f"Motion term stream boundaries mismatch: {record['name']}")

        entries = tokens_per_window * dim
        acc = np.zeros((tokens_per_window, dim), dtype=np.int64)
        trackers = {width: PromotionTracker(width, entries) for width in WIDTHS}
        out_indices = np.arange(dim, dtype=np.int64)
        for head in range(heads):
            for term in range(int(head_offsets[head]), int(head_offsets[head + 1])):
                lane = int(lanes[term])
                if lane < 0 or lane >= 32:
                    raise ValueError(f"Motion lane out of range: {record['name']} term={term}")
                gate = int(gates[term])
                delta = gate * weights[:, head * 32 + lane]
                start = int(term_offsets[term])
                stop = int(term_offsets[term + 1])
                for destination in destinations[start:stop]:
                    token = int(destination)
                    if token < 0 or token >= tokens_per_window:
                        raise ValueError(
                            f"Motion destination out of range: {record['name']} token={token}"
                        )
                    old = acc[token].copy()
                    new = old + delta
                    flat_indices = token * dim + out_indices
                    for tracker in trackers.values():
                        tracker.update(flat_indices, old, new)
                    acc[token] = new

        mismatch = int(np.count_nonzero(acc + bias[None, :] != expected))
        if mismatch:
            raise ValueError(f"Motion {record['name']} Acc32 mismatch={mismatch}")
        record_results.append(
            {
                "name": str(record["name"]),
                "stage": int(record["stage"]),
                "heads": heads,
                "dim": dim,
                "terms": total_terms,
                "events": len(destinations),
                "mismatch": mismatch,
                "widths": {
                    str(width): trackers[width].row(entries) for width in WIDTHS
                },
            }
        )

    return {
        "scope": "current H67 ep35 sample0/window0 all-12-block T450 projection term stream",
        "evidence": "[rtl-vector]+[real-checkpoint-int8]+[software-ordered-miter]",
        "records": len(record_results),
        "mismatch": 0,
        "results": record_results,
        "provenance": {
            "vector_manifest": str(vector_manifest_path.resolve()),
            "vector_manifest_sha256": sha256(vector_manifest_path),
            "source_manifest": str(manifest["source_manifest"]),
            "source_manifest_sha256": str(manifest["source_manifest_sha256"]),
        },
    }


def aggregate_records(records: list[dict]) -> dict:
    summary = {}
    for width in WIDTHS:
        rows = [record["widths"][str(width)] for record in records]
        peak = np.asarray([row["dynamic_peak_fraction"] for row in rows])
        access = np.asarray([row["dynamic_high_access_fraction"] for row in rows])
        state = np.asarray([row["dynamic_state_reduction_vs_acc32"] for row in rows])
        sticky = np.asarray([row["sticky_entries"] for row in rows])
        summary[str(width)] = {
            "dynamic_peak_fraction_mean": float(peak.mean()),
            "dynamic_peak_fraction_max": float(peak.max()),
            "dynamic_high_access_fraction_mean": float(access.mean()),
            "dynamic_high_access_fraction_max": float(access.max()),
            "state_reduction_mean": float(state.mean()),
            "state_reduction_min": float(state.min()),
            "sticky_entries_mean": float(sticky.mean()),
            "sticky_entries_max": int(sticky.max()),
        }
    return summary


def aggregate_local(local: dict) -> dict:
    summary = {}
    for width in WIDTHS:
        rows = [row["widths"][str(width)] for row in local["results"]]
        dynamic_peak = np.asarray([row["dynamic_peak_fraction"] for row in rows])
        dynamic_access = np.asarray([row["dynamic_high_access_fraction"] for row in rows])
        reduction = np.asarray([row["dynamic_state_reduction_vs_acc32"] for row in rows])
        sticky_entries = np.asarray([row["sticky_entries"] for row in rows])
        payload_ratio = np.asarray(
            [row["dynamic_accessed_payload_ratio_vs_acc32"] for row in rows]
        )
        summary[str(width)] = {
            "dynamic_peak_fraction_mean": float(dynamic_peak.mean()),
            "dynamic_peak_fraction_p95": float(np.quantile(dynamic_peak, 0.95)),
            "dynamic_peak_fraction_max": float(dynamic_peak.max()),
            "dynamic_high_access_fraction_mean": float(dynamic_access.mean()),
            "dynamic_high_access_fraction_p95": float(np.quantile(dynamic_access, 0.95)),
            "dynamic_high_access_fraction_max": float(dynamic_access.max()),
            "state_reduction_mean": float(reduction.mean()),
            "state_reduction_p05": float(np.quantile(reduction, 0.05)),
            "state_reduction_min": float(reduction.min()),
            "sticky_entries_mean": float(sticky_entries.mean()),
            "sticky_entries_p95": float(np.quantile(sticky_entries, 0.95)),
            "sticky_entries_max": int(sticky_entries.max()),
            "accessed_payload_ratio_mean": float(payload_ratio.mean()),
            "accessed_payload_ratio_max": float(payload_ratio.max()),
        }
    return summary


def compare_local_strong_baseline(local: dict) -> dict:
    b16_rows = [row["widths"]["16"] for row in local["results"]]
    b18_rows = [row["widths"]["18"] for row in local["results"]]
    state_advantage = np.asarray(
        [
            row18["dynamic_state_bits"] / row16["dynamic_state_bits"]
            for row16, row18 in zip(b16_rows, b18_rows, strict=True)
        ]
    )
    payload_reduction = np.asarray(
        [
            1.0
            - row16["dynamic_accessed_payload_ratio_vs_acc32"]
            / row18["dynamic_accessed_payload_ratio_vs_acc32"]
            for row16, row18 in zip(b16_rows, b18_rows, strict=True)
        ]
    )
    return {
        "baseline": "b18 plus the same exact overflow fallback contract",
        "baseline_observed_promotions": int(
            sum(row["dynamic_peak_entries"] for row in b18_rows)
        ),
        "b16_state_advantage_mean": float(state_advantage.mean()),
        "b16_state_advantage_p05": float(np.quantile(state_advantage, 0.05)),
        "b16_state_advantage_min": float(state_advantage.min()),
        "b16_accessed_payload_reduction_mean": float(payload_reduction.mean()),
        "b16_accessed_payload_reduction_p05": float(
            np.quantile(payload_reduction, 0.05)
        ),
        "b16_accessed_payload_reduction_min": float(payload_reduction.min()),
    }


def compare_record_strong_baseline(records: list[dict]) -> dict:
    b16_rows = [record["widths"]["16"] for record in records]
    b18_rows = [record["widths"]["18"] for record in records]
    state_advantage = np.asarray(
        [
            row18["dynamic_state_bits"] / row16["dynamic_state_bits"]
            for row16, row18 in zip(b16_rows, b18_rows, strict=True)
        ]
    )
    payload_reduction = np.asarray(
        [
            1.0
            - row16["dynamic_accessed_payload_ratio_vs_acc32"]
            / row18["dynamic_accessed_payload_ratio_vs_acc32"]
            for row16, row18 in zip(b16_rows, b18_rows, strict=True)
        ]
    )
    return {
        "baseline": "b18 plus the same exact overflow fallback contract",
        "b16_state_advantage_mean": float(state_advantage.mean()),
        "b16_state_advantage_min": float(state_advantage.min()),
        "b16_accessed_payload_reduction_mean": float(payload_reduction.mean()),
        "b16_accessed_payload_reduction_min": float(payload_reduction.min()),
    }


def render_markdown(report: dict) -> str:
    local = report["local5_summary"]
    lines = [
        "# Exact promoted accumulator profile",
        "",
        f"Status: **{report['status']}** (`{report['evidence']}`)",
        "",
        "## Local5 current population",
        "",
        "| Base width | Peak high occupancy mean/p95/max | High-path access mean/p95/max | State reduction mean/p05/min | Sticky entries mean/p95/max |",
        "|---:|---:|---:|---:|---:|",
    ]
    for width in WIDTHS:
        row = local[str(width)]
        lines.append(
            f"| {width} | {row['dynamic_peak_fraction_mean']:.3%} / "
            f"{row['dynamic_peak_fraction_p95']:.3%} / {row['dynamic_peak_fraction_max']:.3%} | "
            f"{row['dynamic_high_access_fraction_mean']:.3%} / "
            f"{row['dynamic_high_access_fraction_p95']:.3%} / {row['dynamic_high_access_fraction_max']:.3%} | "
            f"{row['state_reduction_mean']:.3f}x / {row['state_reduction_p05']:.3f}x / "
            f"{row['state_reduction_min']:.3f}x | {row['sticky_entries_mean']:.1f} / "
            f"{row['sticky_entries_p95']:.1f} / {row['sticky_entries_max']} |"
        )
    lines.extend(
        [
            "",
            "All 100 groups reproduce the existing Acc32 golden result. Empty groups remain legal zero-update groups.",
            "",
            "## Decision",
            "",
            f"- Local5 b=16 profile gates: `{json.dumps(report['local5_b16_gates'], sort_keys=True)}`.",
            "- The result is conditional profile evidence, not an RTL or PPA result. Exact behavior outside the observed population still requires a bounded spill/replay contract.",
            f"- Against the strong b18 base with the same fallback, b16 state advantage is only {report['strong_baseline']['b16_state_advantage_min']:.3f}x at worst and accessed-payload reduction is only {100*report['strong_baseline']['b16_accessed_payload_reduction_min']:.2f}% at worst; both miss the 20% incremental gate.",
            "- ISCAS 2025 overflow-aware partial-sum management already reduces local guard bits by spilling impending overflow. A2Q/AXE constrain training to avoid overflow, and MGS reorders sums. The present candidate is only defensible if address-resident sparse promotion beats those strong baselines under identical SRAM ports.",
            "",
            "## Motion current T450 vectors",
            "",
            f"- The current ep35 sample0/window0 vector set covers {report['motion_t450']['records']} blocks; all ordered projection outputs match Acc32.",
            f"- Motion b=16 high-path access max is {100*report['motion_t450_summary']['16']['dynamic_high_access_fraction_max']:.3f}% and state reduction min is {report['motion_t450_summary']['16']['state_reduction_min']:.3f}x versus a full Acc32 logical-bit model.",
            f"- Against the same-fallback b=18 baseline, Motion b=16 has only {report['motion_strong_baseline']['b16_state_advantage_min']:.3f}x worst-case state advantage and {100*report['motion_strong_baseline']['b16_accessed_payload_reduction_min']:.2f}% worst-case accessed-payload reduction; negative values mean b=16 is worse.",
            "- This is current T450 evidence, but still only sample0/window0 and not a multisample or full-encoder result.",
            "",
            "## Boundaries",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["claim_boundary"])
    return "\n".join(lines) + "\n"


def build_report(args: argparse.Namespace) -> dict:
    local = local5_profile(args.local_vector_dir, args.local_ordered_payload)
    motion = motion_t450_profile(args.motion_t450_vector_manifest)
    local_summary = aggregate_local(local)
    motion_summary = aggregate_records(motion["results"])
    motion_strong_baseline = compare_record_strong_baseline(motion["results"])
    strong_baseline = compare_local_strong_baseline(local)
    b16 = local_summary["16"]
    gates = {
        "acc32_mismatch_zero": local["mismatch"] == 0,
        "state_reduction_min_ge_1p5x": b16["state_reduction_min"] >= 1.5,
        "high_path_access_max_le_1pct": b16["dynamic_high_access_fraction_max"] <= 0.01,
        "observed_sticky_entries_max_le_64": b16["sticky_entries_max"] <= 64,
        "universal_sidecar_capacity_proven": False,
        "rtl_and_same_macro_ppa_complete": False,
        "incremental_state_advantage_ge_1p2x_vs_b18": strong_baseline[
            "b16_state_advantage_min"
        ]
        >= 1.2,
        "incremental_payload_reduction_ge_20pct_vs_b18": strong_baseline[
            "b16_accessed_payload_reduction_min"
        ]
        >= 0.20,
    }
    return {
        "schema": "exact_promoted_accumulator_profile_v1",
        "status": "NO_GO_AS_DATE_HOLD_AS_IMPLEMENTATION",
        "evidence": "[prof]+[rtl-vector]+[real-checkpoint-int8]+[software-ordered-miter]",
        "local5": local,
        "local5_summary": local_summary,
        "strong_baseline": strong_baseline,
        "motion_t450": motion,
        "motion_t450_summary": motion_summary,
        "motion_strong_baseline": motion_strong_baseline,
        "local5_b16_gates": gates,
        "related_work_boundary": [
            "ISCAS 2025 overflow-aware partial-sum management flushes an impending local overflow to an output buffer; DOI 10.1109/ISCAS56072.2025.11044224.",
            "A2Q/AXE use training or quantization constraints to guarantee low-precision accumulation; this screen does not alter the checkpoint.",
            "MGS changes reduction order to avoid transient overflow; this screen preserves hardware order and promotes the address instead.",
        ],
        "claim_boundary": [
            "No accumulator RTL, spill protocol, or sidecar memory was implemented.",
            "Logical bit counts are not SRAM macro area, dynamic energy, DC, STA, SAIF, or PTPX.",
            "A sidecar sized to the observed maximum is not universally exact; overflow must fail closed into a proved spill/replay or full-width fallback.",
            "The Local5 population is one OUT32 input-head tile per group, pre-bias/pre-BN/pre-requant and not cross-head or encoder.",
            "The Motion profile is current ep35 T450 but only sample0/window0; it cannot support a multisample or full-encoder claim.",
            "The candidate remains outside docs/359 frozen columns and outside the DATE contribution list.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-vector-dir",
        type=Path,
        default=Path("tb_qfit/vectors/local5_joint_ep29_score_projection_realw_sample100_population_out32_v1_20260814"),
    )
    parser.add_argument(
        "--local-ordered-payload",
        type=Path,
        default=Path("results/local5_fullres_bb1e4_joint_heads_profile100_20260809/ordered_term_items.npz"),
    )
    parser.add_argument(
        "--motion-t450-vector-manifest",
        type=Path,
        default=Path("results/h67_fullres_ep35_postconvergence_t450_20260805_checkpoint_projection_rtl/vectors_manifest.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/exact_promoted_accumulator_profile_20260814"),
    )
    args = parser.parse_args()
    report = build_report(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "report.md").write_text(render_markdown(report))
    print(json.dumps({"status": report["status"], "out_dir": str(args.out_dir)}))


if __name__ == "__main__":
    main()
