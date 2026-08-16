#!/usr/bin/env python3
"""评估 H67 的 TTB metadata-first exact cascade 与局部类商候选。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LANES = 32
SPATIAL_TOKENS = 225
DEFAULT_BUNDLES = (1, 2, 4, 8, 16, 32)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def distribution(values: list[int | float]) -> dict[str, float | int]:
    floats = [float(value) for value in values]
    return {
        "mean": sum(floats) / len(floats),
        "p50": percentile(floats, 0.50),
        "p95": percentile(floats, 0.95),
        "p99": percentile(floats, 0.99),
        "max": max(values),
    }


def parse_markdown_profile(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    sample_match = re.search(r"^- samples：(\d+)$", text, flags=re.MULTILINE)
    if sample_match is None:
        raise ValueError("profile markdown missing sample count")

    ttb_match = re.search(
        r"^\| 1 \| (?P<bundles>\d+) \| (?P<density>[0-9.]+)% \| "
        r"(?P<empty>[0-9.]+)% \| (?P<kzero>[0-9.]+)% \| "
        r"(?P<nomotion>[0-9.]+)% \|",
        text,
        flags=re.MULTILINE,
    )
    if ttb_match is None:
        raise ValueError("profile markdown missing True TTB B1 row")

    def metric(name: str) -> float:
        match = re.search(
            rf"^\| {re.escape(name)} \| ([0-9.]+)% \|$",
            text,
            flags=re.MULTILINE,
        )
        if match is None:
            raise ValueError(f"profile markdown missing metric: {name}")
        return float(match.group(1)) / 100.0

    ttb_rows: dict[int, dict[str, float | int]] = {}
    for match in re.finditer(
        r"^\| (?P<size>1|2|4|8) \| (?P<bundles>\d+) \| "
        r"(?P<density>[0-9.]+)% \| (?P<empty>[0-9.]+)% \| "
        r"(?P<kzero>[0-9.]+)% \| (?P<nomotion>[0-9.]+)% \|",
        text,
        flags=re.MULTILINE,
    ):
        size = int(match.group("size"))
        ttb_rows[size] = {
            "bundles": int(match.group("bundles")),
            "q_or_k_density": float(match.group("density")) / 100.0,
            "empty": float(match.group("empty")) / 100.0,
            "kzero": float(match.group("kzero")) / 100.0,
            "no_k_motion": float(match.group("nomotion")) / 100.0,
        }
    if set(ttb_rows) != {1, 2, 4, 8}:
        raise ValueError(f"incomplete TTB rows: {sorted(ttb_rows)}")

    return {
        "samples": int(sample_match.group(1)),
        "spatial_pairs": int(ttb_match.group("bundles")),
        "pair_empty": float(ttb_match.group("empty")) / 100.0,
        "both_kzero": metric("both K slices zero"),
        "no_k_motion": metric("K motion zero"),
        "per_token_kzero": metric("per-token K zero"),
        "paired_h67_equal": metric("H67 paired scores equal"),
        "ttb": ttb_rows,
    }


def parse_sample_csv(path: Path) -> dict[str, Any]:
    fields = (
        "pair_empty_ratio",
        "token_kzero_ratio",
        "s0_pair_empty_ratio",
        "s1_pair_empty_ratio",
        "s2_pair_empty_ratio",
        "s3_pair_empty_ratio",
    )
    columns: dict[str, list[float]] = {field: [] for field in fields}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for field in fields:
                columns[field].append(float(row[field]))
    if not columns[fields[0]]:
        raise ValueError("sample workload CSV is empty")
    return {
        "samples": len(columns[fields[0]]),
        "distributions": {
            field: distribution(values) for field, values in columns.items()
        },
    }


def round_even_silence(count: int, integer_base: int) -> int:
    quotient, remainder = divmod(count, 16)
    if remainder > 8 or (remainder == 8 and (integer_base + quotient) & 1):
        quotient += 1
    return quotient


def score_q7(q: int, current_k: int, peer_k: int) -> int:
    mask = (1 << LANES) - 1
    overlap = (q & current_k).bit_count()
    same_zero = ((~q) & (~current_k) & mask).bit_count()
    motion = (current_k ^ peer_k).bit_count()
    integer_base = 4 * overlap + motion
    return integer_base + round_even_silence(same_zero, integer_base)


def exhaustive_zero_k_closure() -> dict[str, Any]:
    by_qcount = {
        qcount: score_q7((1 << qcount) - 1, 0, 0)
        for qcount in range(LANES + 1)
    }
    ranges = []
    start = 0
    current = by_qcount[0]
    for qcount in range(1, LANES + 2):
        value = by_qcount.get(qcount)
        if value != current:
            ranges.append(
                {"qcount_min": start, "qcount_max": qcount - 1, "score_q7": current}
            )
            start = qcount
            current = value
    return {
        "classes": sorted(set(by_qcount.values())),
        "ranges": ranges,
        "exhaustive_qcount_cases": len(by_qcount),
    }


def load_rows(path: Path) -> tuple[int, list[dict[str, Any]]]:
    tokens = path.read_text(encoding="ascii").split()
    cursor = 0
    row_count = int(tokens[cursor])
    token_count = int(tokens[cursor + 1])
    cursor += 2
    if token_count != 2 * SPATIAL_TOKENS:
        raise ValueError(f"expected T450 rows, got {token_count}")
    rows: list[dict[str, Any]] = []
    for expected_row in range(row_count):
        row_id = int(tokens[cursor])
        stage = int(tokens[cursor + 1])
        block = int(tokens[cursor + 2])
        head = int(tokens[cursor + 3])
        expected_outputs = int(tokens[cursor + 4])
        expected_folded = int(tokens[cursor + 5])
        cursor += 6
        if row_id != expected_row:
            raise ValueError(f"row order mismatch: {row_id} != {expected_row}")
        vectors = []
        for _ in range(token_count):
            q = int(tokens[cursor], 16)
            current_k = int(tokens[cursor + 1], 16)
            peer_k = int(tokens[cursor + 2], 16)
            gate = int(tokens[cursor + 3])
            cursor += 4
            vectors.append((q, current_k, peer_k, gate))
        if sum(current_k != 0 for _, current_k, _, _ in vectors) != expected_outputs:
            raise ValueError(f"row {row_id} active-output ledger mismatch")
        if sum(current_k == 0 for _, current_k, _, _ in vectors) != expected_folded:
            raise ValueError(f"row {row_id} folded-output ledger mismatch")
        rows.append(
            {
                "row": row_id,
                "stage": stage,
                "block": block,
                "head": head,
                "vectors": vectors,
            }
        )
    if cursor != len(tokens):
        raise ValueError("vector file has trailing fields")
    return token_count, rows


def classify_pair(q0: int, q1: int, k0: int, k1: int) -> str:
    if q0 == 0 and q1 == 0 and k0 == 0 and k1 == 0:
        return "empty"
    if k0 == 0 and k1 == 0:
        return "kzero_nonempty"
    if k0 == k1:
        return "motionzero_nonkzero"
    return "full"


def analyze_vectors(
    rows: list[dict[str, Any]],
    bundle_sizes: tuple[int, ...],
) -> dict[str, Any]:
    result: dict[int, dict[str, Any]] = {}
    row_details: dict[int, list[dict[str, Any]]] = {size: [] for size in bundle_sizes}
    category_total: Counter[str] = Counter()
    stage_categories: dict[int, Counter[str]] = defaultdict(Counter)
    paired_equal = 0
    active_paired_equal = 0
    active_pair_total = 0
    total_pairs = 0
    zero_k_score_classes: set[int] = set()

    for row in rows:
        vectors = row["vectors"]
        pairs = []
        for spatial in range(SPATIAL_TOKENS):
            q0, k0, peer0, _ = vectors[spatial]
            q1, k1, peer1, _ = vectors[SPATIAL_TOKENS + spatial]
            if peer0 != k1 or peer1 != k0:
                raise ValueError(
                    f"row {row['row']} spatial {spatial} temporal-peer mismatch"
                )
            score0 = score_q7(q0, k0, k1)
            score1 = score_q7(q1, k1, k0)
            category = classify_pair(q0, q1, k0, k1)
            pairs.append((score0, score1, category, int(k0 != 0) + int(k1 != 0)))
            category_total[category] += 1
            stage_categories[row["stage"]][category] += 1
            paired_equal += int(score0 == score1)
            if category in ("empty", "kzero_nonempty"):
                zero_k_score_classes.update((score0, score1))
            else:
                active_pair_total += 1
                active_paired_equal += int(score0 == score1)
            total_pairs += 1

        for size in bundle_sizes:
            unique_counts = []
            active_counts = []
            class_descriptors = 0
            for start in range(0, SPATIAL_TOKENS, size):
                scores = []
                active_count = 0
                for score0, score1, category, _ in pairs[start:start + size]:
                    scores.extend((score0, score1))
                    active_count += int(category not in ("empty", "kzero_nonempty"))
                unique = len(set(scores))
                unique_counts.append(unique)
                active_counts.append(active_count)
                class_descriptors += unique
            row_details[size].append(
                {
                    "row": row["row"],
                    "stage": row["stage"],
                    "block": row["block"],
                    "head": row["head"],
                    "groups": len(unique_counts),
                    "class_descriptors": class_descriptors,
                    "rqtb_descriptors": sum(
                        1 if score0 == score1 else 2
                        for score0, score1, _, _ in pairs
                    ),
                    "active_pairs": sum(
                        category not in ("empty", "kzero_nonempty")
                        for _, _, category, _ in pairs
                    ),
                    "active_k_tokens": sum(active for _, _, _, active in pairs),
                    "active_rqtb_descriptors": sum(
                        (1 if score0 == score1 else 2)
                        for score0, score1, category, _ in pairs
                        if category not in ("empty", "kzero_nonempty")
                    ),
                    "active_counts_per_group": active_counts,
                    "unique_per_group": unique_counts,
                }
            )

    raw_slots = len(rows) * 2 * SPATIAL_TOKENS
    rqtb_slots = sum(
        detail["rqtb_descriptors"] for detail in row_details[bundle_sizes[0]]
    )
    active_rqtb_slots = sum(
        detail["active_rqtb_descriptors"] for detail in row_details[bundle_sizes[0]]
    )
    for size in bundle_sizes:
        details = row_details[size]
        unique_counts = [
            value for detail in details for value in detail["unique_per_group"]
        ]
        class_descriptors = sum(detail["class_descriptors"] for detail in details)
        result[size] = {
            "groups": sum(detail["groups"] for detail in details),
            "class_descriptors": class_descriptors,
            "reduction_vs_raw": 1.0 - class_descriptors / raw_slots,
            "reduction_vs_rqtb": 1.0 - class_descriptors / rqtb_slots,
            "unique_classes_per_group": distribution(unique_counts),
        }

    return {
        "rows": len(rows),
        "raw_score_slots": raw_slots,
        "spatial_pairs": total_pairs,
        "paired_score_equal": paired_equal / total_pairs,
        "active_paired_score_equal": active_paired_equal / active_pair_total,
        "rqtb_slots": rqtb_slots,
        "active_rqtb_slots": active_rqtb_slots,
        "active_command_reduction_vs_rqtb": 1.0 - active_rqtb_slots / rqtb_slots,
        "zero_k_score_classes": sorted(zero_k_score_classes),
        "categories": dict(category_total),
        "stage_categories": {
            str(stage): dict(counter) for stage, counter in sorted(stage_categories.items())
        },
        "bundles": {str(size): value for size, value in result.items()},
        "row_details": {str(size): value for size, value in row_details.items()},
    }


def simulate_decoupled_bundle_queue(active_counts: list[int]) -> dict[str, int]:
    """一拍接收一个bundle descriptor，一拍消费一个active pair。"""
    queue: list[int] = []
    producer = 0
    cycles = 0
    consumed = 0
    max_descriptors = 0
    max_active_backlog = 0
    while producer < len(active_counts) or queue:
        if producer < len(active_counts):
            active = active_counts[producer]
            producer += 1
            if active:
                queue.append(active)
        max_descriptors = max(max_descriptors, len(queue))
        max_active_backlog = max(max_active_backlog, sum(queue))
        if queue:
            queue[0] -= 1
            consumed += 1
            if queue[0] == 0:
                queue.pop(0)
        cycles += 1
    if consumed != sum(active_counts):
        raise RuntimeError("decoupled queue consumption mismatch")
    return {
        "cycles": cycles,
        "max_bundle_descriptors": max_descriptors,
        "max_active_pair_backlog": max_active_backlog,
    }


def full_profile_cost_model(profile: dict[str, Any], bundle_size: int) -> dict[str, Any]:
    empty = profile["pair_empty"]
    kzero_nonempty = profile["both_kzero"] - empty
    motionzero_nonkzero = profile["no_k_motion"] - profile["both_kzero"]
    full = 1.0 - profile["no_k_motion"]
    categories = {
        "empty": empty,
        "kzero_nonempty": kzero_nonempty,
        "motionzero_nonkzero": motionzero_nonkzero,
        "full": full,
    }
    if any(value < -1e-9 for value in categories.values()):
        raise ValueError(f"non-monotonic full-profile categories: {categories}")
    if abs(sum(categories.values()) - 1.0) > 1e-6:
        raise ValueError(f"full-profile categories do not sum to one: {categories}")

    # 每个 temporal pair 的传统 score 布尔工作量为 2 个 score x 3 个 32-lane 项。
    metadata_assisted_ratio = motionzero_nonkzero * (2.0 / 3.0) + full
    conservative_qcount_ratio = metadata_assisted_ratio + kzero_nonempty / 3.0

    # temporal pair 原始唯一 payload 为 {Q0,Q1,K0,K1}=128 bit。
    payload_without_header = (
        2.0
        + kzero_nonempty * 12.0
        + motionzero_nonkzero * 96.0
        + full * 128.0
    )
    preclassified_payload_without_header = (
        2.0
        + kzero_nonempty * 4.0
        + motionzero_nonkzero * 96.0
        + full * 128.0
    )
    payload = {}
    preclassified_payload = {}
    for header_bits in (0, 32, 64):
        bits_per_pair = payload_without_header + header_bits / bundle_size
        preclassified_bits_per_pair = (
            preclassified_payload_without_header + header_bits / bundle_size
        )
        payload[str(header_bits)] = {
            "bits_per_pair": bits_per_pair,
            "reduction_vs_128b": 1.0 - bits_per_pair / 128.0,
        }
        preclassified_payload[str(header_bits)] = {
            "bits_per_pair": preclassified_bits_per_pair,
            "reduction_vs_128b": 1.0 - preclassified_bits_per_pair / 128.0,
        }

    return {
        "disjoint_categories": categories,
        "score_boolean_lane_work": {
            "baseline_per_pair": 192,
            "metadata_assisted_ratio": metadata_assisted_ratio,
            "metadata_assisted_reduction": 1.0 - metadata_assisted_ratio,
            "conservative_qcount_ratio": conservative_qcount_ratio,
            "conservative_qcount_reduction": 1.0 - conservative_qcount_ratio,
            "motion_xor_branch_reduction": profile["no_k_motion"],
        },
        "payload_model": {
            "baseline_bits_per_pair": 128,
            "category_bits_per_pair": 2,
            "bundle_size": bundle_size,
            "qcount_12bit_header_sensitivity": payload,
            "preclassified_4bit_header_sensitivity": preclassified_payload,
            "header_sensitivity": payload,
        },
        "gated_k_emit_reduction": profile["per_token_kzero"],
    }


def cycle_model(
    vector: dict[str, Any],
    rqtb_report: dict[str, Any],
    bundle_size: int,
) -> dict[str, Any]:
    rows = vector["row_details"][str(bundle_size)]
    baseline_rows = rqtb_report.get("rows_2s", [])
    if len(rows) != len(baseline_rows):
        raise ValueError("RQTB row count differs from vector row count")
    serial_total = 0
    overlap_total = 0
    zk_decoupled_total = 0
    baseline_total = 0
    queue_depths: list[int] = []
    active_backlogs: list[int] = []
    detail_out = []
    for candidate, baseline in zip(rows, baseline_rows):
        if candidate["row"] != int(baseline["row"]):
            raise ValueError("RQTB row order differs from vector rows")
        residual = max(int(baseline["rqtb_cycles"]) - SPATIAL_TOKENS, 0)
        class_commit = math.ceil(candidate["class_descriptors"] / 2)
        serial_front = candidate["groups"] + candidate["active_pairs"] + class_commit
        overlap_front = max(candidate["groups"], candidate["active_pairs"], class_commit)
        zk_queue = simulate_decoupled_bundle_queue(candidate["active_counts_per_group"])
        serial_cycles = residual + serial_front
        overlap_cycles = residual + overlap_front
        zk_decoupled_cycles = residual + zk_queue["cycles"]
        baseline_cycles = int(baseline["rqtb_cycles"])
        serial_total += serial_cycles
        overlap_total += overlap_cycles
        zk_decoupled_total += zk_decoupled_cycles
        baseline_total += baseline_cycles
        queue_depths.append(zk_queue["max_bundle_descriptors"])
        active_backlogs.append(zk_queue["max_active_pair_backlog"])
        detail_out.append(
            {
                "row": candidate["row"],
                "stage": candidate["stage"],
                "baseline_rqtb2s_cycles": baseline_cycles,
                "residual_cycles": residual,
                "metadata_groups": candidate["groups"],
                "active_pair_cycles": candidate["active_pairs"],
                "class_commit_2wide_cycles": class_commit,
                "serial_candidate_cycles": serial_cycles,
                "overlap_candidate_cycles": overlap_cycles,
                "zk_decoupled_front_cycles": zk_queue["cycles"],
                "zk_decoupled_candidate_cycles": zk_decoupled_cycles,
                "zk_max_bundle_descriptors": zk_queue["max_bundle_descriptors"],
                "zk_max_active_pair_backlog": zk_queue["max_active_pair_backlog"],
            }
        )
    recorded_total = int(rqtb_report["cycles"]["totals"]["rqtb_2s"])
    if baseline_total != recorded_total:
        raise ValueError(f"RQTB total mismatch: {baseline_total} != {recorded_total}")
    return {
        "evidence": "[模型] sample0/window0；以实测RQTB2S逐行周期为底座",
        "assumption": (
            "每行仅从RQTB2S周期中扣除固定225个pair ingest cycle；metadata、"
            "active-pair score和2-wide class commit分别串行或理想重叠；"
            "其余SCS/gated-K/反压周期保持不变"
        ),
        "zk_assumption": (
            "TTB metadata每拍接收一个bundle；zero-K三类直接更新专用计数器；"
            "active bundle以指针和mask进入有序FIFO，score核每拍消费一个pair；"
            "Q/K payload保留在现有row SRAM而不复制进FIFO；其余周期保持不变"
        ),
        "baseline_rqtb2s_cycles": baseline_total,
        "serial_candidate_cycles": serial_total,
        "overlap_candidate_cycles": overlap_total,
        "zk_decoupled_candidate_cycles": zk_decoupled_total,
        "serial_speedup": baseline_total / serial_total,
        "overlap_speedup": baseline_total / overlap_total,
        "zk_decoupled_speedup": baseline_total / zk_decoupled_total,
        "serial_cycle_reduction": 1.0 - serial_total / baseline_total,
        "overlap_cycle_reduction": 1.0 - overlap_total / baseline_total,
        "zk_decoupled_cycle_reduction": 1.0 - zk_decoupled_total / baseline_total,
        "zk_queue_depth_distribution": distribution(queue_depths),
        "zk_active_backlog_distribution": distribution(active_backlogs),
        "rows": detail_out,
    }


def make_markdown(report: dict[str, Any]) -> str:
    full = report["full_profile100"]
    model = report["full_profile_cost_model"]
    sample = report["sample0_exact"]
    cycle = report["cycle_model"]
    closure = report["zero_k_exhaustive_closure"]
    selected = report["selected_bundle"]
    lines = [
        "# Motion Exact Metadata-Cascade 与 TTB 类商架构准入",
        "",
        "## 1. 结论",
        "",
        f"- `[prof]` H67 fullres profile100 中，temporal pair empty 为 "
        f"`{full['pair_empty']:.2%}`，both-K-zero 为 `{full['both_kzero']:.2%}`，"
        f"no-K-motion 为 `{full['no_k_motion']:.2%}`；",
        f"- `[模型]` 在 K-zero 路径复用事件元数据 q-count 时，score 布尔 lane 工作"
        f"减少 `{model['score_boolean_lane_work']['metadata_assisted_reduction']:.2%}`；"
        f"即使把 q-count popcount 计回本核，仍减少 "
        f"`{model['score_boolean_lane_work']['conservative_qcount_reduction']:.2%}`；",
        f"- `[模型]` TTB{selected} 含 32-bit header 时，Q/K payload 位数相对"
        f"128-bit temporal pair 减少 "
        f"`{model['payload_model']['preclassified_4bit_header_sensitivity']['32']['reduction_vs_128b']:.2%}`；",
        f"- `[prof-sample0]` TTB{selected} 局部类商把 score-class command 从 RQTB 的 "
        f"`{sample['rqtb_slots']}` 降为 "
        f"`{sample['bundles'][str(selected)]['class_descriptors']}`，减少 "
        f"`{sample['bundles'][str(selected)]['reduction_vs_rqtb']:.2%}`；",
        f"- `[prof-sample0]` 新准入方案只让非 K-zero pair 进入 RQTB，active command "
        f"为 `{sample['active_rqtb_slots']}`，相对原 RQTB 减少 "
        f"`{sample['active_command_reduction_vs_rqtb']:.2%}`；",
        f"- `[模型]` 以真实 RQTB2S 逐行周期为底座，通用类商完全串行前端仅为 "
        f"`{cycle['serial_speedup']:.3f}x`，判为负结果；three-class zero-K direct "
        f"injection 与 active-score descriptor 解耦后为 "
        f"`{cycle['zk_decoupled_speedup']:.3f}x`；",
        f"- 架构准入：`{'PASS' if report['admission']['pass'] else 'FAIL'}`。"
        "PASS 只允许进入最小 RTL，FAIL 则禁止进入；二者都不是性能结论。",
        "",
        "## 2. 候选数据流",
        "",
        "```text",
        "event coder / TTB metadata",
        "  {both-K-zero, zk_class0/1, active_mask, payload pointer}",
        "    |",
        "    +-> zero-K stream：3-class direct multiplicity injection",
        "    |                  （专用3计数器，不经过通用class FIFO）",
        "    |",
        "    +-> active bundle FIFO -> one pair/cycle full H67 score",
        "                            -> temporal RQTB quotient",
        "    |",
        "    +-> L2 motion-zero branch gate（仅非K-zero中的极少数）",
        "    -> shared weighted SCS merge -> gated-K emit",
        "```",
        "",
        "这不是删除 empty/K-zero token。both-K-zero 时，H67 score 由 q-count 精确"
        "化为 0/1/2 三类；三个 multiplicity counter 必须参与 Shiftmax 分母。只有"
        "K=0 的 gated-K payload 可以不写 active bank。",
        "",
        "对全部 33 个 q-count 穷举得到：",
        "",
        "| q-count | exact score class |",
        "|---:|---:|",
    ]
    for item in closure["ranges"]:
        lines.append(
            f"| {item['qcount_min']}..{item['qcount_max']} | {item['score_q7']} |"
        )
    lines.extend(
        [
            "",
            "## 3. Profile100 四级分流",
            "",
            "| 路径 | 占比 | 硬件行为 |",
            "|---|---:|---|",
        ]
    )
    labels = {
        "empty": "L0 empty",
        "kzero_nonempty": "L1 K-zero nonempty",
        "motionzero_nonkzero": "L2 motion-zero non-K-zero",
        "full": "L3 full",
    }
    actions = {
        "empty": "不读 Q/K payload；固定类计数",
        "kzero_nonempty": "只读 q-count；LUT 得到类计数",
        "motionzero_nonkzero": "读 Q 与单份 K；关闭 XOR 分支",
        "full": "读 Q0/Q1/K0/K1；完整 score",
    }
    for key, value in model["disjoint_categories"].items():
        lines.append(f"| {labels[key]} | {value:.4%} | {actions[key]} |")
    lines.extend(
        [
            "",
            "### 3.1 Payload 灵敏度",
            "",
            "以下按 event coder 已输出两个 2-bit zero-K class 计算：",
            "这里的 payload 只指 score 阶段从 row SRAM 的读取，不包含两种方案共同的"
            "Q/K 首次写入。",
            "",
            "| 每 bundle header | bits/temporal pair | 相对 128-bit 减少 |",
            "|---:|---:|---:|",
        ]
    )
    for header, values in model["payload_model"]["preclassified_4bit_header_sensitivity"].items():
        lines.append(
            f"| {header} bit | {values['bits_per_pair']:.3f} | "
            f"{values['reduction_vs_128b']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 4. Sample0 精确类商 DSE",
            "",
            "| spatial/TTB | group | class command | vs raw | vs RQTB | unique/group mean/p95/p99/max |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for size, values in sample["bundles"].items():
        dist = values["unique_classes_per_group"]
        lines.append(
            f"| {size} | {values['groups']} | {values['class_descriptors']} | "
            f"{values['reduction_vs_raw']:.2%} | {values['reduction_vs_rqtb']:.2%} | "
            f"{dist['mean']:.2f}/{dist['p95']:.1f}/{dist['p99']:.1f}/{dist['max']} |"
        )
    lines.extend(
        [
            "",
            "RQTB 是公平强基线；raw 只用于解释代数压缩上限。TTB1 等价于每个"
            "temporal pair 内取 score-class 商，必须与 RQTB slot 数精确一致。",
            "",
            "## 5. 周期模型边界与负结果",
            "",
            f"- RQTB2S `[rtl]`：`{cycle['baseline_rqtb2s_cycles']}` cycle；",
            f"- 通用 TTB{selected} 类商串行前端 `[模型]`："
            f"`{cycle['serial_candidate_cycles']}` cycle，`{cycle['serial_speedup']:.3f}x`；",
            f"- three-class zero-K direct injection + active bundle queue `[模型]`："
            f"`{cycle['zk_decoupled_candidate_cycles']}` cycle，"
            f"`{cycle['zk_decoupled_speedup']:.3f}x`；",
            f"- active bundle FIFO descriptor depth p95/max："
            f"`{cycle['zk_queue_depth_distribution']['p95']:.1f}/"
            f"{cycle['zk_queue_depth_distribution']['max']}`；active-pair backlog p95/max："
            f"`{cycle['zk_active_backlog_distribution']['p95']:.1f}/"
            f"{cycle['zk_active_backlog_distribution']['max']}`；",
            f"- 假设：{cycle['assumption']}。",
            f"- zero-K/active 解耦假设：{cycle['zk_assumption']}。",
            "",
            "通用类商串行版本虽然 command 更少，但周期门槛失败，因此明确否决；"
            "不能只凭压缩率进入 RTL。新模型没有计入新增 metadata SRAM、bundle "
            "assembler、three-counter merge 旁路"
            "和真实 SRAM latency，因此不能作为论文性能结果。最小 RTL 必须与 RQTB2S"
            "共享 score lane 数、FIFO 端口和 SCS backend，并在随机反压下重新测周期。",
            "",
            "## 6. 准入门槛",
            "",
            "| 门槛 | 结果 |",
            "|---|---|",
        ]
    )
    for check in report["admission"]["checks"]:
        lines.append(
            f"| {check['name']} | {'PASS' if check['pass'] else 'FAIL'}："
            f"{check['value']} |"
        )
    lines.extend(
        [
            "",
            "## 7. 证据边界与下一步",
            "",
            "当前允许声称：通用类商串行实现被负结果否决；profile100 支持把"
            "three-class zero-K direct injection 与 active-score descriptor 解耦作为"
            "Motion 新机制候选，其 sample0 周期模型通过准入。",
            "",
            "当前不允许声称：已实现 TTB cascade RTL、已取得端到端加速、已节能、"
            "已优于 Bishop/Prosperity，或该组合本身已经达到 DATE 新颖性要求。",
            "",
            "下一包只做一个最高优先级缺口：构造与 RQTB2S 等 lane/等 FIFO/共享 SCS"
            "的最小 TTB8 three-class injection + active bundle queue RTL，对同一"
            "138-row trace 做 Acc32、随机反压、周期和 payload-toggle 对照；若 RTL"
            "收益低于 10%，停止扩展。",
            "",
            "## 8. 可复现性",
            "",
            f"- profile Markdown SHA-256：`{report['provenance']['profile_md_sha256']}`；",
            f"- sample CSV SHA-256：`{report['provenance']['sample_csv_sha256']}`；",
            f"- row vector SHA-256：`{report['provenance']['vectors_sha256']}`；",
            f"- RQTB report SHA-256：`{report['provenance']['rqtb_report_sha256']}`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-md", type=Path, required=True)
    parser.add_argument("--sample-csv", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--rqtb-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-bundle", type=int, default=8)
    args = parser.parse_args()

    if args.selected_bundle not in DEFAULT_BUNDLES:
        raise ValueError(f"unsupported bundle size: {args.selected_bundle}")
    profile = parse_markdown_profile(args.profile_md)
    sample_csv = parse_sample_csv(args.sample_csv)
    if sample_csv["samples"] != profile["samples"]:
        raise ValueError("profile Markdown and sample CSV sample count differ")
    _, rows = load_rows(args.vectors)
    vector = analyze_vectors(rows, DEFAULT_BUNDLES)
    rqtb = json.loads(args.rqtb_report.read_text(encoding="utf-8"))
    if vector["rqtb_slots"] != int(rqtb["work"]["rqtb_slots"]):
        raise ValueError("TTB1 class quotient does not match recorded RQTB slots")
    full_model = full_profile_cost_model(profile, args.selected_bundle)
    cycles = cycle_model(vector, rqtb, args.selected_bundle)
    selected = vector["bundles"][str(args.selected_bundle)]
    closure = exhaustive_zero_k_closure()

    checks = [
        {
            "name": "profile100 K-zero exact path >= 60%",
            "pass": profile["both_kzero"] >= 0.60,
            "value": f"{profile['both_kzero']:.2%}",
        },
        {
            "name": "含32-bit header的payload减少 >= 60%",
            "pass": full_model["payload_model"]["preclassified_4bit_header_sensitivity"]["32"]
            ["reduction_vs_128b"] >= 0.60,
            "value": (
                f"{full_model['payload_model']['preclassified_4bit_header_sensitivity']['32']['reduction_vs_128b']:.2%}"
            ),
        },
        {
            "name": "保守score lane工作减少 >= 60%",
            "pass": full_model["score_boolean_lane_work"]
            ["conservative_qcount_reduction"] >= 0.60,
            "value": (
                f"{full_model['score_boolean_lane_work']['conservative_qcount_reduction']:.2%}"
            ),
        },
        {
            "name": "zero-K bypass后active command减少 >= 30%",
            "pass": vector["active_command_reduction_vs_rqtb"] >= 0.30,
            "value": f"{vector['active_command_reduction_vs_rqtb']:.2%}",
        },
        {
            "name": "both-K-zero score class 精确闭合为 {0,1,2}",
            "pass": closure["classes"] == [0, 1, 2]
            and vector["zero_k_score_classes"] == closure["classes"],
            "value": (
                f"穷举{closure['exhaustive_qcount_cases']}例={closure['classes']}，"
                f"trace={vector['zero_k_score_classes']}"
            ),
        },
        {
            "name": "zero-K/active 解耦周期模型减少 >= 10%",
            "pass": cycles["zk_decoupled_cycle_reduction"] >= 0.10,
            "value": f"{cycles['zk_decoupled_cycle_reduction']:.2%}",
        },
    ]
    report = {
        "schema": "h67_exact_metadata_cascade_profile_v1",
        "status": "PASS",
        "evidence": "[prof] profile100 + [prof-sample0] exact vector + [模型] cycle/payload",
        "selected_bundle": args.selected_bundle,
        "full_profile100": profile,
        "sample_variability": sample_csv,
        "full_profile_cost_model": full_model,
        "zero_k_exhaustive_closure": closure,
        "sample0_exact": vector,
        "cycle_model": cycles,
        "admission": {
            "pass": all(check["pass"] for check in checks),
            "checks": checks,
            "meaning": "只允许进入等资源最小RTL，不代表性能或DATE新颖性成立",
        },
        "provenance": {
            "profile_md": str(args.profile_md.resolve()),
            "profile_md_sha256": file_sha256(args.profile_md),
            "sample_csv": str(args.sample_csv.resolve()),
            "sample_csv_sha256": file_sha256(args.sample_csv),
            "vectors": str(args.vectors.resolve()),
            "vectors_sha256": file_sha256(args.vectors),
            "rqtb_report": str(args.rqtb_report.resolve()),
            "rqtb_report_sha256": file_sha256(args.rqtb_report),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(make_markdown(report), encoding="utf-8")
    print(
        "PASS H67 exact metadata cascade profile "
        f"bundle={args.selected_bundle} admission={report['admission']['pass']} "
        f"serial_speedup={cycles['serial_speedup']:.4f} "
        f"zk_decoupled_speedup={cycles['zk_decoupled_speedup']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
