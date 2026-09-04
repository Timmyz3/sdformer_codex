#!/usr/bin/env python3
"""Profile source-owned sufficient statistics for exact Local5 Q-silent scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


TOKENS = 450
PLANE_TOKENS = 225
SIDE = 15
ROLES = 5
ROLE_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_memh(path: Path) -> list[int]:
    return [
        int(line.strip(), 16)
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]


def source_id(destination: int, role: int) -> int | None:
    plane, within = divmod(destination, PLANE_TOKENS)
    y, x = divmod(within, SIDE)
    dy, dx = ROLE_OFFSETS[role]
    sy, sx = y + dy, x + dx
    if not (0 <= sy < SIDE and 0 <= sx < SIDE):
        return None
    return plane * PLANE_TOKENS + sy * SIDE + sx


def rne_div16_nonnegative(value: int) -> int:
    quotient, remainder = divmod(value, 16)
    return quotient + int(remainder > 8 or (remainder == 8 and quotient & 1))


def percentile(values: list[int], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "qsilent_destinations",
        "valid_edges",
        "qsilent_edges",
        "nonsilent_edges",
        "unique_qsilent_sources",
        "baseline_qsilent_popcounts",
        "source_owned_all_source_popcounts",
        "source_owned_demand_popcounts",
        "baseline_score_k_read_bits",
        "source_owned_score_k_read_bits",
        "source_stat_route_bits",
        "source_stat_write_bits",
    )
    totals = {field: int(sum(int(row[field]) for row in rows)) for field in fields}
    baseline_pop = totals["baseline_qsilent_popcounts"]
    all_source_pop = totals["source_owned_all_source_popcounts"]
    demand_pop = totals["source_owned_demand_popcounts"]
    baseline_bits = totals["baseline_score_k_read_bits"]
    source_bits = totals["source_owned_score_k_read_bits"]
    return {
        "groups": len(rows),
        "totals": totals,
        "reductions": {
            "popcount_all_source": (
                0.0 if baseline_pop == 0 else 1.0 - all_source_pop / baseline_pop
            ),
            "popcount_demand": (
                0.0 if baseline_pop == 0 else 1.0 - demand_pop / baseline_pop
            ),
            "score_k_read_bits": (
                0.0 if baseline_bits == 0 else 1.0 - source_bits / baseline_bits
            ),
        },
        "distribution": {
            field: {
                "mean": float(np.mean([int(row[field]) for row in rows])),
                "p50": percentile([int(row[field]) for row in rows], 50),
                "p95": percentile([int(row[field]) for row in rows], 95),
                "p99": percentile([int(row[field]) for row in rows], 99),
                "max": max(int(row[field]) for row in rows),
            }
            for field in fields
        },
    }


def analyze(vector_dir: Path) -> dict[str, Any]:
    manifest_path = vector_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "local5_score_projection_vectors_v1":
        raise ValueError("输入不是Local5 score/projection v1向量")
    rows_meta = manifest.get("selection", {}).get("rows")
    if not isinstance(rows_meta, list) or not rows_meta:
        raise ValueError("manifest缺少selection.rows")
    artifacts = manifest.get("artifacts", {})
    required = ("input_q", "input_candidate_k", "input_valid", "expected_scores")
    for name in required:
        item = artifacts.get(name)
        if not isinstance(item, dict):
            raise ValueError(f"manifest缺少{name}")
        path = vector_dir / str(item.get("file", ""))
        if not path.is_file() or sha256(path) != item.get("sha256"):
            raise ValueError(f"{name}文件或SHA不匹配")

    q_values = load_memh(vector_dir / artifacts["input_q"]["file"])
    candidate_values = load_memh(
        vector_dir / artifacts["input_candidate_k"]["file"]
    )
    valid_values = load_memh(vector_dir / artifacts["input_valid"]["file"])
    score_values = load_memh(vector_dir / artifacts["expected_scores"]["file"])
    expected = len(rows_meta) * TOKENS
    if not all(
        len(values) == expected
        for values in (q_values, candidate_values, valid_values, score_values)
    ):
        raise ValueError("T450向量长度不一致")

    rows: list[dict[str, Any]] = []
    edge_k_mismatches = 0
    score_mismatches = 0
    checked_qsilent_edges = 0
    for group_index, meta in enumerate(rows_meta):
        base = group_index * TOKENS
        group_q = q_values[base : base + TOKENS]
        group_k = candidate_values[base : base + TOKENS]
        group_valid = valid_values[base : base + TOKENS]
        group_scores = score_values[base : base + TOKENS]
        self_k = [int(value) & 0xFFFFFFFF for value in group_k]
        touched_sources: set[int] = set()
        qsilent_destinations = 0
        valid_edges = 0
        qsilent_edges = 0
        nonsilent_edges = 0

        for destination in range(TOKENS):
            qsilent = int(group_q[destination]) == 0
            if qsilent:
                qsilent_destinations += 1
            for role in range(ROLES):
                valid = bool((int(group_valid[destination]) >> role) & 1)
                mapped_source = source_id(destination, role)
                if valid != (mapped_source is not None):
                    raise ValueError(
                        f"group {group_index} token {destination} role {role} "
                        "valid与固定拓扑边界不一致"
                    )
                if not valid:
                    continue
                assert mapped_source is not None
                valid_edges += 1
                candidate_k = (int(group_k[destination]) >> (32 * role)) & 0xFFFFFFFF
                if candidate_k != self_k[mapped_source]:
                    edge_k_mismatches += 1
                if qsilent:
                    qsilent_edges += 1
                    touched_sources.add(mapped_source)
                    expected_score = (int(group_scores[destination]) >> (16 * role)) & 0xFFFF
                    computed_score = rne_div16_nonnegative(
                        32 - int(candidate_k).bit_count()
                    )
                    if expected_score != computed_score:
                        score_mismatches += 1
                    checked_qsilent_edges += 1
                else:
                    nonsilent_edges += 1

        rows.append(
            {
                "group": group_index,
                "sample": int(meta["sample"]),
                "stage": int(meta["stage"]),
                "block": int(meta["block"]),
                "window": int(meta["window"]),
                "head": int(meta["head"]),
                "qsilent_destinations": qsilent_destinations,
                "valid_edges": valid_edges,
                "qsilent_edges": qsilent_edges,
                "nonsilent_edges": nonsilent_edges,
                "unique_qsilent_sources": len(touched_sources),
                "baseline_qsilent_popcounts": qsilent_edges,
                "source_owned_all_source_popcounts": TOKENS,
                "source_owned_demand_popcounts": len(touched_sources),
                "baseline_score_k_read_bits": valid_edges * 32,
                "source_owned_score_k_read_bits": (
                    nonsilent_edges * 32 + qsilent_edges * 6
                ),
                "source_stat_route_bits": qsilent_edges * 6,
                "source_stat_write_bits": TOKENS * 6,
            }
        )

    if edge_k_mismatches or score_mismatches:
        raise AssertionError(
            f"source-owned exact检查失败: edge_k={edge_k_mismatches}, "
            f"score={score_mismatches}"
        )

    stage_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        stage_rows[int(row["stage"])].append(row)
    return {
        "schema": "local5_source_owned_qsilent_profile_v1",
        "status": "PROFILE_ONLY",
        "source": {
            "vector_manifest": str(manifest_path.resolve()),
            "vector_manifest_sha256": sha256(manifest_path),
            "evidence": manifest.get("evidence"),
            "groups": len(rows_meta),
            "samples": len({int(row["sample"]) for row in rows_meta}),
            "tokens_per_group": TOKENS,
        },
        "exact_checks": {
            "topology_edge_k_mismatches": edge_k_mismatches,
            "qsilent_score_mismatches": score_mismatches,
            "checked_qsilent_edges": checked_qsilent_edges,
        },
        "global": summarize(rows),
        "by_stage": {
            str(stage): summarize(stage_rows[stage]) for stage in sorted(stage_rows)
        },
        "rows": rows,
        "contracts": [
            "Q=0时raw16(Q,K)=32-popcount(K)，随后使用同一/16 RNE Q7。",
            "source-owned路径每个K source只生成一次6-bit popcount统计量；"
            "demand口径仅生成至少被一个silent query消费的source。",
            "score侧K读bit只比较valid edge：原路径32 bit/edge；候选silent edge"
            "读取6-bit统计量，non-silent edge仍读取32-bit K。",
            "K源流本身的32-bit输入是两边共同工作，不计入score侧读bit削减。",
        ],
        "limits": [
            "这是100-sample真实Q/K向量profile，不是RTL周期、面积、功耗或PPA。",
            "当前模型未计三行统计量buffer、五方向路由、mixed fast/residual join"
            "和反压，因此不能把工作削减直接写成加速。",
            "进入RTL前必须与现行q0_serial比较，并证明任一stage不出现显著退化。",
        ],
    }


def render(report: dict[str, Any]) -> str:
    global_row = report["global"]
    totals = global_row["totals"]
    reductions = global_row["reductions"]
    lines = [
        "# Local5 source-owned Q-silent 充分统计画像",
        "",
        f"- group/sample：{report['source']['groups']}/{report['source']['samples']}",
        f"- Q-silent exact edge：{report['exact_checks']['checked_qsilent_edges']}，"
        "score mismatch=0，topology K mismatch=0",
        "- 证据：[prof]；不是RTL周期或PPA",
        "",
        "| 指标 | 当前destination侧 | source-owned all-source | "
        "source-owned demand |",
        "|---|---:|---:|---:|",
        f"| popcount evaluation | {totals['baseline_qsilent_popcounts']} | "
        f"{totals['source_owned_all_source_popcounts']} | "
        f"{totals['source_owned_demand_popcounts']} |",
        f"| 相对削减 | - | {reductions['popcount_all_source']:.2%} | "
        f"{reductions['popcount_demand']:.2%} |",
        f"| score侧K读取bit | {totals['baseline_score_k_read_bits']} | "
        f"{totals['source_owned_score_k_read_bits']} | 同左 |",
        f"| score侧K读取bit削减 | - | {reductions['score_k_read_bits']:.2%} | 同左 |",
        f"| 6-bit统计量route bit | 0 | {totals['source_stat_route_bits']} | 同左 |",
        f"| 6-bit统计量写入bit | 0 | {totals['source_stat_write_bits']} | "
        "按需值另由实现决定 |",
        "",
        "## 分stage",
        "",
        "| stage | groups | silent destination | silent edge | "
        "all-source popcount削减 | score K-bit削减 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, row in report["by_stage"].items():
        stage_totals = row["totals"]
        stage_reductions = row["reductions"]
        lines.append(
            f"| {stage} | {row['groups']} | "
            f"{stage_totals['qsilent_destinations']} | "
            f"{stage_totals['qsilent_edges']} | "
            f"{stage_reductions['popcount_all_source']:.2%} | "
            f"{stage_reductions['score_k_read_bits']:.2%} |"
        )
    lines.extend(["", "## 合同", ""])
    lines.extend(f"- {item}" for item in report["contracts"])
    lines.extend(["", "## 边界", ""])
    lines.extend(f"- {item}" for item in report["limits"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=Path(
            "tb_qfit/vectors/"
            "local5_joint_ep29_score_projection_realw_sample100_population_v1_20260813"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/local5_source_owned_qsilent_profile_20260814"),
    )
    args = parser.parse_args()
    report = analyze(args.vector_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out / "report.md").write_text(render(report), encoding="utf-8")
    print(args.out / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
