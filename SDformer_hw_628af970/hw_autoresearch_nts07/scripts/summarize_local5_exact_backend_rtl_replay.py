#!/usr/bin/env python3
"""汇总 Local5 全 group Direct/GASR 五 bank 精确 RTL 回放。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


GROUP_RE = re.compile(r"^GROUP (?P<body>.+)$")
FIELD_RE = re.compile(r"(?P<key>[a-zA-Z0-9_]+)=(?P<value>-?\d+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_log(path: Path) -> list[dict[str, int]]:
    rows = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        match = GROUP_RE.match(line)
        if match:
            rows.append(
                {
                    item.group("key"): int(item.group("value"))
                    for item in FIELD_RE.finditer(match.group("body"))
                }
            )
    nonempty = [line for line in text.splitlines() if line.strip()]
    pass_lines = [
        line for line in nonempty if line.startswith("PASS post-G0 active projection")
    ]
    terminal_ok = nonempty[-1] == pass_lines[0] if pass_lines else False
    if (
        pass_lines
        and len(nonempty) >= 2
        and nonempty[-2] == pass_lines[0]
        and nonempty[-1].startswith("- ")
        and nonempty[-1].endswith("Verilog $finish")
    ):
        terminal_ok = True
    if len(pass_lines) != 1 or not terminal_ok:
        raise ValueError(f"RTL日志没有唯一末尾PASS: {path}")
    if any(token in text for token in ("%Error", "MISMATCH", "FAIL", "$fatal")):
        raise ValueError(f"RTL日志含失败标记: {path}")
    if [row["group"] for row in rows] != list(range(len(rows))):
        raise ValueError(f"RTL group顺序不连续: {path}")
    return rows


def statistics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p0": float(np.percentile(values, 0)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "sample_observed_max": float(values.max()),
    }


def weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0 or values.shape != weights.shape or np.any(weights <= 0):
        raise ValueError("weighted percentile输入非法")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    target = percentile / 100.0 * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, target, side="left")), values.size - 1)
    return float(values[order[index]])


def paired_cluster_bootstrap(
    direct: np.ndarray,
    gasr: np.ndarray,
    *,
    trials: int = 20_000,
    seed: int = 20260810,
) -> dict[str, float | int]:
    direct = np.asarray(direct, dtype=np.float64)
    gasr = np.asarray(gasr, dtype=np.float64)
    if direct.size < 2 or direct.shape != gasr.shape or np.any(gasr <= 0):
        raise ValueError("paired cluster bootstrap输入非法")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, direct.size, size=(trials, direct.size))
    ratios = direct[indices].sum(axis=1) / gasr[indices].sum(axis=1)
    return {
        "trials": trials,
        "seed": seed,
        "lower_95": float(np.percentile(ratios, 2.5)),
        "median": float(np.percentile(ratios, 50)),
        "upper_95": float(np.percentile(ratios, 97.5)),
    }


def build_report(manifest_path: Path, direct_path: Path, gasr_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "local5_active_projection_postg0_vectors_v1"
        or manifest.get("selection", {}).get("method")
        != "manifest_order_all_groups"
    ):
        raise ValueError("vector manifest不是all-groups精确回放合同")
    source_manifest_path = Path(str(manifest.get("source_manifest", "")))
    source_payload_path = Path(str(manifest.get("source_payload", "")))
    if (
        not source_manifest_path.is_file()
        or not source_payload_path.is_file()
        or sha256(source_manifest_path) != manifest.get("source_manifest_sha256")
        or sha256(source_payload_path) != manifest.get("source_payload_sha256")
    ):
        raise ValueError("vector manifest绑定的source trace文件或SHA失配")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if (
        source_manifest.get("schema") != "et3_ordered_term_trace_v2"
        or not source_manifest.get("qualification", {}).get("qualified")
    ):
        raise ValueError("source trace未通过post-G0 qualification")
    source_groups = source_manifest.get("groups") or []
    groups_per_block_sample = int(
        source_manifest.get("sampling", {}).get("groups_per_block_sample", 0)
    )
    if groups_per_block_sample <= 0:
        raise ValueError("source trace缺少正的groups_per_block_sample")
    metadata = manifest.get("selection", {}).get("rows") or []
    direct = parse_log(direct_path)
    gasr = parse_log(gasr_path)
    if not metadata or len(direct) != len(metadata) or len(gasr) != len(metadata):
        raise ValueError("metadata与两条RTL日志group数不一致")
    for label, log_rows, expected_mode in (
        ("Direct", direct, 0),
        ("GASR-reset", gasr, 1),
    ):
        if any(
            row.get("new1rw") != 1
            or row.get("mode") != expected_mode
            or row.get("latency") != 1
            for row in log_rows
        ):
            raise ValueError(f"{label}日志的1RW/mode/latency合同不匹配")

    stage_rows: dict[int, list[dict[str, int]]] = defaultdict(list)
    rows = []
    for index, (meta, drow, grow) in enumerate(zip(metadata, direct, gasr, strict=True)):
        if (
            drow["group"] != index
            or grow["group"] != index
            or int(meta["vector_group_index"]) != index
        ):
            raise ValueError(f"group身份错位: {index}")
        source_index = int(meta.get("input_group_index", -1))
        if not 0 <= source_index < len(source_groups):
            raise ValueError(f"group {index}的source index越界")
        source_row = source_groups[source_index]
        identity_fields = (
            "tag",
            "sample",
            "stage",
            "block",
            "window",
            "head",
            "flat_group",
            "batch_windows",
            "heads",
            "module",
            "selection",
        )
        if any(meta.get(field) != source_row.get(field) for field in identity_fields):
            raise ValueError(f"group {index}与source manifest身份不一致")
        for field, meta_field in (
            ("active", "active_sources"),
            ("terms", "terms"),
            ("updates", "updates"),
        ):
            if drow[field] != grow[field] or drow[field] != int(meta[meta_field]):
                raise ValueError(f"group {index} 的 {field} 跨路径不一致")
        row = {
            "group": index,
            "sample": int(meta["sample"]),
            "stage": int(meta["stage"]),
            "block": int(meta["block"]),
            "window": int(meta["window"]),
            "head": int(meta["head"]),
            "population_weight": float(meta["batch_windows"])
            * float(meta["heads"])
            / groups_per_block_sample,
            "active": drow["active"],
            "terms": drow["terms"],
            "updates": drow["updates"],
            "direct_cycles": drow["cycles"],
            "gasr_cycles": grow["cycles"],
            "direct_stall": drow["term_stall"],
            "gasr_stall": grow["term_stall"],
            "direct_sram_reads": drow["sram_reads"],
            "direct_sram_writes": drow["sram_writes"],
            "gasr_sram_reads": grow["sram_reads"],
            "gasr_sram_writes": grow["sram_writes"],
            "direct_sram_transactions": drow["sram_reads"] + drow["sram_writes"],
            "gasr_sram_transactions": grow["sram_reads"] + grow["sram_writes"],
        }
        rows.append(row)
        stage_rows[row["stage"]].append(row)

    direct_cycles = np.asarray([row["direct_cycles"] for row in rows], dtype=np.float64)
    gasr_cycles = np.asarray([row["gasr_cycles"] for row in rows], dtype=np.float64)
    speedups = direct_cycles / gasr_cycles
    population_weights = np.asarray(
        [row["population_weight"] for row in rows], dtype=np.float64
    )
    direct_sum = int(direct_cycles.sum())
    gasr_sum = int(gasr_cycles.sum())
    oracle_cycles = np.minimum(direct_cycles, gasr_cycles)
    oracle_sum = int(oracle_cycles.sum())
    direct_reads = sum(row["direct_sram_reads"] for row in rows)
    direct_writes = sum(row["direct_sram_writes"] for row in rows)
    gasr_reads = sum(row["gasr_sram_reads"] for row in rows)
    gasr_writes = sum(row["gasr_sram_writes"] for row in rows)
    direct_tx = sum(row["direct_sram_transactions"] for row in rows)
    gasr_tx = sum(row["gasr_sram_transactions"] for row in rows)
    population_direct = float(np.dot(population_weights, direct_cycles))
    population_gasr = float(np.dot(population_weights, gasr_cycles))
    population_oracle = float(np.dot(population_weights, oracle_cycles))
    per_stage = []
    for stage in sorted(stage_rows):
        subset = stage_rows[stage]
        stage_direct_cycles = np.asarray(
            [row["direct_cycles"] for row in subset], dtype=np.float64
        )
        stage_gasr_cycles = np.asarray(
            [row["gasr_cycles"] for row in subset], dtype=np.float64
        )
        dsum = sum(row["direct_cycles"] for row in subset)
        gsum = sum(row["gasr_cycles"] for row in subset)
        oracle_stage_sum = sum(
            min(row["direct_cycles"], row["gasr_cycles"]) for row in subset
        )
        ratios = np.asarray(
            [row["direct_cycles"] / row["gasr_cycles"] for row in subset]
        )
        stage_weights = np.asarray(
            [row["population_weight"] for row in subset], dtype=np.float64
        )
        per_stage.append(
            {
                "stage": stage,
                "groups": len(subset),
                "direct_cycles": dsum,
                "gasr_cycles": gsum,
                "aggregate_speedup": dsum / gsum,
                "post_hoc_perfect_mode_oracle_speedup": dsum / oracle_stage_sum,
                "group_speedup": statistics(ratios),
                "direct_cycles_distribution": statistics(stage_direct_cycles),
                "gasr_cycles_distribution": statistics(stage_gasr_cycles),
                "p95_non_regression": float(np.percentile(stage_gasr_cycles, 95))
                <= float(np.percentile(stage_direct_cycles, 95)),
                "population_weighted_speedup": float(
                    np.dot(stage_weights, stage_direct_cycles)
                    / np.dot(stage_weights, stage_gasr_cycles)
                ),
                "population_weighted_direct_p95": weighted_percentile(
                    stage_direct_cycles, stage_weights, 95
                ),
                "population_weighted_gasr_p95": weighted_percentile(
                    stage_gasr_cycles, stage_weights, 95
                ),
            }
        )
    overall_p95_non_regression = float(np.percentile(gasr_cycles, 95)) <= float(
        np.percentile(direct_cycles, 95)
    )
    stage_p95_non_regression = all(row["p95_non_regression"] for row in per_stage)
    p95_non_regression = overall_p95_non_regression and stage_p95_non_regression
    speedup = direct_sum / gasr_sum
    sample_rows: dict[int, list[dict[str, int | float]]] = defaultdict(list)
    for row in rows:
        sample_rows[int(row["sample"])].append(row)
    sample_direct = []
    sample_gasr = []
    for sample in sorted(sample_rows):
        subset = sample_rows[sample]
        sample_direct.append(
            sum(float(row["population_weight"]) * int(row["direct_cycles"]) for row in subset)
        )
        sample_gasr.append(
            sum(float(row["population_weight"]) * int(row["gasr_cycles"]) for row in subset)
        )
    sample_direct_array = np.asarray(sample_direct, dtype=np.float64)
    sample_gasr_array = np.asarray(sample_gasr, dtype=np.float64)
    return {
        "schema": "local5_exact_backend_rtl_replay_v1",
        "status": "EXACT_BACKEND_RTL_DECISION_COMPLETE",
        "evidence": "[rtl]，projection_start到flush/done周期；done后Acc32 readback仅作miter；非ASIC PPA",
        "inputs": {
            "vector_manifest": str(manifest_path.resolve()),
            "vector_manifest_sha256": sha256(manifest_path),
            "source_manifest": manifest.get("source_manifest"),
            "source_manifest_sha256": manifest.get("source_manifest_sha256"),
            "source_payload": manifest.get("source_payload"),
            "source_payload_sha256": manifest.get("source_payload_sha256"),
            "source_sampling": source_manifest.get("sampling"),
            "source_checkpoint": source_manifest.get("checkpoint"),
            "source_checkpoint_sha256": source_manifest.get("checkpoint_sha256"),
            "direct_log": str(direct_path.resolve()),
            "direct_log_sha256": sha256(direct_path),
            "gasr_log": str(gasr_path.resolve()),
            "gasr_log_sha256": sha256(gasr_path),
            "groups": len(rows),
        },
        "aggregate": {
            "direct_cycles": direct_sum,
            "gasr_cycles": gasr_sum,
            "gasr_speedup": speedup,
            "gasr_cycle_reduction": 1.0 - gasr_sum / direct_sum,
            "direct_term_stalls": sum(row["direct_stall"] for row in rows),
            "gasr_term_stalls": sum(row["gasr_stall"] for row in rows),
            "direct_sram_transactions": direct_tx,
            "gasr_sram_transactions": gasr_tx,
            "direct_sram_reads": direct_reads,
            "direct_sram_writes": direct_writes,
            "gasr_sram_reads": gasr_reads,
            "gasr_sram_writes": gasr_writes,
            "sram_transaction_reduction": 1.0 - gasr_tx / direct_tx,
            "post_hoc_perfect_mode_oracle": {
                "cycles": oracle_sum,
                "speedup_over_direct": direct_sum / oracle_sum,
                "maximum_saved_cycles": direct_sum - oracle_sum,
                "mean_saved_cycles_per_group": (direct_sum - oracle_sum) / len(rows),
                "interpretation": "零成本、预知每组更快模式的事后上界；不是真实候选",
            },
            "win_equal_loss": {
                "win": int(np.sum(speedups > 1.0)),
                "equal": int(np.sum(speedups == 1.0)),
                "loss": int(np.sum(speedups < 1.0)),
            },
            "group_speedup": statistics(speedups),
            "direct_cycles_distribution": statistics(direct_cycles),
            "gasr_cycles_distribution": statistics(gasr_cycles),
            "population_weighted_approximation": {
                "weight": "(batch_windows * heads) / groups_per_block_sample；当前每block/sample四个rotating group",
                "direct_weighted_cycles": population_direct,
                "gasr_weighted_cycles": population_gasr,
                "gasr_speedup": population_direct / population_gasr,
                "direct_weighted_p95": weighted_percentile(
                    direct_cycles, population_weights, 95
                ),
                "gasr_weighted_p95": weighted_percentile(
                    gasr_cycles, population_weights, 95
                ),
                "perfect_mode_oracle_speedup": population_direct / population_oracle,
                "sample_ratio_distribution": statistics(
                    sample_direct_array / sample_gasr_array
                ),
                "paired_sample_cluster_bootstrap": paired_cluster_bootstrap(
                    sample_direct_array, sample_gasr_array
                ),
                "inference_boundary": "条件于当前确定性rotating window/head selection，只重采样100个sample cluster",
            },
        },
        "per_stage": per_stage,
        "numeric_miter": {
            "paired_acc32_coordinates": len(rows)
            * int(manifest["shape"]["sources"])
            * int(manifest["shape"]["out_dim"]),
            "comparisons_per_mode": len(rows)
            * int(manifest["shape"]["sources"])
            * int(manifest["shape"]["out_dim"]),
            "total_comparisons_across_two_modes": 2
            * len(rows)
            * int(manifest["shape"]["sources"])
            * int(manifest["shape"]["out_dim"]),
            "mismatches": 0,
            "cycle_includes_readback": False,
        },
        "cycle_scope": "projection_start through relation/frontier/builder/five-bank execution and flush/done; excludes post-done Acc32 readback",
        "transaction_scope": "accumulator-bank execution only; excludes relation SRAM, post-done readback and valid-bitmap clear",
        "local_pre_result_gate": {
            "evidence_boundary": "运行前已写入本地脚本；不是外部时间戳预注册",
            "aggregate_speedup_min": 1.20,
            "overall_and_each_stage_p95_non_regression_required": True,
            "aggregate_speedup_pass": speedup >= 1.20,
            "overall_p95_non_regression_pass": overall_p95_non_regression,
            "each_stage_p95_non_regression_pass": stage_p95_non_regression,
            "tail_pass": p95_non_regression,
            "decision": (
                "PROMOTE_GASR_RESET_PATH"
                if speedup >= 1.20 and p95_non_regression
                else "REJECT_GASR_RESET_PATH"
            ),
        },
        "limits": [
            "GASR每group由run_start清空，不是GASR2C-P跨head preserve。",
            "当前回放逐group独立，不构成同窗全head或full encoder周期。",
            "4800组来自100 sample、12 block、每block/sample抽4个ordered group；不是full-workload totals。",
            "all-head coverage是100-sample cohort上的union coverage，不是每个sample/block穷举全部head。",
            "paired-sample bootstrap条件于当前确定性rotating selection，未覆盖window/head选择不确定性。",
            "周期从projection_start计到flush/done，不含done后的Acc32 readback，也不含外部共同serializer。",
            "向量使用真实descriptor/gate/K workload与synthetic weight；控制周期不依赖weight数值，数值miter覆盖该synthetic weight合同。",
            "SRAM事务只计accumulator bank执行期，不含relation SRAM、done后readback或valid-bitmap清零；不是SAIF、能量或功耗。",
            "1.20x与逐stage p95门槛在本轮结果产生前写入本地脚本，但没有外部不可变时间戳。",
            "Direct/GASR逐group最优选择是看过结果后的零成本oracle，只用于判断自适应双模的理论上限。",
        ],
    }


def render_markdown(report: dict[str, object]) -> str:
    agg = report["aggregate"]
    gate = report["local_pre_result_gate"]
    lines = [
        "# Local5 全 Group 五 Bank 精确 RTL 回放",
        "",
        "## 结论",
        "",
        f"本轮回放 `{report['inputs']['groups']}` 个 group，GASR-reset 相对 Direct 为 "
        f"`{agg['gasr_speedup']:.4f}x`，预提交裁决为 `{gate['decision']}`。",
        "",
        "## 总表",
        "",
        "| 指标 | Direct | GASR-reset | 差异 |",
        "|---|---:|---:|---:|",
        f"| RTL周期 | {agg['direct_cycles']:,} | {agg['gasr_cycles']:,} | {agg['gasr_speedup']:.4f}x |",
        f"| term stall | {agg['direct_term_stalls']:,} | {agg['gasr_term_stalls']:,} | - |",
        f"| 1RW SRAM事务 | {agg['direct_sram_transactions']:,} | {agg['gasr_sram_transactions']:,} | 降低 {agg['sram_transaction_reduction']:.2%} |",
        "",
        f"逐group win/equal/loss=`{agg['win_equal_loss']['win']}/{agg['win_equal_loss']['equal']}/{agg['win_equal_loss']['loss']}`。",
        f"事后零成本逐group最优模式上界仅为 `{agg['post_hoc_perfect_mode_oracle']['speedup_over_direct']:.4f}x`，"
        "不构成可实现结果。",
        f"按 `batch_windows*heads` 加权的 population 近似为 "
        f"`{agg['population_weighted_approximation']['gasr_speedup']:.4f}x`，"
        f"完美双模上界为 `{agg['population_weighted_approximation']['perfect_mode_oracle_speedup']:.4f}x`。",
        "",
        "## 分 Stage",
        "",
        "| Stage | group | Direct周期 | GASR周期 | 聚合加速 | Direct p95 | GASR p95 | p95不回退 |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report["per_stage"]:
        lines.append(
            f"| {row['stage']} | {row['groups']} | {row['direct_cycles']:,} | "
            f"{row['gasr_cycles']:,} | {row['aggregate_speedup']:.4f}x | "
            f"{row['direct_cycles_distribution']['p95']:.1f} | "
            f"{row['gasr_cycles_distribution']['p95']:.1f} | "
            f"{'是' if row['p95_non_regression'] else '否'} |"
        )
    lines += ["", "## 证据边界", ""]
    lines.extend(f"- {item}" for item in report["limits"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--direct-log", type=Path, required=True)
    parser.add_argument("--gasr-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.manifest, args.direct_log, args.gasr_log)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report["local_pre_result_gate"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
