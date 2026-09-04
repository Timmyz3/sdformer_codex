#!/usr/bin/env python3
"""Extract architecture-facing statistics from the existing H67/H68 profile100."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
N_TOKENS = 162
CONTROL_CYCLES = 3
CASES = {
    "H67": {
        "class_cycles": 2.0,
        "profile": REPO
        / "neuron_experiments/H9_bipolar_self_attention/results"
        / "h67_ep19_true_ttb_profile100_20260712/nts11_hardware_p0_profile.json",
    },
    "H68": {
        "class_cycles": 1.0,
        "profile": REPO
        / "neuron_experiments/H9_bipolar_self_attention/results"
        / "h68_ep19_true_ttb_profile100_20260713/nts11_hardware_p0_profile.json",
    },
}
RATIO_COUNTERS = {
    "pair_empty": ("ttb_tok1_empty", "ttb_tok1_total"),
    "pair_kzero": ("ttb_tok1_kzero", "ttb_tok1_total"),
    "pair_motion_zero": ("ttb_tok1_motion_zero", "ttb_tok1_total"),
    "delta_zero": ("delta_zero_update_token_heads", "delta_token_heads"),
}


def ratio(record: dict[str, Any], numerator: str, denominator: str) -> float:
    total = int(record[denominator])
    return int(record[numerator]) / total if total else 0.0


def quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {key: 0.0 for key in ("mean", "p10", "p50", "p90", "p99", "max", "cv")}
    mean = float(array.mean())
    return {
        "mean": mean,
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
        "cv": float(array.std() / mean) if mean else 0.0,
    }


def weighted_mean(records: list[dict[str, Any]], field: str) -> float:
    weights = np.asarray([row_count(record) for record in records], dtype=np.float64)
    values = np.asarray([float(record[field]) for record in records], dtype=np.float64)
    return float(np.average(values, weights=weights)) if weights.sum() else 0.0


def summed_ratio(records: list[dict[str, Any]], numerator: str, denominator: str) -> float:
    total = sum(int(record[denominator]) for record in records)
    return sum(int(record[numerator]) for record in records) / total if total else 0.0


def row_count(record: dict[str, Any]) -> int:
    return int(record["batch_windows"]) * int(record["num_heads"])


def row_cycles(record: dict[str, Any], class_cycles: float) -> float:
    active = float(record["zaf_active_entries_mean"])
    occupied = float(record["zaf_fold_classes_mean"])
    return N_TOKENS + max(active, 1.0) + class_cycles * occupied + active + CONTROL_CYCLES


def backend_cycles(record: dict[str, Any], class_cycles: float) -> float:
    active = float(record["zaf_active_entries_mean"])
    occupied = float(record["zaf_fold_classes_mean"])
    return max(active, 1.0) + class_cycles * occupied + active + CONTROL_CYCLES


def flowshop_cycles(front: list[float], backend: list[float], contexts: int) -> float:
    front_available = 0.0
    backend_available = 0.0
    releases: list[float] = []
    for index, (front_work, backend_work) in enumerate(zip(front, backend, strict=True)):
        context_available = releases[index - contexts] if index >= contexts else 0.0
        front_done = max(front_available, context_available) + front_work
        backend_done = max(backend_available, front_done) + backend_work
        front_available = front_done
        backend_available = backend_done
        releases.append(backend_done)
    return backend_available


def three_stage_flowshop_cycles(
    fetch: list[float],
    commit: list[float],
    backend: list[float],
    contexts: int,
) -> float:
    """Conservative row-granular fetch/commit/backend replay."""

    fetch_available = 0.0
    commit_available = 0.0
    backend_available = 0.0
    releases: list[float] = []
    for index, (fetch_work, commit_work, backend_work) in enumerate(
        zip(fetch, commit, backend, strict=True)
    ):
        context_available = releases[index - contexts] if index >= contexts else 0.0
        fetch_done = max(fetch_available, context_available) + fetch_work
        commit_done = max(commit_available, fetch_done) + commit_work
        backend_done = max(backend_available, commit_done) + backend_work
        fetch_available = fetch_done
        commit_available = commit_done
        backend_available = backend_done
        releases.append(backend_done)
    return backend_available


def pair_category_row_means(record: dict[str, Any]) -> dict[str, float]:
    """Recover exact pair K-state means available in the legacy profile."""

    rows = row_count(record)
    pairs = float(record["ttb_tok1_total"]) / rows
    both_kzero = float(record["ttb_tok1_kzero"]) / rows
    zero_tokens = float(record["zaf_kzero_token_ratio"]) * 2.0 * pairs
    one_kzero = zero_tokens - 2.0 * both_kzero
    tolerance = 1e-4
    if one_kzero < -tolerance:
        raise ValueError("legacy counters imply a negative one-K-zero pair count")
    one_kzero = min(max(one_kzero, 0.0), pairs - both_kzero)
    both_active = pairs - both_kzero - one_kzero
    if both_active < -tolerance:
        raise ValueError("legacy counters imply a negative both-active pair count")
    return {
        "pairs": pairs,
        "both_kzero": both_kzero,
        "one_kzero": one_kzero,
        "both_active": max(both_active, 0.0),
    }


def commit_cycles(record: dict[str, Any], mode: str) -> float:
    pair = pair_category_row_means(record)
    pairs = pair["pairs"]
    both_kzero = pair["both_kzero"]
    one_kzero = pair["one_kzero"]
    both_active = pair["both_active"]
    if mode == "dual_write_ideal":
        return pairs
    if mode == "split_1w_no_merge":
        return 2.0 * both_kzero + one_kzero + 2.0 * both_active
    if mode == "split_1w_perfect_pccc":
        return both_kzero + one_kzero + 2.0 * both_active
    if mode == "unified_1w_no_merge":
        return 2.0 * pairs
    if mode == "unified_1w_perfect_pccc":
        return 2.0 * pairs - both_kzero
    raise ValueError(f"unsupported commit mode: {mode}")


COMMIT_MODES = (
    "dual_write_ideal",
    "split_1w_no_merge",
    "split_1w_perfect_pccc",
    "unified_1w_no_merge",
    "unified_1w_perfect_pccc",
)


def sample_port_aware_dse(group: list[dict[str, Any]], class_cycles: float) -> dict[str, float]:
    backend: list[float] = []
    commits = {mode: [] for mode in COMMIT_MODES}
    row_total = 0
    category_totals = {key: 0.0 for key in ("pairs", "both_kzero", "one_kzero", "both_active")}
    for record in group:
        rows = row_count(record)
        row_total += rows
        backend.extend([backend_cycles(record, class_cycles)] * rows)
        category = pair_category_row_means(record)
        for key in category_totals:
            category_totals[key] += category[key] * rows
        for mode in COMMIT_MODES:
            commits[mode].extend([commit_cycles(record, mode)] * rows)

    result: dict[str, float] = {
        f"pair_{key}_per_row": value / row_total for key, value in category_totals.items()
    }
    for fetch_width, fetch_cycles in ((64, 162.0), (128, 81.0)):
        fetch = [fetch_cycles] * row_total
        for mode in COMMIT_MODES:
            for contexts in (1, 2, 4):
                result[f"fetch{fetch_width}_{mode}_contexts{contexts}"] = three_stage_flowshop_cycles(
                    fetch, commits[mode], backend, contexts
                )
    return result


def sample_pipeline_dse(group: list[dict[str, Any]], class_cycles: float) -> dict[str, float]:
    front: list[float] = []
    backend: list[float] = []
    for record in group:
        rows = row_count(record)
        front.extend([81.0] * rows)
        backend.extend([backend_cycles(record, class_cycles)] * rows)
    current = sum(N_TOKENS + value for value in backend)
    result = {
        "current_serial": current,
        "pair_front_work": sum(front),
        "backend_work": sum(backend),
        "infinite_context_lower_bound": max(sum(front), sum(backend)),
    }
    for contexts in (1, 2, 4, 8):
        result[f"contexts_{contexts}"] = flowshop_cycles(front, backend, contexts)
    return result


def parse_block(name: str) -> int:
    match = re.search(r"\.B(\d+)\.", name)
    if match is None:
        raise ValueError(f"cannot parse block from {name}")
    return int(match.group(1))


def summarize_group(records: list[dict[str, Any]], class_cycles: float) -> dict[str, Any]:
    first = records[0]
    result: dict[str, Any] = {
        "records": len(records),
        "stage": int(first["stage"]),
        "block": parse_block(str(first["name"])),
        "rows_per_frame": row_count(first),
        "active_entries": quantiles(float(record["zaf_active_entries_mean"]) for record in records),
        "fold_classes": quantiles(float(record["zaf_fold_classes_mean"]) for record in records),
        "row_cycles": quantiles(row_cycles(record, class_cycles) for record in records),
        "q_active_density": quantiles(float(record["q_active_density"]) for record in records),
        "k_active_density": quantiles(float(record["k_active_density"]) for record in records),
    }
    for label, (numerator, denominator) in RATIO_COUNTERS.items():
        result[label] = quantiles(ratio(record, numerator, denominator) for record in records)
        result[label]["global"] = summed_ratio(records, numerator, denominator)
    return result


def summarize_stage(stage: int, records: list[dict[str, Any]], class_cycles: float) -> dict[str, Any]:
    block_names = sorted({str(record["name"]) for record in records})
    result: dict[str, Any] = {
        "stage": stage,
        "blocks": len(block_names),
        "rows_per_frame": sum(row_count(record) for record in records[: len(block_names)]),
        "active_entries_weighted_mean": weighted_mean(records, "zaf_active_entries_mean"),
        "fold_classes_weighted_mean": weighted_mean(records, "zaf_fold_classes_mean"),
        "row_cycles": quantiles(row_cycles(record, class_cycles) for record in records),
    }
    for label, (numerator, denominator) in RATIO_COUNTERS.items():
        result[label] = summed_ratio(records, numerator, denominator)
    return result


def validate_sample_order(records: list[dict[str, Any]], samples: int) -> list[list[dict[str, Any]]]:
    names = sorted({str(record["name"]) for record in records})
    if len(records) != samples * len(names):
        raise ValueError("record count does not match samples multiplied by attention blocks")
    groups = [records[index : index + len(names)] for index in range(0, len(records), len(names))]
    expected = [str(record["name"]) for record in groups[0]]
    if len(set(expected)) != len(names):
        raise ValueError("first sample does not contain one record per attention block")
    for index, group in enumerate(groups):
        if [str(record["name"]) for record in group] != expected:
            raise ValueError(f"attention record order changed at sample {index}")
    return groups


def sample_summary(group: list[dict[str, Any]], class_cycles: float) -> dict[str, float]:
    cycles = sum(row_count(record) * row_cycles(record, class_cycles) for record in group)
    total_pairs = sum(int(record["ttb_tok1_total"]) for record in group)
    return {
        "attention_cycles_proxy": cycles,
        "pair_empty": sum(int(record["ttb_tok1_empty"]) for record in group) / total_pairs,
        "pair_kzero": sum(int(record["ttb_tok1_kzero"]) for record in group) / total_pairs,
        "pair_motion_zero": sum(int(record["ttb_tok1_motion_zero"]) for record in group) / total_pairs,
        "delta_zero": (
            sum(int(record["delta_zero_update_token_heads"]) for record in group)
            / sum(int(record["delta_token_heads"]) for record in group)
        ),
    }


def bundle_summary(records: list[dict[str, Any]], bundle: int) -> dict[str, Any]:
    prefix = f"ttb_tok{bundle}"
    total = sum(int(record[f"{prefix}_total"]) for record in records)
    empty = sum(int(record[f"{prefix}_empty"]) for record in records)
    result: dict[str, Any] = {
        "bundle_tokens": bundle,
        "bundles": total,
        "empty_ratio": empty / total if total else 0.0,
    }
    for threshold in (2, 4, 8, 16, 32):
        active = sum(int(record[f"{prefix}_active_le{threshold}"]) for record in records)
        result[f"active_le{threshold}_ratio"] = active / total if total else 0.0
    active_lanes = sum(int(record[f"{prefix}_active_lanes"]) for record in records)
    result["mean_active_lanes_per_nonempty"] = active_lanes / (total - empty) if total > empty else 0.0
    return result


def delta_histogram(records: list[dict[str, Any]]) -> dict[str, float]:
    fields = (
        "delta_update_count_0", "delta_update_count_1", "delta_update_count_2",
        "delta_update_count_3_4", "delta_update_count_5_8", "delta_update_count_9_16",
        "delta_update_count_17_plus",
    )
    counts = {field: sum(int(record[field]) for record in records) for field in fields}
    total = sum(counts.values())
    return {field.removeprefix("delta_update_count_"): value / total for field, value in counts.items()}


def analyze_case(name: str, case: dict[str, Any]) -> dict[str, Any]:
    profile = Path(case["profile"])
    data = json.loads(profile.read_text(encoding="utf-8"))
    records = data["summary"]["h60_records"]
    samples = int(data["samples"])
    class_cycles = float(case["class_cycles"])
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_stage: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_name[str(record["name"])].append(record)
        by_stage[int(record["stage"])].append(record)
    sample_groups = validate_sample_order(records, samples)
    sample_rows = [sample_summary(group, class_cycles) for group in sample_groups]
    sample_pipelines = [sample_pipeline_dse(group, class_cycles) for group in sample_groups]
    sample_port_dse = [sample_port_aware_dse(group, class_cycles) for group in sample_groups]
    whole: dict[str, Any] = {
        "samples": samples,
        "attention_blocks": len(by_name),
        "rows_per_frame": sum(row_count(records_for_block[0]) for records_for_block in by_name.values()),
        "active_entries_weighted_mean": weighted_mean(records, "zaf_active_entries_mean"),
        "fold_classes_weighted_mean": weighted_mean(records, "zaf_fold_classes_mean"),
        "sample_attention_cycles_proxy": quantiles(row["attention_cycles_proxy"] for row in sample_rows),
    }
    pipeline_dse: dict[str, Any] = {
        key: quantiles(row[key] for row in sample_pipelines)
        for key in sample_pipelines[0]
    }
    current_mean = pipeline_dse["current_serial"]["mean"]
    pair_one_mean = pipeline_dse["contexts_1"]["mean"]
    for contexts in (1, 2, 4, 8):
        value = pipeline_dse[f"contexts_{contexts}"]["mean"]
        pipeline_dse[f"reduction_vs_current_{contexts}"] = 1.0 - value / current_mean
        pipeline_dse[f"reduction_vs_pair1_{contexts}"] = 1.0 - value / pair_one_mean
    whole["pair_pipeline_dse"] = pipeline_dse
    port_dse: dict[str, Any] = {
        key: quantiles(row[key] for row in sample_port_dse)
        for key in sample_port_dse[0]
    }
    for key, value in tuple(port_dse.items()):
        if key.startswith("fetch"):
            port_dse[f"reduction_vs_current_{key}"] = 1.0 - value["mean"] / current_mean
    whole["port_aware_pipeline_dse"] = port_dse
    for label, (numerator, denominator) in RATIO_COUNTERS.items():
        whole[label] = summed_ratio(records, numerator, denominator)
        whole[f"sample_{label}"] = quantiles(row[label] for row in sample_rows)
    blocks = [summarize_group(items, class_cycles) for _, items in sorted(by_name.items())]
    stages = [summarize_stage(stage, items, class_cycles) for stage, items in sorted(by_stage.items())]
    return {
        "model": name,
        "profile": str(profile.relative_to(REPO)),
        "scope": {
            "supported": "100 samples, 12 attention blocks, aggregate counters per block invocation",
            "not_supported": "ordered pair/row burst, pair union-index traffic, input-flow correlation, SRAM conflicts",
            "cycle_proxy": "existing 162-token row FSM plus occupied-class scan; excludes projection, ATLIF, SRAM stalls and decoder",
        },
        "whole": whole,
        "stages": stages,
        "blocks": blocks,
        "bundles": [bundle_summary(records, bundle) for bundle in (1, 2, 4, 8)],
        "delta_histogram": delta_histogram(records),
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render(results: list[dict[str, Any]]) -> str:
    lines = [
        "# H67/H68 百样本架构特征统计",
        "",
        "## 结论先行",
        "",
        "1. H67/H68 的全局稀疏分布接近，适合使用同一套可配置执行底座，不支持为两者各做一套物理核。",
        "2. block 间差异远大于 H67/H68 之间差异。固定按 stage 或全局阈值路由会浪费机会，调度粒度至少应到 `stage/block`，后续有序 trace 再判断是否下沉到 row。",
        "3. `T=2` 时间对全空和 Delta=0 比例高，优先级应是时间对联合取数、精确复用和多上下文隐藏可变后端，而不是先冻结异构双核。",
        "4. 本报告没有逐 pair 顺序，不能决定 FIFO 深度、蝶形压紧网络或双路径是否有净 PPA 收益；这些候选仍处于待证伪状态。",
        "",
        "## 全网统计",
        "",
        "| 模型 | 样本 | attention行/帧 | pair全空 | K-zero | motion-zero | Delta=0 | active项/行 | fold类/行 | 周期代理p50 | p99/p50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        whole = result["whole"]
        cycles = whole["sample_attention_cycles_proxy"]
        lines.append(
            f"| {result['model']} | {whole['samples']} | {whole['rows_per_frame']} | "
            f"{pct(whole['pair_empty'])} | {pct(whole['pair_kzero'])} | "
            f"{pct(whole['pair_motion_zero'])} | {pct(whole['delta_zero'])} | "
            f"{whole['active_entries_weighted_mean']:.2f} | {whole['fold_classes_weighted_mean']:.2f} | "
            f"{cycles['p50']:.0f} | {cycles['p99'] / cycles['p50']:.3f} |"
        )
    lines += [
        "",
        "`周期代理` 复用当前 row FSM 的 162-token load/scan/class/emit 口径，只用于比较样本和 block 服务时间，不是整网 FPS。",
        "",
        "## 81-pair 前端与多 context 粗粒度 DSE",
        "",
        "| 模型 | context | 周期均值/帧 | 相对当前162-token | 相对pair单context | p99/p50 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        dse = result["whole"]["pair_pipeline_dse"]
        for contexts in (1, 2, 4, 8):
            cycles = dse[f"contexts_{contexts}"]
            lines.append(
                f"| {result['model']} | {contexts} | {cycles['mean']:.0f} | "
                f"{pct(dse[f'reduction_vs_current_{contexts}'])} | "
                f"{pct(dse[f'reduction_vs_pair1_{contexts}'])} | "
                f"{cycles['p99'] / cycles['p50']:.3f} |"
            )
    lines += [
        "",
        "该模型把每个 block 调用内的 active/fold 均值复制到其全部 row，只保留样本和 block 顺序，未包含逐 row burst、bank conflict、PCCC stall 或 SRAM 延迟。因此它是 context 数的预筛选，不是 RTL cycle result。",
        "",
        "## 供数与双提交端口感知 DSE",
        "",
        "旧 profile 可以精确恢复双 K-zero、单 K-zero 和双 active 三类 pair，但没有双 K-zero 同 class 比例。因此 PCCC 只报告“完全不合并”和“全部同类可合并”上下界。模型按行串联 `fetch -> commit -> SCS`，允许不同行在多个 context 中流水重叠；它比上一节保守，但仍未包含逐 pair 流水、SRAM 延迟和 ordered burst。",
        "",
        "| 模型 | pair类别 | 每行数量 | 占比 |",
        "|---|---|---:|---:|",
    ]
    category_labels = {
        "both_kzero": "双 K-zero",
        "one_kzero": "单 K-zero",
        "both_active": "双 active",
    }
    for result in results:
        dse = result["whole"]["port_aware_pipeline_dse"]
        pairs = dse["pair_pairs_per_row"]["mean"]
        for key, label in category_labels.items():
            value = dse[f"pair_{key}_per_row"]["mean"]
            lines.append(f"| {result['model']} | {label} | {value:.2f} | {pct(value / pairs)} |")
    lines += [
        "",
        "| 模型 | 供数 | commit结构 | context | 周期均值/帧 | 相对当前162-token |",
        "|---|---:|---|---:|---:|---:|",
    ]
    scenario_labels = {
        "dual_write_ideal": "active/hist 各双写口理想下界",
        "split_1w_no_merge": "active/hist 分 bank 单写口、无合并",
        "split_1w_perfect_pccc": "active/hist 分 bank 单写口、PCCC 全合并上界",
        "unified_1w_no_merge": "统一单写口、无合并",
        "unified_1w_perfect_pccc": "统一单写口、PCCC 全合并上界",
    }
    selected = (
        (128, "dual_write_ideal", 2),
        (128, "split_1w_no_merge", 2),
        (128, "split_1w_perfect_pccc", 2),
        (128, "unified_1w_no_merge", 2),
        (128, "unified_1w_perfect_pccc", 2),
        (64, "split_1w_perfect_pccc", 2),
        (128, "split_1w_no_merge", 4),
        (128, "split_1w_perfect_pccc", 4),
    )
    for result in results:
        dse = result["whole"]["port_aware_pipeline_dse"]
        for width, mode, contexts in selected:
            key = f"fetch{width}_{mode}_contexts{contexts}"
            lines.append(
                f"| {result['model']} | {width} bit/拍 | {scenario_labels[mode]} | {contexts} | "
                f"{dse[key]['mean']:.0f} | {pct(dse[f'reduction_vs_current_{key}'])} |"
            )
    lines += [
        "",
        "真实 PCCC 必定位于无合并和全合并两条边界之间。只有 ordered profile 返回双 K-zero 同类率、同拍 collision 和有限队列 stall 后，才能选择端口数并把收益写成实测结论。",
        "",
        "## 分阶段统计",
        "",
        "| 模型 | stage | blocks | 行/帧 | pair全空 | K-zero | Delta=0 | active项/行 | fold类/行 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for stage in result["stages"]:
            lines.append(
                f"| {result['model']} | S{stage['stage']} | {stage['blocks']} | {stage['rows_per_frame']} | "
                f"{pct(stage['pair_empty'])} | {pct(stage['pair_kzero'])} | {pct(stage['delta_zero'])} | "
                f"{stage['active_entries_weighted_mean']:.2f} | {stage['fold_classes_weighted_mean']:.2f} |"
            )
    lines += [
        "",
        "## 分 block 统计",
        "",
        "| 模型 | block | 行/帧 | pair全空均值 | p10-p90 | K-zero | Delta=0 | active项均值 | p10-p90 | fold类均值 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for block in sorted(result["blocks"], key=lambda item: (item["stage"], item["block"])):
            empty = block["pair_empty"]
            active = block["active_entries"]
            lines.append(
                f"| {result['model']} | S{block['stage']}B{block['block']} | {block['rows_per_frame']} | "
                f"{pct(empty['global'])} | {pct(empty['p10'])}-{pct(empty['p90'])} | "
                f"{pct(block['pair_kzero']['global'])} | {pct(block['delta_zero']['global'])} | "
                f"{active['mean']:.2f} | {active['p10']:.2f}-{active['p90']:.2f} | "
                f"{block['fold_classes']['mean']:.2f} |"
            )
    lines += [
        "",
        "## TTB 令牌-时间包路由覆盖",
        "",
        "| 模型 | bundle token数 | empty | active<=2 | active<=4 | active<=8 | active<=16 | 非空bundle平均active lane |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for bundle in result["bundles"]:
            lines.append(
                f"| {result['model']} | {bundle['bundle_tokens']} | {pct(bundle['empty_ratio'])} | "
                f"{pct(bundle['active_le2_ratio'])} | {pct(bundle['active_le4_ratio'])} | "
                f"{pct(bundle['active_le8_ratio'])} | {pct(bundle['active_le16_ratio'])} | "
                f"{bundle['mean_active_lanes_per_nonempty']:.2f} |"
            )
    lines += [
        "",
        "## 对架构的直接约束",
        "",
        "- **主线先做统一同构核。** H67/H68 工作负载近似，H67 作为功能超集，H68 编译期关闭 Motion-XOR 并缩减 class 逻辑。",
        "- **前端按 temporal pair 驻留。** 一次读取 `{Q0,Q1,K0,K1}`，联合生成两个 score；全空 pair 精确注入两个 class-2 项，不能删除 Shiftmax 分母贡献。",
        "- **context 数必须与 commit 结构联合决定。** 两阶段模型支持 2-context，但端口感知三阶段模型表明高 PCCC 合并率下 4-context 仍可能有两位数收益；RTL 应参数化 1/2/4，先实现 2，物理数量等待 ordered trace，暂不考虑 8-context。",
        "- **调度配置至少到 block。** S1B0、S2B3 接近完全静默，而 S0B0、S2B5、S3B1 明显更活跃；统一阈值会造成稀疏路径排队或稠密路径空转。",
        "- **蝶形网络只作为可淘汰候选。** 只有新 profile 证明四向量并集索引包在多数非空 pair 中明显短于 128-bit bitmap，且综合后网络能耗低于 SRAM/互连节省，才实现动态压紧。",
        "- **暂不主张稀疏/稠密双核。** 当前数据证明稀疏和异质性存在，但没有有序 burst、有限 FIFO 和双路径 PPA，不能证明双核优于表示可切换的同构核。",
        "",
        "## 证据边界与待补统计",
        "",
        "已完成：100 样本、12 个 block 的调用级计数、stage/block 分位数、每样本 attention 周期代理、TTB bundle 覆盖、Delta 分桶。",
        "",
        "尚待 GPU 队列释放后完成：逐 pair 四向量事件并集、逐 row active/fold/class collision、有序 burst、有限 FIFO、bank conflict、输入事件/光流幅值/梯度/AEE 相关性。未完成项不得写成论文实测结论。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    results = [analyze_case(name, case) for name, case in CASES.items()]
    output = ROOT / "results/h67_h68_profile100_arch_features.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"schema_version": 1, "results": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output.with_suffix(".md").write_text(render(results), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
