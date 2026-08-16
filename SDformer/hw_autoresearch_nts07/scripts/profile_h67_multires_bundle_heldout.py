#!/usr/bin/env python3
"""用预划分sample评估Motion多分辨率bundle静态调度的held-out上界。"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from scripts.profile_h67_zkqi_multisample_ordered import (
        DEFAULT_PROFILE,
        EXPECTED_NAMES,
        block_identity,
        decode_record,
        distribution,
        receipt,
        ttb_depth1_front_cycles,
        validate_profile_contract,
    )
except ModuleNotFoundError:
    from profile_h67_zkqi_multisample_ordered import (
        DEFAULT_PROFILE,
        EXPECTED_NAMES,
        block_identity,
        decode_record,
        distribution,
        receipt,
        ttb_depth1_front_cycles,
        validate_profile_contract,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/h67_multires_bundle_heldout_20260809"
BUNDLE_SIZES = (4, 8, 16, 32)
PAIRS = 225
CALIBRATION_SAMPLES = tuple(range(50))
HELDOUT_SAMPLES = tuple(range(50, 100))


def bundle_front_cycles(active_pair: np.ndarray, bundle_size: int) -> np.ndarray:
    active = np.asarray(active_pair, dtype=np.bool_)
    if active.ndim != 3 or active.shape[-1] != PAIRS:
        raise ValueError("active_pair必须是[window,head,225]")
    if bundle_size not in BUNDLE_SIZES:
        raise ValueError("bundle_size不在冻结DSE集合")
    windows, heads, _ = active.shape
    groups = math.ceil(PAIRS / bundle_size)
    padded = np.pad(
        active,
        ((0, 0), (0, 0), (0, groups * bundle_size - PAIRS)),
        constant_values=False,
    )
    counts = padded.reshape(windows, heads, groups, bundle_size).sum(
        axis=3, dtype=np.int64
    )
    return ttb_depth1_front_cycles(
        counts.reshape(windows * heads, groups)
    ).reshape(windows, heads)


def choose_best_size(totals: dict[int, int]) -> int:
    if set(totals) != set(BUNDLE_SIZES):
        raise ValueError("候选bundle集合漂移")
    # 周期相同时优先较窄selector，避免用同周期的大mask制造虚假候选。
    return min(BUNDLE_SIZES, key=lambda size: (int(totals[size]), size))


def build_dataset(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    dataset: list[dict[str, Any]] = []
    residuals: list[int] = []
    monotonic_violations = 0
    b32_vs_b8_faster = 0
    b32_vs_b8_equal = 0
    b32_vs_b8_slower = 0
    trace_exact_checks = 0
    for record in records:
        sample = int(record["sample_id"])
        stage, block, name = block_identity(record)
        metrics, checks = decode_record(record)
        for key in (
            "ttb_active_trace_exact", "ttb_k_trace_exact",
            "ttb_motion_trace_exact",
        ):
            if not checks[key]:
                raise ValueError(f"{record.get('name')}: {key}不成立")
            trace_exact_checks += 1
        active_pair = (checks["k_count"] != 0).any(axis=0)
        cycles = {}
        for bundle_size in BUNDLE_SIZES:
            front = bundle_front_cycles(active_pair, bundle_size)
            cycles[bundle_size] = (
                front + metrics["backend_cycles"] + metrics["preload_cycles"]
            ).astype(np.int64, copy=False)
        residuals.extend(
            (cycles[8] - metrics["ttb_e2e_cycles"]).reshape(-1).tolist()
        )
        monotonic_violations += int(
            (
                (cycles[8] > cycles[4])
                | (cycles[16] > cycles[8])
                | (cycles[32] > cycles[16])
            ).sum()
        )
        delta32 = cycles[32] - cycles[8]
        b32_vs_b8_faster += int((delta32 < 0).sum())
        b32_vs_b8_equal += int((delta32 == 0).sum())
        b32_vs_b8_slower += int((delta32 > 0).sum())
        dataset.append(
            {
                "sample": sample,
                "stage": stage,
                "block": block,
                "name": name,
                "cycles": cycles,
                "baseline": metrics["baseline_e2e_cycles"].astype(
                    np.int64, copy=False
                ),
            }
        )
    if not residuals or min(residuals) != 0 or max(residuals) != 0:
        raise ValueError("B8 DSE模型与已校准TTB8模型不一致")
    return dataset, {
        "rows": sum(entry["baseline"].size for entry in dataset),
        "b8_cycle_residual_min": min(residuals),
        "b8_cycle_residual_max": max(residuals),
        "ordered_ttb_trace_exact_checks": trace_exact_checks,
        "coarser_bundle_monotonic_violations": monotonic_violations,
        "b32_vs_b8_rows": {
            "faster": b32_vs_b8_faster,
            "equal": b32_vs_b8_equal,
            "slower": b32_vs_b8_slower,
        },
    }


def total_by_size(
    dataset: list[dict[str, Any]],
    samples: set[int],
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[int, int]:
    result = {size: 0 for size in BUNDLE_SIZES}
    for entry in dataset:
        if entry["sample"] not in samples or (predicate and not predicate(entry)):
            continue
        for size in BUNDLE_SIZES:
            result[size] += int(entry["cycles"][size].sum())
    return result


def freeze_policy(
    dataset: list[dict[str, Any]], calibration: set[int]
) -> dict[str, Any]:
    global_totals = total_by_size(dataset, calibration)
    global_size = choose_best_size(global_totals)
    stage_sizes = {}
    for stage in range(4):
        totals = total_by_size(
            dataset, calibration, lambda entry, stage=stage: entry["stage"] == stage
        )
        stage_sizes[str(stage)] = choose_best_size(totals)
    block_sizes = {}
    for name in EXPECTED_NAMES:
        totals = total_by_size(
            dataset, calibration, lambda entry, name=name: entry["name"] == name
        )
        block_sizes[name] = choose_best_size(totals)
    return {
        "global_totals": {str(key): value for key, value in global_totals.items()},
        "global_size": global_size,
        "stage_sizes": stage_sizes,
        "block_sizes": block_sizes,
    }


def selected_cycles(
    entry: dict[str, Any], strategy: str, policy: dict[str, Any]
) -> np.ndarray:
    if strategy.startswith("fixed_b"):
        return entry["cycles"][int(strategy.removeprefix("fixed_b"))]
    if strategy == "global_static":
        return entry["cycles"][int(policy["global_size"])]
    if strategy == "stage_static":
        return entry["cycles"][int(policy["stage_sizes"][str(entry["stage"])])]
    if strategy == "block_static":
        return entry["cycles"][int(policy["block_sizes"][entry["name"]])]
    if strategy == "candidate_oracle":
        return np.minimum.reduce([entry["cycles"][size] for size in BUNDLE_SIZES])
    raise ValueError(f"未知策略: {strategy}")


def evaluate_strategy(
    dataset: list[dict[str, Any]],
    samples: set[int],
    strategy: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    rows: list[np.ndarray] = []
    baseline_rows: list[np.ndarray] = []
    per_sample = defaultdict(int)
    per_stage = defaultdict(int)
    for entry in dataset:
        if entry["sample"] not in samples:
            continue
        values = selected_cycles(entry, strategy, policy)
        rows.append(values.reshape(-1))
        baseline_rows.append(entry["baseline"].reshape(-1))
        per_sample[entry["sample"]] += int(values.sum())
        per_stage[entry["stage"]] += int(values.sum())
    row_values = np.concatenate(rows)
    baseline_values = np.concatenate(baseline_rows)
    sample_values = np.array(
        [per_sample[sample] for sample in sorted(samples)], dtype=np.int64
    )
    return {
        "rows": int(row_values.size),
        "cycles": int(row_values.sum()),
        "baseline_cycles": int(baseline_values.sum()),
        "speedup_vs_rqtb": float(baseline_values.sum() / row_values.sum()),
        "row_distribution": distribution(row_values),
        "sample_distribution": distribution(sample_values),
        "stage_cycles": {str(stage): per_stage[stage] for stage in range(4)},
    }


def compare_strategy(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    return {
        "cycle_reduction": 1.0 - candidate["cycles"] / reference["cycles"],
        "speedup": reference["cycles"] / candidate["cycles"],
        "row_p95_reduction": 1.0
        - candidate["row_distribution"]["p95"]
        / reference["row_distribution"]["p95"],
        "row_p99_reduction": 1.0
        - candidate["row_distribution"]["p99"]
        / reference["row_distribution"]["p99"],
        "sample_p95_reduction": 1.0
        - candidate["sample_distribution"]["p95"]
        / reference["sample_distribution"]["p95"],
        "sample_p99_reduction": 1.0
        - candidate["sample_distribution"]["p99"]
        / reference["sample_distribution"]["p99"],
    }


def selector_contract() -> dict[str, Any]:
    return {
        str(size): {
            "groups": math.ceil(PAIRS / size),
            "padded_metadata_bits": math.ceil(PAIRS / size) * size,
            "flat_selector_inputs": size,
        }
        for size in BUNDLE_SIZES
    }


def decide_architecture_gate(comparison: dict[str, Any]) -> dict[str, str]:
    block = comparison["block_vs_global"]
    if block["cycle_reduction"] < 0.05:
        return {
            "status": "REJECT_AS_PARAMETER_DSE",
            "reason": "12-block静态策略相对最佳全局固定粒度的held-out周期增益不足5%",
        }
    if block["row_p95_reduction"] < 0 or block["row_p99_reduction"] < 0:
        return {
            "status": "REJECT_TAIL_REGRESSION",
            "reason": "held-out row p95或p99相对最佳全局粒度回退",
        }
    return {
        "status": "WAIT_CANONICAL_RTL_AND_PHYSICAL",
        "reason": "周期门槛通过，但selector/mux/control与固定5 ns物理代价尚未验证",
    }


def render_md(report: dict[str, Any]) -> str:
    policy = report["frozen_policy"]
    heldout = report["heldout"]
    comparison = report["heldout_comparison"]
    verdict = report["architecture_gate"]
    lines = [
        "# Motion多分辨率Bundle静态调度Held-out DSE",
        "",
        "## 结论",
        "",
        f"- calibration/held-out sample固定为`0..49`/`50..99`；B8模型残差="
        f"`{report['model_contract']['b8_cycle_residual_min']}.."
        f"{report['model_contract']['b8_cycle_residual_max']}`。",
        f"- calibration冻结的最佳全局粒度为`B{policy['global_size']}`。",
        f"- 12-block静态策略相对最佳全局固定粒度的held-out含preload周期变化为"
        f"`{comparison['block_vs_global']['cycle_reduction']:.4%}`。",
        f"- 固定B32相对固定B8的held-out含preload周期减少"
        f"`{comparison['fixed_b32_vs_b8']['cycle_reduction']:.4%}`；"
        "尚未计32-input selector代价。",
        f"- 全672000行粗粒度单调性违例为"
        f"`{report['model_contract']['coarser_bundle_monotonic_violations']}`。",
        f"- 架构晋级状态：**{verdict['status']}**；原因：{verdict['reason']}。",
        "- 本报告是`[模型-heldout]`，未计selector/mux/control物理代价，不是RTL或PPA。",
        "",
        "## 1. 冻结策略",
        "",
        f"全局：`B{policy['global_size']}`。",
        "",
        "| Stage | 冻结粒度 |",
        "|---:|---:|",
    ]
    for stage, size in policy["stage_sizes"].items():
        lines.append(f"| S{stage} | B{size} |")
    lines += [
        "",
        "| Block | 冻结粒度 |",
        "|---|---:|",
    ]
    for name, size in policy["block_sizes"].items():
        lines.append(f"| {name} | B{size} |")
    lines += [
        "",
        "## 2. Held-out消融",
        "",
        "| 策略 | 含preload周期 | 相对RQTB2S加速 | row p95 | row p99 |",
        "|---|---:|---:|---:|---:|",
    ]
    order = [
        "fixed_b4", "fixed_b8", "fixed_b16", "fixed_b32",
        "global_static", "stage_static", "block_static", "candidate_oracle",
    ]
    for name in order:
        row = heldout[name]
        lines.append(
            f"| {name} | {row['cycles']} | {row['speedup_vs_rqtb']:.6f}x | "
            f"{row['row_distribution']['p95']} | {row['row_distribution']['p99']} |"
        )
    lines += [
        "",
        "## 3. 相对最佳全局固定粒度",
        "",
        "| 策略 | 周期减少 | 加速 | row p95 | row p99 | sample p95 | sample p99 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in (
        "fixed_b32_vs_b8", "stage_vs_global", "block_vs_global",
        "oracle_vs_global",
    ):
        row = comparison[key]
        lines.append(
            f"| {key} | {row['cycle_reduction']:.4%} | {row['speedup']:.6f}x | "
            f"{row['row_p95_reduction']:.4%} | {row['row_p99_reduction']:.4%} | "
            f"{row['sample_p95_reduction']:.4%} | {row['sample_p99_reduction']:.4%} |"
        )
    lines += [
        "",
        "## 4. 尚未计入的结构代价",
        "",
        "| B | group数 | padded metadata bit | flat selector输入 |",
        "|---:|---:|---:|---:|",
    ]
    for size, row in report["selector_contract"].items():
        lines.append(
            f"| {size} | {row['groups']} | {row['padded_metadata_bits']} | "
            f"{row['flat_selector_inputs']} |"
        )
    lines += [
        "",
        "周期模型没有计入共享225-bit bitmap的多分辨率view、priority selector、"
        "mask mux、2-bit mode和控制扇出。若周期增益未过5%门槛，不应继续用RTL/PnR"
        "包装参数调优；若过门槛，也只能选择一个胜者进入canonical RTL和共同5 ns"
        "开放物理对照。",
        "",
        "## 5. 证据边界",
        "",
        "- held-out只表示未参与本轮bundle选择；此前已经看过同一profile的总体统计，"
        "  因而不是完全盲测数据集；",
        "- candidate_oracle不实现，只给冻结{B4,B8,B16,B32}集合内的逐row上界；",
        "- 所有周期均为row级无反压校准模型，不是encoder FPS；",
        "- 本轮不新增DATE贡献，只有通过周期与后续物理双门槛才允许晋级。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    records = profile.get("summary", {}).get("h60_records") or []
    profile_contract = validate_profile_contract(profile, records)
    dataset, model_contract = build_dataset(records)
    calibration = set(CALIBRATION_SAMPLES)
    heldout_samples = set(HELDOUT_SAMPLES)
    if calibration & heldout_samples or calibration | heldout_samples != set(range(100)):
        raise ValueError("calibration/held-out划分不守恒")

    policy = freeze_policy(dataset, calibration)
    strategies = [
        *(f"fixed_b{size}" for size in BUNDLE_SIZES),
        "global_static", "stage_static", "block_static", "candidate_oracle",
    ]
    calibration_results = {
        name: evaluate_strategy(dataset, calibration, name, policy)
        for name in strategies
    }
    heldout_results = {
        name: evaluate_strategy(dataset, heldout_samples, name, policy)
        for name in strategies
    }
    global_result = heldout_results["global_static"]
    comparison = {
        "fixed_b32_vs_b8": compare_strategy(
            heldout_results["fixed_b32"], heldout_results["fixed_b8"]
        ),
        "stage_vs_global": compare_strategy(
            heldout_results["stage_static"], global_result
        ),
        "block_vs_global": compare_strategy(
            heldout_results["block_static"], global_result
        ),
        "oracle_vs_global": compare_strategy(
            heldout_results["candidate_oracle"], global_result
        ),
    }
    block_gain = comparison["block_vs_global"]["cycle_reduction"]
    architecture_gate = decide_architecture_gate(comparison)

    report = {
        "schema": "h67_multires_bundle_heldout_dse_v1",
        "status": "PASS",
        "evidence_level": "[模型-heldout]",
        "scope": "row-level no-stall calibrated cycle model；不含selector物理代价",
        "split_contract": {
            "calibration_samples": list(CALIBRATION_SAMPLES),
            "heldout_samples": list(HELDOUT_SAMPLES),
            "说明": "held-out未参与本轮bundle选择，但此前总体profile已被观察",
        },
        "profile_contract": profile_contract,
        "model_contract": model_contract,
        "frozen_policy": policy,
        "calibration": calibration_results,
        "heldout": heldout_results,
        "heldout_comparison": comparison,
        "selector_contract": selector_contract(),
        "architecture_gate": architecture_gate,
        "block_choice_histogram": dict(
            sorted(Counter(policy["block_sizes"].values()).items())
        ),
        "source_profile": receipt(args.profile),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        render_md(report), encoding="utf-8"
    )
    print(
        f"PASS global=B{policy['global_size']} block_gain={block_gain:.8%} "
        f"gate={architecture_gate['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
