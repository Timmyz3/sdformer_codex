#!/usr/bin/env python3
"""H67 projection / SCS 周期分账（CPU only，读 compact profile + GCM-P DSE）。

不重跑 GPU profile。不修改 GPT 已有脚本：本文件为独立分账入口。

分账口径：
1) SCS：固定 35 类扫描 vs 占用类扫描（与 score_class_scan_cycle_model 同公式，
   输入改为 compact profile 的 stage 均值）。
2) Projection work-items：direct active-lanes vs final-gate NMF terms / multicast
   delivery（来自 compact 全局计数，精确）。
3) Projection cycles：优先引用 gcmp_*_multicast_dse.json 的 ordered-trace 逐行
   模型结果；另给聚合近似（无 ceil 行效应）作对照。

证据等级：[模型] 周期；[prof] work-item 计数。非 DC/FPS/mW。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPACT = ROOT / "results/profile100_compact_arch_stats_20260714.json"
DEFAULT_DSE = ROOT / "results/gcmp_h67_multicast_dse.json"
DEFAULT_JSON = ROOT / "results/projection_scs_cycle_ledger_20260715.json"
DEFAULT_MD = ROOT / "results/projection_scs_cycle_ledger_20260715.md"

N_TOKENS = 162
CONTROL_CYCLES = 3
H67_FIXED_CLASSES = 35
H67_CLASS_PIPE = 2.0  # 每占用类两拍
FREQUENCY_HZ = 500_000_000
# H67 注意力投影输出通道：GCM-P DSE 反推 ceil(OC/L) 与 192 一致
DEFAULT_OUTPUT_CHANNELS = 192
# G1 top 相序开销（RTL 控制器口径，加到 DSE product||delivery 之外）
DEFAULT_NMF_K_LANES = 32  # NMF 目录扫描 lane 维（K 通道）
DEFAULT_ACC_BANKS = 2
DEFAULT_NMF_SLOTS = 4
FINISH_CYCLES_PER_ROW = 2  # ST_FINISH + ST_DONE

# 架构锁配置（docs/76）：G=1, S=4
LOCK_CLASS_SLOTS = 4
LOCK_CONFIGS = (
    # multicast_width, output_lanes, product_engines
    (1, 8, 1),
    (1, 16, 1),
    (1, 32, 1),
    (4, 8, 1),
    (4, 16, 1),
    (4, 32, 1),
    (4, 32, 2),
    (8, 16, 1),
    (8, 32, 1),
    (8, 32, 2),
)


def scs_row_cycles(active: float, class_cycles: float) -> float:
    return N_TOKENS + max(active, 1.0) + class_cycles + active + CONTROL_CYCLES


def scs_ledger(model: dict[str, Any]) -> dict[str, Any]:
    stages_out = []
    total_fixed = 0.0
    total_scs = 0.0
    total_rows = 0
    for st in model["stages"]:
        rows = int(st["rows_per_frame"])
        active = float(st["zaf_active_entries_mean"])
        fold = float(st["zaf_fold_classes_mean"])
        fixed = scs_row_cycles(active, float(H67_FIXED_CLASSES))
        sparse = scs_row_cycles(active, H67_CLASS_PIPE * fold)
        total_fixed += rows * fixed
        total_scs += rows * sparse
        total_rows += rows
        stages_out.append(
            {
                "stage": int(st["stage"]),
                "rows_per_frame": rows,
                "active_entries_mean": active,
                "fold_classes_mean": fold,
                "kzero_token_ratio": float(st["zaf_kzero_token_ratio"]),
                "ttb2_empty_ratio": float(st["ttb2_empty_ratio"]),
                "fixed_cycles_per_row": fixed,
                "scs_cycles_per_row": sparse,
                "cycle_reduction": (fixed - sparse) / fixed if fixed else 0.0,
                "fixed_cycles_frame": rows * fixed,
                "scs_cycles_frame": rows * sparse,
            }
        )
    weighted = model.get("weighted", {})
    return {
        "schema": "scs_row_engine",
        "tokens_per_row": N_TOKENS,
        "fixed_classes": H67_FIXED_CLASSES,
        "class_pipe_cycles": H67_CLASS_PIPE,
        "control_cycles": CONTROL_CYCLES,
        "rows_per_frame": total_rows,
        "fixed_cycles_per_frame": total_fixed,
        "scs_cycles_per_frame": total_scs,
        "cycle_reduction": (total_fixed - total_scs) / total_fixed if total_fixed else 0.0,
        "fixed_fps_500mhz_kernel_only": FREQUENCY_HZ / total_fixed if total_fixed else 0.0,
        "scs_fps_500mhz_kernel_only": FREQUENCY_HZ / total_scs if total_scs else 0.0,
        "weighted_profile": weighted,
        "stages": stages_out,
        "限制": (
            "单注意力行核无外部停顿模型；不含 Q/K 投影、ATLIF、SRAM 等；"
            "不能写成端到端 FPS。"
        ),
        "evidence": "[模型]+[prof stage 均值]",
    }


def projection_work_items(bt: dict[str, Any]) -> dict[str, Any]:
    baseline = int(bt["projection_baseline_active_lanes"])
    gate_terms = int(bt["projection_gate_class_channel_terms_deploy"])
    score_terms = int(bt["projection_class_channel_terms_h67"])
    g1_terms = int(bt.get("projection_gate_group_terms_g1", gate_terms))
    deliveries = {
        f"m{m}": int(bt[f"projection_gate_multicast_delivery_m{m}"])
        for m in (1, 2, 4, 8, 16)
        if f"projection_gate_multicast_delivery_m{m}" in bt
    }
    group_terms = {
        f"g{g}": int(bt[f"projection_gate_group_terms_g{g}"])
        for g in (1, 2, 4, 8, 16)
        if f"projection_gate_group_terms_g{g}" in bt
    }
    return {
        "schema": "projection_work_items",
        "baseline_active_lanes": baseline,
        "score_class_channel_terms": score_terms,
        "final_gate_class_channel_terms": gate_terms,
        "g1_group_terms": g1_terms,
        "product_reduction_vs_direct": 1.0 - gate_terms / baseline if baseline else 0.0,
        "gate_vs_score_extra_merge": 1.0 - gate_terms / score_terms if score_terms else 0.0,
        "multicast_delivery": deliveries,
        "group_terms": group_terms,
        "row_active_projection_gate_classes_mean_deploy": float(
            bt.get("row_active_projection_gate_classes_mean_deploy", 0.0)
        ),
        "row_active_projection_classes_mean_h67": float(
            bt.get("row_active_projection_classes_mean_h67", 0.0)
        ),
        "pair_empty_ratio": float(bt.get("pair_empty_ratio", 0.0)),
        "token_kzero_ratio": float(bt.get("token_kzero_ratio", 0.0)),
        "evidence": "[prof] compact 全局计数",
    }


def approx_projection_cycles(
    *,
    baseline_lanes: int,
    gate_terms: int,
    delivery: int,
    output_channels: int,
    output_lanes: int,
    product_engines: int,
) -> dict[str, Any]:
    """聚合近似：无 per-row ceil，仅作 DSE 对照，不得单独引用为 DATE 数字。"""
    chunks = math.ceil(output_channels / output_lanes)
    direct = math.ceil(baseline_lanes / product_engines) * chunks
    product = math.ceil(gate_terms / product_engines) * chunks
    deliv = delivery * chunks
    cand = max(product, deliv)
    return {
        "chunks": chunks,
        "approx_direct_cycles": direct,
        "approx_product_cycles": product,
        "approx_delivery_cycles": deliv,
        "approx_candidate_cycles": cand,
        "approx_ideal_speedup": direct / cand if cand else 0.0,
        "approx_product_vs_direct": 1.0 - product / direct if direct else 0.0,
        "bottleneck": "delivery" if deliv >= product else "product",
        "限制": "聚合 ceil 近似，忽略行级 ceil 与 bank 冲突；优先用 DSE 精确行模型。",
    }


def pick_dse_configs(
    dse: dict[str, Any],
    *,
    class_slots: int = LOCK_CLASS_SLOTS,
) -> list[dict[str, Any]]:
    wanted = set(LOCK_CONFIGS)
    out: list[dict[str, Any]] = []
    for cfg in dse.get("configurations", []):
        key = (
            int(cfg["multicast_width"]),
            int(cfg["output_lanes"]),
            int(cfg["product_engines"]),
        )
        if int(cfg["class_slots"]) != class_slots or key not in wanted:
            continue
        out.append(
            {
                "class_slots": int(cfg["class_slots"]),
                "multicast_width": int(cfg["multicast_width"]),
                "output_lanes": int(cfg["output_lanes"]),
                "product_engines": int(cfg["product_engines"]),
                "rows": int(cfg["rows"]),
                "overflow_rows": int(cfg["overflow_rows"]),
                "overflow_ratio": float(cfg["overflow_ratio"]),
                "direct_cycles": int(cfg["direct_cycles"]),
                "candidate_cycles": int(cfg["candidate_cycles"]),
                "product_cycles": int(cfg["product_cycles"]),
                "delivery_cycles": int(cfg["delivery_cycles"]),
                "ideal_speedup": float(cfg["ideal_speedup"]),
                "candidate_p50": float(cfg["candidate_p50"]),
                "candidate_p95": float(cfg["candidate_p95"]),
                "candidate_p99": float(cfg["candidate_p99"]),
                "candidate_max": float(cfg["candidate_max"]),
                "bottleneck": (
                    "delivery"
                    if int(cfg["delivery_cycles"]) >= int(cfg["product_cycles"])
                    else "product"
                ),
                "evidence": "[模型] GCM-P ordered-trace 逐行",
            }
        )
    out.sort(key=lambda r: (r["multicast_width"], r["output_lanes"], r["product_engines"]))
    return out


def projection_phase_overhead(
    *,
    rows: int,
    tokens_per_row: int = N_TOKENS,
    nmf_slots: int = DEFAULT_NMF_SLOTS,
    nmf_k_lanes: int = DEFAULT_NMF_K_LANES,
    acc_banks: int = DEFAULT_ACC_BANKS,
    mean_terms_per_row: float,
    dse_candidate_total: int,
    dse_product_total: int,
    dse_delivery_total: int,
    dse_direct_total: int,
) -> dict[str, Any]:
    """G1 顶层串行相序：NMF 建表 + (DSE product||delivery) + bias-commit + finish。

    与 RTL `hitflow_g1_projection_top` 对齐：
      ST_RUN/BUILD: 逐 token 建目录（约 1 token/cycle）
      ST_DRAIN: 目录扫描；非空项走 product→multicast（已由 DSE candidate 覆盖）
                 空 slot-lane 1 拍前进，可与 product 流水重叠 → **不**加到 wall
      ST_BIAS: 按 token 串行发 bias，bank 并行度 = BANKS；每 token 约 2 拍
               （accept busy + writeback），模型 = ceil(T/BANKS)*2
      ST_FINISH/DONE: 固定 2 拍/行
    """
    if rows <= 0:
        raise ValueError("rows must be positive")
    nmf_build_total = rows * tokens_per_row
    slot_lane_cells = nmf_slots * nmf_k_lanes
    empty_scan_per_row = max(0.0, float(slot_lane_cells) - float(mean_terms_per_row))
    empty_scan_total = int(round(rows * empty_scan_per_row))
    # empty advances absorbed while product pipeline busy; report but do not add
    bias_per_row = math.ceil(tokens_per_row / acc_banks) * 2
    bias_total = rows * bias_per_row
    finish_total = rows * FINISH_CYCLES_PER_ROW
    additive = nmf_build_total + bias_total + finish_total
    total_serial = int(dse_candidate_total) + additive
    total_serial_direct = int(dse_direct_total) + additive
    return {
        "schema": "g1_top_phase_overhead",
        "rows": rows,
        "tokens_per_row": tokens_per_row,
        "nmf_slots": nmf_slots,
        "nmf_k_lanes": nmf_k_lanes,
        "acc_banks": acc_banks,
        "mean_terms_per_row": mean_terms_per_row,
        "phases": {
            "nmf_build_token_stream": nmf_build_total,
            "directory_empty_scan_absorbed_not_added": empty_scan_total,
            "dse_product_cycles": int(dse_product_total),
            "dse_delivery_cycles": int(dse_delivery_total),
            "dse_candidate_product_or_delivery": int(dse_candidate_total),
            "bias_commit": bias_total,
            "finish_done": finish_total,
        },
        "bias_cycles_per_row": bias_per_row,
        "additive_outside_dse": additive,
        "total_serial_with_overhead": total_serial,
        "total_serial_direct_baseline": total_serial_direct,
        "overhead_fraction_of_total": additive / total_serial if total_serial else 0.0,
        "effective_speedup_vs_direct_serial": (
            total_serial_direct / total_serial if total_serial else 0.0
        ),
        "dse_only_speedup": (
            dse_direct_total / dse_candidate_total if dse_candidate_total else 0.0
        ),
        "限制": (
            "建表与 bias 为串行下界模型；未计 weight SRAM 延迟、bank conflict、"
            "NMF 与 product 更深流水重叠；empty scan 假定被 product 路径吸收。"
        ),
        "evidence": "[模型] RTL 相序 + [prof] terms/rows + [模型] DSE candidate",
    }


def attach_phase_overhead_to_dse(
    dse_rows: list[dict[str, Any]],
    *,
    work: dict[str, Any],
    nmf_k_lanes: int,
    acc_banks: int,
) -> list[dict[str, Any]]:
    out = []
    gate_terms = int(work["final_gate_class_channel_terms"])
    for row in dse_rows:
        rows = int(row["rows"])
        mean_terms = gate_terms / rows if rows else 0.0
        phase = projection_phase_overhead(
            rows=rows,
            mean_terms_per_row=mean_terms,
            dse_candidate_total=int(row["candidate_cycles"]),
            dse_product_total=int(row["product_cycles"]),
            dse_delivery_total=int(row["delivery_cycles"]),
            dse_direct_total=int(row["direct_cycles"]),
            nmf_k_lanes=nmf_k_lanes,
            acc_banks=acc_banks,
        )
        enriched = dict(row)
        enriched["phase_overhead"] = phase
        enriched["total_serial_with_overhead"] = phase["total_serial_with_overhead"]
        enriched["effective_speedup_vs_direct_serial"] = phase[
            "effective_speedup_vs_direct_serial"
        ]
        enriched["overhead_fraction_of_total"] = phase["overhead_fraction_of_total"]
        out.append(enriched)
    return out


def architecture_recommendation(
    scs: dict[str, Any],
    work: dict[str, Any],
    dse_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    best = None
    for row in dse_rows:
        if row["overflow_ratio"] > 0.001:
            continue
        # Prefer effective speedup after phase overhead when present
        score = float(
            row.get("effective_speedup_vs_direct_serial", row["ideal_speedup"])
        )
        if best is None or score > float(
            best.get("effective_speedup_vs_direct_serial", best["ideal_speedup"])
        ):
            best = row
    # Prefer balanced lock: S=4,M=4,L=32,P=1 as default paper config if present
    preferred = None
    for row in dse_rows:
        if (
            row["multicast_width"] == 4
            and row["output_lanes"] == 32
            and row["product_engines"] == 1
        ):
            preferred = row
            break
    return {
        "lock": "H67 + SCS + NMF(G=1,S=4) + exact pair/K-zero",
        "scs_cycle_reduction": scs["cycle_reduction"],
        "nmf_product_work_reduction": work["product_reduction_vs_direct"],
        "preferred_gcmp_config": preferred,
        "best_speedup_config_among_lock_set": best,
        "defer": ["G>=2 without EDP>=15%", "PHEA dual-core", "butterfly fabric"],
        "gpu_required": False,
    }


def render_md(result: dict[str, Any]) -> str:
    scs = result["scs"]
    work = result["projection_work_items"]
    lines = [
        "# Projection / SCS 周期分账（CPU）",
        "",
        f"- compact：`{result['inputs']['compact']}`",
        f"- DSE：`{result['inputs']['dse']}`",
        f"- 模型：`{result['variant']}`；GPU：`不需要`",
        "",
        "## 1. SCS 行核周期",
        "",
        f"| 固定扫描周期/帧 | 占用类扫描周期/帧 | 下降 | 500MHz 固定帧率* | 500MHz SCS 帧率* |",
        f"|---:|---:|---:|---:|---:|",
        f"| {scs['fixed_cycles_per_frame']:.0f} | {scs['scs_cycles_per_frame']:.0f} | "
        f"{100*scs['cycle_reduction']:.2f}% | {scs['fixed_fps_500mhz_kernel_only']:.2f} | "
        f"{scs['scs_fps_500mhz_kernel_only']:.2f} |",
        "",
        "\\*仅注意力行核，非端到端。",
        "",
        "| stage | rows | active/row | fold | fixed cyc/row | SCS cyc/row | 下降 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for st in scs["stages"]:
        lines.append(
            f"| {st['stage']} | {st['rows_per_frame']} | {st['active_entries_mean']:.2f} | "
            f"{st['fold_classes_mean']:.2f} | {st['fixed_cycles_per_row']:.2f} | "
            f"{st['scs_cycles_per_row']:.2f} | {100*st['cycle_reduction']:.2f}% |"
        )
    lines += [
        "",
        "## 2. Projection work-items（[prof]）",
        "",
        f"| baseline active lanes | final-gate terms | 乘积减少 | gate 相对 score 合并 |",
        f"|---:|---:|---:|---:|",
        f"| {work['baseline_active_lanes']} | {work['final_gate_class_channel_terms']} | "
        f"{100*work['product_reduction_vs_direct']:.2f}% | "
        f"{100*work['gate_vs_score_extra_merge']:.2f}% |",
        "",
        "### Multicast delivery",
        "",
        "| M | delivery transactions | vs M=1 |",
        "|---:|---:|---:|",
    ]
    m1 = work["multicast_delivery"].get("m1", 0)
    for m in (1, 2, 4, 8, 16):
        key = f"m{m}"
        if key not in work["multicast_delivery"]:
            continue
        d = work["multicast_delivery"][key]
        ratio = d / m1 if m1 else 0.0
        lines.append(f"| {m} | {d} | {100*ratio:.2f}% |")
    lines += [
        "",
        "## 3. GCM-P 周期（DSE 精确行模型，S=4 锁表）",
        "",
        "| M | L | P | overflow | direct | candidate | product | delivery | speedup | p95 | bottleneck |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["projection_dse_lock"]:
        lines.append(
            f"| {row['multicast_width']} | {row['output_lanes']} | {row['product_engines']} | "
            f"{row['overflow_ratio']:.4%} | {row['direct_cycles']} | {row['candidate_cycles']} | "
            f"{row['product_cycles']} | {row['delivery_cycles']} | {row['ideal_speedup']:.3f} | "
            f"{row['candidate_p95']:.1f} | {row['bottleneck']} |"
        )
    lines += [
        "",
        "## 4. G1 相序开销（NMF 建表 + bias-commit + finish）",
        "",
        "在 DSE candidate（product||delivery）之外，按 RTL 控制器串行叠加：",
        "",
        "| M | L | P | DSE candidate | +NMF build | +bias | +finish | **总串行** | 开销占比 | DSE speedup | **有效 speedup** |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["projection_dse_lock"]:
        ph = row.get("phase_overhead", {})
        phases = ph.get("phases", {})
        lines.append(
            f"| {row['multicast_width']} | {row['output_lanes']} | {row['product_engines']} | "
            f"{row['candidate_cycles']} | {phases.get('nmf_build_token_stream', 0)} | "
            f"{phases.get('bias_commit', 0)} | {phases.get('finish_done', 0)} | "
            f"{row.get('total_serial_with_overhead', 0)} | "
            f"{100*float(row.get('overhead_fraction_of_total', 0)):.1f}% | "
            f"{row['ideal_speedup']:.3f} | "
            f"{float(row.get('effective_speedup_vs_direct_serial', 0)):.3f} |"
        )
    if result["projection_dse_lock"]:
        sample = result["projection_dse_lock"][0].get("phase_overhead", {})
        lines += [
            "",
            f"- bias 模型：`ceil(T/BANKS)*2`，BANKS=`{sample.get('acc_banks', DEFAULT_ACC_BANKS)}`，"
            f"T=`{sample.get('tokens_per_row', N_TOKENS)}`；",
            f"- NMF 建表：`T` token/行串行；empty directory scan 假定被 product 路径吸收（报告但不加 wall）；",
            f"- 有效 speedup = (direct+开销)/(candidate+开销)，通常 **低于** 纯 DSE ideal。",
        ]
    lines += [
        "",
        "## 5. 聚合近似（对照，非主数字）",
        "",
        "| M | L | P | approx speedup | bottleneck |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in result["projection_approx"]:
        lines.append(
            f"| {row['multicast_width']} | {row['output_lanes']} | {row['product_engines']} | "
            f"{row['approx']['approx_ideal_speedup']:.3f} | {row['approx']['bottleneck']} |"
        )
    rec = result["recommendation"]
    lines += [
        "",
        "## 6. 架构建议",
        "",
        f"- **锁**：`{rec['lock']}`",
        f"- SCS 周期下降：`{100*rec['scs_cycle_reduction']:.2f}%`（行核模型）",
        f"- NMF work-item 乘积减少：`{100*rec['nmf_product_work_reduction']:.2f}%` [prof]",
        f"- GPU：`{'需要' if rec['gpu_required'] else '不需要'}（本分账）`",
        f"- 暂缓：{', '.join(rec['defer'])}",
        "",
        "## 7. 边界",
        "",
        "- 周期 = backend 模型，不是 DC/SAIF；",
        "- DSE candidate = max(product, delivery)，假设可完全重叠；",
        "- **已含** NMF 建表 token 流、bias-commit、finish 串行下界；",
        "- 仍未含 weight SRAM 延迟、bank conflict 细账；",
        "- S=4 overflow≈0.014% 仍需 fallback 合同，但可后做。",
        "",
    ]
    return "\n".join(lines)


def build_ledger(
    *,
    compact_path: Path,
    dse_path: Path,
    variant: str,
    output_channels: int,
    nmf_k_lanes: int = DEFAULT_NMF_K_LANES,
    acc_banks: int = DEFAULT_ACC_BANKS,
) -> dict[str, Any]:
    compact = json.loads(compact_path.read_text(encoding="utf-8"))
    if variant not in compact["models"]:
        raise KeyError(f"compact 无模型 {variant}")
    model = compact["models"][variant]
    bt = model["binary_temporal_pairs"]
    scs = scs_ledger(model)
    work = projection_work_items(bt)

    dse_rows: list[dict[str, Any]] = []
    if dse_path.is_file():
        dse = json.loads(dse_path.read_text(encoding="utf-8"))
        dse_rows = pick_dse_configs(dse)
        dse_rows = attach_phase_overhead_to_dse(
            dse_rows,
            work=work,
            nmf_k_lanes=nmf_k_lanes,
            acc_banks=acc_banks,
        )

    approx_rows = []
    for m, lanes, engines in LOCK_CONFIGS:
        deliv_key = f"m{m}"
        if deliv_key not in work["multicast_delivery"]:
            continue
        approx = approx_projection_cycles(
            baseline_lanes=work["baseline_active_lanes"],
            gate_terms=work["final_gate_class_channel_terms"],
            delivery=work["multicast_delivery"][deliv_key],
            output_channels=output_channels,
            output_lanes=lanes,
            product_engines=engines,
        )
        approx_rows.append(
            {
                "class_slots": LOCK_CLASS_SLOTS,
                "multicast_width": m,
                "output_lanes": lanes,
                "product_engines": engines,
                "output_channels": output_channels,
                "approx": approx,
            }
        )

    result = {
        "schema_version": 2,
        "variant": variant,
        "gpu_required": False,
        "inputs": {
            "compact": str(compact_path),
            "dse": str(dse_path) if dse_path.is_file() else None,
            "output_channels": output_channels,
            "nmf_k_lanes": nmf_k_lanes,
            "acc_banks": acc_banks,
        },
        "scs": scs,
        "projection_work_items": work,
        "projection_dse_lock": dse_rows,
        "projection_approx": approx_rows,
        "recommendation": architecture_recommendation(scs, work, dse_rows),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--dse", type=Path, default=DEFAULT_DSE)
    parser.add_argument("--variant", default="H67", choices=("H67", "H68", "TTX"))
    parser.add_argument("--output-channels", type=int, default=DEFAULT_OUTPUT_CHANNELS)
    parser.add_argument("--nmf-k-lanes", type=int, default=DEFAULT_NMF_K_LANES)
    parser.add_argument("--acc-banks", type=int, default=DEFAULT_ACC_BANKS)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    # H68/TTX may not have matching default DSE; allow missing
    dse_path = args.dse
    if args.variant == "H68":
        alt = ROOT / "results/gcmp_h68_multicast_dse.json"
        if alt.is_file() and args.dse == DEFAULT_DSE:
            dse_path = alt

    result = build_ledger(
        compact_path=args.compact,
        dse_path=dse_path,
        variant=args.variant,
        output_channels=args.output_channels,
        nmf_k_lanes=args.nmf_k_lanes,
        acc_banks=args.acc_banks,
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.md.write_text(render_md(result), encoding="utf-8")
    print(args.json)
    print(args.md)
    print(
        f"SCS reduction={100*result['scs']['cycle_reduction']:.2f}% "
        f"NMF work-item reduction={100*result['projection_work_items']['product_reduction_vs_direct']:.2f}% "
        f"gpu_required=False"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
