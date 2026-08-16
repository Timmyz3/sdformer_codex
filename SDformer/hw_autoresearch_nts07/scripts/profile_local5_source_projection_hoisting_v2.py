#!/usr/bin/env python3
"""评估 Local5 source projection hoisting，并与现有 DQFS 强基线对照。"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = (
    ROOT / "results/local5_fullres_bb1e4_joint_heads_profile100_20260809"
)
ROLES = 5
TOKENS = 450
LANES = 32
OUT_DIM = 32
MIN_LOCAL5_DEGREE = 3
MAX_LOCAL5_DEGREE = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def extract_top_level_object(path: Path, key: str) -> dict[str, Any]:
    """用 JSON decoder 读取大型 JSON 中靠后的一个顶层 object。"""

    marker = f'\n  "{key}": '.encode("utf-8")
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            position = mapped.rfind(marker)
            if position < 0:
                raise ValueError(f"{path} 缺少顶层 {key}")
            start = position + len(marker)
            tail = mapped[start:].decode("utf-8")
    value, _ = json.JSONDecoder().raw_decode(tail)
    if not isinstance(value, dict):
        raise ValueError(f"{path} 顶层 {key} 不是 object")
    return value


def popcount_u64(values: np.ndarray) -> np.ndarray:
    if values.dtype != np.uint64 or values.ndim != 1:
        raise ValueError("K bitmap 必须是一维 uint64")
    lookup = np.asarray([integer.bit_count() for integer in range(256)], dtype=np.uint8)
    byte_view = values.view(np.uint8).reshape(values.size, 8)
    return lookup[byte_view].sum(axis=1, dtype=np.uint16)


def analyze_descriptors(
    k_bitmap: np.ndarray,
    source_k_popcount: np.ndarray,
    incoming_gates: np.ndarray,
    valid_mask: np.ndarray,
    source_gate_count: np.ndarray,
    source_term_count: np.ndarray,
    source_delivery_count: np.ndarray,
) -> dict[str, Any]:
    descriptors = k_bitmap.size
    if (
        source_k_popcount.shape != (descriptors,)
        or incoming_gates.shape != (descriptors, ROLES)
        or valid_mask.shape != (descriptors,)
        or source_gate_count.shape != (descriptors,)
        or source_term_count.shape != (descriptors,)
        or source_delivery_count.shape != (descriptors,)
        or incoming_gates.dtype != np.uint16
        or valid_mask.dtype != np.uint8
    ):
        raise ValueError("descriptor 数组 shape/dtype 不符合冻结合同")
    if np.any(valid_mask > 0x1F):
        raise ValueError("valid mask 使用了五角色之外的 bit")

    recomputed_k = popcount_u64(k_bitmap)
    if not np.array_equal(recomputed_k, source_k_popcount):
        raise ValueError("source K popcount 与 bitmap 不一致")

    roles = np.arange(ROLES, dtype=np.uint8)
    valid = ((valid_mask[:, None] >> roles[None, :]) & 1).astype(bool)
    active_gate = valid & incoming_gates.astype(np.uint32).astype(bool)
    first_gate = active_gate.copy()
    for role in range(ROLES):
        for prior in range(role):
            first_gate[:, role] &= ~(
                active_gate[:, prior]
                & (incoming_gates[:, prior] == incoming_gates[:, role])
            )
    unique_gate_count = first_gate.sum(axis=1, dtype=np.uint8)
    degree = active_gate.sum(axis=1, dtype=np.uint8)
    active_source = recomputed_k > 0
    expected_gate_count = np.where(active_source, unique_gate_count, 0)
    expected_terms = recomputed_k.astype(np.uint32) * unique_gate_count
    expected_delivery = recomputed_k.astype(np.uint32) * degree
    if (
        not np.array_equal(expected_gate_count, source_gate_count)
        or not np.array_equal(expected_terms, source_term_count)
        or not np.array_equal(expected_delivery, source_delivery_count)
    ):
        raise ValueError("source gate/term/delivery 闭式计数不一致")

    term_ops = source_term_count.astype(np.uint64)
    project_adds = np.where(active_source, recomputed_k - 1, 0).astype(np.uint64)
    # 现有term builder会把相同gate的多个destination合成一次product multicast。
    # project-first强基线也必须保留这项能力，而不能按destination degree重复scale。
    project_scales = np.where(active_source, unique_gate_count, 0).astype(np.uint64)
    project_ops = project_adds + project_scales
    favorable = active_source & (project_ops < term_ops)
    return {
        "descriptors": int(descriptors),
        "active_sources": int(active_source.sum()),
        "active_k_lanes": int(recomputed_k.sum(dtype=np.uint64)),
        "active_source_unique_gates": int(project_scales.sum(dtype=np.uint64)),
        "active_source_edges": int(
            np.where(active_source, degree, 0).sum(dtype=np.uint64)
        ),
        "edge_lane_products": int(source_delivery_count.sum(dtype=np.uint64)),
        "source_quotient_product_rows": int(term_ops.sum(dtype=np.uint64)),
        "project_first_vector_adds": int(project_adds.sum(dtype=np.uint64)),
        "project_first_wide_gate_scales": int(project_scales.sum(dtype=np.uint64)),
        "project_first_total_vector_ops": int(project_ops.sum(dtype=np.uint64)),
        "project_first_favorable_sources": int(favorable.sum()),
        "project_first_favorable_active_source_ratio": (
            float(favorable.sum() / active_source.sum())
            if active_source.any()
            else 0.0
        ),
        "project_first_vs_source_quotient_op_change": (
            float(project_ops.sum(dtype=np.uint64) / term_ops.sum(dtype=np.uint64) - 1.0)
            if term_ops.any()
            else 0.0
        ),
        "project_first_weight_row_reduction_vs_source_quotient": (
            float(1.0 - recomputed_k.sum(dtype=np.uint64) / term_ops.sum(dtype=np.uint64))
            if term_ops.any()
            else 0.0
        ),
    }


def analyze_row_mode_oracle(
    k_bitmap: np.ndarray,
    incoming_gates: np.ndarray,
    valid_mask: np.ndarray,
    descriptor_group_offsets: np.ndarray,
    source_plane: np.ndarray,
    source_y: np.ndarray,
    source_gate_count: np.ndarray,
) -> dict[str, Any]:
    """在(group,time-plane,source-row)粒度比较DQFS与project-first。

    这里只提供显式成本代理和算术免费下界，不把不同位宽运算混成真实性能。
    """

    descriptors = k_bitmap.size
    if (
        incoming_gates.shape != (descriptors, ROLES)
        or valid_mask.shape != (descriptors,)
        or source_plane.shape != (descriptors,)
        or source_y.shape != (descriptors,)
        or source_gate_count.shape != (descriptors,)
        or descriptor_group_offsets.ndim != 1
        or descriptor_group_offsets[0] != 0
        or descriptor_group_offsets[-1] != descriptors
    ):
        raise ValueError("row-mode descriptor shape不符合合同")

    group_sizes = np.diff(descriptor_group_offsets)
    if np.any(group_sizes != TOKENS):
        raise ValueError("row-mode要求每group固定450个source descriptor")
    groups = group_sizes.size
    side = 15
    row_groups_per_group = 2 * side
    group_ids = np.repeat(np.arange(groups, dtype=np.int64), TOKENS)
    row_ids = (
        group_ids * row_groups_per_group
        + source_plane.astype(np.int64) * side
        + source_y.astype(np.int64)
    )
    row_count = groups * row_groups_per_group

    kpop = popcount_u64(k_bitmap).astype(np.int64)
    active = kpop > 0
    project_weight_reads = np.bincount(
        row_ids, weights=kpop, minlength=row_count
    ).astype(np.int64)
    project_vector_adds = np.bincount(
        row_ids,
        weights=np.where(active, kpop - 1, 0),
        minlength=row_count,
    ).astype(np.int64)
    project_wide_scales = np.bincount(
        row_ids,
        weights=np.where(active, source_gate_count, 0),
        minlength=row_count,
    ).astype(np.int64)

    roles = np.arange(ROLES, dtype=np.uint8)
    valid = ((valid_mask[:, None] >> roles[None, :]) & 1).astype(bool)
    active_gate = valid & incoming_gates.astype(bool)
    first_gate = active_gate.copy()
    for role in range(ROLES):
        for prior in range(role):
            first_gate[:, role] &= ~(
                active_gate[:, prior]
                & (incoming_gates[:, prior] == incoming_gates[:, role])
            )

    dqfs_keys: list[np.ndarray] = []
    for role in range(ROLES):
        gate_active = active & first_gate[:, role]
        if not np.any(gate_active):
            continue
        gate = incoming_gates[:, role].astype(np.int64)
        for lane in range(LANES):
            selected = gate_active & (((k_bitmap >> np.uint64(lane)) & 1) != 0)
            if np.any(selected):
                dqfs_keys.append(
                    (row_ids[selected] * LANES + lane) * 257 + gate[selected]
                )
    if not dqfs_keys:
        dqfs_products = np.zeros(row_count, dtype=np.int64)
    else:
        unique_keys = np.unique(np.concatenate(dqfs_keys))
        dqfs_products = np.bincount(
            unique_keys // (LANES * 257), minlength=row_count
        ).astype(np.int64)

    # 下界把project-first的13-bit add和13x9 wide multiply视为零成本，
    # 只比较weight-row read；这是偏向project-first的乐观上界，不是性能预测。
    weight_only_choose_project = project_weight_reads < dqfs_products
    weight_only_reads = np.where(
        weight_only_choose_project, project_weight_reads, dqfs_products
    )

    # 显式、可复现但不具物理含义的unit-vector-op灵敏度：
    # DQFS = read + 9x8 multiply；PF = read + 13-bit add + 13x9 multiply。
    dqfs_unit_cost = 2 * dqfs_products
    project_unit_cost = (
        project_weight_reads + project_vector_adds + project_wide_scales
    )
    unit_choose_project = project_unit_cost < dqfs_unit_cost
    unit_oracle_cost = np.where(
        unit_choose_project, project_unit_cost, dqfs_unit_cost
    )

    return {
        "granularity": "(ordered_group,time_plane,source_row)",
        "row_segments": int(row_count),
        "dqfs_weight_reads_and_narrow_products": int(dqfs_products.sum()),
        "project_first_weight_reads": int(project_weight_reads.sum()),
        "project_first_13b_vector_adds": int(project_vector_adds.sum()),
        "project_first_13x9_wide_products": int(project_wide_scales.sum()),
        "weight_only_free_compute_oracle": {
            "project_first_rows": int(weight_only_choose_project.sum()),
            "project_first_row_ratio": float(weight_only_choose_project.mean()),
            "selected_weight_reads": int(weight_only_reads.sum()),
            "read_reduction_vs_all_dqfs": (
                float(1.0 - weight_only_reads.sum() / dqfs_products.sum())
                if dqfs_products.any()
                else 0.0
            ),
            "boundary": "PF add/wide multiply均假定免费；仅为read reduction绝对乐观上界",
        },
        "unit_vector_op_sensitivity": {
            "definition": "DQFS=read+narrow_mul; PF=read+13b_add+wide_mul;各向量操作等权",
            "all_dqfs_cost": int(dqfs_unit_cost.sum()),
            "all_project_first_cost": int(project_unit_cost.sum()),
            "project_first_rows": int(unit_choose_project.sum()),
            "project_first_row_ratio": float(unit_choose_project.mean()),
            "oracle_cost": int(unit_oracle_cost.sum()),
            "oracle_reduction_vs_all_dqfs": (
                float(1.0 - unit_oracle_cost.sum() / dqfs_unit_cost.sum())
                if dqfs_unit_cost.any()
                else 0.0
            ),
            "boundary": "灵敏度代理，不是cycle、energy或PPA",
        },
    }


def validate_full_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "source_resident_active_k_lanes",
        "source_active_instances",
        "source_gate_lane_terms",
        "dqfs_row_value_product_computes",
        "naive_active_edge_products",
        "zero_gate_entries",
        "source_gate_cardinality_histogram",
    }
    if not keys.issubset(summary):
        raise ValueError("full profile summary 缺少 hoisting 强基线字段")
    scalar_keys = keys - {"source_gate_cardinality_histogram"}
    result: dict[str, Any] = {key: int(summary[key]) for key in scalar_keys}
    histogram = summary["source_gate_cardinality_histogram"]
    if not isinstance(histogram, list) or len(histogram) < 2:
        raise ValueError("full profile缺少source unique-gate histogram")
    result["source_gate_cardinality_histogram"] = [int(value) for value in histogram]
    if (
        any(result[key] < 0 for key in scalar_keys)
        or result["zero_gate_entries"] != 0
        or result["source_active_instances"] > result["source_resident_active_k_lanes"]
        or result["dqfs_row_value_product_computes"]
        > result["source_gate_lane_terms"]
    ):
        raise ValueError("full profile summary 数值关系不成立")
    return result


def full_profile_bounds(summary: dict[str, Any]) -> dict[str, Any]:
    k_lanes = summary["source_resident_active_k_lanes"]
    active_sources = summary["source_active_instances"]
    source_terms = summary["source_gate_lane_terms"]
    dqfs = summary["dqfs_row_value_product_computes"]
    project_adds = k_lanes - active_sources
    unique_gate_scales = sum(
        gate_count * sources
        for gate_count, sources in enumerate(
            summary["source_gate_cardinality_histogram"]
        )
    )
    project_ops = project_adds + unique_gate_scales
    return {
        "active_k_lanes": k_lanes,
        "active_sources": active_sources,
        "edge_lane_products": summary["naive_active_edge_products"],
        "source_quotient_product_rows": source_terms,
        "dqfs_row_value_product_rows": dqfs,
        "project_first_weight_rows": k_lanes,
        "project_first_vector_adds": project_adds,
        "project_first_unique_gate_scale_vectors": unique_gate_scales,
        "project_first_total_vector_ops": project_ops,
        "project_first_weight_row_change_vs_dqfs": k_lanes / dqfs - 1.0,
        "project_first_op_change_vs_dqfs": project_ops / dqfs - 1.0,
        "project_first_op_change_vs_source_quotient": project_ops / source_terms - 1.0,
        "source_quotient_reduction_vs_edge_lane": 1.0 - source_terms / summary["naive_active_edge_products"],
        "dqfs_reduction_vs_source_quotient": 1.0 - dqfs / source_terms,
    }


def render_markdown(report: dict[str, Any]) -> str:
    full = report["full_profile"]
    canonical = report["canonical_ordered_subset"]
    return "\n".join(
        [
            "# Local5 Source Projection Hoisting 强基线裁决",
            "",
            "> 证据：`[prof]+[模型]`；不是 RTL 周期、能量或 PPA。",
            "",
            "## 结论",
            "",
            "`source projection hoisting` 在当前 Local5 workload 下暂不晋级主架构 RTL。",
            "修正版保留了现有 unique-gate multicast。静态 full-profile 下，project-first",
            "仍比DQFS读取更多weight row，并额外引入13-bit向量加与13x9宽乘；row级",
            "hybrid只报告显式成本灵敏度，不作性能贡献主张。",
            "",
            "## Full Profile 强基线",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            f"| edge-lane product | {full['edge_lane_products']:,} |",
            f"| source-quotient product row | {full['source_quotient_product_rows']:,} |",
            f"| DQFS row-value product row | {full['dqfs_row_value_product_rows']:,} |",
            f"| project-first weight row | {full['project_first_weight_rows']:,} |",
            f"| project-first 13-bit vector add | {full['project_first_vector_adds']:,} |",
            f"| project-first 13x9 wide product | {full['project_first_unique_gate_scale_vectors']:,} |",
            f"| project-first arithmetic vectors | {full['project_first_total_vector_ops']:,} |",
            f"| project-first weight row 相对 DQFS | {full['project_first_weight_row_change_vs_dqfs']:+.3%} |",
            f"| project-first arithmetic vectors 相对 DQFS narrow product | {full['project_first_op_change_vs_dqfs']:+.3%} |",
            "",
            "上述最后一行只展示数量关系，不把13-bit add、13x9 wide product和9x8 narrow",
            "product等价成同一能量或周期单位。静态方案不晋级的直接依据是：PF的weight",
            "row比DQFS多19.8%，同时还需要额外宽算术；动态方案仍需同约束PPA后裁决。",
            "",
            "## Ordered Canonical 子集",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            f"| descriptor | {canonical['descriptors']:,} |",
            f"| active source | {canonical['active_sources']:,} |",
            f"| source-quotient vector op | {canonical['source_quotient_product_rows']:,} |",
            f"| project-first 13-bit add | {canonical['project_first_vector_adds']:,} |",
            f"| project-first 13x9 wide product | {canonical['project_first_wide_gate_scales']:,} |",
            f"| project-first 相对source-quotient有利source | {canonical['project_first_favorable_sources']:,} ({canonical['project_first_favorable_active_source_ratio']:.3%}) |",
            "",
            f"- row级weight-only免费算术oracle相对DQFS read减少：{report['row_mode_oracle']['weight_only_free_compute_oracle']['read_reduction_vs_all_dqfs']:.3%}。",
            f"- row级等权vector-op灵敏度oracle相对DQFS代理减少：{report['row_mode_oracle']['unit_vector_op_sensitivity']['oracle_reduction_vs_all_dqfs']:.3%}。",
            "- 前者把PF宽算术视为免费，后者把不同位宽向量操作等权；二者都不是cycle/energy。",
            "",
            "## 精确性与位宽",
            "",
            "当前 TCFM5 直接把 9-bit unsigned gate 与 INT8 weight 相乘后写入 Acc32，",
            "乘前没有右移、舍入或饱和。因此在足够中间位宽和相同最终 Acc32 合同下，",
            "`sum_l(g*K_l*W_l) = g*sum_l(K_l*W_l)` 可保持整数结果。",
            "但 project-first 需要 13-bit source projection 和冻结gate 0..256下最坏21-bit gate-scaled value，",
            "当前ordered payload实际gate最大值32时18-bit足够；",
            "而现有 term-first product 只需 16-bit。DQFS另有目录、term SRAM、重排和",
            "fallback成本，因此最终物理胜负仍需同约束综合，不能由位宽单独推出。",
            "",
            "## 边界",
            "",
            "- full-profile 结果来自既有 joint-head summary；ordered 子集独立重算 descriptor 合同。",
            "- 操作数、weight row和row级oracle不是 cycle、energy 或 ASIC PPA。",
            "- 该负结果不否决 FCSR-RX、DQFS 或 TCFM5；相反，它说明必须以 DQFS 为强基线。",
            "- 不将 source projection hoisting 或动态hybrid写成已实现贡献。",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_PROFILE / "ordered_term_manifest.json",
    )
    parser.add_argument(
        "--feature-json",
        type=Path,
        default=DEFAULT_PROFILE / "local5_hardware_features.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    feature_path = args.feature_json.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "et3_ordered_term_trace_v2":
        raise ValueError("ordered manifest schema 不一致")
    payload_path = manifest_path.parent / str(manifest["payload_file"])
    if sha256(payload_path) != manifest.get("payload_sha256"):
        raise ValueError("ordered payload SHA 不一致")
    groups = manifest.get("groups")
    if not isinstance(groups, list) or len(groups) != 13_800:
        raise ValueError("ordered group 数不等于 13800")

    with np.load(payload_path, allow_pickle=False) as payload:
        descriptor_offsets = payload["descriptor_group_offsets"]
        if (
            descriptor_offsets.dtype != np.int64
            or descriptor_offsets.shape != (len(groups) + 1,)
            or descriptor_offsets[0] != 0
            or np.any(np.diff(descriptor_offsets) != TOKENS)
        ):
            raise ValueError("descriptor group offset 不符合 450/source 合同")
        canonical = analyze_descriptors(
            payload["descriptor_k_bitmap"],
            payload["source_k_popcount"],
            payload["descriptor_incoming_gates"],
            payload["descriptor_valid_mask"],
            payload["source_gate_count"],
            payload["source_term_count"],
            payload["source_delivery_count"],
        )
        ordered_payload_gate_max = int(payload["descriptor_incoming_gates"].max())
        row_mode_oracle = analyze_row_mode_oracle(
            payload["descriptor_k_bitmap"],
            payload["descriptor_incoming_gates"],
            payload["descriptor_valid_mask"],
            descriptor_offsets,
            payload["descriptor_source_plane"],
            payload["descriptor_source_y"],
            payload["source_gate_count"],
        )

    full_summary = validate_full_summary(
        extract_top_level_object(feature_path, "summary")
    )
    full = full_profile_bounds(full_summary)
    report = {
        "schema": "local5_source_projection_hoisting_profile_v2",
        "status": "REJECT_STATIC_PRIMARY_RTL_DYNAMIC_ROW_HYBRID_NEEDS_PPA",
        "evidence": "[prof]+[模型]",
        "inputs": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "payload": str(payload_path),
            "payload_sha256": sha256(payload_path),
            "feature_json": str(feature_path),
            "feature_json_sha256": sha256(feature_path),
            "checkpoint_sha256": manifest.get("checkpoint_sha256"),
            "config_sha256": manifest.get("config_sha256"),
        },
        "full_profile": full,
        "canonical_ordered_subset": canonical,
        "row_mode_oracle": row_mode_oracle,
        "integer_contract": {
            "current_term_product_bits_required": 16,
            "project_first_source_sum_bits_required": 13,
            "project_first_scaled_bits_required": 21,
            "ordered_payload_gate_max": ordered_payload_gate_max,
            "ordered_payload_scaled_bits_required": 18,
            "exact_if": [
                "INT8 weight 与二值 K 先做足位宽整数和",
                "9-bit gate 相乘前无逐 lane 舍入或饱和",
                "最终使用相同 Acc32 加法合同",
            ],
        },
        "boundary": [
            "操作数和 weight-row 数不是 RTL cycle、energy 或 PPA。",
            "ordered canonical 子集不等于全部空间窗口。",
            "row级weight-only oracle把PF算术视为免费，只是read收益绝对乐观上界。",
            "row级unit-vector-op oracle把不同位宽操作等权，只是灵敏度代理。",
            "本结果否决静态全量source projection hoisting作为当前主RTL候选；动态hybrid需同约束PPA后裁决。",
            "本结果不否决既有DQFS/TCFM5。",
        ],
    }
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    script_snapshot = output / Path(__file__).name
    script_snapshot.write_bytes(Path(__file__).read_bytes())
    test_source = Path(__file__).with_name(
        "test_profile_local5_source_projection_hoisting_v2.py"
    )
    test_snapshot = output / test_source.name
    test_snapshot.write_bytes(test_source.read_bytes())
    test_run = subprocess.run(
        [sys.executable, str(test_snapshot)],
        cwd=output,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    test_receipt = output / "unit_tests.log"
    test_receipt.write_text(test_run.stdout, encoding="utf-8")
    if test_run.returncode != 0 or "Ran 5 tests" not in test_run.stdout:
        raise RuntimeError("source projection hoisting v2单元测试未通过")
    report["source_sha256"] = sha256(script_snapshot)
    report["test_source_sha256"] = sha256(test_snapshot)
    report["test_receipt_sha256"] = sha256(test_receipt)
    write_json(output / "report.json", report)
    (output / "report.md").write_text(render_markdown(report), encoding="utf-8")
    complete = {
        "schema": "local5_source_projection_hoisting_complete_v2",
        "status": report["status"],
        "report_sha256": sha256(output / "report.json"),
        "markdown_sha256": sha256(output / "report.md"),
        "source_sha256": sha256(script_snapshot),
        "test_source_sha256": sha256(test_snapshot),
        "test_receipt_sha256": sha256(test_receipt),
        "unit_tests": "5/5 PASS",
    }
    write_json(output / "complete.json", complete)
    print(json.dumps(complete, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
