#!/usr/bin/env python3
"""用 Local5 post-G0 descriptor 评估 active-source TCFM-5 数据流。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROLE_DY = np.asarray([0, 1, -1, 0, 0], dtype=np.int64)
ROLE_DX = np.asarray([0, 0, 0, 1, -1], dtype=np.int64)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bound_file(path: Path, expected_sha256: str, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label}不存在: {resolved}")
    if file_sha256(resolved) != expected_sha256:
        raise ValueError(f"{label} SHA绑定失效: {resolved}")
    return resolved


def percentile_summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": int(values.max()),
    }


def popcount_u64(values: np.ndarray) -> np.ndarray:
    return np.fromiter(
        (int(value).bit_count() for value in values),
        count=len(values),
        dtype=np.uint8,
    )


def analyze_descriptor_chunk(
    gates: np.ndarray,
    valid_mask: np.ndarray,
    k_bitmap: np.ndarray,
    plane: np.ndarray,
    source_y: np.ndarray,
    source_x: np.ndarray,
    *,
    height: int,
    width: int,
) -> dict[str, np.ndarray]:
    """返回逐 source 的 product、update 与各 bank 映射服务周期。"""

    if gates.ndim != 2 or gates.shape[1] != 5:
        raise ValueError("incoming gate 必须为 [sources,5]")
    count = len(gates)
    for values in (valid_mask, k_bitmap, plane, source_y, source_x):
        if len(values) != count:
            raise ValueError("descriptor 数组长度不一致")

    lane_count = popcount_u64(k_bitmap).astype(np.int32)
    role_valid = np.stack(
        [
            (((valid_mask >> role) & 1) != 0) & (gates[:, role] != 0)
            for role in range(5)
        ],
        axis=1,
    )
    unique_gate = np.zeros((count, 5), dtype=bool)
    for role in range(5):
        unique_gate[:, role] = role_valid[:, role]
        for previous in range(role):
            unique_gate[:, role] &= ~(
                role_valid[:, previous]
                & (gates[:, previous] == gates[:, role])
            )

    destination_y = source_y[:, None].astype(np.int64) + ROLE_DY[None, :]
    destination_x = source_x[:, None].astype(np.int64) + ROLE_DX[None, :]
    geometric_valid = (
        (destination_y >= 0)
        & (destination_y < height)
        & (destination_x >= 0)
        & (destination_x < width)
    )
    if np.any(role_valid & ~geometric_valid):
        raise ValueError("valid mask 包含越界 destination")

    destination_id = (
        plane[:, None].astype(np.int64) * height * width
        + destination_y * width
        + destination_x
    )
    bank_cycles = {
        "parity2": np.zeros(count, dtype=np.int32),
        "linear3": np.zeros(count, dtype=np.int32),
        "linear5": np.zeros(count, dtype=np.int32),
        "tcfm5": np.zeros(count, dtype=np.int32),
    }
    bank_ids = {
        "parity2": np.remainder(destination_id, 2),
        "linear3": np.remainder(destination_id, 3),
        "linear5": np.remainder(destination_id, 5),
        "tcfm5": np.remainder(destination_x + 2 * destination_y, 5),
    }
    bank_counts = {"parity2": 2, "linear3": 3, "linear5": 5, "tcfm5": 5}

    for representative in range(5):
        same_gate_roles = (
            role_valid
            & (gates == gates[:, representative : representative + 1])
            & unique_gate[:, representative : representative + 1]
        )
        for name, ids in bank_ids.items():
            loads = np.stack(
                [
                    (((ids == bank) & same_gate_roles).sum(axis=1))
                    for bank in range(bank_counts[name])
                ],
                axis=1,
            )
            bank_cycles[name] += loads.max(axis=1).astype(np.int32)

    unique_count = unique_gate.sum(axis=1).astype(np.int32)
    valid_role_count = role_valid.sum(axis=1).astype(np.int32)
    product_terms = lane_count * unique_count
    result = {
        "active_sources": ((lane_count > 0) & (unique_count > 0)).astype(np.int32),
        "product_terms": product_terms,
        "destination_updates": lane_count * valid_role_count,
    }
    for name, cycles in bank_cycles.items():
        result[f"{name}_cycles"] = lane_count * cycles

    if np.any(result["tcfm5_cycles"] != product_terms):
        raise AssertionError("TCFM-5 未实现每个 gate-lane term 单拍无冲突")
    return result


def reduce_groups(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    if offsets[0] != 0 or offsets[-1] != len(values):
        raise ValueError("descriptor group offsets 与 payload 不一致")
    return np.add.reduceat(values, offsets[:-1]).astype(np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/local5_fullres_postg0_qfsa_profile100_20260730"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/local5_active_tcfm5_postg0_20260803"),
    )
    parser.add_argument("--bitmap-width", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    args = parser.parse_args()
    if args.bitmap_width <= 0 or args.chunk_size <= 0:
        raise SystemExit("bitmap-width 与 chunk-size 必须为正数")

    manifest_path = args.input_dir / "ordered_term_manifest.json"
    payload_path = args.input_dir / "ordered_term_items.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "et3_ordered_term_trace_v2":
        raise ValueError("本分析只接受 post-G0 v2 ordered trace")
    qualification = manifest.get("qualification", {})
    if not qualification.get("qualified"):
        raise ValueError("ordered trace 未通过 qualification")
    verify_bound_file(payload_path, manifest.get("payload_sha256", ""), "ordered trace payload")
    checkpoint_path = verify_bound_file(
        Path(manifest["checkpoint"]), manifest["checkpoint_sha256"], "checkpoint"
    )
    config_path = verify_bound_file(
        Path(manifest["config"]), manifest["config_sha256"], "config"
    )
    selection_plan = manifest.get("sampling", {}).get("selection_plan")
    selection_plan_sha256 = manifest.get("sampling", {}).get("selection_plan_sha256")
    selection_plan_path = None
    if selection_plan or selection_plan_sha256:
        if not selection_plan or not selection_plan_sha256:
            raise ValueError("selection plan路径与SHA必须同时存在")
        selection_plan_path = verify_bound_file(
            Path(selection_plan), selection_plan_sha256, "selection plan"
        )

    payload = np.load(payload_path, mmap_mode="r")
    offsets = payload["descriptor_group_offsets"]
    descriptor_count = int(offsets[-1])
    metric_names = (
        "active_sources",
        "product_terms",
        "destination_updates",
        "parity2_cycles",
        "linear3_cycles",
        "linear5_cycles",
        "tcfm5_cycles",
    )
    descriptor_metrics = {
        name: np.zeros(descriptor_count, dtype=np.int32) for name in metric_names
    }
    for start in range(0, descriptor_count, args.chunk_size):
        stop = min(start + args.chunk_size, descriptor_count)
        chunk = analyze_descriptor_chunk(
            np.asarray(payload["descriptor_incoming_gates"][start:stop]),
            np.asarray(payload["descriptor_valid_mask"][start:stop]),
            np.asarray(payload["descriptor_k_bitmap"][start:stop]),
            np.asarray(payload["descriptor_source_plane"][start:stop]),
            np.asarray(payload["descriptor_source_y"][start:stop]),
            np.asarray(payload["descriptor_source_x"][start:stop]),
            height=15,
            width=15,
        )
        for name in metric_names:
            descriptor_metrics[name][start:stop] = chunk[name]

    group_metrics = {
        name: reduce_groups(values, offsets)
        for name, values in descriptor_metrics.items()
    }
    source_slots = np.diff(offsets).astype(np.int64)
    if np.any(source_slots != 450):
        raise ValueError("Local5 fullres 每组必须恰有 450 个 source descriptor")
    bitmap_scan_cycles = np.ceil(source_slots / args.bitmap_width).astype(np.int64)
    compact_issue = group_metrics["active_sources"]

    for backend in ("parity2", "linear3", "linear5", "tcfm5"):
        service = group_metrics[f"{backend}_cycles"]
        group_metrics[f"{backend}_fixed_scan_pipeline"] = np.maximum(
            source_slots, service
        )
        group_metrics[f"{backend}_active_pipeline"] = (
            bitmap_scan_cycles + np.maximum(compact_issue, service)
        )

    linear5_active = group_metrics["linear5_active_pipeline"]
    tcfm5_active = group_metrics["tcfm5_active_pipeline"]
    groups = manifest["groups"]
    stage_results = []
    for stage in range(4):
        mask = np.asarray([group["stage"] == stage for group in groups], dtype=bool)
        baseline_cycles = int(linear5_active[mask].sum())
        candidate_cycles = int(tcfm5_active[mask].sum())
        stage_results.append(
            {
                "stage": stage,
                "groups": int(mask.sum()),
                "linear5_active_cycles": baseline_cycles,
                "tcfm5_active_cycles": candidate_cycles,
                "speedup": baseline_cycles / candidate_cycles,
            }
        )

    descriptor_bits = 32 + 5 * 9 + 5
    full_descriptor_bits = descriptor_count * descriptor_bits
    compact_descriptor_bits = (
        int(group_metrics["active_sources"].sum()) * descriptor_bits
        + len(groups) * 450
    )
    topology_speedup = (
        int(group_metrics["linear5_cycles"].sum())
        / int(group_metrics["tcfm5_cycles"].sum())
    )
    result: dict[str, Any] = {
        "schema": "local5_active_tcfm5_postg0_v2",
        "status": "PROFILE_MODEL_COMPLETE",
        "evidence": "[prof]+[exact-port-model]",
        "input": {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": file_sha256(manifest_path),
            "payload": str(payload_path.resolve()),
            "payload_sha256": file_sha256(payload_path),
            "groups": len(groups),
            "descriptors": descriptor_count,
            "samples": qualification["processed_samples"],
            "resolution": manifest["resolution"],
            "checkpoint": manifest.get("checkpoint"),
            "checkpoint_sha256": manifest.get("checkpoint_sha256"),
            "config": manifest.get("config"),
            "config_sha256": manifest.get("config_sha256"),
            "sampling": manifest.get("sampling"),
        },
        "contract": {
            "roles": ["self", "down_destination", "up_destination", "right_destination", "left_destination"],
            "gate_zero_skipped": True,
            "same_source_gate_lane_product_reuse": True,
            "accumulator_ports": "每 bank 每拍一个 destination update",
            "active_source_bitmap_width": args.bitmap_width,
            "active_source_bitmap_scan_cycles_per_group": int(bitmap_scan_cycles[0]),
            "compact_issue_model": "bitmap scan串行；descriptor producer与backend取max重叠",
        },
        "execution_receipt": {
            "producer": str(Path(__file__).resolve()),
            "producer_sha256": file_sha256(Path(__file__).resolve()),
            "command": {
                "input_dir": str(args.input_dir.resolve()),
                "output_dir": str(args.output_dir.resolve()),
                "bitmap_width": args.bitmap_width,
                "chunk_size": args.chunk_size,
            },
            "python": sys.version.splitlines()[0],
            "python_executable": str(Path(sys.executable).resolve()),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "totals": {
            name: int(values.sum()) for name, values in group_metrics.items()
        },
        "group_statistics": {
            name: percentile_summary(values) for name, values in group_metrics.items()
        },
        "stage_results": stage_results,
        "derived": {
            "active_source_rate": int(group_metrics["active_sources"].sum()) / descriptor_count,
            "fixed_to_compact_descriptor_bit_reduction": 1 - compact_descriptor_bits / full_descriptor_bits,
            "full_descriptor_bits": full_descriptor_bits,
            "compact_descriptor_bits": compact_descriptor_bits,
            "single_bank_to_tcfm5_raw_speedup": int(group_metrics["destination_updates"].sum()) / int(group_metrics["tcfm5_cycles"].sum()),
            "linear5_to_tcfm5_raw_speedup": topology_speedup,
            "linear5_active_to_tcfm5_active_speedup": int(linear5_active.sum()) / int(tcfm5_active.sum()),
            "linear5_fixed_to_tcfm5_active_speedup": int(group_metrics["linear5_fixed_scan_pipeline"].sum()) / int(tcfm5_active.sum()),
            "tcfm5_fixed_to_tcfm5_active_speedup": int(group_metrics["tcfm5_fixed_scan_pipeline"].sum()) / int(tcfm5_active.sum()),
            "tcfm5_fixed_to_zero_scan_tcfm5_speedup": int(group_metrics["tcfm5_fixed_scan_pipeline"].sum()) / int(group_metrics["tcfm5_cycles"].sum()),
        },
        "limits": [
            "周期仅覆盖source descriptor到Acc update，不含Local5 score、Shiftmax、bias/final和full encoder。",
            "active-source bitmap必须在destination-major score阶段并行生成；若在关系转置后才生成，不能省掉450次关系读取。",
            "五色active bitmap、五bank SRAM宏面积、时钟、块间布线和功耗尚未综合。",
            "ordered trace每个block/sample只预先选定一个window并覆盖其全部head；不是完整帧全部window或总体周期。",
            "相同五bank基线只改变bank映射，product term和单写端口合同保持一致。",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    totals = result["totals"]
    derived = result["derived"]
    lines = [
        "# Local5 Active-Source TCFM-5 Post-G0评估",
        "",
        "## 结论",
        "",
        f"在`{len(groups)}`个真实post-G0 group上，只有"
        f"`{derived['active_source_rate']:.2%}`的source产生非零投影工作。"
        "在相同五个单写Acc bank、相同gate-lane product term下，拓扑着色TCFM-5"
        f"相对普通linear-id mod-5映射获得`{derived['linear5_to_tcfm5_raw_speedup']:.3f}x`"
        "product-delivery周期收益。加入32-bit active-source bitmap扫描后，"
        f"相对同样active前端的linear-5仍为`{derived['linear5_active_to_tcfm5_active_speedup']:.3f}x`。",
        f"相对当前固定450-source扫描的TCFM-5，32-bit bitmap扫描模型为"
        f"`{derived['tcfm5_fixed_to_tcfm5_active_speedup']:.3f}x`；若采用五个bank-local"
        f"zero skipper直接发射active source，则上界为"
        f"`{derived['tcfm5_fixed_to_zero_scan_tcfm5_speedup']:.3f}x`。",
        "",
        "证据等级为 **[prof]+[exact-port-model]**；这不是RTL端到端加速或PPA。",
        "",
        "## 总量结果",
        "",
        "| 指标 | 总量 |",
        "|---|---:|",
        f"| source descriptor | {descriptor_count} |",
        f"| active source | {totals['active_sources']} |",
        f"| gate-lane product term | {totals['product_terms']} |",
        f"| destination update | {totals['destination_updates']} |",
        f"| parity-2 delivery cycles | {totals['parity2_cycles']} |",
        f"| linear-3 delivery cycles | {totals['linear3_cycles']} |",
        f"| linear-5 delivery cycles | {totals['linear5_cycles']} |",
        f"| TCFM-5 delivery cycles | {totals['tcfm5_cycles']} |",
        "",
        "## 同端口强基线",
        "",
        "| 对照 | 加速比 | 含义 |",
        "|---|---:|---|",
        f"| single-bank / TCFM-5 | {derived['single_bank_to_tcfm5_raw_speedup']:.3f}x | 五角色组播并行上界 |",
        f"| linear-5 / TCFM-5 | {derived['linear5_to_tcfm5_raw_speedup']:.3f}x | 相同五bank、只改变拓扑映射 |",
        f"| active linear-5 / active TCFM-5 | {derived['linear5_active_to_tcfm5_active_speedup']:.3f}x | 两侧均计15拍bitmap scan |",
        f"| fixed-scan linear-5 / active TCFM-5 | {derived['linear5_fixed_to_tcfm5_active_speedup']:.3f}x | active-source压缩与拓扑映射联合 |",
        f"| fixed-scan TCFM-5 / active TCFM-5 | {derived['tcfm5_fixed_to_tcfm5_active_speedup']:.3f}x | 只计active-source压缩 |",
        f"| fixed-scan TCFM-5 / zero-scan TCFM-5 | {derived['tcfm5_fixed_to_zero_scan_tcfm5_speedup']:.3f}x | 五bank本地zero skipper理想上界 |",
        "",
        "## Stage结果",
        "",
        "| Stage | group | active linear-5 | active TCFM-5 | 加速 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in stage_results:
        lines.append(
            f"| S{row['stage']} | {row['groups']} | {row['linear5_active_cycles']} | "
            f"{row['tcfm5_active_cycles']} | {row['speedup']:.3f}x |"
        )
    lines += [
        "",
        "## 元数据流量",
        "",
        f"固定流按每source `82 bit`计为`{full_descriptor_bits}` bit；active-source模式"
        f"计450-bit bitmap加活动descriptor，为`{compact_descriptor_bits}` bit，"
        f"降低`{derived['fixed_to_compact_descriptor_bit_reduction']:.2%}`。该值是链路bit"
        "计数，不是SRAM面积或能耗。",
        "",
        "## 证据边界",
        "",
    ]
    lines.extend(f"- {item}" for item in result["limits"])
    lines += [
        "",
        "## 架构晋级判断",
        "",
        "该模型只用于提出`active-source quotient + TCFM-5`的RTL候选与完整cohort趋势。"
        "RTL结果必须在独立报告中按相同SRAM wrapper、相同五bank和相同反压合同给出，"
        "不能由本模型自动晋级为RTL或PPA结论。",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    source_dir = args.output_dir / "source"
    source_dir.mkdir(exist_ok=True)
    producer_snapshot = source_dir / Path(__file__).name
    shutil.copyfile(Path(__file__).resolve(), producer_snapshot)
    bound_inputs = {
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": file_sha256(manifest_path),
        },
        "payload": {
            "path": str(payload_path.resolve()),
            "sha256": file_sha256(payload_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
        },
        "config": {
            "path": str(config_path),
            "sha256": file_sha256(config_path),
        },
    }
    if selection_plan_path is not None:
        bound_inputs["selection_plan"] = {
            "path": str(selection_plan_path),
            "sha256": file_sha256(selection_plan_path),
        }
    package_files = {
        str(path.relative_to(args.output_dir)): file_sha256(path)
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file() and path.name != "complete.json"
    }
    complete = {
        "schema": "local5_active_tcfm5_exact_port_model_package_v1",
        "status": "SEALED",
        "evidence": "[prof]+[exact-port-model]",
        "bound_inputs": bound_inputs,
        "package_files": package_files,
    }
    (args.output_dir / "complete.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
