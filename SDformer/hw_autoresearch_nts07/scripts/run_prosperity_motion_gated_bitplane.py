#!/usr/bin/env python3
"""用 Prosperity 官方 FC CPU 路径评估 Motion 精确 gated-K bit-plane。"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

try:
    from scripts.run_prosperity_motion_bittrace_probe import unpack_bits
    from scripts.run_prosperity_official_probe import (
        ROOT,
        git_commit,
        load_official_api,
        run_official_fc,
        sha256_file,
    )
except ModuleNotFoundError:
    from run_prosperity_motion_bittrace_probe import unpack_bits
    from run_prosperity_official_probe import (
        ROOT,
        git_commit,
        load_official_api,
        run_official_fc,
        sha256_file,
    )


DEFAULT_TRACE = ROOT / "results" / "h67_real_bit_trace_20260717"
DEFAULT_OUT = ROOT / "results" / "prosperity_motion_gated_bitplane_20260802"

SUM_FIELDS = (
    "total_cycles",
    "compute_cycles",
    "preprocess_stall_cycles",
    "memory_stall_cycles",
    "num_ops",
    "dram_reads",
    "dram_writes",
    "g_act_reads",
    "g_wgt_reads",
    "g_psum_reads",
    "g_psum_writes",
)


def make_gated_activation(record: dict, trace_root: Path) -> tuple[np.ndarray, dict]:
    path = Path(record["file"])
    if not path.is_absolute():
        path = trace_root / path
    with np.load(path) as payload:
        k = unpack_bits(payload, "k")
        gate = np.asarray(payload["gate_q17"], dtype=np.uint16)
        weight_shape = tuple(
            int(value) for value in payload["projection_weight_int8"].shape
        )
    if k.ndim != 5 or k.shape[1] != 1:
        raise ValueError(f"不支持的K shape: {k.shape}")
    network_time_steps, _, heads, spatial_tokens, lanes = k.shape
    sequence_length = network_time_steps * spatial_tokens
    if gate.shape != (1, heads, sequence_length):
        raise ValueError(f"gate shape={gate.shape}与K不匹配")
    k_rows = k[:, 0].transpose(1, 0, 2, 3).reshape(
        heads, sequence_length, lanes
    )
    gated = k_rows * gate[0, :, :, None]
    activation = (
        gated.transpose(1, 0, 2)
        .reshape(1, sequence_length, heads * lanes)
        .astype(np.uint16)
    )
    if weight_shape[0] != activation.shape[-1]:
        raise ValueError("投影权重输入维与gated-K不一致")
    return activation, {
        "trace_file": str(path),
        "trace_sha256": sha256_file(path),
        "network_time_steps": network_time_steps,
        "prosperity_time_steps": 1,
        "heads": heads,
        "sequence_length": sequence_length,
        "lanes_per_head": lanes,
        "input_dim": heads * lanes,
        "output_dim": weight_shape[1],
        "gate_min": int(gate.min()),
        "gate_max": int(gate.max()),
        "gate_codes": [int(value) for value in np.unique(gate)],
    }


def split_active_bitplanes(activation: np.ndarray) -> list[tuple[int, np.ndarray]]:
    max_value = int(activation.max(initial=0))
    bit_width = max(1, max_value.bit_length())
    planes = []
    reconstructed = np.zeros_like(activation, dtype=np.uint16)
    for bit in range(bit_width):
        plane = ((activation >> bit) & 1).astype(np.uint8)
        if np.any(plane):
            planes.append((bit, plane))
            reconstructed += plane.astype(np.uint16) << bit
    if not np.array_equal(reconstructed, activation):
        raise AssertionError("bit-plane重建不等于原gated-K")
    return sorted(
        planes,
        key=lambda item: (-int(np.count_nonzero(item[1])), item[0]),
    )


def make_fc(name: str, plane: np.ndarray, output_dim: int):
    _, FC, _, _ = load_official_api()
    time_steps, sequence_length, input_dim = plane.shape
    operator = FC(
        name,
        input_dim=input_dim,
        output_dim=output_dim,
        sequence_length=sequence_length,
        batch_size=1,
        time_steps=time_steps,
    )
    operator.activation_tensor.sparse_map = (
        torch.from_numpy(plane).unsqueeze(0).contiguous()
    )
    return operator


def sum_runs(runs: list[dict]) -> dict:
    result = {field: 0 for field in SUM_FIELDS}
    for run in runs:
        for field in SUM_FIELDS:
            result[field] += int(run[field])
    return result


def evaluate_stage(record: dict, trace_root: Path) -> dict:
    activation, source = make_gated_activation(record, trace_root)
    planes = split_active_bitplanes(activation)

    rows = []
    for order, (bit, plane) in enumerate(planes):
        operator = make_fc(
            f"{record['name'].replace('.', '_')}_gate_bit{bit}",
            plane,
            source["output_dim"],
        )
        resident = order > 0
        product = asdict(
            run_official_fc(
                operator,
                True,
                weight_stored_in_buffer=resident,
            )
        )
        bit_sparse = asdict(
            run_official_fc(
                operator,
                False,
                weight_stored_in_buffer=resident,
            )
        )
        rows.append(
            {
                "execution_order": order,
                "bit": bit,
                "weight_resident": resident,
                "ones": int(np.count_nonzero(plane)),
                "density": float(np.mean(plane)),
                "official_product_sparsity": product,
                "official_bit_sparsity": bit_sparse,
            }
        )

    product_total = sum_runs(
        [row["official_product_sparsity"] for row in rows]
    )
    bit_total = sum_runs(
        [row["official_bit_sparsity"] for row in rows]
    )
    matrix_rows = source["prosperity_time_steps"] * source["sequence_length"]
    merge_cycles = (
        max(0, len(rows) - 1)
        * matrix_rows
        * math.ceil(source["output_dim"] / 128)
    )
    return {
        "stage": int(record["name"].split(".")[0][1:]),
        "source": source,
        "active_bitplanes": len(rows),
        "bitplane_execution_order": [row["bit"] for row in rows],
        "planes": rows,
        "official_product_sparsity_total": product_total,
        "official_bit_sparsity_total": bit_total,
        "official_product_vs_bit_speedup": (
            bit_total["total_cycles"] / product_total["total_cycles"]
            if product_total["total_cycles"]
            else None
        ),
        "shift_accumulate_cycles_unmodeled_lower_bound": merge_cycles,
    }


def build_report(trace_root: Path, stages: tuple[int, ...]) -> dict:
    manifest_path = trace_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = {
        int(record["name"].split(".")[0][1:]): record
        for record in manifest["records"]
    }
    missing = sorted(set(stages) - records.keys())
    if missing:
        raise ValueError(f"trace缺少stage: {missing}")
    rows = [evaluate_stage(records[stage], trace_root) for stage in stages]
    product_total = sum(
        row["official_product_sparsity_total"]["total_cycles"] for row in rows
    )
    bit_total = sum(
        row["official_bit_sparsity_total"]["total_cycles"] for row in rows
    )
    return {
        "schema": "prosperity_motion_gated_bitplane_v1",
        "generated_date": "2026-08-02",
        "trace_manifest": str(manifest_path),
        "trace_manifest_sha256": sha256_file(manifest_path),
        "prosperity_repo": "https://github.com/dubcyfor3/Prosperity",
        "prosperity_commit": git_commit(ROOT / "third_party" / "Prosperity"),
        "method": {
            "numeric_equivalence": "gated-K=sum(bitplane_b*2^b)，逐元素重建断言通过",
            "official_path": "每个非零bit-plane真实调用官方Simulator.run_fc CPU路径",
            "favorable_assumptions": [
                "跳过全零bit-plane",
                "按密度从高到低执行",
                "第一plane后权重常驻片上",
                "官方总周期不加入跨plane移位累加成本",
            ],
        },
        "stages": rows,
        "totals": {
            "official_product_sparsity_cycles": product_total,
            "official_bit_sparsity_cycles": bit_total,
            "official_product_vs_bit_speedup": bit_total / product_total,
            "shift_accumulate_cycles_unmodeled_lower_bound": sum(
                row["shift_accumulate_cycles_unmodeled_lower_bound"]
                for row in rows
            ),
        },
        "evidence_boundary": [
            "输入是Motion sample0/window0真实K与Q1.7 gate，不是密度塑形",
            "bit-plane分解数值精确，但Prosperity官方模拟器不建模plane间移位累加",
            "全零gated-K stage按零个FC plane计，偏置写出与最终输出成本未建模，进一步有利于Prosperity",
            "Prosperity为128输出lane配置，本设计为三bank共96个32-lane产品通路，不能直接宣称面积公平",
            "只有一个sample/window，不能外推数据集mean/p95/p99",
            "本结果是强基线周期证据，不是Prosperity PPA复现",
        ],
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Motion 精确 Gated-K 的 Prosperity 官方 Bit-Plane 评估\n\n",
        "## 1. 结论\n\n",
        "本评估把真实 `gate_code × K` 逐元素分解为二值 bit-plane，每个非零 plane "
        "真实调用 Prosperity 官方 `Simulator.run_fc` CPU 路径。分解前后逐元素重建一致。"
        "为形成有利于对手的强基线，权重在首个 plane 后保持驻留，且官方周期不计跨 plane "
        "移位累加成本。\n\n",
        "| Stage | gate code | active planes | plane顺序 | product cycles | bit-sparse cycles | 官方内部加速 | 未计merge下界 |\n",
        "|---|---|---:|---|---:|---:|---:|---:|\n",
    ]
    for row in report["stages"]:
        p = row["official_product_sparsity_total"]
        b = row["official_bit_sparsity_total"]
        speedup = row["official_product_vs_bit_speedup"]
        speedup_text = f"{speedup:.3f}x" if speedup is not None else "N/A"
        lines.append(
            f"| S{row['stage']} | {row['source']['gate_codes']} | "
            f"{row['active_bitplanes']} | {row['bitplane_execution_order']} | "
            f"{p['total_cycles']} | {b['total_cycles']} | "
            f"{speedup_text} | "
            f"{row['shift_accumulate_cycles_unmodeled_lower_bound']} |\n"
        )
    total = report["totals"]
    lines.extend(
        [
            f"| **总计** | - | - | - | "
            f"{total['official_product_sparsity_cycles']} | "
            f"{total['official_bit_sparsity_cycles']} | "
            f"{total['official_product_vs_bit_speedup']:.3f}x | "
            f"{total['shift_accumulate_cycles_unmodeled_lower_bound']} |\n\n",
            "## 2. 证据边界\n\n",
        ]
    )
    lines.extend(f"- {item}。\n" for item in report["evidence_boundary"])
    lines.extend(
        [
            "\n## 3. 复现\n\n",
            "```bash\n",
            "/opt/conda/envs/sdformerflow/bin/python "
            "scripts/run_prosperity_motion_gated_bitplane.py\n",
            "```\n",
        ]
    )
    path.write_text("".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stages", default="0,1,2,3")
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    stages = tuple(int(value) for value in args.stages.split(",") if value)
    report = build_report(args.trace_root, stages)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    write_markdown(report, args.out / "report.md")
    print(args.out / "report.md")
    print(json.dumps(report["totals"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
