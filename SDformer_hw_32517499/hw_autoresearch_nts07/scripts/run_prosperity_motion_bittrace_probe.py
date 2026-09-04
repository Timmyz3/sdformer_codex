#!/usr/bin/env python3
"""把 Motion 真实 K support bit trace 送入 Prosperity 官方 FC simulator。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from scripts.run_prosperity_official_probe import (
        ROOT,
        load_official_api,
        run_official_fc,
        sha256_file,
    )
except ModuleNotFoundError:
    from run_prosperity_official_probe import (
        ROOT,
        load_official_api,
        run_official_fc,
        sha256_file,
    )


DEFAULT_TRACE = ROOT / "results" / "h67_real_bit_trace_20260717"
DEFAULT_OUT = ROOT / "results" / "prosperity_motion_bittrace_probe_20260729"


def unpack_bits(payload: np.lib.npyio.NpzFile, prefix: str) -> np.ndarray:
    shape = tuple(int(value) for value in payload[f"{prefix}_shape"])
    count = int(np.prod(shape))
    bits = np.unpackbits(
        payload[f"{prefix}_bits_packed"],
        bitorder="little",
    )[:count]
    return bits.reshape(shape).astype(np.uint8)


def make_k_support_fc(record: dict, trace_root: Path):
    _, FC, _, _ = load_official_api()
    path = Path(record["file"])
    if not path.is_absolute():
        path = trace_root / path
    payload = np.load(path)
    k = unpack_bits(payload, "k")
    if k.ndim != 5 or k.shape[1] != 1:
        raise ValueError(f"不支持的 K shape: {k.shape}")
    time_steps, _, heads, sequence_length, lanes = k.shape
    input_dim = heads * lanes
    activation = (
        torch.from_numpy(k)
        .permute(1, 0, 3, 2, 4)
        .reshape(1, time_steps, sequence_length, input_dim)
        .contiguous()
    )
    weight_shape = tuple(int(value) for value in payload["projection_weight_int8"].shape)
    if weight_shape[0] != input_dim:
        raise ValueError(
            f"K input_dim={input_dim} 与 projection weight={weight_shape} 不一致"
        )
    operator = FC(
        f"{record['name'].replace('.', '_')}_k_support",
        input_dim=input_dim,
        output_dim=weight_shape[1],
        sequence_length=sequence_length,
        batch_size=1,
        time_steps=time_steps,
    )
    operator.activation_tensor.sparse_map = activation
    try:
        trace_file = str(path.relative_to(ROOT))
    except ValueError:
        trace_file = str(path)
    return operator, {
        "trace_file": trace_file,
        "trace_sha256": sha256_file(path),
        "k_shape": list(k.shape),
        "fc_activation_shape": list(activation.shape),
        "k_density": float(activation.float().mean().item()),
        "gate_nonzero": int(np.count_nonzero(payload["gate_q17"])),
        "gate_unique_codes": sorted(
            int(value) for value in np.unique(payload["gate_q17"])
        ),
        "weight_shape": list(weight_shape),
    }


def build_report(trace_root: Path, stages: tuple[int, ...]) -> dict:
    manifest = json.loads((trace_root / "manifest.json").read_text())
    records = {
        int(record["name"].split(".")[0][1:]): record
        for record in manifest["records"]
    }
    missing = sorted(set(stages) - records.keys())
    if missing:
        raise ValueError(f"bit trace 缺少 stage: {missing}")

    results = []
    for stage in stages:
        operator, source = make_k_support_fc(records[stage], trace_root)
        product = run_official_fc(operator, True)
        bit = run_official_fc(operator, False)
        results.append(
            {
                "stage": stage,
                "source": source,
                "official_product_sparsity": product.__dict__,
                "official_bit_sparsity": bit.__dict__,
                "official_cycle_speedup": (
                    bit.total_cycles / product.total_cycles
                ),
                "official_g_wgt_read_reduction": (
                    1.0 - product.g_wgt_reads / max(1, bit.g_wgt_reads)
                ),
            }
        )
    return {
        "schema": "prosperity_motion_k_support_probe_v1",
        "generated_date": "2026-07-29",
        "trace_manifest": str((trace_root / "manifest.json").relative_to(ROOT)),
        "trace_manifest_sha256": sha256_file(trace_root / "manifest.json"),
        "stages": results,
        "evidence_boundary": [
            "K bits 来自 Motion 真实网络 sample0/window0，不是密度塑形",
            "真实调用 Prosperity 官方 Simulator.run_fc CPU 路径",
            "输入仅为 binary K support；未表达 Q1.7 gate，因而不是 gated-K 投影等价基线",
            "仅一个 sample/window，不能外推 profile100 mean/p95/p99",
            "Prosperity 与本设计必须在未来相同逐元素 gated activation 合同下再比较",
        ],
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Motion 真实 K Support 的 Prosperity 官方仿真探针\n\n",
        "## 1. 结果\n\n",
        "| Stage | K density | gate code 数 | product cycles | bit-sparse cycles | 官方周期比 | g_wgt 读取降低 |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in report["stages"]:
        source = row["source"]
        product = row["official_product_sparsity"]
        bit = row["official_bit_sparsity"]
        lines.append(
            f"| S{row['stage']} | {source['k_density']:.5f} | "
            f"{len(source['gate_unique_codes'])} | "
            f"{product['total_cycles']} | {bit['total_cycles']} | "
            f"{row['official_cycle_speedup']:.3f}× | "
            f"{100*row['official_g_wgt_read_reduction']:.2f}% |\n"
        )
    lines.extend(
        [
            "\n## 2. 正确解释\n\n",
            "该实验回答的是：对同一真实 Motion K 的 0/1 support，Prosperity 官方"
            " product-sparsity 相对其 bit-sparsity 路径能减少多少周期。它没有表示"
            " Q1.7 gate，因此不能与 gate-aware term/NMF/DCTF 直接比较。\n\n",
            "## 3. 证据边界\n\n",
        ]
    )
    for item in report["evidence_boundary"]:
        lines.append(f"- {item}。\n")
    lines.extend(
        [
            "\n## 4. 复现\n\n",
            "```bash\n",
            "/opt/conda/envs/sdformerflow/bin/python "
            "scripts/run_prosperity_motion_bittrace_probe.py\n",
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
    stages = tuple(int(value) for value in args.stages.split(",") if value)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    report = build_report(args.trace_root, stages)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    write_markdown(report, args.out / "report.md")
    print(args.out / "report.md")
    for row in report["stages"]:
        print(
            f"S{row['stage']}",
            f"official_product_vs_bit={row['official_cycle_speedup']:.3f}x",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
