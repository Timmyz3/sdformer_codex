#!/usr/bin/env python3
"""把 Local5 post-G0 ordered term 重建为 gated bit-plane 并运行 Prosperity。"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

try:
    from scripts.run_prosperity_motion_gated_bitplane import (
        SUM_FIELDS,
        make_fc,
        split_active_bitplanes,
        sum_runs,
    )
    from scripts.run_prosperity_official_probe import (
        ROOT,
        git_commit,
        run_official_fc,
        sha256_file,
    )
except ModuleNotFoundError:
    from run_prosperity_motion_gated_bitplane import (
        SUM_FIELDS,
        make_fc,
        split_active_bitplanes,
        sum_runs,
    )
    from run_prosperity_official_probe import (
        ROOT,
        git_commit,
        run_official_fc,
        sha256_file,
    )


DEFAULT_TRACE = (
    ROOT / "results" / "local5_fullres_postg0_qfsa_profile100_20260730"
)
DEFAULT_OUT = ROOT / "results" / "prosperity_local5_gated_bitplane_20260802"


def reconstruct_group_activation(
    *,
    tokens: int,
    lanes: int,
    gate: np.ndarray,
    lane: np.ndarray,
    multiplicity: np.ndarray,
    destination: np.ndarray,
) -> np.ndarray:
    lengths = {len(gate), len(lane), len(multiplicity), len(destination)}
    if len(lengths) != 1:
        raise ValueError("Local5 term数组长度不一致")
    if len(gate) == 0:
        return np.zeros((tokens, lanes), dtype=np.uint16)
    if np.any(destination >= tokens) or np.any(lane >= lanes):
        raise ValueError("Local5 term destination/lane越界")
    values = gate.astype(np.uint32) * multiplicity.astype(np.uint32)
    activation = np.zeros((tokens, lanes), dtype=np.uint32)
    np.add.at(
        activation,
        (destination.astype(np.int64), lane.astype(np.int64)),
        values,
    )
    if int(activation.max(initial=0)) > np.iinfo(np.uint16).max:
        raise OverflowError("Local5 gated activation超过uint16")
    return activation.astype(np.uint16)


def load_block_head_activations(trace_root: Path) -> tuple[dict, dict]:
    manifest_path = trace_root / "ordered_term_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "et3_ordered_term_trace_v2":
        raise ValueError("不支持的Local5 ordered trace schema")
    if manifest.get("evidence_level") != "post_g0":
        raise ValueError("Prosperity正式基线要求post_g0 trace")
    qualification = manifest.get("qualification", {})
    if not qualification.get("qualified", False):
        raise ValueError("Local5 post_g0 qualification未通过")
    payload_path = trace_root / manifest["payload_file"]
    if sha256_file(payload_path) != manifest["payload_sha256"]:
        raise ValueError("Local5 ordered payload SHA256不匹配")

    with np.load(payload_path, allow_pickle=False) as payload:
        offsets = np.asarray(payload["group_offsets"], dtype=np.int64)
        gates = np.asarray(payload["item_gate_code"], dtype=np.uint16)
        lanes = np.asarray(payload["item_lane_id"], dtype=np.uint16)
        multiplicities = np.asarray(
            payload["item_multiplicity"], dtype=np.uint8
        )
        destinations = np.asarray(
            payload["item_destination"], dtype=np.uint16
        )
    groups = manifest["groups"]
    if len(offsets) != len(groups) + 1 or offsets[0] != 0:
        raise ValueError("group_offsets与manifest groups不一致")
    if offsets[-1] != len(gates):
        raise ValueError("group_offsets尾部与item数量不一致")

    by_key: dict[tuple[int, int, int], list[np.ndarray]] = defaultdict(list)
    metadata: dict[tuple[int, int, int], dict] = {}
    for index, group in enumerate(groups):
        start, end = int(offsets[index]), int(offsets[index + 1])
        key = (int(group["stage"]), int(group["block"]), int(group["head"]))
        activation = reconstruct_group_activation(
            tokens=int(group["tokens"]),
            lanes=int(group["lanes"]),
            gate=gates[start:end],
            lane=lanes[start:end],
            multiplicity=multiplicities[start:end],
            destination=destinations[start:end],
        )
        by_key[key].append(activation)
        current = {
            "stage": key[0],
            "block": key[1],
            "head": key[2],
            "tokens": int(group["tokens"]),
            "lanes": int(group["lanes"]),
            "heads": int(group["heads"]),
        }
        if key in metadata and metadata[key] != current:
            raise ValueError(f"同一stage/block/head元数据不一致: {key}")
        metadata[key] = current

    concatenated = {
        key: np.concatenate(value, axis=0)
        for key, value in by_key.items()
    }
    source = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "payload": str(payload_path),
        "payload_sha256": sha256_file(payload_path),
        "sampled_groups": len(groups),
        "block_head_keys": len(concatenated),
        "sampling": manifest["sampling"],
    }
    return source, {key: (metadata[key], value) for key, value in concatenated.items()}


def evaluate_key(key: tuple[int, int, int], metadata: dict, activation: np.ndarray) -> dict:
    planes = split_active_bitplanes(activation[None, ...])
    rows = []
    output_dim = metadata["heads"] * metadata["lanes"]
    for order, (bit, plane) in enumerate(planes):
        operator = make_fc(
            f"local5_s{key[0]}_b{key[1]}_h{key[2]}_bit{bit}",
            plane,
            output_dim,
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
    product = sum_runs([row["official_product_sparsity"] for row in rows])
    bit_sparse = sum_runs([row["official_bit_sparsity"] for row in rows])
    matrix_rows = int(activation.shape[0])
    return {
        **metadata,
        "sampled_group_count": matrix_rows // metadata["tokens"],
        "matrix_rows": matrix_rows,
        "activation_max": int(activation.max(initial=0)),
        "activation_nonzero_ratio": float(np.count_nonzero(activation) / activation.size),
        "active_bitplanes": len(rows),
        "bitplane_execution_order": [row["bit"] for row in rows],
        "planes": rows,
        "official_product_sparsity_total": product,
        "official_bit_sparsity_total": bit_sparse,
        "official_product_vs_bit_speedup": (
            bit_sparse["total_cycles"] / product["total_cycles"]
            if product["total_cycles"]
            else None
        ),
        "shift_accumulate_cycles_unmodeled_lower_bound": (
            max(0, len(rows) - 1)
            * matrix_rows
            * math.ceil(output_dim / 128)
        ),
    }


def aggregate(rows: list[dict]) -> dict:
    product = {field: 0 for field in SUM_FIELDS}
    bit_sparse = {field: 0 for field in SUM_FIELDS}
    merge = 0
    for row in rows:
        for field in SUM_FIELDS:
            product[field] += row["official_product_sparsity_total"][field]
            bit_sparse[field] += row["official_bit_sparsity_total"][field]
        merge += row["shift_accumulate_cycles_unmodeled_lower_bound"]
    return {
        "official_product_sparsity": product,
        "official_bit_sparsity": bit_sparse,
        "official_product_vs_bit_speedup": (
            bit_sparse["total_cycles"] / product["total_cycles"]
            if product["total_cycles"]
            else None
        ),
        "shift_accumulate_cycles_unmodeled_lower_bound": merge,
    }


def build_report(trace_root: Path) -> dict:
    source, activations = load_block_head_activations(trace_root)
    rows = [
        evaluate_key(key, metadata, activation)
        for key, (metadata, activation) in sorted(activations.items())
    ]
    stage_rows = {}
    for stage in sorted({row["stage"] for row in rows}):
        stage_rows[str(stage)] = aggregate(
            [row for row in rows if row["stage"] == stage]
        )
    return {
        "schema": "prosperity_local5_gated_bitplane_v1",
        "generated_date": "2026-08-02",
        "source": source,
        "prosperity_repo": "https://github.com/dubcyfor3/Prosperity",
        "prosperity_commit": git_commit(ROOT / "third_party" / "Prosperity"),
        "method": {
            "numeric_equivalence": "activation[dest,lane]+=gate*multiplicity，再做exact bit-plane",
            "weight_scope": "每个stage/block/head独立，避免跨head错误复用权重片",
            "favorable_assumptions": [
                "跳过全零bit-plane",
                "按密度从高到低执行",
                "同stage/block/head首plane后权重驻留",
                "官方周期不计跨plane移位累加与偏置/final输出",
            ],
        },
        "block_head_rows": rows,
        "stages": stage_rows,
        "totals": aggregate(rows),
        "evidence_boundary": [
            "输入来自post-G0 ordered term的gate/lane/multiplicity/destination逐元素重建",
            "profile每个block/sample只抽4个ordered group，不是full-frame workload",
            "Prosperity官方模拟器不建模多bit plane合并、偏置和最终输出",
            "结果用于Local5与强基线的相同sampled scope比较，不与Motion full-window周期直接相除",
            "本结果不是Prosperity PPA复现",
        ],
    }


def write_markdown(report: dict, path: Path) -> None:
    total = report["totals"]
    lines = [
        "# Local5 Post-G0 Gated Projection 的 Prosperity 官方 Bit-Plane 评估\n\n",
        "## 1. 证据口径\n\n",
        f"- sampled ordered groups：`{report['source']['sampled_groups']}`；\n",
        f"- stage/block/head keys：`{report['source']['block_head_keys']}`；\n",
        "- 每个group由`gate×multiplicity`按destination/lane精确重建；\n",
        "- 每个非零bit-plane真实调用官方`Simulator.run_fc` CPU路径；\n",
        "- 跨plane merge、bias和final output不计入官方周期。\n\n",
        "## 2. Stage结果\n\n",
        "| Stage | product cycles | bit-sparse cycles | 官方内部加速 | 未计merge下界 |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for stage, row in report["stages"].items():
        product = row["official_product_sparsity"]["total_cycles"]
        bit_sparse = row["official_bit_sparsity"]["total_cycles"]
        speedup = row["official_product_vs_bit_speedup"]
        text = f"{speedup:.3f}x" if speedup is not None else "N/A"
        lines.append(
            f"| S{stage} | {product} | {bit_sparse} | {text} | "
            f"{row['shift_accumulate_cycles_unmodeled_lower_bound']} |\n"
        )
    speedup = total["official_product_vs_bit_speedup"]
    speedup_text = f"{speedup:.3f}x" if speedup is not None else "N/A"
    lines.append(
        f"| **总计** | {total['official_product_sparsity']['total_cycles']} | "
        f"{total['official_bit_sparsity']['total_cycles']} | {speedup_text} | "
        f"{total['shift_accumulate_cycles_unmodeled_lower_bound']} |\n\n"
    )
    lines.append("## 3. 证据边界\n\n")
    lines.extend(f"- {item}。\n" for item in report["evidence_boundary"])
    lines.extend(
        [
            "\n## 4. 复现\n\n",
            "```bash\n",
            "/opt/conda/envs/sdformerflow/bin/python "
            "scripts/run_prosperity_local5_gated_bitplane.py\n",
            "```\n",
        ]
    )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    report = build_report(args.trace_root)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(report, args.out / "report.md")
    print(args.out / "report.md")
    print(json.dumps(report["totals"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
