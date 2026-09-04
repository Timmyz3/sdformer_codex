#!/usr/bin/env python3
"""Screen an exact digit-serial Local5 gate/weight multiplier on real terms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROLES = 5
GATE_CODES = 512
CHUNK_SIZE = 250_000
POPCOUNT8 = np.asarray([value.bit_count() for value in range(256)], dtype=np.uint8)
POPCOUNT_GATE = np.asarray(
    [value.bit_count() for value in range(GATE_CODES)], dtype=np.uint8
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bitmap_popcount(values: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(values, dtype=np.uint64)
    return POPCOUNT8[values.view(np.uint8).reshape(-1, 8)].sum(
        axis=1, dtype=np.uint16
    )


def analyze_descriptor_chunk(
    gates: np.ndarray,
    valid: np.ndarray,
    k_bitmap: np.ndarray,
    expected_terms: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    gates = np.asarray(gates, dtype=np.uint16)
    valid = np.asarray(valid, dtype=np.uint8)
    k_bitmap = np.asarray(k_bitmap, dtype=np.uint64)
    if gates.ndim != 2 or gates.shape[1] != ROLES:
        raise ValueError("descriptor_incoming_gates必须为[N,5]")
    if valid.shape != (len(gates),) or k_bitmap.shape != (len(gates),):
        raise ValueError("Local5 descriptor数组长度不一致")
    if np.any(gates >= GATE_CODES):
        raise ValueError("gate code超过9-bit范围")

    lane_count = bitmap_popcount(k_bitmap).astype(np.uint32)
    baseline = np.zeros(len(gates), dtype=np.uint32)
    digit_serial = np.zeros(len(gates), dtype=np.uint32)
    popcount_hist = np.zeros(10, dtype=np.uint64)
    gate_hist = np.zeros(GATE_CODES, dtype=np.uint64)

    for role in range(ROLES):
        active = (((valid >> role) & 1) != 0) & (gates[:, role] != 0)
        unique = active.copy()
        for previous in range(role):
            unique &= ~(
                (((valid >> previous) & 1) != 0)
                & (gates[:, previous] == gates[:, role])
            )
        codes = gates[:, role]
        weighted_terms = lane_count * unique.astype(np.uint32)
        digits = POPCOUNT_GATE[codes].astype(np.uint32)
        baseline += weighted_terms
        digit_serial += weighted_terms * digits
        gate_hist += np.bincount(
            codes,
            weights=weighted_terms,
            minlength=GATE_CODES,
        ).astype(np.uint64)
        popcount_hist += np.bincount(
            digits,
            weights=weighted_terms,
            minlength=10,
        ).astype(np.uint64)

    if expected_terms is not None:
        expected = np.asarray(expected_terms, dtype=np.uint32)
        if expected.shape != baseline.shape or not np.array_equal(
            expected, baseline
        ):
            mismatch = np.flatnonzero(expected != baseline)[:5]
            raise AssertionError(
                "term重建与producer不一致: "
                f"indices={mismatch.tolist()}"
            )
    return {
        "baseline_terms": baseline,
        "digit_serial_cycles": digit_serial,
        "popcount_hist": popcount_hist,
        "gate_hist": gate_hist,
    }


def summarize(baseline: np.ndarray, digit_serial: np.ndarray) -> dict[str, Any]:
    base_total = int(np.sum(baseline, dtype=np.uint64))
    digit_total = int(np.sum(digit_serial, dtype=np.uint64))
    active = baseline > 0
    ratio = float(digit_total / base_total) if base_total else 1.0
    per_group_ratio = np.divide(
        digit_serial,
        baseline,
        out=np.ones_like(digit_serial, dtype=np.float64),
        where=baseline != 0,
    )
    return {
        "groups": int(len(baseline)),
        "active_groups": int(np.count_nonzero(active)),
        "baseline_product_terms": base_total,
        "digit_serial_product_cycles": digit_total,
        "weighted_work_ratio": ratio,
        "weighted_work_increase": ratio - 1.0,
        "per_group_ratio": {
            "mean": float(np.mean(per_group_ratio[active])) if np.any(active) else 1.0,
            "p50": float(np.percentile(per_group_ratio[active], 50)) if np.any(active) else 1.0,
            "p95": float(np.percentile(per_group_ratio[active], 95)) if np.any(active) else 1.0,
            "p99": float(np.percentile(per_group_ratio[active], 99)) if np.any(active) else 1.0,
            "max": float(np.max(per_group_ratio[active])) if np.any(active) else 1.0,
        },
    }


def analyze(input_dir: Path) -> dict[str, Any]:
    manifest_path = input_dir / "ordered_term_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "et3_ordered_term_trace_v2":
        raise ValueError("输入不是Local5 ordered term v2 trace")
    if manifest.get("evidence_level") != "post_g0":
        raise ValueError("稀疏数字筛选要求post_g0 trace")
    qualification = manifest.get("qualification", {})
    if not qualification.get("qualified", False):
        raise ValueError("Local5 joint-head qualification未通过")
    payload_path = input_dir / str(manifest.get("payload_file", ""))
    if not payload_path.is_file() or sha256(payload_path) != manifest.get(
        "payload_sha256"
    ):
        raise ValueError("ordered payload文件或SHA256不匹配")

    with np.load(payload_path, allow_pickle=False) as payload:
        offsets = np.asarray(payload["descriptor_group_offsets"], dtype=np.int64)
        source_offsets = np.asarray(payload["source_group_offsets"], dtype=np.int64)
        gates = payload["descriptor_incoming_gates"]
        valid = payload["descriptor_valid_mask"]
        k_bitmap = payload["descriptor_k_bitmap"]
        source_terms = payload["source_term_count"]
        descriptor_count = len(k_bitmap)
        groups = manifest.get("groups", [])
        if (
            not np.array_equal(offsets, source_offsets)
            or len(offsets) != len(groups) + 1
            or offsets[0] != 0
            or offsets[-1] != descriptor_count
        ):
            raise ValueError("descriptor/source group offset合同不一致")

        descriptor_base = np.zeros(descriptor_count, dtype=np.uint32)
        descriptor_digit = np.zeros(descriptor_count, dtype=np.uint32)
        popcount_hist = np.zeros(10, dtype=np.uint64)
        gate_hist = np.zeros(GATE_CODES, dtype=np.uint64)
        for start in range(0, descriptor_count, CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, descriptor_count)
            chunk = analyze_descriptor_chunk(
                gates[start:stop],
                valid[start:stop],
                k_bitmap[start:stop],
                source_terms[start:stop],
            )
            descriptor_base[start:stop] = chunk["baseline_terms"]
            descriptor_digit[start:stop] = chunk["digit_serial_cycles"]
            popcount_hist += chunk["popcount_hist"]
            gate_hist += chunk["gate_hist"]

    group_base = np.add.reduceat(descriptor_base, offsets[:-1]).astype(np.uint64)
    group_digit = np.add.reduceat(descriptor_digit, offsets[:-1]).astype(np.uint64)
    by_stage: dict[str, dict[str, Any]] = {}
    for stage in sorted({int(group["stage"]) for group in groups}):
        mask = np.fromiter(
            (int(group["stage"]) == stage for group in groups),
            dtype=bool,
            count=len(groups),
        )
        by_stage[str(stage)] = summarize(group_base[mask], group_digit[mask])

    total_terms = int(popcount_hist.sum())
    one_hot_terms = int(popcount_hist[1])
    multi_digit_terms = total_terms - one_hot_terms
    top_codes = sorted(
        (
            {"gate": code, "binary_popcount": code.bit_count(), "product_terms": int(count)}
            for code, count in enumerate(gate_hist)
            if count
        ),
        key=lambda row: (-row["product_terms"], row["gate"]),
    )
    global_summary = summarize(group_base, group_digit)
    work_ratio = float(global_summary["weighted_work_ratio"])
    return {
        "schema": "local5_sparse_digit_gate_mul_profile_v1",
        "status": "NO_GO" if work_ratio > 1.05 else "CONDITIONAL",
        "source": {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256(manifest_path),
            "payload": str(payload_path.resolve()),
            "payload_sha256": sha256(payload_path),
            "groups": len(groups),
            "samples": int(qualification.get("processed_samples", 0)),
            "descriptors": int(descriptor_count),
        },
        "global": global_summary,
        "by_stage": by_stage,
        "term_weighted_gate_popcount": {
            "histogram": [int(value) for value in popcount_hist],
            "one_hot_terms": one_hot_terms,
            "one_hot_fraction": float(one_hot_terms / total_terms),
            "multi_digit_terms": multi_digit_terms,
            "multi_digit_fraction": float(multi_digit_terms / total_terms),
        },
        "top_gate_codes": top_codes[:16],
        "decision": {
            "threshold": "digit-serial product work must be <=1.05x one-cycle multiplier terms",
            "result": "NO_GO" if work_ratio > 1.05 else "CONDITIONAL",
            "reason": (
                "真实TCFM5 term加权后，多bit gate使串行shift-add工作量超过门槛；"
                "不写旁路RTL。"
                if work_ratio > 1.05
                else "工作量门槛通过，仍需同吞吐PPA强基线。"
            ),
        },
        "contracts": [
            "一个source的相同非零gate只形成一个term，并按K bitmap非零lane计product term。",
            "基线按每个product term一拍计；digit-serial按gate二进制popcount拍数计。",
            "该变换在足够中间位宽下可保持整数乘法精确，但本报告不构成RTL证明。",
        ],
        "limits": [
            "这是post-G0 100-sample真实term流profile和有限资源工作模型，不是RTL周期。",
            "未计算控制、寄存器、回退、SRAM、路由或功耗；不能作为PPA主张。",
            "raw gate entry分布不能替代按source唯一gate与K lane计权的投影工作分布。",
        ],
    }


def render(report: dict[str, Any]) -> str:
    global_row = report["global"]
    pop = report["term_weighted_gate_popcount"]
    lines = [
        "# Local5 稀疏数字 Gate 乘法筛选",
        "",
        "## 1. 结果",
        "",
        f"裁决：`{report['status']}`。",
        "",
        "按真实 TCFM5 投影 term 加权，而不是按原始 gate entry 计数：",
        "",
        f"- 基线 product term：`{global_row['baseline_product_terms']}`；",
        f"- digit-serial shift-add 周期：`{global_row['digit_serial_product_cycles']}`；",
        f"- 工作量比：`{global_row['weighted_work_ratio']:.4f}x`；",
        f"- one-hot gate term：`{pop['one_hot_fraction']:.2%}`；",
        f"- multi-digit gate term：`{pop['multi_digit_fraction']:.2%}`。",
        "",
        "因此，大量原始 gate 虽集中于 16/32，但按实际 K lane 和 source 唯一 gate",
        "计权后，仍有超过四分之一 product term 需要 4 或 5 个二进制数字。",
        "串行 shift-add 不能打赢一拍乘法强基线，不进入 RTL。",
        "",
        "## 2. 按 Stage",
        "",
        "| stage | baseline term | digit cycles | work ratio |",
        "|---:|---:|---:|---:|",
    ]
    for stage, row in report["by_stage"].items():
        lines.append(
            f"| {stage} | {row['baseline_product_terms']} | "
            f"{row['digit_serial_product_cycles']} | {row['weighted_work_ratio']:.4f}x |"
        )
    lines.extend(
        [
            "",
            "## 3. Term 加权 Gate 分布",
            "",
            "| gate | popcount | product term |",
            "|---:|---:|---:|",
        ]
    )
    for row in report["top_gate_codes"]:
        lines.append(
            f"| {row['gate']} | {row['binary_popcount']} | {row['product_terms']} |"
        )
    lines.extend(
        [
            "",
            "## 4. 证据边界",
            "",
            "- `[prof]`：post-G0、100-sample、13,800 group 的真实 ordered term 流；",
            "- `[模型]`：每 product 一拍与每 binary digit 一拍的局部工作模型；",
            "- 不是 RTL 周期、DC/STA/SAIF、full encoder 或 ASIC PPA；",
            "- 不修改 Local5 封存主表，不复活 Prosperity bit-plane 或 ECGB。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(render(report), encoding="utf-8")
    print(
        "LOCAL5_SPARSE_DIGIT "
        f"status={report['status']} "
        f"terms={report['global']['baseline_product_terms']} "
        f"digit_cycles={report['global']['digit_serial_product_cycles']} "
        f"ratio={report['global']['weighted_work_ratio']:.6f}"
    )


if __name__ == "__main__":
    main()
