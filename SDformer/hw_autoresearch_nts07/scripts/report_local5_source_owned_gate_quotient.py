#!/usr/bin/env python3
"""Audit the existing Local5 source-owned gate/lane execution object."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "results/local5_fullres_bb1e4_joint_heads_profile100_20260809"
)
REQUIRED_ARRAYS = {
    "group_offsets",
    "item_mode_multiset",
    "item_multiplicity",
    "descriptor_group_offsets",
    "descriptor_incoming_gates",
    "descriptor_valid_mask",
    "source_term_count",
    "source_gate_count",
    "source_k_popcount",
    "source_delivery_count",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_nonzero_gate_count(
    gates: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    if gates.ndim != 2 or gates.shape[1] != 5:
        raise ValueError("descriptor_incoming_gates must have shape [N,5]")
    if valid_mask.shape != (gates.shape[0],):
        raise ValueError("descriptor_valid_mask shape mismatch")
    roles = np.arange(5, dtype=np.uint8)
    valid = ((valid_mask[:, None] >> roles) & 1) != 0
    masked = np.where(valid, gates, 0)
    count = np.zeros(gates.shape[0], dtype=np.uint8)
    for role in range(5):
        nonzero = masked[:, role] != 0
        seen = np.zeros(gates.shape[0], dtype=np.bool_)
        for prior in range(role):
            seen |= masked[:, prior] == masked[:, role]
        count += nonzero & ~seen
    return count


def analyze_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    missing = sorted(REQUIRED_ARRAYS.difference(arrays))
    if missing:
        raise ValueError(f"ordered term payload missing arrays: {missing}")

    group_offsets = np.asarray(arrays["group_offsets"], dtype=np.int64)
    item_mode = np.asarray(arrays["item_mode_multiset"])
    multiplicity = np.asarray(arrays["item_multiplicity"], dtype=np.int64)
    descriptor_offsets = np.asarray(
        arrays["descriptor_group_offsets"], dtype=np.int64
    )
    gates = np.asarray(arrays["descriptor_incoming_gates"])
    valid_mask = np.asarray(arrays["descriptor_valid_mask"], dtype=np.uint8)
    source_terms = np.asarray(arrays["source_term_count"], dtype=np.int64)
    source_gate_count = np.asarray(
        arrays["source_gate_count"], dtype=np.int64
    )
    source_k_popcount = np.asarray(
        arrays["source_k_popcount"], dtype=np.int64
    )
    source_delivery = np.asarray(
        arrays["source_delivery_count"], dtype=np.int64
    )

    if group_offsets.ndim != 1 or group_offsets.size < 2:
        raise ValueError("group_offsets must be a non-empty prefix sum")
    if descriptor_offsets.shape != group_offsets.shape:
        raise ValueError("descriptor/group offset shapes differ")
    if int(group_offsets[0]) != 0 or int(group_offsets[-1]) != item_mode.size:
        raise ValueError("item group offsets do not cover the payload")
    if int(descriptor_offsets[0]) != 0 or int(descriptor_offsets[-1]) != gates.shape[0]:
        raise ValueError("descriptor group offsets do not cover the payload")
    if multiplicity.shape != item_mode.shape:
        raise ValueError("item mode/multiplicity shapes differ")
    if np.any(item_mode != 1):
        raise ValueError("destination-major payload contains a non-multiset item")
    if multiplicity.size and (
        int(multiplicity.min()) < 1 or int(multiplicity.max()) > 5
    ):
        raise ValueError("item multiplicity is outside [1,5]")

    descriptor_count = gates.shape[0]
    descriptor_vectors = (
        valid_mask,
        source_terms,
        source_gate_count,
        source_k_popcount,
        source_delivery,
    )
    if any(array.shape != (descriptor_count,) for array in descriptor_vectors):
        raise ValueError("source descriptor array shapes differ")

    roles = np.arange(5, dtype=np.uint8)
    valid = ((valid_mask[:, None] >> roles) & 1) != 0
    invalid_nonzero_gate = int(np.count_nonzero((~valid) & (gates != 0)))
    if invalid_nonzero_gate:
        raise ValueError("invalid Local5 candidate carries a nonzero gate")

    active_source = source_k_popcount > 0
    unique_gates = unique_nonzero_gate_count(gates, valid_mask).astype(np.int64)
    expected_gate_count = np.where(active_source, unique_gates, 0)
    gate_count_mismatch = int(
        np.count_nonzero(expected_gate_count != source_gate_count)
    )
    expected_source_terms = source_k_popcount * source_gate_count
    source_term_mismatch = int(
        np.count_nonzero(expected_source_terms != source_terms)
    )
    valid_nonzero_edges = ((gates != 0) & valid).sum(
        axis=1, dtype=np.int64
    )
    expected_delivery = source_k_popcount * valid_nonzero_edges
    source_delivery_mismatch = int(
        np.count_nonzero(expected_delivery != source_delivery)
    )
    if gate_count_mismatch or source_term_mismatch or source_delivery_mismatch:
        raise ValueError(
            "source-owned quotient formulas mismatch: "
            f"gate={gate_count_mismatch} term={source_term_mismatch} "
            f"delivery={source_delivery_mismatch}"
        )

    relation_lane_delivery = int(multiplicity.sum())
    source_delivery_total = int(source_delivery.sum())
    if relation_lane_delivery != source_delivery_total:
        raise ValueError(
            "destination/source transposition does not conserve delivery: "
            f"{relation_lane_delivery} != {source_delivery_total}"
        )
    destination_terms = int(multiplicity.size)
    source_terms_total = int(source_terms.sum())
    if not (source_terms_total <= destination_terms <= relation_lane_delivery):
        raise ValueError("term ordering violates exact quotient bounds")

    histogram = Counter(int(value) for value in multiplicity.tolist())
    return {
        "groups": int(group_offsets.size - 1),
        "source_descriptors": int(descriptor_count),
        "active_source_descriptors": int(np.count_nonzero(active_source)),
        "relation_lane_delivery": relation_lane_delivery,
        "destination_local_mfep_terms": destination_terms,
        "source_owned_gate_lane_terms": source_terms_total,
        "source_unique_gate_instances": int(source_gate_count.sum()),
        "multiplicity_histogram": {
            str(key): value for key, value in sorted(histogram.items())
        },
        "destination_term_reduction_vs_relation_lane": (
            1.0 - destination_terms / relation_lane_delivery
        ),
        "source_term_reduction_vs_relation_lane": (
            1.0 - source_terms_total / relation_lane_delivery
        ),
        "source_term_reduction_vs_destination_mfep": (
            1.0 - source_terms_total / destination_terms
        ),
        "destination_mfep_over_source_term_ratio": (
            destination_terms / source_terms_total
        ),
        "checks": {
            "destination_source_delivery_conserved": True,
            "source_gate_count_is_unique_nonzero_gate_count": True,
            "source_term_equals_k_popcount_times_unique_gate_count": True,
            "source_delivery_equals_k_popcount_times_valid_nonzero_edges": True,
            "invalid_candidate_nonzero_gate_count": 0,
        },
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    ledger = report["ledger"]
    lines = [
        "# Local5 source-owned gate/lane quotient 审计",
        "",
        "## 裁决",
        "",
        "- 状态：**ADMIT_AS_EXISTING_LOCAL5_CORE_EXECUTION_OBJECT**。",
        "- 该机制已存在于 source descriptor/TCFM5 数据流，本报告是恢复并量化，不是新增 RTL 或新发明。",
        "- 证据等级：`[prof]`。term 是 gate x weight product work，不是周期、功耗或 ASIC PPA。",
        "",
        "## 精确数据流",
        "",
        "对固定 source `s`、活动 K lane `l` 和 gate code `g`，把所有满足",
        "`K_s[l]=1 && gate(d,s)=g` 的 destination 放入同一有界 destination set。",
        "硬件只生成一次 `g*W[l,:]`，再由 TCFM5 对该集合执行精确多播。destination",
        "identity 没有被 count folding 删除，delivery 总数在转置前后严格守恒。",
        "",
        "## 同 trace 强基线",
        "",
        "| 执行对象 | term / delivery | 相对 relation-lane 减少 |",
        "|---|---:|---:|",
        f"| raw relation-lane delivery | {ledger['relation_lane_delivery']} | 0.00% |",
        f"| destination-local MFEP multiplicity term | {ledger['destination_local_mfep_terms']} | {100*ledger['destination_term_reduction_vs_relation_lane']:.2f}% |",
        f"| source-owned gate-lane multicast term | {ledger['source_owned_gate_lane_terms']} | {100*ledger['source_term_reduction_vs_relation_lane']:.2f}% |",
        "",
        f"source-owned 相对 destination-local MFEP 再减少 **{100*ledger['source_term_reduction_vs_destination_mfep']:.2f}%** product term；",
        f"MFEP term 数是 source-owned 的 **{ledger['destination_mfep_over_source_term_ratio']:.3f}x**。",
        "",
        "## 证据边界",
        "",
        "- 29.16M destination delivery 没有被删除；减少的是重复 gate x weight product。",
        "- source-major 是强基线胜出后的现行执行对象，MFEP 不恢复为并行主机制。",
        "- DATE 创新表述应与 Query-Silent、编译期有界转置和 TCFM5 组成一条 Local5 数据流，不能拆成四条独立贡献。",
        "- 仍需同端口存储模型、DC/STA/SAIF 和多样本生产 RTL 来证明物理收益。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(source_dir: Path) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    manifest_path = source_dir / "ordered_term_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload_path = source_dir / manifest["payload_file"]
    payload_sha = sha256(payload_path)
    if payload_sha != manifest.get("payload_sha256"):
        raise ValueError("ordered term payload SHA256 mismatch")
    qualification = manifest.get("qualification", {})
    if not qualification.get("qualified") or qualification.get("processed_samples") != 100:
        raise ValueError("source trace is not the qualified profile100 cohort")
    with np.load(payload_path, allow_pickle=False) as payload:
        ledger = analyze_arrays(payload)
    if ledger["groups"] != 13800:
        raise ValueError(f"expected 13800 qualified groups, got {ledger['groups']}")
    return {
        "schema": "local5_source_owned_gate_quotient_profile100_v1",
        "status": "ADMIT_AS_EXISTING_LOCAL5_CORE_EXECUTION_OBJECT",
        "evidence": "[prof]",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "source_payload": str(payload_path),
        "source_payload_sha256": payload_sha,
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "config_sha256": manifest.get("config_sha256"),
        "producer_order_contract": manifest.get("producer_order_contract"),
        "source_descriptor_contract": manifest.get("source_descriptor_contract"),
        "ledger": ledger,
        "claim_boundary": {
            "product_term_not_cycle_or_energy": True,
            "destination_delivery_is_not_reduced": True,
            "existing_mechanism_not_new_rtl": True,
            "not_encoder_speedup": True,
            "not_asic_ppa": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.source_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "report.json"
    md_path = args.output_dir / "report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(md_path, report)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
