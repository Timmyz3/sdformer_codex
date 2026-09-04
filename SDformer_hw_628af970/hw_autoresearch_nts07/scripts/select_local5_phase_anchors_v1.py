#!/usr/bin/env python3
"""从真实 Local5 ordered-term workload 冻结跨 H/序列/密度 phase anchor 计划。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


SCHEMA = "local5_phase_anchor_selection_v1"
STATUS = "FROZEN_PROFILE_SELECTION_NOT_RTL"
PACKAGE_REVISION = "v4_reviewfix"
EXPECTED_HEADS = {0: 3, 1: 6, 2: 12, 3: 24}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层不是 object: {path}")
    return value


def identity_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return row["sample"], row["stage"], row["block"], row["window"]


def aggregate_windows(
    groups: list[dict[str, Any]], arrays: Any
) -> list[dict[str, Any]]:
    required = {
        "group_offsets",
        "descriptor_group_offsets",
        "item_mode_multiset",
        "source_term_count",
        "source_delivery_count",
        "source_service_cycles_pipelined",
    }
    if not required.issubset(arrays.files):
        raise ValueError("ordered-term payload 缺少 anchor 统计所需数组")
    # NPZ 是压缩容器；每列只解压一次，禁止在 13,800-group 循环内反复 __getitem__。
    group_offsets = arrays["group_offsets"]
    descriptor_offsets = arrays["descriptor_group_offsets"]
    item_mode_multiset = arrays["item_mode_multiset"]
    source_term_count = arrays["source_term_count"]
    source_delivery_count = arrays["source_delivery_count"]
    source_service_cycles = arrays["source_service_cycles_pipelined"]
    exact_dtypes = {
        "group_offsets": np.dtype("int64"),
        "descriptor_group_offsets": np.dtype("int64"),
        "item_mode_multiset": np.dtype("uint8"),
        "source_term_count": np.dtype("uint16"),
        "source_delivery_count": np.dtype("uint16"),
        "source_service_cycles_pipelined": np.dtype("uint16"),
    }
    loaded = {
        "group_offsets": group_offsets,
        "descriptor_group_offsets": descriptor_offsets,
        "item_mode_multiset": item_mode_multiset,
        "source_term_count": source_term_count,
        "source_delivery_count": source_delivery_count,
        "source_service_cycles_pipelined": source_service_cycles,
    }
    for name, value in loaded.items():
        if value.dtype != exact_dtypes[name]:
            raise ValueError(f"{name} dtype 不符合冻结合同")
        if value.ndim != 1:
            raise ValueError(f"{name} 不是一维数组")
    if group_offsets.shape != (len(groups) + 1,) or descriptor_offsets.shape != (
        len(groups) + 1,
    ):
        raise ValueError("group offset 形状与 manifest 不一致")
    if int(group_offsets[0]) != 0 or int(descriptor_offsets[0]) != 0:
        raise ValueError("group offset 起点不是 0")
    if np.any(np.diff(group_offsets) < 0) or np.any(np.diff(descriptor_offsets) < 0):
        raise ValueError("group offset 非单调")
    item_count = int(group_offsets[-1])
    descriptor_count = int(descriptor_offsets[-1])
    if item_count != item_mode_multiset.shape[0]:
        raise ValueError("group offset 终点与 item 数组长度不一致")
    for name, value in (
        ("source_term_count", source_term_count),
        ("source_delivery_count", source_delivery_count),
        ("source_service_cycles_pipelined", source_service_cycles),
    ):
        if descriptor_count != value.shape[0]:
            raise ValueError(f"descriptor offset 终点与 {name} 长度不一致")

    aggregate: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for index, group in enumerate(groups):
        required_group = {
            "tag",
            "sample",
            "stage",
            "block",
            "window",
            "head",
            "heads",
            "tokens",
        }
        if not required_group.issubset(group):
            raise ValueError(f"group {index} 缺字段")
        if group["tag"] != index:
            raise ValueError("group tag 不是连续 producer 顺序")
        stage = int(group["stage"])
        heads = int(group["heads"])
        if stage not in EXPECTED_HEADS or heads != EXPECTED_HEADS[stage]:
            raise ValueError("stage/head 拓扑不符合 Local5 冻结合同")
        if int(group["head"]) < 0 or int(group["head"]) >= heads:
            raise ValueError("group head 越界")
        key = (
            int(group["sample"]),
            stage,
            int(group["block"]),
            int(group["window"]),
        )
        row = aggregate.setdefault(
            key,
            {
                "sample": key[0],
                "stage": key[1],
                "block": key[2],
                "window": key[3],
                "heads": heads,
                "tokens": int(group["tokens"]),
                "head_ids": [],
                "term_items": 0,
                "source_descriptors": 0,
                "active_sources": 0,
                "source_terms": 0,
                "source_deliveries": 0,
                "service_cycles": 0,
            },
        )
        if row["heads"] != heads or row["tokens"] != int(group["tokens"]):
            raise ValueError("同一 window 的 heads/tokens 不一致")
        item_start = int(group_offsets[index])
        item_end = int(group_offsets[index + 1])
        descriptor_start = int(descriptor_offsets[index])
        descriptor_end = int(descriptor_offsets[index + 1])
        if item_end < item_start or descriptor_end < descriptor_start:
            raise ValueError("group slice 非法")
        row["head_ids"].append(int(group["head"]))
        row["term_items"] += item_end - item_start
        row["source_descriptors"] += descriptor_end - descriptor_start
        term_counts = source_term_count[descriptor_start:descriptor_end]
        row["active_sources"] += int(np.count_nonzero(term_counts))
        row["source_terms"] += int(np.sum(term_counts, dtype=np.uint64))
        row["source_deliveries"] += int(
            np.sum(
                source_delivery_count[descriptor_start:descriptor_end],
                dtype=np.uint64,
            )
        )
        row["service_cycles"] += int(
            np.sum(
                source_service_cycles[descriptor_start:descriptor_end],
                dtype=np.uint64,
            )
        )

    rows = []
    for key in sorted(aggregate):
        row = aggregate[key]
        expected_heads = list(range(row["heads"]))
        if sorted(row.pop("head_ids")) != expected_heads:
            raise ValueError(f"window {key} 缺失、重复或乱序 head")
        expected_descriptors = row["heads"] * row["tokens"]
        if row["source_descriptors"] != expected_descriptors:
            raise ValueError(f"window {key} source descriptor 数量不匹配")
        row["term_items_per_head"] = row["term_items"] / row["heads"]
        row["active_source_ratio"] = (
            row["active_sources"] / row["source_descriptors"]
        )
        row["service_cycles_per_head"] = row["service_cycles"] / row["heads"]
        rows.append(row)
    return rows


def choose_nearest(
    candidates: list[dict[str, Any]], metric: str, target: float
) -> dict[str, Any]:
    return min(
        candidates,
        key=lambda row: (abs(float(row[metric]) - target), identity_key(row)),
    )


def build_anchor_plan(
    rows: list[dict[str, Any]], sequence_keys: list[str]
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("没有可选 window")
    sample_ids = sorted({row["sample"] for row in rows})
    if sample_ids != list(range(len(sequence_keys))):
        raise ValueError("row sample 与 cohort sequence_keys 不连续对应")
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_heads: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sequence = sequence_keys[row["sample"]]
        if not isinstance(sequence, str) or not sequence:
            raise ValueError("sequence key 非法")
        row["sequence_key"] = sequence
        by_sequence[sequence].append(row)
        by_heads[row["heads"]].append(row)
    if len(by_sequence) != 18:
        raise ValueError(f"正式 cohort 应为 18 个 sequence cluster，实际 {len(by_sequence)}")
    if set(by_heads) != set(EXPECTED_HEADS.values()):
        raise ValueError("未覆盖 H3/H6/H12/H24")

    selected: dict[tuple[int, int, int, int], dict[str, Any]] = {}

    def add(row: dict[str, Any], reason: str) -> None:
        key = identity_key(row)
        if key not in selected:
            selected[key] = {
                key_name: row[key_name]
                for key_name in (
                    "sample",
                    "stage",
                    "block",
                    "window",
                    "heads",
                    "tokens",
                    "sequence_key",
                    "term_items",
                    "term_items_per_head",
                    "active_source_ratio",
                    "source_terms",
                    "source_deliveries",
                    "service_cycles",
                    "service_cycles_per_head",
                )
            }
            selected[key]["reasons"] = []
        if reason not in selected[key]["reasons"]:
            selected[key]["reasons"].append(reason)

    for sequence in sorted(by_sequence):
        candidates = by_sequence[sequence]
        target = float(median(row["term_items_per_head"] for row in candidates))
        add(
            choose_nearest(candidates, "term_items_per_head", target),
            f"SEQUENCE_MEDIAN:{sequence}",
        )

    for heads in sorted(by_heads):
        candidates = by_heads[heads]
        for metric in ("term_items_per_head", "service_cycles_per_head"):
            add(min(candidates, key=lambda row: (row[metric], identity_key(row))), f"H{heads}_{metric}_MIN")
            add(max(candidates, key=lambda row: (row[metric], tuple(-v for v in identity_key(row)))), f"H{heads}_{metric}_MAX")
        active_candidates = [
            row
            for row in candidates
            if row["active_source_ratio"] > 0 and row["service_cycles_per_head"] > 0
        ]
        if not active_candidates:
            raise ValueError(f"H{heads} 没有非零随机反压候选")
        target = float(
            median(row["term_items_per_head"] for row in active_candidates)
        )
        add(
            choose_nearest(active_candidates, "term_items_per_head", target),
            f"H{heads}_NONZERO_RANDOM_BACKPRESSURE_ANCHOR",
        )

    anchors = [selected[key] for key in sorted(selected)]
    for anchor_id, anchor in enumerate(anchors):
        anchor["anchor_id"] = anchor_id
        anchor["reasons"].sort()
    return anchors


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def render_report(plan: dict[str, Any]) -> str:
    coverage = plan["coverage"]
    lines = [
        "# Local5 跨 H 与序列 Phase Anchor 冻结计划",
        "",
        "> 证据等级：`[prof]+[计划]`；本文件不是 RTL 通过证明，formal G0 仍为 **DENY**。",
        "",
        "## 结论",
        "",
        f"从 100 个真实样本、{coverage['windows']} 个 canonical block-window 中，确定性选出 "
        f"{coverage['anchors']} 个 anchor。集合覆盖 {coverage['sequence_clusters']}/18 个序列 "
        f"cluster 和 H3/H6/H12/H24，并在每个 H 内加入 term 密度、服务周期极值及随机反压锚点。",
        "",
        "该集合只回答‘下一批完整逐事件/compact RTL trace 应回放哪些 workload’，不回答周期收益、PPA 或架构创新性。",
        "",
        "## 选择规则",
        "",
        "1. 每个 sequence cluster 选 term-items/head 最接近本 cluster 中位数的窗口。",
        "2. 每个 H 分别选择 sampled canonical window 中 term-items/head 与 service-cycles/head 的最小、最大窗口。",
        "3. 每个 H 从 active_source_ratio>0 且 service_cycles>0 的集合选择中位 term 密度窗口，后续施加固定 seed 的确定性随机反压。",
        "4. 同一窗口命中多个规则时合并 reason，不重复回放。",
        "5. 所有统计来自冻结 ordered-term payload；没有把模型量改名为 RTL_DIRECT。",
        "",
        "## 覆盖",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 输入样本 | {coverage['samples']} |",
        f"| 输入 canonical window | {coverage['windows']} |",
        f"| 序列 cluster | {coverage['sequence_clusters']} |",
        f"| 选中 anchor | {coverage['anchors']} |",
        f"| H 集合 | {', '.join('H'+str(value) for value in coverage['heads'])} |",
        "",
        "该计划只保证 H 与 sequence cluster 的边际覆盖，不声称 4x18 笛卡尔覆盖；机器可读 Hxcluster 矩阵见 `anchor_plan.json`。极值也仅是上游均匀抽取的 1,200 个 canonical window 内的 sampled extrema，不是完整空间窗口总体极值。",
        "",
        "## 后续准入",
        "",
        "- 先对四个 H 的中位锚点做完整 trace 与 compact telemetry 同构；",
        f"- 再跑本计划全部 anchor，并对每个 H 的反压锚点使用冻结 seed {plan['selection_contract']['backpressure_seeds']}；",
        "- 只有真实 RTL count/digest/Acc32 全部绑定后，anchor coverage 才能标为 `[rtl-anchor]`；",
        "- 全 1,200 window Direct compact telemetry 与 admission receipt 仍是 formal G0 独立门槛。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    staging = output.with_name(output.name + f".staging.{os.getpid()}")
    if output.exists() or staging.exists():
        raise ValueError("输出目录已存在")
    staging.mkdir(parents=True)
    try:
        manifest = read_json(args.manifest)
        cohort = read_json(args.cohort)
        if manifest.get("schema") != "et3_ordered_term_trace_v2":
            raise ValueError("ordered-term manifest schema 不匹配")
        if cohort.get("schema") != "ordered_trace_cohort_v2":
            raise ValueError("cohort schema 不匹配")
        if manifest.get("payload_sha256") != sha256_file(args.payload):
            raise ValueError("ordered-term payload SHA 不匹配")
        if manifest.get("cohort_file_sha256") != sha256_file(args.cohort):
            raise ValueError("ordered-term manifest 未绑定本 cohort")
        groups = manifest.get("groups")
        sequence_keys = cohort.get("sequence_keys")
        if not isinstance(groups, list) or not isinstance(sequence_keys, list):
            raise ValueError("manifest/cohort 主数组缺失")
        with np.load(args.payload, allow_pickle=False) as arrays:
            rows = aggregate_windows(groups, arrays)
        anchors = build_anchor_plan(rows, sequence_keys)
        heads = sorted({row["heads"] for row in rows})
        sequence_clusters = sorted(set(sequence_keys))
        hxcluster_matrix = {
            f"H{head}": {
                sequence: sum(
                    1
                    for anchor in anchors
                    if anchor["heads"] == head and anchor["sequence_key"] == sequence
                )
                for sequence in sequence_clusters
            }
            for head in heads
        }
        plan = {
            "schema": SCHEMA,
            "package_revision": PACKAGE_REVISION,
            "status": STATUS,
            "formal_g0": "DENY",
            "evidence": "[prof]+[计划]",
            "input_bindings": {
                "manifest": str(args.manifest.resolve()),
                "manifest_sha256": sha256_file(args.manifest),
                "payload": str(args.payload.resolve()),
                "payload_sha256": sha256_file(args.payload),
                "cohort": str(args.cohort.resolve()),
                "cohort_sha256": sha256_file(args.cohort),
            },
            "selection_contract": {
                "sequence": "nearest cluster median term_items_per_head",
                "per_h_extrema": [
                    "term_items_per_head:min/max",
                    "service_cycles_per_head:min/max",
                ],
                "backpressure": "nearest per-H median term_items_per_head",
                "backpressure_active_filter": "active_source_ratio>0 and service_cycles_per_head>0",
                "backpressure_seeds": [20260813, 20260814],
                "deduplication": "identity tuple with merged sorted reasons",
                "coverage_scope": "marginal H and sequence-cluster coverage; not Hxcluster Cartesian closure",
                "extrema_scope": "sampled extrema over 1200 uniformly selected canonical windows",
            },
            "coverage": {
                "samples": len(sequence_keys),
                "windows": len(rows),
                "sequence_clusters": len(set(sequence_keys)),
                "heads": heads,
                "anchors": len(anchors),
                "hxcluster_nonzero_cells": sum(
                    count > 0
                    for row in hxcluster_matrix.values()
                    for count in row.values()
                ),
                "hxcluster_total_cells": len(heads) * len(sequence_clusters),
            },
            "hxcluster_matrix": hxcluster_matrix,
            "anchors": anchors,
        }
        write_json(staging / "anchor_plan.json", plan)
        (staging / "report.md").write_text(render_report(plan), encoding="utf-8")
        source_copy = staging / Path(__file__).name
        shutil.copy2(Path(__file__).resolve(), source_copy)
        test_source = Path(__file__).resolve().with_name(
            "test_select_local5_phase_anchors_v1.py"
        )
        if not test_source.is_file() or test_source.is_symlink():
            raise ValueError("anchor selector 单测源码缺失或为符号链接")
        test_copy = staging / test_source.name
        shutil.copy2(test_source, test_copy)
        test_argv = [sys.executable, str(test_source)]
        test_completed = subprocess.run(
            test_argv,
            check=False,
            capture_output=True,
            text=True,
            cwd=Path("/tmp"),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        test_receipt = {
            "schema": "local5_phase_anchor_selection_test_receipt_v1",
            "package_revision": PACKAGE_REVISION,
            "status": "PASS" if test_completed.returncode == 0 else "FAIL",
            "returncode": test_completed.returncode,
            "argv": test_argv,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "test_source_sha256": sha256_file(test_source),
            "stdout": test_completed.stdout,
            "stderr": test_completed.stderr,
        }
        write_json(staging / "test_execution.json", test_receipt)
        if test_completed.returncode != 0:
            raise ValueError("anchor selector 冻结单测未通过")
        complete = {
            "schema": "local5_phase_anchor_selection_complete_v1",
            "package_revision": PACKAGE_REVISION,
            "status": STATUS,
            "formal_g0": "DENY",
            "anchor_plan_sha256": sha256_file(staging / "anchor_plan.json"),
            "report_sha256": sha256_file(staging / "report.md"),
            "source_sha256": sha256_file(source_copy),
            "test_source_sha256": sha256_file(test_copy),
            "test_execution_sha256": sha256_file(staging / "test_execution.json"),
        }
        write_json(staging / "complete.json", complete)
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(json.dumps({"status": STATUS, "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
