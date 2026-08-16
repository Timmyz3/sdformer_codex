#!/usr/bin/env python3
"""审计 Local5 formal phase archive 的事件规模与去重下界。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


FORMAL_GROUPS = 13_800
FORMAL_WINDOWS = 1_200
FORMAL_PHASES = 462_600
FORMAL_ACC32 = 198_720_000
TOKENS = 450
V4_EVENT_BYTES = 1 + 4 + 64  # resource:uint8, cycle:uint32, identity:S64
STRUCTURAL_EVENT_BYTES = 1 + 4 + 4  # resource:uint8, cycle:uint32, identity:uint32
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_SOURCE_FILES = (
    SCRIPT_DIR / "audit_local5_formal_phase_archive_scale.py",
    SCRIPT_DIR / "local5_erep_archive_replay_v4.py",
    SCRIPT_DIR / "local5_erep_ledger_replay_v4.py",
    SCRIPT_DIR / "local5_erep_command_schedule_v4.py",
    SCRIPT_DIR / "local5_erep_capacity_baselines_v4.py",
    SCRIPT_DIR / "local5_erep_identity_service_v4.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def estimate_counts(
    unique_item_counts: np.ndarray,
    product_term_counts: np.ndarray,
    delivery_counts: np.ndarray,
    active_records: np.ndarray,
    heads: np.ndarray,
) -> dict[str, int]:
    unique_item_counts = np.asarray(unique_item_counts, dtype=np.int64)
    product_term_counts = np.asarray(product_term_counts, dtype=np.int64)
    delivery_counts = np.asarray(delivery_counts, dtype=np.int64)
    active_records = np.asarray(active_records, dtype=np.int64)
    heads = np.asarray(heads, dtype=np.int64)
    if (
        unique_item_counts.ndim != 1
        or product_term_counts.shape != unique_item_counts.shape
        or delivery_counts.shape != unique_item_counts.shape
        or active_records.shape != unique_item_counts.shape
        or heads.shape != unique_item_counts.shape
        or np.any(unique_item_counts < 0)
        or np.any(product_term_counts < 0)
        or np.any(delivery_counts < product_term_counts)
        or np.any(active_records < 0)
        or np.any(active_records > TOKENS)
        or np.any(heads <= 0)
    ):
        raise ValueError("term/active/head 输入形状或范围不合法")
    unique_items = int(unique_item_counts.sum())
    product_terms = int(product_term_counts.sum())
    deliveries = int(delivery_counts.sum())
    records = int(active_records.sum())
    tile_product_terms = int(np.dot(product_term_counts, heads))
    tile_deliveries = int(np.dot(delivery_counts, heads))
    tile_records = int(np.dot(active_records, heads))
    fill_events = 2 * records
    execute_metadata_events = 3 * tile_records
    two_path_acc_events = 2 * tile_deliveries
    expanded = fill_events + execute_metadata_events + two_path_acc_events
    template = 5 * records + 2 * deliveries
    return {
        "destination_unique_items": unique_items,
        "source_product_terms": product_terms,
        "destination_deliveries": deliveries,
        "active_records": records,
        "tile_expanded_product_terms": tile_product_terms,
        "tile_expanded_deliveries": tile_deliveries,
        "tile_expanded_records": tile_records,
        "fill_events": fill_events,
        "execute_metadata_events": execute_metadata_events,
        "two_path_acc_update_events": two_path_acc_events,
        "v4_main_expanded_events_excluding_common": expanded,
        "head_template_events": template,
    }


def load_workload(root: Path) -> tuple[dict[str, Any], dict[str, int]]:
    manifest_path = root / "ordered_term_manifest.json"
    payload_path = root / "ordered_term_items.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = manifest.get("groups")
    if (
        manifest.get("schema") != "et3_ordered_term_trace_v2"
        or not manifest.get("qualification", {}).get("qualified")
        or not isinstance(groups, list)
        or len(groups) != FORMAL_GROUPS
    ):
        raise ValueError("formal ordered-term manifest 合同不成立")
    with np.load(payload_path, mmap_mode="r", allow_pickle=False) as payload:
        required = {
            "group_offsets",
            "source_group_offsets",
            "source_term_count",
            "source_delivery_count",
        }
        if not required.issubset(payload.files):
            raise ValueError("ordered-term payload 缺少规模审计数组")
        group_offsets = np.asarray(payload["group_offsets"], dtype=np.int64)
        source_offsets = np.asarray(payload["source_group_offsets"], dtype=np.int64)
        source_term_count = np.asarray(payload["source_term_count"])
        source_delivery_count = np.asarray(payload["source_delivery_count"])
        if (
            group_offsets.shape != (FORMAL_GROUPS + 1,)
            or source_offsets.shape != (FORMAL_GROUPS + 1,)
            or int(group_offsets[0]) != 0
            or int(source_offsets[0]) != 0
            or np.any(np.diff(group_offsets) < 0)
            or not np.all(np.diff(source_offsets) == TOKENS)
            or int(source_offsets[-1]) != source_term_count.size
            or source_delivery_count.shape != source_term_count.shape
        ):
            raise ValueError("ordered-term offsets/shape 合同不成立")
        unique_item_counts = np.diff(group_offsets)
        product_term_counts = np.add.reduceat(
            source_term_count.astype(np.int64), source_offsets[:-1]
        )
        delivery_counts = np.add.reduceat(
            source_delivery_count.astype(np.int64), source_offsets[:-1]
        )
        active_records = np.add.reduceat(
            (source_term_count > 0).astype(np.int64), source_offsets[:-1]
        )
    heads = np.asarray([int(row.get("heads", -1)) for row in groups], dtype=np.int64)
    if set(heads.tolist()) != {3, 6, 12, 24}:
        raise ValueError("formal stage head 集合不等于 {3,6,12,24}")
    bindings = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "payload": str(payload_path.resolve()),
        "payload_sha256": sha256(payload_path),
    }
    return bindings, estimate_counts(
        unique_item_counts,
        product_term_counts,
        delivery_counts,
        active_records,
        heads,
    )


def build_report(bindings: dict[str, Any], counts: dict[str, int]) -> dict[str, Any]:
    expanded = counts["v4_main_expanded_events_excluding_common"]
    template = counts["head_template_events"]
    common_events = {
        "one_event_per_common_phase": FORMAL_GROUPS * 2,
        "vector_drain_450": FORMAL_GROUPS * (1 + TOKENS),
        "scalar_drain_450x32": FORMAL_GROUPS * (1 + TOKENS * 32),
    }
    phase_metadata_bytes = FORMAL_PHASES * (2 + 2 + 1 + 2 + 4) + (
        FORMAL_PHASES + 1
    ) * 8
    v4_bytes = {
        key: (expanded + value) * V4_EVENT_BYTES + phase_metadata_bytes
        for key, value in common_events.items()
    }
    base_template_bytes_excluding_tile_patch = {
        key: (
            (template + value) * STRUCTURAL_EVENT_BYTES
            + phase_metadata_bytes
            + FORMAL_PHASES * 4
        )
        for key, value in common_events.items()
    }
    delivery_aligned_cycle_bytes = (
        counts["destination_deliveries"] * 2 * 4
        + counts["active_records"] * 5 * 4
        + phase_metadata_bytes
        + FORMAL_PHASES * 4
    )
    patch_target_events = expanded
    dense_cycle_patch_bytes = patch_target_events * 4
    dense_cycle_identity_patch_bytes = patch_target_events * 8
    model_source_bindings = [
        {"file": str(path), "sha256": sha256(path)} for path in MODEL_SOURCE_FILES
    ]
    return {
        "schema": "local5_formal_phase_archive_scale_audit_v3",
        "status": "PASS_SCALE_AUDIT_REQUIRES_TEMPLATE_ARCHIVE",
        "evidence": "[prof]+[模型]",
        "formal_g0": "DENY",
        "input_bindings": bindings,
        "model_source_bindings": model_source_bindings,
        "formal_shape": {
            "windows": FORMAL_WINDOWS,
            "input_heads": FORMAL_GROUPS,
            "phases": FORMAL_PHASES,
            "acc32_scalars": FORMAL_ACC32,
        },
        "event_counts": counts,
        "storage_model": {
            "v4_event_bytes": V4_EVENT_BYTES,
            "common_event_scenarios": common_events,
            "v4_uncompressed_bytes_by_common_scenario": v4_bytes,
            "v4_uncompressed_gib_by_common_scenario": {
                key: value / (1 << 30) for key, value in v4_bytes.items()
            },
            "base_template_bytes_excluding_tile_patch_by_common_scenario": base_template_bytes_excluding_tile_patch,
            "base_template_gib_excluding_tile_patch_by_common_scenario": {
                key: value / (1 << 30)
                for key, value in base_template_bytes_excluding_tile_patch.items()
            },
            "tile_patch_capacity_envelope": {
                "patch_target_events_excluding_common": patch_target_events,
                "dense_cycle_only_uint32_bytes": dense_cycle_patch_bytes,
                "dense_cycle_only_uint32_gib": dense_cycle_patch_bytes / (1 << 30),
                "dense_cycle_identity_uint32_pair_bytes": dense_cycle_identity_patch_bytes,
                "dense_cycle_identity_uint32_pair_gib": dense_cycle_identity_patch_bytes
                / (1 << 30),
                "sparse_patch_bytes": None,
                "sparse_patch_density": None,
                "boundary": (
                    "完整patch容量未知；单窗canary必须实测cycle/identity差异密度。"
                    "本包络未计稀疏索引、offset或编码头。"
                ),
            },
            "delivery_aligned_cycle_bytes_excluding_common": delivery_aligned_cycle_bytes,
            "delivery_aligned_cycle_gib_excluding_common": delivery_aligned_cycle_bytes
            / (1 << 30),
            "base_event_reuse_factor_excluding_patch": expanded / template,
            "acc32_expected_actual_raw_bytes": FORMAL_ACC32 * 2 * 4,
            "acc32_expected_actual_raw_gib": FORMAL_ACC32 * 8 / (1 << 30),
        },
        "decision": {
            "v4_full_expansion": "DENY_FULL_RUN",
            "required": (
                "每input-head保存参数化fill/direct/execute模板；output tile保存"
                "identity/service-cycle patch；event identity与source/delivery索引对齐"
            ),
            "proof_obligation": [
                "证明OUT_DIM32下数据依赖不随weight数值变化",
                "保留output_tile参与identity-service后的cycle差异",
                "模板+tile patch展开后必须与逐tile事件完全同构",
                "数值Acc32 archive保持逐output-tile，不参与phase模板去重",
            ],
        },
        "boundary": [
            "这是由正式profile100计数驱动的容量模型，不是RTL周期或PPA",
            "base-template容量不含tile identity/service-cycle patch，不能作为完整实现预算",
            "one_event_per_common_phase是语义事件场景，不是v4 schema严格最小值",
            "未实现模板archive/parser前formal G0保持DENY",
            "压缩后磁盘大小取决于NPZ压缩率；本文报告未压缩内存下界",
        ],
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    counts = report["event_counts"]
    storage = report["storage_model"]
    text = f"""# Local5 Formal Phase Archive 规模审计

## 裁决

旧 v4 全展开 archive 暂停全量运行，`formal G0 = DENY`。本容量审计只证明当前
表示会把同一 input-head 的结构事件骨架按 output tile 重复展开；它不假设各 tile
的 identity/service-cycle patch 相同，也不对算法或 RTL 数值正确性作判断。

## 真实工作量

| 指标 | 数值 |
|---|---:|
| input-head group | {report['formal_shape']['input_heads']:,} |
| phase | {report['formal_shape']['phases']:,} |
| destination unique item | {counts['destination_unique_items']:,} |
| source product term | {counts['source_product_terms']:,} |
| multiplicity 展开 delivery/update | {counts['destination_deliveries']:,} |
| tile 展开 product term | {counts['tile_expanded_product_terms']:,} |
| tile 展开 delivery/update | {counts['tile_expanded_deliveries']:,} |
| v4 主流水事件（不含 prepare/drain） | {counts['v4_main_expanded_events_excluding_common']:,} |
| head-template 事件 | {counts['head_template_events']:,} |

    v4 的 `event_resource:uint8 + event_cycle:uint32 + event_identity:S64` 每条至少
{storage['v4_event_bytes']} byte。连同 phase 索引后，未压缩内存下界约
**{storage['v4_uncompressed_gib_by_common_scenario']['one_event_per_common_phase']:.2f} GiB**；
若 drain 为每 source 一次 vector read，则约
**{storage['v4_uncompressed_gib_by_common_scenario']['vector_drain_450']:.2f} GiB**，
若为 `450x32` scalar read，则约
**{storage['v4_uncompressed_gib_by_common_scenario']['scalar_drain_450x32']:.2f} GiB**。
这还没有计入 Python/NumPy 临时副本和 JSON ledger。

## 改造方向

1. 每个 input head 保存一份参数化 fill/direct/execute phase 模板；H 个 output tile
   保存 identity/service-cycle patch。仅 base event 的复用因子为
   **{storage['base_event_reuse_factor_excluding_patch']:.2f} 倍**，不是端到端存储缩减。
2. `S64` identity 改为与正式 ordered-term/source offset 对齐的结构化整数索引；
   每个 common phase 恰有一个语义事件的场景下，**仅 base-template** 约
   **{storage['base_template_gib_excluding_tile_patch_by_common_scenario']['one_event_per_common_phase']:.2f} GiB**。
   该数字不含 tile patch，不能作为完整 archive 容量。
3. 若进一步按 delivery/source 顺序隐式恢复 identity，只保存 direct/execute 与五类
   record cycle，排除 common phase 和 tile patch 的 base 模型下界约
   **{storage['delivery_aligned_cycle_gib_excluding_common']:.2f} GiB**。
4. Acc32 数值 archive 仍逐 tile 保存 expected/actual；其 raw payload 约
   **{storage['acc32_expected_actual_raw_gib']:.2f} GiB**，不能用模板去重替代。
5. 若全部主事件都需要 `uint32 cycle` patch，单 cycle 字段约
   **{storage['tile_patch_capacity_envelope']['dense_cycle_only_uint32_gib']:.2f} GiB**；
   cycle 与 identity 各一个 `uint32` 时约
   **{storage['tile_patch_capacity_envelope']['dense_cycle_identity_uint32_pair_gib']:.2f} GiB**，
   且仍未计索引/offset。实际稀疏 patch 密度必须由 canary 实测。

## 证据边界

- `[prof]`：unique item、product term、delivery、active record 与 head 数来自冻结
  profile100；`ACC_WRITE` 的容量模型按 multiplicity 展开的 delivery/update 计数，
  不能按 unique item 计数。
- `[模型]`：事件展开公式和字节数；不是 RTL 周期、功耗或 ASIC PPA。
- prepare/drain 当前仅有每 common phase 一个语义事件、每 source vector 和每
  source×channel scalar 三个场景；v4 schema 本身允许空事件，准确事件数必须由
  单窗 RTL canary 冻结。
- 模板同构、结构化 identity parser 和单窗 RTL canary 未完成前，不生成 admission。
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("results/local5_fullres_bb1e4_joint_heads_profile100_20260809"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    bindings, counts = load_workload(args.profile.resolve())
    report = build_report(bindings, counts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "phase_archive_scale_audit.json"
    md_path = args.output_dir / "phase_archive_scale_audit.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(md_path, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
