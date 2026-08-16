#!/usr/bin/env python3
"""基于正式 Local5 manifest 与 RTL 探针生成 G0 档案容量/运行预算。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


TOKENS = 450
OUT_DIM = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_time(path: Path) -> dict[str, float]:
    rows: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        rows[key] = float(value)
    return rows


def gib(value: int) -> float:
    return value / (1 << 30)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--canary-report", type=Path, required=True)
    parser.add_argument("--verilator-time", type=Path, required=True)
    parser.add_argument("--icarus-time", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.profile / "ordered_term_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canary = json.loads(args.canary_report.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "et3_ordered_term_trace_v2"
        or (manifest.get("qualification") or {}).get("qualified") is not True
        or canary.get("status")
        != "PASS_INTEGRATED_CROSS_HEAD_CANARY_NOT_G0"
    ):
        raise ValueError("正式manifest或集成canary未准入")
    by_window: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for group in manifest["groups"]:
        key = (
            int(group["sample"]),
            int(group["stage"]),
            int(group["block"]),
            int(group["window"]),
        )
        by_window[key].append(int(group["head"]))
    head_histogram: Counter[int] = Counter()
    per_sample_final: Counter[int] = Counter()
    per_sample_tasks: Counter[int] = Counter()
    for (sample, _, _, _), heads in by_window.items():
        if sorted(heads) != list(range(len(heads))):
            raise ValueError("formal joint window head覆盖不完整")
        count = len(heads)
        head_histogram[count] += 1
        per_sample_final[sample] += count * TOKENS * OUT_DIM
        per_sample_tasks[sample] += count * count
    if (
        len(by_window) != 1200
        or len(manifest["groups"]) != 13800
        or set(per_sample_final) != set(range(100))
    ):
        raise ValueError("formal window/group/sample计数不匹配")
    hxh_tasks = sum(heads * heads * windows for heads, windows in head_histogram.items())
    final_scalars = sum(per_sample_final.values())
    partial_scalars = hxh_tasks * TOKENS * OUT_DIM
    if hxh_tasks != 210600 or final_scalars != 198720000:
        raise ValueError("formal HxH/Acc32标量计数与冻结合同不一致")

    verilator_time = parse_time(args.verilator_time)
    icarus_time = parse_time(args.icarus_time)
    canary_cycles = int(canary["simulators"][0]["cycles"])
    stage0_equivalent_windows = hxh_tasks / 9.0
    estimated_cycles = canary_cycles * stage0_equivalent_windows
    sample_shard_scalars = sorted(set(per_sample_final.values()))
    if sample_shard_scalars != [final_scalars // 100]:
        raise ValueError("每sample formal final Acc32规模不一致")
    per_sample_bytes = sample_shard_scalars[0] * 4

    output = {
        "schema": "local5_formal_archive_capacity_budget_v1",
        "status": "PASS_BUDGET_NOT_G0",
        "evidence": "[prof]+[rtl校准模型]",
        "formal_g0": "DENY",
        "input": {
            "manifest_sha256": sha256(manifest_path),
            "canary_report_sha256": sha256(args.canary_report),
            "verilator_time_sha256": sha256(args.verilator_time),
            "icarus_time_sha256": sha256(args.icarus_time),
        },
        "formal_workload": {
            "joint_windows": len(by_window),
            "input_head_groups": len(manifest["groups"]),
            "window_head_histogram": {
                str(heads): windows
                for heads, windows in sorted(head_histogram.items())
            },
            "hxh_tasks": hxh_tasks,
            "final_acc32_scalars": final_scalars,
            "raw_partial_acc32_scalars_if_materialized": partial_scalars,
            "frozen_phase_records": 462600,
        },
        "storage": {
            "one_final_archive_bytes": final_scalars * 4,
            "one_final_archive_gib": gib(final_scalars * 4),
            "expected_plus_one_actual_gib": gib(final_scalars * 4 * 2),
            "expected_plus_two_actual_gib": gib(final_scalars * 4 * 3),
            "one_raw_partial_archive_gib": gib(partial_scalars * 4),
            "raw_partial_avoided_by_integrated_final_gib": gib(
                (partial_scalars - final_scalars) * 4
            ),
            "per_sample_final_archive_bytes": per_sample_bytes,
            "per_sample_expected_plus_verilator_mib": per_sample_bytes * 2 / (1 << 20),
            "recommended_shards": 100,
        },
        "runtime_probe": {
            "canary_heads": 3,
            "canary_hxh_tasks": 9,
            "canary_cycles": canary_cycles,
            "verilator_wall_seconds": verilator_time["wall_seconds"],
            "icarus_wall_seconds": icarus_time["wall_seconds"],
            "stage0_equivalent_windows_by_hxh": stage0_equivalent_windows,
            "full_verilator_wall_hours_h2_model": (
                verilator_time["wall_seconds"] * stage0_equivalent_windows / 3600
            ),
            "full_icarus_wall_hours_h2_model": (
                icarus_time["wall_seconds"] * stage0_equivalent_windows / 3600
            ),
            "full_cycle_h2_model": estimated_cycles,
            "model_warning": (
                "按H^2任务数从单个stage0窗口外推；未计stage密度、缓存、编译、"
                "分片启动与I/O差异，不是部署吞吐或RTL实测全量时间"
            ),
        },
        "execution_decision": {
            "full_numeric_actual": "Verilator，100个sample分片，可断点续跑",
            "cross_simulator": (
                "Icarus只跑分层canary；已有stage0逐值一致，后续补stage1/2/3"
            ),
            "archive": (
                "只物化output_tile->source->out最终Acc32；禁止物化3.03B partial"
            ),
            "admission": (
                "每分片先只读miter与SHA，再由总receipt归并；全部mismatch=0前DENY"
            ),
        },
        "forbidden_claims": [
            "不是full RTL回放实测时长",
            "不是EREP或full-encoder性能",
            "不是ASIC PPA",
            "不是formal G0 admission",
        ],
    }
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
