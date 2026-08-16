#!/usr/bin/env python3
"""评估Motion时间增量score与等价score descriptor合并。"""

from __future__ import annotations

import argparse
import base64
import json
import math
import zlib
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .evidence_provenance import sha256_file
except ImportError:
    from evidence_provenance import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def file_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"provenance文件不存在: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def profile_contract(profile: dict[str, Any], profile_path: Path) -> dict[str, Any]:
    protocol = profile.get("eval_protocol") or {}
    records = profile.get("summary", {}).get("h60_records") or []
    if not records:
        raise ValueError("profile缺少h60_records")
    temporal_tokens = int(records[0].get("tokens", -1))
    if temporal_tokens <= 0 or any(
        int(record.get("tokens", -1)) != temporal_tokens for record in records
    ):
        raise ValueError("h60_records的temporal token不一致")
    if int(protocol.get("tokens_per_window", -1)) != temporal_tokens:
        raise ValueError("eval_protocol与h60_records的temporal token不一致")
    samples = int(profile.get("samples", -1))
    if samples <= 0:
        raise ValueError("profile samples必须为正数")
    artifact = profile.get("artifact_identity") or {}
    required_artifact = (
        "config_path",
        "config_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
    )
    if any(not artifact.get(key) for key in required_artifact):
        raise ValueError("profile缺少config/checkpoint身份")
    return {
        "profile": str(profile_path.resolve()),
        "experiment": str(profile.get("experiment", "")),
        "samples": samples,
        "h60_records": len(records),
        "resolution": protocol.get("resolution"),
        "crop": protocol.get("crop"),
        "window_size": protocol.get("window_size"),
        "temporal_tokens": temporal_tokens,
        "bn_policy": protocol.get("bn_policy"),
        "config_path": artifact["config_path"],
        "config_sha256": artifact["config_sha256"],
        "checkpoint_path": artifact["checkpoint_path"],
        "checkpoint_sha256": artifact["checkpoint_sha256"],
    }


def decode_i16_trace(encoded: dict[str, Any]) -> np.ndarray:
    if encoded.get("codec") != "zlib_base64" or encoded.get("dtype") != "int16_le":
        raise ValueError("只支持zlib_base64/int16_le ordered trace")
    raw = zlib.decompress(base64.b64decode(encoded["data"]))
    values = np.frombuffer(raw, dtype="<i2")
    shape = tuple(int(value) for value in encoded["shape"])
    if values.size != math.prod(shape):
        raise ValueError("ordered trace shape不守恒")
    return values.reshape(shape)


def h67_score_code(
    q_count: np.ndarray,
    k_count: np.ndarray,
    overlap: np.ndarray,
    motion: np.ndarray,
) -> np.ndarray:
    same_zero = 32 - q_count - k_count + overlap
    integer = 4 * overlap + motion + same_zero // 16
    remainder = same_zero % 16
    increment = (remainder > 8) | ((remainder == 8) & ((integer & 1) != 0))
    return integer + increment


def active_pair_detail(records: list[dict[str, Any]]) -> dict[str, int | float]:
    totals = {
        "pair_total": 0,
        "pair_both_active": 0,
        "pair_both_active_equal": 0,
        "pair_one_active": 0,
        "pair_one_active_equal": 0,
        "pair_both_kzero": 0,
        "pair_both_kzero_equal": 0,
    }
    for record in records:
        k_count = decode_i16_trace(record["pair_k_count_ordered_trace"])
        q_count = decode_i16_trace(record["pair_q_count_ordered_trace"])
        overlap = decode_i16_trace(record["pair_overlap_ordered_trace"])
        motion = decode_i16_trace(record["pair_motion_ordered_trace"])
        score0 = h67_score_code(q_count[0], k_count[0], overlap[0], motion)
        score1 = h67_score_code(q_count[1], k_count[1], overlap[1], motion)
        equal = score0 == score1
        active0 = k_count[0] != 0
        active1 = k_count[1] != 0
        both_active = active0 & active1
        one_active = active0 ^ active1
        both_kzero = ~active0 & ~active1
        totals["pair_total"] += int(equal.size)
        totals["pair_both_active"] += int(both_active.sum())
        totals["pair_both_active_equal"] += int((both_active & equal).sum())
        totals["pair_one_active"] += int(one_active.sum())
        totals["pair_one_active_equal"] += int((one_active & equal).sum())
        totals["pair_both_kzero"] += int(both_kzero.sum())
        totals["pair_both_kzero_equal"] += int((both_kzero & equal).sum())
    both_active = totals["pair_both_active"]
    totals["pair_both_active_equal_rate"] = (
        totals["pair_both_active_equal"] / both_active if both_active else 0.0
    )
    return totals


def evaluate_temporal_equivalence(
    stats: dict[str, Any],
    active_detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total = int(stats["pair_total"])
    empty = int(stats["pair_empty"])
    equal = int(stats["pair_score_equal_h67"])
    histogram = [int(value) for value in stats["update_histogram"]]
    if sum(histogram) != total:
        raise ValueError("update histogram总数与pair_total不一致")
    if equal < empty:
        raise ValueError("empty pair必须属于score equal集合")
    active = total - empty
    unequal = total - equal
    active_equal = equal - empty
    baseline_active_descriptors = 2 * active
    compressed_active_descriptors = active + unequal
    baseline_lane_work = 64 * active
    delta_lane_work = 32 * active + sum(
        update * count for update, count in enumerate(histogram)
    )

    widths = []
    for lanes_per_cycle in (4, 8, 16, 32):
        baseline_cycles = 2 * math.ceil(32 / lanes_per_cycle) * active
        delta_cycles = math.ceil(32 / lanes_per_cycle) * active + sum(
            math.ceil(update / lanes_per_cycle) * count
            for update, count in enumerate(histogram)
            if update > 0
        )
        widths.append(
            {
                "lanes_per_cycle": lanes_per_cycle,
                "baseline_cycles": baseline_cycles,
                "delta_cycles": delta_cycles,
                "speedup": baseline_cycles / delta_cycles,
                "cycle_reduction": 1 - delta_cycles / baseline_cycles,
            }
        )

    result = {
        "pair_total": total,
        "pair_empty": empty,
        "pair_active": active,
        "pair_score_equal": equal,
        "pair_score_unequal": unequal,
        "pair_active_score_equal": active_equal,
        "active_score_equal_rate": active_equal / active,
        "baseline_active_score_descriptors": baseline_active_descriptors,
        "compressed_active_score_descriptors": compressed_active_descriptors,
        "active_descriptor_reduction": 1 - compressed_active_descriptors / baseline_active_descriptors,
        "all_descriptor_reduction": 1 - (total + unequal) / (2 * total),
        "baseline_lane_work": baseline_lane_work,
        "delta_lane_work": delta_lane_work,
        "lane_work_reduction": 1 - delta_lane_work / baseline_lane_work,
        "update_histogram": histogram,
        "width_results": widths,
    }
    if active_detail is not None:
        if int(active_detail["pair_total"]) != total:
            raise ValueError("active类别trace与汇总pair_total不一致")
        both_active = int(active_detail["pair_both_active"])
        one_active = int(active_detail["pair_one_active"])
        equal_both_active = int(active_detail["pair_both_active_equal"])
        baseline_active_entries = 2 * both_active + one_active
        quotient_active_entries = baseline_active_entries - equal_both_active
        fold_classes = int(stats["row_kzero_fold_classes_sum_h67"])
        baseline_exp = baseline_active_entries + fold_classes
        quotient_exp = quotient_active_entries + fold_classes
        result["active_pair_detail"] = active_detail
        result["baseline_scs_active_entries"] = baseline_active_entries
        result["quotient_scs_active_entries"] = quotient_active_entries
        result["scs_active_entry_reduction"] = (
            1 - quotient_active_entries / baseline_active_entries
        )
        result["baseline_scs_exp_transactions_model"] = baseline_exp
        result["quotient_scs_exp_transactions_model"] = quotient_exp
        result["scs_exp_transaction_reduction_model"] = 1 - quotient_exp / baseline_exp
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(
            "/root/private_data/work/sdformer_codex/SDformer/"
            "neuron_experiments/H9_bipolar_self_attention/results/"
            "h67_ep19_ttb_delta_cycle_v2_profile100_20260713/"
            "nts11_hardware_p0_profile.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/motion_temporal_equivalence_20260803"),
    )
    parser.add_argument("--watcher", type=Path, required=True)
    parser.add_argument("--test-log", type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    source = profile_contract(profile, args.profile)
    stats = profile["summary"]["binary_temporal_pairs"]
    detail = active_pair_detail(profile["summary"]["h60_records"])
    analysis = evaluate_temporal_equivalence(stats, detail)
    config_binding = file_binding(Path(source["config_path"]))
    checkpoint_binding = file_binding(Path(source["checkpoint_path"]))
    if config_binding["sha256"] != source["config_sha256"]:
        raise ValueError("profile记录的config SHA与文件不一致")
    if checkpoint_binding["sha256"] != source["checkpoint_sha256"]:
        raise ValueError("profile记录的checkpoint SHA与文件不一致")
    result = {
        "schema": "motion_temporal_equivalence_v2",
        "status": "PROFILE_MODEL_COMPLETE",
        "evidence": "[prof-ordered]+[整数重算模型]；尚无anchor+delta逐元素miter",
        "profile": str(args.profile.resolve()),
        "source": source,
        "provenance": {
            "profile": file_binding(args.profile),
            "config": config_binding,
            "checkpoint": checkpoint_binding,
            "analyzer": file_binding(Path(__file__)),
            "validator": file_binding(ROOT / "scripts/evidence_provenance.py"),
            "watcher": file_binding(args.watcher),
            "test_log": file_binding(args.test_log),
            "tests": [
                file_binding(ROOT / "tests/test_new_dual_line_architecture_models.py"),
                file_binding(ROOT / "tests/test_model_motion_reversible_quotient_bundle.py"),
                file_binding(ROOT / "tests/test_motion_model_provenance.py"),
            ],
        },
        "analysis": analysis,
        "contract": {
            "first_score": "完整32-lane Q/K贡献",
            "second_score": "只在(Q0 xor Q1) or (K0 xor K1)位上计算新旧贡献差",
            "motion_term": "K0 xor K1 popcount在两个时间score间共享",
            "equal_coalescing": "score相等时保存一个score class加2-bit temporal mask",
            "unequal_fallback": "保存两个独立score descriptor",
            "numeric": "先在整数numerator域更新，再执行同一RNE /16",
        },
        "limits": [
            "cycle模型只覆盖score lane issue，不含compactor、状态读取、SCS、projection和SRAM延迟。",
            "32-lane全并行基线下几乎没有周期收益，主要收益是开关活动和descriptor流量。",
            (
                f"结果来自resolution={source['resolution']}、crop={source['crop']}、"
                f"window={source['window_size']}、T={source['temporal_tokens']}、"
                f"samples={source['samples']}的已绑定profile；仍不是RTL cycle、SAIF或PPA。"
            ),
            "必须由RTL逐元素证明增量numerator、RNE score和合并后的SCS multiplicity完全一致。",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Motion时间等价压缩Scoreflow评估",
        "",
        "## 结论",
        "",
        f"H67 profile100中，两时刻K均有效的时间对有"
        f"`{analysis['active_pair_detail']['pair_both_active_equal_rate']:.2%}`产生相同Q7 score。"
        "按归一化域合并、投影前展开，可在不改变Shiftmax分母或K输出的前提下把"
        f"SCS active entry减少`{analysis['scs_active_entry_reduction']:.2%}`，"
        f"全SCS指数事务模型减少`{analysis['scs_exp_transaction_reduction_model']:.2%}`。"
        f"时间增量score的逐lane工作另减少`{analysis['lane_work_reduction']:.2%}`。",
        "",
        "证据等级为 **[prof-ordered]+[整数重算模型]**；尚无anchor+delta逐元素miter，也不是RTL或PPA。",
        "",
        "## Profile结果",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| temporal pair | {analysis['pair_total']} |",
        f"| 非空 pair | {analysis['pair_active']} |",
        f"| 非空且score相等（上游） | {analysis['pair_active_score_equal']} |",
        f"| 非空score相等率（上游） | {analysis['active_score_equal_rate']:.2%} |",
        f"| 双K有效pair | {analysis['active_pair_detail']['pair_both_active']} |",
        f"| 双K有效且score相等 | {analysis['active_pair_detail']['pair_both_active_equal']} |",
        f"| 双K有效score相等率 | {analysis['active_pair_detail']['pair_both_active_equal_rate']:.2%} |",
        f"| SCS active entry基线 | {analysis['baseline_scs_active_entries']} |",
        f"| 商流SCS active entry | {analysis['quotient_scs_active_entries']} |",
        f"| SCS active entry降低 | {analysis['scs_active_entry_reduction']:.2%} |",
        f"| 全SCS指数事务模型降低 | {analysis['scs_exp_transaction_reduction_model']:.2%} |",
        f"| 逐lane score工作降低 | {analysis['lane_work_reduction']:.2%} |",
        "",
        "## 折叠lane周期模型",
        "",
        "| 每拍lane | 稠密双score周期 | 增量周期 | 加速 |",
        "|---:|---:|---:|---:|",
    ]
    for row in analysis["width_results"]:
        lines.append(
            f"| {row['lanes_per_cycle']} | {row['baseline_cycles']} | "
            f"{row['delta_cycles']} | {row['speedup']:.3f}x |"
        )
    lines += [
        "",
        "## 微架构候选",
        "",
        "该结果对应TESC（Temporal-Equivalence Score Coalescer）候选：16-lane"
        "折叠score engine先计算t0，再由toggle compactor发射t1差分；整数RNE后若score"
        "相同，只向SCS写一个`{class, temporal_mask=2'b11}`，否则写两个descriptor。"
        "SCS按mask popcount加入multiplicity；只有双K有效且score相同才减少active entry，"
        "projection前按active mask重新展开两个K destination，"
        "因此不删除token、不改变分母，也不使用预测。",
        "",
        "## 证据边界",
        "",
    ]
    lines.extend(f"- {item}" for item in result["limits"])
    lines += [
        "",
        "## 架构晋级判断",
        "",
        "TESC可进入最小RTL：必须比较32-lane双score、16-lane稠密折叠和16-lane增量"
        "三种同接口结构。只有post-OpenROAD面积归一吞吐至少1.15x，或真实SAIF的"
        "score+SCS能量至少降低15%，才作为Motion独立论文主贡献；否则仅保留为"
        "scoreflow压缩子机制。",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
