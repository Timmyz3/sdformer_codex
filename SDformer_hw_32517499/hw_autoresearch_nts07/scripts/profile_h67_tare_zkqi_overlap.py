#!/usr/bin/env python3
"""Profile TARE residual execution after exact TTB8-ZKQI zero-K filtering."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT / "results/h67_fullres_ep30_t450_all12_bit_trace_20260805/manifest.json"
)
DEFAULT_STRONG_BASELINE = ROOT / "results/h67_zkqi_threeway_20260809/report.json"
DEFAULT_OUTPUT = ROOT / "results/h67_tare_zkqi_overlap_t450_20260810"
WIDTHS = (2, 4, 8, 16, 32)
EXPECTED_NAMES = (
    "S0.B0.attn",
    "S0.B1.attn",
    "S1.B0.attn",
    "S1.B1.attn",
    "S2.B0.attn",
    "S2.B1.attn",
    "S2.B2.attn",
    "S2.B3.attn",
    "S2.B4.attn",
    "S2.B5.attn",
    "S3.B0.attn",
    "S3.B1.attn",
)
BLOCK_RE = re.compile(r"S(?P<stage>\d+)\.B(?P<block>\d+)\.attn")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_bits(payload: np.lib.npyio.NpzFile, prefix: str) -> np.ndarray:
    shape = tuple(int(value) for value in payload[f"{prefix}_shape"])
    count = int(np.prod(shape))
    bits = np.unpackbits(
        payload[f"{prefix}_bits_packed"], bitorder="little"
    )[:count]
    return bits.reshape(shape).astype(np.bool_, copy=False)


def rne_div16_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if np.any(values < 0):
        raise ValueError("RNE input must be non-negative")
    quotient, remainder = np.divmod(values, 16)
    increment = (remainder > 8) | ((remainder == 8) & ((quotient & 1) != 0))
    return quotient + increment.astype(np.int64)


def candidate_metrics(update_hist: np.ndarray, active_pairs: int, width: int) -> dict[str, Any]:
    """Return a lane-only score-engine model against two parallel 32-lane engines."""

    hist = np.asarray(update_hist, dtype=np.int64)
    if hist.shape != (33,) or active_pairs != int(hist.sum()):
        raise ValueError("active update histogram contract mismatch")
    if width <= 0 or width > 32:
        raise ValueError("residual width must be in [1, 32]")

    zero = int(hist[0])
    sparse = int(hist[1 : width + 1].sum())
    dense = int(hist[width + 1 :].sum())
    sparse_lane_work = sum((32 + count) * int(hist[count]) for count in range(1, width + 1))
    score_lane_work = 32 * zero + sparse_lane_work + 64 * dense
    baseline_lane_work = 64 * active_pairs
    serial_cycles = active_pairs + dense
    score_throughput_ratio = active_pairs / serial_cycles if serial_cycles else 1.0
    lane_area_proxy = 32 + width
    area_normalized_score_throughput = score_throughput_ratio * 64 / lane_area_proxy
    return {
        "residual_width": width,
        "threshold": width,
        "zero_pairs": zero,
        "sparse_pairs": sparse,
        "dense_fallback_pairs": dense,
        "dense_fallback_ratio": dense / active_pairs if active_pairs else 0.0,
        "baseline_score_lane_work": baseline_lane_work,
        "candidate_score_lane_work": score_lane_work,
        "score_lane_work_reduction": (
            1.0 - score_lane_work / baseline_lane_work if baseline_lane_work else 0.0
        ),
        "candidate_score_cycles": serial_cycles,
        "score_throughput_ratio_vs_two_direct32": score_throughput_ratio,
        "lane_area_proxy": lane_area_proxy,
        "lane_only_area_normalized_score_throughput": area_normalized_score_throughput,
        "model_exclusions": [
            "update detector/XOR/popcount",
            "priority compactor and residual control",
            "RNE/output skid",
            "TTB8 scanner/SCS/backend/SRAM/frequency/power",
        ],
    }


def _blank_bucket() -> dict[str, Any]:
    return {
        "pairs": 0,
        "active_pairs": 0,
        "active_score_equal_pairs": 0,
        "active_update_sum": 0,
        "active_update_histogram": np.zeros(33, dtype=np.int64),
    }


def _merge_bucket(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in ("pairs", "active_pairs", "active_score_equal_pairs", "active_update_sum"):
        target[key] += int(source[key])
    target["active_update_histogram"] += source["active_update_histogram"]


def profile_record(record: dict[str, Any]) -> dict[str, Any]:
    path = Path(record["file"])
    observed_sha = sha256(path)
    if observed_sha != record["sha256"]:
        raise ValueError(f"trace SHA256 mismatch: {path}")
    with np.load(path) as payload:
        q = unpack_bits(payload, "q")
        k = unpack_bits(payload, "k")
    if q.shape != k.shape or q.ndim != 5 or q.shape[0] != 2 or q.shape[-1] != 32:
        raise ValueError(f"illegal Q/K shape: q={q.shape}, k={k.shape}")
    if q.shape[1] != 1 or q.shape[3] != 225:
        raise ValueError(f"expected one full-resolution T450 window: {q.shape}")

    q_count = q.sum(axis=-1, dtype=np.int64)
    k_count = k.sum(axis=-1, dtype=np.int64)
    overlap = (q & k).sum(axis=-1, dtype=np.int64)
    motion = (k[0] ^ k[1]).sum(axis=-1, dtype=np.int64)

    anchor_raw = 65 * overlap[0] + 32 - q_count[0] - k_count[0] + 16 * motion
    direct_raw = 65 * overlap[1] + 32 - q_count[1] - k_count[1] + 16 * motion
    lane_raw0 = np.where(q[0] & k[0], 64, np.where((~q[0]) & (~k[0]), 1, 0))
    lane_raw1 = np.where(q[1] & k[1], 64, np.where((~q[1]) & (~k[1]), 1, 0))
    delta_raw = (lane_raw1 - lane_raw0).sum(axis=-1, dtype=np.int64)
    reconstructed_raw = anchor_raw + delta_raw
    raw_mismatches = int(np.count_nonzero(reconstructed_raw != direct_raw))
    reconstructed_q7 = rne_div16_array(reconstructed_raw)
    direct_q7 = rne_div16_array(direct_raw)
    q7_mismatches = int(np.count_nonzero(reconstructed_q7 != direct_q7))

    score0 = rne_div16_array(anchor_raw)
    score1 = direct_q7
    active = (k_count != 0).any(axis=0)
    update_count = ((q[0] ^ q[1]) | (k[0] ^ k[1])).sum(axis=-1, dtype=np.int64)
    active_updates = update_count[active]
    update_hist = np.bincount(active_updates, minlength=33).astype(np.int64)
    score_equal = score0 == score1
    equal_hist = np.bincount(update_count[active & score_equal], minlength=33).astype(np.int64)

    result = {
        "name": record["name"],
        "source_file": str(path),
        "source_sha256": observed_sha,
        "windows": int(q.shape[1]),
        "heads": int(q.shape[2]),
        "pairs": int(active.size),
        "active_pairs": int(active.sum()),
        "active_score_equal_pairs": int((active & score_equal).sum()),
        "active_update_sum": int(active_updates.sum()),
        "active_update_histogram": update_hist,
        "active_score_equal_by_update": equal_hist,
        "raw_mismatches": raw_mismatches,
        "q7_mismatches": q7_mismatches,
    }
    return result


def _json_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    active = int(bucket["active_pairs"])
    result = {
        "pairs": int(bucket["pairs"]),
        "active_pairs": active,
        "active_pair_ratio": active / int(bucket["pairs"]) if bucket["pairs"] else 0.0,
        "active_score_equal_pairs": int(bucket["active_score_equal_pairs"]),
        "active_score_equal_ratio": (
            int(bucket["active_score_equal_pairs"]) / active if active else 0.0
        ),
        "active_update_mean": int(bucket["active_update_sum"]) / active if active else 0.0,
        "active_update_histogram": [int(value) for value in bucket["active_update_histogram"]],
    }
    return result


def analyze(manifest_path: Path, strong_baseline_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    names = tuple(record.get("name") for record in records)
    if names != EXPECTED_NAMES:
        raise ValueError(f"expected all 12 H67 blocks in order, observed {names}")
    coverage = manifest.get("coverage", {})
    if (
        manifest.get("sample_limit") != 1
        or manifest.get("first_block_only") is not False
        or coverage.get("four_stage_complete") is not True
        or coverage.get("record_count") != 12
    ):
        raise ValueError("manifest is not the frozen sample0/all-12/full-stage trace")

    baseline = json.loads(strong_baseline_path.read_text(encoding="utf-8"))
    if int(baseline["coverage"]["rows"]) != 138:
        raise ValueError("strong baseline row coverage drifted")
    no_stall = baseline["cycles"]["modes"]["0"]
    ttb8_e2e_cycles = int(no_stall["ttb8_zkqi_e2e_cycles"])

    total = _blank_bucket()
    stages: dict[str, dict[str, Any]] = {}
    blocks: list[dict[str, Any]] = []
    raw_mismatches = 0
    q7_mismatches = 0
    equal_by_update = np.zeros(33, dtype=np.int64)
    source_records = []

    for record in records:
        match = BLOCK_RE.fullmatch(record["name"])
        if match is None:
            raise ValueError(f"illegal block name: {record['name']}")
        stage_name = f"S{match.group('stage')}"
        stage = stages.setdefault(stage_name, _blank_bucket())
        profiled = profile_record(record)
        _merge_bucket(total, profiled)
        _merge_bucket(stage, profiled)
        raw_mismatches += int(profiled["raw_mismatches"])
        q7_mismatches += int(profiled["q7_mismatches"])
        equal_by_update += profiled["active_score_equal_by_update"]
        block_json = _json_bucket(profiled)
        block_json.update(
            {
                "name": profiled["name"],
                "heads": profiled["heads"],
                "tare16_dense_fallback_pairs": int(profiled["active_update_histogram"][17:].sum()),
            }
        )
        blocks.append(block_json)
        source_records.append(
            {
                "name": profiled["name"],
                "file": profiled["source_file"],
                "sha256": profiled["source_sha256"],
            }
        )

    total_json = _json_bucket(total)
    active_pairs = total_json["active_pairs"]
    if total_json["pairs"] != 31050 or active_pairs != 14554:
        raise ValueError(
            "ZKQI identity calibration failed: "
            f"pairs={total_json['pairs']}, active_pairs={active_pairs}"
        )
    if raw_mismatches or q7_mismatches:
        raise ValueError(f"TARE algebra mismatch: raw={raw_mismatches}, q7={q7_mismatches}")

    candidates = [
        candidate_metrics(total["active_update_histogram"], active_pairs, width)
        for width in WIDTHS
    ]
    for candidate in candidates:
        dense = candidate["dense_fallback_pairs"]
        candidate["ttb8_combined_ideal_service_model"] = {
            "optimistic_fully_hidden_cycles": ttb8_e2e_cycles,
            "ideal_serial_fallback_cycles": ttb8_e2e_cycles + dense,
            "ideal_serial_speed_ratio_vs_ttb8_zkqi": ttb8_e2e_cycles / (ttb8_e2e_cycles + dense),
            "excluded_from_serial_model": [
                "detector and 32-to-W compactor latency",
                "atomic T0/T1 packet and pipeline bubbles",
                "backpressure propagation and frequency change",
            ],
        }
        candidate["admission_checks"] = {
            "score_lane_work_reduction_ge_10pct": candidate["score_lane_work_reduction"] >= 0.10,
            "lane_only_area_normalized_score_throughput_gt_1": candidate[
                "lane_only_area_normalized_score_throughput"
            ] > 1.0,
        }
        candidate["conditionally_admitted"] = all(candidate["admission_checks"].values())

    admitted = [candidate for candidate in candidates if candidate["conditionally_admitted"]]
    selected = max(
        admitted,
        key=lambda candidate: candidate["lane_only_area_normalized_score_throughput"],
        default=None,
    )
    selected_width = selected["residual_width"] if selected else None
    rtl_screen_widths = [8, 16] if selected_width is not None else []

    return {
        "schema": "h67_tare_zkqi_overlap_t450_v1",
        "status": "CONDITIONAL_ADMIT" if selected else "REJECT",
        "evidence_levels": {
            "trace_and_algebra": "[prof]",
            "cycle_and_lane_area": "[模型]",
            "integrated_rtl": "[待验证]",
            "asic_ppa": "[待验证]",
        },
        "scope": {
            "line": "H67 Motion",
            "resolution": [480, 640],
            "window": [2, 15, 15],
            "temporal_tokens": 450,
            "spatial_pairs_per_head_row": 225,
            "sample_count": 1,
            "attention_blocks": 12,
            "head_rows": 138,
            "raw_lane_identity_available": True,
            "cross_sample_claim_allowed": False,
        },
        "identity_calibration": {
            "pairs": total_json["pairs"],
            "zkqi_active_pairs": active_pairs,
            "expected_zkqi_active_pairs": 14554,
            "aggregate_active_pair_count_match": active_pairs == 14554,
            "pairwise_active_bitmap_compared": False,
            "active_score_equal_pairs": total_json["active_score_equal_pairs"],
            "active_score_equal_ratio": total_json["active_score_equal_ratio"],
        },
        "exactness": {
            "raw16_mismatches": raw_mismatches,
            "q7_mismatches": q7_mismatches,
            "equation": "raw_target = raw_anchor + sum(updated_lane_raw_target - updated_lane_raw_anchor)",
            "rounding": "single final RNE divide-by-16",
        },
        "workload": {
            "total": total_json,
            "active_score_equal_by_update": [int(value) for value in equal_by_update],
            "stages": {name: _json_bucket(bucket) for name, bucket in stages.items()},
            "blocks": blocks,
        },
        "strong_baseline": {
            "name": "TTB8-ZKQI with two parallel Direct32 score engines",
            "report": str(strong_baseline_path),
            "report_sha256": sha256(strong_baseline_path),
            "no_stall_preload_inclusive_cycles": ttb8_e2e_cycles,
            "score_lane_proxy": 64,
        },
        "candidates": candidates,
        "selection": {
            "selected_width": selected_width,
            "rtl_screen_widths": rtl_screen_widths,
            "width_frozen": False,
            "decision": (
                "TARE-W8/W16仅作为active-score前端面积/能耗候选进入A/B/C RTL筛选；W16为首选点但尚未冻结，不声称周期加速"
                if selected_width == 16
                else "没有候选通过最低准入门槛"
            ),
            "next_required_evidence": [
                "将parameterized TARE-W8/W16接在TTB8 active-pair出口，保持ZKQI三类常量注入",
                "同接口比较双Direct32、TARE-W8、TARE-W16的Icarus/Verilator bit-exact和随机反压周期",
                "候选原子输出T0/T1 score packet，residual16 signed delta至少12 bit",
                "加入detector/compactor/RNE后做同约束Yosys/OpenROAD开放代理",
                "若全前端面积归一吞吐不大于1或总执行EDP无净收益则否决",
            ],
        },
        "traffic_and_control_boundary": {
            "qk_sram_traffic_reduction": 0.0,
            "reason": "update detector仍需读取驻留的Q0/Q1/K0/K1；TARE只减少score lane工作",
            "new_state": "update mask/class、residual delta、dense replay控制与弹性输出",
            "ttb8_scs_backend_cycle_reduction": 0,
        },
        "limitations": [
            "raw temporal lane identity仅覆盖sample0一个window的全部12个block",
            "canonical profile100只有计数与ordered metadata，不能恢复Q0/Q1/K0/K1 update mask",
            "lane-only面积归一吞吐没有计入detector、compactor、控制、SRAM和频率",
            "TARE dense fallback在保守串行边界内只会增加TTB8周期，不会减少SCS/backend周期",
            "没有DC/STA/SAIF/PTPX证据",
        ],
        "source_receipts": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "records": source_records,
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    identity = result["identity_calibration"]
    total = result["workload"]["total"]
    lines = [
        "# H67 Motion：TARE 与 TTB8-ZKQI 独立增量筛选",
        "",
        "## 结论",
        "",
        f"- [prof] 原始 T450 位流覆盖 `{identity['pairs']}` 个 temporal pair；重新得到与 ZKQI 相同的 `{identity['zkqi_active_pairs']}` 个 active pair 总数，aggregate count 校准通过；尚未逐 pair 比较 bitmap。",
        f"- [prof] active pair 的时间 score 相等率为 `{identity['active_score_equal_ratio']:.4%}`；TARE raw16/Q7 均为零失配。",
        "- [模型] TARE 不减少 Q/K SRAM 读取，也不缩短 TTB8 scanner、SCS 或 gated-K backend；它只能尝试用较窄 residual lane 替代两个并行 Direct32 score engine。",
        f"- [模型] 本轮将 `TARE-W8/W16` 作为条件候选带入 A/B/C RTL；`W{result['selection']['selected_width']}` 只是首选点而非冻结最优，仅允许讲面积/能耗机会。",
        "- [待验证] 只有加入 detector/compactor/RNE 后的同接口 RTL 与开放物理代理仍获益，才能将其升格为 Motion 架构机制。",
        "",
        "## 证据边界",
        "",
        "- 输入是 480x640、window=2x15x15、sample0 一个窗口、全 12 attention block 的真实 Q/K 位流；",
        "- 覆盖四个 stage 共 138 个 head-row，但不是跨 sample workload；",
        "- profile100 没有 raw lane identity，不能从计数反推出 temporal update mask；",
        "- 所有 lane 面积、吞吐和联合周期均为 [模型]，不是 ASIC PPA；",
        "- 强基线是当前 TTB8-ZKQI 加两个并行 32-lane score engine，不再使用旧 Direct32 弱基线。",
        "",
        "## 身份与精确性",
        "",
        "| 项目 | 结果 |",
        "|---|---:|",
        f"| temporal pair | {identity['pairs']} |",
        f"| non-K-zero active pair | {identity['zkqi_active_pairs']} |",
        f"| active ratio | {total['active_pair_ratio']:.4%} |",
        f"| active score equal | {identity['active_score_equal_pairs']} / {identity['zkqi_active_pairs']} ({identity['active_score_equal_ratio']:.4%}) |",
        f"| raw16 mismatch | {result['exactness']['raw16_mismatches']} |",
        f"| Q7 mismatch | {result['exactness']['q7_mismatches']} |",
        "",
        "TARE 只在 non-K-zero active path 上评估。每项先算 32-lane anchor；update-count 不超过 W 时由 W-lane residual 重建，否则追加一次 32-lane direct fallback。所有路径只在末端做一次 RNE。",
        "",
        "## 同强基线 DSE",
        "",
        "| W=T | ZERO | SPARSE | dense fallback | score-lane work减少 | 吞吐/双Direct32 | lane-only面积归一吞吐 | TTB8理想串行fallback模型 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in result["candidates"]:
        bound = candidate["ttb8_combined_ideal_service_model"]
        lines.append(
            f"| {candidate['residual_width']} | {candidate['zero_pairs']} | {candidate['sparse_pairs']} | "
            f"{candidate['dense_fallback_pairs']} ({candidate['dense_fallback_ratio']:.4%}) | "
            f"{candidate['score_lane_work_reduction']:.4%} | "
            f"{candidate['score_throughput_ratio_vs_two_direct32']:.4f}x | "
            f"{candidate['lane_only_area_normalized_score_throughput']:.4f}x | "
            f"{bound['ideal_serial_fallback_cycles']} ({bound['ideal_serial_speed_ratio_vs_ttb8_zkqi']:.4f}x) |"
        )
    lines += [
        "",
        "`lane-only面积归一吞吐` 仅按 64 个 baseline score lane 与 `32+W` 个候选 lane 估算，故意排除了 detector、compactor、控制、SRAM 与频率。它只能做准入筛选，不能进论文 PPA 主表。",
        "",
        "## 分 Block 分布",
        "",
        "| Block | heads | pair | active | score equal | update均值 | TARE-16 fallback |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for block in result["workload"]["blocks"]:
        lines.append(
            f"| {block['name']} | {block['heads']} | {block['pairs']} | {block['active_pairs']} | "
            f"{block['active_score_equal_ratio']:.4%} | {block['active_update_mean']:.3f} | "
            f"{block['tare16_dense_fallback_pairs']} |"
        )
    lines += [
        "",
        "分布高度不均：若干 block 完全没有 active pair，而 S2.B0、S3.B0 的 update 较密。因而固定 W=4 在 fullres T450 上已不合适；W=16 才把 dense fallback 压到可控范围。",
        "",
        "## 架构含义",
        "",
        "```text",
        "TTB8 metadata scan",
        "      |",
        "      +-- K-zero --> three-class exact seed --> shared SCS backend",
        "      |",
        "      `-- active pair --> temporal update detector",
        "                            |",
        "                            +-- <=16: Direct32 anchor + residual16",
        "                            `-- >16 : Direct32 anchor + exact replay",
        "                                      |",
        "                                      v",
        "                               RQTB class commit",
        "                                      |",
        "                                      v",
        "                               shared SCS/gated-K",
        "```",
        "",
        "该组合保留 ZKQI 的 exact zero-K 注入和 TTB8 扫描；TARE-16 只替换 active score 前端。它不新增新的 Q/K 流量收益，也不能把 score-lane work 的下降直接乘到端到端周期上。",
        "",
        "## 准入与下一步",
        "",
        f"当前决策：**{result['selection']['decision']}**。",
        "",
        "下一轮只做一个问题：把参数化 TARE-W8/W16 接到 TTB8 active-pair 出口，并与双 Direct32 做同接口、同反压、同原子T0/T1输出的 A/B/C 强基线 RTL。residual16 的 signed delta 至少使用 12 bit。若加入控制后的全前端面积归一吞吐不大于 1，或总执行 EDP 无净收益，则立即否决，不再以 TARE 作为论文贡献。",
        "",
        "## 复现",
        "",
        "```bash",
        "python3 -m unittest tests.test_profile_h67_tare_zkqi_overlap",
        "python3 scripts/profile_h67_tare_zkqi_overlap.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--strong-baseline", type=Path, default=DEFAULT_STRONG_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze(args.manifest, args.strong_baseline)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_json = args.output_dir / "report.json"
    report_md = args.output_dir / "report.md"
    report_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_md.write_text(render_markdown(result), encoding="utf-8")
    print(f"PASS {result['status']} selected_width={result['selection']['selected_width']}")
    print(report_json)
    print(report_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
