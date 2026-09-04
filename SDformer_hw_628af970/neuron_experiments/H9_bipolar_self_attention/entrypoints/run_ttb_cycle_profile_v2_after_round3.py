"""Collect exact TTB/Delta routing histograms after the full software queue."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
GEN = EXP / "configs/generated"
RESULTS = EXP / "results"
PY = Path(sys.executable)
PREV_STATUS = RESULTS / "round4_assignment_after_h78_status.log"
STATUS = RESULTS / "ttb_cycle_profile_v2_after_round3_status.log"
SUMMARY_JSON = RESULTS / "ttb_delta_cycle_profile100_v2_20260713.json"
SUMMARY_MD = RESULTS / "ttb_delta_cycle_profile100_v2_20260713.md"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
HARDWARE_DOC = REPO / "hw_autoresearch_nts07/docs/46_TTB真实分布周期模型与综合协议.md"
PORTFOLIO = REPO / "neuron_autoresearch/DATE_IDEA_PORTFOLIO_20260712.md"

CASES = (
    {
        "id": "TTX",
        "config": GEN / "date11full_ttx_dyadic_txonly_all12_deploy_int8.yml",
        "checkpoint": RESULTS / "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/checkpoint_epoch2.pth",
        "output": RESULTS / "ttx_ep2_ttb_delta_cycle_v2_profile100_20260713",
    },
    {
        "id": "H67",
        "config": GEN / "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_dyadic_int8_deploy_rtl_exact.yml",
        "require_rtl_exact": True,
        "checkpoint": RESULTS / "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_bs8_full30_20260711_setsid/checkpoint_epoch19.pth",
        "output": RESULTS / "h67_ep19_ttb_delta_cycle_v2_profile100_20260713",
    },
    {
        "id": "H68",
        "config": GEN / "h68_allbinary_all12_castling_ttx_deploy_full30_dyadic_int8_deploy_rtl_exact.yml",
        "require_rtl_exact": True,
        "checkpoint": RESULTS / "h68_allbinary_all12_castling_ttx_aux050_toep20_w720_fastlr_full30_bs8_full30_20260711_setsid/checkpoint_epoch19.pth",
        "output": RESULTS / "h68_ep19_ttb_delta_cycle_v2_profile100_20260713",
    },
)
THRESHOLDS = (2, 4, 8, 12, 16)


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def wait_previous() -> None:
    marker = "ALL COMPLETE ROUND4 ASSIGNMENT:"
    while not PREV_STATUS.exists() or marker not in PREV_STATUS.read_text(encoding="utf-8", errors="ignore"):
        record(f"WAIT Round4 H79-H80 full30 queue: {PREV_STATUS}")
        time.sleep(600)


def profile_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        delta = data["summary"]["delta_ttx"]
        ttb = data["summary"]["token_time_bundles"]
        return (
            len(delta.get("delta_update_histogram", [])) == 33
            and "delta_active_lane_sum_le12" in delta
            and all("active_lane_sum_1_12" in row and row.get("active_histogram") for row in ttb)
            and bool(data.get("ordered_trace", False))
            and all(
                "ttb_tok4_active_ordered_trace" in row
                and "ttb_tok8_active_ordered_trace" in row
                and "delta_update_ordered_trace" in row
                and "pair_q_count_ordered_trace" in row
                and "pair_k_count_ordered_trace" in row
                and "pair_overlap_ordered_trace" in row
                and "pair_motion_ordered_trace" in row
                and "pair_update_ordered_trace" in row
                and "pair_four_vector_union_ordered_trace" in row
                and "projection_baseline_active_lanes_ordered_trace" in row
                and "projection_class_channel_terms_ttx_ordered_trace" in row
                and "projection_class_channel_terms_h67_ordered_trace" in row
                and "projection_class_channel_max_fanout_ttx_ordered_trace" in row
                and "projection_class_channel_max_fanout_h67_ordered_trace" in row
                and "projection_active_classes_ttx_ordered_trace" in row
                and "projection_active_classes_h67_ordered_trace" in row
                and "projection_gate_class_channel_terms_deploy_ordered_trace" in row
                and "projection_gate_class_channel_max_fanout_deploy_ordered_trace" in row
                and "projection_active_gate_classes_deploy_ordered_trace" in row
                and all(
                    f"projection_gate_multicast_delivery_m{width}_ordered_trace" in row
                    for width in (1, 2, 4, 8, 16)
                )
                and all(
                    f"projection_gate_group_terms_g{group_windows}_ordered_trace" in row
                    for group_windows in (1, 2, 4, 8, 16)
                )
                and all(
                    f"projection_gate_group_{metric}_g{group_windows}_ordered_trace" in row
                    for metric in ("active_lanes", "active_classes", "max_fanout")
                    for group_windows in (1, 2, 4, 8, 16)
                )
                and all(
                    f"projection_gate_group_delivery_g{group_windows}_m{width}_ordered_trace" in row
                    for group_windows in (1, 2, 4, 8, 16)
                    for width in (1, 2, 4, 8, 16)
                )
                and all(
                    f"projection_multicast_delivery_{variant}_m{width}_ordered_trace" in row
                    for variant in ("ttx", "h67")
                    for width in (1, 2, 4, 8, 16)
                )
                and "spatial_union_count_histogram" in row
                and "spatial_bank4_rowmajor_cycles_histogram" in row
                and "spatial_bank8_diagonal_cycles_histogram" in row
                for row in data["summary"]["h60_records"]
            )
            and bool(data["summary"].get("binary_temporal_pairs"))
            and data["summary"]["binary_temporal_pairs"].get("spatial_row_total", 0) > 0
            and "projection_class_channel_ratio_h67"
            in data["summary"]["binary_temporal_pairs"]
            and "projection_gate_class_channel_ratio_deploy"
            in data["summary"]["binary_temporal_pairs"]
            and len(data["summary"].get("sample_records", [])) == int(data.get("samples", 0))
            and bool(data["summary"].get("sample_correlations"))
            and bool(data["summary"].get("operator_rows"))
            and bool(data["summary"].get("operator_by_scope"))
            and bool(data["summary"].get("cross_sample_by_stage"))
            and all(
                all(field in row for field in ("finite_ratio", "value_min", "value_max", "binary01_ratio", "ternary_ratio"))
                for row in data["summary"].get("activation_records", [])
                if str(row.get("kind", "")).startswith("stage_")
            )
            and all(
                all(field in row for field in (
                    "deployment_dead_result", "quant_sample_events",
                    "parameter_q4_event_mismatch", "parameter_q6_event_mismatch",
                    "parameter_q8_event_mismatch",
                ))
                for row in data["summary"].get("atlif_rows", [])
            )
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def audit_log(log: Path) -> None:
    text = log.read_text(encoding="utf-8", errors="ignore")
    required = (
        r"installed ATLIF modules: 105",
        r"installed H60/Shiftmax modules: 12",
        r"load audit: checkpoint_overlay_keys=210, model_overlay_keys=210, missing=0, unexpected=0",
        r"processed 100/100",
    )
    missing = [pattern for pattern in required if re.search(pattern, text) is None]
    if missing:
        raise RuntimeError(f"TTB cycle-v2 profile audit failed ({missing}): {log}")


def audit_case_config(case: dict[str, Any]) -> None:
    config = Path(case["config"])
    if not config.exists():
        raise FileNotFoundError(f"profile config does not exist: {config}")
    if case.get("require_rtl_exact"):
        text = config.read_text(encoding="utf-8")
        if re.search(r"^\s*hardware_rtl_shiftmax_enabled:\s*true\s*$", text, re.MULTILINE) is None:
            raise RuntimeError(f"{case['id']} profile must use the RTL-exact Shiftmax config: {config}")


def run_profile(case: dict[str, Any]) -> Path:
    audit_case_config(case)
    output = Path(case["output"])
    profile = output / "nts11_hardware_p0_profile.json"
    log = output / "profile.log"
    if profile_complete(profile):
        audit_log(log)
        record(f"REUSE complete {case['id']} TTB/Delta cycle-v2 profile100: {profile}")
        return profile
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "SDFORMER_USE_MLFLOW": "0",
        "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
        "SDFORMER_SNN_BACKEND": "cupy",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    command = [
        str(PY), "-u", str(EXP / "entrypoints/profile_nts11_hardware_p0.py"),
        "--config", str(case["config"]), "--checkpoint", str(case["checkpoint"]),
        "--output-dir", str(output), "--samples", "100", "--num-workers", "0",
        "--ordered-trace",
    ]
    record(f"START {case['id']} TTB/Delta cycle-v2 profile100: {' '.join(command)}")
    with log.open("a", encoding="utf-8") as handle:
        proc = subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
    record(f"END {case['id']} TTB/Delta cycle-v2 profile100: exit_code={proc.returncode}")
    if proc.returncode != 0 or not profile_complete(profile):
        raise RuntimeError(f"TTB cycle-v2 profile failed: {log}")
    audit_log(log)
    return profile


def replay_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return len(data.get("analytical_sweep", [])) == 45 and len(data.get("finite_replay", [])) == 27
    except (TypeError, ValueError, json.JSONDecodeError):
        return False


def run_cycle_replay(case: dict[str, Any], profile: Path) -> Path:
    output = Path(case["output"])
    replay = output / "ttb_finite_fifo_replay.json"
    if replay_complete(replay):
        record(f"REUSE complete {case['id']} finite FIFO replay: {replay}")
        return replay
    command = [
        str(PY), str(EXP / "entrypoints/replay_ttb_dual_path_cycles.py"),
        "--profile-json", str(profile), "--output", str(replay),
    ]
    log = output / "ttb_finite_fifo_replay.log"
    record(f"START {case['id']} finite FIFO replay: {' '.join(command)}")
    with log.open("a", encoding="utf-8") as handle:
        proc = subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
    record(f"END {case['id']} finite FIFO replay: exit_code={proc.returncode}")
    if proc.returncode != 0 or not replay_complete(replay):
        raise RuntimeError(f"finite FIFO replay failed: {log}")
    return replay


def run_pair_dse(case: dict[str, Any], profile: Path) -> Path:
    output = Path(case["output"])
    result = output / "binary_temporal_pair_arch_dse.json"
    if result.exists():
        try:
            data = json.loads(result.read_text(encoding="utf-8"))
            if data.get("schema_version") == 1 and data.get("model_summary", {}).get("pairs", 0) > 0:
                record(f"REUSE complete {case['id']} binary temporal-pair DSE: {result}")
                return result
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    command = [
        str(PY), str(EXP / "entrypoints/analyze_binary_temporal_pair_arch.py"),
        "--profile-json", str(profile), "--output", str(result),
    ]
    log = output / "binary_temporal_pair_arch_dse.log"
    record(f"START {case['id']} binary temporal-pair DSE: {' '.join(command)}")
    with log.open("a", encoding="utf-8") as handle:
        proc = subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
    record(f"END {case['id']} binary temporal-pair DSE: exit_code={proc.returncode}")
    if proc.returncode != 0 or not result.exists():
        raise RuntimeError(f"binary temporal-pair DSE failed: {log}")
    return result


def summarize(case: dict[str, Any], profile: Path, replay: Path, pair_dse: Path) -> dict[str, Any]:
    data = json.loads(profile.read_text(encoding="utf-8"))
    pair_data = json.loads(pair_dse.read_text(encoding="utf-8"))
    return {
        "id": case["id"],
        "config": str(case["config"]),
        "checkpoint": str(case["checkpoint"]),
        "profile": str(profile),
        "cycle_replay": str(replay),
        "pair_dse": str(pair_dse),
        "delta_ttx": data["summary"]["delta_ttx"],
        "token_time_bundles": data["summary"]["token_time_bundles"],
        "binary_temporal_pairs": data["summary"]["binary_temporal_pairs"],
        "pair_dse_summary": pair_data["model_summary"],
    }


def render(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# TTB/精确差分第二版百样本统计",
        "",
        "所有比例均为路由覆盖率，不直接等同于 speedup 或能耗下降。三个 checkpoint 均通过 ATLIF105、attention12、overlay210/210、missing0/unexpected0 加载审计。",
        "",
        "## 精确差分的令牌路由",
        "",
        "| 模型 | 阈值 | 零差分/复用 | 稀疏令牌 | 稠密令牌 | 稀疏变化通道数 | 每个稀疏令牌平均变化通道 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = row["delta_ttx"]
        total = int(delta["delta_token_heads"])
        zero = int(delta["delta_zero_update_token_heads"])
        for threshold in THRESHOLDS:
            sparse = int(delta[f"delta_active_le{threshold}"])
            lanes = int(delta[f"delta_active_lane_sum_le{threshold}"])
            dense = total - zero - sparse
            lines.append(
                f"| {row['id']} | {threshold} | {zero / total:.6%} | {sparse / total:.6%} | "
                f"{dense / total:.6%} | {lanes} | {lanes / sparse if sparse else 0.0:.4f} |"
            )
    lines += [
        "",
        "## 真实 TTB 路由与条件载荷",
        "",
        "| 模型 | 每包令牌数 | 包数量 | 全空比例 | 活跃通道不超过12 | 对应通道总数 | 每个活跃包平均通道数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        for bundle in row["token_time_bundles"]:
            active = int(bundle["active_1_12_count"])
            lanes = int(bundle["active_lane_sum_1_12"])
            lines.append(
                f"| {row['id']} | {bundle['token_bundle']} | {bundle['bundles']} | "
                f"{bundle['empty_ratio']:.6%} | {bundle['active_1_12_ratio']:.6%} | {lanes} | "
                f"{lanes / active if active else 0.0:.4f} |"
            )
    lines += [
        "",
        "完整 `u=0..32` Delta histogram 和每个 bundle 的 `A_b=0..L_b` histogram 保存在 JSON。"
        "这些字段足以重放 theta/kappa=2/4/8/12/16 的 token 数和条件 index payload；当前artifact已按stage/block有序计数执行有限FIFO回放，最终PPA仍需加入SRAM bank transaction、共享Shiftmax backend与投影/decoder时序。",
        "",
        "## 有序有限深度 FIFO 重放工件",
        "",
        "| 模型 | 重放结果 | 覆盖范围 |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | `{Path(row['cycle_replay']).relative_to(REPO)}` | "
            "元数据、稠密/稀疏服务和有限深度 FIFO；不含 Shiftmax 后端、SRAM、投影、解码器与片上网络 |"
        )
    lines += [
        "",
        "## 二值时间对精确表示设计空间探索",
        "",
        "| 模型 | 时间对数 | 全空比例 | 每对事件数 | 每对并集通道数 | H67 双分数相同比例 | 自适应表示流量下降 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        pair = row["pair_dse_summary"]
        lines.append(
            f"| {row['id']} | {pair['pairs']} | {pair['pair_empty_ratio']:.6%} | "
            f"{pair['mean_events_per_pair']:.4f} | {pair['mean_union_lanes_per_pair']:.4f} | "
            f"{pair['score_equal_h67_ratio']:.6%} | "
            f"{pair['adaptive_traffic_reduction_vs_dense']:.6%} |"
        )
    return "\n".join(lines) + "\n"


def append_document(path: Path, body: str) -> None:
    marker = "TTB_DELTA_CYCLE_PROFILE100_V2_20260713"
    current = path.read_text(encoding="utf-8")
    if marker in current:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n\n## TTB/精确差分第二版百样本自动结果\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write(f"- 结果文件：`{SUMMARY_MD.relative_to(REPO)}`\n\n")
        handle.write("\n".join(body.splitlines()[4:]) + "\n")


def main() -> int:
    wait_previous()
    rows = []
    for case in CASES:
        profile = run_profile(case)
        replay = run_cycle_replay(case, profile)
        pair_dse = run_pair_dse(case, profile)
        rows.append(summarize(case, profile, replay, pair_dse))
    body = render(rows)
    SUMMARY_JSON.write_text(json.dumps({"schema_version": 2, "rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    SUMMARY_MD.write_text(body, encoding="utf-8")
    for document in (REDESIGN, HARDWARE_DOC, PORTFOLIO):
        append_document(document, body)
    record(f"ALL COMPLETE TTB/DELTA CYCLE V2: {SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
