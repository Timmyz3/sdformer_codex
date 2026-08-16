#!/usr/bin/env python3
"""汇总Motion RQTB一/双slot同随机反压强基线。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

try:
    from scripts.summarize_h67_rqtb_physical_flow import (
        distribution,
        key_values,
        occupancy_distribution,
        speedup_distribution,
    )
except ModuleNotFoundError:
    from summarize_h67_rqtb_physical_flow import (
        distribution,
        key_values,
        occupancy_distribution,
        speedup_distribution,
    )


ROW_RE = re.compile(r"^RQTB_ROW (?P<body>.+)$")
FINAL_RE = re.compile(r"^PASS H67 RQTB(?: 1S random| 2S)? physical flow (?P<body>.+)$")
AREA_RE = re.compile(r"Chip area for module '\\(?P<top>[^']+)':\s+(?P<area>[0-9.]+)")
COVER_RE = re.compile(r"^RQTB_2S_COVER (?P<body>.+)$")
RESTART_PASS = "PASS H67 RQTB 2S rejected-restart fail-closed"
BUILD_RESTART_PASS = "PASS H67 RQTB 2S build-stage rejected-restart mutation-kill"
EXPECTED_COVER_KEYS = {
    "cross_pair", "same_class", "double_active", "fifo_both", "dual_k",
    "fixed_cross_pair", "rqtb_cross_pair",
    "fixed_same_class", "rqtb_same_class",
    "fixed_double_active", "rqtb_double_active",
    "fixed_fifo_both", "rqtb_fifo_both",
    "fixed_dual_k", "rqtb_dual_k",
}
ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = [
    "rtl_ttx/ttx_ceil_log2_u32.sv",
    "rtl_ttx/ttx_exp2_lut_q8.sv",
    "rtl_ttx/ttx_gate_quant_q17.sv",
    "rtl_h67/h67_motionxor_score_q7.sv",
    "rtl_h67/h67_temporal_slot_encoder.sv",
    "rtl_h67/h67_sync_dual_bank_k_store.sv",
    "rtl_h67/h67_temporal_slot_fifo.sv",
    "rtl_h67/h67_temporal_slot_fifo_2s.sv",
    "rtl_h67/h67_temporal_weighted_scs_directory.sv",
    "rtl_h67/h67_temporal_weighted_scs_directory_2s.sv",
    "rtl_h67/h67_temporal_slot_shiftmax_sync_k_top.sv",
    "rtl_h67/h67_temporal_slot_shiftmax_sync_k_2s_top.sv",
    "verif_h67/h67_temporal_slot_flow_2s_assertions.sv",
    "tb_h67/tb_h67_temporal_slot_flow_real_trace_1s_random.sv",
    "tb_h67/tb_h67_temporal_slot_flow_real_trace_2s.sv",
    "tb_h67/tb_h67_temporal_slot_restart_reject_2s.sv",
    "tb_h67/tb_h67_temporal_slot_build_restart_reject_2s.sv",
    "sim_h67/run_h67_rqtb_strong_baseline_checks.sh",
    "scripts/summarize_h67_rqtb_strong_baseline.py",
    "scripts/summarize_h67_rqtb_physical_flow.py",
    "tests/test_summarize_h67_rqtb_strong_baseline.py",
    "tests/test_summarize_h67_rqtb_physical_flow.py",
    "tests/test_summarize_rqtb_openroad_proxy.py",
    "tests/test_summarize_h67_rqtb_fifo_depth_dse.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"证据文件不存在: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load_vector_identity(manifest_path: Path, vector_path: Path) -> dict[str, Any]:
    """绑定向量、上游trace manifest与checkpoint/config身份。"""

    manifest_path = manifest_path.resolve()
    vector_path = vector_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "h67_checkpoint_t450_score_shiftmax_vectors_v1":
        raise ValueError("向量manifest schema不一致")
    declared_vector = Path(str(manifest.get("vector_file", ""))).resolve()
    if declared_vector != vector_path:
        raise ValueError("向量manifest绑定的vector_file与实际输入不一致")
    vector_sha = sha256(vector_path)
    if vector_sha != manifest.get("vector_sha256"):
        raise ValueError("向量文件SHA与manifest不一致")
    if (
        int(manifest.get("row_count", -1)) != 138
        or int(manifest.get("tokens_per_row", -1)) != 450
        or int(manifest.get("token_vector_count", -1)) != 62_100
    ):
        raise ValueError("向量manifest不满足138x450合同")
    records = manifest.get("records")
    if (
        not isinstance(records, list)
        or len(records) != 12
        or sum(int(record.get("rows", -1)) for record in records) != 138
    ):
        raise ValueError("向量manifest不满足12-block/138-row合同")

    source_manifest = Path(str(manifest.get("source_manifest", ""))).resolve()
    if not source_manifest.is_file():
        raise ValueError("上游trace manifest不存在")
    source_manifest_sha = sha256(source_manifest)
    if source_manifest_sha != manifest.get("source_manifest_sha256"):
        raise ValueError("上游trace manifest SHA不一致")

    run_context = manifest.get("run_context")
    if not isinstance(run_context, dict):
        raise ValueError("向量manifest缺少run_context")
    artifact = run_context.get("artifact_identity")
    protocol = run_context.get("eval_protocol")
    if not isinstance(artifact, dict) or not isinstance(protocol, dict):
        raise ValueError("向量manifest缺少artifact/eval身份")
    required_artifact = {
        "config_path", "config_sha256", "checkpoint_path", "checkpoint_sha256"
    }
    if not required_artifact.issubset(artifact):
        raise ValueError("artifact identity字段不完整")
    if protocol.get("tokens_per_window") != 450:
        raise ValueError("eval protocol的N_tok不等于450")
    config_path = Path(str(artifact["config_path"])).resolve()
    if not config_path.is_file() or sha256(config_path) != artifact["config_sha256"]:
        raise ValueError("config文件SHA与artifact identity不一致")
    checkpoint_path = Path(str(artifact["checkpoint_path"])).resolve()
    if (
        not checkpoint_path.is_file()
        or sha256(checkpoint_path) != artifact["checkpoint_sha256"]
    ):
        raise ValueError("checkpoint文件SHA与artifact identity不一致")
    if (
        "checkpoint_size" in artifact
        and checkpoint_path.stat().st_size != int(artifact["checkpoint_size"])
    ):
        raise ValueError("checkpoint文件大小与artifact identity不一致")
    for record in records:
        source = Path(str(record.get("source", ""))).resolve()
        if not source.is_file() or sha256(source) != record.get("source_sha256"):
            raise ValueError(f"source NPZ SHA不一致: {source}")

    return {
        "vector_manifest": str(manifest_path),
        "vector_manifest_sha256": sha256(manifest_path),
        "vector_file": str(vector_path),
        "vector_sha256": vector_sha,
        "source_trace_manifest": str(source_manifest),
        "source_trace_manifest_sha256": source_manifest_sha,
        "artifact_identity": artifact,
        "eval_protocol": protocol,
        "scope": manifest.get("scope"),
        "records": records,
    }


def parse_log(path: Path) -> tuple[list[dict[str, int]], dict[str, int], list[list[int]]]:
    rows: list[dict[str, int]] = []
    final: dict[str, int] | None = None
    occupancy: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if match := ROW_RE.match(line):
            rows.append(key_values(match.group("body")))
        elif match := FINAL_RE.match(line):
            final = key_values(match.group("body"))
        elif line.startswith("RQTB_OCC "):
            occupancy.append([int(value) for value in line.split("=", 1)[1].split(",")])
    if final is None or len(rows) != 138 or len(occupancy) != 2:
        raise ValueError(f"日志不满足138行完整合同: {path}")
    return rows, final, occupancy


def parse_area(path: Path, expected_top: str) -> float:
    matches = [
        float(match.group("area"))
        for match in AREA_RE.finditer(path.read_text(encoding="utf-8"))
        if match.group("top") == expected_top
    ]
    if not matches:
        raise ValueError(f"映射日志缺少{expected_top}面积: {path}")
    return matches[-1]


def parse_cover(path: Path) -> dict[str, int]:
    matches = [
        key_values(match.group("body"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if (match := COVER_RE.match(line))
    ]
    if len(matches) != 1 or set(matches[0]) != EXPECTED_COVER_KEYS:
        raise ValueError(f"2S关键机制coverage receipt为空或重复: {path}")
    coverage = matches[0]
    required_positive = EXPECTED_COVER_KEYS - {"fixed_cross_pair", "fixed_dual_k"}
    if (coverage["fixed_cross_pair"] != 0
            or coverage["fixed_dual_k"] != 0
            or any(coverage[key] <= 0 for key in required_positive)
            or coverage["cross_pair"] != coverage["rqtb_cross_pair"]
            or coverage["same_class"]
               != coverage["fixed_same_class"] + coverage["rqtb_same_class"]
            or coverage["double_active"]
               != coverage["fixed_double_active"] + coverage["rqtb_double_active"]
            or coverage["fifo_both"]
               != coverage["fixed_fifo_both"] + coverage["rqtb_fifo_both"]
            or coverage["dual_k"] != coverage["rqtb_dual_k"]):
        raise ValueError(f"2S关键机制coverage语义矩阵不成立: {path}")
    return coverage


def require_restart_receipt(path: Path, expected: str = RESTART_PASS) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    if text.count(expected) != 1:
        raise ValueError(f"非法restart fail-closed receipt缺失或重复: {path}")


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    vector_identity = load_vector_identity(args.vector_manifest, args.vectors)
    rows_1s, final_1s, occ_1s = parse_log(args.log_1s)
    rows_2s, final_2s, occ_2s = parse_log(args.log_2s)
    rows_2s_sva, final_2s_sva, _ = parse_log(args.log_2s_sva)
    cover_2s = parse_cover(args.log_2s)
    cover_2s_sva = parse_cover(args.log_2s_sva)
    require_restart_receipt(args.restart_log)
    require_restart_receipt(args.restart_sva_log)
    require_restart_receipt(args.build_restart_log, BUILD_RESTART_PASS)
    require_restart_receipt(args.build_restart_sva_log, BUILD_RESTART_PASS)

    if (rows_2s_sva != rows_2s or final_2s_sva != final_2s
            or cover_2s_sva != cover_2s):
        raise ValueError("2S SVA与主仿真日志不一致")
    invariant_keys = [
        "row", "stage", "block", "head", "active", "equal",
        "fixed_slots", "rqtb_slots", "fixed_desc", "rqtb_desc",
        "fixed_exp", "rqtb_exp",
    ]
    for row_1s, row_2s in zip(rows_1s, rows_2s):
        if any(row_1s[key] != row_2s[key] for key in invariant_keys):
            raise ValueError(f"1S/2S工作量合同不一致: row={row_1s['row']}")
        if row_2s["fixed_slots"] != 450:
            raise ValueError("Fixed slot数不是450")
        if row_2s["rqtb_slots"] != 450 - row_2s["equal"]:
            raise ValueError("RQTB slot商合同不成立")

    metric_names = ("fixed_cycles", "rqtb_cycles")
    cycles = {
        "fixed_1s": [row[metric_names[0]] for row in rows_1s],
        "rqtb_1s": [row[metric_names[1]] for row in rows_1s],
        "fixed_2s": [row[metric_names[0]] for row in rows_2s],
        "rqtb_2s": [row[metric_names[1]] for row in rows_2s],
    }
    totals = {name: sum(values) for name, values in cycles.items()}

    areas = {
        "fixed_1s": parse_area(args.map_fixed_1s, "h67_temporal_slot_shiftmax_sync_k_top"),
        "rqtb_1s": parse_area(args.map_rqtb_1s, "h67_temporal_slot_shiftmax_sync_k_top"),
        "fixed_2s": parse_area(args.map_fixed_2s, "h67_temporal_slot_shiftmax_sync_k_2s_top"),
        "rqtb_2s": parse_area(args.map_rqtb_2s, "h67_temporal_slot_shiftmax_sync_k_2s_top"),
    }
    primary_speedup = totals["fixed_2s"] / totals["rqtb_2s"]
    primary_area_ratio = areas["rqtb_2s"] / areas["fixed_2s"]

    result: dict[str, Any] = {
        "schema": "h67_rqtb_strong_baseline_v1",
        "status": "PASS",
        "evidence_level": "[rtl]+[身份绑定]+[open-map代理]",
        "scope": "H67 checkpoint-bound sample0/window0全12 block、138个真实N_tok=450 head-row；可复现LFSR反压",
        "input_identity": vector_identity,
        "coverage": {
            "rows": 138,
            "tokens": 62100,
            "gated_k_outputs_checked": final_2s["checked"],
            "synthetic_acc32_checksum_values_checked": 4416,
            "synthetic_acc32_checksum_mismatch": final_2s["acc32_mismatch"],
            "sva_rows_2s": len(rows_2s_sva),
            "backpressure": "16-bit fixed-seed LFSR；每个head-row重新播种",
            "key_mechanism_cover_hits": cover_2s,
            "rejected_restart_fail_closed": True,
            "build_stage_rejected_restart_mutation_killed": True,
        },
        "cycles": {
            "totals": totals,
            "distributions": {name: distribution(values) for name, values in cycles.items()},
            "rqtb_vs_fixed_1s": {
                "speedup": totals["fixed_1s"] / totals["rqtb_1s"],
                "cycle_reduction_ratio": 1.0 - totals["rqtb_1s"] / totals["fixed_1s"],
            },
            "rqtb_vs_fixed_2s_primary": {
                "speedup": primary_speedup,
                "cycle_reduction_ratio": 1.0 - totals["rqtb_2s"] / totals["fixed_2s"],
                "per_row_speedup": speedup_distribution(cycles["fixed_2s"], cycles["rqtb_2s"]),
                "rqtb_faster_rows": sum(
                    rqtb < fixed
                    for fixed, rqtb in zip(cycles["fixed_2s"], cycles["rqtb_2s"])
                ),
            },
            "fixed_2s_vs_fixed_1s": {
                "speedup": totals["fixed_1s"] / totals["fixed_2s"],
                "cycle_reduction_ratio": 1.0 - totals["fixed_2s"] / totals["fixed_1s"],
            },
            "rqtb_2s_vs_rqtb_1s": {
                "speedup": totals["rqtb_1s"] / totals["rqtb_2s"],
                "cycle_reduction_ratio": 1.0 - totals["rqtb_2s"] / totals["rqtb_1s"],
            },
        },
        "work": {
            "fixed_slots": final_2s["fixed_slots"],
            "rqtb_slots": final_2s["rqtb_slots"],
            "slot_reduction_ratio": 1.0 - final_2s["rqtb_slots"] / final_2s["fixed_slots"],
            "fixed_exp": final_2s["fixed_exp"],
            "rqtb_exp": final_2s["rqtb_exp"],
            "exp_reduction_ratio": 1.0 - final_2s["rqtb_exp"] / final_2s["fixed_exp"],
            "occupancy_1s": {
                "fixed": occupancy_distribution(occ_1s[0]),
                "rqtb": occupancy_distribution(occ_1s[1]),
            },
            "occupancy_2s": {
                "fixed": occupancy_distribution(occ_2s[0]),
                "rqtb": occupancy_distribution(occ_2s[1]),
            },
        },
        "metric_definitions": {
            "class_transactions": (
                "全行class_present位图的唯一score-class扫描次数；Fixed与RQTB相同"
            ),
            "exp_transactions": (
                "class scan一次exp加每个active descriptor在emit阶段一次exp/gate重建；"
                "这是逻辑活动代理，不是独立cycle，也不是唯一class数"
            ),
            "exp_reduction_origin": (
                "RQTB合并同一temporal pair内同score的active descriptor；"
                "不来自全行class cardinality下降"
            ),
            "strong_baseline_gap": (
                "尚未与class-exp/gate cache基线比较；因此exp reduction不能单独作为"
                "SCS架构优势或能量结论"
            ),
        },
        "open_mapping_proxy": {
            "library": "NangateOpenCellLibrary_typical.lib",
            "constraint": "无SDC、memory未映射；仅逻辑面积代理",
            "logic_area": areas,
            "primary_rqtb_area_overhead_ratio": primary_area_ratio - 1.0,
            "primary_area_normalized_throughput": primary_speedup / primary_area_ratio,
            "fixed_2s_area_overhead_vs_fixed_1s": areas["fixed_2s"] / areas["fixed_1s"] - 1.0,
            "rqtb_2s_area_overhead_vs_rqtb_1s": areas["rqtb_2s"] / areas["rqtb_1s"] - 1.0,
        },
        "negative_results": [
            "早期1S对照把单口解码瓶颈计入RQTB收益，不能作为主基线。",
            "双slot公平基线把RQTB周期收益从约30%收缩到约16%，说明约一半早期收益并非RQTB独有。",
            "当前公平RTL锚点只有sample0/window0的138个head-row，尚不能外推到多样本p95/p99部署分布。",
            "开放逻辑映射没有SDC且未计memory；2S双读/双写存储代价必须以flop-memory OpenROAD或真实多bank SRAM合同评估。",
            "exp事务下降来自active descriptor级gate重建减少；class-gate cache强基线尚未完成。",
        ],
        "claim_boundary": [
            "主张RQTB性能时只使用Fixed2S对RQTB2S的等资源结果。",
            "1S结果只用于说明弱基线会高估RQTB收益。",
            "Acc32是固定合成权重的整数checksum，不是真实投影权重回放。",
        ],
        "rows_1s": rows_1s,
        "rows_2s": rows_2s,
    }
    log_dir = args.log_1s.resolve().parent
    log_artifacts = {
        path.name: artifact(path)
        for path in sorted(log_dir.iterdir())
        if path.is_file()
    }
    result["provenance"] = {
        "log_artifacts": log_artifacts,
        "implementation_artifacts": {
            relative: artifact(ROOT / relative)
            for relative in IMPLEMENTATION_PATHS
        },
    }
    return result


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    cycles = result["cycles"]
    totals = cycles["totals"]
    primary = cycles["rqtb_vs_fixed_2s_primary"]
    mapping = result["open_mapping_proxy"]
    areas = mapping["logic_area"]
    lines = [
        "# Motion RQTB双slot公平强基线报告",
        "",
        "## 结论",
        "",
        "- 状态：**PASS**；证据等级：**[rtl]+[身份绑定]+[open-map代理]**。",
        f"- 主基线Fixed2S→RQTB2S：{totals['fixed_2s']:,}→{totals['rqtb_2s']:,}周期，{primary['speedup']:.3f}×，周期减少{primary['cycle_reduction_ratio']:.2%}。",
        f"- 138行中RQTB2S更快：{primary['rqtb_faster_rows']}/138；{result['coverage']['gated_k_outputs_checked']:,}个gated-K输出和{result['coverage']['synthetic_acc32_checksum_values_checked']:,}个synthetic Acc32 checksum零失配。",
        f"- 关键机制coverage hit：跨pair {result['coverage']['key_mechanism_cover_hits']['cross_pair']:,}、同class冲突 {result['coverage']['key_mechanism_cover_hits']['same_class']:,}、双active append {result['coverage']['key_mechanism_cover_hits']['double_active']:,}、FIFO并发 {result['coverage']['key_mechanism_cover_hits']['fifo_both']:,}、双K读取 {result['coverage']['key_mechanism_cover_hits']['dual_k']:,}。",
        "- 非法提前window_start在Icarus和Verilator+SVA中均被拒绝；构建阶段合法pair与发射阶段held output两类反例均已覆盖。",
        f"- slot减少{result['work']['slot_reduction_ratio']:.2%}，exp事务减少{result['work']['exp_reduction_ratio']:.2%}。",
        "- `exp transaction` 定义为class扫描加active descriptor级gate重建活动代理；它不是唯一class数或独立周期。",
        "- Fixed/RQTB唯一class数相同；exp下降来自descriptor合并，尚未对比class-gate cache强基线。",
        f"- checkpoint SHA：`{result['input_identity']['artifact_identity']['checkpoint_sha256']}`。",
        f"- config SHA：`{result['input_identity']['artifact_identity']['config_sha256']}`。",
        f"- vector SHA：`{result['input_identity']['vector_sha256']}`；`N_tok=450`，不是SNN时间步数。",
        "",
        "## 公平消融",
        "",
        "| 配置 | 总周期 | 相对同slot宽度Fixed | 开放逻辑面积代理 |",
        "|---|---:|---:|---:|",
        f"| Fixed1S | {totals['fixed_1s']:,} | 1.000× | {areas['fixed_1s']:.2f} |",
        f"| RQTB1S | {totals['rqtb_1s']:,} | {cycles['rqtb_vs_fixed_1s']['speedup']:.3f}× | {areas['rqtb_1s']:.2f} |",
        f"| Fixed2S | {totals['fixed_2s']:,} | 1.000× | {areas['fixed_2s']:.2f} |",
        f"| RQTB2S | {totals['rqtb_2s']:,} | {primary['speedup']:.3f}× | {areas['rqtb_2s']:.2f} |",
        "",
        f"- 单纯把Fixed前端从1S加宽到2S已经带来{cycles['fixed_2s_vs_fixed_1s']['speedup']:.3f}×；因此1S的RQTB收益被高估。",
        f"- RQTB自身从1S到2S只有{cycles['rqtb_2s_vs_rqtb_1s']['speedup']:.3f}×，说明商编码已经缓解前端压力。",
        f"- 主配置面积归一吞吐代理：{mapping['primary_area_normalized_throughput']:.3f}×。这不是ASIC PPA。",
        "",
        "## 负结果与边界",
        "",
    ]
    lines.extend(f"- {item}" for item in result["negative_results"])
    lines.extend(["", "## 论文口径", ""])
    lines.extend(f"- {item}" for item in result["claim_boundary"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-1s", type=Path, required=True)
    parser.add_argument("--log-2s", type=Path, required=True)
    parser.add_argument("--log-2s-sva", type=Path, required=True)
    parser.add_argument("--restart-log", type=Path, required=True)
    parser.add_argument("--restart-sva-log", type=Path, required=True)
    parser.add_argument("--build-restart-log", type=Path, required=True)
    parser.add_argument("--build-restart-sva-log", type=Path, required=True)
    parser.add_argument("--map-fixed-1s", type=Path, required=True)
    parser.add_argument("--map-rqtb-1s", type=Path, required=True)
    parser.add_argument("--map-fixed-2s", type=Path, required=True)
    parser.add_argument("--map-rqtb-2s", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--vector-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    snapshot_root = args.output_dir / "source_snapshots"
    snapshot_manifest: dict[str, Any] = {}
    for relative in IMPLEMENTATION_PATHS:
        source = ROOT / relative
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        snapshot_manifest[relative] = artifact(destination)
        if snapshot_manifest[relative]["sha256"] != result["provenance"]["implementation_artifacts"][relative]["sha256"]:
            raise ValueError(f"源码快照SHA不一致: {relative}")
    (args.output_dir / "source_snapshot_manifest.json").write_text(
        json.dumps(snapshot_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(args.output_dir / "report.md", result)
    (args.output_dir / "complete.json").write_text(
        json.dumps(
            {
                "schema": "h67_rqtb_strong_baseline_complete_v1",
                "status": result["status"],
                "report_sha256": sha256(args.output_dir / "report.json"),
                "markdown_sha256": sha256(args.output_dir / "report.md"),
                "source_snapshot_manifest_sha256": sha256(
                    args.output_dir / "source_snapshot_manifest.json"
                ),
                "implementation_artifacts": len(IMPLEMENTATION_PATHS),
                "log_artifacts": len(result["provenance"]["log_artifacts"]),
                "checkpoint_sha256_recomputed": result["input_identity"]["artifact_identity"]["checkpoint_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
