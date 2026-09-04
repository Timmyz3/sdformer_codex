#!/usr/bin/env python3
"""等待TTB-v2完成后自动生成HIT-Flow ordered架构报告与预算。"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
HW = REPO / "hw_autoresearch_nts07"
EXP_RESULTS = REPO / "neuron_experiments/H9_bipolar_self_attention/results"
SOURCE_STATUS = EXP_RESULTS / "ttb_cycle_profile_v2_after_round3_status.log"
STATUS = HW / "results/hit_flow_postprofile_after_ttb_v2_status.log"
H67_PROFILE = EXP_RESULTS / "h67_ep19_ttb_delta_cycle_v2_profile100_20260713/nts11_hardware_p0_profile.json"
H68_PROFILE = EXP_RESULTS / "h68_ep19_ttb_delta_cycle_v2_profile100_20260713/nts11_hardware_p0_profile.json"
ANALYSIS_JSON = HW / "results/hit_flow_ordered_profile_analysis.json"
ANALYSIS_MD = HW / "results/hit_flow_ordered_profile_analysis.md"
BUDGET_JSON = HW / "results/hit_flow_full_encoder_budget_ordered.json"
BUDGET_MD = HW / "results/hit_flow_full_encoder_budget_ordered.md"
GCMP_H67_JSON = HW / "results/gcmp_h67_multicast_dse.json"
GCMP_H67_MD = HW / "results/gcmp_h67_multicast_dse.md"
GCMP_H68_JSON = HW / "results/gcmp_h68_multicast_dse.json"
GCMP_H68_MD = HW / "results/gcmp_h68_multicast_dse.md"
WINDOW_H67_JSON = HW / "results/gate_window_group_h67_dse.json"
WINDOW_H67_MD = HW / "results/gate_window_group_h67_dse.md"
WINDOW_H68_JSON = HW / "results/gate_window_group_h68_dse.json"
WINDOW_H68_MD = HW / "results/gate_window_group_h68_dse.md"
RESEARCH_LOG = HW / "research-log.md"


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def wait_source() -> None:
    marker = "ALL COMPLETE TTB/DELTA CYCLE V2:"
    while not SOURCE_STATUS.exists() or marker not in SOURCE_STATUS.read_text(encoding="utf-8", errors="ignore"):
        record(f"等待TTB-v2完成：{SOURCE_STATUS}")
        time.sleep(600)


def run(command: list[str], label: str) -> None:
    record(f"开始{label}：{' '.join(command)}")
    proc = subprocess.run(command, cwd=REPO)
    record(f"结束{label}：exit_code={proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError(f"{label}失败")


def analysis_complete() -> bool:
    if not ANALYSIS_JSON.exists() or not ANALYSIS_MD.exists():
        return False
    try:
        data = json.loads(ANALYSIS_JSON.read_text(encoding="utf-8"))
        return (
            data.get("schema_version") == 1
            and [row.get("model") for row in data.get("models", [])] == ["H67", "H68"]
            and all(row.get("operator_by_scope") for row in data["models"])
            and all(row.get("cross_sample_by_stage") for row in data["models"])
            and all(row.get("projection_multicast") for row in data["models"])
            and all(
                "deploy_group_g16_reduction_vs_row" in row["projection_multicast"]
                for row in data["models"]
            )
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def budget_complete() -> bool:
    if not BUDGET_JSON.exists() or not BUDGET_MD.exists():
        return False
    try:
        data = json.loads(BUDGET_JSON.read_text(encoding="utf-8"))
        return (
            data.get("schema_version") == 1
            and "逐算子encoder" in data.get("inputs", {}).get("spatial_proxy_source", "")
            and bool(data.get("configurations"))
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return False


def gcmp_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("schema_version") == 1 and len(data.get("configurations", [])) == 135
    except (TypeError, ValueError, json.JSONDecodeError):
        return False


def window_group_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("schema_version") == 1 and len(data.get("configurations", [])) == 675
    except (TypeError, ValueError, json.JSONDecodeError):
        return False


def append_log_once() -> None:
    marker = "HIT_FLOW_ORDERED_POSTPROFILE_20260713"
    text = RESEARCH_LOG.read_text(encoding="utf-8")
    if marker in text:
        return
    with RESEARCH_LOG.open("a", encoding="utf-8") as handle:
        handle.write("\n## Ordered profile自动回填完成\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write(f"- 架构决策报告：`{ANALYSIS_MD.relative_to(REPO)}`；\n")
        handle.write(f"- 逐算子全encoder预算：`{BUDGET_MD.relative_to(REPO)}`；\n")
        handle.write(f"- H67 GCM-P DSE：`{GCMP_H67_MD.relative_to(REPO)}`；\n")
        handle.write(f"- H68 GCM-P DSE：`{GCMP_H68_MD.relative_to(REPO)}`；\n")
        handle.write(f"- H67跨窗口gate-product DSE：`{WINDOW_H67_MD.relative_to(REPO)}`；\n")
        handle.write(f"- H68跨窗口gate-product DSE：`{WINDOW_H68_MD.relative_to(REPO)}`；\n")
        handle.write("- PCCC、bank mapping、RPI、ATLIF量化、persistent-HTT和spatial lane按报告门槛重新冻结。\n")


def main() -> int:
    wait_source()
    for path in (H67_PROFILE, H68_PROFILE):
        if not path.exists():
            raise FileNotFoundError(path)
    if not analysis_complete():
        run([
            sys.executable,
            str(HW / "scripts/analyze_hit_flow_ordered_profiles.py"),
            "--h67", str(H67_PROFILE),
            "--h68", str(H68_PROFILE),
            "--json", str(ANALYSIS_JSON),
            "--md", str(ANALYSIS_MD),
        ], "HIT-Flow ordered profile审计")
    else:
        record(f"复用已完成ordered分析：{ANALYSIS_JSON}")
    if not budget_complete():
        run([
            sys.executable,
            str(HW / "scripts/model_hit_flow_full_encoder_budget.py"),
            "--storage", str(HW / "results/h67_h68_encoder_storage_contract.json"),
            "--profile", str(HW / "results/h67_h68_profile100_arch_features.json"),
            "--sops", str(EXP_RESULTS / "h68_castling_ttx_aux050_s360_retry_20260711_202914/profile_deploy_valid825/sops_summary.json"),
            "--runtime-profile", str(H67_PROFILE),
            "--json", str(BUDGET_JSON),
            "--md", str(BUDGET_MD),
        ], "HIT-Flow逐算子全encoder预算")
    else:
        record(f"复用已完成ordered预算：{BUDGET_JSON}")
    for profile, variant, output_json, output_md, label in (
        (H67_PROFILE, "h67", GCMP_H67_JSON, GCMP_H67_MD, "H67 GCM-P DSE"),
        (H68_PROFILE, "ttx", GCMP_H68_JSON, GCMP_H68_MD, "H68 GCM-P DSE"),
    ):
        if not gcmp_complete(output_json):
            run([
                sys.executable,
                str(HW / "scripts/model_gcmp_multicast_dse.py"),
                "--profile", str(profile),
                "--variant", variant,
                "--json", str(output_json),
                "--md", str(output_md),
            ], label)
        else:
            record(f"复用已完成{label}：{output_json}")
    for profile, output_json, output_md, label in (
        (H67_PROFILE, WINDOW_H67_JSON, WINDOW_H67_MD, "H67跨窗口gate-product DSE"),
        (H68_PROFILE, WINDOW_H68_JSON, WINDOW_H68_MD, "H68跨窗口gate-product DSE"),
    ):
        if not window_group_complete(output_json):
            run([
                sys.executable,
                str(HW / "scripts/model_gate_window_group_dse.py"),
                "--profile", str(profile),
                "--json", str(output_json),
                "--md", str(output_md),
            ], label)
        else:
            record(f"复用已完成{label}：{output_json}")
    if (
        not analysis_complete()
        or not budget_complete()
        or not gcmp_complete(GCMP_H67_JSON)
        or not gcmp_complete(GCMP_H68_JSON)
        or not window_group_complete(WINDOW_H67_JSON)
        or not window_group_complete(WINDOW_H68_JSON)
    ):
        raise RuntimeError("postprofile输出完整性检查失败")
    append_log_once()
    record(f"全部完成HIT-Flow ordered回填：{ANALYSIS_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
