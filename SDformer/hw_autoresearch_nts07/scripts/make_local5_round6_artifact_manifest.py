#!/usr/bin/env python3
"""生成 Local5 第六轮 RTL/模型证据清单。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/local5_round6_artifact_manifest_20260729.json"

ARTIFACTS = [
    "rtl_local5/local5_score_gate_term_top.sv",
    "rtl_local5/local5_row_context_engine.sv",
    "rtl_local5/local5_row_context_tare_engine.sv",
    "rtl_local5/local5_shiftmax5_q17.sv",
    "rtl_local5/local5_mfep_term_builder.sv",
    "rtl_local5/local5_window_attention_top.sv",
    "rtl_local5/local5_linebuf_window_top.sv",
    "tb_local5/tb_local5_score_gate_term_top.sv",
    "tb_local5/tb_local5_row_context_tare.sv",
    "tb_local5/tb_local5_mfep_t450_counter.sv",
    "verif_local5/local5_score_gate_term_assertions.sv",
    "sim_local5/run_local5_parity_checks.sh",
    "sim_local5/run_local5_sva_checks.sh",
    "sim_local5/run_local5_cross_sim_checks.sh",
    "sim_local5/run_local5_tare_yosys_matrix.sh",
    "scripts/report_local5_tare_direct_matrix.py",
    "scripts/phi_prosperity_dual_line_simulator.py",
    "scripts/make_local5_round6_artifact_manifest.py",
    "tests/test_phi_prosperity_dual_line_simulator.py",
    "results/local5_tare_direct_fullflow_20260729.log",
    "results/local5_tare_direct_sva_20260729.log",
    "results/local5_tare_yosys_20260729.log",
    "results/local5_cross_sim_20260729.log",
    "results/local5_tare_direct_arch_eval_20260729/report.json",
    "results/local5_tare_direct_arch_eval_20260729/report.md",
    "results/phi_prosperity_dual_line_sim_20260729/report.json",
    "results/phi_prosperity_dual_line_sim_20260729/report.md",
    "results/phi_prosperity_dual_line_unit_20260729.log",
    "docs/178_Local5_TARE同顶层消融与双线第六轮RTL收口_20260729.md",
    "docs/179_Local5独立RTL审阅与完成反压整改_20260729.md",
    "docs/180_DATE第六轮独立复审与双线主张重构_20260729.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_first_line(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"不可用: {exc}"
    output = (result.stdout + result.stderr).strip()
    return output.splitlines()[0] if output else f"退出码 {result.returncode}"


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> int:
    records = []
    missing = []
    for relative in ARTIFACTS:
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    fullflow = (ROOT / "results/local5_tare_direct_fullflow_20260729.log").read_text(
        errors="replace"
    )
    sva = (ROOT / "results/local5_tare_direct_sva_20260729.log").read_text(
        errors="replace"
    )
    yosys = (ROOT / "results/local5_tare_yosys_20260729.log").read_text(
        errors="replace"
    )
    unit = (ROOT / "results/phi_prosperity_dual_line_unit_20260729.log").read_text(
        errors="replace"
    )

    checks = {
        "fullflow_pass_records": sum(
            1 for line in fullflow.splitlines() if line.startswith("PASS ")
        ),
        "fullflow_complete": "ALL LOCAL5 PARITY+WINDOW CHECKS PASSED" in fullflow,
        "fullflow_has_failure_token": any(
            token in fullflow for token in ("%Error", "FAIL", "TIMEOUT")
        ),
        "sva_complete": "ALL LOCAL5 SVA CHECKS PASSED" in sva,
        "yosys_four_point_complete": (
            "ALL LOCAL5 TARE YOSYS MATRIX CHECKS PASSED" in yosys
        ),
        "cross_sim_complete": (
            "ALL LOCAL5 ICARUS CROSS-SIM CHECKS PASSED"
            in (ROOT / "results/local5_cross_sim_20260729.log").read_text(
                errors="replace"
            )
        ),
        "model_unit_14_pass": "Ran 14 tests" in unit and unit.rstrip().endswith("OK"),
    }
    checks["all_required_pass"] = (
        checks["fullflow_pass_records"] == 21
        and checks["fullflow_complete"]
        and not checks["fullflow_has_failure_token"]
        and checks["sva_complete"]
        and checks["yosys_four_point_complete"]
        and checks["cross_sim_complete"]
        and checks["model_unit_14_pass"]
        and not missing
    )

    manifest = {
        "schema": "local5_round6_artifact_manifest_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "Local5研究原型RTL与双线模型；非DC/STA/SAIF签核",
        "git": {
            "head": git_value("rev-parse", "HEAD"),
            "dirty": bool(git_value("status", "--porcelain")),
        },
        "tools": {
            "python": command_first_line(["python3", "--version"]),
            "iverilog": command_first_line(["iverilog", "-V"]),
            "verilator": command_first_line(["verilator", "--version"]),
            "yosys": command_first_line(["yosys", "-V"]),
        },
        "checks": checks,
        "missing": missing,
        "artifacts": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(OUT)
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if checks["all_required_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
