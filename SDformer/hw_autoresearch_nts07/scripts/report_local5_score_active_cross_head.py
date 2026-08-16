#!/usr/bin/env python3
"""Seal the production score-active Local5 cross-head smoke evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


PASS_RE = re.compile(
    r"PASS Local5 cross-head OUT32 seed=(?P<seed>\d+) "
    r"cycles=(?P<cycles>\d+) heads=(?P<heads>\d+) "
    r"partial=(?P<partial>\d+) final=(?P<final>\d+) "
    r"result_stall=(?P<stall>\d+) group_stall=(?P<group_stall>\d+)"
)
DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def parse_log(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if "ERROR:" in text or "FATAL:" in text or "mismatch" in text.lower():
        raise ValueError(f"failure marker in {path}")
    matches = list(PASS_RE.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one PASS record in {path}")
    return {name: int(value) for name, value in matches[0].groupdict().items()}


def tool_log_is_clean(path: Path, required: str | None = None) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    if re.search(r"(?:%Error|\bERROR:|\bFATAL:)", text, re.IGNORECASE):
        return False
    return required is None or required in text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    evidence = args.evidence if args.evidence.is_absolute() else root / args.evidence

    logs = {
        "iverilog": evidence / "iverilog_seed_17717.log",
        "verilator_sva": evidence / "verilator_seed_17717.log",
    }
    parsed = {name: parse_log(path) for name, path in logs.items()}
    expected = {
        "seed": 17717,
        "heads": 3,
        "partial": 43200,
        "final": 14400,
        "group_stall": 0,
    }
    checks = {
        f"{simulator}_expected_{field}": data[field] == value
        for simulator, data in parsed.items()
        for field, value in expected.items()
    }
    checks["simulators_match"] = parsed["iverilog"] == parsed["verilator_sva"]
    checks["shell_verilator_lint"] = tool_log_is_clean(
        evidence / "verilator_shell_prod_lint.log"
    )
    checks["shell_yosys_check"] = tool_log_is_clean(
        evidence / "yosys_shell_prod.log",
        required="Found and reported 0 problems.",
    )
    docs359 = root / "docs/359_DATE终局冻结_20260813.md"
    checks["docs359_frozen"] = (
        docs359.is_file() and sha256(docs359) == DOCS359_SHA256
    )

    source_paths = [
        root / "rtl_qfit/qfit_local5_active_projection_tile.sv",
        root / "rtl_qfit/qfit_local5_score_active_projection_tile.sv",
        root / "rtl_qfit/qfit_local5_tagged_t450_job_engine.sv",
        root / "rtl_qfit/qfit_local5_cross_head_tile_executor.sv",
        root / "rtl_qfit/qfit_local5_encoder_t450_numeric_shell.sv",
        root / "tb_qfit/tb_qfit_local5_tagged_t450_job_engine.sv",
        root / "tb_qfit/tb_qfit_local5_cross_head_tile_executor.sv",
        root / "sim_qfit/run_local5_score_active_cross_head_checks.sh",
        root / "scripts/report_local5_score_active_cross_head.py",
    ]
    checks["all_sources_present"] = all(path.is_file() for path in source_paths)

    report = {
        "schema": "local5_score_active_cross_head_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "evidence_label": "rtl_component_system_boundary",
        "configuration": {
            "window": "T450_2x15x15",
            "input_heads": 3,
            "out_dim": 32,
            "child_path": "QS_to_FCSR_to_source_owned_to_TCFM5",
            "cross_head_accumulation": "external_Acc32",
            "backpressure": "token_weight_result_LFSR",
        },
        "checks": checks,
        "simulators": parsed,
        "claim_boundary": [
            "one synthetic T450 window using the existing deterministic oracle",
            "production score/relation/projection RTL and three-head OUT32 accumulation",
            "not final Local5 checkpoint evidence",
            "not bias, BN, requant, residual, neuron update, or a full encoder run",
            "12-block shell evidence is structural lint/hierarchy only",
            "no cycle number from this package belongs in docs/359",
        ],
        "source_sha256": {
            str(path.relative_to(root)): sha256(path)
            for path in source_paths
            if path.is_file()
        },
        "docs359_sha256": sha256(docs359) if docs359.is_file() else None,
    }
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (evidence / "report.md").write_text(
        "# Local5 当前生产前端 cross-head OUT32 闭合\n\n"
        f"状态：`{report['status']}`。证据等级：`[rtl组件系统边界]`。\n\n"
        "- Icarus 与 Verilator/SVA：3 heads、43,200 partial、14,400 final，逐项一致。\n"
        "- child 数据流：QS -> FCSR -> source-owned -> TCFM5。\n"
        "- 12-block production 参数壳通过 lint/hierarchy；未跑完整 encoder 数值。\n"
        "- 本包不绑定最终 Local5 checkpoint，不修改 `docs/359`。\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
