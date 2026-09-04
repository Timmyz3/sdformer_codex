#!/usr/bin/env python3
"""Summarize checkpoint-bound Local5 score/Shiftmax RTL evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--rtl-root", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.vector_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    logs = {
        "iverilog": args.result_dir / "iverilog.log",
        "verilator": args.result_dir / "verilator.log",
        "yosys": args.result_dir / "yosys.log",
    }
    expected = int(manifest["vector_count"])
    marker = f"PASS tb_local5_score_shiftmax_vectors vectors={expected}"
    checks = {
        "trace_gate_matches_independent_reference": bool(
            manifest["independent_reference"]["trace_gate_zero_mismatch"]
        ),
        "iverilog_zero_mismatch": marker in logs["iverilog"].read_text(
            encoding="utf-8", errors="replace"
        ),
        "verilator_zero_mismatch": marker in logs["verilator"].read_text(
            encoding="utf-8", errors="replace"
        ),
        "yosys_check_passed": (
            logs["yosys"].read_text(encoding="utf-8", errors="replace").count(
                "End of script"
            )
            == 2
            and "ERROR:"
            not in logs["yosys"].read_text(encoding="utf-8", errors="replace")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Local5 checkpoint score RTL checks failed: {checks}")
    rtl_paths = [
        args.rtl_root / "rtl_local5/local5_axnor_score_q7.sv",
        args.rtl_root / "rtl_local5/local5_shiftmax5_q17.sv",
        args.rtl_root / "tb_local5/tb_local5_score_shiftmax_vectors.sv",
    ]
    result = {
        "schema": "local5_checkpoint_score_rtl_report_v1",
        "status": "PASS",
        "evidence_scope": (
            "checkpoint_bound_post_g0_qk_score_shiftmax_rtl_exact_"
            "not_projection_or_full_network"
        ),
        "vector_manifest": str(manifest_path.resolve()),
        "vector_manifest_sha256": sha256(manifest_path),
        "vectors": expected,
        "groups": int(manifest["selection"]["groups"]),
        "checks": checks,
        "rtl_bindings": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in rtl_paths
        ],
        "logs": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in logs.items()
        },
    }
    (args.result_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.result_dir / "report.md").write_text(
        "\n".join(
            [
                "# Local5 checkpoint-bound score/Shiftmax RTL exact",
                "",
                f"- 状态：**{result['status']}**",
                f"- 真实 post-G0 groups：{result['groups']}",
                f"- T450 token vectors：{result['vectors']}",
                "- Icarus/Verilator：逐 vector Q7 score 与 Q1.7 gate 零失配。",
                "- 证据边界：只覆盖 checkpoint-bound Q/K 到 score/Shiftmax；",
                "  projection 与 full-network 分别签核，不得合并表述。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(args.result_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
