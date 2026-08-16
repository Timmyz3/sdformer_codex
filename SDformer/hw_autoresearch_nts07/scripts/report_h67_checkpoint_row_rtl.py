#!/usr/bin/env python3
"""Report checkpoint-bound H67 T450 row-engine RTL evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
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

    vector_manifest_path = args.vector_dir / "manifest.json"
    vector_manifest = json.loads(vector_manifest_path.read_text(encoding="utf-8"))
    if vector_manifest.get("schema") != "h67_checkpoint_t450_score_shiftmax_vectors_v1":
        raise ValueError("unexpected H67 vector schema")
    if vector_manifest.get("tokens_per_row") != 450:
        raise ValueError("production H67 RTL report requires T450")
    if not vector_manifest.get("independent_reference_matches_trace"):
        raise ValueError("independent software reference did not match trace")

    iverilog_log = (args.result_dir / "iverilog.log").read_text(encoding="utf-8")
    verilator_log = (args.result_dir / "verilator.log").read_text(encoding="utf-8")
    yosys_log = (args.result_dir / "yosys.log").read_text(encoding="utf-8")
    expected = (
        f"PASS tb_h67_checkpoint_rows rows={vector_manifest['row_count']} "
        f"tokens=450 checked_outputs={vector_manifest['expected_active_outputs']}"
    )
    if expected not in iverilog_log or expected not in verilator_log:
        raise ValueError("Icarus/Verilator did not both finish zero-mismatch")
    if "ERROR:" in yosys_log or "Warning: Wire" in yosys_log:
        raise ValueError("Yosys structural check contains errors or undriven wires")

    sources = {
        "generator": args.rtl_root / "scripts/generate_h67_checkpoint_row_vectors.py",
        "reporter": Path(__file__).resolve(),
        "runner": args.rtl_root / "sim_h67/run_h67_checkpoint_row_trace_checks.sh",
        "testbench": args.rtl_root / "tb_h67/tb_h67_checkpoint_rows.sv",
        "row_engine": args.rtl_root / "rtl_h67/h67_score_class_row_engine.sv",
        "score_leaf": args.rtl_root / "rtl_h67/h67_motionxor_score_q7.sv",
        "temporal_adapter": args.rtl_root / "rtl_h67/h67_temporal_pair_adapter.sv",
        "exp2_lut": args.rtl_root / "rtl_ttx/ttx_exp2_lut_q8.sv",
        "ceil_log2": args.rtl_root / "rtl_ttx/ttx_ceil_log2_u32.sv",
        "gate_quant": args.rtl_root / "rtl_ttx/ttx_gate_quant_q17.sv",
    }
    source_sha256 = {name: file_sha256(path) for name, path in sources.items()}
    output = {
        "schema": "h67_checkpoint_t450_score_shiftmax_rtl_report_v1",
        "status": "PASS",
        "scope": "checkpoint_bound_qk_score_scs_shiftmax_component_rtl_exact_not_projection_or_full_network",
        "vector_manifest": str(vector_manifest_path.resolve()),
        "vector_manifest_sha256": file_sha256(vector_manifest_path),
        "source_trace_manifest": vector_manifest["source_manifest"],
        "source_trace_manifest_sha256": vector_manifest["source_manifest_sha256"],
        "run_context": vector_manifest["run_context"],
        "rows": vector_manifest["row_count"],
        "tokens_per_row": 450,
        "token_vectors": vector_manifest["token_vector_count"],
        "active_outputs_checked": vector_manifest["expected_active_outputs"],
        "folded_zero_k_tokens": vector_manifest["expected_folded_tokens"],
        "independent_reference_matches_trace": True,
        "iverilog_zero_mismatch": True,
        "verilator_zero_mismatch": True,
        "yosys_check_passed": True,
        "source_sha256": source_sha256,
        "limits": [
            "This proves the H67 Q/K score, zero-K denominator fold, Shiftmax, and active gate output component.",
            "Projection, complete attention control, full encoder, SRAM macros, and full-network RTL are outside this report.",
        ],
    }
    report_json = args.result_dir / "report.json"
    report_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md = args.result_dir / "report.md"
    report_md.write_text(
        "\n".join(
            [
                "# H67 checkpoint-bound T450 score/Shiftmax RTL report",
                "",
                "- Status: **PASS**",
                f"- Rows: `{output['rows']}`",
                f"- Token vectors: `{output['token_vectors']}`",
                f"- Active outputs checked: `{output['active_outputs_checked']}`",
                f"- Folded zero-K tokens: `{output['folded_zero_k_tokens']}`",
                "- Icarus/Verilator: zero mismatch",
                "- Yosys: hierarchy/proc/opt/check/stat passed at `MAX_TOKENS=450`",
                "",
                "Scope: checkpoint-bound Q/K score, SCS zero-K denominator preservation, Shiftmax, and active gate output component RTL-exact. Projection and full-network RTL are not covered.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(report_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
