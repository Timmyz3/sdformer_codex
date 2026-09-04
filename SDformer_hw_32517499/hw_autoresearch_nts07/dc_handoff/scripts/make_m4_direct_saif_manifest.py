#!/usr/bin/env python3
"""Bind a direct VCS SAIF to the temporal-fenced M4 B400 vector contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def saif_duration_ns(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    duration_match = re.search(r"\(DURATION\s+([0-9]+(?:\.[0-9]+)?)\)", text)
    timescale_match = re.search(
        r"\(TIMESCALE\s+([0-9]+(?:\.[0-9]+)?)\s*(s|ms|us|ns|ps|fs)\)", text
    )
    if duration_match is None or timescale_match is None:
        raise ValueError("SAIF duration/timescale is missing")
    unit_ns = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0, "ps": 1e-3, "fs": 1e-6}
    return (
        float(duration_match.group(1))
        * float(timescale_match.group(1))
        * unit_ns[timescale_match.group(2)]
    )


def balanced_expression(text: str, start: int) -> str:
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("unterminated SAIF expression")


def saif_monitoring_summary(path: Path) -> dict[str, object]:
    """Fail closed if VCS did not actually monitor the M4 DUT.

    VCS can create a syntactically valid SAIF even when the selected UCLI
    policy misses SystemVerilog objects.  Duration and file identity alone do
    not detect that failure, so bind admission to recognizable DUT ports too.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    top_match = re.search(
        r"(?m)^\s*\(INSTANCE\s+tb_qfit_dual_line_descriptor_resident_real\s*$",
        text,
    )
    if top_match is None:
        raise ValueError("SAIF hierarchy lacks the M4 testbench root")
    top_scope = balanced_expression(text, top_match.start())
    dut_match = re.search(r"(?m)^\s*\(INSTANCE\s+dut\s*$", top_scope)
    if dut_match is None:
        raise ValueError("SAIF hierarchy lacks the direct M4 DUT scope")
    dut_scope = balanced_expression(top_scope, dut_match.start())
    net_match = re.search(r"(?m)^\s*\(NET\s*$", dut_scope)
    if net_match is None:
        raise ValueError("SAIF direct M4 DUT scope has no NET block")
    direct_net_scope = balanced_expression(dut_scope, net_match.start())
    record_pattern = re.compile(
        r"(?m)^\s*\(([^()\s]+)\s*\n\s*"
        r"\(T0\s+([0-9]+(?:\.[0-9]+)?)\)\s*"
        r"\(T1\s+([0-9]+(?:\.[0-9]+)?)\)\s*"
        r"\(TX\s+([0-9]+(?:\.[0-9]+)?)\)\s*\n\s*"
        r"\(TC\s+([0-9]+(?:\.[0-9]+)?)\)"
    )
    records = [
        (match.group(1), {
            "t0": float(match.group(2)),
            "t1": float(match.group(3)),
            "tx": float(match.group(4)),
            "tc": float(match.group(5)),
        })
        for match in record_pattern.finditer(direct_net_scope)
    ]
    required_signals = [
        "clk_core",
        "rst_core",
        "descriptor_valid",
        "weight_request_valid",
        "weight_response_valid",
        "output_valid",
        "controller_state",
        "protocol_error",
    ]
    def matching_records(base_name: str) -> list[dict[str, float]]:
        return [
            values
            for name, values in records
            if name == base_name or name.startswith(base_name + r"\[")
        ]

    missing = [name for name in required_signals if not matching_records(name)]
    if len(records) < 20 or missing:
        raise ValueError(
            "SAIF lacks M4 DUT activity: "
            f"direct_dut_records={len(records)} missing={missing}"
        )
    nonzero_required = (
        "clk_core",
        "descriptor_valid",
        "weight_request_valid",
        "weight_response_valid",
        "output_valid",
        "controller_state",
    )
    inactive = [
        name
        for name in nonzero_required
        if sum(item["tc"] for item in matching_records(name)) <= 0
    ]
    if inactive:
        raise ValueError(f"required direct DUT signals have zero activity: {inactive}")
    protocol = matching_records("protocol_error")
    if sum(item["tc"] + item["tx"] for item in protocol) != 0:
        raise ValueError("protocol_error toggled or became unknown during M4 SAIF")
    signal_time = sum(
        item["t0"] + item["t1"] + item["tx"] for _, item in records
    )
    tx_time = sum(item["tx"] for _, item in records)
    critical_activity = {
        name: {
            "records": len(matching_records(name)),
            "tc": sum(item["tc"] for item in matching_records(name)),
            "tx": sum(item["tx"] for item in matching_records(name)),
        }
        for name in (*nonzero_required, "protocol_error", "use_motion_q")
    }
    return {
        "all_saif_signal_records": len(record_pattern.findall(text)),
        "direct_dut_signal_records": len(records),
        "required_direct_dut_signals": required_signals,
        "critical_activity": critical_activity,
        "direct_dut_tx_signal_time_pct": (
            100.0 * tx_time / signal_time if signal_time else 0.0
        ),
        "empty_design_header_warning": re.search(r"\(DESIGN\s*\)", text) is not None,
    }


def vcs_identity(log_text: str) -> dict[str, str]:
    match = re.search(
        r"Compiler version\s+([^;\n]+);\s*Runtime version\s+([^;\n]+);",
        log_text,
    )
    if match is None:
        raise ValueError("VCS compiler/runtime identity is absent from simulation log")
    return {"compiler": match.group(1).strip(), "runtime": match.group(2).strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--simulation-log", type=Path, required=True)
    parser.add_argument("--saif", type=Path, required=True)
    parser.add_argument("--simv", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--ucli-script", type=Path, required=True)
    parser.add_argument("--strip-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (
        args.vector_manifest, args.trace, args.simulation_log, args.saif,
        args.simv, args.runner, args.ucli_script,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing M4 activity evidence: {path}")

    vectors = json.loads(args.vector_manifest.read_text(encoding="utf-8"))
    population = vectors.get("population", {})
    wall_cycles = population.get("m4_wall_cycles")
    if vectors.get("status") != "PASS_CHECKPOINT_BOUND_REAL_BITMAP_DESCRIPTOR_BATCHES" \
            or vectors.get("availability_mode") != "temporal_fenced" \
            or population.get("batches") != 400 \
            or len(vectors.get("sample_batches", {})) != 40 \
            or wall_cycles != 138893:
        raise ValueError("M4 temporal B400 activity population is not frozen")
    if sha256(args.trace) != vectors.get("sha256", {}).get("real_descriptors.txt"):
        raise ValueError("M4 activity trace SHA mismatch")
    log_text = args.simulation_log.read_text(encoding="utf-8", errors="replace")
    if (
        "request_beats=111373 bank_reads=1024946" not in log_text
        or "source_checks=1024946 wall_cycles=138893 ideal=1" not in log_text
        or re.search(r"Fatal:|^Error:", log_text, re.MULTILINE)
    ):
        raise ValueError("M4 direct-SAIF simulation did not preserve the ideal miter")

    duration_ns = saif_duration_ns(args.saif)
    monitoring = saif_monitoring_summary(args.saif)
    simulator_identity = vcs_identity(log_text)
    measured_cycles = round(duration_ns / 3.0)
    measurement_overhead = measured_cycles - wall_cycles
    if measurement_overhead != population["batches"]:
        raise ValueError(
            f"M4 SAIF interval is inconsistent: duration={duration_ns} ns "
            f"cycles={measured_cycles} overhead={measurement_overhead}"
        )
    payload = {
        "schema": "m4_direct_vcs_saif_manifest_v1",
        "status": "PASS_M4_DIRECT_VCS_SAIF_BOUNDED_EXPLORATORY",
        "design_name": "qfit_dual_line_descriptor_resident_engine",
        "activity_source_kind": "direct_vcs_saif",
        "source_activity": str(args.simulation_log.resolve()),
        "source_activity_sha256": sha256(args.simulation_log),
        "source_vcd": "",
        "source_vcd_sha256": "",
        "trace_root": str(args.trace.resolve()),
        "trace_sha256": sha256(args.trace),
        "vector_manifest": str(args.vector_manifest.resolve()),
        "vector_manifest_sha256": sha256(args.vector_manifest),
        "simulator": "Synopsys VCS " + simulator_identity["runtime"],
        "simulator_identity": simulator_identity,
        "simv": str(args.simv.resolve()),
        "simv_sha256": sha256(args.simv),
        "runner": str(args.runner.resolve()),
        "runner_sha256": sha256(args.runner),
        "ucli_script": str(args.ucli_script.resolve()),
        "ucli_script_sha256": sha256(args.ucli_script),
        "manifest_builder": str(Path(__file__).resolve()),
        "manifest_builder_sha256": sha256(Path(__file__).resolve()),
        "strip_path": args.strip_path,
        "warmup_cycles": 0,
        "measured_cycles": measured_cycles,
        "busy_cycles": wall_cycles,
        "measurement_overhead_cycles": measurement_overhead,
        "measurement_scope": "temporal_fenced_full_descriptor_execution_with_batch_fences",
        "activity_purpose": "identity_smoke",
        "paper_power_eligible": False,
        "workload_kind": "dual_line_descriptor_batch",
        "trace_scope": (
            "B400 stratified H67 ep35/Local5 ep44 Local+Motion cohort; all ten samples "
            "per identity, bounded source-kernel activity rather than a full-network run"
        ),
        "identity_root": str(args.vector_manifest.parent.resolve()),
        "saif_sha256": sha256(args.saif),
        "saif_duration_ns": duration_ns,
        "saif_signal_record_count": monitoring["all_saif_signal_records"],
        "saif_direct_dut_signal_record_count": monitoring[
            "direct_dut_signal_records"
        ],
        "saif_required_dut_signals": monitoring["required_direct_dut_signals"],
        "saif_critical_direct_dut_activity": monitoring["critical_activity"],
        "saif_direct_dut_tx_signal_time_pct": monitoring[
            "direct_dut_tx_signal_time_pct"
        ],
        "saif_empty_design_header_warning": monitoring[
            "empty_design_header_warning"
        ],
        "clock_period_ns": 3.0,
        "claim_boundary": (
            "Direct RTL SAIF is suitable for exploratory premacro PTPX only. B400 is "
            "bounded and paper_power_eligible remains false; no SRAM macro/SPEF power is present."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
