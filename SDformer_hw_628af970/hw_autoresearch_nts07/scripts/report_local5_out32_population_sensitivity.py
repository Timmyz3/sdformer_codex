#!/usr/bin/env python3
"""Report current Local5 OUT32 population RTL without changing frozen tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


GROUP_RE = re.compile(
    r"^GROUP .* group=(?P<group>\d+) cycles=(?P<cycles>\d+) "
    r"score_rows=(?P<score_rows>\d+) score_service=(?P<score_service>\d+) "
    r"score_direct_rows=(?P<score_direct>\d+) qsilent_rows=(?P<qsilent>\d+) "
    r"identk_rows=(?P<identk>\d+) overlap=(?P<overlap>\d+) "
    r"active=(?P<active>\d+).* terms=(?P<terms>\d+) updates=(?P<updates>\d+)"
)
PASS_RE = re.compile(
    r"^PASS Local5 score-to-projection .* groups=100 total_cycles=(\d+)"
)
BAD_RE = re.compile(r"%Error|Assertion failed|MISMATCH|\$fatal|\bFAIL\b")
LEDGER_FIELDS = (
    "cycles", "score_rows", "score_service", "score_direct", "qsilent",
    "identk", "overlap", "active", "terms", "updates",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_log(path: Path) -> dict[int, dict[str, int]]:
    text = path.read_text(encoding="utf-8", errors="strict")
    if BAD_RE.search(text):
        raise ValueError(f"failure marker in {path}")
    rows: dict[int, dict[str, int]] = {}
    passes: list[int] = []
    for line in text.splitlines():
        if match := GROUP_RE.fullmatch(line):
            row = {key: int(value) for key, value in match.groupdict().items()}
            group = row.pop("group")
            if group in rows:
                raise ValueError(f"duplicate group {group} in {path}")
            rows[group] = row
        if match := PASS_RE.fullmatch(line):
            passes.append(int(match.group(1)))
    if sorted(rows) != list(range(100)):
        raise ValueError(f"100-group population is incomplete in {path}")
    total = sum(row["cycles"] for row in rows.values())
    if passes != [total]:
        raise ValueError(f"PASS receipt differs from row ledger in {path}")
    if any(set(row) != set(LEDGER_FIELDS) for row in rows.values()):
        raise ValueError(f"GROUP schema differs in {path}")
    return rows


def load_manifest(path: Path, out_dim: int) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    shape = manifest.get("shape", {})
    selection = manifest.get("selection", {})
    if (
        manifest.get("schema") != "local5_score_projection_vectors_v1"
        or shape.get("sources") != 450
        or shape.get("head_dim") != 32
        or shape.get("out_dim") != out_dim
        or selection.get("groups") != 100
        or len(selection.get("rows", [])) != 100
        or manifest.get("weight_mode")
        != "checkpoint_theta_folded_dyadic_int8_head_slice"
    ):
        raise ValueError(f"vector manifest contract mismatch: {path}")
    for key in ("source_manifest", "source_payload"):
        source = Path(manifest[key])
        if not source.is_file() or sha256(source) != manifest[f"{key}_sha256"]:
            raise ValueError(f"source binding differs for {key}: {path}")
    return manifest


def workload_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        row[key]
        for key in (
            "sample", "stage", "block", "window", "head",
            "input_group_index", "active_sources", "terms", "updates",
        )
    )


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def compare(
    baseline: dict[int, dict[str, int]], candidate: dict[int, dict[str, int]]
) -> dict[str, Any]:
    left = sum(row["cycles"] for row in baseline.values())
    right = sum(row["cycles"] for row in candidate.values())
    ratios = [baseline[index]["cycles"] / candidate[index]["cycles"] for index in range(100)]
    return {
        "baseline_cycles": left,
        "candidate_cycles": right,
        "speedup": left / right,
        "cycle_reduction_ratio": 1.0 - right / left,
        "per_group": {
            "min": min(ratios),
            "p50": percentile(ratios, 0.50),
            "p95": percentile(ratios, 0.95),
            "max": max(ratios),
            "wins": sum(value > 1.0 for value in ratios),
            "ties": sum(value == 1.0 for value in ratios),
            "losses": sum(value < 1.0 for value in ratios),
        },
    }


def build_report(
    vector_manifest_path: Path,
    out2_vector_manifest_path: Path,
    t450_log_path: Path,
    rolling_log_path: Path,
    icarus_t450_log_path: Path,
    icarus_rolling_log_path: Path,
    out2_t450_log_path: Path,
    out2_rolling_log_path: Path,
) -> dict[str, Any]:
    out32_manifest = load_manifest(vector_manifest_path, 32)
    out2_manifest = load_manifest(out2_vector_manifest_path, 2)
    out32_rows = out32_manifest["selection"]["rows"]
    out2_rows = out2_manifest["selection"]["rows"]
    if [workload_identity(row) for row in out32_rows] != [
        workload_identity(row) for row in out2_rows
    ]:
        raise ValueError("OUT2 and OUT32 workload populations differ")
    for row in out32_rows:
        channels = row.get("projection_output_channels")
        if (
            not isinstance(channels, list)
            or len(channels) != 32
            or channels != list(range(channels[0], channels[0] + 32))
            or channels[0] % 32
        ):
            raise ValueError("OUT32 row is not one aligned contiguous output tile")

    t450 = parse_log(t450_log_path)
    rolling = parse_log(rolling_log_path)
    icarus_t450 = parse_log(icarus_t450_log_path)
    icarus_rolling = parse_log(icarus_rolling_log_path)
    out2_t450 = parse_log(out2_t450_log_path)
    out2_rolling = parse_log(out2_rolling_log_path)
    if icarus_t450 != t450 or icarus_rolling != rolling:
        raise ValueError("Icarus and Verilator OUT32 ledgers differ")
    for candidate_name, candidate in (
        ("rolling", rolling),
        ("out2_t450", out2_t450),
        ("out2_rolling", out2_rolling),
    ):
        for index in range(100):
            for field in LEDGER_FIELDS[1:]:
                if candidate[index][field] != t450[index][field]:
                    raise ValueError(
                        f"{candidate_name} workload ledger differs at group {index} field {field}"
                    )
    out_dim_cycle_invariant = all(
        t450[index]["cycles"] == out2_t450[index]["cycles"]
        and rolling[index]["cycles"] == out2_rolling[index]["cycles"]
        for index in range(100)
    )
    comparison = compare(t450, rolling)
    checks = 100 * 450 * 32
    return {
        "schema": "local5_out32_population_sensitivity_rtl_v1",
        "status": "PASS",
        "evidence": "[rtl]+[population]+[real-checkpoint-int8]+[OUT_DIM=32]",
        "scope": (
            "100 sample-disjoint population-stage-weighted raw-Q/K groups; "
            "score/Shiftmax5 through relation/TCFM5 to one aligned 32-channel "
            "output tile; pre-bias/pre-BN/pre-requant; not encoder"
        ),
        "population": {
            "groups": 100,
            "stage_counts": out32_manifest["selection"]["stage_counts"],
            "out2_out32_workload_identity_exact": True,
        },
        "correctness": {
            "acc32_values_checked": checks,
            "acc32_mismatch": 0,
            "score_rows_checked": 45_000,
            "score_gate_scalar_checks": 450_000,
            "verilator_assert": True,
            "icarus_verilator_per_group_exact": True,
        },
        "cycles": {
            "t450_qsilent": comparison["baseline_cycles"],
            "rolling_qsilent": comparison["candidate_cycles"],
            "speedup": comparison["speedup"],
            "cycle_reduction_ratio": comparison["cycle_reduction_ratio"],
            "per_group": comparison["per_group"],
            "out2_out32_busy_cycle_invariant": out_dim_cycle_invariant,
        },
        "work": {
            field: sum(row[field] for row in rolling.values())
            for field in LEDGER_FIELDS[1:]
        },
        "physical_width": {
            "accumulator_payload_bits": 5 * 90 * 32 * 32,
            "weight_values_per_group": 32 * 32,
            "output_channels_per_tile": 32,
        },
        "claim_boundary": [
            "busy cycles exclude weight load and Acc32 readback in the testbench",
            "32 channels execute spatially in the packed vector backend; cycle invariance is not a free-area claim",
            "one output tile per selected input head, not all output tiles or cross-head finalization",
            "no full encoder, decoder, DC, STA, SAIF, PTPX, or ASIC PPA claim",
            "this side evidence does not modify docs/359 frozen columns",
        ],
        "provenance": {
            "vector_manifest": str(vector_manifest_path.resolve()),
            "vector_manifest_sha256": sha256(vector_manifest_path),
            "out2_vector_manifest": str(out2_vector_manifest_path.resolve()),
            "out2_vector_manifest_sha256": sha256(out2_vector_manifest_path),
            "logs": {
                str(path.resolve()): sha256(path)
                for path in (
                    t450_log_path, rolling_log_path,
                    icarus_t450_log_path, icarus_rolling_log_path,
                    out2_t450_log_path, out2_rolling_log_path,
                )
            },
        },
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    cycles = report["cycles"]
    correctness = report["correctness"]
    work = report["work"]
    lines = [
        "# Local5 OUT32 Population RTL Sensitivity",
        "",
        f"- Evidence: `{report['evidence']}`.",
        f"- Scope: {report['scope']}.",
        f"- Acc32 checks: `{correctness['acc32_values_checked']:,}`, mismatch `0`.",
        f"- Query-Silent+T450 -> Query-Silent+rolling: `{cycles['t450_qsilent']:,} -> {cycles['rolling_qsilent']:,} = {cycles['speedup']:.4f}x`.",
        f"- Per-group min/p50/p95: `{cycles['per_group']['min']:.4f}/{cycles['per_group']['p50']:.4f}/{cycles['per_group']['p95']:.4f}x`.",
        f"- OUT2/OUT32 projection-busy cycles are row-exact invariant: `{'PASS' if cycles['out2_out32_busy_cycle_invariant'] else 'FAIL'}`.",
        f"- Conserved work: descriptors `{work['active']:,}`, terms `{work['terms']:,}`, updates `{work['updates']:,}`.",
        "",
        "The packed TCFM5 backend computes one 32-channel output tile spatially. The invariant busy-cycle result demonstrates width scaling of the frozen dataflow, while accumulator and multiplier resources scale with OUT_DIM; it is not a zero-cost throughput claim.",
        "",
        "## Claim Boundary",
        "",
    ]
    lines.extend(f"- {item}" for item in report["claim_boundary"])
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-manifest", type=Path, required=True)
    parser.add_argument("--out2-vector-manifest", type=Path, required=True)
    parser.add_argument("--t450-log", type=Path, required=True)
    parser.add_argument("--rolling-log", type=Path, required=True)
    parser.add_argument("--icarus-t450-log", type=Path, required=True)
    parser.add_argument("--icarus-rolling-log", type=Path, required=True)
    parser.add_argument("--out2-t450-log", type=Path, required=True)
    parser.add_argument("--out2-rolling-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        args.vector_manifest, args.out2_vector_manifest,
        args.t450_log, args.rolling_log,
        args.icarus_t450_log, args.icarus_rolling_log,
        args.out2_t450_log, args.out2_rolling_log,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    write_markdown(args.output_dir / "report.md", report)
    print(
        "PASS Local5 OUT32 population RTL "
        f"cycles={report['cycles']['t450_qsilent']}/"
        f"{report['cycles']['rolling_qsilent']} "
        f"acc32={report['correctness']['acc32_values_checked']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
