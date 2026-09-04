#!/usr/bin/env python3
"""Bind a wrapper VCD and measured interval to the real vector population."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


MEASUREMENT_RE = re.compile(
    r"SAIF_MEASUREMENT\s+design\s*=\s*(\S+)\s+start_group\s*=\s*(\d+)"
    r"\s+groups\s*=\s*(\d+)\s+measured_cycles\s*=\s*(\d+)"
    r"\s+scope\s*=\s*(\S+)"
)
LOCAL5_GROUP_RE = re.compile(
    r"GROUP\s+.*?group\s*=\s*(\d+)\s+cycles\s*=\s*(\d+).*?"
    r"score_service\s*=\s*(\d+).*?qsilent_rows\s*=\s*(\d+).*?"
    r"identk_rows\s*=\s*(\d+).*?overlap\s*=\s*(\d+)"
)
MOTION_ROW_RE = re.compile(
    r"MOTION_ACTIVITY_ROW\s+mode\s*=\s*(\S+)\s+row\s*=\s*(\d+)"
    r"\s+cycles\s*=\s*(\d+)\s+slots\s*=\s*(\d+)"
    r"\s+equal\s*=\s*(\d+)\s+emitted\s*=\s*(\d+)"
)


MOTION_PAPER_CONTRACTS = {
    "h67_fixed2s_mssb5_dc_top": {
        "busy_cycles": 112589,
        "slots": 62100,
        "equal": 28001,
    },
    "h67_rqtb2s_mssb5_dc_top": {
        "busy_cycles": 94891,
        "slots": 34099,
        "equal": 28001,
    },
}
DESIGN_CLOCK_PERIOD_PS = {
    "h67_fixed2s_mssb5_dc_top": 10_000,
    "h67_rqtb2s_mssb5_dc_top": 10_000,
    "local5_unified_out2_dc_top": 2_000,
    "local5_unified_out2_1rw_dc_top": 2_000,
}
LOCAL5_PAPER_BUSY_CYCLES = {
    "local5_unified_out2_dc_top": 155791,
    "local5_unified_out2_1rw_dc_top": 170269,
}


def paper_population_contract(
    design: str,
    start: int,
    count: int,
    workload_kind: str,
    workload_rows: list[dict[str, int]],
    busy_cycles: int,
    measurement_scope: str,
) -> tuple[bool, dict[str, bool], dict[str, int | float]]:
    """Apply workload-specific paper-power admission, not a generic size test."""
    if workload_kind == "motion_row" and design in MOTION_PAPER_CONTRACTS:
        anchor = MOTION_PAPER_CONTRACTS[design]
        totals = {
            "slots": sum(row["slots"] for row in workload_rows),
            "equal": sum(row["equal"] for row in workload_rows),
            "emitted_nonzero_rows": sum(row["emitted"] > 0 for row in workload_rows),
        }
        checks = {
            "start_at_zero": start == 0,
            "all_138_rows": count == 138 and len(workload_rows) == 138,
            "fair_lfsr_scope": measurement_scope == "fair_lfsr_row_execution",
            "frozen_busy_cycles": busy_cycles == anchor["busy_cycles"],
            "frozen_slot_total": totals["slots"] == anchor["slots"],
            "frozen_equal_total": totals["equal"] == anchor["equal"],
            "nontrivial_population": totals["emitted_nonzero_rows"] >= 100,
        }
        return all(checks.values()), checks, totals

    if workload_kind == "local5_group" and design in LOCAL5_PAPER_BUSY_CYCLES:
        nontrivial_count = sum(row["score_service"] > 0 for row in workload_rows)
        totals = {
            "score_service": sum(row["score_service"] for row in workload_rows),
            "nontrivial_groups": nontrivial_count,
            "nontrivial_ratio": nontrivial_count / count if count else 0.0,
        }
        checks = {
            "start_at_zero": start == 0,
            "all_100_groups": count == 100 and len(workload_rows) == 100,
            "matched_tile_scope": measurement_scope
            in {"full_load_compute_readback", "busy_projection"},
            "frozen_busy_cycles": busy_cycles == LOCAL5_PAPER_BUSY_CYCLES[design],
            "at_least_30_nontrivial_groups": nontrivial_count >= 30,
        }
        return all(checks.values()), checks, totals

    return False, {"known_paper_population": False}, {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def path_has_content(path: Path) -> bool:
    if path.is_file():
        return path.stat().st_size > 0
    return path.is_dir() and any(
        item.is_file() and item.stat().st_size > 0 for item in path.rglob("*")
    )


def parse_timescale_ps(fields: list[str]) -> float | None:
    units = {
        "s": 1e12, "ms": 1e9, "us": 1e6,
        "ns": 1e3, "ps": 1.0, "fs": 1e-3,
    }
    token = "".join(fields)
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([a-z]+)", token)
    if not match or match.group(2) not in units:
        return None
    return float(match.group(1)) * units[match.group(2)]


def vcd_metadata(path: Path) -> dict[str, object]:
    """Return scopes, timebase, extent, and dump_active intervals."""
    scopes: set[str] = set()
    stack: list[str] = []
    timescale_ps: float | None = None
    timescale_fields: list[str] = []
    in_timescale = False
    dump_active_code: str | None = None
    current_time = 0
    last_time = 0
    active_start: int | None = None
    active_ticks = 0
    active_intervals = 0
    value_changes = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if in_timescale:
                fields = stripped.split()
                if "$end" in fields:
                    timescale_fields.extend(fields[:fields.index("$end")])
                    timescale_ps = parse_timescale_ps(timescale_fields)
                    in_timescale = False
                else:
                    timescale_fields.extend(fields)
                continue
            if stripped.startswith("$timescale"):
                fields = stripped.split()[1:]
                if "$end" in fields:
                    timescale_ps = parse_timescale_ps(fields[:fields.index("$end")])
                else:
                    timescale_fields = fields
                    in_timescale = True
                continue
            if stripped.startswith("$scope "):
                fields = stripped.split()
                if len(fields) >= 4:
                    stack.append(fields[2])
                    scopes.add("/".join(stack))
            elif stripped.startswith("$upscope"):
                if stack:
                    stack.pop()
            elif stripped.startswith("$var "):
                fields = stripped.split()
                if len(fields) >= 6 and fields[4] == "dump_active":
                    dump_active_code = fields[3]
            elif stripped.startswith("#"):
                current_time = int(stripped[1:])
                last_time = current_time
            elif stripped and not stripped.startswith("$"):
                value_changes += 1
                if dump_active_code and stripped == f"1{dump_active_code}":
                    if active_start is not None:
                        raise SystemExit("duplicate dump_active assertion in VCD")
                    active_start = current_time
                    active_intervals += 1
                elif dump_active_code and stripped == f"0{dump_active_code}":
                    if active_start is not None:
                        active_ticks += current_time - active_start
                        active_start = None
    if active_start is not None:
        active_ticks += last_time - active_start
    return {
        "scopes": scopes,
        "timescale_ps": timescale_ps,
        "last_timestamp_ticks": last_time,
        "active_ticks": active_ticks,
        "active_intervals": active_intervals,
        "value_changes": value_changes,
        "dump_active_code": dump_active_code,
    }


def selected_workload(text: str, start: int, count: int) -> tuple[str, list[dict[str, int]]]:
    end = start + count
    local_rows = [
        {
            "index": int(match.group(1)),
            "busy_cycles": int(match.group(2)),
            "score_service": int(match.group(3)),
            "qsilent_rows": int(match.group(4)),
            "identk_rows": int(match.group(5)),
            "overlap": int(match.group(6)),
        }
        for match in LOCAL5_GROUP_RE.finditer(text)
        if start <= int(match.group(1)) < end
    ]
    motion_rows = [
        {
            "index": int(match.group(2)),
            "busy_cycles": int(match.group(3)),
            "slots": int(match.group(4)),
            "equal": int(match.group(5)),
            "emitted": int(match.group(6)),
        }
        for match in MOTION_ROW_RE.finditer(text)
        if start <= int(match.group(2)) < end
    ]
    if local_rows and motion_rows:
        raise SystemExit("log contains both Local5 and Motion workload rows")
    if local_rows:
        return "local5_group", local_rows
    if motion_rows:
        return "motion_row", motion_rows
    return "unknown", []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", required=True)
    parser.add_argument("--vcd", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--strip-path", required=True)
    parser.add_argument(
        "--purpose",
        choices=("identity_smoke", "paper_power_compute", "paper_power_with_io"),
        required=True,
    )
    parser.add_argument("--measurement-scope", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    text = args.log.read_text(encoding="utf-8", errors="replace")
    if re.search(r"(?im)^\s*%?(?:ERROR|FATAL)(?:[-: ]|$)", text):
        raise SystemExit("simulation log contains ERROR/FATAL")
    matches = MEASUREMENT_RE.findall(text)
    if len(matches) != 1:
        raise SystemExit(f"expected one SAIF_MEASUREMENT line, found {len(matches)}")
    design, start_group, groups, measured_cycles, logged_scope = matches[0]
    start = int(start_group)
    count = int(groups)
    workload_kind, workload_rows = selected_workload(text, start, count)
    vcd = vcd_metadata(args.vcd) if args.vcd.is_file() else {}
    scopes = vcd.get("scopes", set())
    busy_cycles = sum(row["busy_cycles"] for row in workload_rows)
    expected_indices = list(range(start, start + count))
    actual_indices = sorted(row["index"] for row in workload_rows)
    indices_exact = actual_indices == expected_indices
    population_eligible, population_checks, population_totals = (
        paper_population_contract(
            design,
            start,
            count,
            workload_kind,
            workload_rows,
            busy_cycles,
            logged_scope,
        )
    )
    clock_period_ps = DESIGN_CLOCK_PERIOD_PS.get(design)
    expected_active_ps = (
        int(measured_cycles) * clock_period_ps if clock_period_ps else None
    )
    active_ps = None
    if vcd.get("timescale_ps") is not None:
        active_ps = int(
            round(float(vcd.get("active_ticks", 0)) * float(vcd["timescale_ps"]))
        )
    simulation_pass_count = (
        text.count("PASS Local5 score-to-projection")
        + text.count("PASS Motion wrapper activity")
    )
    vcd_duration_exact = active_ps == expected_active_ps
    single_interval = int(vcd.get("active_intervals", 0)) == 1
    purpose_matches_scope = (
        (workload_kind == "motion_row" and args.purpose == "paper_power_compute")
        or (
            workload_kind == "local5_group"
            and (
                (
                    logged_scope == "full_load_compute_readback"
                    and args.purpose == "paper_power_with_io"
                )
                or (
                    logged_scope == "busy_projection"
                    and args.purpose == "paper_power_compute"
                )
            )
        )
    )
    paper_power_eligible = (
        args.purpose != "identity_smoke"
        and purpose_matches_scope
        and indices_exact
        and population_eligible
        and vcd_duration_exact
        and single_interval
    )
    checks = {
        "design": design == args.design,
        "simulation_pass_once": simulation_pass_count == 1,
        "vcd_nonempty": args.vcd.is_file() and args.vcd.stat().st_size > 0,
        "trace_nonempty": path_has_content(args.trace_root),
        "measured_cycles": int(measured_cycles) > 0,
        "workload_rows_complete": indices_exact,
        "busy_cycles": busy_cycles > 0,
        "measurement_covers_busy": int(measured_cycles) >= busy_cycles,
        "strip_path_in_vcd": args.strip_path in scopes,
        "measurement_scope": logged_scope == args.measurement_scope,
        "vcd_timescale": vcd.get("timescale_ps") is not None,
        "vcd_has_value_changes": int(vcd.get("value_changes", 0)) > 0,
        "vcd_dump_active_signal": bool(vcd.get("dump_active_code")),
        "vcd_active_duration": vcd_duration_exact,
        "paper_population_single_vcd_interval": (
            args.purpose == "identity_smoke" or single_interval
        ),
        "paper_power_purpose_matches_scope": (
            args.purpose == "identity_smoke" or purpose_matches_scope
        ),
        "paper_population_contract": (
            args.purpose == "identity_smoke" or paper_power_eligible
        ),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "design_name": args.design,
        "source_vcd": str(args.vcd.resolve()),
        "source_vcd_sha256": sha256_file(args.vcd),
        "trace_root": str(args.trace_root.resolve()),
        "trace_sha256": sha256_tree(args.trace_root),
        "simulator": "verilator --assert --trace",
        "strip_path": args.strip_path,
        "warmup_cycles": 0,
        "measured_cycles": int(measured_cycles),
        "busy_cycles": busy_cycles,
        "measurement_overhead_cycles": int(measured_cycles) - busy_cycles,
        "measurement_scope": logged_scope,
        "clock_period_ps": clock_period_ps,
        "vcd_timescale_ps": vcd.get("timescale_ps"),
        "vcd_last_timestamp_ticks": vcd.get("last_timestamp_ticks"),
        "vcd_active_ticks": vcd.get("active_ticks"),
        "vcd_active_duration_ps": active_ps,
        "vcd_active_intervals": vcd.get("active_intervals"),
        "activity_purpose": args.purpose,
        "paper_power_eligible": paper_power_eligible,
        "paper_population_checks": population_checks,
        "paper_population_totals": population_totals,
        "workload_kind": workload_kind,
        "workload_rows": workload_rows,
        "trace_scope": f"start_group={start_group}, groups={groups}",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
