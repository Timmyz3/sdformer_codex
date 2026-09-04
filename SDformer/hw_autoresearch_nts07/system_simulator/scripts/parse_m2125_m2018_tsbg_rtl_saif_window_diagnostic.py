#!/opt/anaconda3/bin/python3
"""Fail-closed parser for the additive M2125 RTL-SAIF diagnostic.

No EDA is invoked here.  M2125 is deliberately diagnostic-only: it asks
whether deterministic RTL initialization plus phase-matched settled-negedge
stops can produce two complete, unknown-free DUT-only SAIF windows while the
frozen M2051/M2018 functional and cycle ledgers remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys


AXES = {
    "ordinary_lru4": {"cycles": 20292, "reads": 14304, "mode": 0},
    "tsbg_b4": {"cycles": 7569, "reads": 4608, "mode": 1},
}
EXPECTED_RECORDS = 93971
CRITICAL = (
    "mem_req_valid", "mem_rsp_valid", "bridge_valid", "commit_valid",
    "mem_req_accept", "mem_rsp_accept", "bridge_accept", "commit_accept",
)


class Failure(RuntimeError):
    pass


def need(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> str:
    need(path.is_file() and not path.is_symlink(), f"missing/symlink: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_runtime(path: Path, axis: str) -> dict[str, object]:
    need(axis in AXES, "unknown axis")
    text = read(path)
    begin = re.findall(
        r"^M2125_RTL_SAIF_WINDOW_BEGIN sampling=settled_negedge .*"
        r"preload_cycles=383 time_ns=([0-9.]+)$", text, re.MULTILINE)
    end = re.findall(
        rf"^M2125_RTL_SAIF_WINDOW_END axis={re.escape(axis)} "
        r"sampling=settled_negedge measurement_cycles=([0-9]+) "
        r"scalar_weight_reads=([0-9]+) duration_ns=([0-9.]+)$",
        text, re.MULTILINE)
    passed = re.findall(
        r"^PASS_M2125_RTL_SAIF_WINDOW_DIAGNOSTIC_AXIS "
        r"ledger_exact=1 initreg_diagnostic_only=1 paper_citable=0$",
        text, re.MULTILINE)
    need(len(begin) == 1 and len(end) == 1 and len(passed) == 1,
         "runtime marker count")
    cycles, reads, duration = int(end[0][0]), int(end[0][1]), float(end[0][2])
    cfg = AXES[axis]
    need(cycles == cfg["cycles"] and reads == cfg["reads"],
         "runtime frozen ledger drift")
    need(math.isclose(duration, cycles * 3.0, rel_tol=0.0, abs_tol=1e-6),
         "runtime phase duration drift")
    need(not re.search(r"(^|\n)(Fatal:|Error:)|Assertion failed|M2125 .*drift",
                       text), "runtime failure token")
    return {
        "axis": axis,
        "sha256": sha256(path),
        "begin_time_ns": float(begin[0]),
        "measurement_cycles": cycles,
        "scalar_weight_reads": reads,
        "duration_ns": duration,
        "completion_ledger_exact": True,
    }


def parse_saif(path: Path, axis: str) -> dict[str, object]:
    need(axis in AXES, "unknown axis")
    text = read(path)
    scale_match = re.findall(
        r"\(TIMESCALE\s+([0-9.eE+-]+)\s+([A-Za-z]+)\)", text)
    duration_match = re.findall(r"\(DURATION\s+([0-9.eE+-]+)\)", text)
    need(len(scale_match) == 1 and len(duration_match) == 1,
         "nonunique SAIF header")
    unit_scale_ns = {
        "s": 1.0e9, "ms": 1.0e6, "us": 1.0e3,
        "ns": 1.0, "ps": 1.0e-3, "fs": 1.0e-6,
    }
    scale, unit = float(scale_match[0][0]), scale_match[0][1]
    need(unit in unit_scale_ns, f"unsupported SAIF unit: {unit}")
    duration_raw = float(duration_match[0])
    duration_ns = duration_raw * scale * unit_scale_ns[unit]
    expected_ns = AXES[axis]["cycles"] * 3.0
    need(math.isclose(duration_ns, expected_ns, rel_tol=0.0, abs_tol=1e-6),
         f"duration {duration_ns} != {expected_ns}")

    records = re.findall(
        r"\(T0\s+([0-9.eE+-]+)\)\s*\(T1\s+([0-9.eE+-]+)\)\s*"
        r"\(TX\s+([0-9.eE+-]+)\)\s*\(TC\s+([0-9.eE+-]+)\)", text)
    need(len(records) == EXPECTED_RECORDS,
         f"DUT-only SAIF record coverage {len(records)} != {EXPECTED_RECORDS}")
    tx_nonzero = 0
    tx_sum = 0.0
    toggled = 0
    conservation_failures = 0
    for values in records:
        t0, t1, tx, tc = map(float, values)
        need(min(t0, t1, tx, tc) >= 0.0, "negative SAIF field")
        tx_nonzero += int(tx != 0.0)
        tx_sum += tx
        toggled += int(tc > 0.0)
        conservation_failures += int(not math.isclose(
            t0 + t1 + tx, duration_raw, rel_tol=0.0, abs_tol=1e-6))
    need(tx_nonzero == 0 and tx_sum == 0.0,
         f"SAIF unknown activity: records={tx_nonzero} sum={tx_sum}")
    need(conservation_failures == 0,
         f"SAIF conservation failures: {conservation_failures}")
    need(toggled >= 20, f"insufficient nonzero-toggle records: {toggled}")

    critical: dict[str, int] = {}
    for token in CRITICAL:
        values = re.findall(
            rf"\({re.escape(token)}(?:\\?\[[^\]]+\])?\s+"
            rf"\(T0\s+[0-9.eE+-]+\)\s*\(T1\s+[0-9.eE+-]+\)\s*"
            rf"\(TX\s+0(?:\.0+)?\)\s*\(TC\s+([0-9.eE+-]+)\)", text)
        count = sum(float(value) > 0.0 for value in values)
        need(count > 0, f"missing/zero critical activity: {token}")
        critical[token] = count
    return {
        "axis": axis,
        "sha256": sha256(path),
        "duration_raw": duration_raw,
        "duration_ns": duration_ns,
        "expected_cycles": AXES[axis]["cycles"],
        "record_count": len(records),
        "nonzero_toggle_record_count": toggled,
        "tx_nonzero_record_count": tx_nonzero,
        "tx_sum": tx_sum,
        "conservation_failures": conservation_failures,
        "critical_nonzero_record_counts": critical,
    }


def final_result(root: Path, output: Path) -> dict[str, object]:
    axes: dict[str, object] = {}
    for axis in AXES:
        axis_root = root / axis
        axes[axis] = {
            "runtime": parse_runtime(axis_root / "rtl_sim.log", axis),
            "rtl_saif": parse_saif(axis_root / "rtl_execute.saif", axis),
        }
    result = {
        "schema": "m2125_m2018_tsbg_rtl_saif_window_diagnostic_result_r1_v1",
        "status": "PASS_RAW_M2127_M2125_RTL_SAIF_DIAGNOSTIC_PENDING_M2128_RESULT_HAMMER",
        "axes": axes,
        "comparison": {
            "ordinary_cycles": AXES["ordinary_lru4"]["cycles"],
            "tsbg_cycles": AXES["tsbg_b4"]["cycles"],
            "cycle_ratio_diagnostic": (
                AXES["ordinary_lru4"]["cycles"] /
                AXES["tsbg_b4"]["cycles"]),
            "ordinary_scalar_weight_reads": AXES["ordinary_lru4"]["reads"],
            "tsbg_scalar_weight_reads": AXES["tsbg_b4"]["reads"],
        },
        "claim_boundary": {
            "vcs_only_rtl_saif_diagnostic": True,
            "fixed_ep34_slot42_directed_int8_weights": True,
            "completion_and_cycle_ledgers_exact": True,
            "compile_initreg_random_instrumentation": True,
            "runtime_initreg_zero_selection": True,
            "initreg_is_silicon_initialization": False,
            "dc_run": False,
            "ptpx_run": False,
            "mapped_netlist_activity": False,
            "power_or_energy": False,
            "component_speedup_admitted": False,
            "system_speedup": False,
            "paper_citable": False,
        },
    }
    write_json(output, result)
    return result


def static_check() -> dict[str, object]:
    source = Path(__file__).read_text()
    checks = {
        "two_axes": set(AXES) == {"ordinary_lru4", "tsbg_b4"},
        "frozen_cycles": [AXES[x]["cycles"] for x in AXES] == [20292, 7569],
        "frozen_reads": [AXES[x]["reads"] for x in AXES] == [14304, 4608],
        "exact_record_gate": "len(records) == EXPECTED_RECORDS" in source,
        "all_tx_zero_gate": "tx_nonzero == 0 and tx_sum == 0.0" in source,
        "exact_conservation_gate": "conservation_failures == 0" in source,
        "critical_nonzero_gate": "missing/zero critical activity" in source,
        "diagnostic_boundary": '"paper_citable": False' in source,
    }
    need(all(checks.values()), f"static checks failed: {checks}")
    return {"status": "PASS_M2125_STATIC_PARSER", "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("static")
    runtime = sub.add_parser("runtime")
    runtime.add_argument("--axis", required=True, choices=AXES)
    runtime.add_argument("--path", required=True, type=Path)
    saif = sub.add_parser("saif")
    saif.add_argument("--axis", required=True, choices=AXES)
    saif.add_argument("--path", required=True, type=Path)
    final = sub.add_parser("final")
    final.add_argument("--root", required=True, type=Path)
    final.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "static":
            value = static_check()
        elif args.command == "runtime":
            value = parse_runtime(args.path, args.axis)
        elif args.command == "saif":
            value = parse_saif(args.path, args.axis)
        else:
            value = final_result(args.root, args.output)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    except Failure as exc:
        print(f"M2125_FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
