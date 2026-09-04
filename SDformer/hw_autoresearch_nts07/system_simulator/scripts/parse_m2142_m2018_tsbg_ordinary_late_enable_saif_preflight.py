#!/opt/anaconda3/bin/python3
"""Fail-closed parser for M2142's ordinary late-enable SAIF preflight.

M2142 is diagnostic only.  It tests whether enabling the native VCS activity
observer before reset/preload and resetting activity history at the first stop
removes M2139's observer-state TX fingerprint.  It admits neither power nor a
TSBG comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys


EXPECTED_CYCLES = 20292
EXPECTED_READS = 14304
EXPECTED_DURATION_NS = 60876.0
EXPECTED_RECORDS = 93971
EXPECTED_INTERNAL_ELEMENTS = 228
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


def one_match(pattern: str, text: str, label: str) -> re.Match[str]:
    rows = list(re.finditer(pattern, text, re.MULTILINE))
    need(len(rows) == 1, f"{label} marker count {len(rows)} != 1")
    return rows[0]


def parse_runtime(path: Path) -> dict[str, object]:
    text = read(path)
    phase_matches: list[re.Match[str]] = []
    expected_actions = {
        1: "power_enable", 2: "run_reset_and_preload",
        3: "first_stop_reached", 4: "power_reset",
        5: "second_stop_reached", 6: "power_disable", 7: "power_report",
    }
    for order, action in expected_actions.items():
        phase_matches.append(one_match(
            rf"^M2142_UCLI_PHASE order={order} action={action}(?: .*)?$",
            text, f"UCLI phase {order}"))
    need(all(left.start() < right.start()
             for left, right in zip(phase_matches, phase_matches[1:])),
         "UCLI phase order drift")

    census = one_match(
        r"^M2142_INTERNAL_KNOWNNESS_CENSUS phase=pre_power_reset "
        r"row_live=([0-9]+)/192 row_live_one=([0-9]+) "
        r"cache_valid=([0-9]+)/4 cache_valid_one=([0-9]+) "
        r"slot_valid=([0-9]+)/8 slot_valid_one=([0-9]+) "
        r"bridge_overflow=([0-9]+)/16 bridge_overflow_one=([0-9]+) "
        r"rsp_shape_legal=([0-9]+)/8 rsp_shape_legal_one=([0-9]+) "
        r"total=([0-9]+)/228 observe_only=1 force=0 deposit=0 mask=0 "
        r"rtl_edit=0$", text, "internal census")
    known = [int(census.group(index)) for index in (1, 3, 5, 7, 9)]
    ones = [int(census.group(index)) for index in (2, 4, 6, 8, 10)]
    total = int(census.group(11))
    need(known == [192, 4, 8, 16, 8]
         and total == EXPECTED_INTERNAL_ELEMENTS,
         f"incomplete internal knownness census: {known}, total={total}")
    need(all(0 <= value <= bound for value, bound in
             zip(ones, [192, 4, 8, 16, 8])), "internal one-count bounds")

    begin = one_match(
        r"^M2142_RTL_SAIF_WINDOW_BEGIN sampling=settled_negedge .*"
        r"preload_cycles=383 time_ns=([0-9.]+) "
        r"next_ucli_action=power_reset$", text, "window begin")
    end = one_match(
        r"^M2142_RTL_SAIF_WINDOW_END axis=ordinary_lru4 "
        r"sampling=settled_negedge measurement_cycles=([0-9]+) "
        r"scalar_weight_reads=([0-9]+) duration_ns=([0-9.]+)$",
        text, "window end")
    passed = one_match(
        r"^PASS_M2142_ORDINARY_LATE_ENABLE_SAIF_PREFLIGHT "
        r"ledger_exact=1 internal_census_exact=1 "
        r"enable_before_reset_preload=1 power_reset_at_first_stop=1 "
        r"initreg_diagnostic_only=1 paper_citable=0$", text, "pass")

    need(phase_matches[1].start() < census.start() < begin.start()
         < phase_matches[2].start() < phase_matches[3].start()
         < end.start() < passed.start() < phase_matches[4].start(),
         "causal marker order: enable/run/census/stop/reset/window/stop")
    cycles, reads, duration = (int(end.group(1)), int(end.group(2)),
                               float(end.group(3)))
    need(cycles == EXPECTED_CYCLES and reads == EXPECTED_READS,
         "frozen ordinary ledger drift")
    need(math.isclose(duration, EXPECTED_DURATION_NS,
                      rel_tol=0.0, abs_tol=1e-6), "runtime duration drift")
    need(not re.search(
        r"(^|\n)(Fatal:|Error:)|Assertion failed|M2142 .*drift|"
        r"M2142 .*failed", text), "runtime failure token")
    return {
        "sha256": sha256(path),
        "axis": "ordinary_lru4",
        "begin_time_ns": float(begin.group(1)),
        "measurement_cycles": cycles,
        "scalar_weight_reads": reads,
        "duration_ns": duration,
        "ucli_phase_order_exact": True,
        "power_enable_before_first_run": True,
        "power_reset_after_first_stop": True,
        "internal_knownness": {
            "row_live_q": known[0], "cache_valid_q": known[1],
            "slot_valid_q": known[2], "bridge_overflow": known[3],
            "rsp_shape_legal": known[4], "total": total,
            "one_counts": ones, "observe_only": True,
        },
        "completion_ledger_exact": True,
    }


def parse_saif(path: Path) -> dict[str, object]:
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
    need(math.isclose(duration_ns, EXPECTED_DURATION_NS,
                      rel_tol=0.0, abs_tol=1e-6),
         f"duration {duration_ns} != {EXPECTED_DURATION_NS}")

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
        "sha256": sha256(path),
        "axis": "ordinary_lru4",
        "duration_raw": duration_raw,
        "duration_ns": duration_ns,
        "expected_cycles": EXPECTED_CYCLES,
        "record_count": len(records),
        "nonzero_toggle_record_count": toggled,
        "tx_nonzero_record_count": tx_nonzero,
        "tx_sum": tx_sum,
        "conservation_failures": conservation_failures,
        "critical_nonzero_record_counts": critical,
    }


def final_result(root: Path, output: Path) -> dict[str, object]:
    result = {
        "schema": "m2142_m2018_tsbg_ordinary_late_enable_saif_preflight_result_r1_v1",
        "status": "PASS_RAW_M2144_M2142_ORDINARY_LATE_ENABLE_SAIF_PREFLIGHT_PENDING_M2145_RESULT_HAMMER",
        "runtime": parse_runtime(root / "rtl_sim.log"),
        "rtl_saif": parse_saif(root / "rtl_execute.saif"),
        "claim_boundary": {
            "ordinary_axis_only": True,
            "vcs_native_rtl_saif_acquisition_preflight": True,
            "late_enable_causal_hypothesis_test": True,
            "power_enable_before_reset_preload": True,
            "power_reset_at_first_stop": True,
            "internal_census_observe_only": True,
            "tsbg_axis_run": False,
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
        "ordinary_only": "tsbg_axis_run\": False" in source,
        "frozen_cycles": "EXPECTED_CYCLES = 20292" in source,
        "frozen_reads": "EXPECTED_READS = 14304" in source,
        "exact_duration": "EXPECTED_DURATION_NS = 60876.0" in source,
        "exact_record_gate": "len(records) == EXPECTED_RECORDS" in source,
        "all_tx_zero_gate": "tx_nonzero == 0 and tx_sum == 0.0" in source,
        "exact_conservation_gate": "conservation_failures == 0" in source,
        "critical_nonzero_gate": "missing/zero critical activity" in source,
        "census_gate": "known == [192, 4, 8, 16, 8]" in source,
        "causal_order_gate": "causal marker order" in source,
        "diagnostic_boundary": '"paper_citable": False' in source,
    }
    need(all(checks.values()), f"static checks failed: {checks}")
    return {"status": "PASS_M2142_STATIC_PARSER", "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("static")
    runtime = sub.add_parser("runtime")
    runtime.add_argument("--path", required=True, type=Path)
    saif = sub.add_parser("saif")
    saif.add_argument("--path", required=True, type=Path)
    final = sub.add_parser("final")
    final.add_argument("--root", required=True, type=Path)
    final.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "static":
            value = static_check()
        elif args.command == "runtime":
            value = parse_runtime(args.path)
        elif args.command == "saif":
            value = parse_saif(args.path)
        else:
            value = final_result(args.root, args.output)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    except Failure as exc:
        print(f"M2142_FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
