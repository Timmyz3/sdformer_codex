#!/opt/anaconda3/bin/python3
"""Fail-closed parser for M2160's report-before-reset native-SAIF preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys


EXPECTED = {
    "cycles": 20292,
    "duration_ns": 60876.0,
    "rows": 149,
    "issues": 1278,
    "products": 29472,
    "commits": 24,
    "bundles": 1788,
    "reads": 14304,
    "records": 93971,
    "internal_elements": 228,
    "prehistory_duration_ns": 1167.01,
}
CRITICAL = (
    "mem_req_valid", "mem_rsp_valid", "bridge_valid", "commit_valid",
    "mem_req_accept", "mem_rsp_accept", "bridge_accept", "commit_accept",
)
DIRECT_FRONTEND = "m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend"
FORBIDDEN_SECOND_AXIS_PATTERNS = (
    r"\btb_m2051\b", r"\bdut_tsbg\b", r"\bload_valid_tsbg\b",
    r"\btsbg_done_cycle\b", r"\bfull_tsbg", r"\bterminal_tsbg\b",
    r"\bexpected_tsbg\b", r"\bobserved_tsbg\b", r"\btsbg\s*\.",
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


def verify_file_seal(path: Path) -> dict[str, str]:
    """Verify a raw file's two-level identity seal before parsing it."""
    need(path.is_file() and not path.is_symlink(), f"raw file: {path}")
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    need(sidecar.is_file() and not sidecar.is_symlink(),
         f"missing/symlink file seal: {sidecar}")
    need(outer.is_file() and not outer.is_symlink(),
         f"missing/symlink outer file seal: {outer}")
    need(sidecar.read_text().split() == [sha256(path), path.name],
         f"raw file sidecar mismatch: {path.name}")
    need(outer.read_text().split() == [sha256(sidecar), sidecar.name],
         f"raw file outer seal mismatch: {path.name}")
    return {
        "sha256": sha256(path),
        "sidecar_sha256": sha256(sidecar),
        "outer_sha256": sha256(outer),
    }


def one_match(pattern: str, text: str, label: str) -> re.Match[str]:
    rows = list(re.finditer(pattern, text, re.MULTILINE))
    need(len(rows) == 1, f"{label} marker count {len(rows)} != 1")
    return rows[0]


def audit_single_axis_source(tb_text: str, filelist_text: str) -> dict[str, object]:
    """Reject any second executable axis while allowing M2018's historic name."""
    instantiations = re.findall(
        rf"\b{re.escape(DIRECT_FRONTEND)}\s*#\s*\(", tb_text)
    need(len(instantiations) == 1,
         f"direct M2018 frontend instance count {len(instantiations)} != 1")
    mode_zero = re.findall(r"\.SCHEDULE_MODE\s*\(\s*0\s*\)", tb_text)
    mode_one = re.findall(r"\.SCHEDULE_MODE\s*\(\s*1\s*\)", tb_text)
    need(len(mode_zero) == 1 and not mode_one,
         f"schedule-mode topology zero={len(mode_zero)} one={len(mode_one)}")
    hits: list[str] = []
    for pattern in FORBIDDEN_SECOND_AXIS_PATTERNS:
        if re.search(pattern, tb_text, flags=re.IGNORECASE | re.MULTILINE):
            hits.append(pattern)
    need(not hits, f"second-axis TB symbols: {hits}")
    lowered_filelist = filelist_text.lower()
    need("tb_m2051" not in lowered_filelist,
         "parent dual-axis testbench in filelist")
    need("m2020_m2018_vcs_public_name_adapter" not in lowered_filelist,
         "public-name adapter in filelist")
    source_rows = [line.strip() for line in filelist_text.splitlines()
                   if line.strip() and not line.lstrip().startswith("#")]
    need(len(source_rows) == 4, f"filelist source count {len(source_rows)} != 4")
    need(source_rows[-1].endswith(
         "tb_m2160_m2018_ordinary_native_saif_report_reset_preflight.sv"),
         "single-axis TB is not final filelist source")
    return {
        "direct_m2018_frontends": 1,
        "schedule_mode_zero_instances": 1,
        "schedule_mode_one_instances": 0,
        "parent_dual_axis_tb_instances": 0,
        "second_axis_symbols": 0,
        "public_name_adapter_in_filelist": False,
        "filelist_source_count": len(source_rows),
    }


def parse_runtime(path: Path) -> dict[str, object]:
    text = read(path)
    actions = {
        1: "power_enable",
        2: "first_run_returned",
        3: "prehistory_power_disable",
        4: "prehistory_power_report",
        5: "power_reset_requested",
        6: "measurement_power_enable",
        7: "second_run_returned",
        8: "measurement_power_disable",
        9: "measurement_power_report",
    }
    phases = [one_match(
        rf"^M2160_UCLI_PHASE order={order} action={action}(?: .*)?$",
        text, f"UCLI phase {order}") for order, action in actions.items()]
    need(all(a.start() < b.start() for a, b in zip(phases, phases[1:])),
         "UCLI phase order drift")

    census = one_match(
        r"^M2160_INTERNAL_KNOWNNESS_CENSUS phase=pre_power_reset "
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
         and total == EXPECTED["internal_elements"],
         f"incomplete internal knownness: {known}, total={total}")
    need(all(0 <= value <= bound for value, bound in
             zip(ones, [192, 4, 8, 16, 8])), "internal one-count bounds")

    begin = one_match(
        r"^M2160_RTL_SAIF_WINDOW_BEGIN sampling=settled_negedge "
        r"global_slot=42 sample=0 layer=28 is_fc2=0 token_start=0 "
        r"source_groups=48 preload_cycles=383 time_ns=([0-9.]+) "
        r"next_ucli_action=disable_report_prehistory_then_reset$", text,
        "window begin")
    end = one_match(
        r"^M2160_RTL_SAIF_WINDOW_END axis=ordinary_lru4 "
        r"sampling=settled_negedge measurement_cycles=([0-9]+) "
        r"rows=([0-9]+) issues=([0-9]+) products=([0-9]+) "
        r"commits=([0-9]+) bundles=([0-9]+) "
        r"scalar_weight_reads=([0-9]+) duration_ns=([0-9.]+)$",
        text, "window end")
    passed = one_match(
        r"^PASS_M2160_ORDINARY_SINGLE_AXIS_NATIVE_SAIF_PREFLIGHT "
        r"ledger_exact=1 arithmetic_scoreboard_exact=1 "
        r"internal_census_exact=1 enable_before_reset_preload=1 "
        r"prehistory_report_requested=1 power_reset_requested=1 "
        r"frontends=1 schedule_mode=0 "
        r"second_axis=0 initreg_diagnostic_only=1 paper_citable=0$",
        text, "pass")
    # UCLI phase 1 precedes the first run.  The census and begin markers are
    # emitted *inside* that run and therefore precede its return marker.  The
    # report/reset/re-enable markers follow.  Likewise end/pass are emitted
    # inside the second run and precede its return marker.
    need(phases[0].start() < census.start() < begin.start()
         < phases[1].start() < phases[2].start() < phases[3].start()
         < phases[4].start() < phases[5].start()
         < end.start() < passed.start() < phases[6].start()
         < phases[7].start() < phases[8].start(), "causal marker order")
    ledgers = [int(end.group(index)) for index in range(1, 8)]
    need(ledgers == [EXPECTED[key] for key in
         ("cycles", "rows", "issues", "products", "commits", "bundles", "reads")],
         f"frozen ordinary ledger drift: {ledgers}")
    duration = float(end.group(8))
    need(math.isclose(duration, EXPECTED["duration_ns"],
                      rel_tol=0.0, abs_tol=1e-6), "runtime duration drift")
    reset_rejection_patterns = (
        r"SAIF_REPORT_BEFORE_RESET",
        r"request to reset power information will be ignored",
        r"power\s+-reset.*ignored",
        r"resetting switching activity.*ignored",
    )
    reset_rejections = [pattern for pattern in reset_rejection_patterns
                        if re.search(pattern, text, re.IGNORECASE)]
    need(not reset_rejections,
         f"simulator rejected/ignored power reset: {reset_rejections}")
    need(not re.search(
        r"(^|\n)(Fatal:|Error:)|Assertion failed|M2160 .*drift|"
        r"M2160 .*failed|M2160 .*mismatch|M2160 .*timeout", text),
        "runtime failure token")
    return {
        "sha256": sha256(path),
        "axis": "ordinary_lru4",
        "single_frontend": True,
        "schedule_mode": 0,
        "second_axis_executed": False,
        "begin_time_ns": float(begin.group(1)),
        "duration_ns": duration,
        "completion_ledger": dict(zip(
            ("cycles", "rows", "issues", "products", "commits", "bundles", "reads"),
            ledgers)),
        "internal_knownness": {
            "row_live_q": known[0], "cache_valid_q": known[1],
            "slot_valid_q": known[2], "bridge_overflow": known[3],
            "rsp_shape_legal": known[4], "total": total,
            "one_counts": ones, "observe_only": True,
        },
        "arithmetic_scoreboard_exact": True,
        "power_enable_before_first_run": True,
        "prehistory_report_before_reset_request": True,
        "power_reset_requested_after_prehistory_report": True,
        "power_reset_rejection_warning_count": 0,
        "power_reset_acceptance_runtime_evidence":
            "warning_absent_and_tb_duration_exact__final_requires_saif_duration",
    }


def parse_saif(path: Path, *, role: str) -> dict[str, object]:
    need(role in {"diagnostic_prehistory", "measurement"},
         f"unsupported SAIF role: {role}")
    seal = verify_file_seal(path)
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
    expected_duration = (EXPECTED["duration_ns"] if role == "measurement"
                         else EXPECTED["prehistory_duration_ns"])
    need(math.isclose(duration_ns, expected_duration,
                      rel_tol=0.0, abs_tol=1e-6),
         f"{role} duration {duration_ns} != {expected_duration}")
    instances = re.findall(r"\(INSTANCE(?:\s|\n)", text)
    need(instances, f"{role} has no INSTANCE block")
    records = re.findall(
        r"\(T0\s+([0-9.eE+-]+)\)\s*\(T1\s+([0-9.eE+-]+)\)\s*"
        r"\(TX\s+([0-9.eE+-]+)\)\s*\(TC\s+([0-9.eE+-]+)\)", text)
    need(len(records) == EXPECTED["records"],
         f"DUT-only record coverage {len(records)} != {EXPECTED['records']}")
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
    if role == "measurement":
        need(tx_nonzero == 0 and tx_sum == 0.0,
             f"SAIF unknown activity: records={tx_nonzero} sum={tx_sum}")
    need(conservation_failures == 0,
         f"SAIF conservation failures: {conservation_failures}")
    need(toggled >= 20, f"insufficient nonzero-toggle records: {toggled}")
    critical: dict[str, int] = {}
    for token in (CRITICAL if role == "measurement" else ("load_valid",)):
        values = re.findall(
            rf"\({re.escape(token)}(?:\\?\[[^\]]+\])?\s+"
            rf"\(T0\s+[0-9.eE+-]+\)\s*\(T1\s+[0-9.eE+-]+\)\s*"
            rf"\(TX\s+[0-9.eE+-]+\)\s*\(TC\s+([0-9.eE+-]+)\)", text)
        count = sum(float(value) > 0.0 for value in values)
        need(count > 0, f"missing/zero critical activity: {token}")
        critical[token] = count
    return {
        "identity_seal": seal, "role": role, "axis": "ordinary_lru4",
        "duration_raw": duration_raw, "duration_ns": duration_ns,
        "record_count": len(records),
        "nonzero_toggle_record_count": toggled,
        "tx_nonzero_record_count": tx_nonzero, "tx_sum": tx_sum,
        "conservation_failures": conservation_failures,
        "critical_nonzero_record_counts": critical,
    }


def final_result(root: Path, output: Path) -> dict[str, object]:
    runtime = parse_runtime(root / "rtl_sim.log")
    diagnostic = parse_saif(root / "rtl_prehistory.saif",
                            role="diagnostic_prehistory")
    measurement = parse_saif(root / "rtl_measurement.saif",
                             role="measurement")
    need(diagnostic["identity_seal"]["sha256"] !=
         measurement["identity_seal"]["sha256"],
         "diagnostic and measurement SAIF content identities collide")
    result = {
        "schema": "m2162_m2160_m2018_ordinary_native_saif_report_reset_preflight_result_r1_v1",
        "status": "PASS_RAW_M2162_M2160_REPORT_RESET_NATIVE_SAIF_PREFLIGHT_PENDING_M2163_RESULT_HAMMER",
        "runtime": runtime,
        "diagnostic_prehistory_saif": diagnostic,
        "measurement_saif": measurement,
        "power_reset_acceptance": {
            "requested_after_diagnostic_report": True,
            "simulator_rejection_warning_absent": True,
            "measurement_duration_ns": measurement["duration_ns"],
            "accepted": True,
        },
        "claim_boundary": {
            "ordinary_axis_only": True,
            "single_frontend": True,
            "schedule_mode": 0,
            "second_axis_run": False,
            "vcs_native_rtl_saif_acquisition_preflight": True,
            "diagnostic_prehistory_never_annotated": True,
            "measurement_saif_candidate_only": True,
            "dc_run": False, "ptpx_run": False, "icc2_run": False,
            "mapped_netlist_activity": False, "power_or_energy": False,
            "component_speedup_admitted": False, "system_speedup": False,
            "paper_citable": False,
        },
    }
    write_json(output, result)
    return result


def static_check() -> dict[str, object]:
    source = Path(__file__).read_text()
    checks = {
        "frozen_cycles": '"cycles": 20292' in source,
        "frozen_reads": '"reads": 14304' in source,
        "exact_duration": '"duration_ns": 60876.0' in source,
        "exact_record_gate": 'len(records) == EXPECTED["records"]' in source,
        "all_tx_zero_gate": "tx_nonzero == 0 and tx_sum == 0.0" in source,
        "conservation_gate": "conservation_failures == 0" in source,
        "critical_toggle_gate": "missing/zero critical activity" in source,
        "census_gate": "known == [192, 4, 8, 16, 8]" in source,
        "single_axis_boundary": '"second_axis_run": False' in source,
        "diagnostic_boundary": '"paper_citable": False' in source,
        "raw_file_double_seal_gate": "verify_file_seal(path)" in source,
        "reset_warning_gate": "SAIF_REPORT_BEFORE_RESET" in source,
        "two_distinct_saif_roles":
            'role in {"diagnostic_prehistory", "measurement"}' in source,
    }
    need(all(checks.values()), f"static checks failed: {checks}")
    return {"status": "PASS_M2160_STATIC_PARSER", "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("static")
    runtime = sub.add_parser("runtime")
    runtime.add_argument("--path", type=Path, required=True)
    saif = sub.add_parser("saif")
    saif.add_argument("--path", type=Path, required=True)
    saif.add_argument("--role", choices=("diagnostic_prehistory", "measurement"),
                      required=True)
    final = sub.add_parser("final")
    final.add_argument("--root", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "static":
        value = static_check()
    elif args.command == "runtime":
        value = parse_runtime(args.path)
    elif args.command == "saif":
        value = parse_saif(args.path, role=args.role)
    else:
        value = final_result(args.root, args.output)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Failure as exc:
        print(f"M2160_PARSE_FAIL_CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
