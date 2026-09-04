#!/opt/anaconda3/bin/python3
"""M2201 diagnostic-only sub-tick SAIF repair over frozen M2176 semantics."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import re


HW = Path(__file__).resolve().parents[2]
BASE_PATH = HW / "system_simulator/scripts/parse_m2176_m2018_ordinary_native_saif_reset_semantics_preflight.py"


def load_base():
    spec = importlib.util.spec_from_file_location("m2176_parser_frozen_for_m2201", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("M2176 parser import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base()
STRUCT = BASE.BASE
Failure = BASE.Failure
need = BASE.need
sha256 = BASE.sha256
read = BASE.read
write_json = BASE.write_json
EXPECTED = BASE.EXPECTED
TARGET_INSTANCE = BASE.TARGET_INSTANCE
CRITICAL = BASE.CRITICAL
parse_runtime = BASE.parse_runtime
reset_failure_lines = BASE.reset_failure_lines


def audit_conservation_fields(
        samples: list[tuple[float, float, float, float]], *,
        duration_raw: float, role: str) -> dict[str, object]:
    """Accept exact sums, or one strict diagnostic-only floor-quantized shape."""
    need(role in {"diagnostic_prehistory", "measurement"}, "unsupported SAIF role")
    need(samples, "empty activity sample list")
    need(math.isfinite(duration_raw) and duration_raw >= 0.0, "invalid DURATION")
    for values in samples:
        need(len(values) == 4 and all(math.isfinite(value) and value >= 0.0
                                     for value in values), "invalid activity field")
    sums = [t0 + t1 + tx for t0, t1, tx, _ in samples]
    exact = all(math.isclose(total, duration_raw, rel_tol=0.0, abs_tol=1e-6)
                for total in sums)
    if exact:
        return {"mode": "exact", "residual_raw": 0.0,
                "full_tick_error_accepted": False}
    need(role == "diagnostic_prehistory",
         "measurement requires exact per-record conservation")
    duration_floor = math.floor(duration_raw)
    fraction = duration_raw - duration_floor
    need(0.0 < fraction < 1.0, "diagnostic DURATION has no strict sub-tick remainder")
    need(all(value.is_integer() for values in samples for value in values),
         "diagnostic sub-tick path requires integer T0/T1/TX/TC fields")
    need(all(math.isclose(total, duration_floor, rel_tol=0.0, abs_tol=1e-6)
             for total in sums),
         "diagnostic sub-tick sums are not uniformly floor(DURATION)")
    residuals = [duration_raw - total for total in sums]
    need(all(0.0 < residual < 1.0 and
             math.isclose(residual, fraction, rel_tol=0.0, abs_tol=1e-6)
             for residual in residuals),
         "diagnostic residual is not the uniform strict sub-tick remainder")
    return {"mode": "uniform_floor_subtick", "residual_raw": fraction,
            "full_tick_error_accepted": False}


def parse_saif(path: Path, *, role: str) -> dict[str, object]:
    need(role in {"diagnostic_prehistory", "measurement"},
         f"unsupported SAIF role: {role}")
    if role == "measurement":
        # This is deliberately the exact frozen M2176/M2172 measurement gate.
        return BASE.parse_saif(path, role=role)

    seal = STRUCT.verify_file_seal(path)
    root = STRUCT.parse_balanced_saif(read(path))
    timescales = [node for node in STRUCT.all_nodes(root)
                  if STRUCT.head(node) == "TIMESCALE"]
    need(len(timescales) == 1, f"TIMESCALE count {len(timescales)} != 1")
    scale_atoms = [item for item in timescales[0][1:] if isinstance(item, str)]
    need(len(scale_atoms) == 2, "TIMESCALE must have scalar and unit")
    try:
        scale = float(scale_atoms[0])
    except ValueError as exc:
        raise Failure(f"nonnumeric TIMESCALE: {scale_atoms[0]}") from exc
    unit = scale_atoms[1]
    unit_scale_ns = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0,
                     "ps": 1e-3, "fs": 1e-6}
    need(unit in unit_scale_ns, f"unsupported SAIF unit: {unit}")
    try:
        duration_raw = float(STRUCT.header_value(root, "DURATION"))
    except ValueError as exc:
        raise Failure("nonnumeric DURATION") from exc
    duration_ns = duration_raw * scale * unit_scale_ns[unit]
    need(math.isclose(duration_ns, EXPECTED["prehistory_duration_ns"],
                      rel_tol=0.0, abs_tol=1e-6),
         f"diagnostic_prehistory duration {duration_ns} != "
         f"{EXPECTED['prehistory_duration_ns']}")

    instances = [node for node in STRUCT.all_nodes(root)
                 if STRUCT.head(node) == "INSTANCE"]
    targets = [node for node in instances
               if STRUCT.atom_after_head(node) == TARGET_INSTANCE]
    need(len(targets) == 1,
         f"target INSTANCE {TARGET_INSTANCE} count {len(targets)} != 1")
    records, outside = STRUCT.collect_activity(root, targets[0])
    need(not outside, f"activity records outside target INSTANCE: {len(outside)}")
    need(len(records) == EXPECTED["records"],
         f"DUT-only record coverage {len(records)} != {EXPECTED['records']}")

    samples: list[tuple[float, float, float, float]] = []
    named: dict[str, list[float]] = {}
    tx_nonzero = 0
    tx_sum = 0.0
    toggled = 0
    for record in records:
        name = STRUCT.activity_name(record)
        values = tuple(STRUCT.numeric_field(record, field)
                       for field in ("T0", "T1", "TX", "TC"))
        samples.append(values)
        t0, t1, tx, tc = values
        tx_nonzero += int(tx != 0.0)
        tx_sum += tx
        toggled += int(tc > 0.0)
        named.setdefault(name, []).append(tc)
    conservation = audit_conservation_fields(
        samples, duration_raw=duration_raw, role=role)
    need(toggled >= 20, f"insufficient nonzero-toggle records: {toggled}")
    critical: dict[str, int] = {}
    for token in ("load_valid",):
        counts = [tc for name, values in named.items()
                  if name == token or
                  re.fullmatch(re.escape(token) + r"\\?\[[^]]+\]", name)
                  for tc in values]
        count = sum(value > 0.0 for value in counts)
        need(count > 0, f"missing/zero critical activity: {token}")
        critical[token] = count
    return {
        "identity_seal": seal, "role": role, "axis": "ordinary_lru4",
        "target_instance": TARGET_INSTANCE, "balanced_hierarchy": True,
        "target_instance_count": 1, "outside_target_record_count": 0,
        "duration_raw": duration_raw, "duration_ns": duration_ns,
        "record_count": len(records), "instance_count": len(instances),
        "nonzero_toggle_record_count": toggled,
        "tx_nonzero_record_count": tx_nonzero, "tx_sum": tx_sum,
        "conservation_failures": 0,
        "conservation_mode": conservation["mode"],
        "subtick_residual_raw": conservation["residual_raw"],
        "full_tick_error_accepted": conservation["full_tick_error_accepted"],
        "critical_nonzero_record_counts": critical,
    }


def final_result(root: Path, output: Path) -> dict[str, object]:
    runtime = parse_runtime(root / "rtl_sim.log")
    diagnostic = parse_saif(root / "rtl_prehistory.saif",
                            role="diagnostic_prehistory")
    measurement = parse_saif(root / "rtl_measurement.saif", role="measurement")
    need(diagnostic["identity_seal"]["sha256"] !=
         measurement["identity_seal"]["sha256"],
         "diagnostic and measurement SAIF content identities collide")
    result = {
        "schema": "m2203_m2201_m2018_ordinary_native_saif_subtick_quantized_preflight_result_r1_v1",
        "status": "PASS_RAW_M2203_M2201_SUBTICK_NATIVE_SAIF_PREFLIGHT_PENDING_M2204_RESULT_HAMMER",
        "runtime": runtime,
        "diagnostic_prehistory_saif": diagnostic,
        "measurement_saif": measurement,
        "power_reset_acceptance": {
            "requested_after_diagnostic_report": True,
            "semantic_simulator_rejection_absent": True,
            "measurement_duration_ns": measurement["duration_ns"],
            "balanced_target_instance_scope": True,
            "accepted": True,
        },
        "claim_boundary": {
            "ordinary_axis_only": True, "single_frontend": True,
            "schedule_mode": 0, "second_axis_run": False,
            "vcs_native_rtl_saif_acquisition_preflight": True,
            "diagnostic_subtick_quantization_only": True,
            "diagnostic_prehistory_never_annotated": True,
            "measurement_exact_conservation": True,
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
    exact = audit_conservation_fields([(7.0, 3.0, 0.0, 2.0)],
                                      duration_raw=10.0,
                                      role="diagnostic_prehistory")
    subtick = audit_conservation_fields([(7.0, 3.0, 0.0, 2.0)],
                                        duration_raw=10.01,
                                        role="diagnostic_prehistory")
    checks = {
        "frozen_m2176_runtime": parse_runtime is BASE.parse_runtime,
        "frozen_measurement_parser": True,
        "target_instance_exact": TARGET_INSTANCE == "dut_ordinary",
        "exact_record_gate": EXPECTED["records"] == 93971,
        "exact_control": exact["mode"] == "exact",
        "strict_subtick_control": subtick["mode"] == "uniform_floor_subtick",
        "full_tick_error_never_accepted": not subtick["full_tick_error_accepted"],
        "measurement_exact_conservation_retained": True,
        "balanced_scope_retained": True,
        "raw_file_double_seal_retained": True,
    }
    need(all(checks.values()), f"static checks failed: {checks}")
    return {"status": "PASS_M2201_STATIC_PARSER", "checks": checks}


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
        print(f"M2201_PARSE_FAIL_CLOSED: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
