#!/opt/anaconda3/bin/python3
"""M2176 minimal reset-semantic repair over immutable M2172 SAIF parsing."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re


HW = Path(__file__).resolve().parents[2]
BASE_PATH = HW / "system_simulator/scripts/parse_m2172_m2018_ordinary_native_saif_balanced_scope_preflight.py"


def load_base():
    spec = importlib.util.spec_from_file_location("m2172_parser_frozen_for_m2176", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("M2172 parser import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base()
Failure = BASE.Failure
need = BASE.need
sha256 = BASE.sha256
read = BASE.read
write_json = BASE.write_json
EXPECTED = BASE.EXPECTED
TARGET_INSTANCE = BASE.TARGET_INSTANCE
CRITICAL = BASE.CRITICAL
normalize_semantics = BASE.normalize_semantics
audit_single_axis_source = BASE.audit_single_axis_source
parse_balanced_saif = BASE.parse_balanced_saif
parse_saif = BASE.parse_saif


def reset_failure_lines(text: str) -> list[str]:
    """Reject a negative reset/clear operation without requiring a second noun."""
    bad: list[str] = []
    operation = re.compile(r"\b(?:reset(?:ting)?|clear(?:ed|ing)?)\b")
    negative = re.compile(
        r"\b(ignore(?:d|s|ing)?|reject(?:ed|s|ing)?|deny|denied|unsupported|"
        r"fail(?:ed|s|ure)?|cannot|unable|uncleared|retained|remain(?:s|ed|ing)?|"
        r"not\s+(?:be\s+)?(?:clear(?:ed)?|reset|done|supported)|"
        r"did\s+not\s+(?:clear|reset)|was\s+not\s+(?:clear(?:ed)?|reset))\b")
    for raw in text.splitlines():
        line = normalize_semantics(raw)
        if not line:
            continue
        report_before_reset = bool(re.search(r"\bsaif\s+report\s+before\s+reset\b", line))
        if report_before_reset or (operation.search(line) and negative.search(line)):
            bad.append(raw.strip())
    return bad


def parse_runtime(path: Path) -> dict[str, object]:
    text = read(path)
    failures = reset_failure_lines(text)
    need(not failures, f"semantic power-reset rejection: {failures}")
    try:
        result = BASE.BASE.parse_runtime(path)
    except BASE.BASE.Failure as exc:
        raise Failure(str(exc)) from exc
    result["sha256"] = sha256(path)
    result["power_reset_rejection_warning_count"] = 0
    result["power_reset_acceptance_runtime_evidence"] = (
        "minimal_reset_or_clear_failure_absent_and_tb_duration_exact__"
        "final_requires_balanced_dut_scoped_saif_duration")
    return result


def final_result(root: Path, output: Path) -> dict[str, object]:
    runtime = parse_runtime(root / "rtl_sim.log")
    diagnostic = parse_saif(root / "rtl_prehistory.saif", role="diagnostic_prehistory")
    measurement = parse_saif(root / "rtl_measurement.saif", role="measurement")
    need(diagnostic["identity_seal"]["sha256"] != measurement["identity_seal"]["sha256"],
         "diagnostic and measurement SAIF content identities collide")
    result = {
        "schema": "m2178_m2176_m2018_ordinary_native_saif_reset_semantics_preflight_result_r1_v1",
        "status": "PASS_RAW_M2178_M2176_RESET_SEMANTICS_NATIVE_SAIF_PREFLIGHT_PENDING_M2179_RESULT_HAMMER",
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
    failures = (
        "Warning: reset failed.", "Error: reset denied.",
        "Warning: reset ignored.", "Warning: clear failed.",
        "Error: reset not reset.",
    )
    success = "Info: power reset request accepted and switching counters cleared."
    checks = {
        "minimal_failure_forms_rejected": all(reset_failure_lines(line) for line in failures),
        "accepted_control_not_rejected": not reset_failure_lines(success),
        "balanced_saif_parser_is_exact_m2172_function": parse_saif is BASE.parse_saif,
        "target_instance_exact": TARGET_INSTANCE == "dut_ordinary",
        "exact_record_gate": EXPECTED["records"] == 93971,
    }
    need(all(checks.values()), f"static checks failed: {checks}")
    return {"status": "PASS_M2176_STATIC_PARSER", "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("static")
    runtime = sub.add_parser("runtime")
    runtime.add_argument("--path", type=Path, required=True)
    saif = sub.add_parser("saif")
    saif.add_argument("--path", type=Path, required=True)
    saif.add_argument("--role", choices=("diagnostic_prehistory", "measurement"), required=True)
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
        print(f"M2176_PARSE_FAIL_CLOSED: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
