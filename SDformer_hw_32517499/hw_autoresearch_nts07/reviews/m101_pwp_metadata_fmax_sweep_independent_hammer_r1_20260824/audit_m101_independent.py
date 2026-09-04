#!/usr/bin/env python3
"""Independent, read-only M101 evidence audit plus production-auditor attacks."""

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PERIODS = (2.750, 3.000, 3.250, 3.500, 3.750, 4.000, 4.250, 4.500)
DESIGNS = {
    "m85": {
        "top": "guarded_wordpacked_pwp_stream",
        "filelist": "dc_handoff/filelists/date_m97_m85_logic_only_dc.f",
        "filelist_sha256": "6e2c6c7f831eecadba604675447f8425c3427e6cf83a6c6310e7a20483789d00",
    },
    "m99": {
        "top": "phase_slack_guarded_wordpacked_pwp_stream",
        "filelist": "dc_handoff/filelists/date_m100_m99_phase_slack_logic_only_dc.f",
        "filelist_sha256": "13c92bdef276680174c564ea5f45e360bbd45e7cd6a38513ca3b247a96b629c0",
    },
}
RUN = ROOT / "dc_handoff/runs/m101_pwp_metadata_fmax_sweep_r1_20260824"
CONTRACT = ROOT / "contracts/m101_pwp_metadata_fmax_sweep_synopsys_contract_r1_20260824.json"
PROD_AUDITOR = ROOT / "dc_handoff/scripts/audit_m101_pwp_metadata_fmax_sweep.py"
PROD_RECEIPT = ROOT / "results/m101_pwp_metadata_fmax_sweep_synopsys_r1_20260824/m101_pwp_metadata_fmax_sweep_receipt_r1.json"
DOC359 = ROOT / "docs/359_DATE终局冻结_20260813.md"

REQUIRED = (
    "dc.log",
    "dc_backend.rc",
    "BACKEND_COMPLETE.txt",
    "point_identity.txt",
    "reports/qor.rpt",
    "reports/area.rpt",
    "reports/clocks.rpt",
    "reports/timing_setup.rpt",
    "reports/timing_hold.rpt",
    "reports/constraint_violators.rpt",
    "reports/check_design_postcompile.rpt",
    "reports/check_timing_postcompile.rpt",
    "reports/references_postcompile.rpt",
    "reports/resources_precompile.rpt",
    "reports/resources_postcompile.rpt",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def qos_field(text: str, label: str, number=float):
    match = re.search(rf"^\s*{re.escape(label)}:\s+(-?[0-9]+(?:\.[0-9]+)?)\s*$", text, re.M)
    require(match is not None, f"missing QoR field {label}")
    return number(match.group(1))


def worst_slack(path: Path):
    values = [
        (float(value), status)
        for status, value in re.findall(
            r"slack\s+\((MET|VIOLATED)\)\s+(-?[0-9]+(?:\.[0-9]+)?)",
            path.read_text(encoding="utf-8", errors="replace"),
        )
    ]
    require(values, f"no slack in {path}")
    return min(values, key=lambda pair: pair[0])


def point_name(design: str, period: float) -> str:
    return f"{design}_{period:.3f}ns".replace(".", "p").replace("pns", "ns")


def audit_point(design: str, period: float) -> dict:
    cfg = DESIGNS[design]
    point = RUN / point_name(design, period)
    require(point.is_dir(), f"missing {point}")
    for rel in REQUIRED:
        require((point / rel).is_file(), f"missing {point / rel}")
    for rel in (
        f"netlist/{cfg['top']}_mapped.v",
        f"netlist/{cfg['top']}_mapped.sdc",
        f"netlist/{cfg['top']}.ddc",
    ):
        require((point / rel).is_file(), f"missing completion netlist {point / rel}")

    identity = (point / "point_identity.txt").read_text().splitlines()
    require(identity[0] == f"design_key={design}", f"design key drift at {point}")
    require(identity[1] == f"design_name={cfg['top']}", f"top drift at {point}")
    require(identity[2] == f"clock_period_ns={period:.3f}", f"period drift at {point}")
    require(identity[3].startswith(cfg["filelist_sha256"] + "  "), f"filelist SHA drift at {point}")

    log = (point / "dc.log").read_text(encoding="utf-8", errors="replace")
    require(not re.search(r"^Error:", log, re.M), f"DC error at {point}")
    require(f"Current design is now '{cfg['top']}'." in log, f"log top mismatch at {point}")
    require("Using operating conditions 'ssg0p9v125c'" in log, f"corner mismatch at {point}")
    require((point / "dc_backend.rc").read_text().strip() == "0", f"nonzero backend at {point}")
    require((point / "BACKEND_COMPLETE.txt").read_text().strip() == "backend_complete=true", f"incomplete at {point}")

    clocks = (point / "reports/clocks.rpt").read_text()
    match = re.search(r"^core_clk\s+([0-9]+(?:\.[0-9]+)?)\s+", clocks, re.M)
    require(match is not None and abs(float(match.group(1)) - period) < 1e-9, f"clock report mismatch at {point}")
    for report in ("qor.rpt", "timing_setup.rpt", "timing_hold.rpt", "references_postcompile.rpt"):
        text = (point / "reports" / report).read_text(encoding="utf-8", errors="replace")
        require(re.search(rf"^Design\s*:\s*{re.escape(cfg['top'])}\s*$", text, re.M) is not None, f"report top mismatch in {report} at {point}")

    setup_slack, setup_status = worst_slack(point / "reports/timing_setup.rpt")
    hold_slack, hold_status = worst_slack(point / "reports/timing_hold.rpt")
    constraints = (point / "reports/constraint_violators.rpt").read_text()
    no_violation_sections = constraints.count("This design has no violated constraints.")
    point_pass = (
        setup_status == "MET"
        and setup_slack >= 0.0
        and hold_status == "MET"
        and hold_slack >= 0.0
        and no_violation_sections == 5
    )
    qor = (point / "reports/qor.rpt").read_text()
    check_timing = (point / "reports/check_timing_postcompile.rpt").read_text()
    refs = (point / "reports/references_postcompile.rpt").read_text()
    return {
        "design": design,
        "top": cfg["top"],
        "period_ns": period,
        "clock_report_period_ns": float(match.group(1)),
        "setup_worst_slack_ns": setup_slack,
        "setup_status": setup_status,
        "hold_worst_slack_ns": hold_slack,
        "hold_status": hold_status,
        "constraint_sections_without_violations": no_violation_sections,
        "point_pass": point_pass,
        "cell_area_um2": qos_field(qor, "Cell Area"),
        "macro_count": qos_field(qor, "Macro Count", int),
        "leaf_cell_count": qos_field(qor, "Leaf Cell Count", int),
        "levels_of_logic": qos_field(qor, "Levels of Logic"),
        "postcompile_check_timing_warning_count": len(re.findall(r"^Warning:", check_timing, re.M)),
        "postcompile_blackbox_named_reference_count": refs.count(".blackbox."),
        "mapped_netlist_present": True,
        "identity_exact": True,
        "point_symlink_count": sum(1 for path in point.rglob("*") if path.is_symlink()),
    }


def nominal_replay() -> dict:
    with tempfile.TemporaryDirectory(prefix="m101-nominal-replay-") as tmp:
        output = Path(tmp) / "receipt.json"
        run = subprocess.run(
            [
                "python3",
                "dc_handoff/scripts/audit_m101_pwp_metadata_fmax_sweep.py",
                "--run-dir",
                "dc_handoff/runs/m101_pwp_metadata_fmax_sweep_r1_20260824",
                "--contract",
                "contracts/m101_pwp_metadata_fmax_sweep_synopsys_contract_r1_20260824.json",
                "--output",
                str(output),
            ],
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            cwd=str(ROOT),
        )
        return {
            "rc": run.returncode,
            "stdout": run.stdout.strip(),
            "stderr": run.stderr.strip(),
            "output_sha256": sha256(output) if output.is_file() else None,
            "byte_identical_to_sealed_receipt": output.is_file() and output.read_bytes() == PROD_RECEIPT.read_bytes(),
        }


def hostile_auditor_attack() -> dict:
    """Alias a 3.000 ns point as 2.750 ns, omit all netlists, and raise the contract threshold."""
    production_required = (
        "dc.log",
        "dc_backend.rc",
        "BACKEND_COMPLETE.txt",
        "point_identity.txt",
        "reports/qor.rpt",
        "reports/area.rpt",
        "reports/timing_setup.rpt",
        "reports/timing_hold.rpt",
        "reports/constraint_violators.rpt",
        "reports/check_design_postcompile.rpt",
        "reports/check_timing_postcompile.rpt",
        "reports/references_postcompile.rpt",
        "reports/resources_precompile.rpt",
        "reports/resources_postcompile.rpt",
    )
    with tempfile.TemporaryDirectory(prefix="m101-hostile-audit-") as tmp:
        base = Path(tmp)
        attack_run = base / "run"
        attack_run.mkdir()
        (attack_run / "admission.txt").symlink_to((RUN / "admission.txt").resolve())
        (attack_run / "BACKEND_COMPLETE_AWAITING_AUDIT.txt").symlink_to(
            (RUN / "BACKEND_COMPLETE_AWAITING_AUDIT.txt").resolve()
        )
        for design in DESIGNS:
            for period in PERIODS:
                requested_name = point_name(design, period)
                source_name = "m99_3p000ns" if requested_name == "m99_2p750ns" else requested_name
                source = RUN / source_name
                target = attack_run / requested_name
                (target / "reports").mkdir(parents=True)
                for rel in production_required:
                    destination = target / rel
                    destination.symlink_to((source / rel).resolve())

        hostile_contract = json.loads(CONTRACT.read_text())
        hostile_contract["acceptance_gates"]["m99_to_m85_achieved_grid_frequency_ratio_min"] = 99.0
        hostile_contract["acceptance_gates"]["all_points_exact_input_identity"] = False
        hostile_contract_path = base / "hostile_contract.json"
        hostile_contract_path.write_text(json.dumps(hostile_contract, indent=2) + "\n")
        output = base / "receipt.json"
        run = subprocess.run(
            ["python3", str(PROD_AUDITOR), "--run-dir", str(attack_run), "--contract", str(hostile_contract_path), "--output", str(output)],
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        receipt = json.loads(output.read_text()) if output.is_file() else {}
        aliased_identity = (attack_run / "m99_2p750ns/point_identity.txt").read_text().splitlines()
        return {
            "attack": "3.000ns evidence aliased under m99_2p750ns; zero mapped netlists; contract ratio minimum changed to 99x; exact-input gate changed to false",
            "rc": run.returncode,
            "stdout": run.stdout.strip(),
            "stderr": run.stderr.strip(),
            "production_status": receipt.get("status"),
            "production_all_acceptance_gates_pass": receipt.get("all_acceptance_gates_pass"),
            "reported_m99_fastest_period_ns": receipt.get("fastest_passing_grid_points", {}).get("m99", {}).get("period_ns"),
            "aliased_point_identity_line": aliased_identity[2],
            "mapped_netlist_count": len(list(attack_run.glob("**/netlist/*"))),
            "hostile_contract_ratio_min": 99.0,
            "hostile_contract_all_points_exact_input_identity": False,
            "production_receipt_gate_keys": sorted(receipt.get("acceptance_gates", {}).keys()),
            "attack_exposes_fail_open": (
                run.returncode == 0
                and receipt.get("all_acceptance_gates_pass") is True
                and receipt.get("fastest_passing_grid_points", {}).get("m99", {}).get("period_ns") == 2.75
                and aliased_identity[2] == "clock_period_ns=3.000"
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(CONTRACT.read_text())
    frozen_hashes = {
        "contract": sha256(CONTRACT),
        "production_auditor": sha256(PROD_AUDITOR),
        "launch_script": sha256(ROOT / "dc_handoff/scripts/run_dc_m101_pwp_metadata_fmax_sweep.sh"),
        "tcl": sha256(ROOT / contract["frozen_sweep"]["tcl"]),
        "sdc": sha256(ROOT / contract["frozen_sweep"]["sdc"]),
        "m85_filelist": sha256(ROOT / DESIGNS["m85"]["filelist"]),
        "m99_filelist": sha256(ROOT / DESIGNS["m99"]["filelist"]),
        "m82_rtl": sha256(ROOT / "rtl_m82/zero_bubble_elastic_pwp_stream.sv"),
        "m85_rtl": sha256(ROOT / "rtl_m85/guarded_wordpacked_pwp_stream.sv"),
        "m99_rtl": sha256(ROOT / "rtl_m99/phase_slack_guarded_wordpacked_pwp_stream.sv"),
        "docs_359": sha256(DOC359),
        "production_receipt": sha256(PROD_RECEIPT),
    }
    expected_hashes = {
        "contract": "dad2b791d505b9532f7924b80e28cd899983e2b097f993f5b1df1c1a97a16c50",
        "production_auditor": "9dbbbd9bde1cbd67c5aab272b978dc7778b85721f0ec57dade9afaad09d4230e",
        "tcl": contract["frozen_sweep"]["tcl_sha256"],
        "sdc": contract["frozen_sweep"]["sdc_sha256"],
        "m85_filelist": DESIGNS["m85"]["filelist_sha256"],
        "m99_filelist": DESIGNS["m99"]["filelist_sha256"],
        "m82_rtl": contract["designs"]["m85_unrolled"]["rtl_sha256"]["rtl_m82/zero_bubble_elastic_pwp_stream.sv"],
        "m85_rtl": contract["designs"]["m85_unrolled"]["rtl_sha256"]["rtl_m85/guarded_wordpacked_pwp_stream.sv"],
        "m99_rtl": contract["designs"]["m99_phase_slack"]["rtl_sha256"]["rtl_m99/phase_slack_guarded_wordpacked_pwp_stream.sv"],
        "docs_359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
        "production_receipt": "3536b3ef1ef6eea628857edefb05bf687568eacb18d04e8774b46f820c11ff45",
    }
    hash_checks = {name: frozen_hashes[name] == value for name, value in expected_hashes.items()}
    require(all(hash_checks.values()), f"frozen hash mismatch: {hash_checks}")

    points = {design: [audit_point(design, period) for period in PERIODS] for design in DESIGNS}
    fastest = {
        design: min((point for point in values if point["point_pass"]), key=lambda point: point["period_ns"])
        for design, values in points.items()
    }
    ratio = fastest["m85"]["period_ns"] / fastest["m99"]["period_ns"]
    area_fraction = fastest["m99"]["cell_area_um2"] / fastest["m85"]["cell_area_um2"]
    matched_area = [
        {
            "period_ns": a["period_ns"],
            "m99_area_fraction_of_m85": b["cell_area_um2"] / a["cell_area_um2"],
        }
        for a, b in zip(points["m85"], points["m99"])
    ]
    sealed = json.loads(PROD_RECEIPT.read_text())
    point_reconciliation = all(
        abs(actual["setup_worst_slack_ns"] - recorded["setup_worst_slack_ns"]) < 1e-12
        and abs(actual["cell_area_um2"] - recorded["cell_area_um2"]) < 1e-12
        and actual["point_pass"] == recorded["point_pass"]
        for design in DESIGNS
        for actual, recorded in zip(points[design], sealed["grid_points"][design])
    )
    attack = hostile_auditor_attack()
    require(attack["attack_exposes_fail_open"], "hostile auditor attack no longer reproduces")

    output = {
        "schema": "m101_pwp_metadata_fmax_sweep_independent_hammer_audit_v1",
        "status": "CONDITIONAL_PASS_CURRENT_EVIDENCE_RECONCILES_BUT_PRODUCTION_SEAL_FAILS_CLOSEDNESS",
        "score": 72,
        "severity_counts": {"P0": 0, "P1": 2, "P2": 3},
        "frozen_hashes": frozen_hashes,
        "frozen_hash_checks": hash_checks,
        "current_evidence": {
            "all_16_points_identity_top_clock_and_backend_exact": True,
            "all_16_points_zero_symlinks": all(point["point_symlink_count"] == 0 for values in points.values() for point in values),
            "all_16_points_mapped_netlists_present": all(point["mapped_netlist_present"] for values in points.values() for point in values),
            "all_16_points_postcompile_check_timing_warning_free": all(point["postcompile_check_timing_warning_count"] == 0 for values in points.values() for point in values),
            "all_16_points_no_named_postcompile_blackbox": all(point["postcompile_blackbox_named_reference_count"] == 0 for values in points.values() for point in values),
            "sealed_receipt_point_reconciliation": point_reconciliation,
            "points": points,
        },
        "independent_metrics": {
            "m85_fastest_passing_grid_period_ns": fastest["m85"]["period_ns"],
            "m85_previous_grid_period_ns": 3.75,
            "m85_previous_grid_setup_wns_ns": next(p["setup_worst_slack_ns"] for p in points["m85"] if p["period_ns"] == 3.75),
            "m99_fastest_passing_grid_period_ns": fastest["m99"]["period_ns"],
            "m99_fastest_is_lower_grid_boundary": True,
            "m99_lower_boundary_setup_wns_ns": fastest["m99"]["setup_worst_slack_ns"],
            "achieved_grid_frequency_ratio": ratio,
            "area_fraction_at_each_design_fastest_point": area_fraction,
            "area_reduction_at_each_design_fastest_point": 1.0 - area_fraction,
            "matched_period_area_fractions": matched_area,
            "matched_period_area_fraction_min": min(row["m99_area_fraction_of_m85"] for row in matched_area),
            "matched_period_area_fraction_max": max(row["m99_area_fraction_of_m85"] for row in matched_area),
            "metric_interpretation": "mapped target-grid closure ratio, not continuous Fmax and not module or system throughput speedup",
        },
        "production_auditor": {
            "nominal_replay": nominal_replay(),
            "hostile_attack": attack,
            "contract_requires_all_points_exact_input_identity": "all_points_exact_input_identity" in contract["acceptance_gates"],
            "sealed_receipt_contains_all_points_exact_input_identity_gate": "all_points_exact_input_identity" in sealed["acceptance_gates"],
            "sealed_manifest_hashes_any_point_report_or_netlist": False,
        },
        "findings": [
            {
                "id": "P1-1",
                "severity": "P1",
                "title": "Production postrun audit is fail-open on point identity, clock, netlists, and contract thresholds",
                "impact": "A 3.000 ns M99 run can be placed under the 2.750 ns directory, all mapped netlists can be absent, and the contract minimum can be raised to 99x while the auditor still emits PASS and 363.636 MHz.",
                "required_fix": "Parse and compare point_identity content; parse report_clocks and report Design/top; verify every frozen RTL/filelist/TCL/SDC/library hash and reject symlinks; require mapped V/DDC/SDC; derive every threshold and gate from the contract.",
            },
            {
                "id": "P1-2",
                "severity": "P1",
                "title": "The durable seal does not cover the run evidence",
                "impact": "SHA256SUMS.complete_r1 hashes the auditor, contract, one marker, and receipt, but no DC log, report, point identity, DDC, mapped Verilog, or mapped SDC. Receipt hashes only four files per point and cannot reconstruct the launched input identity.",
                "required_fix": "Create a canonical manifest over all 16 point directories and all admission inputs, then bind that manifest into the receipt and review.",
            },
            {
                "id": "P2-1",
                "severity": "P2",
                "title": "Same-function wording is stronger than the functional evidence",
                "impact": "M99 is differential-tested after latency alignment on directed and 1728 frozen phases, but it adds a 128-edge audit phase and has no exhaustive cross-RTL formal/refinement proof.",
                "required_fix": "Use 'latency-aligned, frozen-workload differential equivalence' or add an exhaustive sequential refinement/equivalence proof that explicitly models the phase-setup latency.",
            },
            {
                "id": "P2-2",
                "severity": "P2",
                "title": "Both admitted frontier points have negligible timing margin and M99 is censored by the lower grid edge",
                "impact": "M85 4.000 ns reports 0.0000 ns MET and M99 2.750 ns reports +0.0009 ns; there is no failing M99 point below 2.750 ns. This supports target closure but not a stable continuous-Fmax estimate.",
                "required_fix": "Add a finer bracket around both transitions and report a guardbanded point or confidence interval; retain the exact phrase 'frozen-grid target closure ratio'.",
            },
            {
                "id": "P2-3",
                "severity": "P2",
                "title": "Area and frequency remain logic-only pre-macro ideal-clock estimates",
                "impact": "Matched-period area reduction is robust across the grid, but no clock tree, routing parasitics, SRAM macros, workload activity, power, or energy is included.",
                "required_fix": "Keep this as a module-level mapped standard-cell result; do not call it PPA, module speedup, or system speedup until post-layout/macro-aware evidence exists.",
            },
        ],
        "claim_admission": {
            "admit_now": [
                "Current files independently show M99 closes the frozen 2.750 ns target while M85 first closes 4.000 ns under the same pre-macro DC recipe.",
                "The corresponding achieved-grid target-frequency ratio is 1.454545x.",
                "The fastest-point standard-cell area fraction is 0.478168; matched-period fractions range from about 0.390 to 0.492.",
            ],
            "withhold_until_fixes": [
                "fail-closed sealed M101 admission",
                "continuous Fmax",
                "unqualified same-function equivalence",
                "module throughput speedup, system speedup, PPA, power, energy, or DATE headline",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"PASS independent-current-evidence ratio={ratio:.9f}x score=72; FAIL production-seal-closedness")


if __name__ == "__main__":
    main()
