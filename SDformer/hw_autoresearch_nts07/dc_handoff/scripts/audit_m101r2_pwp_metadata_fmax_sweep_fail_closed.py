#!/usr/bin/env python3
"""Fail-closed M101-r2 audit and durable manifest for the frozen DC grid."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


PERIODS = (2.750, 3.000, 3.250, 3.500, 3.750, 4.000, 4.250, 4.500)
DESIGNS = {
    "m85": {
        "top": "guarded_wordpacked_pwp_stream",
        "contract_key": "m85_unrolled",
    },
    "m99": {
        "top": "phase_slack_guarded_wordpacked_pwp_stream",
        "contract_key": "m99_phase_slack",
    },
}
EXPECTED_CONTRACT_SHA256 = (
    "dad2b791d505b9532f7924b80e28cd899983e2b097f993f5b1df1c1a97a16c50")
EXPECTED_GATE_KEYS = {
    "all_16_backends_complete",
    "all_points_exact_input_identity",
    "both_designs_have_at_least_one_passing_point",
    "m85_and_m99_3ns_anchor_repeat_within_area_tolerance",
    "m85_and_m99_3ns_setup_slack_sign_matches_anchor",
    "m99_fastest_passing_grid_period_ns_max",
    "m99_to_m85_achieved_grid_frequency_ratio_min",
    "m99_area_fraction_of_m85_at_each_designs_fastest_passing_point_max",
}
REQUIRED_REPORTS = (
    "qor.rpt",
    "area.rpt",
    "clocks.rpt",
    "timing_setup.rpt",
    "timing_hold.rpt",
    "constraint_violators.rpt",
    "check_design_postcompile.rpt",
    "check_timing_postcompile.rpt",
    "references_postcompile.rpt",
    "resources_precompile.rpt",
    "resources_postcompile.rpt",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise RuntimeError("non-standard JSON constant: " + raw)

    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def require_regular(path, label):
    path = Path(path)
    require(path.exists(), "missing " + label + ": " + str(path))
    require(not path.is_symlink(), "symlink forbidden for " + label + ": " + str(path))
    require(path.is_file(), "not a regular file for " + label + ": " + str(path))
    require(path.stat().st_size > 0, "empty file for " + label + ": " + str(path))


def point_name(design, period):
    return f"{design}_{period:.3f}ns".replace(".", "p").replace("pns", "ns")


def field(text, label, number=float):
    match = re.search(
        rf"^\s*{re.escape(label)}:\s+(-?[0-9]+(?:\.[0-9]+)?)\s*$",
        text, re.M)
    require(match is not None, "missing QoR field " + label)
    return number(match.group(1))


def worst_slack(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    values = [
        (float(value), status)
        for status, value in re.findall(
            r"slack\s+\((MET|VIOLATED)\)\s+(-?[0-9]+(?:\.[0-9]+)?)",
            text)
    ]
    require(values, "no timing slack records in " + str(path))
    return min(values, key=lambda item: item[0])


def report_top(path, top):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    require(re.search(rf"^Design\s*:\s*{re.escape(top)}\s*$", text, re.M),
            "report top mismatch in " + str(path))


def verify_frozen_inputs(contract, run_dir):
    for design_key, cfg in DESIGNS.items():
        frozen = contract["designs"][cfg["contract_key"]]
        require(frozen["top"] == cfg["top"], "contract top drift " + design_key)
        for path_key, sha_key in (
            ("filelist", "filelist_sha256"),
            ("functional_contract", "functional_contract_sha256"),
            ("sealed_vcs_completion", "sealed_vcs_completion_sha256"),
        ):
            path = Path(frozen[path_key])
            require_regular(path, design_key + " " + path_key)
            require(sha256(path) == frozen[sha_key],
                    design_key + " frozen input SHA drift: " + str(path))
        for path, expected in frozen["rtl_sha256"].items():
            require_regular(path, design_key + " rtl")
            require(sha256(path) == expected,
                    design_key + " RTL SHA drift: " + path)

    sweep = contract["frozen_sweep"]
    require(tuple(sweep["period_grid_ns"]) == PERIODS,
            "contract period grid drift")
    for path_key, sha_key in (("tcl", "tcl_sha256"), ("sdc", "sdc_sha256")):
        path = Path(sweep[path_key])
        require_regular(path, "sweep " + path_key)
        require(sha256(path) == sweep[sha_key], "sweep input SHA drift: " + str(path))

    admission = (run_dir / "admission.txt").read_text(encoding="utf-8")
    setup_match = re.search(r"^setup_library=(.+)$", admission, re.M)
    hold_match = re.search(r"^hold_library=(.+)$", admission, re.M)
    require(setup_match and hold_match, "admission library paths missing")
    setup_library = Path(setup_match.group(1))
    hold_library = Path(hold_match.group(1))
    require_regular(setup_library, "setup library")
    require_regular(hold_library, "hold library")
    require(sha256(setup_library) == sweep["setup_library_sha256"],
            "setup library SHA drift")
    require(sha256(hold_library) == sweep["hold_library_sha256"],
            "hold library SHA drift")
    return setup_library, hold_library


def audit_point(run_dir, contract, design_key, period):
    cfg = DESIGNS[design_key]
    frozen = contract["designs"][cfg["contract_key"]]
    point = run_dir / point_name(design_key, period)
    require(point.is_dir() and not point.is_symlink(), "missing/symlink point " + str(point))
    for path in point.rglob("*"):
        require(not path.is_symlink(), "point evidence symlink forbidden: " + str(path))

    required = ["dc.log", "dc_backend.rc", "BACKEND_COMPLETE.txt", "point_identity.txt"]
    required += ["reports/" + name for name in REQUIRED_REPORTS]
    required += [
        "netlist/" + cfg["top"] + "_mapped.v",
        "netlist/" + cfg["top"] + "_mapped.sdc",
        "netlist/" + cfg["top"] + ".ddc",
        "netlist/" + cfg["top"] + ".svf",
    ]
    for relative in required:
        require_regular(point / relative, "point evidence")

    identity = (point / "point_identity.txt").read_text(encoding="utf-8").splitlines()
    require(len(identity) == 4, "point identity line count drift at " + str(point))
    require(identity[0] == "design_key=" + design_key, "design key drift at " + str(point))
    require(identity[1] == "design_name=" + cfg["top"], "design top drift at " + str(point))
    require(identity[2] == f"clock_period_ns={period:.3f}", "period identity drift at " + str(point))
    require(identity[3].split()[0] == frozen["filelist_sha256"],
            "filelist identity drift at " + str(point))

    require((point / "dc_backend.rc").read_text().strip() == "0",
            "backend rc != 0 at " + str(point))
    require((point / "BACKEND_COMPLETE.txt").read_text().strip()
            == "backend_complete=true", "backend incomplete at " + str(point))
    log = (point / "dc.log").read_text(encoding="utf-8", errors="replace")
    require(not re.search(r"^Error:", log, re.M), "DC Error line at " + str(point))
    require("Current design is now '" + cfg["top"] + "'." in log,
            "DC log top mismatch at " + str(point))
    require("Using operating conditions '" + contract["frozen_sweep"]["operating_condition"] + "'" in log,
            "DC operating condition mismatch at " + str(point))

    clocks_path = point / "reports/clocks.rpt"
    report_top(clocks_path, cfg["top"])
    clocks = clocks_path.read_text(encoding="utf-8", errors="replace")
    clock_match = re.search(r"^core_clk\s+([0-9]+(?:\.[0-9]+)?)\s+", clocks, re.M)
    require(clock_match and math.isclose(float(clock_match.group(1)), period,
                                         abs_tol=1e-9),
            "clock report period mismatch at " + str(point))
    for name in ("qor.rpt", "timing_setup.rpt", "timing_hold.rpt",
                 "references_postcompile.rpt"):
        report_top(point / "reports" / name, cfg["top"])

    mapped = (point / "netlist" / (cfg["top"] + "_mapped.v")).read_text(
        encoding="utf-8", errors="replace")
    require(re.search(rf"\bmodule\s+{re.escape(cfg['top'])}\b", mapped),
            "mapped netlist top missing at " + str(point))

    qor = (point / "reports/qor.rpt").read_text(encoding="utf-8", errors="replace")
    setup_slack, setup_status = worst_slack(point / "reports/timing_setup.rpt")
    hold_slack, hold_status = worst_slack(point / "reports/timing_hold.rpt")
    no_violations = (point / "reports/constraint_violators.rpt").read_text(
        encoding="utf-8", errors="replace").count(
            "This design has no violated constraints.")
    setup_tns = field(qor, "Total Negative Slack")
    setup_violations = field(qor, "No. of Violating Paths")
    hold_tns = field(qor, "Total Hold Violation")
    hold_violations = field(qor, "No. of Hold Violations")
    point_pass = (
        setup_status == "MET" and setup_slack >= 0.0
        and hold_status == "MET" and hold_slack >= 0.0
        and setup_tns == 0.0 and setup_violations == 0.0
        and hold_tns == 0.0 and hold_violations == 0.0
        and no_violations == 5)
    return {
        "design_key": design_key,
        "top": cfg["top"],
        "period_ns": period,
        "clock_report_period_ns": float(clock_match.group(1)),
        "setup_worst_slack_ns": setup_slack,
        "setup_status": setup_status,
        "hold_worst_slack_ns": hold_slack,
        "hold_status": hold_status,
        "setup_tns_ns": setup_tns,
        "setup_violating_paths": setup_violations,
        "hold_tns_ns": hold_tns,
        "hold_violating_paths": hold_violations,
        "constraint_sections_without_violations": no_violations,
        "point_pass": point_pass,
        "levels_of_logic": field(qor, "Levels of Logic"),
        "critical_path_length_ns": field(qor, "Critical Path Length"),
        "cell_area_um2": field(qor, "Cell Area"),
        "leaf_cell_count": field(qor, "Leaf Cell Count", int),
        "combinational_cell_count": field(qor, "Combinational Cell Count", int),
        "sequential_cell_count": field(qor, "Sequential Cell Count", int),
        "macro_count": field(qor, "Macro Count", int),
        "identity_exact": True,
        "mapped_artifacts_present": True,
        "point_symlink_count": 0,
        "point_identity_sha256": sha256(point / "point_identity.txt"),
        "qor_sha256": sha256(point / "reports/qor.rpt"),
        "setup_sha256": sha256(point / "reports/timing_setup.rpt"),
        "hold_sha256": sha256(point / "reports/timing_hold.rpt"),
        "mapped_verilog_sha256": sha256(point / "netlist" / (cfg["top"] + "_mapped.v")),
        "mapped_sdc_sha256": sha256(point / "netlist" / (cfg["top"] + "_mapped.sdc")),
        "ddc_sha256": sha256(point / "netlist" / (cfg["top"] + ".ddc")),
    }


def build_manifest(run_dir, contract_path, auditor_path, external_paths, output):
    entries = []
    for path in sorted(run_dir.rglob("*"), key=lambda item: str(item)):
        require(not path.is_symlink(), "run evidence symlink forbidden: " + str(path))
        if path.is_file():
            require(path.stat().st_size > 0, "empty run evidence: " + str(path))
            entries.append((sha256(path), "run/" + str(path.relative_to(run_dir))))
    for label, path in (("contract", contract_path), ("auditor", auditor_path)):
        entries.append((sha256(path), label + "/" + path.name))
    for label, path in external_paths:
        entries.append((sha256(path), "external/" + label + "/" + path.name))
    require(len(entries) > 16 * 20, "durable manifest unexpectedly small")
    output.write_text("".join(digest + "  " + label + "\n"
                              for digest, label in entries), encoding="utf-8")
    return len(entries), sha256(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    args = parser.parse_args()
    require(args.run_dir.is_dir() and not args.run_dir.is_symlink(), "bad run directory")
    require(not args.output.exists(), "refusing receipt overwrite")
    require(not args.manifest_output.exists(), "refusing manifest overwrite")
    require_regular(args.contract, "contract")
    require(sha256(args.contract) == EXPECTED_CONTRACT_SHA256,
            "frozen M101 contract identity drift")
    contract = strict_json(args.contract)
    require(set(contract["acceptance_gates"]) == EXPECTED_GATE_KEYS,
            "contract acceptance gate set drift")
    require_regular(args.run_dir / "admission.txt", "admission")
    require_regular(args.run_dir / "BACKEND_COMPLETE_AWAITING_AUDIT.txt",
                    "backend grid marker")
    setup_library, hold_library = verify_frozen_inputs(contract, args.run_dir)

    points = {
        design: [audit_point(args.run_dir, contract, design, period)
                 for period in PERIODS]
        for design in DESIGNS
    }
    fastest = {}
    for design, values in points.items():
        passing = [point for point in values if point["point_pass"]]
        require(passing, design + " has no passing point")
        fastest[design] = min(passing, key=lambda point: point["period_ns"])

    m85_fast = fastest["m85"]
    m99_fast = fastest["m99"]
    ratio = m85_fast["period_ns"] / m99_fast["period_ns"]
    area_fraction = m99_fast["cell_area_um2"] / m85_fast["cell_area_um2"]
    anchors = {
        design: next(point for point in points[design]
                     if math.isclose(point["period_ns"], 3.0))
        for design in DESIGNS
    }
    anchor = contract["anchor_identity"]
    tolerance = anchor["repeat_area_relative_tolerance"]
    area_repeat = (
        abs(anchors["m85"]["cell_area_um2"] - anchor["m97_3ns_cell_area_um2"])
        / anchor["m97_3ns_cell_area_um2"] <= tolerance
        and abs(anchors["m99"]["cell_area_um2"] - anchor["m100_3ns_cell_area_um2"])
        / anchor["m100_3ns_cell_area_um2"] <= tolerance)
    sign_repeat = (anchors["m85"]["setup_worst_slack_ns"] < 0.0
                   and anchors["m99"]["setup_worst_slack_ns"] >= 0.0)
    thresholds = contract["acceptance_gates"]
    gates = {
        "all_16_backends_complete": sum(len(values) for values in points.values()) == 16,
        "all_points_exact_input_identity": all(
            point["identity_exact"] and point["mapped_artifacts_present"]
            and point["point_symlink_count"] == 0
            for values in points.values() for point in values),
        "both_designs_have_at_least_one_passing_point": True,
        "m85_and_m99_3ns_anchor_repeat_within_area_tolerance": area_repeat,
        "m85_and_m99_3ns_setup_slack_sign_matches_anchor": sign_repeat,
        "m99_fastest_passing_grid_period_ns_max":
            m99_fast["period_ns"] <= thresholds[
                "m99_fastest_passing_grid_period_ns_max"],
        "m99_to_m85_achieved_grid_frequency_ratio_min":
            ratio >= thresholds[
                "m99_to_m85_achieved_grid_frequency_ratio_min"],
        "m99_area_fraction_of_m85_at_each_designs_fastest_passing_point_max":
            area_fraction <= thresholds[
                "m99_area_fraction_of_m85_at_each_designs_fastest_passing_point_max"],
    }
    require(set(gates) == EXPECTED_GATE_KEYS, "receipt gate set drift")
    require(all(gates.values()), "one or more frozen M101-r2 gates failed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_count, manifest_sha = build_manifest(
        args.run_dir, args.contract, Path(__file__).resolve(),
        (("setup_library", setup_library), ("hold_library", hold_library)),
        args.manifest_output)
    receipt = {
        "schema": "m101r2_pwp_metadata_fmax_sweep_fail_closed_receipt_v1",
        "status": "PASS_FAIL_CLOSED_FROZEN_GRID_TARGET_CLOSURE",
        "identity": {
            "contract_sha256": sha256(args.contract),
            "auditor_sha256": sha256(Path(__file__).resolve()),
            "durable_run_manifest_sha256": manifest_sha,
            "durable_run_manifest_entries": manifest_count,
        },
        "grid_points": points,
        "fastest_passing_grid_points": fastest,
        "comparison": {
            "m85_achieved_frozen_grid_target_mhz": 1000.0 / m85_fast["period_ns"],
            "m99_achieved_frozen_grid_target_mhz": 1000.0 / m99_fast["period_ns"],
            "frozen_grid_target_closure_ratio": ratio,
            "fastest_point_standard_cell_area_fraction": area_fraction,
            "fastest_point_standard_cell_area_reduction_fraction": 1.0 - area_fraction,
        },
        "acceptance_gates": gates,
        "all_acceptance_gates_pass": True,
        "claim_boundary": {
            "latency_aligned_frozen_trace_differential_equivalence": True,
            "same_recipe_logic_only_pre_macro": True,
            "frozen_grid_target_closure_ratio": True,
            "continuous_fmax": False,
            "postlayout_fmax": False,
            "macro_inclusive_ppa": False,
            "module_throughput_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS M101-r2 m85={:.3f}ns m99={:.3f}ns ratio={:.9f}x area_fraction={:.9f} manifest_entries={}".format(
        m85_fast["period_ns"], m99_fast["period_ns"], ratio,
        area_fraction, manifest_count), flush=True)


if __name__ == "__main__":
    main()
