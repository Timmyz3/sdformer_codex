#!/usr/bin/env python3
"""Fail-closed postrun audit for the frozen M101 Synopsys period grid."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Tuple


PERIODS = (2.750, 3.000, 3.250, 3.500, 3.750, 4.000, 4.250, 4.500)
DESIGNS = {
    "m85": "guarded_wordpacked_pwp_stream",
    "m99": "phase_slack_guarded_wordpacked_pwp_stream",
}
REQUIRED = (
    "dc.log",
    "dc_backend.rc",
    "BACKEND_COMPLETE.txt",
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def field(text: str, label: str, number: type = float):
    match = re.search(rf"^\s*{re.escape(label)}:\s+(-?[0-9]+(?:\.[0-9]+)?)\s*$", text, re.M)
    require(match is not None, f"missing QoR field {label}")
    return number(match.group(1))


def worst_slack(path: Path) -> Tuple[float, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    values = [
        (float(value), status)
        for status, value in re.findall(
            r"slack\s+\((MET|VIOLATED)\)\s+(-?[0-9]+(?:\.[0-9]+)?)", text
        )
    ]
    require(values, f"no timing slack records in {path}")
    return min(values, key=lambda item: item[0])


def audit_point(run_dir: Path, design_key: str, period: float) -> dict:
    point_dir = run_dir / f"{design_key}_{period:.3f}ns".replace(".", "p")
    # Restore the extension separator changed by the mechanical replacement.
    point_dir = Path(str(point_dir).replace("pns", "ns"))
    require(point_dir.is_dir(), f"missing grid directory {point_dir}")
    missing = [name for name in REQUIRED if not (point_dir / name).is_file()]
    require(not missing, f"missing M101 evidence at {point_dir}: {missing}")
    require((point_dir / "dc_backend.rc").read_text().strip() == "0", f"backend rc != 0 at {point_dir}")
    require((point_dir / "BACKEND_COMPLETE.txt").read_text().strip() == "backend_complete=true", f"backend incomplete at {point_dir}")

    log = (point_dir / "dc.log").read_text(encoding="utf-8", errors="replace")
    require(not re.search(r"^Error:", log, re.M), f"DC Error line at {point_dir}")
    qor = (point_dir / "reports/qor.rpt").read_text(encoding="utf-8", errors="replace")
    setup_slack, setup_status = worst_slack(point_dir / "reports/timing_setup.rpt")
    hold_slack, hold_status = worst_slack(point_dir / "reports/timing_hold.rpt")
    violators = (point_dir / "reports/constraint_violators.rpt").read_text(
        encoding="utf-8", errors="replace"
    )
    no_violation_sections = violators.count("This design has no violated constraints.")
    point_pass = (
        setup_slack >= 0.0
        and setup_status == "MET"
        and hold_slack >= 0.0
        and hold_status == "MET"
        and no_violation_sections == 5
    )
    return {
        "design_key": design_key,
        "top": DESIGNS[design_key],
        "period_ns": period,
        "setup_worst_slack_ns": setup_slack,
        "setup_status": setup_status,
        "hold_worst_slack_ns": hold_slack,
        "hold_status": hold_status,
        "constraint_sections_without_violations": no_violation_sections,
        "point_pass": point_pass,
        "levels_of_logic": field(qor, "Levels of Logic"),
        "critical_path_length_ns": field(qor, "Critical Path Length"),
        "cell_area_um2": field(qor, "Cell Area"),
        "leaf_cell_count": field(qor, "Leaf Cell Count", int),
        "combinational_cell_count": field(qor, "Combinational Cell Count", int),
        "sequential_cell_count": field(qor, "Sequential Cell Count", int),
        "macro_count": field(qor, "Macro Count", int),
        "point_identity_sha256": sha256(point_dir / "point_identity.txt"),
        "qor_sha256": sha256(point_dir / "reports/qor.rpt"),
        "setup_sha256": sha256(point_dir / "reports/timing_setup.rpt"),
        "hold_sha256": sha256(point_dir / "reports/timing_hold.rpt"),
    }


def close_fraction(actual: float, reference: float, tolerance: float) -> bool:
    return abs(actual - reference) / reference <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    require(args.run_dir.is_dir(), f"missing run directory {args.run_dir}")
    require((args.run_dir / "BACKEND_COMPLETE_AWAITING_AUDIT.txt").is_file(), "grid backend completion marker missing")
    contract = json.loads(args.contract.read_text())
    require(tuple(contract["frozen_sweep"]["period_grid_ns"]) == PERIODS, "contract period grid drift")

    points = {
        design: [audit_point(args.run_dir, design, period) for period in PERIODS]
        for design in DESIGNS
    }
    fastest = {}
    for design, design_points in points.items():
        passing = [point for point in design_points if point["point_pass"]]
        require(passing, f"{design} has no passing M101 grid point")
        fastest[design] = min(passing, key=lambda point: point["period_ns"])

    m85_fast = fastest["m85"]
    m99_fast = fastest["m99"]
    ratio = m85_fast["period_ns"] / m99_fast["period_ns"]
    area_fraction = m99_fast["cell_area_um2"] / m85_fast["cell_area_um2"]
    anchor_m85 = next(point for point in points["m85"] if math.isclose(point["period_ns"], 3.0))
    anchor_m99 = next(point for point in points["m99"] if math.isclose(point["period_ns"], 3.0))
    tolerance = contract["anchor_identity"]["repeat_area_relative_tolerance"]
    gates = {
        "all_16_backends_complete": len(points["m85"]) + len(points["m99"]) == 16,
        "all_points_macro_count_zero": all(point["macro_count"] == 0 for values in points.values() for point in values),
        "both_designs_have_at_least_one_passing_point": True,
        "m85_3ns_anchor_area_repeat": close_fraction(
            anchor_m85["cell_area_um2"], contract["anchor_identity"]["m97_3ns_cell_area_um2"], tolerance
        ),
        "m99_3ns_anchor_area_repeat": close_fraction(
            anchor_m99["cell_area_um2"], contract["anchor_identity"]["m100_3ns_cell_area_um2"], tolerance
        ),
        "m85_3ns_anchor_setup_sign_repeat": anchor_m85["setup_worst_slack_ns"] < 0.0,
        "m99_3ns_anchor_setup_sign_repeat": anchor_m99["setup_worst_slack_ns"] >= 0.0,
        "m99_fastest_passing_period_le_3ns": m99_fast["period_ns"] <= 3.0,
        "achieved_grid_frequency_ratio_ge_1p25": ratio >= 1.25,
        "m99_area_fraction_at_fastest_points_le_0p5": area_fraction <= 0.5,
    }
    all_gates = all(gates.values())
    receipt = {
        "schema": "m101_pwp_metadata_fmax_sweep_synopsys_receipt_v1",
        "status": "PASS_SCOPED_SAME_FUNCTION_GRID" if all_gates else "PARTIAL_PASS_ONE_OR_MORE_FROZEN_GATES_FAILED",
        "identity": {
            "contract": str(args.contract),
            "contract_sha256": sha256(args.contract),
            "auditor": str(Path(__file__).resolve()),
            "auditor_sha256": sha256(Path(__file__).resolve()),
            "admission_sha256": sha256(args.run_dir / "admission.txt"),
        },
        "grid_points": points,
        "fastest_passing_grid_points": fastest,
        "comparison": {
            "m85_achieved_grid_frequency_mhz": 1000.0 / m85_fast["period_ns"],
            "m99_achieved_grid_frequency_mhz": 1000.0 / m99_fast["period_ns"],
            "m99_to_m85_achieved_grid_frequency_ratio": ratio,
            "m99_area_fraction_at_each_fastest_passing_point": area_fraction,
            "m99_area_reduction_fraction": 1.0 - area_fraction,
        },
        "acceptance_gates": gates,
        "all_acceptance_gates_pass": all_gates,
        "claim_boundary": {
            "same_function_mapped_grid_ratio": all_gates,
            "continuous_or_postlayout_fmax": False,
            "bit_sparse_physical_baseline": False,
            "m88_cycle_times_frequency_product": False,
            "complete_sram_timing_or_ppa": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        "M101 status={} m85={}ns m99={}ns ratio={:.6f}x area_fraction={:.6f}".format(
            receipt["status"], m85_fast["period_ns"], m99_fast["period_ns"], ratio, area_fraction
        )
    )


if __name__ == "__main__":
    main()
