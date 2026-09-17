"""Parse the two existing DC points only; no EDA or functional rerun."""
from pathlib import Path
import json
import re

P = Path(__file__).resolve().parent


def scalar(text, label):
    hit = re.search(r"^" + re.escape(label) + r"\s*:\s*([0-9.]+)", text, re.M)
    assert hit, label
    return float(hit.group(1))


def paths(file):
    out = []
    for block in file.read_text().split("  Startpoint: ")[1:]:
        endpoint = re.search(r"Endpoint:\s*(\S+)", block)
        arrival = re.search(r"data arrival time\s+([0-9.]+)", block)
        required = re.search(r"data required time\s+([0-9.]+)", block)
        slack = re.search(r"slack \((MET|VIOLATED)\)\s+(-?[0-9.]+)", block)
        assert endpoint and arrival and required and slack, file
        out.append({
            "startpoint": block.splitlines()[0].split(" (")[0].strip(),
            "endpoint": endpoint.group(1),
            "arrival_ns": float(arrival.group(1)),
            "required_ns": float(required.group(1)),
            "slack_ns": float(slack.group(2)),
            "status": slack.group(1),
        })
    assert out, file
    return out


receipt = json.loads((P / "DC_RUN.json").read_text())
assert [r["arm"] for r in receipt] == [0, 1]
assert all(r["returncode"] == 0 and r["area_report"] and r["timing_report"] for r in receipt)
arms = []
for arm in range(2):
    directory = P / f"dc_{arm}" / "reports"
    area = (directory / "area.rpt").read_text()
    qor = (directory / "qor.rpt").read_text()
    all_paths = paths(directory / "timing_all.rpt")
    decision = paths(directory / "timing_decision.rpt")
    errors = [f.name for f in directory.glob("timing_*_to_decision.rpt") if "Error:" in f.read_text()]
    assert scalar(area, "Number of macros/black boxes") == 0
    assert "No. of Violating Paths:        0.00" in qor
    assert min(r["slack_ns"] for r in all_paths) >= 0
    arms.append({
        "arm": arm,
        "name": "original" if arm == 0 else "normalized",
        "cell_area_library_units": scalar(area, "Total cell area"),
        "combinational_area_library_units": scalar(area, "Combinational area"),
        "sequential_area_library_units": scalar(area, "Noncombinational area"),
        "sequential_cells": int(scalar(area, "Number of sequential cells")),
        "cells": int(scalar(area, "Number of cells")),
        "macros_or_blackboxes": 0,
        "global_worst": all_paths[0],
        "decision_paths": decision,
        "global_top20_paths": all_paths,
        "invalid_optional_source_filter_reports": errors,
    })

summary = {
    "scope": "one decision lane with private runtime two-half LUT; logic mapping only",
    "exactly_two_compile_points": True,
    "functional": json.loads((P / "FUNCTIONAL.json").read_text()),
    "constraints": {
        "corner": "tcbn28hpcplusbwp35p140ssg0p9v125c", "period_ns": 3,
        "setup_uncertainty_ns": 0.2, "hold_uncertainty_ns": 0.05,
        "input_output_delay_ns": 0.25, "input_transition_ns": 0.1,
        "output_load": 0.01, "wireload": "ZeroWireload", "max_fanout": 32,
    },
    "arms": arms,
    "normalized_vs_original": {
        "cell_area_reduction_percent": 100 * (1 - arms[1]["cell_area_library_units"] / arms[0]["cell_area_library_units"]),
        "worst_arrival_reduction_ns": arms[0]["global_worst"]["arrival_ns"] - arms[1]["global_worst"]["arrival_ns"],
        "worst_arrival_reduction_percent": 100 * (1 - arms[1]["global_worst"]["arrival_ns"] / arms[0]["global_worst"]["arrival_ns"]),
    },
    "limitations": [
        "No PT, FM, CTS, placed routing, power measurement, or full-kernel Fmax admission.",
        "Do not multiply private lane area by 80; parent shares LUT/P/N/tails and has other paths.",
        "Latest normalized_bound stores P-1/N-1 and uses carry-in; this frozen probe stores P/N and explicitly subtracts delta in GROUP.",
        "Source-filtered reports failed DC -from cell typing; global and decision endpoint reports are valid. No extra compile was run.",
        "Timing includes unspecialized GROUP/PLANE controls; q-D paths from LUT are not evidence of an active GROUP LUT dependency.",
        "Both compile reports include a zero max_leakage_power objective violation; no energy or leakage comparison is claimed.",
    ],
}
(P / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary["normalized_vs_original"], indent=2))
print("PASS parsed two existing mappings; no EDA rerun")
