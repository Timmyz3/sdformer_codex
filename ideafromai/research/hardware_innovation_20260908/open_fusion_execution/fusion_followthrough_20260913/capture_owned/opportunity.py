#!/usr/bin/env python3
"""Reproduce the fixed offline opportunity table with Python 3.12 and NumPy.

From this directory (all input paths resolve relative to this script):
    /opt/anaconda3/bin/python3.12 opportunity.py --check
    /opt/anaconda3/bin/python3.12 opportunity.py --output /tmp/opportunity.json

With no arguments, write the regenerated JSON to stdout. --check recomputes
from the four NPZ files and compares with opportunity.json without writing it.
This is adjacent-frame gold analysis, not a Machine/cache-policy simulation.
The two pixels, H8 grouping and ONE three-level encoder are fixed, not options.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE_PROGRAM = "../../breadth_20260912/source_execution/dense/program.json"
RADII = (64, 256, 1024)
PIXELS = ((0, 0), (0, 1))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tier(radius):
    return max((r for r in RADII if r <= radius), default=0)


def distribution(values):
    a = np.asarray(values, dtype=np.int64)
    return {
        "min": int(a.min()),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "max": int(a.max()),
    }


def summarize(rows):
    lane_equal = sum(r["per_lane_gate10_equal_count"] for r in rows)
    return {
        "group_transitions": len(rows),
        "exact_I24": sum(r["exact_I24"] for r in rows),
        "gate80_equal": sum(r["gate80_equal"] for r in rows),
        "gate80_equal_nonexact": sum(
            r["gate80_equal"] and not r["exact_I24"] for r in rows
        ),
        "fixed_three_level_accepts": sum(r["fixed_three_level_accepts"] for r in rows),
        "maximum_radius_accepts": sum(r["maximum_radius_accepts"] for r in rows),
        "per_lane_gate10_equal": lane_equal,
        "per_lane_gate10_total": len(rows) * 8,
        "per_lane_gate10_equal_rate": lane_equal / (len(rows) * 8),
        "per_lane_maximum_radius_accepts": sum(
            sum(r["per_lane_maximum_radius_accepts"]) for r in rows
        ),
        "gate_bit_flips": sum(r["gate_bit_flips"] for r in rows),
        "reference_three_level_histogram": {
            str(level): sum(r["reference_three_level_radius"] == level for r in rows)
            for level in (0, *RADII)
        },
        "actual_max_abs_delta_I24": distribution(
            [r["actual_max_abs_delta_I24"] for r in rows]
        ),
        "reference_maximum_Linf_safe_integer_radius": distribution(
            [r["reference_maximum_Linf_safe_integer_radius"] for r in rows]
        ),
    }


def regenerate():
    index = json.loads((HERE / "index.json").read_text())
    require(index["complete"] and len(index["frames"]) == 4, "Need all four captures")
    require(index["structure"] == "dense", "This fixed analysis is for matched dense")
    with np.load(HERE / index["parameters"], allow_pickle=False) as parameters:
        require(len(parameters.files) == 32, "Unexpected deployed parameter contract")
        a = parameters["As_q16"].astype(np.int64)
    require(a.shape == (10, 10), "Expected T10 source matrix")
    row_l1 = np.abs(a).sum(axis=1)
    require(np.all(row_l1 > 0), "Fixed source requires nonzero rows")
    program = json.loads((HERE / SOURCE_PROGRAM).read_text())
    gates = sorted((op for op in program if op["kind"] == "gate"), key=lambda op: op["output_t"])
    require([op["output_t"] for op in gates] == list(range(10)), "Expected ten source gates")
    for op in gates:
        require(op["constant"] == -1, "Fixed source expects nonconstant gates")
        require(op["direction"] in (-1, 1), "Unsupported gate direction")
        operands = op["operands"]
        require(len(operands) == 1 and operands[0]["shift"] == 0
                and operands[0]["sign"] == 1, "Expected unshifted positive S gate operand")
    thresholds = np.array([op["threshold"] for op in gates], dtype=np.int64)
    directions = np.array([op["direction"] for op in gates], dtype=np.int64)
    frames = []
    geometry = None
    checked = 0
    for frame in index["frames"]:
        with np.load(HERE / frame["capture"], allow_pickle=False) as capture:
            current_geometry = json.loads(capture["window_geometry_json"].item())["interior"]
            if geometry is None:
                geometry = current_geometry
            require(current_geometry == geometry, "Frame geometry changed")
            x_all = capture["interior_I24"].astype(np.int64)
            s_all = capture["interior_source_S48"].astype(np.int64)
            g_all = capture["interior_sn1_gate"]
        require(x_all.shape == s_all.shape == g_all.shape == (10, 96, 11, 11),
                "Unexpected interior capture shape")
        entries = []
        for y, x in PIXELS:
            for h in range(0, 96, 8):
                inputs = x_all[:, h:h + 8, y, x]
                s = s_all[:, h:h + 8, y, x]
                g = g_all[:, h:h + 8, y, x]
                require(np.all((g == 0) | (g == 1)), "Nonbinary source gate")
                g = g.astype(bool)
                require(np.all((-2**23 <= inputs) & (inputs < 2**23)), "I24 overflow")
                require(np.all((-2**47 <= s) & (s < 2**47)), "S48 overflow")
                require(np.array_equal(a @ inputs, s), "Captured S disagrees with As @ I24")
                k = thresholds[:, None]
                positive = directions[:, None] == 1
                require(np.array_equal(np.where(positive, s >= k, s <= k), g),
                        "Captured gate disagrees with integer program cutoff")
                # Inclusive safe margin: no integer point at the first flip is included.
                margin = np.where(positive, np.where(g, s - k, k - 1 - s),
                                  np.where(g, k - s, s - k - 1))
                require(np.all(margin >= 0), "Negative safe margin")
                radii = margin // row_l1[:, None]
                entries.append({"inputs": inputs, "gates": g, "margin": margin,
                                "radii": radii, "radius": int(radii.min())})
                checked += g.size
        frames.append(entries)

    rows = []
    origin = geometry["source_origin"]
    for transition in range(1, 4):
        for entry, (old, new) in enumerate(zip(frames[transition - 1], frames[transition])):
            pixel, group = divmod(entry, 12)
            y, x = PIXELS[pixel]
            delta = np.abs(new["inputs"] - old["inputs"])
            max_delta = int(delta.max())
            lane_delta = delta.max(axis=0)
            lane_equal = np.all(new["gates"] == old["gates"], axis=0)
            lane_radius = old["radii"].min(axis=0)
            lane_accept = lane_delta <= lane_radius
            radius = old["radius"]
            encoded_radius = tier(radius)
            accepted = max_delta <= radius
            # Code zero carries no nonzero certificate; exact inputs are listed separately.
            tier_accepted = encoded_radius > 0 and max_delta <= encoded_radius
            same_gates = bool(lane_equal.all())
            require(not accepted or same_gates, "False-positive group certificate")
            require(not tier_accepted or same_gates, "False-positive encoded certificate")
            require(np.all(~lane_accept | lane_equal), "False-positive lane certificate")
            limiting_t, limiting_lane = np.unravel_index(np.argmin(old["radii"]), (10, 8))
            rows.append({
                "transition": transition,
                "reference_frame": index["frames"][transition - 1]["file"],
                "current_frame": index["frames"][transition]["file"],
                "entry": entry,
                "pixel": pixel,
                "local_yx": [y, x],
                "global_yx": [origin[0] + y, origin[1] + x],
                "h_start": group * 8,
                "actual_max_abs_delta_I24": max_delta,
                "exact_I24": max_delta == 0,
                "gate80_equal": same_gates,
                "gate_bit_flips": int(np.count_nonzero(new["gates"] != old["gates"])),
                "reference_three_level_radius": encoded_radius,
                "reference_maximum_Linf_safe_integer_radius": radius,
                "maximum_radius_accepts": accepted,
                "fixed_three_level_accepts": tier_accepted,
                "reference_limiting_t": int(limiting_t),
                "reference_limiting_lane": int(limiting_lane),
                "reference_limiting_margin": int(old["margin"][limiting_t, limiting_lane]),
                "reference_limiting_row_L1": int(row_l1[limiting_t]),
                "per_lane_gate10_equal": lane_equal.tolist(),
                "per_lane_gate10_equal_count": int(lane_equal.sum()),
                "per_lane_gate10_equal_rate": float(lane_equal.mean()),
                "per_lane_max_abs_delta_I24": lane_delta.tolist(),
                "per_lane_maximum_Linf_safe_integer_radius": lane_radius.tolist(),
                "per_lane_maximum_radius_accepts": lane_accept.tolist(),
                "current_three_level_radius": tier(new["radius"]),
                "current_maximum_Linf_safe_integer_radius": new["radius"],
            })
    return {
        "complete": True,
        "scope": "Offline gold/upper-bound explanation only; no Machine execution, no controller input, no parameter/group sweep.",
        "captures": "index.json",
        "parent": "matched dense stage320; 32 deployed arrays exactly matched",
        "window": "interior",
        "fixed_pixels": [{"pixel": i, "local_yx": [y, x],
                          "global_yx": [origin[0] + y, origin[1] + x]}
                         for i, (y, x) in enumerate(PIXELS)],
        "H8_groups_per_pixel": 12,
        "reference_protocol": "Exactly previous adjacent captured inference frame, all three transitions; this is not a cache residency/hit policy simulation.",
        "fixed_three_level_encoder": list(RADII),
        "source_program": SOURCE_PROGRAM,
        "thresholds_by_time": thresholds.tolist(),
        "directions_by_time": directions.tolist(),
        "constant_gates_by_time": [op["constant"] for op in gates],
        "As_row_L1": row_l1.tolist(),
        "safe_radius_definition": "m is inclusive safe integer slack: >= true S-K, false K-1-S; <= true K-S, false S-K-1. Per gate/lane floor(m/L), group minimum across all 80. Ignores signed24 domain clipping; this is the exact maximum admitted by this L1-margin certificate, not all possible certificates.",
        "ordinary_permission": "exact-I24 and already equal output opportunities are separated; no history/compare/create-radius cost charged here.",
        "whole_gate_equality_bound": "gate80_equal is an oracle upper bound for an exact whole-H8 source-output reuse; not an executable detection mechanism.",
        "fixed_radius_vs_unquantized": "Unquantized floor(m/L) tests whether three-level quantization is decisive under the same isotropic H8 criterion.",
        "validation": {"source_S48_and_program_gate_differences": 0,
                       "checked_source_gate_values": checked,
                       "accepted_certificate_false_positives": 0},
        "totals": summarize(rows),
        "transitions": [{"transition": t, **summarize([r for r in rows if r["transition"] == t])}
                        for t in range(1, 4)],
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Recompute and compare with existing opportunity.json; no writes")
    mode.add_argument("--output", type=Path, help="Write regenerated JSON here; default is stdout")
    args = parser.parse_args()
    result = regenerate()
    if args.check:
        expected = json.loads((HERE / "opportunity.json").read_text())
        require(result == expected, "Regenerated table differs from opportunity.json")
        print("PASS: all 72 rows, summaries and metadata match opportunity.json exactly")
    else:
        serialized = json.dumps(result, indent=2) + "\n"
        if args.output is None:
            sys.stdout.write(serialized)
        else:
            args.output.write_text(serialized)


if __name__ == "__main__":
    main()
