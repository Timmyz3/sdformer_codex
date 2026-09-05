#!/usr/bin/env python3
"""Summarize six measured PTPX points, with the actual activity/energy scope.

No run authorization or recursive sealing. Missing results stay missing.
"""
import argparse
import json
from pathlib import Path
import re

HW = Path(__file__).resolve().parents[2]
FLOAT = r"[-+\d.eE]+"


def saif_duration_ns(path):
    text = path.read_text()
    duration = float(re.search(r"\(DURATION\s+([\d.]+)\)", text).group(1))
    magnitude, unit = re.search(r"\(TIMESCALE\s+([\d.]+)\s+(\w+)\)", text).groups()
    return duration * float(magnitude) * {"ps": .001, "ns": 1, "us": 1000}[unit]


def annotation_counts(path):
    # report includes toggle-rate and probability sections. Check both.
    lines = re.findall(r"^ Nets\s+(.+)$", path.read_text(), re.M)
    counts = []
    for line in lines:
        values = [int(x) for x in re.findall(r"(\d+)\([\d.]+%\)", line)]
        total = int(line.split()[-1])
        if len(values) != 10 or sum(values[:3]) != total:
            raise ValueError("Input/state activity is not all mapped from file: " + str(path))
        counts.append(total)
    if len(counts) != 2 or counts[0] != counts[1]:
        raise ValueError("Missing activity/probability section: " + str(path))
    return counts[0]


def read_power(path):
    text = path.read_text()
    result = {}
    for key, label in (("total_mw", "Total Power"), ("internal_mw", "Cell Internal Power"),
                       ("switching_mw", "Net Switching Power"), ("leakage_mw", "Cell Leakage Power")):
        result[key] = float(re.search(re.escape(label) + r"\s*=\s*(" + FLOAT + r")", text).group(1))
    clock = re.search(r"^clock_network\s+(" + FLOAT + r")", text, re.M)
    result["clock_pin_internal_mw"] = float(clock.group(1))
    result["nonclock_dynamic_mw"] = result["internal_mw"] + result["switching_mw"] - result["clock_pin_internal_mw"]
    return result


def constant_ramp_warnings(report, netlist):
    warnings = re.findall(r"ramp ([\d.]+) at pin '([^']+)' of cell '([^']+)'", report.read_text())
    cells = {cell for _,_,cell in warnings}
    input_nets, constants = {}, set()
    # Parse only affected instances and tie-cell outputs, not arbitrary Verilog.
    for statement in netlist.read_text().split(";"):
        match = re.search(r"\b(\w+)\s+(U\d+)\s*\((.*)\)\s*$", statement, re.S)
        if not match:
            continue
        kind, name, pins = match.groups()
        ports = dict(re.findall(r"\.(\w+)\(\s*([^\s)]+)\s*\)", pins))
        if kind.startswith(("TIEH", "TIEL")):
            constants.update(ports.values())
        if name in cells:
            input_nets[name] = ports
    for ramp, pin, cell in warnings:
        if float(ramp) != 0 or input_nets.get(cell, {}).get(pin) not in constants:
            raise ValueError("A nonconstant power-table extrapolation needs review: " + cell)
    return len(warnings)


def summarize(root):
    selections = json.loads((HW / "tb_m2018/fixtures/m2217_ep34_tsbg_matched_power_windows.json").read_text())["selections"]
    rows = []
    for selection in selections:
        for axis, identity in (("ordinary_lru4", "ordinary"), ("tsbg_b4", "tsbg")):
            window = selection["stratum"]
            point = root / axis / window
            if not (point / "COMPLETE.txt").is_file():
                raise ValueError("PTPX result pending: " + str(point))
            log = (point / "ptpx.log").read_text()
            if re.search(r"^Error:", log, re.M):
                raise ValueError("PTPX tool error: " + str(point))
            power = read_power(point / "power.rpt")
            state_count = annotation_counts(point / "sequential_sources_before.rpt")
            input_count = annotation_counts(point / "primary_inputs_sources_before.rpt")
            dc = (HW / "results/m2242_tsbg_power_continue_20260905/ordinary_lru4/dc"
                  if axis == "ordinary_lru4" else root / "tsbg_b4/dc")
            constant_warnings = constant_ramp_warnings(point / "check_power.rpt",
                dc / "netlist/m2018_axis_mapped.v")
            if "No nets found with activity info from UNINITIALIZED" not in (point / "activity_no_switching_activity.rpt").read_text():
                raise ValueError("Uninitialized activity remains")
            cycles = selection[identity]["cycles"]
            saif = HW / "results/m2242_tsbg_power_continue_20260905" / axis / window / "rtl_measurement.saif"
            state_saif = HW / "results/m2247_state_probe_windowed" / axis / f"{window}_state.saif"
            if saif_duration_ns(saif) != cycles * 3 or saif_duration_ns(state_saif) != cycles * 3:
                raise ValueError("Cycle/SAIF measurement window mismatch")
            power.update(axis=axis, window=window, slot=selection["global_slot"],
                reuse_density=selection["selected_density_fraction"][0] / selection["selected_density_fraction"][1],
                cycles=cycles, duration_ns=cycles*3,
                accepted_bank_reads=selection[identity]["accepted_bank_requests"],
                # mW * ns = pJ; /1000 = nJ.
                logic_energy_nj=power["total_mw"] * cycles * 3 / 1000,
                annotated_sequential_nets=state_count, annotated_primary_input_nets=input_count,
                constant_tie_input_ramp_warnings=constant_warnings)
            rows.append(power)
    comparisons = []
    for selection in selections:
        base, candidate = [r for r in rows if r["window"] == selection["stratum"]]
        comparisons.append(dict(window=selection["stratum"],
            cycle_speedup=base["cycles"]/candidate["cycles"],
            bank_read_reduction=1-candidate["accepted_bank_reads"]/base["accepted_bank_reads"],
            logic_energy_reduction=1-candidate["logic_energy_nj"]/base["logic_energy_nj"],
            nonclock_dynamic_energy_reduction=1-(candidate["nonclock_dynamic_mw"]*candidate["cycles"])/(
                base["nonclock_dynamic_mw"]*base["cycles"]),
            average_power_reduction=1-candidate["total_mw"]/base["total_mw"]))
    return dict(status="SIX_MATCHED_PRELAYOUT_POINTS", rows=rows, comparisons=comparisons,
        scope="Three preselected reuse-density windows; one workload per tercile, not a population energy estimate",
        method="RTL-SAIF mapped/propagated PrimeTime PX, TT 0.9V 25C, 3ns, ideal clock 0.1ns slew, ZeroWireload",
        caveats=["No CTS/SPEF, no SRAM bank power in this logic-only number",
            "Deterministic verification INT8 weights, not quantized ep34 FC weights",
            "Ungated mapped baseline and candidate; clock gating has not been compared",
            "Hold repair and post-repair equivalence remain separate work"],
        system_energy_or_speedup=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=HW / "results/m2248_matched_power")
    args = ap.parse_args()
    result = summarize(args.root)
    (args.root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
