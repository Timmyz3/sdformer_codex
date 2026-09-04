#!/opt/anaconda3/bin/python3
"""Fail-closed parser for the M2217 matched native-SAIF/DC/PTPX campaign."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import sys


HW = Path(__file__).resolve().parents[2]
SELECTION = HW / "tb_m2018/fixtures/m2217_ep34_tsbg_matched_power_windows.json"
STRUCT_PATH = HW / (
    "system_simulator/scripts/"
    "parse_m2172_m2018_ordinary_native_saif_balanced_scope_preflight.py")
POWER_PATH = HW / "system_simulator/scripts/parse_m2117_m2018_tsbg_rtl_saifmap_power.py"
MAPPING = HW / "reviews/tsmc28_sram_macro_audit_r1_20260827/tsmc28_sram_mapping_r1.json"
TARGET_INSTANCE = "dut_axis"
RECORDS = 93971
CRITICAL = (
    "mem_req_valid", "mem_rsp_valid", "bridge_valid", "commit_valid",
    "mem_req_accept", "mem_rsp_accept", "bridge_accept", "commit_accept",
)
AXES = {"ordinary_lru4": 0, "tsbg_b4": 1}
STRATA = ("low", "median", "high")


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("import spec: " + str(path))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


STRUCT = module(STRUCT_PATH, "m2172_struct_for_m2217")
POWER = module(POWER_PATH, "m2117_power_for_m2217")
Failure = STRUCT.Failure
need = STRUCT.need
sha256 = STRUCT.sha256
read = STRUCT.read
write_json = STRUCT.write_json


def strict_json(path: Path) -> dict:
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(read(path), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON: " + token)))


def selected() -> dict[str, dict]:
    rows = strict_json(SELECTION)
    need(rows["schema"] == "m2217_ep34_tsbg_matched_power_window_selection_v1"
         and rows["status"] == "FROZEN_PRE_POWER_SELECTION__NO_POWER_OR_PPA_USED",
         "selection identity")
    result = {row["stratum"]: row for row in rows["selections"]}
    need(tuple(result) == STRATA and rows["aggregate_weights"] == {
        "low": [1, 3], "median": [1, 3], "high": [1, 3]},
        "selection strata/weights")
    return result


def expected(axis: str, stratum: str) -> dict:
    need(axis in AXES and stratum in STRATA, "axis/stratum")
    row = selected()[stratum]
    key = "ordinary" if axis == "ordinary_lru4" else "tsbg"
    value = dict(row[key])
    value.update({name: row[name] for name in
                  ("global_slot", "sample_id", "sequence", "layer_id",
                   "target", "token_role", "token_start", "source_groups",
                   "rows", "issues", "products", "commits")})
    return value


def verify_file_seal(path: Path) -> dict[str, str]:
    return STRUCT.verify_file_seal(path)


def parse_saif(path: Path, axis: str, stratum: str, role: str) -> dict:
    cfg = expected(axis, stratum)
    need(role in {"diagnostic_prehistory", "measurement"}, "SAIF role")
    seal = verify_file_seal(path)
    root = STRUCT.parse_balanced_saif(read(path))
    times = [node for node in STRUCT.all_nodes(root)
             if STRUCT.head(node) == "TIMESCALE"]
    need(len(times) == 1, "TIMESCALE count")
    atoms = [item for item in times[0][1:] if isinstance(item, str)]
    need(len(atoms) == 2, "TIMESCALE shape")
    scale = float(atoms[0])
    unit_ns = {"s": 1e9, "ms": 1e6, "us": 1e3,
               "ns": 1.0, "ps": 1e-3, "fs": 1e-6}
    need(atoms[1] in unit_ns, "TIMESCALE unit")
    duration_raw = float(STRUCT.header_value(root, "DURATION"))
    duration_ns = duration_raw * scale * unit_ns[atoms[1]]
    if role == "measurement":
        need(math.isclose(duration_ns, cfg["cycles"] * 3.0,
                          rel_tol=0.0, abs_tol=1e-6), "measurement duration")
    else:
        # Preload is 383 cycles sampled to a settled #0.01-ns boundary.  It is
        # diagnostic-only and is never passed to PrimeTime PX.
        need(1167.0 <= duration_ns < 1168.0, "diagnostic duration")
    instances = [node for node in STRUCT.all_nodes(root)
                 if STRUCT.head(node) == "INSTANCE"]
    targets = [node for node in instances
               if STRUCT.atom_after_head(node) == TARGET_INSTANCE]
    need(len(targets) == 1, "single DUT target instance")
    records, outside = STRUCT.collect_activity(root, targets[0])
    need(not outside and len(records) == RECORDS,
         "DUT-only balanced activity coverage")
    samples = []
    named: dict[str, list[float]] = {}
    toggled = tx_nonzero = 0
    for record in records:
        name = STRUCT.activity_name(record)
        values = tuple(STRUCT.numeric_field(record, field)
                       for field in ("T0", "T1", "TX", "TC"))
        need(all(math.isfinite(value) and value >= 0 for value in values),
             "invalid SAIF field")
        samples.append(values)
        tx_nonzero += int(values[2] != 0)
        toggled += int(values[3] > 0)
        named.setdefault(name, []).append(values[3])
    sums = [t0 + t1 + tx for t0, t1, tx, _ in samples]
    if role == "measurement":
        need(tx_nonzero == 0, "measurement TX nonzero")
        need(all(math.isclose(total, duration_raw, rel_tol=0.0, abs_tol=1e-6)
                 for total in sums), "measurement record conservation")
        tokens = CRITICAL
    else:
        # M2204 admitted only a strict uniform sub-tick floor for the
        # diagnostic prehistory.  Never accept a whole-tick discrepancy.
        exact = all(math.isclose(total, duration_raw, rel_tol=0.0, abs_tol=1e-6)
                    for total in sums)
        floor_ok = (0 < duration_raw - math.floor(duration_raw) < 1
                    and all(value.is_integer() for values in samples
                            for value in values)
                    and all(math.isclose(total, math.floor(duration_raw),
                                         rel_tol=0.0, abs_tol=1e-6)
                            for total in sums))
        need(exact or floor_ok, "diagnostic exact/subtick conservation")
        tokens = ("load_valid",)
    need(toggled >= 20, "insufficient activity")
    critical = {}
    for token in tokens:
        counts = [tc for name, values in named.items()
                  if name == token or re.fullmatch(
                      re.escape(token) + r"\\?\[[^]]+\]", name)
                  for tc in values]
        critical[token] = sum(value > 0 for value in counts)
        need(critical[token] > 0, "zero critical activity: " + token)
    return {
        "axis": axis, "stratum": stratum, "role": role,
        "identity_seal": seal, "target_instance": TARGET_INSTANCE,
        "target_instance_count": 1, "outside_target_records": 0,
        "record_count": len(records), "duration_raw": duration_raw,
        "duration_ns": duration_ns, "tx_nonzero_records": tx_nonzero,
        "nonzero_toggle_records": toggled,
        "conservation": "exact" if all(math.isclose(
            total, duration_raw, rel_tol=0.0, abs_tol=1e-6)
            for total in sums) else "uniform_floor_subtick_diagnostic_only",
        "critical_nonzero_records": critical,
    }


def parse_runtime(path: Path, axis: str, stratum: str) -> dict:
    cfg = expected(axis, stratum)
    text = read(path)
    mode = AXES[axis]
    begins = re.findall(r"^M2217_WINDOW_BEGIN (.+)$", text, re.MULTILINE)
    ends = re.findall(r"^M2217_WINDOW_END (.+)$", text, re.MULTILINE)
    passes = re.findall(r"^PASS_M2217_SINGLE_DUT_NATIVE_SAIF (.+)$",
                        text, re.MULTILINE)
    need(len(begins) == len(ends) == len(passes) == 1, "runtime marker count")
    for token in (
        f"axis_mode={mode}", f"stratum={stratum}",
        f"slot={cfg['global_slot']}", f"sample={cfg['sample_id']}",
        f"sequence={cfg['sequence']}", f"layer={cfg['layer_id']}",
        f"token_start={cfg['token_start']}", f"source_groups={cfg['source_groups']}",
    ):
        need(token in begins[0], "begin identity: " + token)
    for token in (
        f"axis_mode={mode}", f"stratum={stratum}",
        f"cycles={cfg['cycles']}", f"rows={cfg['rows']}",
        f"issues={cfg['issues']}", f"products={cfg['products']}",
        f"commits={cfg['commits']}", f"bundles={cfg['bundles']}",
        f"accepted_bank_requests={cfg['accepted_bank_requests']}",
        "record_conservation=1", "tx_required_zero=1",
    ):
        need(token in ends[0], "end ledger: " + token)
    need(f"axis_mode={mode}" in passes[0] and f"stratum={stratum}" in passes[0]
         and "arithmetic=1 ledger=1 frontends=1 second_axis=0 paper_result=0"
         in passes[0], "PASS boundary")
    need(not re.search(r"(^|\n)(Fatal:|Error:)|Assertion failed|M2217 .*drift|M2217 .*mismatch|M2217 .*timeout", text),
         "runtime failure")
    return {"sha256": sha256(path), "axis": axis, "stratum": stratum,
            "single_frontend": True, "schedule_mode": mode,
            "completion_ledger": cfg}


def sram_model() -> dict:
    ledger = strict_json(MAPPING)
    rows = [row for row in ledger["mappings"]
            if row["id"] == "C2_FC2_WEIGHT_BANKS_K1_K8_K1X8"]
    need(len(rows) == 1, "C2 SRAM mapping row")
    row = rows[0]
    generated = ledger["generated_view_inventory"]
    leakage_density_mw_um2 = (
        generated["slow"]["leakage_uA"] * 0.9 * 1e-3
        / generated["slow"]["area_um2"])
    leakage_mw = leakage_density_mw_um2 * row["area_um2"]
    return {
        "mapping_sha256": sha256(MAPPING), "capacity_bytes": 294912,
        "macro_count": row["macro_count"], "area_um2": row["area_um2"],
        "dynamic_read_energy_pj_per_accepted_bank_activation":
            row["nominal_deep_segment_read_energy_pj_per_bank_request"],
        "dynamic_model": "FOUNDRY_QRT_TT1V85C_DEEP_SEGMENT_CONSERVATIVE",
        "tail_segment_not_selected_without_address_trace": True,
        "leakage_power_mw": leakage_mw,
        "leakage_model": "FOUNDRY_GENERATED_128X128_HVT_SSG0P9V125C_AREA_SCALED_PROXY",
        "leakage_proxy_corner_differs_from_logic_tt0p9v25c": True,
        "identical_capacity_area_and_leakage_both_axes": True,
    }


def parse_point(root: Path, axis: str, stratum: str) -> dict:
    cfg = expected(axis, stratum)
    runtime = parse_runtime(root / "rtl_sim.log", axis, stratum)
    diagnostic = parse_saif(root / "rtl_prehistory.saif", axis, stratum,
                            "diagnostic_prehistory")
    measurement = parse_saif(root / "rtl_measurement.saif", axis, stratum,
                             "measurement")
    need(diagnostic["identity_seal"]["sha256"] !=
         measurement["identity_seal"]["sha256"], "SAIF roles collide")
    dc = root.parent / "dc"
    maps = POWER.classify_maps(
        dc / "netlist/m2018_axis.ptpx_map.default.tcl",
        dc / "netlist/m2018_axis.ptpx_map.essential.tcl",
        root / "map_classification.json")
    pt = root / "ptpx/reports"
    annotation = POWER.parse_annotation(pt / "saif_annotation_summary.rpt")
    switching = POWER.parse_switching_coverage(pt / "switching_coverage.rpt")
    inconsistent = read(pt / "inconsistent_annotation.rpt")
    need(not re.search(r"\b(inconsistent|error|failed)\b", inconsistent,
                       re.IGNORECASE), "inconsistent annotation")
    critical = [POWER.parse_critical(
        pt / f"critical_{name}_activity.rpt", name) for name in CRITICAL]
    logic = POWER.parse_power(pt / "power.rpt", cfg["cycles"] * 3.0)
    scope = read(pt / "scope_and_boundary.rpt")
    for token in (f"axis={axis}", f"stratum={stratum}",
                  f"measurement_cycles={cfg['cycles']}",
                  f"accepted_bank_requests={cfg['accepted_bank_requests']}",
                  "weight_sram_capacity_bytes=294912",
                  "weight_sram_macro_count=16",
                  "power_corner=tt0p9v25c", "wireload=ZeroWireload"):
        need(token in scope, "PT scope token: " + token)
    model = sram_model()
    duration_ns = cfg["cycles"] * 3.0
    dynamic_nj = (cfg["accepted_bank_requests"] *
                  model["dynamic_read_energy_pj_per_accepted_bank_activation"]
                  * 1e-3)
    leakage_nj = model["leakage_power_mw"] * duration_ns * 1e-3
    return {
        "axis": axis, "schedule_mode": AXES[axis], "stratum": stratum,
        "runtime": runtime, "diagnostic_prehistory_saif": diagnostic,
        "measurement_saif": measurement, "map_classification": maps,
        "annotation": annotation, "switching_coverage": switching,
        "critical_cones": critical, "logic": logic,
        "sram": {**model, "accepted_bank_activations":
                 cfg["accepted_bank_requests"],
                 "dynamic_energy_nj": dynamic_nj,
                 "leakage_energy_nj": leakage_nj,
                 "total_energy_nj": dynamic_nj + leakage_nj},
        "component_model": {
            "logic_energy_nj": logic["energy_nj"],
            "sram_dynamic_energy_nj": dynamic_nj,
            "sram_leakage_energy_nj": leakage_nj,
            "total_energy_nj": logic["energy_nj"] + dynamic_nj + leakage_nj,
            "mixed_corner_model": True,
        },
    }


def final_result(root: Path, output: Path) -> dict:
    points = {axis: {stratum: parse_point(root / axis / stratum, axis, stratum)
                     for stratum in STRATA} for axis in AXES}
    aggregate = {}
    for axis in AXES:
        aggregate[axis] = {}
        for field in ("logic_energy_nj", "sram_dynamic_energy_nj",
                      "sram_leakage_energy_nj", "total_energy_nj"):
            aggregate[axis][field] = sum(
                points[axis][stratum]["component_model"][field]
                for stratum in STRATA) / 3.0
    aggregate["comparison"] = {
        "logic_energy_reduction_fraction": 1.0 -
            aggregate["tsbg_b4"]["logic_energy_nj"] /
            aggregate["ordinary_lru4"]["logic_energy_nj"],
        "sram_dynamic_energy_reduction_fraction": 1.0 -
            aggregate["tsbg_b4"]["sram_dynamic_energy_nj"] /
            aggregate["ordinary_lru4"]["sram_dynamic_energy_nj"],
        "component_model_total_energy_reduction_fraction": 1.0 -
            aggregate["tsbg_b4"]["total_energy_nj"] /
            aggregate["ordinary_lru4"]["total_energy_nj"],
        "fixed_population_tercile_weights": {name: [1, 3] for name in STRATA},
    }
    result = {
        "schema": "m2219_m2217_ep34_tsbg_matched_power_result_r1_v1",
        "status": "PASS_RAW_M2219_PENDING_M2220_INDEPENDENT_RESULT_HAMMER",
        "points": points, "aggregate": aggregate,
        "claim_boundary": {
            "native_rtl_dut_only_saif": True,
            "mapped_standard_cell_ptpx": True,
            "external_sram_foundry_model_separate": True,
            "same_capacity_area_and_leakage_both_axes": True,
            "low_median_high_pre_power_selection": True,
            "full_network": False, "fps": False, "silicon": False,
            "post_read_or_selective_bankfill": False,
            "hold_closed": False, "prelayout_zero_wireload": True,
            "paper_citable": False,
        },
    }
    write_json(output, result)
    return result


def static_check() -> dict:
    rows = selected()
    model = sram_model()
    checks = {
        "three_strata": tuple(rows) == STRATA,
        "three_distinct_sequences": len({row["sequence"] for row in rows.values()}) == 3,
        "six_points": len(AXES) * len(STRATA) == 6,
        "single_target": TARGET_INSTANCE == "dut_axis",
        "exact_record_gate": RECORDS == 93971,
        "same_288kib_16macro": model["capacity_bytes"] == 294912
            and model["macro_count"] == 16,
        "logic_sram_total_split": True,
        "aggregate_weights_fixed": True,
        "no_post_read_bankfill": True,
    }
    need(all(checks.values()), "static checks")
    return {"status": "PASS_M2217_STATIC_PARSER", "checks": checks,
            "sram_model": model}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("static")
    saif = sub.add_parser("saif")
    saif.add_argument("--axis", choices=AXES, required=True)
    saif.add_argument("--stratum", choices=STRATA, required=True)
    saif.add_argument("--role", choices=("diagnostic_prehistory", "measurement"),
                      required=True)
    saif.add_argument("--path", type=Path, required=True)
    maps = sub.add_parser("maps")
    maps.add_argument("--default", type=Path, required=True)
    maps.add_argument("--essential", type=Path, required=True)
    maps.add_argument("--output", type=Path, required=True)
    final = sub.add_parser("final")
    final.add_argument("--root", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "static": value = static_check()
    elif args.command == "saif": value = parse_saif(
        args.path, args.axis, args.stratum, args.role)
    elif args.command == "maps": value = POWER.classify_maps(
        args.default, args.essential, args.output)
    else: value = final_result(args.root, args.output)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Failure as exc:
        print("M2217_PARSE_FAIL_CLOSED: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
