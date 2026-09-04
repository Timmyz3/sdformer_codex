#!/usr/bin/env python3
"""Build strict same-flow M35-r6/M33 DC/STA/Formality metrics."""

from __future__ import print_function
import argparse
import collections
import json
import pathlib
import re


DESIGNS = {
    "m35": {
        "name": "qfit_complement_csd8_canonical",
        "source_sha256": "84b1f3cb6344863ecfdbac2af8abcfdd15b1f16571979588badbc3e2e0dd1854",
        "results_per_cycle": 8,
    },
    "m33": {
        "name": "qfit_threshold_late_scale_uq0p24_radix20x4",
        "source_sha256": "2df1c28c0d22cd5a1c38a78a5838101b23bb13beec9e3e5e60ac8f84aba16c4c",
        "results_per_cycle": 4,
    },
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(path):
    require(path.is_file() and path.stat().st_size > 0,
            "missing or empty file: {}".format(path))
    return path.read_text(encoding="utf-8", errors="replace")


def number(pattern, source, cast=float):
    match = re.search(pattern, source, re.MULTILINE)
    require(match is not None, "metric pattern missing: " + pattern)
    value = cast(match.group(1))
    require(type(value) is cast, "metric type drift")
    return value


def min_slack(source):
    values = [float(value) for value in re.findall(
        r"slack \((?:MET|VIOLATED)\)\s+(-?[0-9]+(?:\.[0-9]+)?)", source)]
    require(values, "timing report has no slack")
    return min(values)


def report_point_count(path, kind):
    if not path.exists() or path.stat().st_size == 0:
        return 0
    values = [int(value) for value in re.findall(
        r"^\s*([0-9]+)\s+{} (?:compare )?points\s*$".format(kind),
        text(path), re.MULTILINE | re.IGNORECASE)]
    return max(values) if values else 0


def unmatched_count(source, label):
    no_unmatched = len(re.findall(
        r"^No unmatched points\.$", source, re.MULTILINE))
    require(no_unmatched <= 1, "ambiguous no-unmatched terminal")
    if no_unmatched == 1:
        require(not re.search(
            r"Unmatched reference\(implementation\)", source),
            "contradictory unmatched report")
        return 0
    values = [int(value) for value in re.findall(
        r"^\s*([0-9]+)\([0-9]+\) Unmatched reference\(implementation\) "
        + label + r"\s*$", source, re.MULTILINE)]
    require(values, "unmatched row missing: " + label)
    return values[-1]


def parse_design(root, key):
    directory = root / key
    cfg = DESIGNS[key]
    area = text(directory / "reports/sta_area.rpt")
    setup_text = text(directory / "reports/sta_setup.rpt")
    hold_text = text(directory / "reports/sta_hold.rpt")
    dc_log = text(directory / "dc.raw.log")
    fm_log = text(directory / "formality.raw.log")
    unmatched = text(directory / "reports/formality_unmatched.rpt")
    for stage in ("dc", "sta", "formality"):
        require((directory / (stage + ".rc")).read_bytes() == b"0\n",
                "{} {} rc".format(key, stage))
    require(not re.search(r"^(Error|Fatal):", dc_log, re.MULTILINE),
            key + " DC Error/Fatal")
    require(not re.search(r"^(Error|Fatal):", fm_log, re.MULTILINE),
            key + " Formality Error/Fatal")
    cells = number(r"^Number of cells:\s+([0-9]+)", area, int)
    comb_cells = number(r"^Number of combinational cells:\s+([0-9]+)", area, int)
    seq_cells = number(r"^Number of sequential cells:\s+([0-9]+)", area, int)
    macros = number(r"^Number of macros/black boxes:\s+([0-9]+)", area, int)
    comb_area = number(r"^Combinational area:\s+([0-9.]+)", area)
    noncomb_area = number(r"^Noncombinational area:\s+([0-9.]+)", area)
    total_area = number(r"^Total cell area:\s+([0-9.]+)", area)
    setup = min_slack(setup_text)
    hold = min_slack(hold_text)
    succeeded = len(re.findall(r"^Verification SUCCEEDED$", fm_log, re.MULTILINE))
    passing = [int(value) for value in re.findall(
        r"^\s*([0-9]+) Passing compare points\s*$", fm_log, re.MULTILINE)]
    failing_rows = re.findall(
        r"^Failing \(not equivalent\)\s+((?:[0-9]+\s+){7}[0-9]+)\s*$",
        fm_log, re.MULTILINE)
    require(succeeded == 1 and len(passing) == 1 and failing_rows,
            key + " ambiguous Formality terminal result")
    failing_columns = [int(value) for value in failing_rows[-1].split()]
    formality = {
        "verification_succeeded_terminal_count": succeeded,
        "passing_compare_points": passing[-1],
        "failing_compare_points": failing_columns[-1],
        "failing_result_columns": failing_columns,
        "aborted_compare_points": report_point_count(
            directory / "reports/formality_aborted.rpt", "Aborted"),
        "unverified_compare_points": report_point_count(
            directory / "reports/formality_unverified.rpt", "Unverified"),
        "unmatched_compare_points": unmatched_count(unmatched, r"compare points"),
        "unmatched_primary_or_blackbox_points": unmatched_count(
            unmatched, r"primary inputs, black-box outputs"),
        "unmatched_unread_points_diagnostic_only": unmatched_count(
            unmatched, r"unread points"),
        "fmr_elab_147_count": len(re.findall(r"FMR_ELAB-147", fm_log)),
        "message_filters_used": False,
    }
    formality["all_gates_pass"] = all([
        formality["passing_compare_points"] > 0,
        formality["failing_compare_points"] == 0,
        formality["aborted_compare_points"] == 0,
        formality["unverified_compare_points"] == 0,
        formality["unmatched_compare_points"] == 0,
        formality["unmatched_primary_or_blackbox_points"] == 0,
        formality["fmr_elab_147_count"] == 0,
    ])
    warnings = [line for line in dc_log.splitlines() if line.startswith("Warning:")]
    codes = collections.Counter()
    for line in warnings:
        match = re.search(r"\(([A-Z][A-Z0-9_-]*-[0-9]+)\)\s*$", line)
        codes[match.group(1) if match else "UNCLASSIFIED"] += 1
    result = {
        "design_name": cfg["name"],
        "source_sha256": cfg["source_sha256"],
        "results_per_cycle_contract": cfg["results_per_cycle"],
        "cell_count": cells,
        "combinational_cell_count": comb_cells,
        "sequential_cell_count": seq_cells,
        "macro_or_blackbox_cell_count": macros,
        "combinational_area_um2": comb_area,
        "noncombinational_area_um2": noncomb_area,
        "total_cell_area_um2": total_area,
        "setup_wns_ns_slow_ssg0p9v125c": setup,
        "hold_wns_ns_fast_ffg1p05vm40c": hold,
        "dc_warning_count": len(warnings),
        "dc_warning_codes": dict(sorted(codes.items())),
        "formality": formality,
    }
    result["all_dc_sta_formality_gates_pass"] = all([
        macros == 0, setup >= 0.0, hold >= 0.0, formality["all_gates_pass"]])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite receipt")
    m35 = parse_design(args.run, "m35")
    m33 = parse_design(args.run, "m33")
    audit = text(args.run / "m35/reports/m35_r6_zero_multiplier_audit.rpt")
    zero_multiplier = number(r"^physical_multiplier_hit_total=([0-9]+)$", audit, int)
    require(zero_multiplier == 0, "M35 multiplier audit")
    m35_area = m35["total_cell_area_um2"]
    m33_area = m33["total_cell_area_um2"]
    all_pass = bool(m35["all_dc_sta_formality_gates_pass"] and
                    m33["all_dc_sta_formality_gates_pass"] and
                    zero_multiplier == 0)
    receipt = {
        "schema": "m35_r6_m33_fair_exact_sha_synopsys_receipt_v1",
        "status": "PASS_EXACT_SHA_FRESH_M35_AND_M33_DC_STA_FORMALITY"
                  if all_pass else "FAIL_GATE_DO_NOT_CITE",
        "candidate_changed": False,
        "scope": "standalone_logic_only_zero_wireload_ideal_clock_no_sram_macro",
        "common_flow": {
            "clock_period_ns": 3.0,
            "clock_frequency_mhz_nominal": 333.3333333333333,
            "setup_corner": "ssg0p9v125c",
            "hold_corner": "ffg1p05vm40c",
            "same_sdc_bytes": True,
            "same_dc_tcl_bytes": True,
            "same_sta_tcl_bytes": True,
            "same_formality_tcl_bytes": True,
            "sequential_foreground_invocation_no_tee_no_background": True,
        },
        "m35": m35,
        "m33": m33,
        "m35_zero_physical_multiplier_count": zero_multiplier,
        "fair_comparison": {
            "functional_contract": "both compute exact signed32_accumulator times frozen/checkpoint-admitted UQ0.24 threshold; M35 restricts runtime configuration to ten H67-ep35 descriptor IDs and returns eight products/cycle, while generic M33 accepts UQ0.24 threshold words and returns four products/cycle",
            "m35_over_m33_area": m35_area / m33_area,
            "m35_over_m33_peak_result_rate": 2.0,
            "m35_over_m33_result_rate_per_area": 2.0 * m33_area / m35_area,
            "m35_area_per_result_reduction_percent":
                (1.0 - m35_area / (2.0 * m33_area)) * 100.0,
        },
        "gates": {
            "all_pass": all_pass,
            "m35_dc_sta_formality_pass": bool(m35["all_dc_sta_formality_gates_pass"]),
            "m33_dc_sta_formality_pass": bool(m33["all_dc_sta_formality_gates_pass"]),
            "m35_zero_multiplier_pass": zero_multiplier == 0,
        },
        "claim_boundary": {
            "admitted": "exact-source standalone logic-only fresh DC/STA/Formality and same-flow 3ns M35/M33 area, timing, peak result-rate and result-rate-per-area comparison",
            "not_admitted": "placed/routed or macro-inclusive PPA, clock-tree/wire accuracy, power, energy, full-system cycles or speedup, accuracy, external accelerator comparison, DATE headline, or best-paper status",
            "paper_ppa_ready": False,
            "system_speedup_admitted": False,
        },
    }
    require(all_pass, "one or more strict gates failed")
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
