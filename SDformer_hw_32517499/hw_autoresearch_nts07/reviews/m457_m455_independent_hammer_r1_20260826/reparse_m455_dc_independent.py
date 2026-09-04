#!/usr/bin/env python3
"""Independently reparse M455 and its sealed M433 reference without its receipt."""

import hashlib
import json
import re
import sys
from decimal import Decimal, getcontext
from pathlib import Path


getcontext().prec = 50
HW = Path(__file__).resolve().parents[2]
M455 = HW / "dc_handoff/runs/m455_m451_vs_m433_standalone_dc_3p000ns_r1_20260826"
M439 = HW / "dc_handoff/runs/m439_serial_vs_dualcoread_adapters_dc_3p000ns_r1_20260826"
M449 = HW / "results/m449_m447_independent_hammer_r1_20260826"

FROZEN_SHA256 = {
    "contracts/m455_m451_vs_m433_standalone_dc_contract_r1_20260826.json": "ca97c75c53299a60325be2fb16f3d411b3fe675a588a59eea9f28a48356e9444",
    "dc_handoff/scripts/run_dc_m455_m451_vs_m433_standalone_exact_sha.sh": "2935b514fe4c813d856980a3af146627d298c7095283031721b98df58116cf84",
    "dc_handoff/runs/m455_m451_vs_m433_standalone_dc_3p000ns_r1_20260826/evidence_manifest.sha256": "f58b2a12feba9333a841199346161c2ee092390221dee40fcb83b10016ac2758",
    "dc_handoff/runs/m455_m451_vs_m433_standalone_dc_3p000ns_r1_20260826/evidence_manifest.seal.sha256": "42014d7472e55a10cdfaf4ec0399651f16adb60276a8e7aa1234f9656f85d244",
    "rtl_m451/m451_exact_k1_fused_pwp_correction_adapter.sv": "b09172c5ca5c6fccddad0ccd19f37ffaae032cfe26350297f9ffcb3df65e2307",
    "results/m451_exact_k1_fused_pwp_correction_directed_vcs_r1_20260826/RUN_MANIFEST.sha256": "ea2216fc927312422581ae834cba66f2c424f491dcf7078a60a9af02f1b73743",
    "results/m451_exact_k1_fused_pwp_correction_directed_vcs_r1_20260826/RUN_MANIFEST.seal.sha256": "9b6fad46290411d90e9d28e40202981b64d8ccb178f607f23370ce213c6fd3e3",
    "results/m452_m451_independent_hammer_r1_20260826/RUN_MANIFEST.sha256": "f5718f23655e5b92f8a2d7aee34e7f0cdd9c4d52fd25d0cc2dae6efc8a614408",
    "results/m452_m451_independent_hammer_r1_20260826/RUN_MANIFEST.seal.sha256": "13873fcd25dbe9b74bfd8095f2a13ac10115f037f052aa7b32b9b8d2ae16598e",
    "contracts/m439_m405_serial_vs_m433_dualcoread_adapter_dc_contract_r1_20260826.json": "f59a58be539a734b04fbbb8f4de9cdb4f7f33661cf02fd4ed49938ea1782698a",
    "dc_handoff/scripts/run_dc_m439_serial_vs_dualcoread_adapters_exact_sha.sh": "2eb83a48cbd2876154579e4dd6034cd0eb19a161e5ed0a7e47dfcd8f70d2be28",
    "dc_handoff/runs/m439_serial_vs_dualcoread_adapters_dc_3p000ns_r1_20260826/evidence_manifest.sha256": "2564ab5305115a5da7af98d47d32414430b6548c2dfcd36d3356b3d90773423b",
    "dc_handoff/runs/m439_serial_vs_dualcoread_adapters_dc_3p000ns_r1_20260826/evidence_manifest.seal.sha256": "98696f3bd166172aa294d2d24fb5d16f6fa7211a8da939fb99c035506d3eaa1a",
    "rtl_m433/m433_exact_dualbank_coread_pwp_adapter.sv": "75ad462a584ea46bd1043bb6a21d82b5687e7ab392995b28d707c248a5f96046",
    "dc_handoff/scripts/run_dc_m362_m356_failclosed_q128_matcher_exact_sha.tcl": "b4da812ed639e48a69f04c45d1393edcc46d3f39a638db450b375a0352dc995f",
    "dc_handoff/constraints/date_m439_pwp_adapter_3ns.sdc": "565f486c7537484b0b6c11db7e53e4afc6962f2f73827a30764c3fe70bf3bb29",
    "results/m447_m430_delta_domain_correction_fold_dse_r1_20260826/SHA256SUMS": "63e57fc0fa59a4779177756411a58352fe26b61c854d64843eb6cf71146201a1",
    "results/m447_m430_delta_domain_correction_fold_dse_r1_20260826/SHA256SUMS.seal.sha256": "1be3dc212f8c256ea8a6fc32766a3b1bfe0c0e5d2ff8f4846178f475a6668da1",
    "results/m447_m430_delta_domain_correction_fold_dse_r1_20260826/m447_m430_delta_domain_correction_fold_dse_r1.json": "d00a98b98309a77e2461427077d01eda171aa13fdaefedb701a5c07b5a99cc3d",
    "results/m449_m447_independent_hammer_r1_20260826/SHA256SUMS": "40831b3df313f90aae27124f9058b1332ebaaf42f29af1ff59f5ba438443da0f",
    "results/m449_m447_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256": "a7fe306a91a1efc7b05340fdfa4bfd859e9f7aa830db01e022b046e1fb14b96a",
    "docs/359_DATE终局冻结_20260813.md": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

TOOL_SHA256 = {
    Path("/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"): "23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2",
    Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"): "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af",
    Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"): "a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(manifest: Path) -> int:
    checked = 0
    for raw in manifest.read_text().splitlines():
        if not raw.strip():
            continue
        expected, name = raw.split(maxsplit=1)
        name = name.lstrip(" *")
        target = Path(name) if Path(name).is_absolute() else manifest.parent / name
        actual = sha256(target.resolve())
        if actual != expected:
            raise AssertionError(f"manifest mismatch: {target}: {actual} != {expected}")
        checked += 1
    return checked


def one(pattern: str, text: str, cast=float):
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise AssertionError(f"missing pattern {pattern!r}")
    return cast(match.group(1))


def parse_point(point: Path, top: str, role: str) -> dict:
    reports = point / "reports"
    area_text = (reports / "area.rpt").read_text(errors="replace")
    qor_text = (reports / "qor.rpt").read_text(errors="replace")
    setup_text = (reports / "timing_setup.rpt").read_text(errors="replace")
    hold_text = (reports / "timing_hold.rpt").read_text(errors="replace")
    constraint_text = (reports / "constraint_violators.rpt").read_text(errors="replace")
    design_text = (reports / "check_design_postcompile.rpt").read_text(errors="replace")
    timing_check_text = (reports / "check_timing_postcompile.rpt").read_text(errors="replace")
    hold_guard_text = (reports / "hold_guard_contract.rpt").read_text(errors="replace")
    log_text = (point / "dc.log").read_text(errors="replace")
    netlist_text = (point / f"netlist/{top}_mapped.v").read_text(errors="replace")
    mapped_sdc = (point / f"netlist/{top}_mapped.sdc").read_text(errors="replace")

    setup_slacks = [Decimal(x) for x in re.findall(r"slack \(MET\)\s+([-0-9.]+)", setup_text)]
    hold_slacks = [Decimal(x) for x in re.findall(r"slack \(MET\)\s+([-0-9.]+)", hold_text)]
    assert setup_slacks and hold_slacks
    assert "slack (VIOLATED)" not in setup_text + hold_text
    assert constraint_text.count("This design has no violated constraints.") == 5
    assert "Warning:" not in timing_check_text and "Error:" not in timing_check_text
    assert not re.search(r"unresolved reference|inferred latch|timing loop", design_text, re.I)
    if role == "candidate_m451":
        assert design_text.strip() == "1"
        assert len(re.findall(r"signed to unsigned assignment occurs\. \(VER-318\)", log_text)) == 6
    else:
        assert design_text.count("LINT-31") == 2
        assert "busy' is connected directly to output port 'debug_output_full" in design_text
        assert "VER-318" not in log_text
    assert (point / "dc.rc").read_text().strip() == "0"
    assert "Thank you..." in log_text
    assert not re.search(r"^(?:Error|Fatal):|ELAB-312|TIM-209|OPT-150", log_text, re.MULTILINE)
    assert log_text.count("contains 1 high-fanout nets") == 5
    assert "tcbn28hpcplusbwp35p140ssg0p9v125c.db" in log_text
    assert "tcbn28hpcplusbwp35p140ffg1p05vm40c.db" in log_text
    assert "set_min_library $lib_db -min_version $min_lib_db" in log_text
    assert "Using operating conditions 'ssg0p9v125c'" in log_text
    assert "set_wire_load_model -name ZeroWireload" in log_text
    assert netlist_text.count("\nmodule ") == 1 and netlist_text.count("\nendmodule") == 1
    assert not re.search(r"^\s+[A-Z0-9_]*(?:LAT|LATCH)[A-Z0-9_]*\s+\w+\s*\(", netlist_text, re.MULTILINE)
    dff_count = len(re.findall(r"^\s+DFCNQD1BWP35P140\s+", netlist_text, re.MULTILINE))
    assert one(r"create_clock .* -period ([0-9.]+)", mapped_sdc, Decimal) == Decimal("3")
    assert one(r"set_clock_uncertainty ([0-9.]+)", mapped_sdc, Decimal) == Decimal("0.1")
    assert not re.search(r"^set_(?:input_transition|driving_cell)", mapped_sdc, re.MULTILINE)
    assert "set_false_path   -from [get_ports reset_n]" in mapped_sdc
    assert "synthesis_hold_uncertainty_ns=0.125" in hold_guard_text
    assert "publication_hold_uncertainty_ns=0.100" in hold_guard_text
    assert "additional_hold_guard_ns=0.025" in hold_guard_text
    assert "guard_removed_before_final_reports_and_write_sdc=true" in hold_guard_text

    result = {
        "role": role,
        "top": top,
        "report_date": one(r"^Date\s+:\s+(.+)$", area_text, str),
        "ports": one(r"Number of ports:\s+(\d+)", area_text, int),
        "nets": one(r"Number of nets:\s+(\d+)", area_text, int),
        "cells": one(r"Number of cells:\s+(\d+)", area_text, int),
        "combinational_cells": one(r"Number of combinational cells:\s+(\d+)", area_text, int),
        "sequential_cells": one(r"Number of sequential cells:\s+(\d+)", area_text, int),
        "mapped_dff_instances": dff_count,
        "macros_blackboxes": one(r"Number of macros/black boxes:\s+(\d+)", area_text, int),
        "buf_inv_cells": one(r"Number of buf/inv:\s+(\d+)", area_text, int),
        "references": one(r"Number of references:\s+(\d+)", area_text, int),
        "combinational_area_um2": str(one(r"Combinational area:\s+([0-9.]+)", area_text, Decimal)),
        "noncombinational_area_um2": str(one(r"Noncombinational area:\s+([0-9.]+)", area_text, Decimal)),
        "macro_blackbox_area_um2": str(one(r"Macro/Black Box area:\s+([0-9.]+)", area_text, Decimal)),
        "cell_area_um2": str(one(r"Total cell area:\s+([0-9.]+)", area_text, Decimal)),
        "logic_levels": str(one(r"Levels of Logic:\s+([0-9.]+)", qor_text, Decimal)),
        "critical_path_length_ns": str(one(r"Critical Path Length:\s+([0-9.]+)", qor_text, Decimal)),
        "setup_worst_slack_ns": str(min(setup_slacks)),
        "hold_worst_slack_ns": str(min(hold_slacks)),
        "constraint_report_sections_without_violations": 5,
        "clock_period_ns": "3",
        "clock_uncertainty_ns": "0.1",
        "mapping_hold_uncertainty_ns": "0.125",
        "additional_mapping_hold_guard_ns": "0.025",
        "input_delay_ns": "0.2",
        "output_delay_ns": "0.2",
        "output_load_pf": "0.01",
        "input_transition_or_driving_cell_constraints": 0,
        "wireload": "ZeroWireload",
        "clock_network": "ideal",
        "high_fanout_1000_proxy_warning_occurrences": 5,
        "signed_to_unsigned_VER318_warning_occurrences": 6 if role == "candidate_m451" else 0,
        "no_latch_blackbox_timing_loop": True,
    }
    assert result["sequential_cells"] == result["mapped_dff_instances"]
    assert result["macros_blackboxes"] == 0
    assert Decimal(result["macro_blackbox_area_um2"]) == 0
    return result


def main() -> None:
    frozen_checks = {}
    for relative, expected in FROZEN_SHA256.items():
        actual = sha256(HW / relative)
        assert actual == expected, f"frozen input drift: {relative}: {actual} != {expected}"
        frozen_checks[relative] = actual
    tool_checks = {}
    for path, expected in TOOL_SHA256.items():
        actual = sha256(path)
        assert actual == expected, f"tool/library drift: {path}: {actual} != {expected}"
        tool_checks[str(path)] = actual

    manifest_counts = {
        "m451_inner": verify_manifest(HW / "results/m451_exact_k1_fused_pwp_correction_directed_vcs_r1_20260826/RUN_MANIFEST.sha256"),
        "m451_outer": verify_manifest(HW / "results/m451_exact_k1_fused_pwp_correction_directed_vcs_r1_20260826/RUN_MANIFEST.seal.sha256"),
        "m452_inner": verify_manifest(HW / "results/m452_m451_independent_hammer_r1_20260826/RUN_MANIFEST.sha256"),
        "m452_outer": verify_manifest(HW / "results/m452_m451_independent_hammer_r1_20260826/RUN_MANIFEST.seal.sha256"),
        "m439_inner": verify_manifest(M439 / "evidence_manifest.sha256"),
        "m439_outer": verify_manifest(M439 / "evidence_manifest.seal.sha256"),
        "m455_inner": verify_manifest(M455 / "evidence_manifest.sha256"),
        "m455_outer": verify_manifest(M455 / "evidence_manifest.seal.sha256"),
        "m447_inner": verify_manifest(HW / "results/m447_m430_delta_domain_correction_fold_dse_r1_20260826/SHA256SUMS"),
        "m447_outer": verify_manifest(HW / "results/m447_m430_delta_domain_correction_fold_dse_r1_20260826/SHA256SUMS.seal.sha256"),
        "m449_inner": verify_manifest(M449 / "SHA256SUMS"),
        "m449_outer": verify_manifest(M449 / "SHA256SUMS.seal.sha256"),
    }

    candidate = parse_point(M455 / "candidate", "m451_exact_k1_fused_pwp_correction_adapter", "candidate_m451")
    reference = parse_point(M439 / "dual_coread", "m433_exact_dualbank_coread_pwp_adapter", "reference_m433")
    assert candidate["cell_area_um2"] == "12952.043867"
    assert reference["cell_area_um2"] == "8351.405814"
    assert candidate["cells"] == 12802 and reference["cells"] == 7139
    assert candidate["sequential_cells"] == 1445 and reference["sequential_cells"] == 1348
    assert candidate["logic_levels"] == "42.00" and reference["logic_levels"] == "52.00"
    assert candidate["setup_worst_slack_ns"] == "0.8828"
    assert reference["setup_worst_slack_ns"] == "0.8411"
    assert candidate["hold_worst_slack_ns"] == reference["hold_worst_slack_ns"] == "0.0251"

    m449 = json.loads((M449 / "m449_independent_recomputation.json").read_text())
    points = {point["name"]: point for point in m449["six_points"]}
    separate_cycles = Decimal(points["k1_separate_fold"]["cycles"])
    fused_cycles = Decimal(points["k1_fused_delta_composer"]["cycles"])
    opportunity = separate_cycles / fused_cycles
    assert separate_cycles == Decimal(517041352) and fused_cycles == Decimal(430154216)
    assert abs(opportunity - Decimal("1.2019906646689706")) < Decimal("1e-15")

    candidate_area = Decimal(candidate["cell_area_um2"])
    reference_area = Decimal(reference["cell_area_um2"])
    area_ratio = candidate_area / reference_area
    ff_ratio = Decimal(candidate["sequential_cells"]) / Decimal(reference["sequential_cells"])
    cell_ratio = Decimal(candidate["cells"]) / Decimal(reference["cells"])
    throughput_area = opportunity / area_ratio
    comparison = {
        "m451_to_m433_area_ratio": str(area_ratio),
        "m451_to_m433_area_delta_percent": str((area_ratio - 1) * 100),
        "m451_to_m433_cell_ratio": str(cell_ratio),
        "m451_to_m433_cell_delta_percent": str((cell_ratio - 1) * 100),
        "m451_to_m433_ff_ratio": str(ff_ratio),
        "m451_to_m433_ff_delta_percent": str((ff_ratio - 1) * 100),
        "m451_minus_m433_area_um2": str(candidate_area - reference_area),
        "m451_minus_m433_combinational_area_um2": str(Decimal(candidate["combinational_area_um2"]) - Decimal(reference["combinational_area_um2"])),
        "m451_minus_m433_noncombinational_area_um2": str(Decimal(candidate["noncombinational_area_um2"]) - Decimal(reference["noncombinational_area_um2"])),
        "m451_minus_m433_logic_levels": str(Decimal(candidate["logic_levels"]) - Decimal(reference["logic_levels"])),
        "m451_minus_m433_setup_slack_ns": str(Decimal(candidate["setup_worst_slack_ns"]) - Decimal(reference["setup_worst_slack_ns"])),
        "k1_trace_cycle_opportunity": str(opportunity),
        "standalone_opportunity_throughput_per_area_ratio": str(throughput_area),
        "standalone_opportunity_throughput_per_area_delta_percent": str((throughput_area - 1) * 100),
        "candidate_physical_input_bytes_per_fused_cycle": 256,
        "reference_physical_input_bytes_per_wide_cycle": 160,
        "physical_input_bandwidth_ratio": str(Decimal(256) / Decimal(160)),
    }

    rtl451 = (HW / "rtl_m451/m451_exact_k1_fused_pwp_correction_adapter.sv").read_text()
    rtl433 = (HW / "rtl_m433/m433_exact_dualbank_coread_pwp_adapter.sv").read_text()
    header451 = re.search(r"module\s+m451_exact_k1_fused_pwp_correction_adapter.*?\)\s*\((.*?)\);", rtl451, re.S).group(1)
    header433 = re.search(r"module\s+m433_exact_dualbank_coread_pwp_adapter.*?\)\s*\((.*?)\);", rtl433, re.S).group(1)
    assert not re.search(r"old_psum|address|addr|sram|memory", header451, re.I)
    assert not re.search(r"old_psum|address|addr|sram|memory", header433, re.I)

    output = {
        "schema": "m457_m455_dc_independent_reparse_v1",
        "status": "PASS_RAW_STANDALONE_DC_REPARSE_PERFORMANCE_MAINLINE_NO_GO",
        "source_m455_receipt_consumed": False,
        "tool": "Synopsys Design Compiler V-2023.12-SP3",
        "technology": "TSMC28 HPC+ standard cells",
        "frozen_sha256_checks": frozen_checks,
        "tool_library_sha256_checks": tool_checks,
        "manifest_files_verified": manifest_counts,
        "candidate_m451": candidate,
        "reference_m433": reference,
        "comparison": comparison,
        "flow_fairness": {
            "same_dc_engine_tcl_sha256": True,
            "same_sdc_sha256": True,
            "same_slow_max_library_sha256": True,
            "same_fast_min_library_sha256": True,
            "same_operating_condition": "ssg0p9v125c",
            "same_clock_period_ns": "3",
            "same_zero_wireload": True,
            "same_ideal_clock": True,
            "paired_in_one_new_runner": False,
            "reference_from_older_sealed_m439_run": True,
            "assessment": "Comparable for sealed standalone mechanism-local logic cost because the engine Tcl, SDC, tool, libraries and runtime assumptions match exactly; not a new paired-run or integrated iso-resource comparison.",
        },
        "semantic_resource_boundary": {
            "m451_incremental_signed_preadder_lanes": 96,
            "m451_output_bits_per_lane": 13,
            "m433_output_bits_per_lane": 12,
            "extra_registered_lane_bits_plus_fused_flag": 97,
            "observed_extra_ff": candidate["sequential_cells"] - reference["sequential_cells"],
            "m451_fused_input_physical_bytes": 256,
            "m433_wide_input_physical_bytes": 160,
            "address_generation_in_either_top": False,
            "memory_macros_in_either_top": False,
            "old_psum_accumulator_in_either_top": False,
            "baseline_separate_correction_compute_area_in_m433_denominator": False,
            "remaining_nonfused_correction_path_in_candidate": False,
            "cycle_opportunity_is_rtl_measured": False,
            "cycle_opportunity_is_resource_normalized": False,
        },
        "claim_boundary": {
            "standalone_logic_only_dc": True,
            "standalone_area_and_timing_comparison": True,
            "cycle_opportunity": True,
            "standalone_opportunity_throughput_per_area_diagnostic": True,
            "formality": False,
            "primetime": False,
            "integrated_amortization": False,
            "memory_port_concurrency": False,
            "conv_speedup": False,
            "system_speedup": False,
            "power_or_energy": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
    }
    json.dump(output, sys.stdout, indent=2, sort_keys=True, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
