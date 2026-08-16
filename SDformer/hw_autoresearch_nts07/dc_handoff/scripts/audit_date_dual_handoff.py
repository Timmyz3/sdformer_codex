#!/usr/bin/env python3
"""Fail-closed static audit for the frozen Motion/Local5 DC handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


DESIGNS = {
    "h67_fixed2s_mssb5_dc_top": {
        "filelist": "dc_handoff/filelists/date_motion_2s.f",
        "wrapper": "dc_handoff/rtl/date_motion_dc_tops.sv",
        "must": [
            "module h67_fixed2s_mssb5_dc_top",
            ".QUOTIENT_ENABLE(1'b0)",
            ".MSSB5_SCORE_FRONT(1'b1)",
        ],
        "must_not": [".QUOTIENT_ENABLE(1'b1)"],
        "boundary": "Motion T450 attention row slice; not full encoder",
    },
    "h67_rqtb2s_mssb5_dc_top": {
        "filelist": "dc_handoff/filelists/date_motion_2s.f",
        "wrapper": "dc_handoff/rtl/date_motion_dc_tops.sv",
        "must": [
            "module h67_rqtb2s_mssb5_dc_top",
            ".QUOTIENT_ENABLE(1'b1)",
            ".MSSB5_SCORE_FRONT(1'b1)",
        ],
        "must_not": [".QUOTIENT_ENABLE(1'b0)"],
        "boundary": "Motion T450 attention row slice; not full encoder",
    },
    "local5_unified_out2_dc_top": {
        "filelist": "dc_handoff/filelists/date_local5_out2.f",
        "wrapper": "dc_handoff/rtl/date_local5_dc_top.sv",
        "must": [
            "module local5_unified_out2_dc_top",
            ".OUT_DIM(2)",
            ".RELATION_SCHED_MODE(0)",
            ".BACKEND_KIND(0)",
            ".ACC_BACKEND_KIND(0)",
            ".ACC_MEMORY_IMPL(0)",
            ".ARCH_QSILENT(1'b1)",
            ".ARCH_IDENTK(1'b1)",
            ".ARCH_QSILENT_OVERLAP(1'b1)",
        ],
        "must_not": [".OUT_DIM(32)", ".BACKEND_KIND(1)"],
        "boundary": "Local5 OUT_DIM=2 score-to-Acc32 tile; not encoder",
    },
    "local5_unified_out2_1rw_dc_top": {
        "filelist": "dc_handoff/filelists/date_local5_out2_1rw.f",
        "wrapper": "dc_handoff/rtl/date_local5_1rw_dc_top.sv",
        "must": [
            "module local5_unified_out2_1rw_dc_top",
            ".OUT_DIM(2)",
            ".RELATION_SCHED_MODE(0)",
            ".BACKEND_KIND(0)",
            ".ACC_BACKEND_KIND(1)",
            ".ACC_MEMORY_IMPL(0)",
            ".ARCH_QSILENT(1'b1)",
            ".ARCH_IDENTK(1'b1)",
            ".ARCH_QSILENT_OVERLAP(1'b1)",
        ],
        "must_not": [".OUT_DIM(32)", ".ACC_BACKEND_KIND(0)"],
        "boundary": (
            "Local5 OUT_DIM=2 score-to-Acc32 tile with legal 1RW Acc; "
            "physical-sensitivity baseline, not encoder"
        ),
    },
}
ACTIVITY_CONTRACTS = {
    "motion_fixed": {
        "path": "dc_handoff/runs/motion_fixed_dc_activity_population138_fair/activity_contract.json",
        "design": "h67_fixed2s_mssb5_dc_top",
        "purpose": "paper_power_compute",
        "busy": 112589,
        "measured": 113141,
    },
    "motion_rqtb": {
        "path": "dc_handoff/runs/motion_rqtb_dc_activity_population138_fair/activity_contract.json",
        "design": "h67_rqtb2s_mssb5_dc_top",
        "purpose": "paper_power_compute",
        "busy": 94891,
        "measured": 95443,
    },
    "local5_full": {
        "path": "dc_handoff/runs/local5_dc_activity_full_population100/activity_contract.json",
        "design": "local5_unified_out2_dc_top",
        "purpose": "paper_power_with_io",
        "busy": 155791,
        "measured": 349088,
    },
    "local5_1rw_full": {
        "path": "dc_handoff/runs/local5_1rw_activity_population100_full/activity_contract.json",
        "design": "local5_unified_out2_1rw_dc_top",
        "purpose": "paper_power_with_io",
        "busy": 170269,
        "measured": 397024,
    },
}
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    if path.is_file():
        return sha256(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    checks: dict[str, bool] = {}
    hashes: dict[str, str] = {}
    module_texts: dict[str, str] = {}

    handoff_files = [
        "dc_handoff/run_dc.sh",
        "dc_handoff/run_formality.sh",
        "dc_handoff/run_ptsta.sh",
        "dc_handoff/run_ptpx.sh",
        "dc_handoff/run_local5_activity.sh",
        "dc_handoff/run_local5_1rw_activity.sh",
        "dc_handoff/run_motion_activity.sh",
        "dc_handoff/scripts/run_dc.tcl",
        "dc_handoff/scripts/run_formality.tcl",
        "dc_handoff/scripts/run_ptsta.tcl",
        "dc_handoff/scripts/run_ptpx.tcl",
        "dc_handoff/scripts/audit_dc_artifacts.py",
        "dc_handoff/scripts/audit_synopsys_postrun.py",
        "dc_handoff/scripts/audit_saif_manifest.py",
        "dc_handoff/scripts/audit_expected_macro_refs.py",
        "dc_handoff/scripts/write_synopsys_run_manifest.py",
        "dc_handoff/scripts/report_activity_vcd.py",
        "dc_handoff/scripts/make_saif_manifest.py",
        "dc_handoff/scripts/compare_motion_activity_contracts.py",
        "dc_handoff/config/date_dual_constraints.yaml",
        "dc_handoff/config/saif_manifest.example.json",
    ]
    for name in handoff_files:
        path = root / name
        checks[f"handoff:{name}"] = path.is_file() and path.stat().st_size > 0
        if path.is_file():
            hashes[name] = sha256(path)

    ptsta_text = (root / "dc_handoff/run_ptsta.sh").read_text(encoding="utf-8")
    ptpx_text = (root / "dc_handoff/run_ptpx.sh").read_text(encoding="utf-8")
    dc_text = (root / "dc_handoff/run_dc.sh").read_text(encoding="utf-8")
    dc_tcl_text = (root / "dc_handoff/scripts/run_dc.tcl").read_text(encoding="utf-8")
    activity_text = (root / "dc_handoff/scripts/report_activity_vcd.py").read_text(
        encoding="utf-8"
    )
    postrun_text = (root / "dc_handoff/scripts/audit_synopsys_postrun.py").read_text(
        encoding="utf-8"
    )
    checks["spef_requires_explicit_pnr_netlist"] = (
        "读取SPEF时必须通过NETLIST_FILE" in ptsta_text
        and "读取SPEF时必须通过NETLIST_FILE" in ptpx_text
    )
    checks["pt_requires_operating_condition"] = (
        "OPERATING_CONDITION" in ptsta_text and "OPERATING_CONDITION" in ptpx_text
    )
    checks["ptpx_requires_saif_manifest_and_coverage"] = (
        "SAIF_MANIFEST" in ptpx_text and "MIN_SAIF_COVERAGE_PCT" in ptpx_text
    )
    checks["paper_ppa_requires_macro_identity"] = (
        "PPA_ADMISSION" in dc_text and "EXPECTED_MACRO_REFS" in dc_text
    )
    checks["dc_saif_requires_manifest"] = (
        "audit_saif_manifest.py" in dc_text and "SAIF_MANIFEST" in dc_text
    )
    checks["dc_without_saif_does_not_report_power"] = (
        "NO_SAIF_POWER_NOT_RUN" in dc_tcl_text
        and 'if {[info exists ::env(SAIF_FILE)]' in dc_tcl_text
    )
    checks["ptpx_requires_paper_power_population"] = (
        "--require-paper-power-eligible" in ptpx_text
    )
    checks["activity_contract_checks_vcd_scope"] = (
        "vcd_metadata" in activity_text and "strip_path_in_vcd" in activity_text
    )
    checks["activity_contract_separates_busy_and_measured_cycles"] = (
        '"busy_cycles"' in activity_text and '"measured_cycles"' in activity_text
    )
    checks["activity_contract_locks_frozen_paper_populations"] = all(
        token in activity_text
        for token in ("112589", "94891", "62100", "34099", "28001", "155791")
    )
    checks["activity_contract_checks_vcd_time_axis"] = all(
        token in activity_text
        for token in ("vcd_active_duration", "active_intervals", "clock_period_ps")
    )
    saif_manifest_text = (
        root / "dc_handoff/scripts/audit_saif_manifest.py"
    ).read_text(encoding="utf-8")
    checks["saif_manifest_rehashes_source_identity"] = all(
        token in saif_manifest_text
        for token in ("sha256(source_vcd)", "sha256_tree(trace_root)",
                      "sha256(activity_contract)")
    )
    checks["ptpx_requires_explicit_zero_unannotated_objects"] = (
        "unannotated_object_count" in postrun_text
        and 'unannotated == 0' in postrun_text
    )

    sdc = root / "dc_handoff/constraints/date_dual_core.sdc"
    checks["sdc_exists"] = sdc.is_file()
    if sdc.is_file():
        sdc_text = sdc.read_text(encoding="utf-8")
        checks["sdc_clk_core"] = "[get_ports clk_core]" in sdc_text
        checks["sdc_period_override"] = "CLOCK_PERIOD_NS" in sdc_text
        hashes[str(sdc.relative_to(root))] = sha256(sdc)

    for design, cfg in DESIGNS.items():
        filelist = root / cfg["filelist"]
        wrapper = root / cfg["wrapper"]
        checks[f"{design}:filelist"] = filelist.is_file()
        checks[f"{design}:wrapper"] = wrapper.is_file()
        if not filelist.is_file() or not wrapper.is_file():
            continue
        entries = [
            line.strip() for line in filelist.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        missing = [entry for entry in entries if not (root / entry).is_file()]
        checks[f"{design}:all_rtl_files"] = not missing
        wrapper_text = wrapper.read_text(encoding="utf-8")
        module_marker = f"module {design}"
        module_start = wrapper_text.find(module_marker)
        module_end = wrapper_text.find("endmodule", module_start)
        checks[f"{design}:module_scope"] = module_start >= 0 and module_end >= 0
        module_text = (
            wrapper_text[module_start:module_end]
            if module_start >= 0 and module_end >= 0
            else ""
        )
        module_texts[design] = module_text
        for token in cfg["must"]:
            checks[f"{design}:token:{token}"] = token in module_text
        for token in cfg["must_not"]:
            checks[f"{design}:forbidden:{token}"] = token not in module_text
        hashes[str(filelist.relative_to(root))] = sha256(filelist)
        hashes[str(wrapper.relative_to(root))] = sha256(wrapper)
        for entry in entries:
            path = root / entry
            if path.is_file():
                hashes[str(path.relative_to(root))] = sha256(path)

    fixed = module_texts.get("h67_fixed2s_mssb5_dc_top", "")
    rqtb = module_texts.get("h67_rqtb2s_mssb5_dc_top", "")
    if fixed and rqtb:
        def normalize_motion(text: str) -> str:
            text = re.sub(r"module h67_(?:fixed|rqtb)2s_mssb5_dc_top", "module MOTION_TOP", text)
            text = re.sub(r"\.QUOTIENT_ENABLE\(1'b[01]\)", ".QUOTIENT_ENABLE(Q)", text)
            return text

        checks["motion_wrappers_only_quotient_differs"] = (
            normalize_motion(fixed) == normalize_motion(rqtb)
        )

    docs359 = root / "docs/359_DATE终局冻结_20260813.md"
    checks["docs359_frozen_sha256"] = (
        docs359.is_file() and sha256(docs359) == DOCS359_SHA256
    )
    if docs359.is_file():
        hashes[str(docs359.relative_to(root))] = sha256(docs359)

    for label, cfg in ACTIVITY_CONTRACTS.items():
        path = root / cfg["path"]
        contract = {}
        if path.is_file():
            contract = json.loads(path.read_text(encoding="utf-8"))
            hashes[str(path.relative_to(root))] = sha256(path)
        source_vcd = Path(str(contract.get("source_vcd", "")))
        trace_root = Path(str(contract.get("trace_root", "")))
        if not source_vcd.is_absolute():
            source_vcd = root / source_vcd
        if not trace_root.is_absolute():
            trace_root = root / trace_root
        checks[f"activity:{label}:contract"] = bool(contract) and all(
            (
                contract.get("status") == "PASS",
                contract.get("design_name") == cfg["design"],
                contract.get("activity_purpose") == cfg["purpose"],
                contract.get("paper_power_eligible") is True,
                contract.get("busy_cycles") == cfg["busy"],
                contract.get("measured_cycles") == cfg["measured"],
                contract.get("vcd_active_intervals") == 1,
            )
        )
        checks[f"activity:{label}:source_vcd_identity"] = (
            source_vcd.is_file()
            and contract.get("source_vcd_sha256") == sha256(source_vcd)
        )
        checks[f"activity:{label}:trace_identity"] = (
            trace_root.exists()
            and contract.get("trace_sha256") == sha256_tree(trace_root)
        )

    motion_pair = root / "dc_handoff/runs/motion_fair_activity_pair_20260814.json"
    pair_data = {}
    if motion_pair.is_file():
        pair_data = json.loads(motion_pair.read_text(encoding="utf-8"))
        hashes[str(motion_pair.relative_to(root))] = sha256(motion_pair)
    checks["activity:motion_pair_all_checks"] = (
        pair_data.get("status") == "PASS"
        and bool(pair_data.get("checks"))
        and all(pair_data["checks"].values())
    )

    passed = all(checks.values())
    result = {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "design_boundaries": {
            design: cfg["boundary"] for design, cfg in DESIGNS.items()
        },
        "evidence_label": "handoff-ready only; no DC/STA/PPA claim",
        "sha256": dict(sorted(hashes.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
