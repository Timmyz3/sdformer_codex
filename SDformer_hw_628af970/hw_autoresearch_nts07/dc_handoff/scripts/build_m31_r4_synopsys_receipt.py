#!/usr/bin/env python3
"""Build the fail-closed M31-r4 fresh-DC/Formality receipt.

The receipt is deliberately generated only after the report auditors, all live
ledgers, and the self-contained Formality snapshot have been revalidated.  It
does not admit placed/routed PPA, power, energy, or system performance.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from audit_m31_r4_dc_reports import build as rebuild_dc_audit
from audit_m31_r4_formality import build as rebuild_formality_audit


DESIGN = "qfit_atlif_unified_t10_t2_stream_core"
R4_RECEIPT_SHA256 = (
    "bae2f05e74ffa8863195bda9f222c22fc06364ade872e9cf83d3cd4106e5b77d")
R4_ADMISSION_SHA256 = (
    "e8bd1b6452280396a5c8fc83ce79f34d1ae08256f97b469613207418dcfd0ff6")
R1_RECEIPT_RELATIVE = (
    "hw_autoresearch_nts07/contracts/m31_synopsys_receipt_r1_20260822.json")
RECEIPT_SCHEMA = "m31_synopsys_receipt_r2_fresh_r4_v1"
RECEIPT_STATUS = (
    "PASS_M31_R4_FRESH_VCS_DC_STA_STRICT_FORMALITY_EXACT96_"
    "PREMACRO_LOGIC_ONLY_NO_SYSTEM_OR_PAPER_PPA_CLAIM")
DC_SCHEMA = "m31_r4_fresh_dc_report_audit_v1"
DC_STATUS = "PASS_M31_R4_EXACT96_ZERO_WIRE_IDEAL_CLOCK_3NS_LOGIC_ONLY"
FM_SCHEMA = "m31_r4_fresh_formality_audit_v1"
FM_STATUS = "PASS_M31_R4_RTL_TO_FRESH_MAPPED_NETLIST_FORMALITY_STRICT"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path, label):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("missing M31 receipt {}".format(label))
    return path.resolve()


def load_json_no_duplicates(path, label):
    def hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key in M31 {}".format(label))
            result[key] = value
        return result
    return json.loads(require_file(path, label).read_text(encoding="utf-8"),
                      object_pairs_hook=hook)


def require_exact_keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("M31 {} exact schema drift".format(label))


def assert_exact_json(path, rebuilt, label):
    recorded = load_json_no_duplicates(path, label)
    if recorded != rebuilt:
        raise ValueError("M31 {} does not equal a live rebuild".format(label))
    return recorded


def parse_ledger(ledger_path, base_dir, label):
    ledger = require_file(ledger_path, label)
    base = Path(base_dir).resolve()
    entries = []
    canonical_seen = set()
    for line_number, raw in enumerate(
            ledger.read_text(encoding="utf-8").splitlines(), 1):
        if not raw:
            continue
        match = re.match(r"^([0-9a-f]{64})  ([^\0]+)$", raw)
        if not match:
            raise ValueError("malformed M31 {} line {}".format(
                label, line_number))
        expected, raw_path = match.groups()
        if raw_path.startswith("*"):
            raw_path = raw_path[1:]
        path = Path(raw_path)
        if not path.is_absolute():
            path = base / path
        path = require_file(path, "{} entry".format(label))
        canonical = str(path)
        if canonical in canonical_seen:
            raise ValueError("duplicate canonical path in M31 {}".format(label))
        canonical_seen.add(canonical)
        if sha256(path) != expected:
            raise ValueError("M31 {} content hash drift: {}".format(label, path))
        entries.append((expected, path, raw_path))
    if not entries:
        raise ValueError("empty M31 {}".format(label))
    return ledger, entries


def entry_paths(entries):
    return set(path for _, path, _ in entries)


def require_ledger_paths(entries, required, label):
    present = entry_paths(entries)
    missing = set(Path(path).resolve() for path in required) - present
    if missing:
        raise ValueError("M31 {} required population drift: {}".format(
            label, sorted(str(path) for path in missing)))


def validate_functional_anchor(receipt_path, admission_path):
    receipt = require_file(receipt_path, "r4 functional receipt")
    admission = require_file(admission_path, "r4 functional admission")
    if sha256(receipt) != R4_RECEIPT_SHA256:
        raise ValueError("M31 exact r4 functional receipt identity drift")
    if sha256(admission) != R4_ADMISSION_SHA256:
        raise ValueError("M31 exact r4 functional admission identity drift")
    receipt_json = load_json_no_duplicates(receipt, "r4 functional receipt")
    admission_json = load_json_no_duplicates(admission, "r4 functional admission")
    if (receipt_json.get("schema") != "m31_output_receipt_v4"
            or receipt_json.get("headline_admitted") is not False):
        raise ValueError("M31 r4 functional receipt schema/admission drift")
    if (admission_json.get("schema")
            != "m31_r4_static_phase_vcs_machine_admission_v1"
            or admission_json.get("status")
            != "PASS_EXACT_FROZEN_M31_R4_STATIC_PHASE_VCS_ONLY"):
        raise ValueError("M31 r4 functional machine admission drift")
    admission_flags = admission_json.get("admission", {})
    if (admission_flags.get("current_r4_vcs_source_admitted") is not True
            or any(admission_flags.get(name) is not False for name in (
                "dc_sta_admitted", "formality_admitted", "headline_admitted",
                "phase_fault_recovery_admitted", "ppa_power_energy_admitted",
                "system_speedup_admitted"))):
        raise ValueError("M31 r4 functional admission boundary drift")
    if admission_json.get("identity", {}).get("receipt_sha256") != sha256(receipt):
        raise ValueError("M31 r4 functional admission/receipt linkage drift")
    return receipt, admission


def validate_dc(run, dc_audit_path):
    dc_audit_path = require_file(dc_audit_path, "fresh DC machine audit")
    rebuilt = rebuild_dc_audit(run, 3.000)
    audit = assert_exact_json(dc_audit_path, rebuilt, "fresh DC machine audit")
    require_exact_keys(audit, {
        "schema", "status", "identity", "resource_audit", "cell_accounting",
        "physical_assumptions", "timing", "area", "admission",
    }, "fresh DC audit top level")
    if audit["schema"] != DC_SCHEMA or audit["status"] != DC_STATUS:
        raise ValueError("M31 fresh DC exact schema/status drift")
    cells = audit["cell_accounting"]
    if (cells["total_cell_instances_including_hierarchy"]
            != cells["hierarchical_cell_instances"]
            + cells["leaf_mapped_cell_instances"]):
        raise ValueError("M31 fresh DC total/hierarchical/leaf accounting drift")
    if (cells["leaf_mapped_cell_instances"]
            != cells["combinational_leaf_cell_instances"]
            + cells["sequential_leaf_cell_instances"]):
        raise ValueError("M31 fresh DC leaf accounting drift")
    resource = audit["resource_audit"]
    if resource != dict(resource,
            pool_hierarchy_instances=1,
            multiplier_leaf_hierarchy_instances=96,
            pool_external_multiplier_leaf_instances=0,
            empty_mapped_multiplier_leaf_instances=0):
        raise ValueError("M31 fresh DC exact multiplier resource drift")
    physical = audit["physical_assumptions"]
    if (physical.get("clock_period_ns") != 3.0
            or physical.get("clock_count") != 1
            or physical.get("clock_network_model") != "IDEAL_UNPROPAGATED"
            or physical.get("interconnect_area_model") != "ZERO_WIRE_LOAD"
            or physical.get("net_interconnect_area_um2") != 0.0
            or physical.get("macro_timing_models") != "NONE"):
        raise ValueError("M31 fresh DC physical-assumption boundary drift")
    if audit["timing"].get("setup_and_hold_met") is not True:
        raise ValueError("M31 fresh DC timing is not met")
    if audit["area"].get("placed_or_routed_area_admitted") is not False:
        raise ValueError("M31 fresh DC improperly admits physical area")
    expected_admission = {
        "fresh_current_source_dc_sta_admitted": True,
        "formality_admitted": False,
        "paper_ppa_ready": False,
        "power_energy_admitted": False,
        "system_speedup_admitted": False,
        "headline_admitted": False,
    }
    if audit["admission"] != expected_admission:
        raise ValueError("M31 fresh DC claim boundary drift")
    return dc_audit_path, audit


def validate_formality(run, attempt, fm_audit_path):
    fm_audit_path = require_file(fm_audit_path, "strict Formality machine audit")
    recorded = load_json_no_duplicates(fm_audit_path,
                                       "strict Formality machine audit")
    expected_passing = recorded.get("verification", {}).get(
        "passing_compare_points")
    if not isinstance(expected_passing, int) or expected_passing <= 0:
        raise ValueError("M31 Formality passing population is invalid")
    rebuilt = rebuild_formality_audit(run, attempt, expected_passing)
    audit = assert_exact_json(fm_audit_path, rebuilt,
                              "strict Formality machine audit")
    require_exact_keys(audit, {
        "schema", "status", "identity", "verification", "admission",
    }, "strict Formality audit top level")
    if audit["schema"] != FM_SCHEMA or audit["status"] != FM_STATUS:
        raise ValueError("M31 strict Formality exact schema/status drift")
    verification = audit["verification"]
    exact_zero = (
        "failing_compare_points",
        "unmatched_reference_compare_points",
        "unmatched_implementation_compare_points",
        "unmatched_reference_primary_or_blackbox_points",
        "unmatched_implementation_primary_or_blackbox_points",
        "fmr_elab_147_diagnostics",
        "logic_simulator_disagreement_warnings",
    )
    if any(verification.get(name) != 0 for name in exact_zero):
        raise ValueError("M31 strict Formality zero-population contract drift")
    if audit["admission"] != {
            "rtl_to_exact_mapped_netlist_equivalence_admitted": True,
            "dc_sta_identity_inherited_from_external_machine_audit": True,
            "ppa_power_energy_admitted": False,
            "system_speedup_admitted": False,
            "headline_admitted": False}:
        raise ValueError("M31 strict Formality claim boundary drift")
    return fm_audit_path, audit


def validate_snapshot(run, snapshot_tag, snapshot_ledger_path, attempt):
    snapshot_dir = (run / "sealed_formality_{}".format(snapshot_tag)).resolve()
    canonical_snapshot_ledger = (
        run / "sealed_formality_evidence_{}.sha256".format(snapshot_tag)
    ).resolve()
    if Path(snapshot_ledger_path).resolve() != canonical_snapshot_ledger:
        raise ValueError("M31 snapshot ledger is not the canonical run-local path")
    if not snapshot_dir.is_dir():
        raise ValueError("missing M31 self-contained Formality snapshot")
    snapshot_ledger, entries = parse_ledger(
        snapshot_ledger_path, run, "snapshot ledger")
    listed = set()
    for _, path, _ in entries:
        try:
            path.relative_to(snapshot_dir)
        except ValueError:
            raise ValueError("M31 snapshot ledger entry escapes snapshot")
        listed.add(path)
    actual = set(path.resolve() for path in snapshot_dir.rglob("*")
                 if path.is_file())
    if listed != actual:
        raise ValueError("M31 snapshot ledger is not exactly closed")
    required = {
        snapshot_dir / "external_identity.sha256",
        snapshot_dir / "source_map.tsv",
        snapshot_dir / "formality_live_evidence.sha256",
        snapshot_dir / "formality_run_manifest.json",
        snapshot_dir / "seal_formality_snapshot_r2.sh",
        snapshot_dir / "inputs/run/netlist/{}_mapped.v".format(DESIGN),
        snapshot_dir / "inputs/run/netlist/{}.svf".format(DESIGN),
        snapshot_dir / "outputs/formality_{}.log".format(attempt),
        snapshot_dir / "outputs/formality_{}.exit_status".format(attempt),
        snapshot_dir / "outputs/reports/formality_status.txt",
        snapshot_dir / "outputs/reports/formality_unmatched.rpt",
        snapshot_dir / "outputs/reports/formality_verify.rpt",
        snapshot_dir / "outputs/formality_machine_audit_{}.json".format(
            attempt),
        snapshot_dir / "outputs/formality_admission_{}.txt".format(attempt),
        snapshot_dir / "outputs/formality_run_manifest.json",
    }
    if not required.issubset(listed):
        raise ValueError("M31 snapshot required artifact population drift")
    required_suffixes = {
        "rtl_m31/qfit_signed_int8_mul96_pool.sv",
        "rtl_m31/qfit_atlif_unified_t10_t2_stream_core.sv",
        "dc_handoff/filelists/date_m31_unified_t10_t2_dc.f",
        "dc_handoff/scripts/build_m31_r4_synopsys_receipt.py",
    }
    suffixes = set(str(path.relative_to(snapshot_dir)) for path in listed)
    for suffix in required_suffixes:
        matches = [name for name in suffixes if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError("M31 snapshot RTL/filelist population drift")
    return snapshot_dir, snapshot_ledger


def build(args):
    run = Path(args.run_dir).resolve()
    if not run.is_dir():
        raise ValueError("missing M31 fresh Synopsys run directory")
    if not re.match(r"^[A-Za-z0-9_.-]+$", args.attempt):
        raise ValueError("unsafe M31 Formality attempt tag")
    if not re.match(r"^[A-Za-z0-9_.-]+$", args.snapshot_tag):
        raise ValueError("unsafe M31 snapshot tag")
    if args.independent_review_score < 0 or args.independent_review_score > 100:
        raise ValueError("M31 independent review score is outside 0..100")
    builder_source = require_file(__file__, "Synopsys receipt builder source")
    expected_dc_audit = (
        run / "reports/m31_r4_dc_machine_audit.json").resolve()
    expected_fm_audit = (
        run / "formality_machine_audit_{}.json".format(args.attempt)).resolve()
    if Path(args.dc_audit).resolve() != expected_dc_audit:
        raise ValueError("M31 DC audit is not the canonical run-local path")
    if Path(args.formality_audit).resolve() != expected_fm_audit:
        raise ValueError("M31 Formality audit is not the canonical run-local path")

    functional_receipt, functional_admission = validate_functional_anchor(
        args.functional_receipt, args.functional_admission)
    dc_audit_path, dc = validate_dc(run, args.dc_audit)
    fm_audit_path, fm = validate_formality(run, args.attempt,
                                           args.formality_audit)
    if (dc["identity"]["mapped_netlist_sha256"]
            != fm["identity"]["mapped_netlist_sha256"]):
        raise ValueError("M31 DC/Formality mapped-netlist identity drift")

    dc_admission = require_file(run / "admission.txt", "DC admission")
    dc_evidence, dc_live_entries = parse_ledger(
        run / "evidence.sha256", run, "DC live evidence ledger")
    require_ledger_paths(dc_live_entries, {
        dc_audit_path, dc_admission, run / "dc.log",
        run / "reports/m31_resource_audit_postcompile.rpt",
        run / "reports/qor.rpt", run / "reports/area.rpt",
        run / "reports/clocks.rpt",
        run / "reports/references_postcompile.rpt",
        run / "reports/timing_setup.rpt", run / "reports/timing_hold.rpt",
        run / "netlist/{}_mapped.v".format(DESIGN),
        run / "netlist/{}.svf".format(DESIGN),
    }, "DC live evidence ledger")
    sealed_dc, sealed_dc_entries = parse_ledger(
        run / "sealed_dc_evidence.sha256", run,
        "sealed DC evidence ledger")
    required_sealed_dc = {dc_evidence}
    required_sealed_dc.update(
        path for path in entry_paths(dc_live_entries)
        if path == run or run in path.parents)
    require_ledger_paths(sealed_dc_entries, required_sealed_dc,
                         "sealed DC evidence ledger")
    fm_admission = require_file(
        run / "formality_admission_{}.txt".format(args.attempt),
        "Formality admission")
    fm_evidence, fm_live_entries = parse_ledger(
        run / "formality_evidence_{}.sha256".format(args.attempt), run,
        "Formality live evidence ledger")
    require_ledger_paths(fm_live_entries, {
        fm_audit_path, fm_admission,
        run / "formality_{}.log".format(args.attempt),
        run / "formality_{}.exit_status".format(args.attempt),
        run / "reports/formality_status.txt",
        run / "reports/formality_unmatched.rpt",
        run / "reports/formality_verify.rpt",
        run / "formality_run_manifest.json",
        builder_source,
    }, "Formality live evidence ledger")
    snapshot_dir, snapshot_ledger = validate_snapshot(
        run, args.snapshot_tag, args.snapshot_ledger, args.attempt)

    cells = dc["cell_accounting"]
    verification = fm["verification"]
    result = {
        "schema": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "date": args.date,
        "generation": {
            "builder_path": str(builder_source),
            "builder_sha256": sha256(builder_source),
            "output_policy": "CREATE_ONLY_NO_OVERWRITE",
            "live_audits_rebuilt_before_write": True,
            "all_ledgers_rehashed_before_write": True,
            "snapshot_exact_ledger_closure_required": True,
        },
        "functional_anchor": {
            "receipt_path": str(functional_receipt),
            "receipt_sha256": sha256(functional_receipt),
            "machine_admission_path": str(functional_admission),
            "machine_admission_sha256": sha256(functional_admission),
            "independent_review_score": args.independent_review_score,
            "review_scope": (
                "frozen M31-r4 current-source VCS/SVA and exact machine "
                "admission; Synopsys evidence is reviewed independently"),
        },
        "dc_sta": {
            "directory": str(run),
            "dc_machine_audit_path": str(dc_audit_path),
            "dc_machine_audit_sha256": sha256(dc_audit_path),
            "dc_live_evidence_ledger_sha256": sha256(dc_evidence),
            "sealed_dc_evidence_ledger_sha256": sha256(sealed_dc),
            "admission_sha256": sha256(dc_admission),
            "mapped_netlist_sha256": dc["identity"]["mapped_netlist_sha256"],
            "svf_sha256": dc["identity"]["svf_sha256"],
            "clock_period_ns": dc["physical_assumptions"]["clock_period_ns"],
            "setup_wns_ns": dc["timing"]["setup_wns_ns"],
            "hold_wns_ns": dc["timing"]["hold_wns_ns"],
            "total_cell_area_um2": dc["area"]["total_cell_area_um2"],
            "cell_accounting": cells,
            "resource_audit": dc["resource_audit"],
            "interconnect_model": "ZERO_WIRE_LOAD",
            "clock_network_model": "IDEAL_UNPROPAGATED",
            "paper_ppa_ready": False,
        },
        "formality": {
            "attempt": args.attempt,
            "machine_audit_path": str(fm_audit_path),
            "machine_audit_sha256": sha256(fm_audit_path),
            "live_evidence_ledger_sha256": sha256(fm_evidence),
            "admission_sha256": sha256(fm_admission),
            "passing_compare_points": verification["passing_compare_points"],
            "failing_compare_points": 0,
            "unmatched_reference_compare_points": 0,
            "unmatched_implementation_compare_points": 0,
            "unread_reference_points": verification["unread_reference_points"],
            "unread_implementation_points": verification[
                "unread_implementation_points"],
            "fmr_elab_147_diagnostics": 0,
            "logic_simulator_disagreement_warnings": 0,
            "self_contained_snapshot": {
                "canonical_directory": str(snapshot_dir),
                "ledger_path": str(snapshot_ledger),
                "ledger_sha256": sha256(snapshot_ledger),
                "ledger_entry_count": len(parse_ledger(
                    snapshot_ledger, run, "snapshot ledger final")[1]),
                "exact_ledger_closure": True,
            },
        },
        "supersedes": {
            "path": R1_RECEIPT_RELATIVE,
            "state": "ERROR_STALE_SUPERSEDED_DO_NOT_CITE",
            "reason": (
                "r1 predates the r4 static-phase source and mislabels the "
                "hierarchy-inclusive total cell count as leaf cells; its "
                "Formality log also contains FMR_ELAB-147 diagnostics and a "
                "logic-simulator-disagreement warning"),
        },
        "claim_boundary": {
            "permitted": (
                "current-source M31-r4 standalone exact VCS/SVA; one mapped "
                "multiplier-pool hierarchy with exactly 96 nonempty signed-INT8 "
                "multiplier leaves; 28nm zero-wire ideal-clock logic-only "
                "DC/STA at 3.000ns; and strict RTL-to-that-exact-mapped-netlist "
                "Formality equivalence"),
            "forbidden": (
                "robust or post-layout frequency, placed/routed area or timing, "
                "clock tree or extracted interconnect, SRAM/DRAM macros, "
                "SAIF/PTPX power or energy, full-network cycles/FPS/speedup, "
                "trained accuracy, comparison with Prosperity/Phi, or any DATE "
                "headline"),
        },
        "independent_synopsys_review_required": True,
        "headline_admitted": False,
    }
    require_exact_keys(result, {
        "schema", "status", "date", "generation", "functional_anchor", "dc_sta",
        "formality", "supersedes", "claim_boundary",
        "independent_synopsys_review_required", "headline_admitted",
    }, "generated receipt top level")
    return result


def write_output(path, result):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError("refusing to overwrite M31 Synopsys receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(str(path), flags, 0o644)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--snapshot-tag", required=True)
    parser.add_argument("--snapshot-ledger", type=Path, required=True)
    parser.add_argument("--dc-audit", type=Path, required=True)
    parser.add_argument("--formality-audit", type=Path, required=True)
    parser.add_argument("--functional-receipt", type=Path, required=True)
    parser.add_argument("--functional-admission", type=Path, required=True)
    parser.add_argument("--independent-review-score", type=int, required=True)
    parser.add_argument("--date", default="2026-08-22")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args)
    write_output(args.output, result)
    print(args.output)


if __name__ == "__main__":
    sys.exit(main())
