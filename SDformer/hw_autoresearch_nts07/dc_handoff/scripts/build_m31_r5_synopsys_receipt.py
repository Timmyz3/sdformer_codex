#!/usr/bin/env python3
"""Build M31-r5 from the frozen r4 tool evidence without rerunning tools.

R5 tightens only evidence identity and admission.  It rejects symlinks and path
replacement, consumes exact-set relative ledgers, cross-binds both Formality
RTL inputs to the frozen VCS snapshot, and parses the complete DC clock table.
It does not expand the r4 logic-only claim boundary.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re


RUN_NAME = "m31_unified_t10_t2_dc_3p000ns_r4_static_phase_20260822"
ATTEMPT = "m31_r4_static_phase_fm1_20260822"
FM_SNAPSHOT_TAG = "m31_r4_static_phase_fm1_c094849e_r2_20260822"
VCS_SNAPSHOT_NAME = "m31_r4_vcs_inputs_c094849e_20260822"
CANONICAL_WORK_ROOT = Path("/home/zhumd/work")
REANCHOR_RELATIVE = (
    "sdformer_codex/SDformer/hw_autoresearch_nts07/results/"
    "m31_r5_frozen_evidence_reanchor_20260822/"
    "m31_r5_live_reanchor_relative_exact.sha256")
R5_RECEIPT_RELATIVE = (
    "hw_autoresearch_nts07/contracts/"
    "m31_synopsys_receipt_r5_evidence_hardened_20260822.json")

R4_VCS_RECEIPT_SHA256 = (
    "bae2f05e74ffa8863195bda9f222c22fc06364ade872e9cf83d3cd4106e5b77d")
R4_VCS_ADMISSION_SHA256 = (
    "e8bd1b6452280396a5c8fc83ce79f34d1ae08256f97b469613207418dcfd0ff6")
R4_VCS_SNAPSHOT_LEDGER_SHA256 = (
    "41009ec9ec86d4e19489bd49816634ca148340a0f19f784bd2d18bf2d3d0f22d")
R4_SYNOPSYS_RECEIPT_SHA256 = (
    "5cf35c5ef92e174e04d4169c2c924a5ac6962ab19dcf0fe4fa48c2f5e0e5c561")
R4_DC_AUDIT_SHA256 = (
    "4e4d24d25651d6fe37c018eb34bd7e9472a4869fd4c0da669a3db716e472f095")
R4_DC_SEALED_LEDGER_SHA256 = (
    "919f6e54ebb0841855306870e8ab3aa402ebc4ddfc00cefc9a1e14e0be9bc4fe")
R4_FM_AUDIT_SHA256 = (
    "aff84cef659e40930156d569a8f10d95d95e9bf2b8bdd2faa757ce7ab94f8f88")
R4_FM_SNAPSHOT_LEDGER_SHA256 = (
    "0b9586ae19bafbbb6d968067271c39b6c960446590b96a710cacccfd98ed0528")
CORE_SHA256 = (
    "c094849e88c0d9fc3a390d0cf6fc9adf10ff4dc31d77e265e425e5cf71b5ef15")
POOL_SHA256 = (
    "7872d25c01c112f07a7d8e3cfe728029eef1f68e0f7bf87bdf2a50416776ea18")
FM_FILELIST_SHA256 = (
    "850a3f1a44fadecd6e31278a1fc0016dc8b86abad6af13fea233b076aa6af861")
MAPPED_NETLIST_SHA256 = (
    "13cfe86f5004fd54d954ce124f63e660c2be9eea1a426bc26dd73f261f678d14")

SCHEMA = "m31_synopsys_receipt_r5_frozen_evidence_hardened_v1"
STATUS = (
    "PASS_M31_R5_EVIDENCE_HARDENING_ADVANCES_R4_FROZEN_TOOL_EVIDENCE_"
    "NO_TOOL_RERUN_LOGIC_ONLY")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_no_duplicates(path, label):
    def hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key in M31-r5 {}".format(label))
            result[key] = value
        return result
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=hook)


def require_exact_keys(value, expected, label):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("M31-r5 {} exact schema drift".format(label))


def ensure_no_symlink_chain(path, root, label):
    path = Path(os.path.abspath(str(path)))
    root = Path(os.path.abspath(str(root)))
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ValueError("M31-r5 {} escapes its allowed root".format(label))
    cursor = root
    if cursor.is_symlink():
        raise ValueError("M31-r5 {} root is a symlink".format(label))
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("M31-r5 {} contains a symlink".format(label))


def canonical_dir(raw, allowed_root, label, exact_relative=None):
    lexical = Path(os.path.abspath(str(raw)))
    root_lexical = Path(os.path.abspath(str(allowed_root)))
    ensure_no_symlink_chain(lexical, root_lexical, label)
    resolved_root = root_lexical.resolve()
    resolved = lexical.resolve()
    if resolved != lexical or not resolved.is_dir():
        raise ValueError("M31-r5 {} is missing or resolves through a link".format(
            label))
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError:
        raise ValueError("M31-r5 {} canonical containment failed".format(label))
    relative_text = "" if str(relative) == "." else str(relative)
    if exact_relative is not None and relative_text != exact_relative:
        raise ValueError("M31-r5 {} exact path identity drift".format(label))
    return resolved


def canonical_file(raw, allowed_root, label, exact_relative=None):
    lexical = Path(os.path.abspath(str(raw)))
    root_lexical = Path(os.path.abspath(str(allowed_root)))
    ensure_no_symlink_chain(lexical, root_lexical, label)
    resolved_root = root_lexical.resolve()
    resolved = lexical.resolve()
    if (resolved != lexical or not resolved.is_file()
            or resolved.stat().st_size == 0):
        raise ValueError("M31-r5 {} is missing, empty, or linked".format(label))
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError:
        raise ValueError("M31-r5 {} canonical containment failed".format(label))
    if exact_relative is not None and str(relative) != exact_relative:
        raise ValueError("M31-r5 {} exact path identity drift".format(label))
    return resolved


def parse_exact_relative_ledger(ledger_path, base_root, expected_relative,
                                label, expected_ledger_sha=None):
    ledger = canonical_file(ledger_path, base_root, label)
    if expected_ledger_sha is not None and sha256(ledger) != expected_ledger_sha:
        raise ValueError("M31-r5 {} identity drift".format(label))
    expected_relative = set(expected_relative)
    rows = {}
    canonical_targets = set()
    for line_number, line in enumerate(
            ledger.read_text(encoding="utf-8").splitlines(), 1):
        match = re.match(r"^([0-9a-f]{64})  ([^\0]+)$", line)
        if not match:
            raise ValueError("malformed M31-r5 {} line {}".format(
                label, line_number))
        expected_sha, relative = match.groups()
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("M31-r5 {} rejects absolute/path-escape entry".format(
                label))
        normalized = str(relative_path)
        if normalized != relative or relative in rows:
            raise ValueError("M31-r5 {} normalized/duplicate target drift".format(
                label))
        target = canonical_file(Path(base_root) / relative_path, base_root,
                                "{} entry".format(label), relative)
        if target in canonical_targets:
            raise ValueError("M31-r5 {} canonical target collision".format(label))
        canonical_targets.add(target)
        if sha256(target) != expected_sha:
            raise ValueError("M31-r5 {} content hash drift".format(label))
        rows[relative] = {"sha256": expected_sha, "path": target}
    if set(rows) != expected_relative:
        raise ValueError("M31-r5 {} must equal the exact expected set".format(label))
    return ledger, rows


def assert_exact_directory_files(directory, expected_paths, label):
    directory = Path(directory)
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("M31-r5 {} contains a symlink".format(label))
    actual = set(path.resolve() for path in directory.rglob("*")
                 if path.is_file())
    expected = set(Path(path).resolve() for path in expected_paths)
    if actual != expected:
        raise ValueError("M31-r5 {} exact file closure drift".format(label))


def dc_sealed_expected():
    return {
        "sealed_dc/inputs/dc_handoff/constraints/date_m31_unified_t10_t2.sdc",
        "sealed_dc/inputs/dc_handoff/filelists/date_m31_unified_t10_t2_dc.f",
        "sealed_dc/inputs/dc_handoff/scripts/audit_m31_r4_dc_reports.py",
        "sealed_dc/inputs/dc_handoff/scripts/run_dc_m31_unified_t10_t2.sh",
        "sealed_dc/inputs/dc_handoff/scripts/run_dc_m31_unified_t10_t2.tcl",
        "sealed_dc/inputs/rtl_m31/qfit_atlif_unified_t10_t2_stream_core.sv",
        "sealed_dc/inputs/rtl_m31/qfit_signed_int8_mul96_pool.sv",
        "sealed_dc/library_identity.sha256",
        "sealed_dc/seal_dc_evidence.sh",
        "sealed_dc/source_map.tsv",
        "evidence.sha256", "dc.log", "admission.txt",
        "reports/m31_r4_dc_machine_audit.json",
        "reports/m31_resource_audit_precompile.rpt",
        "reports/m31_resource_audit_postcompile.rpt",
        "reports/m31_lint33_audit.rpt", "reports/qor.rpt",
        "reports/area.rpt", "reports/clocks.rpt",
        "reports/references_postcompile.rpt",
        "reports/check_design_postcompile.rpt",
        "reports/check_timing_postcompile.rpt",
        "reports/timing_setup.rpt", "reports/timing_hold.rpt",
        "netlist/qfit_atlif_unified_t10_t2_stream_core_mapped.v",
        "netlist/qfit_atlif_unified_t10_t2_stream_core.svf",
        "netlist/qfit_atlif_unified_t10_t2_stream_core.ddc",
    }


def fm_snapshot_expected():
    prefix = "sealed_formality_{}".format(FM_SNAPSHOT_TAG)
    return set("{}/{}".format(prefix, relative) for relative in {
        "external_identity.sha256", "formality_live_evidence.sha256",
        "formality_run_manifest.json",
        "inputs/hw_root/dc_handoff/filelists/date_m31_unified_t10_t2_dc.f",
        "inputs/hw_root/dc_handoff/run_formality.sh",
        "inputs/hw_root/dc_handoff/scripts/audit_m31_r4_formality.py",
        "inputs/hw_root/dc_handoff/scripts/build_m31_r4_synopsys_receipt.py",
        "inputs/hw_root/dc_handoff/scripts/run_formality.tcl",
        "inputs/hw_root/dc_handoff/scripts/seal_formality_evidence.sh",
        "inputs/hw_root/dc_handoff/scripts/write_synopsys_run_manifest.py",
        "inputs/hw_root/rtl_m31/qfit_atlif_unified_t10_t2_stream_core.sv",
        "inputs/hw_root/rtl_m31/qfit_signed_int8_mul96_pool.sv",
        "inputs/run/netlist/qfit_atlif_unified_t10_t2_stream_core.svf",
        "inputs/run/netlist/qfit_atlif_unified_t10_t2_stream_core_mapped.v",
        "outputs/formality_admission_{}.txt".format(ATTEMPT),
        "outputs/formality_{}.exit_status".format(ATTEMPT),
        "outputs/formality_{}.log".format(ATTEMPT),
        "outputs/formality_machine_audit_{}.json".format(ATTEMPT),
        "outputs/formality_run_manifest.json",
        "outputs/reports/formality_status.txt",
        "outputs/reports/formality_unmatched.rpt",
        "outputs/reports/formality_verify.rpt",
        "seal_formality_snapshot_r2.sh", "source_map.tsv",
    })


def vcs_snapshot_expected():
    prefix = VCS_SNAPSHOT_NAME
    return set("{}/{}".format(prefix, relative) for relative in {
        "input_sha256.txt",
        "inputs/hw_root/dc_handoff/filelists/date_m31_unified_t10_t2_vcs.f",
        "inputs/hw_root/dc_handoff/scripts/run_vcs_m31_unified_t10_t2_sva.sh",
        "inputs/hw_root/rtl_m31/qfit_atlif_unified_t10_t2_stream_core.sv",
        "inputs/hw_root/rtl_m31/qfit_signed_int8_mul96_pool.sv",
        "inputs/hw_root/tb_m31/tb_qfit_atlif_unified_t10_t2_stream_core.sv",
        "inputs/hw_root/verif_m31/qfit_atlif_unified_t10_t2_stream_assertions.sv",
        "snapshot_admission.json", "source_map.tsv",
        "tools/seal_m31_r4_vcs_inputs.py",
    })


def parse_unique_clock_report(path):
    text = Path(path).read_text(encoding="utf-8")
    separators = [index for index, line in enumerate(text.splitlines())
                  if line == "-" * 80]
    if len(separators) != 2 or separators[1] != separators[0] + 2:
        raise ValueError("M31-r5 clock table population drift")
    lines = text.splitlines()
    row = lines[separators[0] + 1]
    match = re.match(
        r"^core_clk\s+3\.00\s+\{0 1\.5\}\s+f\s+\{clk_core\}\s*$", row)
    if not match:
        raise ValueError("M31-r5 unique core clock contract drift")
    trailing = [line.strip() for line in lines[separators[1] + 1:]
                if line.strip()]
    if trailing != ["1"]:
        raise ValueError("M31-r5 exact clock-count footer drift")
    return {"clock_count": 1, "clock_name": "core_clk",
            "period_ns": 3.0, "attributes": "f",
            "network": "IDEAL_UNPROPAGATED"}


def parse_exact_two_rtl_filelist(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    expected = [
        "rtl_m31/qfit_signed_int8_mul96_pool.sv",
        "rtl_m31/qfit_atlif_unified_t10_t2_stream_core.sv",
    ]
    if lines != expected or sha256(path) != FM_FILELIST_SHA256:
        raise ValueError("M31-r5 Formality exact two-RTL filelist drift")
    return lines


def validate_exact_rtl_cross_binding(fm_dir, vcs_dir, dc_dir):
    bindings = {
        "rtl_m31/qfit_signed_int8_mul96_pool.sv": POOL_SHA256,
        "rtl_m31/qfit_atlif_unified_t10_t2_stream_core.sv": CORE_SHA256,
    }
    filelist = fm_dir / (
        "inputs/hw_root/dc_handoff/filelists/date_m31_unified_t10_t2_dc.f")
    parse_exact_two_rtl_filelist(filelist)
    manifest = vcs_dir / "input_sha256.txt"
    manifest_rows = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([0-9a-f]{64})  ([^\0]+)$", line)
        if not match or match.group(2) in manifest_rows:
            raise ValueError("M31-r5 frozen VCS manifest population drift")
        manifest_rows[match.group(2)] = match.group(1)
    expected_manifest_paths = {
        "dc_handoff/filelists/date_m31_unified_t10_t2_vcs.f",
        "dc_handoff/scripts/run_vcs_m31_unified_t10_t2_sva.sh",
        "rtl_m31/qfit_atlif_unified_t10_t2_stream_core.sv",
        "rtl_m31/qfit_signed_int8_mul96_pool.sv",
        "tb_m31/tb_qfit_atlif_unified_t10_t2_stream_core.sv",
        "verif_m31/qfit_atlif_unified_t10_t2_stream_assertions.sv",
    }
    if set(manifest_rows) != expected_manifest_paths:
        raise ValueError("M31-r5 frozen VCS manifest is not exact six-input")
    for relative, expected_sha in manifest_rows.items():
        target = canonical_file(
            vcs_dir / "inputs/hw_root" / relative,
            vcs_dir, "frozen VCS six-input manifest target",
            "inputs/hw_root/{}".format(relative))
        if sha256(target) != expected_sha:
            raise ValueError("M31-r5 frozen VCS manifest content drift")
    for relative, expected_sha in bindings.items():
        fm_rtl = fm_dir / "inputs/hw_root" / relative
        vcs_rtl = vcs_dir / "inputs/hw_root" / relative
        dc_rtl = dc_dir / "inputs" / relative
        if any(sha256(path) != expected_sha for path in (fm_rtl, vcs_rtl, dc_rtl)):
            raise ValueError("M31-r5 FM/VCS/DC frozen RTL cross-binding drift")
        if manifest_rows.get(relative) != expected_sha:
            raise ValueError("M31-r5 VCS manifest RTL identity drift")
    return bindings


def expected_reanchor_files(work_root, repo_root, run):
    hw = repo_root / "hw_autoresearch_nts07"
    builder = Path(__file__)
    sealer = builder.parent / "seal_m31_r5_evidence_reanchor.py"
    files = {
        hw / "contracts/m31_output_receipt_r4_static_phase_20260822.json",
        hw / (
            "results/m31_r4_static_phase_vcs_machine_admission_20260822/"
            "m31_r4_static_phase_vcs_machine_admission.json"),
        hw / (
            "system_simulator/evidence/"
            "m31_r4_vcs_inputs_c094849e_20260822.sha256"),
        hw / "contracts/m31_synopsys_receipt_r2_static_phase_20260822.json",
        run / "reports/m31_r4_dc_machine_audit.json",
        run / "sealed_dc_evidence.sha256",
        run / "formality_machine_audit_{}.json".format(ATTEMPT),
        run / "sealed_formality_evidence_{}.sha256".format(FM_SNAPSHOT_TAG),
        builder, sealer,
    }
    relative = set()
    for path in files:
        path = canonical_file(path, work_root, "r5 reanchor source")
        relative.add(str(path.relative_to(work_root)))
    return relative


def validate_roots(work_root, repo_root, runs_root, run):
    work = canonical_dir(
        work_root, CANONICAL_WORK_ROOT, "work root", "")
    repo = canonical_dir(repo_root, work, "repo root", "sdformer_codex/SDformer")
    runs = canonical_dir(
        runs_root, work, "Synopsys runs root",
        "synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs")
    run = canonical_dir(run, runs, "fresh M31 r4 run", RUN_NAME)
    return work, repo, runs, run


def require_sha(path, expected, label):
    if sha256(path) != expected:
        raise ValueError("M31-r5 {} SHA-256 drift".format(label))


def validate_snapshot_cross_identity(run, fm_dir):
    """Bind live audit locations to the exact files carried in each snapshot."""
    pairs = (
        (run / "formality_machine_audit_{}.json".format(ATTEMPT),
         fm_dir / "outputs/formality_machine_audit_{}.json".format(ATTEMPT),
         R4_FM_AUDIT_SHA256, "Formality machine audit"),
        (run / "netlist/qfit_atlif_unified_t10_t2_stream_core_mapped.v",
         fm_dir / "inputs/run/netlist/"
         "qfit_atlif_unified_t10_t2_stream_core_mapped.v",
         MAPPED_NETLIST_SHA256, "Formality mapped-netlist input"),
    )
    for live, frozen, expected_sha, label in pairs:
        if sha256(live) != expected_sha or sha256(frozen) != expected_sha:
            raise ValueError("M31-r5 {} live/frozen cross-identity drift".format(
                label))


def build(args):
    work, repo, _, run = validate_roots(
        args.work_root, args.repo_root, args.runs_root, args.run_dir)
    hw = repo / "hw_autoresearch_nts07"
    reanchor = canonical_file(
        args.reanchor_ledger, repo, "r5 relative reanchor ledger",
        "hw_autoresearch_nts07/results/"
        "m31_r5_frozen_evidence_reanchor_20260822/"
        "m31_r5_live_reanchor_relative_exact.sha256")
    expected_top = expected_reanchor_files(work, repo, run)
    _, top_rows = parse_exact_relative_ledger(
        reanchor, work, expected_top, "r5 relative reanchor ledger")

    r4_vcs_receipt = hw / (
        "contracts/m31_output_receipt_r4_static_phase_20260822.json")
    r4_vcs_admission = hw / (
        "results/m31_r4_static_phase_vcs_machine_admission_20260822/"
        "m31_r4_static_phase_vcs_machine_admission.json")
    vcs_ledger = hw / (
        "system_simulator/evidence/"
        "m31_r4_vcs_inputs_c094849e_20260822.sha256")
    r4_receipt_path = hw / (
        "contracts/m31_synopsys_receipt_r2_static_phase_20260822.json")
    dc_audit_path = run / "reports/m31_r4_dc_machine_audit.json"
    dc_ledger = run / "sealed_dc_evidence.sha256"
    fm_audit_path = run / "formality_machine_audit_{}.json".format(ATTEMPT)
    fm_ledger = run / "sealed_formality_evidence_{}.sha256".format(
        FM_SNAPSHOT_TAG)
    for path, expected, label in (
            (r4_vcs_receipt, R4_VCS_RECEIPT_SHA256, "r4 VCS receipt"),
            (r4_vcs_admission, R4_VCS_ADMISSION_SHA256, "r4 VCS admission"),
            (vcs_ledger, R4_VCS_SNAPSHOT_LEDGER_SHA256,
             "r4 VCS snapshot ledger"),
            (r4_receipt_path, R4_SYNOPSYS_RECEIPT_SHA256,
             "r4 Synopsys receipt"),
            (dc_audit_path, R4_DC_AUDIT_SHA256, "r4 DC audit"),
            (dc_ledger, R4_DC_SEALED_LEDGER_SHA256, "r4 DC sealed ledger"),
            (fm_audit_path, R4_FM_AUDIT_SHA256, "r4 Formality audit"),
            (fm_ledger, R4_FM_SNAPSHOT_LEDGER_SHA256,
             "r4 Formality snapshot ledger")):
        require_sha(path, expected, label)

    _, dc_rows = parse_exact_relative_ledger(
        dc_ledger, run, dc_sealed_expected(), "DC frozen exact-set ledger",
        R4_DC_SEALED_LEDGER_SHA256)
    dc_sealed_dir = run / "sealed_dc"
    dc_sealed_files = [item["path"] for relative, item in dc_rows.items()
                       if relative.startswith("sealed_dc/")]
    assert_exact_directory_files(dc_sealed_dir, dc_sealed_files,
                                 "DC frozen snapshot")

    _, fm_rows = parse_exact_relative_ledger(
        fm_ledger, run, fm_snapshot_expected(),
        "Formality frozen exact-set ledger", R4_FM_SNAPSHOT_LEDGER_SHA256)
    fm_dir = run / "sealed_formality_{}".format(FM_SNAPSHOT_TAG)
    assert_exact_directory_files(fm_dir,
                                 [item["path"] for item in fm_rows.values()],
                                 "Formality frozen snapshot")

    vcs_base = vcs_ledger.parent
    _, vcs_rows = parse_exact_relative_ledger(
        vcs_ledger, vcs_base, vcs_snapshot_expected(),
        "VCS frozen exact-set ledger", R4_VCS_SNAPSHOT_LEDGER_SHA256)
    vcs_dir = vcs_base / VCS_SNAPSHOT_NAME
    assert_exact_directory_files(vcs_dir,
                                 [item["path"] for item in vcs_rows.values()],
                                 "VCS frozen snapshot")
    rtl_binding = validate_exact_rtl_cross_binding(
        fm_dir, vcs_dir, dc_sealed_dir)
    validate_snapshot_cross_identity(run, fm_dir)
    clock = parse_unique_clock_report(run / "reports/clocks.rpt")

    r4_receipt = load_json_no_duplicates(r4_receipt_path, "r4 receipt")
    require_exact_keys(r4_receipt, {
        "schema", "status", "date", "generation", "functional_anchor",
        "dc_sta", "formality", "supersedes", "claim_boundary",
        "independent_synopsys_review_required", "headline_admitted",
    }, "r4 receipt top level")
    if (r4_receipt["schema"] != "m31_synopsys_receipt_r2_fresh_r4_v1"
            or r4_receipt["headline_admitted"] is not False):
        raise ValueError("M31-r5 r4 receipt identity/admission drift")
    dc_audit = load_json_no_duplicates(dc_audit_path, "r4 DC audit")
    fm_audit = load_json_no_duplicates(fm_audit_path, "r4 Formality audit")
    if (dc_audit.get("status")
            != "PASS_M31_R4_EXACT96_ZERO_WIRE_IDEAL_CLOCK_3NS_LOGIC_ONLY"
            or fm_audit.get("status")
            != "PASS_M31_R4_RTL_TO_FRESH_MAPPED_NETLIST_FORMALITY_STRICT"):
        raise ValueError("M31-r5 frozen tool audit status drift")
    # The r4 audit and the newly parsed full clock table use intentionally
    # different field names.  Cross-check their exact values and types here.
    physical = dc_audit["physical_assumptions"]
    if not (type(physical.get("clock_count")) is int
            and physical["clock_count"] == 1
            and type(physical.get("clock_period_ns")) is float
            and physical["clock_period_ns"] == 3.0
            and physical.get("clock_attributes") == "f"
            and physical.get("clock_network_model") == "IDEAL_UNPROPAGATED"):
        raise ValueError("M31-r5 DC audit/unique-clock cross-check drift")
    verification = fm_audit.get("verification", {})
    for field in (
            "failing_compare_points", "unmatched_reference_compare_points",
            "unmatched_implementation_compare_points",
            "unmatched_reference_primary_or_blackbox_points",
            "unmatched_implementation_primary_or_blackbox_points",
            "fmr_elab_147_diagnostics",
            "logic_simulator_disagreement_warnings"):
        if type(verification.get(field)) is not int or verification[field] != 0:
            raise ValueError("M31-r5 strict Formality zero contract drift")
    if (type(verification.get("passing_compare_points")) is not int
            or verification["passing_compare_points"] != 7071):
        raise ValueError("M31-r5 strict Formality passing population drift")
    if fm_audit["identity"].get("mapped_netlist_sha256") != MAPPED_NETLIST_SHA256:
        raise ValueError("M31-r5 frozen mapped-netlist identity drift")

    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "date": args.date,
        "advances": {
            "receipt_path": str(r4_receipt_path),
            "receipt_sha256": R4_SYNOPSYS_RECEIPT_SHA256,
            "state": "FROZEN_TOOL_EVIDENCE_REANCHORED_BY_R5",
            "dc_or_formality_rerun": False,
            "reason": (
                "r5 hardens path, ledger, RTL, VCS-snapshot, and clock identity; "
                "it does not replace or reinterpret the frozen r4 tool results"),
        },
        "relative_exact_reanchor": {
            "ledger_path": str(reanchor),
            "ledger_sha256": sha256(reanchor),
            "entry_count": len(top_rows),
            "absolute_entries": 0,
            "path_escape_entries": 0,
            "extra_entries": 0,
            "symlink_entries": 0,
            "mutable_legacy_absolute_live_ledgers_admitted": False,
        },
        "frozen_vcs_anchor": {
            "receipt_sha256": R4_VCS_RECEIPT_SHA256,
            "machine_admission_sha256": R4_VCS_ADMISSION_SHA256,
            "snapshot_ledger_sha256": R4_VCS_SNAPSHOT_LEDGER_SHA256,
            "snapshot_file_count": len(vcs_rows),
            "exact_six_input_manifest": True,
            "formality_rtl_binding": rtl_binding,
            "formality_filelist_sha256": FM_FILELIST_SHA256,
        },
        "frozen_dc_sta": {
            "machine_audit_sha256": R4_DC_AUDIT_SHA256,
            "sealed_exact_set_ledger_sha256": R4_DC_SEALED_LEDGER_SHA256,
            "sealed_ledger_entry_count": len(dc_rows),
            "unique_clock_contract": clock,
            "cell_accounting": dc_audit["cell_accounting"],
            "resource_audit": dc_audit["resource_audit"],
            "timing": dc_audit["timing"],
            "area": dc_audit["area"],
            "interconnect_model": "ZERO_WIRE_LOAD",
            "macro_or_black_box_instances": 0,
            "paper_ppa_ready": False,
        },
        "frozen_formality": {
            "machine_audit_sha256": R4_FM_AUDIT_SHA256,
            "snapshot_exact_set_ledger_sha256": R4_FM_SNAPSHOT_LEDGER_SHA256,
            "snapshot_ledger_entry_count": len(fm_rows),
            "mapped_netlist_sha256": MAPPED_NETLIST_SHA256,
            "verification": verification,
        },
        "path_security": {
            "work_root": str(work), "repo_root": str(repo),
            "run_directory": str(run),
            "all_admitted_paths_realpath_contained": True,
            "all_admitted_path_components_symlink_free": True,
            "all_admitted_ledgers_relative_exact_set": True,
            "coherent_rehash_path_replacement_admitted": False,
        },
        "claim_boundary": r4_receipt["claim_boundary"],
        "independent_review_required": True,
        "headline_admitted": False,
    }
    require_exact_keys(result, {
        "schema", "status", "date", "advances", "relative_exact_reanchor",
        "frozen_vcs_anchor", "frozen_dc_sta", "frozen_formality",
        "path_security", "claim_boundary", "independent_review_required",
        "headline_admitted",
    }, "r5 receipt top level")
    return result


def write_output(path, result, repo_root=None):
    repo = Path(repo_root) if repo_root is not None else CANONICAL_WORK_ROOT / (
        "sdformer_codex/SDformer")
    repo = canonical_dir(
        repo, CANONICAL_WORK_ROOT, "receipt output repo root",
        "sdformer_codex/SDformer")
    path = Path(os.path.abspath(str(path)))
    expected = repo / R5_RECEIPT_RELATIVE
    if path != expected:
        raise ValueError("M31-r5 receipt output exact path identity drift")
    if path.exists() or path.is_symlink():
        raise ValueError("refusing to overwrite M31-r5 receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical_dir(
        path.parent, repo, "receipt output parent",
        str(Path(R5_RECEIPT_RELATIVE).parent))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(path), flags, 0o444)
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
    path.chmod(0o444)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reanchor-ledger", type=Path, required=True)
    parser.add_argument("--date", default="2026-08-22")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args)
    write_output(args.output, result, args.repo_root)
    print(args.output)


if __name__ == "__main__":
    main()
