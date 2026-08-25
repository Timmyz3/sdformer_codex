#!/usr/bin/env python3
"""Type-strict producer validator for M43-r3 exact-SHA Synopsys evidence."""

from __future__ import print_function

import argparse
import hashlib
import json
import pathlib
import re


MANIFEST_SCHEMA = "m43_r3_exact_sha_synopsys_launch_manifest_v2"
CONTRACT_SCHEMA = "m43_r3_exact_sha_synopsys_contract_v2"
CONTRACT_STATUS = "FROZEN_EXACT_SOURCE_FRESH_STANDALONE_3NS_DC_STA_FORMALITY_R2"
RECEIPT_SCHEMA = "m43_r3_exact_sha_synopsys_receipt_v1"
CONTRACT_REL = (
    "hw_autoresearch_nts07/contracts/"
    "m43_r3_exact_sha_synopsys_contract_r2_20260823.json"
)
PATHS = {
    "candidate": "hw_autoresearch_nts07/rtl_m43/qfit_parent_delta_p8_l96_multicontext.sv",
    "vcs_receipt": "hw_autoresearch_nts07/contracts/m43_r2_exact_sha_vcs_receipt_r1_20260823.json",
    "review": "hw_autoresearch_nts07/results/m43_r2_independent_hammer_review_20260823/m43_r2_independent_hammer_review.json",
    "review_validator": "hw_autoresearch_nts07/results/m43_r2_independent_hammer_review_20260823/validate_m43_r2_independent_hammer_review.py",
    "filelist": "hw_autoresearch_nts07/dc_handoff/filelists/date_m43_r3_parent_delta_p8_l96_dc.f",
    "sdc": "hw_autoresearch_nts07/dc_handoff/constraints/date_m43_r3_parent_delta_p8_l96_3ns.sdc",
    "dc_tcl": "hw_autoresearch_nts07/dc_handoff/scripts/run_dc_m43_r3_exact_sha.tcl",
    "sta_tcl": "hw_autoresearch_nts07/dc_handoff/scripts/run_sta_m43_r3_exact_sha.tcl",
    "formality_tcl": "hw_autoresearch_nts07/dc_handoff/scripts/run_formality_m43_r3_exact_sha.tcl",
    "auditor": "hw_autoresearch_nts07/dc_handoff/scripts/audit_m43_r3_structural.py",
    "builder": "hw_autoresearch_nts07/dc_handoff/scripts/build_m43_r3_synopsys_receipt.py",
    "validator": "hw_autoresearch_nts07/dc_handoff/scripts/validate_m43_r3_exact_sha_synopsys.py",
    "runner": "hw_autoresearch_nts07/dc_handoff/scripts/run_m43_r3_exact_sha_synopsys.sh",
    "dc_binary": "tools/dc_resolved_binary",
    "fm_binary": "tools/formality_resolved_binary",
    "slow_lib": "libraries/tcbn28hpcplusbwp35p140ssg0p9v125c.db",
    "fast_lib": "libraries/tcbn28hpcplusbwp35p140ffg1p05vm40c.db",
}
EXPECTED = {
    "candidate": "e70239b1ec9a7d4541b0ae8d0a8f55e252fa6c804b364ab126d8201e108e0deb",
    "vcs_receipt": "3e416d615829c9b82206547ef3ab23178bfe3e01eeb0b0ff5a789bec116fe51a",
    "review": "8151f8f5ab0d1038fcdfc601a78da97300613304e46991ebfac2520d180d181a",
    "review_validator": "60e1488cff5e867005f32d168e1b66e62533d815d94e753777a4a2b397b3bd87",
    "dc_binary": "23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2",
    "fm_binary": "aceb24fb490927bf292dba8ce6a783fbad1dd648bb7e41710fc750b2dafed53b",
    "slow_lib": "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af",
    "fast_lib": "a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(
        pathlib.Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("invalid JSON constant: " + value)),
    )


def exact_int(value, minimum, label):
    require(type(value) is int and value >= minimum, label)


def exact_float(value, label):
    require(type(value) is float, label)


def parse_sha_manifest(path):
    entries = {}
    for number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\x00-\x1f]+)", line)
        require(match is not None, "SHA manifest syntax line {}".format(number))
        require(match.group(2) not in entries, "duplicate SHA manifest member")
        entries[match.group(2)] = match.group(1)
    return entries


def parse_key_values(path):
    result = {}
    for number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        key, separator, value = line.partition("=")
        require(separator and key and key not in result,
                "key/value syntax line {} in {}".format(number, path))
        result[key] = value
    return result


def validate(run):
    require(run.is_dir() and not run.is_symlink(), "run missing or symlinked")
    snapshot = run / "snapshot/inputs"
    launch = read_json(snapshot / "launch_manifest.json")
    require(set(launch) == {"schema", "entries"}, "launch manifest keys")
    require(launch["schema"] == MANIFEST_SCHEMA, "launch schema")
    launch_sha = (run / "launch_manifest.sha256").read_text(
        encoding="ascii").strip()
    require(re.fullmatch(r"[0-9a-f]{64}", launch_sha) is not None,
            "launch SHA syntax")
    require(sha(snapshot / "launch_manifest.json") == launch_sha,
            "launch manifest SHA drift")

    launch_entries = {}
    for entry in launch["entries"]:
        require(set(entry) == {"source", "snapshot", "sha256"},
                "launch entry keys")
        relative = pathlib.Path(entry["snapshot"])
        require(not relative.is_absolute() and ".." not in relative.parts,
                "snapshot path escape")
        name = str(relative)
        require(name not in launch_entries, "duplicate launch destination")
        member = snapshot / relative
        require(member.is_file() and not member.is_symlink(),
                "snapshot member missing/symlink: " + name)
        require(sha(member) == entry["sha256"], "snapshot SHA drift: " + name)
        launch_entries[name] = entry["sha256"]

    input_entries = parse_sha_manifest(run / "snapshot_input_sha256.txt")
    require(len(input_entries) == len(launch_entries) + 1,
            "snapshot input manifest population")
    require(input_entries.get("./launch_manifest.json") == launch_sha,
            "snapshot launch identity")
    for name, expected_sha in input_entries.items():
        require(name.startswith("./"), "snapshot manifest path form")
        member = snapshot / name[2:]
        require(member.is_file() and not member.is_symlink()
                and sha(member) == expected_sha,
                "snapshot input drift: " + name)

    contract = read_json(snapshot / CONTRACT_REL)
    require(contract["schema"] == CONTRACT_SCHEMA, "contract schema")
    require(contract["status"] == CONTRACT_STATUS, "contract status")
    require(CONTRACT_REL in launch_entries, "contract absent from launch")
    expected_map = dict(launch_entries)
    del expected_map[CONTRACT_REL]
    require(contract["exact_snapshot_sha256"] == expected_map,
            "contract exact snapshot map != launch")
    for key, expected_sha in EXPECTED.items():
        require(PATHS[key] in launch_entries, key + " absent from launch")
        require(launch_entries[PATHS[key]] == expected_sha, key + " SHA drift")

    admission = contract["m43_r2_admission"]
    require(admission["candidate_sha256"] == EXPECTED["candidate"],
            "candidate admission SHA")
    require(admission["vcs_receipt_sha256"] == EXPECTED["vcs_receipt"],
            "VCS receipt admission SHA")
    require(admission["independent_review_sha256"] == EXPECTED["review"],
            "review admission SHA")
    require(admission["independent_review_validator_sha256"]
            == EXPECTED["review_validator"], "review validator admission SHA")
    require(admission["independent_verdict"]
            == "GO_STANDALONE_EXACT_SHA_VCS_SVA_ONLY", "independent verdict")
    flow = contract["synopsys_flow"]
    require(flow["resolved_dc_binary_sha256"] == EXPECTED["dc_binary"],
            "DC binary contract SHA")
    require(flow["resolved_formality_binary_sha256"] == EXPECTED["fm_binary"],
            "Formality binary contract SHA")
    require(flow["setup_library_sha256"] == EXPECTED["slow_lib"],
            "setup library contract SHA")
    require(flow["hold_library_sha256"] == EXPECTED["fast_lib"],
            "hold library contract SHA")
    for contract_key, path_key in (
            ("filelist_sha256", "filelist"), ("sdc_sha256", "sdc"),
            ("dc_tcl_sha256", "dc_tcl"), ("sta_tcl_sha256", "sta_tcl"),
            ("formality_tcl_sha256", "formality_tcl"),
            ("structural_auditor_sha256", "auditor"),
            ("receipt_builder_sha256", "builder"),
            ("producer_validator_sha256", "validator"),
            ("snapshot_runner_sha256", "runner")):
        require(flow[contract_key] == launch_entries[PATHS[path_key]],
                contract_key + " drift")

    vcs = read_json(snapshot / PATHS["vcs_receipt"])
    review = read_json(snapshot / PATHS["review"])
    require(vcs["status"]
            == "PASS_EXACT_SHA_STANDALONE_VCS_SVA_PENDING_INDEPENDENT_HAMMER",
            "predecessor VCS status")
    require(review["status"] == "PASS_INDEPENDENT_HAMMER", "review status")
    require(review["verdict"] == "GO_STANDALONE_EXACT_SHA_VCS_SVA_ONLY",
            "review verdict")
    require(review["producer_receipt_attack"]["mismatch_count"] == 0,
            "producer receipt mismatch count")

    launch_receipt = parse_key_values(run / "external_launch_receipt.txt")
    require(set(launch_receipt) == {"launcher_sha256", "manifest_sha256"},
            "external launch receipt keys")
    require(re.fullmatch(r"[0-9a-f]{64}", launch_receipt["launcher_sha256"])
            is not None, "launcher SHA missing or malformed")
    require(launch_receipt["manifest_sha256"] == launch_sha,
            "external launcher manifest pin drift")

    external = parse_key_values(run / "external_tool_identity.txt")
    require(external == {
        "dc_launcher": "/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell",
        "dc_resolved": "/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell",
        "dc_resolved_sha256": EXPECTED["dc_binary"],
        "formality_launcher": "/opt/synopsys/fm/V-2023.12-SP3/bin/fm_shell",
        "formality_resolved": "/opt/synopsys/fm/V-2023.12-SP3/bin/fm_shell",
        "formality_resolved_sha256": EXPECTED["fm_binary"],
    }, "external tool identity drift")

    for stage in ("dc", "sta", "formality"):
        require((run / (stage + ".rc")).read_bytes() == b"0\n", stage + " rc")
        raw = (run / (stage + ".raw.log")).read_text(
            encoding="utf-8", errors="replace")
        require(not re.search(r"^(Error|Fatal):", raw, re.MULTILINE),
                stage + " Error/Fatal")
        require("Thank you" in raw, stage + " terminal marker")
    for stage in ("DC", "STA", "FORMALITY"):
        marker = (run / (stage + "_INTERNAL_COMPLETE.txt")).read_text(
            encoding="ascii")
        require(marker.count("M43_R3_{}_INTERNAL_COMPLETE=PASS".format(stage)) == 1,
                stage + " completion marker")

    physical = (run / "reports/constraint_contract_postcompile.rpt").read_text(
        encoding="utf-8", errors="replace")
    clocks = (run / "reports/clocks.rpt").read_text(
        encoding="utf-8", errors="replace")
    require("physical_contract=ZERO_WIRELOAD_IDEAL_CLOCK_NO_SRAM_MACRO" in physical,
            "physical contract marker")
    require(re.search(r"^Name\s+:\s+ZeroWireload\s*$", physical, re.MULTILINE),
            "explicit ZeroWireload evidence")
    require("Design allows ideal nets on clock nets." in physical,
            "ideal clock evidence")
    clock_match = re.search(
        r"^core_clk\s+([0-9.]+)\s+\{[^}]+\}\s+(\S+)\s+\{clk_core\}\s*$",
        clocks, re.MULTILINE)
    require(clock_match is not None and abs(float(clock_match.group(1)) - 3.0) < 1e-12,
            "3ns core clock")
    require("p" not in clock_match.group(2), "propagated clock forbidden")

    audit = (run / "reports/m43_r3_structural_audit.rpt").read_text(
        encoding="utf-8", errors="replace")
    for field in ("physical_multiplier_hit_total",
                  "postcompile_reference_blackbox_attribute_count",
                  "area_macro_or_blackbox_cell_count",
                  "unresolved_link_signature_count"):
        require(re.search(r"^{}=0$".format(field), audit, re.MULTILINE),
                field + " nonzero")

    receipt_path = run / "m43_r3_synopsys_receipt.json"
    receipt = read_json(receipt_path)
    require(receipt["schema"] == RECEIPT_SCHEMA, "receipt schema")
    require(receipt["status"] == "PASS_EXACT_SHA_FRESH_M43_R3_DC_STA_FORMALITY",
            "receipt status")
    require(type(receipt["candidate_changed"]) is bool
            and not receipt["candidate_changed"], "candidate changed")
    require(receipt["exact_identity"] == {
        "candidate_rtl_sha256": EXPECTED["candidate"],
        "vcs_receipt_sha256": EXPECTED["vcs_receipt"],
        "independent_review_sha256": EXPECTED["review"],
        "independent_review_validator_sha256": EXPECTED["review_validator"],
        "dc_resolved_binary_sha256": EXPECTED["dc_binary"],
        "formality_resolved_binary_sha256": EXPECTED["fm_binary"],
        "setup_library_sha256": EXPECTED["slow_lib"],
        "hold_library_sha256": EXPECTED["fast_lib"],
    }, "receipt exact identity")
    ppa = receipt["logic_only_ppa"]
    exact_int(ppa["cell_count"], 1, "cell count")
    exact_int(ppa["macro_or_blackbox_cell_count"], 0, "macro count")
    require(ppa["macro_or_blackbox_cell_count"] == 0, "macro count nonzero")
    for field in ("combinational_area_um2", "noncombinational_area_um2",
                  "total_cell_area_um2", "setup_wns_ns_slow_ssg0p9v125c",
                  "hold_wns_ns_fast_ffg1p05vm40c"):
        exact_float(ppa[field], field + " type")
    require(ppa["total_cell_area_um2"] > 0.0, "area nonpositive")
    require(ppa["setup_wns_ns_slow_ssg0p9v125c"] >= 0.0, "setup WNS")
    require(ppa["hold_wns_ns_fast_ffg1p05vm40c"] >= 0.0, "hold WNS")

    fm = receipt["formality"]
    exact_int(fm["passing_compare_points"], 1, "passing points")
    for field in ("failing_compare_points", "aborted_compare_points",
                  "unverified_compare_points", "unmatched_compare_points",
                  "unmatched_primary_or_blackbox_points", "fmr_elab_147_count"):
        require(type(fm[field]) is int and fm[field] == 0,
                "Formality " + field)
    require(type(fm["all_gates_pass"]) is bool and fm["all_gates_pass"],
            "Formality all gates")

    density = receipt["peak_compute_contract"]
    require(density["implemented_peak_signed_adds_per_cycle"] == 768,
            "implemented peak adds")
    require(density["conditional_k2_dual_destination_adds_per_cycle"] == 1536,
            "conditional K2 adds")
    area_um2 = ppa["total_cell_area_um2"]
    exact_float(density["implemented_peak_signed_adds_per_cycle_per_mm2"],
                "implemented density type")
    exact_float(density["conditional_k2_dual_destination_adds_per_cycle_per_mm2"],
                "conditional density type")
    require(abs(density["implemented_peak_signed_adds_per_cycle_per_mm2"]
                - 768.0e6 / area_um2) < 1e-9, "implemented density formula")
    require(abs(density["conditional_k2_dual_destination_adds_per_cycle_per_mm2"]
                - 1536.0e6 / area_um2) < 1e-9, "conditional density formula")
    require(density["conditional_k2_status"]
            == "ARCHITECTURAL_PROJECTION_NOT_IMPLEMENTED_OR_MEASURED_IN_THIS_BLOCK",
            "conditional K2 boundary")
    require(type(receipt["gates"]["all_pass"]) is bool
            and receipt["gates"]["all_pass"], "receipt gates")
    for boundary in ("paper_ppa_ready", "system_speedup_admitted",
                     "power_or_energy_admitted"):
        require(type(receipt["claim_boundary"][boundary]) is bool
                and not receipt["claim_boundary"][boundary], boundary)
    return receipt, contract, launch_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=pathlib.Path, required=True)
    args = parser.parse_args()
    receipt, contract, launch_sha = validate(args.run)
    print(
        "PASS M43-r3 exact-SHA Synopsys validator receipt_sha256={} "
        "contract_status={} launch_manifest_sha256={} area_um2={:.6f} "
        "setup_wns_ns={:.4f} hold_wns_ns={:.4f} fm_pass={} "
        "adds_per_cycle_per_mm2={:.6f} conditional_k2_destination_adds_per_cycle_per_mm2={:.6f}".format(
            sha(args.run / "m43_r3_synopsys_receipt.json"),
            contract["status"], launch_sha,
            receipt["logic_only_ppa"]["total_cell_area_um2"],
            receipt["logic_only_ppa"]["setup_wns_ns_slow_ssg0p9v125c"],
            receipt["logic_only_ppa"]["hold_wns_ns_fast_ffg1p05vm40c"],
            receipt["formality"]["passing_compare_points"],
            receipt["peak_compute_contract"]["implemented_peak_signed_adds_per_cycle_per_mm2"],
            receipt["peak_compute_contract"]["conditional_k2_dual_destination_adds_per_cycle_per_mm2"],
        )
    )


if __name__ == "__main__":
    main()
