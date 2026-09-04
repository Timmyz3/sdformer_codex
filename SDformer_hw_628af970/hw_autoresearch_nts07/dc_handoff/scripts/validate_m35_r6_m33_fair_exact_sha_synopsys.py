#!/usr/bin/env python3
"""Type-strict producer validator for the fresh M35-r7/M33 fair rerun.

In addition to tool/result gates, this validator treats the frozen contract as
executable policy.  Every SHA declared by the contract must resolve to a
member of the immutable snapshot and agree with the launch manifest.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import pathlib
import re


EXPECTED_MANIFEST_SCHEMA = "m35_r6_m33_fair_launch_manifest_v1"
EXPECTED_CONTRACT_SCHEMA = "m35_r7_m33_fair_exact_sha_synopsys_contract_v1"
EXPECTED_CONTRACT_STATUS = (
    "FROZEN_EXACT_SOURCE_FRESH_SEQUENTIAL_SAME_FLOW_3NS_DC_STA_FORMALITY_R7"
)
EXPECTED_RECEIPT_SCHEMA = "m35_r6_m33_fair_exact_sha_synopsys_receipt_v1"
EXPECTED_CANDIDATE = "84b1f3cb6344863ecfdbac2af8abcfdd15b1f16571979588badbc3e2e0dd1854"
EXPECTED_FIXED_BUILDER = "4525c438cca1ca9a2f29ae08209a4a3fb790d3ac733f6e7264893f3197b4cca5"
CONTRACT_REL = (
    "hw_autoresearch_nts07/contracts/"
    "m35_r7_m33_fair_exact_sha_synopsys_contract_r1_20260823.json"
)

PATHS = {
    "candidate": "hw_autoresearch_nts07/rtl_m35_r4/qfit_complement_csd8_canonical.sv",
    "m33_pool": "hw_autoresearch_nts07/rtl_m31/qfit_signed_int8_mul96_pool.sv",
    "m33_source": "hw_autoresearch_nts07/rtl_m33/qfit_threshold_late_scale_uq0p24_radix20x4.sv",
    "m35_filelist": "hw_autoresearch_nts07/dc_handoff/filelists/date_m35_r6_canonical_dc.f",
    "m33_filelist": "hw_autoresearch_nts07/dc_handoff/filelists/date_m33_r6_fair_dc.f",
    "sdc": "hw_autoresearch_nts07/dc_handoff/constraints/date_m35_r6_m33_fair_3ns.sdc",
    "dc_tcl": "hw_autoresearch_nts07/dc_handoff/scripts/run_dc_m35_r6_m33_fair_exact_sha.tcl",
    "sta_tcl": "hw_autoresearch_nts07/dc_handoff/scripts/run_sta_m35_r6_m33_fair_exact_sha.tcl",
    "formality_tcl": "hw_autoresearch_nts07/dc_handoff/scripts/run_formality_m35_r6_m33_fair_exact_sha.tcl",
    "auditor": "hw_autoresearch_nts07/dc_handoff/scripts/audit_m35_r6_zero_multiplier.py",
    "builder": "hw_autoresearch_nts07/dc_handoff/scripts/build_m35_r6_m33_fair_receipt.py",
    "validator": "hw_autoresearch_nts07/dc_handoff/scripts/validate_m35_r6_m33_fair_exact_sha_synopsys.py",
    "runner": "hw_autoresearch_nts07/dc_handoff/scripts/run_m35_r6_m33_fair_exact_sha_synopsys.sh",
    "descriptor_contract": "hw_autoresearch_nts07/contracts/m35_canonical_descriptor_contract_r4_20260822.json",
    "m35_review": "hw_autoresearch_nts07/results/m35_r5_independent_hammer_review_20260823/m35_r5_independent_hammer_review.json",
    "m35_review_validator": "hw_autoresearch_nts07/results/m35_r5_independent_hammer_review_20260823/validate_m35_r5_independent_hammer_review.py",
    "r6_no_go_review": "hw_autoresearch_nts07/results/m35_r6_m33_fair_independent_hammer_review_20260823/m35_r6_m33_fair_independent_hammer_review.json",
    "r6_no_go_validator": "hw_autoresearch_nts07/dc_handoff/scripts/validate_m35_r6_m33_fair_independent_hammer_review.py",
    "m35_vcs_receipt": "provenance/m35_r5_vcs_receipt.json",
    "m33_receipt": "hw_autoresearch_nts07/contracts/m33_output_receipt_r1_20260822.json",
    "slow_lib": "libraries/tcbn28hpcplusbwp35p140ssg0p9v125c.db",
    "fast_lib": "libraries/tcbn28hpcplusbwp35p140ffg1p05vm40c.db",
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
        require(match is not None, "manifest syntax line {}".format(number))
        require(match.group(2) not in entries, "duplicate manifest member")
        entries[match.group(2)] = match.group(1)
    return entries


def collect_sha_values(value):
    result = []
    if isinstance(value, dict):
        for item in value.values():
            result.extend(collect_sha_values(item))
    elif isinstance(value, list):
        for item in value:
            result.extend(collect_sha_values(item))
    elif isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
        result.append(value)
    return result


def validate_contract_against_launch(contract, launch_entries):
    """Fail closed if any frozen-contract SHA drifts from the launch snapshot."""
    require(contract["schema"] == EXPECTED_CONTRACT_SCHEMA, "contract schema")
    require(contract["status"] == EXPECTED_CONTRACT_STATUS, "contract status")
    require(CONTRACT_REL in launch_entries, "contract absent from launch manifest")

    expected_snapshot = dict(launch_entries)
    del expected_snapshot[CONTRACT_REL]
    declared_snapshot = contract["exact_snapshot_sha256"]
    require(type(declared_snapshot) is dict, "contract exact snapshot map type")
    require(declared_snapshot == expected_snapshot,
            "contract exact snapshot map != launch manifest")

    m35 = contract["m35_admission"]
    m33 = contract["m33_fair_baseline"]
    flow = contract["common_synopsys_flow"]
    repair = contract["repair_provenance"]
    semantic_pairs = [
        (m35["candidate_sha256"], PATHS["candidate"], "M35 candidate"),
        (m35["vcs_receipt_sha256"], PATHS["m35_vcs_receipt"], "M35 VCS receipt"),
        (m35["independent_hammer_review_sha256"], PATHS["m35_review"], "M35 review"),
        (m33["source_sha256"][0], PATHS["m33_pool"], "M33 multiplier pool"),
        (m33["source_sha256"][1], PATHS["m33_source"], "M33 source"),
        (m33["recursive_output_receipt_sha256"], PATHS["m33_receipt"], "M33 receipt"),
        (flow["setup_library_sha256"], PATHS["slow_lib"], "setup library"),
        (flow["hold_library_sha256"], PATHS["fast_lib"], "hold library"),
        (flow["common_sdc_sha256"], PATHS["sdc"], "common SDC"),
        (flow["m35_filelist_sha256"], PATHS["m35_filelist"], "M35 filelist"),
        (flow["m33_filelist_sha256"], PATHS["m33_filelist"], "M33 filelist"),
        (flow["dc_tcl_sha256"], PATHS["dc_tcl"], "DC Tcl"),
        (flow["sta_tcl_sha256"], PATHS["sta_tcl"], "STA Tcl"),
        (flow["formality_tcl_sha256"], PATHS["formality_tcl"], "Formality Tcl"),
        (flow["m35_structural_auditor_sha256"], PATHS["auditor"], "M35 auditor"),
        (flow["receipt_builder_sha256"], PATHS["builder"], "receipt builder"),
        (flow["independent_run_validator_sha256"], PATHS["validator"], "validator"),
        (flow["snapshot_runner_sha256"], PATHS["runner"], "snapshot runner"),
        (repair["r6_no_go_review_sha256"], PATHS["r6_no_go_review"],
         "r6 NO-GO review"),
        (repair["r6_no_go_validator_sha256"], PATHS["r6_no_go_validator"],
         "r6 NO-GO validator"),
    ]
    for declared, relative, label in semantic_pairs:
        require(relative in launch_entries, label + " path missing from launch")
        require(declared == launch_entries[relative], label + " SHA mismatch")
    require(flow["receipt_builder_sha256"] == EXPECTED_FIXED_BUILDER,
            "contract does not pin fixed r6 receipt parser")
    require(m35["candidate_sha256"] == EXPECTED_CANDIDATE,
            "contract candidate identity drift")

    launch_hashes = set(launch_entries.values())
    for declared in collect_sha_values(contract):
        require(declared in launch_hashes,
                "contract SHA not backed by a launch/snapshot member: " + declared)
    return {
        "contract_sha256": launch_entries[CONTRACT_REL],
        "declared_snapshot_member_count": len(declared_snapshot),
        "all_contract_sha_values_launch_backed": True,
        "receipt_builder_sha256": flow["receipt_builder_sha256"],
        "validator_sha256": flow["independent_run_validator_sha256"],
        "runner_sha256": flow["snapshot_runner_sha256"],
    }


def validate(run):
    require(run.is_dir() and not run.is_symlink(), "run missing or symlinked")
    snapshot = run / "snapshot/inputs"
    launch = read_json(snapshot / "launch_manifest.json")
    require(set(launch) == {"schema", "entries"}, "launch manifest key drift")
    require(launch["schema"] == EXPECTED_MANIFEST_SCHEMA, "launch schema")
    launch_sha = (run / "launch_manifest.sha256").read_text(encoding="ascii").strip()
    require(re.fullmatch(r"[0-9a-f]{64}", launch_sha) is not None, "launch SHA syntax")
    require(sha(snapshot / "launch_manifest.json") == launch_sha, "launch SHA drift")

    launch_entries = {}
    for entry in launch["entries"]:
        require(set(entry) == {"source", "snapshot", "sha256"}, "launch entry keys")
        relative = pathlib.Path(entry["snapshot"])
        require(not relative.is_absolute() and ".." not in relative.parts,
                "snapshot path escape")
        name = str(relative)
        require(name not in launch_entries, "duplicate snapshot destination")
        member = snapshot / relative
        require(member.is_file() and not member.is_symlink(), "snapshot member")
        require(sha(member) == entry["sha256"], "snapshot member SHA drift: " + name)
        launch_entries[name] = entry["sha256"]

    input_entries = parse_sha_manifest(run / "snapshot_input_sha256.txt")
    require(len(input_entries) == len(launch_entries) + 1,
            "snapshot input manifest population")
    for name, expected in input_entries.items():
        require(name.startswith("./"), "snapshot manifest path form")
        member = snapshot / name[2:]
        require(member.is_file() and not member.is_symlink() and sha(member) == expected,
                "snapshot manifest member drift")
    require(input_entries["./launch_manifest.json"] == launch_sha,
            "snapshot input manifest launch identity")

    contract = read_json(snapshot / CONTRACT_REL)
    exact_inputs = validate_contract_against_launch(contract, launch_entries)

    external = {}
    for line in (run / "external_launch_receipt.txt").read_text(
            encoding="ascii").splitlines():
        key, separator, value = line.partition("=")
        require(separator and key not in external, "external launch receipt syntax")
        external[key] = value
    require(set(external) == {"launcher_sha256", "manifest_sha256"},
            "external launch receipt keys")
    require(re.fullmatch(r"[0-9a-f]{64}", external["launcher_sha256"]) is not None,
            "launcher SHA missing or malformed")
    require(external["manifest_sha256"] == launch_sha,
            "external launcher manifest pin drift")

    for key in ("m35", "m33"):
        directory = run / key
        for stage in ("dc", "sta", "formality"):
            require((directory / (stage + ".rc")).read_bytes() == b"0\n",
                    key + " " + stage + " rc")
        for stage in ("DC", "STA", "FORMALITY"):
            marker = directory / (stage + "_INTERNAL_COMPLETE.txt")
            require(marker.read_text().count(
                "M35_R6_M33_FAIR_{}_INTERNAL_COMPLETE=PASS".format(stage)) == 1,
                key + " " + stage + " marker")
        for stage in ("dc", "sta", "formality"):
            raw = (directory / (stage + ".raw.log")).read_text(
                encoding="utf-8", errors="replace")
            require(not re.search(r"^(Error|Fatal):", raw, re.MULTILINE),
                    key + " " + stage + " Error/Fatal")
            require("Thank you" in raw, key + " " + stage + " terminal marker")
        fm = (directory / "formality.raw.log").read_text(
            encoding="utf-8", errors="replace")
        require(len(re.findall(r"^Verification SUCCEEDED$", fm, re.MULTILINE)) == 1,
                key + " Formality terminal population")

    audit = (run / "m35/reports/m35_r6_zero_multiplier_audit.rpt").read_text()
    require(re.search(r"^physical_multiplier_hit_total=0$", audit, re.MULTILINE),
            "M35 zero multiplier audit")
    require(re.search(r"^postcompile_blackbox_attribute_count=0$", audit, re.MULTILINE),
            "M35 blackbox audit")
    require(re.search(r"^unresolved_link_signature_count=0$", audit, re.MULTILINE),
            "M35 unresolved-link audit")

    receipt_path = run / "m35_r6_m33_fair_receipt.json"
    receipt = read_json(receipt_path)
    require(receipt["schema"] == EXPECTED_RECEIPT_SCHEMA, "receipt schema")
    require(receipt["status"] == "PASS_EXACT_SHA_FRESH_M35_AND_M33_DC_STA_FORMALITY",
            "receipt status")
    require(type(receipt["candidate_changed"]) is bool and not receipt["candidate_changed"],
            "candidate changed type/value")
    require(type(receipt["gates"]["all_pass"]) is bool and receipt["gates"]["all_pass"],
            "all pass type/value")
    for key in ("m35", "m33"):
        item = receipt[key]
        exact_int(item["cell_count"], 1, key + " cell count")
        exact_int(item["macro_or_blackbox_cell_count"], 0, key + " macro count")
        require(item["macro_or_blackbox_cell_count"] == 0, key + " macro nonzero")
        for field in ("total_cell_area_um2", "setup_wns_ns_slow_ssg0p9v125c",
                      "hold_wns_ns_fast_ffg1p05vm40c"):
            exact_float(item[field], key + " " + field + " type")
        require(item["setup_wns_ns_slow_ssg0p9v125c"] >= 0.0, key + " setup")
        require(item["hold_wns_ns_fast_ffg1p05vm40c"] >= 0.0, key + " hold")
        formal = item["formality"]
        exact_int(formal["passing_compare_points"], 1, key + " passing points")
        for field in ("failing_compare_points", "aborted_compare_points",
                      "unverified_compare_points", "unmatched_compare_points",
                      "unmatched_primary_or_blackbox_points", "fmr_elab_147_count"):
            require(type(formal[field]) is int and formal[field] == 0,
                    key + " Formality " + field)
        require(type(formal["message_filters_used"]) is bool
                and not formal["message_filters_used"], key + " filters")
    comparison = receipt["fair_comparison"]
    for field in ("m35_over_m33_area", "m35_over_m33_peak_result_rate",
                  "m35_over_m33_result_rate_per_area",
                  "m35_area_per_result_reduction_percent"):
        exact_float(comparison[field], "comparison " + field)
    m35_area = receipt["m35"]["total_cell_area_um2"]
    m33_area = receipt["m33"]["total_cell_area_um2"]
    require(abs(comparison["m35_over_m33_area"] - m35_area / m33_area) < 1e-12,
            "area ratio formula")
    require(abs(comparison["m35_over_m33_peak_result_rate"] - 2.0) < 1e-12,
            "peak rate formula")
    require(abs(comparison["m35_over_m33_result_rate_per_area"]
                - 2.0 * m33_area / m35_area) < 1e-12, "density formula")
    require(type(receipt["claim_boundary"]["paper_ppa_ready"]) is bool
            and not receipt["claim_boundary"]["paper_ppa_ready"], "paper PPA boundary")
    require(type(receipt["claim_boundary"]["system_speedup_admitted"]) is bool
            and not receipt["claim_boundary"]["system_speedup_admitted"],
            "system speedup boundary")
    return receipt, exact_inputs, external


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=pathlib.Path, required=True)
    args = parser.parse_args()
    receipt, exact_inputs, external = validate(args.run)
    print(
        "PASS M35-r7/M33 exact-SHA same-flow validator "
        "receipt_sha256={} contract_sha256={} launcher_sha256={} "
        "area_ratio={:.12f} density_ratio={:.12f}".format(
            sha(args.run / "m35_r6_m33_fair_receipt.json"),
            exact_inputs["contract_sha256"], external["launcher_sha256"],
            receipt["fair_comparison"]["m35_over_m33_area"],
            receipt["fair_comparison"]["m35_over_m33_result_rate_per_area"],
        )
    )


if __name__ == "__main__":
    main()
