#!/usr/bin/env python3
"""Read-only, different-author hammer for the M1665 C1 recovered result.

This program never invokes EDA and never writes into the result tree.  Its
``--attack`` switch mutates only in-memory audit state so negative tests cannot
alter sealed evidence.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path


EXPECTED = {
    "m1649_runner_sha256": "8a1688206acf75ee0942c7bf6acb20b16c3017c7bf54451ab11d84953a4474e3",
    "m1649_contract_sha256": "5ca134044f1e100c925785db8025b8a7dce3e23daf5c3964608ca039ace84fb3",
    "m1650_review_sha256": "1ed6522019a7c34109ce44e0a7f5a959343e61f28151f08ec27dbb66546589bb",
    "m1651_release_sha256": "5e68e99c49a5e7ab04b0883b06537398b5cf41c76d6812d08b9c87fc988771ef",
    "m1655_review_sha256": "4d6f3e2cb238fbe77038cfc213d31ce061e17d49f43badcbc6b30ee8ffb825b2",
    "m1659_source_sha256": "cfd06bc58023869350668ab256311f97728e86db1f5d19d1933e2c9753960e73",
    "m1659_contract_sha256": "9516194a6e3195cae1803857967138b5e92eccc67ba4af9831bc2b2aa4099ec6",
    "m1660_review_sha256": "dee5ea244e0913d77b5995bd6e714ce6a9af9449d3866629807d31fd7f939ae8",
    "m1664_release_sha256": "4e261d4a86e98e26b359bc7b82d8004cd658227e656243a6fd92318417ce060f",
}


def fail(message):
    raise AssertionError(message)


def require(condition, message):
    if not condition:
        fail(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key: " + key)
        out[key] = value
    return out


def reject_constant(value):
    raise ValueError("non-finite JSON number: " + value)


def load_json(path):
    return json.loads(path.read_text(), object_pairs_hook=unique_object,
                      parse_constant=reject_constant)


def parse_manifest(path):
    entries = {}
    line_re = re.compile(r"^([0-9a-f]{64})  ([^/].*)$")
    for line in path.read_text().splitlines():
        match = line_re.match(line)
        require(match is not None, "malformed manifest line in " + str(path))
        digest, rel = match.groups()
        require(rel not in entries, "duplicate manifest member " + rel)
        parts = Path(rel).parts
        require(".." not in parts and "." not in parts,
                "unsafe manifest member " + rel)
        entries[rel] = digest
    return entries


def regular_files(root):
    files = set()
    for path in root.rglob("*"):
        require(not path.is_symlink(), "symlink rejected: " + str(path))
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
    return files


def verify_manifest(root, expected_members, unlisted_allowed=None):
    unlisted_allowed = set(unlisted_allowed or [])
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    entries = parse_manifest(manifest)
    require(len(entries) == expected_members,
            "member count mismatch for " + str(root))
    allowed = set(entries) | {"SHA256SUMS", "SHA256SUMS.seal.sha256"} | unlisted_allowed
    require(regular_files(root) == allowed, "sealed topology mismatch for " + str(root))
    for rel, digest in entries.items():
        path = root / rel
        require(path.is_file() and not path.is_symlink(), "non-regular member " + rel)
        require(sha256(path) == digest, "member digest mismatch " + rel)
    outer_entries = parse_manifest(outer)
    require(outer_entries == {"SHA256SUMS": sha256(manifest)},
            "outer seal mismatch for " + str(root))
    return entries


def verify_file_seal(path):
    manifest = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(parse_manifest(manifest) == {path.name: sha256(path)},
            "file manifest mismatch " + str(path))
    require(parse_manifest(outer) == {manifest.name: sha256(manifest)},
            "file outer seal mismatch " + str(path))


def keyvals(path):
    out = {}
    for raw in path.read_text().splitlines():
        if "=" in raw:
            key, value = raw.split("=", 1)
            require(key not in out, "duplicate key in " + str(path))
            out[key] = value
    return out


def close(a, b, eps=1e-12):
    return math.fabs(float(a) - float(b)) <= eps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--attack", default="none", choices=[
        "none", "source_byte", "target_count", "target_manifest",
        "source_manifest_identity", "dc_rerun", "eda_run", "setup_wns",
        "hold_wns", "area", "macro_count", "drc", "paper_citable",
        "error_line", "provenance_identity",
    ])
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    base = repo / "hw_autoresearch_nts07"
    q = base / "dc_handoff/runs/m1649_m1630_c1_resource_gate_successor_dc_r1_20260901.failed_or_incomplete.519344.quarantine"
    target = base / "dc_handoff/runs/m1665_m1664_m1659_m1649_c1_residual_hold_closed_dc_recovered_canonical_r1_20260901"
    copied = target / "original_quarantine"
    source_attempt = base / "dc_handoff/runs/.m1649_m1630_c1_resource_gate_successor_dc_attempt_consumed"
    recovery_attempt = base / "dc_handoff/runs/.m1665_m1659_c1_canonical_recovery_attempt_consumed"

    paths = {
        "m1649_runner_sha256": base / "dc_handoff/scripts/run_dc_m1649_m1630_c1_resource_gate_successor_exact_sha_r1.sh",
        "m1649_contract_sha256": base / "contracts/m1649_m1630_c1_resource_gate_successor_dc_source_contract_r1_20260901.json",
        "m1650_review_sha256": base / "reviews/m1650_m1649_m1630_c1_resource_gate_successor_dc_source_hammer_r1_20260901/review.json",
        "m1651_release_sha256": base / "contracts/m1651_m1650_m1649_m1630_c1_resource_gate_successor_dc_launch_release_r1_20260901.json",
        "m1655_review_sha256": base / "reviews/m1655_m1649_c1_quarantine_forensic_recovery_review_r1_20260901/review.json",
        "m1659_source_sha256": base / "dc_handoff/scripts/promote_m1659_m1649_c1_quarantine_atomic_canonical_recovery_r1.sh",
        "m1659_contract_sha256": base / "contracts/m1659_m1649_c1_atomic_canonical_recovery_source_contract_r1_20260901.json",
        "m1660_review_sha256": base / "reviews/m1660_m1659_c1_canonical_recovery_source_independent_review_r1_20260901/review.json",
        "m1664_release_sha256": base / "contracts/m1664_m1660_m1659_c1_canonical_recovery_release_r1_20260901.json",
    }

    # All predecessor authorities are exact, and all JSON/review authorities
    # retain their own manifest plus outer seal.
    actual_identity = {key: sha256(path) for key, path in paths.items()}
    if args.attack == "provenance_identity":
        actual_identity["m1664_release_sha256"] = "0" * 64
    require(actual_identity == EXPECTED, "predecessor identity chain mismatch")
    for key in ("m1649_contract_sha256", "m1651_release_sha256",
                "m1659_contract_sha256", "m1664_release_sha256"):
        verify_file_seal(paths[key])
    for key in ("m1650_review_sha256", "m1655_review_sha256", "m1660_review_sha256"):
        review_dir = paths[key].parent
        review_entries = parse_manifest(review_dir / "SHA256SUMS")
        verify_manifest(review_dir, len(review_entries))
    require(sha256(base / "docs/359_DATE终局冻结_20260813.md") ==
            "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
            "docs/359 identity drift")

    q_entries = verify_manifest(q, 39)
    copied_entries = verify_manifest(copied, 39)
    require(q_entries == copied_entries, "copied manifest entry mismatch")
    source_copy_equal = True
    for rel in sorted(regular_files(q)):
        if sha256(q / rel) != sha256(copied / rel):
            source_copy_equal = False
    if args.attack == "source_byte":
        source_copy_equal = False
    require(source_copy_equal, "original quarantine was not copied byte-exact")

    target_entries = verify_manifest(
        target, 42,
        {"original_quarantine/SHA256SUMS",
         "original_quarantine/SHA256SUMS.seal.sha256"})
    if args.attack == "target_count":
        require(len(target_entries) == 41, "injected target count mutation")
    require(len(target_entries) == 42, "target does not seal 42 members")
    if args.attack == "target_manifest":
        fail("injected target manifest digest mutation")

    verify_manifest(source_attempt, 1)
    verify_manifest(recovery_attempt, 2)
    recovery_identity = keyvals(recovery_attempt / "IDENTITY.txt")
    require(recovery_identity == {
        "source_sha256": EXPECTED["m1659_source_sha256"],
        "release_sha256": EXPECTED["m1664_release_sha256"],
        "source_quarantine_manifest_sha256": sha256(q / "SHA256SUMS"),
    }, "recovery attempt identity mismatch")
    attempt_text = (recovery_attempt / "ATTEMPT_CONSUMED.txt").read_text()
    require("max_recoveries=1" in attempt_text and "retry=false" in attempt_text,
            "one-shot attempt contract mismatch")

    provenance = load_json(target / "M1665_RECOVERY_PROVENANCE.json")
    receipt = load_json(target / "m1665_recovered_c1_dc_receipt.json")
    run_complete = keyvals(target / "RUN_COMPLETE_RECOVERED.txt")
    if args.attack == "source_manifest_identity":
        provenance["source_manifest_sha256"] = "0" * 64
    if args.attack == "dc_rerun":
        provenance["mutation"]["dc_rerun"] = True
    if args.attack == "eda_run":
        provenance["mutation"]["eda_run"] = True
    if args.attack == "setup_wns":
        receipt["setup"]["wns_ns"] = -0.001
    if args.attack == "hold_wns":
        receipt["hold"]["wns_ns"] = -0.001
    if args.attack == "area":
        receipt["area"]["recovered_um2"] = 1.0
    if args.attack == "macro_count":
        receipt["macros"]["count"] = 8
    if args.attack == "drc":
        receipt["drc_violating_nets"] = 1
    if args.attack == "paper_citable":
        receipt["claim_boundary"]["paper_citable"] = True

    require(provenance["status"] ==
            "COPY_ONLY_RECOVERY_OF_DC_COMPLETE_SEALED_M1649_QUARANTINE",
            "provenance status mismatch")
    require(provenance["source_members"] == 39, "source member claim mismatch")
    require(provenance["source_manifest_sha256"] == sha256(q / "SHA256SUMS"),
            "source manifest identity mismatch")
    require(provenance["source_outer_seal_file_sha256"] == sha256(q / "SHA256SUMS.seal.sha256"),
            "source outer-seal identity mismatch")
    require(provenance["identity"] == EXPECTED, "provenance authority identity mismatch")
    require(receipt["identity"] == EXPECTED, "receipt authority identity mismatch")
    require(provenance["mutation"] == {
        "source_quarantine_modified": False,
        "dc_rerun": False,
        "copied_artifact_bytes_changed": False,
        "eda_run": False,
    }, "recovery mutation boundary mismatch")
    require(provenance["concurrency"] == {
        "atomic_launch_lock": True,
        "one_shot_attempt_before_copy": True,
        "fixed_work_identity": True,
        "no_replace_atomic_publish": True,
        "retry": False,
    }, "recovery concurrency boundary mismatch")

    # Directly rederive the sole waived startup line and all post-flow facts
    # from the copied, sealed bytes rather than trusting the new receipt.
    require((copied / "dc.rc").read_text().strip() == "0", "dc rc is not zero")
    terminal = keyvals(copied / "TCL_INTERNAL_COMPLETE.txt")
    require(terminal["status"] == "M1630_DC_INTERNAL_COMPLETE__RUNNER_GATE_REQUIRED",
            "Tcl terminal marker mismatch")
    require(terminal["input_generation"] == "original_m993_m1006_admitted_ddc",
            "wrong mapped input generation")
    require(terminal["failed_m1614_output_used"] == "false",
            "failed M1614 output was used")
    require(terminal["hold_only_incremental_mapping_count"] == "1",
            "incremental mapping count mismatch")

    log_lines = (copied / "dc.log").read_text(errors="replace").splitlines()
    errors = [(idx + 1, line) for idx, line in enumerate(log_lines)
              if re.match(r"^(Error|Fatal):", line)]
    if args.attack == "error_line":
        errors = [(33, errors[0][1])]
    require(errors == [(32, "Error: Error during sourcing of /opt/synopsys/syn/V-2023.12-SP3/auxx/gui/dv/.synopsys_dv.tcl")],
            "error/fatal classification mismatch")
    current = next(idx for idx, line in enumerate(log_lines)
                   if line.startswith("Current time:"))
    require(current + 1 > 32, "flow started before waived startup line")
    require(not any(re.match(r"^(Error|Fatal):", line)
                    for line in log_lines[current + 1:]),
            "in-flow Error/Fatal found")
    full_log = "\n".join(log_lines)
    require('no such variable\n    (read trace on "::env(HOME)")' in full_log,
            "HOME/dv.tcl root cause missing")
    require("Writing verilog file '" in full_log and "Writing ddc file '" in full_log,
            "netlist/DDC completion evidence missing")
    require("set_svf -off" in full_log and "Thank you..." in full_log,
            "normal DC shutdown evidence missing")

    setup = keyvals(copied / "reports/setup_posthold_summary_machine.txt")
    hold = keyvals(copied / "reports/hold_posthold_summary_machine.txt")
    require(setup == {
        "phase": "POST_RESTORE_REPORTED", "delay_type": "max", "status": "MET",
        "wns_ns": "0.002221110", "tns_ns": "0.000000000",
        "violating_paths": "0", "negative_path_ceiling": "200000",
    }, "setup summary mismatch")
    require(hold == {
        "phase": "POST_RESTORE_REPORTED", "delay_type": "min", "status": "MET",
        "wns_ns": "0.000999451", "tns_ns": "0.000000000",
        "violating_paths": "0", "negative_path_ceiling": "200000",
    }, "hold summary mismatch")
    area_text = (copied / "reports/area_posthold.rpt").read_text()
    area_match = re.search(r"^Total cell area:\s+([0-9.]+)$", area_text, re.M)
    require(area_match is not None, "area value missing")
    area = float(area_match.group(1))
    require(close(area, 152898.625984), "rederived area mismatch")
    macro = keyvals(copied / "reports/macro_binding_audit.txt")
    require(macro["status"] == "PASS_M1630_RESOLVED_LIBRARY_MACRO_STRUCTURE",
            "macro binding status mismatch")
    require(macro["macro_cell"] == "TS1N28HPCPHVTB128X128M4S",
            "macro cell mismatch")
    require(macro["macro_count_pre"] == "9" and macro["macro_count_post"] == "9" and
            macro["expected_macro_count"] == "9", "macro count mismatch")
    drc_text = (copied / "reports/constraint_design_rules_posthold.rpt").read_text()
    require(drc_text.count("This design has no violated constraints.") == 5,
            "DRC report mismatch")

    net = copied / "netlist"
    artifacts = {
        "ddc": net / "m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed.ddc",
        "svf": net / "m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed.svf",
        "sdc": net / "m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed_mapped.sdc",
        "mapped": net / "m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed_mapped.v",
    }
    artifact_hashes = {
        "ddc": "2a46429aefb9a772e1e77a7914449d052ad6f888af033d7413f8b03f3d2569b0",
        "svf": "7c15c1a30827df74c0da35f24f7e88723484c2a211edd3d6c049f52e21dec274",
        "sdc": "5ab21dbeb46baabf6e0bec2ea2a8f8542e114308e77ded25486fa022e4c3e198",
        "mapped": "842d100f6a3fc26684e13a8065191028af7840685aaf4b7cfa77a4fe998c46ee",
    }
    require({key: sha256(path) for key, path in artifacts.items()} == artifact_hashes,
            "DDC/SVF/SDC/mapped artifact identity mismatch")
    require(all(path.stat().st_size > 0 for path in artifacts.values()),
            "empty mapped artifact")
    mapped = artifacts["mapped"].read_text(errors="replace")
    require(mapped.rstrip().endswith("endmodule"), "mapped netlist is truncated")
    require(len(re.findall(r"\bTS1N28HPCPHVTB128X128M4S\b", mapped)) == 9,
            "mapped netlist macro count mismatch")

    expected_overhead = (area / 147246.392090 - 1.0) * 100.0
    require(receipt["status"] ==
            "PASS_RECOVERED_M1649_C1_RESIDUAL_HOLD_CLOSED_DC_CANDIDATE_PENDING_FORMALITY_PT_POWER",
            "receipt status mismatch")
    require(close(receipt["setup"]["wns_ns"], 0.002221110), "receipt setup WNS mismatch")
    require(close(receipt["setup"]["tns_ns"], 0.0), "receipt setup TNS mismatch")
    require(close(receipt["hold"]["wns_ns"], 0.000999451), "receipt hold WNS mismatch")
    require(close(receipt["hold"]["tns_ns"], 0.0), "receipt hold TNS mismatch")
    require(receipt["setup"]["violating_paths"] == 0 and
            receipt["hold"]["violating_paths"] == 0, "receipt timing violations")
    require(close(receipt["area"]["recovered_um2"], area), "receipt area mismatch")
    require(close(receipt["area"]["overhead_percent"], expected_overhead),
            "receipt area overhead mismatch")
    require(receipt["area"]["within_five_percent"] is True and expected_overhead < 5.0,
            "area overhead gate mismatch")
    require(receipt["macros"] == {"cell": "TS1N28HPCPHVTB128X128M4S", "count": 9},
            "receipt macro mismatch")
    require(receipt["drc_violating_nets"] == 0, "receipt DRC mismatch")
    require(receipt["input_generation"] == "original_m993_m1006_admitted_ddc",
            "receipt input generation mismatch")
    require(receipt["failed_m1614_output_used"] is False,
            "receipt used failed M1614 output")
    require(receipt["hold_only_incremental_mapping_count"] == 1,
            "receipt incremental map count mismatch")
    require(receipt["claim_boundary"] == {
        "dc_setup_hold_area_macro_drc_candidate": True,
        "formality": False, "independent_pt": False, "power": False,
        "energy": False, "cycle_speedup": False, "system_speedup": False,
        "paper_ppa_ready": False, "paper_citable": False, "headline": False,
    }, "claim boundary was widened")
    require(run_complete == {
        "status": receipt["status"], "source_failure_marker_preserved": "true",
        "dc_rerun": "false", "formality": "false", "independent_pt": "false",
        "power": "false", "energy": "false", "cycle_speedup": "false",
        "system_speedup": "false", "paper_ppa_ready": "false",
        "paper_citable": "false",
    }, "terminal receipt mismatch")

    # Copy-only evidence: exact reviewed source, zero-EDA release, sealed
    # one-shot attempt, byte-identical copied artifacts and no executable EDA
    # command in the recovery source.  This proves the recovery itself did not
    # rerun EDA; it is not a claim about unrelated host processes.
    release = load_json(paths["m1664_release_sha256"])
    require(release["authorization"] == {"copy_only_recoveries": 1, "all_eda_runs": 0},
            "release authorized EDA")
    recovery_source = paths["m1659_source_sha256"].read_text()
    executable_eda = re.findall(
        r"(?m)^\s*(?:env\s+[^\n]*\s+)?(?:dc_shell|fm_shell|pt_shell|vcs|simv|ptpx)\b",
        recovery_source)
    require(executable_eda == [], "recovery source contains executable EDA")
    require("cp -a --no-dereference" in recovery_source and "mv -T --" in recovery_source,
            "copy-only publish primitives missing")

    output = {
        "schema": "m1667_m1665_c1_canonical_recovery_result_hammer_output_r1_v1",
        "status": "PASS_M1667_M1665_C1_CANONICAL_RECOVERY_RESULT_HAMMER",
        "python": sys.version.split()[0],
        "source_quarantine_manifest_members": len(q_entries),
        "source_copy_all_files_byte_exact": source_copy_equal,
        "target_manifest_members": len(target_entries),
        "target_manifest_and_outer_seal_pass": True,
        "embedded_source_manifest_and_outer_seal_pass": True,
        "predecessor_identity_chain_pass": True,
        "dc_process_return_code": 0,
        "sole_error_line": 32,
        "sole_error_phase": "pre_flow_HOME_unset_dv_tcl_startup",
        "in_flow_error_or_fatal_count": 0,
        "setup_wns_ns": float(setup["wns_ns"]),
        "setup_tns_ns": float(setup["tns_ns"]),
        "hold_wns_ns": float(hold["wns_ns"]),
        "hold_tns_ns": float(hold["tns_ns"]),
        "area_um2": area,
        "area_overhead_percent": expected_overhead,
        "macro_count": int(macro["macro_count_post"]),
        "drc_violating_nets": 0,
        "ddc_svf_sdc_mapped_verilog_exact": True,
        "one_shot_copy_only_attempt": True,
        "eda_rerun_by_recovery": False,
        "claim_boundary_remains_non_citable_pending_formality_pt_power": True,
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("HAMMER_REJECT: " + str(exc), file=sys.stderr)
        sys.exit(2)
