#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent M1006 hammer for the copy-only M993 recovered C1 result."""

import hashlib
import json
import os
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
TARGET = HW / "dc_handoff/runs/m993_m989_m962_m935_macro_aware_dc_recovered_canonical_r1_20260829"
INNER = TARGET / "original_quarantine"
SOURCE = HW / "dc_handoff/runs/m962_m935_three_stage_match_macro_aware_dc_3p000ns_r1_20260829.failed_or_incomplete.3868703.quarantine"
ATTEMPT = HW / "dc_handoff/runs/.m993_m989_m962_copy_promotion_attempt_consumed"
RUNS = HW / "dc_handoff/runs"
SCRIPT = HW / "dc_handoff/scripts/promote_m989_m962_quarantine_atomic_one_shot_copy_only_r1.sh"
M991 = HW / "contracts/m991_m990_m989_atomic_one_shot_copy_only_promotion_release_r1_20260829.json"
M992 = HW / "reviews/m992_m991_m990_m989_atomic_one_shot_promotion_release_hammer_r1_20260829/review.json"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "target_manifest": "8aeda1372387692201badb90a7d81eb7d908f803c6cd652aab22dace5043d093",
    "target_outer": "0cc3b953342d6f149183e5fdf55b97174f69f97701574b0a79f05a5068ff6689",
    "inner_manifest": "9a1649638c0c2aa7b533fdb16cd763c87e6280dfc5a3c291240818cf1022eafe",
    "inner_outer": "a213df2a38ff231f9d0dbd78c379ef13b3731caf3b5335c37d6d17bf20927997",
    "attempt_manifest": "c7d5192ea52d5009478cd06c17a6a548faa17ac1f018ccc4af5124d6d6cf257a",
    "attempt_outer": "b553386d97054acfc468d9a9cbe218932051ef6ed3c526defa8f587a4006455f",
    "script": "7b63668f5fb68ac8d60acf4e43925313ab1c0bdc84caeefcbfb0e238871c4be9",
    "m991": "4cd6bd1777407dd0b5282713a12b945d53ea3cbcbb63ad0a2409d161c85992e7",
    "m992": "885495907ba90c9b7fa8a2d762c530ed85ba6ff19e3efdbe7d5ff0a32ceb2d08",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_sealed_directory(directory, manifest_sha, outer_sha,
                            exclude_nested_seal_names=False):
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink(), "directory missing/symlink")
    require(manifest.is_file() and outer.is_file() and
            not manifest.is_symlink() and not outer.is_symlink(), "seals missing/symlink")
    require(sha(manifest) == manifest_sha and sha(outer) == outer_sha,
            "seal identity drift")
    require(outer.read_text().split() == [manifest_sha, "SHA256SUMS"],
            "outer content drift")
    listed = {}
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(None, 1); rel = rel.lstrip("*")
        require(rel not in listed and ".." not in Path(rel).parts,
                "unsafe/duplicate manifest path")
        member = directory / rel
        require(member.is_file() and not member.is_symlink() and sha(member) == digest,
                "manifest member drift: " + rel)
        listed[rel] = digest
    actual = set()
    links = []
    for root, dirs, files in os.walk(directory, followlinks=False):
        root = Path(root)
        for name in list(dirs):
            if (root / name).is_symlink(): links.append(str((root / name).relative_to(directory)))
        dirs[:] = [name for name in dirs if not (root / name).is_symlink()]
        for name in files:
            path = root / name
            if path.is_symlink(): links.append(str(path.relative_to(directory))); continue
            if name in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
                if path.parent == directory or exclude_nested_seal_names: continue
            actual.add(str(path.relative_to(directory)))
    require(not links, "symlink in sealed tree")
    require(set(listed) == actual,
            "exact-set drift missing=%r extra=%r" %
            (sorted(set(listed) - actual), sorted(actual - set(listed))))
    return {"entries": len(listed), "manifest_sha256": sha(manifest),
            "outer_seal_file_sha256": sha(outer), "exact_set": True}


def require_float(text, pattern, expected, tolerance=1e-9):
    match = re.search(pattern, text, re.MULTILINE)
    require(match is not None, "missing numeric anchor: " + pattern)
    value = float(match.group(1))
    require(abs(value - expected) <= tolerance,
            "numeric anchor drift: %r != %r" % (value, expected))
    return value


def main():
    target_seal = verify_sealed_directory(
        TARGET, EXPECTED["target_manifest"], EXPECTED["target_outer"],
        exclude_nested_seal_names=True)
    inner_seal = verify_sealed_directory(
        INNER, EXPECTED["inner_manifest"], EXPECTED["inner_outer"])
    source_seal = verify_sealed_directory(
        SOURCE, EXPECTED["inner_manifest"], EXPECTED["inner_outer"])
    attempt_seal = verify_sealed_directory(
        ATTEMPT, EXPECTED["attempt_manifest"], EXPECTED["attempt_outer"])
    require(inner_seal == source_seal, "copied source seal differs from frozen source")
    require(sha(SCRIPT) == EXPECTED["script"] and sha(M991) == EXPECTED["m991"] and
            sha(M992) == EXPECTED["m992"] and sha(DOC359) == EXPECTED["docs359"],
            "authority/docs identity drift")

    provenance = json.loads((TARGET / "M993_PROMOTION_PROVENANCE.json").read_text())
    receipt = json.loads((TARGET / "m993_recovered_dc_receipt.json").read_text())
    require(provenance["status"] ==
            "COPY_ONLY_RECOVERY_OF_SYNTHESIS_COMPLETE_M962_QUARANTINE" and
            provenance["source_manifest_sha256"] == EXPECTED["inner_manifest"] and
            provenance["source_outer_seal_file_sha256"] == EXPECTED["inner_outer"] and
            provenance["mutation"] == {"source_modified": False,
                                       "runner_modified": False,
                                       "eda_rerun": False,
                                       "copied_payload_changed": False},
            "promotion provenance drift")
    require(provenance["identity"]["promotion_script_sha256"] == EXPECTED["script"] and
            provenance["identity"]["m991_release_sha256"] == EXPECTED["m991"] and
            provenance["identity"]["m992_hammer_sha256"] == EXPECTED["m992"],
            "promotion authority binding drift")
    require(provenance["runner_bug"] == {
        "cause": "env -i omitted HOME; nonfatal Design Vision startup Tcl error matched over-broad grep",
        "dc_shell_exit_code": 0, "log_hit_line": 32, "runner_exit_code": 9},
        "runner failure forensic drift")

    attempt = (ATTEMPT / "ATTEMPT_CONSUMED.txt").read_text()
    identity = (ATTEMPT / "IDENTITY.txt").read_text()
    require("status=M993_M989_COPY_PROMOTION_ATTEMPT_CONSUMED" in attempt and
            "max_promotions=1" in attempt and "retry=false" in attempt and
            ("promotion_script_sha256=" + EXPECTED["script"]) in identity and
            ("source_manifest_sha256=" + EXPECTED["inner_manifest"]) in identity,
            "attempt identity drift")
    absent = {
        "lock": not (RUNS / ".m993_m989_m962_copy_promotion_launch_lock").exists(),
        "work": not (RUNS / ".m993_m989_m962_copy_promotion_work").exists(),
        "failure_quarantine": not (RUNS / "m993_m989_m962_copy_promotion_failed_or_incomplete.quarantine").exists(),
    }
    require(all(absent.values()), "post-publication state drift")

    failure_marker = (INNER / "RUN_FAILED_OR_INCOMPLETE.txt").read_text()
    require(failure_marker ==
            "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=9\nsetup_admitted=false\n",
            "original FAILED marker not preserved exactly")
    require((INNER / "dc.rc").read_text().strip() == "0",
            "dc_shell exit code is not zero")

    setup_text = (INNER / "reports/setup_summary_machine.txt").read_text()
    macro_text = (INNER / "reports/macro_binding_audit.txt").read_text()
    area_text = (INNER / "reports/area_hierarchy.rpt").read_text()
    qor_text = (INNER / "reports/qor.rpt").read_text()
    top100_text = (INNER / "reports/timing_setup_top100.rpt").read_text()
    require("status=MET" in setup_text and "setup_violating_paths=0" in setup_text,
            "setup summary status drift")
    require_float(setup_text, r"^setup_wns_ns=([0-9.\-]+)$", 0.001795)
    require_float(setup_text, r"^setup_tns_ns=([0-9.\-]+)$", 0.0)
    require_float(setup_text, r"^clock_period_ns=([0-9.\-]+)$", 3.0)
    require("macro_cell=TS1N28HPCPHVTB128X128M4S" in macro_text and
            "macro_count_pre=9" in macro_text and "macro_count_post=9" in macro_text and
            "expected_macro_count=9" in macro_text and
            "behavioral_macro_verilog_read_by_dc=false" in macro_text and
            "inferred_parent_array_allowed=false" in macro_text,
            "macro binding anchor drift")
    require_float(area_text, r"Total cell area:\s+([0-9.]+)", 147246.392090, 1e-6)
    require_float(qor_text, r"Cell Area:\s+([0-9.]+)", 147246.392090, 1e-6)
    require(top100_text.count("  Path Group: core_clk") == 100 and
            top100_text.count("  slack (MET)") == 100 and
            "slack (VIOLATED)" not in top100_text,
            "top100 setup path anchor drift")

    expected_boundary = {
        "energy": False, "full_213376B_storage_integrated": False,
        "headline": False, "hold_signoff": False, "paper_ppa_ready": False,
        "power": False, "rtl_cycles_measured": False,
        "setup_area_component_candidate": True, "speedup": False,
        "system_speedup": False,
    }
    require(receipt["status"] ==
            "PASS_RECOVERED_RAW_M962_3NS_SETUP_AREA_COMPONENT_CANDIDATE" and
            receipt["claim_boundary"] == expected_boundary and
            receipt["clock_period_ns"] == 3.0 and receipt["ideal_clock"] is True and
            receipt["wireload"] == "ZeroWireload" and
            receipt["macro_cell"] == "TS1N28HPCPHVTB128X128M4S" and
            receipt["macro_count"] == 9 and
            receipt["total_cell_area_um2_dc_reported"] == 147246.39209 and
            receipt["setup"] == {"met": True, "wns_ns": 0.001795,
                                  "tns_ns": 0.0, "violating_paths": 0,
                                  "top100_reported_paths": 100},
            "recovered receipt/report boundary drift")

    return {
        "schema": "m1006_m993_recovered_c1_component_result_hammer_v1",
        "status": "PASS_M1006_M993_RECOVERED_C1_COMPONENT_RESULT_HAMMER",
        "verdict": "GO_C1_3NS_SETUP_AREA_COMPONENT_CANDIDATE_ONLY",
        "score_out_of_100": 99,
        "p0_count": 0, "p1_count": 0, "p2_count": 1,
        "seals": {"target": target_seal, "original_quarantine": inner_seal,
                  "attempt": attempt_seal},
        "state": {"attempt_consumed_and_sealed": True, **absent},
        "anchors": {
            "dc_shell_exit_code": 0, "runner_exit_code": 9,
            "clock_period_ns": 3.0, "setup_met": True,
            "setup_wns_ns": 0.001795, "setup_tns_ns": 0.0,
            "setup_violating_paths": 0, "top100_met_paths": 100,
            "macro_cell": "TS1N28HPCPHVTB128X128M4S", "macro_count": 9,
            "total_cell_area_um2": 147246.39209,
            "original_failed_marker_preserved": True,
        },
        "decision": {
            "component_setup_area_candidate_admitted": True,
            "rtl_cycle_speedup_admitted": False,
            "cpu_same_ledger_1p746753_promoted_to_rtl_cycle": False,
            "full_storage_system_ppa_admitted": False,
        },
        "p2": [{"id": "P2_HOLD_POWER_FULL_STORAGE_REMAIN_OPEN",
                "finding": "Hold has 9992 diagnostic violations (worst -0.09 ns); power/energy and full 213376-B storage are not integrated."}],
        "scope": {"read_only_result_hammer": True, "eda_runs": 0,
                  "target_modified": False, "docs359_modified": False},
        "claim_boundary": expected_boundary,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
