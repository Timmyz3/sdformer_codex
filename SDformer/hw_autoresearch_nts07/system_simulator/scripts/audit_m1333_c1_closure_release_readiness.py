#!/usr/bin/env python3
"""Read-only C1 closure/release audit.  This script launches no EDA."""
import argparse
import hashlib
import json
import re
from pathlib import Path

HW = Path(__file__).resolve().parents[2]
EXPECTED = {
    "reviews/m879_m863_c1_r21_unit_delay_vcs_result_hammer_r1_20260829/review.json": "54daccf7be1deab09edc1ebead9f478777ad0de474129c533e53f0280f345754",
    "reviews/m1006_m993_m989_m962_recovered_c1_component_result_hammer_r1_20260829/review.json": "d7b30ff3a82a099c080f3aa3dd32c13c1d2d5b5e278112eb9e3b1c24588809ea",
    "contracts/m1116c_m1114_m1006_m963_m959_m935_full_storage_common_charge_source_contract_r1_20260830.json": "82b7f1b6faea7e39f03f32c1bc1fbd924259147a8e2d5d9c58516c41646e7e30",
    "reviews/m1160_m1116c_c1_full_storage_common_charge_source_independent_hammer_r1_20260830/review.json": "418980de0deddf2cb223b813d1372dc9d61f51bb230c20d3c9405cf219ba30a4",
    "contracts/m1162_m1160_m1116c_c1_common_charge_protocol_repair_source_contract_r1_20260830.json": "5787f3302aa3308485e357c41385e69da93e6b41bfdea92410690af5a95ecbdc",
    "reviews/m1166_m1162_c1_common_charge_protocol_repair_independent_hammer_r1_20260830/review.json": "7f2cdf4cb1f979c0680b491c27c1088bc35624a2fd801b97c304c5b403076b4c",
    "contracts/m1265_c1_r12_exact_byte_vcs_launch_release_r1_20260830.json": "2ee20e2a773ab3c778fa09758f052b16538795903e4830122a0db8f2c6f0e022",
    "reviews/m1266_m1265_c1_r12_exact_byte_vcs_release_independent_hammer_r1_20260830/review.json": "8e82f72545f8027e203ff0c0221a2c9cf48dd127aba4ee6b1a8fd28bbde2ba78",
    "results/.m1265_m1258r12_m1162_c1_common_charge_protocol_vcs_r12_attempt_consumed/identity.txt": "9a8f73b89d59ac6c2a7f0aec2443ffbc0190ecd5daf7e6985e3abe34579a7104",
    "results/m1265_m1258r12_m1162_c1_common_charge_protocol_unit_delay_vcs_r12_20260830.failed_or_incomplete.2521081.quarantine/failure_phase_and_timeout_dump.txt": "adb8a354917224dd251fa9516045dfecea467f50c52aecb89e74f8f8ea71b160",
    "results/m1265_m1258r12_m1162_c1_common_charge_protocol_unit_delay_vcs_r12_20260830.failed_or_incomplete.2521081.quarantine/sim.log": "dbc340e87ea4cea4dda4e27f174cb5acb23fde324ea2380b630f127e84870dc0",
    "reviews/m1268_m1265_c1_r12_nonfirst_psum_failure_forensic_r1_20260830/review.json": "39102bd7214e430eb517a104062c754b0018e88e2ba62dbcd9323adb8702287b",
    "contracts/m1270_c1_r13_real_m935_integrated_protocol_source_contract_r1_20260830.json": "f17a02226b4d8a391d6cbb5830e16f7e0716b7a9f1e342457add79e0438e15ee",
    "reviews/m1273_m1272_m1270_c1_r13_checker_final_independent_hammer_r1_20260830/review.json": "caf61dd7de32f546e0c0e681b020c8717e8a4aca536ab17df62854483dc4749a",
    "reviews/m1275_c1_memory_energy_admission_audit_r1_20260830/review.json": "0217142ade5b04c6a31d1d24e2796b58af8937203a652ab46c73b9cb13a44520",
    "docs/359_DATE终局冻结_20260813.md": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(rel):
    return json.loads((HW / rel).read_text())


def must(condition, message):
    if not condition:
        raise SystemExit("FAIL_M1333_STATIC_AUDIT: " + message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checked = {}
    for rel, expected in EXPECTED.items():
        path = HW / rel
        must(path.is_file() and sha(path) == expected, "identity drift: " + rel)
        checked[rel] = expected

    m879 = load("reviews/m879_m863_c1_r21_unit_delay_vcs_result_hammer_r1_20260829/review.json")
    must(m879["status"] == "PASS100_M863_C1_R21_SYNOPSYS_VCS_E3_FUNCTIONAL_RESULT_ADMITTED",
         "core VCS authority drift")
    must(m879["claim_boundary"]["rtl_cycle_speedup_verified"] is False,
         "core VCS falsely promoted")
    m1006 = load("reviews/m1006_m993_m989_m962_recovered_c1_component_result_hammer_r1_20260829/review.json")
    must(m1006["anchors"]["setup_met"] is True
         and m1006["anchors"]["macro_count_pre_post_expected"] == [9, 9, 9],
         "M1006 component point drift")
    must(m1006["claim_boundary"]["hold_signoff"] is False
         and m1006["claim_boundary"]["power"] is False
         and m1006["claim_boundary"]["full_213376B_storage_integrated"] is False,
         "M1006 boundary drift")

    m1116 = load("contracts/m1116c_m1114_m1006_m963_m959_m935_full_storage_common_charge_source_contract_r1_20260830.json")
    exact = m1116["exact_once_capacity_boundary"]
    must(exact["represented_ledger_bytes"] == 214912
         and exact["internal_foundry_macro_bytes"] == 18432
         and exact["external_common_charge_bytes"] == 196480,
         "M1116C storage split drift")
    must(exact["full_214912B_physically_integrated"] is False,
         "M1116C falsely physicalized full storage")
    m1160 = load("reviews/m1160_m1116c_c1_full_storage_common_charge_source_independent_hammer_r1_20260830/review.json")
    must(m1160["status"].startswith("STOP_M1160"), "old common-charge wrapper no longer stopped")
    m1166 = load("reviews/m1166_m1162_c1_common_charge_protocol_repair_independent_hammer_r1_20260830/review.json")
    must(m1166["authorization"]["direct_vcs_run_on_current_tb"] is False
         and m1166["authorization"]["dc_now"] is False,
         "M1162 source hammer unexpectedly authorizes execution")

    attempt = (HW / "results/.m1265_m1258r12_m1162_c1_common_charge_protocol_vcs_r12_attempt_consumed/identity.txt").read_text()
    must("automatic_retry=false" in attempt, "R12 attempt consumption drift")
    m1268 = load("reviews/m1268_m1265_c1_r12_nonfirst_psum_failure_forensic_r1_20260830/review.json")
    must(m1268["status"] == "PASS_FORENSIC__TB_CHILD_OUTPUT_SEAM_FAILURE__NOT_RTL_EXTRA_PSUM_EVIDENCE",
         "R12 forensic boundary drift")
    must(m1268["unique_failure"]["r12_functional_pass"] is False
         and m1268["verdict"]["retry_authorized"] is False,
         "R12 unexpectedly reusable")

    m1270 = load("contracts/m1270_c1_r13_real_m935_integrated_protocol_source_contract_r1_20260830.json")
    must(m1270["launch_authorized"] is False, "R13 source unexpectedly launchable")
    m1273 = load("reviews/m1273_m1272_m1270_c1_r13_checker_final_independent_hammer_r1_20260830/review.json")
    must(m1273["status"] == "SOURCE_NO_GO__NO_RELEASE_NO_VCS__CHECKER_EXPANSION_STOPPED"
         and m1273["vcs_authorized"] is False,
         "R13 final hammer boundary drift")

    contracts = [p.name for p in (HW / "contracts").glob("*.json")]
    r13_or_later_releases = sorted(name for name in contracts
        if re.search(r"m12(?:7[4-9]|[89][0-9])|m13[0-9][0-9]", name)
        and "c1" in name.lower() and ("release" in name.lower() or "launch" in name.lower()))
    full_storage_releases = sorted(name for name in contracts
        if ("m1116c" in name.lower() or "m1162" in name.lower())
        and ("dc" in name.lower() or "pt" in name.lower())
        and ("release" in name.lower() or "launch" in name.lower()))
    must(not r13_or_later_releases, "new C1 release appeared; refresh audit")
    must(not full_storage_releases, "full-storage DC/PT release appeared; refresh audit")

    m1275 = load("reviews/m1275_c1_memory_energy_admission_audit_r1_20260830/review.json")
    must(m1275["verdict"]["candidate_vs_baseline_c1_energy_ratio_now"] is False
         and m1275["verdict"]["total_c1_energy_now"] is False,
         "C1 energy boundary drift")

    out = {
        "schema": "m1333_c1_closure_release_readiness_readonly_audit_r1_v1",
        "status": "NO_GO_DIRECT_C1_VCS_DC_PT__NO_UNCONSUMED_ADMITTED_RELEASE",
        "score": 100,
        "eda_launched": False,
        "checked_identity": checked,
        "release_inventory": {
            "unconsumed_admitted_wrapper_vcs_release": None,
            "r13_or_later_c1_launch_release_contracts": r13_or_later_releases,
            "full_storage_dc_pt_launch_release_contracts": full_storage_releases,
            "direct_command": None,
            "direct_namespace": None,
            "collision_gate_applicable_now": False,
        },
        "evidence_split": {
            "core_vcs": {
                "status": "PASS",
                "scope": "M528 dead-write-only 1RW core, foundry UNIT_DELAY functional",
                "rtl_cycle_speedup": False,
            },
            "common_charge_wrapper_vcs": {
                "status": "FAIL_CLOSED",
                "latest_consumed_attempt": "M1265 R12",
                "failure": "TB child-output seam at directed non-first phase; not proof of RTL extra psum",
                "retry": False,
                "r13_status": "source NO-GO; no release/VCS",
            },
            "full_storage_214912B": {
                "represented_ledger_bytes": 214912,
                "physically_integrated_bytes": 18432,
                "external_common_charge_bytes": 196480,
                "numeric_external_area_energy": False,
                "full_storage_dc_pt_release": False,
            },
            "physical_component": {
                "source": "M1006",
                "area_um2": 147246.39209,
                "setup_wns_ns": 0.001795,
                "parent_macros": 9,
                "hold_wns_ns": -0.09,
                "hold_closed": False,
                "power": False,
                "energy": False,
            },
        },
        "minimum_p0_gaps": [
            "No admitted wrapper-level VCS PASS through real frozen M935 and M1162; R12 is consumed failed and R13 has no source GO or release.",
            "No executable full-storage DC/PT top binds repaired M1162 while accounting every 214912 bytes; only 18432 bytes are physically integrated and 196480 bytes remain external common charge.",
            "No fast-view hold closure for the current wrapper/full-storage identity and no matched candidate/baseline SAIF/PTPX power evidence.",
        ],
        "unique_next_source": {
            "name": "additive R14 real-M935 runtime-witness wrapper VCS source package",
            "why_first": "Wrapper functional admission is a prerequisite for any repaired full-storage DC/PT source; launching physical tools before it would synthesize an unverified boundary.",
            "must_freeze": ["M528", "M935", "M1162", "R3 SVA", "214912-byte common-charge ledger"],
            "must_replace": "Only the R13 verification/control-flow proof surface, not design RTL",
            "required_structure": [
                "one unconditional real-M935 two-beat path with natural first and non-first beats",
                "a small monotonic runtime witness FSM/counter block rather than further regex-only checker expansion",
                "zero issue-request force/assignment seam",
                "operand-complete fatal oracle and exact runtime counts for 2 weight, 1 psum, 2 issue accepts, 1 commit/row/task completion",
                "fresh exact filelist, static checker/tests, independent source hammer; release authored only after source GO",
            ],
            "does_not_authorize_execution": True,
        },
        "forbidden_now": [
            "retry M1265/R12",
            "execute M1116C candidate DC Tcl/filelist",
            "relabel M1006 as 214912-byte full-storage PPA",
            "quote 1.759x as RTL cycle speedup",
            "run PT/PTPX from component netlist",
        ],
        "claim_boundary": {
            "core_functional_vcs": True,
            "wrapper_functional_vcs": False,
            "full_storage_dc_pt": False,
            "hold": False,
            "power": False,
            "energy": False,
            "rtl_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
        "docs359_sha256": EXPECTED["docs/359_DATE终局冻结_20260813.md"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(out["status"])


if __name__ == "__main__":
    main()
