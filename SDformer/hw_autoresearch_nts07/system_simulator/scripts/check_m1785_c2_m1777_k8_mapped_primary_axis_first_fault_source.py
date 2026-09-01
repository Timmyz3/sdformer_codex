#!/usr/bin/env python3
"""CPU-only M1785 source/failure checker; never launches EDA."""

from __future__ import print_function

import hashlib
import json
from pathlib import Path
import re
import stat


HW = Path(__file__).resolve().parents[2]
TB = HW / "dc_handoff/tb/tb_m1785_c2_m1777_k8_mapped_primary_axis_first_fault_diagnostic.sv"
FILELIST = HW / "dc_handoff/filelists/iscas_m1785_c2_m1777_k8_mapped_primary_axis_first_fault_diagnostic.f"
RUNNER = HW / "dc_handoff/scripts/run_m1785_c2_m1777_k8_mapped_primary_axis_first_fault_one_shot.py"
TEST = HW / "system_simulator/tests/test_m1785_c2_m1777_k8_mapped_primary_axis_first_fault_source.py"
CONTRACT = HW / "contracts/m1785_c2_m1777_k8_mapped_primary_axis_first_fault_source_contract_r1_20260902.json"
FAILURE = HW / "results/m1777_c2_two_axis_equal_bandwidth_energy_r1_20260902.failed_or_incomplete.quarantine"
PRIVATE = HW / "results/m1777_c2_two_axis_equal_bandwidth_energy_r1_20260902.private_build.unsealed_do_not_cite"
NETLIST = HW / "dc_handoff/runs/m1661_m1652_c2_resource_gate_successor_three_axis_logic_only_dc_3p000ns_r1_20260901/k8/netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v"
M1684_ASSERT = HW / "dc_handoff/tb/m1684_c2_m1609_production_binary_fault_assertions.sv"
M1684_WRAPPER = HW / "dc_handoff/tb/tb_m1684_c2_m1609_fresh_mapped_production_energy.sv"
M979_TB = HW / "dc_handoff/tb/tb_m979_c2_three_axis_mapped_gate_case_saif.sv"
MEMORY = HW / "dc_handoff/tb/m1334_c2_production_activity_reset_safe_memory_model.sv"
M1609 = HW / "rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv"
M1613_RECEIPT = HW / "results/m1613_c2_m1609_registered_fault_directed_vcs_r1_20260901/receipt.json"
M1594_REVIEW = HW / "reviews/m1594_m1593_c2_first_fault_independent_cone_review_r1_20260901/review.json"
M1606_REVIEW = HW / "reviews/m1606_m1604_c2_settled_result_semantics_independent_review_r1_20260901/review.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

PINS = {
    FAILURE / "failure.json": "6b1a135ea47161d299abdc804428be4caea8c421cdba1fdd65115abfdbd23570",
    FAILURE / "SHA256SUMS": "c99303b4362f62b4e6dde5c628e69cefb64344a8e22f349942e37a6e09d5bc1f",
    PRIVATE / "candidate/k8_case0.log": "00a2fb2b0073a4df364e9cbf0c35286f2979fb1426c35a6254007ae7f5770831",
    PRIVATE / "candidate/k8_case0.assert.report": "c7eee3a8bbae873e092c90b3065a6502963a21931b686110ab0d9c3e1ac7feca",
    PRIVATE / "build/k8/compile.log": "8c080598dd75f9c0119fee265ed5d3e10ecec68e32d416ce4319b50258938135",
    NETLIST: "6c62d99b444ba25f8eb3f1e491479b44f5613b0323e032af8150e81c84f393c4",
    M1684_ASSERT: "39fdc0f47628272a6f1a7b6887da52fdbf4d71f1f5fe6557d4a7022f06bc62b1",
    M1684_WRAPPER: "034934d1cdb6dc683ffa51811bd363fadd02673a1311a5b715b5c4b0e3cb5a2e",
    M979_TB: "cce12a93c4c8fd8d424fbf9f6354ba30e2870a05a7480fc7de26b3b29c87266c",
    MEMORY: "f9b0d87dd3b951a24b79545555c09b32bbce695e85cc71df2948e5065981c7c3",
    M1609: "7ee28b3912ae34c99c795a48e80be29df2b59b363e5de2d2b359175ec9dda931",
    M1613_RECEIPT: "1d11ba89467dd6a6475b70c4dfaa899ed06e44540608c7345593e1e8a537e438",
    M1594_REVIEW: "97370ae3eeae00ad79e3647b6ec34df3e114c88b807d773a69b250b0cfac324e",
    M1606_REVIEW: "5fe04f768e23fee6e57b150c14459b4a5f14bc407aeaec5b040959183935a8a9",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class Failure(RuntimeError):
    pass


def need(value, message):
    if not value:
        raise Failure(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            need(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=pairs,
                         parse_constant=lambda token: (_ for _ in ()).throw(
                             Failure("nonfinite JSON: " + token)))


def verify_regular(path, expected=None):
    path = Path(path)
    need(path.is_file() and not path.is_symlink()
         and stat.S_ISREG(path.lstat().st_mode), "nonregular: " + str(path))
    if expected is not None:
        need(sha256(path) == expected, "identity drift: " + str(path))


def verify_sealed_failure():
    verify_regular(FAILURE / "SHA256SUMS", PINS[FAILURE / "SHA256SUMS"])
    outer = FAILURE / "SHA256SUMS.seal.sha256"
    verify_regular(outer)
    need(outer.read_text(encoding="utf-8").strip()
         == PINS[FAILURE / "SHA256SUMS"] + "  SHA256SUMS",
         "failure outer seal")
    manifest = (FAILURE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    need(manifest == [PINS[FAILURE / "failure.json"] + "  failure.json"],
         "failure population/manifest")
    failure = strict_json(FAILURE / "failure.json")
    need(failure["status"] == "FAILED_OR_INCOMPLETE", "failure status")
    need(failure["phase"] == "SIM_k8_0", "failure phase")
    need(failure["counts"] == {"ptpx_runs": 0, "saif_files": 0,
         "simv_runs": 1, "vcs_compiles": 1}, "failure counts")
    need(failure["attempt_consumed"] is True, "attempt not consumed")
    need(failure["automatic_retry"] is False, "automatic retry")
    need(failure["partial_axis_citable"] is False, "partial axis citable")
    return failure


def audit_tb(text):
    required = [
        "tb_m1684_c2_m1609_fresh_mapped_production_energy sealed();",
        "m1785_c2_public_first_fault_monitor diagnostic (",
        "#1ps;",
        "always @(protocol_error or numeric_overflow or stale_response_seen",
        "M1785_FIRST_UNKNOWN code=%0d class=%s field=%s time_ps=%0t",
        "M1785_SETTLED_TRACE sample=%0d",
        "M1785_FINAL first_unknown_seen=%0d",
        "exact_m1684_assertion_preserved=1 initreg=0 force=0 ignore_x=0",
        "flag_unknown(1, \"FAULT\", \"protocol_error\")",
        "flag_unknown(2, \"FAULT\", \"numeric_overflow\")",
        "flag_unknown(3, \"FAULT\", \"stale_response_seen\")",
        "flag_unknown(37, \"DIAGNOSTIC_TAP\"",
        ".endpoint_fault(sealed.endpoint_fault)",
        ".registered_fault_taps(registered_fault_taps)",
    ]
    for token in required:
        need(text.count(token) == 1, "TB token count: " + token)
    need(text.count("flag_unknown(4 + bank") == 1, "endpoint fault loop")
    for code in range(12, 37):
        need(text.count("flag_unknown(" + str(code) + ",") == 1,
             "public field code: " + str(code))
    need("force " not in text.lower(), "active force in TB")
    need("initreg" not in text.lower().replace("initreg=0", ""),
         "initreg in TB")
    need("ignore_x=1" not in text and "ignorex" not in text.lower(),
         "X suppression in TB")
    need("$isunknown" in text and "=== 1'b1" in text,
         "four-state observation missing")


def audit_filelist(text):
    rows = [row.strip() for row in text.splitlines() if row.strip()]
    need(len(rows) == 10, "filelist length")
    need(rows[:2] == ["+define+M979_AXIS_K8", "+define+SVA_RUNTIME_ENABLED"],
         "K8/SVA defines")
    need(rows[-1].endswith(
        "tb_m1785_c2_m1777_k8_mapped_primary_axis_first_fault_diagnostic.sv"),
        "diagnostic TB last")
    for required in (str(NETLIST), str(MEMORY), str(M979_TB),
                     str(M1684_ASSERT), str(M1684_WRAPPER)):
        need(rows.count(required) == 1, "filelist member: " + required)
    need(not any("rtl_m" in row for row in rows), "RTL mixed into mapped run")


def audit_runner(text):
    required = [
        "M1785_EXPECTED_RUNNER_SHA256",
        "M1785_EXPECTED_M1786_REVIEW_SHA256",
        "PASS_M1786_M1785_C2_K8_MAPPED_FIRST_FAULT_SOURCE_HAMMER__AUTHORIZE_ONE_ATTEMPT",
        "ATTEMPT.mkdir()",
        "automatic_retry\": False",
        "tb_m1785_c2_m1777_k8_mapped_primary_axis_first_fault_diagnostic",
        "+M979_CASE=0",
        "checker.check_runtime_text(text)",
        "vcs_compiles\": 1",
        "simv_runs\": 1",
        "saif_files\": 0",
        "ptpx_runs\": 0",
        "/tmp/date_dual_synopsys_same_uid_eda_queue.lock",
    ]
    for token in required:
        need(token in text, "runner token: " + token)
    for forbidden in ("power -enable", "-ucli", "+vcs+initreg"):
        need(forbidden not in text.lower(), "runner forbidden: " + forbidden)
    need(not re.search(r"(?im)(?:^|[;{])\s*force\s+", text),
         "runner active force")
    need("subprocess.run(compile_command" in text
         and "subprocess.run(sim_command" in text,
         "only named compile/sim commands may launch")
    need(text.count("subprocess.run(compile_command") == 1,
         "compile budget source")
    need(text.count("subprocess.run(sim_command") == 1,
         "sim budget source")


def check_runtime_text(text):
    need(text.count("M1684 mapped fault vector contains X/Z") == 1,
         "exact M1684 X/Z failure absent/duplicated")
    need("PASS M1684 M1609 binary-clean production case=" not in text,
         "unexpected M1684 pass")
    rows = re.findall(
        r"^M1785_FIRST_UNKNOWN code=(\d+) class=([A-Z_]+) field=([^ ]+) time_ps=(\d+)$",
        text, flags=re.MULTILINE)
    need(rows, "M1785 field localization absent")
    final = re.findall(
        r"^M1785_FINAL first_unknown_seen=1 first_unknown_code=(\d+) first_unknown_time_ps=(\d+) settled_samples=(\d+) exact_m1684_assertion_preserved=1 initreg=0 force=0 ignore_x=0$",
        text, flags=re.MULTILINE)
    need(len(final) == 1, "M1785 final absent/duplicated")
    codes = [int(row[0]) for row in rows]
    need(int(final[0][0]) in codes, "first code not reported")
    need(int(final[0][2]) > 0, "no settled samples")
    public = [code for code in codes if code <= 36]
    need(public, "only private diagnostic tap localized")
    first = int(final[0][0])
    if first == 1:
        classification = "PROTOCOL_ERROR_PUBLIC_XZ"
    elif first == 2:
        classification = "NUMERIC_OVERFLOW_PUBLIC_XZ"
    elif first == 3:
        classification = "STALE_RESPONSE_PUBLIC_XZ"
    elif 4 <= first <= 11:
        classification = "ENDPOINT_FAULT_PUBLIC_XZ"
    elif 12 <= first <= 16:
        classification = "SOURCE_PUBLIC_XZ"
    elif 17 <= first <= 22:
        classification = "ENDPOINT_INTERFACE_PUBLIC_XZ"
    elif 23 <= first <= 27:
        classification = "RESULT_PUBLIC_XZ"
    elif 28 <= first <= 31:
        classification = "DONE_PUBLIC_XZ"
    elif 32 <= first <= 36:
        classification = "STATUS_PUBLIC_XZ"
    else:
        classification = "PRIVATE_TAP_FIRST_REQUIRES_PUBLIC_FOLLOWUP"
    return {"status": "PASS_M1785_DIAGNOSTIC_LOCALIZATION",
            "classification": classification,
            "first_unknown_code": first,
            "first_unknown_time_ps": int(final[0][1]),
            "reported_unknown_codes": codes,
            "settled_samples": int(final[0][2])}


def main():
    for path, expected in PINS.items():
        verify_regular(path, expected)
    failure = verify_sealed_failure()
    audit_tb(TB.read_text(encoding="utf-8"))
    audit_filelist(FILELIST.read_text(encoding="utf-8"))
    audit_runner(RUNNER.read_text(encoding="utf-8"))
    contract = strict_json(CONTRACT)
    need(contract["status"] ==
         "SOURCE_ONLY_READY_FOR_M1786_HAMMER__NO_EDA_NO_ATTEMPT_NO_CLAIM",
         "contract status")
    need(contract["future_execution"]["authorized_now"] is False,
         "future execution prematurely authorized")
    rows = contract["source_files"]
    need(len(rows) == 5, "source inventory length")
    for row in rows:
        path = HW / row["path"]
        verify_regular(path, row["sha256"])
    need(contract["claim_boundary"] == {
        "diagnostic_only": True, "energy": False, "headline": False,
        "mapped_functionality": False, "paper_citable": False,
        "performance": False, "power": False, "system_speedup": False,
        "timing_verified": False}, "claim boundary")
    return {"schema": "m1785_c2_m1777_k8_first_fault_source_check_r1_v1",
            "status": "PASS_M1785_SOURCE_ONLY_NO_EDA_NO_ATTEMPT",
            "m1777_failure_phase": failure["phase"],
            "m1777_counts": failure["counts"],
            "m1777_root_cause": "M1684_K8_PUBLIC_FAULT_VECTOR_XZ_AT_27000PS",
            "checker_traceback_is_downstream": True,
            "mapped_netlist_sha256": PINS[NETLIST],
            "docs359_sha256": PINS[DOCS359],
            "future_vcs_compiles": 1, "future_simv_runs": 1,
            "eda_runs_now": 0}


if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True, separators=(",", ":")))
