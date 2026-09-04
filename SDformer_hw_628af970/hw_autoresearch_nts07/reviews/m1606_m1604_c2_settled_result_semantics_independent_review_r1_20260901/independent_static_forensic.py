#!/usr/bin/env python3
"""Read-only M1604 result audit and synchronous ready/valid semantic proof."""

from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RESULT = HW / "results/m1604_c2_rtl_mapped_k8_case0_settled_first_fault_r1_20260901"
ATTEMPT = HW / "results/.m1604_c2_rtl_mapped_k8_case0_settled_first_fault_attempt_consumed"
COMPILE = RESULT / "compile.log"
SIM = RESULT / "sim.log"
SIMV = RESULT / "simv"
FILELIST = HW / "dc_handoff/filelists/date_m1601_c2_rtl_vs_mapped_k8_case0_settled_first_fault_source.f"
TB = HW / "dc_handoff/tb/tb_m1601_c2_rtl_vs_mapped_k8_case0_settled_first_fault.sv"
COMPACTOR = HW / "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv"
FRONTEND = HW / "rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv"
CORE = HW / "rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv"
K8 = HW / "rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv"
MAPPED = HW / ("dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829/"
               "k8/netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v")
MEMORY = HW / "dc_handoff/tb/m1334_c2_production_activity_reset_safe_memory_model.sv"
M1603 = HW / "reviews/m1603_m1601_c2_settled_first_fault_source_independent_review_r1_20260901"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    COMPILE: "69629f31870743d66df51b32ee7ad713e12a11b5a4803d8aa118e41cf92e7245",
    SIM: "75343367517c7a232545002ab62620e3373a7e3f97641e3162d4d0569aa4956d",
    SIMV: "36fbb9607802e7c97da0262854e58606c472bcbe17560367267e3a89baf1e84e",
    FILELIST: "b6e384a3b7de9541a66af0302722c9ae9ca12b50e5e57a1ac764bf1576a39a53",
    TB: "3e8a9254fd9104aeeb4d3f05077a9f2b8ae33a9617d3236447108a5b666ba8e4",
    COMPACTOR: "e278da8b0deaa0dda07b0477930453daa40b0331399a3941b743d604d0b102a5",
    FRONTEND: "529e463802fec72716ac6592d31e7668104a5463ff92499a98ec7314c8e88267",
    CORE: "010fe9e6786db1d3bbcad7759bda17a783ce5cfe15cae02c5b4c9ebf96e9950b",
    K8: "2588f890213d29aab6829dff679719c0f9ce4762c17bb061d1869b27a2f1d50e",
    MAPPED: "6b745030df6c041a0501d041ee277459c726c52263b4eec6ab5712f14d156de5",
    MEMORY: "f9b0d87dd3b951a24b79545555c09b32bbce695e85cc71df2948e5065981c7c3",
    M1603 / "review.json": "ade8ea24cc96b768e8c6dac01d091519440b3915a333178455380c6833bf0a49",
    M1603 / "SHA256SUMS": "497388480c66519e51f2066a86ddc4f504cd0608e47df832d28c9012be6a2d9b",
    M1603 / "SHA256SUMS.seal.sha256": "d3df0f82da13086cd69f6e8cfbff0a8dc569e802fda7b43b4332e7085910cf83",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

TOP = "tb_m1601_c2_rtl_vs_mapped_k8_case0_settled_first_fault"


class ForensicError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ForensicError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(text):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          ForensicError("nonfinite JSON: " + token)))


def normalize_command(lines, index):
    row = lines[index]
    while row.rstrip().endswith("\\"):
        index += 1
        require(index < len(lines), "unterminated command continuation")
        row = row.rstrip()[:-1] + " " + lines[index].strip()
    return " ".join(row.split())


def verify_sealed_m1603():
    manifest = M1603 / "SHA256SUMS"
    outer = M1603 / "SHA256SUMS.seal.sha256"
    require(outer.read_text(encoding="ascii").split() ==
            [EXPECTED[manifest], "SHA256SUMS"], "M1603 outer content drift")
    expected = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, name = line.split(None, 1)
        name = name.strip(); rel = Path(name)
        require(name not in expected and name == rel.as_posix() and
                not rel.is_absolute() and ".." not in rel.parts,
                "unsafe M1603 manifest row")
        expected[name] = digest
    actual = set()
    for member in M1603.rglob("*"):
        rel = member.relative_to(M1603).as_posix()
        if rel in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "M1603 symlink")
        if stat.S_ISREG(mode):
            actual.add(rel)
        else:
            require(stat.S_ISDIR(mode), "M1603 special member")
    require(actual == set(expected), "M1603 member-set drift")
    for name, digest in expected.items():
        require(sha256(M1603 / name) == digest, "M1603 member drift: " + name)
    review = strict_json((M1603 / "review.json").read_text(encoding="utf-8"))
    require(review["status"] ==
            "PASS_M1603_M1601_SETTLED_SOURCE__ONE_NEW_IDENTITY_COMPILE_AND_CASE0_SIM_AUTHORIZED__NOT_EXECUTED",
            "M1603 status drift")
    auth = review["authorization"]
    require(auth["result_identity"] == RESULT.name and auth["vcs_compiles"] == 1 and
            auth["simv_runs"] == 1 and auth["case"] == "k8_case0" and
            auth["retry"] is False, "M1603 execution authority drift")
    return {"full_tree_sealed": True, "exact_result_identity": True,
            "compile_budget": 1, "simulation_budget": 1}


def audit_compile(text, filelist_rows):
    lines = text.splitlines()
    rows = [i for i, row in enumerate(lines) if row.startswith("Command: vcs ")]
    require(len(rows) == 1, "compile command count is not one")
    command = normalize_command(lines, rows[0])
    require(("-top " + TOP) in command and
            "-f dc_handoff/filelists/" + FILELIST.name in command,
            "executed top/filelist drift")
    lower = command.lower()
    for token in ("ucli", "initreg", "saif", "ptpx"):
        require(token not in lower, "prohibited compile option: " + token)
    parsed = re.findall(r"^Parsing design file '([^']+)'", text, flags=re.M)
    require(parsed == filelist_rows, "compile parsing order drift")
    require("Top Level Modules:\n       " + TOP in text and
            "TimeScale is 1 ns / 1 ps" in text and
            "71 modules and 3 UDPs read." in text,
            "compile elaboration receipt drift")
    require(len(re.findall(r"^Warning-\[", text, flags=re.M)) == 4 and
            len(re.findall(r"^Warning-\[TFIPC\]", text, flags=re.M)) == 3 and
            re.search(r"^Error-\[", text, flags=re.M) is None,
            "compile diagnostic population drift")
    return {"executed_commands": 1, "parsed_files": 16,
            "compile_errors": 0, "warnings": 4, "tfipc_unused_carry": 3}


TRACE = re.compile(
    r"^M1601_TRACE cycle=(\d+) header=([^ ]+) source=([^ ]+) endpoint=([^ ]+) "
    r"mem=([^ ]+) commit=([^ ]+) done=([^ ]+) top_pns=([^ ]+) "
    r"endpoint_fault=([^ ]+) taps_csfamS=([^ ]+)$")


def audit_sim(text):
    lines = text.splitlines()
    commands = [row for row in lines if row.startswith("Command: ") and "/simv " in row]
    require(len(commands) == 1, "simv command count is not one")
    lower = commands[0].lower()
    for token in ("ucli", "initreg", "saif", "ptpx"):
        require(token not in lower, "prohibited sim option: " + token)
    traces = [TRACE.match(row).groups() for row in lines if TRACE.match(row)]
    require([int(row[0]) for row in traces] == [1, 2, 3, 4],
            "trace population drift")
    require(all(row[1:] == ("0/0", "0/0", "0/0", "0/0", "0/0", "0/0",
                                 "000/000", "00000000/00000000", "000000/000000")
                for row in traces[:3]), "cycles 1-3 not clean")
    cycle4 = traces[3]
    require(cycle4[1:7] == ("0/0", "0/0", "0/0", "0/0", "0/0", "0/0") and
            cycle4[7] == "100/100" and
            cycle4[8] == "00000000/00000000" and
            cycle4[9] == "000000/000000", "cycle-4 signature drift")
    stop = [row for row in lines if row.startswith("M1601_FIRST_STOP ")]
    require(len(stop) == 1, "first-stop population drift")
    for token in ("FAULT_OR_X", "cycle=4", "first_difference_cycle=-1",
                  "first_fault_cycle=4", "rtl_top_pns=100",
                  "mapped_top_pns=100", "rtl_endpoint_fault=00000000",
                  "mapped_endpoint_fault=00000000", "rtl_taps=000000",
                  "mapped_taps=000000"):
        require(token in stop[0], "first-stop field drift: " + token)
    require("$finish at simulation time                22501" in text,
            "settled sample timestamp drift")
    return {"executed_commands": 1, "first_stop_cycle": 4,
            "sample_time_ps": 22501, "first_difference_cycle": -1,
            "rtl_top_pns": "100", "mapped_top_pns": "100",
            "endpoint_faults": "all_zero", "registered_fault_taps": "all_zero"}


def source_semantics():
    tb = TB.read_text(encoding="utf-8")
    compactor = COMPACTOR.read_text(encoding="utf-8")
    frontend = FRONTEND.read_text(encoding="utf-8")
    core = CORE.read_text(encoding="utf-8")
    k8 = K8.read_text(encoding="utf-8")

    for token in ("always #1.5 clk_core = ~clk_core;",
                  "repeat (4) @(negedge clk_core);", "rst_core = 1'b0;",
                  "repeat (2) @(posedge clk_core);", "@(negedge clk_core);\n            header_valid = 1'b1;",
                  "header_raw_beat_count = 6'd4;", "header_window_depth = 4'd2;",
                  "header_output_blocks = 4'd1;", "raw_lane_valid = 4'b1111;",
                  "raw_last = 1'b1;", "raw_valid = 1'b1;",
                  "@(negedge clk_core);\n            raw_valid = 1'b0;"):
        require(token in tb, "TB timing/stimulus token drift: " + token)
    require("#1ps;\n            trace_edge();" in tb,
            "post-edge settled checker drift")

    for token in (
        "raw_packet_legal = token_active_q && !raw_done_q",
        "raw_beats_accepted_q + raw_lane_count\n                <= raw_beat_count_q && raw_last_legal;",
        "assign raw_ready = !fault_q && raw_packet_legal",
        "assign raw_accept = raw_valid && raw_ready;",
        "|| (raw_valid && !raw_packet_legal);",
        "assign protocol_error = fault_q || illegal_request;",
        "if (illegal_request) fault_q <= 1;",
        "if (raw_accept) begin",
        "if (raw_last) raw_done_q <= 1;",
    ):
        require(token in compactor, "compactor semantic token drift: " + token)
    require(".raw_valid(raw_valid), .raw_ready(raw_ready)" in frontend and
            ".protocol_error(m202_protocol_error)" in frontend and
            "|| m202_protocol_error || m204_protocol_error" in frontend,
            "compactor protocol propagation drift")
    require("|| fe_protocol_error || svc_protocol_error;" in core,
            "frontend-to-core protocol propagation drift")
    require("assign protocol_error = core_protocol_error || adapter_protocol_error" in k8 and
            "|| consistency_fault_q || consistency_fault_now;" in k8,
            "core-to-top protocol propagation drift")

    return {
        "clock_half_period_ns": 1.5,
        "reset_deassert_ps": 12000,
        "header_assert_negedge_ps": 18000,
        "header_accept_posedge_ps": 19500,
        "raw_assert_negedge_ps": 21000,
        "raw_accept_posedge_ps": 22500,
        "settled_observation_ps": 22501,
        "scheduled_raw_deassert_negedge_ps": 24000,
        "pre_edge": {"raw_valid": 1, "raw_done_q": 0,
                     "raw_packet_legal": 1, "raw_accept": 1,
                     "illegal_request": 0, "compactor_fault_q": 0,
                     "public_protocol_error": 0},
        "post_edge_settled": {"raw_valid": 1, "raw_done_q": 1,
                              "raw_packet_legal": 0, "raw_accept": 0,
                              "illegal_request": 1, "compactor_fault_q": 0,
                              "public_protocol_error": 1},
        "after_producer_negedge_withdrawal_counterfactual": {
            "raw_valid": 0, "illegal_request": 0,
            "compactor_fault_q": 0, "public_protocol_error": 0},
        "causal_term": "compactor: raw_valid && !raw_packet_legal after raw_done_q advances",
        "registered_fault_not_set_reason": "illegal_request was zero at the accepting edge",
    }


def main(output):
    for path, expected in EXPECTED.items():
        mode = path.lstat().st_mode
        require(stat.S_ISREG(mode) and not path.is_symlink(),
                "nonregular frozen identity: " + str(path))
        require(sha256(path) == expected, "frozen identity drift: " + str(path))
    require(ATTEMPT.is_dir() and not ATTEMPT.is_symlink() and
            not any(ATTEMPT.iterdir()), "consumed-attempt marker drift")
    authority = verify_sealed_m1603()
    filelist_rows = [row for row in FILELIST.read_text(encoding="utf-8").splitlines()
                     if row.strip()]
    require(len(filelist_rows) == 16, "filelist population drift")
    compile_result = audit_compile(COMPILE.read_text(encoding="utf-8"), filelist_rows)
    sim_result = audit_sim(SIM.read_text(encoding="utf-8"))
    semantics = source_semantics()

    value = {
        "schema": "m1606_m1604_c2_settled_result_semantics_independent_review_r1_v1",
        "status": "PASS_M1606_RESULT_AUDIT__M1604_LEGAL_ACCEPT_POSTEDGE_COMBINATIONAL_FALSE_ERROR__NO_TOOL_AUTHORITY",
        "identity": {str(path.relative_to(ROOT)): digest
                     for path, digest in EXPECTED.items()},
        "authority": authority,
        "m1604_execution": {"compile": compile_result, "simulation": sim_result,
                            "attempt_consumed": True},
        "synchronous_semantics": semantics,
        "classification": {
            "rtl_mapped_difference": False,
            "stable_x": False,
            "actual_protocol_violation_at_accepting_edge": False,
            "postaccept_combinational_protocol_error_pulse": True,
            "root": "LEGAL_READY_VALID_ACCEPT_REINTERPRETED_BY_POSTEDGE_STATE_WHILE_VALID_HELD",
            "m1594_settle_repair_effective": True,
            "m1604_result": "FAILED_DIAGNOSTIC__SEMANTIC_FALSE_POSITIVE__NOT_CITABLE",
        },
        "repair_comparison": {
            "pre_edge_sampling": {
                "would_observe": "raw_accept=1, protocol_error=0",
                "advantage": "matches accepting-edge transaction legality",
                "fatal_limitation": "masks a real public-output pulse and naive posedge sampling recreates the mapped active-region race",
                "verdict": "DO_NOT_USE_AS_FIX"},
            "post_edge_settled_sampling": {
                "observed": "raw_accept=0, protocol_error=1, registered fault_q=0",
                "advantage": "race-free and faithfully observes the implemented public interface",
                "verdict": "KEEP_AS_CHECKER"},
            "registered_fault_only_public_protocol_error": {
                "would_observe_cycle4": 0,
                "true_illegal_edge_behavior": "illegal_request sampled high sets sticky fault_q for postedge visibility",
                "same_edge_acceptance": "ready/raw_packet_legal already prevent malformed acceptance",
                "verdict": "PREFERRED_SEMANTIC_REPAIR"},
        },
        "unique_minimum_next_step": {
            "action": "author source-only RTL successor changing only compactor public protocol_error from fault_q||illegal_request to fault_q",
            "preserve": ["illegal_request expression", "fault_q latch", "raw_ready/raw_packet_legal gate",
                         "M1601 1ps settled checker", "stimulus", "mapped netlist until later DC"],
            "source_review_before_tools": True,
            "resynthesis_eventually_required": True,
            "vcs_authorized_now": False, "dc_authorized_now": False,
            "ptpx_authorized_now": False, "file_edits_by_m1606": False,
        },
        "claim_boundary": {"paper_citable": False, "rtl_mapped_pass": False,
                           "timing_verified": False, "power": False,
                           "ppa": False, "system_speedup": False},
    }
    Path(output).write_text(json.dumps(value, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.output)
