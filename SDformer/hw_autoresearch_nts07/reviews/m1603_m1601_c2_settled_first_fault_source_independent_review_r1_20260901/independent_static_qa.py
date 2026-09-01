#!/usr/bin/env python3
"""Different-author, simulator-free QA for the M1601 settled checker source."""

from __future__ import print_function

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import stat


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
OLD_TB = HW / "dc_handoff/tb/tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault.sv"
NEW_TB = HW / "dc_handoff/tb/tb_m1601_c2_rtl_vs_mapped_k8_case0_settled_first_fault.sv"
OLD_FL = HW / "dc_handoff/filelists/date_m1578_c2_rtl_vs_mapped_k8_case0_first_fault_source.f"
NEW_FL = HW / "dc_handoff/filelists/date_m1601_c2_rtl_vs_mapped_k8_case0_settled_first_fault_source.f"
CHECKER = HW / "dc_handoff/scripts/check_m1601_c2_settled_first_fault_source.py"
TEST = HW / "system_simulator/tests/test_m1601_c2_settled_first_fault_source.py"
CONTRACT = HW / "contracts/m1601_c2_settled_first_fault_source_contract_r1_20260901.json"
CONTRACT_INNER = Path(str(CONTRACT) + ".sha256")
CONTRACT_OUTER = Path(str(CONTRACT) + ".sha256.seal.sha256")
M1594 = HW / "reviews/m1594_m1593_c2_first_fault_independent_cone_review_r1_20260901"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    OLD_TB: "4a2ef4c40037274aadd936db8dbe38258aa39fa14a7e0322741f92acd958c435",
    NEW_TB: "3e8a9254fd9104aeeb4d3f05077a9f2b8ae33a9617d3236447108a5b666ba8e4",
    OLD_FL: "09166d29aedc0a03266f9726ec006ac96efdd396c5290edb423ae303ad2548f1",
    NEW_FL: "b6e384a3b7de9541a66af0302722c9ae9ca12b50e5e57a1ac764bf1576a39a53",
    CHECKER: "596cb2f3dfe6ab58365d9cf01b5352a865137a2210c007db41205ce222507989",
    TEST: "394b68c8d3e78e5d7fd9c1a917afee71b2267f5644a9a156ef82cbcf2859066b",
    CONTRACT: "7ae75fca150799c83be7869ab5c9d01fe4c6636deab9eccc4fc7c61458fe57fe",
    CONTRACT_INNER: "171e497373caa6f703da28c9e3e8c6ca44921e03e482dc2f754a7848600ad64d",
    CONTRACT_OUTER: "c06605a25a06839bd1b7a5660e6eac20f1e0ad296bbf7036426db4c45d2cab87",
    M1594 / "review.json": "97370ae3eeae00ad79e3647b6ec34df3e114c88b807d773a69b250b0cfac324e",
    M1594 / "SHA256SUMS": "b629c1eef0489e96840e2444f2ef252ee2d261ccc47159de17fed9180dd8ae53",
    M1594 / "SHA256SUMS.seal.sha256": "1785184a2a61bba0f43c6727f93225919adc7bda2f728d646ff3e6f3b62951f5",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

PREFIX_SHA = (
    "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a",
    "e278da8b0deaa0dda07b0477930453daa40b0331399a3941b743d604d0b102a5",
    "8295393bf91a9bfc64a2253aaff60db97df5df587ab9b77d56996afee82cb2a0",
    "529e463802fec72716ac6592d31e7668104a5463ff92499a98ec7314c8e88267",
    "f6537081977e9dc09e968fad800b333604b4573ee2e9361960483349fe1e8ad1",
    "44f7df331af66ba62fadf5e336b9c0c00d00f809e215aa8e091e9de011c5627e",
    "3811998fc48d31e6519ecc6c6cfb8f5d38db6fc6dd070e09d73a5f70b7579871",
    "010fe9e6786db1d3bbcad7759bda17a783ce5cfe15cae02c5b4c9ebf96e9950b",
    "6ea038ef935b1144d5424634e75446301270362c259341a8e7e7117523b25815",
    "11080d39c06672cebb64988e931c41e1d4c04134a312aeb8e250d01f0ac576ff",
    "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
    "2588f890213d29aab6829dff679719c0f9ce4762c17bb061d1869b27a2f1d50e",
    "3328e52d8cf1eec6098ebb7b0525ac55cd8bd6b2fe5b5e504b337d1a678e3c4b",
    "6b745030df6c041a0501d041ee277459c726c52263b4eec6ab5712f14d156de5",
    "f9b0d87dd3b951a24b79545555c09b32bbce695e85cc71df2948e5065981c7c3",
)

OLD_TOP = "tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault"
NEW_TOP = "tb_m1601_c2_rtl_vs_mapped_k8_case0_settled_first_fault"


class QAError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise QAError(message)


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
                          QAError("nonfinite JSON: " + token)))


SETTLE_BLOCK = (
    "\n            // Gate-level sequential cells and their zero-delay combinational\n"
    "            // fanout update in later event regions than this posedge callback.\n"
    "            // One timeprecision step preserves the cycle while removing the\n"
    "            // active-region race diagnosed by the sealed M1594 review.\n"
    "            #1ps;\n"
)


def normalize_new_tb(text):
    replacements = (
        (NEW_TOP, OLD_TOP),
        ("M1601_TRACE", "M1578_TRACE"),
        ("M1601_FIRST_STOP", "M1578_FIRST_STOP"),
        ("M1601 diagnostic watchdog", "M1578 diagnostic watchdog"),
        ("M1601 absolute watchdog", "M1578 absolute watchdog"),
    )
    expected_counts = (1, 1, 1, 1, 1)
    for (new, old), count in zip(replacements, expected_counts):
        require(text.count(new) == count, "identifier count drift: " + new)
        text = text.replace(new, old)
    require(text.count(SETTLE_BLOCK) == 1, "authorized settle block count drift")
    return text.replace(SETTLE_BLOCK, "\n")


def verify_tb(old_text, new_text):
    require(normalize_new_tb(new_text) == old_text,
            "new TB differs beyond identifier rename and one settle block")
    require(old_text.count("#1ps;") == 0 and new_text.count("#1ps;") == 1,
            "1 ps delay population drift")
    checker_start = new_text.index("always @(posedge clk_core) begin")
    checker_end = new_text.index("task automatic initialize_inputs", checker_start)
    block = new_text[checker_start:checker_end]
    increment = block.index("cycle_ordinal = cycle_ordinal + 1;")
    delay = block.index("#1ps;")
    trace = block.index("trace_edge();")
    first_decision = block.index("if (difference_now")
    require(increment < delay < trace < first_decision,
            "settle is not after cycle increment and before trace/decisions")
    for token in ('print_stop("FAULT_OR_X")',
                  'print_stop("FIRST_RTL_MAPPED_DIFFERENCE")',
                  'print_stop("BOTH_CLEAN_TO_DONE")', "$finish;"):
        require(delay < block.index(token), "settle occurs after stop path: " + token)
    require(new_text.count("always @(posedge clk_core)") == 1,
            "checker process population drift")
    require("`timescale 1ns/1ps" in new_text,
            "1 ps is not one declared timeprecision step")
    return {"identifier_renames": 5, "settle_count": 1,
            "settle_after_cycle_increment": True,
            "settle_before_trace_and_all_stop_decisions": True,
            "normalized_byte_equal_to_m1578": True}


def resolve_row(row):
    path = Path(row)
    return path if path.is_absolute() else HW / path


def verify_filelists(old_rows, new_rows):
    require(len(old_rows) == len(new_rows) == 16, "filelist must have 16 entries")
    require(old_rows[:15] == new_rows[:15], "filelist prefix drift")
    require(old_rows[15] == "dc_handoff/tb/tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault.sv",
            "old TB binding drift")
    require(new_rows[15] ==
            "dc_handoff/tb/tb_m1601_c2_rtl_vs_mapped_k8_case0_settled_first_fault.sv",
            "new TB binding drift")
    observed = tuple(sha256(resolve_row(row)) for row in new_rows[:15])
    require(observed == PREFIX_SHA, "frozen library/RTL/netlist/memory identity drift")
    return {"entries": 16, "unchanged_prefix_entries": 15,
            "changed_entry": 16, "prefix_sha256": list(observed),
            "rtl_unchanged": True, "mapped_netlist_unchanged": True,
            "memory_model_unchanged": True}


def verify_sealed_m1594():
    manifest = M1594 / "SHA256SUMS"
    outer = M1594 / "SHA256SUMS.seal.sha256"
    require(outer.read_text(encoding="ascii").split() ==
            [EXPECTED[manifest], "SHA256SUMS"], "M1594 outer content drift")
    expected = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, name = line.split(None, 1)
        name = name.strip()
        rel = Path(name)
        require(name not in expected and name == rel.as_posix() and
                not rel.is_absolute() and ".." not in rel.parts,
                "unsafe M1594 manifest row")
        expected[name] = digest
    actual = set()
    for member in M1594.rglob("*"):
        rel = member.relative_to(M1594).as_posix()
        if rel in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "M1594 symlink")
        if stat.S_ISREG(mode):
            actual.add(rel)
        else:
            require(stat.S_ISDIR(mode), "M1594 special member")
    require(actual == set(expected), "M1594 member set drift")
    for name, digest in expected.items():
        require(sha256(M1594 / name) == digest, "M1594 member drift: " + name)
    review = strict_json((M1594 / "review.json").read_text(encoding="utf-8"))
    require(review["status"] == "PASS_REVIEW__M1593_FAILED_DO_NOT_CITE",
            "M1594 status drift")
    decision = review["decision"]
    require(decision["primary_classification"] ==
            "CHECKER_ACTIVE_REGION_SAMPLING_DEFECT_WITH_COMBINATIONAL_X_OBSERVATION" and
            decision["minimum_repair"] ==
            "add exactly one 1ps/one-timeprecision post-posedge settle before trace and all stop decisions" and
            decision["resynthesis_required_for_minimum_repair"] is False,
            "M1594 decision drift")
    return {"full_tree_sealed": True, "minimum_repair_exact": True,
            "resynthesis_required": False}


def verify_contract(value):
    require(value["schema"] == "m1601_c2_settled_first_fault_source_contract_r1_v1",
            "contract schema drift")
    require(value["predecessor"] == {
        "failed_attempt": "M1593", "independent_review": "M1594",
        "classification": "CHECKER_ACTIVE_REGION_SAMPLING_DEFECT_WITH_COMBINATIONAL_X_OBSERVATION",
        "review_sha256": EXPECTED[M1594 / "review.json"],
        "manifest_sha256": EXPECTED[M1594 / "SHA256SUMS"],
        "outer_seal_file_sha256": EXPECTED[M1594 / "SHA256SUMS.seal.sha256"]},
        "contract predecessor drift")
    require(value["sole_semantic_delta"] == {
        "sample_event": "posedge plus 1 ps", "timeprecision_delay": "#1ps",
        "delay_location": "after cycle ordinal increment and before trace plus every stop decision",
        "stimulus_changed": False, "dut_or_mapped_netlist_changed": False,
        "memory_changed": False, "fault_or_comparison_semantics_changed": False},
        "contract semantic delta drift")
    require(value["future_execution"] == {
        "authorized_now": False, "different_author_hammer_required": True,
        "budget_after_pass": {"vcs_compiles": 1, "simv_runs": 1,
                              "case": "k8_case0", "ucli": 0,
                              "initreg": 0, "saif": 0, "ptpx": 0}},
        "contract execution budget drift")
    require(value["claim_boundary"] == {
        "diagnostic_only": True, "paper_citable": False, "rtl_pass": False,
        "mapped_pass": False, "power": False, "energy": False,
        "system_speedup": False, "headline": False},
        "contract claim boundary drift")


def rejected(function):
    try:
        function()
    except (QAError, KeyError, TypeError, ValueError):
        return True
    return False


def mutation_hammer(old_text, new_text, old_rows, new_rows, contract):
    mutations = []
    mutations.append(("missing_settle", lambda: verify_tb(old_text,
        new_text.replace(SETTLE_BLOCK, "\n"))))
    mutations.append(("duplicate_settle", lambda: verify_tb(old_text,
        new_text.replace(SETTLE_BLOCK, SETTLE_BLOCK + SETTLE_BLOCK))))
    mutations.append(("two_ps", lambda: verify_tb(old_text,
        new_text.replace("#1ps;", "#2ps;"))))
    mutations.append(("stimulus_beat", lambda: verify_tb(old_text,
        new_text.replace("header_raw_beat_count = 6'd4;",
                         "header_raw_beat_count = 6'd5;"))))
    mutations.append(("stimulus_tag", lambda: verify_tb(old_text,
        new_text.replace("header_tag = 24'h979000;", "header_tag = 24'h979001;"))))
    mutations.append(("reset_length", lambda: verify_tb(old_text,
        new_text.replace("repeat (4) @(negedge clk_core);",
                         "repeat (5) @(negedge clk_core);"))))
    mutations.append(("compare_semantics", lambda: verify_tb(old_text,
        new_text.replace("!== {mapped_header_accept", "!= {mapped_header_accept"))))
    mutations.append(("fault_semantics", lambda: verify_tb(old_text,
        new_text.replace("mapped_protocol_error !== 1'b0",
                         "mapped_protocol_error == 1'b1"))))
    mutations.append(("watchdog", lambda: verify_tb(old_text,
        new_text.replace("cycle_ordinal >= 4096", "cycle_ordinal >= 4095"))))
    mutations.append(("memory_binding", lambda: verify_tb(old_text,
        new_text.replace("m1578_case0_memory_fabric mapped_memory",
                         "m1578_case0_memory_fabric rtl_memory"))))
    mutations.append(("mapped_binding", lambda: verify_tb(old_text,
        new_text.replace("mapped_dut (", "mapped_dut_changed ("))))
    mutations.append(("rtl_binding", lambda: verify_tb(old_text,
        new_text.replace(") rtl_dut (", ") rtl_dut_changed ("))))

    before_increment = new_text.replace(
        "            cycle_ordinal = cycle_ordinal + 1;" + SETTLE_BLOCK,
        SETTLE_BLOCK + "            cycle_ordinal = cycle_ordinal + 1;\n")
    mutations.append(("settle_before_increment",
                      lambda: verify_tb(old_text, before_increment)))
    after_trace = new_text.replace(
        SETTLE_BLOCK + "            trace_edge();\n",
        "\n            trace_edge();" + SETTLE_BLOCK)
    mutations.append(("settle_after_trace", lambda: verify_tb(old_text, after_trace)))

    for name, rows in (
            ("filelist_prefix", ["rtl_changed.sv"] + new_rows[1:]),
            ("filelist_old_last", new_rows[:15] + [old_rows[15]]),
            ("filelist_extra", new_rows + [new_rows[-1]]),
            ("filelist_missing", new_rows[:-1])):
        mutations.append((name, lambda rows=rows: verify_filelists(old_rows, rows)))

    bad = copy.deepcopy(contract); bad["future_execution"]["authorized_now"] = True
    mutations.append(("contract_preauthorize", lambda bad=bad: verify_contract(bad)))
    bad = copy.deepcopy(contract); bad["future_execution"]["budget_after_pass"]["simv_runs"] = 2
    mutations.append(("contract_second_sim", lambda bad=bad: verify_contract(bad)))
    bad = copy.deepcopy(contract); bad["sole_semantic_delta"]["stimulus_changed"] = True
    mutations.append(("contract_stimulus", lambda bad=bad: verify_contract(bad)))
    bad = copy.deepcopy(contract); bad["claim_boundary"]["paper_citable"] = True
    mutations.append(("contract_claim", lambda bad=bad: verify_contract(bad)))

    for name, function in mutations:
        require(rejected(function), "mutation accepted: " + name)
    return len(mutations)


def main(output):
    for path, expected in EXPECTED.items():
        mode = path.lstat().st_mode
        require(stat.S_ISREG(mode) and not path.is_symlink(),
                "nonregular frozen identity: " + str(path))
        require(sha256(path) == expected, "frozen identity drift: " + str(path))
    require(CONTRACT_INNER.read_text(encoding="ascii").split() ==
            [EXPECTED[CONTRACT], CONTRACT.name], "contract inner seal drift")
    require(CONTRACT_OUTER.read_text(encoding="ascii").split() ==
            [EXPECTED[CONTRACT_INNER], CONTRACT_INNER.name],
            "contract outer seal drift")

    m1594 = verify_sealed_m1594()
    old_text = OLD_TB.read_text(encoding="utf-8")
    new_text = NEW_TB.read_text(encoding="utf-8")
    tb = verify_tb(old_text, new_text)
    old_rows = [row for row in OLD_FL.read_text(encoding="utf-8").splitlines()
                if row.strip()]
    new_rows = [row for row in NEW_FL.read_text(encoding="utf-8").splitlines()
                if row.strip()]
    filelist = verify_filelists(old_rows, new_rows)
    contract = strict_json(CONTRACT.read_text(encoding="utf-8"))
    verify_contract(contract)
    mutation_count = mutation_hammer(old_text, new_text, old_rows, new_rows,
                                     contract)

    value = {
        "schema": "m1603_m1601_c2_settled_first_fault_source_independent_qa_r1_v1",
        "status": "PASS_M1603_M1601_SETTLED_SOURCE__ONE_NEW_IDENTITY_COMPILE_AND_CASE0_SIM_AUTHORIZED__NOT_EXECUTED",
        "identity": {str(path.relative_to(ROOT)): digest
                     for path, digest in EXPECTED.items()},
        "m1594": m1594,
        "testbench": tb,
        "filelist": filelist,
        "python": {"author_test_cpython310": "3/3",
                   "independent_mutations_rejected": mutation_count},
        "execution_by_m1603": {"vcs": 0, "simv": 0, "dc": 0, "ptpx": 0},
        "authorization": {
            "result_identity": "m1604_c2_rtl_mapped_k8_case0_settled_first_fault_r1_20260901",
            "vcs_compiles": 1, "simv_runs": 1, "case": "k8_case0",
            "filelist": NEW_FL.name, "top": NEW_TOP,
            "ucli": 0, "initreg": 0, "saif": 0, "ptpx": 0,
            "retry": False, "paper_claim": False,
            "independent_result_review_required": True,
        },
        "decision_boundary": {
            "both_clean_to_done": "checker sampling race closed; later production-activity source may be reviewed",
            "x_or_difference_persists": "repair RTL validity/reset isolation and rerun DC before any further mapped simulation",
        },
    }
    Path(output).write_text(json.dumps(value, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.output)
