#!/usr/bin/env python3
"""Run allowed Python checks and seal the M1588 incremental review."""

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
OUT = Path(__file__).resolve().parent
PY310 = "/opt/anaconda3/envs/pytorch310/bin/python3.10"
TEST = HW / "system_simulator/tests/test_m1578_c2_rtl_vs_mapped_k8_case0_first_fault_source.py"
RUNNER = HW / "dc_handoff/scripts/run_m1578_c2_rtl_vs_mapped_k8_case0_first_fault_source.py"
R310 = OUT / "cpython310_qa.json"
R36 = OUT / "cpython36_qa.json"
STATUS = "PASS_M1588_INCREMENTAL_SOURCE_QA__AUTHORIZE_EXACTLY_ONE_NEW_IDENTITY_COMPILE_AND_ONE_K8_CASE0_SIM__NO_TOOL_RUN"
RUNTIME_STATUS = "PASS_M1588_INCREMENTAL_SOURCE_QA__AUTHORIZE_ONE_NEW_IDENTITY_COMPILE_AND_ONE_CASE0_SIM__NO_TOOL_RUN"
TOP = "tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault"
FILELIST_SHA = "09166d29aedc0a03266f9726ec006ac96efdd396c5290edb423ae303ad2548f1"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def execute(command):
    result = subprocess.run(command, cwd=str(ROOT), text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=60,
                            check=False)
    require(result.returncode == 0,
            "allowed Python check failed: {}\n{}".format(
                command, result.stdout))
    return result.stdout


def main():
    r310 = load(R310)
    r36 = load(R36)
    for result, version in ((r310, "3.10.18"), (r36, "3.6.8")):
        require(result["status"] == RUNTIME_STATUS and
                result["runtime"]["version"] == version,
                "dual-runtime incremental QA drift")
        require(result["robustness_mutations"]["passed"] == 10 and
                result["robustness_mutations"]["count"] == 10 and
                all(row["rejected"] for row in
                    result["robustness_mutations"]["rows"]),
                "robustness mutation drift")
        require(result["m1586_compile_log"] == {
            "failed_line": 350, "failed_token": "tri",
            "other_errors": 0, "parse_errors": 1, "warnings": 1},
            "M1586 parse receipt drift")
        require(result["execution_by_m1588"] == {
            "attempt_consumed": False, "eda_runs": 0,
            "simv_runs": 0, "vcs_compiles": 0},
            "M1588 execution boundary drift")

    test_output = execute([PY310, str(TEST)])
    static_output = execute([PY310, str(RUNNER), "--static-check"])
    describe_output = execute([PY310, str(RUNNER), "--describe"])
    require("Ran 9 tests" in test_output and test_output.rstrip().endswith("OK"),
            "existing Python test receipt drift")
    static = json.loads(static_output)
    describe = json.loads(describe_output)
    require(static["status"] ==
            "PASS_SOURCE_ONLY_READY_FOR_INDEPENDENT_HAMMER__NO_TOOL_RUN" and
            static["execution"] == {"attempt_consumed": False, "ptpx": 0,
                                    "saif": 0, "simv_runs": 0,
                                    "vcs_compiles": 0},
            "author static check drift")
    require(describe["case"] == "exact M979 K8 case0" and
            describe["memory"] ==
            "two independent instances of the same reset-safe model",
            "author description drift")

    (OUT / "existing_python_tests.txt").write_text(test_output,
                                                    encoding="utf-8")
    (OUT / "existing_static_check.json").write_text(
        json.dumps(static, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "existing_describe.json").write_text(
        json.dumps(describe, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    authorization = {
        "authorized_after_m1588": True,
        "new_result_identity_required": True,
        "vcs_compiles": 1,
        "simv_runs": 1,
        "cases": ["k8_case0"],
        "required_top": TOP,
        "required_filelist_sha256": FILELIST_SHA,
        "reuse_m1586_or_m1502_simv": False,
        "ucli": False, "initreg": False, "saif": False,
        "ptpx": False, "force_release": False,
    }
    review = {
        "schema": "m1588_m1587_reserved_word_repair_independent_incremental_review_r1_v1",
        "status": STATUS,
        "score": 100,
        "identity": {
            "source_commit": "842da3aa7e9aa2441e23953b643a836198d568d0",
            "m1587_contract_sha256":
                "b5feba8a0a00318c1c767dda21892e26ee62b0f105112adb6da204161ea1a52b",
            "m1586_compile_log_sha256":
                "21fa1fb0021acfbcf585989f0b0ccf98a8495a41010b252309f0f72c60448eaa",
            "repaired_tb_sha256":
                "4a2ef4c40037274aadd936db8dbe38258aa39fa14a7e0322741f92acd958c435",
            "filelist_sha256": FILELIST_SHA,
            "rtl_wrapper_sha256":
                "3328e52d8cf1eec6098ebb7b0525ac55cd8bd6b2fe5b5e504b337d1a678e3c4b",
            "mapped_netlist_sha256":
                "6b745030df6c041a0501d041ee277459c726c52263b4eec6ab5712f14d156de5",
            "memory_model_sha256":
                "f9b0d87dd3b951a24b79545555c09b32bbce695e85cc71df2948e5065981c7c3",
        },
        "m1586_failure": {
            "compile_attempts": 1, "simulation_attempts": 0,
            "unique_parse_error": True, "error_class": "Error-[SE]",
            "file": "tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault.sv",
            "line": 350, "token": "tri",
            "dut_elaboration_reached": False,
        },
        "incremental_delta": {
            "old_identifier": "tri",
            "new_identifier": "tri_state_char",
            "identifier_occurrences": 24,
            "declaration_or_calls": 21,
            "assignments": 3,
            "reverse_rename_is_byte_exact_to_m1578": True,
            "filelist_unchanged": True,
            "dut_sources_unchanged": True,
            "mapped_netlist_unchanged": True,
            "memory_unchanged": True,
            "stimulus_unchanged": True,
            "four_state_semantics_unchanged": True,
        },
        "checks": {
            "existing_python_tests": "PASS_9_OF_9",
            "existing_static_check": static["status"],
            "independent_source_parser": {
                "cpython310": "PASS_10_OF_10_MUTATIONS",
                "cpython36": "PASS_10_OF_10_MUTATIONS"},
            "standalone_eda_parser_invoked": False,
            "reason": "M1588 is constrained to Python and static source analysis; Verilator was detected but intentionally not invoked.",
        },
        "authorization": authorization,
        "claim_boundary": {
            "source_qa_only": True,
            "vcs_compile_executed_by_m1588": False,
            "simv_executed_by_m1588": False,
            "eda_executed_by_m1588": False,
            "attempt_consumed_by_m1588": False,
            "rtl_pass": False, "mapped_pass": False,
            "paper_citable": False, "timing": False,
            "power": False, "ppa": False, "speedup": False,
        },
    }
    review_md = """# M1588 — M1587 C2 reserved-word repair incremental QA

Decision: **PASS incremental source QA; authorize exactly one new-identity VCS
compile and one `k8_case0` simulation.** M1588 ran no VCS, `simv`, or EDA tool
and consumed no attempt.

The consumed M1586 compile log contains one and only one compiler error:
`Error-[SE]` at testbench line 350, whose token is the SystemVerilog reserved
word `tri`. It did not reach DUT elaboration and produced no simulation.

The repaired testbench changes the identifier `tri` to `tri_state_char` at 24
locations: 21 declaration/call occurrences and three function assignments.
Replacing the new identifier with the old one reconstructs the frozen M1578
testbench byte for byte. Consequently the DUT pair, ordered filelist, hard-wired
M979 K8 case0 stimulus, two independent memory fabrics, event schedule, and
four-state `0/1/X` reporting are unchanged. The RTL wrapper, mapped netlist and
memory model hashes also remain frozen.

The existing Python suite passes 9/9. The independent conservative source
parser rejects 10/10 robustness mutations under CPython 3.10.18 and 3.6.8.
Verilator was detected but intentionally not invoked because this review is
limited to Python and static source analysis.

The future run must use a new result identity, the frozen filelist, and explicit
`-top tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault`. It may consume exactly
one compile and one `k8_case0` simulation. Reuse of the failed M1586 or M1502
binary, UCLI, initreg, force/release, SAIF, PTPX, a second compile, or a second
simulation is not authorized. No RTL/mapped PASS or paper claim exists until
the future run is independently reviewed.
"""
    mechanical = {
        "schema": "m1588_incremental_mechanical_checks_r1_v1",
        "status": STATUS,
        "commands": [
            {"kind": "python_test", "result": "PASS_9_OF_9",
             "vcs": False, "simv": False, "eda": False},
            {"kind": "author_static_check", "result": static["status"],
             "vcs": False, "simv": False, "eda": False},
            {"kind": "independent_python_parser_cpython310",
             "result": "PASS_10_OF_10_MUTATIONS",
             "vcs": False, "simv": False, "eda": False},
            {"kind": "independent_python_parser_cpython36",
             "result": "PASS_10_OF_10_MUTATIONS",
             "vcs": False, "simv": False, "eda": False},
        ],
        "authorization": authorization,
    }

    targets = [OUT / "review.json", OUT / "review.md",
               OUT / "mechanical_checks.json", OUT / "RUN_COMPLETE.txt",
               OUT / "SHA256SUMS", OUT / "SHA256SUMS.seal.sha256"]
    require(all(not path.exists() for path in targets), "refuse overwrite")
    (OUT / "review.json").write_text(
        json.dumps(review, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    (OUT / "review.md").write_text(review_md, encoding="utf-8")
    (OUT / "mechanical_checks.json").write_text(
        json.dumps(mechanical, indent=2, sort_keys=True,
                   allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "RUN_COMPLETE.txt").write_text(STATUS + "\n", encoding="ascii")
    members = [OUT / "independent_incremental_qa.py", R310, R36,
               Path(__file__).resolve(), OUT / "existing_python_tests.txt",
               OUT / "existing_static_check.json", OUT / "existing_describe.json",
               OUT / "mechanical_checks.json", OUT / "review.json",
               OUT / "review.md", OUT / "RUN_COMPLETE.txt"]
    sums = OUT / "SHA256SUMS"
    sums.write_text("".join("{}  {}\n".format(sha256(path), path.name)
                             for path in members), encoding="ascii")
    (OUT / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(sums)), encoding="ascii")
    print(STATUS)


if __name__ == "__main__":
    main()
