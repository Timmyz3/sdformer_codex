#!/usr/bin/env python3
"""Run only the allowed Python checks and seal the M1580 review."""

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
OUT = Path(__file__).resolve().parent
TEST = HW / "system_simulator/tests/test_m1578_c2_rtl_vs_mapped_k8_case0_first_fault_source.py"
RUNNER = HW / "dc_handoff/scripts/run_m1578_c2_rtl_vs_mapped_k8_case0_first_fault_source.py"
R310 = OUT / "cpython310_qa.json"
R36 = OUT / "cpython36_qa.json"
PY310 = "/opt/anaconda3/envs/pytorch310/bin/python3.10"
STATUS = "PASS_M1580_INDEPENDENT_SOURCE_QA__AUTHORIZE_EXACTLY_ONE_FUTURE_COMPILE_AND_ONE_K8_CASE0_SIM__NO_TOOL_RUN_IN_REVIEW"
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
    completed = subprocess.run(command, cwd=str(ROOT), text=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=60,
                               check=False)
    require(completed.returncode == 0,
            "Python check failed: {}\n{}".format(command, completed.stdout))
    return completed.stdout


def main():
    r310 = load(R310)
    r36 = load(R36)
    expected_runtime_status = (
        "PASS_M1580_SOURCE_QA__AUTHORIZE_ONE_FUTURE_COMPILE_AND_ONE_CASE0_SIM__NO_TOOL_RUN_IN_REVIEW")
    for result, version in ((r310, "3.10.18"), (r36, "3.6.8")):
        require(result["status"] == expected_runtime_status and
                result["runtime"]["version"] == version,
                "dual-runtime result drift")
        require(result["robustness_mutations"]["count"] == 16 and
                result["robustness_mutations"]["passed"] == 16 and
                all(row["rejected"] for row in
                    result["robustness_mutations"]["rows"]),
                "robustness mutation drift")
        require(result["pre_review_execution"] == {
            "attempt_consumed": False, "initreg": False, "ptpx": False,
            "saif": False, "simv_runs": 0, "ucli": False,
            "vcs_compiles": 0}, "pre-review execution drift")
        authorization = result["authorization_recommendation"]
        require(authorization["vcs_compiles"] == 1 and
                authorization["simv_runs"] == 1 and
                authorization["cases"] == ["k8_case0"] and
                authorization["required_top"] == TOP and
                authorization["required_filelist_sha256"] == FILELIST_SHA,
                "authorization recommendation drift")

    test_output = execute([PY310, str(TEST)])
    static_output = execute([PY310, str(RUNNER), "--static-check"])
    describe_output = execute([PY310, str(RUNNER), "--describe"])
    require("Ran 9 tests" in test_output and test_output.rstrip().endswith("OK"),
            "existing unit-test receipt drift")
    static = json.loads(static_output)
    describe = json.loads(describe_output)
    require(static["status"] ==
            "PASS_SOURCE_ONLY_READY_FOR_INDEPENDENT_HAMMER__NO_TOOL_RUN" and
            static["execution"] == {"attempt_consumed": False, "ptpx": 0,
                                    "saif": 0, "simv_runs": 0,
                                    "vcs_compiles": 0},
            "author static-check drift")
    require(describe["case"] == "exact M979 K8 case0" and
            describe["memory"] ==
            "two independent instances of the same reset-safe model",
            "author description drift")

    (OUT / "existing_unit_tests_output.txt").write_text(test_output,
                                                         encoding="utf-8")
    (OUT / "existing_static_check.json").write_text(
        json.dumps(static, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "existing_describe.json").write_text(
        json.dumps(describe, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    review = {
        "schema": "m1580_m1578_c2_rtl_mapped_case0_source_independent_qa_r1_v1",
        "status": STATUS,
        "score": 99,
        "identity": {
            "source_commit": "23be26d73606a26933bdc62f5c88236e5abbf8a0",
            "contract_sha256": "29fdfd080ade36ca373a8e716771ebc896c156ffadea0590c7c3b00c3c616d2d",
            "tb_sha256": "1c4659304c63b84cb9be443dbec33c71c61a92db092fed55718c0453d7099308",
            "filelist_sha256": FILELIST_SHA,
            "runner_sha256": "4c2dcca813329b4f1aaac906b3e198720961a8de8276754986b1ca1c9bc405b6",
            "mapped_netlist_sha256": "6b745030df6c041a0501d041ee277459c726c52263b4eec6ab5712f14d156de5",
            "rtl_wrapper_sha256": "3328e52d8cf1eec6098ebb7b0525ac55cd8bd6b2fe5b5e504b337d1a678e3c4b",
            "memory_model_sha256": "f9b0d87dd3b951a24b79545555c09b32bbce695e85cc71df2948e5065981c7c3",
            "docs359_sha256": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
        },
        "checks": {
            "existing_unit_tests": "PASS_9_OF_9",
            "existing_static_check": static["status"],
            "independent_dual_runtime": {"cpython310": "PASS_16_OF_16",
                                         "cpython36": "PASS_16_OF_16"},
            "ordered_filelist_entries": 16,
            "required_top": TOP,
            "top_source_is_last_filelist_entry": True,
            "mapped_arch_mode1_definition_count": 1,
            "rtl_arch_mode1_instance_count": 1,
            "mapped_dut_instance_count": 1,
            "independent_memory_fabric_instances": 2,
            "independent_memory_signal_namespaces": True,
            "four_state_log_preserved": True,
            "x_to_zero_conversion": False,
            "first_difference_cycle_recorded": True,
            "first_fault_cycle_recorded": True,
            "events_recorded": ["header", "source", "endpoint_request",
                                "memory_response", "commit", "done"],
        },
        "first_fault_classification": {
            "reset_or_uninitialized": "control_unknown and 0/1/X render",
            "protocol": "protocol_error plus endpoint_fault and six named taps",
            "stale": "stale_response_seen plus memory-adapter stale tap",
            "numeric": "numeric_overflow",
            "stage": "same-cycle trace precedes the first-stop record and carries six event stages",
            "sufficient_for_requested_localization": True,
        },
        "authorization": {
            "pre_review_authorized_now_was_false": True,
            "authorized_after_m1580": True,
            "vcs_compiles": 1,
            "simv_runs": 1,
            "cases": ["k8_case0"],
            "required_top": TOP,
            "required_filelist_sha256": FILELIST_SHA,
            "execution_wrapper_requirement":
                "The future one-shot invocation must explicitly bind -top to required_top; the source-only filelist intentionally contains paths only.",
            "reuse_m1502_simv": False,
            "ucli": False, "initreg": False, "saif": False,
            "ptpx": False, "force_release": False,
        },
        "claim_boundary": {
            "source_qa_only": True,
            "vcs_compile_executed_by_m1580": False,
            "simv_executed_by_m1580": False,
            "attempt_consumed_by_m1580": False,
            "rtl_pass": False, "mapped_pass": False,
            "timing": False, "power": False, "ppa": False,
            "speedup": False, "paper_citable": False,
        },
    }
    review_md = """# M1580 — M1578 C2 RTL/mapped case0 independent source QA

Decision: **PASS source-only QA; authorize exactly one future VCS compile and
one `k8_case0` simulation.** M1580 itself ran no VCS, `simv`, or EDA tool and
consumed no attempt.

The existing Python suite passes 9/9 and the author static check remains
`PASS_SOURCE_ONLY_READY_FOR_INDEPENDENT_HAMMER__NO_TOOL_RUN`. An independent
text-and-identity checker also passes 16/16 robustness mutations under both
CPython 3.10.18 and 3.6.8.

The 16-entry filelist binds the frozen RTL wrapper, the exact mapped
`ARCH_MODE1` netlist, the reset-safe memory model, and the M1578 top source.
The top instantiates RTL `ARCH_MODE=1` and the mapped module once each. The two
memory fabrics have distinct RTL/mapped signal namespaces, so neither DUT can
consume the other's requests or responses.

Four-state information is preserved with case equality/inequality,
`$isunknown`, and explicit `0/1/X` rendering. The same-cycle trace precedes the
stop record and reports header, source, endpoint request, memory response,
commit and done, plus top protocol/numeric/stale bits, eight endpoint bits and
six internal taps per DUT. This is sufficient to separate reset/X, protocol,
stale and numeric first-fault classes.

The future one-shot wrapper must explicitly pass
`-top tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault` and the frozen filelist
SHA. The filelist itself contains source paths only. Reusing M1502 `simv`, UCLI,
initreg, SAIF, PTPX, force/release, a second compile, or a second simulation is
not authorized. No RTL/mapped PASS or paper claim exists until that one run is
independently reviewed.
"""
    mechanical = {
        "schema": "m1580_m1578_independent_source_qa_mechanical_checks_r1_v1",
        "status": STATUS,
        "commands": [
            {"command": PY310 + " " + str(TEST), "result": "PASS_9_OF_9",
             "vcs": False, "simv": False, "eda": False},
            {"command": PY310 + " " + str(RUNNER) + " --static-check",
             "result": static["status"], "vcs": False, "simv": False,
             "eda": False},
            {"command": "CPython3.10 independent_static_qa.py",
             "result": "PASS_16_OF_16", "vcs": False, "simv": False,
             "eda": False},
            {"command": "CPython3.6 independent_static_qa.py",
             "result": "PASS_16_OF_16", "vcs": False, "simv": False,
             "eda": False},
        ],
        "pre_review_execution": r310["pre_review_execution"],
        "authorization": review["authorization"],
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
        json.dumps(mechanical, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    (OUT / "RUN_COMPLETE.txt").write_text(STATUS + "\n", encoding="ascii")
    members = [OUT / "independent_static_qa.py", R310, R36,
               Path(__file__).resolve(), OUT / "existing_unit_tests_output.txt",
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
