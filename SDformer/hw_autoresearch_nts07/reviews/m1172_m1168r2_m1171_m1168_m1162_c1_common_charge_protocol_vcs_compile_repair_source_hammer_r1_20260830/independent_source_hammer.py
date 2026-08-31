#!/usr/bin/env python3
"""Fresh different-author source hammer for M1168R2; never runs VCS/EDA."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CONTRACT = HW / "contracts/m1168r2_m1171_m1168_m1162_c1_common_charge_protocol_vcs_compile_repair_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1168r2_m1171_m1168_m1162_c1_common_charge_protocol_vcs_compile_repair_source_author_receipt_r1_20260830"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1168r2_m1162_c1_common_charge_protocol_exact_sha_r2.sh"
STATIC = HW / "verif_m1168r2_c1_common_charge_protocol/static_check_m1168r2_m1162_vcs_source.py"
TB = HW / "verif_m1168r2_c1_common_charge_protocol/tb_m1168r2_m1162_common_charge_protocol_unit_delay_r2.sv"
SVA = HW / "verif_m1168r2_c1_common_charge_protocol/m1168r2_m1162_common_charge_protocol_assertions_r2.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
R1_TB = HW / "verif_m1168_c1_common_charge_protocol/tb_m1168_m1162_common_charge_protocol_unit_delay_r1.sv"
R1_SVA = HW / "verif_m1168_c1_common_charge_protocol/m1168_m1162_common_charge_protocol_assertions_r1.sv"
R1_FILELIST = HW / "dc_handoff/filelists/date_m1168_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
WRAPPER = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
PARENT = HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
R1_ATTEMPT = HW / "results/.m1168_m1162_c1_common_charge_protocol_vcs_r1_attempt_consumed"
R1_QUARANTINE = HW / "results/m1168_m1162_c1_common_charge_protocol_unit_delay_vcs_r1_20260830.failed_or_incomplete.3074649.quarantine"
R2_ATTEMPT = HW / "results/.m1168r2_m1162_c1_common_charge_protocol_vcs_r2_attempt_consumed"
R2_RESULT = HW / "results/m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs_r2_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    CONTRACT: "7abf99b60fce68ee0823b0e087f3276dccbc33b4d6921c5e6fe34bf3e16abe21",
    Path(str(CONTRACT) + ".sha256"): "d6f0e14eaf2a23a7369a86b9783b194b05c67cd9dbd5dfa2bb0ad5fe30e6c9f4",
    Path(str(CONTRACT) + ".sha256.seal.sha256"): "06c134e50fec169fd5609956fdc723d9ddfe9297ec132b5a4e29869bf0692d44",
    RUNNER: "4a661d50ca1929968b31258dd4950945bdd792311c090389f6a882e52aba58c3",
    STATIC: "022cf2d61d29cb22547db78de3dc8f5dbbbc8e0b03443c7469abd4f56d6beae8",
    TB: "bd5a2c3ce1ab9f03a7017756c96d5013577116583fc7d007ef3374593272ee35",
    SVA: "59ff9141175159e9043d86dd5932a4113fde88582005487f1eb65e372c6a684f",
    FILELIST: "96331eb20fb6d4e72e157d23c579841a121103053ed6246f0b76f812399f1411",
    R1_TB: "ae04c1c9e5104e4e4272632b0aa595fa2b8f93cef7c98ef40210afa0af7d28cc",
    R1_SVA: "9f7d4dcc9edb4ceb66469e2095fc4ae0043d625db309fb6fb00fc8fb197e261b",
    R1_FILELIST: "a6d0a90e0132771992dd5c5f9c3fc1e185020e724baa5eb0648632a7a0d593be",
    WRAPPER: "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    M935: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    PARENT: "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    R1_ATTEMPT / "identity.txt": "7b624fd913046f028506594e1b354bbb76c777a7c6467e1652c178fc7e05faae",
    R1_QUARANTINE / "RUN_FAILED_OR_INCOMPLETE.txt": "a93d50e1ee3170f2e688c250fe7f75861f79176ff4ca60b407a0fb07515e185b",
    R1_QUARANTINE / "compile.log": "39765d45f5e53de02a4c9139915253b0d0d8190f042027b70344dea08b0037ff",
    R1_QUARANTINE / "SHA256SUMS": "6f7d480bc752ea5835c3442de72f8e5e484ae41db3a5377b49e593e13838614c",
    R1_QUARANTINE / "SHA256SUMS.seal.sha256": "72ec416eb80888bb5c30a448c870b0859912097d43564662a3a88953182316c7",
    AUTHOR / "review.json": "7d5b94241eb726a9287619f69816c15d6ff76feac3f64cf0829806c41520c002",
    AUTHOR / "SHA256SUMS": "86e27f8170cdeabd05fa98549f04fb15ce6700256368d0fb79013322c0e49197",
    AUTHOR / "SHA256SUMS.seal.sha256": "acae14e78699d817cf20a989e41926c42fffc222c0a189481840f1a2557ca756",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

checks = 0
mutations = 0


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json_text(text: str):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(
        text,
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            RuntimeError("non-finite JSON: " + token)),
    )


def regular_exact(path: Path, digest: str) -> None:
    require(path.exists(), "missing identity: " + str(path))
    require(stat.S_ISREG(path.lstat().st_mode), "non-regular identity: " + str(path))
    require(not path.is_symlink(), "symlink identity: " + str(path))
    require(sha(path) == digest, "SHA drift: " + str(path))


def parse_manifest(directory: Path) -> dict[str, str]:
    result = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        require(bool(line.strip()), "blank manifest line")
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in result, "duplicate manifest member")
        result[name] = digest
    return result


def verify_recursive_seal(directory: Path, expected_manifest: str, expected_outer: str) -> None:
    regular_exact(directory / "SHA256SUMS", expected_manifest)
    regular_exact(directory / "SHA256SUMS.seal.sha256", expected_outer)
    listed = parse_manifest(directory)
    actual = set()
    for root, dirs, files in os.walk(directory, followlinks=False):
        root_path = Path(root)
        dirs[:] = [name for name in dirs if not (root_path / name).is_symlink()]
        for name in files:
            path = root_path / name
            if path.is_symlink():
                raise RuntimeError("symlink inside recursive seal")
            if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
                continue
            if stat.S_ISREG(path.lstat().st_mode):
                actual.add(str(path.relative_to(directory)))
    require(set(listed) == actual, "recursive seal membership mismatch")
    for name, digest in listed.items():
        require(sha(directory / name) == digest, "recursive seal member drift: " + name)
    outer_tokens = (directory / "SHA256SUMS.seal.sha256").read_text().split()
    require(outer_tokens == [expected_manifest, "SHA256SUMS"], "outer seal content mismatch")


STAGE_DECLS = (
    "logic force_stage_first_q, force_stage_last_q;",
    "logic [15:0] force_stage_epoch_q;",
    "logic [5:0] force_stage_row_q;",
    "logic [3:0] force_stage_source_q;",
)
STAGE_MAP = (
    ("force_stage_first_q", "first", "issue_request_first"),
    ("force_stage_last_q", "last", "issue_request_last"),
    ("force_stage_epoch_q", "epoch", "issue_request_epoch"),
    ("force_stage_row_q", "row", "issue_request_row_id"),
    ("force_stage_source_q", "source", "issue_request_source_index"),
)
FORCE_LIST = (
    ("issue_request_valid", "1'b1"),
    ("issue_request_epoch", "force_stage_epoch_q"),
    ("issue_request_row_id", "force_stage_row_q"),
    ("issue_request_first", "force_stage_first_q"),
    ("issue_request_last", "force_stage_last_q"),
    ("issue_request_source_valid", "1'b1"),
    ("issue_request_source_index", "force_stage_source_q"),
    ("issue_request_parent_valid", "1'b0"),
    ("issue_request_parent_id", "6'b0"),
    ("core_issue_data_ready", "1'b1"),
)


def force_request_body(tb: str) -> tuple[str, int]:
    match = re.search(r"task\s+automatic\s+force_request\s*\((.*?)endtask", tb, re.S)
    require(match is not None, "force_request task absent")
    return match.group(1), match.start()


def validate_force_staging(tb: str) -> None:
    body, task_start = force_request_body(tb)
    prefix = tb[:task_start]
    for decl in STAGE_DECLS:
        require(len(re.findall(r"^\s{4}" + re.escape(decl) + r"\s*$", prefix, re.M)) == 1,
                "module-scope staging declaration drift: " + decl)
    require("automatic logic" not in prefix, "automatic staging lifetime is forbidden")
    for stage, formal, target in STAGE_MAP:
        assignment = stage + " = " + formal + ";"
        forced = "force dut." + target + " = " + stage + ";"
        require(body.count(assignment) == 1, "stage assignment missing/duplicated: " + stage)
        require(tb.count(assignment) == 1, "stage variable assigned outside force_request: " + stage)
        require(body.count(forced) == 1, "stage force mapping missing/duplicated: " + stage)
        require(body.index(assignment) < body.index(forced), "force precedes assignment: " + stage)
    first_force = body.index("force dut.")
    for stage, formal, _ in STAGE_MAP:
        require(body.index(stage + " = " + formal + ";") < first_force,
                "all five staging fields must be assigned before any force")
    statements = re.findall(r"force\s+dut\.([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+?)\s*;", body)
    require(tuple(statements) == FORCE_LIST, "ten true hierarchical DUT force statements changed")
    require(len({target for target, _ in statements}) == 10, "force target alias/duplication")
    for _, rhs in statements:
        require(rhs not in {"first", "last", "epoch", "row", "source"},
                "automatic formal leaked to force RHS")
    require(body.count("force dut.") == 10, "hierarchical force cardinality changed")


def reject_tb_mutation(base: str, mutated: str, label: str) -> None:
    global mutations
    require(mutated != base, "mutation anchor absent: " + label)
    try:
        validate_force_staging(mutated)
    except RuntimeError:
        mutations += 1
        return
    raise RuntimeError("TB mutation accepted: " + label)


def validate_contract(data) -> None:
    require(data["status"] == "SOURCE_READY_FOR_FRESH_M1172_HAMMER__NO_VCS_RELEASE", "contract status")
    require(data["identity"] == {
        "runner_path": "dc_handoff/scripts/run_vcs_m1168r2_m1162_c1_common_charge_protocol_exact_sha_r2.sh",
        "runner_sha256": EXPECTED[RUNNER],
    }, "runner identity")
    forensic = data["r1_failure_forensics"]
    require(forensic["attempt_reusable"] is False, "old attempt made reusable")
    require(forensic["compile_exit_code"] == 255 and forensic["simulation_started"] is False,
            "r1 failure boundary drift")
    require(forensic["dtinpcil_errors"] == 5 and forensic["irfpca_autovar_errors"] == 5,
            "r1 error count drift")
    repair = data["repair"]
    require(repair == {
        "method": repair["method"],
        "module_scope_stage_fields": 5,
        "automatic_formals_on_force_rhs": 0,
        "hierarchical_dut_force_statements": 10,
        "all_stage_fields_assigned_before_force": True,
        "force_target_preserved": True,
        "lrm_compile_mode_mutations_rejected": 5,
        "functional_behavior_claimed": False,
    }, "repair contract drift")
    preserve = data["preserved_verification"]
    require(preserve == {
        "assert_properties": 16,
        "cover_properties": 6,
        "directed_protocol_cases": 18,
        "deterministic_random_transactions": 24,
        "protocol_attacks": 7,
        "service_assumption_attacks": 2,
        "reset_pending_states": 3,
        "minimum_completed_issue_ii_target": 2,
        "normal_frozen_m935_rows": 1,
        "normal_frozen_m935_tasks": 1,
    }, "verification preservation contract drift")
    unique = data["unique_r2_attempt"]
    require(unique["attempt_path"] == "results/.m1168r2_m1162_c1_common_charge_protocol_vcs_r2_attempt_consumed",
            "r2 attempt namespace drift")
    require(unique["result_path"] == "results/m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs_r2_20260830",
            "r2 result namespace drift")
    require(unique["r1_namespace_reuse"] is False and unique["single_attempt_after_future_release"] is True
            and unique["automatic_retry"] is False, "exactly-once contract weakened")
    gates = data["future_gates"]
    require(gates["fresh_different_author_hammer_required"] is True
            and gates["separate_release_required_after_hammer"] is True
            and gates["direct_r2_execution_now"] is False, "future gates weakened")
    require(data["authorization"]["vcs_compiles_now"] == 0
            and data["authorization"]["simv_runs_now"] == 0
            and data["authorization"]["all_eda_runs_now"] == 0,
            "source authorization widened")
    for key in ("functional_vcs_verified", "timing_verified", "cycles_measured", "speedup",
                "ppa", "power", "energy", "system_speedup", "paper_citable", "headline"):
        require(data["claim_boundary"][key] is False, "premature claim opened: " + key)


def reject_contract_mutation(base, mutate, label: str) -> None:
    global mutations
    trial = copy.deepcopy(base)
    mutate(trial)
    try:
        validate_contract(trial)
    except (KeyError, RuntimeError, TypeError):
        mutations += 1
        return
    raise RuntimeError("contract mutation accepted: " + label)


def transform_r1_tb_to_expected_r2(r1: str) -> str:
    replacements = (
        (
            "// M1168 source-only verification package for the repaired M1162 common-charge\n"
            "// protocol.  This file is intentionally not executed by its author.  A fresh\n"
            "// different-author hammer and a separately sealed release must precede the\n"
            "// single permitted foundry UNIT_DELAY VCS attempt.\n"
            "module tb_m1168_m1162_common_charge_protocol_unit_delay_r1;",
            "// M1168R2 source-only repair of the M1168 verification package.  The original\n"
            "// r1 compile failed because a procedural force RHS referenced automatic task\n"
            "// arguments.  R2 stages those values into module-scope variables before it\n"
            "// forces the same DUT-internal request state.  This file is intentionally not\n"
            "// executed by its author.  A fresh different-author hammer and a separately\n"
            "// sealed release must precede the single permitted r2 UNIT_DELAY VCS attempt.\n"
            "module tb_m1168r2_m1162_common_charge_protocol_unit_delay_r2;",
        ),
        (
            "    integer unsigned prng_q;\n\n",
            "    integer unsigned prng_q;\n\n"
            "    // LRM-legal procedural-force staging.  These variables have static module\n"
            "    // lifetime; none is an automatic task formal.  Every call assigns all five\n"
            "    // fields before the hierarchical DUT force statements below.  Calls are\n"
            "    // sequential and release_request ends the prior force before reuse.\n"
            "    logic force_stage_first_q, force_stage_last_q;\n"
            "    logic [15:0] force_stage_epoch_q;\n"
            "    logic [5:0] force_stage_row_q;\n"
            "    logic [3:0] force_stage_source_q;\n\n",
        ),
        ("m1168_m1162_common_charge_protocol_assertions_r1 u_protocol_sva",
         "m1168r2_m1162_common_charge_protocol_assertions_r2 u_protocol_sva"),
        (
            "        begin\n"
            "            force dut.issue_request_valid = 1'b1;\n"
            "            force dut.issue_request_epoch = epoch;\n"
            "            force dut.issue_request_row_id = row;\n"
            "            force dut.issue_request_first = first;\n"
            "            force dut.issue_request_last = last;\n"
            "            force dut.issue_request_source_valid = 1'b1;\n"
            "            force dut.issue_request_source_index = source;\n",
            "        begin\n"
            "            force_stage_first_q = first;\n"
            "            force_stage_last_q = last;\n"
            "            force_stage_epoch_q = epoch;\n"
            "            force_stage_row_q = row;\n"
            "            force_stage_source_q = source;\n"
            "            force dut.issue_request_valid = 1'b1;\n"
            "            force dut.issue_request_epoch = force_stage_epoch_q;\n"
            "            force dut.issue_request_row_id = force_stage_row_q;\n"
            "            force dut.issue_request_first = force_stage_first_q;\n"
            "            force dut.issue_request_last = force_stage_last_q;\n"
            "            force dut.issue_request_source_valid = 1'b1;\n"
            "            force dut.issue_request_source_index = force_stage_source_q;\n",
        ),
        ("COVERAGE_M1168_PROTOCOL", "COVERAGE_M1168R2_PROTOCOL"),
        ("COVERAGE_M1168_RESETS_ATTACKS", "COVERAGE_M1168R2_RESETS_ATTACKS"),
        ("COVERAGE_M1168_SERVICE_ASSUMPTIONS", "COVERAGE_M1168R2_SERVICE_ASSUMPTIONS"),
        ("COVERAGE_M1168_FROZEN_M935", "COVERAGE_M1168R2_FROZEN_M935"),
        ("PASS_M1168_M1162_COMMON_CHARGE_PROTOCOL_UNIT_DELAY_CANDIDATE",
         "PASS_M1168R2_M1162_COMMON_CHARGE_PROTOCOL_UNIT_DELAY_CANDIDATE"),
    )
    result = r1
    for old, new in replacements:
        require(result.count(old) == 1, "r1 transform anchor drift")
        result = result.replace(old, new, 1)
    return result


def validate_runner(runner: str) -> None:
    active_runner = "\n".join(line for line in runner.splitlines()
                              if not line.lstrip().startswith("#"))
    for token in (
        "M1168R2_EXPECTED_RELEASE_SHA256",
        "M1168R2_EXPECTED_HAMMER_REVIEW_SHA256",
        "M1168R2_EXPECTED_HAMMER_OUTER_SHA256",
        "verify_recursive_seal \"${R1_QUARANTINE}\"",
        "verify_recursive_seal \"${HAMMER_DIR}\"",
        "sha_exact \"${M1168R2_EXPECTED_RELEASE_SHA256}\" \"${RELEASE}\"",
        "SOURCE_READY_FOR_FRESH_M1172_HAMMER__NO_VCS_RELEASE",
        "PASS_M1172_M1168R2_VCS_SOURCE_HAMMER__AUTHORIZE_RELEASE",
        "AUTHORIZE_EXACTLY_ONE_M1168R2_FUNCTIONAL_VCS_ATTEMPT",
        "[[ ! -e \"${RESULT}\" && ! -e \"${ATTEMPT}\" && ! -e \"${WORK}\" ]]",
        "mv -- \"${WORK}\" \"${RESULT}\"",
    ):
        require(token in runner, "runner gate absent: " + token)
    require(runner.count('"${VCS_BIN}" -full64') == 1, "VCS compile cardinality is not one")
    require(runner.count('./simv -no_save') == 1, "simv cardinality is not one")
    require(runner.count('mkdir -- "${ATTEMPT}"') == 1, "attempt consume cardinality is not one")
    require("while true" not in active_runner and "until " not in active_runner
            and "retry" not in active_runner.lower(),
            "automatic retry loop/text found")
    require('ATTEMPT="${HW_ROOT}/results/.m1168r2_m1162_c1_common_charge_protocol_vcs_r2_attempt_consumed"' in runner,
            "fresh r2 attempt path absent")
    require('RESULT="${HW_ROOT}/results/m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs_r2_20260830"' in runner,
            "fresh r2 result path absent")
    require('R1_ATTEMPT_ID="${HW_ROOT}/results/.m1168_m1162_c1_common_charge_protocol_vcs_r1_attempt_consumed/identity.txt"' in runner,
            "r1 forensics path absent")
    require('ATTEMPT="${HW_ROOT}/results/.m1168_m1162_' not in runner,
            "old attempt namespace reused as write target")
    require(runner.index('verify_recursive_seal "${HAMMER_DIR}"') < runner.index('mkdir -- "${ATTEMPT}"'),
            "attempt consumed before hammer verification")
    require(runner.index('sha_exact "${M1168R2_EXPECTED_RELEASE_SHA256}" "${RELEASE}"') < runner.index('mkdir -- "${ATTEMPT}"'),
            "attempt consumed before release verification")
    for path, digest in ((TB, EXPECTED[TB]), (SVA, EXPECTED[SVA]),
                         (FILELIST, EXPECTED[FILELIST]), (STATIC, EXPECTED[STATIC])):
        require(("sha_exact " + digest) in runner and str(path.relative_to(HW)) in runner,
                "runner source SHA pin absent: " + str(path))


def main() -> None:
    for path, digest in EXPECTED.items():
        regular_exact(path, digest)
    require(Path(str(CONTRACT) + ".sha256").read_text().split() == [EXPECTED[CONTRACT], CONTRACT.name],
            "contract sidecar content")
    require(Path(str(CONTRACT) + ".sha256.seal.sha256").read_text().split() ==
            [EXPECTED[Path(str(CONTRACT) + ".sha256")], CONTRACT.name + ".sha256"],
            "contract outer sidecar content")
    verify_recursive_seal(AUTHOR, EXPECTED[AUTHOR / "SHA256SUMS"],
                          EXPECTED[AUTHOR / "SHA256SUMS.seal.sha256"])
    verify_recursive_seal(R1_QUARANTINE, EXPECTED[R1_QUARANTINE / "SHA256SUMS"],
                          EXPECTED[R1_QUARANTINE / "SHA256SUMS.seal.sha256"])
    require(parse_manifest(R1_QUARANTINE) == {
        "RUN_FAILED_OR_INCOMPLETE.txt": EXPECTED[R1_QUARANTINE / "RUN_FAILED_OR_INCOMPLETE.txt"],
        "compile.log": EXPECTED[R1_QUARANTINE / "compile.log"],
    }, "r1 quarantine membership is not the exact failed compile")
    compile_log = (R1_QUARANTINE / "compile.log").read_text()
    require(compile_log.count("Error-[DTINPCIL]") == 5, "r1 DTINPCIL count drift")
    require(compile_log.count("Error-[IRFPCA-AUTOVAR]") == 5, "r1 AUTOVAR count drift")
    require("Automatic variable may not be used in non-procedural context" in compile_log,
            "r1 root-cause evidence absent")
    require(not os.path.lexists(R2_ATTEMPT), "r2 attempt already consumed")
    require(not os.path.lexists(R2_RESULT), "r2 result namespace already exists")
    require(not list((HW / "results").glob(".m1168r2_m1162_c1_common_charge_protocol_vcs_r2_work.*")),
            "r2 work namespace already exists")

    contract = strict_json_text(CONTRACT.read_text())
    validate_contract(contract)
    for mutation, label in (
        (lambda d: d["r1_failure_forensics"].__setitem__("attempt_reusable", True), "old attempt reusable"),
        (lambda d: d["repair"].__setitem__("automatic_formals_on_force_rhs", 1), "automatic RHS"),
        (lambda d: d["repair"].__setitem__("all_stage_fields_assigned_before_force", False), "missing stage"),
        (lambda d: d["repair"].__setitem__("force_target_preserved", False), "target drift"),
        (lambda d: d["unique_r2_attempt"].__setitem__("attempt_path", "results/.m1168_m1162_c1_common_charge_protocol_vcs_r1_attempt_consumed"), "old namespace"),
        (lambda d: d["unique_r2_attempt"].__setitem__("automatic_retry", True), "retry"),
        (lambda d: d["future_gates"].__setitem__("separate_release_required_after_hammer", False), "release bypass"),
        (lambda d: d["authorization"].__setitem__("vcs_compiles_now", 1), "premature compile"),
        (lambda d: d["claim_boundary"].__setitem__("functional_vcs_verified", True), "premature VCS claim"),
    ):
        reject_contract_mutation(contract, mutation, label)

    author = strict_json_text((AUTHOR / "review.json").read_text())
    require(author["status"] == "PASS_M1168R2_COMPILE_REPAIR_SOURCE_ONLY__FRESH_M1172_HAMMER_AND_M1173_RELEASE_REQUIRED__NO_VCS_NO_EDA",
            "author receipt status drift")
    require(author["verdict"] == "GO_FRESH_DIFFERENT_AUTHOR_SOURCE_HAMMER_ONLY__NO_DIRECT_EXECUTION",
            "author verdict drift")
    require(author["issue_counts"] == {"P0": 0, "P1": 0, "P2": 0}, "author issue counts")
    require(author["execution_audit"] == {
        "runner_invocations": 0, "vcs_compiles": 0, "simv_runs": 0,
        "all_eda_runs": 0, "license_queries": 0, "attempts_consumed": 0,
        "results_created": 0,
    }, "author source-only boundary drift")

    tb = TB.read_text()
    validate_force_staging(tb)
    for old, new, label in (
        ("force_stage_epoch_q = epoch;", "", "staging assignment deleted"),
        ("force_stage_row_q = row;", "force_stage_row_q = epoch;", "staging alias"),
        ("force dut.issue_request_epoch = force_stage_epoch_q;", "force dut.issue_request_epoch = epoch;", "automatic formal RHS"),
        ("force dut.issue_request_row_id = force_stage_row_q;", "force dut.issue_request_parent_id = force_stage_row_q;", "force target changed"),
        ("force dut.issue_request_first = force_stage_first_q;", "force dut.issue_request_last = force_stage_first_q;", "force target duplicated"),
        ("force_stage_source_q = source;", "force_stage_source_q = row;", "source alias"),
        ("logic [15:0] force_stage_epoch_q;", "automatic logic [15:0] force_stage_epoch_q;", "stage lifetime changed"),
    ):
        reject_tb_mutation(tb, tb.replace(old, new, 1), label)
    require(transform_r1_tb_to_expected_r2(R1_TB.read_text()) == tb,
            "r2 TB changed behavior beyond the five staging repair plus identity strings")
    require(SVA.read_text() == R1_SVA.read_text().replace(
        "module m1168_m1162_common_charge_protocol_assertions_r1 (",
        "module m1168r2_m1162_common_charge_protocol_assertions_r2 (", 1),
        "r2 SVA changed beyond module identity")
    expected_filelist = R1_FILELIST.read_text().replace(
        "/verif_m1168_c1_common_charge_protocol/m1168_m1162_common_charge_protocol_assertions_r1.sv",
        "/verif_m1168r2_c1_common_charge_protocol/m1168r2_m1162_common_charge_protocol_assertions_r2.sv",
    ).replace(
        "/verif_m1168_c1_common_charge_protocol/tb_m1168_m1162_common_charge_protocol_unit_delay_r1.sv",
        "/verif_m1168r2_c1_common_charge_protocol/tb_m1168r2_m1162_common_charge_protocol_unit_delay_r2.sv",
    )
    require(FILELIST.read_text() == expected_filelist, "r2 filelist changed beyond TB/SVA identities")
    sva = SVA.read_text()
    require(sva.count("assert property") == 16, "assertion count drift")
    require(sva.count("cover property") == 6, "cover count drift")
    for token in (
        "directed_weight_first();", "directed_psum_first_and_backpressure();",
        "directed_nonfirst();", "directed_ii2();", "reset_pending_cases();",
        "sticky_fault_attacks();", "service_assumption_attacks();",
        "normal_m935_completion();",
        "random_legal_transaction(test_index);",
        "protocol_attacks=7", "service_assumption_attacks=2", "reset_states=3",
        "ii=2", "normal_m935_rows=1", "normal_m935_tasks=1",
    ):
        require(token in tb, "preserved verification token absent: " + token)
    require(re.search(r"for \(integer test_index = 0; test_index < 24;\s*"
                      r"test_index = test_index \+ 1\)", tb),
            "24-transaction deterministic random loop absent")

    runner = RUNNER.read_text()
    validate_runner(runner)
    for old, new, label in (
        ('.m1168r2_m1162_c1_common_charge_protocol_vcs_r2_attempt_consumed',
         '.m1168_m1162_c1_common_charge_protocol_vcs_r1_attempt_consumed', "old attempt reuse"),
        ('verify_recursive_seal "${HAMMER_DIR}"', ': # hammer bypass', "hammer bypass"),
        ('sha_exact "${M1168R2_EXPECTED_RELEASE_SHA256}" "${RELEASE}"', ': # release bypass', "release bypass"),
    ):
        mutated = runner.replace(old, new, 1)
        require(mutated != runner, "runner mutation anchor absent")
        try:
            validate_runner(mutated)
        except RuntimeError:
            global mutations
            mutations += 1
        else:
            raise RuntimeError("runner mutation accepted: " + label)

    print(json.dumps({
        "schema": "m1172_m1168r2_compile_repair_independent_source_hammer_r1_v1",
        "status": "PASS_M1172_M1168R2_VCS_SOURCE_HAMMER__AUTHORIZE_RELEASE",
        "verdict": "GO",
        "score": 99,
        "checks_passed": checks,
        "mutations_rejected": mutations,
        "force_repair": {
            "module_scope_staging_fields": 5,
            "all_staging_fields_assigned_before_first_force": True,
            "automatic_task_formals_on_force_rhs": 0,
            "true_hierarchical_dut_forces": 10,
        },
        "preserved_verification": {
            "assertions": 16, "covers": 6, "directed_cases": 18,
            "random_legal_transactions": 24, "protocol_attacks": 7,
            "service_attacks": 2, "reset_states": 3, "minimum_ii": 2,
            "normal_m935_rows": 1, "normal_m935_tasks": 1,
        },
        "namespace": {"r1_attempt_reusable": False, "r2_namespace_fresh": True,
                      "future_runner_exactly_once": True},
        "execution_audit": {"runner_invocations": 0, "vcs_compiles": 0,
                            "simv_runs": 0, "eda_runs": 0, "license_queries": 0},
        "claim_boundary": {"functional_vcs_verified": False, "timing_verified": False,
                           "cycles_measured": False, "speedup": False, "ppa": False,
                           "power": False, "energy": False, "system_speedup": False,
                           "paper_citable": False, "headline": False},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
