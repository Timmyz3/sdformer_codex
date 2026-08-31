#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author M1336 C2 release-source hammer; never launches EDA."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1336_c2_headline_mapped_production_activity_one_shot_exact_sha.sh"
CONTRACT = HW / "contracts/m1336_c2_headline_mapped_production_activity_vcs_release_source_contract_r1_20260831.json"
CONTRACT_SUM = Path(str(CONTRACT) + ".sha256")
CONTRACT_OUTER = Path(str(CONTRACT) + ".sha256.seal.sha256")
CHECKER = HW / "verif_m1336_c2_activity_release/static_check_m1336_c2_activity_vcs_release_source.py"
TEST = HW / "verif_m1336_c2_activity_release/test_m1336_c2_activity_vcs_release_source.py"
AUTHOR = HW / "reviews/m1336_c2_headline_mapped_production_activity_vcs_release_source_author_r1_20260831"
M1334_AUTHOR = HW / "reviews/m1334_c2_headline_mapped_production_activity_source_author_r1_20260831"
M1335_BLIND = HW / "reviews/m1335_m1334_c2_headline_mapped_production_activity_source_blind_hammer_r1_20260831"
UCLI = HW / "dc_handoff/scripts/m1334_c2_headline_mapped_production_activity.ucli.tcl"
SOURCE_CHECKER = HW / "system_simulator/scripts/check_m1334_c2_headline_mapped_production_activity_source.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUTPUT = Path(__file__).resolve().parent / "hammer_output.json"

EXPECTED = {
    RUNNER: "bf887a12aa5687f34a3d2198cbec3f0c6bcc4ed9aa47a38cfd6853128935bd35",
    CONTRACT: "cc0ea28407da69ef87a9270a9982615cde182b7a195564f5a4e26d0c6d9f0288",
    CONTRACT_SUM: "6860b9a35e914ea68ea578c830497ecc76e4463dbf9af85884e823ff51f6514a",
    CONTRACT_OUTER: "6ddf6d4e4f4588a77055bb52210e4fcaa7fc86724b8eb314883a9633f69bb9d1",
    CHECKER: "6c2f67051a5ae9796f2f95a5f0dd905b5e8f9d1fe07364dbe5442a207dc24c38",
    TEST: "daf8b5c6991c31fa4597b555e7b0e710c26cfe5bb3cc3119c5c26048a3653ca7",
    UCLI: "c90153dfd58ff4e653852a54b31ad3b19cb8fabd993e15c21d9071b555cbebc1",
    SOURCE_CHECKER: "c9326ff934239e8773e9f991e6bf0be94bba9c9c602be199433c22d1cd4c9da9",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load("m1337_release_checker", CHECKER)


def result_namespaces() -> list[Path]:
    result = HW / "results/m1336_c2_headline_mapped_production_activity_vcs_r1_20260831"
    return [
        HW / "results/.m1336_c2_headline_mapped_production_activity_vcs_attempt_consumed",
        result,
        Path(str(result) + ".failed_or_incomplete.quarantine"),
        Path(str(result) + ".private_build.unsealed_do_not_cite"),
        Path(str(result) + ".failed_private_build.unsealed_do_not_cite"),
    ]


def main() -> None:
    for path, digest in EXPECTED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "identity drift: " + str(path))
    M.verify_dir(M1334_AUTHOR)
    M.verify_dir(M1335_BLIND)
    M.verify_dir(AUTHOR)
    author_rows = {}
    for line in (AUTHOR / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        author_rows[name.lstrip("*")] = digest
    need(author_rows.get("review.json") ==
         "a628086eb3cb811977306deda9ae4f26a4d9293f93db574f84e5a4d5f99facf4",
         "author review drift")
    need(sha(AUTHOR / "SHA256SUMS") ==
         "38f21ed864cf4a2a5cf0b8c5ede173e4f4d077f7715404624be614ab87002977",
         "author manifest drift")
    need(sha(AUTHOR / "SHA256SUMS.seal.sha256") ==
         "62cc3cbfd5301f1dfec550dcdbdafb6b5acb8d2a0a4fc312b83998056851b1bc",
         "author outer drift")

    contract = json.loads(CONTRACT.read_text())
    author = json.loads((AUTHOR / "review.json").read_text())
    blind = json.loads((M1335_BLIND / "review.json").read_text())
    need(blind["status"] ==
         "PASS_SOURCE_ADMITTED__EXACT_SHA_VCS_RELEASE_CONTRACT_MAY_BE_AUTHORED" and
         blind["predecessor"]["remaining_false_negative_count"] == 0,
         "M1335 blind PASS root drift")
    need(contract["identity"] == {
        "runner": "dc_handoff/scripts/run_vcs_m1336_c2_headline_mapped_production_activity_one_shot_exact_sha.sh",
        "runner_sha256": EXPECTED[RUNNER],
        "checker": "verif_m1336_c2_activity_release/static_check_m1336_c2_activity_vcs_release_source.py",
        "checker_sha256": EXPECTED[CHECKER],
        "test": "verif_m1336_c2_activity_release/test_m1336_c2_activity_vcs_release_source.py",
        "test_sha256": EXPECTED[TEST]}, "contract identity drift")
    need(author["bindings"]["runner_sha256"] == EXPECTED[RUNNER] and
         author["bindings"]["source_contract_sha256"] == EXPECTED[CONTRACT],
         "author binding drift")
    need(contract["workloads"] == {
        "events": [20, 41, 90, 110, 0],
        "k8_cycles": [51, 131, 486, 1231, 14],
        "k1x8_cycles": [53, 133, 499, 1246, 14]}, "workload anchor drift")
    need(contract["future_execution"]["vcs_compiles"] == 2 and
         contract["future_execution"]["simv_runs"] == 10 and
         contract["future_execution"]["automatic_retry"] is False,
         "future cardinality/retry drift")
    need(all(contract["claim_boundary"][key] is False for key in
             ("functional_vcs_verified", "production_saif", "ptpx", "power",
              "energy", "performance", "system_speedup", "paper_ppa_ready",
              "headline")), "claim boundary drift")
    need(all(not os.path.lexists(str(path)) for path in result_namespaces()),
         "result/attempt namespace already consumed")

    runner = RUNNER.read_text()
    mutations = {
        "attempt_before_resource": runner.replace(
            'phase="RESOURCE_PREFLIGHT"', 'phase="ATTEMPT_CONSUME"', 1),
        "same_uid_vcs1_removed": runner.replace("'vcs1',", "", 1),
        "compile_count_drift": runner.replace('compile_count=$((compile_count+1))',
                                               'compile_count=$((compile_count+2))', 1),
        "sim_count_drift": runner.replace('sim_count=$((sim_count+1))',
                                           'sim_count=$((sim_count+2))', 1),
        "axis_removed": runner.replace("for axis in k8 k1x8; do", "for axis in k8; do", 1),
        "workload_removed": runner.replace("for case_id in 0 1 2 3 4; do",
                                            "for case_id in 0 1 2 3; do", 1),
        "k8_cycle_drift": runner.replace("cycles=(51 131 486 1231 14)",
                                          "cycles=(51 131 486 1230 14)", 1),
        "k1x8_cycle_drift": runner.replace("cycles=(53 133 499 1246 14)",
                                            "cycles=(53 133 499 1245 14)", 1),
        "event_anchor_drift": runner.replace("events=(20 41 90 110 0)",
                                              "events=(20 41 90 109 0)", 1),
        "saif_scope_env_drift": runner.replace("M1334_SAIF_FILE", "M1334_BAD_SAIF_FILE", 1),
        "failure_seal_removed": runner.replace('seal_dir "${FAILURE_STAGE}"', ":", 1),
        "success_seal_removed": runner.replace('seal_dir "${RESULT_STAGE}"', ":", 1),
        "rename_fallback_overwrite": runner.replace("publish_no_replace", "mv_overwrite", 1),
        "retry_enabled": runner.replace("automatic_retry=false", "automatic_retry=true", 1),
        "release_sha_gate_missing": runner.replace(
            "M1336_EXPECTED_LAUNCH_RELEASE_SHA256 \\", "M1336_OMITTED_RELEASE_SHA \\", 1),
        "ucli_workspace_key_read": runner.replace("-no_save", "-save ucli.key", 1),
        "pass_token_drift": runner.replace(
            "PASS_M1336_C2_MAPPED_PRODUCTION_ACTIVITY_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
            "PASS_M1336_PERFORMANCE_VERIFIED", 1),
        "performance_boundary_lifted": runner.replace("'performance':False",
                                                       "'performance':True", 1),
    }
    rejected = []
    for label, mutant in mutations.items():
        need(mutant != runner, "mutation did not change runner: " + label)
        need(not M.exact_runner_gate(mutant, contract["identity"]["runner_sha256"]),
             "runner mutation escaped exact SHA: " + label)
        rejected.append(label)

    # P0: the runner requires all three future artifacts before invoking a
    # source-only checker whose own main requires those paths all to be absent.
    verify_source_hammer = runner.index('verify_recursive_seal "${SOURCE_HAMMER}"')
    verify_launch = runner.index('verify_file_sidecar "${LAUNCH_RELEASE}"')
    verify_final = runner.index('verify_recursive_seal "${FINAL_HAMMER}"')
    invoke_release_checker = runner.index('"${PYTHON}" -I "${RELEASE_CHECKER}"')
    need(max(verify_source_hammer, verify_launch, verify_final) < invoke_release_checker,
         "runner future-chain order changed")
    checker_text = CHECKER.read_text()
    need('assert all(not os.path.lexists(path) for path in future)' in checker_text,
         "source-only future-absence assertion changed")

    # P1: neither attempt.txt nor candidate receipt binds the source-hammer or
    # final-hammer review/manifest/outer SHA used for this execution.
    attempt_block = runner[runner.index("printf 'status=M1336_ATTEMPT_CONSUMED"):
                           runner.index('seal_dir "${ATTEMPT_STAGE}"')]
    receipt_block = runner[runner.index("d={'schema':'m1336_c2_mapped_production_activity"):
                           runner.index("out.write_text", runner.index("d={'schema':'m1336_c2_mapped_production_activity"))]
    for block, label in ((attempt_block, "attempt"), (receipt_block, "candidate receipt")):
        need("source_hammer" not in block.lower() and "final_hammer" not in block.lower(),
             label + " unexpectedly binds full authorization chain")

    OUTPUT.write_text(json.dumps({
        "status": "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED",
        "author_static_checks": "80/80 PASS",
        "author_directed_tests": "10/10 PASS",
        "independent_exact_sha_mutations_rejected": rejected,
        "independent_mutation_count": len(rejected),
        "false_negative_count": 2,
        "p0_runtime_checker_future_chain_contradiction": True,
        "p1_attempt_and_candidate_receipt_missing_full_release_chain_sha": True,
        "license_queries": 0,
        "vcs_runs": 0,
        "simv_runs": 0,
        "saif_runs": 0,
        "eda_runs": 0,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("M1337_FAIL_M1336_C2_RELEASE_SOURCE__P0_RUNTIME_CHECKER_CONTRADICTION")


if __name__ == "__main__":
    main()
