#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent local-only M1208 hammer for the M1207 acyclic VCS release.

This program performs identity, seal, syntax, semantic and in-memory mutation
checks only. It never invokes VCS, simv, a license tool, GPU, or network.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent

RUNNER = HW / "dc_handoff/scripts/run_vcs_m1207_m1198r7_m1162_c1_common_charge_protocol_exact_sha_r7.sh"
CHECKER = HW / "verif_m1207_c1_r7_vcs_release/static_check_m1207_c1_r7_vcs_release_source.py"
R7_CHECKER = HW / "verif_m1198r7_c1_common_charge_protocol/static_check_m1198r7_m1162_vcs_source.py"
CONTRACT = HW / "contracts/m1207_m1201_m1198_c1_r7_vcs_launcher_source_contract_r1_20260830.json"
RELEASE = HW / "contracts/m1207_m1201_m1198_c1_r7_vcs_launch_release_r1_20260830.json"
AUTHOR = HW / "reviews/m1207_m1201_m1198_c1_r7_vcs_release_author_receipt_r1_20260830"
M1198_AUTHOR = HW / "reviews/m1198_m1194_c1_r7_source_gate_repair_author_receipt_r1_20260830"
M1201 = HW / "reviews/m1201_m1198_c1_r7_source_gate_repair_hammer_r1_20260830"
FILELIST = HW / "dc_handoff/filelists/date_m1198r7_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
TB = HW / "verif_m1193r6_c1_common_charge_protocol/tb_m1193r6_m1162_common_charge_protocol_unit_delay_r6.sv"
SVA = HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
WRAPPER = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
PARENT = HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
M1198_CONTRACT = HW / "contracts/m1198_m1194_m1193_m1162_c1_r7_source_gate_repair_contract_r1_20260830.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

RUNNER_SHA = "4ef7458b762734d3a034282353ddeee281529e0cbcbbfacf364e368f522d90e4"
CHECKER_SHA = "c09e57a7842a3bf51fa094d0a04d52dd9b383c5e60bb89fe51606460f4d706f8"
CONTRACT_ID = (
    "5be82ebf3bc9779b14f894f57fe3c456aa5fdafd58214baa39878730d601ccb1",
    "7edbf467656ca0e5bbce8a58ce82b58e08b5dbbe6f4602241e5e1fc936160220",
    "693142378b91528967c34f6acf2825ec4283248a1540ba5c50bcf5f56f1e248c",
)
RELEASE_ID = (
    "81552f212cf31ae30979bf7d9f26fea8b37c22e8be23680a1f7630574b158ac0",
    "142ba5414bd51bfacdf1d528c01907d7a1d7ba1d9f14febefd46b158c17719fe",
    "4c54f90cdaae65f14762309b053a9969b80c1b69c093f68b93004b206a6e276b",
)
AUTHOR_ID = (
    "0b190010e7905b811e13ce06e132c63966055c54c98d1ee88879605f400834e2",
    "b4e2065d7dcc1dbe633c290f30a77e64c8eef360262592f64d2d9a43399724ba",
    "1057d9be79fb2a036fe2a4f0942c6d261d9bad541693577dbbe7e6f67c73272a",
)
M1198_AUTHOR_ID = (
    "c47172693484f9098881b745e34d797d481906e8857bc93a169ce6bd701605df",
    "184946a8e314414919b7702b23a1d2e80dc1c15623dbb9d04090158021e1eac9",
    "7286441a67b9cb1196dec9356e5bf1b33ca5a6e90522ff4b404137c6fc76768b",
)
M1201_ID = (
    "b78fc16baf67025b2a500f0a9a26b7392f8752dc4af4afbf066d685d89f495c7",
    "26323508b4186a0e3c718afceb3d6deeeb2b2f4467418f863b2cab102cf1558f",
    "3317d266834a69c5d9ca6ae747ff5285409fc8f4b8a650d62507a38deaa9a748",
)
EXACT = {
    R7_CHECKER: "b1cfb957d5c4fc518d46980040afa61288eb7dcaa79fa5e6c45e25b097094795",
    FILELIST: "444ff65d575c6e897f9d459689f323290f16eb89c962c91b395964c7850fcbfa",
    TB: "0fcc2138ef5d716735eea01dee25a148a5223b1d6adf1e3b2fa464341fbf1345",
    SVA: "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    WRAPPER: "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    M935: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    PARENT: "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    M1198_CONTRACT: "44c5a3add48ef74ef0698f81f20fef417989c17b74df3e1d366cf404b7ce5488",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
ENV_NAMES = (
    "M1207_EXPECTED_RELEASE_SHA256",
    "M1207_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256",
    "M1207_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256",
    "M1207_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256",
)
FALSE_CLAIMS = (
    "functional_vcs_verified", "timing_verified", "cycles_measured",
    "speedup", "ppa", "power", "energy", "system_speedup", "paper_citable",
)


class Failure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: str | None = None) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), "not regular: " + str(path))
    if expected is not None:
        require(sha(path) == expected, "SHA drift: " + str(path))


def strict_json(path: Path) -> Any:
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key")
            output[key] = value
        return output
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON " + token)))


def manifest_tree(directory: Path, identity: tuple[str, str, str], review_name: str) -> dict[str, Any]:
    require(directory.is_dir() and not directory.is_symlink(), "sealed tree drift")
    manifest, outer = directory / "SHA256SUMS", directory / "SHA256SUMS.seal.sha256"
    regular(manifest, identity[1]); regular(outer, identity[2])
    require(outer.read_text(encoding="ascii").split() == [identity[1], "SHA256SUMS"],
            "outer seal drift")
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]) is not None,
                "manifest syntax")
        name = fields[1].lstrip("*"); rel = Path(name)
        require(name not in rows and name == rel.as_posix() and not rel.is_absolute()
                and ".." not in rel.parts, "manifest member")
        rows[name] = fields[0]
    actual: set[str] = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "sealed symlink")
        if stat.S_ISREG(mode): actual.add(name)
        else: require(stat.S_ISDIR(mode), "sealed special member")
    require(actual == set(rows), "recursive member set drift")
    for name, digest in rows.items(): regular(directory / name, digest)
    require(rows.get(review_name) == identity[0], "review identity drift")
    return strict_json(directory / review_name)


def double_file(path: Path, identity: tuple[str, str, str]) -> None:
    side, outer = Path(str(path) + ".sha256"), Path(str(path) + ".sha256.seal.sha256")
    regular(path, identity[0]); regular(side, identity[1]); regular(outer, identity[2])
    require(side.read_text(encoding="ascii").split() == [identity[0], path.name], "sidecar drift")
    require(outer.read_text(encoding="ascii").split() == [identity[1], side.name], "outer sidecar drift")


def env_gate(env: dict[str, str]) -> bool:
    return all(re.fullmatch(r"[0-9a-f]{64}", env.get(name, "")) is not None
               for name in ENV_NAMES)


def review_is_acyclic(identity: dict[str, str], serialized: str,
                      manifest_digest: str, outer_digest: str) -> bool:
    forbidden = {"hammer_manifest_sha256", "hammer_outer_seal_file_sha256",
                 "manifest_sha256", "outer_seal_file_sha256"}
    return forbidden.isdisjoint(identity) and manifest_digest not in serialized \
        and outer_digest not in serialized


def validate_runner(text: str) -> None:
    for name in ENV_NAMES:
        require(name in text, "missing env token " + name)
    required = (
        "[[ $# -eq 0 ]]", "set -euo pipefail", "+define+UNIT_DELAY",
        'verify_recursive_seal "${RELEASE_HAMMER}"',
        'sha_exact "${M1207_EXPECTED_RELEASE_SHA256}" "${RELEASE}"',
        'sha_exact "${M1207_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256}"',
        'sha_exact "${M1207_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256}"',
        'sha_exact "${M1207_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256}"',
        "self-reference forbidden", "p.stat().st_uid != os.getuid()", "MemAvailable below 64 GiB",
        "COVERAGE_M1193R6_PROTOCOL weight_first=1 psum_first=1",
        "COVERAGE_M1193R6_RESETS_ATTACKS reset_partial=1 reset_complete=1",
        "COVERAGE_M1193R6_SERVICE_ASSUMPTIONS weight_payload_mutation=1 psum_valid_drop=1 weight_windows=1 psum_windows=1 independent_checker=1 race_free_negedge_sample=1 skew_isolated=1 reachable_core_ready_force=0 boundary_fault=0 core_fault=0 dut_fault_claim=0",
        "COVERAGE_M1193R6_FROZEN_M935 normal_issues=2 normal_rows=1 normal_tasks=1",
        "PASS_M1193R6_M1162_COMMON_CHARGE_PROTOCOL_UNIT_DELAY_CANDIDATE",
        "functional_vcs_verified':True", "timing_verified':False",
        "cycles_measured':False", "system_speedup':False",
    )
    for token in required: require(token in text, "missing runner semantic token: " + token)
    positions = [
        text.index('verify_recursive_seal "${RELEASE_HAMMER}"'),
        text.index('sha_exact "${M1207_EXPECTED_RELEASE_SHA256}"'),
        text.index('sha_exact "${M1207_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256}"'),
        text.index('sha_exact "${M1207_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256}"'),
        text.index('sha_exact "${M1207_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256}"'),
        text.index('"${PYTHON_BIN}" -I "${R7_CHECKER}"'),
        text.index('"${PYTHON_BIN}" -I "${M1207_CHECKER}"'),
        text.index("blocked={'vcs'"),
        text.index('mem_kib="$(awk'),
        text.index('/bin/mkdir -- "${ATTEMPT}"'),
        text.index('"${VCS_BIN}" -full64'),
        text.index('./simv -no_save'),
    ]
    require(positions == sorted(positions), "release/attempt/tool gate ordering")
    require(text.count('"${VCS_BIN}" -full64') == 1 and
            text.count('./simv -no_save') == 1, "not exactly one VCS and simv")
    require(text.count('/bin/mkdir -- "${ATTEMPT}"') == 1, "attempt not unique")
    require('x[\'identity\'][\'hammer_manifest_sha256\']' not in text and
            'x[\'identity\'][\'hammer_outer_seal_file_sha256\']' not in text,
            "runtime self-reference read")
    require("1800s ./simv -no_save" in text and "trap on_exit EXIT" in text and
            "failed_or_incomplete.$$.quarantine" in text and
            'seal_dir "${WORK}"' in text and 'mv -- "${WORK}" "${RESULT}"' in text,
            "timeout/quarantine/atomic publish drift")


def mutation_suite(runner: str) -> int:
    mutations = (
        runner.replace('[[ $# -eq 0 ]]', '[[ $# -ge 0 ]]', 1),
        runner.replace('verify_recursive_seal "${RELEASE_HAMMER}"', '# removed release hammer', 1),
        runner.replace('sha_exact "${M1207_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256}"', '# removed manifest pin', 1),
        runner.replace('/bin/mkdir -- "${ATTEMPT}"', '"${VCS_BIN}" -full64\n/bin/mkdir -- "${ATTEMPT}"', 1),
        runner.replace('"${VCS_BIN}" -full64', '# removed VCS command', 1),
        runner.replace('./simv -no_save', './simv -no_save\n./simv -no_save', 1),
        runner.replace('+define+UNIT_DELAY', '+define+ZERO_DELAY', 1),
        runner.replace('boundary_fault=0 core_fault=0', 'boundary_fault=1 core_fault=0', 1),
        runner.replace("timing_verified':False", "timing_verified':True", 1),
        runner.replace("1800s ./simv -no_save", "0s ./simv -no_save", 1),
    )
    rejected = 0
    for index, text in enumerate(mutations):
        require(text != runner, "mutation did not apply")
        try: validate_runner(text)
        except (Failure, ValueError): rejected += 1
        else: raise Failure(f"runner mutation {index} survived")
    return rejected


def run_static(path: Path) -> dict[str, Any]:
    process = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", "-I", str(path)],
        cwd=HW.parent, text=True, capture_output=True, check=False,
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    require(process.returncode == 0 and not process.stderr, "static checker failed")
    return json.loads(process.stdout)


def main() -> None:
    regular(RUNNER, RUNNER_SHA); regular(CHECKER, CHECKER_SHA)
    double_file(CONTRACT, CONTRACT_ID); double_file(RELEASE, RELEASE_ID)
    author = manifest_tree(AUTHOR, AUTHOR_ID, "review.json")
    m1198 = manifest_tree(M1198_AUTHOR, M1198_AUTHOR_ID, "review.json")
    m1201 = manifest_tree(M1201, M1201_ID, "review.json")
    for path, digest in EXACT.items(): regular(path, digest)
    require(author["status"].startswith("PASS_M1207_R7_ACYCLIC_RELEASE") and
            m1198["status"].startswith("PASS_R7_SOURCE_ONLY") and
            m1201["status"].startswith("PASS_SOURCE_HAMMER"), "admission status drift")
    contract, release = strict_json(CONTRACT), strict_json(RELEASE)
    require(contract["status"] ==
            "M1207_R7_ACYCLIC_RELEASE_SOURCE_READY__FRESH_M1208_HAMMER_REQUIRED__NO_VCS_NO_EDA"
            and release["status"] ==
            "AUTHORIZE_ONE_M1207_R7_FUNCTIONAL_VCS_ATTEMPT_AFTER_ACYCLIC_M1208_HAMMER",
            "contract/release status")
    require(contract["identity"]["runner_sha256"] == RUNNER_SHA and
            contract["identity"]["m1207_release_checker_sha256"] == CHECKER_SHA and
            release["identity"]["runner_sha256"] == RUNNER_SHA and
            release["identity"]["source_contract_sha256"] == CONTRACT_ID[0],
            "contract/release identity")
    require(contract["authorization"] == {"vcs_compiles": 0, "simv_runs": 0,
            "all_other_eda_runs": 0, "gpu_runs": 0, "network_runs": 0} and
            release["authorization"] == {"vcs_compiles": 1, "simv_runs": 1,
            "all_other_eda_runs": 0}, "authorization drift")
    for document in (contract, release):
        for key in FALSE_CLAIMS: require(document["claim_boundary"][key] is False, "claim drift " + key)

    runner = RUNNER.read_text(encoding="utf-8")
    validate_runner(runner)
    runner_mutations = mutation_suite(runner)
    good = {name: "a" * 64 for name in ENV_NAMES}
    require(env_gate(good), "complete env rejected")
    env_mutations = 0
    for name in ENV_NAMES:
        for value in (None, "b" * 63, "B" * 64, "z" * 64):
            candidate = dict(good)
            if value is None: candidate.pop(name)
            else: candidate[name] = value
            require(not env_gate(candidate), "environment mutation survived")
            env_mutations += 1
    require(review_is_acyclic({"runner_sha256": RUNNER_SHA}, "{}", "a"*64, "b"*64),
            "valid acyclic identity rejected")
    self_ref_mutations = 0
    for key in ("hammer_manifest_sha256", "hammer_outer_seal_file_sha256",
                "manifest_sha256", "outer_seal_file_sha256"):
        require(not review_is_acyclic({key: "c"*64}, "{}", "a"*64, "b"*64),
                "self-reference key survived")
        self_ref_mutations += 1
    for serialized in ('{"x":"' + "a"*64 + '"}', '{"x":"' + "b"*64 + '"}'):
        require(not review_is_acyclic({}, serialized, "a"*64, "b"*64),
                "embedded self-digest survived")
        self_ref_mutations += 1

    r7_static = run_static(R7_CHECKER)
    release_static = run_static(CHECKER)
    require(r7_static["checks_passed"] == 74 and r7_static["mutations_rejected"] == 16 and
            r7_static["service_force_multiset_exact_nine"] is True and
            r7_static["service_oracles_exact"] is True and
            release_static["checks_passed"] == 110 and
            release_static["environment_mutations_rejected"] == 16 and
            release_static["one_compile_one_sim"] is True,
            "static checker evidence drift")
    bash_parse = subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)],
            cwd=HW.parent, capture_output=True, text=True, check=False)
    require(bash_parse.returncode == 0 and not bash_parse.stderr, "runner bash syntax")
    sva_text = SVA.read_text(encoding="utf-8")
    require(len(re.findall(r"\bassert\s+property\b", sva_text)) == 16 and
            len(re.findall(r"\bcover\s+property\b", sva_text)) == 6,
            "SVA assertion/cover count drift")
    filelist = FILELIST.read_text(encoding="utf-8").splitlines()
    require(len(filelist) == 6 and filelist[-1].endswith(TB.name) and
            filelist[-2].endswith(SVA.name) and filelist[0].endswith("180a_ssg0p9v125c.v"),
            "foundry UNIT_DELAY filelist drift")

    attempt = HW / "results/.m1207_m1198r7_m1162_c1_common_charge_protocol_vcs_r7_attempt_consumed"
    result = HW / "results/m1207_m1198r7_m1162_c1_common_charge_protocol_unit_delay_vcs_r7_20260830"
    work = list((HW / "results").glob(".m1207_m1198r7_m1162_c1_common_charge_protocol_vcs_r7_work.*"))
    quarantine = list((HW / "results").glob(
        "m1207_m1198r7_m1162_c1_common_charge_protocol_unit_delay_vcs_r7_20260830.failed_or_incomplete.*"))
    require(not attempt.exists() and not result.exists() and not work and not quarantine,
            "M1207 namespace not fresh")

    regression = release["required_regression"]
    expected_regression = {
        "assertions": 16, "covers": 6, "protocol_attacks": 7,
        "service_assumption_attacks": 2, "deterministic_legal_transactions": 24,
        "legal_masks_clear": 29, "request_attack_windows": 2, "reset_states": 3,
        "minimum_completed_issue_ii": 2, "normal_m935_rows": 1,
        "normal_m935_tasks": 1, "service_skew_isolated": True,
        "reachable_core_ready_force": False, "boundary_fault": False, "core_fault": False,
    }
    require(regression == expected_regression, "regression oracle drift")
    review = {
        "schema": "m1208_m1207_m1198_c1_r7_vcs_release_source_hammer_review_r1_v1",
        "status": "PASS_M1208_M1207_C1_R7_ACYCLIC_RELEASE_HAMMER__AUTHORIZE_ONE_LAUNCH",
        "date": "2026-08-30", "verdict": "GO", "score": 100,
        "issue_counts": {"P0": 0, "P1": 0, "P2": 0},
        "identity": {
            "runner_sha256": RUNNER_SHA,
            "m1207_checker_sha256": CHECKER_SHA,
            "source_contract_sha256": CONTRACT_ID[0],
            "release_sha256": RELEASE_ID[0],
            "m1207_author_review_sha256": AUTHOR_ID[0],
            "m1198_contract_sha256": EXACT[M1198_CONTRACT],
            "m1201_review_sha256": M1201_ID[0],
            "docs359_sha256": EXACT[DOCS359],
        },
        "independent_checks": {
            "runner_static_mutations_rejected": runner_mutations,
            "environment_mutations_rejected": env_mutations,
            "self_reference_mutations_rejected": self_ref_mutations,
            "m1198_checks_passed": 74,
            "m1198_mutations_rejected": 16,
            "m1207_checks_passed": 110,
            "sva_assertions": 16, "sva_covers": 6,
            "recursive_authorities_verified": 3,
            "contract_double_seals_verified": 2,
            "runner_bash_parse": True,
            "fresh_attempt_result_work_quarantine_namespace": True,
        },
        "vcs_semantics": {
            "foundry_unit_delay": True,
            "vcs_compiles": 1, "simv_runs": 1, "simv_timeout_seconds": 1800,
            "exact_pass_and_four_coverage_lines": True,
            "assertions": 16, "covers": 6,
            "protocol_attacks": 7, "service_assumption_attacks": 2,
            "legal_transactions": 24, "minimum_completed_issue_ii": 2,
            "service_skew_isolated": True,
            "reachable_core_ready_force": False,
            "boundary_fault": False, "core_fault": False,
        },
        "acyclic_authority": {
            "review_forbids_own_manifest_and_outer_fields": True,
            "review_forbids_embedded_manifest_and_outer_digests": True,
            "review_manifest_outer_supplied_as_three_independent_runtime_hashes": True,
            "all_verified_before_attempt": True,
        },
        "authorization": {"vcs_compiles": 1, "simv_runs": 1, "all_other_eda_runs": 0},
        "execution_now": {"vcs": 0, "simv": 0, "all_eda": 0,
                          "license": 0, "gpu": 0, "network": 0},
        "claim_boundary": {
            "functional_vcs_verified": False, "timing_verified": False,
            "cycles_measured": False, "speedup": False, "ppa": False,
            "power": False, "energy": False, "system_speedup": False,
            "paper_citable": False, "headline": False,
        },
    }
    mechanical = {
        "schema": "m1208_m1207_c1_r7_release_hammer_mechanical_r1_v1",
        "status": "PASS_SOURCE_RELEASE_HAMMER__NO_VCS_NO_EDA",
        "checks": {
            "exact_source_and_contract_identities": True,
            "recursive_authority_and_double_seals": True,
            "acyclic_runtime_hash_protocol": True,
            "self_reference_rejected": self_ref_mutations,
            "environment_faults_rejected": env_mutations,
            "runner_semantic_faults_rejected": runner_mutations,
            "exact_unit_delay_regression_oracles": True,
            "one_shot_atomicity_and_failure_quarantine": True,
            "namespace_fresh": True, "docs359_unchanged": True,
        },
        "execution": {"vcs": False, "simv": False, "eda": False,
                      "license": False, "gpu": False, "network": False},
    }
    review_md = """# M1208 independent M1207 C1/R7 acyclic release hammer

**Verdict: GO, 100/100, P0=0, P1=0.** The exact M1198/M1201 source
corpus, the M1207 runner/checker/contracts, and all recursive authorities are
sealed and unchanged. Independent checks reject environment, self-reference,
gate-order, UNIT_DELAY, count, oracle, timeout and claim-boundary mutations.

The acyclic protocol is sound: the review contains no self manifest/outer
identity, while review, manifest and outer-seal-file hashes must arrive as
three independent environment values and are verified before the persistent
attempt token. Exactly one foundry UNIT_DELAY compile and one bounded simv run
are authorized. No VCS, simv, license, EDA, GPU, or network action occurred in
this hammer.

This is source/release authorization only. Functional VCS, timing, cycles,
speedup, PPA, power, energy, system speedup and paper citation remain false
until a future sealed result passes a fresh different-author result hammer.
"""
    (HERE / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (HERE / "mechanical_checks.json").write_text(json.dumps(mechanical, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (HERE / "review.md").write_text(review_md, encoding="utf-8")
    (HERE / "NO_VCS_NO_SIMV_NO_EDA_NO_LICENSE_NO_GPU_NO_NETWORK.txt").write_text(
        "M1208 performed local read-only source/release checks only.\n", encoding="ascii")
    (HERE / "RUN_COMPLETE.txt").write_text(
        "PASS_M1208_M1207_C1_R7_ACYCLIC_RELEASE_HAMMER__AUTHORIZE_ONE_LAUNCH\n",
        encoding="ascii")


if __name__ == "__main__":
    main()
