#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Exact-contract source gate for the fresh M1363 C1/R16 one-shot runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any


HERE = Path(__file__).resolve().parent
HW = HERE.parent
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1363_m1337r15_m1162_c1_real_m935_runtime_witness_unit_delay_exact_sha.sh"
RUNNER_SHA256 = "ac473072accc6d48ec15c1e541d3fd7caad64638a2942655766dad14a1879de3"
FILELIST = HW / "verif_m1337r15_c1_real_m935_runtime_witness/m1337r15_unit_delay_filelist.f"
WITNESS = HW / "verif_m1337r15_c1_real_m935_runtime_witness/m1337r15_m935_runtime_witness.sv"
TB = HW / "verif_m1270r13_c1_real_m935_protocol/tb_m1270r13_m1162_real_m935_protocol_unit_delay_r13.sv"
PARENT = HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
WRAPPER = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
SVA = HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
R16_CHECKER = HW / "verif_m1345r16_c1_real_m935_runtime_witness/check_m1345r16_source.py"
R16_TESTS = HW / "verif_m1345r16_c1_real_m935_runtime_witness/test_m1345r16_source.py"
R16_CONTRACT = HW / "contracts/m1345_c1_r16_real_m935_runtime_witness_source_contract_r1_20260831.json"
R16_AUTHOR = HW / "reviews/m1345_c1_r16_real_m935_runtime_witness_source_author_r1_20260831"
R16_HAMMER = HW / "reviews/m1352_m1345_c1_r16_runtime_witness_source_blind_hammer_r1_20260831"
M1354_CHECKER = HW / "verif_m1354_c1_r16_vcs_release/check_m1354_c1_r16_vcs_release_source.py"
M1354_TESTS = HW / "verif_m1354_c1_r16_vcs_release/test_m1354_c1_r16_vcs_release_source.py"
M1354_CONTRACT = HW / "contracts/m1354_c1_r16_real_m935_runtime_witness_vcs_release_source_contract_r1_20260831.json"
M1354_AUTHOR = HW / "reviews/m1354_c1_r16_real_m935_runtime_witness_vcs_release_source_author_r1_20260831"
M1355_FAIL = HW / "reviews/m1355_m1354_c1_r16_real_m935_runtime_witness_vcs_release_blind_hammer_r1_20260831"
CHECKER = Path(__file__).resolve()
TESTS = HERE / "test_m1363_c1_r16_vcs_release_exact_source.py"
CONTRACT = HW / "contracts/m1363_c1_r16_real_m935_runtime_witness_vcs_release_exact_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1363_c1_r16_real_m935_runtime_witness_vcs_release_exact_source_author_r1_20260831"
FUTURE_HAMMER = HW / "reviews/m1364_m1363_c1_r16_real_m935_runtime_witness_vcs_release_source_blind_hammer_r1_20260831"
FUTURE_RELEASE = HW / "contracts/m1365_m1364_m1363_c1_r16_real_m935_runtime_witness_vcs_launch_release_r1_20260831.json"
FUTURE_FINAL = HW / "reviews/m1366_m1365_m1363_c1_r16_real_m935_runtime_witness_vcs_final_launch_hammer_r1_20260831"
FOUNDRY = Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v")
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
ATTEMPT = HW / "results/.m1363_c1_r16_real_m935_runtime_witness_vcs_attempt_consumed"
RESULT = HW / "results/m1363_c1_r16_real_m935_runtime_witness_unit_delay_vcs_r1_20260831"
QUARANTINE = Path(str(RESULT) + ".failed_or_incomplete.quarantine")

CLAIMS = ("source_only", "functional_vcs", "timing_verified", "cycles_measured",
          "speedup", "ppa", "power", "energy", "system_speedup", "headline")
EXACT_CLAIMS = {key: key == "source_only" for key in CLAIMS}
EXPECTED = {
    RUNNER: RUNNER_SHA256,
    FILELIST: "87a8b5e7500808a8afbd4339668aae3a44db2de7924a948020e2c7bffce4289e",
    WITNESS: "0ec7179e36f9af09e3020f76a5a927298d877b3cc20c6ac9ab4686bf465d18af",
    TB: "b749c7d635dc5b65669320aec7b7edb40cd5e2a5d781a9e474e3d28cbb054263",
    PARENT: "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    M935: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    WRAPPER: "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    SVA: "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    R16_CHECKER: "b570eeb7a49bb042de2abca2f6739df09ab1895f208103dbe4dfdac2e340cea4",
    R16_TESTS: "5427063ef93e89cd7059b6e48422626a71fd0913427f9614da65faf9fca29929",
    R16_CONTRACT: "c9749b4a7f9e3e6f8b38cbaf4735b036d7753f79a407e208d28f09aecd375f33",
    M1354_CHECKER: "1eafcbe14319fcb47e53dcdd9369393f602201056498eec2b652a6263a378c4d",
    M1354_TESTS: "6739b5234e1a5fd8d81af4089d8273e568b4aade9a1502ffac870958debb029f",
    M1354_CONTRACT: "39ff4b3675a70266cf8d2e695078856880bb240a767c2786938e428f94409559",
    FOUNDRY: "8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d",
    VCS: "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    PYTHON: "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
SEALS = {
    R16_AUTHOR: ("a5b136fce2bc3c5b5a5920b1e88cff092b1228b49a7ff6fd9959ff95e06772e5",
                 "bd875634a0be33cb5dc2f0600734fa90e014ade961658c3d1f480ce40425a616",
                 "c9700d4411dd087b12494e4aaf2f5fde0de52f7e30b7397573b205371837e99f"),
    R16_HAMMER: ("74969404ea26e5a522c205328c05a3527fca6daeefb74f6fb103cacb990e94ea",
                 "d703fb23ff2a7726049f58d09e7d304d0e4e8adcaa781f34856115dcb4de40e6",
                 "29c6bf6de6a7ed91dc523dfc3360d7731c324a24cd3548a0fe3a346018e37ec7"),
    M1354_AUTHOR: ("378ce7f6e8b0ae20f98c94d197c2fad1dcd7e1082fa269320041480319daddae",
                   "799616b204bb88333193baad0188aac846cdca9a0493c19476f31ca1f7f866f2",
                   "862b93fa2e781f48e4c1a59cc63262fe6541787e32171f28109ce6fd3eb0cbb6"),
    M1355_FAIL: ("7c06c50e2087e2794957508cf042d6931d73cb22ce3a3cada5628a2d55ae4c8d",
                 "9709d1c21ce13df3b84efa19d4dfa47d2116fa661327f18d0666b17d924ec5f8",
                 "8b7aea4d1bc0764c1e9137196e2fc0ea3b86cee27baf6ab459c2c717bd201105"),
}


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regular(path: Path) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), "not regular: " + str(path))


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    regular(path)
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           AssertionError("nonfinite JSON token: " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def verify_dir(root: Path, pins: tuple[str, str, str] | None = None) -> dict[str, Any]:
    require(root.is_dir() and not root.is_symlink(), "sealed directory invalid")
    manifest, outer = root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"
    regular(manifest); regular(outer)
    if pins is not None:
        require(sha(root / "review.json") == pins[0] and sha(manifest) == pins[1] and sha(outer) == pins[2],
                "sealed directory exact pin drift")
    require(outer.read_text().split() == [sha(manifest), "SHA256SUMS"], "outer seal drift")
    listed = set()
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*"); rel = Path(name)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and name not in listed and
                not rel.is_absolute() and ".." not in rel.parts, "manifest row invalid")
        member = root / rel; regular(member); require(sha(member) == digest, "manifest member drift")
        listed.add(name)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == listed, "sealed directory population drift")
    return strict_json(root / "review.json")


def expected_contract() -> dict[str, Any]:
    return {
        "schema": "m1363_c1_r16_real_m935_runtime_witness_vcs_release_exact_source_contract_r1_v1",
        "status": "M1363_C1_R16_VCS_RELEASE_EXACT_SOURCE_READY__FRESH_M1364_REQUIRED__NO_LAUNCH",
        "date": "2026-08-31",
        "purpose": "Fresh-path additive successor to M1354 that exact-checks all M1355 contract false-negative fields while preserving the admitted R16 technical corpus.",
        "identity": {
            "runner_path": RUNNER.relative_to(HW).as_posix(), "runner_sha256": RUNNER_SHA256,
            "filelist_path": FILELIST.relative_to(HW).as_posix(), "filelist_sha256": EXPECTED[FILELIST],
            "source_checker_path": CHECKER.relative_to(HW).as_posix(), "source_checker_sha256": sha(CHECKER),
            "source_tests_path": TESTS.relative_to(HW).as_posix(), "source_tests_sha256": sha(TESTS),
        },
        "failed_predecessor": {
            "m1354_checker_sha256": EXPECTED[M1354_CHECKER],
            "m1354_tests_sha256": EXPECTED[M1354_TESTS],
            "m1354_contract_sha256": EXPECTED[M1354_CONTRACT],
            "m1354_author_review_sha256": SEALS[M1354_AUTHOR][0],
            "m1354_author_manifest_sha256": SEALS[M1354_AUTHOR][1],
            "m1354_author_outer_file_sha256": SEALS[M1354_AUTHOR][2],
            "m1355_review_sha256": SEALS[M1355_FAIL][0],
            "m1355_manifest_sha256": SEALS[M1355_FAIL][1],
            "m1355_outer_file_sha256": SEALS[M1355_FAIL][2],
            "m1355_false_negatives": 16,
            "m1354_runner_reused": False,
        },
        "r16_authority": {
            "checker_sha256": EXPECTED[R16_CHECKER], "tests_sha256": EXPECTED[R16_TESTS],
            "contract_sha256": EXPECTED[R16_CONTRACT],
            "author_review_sha256": SEALS[R16_AUTHOR][0], "author_manifest_sha256": SEALS[R16_AUTHOR][1],
            "author_outer_file_sha256": SEALS[R16_AUTHOR][2],
            "blind_review_sha256": SEALS[R16_HAMMER][0], "blind_manifest_sha256": SEALS[R16_HAMMER][1],
            "blind_outer_file_sha256": SEALS[R16_HAMMER][2], "source_admitted": True,
        },
        "future_release": {
            "source_hammer_path": FUTURE_HAMMER.relative_to(HW).as_posix(),
            "launch_release_path": FUTURE_RELEASE.relative_to(HW).as_posix(),
            "final_hammer_path": FUTURE_FINAL.relative_to(HW).as_posix(),
            "fresh_different_author_hammers_required": True,
            "launch_authorized": False, "vcs_compiles_now": 0, "simv_runs_now": 0,
            "automatic_retry": False,
        },
        "future_execution": {
            "maximum_vcs_compiles": 1, "maximum_simv_runs": 1, "all_other_eda_runs": 0,
            "compile_timeout_seconds": 1200, "simulation_timeout_seconds": 1800,
            "attempt_consumed_before_tool": True,
            "fresh_attempt_namespace": ATTEMPT.relative_to(HW).as_posix(),
            "fresh_result_namespace": RESULT.relative_to(HW).as_posix(),
            "failure_quarantine_recursive_seal": True, "automatic_retry": False,
        },
        "author_execution": {
            "source_authoring": True, "source_only_tests": True,
            "different_author_blind_hammer": True, "release": False,
            "license_query": False, "vcs": False, "simv": False, "dc": False,
            "pt": False, "ptpx": False, "eda": False, "gpu": False, "remote": False,
        },
        "claim_boundary": dict(EXACT_CLAIMS),
        "protected_files": {
            "docs359": {"path": DOCS359.relative_to(HW).as_posix(), "sha256": EXPECTED[DOCS359]},
            "foundry_unit_delay": {"path": str(FOUNDRY), "sha256": EXPECTED[FOUNDRY]},
        },
    }


def check_contract_dict(contract: dict[str, Any]) -> None:
    require(contract == expected_contract(), "M1363 contract exact-set/value drift")


def env_gate(env: dict[str, str]) -> bool:
    names = (
        "M1363_EXPECTED_RUNNER_SHA256", "M1363_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256",
        "M1363_EXPECTED_SOURCE_HAMMER_MANIFEST_SHA256", "M1363_EXPECTED_SOURCE_HAMMER_OUTER_FILE_SHA256",
        "M1363_EXPECTED_LAUNCH_RELEASE_SHA256", "M1363_EXPECTED_FINAL_HAMMER_REVIEW_SHA256",
        "M1363_EXPECTED_FINAL_HAMMER_MANIFEST_SHA256", "M1363_EXPECTED_FINAL_HAMMER_OUTER_FILE_SHA256")
    return all(re.fullmatch(r"[0-9a-f]{64}", env.get(name, "")) is not None for name in names)


def audit_runner(text: str) -> dict[str, Any]:
    require(text.count('"${VCS_BIN}" -full64') == 1, "compile cardinality drift")
    require(text.count('./simv -no_save') == 1, "sim cardinality drift")
    require(text.count('/usr/bin/timeout --signal=TERM --kill-after=30s') == 2, "timeout cardinality drift")
    attempt = text.index('phase="ATTEMPT_CONSUME"')
    compile_at = text.index('phase="COMPILE"')
    require(attempt < compile_at and text.index('publish_no_replace "${ATTEMPT_STAGE}" "${ATTEMPT}"') < compile_at,
            "attempt does not dominate tool")
    resource = text.index('phase="RESOURCE_PREFLIGHT"')
    require(text.count("collision_gate\n", resource, attempt) == 2 and resource < attempt,
            "collision gates do not dominate attempt")
    require('seal_dir "${FAILURE_STAGE}"' in text and
            'publish_no_replace "${FAILURE_STAGE}" "${QUARANTINE}"' in text,
            "failure quarantine seal/publish drift")
    require("automatic_retry=true" not in text and "rm -" not in text, "retry/destructive drift")
    return {"one_compile": True, "one_sim": True, "two_timeouts": True,
            "attempt_before_tool": True, "collision_gates": 2,
            "failure_quarantine_recursive_seal": True}


def validate_future(mode: str) -> dict[str, Any]:
    paths = (FUTURE_HAMMER, FUTURE_RELEASE, FUTURE_FINAL)
    if mode == "source_absent":
        require(all(not os.path.lexists(str(path)) for path in paths), "future release residue")
        require(all(not os.path.lexists(str(path)) for path in (ATTEMPT, RESULT, QUARANTINE)),
                "one-shot result namespace residue")
        return {"mode": mode, "future_absent": True, "one_shot_fresh": True}
    require(mode == "runtime_present", "unknown mode")
    require(FUTURE_HAMMER.is_dir() and FUTURE_RELEASE.is_file() and FUTURE_FINAL.is_dir(),
            "runtime release chain absent")
    require(all(not os.path.lexists(str(path)) for path in (ATTEMPT, RESULT, QUARANTINE)),
            "one-shot result namespace residue")
    return {"mode": mode, "future_present": True, "one_shot_fresh": True}


def validate_common(skip_author: bool = False) -> dict[str, Any]:
    for path, digest in EXPECTED.items():
        regular(path); require(sha(path) == digest, "frozen exact-byte drift: " + str(path))
    expected_filelist = [str(path) for path in (FOUNDRY, PARENT, M935, WRAPPER, SVA, TB, WITNESS)]
    require(FILELIST.read_text().splitlines() == expected_filelist, "filelist/order drift")
    for root, pins in SEALS.items(): verify_dir(root, pins)
    m1355 = strict_json(M1355_FAIL / "review.json")
    require(m1355.get("status") == "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED" and
            m1355.get("fresh_hammer", {}).get("false_negative_count") == 16 and
            m1355.get("authorization", {}).get("additive_source_successor") is True and
            m1355.get("claim_boundary") == EXACT_CLAIMS, "M1355 failure authority drift")
    syntax = subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, check=False)
    require(syntax.returncode == 0, "runner shell syntax")
    runner_audit = audit_runner(RUNNER.read_text())
    check_contract_dict(strict_json(CONTRACT))
    if not skip_author:
        author = verify_dir(AUTHOR)
        require(author.get("status") == "PASS_M1363_C1_R16_VCS_RELEASE_EXACT_SOURCE__FRESH_M1364_REQUIRED" and
                author.get("bindings") == {
                    "runner_sha256": RUNNER_SHA256, "source_checker_sha256": sha(CHECKER),
                    "source_tests_sha256": sha(TESTS), "source_contract_sha256": sha(CONTRACT),
                    "m1354_author_review_sha256": SEALS[M1354_AUTHOR][0],
                    "m1355_review_sha256": SEALS[M1355_FAIL][0]} and
                author.get("authorization", {}).get("release") is False and
                author.get("claim_boundary") == EXACT_CLAIMS, "author seal drift")
    return {"exact_byte_members": len(EXPECTED), "sealed_authorities": len(SEALS),
            "m1355_false_negatives_bound": 16, "runner": runner_audit,
            "claim_boundary_exact": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source_absent", "runtime_present"), required=True)
    parser.add_argument("--skip-author", action="store_true")
    args = parser.parse_args()
    common = validate_common(skip_author=args.skip_author)
    future = validate_future(args.mode)
    print(json.dumps({
        "schema": "m1363_c1_r16_vcs_release_exact_source_check_r1_v1",
        "status": "PASS_M1363_SOURCE_ONLY__NO_VCS_NO_EDA",
        "common": common, "future": future, "launch_authorized": False,
        "license_queries": 0, "vcs_runs": 0, "simv_runs": 0, "eda_runs": 0,
        "docs359_sha256": sha(DOCS359),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
