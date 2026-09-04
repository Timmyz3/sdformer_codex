#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Two-mode source/runtime checker for the M1344 C2 one-shot release."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys


HW = Path(__file__).resolve().parents[1]
OLD_CHECKER = HW / "verif_m1336_c2_activity_release/static_check_m1336_c2_activity_vcs_release_source.py"
OLD_CHECKER_SHA256 = "6c2f67051a5ae9796f2f95a5f0dd905b5e8f9d1fe07364dbe5442a207dc24c38"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1344_c2_headline_mapped_production_activity_one_shot_exact_sha.sh"
CONTRACT = HW / "contracts/m1344_c2_headline_mapped_production_activity_vcs_release_source_contract_r1_20260831.json"
CHECKER = Path(__file__).resolve()
TEST = HW / "verif_m1344_c2_activity_release/test_m1344_c2_activity_vcs_release_source.py"
AUTHOR = HW / "reviews/m1344_c2_headline_mapped_production_activity_vcs_release_source_author_r1_20260831"
M1337_FAIL = HW / "reviews/m1337_m1336_c2_headline_mapped_production_activity_vcs_release_source_blind_hammer_r1_20260831"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
CYCLES = {"k8": [51, 131, 486, 1231, 14],
          "k1x8": [53, 133, 499, 1246, 14]}
EVENTS = [20, 41, 90, 110, 0]
ENV_NAMES = (
    "M1344_EXPECTED_RUNNER_SHA256",
    "M1344_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256",
    "M1344_EXPECTED_SOURCE_HAMMER_MANIFEST_SHA256",
    "M1344_EXPECTED_SOURCE_HAMMER_OUTER_FILE_SHA256",
    "M1344_EXPECTED_LAUNCH_RELEASE_SHA256",
    "M1344_EXPECTED_FINAL_HAMMER_REVIEW_SHA256",
    "M1344_EXPECTED_FINAL_HAMMER_MANIFEST_SHA256",
    "M1344_EXPECTED_FINAL_HAMMER_OUTER_FILE_SHA256",
)
CLAIMS = ("functional_vcs_verified", "production_saif", "ptpx", "power",
          "energy", "performance", "system_speedup", "paper_ppa_ready", "headline")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def load_old():
    need(sha(OLD_CHECKER) == OLD_CHECKER_SHA256, "sealed M1336 checker drift")
    spec = importlib.util.spec_from_file_location("m1344_sealed_m1336_checker", OLD_CHECKER)
    need(spec is not None and spec.loader is not None, "cannot load sealed checker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OLD = load_old()


def verify_file_sidecar(path: Path) -> None:
    sums = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    need(path.is_file() and not path.is_symlink() and
         sums.is_file() and not sums.is_symlink() and
         outer.is_file() and not outer.is_symlink(), "sidecar absent")
    need(sums.read_text().split() == [sha(path), path.name], "file sidecar mismatch")
    need(outer.read_text().split() == [sha(sums), sums.name], "outer sidecar mismatch")


def future_paths(root: Path = HW) -> dict[str, Path]:
    return {
        "source_hammer": root / "reviews/m1345_m1344_c2_headline_mapped_production_activity_vcs_release_source_blind_hammer_r1_20260831",
        "launch_release": root / "contracts/m1346_m1344_c2_headline_mapped_production_activity_vcs_launch_release_r1_20260831.json",
        "final_hammer": root / "reviews/m1347_m1346_m1344_c2_headline_mapped_production_activity_vcs_final_launch_hammer_r1_20260831",
    }


def env_expected(env: dict[str, str]) -> dict[str, str]:
    need(all(re.fullmatch(r"[0-9a-f]{64}", env.get(name, "")) for name in ENV_NAMES),
         "runtime external SHA environment absent/invalid")
    return {name: env[name] for name in ENV_NAMES}


def validate_future(mode: str, paths: dict[str, Path] | None = None,
                    expected: dict[str, str] | None = None) -> dict:
    paths = future_paths() if paths is None else paths
    need(set(paths) == {"source_hammer", "launch_release", "final_hammer"},
         "future path key set drift")
    if mode == "source_absent":
        need(expected is None, "source_absent must not consume future SHA")
        need(all(not os.path.lexists(str(path)) for path in paths.values()),
             "source_absent mode found future authority residue")
        return {"mode": mode, "future_absent": True}
    need(mode == "runtime_present", "unknown checker mode")
    need(expected is not None and set(expected) == set(ENV_NAMES),
         "runtime expected SHA key set drift")
    source = paths["source_hammer"]
    release = paths["launch_release"]
    final = paths["final_hammer"]
    OLD.verify_dir(source)
    verify_file_sidecar(release)
    OLD.verify_dir(final)
    exact = {
        source / "review.json": expected["M1344_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256"],
        source / "SHA256SUMS": expected["M1344_EXPECTED_SOURCE_HAMMER_MANIFEST_SHA256"],
        source / "SHA256SUMS.seal.sha256": expected["M1344_EXPECTED_SOURCE_HAMMER_OUTER_FILE_SHA256"],
        release: expected["M1344_EXPECTED_LAUNCH_RELEASE_SHA256"],
        final / "review.json": expected["M1344_EXPECTED_FINAL_HAMMER_REVIEW_SHA256"],
        final / "SHA256SUMS": expected["M1344_EXPECTED_FINAL_HAMMER_MANIFEST_SHA256"],
        final / "SHA256SUMS.seal.sha256": expected["M1344_EXPECTED_FINAL_HAMMER_OUTER_FILE_SHA256"],
    }
    for path, digest in exact.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "runtime future identity drift: " + str(path))
    need(sha(RUNNER) == expected["M1344_EXPECTED_RUNNER_SHA256"],
         "runtime runner external SHA drift")
    source_json = json.loads((source / "review.json").read_text())
    release_json = json.loads(release.read_text())
    final_json = json.loads((final / "review.json").read_text())
    runner_sha, contract_sha = sha(RUNNER), sha(CONTRACT)
    need(source_json["status"] ==
         "PASS_M1345_M1344_C2_ACTIVITY_RELEASE_SOURCE__LAUNCH_RELEASE_MAY_BE_AUTHORED",
         "source-hammer runtime status drift")
    need(source_json["bindings"]["runner_sha256"] == runner_sha and
         source_json["bindings"]["source_contract_sha256"] == contract_sha,
         "source-hammer runtime binding drift")
    need(release_json["status"] ==
         "AUTHORIZE_ONE_M1344_C2_MAPPED_PRODUCTION_ACTIVITY_VCS_ATTEMPT" and
         release_json["launch_now"] is True,
         "launch-release runtime status drift")
    need(release_json["identity"]["runner_sha256"] == runner_sha and
         release_json["identity"]["source_contract_sha256"] == contract_sha and
         release_json["identity"]["source_hammer_review_sha256"] == sha(source / "review.json"),
         "launch-release runtime identity drift")
    authorization = {"vcs_compiles": 2, "simv_runs": 10,
                     "all_other_eda_runs": 0, "automatic_retry": False}
    need(release_json["authorization"] == authorization,
         "launch-release cardinality drift")
    need(final_json["status"] ==
         "PASS_M1347_AUTHORIZE_ONE_M1344_C2_MAPPED_PRODUCTION_ACTIVITY_VCS_LAUNCH",
         "final-hammer runtime status drift")
    for key, path in (("runner_sha256", RUNNER),
                      ("source_contract_sha256", CONTRACT),
                      ("source_hammer_review_sha256", source / "review.json"),
                      ("launch_release_sha256", release)):
        need(final_json["bindings"][key] == sha(path),
             "final-hammer runtime binding drift: " + key)
    need(final_json["authorization"] == authorization,
         "final-hammer runtime cardinality drift")
    for document in (source_json, release_json, final_json):
        need(all(document["claim_boundary"][key] is False for key in CLAIMS),
             "future claim boundary lifted")
    return {"mode": mode, "future_present": True,
            "authorization": authorization}


def namespaces() -> list[Path]:
    result = HW / "results/m1344_c2_headline_mapped_production_activity_vcs_r1_20260831"
    return [HW / "results/.m1344_c2_headline_mapped_production_activity_vcs_attempt_consumed",
            result, Path(str(result) + ".failed_or_incomplete.quarantine"),
            Path(str(result) + ".private_build.unsealed_do_not_cite"),
            Path(str(result) + ".failed_private_build.unsealed_do_not_cite")]


def validate_common(skip_author: bool = False) -> int:
    checks = 0
    for path, digest in OLD.EXPECTED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "inherited identity drift: " + str(path)); checks += 1
    OLD.verify_dir(OLD.M1334_AUTHOR); OLD.verify_dir(OLD.M1335_BLIND); checks += 2
    OLD.verify_dir(M1337_FAIL); checks += 1
    need(sha(M1337_FAIL / "review.json") ==
         "84a898e2b894e6754ab9ef70464b6a3f6e857b44e076d9bc1c93cf8e53faa946" and
         sha(M1337_FAIL / "SHA256SUMS") ==
         "31ae8689016cac5482a004b355a0f640251b3ad128cba7535337520552b9a0f0" and
         sha(M1337_FAIL / "SHA256SUMS.seal.sha256") ==
         "a5fe53b7def3be354aaf7ef87e4e6d779be7a2c326a10097cd4dbcad2e45e1c8",
         "M1337 FAIL root drift"); checks += 1
    syntax = subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, check=False)
    need(syntax.returncode == 0, "runner bash syntax failed"); checks += 1
    verify_file_sidecar(CONTRACT); contract = json.loads(CONTRACT.read_text()); checks += 1
    need(contract["status"] ==
         "M1344_C2_ACTIVITY_ONE_SHOT_RELEASE_SOURCE_READY__FRESH_M1345_REQUIRED",
         "contract status drift")
    need(contract["identity"] == {
        "runner": str(RUNNER.relative_to(HW)), "runner_sha256": sha(RUNNER),
        "checker": str(CHECKER.relative_to(HW)), "checker_sha256": sha(CHECKER),
        "test": str(TEST.relative_to(HW)), "test_sha256": sha(TEST)},
        "contract source identity drift")
    need(contract["workloads"] == {"events": EVENTS, "k8_cycles": CYCLES["k8"],
                                     "k1x8_cycles": CYCLES["k1x8"]},
         "workload anchors drift")
    need(contract["future_execution"] == {
        "vcs_compiles": 2, "simv_runs": 10, "automatic_retry": False,
        "compile_timeout_seconds": 1800, "simulation_timeout_seconds": 600,
        "attempt_consumed_before_first_vcs": True}, "future cardinality drift")
    need(all(contract["claim_boundary"][key] is False for key in CLAIMS),
         "contract claim boundary lifted")
    runner = RUNNER.read_text()
    markers = (
        "for axis in k8 k1x8; do", "for case_id in 0 1 2 3 4; do",
        '[[ "${compile_count}" -eq 2 && "${sim_count}" -eq 10 ]]',
        "cycles=(51 131 486 1231 14)", "cycles=(53 133 499 1246 14)",
        "events=(20 41 90 110 0)",
        '"${RELEASE_CHECKER}" --mode runtime_present',
        'publish_no_replace "${ATTEMPT_STAGE}" "${ATTEMPT}"',
        'publish_no_replace "${RESULT_STAGE}" "${RESULT}"',
        'seal_dir "${FAILURE_STAGE}"', 'seal_dir "${RESULT_STAGE}"',
        "automatic_retry=false", "CANDIDATE_UNSEALED_DO_NOT_CITE")
    need(all(marker in runner for marker in markers), "runner semantic marker absent")
    need(runner.index('phase="RESOURCE_PREFLIGHT"') < runner.index('phase="ATTEMPT_CONSUME"') and
         runner.index('phase="LICENSE_PREFLIGHT"') < runner.index('phase="ATTEMPT_CONSUME"'),
         "preflight/attempt order drift")
    need("ucli.key" not in runner and re.search(r"(^|\s)rm([\s-]|$)", runner) is None,
         "workspace UCLI/removal path introduced")
    for key in ("source_hammer_review_sha256", "source_hammer_manifest_sha256",
                "source_hammer_outer_file_sha256", "final_hammer_review_sha256",
                "final_hammer_manifest_sha256", "final_hammer_outer_file_sha256"):
        need(runner.count(key) >= 3, "full chain identity absent from receipts: " + key)
    checks += len(markers) + 9
    need(all(not os.path.lexists(str(path)) for path in namespaces()),
         "M1344 result namespace residue")
    checks += len(namespaces())
    if not skip_author:
        OLD.verify_dir(AUTHOR)
        review = json.loads((AUTHOR / "review.json").read_text())
        need(review["status"] ==
             "PASS_M1344_C2_ACTIVITY_RELEASE_SOURCE__FRESH_M1345_HAMMER_REQUIRED" and
             review["bindings"]["runner_sha256"] == sha(RUNNER) and
             review["bindings"]["source_contract_sha256"] == sha(CONTRACT),
             "author binding drift")
        checks += 1
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source_absent", "runtime_present"), required=True)
    parser.add_argument("--skip-author", action="store_true")
    args = parser.parse_args()
    checks = validate_common(skip_author=args.skip_author)
    expected = None if args.mode == "source_absent" else env_expected(dict(os.environ))
    future = validate_future(args.mode, expected=expected)
    print(json.dumps({
        "schema": "m1344_c2_activity_vcs_release_two_mode_check_r1",
        "status": ("PASS_M1344_SOURCE_ABSENT__NO_EDA" if args.mode == "source_absent" else
                   "PASS_M1344_RUNTIME_PRESENT__ONE_SHOT_CHAIN_ADMITTED"),
        "mode": args.mode, "checks_passed": checks, "future": future,
        "vcs_runs": 0, "simv_runs": 0, "license_queries": 0,
        "docs359_sha256": sha(DOCS359),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
