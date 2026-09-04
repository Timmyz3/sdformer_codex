#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Read-only static gate for the M1250 C1/R11 one-shot VCS release."""
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


HW = Path(__file__).resolve().parents[1]
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1250_m1232r11_m1162_c1_common_charge_protocol_exact_sha_r11.sh"
FILELIST = HW / "dc_handoff/filelists/date_m1250_m1232r11_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
TB = HW / "verif_m1232r11_c1_common_charge_protocol/tb_m1232r11_m1162_common_charge_protocol_unit_delay_r11.sv"
R11_CHECKER = HW / "verif_m1232r11_c1_common_charge_protocol/check_m1232r11_source.py"
R11_TESTS = HW / "verif_m1232r11_c1_common_charge_protocol/test_m1232r11_source.py"
PARENT = HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
WRAPPER = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
SVA = HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
M1246_CONTRACT = HW / "contracts/m1246_m1242_m1239_c1_r11_checker_source_contract_r1_20260830.json"
M1246_AUTHOR = HW / "reviews/m1246_m1242_c1_r11_checker_hardening_source_author_r1_20260830"
M1247 = HW / "reviews/m1247_m1246_c1_r11_checker_tests_independent_hammer_r1_20260830"
SOURCE_CONTRACT = HW / "contracts/m1250_m1247_m1246_c1_r11_vcs_release_source_contract_r1_20260830.json"
RELEASE = HW / "contracts/m1250_m1247_m1246_c1_r11_vcs_launch_release_r1_20260830.json"
AUTHOR = HW / "reviews/m1250_m1247_m1246_c1_r11_vcs_release_author_r1_20260830"
FOUNDRY = Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v")
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    RUNNER: "39edc512856693516f6bcf145ca184eb5254aa44bc7231d0d063e811f1b4393e",
    FILELIST: "87cc365423baa9cc2b99f9e2eac3f5a836fc8007f7136df2b671600437cab08e",
    TB: "850881df0212a9461e47e36b6829a993b9cf25af2c9faa3b7921e08fa141c776",
    R11_CHECKER: "154860a16dfa3e2175653e81c14db645da3718af2c8d659c35299d80248e68fd",
    R11_TESTS: "de89c87210e8782d38b84b8202d229a418ebb153583a02043f4080e25aac4605",
    PARENT: "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    M935: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    WRAPPER: "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    SVA: "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    M1246_CONTRACT: "7f956c2343a596da25dc8658a79e6f50da462370fa3ddd4c7b4a650ab8c6c88d",
    M1246_AUTHOR / "mechanical_checks.json": "d6cbcb89e1712c9fd1690a9f9ff5ba038275e75c882f7b2620c95837d4d6c8eb",
    M1246_AUTHOR / "SHA256SUMS": "a67fdce5307e7a35e84e162aa748404b1c56cccfc902458e1679b09b1acc1c52",
    M1246_AUTHOR / "SHA256SUMS.seal.sha256": "ff6d811aa64078feea2ad01fb30b19f10edf550d8da550ddb850939e5473f144",
    M1247 / "review.json": "32bdfcdafe3039eb9e44f318c2133e997cb182227fc0c18367d3ba9393bc807b",
    M1247 / "SHA256SUMS": "8440f0f6111f6df9df1cfe0f85847fc2743ce2b9cd6f857c33d2581fa6ec0132",
    M1247 / "SHA256SUMS.seal.sha256": "b9eb60767d829ecb0bde4e95bacc73c533c4da764702541803c69a4fa062c57d",
    FOUNDRY: "8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d",
    VCS: "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    PYTHON: "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def sidecar(path: Path) -> None:
    sums = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(sums.read_text().split() == [sha(path), path.name], "sidecar " + path.name)
    require(outer.read_text().split() == [sha(sums), sums.name], "outer " + path.name)


def verify_dir(root: Path) -> None:
    require(root.is_dir() and not root.is_symlink(), "sealed dir " + str(root))
    sums, outer = root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"
    require(outer.read_text().split() == [sha(sums), "SHA256SUMS"], "outer seal")
    listed: dict[str, str] = {}
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*")
        require(name not in listed and not Path(name).is_absolute()
                and ".." not in Path(name).parts, "unsafe member")
        listed[name] = digest
    actual: set[str] = set()
    for base_text, dirs, files in os.walk(root, followlinks=False):
        base = Path(base_text)
        dirs[:] = [name for name in dirs if not (base / name).is_symlink()]
        for name in files:
            path = base / name; rel = path.relative_to(root).as_posix()
            if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"} or path.is_symlink():
                continue
            if stat.S_ISREG(path.lstat().st_mode): actual.add(rel)
    require(actual == set(listed), "sealed population")
    for name, digest in listed.items():
        require(sha(root / name) == digest, "member drift " + name)


def audit_filelist(text: str) -> list[str]:
    errors = []
    expected = [str(path) for path in (FOUNDRY, PARENT, M935, WRAPPER, SVA, TB)]
    lines = text.splitlines()
    if lines != expected: errors.append("filelist exact ordered population")
    if any("m1219r9" in line or "m1226r10" in line for line in lines):
        errors.append("older TB leaked")
    return errors


def audit_runner(text: str) -> list[str]:
    errors = []
    def need(token: str, label: str) -> None:
        if token not in text: errors.append(label)
    for token in (
            "M1250_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256",
            "M1250_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256",
            "M1250_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256",
            "COMPILE_TIMEOUT_SECONDS=1200", "SIM_TIMEOUT_SECONDS=1800",
            "failure_phase_and_timeout_dump.txt", "TIMEOUT_M1219R9",
            "PHASE_M1219R9_RANDOM_TRANSACTION_ENTER",
            "PHASE_M1219R9_RANDOM_TRANSACTION_COMPLETE",
            "RUN_FAILED_OR_INCOMPLETE.txt", "failed_or_incomplete.$$.quarantine",
            "automatic_retry=false", "PASS_M1232R11_M1162_COMMON_CHARGE_PROTOCOL_SOURCE_CANDIDATE",
            "zero_sva_failures_required=true", "normal_m935_rows=1",
            "normal_m935_tasks=1", "compile.log sim.log"):
        need(token, "missing runner token " + token)
    if text.count('"${VCS_BIN}" -full64') != 1: errors.append("compile count")
    if text.count('./simv -no_save') != 1: errors.append("sim count")
    if text.count('/usr/bin/timeout --signal=TERM --kill-after=30s') != 2:
        errors.append("separate timeout count")
    if "rm -" in text or "automatic_retry=true" in text:
        errors.append("destructive/retry behavior")
    try:
        prior = text.index('for sealed in "${M1246_AUTHOR}"')
        hammer = text.index('sha_exact "${M1250_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256}"')
        attempt = text.index('/bin/mkdir -- "${ATTEMPT}"')
        work = text.index('/bin/mkdir -- "${WORK}"')
        compile_at = text.index('"${VCS_BIN}" -full64')
        sim_at = text.index('./simv -no_save')
        gates_at = text.index('for phase in DIRECTED')
        move_at = text.index('mv -- "${WORK}" "${RESULT}"')
        if not (prior < hammer < attempt < work < compile_at < sim_at < gates_at < move_at):
            errors.append("gate order")
    except ValueError:
        errors.append("gate anchors")
    phase_loop = re.search(r"for phase in ([A-Z0-9_ ]+); do", text)
    expected_phases = ["DIRECTED", "RESET_PENDING", "STICKY_ATTACKS",
                       "SERVICE_ATTACKS", "RANDOM", "NORMAL_M935", "CLEAN_RESET_PREP"]
    if phase_loop is None or phase_loop.group(1).split() != expected_phases:
        errors.append("phase population")
    need('^PHASE_M1219R9_${phase}_ENTER( |$)', "phase enter gate")
    need('^PHASE_M1219R9_${phase}_COMPLETE( |$)', "phase complete gate")
    need("for index in $(seq 0 23)", "24 random gate")
    need("if rg -qi '(^|[^[:alnum:]_])(Error|Fatal|Assertion|\\$error|\\$fatal)([^[:alnum:]_]|$)' compile.log sim.log; then exit 35; fi",
         "error/assertion/fatal rejection")
    return errors


def env_gate(env: dict[str, str]) -> bool:
    names = ("M1250_EXPECTED_RELEASE_SHA256",
             "M1250_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256",
             "M1250_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256",
             "M1250_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256")
    return all(re.fullmatch(r"[0-9a-f]{64}", env.get(name, "")) is not None
               for name in names)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--skip-author", action="store_true")
    args = parser.parse_args(); checks = 0
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "identity drift " + str(path)); checks += 1
    for root in (M1246_AUTHOR, M1247): verify_dir(root); checks += 1
    for artifact in (M1246_CONTRACT, SOURCE_CONTRACT, RELEASE): sidecar(artifact); checks += 2
    require(not audit_filelist(FILELIST.read_text()), "filelist audit"); checks += 1
    require(not audit_runner(RUNNER.read_text()), "runner audit"); checks += 1
    bash_check = subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, check=False)
    require(bash_check.returncode == 0, "bash syntax"); checks += 1
    completed = subprocess.run([str(PYTHON), "-I", str(R11_CHECKER)], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    require(completed.returncode == 0, "M1246 checker failed"); checks += 1
    tests = subprocess.run([str(PYTHON), "-I", str(R11_TESTS)],
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           check=False, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    require(tests.returncode == 0 and "Ran 24 tests" in tests.stderr,
            "M1246 tests failed"); checks += 1
    contract = json.loads(SOURCE_CONTRACT.read_text()); release = json.loads(RELEASE.read_text())
    require(contract["status"] ==
            "M1250_C1_R11_ONE_SHOT_RELEASE_SOURCE_READY__FRESH_M1251_REQUIRED__NO_VCS_NO_EDA",
            "source status"); checks += 1
    require(release["status"] ==
            "AUTHORIZE_ONE_M1250_R11_UNIT_DELAY_VCS_ATTEMPT_AFTER_FRESH_M1251_HAMMER",
            "release status"); checks += 1
    require(contract["identity"]["runner_sha256"] == sha(RUNNER)
            == release["identity"]["runner_sha256"], "runner binding"); checks += 1
    require(contract["identity"]["release_checker_sha256"] == sha(Path(__file__)),
            "checker binding"); checks += 1
    require(release["identity"]["source_contract_sha256"] == sha(SOURCE_CONTRACT),
            "contract binding"); checks += 1
    prior = json.loads((M1247 / "review.json").read_text())
    require(prior["status"] == "PASS_M1247_RELEASE_AUTHORING_GO"
            and prior["score"] == 100 and prior["p0_count"] == 0
            and prior["p1_count"] == 0 and prior["p2_count"] == 0
            and prior["authorization"]["fresh_disjoint_release_authoring"] is True,
            "M1247 authorization"); checks += 1
    if not args.skip_author:
        verify_dir(AUTHOR); checks += 1
        author = json.loads((AUTHOR / "review.json").read_text())
        require(author["schema"] ==
                "m1250_m1247_m1246_c1_r11_vcs_release_author_review_r1_v1"
                and author["status"] ==
                "PASS_M1250_R11_ONE_SHOT_RELEASE_SOURCE__FRESH_M1251_HAMMER_REQUIRED"
                and author["bindings"]["runner_sha256"] == sha(RUNNER)
                and author["bindings"]["source_contract_sha256"] == sha(SOURCE_CONTRACT)
                and author["bindings"]["release_sha256"] == sha(RELEASE),
                "author binding"); checks += 1
    good = {name: "a" * 64 for name in (
        "M1250_EXPECTED_RELEASE_SHA256", "M1250_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256",
        "M1250_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256",
        "M1250_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256")}
    require(env_gate(good), "good env"); checks += 1
    for name in tuple(good):
        for value in (None, "b" * 63, "B" * 64):
            changed = dict(good)
            if value is None: changed.pop(name)
            else: changed[name] = value
            require(not env_gate(changed), "env mutation " + name); checks += 1
    attempt = HW / "results/.m1250_m1232r11_m1162_c1_common_charge_protocol_vcs_r11_attempt_consumed"
    result = HW / "results/m1250_m1232r11_m1162_c1_common_charge_protocol_unit_delay_vcs_r11_20260830"
    require(not os.path.lexists(attempt) and not os.path.lexists(result), "fresh namespace"); checks += 1
    require(not list((HW / "results").glob(".m1250_m1232r11_m1162_c1_common_charge_protocol_vcs_r11_work.*")), "stale work"); checks += 1
    require(not list((HW / "results").glob("m1250_m1232r11_m1162_c1_common_charge_protocol_unit_delay_vcs_r11_20260830.failed_or_incomplete.*")), "stale quarantine"); checks += 1
    for data in (contract, release):
        for key in ("functional_vcs_verified", "timing_verified", "cycles_measured",
                    "speedup", "ppa", "power", "energy", "system_speedup",
                    "paper_citable"):
            require(data["claim_boundary"][key] is False, "claim " + key); checks += 1
    print(json.dumps({"schema":"m1250_c1_r11_vcs_release_static_check_r1_v1",
                      "status":"PASS_M1250_SOURCE_ONLY__FRESH_M1251_REQUIRED__NO_VCS_NO_EDA",
                      "checks_passed":checks,"one_compile":True,"one_sim":True,
                      "compile_timeout_seconds":1200,"sim_timeout_seconds":1800,
                      "phase_pairs_required":7,"random_phase_pairs_required":24,
                      "zero_sva_error_fatal_required":True,
                      "failure_quarantine_recursive_seal":True,"automatic_retry":False,
                      "vcs_runs":0,"simv_runs":0,"all_eda_runs":0,
                      "docs359_sha256":sha(DOCS359)},indent=2,sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
