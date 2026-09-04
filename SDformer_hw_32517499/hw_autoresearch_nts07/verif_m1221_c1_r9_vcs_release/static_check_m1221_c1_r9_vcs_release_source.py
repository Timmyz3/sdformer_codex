#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Read-only static gate for the M1221 C1/R9 one-shot VCS release."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any


HW = Path(__file__).resolve().parents[1]
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1221_m1219r9_m1162_c1_common_charge_protocol_exact_sha_r9.sh"
FILELIST = HW / "dc_handoff/filelists/date_m1221_m1219r9_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
TB = HW / "verif_m1219r9_c1_common_charge_protocol/tb_m1219r9_m1162_common_charge_protocol_unit_delay_r9.sv"
R9_CHECKER = HW / "verif_m1219r9_c1_common_charge_protocol/check_m1219r9_source.py"
R9_TESTS = HW / "verif_m1219r9_c1_common_charge_protocol/test_m1219r9_source.py"
PARENT = HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
WRAPPER = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
SVA = HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
M1219_CONTRACT = HW / "contracts/m1219_m1218_m1213_c1_r9_observability_source_contract_r1_20260830.json"
M1218_FAILURE = HW / "reviews/m1218_m1213_c1_r8_vcs_timeout_failure_review_r1_20260830"
M1219_AUTHOR = HW / "reviews/m1219_m1218_c1_r9_observability_source_author_r1_20260830"
M1220 = HW / "reviews/m1220_m1219_c1_r9_observability_source_hammer_r1_20260830"
SOURCE_CONTRACT = HW / "contracts/m1221_m1220_m1219_c1_r9_vcs_launcher_source_contract_r1_20260830.json"
RELEASE = HW / "contracts/m1221_m1220_m1219_c1_r9_vcs_launch_release_r1_20260830.json"
AUTHOR = HW / "reviews/m1221_m1220_m1219_c1_r9_vcs_release_author_r1_20260830"
FOUNDRY = Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v")
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    RUNNER: "e018fa988cdf5f1a60033884b3ea5e95e4c4985ac799ead140aaba61e52df1d1",
    FILELIST: "e3819d608f7245ba0145164e5f8d6d3d8bc35fba22f48427aec1f3e9b2b70fc5",
    TB: "9666e086c69ecda4670622e063e9d54c89f94f2c77cd5eb012da54ca23492a75",
    R9_CHECKER: "2639ecfe321f004939ffe4d5de65586191ecb26c9f31f772473d92fdc7456268",
    R9_TESTS: "b365f3b8afef707359dbb54945684da953bbdd28a334201e438c7baebeaab563",
    PARENT: "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    M935: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    WRAPPER: "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    SVA: "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    M1219_CONTRACT: "fd4a23ea97395f47c49fd9183a51b842156b3535c494062028d9b72a1c389c67",
    M1218_FAILURE / "review.json": "fa397c84821ecd36c66eca0f598f6345ba7463943ad701e7b90acca119a7c460",
    M1218_FAILURE / "SHA256SUMS": "63cbc4084a883e0e83dd1ce6b1036db89aab3fbe86c37ecc6a43abc159a71261",
    M1218_FAILURE / "SHA256SUMS.seal.sha256": "73817b962f5fd76525f9f984184441811f1c2da00aa785a9a9dc4252cbb0e370",
    M1219_AUTHOR / "author_review.md": "0aa4b4515d0e507eee42d8d01dba07cffd24946c0bcad3e35764b0cef5c8d966",
    M1219_AUTHOR / "SHA256SUMS": "3924a2b4dc976de6e4c121c0e2a7254722078a2328891fee0f547e85d66b9647",
    M1219_AUTHOR / "SHA256SUMS.seal.sha256": "5a7007cecaaffa76cc5951951965dfa237e8fb42e4e06cb0ab2444380881f01e",
    M1220 / "review.json": "7004b6f30793971b3d297502d587edb99222e4a90f683a301b1e23dc84356572",
    M1220 / "SHA256SUMS": "d3a064202c2bfa8b257c190d898855451726b72b8ec9dce5736208c8f1daaa04",
    M1220 / "SHA256SUMS.seal.sha256": "fc05610ec4ea83059f8f61bd9263de7130e24c63cb88800fc1979f30ab91ba4d",
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
    sums = Path(str(path) + ".sha256"); outer = Path(str(path) + ".sha256.seal.sha256")
    require(sums.read_text().split() == [sha(path), path.name], "sidecar " + path.name)
    require(outer.read_text().split() == [sha(sums), sums.name], "outer " + path.name)


def verify_dir(root: Path) -> None:
    require(root.is_dir() and not root.is_symlink(), "sealed dir " + str(root))
    sums, outer = root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"
    require(outer.read_text().split() == [sha(sums), "SHA256SUMS"], "outer seal")
    listed: dict[str, str] = {}
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*")
        require(name not in listed and not Path(name).is_absolute() and ".." not in Path(name).parts,
                "unsafe member")
        listed[name] = digest
    actual: set[str] = set()
    for base_text, dirs, files in os.walk(root, followlinks=False):
        base = Path(base_text); dirs[:] = [name for name in dirs if not (base / name).is_symlink()]
        for name in files:
            path = base / name; rel = path.relative_to(root).as_posix()
            if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"} or path.is_symlink(): continue
            if stat.S_ISREG(path.lstat().st_mode): actual.add(rel)
    require(actual == set(listed), "sealed population")
    for name, digest in listed.items(): require(sha(root / name) == digest, "member drift " + name)


def audit_filelist(text: str) -> list[str]:
    errors = []
    expected = [str(path) for path in (FOUNDRY, PARENT, M935, WRAPPER, SVA, TB)]
    lines = text.splitlines()
    if lines != expected: errors.append("filelist exact ordered population")
    if any("m1210r8" in line for line in lines): errors.append("R8 TB leaked")
    return errors


def audit_runner(text: str) -> list[str]:
    errors = []
    def need(token: str, label: str) -> None:
        if token not in text: errors.append(label)
    for token in ("M1221_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256",
                  "M1221_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256",
                  "M1221_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256",
                  "phase_watchdog_timeout_dump.txt", "TIMEOUT_M1219R9",
                  "PHASE_M1219R9_RANDOM_TRANSACTION_ENTER",
                  "PHASE_M1219R9_RANDOM_TRANSACTION_COMPLETE",
                  "if rg -q '^TIMEOUT_M1219R9 '", "SIM_TIMEOUT_SECONDS=1800",
                  "RUN_FAILED_OR_INCOMPLETE.txt", "failed_or_incomplete.$$.quarantine",
                  "automatic_retry=false"):
        need(token, "missing runner token " + token)
    if text.count('"${VCS_BIN}" -full64') != 1: errors.append("compile count")
    if text.count('./simv -no_save') != 1: errors.append("sim count")
    if "rm -" in text: errors.append("destructive cleanup")
    try:
        author = text.index('for sealed in "${M1218_FAILURE}"')
        hammer = text.index('sha_exact "${M1221_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256}"')
        attempt = text.index('/bin/mkdir -- "${ATTEMPT}"')
        work = text.index('/bin/mkdir -- "${WORK}"')
        compile_at = text.index('"${VCS_BIN}" -full64')
        sim_at = text.index('./simv -no_save')
        if not (author < hammer < attempt < work < compile_at < sim_at): errors.append("gate order")
    except ValueError: errors.append("gate anchors")
    phase_loop = re.search(r"for phase in ([A-Z0-9_ ]+); do", text)
    expected_phases = ["DIRECTED", "RESET_PENDING", "STICKY_ATTACKS", "SERVICE_ATTACKS",
                       "RANDOM", "NORMAL_M935", "CLEAN_RESET_PREP"]
    if phase_loop is None or phase_loop.group(1).split() != expected_phases:
        errors.append("phase population")
    need('^PHASE_M1219R9_${phase}_ENTER( |$)', "phase enter gate")
    need('^PHASE_M1219R9_${phase}_COMPLETE( |$)', "phase complete gate")
    return errors


def env_gate(env: dict[str, str]) -> bool:
    names = ("M1221_EXPECTED_RELEASE_SHA256", "M1221_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256",
             "M1221_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256",
             "M1221_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256")
    return all(re.fullmatch(r"[0-9a-f]{64}", env.get(name, "")) is not None for name in names)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--skip-author", action="store_true")
    args = parser.parse_args(); checks = 0
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "identity drift " + str(path)); checks += 1
    for root in (M1218_FAILURE, M1219_AUTHOR, M1220): verify_dir(root); checks += 1
    for artifact in (M1219_CONTRACT, SOURCE_CONTRACT, RELEASE): sidecar(artifact); checks += 2
    require(not audit_filelist(FILELIST.read_text()), "filelist audit"); checks += 1
    require(not audit_runner(RUNNER.read_text()), "runner audit"); checks += 1
    completed = subprocess.run([str(PYTHON), "-I", str(R9_CHECKER)], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    require(completed.returncode == 0, "M1219 checker failed"); checks += 1
    contract = json.loads(SOURCE_CONTRACT.read_text()); release = json.loads(RELEASE.read_text())
    require(contract["status"] ==
            "M1221_C1_R9_ONE_SHOT_RELEASE_SOURCE_READY__FRESH_M1222_REQUIRED__NO_VCS_NO_EDA",
            "source status"); checks += 1
    require(release["status"] ==
            "AUTHORIZE_ONE_M1221_R9_UNIT_DELAY_VCS_ATTEMPT_AFTER_FRESH_M1222_HAMMER",
            "release status"); checks += 1
    require(contract["identity"]["runner_sha256"] == sha(RUNNER) == release["identity"]["runner_sha256"],
            "runner binding"); checks += 1
    require(contract["identity"]["release_checker_sha256"] == sha(Path(__file__)),
            "checker binding"); checks += 1
    require(release["identity"]["source_contract_sha256"] == sha(SOURCE_CONTRACT),
            "contract binding"); checks += 1
    if not args.skip_author:
        verify_dir(AUTHOR); checks += 1
        author = json.loads((AUTHOR / "review.json").read_text())
        require(contract["future_m1221_author"] == {
            "path": "reviews/m1221_m1220_m1219_c1_r9_vcs_release_author_r1_20260830",
            "schema": "m1221_m1220_m1219_c1_r9_vcs_release_author_review_r1_v1",
            "status": "PASS_M1221_R9_ONE_SHOT_RELEASE_SOURCE__FRESH_M1222_HAMMER_REQUIRED"},
            "future author declaration"); checks += 1
        require(author["schema"] == contract["future_m1221_author"]["schema"] and
                author["status"] == contract["future_m1221_author"]["status"] and
                author["bindings"]["runner_sha256"] == sha(RUNNER) and
                author["bindings"]["source_contract_sha256"] == sha(SOURCE_CONTRACT) and
                author["bindings"]["release_sha256"] == sha(RELEASE),
                "author binding"); checks += 1
    good = {name: "a" * 64 for name in (
        "M1221_EXPECTED_RELEASE_SHA256", "M1221_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256",
        "M1221_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256",
        "M1221_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256")}
    require(env_gate(good), "good env"); checks += 1
    for name in tuple(good):
        for value in (None, "b" * 63, "B" * 64):
            changed = dict(good)
            if value is None: changed.pop(name)
            else: changed[name] = value
            require(not env_gate(changed), "env mutation " + name); checks += 1
    attempt = HW / "results/.m1221_m1219r9_m1162_c1_common_charge_protocol_vcs_r9_attempt_consumed"
    result = HW / "results/m1221_m1219r9_m1162_c1_common_charge_protocol_unit_delay_vcs_r9_20260830"
    require(not os.path.lexists(attempt) and not os.path.lexists(result), "fresh namespace"); checks += 1
    require(not list((HW / "results").glob(".m1221_m1219r9_m1162_c1_common_charge_protocol_vcs_r9_work.*")), "stale work"); checks += 1
    require(not list((HW / "results").glob("m1221_m1219r9_m1162_c1_common_charge_protocol_unit_delay_vcs_r9_20260830.failed_or_incomplete.*")), "stale quarantine"); checks += 1
    for data in (contract, release):
        for key in ("functional_vcs_verified", "timing_verified", "cycles_measured", "speedup",
                    "ppa", "power", "energy", "system_speedup", "paper_citable"):
            require(data["claim_boundary"][key] is False, "claim " + key); checks += 1
    print(json.dumps({"schema":"m1221_c1_r9_vcs_release_static_check_r1_v1",
                      "status":"PASS_M1221_SOURCE_ONLY__FRESH_M1222_REQUIRED__NO_VCS_NO_EDA",
                      "checks_passed":checks,"one_compile":True,"one_bounded_sim":True,
                      "phase_pairs_required":7,"random_phase_pairs_required":24,
                      "timeout_dump_and_failure_quarantine":True,"automatic_retry":False,
                      "vcs_runs":0,"simv_runs":0,"all_eda_runs":0,
                      "docs359_sha256":sha(DOCS359)},indent=2,sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
