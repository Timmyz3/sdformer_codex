#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Fail-closed source gate for the one-shot C1/R16 functional VCS runner."""
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
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1354_m1337r15_m1162_c1_real_m935_runtime_witness_unit_delay_exact_sha.sh"
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
CHECKER = Path(__file__).resolve()
TESTS = HERE / "test_m1354_c1_r16_vcs_release_source.py"
CONTRACT = HW / "contracts/m1354_c1_r16_real_m935_runtime_witness_vcs_release_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1354_c1_r16_real_m935_runtime_witness_vcs_release_source_author_r1_20260831"
FUTURE_HAMMER = HW / "reviews/m1355_m1354_c1_r16_real_m935_runtime_witness_vcs_release_blind_hammer_r1_20260831"
FUTURE_RELEASE = HW / "contracts/m1356_m1355_m1354_c1_r16_real_m935_runtime_witness_vcs_launch_release_r1_20260831.json"
FOUNDRY = Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v")
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
ATTEMPT = HW / "results/.m1354_c1_r16_real_m935_runtime_witness_vcs_attempt_consumed"
RESULT = HW / "results/m1354_c1_r16_real_m935_runtime_witness_unit_delay_vcs_r1_20260831"

EXPECTED = {
    RUNNER: "b95d4568fa8497ebe47cb96e8a4fe0fa4f8320eb2d9f3878214122d444cd3ec6",
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
    FOUNDRY: "8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d",
    VCS: "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    PYTHON: "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
R16_SEALS = {
    R16_AUTHOR: {
        "review": "a5b136fce2bc3c5b5a5920b1e88cff092b1228b49a7ff6fd9959ff95e06772e5",
        "manifest": "bd875634a0be33cb5dc2f0600734fa90e014ade961658c3d1f480ce40425a616",
        "outer": "c9700d4411dd087b12494e4aaf2f5fde0de52f7e30b7397573b205371837e99f",
    },
    R16_HAMMER: {
        "review": "74969404ea26e5a522c205328c05a3527fca6daeefb74f6fb103cacb990e94ea",
        "manifest": "d703fb23ff2a7726049f58d09e7d304d0e4e8adcaa781f34856115dcb4de40e6",
        "outer": "29c6bf6de6a7ed91dc523dfc3360d7731c324a24cd3548a0fe3a346018e37ec7",
    },
}


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), "not regular: " + str(path))


def exact_byte_gate(path: Path, data: bytes) -> bool:
    return path in EXPECTED and hashlib.sha256(data).hexdigest() == EXPECTED[path]


def env_gate(env: dict[str, str]) -> bool:
    names = (
        "M1354_EXPECTED_RELEASE_SHA256",
        "M1354_EXPECTED_HAMMER_REVIEW_SHA256",
        "M1354_EXPECTED_HAMMER_MANIFEST_SHA256",
        "M1354_EXPECTED_HAMMER_OUTER_SEAL_FILE_SHA256",
    )
    return all(re.fullmatch(r"[0-9a-f]{64}", env.get(name, "")) is not None
               for name in names)


def verify_dir(root: Path, pins: dict[str, str] | None = None) -> None:
    regular(root / "SHA256SUMS")
    regular(root / "SHA256SUMS.seal.sha256")
    require((root / "SHA256SUMS.seal.sha256").read_text().split() ==
            [sha(root / "SHA256SUMS"), "SHA256SUMS"], "outer seal drift")
    listed: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in listed and not Path(name).is_absolute()
                and ".." not in Path(name).parts, "unsafe/duplicate manifest member")
        listed[name] = digest
    actual: set[str] = set()
    for base_text, dirs, files in os.walk(root, followlinks=False):
        base = Path(base_text)
        dirs[:] = [name for name in dirs if not (base / name).is_symlink()]
        for name in files:
            path = base / name
            rel = path.relative_to(root).as_posix()
            if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
                continue
            regular(path)
            actual.add(rel)
    require(actual == set(listed), "recursive manifest membership drift")
    for name, digest in listed.items():
        require(sha(root / name) == digest, "manifest digest drift: " + name)
    if pins is not None:
        require(sha(root / "review.json") == pins["review"], "review pin drift")
        require(sha(root / "SHA256SUMS") == pins["manifest"], "manifest pin drift")
        require(sha(root / "SHA256SUMS.seal.sha256") == pins["outer"], "outer pin drift")


def check_contract_dict(contract: dict[str, Any]) -> None:
    require(contract.get("schema") ==
            "m1354_c1_r16_real_m935_runtime_witness_vcs_release_source_contract_r1_v1",
            "schema drift")
    require(contract.get("status") ==
            "M1354_C1_R16_VCS_RELEASE_SOURCE_READY__FRESH_M1355_REQUIRED__NO_LAUNCH",
            "status drift")
    identity = contract.get("identity", {})
    expected_identity = {
        "runner_path": RUNNER.relative_to(HW).as_posix(),
        "runner_sha256": sha(RUNNER),
        "filelist_path": FILELIST.relative_to(HW).as_posix(),
        "filelist_sha256": sha(FILELIST),
        "source_checker_path": CHECKER.relative_to(HW).as_posix(),
        "source_checker_sha256": sha(CHECKER),
        "source_tests_path": TESTS.relative_to(HW).as_posix(),
        "source_tests_sha256": sha(TESTS),
    }
    require(identity == expected_identity, "source identity drift")
    require(contract.get("r16_authority") == {
        "checker_sha256": EXPECTED[R16_CHECKER],
        "tests_sha256": EXPECTED[R16_TESTS],
        "contract_sha256": EXPECTED[R16_CONTRACT],
        "author_review_sha256": R16_SEALS[R16_AUTHOR]["review"],
        "author_manifest_sha256": R16_SEALS[R16_AUTHOR]["manifest"],
        "author_outer_file_sha256": R16_SEALS[R16_AUTHOR]["outer"],
        "blind_review_sha256": R16_SEALS[R16_HAMMER]["review"],
        "blind_manifest_sha256": R16_SEALS[R16_HAMMER]["manifest"],
        "blind_outer_file_sha256": R16_SEALS[R16_HAMMER]["outer"],
        "source_admitted": True,
    }, "R16 authority drift")
    require(contract.get("future_release") == {
        "hammer_path": FUTURE_HAMMER.relative_to(HW).as_posix(),
        "release_path": FUTURE_RELEASE.relative_to(HW).as_posix(),
        "fresh_different_author_hammer_required": True,
        "launch_authorized": False,
        "vcs_compiles_now": 0,
        "simv_runs_now": 0,
        "automatic_retry": False,
    }, "future release drift")
    execution = contract.get("author_execution", {})
    require(all(execution.get(key) is False for key in
                ("release", "vcs", "simv", "dc", "pt", "ptpx", "eda", "gpu", "remote")),
            "author execution drift")
    boundary = contract.get("claim_boundary", {})
    require(boundary.get("source_only") is True and all(boundary.get(key) is False
            for key in ("functional_vcs", "timing_verified", "cycles_measured",
                        "speedup", "ppa", "power", "energy", "system_speedup",
                        "headline")), "claim boundary drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-author", action="store_true")
    args = parser.parse_args()
    checks = 0
    for path, digest in EXPECTED.items():
        regular(path)
        require(sha(path) == digest, "frozen exact-byte drift: " + str(path))
        checks += 1
    expected_filelist = [str(path) for path in (FOUNDRY, PARENT, M935, WRAPPER, SVA, TB, WITNESS)]
    require(FILELIST.read_text().splitlines() == expected_filelist, "filelist/order drift")
    checks += 1
    syntax = subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, check=False)
    require(syntax.returncode == 0, "runner shell syntax")
    runner = RUNNER.read_text()
    require(runner.count('"${VCS_BIN}" -full64') == 1, "one compile")
    require(runner.count('./simv -no_save') == 1, "one simulation")
    require(runner.count('/usr/bin/timeout --signal=TERM --kill-after=30s') == 2,
            "two bounded tool calls")
    require("automatic_retry=true" not in runner and "rm -" not in runner,
            "retry/destructive drift")
    checks += 5
    for root, pins in R16_SEALS.items():
        verify_dir(root, pins)
        checks += 1
    regular(CONTRACT)
    check_contract_dict(json.loads(CONTRACT.read_text()))
    checks += 1
    if not args.skip_author:
        verify_dir(AUTHOR)
        review = json.loads((AUTHOR / "review.json").read_text())
        require(review.get("status") ==
                "PASS_M1354_C1_R16_VCS_RELEASE_SOURCE__FRESH_M1355_REQUIRED",
                "author status drift")
        require(review.get("bindings") == {
            "runner_sha256": sha(RUNNER),
            "source_checker_sha256": sha(CHECKER),
            "source_tests_sha256": sha(TESTS),
            "source_contract_sha256": sha(CONTRACT),
        }, "author bindings drift")
        checks += 1
    require(not os.path.lexists(ATTEMPT) and not os.path.lexists(RESULT),
            "attempt/result namespace not fresh")
    require(not FUTURE_HAMMER.exists() and not FUTURE_RELEASE.exists(),
            "future release unexpectedly exists during source authoring")
    checks += 2
    good = {name: "a" * 64 for name in (
        "M1354_EXPECTED_RELEASE_SHA256",
        "M1354_EXPECTED_HAMMER_REVIEW_SHA256",
        "M1354_EXPECTED_HAMMER_MANIFEST_SHA256",
        "M1354_EXPECTED_HAMMER_OUTER_SEAL_FILE_SHA256",
    )}
    require(env_gate(good), "good env rejected")
    for name in tuple(good):
        bad = dict(good); bad.pop(name)
        require(not env_gate(bad), "missing env accepted: " + name)
    checks += 5
    print(json.dumps({
        "schema": "m1354_c1_r16_vcs_release_source_check_r1_v1",
        "status": "PASS_M1354_SOURCE_ONLY__FRESH_M1355_REQUIRED__NO_VCS_NO_EDA",
        "checks_passed": checks,
        "exact_byte_corpus": True,
        "one_compile": True,
        "one_sim": True,
        "automatic_retry": False,
        "vcs_runs": 0,
        "simv_runs": 0,
        "eda_runs": 0,
        "docs359_sha256": sha(DOCS359),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
