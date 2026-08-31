#!/usr/bin/env python3
"""Read-only source gate for the M1213 C1/R8 one-shot VCS release."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path


HW = Path(__file__).resolve().parents[1]
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1213_m1210r8_m1162_c1_common_charge_protocol_exact_sha_r8.sh"
SOURCE_CONTRACT = HW / "contracts/m1213_m1212_m1210_c1_r8_vcs_launcher_source_contract_r1_20260830.json"
RELEASE = HW / "contracts/m1213_m1212_m1210_c1_r8_vcs_launch_release_r1_20260830.json"
R8_CHECKER = HW / "verif_m1210r8_c1_common_charge_protocol/static_check_m1210r8_m1162_vcs_source.py"
FILELIST = HW / "dc_handoff/filelists/date_m1210r8_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
TB = HW / "verif_m1210r8_c1_common_charge_protocol/tb_m1210r8_m1162_common_charge_protocol_unit_delay_r8.sv"
SVA = HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
WRAPPER = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
PARENT = HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
M1210_CONTRACT = HW / "contracts/m1210_m1207_m1198_m1162_c1_r8_random_request_quiesce_source_contract_r1_20260830.json"
M1210_AUTHOR = HW / "reviews/m1210_m1207_c1_r8_random_request_quiesce_author_receipt_r1_20260830"
M1212 = HW / "reviews/m1212_m1210_c1_r8_random_request_quiesce_source_hammer_r1_20260830"
FOUNDRY = Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v")
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    R8_CHECKER: "cce8219a13d7584f1c35e262ac4de3e4a935fddc53652d6ce322e7e5f94daa96",
    FILELIST: "048253d22301df9fb84502ff35f5129459a5b43e4ff9e8d11ea62973f7047af6",
    TB: "060ec9d5ae6085a0dd013160d22f63e21615730384ddaef342eb3fa77e17947b",
    SVA: "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    WRAPPER: "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    M935: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    PARENT: "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    M1210_CONTRACT: "26ca340e8f33ca936b169c638862bc3a76f7233035d680cc14ddb7389bcc5d07",
    M1210_AUTHOR / "review.json": "d9671bff7efa1e808d5008c23d02df119df4553b60d5782fb2e0ba8bb73efc4a",
    M1210_AUTHOR / "SHA256SUMS": "cf9e56adcc15c33ca7663502cdad741c1287dc64d8e2f79df55b9120d986cc5a",
    M1210_AUTHOR / "SHA256SUMS.seal.sha256": "28a209d39c1211a0c9c20b43b471cea68d1e5492d516c332120cc1098a773826",
    M1212 / "review.json": "550d4459ce34f0b01c43ac913123e247270b66c7bd83678d01228b227839fe4d",
    M1212 / "SHA256SUMS": "349306f94a43c93acbee71e926ef36474d2bdf0bb1c12f597037fd8b597165a7",
    M1212 / "SHA256SUMS.seal.sha256": "92e1640e01288841a768d165dc66bbb5bd87fa3f0385bfc88f5843099ece9909",
    FOUNDRY: "8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d",
    VCS: "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    PYTHON: "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

checks = 0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise AssertionError(message)


def verify_dir(directory: Path) -> None:
    require(directory.is_dir() and not directory.is_symlink(), "sealed dir " + str(directory))
    sums, outer = directory / "SHA256SUMS", directory / "SHA256SUMS.seal.sha256"
    require(outer.read_text().split() == [sha(sums), "SHA256SUMS"], "outer seal")
    listed: dict[str, str] = {}
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in listed and not Path(name).is_absolute()
                and ".." not in Path(name).parts, "safe member")
        listed[name] = digest
    actual: set[str] = set()
    for root, dirs, files in os.walk(directory, followlinks=False):
        base = Path(root)
        dirs[:] = [name for name in dirs if not (base / name).is_symlink()]
        for name in files:
            member = base / name
            rel = member.relative_to(directory).as_posix()
            if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"} or member.is_symlink():
                continue
            if stat.S_ISREG(member.lstat().st_mode):
                actual.add(rel)
    require(actual == set(listed), "complete sealed membership")
    for name, digest in listed.items():
        require(sha(directory / name) == digest, "sealed member drift " + name)


def sidecar(path: Path) -> None:
    sums = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(sums.read_text().split() == [sha(path), path.name], "sidecar " + path.name)
    require(outer.read_text().split() == [sha(sums), sums.name], "sidecar outer " + path.name)


def env_gate(env: dict[str, str]) -> bool:
    names = (
        "M1213_EXPECTED_RELEASE_SHA256",
        "M1213_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256",
        "M1213_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256",
        "M1213_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256",
    )
    return all(re.fullmatch(r"[0-9a-f]{64}", env.get(name, "")) is not None for name in names)


def main() -> None:
    for path in (RUNNER, SOURCE_CONTRACT, RELEASE):
        require(path.is_file() and not path.is_symlink(), "M1213 source " + str(path))
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "identity drift " + str(path))
    verify_dir(M1210_AUTHOR)
    verify_dir(M1212)
    for path in (M1210_CONTRACT, SOURCE_CONTRACT, RELEASE):
        sidecar(path)

    contract = json.loads(SOURCE_CONTRACT.read_text())
    release = json.loads(RELEASE.read_text())
    runner = RUNNER.read_text()
    require(contract["status"] ==
            "M1213_R8_ACYCLIC_RELEASE_SOURCE_READY__FRESH_RELEASE_HAMMER_REQUIRED__NO_VCS_NO_EDA",
            "source contract status")
    require(release["status"] ==
            "AUTHORIZE_ONE_M1213_R8_FUNCTIONAL_VCS_ATTEMPT_AFTER_FRESH_RELEASE_HAMMER",
            "release status")
    require(contract["identity"]["runner_sha256"] == sha(RUNNER)
            and release["identity"]["runner_sha256"] == sha(RUNNER), "runner pin")
    require(contract["identity"]["m1213_release_checker_sha256"] == sha(Path(__file__)),
            "checker pin")
    require(release["identity"]["source_contract_sha256"] == sha(SOURCE_CONTRACT),
            "source contract pin")
    for token in (
        "verify_recursive_seal \"${RELEASE_HAMMER}\"",
        "M1213_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256",
        "M1213_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256",
        "M1213_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256",
        "self-reference forbidden",
        "COVERAGE_M1210R8_PROTOCOL weight_first=1",
        "random_request_quiesce=24",
        "exactly_one_random_request_handshake=1",
        "COVERAGE_M1210R8_RESETS_ATTACKS reset_partial=1",
        "COVERAGE_M1210R8_SERVICE_ASSUMPTIONS weight_payload_mutation=1",
        "COVERAGE_M1210R8_FROZEN_M935 normal_issues=2",
        "PASS_M1210R8_M1162_COMMON_CHARGE_PROTOCOL_UNIT_DELAY_CANDIDATE",
        "functional_vcs_verified':True", "timing_verified':False",
        "automatic_retry':False",
    ):
        require(token in runner, "runner token " + token)
    seal_gate = runner.index('verify_recursive_seal "${RELEASE_HAMMER}"')
    review_gate = runner.index('sha_exact "${M1213_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256}"')
    manifest_gate = runner.index('sha_exact "${M1213_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256}"')
    outer_gate = runner.index('sha_exact "${M1213_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256}"')
    attempt = runner.index('/bin/mkdir -- "${ATTEMPT}"')
    compile_call = runner.index('"${VCS_BIN}" -full64')
    require(seal_gate < review_gate < manifest_gate < outer_gate < attempt < compile_call,
            "hammer gates before attempt before VCS")
    require(runner.count('"${VCS_BIN}" -full64') == 1
            and runner.count('./simv -no_save') == 1, "one compile one sim")
    require("rm -" not in runner and "retry" not in runner.lower().replace("automatic_retry", ""),
            "no destructive cleanup or retry loop")

    names = tuple(release["required_environment"])
    good = {name: "a" * 64 for name in names if name.startswith("M1213_")}
    require(env_gate(good), "complete environment accepted")
    for name in tuple(good):
        missing = dict(good); missing.pop(name)
        require(not env_gate(missing), "missing env rejected " + name)
        short = dict(good); short[name] = "b" * 63
        require(not env_gate(short), "short env rejected " + name)
        upper = dict(good); upper[name] = "B" * 64
        require(not env_gate(upper), "uppercase env rejected " + name)

    attempt_path = HW / "results/.m1213_m1210r8_m1162_c1_common_charge_protocol_vcs_r8_attempt_consumed"
    result_path = HW / "results/m1213_m1210r8_m1162_c1_common_charge_protocol_unit_delay_vcs_r8_20260830"
    require(not attempt_path.exists() and not result_path.exists(), "fresh M1213 namespace")
    require(not list((HW / "results").glob(".m1213_m1210r8_m1162_c1_common_charge_protocol_vcs_r8_work.*")),
            "no stale work")
    require(not list((HW / "results").glob(
            "m1213_m1210r8_m1162_c1_common_charge_protocol_unit_delay_vcs_r8_20260830.failed_or_incomplete.*")),
            "no prior quarantine")
    for data in (contract, release):
        for key in ("functional_vcs_verified", "timing_verified", "cycles_measured",
                    "speedup", "ppa", "power", "energy", "system_speedup", "paper_citable"):
            require(data["claim_boundary"][key] is False, "claim boundary " + key)
    require(contract["authorization"] == {"vcs_compiles": 0, "simv_runs": 0,
            "all_other_eda_runs": 0, "gpu_runs": 0, "network_runs": 0}, "source inert")
    require(release["authorization"] == {"vcs_compiles": 1, "simv_runs": 1,
            "all_other_eda_runs": 0}, "one-shot authorization")
    print(json.dumps({
        "schema": "m1213_c1_r8_vcs_release_source_static_check_v1",
        "status": "PASS_M1213_RELEASE_SOURCE_ONLY__FRESH_DIFFERENT_AUTHOR_RELEASE_HAMMER_REQUIRED__NO_VCS_NO_EDA",
        "checks_passed": checks,
        "m1210_m1212_recursive_seals": True,
        "exact_source_tool_library_pins": True,
        "acyclic_three_runtime_hammer_hashes": True,
        "environment_mutations_rejected": 12,
        "fresh_namespace": True,
        "one_compile_one_sim": True,
        "automatic_retry": False,
        "vcs_runs": 0, "simv_runs": 0, "all_eda_runs": 0,
        "docs359_sha256": sha(DOCS359),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
