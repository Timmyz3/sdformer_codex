#!/usr/bin/env python3
"""Read-only M1204 R7 launcher/release source gate.  Invokes no EDA."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

HW = Path(__file__).resolve().parents[1]
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1204_m1198r7_m1162_c1_common_charge_protocol_exact_sha_r7.sh"
CONTRACT = HW / "contracts/m1204_m1201_m1198_c1_r7_vcs_launcher_source_contract_r1_20260830.json"
RELEASE = HW / "contracts/m1204_m1201_m1198_c1_r7_vcs_launch_release_r1_20260830.json"
R7_CHECKER = HW / "verif_m1198r7_c1_common_charge_protocol/static_check_m1198r7_m1162_vcs_source.py"
FILELIST = HW / "dc_handoff/filelists/date_m1198r7_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
TB = HW / "verif_m1193r6_c1_common_charge_protocol/tb_m1193r6_m1162_common_charge_protocol_unit_delay_r6.sv"
SVA = HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
M1198_CONTRACT = HW / "contracts/m1198_m1194_m1193_m1162_c1_r7_source_gate_repair_contract_r1_20260830.json"
M1198_AUTHOR = HW / "reviews/m1198_m1194_c1_r7_source_gate_repair_author_receipt_r1_20260830"
M1201 = HW / "reviews/m1201_m1198_c1_r7_source_gate_repair_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    R7_CHECKER: "b1cfb957d5c4fc518d46980040afa61288eb7dcaa79fa5e6c45e25b097094795",
    FILELIST: "444ff65d575c6e897f9d459689f323290f16eb89c962c91b395964c7850fcbfa",
    TB: "0fcc2138ef5d716735eea01dee25a148a5223b1d6adf1e3b2fa464341fbf1345",
    SVA: "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    M1198_CONTRACT: "44c5a3add48ef74ef0698f81f20fef417989c17b74df3e1d366cf404b7ce5488",
    M1198_AUTHOR / "review.json": "c47172693484f9098881b745e34d797d481906e8857bc93a169ce6bd701605df",
    M1198_AUTHOR / "SHA256SUMS": "184946a8e314414919b7702b23a1d2e80dc1c15623dbb9d04090158021e1eac9",
    M1198_AUTHOR / "SHA256SUMS.seal.sha256": "7286441a67b9cb1196dec9356e5bf1b33ca5a6e90522ff4b404137c6fc76768b",
    M1201 / "review.md": "a431090064fa8a8b0180c1ad188866ad3b44bacb7208e2ea1233618719ef675a",
    M1201 / "review.json": "b78fc16baf67025b2a500f0a9a26b7392f8752dc4af4afbf066d685d89f495c7",
    M1201 / "SHA256SUMS": "26323508b4186a0e3c718afceb3d6deeeb2b2f4467418f863b2cab102cf1558f",
    M1201 / "SHA256SUMS.seal.sha256": "3317d266834a69c5d9ca6ae747ff5285409fc8f4b8a650d62507a38deaa9a748",
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
    require(directory.is_dir() and not directory.is_symlink(), f"sealed dir {directory}")
    sums = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(outer.read_text().split() == [sha(sums), "SHA256SUMS"],
            f"outer seal {directory}")
    listed: dict[str, str] = {}
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*")
        require(name not in listed and not Path(name).is_absolute()
                and ".." not in Path(name).parts, "safe manifest member")
        listed[name] = digest
    actual: set[str] = set()
    for root, dirs, files in os.walk(directory, followlinks=False):
        base = Path(root); dirs[:] = [d for d in dirs if not (base / d).is_symlink()]
        for name in files:
            member = base / name
            rel = member.relative_to(directory).as_posix()
            if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"} or member.is_symlink():
                continue
            if stat.S_ISREG(member.lstat().st_mode): actual.add(rel)
    require(actual == set(listed), "complete recursive membership")
    for name, digest in listed.items():
        require(sha(directory / name) == digest, "sealed member drift " + name)


def sidecar(path: Path) -> None:
    sums = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(sums.read_text().split() == [sha(path), path.name], "sidecar " + path.name)
    require(outer.read_text().split() == [sha(sums), sums.name], "outer sidecar " + path.name)


def main() -> None:
    for path in (RUNNER, CONTRACT, RELEASE):
        require(path.is_file() and not path.is_symlink(), "M1204 source file " + str(path))
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "identity drift " + str(path))
    verify_dir(M1198_AUTHOR)
    verify_dir(M1201)
    for path in (M1198_CONTRACT, CONTRACT, RELEASE): sidecar(path)

    c = json.loads(CONTRACT.read_text()); r = json.loads(RELEASE.read_text())
    a = json.loads((M1198_AUTHOR / "review.json").read_text())
    h = json.loads((M1201 / "review.json").read_text())
    require(c["status"] ==
            "M1204_R7_RELEASE_SOURCE_READY__FRESH_M1206_HAMMER_REQUIRED__NO_VCS_NO_EDA",
            "source contract status")
    require(r["status"] ==
            "AUTHORIZE_ONE_M1204_R7_FUNCTIONAL_VCS_ATTEMPT_AFTER_FRESH_M1206_HAMMER",
            "release status")
    require(c["identity"]["runner_sha256"] == sha(RUNNER)
            and r["identity"]["runner_sha256"] == sha(RUNNER), "runner identity")
    require(r["identity"]["source_contract_sha256"] == sha(CONTRACT),
            "release contract identity")
    require(a["status"] ==
            "PASS_R7_SOURCE_ONLY__FRESH_DIFFERENT_AUTHOR_HAMMER_REQUIRED__NO_VCS_NO_EDA",
            "M1198 status")
    require(h["status"] ==
            "PASS_SOURCE_HAMMER__SUCCESSOR_RELEASE_AUTHORING_ONLY__NO_VCS_NO_EDA",
            "M1201 status")
    expected_regression = {"assertions": 16, "covers": 6,
        "protocol_attacks": 7, "service_assumption_attacks": 2,
        "deterministic_legal_transactions": 24, "legal_masks_clear": 29,
        "reset_states": 3, "minimum_completed_issue_ii": 2,
        "normal_m935_rows": 1, "normal_m935_tasks": 1}
    require(h["preserved_regression"] == expected_regression, "M1201 regression")

    runner = RUNNER.read_text()
    required = [
        "M1204_EXPECTED_RELEASE_SHA256",
        "M1204_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256",
        "M1204_EXPECTED_RELEASE_HAMMER_OUTER_SHA256",
        "m1206_m1204_m1198_c1_r7_vcs_release_source_hammer_review_r1_v1",
        "PASS_M1206_M1204_C1_R7_VCS_RELEASE_HAMMER__AUTHORIZE_ONE_LAUNCH",
        "M1204_R7_RELEASE_SOURCE_READY__FRESH_M1206_HAMMER_REQUIRED__NO_VCS_NO_EDA",
        "AUTHORIZE_ONE_M1204_R7_FUNCTIONAL_VCS_ATTEMPT_AFTER_FRESH_M1206_HAMMER",
        "COVERAGE_M1193R6_PROTOCOL weight_first=1",
        "COVERAGE_M1193R6_RESETS_ATTACKS reset_partial=1",
        "COVERAGE_M1193R6_SERVICE_ASSUMPTIONS weight_payload_mutation=1",
        "COVERAGE_M1193R6_FROZEN_M935 normal_issues=2",
        "service_skew_isolated=1 reachable_core_ready_force=0 boundary_fault=0 core_fault=0",
        "directed_random=24 protocol_attacks=7 service_assumption_attacks=2",
        "minimum_completed_issue_ii':2",
        "functional_vcs_verified':True",
        "timing_verified':False",
    ]
    for token in required: require(token in runner, "runner token " + token)
    hammer_gate = runner.index('verify_recursive_seal "${RELEASE_HAMMER}"')
    attempt = runner.index('/bin/mkdir -- "${ATTEMPT}"')
    vcs = runner.index('"${VCS_BIN}" -full64')
    require(hammer_gate < attempt < vcs, "hammer-before-attempt-before-VCS order")
    require(runner.count('"${VCS_BIN}" -full64') == 1, "one VCS compile")
    require(runner.count('./simv -no_save') == 1, "one simv run")
    require("dc_shell" in runner and "pt_shell" in runner and "fm_shell" in runner,
            "EDA collision list")
    require("SNPSLMD_LICENSE_FILE" in runner and "UNIT_DELAY" in runner,
            "foundry functional setup")

    attempt_path = HW / "results/.m1204_m1198r7_m1162_c1_common_charge_protocol_vcs_r7_attempt_consumed"
    result_path = HW / "results/m1204_m1198r7_m1162_c1_common_charge_protocol_unit_delay_vcs_r7_20260830"
    require(not attempt_path.exists() and not result_path.exists(), "fresh M1204 namespace")
    for d in (c, r):
        b = d["claim_boundary"]
        for key in ("functional_vcs_verified", "timing_verified", "cycles_measured",
                    "speedup", "ppa", "power", "energy", "system_speedup",
                    "paper_citable"):
            require(b[key] is False, "claim boundary " + key)
    require(c["authorization"]["vcs_compiles"] == 0
            and c["authorization"]["simv_runs"] == 0, "source inert")
    require(r["authorization"] ==
            {"vcs_compiles": 1, "simv_runs": 1, "all_other_eda_runs": 0},
            "release one-shot authorization")

    print(json.dumps({
        "schema": "m1204_c1_r7_vcs_release_source_static_check_v1",
        "status": "PASS_M1204_RELEASE_SOURCE_ONLY__FRESH_M1206_HAMMER_REQUIRED__NO_VCS_NO_EDA",
        "checks_passed": checks,
        "m1198_m1201_authority_seals": True,
        "r7_source_identity": True,
        "hammer_before_attempt_before_vcs": True,
        "exact_regression_gate": expected_regression,
        "namespace_fresh": True,
        "vcs_runs": 0, "simv_runs": 0, "all_eda_runs": 0,
        "docs359_sha256": sha(DOCS359),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
