#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Static source checker for the additive M1018/M1019/M1022 repair chain.

It performs no EDA and intentionally accepts an absent future M1020 independent
hammer.  M1022 itself remains inert until that sealed authority exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


HW = Path(__file__).resolve().parents[2]
RUNNER = HW / "dc_handoff/scripts/run_m1022_m1001_c2_mapped_gate_saif_one_shot_r3.sh"
RELEASE = HW / "contracts/m1019_m1018_m1001_c2_mapped_gate_saif_launch_release_r3_20260829.json"
AUDIT = HW / "reviews/m1018_m1013_c2_saif_compile_failure_audit_r1_20260829"
M1020 = HW / "reviews/m1020_m1019_m1018_m1022_c2_saif_release_hammer_r1_20260829"
RESULTS = HW / "results"

EXPECTED = {
    "runner": "dbaa5b0b9619cb60b556a42f27e9e926a56bcb22d4627c13048b70a3fc74af1b",
    "release": "fcd5a659a8b057f5e84e1b8691645cf1cf66ecafed0cf6dec0827ed9a571f609",
    "release_sidecar": "56a92631e49884614eb689abccd4d7022367735f736defbc2dfea28ef11fd2ea",
    "release_outer": "6e839e3d37b95f40bdc276441ad8068c47980dd665f4f36db69cccf3ab177e7e",
    "audit_review": "55b1656b9f903e684ff5c418081f89dee71910585067ca807b33032d923a95a8",
    "audit_manifest": "223c80b45beaf81f2ce8dd2d8eb265c7bb46a7bb404594ce661d787082bfcf12",
    "audit_outer": "5c096458a507e99e922ba6a0658ac7e28bf4f2710a2f49cc971d15c738f32146",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def audit_runner_text(text: str) -> None:
    required = (
        'readonly expected_vcs_home=/opt/synopsys/vcs/V-2023.12-SP1',
        'export VCS_HOME="${expected_vcs_home}"',
        'export PATH="${VCS_HOME}/bin:/usr/bin:/bin"',
        'vcs="${VCS_HOME}/bin/vcs"',
        'vcs_msg_report="${VCS_HOME}/bin/vcsMsgReport"',
        'b34e06a92b05856532f868d32c0c81f1708506096856ad9a97bd27e2bd60215b',
        'M1022_EXPECTED_RUNNER_SHA256',
        'M1022_EXPECTED_M1002_OUTER_SHA256',
        'M1022_EXPECTED_M1018_OUTER_SHA256',
        'M1022_EXPECTED_M1020_OUTER_SHA256',
        'PASS_M1020_M1019_M1018_M1022_C2_SAIF_RELEASE_HAMMER',
        'result="${hw_root}/results/m1022_m1001_c2_three_axis_mapped_gate_saif_r3_20260829"',
        'attempt="${hw_root}/results/.m1022_m1001_c2_three_axis_mapped_gate_saif_attempt_consumed"',
        'for axis in k1 k8 k1x8; do',
        'for case_id in 0 1 2 3 4; do',
        'collision_gate',
        'mkdir "${attempt}"',
    )
    for token in required:
        require(token in text, "runner token absent: " + token)
    require(text.index('expect_sha "${vcs_msg_report}"') < text.index('mkdir "${attempt}"'),
            "vcsMsgReport identity is not pre-attempt")
    require(text.index('verify_seal "${release_hammer}"') < text.index('mkdir "${attempt}"'),
            "M1020 authority is not pre-attempt")
    require(text.index('"${vcs}" -full64') < text.index('for case_id in 0 1 2 3 4; do'),
            "compile is not fresh per axis")
    ucli = HW / "dc_handoff/scripts/m979_c2_mapped_gate_per_case_saif.ucli.tcl"
    require("power tb_m979_c2_three_axis_mapped_gate_case_saif.dut" in ucli.read_text(),
            "UCLI is not DUT-only")
    require("M1005_EXPECTED_" not in text and
            'result="${hw_root}/results/m1013_' not in text and
            'attempt="${hw_root}/results/.m1013_' not in text,
            "stale execution namespace in runner")


def verify_seal(directory: Path, review_sha: str, manifest_sha: str, outer_sha: str) -> None:
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(sha(review) == review_sha and sha(manifest) == manifest_sha and sha(outer) == outer_sha,
            "sealed audit identity drift")
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=directory,
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["sha256sum", "-c", "SHA256SUMS.seal.sha256"], cwd=directory,
                   check=True, stdout=subprocess.DEVNULL)


def main() -> dict:
    require(sha(RUNNER) == EXPECTED["runner"], "runner identity drift")
    audit_runner_text(RUNNER.read_text())
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    sidecar = Path(str(RELEASE) + ".sha256")
    outer = Path(str(RELEASE) + ".sha256.seal.sha256")
    require(sha(RELEASE) == EXPECTED["release"] and sha(sidecar) == EXPECTED["release_sidecar"] and
            sha(outer) == EXPECTED["release_outer"], "release identity drift")
    subprocess.run(["sha256sum", "-c", sidecar.name], cwd=RELEASE.parent,
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["sha256sum", "-c", outer.name], cwd=RELEASE.parent,
                   check=True, stdout=subprocess.DEVNULL)
    release = json.loads(RELEASE.read_text())
    require(release["status"] == "PASS_M1019_M1018_M1001_C2_SAIF_LAUNCH_RELEASE_R3" and
            release["launch_now"] is True and release["max_attempts"] == 1 and
            release["runner_sha256"] == EXPECTED["runner"], "release content drift")
    require(release["execution"]["axes"] == ["k1", "k8", "k1x8"] and
            release["execution"]["cases_per_axis"] == 5 and
            release["execution"]["total_gate_simulations"] == 15 and
            release["execution"]["fresh_compile_per_axis"] is True and
            release["execution"]["dut_only_saif"] is True, "execution geometry drift")
    require(release["independent_hammer"]["required_status"] ==
            "PASS_M1020_M1019_M1018_M1022_C2_SAIF_RELEASE_HAMMER" and
            release["independent_hammer"]["authored_by_release_author"] is False and
            release["independent_hammer"]["present_at_release_authoring"] is False,
            "independent hammer boundary drift")
    require(all(release["authorization"][key] is False for key in
                ("automatic_retry", "m1013_retry", "pt", "ptpx", "dc", "gpu_remote")),
            "authorization drift")
    verify_seal(AUDIT, EXPECTED["audit_review"], EXPECTED["audit_manifest"], EXPECTED["audit_outer"])
    audit = json.loads((AUDIT / "review.json").read_text())
    require(audit["failure_boundary"]["gate_simulations_completed"] == 0 and
            audit["failure_boundary"]["saif_files_created"] == 0 and
            audit["failure_boundary"]["m1013_retry_authorized"] is False,
            "failure boundary drift")
    require(not M1020.exists(), "M1020 must be independently authored, not pre-created")
    require(not (RESULTS / ".m1022_m1001_c2_three_axis_mapped_gate_saif_attempt_consumed").exists() and
            not (RESULTS / "m1022_m1001_c2_three_axis_mapped_gate_saif_r3_20260829").exists(),
            "M1022 namespace already consumed")
    return {
        "status": "PASS_M1021_M1018_M1019_M1022_ENVIRONMENT_REPAIR_SOURCE",
        "runner_sha256": sha(RUNNER),
        "release_sha256": sha(RELEASE),
        "audit_outer_sha256": sha(AUDIT / "SHA256SUMS.seal.sha256"),
        "future_m1020_absent": True,
        "m1022_executed": False,
        "eda_runs": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(main(), indent=2, sort_keys=True))
