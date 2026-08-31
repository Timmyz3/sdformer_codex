#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Static/fault-injection checker for M1029/M1030/M1031.

No M1033 runner or EDA tool is invoked here.  License-routing values are
treated as secrets: only the boolean presence of at least one nonempty route
may leave ``audit_license_environment``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Mapping


HW = Path(__file__).resolve().parents[2]
RUNNER = HW / "dc_handoff/scripts/run_m1033_m1001_c2_mapped_gate_saif_one_shot_r4.sh"
TINY = HW / "dc_handoff/tb/tb_m1030_vcs_license_checkout_preflight.sv"
RELEASE = HW / "contracts/m1031_m1029_m1001_c2_mapped_gate_saif_launch_release_r4_20260829.json"
M1002 = HW / "reviews/m1002_m1001_c2_mapped_gate_saif_rekey_source_hammer_r1_20260829"
M1018 = HW / "reviews/m1018_m1013_c2_saif_compile_failure_audit_r1_20260829"
M1029 = HW / "reviews/m1029_m1022_c2_saif_license_failure_audit_r1_20260829"
M1032 = HW / "reviews/m1032_m1031_m1029_m1030_m1033_c2_saif_release_hammer_r1_20260829"
M1033_ATTEMPT = HW / "results/.m1033_m1001_c2_three_axis_mapped_gate_saif_attempt_consumed"
M1033_RESULT = HW / "results/m1033_m1001_c2_three_axis_mapped_gate_saif_r4_20260829"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "runner": "672bcc595b37a2aa0c3864262b89e342170e3bce33efe9a9c97b317db7847f66",
    "tiny": "6569e08194ecc0976e9730c735240fbbe7cc95d330f04be382e10d9283409371",
    "release": "f6a716d7654162832c49a42e6351b9e31536e460da40e28a0eda1abdd64c75c2",
    "release_sidecar": "1a3aabd8d377f76acfbf1edebf56163e514f2a8a3a2124aaba2205978ea97ba6",
    "m1002_outer": "d489e1cc3893e9c2a265ad5d35213e349f6eb44a5b4e2e15189711b1c82f5b85",
    "m1018_outer": "5c096458a507e99e922ba6a0658ac7e28bf4f2710a2f49cc971d15c738f32146",
    "m1029_review": "d431da2d76bb970361d2d57d4afaca4ea54b5e21888720445e2f05bb172dc9ee",
    "m1029_manifest": "23612b4326234fa22276bc300fc2999e5003a9c18714bbbef1815e45ad52a1c0",
    "m1029_outer": "bb3c7c8ddf3d19e73bf9d03a0e63c2ac4efcae6d5861e70f4e052ff9893fecbe",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
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


def verify_flat(directory: Path, expected_outer: str) -> None:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and manifest.is_file() and outer.is_file(),
            "sealed directory absent")
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=directory,
                   stdout=subprocess.DEVNULL, check=True)
    subprocess.run(["sha256sum", "-c", "SHA256SUMS.seal.sha256"],
                   cwd=directory, stdout=subprocess.DEVNULL, check=True)
    require(sha(outer) == expected_outer, "outer seal drift")


def audit_license_environment(environment: Mapping[str, str]) -> dict:
    present = bool(str(environment.get("LM_LICENSE_FILE", ""))) or bool(
        str(environment.get("SNPSLMD_LICENSE_FILE", "")))
    require(present, "nonempty license route required")
    # Deliberately return no key, value, hash, length, prefix, or endpoint.
    return {"license_route_present": True, "license_value_recorded": False}


def audit_namespace(paths: Mapping[str, bool]) -> None:
    require(not any(bool(value) for value in paths.values()),
            "result/attempt/work collision")


def audit_collision(active_process_names) -> None:
    forbidden = {"vcs1", "vlogan", "dc_shell", "dc_shell-t", "fm_shell", "pt_shell"}
    require(not forbidden.intersection(set(active_process_names)),
            "VCS/DC/FM/PT collision")


def audit_runner_text(text: str) -> None:
    required = (
        'if [[ -n "${LM_LICENSE_FILE:-}" || -n "${SNPSLMD_LICENSE_FILE:-}" ]]',
        'license_route_present=1',
        'nonempty license route required',
        'export VCS_HOME="${expected_vcs_home}"',
        'export PATH="${VCS_HOME}/bin:/usr/bin:/bin"',
        'expect_sha "${vcs_msg_report}" b34e06a92b05856532f868d32c0c81f1708506096856ad9a97bd27e2bd60215b',
        'expect_sha "${tiny_sv}" 6569e08194ecc0976e9730c735240fbbe7cc95d330f04be382e10d9283409371',
        '"${vcs}" -full64 -sverilog',
        '-top tb_m1030_vcs_license_checkout_preflight',
        'run_license_preflight',
        'PASS_M1030_TINY_SV_FULL64_LICENSE_CHECKOUT_PREFLIGHT',
        'license_value_recorded":false',
        'M1033_EXPECTED_M1032_OUTER_SHA256',
        'PASS_M1032_M1031_M1029_M1030_M1033_C2_SAIF_RELEASE_HAMMER',
        'result="${hw_root}/results/m1033_m1001_c2_three_axis_mapped_gate_saif_r4_20260829"',
        'attempt="${hw_root}/results/.m1033_m1001_c2_three_axis_mapped_gate_saif_attempt_consumed"',
        'for axis in k1 k8 k1x8; do',
        'for case_id in 0 1 2 3 4; do',
        'collision_gate',
        'mkdir "${attempt}"',
    )
    for token in required:
        require(token in text, "runner token absent: " + token)
    require(text.index('expect_sha "${tiny_sv}"') < text.rindex('run_license_preflight'),
            "tiny identity is not before preflight")
    # The function definition appears early; the actual call is the final
    # occurrence and must precede attempt consumption.
    require(text.rindex("run_license_preflight") < text.index('mkdir "${attempt}"'),
            "license checkout is not before attempt")
    require(text.rindex("collision_gate") < text.index('mkdir "${attempt}"'),
            "final collision gate is not before attempt")
    require('>/dev/null 2>&1' in text, "preflight compiler output is not suppressed")
    for line in text.splitlines():
        if "printf" in line:
            require("LM_LICENSE_FILE" not in line and "SNPSLMD_LICENSE_FILE" not in line,
                    "license value can reach printf")
    require("M1022_EXPECTED_" not in text and
            'result="${hw_root}/results/m1022_' not in text and
            'attempt="${hw_root}/results/.m1022_' not in text,
            "stale consumed namespace in runner")


def main() -> dict:
    require(sha(RUNNER) == EXPECTED["runner"], "runner identity drift")
    require(sha(TINY) == EXPECTED["tiny"], "tiny SV identity drift")
    require(sha(RELEASE) == EXPECTED["release"], "release identity drift")
    sidecar = Path(str(RELEASE) + ".sha256")
    outer = Path(str(RELEASE) + ".sha256.seal.sha256")
    require(sha(sidecar) == EXPECTED["release_sidecar"], "release sidecar drift")
    subprocess.run(["sha256sum", "-c", sidecar.name], cwd=RELEASE.parent,
                   stdout=subprocess.DEVNULL, check=True)
    subprocess.run(["sha256sum", "-c", outer.name], cwd=RELEASE.parent,
                   stdout=subprocess.DEVNULL, check=True)
    verify_flat(M1002, EXPECTED["m1002_outer"])
    verify_flat(M1018, EXPECTED["m1018_outer"])
    verify_flat(M1029, EXPECTED["m1029_outer"])
    require(sha(M1029 / "review.json") == EXPECTED["m1029_review"] and
            sha(M1029 / "SHA256SUMS") == EXPECTED["m1029_manifest"],
            "M1029 identity drift")
    audit = json.loads((M1029 / "review.json").read_text(encoding="utf-8"))
    require(audit["failure_boundary"]["m1022_retry_authorized"] is False and
            audit["failure_boundary"]["gate_simulations_completed"] == 0 and
            audit["failure_boundary"]["saif_files_created"] == 0,
            "M1029 failure boundary drift")
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    require(release["status"] ==
            "PASS_M1031_M1029_M1001_C2_SAIF_LAUNCH_RELEASE_R4" and
            release["launch_now"] is True and release["max_attempts"] == 1 and
            release["runner_sha256"] == EXPECTED["runner"] and
            release["tiny_sv"]["sha256"] == EXPECTED["tiny"],
            "release content drift")
    require(release["independent_hammer"]["required_status"] ==
            "PASS_M1032_M1031_M1029_M1030_M1033_C2_SAIF_RELEASE_HAMMER" and
            release["independent_hammer"]["present_at_release_authoring"] is False,
            "independent hammer boundary drift")
    require(release["execution"]["axes"] == ["k1", "k8", "k1x8"] and
            release["execution"]["total_gate_simulations"] == 15 and
            release["execution"]["dut_only_saif"] is True,
            "execution geometry drift")
    audit_runner_text(RUNNER.read_text(encoding="utf-8"))
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    require(not M1032.exists(), "M1032 must be independently authored")
    audit_namespace({"attempt": M1033_ATTEMPT.exists(), "result": M1033_RESULT.exists()})
    require(sha(DOC359) == EXPECTED["docs359"], "docs/359 drift")
    return {
        "status": "PASS_M1031_M1029_M1030_M1033_LICENSE_COMPLETE_SOURCE",
        "runner_sha256": sha(RUNNER),
        "tiny_sv_sha256": sha(TINY),
        "release_sha256": sha(RELEASE),
        "m1029_outer_sha256": sha(M1029 / "SHA256SUMS.seal.sha256"),
        "future_m1032_absent": True,
        "m1033_executed": False,
        "license_value_recorded": False,
        "eda_runs": 0,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
