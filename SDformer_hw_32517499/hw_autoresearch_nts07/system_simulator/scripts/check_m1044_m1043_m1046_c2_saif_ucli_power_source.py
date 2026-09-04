#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Static and fault-injection gate for the M1044/M1046 UCLI-power repair.

This checker never launches VCS or consumes the M1046 namespace.  It proves
that the tiny preflight and all three production axes use the same debug/LCA
flags, and models the fail-closed pre-attempt boundary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Mapping


HW = Path(__file__).resolve().parents[2]
RUNNER = HW / "dc_handoff/scripts/run_m1046_m1001_c2_mapped_gate_saif_one_shot_r5.sh"
PREFLIGHT_TB = HW / "dc_handoff/tb/tb_m1044_vcs_ucli_power_saif_preflight.sv"
PREFLIGHT_UCLI = HW / "dc_handoff/scripts/m1044_vcs_ucli_power_saif_preflight.ucli.tcl"
RELEASE = HW / "contracts/m1044_m1043_m1001_c2_mapped_gate_saif_launch_release_r5_20260829.json"
M1002 = HW / "reviews/m1002_m1001_c2_mapped_gate_saif_rekey_source_hammer_r1_20260829"
M1018 = HW / "reviews/m1018_m1013_c2_saif_compile_failure_audit_r1_20260829"
M1029 = HW / "reviews/m1029_m1022_c2_saif_license_failure_audit_r1_20260829"
M1043 = HW / "reviews/m1043_m1033_c2_saif_ucli_failure_audit_r1_20260829"
M1045 = HW / "reviews/m1045_m1044_m1043_m1046_c2_saif_release_hammer_r1_20260829"
M1046_ATTEMPT = HW / "results/.m1046_m1001_c2_three_axis_mapped_gate_saif_attempt_consumed"
M1046_RESULT = HW / "results/m1046_m1001_c2_three_axis_mapped_gate_saif_r5_20260829"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "runner": "381afcf82fd8a95a2966320cc2fc25d7965d5d0e74f060c8ab6aaef8027e4856",
    "preflight_tb": "068f3caca609864eb8065814d07236b85bcd29c77f4da047ef57bdb5e08d735e",
    "preflight_ucli": "1107aa2b8d30b14e7e4f9237ff461fb058ae4e07c8a5bed30bef3ad3eb9c30ac",
    "release": "a6f154cf5b0a9c31d204d8c174702bcea1447f1f1e0813ab4131d27258663db4",
    "release_sidecar": "b95ec79b5c839f96f56dc4ce4c7f8f3238eeb6012ab24ab84794ce0f379ffc6e",
    "m1002_outer": "d489e1cc3893e9c2a265ad5d35213e349f6eb44a5b4e2e15189711b1c82f5b85",
    "m1018_outer": "5c096458a507e99e922ba6a0658ac7e28bf4f2710a2f49cc971d15c738f32146",
    "m1029_outer": "bb3c7c8ddf3d19e73bf9d03a0e63c2ac4efcae6d5861e70f4e052ff9893fecbe",
    "m1043_review": "2f1928c50adfc329987da202ed27291d411d80edb8a9598173e2ae57eb2499fb",
    "m1043_manifest": "e51bdc56936007e42a0ae89464943ed5498026317d991949023621956f1aa27f",
    "m1043_outer": "200921506c25ad2c05b0fc65d46101ba7a99c9b5bf8fc6e9979af1dc2efd21db",
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
    require(directory.is_dir(), "sealed directory absent")
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=directory,
                   stdout=subprocess.DEVNULL, check=True)
    subprocess.run(["sha256sum", "-c", "SHA256SUMS.seal.sha256"], cwd=directory,
                   stdout=subprocess.DEVNULL, check=True)
    require(sha(directory / "SHA256SUMS.seal.sha256") == expected_outer,
            "outer seal drift")


def audit_license_environment(environment: Mapping[str, str]) -> dict:
    present = bool(str(environment.get("LM_LICENSE_FILE", ""))) or bool(
        str(environment.get("SNPSLMD_LICENSE_FILE", "")))
    require(present, "nonempty license route required")
    return {"license_route_present": True, "license_value_recorded": False}


def audit_namespace(paths: Mapping[str, bool]) -> None:
    require(not any(bool(value) for value in paths.values()), "namespace collision")


def audit_preflight_outcome(*, compile_rc: int = 0, sim_rc: int = 0,
                            simv_created: bool = True, saif_exists: bool = True,
                            saif_bytes: int = 1, top_hierarchy: bool = True,
                            dut_hierarchy: bool = True, duration_ns: int = 1) -> dict:
    require(compile_rc == 0 and sim_rc == 0 and simv_created,
            "UCLI power preflight execution failure")
    require(saif_exists and saif_bytes > 0, "preflight SAIF missing or empty")
    require(top_hierarchy and dut_hierarchy and duration_ns > 0,
            "preflight SAIF hierarchy/duration failure")
    return {"preflight_passed": True, "attempt_consumed": False}


def audit_runner_text(text: str) -> None:
    required = (
        'if [[ -n "${LM_LICENSE_FILE:-}" || -n "${SNPSLMD_LICENSE_FILE:-}" ]]',
        'expect_sha "${preflight_tb}" 068f3caca609864eb8065814d07236b85bcd29c77f4da047ef57bdb5e08d735e',
        'expect_sha "${preflight_ucli}" 1107aa2b8d30b14e7e4f9237ff461fb058ae4e07c8a5bed30bef3ad3eb9c30ac',
        '-debug_access+r -lca -Mdir=csrc',
        'power_enable_disable_report_executed":true',
        "grep -q '(INSTANCE dut'",
        'phase=UCLI_POWER_SAIF_PREFLIGHT\nrun_ucli_power_preflight',
        'phase=ATTEMPT_ATOMIC_CONSUME',
        'mkdir "${attempt}"',
        'for axis in k1 k8 k1x8; do',
        'for case_id in 0 1 2 3 4; do',
        '"${vcs}" -full64 -sverilog -debug_access+r -lca +v2k',
        'PASS_M1045_M1044_M1043_M1046_C2_SAIF_RELEASE_HAMMER',
        'one_m1046_vcs_mapped_gate_saif_attempt',
        'result="${hw_root}/results/m1046_m1001_c2_three_axis_mapped_gate_saif_r5_20260829"',
        'attempt="${hw_root}/results/.m1046_m1001_c2_three_axis_mapped_gate_saif_attempt_consumed"',
    )
    for token in required:
        require(token in text, "runner token absent: " + token)
    call = text.rindex("run_ucli_power_preflight")
    consume = text.index('mkdir "${attempt}"')
    require(call < consume, "preflight is not before attempt consumption")
    require(text.count('"${vcs}" -full64 -sverilog -debug_access+r -lca') >= 1,
            "production debug/LCA flags absent")
    require('"$(jq -r \'.launch_now\' "${release}")" == false' in text,
            "release must remain launch_now=false")
    require('result="${hw_root}/results/m1033_' not in text and
            'attempt="${hw_root}/results/.m1033_' not in text,
            "consumed M1033 namespace reused")
    for line in text.splitlines():
        if "printf" in line:
            require("LM_LICENSE_FILE" not in line and "SNPSLMD_LICENSE_FILE" not in line,
                    "license value can reach output")


def main() -> dict:
    require(sha(RUNNER) == EXPECTED["runner"], "runner identity drift")
    require(sha(PREFLIGHT_TB) == EXPECTED["preflight_tb"], "preflight TB drift")
    require(sha(PREFLIGHT_UCLI) == EXPECTED["preflight_ucli"], "preflight UCLI drift")
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
    verify_flat(M1043, EXPECTED["m1043_outer"])
    require(sha(M1043 / "review.json") == EXPECTED["m1043_review"] and
            sha(M1043 / "SHA256SUMS") == EXPECTED["m1043_manifest"],
            "M1043 identity drift")
    audit = json.loads((M1043 / "review.json").read_text(encoding="utf-8"))
    require(audit["failure_boundary"]["m1033_retry_authorized"] is False,
            "M1033 retry boundary drift")
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    require(release["status"] == "PASS_M1044_M1043_M1001_C2_SAIF_LAUNCH_RELEASE_R5" and
            release["launch_now"] is False and release["max_attempts"] == 1,
            "release status/launch boundary drift")
    require(release["preflight"]["compile_flags"] ==
            ["-full64", "-sverilog", "-debug_access+r", "-lca"] and
            release["execution"]["production_compile_flags"] ==
            release["preflight"]["compile_flags"], "preflight/production flag mismatch")
    require(release["execution"]["axes"] == ["k1", "k8", "k1x8"] and
            release["execution"]["total_gate_simulations"] == 15 and
            release["execution"]["dut_only_saif"] is True,
            "production geometry drift")
    require(release["independent_hammer"]["required_status"] ==
            "PASS_M1045_M1044_M1043_M1046_C2_SAIF_RELEASE_HAMMER" and
            release["independent_hammer"]["present_at_release_authoring"] is False,
            "independent hammer boundary drift")
    audit_runner_text(RUNNER.read_text(encoding="utf-8"))
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    require(not M1045.exists(), "M1045 must be independently authored")
    audit_namespace({"attempt": M1046_ATTEMPT.exists(), "result": M1046_RESULT.exists()})
    require(sha(DOC359) == EXPECTED["docs359"], "docs/359 drift")
    return {
        "status": "PASS_M1044_M1043_M1046_UCLI_POWER_COMPLETE_SOURCE",
        "runner_sha256": sha(RUNNER),
        "release_sha256": sha(RELEASE),
        "preflight_tb_sha256": sha(PREFLIGHT_TB),
        "preflight_ucli_sha256": sha(PREFLIGHT_UCLI),
        "future_m1045_absent": True,
        "m1046_executed": False,
        "license_value_recorded": False,
        "eda_runs": 0,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
