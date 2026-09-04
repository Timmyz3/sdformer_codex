#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Receipt-blind M1032 static/release hammer after independent tiny VCS."""

import hashlib
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
RUNNER = HW / "dc_handoff/scripts/run_m1033_m1001_c2_mapped_gate_saif_one_shot_r4.sh"
TINY = HW / "dc_handoff/tb/tb_m1030_vcs_license_checkout_preflight.sv"
RELEASE = HW / "contracts/m1031_m1029_m1001_c2_mapped_gate_saif_launch_release_r4_20260829.json"
M1002 = HW / "reviews/m1002_m1001_c2_mapped_gate_saif_rekey_source_hammer_r1_20260829"
M1018 = HW / "reviews/m1018_m1013_c2_saif_compile_failure_audit_r1_20260829"
M1029 = HW / "reviews/m1029_m1022_c2_saif_license_failure_audit_r1_20260829"
M1031 = HW / "reviews/m1031_m1029_m1030_m1033_c2_saif_license_complete_source_receipt_r1_20260829"
M1022_ATTEMPT = HW / "results/.m1022_m1001_c2_three_axis_mapped_gate_saif_attempt_consumed"
M1033_ATTEMPT = HW / "results/.m1033_m1001_c2_three_axis_mapped_gate_saif_attempt_consumed"
M1033_RESULT = HW / "results/m1033_m1001_c2_three_axis_mapped_gate_saif_r4_20260829"
PREFLIGHT = HERE / "tiny_preflight_receipt.json"

EXPECTED = {
    RUNNER: "672bcc595b37a2aa0c3864262b89e342170e3bce33efe9a9c97b317db7847f66",
    TINY: "6569e08194ecc0976e9730c735240fbbe7cc95d330f04be382e10d9283409371",
    RELEASE: "f6a716d7654162832c49a42e6351b9e31536e460da40e28a0eda1abdd64c75c2",
}
SEALS = {
    M1002: "d489e1cc3893e9c2a265ad5d35213e349f6eb44a5b4e2e15189711b1c82f5b85",
    M1018: "5c096458a507e99e922ba6a0658ac7e28bf4f2710a2f49cc971d15c738f32146",
    M1029: "bb3c7c8ddf3d19e73bf9d03a0e63c2ac4efcae6d5861e70f4e052ff9893fecbe",
    M1031: "92e99f7d957f3a27a434c46f0d6f2c9497a9030a4e02cf2bca22fa9fc69b3826",
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(rows):
        output = {}
        for key, value in rows:
            require(key not in output, "duplicate JSON key")
            output[key] = value
        return output
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs)


def verify_seal(directory, expected_outer):
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=directory,
                   stdout=subprocess.DEVNULL, check=True)
    subprocess.run(["sha256sum", "-c", "SHA256SUMS.seal.sha256"],
                   cwd=directory, stdout=subprocess.DEVNULL, check=True)
    require(sha256(directory / "SHA256SUMS.seal.sha256") == expected_outer,
            "outer seal drift: " + directory.name)


def preattempt_order(text):
    consume = text.index('mkdir "${attempt}"')
    checks = {
        "missing_license": text.index('[[ "${license_route_present}" -eq 1 ]]'),
        "tiny_identity": text.index('expect_sha "${tiny_sv}"'),
        "wrong_outer": text.index('verify_seal "${source_hammer}"'),
        "wrong_status": text.index('[[ "$(jq -r'),
        "namespace_collision": text.index('[[ ! -e "${result}"'),
        "preflight_compile_or_simv_failure": text.rindex("run_license_preflight", 0, consume),
        "tool_collision": text.rindex("collision_gate", 0, consume),
    }
    require(all(position < consume for position in checks.values()),
            "one failure gate occurs after attempt consume")
    require(text.index('if [[ "${rc}" -ne 0 || "${simv_created}" -ne 1 ]]') <
            checks["preflight_compile_or_simv_failure"],
            "preflight failure branch missing")
    return {name: "REJECTS_BEFORE_M1033_ATTEMPT" for name in checks}


def main():
    for path, expected in EXPECTED.items():
        require(sha256(path) == expected, "identity drift: " + path.name)
    for directory, expected in SEALS.items():
        verify_seal(directory, expected)
    subprocess.run(["sha256sum", "-c", RELEASE.name + ".sha256"],
                   cwd=RELEASE.parent, stdout=subprocess.DEVNULL, check=True)
    subprocess.run(["sha256sum", "-c", RELEASE.name + ".sha256.seal.sha256"],
                   cwd=RELEASE.parent, stdout=subprocess.DEVNULL, check=True)
    release = strict_json(RELEASE)
    m1002 = strict_json(M1002 / "review.json")
    m1018 = strict_json(M1018 / "review.json")
    m1029 = strict_json(M1029 / "review.json")
    m1031 = strict_json(M1031 / "review.json")
    require(m1002["status"] == "PASS_M1002_M1001_SOURCE_HAMMER",
            "M1002 status drift")
    require(m1018["status"] ==
            "PASS_M1018_M1013_FAILURE_AUDIT__M1013_DO_NOT_RETRY" and
            m1018["failure_boundary"]["m1013_retry_authorized"] is False,
            "M1018 no-retry drift")
    require(m1029["status"] ==
            "PASS_M1029_M1022_FAILURE_AUDIT__M1022_DO_NOT_RETRY" and
            m1029["failure_boundary"]["m1022_retry_authorized"] is False and
            m1029["failure_boundary"]["gate_simulations_completed"] == 0 and
            m1029["failure_boundary"]["saif_files_created"] == 0,
            "M1029/M1022 failure boundary drift")
    require(m1031["status"] ==
            "PASS_M1031_M1029_M1030_M1033_LICENSE_COMPLETE_SOURCE" and
            m1031["score_out_of_100"] == 100,
            "M1031 source receipt drift")
    require(release["status"] ==
            "PASS_M1031_M1029_M1001_C2_SAIF_LAUNCH_RELEASE_R4" and
            release["launch_now"] is True and release["max_attempts"] == 1 and
            release["runner_sha256"] == EXPECTED[RUNNER] and
            release["tiny_sv"]["sha256"] == EXPECTED[TINY],
            "release content drift")
    require(M1022_ATTEMPT.is_dir(), "consumed M1022 attempt absent")
    require(not M1033_ATTEMPT.exists() and not M1033_RESULT.exists(),
            "M1033 namespace collision")
    preflight = strict_json(PREFLIGHT)
    require(preflight["status"] ==
            "PASS_M1032_INDEPENDENT_TINY_FULL64_COMPILE_AND_SIMV" and
            preflight["compile_return_code"] == 0 and
            preflight["simv_return_code"] == 0 and
            preflight["simv_created"] is True and
            preflight["license_route_present"] is True and
            preflight["license_value_recorded"] is False and
            preflight["canonical_m1033_attempt"] is False,
            "independent tiny preflight drift")
    text = RUNNER.read_text(encoding="utf-8")
    for line in text.splitlines():
        if "printf" in line:
            require("LM_LICENSE_FILE" not in line and
                    "SNPSLMD_LICENSE_FILE" not in line,
                    "license value can reach output")
    require("printenv" not in text and "license_value_recorded\":false" in text,
            "license privacy guard drift")
    attacks = preattempt_order(text)
    require(not M1033_ATTEMPT.exists() and not M1033_RESULT.exists(),
            "M1033 appeared during hammer")
    return {
        "status": "PASS_M1032_M1031_M1029_M1030_M1033_C2_SAIF_RELEASE_HAMMER",
        "score": 100,
        "p0": 0, "p1": 0, "p2": 0,
        "identity": {
            "m1030_runner_sha256": sha256(RUNNER),
            "m1030_tiny_sv_sha256": sha256(TINY),
            "m1031_release_sha256": sha256(RELEASE),
            "m1029_outer_seal_file_sha256":
                sha256(M1029 / "SHA256SUMS.seal.sha256"),
            "m1031_source_receipt_outer_seal_file_sha256":
                sha256(M1031 / "SHA256SUMS.seal.sha256"),
        },
        "pre_attempt_faults": attacks,
        "tiny_full64_compile_simv": "PASS",
        "license_route_present": True,
        "license_value_recorded": False,
        "m1022_retry_authorized": False,
        "m1033_attempt_consumed": False,
        "m1033_result_created": False,
        "formal_m1033_run": False,
        "saif_created": False,
        "pt_ptpx_dc_run": False,
        "authorization": "ONE_M1033_VCS_SAIF_ATTEMPT_ONLY",
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
