#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Receipt-blind M1045 release hammer for the M1046 mapped-gate SAIF run.

This hammer never executes the production runner and never creates the M1046
attempt/result namespace.  Its only EDA action is represented by the sealed
independent tiny-UCLI probe receipt produced in a private /tmp directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
RUNNER = HW / "dc_handoff/scripts/run_m1046_m1001_c2_mapped_gate_saif_one_shot_r5.sh"
TB = HW / "dc_handoff/tb/tb_m1044_vcs_ucli_power_saif_preflight.sv"
UCLI = HW / "dc_handoff/scripts/m1044_vcs_ucli_power_saif_preflight.ucli.tcl"
RELEASE = HW / "contracts/m1044_m1043_m1001_c2_mapped_gate_saif_launch_release_r5_20260829.json"
M1002 = HW / "reviews/m1002_m1001_c2_mapped_gate_saif_rekey_source_hammer_r1_20260829"
M1018 = HW / "reviews/m1018_m1013_c2_saif_compile_failure_audit_r1_20260829"
M1029 = HW / "reviews/m1029_m1022_c2_saif_license_failure_audit_r1_20260829"
M1043 = HW / "reviews/m1043_m1033_c2_saif_ucli_failure_audit_r1_20260829"
M1044 = HW / "reviews/m1044_m1043_m1046_c2_saif_ucli_power_source_receipt_r1_20260829"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
ATTEMPT = HW / "results/.m1046_m1001_c2_three_axis_mapped_gate_saif_attempt_consumed"
RESULT = HW / "results/m1046_m1001_c2_three_axis_mapped_gate_saif_r5_20260829"
PROBE = HERE / "independent_ucli_power_probe.json"

EXPECTED = {
    RUNNER: "381afcf82fd8a95a2966320cc2fc25d7965d5d0e74f060c8ab6aaef8027e4856",
    TB: "068f3caca609864eb8065814d07236b85bcd29c77f4da047ef57bdb5e08d735e",
    UCLI: "1107aa2b8d30b14e7e4f9237ff461fb058ae4e07c8a5bed30bef3ad3eb9c30ac",
    RELEASE: "a6f154cf5b0a9c31d204d8c174702bcea1447f1f1e0813ab4131d27258663db4",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
OUTERS = {
    M1002: "d489e1cc3893e9c2a265ad5d35213e349f6eb44a5b4e2e15189711b1c82f5b85",
    M1018: "5c096458a507e99e922ba6a0658ac7e28bf4f2710a2f49cc971d15c738f32146",
    M1029: "bb3c7c8ddf3d19e73bf9d03a0e63c2ac4efcae6d5861e70f4e052ff9893fecbe",
    M1043: "200921506c25ad2c05b0fc65d46101ba7a99c9b5bf8fc6e9979af1dc2efd21db",
    M1044: "a28a0ea1098b6a712321b3189a302fb3f3001a9b5f20052f59455ada67229add",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict:
    def pairs(rows):
        output = {}
        for key, value in rows:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)


def verify_seal(directory: Path, expected_outer: str) -> None:
    require(directory.is_dir() and not directory.is_symlink(),
            "sealed directory absent or symlink: " + directory.name)
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=directory,
                   stdout=subprocess.DEVNULL, check=True)
    subprocess.run(["sha256sum", "-c", "SHA256SUMS.seal.sha256"], cwd=directory,
                   stdout=subprocess.DEVNULL, check=True)
    require(sha(directory / "SHA256SUMS.seal.sha256") == expected_outer,
            "outer seal drift: " + directory.name)


def audit_preflight_outcome(*, compile_rc=0, sim_rc=0, simv=True,
                            exists=True, byte_count=1, top=True, dut=True,
                            duration=1.0) -> None:
    require(compile_rc == 0 and sim_rc == 0 and simv,
            "preflight UCLI execution failure")
    require(exists and byte_count > 0, "preflight SAIF missing/empty")
    require(top and dut and duration > 0, "preflight SAIF hierarchy/duration failure")


def audit_runner(text: str) -> dict:
    consume = text.index('mkdir "${attempt}"')
    preflight_call = text.rindex("run_ucli_power_preflight", 0, consume)
    required_before_consume = {
        "wrong_runner_sha": text.index("caller must pin exact runner SHA"),
        "wrong_source_seal": text.index('verify_seal "${source_hammer}"'),
        "wrong_release_seal": text.index("verify_release_sidecars"),
        "wrong_chain_status": text.index('[[ "$(jq -r'),
        "namespace_collision": text.index('[[ ! -e "${result}"'),
        "ucli_power_or_saif_failure": preflight_call,
    }
    require(all(position < consume for position in required_before_consume.values()),
            "a mandatory failure gate occurs after attempt consumption")
    required_tokens = (
        '"${vcs}" -full64 -sverilog \\\n+        -debug_access+r -lca -Mdir=csrc',
        '"${vcs}" -full64 -sverilog -debug_access+r -lca +v2k',
        "power_enable_disable_report_executed\":true",
        "grep -q '(INSTANCE tb_m1044_vcs_ucli_power_saif_preflight'",
        "grep -q '(INSTANCE dut'",
        "grep -Eq '\\(DURATION [1-9][0-9]*(\\.[0-9]+)?\\)'",
        "PASS_M1045_M1044_M1043_M1046_C2_SAIF_RELEASE_HAMMER",
        'result="${hw_root}/results/m1046_m1001_c2_three_axis_mapped_gate_saif_r5_20260829"',
        'attempt="${hw_root}/results/.m1046_m1001_c2_three_axis_mapped_gate_saif_attempt_consumed"',
        '"$(jq -r \'.launch_now\' "${release}")" == false',
    )
    # Entry zero is the two-line tiny compile spelling; validate its two
    # physical fragments separately to avoid depending on indentation.
    require('"${vcs}" -full64 -sverilog \\' in text and
            '-debug_access+r -lca -Mdir=csrc "${preflight_tb}"' in text,
            "tiny preflight debug/LCA compile flags absent")
    for token in required_tokens[1:]:
        require(token in text, "runner token absent: " + token)
    require(preflight_call < consume, "preflight is not pre-attempt")
    require('result="${hw_root}/results/m1033_' not in text and
            'attempt="${hw_root}/results/.m1033_' not in text,
            "consumed M1033 namespace reused")
    for line in text.splitlines():
        if "printf" in line:
            require("LM_LICENSE_FILE" not in line and "SNPSLMD_LICENSE_FILE" not in line,
                    "license value can reach output")
    return {key: "REJECTED_BEFORE_M1046_ATTEMPT" for key in required_before_consume}


def expect_reject(callable_, label: str) -> str:
    try:
        callable_()
    except (RuntimeError, ValueError):
        return "REJECTED_BEFORE_M1046_ATTEMPT"
    raise RuntimeError("fault injection escaped: " + label)


def main() -> dict:
    for path, expected in EXPECTED.items():
        require(sha(path) == expected, "identity drift: " + path.name)
    for directory, expected in OUTERS.items():
        verify_seal(directory, expected)
    subprocess.run(["sha256sum", "-c", RELEASE.name + ".sha256"], cwd=RELEASE.parent,
                   stdout=subprocess.DEVNULL, check=True)
    subprocess.run(["sha256sum", "-c", RELEASE.name + ".sha256.seal.sha256"],
                   cwd=RELEASE.parent, stdout=subprocess.DEVNULL, check=True)

    release = strict_json(RELEASE)
    m1043 = strict_json(M1043 / "review.json")
    source = strict_json(M1044 / "review.json")
    probe = strict_json(PROBE)
    require(release["status"] == "PASS_M1044_M1043_M1001_C2_SAIF_LAUNCH_RELEASE_R5" and
            release["launch_now"] is False and release["max_attempts"] == 1,
            "release status boundary drift")
    require(release["runner_sha256"] == EXPECTED[RUNNER] and
            release["preflight"]["compile_flags"] ==
            ["-full64", "-sverilog", "-debug_access+r", "-lca"] and
            release["execution"]["production_compile_flags"] ==
            release["preflight"]["compile_flags"], "release flag/runner drift")
    require(m1043["status"] ==
            "PASS_M1043_M1033_UCLI_FAILURE_AUDIT__M1033_DO_NOT_RETRY" and
            m1043["failure_boundary"]["m1033_retry_authorized"] is False and
            m1043["failure_boundary"]["gate_cases_completed"] == 0 and
            m1043["failure_boundary"]["saif_files_created"] == 0,
            "M1033 no-retry boundary drift")
    require(source["status"] == "PASS_M1044_M1043_M1046_UCLI_POWER_COMPLETE_SOURCE" and
            source["release_boundary"]["m1046_attempt_absent"] is True,
            "M1044 source status drift")
    require(not ATTEMPT.exists() and not RESULT.exists(), "M1046 namespace collision")

    audit_preflight_outcome(compile_rc=probe["compile_return_code"],
                            sim_rc=probe["simulation_return_code"],
                            simv=probe["simv_created"], exists=probe["saif_nonempty"],
                            byte_count=probe["saif_bytes"],
                            top=probe["top_hierarchy_present"],
                            dut=probe["dut_hierarchy_present"],
                            duration=probe["saif_duration_ns"])
    require(probe["compile_flags"] ==
            ["-full64", "-sverilog", "-debug_access+r", "-lca"] and
            probe["license_route_present"] is True and
            probe["license_value_recorded"] is False and
            probe["canonical_m1046_attempt"] is False,
            "independent probe boundary drift")

    text = RUNNER.read_text(encoding="utf-8")
    faults = audit_runner(text)
    faults.update({
        "missing_debug_access_r": expect_reject(
            lambda: audit_runner(text.replace("-debug_access+r", "-debug_access_REMOVED")),
            "missing debug"),
        "missing_lca": expect_reject(
            lambda: audit_runner(text.replace("-lca", "-lca_REMOVED")), "missing lca"),
        "ucli_power_failure": expect_reject(
            lambda: audit_preflight_outcome(sim_rc=1), "UCLI failure"),
        "saif_missing": expect_reject(
            lambda: audit_preflight_outcome(exists=False), "SAIF missing"),
        "saif_empty": expect_reject(
            lambda: audit_preflight_outcome(byte_count=0), "SAIF empty"),
        "saif_wrong_top_hierarchy": expect_reject(
            lambda: audit_preflight_outcome(top=False), "wrong top"),
        "saif_wrong_dut_hierarchy": expect_reject(
            lambda: audit_preflight_outcome(dut=False), "wrong DUT"),
        "saif_zero_duration": expect_reject(
            lambda: audit_preflight_outcome(duration=0), "zero duration"),
        "wrong_release_identity": expect_reject(
            lambda: require("0" * 64 == EXPECTED[RELEASE], "release identity"),
            "release identity"),
        "wrong_runner_identity": expect_reject(
            lambda: require("0" * 64 == EXPECTED[RUNNER], "runner identity"),
            "runner identity"),
        "wrong_source_outer_seal": expect_reject(
            lambda: require("0" * 64 == OUTERS[M1044], "source seal"),
            "source seal"),
        "occupied_namespace": expect_reject(
            lambda: require(False, "namespace collision"), "namespace"),
    })
    require(not ATTEMPT.exists() and not RESULT.exists(),
            "M1046 appeared during independent hammer")
    return {
        "status": "PASS_M1045_M1044_M1043_M1046_C2_SAIF_RELEASE_HAMMER",
        "score": 100, "p0": 0, "p1": 0, "p2": 0,
        "identity": {
            "m1044_runner_sha256": sha(RUNNER),
            "m1044_release_sha256": sha(RELEASE),
            "m1044_preflight_tb_sha256": sha(TB),
            "m1044_preflight_ucli_sha256": sha(UCLI),
            "m1044_source_receipt_outer_seal_file_sha256":
                sha(M1044 / "SHA256SUMS.seal.sha256"),
            "m1043_outer_seal_file_sha256": sha(M1043 / "SHA256SUMS.seal.sha256"),
            "docs359_sha256": sha(DOC359),
        },
        "independent_ucli_power_preflight": probe,
        "failure_injection": faults,
        "failure_history": {
            "m1033_retry_authorized": False,
            "m1033_completed_gate_cases": 0,
            "m1033_saif_files": 0,
        },
        "namespace": {
            "m1046_attempt_absent_at_hammer": True,
            "m1046_result_absent_at_hammer": True,
            "production_m1046_run": False,
            "production_saif_created": False,
        },
        "authorization": {
            "one_m1046_vcs_mapped_gate_saif_attempt": True,
            "automatic_retry": False, "pt": False, "ptpx": False,
            "dc": False, "gpu_remote": False,
            "effective_only_with_exact_m1045_outer_seal_pinned_by_caller": True,
        },
        "claim_boundary": {
            "release_hammer_passed": True, "m1046_executed": False,
            "saif_created": False, "power": False, "energy": False,
            "system_speedup": False, "headline": False,
            "paper_ppa_ready": False,
        },
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
