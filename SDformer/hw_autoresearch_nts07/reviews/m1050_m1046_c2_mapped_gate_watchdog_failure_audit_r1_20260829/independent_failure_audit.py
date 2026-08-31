#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Read-only M1050 audit of the consumed M1046 mapped-gate SAIF failure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
ATTEMPT = HW / "results/.m1046_m1001_c2_three_axis_mapped_gate_saif_attempt_consumed"
PREFLIGHT = HW / "results/m1046_m1001_c2_ucli_power_preflight.2027456.sealed"
QUARANTINE = HW / "results/m1046_m1001_c2_three_axis_mapped_gate_saif_r5_20260829.failed_or_incomplete.2027456.quarantine"
RESULT = HW / "results/m1046_m1001_c2_three_axis_mapped_gate_saif_r5_20260829"
RUNNER = HW / "dc_handoff/scripts/run_m1046_m1001_c2_mapped_gate_saif_one_shot_r5.sh"
TB = HW / "dc_handoff/tb/tb_m979_c2_three_axis_mapped_gate_case_saif.sv"
NETLIST = HW / "dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829/k1/netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v"
M867 = HW / "reviews/m867_m859_c2_r25_shared_whitelist_vcs_result_hammer_r1_20260829"
M1045 = HW / "reviews/m1045_m1044_m1043_m1046_c2_saif_release_hammer_r1_20260829"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
DIAG = HERE / "diagnostic_summary.json"

EXPECTED_SHA = {
    RUNNER: "381afcf82fd8a95a2966320cc2fc25d7965d5d0e74f060c8ab6aaef8027e4856",
    TB: "cce12a93c4c8fd8d424fbf9f6354ba30e2870a05a7480fc7de26b3b29c87266c",
    NETLIST: "060e7cd00e5a0f79860430c823439424ae88211cd2ff0d71bc787c9e6691d6b3",
    QUARANTINE / "k1/case0.log": "d129e34047ee882435e06b55ca37c1ae843a03662b8f86edad747251409c765a",
    QUARANTINE / "k1/compile.log": "15d9f5edbbc394f0f03182f824ab45162c0ea36f1800ed57aa326a88ca45b1ea",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
OUTERS = {
    ATTEMPT: "e0d70c4a39ca3efd79893b354d5551bd402ad81608ae3e31d9ceba11eae7dd13",
    PREFLIGHT: "f9bac1e8638e3b82e4aed19f7fec8405b292d077aee04197c6c60453a508bdb7",
    QUARANTINE: "cb6f6b69e2cb51d60556f5bcb8a7748865f72ee2bdbe2f178925a624d9e9d705",
    M867: "1c6e1999b9a23fd8db9c025011b83e4d532f86a42deb30adaef3931e11ed0041",
    M1045: "7c1dcdb02f1c259e3150b56ba995b397e0f65917b779f4f85b0a756b66c6011c",
}


def require(value: bool, message: str) -> None:
    if not value:
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
            "sealed directory absent/symlink: " + directory.name)
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=directory,
                   stdout=subprocess.DEVNULL, check=True)
    subprocess.run(["sha256sum", "-c", "SHA256SUMS.seal.sha256"], cwd=directory,
                   stdout=subprocess.DEVNULL, check=True)
    require(sha(directory / "SHA256SUMS.seal.sha256") == expected_outer,
            "outer seal drift: " + directory.name)


def main() -> dict:
    for path, expected in EXPECTED_SHA.items():
        require(sha(path) == expected, "identity drift: " + path.name)
    for directory, expected in OUTERS.items():
        verify_seal(directory, expected)
    require(not RESULT.exists(), "canonical M1046 result must be absent")

    failure = strict_json(QUARANTINE / "failure.json")
    attempt = strict_json(ATTEMPT / "attempt.json")
    preflight = strict_json(PREFLIGHT / "preflight.json")
    m867 = strict_json(M867 / "review.json")
    diag = strict_json(DIAG)
    case0 = (QUARANTINE / "k1/case0.log").read_text(encoding="utf-8")
    compile_log = (QUARANTINE / "k1/compile.log").read_text(encoding="utf-8")
    tb_text = TB.read_text(encoding="utf-8")

    require(failure == {"status": "FAILED_OR_INCOMPLETE",
                        "phase": "RUN_k1_CASE0", "return_code": 1},
            "M1046 failure boundary drift")
    require(attempt["status"] == "M1046_ATTEMPT_CONSUMED" and
            attempt["ucli_power_preflight_passed"] is True,
            "attempt boundary drift")
    require(preflight["status"] == "PASS_M1044_TINY_UCLI_POWER_SAIF_PREFLIGHT" and
            preflight["power_enable_disable_report_executed"] is True and
            preflight["saif_nonempty"] is True,
            "tiny UCLI preflight drift")
    require("simv up to date" in compile_log and
            re.search(r"CPU time: [0-9.]+ seconds to compile", compile_log),
            "mapped K1 compile/link completion absent")
    require("M979_SAIF_WINDOW_START axis=K1 case=0 edge=3" in case0 and
            "at time 300015000 ps" in case0 and "M979 watchdog" in case0,
            "case0 watchdog signature drift")
    require("PASS M979 mapped replay" not in case0,
            "failed case unexpectedly contains PASS")
    require(len(list(QUARANTINE.rglob("case*.saif"))) == 0 and
            len(list(QUARANTINE.rglob("case*.saif_check.json"))) == 0,
            "completed production SAIF/case-check unexpectedly present")
    require(m867["vcs_evidence"]["equal_bandwidth"]["clean_cases"] == 10 and
            set(m867["vcs_evidence"]["equal_bandwidth"]["exact_cycles"]) ==
            {"k8", "k1x8"}, "M867 scope drift")
    require("return -1;" in tb_text and
            "if(axis==1)" in tb_text and "if(axis==2)" in tb_text,
            "K1 missing-cycle-anchor audit drift")

    patterns = diag["compile_time_initreg_diagnostic"]["patterns"]
    require(set(patterns) == {"all_zero", "all_one", "random_seed_1",
                              "random_seed_7", "random_seed_29"},
            "diagnostic initialization population drift")
    for row in patterns.values():
        require(row["pass"] is True and row["cycles"] == 259 and
                row["events"] == 20 and row["mismatches"] == 0,
                "initialization diagnostic did not converge identically")
    require(diag["without_ucli"]["watchdog_reproduced"] is True and
            diag["signal_probe"]["first_x_window_ns"] == [25, 28] and
            diag["signal_probe"]["requests_before_x"] == 0,
            "non-UCLI/X-propagation diagnostic drift")
    return {
        "status": "PASS_M1050_M1046_WATCHDOG_FAILURE_AUDIT__M1046_DO_NOT_RETRY",
        "score": 100, "p0": 0, "p1": 0, "p2": 0,
        "root_cause": "mapped K1 uninitialized-state X propagation after first raw accept",
        "excluded": ["license", "mapped_compile", "plusarg", "UCLI_stop_resume",
                     "SAIF_command", "long_latency", "raw_protocol"],
        "completed_gate_cases": 0,
        "production_saif_files": 0,
        "m1046_retry_authorized": False,
        "diagnostic_runs_only": 7,
        "diagnostic_saif_created": False,
        "docs359_sha256": sha(DOC359),
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
