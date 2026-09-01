#!/usr/bin/env python3
from __future__ import print_function

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess


HW = Path(__file__).resolve().parents[2]
OLD = HW / "dc_handoff/scripts/run_vcs_m1613_c2_m1609_registered_fault_directed_exact_sha_r1.sh"
NEW = HW / "dc_handoff/scripts/run_vcs_m1621_m1613_c2_m1609_registered_fault_directed_exact_sha_r1.sh"
CONTRACT = HW / "contracts/m1621_m1613_python36_regular_path_runner_successor_source_contract_r1_20260901.json"
RESULT = HW / "results/m1613_c2_m1609_registered_fault_directed_vcs_r1_20260901"
ATTEMPT = HW / "results/.m1613_c2_m1609_registered_fault_directed_vcs_attempt_consumed"
M1622 = HW / "reviews/m1622_m1621_m1613_c2_registered_fault_directed_runner_source_hammer_r1_20260901"
M1623 = HW / "contracts/m1623_m1622_m1621_m1613_c2_registered_fault_directed_vcs_launch_release_r1_20260901.json"
OLD_SHA = "f2b3888879cb5a6af4396eb8b4971510453a47622299e17dd6702925587c0b29"
NEW_SHA = "11da68ff4eb9da70c83b56ae7dd2dbff26f125833224beb08f165fe97a0ea30b"
PYTHON_SHA = "9c9502e21917eff03ffe4672c4e61cf8ce651aabeaf5118e423782feba58787f"


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def absent(path):
    path = Path(path)
    return not path.exists() and not path.is_symlink()


def normalized_new():
    value = NEW.read_text(encoding="utf-8")
    substitutions = (
        ("# M1621 additive M1613 one-shot directed VCS runner source.",
         "# M1613 one-shot directed VCS runner source."),
        ("python36=/usr/libexec/platform-python3.6",
         "python36=/usr/bin/python3.6"),
        ("reviews/m1622_m1621_m1613_c2_registered_fault_directed_runner_source_hammer_r1_20260901",
         "reviews/m1617_m1613_c2_m1609_registered_fault_directed_source_hammer_r1_20260901"),
        ("contracts/m1623_m1622_m1621_m1613_c2_registered_fault_directed_vcs_launch_release_r1_20260901.json",
         "contracts/m1618_m1617_m1613_c2_registered_fault_directed_vcs_launch_release_r1_20260901.json"),
        ("M1621_EXPECTED_RUNNER_SHA256", "M1613_EXPECTED_RUNNER_SHA256"),
        ("PASS_M1622_M1621_M1613_C2_REGISTERED_FAULT_RUNNER_HAMMER__AUTHORIZE_ONE_FUTURE_VCS_ATTEMPT",
         "PASS_M1617_M1613_C2_REGISTERED_FAULT_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_VCS_ATTEMPT"),
        ("AUTHORIZE_ONE_M1621_M1613_C2_REGISTERED_FAULT_DIRECTED_VCS_ATTEMPT",
         "AUTHORIZE_ONE_M1613_C2_REGISTERED_FAULT_DIRECTED_VCS_ATTEMPT"),
        ("M1621_EXPECTED_RELEASE_SHA256", "M1613_EXPECTED_RELEASE_SHA256"),
    )
    for before, after in substitutions:
        require(before in value, "successor substitution anchor missing")
        value = value.replace(before, after)
    return value


def run_pre_attempt(runner, environment_name, runner_sha):
    environment = dict(os.environ)
    environment[environment_name] = runner_sha
    process = subprocess.run([str(runner)], cwd=str(HW.parent), env=environment,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True, timeout=60)
    require(process.returncode == 3, "pre-attempt runner did not fail closed")
    require(absent(RESULT) and absent(ATTEMPT),
            "pre-attempt dry check consumed namespace")
    return {"returncode": process.returncode, "stdout": process.stdout,
            "stderr": process.stderr}


def main():
    require(sha256(OLD) == OLD_SHA and sha256(NEW) == NEW_SHA,
            "runner SHA drift")
    require(stat.S_ISLNK(Path("/usr/bin/python3.6").lstat().st_mode),
            "predecessor Python path is no longer a symlink")
    mode = Path("/usr/libexec/platform-python3.6").lstat().st_mode
    require(stat.S_ISREG(mode) and not Path(
        "/usr/libexec/platform-python3.6").is_symlink(),
        "successor Python path is not a regular non-symlink")
    require(sha256("/usr/bin/python3.6") == PYTHON_SHA and
            sha256("/usr/libexec/platform-python3.6") == PYTHON_SHA,
            "resolved Python content differs")
    require(normalized_new() == OLD.read_text(encoding="utf-8"),
            "successor changes more than the admitted path/control identities")
    require(os.access(str(NEW), os.X_OK), "successor runner is not executable")
    subprocess.check_call(["bash", "-n", str(NEW)])
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    require(contract["status"] ==
            "SOURCE_ONLY__ADDITIVE_RUNNER_PATH_REPAIR__NO_VCS_NO_ATTEMPT" and
            contract["identity"]["successor_runner_sha256"] == NEW_SHA and
            contract["namespace_frozen"]["attempt_exists_at_m1621_authoring"] is False,
            "M1621 contract drift")
    require(absent(RESULT) and absent(ATTEMPT) and absent(M1622) and absent(M1623),
            "authoring/future authority namespace is not fresh")

    old_result = run_pre_attempt(OLD, "M1613_EXPECTED_RUNNER_SHA256", OLD_SHA)
    require("missing/nonregular/SHA mismatch: /usr/bin/python3.6" in
            old_result["stderr"], "predecessor failure signature drift")
    new_result = run_pre_attempt(NEW, "M1621_EXPECTED_RUNNER_SHA256", NEW_SHA)
    require("sealed directory absent/nonregular: " in new_result["stderr"] and
            "m1622_m1621_m1613" in new_result["stderr"] and
            "/usr/libexec/platform-python3.6" not in new_result["stderr"],
            "successor did not pass Python gate and stop at future M1622")
    require("VCS_COMPILE" not in new_result["stdout"] + new_result["stderr"] and
            absent(RESULT) and absent(ATTEMPT),
            "static test reached VCS or consumed attempt")
    print("PASS M1621 additive runner python36_regular=1 same_sha=1 "
          "old_pre_attempt_fail=1 new_future_m1622_stop=1 attempt=0 vcs=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
