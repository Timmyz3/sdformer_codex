#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent M1039 C1 full-replay release hammer; never runs M1040."""

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
RUNNER = HW / "system_simulator/scripts/run_m1040_m1016_c1_full_matched_address_replay_one_shot.sh"
RELEASE = HW / "contracts/m1038_m1037_m1016_c1_full_matched_address_replay_launch_release_r1_20260829.json"
M1025 = HW / "reviews/m1025_m1016_c1_full_matched_address_replay_source_hammer_r1_20260829"
M1036 = HW / "reviews/m1036_m1026_m1027_m1028_c1_full_replay_cross_hammer_r1_20260829"
M1037 = HW / "reviews/m1037_m1038_m1040_c1_full_replay_collision_resource_source_receipt_r1_20260829"
TEST_SOURCE = HW / "system_simulator/tests/test_m1037_m1038_m1040_c1_full_replay_collision_resource_source.py"
ATTEMPT = HW / "results/.m1040_m1016_c1_full_matched_address_replay_attempt_consumed"
RESULT = HW / "results/m1040_m1016_c1_full_matched_address_replay_r1_20260829"
LOCKFILE = HW / "results/.c1_full_matched_address_replay_global.lock"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_seal(directory, expected):
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=directory,
                   stdout=subprocess.DEVNULL, check=True)
    subprocess.run(["sha256sum", "-c", "SHA256SUMS.seal.sha256"], cwd=directory,
                   stdout=subprocess.DEVNULL, check=True)
    require(sha(directory / "SHA256SUMS.seal.sha256") == expected,
            "outer seal drift: " + directory.name)


def load_test_support():
    require(sha(TEST_SOURCE) ==
            "9f0f275fc9023346e541ea99fbc2f49365b79e8ea732087e53151d9afe8bfe89",
            "sandbox support drift")
    spec = importlib.util.spec_from_file_location("m1039_sandbox", TEST_SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load sandbox")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_faults(support):
    outcomes = {}
    for name in ("vcs1", "vlogan", "dc_shell", "dc_shell-t",
                 "fm_shell", "pt_shell"):
        with support.sandbox() as box:
            tool = box["root"] / "tool"
            tool.mkdir()
            fake = tool / name
            fake.symlink_to("/usr/bin/sleep")
            blocker = subprocess.Popen([str(fake), "3"])
            try:
                for _ in range(100):
                    if subprocess.run(["/usr/bin/pgrep", "-x", name],
                                      stdout=subprocess.DEVNULL).returncode == 0:
                        break
                    time.sleep(0.01)
                proc = support.run(box)
                require(proc.returncode != 0 and
                        "process collision: " + name in proc.stderr and
                        not box["attempt"].exists(),
                        "EDA collision reached attempt: " + name)
                outcomes["pgrep_" + name] = "REJECTED_PRE_ATTEMPT_ENGINE_NOT_RUN"
            finally:
                blocker.terminate()
                blocker.wait(timeout=2)
    with support.sandbox() as box:
        fd = os.open(str(box["lockfile"]), os.O_CREAT | os.O_WRONLY, 0o664)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            proc = support.run(box)
            require("C1 full replay lock collision" in proc.stderr and
                    not box["attempt"].exists(), "lock fault reached attempt")
            outcomes["flock_occupied"] = "REJECTED_PRE_ATTEMPT_ENGINE_NOT_RUN"
        finally:
            os.close(fd)
    resource_cases = {
        "commit_headroom_below_16gib": (
            "CommitLimit: 20000000 kB\nCommitted_AS: 10000000 kB\nMemAvailable: 100000000 kB\n",
            "CommitLimit-Committed_AS below 16GiB floor"),
        "memavailable_below_16gib": (
            "CommitLimit: 100000000 kB\nCommitted_AS: 10000000 kB\nMemAvailable: 10000000 kB\n",
            "MemAvailable below 16GiB floor"),
    }
    for key, (meminfo, message) in resource_cases.items():
        with support.sandbox(meminfo) as box:
            proc = support.run(box)
            require(message in proc.stderr and not box["attempt"].exists(),
                    key + " reached attempt")
            outcomes[key] = "REJECTED_PRE_ATTEMPT_ENGINE_NOT_RUN"
    with support.sandbox() as box:
        box["env"]["M1040_EXPECTED_M1025_OUTER_SHA256"] = "0" * 64
        proc = support.run(box)
        require("caller must pin exact M1025, M1036 and M1039 outer SHAs" in
                proc.stderr and not box["attempt"].exists(),
                "wrong outer reached attempt")
        outcomes["wrong_outer"] = "REJECTED_PRE_ATTEMPT_ENGINE_NOT_RUN"
    with support.sandbox() as box:
        value = json.loads(box["release"].read_text())
        value["status"] = "WRONG_RELEASE_STATUS"
        box["release"].write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        sidecar = Path(str(box["release"]) + ".sha256")
        sidecar.write_text("{}  {}\n".format(support.sha(box["release"]),
                                              box["release"].name))
        Path(str(box["release"]) + ".sha256.seal.sha256").write_text(
            "{}  {}\n".format(support.sha(sidecar), sidecar.name))
        hammer_review = json.loads((box["hammer"] / "review.json").read_text())
        hammer_review["identity"]["m1038_release_sha256"] = support.sha(box["release"])
        (box["hammer"] / "review.json").write_text(json.dumps(hammer_review) + "\n")
        box["env"]["M1040_EXPECTED_M1039_OUTER_SHA256"] = support.seal(box["hammer"])
        proc = support.run(box)
        require("hardcoded M1040 execution authority content mismatch" in
                proc.stderr and not box["attempt"].exists(),
                "wrong status reached attempt")
        outcomes["wrong_status"] = "REJECTED_PRE_ATTEMPT_ENGINE_NOT_RUN"
    with support.sandbox() as box:
        box["result"].mkdir()
        proc = support.run(box)
        require("M1040 result/attempt/work collision" in proc.stderr and
                not box["attempt"].exists(), "namespace fault reached attempt")
        outcomes["namespace_collision"] = "REJECTED_PRE_ATTEMPT_ENGINE_NOT_RUN"
    return outcomes


def normal_host_gate():
    active = [name for name in ("vcs1", "vlogan", "dc_shell", "dc_shell-t",
                                "fm_shell", "pt_shell")
              if subprocess.run(["/usr/bin/pgrep", "-x", name],
                                stdout=subprocess.DEVNULL).returncode == 0]
    require(not active, "normal host has active EDA collision")
    rows = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("CommitLimit", "Committed_AS", "MemAvailable"):
            rows[key] = int(value.strip().split()[0])
    headroom = rows["CommitLimit"] - rows["Committed_AS"]
    require(headroom >= 16777216 and rows["MemAvailable"] >= 16777216,
            "normal host resource floor failed")
    fd = os.open(str(LOCKFILE), os.O_CREAT | os.O_WRONLY, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)
    return {"eda_collisions": [], "global_lock_available": True,
            "commit_headroom_kb": headroom,
            "mem_available_kb": rows["MemAvailable"],
            "full_replay_run": False}


def main():
    require(sha(RUNNER) ==
            "47d73bcff61cc0721d79223c3e2f398e406ad87aba5359d2c1418674990d2c34",
            "runner drift")
    require(sha(RELEASE) ==
            "ce96a98abcf8fbbb75e98c0ef1c407c2b804aa6d231e36c12a4c13f9d03fd8d5",
            "release drift")
    verify_seal(M1025,
                "7004ab978588ebaed6b94e57c9c30bbaadb4c9502a57921dc1b1e40cfe7743ff")
    verify_seal(M1036,
                "476f0779ad32d40831dbcdaa5d4c223d7f6a50d9aecb196e63107ee4c1c8f5ae")
    verify_seal(M1037,
                "bea2c51aa7cea7e7f243093e9838c74e16ba1302c5610c33283f3d1817567a7c")
    subprocess.run(["sha256sum", "-c", RELEASE.name + ".sha256"],
                   cwd=RELEASE.parent, stdout=subprocess.DEVNULL, check=True)
    subprocess.run(["sha256sum", "-c", RELEASE.name + ".sha256.seal.sha256"],
                   cwd=RELEASE.parent, stdout=subprocess.DEVNULL, check=True)
    m1025 = json.loads((M1025 / "review.json").read_text())
    m1036 = json.loads((M1036 / "review.json").read_text())
    m1037 = json.loads((M1037 / "review.json").read_text())
    release = json.loads(RELEASE.read_text())
    require(m1025["status"] ==
            "PASS_M1025_M1016_C1_FULL_MATCHED_ADDRESS_REPLAY_SOURCE_HAMMER",
            "M1025 status drift")
    require(m1036["status"] ==
            "FAIL_M1036_M1026_M1027_M1028_C1_FULL_REPLAY_CROSS_HAMMER" and
            m1036["p0_count"] == 1, "M1036 STOP authority drift")
    require(m1037["status"] ==
            "PASS_M1037_M1038_M1040_COLLISION_RESOURCE_SOURCE" and
            m1037["score_out_of_100"] == 100, "M1037 receipt drift")
    require(release["status"] ==
            "PASS_M1038_M1037_M1016_C1_FULL_REPLAY_LAUNCH_RELEASE" and
            release["launch_now"] is True and release["max_attempts"] == 1,
            "M1038 release drift")
    require(not ATTEMPT.exists() and not RESULT.exists(),
            "M1040 namespace is not fresh")
    support = load_test_support()
    faults = run_faults(support)
    host = normal_host_gate()
    require(not ATTEMPT.exists() and not RESULT.exists(),
            "M1040 was consumed during hammer")
    return {
        "status": "PASS_M1039_M1038_M1036_M1040_C1_FULL_REPLAY_RELEASE_HAMMER",
        "score": 100, "p0": 0, "p1": 0, "p2": 0,
        "identity": {
            "m1040_runner_sha256": sha(RUNNER),
            "m1038_release_sha256": sha(RELEASE),
            "m1025_outer_seal_file_sha256":
                sha(M1025 / "SHA256SUMS.seal.sha256"),
            "m1036_outer_seal_file_sha256":
                sha(M1036 / "SHA256SUMS.seal.sha256"),
            "m1037_source_receipt_outer_seal_file_sha256":
                sha(M1037 / "SHA256SUMS.seal.sha256"),
        },
        "faults": faults,
        "normal_host_gate_only": host,
        "m1040_attempt_consumed": False,
        "m1040_result_created": False,
        "engine_runs": 0,
        "eda_runs": 0,
        "authorization": "ONE_M1040_CPU_FULL_REPLAY_ATTEMPT_ONLY",
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
