#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Static source checker for the M1037/M1038/M1040 repair chain."""
import hashlib
import json
from pathlib import Path
import subprocess


HW = Path(__file__).resolve().parents[2]
RUNNER = HW / "system_simulator/scripts/run_m1040_m1016_c1_full_matched_address_replay_one_shot.sh"
RELEASE = HW / "contracts/m1038_m1037_m1016_c1_full_matched_address_replay_launch_release_r1_20260829.json"
M1025 = HW / "reviews/m1025_m1016_c1_full_matched_address_replay_source_hammer_r1_20260829"
M1036 = HW / "reviews/m1036_m1026_m1027_m1028_c1_full_replay_cross_hammer_r1_20260829"
M1039 = HW / "reviews/m1039_m1038_m1036_m1040_c1_full_replay_release_hammer_r1_20260829"
RESULTS = HW / "results"

EXPECTED = {
    "runner": "47d73bcff61cc0721d79223c3e2f398e406ad87aba5359d2c1418674990d2c34",
    "release": "ce96a98abcf8fbbb75e98c0ef1c407c2b804aa6d231e36c12a4c13f9d03fd8d5",
    "release_sidecar": "048f83f470adbf00fd5fdfd9a89e257d2dd72fbd3b368b979b6e9d476703de9d",
    "release_outer": "8e4140902a36c908eeb4f3308067b8d3237c99f6818afce70ce27c32050b5f0e",
    "m1025_outer": "7004ab978588ebaed6b94e57c9c30bbaadb4c9502a57921dc1b1e40cfe7743ff",
    "m1036_outer": "476f0779ad32d40831dbcdaa5d4c223d7f6a50d9aecb196e63107ee4c1c8f5ae",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def audit_runner_text(text: str) -> None:
    names = "for name in vcs1 vlogan dc_shell dc_shell-t fm_shell pt_shell; do"
    required = (names, '"${pgrep_bin}" -x "${name}"',
                'lockfile="${hw_root}/results/.c1_full_matched_address_replay_global.lock"',
                '"${flock_bin}" -n "${lock_fd}"', "resource_gate",
                'CommitLimit-Committed_AS below 16GiB floor',
                'MemAvailable below 16GiB floor',
                "M1040_EXPECTED_M1039_OUTER_SHA256",
                "PASS_M1039_M1038_M1036_M1040_C1_FULL_REPLAY_RELEASE_HAMMER",
                'result="${hw_root}/results/m1040_',
                'attempt="${hw_root}/results/.m1040_',
                'phase=ATTEMPT_ATOMIC_CONSUME', 'mkdir "${attempt}"')
    for token in required:
        require(token in text, "runner token absent: " + token)
    attempt = text.index("phase=ATTEMPT_ATOMIC_CONSUME")
    require(text.index("process_collision_gate\n") < attempt and
            text.index('"${flock_bin}" -n "${lock_fd}"') < attempt and
            text.index("resource_gate\n") < attempt,
            "runtime gate occurs after attempt")
    require("pgrep -f" not in text and "pgrep -a" not in text,
            "generic process scan present")
    require("m1028_m1016_c1_full_matched_address_replay_r1" not in text,
            "old result namespace survived")


def main() -> dict:
    require(sha(RUNNER) == EXPECTED["runner"], "runner identity drift")
    sidecar = Path(str(RELEASE) + ".sha256")
    outer = Path(str(RELEASE) + ".sha256.seal.sha256")
    require(sha(RELEASE) == EXPECTED["release"] and
            sha(sidecar) == EXPECTED["release_sidecar"] and
            sha(outer) == EXPECTED["release_outer"], "release identity drift")
    subprocess.run(["sha256sum", "-c", sidecar.name], cwd=RELEASE.parent,
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["sha256sum", "-c", outer.name], cwd=RELEASE.parent,
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    audit_runner_text(RUNNER.read_text())
    release = json.loads(RELEASE.read_text())
    require(release["status"] == "PASS_M1038_M1037_M1016_C1_FULL_REPLAY_LAUNCH_RELEASE" and
            release["runner_sha256"] == EXPECTED["runner"] and
            release["m1025"]["outer_seal_file_sha256"] == EXPECTED["m1025_outer"] and
            release["m1036"]["outer_seal_file_sha256"] == EXPECTED["m1036_outer"],
            "release binding drift")
    require(release["runtime_gates"]["exact_process_names"] ==
            ["vcs1", "vlogan", "dc_shell", "dc_shell-t", "fm_shell", "pt_shell"] and
            release["runtime_gates"]["minimum_commit_headroom_kb"] == 16_777_216 and
            release["runtime_gates"]["minimum_mem_available_kb"] == 16_777_216 and
            release["runtime_gates"]["generic_cpu_process_blacklist"] is False,
            "runtime gate contract drift")
    require(not M1039.exists(), "M1039 must be independently authored")
    require(not (RESULTS / ".m1040_m1016_c1_full_matched_address_replay_attempt_consumed").exists() and
            not (RESULTS / "m1040_m1016_c1_full_matched_address_replay_r1_20260829").exists() and
            not (RESULTS / ".m1028_m1016_c1_full_matched_address_replay_attempt_consumed").exists(),
            "M1040/M1028 namespace consumed")
    return {"status": "PASS_M1037_M1038_M1040_COLLISION_RESOURCE_SOURCE",
            "runner_sha256": sha(RUNNER), "release_sha256": sha(RELEASE),
            "future_m1039_absent": True, "m1040_executed": False,
            "full_replay": False, "eda": False}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
