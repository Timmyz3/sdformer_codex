#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Cross-author M1036 hammer for the M1026/M1027/M1028 C1 replay chain.

The production M1028 runner and 51.84M-row engine are never invoked. Dynamic
faults execute a temporary path-redirected clone whose engine is /bin/false.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "system_simulator/scripts/run_m1028_m1016_c1_full_matched_address_replay_one_shot.sh"
RELEASE = HW / "contracts/m1026_m1016_c1_full_matched_address_replay_launch_release_r1_20260829.json"
RELEASE_SHA = Path(str(RELEASE) + ".sha256")
RELEASE_OUTER = Path(str(RELEASE) + ".sha256.seal.sha256")
M1025 = HW / "reviews/m1025_m1016_c1_full_matched_address_replay_source_hammer_r1_20260829"
M1027 = HW / "reviews/m1027_m1026_m1016_c1_full_matched_address_replay_release_hammer_r1_20260829"
RECEIPT = HW / "reviews/m1026_m1028_m1016_c1_full_matched_address_replay_execution_source_receipt_r1_20260829"
CONTRACT = HW / "contracts/m1016_m1010_c1_full_matched_address_replay_source_contract_r1_20260829.json"
ENGINE = HW / "system_simulator/scripts/run_m1016_c1_full_matched_address_replay.py"
ROWS = HW / "results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/m410r2_h67_q32_runtime_rows_32.memh"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
RESULTS = HW / "results"

EXPECTED = {
    "runner": "f557c0e65e745500579873d3b7df0d53fad8103d93452a0e7f7bc7327722dc47",
    "release": "96e9685ba9eae5f3545ca0745cfeff9acd2255000238b70f2d5f1b0b1ded1afd",
    "release_sidecar": "ab24498486fb371c67629329b98e9cf91610f6a4de000d58d4a3be5a9bb35553",
    "release_outer": "6b495010bfad7746479c1930cd724961e58c8f0896335a1b99f6c5f0b12e4fc4",
    "m1025_review": "6c86079035a52af4a36a3156ed4cdd6cb0bb71b51b198e71eb79512dfc361703",
    "m1025_manifest": "89155180dd298b752b4ef92ae70fe907e393733d23cb0a4d9be1b2a5f2784e16",
    "m1025_outer": "7004ab978588ebaed6b94e57c9c30bbaadb4c9502a57921dc1b1e40cfe7743ff",
    "m1027_review": "6cb83aee03fa1d83a622f3561d10b422505514e303df75baa97ffccf23bdfb7e",
    "m1027_manifest": "31f370ce1ecf3551b53f9a01b93b2e8b4d08da5a9d1ae85efea4a9991a3ea964",
    "m1027_outer": "fc7889262b9203686c7a672a08484f8da042841db6a29359776ec67189c3057a",
    "receipt_review": "29d336786eec1cd327d25b65c9694f087e4e333d07362f33ee596475124afa75",
    "receipt_manifest": "639480bb84bf4e483325fe671994a4f068ace21b066940c4609613e0651c0122",
    "receipt_outer": "eb58252f85af3f097995f5724a9e038f8403623557d36f9b267e4d4b3c8df9aa",
    "contract": "b980f51017778b1958845547601de5d343ba5a1f3db1b046963afa7549644c90",
    "engine": "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa",
    "rows": "6e03352b89eff1955825334b4dedd991db8c975a9ef6662fe0317e73ccfa8334",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(
        RuntimeError("nonfinite JSON: " + value)))


def verify_flat(directory: Path, expected: tuple[str, str, str]) -> dict:
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require((sha(review), sha(manifest), sha(outer)) == expected,
            "sealed identity drift: " + directory.name)
    listed = set()
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(None, 1)
        rel = rel.lstrip("./*")
        member = directory / rel
        require(rel not in listed and member.is_file() and not member.is_symlink() and
                sha(member) == digest, "sealed member drift: " + str(member))
        listed.add(rel)
    require(outer.read_text().split() == [expected[1], "SHA256SUMS"],
            "outer content drift: " + directory.name)
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and not path.is_symlink() and
              path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256") and
              "__pycache__" not in path.parts}
    require(listed == actual, "sealed exact-set drift: " + directory.name)
    return load_json(review)


def sandbox_runner(root: Path) -> tuple[Path, Path, Path, Path]:
    """Redirect all mutable paths and replace the huge engine with /bin/false."""
    runner = root / "runner.sh"
    release = root / "release.json"
    shutil.copy2(RELEASE, release)
    result = root / "result"
    attempt = root / "attempt"
    work = root / "work.$$"
    failure = root / "failure.$$.quarantine"
    false_sha = sha(Path("/bin/false"))
    text = RUNNER.read_text()
    replacements = {
        'hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"': f'hw_root="{HW}"',
        'engine="${hw_root}/system_simulator/scripts/run_m1016_c1_full_matched_address_replay.py"': 'engine="/bin/false"',
        'release="${hw_root}/contracts/m1026_m1016_c1_full_matched_address_replay_launch_release_r1_20260829.json"': f'release="{release}"',
        'result="${hw_root}/results/m1028_m1016_c1_full_matched_address_replay_r1_20260829"': f'result="{result}"',
        'attempt="${hw_root}/results/.m1028_m1016_c1_full_matched_address_replay_attempt_consumed"': f'attempt="{attempt}"',
        'work="${hw_root}/results/.m1028_m1016_c1_full_matched_address_replay_work.$$"': f'work="{work}"',
        'failure="${hw_root}/results/m1028_m1016_c1_full_matched_address_replay_r1_20260829.failed_or_incomplete.$$.quarantine"': f'failure="{failure}"',
        'expected_engine_sha=d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa': f'expected_engine_sha={false_sha}',
    }
    for old, new in replacements.items():
        require(text.count(old) == 1, "sandbox patch anchor drift: " + old)
        text = text.replace(old, new)
    runner.write_text(text)
    runner.chmod(0o755)
    value = load_json(release)
    value["engine_sha256"] = false_sha
    value["runner_sha256"] = sha(runner)
    release.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return runner, release, result, attempt


def invoke_fault(name: str, mutate, expected_fragment: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="m1036_fault_") as td:
        root = Path(td)
        runner, release, result, attempt = sandbox_runner(root)
        env = {"PATH": "/usr/bin:/bin",
               "M1028_EXPECTED_RUNNER_SHA256": sha(runner),
               "M1028_EXPECTED_M1025_OUTER_SHA256": EXPECTED["m1025_outer"],
               "M1028_EXPECTED_M1027_OUTER_SHA256": EXPECTED["m1027_outer"]}
        mutate(root, runner, release, result, attempt, env)
        before = attempt.exists()
        proc = subprocess.run([str(runner)], text=True, capture_output=True, env=env,
                              check=False, timeout=20)
        combined = proc.stdout + proc.stderr
        require(proc.returncode != 0 and expected_fragment in combined,
                name + " wrong failure: " + combined[-500:])
        require(attempt.exists() == before, name + " crossed attempt boundary")
        require(not result.exists(), name + " created result")
        require(not list(root.glob("failure.*")) and not list(root.glob("work.*")),
                name + " created post-attempt artifacts")
        return {"fault": name, "return_code": proc.returncode,
                "attempt_created": False, "full_replay": False, "eda": False}


def main() -> dict:
    require(sha(RUNNER) == EXPECTED["runner"] and sha(RELEASE) == EXPECTED["release"] and
            sha(RELEASE_SHA) == EXPECTED["release_sidecar"] and
            sha(RELEASE_OUTER) == EXPECTED["release_outer"], "runner/release identity drift")
    require(RELEASE_SHA.read_text().split() == [EXPECTED["release"], RELEASE.name] and
            RELEASE_OUTER.read_text().split() == [EXPECTED["release_sidecar"], RELEASE_SHA.name],
            "release double sidecar drift")
    for path, key in ((CONTRACT, "contract"), (ENGINE, "engine"), (ROWS, "rows"),
                      (DOC359, "docs359")):
        require(sha(path) == EXPECTED[key], key + " authority drift")
    m1025 = verify_flat(M1025, (EXPECTED["m1025_review"], EXPECTED["m1025_manifest"],
                               EXPECTED["m1025_outer"]))
    m1027 = verify_flat(M1027, (EXPECTED["m1027_review"], EXPECTED["m1027_manifest"],
                               EXPECTED["m1027_outer"]))
    receipt = verify_flat(RECEIPT, (EXPECTED["receipt_review"], EXPECTED["receipt_manifest"],
                                   EXPECTED["receipt_outer"]))
    require(m1025["status"] == "PASS_M1025_M1016_C1_FULL_MATCHED_ADDRESS_REPLAY_SOURCE_HAMMER" and
            m1025["authorization"]["execute_51840000_replay"] is False,
            "M1025 scope drift")
    require(m1027["status"] == "PASS_M1027_M1026_M1016_C1_FULL_REPLAY_RELEASE_HAMMER" and
            m1027["identity"]["runner_sha256"] == EXPECTED["runner"] and
            m1027["identity"]["release_sha256"] == EXPECTED["release"] and
            m1027["identity"]["m1025_outer_sha256"] == EXPECTED["m1025_outer"],
            "M1027 authority drift")
    require(receipt["status"] == "PASS_M1026_M1027_M1028_EXECUTION_SOURCE_CHAIN__NO_EXECUTION",
            "execution source receipt drift")
    release = load_json(RELEASE)
    require(release["runner_sha256"] == EXPECTED["runner"] and
            release["engine_sha256"] == EXPECTED["engine"] and
            release["m1025_outer_seal_file_sha256"] == EXPECTED["m1025_outer"] and
            release["execution"]["raw_rows"] == 51_840_000 and
            release["execution"]["samples"] == 10 and
            release["execution"]["operators"] == 4 and
            release["execution"]["partitions"] == 432 and
            release["execution"]["rows_per_phase"] == 3000 and
            release["execution"]["output_blocks"] == 8,
            "release identity/51.84M geometry drift")
    require(release["execution"]["hardcoded_authority_paths"] is True and
            release["execution"]["caller_selectable_release_path"] is False and
            release["execution"]["caller_selectable_hammer_path"] is False,
            "authority path scope drift")
    auth = release["authorization"]
    require(auth["one_m1028_attempt"] is True and auth["automatic_retry"] is False and
            auth["cpu_full_replay"] is True and all(auth[key] is False for key in
            ("vcs", "dc", "formality", "pt", "ptpx", "gpu", "remote")),
            "CPU-only authorization drift")
    require(release["expected_raw_result_boundary"]["capacity_only_214912B_admitted"] is False and
            release["expected_raw_result_boundary"]["matched_cycles_admitted"] is False and
            release["expected_raw_result_boundary"]["speedup_admitted"] is False,
            "raw result claim boundary drift")
    text = RUNNER.read_text()
    for token in ("cleanup()", "seal_dir()", "ATTEMPT_ATOMIC_CONSUME",
                  "FULL_51840000_CPU_REPLAY", "mkdir \"${attempt}\"",
                  "capacity_only_214912B_admitted", "speedup_admitted"):
        require(token in text, "runner token absent: " + token)
    for forbidden in ("M1016_RELEASE_JSON", "M1016_RELEASE_HAMMER_DIR", "/opt/synopsys",
                      "dc_shell", "pt_shell", "nvidia-smi", "ssh "):
        require(forbidden not in text, "non-CPU/generic path in runner: " + forbidden)
    require(not (RESULTS / ".m1028_m1016_c1_full_matched_address_replay_attempt_consumed").exists() and
            not (RESULTS / "m1028_m1016_c1_full_matched_address_replay_r1_20260829").exists(),
            "M1028 namespace consumed")
    subprocess.run(["bash", "-n", str(RUNNER)], check=True, timeout=10)

    faults = []
    faults.append(invoke_fault("wrong_runner_sha",
        lambda root, runner, release, result, attempt, env:
            env.__setitem__("M1028_EXPECTED_RUNNER_SHA256", "0" * 64),
        "caller must pin exact M1028 runner SHA"))
    faults.append(invoke_fault("wrong_m1025_outer",
        lambda root, runner, release, result, attempt, env:
            env.__setitem__("M1028_EXPECTED_M1025_OUTER_SHA256", "1" * 64),
        "caller must pin exact hardcoded M1025 outer SHA"))
    faults.append(invoke_fault("wrong_m1027_outer",
        lambda root, runner, release, result, attempt, env:
            env.__setitem__("M1028_EXPECTED_M1027_OUTER_SHA256", "2" * 64),
        "outer seal identity drift"))

    def bad_status(root, runner, release, result, attempt, env):
        value = load_json(release)
        value["status"] = "STOP_INJECTED"
        release.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    faults.append(invoke_fault("wrong_release_status", bad_status,
                               "hardcoded execution authority content mismatch"))

    def bad_engine_pin(root, runner, release, result, attempt, env):
        value = load_json(release)
        value["engine_sha256"] = "3" * 64
        release.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    faults.append(invoke_fault("wrong_release_engine_sha", bad_engine_pin,
                               "hardcoded execution authority content mismatch"))

    def occupied(root, runner, release, result, attempt, env):
        attempt.mkdir()
        (attempt / "SENTINEL").write_text("occupied\n")

    faults.append(invoke_fault("occupied_namespace", occupied,
                               "M1028 result/attempt/work collision"))

    # Collision injection uses a fake pgrep that reports an active tool. A
    # correct collision gate would stop before attempt creation. The sandbox
    # engine is /bin/false, so even a fail-open runner cannot start full replay.
    collision_attempt_created = False
    collision_cleanup_sealed = False
    with tempfile.TemporaryDirectory(prefix="m1036_collision_") as td:
        root = Path(td)
        runner, release, result, attempt = sandbox_runner(root)
        fake_bin = root / "bin"
        fake_bin.mkdir()
        pgrep = fake_bin / "pgrep"
        pgrep.write_text("#!/bin/sh\nexit 0\n")
        pgrep.chmod(0o755)
        env = {"PATH": f"{fake_bin}:/usr/bin:/bin",
               "M1028_EXPECTED_RUNNER_SHA256": sha(runner),
               "M1028_EXPECTED_M1025_OUTER_SHA256": EXPECTED["m1025_outer"],
               "M1028_EXPECTED_M1027_OUTER_SHA256": EXPECTED["m1027_outer"]}
        proc = subprocess.run([str(runner)], text=True, capture_output=True, env=env,
                              check=False, timeout=20)
        collision_attempt_created = attempt.exists()
        quarantines = list(root.glob("failure.*.quarantine"))
        if quarantines:
            quarantine = quarantines[0]
            try:
                subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=quarantine,
                               check=True, stdout=subprocess.DEVNULL)
                subprocess.run(["sha256sum", "-c", "SHA256SUMS.seal.sha256"], cwd=quarantine,
                               check=True, stdout=subprocess.DEVNULL)
                collision_cleanup_sealed = True
            except subprocess.CalledProcessError:
                collision_cleanup_sealed = False
        require(proc.returncode != 0 and not result.exists(),
                "collision sandbox unexpectedly completed")

    p0 = []
    if collision_attempt_created:
        p0.append({
            "id": "P0_CPU_REPLAY_COLLISION_GATE_ABSENT",
            "finding": "M1028 contains no collision gate. A sandbox pgrep oracle reporting an active tool was never queried; the runner consumed its attempt and reached the harmless replacement engine.",
            "required_repair": "Add a pre-attempt collision gate for the frozen conflicting processes/CPU replay lock, independently fault-test it, and use a fresh additive namespace because no production M1028 attempt was consumed.",
        })

    return {
        "schema": "m1036_m1026_m1027_m1028_c1_full_replay_cross_hammer_r1_v1",
        "date": "2026-08-29", "milestone": "M1036",
        "status": ("FAIL_M1036_M1026_M1027_M1028_C1_FULL_REPLAY_CROSS_HAMMER"
                   if p0 else "PASS_M1036_M1026_M1027_M1028_C1_FULL_REPLAY_CROSS_HAMMER"),
        "verdict": ("STOP_AUTHOR_ADDITIVE_COLLISION_GATE_REPAIR"
                    if p0 else "GO_ONE_M1028_CPU_FULL_REPLAY_ATTEMPT_ONLY"),
        "score_out_of_100": 88 if p0 else 100,
        "p0_count": len(p0), "p1_count": 0, "p2_count": 0,
        "identity": {
            "runner_sha256": sha(RUNNER), "release_sha256": sha(RELEASE),
            "m1025_outer_seal_file_sha256": sha(M1025 / "SHA256SUMS.seal.sha256"),
            "m1027_outer_seal_file_sha256": sha(M1027 / "SHA256SUMS.seal.sha256"),
            "contract_sha256": sha(CONTRACT), "engine_sha256": sha(ENGINE),
            "m410_rows_sha256": sha(ROWS), "docs359_sha256": sha(DOC359),
        },
        "mechanical": {
            "hardcoded_authority_paths": True, "new_namespace_fresh": True,
            "one_shot_atomic_attempt": True, "cleanup_quarantine": True,
            "cleanup_sealed_in_collision_sandbox": collision_cleanup_sealed,
            "raw_rows": 51_840_000, "samples": 10, "operators": 4,
            "partitions": 432, "rows_per_phase": 3000, "output_blocks": 8,
            "cpu_only": True, "bash_n": "PASS",
            "pre_attempt_faults_rejected": faults,
            "collision_oracle_attempt_created": collision_attempt_created,
        },
        "p0": p0,
        "authorization": {
            "m1028_cpu_full_replay": not p0, "automatic_retry": False,
            "eda": False, "gpu": False, "remote": False,
            "capacity_214912B_admitted": False,
            "matched_cycles_admitted": False, "speedup_admitted": False,
        },
        "scope": {
            "cross_author_hammer": True, "production_m1028_runs": 0,
            "full_51840000_replays": 0, "eda_runs": 0,
            "sandbox_engine": "/bin/false", "docs359_modified": False,
        },
        "claim_boundary": {
            "full_result": False, "capacity_214912B": False,
            "matched_cycles": False, "speedup": False, "paper_ppa_ready": False,
        },
    }


if __name__ == "__main__":
    result = main()
    (HERE / "review.json").write_text(json.dumps(result, indent=2, sort_keys=True,
                                                  allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
