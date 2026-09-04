#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent M1020 hammer for the repaired C2 mapped-gate SAIF chain.

The only Synopsys invocation is a clean-environment ``vcs -full64 -ID``
frontend identity smoke.  No design is compiled and M1022 is never run.
Dynamic runner faults operate on temporary clones and must stop before their
temporary attempt namespace is consumed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "dc_handoff/scripts/run_m1022_m1001_c2_mapped_gate_saif_one_shot_r3.sh"
RELEASE = HW / "contracts/m1019_m1018_m1001_c2_mapped_gate_saif_launch_release_r3_20260829.json"
CONTRACT = HW / "contracts/m1001_m979_c2_mapped_gate_saif_rekey_source_contract_r1_20260829.json"
M1002 = HW / "reviews/m1002_m1001_c2_mapped_gate_saif_rekey_source_hammer_r1_20260829"
M1018 = HW / "reviews/m1018_m1013_c2_saif_compile_failure_audit_r1_20260829"
M1021 = HW / "reviews/m1021_m1018_m1019_m1022_c2_saif_environment_repair_source_receipt_r1_20260829"
M1013_ATTEMPT = HW / "results/.m1013_m1001_c2_three_axis_mapped_gate_saif_attempt_consumed"
M1013_QUARANTINE = HW / "results/m1013_m1001_c2_three_axis_mapped_gate_saif_r2_20260829.failed_or_incomplete.1554180.quarantine"
M1022_ATTEMPT = HW / "results/.m1022_m1001_c2_three_axis_mapped_gate_saif_attempt_consumed"
M1022_RESULT = HW / "results/m1022_m1001_c2_three_axis_mapped_gate_saif_r3_20260829"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
VCS_HOME = Path("/opt/synopsys/vcs/V-2023.12-SP1")
VCS = VCS_HOME / "bin/vcs"
VCS_MSG = VCS_HOME / "bin/vcsMsgReport"

EXPECTED = {
    "runner": "dbaa5b0b9619cb60b556a42f27e9e926a56bcb22d4627c13048b70a3fc74af1b",
    "release": "fcd5a659a8b057f5e84e1b8691645cf1cf66ecafed0cf6dec0827ed9a571f609",
    "release_sidecar": "56a92631e49884614eb689abccd4d7022367735f736defbc2dfea28ef11fd2ea",
    "release_outer": "6e839e3d37b95f40bdc276441ad8068c47980dd665f4f36db69cccf3ab177e7e",
    "contract": "7afc4c093b802bdfd97aea101c803735e993c2eef57983311d3eb1a3d6bd36c6",
    "m1002_review": "e747c73b3add43e7010fc539f9f06d35f5e6e69219a9e66fc7f0e25e511045d7",
    "m1002_manifest": "019b810281f815d44d0024b89556ac7cacaea2c28885aa4ce79ead37761cc6eb",
    "m1002_outer": "d489e1cc3893e9c2a265ad5d35213e349f6eb44a5b4e2e15189711b1c82f5b85",
    "m1018_review": "55b1656b9f903e684ff5c418081f89dee71910585067ca807b33032d923a95a8",
    "m1018_manifest": "223c80b45beaf81f2ce8dd2d8eb265c7bb46a7bb404594ce661d787082bfcf12",
    "m1018_outer": "5c096458a507e99e922ba6a0658ac7e28bf4f2710a2f49cc971d15c738f32146",
    "m1021_review": "e9979dd58a65b1cda43dea73edd9a8beade2592a9350250b564e3160c2c03f43",
    "m1021_manifest": "a388553c6819ae7173101a651fbf557b3ebab120cbdf7f1cd55e1152458c874d",
    "m1021_outer": "b6383c2408cbd5c09536b62d8dda2799b479a7295f425d76e0726056acbeedfa",
    "m1013_attempt_manifest": "fe9638b1eaac54b6c4c55ce2530a325f18f071475cb8bae56d0af2bacc8d85d6",
    "m1013_attempt_outer": "9003e7419d29b8ff44237ef607dd546c51b070d20654aac327c78b478e656345",
    "m1013_quarantine_manifest": "a699d5776eac7eb803ba98c9f4d2b6551a9d702a57ade6f1053c294fcbfb5457",
    "m1013_quarantine_outer": "46d310564ce1c6332240ac9529f91640b56a5a71dad63d3c769b8af0439edc1a",
    "vcs": "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    "vcs_msg": "b34e06a92b05856532f868d32c0c81f1708506096856ad9a97bd27e2bd60215b",
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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"),
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + token)))


def verify_flat(directory: Path, review_sha: str | None = None,
                manifest_sha: str | None = None,
                outer_sha: str | None = None,
                require_review: bool = True) -> dict:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink(),
            "missing/symlink sealed directory: " + str(directory))
    require(manifest.is_file() and outer.is_file(), "seal absent: " + str(directory))
    if manifest_sha:
        require(sha(manifest) == manifest_sha, "manifest drift: " + directory.name)
    if outer_sha:
        require(sha(outer) == outer_sha, "outer drift: " + directory.name)
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(None, 1)
        relative = Path(relative.lstrip("*")).as_posix()
        member = directory / relative
        require(relative not in listed and member.is_file() and not member.is_symlink(),
                "sealed member absent/duplicate/symlink: " + relative)
        require(sha(member) == digest, "sealed member drift: " + relative)
        listed[relative] = digest
    require(outer.read_text(encoding="utf-8").split() == [sha(manifest), "SHA256SUMS"],
            "outer content drift: " + directory.name)
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and not path.is_symlink() and
              path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    require(set(listed) == actual, "seal exact-set drift: " + directory.name)
    if require_review:
        review = directory / "review.json"
        require(review.is_file() and (not review_sha or sha(review) == review_sha),
                "review drift: " + directory.name)
        return load_json(review)
    return {"member_count": len(listed)}


def reseal(directory: Path) -> str:
    members = sorted(path for path in directory.rglob("*") if path.is_file() and
                     path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    manifest = directory / "SHA256SUMS"
    manifest.write_text("".join(
        f"{sha(path)}  {path.relative_to(directory)}\n" for path in members),
        encoding="utf-8")
    outer = directory / "SHA256SUMS.seal.sha256"
    outer.write_text(f"{sha(manifest)}  SHA256SUMS\n", encoding="utf-8")
    return sha(outer)


def clean_env_frontend_smoke() -> dict:
    env = {
        "VCS_HOME": str(VCS_HOME),
        "PATH": f"{VCS_HOME}/bin:/usr/bin:/bin",
    }
    completed = subprocess.run([str(VCS), "-full64", "-ID"], env=env,
                               text=True, capture_output=True, timeout=30,
                               check=False)
    combined = completed.stdout + completed.stderr
    require(completed.returncode == 0, "clean-env -full64 frontend smoke failed")
    require("Compiler version = VCS V-2023.12-SP1_Full64" in combined,
            "frontend compiler identity absent")
    require("/bin/vcsMsgReport" not in combined and
            "Cannot find vcsMsgReport" not in combined,
            "frontend regressed to /bin/vcsMsgReport")
    return {
        "command": "env -i VCS_HOME=<exact> PATH=<exact>/bin:/usr/bin:/bin vcs -full64 -ID",
        "return_code": completed.returncode,
        "compiler": "VCS V-2023.12-SP1_Full64",
        "machine_type": "linux64",
        "bin_vcs_msg_report_error": False,
        "design_compile": False,
    }


def early_source_fault(name: str, mutate, expected: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="m1020_early_fault_") as temporary:
        runner = Path(temporary) / "runner.sh"
        text = mutate(RUNNER.read_text(encoding="utf-8"))
        runner.write_text(text, encoding="utf-8")
        runner.chmod(0o755)
        completed = subprocess.run([str(runner)], env={"PATH": "/usr/bin:/bin"},
                                   text=True, capture_output=True, timeout=20,
                                   check=False)
        combined = completed.stdout + completed.stderr
        require(completed.returncode != 0 and expected in combined,
                name + " wrong fail-closed result: " + combined[-600:])
        require(not M1022_ATTEMPT.exists() and not M1022_RESULT.exists(),
                name + " touched production namespace")
        return {"name": name, "return_code": completed.returncode,
                "expected_failure": expected, "attempt_created": False,
                "vcs_compile_invoked": False}


def sandbox_runner(root: Path):
    source_hammer = root / "m1002"
    failure_audit = root / "m1018"
    release_hammer = root / "m1020"
    shutil.copytree(M1002, source_hammer)
    shutil.copytree(M1018, failure_audit)
    release = root / "m1019_release.json"
    shutil.copy2(RELEASE, release)
    runner = root / "runner.sh"
    result = root / "result"
    attempt = root / "attempt"
    work = root / "work.$$"
    failure = root / "failure.$$.quarantine"
    text = RUNNER.read_text(encoding="utf-8")
    replacements = {
        'dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"': f'dc_root="{HW / "dc_handoff"}"',
        'hw_root="$(cd "${dc_root}/.." && pwd)"': f'hw_root="{HW}"',
        'source_hammer="${hw_root}/reviews/m1002_m1001_c2_mapped_gate_saif_rekey_source_hammer_r1_20260829"': f'source_hammer="{source_hammer}"',
        'failure_audit="${hw_root}/reviews/m1018_m1013_c2_saif_compile_failure_audit_r1_20260829"': f'failure_audit="{failure_audit}"',
        'release="${hw_root}/contracts/m1019_m1018_m1001_c2_mapped_gate_saif_launch_release_r3_20260829.json"': f'release="{release}"',
        'release_hammer="${hw_root}/reviews/m1020_m1019_m1018_m1022_c2_saif_release_hammer_r1_20260829"': f'release_hammer="{release_hammer}"',
        'result="${hw_root}/results/m1022_m1001_c2_three_axis_mapped_gate_saif_r3_20260829"': f'result="{result}"',
        'attempt="${hw_root}/results/.m1022_m1001_c2_three_axis_mapped_gate_saif_attempt_consumed"': f'attempt="{attempt}"',
        'work="${hw_root}/results/.m1022_m1001_c2_three_axis_mapped_gate_saif_work.$$"': f'work="{work}"',
        'failure="${hw_root}/results/m1022_m1001_c2_three_axis_mapped_gate_saif_r3_20260829.failed_or_incomplete.$$.quarantine"': f'failure="{failure}"',
    }
    for old, new in replacements.items():
        require(text.count(old) == 1, "sandbox patch anchor drift: " + old)
        text = text.replace(old, new)
    runner.write_text(text, encoding="utf-8")
    runner.chmod(0o755)

    release_data = load_json(release)
    release_data["runner_sha256"] = sha(runner)
    release.write_text(json.dumps(release_data, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    sidecar = Path(str(release) + ".sha256")
    sidecar.write_text(f"{sha(release)}  {release.name}\n", encoding="utf-8")
    outer = Path(str(release) + ".sha256.seal.sha256")
    outer.write_text(f"{sha(sidecar)}  {sidecar.name}\n", encoding="utf-8")

    release_hammer.mkdir()
    fake_review = {
        "status": "PASS_M1020_M1019_M1018_M1022_C2_SAIF_RELEASE_HAMMER",
        "identity": {
            "m1019_release_sha256": sha(release),
            "m1022_runner_sha256": sha(runner),
            "m1002_outer_seal_file_sha256": sha(source_hammer / "SHA256SUMS.seal.sha256"),
            "m1018_outer_seal_file_sha256": sha(failure_audit / "SHA256SUMS.seal.sha256"),
        },
    }
    (release_hammer / "review.json").write_text(
        json.dumps(fake_review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    release_outer = reseal(release_hammer)
    env = {
        "PATH": "/usr/bin:/bin",
        "SNPSLMD_LICENSE_FILE": os.environ.get("SNPSLMD_LICENSE_FILE", ""),
        "M1022_EXPECTED_RUNNER_SHA256": sha(runner),
        "M1022_EXPECTED_M1002_OUTER_SHA256": sha(source_hammer / "SHA256SUMS.seal.sha256"),
        "M1022_EXPECTED_M1018_OUTER_SHA256": sha(failure_audit / "SHA256SUMS.seal.sha256"),
        "M1022_EXPECTED_M1020_OUTER_SHA256": release_outer,
    }
    return runner, attempt, result, env


def occupied_attempt_fault() -> dict:
    with tempfile.TemporaryDirectory(prefix="m1020_occupied_") as temporary:
        root = Path(temporary)
        runner, attempt, result, env = sandbox_runner(root)
        attempt.mkdir()
        (attempt / "SENTINEL_DO_NOT_TOUCH").write_text("occupied\n", encoding="utf-8")
        completed = subprocess.run([str(runner)], env=env, text=True,
                                   capture_output=True, timeout=30, check=False)
        combined = completed.stdout + completed.stderr
        require(completed.returncode != 0 and
                "result/attempt/work collision" in combined,
                "occupied namespace did not fail closed: " + combined[-800:])
        require((attempt / "SENTINEL_DO_NOT_TOUCH").read_text() == "occupied\n",
                "occupied namespace was mutated")
        require(not result.exists() and not list(root.glob("work.*")) and
                not list(root.glob("failure.*.quarantine")),
                "occupied fault crossed attempt boundary")
        return {"name": "occupied_attempt_namespace",
                "return_code": completed.returncode,
                "expected_failure": "result/attempt/work collision",
                "attempt_sentinel_unchanged": True,
                "new_attempt_created": False, "vcs_compile_invoked": False}


def active_collision_fault() -> dict:
    with tempfile.TemporaryDirectory(prefix="m1020_collision_") as temporary:
        root = Path(temporary)
        runner, attempt, result, env = sandbox_runner(root)
        fake = root / "vcs1"
        fake.symlink_to("/usr/bin/sleep")
        blocker = subprocess.Popen([str(fake), "10"])
        try:
            time.sleep(0.1)
            completed = subprocess.run([str(runner)], env=env, text=True,
                                       capture_output=True, timeout=30,
                                       check=False)
            combined = completed.stdout + completed.stderr
            require(completed.returncode != 0 and
                    "VCS/DC/FM/PT collision" in combined,
                    "active collision did not fail closed: " + combined[-800:])
            require(not attempt.exists() and not result.exists(),
                    "collision consumed attempt")
            return {"name": "active_vcs1_collision",
                    "return_code": completed.returncode,
                    "expected_failure": "VCS/DC/FM/PT collision",
                    "attempt_created": False, "vcs_compile_invoked": False}
        finally:
            blocker.terminate()
            try:
                blocker.wait(timeout=2)
            except subprocess.TimeoutExpired:
                blocker.kill()
                blocker.wait(timeout=2)


def main() -> dict:
    require(sha(RUNNER) == EXPECTED["runner"], "runner identity drift")
    require(sha(RELEASE) == EXPECTED["release"], "release identity drift")
    release_sidecar = Path(str(RELEASE) + ".sha256")
    release_outer = Path(str(RELEASE) + ".sha256.seal.sha256")
    require(sha(release_sidecar) == EXPECTED["release_sidecar"] and
            sha(release_outer) == EXPECTED["release_outer"],
            "release sidecar drift")
    require(release_sidecar.read_text().split() == [EXPECTED["release"], RELEASE.name] and
            release_outer.read_text().split() == [EXPECTED["release_sidecar"], release_sidecar.name],
            "release sidecar content drift")
    require(sha(CONTRACT) == EXPECTED["contract"], "M1001 contract drift")
    require(sha(VCS) == EXPECTED["vcs"] and sha(VCS_MSG) == EXPECTED["vcs_msg"] and
            os.access(VCS, os.X_OK) and os.access(VCS_MSG, os.X_OK) and
            not VCS.is_symlink() and not VCS_MSG.is_symlink(),
            "VCS executable/support identity drift")
    require(sha(DOC359) == EXPECTED["docs359"], "docs/359 drift")

    m1002 = verify_flat(M1002, EXPECTED["m1002_review"],
                        EXPECTED["m1002_manifest"], EXPECTED["m1002_outer"])
    m1018 = verify_flat(M1018, EXPECTED["m1018_review"],
                        EXPECTED["m1018_manifest"], EXPECTED["m1018_outer"])
    m1021 = verify_flat(M1021, EXPECTED["m1021_review"],
                        EXPECTED["m1021_manifest"], EXPECTED["m1021_outer"])
    verify_flat(M1013_ATTEMPT, manifest_sha=EXPECTED["m1013_attempt_manifest"],
                outer_sha=EXPECTED["m1013_attempt_outer"], require_review=False)
    verify_flat(M1013_QUARANTINE,
                manifest_sha=EXPECTED["m1013_quarantine_manifest"],
                outer_sha=EXPECTED["m1013_quarantine_outer"],
                require_review=False)

    require(m1002["status"] == "PASS_M1002_M1001_SOURCE_HAMMER",
            "M1002 authority drift")
    require(m1018["status"] ==
            "PASS_M1018_M1013_FAILURE_AUDIT__M1013_DO_NOT_RETRY" and
            m1018["failure_boundary"]["gate_simulations_completed"] == 0 and
            m1018["failure_boundary"]["saif_files_created"] == 0 and
            m1018["failure_boundary"]["m1013_retry_authorized"] is False,
            "M1018 failure boundary drift")
    require(m1021["status"] ==
            "PASS_M1021_M1018_M1019_M1022_ENVIRONMENT_REPAIR_SOURCE",
            "M1021 source receipt drift")
    release = load_json(RELEASE)
    require(release["status"] ==
            "PASS_M1019_M1018_M1001_C2_SAIF_LAUNCH_RELEASE_R3" and
            release["runner_sha256"] == EXPECTED["runner"] and
            release["max_attempts"] == 1 and release["launch_now"] is True,
            "M1019 release content drift")
    require(release["execution"]["axes"] == ["k1", "k8", "k1x8"] and
            release["execution"]["cases_per_axis"] == 5 and
            release["execution"]["total_gate_simulations"] == 15 and
            release["execution"]["fresh_compile_per_axis"] is True and
            release["execution"]["old_simv_reuse"] is False and
            release["execution"]["dut_only_saif"] is True,
            "execution geometry drift")
    require(release["independent_hammer"]["required_status"] ==
            "PASS_M1020_M1019_M1018_M1022_C2_SAIF_RELEASE_HAMMER",
            "required hammer status drift")
    require(all(release["authorization"][key] is False for key in
                ("automatic_retry", "m1013_retry", "pt", "ptpx", "dc", "gpu_remote")),
            "release authorization expanded")

    text = RUNNER.read_text(encoding="utf-8")
    require(text.count('for axis in k1 k8 k1x8; do') == 1 and
            text.count('for case_id in 0 1 2 3 4; do') == 1 and
            text.count('"${vcs}" -full64') == 1 and
            text.index('"${vcs}" -full64') < text.index('for case_id in 0 1 2 3 4; do'),
            "fresh 3x5 compile geometry drift")
    require('power tb_m979_c2_three_axis_mapped_gate_case_saif.dut' in
            (HW / "dc_handoff/scripts/m979_c2_mapped_gate_per_case_saif.ucli.tcl").read_text(),
            "SAIF is not DUT-only")
    require(text.index('expect_sha "${vcs_msg_report}"') <
            text.index('mkdir "${attempt}"') and
            text.index('verify_seal "${release_hammer}"') <
            text.index('mkdir "${attempt}"') and
            text.index('collision_gate') < text.index('mkdir "${attempt}"'),
            "pre-attempt ordering drift")
    require('result="${hw_root}/results/m1022_' in text and
            'attempt="${hw_root}/results/.m1022_' in text and
            "M1013_EXPECTED_" not in text and "m1013_m1001_c2_three_axis" not in text,
            "fresh namespace drift")
    require(not M1022_ATTEMPT.exists() and not M1022_RESULT.exists(),
            "M1022 namespace already consumed")
    subprocess.run(["bash", "-n", str(RUNNER)], check=True, timeout=10)

    frontend = clean_env_frontend_smoke()
    faults = [
        early_source_fault(
            "missing_vcs_home_export",
            lambda source: source.replace('export VCS_HOME="${expected_vcs_home}"', ""),
            "VCS_HOME: unbound variable"),
        early_source_fault(
            "wrong_vcs_home_root",
            lambda source: source.replace(
                "readonly expected_vcs_home=/opt/synopsys/vcs/V-2023.12-SP1",
                "readonly expected_vcs_home=/opt/synopsys/vcs/INJECTED_WRONG"),
            "VCS_HOME installation root drift"),
        early_source_fault(
            "wrong_vcs_msg_report_sha",
            lambda source: source.replace(EXPECTED["vcs_msg"], "0" * 64),
            "identity drift"),
        occupied_attempt_fault(),
        active_collision_fault(),
    ]

    return {
        "schema": "m1020_m1019_m1018_m1022_c2_saif_release_hammer_r1_v1",
        "date": "2026-08-29",
        "milestone": "M1020",
        "status": "PASS_M1020_M1019_M1018_M1022_C2_SAIF_RELEASE_HAMMER",
        "verdict": "GO_ONE_M1022_VCS_SAIF_ATTEMPT_ONLY",
        "score_out_of_100": 100,
        "p0_count": 0,
        "p1_count": 0,
        "p2_count": 0,
        "identity": {
            "m1001_contract_sha256": sha(CONTRACT),
            "m1002_review_sha256": sha(M1002 / "review.json"),
            "m1002_manifest_sha256": sha(M1002 / "SHA256SUMS"),
            "m1002_outer_seal_file_sha256": sha(M1002 / "SHA256SUMS.seal.sha256"),
            "m1018_review_sha256": sha(M1018 / "review.json"),
            "m1018_manifest_sha256": sha(M1018 / "SHA256SUMS"),
            "m1018_outer_seal_file_sha256": sha(M1018 / "SHA256SUMS.seal.sha256"),
            "m1019_release_sha256": sha(RELEASE),
            "m1021_outer_seal_file_sha256": sha(M1021 / "SHA256SUMS.seal.sha256"),
            "m1022_runner_sha256": sha(RUNNER),
            "vcs_sha256": sha(VCS),
            "vcs_msg_report_sha256": sha(VCS_MSG),
            "docs359_sha256": sha(DOC359),
        },
        "m1013_failure_boundary": {
            "attempt_consumed_and_sealed": True,
            "attempt_outer_seal_file_sha256": sha(M1013_ATTEMPT / "SHA256SUMS.seal.sha256"),
            "quarantine_sealed": True,
            "quarantine_outer_seal_file_sha256": sha(M1013_QUARANTINE / "SHA256SUMS.seal.sha256"),
            "completed_gate_simulations": 0,
            "saif_files": 0,
            "retry": False,
        },
        "frontend_smoke": frontend,
        "execution_geometry": {
            "axes": ["k1", "k8", "k1x8"],
            "cases_per_axis": 5,
            "total_gate_simulations": 15,
            "fresh_compile_per_axis": True,
            "old_simv_reuse": False,
            "dut_only_saif": True,
        },
        "one_shot_and_namespace": {
            "m1022_namespace_fresh_at_review": True,
            "atomic_attempt_mkdir": True,
            "support_identity_before_attempt": True,
            "authority_before_attempt": True,
            "collision_gate_before_attempt": True,
            "automatic_retry": False,
        },
        "fault_injection": faults,
        "authorization": {
            "one_m1022_vcs_saif_attempt": True,
            "vcs_mapped_gate": True,
            "saif_generation": True,
            "pt": False,
            "ptpx": False,
            "dc": False,
            "gpu_remote": False,
        },
        "scope": {
            "clean_env_vcs_frontend_identity_smokes": 1,
            "vcs_design_compiles": 0,
            "m1022_runs": 0,
            "saif_created": False,
            "pt_runs": 0,
            "ptpx_runs": 0,
            "dc_runs": 0,
            "gpu_remote_runs": 0,
            "docs359_modified": False,
        },
        "claim_boundary": {
            "launch_chain_ready": True,
            "saif_activity": False,
            "power": False,
            "energy": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
        },
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
