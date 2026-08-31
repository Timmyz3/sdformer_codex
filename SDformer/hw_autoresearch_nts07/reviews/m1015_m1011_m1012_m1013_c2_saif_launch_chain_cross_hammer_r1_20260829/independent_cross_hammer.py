#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent M1015 cross-hammer for the repaired C2 mapped-gate SAIF launch chain.

This script is deliberately static/preflight-only.  It never invokes VCS or any
other EDA tool.  Dynamic fault injection uses a temporary clone of the runner
whose result/attempt namespaces and authority inputs are redirected to a
temporary directory; every injected case must fail before attempt creation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RELEASE = HW / "contracts/m1011_m1001_c2_mapped_gate_saif_launch_release_r2_20260829.json"
RELEASE_SHA = Path(str(RELEASE) + ".sha256")
RELEASE_OUTER = Path(str(RELEASE) + ".sha256.seal.sha256")
CONTRACT = HW / "contracts/m1001_m979_c2_mapped_gate_saif_rekey_source_contract_r1_20260829.json"
RUNNER = HW / "dc_handoff/scripts/run_m1013_m1001_c2_mapped_gate_saif_one_shot_r2.sh"
M1001 = HW / "reviews/m1001_m979_c2_mapped_gate_saif_rekey_source_receipt_r1_20260829"
M1002 = HW / "reviews/m1002_m1001_c2_mapped_gate_saif_rekey_source_hammer_r1_20260829"
M1011 = HW / "reviews/m1011_m1001_c2_mapped_gate_saif_launch_chain_source_receipt_r2_20260829"
M1012 = HW / "reviews/m1012_m1011_m1001_c2_mapped_gate_saif_release_hammer_r2_20260829"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
RESULTS = HW / "results"

EXPECTED = {
    "contract": "7afc4c093b802bdfd97aea101c803735e993c2eef57983311d3eb1a3d6bd36c6",
    "runner": "d9a7876a53c1becbba0155298b8f05aafba78dfedf42767ff298649fe13a9d14",
    "release": "7f765d317de1164fb3268d0b801886d05aa826002c8e73087f0df1dd4947ea67",
    "release_sidecar": "f720ebd01636209d37a9ee494851e3a895f51c25c08ceddd73fb5b84735947a6",
    "release_outer": "31d0f41819670d44bcd7310b089508d573bf71013054c0b4940cd44a77819760",
    "m1002_review": "e747c73b3add43e7010fc539f9f06d35f5e6e69219a9e66fc7f0e25e511045d7",
    "m1002_manifest": "019b810281f815d44d0024b89556ac7cacaea2c28885aa4ce79ead37761cc6eb",
    "m1002_outer": "d489e1cc3893e9c2a265ad5d35213e349f6eb44a5b4e2e15189711b1c82f5b85",
    "m1012_review": "437c1e3627b7195788ac3898d2fd3a6a60ecee817e4c2b4f74acad85130d8d5b",
    "m1012_manifest": "991eec62aef0e1604cab7b4edac88b7c98a0fda26b66d7988d50dbfe589b6c88",
    "m1012_outer": "b921af5dc801d3b44f669e5673493a5b24c50ed4d2b8b4b865805d3d8c33b4a8",
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
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(
        RuntimeError("nonfinite JSON: " + value)))


def verify_flat(directory: Path, review_sha: str | None = None,
                manifest_sha: str | None = None, outer_sha: str | None = None) -> dict:
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink(), "missing/symlink seal dir: " + str(directory))
    require(review.is_file() and manifest.is_file() and outer.is_file(), "incomplete seal: " + directory.name)
    if review_sha:
        require(sha(review) == review_sha, "review identity drift: " + directory.name)
    if manifest_sha:
        require(sha(manifest) == manifest_sha, "manifest identity drift: " + directory.name)
    if outer_sha:
        require(sha(outer) == outer_sha, "outer identity drift: " + directory.name)
    listed: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(None, 1)
        rel = rel.lstrip("*")
        member = directory / rel
        require(rel not in listed and member.is_file() and not member.is_symlink(),
                "sealed member missing/duplicate/symlink: " + str(member))
        require(sha(member) == digest, "sealed member drift: " + str(member))
        listed[rel] = digest
    require(outer.read_text().split() == [sha(manifest), "SHA256SUMS"],
            "outer content drift: " + directory.name)
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and not path.is_symlink() and
              path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    require(set(listed) == actual, "sealed exact-set drift: " + directory.name)
    return load_json(review)


def reseal(directory: Path) -> str:
    members = sorted(path for path in directory.rglob("*") if path.is_file() and
                     path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    manifest = directory / "SHA256SUMS"
    manifest.write_text("".join(f"{sha(path)}  {path.relative_to(directory)}\n" for path in members))
    outer = directory / "SHA256SUMS.seal.sha256"
    outer.write_text(f"{sha(manifest)}  SHA256SUMS\n")
    return sha(outer)


def sandbox_runner(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    """Clone only launch authorities and redirect every mutable namespace."""
    source_hammer = root / "m1002"
    release_hammer = root / "m1012"
    shutil.copytree(M1002, source_hammer)
    shutil.copytree(M1012, release_hammer)
    release = root / "m1011_release.json"
    shutil.copy2(RELEASE, release)
    runner = root / "runner.sh"
    result = root / "result"
    attempt = root / "attempt"
    work_pattern = root / "work.$$"
    failure_pattern = root / "failure.$$.quarantine"
    text = RUNNER.read_text()
    replacements = {
        'dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"': f'dc_root="{HW / "dc_handoff"}"',
        'hw_root="$(cd "${dc_root}/.." && pwd)"': f'hw_root="{HW}"',
        'source_hammer="${hw_root}/reviews/m1002_m1001_c2_mapped_gate_saif_rekey_source_hammer_r1_20260829"': f'source_hammer="{source_hammer}"',
        'release="${hw_root}/contracts/m1011_m1001_c2_mapped_gate_saif_launch_release_r2_20260829.json"': f'release="{release}"',
        'release_hammer="${hw_root}/reviews/m1012_m1011_m1001_c2_mapped_gate_saif_release_hammer_r2_20260829"': f'release_hammer="{release_hammer}"',
        'result="${hw_root}/results/m1013_m1001_c2_three_axis_mapped_gate_saif_r2_20260829"': f'result="{result}"',
        'attempt="${hw_root}/results/.m1013_m1001_c2_three_axis_mapped_gate_saif_attempt_consumed"': f'attempt="{attempt}"',
        'work="${hw_root}/results/.m1013_m1001_c2_three_axis_mapped_gate_saif_work.$$"': f'work="{work_pattern}"',
        'failure="${hw_root}/results/m1013_m1001_c2_three_axis_mapped_gate_saif_r2_20260829.failed_or_incomplete.$$.quarantine"': f'failure="{failure_pattern}"',
    }
    for old, new in replacements.items():
        require(text.count(old) == 1, "runner patch anchor drift: " + old)
        text = text.replace(old, new)
    runner.write_text(text)
    runner.chmod(0o755)
    # Preserve the production runner's exact self-binding semantics after the
    # temporary path redirection changes the clone's identity.
    release_data = load_json(release)
    release_data["runner_sha256"] = sha(runner)
    release.write_text(json.dumps(release_data, indent=2, sort_keys=True) + "\n")
    return runner, source_hammer, release, release_hammer, attempt


def invoke_fault(name: str, mutate, expect_fragment: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="m1015_fault_") as td:
        root = Path(td)
        runner, m1002, release, m1012, attempt = sandbox_runner(root)
        env = {"PATH": "/usr/bin:/bin",
               "M1013_EXPECTED_RUNNER_SHA256": sha(runner),
               "M1013_EXPECTED_M1002_OUTER_SHA256": sha(m1002 / "SHA256SUMS.seal.sha256"),
               "M1013_EXPECTED_M1012_OUTER_SHA256": sha(m1012 / "SHA256SUMS.seal.sha256")}
        mutate(root, runner, m1002, release, m1012, attempt, env)
        before = attempt.exists()
        proc = subprocess.run([str(runner)], text=True, capture_output=True, env=env,
                              timeout=20, check=False)
        after = attempt.exists()
        combined = proc.stdout + proc.stderr
        require(proc.returncode != 0, name + " unexpectedly passed")
        require(expect_fragment in combined, name + " wrong failure: " + combined[-500:])
        require(before == after, name + " mutated attempt namespace")
        require(not (root / "result").exists(), name + " created result")
        require(not list(root.glob("work.*")) and not list(root.glob("failure.*")),
                name + " crossed attempt/work boundary")
        return {"name": name, "return_code": proc.returncode,
                "expected_failure": expect_fragment, "attempt_created": False,
                "eda_invoked": False}


def main() -> dict:
    require(sha(CONTRACT) == EXPECTED["contract"], "actual M1001 contract SHA drift")
    require(sha(RUNNER) == EXPECTED["runner"], "M1013 runner SHA drift")
    require(sha(RELEASE) == EXPECTED["release"] and
            sha(RELEASE_SHA) == EXPECTED["release_sidecar"] and
            sha(RELEASE_OUTER) == EXPECTED["release_outer"], "M1011 release identity drift")
    require(RELEASE_SHA.read_text().split() == [EXPECTED["release"], RELEASE.name] and
            RELEASE_OUTER.read_text().split() == [EXPECTED["release_sidecar"], RELEASE_SHA.name],
            "M1011 double sidecar content drift")
    require(sha(DOC359) == EXPECTED["docs359"], "docs/359 drift")

    verify_flat(M1001)
    m1002 = verify_flat(M1002, EXPECTED["m1002_review"], EXPECTED["m1002_manifest"],
                        EXPECTED["m1002_outer"])
    verify_flat(M1011)
    m1012 = verify_flat(M1012, EXPECTED["m1012_review"], EXPECTED["m1012_manifest"],
                        EXPECTED["m1012_outer"])
    release = load_json(RELEASE)
    runner_text = RUNNER.read_text()

    require(m1002["status"] == "PASS_M1002_M1001_SOURCE_HAMMER" and m1002["p0_count"] == 0,
            "M1002 authority status drift")
    require(m1012["status"] == "PASS_M1012_M1011_M1001_RELEASE_HAMMER_R2" and
            m1012["verdict"] == "GO_ONE_M1013_VCS_SAIF_ATTEMPT_ONLY" and
            m1012["p0_count"] == 0 and m1012["p1_count"] == 0,
            "M1012 authority status drift")
    require(m1012["identity"]["m1001_contract_sha256"] == EXPECTED["contract"] and
            m1012["identity"]["m1013_runner_sha256"] == EXPECTED["runner"] and
            m1012["identity"]["m1002_outer_sha256"] == EXPECTED["m1002_outer"],
            "M1012 identity binding drift")
    require(release["status"] == "PASS_M1011_M1001_LAUNCH_RELEASE_R2" and
            release["runner_sha256"] == EXPECTED["runner"] and
            release["source_contract_sha256"] == EXPECTED["contract"] and
            release["source_hammer"]["outer_seal_file_sha256"] == EXPECTED["m1002_outer"],
            "M1011 authority content drift")
    require(release["execution"]["axes"] == ["k1", "k8", "k1x8"] and
            release["execution"]["cases_per_axis"] == 5 and
            release["execution"]["total_gate_simulations"] == 15 and
            release["execution"]["fresh_compile_per_axis"] is True and
            release["execution"]["old_simv_reuse"] is False and
            release["execution"]["dut_only_saif"] is True,
            "M1011 3-axis x 5-case execution geometry drift")
    auth = release["authorization"]
    require(auth["one_m1013_run"] is True and auth["automatic_retry"] is False and
            auth["vcs_mapped_gate"] is True and auth["saif_generation"] is True and
            all(auth[key] is False for key in ("pt", "ptpx", "dc", "gpu_remote")),
            "M1011 authorization drift")

    require(runner_text.count('for axis in k1 k8 k1x8; do') == 1 and
            runner_text.count('for case_id in 0 1 2 3 4; do') == 1,
            "runner 3-axis x 5-case loop drift")
    require(runner_text.count('"${vcs}" -full64') == 1 and
            runner_text.index('"${vcs}" -full64') < runner_text.index('for case_id in 0 1 2 3 4; do'),
            "fresh compile placement drift")
    require('power tb_m979_c2_three_axis_mapped_gate_case_saif.dut' in
            (HW / "dc_handoff/scripts/m979_c2_mapped_gate_per_case_saif.ucli.tcl").read_text(),
            "DUT-only SAIF scope drift")
    require(all(token in runner_text for token in
                ("M1013_EXPECTED_RUNNER_SHA256", "M1013_EXPECTED_M1002_OUTER_SHA256",
                 "M1013_EXPECTED_M1012_OUTER_SHA256", "mkdir \"${attempt}\"",
                 "pgrep -x vcs1", "pgrep -x vlogan", "pgrep -x dc_shell",
                 "pgrep -x fm_shell", "pgrep -x pt_shell")),
            "runner preflight/one-shot/collision gate drift")
    require(all(stale not in runner_text for stale in
                ("M1005_EXPECTED_", "m1005_m1001_c2_three_axis_mapped_gate_saif_r1")),
            "old M1005 namespace survived in repaired runner")
    require(not any((RESULTS / name).exists() for name in
                    (".m1005_m1001_c2_three_axis_mapped_gate_saif_attempt_consumed",
                     "m1005_m1001_c2_three_axis_mapped_gate_saif_r1_20260829")),
            "old M1005 attempt/result consumed")
    require(not any((RESULTS / name).exists() for name in
                    (".m1013_m1001_c2_three_axis_mapped_gate_saif_attempt_consumed",
                     "m1013_m1001_c2_three_axis_mapped_gate_saif_r2_20260829")),
            "new M1013 namespace not fresh")
    subprocess.run(["bash", "-n", str(RUNNER)], check=True, timeout=10)
    subprocess.run(["/opt/anaconda3/envs/pytorch310/bin/python3.10",
                    str(HW / "system_simulator/scripts/check_m1001_m979_c2_mapped_gate_saif_rekey_source.py"),
                    "--contract", str(CONTRACT)], check=True, stdout=subprocess.DEVNULL, timeout=30)

    faults = []
    faults.append(invoke_fault(
        "wrong_runner_sha",
        lambda root, runner, m1002d, releasef, m1012d, attempt, env:
            env.__setitem__("M1013_EXPECTED_RUNNER_SHA256", "0" * 64),
        "caller must pin exact runner SHA"))
    faults.append(invoke_fault(
        "wrong_m1002_outer",
        lambda root, runner, m1002d, releasef, m1012d, attempt, env:
            env.__setitem__("M1013_EXPECTED_M1002_OUTER_SHA256", "1" * 64),
        "outer seal mismatch"))

    def wrong_status(root, runner, m1002d, releasef, m1012d, attempt, env):
        data = load_json(releasef)
        data["status"] = "STOP_INJECTED_WRONG_STATUS"
        releasef.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    faults.append(invoke_fault("wrong_release_status", wrong_status,
                               "release chain content mismatch"))

    def occupied_namespace(root, runner, m1002d, releasef, m1012d, attempt, env):
        attempt.mkdir()
        (attempt / "SENTINEL_DO_NOT_TOUCH").write_text("injected occupied namespace\n")

    faults.append(invoke_fault("occupied_attempt_namespace", occupied_namespace,
                               "result/attempt/work collision"))

    # The collision gate is dynamically exercised with a harmless process whose
    # comm name is vcs1.  The temporary runner must stop before attempt creation.
    with tempfile.TemporaryDirectory(prefix="m1015_collision_") as td:
        root = Path(td)
        runner, m1002d, releasef, m1012d, attempt = sandbox_runner(root)
        fake = root / "vcs1"
        fake.symlink_to("/usr/bin/sleep")
        blocker = subprocess.Popen([str(fake), "10"])
        try:
            time.sleep(0.1)
            env = {"PATH": "/usr/bin:/bin",
                   "M1013_EXPECTED_RUNNER_SHA256": sha(runner),
                   "M1013_EXPECTED_M1002_OUTER_SHA256": sha(m1002d / "SHA256SUMS.seal.sha256"),
                   "M1013_EXPECTED_M1012_OUTER_SHA256": sha(m1012d / "SHA256SUMS.seal.sha256")}
            proc = subprocess.run([str(runner)], text=True, capture_output=True, env=env,
                                  timeout=20, check=False)
            combined = proc.stdout + proc.stderr
            require(proc.returncode != 0 and "VCS/DC/FM/PT collision" in combined,
                    "collision gate did not fail closed: " + combined[-500:])
            require(not attempt.exists(), "collision gate consumed attempt")
            faults.append({"name": "active_vcs1_collision", "return_code": proc.returncode,
                           "expected_failure": "VCS/DC/FM/PT collision",
                           "attempt_created": False, "eda_invoked": False})
        finally:
            blocker.terminate()
            try:
                blocker.wait(timeout=2)
            except subprocess.TimeoutExpired:
                blocker.kill()
                blocker.wait(timeout=2)

    return {
        "schema": "m1015_m1011_m1012_m1013_c2_saif_launch_chain_cross_hammer_r1_v1",
        "date": "2026-08-29",
        "milestone": "M1015",
        "status": "PASS_M1015_M1011_M1012_M1013_C2_SAIF_LAUNCH_CHAIN_CROSS_HAMMER",
        "verdict": "GO_ONE_M1013_VCS_SAIF_ATTEMPT_ONLY",
        "score_out_of_100": 100,
        "p0_count": 0,
        "p1_count": 0,
        "p2_count": 0,
        "identity": {
            "m1001_contract_sha256": sha(CONTRACT),
            "m1013_runner_sha256": sha(RUNNER),
            "m1011_release_sha256": sha(RELEASE),
            "m1012_status": m1012["status"],
            "m1012_review_sha256": sha(M1012 / "review.json"),
            "m1012_manifest_sha256": sha(M1012 / "SHA256SUMS"),
            "m1012_outer_seal_file_sha256": sha(M1012 / "SHA256SUMS.seal.sha256"),
            "m1002_outer_seal_file_sha256": sha(M1002 / "SHA256SUMS.seal.sha256"),
            "docs359_sha256": sha(DOC359),
        },
        "execution_geometry": {
            "axes": ["k1", "k8", "k1x8"],
            "cases_per_axis": 5,
            "total_gate_simulations": 15,
            "fresh_compile_per_axis": True,
            "old_simv_reuse": False,
            "dut_only_saif": True,
        },
        "one_shot_and_namespace": {
            "old_m1005_attempt_consumed": False,
            "old_m1005_result_created": False,
            "m1013_namespace_fresh_at_review": True,
            "atomic_attempt_mkdir": True,
            "automatic_retry": False,
            "collision_gate": True,
        },
        "fault_injection": faults,
        "authorization": {
            "vcs_mapped_gate": True,
            "saif_generation": True,
            "pt": False,
            "ptpx": False,
            "dc": False,
            "gpu_remote": False,
        },
        "scope": {
            "static_and_sandboxed_preflight_only": True,
            "m1013_runs": 0,
            "eda_runs": 0,
            "saif_created": False,
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
    result = main()
    (HERE / "review.json").write_text(json.dumps(result, indent=2, sort_keys=True,
                                                  allow_nan=False) + "\n")
    checks = [
        "PASS exact M1001 contract SHA " + result["identity"]["m1001_contract_sha256"],
        "PASS exact M1013 runner SHA " + result["identity"]["m1013_runner_sha256"],
        "PASS M1012 status " + result["identity"]["m1012_status"],
        "PASS M1012 outer " + result["identity"]["m1012_outer_seal_file_sha256"],
        "PASS geometry k1/k8/k1x8 x five cases = 15",
        "PASS fresh compile per axis; old simv reuse false",
        "PASS UCLI scopes activity to tb_m979_c2_three_axis_mapped_gate_case_saif.dut",
        "PASS old M1005 result/attempt absent; M1013 namespace fresh",
        "PASS one-shot atomic attempt mkdir and collision gate",
    ]
    checks.extend("PASS fault {} rc={} before_attempt=true".format(
        fault["name"], fault["return_code"]) for fault in result["fault_injection"])
    checks.extend([
        "PASS authorization VCS+SAIF only; PT/PTPX/DC/GPU false",
        "PASS docs/359 SHA " + result["identity"]["docs359_sha256"],
        "PASS no M1013/EDA execution and no SAIF generated by M1015",
    ])
    (HERE / "mechanical_checks.txt").write_text("\n".join(checks) + "\n")
    (HERE / "review.md").write_text(
        "# M1015 repaired C2 SAIF launch-chain cross-hammer\n\n"
        "**Verdict: GO for exactly one M1013 mapped-gate VCS+SAIF attempt.** "
        "Score 100/100; P0/P1/P2 = 0/0/0.\n\n"
        "The independently recomputed identities are the actual M1001 contract "
        "`7afc4c093b802bdfd97aea101c803735e993c2eef57983311d3eb1a3d6bd36c6` "
        "and repaired M1013 runner "
        "`d9a7876a53c1becbba0155298b8f05aafba78dfedf42767ff298649fe13a9d14`. "
        "M1012 has status `PASS_M1012_M1011_M1001_RELEASE_HAMMER_R2` and outer "
        "seal `b921af5dc801d3b44f669e5673493a5b24c50ed4d2b8b4b865805d3d8c33b4a8`.\n\n"
        "The runner performs a fresh mapped-netlist compile for each of K1, K8, "
        "and equal-bandwidth K1x8, then runs five frozen cases per axis (15 gate "
        "simulations). UCLI activity is DUT-only. The old M1005 attempt/result and "
        "the new M1013 attempt/result namespaces were absent at review time.\n\n"
        "Sandboxed fault injection changed runner SHA, authority outer seal, release "
        "status, and attempt namespace independently; every case returned rc=3 before "
        "attempt creation. A harmless process named `vcs1` also exercised the collision "
        "gate and stopped before attempt creation. No M1013 runner or EDA executable "
        "was launched by this review.\n\n"
        "Authorization is limited to one VCS mapped-gate plus SAIF attempt. This does "
        "not authorize PT, PTPX, DC, remote GPU work, power, energy, system speedup, "
        "or paper-ready PPA claims.\n")
    (HERE / "RUN_COMPLETE.txt").write_text(
        "PASS_M1015_M1011_M1012_M1013_C2_SAIF_LAUNCH_CHAIN_CROSS_HAMMER\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
