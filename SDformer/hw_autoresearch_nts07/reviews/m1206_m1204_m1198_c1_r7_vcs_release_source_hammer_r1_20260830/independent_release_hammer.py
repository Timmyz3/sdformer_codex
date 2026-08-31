#!/usr/bin/env python3
"""Independent, read-only M1206 hammer for the M1204 C1/R7 release source.

This checker deliberately does not invoke VCS, simv, any license client, GPU,
or network operation.  It also proves whether the proposed recursively sealed
release-hammer identity is constructible before any attempt token may exist.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

HW = Path(__file__).resolve().parents[2]
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1204_m1198r7_m1162_c1_common_charge_protocol_exact_sha_r7.sh"
CHECKER = HW / "verif_m1204_c1_r7_vcs_release/static_check_m1204_c1_r7_vcs_release_source.py"
R7_CHECKER = HW / "verif_m1198r7_c1_common_charge_protocol/static_check_m1198r7_m1162_vcs_source.py"
SOURCE = HW / "contracts/m1204_m1201_m1198_c1_r7_vcs_launcher_source_contract_r1_20260830.json"
RELEASE = HW / "contracts/m1204_m1201_m1198_c1_r7_vcs_launch_release_r1_20260830.json"
M1198 = HW / "contracts/m1198_m1194_m1193_m1162_c1_r7_source_gate_repair_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1204_m1201_m1198_c1_r7_vcs_release_author_receipt_r1_20260830"
R7_AUTHOR = HW / "reviews/m1198_m1194_c1_r7_source_gate_repair_author_receipt_r1_20260830"
R7_HAMMER = HW / "reviews/m1201_m1198_c1_r7_source_gate_repair_hammer_r1_20260830"
TB = HW / "verif_m1193r6_c1_common_charge_protocol/tb_m1193r6_m1162_common_charge_protocol_unit_delay_r6.sv"
SVA = HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1198r7_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
ATTEMPT = HW / "results/.m1204_m1198r7_m1162_c1_common_charge_protocol_vcs_r7_attempt_consumed"
RESULT = HW / "results/m1204_m1198r7_m1162_c1_common_charge_protocol_unit_delay_vcs_r7_20260830"

EXPECTED = {
    RUNNER: "323d772fa6c2a2da05d52d9199019e620f1865d995bd13232d5a761f2df6f7ef",
    CHECKER: "d0fa10804209688af1984eb23e1793d1ea5b0c2aabca6fbb8bf8ad025abdae1e",
    R7_CHECKER: "b1cfb957d5c4fc518d46980040afa61288eb7dcaa79fa5e6c45e25b097094795",
    SOURCE: "a16e16336af0c3db475d2dcd6a725b6adec0ae82623c7d2fc89e84398032f64d",
    RELEASE: "b4e2e494aca2653ff88435ca92e96c919fd8b869fa99b1077dbbc8e717e927f4",
    M1198: "44c5a3add48ef74ef0698f81f20fef417989c17b74df3e1d366cf404b7ce5488",
    TB: "0fcc2138ef5d716735eea01dee25a148a5223b1d6adf1e3b2fa464341fbf1345",
    SVA: "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    FILELIST: "444ff65d575c6e897f9d459689f323290f16eb89c962c91b395964c7850fcbfa",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

checks = 0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def req(ok: bool, msg: str) -> None:
    global checks
    checks += 1
    if not ok:
        raise AssertionError(msg)


def verify_seal(directory: Path) -> None:
    req(directory.is_dir() and not directory.is_symlink(), f"sealed directory {directory}")
    sums = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    req(outer.read_text().split() == [sha(sums), "SHA256SUMS"], f"outer seal {directory}")
    listed: dict[str, str] = {}
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        req(name not in listed and not Path(name).is_absolute() and ".." not in Path(name).parts,
            "safe unique manifest member")
        listed[name] = digest
    actual: set[str] = set()
    for root, dirs, files in os.walk(directory, followlinks=False):
        base = Path(root)
        dirs[:] = [name for name in dirs if not (base / name).is_symlink()]
        for name in files:
            path = base / name
            rel = path.relative_to(directory).as_posix()
            if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"} or path.is_symlink():
                continue
            if stat.S_ISREG(path.lstat().st_mode):
                actual.add(rel)
    req(actual == set(listed), f"complete recursive manifest {directory}")
    for name, digest in listed.items():
        req(sha(directory / name) == digest, f"sealed member drift {directory}/{name}")


def verify_sidecar(path: Path) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    req(side.read_text().split() == [sha(path), path.name], f"sidecar {path.name}")
    req(outer.read_text().split() == [sha(side), side.name], f"outer sidecar {path.name}")


def structural_gate(runner: str, source: dict, release: dict) -> None:
    """Independent release semantics used for in-memory mutation rejection."""
    required_tokens = [
        "M1204_EXPECTED_RELEASE_SHA256",
        "M1204_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256",
        "M1204_EXPECTED_RELEASE_HAMMER_OUTER_SHA256",
        "PASS_M1206_M1204_C1_R7_VCS_RELEASE_HAMMER__AUTHORIZE_ONE_LAUNCH",
        "COVERAGE_M1193R6_PROTOCOL weight_first=1",
        "COVERAGE_M1193R6_RESETS_ATTACKS reset_partial=1",
        "COVERAGE_M1193R6_SERVICE_ASSUMPTIONS weight_payload_mutation=1",
        "COVERAGE_M1193R6_FROZEN_M935 normal_issues=2",
        "directed_random=24 protocol_attacks=7 service_assumption_attacks=2",
        "service_skew_isolated=1 reachable_core_ready_force=0 boundary_fault=0 core_fault=0",
        "SNPSLMD_LICENSE_FILE", "+define+UNIT_DELAY", "dc_shell", "pt_shell", "fm_shell",
    ]
    if any(token not in runner for token in required_tokens):
        raise ValueError("required runner token")
    if runner.count('"${VCS_BIN}" -full64') != 1 or runner.count('./simv -no_save') != 1:
        raise ValueError("one compile and one simulation")
    hammer = runner.index('verify_recursive_seal "${RELEASE_HAMMER}"')
    attempt = runner.index('/bin/mkdir -- "${ATTEMPT}"')
    vcs = runner.index('"${VCS_BIN}" -full64')
    if not hammer < attempt < vcs:
        raise ValueError("hammer/attempt/VCS ordering")
    if source["identity"]["runner_sha256"] != EXPECTED[RUNNER]:
        raise ValueError("source runner identity")
    if release["identity"]["runner_sha256"] != EXPECTED[RUNNER]:
        raise ValueError("release runner identity")
    if release["identity"]["source_contract_sha256"] != EXPECTED[SOURCE]:
        raise ValueError("source contract identity")
    if release["unique_attempt"]["attempt_path"] != str(ATTEMPT.relative_to(HW)):
        raise ValueError("attempt namespace")
    if release["unique_attempt"]["result_path"] != str(RESULT.relative_to(HW)):
        raise ValueError("result namespace")
    if release["authorization"] != {"vcs_compiles": 1, "simv_runs": 1, "all_other_eda_runs": 0}:
        raise ValueError("authorization")
    if release["required_regression"] != {
        "assertions": 16, "covers": 6, "protocol_attacks": 7,
        "service_assumption_attacks": 2, "deterministic_legal_transactions": 24,
        "legal_masks_clear": 29, "request_attack_windows": 2, "reset_states": 3,
        "minimum_completed_issue_ii": 2, "normal_m935_rows": 1,
        "normal_m935_tasks": 1, "service_skew_isolated": True,
        "reachable_core_ready_force": False, "boundary_fault": False, "core_fault": False,
    }:
        raise ValueError("regression identity")
    for doc in (source, release):
        for key in ("functional_vcs_verified", "timing_verified", "cycles_measured", "speedup",
                    "ppa", "power", "energy", "system_speedup", "paper_citable", "headline"):
            if doc["claim_boundary"][key] is not False:
                raise ValueError("claim boundary")


def mutate_and_reject(runner: str, source: dict, release: dict) -> int:
    cases: list[tuple[str, str, dict, dict]] = []
    def clone(obj: dict) -> dict:
        return json.loads(json.dumps(obj))
    s = clone(source); s["identity"]["runner_sha256"] = "0" * 64
    cases.append(("source runner SHA", runner, s, clone(release)))
    r = clone(release); r["identity"]["runner_sha256"] = "0" * 64
    cases.append(("release runner SHA", runner, clone(source), r))
    r = clone(release); r["identity"]["source_contract_sha256"] = "0" * 64
    cases.append(("source contract SHA", runner, clone(source), r))
    r = clone(release); r["unique_attempt"]["attempt_path"] += ".alias"
    cases.append(("attempt alias", runner, clone(source), r))
    r = clone(release); r["unique_attempt"]["result_path"] += ".alias"
    cases.append(("result alias", runner, clone(source), r))
    cases.append(("remove VCS", runner.replace('"${VCS_BIN}" -full64', '"${VCS_BIN}" -noop', 1), clone(source), clone(release)))
    cases.append(("duplicate VCS", runner + '\n"${VCS_BIN}" -full64\n', clone(source), clone(release)))
    cases.append(("remove simv", runner.replace('./simv -no_save', './simv -disabled', 1), clone(source), clone(release)))
    cases.append(("PASS token", runner.replace("directed_random=24", "directed_random=23"), clone(source), clone(release)))
    cases.append(("coverage token", runner.replace("normal_issues=2", "normal_issues=1"), clone(source), clone(release)))
    cases.append(("attack oracle", runner.replace("boundary_fault=0 core_fault=0", "boundary_fault=1 core_fault=0"), clone(source), clone(release)))
    cases.append(("UNIT_DELAY", runner.replace("+define+UNIT_DELAY", "+define+BROKEN_DELAY"), clone(source), clone(release)))
    cases.append(("license guard", runner.replace("SNPSLMD_LICENSE_FILE", "REMOVED_LICENSE_GUARD"), clone(source), clone(release)))
    cases.append(("same UID collision", runner.replace("dc_shell", "removed_dc", 1), clone(source), clone(release)))
    r = clone(release); r["required_regression"]["assertions"] = 15
    cases.append(("assertion count", runner, clone(source), r))
    r = clone(release); r["claim_boundary"]["speedup"] = True
    cases.append(("claim inflation", runner, clone(source), r))
    rejected = 0
    for name, text, sdoc, rdoc in cases:
        try:
            structural_gate(text, sdoc, rdoc)
        except (ValueError, KeyError, AssertionError):
            rejected += 1
        else:
            raise AssertionError("mutation accepted: " + name)
    return rejected


def main() -> None:
    for path, digest in EXPECTED.items():
        req(path.is_file() and not path.is_symlink() and sha(path) == digest,
            f"identity {path}")
    for directory in (AUTHOR, R7_AUTHOR, R7_HAMMER):
        verify_seal(directory)
    for path in (SOURCE, RELEASE, M1198):
        verify_sidecar(path)
    req(not ATTEMPT.exists() and not RESULT.exists(), "fresh attempt/result")
    req(not list((HW / "results").glob(".m1204_m1198r7_m1162_c1_common_charge_protocol_vcs_r7_work.*")),
        "fresh work namespace")
    req(not list((HW / "results").glob("m1204_m1198r7_m1162_c1_common_charge_protocol_unit_delay_vcs_r7_20260830.failed_or_incomplete.*")),
        "fresh quarantine namespace")
    source = json.loads(SOURCE.read_text())
    release = json.loads(RELEASE.read_text())
    runner = RUNNER.read_text()
    structural_gate(runner, source, release)
    req(subprocess.run(["/bin/bash", "-n", str(RUNNER)], check=False).returncode == 0,
        "runner shell parse")
    author_check = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", "-I", str(CHECKER)],
        check=False, capture_output=True, text=True)
    req(author_check.returncode == 0 and "PASS_M1204_RELEASE_SOURCE_ONLY" in author_check.stdout,
        "author checker")
    r7_check = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", "-I", str(R7_CHECKER)],
        check=False, capture_output=True, text=True)
    req(r7_check.returncode == 0 and "PASS_R7_SOURCE_ONLY" in r7_check.stdout,
        "R7 checker")
    rejected = mutate_and_reject(runner, source, release)
    req(rejected == 16, "all mutations rejected")

    # P0 constructibility proof.  The release runtime requires the hammer's
    # review.json to contain hashes of SHA256SUMS and its outer seal.  The same
    # runtime first requires a complete recursive seal, so review.json must be
    # in SHA256SUMS.  Therefore changing either embedded hash changes
    # review.json, which changes SHA256SUMS, which changes both requested
    # embedded hashes.  Producing this cryptographic fixed point is not a
    # reproducible sealing procedure and no authoring algorithm is supplied.
    req('assert sha(manifest)==x[\'identity\'][\'hammer_manifest_sha256\']' in runner,
        "runtime manifest self-pin present")
    req('assert sha(outer)==x[\'identity\'][\'hammer_outer_seal_file_sha256\']' in runner,
        "runtime outer self-pin present")
    req('verify_recursive_seal "${RELEASE_HAMMER}"' in runner,
        "recursive membership required before runtime pins")

    print(json.dumps({
        "schema": "m1206_m1204_m1198_c1_r7_vcs_release_source_hammer_mechanical_r1_v1",
        "status": "FAIL_UNCONSTRUCTIBLE_RECURSIVE_HAMMER_IDENTITY__DO_NOT_LAUNCH",
        "checks_passed_before_p0": checks,
        "mutations_rejected": rejected,
        "identities_and_prior_seals_valid": True,
        "fresh_namespace": True,
        "runner_shell_parse": True,
        "author_and_r7_static_checkers_pass": True,
        "p0": "review.json embeds SHA256SUMS and outer-seal hashes while review.json must itself be in SHA256SUMS",
        "vcs_runs": 0,
        "simv_runs": 0,
        "all_eda_runs": 0,
        "gpu_runs": 0,
        "network_runs": 0,
        "docs359_sha256": sha(DOCS359),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
