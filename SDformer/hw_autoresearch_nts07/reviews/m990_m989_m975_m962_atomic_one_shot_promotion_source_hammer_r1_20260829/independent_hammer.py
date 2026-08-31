#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent source/concurrency hammer for inert M989; no promotion or EDA."""

import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import time


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SCRIPT = HW / "dc_handoff/scripts/promote_m989_m962_quarantine_atomic_one_shot_copy_only_r1.sh"
CONTRACT = HW / "contracts/m989_m975_m962_atomic_one_shot_copy_only_promotion_source_contract_r1_20260829.json"
M989 = HW / "reviews/m989_m975_m962_atomic_one_shot_copy_only_promotion_source_r1_20260829"
SOURCE = HW / "dc_handoff/runs/m962_m935_three_stage_match_macro_aware_dc_3p000ns_r1_20260829.failed_or_incomplete.3868703.quarantine"
M975_CONTRACT = HW / "contracts/m975_m962_quarantine_copy_only_canonical_recovery_source_contract_r1_20260829.json"
M975 = HW / "reviews/m975_m962_quarantine_rc9_forensic_and_promotion_source_hammer_r1_20260829"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
RUNS = HW / "dc_handoff/runs"

EXPECTED = {
    "script": "7b63668f5fb68ac8d60acf4e43925313ab1c0bdc84caeefcbfb0e238871c4be9",
    "script_sidecar": "f164608c758c62198be918871b10fb2225730e8fb36279f922590c453bc8e575",
    "script_outer": "0489b3a7a69cde6f1d868f56346de2a214b950b6b778f3f136a846b233da18e2",
    "contract": "9be7b8640f3f1674de5e603992cf69c4f091000c1c09c9bcfa4f67745a037030",
    "contract_sidecar": "7e3e3a22b33db619254856eb3cfe2e692ff2bb3b8584677ca06d2dd18e5fbb40",
    "contract_outer": "6a8a978befe6348182a3d2a6e90f57fda7a13574f041bd262553aa21f35b890c",
    "m989_review": "801abbe4d19a544b30d0b57f05c28a1f99d5a0fbe594261861d4ea3f05d2e6d3",
    "m989_manifest": "4af382fcc6c6b9ced71a2d34dad2f037221530e09e0c34801930259780221edf",
    "m989_outer": "ac4a58e4516f7e7ceabad14b42077d148855e4c3f4f1d2a20533b887117fdaf7",
    "m989_attack": "e33510ebc793853d829871b5ebe27c322bb56feceda96e4de6b8271be0f8b051",
    "m989_attack_result": "0d12b22dbd348735de8dff53c20d4c48505316e6ae8f6affdac68ac25fd28bc0",
    "source_manifest": "9a1649638c0c2aa7b533fdb16cd763c87e6280dfc5a3c291240818cf1022eafe",
    "source_outer": "a213df2a38ff231f9d0dbd78c379ef13b3731caf3b5335c37d6d17bf20927997",
    "m975_contract": "b46e1dcc6e9691e828fafb811e22e7e523d5b226a28955c9f2a1366424c6211a",
    "m975_review": "5819ac17810e3f3680e1f5cc2320f3f91aa742aa0086bed69ba5a9fd9ad2e429",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_file_seal(payload, sidecar_sha, outer_sha):
    sidecar = Path(str(payload) + ".sha256")
    outer = Path(str(payload) + ".sha256.seal.sha256")
    require(all(path.is_file() and not path.is_symlink()
                for path in (payload, sidecar, outer)), "file seal missing/symlink")
    require(sha(sidecar) == sidecar_sha and sha(outer) == outer_sha,
            "file seal identity drift")
    require(sidecar.read_text().split() == [sha(payload), payload.name],
            "sidecar content drift")
    require(outer.read_text().split() == [sha(sidecar), sidecar.name],
            "outer sidecar content drift")


def verify_directory(directory, manifest_sha, outer_sha):
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink(), "directory missing/symlink")
    require(manifest.is_file() and outer.is_file() and
            not manifest.is_symlink() and not outer.is_symlink(), "directory seal missing")
    require(sha(manifest) == manifest_sha and sha(outer) == outer_sha,
            "directory seal identity drift")
    require(outer.read_text().split() == [manifest_sha, "SHA256SUMS"],
            "directory outer content drift")
    listed = {}
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(None, 1)
        rel = rel.lstrip("*")
        require(rel not in listed and ".." not in Path(rel).parts,
                "unsafe/duplicate manifest path")
        member = directory / rel
        require(member.is_file() and not member.is_symlink() and sha(member) == digest,
                "manifest member drift: " + rel)
        listed[rel] = digest
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and not path.is_symlink() and
              path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    require(set(listed) == actual, "recursive exact-set drift")
    require(not [path for path in directory.rglob("*") if path.is_symlink()],
            "recursive symlink found")
    return {"entries": len(listed), "manifest_sha256": sha(manifest),
            "outer_seal_file_sha256": sha(outer)}


def contender(root, gate, results, hold):
    root = Path(root); lock = root / "lock"; attempt = root / "attempt"
    gate.wait()
    try:
        os.mkdir(lock)
    except FileExistsError:
        results.put("STOP_LOCK")
        return
    try:
        try:
            os.mkdir(attempt)
        except FileExistsError:
            results.put("STOP_ATTEMPT")
            return
        (root / "copy_started").write_text("one\n")
        time.sleep(hold)
        results.put("COPY_WINNER")
    finally:
        os.rmdir(lock)


def wave(root, workers, hold):
    gate = mp.Barrier(workers); results = mp.Queue()
    jobs = [mp.Process(target=contender, args=(str(root), gate, results, hold))
            for _ in range(workers)]
    for job in jobs: job.start()
    for job in jobs:
        job.join(10)
        require(not job.is_alive() and job.exitcode == 0, "worker failed/hung")
    output = []
    for _ in jobs:
        try: output.append(results.get(timeout=2))
        except queue.Empty as error: raise RuntimeError("missing worker result") from error
    return output


def independent_state_attacks():
    with tempfile.TemporaryDirectory(prefix="m990_state_hammer_") as temporary:
        root = Path(temporary)
        first = wave(root, 24, 0.06)
        second = wave(root, 24, 0.0)
        require(first.count("COPY_WINNER") == 1 and
                sum(item.startswith("STOP_") for item in first) == 23,
                "first concurrency wave drift")
        require(second.count("COPY_WINNER") == 0 and
                sum(item.startswith("STOP_") for item in second) == 24,
                "permanent attempt did not block second wave")

        # Crash immediately after canonical attempt mkdir: lock cleanup may run,
        # but the permanent attempt remains and the next owner cannot copy.
        crash = root / "crash"; crash.mkdir()
        lock = crash / "lock"; attempt = crash / "attempt"
        os.mkdir(lock); os.mkdir(attempt); os.rmdir(lock)
        os.mkdir(lock)
        retry_blocked = False
        try: os.mkdir(attempt)
        except FileExistsError: retry_blocked = True
        finally: os.rmdir(lock)
        require(retry_blocked, "post-attempt crash retry was not blocked")

        # Literal target publication cannot nest WORK below an existing target.
        work = root / "work"; target = root / "target"
        work.mkdir(); target.mkdir(); (work / "payload").write_text("x\n")
        mv_existing = subprocess.run(
            ["/usr/bin/mv", "-T", "--", str(work), str(target)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        no_nested_work = not (target / work.name).exists()
        require(no_nested_work, "mv -T nested work under target")

        return {
            "first_wave_workers": 24,
            "first_wave_copy_winners": first.count("COPY_WINNER"),
            "first_wave_stopped_before_copy": sum(x.startswith("STOP_") for x in first),
            "second_wave_workers": 24,
            "second_wave_copy_winners": second.count("COPY_WINNER"),
            "second_wave_stopped_before_copy": sum(x.startswith("STOP_") for x in second),
            "crash_after_canonical_attempt_blocks_retry": retry_blocked,
            "mv_T_existing_target_return_code": mv_existing.returncode,
            "mv_T_no_nesting": no_nested_work,
        }


def main():
    require(sha(SCRIPT) == EXPECTED["script"], "script SHA drift")
    require(sha(CONTRACT) == EXPECTED["contract"], "contract SHA drift")
    require(sha(M989 / "review.json") == EXPECTED["m989_review"], "M989 review drift")
    require(sha(M989 / "static_concurrency_attack.py") == EXPECTED["m989_attack"],
            "M989 attack script drift")
    require(sha(M989 / "static_concurrency_attack_result.json") ==
            EXPECTED["m989_attack_result"], "M989 attack result drift")
    require(sha(M975_CONTRACT) == EXPECTED["m975_contract"] and
            sha(M975 / "review.json") == EXPECTED["m975_review"],
            "M975 authority drift")
    require(sha(DOC359) == EXPECTED["docs359"], "docs359 drift")
    verify_file_seal(SCRIPT, EXPECTED["script_sidecar"], EXPECTED["script_outer"])
    verify_file_seal(CONTRACT, EXPECTED["contract_sidecar"], EXPECTED["contract_outer"])
    m989_seal = verify_directory(M989, EXPECTED["m989_manifest"], EXPECTED["m989_outer"])
    source_seal = verify_directory(SOURCE, EXPECTED["source_manifest"], EXPECTED["source_outer"])

    contract = json.loads(CONTRACT.read_text())
    source_review = json.loads((M989 / "review.json").read_text())
    require(contract["status"] == "SOURCE_READY__M989_PROMOTION_NOT_AUTHORIZED_NOW" and
            contract["authorization"] == {"promotion_runs_now": 0,
                                           "future_copy_only_promotions_max": 1,
                                           "eda_runs": 0}, "contract scope drift")
    require(source_review["status"] ==
            "PASS_M989_ATOMIC_ONE_SHOT_COPY_ONLY_PROMOTION_SOURCE__FUTURE_M990_HAMMER_REQUIRED" and
            source_review["scope"]["promotion_executed"] is False,
            "M989 source review scope drift")

    author_attack = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", "-I",
         str(M989 / "static_concurrency_attack.py")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        check=False, timeout=30)
    require(author_attack.returncode == 0 and
            "PASS_M989_STATIC_CONCURRENCY_ATTACKS" in author_attack.stdout,
            "M989 author attack failed")
    require(subprocess.run(["/usr/bin/bash", "-n", str(SCRIPT)], check=False).returncode == 0,
            "promotion script bash syntax failed")

    names = ["m993_m989_m962_m935_macro_aware_dc_recovered_canonical_r1_20260829",
             ".m993_m989_m962_copy_promotion_launch_lock",
             ".m993_m989_m962_copy_promotion_attempt_consumed",
             ".m993_m989_m962_copy_promotion_work",
             "m993_m989_m962_copy_promotion_failed_or_incomplete.quarantine"]
    before = {name: (RUNS / name).exists() or (RUNS / name).is_symlink() for name in names}
    require(not any(before.values()), "M993 namespace not fresh before inert check")
    inert = subprocess.run(
        [str(SCRIPT)], cwd=str(HW.parent),
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        check=False, timeout=15)
    after = {name: (RUNS / name).exists() or (RUNS / name).is_symlink() for name in names}
    require(inert.returncode == 3 and not any(after.values()),
            "inert source touched M993 namespace")

    source = SCRIPT.read_text()
    ordered = [
        'verify_dir_seal "$(dirname -- "${M992_HAMMER}")"',
        'if ! mkdir -- "${LOCK}"; then',
        'if ! mkdir -- "${ATTEMPT}"; then',
        'seal_dir "${ATTEMPT}"',
        'mkdir -- "${WORK}"',
        'cp -a --no-dereference "${SOURCE}/." "${WORK}/original_quarantine/"',
        'seal_dir "${WORK}"',
        '[[ ! -e "${TARGET}" ]] || exit 6',
        'mv -T -- "${WORK}" "${TARGET}"',
    ]
    positions = []; cursor = 0
    for token in ordered:
        position = source.index(token, cursor); positions.append(position)
        cursor = position + len(token)
    require(positions == sorted(positions), "state order drift")
    require('WORK="${TARGET}.copy_work.$$"' not in source and
            'WORK="${HW_ROOT}/dc_handoff/runs/.m993_m989_m962_copy_promotion_work"' in source,
            "fixed work identity drift")
    require("rm -rf" not in source and not any(
            line.strip().startswith(("rm ", "rmdir ")) and "ATTEMPT" in line
            for line in source.splitlines()), "attempt can be removed")
    trap = source[source.index("on_exit() {"):source.index("trap on_exit EXIT INT TERM")]
    require('mv -T -- "${WORK}" "${FAILQ}"' in trap and
            'mv -T -- "${WORK}" "${TARGET}"' not in trap,
            "failure trap target pollution")

    state = independent_state_attacks()
    nested_seal_boundary = (
        "find -P recursively excludes every nested file named SHA256SUMS or "
        "SHA256SUMS.seal.sha256, and final TARGET verification does not separately "
        "verify TARGET/original_quarantine. Actual payload files remain covered by "
        "the root manifest and expected inner hashes remain in root-covered provenance."
    )
    return {
        "schema": "m990_m989_atomic_one_shot_copy_only_promotion_source_hammer_v1",
        "status": "PASS_M990_M989_COPY_ONLY_PROMOTION_SOURCE_HAMMER",
        "verdict": "GO_AUTHOR_M991_RELEASE_ONLY",
        "score_out_of_100": 96,
        "p0_count": 0, "p1_count": 0, "p2_count": 2,
        "pins": {"promotion_script_sha256": EXPECTED["script"],
                 "source_contract_sha256": EXPECTED["contract"],
                 "m989_review_sha256": EXPECTED["m989_review"],
                 "m975_contract_sha256": EXPECTED["m975_contract"],
                 "m975_review_sha256": EXPECTED["m975_review"],
                 "docs359_sha256": EXPECTED["docs359"]},
        "source_seals": {"m989": m989_seal, "m962_quarantine": source_seal},
        "state_attacks": state,
        "positive": {
            "m987_concurrent_nesting_p0_repaired": True,
            "fixed_atomic_launch_lock": True,
            "canonical_attempt_consumed_before_copy": True,
            "attempt_is_permanent": True,
            "fixed_work_identity": True,
            "target_rechecked_after_complete_seal": True,
            "mv_T_literal_publication": True,
            "failure_trap_does_not_write_target": True,
            "production_promotion_executed": False,
            "eda_runs": 0,
        },
        "p2": [
            {"id": "P2_SAME_UID_NONCOOPERATING_MUTATION_OUT_OF_SCOPE",
             "finding": "The protocol serializes cooperating invocations; it is not a same-UID hostile-process defense."},
            {"id": "P2_NESTED_SEAL_FILES_NOT_ROOT_MANIFEST_MEMBERS",
             "finding": nested_seal_boundary,
             "remediation": "At result hammer time, separately verify original_quarantine or include nested seal files in the outer root manifest."},
        ],
        "decision": {"concurrency_protocol_admitted": True,
                     "m991_release_authoring_authorized": True,
                     "m993_execution_authorized_now": False},
        "scope": {"source_only": True, "promotion_executed": False,
                  "eda_runs": 0, "docs359_modified": False},
        "claim_boundary": {"setup_area_promoted": False, "paper_citable": False,
                           "speedup": False, "system_speedup": False,
                           "paper_ppa_ready": False},
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
