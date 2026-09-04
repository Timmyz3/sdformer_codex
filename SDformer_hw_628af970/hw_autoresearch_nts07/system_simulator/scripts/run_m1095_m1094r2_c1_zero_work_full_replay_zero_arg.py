#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1095 zero-argument, hardcoded-authority C1 CPU replay launcher.

SOURCE ONLY.  M1098 must independently hammer this exact file and issue a
separate launch release before it may be executed.  A successful execution
remains raw CPU-model evidence pending M1099 result hammer.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from typing import Any


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
PYTHON_SHA = "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
ENGINE = HERE / "execute_m1094_m1086_c1_zero_work_exact_1rw_full_replay_one_shot.py"
ENGINE_SHA = "c8808c0d4cf37a8f279afa128e089c08af3718606061658db8f2047c198c824a"
NON_LAUNCH_STUB = HERE / "run_m1094_m1086_c1_zero_work_exact_1rw_full_replay_one_shot.sh"
NON_LAUNCH_STUB_SHA = "745b6e112d0e33457a64a6411b1563afa58418a3f9c175039ebc9ecadb902e10"
M1094_CONTRACT = HW / "contracts/m1094r2_m1087r3_m1086r2_c1_zero_work_full_replay_atomic_library_source_contract_r1_20260830.json"
M1094_CONTRACT_SHA = "5278c5fa03a74cf9e3364325865b1bd52a5f75f372de15d5172b0b38bda64be4"
M1094_CONTRACT_SIDECAR_SHA = "963315ed0cd04080eeeb7271dab2da0fa808891919d6aa119f4ed89d4b44fffa"
M1094_CONTRACT_OUTER_SHA = "c35cdf984fb51c584c9ca99f5ff7a638884eb7db3aabab994a62ddc0221b4c5f"
M1094_RECEIPT = HW / "reviews/m1094r2_m1087r3_m1086r2_c1_atomic_library_source_receipt_r1_20260830"
M1094_RECEIPT_ID = (
    "1ff58eb7cf13f1d9e90aff6a45dad5d62fe42879718bef6414b0cdbca233c315",
    "85f0a558966546cccfced52cce3b265a2258185c3444c9036988ba69bf3c4604",
    "3bbeb9624b064021298c7f9d4e4cb2b91777dc9b274326570ec54671f7b7336b",
)
M1095A_HAMMER = HW / "reviews/m1095a_m1094r2_c1_atomic_library_independent_hammer_r1_20260830"
M1095A_HAMMER_ID = (
    "2d8fbe116cb378e558a7d93245debcfbf54fa7a2e3cc98613f1c3bf1bff633d8",
    "b01a44474b50273d3aefde39f7110cc3f3a7930184b0fc497a0c4ba7a06ce146",
    "acf88f44338d8bda95b07cc0694763da34ec1969e69c270ff43c1329aff9a650",
)
M1087R3 = HW / "reviews/m1087r3_m1086r2_c1_zero_work_population_source_hammer_r1_20260830"
M1087R3_ID = (
    "a3b9e35079444a6272ee91040e0250f16d1284c00a3e62c8b5ebc462366d1974",
    "70a5641bc0ad8dde7cb921361e4cd9938737b9cd009747b4f5fcb128b164d1ca",
    "c8901ff70a8a22fa171f0fc47ae6ea40ee91c3af793c9dc5ca09670113369ae5",
)
M1086_SOURCE = HERE / "run_m1086_c1_zero_work_exact_1rw_source.py"
M1086_SOURCE_SHA = "3925c97de922393786b4aa8ae6ca6b4942489e3cf10485f5d1b6cd423e797a51"
M1086R2_CONTRACT = HW / "contracts/m1086r2_c1_zero_work_exact_1rw_population_repair_contract_r1_20260830.json"
M1086R2_CONTRACT_SHA = "351bbec8d7c4b538f035077f18f670ec6deccae4d4a995ec4ce250a6e960ed6f"
M1086R2_CONTRACT_SIDECAR_SHA = "d1e8cde70b0078e8aa80bff681569a0f472954dca31a71ba2e047451f33d3fff"
M1086R2_CONTRACT_OUTER_SHA = "a45bf483f0bf77a48ddce23e7d1d5e0194bc7c5ef2c3893afe29211577dc4243"
M1086R2_RECEIPT = HW / "reviews/m1086r2_c1_zero_work_exact_1rw_population_repair_source_receipt_r1_20260830"
M1086R2_RECEIPT_ID = (
    "26443f683de23495be2258376d5d6a86327f4e3c27c3dc1a4c1bc56401927108",
    "f89bad7e5c31ca329d27e6eba9462e08a634bdd0b2713fe5ef91b53c22cfdf66",
    "8447490431b8474d67e08539bd7cd52aefd4457e7c49262a92bbd7e1d6a5e837",
)
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

TASKS = 812160
DESIGNS = ("candidate", "strongest_zero", "same_coordinate_bit")
VALUES = 2436480
MIN_MEM_AVAILABLE_KIB = 4 * 1024 * 1024
MIN_COMMIT_HEADROOM_KIB = 8 * 1024 * 1024


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + token)))


def verify_flat(directory: Path, identity: tuple[str, str, str]) -> None:
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink() and
            (sha256(review), sha256(manifest), sha256(outer)) == identity,
            "sealed authority identity drift")
    seen = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        member = directory / relative
        require(relative not in seen and member.is_file() and
                not member.is_symlink() and sha256(member) == expected,
                "sealed authority member drift")
        seen.add(relative)
    require(outer.read_text(encoding="utf-8").split() ==
            [sha256(manifest), "SHA256SUMS"], "authority outer content drift")


def verify_double_seal(path: Path, file_sha: str, side_sha: str,
                       outer_sha: str) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(path.is_file() and not path.is_symlink() and sha256(path) == file_sha and
            side.is_file() and not side.is_symlink() and sha256(side) == side_sha and
            outer.is_file() and not outer.is_symlink() and sha256(outer) == outer_sha and
            side.read_text(encoding="utf-8").split() == [file_sha, path.name] and
            outer.read_text(encoding="utf-8").split() == [side_sha, side.name],
            "double seal identity/content drift")


def load_engine():
    require(ENGINE.is_file() and not ENGINE.is_symlink() and
            sha256(ENGINE) == ENGINE_SHA, "M1094r2 engine identity drift")
    spec = importlib.util.spec_from_file_location("m1095_frozen_m1094r2", ENGINE)
    require(spec is not None and spec.loader is not None, "cannot load M1094r2")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1094 = load_engine()


def validate_hardcoded_authorities() -> dict[str, Any]:
    require(len(sys.argv) == 1, "M1095 accepts zero arguments")
    require(Path(sys.executable).resolve() == PYTHON and sha256(PYTHON) == PYTHON_SHA and
            tuple(sys.version_info[:3]) == (3, 10, 18) and sys.flags.isolated == 1 and
            sys.flags.no_user_site == 1,
            "M1095 isolated Python identity drift")
    for path, expected in ((NON_LAUNCH_STUB, NON_LAUNCH_STUB_SHA),
                           (M1086_SOURCE, M1086_SOURCE_SHA),
                           (DOCS359, DOCS359_SHA)):
        require(path.is_file() and not path.is_symlink() and sha256(path) == expected,
                "hardcoded file identity drift")
    verify_double_seal(M1094_CONTRACT, M1094_CONTRACT_SHA,
                       M1094_CONTRACT_SIDECAR_SHA, M1094_CONTRACT_OUTER_SHA)
    verify_double_seal(M1086R2_CONTRACT, M1086R2_CONTRACT_SHA,
                       M1086R2_CONTRACT_SIDECAR_SHA, M1086R2_CONTRACT_OUTER_SHA)
    for directory, identity in ((M1094_RECEIPT, M1094_RECEIPT_ID),
                                (M1095A_HAMMER, M1095A_HAMMER_ID),
                                (M1087R3, M1087R3_ID),
                                (M1086R2_RECEIPT, M1086R2_RECEIPT_ID)):
        verify_flat(directory, identity)
    contract = strict_json(M1094_CONTRACT)
    population = contract["canonical_population"]
    require(population["tasks"] == TASKS and population["designs"] == list(DESIGNS) and
            population["design_count"] == len(DESIGNS) and
            population["task_design_work_values"] == VALUES and
            population["required_preflight_values_checked"] == VALUES,
            "M1095 population authority drift")
    # M1094 performs an independent frozen-source/fresh-namespace validation.
    library = M1094.validate_source_contract(require_fresh=True)
    require(library["tasks"] == TASKS and library["design_count"] == len(DESIGNS) and
            library["task_design_work_values"] == VALUES and
            library["canonical_payload_opened_or_hashed"] is False,
            "M1094 source validation drift")
    return {"status": "PASS_M1095_HARDCODED_AUTHORITIES_NO_PAYLOAD",
            "tasks": TASKS, "design_count": len(DESIGNS),
            "task_design_work_values": VALUES,
            "canonical_payload_opened_or_hashed": False}


def read_meminfo() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        fields = raw.split()
        if fields and fields[0].isdigit():
            values[key] = int(fields[0])
    require(all(key in values for key in
                ("MemAvailable", "CommitLimit", "Committed_AS")),
            "M1095 meminfo schema drift")
    return values


def validate_process_resource_freshness() -> dict[str, Any]:
    competing = []
    self_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == self_pid:
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        decoded = [part.decode("utf-8", "replace") for part in argv if part]
        # Match an actual Python script argv coordinate, not shell command text.
        if any(Path(value).name == Path(__file__).name for value in decoded[1:3]):
            competing.append(int(entry.name))
    require(not competing, "M1095 competing launcher process")
    info = read_meminfo()
    headroom = info["CommitLimit"] - info["Committed_AS"]
    require(info["MemAvailable"] >= MIN_MEM_AVAILABLE_KIB and
            headroom >= MIN_COMMIT_HEADROOM_KIB,
            "M1095 insufficient memory/commit headroom")
    require(not M1094.RESULT.exists() and not M1094.ATTEMPT.exists() and
            not M1094.LOCK.exists() and
            not any(M1094.RESULT.parent.glob(M1094.WORK_PREFIX + "*")) and
            not any(M1094.RESULT.parent.glob(M1094.FAILURE_PREFIX + "*")),
            "M1095 runtime namespace not fresh")
    return {"status": "PASS_M1095_PROCESS_RESOURCE_FRESHNESS",
            "competing_launcher_pids": competing,
            "mem_available_kib": info["MemAvailable"],
            "commit_headroom_kib": headroom,
            "minimum_mem_available_kib": MIN_MEM_AVAILABLE_KIB,
            "minimum_commit_headroom_kib": MIN_COMMIT_HEADROOM_KIB}


def hardcoded_result_authority() -> dict[str, Any]:
    """Only immutable predecessor identities; no caller or environment values."""
    return {
        "schema": "m1095_hardcoded_c1_launch_authority_v1",
        "status": "PASS_DIFFERENT_AUTHOR_HARDCODED_LAUNCH_AUTHORITY",
        "m1094r2_engine_sha256": ENGINE_SHA,
        "m1094r2_contract_sha256": M1094_CONTRACT_SHA,
        "m1094r2_contract_outer_seal_file_sha256": M1094_CONTRACT_OUTER_SHA,
        "m1094r2_receipt_outer_seal_file_sha256": M1094_RECEIPT_ID[2],
        "m1095a_library_hammer_outer_seal_file_sha256": M1095A_HAMMER_ID[2],
        "m1087r3_outer_seal_file_sha256": M1087R3_ID[2],
        "m1086_source_sha256": M1086_SOURCE_SHA,
        "m1086r2_contract_outer_seal_file_sha256": M1086R2_CONTRACT_OUTER_SHA,
        "m1086r2_receipt_outer_seal_file_sha256": M1086R2_RECEIPT_ID[2],
        "python_sha256": PYTHON_SHA,
        "docs359_sha256": DOCS359_SHA,
        "m1098_independent_launch_hammer_required_before_execution": True,
        "m1099_independent_result_hammer_required_after_success": True,
        "legacy_m1094_schema_m1096_token_is_superseded_by_m1099": True,
    }


def consume_attempt_atomically(authority: dict[str, Any]) -> dict[str, Any]:
    require(type(authority) is dict and authority == hardcoded_result_authority(),
            "M1095 authority must equal source-hardcoded identity map")
    require(not M1094.ATTEMPT.exists() and not M1094.RESULT.exists(),
            "M1095 attempt/result collision")
    try:
        M1094.ATTEMPT.mkdir(mode=0o700)
    except FileExistsError as error:
        raise RuntimeError("M1095 attempt collision") from error
    M1094.fsync_dir(M1094.ATTEMPT.parent)
    receipt = {
        "schema": "m1095_c1_full_replay_attempt_r1_v1",
        "status": "CONSUMED_BEFORE_CANONICAL_PAYLOAD_ACCESS",
        "maximum_attempts": 1,
        "automatic_retry": False,
        "authority": authority,
        "canonical_payload_opened_or_hashed_before_attempt": False,
    }
    M1094.write_exclusive(M1094.ATTEMPT / "attempt.json",
                          (json.dumps(receipt, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n").encode())
    return {"receipt": receipt, "seal": M1094.atomic_seal(M1094.ATTEMPT)}


def acquire_lock() -> None:
    require(M1094.LOCK.parent.resolve() == M1094.RESULT.parent.resolve() and
            M1094.LOCK.name == ".m1094_c1_zero_work_exact_1rw_full_replay.lock",
            "M1095 lock identity drift")
    try:
        M1094.LOCK.mkdir(mode=0o700)
    except FileExistsError as error:
        raise RuntimeError("M1095 launch lock collision") from error
    M1094.write_exclusive(M1094.LOCK / "owner.json", (json.dumps({
        "schema": "m1095_c1_launch_lock_v1", "pid": os.getpid(),
        "automatic_retry": False}, sort_keys=True) + "\n").encode())
    M1094.fsync_dir(M1094.LOCK.parent)


def release_lock() -> None:
    if M1094.LOCK.is_dir() and not M1094.LOCK.is_symlink():
        owner = M1094.LOCK / "owner.json"
        require(owner.is_file() and not owner.is_symlink(), "M1095 lock owner drift")
        owner.unlink()
        M1094.LOCK.rmdir()
        M1094.fsync_dir(M1094.LOCK.parent)


def interrupted(signum, _frame) -> None:
    raise RuntimeError("M1095 interrupted by signal %d" % int(signum))


def main() -> int:
    # M1098 must run exactly this file through the pinned interpreter with -I.
    validate_hardcoded_authorities()
    validate_process_resource_freshness()
    authority = hardcoded_result_authority()
    locked = False
    attempt_consumed = False
    work = M1094.RESULT.parent / (M1094.WORK_PREFIX +
            "%d.%d" % (os.getpid(), time.time_ns()))
    quarantine = M1094.RESULT.parent / (M1094.FAILURE_PREFIX +
            "%d.%d.quarantine" % (os.getpid(), time.time_ns()))
    phase = "PRE_ATTEMPT"
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, interrupted)
    try:
        acquire_lock(); locked = True
        # Recheck freshness under the unique lock, excluding the lock itself.
        require(not M1094.RESULT.exists() and not M1094.ATTEMPT.exists() and
                not any(M1094.RESULT.parent.glob(M1094.WORK_PREFIX + "*")) and
                not any(M1094.RESULT.parent.glob(M1094.FAILURE_PREFIX + "*")),
                "M1095 post-lock freshness drift")
        phase = "CONSUME_ATTEMPT"
        consume_attempt_atomically(authority); attempt_consumed = True
        phase = "PREFLIGHT_THEN_FULL_REPLAY"
        # execute_full's first canonical payload operation is the zero-argument
        # preflight; it then calls the zero-argument iterator exactly once.
        M1094.execute_full(authority, work)
        phase = "ATOMIC_NO_REPLACE_PUBLISH"
        published = M1094.publish_result(work)
        require(published["status"] == M1094.RESULT_STATUS,
                "M1095 publication status drift")
        print(json.dumps({"status": "PASS_M1095_RAW_CPU_MODEL_PUBLISHED_PENDING_M1099",
                          "result": str(M1094.RESULT),
                          "speedup_admitted": False,
                          "paper_citable": False}, sort_keys=True))
        return 0
    except BaseException:
        failure = traceback.format_exc()
        if attempt_consumed:
            try:
                M1094.quarantine_work(work, quarantine, 1, phase)
            except BaseException:
                # Attempt remains consumed and therefore cannot retry even if
                # quarantine itself is interrupted.
                sys.stderr.write("M1095_QUARANTINE_FAILURE\n" + traceback.format_exc())
        sys.stderr.write("M1095_FAIL_CLOSED phase=" + phase + "\n" + failure)
        return 1
    finally:
        if locked:
            release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
