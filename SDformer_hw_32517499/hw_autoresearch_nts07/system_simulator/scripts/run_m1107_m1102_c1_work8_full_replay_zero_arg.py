#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1107 zero-argument launcher for the frozen M1102 C1 replay.

SOURCE ONLY.  A different-author final launcher hammer must approve this exact
file before execution.  The launcher accepts no argv/configuration/authority
from its caller, consumes at most one attempt, and never retries.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import sys
import time
import traceback
from typing import Any


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
PYTHON_SHA = "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
ATOMIC = HERE / "execute_m1102_c1_work8_exact_1rw_full_replay_atomic.py"
ATOMIC_SHA = "0325a4c901e945656ad6d74b12cae6b066f5b75bb426326143f8b0a8f24d1157"
SOURCE = HERE / "run_m1102_c1_work8_exact_1rw_source.py"
SOURCE_SHA = "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc"
CONTRACT = HW / "contracts/m1102_c1_legal_work8_exact_1rw_additive_source_contract_r1_20260830.json"
CONTRACT_ID = (
    "fad9c381fc1e55fc78d6cf4b95ad0959b5a7089989a7acce1ccfafa73714db6e",
    "e6754574c804a7ed2cfd39e5a99c991db38402389901fef570359decf43e3607",
    "b17774b1b3fad06f104081b2ab2b0de4b3b539c72fd9e6adcb2171a46d55770c",
)
SOURCE_RECEIPT = HW / "reviews/m1102_c1_legal_work8_exact_1rw_additive_source_receipt_r1_20260830"
SOURCE_RECEIPT_ID = (
    "5b5383a062672844d07a35a671aa6bc61d9efa660ad934afbe0ea42e51a16797",
    "6f705cf7b2064aba5e54ecb4a2d0399c5c5462de07781cb0ac03c37e0a986ed5",
    "326cc8ba37dd839a8447d89cdbb7156b623207bf6405ae57a0954c71a8db6377",
)
M1104 = HW / "reviews/m1104_m1102_c1_source_atomic_independent_hammer_r1_20260830"
M1104_ID = (
    "341026dc3c28bbea421bf29c1281f0aadfa58ce2cd2a59af85e6ef8fd0ceb89f",
    "f9947c686b98c062576b6af2207e3e0ed152b0278e44ee4393ba27e0e157ff61",
    "a3c28bb2e7c5040f83199dba4e70eefa46e86dc95a06eb5709b3be20a4bed237",
)
M1104_HAMMER_SHA = "94bc3b3a0186b0f5ccf8416b8292e1dba3204fc5937c29c07e4f92e566740013"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
M1100_OUTER = "867102e3529a8c4bc10b4ad3fe2336e4ddfcc6350cdcc3d38fdb783c7dc71376"
M1101_OUTER = "d9f95f7c9b3fb15bef9f369c365603dd7060529b08b4bab5f0626f06d5bb7539"

TASKS = 812160
VALUES = 2436480
DESIGNS = ("candidate", "strongest_zero", "same_coordinate_bit")
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
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + token)))


def verify_regular(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and sha256(path) == expected,
            "pinned regular file identity drift: " + str(path))


def verify_double(path: Path, identity: tuple[str, str, str]) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    verify_regular(path, identity[0]); verify_regular(side, identity[1])
    verify_regular(outer, identity[2])
    require(side.read_text(encoding="utf-8").split() == [identity[0], path.name] and
            outer.read_text(encoding="utf-8").split() == [identity[1], side.name],
            "double-seal content drift")


def verify_flat(directory: Path, identity: tuple[str, str, str], status: str) -> None:
    require(directory.is_dir() and not directory.is_symlink(), "authority directory drift")
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require((sha256(review), sha256(manifest), sha256(outer)) == identity,
            "authority root identity drift")
    require(outer.read_text(encoding="utf-8").split() == [identity[1], "SHA256SUMS"],
            "authority outer content drift")
    seen = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        require(relative not in seen and not Path(relative).is_absolute() and
                ".." not in Path(relative).parts, "authority member path drift")
        verify_regular(directory / relative, expected)
        seen.add(relative)
    require(strict_json(review).get("status") == status, "authority status drift")


def load_atomic():
    verify_regular(ATOMIC, ATOMIC_SHA)
    spec = importlib.util.spec_from_file_location("m1107_frozen_m1102_atomic", ATOMIC)
    require(spec is not None and spec.loader is not None, "cannot load M1102 atomic")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1102 = load_atomic()


def hardcoded_authority() -> dict[str, str]:
    return {
        "status": "PASS_DIFFERENT_AUTHOR_M1102_HARDCODED_LAUNCH_AUTHORITY",
        "launch_wrapper_sha256": sha256(Path(__file__).resolve()),
        "launch_hammer_outer_seal_file_sha256": M1104_ID[2],
        "m1102_atomic_library_sha256": ATOMIC_SHA,
        "m1102_semantic_source_sha256": SOURCE_SHA,
        "m1102_contract_sha256": CONTRACT_ID[0],
        "m1100_outer_seal_file_sha256": M1100_OUTER,
        "m1101_outer_seal_file_sha256": M1101_OUTER,
    }


def validate_hardcoded_authorities(enforce_runtime: bool) -> dict[str, Any]:
    if enforce_runtime:
        require(len(sys.argv) == 1, "M1107 accepts zero arguments")
        require(Path(sys.executable).resolve() == PYTHON and
                tuple(sys.version_info[:3]) == (3, 10, 18) and
                sys.flags.isolated == 1 and sys.flags.no_user_site == 1,
                "M1107 requires pinned isolated Python")
    verify_regular(PYTHON, PYTHON_SHA)
    verify_regular(SOURCE, SOURCE_SHA)
    verify_regular(DOCS359, DOCS359_SHA)
    verify_double(CONTRACT, CONTRACT_ID)
    verify_flat(SOURCE_RECEIPT, SOURCE_RECEIPT_ID,
                "PASS_M1102_ADDITIVE_SOURCE_AUTHOR_RECEIPT__DIFFERENT_AUTHOR_HAMMER_REQUIRED")
    verify_flat(M1104, M1104_ID,
                "PASS_M1104_M1102_SOURCE_ATOMIC_HAMMER__AUTHOR_ZERO_ARG_LAUNCHER_ONLY")
    verify_regular(M1104 / "independent_hammer.py", M1104_HAMMER_SHA)
    contract = strict_json(CONTRACT)
    semantics = contract["semantics"]
    population = contract["canonical_population"]
    require(semantics["generic_domain"] == "exact int; work==0 or work>=8" and
            semantics["canonical_domain"] ==
                "exact int; work%8==0 and (work==0 or work>=8)" and
            semantics["work_1_to_7"] == "fail closed" and
            population["tasks"] == TASKS and
            population["task_design_work_values"] == VALUES and
            population["designs"] == list(DESIGNS), "work-domain authority drift")
    M1102._validate_launch_authority(hardcoded_authority())
    return {"status": "PASS_M1107_HARDCODED_AUTHORITIES_NO_ATTEMPT",
            "tasks": TASKS, "values": VALUES, "designs": list(DESIGNS),
            "generic_domain": semantics["generic_domain"],
            "canonical_domain": semantics["canonical_domain"]}


def sanitize_environment() -> None:
    """Discard all caller variables; replay code receives constants only."""
    os.environ.clear()
    os.environ.update({"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                       "PATH": "/usr/bin:/bin", "TMPDIR": "/tmp",
                       "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})


def read_meminfo() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        fields = raw.split()
        if fields and fields[0].isdigit():
            values[key] = int(fields[0])
    require(all(key in values for key in ("MemAvailable", "CommitLimit", "Committed_AS")),
            "meminfo schema drift")
    return values


def namespace_freshness(ignore_lock: bool = False) -> dict[str, Any]:
    competing = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = [part.decode("utf-8", "replace") for part in
                    (entry / "cmdline").read_bytes().split(b"\0") if part]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(Path(value).name == Path(__file__).name for value in argv[1:3]):
            competing.append(int(entry.name))
    require(not competing, "competing M1107 launcher")
    require(not M1102.RESULT.exists() and not M1102.ATTEMPT.exists() and
            (ignore_lock or not M1102.LOCK.exists()) and
            not any(M1102.RESULT.parent.glob(M1102.WORK_PREFIX + "*")) and
            not any(M1102.RESULT.parent.glob(M1102.FAILURE_PREFIX + "*")),
            "M1107 runtime namespace not fresh")
    info = read_meminfo()
    headroom = info["CommitLimit"] - info["Committed_AS"]
    require(info["MemAvailable"] >= MIN_MEM_AVAILABLE_KIB and
            headroom >= MIN_COMMIT_HEADROOM_KIB, "insufficient memory/commit headroom")
    return {"status": "PASS_M1107_NAMESPACE_RESOURCE_FRESHNESS",
            "competing_launcher_pids": competing,
            "mem_available_kib": info["MemAvailable"],
            "commit_headroom_kib": headroom}


def acquire_lock() -> None:
    try:
        M1102.LOCK.mkdir(mode=0o700)
    except FileExistsError as error:
        raise RuntimeError("M1107 launch lock collision") from error
    M1102.write_exclusive(M1102.LOCK / "owner.json", (json.dumps({
        "schema": "m1107_c1_launch_lock_v1", "pid": os.getpid(),
        "maximum_attempts": 1, "automatic_retry": False}, sort_keys=True) + "\n").encode())
    M1102.fsync_dir(M1102.LOCK.parent)


def release_lock() -> None:
    if M1102.LOCK.is_dir() and not M1102.LOCK.is_symlink():
        owner = M1102.LOCK / "owner.json"
        require(owner.is_file() and not owner.is_symlink(), "lock owner drift")
        owner.unlink(); M1102.LOCK.rmdir(); M1102.fsync_dir(M1102.LOCK.parent)


def source_static_self_test() -> dict[str, Any]:
    require(not M1102.ATTEMPT.exists() and not M1102.RESULT.exists(),
            "self-test refuses existing production namespace")
    identities = validate_hardcoded_authorities(enforce_runtime=False)
    before = (M1102.ATTEMPT.exists(), M1102.RESULT.exists())
    oracle = M1102.M1102.source_small_oracle()
    require(oracle.get("status") == "PASS_M1102_WORK8_SOURCE_SMALL_ORACLE" and
            oracle.get("attempt_created") is False, "small oracle drift")
    require(before == (False, False) and not M1102.ATTEMPT.exists() and
            not M1102.RESULT.exists(), "self-test created production evidence")
    return {"status": "PASS_M1107_ZERO_ARG_LAUNCHER_SOURCE_SELF_TEST__NO_ATTEMPT",
            "identities": identities, "small_oracle": oracle,
            "launcher_executed": False, "attempt_created": False,
            "full_replay_executed": False, "automatic_retry": False}


def interrupted(signum, _frame) -> None:
    raise RuntimeError("M1107 interrupted by signal %d" % int(signum))


def main() -> int:
    validate_hardcoded_authorities(enforce_runtime=True)
    namespace_freshness()
    sanitize_environment()
    authority = hardcoded_authority()
    locked = False
    attempt_consumed = False
    work = M1102.RESULT.parent / (M1102.WORK_PREFIX +
            "%d.%d" % (os.getpid(), time.time_ns()))
    quarantine = M1102.RESULT.parent / (M1102.FAILURE_PREFIX +
            "%d.%d.quarantine" % (os.getpid(), time.time_ns()))
    phase = "PRE_ATTEMPT"
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, interrupted)
    try:
        acquire_lock(); locked = True
        namespace_freshness(ignore_lock=True)
        phase = "CONSUME_ATTEMPT"
        M1102.consume_attempt(authority); attempt_consumed = True
        phase = "EXHAUSTIVE_PREFLIGHT_THEN_FULL_REPLAY"
        M1102.execute_full(authority, work)
        phase = "ATOMIC_NO_REPLACE_PUBLISH"
        published = M1102.publish_result(work)
        print(json.dumps({"status": published["status"],
                          "result": str(M1102.RESULT),
                          "independent_result_hammer_required": True,
                          "speedup_admitted": False}, sort_keys=True))
        return 0
    except BaseException:
        failure = traceback.format_exc()
        if attempt_consumed:
            try:
                M1102.quarantine_work(work, quarantine, 1, phase)
            except BaseException:
                sys.stderr.write("M1107_QUARANTINE_FAILURE\n" + traceback.format_exc())
        sys.stderr.write("M1107_FAIL_CLOSED phase=" + phase + "\n" + failure)
        return 1
    finally:
        if locked:
            release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
