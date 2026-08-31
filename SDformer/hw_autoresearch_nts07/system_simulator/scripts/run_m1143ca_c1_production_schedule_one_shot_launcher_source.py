#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1143CA immutable zero-argument one-shot launcher source.

Source only until a different-author final launch hammer.  The static self-test
does not open M410 and never invokes the production child.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import traceback
from typing import Any

sys.dont_write_bytecode = True
SOURCE_FILE = Path(__file__).resolve()
HERE = SOURCE_FILE.parent
HW = HERE.parent.parent
RESULTS = HW / "results"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
PYTHON_SHA = "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
CHILD = HERE / "run_m1141ca_c1_production_schedule_release_source.py"
CHILD_SHA = "e2f5d4e0bab472b3a5c7ec5259a805641b800efd3c0e82884e81152eb41cb611"
CHILD_CONTRACT = HW / "contracts/m1141ca_c1_production_schedule_release_source_contract_r1_20260830.json"
CHILD_CONTRACT_ID = (
    "4fe7ba960516e889cb1f7140315e1e37a5b42dd00337f136b22a25f1c7ac06d4",
    "128d813d63cba813173a5e282dd6f3247ff2f443a5428878d76bef36230d0263",
    "6e5561e52fab6b4ae3018f8995f4b71f4c8eaeaf02c83ea192421081b5af8184",
)
CHILD_AUTHOR = HW / "reviews/m1141ca_c1_production_schedule_release_source_author_receipt_r1_20260830"
CHILD_AUTHOR_ID = (
    "60ae32a718c275336cc83b1943db61fd99ec20516ccf7feeb70eafe4033fe76b",
    "4a7e1701ae822ceec0be3686c79a277b20c19ec61afc3212b5be86c5bfab0bb7",
    "b5602b120cc7c02769a54e67c78588c481776af9f40f3d3359a2938bf2f8b825",
)
M1142 = HW / "reviews/m1142ca_m1141ca_c1_production_schedule_release_hammer_r1_20260830"
M1142_ID = (
    "4c815c711242967777338369c3093422575f66acc452c51e08f54838ce006fe9",
    "8133859d3a28296d95ab5615d70c6a5252a637af613421c120e63be1fe94096a",
    "7a8f8da04bb81a0097d819f98a3bed6e9e40b86a32aef055134f3306bb1850e8",
)
CONTRACT = HW / "contracts/m1143ca_c1_one_shot_production_schedule_launcher_source_contract_r1_20260830.json"
CONTRACT_ID = (
    "07f27437e12c7e52bf95604f8ee3cba4e0f72f83d795fc92ad2536e7eb555be1",
    "738e5f1e7912421cb1c6d982fb88a2018121fb190ddaa673764a90f779831919",
    "e2a4a9f1a9962d485574c5193895aad0e573b7ff2ebc37a5b20c34de424adb80",
)
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
M410_SHA = "6e03352b89eff1955825334b4dedd991db8c975a9ef6662fe0317e73ccfa8334"
M1016_SHA = "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa"
M1139_SHA = "d18137661517538a8273b696b5f2ada09aff9847c16da0d3a723037e901153a9"
M1140CA_OUTER_SHA = "f73cafa73ed047abd59730749bf48fcb3f463fca77609aec6085f5b3389fa352"
EXPECTED_UID = 1913
MIN_CPUS = 4
MIN_MEM_AVAILABLE = 4 * (1 << 30)
MIN_COMMIT_HEADROOM = 8 * (1 << 30)
MIN_DISK_FREE = 16 * (1 << 30)
CHILD_TIMEOUT_SECONDS = 172_800
MANIFEST = "SHA256SUMS"
OUTER = "SHA256SUMS.seal.sha256"
RECORDS_NAME = "m1141ca_per_task_schedule_records.jsonl"
RELEASE_NAME = "m1141ca_schedule_release.json"
AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
CHILD_STATUS = "PASS_EXACT_PRODUCTION_SCHEDULE_RELEASE__DIFFERENT_AUTHOR_RESULT_HAMMER_REQUIRED"
CHILD_ENVIRONMENT = {
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "TZ": "UTC",
}


class Failure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_regular(path: Path, expected: str, owner_uid: int | None = None) -> None:
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and not path.is_symlink() and
            sha256(path) == expected and
            (owner_uid is None or value.st_uid == owner_uid),
            "identity/owner drift: " + str(path))


def strict_json(path: Path) -> Any:
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON: " + token)))


def verify_double(path: Path, identity: tuple[str, str, str]) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    verify_regular(path, identity[0], EXPECTED_UID)
    verify_regular(side, identity[1], EXPECTED_UID)
    verify_regular(outer, identity[2], EXPECTED_UID)
    require(side.read_text(encoding="utf-8").split() == [identity[0], path.name] and
            outer.read_text(encoding="utf-8").split() == [identity[1], side.name],
            "double seal content drift")


def verify_tree(directory: Path, identity: tuple[str, str, str]) -> dict[str, Any]:
    require(directory.is_dir() and not directory.is_symlink() and
            directory.stat().st_uid == EXPECTED_UID, "sealed directory owner/type drift")
    review = directory / "review.json"; manifest = directory / MANIFEST
    outer = directory / OUTER
    verify_regular(review, identity[0], EXPECTED_UID)
    verify_regular(manifest, identity[1], EXPECTED_UID)
    verify_regular(outer, identity[2], EXPECTED_UID)
    require(outer.read_text(encoding="utf-8").split() == [identity[1], MANIFEST],
            "sealed outer content drift")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*"); rel = Path(name)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and
                name not in expected and name == rel.as_posix() and
                not rel.is_absolute() and ".." not in rel.parts,
                "sealed manifest member drift")
        expected[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {MANIFEST, OUTER}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "sealed symlink")
        if stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), "sealed special member")
    require(actual == set(expected), "sealed exact member set drift")
    for name, digest in expected.items():
        verify_regular(directory / name, digest, EXPECTED_UID)
    return strict_json(review)


def source_preflight() -> dict[str, Any]:
    verify_regular(CHILD, CHILD_SHA, EXPECTED_UID)
    verify_double(CHILD_CONTRACT, CHILD_CONTRACT_ID)
    author = verify_tree(CHILD_AUTHOR, CHILD_AUTHOR_ID)
    hammer = verify_tree(M1142, M1142_ID)
    verify_double(CONTRACT, CONTRACT_ID)
    verify_regular(PYTHON, PYTHON_SHA)
    verify_regular(DOCS359, DOCS359_SHA, EXPECTED_UID)
    child_contract = strict_json(CHILD_CONTRACT)
    require(CHILD_ENVIRONMENT == {
                "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PATH": "/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1", "TZ": "UTC"},
            "clean child environment drift")
    require(author["status"] ==
            "PASS_M1141CA_SOURCE_AND_CONTROLLED_FAKE_RELEASE__DIFFERENT_AUTHOR_HAMMER_ONLY" and
            author["authorization"]["production_execution"] is False and
            hammer["status"] ==
            "PASS_M1142CA_INDEPENDENT_RELEASE_HAMMER__AUTHOR_ONE_SHOT_PRODUCTION_LAUNCHER_SOURCE_ONLY" and
            hammer["authorization"]["production_execution"] is False and
            hammer["authorization"]["one_shot_production_schedule_execution_launcher_source_authoring"] is True and
            child_contract["source"]["arguments"] == 0 and
            child_contract["source"]["automatic_retry"] is False and
            child_contract["production_geometry"]["records"] == 2_436_480,
            "M1141CA/M1142CA authorization drift")
    return {
        "status": "PASS_M1143CA_LAUNCHER_SOURCE_PREFLIGHT__NO_CHILD_NO_M410",
        "m1141ca_source_sha256": CHILD_SHA,
        "m1141ca_contract_outer_seal_file_sha256": CHILD_CONTRACT_ID[2],
        "m1141ca_author_outer_seal_file_sha256": CHILD_AUTHOR_ID[2],
        "m1142ca_outer_seal_file_sha256": M1142_ID[2],
        "child_processes": 0, "m410_opened": False,
    }


@dataclass(frozen=True)
class LaunchLayout:
    result_root: Path
    launcher_result: Path
    attempt: Path
    lock: Path
    launcher_work_prefix: str
    launcher_failure_prefix: str
    child_result: Path
    child_work_prefix: str
    child_failure_prefix: str


@dataclass(frozen=True)
class ChildExpectation:
    status: str
    tasks: int
    records: int
    rows_sha256: str


PRODUCTION_LAYOUT = LaunchLayout(
    RESULTS,
    RESULTS / "m1143ca_c1_production_schedule_one_shot_launch_r1_20260830",
    RESULTS / ".m1143ca_c1_production_schedule_one_shot_attempt_consumed",
    Path("/tmp/m1143ca_c1_production_schedule_one_shot.lock"),
    ".m1143ca_c1_production_schedule_one_shot_work.",
    "m1143ca_c1_production_schedule_one_shot_launch_r1_20260830.failed_or_incomplete.",
    RESULTS / "m1141ca_c1_production_schedule_release_r1_20260830",
    ".m1141ca_c1_production_schedule_release_work.",
    "m1141ca_c1_production_schedule_release_r1_20260830.failed_or_incomplete.",
)
PRODUCTION_EXPECTATION = ChildExpectation(CHILD_STATUS, 812_160, 2_436_480, M410_SHA)


def _namespace_collisions(layout: LaunchLayout, ignore_lock: bool = False) -> tuple[str, ...]:
    found = []
    for path in (layout.launcher_result, layout.attempt, layout.child_result):
        if path.exists() or path.is_symlink():
            found.append(str(path))
    if not ignore_lock and (layout.lock.exists() or layout.lock.is_symlink()):
        found.append(str(layout.lock))
    for prefix in (layout.launcher_work_prefix, layout.launcher_failure_prefix,
                   layout.child_work_prefix, layout.child_failure_prefix):
        found.extend(str(path) for path in layout.result_root.glob(prefix + "*"))
    return tuple(sorted(found))


def _meminfo() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"(MemAvailable|CommitLimit|Committed_AS):\s+(\d+)\s+kB", line)
        if match:
            values[match.group(1)] = int(match.group(2)) * 1024
    require(set(values) == {"MemAvailable", "CommitLimit", "Committed_AS"},
            "meminfo resource fields absent")
    return values


def _same_uid_conflicting_processes() -> tuple[int, ...]:
    conflicts = []
    exact_tokens = {str(CHILD), str(SOURCE_FILE), CHILD.name, SOURCE_FILE.name}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            status_text = (entry / "status").read_text(encoding="utf-8")
            uid_match = re.search(r"^Uid:\s+(\d+)", status_text, re.M)
            raw = (entry / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        tokens = {item.decode("utf-8", errors="replace") for item in raw if item}
        if (uid_match is not None and
                int(uid_match.group(1)) == EXPECTED_UID and
                tokens & exact_tokens):
            conflicts.append(int(entry.name))
    return tuple(sorted(conflicts))


def _external_resource_preflight(layout: LaunchLayout) -> dict[str, Any]:
    require(os.getuid() == EXPECTED_UID and HW.stat().st_uid == EXPECTED_UID and
            layout.result_root.stat().st_uid == EXPECTED_UID,
            "launcher/workspace/result UID mismatch")
    cpu_count = os.cpu_count()
    require(type(cpu_count) is int and cpu_count >= MIN_CPUS,
            "insufficient CPU resource")
    memory = _meminfo()
    commit_headroom = memory["CommitLimit"] - memory["Committed_AS"]
    disk_free = shutil.disk_usage(layout.result_root).free
    conflicts = _same_uid_conflicting_processes()
    require(memory["MemAvailable"] >= MIN_MEM_AVAILABLE,
            "insufficient MemAvailable resource")
    require(commit_headroom >= MIN_COMMIT_HEADROOM,
            "insufficient commit headroom")
    require(disk_free >= MIN_DISK_FREE, "insufficient result filesystem capacity")
    require(conflicts == (), "same-UID conflicting child/launcher process")
    return {
        "uid": EXPECTED_UID, "cpu_count": cpu_count,
        "mem_available_bytes": memory["MemAvailable"],
        "commit_headroom_bytes": commit_headroom,
        "result_filesystem_free_bytes": disk_free,
        "same_uid_conflicting_processes": 0,
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    finally:
        os.close(fd)


def _write_json(path: Path, value: Any) -> None:
    _write_exclusive(path, (json.dumps(value, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n").encode())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _rename_noreplace(source: Path, destination: Path) -> None:
    require(not destination.exists() and not destination.is_symlink(),
            "publish destination collision")
    libc = ctypes.CDLL(None, use_errno=True)
    call = getattr(libc, "renameat2", None)
    require(call is not None, "renameat2 unavailable")
    call.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                     ctypes.c_char_p, ctypes.c_uint]
    call.restype = ctypes.c_int
    if call(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        raise Failure("renameat2 noreplace failed: " + os.strerror(ctypes.get_errno()))


def _seal_tree(directory: Path) -> tuple[str, str]:
    members = []
    for path in directory.rglob("*"):
        if path.name in {MANIFEST, OUTER}:
            continue
        mode = path.lstat().st_mode
        require(not stat.S_ISLNK(mode), "output symlink")
        if stat.S_ISREG(mode):
            members.append(path)
        else:
            require(stat.S_ISDIR(mode), "output special member")
    members.sort(key=lambda path: path.relative_to(directory).as_posix())
    require(members, "empty output tree")
    manifest = "".join(
        f"{sha256(path)}  {path.relative_to(directory).as_posix()}\n" for path in members)
    _write_exclusive(directory / MANIFEST, manifest.encode())
    manifest_sha = sha256(directory / MANIFEST)
    _write_exclusive(directory / OUTER, f"{manifest_sha}  {MANIFEST}\n".encode())
    _fsync_dir(directory)
    return manifest_sha, sha256(directory / OUTER)


def _verify_child_result(path: Path, expectation: ChildExpectation) -> dict[str, Any]:
    require(path.is_dir() and not path.is_symlink(), "child result absent/non-directory")
    manifest = path / MANIFEST; outer = path / OUTER
    require(manifest.is_file() and outer.is_file() and not manifest.is_symlink() and
            not outer.is_symlink(), "child seal absent")
    manifest_sha, manifest_name = outer.read_text(encoding="utf-8").split()
    require(manifest_name == MANIFEST and sha256(manifest) == manifest_sha,
            "child outer seal mismatch")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*"); rel = Path(name)
        require(name not in expected and name == rel.as_posix() and
                not rel.is_absolute() and ".." not in rel.parts,
                "child manifest member drift")
        expected[name] = digest
    require(set(expected) == {RECORDS_NAME, RELEASE_NAME},
            "child exact result member set drift")
    actual = {member.name for member in path.iterdir()
              if member.name not in {MANIFEST, OUTER}}
    require(actual == set(expected) and all(
        (path / name).is_file() and not (path / name).is_symlink() and
        sha256(path / name) == digest for name, digest in expected.items()),
        "child result exact census/hash drift")
    release = strict_json(path / RELEASE_NAME)
    records_sha = sha256(path / RECORDS_NAME)
    require(release["schema"] == "m1141ca_c1_production_schedule_release_r1_v1" and
            release["status"] == expectation.status and
            release["source_rows"]["sha256"] == expectation.rows_sha256 and
            release["geometry"]["tasks"] == expectation.tasks and
            release["geometry"]["records"] == expectation.records and
            release["geometry"]["axes"] == list(AXES) and
            release["records"]["file"] == RECORDS_NAME and
            release["records"]["count"] == expectation.records and
            release["records"]["sha256"] == records_sha and
            release["records"]["axis_order_within_each_task"] == list(AXES) and
            release["records"]["axis_counts"] == {
                axis: expectation.tasks for axis in AXES} and
            release["authority"]["m1016_source_sha256"] == M1016_SHA and
            release["authority"]["m1139ca_source_sha256"] == M1139_SHA and
            release["authority"]["m1140ca_outer_seal_file_sha256"] ==
                M1140CA_OUTER_SHA and
            release["claim_boundary"]["digest_compiler"] is False and
            release["claim_boundary"]["real_driver"] is False and
            release["claim_boundary"]["paper_citable"] is False,
            "child release semantic drift")
    return {"manifest_sha256": manifest_sha,
            "outer_seal_file_sha256": sha256(outer),
            "records_sha256": records_sha, "records": expectation.records}


def _execute_once(layout: LaunchLayout, child_source: Path, child_sha: str,
                  expectation: ChildExpectation, require_real_authority: bool) -> dict[str, Any]:
    """Private fixture-capable core; production_main binds every real argument."""
    if require_real_authority:
        source_preflight()
    require(layout.result_root.is_dir() and not layout.result_root.is_symlink(),
            "result root drift")
    verify_regular(child_source, child_sha, EXPECTED_UID)
    require(_namespace_collisions(layout) == (), "launcher/child namespace collision")
    resource_before = _external_resource_preflight(layout)
    lock_fd = os.open(layout.lock,
                      os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    attempt_consumed = False
    work = layout.result_root / (layout.launcher_work_prefix +
                                 f"{os.getpid()}.{time.time_ns()}")
    failure = layout.result_root / (layout.launcher_failure_prefix +
                                    f"{os.getpid()}.{time.time_ns()}.quarantine")
    phase = "LOCKED_PREFLIGHT"
    child_processes = 0
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(_namespace_collisions(layout, ignore_lock=True) == (),
                "namespace changed under lock")
        resource_locked = _external_resource_preflight(layout)
        phase = "CONSUME_SINGLE_ATTEMPT"
        layout.attempt.mkdir(mode=0o700)
        _write_json(layout.attempt / "attempt.json", {
            "schema": "m1143ca_c1_production_schedule_one_shot_attempt_r1_v1",
            "status": "M1143CA_SINGLE_ATTEMPT_CONSUMED__NO_AUTO_RETRY",
            "m1141ca_source_sha256": CHILD_SHA,
            "m1141ca_contract_outer_seal_file_sha256": CHILD_CONTRACT_ID[2],
            "m1141ca_author_outer_seal_file_sha256": CHILD_AUTHOR_ID[2],
            "m1142ca_outer_seal_file_sha256": M1142_ID[2],
            "expected_child_processes": 1, "automatic_retry": False,
        })
        _seal_tree(layout.attempt); _fsync_dir(layout.result_root)
        attempt_consumed = True
        work.mkdir(mode=0o700)
        phase = "SPAWN_EXACTLY_ONE_ZERO_ARGUMENT_CHILD"
        command = [str(PYTHON), str(child_source)]
        require(command == [str(PYTHON), str(child_source)] and len(command) == 2,
                "child command/zero-argument drift")
        with (work / "child.stdout.log").open("wb") as stdout, \
                (work / "child.stderr.log").open("wb") as stderr:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                cwd=str(HW), env=dict(CHILD_ENVIRONMENT), close_fds=True,
                start_new_session=True)
            child_processes += 1
            require(child_processes == 1, "more than one child process")
            try:
                child_rc = process.wait(timeout=CHILD_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=10)
                raise Failure("production child timeout; no retry")
        require(child_rc == 0, "production child failed; no retry")
        phase = "VERIFY_CHILD_RESULT_FAILURE_EXCLUSION"
        child_failures = tuple(layout.result_root.glob(layout.child_failure_prefix + "*"))
        child_works = tuple(layout.result_root.glob(layout.child_work_prefix + "*"))
        require(child_failures == () and child_works == () and
                layout.child_result.exists() and not layout.child_result.is_symlink(),
                "child result/failure/work mutual exclusion drift")
        child_receipt = _verify_child_result(layout.child_result, expectation)
        phase = "SEAL_AND_PUBLISH_LAUNCH_RECEIPT"
        _write_json(work / "receipt.json", {
            "schema": "m1143ca_c1_production_schedule_one_shot_launch_receipt_r1_v1",
            "status": "PASS_M1143CA_ONE_SHOT_CHILD_COMPLETE__RESULT_HAMMER_REQUIRED",
            "source_sha256": sha256(SOURCE_FILE),
            "child": {"source_sha256": child_sha, "processes": child_processes,
                      "arguments": 0, "returncode": child_rc,
                      "result": child_receipt},
            "resources_before": resource_before,
            "resources_under_lock": resource_locked,
            "clean_environment_keys": sorted(CHILD_ENVIRONMENT),
            "attempt_consumed": True, "automatic_retry": False,
            "result_failure_mutually_exclusive": True,
            "claim_boundary": {"result_hammer_required": True,
                               "traffic_cycles_energy_speedup": False,
                               "paper_citable": False},
        })
        _write_exclusive(work / "RUN_COMPLETE.txt",
                         b"PASS_M1143CA_ONE_SHOT_CHILD_COMPLETE__RESULT_HAMMER_REQUIRED\n")
        manifest_sha, outer_sha = _seal_tree(work)
        _rename_noreplace(work, layout.launcher_result); _fsync_dir(layout.result_root)
        require(not tuple(layout.result_root.glob(layout.launcher_failure_prefix + "*")),
                "launcher result/failure mutual exclusion drift")
        return {"status": "PASS_M1143CA_ONE_SHOT_CHILD_COMPLETE__RESULT_HAMMER_REQUIRED",
                "launcher_result": str(layout.launcher_result),
                "child_processes": child_processes, "automatic_retry": False,
                "manifest_sha256": manifest_sha, "outer_seal_file_sha256": outer_sha}
    except BaseException:
        reason = traceback.format_exc()
        if attempt_consumed:
            try:
                work.mkdir(mode=0o700, exist_ok=True)
                _write_json(work / "failure.json", {
                    "schema": "m1143ca_c1_production_schedule_one_shot_failure_r1_v1",
                    "status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE",
                    "phase": phase, "message": reason,
                    "attempt_consumed": True, "child_processes": child_processes,
                    "automatic_retry": False,
                })
                _seal_tree(work); _rename_noreplace(work, failure)
                _fsync_dir(layout.result_root)
                require(not layout.launcher_result.exists(),
                        "launcher failure/result mutual exclusion drift")
            except BaseException:
                pass
        raise
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
            try:
                layout.lock.unlink()
            except FileNotFoundError:
                pass


def source_static_self_test() -> dict[str, Any]:
    preflight = source_preflight()
    require(_namespace_collisions(PRODUCTION_LAYOUT) == (),
            "production namespace not fresh")
    resources = _external_resource_preflight(PRODUCTION_LAYOUT)
    return {
        "status": "PASS_M1143CA_LAUNCHER_STATIC_SELF_TEST__NO_CHILD_NO_M410_NO_PRODUCTION",
        "preflight": preflight, "resources": resources,
        "zero_argument": True,
        "hardcoded_child_command": [str(PYTHON), str(CHILD)],
        "expected_child_processes_future": 1,
        "attempt_created": False, "child_processes": 0,
        "m410_opened": False, "production_records": 0,
        "automatic_retry": False,
    }


def production_main() -> dict[str, Any]:
    return _execute_once(PRODUCTION_LAYOUT, CHILD, CHILD_SHA,
                         PRODUCTION_EXPECTATION, True)


def main() -> int:
    require(len(sys.argv) == 1, "M1143CA accepts zero arguments")
    print(json.dumps(production_main(), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
