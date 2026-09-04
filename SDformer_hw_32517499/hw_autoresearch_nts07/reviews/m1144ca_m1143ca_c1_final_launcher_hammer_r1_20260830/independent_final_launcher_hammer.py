#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1144CA independent final launcher hammer.

All child executions are controlled Python mocks.  This program never opens
M410, never invokes M1141CA, and never touches the production namespace.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/run_m1143ca_c1_production_schedule_one_shot_launcher_source.py"
CONTRACT = HW / "contracts/m1143ca_c1_one_shot_production_schedule_launcher_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1143ca_c1_one_shot_production_schedule_launcher_author_receipt_r1_20260830"
M1141_SOURCE = HW / "system_simulator/scripts/run_m1141ca_c1_production_schedule_release_source.py"
M1142 = HW / "reviews/m1142ca_m1141ca_c1_production_schedule_release_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
REAL_RESULT = HW / "results/m1143ca_c1_production_schedule_one_shot_launch_r1_20260830"
REAL_ATTEMPT = HW / "results/.m1143ca_c1_production_schedule_one_shot_attempt_consumed"
REAL_CHILD_RESULT = HW / "results/m1141ca_c1_production_schedule_release_r1_20260830"
EXPECTED = {
    "source": "184528ee978f3260e7e52d1048d96ecd99a3488d516c85ee3dbc0bcdd2d56be7",
    "contract": "07f27437e12c7e52bf95604f8ee3cba4e0f72f83d795fc92ad2536e7eb555be1",
    "contract_side": "738e5f1e7912421cb1c6d982fb88a2018121fb190ddaa673764a90f779831919",
    "contract_outer": "e2a4a9f1a9962d485574c5193895aad0e573b7ff2ebc37a5b20c34de424adb80",
    "author_outer": "961897966dc7bd96de087d62933569a51c1ad3277f5171381ac5336b09289615",
    "m1141_source": "e2f5d4e0bab472b3a5c7ec5259a805641b800efd3c0e82884e81152eb41cb611",
    "m1142_outer": "7a8f8da04bb81a0097d819f98a3bed6e9e40b86a32aef055134f3306bb1850e8",
    "python": "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
checks = 0
attacks: dict[str, str] = {}


class CheckFailure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise CheckFailure(message)


def rejected(label: str, action: Callable[[], Any], contains: str | None = None) -> None:
    try:
        action()
    except BaseException as error:
        if contains is not None:
            require(contains in str(error), label + " wrong rejection: " + str(error))
        attacks[label] = type(error).__name__ + ": " + str(error)
        return
    raise CheckFailure("attack accepted: " + label)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: str, owner: int | None = 1913) -> None:
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink() and
            (owner is None or path.stat().st_uid == owner) and sha(path) == expected,
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
                          CheckFailure("nonfinite JSON: " + token)))


def verify_double(path: Path) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    regular(path, EXPECTED["contract"]); regular(side, EXPECTED["contract_side"])
    regular(outer, EXPECTED["contract_outer"])
    require(side.read_text(encoding="utf-8").split() == [EXPECTED["contract"], path.name] and
            outer.read_text(encoding="utf-8").split() == [EXPECTED["contract_side"], side.name],
            "contract double seal content drift")


def verify_tree(directory: Path, expected_outer: str) -> dict[str, Any]:
    require(directory.is_dir() and not directory.is_symlink() and
            directory.stat().st_uid == 1913, "sealed tree owner/type drift")
    manifest = directory / "SHA256SUMS"; outer = directory / "SHA256SUMS.seal.sha256"
    regular(outer, expected_outer)
    require(outer.read_text(encoding="utf-8").split() == [sha(manifest), "SHA256SUMS"],
            "outer seal content drift")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*"); rel = Path(name)
        require(name not in expected and name == rel.as_posix() and
                not rel.is_absolute() and ".." not in rel.parts,
                "manifest unsafe/duplicate member")
        expected[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "sealed symlink")
        if stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), "sealed special member")
    require(actual == set(expected), "sealed exact member set drift")
    for name, digest in expected.items():
        regular(directory / name, digest)
    return strict_json(directory / "review.json")


def production_names() -> tuple[str, ...]:
    root = HW / "results"; names = []
    for path in (REAL_RESULT, REAL_ATTEMPT, REAL_CHILD_RESULT):
        if path.exists() or path.is_symlink(): names.append(path.name)
    for pattern in (
        ".m1143ca_c1_production_schedule_one_shot_work.*",
        "m1143ca_c1_production_schedule_one_shot_launch_r1_20260830.failed_or_incomplete.*",
        ".m1141ca_c1_production_schedule_release_work.*",
        "m1141ca_c1_production_schedule_release_r1_20260830.failed_or_incomplete.*",
    ):
        names.extend(path.name for path in root.glob(pattern))
    lock = Path("/tmp/m1143ca_c1_production_schedule_one_shot.lock")
    if lock.exists() or lock.is_symlink(): names.append(str(lock))
    return tuple(sorted(names))


def load_subject():
    spec = importlib.util.spec_from_file_location("m1144ca_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "subject import spec")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def static_checks(module) -> dict[str, Any]:
    regular(SOURCE, EXPECTED["source"]); verify_double(CONTRACT)
    author = verify_tree(AUTHOR, EXPECTED["author_outer"])
    regular(M1141_SOURCE, EXPECTED["m1141_source"])
    verify_tree(M1142, EXPECTED["m1142_outer"])
    regular(module.PYTHON, EXPECTED["python"], None); regular(DOCS359, EXPECTED["docs359"])
    contract = strict_json(CONTRACT)
    text = SOURCE.read_text(encoding="utf-8"); tree = ast.parse(text)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    production_text = ast.unparse(functions["production_main"])
    main_text = ast.unparse(functions["main"])
    execute = functions["_execute_once"]; execute_text = ast.unparse(execute)
    popens = [node for node in ast.walk(execute) if isinstance(node, ast.Call) and
             isinstance(node.func, ast.Attribute) and node.func.attr == "Popen"]
    require(len(functions["production_main"].args.args) == 0 and
            len(functions["main"].args.args) == 0 and "len(sys.argv) == 1" in main_text,
            "launcher is not immutable zero argument")
    require("_execute_once(PRODUCTION_LAYOUT, CHILD, CHILD_SHA, PRODUCTION_EXPECTATION, True)" in
            production_text, "fixed production child binding drift")
    require(len(popens) == 1, "launcher must contain exactly one Popen")
    popen_text = ast.unparse(popens[0])
    require(all(token in popen_text for token in (
                "stdin=subprocess.DEVNULL", "stdout=stdout", "stderr=stderr",
                "cwd=str(HW)", "env=dict(CHILD_ENVIRONMENT)", "close_fds=True",
                "start_new_session=True")), "Popen isolation drift")
    require(module.CHILD_ENVIRONMENT == {
                "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PATH": "/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1", "TZ": "UTC"}, "clean seven-key env drift")
    require(execute_text.count("_namespace_collisions") >= 2 and
            execute_text.count("_external_resource_preflight") == 2 and
            execute_text.index("layout.attempt.mkdir") < execute_text.index("subprocess.Popen"),
            "double preflight or attempt-before-child drift")
    require("fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)" in execute_text and
            "ignore_lock=True" in execute_text and "_rename_noreplace(work, layout.launcher_result)" in
            execute_text and "'automatic_retry': False" in execute_text,
            "lock/noreplace/no-retry drift")
    require(author["authorization"]["launcher_execution"] is False and
            contract["launcher"]["arguments"] == 0 and
            contract["launcher"]["exactly_one_child"] is True and
            contract["launcher"]["automatic_retry"] is False and
            contract["authorization"]["different_author_final_launch_hammer_only_next"] is True,
            "author/contract authorization drift")
    static = module.source_static_self_test()
    require(static["child_processes"] == 0 and static["m410_opened"] is False and
            static["production_records"] == 0 and static["attempt_created"] is False,
            "static test escaped no-production boundary")
    return {"source_identity": True, "all_live_exact_authorities": True,
            "zero_argument_fixed_child": True, "popen_sites": 1,
            "clean_environment_keys": sorted(module.CHILD_ENVIRONMENT),
            "double_preflight": True, "attempt_before_child": True,
            "atomic_noreplace_publish": True, "automatic_retry": False}


def layout(module, root: Path, stem: str):
    return module.LaunchLayout(
        root, root / (stem + "_launch_result"), root / ("." + stem + "_attempt"),
        root / ("." + stem + ".lock"), "." + stem + "_work.",
        stem + "_launch.failed_or_incomplete.", root / (stem + "_child_result"),
        "." + stem + "_child_work.", stem + "_child.failed_or_incomplete.")


def seal_fake_child(module, target: Path, expectation, bad_status: bool = False) -> None:
    target.mkdir(mode=0o700)
    records = b'{"controlled_fake":true}\n'
    (target / module.RECORDS_NAME).write_bytes(records)
    release = {
        "schema": "m1141ca_c1_production_schedule_release_r1_v1",
        "status": "ATTACK" if bad_status else expectation.status,
        "source_rows": {"sha256": expectation.rows_sha256},
        "geometry": {"tasks": expectation.tasks, "records": expectation.records,
                     "axes": list(module.AXES)},
        "records": {"file": module.RECORDS_NAME, "count": expectation.records,
                    "sha256": hashlib.sha256(records).hexdigest(),
                    "axis_order_within_each_task": list(module.AXES),
                    "axis_counts": {axis: expectation.tasks for axis in module.AXES}},
        "authority": {"m1016_source_sha256": module.M1016_SHA,
                      "m1139ca_source_sha256": module.M1139_SHA,
                      "m1140ca_outer_seal_file_sha256": module.M1140CA_OUTER_SHA},
        "claim_boundary": {"digest_compiler": False, "real_driver": False,
                           "paper_citable": False},
    }
    (target / module.RELEASE_NAME).write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    module._seal_tree(target)


class MockPopen:
    calls = 0
    callback: Callable[[], Any] | None = None
    returncode = 0
    expected_command: list[str] = []
    attempt: Path | None = None
    pid = 99_999_971

    def __init__(self, command, **kwargs):
        type(self).calls += 1
        require(command == type(self).expected_command and len(command) == 2,
                "mock observed arguments or command drift")
        require(type(self).attempt is not None and type(self).attempt.is_dir() and
                (type(self).attempt / "SHA256SUMS.seal.sha256").is_file(),
                "child spawned before sealed attempt")
        require(kwargs["stdin"] is subprocess.DEVNULL and kwargs["close_fds"] is True and
                kwargs["start_new_session"] is True and kwargs["cwd"] == str(HW),
                "mock Popen isolation drift")
        require(kwargs["env"] == {
            "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1", "TZ": "UTC"}, "mock inherited environment")
        require(kwargs["stdout"].name.endswith("child.stdout.log") and
                kwargs["stderr"].name.endswith("child.stderr.log"), "mock log binding drift")
        if type(self).callback is not None: type(self).callback()

    def wait(self, timeout=None):
        require(timeout == 172_800, "child timeout drift")
        return type(self).returncode


def configure(module, child: Path, target_layout, callback, rc: int = 0) -> None:
    MockPopen.calls = 0; MockPopen.callback = callback; MockPopen.returncode = rc
    MockPopen.expected_command = [str(module.PYTHON), str(child)]
    MockPopen.attempt = target_layout.attempt


def fake_resources(module, calls: list[int]):
    def value(target_layout):
        calls.append(len(calls) + 1)
        return {"uid": 1913, "cpu_count": 8, "mem_available_bytes": 16 << 30,
                "commit_headroom_bytes": 16 << 30,
                "result_filesystem_free_bytes": 32 << 30,
                "same_uid_conflicting_processes": 0}
    return value


def success_case(module, root: Path) -> dict[str, Any]:
    child = root / "success_child.py"; child.write_text("# fake\n", encoding="utf-8")
    target = layout(module, root, "success")
    expectation = module.ChildExpectation(module.CHILD_STATUS, 2, 6, "f" * 64)
    configure(module, child, target, lambda: seal_fake_child(module, target.child_result, expectation))
    resource_calls: list[int] = []
    with patch.object(module.subprocess, "Popen", MockPopen), \
         patch.object(module, "_external_resource_preflight", fake_resources(module, resource_calls)):
        result = module._execute_once(target, child, sha(child), expectation, False)
        require(MockPopen.calls == 1 and resource_calls == [1, 2] and
                result["child_processes"] == 1 and target.attempt.is_dir() and
                target.launcher_result.is_dir() and target.child_result.is_dir(),
                "success conservation/double resource preflight")
        rejected("second_attempt", lambda: module._execute_once(
            target, child, sha(child), expectation, False), "collision")
        require(MockPopen.calls == 1, "second attempt spawned child")
    return {"controlled_child_calls": 1, "resource_preflight_calls": 2,
            "sealed_attempt_before_child": True, "sealed_atomic_result": True,
            "second_attempt_rejected": True}


def failure_case(module, root: Path) -> dict[str, Any]:
    child = root / "failure_child.py"; child.write_text("# fake\n", encoding="utf-8")
    target = layout(module, root, "failure")
    expectation = module.ChildExpectation(module.CHILD_STATUS, 2, 6, "e" * 64)
    configure(module, child, target, None, 7); calls: list[int] = []
    with patch.object(module.subprocess, "Popen", MockPopen), \
         patch.object(module, "_external_resource_preflight", fake_resources(module, calls)):
        rejected("child_nonzero_no_retry", lambda: module._execute_once(
            target, child, sha(child), expectation, False), "no retry")
    failures = tuple(root.glob(target.launcher_failure_prefix + "*"))
    require(MockPopen.calls == 1 and calls == [1, 2] and target.attempt.is_dir() and
            len(failures) == 1 and not target.launcher_result.exists() and
            (failures[0] / module.MANIFEST).is_file() and
            (failures[0] / module.OUTER).is_file(), "failure quarantine/no-retry drift")
    return {"controlled_child_calls": 1, "attempt_persistent": True,
            "sealed_failure_quarantine": True, "automatic_retry": False}


def coexistence_case(module, root: Path) -> None:
    child = root / "coexist_child.py"; child.write_text("# fake\n", encoding="utf-8")
    target = layout(module, root, "coexist")
    expectation = module.ChildExpectation(module.CHILD_STATUS, 2, 6, "d" * 64)
    def callback():
        seal_fake_child(module, target.child_result, expectation)
        (root / (target.child_failure_prefix + "attack")).mkdir()
    configure(module, child, target, callback); calls: list[int] = []
    with patch.object(module.subprocess, "Popen", MockPopen), \
         patch.object(module, "_external_resource_preflight", fake_resources(module, calls)):
        rejected("child_result_failure_coexistence", lambda: module._execute_once(
            target, child, sha(child), expectation, False), "mutual exclusion")
    require(MockPopen.calls == 1 and not target.launcher_result.exists(),
            "coexistence attack published result")


def malformed_result_case(module, root: Path) -> None:
    child = root / "malformed_child.py"; child.write_text("# fake\n", encoding="utf-8")
    target = layout(module, root, "malformed")
    expectation = module.ChildExpectation(module.CHILD_STATUS, 2, 6, "c" * 64)
    configure(module, child, target,
              lambda: seal_fake_child(module, target.child_result, expectation, True))
    calls: list[int] = []
    with patch.object(module.subprocess, "Popen", MockPopen), \
         patch.object(module, "_external_resource_preflight", fake_resources(module, calls)):
        rejected("malformed_child_semantics", lambda: module._execute_once(
            target, child, sha(child), expectation, False), "semantic")
    require(MockPopen.calls == 1 and not target.launcher_result.exists(),
            "malformed child published launcher result")


def namespace_attacks(module, root: Path) -> None:
    markers = ("launcher_result", "attempt", "child_result", "launcher_work",
               "launcher_failure", "child_work", "child_failure", "lock")
    for marker in markers:
        case_root = root / ("ns_" + marker); case_root.mkdir()
        target = layout(module, case_root, marker)
        path = {
            "launcher_result": target.launcher_result, "attempt": target.attempt,
            "child_result": target.child_result, "launcher_work": case_root / (target.launcher_work_prefix + "x"),
            "launcher_failure": case_root / (target.launcher_failure_prefix + "x"),
            "child_work": case_root / (target.child_work_prefix + "x"),
            "child_failure": case_root / (target.child_failure_prefix + "x"), "lock": target.lock,
        }[marker]
        path.mkdir() if marker != "lock" else path.touch()
        require(module._namespace_collisions(target), "namespace marker missed: " + marker)
    target = layout(module, root, "under_lock_namespace")
    child = root / "under_lock_child.py"; child.write_text("# fake\n", encoding="utf-8")
    expectation = module.ChildExpectation(module.CHILD_STATUS, 2, 6, "b" * 64)
    configure(module, child, target, None); namespace_calls = []
    def collide(_layout, ignore_lock=False):
        namespace_calls.append(ignore_lock)
        return () if len(namespace_calls) == 1 else ("injected-under-lock",)
    resource_calls: list[int] = []
    with patch.object(module, "_namespace_collisions", collide), \
         patch.object(module, "_external_resource_preflight", fake_resources(module, resource_calls)), \
         patch.object(module.subprocess, "Popen", MockPopen):
        rejected("namespace_changed_under_lock", lambda: module._execute_once(
            target, child, sha(child), expectation, False), "under lock")
    require(namespace_calls == [False, True] and resource_calls == [1] and
            MockPopen.calls == 0 and not target.attempt.exists() and not target.lock.exists(),
            "under-lock namespace double preflight conservation")


def resource_attacks(module, root: Path) -> None:
    target = layout(module, root, "resources")
    good = {"MemAvailable": 16 << 30, "CommitLimit": 32 << 30,
            "Committed_AS": 8 << 30}
    with patch.object(module.os, "getuid", return_value=0):
        rejected("wrong_uid", lambda: module._external_resource_preflight(target), "UID")
    with patch.object(module.os, "cpu_count", return_value=1):
        rejected("cpu_floor", lambda: module._external_resource_preflight(target), "CPU")
    with patch.object(module, "_meminfo", return_value={**good, "MemAvailable": 1}):
        rejected("memory_floor", lambda: module._external_resource_preflight(target), "MemAvailable")
    with patch.object(module, "_meminfo", return_value={**good, "Committed_AS": (32 << 30) - 1}):
        rejected("commit_floor", lambda: module._external_resource_preflight(target), "commit")
    with patch.object(module.shutil, "disk_usage", return_value=SimpleNamespace(free=1)):
        rejected("disk_floor", lambda: module._external_resource_preflight(target), "filesystem")
    with patch.object(module, "_same_uid_conflicting_processes", return_value=(1234,)):
        rejected("same_uid_process", lambda: module._external_resource_preflight(target), "conflicting")

    child = root / "resource_recheck_child.py"; child.write_text("# fake\n", encoding="utf-8")
    expectation = module.ChildExpectation(module.CHILD_STATUS, 2, 6, "a" * 64)
    configure(module, child, target, None); calls = []
    def second_fails(_layout):
        calls.append(len(calls) + 1)
        if len(calls) == 2: raise module.Failure("injected locked resource drift")
        return {"ok": True}
    with patch.object(module, "_external_resource_preflight", second_fails), \
         patch.object(module.subprocess, "Popen", MockPopen):
        rejected("resource_changed_under_lock", lambda: module._execute_once(
            target, child, sha(child), expectation, False), "resource drift")
    require(calls == [1, 2] and MockPopen.calls == 0 and not target.attempt.exists() and
            not target.lock.exists(), "locked resource recheck conservation")


def main() -> None:
    before_names = production_names()
    require(before_names == (), "production namespace not fresh before independent hammer")
    frozen = (SOURCE, CONTRACT, Path(str(CONTRACT) + ".sha256"),
              Path(str(CONTRACT) + ".sha256.seal.sha256"), M1141_SOURCE, DOCS359)
    before = {str(path): sha(path) for path in frozen}
    module = load_subject(); static = static_checks(module)
    with tempfile.TemporaryDirectory(prefix="m1144ca_controlled.", dir="/tmp") as name:
        root = Path(name)
        positive = success_case(module, root)
        failure = failure_case(module, root)
        coexistence_case(module, root)
        malformed_result_case(module, root)
        namespace_attacks(module, root)
        resource_attacks(module, root)
    after = {str(path): sha(path) for path in frozen}
    require(before == after and production_names() == before_names,
            "frozen identities or production namespace changed")
    result = {
        "schema": "m1144ca_m1143ca_c1_final_launcher_hammer_mechanical_r1_v1",
        "status": "PASS_M1144CA_INDEPENDENT_FINAL_LAUNCHER_HAMMER__ROOT_EXTERNAL_PREFLIGHT_THEN_ONE_EXACT_EXECUTION_ONLY",
        "checks": checks, "attacks_rejected": attacks, "static": static,
        "controlled_success": positive, "controlled_failure": failure,
        "real_boundary": {"real_child_processes": 0, "m410_open_count": 0,
                          "production_records": 0, "production_execution": False,
                          "production_namespace_unchanged": True},
        "identity": {"launcher_source_sha256": EXPECTED["source"],
                     "launcher_contract_outer_seal_file_sha256": EXPECTED["contract_outer"],
                     "launcher_author_outer_seal_file_sha256": EXPECTED["author_outer"],
                     "m1141ca_source_sha256": EXPECTED["m1141_source"],
                     "m1142ca_outer_seal_file_sha256": EXPECTED["m1142_outer"],
                     "docs359_sha256": EXPECTED["docs359"]},
        "authorization": {"root_external_preflight_required": True,
                          "exact_launcher_executions_authorized": 1,
                          "exact_command": [str(module.PYTHON), str(SOURCE)],
                          "arguments": 0, "automatic_retry": False,
                          "hammer_execution": False},
        "claim_boundary": {"launcher_only": True, "child_result_hammer_required": True,
                           "traffic_cycles_energy_speedup": False,
                           "paper_citable": False, "paper_ppa_ready": False},
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
