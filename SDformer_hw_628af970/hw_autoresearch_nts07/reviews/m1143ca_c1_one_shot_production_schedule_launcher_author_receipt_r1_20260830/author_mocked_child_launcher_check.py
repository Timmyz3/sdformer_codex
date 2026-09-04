#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1143CA author check with mocked Popen; no real child and no M410 open."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
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
M1141_SOURCE = HW / "system_simulator/scripts/run_m1141ca_c1_production_schedule_release_source.py"
M1141_CONTRACT = HW / "contracts/m1141ca_c1_production_schedule_release_source_contract_r1_20260830.json"
M1141_AUTHOR = HW / "reviews/m1141ca_c1_production_schedule_release_source_author_receipt_r1_20260830"
M1142 = HW / "reviews/m1142ca_m1141ca_c1_production_schedule_release_hammer_r1_20260830"
REAL_RESULT = HW / "results/m1143ca_c1_production_schedule_one_shot_launch_r1_20260830"
REAL_ATTEMPT = HW / "results/.m1143ca_c1_production_schedule_one_shot_attempt_consumed"
REAL_CHILD_RESULT = HW / "results/m1141ca_c1_production_schedule_release_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    "source": "184528ee978f3260e7e52d1048d96ecd99a3488d516c85ee3dbc0bcdd2d56be7",
    "contract": "07f27437e12c7e52bf95604f8ee3cba4e0f72f83d795fc92ad2536e7eb555be1",
    "contract_side": "738e5f1e7912421cb1c6d982fb88a2018121fb190ddaa673764a90f779831919",
    "contract_outer": "e2a4a9f1a9962d485574c5193895aad0e573b7ff2ebc37a5b20c34de424adb80",
    "m1141_source": "e2f5d4e0bab472b3a5c7ec5259a805641b800efd3c0e82884e81152eb41cb611",
    "m1141_contract": "4fe7ba960516e889cb1f7140315e1e37a5b42dd00337f136b22a25f1c7ac06d4",
    "m1141_author_outer": "b5602b120cc7c02769a54e67c78588c481776af9f40f3d3359a2938bf2f8b825",
    "m1142_outer": "7a8f8da04bb81a0097d819f98a3bed6e9e40b86a32aef055134f3306bb1850e8",
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


def regular(path: Path, expected: str) -> None:
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink() and
            sha(path) == expected, "identity drift: " + str(path))


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


def production_names() -> tuple[str, ...]:
    root = HW / "results"
    names = []
    for path in (REAL_RESULT, REAL_ATTEMPT, REAL_CHILD_RESULT):
        if path.exists() or path.is_symlink():
            names.append(path.name)
    for pattern in (
        ".m1143ca_c1_production_schedule_one_shot_work.*",
        "m1143ca_c1_production_schedule_one_shot_launch_r1_20260830.failed_or_incomplete.*",
        ".m1141ca_c1_production_schedule_release_work.*",
        "m1141ca_c1_production_schedule_release_r1_20260830.failed_or_incomplete.*",
    ):
        names.extend(path.name for path in root.glob(pattern))
    return tuple(sorted(names))


def load_subject():
    spec = importlib.util.spec_from_file_location("m1143ca_author_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "subject spec")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def static_checks(module) -> dict[str, Any]:
    regular(SOURCE, EXPECTED["source"]); regular(CONTRACT, EXPECTED["contract"])
    regular(Path(str(CONTRACT) + ".sha256"), EXPECTED["contract_side"])
    regular(Path(str(CONTRACT) + ".sha256.seal.sha256"), EXPECTED["contract_outer"])
    regular(M1141_SOURCE, EXPECTED["m1141_source"])
    regular(M1141_CONTRACT, EXPECTED["m1141_contract"])
    regular(M1141_AUTHOR / "SHA256SUMS.seal.sha256", EXPECTED["m1141_author_outer"])
    regular(M1142 / "SHA256SUMS.seal.sha256", EXPECTED["m1142_outer"])
    regular(DOCS359, EXPECTED["docs359"])
    contract = strict_json(CONTRACT)
    text = SOURCE.read_text(encoding="utf-8"); tree = ast.parse(text)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    require(len(functions["production_main"].args.args) == 0 and
            len(functions["main"].args.args) == 0 and
            "len(sys.argv) == 1" in ast.unparse(functions["main"]),
            "zero-argument launcher drift")
    production = ast.unparse(functions["production_main"])
    require("_execute_once(PRODUCTION_LAYOUT, CHILD, CHILD_SHA, PRODUCTION_EXPECTATION, True)" in
            production, "production binding is not immutable")
    execute = functions["_execute_once"]
    popens = [node for node in ast.walk(execute) if isinstance(node, ast.Call) and
             isinstance(node.func, ast.Attribute) and node.func.attr == "Popen"]
    require(len(popens) == 1, "exactly one Popen call in launcher core")
    popen_text = ast.unparse(popens[0])
    require("stdin=subprocess.DEVNULL" in popen_text and
            "env=dict(CHILD_ENVIRONMENT)" in popen_text and
            "close_fds=True" in popen_text and "start_new_session=True" in popen_text,
            "clean child process contract drift")
    require("ROWS" not in {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} and
            "m410r2_h67_q32_runtime_rows_32.memh" not in text,
            "launcher source can directly open M410")
    require(contract["sole_authorization"]["outer_seal_file_sha256"] ==
            EXPECTED["m1142_outer"] and
            contract["launcher"]["arguments"] == 0 and
            contract["launcher"]["exactly_one_child"] is True and
            contract["launcher"]["automatic_retry"] is False and
            contract["authorization"]["different_author_final_launch_hammer_only_next"] is True and
            contract["this_milestone_execution"]["real_child_processes"] == 0,
            "contract authorization drift")
    static = module.source_static_self_test()
    require(static["attempt_created"] is False and static["child_processes"] == 0 and
            static["m410_opened"] is False and static["production_records"] == 0,
            "source static test escaped boundary")
    return {"zero_argument": True, "production_binding_immutable": True,
            "popen_calls_in_core": 1, "direct_m410_path_or_open": False,
            "clean_environment_keys": sorted(module.CHILD_ENVIRONMENT),
            "resource_preflight_pass": True}


def layout(module, root: Path, stem: str):
    return module.LaunchLayout(
        root, root / (stem + "_launch_result"), root / ("." + stem + "_attempt"),
        root / ("." + stem + ".lock"), "." + stem + "_work.",
        stem + "_launch.failed_or_incomplete.", root / (stem + "_child_result"),
        "." + stem + "_child_work.", stem + "_child.failed_or_incomplete.")


def seal_fake_child(module, target: Path, expectation) -> None:
    target.mkdir(mode=0o700)
    records = b'{"controlled_fake":true}\n'
    (target / module.RECORDS_NAME).write_bytes(records)
    release = {
        "schema": "m1141ca_c1_production_schedule_release_r1_v1",
        "status": expectation.status,
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
    callback = None
    returncode = 0
    expected_command = None
    environments = []
    pid = 99_999_991

    def __init__(self, command, **kwargs):
        type(self).calls += 1
        require(command == type(self).expected_command, "mock observed wrong command")
        require(set(command) == set(type(self).expected_command) and len(command) == 2,
                "mock observed child arguments")
        require(kwargs["stdin"] is not None and kwargs["close_fds"] is True and
                kwargs["start_new_session"] is True and
                kwargs["cwd"] == str(HW), "mock child process option drift")
        require(kwargs["env"] == {
            "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1", "TZ": "UTC"}, "mock child env not clean")
        type(self).environments.append(dict(kwargs["env"]))
        if type(self).callback is not None:
            type(self).callback()

    def wait(self, timeout=None):
        require(timeout == 172_800, "child timeout drift")
        return type(self).returncode


def configure_mock(module, child: Path, callback, returncode: int = 0) -> None:
    MockPopen.calls = 0; MockPopen.callback = callback
    MockPopen.returncode = returncode
    MockPopen.expected_command = [str(module.PYTHON), str(child)]
    MockPopen.environments = []


def controlled_positive(module, root: Path) -> dict[str, Any]:
    child = root / "controlled_fake_child.py"
    child.write_text("# controlled fake; never executed\n", encoding="utf-8")
    expected = module.ChildExpectation(module.CHILD_STATUS, 2, 6, "f" * 64)
    target_layout = layout(module, root, "positive")
    configure_mock(module, child,
                   lambda: seal_fake_child(module, target_layout.child_result, expected))
    with patch.object(module.subprocess, "Popen", MockPopen):
        result = module._execute_once(target_layout, child, sha(child), expected, False)
        require(MockPopen.calls == 1 and result["child_processes"] == 1 and
                target_layout.attempt.is_dir() and target_layout.launcher_result.is_dir() and
                target_layout.child_result.is_dir() and
                not tuple(root.glob(target_layout.launcher_failure_prefix + "*")),
                "mocked positive one-shot conservation")
        rejected("second_attempt", lambda: module._execute_once(
            target_layout, child, sha(child), expected, False), "collision")
        require(MockPopen.calls == 1, "second attempt spawned another child")
    return {"mock_child_calls": 1, "real_child_processes": 0,
            "attempt_persistent": True, "launcher_result": True,
            "child_result_verified": True, "second_attempt_rejected": True}


def controlled_failure(module, root: Path) -> dict[str, Any]:
    child = root / "controlled_failure_child.py"
    child.write_text("# controlled failure; never executed\n", encoding="utf-8")
    expected = module.ChildExpectation(module.CHILD_STATUS, 2, 6, "e" * 64)
    target_layout = layout(module, root, "failure")
    configure_mock(module, child, None, returncode=7)
    with patch.object(module.subprocess, "Popen", MockPopen):
        rejected("mock_child_nonzero", lambda: module._execute_once(
            target_layout, child, sha(child), expected, False), "no retry")
    failures = tuple(root.glob(target_layout.launcher_failure_prefix + "*"))
    require(MockPopen.calls == 1 and target_layout.attempt.is_dir() and
            len(failures) == 1 and not target_layout.launcher_result.exists() and
            (failures[0] / module.MANIFEST).is_file() and
            (failures[0] / module.OUTER).is_file(),
            "mocked failure one-shot/quarantine conservation")
    return {"mock_child_calls": 1, "real_child_processes": 0,
            "attempt_persistent": True, "failure_quarantine": True,
            "launcher_result": False, "automatic_retry": False}


def result_failure_attack(module, root: Path) -> None:
    child = root / "controlled_both_child.py"
    child.write_text("# controlled result+failure; never executed\n", encoding="utf-8")
    expected = module.ChildExpectation(module.CHILD_STATUS, 2, 6, "d" * 64)
    target_layout = layout(module, root, "both")
    def callback():
        seal_fake_child(module, target_layout.child_result, expected)
        (root / (target_layout.child_failure_prefix + "attack")).mkdir()
    configure_mock(module, child, callback)
    with patch.object(module.subprocess, "Popen", MockPopen):
        rejected("child_result_failure_coexistence", lambda: module._execute_once(
            target_layout, child, sha(child), expected, False), "mutual exclusion")
    require(MockPopen.calls == 1 and not target_layout.launcher_result.exists(),
            "result/failure attack published launcher result")


def preflight_attacks(module, root: Path) -> None:
    target_layout = layout(module, root, "preflight")
    with patch.object(module.os, "getuid", return_value=0):
        rejected("wrong_uid", lambda: module._external_resource_preflight(target_layout),
                 "UID")
    good = {"MemAvailable": 16 << 30, "CommitLimit": 32 << 30,
            "Committed_AS": 8 << 30}
    with patch.object(module, "_meminfo", return_value={**good, "MemAvailable": 1}):
        rejected("memory_resource", lambda: module._external_resource_preflight(target_layout),
                 "MemAvailable")
    with patch.object(module, "_meminfo", return_value={
            **good, "Committed_AS": (32 << 30) - 1}):
        rejected("commit_headroom", lambda: module._external_resource_preflight(target_layout),
                 "commit headroom")
    with patch.object(module.shutil, "disk_usage", return_value=SimpleNamespace(free=1)):
        rejected("disk_resource", lambda: module._external_resource_preflight(target_layout),
                 "filesystem")
    with patch.object(module, "_same_uid_conflicting_processes", return_value=(123,)):
        rejected("same_uid_process_collision", lambda:
                 module._external_resource_preflight(target_layout), "conflicting")
    target_layout.launcher_result.mkdir()
    require(module._namespace_collisions(target_layout), "namespace collision undetected")
    target_layout.launcher_result.rmdir()
    original = dict(module.CHILD_ENVIRONMENT)
    try:
        module.CHILD_ENVIRONMENT["INHERITED_ATTACK"] = "1"
        rejected("unclean_environment", module.source_preflight, "environment")
    finally:
        module.CHILD_ENVIRONMENT.clear(); module.CHILD_ENVIRONMENT.update(original)
    original_python = module.PYTHON
    try:
        module.PYTHON = Path("/bin/false")
        rejected("wrong_command_binary", module.source_preflight, "identity")
    finally:
        module.PYTHON = original_python


def main() -> None:
    before_names = production_names()
    require(before_names == (), "production namespace not fresh before author check")
    before = {path: sha(path) for path in
              (SOURCE, CONTRACT, M1141_SOURCE, M1141_CONTRACT, DOCS359)}
    module = load_subject(); static = static_checks(module)
    with tempfile.TemporaryDirectory(prefix="m1143ca_mocked_child_") as name:
        root = Path(name)
        positive = controlled_positive(module, root)
        failure = controlled_failure(module, root)
        result_failure_attack(module, root)
        preflight_attacks(module, root)
    after = {path: sha(path) for path in
             (SOURCE, CONTRACT, M1141_SOURCE, M1141_CONTRACT, DOCS359)}
    require(before == after and production_names() == before_names,
            "frozen identity or production namespace changed")
    result = {
        "schema": "m1143ca_one_shot_launcher_author_mechanical_r1_v1",
        "status": "PASS_M1143CA_ONE_SHOT_LAUNCHER_AUTHOR__MOCKED_CHILD_ONLY_NO_PRODUCTION",
        "checks": checks, "attacks_rejected": attacks,
        "static": static, "mocked_positive": positive,
        "mocked_failure": failure,
        "real_boundary": {"m410_open_count": 0, "production_execution": False,
                          "real_child_processes": 0, "production_records": 0,
                          "production_result_created": False,
                          "digest_compiler_driver_full_eda": False},
        "identity_before_after_equal": True,
        "source_sha256": EXPECTED["source"],
        "contract_identity": [EXPECTED["contract"], EXPECTED["contract_side"],
                              EXPECTED["contract_outer"]],
        "m1141ca_source_sha256": EXPECTED["m1141_source"],
        "m1141ca_author_outer_seal_file_sha256": EXPECTED["m1141_author_outer"],
        "m1142ca_outer_seal_file_sha256": EXPECTED["m1142_outer"],
        "authorization": {"different_author_final_launch_hammer_only_next": True,
                          "launcher_execution": False},
        "docs359_sha256": EXPECTED["docs359"],
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
