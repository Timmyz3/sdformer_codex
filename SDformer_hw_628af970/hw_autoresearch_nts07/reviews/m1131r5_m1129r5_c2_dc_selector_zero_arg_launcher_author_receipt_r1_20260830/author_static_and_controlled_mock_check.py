#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1131r5 launcher author check; static/controlled mock only, no launch or EDA."""
from __future__ import annotations

import ast
import copy
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
HW = HERE.parent.parent
LAUNCHER = HW / "dc_handoff/scripts/run_m1129r5_c2_dc_selector_async_observation_authorized_launch_r1.py"
R4_LAUNCHER = HW / "dc_handoff/scripts/run_m1122r4_c2_dc_selector_async_observation_authorized_launch_r1.py"
RECEIPT = HW / "contracts/m1129r5_c2_dc_selector_async_observation_authorized_launch_receipt_r1_20260830.json"
ENGINE = HW / "dc_handoff/scripts/m1129r5_c2_real_module_async_observation_engine_source_r1.py"
CONTRACT = HW / "contracts/m1129r5_c2_dc_selector_async_observation_source_contract_r1_20260830.json"
ENGINE_AUTHOR = HW / "reviews/m1129r5_c2_real_module_engine_author_receipt_r1_20260830"
M1130 = HW / "reviews/m1130r5_m1129r5_c2_dc_selector_engine_hammer_r1_20260830"
M1121 = HW / "reviews/m1121_m1112r3_c2_dc_invocation_failure_audit_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    "launcher": "0a66b0c7c5c514d7b5a17872069701f9dcd12ace6c8bcb82b4e7282706248c64",
    "r4_launcher": "405cf1bd8a6af412ce44a727b47b27e767cb6bfc1fdcba4c9cb1cae4dc8f63" if False else
                   "405cf1bd8a6af412ce44a727b47db90c14923054678946378bdfe2646a95ec78",
    "receipt": "801af8a4c35aae5c18f1aad7ab90127d2095e1431a739395975f75888e1b89db",
    "receipt_side": "509e0d9d93d286592fa9ef11d0c283b8c40d397ff0350f61a51b17d3b9f23b65",
    "receipt_outer": "170478cf4f9ccd554d930820db7b908072b270fd96f8179b9008d523314c6cc3",
    "engine": "c8fd3366ecf6c4377b62e5717d959348c08192ea8bdbd0afd3b0e566bd6fbd0b",
    "contract": "25cfbf9e2d75333e27a1162ab202b9b6a9b305876ee92ce6ed9f6d30513f370d",
    "contract_outer": "b5a389b2b76a83f6449bfcbc928c416df877f611cfbd987d828552cb4bdf50cf",
    "engine_author_outer": "f31e0b11049229d17d2c91eb6290ff98f5fe963dd32d0329403237d894ce2ef3",
    "m1130_outer": "71ec2e0bfa68d63d971d60e42c6bf4d8e7e990739c2647b0045f15245b0a3ad0",
    "m1121_outer": "dc0135b61750134c37b6e3eba47350a0d9838c9ed0a07ca5ecab3bb93c3ff828",
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


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_regular(path: Path, digest: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and sha(path) == digest,
            "identity drift: " + str(path))


def strict_json(path: Path) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          CheckFailure("nonfinite JSON: " + token)))


def verify_double(path: Path) -> None:
    side = Path(str(path) + ".sha256"); outer = Path(str(path) + ".sha256.seal.sha256")
    verify_regular(path, EXPECTED["receipt"]); verify_regular(side, EXPECTED["receipt_side"])
    verify_regular(outer, EXPECTED["receipt_outer"])
    require(side.read_text(encoding="utf-8").split() ==
            [EXPECTED["receipt"], path.relative_to(HW).as_posix()], "receipt side content")
    require(outer.read_text(encoding="utf-8").split() ==
            [EXPECTED["receipt_side"], side.relative_to(HW).as_posix()], "receipt outer content")


def verify_flat_outer(directory: Path, outer_sha: str) -> dict[str, Any]:
    require(stat.S_ISDIR(directory.lstat().st_mode) and not directory.is_symlink(),
            "sealed directory drift")
    manifest = directory / "SHA256SUMS"; outer = directory / "SHA256SUMS.seal.sha256"
    verify_regular(outer, outer_sha)
    require(outer.read_text(encoding="utf-8").split() == [sha(manifest), "SHA256SUMS"],
            "sealed outer content")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*"); rel = Path(name)
        require(name not in expected and not rel.is_absolute() and ".." not in rel.parts,
                "unsafe manifest")
        expected[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}: continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "sealed symlink")
        if stat.S_ISREG(mode): actual.add(name)
        else: require(stat.S_ISDIR(mode), "sealed special")
    require(actual == set(expected), "sealed exact-member mismatch")
    for name, digest in expected.items(): verify_regular(directory / name, digest)
    return strict_json(directory / "review.json")


def function_node(source: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    node = next((item for item in tree.body
                 if isinstance(item, ast.FunctionDef) and item.name == name), None)
    require(node is not None, "missing function: " + name)
    return node


def analyze(source: str, receipt: dict[str, Any]) -> dict[str, Any]:
    tree = ast.parse(source)
    main = function_node(source, "main")
    validate = function_node(source, "validate_hardcoded_authorities")
    namespace = function_node(source, "namespace_resource_gate")
    collision = function_node(source, "collision_gate")
    clean = function_node(source, "clean_child_environment")
    main_text = ast.unparse(main); validate_text = ast.unparse(validate)
    namespace_text = ast.unparse(namespace); collision_text = ast.unparse(collision)
    clean_text = ast.unparse(clean)
    require('len(sys.argv) == 1' in validate_text and 'os.environ == ROOT_ENV' in validate_text,
            "zero-arg/env-i gate drift")
    require("ATTEMPT.exists()" in namespace_text and "RESULT.exists()" in namespace_text and
            "LOCK.exists()" in namespace_text and "WORK_GLOB" in namespace_text and
            "FAILURE_GLOB" in namespace_text, "fresh namespace gate drift")
    require("EDA_PROCESS_NAMES" in collision_text and "/usr/bin/pgrep" in collision_text and
            "MemAvailable" in source and "CommitLimit" in source and "Committed_AS" in source and
            "MIN_MEM_AVAILABLE_KIB = 8 * 1024 * 1024" in source and
            "MIN_COMMIT_HEADROOM_KIB = 8 * 1024 * 1024" in source,
            "collision/memory gate drift")
    require(main_text.count("subprocess.run") == 1 and
            '[str(PYTHON), \'-I\', str(ENGINE), \'--authorized-launch\']' in main_text and
            "env=clean_child_environment(private_home)" in main_text and
            "return completed.returncode" in main_text and not any(
                isinstance(item, (ast.For, ast.While)) for item in ast.walk(main)),
            "single child/no retry drift")
    require("os.environ" not in clean_text and "HOME" in clean_text and
            "SNPSLMD_LICENSE_FILE" in clean_text and "LM_LICENSE_FILE" in clean_text,
            "clean child env drift")
    require(EXPECTED["engine"] in source and EXPECTED["contract_outer"] in source and
            EXPECTED["engine_author_outer"] in source and EXPECTED["m1130_outer"] in source and
            EXPECTED["m1121_outer"] in source,
            "hardcoded authority binding drift")
    require(receipt == {
        "arguments": 0, "attempt_now": False, "automatic_retry": False,
        "caller_environment_forwarded": False, "caller_selected_authority_allowed": False,
        "dc_now": False, "engine_author_receipt_outer_seal_file_sha256": EXPECTED["engine_author_outer"],
        "engine_contract_outer_seal_file_sha256": EXPECTED["contract_outer"],
        "engine_contract_sha256": EXPECTED["contract"], "engine_sha256": EXPECTED["engine"],
        "launch_now": False, "launcher_sha256": EXPECTED["launcher"],
        "m1121_outer_seal_file_sha256": EXPECTED["m1121_outer"],
        "m1130r5_outer_seal_file_sha256": EXPECTED["m1130_outer"],
        "m1132r5_required": True, "mapped_vcs_now": False, "maximum_attempts": 1,
        "paper_citable": False,
        "schema": "m1129r5_c2_dc_selector_authorized_launch_receipt_r1_v1",
        "status": "M1129R5_LAUNCH_SOURCE_FROZEN__M1132R5_REQUIRED__NO_EDA",
    }, "launch receipt exact schema/boundary drift")
    require("170478cf4f9ccd554d930820db7b908072b270fd96f8179b9008d523314c6cc3" not in source and
            "m1132r5_outer_seal_file_sha256" not in source and
            "m1132r5_outer_seal_file_sha256" not in receipt,
            "future launch hammer hash cycle")
    return {"zero_arguments": True, "env_i_root": True, "fresh_namespace": True,
            "collision_memory_gates": True, "single_child": True, "automatic_retry": False,
            "future_hammer_hash_cycle": False}


def load_launcher():
    spec = importlib.util.spec_from_file_location("m1131r5_launcher_subject", LAUNCHER)
    require(spec is not None and spec.loader is not None, "cannot load launcher")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module); return module


def rejected(label: str, action: Callable[[], Any]) -> None:
    try: action()
    except Exception as error:
        attacks[label] = type(error).__name__ + ": " + str(error); return
    raise CheckFailure("attack accepted: " + label)


def namespace_snapshot() -> dict[str, Any]:
    return {"attempt": (HW / "results/.m1129r5_c2_dc_selector_async_observation_dc_mapped_vcs_attempt_consumed").exists(),
            "result": (HW / "results/m1129r5_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830").exists(),
            "work": sorted(path.name for path in (HW / "results").glob(".m1129r5_c2_dc_selector_async_observation_dc_mapped_vcs_work.*")),
            "failure": sorted(path.name for path in (HW / "results").glob("m1129r5_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.*")),
            "lock": Path("/tmp/m1129r5_c2_dc_selector_async_observation_eda.lock").exists()}


def main() -> None:
    paths = (LAUNCHER, R4_LAUNCHER, RECEIPT, ENGINE, CONTRACT, DOCS359)
    before = {path: sha(path) for path in paths}; namespace_before = namespace_snapshot()
    require(namespace_before == {"attempt": False, "result": False, "work": [],
                                 "failure": [], "lock": False}, "r5 namespace not fresh")
    verify_regular(LAUNCHER, EXPECTED["launcher"]); verify_regular(R4_LAUNCHER, EXPECTED["r4_launcher"])
    verify_double(RECEIPT); verify_regular(ENGINE, EXPECTED["engine"])
    verify_regular(CONTRACT, EXPECTED["contract"]); verify_regular(DOCS359, EXPECTED["docs359"])
    engine_author = verify_flat_outer(ENGINE_AUTHOR, EXPECTED["engine_author_outer"])
    m1130 = verify_flat_outer(M1130, EXPECTED["m1130_outer"])
    m1121 = verify_flat_outer(M1121, EXPECTED["m1121_outer"])
    require(engine_author["status"] ==
            "PASS_M1129R5_REAL_MODULE_ENGINE_SOURCE_AUTHOR_RECEIPT__M1130R5_REQUIRED__NO_EDA" and
            m1130["status"] ==
            "PASS_M1130R5_M1129R5_ENGINE_HAMMER__AUTHOR_ZERO_ARG_LAUNCHER_ONLY__NO_EDA" and
            m1121["status"] ==
            "PASS_M1121_FAILURE_AUDIT__M1112R3_DO_NOT_RETRY__ADDITIVE_R4_INVOCATION_SELECTOR_REPAIR_ONLY",
            "authority status drift")
    source = LAUNCHER.read_text(encoding="utf-8"); receipt = strict_json(RECEIPT)
    static = analyze(source, receipt)
    launcher = load_launcher()
    clean = launcher.clean_child_environment(Path("/tmp/m1131r5_mock_home"))
    require(clean == {"HOME": "/tmp/m1131r5_mock_home", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                      "PATH": "/usr/bin:/bin", "TMPDIR": "/tmp", "PYTHONNOUSERSITE": "1",
                      "PYTHONDONTWRITEBYTECODE": "1", "SNPSLMD_LICENSE_FILE": "27030@ic.ismd-nemo",
                      "LM_LICENSE_FILE": "/opt/synopsys/Synopsys.dat"},
            "clean child environment exact constants drift")
    with patch.object(launcher, "collision_gate", return_value=[]), \
         patch.object(launcher, "read_meminfo", return_value={
             "MemAvailable": 9 * 1024 * 1024, "CommitLimit": 20 * 1024 * 1024,
             "Committed_AS": 10 * 1024 * 1024}):
        resource = launcher.namespace_resource_gate()
    require(resource["eda_collisions"] == [] and resource["commit_headroom_kib"] == 10 * 1024 * 1024,
            "controlled collision/memory gate drift")
    child_calls = []
    def fake_run(argv, cwd, env, close_fds, check):
        child_calls.append({"argv": argv, "cwd": cwd, "env": env,
                            "close_fds": close_fds, "check": check})
        return SimpleNamespace(returncode=7)
    with patch.object(launcher, "validate_hardcoded_authorities", return_value={}), \
         patch.object(launcher, "namespace_resource_gate", return_value={}), \
         patch.object(launcher.subprocess, "run", side_effect=fake_run):
        rc = launcher.main()
    require(rc == 7 and len(child_calls) == 1 and child_calls[0]["argv"] == [
        str(launcher.PYTHON), "-I", str(launcher.ENGINE), "--authorized-launch"] and
        child_calls[0]["cwd"] == str(HW) and child_calls[0]["close_fds"] is True and
        child_calls[0]["check"] is False and set(child_calls[0]["env"]) == {
            "HOME", "LANG", "LC_ALL", "PATH", "TMPDIR", "PYTHONNOUSERSITE",
            "PYTHONDONTWRITEBYTECODE", "SNPSLMD_LICENSE_FILE", "LM_LICENSE_FILE"},
            "controlled single-child launch mock drift")

    bad = copy.deepcopy(receipt); bad["arguments"] = 1
    rejected("receipt_arguments", lambda: analyze(source, bad))
    bad = copy.deepcopy(receipt); bad["caller_environment_forwarded"] = True
    rejected("receipt_caller_env", lambda: analyze(source, bad))
    bad = copy.deepcopy(receipt); bad["maximum_attempts"] = 2
    rejected("receipt_max_two", lambda: analyze(source, bad))
    bad = copy.deepcopy(receipt); bad["automatic_retry"] = True
    rejected("receipt_retry", lambda: analyze(source, bad))
    bad = copy.deepcopy(receipt); bad["m1132r5_outer_seal_file_sha256"] = "0" * 64
    rejected("receipt_future_hash_cycle", lambda: analyze(source, bad))
    rejected("launcher_engine_drift", lambda: analyze(source.replace(EXPECTED["engine"], "0" * 64), receipt))
    rejected("launcher_no_zero_arg", lambda: analyze(source.replace("len(sys.argv) == 1", "len(sys.argv) >= 1"), receipt))
    rejected("launcher_env_forward", lambda: analyze(source.replace("env=clean_child_environment(private_home)", "env=os.environ.copy()"), receipt))
    rejected("launcher_retry_loop", lambda: analyze(source.replace(
        "try:\n        completed = subprocess.run(",
        "try:\n        for _retry in range(2):\n            completed = subprocess.run(", 1), receipt))

    require({path: sha(path) for path in paths} == before and namespace_snapshot() == namespace_before,
            "author tests modified subject or created r5 namespace")
    print(json.dumps({
        "schema": "m1131r5_launcher_author_static_controlled_mock_checks_v1",
        "status": "PASS_M1131R5_ZERO_ARG_LAUNCHER_AUTHOR_TESTS__M1132R5_REQUIRED__NO_EDA",
        "checks": checks, "attacks_rejected": len(attacks), "attacks": attacks,
        "static": static, "controlled_resource_gate": resource,
        "controlled_single_child": {"calls": len(child_calls), "returncode": rc,
                                     "exact_argv": True, "clean_env": True},
        "execution": {"launcher": False, "engine": False, "attempt": False,
                      "dc": False, "vcs": False, "r5_namespace_created": False},
    }, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
