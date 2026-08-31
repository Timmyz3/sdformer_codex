#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1136r6 independent final launch hammer; full controlled path, no real launch/EDA/VCS."""
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
HW = HERE.parents[1]
LAUNCHER = HW / "dc_handoff/scripts/run_m1133r6_c2_authority_schema_repair_authorized_launch_r1.py"
RECEIPT = HW / "contracts/m1133r6_c2_authority_schema_repair_authorized_launch_receipt_r1_20260830.json"
ENGINE = HW / "dc_handoff/scripts/m1133r6_c2_authority_schema_repair_engine_source_r1.py"
ENGINE_CONTRACT = HW / "contracts/m1133r6_c2_authority_schema_repair_engine_source_contract_r1_20260830.json"
ENGINE_AUTHOR = HW / "reviews/m1133r6_c2_authority_schema_repair_engine_author_receipt_r1_20260830"
M1121 = HW / "reviews/m1121_m1112r3_c2_dc_invocation_failure_audit_r1_20260830"
M1132R5_STOP = HW / "reviews/m1132r5_m1129r5_c2_dc_selector_launch_hammer_r1_20260830"
M1134R6 = HW / "reviews/m1134r6_m1133r6_c2_authority_schema_engine_hammer_r1_20260830"
M1135R6 = HW / "reviews/m1135r6_m1133r6_c2_zero_arg_launcher_author_receipt_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUTPUT = HERE / "mechanical_checks.json"

EXPECTED = {
    "launcher": "43ab981e0272b524479f3ba60a321bd968f1065ae9bd009d9b9c6bff898c9999",
    "receipt": "797834f2125665652aa5cdaa648823413d56ccdafcf419383d64472ac4828e36",
    "receipt_side": "23605bf55ae1a6da374b5fbeee3b22e2e1740457e95c801c6580d2b31edb1e51",
    "receipt_outer": "19c71a33e4be2618c7c0500a6d58af12fea88bb280d6b670beef186f50e8d9a6",
    "engine": "1f8a190d7d1c8b7804e7302c8b6a38c30a49df466b6394a82e8f0cf4cec2ee40",
    "contract": "4dc16ffccb3c4a145f69f565500d67407ca821304ee838f93659918055a3ac8a",
    "contract_outer": "82b6d6a6568fc8fc95f1a1b7b6bf05690e06e064a143de41eadfa0e76ac9b849",
    "engine_author_outer": "5b2e0a659992c006d5caee72f5bcd72fd28dfdc07266d7edd2c814f1bc4a3b68",
    "m1121_outer": "dc0135b61750134c37b6e3eba47350a0d9838c9ed0a07ca5ecab3bb93c3ff828",
    "m1132r5_stop_outer": "bc073b90787189710986381b74c18b9a3afbe4ccd2f7969e85b596d3df1adf48",
    "m1134r6_outer": "7c61ff53aaee7711fda0f79fd3a3bc9d99decfc9b7fbda377f22caf29fa72226",
    "m1135r6_review": "340cd358109f02829e04aa9c0fdb75fc6ec947b9c1c7f2f630cbb00babc17367",
    "m1135r6_manifest": "782a396fbcb4deee133d8dd877f8a619d46959d5a5e85eee5069d256f6126a77",
    "m1135r6_outer": "b72f21d7fd6cbf97e2c7888b9dbceaf52c5b7d28d38d86092fadea5ec81a615e",
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


def rejected(label: str, action: Callable[[], Any]) -> None:
    try:
        action()
    except Exception as error:
        attacks[label] = type(error).__name__ + ": " + str(error)
        return
    raise CheckFailure("attack accepted: " + label)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_regular(path: Path, expected: str) -> None:
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink() and
            sha(path) == expected, "identity drift: " + str(path))


def strict_json(path: Path):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          CheckFailure("nonfinite JSON: " + token)))


def verify_double() -> None:
    side = Path(str(RECEIPT) + ".sha256")
    outer = Path(str(RECEIPT) + ".sha256.seal.sha256")
    for path, expected in zip((RECEIPT, side, outer),
                              (EXPECTED["receipt"], EXPECTED["receipt_side"],
                               EXPECTED["receipt_outer"])):
        verify_regular(path, expected)
    require(side.read_text(encoding="utf-8").split() ==
            [EXPECTED["receipt"], RECEIPT.relative_to(HW).as_posix()] and
            outer.read_text(encoding="utf-8").split() ==
            [EXPECTED["receipt_side"], side.relative_to(HW).as_posix()],
            "launch receipt double seal content")


def verify_flat(directory: Path, expected_outer: str) -> dict[str, Any]:
    require(stat.S_ISDIR(directory.lstat().st_mode) and not directory.is_symlink(),
            "sealed authority directory")
    manifest = directory / "SHA256SUMS"; outer = directory / "SHA256SUMS.seal.sha256"
    verify_regular(outer, expected_outer)
    require(outer.read_text(encoding="utf-8").split() == [sha(manifest), "SHA256SUMS"],
            "sealed outer content")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*"); rel = Path(name)
        require(name not in expected and name == rel.as_posix() and
                not rel.is_absolute() and ".." not in rel.parts,
                "sealed safe manifest")
        expected[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "sealed symlink")
        if stat.S_ISREG(mode): actual.add(name)
        else: require(stat.S_ISDIR(mode), "sealed special member")
    require(actual == set(expected), "sealed exact member set")
    for name, digest in expected.items(): verify_regular(directory / name, digest)
    return strict_json(directory / "review.json")


def function_node(source: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    value = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == name), None)
    require(value is not None, "missing function: " + name)
    return value


def expected_receipt() -> dict[str, Any]:
    return {
        "arguments": 0, "attempt_now": False, "automatic_retry": False,
        "caller_environment_forwarded": False,
        "caller_selected_authority_allowed": False, "dc_now": False,
        "engine_author_receipt_outer_seal_file_sha256": EXPECTED["engine_author_outer"],
        "engine_contract_outer_seal_file_sha256": EXPECTED["contract_outer"],
        "engine_contract_sha256": EXPECTED["contract"],
        "engine_sha256": EXPECTED["engine"], "launch_now": False,
        "launcher_sha256": EXPECTED["launcher"],
        "m1121_outer_seal_file_sha256": EXPECTED["m1121_outer"],
        "m1132r5_stop_outer_seal_file_sha256": EXPECTED["m1132r5_stop_outer"],
        "m1134r6_outer_seal_file_sha256": EXPECTED["m1134r6_outer"],
        "m1136r6_required": True, "mapped_vcs_now": False,
        "maximum_attempts": 1, "paper_citable": False,
        "schema": "m1133r6_c2_authority_schema_repair_authorized_launch_receipt_r1_v1",
        "status": "M1133R6_LAUNCH_SOURCE_FROZEN__M1136R6_REQUIRED__NO_EDA",
    }


def analyze(source: str, receipt: dict[str, Any]) -> dict[str, Any]:
    main = function_node(source, "main")
    validate = function_node(source, "validate_hardcoded_authorities")
    namespace = function_node(source, "namespace_resource_gate")
    collision = function_node(source, "collision_gate")
    clean = function_node(source, "clean_child_environment")
    main_text = ast.unparse(main); validate_text = ast.unparse(validate)
    namespace_text = ast.unparse(namespace); collision_text = ast.unparse(collision)
    clean_text = ast.unparse(clean)
    require("len(sys.argv) == 1" in validate_text and "os.environ == ROOT_ENV" in validate_text,
            "zero argument exact env-i gate")
    require(all(token in namespace_text for token in (
                "R5_ATTEMPT", "R5_RESULT", "R5_LOCK", "R5_WORK_GLOB",
                "R5_FAILURE_GLOB", "ATTEMPT", "RESULT", "LOCK", "WORK_GLOB",
                "FAILURE_GLOB")), "r5 STOP plus fresh r6 namespace gate")
    require("EDA_PROCESS_NAMES" in collision_text and "/usr/bin/pgrep" in collision_text and
            "MIN_MEM_AVAILABLE_KIB = 8 * 1024 * 1024" in source and
            "MIN_COMMIT_HEADROOM_KIB = 8 * 1024 * 1024" in source,
            "collision and memory gates")
    require(main_text.count("subprocess.run") == 1 and
            "[str(PYTHON), '-I', str(ENGINE), '--authorized-launch']" in main_text and
            "env=clean_child_environment(private_home)" in main_text and
            "return completed.returncode" in main_text and
            not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(main)),
            "single child and no retry")
    require("os.environ" not in clean_text and "SNPSLMD_LICENSE_FILE" in clean_text and
            "LM_LICENSE_FILE" in clean_text, "clean child environment is constant")
    require(all(EXPECTED[key] in source for key in (
                "engine", "contract_outer", "engine_author_outer", "m1121_outer",
                "m1132r5_stop_outer", "m1134r6_outer")), "exact authority binding")
    require(receipt == expected_receipt(), "launch receipt exact schema/value drift")
    receipt_text = json.dumps(receipt, sort_keys=True)
    require("m1136r6_outer_seal_file_sha256" not in receipt and
            "m1136r6_outer_seal_file_sha256" not in source and
            EXPECTED["receipt_outer"] not in source and EXPECTED["receipt_outer"] not in receipt_text,
            "future M1136r6/receipt hash cycle")
    return {"zero_arguments": True, "exact_env_i_root": True,
            "fresh_r6_namespace": True, "stopped_r5_namespace": True,
            "collision_memory_gates": True, "single_child": True,
            "automatic_retry": False, "future_m1136r6_hash_cycle": False}


def load_launcher():
    spec = importlib.util.spec_from_file_location("m1136r6_launcher_subject", LAUNCHER)
    require(spec is not None and spec.loader is not None, "launcher module spec")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def namespace_snapshot() -> dict[str, Any]:
    results = HW / "results"
    return {
        "r6_attempt": (results / ".m1133r6_c2_authority_schema_repair_dc_mapped_vcs_attempt_consumed").exists(),
        "r6_result": (results / "m1133r6_c2_authority_schema_repair_dc_mapped_vcs_r1_20260830").exists(),
        "r6_work": sorted(path.name for path in results.glob(
            ".m1133r6_c2_authority_schema_repair_dc_mapped_vcs_work.*")),
        "r6_failure": sorted(path.name for path in results.glob(
            "m1133r6_c2_authority_schema_repair_dc_mapped_vcs_r1_20260830.failed_or_incomplete.*")),
        "r6_lock": Path("/tmp/m1133r6_c2_authority_schema_repair_eda.lock").exists(),
        "r5_attempt": (results / ".m1129r5_c2_dc_selector_async_observation_dc_mapped_vcs_attempt_consumed").exists(),
        "r5_result": (results / "m1129r5_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830").exists(),
        "r5_work": sorted(path.name for path in results.glob(
            ".m1129r5_c2_dc_selector_async_observation_dc_mapped_vcs_work.*")),
        "r5_failure": sorted(path.name for path in results.glob(
            "m1129r5_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.*")),
        "r5_lock": Path("/tmp/m1129r5_c2_dc_selector_async_observation_eda.lock").exists(),
    }


def controlled_namespace_case(launcher, marker: str | None = None) -> None:
    with tempfile.TemporaryDirectory(prefix="m1136r6_namespace.", dir="/tmp") as raw:
        root = Path(raw); hw = root / "hw"; results = hw / "results"; results.mkdir(parents=True)
        values = {
            "ATTEMPT": results / ".r6_attempt", "RESULT": results / "r6_result",
            "LOCK": root / "r6_lock", "WORK_GLOB": ".r6_work.*",
            "FAILURE_GLOB": "r6_failure.*", "R5_ATTEMPT": results / ".r5_attempt",
            "R5_RESULT": results / "r5_result", "R5_LOCK": root / "r5_lock",
            "R5_WORK_GLOB": ".r5_work.*", "R5_FAILURE_GLOB": "r5_failure.*",
        }
        targets = {
            "r6_attempt": values["ATTEMPT"], "r6_result": values["RESULT"],
            "r6_lock": values["LOCK"], "r6_work": results / ".r6_work.1",
            "r6_failure": results / "r6_failure.1", "r5_attempt": values["R5_ATTEMPT"],
            "r5_result": values["R5_RESULT"], "r5_lock": values["R5_LOCK"],
            "r5_work": results / ".r5_work.1", "r5_failure": results / "r5_failure.1",
        }
        if marker is not None:
            targets[marker].touch()
        with patch.multiple(launcher, HW=hw, **values), \
             patch.object(launcher, "collision_gate", return_value=[]), \
             patch.object(launcher, "read_meminfo", return_value={
                 "MemAvailable": 9 * 1024 * 1024,
                 "CommitLimit": 20 * 1024 * 1024,
                 "Committed_AS": 10 * 1024 * 1024}):
            launcher.namespace_resource_gate()


def main() -> None:
    paths = (LAUNCHER, RECEIPT, ENGINE, ENGINE_CONTRACT, DOCS359)
    before = {path: sha(path) for path in paths}; namespace_before = namespace_snapshot()
    require(not any(namespace_before.values()), "r5/r6 namespace not absent")
    verify_regular(LAUNCHER, EXPECTED["launcher"]); verify_double()
    verify_regular(ENGINE, EXPECTED["engine"]); verify_regular(ENGINE_CONTRACT, EXPECTED["contract"])
    verify_regular(DOCS359, EXPECTED["docs359"])
    engine_author = verify_flat(ENGINE_AUTHOR, EXPECTED["engine_author_outer"])
    m1121 = verify_flat(M1121, EXPECTED["m1121_outer"])
    stop = verify_flat(M1132R5_STOP, EXPECTED["m1132r5_stop_outer"])
    hammer = verify_flat(M1134R6, EXPECTED["m1134r6_outer"])
    launcher_author = verify_flat(M1135R6, EXPECTED["m1135r6_outer"])
    verify_regular(M1135R6 / "review.json", EXPECTED["m1135r6_review"])
    verify_regular(M1135R6 / "SHA256SUMS", EXPECTED["m1135r6_manifest"])
    require(engine_author["status"].startswith("PASS_M1133R6_AUTHORITY_SCHEMA_REPAIR") and
            m1121["status"].startswith("PASS_M1121_FAILURE_AUDIT") and
            stop["status"].startswith("FAIL_M1132R5") and
            stop["authorization"]["r5_command_withdrawn"] is True and
            hammer["status"].startswith("PASS_M1134R6_M1133R6_ENGINE_HAMMER") and
            launcher_author["status"] ==
                "PASS_M1135R6_M1133R6_ZERO_ARG_LAUNCHER_AUTHOR_RECEIPT__M1136R6_REQUIRED__NO_EDA",
            "authority status chain")
    source = LAUNCHER.read_text(encoding="utf-8"); receipt = strict_json(RECEIPT)
    static = analyze(source, receipt)
    launcher = load_launcher()
    authority = launcher.validate_hardcoded_authorities(enforce_runtime=False)
    require(authority["m1121_outer_seal_file_sha256"] == EXPECTED["m1121_outer"] and
            authority["m1132r5_stop_outer_seal_file_sha256"] == EXPECTED["m1132r5_stop_outer"] and
            authority["m1134r6_outer_seal_file_sha256"] == EXPECTED["m1134r6_outer"] and
            authority["future_m1136r6_discovery_acyclic"] is True,
            "launcher authority return")
    with patch.object(launcher.sys, "argv", [str(LAUNCHER)]), \
         patch.dict(launcher.os.environ, launcher.ROOT_ENV, clear=True):
        exact_env_authority = launcher.validate_hardcoded_authorities(enforce_runtime=True)
    require(exact_env_authority == authority, "exact env-i runtime positive gate")
    with patch.object(launcher.sys, "argv", [str(LAUNCHER), "extra"]), \
         patch.dict(launcher.os.environ, launcher.ROOT_ENV, clear=True):
        rejected("runtime_argument", lambda:
                 launcher.validate_hardcoded_authorities(enforce_runtime=True))
    bad_env = dict(launcher.ROOT_ENV); bad_env["UNAUTHORIZED"] = "1"
    with patch.object(launcher.sys, "argv", [str(LAUNCHER)]), \
         patch.dict(launcher.os.environ, bad_env, clear=True):
        rejected("runtime_environment", lambda:
                 launcher.validate_hardcoded_authorities(enforce_runtime=True))

    clean = launcher.clean_child_environment(Path("/tmp/m1135r6_mock_home"))
    require(clean == {"HOME": "/tmp/m1135r6_mock_home", "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8", "PATH": "/usr/bin:/bin", "TMPDIR": "/tmp",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "SNPSLMD_LICENSE_FILE": "27030@ic.ismd-nemo",
        "LM_LICENSE_FILE": "/opt/synopsys/Synopsys.dat"}, "clean child exact values")
    with patch.object(launcher, "collision_gate", return_value=[]), \
         patch.object(launcher, "read_meminfo", return_value={
             "MemAvailable": 9 * 1024 * 1024, "CommitLimit": 20 * 1024 * 1024,
             "Committed_AS": 10 * 1024 * 1024}):
        resource = launcher.namespace_resource_gate()
    require(resource["eda_collisions"] == [] and
            resource["commit_headroom_kib"] == 10 * 1024 * 1024,
            "controlled collision/memory resource gate")
    controlled_namespace_case(launcher)
    for marker in ("r5_attempt", "r5_result", "r5_lock", "r5_work", "r5_failure",
                   "r6_attempt", "r6_result", "r6_lock", "r6_work", "r6_failure"):
        rejected("namespace_" + marker,
                 lambda marker=marker: controlled_namespace_case(launcher, marker))

    child_calls = []; pgrep_calls = []
    def fake_run(argv, **kwargs):
        if argv and argv[0] == "/usr/bin/pgrep":
            pgrep_calls.append({"argv": argv, "kwargs": kwargs})
            return SimpleNamespace(returncode=1)
        child_calls.append({"argv": argv, **kwargs})
        return SimpleNamespace(returncode=7)
    controlled_mem = {"MemAvailable": 9 * 1024 * 1024,
                      "CommitLimit": 20 * 1024 * 1024,
                      "Committed_AS": 10 * 1024 * 1024}
    with patch.object(launcher.sys, "argv", [str(LAUNCHER)]), \
         patch.dict(launcher.os.environ, launcher.ROOT_ENV, clear=True), \
         patch.object(launcher, "read_meminfo", return_value=controlled_mem), \
         patch.object(launcher.subprocess, "run", side_effect=fake_run):
        returncode = launcher.main()
    require(returncode == 7 and len(child_calls) == 1 and child_calls[0]["argv"] == [
                str(launcher.PYTHON), "-I", str(launcher.ENGINE), "--authorized-launch"] and
            child_calls[0]["cwd"] == str(HW) and child_calls[0]["close_fds"] is True and
            child_calls[0]["check"] is False and set(child_calls[0]["env"]) == {
                "HOME", "LANG", "LC_ALL", "PATH", "TMPDIR", "PYTHONNOUSERSITE",
                "PYTHONDONTWRITEBYTECODE", "SNPSLMD_LICENSE_FILE", "LM_LICENSE_FILE"},
            "controlled exactly one pinned child")
    require(len(pgrep_calls) == len(launcher.EDA_PROCESS_NAMES) and
            [call["argv"][-1] for call in pgrep_calls] == list(launcher.EDA_PROCESS_NAMES) and
            all(call["argv"][0:4] == ["/usr/bin/pgrep", "-u", str(os.getuid()), "-x"]
                for call in pgrep_calls),
            "full same-UID collision path before fake child")

    for label, key, value in (
        ("receipt_arguments", "arguments", 1),
        ("receipt_caller_env", "caller_environment_forwarded", True),
        ("receipt_retry", "automatic_retry", True),
        ("receipt_max_attempts", "maximum_attempts", 2),
        ("receipt_wrong_m1121", "m1121_outer_seal_file_sha256", "0" * 64),
        ("receipt_wrong_stop", "m1132r5_stop_outer_seal_file_sha256", "0" * 64),
        ("receipt_wrong_hammer", "m1134r6_outer_seal_file_sha256", "0" * 64),
    ):
        bad = copy.deepcopy(receipt); bad[key] = value
        rejected(label, lambda bad=bad: analyze(source, bad))
    bad = copy.deepcopy(receipt); bad["m1136r6_outer_seal_file_sha256"] = "0" * 64
    rejected("receipt_future_hash_cycle", lambda: analyze(source, bad))
    rejected("launcher_engine_drift", lambda: analyze(
        source.replace(EXPECTED["engine"], "0" * 64), receipt))
    rejected("launcher_no_zero_arg", lambda: analyze(
        source.replace("len(sys.argv) == 1", "len(sys.argv) >= 1"), receipt))
    rejected("launcher_env_forward", lambda: analyze(source.replace(
        "env=clean_child_environment(private_home)", "env=os.environ.copy()"), receipt))
    rejected("launcher_retry_loop", lambda: analyze(source.replace(
        "try:\n        completed = subprocess.run(",
        "try:\n        for _retry in range(2):\n            completed = subprocess.run(", 1), receipt))

    require({path: sha(path) for path in paths} == before and
            namespace_snapshot() == namespace_before,
            "controlled tests modified subjects or namespaces")
    result = {
        "schema": "m1135r6_launcher_author_static_controlled_mock_checks_v1",
        "status": "PASS_M1136R6_FINAL_LAUNCH_HAMMER_PRESEAL__NO_REAL_LAUNCH_NO_EDA",
        "checks": checks, "attacks_rejected": len(attacks), "attacks": attacks,
        "static": static, "authority": authority,
        "controlled_resource_gate": resource,
        "controlled_single_child": {"calls": len(child_calls),
            "returncode": returncode, "exact_argv": True, "clean_env": True,
            "authority_path_real": True, "resource_path_real": True,
            "same_uid_collision_checks": len(pgrep_calls)},
        "execution": {"launcher": False, "engine": False, "attempt": False,
            "dc": False, "vcs": False, "r5_namespace_created": False,
            "r6_namespace_created": False},
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
