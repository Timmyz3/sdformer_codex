#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent M1125r4 launch hammer using static checks and controlled mocks only.

This program never calls the real launcher main, engine main, pgrep, lmutil, DC,
VCS, or another EDA binary.  Canonical attempt/result/work/failure/lock paths
are snapshotted and must be unchanged after all tests.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
LAUNCHER = HW / "dc_handoff/scripts/run_m1122r4_c2_dc_selector_async_observation_authorized_launch_r1.py"
ENGINE = HW / "dc_handoff/scripts/m1122r4_c2_dc_selector_async_observation_engine_source_r1.py"
SOURCE_CONTRACT = HW / "contracts/m1124r4_m1122r4_c2_dc_selector_zero_arg_launcher_source_contract_r1_20260830.json"
LAUNCH_RECEIPT = HW / "contracts/m1122r4_c2_dc_selector_async_observation_authorized_launch_receipt_r1_20260830.json"
AUTHOR = HW / "reviews/m1124r4_m1122r4_c2_dc_selector_zero_arg_launcher_author_receipt_r1_20260830"
ENGINE_CONTRACT = HW / "contracts/m1122r4_c2_dc_selector_async_observation_source_contract_r1_20260830.json"
ENGINE_AUTHOR = HW / "reviews/m1122r4_c2_dc_selector_engine_author_receipt_r1_20260830"
M1121 = HW / "reviews/m1121_m1112r3_c2_dc_invocation_failure_audit_r1_20260830"
M1123R4 = HW / "reviews/m1123r4_m1122r4_c2_dc_selector_engine_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUTPUT = HERE / "mechanical_checks.json"

PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
LICENSE = Path("/opt/synopsys/Synopsys.dat")
LAUNCHER_SHA = "405cf1bd8a6af412ce44a727b47db90c14923054678946378bdfe2646a95ec78"
ENGINE_SHA = "f278052d251af0c2d150872391306c2f3922049ca04c7df2a0d9d3d074b55007"
SOURCE_CONTRACT_ID = (
    "e2e55ebe7bd9699e020d1a35a9b1d6191071c9ae3f58cb1e00d4daab2aafbb89",
    "719c1edc38a24ef9efd4f3625bc31fa058872d144dc440f227ad522dbb1d5924",
    "b5096402619ac2a9a49bda55c956a11a84fb104f2372fdb0dee6ec910a7e58d3",
)
LAUNCH_RECEIPT_ID = (
    "bcdf5c00e32a8b39f40782232e197d3f450a3f0cc650da9a35ecf6d2da5cf138",
    "b7aaa32cfbe821fc7313839d3c498b3febc6fca6a59c14cb1213d1156bcf6754",
    "532a1d07744d139288618bb7995291b9add2dd1c0b7376c7a2c917fa5a6d8113",
)
AUTHOR_OUTER = "862307313404f356f0cfc21258f47b27e767cb6bfc1fdcba4c9cb1cae4dc8f63"
ENGINE_CONTRACT_ID = (
    "cee4ddc66c244bf4e19e2ce193573b55bf4fd973c7c1bcd53d609d77a9b8cea3",
    "0a1ed1ad054b8a778c17c71eb9fdd82d5df943d77989280bd43660736795b617",
    "373e6b86bdfdf94584f289f8c0fc1af1dc9a7ea19be656cba93159b3efb06987",
)
ENGINE_AUTHOR_OUTER = "c36311a8ac2d5b425c2e3b45a7fee665d9f93cd07e06fd4a095746d7c7c99c9b"
M1121_OUTER = "dc0135b61750134c37b6e3eba47350a0d9838c9ed0a07ca5ecab3bb93c3ff828"
M1123R4_OUTER = "c90174a5a981668d3d27a65c9c73d27370e96039bccaefc68f97a775c9640d5a"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
PYTHON_SHA = "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
LICENSE_SHA = "fc6e1face2ac074043db2bef5c789d5ef747ef76333bc17e62d45389f48a3490"

ROOT_ENV = {
    "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PATH": "/usr/bin:/bin",
    "TMPDIR": "/tmp", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
}
EXPECTED_CHILD_ENV_BASE = {
    "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PATH": "/usr/bin:/bin",
    "TMPDIR": "/tmp", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
    "SNPSLMD_LICENSE_FILE": "27030@ic.ismd-nemo",
    "LM_LICENSE_FILE": "/opt/synopsys/Synopsys.dat",
}
EXPECTED_CHILD_ARGV = [str(PYTHON), "-I", str(ENGINE), "--authorized-launch"]
EXPECTED_ROOT_COMMAND = (
    "/usr/bin/env -i LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/usr/bin:/bin TMPDIR=/tmp "
    "PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 " + str(PYTHON) + " " + str(LAUNCHER)
)
EDA_NAMES = (
    "vcs", "vcs1", "vlogan", "dc_shell", "dc_shell-t", "fm_shell",
    "pt_shell", "simv", "common_shell_exec", "common_shell_exe",
)


class Reject(RuntimeError):
    pass


checks = 0
attacks: list[str] = []


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise Reject(message)


def expect_reject(label: str, operation, exception=(Exception,)) -> None:
    try:
        operation()
    except exception:
        attacks.append(label)
        return
    raise Reject("attack accepted: " + label)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_pairs(rows):
    result = {}
    for key, value in rows:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def strict_load(path: Path):
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(), "direct regular JSON")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(Reject("nonfinite " + token)))


def verify_regular(path: Path, expected: str) -> None:
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink() and sha(path) == expected,
            "regular identity " + str(path))


def verify_double(path: Path, identity: tuple[str, str, str]) -> dict:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    for member, digest in zip((path, side, outer), identity):
        verify_regular(member, digest)
    require(side.read_text(encoding="utf-8").split() == [identity[0], path.relative_to(HW).as_posix()],
            "double side content")
    require(outer.read_text(encoding="utf-8").split() == [identity[1], side.relative_to(HW).as_posix()],
            "double outer content")
    return strict_load(path)


def manifest_rows(path: Path) -> dict[str, str]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]) is not None,
                "manifest syntax")
        name = fields[1].lstrip("*")
        rel = Path(name)
        require(name and name not in rows and name == rel.as_posix() and not rel.is_absolute() and
                ".." not in rel.parts, "manifest member safety")
        rows[name] = fields[0]
    return rows


def verify_flat(directory: Path, expected_outer: str) -> dict:
    require(stat.S_ISDIR(directory.lstat().st_mode) and not directory.is_symlink(), "sealed root")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    verify_regular(outer, expected_outer)
    expected = manifest_rows(manifest)
    require(outer.read_text(encoding="utf-8").split() == [sha(manifest), "SHA256SUMS"], "outer content")
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "live sealed symlink")
        if stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), "sealed special member")
    require(actual == set(expected), "sealed exact member set")
    for name, digest in expected.items():
        verify_regular(directory / name, digest)
    return strict_load(directory / "review.json")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function_text(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(item for item in tree.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and
                item.name == name)
    return ast.get_source_segment(source, node) or ast.unparse(node)


def canonical_snapshot() -> dict:
    results = HW / "results"
    return {
        "attempt": (results / ".m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_attempt_consumed").exists(),
        "result": (results / "m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830").exists(),
        "work": sorted(path.name for path in results.glob(".m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_work.*")),
        "failure": sorted(path.name for path in results.glob("m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.*")),
        "lock": Path("/tmp/m1122r4_c2_dc_selector_async_observation_eda.lock").exists(),
    }


def test_static_and_authority(launcher, engine) -> dict:
    contract = verify_double(SOURCE_CONTRACT, SOURCE_CONTRACT_ID)
    receipt = verify_double(LAUNCH_RECEIPT, LAUNCH_RECEIPT_ID)
    verify_double(ENGINE_CONTRACT, ENGINE_CONTRACT_ID)
    author = verify_flat(AUTHOR, AUTHOR_OUTER)
    verify_flat(ENGINE_AUTHOR, ENGINE_AUTHOR_OUTER)
    verify_flat(M1121, M1121_OUTER)
    verify_flat(M1123R4, M1123R4_OUTER)
    verify_regular(LAUNCHER, LAUNCHER_SHA)
    verify_regular(ENGINE, ENGINE_SHA)
    verify_regular(DOCS359, DOCS359_SHA)
    verify_regular(PYTHON, PYTHON_SHA)
    verify_regular(LICENSE, LICENSE_SHA)
    require(contract["launcher"]["unique_future_command"] == EXPECTED_ROOT_COMMAND, "unique root command")
    require(contract["launcher"]["exact_parent_cmdline_for_engine"] == [str(PYTHON), str(LAUNCHER)],
            "exact parent cmdline")
    require(contract["launcher"]["sha256"] == LAUNCHER_SHA and
            contract["launch_receipt"]["outer_seal_file_sha256"] == LAUNCH_RECEIPT_ID[2] and
            contract["acyclicity"]["future_m1125r4_discovers_its_own_self_consistent_outer"] is True and
            contract["acyclicity"]["sha256_fixed_point_required"] is False, "source contract boundary")
    require(receipt["arguments"] == 0 and receipt["maximum_attempts"] == 1 and
            receipt["automatic_retry"] is False and receipt["caller_environment_forwarded"] is False and
            receipt["caller_selected_authority_allowed"] is False and receipt["m1125r4_required"] is True and
            "m1125r4_outer_seal_file_sha256" not in receipt, "launch receipt boundary")
    require(author["status"] ==
            "PASS_M1124R4_M1122R4_ZERO_ARG_LAUNCHER_AUTHOR_RECEIPT__M1125R4_REQUIRED__NO_EDA",
            "author status")

    lsource = LAUNCHER.read_text(encoding="utf-8")
    esource = ENGINE.read_text(encoding="utf-8")
    main = function_text(lsource, "main")
    validation = function_text(lsource, "validate_hardcoded_authorities")
    child_env = function_text(lsource, "clean_child_environment")
    namespace = function_text(lsource, "namespace_resource_gate")
    collision = function_text(lsource, "collision_gate")
    future = function_text(esource, "verify_future_authority")
    estatic = function_text(esource, "static_gate")
    eflow = function_text(esource, "flow")
    require(main.count("subprocess.run(") == 1 and "subprocess.Popen" not in main and
            main.index("validate_hardcoded_authorities") < main.index("namespace_resource_gate") <
            main.index("tempfile.mkdtemp") < main.index("subprocess.run"), "one ordered child site")
    require('[str(PYTHON), "-I", str(ENGINE), "--authorized-launch"]' in main and
            'cwd=str(HW)' in main and "close_fds=True" in main and "check=False" in main,
            "exact child call")
    require("finally:" in main and "shutil.rmtree(private_home)" in main and
            'private_home.parent == Path("/tmp")' in main and 'private_home.name.startswith("m1122r4_c2_home.")' in main,
            "cleanup identity")
    require("len(sys.argv) == 1" in validation and "tuple(sys.version_info[:3]) == (3, 10, 18)" in validation and
            "os.environ == ROOT_ENV" in validation, "runtime root authority")
    require("os.environ" not in child_env and "os.getenv" not in child_env and
            '"SNPSLMD_LICENSE_FILE": SNPSLMD_LICENSE_FILE' in child_env and
            '"LM_LICENSE_FILE": str(LICENSE_FILE)' in child_env, "constant child environment")
    require("glob(WORK_GLOB)" in namespace and "glob(FAILURE_GLOB)" in namespace and
            "prior failure/quarantine forbids retry" in namespace and "MIN_MEM_AVAILABLE_KIB" in namespace and
            "MIN_COMMIT_HEADROOM_KIB" in namespace, "namespace/resource fail closed")
    require('"common_shell_exec", "common_shell_exe"' in lsource and
            '["/usr/bin/pgrep", "-u", uid, "-x", name]' in collision and
            'completed.returncode == 1' in collision, "same UID collision semantics")
    require('sys.argv[1:] != ["--authorized-launch"]' in estatic and
            'verify_flat_self_consistent(M1125R4)' in future and
            '"PASS_M1125R4_M1122R4_LAUNCH_HAMMER__GO_ONE_ATTEMPT"' in future and
            'verify_parent_launcher(receipt)' in future, "engine future authority")
    require(eflow.count("ATTEMPT.mkdir()") == 1 and eflow.count("run_dc_with_selector_capture(") == 1 and
            eflow.index("ATTEMPT.mkdir()") < eflow.index("run_dc_with_selector_capture(") <
            eflow.index("structural_reset_gate(netlist)") < eflow.index("str(VCS)"), "one attempt ordered flow")
    require(LAUNCH_RECEIPT_ID[2] not in lsource and "m1125r4_outer_seal_file_sha256" not in lsource,
            "launcher no future/hash cycle")
    require("m1125r4_outer_seal_file_sha256" not in receipt, "receipt no future/hash cycle")

    # Full existing authority verification is read-only and deliberately omits runtime execution.
    authority = launcher.validate_hardcoded_authorities(enforce_runtime=False)
    require(authority["engine_sha256"] == ENGINE_SHA and
            authority["m1123r4_outer_seal_file_sha256"] == M1123R4_OUTER, "launcher authority verifier")
    return {"source_contract": contract, "launch_receipt": receipt}


def test_runtime_root(launcher) -> None:
    require(Path(sys.executable) == PYTHON and tuple(sys.version_info[:3]) == (3, 10, 18),
            "hammer itself uses pinned Python")
    with mock.patch.dict(launcher.os.environ, ROOT_ENV, clear=True), \
            mock.patch.object(launcher.sys, "argv", [str(LAUNCHER)]):
        launcher.validate_hardcoded_authorities(enforce_runtime=True)
    with mock.patch.dict(launcher.os.environ, ROOT_ENV, clear=True), \
            mock.patch.object(launcher.sys, "argv", [str(LAUNCHER), "--inject"]):
        expect_reject("launcher_extra_argument", lambda: launcher.validate_hardcoded_authorities(True), RuntimeError)
    poisoned = dict(ROOT_ENV); poisoned["HOME"] = "/attacker"
    with mock.patch.dict(launcher.os.environ, poisoned, clear=True), \
            mock.patch.object(launcher.sys, "argv", [str(LAUNCHER)]):
        expect_reject("launcher_caller_environment", lambda: launcher.validate_hardcoded_authorities(True), RuntimeError)
    with mock.patch.dict(launcher.os.environ, ROOT_ENV, clear=True), \
            mock.patch.object(launcher.sys, "argv", [str(LAUNCHER)]), \
            mock.patch.object(launcher.sys, "executable", "/usr/bin/python3"):
        expect_reject("launcher_unpinned_python", lambda: launcher.validate_hardcoded_authorities(True), RuntimeError)
    with mock.patch.dict(launcher.os.environ, ROOT_ENV, clear=True), \
            mock.patch.object(launcher.sys, "argv", [str(LAUNCHER)]), \
            mock.patch.object(launcher.sys, "version_info", (3, 10, 17)):
        expect_reject("launcher_python_version_drift", lambda: launcher.validate_hardcoded_authorities(True), RuntimeError)


def run_mocked_main(launcher, returncode=0, raised=None) -> tuple[int | None, dict, Path]:
    private = Path(tempfile.mkdtemp(prefix="m1122r4_c2_home.", dir="/tmp"))
    record = {"order": []}
    def authority(enforce_runtime):
        require(enforce_runtime is True, "main runtime validation flag"); record["order"].append("authority")
    def namespace():
        record["order"].append("namespace"); return {"status": "mock"}
    def fake_run(argv, **kwargs):
        record["order"].append("child"); record["argv"] = argv; record["kwargs"] = kwargs
        if raised is not None:
            raise raised
        return SimpleNamespace(returncode=returncode)
    outcome = None
    try:
        with mock.patch.object(launcher, "validate_hardcoded_authorities", side_effect=authority), \
                mock.patch.object(launcher, "namespace_resource_gate", side_effect=namespace), \
                mock.patch.object(launcher.tempfile, "mkdtemp", return_value=str(private)), \
                mock.patch.object(launcher.subprocess, "run", side_effect=fake_run), \
                mock.patch.dict(launcher.os.environ, {"HOME": "/poison", "LM_LICENSE_FILE": "attacker"}, clear=True):
            outcome = launcher.main()
    finally:
        record["cleaned"] = not private.exists()
        if private.exists():
            private.rmdir()
    return outcome, record, private


def test_main_and_child_environment(launcher) -> None:
    rc, record, private = run_mocked_main(launcher)
    require(rc == 0 and record["order"] == ["authority", "namespace", "child"], "positive main ordering")
    require(record["argv"] == EXPECTED_CHILD_ARGV, "positive exact child argv")
    require(record["kwargs"]["cwd"] == str(HW) and record["kwargs"]["close_fds"] is True and
            record["kwargs"]["check"] is False, "positive child controls")
    expected_env = dict(EXPECTED_CHILD_ENV_BASE); expected_env["HOME"] = str(private)
    require(record["kwargs"]["env"] == expected_env and record["cleaned"], "caller blind env and cleanup")
    rc, record, _ = run_mocked_main(launcher, returncode=37)
    require(rc == 37 and record["cleaned"], "nonzero child status propagated")
    holder = {}
    exception_home = Path(tempfile.mkdtemp(prefix="m1122r4_c2_home.", dir="/tmp"))
    def raises():
        def child_failure(*_args, **_kwargs):
            raise OSError("mock child exec failure")
        try:
            with mock.patch.object(launcher, "validate_hardcoded_authorities", return_value={}), \
                    mock.patch.object(launcher, "namespace_resource_gate", return_value={}), \
                    mock.patch.object(launcher.tempfile, "mkdtemp", return_value=str(exception_home)), \
                    mock.patch.object(launcher.subprocess, "run", side_effect=child_failure):
                launcher.main()
        except OSError as error:
            holder["message"] = str(error)
            holder["cleaned"] = not exception_home.exists()
            raise
    try:
        expect_reject("child_exception_propagation_and_cleanup", raises, OSError)
    finally:
        if exception_home.exists():
            exception_home.rmdir()
    require(holder == {"message": "mock child exec failure", "cleaned": True},
            "child exception identity and cleanup")
    with mock.patch.dict(launcher.os.environ,
                         {"SNPSLMD_LICENSE_FILE": "evil", "LM_LICENSE_FILE": "evil2", "HOME": "/evil"}, clear=True):
        env = launcher.clean_child_environment(Path("/tmp/m1125r4_controlled_home"))
    expected = dict(EXPECTED_CHILD_ENV_BASE); expected["HOME"] = "/tmp/m1125r4_controlled_home"
    require(env == expected, "standalone constant child env")
    for index, mutated in enumerate(EXPECTED_CHILD_ARGV):
        wrong = list(EXPECTED_CHILD_ARGV); wrong[index] = mutated + ".attack"
        expect_reject("child_argv_mutation_" + str(index), lambda wrong=wrong: require(wrong == EXPECTED_CHILD_ARGV,
                                                                                       "child argv"), Reject)


def namespace_case(launcher, marker: str | None = None, mem=None) -> None:
    with tempfile.TemporaryDirectory(prefix="m1125r4_namespace.", dir="/tmp") as raw:
        root = Path(raw); hw = root / "hw"; results = hw / "results"; results.mkdir(parents=True)
        attempt = results / ".attempt"; result = results / "result"; lock = root / "lock"
        if marker == "attempt": attempt.mkdir()
        elif marker == "attempt_symlink": attempt.symlink_to(results)
        elif marker == "result": result.mkdir()
        elif marker == "lock": lock.touch()
        elif marker == "work": (results / ".work.1").touch()
        elif marker == "failure": (results / "failure.1").touch()
        info = mem or {"MemAvailable": 9 * 1024 * 1024, "CommitLimit": 20 * 1024 * 1024,
                       "Committed_AS": 10 * 1024 * 1024}
        with mock.patch.multiple(launcher, HW=hw, ATTEMPT=attempt, RESULT=result, LOCK=lock,
                                 WORK_GLOB=".work.*", FAILURE_GLOB="failure.*"), \
                mock.patch.object(launcher, "collision_gate", return_value=[]), \
                mock.patch.object(launcher, "read_meminfo", return_value=info):
            launcher.namespace_resource_gate()


def test_namespace_resource_collision(launcher) -> None:
    namespace_case(launcher)
    for marker in ("attempt", "attempt_symlink", "result", "lock", "work", "failure"):
        expect_reject("namespace_" + marker, lambda marker=marker: namespace_case(launcher, marker), RuntimeError)
    expect_reject("resource_memavailable", lambda: namespace_case(
        launcher, mem={"MemAvailable": 8 * 1024 * 1024 - 1, "CommitLimit": 20 * 1024 * 1024,
                       "Committed_AS": 10 * 1024 * 1024}), RuntimeError)
    expect_reject("resource_commit_headroom", lambda: namespace_case(
        launcher, mem={"MemAvailable": 9 * 1024 * 1024, "CommitLimit": 18 * 1024 * 1024 - 1,
                       "Committed_AS": 10 * 1024 * 1024}), RuntimeError)

    calls = []
    def fake_positive(argv, **kwargs):
        calls.append((argv, kwargs)); return SimpleNamespace(returncode=1)
    with mock.patch.object(launcher.subprocess, "run", side_effect=fake_positive):
        require(launcher.collision_gate() == [], "collision positive")
    uid = str(os.getuid())
    require(len(calls) == len(EDA_NAMES) and [item[0][-1] for item in calls] == list(EDA_NAMES) and
            all(item[0][:-1] == ["/usr/bin/pgrep", "-u", uid, "-x"] for item in calls),
            "same UID exact pgrep calls")
    counter = {"value": 0}
    def fake_collision(argv, **kwargs):
        counter["value"] += 1
        return SimpleNamespace(returncode=0 if counter["value"] == 3 else 1)
    with mock.patch.object(launcher.subprocess, "run", side_effect=fake_collision):
        expect_reject("same_uid_collision", launcher.collision_gate, RuntimeError)
    with mock.patch.object(launcher.subprocess, "run", return_value=SimpleNamespace(returncode=2)):
        expect_reject("pgrep_diagnostic_failure", launcher.collision_gate, RuntimeError)


def test_engine_license_and_future_seal(engine) -> None:
    calls = []
    def good(argv, **kwargs): calls.append((argv, kwargs)); return SimpleNamespace(returncode=0)
    with mock.patch.dict(engine.os.environ, EXPECTED_CHILD_ENV_BASE, clear=True), \
            mock.patch.object(engine.subprocess, "run", side_effect=good):
        engine.license_gate()
    require(len(calls) == 1 and calls[0][0] == [str(engine.LMUTIL), "lmstat", "-a", "-c", "27030@ic.ismd-nemo"] and
            calls[0][1]["timeout"] == 60 and calls[0][1]["check"] is False, "constant license route semantics")
    fallback = dict(EXPECTED_CHILD_ENV_BASE); fallback["SNPSLMD_LICENSE_FILE"] = ""
    calls.clear()
    with mock.patch.dict(engine.os.environ, fallback, clear=True), \
            mock.patch.object(engine.subprocess, "run", side_effect=good):
        engine.license_gate()
    require(calls[0][0][-1] == str(LICENSE), "license file fallback")
    with mock.patch.dict(engine.os.environ, {}, clear=True), \
            mock.patch.object(engine.subprocess, "run", side_effect=lambda *_a, **_k: (_ for _ in ()).throw(
                Reject("lmstat must not run"))):
        expect_reject("license_route_absent", engine.license_gate, engine.GateFailure)
    with mock.patch.dict(engine.os.environ, EXPECTED_CHILD_ENV_BASE, clear=True), \
            mock.patch.object(engine.subprocess, "run", return_value=SimpleNamespace(returncode=1)):
        expect_reject("license_lmstat_failure", engine.license_gate, engine.GateFailure)

    with tempfile.TemporaryDirectory(prefix="m1125r4_future_seal.", dir="/tmp") as raw:
        root = Path(raw)
        review = root / "review.json"
        review.write_text(json.dumps({"status": "PASS_M1125R4_M1122R4_LAUNCH_HAMMER__GO_ONE_ATTEMPT",
                                      "identity": {"launcher_sha256": LAUNCHER_SHA}}, sort_keys=True) + "\n",
                          encoding="utf-8")
        manifest = root / "SHA256SUMS"
        manifest.write_text(f"{sha(review)}  review.json\n", encoding="utf-8")
        outer = root / "SHA256SUMS.seal.sha256"
        outer.write_text(f"{sha(manifest)}  SHA256SUMS\n", encoding="utf-8")
        require(engine.verify_flat_self_consistent(root) == sha(outer), "future self-consistent discovery")
        extra = root / "extra"; extra.write_text("attack\n", encoding="utf-8")
        expect_reject("future_seal_live_extra", lambda: engine.verify_flat_self_consistent(root), engine.GateFailure)
        extra.unlink()
        link = root / "link"; link.symlink_to(review)
        expect_reject("future_seal_symlink", lambda: engine.verify_flat_self_consistent(root), engine.GateFailure)
        link.unlink()
        original = outer.read_text(encoding="utf-8"); outer.write_text("0" * 64 + "  SHA256SUMS\n", encoding="utf-8")
        expect_reject("future_seal_outer_mutation", lambda: engine.verify_flat_self_consistent(root), engine.GateFailure)
        outer.write_text(original, encoding="utf-8")


def main() -> int:
    before = canonical_snapshot()
    require(before == {"attempt": False, "result": False, "work": [], "failure": [], "lock": False},
            "canonical M1122r4 namespaces fresh before hammer")
    launcher = load_module(LAUNCHER, "m1125r4_safe_launcher_module")
    engine = load_module(ENGINE, "m1125r4_safe_engine_module")
    test_static_and_authority(launcher, engine)
    test_runtime_root(launcher)
    test_main_and_child_environment(launcher)
    test_namespace_resource_collision(launcher)
    test_engine_license_and_future_seal(engine)
    after = canonical_snapshot()
    require(after == before, "canonical namespaces unchanged by hammer")
    result = {
        "schema": "m1125r4_m1122r4_c2_dc_selector_launch_hammer_mechanical_checks_r1_v1",
        "status": "PASS_M1125R4_STATIC_AND_CONTROLLED_MOCK_ONLY__NO_LAUNCH_NO_EDA",
        "checks_passed": checks,
        "attacks_rejected": len(attacks),
        "attack_labels": attacks,
        "identity": {
            "launcher_sha256": sha(LAUNCHER),
            "launch_receipt_outer_seal_file_sha256": sha(Path(str(LAUNCH_RECEIPT) + ".sha256.seal.sha256")),
            "source_contract_outer_seal_file_sha256": sha(Path(str(SOURCE_CONTRACT) + ".sha256.seal.sha256")),
            "author_receipt_outer_seal_file_sha256": sha(AUTHOR / "SHA256SUMS.seal.sha256"),
            "engine_sha256": sha(ENGINE),
            "engine_contract_outer_seal_file_sha256": sha(Path(str(ENGINE_CONTRACT) + ".sha256.seal.sha256")),
            "engine_author_receipt_outer_seal_file_sha256": sha(ENGINE_AUTHOR / "SHA256SUMS.seal.sha256"),
            "m1121_outer_seal_file_sha256": sha(M1121 / "SHA256SUMS.seal.sha256"),
            "m1123r4_outer_seal_file_sha256": sha(M1123R4 / "SHA256SUMS.seal.sha256"),
            "docs359_sha256": sha(DOCS359),
        },
        "mock_boundaries": {
            "real_launcher_executed": False, "real_engine_executed": False,
            "real_pgrep_or_lmstat_executed": False, "eda_executed": False,
            "attempt_result_work_failure_or_lock_created": False,
        },
        "canonical_namespace_before": before,
        "canonical_namespace_after": after,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks_passed": checks,
                      "attacks_rejected": len(attacks)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
