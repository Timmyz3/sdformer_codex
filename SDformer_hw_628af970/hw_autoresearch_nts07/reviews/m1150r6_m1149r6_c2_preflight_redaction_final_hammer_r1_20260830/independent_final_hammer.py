#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent M1150R6 hammer for M1149R6; mocks every subprocess.

This program must never invoke real lmstat, VCS, or DC.  All future-execution
checks run inside temporary namespaces with a process factory that rejects any
command outside the frozen lmstat/compile/case0 triplet.
"""
from __future__ import annotations

import ast
from contextlib import ExitStack, contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterator
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "dc_handoff/scripts/run_m1149r6_c2_preflight_redaction_repair_source_r1.py"
CONTRACT = HW / "contracts/m1149r6_c2_preflight_redaction_repair_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1149r6_c2_preflight_redaction_repair_author_receipt_r1_20260830"
M1146 = HW / "reviews/m1146r6_c2_additive_license_route_successor_author_receipt_r1_20260830"
M1147 = HW / "reviews/m1147r6_m1146r6_c2_license_route_final_source_hammer_r1_20260830"
M1148 = HW / "reviews/m1148r6_m1146r6_c2_preflight_redaction_failure_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    "source": "affeba86cc465462ef00a50afe283d9e0677c06362c9830a1bf9ea5f502363ce",
    "contract": "1c4daec4c4ab9dd4ab579edc247068c514d27c18136c91852828cb6077c13802",
    "contract_side": "de2b22ed44d28744f7ecba260fda1c7c8c603fddc6891341f6432334170bba9d",
    "contract_outer": "bbdbc9fcc974a8763bb54d53c7c8b65608e3e535fd6076db227ab2aa52d9795a",
    "author_review": "ee7939e79c0b95fba1e3d955f245055324b46afd09c3f0ca2c7df65c46789bb9",
    "author_manifest": "4c9c2622204253d96da58cb47bf7f1e8cc7aacf5190f69e042127b52b02668c3",
    "author_outer": "b2cefd9935c5b27c6fe28ade8b6ca6e599d17da359cfd0a367315c38ddd22be9",
    "m1146_outer": "513813aa1915e72af18c1b059cfae77947c9ece37fc8699582cc202c489b98d1",
    "m1147_outer": "64007fe4ec37a26c54c197b80ae9f9565e8272c06fecfe3510c24aeb7c74d7e9",
    "m1148_outer": "b60fb9ecd875d87dd0b1f05a8cd448c85ce551a4f589a988f6a0fa8785defb32",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
checks = 0
attacks: dict[str, bool] = {}


class HammerFailure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise HammerFailure(message)


def rejected(label: str, action: Callable[[], Any], contains: str | None = None,
             forbidden: str | None = None) -> None:
    try:
        action()
    except BaseException as error:
        message = str(error)
        if contains is not None:
            require(contains in message, label + " wrong rejection")
        if forbidden is not None:
            require(forbidden not in message, label + " leaked route")
        attacks[label] = True
        return
    raise HammerFailure("attack accepted: " + label)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and sha(path) == expected,
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
                          HammerFailure("nonfinite JSON: " + token)))


def verify_tree(directory: Path, identity: tuple[str, str, str]) -> dict[str, Any]:
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    regular(review, identity[0]); regular(manifest, identity[1]); regular(outer, identity[2])
    require(outer.read_text(encoding="utf-8").split() == [identity[1], "SHA256SUMS"],
            "outer seal content drift")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*"); rel = Path(name)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and
                name not in expected and name == rel.as_posix() and
                not rel.is_absolute() and ".." not in rel.parts,
                "unsafe sealed member")
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
    require(actual == set(expected), "sealed exact member census")
    for name, digest in expected.items():
        regular(directory / name, digest)
    return strict_json(review)


def verify_double(path: Path, identity: tuple[str, str, str]) -> dict[str, Any]:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    regular(path, identity[0]); regular(side, identity[1]); regular(outer, identity[2])
    require(side.read_text(encoding="utf-8").split() == [identity[0], path.name] and
            outer.read_text(encoding="utf-8").split() == [identity[1], side.name],
            "double seal content drift")
    return strict_json(path)


def verify_inputs() -> dict[str, Any]:
    regular(SOURCE, EXPECTED["source"]); regular(DOCS359, EXPECTED["docs359"])
    contract = verify_double(CONTRACT, (EXPECTED["contract"], EXPECTED["contract_side"],
                                        EXPECTED["contract_outer"]))
    author = verify_tree(AUTHOR, (EXPECTED["author_review"], EXPECTED["author_manifest"],
                                  EXPECTED["author_outer"]))
    m1146 = verify_tree(M1146, ("b011596046c724665b71352045c23e82044eaabd5a6ce849be5892b362781fe4",
                                "13fe84f0aef4dfc000278a9e1629368b6256742c6c1ad7f2e769174ac1a6360c",
                                EXPECTED["m1146_outer"]))
    m1147 = verify_tree(M1147, ("d4434283285d6f536b30f3183e86b05e9bbbedbcb9689362df6114d29f9844c9",
                                "b03909b2d54d971a601d52a42f0dd2f1203cde20cb3ee8c31daf27bcf7e877c1",
                                EXPECTED["m1147_outer"]))
    m1148 = verify_tree(M1148, ("5d9f9367a676be794ff39a2d1b3384d83ff60e2cdb77f846b6e99a0e9d770be5",
                                "512159f0be381dcb948866cb71413cb32f1949a50e03db9fd1186d8ebbf9130d",
                                EXPECTED["m1148_outer"]))
    require(contract["source"]["sha256"] == EXPECTED["source"] and
            contract["license_route"]["precedence"] ==
            ["SNPSLMD_LICENSE_FILE", "LM_LICENSE_FILE"] and
            contract["license_route"]["home_key_absent"] is True and
            contract["lmstat_helper"]["decision"] == "process returncode == 0 only" and
            contract["namespace"]["policy"] == "reuse still-fresh M1146 namespace" and
            contract["namespace"]["maximum_attempts"] == 1 and
            contract["preserved_future_execution"]["vcs_compile_attempts"] == 1 and
            contract["preserved_future_execution"]["case0_attempts"] == 1 and
            contract["preserved_future_execution"]["window_cycles"] == 128 and
            contract["preserved_future_execution"]["dc_attempts"] == 0 and
            contract["authorization"]["automatic_retry"] is False,
            "contract semantic drift")
    require(author["status"] ==
            "PASS_M1149R6_RC_ONLY_REDACTION_SOURCE_MOCK__DIFFERENT_AUTHOR_FINAL_HAMMER_ONLY" and
            author["authorization"]["different_author_final_hammer_only"] is True and
            all(author["authorization"][key] is False
                for key in ("real_lmstat", "launch", "vcs", "dc", "automatic_retry")),
            "M1149 author authority drift")
    require(m1146["status"].startswith("PASS_M1146R6_SOURCE_CONTRACT") and
            m1147["status"].startswith("PASS_M1147R6_FINAL_SOURCE_HAMMER") and
            m1148["status"].startswith("PASS_M1148R6_M1146R6_REAL_LMSTAT_PREFLIGHT_FALSE_NEGATIVE"),
            "M1146/M1147/M1148 status chain drift")
    return contract


def load_subject():
    spec = importlib.util.spec_from_file_location("m1150r6_independent_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.configure_base()
    return module


def static_hammer(module) -> dict[str, Any]:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    helper = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
              node.name == "_run_lmstat_redacted"]
    require(len(helper) == 1, "lmstat helper count drift")
    text = ast.unparse(helper[0])
    for token in ("stdout=subprocess.PIPE", "stderr=subprocess.PIPE",
                  "return process.returncode == 0", "stdout = b''", "stderr = b''",
                  "raise Failure('lmstat invocation failed') from None"):
        require(token in text, "missing rc-only/redaction token: " + token)
    require("stderr=subprocess.STDOUT" not in text and "write_text" not in text and
            "write_bytes" not in text and "json.dumps" not in text,
            "lmstat helper persistence surface drift")
    require(module.RESULT == module.BASE.RESULT and module.ATTEMPT == module.BASE.ATTEMPT and
            module.LICENSE_KEYS == ("SNPSLMD_LICENSE_FILE", "LM_LICENSE_FILE"),
            "namespace/precedence drift")
    return {"rc_only": True, "separate_transient_pipes": True,
            "finally_discards_both_buffers": True, "fixed_unchained_exception": True,
            "persistent_write_calls_in_helper": 0}


class MockProcess:
    next_pid = 19000

    def __init__(self, command, rc: int, stdout: bytes = b"", stderr: bytes = b"",
                 timeout: bool = False, cwd: Path | None = None, simv: bool = False):
        self.command = list(command); self.returncode = rc
        self._stdout = stdout; self._stderr = stderr; self._timeout = timeout
        self.pid = MockProcess.next_pid; MockProcess.next_pid += 1
        if simv:
            require(cwd is not None, "mock simv cwd missing")
            (Path(cwd) / "simv").write_text("MOCK_ONLY\n", encoding="utf-8")

    def communicate(self, timeout=None):
        if self._timeout:
            raise subprocess.TimeoutExpired(self.command, timeout,
                                            output=self._stdout, stderr=self._stderr)
        return self._stdout, self._stderr

    def wait(self, timeout=None):
        self.returncode = -15
        return self.returncode


def route_and_helper_hammer(module) -> dict[str, Any]:
    primary = "27000@M1150_PRIMARY_PRIVATE_ROUTE"
    fallback = "27001@M1150_FALLBACK_PRIVATE_ROUTE"
    key, value, metadata = module.BASE._select_license_route({
        "SNPSLMD_LICENSE_FILE": primary, "LM_LICENSE_FILE": fallback})
    require((key, value) == ("SNPSLMD_LICENSE_FILE", primary) and
            set(metadata) == {"selected_variable", "present", "byte_length", "sha256"} and
            primary not in json.dumps(metadata, sort_keys=True), "SNPS priority/metadata drift")
    key2, value2, metadata2 = module.BASE._select_license_route({"LM_LICENSE_FILE": fallback})
    require((key2, value2) == ("LM_LICENSE_FILE", fallback) and
            set(metadata2) == set(metadata), "LM fallback drift")
    for selected_key, selected_value in ((key, value), (key2, value2)):
        child = module.BASE._child_environment(selected_key, selected_value)
        require("HOME" not in child and set(child) ==
                {"LANG", "LC_ALL", "PATH", "VCS_HOME", selected_key},
                "clean child environment drift")

    child = module.BASE._child_environment(key, value)
    popen_calls = []
    def echo_factory(command, **kwargs):
        popen_calls.append(list(command))
        require(kwargs["stdout"] == subprocess.PIPE and kwargs["stderr"] == subprocess.PIPE and
                kwargs["env"] == child and "HOME" not in kwargs["env"],
                "echo subprocess boundary drift")
        return MockProcess(command, 0, ("out:" + primary).encode(),
                           ("err:" + primary).encode())
    with patch.object(subprocess, "Popen", echo_factory):
        require(module._run_lmstat_redacted(key, value, child) is True,
                "rc0 echo should pass")
    require(len(popen_calls) == 1, "rc0 call count drift")

    def nonzero_factory(command, **kwargs):
        return MockProcess(command, 23, ("badout:" + primary).encode(),
                           ("baderr:" + primary).encode())
    with patch.object(subprocess, "Popen", nonzero_factory):
        require(module._run_lmstat_redacted(key, value, child) is False,
                "nonzero should be false")

    def exception_factory(*args, **kwargs):
        raise RuntimeError("underlying:" + primary)
    with patch.object(subprocess, "Popen", exception_factory):
        caught = None
        try:
            module._run_lmstat_redacted(key, value, child)
        except BaseException as error:
            caught = error
        require(caught is not None and str(caught) == "lmstat invocation failed" and
                primary not in str(caught) and caught.__suppress_context__ is True,
                "Popen exception not fixed/redacted/unchained")
        attacks["popen_exception_fixed"] = True

    killed = []
    def timeout_factory(command, **kwargs):
        return MockProcess(command, 99, ("tout:" + primary).encode(),
                           ("terr:" + primary).encode(), timeout=True)
    with patch.object(subprocess, "Popen", timeout_factory), \
            patch.object(os, "killpg", lambda pid, sig: killed.append((pid, sig))):
        require(module._run_lmstat_redacted(key, value, child) is False and
                killed and killed[0][1] == signal.SIGTERM, "timeout cleanup drift")
    attacks["timeout_safe_false"] = True
    return {"snps_priority": True, "lm_fallback": True, "home_absent": True,
            "metadata_fields": sorted(metadata), "rc0_route_echo": True,
            "rc_nonzero_false": True, "timeout_false": True,
            "popen_exception_fixed": True, "raw_output_persisted": False}


def verify_mock_tree(directory: Path, primary: str, secret: str) -> tuple[dict[str, Any], str]:
    manifest = directory / "SHA256SUMS"; outer = directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink(), "mock output missing")
    manifest_sha = sha(manifest)
    require(outer.read_text(encoding="utf-8").split() == [manifest_sha, "SHA256SUMS"],
            "mock outer content drift")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*")
        require(name not in expected and sha(directory / name) == digest,
                "mock member drift")
        expected[name] = digest
    actual = {item.relative_to(directory).as_posix() for item in directory.rglob("*")
              if item.is_file() and item.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(expected), "mock exact census drift")
    for item in directory.rglob("*"):
        if item.is_file():
            payload = item.read_bytes()
            require(secret.encode() not in payload and b"M1150_RAW_STDOUT_SENTINEL" not in payload and
                    b"M1150_RAW_STDERR_SENTINEL" not in payload,
                    "secret/raw lmstat bytes entered output")
    return strict_json(directory / primary), sha(outer)


@contextmanager
def isolated_namespace(module, root: Path) -> Iterator[dict[str, Path | str]]:
    values = {"RESULTS": root, "RESULT": root / "result", "ATTEMPT": root / ".attempt",
              "WORK_PREFIX": ".work.", "FAILURE_PREFIX": "result.failed_or_incomplete.",
              "LOCK": root / ".lock"}
    with ExitStack() as stack:
        stack.enter_context(patch.multiple(module.BASE, **values))
        stack.enter_context(patch.multiple(module, RESULT=values["RESULT"],
                                           ATTEMPT=values["ATTEMPT"],
                                           WORK_PREFIX=values["WORK_PREFIX"],
                                           FAILURE_PREFIX=values["FAILURE_PREFIX"],
                                           LOCK=values["LOCK"]))
        yield values


class FlowFactory:
    def __init__(self, module, secret: str, mode: str):
        self.module = module; self.secret = secret; self.mode = mode; self.commands = []

    def __call__(self, command, **kwargs):
        command = list(command); self.commands.append(command)
        require("HOME" not in kwargs["env"], "HOME leaked to process")
        if command[:2] == [str(self.module.BASE.LMUTIL), "lmstat"]:
            require(kwargs["stdout"] == subprocess.PIPE and kwargs["stderr"] == subprocess.PIPE,
                    "lmstat capture not separated")
            return MockProcess(command, 0,
                               ("M1150_RAW_STDOUT_SENTINEL:" + self.secret).encode(),
                               ("M1150_RAW_STDERR_SENTINEL:" + self.secret).encode())
        if command[0] == str(self.module.BASE.VCS):
            require(len(command) == 14, "frozen compile argv drift")
            return MockProcess(command, 41 if self.mode == "compilefail" else 0,
                               ("compile:" + self.secret).encode(), b"",
                               cwd=kwargs.get("cwd"), simv=self.mode != "compilefail")
        require(command[0].endswith("/simv") and command[1:] == ["-no_save"],
                "unexpected process command")
        payload = self.module.BASE.PASS_TOKEN + "\ncase:" + self.secret + "\n"
        return MockProcess(command, 0, payload.encode(), b"")


def full_flow(module, root: Path, mode: str) -> dict[str, Any]:
    secret = "27000@M1150_FULL_FLOW_PRIVATE_ROUTE"
    environment = {"SNPSLMD_LICENSE_FILE": secret,
                   "LM_LICENSE_FILE": "27001@M1150_UNUSED_FALLBACK",
                   "HOME": "/must/not/propagate"}
    factory = FlowFactory(module, secret, mode)
    with isolated_namespace(module, root), patch.dict(os.environ, environment, clear=True), \
            patch.object(subprocess, "Popen", factory):
        if mode == "success":
            summary = module.production_main()
            require(len(factory.commands) == 3 and module.BASE.RESULT.is_dir() and
                    module.BASE.ATTEMPT.is_dir() and not module.BASE.LOCK.exists() and
                    not list(root.glob(".work.*")) and
                    not list(root.glob("result.failed_or_incomplete.*")),
                    "success namespace/command count drift")
            receipt, result_outer = verify_mock_tree(module.BASE.RESULT, "receipt.json", secret)
            attempt, attempt_outer = verify_mock_tree(module.BASE.ATTEMPT, "attempt.json", secret)
            require(summary["status"] ==
                    "PASS_M1146R6_LICENSE_ROUTE_FROZEN_NETLIST_MAPPED_CASE0_128" and
                    receipt["vcs_compile_attempts"] == 1 and receipt["case0_attempts"] == 1 and
                    receipt["window_cycles"] == 128 and receipt["dc_attempts"] == 0 and
                    receipt["automatic_retry"] is False and
                    receipt["preflight"]["redaction_repair"]["status"] ==
                    "PASS_LMSTAT_RC_ONLY__RAW_STDOUT_STDERR_DISCARDED" and
                    set(receipt["license_route"]) ==
                    {"selected_variable", "present", "byte_length", "sha256"} and
                    attempt["compile_attempts"] == 1 and attempt["case0_attempts"] == 1 and
                    attempt["dc_attempts"] == 0 and attempt["automatic_retry"] is False,
                    "success receipt semantics drift")
            rejected("retry_after_success", module.production_main, "fresh", secret)
            require(len(factory.commands) == 3, "retry launched a process")
            return {"lmstat_compile_case0": [1, 1, 1], "window_cycles": 128,
                    "dc_attempts": 0, "automatic_retry": False,
                    "result_outer_seal_file_sha256": result_outer,
                    "attempt_outer_seal_file_sha256": attempt_outer,
                    "route_or_raw_lmstat_in_tree": False}

        rejected("compile_failure", module.production_main, "mapped VCS compile failed", secret)
        require(len(factory.commands) == 2 and module.BASE.ATTEMPT.is_dir() and
                not module.BASE.RESULT.exists() and not module.BASE.LOCK.exists() and
                not list(root.glob(".work.*")), "compile failure namespace drift")
        failures = list(root.glob("result.failed_or_incomplete.*.quarantine"))
        require(len(failures) == 1, "compile failure quarantine count drift")
        failure, failure_outer = verify_mock_tree(failures[0], "failure.json", secret)
        require(failure["status"] == "FAILED_OR_INCOMPLETE_DO_NOT_CITE" and
                failure["phase"] == "FROZEN_NETLIST_VCS_COMPILE_ONCE" and
                failure["attempt_consumed"] is True and failure["dc_attempts"] == 0 and
                failure["automatic_retry"] is False, "failure snapshot semantics drift")
        rejected("retry_after_failure", module.production_main, "fresh", secret)
        require(len(factory.commands) == 2, "failed retry launched process")
        return {"lmstat_compile_case0": [1, 1, 0], "attempt_consumed": True,
                "quarantine_count": 1, "automatic_retry": False,
                "failure_outer_seal_file_sha256": failure_outer,
                "route_or_raw_lmstat_in_tree": False}


def collision_hammer(module, root: Path) -> dict[str, Any]:
    secret = "27000@M1150_COLLISION_PRIVATE_ROUTE"
    factory = FlowFactory(module, secret, "success")
    with isolated_namespace(module, root), patch.dict(os.environ, {
            "SNPSLMD_LICENSE_FILE": secret}, clear=True), patch.object(subprocess, "Popen", factory):
        module.BASE.RESULT.mkdir()
        rejected("namespace_collision", module.production_main, "fresh", secret)
        require(len(factory.commands) == 0 and not module.BASE.ATTEMPT.exists(),
                "collision reached subprocess/attempt")
    return {"collision_rejected_before_lmstat": True, "processes": 0,
            "attempt_created": False}


def argument_attack(module) -> None:
    with patch.object(sys, "argv", [str(SOURCE), "extra"]):
        module.main()


def main() -> None:
    contract = verify_inputs()
    module = load_subject()
    static = static_hammer(module)
    before = {"source": sha(SOURCE), "base": sha(module.BASE_SOURCE),
              "contract": sha(CONTRACT), "author_outer": sha(AUTHOR / "SHA256SUMS.seal.sha256"),
              "docs359": sha(DOCS359)}
    helper = route_and_helper_hammer(module)
    with tempfile.TemporaryDirectory(prefix="m1150r6_independent_mock_") as temporary:
        temp = Path(temporary)
        success_root = temp / "success"; success_root.mkdir()
        success = full_flow(module, success_root, "success")
        failure_root = temp / "failure"; failure_root.mkdir()
        failure = full_flow(module, failure_root, "compilefail")
        collision_root = temp / "collision"; collision_root.mkdir()
        collision = collision_hammer(module, collision_root)
    rejected("nonzero_argument", lambda: argument_attack(module), "zero arguments")
    after = {"source": sha(SOURCE), "base": sha(module.BASE_SOURCE),
             "contract": sha(CONTRACT), "author_outer": sha(AUTHOR / "SHA256SUMS.seal.sha256"),
             "docs359": sha(DOCS359)}
    require(before == after and before == {
        "source": EXPECTED["source"], "base": module.BASE_SOURCE_SHA,
        "contract": EXPECTED["contract"], "author_outer": EXPECTED["author_outer"],
        "docs359": EXPECTED["docs359"]} and module.BASE.namespace_fresh(),
        "frozen identities or real M1146 namespace changed")
    report = {
        "schema": "m1150r6_m1149r6_c2_preflight_redaction_final_hammer_r1_v1",
        "status": "PASS_M1150R6_M1149R6_FINAL_REDACTION_HAMMER__ROOT_EXTERNAL_REAL_LMSTAT_PREFLIGHT_THEN_ONE_EXACT_EXECUTION_ONLY",
        "score": 100, "checks": checks, "attacks": attacks,
        "identity": {
            "source_sha256": EXPECTED["source"],
            "contract_sha256": EXPECTED["contract"],
            "contract_sidecar_sha256": EXPECTED["contract_side"],
            "contract_outer_seal_file_sha256": EXPECTED["contract_outer"],
            "author_outer_seal_file_sha256": EXPECTED["author_outer"],
            "m1146_outer_seal_file_sha256": EXPECTED["m1146_outer"],
            "m1147_outer_seal_file_sha256": EXPECTED["m1147_outer"],
            "m1148_outer_seal_file_sha256": EXPECTED["m1148_outer"],
            "docs359_sha256": EXPECTED["docs359"],
        },
        "static": static, "license_route_and_lmstat": helper,
        "controlled_success": success, "controlled_compile_failure": failure,
        "collision": collision,
        "execution": {"real_lmstat": 0, "real_vcs": 0, "real_dc": 0,
                      "real_launch": 0, "real_attempt_created": False,
                      "real_result_created": False},
        "authorization": {
            "root_external_real_lmstat_preflight_required": True,
            "then_one_exact_zero_argument_execution": True,
            "future_namespace": "M1146R6",
            "future_vcs_compile_attempts": 1, "future_case0_attempts": 1,
            "future_case0_window_cycles": 128, "future_dc_attempts": 0,
            "automatic_retry": False, "any_second_attempt": False,
        },
        "claim_boundary": {"source_and_mock_only": True, "mapped_functionality": False,
                           "area_timing_power_energy": False, "cycles_speedup": False,
                           "paper_citable": False, "paper_ppa_ready": False},
    }
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    require("PRIVATE_ROUTE" not in encoded and "RAW_STDOUT_SENTINEL" not in encoded and
            "RAW_STDERR_SENTINEL" not in encoded, "review report leaked mock secret/raw bytes")
    print(encoded)


if __name__ == "__main__":
    main()
