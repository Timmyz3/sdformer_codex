#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1149R6 author mock: intercept every process; no real lmstat/VCS/DC/launch."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
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
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    "source": "affeba86cc465462ef00a50afe283d9e0677c06362c9830a1bf9ea5f502363ce",
    "contract": "1c4daec4c4ab9dd4ab579edc247068c514d27c18136c91852828cb6077c13802",
    "contract_side": "de2b22ed44d28744f7ecba260fda1c7c8c603fddc6891341f6432334170bba9d",
    "contract_outer": "bbdbc9fcc974a8763bb54d53c7c8b65608e3e535fd6076db227ab2aa52d9795a",
    "m1146_outer": "513813aa1915e72af18c1b059cfae77947c9ece37fc8699582cc202c489b98d1",
    "m1147_outer": "64007fe4ec37a26c54c197b80ae9f9565e8272c06fecfe3510c24aeb7c74d7e9",
    "m1148_outer": "b60fb9ecd875d87dd0b1f05a8cd448c85ce551a4f589a988f6a0fa8785defb32",
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


def rejected(label: str, action: Callable[[], Any], contains: str | None = None,
             forbidden: str | None = None) -> None:
    try:
        action()
    except Exception as error:
        message = str(error)
        if contains is not None:
            require(contains in message, label + " wrong rejection: " + message)
        if forbidden is not None:
            require(forbidden not in message, label + " leaked secret: " + message)
        attacks[label] = type(error).__name__ + ": " + message
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


def strict_json(path: Path) -> Any:
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          CheckFailure("nonfinite JSON: " + token)))


def verify_contract() -> dict[str, Any]:
    verify_regular(SOURCE, EXPECTED["source"])
    verify_regular(CONTRACT, EXPECTED["contract"])
    side = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    verify_regular(side, EXPECTED["contract_side"])
    verify_regular(outer, EXPECTED["contract_outer"])
    require(side.read_text(encoding="utf-8").split() ==
            [EXPECTED["contract"], CONTRACT.name] and
            outer.read_text(encoding="utf-8").split() ==
            [EXPECTED["contract_side"], side.name], "contract double seal drift")
    contract = strict_json(CONTRACT)
    require(contract["status"] ==
            "SOURCE_ONLY__CONTROLLED_MOCK_ONLY__DIFFERENT_AUTHOR_FINAL_HAMMER_REQUIRED__NO_LMSTAT_NO_VCS" and
            contract["source"]["arguments"] == 0 and
            contract["authority_chain"]["m1148_redaction_failure_hammer"][
                "outer_seal_file_sha256"] == EXPECTED["m1148_outer"] and
            contract["namespace"]["policy"] == "reuse still-fresh M1146 namespace" and
            contract["namespace"]["maximum_attempts"] == 1 and
            contract["lmstat_helper"]["decision"] == "process returncode == 0 only" and
            contract["authorization"]["real_lmstat_now"] is False and
            contract["authorization"]["vcs_now"] is False,
            "contract semantic drift")
    return contract


def load_subject():
    spec = importlib.util.spec_from_file_location("m1149r6_author_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "subject module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.configure_base()
    return module


def static_checks(module, contract: dict[str, Any]) -> dict[str, Any]:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    helper = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
              node.name == "_run_lmstat_redacted"]
    main = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
            node.name == "main"]
    require(len(helper) == len(main) == 1 and len(helper[0].args.args) == 3 and
            len(main[0].args.args) == 0 and "len(sys.argv) == 1" in ast.unparse(main[0]),
            "helper/main schema drift")
    helper_text = ast.unparse(helper[0])
    require("stdout=subprocess.PIPE" in helper_text and
            "stderr=subprocess.PIPE" in helper_text and
            "return process.returncode == 0" in helper_text and
            "stdout = b''" in helper_text and "stderr = b''" in helper_text and
            "raise Failure('lmstat invocation failed') from None" in helper_text and
            "value.encode() not in" not in helper_text and
            "write" not in helper_text.lower() and "json" not in helper_text.lower(),
            "rc-only discard helper drift")
    require(module.LICENSE_KEYS == ("SNPSLMD_LICENSE_FILE", "LM_LICENSE_FILE") and
            module.RESULT == module.BASE.RESULT and module.ATTEMPT == module.BASE.ATTEMPT and
            contract["license_route"]["home_key_absent"] is True and
            contract["preserved_future_execution"]["vcs_compile_attempts"] == 1 and
            contract["preserved_future_execution"]["case0_attempts"] == 1 and
            contract["preserved_future_execution"]["window_cycles"] == 128 and
            contract["preserved_future_execution"]["dc_attempts"] == 0,
            "priority/namespace/future budget drift")
    return {"zero_argument": True, "lmstat_decision": "returncode_only",
            "raw_output_persistence_calls": 0,
            "snps_priority_lm_fallback": True, "home_absent": True,
            "namespace_reused": "M1146R6", "maximum_attempts": 1,
            "future_compile": 1, "future_case0": 1, "future_dc": 0}


class ControlledProcess:
    next_pid = 8000
    def __init__(self, command, *, rc: int, stdout: bytes, stderr: bytes,
                 timeout: bool = False, cwd: Path | None = None,
                 create_simv: bool = False):
        self.command = list(command); self.returncode = rc
        self.stdout = stdout; self.stderr = stderr; self.timeout = timeout
        self.pid = ControlledProcess.next_pid; ControlledProcess.next_pid += 1
        if create_simv and cwd is not None:
            (Path(cwd) / "simv").write_text("CONTROLLED_SIMV\n", encoding="utf-8")

    def communicate(self, timeout=None):
        if self.timeout:
            raise subprocess.TimeoutExpired(self.command, timeout)
        return self.stdout, self.stderr

    def wait(self, timeout=None):
        self.returncode = -15
        return self.returncode


def helper_oracles(module) -> dict[str, Any]:
    secret = "27000@controlled-secret-route"
    key = "SNPSLMD_LICENSE_FILE"
    child = module.BASE._child_environment(key, secret)
    calls = 0
    def factory_success(command, **kwargs):
        nonlocal calls
        calls += 1
        require(kwargs["stderr"] == subprocess.PIPE and kwargs["stdout"] == subprocess.PIPE and
                kwargs["env"] == child and "HOME" not in kwargs["env"],
                "success helper subprocess boundary drift")
        return ControlledProcess(command, rc=0,
                                 stdout=("stdout " + secret).encode(),
                                 stderr=("stderr " + secret).encode())
    with patch.object(subprocess, "Popen", factory_success):
        result = module._run_lmstat_redacted(key, secret, child)
    require(result is True and calls == 1, "rc0 echo oracle drift")

    def factory_failure(command, **kwargs):
        return ControlledProcess(command, rc=7,
                                 stdout=("failure stdout " + secret).encode(),
                                 stderr=("failure stderr " + secret).encode())
    with patch.object(subprocess, "Popen", factory_failure):
        require(module._run_lmstat_redacted(key, secret, child) is False,
                "rc-nonzero oracle drift")

    def factory_exception(*_args, **_kwargs):
        raise RuntimeError("underlying process exception leaked " + secret)
    with patch.object(subprocess, "Popen", factory_exception):
        rejected("lmstat_popen_exception", lambda:
                 module._run_lmstat_redacted(key, secret, child),
                 "lmstat invocation failed", secret)

    killed = []
    def factory_timeout(command, **kwargs):
        return ControlledProcess(command, rc=1, stdout=secret.encode(), stderr=secret.encode(),
                                 timeout=True)
    with patch.object(subprocess, "Popen", factory_timeout), \
            patch.object(os, "killpg", lambda pid, sig: killed.append((pid, sig))):
        require(module._run_lmstat_redacted(key, secret, child) is False and len(killed) == 1,
                "timeout safe false oracle drift")
    return {"rc0_with_route_echo": True, "rc_nonzero": False,
            "popen_exception_safe": True, "timeout_safe_false": True,
            "raw_stdout_returned": False, "raw_stderr_returned": False,
            "route_returned": False}


def selection_oracles(module) -> dict[str, Any]:
    snps = "snps://controlled-primary"
    lm = "lm://controlled-fallback"
    key, value, metadata = module.BASE._select_license_route({
        "SNPSLMD_LICENSE_FILE": snps, "LM_LICENSE_FILE": lm})
    require((key, value) == ("SNPSLMD_LICENSE_FILE", snps) and
            set(metadata) == {"selected_variable", "present", "byte_length", "sha256"} and
            value not in json.dumps(metadata, sort_keys=True) and
            "HOME" not in module.BASE._child_environment(key, value),
            "SNPS priority metadata/HOME drift")
    key2, value2, metadata2 = module.BASE._select_license_route({"LM_LICENSE_FILE": lm})
    require((key2, value2) == ("LM_LICENSE_FILE", lm) and
            value2 not in json.dumps(metadata2, sort_keys=True) and
            "HOME" not in module.BASE._child_environment(key2, value2),
            "LM fallback metadata/HOME drift")
    return {"snps_priority": True, "lm_fallback": True,
            "persistent_fields": sorted(metadata), "home_absent": True}


def verify_sealed_tree(directory: Path, primary: str, secret: str) -> tuple[dict[str, Any], str]:
    require(directory.is_dir() and not directory.is_symlink(), "sealed tree drift")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    manifest_sha = sha(manifest)
    require(outer.read_text(encoding="utf-8").split() == [manifest_sha, "SHA256SUMS"],
            "outer seal content drift")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*")
        require(name not in expected and sha(directory / name) == digest,
                "sealed member drift")
        expected[name] = digest
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(expected), "sealed exact census drift")
    for path in directory.rglob("*"):
        if path.is_file():
            require(secret.encode() not in path.read_bytes(),
                    "route leaked into sealed tree: " + str(path))
    return strict_json(directory / primary), sha(outer)


@contextmanager
def fake_namespace(module, root: Path):
    base = module.BASE
    values = {"RESULTS": root, "RESULT": root / "result", "ATTEMPT": root / ".attempt",
              "WORK_PREFIX": ".work.", "FAILURE_PREFIX": "result.failed_or_incomplete.",
              "LOCK": root / ".lock"}
    with patch.multiple(base, **values):
        yield values


class FullFlowFactory:
    def __init__(self, module, secret: str, mode: str):
        self.module = module; self.secret = secret; self.mode = mode
        self.commands = []

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        require("HOME" not in kwargs["env"], "HOME entered child environment")
        if command[:2] == [str(self.module.BASE.LMUTIL), "lmstat"]:
            require(kwargs["stdout"] == subprocess.PIPE and kwargs["stderr"] == subprocess.PIPE,
                    "lmstat pipe separation drift")
            return ControlledProcess(command, rc=0,
                                     stdout=("lmout " + self.secret).encode(),
                                     stderr=("lmerr " + self.secret).encode())
        if command[0] == str(self.module.BASE.VCS):
            return ControlledProcess(command, rc=1 if self.mode == "compile_failure" else 0,
                                     stdout=("compile " + self.secret).encode(), stderr=b"",
                                     cwd=kwargs.get("cwd"),
                                     create_simv=self.mode != "compile_failure")
        require(command[0].endswith("/simv"), "unexpected controlled command")
        text = self.module.BASE.PASS_TOKEN + "\ncase " + self.secret + "\n"
        return ControlledProcess(command, rc=0, stdout=text.encode(), stderr=b"")


def run_full_flow(module, root: Path, mode: str) -> dict[str, Any]:
    secret = "27000@full-flow-secret-route"
    factory = FullFlowFactory(module, secret, mode)
    environment = {"SNPSLMD_LICENSE_FILE": secret,
                   "LM_LICENSE_FILE": "27000@unused-fallback",
                   "HOME": "/caller/home/must/not/propagate"}
    with fake_namespace(module, root), patch.dict(os.environ, environment, clear=True), \
            patch.object(subprocess, "Popen", factory):
        if mode == "success":
            summary = module.production_main()
            require(len(factory.commands) == 3 and module.BASE.RESULT.is_dir() and
                    module.BASE.ATTEMPT.is_dir() and not module.BASE.LOCK.exists() and
                    not list(root.glob(".work.*")) and
                    not list(root.glob("result.failed_or_incomplete.*")),
                    "success one-shot namespace drift")
            result, result_outer = verify_sealed_tree(
                module.BASE.RESULT, "receipt.json", secret)
            attempt, attempt_outer = verify_sealed_tree(
                module.BASE.ATTEMPT, "attempt.json", secret)
            require(summary["status"] ==
                    "PASS_M1146R6_LICENSE_ROUTE_FROZEN_NETLIST_MAPPED_CASE0_128" and
                    result["vcs_compile_attempts"] == result["case0_attempts"] == 1 and
                    result["window_cycles"] == 128 and result["dc_attempts"] == 0 and
                    result["automatic_retry"] is False and
                    result["preflight"]["redaction_repair"]["status"] ==
                    "PASS_LMSTAT_RC_ONLY__RAW_STDOUT_STDERR_DISCARDED" and
                    attempt["compile_attempts"] == attempt["case0_attempts"] == 1 and
                    attempt["dc_attempts"] == 0,
                    "success receipt/attempt drift")
            logs = [(module.BASE.RESULT / "mapped_vcs/compile.log").read_text(),
                    (module.BASE.RESULT / "mapped_vcs/case0.log").read_text()]
            require(all(secret not in text and "<REDACTED_LICENSE_ROUTE>" in text
                        for text in logs), "command log redaction drift")
            return {"intercepted_processes": 3, "lmstat": 1, "compile": 1, "case0": 1,
                    "result_outer_seal_file_sha256": result_outer,
                    "attempt_outer_seal_file_sha256": attempt_outer,
                    "raw_route_in_any_sealed_member": False}
        rejected("full_compile_failure", module.production_main,
                 "mapped VCS compile failed", secret)
        require(len(factory.commands) == 2 and module.BASE.ATTEMPT.is_dir() and
                not module.BASE.RESULT.exists() and not module.BASE.LOCK.exists() and
                not list(root.glob(".work.*")), "failure namespace drift")
        quarantines = list(root.glob("result.failed_or_incomplete.*.quarantine"))
        require(len(quarantines) == 1, "failure exact quarantine drift")
        failure, outer = verify_sealed_tree(quarantines[0], "failure.json", secret)
        require(failure["status"] == "FAILED_OR_INCOMPLETE_DO_NOT_CITE" and
                failure["attempt_consumed"] is True and failure["dc_attempts"] == 0 and
                failure["automatic_retry"] is False,
                "failure receipt drift")
        rejected("full_compile_failure_retry", module.production_main, "fresh", secret)
        return {"intercepted_processes": 2, "attempt_consumed": True,
                "quarantines": 1, "automatic_retry": False,
                "outer_seal_file_sha256": outer,
                "raw_route_in_any_sealed_member": False}


def runtime_argument_attack(module) -> None:
    with patch.object(sys, "argv", [str(SOURCE), "unexpected"]):
        module.main()


def main() -> None:
    contract = verify_contract()
    module = load_subject()
    static = static_checks(module, contract)
    before = {"source": sha(SOURCE), "base": sha(module.BASE_SOURCE),
              "docs359": sha(DOCS359)}
    selection = selection_oracles(module)
    helper = helper_oracles(module)
    with tempfile.TemporaryDirectory(prefix="m1149r6_author_mock_") as temp:
        root = Path(temp)
        success_root = root / "success"; success_root.mkdir()
        success = run_full_flow(module, success_root, "success")
        failure_root = root / "failure"; failure_root.mkdir()
        failure = run_full_flow(module, failure_root, "compile_failure")
    rejected("runtime_argument", lambda: runtime_argument_attack(module), "zero arguments")
    after = {"source": sha(SOURCE), "base": sha(module.BASE_SOURCE),
             "docs359": sha(DOCS359)}
    require(before == after == {"source": EXPECTED["source"],
                                "base": module.BASE_SOURCE_SHA,
                                "docs359": EXPECTED["docs359"]} and
            module.BASE.namespace_fresh(),
            "frozen identity or real M1146 namespace changed")
    report = {
        "schema": "m1149r6_author_static_controlled_mock_checks_r1_v1",
        "status": "PASS_M1149R6_RC_ONLY_REDACTION_SOURCE_MOCK__DIFFERENT_AUTHOR_FINAL_HAMMER_ONLY",
        "checks_passed": checks, "attacks_rejected": len(attacks), "attacks": attacks,
        "static": static, "selection": selection, "lmstat_helper": helper,
        "controlled_full_success": success,
        "controlled_full_failure": failure,
        "execution": {"real_lmstat": 0, "real_vcs": 0, "real_dc": 0,
                      "real_launch": 0, "attempt_created": False,
                      "result_created": False},
        "identity": {"source_sha256": EXPECTED["source"],
                     "contract_sha256": EXPECTED["contract"],
                     "contract_sidecar_sha256": EXPECTED["contract_side"],
                     "contract_outer_seal_file_sha256": EXPECTED["contract_outer"],
                     "m1146_outer_seal_file_sha256": EXPECTED["m1146_outer"],
                     "m1147_outer_seal_file_sha256": EXPECTED["m1147_outer"],
                     "m1148_outer_seal_file_sha256": EXPECTED["m1148_outer"],
                     "docs359_sha256": EXPECTED["docs359"]},
        "authorization": {"different_author_final_hammer_only": True,
                          "real_lmstat": False, "launch": False,
                          "vcs": False, "dc": False, "automatic_retry": False},
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
