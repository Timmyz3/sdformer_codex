#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1146R6 author mock check; no VCS, DC, lmstat, or EDA process is launched."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterator
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent; HW = HERE.parents[1]
SOURCE = HW / "dc_handoff/scripts/run_m1146r6_c2_license_route_frozen_netlist_mapped_vcs_successor_source_r1.py"
CONTRACT = HW / "contracts/m1146r6_c2_additive_license_route_successor_source_contract_r1_20260830.json"
AUTHORITY = HW / "reviews/m1145r6_m1143r6_c2_license_environment_failure_hammer_r1_20260830"
EXPECTED = {"source": "69c30ccfdc884aecca407b6b86b66bc82f97dd02abdb353561daa083934d591c",
            "contract": "dda71d8842325b3c26cd3046d1e93781103f5e5b4277af71602793c68ef7bfd5",
            "contract_side": "1182ef948471bd0235ff8d57817f4dc09295165f2be6ef600d7591348f4f5975",
            "authority_outer": "9edbc8abd3b47bbec576b35d00602cba5abca01cbee320081f954cca9e820148"}
checks = 0; attacks = {}


class CheckFailure(RuntimeError): pass
def require(value: bool, message: str) -> None:
    global checks; checks += 1
    if not value: raise CheckFailure(message)
def rejected(label: str, action: Callable[[], Any]) -> None:
    try: action()
    except Exception as error: attacks[label] = type(error).__name__ + ": " + str(error); return
    raise CheckFailure("attack accepted: " + label)
def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source():
    require(sha(SOURCE) == EXPECTED["source"] and sha(CONTRACT) == EXPECTED["contract"] and
            sha(Path(str(CONTRACT) + ".sha256")) == EXPECTED["contract_side"] and
            sha(AUTHORITY / "SHA256SUMS.seal.sha256") == EXPECTED["authority_outer"], "identity drift")
    spec = importlib.util.spec_from_file_location("m1146r6_author_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "module spec")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module); return module


class FakeProcess:
    def __init__(self, command, environment, output=b"ok", rc=0):
        self.command = list(command); self.environment = dict(environment); self.output = output; self.returncode = rc; self.pid = 999999
    def communicate(self, timeout): require(timeout in (30, 300, 1800), "timeout drift"); return self.output, None
    def wait(self, timeout=None): return self.returncode


def license_routes(module) -> dict[str, Any]:
    snps = "27000@snps.example"; lm = "27001@lm.example"
    key, value, meta = module._select_license_route({"SNPSLMD_LICENSE_FILE": snps, "LM_LICENSE_FILE": lm})
    require((key, value, meta["byte_length"], meta["sha256"]) ==
            ("SNPSLMD_LICENSE_FILE", snps, len(snps.encode()), hashlib.sha256(snps.encode()).hexdigest()),
            "SNPS priority drift")
    key2, value2, _ = module._select_license_route({"SNPSLMD_LICENSE_FILE": "", "LM_LICENSE_FILE": lm})
    require((key2, value2) == ("LM_LICENSE_FILE", lm), "LM fallback drift")
    for selected_key, selected_value in ((key, value), (key2, value2)):
        child = module._child_environment(selected_key, selected_value)
        require("HOME" not in child and selected_key in child and child[selected_key] == selected_value and
                not ({"SNPSLMD_LICENSE_FILE", "LM_LICENSE_FILE"} - {selected_key}) & set(child),
                "clean child env drift")
    rejected("missing_route", lambda: module._select_license_route({}))
    rejected("control_character", lambda: module._select_license_route({"SNPSLMD_LICENSE_FILE": "bad\nroute"}))
    calls = []
    def fake_popen(command, stdout, stderr, env, start_new_session):
        calls.append((list(command), dict(env)))
        return FakeProcess(command, env, b"license status available", 0)
    child = module._child_environment(key, value)
    with patch.object(subprocess, "Popen", fake_popen):
        require(module._run_lmstat(key, value, child) is True, "mock lmstat unavailable")
    require(calls == [([str(module.LMUTIL), "lmstat", "-c", value], child)] and
            "HOME" not in calls[0][1], "lmstat command/env drift")
    return {"precedence": ["SNPSLMD_LICENSE_FILE", "LM_LICENSE_FILE"],
            "snps_selected": True, "lm_fallback": True, "lmstat_mock_available": True,
            "home_key_absent": True, "persistent_value": False}


def redaction_check(module) -> dict[str, Any]:
    secret = "27000@secret.route"; key = "SNPSLMD_LICENSE_FILE"; child = module._child_environment(key, secret)
    with tempfile.TemporaryDirectory(prefix="m1146r6_redact_") as temp:
        root = Path(temp); log = root / "compile.log"
        def fake_popen(command, stdout, stderr, cwd, env, start_new_session):
            require("HOME" not in env and env[key] == secret, "run child env drift")
            return FakeProcess(command, env, ("prefix " + secret + " suffix\n").encode(), 0)
        with patch.object(subprocess, "Popen", fake_popen):
            require(module._run_command(["fake"], log, 1800, child, secret) == 0, "fake command rc")
        require(secret.encode() not in log.read_bytes() and b"<REDACTED_LICENSE_ROUTE>" in log.read_bytes(),
                "route log redaction failure")
        rejected("json_secret", lambda: module._write_json(root / "bad.json", {"secret": secret}, secret))
        (root / "bad.bin").write_text(secret, encoding="utf-8")
        rejected("seal_secret", lambda: module._seal_tree(root, secret))
    return {"compile_log_redacted_before_write": True, "json_secret_rejected": True,
            "seal_secret_rejected": True}


@contextmanager
def fake_namespace(module, root: Path, preflight, key, secret, child) -> Iterator[None]:
    with patch.multiple(module, RESULTS=root, RESULT=root / "result", ATTEMPT=root / ".attempt",
                        LOCK=root / ".lock", WORK_PREFIX=".work.",
                        FAILURE_PREFIX="result.failed_or_incomplete."), \
         patch.object(module, "source_preflight",
                      lambda require_fresh=True: (preflight, key, secret, child)):
        yield


class FakeRunner:
    def __init__(self, module, secret): self.module = module; self.secret = secret; self.commands = []
    def __call__(self, command, log, timeout, environment, secret):
        require(secret == self.secret and "HOME" not in environment and secret in environment.values(),
                "future child env/secret drift")
        self.commands.append(list(command))
        if len(self.commands) == 1:
            require(len(command) == 14 and timeout == 1800 and command == self.module._compile_command(log.parent),
                    "future compile drift")
            log.write_text("MOCK COMPILE <REDACTED_LICENSE_ROUTE>\n", encoding="utf-8")
            (log.parent / "simv").write_text("MOCK SIMV\n", encoding="utf-8"); return 0
        require(len(self.commands) == 2 and timeout == 300 and
                command == self.module._case0_command(log.parent), "future case0 drift")
        log.write_text(self.module.PASS_TOKEN + "\n", encoding="utf-8"); return 0


def atomic_mock(module) -> dict[str, Any]:
    secret = "27000@mock.route"; key = "SNPSLMD_LICENSE_FILE"; child = module._child_environment(key, secret)
    route = {"selected_variable": key, "present": True, "byte_length": len(secret.encode()),
             "sha256": hashlib.sha256(secret.encode()).hexdigest()}
    preflight = {"status": "MOCK_PREFLIGHT", "route": route, "lmstat_available": True,
                 "home_key_in_child_environment": False,
                 "structural_reset_gate": {"shadow_register_bits": 337, "active_low_clear_nets": 12,
                    "direct_inverter_registers": 75, "buffered_then_inverter_registers": 262,
                    "maximum_chain_cells": 2}, "dc_attempts": 0}
    with tempfile.TemporaryDirectory(prefix="m1146r6_atomic_") as temp:
        root = Path(temp); runner = FakeRunner(module, secret)
        with fake_namespace(module, root, preflight, key, secret, child), \
             patch.object(module, "_run_command", runner), \
             patch.object(subprocess, "Popen", side_effect=CheckFailure("real process forbidden")):
            summary = module._future_execute_once()
            require(len(runner.commands) == 2 and module.RESULT.is_dir() and module.ATTEMPT.is_dir() and
                    not module.LOCK.exists() and not list(root.glob(".work.*")) and
                    secret not in "".join(p.read_text(encoding="utf-8", errors="ignore")
                                          for p in root.rglob("*") if p.is_file()),
                    "atomic mock result/secret boundary")
            attempt = json.loads((module.ATTEMPT / "attempt.json").read_text())
            receipt = json.loads((module.RESULT / "receipt.json").read_text())
            require(attempt["dc_attempts"] == receipt["dc_attempts"] == 0 and
                    attempt["compile_attempts"] == attempt["case0_attempts"] == 1 and
                    receipt["vcs_compile_attempts"] == receipt["case0_attempts"] == 1 and
                    receipt["window_cycles"] == 128 and receipt["pass_token_count"] == 1 and
                    summary["status"] == "PASS_M1146R6_LICENSE_ROUTE_FROZEN_NETLIST_MAPPED_CASE0_128",
                    "atomic receipt budget drift")
            rejected("mock_retry", module._future_execute_once)
    return {"compile_attempts": 1, "case0_attempts": 1, "window_cycles": 128,
            "dc_attempts": 0, "retry_rejected": True, "secret_persisted": False,
            "atomic_result": True}


def main() -> None:
    module = load_source(); text = SOURCE.read_text(encoding="utf-8"); tree = ast.parse(text)
    require("HOME" not in module._child_environment("SNPSLMD_LICENSE_FILE", "x") and
            "os.environ.copy" not in text and "automatic_retry" in text, "static env/retry boundary")
    command = module._compile_command(Path("/mock"))
    require(len(command) == 14 and module._case0_command(Path("/mock")) == ["/mock/simv", "-no_save"],
            "14-argument/case0 command drift")
    route = license_routes(module); redaction = redaction_check(module); atomic = atomic_mock(module)
    require(module.namespace_fresh() and not module.RESULT.exists() and not module.ATTEMPT.exists(),
            "real namespace changed")
    result = {"schema": "m1146r6_license_route_successor_author_mock_r1_v1",
              "status": "PASS_M1146R6_SOURCE_CONTRACT_CONTROLLED_MOCK__DIFFERENT_AUTHOR_HAMMER_REQUIRED",
              "checks": checks, "attacks": attacks, "license_route": route,
              "redaction": redaction, "atomic_mock": atomic,
              "execution": {"vcs": False, "dc": False, "lmstat_real": False,
                            "launch": False, "attempt_created": False, "result_created": False},
              "authorization": {"different_author_hammer_only_next": True,
                                "vcs": False, "dc": False, "launch": False,
                                "automatic_retry": False},
              "source_sha256": EXPECTED["source"], "contract_sha256": EXPECTED["contract"],
              "authority_outer_seal_file_sha256": EXPECTED["authority_outer"]}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__": main()
