#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1143R6 author check: real static 337 gate plus controlled fake runner only."""
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
SOURCE = HW / "dc_handoff/scripts/run_m1143r6_c2_frozen_netlist_mapped_vcs_successor_source_r1.py"
CONTRACT = HW / "contracts/m1143r6_c2_frozen_netlist_mapped_vcs_successor_source_contract_r1_20260830.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
ORIGINAL_FAILURE = HW / ("results/m1133r6_c2_authority_schema_repair_dc_mapped_vcs_r1_"
                         "20260830.failed_or_incomplete.1172090.quarantine")
NETLIST = ORIGINAL_FAILURE / ("dc/netlist/"
                              "m1129r5_c2_k1_async_observation_shadow_wrapper_mapped.v")
EXPECTED = {
    "source": "d112129e9c068d4b609852fc8e824dd986f6d3f923bf2cf132b3a6ac28298471",
    "contract": "6a9b5124dcc33b7002a17ab15af0f5e6e74b561ae5afc967843372c311511c13",
    "contract_side": "e0ef599110e948317b4db39a82b2177834a398d6f6a9f7b53d78d8de5c618fc2",
    "contract_outer": "003cec4159567311dcf6c0bb1656a343ec2d7b317b02f44df5bcf236a562c63a",
    "m1142_outer": "558b2855abd85b147ee18456796fde728623e0f43777438a8194d6de85c6c793",
    "netlist": "362e855cd3b4391d31dc7a08e5388d9545f289c81d291c512d25294a8539cbc4",
    "original_failure_manifest": "cbac2199f94723aa39ec3ae2e3b535dfa03e509cedb0b6ac226269b8eab7dd7e",
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
    except Exception as error:
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
            "SOURCE_ONLY__CONTROLLED_FAKE_ONLY__DIFFERENT_AUTHOR_HAMMER_REQUIRED__NO_VCS_NO_EDA" and
            contract["source"]["arguments"] == 0 and
            contract["sole_authorization"]["outer_seal_file_sha256"] ==
            EXPECTED["m1142_outer"] and
            contract["future_case0"]["compile_attempts"] == 1 and
            contract["future_case0"]["simulation_attempts"] == 1 and
            contract["future_case0"]["window_cycles"] == 128 and
            contract["future_case0"]["sdf_option"] is False and
            contract["future_case0"]["dc_rerun"] is False and
            contract["authorization"]["mapped_vcs_now"] is False and
            contract["authorization"]["automatic_retry"] is False,
            "contract semantic drift")
    return contract


def load_subject():
    spec = importlib.util.spec_from_file_location("m1143r6_author_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "subject module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def static_checks(module, contract: dict[str, Any]) -> dict[str, Any]:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    main = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
            node.name == "main"]
    production = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                  node.name == "production_main"]
    require(len(main) == len(production) == 1 and
            len(main[0].args.args) == len(production[0].args.args) == 0,
            "zero argument entry drift")
    require("len(sys.argv) == 1" in ast.unparse(main[0]) and
            "_future_execute_once()" in ast.unparse(production[0]),
            "zero argument binding drift")
    compile_function = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                        node.name == "_compile_command"]
    require(len(compile_function) == 1, "compile builder drift")
    command = module._compile_command(Path("/controlled/fake_mapped"))
    require(command == [
        str(module.VCS), "-full64", "-sverilog", "+v2k", "-timescale=1ns/1ps",
        "-Mdir=/controlled/fake_mapped/csrc", str(module.CELL),
        str(module.NETLIST), str(module.MEMORY), str(module.TB), "-top",
        module.TB_TOP, "-o", "/controlled/fake_mapped/simv"],
        "exact original case0 compile command drift")
    require(not any(item.lower().startswith(("-sdf", "+sdf")) for item in command) and
            module._case0_command(Path("/controlled/fake_mapped")) ==
            ["/controlled/fake_mapped/simv", "-no_save"] and
            "run_dc" not in text and "dc_shell" not in text,
            "SDF/DC/case0 source boundary drift")
    require(contract["fresh_namespace"]["maximum_attempts"] == 1 and
            contract["fresh_namespace"]["automatic_retry"] is False,
            "one-shot contract drift")
    return {"zero_argument": True, "compile_command_arguments": len(command),
            "sdf_options": 0, "dc_invocations": 0,
            "future_compile_attempts": 1, "future_case0_attempts": 1}


def verify_tree(directory: Path, primary_name: str) -> tuple[dict[str, Any], str, str]:
    primary = directory / primary_name
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    manifest_sha = sha(manifest)
    require(outer.read_text(encoding="utf-8").split() ==
            [manifest_sha, "SHA256SUMS"], "output outer content drift")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*")
        require(name not in expected and sha(directory / name) == digest,
                "output member drift")
        expected[name] = digest
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(expected), "output exact member census drift")
    return strict_json(primary), manifest_sha, sha(outer)


@contextmanager
def fake_namespace(module, root: Path, preflight: dict[str, Any]) -> Iterator[dict[str, Path]]:
    paths = {
        "RESULTS": root,
        "RESULT": root / "result",
        "ATTEMPT": root / ".attempt",
        "LOCK": root / ".lock",
    }
    with patch.multiple(module, RESULTS=paths["RESULTS"], RESULT=paths["RESULT"],
                        ATTEMPT=paths["ATTEMPT"], LOCK=paths["LOCK"],
                        WORK_PREFIX=".work.", FAILURE_PREFIX="result.failed_or_incomplete."), \
            patch.object(module, "source_preflight", lambda require_fresh=True: preflight):
        yield paths


class FakeRunner:
    def __init__(self, module, mode: str):
        self.module = module
        self.mode = mode
        self.commands: list[list[str]] = []

    def __call__(self, command, log, timeout, environment):
        self.commands.append(list(command))
        require(environment == {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                                "PATH": "/opt/synopsys/vcs/V-2023.12-SP1/bin:/usr/bin:/bin",
                                "VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                                "HOME": "/tmp"}, "controlled environment drift")
        if len(self.commands) == 1:
            require(command[0] == str(self.module.VCS) and timeout == 1800 and
                    command == self.module._compile_command(log.parent) and
                    not any(item.lower().startswith(("-sdf", "+sdf")) for item in command),
                    "fake compile command drift")
            log.write_text("CONTROLLED_FAKE_VCS_COMPILE\n", encoding="utf-8")
            if self.mode == "compile_failure":
                return 1
            (log.parent / "simv").write_text("CONTROLLED_FAKE_SIMV\n", encoding="utf-8")
            return 0
        require(len(self.commands) == 2 and timeout == 300 and
                command == self.module._case0_command(log.parent),
                "fake case0 command drift")
        token = self.module.PASS_TOKEN
        if self.mode == "missing_token":
            text = "CONTROLLED_FAKE_CASE0_NO_PASS\n"
        elif self.mode == "duplicate_token":
            text = token + "\n" + token + "\n"
        elif self.mode == "x_token":
            text = token + "\nM1112_FIRST_X controlled\n"
        else:
            text = token + "\n"
        log.write_text(text, encoding="utf-8")
        return 0


def run_success(module, root: Path, preflight: dict[str, Any]) -> dict[str, Any]:
    runner = FakeRunner(module, "success")
    with fake_namespace(module, root, preflight), \
            patch.object(module, "_run_command", runner), \
            patch.object(subprocess, "Popen", side_effect=CheckFailure("real subprocess forbidden")):
        summary = module._future_execute_once()
        result = module.RESULT; attempt = module.ATTEMPT
        require(len(runner.commands) == 2 and result.is_dir() and attempt.is_dir() and
                not module.LOCK.exists() and not list(root.glob(".work.*")) and
                not list(root.glob("result.failed_or_incomplete.*")),
                "success atomic namespace drift")
        receipt, manifest_sha, outer_sha = verify_tree(result, "receipt.json")
        attempt_value, _, attempt_outer = verify_tree(attempt, "attempt.json")
        require(receipt["status"] ==
                "PASS_M1143R6_FROZEN_NETLIST_STRUCTURAL_337_MAPPED_CASE0_128__RESULT_HAMMER_REQUIRED" and
                receipt["vcs_compile_attempts"] == receipt["case0_attempts"] == 1 and
                receipt["window_cycles"] == 128 and receipt["sdf_mode"] == "NONE" and
                receipt["dc_rerun"] is False and receipt["automatic_retry"] is False and
                attempt_value["compile_attempts"] == attempt_value["case0_attempts"] == 1 and
                attempt_value["dc_attempts"] == 0 and
                summary["outer_seal_file_sha256"] == outer_sha,
                "success receipt/attempt drift")
    return {"commands": 2, "compile_commands": 1, "case0_commands": 1,
            "manifest_sha256": manifest_sha, "outer_seal_file_sha256": outer_sha,
            "attempt_outer_seal_file_sha256": attempt_outer,
            "window_cycles": 128, "pass_token_count": 1}


def run_failure(module, root: Path, preflight: dict[str, Any], mode: str) -> dict[str, Any]:
    runner = FakeRunner(module, mode)
    with fake_namespace(module, root, preflight), \
            patch.object(module, "_run_command", runner), \
            patch.object(subprocess, "Popen", side_effect=CheckFailure("real subprocess forbidden")):
        rejected(mode, module._future_execute_once)
        require(module.ATTEMPT.is_dir() and not module.RESULT.exists() and
                not module.LOCK.exists() and not list(root.glob(".work.*")),
                mode + " namespace drift")
        failures = list(root.glob("result.failed_or_incomplete.*.quarantine"))
        require(len(failures) == 1, mode + " exact quarantine drift")
        failure, _, outer_sha = verify_tree(failures[0], "failure.json")
        require(failure["status"] == "FAILED_OR_INCOMPLETE_DO_NOT_CITE" and
                failure["attempt_consumed"] is True and
                failure["dc_rerun"] is False and
                failure["automatic_retry"] is False,
                mode + " failure boundary drift")
        rejected(mode + "_retry", module._future_execute_once, "namespace")
    return {"commands_before_failure": len(runner.commands),
            "attempt_consumed": True, "quarantines": 1,
            "automatic_retry": False, "outer_seal_file_sha256": outer_sha}


def collision_attack(module, root: Path, preflight: dict[str, Any]) -> None:
    result = root / "result"; result.mkdir()
    runner = FakeRunner(module, "success")
    with fake_namespace(module, root, preflight), patch.object(module, "_run_command", runner):
        rejected("result_collision", module._future_execute_once, "namespace")
        require(not module.ATTEMPT.exists() and len(runner.commands) == 0 and
                not list(root.glob("result.failed_or_incomplete.*")),
                "collision consumed attempt or ran command")


def runtime_argument_attack(module) -> None:
    with patch.object(sys, "argv", [str(SOURCE), "unexpected"]):
        module.main()


def main() -> None:
    contract = verify_contract()
    module = load_subject()
    static = static_checks(module, contract)
    before = {"source": sha(SOURCE), "netlist": sha(NETLIST),
              "failure_manifest": sha(ORIGINAL_FAILURE / "SHA256SUMS"),
              "docs359": sha(DOCS359)}
    real_process_calls = 0
    def forbid_process(*_args, **_kwargs):
        nonlocal real_process_calls
        real_process_calls += 1
        raise CheckFailure("EDA/subprocess forbidden in author check")
    with patch.object(subprocess, "Popen", forbid_process):
        preflight = module.source_preflight(require_fresh=True)
        static_oracle = module.source_static_self_test()
    require(preflight["structural_reset_gate"] == {
                "shadow_register_bits": 337, "active_low_clear_nets": 12,
                "direct_inverter_registers": 75,
                "buffered_then_inverter_registers": 262,
                "maximum_chain_cells": 2} and
            preflight["sdf_mode"] == "NONE__PRESERVE_ORIGINAL_CASE0_CONTRACT" and
            static_oracle["vcs_executed"] is False and real_process_calls == 0,
            "real preflight/structural boundary drift")
    with tempfile.TemporaryDirectory(prefix="m1143r6_author_fake_") as temp:
        root = Path(temp)
        success_root = root / "success"; success_root.mkdir()
        success = run_success(module, success_root, preflight)
        failures = {}
        for mode in ("compile_failure", "missing_token", "duplicate_token", "x_token"):
            case = root / mode; case.mkdir()
            failures[mode] = run_failure(module, case, preflight, mode)
        collision = root / "collision"; collision.mkdir()
        collision_attack(module, collision, preflight)
    rejected("runtime_argument", lambda: runtime_argument_attack(module), "zero arguments")
    after = {"source": sha(SOURCE), "netlist": sha(NETLIST),
             "failure_manifest": sha(ORIGINAL_FAILURE / "SHA256SUMS"),
             "docs359": sha(DOCS359)}
    require(before == after == {"source": EXPECTED["source"],
                                "netlist": EXPECTED["netlist"],
                                "failure_manifest": EXPECTED["original_failure_manifest"],
                                "docs359": EXPECTED["docs359"]},
            "frozen subject/failure identity changed")
    require(module.namespace_fresh() and not module.ATTEMPT.exists() and
            not module.RESULT.exists(), "real M1143 namespace changed")
    report = {
        "schema": "m1143r6_author_static_controlled_fake_checks_r1_v1",
        "status": "PASS_M1143R6_SOURCE_STRUCTURAL_337_CONTROLLED_FAKE__DIFFERENT_AUTHOR_HAMMER_ONLY",
        "checks_passed": checks,
        "attacks_rejected": len(attacks), "attacks": attacks,
        "static": static,
        "real_read_only_preflight": preflight,
        "controlled_fake_success": success,
        "controlled_fake_failures": failures,
        "production_boundary": {"real_process_calls": real_process_calls,
                                "vcs_compile": 0, "case0_runs": 0,
                                "dc_rerun": 0, "attempt_created": False,
                                "result_created": False,
                                "original_subject_or_failure_modified": False},
        "identity": {"source_sha256": EXPECTED["source"],
                     "contract_sha256": EXPECTED["contract"],
                     "contract_sidecar_sha256": EXPECTED["contract_side"],
                     "contract_outer_seal_file_sha256": EXPECTED["contract_outer"],
                     "m1142_outer_seal_file_sha256": EXPECTED["m1142_outer"],
                     "netlist_sha256": EXPECTED["netlist"],
                     "docs359_sha256": EXPECTED["docs359"]},
        "authorization": {"different_author_hammer_only": True,
                          "mapped_vcs_execution": False, "dc": False,
                          "launch": False, "automatic_retry": False},
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
