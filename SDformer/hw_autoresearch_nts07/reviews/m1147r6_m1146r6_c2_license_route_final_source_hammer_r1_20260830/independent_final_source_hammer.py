#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1147R6 independent final source hammer; never launches lmstat, VCS, or DC."""
from __future__ import annotations

import ast
from contextlib import contextmanager
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
from typing import Any, Callable, Iterator
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "dc_handoff/scripts/run_m1146r6_c2_license_route_frozen_netlist_mapped_vcs_successor_source_r1.py"
CONTRACT = HW / "contracts/m1146r6_c2_additive_license_route_successor_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1146r6_c2_additive_license_route_successor_author_receipt_r1_20260830"
NETLIST = HW / ("results/m1133r6_c2_authority_schema_repair_dc_mapped_vcs_r1_20260830."
                "failed_or_incomplete.1172090.quarantine/dc/netlist/"
                "m1129r5_c2_k1_async_observation_shadow_wrapper_mapped.v")
CHECKER = HW / "dc_handoff/scripts/m1141r6_c2_additive_structural_reset_chain_checker_source_r1.py"
CELL = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/"
            "TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/"
            "tcbn28hpcplusbwp35p140.v")
MEMORY = HW / "tb_m349/m349_fc2_scalar_bank_memory_model.sv"
TB = HW / "dc_handoff/tb/tb_m1129r5_c2_k1_async_observation_shadow_case0_short.sv"
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    "source": "69c30ccfdc884aecca407b6b86b66bc82f97dd02abdb353561daa083934d591c",
    "contract": "dda71d8842325b3c26cd3046d1e93781103f5e5b4277af71602793c68ef7bfd5",
    "contract_side": "1182ef948471bd0235ff8d57817f4dc09295165f2be6ef600d7591348f4f5975",
    "contract_outer": "b28d565b1c1ef7b3c79724bf06bc4be202e55010f88b7b0274adb068a9fb82e6",
    "author_review": "b011596046c724665b71352045c23e82044eaabd5a6ce849be5892b362781fe4",
    "author_manifest": "13fe84f0aef4dfc000278a9e1629368b6256742c6c1ad7f2e769174ac1a6360c",
    "author_outer": "513813aa1915e72af18c1b059cfae77947c9ece37fc8699582cc202c489b98d1",
    "netlist": "362e855cd3b4391d31dc7a08e5388d9545f289c81d291c512d25294a8539cbc4",
    "checker": "86ccd46fdaffcad77444ca105bde1593394dd7643febba1f6a45680bf515965e",
    "cell": "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a",
    "memory": "4375072b6bd09ada3dc3fd585c12102346ea897192a13630b0c44acf72ff63fa",
    "tb": "c08d22d69c222b8c527bdb70cc5b49392c5467bc3142ebc22ec577da6918147b",
    "vcs": "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    "lmutil": "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
checks = 0
attacks: dict[str, str] = {}


class HammerFailure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise HammerFailure(message)


def rejected(label: str, action: Callable[[], Any]) -> None:
    try:
        action()
    except Exception as error:
        attacks[label] = type(error).__name__ + ": " + str(error)
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


def sealed_tree(directory: Path, primary: str,
                identities: tuple[str, str, str]) -> dict[str, Any]:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    regular(directory / primary, identities[0])
    regular(manifest, identities[1])
    regular(outer, identities[2])
    require(outer.read_text(encoding="utf-8").split() == [identities[1], "SHA256SUMS"],
            "outer seal content drift")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        rel = Path(name)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and name not in expected and
                name == rel.as_posix() and not rel.is_absolute() and ".." not in rel.parts,
                "unsafe manifest member")
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
    return strict_json(directory / primary)


def verify_inputs() -> dict[str, str]:
    for path, key in (
        (SOURCE, "source"), (CONTRACT, "contract"),
        (Path(str(CONTRACT) + ".sha256"), "contract_side"),
        (Path(str(CONTRACT) + ".sha256.seal.sha256"), "contract_outer"),
        (NETLIST, "netlist"), (CHECKER, "checker"), (CELL, "cell"),
        (MEMORY, "memory"), (TB, "tb"), (VCS, "vcs"), (LMUTIL, "lmutil"),
        (DOCS359, "docs359"),
    ):
        regular(path, EXPECTED[key])
    # The outer seal is b28d..., not the hand-transcription family beginning 6632.
    require(EXPECTED["contract_outer"].startswith("b28d565b") and
            not EXPECTED["contract_outer"].startswith("6632"),
            "contract outer hand-typo accepted")
    author = sealed_tree(AUTHOR, "review.json", (EXPECTED["author_review"],
                         EXPECTED["author_manifest"], EXPECTED["author_outer"]))
    require(author["status"] ==
            "PASS_M1146R6_SOURCE_CONTRACT_CONTROLLED_MOCK__DIFFERENT_AUTHOR_HAMMER_REQUIRED" and
            author["authorization"]["different_author_hammer_only_next"] is True and
            all(author["authorization"][key] is False
                for key in ("launch", "vcs", "dc", "automatic_retry")),
            "author receipt semantic drift")
    contract = strict_json(CONTRACT)
    require(contract["source"]["sha256"] == EXPECTED["source"] and
            contract["source"]["arguments"] == 0 and
            contract["source"]["new_namespace"] is True and
            contract["source"]["automatic_retry"] is False and
            contract["future_budget"] == {
                "compile_attempts": 1, "case0_attempts": 1, "window_cycles": 128,
                "compile_command_arguments": 14, "case0_arguments": ["simv", "-no_save"],
                "dc_attempts": 0, "automatic_retry": False, "atomic_result_or_failure": True,
            }, "contract semantic drift")
    return {"source": sha(SOURCE), "contract": sha(CONTRACT),
            "contract_side": sha(Path(str(CONTRACT) + ".sha256")),
            "contract_outer": sha(Path(str(CONTRACT) + ".sha256.seal.sha256")),
            "author_outer": sha(AUTHOR / "SHA256SUMS.seal.sha256"),
            "netlist": sha(NETLIST), "checker": sha(CHECKER), "cell": sha(CELL),
            "memory": sha(MEMORY), "tb": sha(TB), "vcs": sha(VCS),
            "lmutil": sha(LMUTIL), "docs359": sha(DOCS359)}


def load_source():
    spec = importlib.util.spec_from_file_location("m1147r6_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_output(directory: Path, primary: str, secret: str) -> tuple[dict[str, Any], str]:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    manifest_sha = sha(manifest)
    require(outer.read_text(encoding="utf-8").split() == [manifest_sha, "SHA256SUMS"],
            "mock output outer seal")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in expected and sha(directory / name) == digest,
                "mock output member drift")
        expected[name] = digest
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(expected), "mock output exact census")
    for path in directory.rglob("*"):
        if path.is_file():
            require(secret.encode() not in path.read_bytes(), "secret persisted in mock output")
    return strict_json(directory / primary), sha(outer)


def route_and_preflight_hammer(module) -> dict[str, Any]:
    snps = "27000@snps.m1147.invalid"
    lm = "27001@lm.m1147.invalid"
    key, value, meta = module._select_license_route(
        {"SNPSLMD_LICENSE_FILE": snps, "LM_LICENSE_FILE": lm})
    require((key, value) == ("SNPSLMD_LICENSE_FILE", snps) and meta == {
        "selected_variable": "SNPSLMD_LICENSE_FILE", "present": True,
        "byte_length": len(snps.encode()), "sha256": hashlib.sha256(snps.encode()).hexdigest(),
    }, "SNPS precedence/presence-length-SHA drift")
    key2, value2, meta2 = module._select_license_route(
        {"SNPSLMD_LICENSE_FILE": "", "LM_LICENSE_FILE": lm})
    require((key2, value2) == ("LM_LICENSE_FILE", lm) and meta2 == {
        "selected_variable": "LM_LICENSE_FILE", "present": True,
        "byte_length": len(lm.encode()), "sha256": hashlib.sha256(lm.encode()).hexdigest(),
    }, "LM fallback/presence-length-SHA drift")
    expected_fixed = {"LANG", "LC_ALL", "PATH", "VCS_HOME"}
    for selected_key, selected_value in ((key, value), (key2, value2)):
        child = module._child_environment(selected_key, selected_value)
        require(set(child) == expected_fixed | {selected_key} and "HOME" not in child and
                child[selected_key] == selected_value and
                not ({"SNPSLMD_LICENSE_FILE", "LM_LICENSE_FILE"} - {selected_key}) & set(child),
                "child environment is not exact selected-license-only")
    rejected("missing_license_route", lambda: module._select_license_route({}))
    rejected("empty_license_route", lambda: module._select_license_route(
        {"SNPSLMD_LICENSE_FILE": "", "LM_LICENSE_FILE": ""}))
    for label, bad in (("newline", "27000@bad\nnext"), ("carriage_return", "bad\rroute"),
                       ("nul", "bad\x00route")):
        rejected("control_character_" + label,
                 lambda bad=bad: module._select_license_route({"SNPSLMD_LICENSE_FILE": bad}))
    child = module._child_environment(key, value)
    rejected("route_mutation_value", lambda: module._run_lmstat(key, value + "x", child))
    child_home = dict(child); child_home["HOME"] = "/tmp"
    rejected("route_mutation_home", lambda: module._run_lmstat(key, value, child_home))
    child_extra = dict(child); child_extra["LM_LICENSE_FILE"] = lm
    rejected("route_mutation_second_license", lambda: module._run_lmstat(key, value, child_extra))

    seen = []
    def fake_lmstat(selected_key, selected_value, environment):
        seen.append((selected_key, selected_value, dict(environment)))
        require(selected_key == "SNPSLMD_LICENSE_FILE" and selected_value == snps and
                environment == module._child_environment(selected_key, selected_value),
                "source preflight lmstat route/env drift")
        return True
    env = {"SNPSLMD_LICENSE_FILE": snps, "LM_LICENSE_FILE": lm}
    with patch.dict(os.environ, env, clear=True), patch.object(module, "_run_lmstat", fake_lmstat), \
         patch.object(subprocess, "Popen", side_effect=HammerFailure("real process forbidden")):
        public, pkey, pvalue, pchild = module.source_preflight(True)
    require(len(seen) == 1 and (pkey, pvalue, pchild) == (key, value, child) and
            public["route"] == meta and public["lmstat_available"] is True and
            public["home_key_in_child_environment"] is False and public["dc_attempts"] == 0 and
            value not in json.dumps(public, sort_keys=True), "public preflight secret/route drift")
    return {"precedence": ["SNPSLMD_LICENSE_FILE", "LM_LICENSE_FILE"],
            "snps_priority": True, "lm_fallback": True,
            "persistent_metadata": ["selected_variable", "present", "byte_length", "sha256"],
            "lmstat_calls_mocked": len(seen), "home_absent": True,
            "only_selected_license_variable": True,
            "structural_reset_gate": public["structural_reset_gate"]}


def secret_boundary_hammer(module) -> dict[str, Any]:
    secret = "27000@secret.m1147.invalid"
    key = "SNPSLMD_LICENSE_FILE"
    child = module._child_environment(key, secret)
    with tempfile.TemporaryDirectory(prefix="m1147r6_secret_") as temp:
        root = Path(temp)
        def fake_popen(command, stdout, stderr, cwd, env, start_new_session):
            require(env == child and "HOME" not in env, "secret test child env drift")
            return FakeProcess(command, env, ("prefix " + secret + " suffix\n").encode(), 0)
        with patch.object(subprocess, "Popen", fake_popen):
            require(module._run_command(["fake"], root / "command.log", 300, child, secret) == 0,
                    "mock redaction command rc")
        log = (root / "command.log").read_bytes()
        require(secret.encode() not in log and b"<REDACTED_LICENSE_ROUTE>" in log,
                "command output secret not redacted before write")
        rejected("secret_leak_json", lambda: module._write_json(
            root / "leak.json", {"route": secret}, secret))
        leak = root / "leak.bin"; leak.write_text(secret, encoding="utf-8")
        rejected("secret_leak_seal", lambda: module._seal_tree(root, secret))
    return {"log_redacted_before_write": True, "json_leak_rejected": True,
            "sealed_member_leak_rejected": True}


class FakeProcess:
    def __init__(self, command, environment, output=b"ok", returncode=0):
        self.command = list(command)
        self.environment = dict(environment)
        self.output = output
        self.returncode = returncode
        self.pid = 999999

    def communicate(self, timeout):
        require(timeout in (30, 300, 1800), "timeout drift")
        return self.output, None

    def wait(self, timeout=None):
        return self.returncode


@contextmanager
def fake_namespace(module, root: Path, preflight: dict[str, Any], key: str,
                   secret: str, child: dict[str, str]) -> Iterator[dict[str, Path]]:
    paths = {"result": root / "result", "attempt": root / ".attempt", "lock": root / ".lock"}
    with patch.multiple(module, RESULTS=root, RESULT=paths["result"], ATTEMPT=paths["attempt"],
                        LOCK=paths["lock"], WORK_PREFIX=".work.",
                        FAILURE_PREFIX="result.failed_or_incomplete."), \
         patch.object(module, "source_preflight",
                      lambda require_fresh=True: (preflight, key, secret, child)):
        yield paths


class FakeRunner:
    def __init__(self, module, secret: str, child: dict[str, str], mode: str):
        self.module = module
        self.secret = secret
        self.child = child
        self.mode = mode
        self.commands: list[list[str]] = []

    def __call__(self, command, log, timeout, environment, secret):
        require(secret == self.secret and environment == self.child and "HOME" not in environment and
                set(environment) == {"LANG", "LC_ALL", "PATH", "VCS_HOME", "SNPSLMD_LICENSE_FILE"},
                "execution child environment/secret drift")
        self.commands.append(list(command))
        if len(self.commands) == 1:
            require(command == self.module._compile_command(log.parent) and len(command) == 14 and
                    timeout == 1800 and str(self.module.NETLIST) in command and
                    str(self.module.CELL) in command and str(self.module.MEMORY) in command and
                    str(self.module.TB) in command, "frozen compile command drift")
            log.write_text("MOCK COMPILE <REDACTED_LICENSE_ROUTE>\n", encoding="utf-8")
            if self.mode == "compile_fail":
                return 1
            if self.mode != "missing_simv":
                (log.parent / "simv").write_text("MOCK SIMV\n", encoding="utf-8")
            return 0
        require(len(self.commands) == 2 and command == self.module._case0_command(log.parent) and
                command == [str(log.parent / "simv"), "-no_save"] and timeout == 300,
                "single case0 command drift")
        token = self.module.PASS_TOKEN
        text = {"missing_pass": "NO_PASS\n", "duplicate_pass": token + "\n" + token + "\n",
                "first_x": token + "\nM1112_FIRST_X controlled\n"}.get(self.mode, token + "\n")
        log.write_text(text, encoding="utf-8")
        return 0


def atomic_case(module, root: Path, mode: str) -> dict[str, Any]:
    secret = "27000@atomic.m1147.invalid"
    key = "SNPSLMD_LICENSE_FILE"
    child = module._child_environment(key, secret)
    route = {"selected_variable": key, "present": True, "byte_length": len(secret.encode()),
             "sha256": hashlib.sha256(secret.encode()).hexdigest()}
    preflight = {"status": "MOCK_PREFLIGHT", "route": route, "lmstat_available": True,
                 "home_key_in_child_environment": False, "dc_attempts": 0,
                 "structural_reset_gate": {"shadow_register_bits": 337,
                    "active_low_clear_nets": 12, "direct_inverter_registers": 75,
                    "buffered_then_inverter_registers": 262, "maximum_chain_cells": 2}}
    runner = FakeRunner(module, secret, child, mode)
    with fake_namespace(module, root, preflight, key, secret, child), \
         patch.object(module, "_run_command", runner), \
         patch.object(subprocess, "Popen", side_effect=HammerFailure("real process forbidden")):
        if mode == "success":
            summary = module._future_execute_once()
            require(module.RESULT.is_dir() and module.ATTEMPT.is_dir() and
                    not module.LOCK.exists() and not list(root.glob(".work.*")) and
                    not list(root.glob("result.failed_or_incomplete.*")), "success not atomic")
            receipt, outer = verify_output(module.RESULT, "receipt.json", secret)
            attempt, _ = verify_output(module.ATTEMPT, "attempt.json", secret)
            require(len(runner.commands) == 2 and receipt["vcs_compile_attempts"] == 1 and
                    receipt["case0_attempts"] == 1 and receipt["window_cycles"] == 128 and
                    receipt["pass_token_count"] == 1 and receipt["dc_attempts"] == 0 and
                    receipt["automatic_retry"] is False and
                    attempt["compile_attempts"] == attempt["case0_attempts"] == 1 and
                    attempt["dc_attempts"] == 0 and attempt["automatic_retry"] is False and
                    summary["outer_seal_file_sha256"] == outer,
                    "success receipt/budget drift")
            rejected("retry_after_success", module._future_execute_once)
            return {"commands": 2, "compile_attempts": 1, "case0_attempts": 1,
                    "window_cycles": 128, "dc_attempts": 0, "atomic_result": True,
                    "retry_rejected": True, "outer": outer}
        rejected(mode, module._future_execute_once)
        require(module.ATTEMPT.is_dir() and not module.RESULT.exists() and
                not module.LOCK.exists() and not list(root.glob(".work.*")),
                "failure not atomically quarantined")
        quarantines = list(root.glob("result.failed_or_incomplete.*.quarantine"))
        require(len(quarantines) == 1, "failure quarantine cardinality")
        failure, outer = verify_output(quarantines[0], "failure.json", secret)
        require(failure["status"] == "FAILED_OR_INCOMPLETE_DO_NOT_CITE" and
                failure["attempt_consumed"] is True and failure["dc_attempts"] == 0 and
                failure["automatic_retry"] is False and
                failure["home_key_in_child_environment"] is False,
                "failure receipt drift")
        rejected(mode + "_retry", module._future_execute_once)
        return {"commands": len(runner.commands), "attempt_consumed": True,
                "quarantines": 1, "retry_rejected": True, "dc_attempts": 0, "outer": outer}


def main() -> None:
    before = verify_inputs()
    module = load_source()
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    main_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"]
    require(len(main_nodes) == 1 and len(main_nodes[0].args.args) == 0 and
            "len(sys.argv) == 1" in ast.unparse(main_nodes[0]) and
            "os.environ.copy" not in text and "automatic_retry" in text,
            "zero-argument/clean-environment/no-retry static boundary")
    command = module._compile_command(Path("/controlled/m1147"))
    require(len(command) == 14 and command == [str(VCS), "-full64", "-sverilog", "+v2k",
            "-timescale=1ns/1ps", "-Mdir=/controlled/m1147/csrc", str(CELL), str(NETLIST),
            str(MEMORY), str(TB), "-top", module.TB_TOP, "-o", "/controlled/m1147/simv"] and
            module._case0_command(Path("/controlled/m1147")) ==
            ["/controlled/m1147/simv", "-no_save"], "frozen 14-argument command drift")
    route = route_and_preflight_hammer(module)
    secret = secret_boundary_hammer(module)
    cases = {}
    with tempfile.TemporaryDirectory(prefix="m1147r6_atomic_") as temp:
        base = Path(temp)
        for mode in ("success", "compile_fail", "missing_simv", "missing_pass",
                     "duplicate_pass", "first_x"):
            root = base / mode
            root.mkdir()
            cases[mode] = atomic_case(module, root, mode)
        collision = base / "collision"
        collision.mkdir()
        (collision / "result").mkdir()
        collision_secret = "27000@collision.m1147.invalid"
        collision_key = "SNPSLMD_LICENSE_FILE"
        collision_child = module._child_environment(collision_key, collision_secret)
        collision_preflight = {"route": {"selected_variable": collision_key, "present": True,
            "byte_length": len(collision_secret.encode()),
            "sha256": hashlib.sha256(collision_secret.encode()).hexdigest()}}
        runner = FakeRunner(module, collision_secret, collision_child, "success")
        with fake_namespace(module, collision, collision_preflight, collision_key,
                            collision_secret, collision_child), patch.object(module, "_run_command", runner):
            rejected("namespace_collision", module._future_execute_once)
            require(not module.ATTEMPT.exists() and not runner.commands,
                    "collision consumed attempt or launched command")
    with patch.object(sys, "argv", [str(SOURCE), "extra"]):
        rejected("nonzero_argument", module.main)
    after = verify_inputs()
    require(before == after and module.namespace_fresh() and not module.ATTEMPT.exists() and
            not module.RESULT.exists(), "real namespace or frozen identity changed")
    result = {
        "schema": "m1147r6_m1146r6_c2_license_route_final_source_hammer_r1_v1",
        "status": "PASS_M1147R6_FINAL_SOURCE_HAMMER__ROOT_EXTERNAL_PREFLIGHT_THEN_ONE_EXACT_LICENSE_ROUTED_MAPPED_VCS_EXECUTION_ONLY",
        "checks": checks, "attacks": attacks, "exact_identities": before,
        "contract_outer_typo_guard": {"accepted": EXPECTED["contract_outer"],
                                       "rejected_prefix": "6632", "passed": True},
        "license_route": route, "secret_boundary": secret, "controlled_mock": cases,
        "command_contract": {"compile_arguments": 14, "compile_attempts": 1,
            "case0_arguments": ["simv", "-no_save"], "case0_attempts": 1,
            "window_cycles": 128, "pass_token_count": 1, "forbidden_first_x": True,
            "dc_attempts": 0, "automatic_retry": False, "new_namespace": True,
            "atomic_result_or_failure": True},
        "production_boundary": {"real_lmstat": 0, "real_vcs": 0, "real_dc": 0,
            "real_subprocess": 0, "attempt_created": False, "result_created": False},
        "authorization": {"root_external_preflight_required": True,
            "one_exact_mapped_vcs_execution_after_preflight": True,
            "automatic_retry": False, "any_second_attempt": False, "dc": False,
            "modify_frozen_inputs": False},
        "claim_boundary": {"mapped_functionality": False, "ppa": False,
            "cycles_speedup": False, "paper_citable": False},
        "source_sha256": EXPECTED["source"],
        "author_outer_seal_file_sha256": EXPECTED["author_outer"],
        "contract_outer_seal_file_sha256": EXPECTED["contract_outer"],
        "docs359_sha256": EXPECTED["docs359"],
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
