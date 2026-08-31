#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1144R6 independent static/controlled-fake hammer; never launches EDA."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
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
SOURCE = HW / "dc_handoff/scripts/run_m1143r6_c2_frozen_netlist_mapped_vcs_successor_source_r1.py"
CONTRACT = HW / "contracts/m1143r6_c2_frozen_netlist_mapped_vcs_successor_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1143r6_c2_frozen_netlist_mapped_vcs_successor_author_receipt_r1_20260830"
FAILURE = HW / ("results/m1133r6_c2_authority_schema_repair_dc_mapped_vcs_r1_"
                "20260830.failed_or_incomplete.1172090.quarantine")
NETLIST = FAILURE / "dc/netlist/m1129r5_c2_k1_async_observation_shadow_wrapper_mapped.v"
CELL = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/"
            "TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/"
            "tcbn28hpcplusbwp35p140.v")
TB = HW / "dc_handoff/tb/tb_m1129r5_c2_k1_async_observation_shadow_case0_short.sv"
MEMORY = HW / "tb_m349/m349_fc2_scalar_bank_memory_model.sv"
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    "source": "d112129e9c068d4b609852fc8e824dd986f6d3f923bf2cf132b3a6ac28298471",
    "contract": "6a9b5124dcc33b7002a17ab15af0f5e6e74b561ae5afc967843372c311511c13",
    "contract_side": "e0ef599110e948317b4db39a82b2177834a398d6f6a9f7b53d78d8de5c618fc2",
    "contract_outer": "003cec4159567311dcf6c0bb1656a343ec2d7b317b02f44df5bcf236a562c63a",
    "author_review": "be22c299cc4727eb9825842f428da6cfb7474ecb3c184e2614b4da406b983d80",
    "author_manifest": "406d7b9f465a5791bfd08379e2e68c747b649250c8a3c13824f2ae318c658669",
    "author_outer": "7845dcb40c198c2ac92eb4324f16cf3a007e02b7112ac974baceb973f7d2cc31",
    "failure_primary": "e0780bf99273c497bba6ecc4d966df54138681715b5072f631922ad199c9b832",
    "failure_manifest": "cbac2199f94723aa39ec3ae2e3b535dfa03e509cedb0b6ac226269b8eab7dd7e",
    "failure_outer": "08ed7238836c58df1d9f6ccf58e530468413df82d18db5a9d3aabce79a1f3455",
    "netlist": "362e855cd3b4391d31dc7a08e5388d9545f289c81d291c512d25294a8539cbc4",
    "cell": "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a",
    "tb": "c08d22d69c222b8c527bdb70cc5b49392c5467bc3142ebc22ec577da6918147b",
    "memory": "4375072b6bd09ada3dc3fd585c12102346ea897192a13630b0c44acf72ff63fa",
    "vcs": "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
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


def sealed_tree(directory: Path, primary: str, identities: tuple[str, str, str]) -> dict[str, Any]:
    manifest = directory / "SHA256SUMS"; outer = directory / "SHA256SUMS.seal.sha256"
    regular(directory / primary, identities[0]); regular(manifest, identities[1]); regular(outer, identities[2])
    require(outer.read_text(encoding="utf-8").split() == [identities[1], "SHA256SUMS"],
            "outer seal content")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*"); rel = Path(name)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and name not in expected and
                name == rel.as_posix() and not rel.is_absolute() and ".." not in rel.parts,
                "unsafe manifest member")
        expected[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}: continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "sealed symlink")
        if stat.S_ISREG(mode): actual.add(name)
        else: require(stat.S_ISDIR(mode), "sealed special member")
    require(actual == set(expected), "sealed exact member census")
    for name, digest in expected.items(): regular(directory / name, digest)
    return strict_json(directory / primary)


def input_identities() -> dict[str, str]:
    for path, key in ((SOURCE, "source"), (CONTRACT, "contract"),
                      (Path(str(CONTRACT) + ".sha256"), "contract_side"),
                      (Path(str(CONTRACT) + ".sha256.seal.sha256"), "contract_outer"),
                      (NETLIST, "netlist"), (CELL, "cell"), (TB, "tb"),
                      (MEMORY, "memory"), (VCS, "vcs"), (DOCS359, "docs359")):
        regular(path, EXPECTED[key])
    author = sealed_tree(AUTHOR, "review.json", (EXPECTED["author_review"],
                         EXPECTED["author_manifest"], EXPECTED["author_outer"]))
    frozen = sealed_tree(FAILURE, "failure.json", (EXPECTED["failure_primary"],
                         EXPECTED["failure_manifest"], EXPECTED["failure_outer"]))
    require(author["status"] ==
            "PASS_M1143R6_SOURCE_STRUCTURAL_337_CONTROLLED_FAKE__DIFFERENT_AUTHOR_HAMMER_ONLY" and
            all(author["authorization"][key] is False
                for key in ("launch", "vcs", "mapped_vcs", "dc", "automatic_retry")),
            "author authorization drift")
    require(frozen["status"] == "FAILED_DIAGNOSTIC_DO_NOT_CITE" and
            frozen["phase"] == "MAPPED_RESET_PROVENANCE_337" and
            frozen["m1133r6_retry"] is False, "frozen failure semantic drift")
    return {"source": sha(SOURCE), "failure_manifest": sha(FAILURE / "SHA256SUMS"),
            "netlist": sha(NETLIST), "cell": sha(CELL), "tb": sha(TB),
            "memory": sha(MEMORY), "vcs": sha(VCS), "docs359": sha(DOCS359),
            "author_outer": sha(AUTHOR / "SHA256SUMS.seal.sha256")}


def load_source():
    spec = importlib.util.spec_from_file_location("m1144r6_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "module spec")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module); return module


def verify_output(directory: Path, primary: str) -> tuple[dict[str, Any], str]:
    manifest = directory / "SHA256SUMS"; outer = directory / "SHA256SUMS.seal.sha256"
    manifest_sha = sha(manifest)
    require(outer.read_text(encoding="utf-8").split() == [manifest_sha, "SHA256SUMS"],
            "fake output seal")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*")
        require(name not in expected and sha(directory / name) == digest, "fake output member")
        expected[name] = digest
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(expected), "fake output exact census")
    return strict_json(directory / primary), sha(outer)


@contextmanager
def namespace(module, root: Path, preflight: dict[str, Any]) -> Iterator[dict[str, Path]]:
    paths = {"RESULTS": root, "RESULT": root / "result", "ATTEMPT": root / ".attempt",
             "LOCK": root / ".lock"}
    with patch.multiple(module, RESULTS=root, RESULT=paths["RESULT"], ATTEMPT=paths["ATTEMPT"],
                        LOCK=paths["LOCK"], WORK_PREFIX=".work.",
                        FAILURE_PREFIX="result.failed_or_incomplete."), \
         patch.object(module, "source_preflight", lambda require_fresh=True: preflight):
        yield paths


class FakeRunner:
    def __init__(self, module, mode: str): self.module = module; self.mode = mode; self.commands = []
    def __call__(self, command, log, timeout, environment):
        self.commands.append(list(command))
        require(environment["VCS_HOME"] == "/opt/synopsys/vcs/V-2023.12-SP1" and
                environment["HOME"] == "/tmp", "environment drift")
        if len(self.commands) == 1:
            require(len(command) == 14 and command == self.module._compile_command(log.parent) and
                    timeout == 1800 and not any(x.lower().startswith(("-sdf", "+sdf")) for x in command),
                    "compile command drift")
            log.write_text("CONTROLLED_FAKE_COMPILE\n", encoding="utf-8")
            if self.mode == "compile_fail": return 1
            if self.mode != "missing_simv":
                (log.parent / "simv").write_text("CONTROLLED_FAKE_SIMV\n", encoding="utf-8")
            return 0
        require(len(self.commands) == 2 and command == [str(log.parent / "simv"), "-no_save"] and
                timeout == 300, "sim command drift")
        token = self.module.PASS_TOKEN
        text = {"missing_pass": "NO_PASS\n", "duplicate_pass": token + "\n" + token + "\n",
                "first_x": token + "\nM1112_FIRST_X controlled\n"}.get(self.mode, token + "\n")
        log.write_text(text, encoding="utf-8"); return 0


def fake_case(module, root: Path, preflight: dict[str, Any], mode: str) -> dict[str, Any]:
    runner = FakeRunner(module, mode)
    with namespace(module, root, preflight), patch.object(module, "_run_command", runner), \
         patch.object(subprocess, "Popen", side_effect=HammerFailure("real subprocess forbidden")):
        if mode == "success":
            summary = module._future_execute_once()
            require(module.RESULT.is_dir() and module.ATTEMPT.is_dir() and
                    not module.LOCK.exists() and not list(root.glob(".work.*")) and
                    not list(root.glob("result.failed_or_incomplete.*")), "success atomic publish")
            receipt, outer = verify_output(module.RESULT, "receipt.json")
            attempt, _ = verify_output(module.ATTEMPT, "attempt.json")
            require(len(runner.commands) == 2 and receipt["vcs_compile_attempts"] == 1 and
                    receipt["case0_attempts"] == 1 and receipt["window_cycles"] == 128 and
                    receipt["pass_token_count"] == 1 and receipt["dc_rerun"] is False and
                    attempt["compile_attempts"] == attempt["case0_attempts"] == 1 and
                    attempt["dc_attempts"] == 0 and attempt["automatic_retry"] is False and
                    summary["outer_seal_file_sha256"] == outer, "success receipt drift")
            return {"commands": 2, "compile": 1, "sim": 1, "dc": 0, "outer": outer}
        rejected(mode, module._future_execute_once)
        require(module.ATTEMPT.is_dir() and not module.RESULT.exists() and
                not module.LOCK.exists() and not list(root.glob(".work.*")), "failure atomic cleanup")
        quarantines = list(root.glob("result.failed_or_incomplete.*.quarantine"))
        require(len(quarantines) == 1, "failure quarantine cardinality")
        failure, outer = verify_output(quarantines[0], "failure.json")
        require(failure["status"] == "FAILED_OR_INCOMPLETE_DO_NOT_CITE" and
                failure["attempt_consumed"] is True and failure["dc_rerun"] is False and
                failure["automatic_retry"] is False, "failure receipt drift")
        rejected(mode + "_retry", module._future_execute_once)
        return {"commands": len(runner.commands), "attempt_consumed": True,
                "quarantines": 1, "retry_rejected": True, "outer": outer}


def main() -> None:
    before = input_identities(); module = load_source()
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    main_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    require(len(main_nodes) == 1 and len(main_nodes[0].args.args) == 0 and
            "len(sys.argv) == 1" in ast.unparse(main_nodes[0]), "zero-argument binding")
    command = module._compile_command(Path("/controlled/fake"))
    require(len(command) == 14 and command == [str(VCS), "-full64", "-sverilog", "+v2k",
            "-timescale=1ns/1ps", "-Mdir=/controlled/fake/csrc", str(CELL), str(NETLIST),
            str(MEMORY), str(TB), "-top", module.TB_TOP, "-o", "/controlled/fake/simv"] and
            module._case0_command(Path("/controlled/fake")) == ["/controlled/fake/simv", "-no_save"],
            "exact 14-argument compile/sim contract")
    real_calls = 0
    def forbid(*_args, **_kwargs):
        nonlocal real_calls; real_calls += 1; raise HammerFailure("real EDA forbidden")
    with patch.object(subprocess, "Popen", forbid): preflight = module.source_preflight(True)
    require(real_calls == 0 and preflight["structural_reset_gate"] == {
        "shadow_register_bits": 337, "active_low_clear_nets": 12,
        "direct_inverter_registers": 75, "buffered_then_inverter_registers": 262,
        "maximum_chain_cells": 2}, "M1141 structural result drift")
    evidence = {}
    with tempfile.TemporaryDirectory(prefix="m1144r6_fake_") as temp:
        base = Path(temp)
        for mode in ("success", "compile_fail", "missing_simv", "missing_pass",
                     "duplicate_pass", "first_x"):
            root = base / mode; root.mkdir(); evidence[mode] = fake_case(module, root, preflight, mode)
        collision = base / "collision"; collision.mkdir(); (collision / "result").mkdir()
        runner = FakeRunner(module, "success")
        with namespace(module, collision, preflight), patch.object(module, "_run_command", runner):
            rejected("result_collision", module._future_execute_once)
            require(not module.ATTEMPT.exists() and not runner.commands, "collision consumed attempt")
    with patch.object(sys, "argv", [str(SOURCE), "extra"]):
        rejected("nonzero_argument", module.main)
    after = input_identities()
    require(before == after and module.namespace_fresh() and not module.ATTEMPT.exists() and
            not module.RESULT.exists(), "real namespace/identity changed")
    result = {
        "schema": "m1144r6_m1143r6_final_source_launch_hammer_r1_v1",
        "status": "PASS_M1144R6_FINAL_SOURCE_LAUNCH_HAMMER__ROOT_EXTERNAL_PREFLIGHT_THEN_ONE_EXACT_MAPPED_VCS_EXECUTION_ONLY",
        "checks": checks, "attacks": attacks, "controlled_fake": evidence,
        "exact_identities": before, "m1141_structural_reset_gate": preflight["structural_reset_gate"],
        "command_contract": {"compile_arguments": 14, "compile_attempts": 1,
                             "simulation_attempts": 1, "simulation_arguments": ["simv", "-no_save"],
                             "window_cycles": 128, "pass_token_count": 1,
                             "forbidden_first_x": True, "dc_attempts": 0, "automatic_retry": False},
        "production_boundary": {"real_vcs": 0, "real_dc": 0, "real_eda": 0,
                                "attempt_created": False, "result_created": False},
        "authorization": {"root_external_preflight_required": True,
                          "one_exact_mapped_vcs_execution_after_preflight": True,
                          "automatic_retry": False, "dc": False, "any_second_attempt": False,
                          "modify_frozen_inputs": False},
        "claim_boundary": {"mapped_functionality": False, "ppa": False,
                           "cycles_speedup": False, "paper_citable": False},
        "source_sha256": EXPECTED["source"], "author_outer_seal_file_sha256": EXPECTED["author_outer"],
        "docs359_sha256": EXPECTED["docs359"],
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__": main()
