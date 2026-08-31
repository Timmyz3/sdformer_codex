#!/usr/bin/python3
"""M1292 additive remote-Python compatibility successor to frozen M1257.

Only the real production interpreter identity changes: the frozen remote repo,
candidate/config/checkpoint/profile topology, sealed execution bytes, snapshots,
O_EXCL attempt and no-retry semantics remain M1257-exact.  Import is inert.
Production remains unauthorized until a fresh different-author M1292 hammer.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
from types import ModuleType
from typing import Any, Callable, Sequence


M1257_SOURCE = Path(__file__).with_name(
    "run_m1257_m1253_motion_cross_run_final_checkpoint_binder_one_shot_release_successor.py")
M1257_SOURCE_SHA256 = "ce539d625c0583542dd795a0fdfacff2050c4475995b40371ce599109ce001b6"
M1257_TEST = Path(__file__).parents[1] / "tests/test_run_m1257_m1253_motion_cross_run_final_checkpoint_binder_one_shot_release_successor.py"
M1257_TEST_SHA256 = "2684a84d91cfdc09251d4cec76a10b55ebb811214eba464451994bdb4c179e49"
M1257_CONTRACT = Path(__file__).parents[1] / "contracts/m1257_m1253_motion_cross_run_final_checkpoint_binder_one_shot_release_successor_source_contract_r1_20260830.json"
M1257_CONTRACT_SHA256 = "0a25fe22140a0401d0c13ef37d5ab3d9c16a2f02ab1b9f791d30b4ff013c0a8f"

TARGET_REPO = Path("/root/private_data/work/sdformer_codex/SDformer")
TARGET_INTERPRETER = Path("/usr/bin/python3")
TARGET_PYTHON_VERSION = "3.12.3"

RUNTIME_KEYS = frozenset({
    "interpreter", "version", "os_memfd_create", "os_mfd_allow_sealing",
    "fcntl_add_seals", "fcntl_get_seals", "fcntl_seal_write",
    "fcntl_seal_grow", "fcntl_seal_shrink", "fcntl_seal_seal",
    "sealed_launcher_compiles", "child_stdlib_available",
})
CHILD_STDLIB_MODULES = (
    "argparse", "csv", "dataclasses", "decimal", "hashlib", "importlib.util",
    "json", "math", "os", "pathlib", "stat", "sys", "tempfile", "typing",
    "types",
)
CONTRACT_CLAIM_BOUNDARY = {
    "checkpoint_selected_now": False,
    "hardware_rebind_authorized": False,
    "hardware_speedup": False,
    "system_speedup": False,
    "power_or_energy": False,
    "paper_metric": False,
    "remote_execution_authorized": False,
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_m1257() -> ModuleType:
    payload = M1257_SOURCE.read_bytes()
    if hashlib.sha256(payload).hexdigest() != M1257_SOURCE_SHA256:
        raise RuntimeError("frozen M1257 source SHA drift")
    module = ModuleType("m1292_frozen_m1257")
    module.__file__ = str(M1257_SOURCE); module.__package__ = ""
    sys.modules[module.__name__] = module
    exec(compile(payload, str(M1257_SOURCE), "exec"), module.__dict__)
    return module


M = _load_m1257()


def _exact_str(value: Any, expected: str, label: str) -> None:
    M.B.require(type(value) is str and value == expected, label + " mismatch")


def _exact_bool(value: Any, expected: bool, label: str) -> None:
    M.B.require(type(value) is bool and value is expected, label + " must be exact boolean")


def rebind_interpreter(old: Any) -> Any:
    """Return an M1257 policy with only interpreter and version replaced."""
    base = old.base
    rebound_base = M.B.Policy(
        base.repo, TARGET_INTERPRETER, TARGET_PYTHON_VERSION,
        dict(base.authority_pins), dict(base.execution_pins), dict(base.aux_pins),
        base.review_rel, base.review_manifest_sha256, base.review_outer_sha256,
        base.docs_rel, base.docs_sha256, base.candidates, base.manifest_rel,
        base.output_rel, base.attempt_rel, base.log_rel,
    )
    rebound = M.Policy(
        rebound_base, old.successor_review_rel, old.successor_manifest_sha256,
        old.successor_outer_sha256,
    )
    for field in M.B.Policy.__dataclass_fields__:
        if field in ("interpreter", "python_version"):
            continue
        M.B.require(getattr(rebound.base, field) == getattr(base, field),
                    "non-interpreter policy drift: " + field)
    for field in M.Policy.__dataclass_fields__:
        if field != "base":
            M.B.require(getattr(rebound, field) == getattr(old, field),
                        "outer policy drift: " + field)
    return rebound


PRODUCTION_POLICY = rebind_interpreter(M.PRODUCTION_POLICY)
M.B.require(PRODUCTION_POLICY.base.repo == TARGET_REPO, "remote repository path drift")


@dataclass(frozen=True)
class RuntimeProbe:
    interpreter: str
    version: str
    os_memfd_create: bool
    os_mfd_allow_sealing: bool
    fcntl_add_seals: bool
    fcntl_get_seals: bool
    fcntl_seal_write: bool
    fcntl_seal_grow: bool
    fcntl_seal_shrink: bool
    fcntl_seal_seal: bool
    sealed_launcher_compiles: bool
    child_stdlib_available: bool

    def as_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in RUNTIME_KEYS}


def probe_current_runtime(executable_path: Path, version: str) -> RuntimeProbe:
    launcher_compiles = True
    try:
        compile(M.SEALED_LAUNCHER, "<m1292_sealed_launcher>", "exec")
    except (SyntaxError, ValueError, TypeError):
        launcher_compiles = False
    stdlib = all(importlib.util.find_spec(name) is not None
                 for name in CHILD_STDLIB_MODULES)
    return RuntimeProbe(
        str(executable_path), version,
        hasattr(os, "memfd_create"), hasattr(os, "MFD_ALLOW_SEALING"),
        hasattr(fcntl, "F_ADD_SEALS"), hasattr(fcntl, "F_GET_SEALS"),
        hasattr(fcntl, "F_SEAL_WRITE"), hasattr(fcntl, "F_SEAL_GROW"),
        hasattr(fcntl, "F_SEAL_SHRINK"), hasattr(fcntl, "F_SEAL_SEAL"),
        launcher_compiles, stdlib,
    )


def validate_runtime_probe(value: Any) -> dict[str, Any]:
    observed = value.as_dict() if type(value) is RuntimeProbe else value
    M.B.require(type(observed) is dict and set(observed) == RUNTIME_KEYS,
                "runtime probe exact key mismatch")
    _exact_str(observed["interpreter"], str(TARGET_INTERPRETER),
               "runtime interpreter")
    _exact_str(observed["version"], TARGET_PYTHON_VERSION, "runtime version")
    for key in RUNTIME_KEYS - {"interpreter", "version"}:
        _exact_bool(observed[key], True, "runtime " + key)
    return dict(observed)


def validate_contract_claim_boundary(value: Any) -> None:
    M.B.require(type(value) is dict and set(value) == set(CONTRACT_CLAIM_BOUNDARY),
                "contract claim boundary exact key mismatch")
    for key, expected in CONTRACT_CLAIM_BOUNDARY.items():
        _exact_bool(value[key], expected, "contract claim " + key)


Probe = Callable[[Path, str], RuntimeProbe | dict[str, Any]]


def _validate_identity(executable_path: Path, version: str, cwd: Path,
                       policy: Any) -> None:
    M.B.require(isinstance(executable_path, Path) and
                executable_path == TARGET_INTERPRETER and
                executable_path == policy.base.interpreter,
                "interpreter path mismatch")
    _exact_str(version, TARGET_PYTHON_VERSION, "interpreter version")
    M.B.require(type(cwd) is type(policy.base.repo) and cwd == policy.base.repo,
                "repository cwd mismatch")


def _compile_sealed_children(prepared: Any) -> None:
    names = ("m1241", "m1234", "m1228")
    M.B.require(len(prepared.source_fds) == len(names),
                "sealed child descriptor population mismatch")
    for descriptor, name in zip(prepared.source_fds, names):
        blocks = []
        offset = 0
        while True:
            block = os.pread(descriptor, 1 << 20, offset)
            if not block:
                break
            blocks.append(block); offset += len(block)
        compile(b"".join(blocks), "<sealed_{}>".format(name), "exec")


def prepare(policy: Any, executable_path: Path, version: str, cwd: Path,
            probe: Probe = probe_current_runtime):
    _validate_identity(executable_path, version, cwd, policy)
    validate_runtime_probe(probe(executable_path, version))
    prepared = M.prepare(policy, executable_path, version, cwd)
    try:
        _compile_sealed_children(prepared)
        M.B.require(prepared.command[0] == str(TARGET_INTERPRETER),
                    "child interpreter command drift")
        return prepared
    except Exception:
        prepared.close()
        raise


def verify_receipt(output: Path, prepared: Any) -> dict[str, Any]:
    result = M.verify_receipt(output, prepared)
    claim = result.get("claim_boundary")
    expected = dict(M.B.EXACT_CLAIM_BOUNDARY)
    M.B.require(type(claim) is dict and set(claim) == set(expected),
                "result claim boundary exact key mismatch")
    for key, wanted in expected.items():
        _exact_bool(claim[key], wanted, "result claim " + key)
    return result


Runner = Callable[[Sequence[str], Path, tuple[int, ...]], subprocess.CompletedProcess[str]]


def default_runner(command: Sequence[str], cwd: Path, pass_fds: tuple[int, ...]):
    return subprocess.run(list(command), cwd=cwd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False, pass_fds=pass_fds,
                          env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})


def execute_once(policy: Any, executable_path: Path, version: str, cwd: Path,
                 runner: Runner = default_runner,
                 probe: Probe = probe_current_runtime):
    prepared = prepare(policy, executable_path, version, cwd, probe=probe)
    try:
        M.consume_attempt(prepared)
        completed = runner(prepared.command, policy.base.repo, prepared.source_fds)
        M.publish_log(prepared, completed)
        M.B.require(completed.returncode == 0,
                    "single sealed M1241 child failed after attempt; no retry authorized")
        M.B.require(completed.stdout.count(M.CHILD_TOKEN) == 1,
                    "sealed child terminal stdout mismatch")
        verify_receipt(policy.base.repo / policy.base.output_rel, prepared)
        return completed
    finally:
        prepared.close()


def verify_frozen_authorities() -> None:
    for path, expected, label in (
        (M1257_SOURCE, M1257_SOURCE_SHA256, "M1257 source"),
        (M1257_TEST, M1257_TEST_SHA256, "M1257 test"),
        (M1257_CONTRACT, M1257_CONTRACT_SHA256, "M1257 contract"),
    ):
        mode = path.lstat().st_mode
        M.B.require(stat.S_ISREG(mode) and not path.is_symlink() and
                    _sha(path) == expected, label + " identity drift")


def main() -> int:
    M.B.require(len(sys.argv) == 1, "production M1292 release accepts zero arguments")
    verify_frozen_authorities()
    completed = execute_once(PRODUCTION_POLICY, Path(sys.executable),
                             platform.python_version(), Path.cwd())
    sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    print("PASS_M1292_ONE_SHOT_SELECTION_RECEIPT__FRESH_RESULT_HAMMER_REQUIRED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
