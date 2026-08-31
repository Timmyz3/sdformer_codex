#!/usr/bin/python3.12
"""M1297 additive interpreter-entity/TOCTOU successor to frozen M1292.

The production entry point is zero-argument.  It opens and validates the exact
remote Python entity before any checkpoint snapshot, retains that descriptor,
binds it into the O_EXCL attempt record, and executes via /proc/self/fd rather
than reopening /usr/bin/python3 after the attempt is consumed.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import ModuleType
from typing import Any, Callable, Sequence


M1292_SOURCE = Path(__file__).with_name(
    "run_m1292_m1257_motion_final_checkpoint_binder_remote_python_compat_successor.py")
M1292_SOURCE_SHA256 = "76f04b076cef298799e2899670bf60f4671d2fcb1864cb63ee476e4c8f8c49e9"
TARGET_LINK = Path("/usr/bin/python3")
TARGET_REALPATH = Path("/usr/bin/python3.12")
TARGET_VERSION = "3.12.3"
TARGET_ENTITY = {
    "device": 1048625,
    "inode": 1347357695,
    "mode": 0x81ED,
    "size_bytes": 8020928,
    "mtime_sec": 1774292672,
    "sha256": "e1efa562c2cc2e35521a5c9c9b9939921001ff8ca9708a13ef15ace68cc2ccd7",
    "version": TARGET_VERSION,
    "memfd_and_all_seals": True,
}
ENTITY_KEYS = frozenset(TARGET_ENTITY)
COMPLETE_TOKEN = "PASS_M1297_ONE_SHOT_SELECTION_RECEIPT__FRESH_RESULT_HAMMER_REQUIRED"


def _load_m1292() -> ModuleType:
    payload = M1292_SOURCE.read_bytes()
    if hashlib.sha256(payload).hexdigest() != M1292_SOURCE_SHA256:
        raise RuntimeError("frozen M1292 source SHA drift")
    module = ModuleType("m1297_frozen_m1292")
    module.__file__ = str(M1292_SOURCE); module.__package__ = ""
    sys.modules[module.__name__] = module
    exec(compile(payload, str(M1292_SOURCE), "exec"), module.__dict__)
    return module


M = _load_m1292()


def require(value: bool, message: str) -> None:
    M.M.B.require(value, message)


def sha_fd(descriptor: int) -> str:
    digest = hashlib.sha256(); offset = 0
    while True:
        block = os.pread(descriptor, 1 << 20, offset)
        if not block:
            return digest.hexdigest()
        digest.update(block); offset += len(block)


def entity_from_fd(descriptor: int, version: str,
                   memfd_and_all_seals: bool) -> dict[str, Any]:
    observed = os.fstat(descriptor)
    return {
        "device": observed.st_dev, "inode": observed.st_ino,
        "mode": observed.st_mode, "size_bytes": observed.st_size,
        "mtime_sec": observed.st_mtime_ns // 1_000_000_000,
        "sha256": sha_fd(descriptor), "version": version,
        "memfd_and_all_seals": memfd_and_all_seals,
    }


def validate_entity(observed: Any, expected: Any) -> dict[str, Any]:
    require(type(observed) is dict and type(expected) is dict and
            set(observed) == ENTITY_KEYS and set(expected) == ENTITY_KEYS,
            "interpreter entity exact key drift")
    for key in ("device", "inode", "mode", "size_bytes", "mtime_sec"):
        require(type(observed[key]) is int and type(expected[key]) is int and
                observed[key] == expected[key], "interpreter entity {} drift".format(key))
    for key in ("sha256", "version"):
        require(type(observed[key]) is str and observed[key] == expected[key],
                "interpreter entity {} drift".format(key))
    require(type(observed["memfd_and_all_seals"]) is bool and
            observed["memfd_and_all_seals"] is True and
            type(expected["memfd_and_all_seals"]) is bool and
            expected["memfd_and_all_seals"] is True,
            "interpreter entity capability drift")
    require(stat.S_ISREG(observed["mode"]) and observed["mode"] & 0o111 != 0,
            "interpreter entity must be executable regular file")
    return dict(observed)


FD_PROBE = r'''
import fcntl,importlib.util,json,os,platform
names=("argparse","csv","dataclasses","decimal","hashlib","importlib.util","json","math","os","pathlib","stat","sys","tempfile","typing","types")
keys=("F_ADD_SEALS","F_GET_SEALS","F_SEAL_WRITE","F_SEAL_GROW","F_SEAL_SHRINK","F_SEAL_SEAL")
ok=hasattr(os,"memfd_create") and hasattr(os,"MFD_ALLOW_SEALING") and all(hasattr(fcntl,k) for k in keys) and all(importlib.util.find_spec(n) is not None for n in names)
print(json.dumps({"version":platform.python_version(),"memfd_and_all_seals":bool(ok)},sort_keys=True,separators=(",",":")))
'''


Probe = Callable[[int], dict[str, Any]]


def probe_fd_runtime(descriptor: int) -> dict[str, Any]:
    completed = subprocess.run(
        ["/proc/self/fd/{}".format(descriptor), "-I", "-B", "-c", FD_PROBE],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        pass_fds=(descriptor,), env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
    require(completed.returncode == 0 and completed.stderr == "",
            "fd-bound interpreter runtime probe failed")
    value = json.loads(completed.stdout)
    require(type(value) is dict and set(value) == {"version", "memfd_and_all_seals"},
            "fd-bound runtime probe shape drift")
    return value


@dataclass
class InterpreterHandle:
    logical_path: Path
    real_path: Path
    descriptor: int
    identity: dict[str, Any]

    def close(self) -> None:
        try:
            os.close(self.descriptor)
        except OSError:
            pass


def open_interpreter_entity(logical_path: Path, real_path: Path,
                            expected: dict[str, Any], probe: Probe = probe_fd_runtime) -> InterpreterHandle:
    require(isinstance(logical_path, Path) and isinstance(real_path, Path),
            "interpreter paths must be exact Path")
    link_stat = logical_path.lstat()
    require(stat.S_ISLNK(link_stat.st_mode), "logical interpreter must be symlink")
    require(logical_path.resolve(strict=True) == real_path,
            "logical interpreter realpath drift")
    parent_fd = os.open(real_path.parent,
                        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0))
    try:
        descriptor = os.open(real_path.name,
                             os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
                             dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    try:
        runtime = probe(descriptor)
        require(type(runtime) is dict and set(runtime) == {"version", "memfd_and_all_seals"},
                "interpreter runtime probe exact key drift")
        observed = entity_from_fd(descriptor, runtime["version"],
                                  runtime["memfd_and_all_seals"])
        validate_entity(observed, expected)
        return InterpreterHandle(logical_path, real_path, descriptor, observed)
    except Exception:
        os.close(descriptor)
        raise


def revalidate_entity(handle: InterpreterHandle, expected: dict[str, Any]) -> None:
    validate_entity(entity_from_fd(handle.descriptor, handle.identity["version"],
                                   handle.identity["memfd_and_all_seals"]), expected)
    require(handle.logical_path.resolve(strict=True) == handle.real_path,
            "logical interpreter changed after snapshot")
    fresh = open_interpreter_entity(handle.logical_path, handle.real_path, expected,
                                    probe=lambda _fd: {
                                        "version": handle.identity["version"],
                                        "memfd_and_all_seals": handle.identity["memfd_and_all_seals"]})
    try:
        require(os.fstat(fresh.descriptor).st_dev == os.fstat(handle.descriptor).st_dev and
                os.fstat(fresh.descriptor).st_ino == os.fstat(handle.descriptor).st_ino,
                "logical path no longer names pinned interpreter entity")
    finally:
        fresh.close()


@dataclass
class Prepared:
    inherited: Any
    interpreter: InterpreterHandle
    command: list[str]

    @property
    def policy(self): return self.inherited.policy
    @property
    def snapshots(self): return self.inherited.snapshots
    @property
    def source_fds(self): return self.inherited.source_fds
    @property
    def pass_fds(self): return self.inherited.source_fds + (self.interpreter.descriptor,)
    def close(self) -> None:
        self.inherited.close(); self.interpreter.close()


def rebind_policy(old: Any, interpreter: Path, version: str) -> Any:
    base = old.base
    new_base = M.M.B.Policy(
        base.repo, interpreter, version, dict(base.authority_pins),
        dict(base.execution_pins), dict(base.aux_pins), base.review_rel,
        base.review_manifest_sha256, base.review_outer_sha256, base.docs_rel,
        base.docs_sha256, base.candidates, base.manifest_rel, base.output_rel,
        base.attempt_rel, base.log_rel)
    return M.M.Policy(new_base, old.successor_review_rel,
                      old.successor_manifest_sha256, old.successor_outer_sha256)


PRODUCTION_POLICY = rebind_policy(M.PRODUCTION_POLICY, TARGET_LINK, TARGET_VERSION)


def prepare(policy: Any, cwd: Path, logical_path: Path, real_path: Path,
            expected: dict[str, Any], probe: Probe = probe_fd_runtime) -> Prepared:
    # Entity is opened before inherited artifact/checkpoint snapshots.
    handle = open_interpreter_entity(logical_path, real_path, expected, probe=probe)
    try:
        inherited = M.M.prepare(policy, logical_path, expected["version"], cwd)
        M._compile_sealed_children(inherited)
        command = list(inherited.command)
        command[0] = "/proc/self/fd/{}".format(handle.descriptor)
        return Prepared(inherited, handle, command)
    except Exception:
        handle.close()
        raise


def consume_attempt(prepared: Prepared) -> None:
    attempt = prepared.policy.repo / prepared.policy.attempt_rel
    descriptor = os.open(attempt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        population = {key: M.M._snapshot_identity(prepared.snapshots[key])
                      for key in sorted(prepared.snapshots)}
        body = (
            "M1297_PRODUCTION_BINDER_ATTEMPT_CONSUMED_BEFORE_FD_BOUND_CHILD\n"
            "automatic_retry=false\ninput_snapshot_sha256={}\n"
            "interpreter_entity_sha256={}\ncommand_sha256={}\n".format(
                M.M.B.sha256_bytes(json.dumps(population, sort_keys=True,
                                              separators=(",", ":")).encode()),
                M.M.B.sha256_bytes(json.dumps(prepared.interpreter.identity, sort_keys=True,
                                              separators=(",", ":")).encode()),
                M.M.B.sha256_bytes("\0".join(prepared.command).encode())))
        os.write(descriptor, body.encode()); os.fsync(descriptor)
    finally:
        os.close(descriptor)


Runner = Callable[[Sequence[str], Path, tuple[int, ...]], subprocess.CompletedProcess[str]]


def default_runner(command: Sequence[str], cwd: Path, pass_fds: tuple[int, ...]):
    return subprocess.run(list(command), cwd=cwd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False, pass_fds=pass_fds,
                          env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})


def execute_once(policy: Any, cwd: Path, logical_path: Path, real_path: Path,
                 expected: dict[str, Any], runner: Runner = default_runner,
                 probe: Probe = probe_fd_runtime):
    prepared = prepare(policy, cwd, logical_path, real_path, expected, probe=probe)
    try:
        # Final logical-path and retained-fd identity check occurs immediately
        # before irreversible attempt consumption.  Exec itself stays fd-bound.
        revalidate_entity(prepared.interpreter, expected)
        consume_attempt(prepared)
        completed = runner(prepared.command, policy.base.repo, prepared.pass_fds)
        M.M.publish_log(prepared.inherited, completed)
        require(completed.returncode == 0,
                "single fd-bound child failed after attempt; no retry authorized")
        require(completed.stdout.count(M.M.CHILD_TOKEN) == 1,
                "sealed child terminal stdout mismatch")
        M.verify_receipt(policy.base.repo / policy.base.output_rel, prepared.inherited)
        return completed
    finally:
        prepared.close()


def main() -> int:
    require(len(sys.argv) == 1, "production M1297 release accepts zero arguments")
    M.verify_frozen_authorities()
    completed = execute_once(PRODUCTION_POLICY, Path.cwd(), TARGET_LINK,
                             TARGET_REALPATH, dict(TARGET_ENTITY))
    sys.stdout.write(completed.stdout)
    if completed.stderr: sys.stderr.write(completed.stderr)
    print(COMPLETE_TOKEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
