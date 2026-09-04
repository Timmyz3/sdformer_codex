#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1146R6 license-routed frozen-netlist mapped-VCS successor; source only.

No execution is authorized until a different-author hammer is double sealed.
The license route value is never returned, logged, sealed, or written to disk.
"""
from __future__ import annotations

import ctypes
import fcntl
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
import time
import traceback
from typing import Any

sys.dont_write_bytecode = True
SOURCE_FILE = Path(__file__).resolve()
HW = SOURCE_FILE.parent.parent.parent
RESULTS = HW / "results"
AUTHORITY = HW / "reviews/m1145r6_m1143r6_c2_license_environment_failure_hammer_r1_20260830"
AUTHORITY_ID = (
    "e218d0df7e9dea0df2f178340598f7f25ef52d502a8856623bf46ba24251e0a5",
    "857a07a684b310478104a5ceeaa960d76638c27ad70e8025a644de1ac952b764",
    "9edbc8abd3b47bbec576b35d00602cba5abca01cbee320081f954cca9e820148",
)
CHECKER = HW / "dc_handoff/scripts/m1141r6_c2_additive_structural_reset_chain_checker_source_r1.py"
CHECKER_SHA = "86ccd46fdaffcad77444ca105bde1593394dd7643febba1f6a45680bf515965e"
M1143_SOURCE = HW / "dc_handoff/scripts/run_m1143r6_c2_frozen_netlist_mapped_vcs_successor_source_r1.py"
M1143_SOURCE_SHA = "d112129e9c068d4b609852fc8e824dd986f6d3f923bf2cf132b3a6ac28298471"
M1143_ATTEMPT = RESULTS / ".m1143r6_c2_frozen_netlist_mapped_vcs_successor_attempt_consumed"
M1143_ATTEMPT_ID = (
    "f25e3613e773b2c2ce445e1fae0f9a7e22de9282ad8a3148828fd171e284e653",
    "5f993be14eee3273a7cf2c2e9f6299b97c20ad709506ccb3cc727313d084b170",
    "ebdb51e51cbb7a585a4d4b9bab20e48b2c1c510211de6eed014f0e3a2bdd527d",
)
M1143_FAILURE = RESULTS / ("m1143r6_c2_frozen_netlist_mapped_vcs_successor_r1_20260830."
                           "failed_or_incomplete.1671825.1788051182241746176.quarantine")
M1143_FAILURE_ID = (
    "636a41ad428a6fd0f8eb22663613f89f80c3d7a3622cea28f9717318f4ea33d7",
    "02b200a83083e5e4db5237f0d435257c046e293f08ff69c872a3ad0a97bfa0bb",
    "7607b08d35b8c76116f5b85b30e236e93a5339a2670185df025435909d03c06b",
)
NETLIST = HW / ("results/m1133r6_c2_authority_schema_repair_dc_mapped_vcs_r1_20260830."
                "failed_or_incomplete.1172090.quarantine/dc/netlist/"
                "m1129r5_c2_k1_async_observation_shadow_wrapper_mapped.v")
NETLIST_SHA = "362e855cd3b4391d31dc7a08e5388d9545f289c81d291c512d25294a8539cbc4"
CELL = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/"
            "TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/"
            "tcbn28hpcplusbwp35p140.v")
CELL_SHA = "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a"
MEMORY = HW / "tb_m349/m349_fc2_scalar_bank_memory_model.sv"
MEMORY_SHA = "4375072b6bd09ada3dc3fd585c12102346ea897192a13630b0c44acf72ff63fa"
TB = HW / "dc_handoff/tb/tb_m1129r5_c2_k1_async_observation_shadow_case0_short.sv"
TB_SHA = "c08d22d69c222b8c527bdb70cc5b49392c5467bc3142ebc22ec577da6918147b"
TB_TOP = "tb_m1129r5_c2_k1_async_observation_shadow_case0_short"
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
VCS_SHA = "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287"
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
LMUTIL_SHA = "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
PASS_TOKEN = ("PASS_M1112_ASYNC_OBSERVATION_SHORT_WINDOW cycles=128 raw_seen=1 "
              "unknown_bitmap=000000 diagnostic_only=1")
RESULT = RESULTS / "m1146r6_c2_license_route_frozen_netlist_mapped_vcs_successor_r1_20260830"
ATTEMPT = RESULTS / ".m1146r6_c2_license_route_frozen_netlist_mapped_vcs_successor_attempt_consumed"
WORK_PREFIX = ".m1146r6_c2_license_route_frozen_netlist_mapped_vcs_successor_work."
FAILURE_PREFIX = RESULT.name + ".failed_or_incomplete."
LOCK = Path("/tmp/m1146r6_c2_license_route_frozen_netlist_mapped_vcs_successor.lock")
MANIFEST = "SHA256SUMS"; OUTER = "SHA256SUMS.seal.sha256"
LICENSE_KEYS = ("SNPSLMD_LICENSE_FILE", "LM_LICENSE_FILE")


class Failure(RuntimeError): pass


def require(value: bool, message: str) -> None:
    if not value: raise Failure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def verify_regular(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and sha256(path) == expected,
            "identity drift: " + str(path))


def strict_json(path: Path) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(Failure("nonfinite JSON")))


def verify_tree(directory: Path, identity: tuple[str, str, str], primary: str) -> dict[str, Any]:
    manifest = directory / MANIFEST; outer = directory / OUTER
    verify_regular(directory / primary, identity[0]); verify_regular(manifest, identity[1]); verify_regular(outer, identity[2])
    require(outer.read_text(encoding="utf-8").split() == [identity[1], MANIFEST], "outer seal drift")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*"); rel = Path(name)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) and name not in expected and
                name == rel.as_posix() and not rel.is_absolute() and ".." not in rel.parts,
                "unsafe manifest")
        expected[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {MANIFEST, OUTER}: continue
        mode = member.lstat().st_mode; require(not stat.S_ISLNK(mode), "sealed symlink")
        if stat.S_ISREG(mode): actual.add(name)
        else: require(stat.S_ISDIR(mode), "sealed special member")
    require(actual == set(expected), "sealed exact member census")
    for name, digest in expected.items(): verify_regular(directory / name, digest)
    return strict_json(directory / primary)


def _load_checker():
    verify_regular(CHECKER, CHECKER_SHA)
    spec = importlib.util.spec_from_file_location("m1146r6_checker", CHECKER)
    require(spec is not None and spec.loader is not None, "checker module spec")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module); return module


def _select_license_route(environment: dict[str, str]) -> tuple[str, str, dict[str, Any]]:
    selected = None
    for key in LICENSE_KEYS:
        value = environment.get(key)
        if isinstance(value, str) and value:
            selected = (key, value); break
    require(selected is not None, "no usable Synopsys license route")
    key, value = selected
    encoded = value.encode("utf-8", errors="strict")
    require(b"\x00" not in encoded and b"\n" not in encoded and b"\r" not in encoded,
            "invalid license route control character")
    return key, value, {"selected_variable": key, "present": True,
                        "byte_length": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def _child_environment(key: str, value: str) -> dict[str, str]:
    require(key in LICENSE_KEYS and value, "invalid selected license route")
    result = {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
              "PATH": "/opt/synopsys/vcs/V-2023.12-SP1/bin:/opt/synopsys/scl/2025.03/linux64/bin:/usr/bin:/bin",
              "VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1", key: value}
    require("HOME" not in result and set(result) == {"LANG", "LC_ALL", "PATH", "VCS_HOME", key},
            "clean child environment drift")
    return result


def _run_lmstat(key: str, value: str, environment: dict[str, str]) -> bool:
    require(environment == _child_environment(key, value), "lmstat environment drift")
    process = subprocess.Popen([str(LMUTIL), "lmstat", "-c", value], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, env=environment, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM); process.wait(); return False
    require(value.encode() not in output, "lmstat output echoed secret route")
    return process.returncode == 0


def namespace_fresh(ignore_lock: bool = False) -> bool:
    return (not RESULT.exists() and not RESULT.is_symlink() and not ATTEMPT.exists() and
            not ATTEMPT.is_symlink() and (ignore_lock or (not LOCK.exists() and not LOCK.is_symlink())) and
            not any(RESULTS.glob(WORK_PREFIX + "*")) and not any(RESULTS.glob(FAILURE_PREFIX + "*")))


def source_preflight(require_fresh: bool = True) -> tuple[dict[str, Any], str, str, dict[str, str]]:
    authority = verify_tree(AUTHORITY, AUTHORITY_ID, "review.json")
    prior_attempt = verify_tree(M1143_ATTEMPT, M1143_ATTEMPT_ID, "attempt.json")
    prior_failure = verify_tree(M1143_FAILURE, M1143_FAILURE_ID, "failure.json")
    for path, digest in ((M1143_SOURCE, M1143_SOURCE_SHA), (NETLIST, NETLIST_SHA),
                         (CELL, CELL_SHA), (MEMORY, MEMORY_SHA), (TB, TB_SHA),
                         (VCS, VCS_SHA), (LMUTIL, LMUTIL_SHA), (DOCS359, DOCS359_SHA)):
        verify_regular(path, digest)
    require(authority["status"] ==
            "PASS_M1145R6_M1143R6_LICENSE_ENVIRONMENT_OMISSION_FAILURE__AUTHOR_ADDITIVE_LICENSE_ROUTE_SUCCESSOR_SOURCE_ONLY" and
            prior_attempt["dc_attempts"] == 0 and prior_attempt["compile_attempts"] == 1 and
            prior_attempt["case0_attempts"] == 1 and prior_attempt["automatic_retry"] is False and
            prior_failure["status"] == "FAILED_OR_INCOMPLETE_DO_NOT_CITE" and
            prior_failure["phase"] == "FROZEN_NETLIST_VCS_COMPILE_ONCE" and
            prior_failure["automatic_retry"] is False,
            "M1145/prior attempt semantic drift")
    checker = _load_checker()
    reset = checker.structural_reset_chain_gate_text(NETLIST.read_text(encoding="utf-8"), 337)
    require((reset["shadow_register_bits"], reset["active_low_clear_nets"],
             reset["direct_inverter_registers"], reset["buffered_then_inverter_registers"],
             reset["maximum_chain_cells"]) == (337, 12, 75, 262, 2), "reset oracle drift")
    key, value, route = _select_license_route(dict(os.environ))
    child = _child_environment(key, value)
    require(_run_lmstat(key, value, child), "selected license route lmstat unavailable")
    if require_fresh: require(namespace_fresh(), "M1146R6 namespace not fresh")
    public = {"status": "PASS_M1146R6_LICENSE_ROUTE_PREFLIGHT__NO_VALUE_PERSISTED",
              "authority_outer_seal_file_sha256": AUTHORITY_ID[2],
              "route": route, "lmstat_available": True,
              "structural_reset_gate": {key: reset[key] for key in
                  ("shadow_register_bits", "active_low_clear_nets", "direct_inverter_registers",
                   "buffered_then_inverter_registers", "maximum_chain_cells")},
              "home_key_in_child_environment": False, "dc_attempts": 0}
    require(value not in json.dumps(public, sort_keys=True), "license route leaked into preflight")
    return public, key, value, child


def _write_exclusive(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    finally: os.close(fd)


def _write_json(path: Path, value: Any, secret: str) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    require(secret.encode() not in payload, "license route would enter JSON")
    _write_exclusive(path, payload)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try: os.fsync(fd)
    finally: os.close(fd)


def _rename_noreplace(source: Path, destination: Path) -> None:
    require(not destination.exists() and not destination.is_symlink(), "rename destination collision")
    libc = ctypes.CDLL(None, use_errno=True); call = getattr(libc, "renameat2", None)
    require(call is not None, "renameat2 unavailable")
    call.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    call.restype = ctypes.c_int
    if call(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        raise Failure("renameat2 noreplace failed: " + os.strerror(ctypes.get_errno()))


def _seal_tree(directory: Path, secret: str) -> tuple[str, str]:
    members = []
    for path in directory.rglob("*"):
        if path.name in {MANIFEST, OUTER}: continue
        mode = path.lstat().st_mode; require(not stat.S_ISLNK(mode), "output symlink")
        if stat.S_ISREG(mode):
            require(secret.encode() not in path.read_bytes(), "license route leaked into sealed member")
            members.append(path)
        else: require(stat.S_ISDIR(mode), "output special member")
    members.sort(key=lambda path: path.relative_to(directory).as_posix())
    lines = "".join(f"{sha256(path)}  {path.relative_to(directory).as_posix()}\n" for path in members)
    _write_exclusive(directory / MANIFEST, lines.encode()); manifest_sha = sha256(directory / MANIFEST)
    _write_exclusive(directory / OUTER, f"{manifest_sha}  {MANIFEST}\n".encode()); _fsync_dir(directory)
    return manifest_sha, sha256(directory / OUTER)


def _compile_command(mapped: Path) -> list[str]:
    return [str(VCS), "-full64", "-sverilog", "+v2k", "-timescale=1ns/1ps",
            f"-Mdir={mapped / 'csrc'}", str(CELL), str(NETLIST), str(MEMORY), str(TB),
            "-top", TB_TOP, "-o", str(mapped / "simv")]


def _case0_command(mapped: Path) -> list[str]: return [str(mapped / "simv"), "-no_save"]


def _run_command(command: list[str], log: Path, timeout: int,
                 environment: dict[str, str], secret: str) -> int:
    require("HOME" not in environment and secret and secret in environment.values(),
            "child environment/license route drift")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               cwd=log.parent, env=environment, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
        raise Failure("command timeout")
    redacted = output.replace(secret.encode(), b"<REDACTED_LICENSE_ROUTE>")
    _write_exclusive(log, redacted)
    return process.returncode


def _future_execute_once() -> dict[str, Any]:
    preflight, key, secret, child = source_preflight(True)
    lock_fd = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    attempted = False; work = RESULTS / (WORK_PREFIX + f"{os.getpid()}.{time.time_ns()}")
    failure = RESULTS / (FAILURE_PREFIX + f"{os.getpid()}.{time.time_ns()}.quarantine")
    phase = "LOCKED_PREFLIGHT"
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(namespace_fresh(ignore_lock=True), "namespace changed under lock")
        phase = "CONSUME_SINGLE_ATTEMPT"; ATTEMPT.mkdir(mode=0o700)
        _write_json(ATTEMPT / "attempt.json", {"status": "M1146R6_SINGLE_ATTEMPT_CONSUMED__NO_RETRY",
            "compile_attempts": 1, "case0_attempts": 1, "dc_attempts": 0,
            "automatic_retry": False, "license_route": preflight["route"],
            "home_key_in_child_environment": False}, secret)
        _seal_tree(ATTEMPT, secret); _fsync_dir(RESULTS); attempted = True
        work.mkdir(mode=0o700); mapped = work / "mapped_vcs"; mapped.mkdir()
        phase = "FROZEN_NETLIST_VCS_COMPILE_ONCE"
        compile_rc = _run_command(_compile_command(mapped), mapped / "compile.log", 1800, child, secret)
        require(compile_rc == 0 and (mapped / "simv").is_file(), "mapped VCS compile failed")
        phase = "FROZEN_NETLIST_CASE0_128_ONCE"
        case0_rc = _run_command(_case0_command(mapped), mapped / "case0.log", 300, child, secret)
        case0_text = (mapped / "case0.log").read_text(encoding="utf-8", errors="replace")
        require(case0_rc == 0 and case0_text.count(PASS_TOKEN) == 1 and
                "M1112_FIRST_X" not in case0_text, "mapped case0 contract failed")
        phase = "SEAL_AND_PUBLISH_EXACT_RESULT"
        _write_json(work / "receipt.json", {"status": "PASS_M1146R6_LICENSE_ROUTE_FROZEN_NETLIST_MAPPED_CASE0_128__RESULT_HAMMER_REQUIRED",
            "preflight": preflight, "license_route": preflight["route"],
            "vcs_compile_attempts": 1, "case0_attempts": 1, "window_cycles": 128,
            "pass_token_count": 1, "dc_attempts": 0, "automatic_retry": False,
            "home_key_in_child_environment": False, "mapped_functionality_pending_result_hammer": True}, secret)
        _write_exclusive(work / "RUN_COMPLETE.txt",
                         b"PASS_M1146R6_LICENSE_ROUTE_FROZEN_NETLIST_MAPPED_CASE0_128__RESULT_HAMMER_REQUIRED\n")
        manifest, outer = _seal_tree(work, secret); _rename_noreplace(work, RESULT); _fsync_dir(RESULTS)
        return {"status": "PASS_M1146R6_LICENSE_ROUTE_FROZEN_NETLIST_MAPPED_CASE0_128",
                "result": str(RESULT), "manifest_sha256": manifest, "outer_seal_file_sha256": outer}
    except BaseException:
        reason = traceback.format_exc().replace(secret, "<REDACTED_LICENSE_ROUTE>")
        if attempted:
            try:
                work.mkdir(mode=0o700, exist_ok=True)
                _write_json(work / "failure.json", {"status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE",
                    "phase": phase, "message": reason, "attempt_consumed": True,
                    "license_route": preflight["route"], "dc_attempts": 0,
                    "automatic_retry": False, "home_key_in_child_environment": False}, secret)
                _seal_tree(work, secret); _rename_noreplace(work, failure); _fsync_dir(RESULTS)
            except BaseException: pass
        raise
    finally:
        try: fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
            try: LOCK.unlink()
            except FileNotFoundError: pass


def production_main() -> dict[str, Any]: return _future_execute_once()


def main() -> int:
    require(len(sys.argv) == 1, "M1146R6 accepts zero arguments")
    print(json.dumps(production_main(), sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
