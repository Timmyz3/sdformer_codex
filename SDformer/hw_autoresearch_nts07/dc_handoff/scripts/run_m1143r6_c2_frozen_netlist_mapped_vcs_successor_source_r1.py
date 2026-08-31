#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1143R6 frozen-netlist mapped-VCS successor; source only.

This zero-argument runner is not execution-authorized until a different-author
hammer seals it.  Import and source_preflight perform only read-only identity
and structural checks; they never invoke VCS or any EDA process.
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

CHECKER = HW / "dc_handoff/scripts/m1141r6_c2_additive_structural_reset_chain_checker_source_r1.py"
CHECKER_SHA = "86ccd46fdaffcad77444ca105bde1593394dd7643febba1f6a45680bf515965e"
CHECKER_CONTRACT = HW / "contracts/m1141r6_c2_additive_structural_reset_chain_checker_source_contract_r1_20260830.json"
CHECKER_CONTRACT_ID = (
    "60577bc578ca1c9aaa8de5b446f712fc416738a2979e4dc4b86e7ba9b1bf5b37",
    "188ae927bc8ef40085e0e550a29d4892d94647717624487a2d9a4fdb34fa9196",
    "3f4c7cc217fdac94ca061883314bb1b2160a0ee3436ff27f3321a3f70e7a4479",
)
CHECKER_AUTHOR = HW / "reviews/m1141r6_c2_additive_structural_reset_chain_checker_author_receipt_r1_20260830"
CHECKER_AUTHOR_ID = (
    "e1cf31a3e6aef9a57582ec414788b7ef22992bcb1a42902ac1aaa5df7743e7c1",
    "b9d42229914cf11f62f0e76f91378f75b111eda87b055b1d48d1222490923aa6",
    "f47aa96a21b736607c55341569555ce59ad906fd9746125b14035557a3346e97",
)
M1142 = HW / "reviews/m1142r6_m1141r6_c2_structural_reset_chain_checker_hammer_r1_20260830"
M1142_ID = (
    "f3f2b897b579a08bf5e8edda03291ae89499906769356841b7a1153bcc206d4c",
    "1458e531451bcab23735eddc2c38654b8808265ca7aa2690e76d9944cd76cf18",
    "558b2855abd85b147ee18456796fde728623e0f43777438a8194d6de85c6c793",
)

ORIGINAL_ATTEMPT = RESULTS / ".m1133r6_c2_authority_schema_repair_dc_mapped_vcs_attempt_consumed"
ORIGINAL_ATTEMPT_ID = (
    "cfc25412e18d126614768a1a39f38fba101f9a77b556b548bee17f72a13cf317",
    "684c15bdd5a3f4317115b42eeda04ba404e3ddbbe65366dfb0165bd723b38036",
    "83f5e0a0bc5215242c75940e9b2d9560dda999f9fa5e7a467d9a9b006ee9129a",
)
ORIGINAL_FAILURE = RESULTS / ("m1133r6_c2_authority_schema_repair_dc_mapped_vcs_r1_"
                              "20260830.failed_or_incomplete.1172090.quarantine")
ORIGINAL_FAILURE_ID = (
    "e0780bf99273c497bba6ecc4d966df54138681715b5072f631922ad199c9b832",
    "cbac2199f94723aa39ec3ae2e3b535dfa03e509cedb0b6ac226269b8eab7dd7e",
    "08ed7238836c58df1d9f6ccf58e530468413df82d18db5a9d3aabce79a1f3455",
)
ORIGINAL_RESULT = RESULTS / "m1133r6_c2_authority_schema_repair_dc_mapped_vcs_r1_20260830"
NETLIST = ORIGINAL_FAILURE / ("dc/netlist/"
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
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
PASS_TOKEN = ("PASS_M1112_ASYNC_OBSERVATION_SHORT_WINDOW cycles=128 raw_seen=1 "
              "unknown_bitmap=000000 diagnostic_only=1")

RESULT = RESULTS / "m1143r6_c2_frozen_netlist_mapped_vcs_successor_r1_20260830"
ATTEMPT = RESULTS / ".m1143r6_c2_frozen_netlist_mapped_vcs_successor_attempt_consumed"
WORK_PREFIX = ".m1143r6_c2_frozen_netlist_mapped_vcs_successor_work."
FAILURE_PREFIX = RESULT.name + ".failed_or_incomplete."
LOCK = Path("/tmp/m1143r6_c2_frozen_netlist_mapped_vcs_successor.lock")
MANIFEST = "SHA256SUMS"
OUTER = "SHA256SUMS.seal.sha256"


class Failure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON: " + token)))


def verify_regular(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and
            sha256(path) == expected, "identity drift: " + str(path))


def verify_double(path: Path, identity: tuple[str, str, str]) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    verify_regular(path, identity[0]); verify_regular(side, identity[1])
    verify_regular(outer, identity[2])
    require(side.read_text(encoding="utf-8").split() == [identity[0], path.name] and
            outer.read_text(encoding="utf-8").split() == [identity[1], side.name],
            "double seal content drift")


def verify_tree(directory: Path, identity: tuple[str, str, str],
                primary_name: str) -> dict[str, Any]:
    require(directory.is_dir() and not directory.is_symlink(),
            "sealed directory identity drift")
    primary = directory / primary_name
    manifest = directory / MANIFEST
    outer = directory / OUTER
    verify_regular(primary, identity[0]); verify_regular(manifest, identity[1])
    verify_regular(outer, identity[2])
    require(outer.read_text(encoding="utf-8").split() == [identity[1], MANIFEST],
            "sealed outer content drift")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        relative = Path(name)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and
                name not in expected and name == relative.as_posix() and
                not relative.is_absolute() and ".." not in relative.parts,
                "sealed manifest member drift")
        expected[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {MANIFEST, OUTER}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "sealed tree symlink")
        if stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), "sealed tree special member")
    require(actual == set(expected), "sealed exact member census drift")
    for name, digest in expected.items():
        verify_regular(directory / name, digest)
    return strict_json(primary)


def _load_checker():
    verify_regular(CHECKER, CHECKER_SHA)
    spec = importlib.util.spec_from_file_location("m1143r6_frozen_m1141r6", CHECKER)
    require(spec is not None and spec.loader is not None, "checker module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def namespace_fresh(ignore_lock: bool = False) -> bool:
    return (not RESULT.exists() and not RESULT.is_symlink() and
            not ATTEMPT.exists() and not ATTEMPT.is_symlink() and
            (ignore_lock or (not LOCK.exists() and not LOCK.is_symlink())) and
            not any(RESULTS.glob(WORK_PREFIX + "*")) and
            not any(RESULTS.glob(FAILURE_PREFIX + "*")))


def source_preflight(require_fresh: bool = True) -> dict[str, Any]:
    verify_double(CHECKER_CONTRACT, CHECKER_CONTRACT_ID)
    author = verify_tree(CHECKER_AUTHOR, CHECKER_AUTHOR_ID, "review.json")
    hammer = verify_tree(M1142, M1142_ID, "review.json")
    attempt = verify_tree(ORIGINAL_ATTEMPT, ORIGINAL_ATTEMPT_ID, "attempt.json")
    failure = verify_tree(ORIGINAL_FAILURE, ORIGINAL_FAILURE_ID, "failure.json")
    verify_regular(NETLIST, NETLIST_SHA); verify_regular(CELL, CELL_SHA)
    verify_regular(MEMORY, MEMORY_SHA); verify_regular(TB, TB_SHA)
    verify_regular(VCS, VCS_SHA); verify_regular(DOCS359, DOCS359_SHA)
    require(not ORIGINAL_RESULT.exists() and not ORIGINAL_RESULT.is_symlink() and
            len(list(RESULTS.glob(
                "m1133r6_c2_authority_schema_repair_dc_mapped_vcs_r1_20260830."
                "failed_or_incomplete.*"))) == 1,
            "original result/failure namespace drift")
    require(attempt["status"] == "M1133R6_ATTEMPT_CONSUMED_AFTER_M1134R6_M1136R6" and
            attempt["dc_attempts"] == 1 and attempt["mapped_cases"] == 1 and
            failure["status"] == "FAILED_DIAGNOSTIC_DO_NOT_CITE" and
            failure["phase"] == "MAPPED_RESET_PROVENANCE_337" and
            failure["m1133r6_retry"] is False,
            "original attempt/failure semantic drift")
    require(author["status"] ==
            "PASS_M1141R6_ADDITIVE_STRUCTURAL_CHECKER_AUTHOR__BOUNDED_STATIC_ONLY_NO_EDA" and
            hammer["status"] ==
            "PASS_M1142R6_INDEPENDENT_STRUCTURAL_RESET_CHAIN_HAMMER__AUTHOR_ADDITIVE_FROZEN_NETLIST_MAPPED_VCS_SUCCESSOR_SOURCE_ONLY" and
            hammer["authorization"]["mapped_vcs_execution"] is False and
            hammer["authorization"]["dc_retry"] is False,
            "checker authority drift")
    checker = _load_checker()
    checker.source_preflight()
    reset = checker.structural_reset_chain_gate_text(
        NETLIST.read_text(encoding="utf-8", errors="strict"), 337)
    require(reset["shadow_register_bits"] == 337 and
            reset["active_low_clear_nets"] == 12 and
            reset["direct_inverter_registers"] == 75 and
            reset["buffered_then_inverter_registers"] == 262 and
            reset["maximum_chain_cells"] == 2 and
            reset["all_paths_exactly_one_inverter"] is True and
            reset["all_paths_end_rst_core"] is True,
            "337-bit structural reset gate drift")
    require(not list(ORIGINAL_FAILURE.rglob("*.sdf")),
            "unexpected SDF in frozen quarantine")
    if require_fresh:
        require(namespace_fresh(), "M1143R6 namespace not fresh")
    return {
        "status": "PASS_M1143R6_SOURCE_PREFLIGHT__FROZEN_NETLIST_STRUCTURAL_337__NO_VCS_NO_EDA",
        "m1142_outer_seal_file_sha256": M1142_ID[2],
        "original_attempt_outer_seal_file_sha256": ORIGINAL_ATTEMPT_ID[2],
        "original_failure_outer_seal_file_sha256": ORIGINAL_FAILURE_ID[2],
        "mapped_netlist_sha256": NETLIST_SHA,
        "cell_library_sha256": CELL_SHA,
        "checker_sha256": CHECKER_SHA,
        "structural_reset_gate": {
            "shadow_register_bits": 337, "active_low_clear_nets": 12,
            "direct_inverter_registers": 75,
            "buffered_then_inverter_registers": 262,
            "maximum_chain_cells": 2,
        },
        "sdf_mode": "NONE__PRESERVE_ORIGINAL_CASE0_CONTRACT",
        "vcs_executed": False, "eda_executed": False,
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    finally:
        os.close(fd)


def _write_json(path: Path, value: Any) -> None:
    _write_exclusive(path, (json.dumps(value, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n").encode())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _rename_noreplace(source: Path, destination: Path) -> None:
    require(not destination.exists() and not destination.is_symlink(),
            "rename destination collision")
    libc = ctypes.CDLL(None, use_errno=True)
    call = getattr(libc, "renameat2", None)
    require(call is not None, "renameat2 unavailable")
    call.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                     ctypes.c_char_p, ctypes.c_uint]
    call.restype = ctypes.c_int
    if call(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        raise Failure("renameat2 noreplace failed: " + os.strerror(ctypes.get_errno()))


def _seal_tree(directory: Path) -> tuple[str, str]:
    members = []
    for path in directory.rglob("*"):
        if path.name in {MANIFEST, OUTER}:
            continue
        mode = path.lstat().st_mode
        require(not stat.S_ISLNK(mode), "output symlink")
        if stat.S_ISREG(mode):
            members.append(path)
        else:
            require(stat.S_ISDIR(mode), "output special member")
    members.sort(key=lambda path: path.relative_to(directory).as_posix())
    require(members, "empty sealed tree")
    lines = "".join(f"{sha256(path)}  {path.relative_to(directory).as_posix()}\n"
                    for path in members)
    _write_exclusive(directory / MANIFEST, lines.encode())
    manifest_sha = sha256(directory / MANIFEST)
    _write_exclusive(directory / OUTER, f"{manifest_sha}  {MANIFEST}\n".encode())
    _fsync_dir(directory)
    return manifest_sha, sha256(directory / OUTER)


def _run_command(command: list[str], log: Path, timeout: int,
                 environment: dict[str, str]) -> int:
    with log.open("wb") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   cwd=log.parent, env=environment,
                                   start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            raise Failure("command timeout")


def _compile_command(mapped: Path) -> list[str]:
    return [
        str(VCS), "-full64", "-sverilog", "+v2k", "-timescale=1ns/1ps",
        f"-Mdir={mapped / 'csrc'}", str(CELL), str(NETLIST), str(MEMORY),
        str(TB), "-top", TB_TOP, "-o", str(mapped / "simv"),
    ]


def _case0_command(mapped: Path) -> list[str]:
    return [str(mapped / "simv"), "-no_save"]


def _future_execute_once() -> dict[str, Any]:
    """Future one-shot body. Only production_main calls this with hardcoded globals."""
    preflight = source_preflight(require_fresh=True)
    lock_fd = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    attempted = False
    work = RESULTS / (WORK_PREFIX + "%d.%d" % (os.getpid(), time.time_ns()))
    failure = RESULTS / (FAILURE_PREFIX + "%d.%d.quarantine" %
                         (os.getpid(), time.time_ns()))
    phase = "LOCKED_PREFLIGHT"
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(namespace_fresh(ignore_lock=True), "namespace changed under lock")
        phase = "CONSUME_SINGLE_ATTEMPT"
        ATTEMPT.mkdir(mode=0o700)
        _write_json(ATTEMPT / "attempt.json", {
            "schema": "m1143r6_c2_frozen_netlist_mapped_vcs_attempt_r1_v1",
            "status": "M1143R6_SINGLE_ATTEMPT_CONSUMED__NO_RETRY",
            "mapped_netlist_sha256": NETLIST_SHA,
            "m1142_outer_seal_file_sha256": M1142_ID[2],
            "compile_attempts": 1, "case0_attempts": 1,
            "dc_attempts": 0, "automatic_retry": False,
        })
        _seal_tree(ATTEMPT); _fsync_dir(RESULTS); attempted = True
        work.mkdir(mode=0o700)
        phase = "STRUCTURAL_RESET_GATE_337"
        checker = _load_checker()
        reset = checker.structural_reset_chain_gate_text(
            NETLIST.read_text(encoding="utf-8", errors="strict"), 337)
        require(reset["status"] ==
                "PASS_SINGLE_DRIVER_UNARY_CHAIN__EXACTLY_ONE_INVERTER_TO_RST_CORE" and
                reset["shadow_register_bits"] == 337,
                "future structural reset gate failed")
        mapped = work / "mapped_vcs"; mapped.mkdir()
        environment = {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                       "PATH": "/opt/synopsys/vcs/V-2023.12-SP1/bin:/usr/bin:/bin",
                       "VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                       "HOME": "/tmp"}
        phase = "FROZEN_NETLIST_VCS_COMPILE_ONCE"
        compile_command = _compile_command(mapped)
        require(not any(argument.lower().startswith(("-sdf", "+sdf"))
                        for argument in compile_command),
                "SDF option forbidden for original case0 contract")
        compile_rc = _run_command(compile_command, mapped / "compile.log", 1800,
                                  environment)
        require(compile_rc == 0 and (mapped / "simv").is_file(),
                "mapped VCS compile failed")
        phase = "FROZEN_NETLIST_CASE0_128_ONCE"
        case0_rc = _run_command(_case0_command(mapped), mapped / "case0.log", 300,
                                environment)
        case0_text = (mapped / "case0.log").read_text(
            encoding="utf-8", errors="replace")
        require(case0_rc == 0 and case0_text.count(PASS_TOKEN) == 1 and
                "M1112_FIRST_X" not in case0_text,
                "mapped case0 128-cycle contract failed")
        phase = "SEAL_AND_PUBLISH_EXACT_RESULT"
        _write_json(work / "receipt.json", {
            "schema": "m1143r6_c2_frozen_netlist_mapped_vcs_result_r1_v1",
            "status": "PASS_M1143R6_FROZEN_NETLIST_STRUCTURAL_337_MAPPED_CASE0_128__RESULT_HAMMER_REQUIRED",
            "preflight": preflight,
            "mapped_netlist_sha256": NETLIST_SHA,
            "cell_library_sha256": CELL_SHA,
            "tb_sha256": TB_SHA, "memory_model_sha256": MEMORY_SHA,
            "checker_sha256": CHECKER_SHA,
            "structural_reset_gate": {
                "shadow_register_bits": reset["shadow_register_bits"],
                "active_low_clear_nets": reset["active_low_clear_nets"],
                "direct_inverter_registers": reset["direct_inverter_registers"],
                "buffered_then_inverter_registers":
                    reset["buffered_then_inverter_registers"],
            },
            "vcs_compile_attempts": 1, "case0_attempts": 1,
            "window_cycles": 128, "pass_token_count": 1,
            "unknown_bitmap": "000000", "sdf_mode": "NONE",
            "dc_rerun": False, "automatic_retry": False,
            "claim_boundary": {"mapped_functionality_pending_result_hammer": True,
                               "area_timing_power_energy": False,
                               "cycles_speedup": False, "paper_citable": False},
        })
        _write_exclusive(work / "RUN_COMPLETE.txt", (
            "PASS_M1143R6_FROZEN_NETLIST_STRUCTURAL_337_MAPPED_CASE0_128__RESULT_HAMMER_REQUIRED\n"
        ).encode())
        manifest_sha, outer_sha = _seal_tree(work)
        _rename_noreplace(work, RESULT); _fsync_dir(RESULTS)
        return {"status": "PASS_M1143R6_FROZEN_NETLIST_MAPPED_CASE0_128",
                "result": str(RESULT), "manifest_sha256": manifest_sha,
                "outer_seal_file_sha256": outer_sha}
    except BaseException:
        reason = traceback.format_exc()
        if attempted:
            try:
                work.mkdir(mode=0o700, exist_ok=True)
                _write_json(work / "failure.json", {
                    "schema": "m1143r6_c2_frozen_netlist_mapped_vcs_failure_r1_v1",
                    "status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE",
                    "phase": phase, "message": reason,
                    "attempt_consumed": True, "dc_rerun": False,
                    "automatic_retry": False,
                })
                _seal_tree(work); _rename_noreplace(work, failure); _fsync_dir(RESULTS)
            except BaseException:
                pass
        raise
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
            try:
                LOCK.unlink()
            except FileNotFoundError:
                pass


def source_static_self_test() -> dict[str, Any]:
    preflight = source_preflight(require_fresh=True)
    compile_command = _compile_command(Path("/private/fake_mapped"))
    require(len(compile_command) == 14 and compile_command[0] == str(VCS) and
            compile_command[6] == str(CELL) and compile_command[7] == str(NETLIST) and
            compile_command[8] == str(MEMORY) and compile_command[9] == str(TB) and
            compile_command[10:12] == ["-top", TB_TOP] and
            not any(item.lower().startswith(("-sdf", "+sdf"))
                    for item in compile_command),
            "original mapped case0 compile contract drift")
    return {
        "status": "PASS_M1143R6_SOURCE_STATIC_SELF_TEST__STRUCTURAL_337__NO_VCS_NO_EDA",
        "preflight": preflight,
        "future_compile_commands": 1, "future_case0_runs": 1,
        "window_cycles": 128, "sdf_mode": "NONE",
        "production_attempt_created": False, "vcs_executed": False,
        "dc_executed": False, "automatic_retry": False,
    }


def production_main() -> dict[str, Any]:
    return _future_execute_once()


def main() -> int:
    require(len(sys.argv) == 1, "M1143R6 accepts zero arguments")
    print(json.dumps(production_main(), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
