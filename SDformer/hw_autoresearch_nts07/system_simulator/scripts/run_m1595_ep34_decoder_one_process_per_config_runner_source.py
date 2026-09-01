#!/usr/bin/env python3
"""M1595 source-only D0/call0 one-process-per-config runner.

No execution is authorized by this source or its author receipt.  A future
independent release may call ``run_once``.  That path consumes a global attempt
before launching three sequential child interpreters, one admitted non-product
configuration per child.  Unit tests use only an injected synthetic launcher;
they never call the exact M1583 worker or open the decoder payload.
"""
from __future__ import print_function

import argparse
import ctypes
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time


SOURCE_FILE = Path(__file__).resolve()
HERE = SOURCE_FILE.parent
HW = HERE.parent.parent
RESULTS = HW / "results"
ENGINE = HERE / "build_m1583_ep34_decoder_one_process_one_config_source.py"
ENGINE_SHA256 = "f92c91f0a6f3a3d79e53ec232fee339ead72edcf14d22a2d51e6f9e86e3f48c4"
M1592 = HW / "reviews/m1592_m1583_decoder_one_process_one_config_engineering_qa_r1_20260901"
M1592_REVIEW_SHA256 = "e2a46df1db6b13ed7dff801427cecb77cd00b0331e6120a976706db32a57fe80"
M1592_MANIFEST_SHA256 = "ba4192f11aa531c19401da7bbb6a75f82d2cb53577fd2cadbefb5c45295d883a"
M1592_OUTER_SHA256 = "f3b0805ccc50c391d541e934a5615f753bf6e589d0651d44c66e39547cc02ef8"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
PYTHON_SHA256 = "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"

SCHEMA = "m1595_ep34_decoder_one_process_per_config_runner_source_r1_v1"
STATUS = "SOURCE_ONLY__ONE_FRESH_PROCESS_PER_CONFIG__INDEPENDENT_HAMMER_REQUIRED__NO_ACTUAL"
CONFIGS = ("DENSE_TYPED_K8", "BIT_EQUAL_SERVICE_K1X8", "BIT_TYPED_K8")
FORBIDDEN_CONFIG = "PRODUCT_CAPTURE_TYPED_K8"
RESOURCE_SHA256 = "64661d825ee8ddbdccad9c3e09ca5e41c5ea9cfc75bcea394667dcfd91b4de10"
RSS_LIMIT_KIB = 8 * 1024 * 1024
CHILD_TIMEOUT_SECONDS = 172800
MANIFEST = "SHA256SUMS"
OUTER = "SHA256SUMS.seal.sha256"
CHILD_ENV_BASE = {
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "TZ": "UTC",
}


class M1595Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1595Error(message)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def canonical_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def canonical_sha(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def strict_json(path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           M1595Error("nonfinite JSON: " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def verify_tree(directory, review_sha, manifest_sha, outer_sha):
    review = directory / "review.json"
    manifest = directory / MANIFEST
    outer = directory / OUTER
    regular_exact(review, review_sha, "M1592 review")
    regular_exact(manifest, manifest_sha, "M1592 manifest")
    regular_exact(outer, outer_sha, "M1592 outer")
    require(outer.read_text(encoding="ascii").split() ==
            [manifest_sha, MANIFEST], "M1592 outer content drift")
    expected = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, name = line.split(None, 1)
        name = name.strip()
        rel = Path(name)
        require(name not in expected and name == rel.as_posix() and
                not rel.is_absolute() and ".." not in rel.parts,
                "M1592 manifest row invalid")
        expected[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in (MANIFEST, OUTER):
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "M1592 symlink")
        if stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), "M1592 special member")
    require(actual == set(expected), "M1592 manifest set drift")
    for name, digest in expected.items():
        regular_exact(directory / name, digest, "M1592 member " + name)
    return strict_json(review)


def load_engine():
    regular_exact(ENGINE, ENGINE_SHA256, "M1583 engine")
    spec = importlib.util.spec_from_file_location("m1595_exact_m1583", str(ENGINE))
    require(spec is not None and spec.loader is not None, "cannot import M1583")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(tuple(module.CONFIGS) == CONFIGS and
            module.FORBIDDEN_CONFIG == FORBIDDEN_CONFIG and
            module.RESOURCE_SHA256 == RESOURCE_SHA256 and
            module.RSS_LIMIT_KIB == RSS_LIMIT_KIB,
            "M1583 boundary drift")
    return module


M = load_engine()


class Layout(object):
    __slots__ = ("result", "attempt", "work", "failure", "lock")

    def __init__(self, result, attempt, work, failure, lock):
        self.result = Path(result)
        self.attempt = Path(attempt)
        self.work = Path(work)
        self.failure = Path(failure)
        self.lock = Path(lock)


PRODUCTION_LAYOUT = Layout(
    RESULTS / "m1595_ep34_decoder_one_process_per_config_r1_20260901",
    RESULTS / ".m1595_ep34_decoder_one_process_per_config_attempt_consumed",
    RESULTS / ".m1595_ep34_decoder_one_process_per_config_work",
    RESULTS / "m1595_ep34_decoder_one_process_per_config_r1_20260901.failed_or_incomplete",
    Path("/tmp/m1595_ep34_decoder_one_process_per_config.lock"),
)


def verify_authorities():
    regular_exact(DOCS359, DOCS359_SHA256, "docs359")
    regular_exact(PYTHON, PYTHON_SHA256, "Python runtime")
    review = verify_tree(M1592, M1592_REVIEW_SHA256,
                         M1592_MANIFEST_SHA256, M1592_OUTER_SHA256)
    require(review.get("status") ==
            "PASS_M1592_M1583_SOURCE_ENGINEERING_QA__INDEPENDENT_PROCESS_RUNNER_SOURCE_AUTHORING_AUTHORIZED__ACTUAL_NOT_AUTHORIZED" and
            review.get("authorization", {}).get(
                "independent_process_runner_source_authoring") is True and
            review.get("authorization", {}).get("actual_execution") is False,
            "M1592 authority drift")
    description = M.describe()
    require(description["status"] == M.STATUS and
            description["fresh_interpreter_per_configuration"] is True and
            description["one_call_token_consumed_before_payload"] is True and
            description["claim_boundary"]["actual_execution"] is False,
            "M1583 description drift")
    return {"m1592_review_sha256": M1592_REVIEW_SHA256,
            "m1583_source_sha256": ENGINE_SHA256,
            "docs359_sha256": DOCS359_SHA256,
            "python_sha256": PYTHON_SHA256}


def layout_collisions(layout):
    return tuple(str(path) for path in
                 (layout.result, layout.attempt, layout.work, layout.failure)
                 if path.exists() or path.is_symlink())


def preflight(layout=PRODUCTION_LAYOUT):
    authority = verify_authorities()
    require(layout.result.parent.is_dir() and
            not layout.result.parent.is_symlink(), "result parent invalid")
    require(layout_collisions(layout) == (), "attempt namespace not fresh")
    return {"schema": SCHEMA,
            "status": "PASS_M1595_RUNNER_SOURCE_PREFLIGHT__NO_ACTUAL",
            "authority": authority,
            "configurations": list(CONFIGS),
            "pilot": {"decoder_stage": "D0", "call_ordinal": 0,
                      "module_ordinal": 0, "timesteps": 10},
            "attempt_consumed": False, "child_processes": 0,
            "actual_execution": False}


def write_new(path, value):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(path), flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def seal_tree(directory):
    members = []
    for member in directory.rglob("*"):
        if member.name in (MANIFEST, OUTER):
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "result symlink")
        if stat.S_ISREG(mode):
            members.append(member)
        else:
            require(stat.S_ISDIR(mode), "result special member")
    members.sort(key=lambda path: path.relative_to(directory).as_posix())
    require(members, "empty result")
    manifest = "".join("{}  {}\n".format(
        sha256(path), path.relative_to(directory).as_posix()) for path in members)
    (directory / MANIFEST).write_text(manifest, encoding="ascii")
    (directory / OUTER).write_text(
        "{}  {}\n".format(sha256(directory / MANIFEST), MANIFEST),
        encoding="ascii")


def rename_noreplace(source, destination):
    require(not destination.exists() and not destination.is_symlink(),
            "publish destination collision")
    library = ctypes.CDLL(None, use_errno=True)
    call = getattr(library, "renameat2", None)
    require(call is not None, "renameat2 unavailable")
    call.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                     ctypes.c_char_p, ctypes.c_uint]
    call.restype = ctypes.c_int
    result = call(-100, os.fsencode(str(source)), -100,
                  os.fsencode(str(destination)), 1)
    require(result == 0, "renameat2 noreplace failed")


def child_ticket(parent_nonce, config, target, parent_pid):
    raw = "\0".join((str(parent_nonce), str(config), str(Path(target).resolve()),
                     str(int(parent_pid)), ENGINE_SHA256, sha256(SOURCE_FILE)))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_child_envelope(config, envelope, parent_pid, expected_ticket):
    require(type(envelope) is dict and envelope.get("schema") ==
            "m1595_ep34_decoder_child_result_r1_v1",
            "child envelope schema drift")
    require(envelope.get("configuration") == config and
            envelope.get("parent_pid") == int(parent_pid) and
            type(envelope.get("child_pid")) is int and
            envelope["child_pid"] > 1 and
            envelope["child_pid"] != int(parent_pid) and
            envelope.get("ticket_sha256") == expected_ticket and
            envelope.get("m1583_source_sha256") == ENGINE_SHA256,
            "child process/identity drift")
    row = envelope.get("result")
    validated = M.validate_result(config, row)
    require(canonical_sha(validated) == envelope.get("result_sha256") and
            canonical_sha(row) == envelope.get("result_sha256"),
            "child result digest drift")
    return dict(row)


def child_main(config, target):
    require(config in CONFIGS and config != FORBIDDEN_CONFIG,
            "child configuration forbidden")
    parent_pid = int(os.environ.get("M1595_PARENT_PID", "0"))
    parent_nonce = os.environ.get("M1595_PARENT_NONCE", "")
    expected_ticket = os.environ.get("M1595_CHILD_TICKET", "")
    require(parent_pid > 1 and os.getppid() == parent_pid and
            len(parent_nonce) == 64 and len(expected_ticket) == 64,
            "child parent authority absent")
    observed_ticket = child_ticket(parent_nonce, config, target, parent_pid)
    require(observed_ticket == expected_ticket, "child ticket mismatch")
    target = Path(target)
    require(target.parent.is_dir() and not target.parent.is_symlink() and
            not target.exists() and not target.is_symlink(),
            "child target invalid")
    row = M.one_shot_worker_entry(config)
    validated = M.validate_result(config, row)
    envelope = {"schema": "m1595_ep34_decoder_child_result_r1_v1",
        "configuration": config, "parent_pid": parent_pid,
        "child_pid": os.getpid(), "ticket_sha256": observed_ticket,
        "m1583_source_sha256": ENGINE_SHA256,
        "result_sha256": canonical_sha(validated), "result": validated}
    write_new(target, envelope)
    return 0


def launch_real_child(config, target, parent_pid, parent_nonce):
    ticket = child_ticket(parent_nonce, config, target, parent_pid)
    environment = dict(CHILD_ENV_BASE)
    environment["M1595_PARENT_PID"] = str(int(parent_pid))
    environment["M1595_PARENT_NONCE"] = parent_nonce
    environment["M1595_CHILD_TICKET"] = ticket
    command = [str(PYTHON), str(SOURCE_FILE), "--child-config", config,
               "--child-output", str(Path(target).resolve())]
    completed = subprocess.run(command, env=environment, cwd=str(HW.parent),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=CHILD_TIMEOUT_SECONDS)
    require(completed.returncode == 0, "fresh child failed for " + config)
    require(Path(target).is_file() and not Path(target).is_symlink(),
            "fresh child result absent")
    return strict_json(target)


def execute_controlled(layout, launcher):
    """Control plane used by the future run and synthetic unit tests.

    ``launcher`` must create the requested child envelope.  Production passes
    ``launch_real_child``; author tests pass a no-payload in-memory witness.
    """
    layout.lock.parent.mkdir(parents=True, exist_ok=True)
    with layout.lock.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise M1595Error("another M1595 attempt holds the lock") from error
        before = preflight(layout)
        parent_pid = os.getpid()
        write_new(layout.attempt, {"schema": SCHEMA,
            "status": "ATTEMPT_CONSUMED_BEFORE_CHILD",
            "attempt_consumed": True, "automatic_retry": False,
            "parent_pid": parent_pid, "configurations": list(CONFIGS),
            "started_unix": time.time()})
        layout.work.mkdir()
        rows = []
        child_pids = []
        ticket_hashes = []
        try:
            for ordinal, config in enumerate(CONFIGS):
                target = layout.work / ("child_%d_%s.json" % (ordinal, config))
                parent_nonce = os.urandom(32).hex()
                expected_ticket = child_ticket(parent_nonce, config, target,
                                               parent_pid)
                envelope = launcher(config, target, parent_pid, parent_nonce)
                row = verify_child_envelope(config, envelope, parent_pid,
                                            expected_ticket)
                rows.append(row)
                child_pids.append(envelope["child_pid"])
                ticket_hashes.append(envelope["ticket_sha256"])
            require(len(set(child_pids)) == len(CONFIGS),
                    "configuration children did not use distinct processes")
            require(len(set(ticket_hashes)) == len(CONFIGS),
                    "configuration child tickets not unique")
            require(len(set(row["resource_manifest_sha256"] for row in rows)) == 1 and
                    rows[0]["resource_manifest_sha256"] == RESOURCE_SHA256,
                    "cross-config resource identity drift")
            require(len(set(row["commit_sequence_sha256"] for row in rows)) == 1,
                    "cross-config commit sequence drift")
            result = {"schema": "m1595_ep34_decoder_one_process_per_config_result_r1_v1",
                "status": "PASS_M1595_D0_CALL0_THREE_PROCESS_DIAGNOSTIC__INDEPENDENT_RESULT_HAMMER_REQUIRED",
                "identity": before["authority"],
                "population": {"decoder_stage": "D0", "call_ordinal": 0,
                    "module_ordinal": 0, "timesteps": 10,
                    "configurations": list(CONFIGS),
                    "fresh_child_processes": len(child_pids)},
                "child_pids": child_pids, "results": rows,
                "attempt_consumed": True, "automatic_retry": False,
                "claim_boundary": {"diagnostic_only": True,
                    "paper_citable_performance": False,
                    "system_speedup": False, "energy": False,
                    "rtl": False, "eda": False, "production": False}}
            write_new(layout.work / "result.json", result)
            write_new(layout.work / "RUN_COMPLETE.json",
                      {"status": result["status"]})
            seal_tree(layout.work)
            rename_noreplace(layout.work, layout.result)
            return result
        except Exception as error:
            if layout.work.is_dir():
                failure_path = layout.work / "FAILED_OR_INCOMPLETE.json"
                if not failure_path.exists():
                    write_new(failure_path, {"schema": SCHEMA,
                        "status": "FAILED_OR_INCOMPLETE",
                        "attempt_consumed": True, "automatic_retry": False,
                        "completed_configurations": len(rows),
                        "exception_type": type(error).__name__,
                        "exception": str(error)})
                seal_tree(layout.work)
                if not layout.failure.exists():
                    rename_noreplace(layout.work, layout.failure)
            raise


def run_once():
    """Future release entry. Current M1595 authoring does not authorize it."""
    return execute_controlled(PRODUCTION_LAYOUT, launch_real_child)


def describe():
    return {"schema": SCHEMA, "status": STATUS,
        "population": {"decoder_stage": "D0", "call_ordinal": 0,
            "module_ordinal": 0, "timesteps": 10,
            "configurations": list(CONFIGS)},
        "process_model": {"fresh_python_per_configuration": True,
            "sequential_children": True, "parent_pid_distinct": True,
            "unique_child_pids_required": True,
            "one_m1583_call_per_child": True},
        "attempt": {"consumed_before_first_child": True,
            "failure_permanent": True, "automatic_retry": False},
        "result": {"config_resource_identity": True,
            "request_conservation": True, "digest_binding": True,
            "rss_gate_calls_positive": True,
            "strict_rss_limit_kib": RSS_LIMIT_KIB,
            "child_result_passthrough": True},
        "execution": {"attempt_consumed": False, "child_processes": 0,
            "actual": False, "payload": False, "gpu": False, "eda": False},
        "next_gate": "different-author runner source hammer before actual",
        "claim_boundary": {"source_only": True, "cycles": False,
            "traffic": False, "speedup": False, "system_speedup": False,
            "energy": False, "rtl": False, "eda": False,
            "paper_result": False}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--child-config", choices=CONFIGS)
    parser.add_argument("--child-output", type=Path)
    args = parser.parse_args(argv)
    if args.child_config is not None:
        require(args.child_output is not None and not args.describe and
                not args.preflight and not args.run,
                "child invocation shape invalid")
        return child_main(args.child_config, args.child_output)
    require(args.child_output is None, "parent mode accepts no child output")
    if args.describe:
        value = describe()
    elif args.preflight:
        value = preflight()
    else:
        run_once()
        return 0
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
