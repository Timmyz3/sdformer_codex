#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author bounded final hammer for the M1148CA launcher source.

This program never opens the production M1141CA JSONL, never invokes the
zero-argument production entry, and never runs production, replay, or EDA.
All execution-path tests use nine bounded synthetic schedule records inside
private temporary namespaces.
"""
from __future__ import annotations

import ast
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
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/run_m1148ca_c1_production_expected_digest_compiler_one_shot_launcher_source.py"
COMPILER = HW / "system_simulator/scripts/build_m1146ca_c1_independent_expected_digest_compiler_source.py"
CONTRACT = HW / "contracts/m1148ca_c1_production_expected_digest_compiler_launcher_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1148ca_c1_production_expected_digest_launcher_author_receipt_r1_20260830"
M1146_AUTHOR = HW / "reviews/m1146ca_c1_independent_expected_digest_compiler_author_receipt_r1_20260830"
M1147_HAMMER = HW / "reviews/m1147ca_m1146ca_c1_independent_expected_digest_compiler_hammer_r1_20260830"
M1141 = HW / "results/m1141ca_c1_production_schedule_release_r1_20260830"
PRODUCTION_JSONL = M1141 / "m1141ca_per_task_schedule_records.jsonl"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUT = HERE

EXPECTED = {
    "source": "00bd132ca162d15b9aab1d5972c1d4da37e1f43288c1973f8708b5322f14a781",
    "compiler": "7b1f5cd2cd4c4bb0a771d0360f8be924d075215e8dd660728a8decac0c886e73",
    "contract": "50ff6a357a4497aa9ee1950ecc7dbebce325e7ec4258d38a7293cd5266164b10",
    "contract_side": "5ddb4a7634a88d972bb44f21ffef756c836483603b19a0b0a2a866e1876122fd",
    "contract_outer": "6543696b35d014879cf89c20e9559d60b6cc7945c6a169ef4ab3283b8a7ad554",
    "author_outer": "f2d0d3888d88b064d8a7d87b5ff04a7689a71101dce51fc6c1fc127a552e7461",
    "m1146_author_outer": "9aa612c53b3d4064f4fb80ac057f936459624cc7a211373664a9fd04c3650414",
    "m1147_hammer_outer": "b18cfb733ae43eb7c07ebf7725b4f0a3de028100b51c51adfa15a0b227072de9",
    "m1141_release": "4c4d264a9ac1e084c8c0acf0a6d150140f95ee96ee967b038ea4c1eefcc2b58c",
    "m1141_manifest": "852b48c0d8098ef69a619925f82a8e1a308e87f2faf9ea76becabf51d52caace",
    "m1141_outer": "0b6549ce38a62bcb22e8a97d0c038860f5698fabc0d9bff162dc6af95d4f043a",
    "m1141_records": "4d4e0e6396ac1061aca7ada142bc2761bf12a785e5373640a28503e3d73a0a81",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
checks = 0
attacks: dict[str, str] = {}
production_jsonl_open_attempts = 0


class HammerFailure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise HammerFailure(message)


def rejected(label: str, action: Callable[[], Any], contains: str | None = None) -> None:
    try:
        action()
    except Exception as error:
        if contains is not None:
            require(contains.lower() in str(error).lower(),
                    f"{label}: wrong rejection: {error}")
        attacks[label] = f"{type(error).__name__}: {error}"
        return
    raise HammerFailure("attack accepted: " + label)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_regular(path: Path, expected: str, expected_uid: int = 1913) -> None:
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and not path.is_symlink() and
            value.st_uid == expected_uid and sha256(path) == expected,
            "identity drift: " + str(path))


def strict_json_bytes(payload: bytes) -> Any:
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    return json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          HammerFailure("nonfinite JSON: " + token)))


def strict_json(path: Path) -> Any:
    return strict_json_bytes(path.read_bytes())


def verify_double(path: Path, identity: tuple[str, str, str]) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    verify_regular(path, identity[0]); verify_regular(side, identity[1])
    verify_regular(outer, identity[2])
    require(side.read_text(encoding="utf-8").split() == [identity[0], path.name],
            "side seal content")
    require(outer.read_text(encoding="utf-8").split() == [identity[1], side.name],
            "outer seal content")


def verify_flat(directory: Path, expected_outer: str) -> dict[str, Any]:
    require(directory.is_dir() and not directory.is_symlink() and
            directory.stat().st_uid == 1913, "receipt directory drift")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    verify_regular(outer, expected_outer)
    outer_parts = outer.read_text(encoding="utf-8").split()
    require(len(outer_parts) == 2 and outer_parts[1] == "SHA256SUMS" and
            re.fullmatch(r"[0-9a-f]{64}", outer_parts[0]) is not None,
            "receipt outer syntax")
    verify_regular(manifest, outer_parts[0])
    listed: dict[str, str] = {}
    for row in manifest.read_text(encoding="utf-8").splitlines():
        fields = row.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]) is not None,
                "receipt manifest row")
        name = fields[1].lstrip("*"); relative = Path(name)
        require(name not in listed and name == relative.as_posix() and
                not relative.is_absolute() and ".." not in relative.parts,
                "receipt manifest member")
        listed[name] = fields[0]
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "receipt symlink")
        if stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), "receipt special member")
    require(actual == set(listed), "receipt exact member set")
    for name, digest in listed.items():
        verify_regular(directory / name, digest)
    return strict_json(directory / "review.json")


def verify_identities() -> dict[str, Any]:
    verify_regular(SOURCE, EXPECTED["source"])
    verify_regular(COMPILER, EXPECTED["compiler"])
    verify_double(CONTRACT, (EXPECTED["contract"], EXPECTED["contract_side"],
                             EXPECTED["contract_outer"]))
    verify_regular(DOCS359, EXPECTED["docs359"])
    author = verify_flat(AUTHOR, EXPECTED["author_outer"])
    m1146 = verify_flat(M1146_AUTHOR, EXPECTED["m1146_author_outer"])
    m1147 = verify_flat(M1147_HAMMER, EXPECTED["m1147_hammer_outer"])
    release = M1141 / "m1141ca_schedule_release.json"
    manifest = M1141 / "SHA256SUMS"
    outer = M1141 / "SHA256SUMS.seal.sha256"
    verify_regular(release, EXPECTED["m1141_release"])
    verify_regular(manifest, EXPECTED["m1141_manifest"])
    verify_regular(outer, EXPECTED["m1141_outer"])
    require(outer.read_text(encoding="utf-8").split() ==
            [EXPECTED["m1141_manifest"], "SHA256SUMS"], "M1141 outer content")
    listed = {}
    for row in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = row.split(maxsplit=1)
        listed[name.lstrip("*")] = digest
    require(listed == {
        "m1141ca_per_task_schedule_records.jsonl": EXPECTED["m1141_records"],
        "m1141ca_schedule_release.json": EXPECTED["m1141_release"],
    }, "M1141 manifest identity")
    record_stat = PRODUCTION_JSONL.lstat()
    require(stat.S_ISREG(record_stat.st_mode) and not PRODUCTION_JSONL.is_symlink() and
            record_stat.st_uid == 1913 and record_stat.st_size == 836_268_740,
            "M1141 records metadata drift")
    release_value = strict_json(release)
    require(release_value["records"]["count"] == 2_436_480 and
            release_value["records"]["sha256"] == EXPECTED["m1141_records"] and
            release_value["geometry"]["axes"] == list(AXES) and
            release_value["geometry"]["tasks"] == 812_160 and
            release_value["retained_record_or_key_history"] is False,
            "M1141 release schema drift")
    require(author["identity"]["launcher_source_sha256"] == EXPECTED["source"] and
            tuple(author["identity"]["launcher_contract_identity"]) ==
                (EXPECTED["contract"], EXPECTED["contract_side"], EXPECTED["contract_outer"]) and
            author["authorization"]["production_execution"] is False,
            "M1148 author boundary drift")
    require(m1146["subject"]["source_sha256"] == EXPECTED["compiler"] and
            m1146["authorization"]["production_digest_compiler_execution"] is False,
            "M1146 identity/boundary drift")
    require(m1147["subject"]["source_sha256"] == EXPECTED["compiler"] and
            m1147["authorization"]["one_shot_production_digest_compiler_launcher_source_next"] is True and
            m1147["authorization"]["production_digest_compiler_execution_by_this_hammer"] is False,
            "M1147 identity/boundary drift")
    return {"author": author, "m1146": m1146, "m1147": m1147,
            "m1141": release_value}


def load_subject():
    spec = importlib.util.spec_from_file_location("m1149ca_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "subject import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source_static_checks() -> dict[str, Any]:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    require("main" in functions and len(functions["main"].args.args) == 0 and
            functions["main"].args.vararg is None and functions["main"].args.kwarg is None,
            "main is not zero argument")
    production = functions["production_main"]
    require(not any(isinstance(node, (ast.For, ast.While, ast.AsyncFor))
                    for node in ast.walk(production)), "retry/loop in production_main")
    attempt_lines = [node.lineno for node in ast.walk(production)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                     node.func.attr == "mkdir" and isinstance(node.func.value, ast.Name) and
                     node.func.value.id == "ATTEMPT"]
    stream_lines = [node.lineno for node in ast.walk(production)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
                    node.func.id == "_compile_stream"]
    require(len(attempt_lines) == 1 and len(stream_lines) == 1 and
            attempt_lines[0] < stream_lines[0], "attempt-before-stream static order")
    main_calls = [node for node in ast.walk(functions["main"])
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
                  node.func.id == "production_main"]
    require(len(main_calls) == 1 and 'if __name__ == "__main__":' in text and
            "automatic_retry\": False" in text and "renameat2" in text,
            "one-shot/atomic static structure")
    require("retained_event_row_or_key_history\": False" in text and
            "O(axes + axes*24)" in text and "expected-digest authority" not in text.lower(),
            "bounded-state/small-output static structure")
    return {"main_zero_arguments": True, "attempt_line": attempt_lines[0],
            "stream_line": stream_lines[0], "production_main_loops": 0}


def bounded_payload(compiler) -> bytes:
    rows = []
    for record in compiler.bounded_schedule_records():
        mapping = {field: getattr(record, field) for field in compiler.SCHEDULE_FIELDS}
        rows.append((json.dumps(mapping, sort_keys=True, separators=(",", ":"),
                                allow_nan=False) + "\n").encode("utf-8"))
    require(len(rows) == 9, "bounded rows")
    return b"".join(rows)


def run_stream_attacks(subject, compiler) -> dict[str, Any]:
    payload = bounded_payload(compiler)
    expected_sha = hashlib.sha256(payload).hexdigest()
    with tempfile.TemporaryDirectory(prefix="m1149ca_stream_") as raw:
        root = Path(raw)
        good = root / "good.jsonl"; good.write_bytes(payload)
        value = subject._compile_stream(good, expected_sha, 9,
                                        compiler.BOUNDED_GEOMETRY, compiler)
        require(value["schedule_records"] == 9 and value["schedule_bytes"] == len(payload) and
                value["schedule_sha256"] == expected_sha and
                value["authority"]["expected_count_by_axis"] == {axis: 8 for axis in AXES} and
                value["authority"]["expected_digest_by_axis"] == compiler.BOUNDED_GOLDEN_DIGESTS,
                "bounded exact stream mismatch")

        rows = payload.splitlines(keepends=True)
        partial = root / "partial.jsonl"; partial.write_bytes(b"".join(rows[:-1]))
        rejected("partial_record_count", lambda: subject._compile_stream(
            partial, hashlib.sha256(partial.read_bytes()).hexdigest(), 8,
            compiler.BOUNDED_GEOMETRY, compiler), "partial")
        reordered = root / "reordered.jsonl"
        reordered.write_bytes(rows[1] + rows[0] + b"".join(rows[2:]))
        rejected("record_order", lambda: subject._compile_stream(
            reordered, hashlib.sha256(reordered.read_bytes()).hexdigest(), 9,
            compiler.BOUNDED_GEOMETRY, compiler), "reorder")
        rejected("stream_sha", lambda: subject._compile_stream(
            good, "0" * 64, 9, compiler.BOUNDED_GEOMETRY, compiler), "identity")
        rejected("expected_record_count", lambda: subject._compile_stream(
            good, expected_sha, 8, compiler.BOUNDED_GEOMETRY, compiler), "identity")
        duplicate = root / "duplicate_key.jsonl"
        duplicate.write_bytes(b'{"axis":"candidate","axis":"candidate"}\n')
        rejected("duplicate_json_key", lambda: subject._compile_stream(
            duplicate, hashlib.sha256(duplicate.read_bytes()).hexdigest(), 1,
            compiler.BOUNDED_GEOMETRY, compiler), "duplicate")
        crlf = root / "crlf.jsonl"; crlf.write_bytes(rows[0][:-1] + b"\r\n")
        rejected("crlf_framing", lambda: subject._compile_stream(
            crlf, hashlib.sha256(crlf.read_bytes()).hexdigest(), 1,
            compiler.BOUNDED_GEOMETRY, compiler), "framing")
        no_newline = root / "no_newline.jsonl"; no_newline.write_bytes(rows[0][:-1])
        rejected("partial_line", lambda: subject._compile_stream(
            no_newline, hashlib.sha256(no_newline.read_bytes()).hexdigest(), 1,
            compiler.BOUNDED_GEOMETRY, compiler), "framing")

    state = compiler.IndependentExpectedDigestCompiler(compiler.BOUNDED_GEOMETRY)
    for record in compiler.bounded_schedule_records():
        state.consume_schedule_record(record)
    snapshot = state.snapshot()
    require(set(snapshot["axes"]) == set(AXES) and
            sum(len(axis["next_free_cycle"]) for axis in snapshot["axes"].values()) == 72 and
            all(len(axis["next_free_cycle"]) == 24 for axis in snapshot["axes"].values()) and
            snapshot["state_complexity"] == "O(axes + axes*24)",
            "state complexity/history drift")
    return {"records": 9, "events": 24, "stream_sha256": expected_sha,
            "scheduler_state_entries": 72,
            "expected_digest_by_axis": value["authority"]["expected_digest_by_axis"]}


class NamespacePatch:
    def __init__(self, subject, root: Path, records: Path, payload_sha: str,
                 payload_bytes: int):
        self.subject = subject
        self.values = {
            "RESULTS": root,
            "RESULT": root / "result",
            "ATTEMPT": root / ".attempt_consumed",
            "LOCK": root / ".launcher.lock",
            "WORK_PREFIX": ".work.",
            "FAILURE_PREFIX": "failure.",
            "M1141_RECORDS": records,
            "M1141_RECORDS_SHA": payload_sha,
            "M1141_RECORDS_BYTES": payload_bytes,
            "EXPECTED_RECORDS": 9,
            "EXPECTED_EVENTS": 24,
        }
        self.old = {}

    def __enter__(self):
        for name, value in self.values.items():
            self.old[name] = getattr(self.subject, name)
            setattr(self.subject, name, value)
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, value in self.old.items():
            setattr(self.subject, name, value)


def sealed_tree_valid(subject, directory: Path) -> None:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and outer.is_file(), "sandbox seal absent")
    manifest_sha, manifest_name = outer.read_text(encoding="utf-8").split()
    require(manifest_name == "SHA256SUMS" and subject.sha256(manifest) == manifest_sha,
            "sandbox outer invalid")
    listed = {}
    for row in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = row.split(maxsplit=1); listed[name.lstrip("*")] = digest
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(listed), "sandbox member set")
    for name, digest in listed.items():
        require(subject.sha256(directory / name) == digest, "sandbox member digest")


def run_atomic_sandboxes(subject, compiler) -> dict[str, Any]:
    payload = bounded_payload(compiler); payload_sha = hashlib.sha256(payload).hexdigest()
    healthy = {"cpus": 8, "mem_available_bytes": 16 << 30,
               "commit_headroom_bytes": 16 << 30, "disk_free_bytes": 16 << 30,
               "same_uid_conflicts": 0}
    original_compile = subject._compile_stream
    with tempfile.TemporaryDirectory(prefix="m1149ca_success_") as raw:
        root = Path(raw); records = root / "bounded.jsonl"; records.write_bytes(payload)
        open_calls = []
        def observed_compile(path, expected_sha, expected_records, geometry, module):
            require(subject.ATTEMPT.is_dir() and
                    (subject.ATTEMPT / "attempt.json").is_file() and
                    (subject.ATTEMPT / "SHA256SUMS.seal.sha256").is_file(),
                    "input opened before durable attempt")
            open_calls.append(str(path))
            return original_compile(path, expected_sha, expected_records, geometry, module)
        with NamespacePatch(subject, root, records, payload_sha, len(payload)), \
                patch.object(subject, "resource_preflight", return_value=healthy), \
                patch.object(subject, "load_compiler", return_value=compiler), \
                patch.object(compiler, "PRODUCTION_GEOMETRY", compiler.BOUNDED_GEOMETRY), \
                patch.object(subject, "_compile_stream", side_effect=observed_compile):
            result = subject.production_main()
            require(result["events_compiled"] == 24 and len(open_calls) == 1 and
                    subject.RESULT.is_dir() and subject.ATTEMPT.is_dir() and
                    not subject.LOCK.exists() and not tuple(root.glob("failure.*")) and
                    not tuple(root.glob(".work.*")), "sandbox success lifecycle")
            sealed_tree_valid(subject, subject.RESULT)
            receipt = strict_json(subject.RESULT / "receipt.json")
            authority = strict_json(subject.RESULT / "expected_digest_authority.json")
            require(receipt["attempt_consumed"] is True and
                    receipt["automatic_retry"] is False and
                    receipt["event_output_written"] is False and
                    authority["expected_digest_by_axis"] == compiler.BOUNDED_GOLDEN_DIGESTS and
                    set(path.name for path in subject.RESULT.iterdir()) == {
                        "RUN_COMPLETE.txt", "SHA256SUMS", "SHA256SUMS.seal.sha256",
                        "expected_digest_authority.json", "receipt.json", "runtime_resources.json"},
                    "sandbox success output/boundary")
            calls_before = len(open_calls)
            rejected("success_no_retry", subject.production_main, "fresh")
            require(len(open_calls) == calls_before, "retry reopened bounded input")

    with tempfile.TemporaryDirectory(prefix="m1149ca_failure_") as raw:
        root = Path(raw); records = root / "bounded.jsonl"; records.write_bytes(payload)
        attempted = []
        def fail_after_attempt(*_args, **_kwargs):
            require(subject.ATTEMPT.is_dir() and
                    (subject.ATTEMPT / "SHA256SUMS.seal.sha256").is_file(),
                    "failure before durable attempt")
            attempted.append(True)
            raise HammerFailure("bounded injected stream failure")
        with NamespacePatch(subject, root, records, payload_sha, len(payload)), \
                patch.object(subject, "resource_preflight", return_value=healthy), \
                patch.object(subject, "load_compiler", return_value=compiler), \
                patch.object(compiler, "PRODUCTION_GEOMETRY", compiler.BOUNDED_GEOMETRY), \
                patch.object(subject, "_compile_stream", side_effect=fail_after_attempt):
            rejected("failure_quarantine", subject.production_main, "bounded injected")
            failures = tuple(root.glob("failure.*"))
            require(len(attempted) == 1 and subject.ATTEMPT.is_dir() and
                    not subject.RESULT.exists() and len(failures) == 1 and
                    not subject.LOCK.exists() and not tuple(root.glob(".work.*")),
                    "sandbox failure lifecycle")
            sealed_tree_valid(subject, failures[0])
            failure = strict_json(failures[0] / "failure.json")
            require(failure["attempt_consumed"] is True and
                    failure["automatic_retry"] is False and
                    failure["status"] == "FAILED_OR_INCOMPLETE_DO_NOT_CITE",
                    "failure receipt boundary")
            calls_before = len(attempted)
            rejected("failure_no_retry", subject.production_main, "fresh")
            require(len(attempted) == calls_before, "failure retry reached stream")

    with tempfile.TemporaryDirectory(prefix="m1149ca_noreplace_") as raw:
        root = Path(raw); source = root / "source"; destination = root / "destination"
        source.write_text("source", encoding="utf-8")
        destination.write_text("destination", encoding="utf-8")
        rejected("atomic_noreplace", lambda: subject._rename_noreplace(source, destination))
        require(source.read_text(encoding="utf-8") == "source" and
                destination.read_text(encoding="utf-8") == "destination",
                "noreplace clobbered path")
    return {"success_stream_calls": 1, "success_retry_stream_calls": 0,
            "failure_stream_calls": 1, "failure_retry_stream_calls": 0,
            "attempt_before_open": True, "success_failure_mutual_exclusion": True,
            "atomic_noreplace": True}


def run_resource_and_collision_gates(subject) -> dict[str, Any]:
    real_results = subject.RESULTS
    subject.RESULTS = HW / "results"
    good_mem = {"MemAvailable": 8 << 30, "CommitLimit": 32 << 30,
                "Committed_AS": 16 << 30}
    good_vfs = SimpleNamespace(f_bavail=8 << 30, f_frsize=1)
    try:
        with patch.object(subject, "_meminfo", return_value=good_mem), \
                patch.object(subject.os, "sched_getaffinity", return_value=set(range(8))), \
                patch.object(subject.os, "statvfs", return_value=good_vfs), \
                patch.object(subject, "_conflicting_processes", return_value=()):
            healthy = subject.resource_preflight()
            require(healthy["cpus"] == 8 and healthy["same_uid_conflicts"] == 0,
                    "healthy resource gate")
        cases = {
            "cpu_gate": ({"affinity": {0, 1, 2}}, good_mem, good_vfs, ()),
            "memory_gate": ({"affinity": set(range(8))},
                            {**good_mem, "MemAvailable": (4 << 30) - 1}, good_vfs, ()),
            "commit_gate": ({"affinity": set(range(8))},
                            {"MemAvailable": 8 << 30, "CommitLimit": 20 << 30,
                             "Committed_AS": (12 << 30) + 1}, good_vfs, ()),
            "disk_gate": ({"affinity": set(range(8))}, good_mem,
                          SimpleNamespace(f_bavail=(2 << 30) - 1, f_frsize=1), ()),
            "same_uid_conflict_gate": ({"affinity": set(range(8))}, good_mem, good_vfs, (424242,)),
        }
        for label, (cpu, mem, vfs, conflicts) in cases.items():
            with patch.object(subject, "_meminfo", return_value=mem), \
                    patch.object(subject.os, "sched_getaffinity", return_value=cpu["affinity"]), \
                    patch.object(subject.os, "statvfs", return_value=vfs), \
                    patch.object(subject, "_conflicting_processes", return_value=conflicts):
                rejected(label, subject.resource_preflight, "resource")

        sleeper = subprocess.Popen([
            "/opt/anaconda3/envs/pytorch310/bin/python3.10", "-c",
            "import time; time.sleep(15)", str(SOURCE),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            found = subject._conflicting_processes()
            require(sleeper.pid in found, "real same-UID argv collision not detected")
        finally:
            sleeper.terminate()
            try:
                sleeper.wait(timeout=3)
            except subprocess.TimeoutExpired:
                sleeper.kill(); sleeper.wait(timeout=3)
    finally:
        subject.RESULTS = real_results
    return {"healthy_gate": True, "cpu_gate": True, "memory_gate": True,
            "commit_gate": True, "disk_gate": True, "same_uid_gate": True,
            "real_same_uid_argv_collision_detected": True}


def monitor_no_production_open(subject, action: Callable[[], Any]) -> Any:
    global production_jsonl_open_attempts
    original_os_open = os.open
    original_path_open = Path.open
    production = str(PRODUCTION_JSONL.resolve())
    def guarded_os_open(path, *args, **kwargs):
        global production_jsonl_open_attempts
        try:
            candidate = str(Path(path).resolve())
        except TypeError:
            candidate = ""
        if candidate == production:
            production_jsonl_open_attempts += 1
            raise HammerFailure("production M1141 JSONL open forbidden")
        return original_os_open(path, *args, **kwargs)
    def guarded_path_open(path_self, *args, **kwargs):
        global production_jsonl_open_attempts
        if str(path_self.resolve()) == production:
            production_jsonl_open_attempts += 1
            raise HammerFailure("production M1141 JSONL Path.open forbidden")
        return original_path_open(path_self, *args, **kwargs)
    with patch.object(os, "open", side_effect=guarded_os_open), \
            patch.object(Path, "open", new=guarded_path_open):
        return action()


def write_outputs(identity: dict[str, Any], static: dict[str, Any],
                  stream: dict[str, Any], atomic: dict[str, Any],
                  resources: dict[str, Any]) -> None:
    review = {
        "schema": "m1149ca_m1148ca_c1_production_expected_digest_launcher_final_hammer_r1_v1",
        "status": "PASS_M1149CA_DIFFERENT_AUTHOR_FINAL_BOUNDED_LAUNCHER_HAMMER__ROOT_EXTERNAL_PREFLIGHT_THEN_ONE_SHOT_PRODUCTION_NEXT",
        "date": "2026-08-30",
        "subject": {
            "source": str(SOURCE.relative_to(HW)),
            "source_sha256": EXPECTED["source"],
            "author_receipt_outer_sha256": EXPECTED["author_outer"],
            "contract_identity": [EXPECTED["contract"], EXPECTED["contract_side"],
                                  EXPECTED["contract_outer"]],
            "m1146_source_sha256": EXPECTED["compiler"],
            "m1146_author_outer_sha256": EXPECTED["m1146_author_outer"],
            "m1147_hammer_outer_sha256": EXPECTED["m1147_hammer_outer"],
            "m1141_release_sha256": EXPECTED["m1141_release"],
            "m1141_manifest_sha256": EXPECTED["m1141_manifest"],
            "m1141_outer_sha256": EXPECTED["m1141_outer"],
            "m1141_records_sha256_expected": EXPECTED["m1141_records"],
            "docs359_sha256": EXPECTED["docs359"],
        },
        "evidence": {
            "checks": checks,
            "attacks_rejected": len(attacks),
            "attack_labels": sorted(attacks),
            "static": static,
            "bounded_stream": stream,
            "atomic_sandboxes": atomic,
            "resource_and_collision_gates": resources,
            "production_schedule_jsonl_open_attempts": production_jsonl_open_attempts,
            "production_events_compiled": 0,
            "production_namespace_mutated": False,
            "bounded_fake_only": True,
        },
        "verdict": {
            "strict_duplicate_key_rejection": True,
            "stream_sha_and_exact_count": True,
            "record_order_and_partial_rejection": True,
            "state_O_axes_plus_72": True,
            "attempt_consumed_before_input_open": True,
            "fresh_namespace_and_no_retry": True,
            "atomic_success_failure_mutual_exclusion": True,
            "same_uid_collision_and_resource_gate": True,
            "zero_argument_entry_not_executed_by_hammer": True,
        },
        "authorization": {
            "root_external_source_preflight_next": True,
            "one_shot_production_execution_after_successful_external_preflight": True,
            "production_execution_by_this_hammer": False,
            "automatic_retry": False,
            "real_producer_replay": False,
            "eda": False,
        },
        "claim_boundary": {
            "source_and_bounded_fake_only": True,
            "production_expected_digest_authority": False,
            "traffic_cycles_energy_speedup": False,
            "paper_citable_performance": False,
            "paper_ppa_ready": False,
        },
    }
    mechanical = {"schema": "m1149ca_launcher_mechanical_checks_r1_v1",
                  "checks": checks, "attacks": attacks,
                  "production_jsonl_open_attempts": production_jsonl_open_attempts}
    (OUT / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True,
                                                 allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "mechanical_checks.json").write_text(json.dumps(
        mechanical, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "review.md").write_text(
        "# M1149CA different-author final bounded launcher hammer\n\n"
        "PASS. The frozen M1148CA source, author receipt, contract triple, M1146/M1147 "
        "authorities, M1141 release metadata, and docs/359 identity are pinned. Independent "
        "bounded fakes reject duplicate keys, wrong SHA/count/order/partial framing, enforce "
        "O(axes+72) state, consume the attempt before opening the bounded input, preserve "
        "atomic success/failure exclusion, reject retry, and enforce resource/same-UID gates.\n\n"
        "This hammer opened the production M1141 JSONL zero times, compiled zero production "
        "events, did not invoke the zero-argument entry, and ran no replay or EDA. It authorizes "
        "only root external source preflight followed, if successful, by the unique one-shot "
        "production execution.\n", encoding="utf-8")
    (OUT / "BOUNDED_FAKE_ONLY_NO_PRODUCTION_JSONL_NO_ENTRY_NO_REPLAY_NO_EDA.txt").write_text(
        "production_schedule_jsonl_opens=0\nproduction_events_compiled=0\n"
        "zero_argument_entry_invocations=0\nreal_replay=0\neda=0\n", encoding="utf-8")
    (OUT / "RUN_COMPLETE.txt").write_text(
        review["status"] + "\n", encoding="utf-8")
    members = sorted(path for path in OUT.iterdir()
                     if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    manifest = "".join(f"{sha256(path)}  {path.name}\n" for path in members)
    (OUT / "SHA256SUMS").write_text(manifest, encoding="utf-8")
    manifest_sha = sha256(OUT / "SHA256SUMS")
    (OUT / "SHA256SUMS.seal.sha256").write_text(
        f"{manifest_sha}  SHA256SUMS\n", encoding="utf-8")


def main() -> int:
    identity = verify_identities()
    static = source_static_checks()
    subject = load_subject()
    compiler = subject.load_compiler()
    def dynamic():
        stream = run_stream_attacks(subject, compiler)
        atomic = run_atomic_sandboxes(subject, compiler)
        resources = run_resource_and_collision_gates(subject)
        return stream, atomic, resources
    stream, atomic, resources = monitor_no_production_open(subject, dynamic)
    require(production_jsonl_open_attempts == 0, "production JSONL open attempted")
    require(subject._namespace_paths() == (), "real production namespace mutated")
    write_outputs(identity, static, stream, atomic, resources)
    print(json.dumps({"status": "PASS", "checks": checks,
                      "attacks_rejected": len(attacks),
                      "production_jsonl_open_attempts": production_jsonl_open_attempts,
                      "outer_seal_file_sha256": sha256(OUT / "SHA256SUMS.seal.sha256")},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
