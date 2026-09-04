#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent M1142CA hammer; controlled fixtures only, never real M410."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Callable
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/run_m1141ca_c1_production_schedule_release_source.py"
CONTRACT = HW / "contracts/m1141ca_c1_production_schedule_release_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1141ca_c1_production_schedule_release_source_author_receipt_r1_20260830"
M1016 = HW / "system_simulator/scripts/run_m1016_c1_full_matched_address_replay.py"
REAL_ROWS = HW / "results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/m410r2_h67_q32_runtime_rows_32.memh"
REAL_RESULT = HW / "results/m1141ca_c1_production_schedule_release_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    "source": "e2f5d4e0bab472b3a5c7ec5259a805641b800efd3c0e82884e81152eb41cb611",
    "contract": "4fe7ba960516e889cb1f7140315e1e37a5b42dd00337f136b22a25f1c7ac06d4",
    "contract_side": "128d813d63cba813173a5e282dd6f3247ff2f443a5428878d76bef36230d0263",
    "contract_outer": "6e5561e52fab6b4ae3018f8995f4b71f4c8eaeaf02c83ea192421081b5af8184",
    "author_outer": "b5602b120cc7c02769a54e67c78588c481776af9f40f3d3359a2938bf2f8b825",
    "m1016": "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa",
    "rows": "6e03352b89eff1955825334b4dedd991db8c975a9ef6662fe0317e73ccfa8334",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
checks = 0
attacks: dict[str, str] = {}


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
    except BaseException as error:
        if contains is not None:
            require(contains in str(error), f"{label}: wrong rejection {error}")
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
                          HammerFailure("nonfinite JSON " + token)))


def verify_flat(directory: Path, expected_outer: str) -> dict[str, Any]:
    outer = directory / "SHA256SUMS.seal.sha256"
    manifest = directory / "SHA256SUMS"
    regular(outer, expected_outer)
    manifest_sha, manifest_name = outer.read_text(encoding="utf-8").split()
    require(manifest_name == manifest.name and sha(manifest) == manifest_sha,
            "outer seal content drift")
    listed: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        rel = Path(name)
        require(name not in listed and not rel.is_absolute() and ".." not in rel.parts,
                "unsafe/duplicate manifest name")
        listed[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {manifest.name, outer.name}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "sealed symlink")
        if stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), "sealed special member")
    require(actual == set(listed), "sealed exact member set drift")
    for name, digest in listed.items():
        regular(directory / name, digest)
    return strict_json(directory / "review.json")


def production_names() -> tuple[str, ...]:
    parent = REAL_RESULT.parent
    names = []
    if REAL_RESULT.exists() or REAL_RESULT.is_symlink():
        names.append(REAL_RESULT.name)
    names.extend(p.name for p in parent.glob(".m1141ca_c1_production_schedule_release_work.*"))
    names.extend(p.name for p in parent.glob(
        "m1141ca_c1_production_schedule_release_r1_20260830.failed_or_incomplete.*"))
    return tuple(sorted(names))


def load_subject():
    spec = importlib.util.spec_from_file_location("m1142ca_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "subject spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def static_and_identity(module) -> tuple[dict[str, Any], dict[str, Any]]:
    before = production_names()
    require(before == (), "production namespace not empty before hammer")
    regular(SOURCE, EXPECTED["source"])
    regular(CONTRACT, EXPECTED["contract"])
    regular(Path(str(CONTRACT) + ".sha256"), EXPECTED["contract_side"])
    regular(Path(str(CONTRACT) + ".sha256.seal.sha256"), EXPECTED["contract_outer"])
    regular(M1016, EXPECTED["m1016"])
    regular(DOCS359, EXPECTED["docs359"])
    author = verify_flat(AUTHOR, EXPECTED["author_outer"])
    contract = strict_json(CONTRACT)
    require(author["identity"]["source_sha256"] == EXPECTED["source"] and
            author["verdict"].startswith("GO_DIFFERENT_AUTHOR_HAMMER_ONLY") and
            author["authorization"]["production_execution"] is False,
            "author receipt authority drift")
    require(contract["source"]["arguments"] == 0 and
            contract["source"]["automatic_retry"] is False and
            contract["authorization"]["production_execution"] is False and
            contract["production_geometry"]["tasks"] == 812_160 and
            contract["production_geometry"]["records"] == 2_436_480,
            "contract semantics drift")
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    funcs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    require(len(funcs["production_main"].args.args) == 0 and
            len(funcs["main"].args.args) == 0, "zero-argument entry drift")
    prod = ast.unparse(funcs["production_main"])
    main = ast.unparse(funcs["main"])
    execute = ast.unparse(funcs["_execute_release"])
    require("_execute_release(ROWS, ROWS_SHA, ROWS_BYTES, PRODUCTION_GEOMETRY, RESULT)" in prod and
            "len(sys.argv) == 1" in main and "production_main()" in main,
            "hard-bound zero-argument entry drift")
    require("O_NOFOLLOW" in execute and execute.count("os.pread(fd") == 1 and
            "_verify_open_identity(rows, fd, opened, expected_sha)" in execute and
            "_rename_noreplace(stage, result)" in execute and
            "_rename_noreplace(stage, quarantine)" in execute,
            "single-FD/no-follow/atomic path drift")
    recurrence = next(node for node in tree.body if isinstance(node, ast.ClassDef) and
                      node.name == "ExactScheduleRecurrence")
    retained = []
    for node in ast.walk(recurrence):
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.Set)):
            retained.append(type(node).__name__)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                node.func.attr in {"append", "extend", "add"}):
            retained.append(node.func.attr)
    require(retained == [], "O(N) recurrence history primitive")
    # Import plus the explicit self-test must remain non-opening/non-producing.
    static = module.source_static_self_test()
    require(static["canonical_opened"] is False and static["production_records"] == 0 and
            static["production_result_created"] is False and production_names() == before,
            "source self-test escaped boundary")
    # Independent production geometry, not copied from module properties.
    tasks = 10 * 4 * math.ceil(3000 / 64) * 432
    require(tasks == 812_160 and tasks * len(AXES) == 2_436_480 and
            10 * 4 * 432 * 3000 * 9 == 466_560_000,
            "independent production geometry mismatch")
    return contract, {"zero_argument": True, "single_fd": True,
                      "pre_post_inode_sha": True, "o_axes": True,
                      "bounded_rows": 64, "production_tasks": tasks,
                      "production_records": tasks * len(AXES)}


class FakeParent:
    fail_after = -1
    calls = 0

    def parent_cycle_trace(self, masks):
        type(self).calls += 1
        if type(self).fail_after >= 0 and type(self).calls > type(self).fail_after:
            raise RuntimeError("M1142CA controlled midstream stop")
        return tuple(masks)

    @staticmethod
    def parent_summary(trace):
        return {"cycles": sum(value != 0 for value in trace) + 2}


def payload() -> bytes:
    # 128 exact rows => two bounded 64-row tasks. Both one- and multi-parent rows.
    values = [0, 1, 3, 5, 7, 0x10, 0x33, 0xffff]
    rows = [f"{values[index % len(values)]:08x}" for index in range(128)]
    return ("\n".join(rows) + "\n").encode()


def geometry(module):
    return module.Geometry(1, 1, 1, 128, 64, 8, 7, 5, 3)


def quota(total: int, index: int, population: int) -> int:
    return ((index + 1) * total) // population - (index * total) // population


def u64(value: int) -> bytes:
    return value.to_bytes(8, "big")


def independent_source_provenance(module, task: int, raw: bytes,
                                  preprocess: dict[str, int], work: dict[str, int]) -> str:
    payload_parts = [b"M1139CA_PRIOR_TASK\x00\x01", bytes.fromhex(module.M1016_SOURCE_SHA),
                     bytes.fromhex(module.M1102_SOURCE_SHA), u64(task), u64(0), u64(0),
                     u64(task), u64(0), hashlib.sha256(raw).digest()]
    for axis in AXES:
        payload_parts.extend((u64(preprocess[axis]), u64(work[axis])))
    return hashlib.sha256(b"".join(payload_parts)).hexdigest()


def independent_record_provenance(module, record: dict[str, Any]) -> str:
    body = b"".join((
        b"M1139CA_SCHEDULE_RECORD\x00\x01", bytes.fromhex(module.M1016_SOURCE_SHA),
        bytes.fromhex(module.M1102_SOURCE_SHA), bytes.fromhex(module.M1137_SOURCE_SHA),
        bytes((AXES.index(record["axis"]),)), u64(record["task_sequence_ordinal"]),
        u64(record["sample"]), u64(record["operator"]), u64(record["chunk"]),
        u64(record["partition"]), u64(record["requested_cycle_first"]),
        bytes.fromhex(record["source_task_provenance_sha256"])))
    return hashlib.sha256(body).hexdigest()


def independent_expected(module, raw_payload: bytes) -> list[dict[str, Any]]:
    result = []
    states = {axis: {"start": None, "work": 0, "offset": 0} for axis in AXES}
    for task in range(2):
        raw = raw_payload[task * 64 * 9:(task + 1) * 64 * 9]
        masks = [int(line, 16) & 0xffff for line in raw.splitlines()]
        weight = quota(5, task, 2)
        dma = quota(3, task, 2)
        common = max(1, weight, dma, 16)
        capture = 8
        search = sum(mask.bit_count() > 1 for mask in masks)
        frontend = {"candidate": capture + search + 17 * capture + 2,
                    "strongest_zero": 69, "same_coordinate_bit": 10}
        pre = {axis: max(frontend[axis], common) for axis in AXES}
        nnz = sum(mask.bit_count() for mask in masks)
        work = {"candidate": (sum(mask != 0 for mask in masks) + 2) * 8,
                "strongest_zero": nnz * 8, "same_coordinate_bit": nnz * 8}
        source_prov = independent_source_provenance(module, task, raw, pre, work)
        for axis in AXES:
            state = states[axis]
            start = pre[axis] if state["start"] is None else (
                state["start"] + max(state["work"], pre[axis]) + 2)
            requested = state["offset"] + start - pre[axis]
            record = {"axis": axis, "task_sequence_ordinal": task, "sample": 0,
                      "operator": 0, "chunk": task, "partition": 0,
                      "requested_cycle_first": requested,
                      "source_task_provenance_sha256": source_prov}
            record["schedule_record_provenance_sha256"] = independent_record_provenance(
                module, record)
            result.append(record)
            state["start"], state["work"] = start, work[axis]
    return result


def verify_sealed(module, directory: Path, failure: bool = False) -> dict[str, Any]:
    require(directory.is_dir() and not directory.is_symlink(), "sealed directory absent")
    require(stat.S_IMODE(directory.stat().st_mode) == 0o700, "private staging mode drift")
    manifest = directory / module.MANIFEST_NAME
    outer = directory / module.OUTER_NAME
    manifest_sha = sha(manifest)
    require(outer.read_text(encoding="utf-8").split() ==
            [manifest_sha, module.MANIFEST_NAME], "outer seal drift")
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in listed and sha(directory / name) == digest,
                "sealed payload drift")
        listed[name] = digest
    actual = {p.name for p in directory.iterdir()
              if p.name not in {module.MANIFEST_NAME, module.OUTER_NAME}}
    require(actual == set(listed), "sealed exact member mismatch")
    if failure:
        data = strict_json(directory / "failure.json")
        require(data["status"] == "FAILED_OR_INCOMPLETE_DO_NOT_CITE" and
                data["automatic_retry"] is False and "failure.txt" in listed,
                "failure quarantine semantics drift")
        return data
    return strict_json(directory / module.RELEASE_NAME)


def positive(module, root: Path) -> dict[str, Any]:
    rows = root / "bounded_128_rows.memh"
    raw = payload()
    rows.write_bytes(raw)
    result = root / "positive_result"
    FakeParent.calls, FakeParent.fail_after = 0, -1
    opens: list[int] = []
    preads: list[int] = []
    real_open, real_pread = os.open, os.pread

    def tracked_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        try:
            same = Path(path).resolve() == rows.resolve()
        except Exception:
            same = False
        if same:
            opens.append(fd)
            require(flags & os.O_NOFOLLOW and flags & getattr(os, "O_CLOEXEC", 0),
                    "input open lacks NOFOLLOW/CLOEXEC")
        return fd

    def tracked_pread(fd, count, offset):
        if fd in opens:
            preads.append(fd)
        return real_pread(fd, count, offset)

    with patch.object(module, "_load_m1007", return_value=FakeParent()), \
         patch.object(module.os, "open", side_effect=tracked_open), \
         patch.object(module.os, "pread", side_effect=tracked_pread):
        summary = module._execute_release(rows, hashlib.sha256(raw).hexdigest(),
                                          len(raw), geometry(module), result)
    release = verify_sealed(module, result)
    records = [json.loads(line) for line in
               (result / module.RECORDS_NAME).read_text(encoding="utf-8").splitlines()]
    expected = independent_expected(module, raw)
    require(records == expected, "M410+M1016 independent three-axis recurrence mismatch")
    require(len(opens) == 1 and preads and set(preads) == {opens[0]},
            "not a single canonical FD")
    require(summary["records"] == 6 and release["geometry"]["tasks"] == 2 and
            release["geometry"]["records"] == 6 and
            release["records"]["axis_counts"] == {axis: 2 for axis in AXES} and
            release["state_complexity"] == "O(axes) plus one bounded row tile" and
            release["retained_record_or_key_history"] is False,
            "positive release conservation/state drift")
    require(not any(root.glob(".positive_result.private_staging.*")),
            "private staging remains after publish")
    return {"tasks": 2, "records": 6,
            "requested_cycle_first": [r["requested_cycle_first"] for r in records],
            "records_sha256": sha(result / module.RECORDS_NAME),
            "schedule_provenance_sha256": release["records"]["schedule_provenance_sha256"],
            "canonical_open_count": len(opens), "canonical_fd_count": len(set(preads)),
            "result_mode": "0700"}


def sink_attacks(module) -> None:
    raw = payload()[:64 * 9]
    expected = independent_expected(module, payload())[0]
    record = module.ScheduleRecord(**expected)

    def missing():
        sink = module._StreamingRecordSink(io.BytesIO())
        sink(record)
        sink.finalize(1)

    def duplicate():
        sink = module._StreamingRecordSink(io.BytesIO())
        sink(record)
        sink(record)

    def wrong_axis():
        sink = module._StreamingRecordSink(io.BytesIO())
        changed = dict(expected)
        changed["axis"] = AXES[1]
        changed["schedule_record_provenance_sha256"] = independent_record_provenance(
            module, changed)
        sink(module.ScheduleRecord(**changed))

    def wrong_task():
        sink = module._StreamingRecordSink(io.BytesIO())
        changed = dict(expected)
        changed["task_sequence_ordinal"] = 1
        changed["chunk"] = 1
        changed["schedule_record_provenance_sha256"] = independent_record_provenance(
            module, changed)
        sink(module.ScheduleRecord(**changed))

    def provenance():
        sink = module._StreamingRecordSink(io.BytesIO())
        changed = dict(expected)
        changed["schedule_record_provenance_sha256"] = "0" * 64
        sink(module.ScheduleRecord(**changed))

    rejected("missing_record", missing, "conservation")
    rejected("duplicate_record", duplicate, "out of order")
    rejected("out_of_order_axis", wrong_axis, "out of order")
    rejected("out_of_order_task", wrong_task, "out of order")
    rejected("record_provenance_drift", provenance, "provenance")


def run_failure(module, root: Path, label: str, rows: Path, expected_sha: str,
                expected_bytes: int, setup: Callable[[], Any] | None = None,
                error_contains: str | None = None) -> None:
    result = root / (label + "_result")
    before = set(root.iterdir())
    FakeParent.calls, FakeParent.fail_after = 0, -1
    if setup is not None:
        setup()
    rejected(label, lambda: module._execute_release(
        rows, expected_sha, expected_bytes, geometry(module), result), error_contains)
    require(not result.exists() and not result.is_symlink(), label + " published result")
    quarantines = list(root.glob(result.name + ".failed_or_incomplete.*.quarantine"))
    require(len(quarantines) == 1, label + " missing unique quarantine")
    verify_sealed(module, quarantines[0], failure=True)
    require(not any(root.glob("." + result.name + ".private_staging.*")),
            label + " leaked staging")


def file_and_atomic_attacks(module, root: Path) -> None:
    raw = payload()
    good = root / "attack_good.memh"
    good.write_bytes(raw)
    run_failure(module, root, "short_input", root / "short.memh",
                hashlib.sha256(raw).hexdigest(), len(raw),
                lambda: (root / "short.memh").write_bytes(raw[:-9]), "size")
    malformed = bytearray(raw)
    malformed[0] = ord("z")
    run_failure(module, root, "malformed_input", root / "malformed.memh",
                hashlib.sha256(bytes(malformed)).hexdigest(), len(raw),
                lambda: (root / "malformed.memh").write_bytes(malformed), "parse")
    run_failure(module, root, "sha_drift", good, "0" * 64, len(raw), None, "SHA-256")
    target = root / "symlink_target.memh"
    target.write_bytes(raw)
    link = root / "symlink.memh"
    link.symlink_to(target)
    run_failure(module, root, "symlink_input", link, hashlib.sha256(raw).hexdigest(),
                len(raw), None, "not regular")

    # Replacement after open: the open FD remains valid, but the pathname identity changes.
    replace_rows = root / "replace.memh"
    replace_rows.write_bytes(raw)
    original_verify = module._verify_open_identity

    def replace_then_verify(path, fd, opened, expected):
        moved = path.with_name(path.name + ".opened_inode")
        path.rename(moved)
        path.write_bytes(raw)
        return original_verify(path, fd, opened, expected)

    result = root / "path_replacement_result"
    FakeParent.calls, FakeParent.fail_after = 0, -1
    with patch.object(module, "_load_m1007", return_value=FakeParent()), \
         patch.object(module, "_verify_open_identity", side_effect=replace_then_verify):
        rejected("path_replacement", lambda: module._execute_release(
            replace_rows, hashlib.sha256(raw).hexdigest(), len(raw), geometry(module),
            result), "replacement")
    require(not result.exists(), "path replacement published")
    quarantine = list(root.glob(result.name + ".failed_or_incomplete.*.quarantine"))
    require(len(quarantine) == 1, "path replacement quarantine count")
    verify_sealed(module, quarantine[0], failure=True)

    # Midstream stop after one bounded tile: partial records are sealed, never published.
    mid_rows = root / "midstream.memh"
    mid_rows.write_bytes(raw)
    mid_result = root / "midstream_result"
    FakeParent.calls, FakeParent.fail_after = 0, 1
    with patch.object(module, "_load_m1007", return_value=FakeParent()):
        rejected("midstream_failure", lambda: module._execute_release(
            mid_rows, hashlib.sha256(raw).hexdigest(), len(raw), geometry(module),
            mid_result), "controlled midstream")
    require(not mid_result.exists(), "midstream published")
    mid_q = list(root.glob(mid_result.name + ".failed_or_incomplete.*.quarantine"))
    require(len(mid_q) == 1, "midstream quarantine count")
    failure = verify_sealed(module, mid_q[0], failure=True)
    require(failure["phase"] == "STREAM_EXACT_SCHEDULE", "midstream phase drift")

    collision_rows = root / "collision.memh"
    collision_rows.write_bytes(raw)
    collision = root / "collision_result"
    collision.mkdir()
    rejected("result_collision", lambda: module._execute_release(
        collision_rows, hashlib.sha256(raw).hexdigest(), len(raw), geometry(module),
        collision), "collision")
    require(list(root.glob("collision_result.failed_or_incomplete.*")) == [],
            "preflight collision created quarantine/retry")


def main() -> int:
    module = load_subject()
    contract, static = static_and_identity(module)
    sink_attacks(module)
    with tempfile.TemporaryDirectory(prefix="m1142ca_independent_") as name:
        root = Path(name)
        positive_result = positive(module, root)
        file_and_atomic_attacks(module, root)
    require(production_names() == (), "hammer created production namespace")
    require(sha(REAL_ROWS) != "", "UNREACHABLE") if False else None  # never open M410
    require(sha(DOCS359) == EXPECTED["docs359"], "docs359 changed")
    summary = {
        "schema": "m1142ca_independent_schedule_release_hammer_checks_r1_v1",
        "status": "PASS_M1142CA_INDEPENDENT_RELEASE_HAMMER__AUTHOR_ONE_SHOT_PRODUCTION_LAUNCHER_SOURCE_ONLY",
        "checks_passed": checks,
        "attacks_rejected": len(attacks),
        "attacks": attacks,
        "identity": {"source_sha256": sha(SOURCE), "contract_sha256": sha(CONTRACT),
                     "author_receipt_outer_seal_file_sha256": EXPECTED["author_outer"],
                     "m1016_source_sha256": sha(M1016), "m410_expected_sha256": EXPECTED["rows"],
                     "docs359_sha256": sha(DOCS359)},
        "static": static,
        "controlled_positive": positive_result,
        "atomic_failure": {"private_staging_mode": "0700",
                           "publish": "renameat2_RENAME_NOREPLACE",
                           "failure_quarantine_double_sealed": True,
                           "automatic_retry": False},
        "production_boundary": {"m410_open_count": 0, "production_records": 0,
                                "production_result_created": False,
                                "digest_compiler": False, "real_driver": False,
                                "full_replay": False, "eda_gpu_remote": False},
        "authorization": {"one_shot_production_schedule_execution_launcher_source_authoring": True,
                          "production_execution": False, "open_real_m410": False,
                          "digest_compiler_driver_full_eda": False},
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
