#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Fresh-author bounded hammer for the M1161CA production replay driver.

This checker never opens the 836 MB production JSONL and never calls the
production entry point.  It checks the sealed dependency chain, attacks the
bounded streaming protocol, and emits only a source-authorization review.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import stat
import sys
import tracemalloc
from typing import Any

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HW / "system_simulator/scripts/run_m1161ca_c1_production_real_replay_driver_one_shot_source.py"
SOURCE_SHA = "d7ffb8dbab289e83fd8a32f4ed5244cd005a4b6d0785b586df932fd6a97ee20d"
M1135 = HW / "system_simulator/scripts/build_m1135c_c1_oaxes_streaming_weight_validator_sink_source.py"
M1135_SHA = "4c282b4ece5705b5c8dcd039c29003c14e544ffef5e8c4234afab0ac31ac7571"
M1137 = HW / "system_simulator/scripts/build_m1137c_c1_real_per_task_weight_beat_hook_source.py"
M1137_SHA = "9ec640ae8c9fa75f9cbf706e15d2d26a4233def77e5be4d67e94c084347b20a6"
CONTRACT = HW / "contracts/m1161ca_c1_production_real_replay_driver_source_contract_r1_20260830.json"
CONTRACT_ID = (
    "93471a51d5f9d9270ece1629688b10b0cf88047abed9a5e7b6e71048cd63ef63",
    "89345e94816a72f3672920d4eb9c984afa085789fc47213ef8c981b824f437ea",
    "5c7fdc73e9a69211fea340fa6c9862d19531df551176aa0351f6c914a2f12272",
)
AUTHOR = HW / "reviews/m1161ca_c1_production_real_replay_driver_author_receipt_r1_20260830"
AUTHOR_ID = (
    "7d2dbd0f7019f7bf9f462bf9e5fb0575313a896b29d9bda7a673d6699a4b763c",
    "b6361e95b5e4f16414e923a0c1b56928028c81102a3be051670e0da4988f97bf",
    "9d3e6dbad63761090eb60e06fb4dfa220690a3651643e26c96f9948ec10f71f4",
)
M1138 = HW / "reviews/m1138c_m1137c_c1_real_per_task_weight_beat_hook_hammer_r1_20260830"
M1138_ID = (
    "83356f85ce1d7a3be950d50fc226dd193b1c19e537c6764d94bd07cb6d9fe41a",
    "67bb65e27418fb83657e815cc4ef95d190d9e09c69d2d86cb1306bae4e9c2c39",
    "f55db3e6daed3f10c44e60caea81e419af36db08f71ca164b076eac7baea72fc",
)
M1141 = HW / "results/m1141ca_c1_production_schedule_release_r1_20260830"
M1141_RECORDS = M1141 / "m1141ca_per_task_schedule_records.jsonl"
M1141_RECORDS_SHA = "4d4e0e6396ac1061aca7ada142bc2761bf12a785e5373640a28503e3d73a0a81"
M1141_RECORDS_BYTES = 836_268_740
M1141_RELEASE_SHA = "4c4d264a9ac1e084c8c0acf0a6d150140f95ee96ee967b038ea4c1eefcc2b58c"
M1141_MANIFEST_SHA = "852b48c0d8098ef69a619925f82a8e1a308e87f2faf9ea76becabf51d52caace"
M1141_OUTER_SHA = "0b6549ce38a62bcb22e8a97d0c038860f5698fabc0d9bff162dc6af95d4f043a"
M1145 = HW / "reviews/m1145ca_m1143ca_c1_production_result_hammer_r1_20260830"
M1145_ID = (
    "cfe7bf030743c4bcc098d267c69422ae1e76238696902e8dc601ea8143ee208d",
    "7dbc93256d915962a4f83e860aff9aac0bb3b62b1c76113509daef74b852eb4c",
    "8dcc8e84ec8c6273f155c418078fac92b96ef851768ec6cb2066c64ab3d3423e",
)
M1148 = HW / "results/m1148ca_c1_production_expected_digest_compiler_r1_20260830"
M1148_AUTHORITY = M1148 / "expected_digest_authority.json"
M1148_AUTHORITY_SHA = "c45fd835db7fddca268a8891051a5d24bf9492806c6e3610b8e52b8730e705b2"
M1148_MANIFEST_SHA = "6fc0048c84409cc7afc114f540ad17c83a2a00d0d1db19b0684881d8f2dadf5f"
M1148_OUTER_SHA = "98d69e2799af300b2babe72ac3cceb97f3ecc9a435ac7d12c6c7b8fdd13979d1"
M1157 = HW / "reviews/m1157ca_m1148ca_c1_production_expected_digest_result_hammer_r1_20260830"
M1157_ID = (
    "495fcca0bc853a993eb413d64d64b169838928b9e571291b7d4906e343150417",
    "d5a5b568134cc9bba013b6e501e48c04aecb25bf988b229ebeda0509c14c3280",
    "0dde25832d4af29f983bf6e9aa4573de55835677f668541f2076a531e2b913ee",
)
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


class HammerFailure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise HammerFailure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json_bytes(payload: bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          HammerFailure("nonfinite JSON: " + token)))


def strict_json(path: Path) -> Any:
    return strict_json_bytes(path.read_bytes())


def verify_regular(path: Path, expected: str) -> None:
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and not path.is_symlink() and
            sha256(path) == expected, "identity drift: " + str(path))


def manifest_rows(directory: Path, manifest_sha: str,
                  outer_sha: str, skip_content: set[str] | None = None) -> dict[str, str]:
    skip_content = skip_content or set()
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    verify_regular(manifest, manifest_sha)
    verify_regular(outer, outer_sha)
    require(outer.read_text(encoding="utf-8").split() ==
            [manifest_sha, "SHA256SUMS"], "outer content drift")
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]),
                "manifest row syntax")
        name = fields[1].lstrip("*")
        rel = Path(name)
        require(name not in rows and name == rel.as_posix() and
                not rel.is_absolute() and ".." not in rel.parts,
                "manifest member syntax")
        rows[name] = fields[0]
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
    require(actual == set(rows), "sealed exact member set")
    for name, expected in rows.items():
        if name not in skip_content:
            verify_regular(directory / name, expected)
    return rows


def verify_tree(directory: Path, identity: tuple[str, str, str]) -> dict[str, Any]:
    require(directory.is_dir() and not directory.is_symlink(),
            "authority directory drift")
    rows = manifest_rows(directory, identity[1], identity[2])
    require(rows.get("review.json") == identity[0], "review identity drift")
    return strict_json(directory / "review.json")


def verify_double(path: Path, identity: tuple[str, str, str]) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    verify_regular(path, identity[0]); verify_regular(side, identity[1])
    verify_regular(outer, identity[2])
    require(side.read_text(encoding="utf-8").split() == [identity[0], path.name] and
            outer.read_text(encoding="utf-8").split() == [identity[1], side.name],
            "double seal content drift")


def load_subject():
    verify_regular(SOURCE, SOURCE_SHA)
    spec = importlib.util.spec_from_file_location("m1164_frozen_m1161ca", SOURCE)
    require(spec is not None and spec.loader is not None, "subject import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixture_driver(module):
    m1137 = module.load_m1137()
    sink = module.OAxesRowReceiptSink()
    hook = m1137.M1016SuccessorPerTaskWeightBeatHook(m1137.bounded_authority(), sink)
    return module.ScheduleReplayDriver(hook, "bounded_synthetic", 2), sink


def expect_replay_reject(module, payload: bytes, expected_records: int,
                         expected_sha: str | None = None) -> str:
    driver, _ = fixture_driver(module)
    import io
    try:
        module._parse_and_replay(
            io.BytesIO(payload), expected_records, len(payload),
            expected_sha or hashlib.sha256(payload).hexdigest(), driver)
    except BaseException as error:
        return type(error).__name__ + ": " + str(error)
    raise HammerFailure("attacked replay escaped")


def main() -> None:
    # Independently verify the whole pinned authority chain.  The only skipped
    # payload is the production JSONL itself: metadata and its sealed digest are
    # checked, but opening it is prohibited in this hammer.
    verify_regular(DOCS359, DOCS359_SHA)
    verify_regular(M1135, M1135_SHA); verify_regular(M1137, M1137_SHA)
    verify_double(CONTRACT, CONTRACT_ID)
    author = verify_tree(AUTHOR, AUTHOR_ID)
    m1138 = verify_tree(M1138, M1138_ID)
    m1145 = verify_tree(M1145, M1145_ID)
    m1157 = verify_tree(M1157, M1157_ID)
    rows1141 = manifest_rows(M1141, M1141_MANIFEST_SHA, M1141_OUTER_SHA,
                             {M1141_RECORDS.name})
    value = M1141_RECORDS.lstat()
    require(rows1141.get(M1141_RECORDS.name) == M1141_RECORDS_SHA and
            rows1141.get("m1141ca_schedule_release.json") == M1141_RELEASE_SHA and
            stat.S_ISREG(value.st_mode) and not M1141_RECORDS.is_symlink() and
            value.st_size == M1141_RECORDS_BYTES,
            "M1141 production schedule sealed metadata drift")
    rows1148 = manifest_rows(M1148, M1148_MANIFEST_SHA, M1148_OUTER_SHA)
    require(rows1148.get(M1148_AUTHORITY.name) == M1148_AUTHORITY_SHA,
            "M1148 authority identity drift")
    authority_mapping = strict_json(M1148_AUTHORITY)
    require(author["status"] ==
                "PASS_M1161CA_SOURCE_AND_BOUNDED_LIVE_HOOK__DIFFERENT_AUTHOR_HAMMER_REQUIRED" and
            author["subject"]["sha256"] == SOURCE_SHA and
            m1138["status"].startswith("PASS_M1138C_") and
            m1145["status"].startswith("PASS_M1145CA_") and
            m1157["status"] ==
                "PASS_M1157CA_DIFFERENT_AUTHOR_RESULT_HAMMER__EXPECTED_DIGEST_AUTHORITY_ONLY",
            "upstream authorization status drift")

    opened_production: list[str] = []
    original_builtin_open = builtins.open
    original_os_open = os.open
    original_path_open = Path.open

    def guarded_builtin_open(file, *args, **kwargs):
        try:
            resolved = Path(file).resolve()
        except (TypeError, OSError):
            resolved = None
        if resolved == M1141_RECORDS.resolve():
            opened_production.append("builtins.open")
            raise HammerFailure("production JSONL open forbidden")
        return original_builtin_open(file, *args, **kwargs)

    def guarded_os_open(file, *args, **kwargs):
        try:
            resolved = Path(file).resolve()
        except (TypeError, OSError):
            resolved = None
        if resolved == M1141_RECORDS.resolve():
            opened_production.append("os.open")
            raise HammerFailure("production JSONL os.open forbidden")
        return original_os_open(file, *args, **kwargs)

    def guarded_path_open(self, *args, **kwargs):
        if self.resolve() == M1141_RECORDS.resolve():
            opened_production.append("Path.open")
            raise HammerFailure("production JSONL Path.open forbidden")
        return original_path_open(self, *args, **kwargs)

    namespace_before: tuple[str, ...]
    attacks: dict[str, str] = {}
    tracemalloc.start()
    try:
        builtins.open = guarded_builtin_open
        os.open = guarded_os_open
        Path.open = guarded_path_open
        module = load_subject()
        namespace_before = tuple(map(str, module._namespace_paths()))
        require(namespace_before == (), "production namespace not fresh")
        preflight = module.source_preflight(True)
        bounded = module.source_bounded_self_test()

        # Zero argument enforcement must precede any production call.
        called = {"production": False}
        original_production_main = module.production_main
        module.production_main = lambda: called.__setitem__("production", True)
        saved_argv = sys.argv
        try:
            sys.argv = [str(SOURCE), "unexpected"]
            try:
                module.main()
            except BaseException as error:
                attacks["nonzero_argument"] = type(error).__name__ + ": " + str(error)
            else:
                raise HammerFailure("nonzero argument escaped")
        finally:
            sys.argv = saved_argv
            module.production_main = original_production_main
        require(called["production"] is False, "argument rejection called production")

        base = module._fixture_payload()
        lines = base.splitlines(keepends=True)
        require(len(lines) == 6, "bounded fixture geometry drift")
        attacks["reorder"] = expect_replay_reject(
            module, b"".join([lines[1], lines[0], *lines[2:]]), 6)
        attacks["duplicate"] = expect_replay_reject(
            module, b"".join([lines[0], lines[0], *lines[2:]]), 6)
        attacks["drop"] = expect_replay_reject(module, b"".join(lines[:-1]), 6)
        tampered = strict_json_bytes(lines[0][:-1])
        tampered["requested_cycle_first"] += 1
        tampered_line = (json.dumps(tampered, sort_keys=True,
                                    separators=(",", ":")) + "\n").encode()
        attacks["tamper"] = expect_replay_reject(
            module, b"".join([tampered_line, *lines[1:]]), 6)
        bad_coord = strict_json_bytes(lines[0][:-1])
        bad_coord["partition"] = module.PARTITIONS
        bad_coord["task_sequence_ordinal"] = module.PARTITIONS
        bad_coord["schedule_record_provenance_sha256"] = module.record_provenance(
            bad_coord["axis"], bad_coord["task_sequence_ordinal"],
            bad_coord["sample"], bad_coord["operator"], bad_coord["chunk"],
            bad_coord["partition"], bad_coord["requested_cycle_first"],
            bad_coord["source_task_provenance_sha256"])
        bad_coord_line = (json.dumps(bad_coord, sort_keys=True,
                                     separators=(",", ":")) + "\n").encode()
        attacks["coordinate"] = expect_replay_reject(
            module, b"".join([bad_coord_line, *lines[1:]]), 6)

        m1137_module = module.load_m1137()
        authority_attack_count = 0
        for mutation in ("id", "count", "digest", "extra"):
            attacked = json.loads(json.dumps(authority_mapping))
            if mutation == "id":
                attacked["authority_id_sha256"] = "0" * 64
            elif mutation == "count":
                attacked["expected_count_by_axis"]["candidate"] -= 1
            elif mutation == "digest":
                attacked["expected_digest_by_axis"]["candidate"] = "z" * 64
            else:
                attacked["unexpected"] = True
            try:
                module._authority_from_sealed_json(m1137_module, attacked)
            except BaseException as error:
                attacks["authority_" + mutation] = type(error).__name__ + ": " + str(error)
                authority_attack_count += 1
        require(authority_attack_count == 4, "authority attack escaped")
        valid_production_authority = module._authority_from_sealed_json(
            m1137_module, authority_mapping)
        try:
            m1137_module.M1016SuccessorPerTaskWeightBeatHook(
                valid_production_authority, lambda row: None)
        except BaseException as error:
            attacks["authority_not_injected"] = type(error).__name__ + ": " + str(error)
        else:
            raise HammerFailure("production authority usable without sealed injection")

        # Inject one downstream failure and prove that validator, producer cursor,
        # and row receipt all retain the last committed beat and can retry exactly.
        receipt_sink = module.OAxesRowReceiptSink()
        class FailOnce:
            def __init__(self):
                self.calls = 0
                self.failed = False
            def __call__(self, row):
                self.calls += 1
                if self.calls == 2 and not self.failed:
                    self.failed = True
                    raise RuntimeError("M1164 injected sink exception")
                receipt_sink(row)
        fail_once = FailOnce()
        hook = m1137_module.M1016SuccessorPerTaskWeightBeatHook(
            m1137_module.bounded_authority(), fail_once)
        try:
            hook.stream_bounded_task(axis="candidate", task_id=0,
                                     requested_cycle_first=5)
        except RuntimeError as error:
            attacks["sink_exception"] = str(error)
        else:
            raise HammerFailure("sink exception did not propagate")
        snap = hook.snapshot()
        require(snap["successor"]["candidate"]["emitted"] == 1 and
                snap["validator"]["candidate"]["event_count"] == 1 and
                receipt_sink._axis["candidate"].count == 1,
                "sink exception partially committed state")
        hook.stream_bounded_task(axis="candidate", task_id=0,
                                 requested_cycle_first=5)
        hook.stream_bounded_task(axis="candidate", task_id=1,
                                 requested_cycle_first=6)
        for axis in ("strongest_zero", "same_coordinate_bit"):
            hook.stream_bounded_task(axis=axis, task_id=0, requested_cycle_first=5)
            hook.stream_bounded_task(axis=axis, task_id=1, requested_cycle_first=6)
        terminal_after_retry = hook.finalize()
        row_after_retry = receipt_sink.finalize(4)
        require(fail_once.calls == 13 and
                all(terminal_after_retry["events_per_axis"][axis] == 4 and
                    row_after_retry["axes"][axis]["rows"] == 4 and
                    row_after_retry["axes"][axis]["stall_cycles"] == 2 and
                    row_after_retry["axes"][axis]["weight_service_makespan_coordinate"] == 9
                    for axis in module.AXES),
                "sink retry terminal conservation/cycle drift")

        # Namespace collision is tested in a private temp root by rebinding only
        # the source's namespace constants.  No real production path is mutated.
        import tempfile
        saved_namespace = (module.RESULTS, module.RESULT, module.ATTEMPT,
                           module.LOCK, module.WORK_PREFIX, module.FAILURE_PREFIX)
        try:
            with tempfile.TemporaryDirectory(prefix="m1164_namespace_") as temp:
                temp_root = Path(temp)
                module.RESULTS = temp_root
                module.RESULT = temp_root / "result"
                module.ATTEMPT = temp_root / "attempt"
                module.LOCK = temp_root / "lock"
                module.WORK_PREFIX = ".work."
                module.FAILURE_PREFIX = "failure."
                module.ATTEMPT.mkdir()
                try:
                    module.source_preflight(True)
                except BaseException as error:
                    attacks["namespace_collision"] = type(error).__name__ + ": " + str(error)
                else:
                    raise HammerFailure("namespace collision escaped")
        finally:
            (module.RESULTS, module.RESULT, module.ATTEMPT, module.LOCK,
             module.WORK_PREFIX, module.FAILURE_PREFIX) = saved_namespace

        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        require('require(len(sys.argv) == 1, "M1161CA accepts zero arguments")' in
                SOURCE.read_text(encoding="utf-8"), "zero-arg guard absent")
        row_source = inspect.getsource(module.OAxesRowReceiptSink)
        driver_source = inspect.getsource(module.ScheduleReplayDriver)
        require("append(" not in row_source and "append(" not in driver_source and
                "retained_schedule_event_row_or_key_history\": False" in
                    SOURCE.read_text(encoding="utf-8") and
                any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                    node.func.attr == "stream_production_task" for node in ast.walk(tree)),
                "O(axes)/live producer static boundary drift")
        current, peak = tracemalloc.get_traced_memory()
        resources = module.resource_preflight()
    finally:
        Path.open = original_path_open
        os.open = original_os_open
        builtins.open = original_builtin_open
        tracemalloc.stop()

    namespace_after = tuple(map(str, module._namespace_paths()))
    require(opened_production == [] and namespace_before == namespace_after == () and
            bounded["records"] == 6 and bounded["events"] == 12 and
            len(attacks) == 13 and resources["same_uid_conflicts"] == 0,
            "terminal source hammer conservation drift")
    review = {
        "schema": "m1164_m1161ca_c1_production_real_replay_driver_hammer_r1_v1",
        "status": "PASS_M1164_M1161CA_DIFFERENT_AUTHOR_SOURCE_HAMMER__EXACTLY_ONE_PRODUCTION_LAUNCH_AUTHORIZED_AFTER_FRESH_PREFLIGHT",
        "date": "2026-08-30",
        "verdict": "GO_EXACTLY_ONE_ZERO_ARGUMENT_PRODUCTION_REPLAY_LAUNCH__NO_RETRY__RESULT_HAMMER_MANDATORY",
        "subject": {"path": SOURCE.relative_to(HW).as_posix(),
                    "sha256": SOURCE_SHA},
        "sealed_chain": {
            "m1135_source_sha256": M1135_SHA,
            "m1137_source_sha256": M1137_SHA,
            "m1138_outer_seal_file_sha256": M1138_ID[2],
            "m1141_records_sha256_expected_without_hammer_open": M1141_RECORDS_SHA,
            "m1141_records_bytes": M1141_RECORDS_BYTES,
            "m1141_outer_seal_file_sha256": M1141_OUTER_SHA,
            "m1145_outer_seal_file_sha256": M1145_ID[2],
            "m1148_authority_sha256": M1148_AUTHORITY_SHA,
            "m1148_outer_seal_file_sha256": M1148_OUTER_SHA,
            "m1157_outer_seal_file_sha256": M1157_ID[2],
            "author_receipt_outer_seal_file_sha256": AUTHOR_ID[2],
        },
        "bounded_evidence": {
            "records": bounded["records"], "events": bounded["events"],
            "terminal_events_per_axis": 4,
            "row_stall_cycles_per_axis": 2,
            "weight_service_makespan_coordinate_per_axis": 9,
            "sink_calls_with_one_exception": fail_once.calls,
            "attacks_rejected": attacks,
            "production_schedule_open_count": len(opened_production),
            "namespace_before": list(namespace_before),
            "namespace_after": list(namespace_after),
            "state_complexity": "O(axes + axes*24) plus one JSON line and one row",
            "tracemalloc_current_bytes": current,
            "tracemalloc_peak_bytes": peak,
        },
        "production_geometry": {
            "schedule_records": 2_436_480,
            "records_per_axis": 812_160,
            "events": 212_559_552,
            "events_per_axis": 70_853_184,
            "strict_task_major_axis_minor": True,
            "exact_nine_schedule_fields_and_provenance": True,
        },
        "resource_preflight_observed": resources,
        "authorization": {
            "exactly_one_production_launch": True,
            "zero_arguments": True,
            "fresh_source_and_authorities_reverify_required": True,
            "fresh_namespace_required": True,
            "resource_and_same_uid_preflight_required": True,
            "automatic_retry": False,
            "different_author_result_hammer_required": True,
            "eda_gpu_remote": False,
        },
        "claim_boundary": {
            "source_and_bounded_fixture_hammer": True,
            "production_replay_executed_by_this_hammer": False,
            "production_jsonl_opened_by_this_hammer": False,
            "future_result_is_real_m1137_to_m1135_replay": True,
            "future_cycles_are_1rw_weight_service_schedule_coordinates_only": True,
            "rtl_cycle_or_system_speedup": False,
            "paper_citable_performance": False,
            "paper_ppa_ready": False,
        },
        "docs359_sha256": DOCS359_SHA,
    }
    mechanical = {
        "source_sha256": SOURCE_SHA,
        "production_schedule_open_count": len(opened_production),
        "namespace_unchanged": namespace_before == namespace_after == (),
        "bounded_records": 6,
        "bounded_events": 12,
        "attack_classes_rejected": len(attacks),
        "sink_exception_retry_exact": True,
        "same_uid_conflicts": resources["same_uid_conflicts"],
        "docs359_sha256": sha256(DOCS359),
    }
    (HERE / "bounded_hammer.json").write_text(
        json.dumps({"bounded": bounded, "attacks": attacks,
                    "terminal_after_retry": terminal_after_retry,
                    "row_after_retry": row_after_retry},
                   indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    (HERE / "mechanical_checks.json").write_text(
        json.dumps(mechanical, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    (HERE / "review.json").write_text(
        json.dumps(review, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    (HERE / "review.md").write_text(
        "# M1164 / M1161CA different-author source hammer\n\n"
        "PASS. The sealed live M1137C producer, M1135C validator, M1141CA "
        "schedule metadata, and M1148CA/M1157CA digest authority chain were "
        "independently rechecked. Reorder, drop, duplicate, provenance/coordinate "
        "tamper, four authority mutations, namespace collision, bad argv, and a "
        "downstream sink exception were rejected. The exception path retried "
        "without partial producer, validator, or row-receipt commit. The 836 MB "
        "production JSONL was never opened. Exactly one zero-argument production "
        "replay may now launch after its own fresh resource/same-UID preflight; "
        "automatic retry is forbidden and a fresh result hammer is mandatory. "
        "Any cycle field remains a 1RW weight-service schedule coordinate, never "
        "an RTL cycle or system speedup.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
