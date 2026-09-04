#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""One-shot author check for the M1161CA source; bounded fixture only."""
from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tracemalloc

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HW / "system_simulator/scripts/run_m1161ca_c1_production_real_replay_driver_one_shot_source.py"
SOURCE_SHA = "d7ffb8dbab289e83fd8a32f4ed5244cd005a4b6d0785b586df932fd6a97ee20d"
PRODUCTION_RECORDS = (HW / "results/m1141ca_c1_production_schedule_release_r1_20260830/"
                      "m1141ca_per_task_schedule_records.jsonl").resolve()
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusive(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(payload); stream.flush(); os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path: Path, value) -> None:
    exclusive(path, (json.dumps(value, indent=2, sort_keys=True,
                                allow_nan=False) + "\n").encode())


def main() -> None:
    require(sha256(SOURCE) == SOURCE_SHA and sha256(DOCS359) == DOCS359_SHA,
            "source/docs identity drift")
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    attributes = [node.func.attr for node in calls if isinstance(node.func, ast.Attribute)]
    names = [node.func.id for node in calls if isinstance(node.func, ast.Name)]
    require("stream_production_task" in attributes and
            "stream_bounded_task" in attributes and
            "production_main" in names and
            'require(len(sys.argv) == 1, "M1161CA accepts zero arguments")' in text and
            "producer_replay_and_1rw_schedule_receipt" in text and
            '"rtl_cycle_or_system_speedup": False' in text and
            "event_output_written" in text and
            "ATTEMPT.mkdir" in text and "_rename_noreplace" in text,
            "static driver/one-shot/claim-boundary construct absent")

    result = HW / "results/m1161ca_c1_production_real_replay_r1_20260830"
    attempt = HW / "results/.m1161ca_c1_production_real_replay_attempt_consumed"
    failures = tuple((HW / "results").glob(
        "m1161ca_c1_production_real_replay_r1_20260830.failed_or_incomplete.*"))
    work = tuple((HW / "results").glob(".m1161ca_c1_production_real_replay_work.*"))
    namespace_before = (result.exists(), attempt.exists(), len(failures), len(work))
    require(namespace_before == (False, False, 0, 0), "production namespace not fresh")

    opened_production = []
    original_builtin_open = builtins.open
    original_os_open = os.open
    original_path_open = Path.open

    def checked_builtin_open(file, *args, **kwargs):
        try:
            resolved = Path(file).resolve()
        except TypeError:
            resolved = None
        if resolved == PRODUCTION_RECORDS:
            opened_production.append("builtins.open")
            raise RuntimeError("bounded author test attempted production schedule open")
        return original_builtin_open(file, *args, **kwargs)

    def checked_os_open(file, *args, **kwargs):
        try:
            resolved = Path(file).resolve()
        except TypeError:
            resolved = None
        if resolved == PRODUCTION_RECORDS:
            opened_production.append("os.open")
            raise RuntimeError("bounded author test attempted production schedule os.open")
        return original_os_open(file, *args, **kwargs)

    def checked_path_open(self, *args, **kwargs):
        if self.resolve() == PRODUCTION_RECORDS:
            opened_production.append("Path.open")
            raise RuntimeError("bounded author test attempted production schedule Path.open")
        return original_path_open(self, *args, **kwargs)

    spec = importlib.util.spec_from_file_location("m1161ca_author_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "subject import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    tracemalloc.start()
    try:
        builtins.open = checked_builtin_open
        os.open = checked_os_open
        Path.open = checked_path_open
        spec.loader.exec_module(module)
        bounded = module.source_bounded_self_test()
        current, peak = tracemalloc.get_traced_memory()
    finally:
        Path.open = original_path_open
        os.open = original_os_open
        builtins.open = original_builtin_open
        tracemalloc.stop()
    namespace_after = (result.exists(), attempt.exists(),
                       len(tuple((HW / "results").glob(
                           "m1161ca_c1_production_real_replay_r1_20260830.failed_or_incomplete.*"))),
                       len(tuple((HW / "results").glob(
                           ".m1161ca_c1_production_real_replay_work.*"))))
    require(opened_production == [] and namespace_after == namespace_before and
            bounded["status"] ==
                "PASS_BOUNDED_TWO_TASK_LIVE_M1137_REPLAY__PRODUCTION_STOP" and
            bounded["records"] == 6 and bounded["events"] == 12 and
            bounded["attacks_rejected"] == 5 and
            bounded["production_schedule_opened"] is False and
            bounded["production_events_replayed"] == 0 and
            bounded["production_namespace_mutated"] is False and
            all(bounded["driver_terminal"]["m1137c_terminal"]
                    ["m1135c_terminal"]["axes"][axis]["events"] == 4
                for axis in module.AXES),
            "bounded author oracle drift")

    review = {
        "schema": "m1161ca_c1_production_real_replay_driver_author_review_r1_v1",
        "status": "PASS_M1161CA_SOURCE_AND_BOUNDED_LIVE_HOOK__DIFFERENT_AUTHOR_HAMMER_REQUIRED",
        "date": "2026-08-30",
        "verdict": "GO_DIFFERENT_AUTHOR_SOURCE_HAMMER_ONLY__STOP_PRODUCTION_REPLAY_EDA_GPU_REMOTE",
        "subject": {
            "path": SOURCE.relative_to(HW).as_posix(),
            "sha256": SOURCE_SHA,
            "contract_path": "contracts/m1161ca_c1_production_real_replay_driver_source_contract_r1_20260830.json",
        },
        "bounded_evidence": {
            "records": 6, "events": 12, "axes": 3,
            "tasks_per_axis": 2, "attacks_rejected": 5,
            "production_schedule_open_count": len(opened_production),
            "production_namespace_before": list(namespace_before),
            "production_namespace_after": list(namespace_after),
            "tracemalloc_current_bytes": current,
            "tracemalloc_peak_bytes": peak,
            "state_complexity": "O(axes + axes*24)",
        },
        "findings": {
            "sealed_schedule_record_parser": True,
            "independent_record_provenance": True,
            "per_record_live_m1137_stream_production_task_path_present": True,
            "bounded_live_m1137_stream_bounded_task_executed": True,
            "m1135_terminal_counts_and_digests_closed": True,
            "oaxes_row_sink_counts_digest_and_1rw_cycles_closed": True,
            "one_shot_attempt_before_schedule_open": True,
            "failure_quarantine_and_atomic_publish": True,
            "automatic_retry": False,
            "per_event_output": False,
        },
        "authorization": {
            "different_author_source_hammer_next": True,
            "production_replay_execution": False,
            "eda_rtl_gpu_remote": False,
        },
        "claim_boundary": {
            "source_and_bounded_fixture_only": True,
            "real_producer_replay": False,
            "producer_replay_cycles": False,
            "rtl_cycle_or_system_speedup": False,
            "paper_citable_performance": False,
            "paper_ppa_ready": False,
        },
        "docs359_sha256": DOCS359_SHA,
    }
    write_json(HERE / "bounded_oracle.json", bounded)
    write_json(HERE / "mechanical_checks.json", {
        "source_sha256": SOURCE_SHA,
        "production_schedule_open_count": len(opened_production),
        "namespace_unchanged": namespace_after == namespace_before,
        "static_production_call_path_present": True,
        "bounded_events": 12,
        "attacks_rejected": 5,
        "tracemalloc_peak_bytes": peak,
    })
    write_json(HERE / "review.json", review)
    exclusive(HERE / "review.md", (
        "# M1161CA author receipt\n\n"
        "PASS for additive source plus a two-task bounded live-hook fixture only. "
        "The production schedule was not opened and the one-shot namespace was not changed. "
        "A fresh different-author hammer is mandatory before the 212,559,552-event replay. "
        "Any later cycle receipt is a weight-service schedule coordinate, not RTL or system speedup.\n"
    ).encode())
    exclusive(HERE / "SOURCE_BOUNDED_TWO_TASK_ONLY_NO_PRODUCTION_REPLAY_NO_EDA.txt",
              b"SOURCE_BOUNDED_TWO_TASK_ONLY_NO_PRODUCTION_REPLAY_NO_EDA\n")
    exclusive(HERE / "RUN_COMPLETE.txt",
              b"PASS_M1161CA_SOURCE_AND_BOUNDED_LIVE_HOOK__DIFFERENT_AUTHOR_HAMMER_REQUIRED\n")
    members = []
    for member in HERE.rglob("*"):
        if member.name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "receipt symlink")
        if stat.S_ISREG(mode):
            members.append(member)
        else:
            require(stat.S_ISDIR(mode), "receipt special member")
    lines = [f"{sha256(member)}  {member.relative_to(HERE).as_posix()}"
             for member in sorted(members, key=lambda item: item.name)]
    exclusive(HERE / "SHA256SUMS", ("\n".join(lines) + "\n").encode())
    manifest_sha = sha256(HERE / "SHA256SUMS")
    exclusive(HERE / "SHA256SUMS.seal.sha256",
              f"{manifest_sha}  SHA256SUMS\n".encode())
    fd = os.open(HERE, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps({"status": review["status"], "source_sha256": SOURCE_SHA,
                      "bounded_events": 12, "production_schedule_open_count": 0,
                      "outer_seal_file_sha256": sha256(HERE / "SHA256SUMS.seal.sha256")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
