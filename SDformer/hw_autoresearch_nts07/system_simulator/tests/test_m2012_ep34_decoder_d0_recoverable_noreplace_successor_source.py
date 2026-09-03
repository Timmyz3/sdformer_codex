#!/usr/bin/env python3
"""No-production regressions for M2012 recovery and process re-read."""
from __future__ import print_function

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / (
    "build_m2012_ep34_decoder_d0_recoverable_noreplace_successor_source.py")
spec = importlib.util.spec_from_file_location("m2012_test_target", str(SOURCE))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def expect_failure(action):
    try:
        action()
    except (M.M2012Error, OSError):
        return
    raise AssertionError("expected fail-closed rejection")


def record(pid, ppid, raw, start=None):
    return {"pid": pid, "ppid": ppid,
        "starttime_ticks": pid * 10 if start is None else start,
        "cmdline_raw_hex": raw.hex(),
        "cmdline_sha256": hashlib.sha256(raw).hexdigest(),
        "cmdline_text": raw.replace(b"\0", b" ").decode().strip(),
        "cwd": str(M.HW.parent)}


def main():
    launch = (b"bash build_m1704_ep34_decoder_d0_execution_authority_"
              b"adapter_source.py range(1,8700)\0")
    raw = [record(10, 1, launch), record(11, 10, b"python3\0-\0"),
           record(12, 11, b"python3\0-\0"),
           record(13, 11, b"python3\0-\0"),
           record(14, 11, b"python3\0-\0")]
    rows = M.classify_process_records(raw)
    table = dict((row["pid"], dict(row)) for row in raw)
    assert M.reread_process_records(rows, lambda pid: table[pid]) == rows
    changed = dict(table)
    changed[14] = record(14, 11, b"python3\0-\0", start=999999)
    expect_failure(lambda: M.reread_process_records(
        rows, lambda pid: changed[pid]))

    old_root = M.QUARANTINE_ROOT
    try:
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            M.QUARANTINE_ROOT = root / "quarantine"
            partial = root / "target.m2012_import_work"
            partial.mkdir()
            (partial / "result.json").write_text("{}\n")
            assert M.inspect_import_work_topology(partial) == "incomplete"
            first = M.quarantine_partial_import_work(partial, root / "target")
            assert first.exists() and not partial.exists()

            # A pre-existing quarantine slot must not be overwritten; the
            # next normal partial transaction advances to a new no-replace slot.
            partial.mkdir()
            (partial / "SHA256SUMS").write_text("partial\n")
            second = M.quarantine_partial_import_work(partial, root / "target")
            assert second.exists() and second != first and first.exists()

            alien = root / "alien"
            alien.mkdir()
            (alien / "unexpected").write_text("x")
            expect_failure(lambda: M.inspect_import_work_topology(alien))

            linked = root / "linked"
            linked.mkdir()
            (linked / "result.json").symlink_to(first / "result.json")
            expect_failure(lambda: M.inspect_import_work_topology(linked))
    finally:
        M.QUARANTINE_ROOT = old_root

    desc = M.describe()
    assert desc["repairs"]["partial_import_work_noreplace_quarantine"]
    assert desc["repairs"][
        "live_process_identity_reread_immediately_before_publish"]
    text = SOURCE.read_text()
    assert "shutil.rmtree" not in text and "os.replace" not in text
    print(json.dumps({"status": "PASS_M2012_SOURCE_TEST",
        "stable_reread_passed": True, "pid_reuse_rejected": True,
        "partial_copy_quarantined": True,
        "preexisting_quarantine_preserved": True,
        "unexpected_entry_rejected": True, "symlink_rejected": True,
        "production_process_capture": False, "archive_open": False,
        "merge": False, "reducer": False, "gpu": False, "eda": False},
        sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
