#!/usr/bin/env python3
"""No-production regressions for M2009 no-replace publication/resume."""
from __future__ import print_function

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / (
    "build_m2009_ep34_decoder_d0_noreplace_resume_successor_source.py")
spec = importlib.util.spec_from_file_location("m2009_test_target", str(SOURCE))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def expect_failure(action):
    try:
        action()
    except (M.M2009Error, OSError):
        return
    raise AssertionError("expected fail-closed rejection")


def record(pid, ppid, raw):
    return {"pid": pid, "ppid": ppid, "starttime_ticks": pid * 10,
        "cmdline_raw_hex": raw.hex(),
        "cmdline_sha256": hashlib.sha256(raw).hexdigest(),
        "cmdline_text": raw.replace(b"\0", b" ").decode().strip(),
        "cwd": str(M.HW.parent)}


def main():
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        source = root / "source"
        target = root / "target"
        source.write_text("source")
        target.write_text("target")
        expect_failure(lambda: M.rename_noreplace(source, target))
        assert source.read_text() == "source" and target.read_text() == "target"
        target.unlink()
        assert M.rename_noreplace(source, target) is True
        assert not source.exists() and target.read_text() == "source"

    launch = (b"bash build_m1704_ep34_decoder_d0_execution_authority_"
              b"adapter_source.py range(1,8700)\0")
    rows = [record(10, 1, launch), record(11, 10, b"python3\0-\0"),
            record(12, 11, b"python3\0-\0"),
            record(13, 11, b"python3\0-\0"),
            record(14, 11, b"python3\0-\0")]
    classified = M.classify_process_records(rows)
    assert tuple(row["role"] for row in classified) == M.EXPECTED_ROLES
    bad = [dict(row) for row in rows]
    bad[0]["cmdline_sha256"] = "0" * 64
    expect_failure(lambda: M.classify_process_records(bad))
    bad = [dict(row) for row in rows]
    bad[0]["unknown"] = True
    expect_failure(lambda: M.classify_process_records(bad))
    expect_failure(lambda: M.classify_process_records([]))

    assert M.describe()["repairs"][
        "renameat2_noreplace_all_canonical_publish"] is True
    assert M.describe()["repairs"][
        "orphan_import_work_validated_and_promoted"] is True
    source_text = SOURCE.read_text()
    assert "Path.rename(" not in source_text and ".rename(" not in source_text
    assert '"campaign_archive_open_count": 1' in source_text
    assert '"resume_leg_archive_open_count": 0 if resumed else None' in source_text
    print(json.dumps({"status": "PASS_M2009_SOURCE_TEST",
        "noreplace_race_rejected": True, "source_preserved": True,
        "target_preserved": True, "cmdline_hash_attack_rejected": True,
        "unknown_process_key_rejected": True,
        "empty_process_population_rejected": True,
        "production_archive_open": False, "merge": False,
        "reducer": False, "gpu": False, "eda": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
