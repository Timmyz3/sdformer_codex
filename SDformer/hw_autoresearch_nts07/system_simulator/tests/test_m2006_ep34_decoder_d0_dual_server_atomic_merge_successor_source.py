#!/usr/bin/env python3
"""No-production security regressions for M2006."""
from __future__ import print_function

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / (
    "build_m2006_ep34_decoder_d0_dual_server_atomic_merge_successor_source.py")
spec = importlib.util.spec_from_file_location("m2006_test_target", str(SOURCE))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def expect_failure(action):
    try:
        action()
    except (M.M2006Error, M.M2003.M2003Error, OSError, tarfile.TarError):
        return
    raise AssertionError("expected fail-closed rejection")


def fake_record(pid, ppid, text):
    return {"pid": pid, "ppid": ppid, "starttime_ticks": pid * 10,
        "cmdline_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "cmdline_text": text, "cwd": str(M.HW.parent)}


def actual_row():
    keys = {"schema", "status", "source_sha256", "release_sha256",
        "attempt_sha256", "checkpoint_sha256", "resource_manifest_sha256",
        "shard_ordinal", "shard", "configuration_order", "metrics",
        "integer_ratio_inputs", "payload_fd_sha256", "payload_fd_size",
        "rss", "automatic_retry", "shard_isolated", "monolithic_full_call",
        "full_decoder", "system_speedup", "paper_result",
        "independent_result_hammer_pending"}
    return dict((key, key) for key in keys)


def make_two_ordinal_tar(path, duplicate=False, symlink=False):
    with tarfile.open(str(path), "w") as stream:
        for ordinal in range(4500, 4502):
            names = M.M2003._archive_names(ordinal)
            directory = tarfile.TarInfo(names["directory"])
            directory.type = tarfile.DIRTYPE
            stream.addfile(directory)
            for key in ("attempt", "result_json", "manifest", "outer"):
                info = tarfile.TarInfo(names[key])
                payload = b"x\n"
                info.size = len(payload)
                info.mode = 0o400 if key == "attempt" else 0o600
                if symlink and ordinal == 4501 and key == "outer":
                    info.type = tarfile.SYMTYPE
                    info.linkname = "../escape"
                    info.size = 0
                    stream.addfile(info)
                else:
                    stream.addfile(info, io.BytesIO(payload))
            if duplicate and ordinal == 4501:
                info = tarfile.TarInfo(names["manifest"])
                info.size = 2
                stream.addfile(info, io.BytesIO(b"x\n"))


def main():
    launcher_text = (
        "bash build_m1704_ep34_decoder_d0_execution_authority_adapter_source.py "
        "range(1,8700)")
    records = [fake_record(10, 1, launcher_text),
        fake_record(11, 10, "python3 -"),
        fake_record(12, 11, "python3 -"),
        fake_record(13, 11, "python3 -"),
        fake_record(14, 11, "python3 -")]
    classified = M.classify_process_records(records)
    assert tuple(row["role"] for row in classified) == M.EXPECTED_ROLES
    expect_failure(lambda: M.classify_process_records([]))
    expect_failure(lambda: M.classify_process_records(records[:1]))
    unrelated = list(records)
    unrelated[0] = fake_record(10, 1, "bash unrelated")
    expect_failure(lambda: M.classify_process_records(unrelated))

    row = actual_row()
    core = M.exact_receipt_core(row)
    assert "rss" not in core
    assert "independent_result_hammer_pending" in core
    changed = dict(row)
    changed["independent_result_hammer_pending"] = "changed"
    assert M.exact_receipt_core(row) != M.exact_receipt_core(changed)
    changed = dict(row)
    changed["unknown"] = True
    expect_failure(lambda: M.exact_receipt_core(changed))

    events = []
    def verifier(_stage, ordinal):
        events.append("verify{}".format(ordinal))
        if ordinal == 4501:
            raise M.M2006Error("late corrupt shard")
        return {"row": row, "attempt_sha256": "a" * 64,
                "seal": {"manifest_sha256": "b" * 64}}
    expect_failure(lambda: M.verify_all_remote_before_mutation(
        Path("unused"), 4500, 4502, verifier,
        lambda _stage, _ordinal, _verified: {
            "result_json_sha256": "c" * 64,
            "manifest_sha256": "b" * 64}))
    assert events == ["verify4500", "verify4501"]

    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        good = root / "good.tar"
        make_two_ordinal_tar(good)
        digest = M.sha256(good)
        stage, inspection = M.single_fd_verify_and_extract(
            good, digest, 4500, 4502, root / "stage")
        assert inspection == {"archive_sha256": digest,
            "directories": 2, "files": 8, "ordinals": 2}
        assert stage.is_dir()
        duplicate = root / "duplicate.tar"
        make_two_ordinal_tar(duplicate, duplicate=True)
        expect_failure(lambda: M.single_fd_verify_and_extract(
            duplicate, M.sha256(duplicate), 4500, 4502,
            root / "duplicate_stage"))
        link = root / "link.tar"
        make_two_ordinal_tar(link, symlink=True)
        expect_failure(lambda: M.single_fd_verify_and_extract(
            link, M.sha256(link), 4500, 4502, root / "link_stage"))
    assert M.B.G.TOTAL_SHARDS == 8700
    print(json.dumps({"status": "PASS_M2006_SECURITY_TEST",
        "empty_pid_rejected": True, "unrelated_pid_rejected": True,
        "late_corrupt_rejected_before_mutation": True,
        "single_fd_archive": True, "duplicate_rejected": True,
        "symlink_rejected": True, "rss_only_exclusion": True,
        "archive_opened_production": False, "merge": False,
        "reducer": False, "gpu": False, "eda": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
