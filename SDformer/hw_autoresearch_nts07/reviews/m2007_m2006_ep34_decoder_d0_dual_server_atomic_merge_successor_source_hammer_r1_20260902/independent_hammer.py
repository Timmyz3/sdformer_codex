#!/usr/bin/env python3
"""Different-author, no-production hammer for M2006.

All filesystem mutation is confined to TemporaryDirectory.  The production
archive, shard namespaces, payloads, reducer, GPU, and EDA are never opened.
CPython 3.6 safe.
"""
from __future__ import print_function

import hashlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import stat
import tarfile
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HW / "system_simulator/scripts/build_m2006_ep34_decoder_d0_dual_server_atomic_merge_successor_source.py"
spec = importlib.util.spec_from_file_location("m2007_target", str(SOURCE))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def rejected(action):
    try:
        action()
    except (M.M2006Error, M.M2003.M2003Error, OSError, tarfile.TarError,
            KeyError, ValueError):
        return True
    return False


def fake_record(pid, ppid, text):
    payload = text.encode("utf-8")
    return {"pid": pid, "ppid": ppid, "starttime_ticks": pid * 101,
            "cmdline_sha256": hashlib.sha256(payload).hexdigest(),
            "cmdline_text": text, "cwd": str(M.HW.parent)}


def receipt_row(release=None):
    release = release or M.M1706_RELEASE_SHA256
    return {"schema": "s", "status": "ok", "source_sha256": "1" * 64,
        "release_sha256": release, "attempt_sha256": "2" * 64,
        "checkpoint_sha256": M.B.G.CHECKPOINT_SHA256,
        "resource_manifest_sha256": M.B.G.RESOURCE_SHA256,
        "shard_ordinal": 0, "shard": {}, "configuration_order": [],
        "metrics": [], "integer_ratio_inputs": {},
        "payload_fd_sha256": "3" * 64, "payload_fd_size": 0,
        "rss": 7, "automatic_retry": False, "shard_isolated": True,
        "monolithic_full_call": False, "full_decoder": False,
        "system_speedup": False, "paper_result": False,
        "independent_result_hammer_pending": True}


def add_member(stream, name, kind="file", payload=b"x\n", mode=0o600):
    info = tarfile.TarInfo(name)
    info.mode = mode
    if kind == "dir":
        info.type = tarfile.DIRTYPE
        info.size = 0
        stream.addfile(info)
    elif kind == "symlink":
        info.type = tarfile.SYMTYPE
        info.linkname = "../escape"
        info.size = 0
        stream.addfile(info)
    elif kind == "hardlink":
        info.type = tarfile.LNKTYPE
        info.linkname = "../escape"
        info.size = 0
        stream.addfile(info)
    elif kind == "fifo":
        info.type = tarfile.FIFOTYPE
        info.size = 0
        stream.addfile(info)
    elif kind == "char":
        info.type = tarfile.CHRTYPE
        info.size = 0
        stream.addfile(info)
    else:
        info.size = len(payload)
        stream.addfile(info, io.BytesIO(payload))


def make_archive(path, attack=None):
    with tarfile.open(str(path), "w") as stream:
        for ordinal in range(4500, 4502):
            names = M.M2003._archive_names(ordinal)
            directory_name = names["directory"]
            if attack == "traversal" and ordinal == 4501:
                directory_name = "../escape"
            add_member(stream, directory_name, "dir")
            for key in ("attempt", "result_json", "manifest", "outer"):
                kind = "file"
                if ordinal == 4501 and key == "outer" and attack in (
                        "symlink", "hardlink", "fifo", "char"):
                    kind = attack
                add_member(stream, names[key], kind,
                           mode=0o400 if key == "attempt" else 0o600)
            if attack == "duplicate" and ordinal == 4501:
                add_member(stream, names["manifest"])


def main():
    findings = {}
    positives = {}

    launcher = ("bash build_m1704_ep34_decoder_d0_execution_authority_adapter_source.py "
                "range(1,8700)")
    records = [fake_record(101, 1, launcher),
        fake_record(102, 101, "python3 -"),
        fake_record(103, 102, "python3 -"),
        fake_record(104, 102, "python3 -"),
        fake_record(105, 102, "python3 -")]
    roles = M.classify_process_records(records)
    positives["five_exact_roles"] = [row["role"] for row in roles]
    positives["empty_process_set_rejected"] = rejected(
        lambda: M.classify_process_records([]))
    unrelated = list(records)
    unrelated[0] = fake_record(101, 1, "bash unrelated")
    positives["unrelated_process_set_rejected"] = rejected(
        lambda: M.classify_process_records(unrelated))
    inconsistent = [dict(row) for row in records]
    inconsistent[0]["cmdline_sha256"] = "0" * 64
    findings["inconsistent_process_cmdline_hash_accepted"] = not rejected(
        lambda: M.classify_process_records(inconsistent))

    exact = receipt_row()
    core = M.exact_receipt_core(exact)
    positives["rss_only_excluded"] = (
        set(exact) - set(core) == {"rss"} and
        "independent_result_hammer_pending" in core)
    missing = dict(exact)
    del missing["paper_result"]
    unknown = dict(exact)
    unknown["unknown"] = True
    positives["missing_receipt_key_rejected"] = rejected(
        lambda: M.exact_receipt_core(missing))
    positives["unknown_receipt_key_rejected"] = rejected(
        lambda: M.exact_receipt_core(unknown))

    original_verify = M.M1704.M1688.verify_sealed_shard
    try:
        M.M1704.M1688.verify_sealed_shard = lambda _ordinal: {
            "row": receipt_row("f" * 64)}
        positives["wrong_local_release_rejected"] = rejected(
            lambda: M.verify_local_shard(7))
    finally:
        M.M1704.M1688.verify_sealed_shard = original_verify

    events = []
    def late_verifier(_stage, ordinal):
        events.append("verify{}".format(ordinal))
        if ordinal == 4501:
            raise M.M2006Error("late corrupt remote")
        return {"row": exact, "attempt_sha256": "a" * 64,
                "seal": {"manifest_sha256": "b" * 64}}
    positives["late_remote_corruption_rejected"] = rejected(lambda:
        M.verify_all_remote_before_mutation(Path("unused"), 4500, 4502,
            late_verifier, lambda _s, _o, _v: {
                "result_json_sha256": "c" * 64,
                "manifest_sha256": "b" * 64}))
    positives["late_remote_has_verify_only_trace"] = (
        events == ["verify4500", "verify4501"])

    merge_source = inspect.getsource(M.merge_and_reduce)
    resume_source = inspect.getsource(M.manual_resume_from_verified_plan)
    positives["attempt_precedes_archive_open_in_source"] = (
        merge_source.index("_consume_attempt") <
        merge_source.index("single_fd_verify_and_extract"))
    positives["plan_precedes_canonical_merge_in_source"] = (
        merge_source.index("_publish_verified_plan") <
        merge_source.index("_finish_merge"))
    positives["resume_does_not_reopen_archive_in_source"] = (
        "single_fd_verify_and_extract" not in resume_source and
        "tarfile" not in resume_source)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        good = root / "good.tar"
        make_archive(good)
        original_open = M.os.open
        archive_open_count = [0]
        def counting_open(path, flags, *args):
            if os.path.abspath(str(path)) == os.path.abspath(str(good)):
                archive_open_count[0] += 1
            return original_open(path, flags, *args)
        M.os.open = counting_open
        try:
            stage, inspection = M.single_fd_verify_and_extract(
                good, M.sha256(good), 4500, 4502, root / "good_stage")
        finally:
            M.os.open = original_open
        positives["archive_open_count_exactly_one"] = (
            archive_open_count[0] == 1 and inspection["ordinals"] == 2 and
            stage.is_dir())

        for attack in ("traversal", "duplicate", "symlink", "hardlink",
                       "fifo", "char"):
            archive = root / (attack + ".tar")
            make_archive(archive, attack)
            positives[attack + "_archive_rejected"] = rejected(lambda p=archive:
                M.single_fd_verify_and_extract(
                    p, M.sha256(p), 4500, 4502,
                    root / (p.stem + "_stage")))

        old_attempt = M.ATTEMPT
        old_prestop = M.PRESTOP
        try:
            M.ATTEMPT = root / "overall.attempt"
            M.PRESTOP = root / "prestop"
            M.PRESTOP.mkdir()
            (M.PRESTOP / "result.json").write_text("{}\n")
            release = {"archive_sha256": "d" * 64}
            M._consume_attempt(release, "e" * 64)
            positives["overall_attempt_mode_0400"] = (
                stat.S_IMODE(M.ATTEMPT.stat().st_mode) == 0o400)
            positives["overall_attempt_second_consume_rejected"] = rejected(
                lambda: M._consume_attempt(release, "e" * 64))
        finally:
            M.ATTEMPT = old_attempt
            M.PRESTOP = old_prestop

        # Reproduce a crash after creation of the per-result import work.
        # M2006 reuses M2003._copy_result_tree; the subsequent plan-resume call
        # rejects the orphan instead of quarantining/continuing it.
        source = root / "source_result"
        source.mkdir()
        (source / "result.json").write_text("{}\n")
        target = root / "canonical_result"
        original_copytree = M.M2003.shutil.copytree
        def interrupted_copytree(src, dst, *args, **kwargs):
            original_copytree(src, dst, *args, **kwargs)
            raise OSError("synthetic interruption after import-work creation")
        M.M2003.shutil.copytree = interrupted_copytree
        first_failed = rejected(lambda:
            M.M2003._copy_result_tree(source, target))
        M.M2003.shutil.copytree = original_copytree
        orphan = Path(str(target) + ".m2003_import_work")
        second_failed = rejected(lambda:
            M.M2003._copy_result_tree(source, target))
        findings["interrupted_import_work_strands_manual_resume"] = (
            first_failed and orphan.exists() and not target.exists() and
            second_failed)

        # Reproduce the check-then-rename no-overwrite race in plan publish.
        old_plan = M.PLAN
        original_verify_tree = M.B.verify_sealed_tree
        plan_target = root / "verified_plan"
        raced_inode = [None]
        def verify_then_race(plan_work, *args, **kwargs):
            result = original_verify_tree(plan_work, *args, **kwargs)
            plan_target.mkdir()
            raced_inode[0] = plan_target.stat().st_ino
            return result
        try:
            M.PLAN = plan_target
            M.B.verify_sealed_tree = verify_then_race
            M._publish_verified_plan([], "a" * 64, root / "stage")
            findings["verified_plan_atomic_publish_overwrites_raced_empty_target"] = (
                plan_target.is_dir() and
                plan_target.stat().st_ino != raced_inode[0] and
                (plan_target / "result.json").exists())
        except (OSError, M.M2006Error):
            findings["verified_plan_atomic_publish_overwrites_raced_empty_target"] = False
        finally:
            M.PLAN = old_plan
            M.B.verify_sealed_tree = original_verify_tree

        # A resumed result records zero archive opens although the campaign's
        # initial leg necessarily opened the one authorized archive.
        old_result = M.RESULT
        try:
            M.RESULT = root / "resumed_result"
            resumed = M._publish_result("1" * 64, "2" * 64,
                {"manifest_sha256": "3" * 64}, {}, [], {}, True)
            findings["resumed_receipt_reports_campaign_archive_open_zero"] = (
                resumed["archive_open_count"] == 0)
        finally:
            M.RESULT = old_result

    assert all(value is True or isinstance(value, list)
               for value in positives.values()), positives
    expected_findings = {
        "interrupted_import_work_strands_manual_resume": True,
        "verified_plan_atomic_publish_overwrites_raced_empty_target": True,
        "resumed_receipt_reports_campaign_archive_open_zero": True,
        "inconsistent_process_cmdline_hash_accepted": True}
    assert findings == expected_findings, findings
    print(json.dumps({"status": "PASS_M2007_INDEPENDENT_HAMMER__SOURCE_FAILINGS_REPRODUCED",
        "positive_checks": positives, "findings": findings,
        "production_archive_opened": False, "canonical_namespace_mutated": False,
        "merge_executed": False, "reducer_executed": False,
        "payload_opened": False, "gpu": False, "eda": False},
        sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
