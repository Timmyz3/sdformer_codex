#!/usr/bin/env python3
"""Different-author, source-only M2009 transaction hammer.

All executable mutations are confined to TemporaryDirectory namespaces.  The
production process set, archive, shard namespace, reducer, payloads, GPU and
EDA tools are never touched.
"""
from __future__ import print_function

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / (
    "system_simulator/scripts/"
    "build_m2009_ep34_decoder_d0_noreplace_resume_successor_source.py")
EXPECTED_SOURCE_SHA256 = (
    "188619aaeabb381bbc1581e02392c40e6c61b2ddf3faf313186c6f750b94d8d9")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


require(sha256(SOURCE) == EXPECTED_SOURCE_SHA256, "M2009 source SHA drift")
spec = importlib.util.spec_from_file_location("m2010_target", str(SOURCE))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def expect_failure(action):
    try:
        action()
    except (M.M2009Error, OSError, ValueError, KeyError):
        return True
    raise AssertionError("expected fail-closed rejection")


def exact_receipt(ordinal=4500):
    return {
        "schema": "synthetic_shard", "status": "synthetic_complete",
        "source_sha256": "1" * 64, "release_sha256": "2" * 64,
        "attempt_sha256": "3" * 64,
        "checkpoint_sha256": M.B.G.CHECKPOINT_SHA256,
        "resource_manifest_sha256": M.B.G.RESOURCE_SHA256,
        "shard_ordinal": ordinal, "shard": [ordinal, ordinal + 1],
        "configuration_order": ["A1"], "metrics": {"cycles": 1},
        "integer_ratio_inputs": {"numerator": 1, "denominator": 1},
        "payload_fd_sha256": "4" * 64, "payload_fd_size": 1,
        "rss": 123, "automatic_retry": False, "shard_isolated": True,
        "monolithic_full_call": True, "full_decoder": False,
        "system_speedup": False, "paper_result": False,
        "independent_result_hammer_pending": True}


def make_sealed_import(path, row):
    path.mkdir(parents=True, mode=0o700)
    (path / "result.json").write_text(json.dumps(
        row, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    M.B.seal_work_tree(path)
    return M.B.verify_sealed_tree(
        path, allow_ignored_pycache=False, label="synthetic import")


def plan_row(path, row, seal):
    return {"ordinal": row["shard_ordinal"],
        "attempt_sha256": row["attempt_sha256"],
        "result_json_sha256": sha256(path / "result.json"),
        "manifest_sha256": seal["manifest_sha256"],
        "deterministic_core_sha256": M.M2006.canonical_sha(
            M.M2006.exact_receipt_core(row))}


def process_record(pid, ppid, raw):
    return {"pid": pid, "ppid": ppid, "starttime_ticks": pid * 10,
        "cmdline_raw_hex": raw.hex(),
        "cmdline_sha256": hashlib.sha256(raw).hexdigest(),
        "cmdline_text": raw.replace(b"\0", b" ").decode().strip(),
        "cwd": str(M.HW.parent)}


def process_rows():
    launch = (b"bash build_m1704_ep34_decoder_d0_execution_authority_"
              b"adapter_source.py range(1,8700)\0")
    return [process_record(10, 1, launch),
            process_record(11, 10, b"python3\0-\0"),
            process_record(12, 11, b"python3\0-\0"),
            process_record(13, 11, b"python3\0-\0"),
            process_record(14, 11, b"python3\0-\0")]


def main():
    positive = {}
    findings = {}

    with tempfile.TemporaryDirectory() as root_text:
        root = Path(root_text)

        # Atomic target collision after the verify phase must preserve both
        # the raced target and the already sealed source work tree.
        work = root / "publish.work"
        target = root / "publish"
        work.mkdir()
        (work / "result.json").write_text("{}\n")
        original_verify = M.B.verify_sealed_tree

        def raced_verify(*args, **kwargs):
            result = original_verify(*args, **kwargs)
            target.mkdir()
            (target / "attacker-marker").write_text("preserve\n")
            return result

        M.B.verify_sealed_tree = raced_verify
        try:
            expect_failure(lambda: M._publish_sealed_work(
                work, target, "synthetic raced publish"))
        finally:
            M.B.verify_sealed_tree = original_verify
        require((target / "attacker-marker").read_text() == "preserve\n",
                "raced canonical target was overwritten")
        require(work.is_dir(), "sealed source work disappeared on collision")
        positive["renameat2_target_race_preserves_both_trees"] = True

        # A complete, sealed orphan matching the immutable plan is promotable.
        valid_target = root / "valid_result"
        valid_import = Path(str(valid_target) + ".m2009_import_work")
        row = exact_receipt()
        seal = make_sealed_import(valid_import, row)
        expected = plan_row(valid_import, row, seal)
        M._promote_result_resumable(root / "unused_source",
                                    valid_target, expected)
        require(valid_target.is_dir() and not valid_import.exists(),
                "valid sealed orphan was not promoted")
        positive["sealed_plan_matching_orphan_promoted"] = True

        # A fully sealed but plan-mismatched orphan is rejected and preserved.
        evil_target = root / "evil_result"
        evil_import = Path(str(evil_target) + ".m2009_import_work")
        evil_row = exact_receipt(4501)
        evil_seal = make_sealed_import(evil_import, evil_row)
        evil_plan = plan_row(evil_import, evil_row, evil_seal)
        evil_plan["deterministic_core_sha256"] = "f" * 64
        expect_failure(lambda: M._promote_result_resumable(
            root / "unused_source", evil_target, evil_plan))
        require(evil_import.is_dir() and not evil_target.exists(),
                "malicious orphan was not preserved/rejected")
        positive["malicious_orphan_rejected_without_overwrite"] = True

        # The original P1 covered a normal crash during copytree.  M2009 only
        # handles an orphan that was already completely sealed.  A partial
        # copy remains at the fixed name and every manual-resume call rejects
        # it again, so the sole recovery path is still stranded.
        partial_target = root / "partial_result"
        partial_import = Path(str(partial_target) + ".m2009_import_work")
        partial_import.mkdir()
        (partial_import / "result.json").write_text("{\"partial\":true}\n")
        partial_plan = dict(expected)
        first = expect_failure(lambda: M._promote_result_resumable(
            root / "unused_source", partial_target, partial_plan))
        second = expect_failure(lambda: M._promote_result_resumable(
            root / "unused_source", partial_target, partial_plan))
        require(first and second and partial_import.is_dir() and
                not partial_target.exists(),
                "partial-copy orphan behavior did not reproduce")
        findings["partial_copy_orphan_permanently_strands_resume"] = True

        # Both the campaign-total and resume-leg counts are now explicit.
        original_result = M.RESULT
        original_publish = M._publish_sealed_work
        M._publish_sealed_work = lambda work, target, label: {"synthetic": True}
        try:
            M.RESULT = root / "result_resumed"
            resumed = M._publish_result(
                "a" * 64, "b" * 64, {"manifest_sha256": "c" * 64},
                {}, [], {}, True)
            M.RESULT = root / "result_initial"
            initial = M._publish_result(
                "a" * 64, "b" * 64, {"manifest_sha256": "c" * 64},
                {}, [], {}, False)
        finally:
            M.RESULT = original_result
            M._publish_sealed_work = original_publish
        require(resumed["campaign_archive_open_count"] == 1 and
                resumed["resume_leg_archive_open_count"] == 0 and
                initial["campaign_archive_open_count"] == 1 and
                initial["resume_leg_archive_open_count"] is None,
                "archive-open accounting drift")
        positive["campaign_and_resume_archive_counts_separated"] = True

        # Raw bytes, decoded text, hash and exact keys are mutually bound.
        rows = process_rows()
        classified = M.classify_process_records(rows)
        require(tuple(item["role"] for item in classified) ==
                M.EXPECTED_ROLES, "five-role classification drift")
        bad = [dict(item) for item in rows]
        bad[0]["cmdline_sha256"] = "0" * 64
        expect_failure(lambda: M.classify_process_records(bad))
        bad = [dict(item) for item in rows]
        bad[0]["cmdline_raw_hex"] = b"different\0".hex()
        expect_failure(lambda: M.classify_process_records(bad))
        bad = [dict(item) for item in rows]
        bad[0]["cmdline_text"] = "different"
        expect_failure(lambda: M.classify_process_records(bad))
        bad = [dict(item) for item in rows]
        bad[0]["unknown"] = True
        expect_failure(lambda: M.classify_process_records(bad))
        positive["process_exact_keys_raw_hex_text_sha_bound"] = True

        # M2007 required a second PID/starttime read after classification and
        # before publication.  Drive the capture seam with records that would
        # change identity on a second read.  Success after exactly five calls
        # demonstrates that M2009 still publishes from a single stale read.
        original_prestop = M.PRESTOP
        original_strict_json = M.B.strict_json
        original_proc_record = M._proc_record
        original_publish = M._publish_sealed_work
        capture_target = root / "capture"
        calls = {"count": 0}
        by_pid = dict((item["pid"], item) for item in rows)

        def changing_record(pid):
            calls["count"] += 1
            item = dict(by_pid[pid])
            if calls["count"] > 5:
                item["starttime_ticks"] += 1
            return item

        M.PRESTOP = capture_target
        M.B.strict_json = lambda path: {
            "status": M.REVIEW_STATUS, "identity": M.identity(),
            "authorization": {"process_identity_capture": 1,
                "m2011_release_authoring": 1, "archive_open": 0,
                "merge": 0, "reducer": 0, "payload_opens": 0,
                "gpu_runs": 0, "eda_runs": 0}}
        M._proc_record = changing_record
        M._publish_sealed_work = lambda work, target, label: {"synthetic": True}
        try:
            capture = M.capture_process_identity([10, 11, 12, 13, 14])
        finally:
            M.PRESTOP = original_prestop
            M.B.strict_json = original_strict_json
            M._proc_record = original_proc_record
            M._publish_sealed_work = original_publish
        require(capture["captured_all_five_live"] is True and
                calls["count"] == 5,
                "expected stale single-read process capture not reproduced")
        findings["processes_not_reread_before_live_receipt_publish"] = True

    source_text = SOURCE.read_text(encoding="utf-8")
    require("Path.rename(" not in source_text and ".rename(" not in source_text,
            "check-then-rename publication remains")
    require("single_fd_verify_and_extract" in source_text and
            "verify_all_remote_before_mutation" in source_text and
            "M2006.verify_local_shard" in source_text and
            "M2006.exact_receipt_core" in source_text,
            "M2006 inherited closure bindings drift")
    positive["m2006_single_fd_all_before_mutate_m1706_exact_minus_rss_bound"] = True

    print(json.dumps({
        "status": "PASS_M2010_INDEPENDENT_HAMMER__M2009_FAILINGS_REPRODUCED",
        "positive_checks": positive, "findings": findings,
        "production_process_capture": False,
        "production_archive_open": False,
        "canonical_namespace_mutated": False,
        "merge_executed": False, "reducer_executed": False,
        "payload_opened": False, "gpu": False, "eda": False
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
