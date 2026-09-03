#!/usr/bin/env python3
"""Different-author, source-only transaction hammer for M2012.

All mutations are confined to TemporaryDirectory namespaces.  The production
process set, archive, canonical shard namespace, reducer, payloads, GPU, and
EDA tools are never touched.
"""
from __future__ import print_function

import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / (
    "system_simulator/scripts/"
    "build_m2012_ep34_decoder_d0_recoverable_noreplace_successor_source.py")
EXPECTED_SOURCE_SHA256 = (
    "437faf4278acd9701ab6495fabfb08eaafba22065fb8aa63cc4194589d1de872")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


require(sha256(SOURCE) == EXPECTED_SOURCE_SHA256, "M2012 source SHA drift")
spec = importlib.util.spec_from_file_location("m2013_target", str(SOURCE))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def expect_failure(action):
    try:
        action()
    except (M.M2012Error, RuntimeError, OSError, ValueError, KeyError,
            AssertionError):
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


def make_sealed_tree(path, row=None):
    path = Path(path)
    path.mkdir(parents=True, mode=0o700)
    row = exact_receipt() if row is None else row
    (path / "result.json").write_text(json.dumps(
        row, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    M.B.seal_work_tree(path)
    seal = M.B.verify_sealed_tree(
        path, allow_ignored_pycache=False, label="synthetic sealed tree")
    return row, seal


def plan_row(path, row, seal):
    return {"ordinal": row["shard_ordinal"],
        "attempt_sha256": row["attempt_sha256"],
        "result_json_sha256": sha256(Path(path) / "result.json"),
        "manifest_sha256": seal["manifest_sha256"],
        "deterministic_core_sha256": M.M2006.canonical_sha(
            M.M2006.exact_receipt_core(row))}


def process_record(pid, ppid, raw, start=None):
    return {"pid": pid, "ppid": ppid,
        "starttime_ticks": pid * 10 if start is None else start,
        "cmdline_raw_hex": raw.hex(),
        "cmdline_sha256": hashlib.sha256(raw).hexdigest(),
        "cmdline_text": raw.replace(b"\0", b" ").decode().strip(),
        "cwd": str(M.HW.parent)}


def process_records():
    launcher = (b"bash build_m1704_ep34_decoder_d0_execution_authority_"
                b"adapter_source.py range(1,8700)\0")
    return [process_record(10, 1, launcher),
            process_record(11, 10, b"python3\0-\0"),
            process_record(12, 11, b"python3\0-\0"),
            process_record(13, 11, b"python3\0-\0"),
            process_record(14, 11, b"python3\0-\0")]


def main():
    positive = {}
    findings = {}
    with tempfile.TemporaryDirectory() as root_text:
        root = Path(root_text)
        old_quarantine = M.QUARANTINE_ROOT
        M.QUARANTINE_ROOT = root / "quarantine"
        try:
            # Immutable source and plan used by all recovery crash points.
            staged = root / "staged"
            row, seal = make_sealed_tree(staged)
            expected = plan_row(staged, row, seal)

            # Interruption before copy: a fresh exclusive copy must publish.
            before_target = root / "before_copy"
            M._promote_result_resumable(staged, before_target, expected)
            M._verified_tree_matches_plan(
                before_target, expected, "before-copy recovery result")
            positive["interruption_before_copy_recovers"] = True

            # Mid-copy with a strict subset of allowed names is quarantined;
            # a fresh verified copy then publishes, retaining evidence.
            subset_target = root / "mid_copy_subset"
            subset_work = Path(str(subset_target) + ".m2012_import_work")
            subset_work.mkdir()
            (subset_work / "result.json").write_text("{\"partial\":true}\n")
            M._promote_result_resumable(staged, subset_target, expected)
            require(subset_target.is_dir() and not subset_work.exists(),
                    "strict-subset crash did not recover")
            subset_quarantine = sorted(M.QUARANTINE_ROOT.iterdir())
            require(len(subset_quarantine) == 1 and
                    (subset_quarantine[0] / "result.json").exists(),
                    "strict-subset evidence was not retained")
            positive["mid_copy_strict_subset_quarantined_and_recovers"] = True

            # A preoccupied quarantine slot, including a symlink, cannot be
            # overwritten.  Recovery must advance to the next no-replace slot.
            preoccupied_target = root / "preoccupied"
            preoccupied_work = Path(
                str(preoccupied_target) + ".m2012_import_work")
            preoccupied_work.mkdir()
            (preoccupied_work / "SHA256SUMS").write_text("partial\n")
            stem = preoccupied_target.name + ".partial_import_work.0000"
            marker = root / "marker"
            marker.write_text("preserve\n")
            (M.QUARANTINE_ROOT / stem).symlink_to(marker)
            M._promote_result_resumable(
                staged, preoccupied_target, expected)
            require((M.QUARANTINE_ROOT / stem).is_symlink() and
                    marker.read_text() == "preserve\n" and
                    (M.QUARANTINE_ROOT /
                     (preoccupied_target.name +
                      ".partial_import_work.0001")).is_dir(),
                    "preoccupied quarantine slot was not preserved")
            positive["preoccupied_quarantine_preserved"] = True

            # A complete and valid orphan is the after-seal/before-publish
            # state.  It must promote without opening/copying source again.
            sealed_target = root / "after_seal"
            sealed_work = Path(str(sealed_target) + ".m2012_import_work")
            shutil.copytree(str(staged), str(sealed_work))
            M._promote_result_resumable(
                root / "must_not_be_opened", sealed_target, expected)
            require(sealed_target.is_dir() and not sealed_work.exists(),
                    "sealed plan-matching orphan did not promote")
            positive["after_seal_before_publish_orphan_promotes"] = True

            # Simulate interruption at the final no-replace publication.  The
            # complete work tree remains and a later call promotes it.
            publish_target = root / "before_publish"
            original_rename = M.rename_noreplace
            hit = {"done": False}

            def interrupt_final(source, target):
                if Path(target) == publish_target and not hit["done"]:
                    hit["done"] = True
                    raise OSError(errno.EIO, "synthetic interruption")
                return original_rename(source, target)

            M.rename_noreplace = interrupt_final
            try:
                expect_failure(lambda: M._promote_result_resumable(
                    staged, publish_target, expected))
            finally:
                M.rename_noreplace = original_rename
            publish_work = Path(
                str(publish_target) + ".m2012_import_work")
            require(publish_work.is_dir() and not publish_target.exists(),
                    "pre-publish interruption did not preserve work")
            M._promote_result_resumable(staged, publish_target, expected)
            positive["interruption_immediately_before_publish_recovers"] = True

            # A canonical target introduced only at publication must win;
            # neither it nor the verified import-work may be overwritten.
            raced_target = root / "raced_target"
            original_rename = M.rename_noreplace
            raced = {"done": False}

            def inject_target(source, target):
                if Path(target) == raced_target and not raced["done"]:
                    raced["done"] = True
                    raced_target.mkdir()
                    (raced_target / "attacker").write_text("preserve\n")
                return original_rename(source, target)

            M.rename_noreplace = inject_target
            try:
                expect_failure(lambda: M._promote_result_resumable(
                    staged, raced_target, expected))
            finally:
                M.rename_noreplace = original_rename
            raced_work = Path(str(raced_target) + ".m2012_import_work")
            require((raced_target / "attacker").read_text() == "preserve\n"
                    and raced_work.is_dir(),
                    "raced target or import evidence was overwritten")
            positive["canonical_target_race_preserves_both_trees"] = True

            # A validly sealed but plan-mismatched complete orphan remains a
            # hard failure and must be preserved.
            mismatch_target = root / "mismatch"
            mismatch_work = Path(str(mismatch_target) +
                                 ".m2012_import_work")
            wrong_row, wrong_seal = make_sealed_tree(
                mismatch_work, exact_receipt(4501))
            wrong_plan = plan_row(mismatch_work, wrong_row, wrong_seal)
            wrong_plan["deterministic_core_sha256"] = "f" * 64
            expect_failure(lambda: M._promote_result_resumable(
                staged, mismatch_target, wrong_plan))
            require(mismatch_work.is_dir() and not mismatch_target.exists(),
                    "valid sealed mismatch was not preserved/rejected")
            positive["valid_sealed_plan_mismatch_rejected"] = True

            # Alien names, symlinks and special files are rejected in place.
            for kind in ("alien", "symlink", "special"):
                target = root / ("malicious_" + kind)
                work = Path(str(target) + ".m2012_import_work")
                work.mkdir()
                if kind == "alien":
                    (work / "unexpected").write_text("x\n")
                elif kind == "symlink":
                    (work / "result.json").symlink_to(staged / "result.json")
                else:
                    os.mkfifo(str(work / "result.json"))
                expect_failure(lambda w=work, t=target:
                               M._promote_result_resumable(
                                   staged, t, expected))
                require(os.path.lexists(str(work)) and not target.exists(),
                        kind + " orphan was not preserved/rejected")
            positive["alien_symlink_special_orphans_rejected"] = True

            # Critical normal copy boundary: copytree may have created all
            # three allowed names but still be writing the final seal file.
            # M2012 calls this topology 'complete', treats it like malicious
            # sealed evidence, and permanently rejects instead of quarantining
            # and restarting.  Reproduce two consecutive failed resumes.
            truncated_target = root / "three_names_truncated"
            truncated_work = Path(
                str(truncated_target) + ".m2012_import_work")
            truncated_work.mkdir()
            shutil.copyfile(str(staged / "result.json"),
                            str(truncated_work / "result.json"))
            shutil.copyfile(str(staged / "SHA256SUMS"),
                            str(truncated_work / "SHA256SUMS"))
            (truncated_work / "SHA256SUMS.seal.sha256").write_bytes(b"0")
            require(M.inspect_import_work_topology(truncated_work) ==
                    "complete", "three-name boundary was not classified")
            first = expect_failure(lambda: M._promote_result_resumable(
                staged, truncated_target, expected))
            second = expect_failure(lambda: M._promote_result_resumable(
                staged, truncated_target, expected))
            require(first and second and truncated_work.is_dir() and
                    not truncated_target.exists(),
                    "three-name truncated orphan did not strand recovery")
            findings[
                "three_allowed_names_with_truncated_seal_strand_resume"] = True
        finally:
            M.QUARANTINE_ROOT = old_quarantine

        # Exact second reads reject stable-PID reuse and exit.
        records = process_records()
        initial = M.classify_process_records(records)
        by_pid = dict((row["pid"], dict(row)) for row in records)
        changed = dict((pid, dict(row)) for pid, row in by_pid.items())
        changed[10] = process_record(10, 1,
            bytes.fromhex(by_pid[10]["cmdline_raw_hex"]), start=99999)
        expect_failure(lambda: M.reread_process_records(
            initial, lambda pid: changed[pid]))

        def exited(pid):
            if pid == 10:
                raise FileNotFoundError("synthetic exited PID")
            return by_pid[pid]

        expect_failure(lambda: M.reread_process_records(initial, exited))
        positive["pid_reuse_and_exit_rejected_on_second_read"] = True

        # Control-plane attack: capture_process_identity accepts a plain,
        # unsealed review.json and does not inspect score/severity.  A forged
        # document can consume the no-replace PRESTOP namespace before an
        # actual >=95, zero-finding review exists.
        old_review = M.FUTURE_REVIEW
        old_prestop = M.PRESTOP
        old_reader = M._proc_record
        try:
            fake_review = root / "unsealed_fake_review"
            fake_review.mkdir()
            fake = {"status": M.REVIEW_STATUS, "identity": M.identity(),
                "score_over_100": 0,
                "severity_counts": {"p0": 9, "p1": 9, "p2": 9},
                "authorization": {"process_identity_capture": 1,
                    "m2014_release_authoring": 1, "archive_open": 0,
                    "merge": 0, "reducer": 0, "payload_opens": 0,
                    "gpu_runs": 0, "eda_runs": 0}}
            (fake_review / "review.json").write_text(json.dumps(
                fake, indent=2, sort_keys=True, allow_nan=False) + "\n")
            M.FUTURE_REVIEW = fake_review
            M.PRESTOP = root / "forged_prestop"
            M._proc_record = lambda pid: dict(by_pid[pid])
            M.capture_process_identity([10, 11, 12, 13, 14])
            require(M.PRESTOP.is_dir() and
                    not (fake_review / "SHA256SUMS").exists() and
                    not (fake_review / "SHA256SUMS.seal.sha256").exists(),
                    "unsealed review bypass did not reproduce")
            findings["unsealed_zero_score_review_authorizes_capture"] = True
        finally:
            M.FUTURE_REVIEW = old_review
            M.PRESTOP = old_prestop
            M._proc_record = old_reader

        # The wrapper is dynamically bound to M2012's production constants at
        # import time; ensure the inherited P runtime did not retain M2009 paths.
        for name in ("PRESTOP", "ATTEMPT", "PLAN", "RESULT", "FAILURE",
                     "STAGING_PARENT", "SCHEMA", "REVIEW_STATUS",
                     "RELEASE_SCHEMA", "RELEASE_STATUS"):
            require(getattr(M.P, name) == getattr(M, name),
                    "wrapper runtime binding drift: " + name)
        require(M.P._promote_result_resumable is M._promote_result_resumable
                and M.P.identity is M.identity,
                "wrapper callable/identity binding drift")
        positive["m2009_wrapper_runtime_and_identity_bound_to_m2012"] = True

    output = {"status": "FAIL_M2013_INDEPENDENT_SOURCE_HAMMER",
        "positive": positive, "findings": findings,
        "production_process_capture": False, "archive_open": False,
        "canonical_payload_open": False, "merge": False, "reducer": False,
        "gpu": False, "eda": False}
    require(set(findings) == {
        "three_allowed_names_with_truncated_seal_strand_resume",
        "unsealed_zero_score_review_authorizes_capture"},
        "finding set drift")
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
