#!/usr/bin/env python3
"""Process-isolated continuation of the fresh M1681 D0 shard tail.

This launcher does not create a new decoder algorithm, shard schema, or shard
execution budget.  It consumes the 1,137 still-fresh ordinals (7563..8699)
from the already sealed M1706 campaign through the exact M1704 authority
adapter.  M1705 permits this only when concurrency is process isolated and the
launcher is separately reviewed, so the three workers use ``spawn`` and each
worker invokes M1704 serially for a fixed stride of 379 ordinals.

Ordinals 7560..7562 are deliberately excluded: their attempts are already
consumed and their empty work directories require a distinct manual-recovery
authority.  Reduction is also excluded.  The CLI is fail closed; execution is
available only after a sealed M2091 review and M2092 release exist.
"""
from __future__ import print_function

import argparse
import ctypes
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import stat
import sys
import traceback


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = Path(__file__).resolve()
CONTRACT = HW / (
    "contracts/m2090_ep34_decoder_d0_fresh_tail_process_isolated_"
    "resume_source_contract_r1_20260904.json")
M1704_SOURCE = HERE / (
    "build_m1704_ep34_decoder_d0_execution_authority_adapter_source.py")
M1705_REVIEW = HW / (
    "reviews/m1705_m1704_ep34_decoder_d0_execution_authority_adapter_"
    "source_independent_review_r1_20260901")
M1706_RELEASE = HW / (
    "contracts/m1706_m1705_m1704_ep34_decoder_d0_8700_shard_"
    "campaign_release_r1_20260901.json")
FUTURE_REVIEW = HW / (
    "reviews/m2091_m2090_ep34_decoder_d0_fresh_tail_process_isolated_"
    "resume_source_hammer_r1_20260904")
FUTURE_RELEASE = HW / (
    "contracts/m2092_m2091_m2090_ep34_decoder_d0_fresh_tail_"
    "process_isolated_resume_release_r1_20260904.json")
ATTEMPT = HW / (
    "results/.m2090_ep34_decoder_d0_fresh_tail_resume_attempt_consumed")
RESULT = HW / (
    "results/m2090_ep34_decoder_d0_fresh_tail_resume_r1_20260904")
WORK = HW / (
    "results/.m2090_ep34_decoder_d0_fresh_tail_resume_r1_20260904.work")
FAILURE = HW / (
    "results/m2090_ep34_decoder_d0_fresh_tail_resume_r1_20260904."
    "failed_no_retry")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

SCHEMA = "m2090_ep34_decoder_d0_fresh_tail_process_isolated_resume_r1_v1"
STATUS = "SOURCE_ONLY__M2091_REVIEW_AND_M2092_RELEASE_REQUIRED"
REVIEW_STATUS = (
    "PASS_M2091_M2090_DECODER_D0_FRESH_TAIL_PROCESS_ISOLATED_RESUME_"
    "SOURCE__AUTHORIZE_M2092_RELEASE_ONLY")
RELEASE_SCHEMA = (
    "m2092_m2091_m2090_ep34_decoder_d0_fresh_tail_process_isolated_"
    "resume_release_r1_v1")
RELEASE_STATUS = (
    "AUTHORIZE_M2090_ONE_DETACHED_THREE_PROCESS_FRESH_TAIL_RESUME")
M1704_SHA256 = (
    "abc052025a5ed6975fa9d200581b182fa723a7b4b0330bc341b4bc2712204820")
M1705_REVIEW_SHA256 = (
    "c10f9b2c2eef2784fccd22a01c320a9c2ea45df17dc63a9c2cc3fafae9b28f1b")
M1705_MANIFEST_SHA256 = (
    "dbd50234eac3f82b7ba832bb3bcc82ab52d5739a512983cca11d2b8fdde84d38")
M1705_OUTER_SHA256 = (
    "616624711b3735d3f2ee30334bab506119ffbd2630652c52741c9050b8498bc9")
M1706_SHA256 = (
    "43c7096fe90263abf7593d41c3222675bc9153ca4529436b3a57405c550fe7e0")
DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
START = 7563
STOP = 8700
WORKERS = 3
PER_WORKER = 379
AT_FDCWD = -100
RENAME_NOREPLACE = 1


class M2090Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M2090Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def load_m1704():
    regular_exact(M1704_SOURCE, M1704_SHA256, "exact M1704 source")
    spec = importlib.util.spec_from_file_location("m2090_exact_m1704",
                                                  str(M1704_SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot import exact M1704")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.SCHEMA ==
            "m1704_ep34_decoder_d0_execution_authority_adapter_source_r1_v1",
            "M1704 schema drift")
    return module


M1704 = load_m1704()
B = M1704.B


def _identity():
    return {
        "source_sha256": sha256(SOURCE),
        "contract_sha256": sha256(CONTRACT),
        "m1704_source_sha256": M1704_SHA256,
        "m1705_review_sha256": M1705_REVIEW_SHA256,
        "m1705_manifest_sha256": M1705_MANIFEST_SHA256,
        "m1705_outer_file_sha256": M1705_OUTER_SHA256,
        "m1706_release_sha256": M1706_SHA256,
        "checkpoint_sha256": B.G.CHECKPOINT_SHA256,
        "resource_manifest_sha256": B.G.RESOURCE_SHA256,
        "docs359_sha256": DOCS359_SHA256,
    }


def _absent(path, label):
    require(not os.path.lexists(str(path)), label + " exists")


def _rename_noreplace(source, target):
    libc = ctypes.CDLL(None, use_errno=True)
    require(hasattr(libc, "renameat2"), "renameat2 unavailable")
    result = libc.renameat2(
        ctypes.c_int(AT_FDCWD), ctypes.c_char_p(os.fsencode(str(source))),
        ctypes.c_int(AT_FDCWD), ctypes.c_char_p(os.fsencode(str(target))),
        ctypes.c_uint(RENAME_NOREPLACE))
    if result != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), str(target))


def _verify_fixed_authority():
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    B.verify_double_sealed_file(CONTRACT, "M2090 source contract")
    regular_exact(M1704_SOURCE, M1704_SHA256, "exact M1704 source")
    B.verify_sealed_tree(M1705_REVIEW, M1705_REVIEW_SHA256,
        M1705_MANIFEST_SHA256, M1705_OUTER_SHA256, False, "M1705")
    regular_exact(M1706_RELEASE, M1706_SHA256, "exact M1706 release")
    B.verify_double_sealed_file(M1706_RELEASE, "M1706 release")
    M1704.validate_future_review_and_release()


def _orphan_exact(ordinal):
    paths = B.namespace_paths(ordinal)
    attempt_mode = paths["attempt"].lstat().st_mode
    require(stat.S_ISREG(attempt_mode) and not paths["attempt"].is_symlink()
            and stat.S_IMODE(attempt_mode) == 0o400,
            "orphan attempt topology drift")
    require(paths["work"].is_dir() and not paths["work"].is_symlink()
            and not any(paths["work"].iterdir()),
            "orphan work must be an empty regular directory")
    require(not os.path.lexists(str(paths["result"])) and
            not os.path.lexists(str(paths["failure"])),
            "orphan result/failure unexpectedly exists")


def validate_current_topology(verify_prefix=True):
    """Prove the completed prefix, three orphans, and fresh tail."""
    _verify_fixed_authority()
    chain = hashlib.sha256()
    if verify_prefix:
        for ordinal in range(START - WORKERS):
            verified = B.verify_sealed_shard(ordinal)
            chain.update((str(ordinal) + ":" +
                verified["seal"]["manifest_sha256"] + "\n").encode("ascii"))
    for ordinal in range(START - WORKERS, START):
        _orphan_exact(ordinal)
    for ordinal in range(START, STOP):
        paths = B.namespace_paths(ordinal)
        require(all(not os.path.lexists(str(path))
                    for path in paths.values()),
                "fresh tail topology drift at ordinal {}".format(ordinal))
    return {
        "verified_prefix_shards": START - WORKERS if verify_prefix else 0,
        "verified_prefix_manifest_chain": chain.hexdigest()
            if verify_prefix else None,
        "preserved_orphan_ordinals": list(range(START - WORKERS, START)),
        "fresh_tail_start": START,
        "fresh_tail_stop_exclusive": STOP,
        "fresh_tail_shards": STOP - START,
    }


def _validate_future_gate():
    seal = B.verify_sealed_tree(FUTURE_REVIEW,
        allow_ignored_pycache=False, label="M2091")
    review = B.strict_json(FUTURE_REVIEW / "review.json")
    require(review.get("status") == REVIEW_STATUS and
            review.get("score_over_100", 0) >= 95 and
            review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0}
            and review.get("identity") == _identity() and
            review.get("authorization") == {
                "m2092_release_authoring": 1,
                "resume_execution": 0,
                "shard_execution": 0,
                "reducer_execution": 0,
            }, "M2091 review authority drift")
    release_sha = B.verify_double_sealed_file(FUTURE_RELEASE, "M2092")
    release = B.strict_json(FUTURE_RELEASE)
    identity = dict(_identity(),
        review_sha256=sha256(FUTURE_REVIEW / "review.json"),
        review_manifest_sha256=seal["manifest_sha256"],
        review_outer_file_sha256=seal["outer_file_sha256"])
    require(release.get("schema") == RELEASE_SCHEMA and
            release.get("status") == RELEASE_STATUS and
            release.get("identity") == identity and
            release.get("authorization") == {
                "detached_launcher_runs": 1,
                "process_workers": WORKERS,
                "m1706_remaining_shard_runs_consumed": STOP - START,
                "new_shard_runs": 0,
                "new_m1681_shard_attempt_budget": 0,
                "outer_orchestration_attempt_writes": 1,
                "automatic_retry": False,
                "reducer_runs": 0,
                "gpu_runs": 0,
                "eda_runs": 0,
            } and release.get("fixed_partition") == {
                "start": START, "stop_exclusive": STOP,
                "workers": WORKERS, "stride": WORKERS,
                "shards_per_worker": PER_WORKER,
            } and release.get("claim_boundary") == {
                "orchestration_only": True,
                "exact_m1704_m1681_shard_execution": True,
                "full_d0_result": False,
                "full_decoder": False,
                "system_speedup": False,
                "paper_result": False,
            }, "M2092 release drift")
    return release_sha


def _validate_detached_launch():
    """Require the production process to be its own non-interactive session."""
    require(os.environ.get("M2090_DETACHED_LAUNCH") == "1",
            "explicit detached-launch token missing")
    require(os.getsid(0) == os.getpid(),
            "production launcher must be a setsid session leader")
    require(not any(os.isatty(descriptor) for descriptor in (0, 1, 2)),
            "production launcher must not retain a terminal")


def _consume_attempt(release_sha, topology):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(ATTEMPT), flags, 0o400)
    try:
        row = {"schema": SCHEMA, "source_sha256": sha256(SOURCE),
            "release_sha256": release_sha, "pid": os.getpid(),
            "fixed_partition": {"start": START, "stop_exclusive": STOP,
                "workers": WORKERS, "stride": WORKERS,
                "shards_per_worker": PER_WORKER},
            "topology": topology, "automatic_retry": False,
            "payload_opened_before_attempt": False}
        payload = (json.dumps(row, sort_keys=True,
                    separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(str(ATTEMPT), 0o400)
    return sha256(ATTEMPT)


def _worker(worker_id, output_queue):
    completed = 0
    first = START + worker_id
    last = None
    try:
        require(0 <= worker_id < WORKERS, "worker id out of range")
        for ordinal in range(first, STOP, WORKERS):
            M1704._run_authorized_shard(ordinal)
            completed += 1
            last = ordinal
        require(completed == PER_WORKER and last == STOP - WORKERS + worker_id,
                "worker partition cardinality drift")
        output_queue.put({"worker_id": worker_id, "pid": os.getpid(),
            "first": first, "last": last, "completed": completed,
            "status": "PASS"})
    except BaseException as error:
        output_queue.put({"worker_id": worker_id, "pid": os.getpid(),
            "first": first, "last": last, "completed": completed,
            "status": "FAIL", "error": repr(error),
            "traceback": traceback.format_exc()})
        raise


def execute():
    _validate_detached_launch()
    release_sha = _validate_future_gate()
    for path, label in ((ATTEMPT, "attempt"), (RESULT, "result"),
                        (WORK, "work"), (FAILURE, "failure")):
        _absent(path, label)
    topology = validate_current_topology(True)
    attempt_sha = _consume_attempt(release_sha, topology)
    WORK.mkdir(mode=0o700)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=_worker, args=(worker, queue))
                 for worker in range(WORKERS)]
    published = False
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        rows = [queue.get(timeout=5) for _unused in range(WORKERS)]
        rows.sort(key=lambda row: row["worker_id"])
        require(all(process.exitcode == 0 for process in processes),
                "one or more workers failed")
        require(all(row["status"] == "PASS" and
                    row["completed"] == PER_WORKER for row in rows),
                "worker receipt drift")
        require(sum(row["completed"] for row in rows) == STOP - START,
                "completed tail cardinality drift")
        for ordinal in range(START, STOP):
            B.verify_sealed_shard(ordinal)
        receipt = {"schema": SCHEMA,
            "status": "PASS_M2090_FRESH_TAIL_1137_SHARDS_THREE_PROCESS",
            "source_sha256": sha256(SOURCE),
            "release_sha256": release_sha,
            "attempt_sha256": attempt_sha,
            "workers": rows, "completed_shards": STOP - START,
            "ordinal_start": START, "ordinal_stop_exclusive": STOP,
            "preserved_orphans": list(range(START - WORKERS, START)),
            "exact_m1704_m1681_shard_execution": True,
            "reducer_executed": False, "full_d0_result": False,
            "full_decoder": False, "system_speedup": False,
            "paper_result": False, "independent_result_hammer_pending": True}
        (WORK / "result.json").write_text(json.dumps(
            receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        B.seal_work_tree(WORK)
        B.verify_sealed_tree(WORK, allow_ignored_pycache=False,
                             label="M2090 result work")
        _rename_noreplace(WORK, RESULT)
        published = True
        return receipt
    except BaseException as error:
        if WORK.is_dir() and not os.path.lexists(str(FAILURE)):
            (WORK / "failure.json").write_text(json.dumps({
                "schema": SCHEMA, "status": "FAILED_NO_RETRY",
                "error": repr(error), "traceback": traceback.format_exc(),
                "worker_exitcodes": [process.exitcode for process in processes],
                "automatic_retry": False}, indent=2, sort_keys=True,
                allow_nan=False) + "\n", encoding="utf-8")
            B.seal_work_tree(WORK)
            _rename_noreplace(WORK, FAILURE)
        raise
    finally:
        require(published or not WORK.exists(),
                "unpublished work directory remains")


def describe():
    return {"schema": SCHEMA, "status": STATUS,
        "fixed_partition": {"start": START, "stop_exclusive": STOP,
            "workers": WORKERS, "stride": WORKERS,
            "shards_per_worker": PER_WORKER},
        "preserved_orphan_ordinals": list(range(START - WORKERS, START)),
        "execution": "exact M1704 adapter -> exact M1681 shard implementation",
        "claim_boundary": {"source_only": True,
            "new_algorithm": False, "new_shard_budget": False,
            "automatic_retry": False, "reducer_execution": False,
            "full_d0_result": False, "full_decoder": False,
            "system_speedup": False, "paper_result": False}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        output = execute()
    elif args.preflight:
        output = {"schema": SCHEMA,
            "status": "PASS_M2090_SOURCE_PREFLIGHT_NO_EXECUTION",
            "identity": _identity(),
            "topology": validate_current_topology(True),
            "runtime_namespaces_absent": all(not os.path.lexists(str(path))
                for path in (ATTEMPT, RESULT, WORK, FAILURE)),
            "execution": False}
    else:
        output = describe()
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
