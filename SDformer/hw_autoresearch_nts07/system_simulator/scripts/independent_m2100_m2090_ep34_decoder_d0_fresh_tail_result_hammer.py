#!/usr/bin/env python3
"""Independent result hammer for the M2090 1,137-shard fresh tail.

``--static`` verifies only immutable source/authority identities and never
opens an M2090 runtime namespace.  ``--publish`` is a read-only consumer of an
already-published M2090 success: it strongly verifies every M1681 shard,
recomputes all integer metric invariants, and atomically publishes a
double-sealed M2100 review.  It never opens decoder payload, executes a shard
or reducer, or invokes GPU/EDA work.
"""
from __future__ import print_function

import argparse
from collections import OrderedDict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import sys


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
CHECKER = Path(__file__).resolve()
SOURCE = HERE / (
    "run_m2090_ep34_decoder_d0_fresh_tail_process_isolated_resume.py")
CONTRACT = HW / (
    "contracts/m2090_ep34_decoder_d0_fresh_tail_process_isolated_"
    "resume_source_contract_r1_20260904.json")
M2091 = HW / (
    "reviews/m2091_m2090_ep34_decoder_d0_fresh_tail_process_isolated_"
    "resume_source_hammer_r1_20260904")
M2092 = HW / (
    "contracts/m2092_m2091_m2090_ep34_decoder_d0_fresh_tail_"
    "process_isolated_resume_release_r1_20260904.json")
M1706 = HW / (
    "contracts/m1706_m1705_m1704_ep34_decoder_d0_8700_shard_"
    "campaign_release_r1_20260901.json")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
ATTEMPT = HW / (
    "results/.m2090_ep34_decoder_d0_fresh_tail_resume_attempt_consumed")
RESULT = HW / (
    "results/m2090_ep34_decoder_d0_fresh_tail_resume_r1_20260904")
WORK = HW / (
    "results/.m2090_ep34_decoder_d0_fresh_tail_resume_r1_20260904.work")
FAILURE = HW / (
    "results/m2090_ep34_decoder_d0_fresh_tail_resume_r1_20260904."
    "failed_no_retry")
REVIEW = HW / (
    "reviews/m2100_m2090_ep34_decoder_d0_fresh_tail_result_hammer_"
    "r1_20260904")

SOURCE_SHA256 = (
    "23b5c41ac50a13de8a3c2e7e5f46c666de3ed7326f629c6d40fc4b4f577017c7")
CONTRACT_SHA256 = (
    "a61e03a9f4e3d25e0ac82a5c38d1cdf4fc63403b8d85fa18c36ea2784637a6ab")
M2091_REVIEW_SHA256 = (
    "37bd79c50ca33807b38372f10cf80e67ae1f894813298ee98b2f117d1aa1630b")
M2091_MANIFEST_SHA256 = (
    "5d4ca74abc1793668d5569f3fbc6999fb56fc843a582b7f78b3afa93509d03da")
M2091_OUTER_SHA256 = (
    "51849581c3007a1dba96b83c0ede18932ebd64de4ebc2a64f9f1dd92227d9976")
M2092_SHA256 = (
    "9e88449cc0e41b20cc151225381bb8cb3f17f665778b8371c707cacb8132cc09")
M1706_SHA256 = (
    "43c7096fe90263abf7593d41c3222675bc9153ca4529436b3a57405c550fe7e0")
M1681_SOURCE_SHA256 = (
    "006535679b38e2aa207fadde05e9207d2e72dae0464315dceea4a3c96da77a6f")
DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
START = 7563
STOP = 8700
WORKERS = 3
PER_WORKER = 379
SCHEMA = "m2100_m2090_ep34_decoder_d0_fresh_tail_result_hammer_r1_v1"
STATUS = (
    "PASS_M2100_M2090_FRESH_TAIL_1137_SHARDS_RESULT__"
    "D0_REDUCER_INPUT_ONLY")


class M2100Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M2100Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path, expected, label, mode=None):
    path = Path(path)
    current = path.lstat().st_mode
    require(stat.S_ISREG(current) and not path.is_symlink(),
            label + " must be a regular non-symlink")
    if mode is not None:
        require(stat.S_IMODE(current) == mode, label + " mode drift")
    require(sha256(path) == expected, label + " SHA drift")
    return path


def load_source():
    regular_exact(SOURCE, SOURCE_SHA256, "exact M2090 source")
    spec = importlib.util.spec_from_file_location("m2100_exact_m2090", str(SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot import exact M2090 source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.SCHEMA ==
            "m2090_ep34_decoder_d0_fresh_tail_process_isolated_resume_r1_v1",
            "M2090 schema drift")
    return module


M = load_source()
B = M.B


def verify_static():
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    regular_exact(CONTRACT, CONTRACT_SHA256, "M2090 source contract")
    B.verify_double_sealed_file(CONTRACT, "M2090 source contract")
    seal = B.verify_sealed_tree(M2091, M2091_REVIEW_SHA256,
        M2091_MANIFEST_SHA256, M2091_OUTER_SHA256, False, "M2091")
    review = B.strict_json(M2091 / "review.json")
    require(review.get("status") == M.REVIEW_STATUS and
            review.get("score_over_100", 0) >= 95 and
            review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0},
            "M2091 review admission drift")
    regular_exact(M2092, M2092_SHA256, "M2092 release")
    B.verify_double_sealed_file(M2092, "M2092 release")
    regular_exact(M1706, M1706_SHA256, "M1706 release")
    B.verify_double_sealed_file(M1706, "M1706 release")
    require(sha256(B.SOURCE) == M1681_SOURCE_SHA256,
            "exact M1681 source SHA drift")
    require(M._validate_future_gate() == M2092_SHA256,
            "M2091/M2092 semantic authority drift")
    return {"checker_sha256": sha256(CHECKER),
        "m2090_source_sha256": SOURCE_SHA256,
        "m2090_contract_sha256": CONTRACT_SHA256,
        "m2091_review_sha256": M2091_REVIEW_SHA256,
        "m2091_manifest_sha256": seal["manifest_sha256"],
        "m2091_outer_file_sha256": seal["outer_file_sha256"],
        "m2092_release_sha256": M2092_SHA256,
        "m1706_release_sha256": M1706_SHA256,
        "m1681_source_sha256": M1681_SOURCE_SHA256,
        "docs359_sha256": DOCS359_SHA256,
        "runtime_namespace_read": False, "production_execution": False}


def exact_shard_attempt(ordinal):
    return {"schema": B.SCHEMA, "shard_ordinal": ordinal,
        "shard": B.G.shard_descriptor(ordinal),
        "source_sha256": M1681_SOURCE_SHA256,
        "release_sha256": M1706_SHA256,
        "automatic_retry": False,
        "payload_opened_before_attempt": False}


def empty_counter():
    return OrderedDict((name, 0) for name in B.CONFIGS)


def nested_counter():
    return OrderedDict((name, {}) for name in B.CONFIGS)


def add_map(target, source):
    require(type(source) is dict, "metric counter must be a dictionary")
    for key, value in source.items():
        require(type(value) is int and value >= 0,
                "metric counter contains negative/noninteger value")
        target[key] = target.get(key, 0) + value


def verify_shards():
    cycles = empty_counter()
    requests = empty_counter()
    kind_counts = nested_counter()
    byte_counts = nested_counter()
    attempt_chain = hashlib.sha256()
    manifest_chain = hashlib.sha256()
    payload_chain = hashlib.sha256()
    ratio_chain = hashlib.sha256()
    call_counts = {}
    timestep_counts = {}
    sample_counts = {}
    first = None
    last = None
    for ordinal in range(START, STOP):
        paths = B.namespace_paths(ordinal)
        present = dict((key, os.path.lexists(str(path)))
                       for key, path in paths.items())
        require(present == {"result": True, "attempt": True,
                            "work": False, "failure": False},
                "shard sibling topology drift at {}".format(ordinal))
        attempt = paths["attempt"]
        current = attempt.lstat().st_mode
        require(stat.S_ISREG(current) and not attempt.is_symlink() and
                stat.S_IMODE(current) == 0o400,
                "shard attempt is not regular mode 0400 at {}".format(ordinal))
        attempt_row = B.strict_json(attempt)
        require(attempt_row == exact_shard_attempt(ordinal),
                "shard attempt semantics drift at {}".format(ordinal))
        attempt_sha = sha256(attempt)

        verified = B.verify_sealed_shard(ordinal)
        row = B.strict_json(paths["result"] / "result.json")
        require(verified["row"] == row and
                verified["attempt_sha256"] == attempt_sha,
                "strong verifier return drift at {}".format(ordinal))
        B.validate_shard_receipt(row, ordinal, attempt_sha, M1706_SHA256)
        B.validate_metric_bundle(row["metrics"], row["shard"])
        ratios = B.G.validate_three_configuration_metrics(
            row["metrics"], row["shard"])
        require(row["integer_ratio_inputs"] == ratios,
                "integer ratio recompute drift at {}".format(ordinal))
        require(row.get("independent_result_hammer_pending") is True,
                "pre-hammer shard claim drift at {}".format(ordinal))
        require(row["shard"] == B.G.shard_descriptor(ordinal),
                "shard descriptor/ordinal drift")

        for metric in row["metrics"]:
            name = metric["configuration"]
            require(name in cycles, "unknown configuration")
            cycles[name] += metric["total_cycles"]
            requests[name] += metric["request_count"]
            add_map(kind_counts[name], metric["kind_counts"])
            add_map(byte_counts[name], metric["byte_counts"])
        shard = row["shard"]
        call_counts[str(shard["call_ordinal"])] = (
            call_counts.get(str(shard["call_ordinal"]), 0) + 1)
        timestep_counts[str(shard["timestep"])] = (
            timestep_counts.get(str(shard["timestep"]), 0) + 1)
        sample_counts[str(shard["sample_ordinal"])] = (
            sample_counts.get(str(shard["sample_ordinal"]), 0) + 1)
        attempt_chain.update((str(ordinal) + ":" + attempt_sha + "\n").encode("ascii"))
        manifest_chain.update((str(ordinal) + ":" +
            verified["seal"]["manifest_sha256"] + "\n").encode("ascii"))
        payload_chain.update((str(ordinal) + ":" + row["payload_fd_sha256"] +
                              "\n").encode("ascii"))
        ratio_chain.update((str(ordinal) + ":" + B.canonical_sha(ratios) +
                            "\n").encode("ascii"))
        if first is None:
            first = {"ordinal": ordinal, "shard": shard,
                "attempt_sha256": attempt_sha,
                "result_manifest_sha256": verified["seal"]["manifest_sha256"]}
        last = {"ordinal": ordinal, "shard": shard,
            "attempt_sha256": attempt_sha,
            "result_manifest_sha256": verified["seal"]["manifest_sha256"]}

    require(sum(call_counts.values()) == STOP - START and
            sum(timestep_counts.values()) == STOP - START and
            sum(sample_counts.values()) == STOP - START,
            "coverage population conservation drift")
    return {"ordinal_start": START, "ordinal_stop_exclusive": STOP,
        "shards_verified": STOP - START, "attempts_mode_0400": STOP - START,
        "work_residue": 0, "failure_residue": 0,
        "first": first, "last": last,
        "worker_partition_recomputed": [
            {"worker_id": worker, "first": START + worker,
             "last": STOP - WORKERS + worker, "completed": PER_WORKER}
            for worker in range(WORKERS)],
        "coverage": {"call_ordinal_counts": call_counts,
            "timestep_counts": timestep_counts,
            "sample_ordinal_counts": sample_counts},
        "integer_aggregates": {"total_cycles": cycles,
            "request_count": requests, "kind_counts": kind_counts,
            "byte_counts": byte_counts},
        "identity_chains": {
            "attempt_chain_sha256": attempt_chain.hexdigest(),
            "result_manifest_chain_sha256": manifest_chain.hexdigest(),
            "payload_fd_chain_sha256": payload_chain.hexdigest(),
            "integer_ratio_input_chain_sha256": ratio_chain.hexdigest()}}


def verify_result():
    static = verify_static()
    require(not os.path.lexists(str(WORK)) and
            not os.path.lexists(str(FAILURE)),
            "M2090 overall work/failure residue exists")
    attempt_mode = ATTEMPT.lstat().st_mode
    require(stat.S_ISREG(attempt_mode) and not ATTEMPT.is_symlink() and
            stat.S_IMODE(attempt_mode) == 0o400,
            "M2090 overall attempt must be regular mode 0400")
    attempt = B.strict_json(ATTEMPT)
    require(set(attempt) == {"automatic_retry", "fixed_partition", "pid",
            "payload_opened_before_attempt", "release_sha256", "schema",
            "source_sha256", "topology"}, "M2090 overall attempt key drift")
    require(type(attempt["pid"]) is int and attempt["pid"] > 1 and
            attempt["schema"] == M.SCHEMA and
            attempt["source_sha256"] == SOURCE_SHA256 and
            attempt["release_sha256"] == M2092_SHA256 and
            attempt["automatic_retry"] is False and
            attempt["payload_opened_before_attempt"] is False and
            attempt["fixed_partition"] == {"start": START,
                "stop_exclusive": STOP, "workers": WORKERS,
                "stride": WORKERS, "shards_per_worker": PER_WORKER} and
            attempt["topology"] == {"verified_prefix_shards": 7560,
                "verified_prefix_manifest_chain":
                    "3dca65c0c27380017952dec3f036fcfdb8c3e3d59588d1492d4f1547bf4467b2",
                "preserved_orphan_ordinals": [7560, 7561, 7562],
                "fresh_tail_start": START, "fresh_tail_stop_exclusive": STOP,
                "fresh_tail_shards": STOP - START},
            "M2090 overall attempt semantics drift")
    attempt_sha = sha256(ATTEMPT)

    result_seal = B.verify_sealed_tree(
        RESULT, allow_ignored_pycache=False, label="M2090 overall result")
    members = set(path.relative_to(RESULT).as_posix()
        for path in RESULT.rglob("*") if path.is_file() and
        path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    require(members == {"result.json"}, "M2090 result population drift")
    result = B.strict_json(RESULT / "result.json")
    workers = result.get("workers")
    require(type(workers) is list and len(workers) == WORKERS,
            "M2090 worker receipt cardinality drift")
    expected_workers = []
    worker_pids = set()
    for worker, row in enumerate(workers):
        require(type(row) is dict and set(row) == {
            "worker_id", "pid", "first", "last", "completed", "status"},
            "worker receipt keys drift")
        require(type(row["pid"]) is int and row["pid"] > 1,
                "worker pid invalid")
        worker_pids.add(row["pid"])
        expected = {"worker_id": worker, "first": START + worker,
            "last": STOP - WORKERS + worker, "completed": PER_WORKER,
            "status": "PASS"}
        require(dict((key, row[key]) for key in expected) == expected,
                "worker partition/status drift")
        expected_workers.append(dict(expected, pid=row["pid"]))
    require(len(worker_pids) == WORKERS,
            "worker receipts do not identify three distinct processes")
    require(result == {"schema": M.SCHEMA,
        "status": "PASS_M2090_FRESH_TAIL_1137_SHARDS_THREE_PROCESS",
        "source_sha256": SOURCE_SHA256, "release_sha256": M2092_SHA256,
        "attempt_sha256": attempt_sha, "workers": expected_workers,
        "completed_shards": STOP - START, "ordinal_start": START,
        "ordinal_stop_exclusive": STOP,
        "preserved_orphans": [7560, 7561, 7562],
        "exact_m1704_m1681_shard_execution": True,
        "reducer_executed": False, "full_d0_result": False,
        "full_decoder": False, "system_speedup": False,
        "paper_result": False, "independent_result_hammer_pending": True},
        "M2090 overall result receipt drift")

    shards = verify_shards()
    return {"schema": SCHEMA, "status": STATUS, "score_over_100": 100,
        "severity_counts": {"p0": 0, "p1": 0, "p2": 0},
        "production_execution": False, "payload_opened": False,
        "static_identity": static,
        "overall": {"attempt_sha256": attempt_sha,
            "attempt_mode": "0400", "attempt_regular_non_symlink": True,
            "result_json_sha256": sha256(RESULT / "result.json"),
            "result_manifest_sha256": result_seal["manifest_sha256"],
            "result_outer_file_sha256": result_seal["outer_file_sha256"],
            "result_double_sealed": True,
            "worker_processes": WORKERS,
            "shards_per_worker": PER_WORKER,
            "work_and_failure_absent": True},
        "shards": shards,
        "authorization": {"d0_reducer_input": True,
            "reducer_execution": False, "payload_open": False,
            "shard_execution": False, "gpu_runs": 0, "eda_runs": 0},
        "claim_boundary": {"fresh_tail_shards_admitted": STOP - START,
            "d0_reducer_input_only": True, "reducer_executed": False,
            "full_d0_result": False, "full_decoder": False,
            "cycles": False, "traffic": False, "speedup": False,
            "energy": False, "system_speedup": False,
            "paper_result": False}}


def review_markdown(row):
    totals = row["shards"]["integer_aggregates"]["total_cycles"]
    requests = row["shards"]["integer_aggregates"]["request_count"]
    lines = ["# M2100 independent M2090 fresh-tail result hammer", "",
        "Verdict: **PASS (100/100; P0/P1/P2 = 0/0/0).**", "",
        "The already-published M2090 result and all 1,137 fresh-tail shard "
        "receipts (ordinals 7563--8699) were independently re-opened and "
        "strongly verified. The three process partitions contain exactly "
        "379 shards each. Every shard has one regular mode-0400 attempt, one "
        "double-sealed result, and no work or failure sibling.", "",
        "Integer-only aggregate cycle counts: `{}`. Integer-only aggregate "
        "request counts: `{}`.".format(json.dumps(totals, sort_keys=True),
                                        json.dumps(requests, sort_keys=True)),
        "", "Claim ceiling: these 1,137 receipts are admitted only as inputs "
        "to the future exact D0 reducer. This review is not a reducer run, "
        "full-D0/full-decoder result, cycle or traffic comparison, speedup, "
        "energy result, system result, or paper result.", ""]
    return "\n".join(lines)


def publish_review():
    require(not os.path.lexists(str(REVIEW)), "M2100 review already exists")
    row = verify_result()
    work = REVIEW.parent / ("." + REVIEW.name + ".work." + str(os.getpid()))
    require(not os.path.lexists(str(work)), "M2100 review work exists")
    work.mkdir(mode=0o700)
    published = False
    try:
        shutil.copyfile(str(CHECKER), str(work / "independent_hammer.py"))
        (work / "mechanical_checks.json").write_text(json.dumps(
            row, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        (work / "review.json").write_text(json.dumps(
            row, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        (work / "review.md").write_text(review_markdown(row), encoding="utf-8")
        (work / "RUN_COMPLETE.txt").write_text(STATUS + "\n", encoding="ascii")
        B.seal_work_tree(work)
        B.verify_sealed_tree(work, allow_ignored_pycache=False,
                             label="M2100 review work")
        M._rename_noreplace(work, REVIEW)
        published = True
    finally:
        require(published or not work.exists(),
                "unpublished M2100 review work remains")
    return row


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static", action="store_true")
    mode.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    if args.static:
        output = {"schema": SCHEMA,
            "status": "PASS_M2100_CHECKER_STATIC_ONLY__NO_RUNTIME_NAMESPACE_READ",
            "static_identity": verify_static(), "result_read": False,
            "production_execution": False}
    else:
        output = publish_review()
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
