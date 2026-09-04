#!/usr/bin/env python3
"""One-shot publisher for the complete sealed Motion-ep34 D0 shard ledger.

M2096 adds no payload, shard, scheduler, or metric execution.  After the
M2090 fresh-tail continuation and the M2093 three-orphan recovery have each
published a sealed success receipt, this launcher consumes one outer O_EXCL
attempt and calls the exact M1704 reducer.  M1704 delegates to M1688, whose
strong verifier requires all 8,700 sibling namespaces to be exactly
``result=True, attempt=True, work=False, failure=False`` and every attempt to
be a regular non-symlink with mode 0400.

The source-stage CLI never opens a production result or shard namespace.
Execution additionally requires a different-author M2097 source review, a
double-sealed M2098 release, and a detached non-interactive session leader.
The published reduction remains a candidate pending an independent 8,700-
receipt result hammer; it is not a monolithic call, full decoder, system, or
paper result.  CPython 3.6 safe.
"""
from __future__ import print_function

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import traceback


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = Path(__file__).resolve()
CONTRACT = HW / (
    "contracts/m2096_ep34_decoder_d0_8700_shard_one_shot_reducer_"
    "publisher_source_contract_r1_20260904.json")
M1704_SOURCE = HERE / (
    "build_m1704_ep34_decoder_d0_execution_authority_adapter_source.py")
M2090_SOURCE = HERE / (
    "run_m2090_ep34_decoder_d0_fresh_tail_process_isolated_resume.py")
M2090_CONTRACT = HW / (
    "contracts/m2090_ep34_decoder_d0_fresh_tail_process_isolated_"
    "resume_source_contract_r1_20260904.json")
M2091_REVIEW = HW / (
    "reviews/m2091_m2090_ep34_decoder_d0_fresh_tail_process_isolated_"
    "resume_source_hammer_r1_20260904")
M2092_RELEASE = HW / (
    "contracts/m2092_m2091_m2090_ep34_decoder_d0_fresh_tail_process_"
    "isolated_resume_release_r1_20260904.json")
M2093_SOURCE = HERE / (
    "run_m2093_ep34_decoder_d0_three_orphan_manual_recovery.py")
M2093_CONTRACT = HW / (
    "contracts/m2093_ep34_decoder_d0_three_orphan_manual_recovery_"
    "source_contract_r1_20260904.json")
M2094_REVIEW = HW / (
    "reviews/m2094_m2093_ep34_decoder_d0_three_orphan_manual_recovery_"
    "source_hammer_r1_20260904")
M2095_RELEASE = HW / (
    "contracts/m2095_m2094_m2093_ep34_decoder_d0_three_orphan_manual_"
    "recovery_release_r1_20260904.json")
FUTURE_REVIEW = HW / (
    "reviews/m2097_m2096_ep34_decoder_d0_8700_shard_one_shot_reducer_"
    "publisher_source_hammer_r1_20260904")
FUTURE_RELEASE = HW / (
    "contracts/m2098_m2097_m2096_ep34_decoder_d0_8700_shard_one_shot_"
    "reducer_publisher_release_r1_20260904.json")
ATTEMPT = HW / (
    "results/.m2096_ep34_decoder_d0_8700_shard_reducer_attempt_consumed")
RESULT = HW / (
    "results/m2096_ep34_decoder_d0_8700_shard_reducer_r1_20260904")
WORK = HW / (
    "results/.m2096_ep34_decoder_d0_8700_shard_reducer_r1_20260904.work")
FAILURE = HW / (
    "results/m2096_ep34_decoder_d0_8700_shard_reducer_r1_20260904."
    "failed_no_retry")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

SCHEMA = "m2096_ep34_decoder_d0_8700_shard_one_shot_reducer_r1_v1"
STATUS = "SOURCE_ONLY__M2097_REVIEW_AND_M2098_RELEASE_REQUIRED"
RESULT_STATUS = (
    "CANDIDATE_COMPLETE_D0_8700_SEALED_SHARD_REDUCTION__"
    "INDEPENDENT_8700_RECEIPT_HAMMER_REQUIRED")
REVIEW_STATUS = (
    "PASS_M2097_M2096_DECODER_D0_8700_SHARD_ONE_SHOT_REDUCER_"
    "PUBLISHER_SOURCE__AUTHORIZE_M2098_RELEASE_ONLY")
RELEASE_SCHEMA = (
    "m2098_m2097_m2096_ep34_decoder_d0_8700_shard_one_shot_reducer_"
    "publisher_release_r1_v1")
RELEASE_STATUS = (
    "AUTHORIZE_M2096_ONE_DETACHED_D0_8700_SHARD_REDUCER_PUBLISH")
M1704_SHA256 = (
    "abc052025a5ed6975fa9d200581b182fa723a7b4b0330bc341b4bc2712204820")
M2090_SHA256 = (
    "23b5c41ac50a13de8a3c2e7e5f46c666de3ed7326f629c6d40fc4b4f577017c7")
M2090_CONTRACT_SHA256 = (
    "a61e03a9f4e3d25e0ac82a5c38d1cdf4fc63403b8d85fa18c36ea2784637a6ab")
M2091_REVIEW_SHA256 = (
    "37bd79c50ca33807b38372f10cf80e67ae1f894813298ee98b2f117d1aa1630b")
M2091_MANIFEST_SHA256 = (
    "5d4ca74abc1793668d5569f3fbc6999fb56fc843a582b7f78b3afa93509d03da")
M2091_OUTER_SHA256 = (
    "51849581c3007a1dba96b83c0ede18932ebd64de4ebc2a64f9f1dd92227d9976")
M2092_RELEASE_SHA256 = (
    "9e88449cc0e41b20cc151225381bb8cb3f17f665778b8371c707cacb8132cc09")
M2093_SHA256 = (
    "4238f72026442983d3d8c2bf0ea69d09470c56d5b45784100fb27fa88730b757")
M2093_CONTRACT_SHA256 = (
    "1c2a5fa7b27ddc2abbfab5545c83d959d44c3b6bfca5bd9dea9f42d81fde825e")
M2094_REVIEW_SHA256 = (
    "2ad8d70096f5a4a4f9cba4a99953f2bbcb13d5ff7a8c70d8a026a2dcb7d5d5cd")
M2094_MANIFEST_SHA256 = (
    "4848942860f2bd3ec4dae71bd9d01a9035b1097183031b78baced84816067e4d")
M2094_OUTER_SHA256 = (
    "5cc7f7c66579b71834cfb19b475dd2aff619cb7df212183d283e77ea82119806")
M2095_RELEASE_SHA256 = (
    "87b43efddf972c3d7a2022a0b8ce55eef53277f37ec322ad7d7b41633a93b8c1")
DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
TOTAL_SHARDS = 8700
AT_FDCWD = -100
RENAME_NOREPLACE = 1


class M2096Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M2096Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise M2096Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def load_m2093():
    regular_exact(M2093_SOURCE, M2093_SHA256, "exact M2093 source")
    spec = importlib.util.spec_from_file_location("m2096_exact_m2093",
                                                  str(M2093_SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot import exact M2093")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.SCHEMA ==
            "m2093_ep34_decoder_d0_three_orphan_manual_recovery_r1_v1" and
            module.M2090.SCHEMA ==
            "m2090_ep34_decoder_d0_fresh_tail_process_isolated_resume_r1_v1" and
            module.M2090.M1704.SCHEMA ==
            "m1704_ep34_decoder_d0_execution_authority_adapter_source_r1_v1" and
            module.B.G.TOTAL_SHARDS == TOTAL_SHARDS,
            "M2093/M2090/M1704 boundary drift")
    return module


M2093 = load_m2093()
M2090 = M2093.M2090
M1704 = M2090.M1704
B = M2093.B


def _absent(path, label):
    require(not os.path.lexists(str(path)), label + " exists")


def _absent_with_sidecars(path, label):
    for candidate in (Path(path), Path(str(path) + ".sha256"),
                      Path(str(path) + ".sha256.seal.sha256")):
        _absent(candidate, label)


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


def _identity():
    return {"source_sha256": sha256(SOURCE),
        "contract_sha256": sha256(CONTRACT),
        "m1704_source_sha256": M1704_SHA256,
        "m2090_source_sha256": M2090_SHA256,
        "m2090_contract_sha256": M2090_CONTRACT_SHA256,
        "m2091_review_sha256": M2091_REVIEW_SHA256,
        "m2091_manifest_sha256": M2091_MANIFEST_SHA256,
        "m2091_outer_file_sha256": M2091_OUTER_SHA256,
        "m2092_release_sha256": M2092_RELEASE_SHA256,
        "m2093_source_sha256": M2093_SHA256,
        "m2093_contract_sha256": M2093_CONTRACT_SHA256,
        "m2094_review_sha256": M2094_REVIEW_SHA256,
        "m2094_manifest_sha256": M2094_MANIFEST_SHA256,
        "m2094_outer_file_sha256": M2094_OUTER_SHA256,
        "m2095_release_sha256": M2095_RELEASE_SHA256,
        "checkpoint_sha256": B.G.CHECKPOINT_SHA256,
        "resource_manifest_sha256": B.G.RESOURCE_SHA256,
        "docs359_sha256": DOCS359_SHA256}


def _verify_fixed_source_authority(require_future_absent):
    """Verify only immutable source/review/release authority, never results."""
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    B.verify_double_sealed_file(CONTRACT, "M2096 source contract")
    regular_exact(M1704_SOURCE, M1704_SHA256, "exact M1704 source")
    regular_exact(M2090_SOURCE, M2090_SHA256, "exact M2090 source")
    regular_exact(M2090_CONTRACT, M2090_CONTRACT_SHA256,
                  "exact M2090 contract")
    B.verify_double_sealed_file(M2090_CONTRACT, "M2090 contract")
    B.verify_sealed_tree(M2091_REVIEW, M2091_REVIEW_SHA256,
        M2091_MANIFEST_SHA256, M2091_OUTER_SHA256, False, "M2091")
    regular_exact(M2092_RELEASE, M2092_RELEASE_SHA256,
                  "exact M2092 release")
    B.verify_double_sealed_file(M2092_RELEASE, "M2092 release")
    regular_exact(M2093_SOURCE, M2093_SHA256, "exact M2093 source")
    regular_exact(M2093_CONTRACT, M2093_CONTRACT_SHA256,
                  "exact M2093 contract")
    B.verify_double_sealed_file(M2093_CONTRACT, "M2093 contract")
    B.verify_sealed_tree(M2094_REVIEW, M2094_REVIEW_SHA256,
        M2094_MANIFEST_SHA256, M2094_OUTER_SHA256, False, "M2094")
    regular_exact(M2095_RELEASE, M2095_RELEASE_SHA256,
                  "exact M2095 release")
    B.verify_double_sealed_file(M2095_RELEASE, "M2095 release")
    require(M2090._validate_future_gate() == M2092_RELEASE_SHA256,
            "M2090 review/release semantic authority drift")
    require(M2093._validate_future_gate() == M2095_RELEASE_SHA256,
            "M2093 review/release semantic authority drift")
    require(M1704.reduce_complete_sealed_shards.__module__ ==
            M1704.__name__ and
            M1704.M1688.reduce_complete_sealed_shards.__module__ ==
            M1704.M1688.__name__, "exact M1704/M1688 reducer edge drift")
    if require_future_absent:
        require(not FUTURE_REVIEW.exists(), "future M2097 review exists")
        _absent_with_sidecars(FUTURE_RELEASE, "future M2098 release")
    return {"identity": _identity(), "fixed_source_reviews_releases": True,
        "production_results_opened": False,
        "shard_namespaces_opened": False, "payload_opens": 0,
        "shard_runs": 0, "gpu_runs": 0, "eda_runs": 0}


def _validate_future_gate():
    seal = B.verify_sealed_tree(FUTURE_REVIEW,
        allow_ignored_pycache=False, label="M2097")
    review = B.strict_json(FUTURE_REVIEW / "review.json")
    require(review.get("status") == REVIEW_STATUS and
            review.get("score_over_100", 0) >= 95 and
            review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0}
            and review.get("identity") == _identity() and
            review.get("authorization") == {
                "m2098_release_authoring": 1,
                "reducer_execution": 0,
                "payload_open": 0,
                "shard_execution": 0},
            "M2097 review authority drift")
    release_sha = B.verify_double_sealed_file(FUTURE_RELEASE, "M2098")
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
                "outer_attempt_writes": 1,
                "reducer_runs": 1,
                "sealed_shard_receipt_reads": TOTAL_SHARDS,
                "payload_opens": 0,
                "shard_runs": 0,
                "automatic_retry": False,
                "gpu_runs": 0,
                "eda_runs": 0} and
            release.get("reducer") == {
                "implementation": "exact M1704.reduce_complete_sealed_shards",
                "strong_verifier": "exact M1688.verify_sealed_shard",
                "required_shards": TOTAL_SHARDS,
                "ratio_policy": "integer ratio-of-sums"} and
            release.get("claim_boundary") == {
                "d0_candidate_pending_independent_hammer": True,
                "monolithic_full_call": False,
                "full_decoder": False,
                "system_speedup": False,
                "paper_result": False},
            "M2098 release identity/authorization drift")
    return release_sha


def _regular_attempt(path, label):
    path = Path(path)
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and
            stat.S_IMODE(mode) == 0o400,
            label + " must be regular non-symlink mode 0400")
    return sha256(path)


def _exact_outer_success_topology(source_module, label):
    present = {"result": os.path.lexists(str(source_module.RESULT)),
        "attempt": os.path.lexists(str(source_module.ATTEMPT)),
        "work": os.path.lexists(str(source_module.WORK)),
        "failure": os.path.lexists(str(source_module.FAILURE))}
    require(present == {"result": True, "attempt": True,
                        "work": False, "failure": False},
            label + " outer success topology drift")


def _verify_m2090_success():
    _exact_outer_success_topology(M2090, "M2090")
    attempt_sha = _regular_attempt(M2090.ATTEMPT, "M2090 attempt")
    attempt = B.strict_json(M2090.ATTEMPT)
    require(attempt.get("schema") == M2090.SCHEMA and
            attempt.get("source_sha256") == M2090_SHA256 and
            attempt.get("release_sha256") == M2092_RELEASE_SHA256 and
            attempt.get("fixed_partition") == {
                "start": 7563, "stop_exclusive": TOTAL_SHARDS,
                "workers": 3, "stride": 3, "shards_per_worker": 379} and
            attempt.get("automatic_retry") is False and
            attempt.get("payload_opened_before_attempt") is False,
            "M2090 attempt semantic identity drift")
    seal = B.verify_sealed_tree(M2090.RESULT,
        allow_ignored_pycache=False, label="M2090 success")
    row = B.strict_json(M2090.RESULT / "result.json")
    require(row.get("schema") == M2090.SCHEMA and
            row.get("status") ==
                "PASS_M2090_FRESH_TAIL_1137_SHARDS_THREE_PROCESS" and
            row.get("source_sha256") == M2090_SHA256 and
            row.get("release_sha256") == M2092_RELEASE_SHA256 and
            row.get("attempt_sha256") == attempt_sha and
            row.get("completed_shards") == 1137 and
            row.get("ordinal_start") == 7563 and
            row.get("ordinal_stop_exclusive") == TOTAL_SHARDS and
            row.get("preserved_orphans") == [7560, 7561, 7562] and
            row.get("exact_m1704_m1681_shard_execution") is True and
            row.get("reducer_executed") is False and
            row.get("full_d0_result") is False and
            row.get("full_decoder") is False and
            row.get("system_speedup") is False and
            row.get("paper_result") is False and
            row.get("independent_result_hammer_pending") is True,
            "M2090 success semantic identity drift")
    workers = row.get("workers")
    require(type(workers) is list and len(workers) == 3 and
            [item.get("worker_id") for item in workers] == [0, 1, 2] and
            all(item.get("status") == "PASS" and
                item.get("completed") == 379 for item in workers) and
            sum(item["completed"] for item in workers) == 1137,
            "M2090 worker success ledger drift")
    return {"result_json_sha256": sha256(M2090.RESULT / "result.json"),
        "manifest_sha256": seal["manifest_sha256"],
        "outer_file_sha256": seal["outer_file_sha256"],
        "attempt_sha256": attempt_sha,
        "release_sha256": M2092_RELEASE_SHA256}


def _verify_m2093_success():
    _exact_outer_success_topology(M2093, "M2093")
    attempt_sha = _regular_attempt(M2093.ATTEMPT, "M2093 attempt")
    attempt = B.strict_json(M2093.ATTEMPT)
    require(attempt.get("schema") == M2093.SCHEMA and
            attempt.get("source_sha256") == M2093_SHA256 and
            attempt.get("release_sha256") == M2095_RELEASE_SHA256 and
            attempt.get("ordinals") == [7560, 7561, 7562] and
            attempt.get("new_m1681_shard_attempt_writes") == 0 and
            attempt.get("automatic_retry") is False and
            attempt.get("payload_opened_before_attempt") is False,
            "M2093 attempt semantic identity drift")
    seal = B.verify_sealed_tree(M2093.RESULT,
        allow_ignored_pycache=False, label="M2093 success")
    row = B.strict_json(M2093.RESULT / "result.json")
    require(row.get("schema") == M2093.SCHEMA and
            row.get("status") ==
                "PASS_M2093_THREE_ORPHAN_MANUAL_RECOVERY" and
            row.get("source_sha256") == M2093_SHA256 and
            row.get("release_sha256") == M2095_RELEASE_SHA256 and
            row.get("attempt_sha256") == attempt_sha and
            row.get("ordinals") == [7560, 7561, 7562] and
            row.get("original_attempts_preserved") is True and
            row.get("new_m1681_shard_attempt_writes") == 0 and
            row.get("reducer_executed") is False and
            row.get("full_d0_result") is False and
            row.get("full_decoder") is False and
            row.get("system_speedup") is False and
            row.get("paper_result") is False and
            row.get("independent_result_hammer_pending") is True,
            "M2093 success semantic identity drift")
    recovered = row.get("recovered")
    require(type(recovered) is list and
            [item.get("ordinal") for item in recovered] ==
                [7560, 7561, 7562],
            "M2093 recovered ordinal ledger drift")
    for item in recovered:
        paths = B.namespace_paths(item["ordinal"])
        require(item.get("attempt_sha256") == sha256(paths["attempt"]) and
                item.get("result_manifest_sha256") ==
                    sha256(paths["result"] / "SHA256SUMS"),
                "M2093 recovered receipt identity drift")
    return {"result_json_sha256": sha256(M2093.RESULT / "result.json"),
        "manifest_sha256": seal["manifest_sha256"],
        "outer_file_sha256": seal["outer_file_sha256"],
        "attempt_sha256": attempt_sha,
        "release_sha256": M2095_RELEASE_SHA256}


def _verify_predecessor_successes():
    """Called only after the M2096 outer attempt has been consumed."""
    return {"m2090_fresh_tail": _verify_m2090_success(),
            "m2093_three_orphans": _verify_m2093_success()}


def _validate_detached_launch():
    require(os.environ.get("M2096_REDUCER_PUBLISH") == "1",
            "explicit reducer-publish token missing")
    require(os.getsid(0) == os.getpid(),
            "reducer publisher must be a setsid session leader")
    require(not any(os.isatty(descriptor) for descriptor in (0, 1, 2)),
            "reducer publisher must not retain a terminal")


def _consume_attempt(release_sha):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(ATTEMPT), flags, 0o400)
    try:
        row = {"schema": SCHEMA, "source_sha256": sha256(SOURCE),
            "release_sha256": release_sha, "pid": os.getpid(),
            "required_shards": TOTAL_SHARDS,
            "production_results_opened_before_attempt": False,
            "shard_receipts_read_before_attempt": 0,
            "payload_opens": 0, "shard_runs": 0,
            "automatic_retry": False}
        payload = (json.dumps(row, sort_keys=True,
            separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(str(ATTEMPT), 0o400)
    return sha256(ATTEMPT)


def _validate_reduction(reduced):
    require(type(reduced) is dict and
            reduced.get("schema") == M1704.M1688.SCHEMA and
            reduced.get("status") ==
                "COMPLETE_8700_EXACT_TOPOLOGY_SEALED_SHARDS__INDEPENDENT_HAMMER_REQUIRED" and
            reduced.get("complete_shards") == TOTAL_SHARDS and
            reduced.get("exact_sibling_topology") is True and
            reduced.get("attempt_regular_nonsymlink_mode_0400") is True and
            reduced.get("shard_isolated") is True and
            reduced.get("monolithic_full_call") is False and
            reduced.get("full_decoder") is False and
            reduced.get("system_speedup") is False and
            reduced.get("paper_result_pending_independent_hammer") is True,
            "exact M1704/M1688 reduction boundary drift")
    totals = reduced.get("configuration_totals")
    ratios = reduced.get("ratio_of_sums")
    require(type(totals) is dict and set(totals) == set(M1704.M1688.CONFIGS)
            and type(ratios) is dict,
            "reduction totals/ratio ledger drift")
    dense = totals["DENSE_TYPED_K8"]["cycles"]
    equal = totals["BIT_EQUAL_SERVICE_K1X8"]["cycles"]
    typed = totals["BIT_TYPED_K8"]["cycles"]
    require(all(type(value) is int and value > 0
                for value in (dense, equal, typed)) and
            ratios == {"dense_to_bit_typed": {"numerator": dense,
                "denominator": typed},
                "bit_equal_to_bit_typed": {"numerator": equal,
                "denominator": typed}},
            "integer ratio-of-sums operands drift")
    return reduced


def execute():
    _validate_detached_launch()
    _verify_fixed_source_authority(False)
    release_sha = _validate_future_gate()
    for path, label in ((ATTEMPT, "M2096 attempt"),
                        (RESULT, "M2096 result"),
                        (WORK, "M2096 work"),
                        (FAILURE, "M2096 failure")):
        _absent(path, label)
    attempt_sha = _consume_attempt(release_sha)
    WORK.mkdir(mode=0o700)
    success_published = False
    try:
        predecessors = _verify_predecessor_successes()
        reduced = _validate_reduction(
            M1704.reduce_complete_sealed_shards())
        receipt = {"schema": SCHEMA, "status": RESULT_STATUS,
            "source_sha256": sha256(SOURCE),
            "release_sha256": release_sha,
            "attempt_sha256": attempt_sha,
            "predecessor_success_identities": predecessors,
            "reduction": reduced,
            "reducer_implementation":
                "exact M1704.reduce_complete_sealed_shards",
            "strong_verifier": "exact M1688.verify_sealed_shard",
            "sealed_shard_receipts_read": TOTAL_SHARDS,
            "payload_opens": 0, "shard_runs": 0,
            "gpu_runs": 0, "eda_runs": 0,
            "automatic_retry": False,
            "d0_candidate_pending_independent_hammer": True,
            "full_d0_result": False,
            "monolithic_full_call": False,
            "full_decoder": False, "system_speedup": False,
            "paper_result": False}
        (WORK / "result.json").write_text(json.dumps(
            receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        B.seal_work_tree(WORK)
        B.verify_sealed_tree(WORK, allow_ignored_pycache=False,
                             label="M2096 result work")
        _rename_noreplace(WORK, RESULT)
        success_published = True
        return receipt
    except BaseException as error:
        if WORK.is_dir() and not os.path.lexists(str(FAILURE)):
            (WORK / "failure.json").write_text(json.dumps({
                "schema": SCHEMA, "status": "FAILED_NO_RETRY",
                "attempt_sha256": attempt_sha,
                "error": repr(error), "traceback": traceback.format_exc(),
                "automatic_retry": False, "payload_opens": 0,
                "shard_runs": 0, "gpu_runs": 0, "eda_runs": 0,
                "full_decoder": False, "system_speedup": False,
                "paper_result": False}, indent=2, sort_keys=True,
                allow_nan=False) + "\n", encoding="utf-8")
            B.seal_work_tree(WORK)
            B.verify_sealed_tree(WORK, allow_ignored_pycache=False,
                                 label="M2096 failure work")
            _rename_noreplace(WORK, FAILURE)
        raise
    finally:
        require(success_published or not WORK.exists(),
                "unpublished M2096 work remains")


def validate_source_stage():
    authority = _verify_fixed_source_authority(True)
    return {"authority": authority,
        "required_shards_at_future_execution": TOTAL_SHARDS,
        "m2090_result_opened": False, "m2093_result_opened": False,
        "shard_topology_checked": False,
        "reducer_executed": False, "payload_opens": 0,
        "shard_runs": 0, "gpu_runs": 0, "eda_runs": 0}


def describe():
    return {"schema": SCHEMA, "status": STATUS,
        "reducer": {
            "implementation": "exact M1704.reduce_complete_sealed_shards",
            "strong_verifier": "exact M1688.verify_sealed_shard",
            "required_shards": TOTAL_SHARDS,
            "required_sibling_topology": {"result": True,
                "attempt": True, "work": False, "failure": False},
            "attempt_type_mode": "regular non-symlink 0400",
            "ratio_policy": "integer ratio-of-sums"},
        "execution_gate": {
            "predecessor_successes": ["M2090", "M2093"],
            "source_review": "M2097", "release": "M2098",
            "outer_attempt_before_production_results_or_shards": True,
            "detached_token": "M2096_REDUCER_PUBLISH=1",
            "setsid_session_leader": True, "tty_fds": 0,
            "automatic_retry": False},
        "publication": {"temporary_sealed_work_tree": True,
            "rename_noreplace": True,
            "independent_8700_receipt_hammer_pending": True},
        "claim_boundary": {"source_only": True,
            "production_results_opened": False,
            "shard_receipts_read": 0, "payload_opens": 0,
            "shard_runs": 0, "gpu_runs": 0, "eda_runs": 0,
            "cycles": False, "traffic": False, "speedup": False,
            "energy": False, "rtl": False,
            "d0_candidate": False, "full_d0_result": False,
            "monolithic_full_call": False, "full_decoder": False,
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
            "status": "PASS_M2096_SOURCE_PREFLIGHT_NO_PRODUCTION_RESULT_NO_SHARD_READ",
            "source_stage": validate_source_stage(),
            "claim_boundary": describe()["claim_boundary"]}
    else:
        output = describe()
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
