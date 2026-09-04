#!/usr/bin/env python3
"""Read-only independent result hammer for the M2093 three-shard recovery.

This checker opens no decoder payload and executes no shard, reducer, EDA, or
GPU workload.  It verifies the already-published M2093 orchestration receipt,
the three canonical M1681 shard receipts, their original immutable attempts,
and the preserved empty-work evidence.
"""
from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HW / (
    "system_simulator/scripts/"
    "run_m2093_ep34_decoder_d0_three_orphan_manual_recovery.py")
RESULT = HW / (
    "results/m2093_ep34_decoder_d0_three_orphan_manual_recovery_"
    "r1_20260904")
ATTEMPT = HW / (
    "results/.m2093_ep34_decoder_d0_three_orphan_manual_recovery_"
    "attempt_consumed")
FAILURE = HW / (
    "results/m2093_ep34_decoder_d0_three_orphan_manual_recovery_"
    "r1_20260904.failed_no_retry")
WORK = HW / (
    "results/.m2093_ep34_decoder_d0_three_orphan_manual_recovery_"
    "r1_20260904.work")
QUARANTINE = HW / (
    "recovery_quarantine/m2093_decoder_d0_three_orphan_original_empty_work")
OUT = HERE / "mechanical_checks.json"
EXPECTED_SOURCE = (
    "4238f72026442983d3d8c2bf0ea69d09470c56d5b45784100fb27fa88730b757")
EXPECTED_M1681_SOURCE = (
    "006535679b38e2aa207fadde05e9207d2e72dae0464315dceea4a3c96da77a6f")
EXPECTED_M1706_RELEASE = (
    "43c7096fe90263abf7593d41c3222675bc9153ca4529436b3a57405c550fe7e0")
EXPECTED_M2095_RELEASE = (
    "87b43efddf972c3d7a2022a0b8ce55eef53277f37ec322ad7d7b41633a93b8c1")
EXPECTED_DOCS359 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
ORDINALS = (7560, 7561, 7562)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path, label, mode=None):
    path = Path(path)
    current = path.lstat().st_mode
    require(stat.S_ISREG(current) and not path.is_symlink(),
            label + " must be a regular non-symlink")
    if mode is not None:
        require(stat.S_IMODE(current) == mode, label + " mode drift")
    return path


def strict_json(path):
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    return json.loads(Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            RuntimeError("nonfinite JSON: " + token)))


def load_source():
    regular(SOURCE, "M2093 source")
    require(sha256(SOURCE) == EXPECTED_SOURCE, "M2093 source SHA drift")
    spec = importlib.util.spec_from_file_location(
        "m2099_reviewed_m2093", str(SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot import exact M2093 source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_source()
B = M.B


def verify_double_sealed_file(path, expected, label):
    path = regular(path, label)
    require(sha256(path) == expected, label + " SHA drift")
    sidecar = regular(Path(str(path) + ".sha256"), label + " sidecar")
    outer = regular(Path(str(sidecar) + ".seal.sha256"), label + " outer")
    require(sidecar.read_text(encoding="ascii") ==
            expected + "  " + path.name + "\n", label + " sidecar drift")
    require(outer.read_text(encoding="ascii") ==
            sha256(sidecar) + "  " + sidecar.name + "\n",
            label + " outer drift")
    return {"file_sha256": expected, "sidecar_sha256": sha256(sidecar),
            "outer_file_sha256": sha256(outer)}


def verify_empty_quarantine():
    mode = QUARANTINE.lstat().st_mode
    require(stat.S_ISDIR(mode) and not QUARANTINE.is_symlink(),
            "recovery quarantine must be a directory non-symlink")
    expected = set("shard_%d_original_empty_work" % item for item in ORDINALS)
    actual = set(path.name for path in QUARANTINE.iterdir())
    require(actual == expected, "recovery quarantine population drift")
    rows = []
    for ordinal in ORDINALS:
        path = QUARANTINE / ("shard_%d_original_empty_work" % ordinal)
        current = path.lstat().st_mode
        require(stat.S_ISDIR(current) and not path.is_symlink(),
                "preserved work must be a directory non-symlink")
        require(not any(path.iterdir()), "preserved original work is not empty")
        rows.append({"ordinal": ordinal, "path": str(path.relative_to(HW)),
                     "mode": "%04o" % stat.S_IMODE(current), "empty": True})
    return rows


def expected_attempt(ordinal):
    return {"schema": B.SCHEMA, "shard_ordinal": ordinal,
        "shard": B.G.shard_descriptor(ordinal),
        "source_sha256": EXPECTED_M1681_SOURCE,
        "release_sha256": EXPECTED_M1706_RELEASE,
        "automatic_retry": False,
        "payload_opened_before_attempt": False}


def verify_one_shard(ordinal, release_sha):
    paths = B.namespace_paths(ordinal)
    present = {key: os.path.lexists(str(value)) for key, value in paths.items()}
    require(present == {"attempt": True, "work": False,
                        "result": True, "failure": False},
            "canonical sibling topology drift for %d" % ordinal)
    attempt = regular(paths["attempt"], "M1681 attempt", 0o400)
    attempt_row = strict_json(attempt)
    require(attempt_row == expected_attempt(ordinal),
            "original M1681 attempt semantics drift for %d" % ordinal)
    attempt_sha = sha256(attempt)

    verified = B.verify_sealed_shard(ordinal)
    row = strict_json(paths["result"] / "result.json")
    require(verified["row"] == row and verified["attempt_sha256"] == attempt_sha,
            "M1681 sealed-shard return drift")
    # Re-run the frozen receipt, metric-bundle, and integer-ratio validators
    # against identities pinned by this independent review.
    B.validate_shard_receipt(row, ordinal, attempt_sha,
                             EXPECTED_M1706_RELEASE)
    B.validate_metric_bundle(row["metrics"], row["shard"])
    ratios = B.G.validate_three_configuration_metrics(
        row["metrics"], row["shard"])
    require(row["integer_ratio_inputs"] == ratios,
            "recomputed integer ratio inputs drift")

    expected_provenance = {
        "source_sha256": EXPECTED_SOURCE,
        "release_sha256": release_sha,
        "original_attempt_preserved": True,
        "original_empty_work_quarantine":
            "recovery_quarantine/m2093_decoder_d0_three_orphan_"
            "original_empty_work/shard_%d_original_empty_work" % ordinal,
        "new_m1681_shard_attempt_written": False}
    require(row.get("manual_recovery") == expected_provenance,
            "manual-recovery provenance drift for %d" % ordinal)
    require(row.get("independent_result_hammer_pending") is True,
            "pre-hammer shard boundary drift")

    metrics = []
    for metric in row["metrics"]:
        metrics.append({"configuration": metric["configuration"],
            "total_cycles": metric["total_cycles"],
            "request_count": metric["request_count"],
            "metric_sha256": B.canonical_sha(metric)})
    return {"ordinal": ordinal, "attempt_sha256": attempt_sha,
        "attempt_mode": "0400", "attempt_semantics_exact": True,
        "result_json_sha256": sha256(paths["result"] / "result.json"),
        "result_manifest_sha256": verified["seal"]["manifest_sha256"],
        "result_outer_file_sha256": verified["seal"]["outer_file_sha256"],
        "payload_fd_sha256": row["payload_fd_sha256"],
        "payload_fd_size": row["payload_fd_size"],
        "integer_ratio_inputs": ratios, "metrics": metrics}


def run():
    require(sha256(M.DOCS359) == EXPECTED_DOCS359,
            "protected docs359 SHA drift")
    require(sha256(B.SOURCE) == EXPECTED_M1681_SOURCE,
            "frozen M1681 source SHA drift")
    require(sha256(M.M2090.M1706_RELEASE) == EXPECTED_M1706_RELEASE,
            "M1706 release SHA drift")
    m1706_seal = verify_double_sealed_file(
        M.M2090.M1706_RELEASE, EXPECTED_M1706_RELEASE, "M1706 release")
    m2095_seal = verify_double_sealed_file(
        M.FUTURE_RELEASE, EXPECTED_M2095_RELEASE, "M2095 release")
    require(M._validate_future_gate() == EXPECTED_M2095_RELEASE,
            "M2095 authority gate drift")

    overall_attempt = regular(ATTEMPT, "M2093 overall attempt", 0o400)
    overall_attempt_sha = sha256(overall_attempt)
    attempt_row = strict_json(overall_attempt)
    require(attempt_row == {
        "schema": M.SCHEMA, "source_sha256": EXPECTED_SOURCE,
        "release_sha256": EXPECTED_M2095_RELEASE,
        "ordinals": list(ORDINALS),
        "original_orphans": [{"ordinal": item,
            "attempt_sha256": sha256(B.namespace_paths(item)["attempt"]),
            "attempt_mode": "0400", "empty_work": True}
            for item in ORDINALS],
        "new_m1681_shard_attempt_writes": 0,
        "automatic_retry": False,
        "payload_opened_before_attempt": False},
        "M2093 overall attempt semantics drift")

    require(not os.path.lexists(str(WORK)) and
            not os.path.lexists(str(FAILURE)),
            "M2093 work/failure residue exists")
    result_seal = B.verify_sealed_tree(
        RESULT, allow_ignored_pycache=False, label="M2093 overall result")
    members = set(path.relative_to(RESULT).as_posix()
        for path in RESULT.rglob("*") if path.is_file() and
        path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    require(members == {"result.json"},
            "M2093 overall result population drift")
    result_row = strict_json(RESULT / "result.json")

    quarantine = verify_empty_quarantine()
    recovered = [verify_one_shard(item, EXPECTED_M2095_RELEASE)
                 for item in ORDINALS]
    expected_recovered = [{"ordinal": item["ordinal"],
        "attempt_sha256": item["attempt_sha256"],
        "result_manifest_sha256": item["result_manifest_sha256"],
        "payload_fd_sha256": item["payload_fd_sha256"]}
        for item in recovered]
    require(result_row == {
        "schema": M.SCHEMA,
        "status": "PASS_M2093_THREE_ORPHAN_MANUAL_RECOVERY",
        "source_sha256": EXPECTED_SOURCE,
        "release_sha256": EXPECTED_M2095_RELEASE,
        "attempt_sha256": overall_attempt_sha,
        "ordinals": list(ORDINALS), "recovered": expected_recovered,
        "original_attempts_preserved": True,
        "new_m1681_shard_attempt_writes": 0,
        "reducer_executed": False, "full_d0_result": False,
        "full_decoder": False, "system_speedup": False,
        "paper_result": False,
        "independent_result_hammer_pending": True},
        "M2093 overall receipt or recomputed shard identities drift")

    return {
        "schema": "m2099_m2093_ep34_decoder_d0_three_orphan_manual_"
                  "recovery_result_hammer_r1_v1",
        "status": "PASS_M2099_M2093_THREE_ORPHAN_RECOVERY_RESULT",
        "production_execution": False,
        "identities": {"m2093_source_sha256": EXPECTED_SOURCE,
            "m1681_source_sha256": EXPECTED_M1681_SOURCE,
            "m1706_release": m1706_seal, "m2095_release": m2095_seal,
            "docs359_sha256": EXPECTED_DOCS359},
        "overall": {"attempt_sha256": overall_attempt_sha,
            "attempt_mode": "0400", "attempt_regular_non_symlink": True,
            "result_json_sha256": sha256(RESULT / "result.json"),
            "result_manifest_sha256": result_seal["manifest_sha256"],
            "result_outer_file_sha256": result_seal["outer_file_sha256"],
            "result_double_sealed": True,
            "work_and_failure_absent": True},
        "quarantine": quarantine, "recovered_shards": recovered,
        "claim_boundary": {"recovered_shards_admitted": 3,
            "future_8700_shard_reducer_input_only": True,
            "reducer_executed": False, "full_d0_result": False,
            "full_decoder": False, "cycles": False, "traffic": False,
            "speedup": False, "energy": False,
            "system_speedup": False, "paper_result": False}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    output = run()
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True,
        allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
