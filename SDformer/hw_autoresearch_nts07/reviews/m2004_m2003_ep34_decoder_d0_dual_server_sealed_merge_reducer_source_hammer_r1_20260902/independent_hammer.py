#!/usr/bin/env python3
"""Different-author source hammer for M2003; synthetic/temp data only."""
from __future__ import print_function

import importlib.util
import json
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "system_simulator/scripts/"
    "build_m2003_ep34_decoder_d0_dual_server_sealed_merge_reducer_source.py")


def load_target():
    spec = importlib.util.spec_from_file_location("m2004_target", str(SOURCE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def empty_pid_attack(M):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        review = root / "review"
        review.mkdir()
        (review / "review.json").write_text("{}\n", encoding="ascii")
        release = root / "release.json"
        release.write_text("{}\n", encoding="ascii")
        M.FUTURE_REVIEW = review
        M.FUTURE_RELEASE = release
        identity = M.identity()
        seal = {"manifest_sha256": "1" * 64,
                "outer_file_sha256": "2" * 64}
        review_row = {"status": M.REVIEW_STATUS, "score_over_100": 100,
            "severity_counts": {"p0": 0, "p1": 0, "p2": 0},
            "identity": identity,
            "authorization": {"m2005_release_authoring": True,
                "archive_open": False, "merge": False, "reducer": False,
                "shard_runs": 0, "payload_opens": 0,
                "eda_runs": 0, "gpu_runs": 0}}
        release_identity = dict(identity,
            m2004_review_sha256=M.sha256(review / "review.json"),
            m2004_manifest_sha256=seal["manifest_sha256"],
            m2004_outer_file_sha256=seal["outer_file_sha256"])
        release_row = {"schema": M.RELEASE_SCHEMA,
            "status": M.RELEASE_STATUS, "identity": release_identity,
            "archive_sha256": "a" * 64,
            "remote_range": [M.REMOTE_START, M.REMOTE_STOP],
            "local_required_range": [0, M.LOCAL_STOP],
            "stopped_pids": [],
            "authorization": {"archive_open": 1, "archive_extract": 1,
                "merge": 1, "reducer": 1, "result_publish": 1,
                "shard_runs": 0, "payload_opens": 0, "deletes": 0,
                "overwrites": 0, "eda_runs": 0, "gpu_runs": 0}}
        old_tree = M.B.verify_sealed_tree
        old_json = M.B.strict_json
        old_release = M.B.verify_double_sealed_file
        M.B.verify_sealed_tree = lambda *args, **kwargs: seal
        M.B.strict_json = lambda path: (
            review_row if Path(path) == review / "review.json"
            else release_row)
        M.B.verify_double_sealed_file = lambda *args, **kwargs: "b" * 64
        try:
            return M.validate_runtime_authority("a" * 64, []) == "b" * 64
        finally:
            M.B.verify_sealed_tree = old_tree
            M.B.strict_json = old_json
            M.B.verify_double_sealed_file = old_release


def late_verification_attack(M):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        M.RESULT = root / "final_result"
        M.LOCAL_STOP = 0
        M.REMOTE_START = 0
        M.REMOTE_STOP = 2
        events = []
        M.validate_runtime_authority = lambda *args, **kwargs: "c" * 64
        M.extract_archive_once = lambda *args, **kwargs: (
            root / "stage", {"archive_sha256": "d" * 64})
        attempt = root / "attempt0"
        attempt.write_bytes(b"A")

        def paths(ordinal):
            return {"result": root / ("result%d" % ordinal),
                "attempt": attempt if ordinal == 0 else root / "attempt1",
                "work": root / ("work%d" % ordinal),
                "failure": root / ("failure%d" % ordinal)}

        def verify_remote(stage, ordinal):
            events.append("verify%d" % ordinal)
            if ordinal == 1:
                raise M.M2003Error("late corrupt shard")
            return {"row": {}, "attempt_sha256": M.sha256(attempt),
                    "seal": {}}

        M.B.namespace_paths = paths
        M.verify_staged_shard = verify_remote
        M._copy_result_tree = lambda source, target: events.append("install0")
        M._staged_paths = lambda stage, ordinal: {
            "directory": root / "remote0", "attempt": root / "remote_attempt"}
        M.M1704.M1688.verify_sealed_shard = lambda ordinal: (
            events.append("postinstall_verify%d" % ordinal) or {"row": {}})
        failure = None
        try:
            M.merge_and_reduce(root / "archive", "d" * 64, [],
                               root / "staging", root / "quarantine")
        except M.M2003Error as error:
            failure = str(error)
        return {"events": events, "failure": failure,
            "installed_before_all_remote_verified": (
                events.index("install0") < events.index("verify1"))}


def overlap_core_attack(M):
    keys = ("schema", "status", "source_sha256", "release_sha256",
        "attempt_sha256", "checkpoint_sha256", "resource_manifest_sha256",
        "shard_ordinal", "shard", "configuration_order", "metrics",
        "integer_ratio_inputs", "payload_fd_sha256", "payload_fd_size",
        "automatic_retry", "shard_isolated", "monolithic_full_call",
        "full_decoder", "system_speedup", "paper_result")
    left = dict((key, key) for key in keys)
    left["rss"] = {"host": 1}
    left["independent_result_hammer_pending"] = True
    right = dict(left)
    right["independent_result_hammer_pending"] = False
    core = M.deterministic_core(left)
    return {"claim_flag_drift_accepted":
                core == M.deterministic_core(right),
            "excluded_keys": sorted(set(left) - set(core))}


def main():
    M = load_target()
    source_text = SOURCE.read_text(encoding="utf-8")
    row = {"empty_stopped_pids_accepted": empty_pid_attack(M)}
    M = load_target()
    row["late_remote_verification"] = late_verification_attack(M)
    M = load_target()
    row["overlap_core"] = overlap_core_attack(M)
    row["archive_open_call_sites"] = source_text.count("tarfile.open(")
    row["overall_attempt_namespace_present"] = (
        "merge_attempt_consumed" in source_text)
    row["local_prefix_release_explicitly_bound"] = (
        'local["row"].get("release_sha256") == M1706_RELEASE_SHA256'
        in source_text)
    print(json.dumps(row, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
