#!/usr/bin/env python3
"""Source-stage tests for M2003; no archive, merge, reducer, GPU or EDA."""
from __future__ import print_function

import importlib.util
import json
from pathlib import Path
import tempfile


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / (
    "build_m2003_ep34_decoder_d0_dual_server_sealed_merge_reducer_source.py")
spec = importlib.util.spec_from_file_location("m2003_test_target", str(SOURCE))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def expect_failure(action):
    try:
        action()
    except M.M2003Error:
        return
    raise AssertionError("expected M2003Error")


def main():
    description = M.describe()
    assert description["claim_boundary"]["source_only"] is True
    assert description["claim_boundary"]["archive_open"] is False
    assert description["ranges"] == {
        "local_required": [0, 4500], "remote_archive": [4500, 8700],
        "total_shards": 8700}
    population = M.expected_archive_population(4500, 4502)
    assert len(population["directories"]) == 2
    assert len(population["files"]) == 8
    assert M._safe_member_name(
        "hw_autoresearch_nts07/results/example/result.json") == (
        "hw_autoresearch_nts07/results/example/result.json")
    for unsafe in ("/absolute", "../escape", "./relative", "a\\b"):
        expect_failure(lambda value=unsafe: M._safe_member_name(value))
    row = dict((key, key) for key in (
        "schema", "status", "source_sha256", "release_sha256",
        "attempt_sha256", "checkpoint_sha256", "resource_manifest_sha256",
        "shard_ordinal", "shard", "configuration_order", "metrics",
        "integer_ratio_inputs", "payload_fd_sha256", "payload_fd_size",
        "automatic_retry", "shard_isolated", "monolithic_full_call",
        "full_decoder", "system_speedup", "paper_result"))
    row["rss"] = {"machine_specific": 1}
    core = M.deterministic_core(row)
    assert "rss" not in core and len(core) == 20
    other = dict(row)
    other["rss"] = {"machine_specific": 2}
    assert M.deterministic_core(row) == M.deterministic_core(other)
    with tempfile.TemporaryDirectory() as root:
        missing = Path(root) / "missing.tar"
        expect_failure(lambda: M.inspect_archive(missing, "0" * 64))
    assert M.B.G.TOTAL_SHARDS == 8700
    print(json.dumps({"status": "PASS_M2003_SOURCE_TEST",
        "archive_opened": False, "merge": False, "reducer": False,
        "gpu": False, "eda": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
