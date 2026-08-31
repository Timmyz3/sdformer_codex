#!/usr/bin/env python3
"""Wrong-SHA negative and clean pre-mkdir source dry-run for M803 R16."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


SENTINEL = "M803_R16_STUB_REACHED_LIVE_VCS_LICENSE_BOUNDARY__NO_LIVE_PROBE__NO_RESULT_MKDIR"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_environment(root, nonce, trace, runner_sha):
    return {
        "PATH": "/usr/bin:/bin", "LANG": "C.utf8", "LC_ALL": "C.utf8",
        "M803_R16_EXPECTED_VCS_RUNNER_SHA256": runner_sha,
        "M803_R16_SOURCE_DRY_RUN": "1",
        "M803_R16_SOURCE_DRY_RUN_ROOT": str(root),
        "M803_R16_SOURCE_DRY_RUN_NONCE": str(nonce),
        "M803_R16_SOURCE_DRY_RUN_TRACE": str(trace),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runner", type=Path)
    args = parser.parse_args()
    runner = args.runner.resolve(strict=True)
    if not runner.is_file() or runner.is_symlink():
        raise RuntimeError("runner must be regular")
    actual = digest(runner)
    prospective_result = runner.parents[2] / "results/m803_c2_r16_channel_split_vcs_r1_20260828"
    prospective_attempt = runner.parents[2] / "results/.m803_c2_r16_channel_split_vcs_attempt_consumed"
    if prospective_result.exists() or prospective_attempt.exists():
        raise RuntimeError("prospective R16 result/attempt already exists")

    with tempfile.TemporaryDirectory(prefix="m803_r16_source_dryrun.") as raw:
        root = Path(raw)
        nonce = root / "SOURCE_HAMMER_NONCE"
        nonce.write_text("M803_R16_SOURCE_HAMMER_ONLY\n", encoding="utf-8")
        wrong_trace = root / "wrong_sha_trace.jsonl"
        wrong = subprocess.run([str(runner)], cwd=root,
                               env=base_environment(root, nonce, wrong_trace, "0" * 64),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True, timeout=180)
        if wrong.returncode != 3 or wrong_trace.exists():
            raise RuntimeError("wrong-SHA call did not fail before trace")

        trace = root / "positive_trace.jsonl"
        positive = subprocess.run([str(runner)], cwd=root,
                                  env=base_environment(root, nonce, trace, actual),
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  universal_newlines=True, timeout=180)
        if positive.returncode != 86:
            raise RuntimeError("positive dry-run rc={} stderr={}".format(
                positive.returncode, positive.stderr[-2000:]))
        combined = positive.stdout + positive.stderr
        if combined.count(SENTINEL) != 1:
            raise RuntimeError("unique dry-run sentinel missing")
        events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        names = [event["event"] for event in events]
        expected = ["stub_collision_initial", "stub_cgroup", "stub_resource",
                    "stub_collision_final", "live_probe_boundary_stop"]
        if names != expected:
            raise RuntimeError("event sequence mismatch: {}".format(names))
        zero = {"vcs_identity_probe_runs": 0, "license_server_queries": 0,
                "vcs_compile_runs": 0, "simv_runs": 0,
                "result_directories_created": 0}
        if any(event["totals"] != zero for event in events):
            raise RuntimeError("nonzero side-effect ledger")
        if prospective_result.exists() or prospective_attempt.exists():
            raise RuntimeError("dry-run created prospective identity")
        print(json.dumps({
            "schema": "m803_r16_runner_premkdir_dry_run_v1",
            "status": "PASS_WRONG_SHA_NEGATIVE_AND_ZERO_SIDE_EFFECT_BOUNDARY",
            "runner_sha256": actual,
            "wrong_sha_rc": wrong.returncode,
            "positive_rc": positive.returncode,
            "events": names,
            "totals": zero,
            "home_absent": "HOME" not in base_environment(root, nonce, trace, actual),
            "vcs_executed": False, "license_queried": False,
            "result_created": False, "attempt_created": False,
        }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
