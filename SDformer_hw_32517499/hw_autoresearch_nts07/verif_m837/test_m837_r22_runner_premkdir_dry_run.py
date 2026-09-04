#!/usr/bin/env python3
"""Outer-C wrong-SHA and zero-formal-side-effect dry-run for C2 R22."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


SENTINEL = "M837_R22_STUB_REACHED_LIVE_VCS_LICENSE_BOUNDARY__NO_LIVE_PROBE__NO_FORMAL_IDENTITY"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(root, nonce, trace, runner_sha):
    return {
        "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
        "M837_R22_EXPECTED_VCS_RUNNER_SHA256": runner_sha,
        "M837_R22_SOURCE_DRY_RUN": "1",
        "M837_R22_SOURCE_DRY_RUN_ROOT": str(root),
        "M837_R22_SOURCE_DRY_RUN_NONCE": str(nonce),
        "M837_R22_SOURCE_DRY_RUN_TRACE": str(trace),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runner", type=Path)
    args = parser.parse_args()
    runner = args.runner.resolve(strict=True)
    actual = digest(runner)
    hw = runner.parents[2]
    formal = [
        hw / "results/.m837_c2_r22_unicode_channel_split_vcs_attempt_consumed",
        hw / "results/m837_c2_r22_unicode_channel_split_vcs_r1_20260829",
    ]
    if any(path.exists() or path.is_symlink() for path in formal):
        raise RuntimeError("formal M837 identity already exists")
    before = set((hw / "results").glob(
        "m837_c2_r22_unicode_channel_split_vcs_r1_20260829.failed_or_incomplete.*"))
    with tempfile.TemporaryDirectory(prefix="m837_r22_dryrun.") as raw:
        root = Path(raw)
        nonce = root / "NONCE"
        nonce.write_text("M837_R22_SOURCE_HAMMER_ONLY\n", encoding="utf-8")
        wrong_trace = root / "wrong.jsonl"
        wrong = subprocess.run([str(runner)], cwd=root,
            env=environment(root, nonce, wrong_trace, "0" * 64),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=180)
        if wrong.returncode != 3 or wrong_trace.exists():
            raise RuntimeError("wrong SHA did not fail before trace")
        trace = root / "positive.jsonl"
        positive = subprocess.run([str(runner)], cwd=root,
            env=environment(root, nonce, trace, actual),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=180)
        if positive.returncode != 86:
            raise RuntimeError("positive dry-run rc={} stderr={}".format(
                positive.returncode, positive.stderr[-3000:]))
        if (positive.stdout + positive.stderr).count(SENTINEL) != 1:
            raise RuntimeError("unique source-dry-run sentinel missing")
        events = [json.loads(line) for line in
                  trace.read_text(encoding="utf-8").splitlines()]
        names = [event["event"] for event in events]
        expected = ["source_contract_verified",
                    "m834_r21_and_m832_spent_authorities_verified",
                    "atomic_guard_selftest",
                    "live_vcs_license_boundary_stop"]
        if names != expected:
            raise RuntimeError("dry-run event sequence drift: " + repr(names))
        zero = {"vcs_identity_probe_runs": 0, "license_server_queries": 0,
                "vcs_compile_runs": 0, "simv_runs": 0,
                "formal_attempts_created": 0, "formal_results_created": 0,
                "failure_quarantines_created": 0}
        if any(event["totals"] != zero for event in events):
            raise RuntimeError("nonzero dry-run side-effect ledger")
    after = set((hw / "results").glob(
        "m837_c2_r22_unicode_channel_split_vcs_r1_20260829.failed_or_incomplete.*"))
    if any(path.exists() or path.is_symlink() for path in formal) or after != before:
        raise RuntimeError("source dry-run created formal identity/quarantine")
    print(json.dumps({
        "schema": "m837_r22_runner_premkdir_dry_run_v1",
        "status": "PASS_M837_R22_OUTER_C_UNICODE_AND_ZERO_FORMAL_SIDE_EFFECT_DRY_RUN",
        "runner_sha256": actual, "outer_lang": "C", "outer_lc_all": "C",
        "wrong_sha_rc": wrong.returncode, "positive_rc": positive.returncode,
        "events": names, "totals": zero, "home_absent": True,
        "vcs_executed": False, "license_queried": False,
        "attempt_created": False, "result_created": False,
        "failure_quarantine_created": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

