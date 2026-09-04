#!/usr/bin/env python3
"""Execute the exact M863/R21 pre-mkdir prefix in fail-closed stub mode.

The runner itself owns the stub boundary.  This harness supplies a fresh
temporary root and nonce, executes under a clean environment, and admits only
the unique rc/sentinel tuple emitted immediately before the live VCS/lmutil
boundary.  It rejects any VCS/lmutil/simv trace or prospective result mkdir.
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


SENTINEL = "M863_R21_STUB_REACHED_LIVE_VCS_LICENSE_BOUNDARY__NO_LIVE_PROBE__NO_RESULT_MKDIR"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runner", type=Path)
    args = parser.parse_args()
    runner = args.runner.resolve(strict=True)
    if runner.is_symlink() or not runner.is_file():
        raise RuntimeError("runner must be regular")
    with tempfile.TemporaryDirectory(prefix="m863_r21_premkdir_dryrun.") as raw:
        root = Path(raw)
        nonce = root / "SOURCE_HAMMER_NONCE"
        nonce.write_text("M863_R21_SOURCE_HAMMER_ONLY\n", encoding="utf-8")
        trace = root / "side_effect_trace.jsonl"
        env = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
            "VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
            "VCS_ARCH_OVERRIDE": "linux",
            "SNPSLMD_LICENSE_FILE": "27030@ic.ismd-nemo",
            "LM_LICENSE_FILE": "/opt/synopsys/Synopsys.dat",
            "M863_R21_SOURCE_DRY_RUN": "1",
            "M863_R21_SOURCE_DRY_RUN_ROOT": str(root),
            "M863_R21_SOURCE_DRY_RUN_NONCE": str(nonce),
            "M863_R21_SOURCE_DRY_RUN_TRACE": str(trace),
        }
        completed = subprocess.run([str(runner)], cwd=root, env=env,
                                   universal_newlines=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, timeout=180)
        if completed.returncode != 86:
            raise RuntimeError(f"dry-run rc {completed.returncode}, stderr={completed.stderr[-2000:]}")
        combined = completed.stdout + completed.stderr
        if combined.count(SENTINEL) != 1:
            raise RuntimeError("unique live-boundary sentinel missing")
        if not trace.is_file() or trace.is_symlink():
            raise RuntimeError("stub trace missing/non-regular")
        events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line]
        names = [event["event"] for event in events]
        expected = [
            "stub_collision_initial", "stub_cgroup", "stub_resource",
            "stub_collision_final", "live_probe_boundary_stop",
        ]
        if names != expected:
            raise RuntimeError(f"stub event sequence mismatch: {names}")
        totals = events[-1]["totals"]
        expected_zero = {
            "vcs_identity_probe_runs": 0, "license_server_queries": 0,
            "vcs_compile_runs": 0, "simv_runs": 0,
            "result_directories_created": 0,
        }
        if totals != expected_zero:
            raise RuntimeError(f"side-effect totals not zero: {totals}")
        prospective = runner.parents[2] / "results" / "m863_m533_m528_dead_write_only_1rw_unit_delay_vcs_r21_20260829"
        if prospective.exists():
            raise RuntimeError("prospective result identity exists after stub dry-run")
        if any(root.glob("**/simv")):
            raise RuntimeError("simv artifact appeared in clean dry-run root")
        print(json.dumps({
            "schema": "m863_r21_premkdir_stub_dry_run_v1",
            "status": "PASS_REACHED_LIVE_PROBE_BOUNDARY_WITH_ZERO_SIDE_EFFECTS",
            "runner_rc": 86,
            "sentinel_count": 1,
            "events": names,
            "totals": totals,
            "prospective_result_absent": True,
        }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

