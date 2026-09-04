#!/usr/bin/env python3
"""Execute the exact M770 predicate heredoc embedded in the M784/r15 runner.

This is a source-static test.  It never executes the runner, VCS, simv, lmutil,
or any other EDA command.  The negative cases deliberately delete or misspell
the sealed M770 decision key and require the *extracted runner heredoc* to fail.
"""

import json
import subprocess
import tempfile
from pathlib import Path


HW_ROOT = Path(__file__).resolve().parents[1]
RUNNER = HW_ROOT / "dc_handoff/scripts/run_vcs_m784_m533_m528_dead_write_only_1rw_unit_delay_r15_exact_sha.sh"
R13_FAILURE = HW_ROOT / "results/m758_m533_m528_dead_write_only_1rw_unit_delay_vcs_r13_20260828/RUN_FAILED_OR_INCOMPLETE.json"
M770 = HW_ROOT / "reviews/m770_m533_r13_vcs_home_failure_fresh_hammer_r1_20260828/review.json"
M782 = HW_ROOT / "reviews/m782_m533_r14_premkdir_launch_boundary_failure_hammer_r1_20260828/review.json"
PREFLIGHT = HW_ROOT / "reviews/m772_m533_r14_vcs_environment_preflight_r1_20260828/preflight.json"

START = '  python3 -I - "${R13_FAILED_RECEIPT}" "${M770_REVIEW}" "${M782_REVIEW}" "${AUTHOR_ENV_PREFLIGHT}" <<\'PY2\''
END = "PY2"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def extract_real_heredoc():
    lines = RUNNER.read_text(encoding="utf-8").splitlines()
    starts = [index for index, line in enumerate(lines) if line == START]
    require(len(starts) == 1, f"expected one exact M770 heredoc start, got {len(starts)}")
    start = starts[0] + 1
    ends = [index for index in range(start, len(lines)) if lines[index] == END]
    require(ends, "unterminated M770 heredoc")
    code = "\n".join(lines[start : ends[0]]) + "\n"
    require(
        'audit.get("decision", {}).get("r14_launch_authorized_now") is False' in code,
        "real runner heredoc does not request decision.r14_launch_authorized_now",
    )
    require(
        'audit.get("decision", {}).get("vcs_launch_authorized_now")' not in code,
        "withdrawn nonexistent M770 launch key remains in real runner heredoc",
    )
    return code


def run_extracted(code, m770_path):
    return subprocess.run(
        ["python3", "-I", "-", str(R13_FAILURE), str(m770_path), str(M782), str(PREFLIGHT)],
        input=code,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def main():
    code = extract_real_heredoc()
    good = run_extracted(code, M770)
    require(good.returncode == 0, f"sealed M770 should pass real heredoc: {good.stderr}")

    sealed = json.loads(M770.read_text(encoding="utf-8"))
    require(sealed["decision"]["r14_launch_authorized_now"] is False, "sealed M770 key/value drift")
    with tempfile.TemporaryDirectory(prefix="m784_r15_m770_predicate_static.") as raw:
        tmp = Path(raw)

        missing = json.loads(json.dumps(sealed))
        del missing["decision"]["r14_launch_authorized_now"]
        missing_path = tmp / "m770_missing_key.json"
        missing_path.write_text(json.dumps(missing), encoding="utf-8")
        missing_run = run_extracted(code, missing_path)
        require(missing_run.returncode != 0, "missing M770 key was incorrectly accepted")
        require("M770 launch boundary" in missing_run.stderr, "missing-key failure did not come from real predicate")

        wrong = json.loads(json.dumps(sealed))
        value = wrong["decision"].pop("r14_launch_authorized_now")
        wrong["decision"]["vcs_launch_authorized_now"] = value
        wrong_path = tmp / "m770_wrong_key.json"
        wrong_path.write_text(json.dumps(wrong), encoding="utf-8")
        wrong_run = run_extracted(code, wrong_path)
        require(wrong_run.returncode != 0, "misspelled M770 key was incorrectly accepted")
        require("M770 launch boundary" in wrong_run.stderr, "wrong-key failure did not come from real predicate")

    print(
        "PASS_M784_R15_REAL_RUNNER_M770_HEREDOC "
        "sealed=pass missing_key=fail wrong_key=fail runner_executions=0 eda_runs=0"
    )


if __name__ == "__main__":
    main()
