#!/usr/bin/env python3
"""Source-only fail-closed tests for the M849/R20 wall-clock simv wrapper.

No VCS, license or EDA command is executed.  Tiny temporary fake-simv programs
exercise GNU timeout's fast, TERM and TERM-to-KILL paths, a failing tee, and a
double-sealed infrastructure-timeout receipt model.  Short test durations do
not alter the production literal pinned in the runner.
"""

import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import tempfile
import time
from pathlib import Path


TIMEOUT = Path("/usr/bin/timeout")
TEE = Path("/usr/bin/tee")
PYTHON36 = Path("/usr/libexec/platform-python3.6")
TIMEOUT_SHA256 = "2d5662f0e08f558aa842d7bc99aa00ea534b4fb46c7e72f6a4c834220cafbf02"
PRODUCTION_LITERAL = "/usr/bin/timeout --signal=TERM --kill-after=30s 300s ./simv -no_save"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_fake(path, body):
    path.write_text("#!{}\n{}\n".format(PYTHON36, body), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def run_pipeline(fake, log, timeout_seconds="0.30s", kill_after="0.20s", tee=TEE):
    first = subprocess.Popen(
        [str(TIMEOUT), "--signal=TERM", "--kill-after=" + kill_after,
         timeout_seconds, str(fake), "-no_save"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    second = subprocess.Popen([str(tee), str(log)], stdin=first.stdout,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    first.stdout.close()
    tee_rc = second.wait(timeout=5)
    timeout_rc = first.wait(timeout=5)
    if timeout_rc < 0:
        timeout_rc = 128 - timeout_rc
    return timeout_rc, tee_rc


def process_absent(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def seal_timeout_receipt(root, child_rc):
    sim = root / "sim.log"
    sim.write_text("Chronologic VCS simulator runtime banner only\n", encoding="utf-8")
    receipt = {
        "schema": "m849_r20_fake_timeout_receipt_v1",
        "status": "FAILED_DO_NOT_CITE",
        "kind": "failure",
        "phase": "infrastructure_timeout_before_verilog_time",
        "runner_exit_rc": 124,
        "child_rc": child_rc,
        "failure_message": "infrastructure_timeout_before_verilog_time",
        "paper_citable": False,
        "functional_vcs_verified": False,
        "attribute_to_rtl": False,
    }
    receipt_path = root / "RUN_FAILED_OR_INCOMPLETE.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    marker = root / "FAILED_DO_NOT_CITE"
    marker.write_text(
        "FAILED_DO_NOT_CITE phase=infrastructure_timeout_before_verilog_time "
        "runner_rc=124 child_rc={} monitor_status=fake_final_ack_pass\n".format(child_rc),
        encoding="utf-8")
    members = [marker, receipt_path, sim]
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join("{}  {}\n".format(sha256(p), p.name) for p in members),
                        encoding="utf-8")
    seal = root / "SHA256SUMS.seal.sha256"
    seal.write_text("{}  SHA256SUMS\n".format(sha256(manifest)), encoding="utf-8")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if sha256(root / name) != digest:
            raise RuntimeError("member seal mismatch: " + name)
    if sha256(manifest) != seal.read_text(encoding="utf-8").split()[0]:
        raise RuntimeError("outer seal mismatch")
    return sha256(receipt_path), sha256(manifest), sha256(seal)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runner", type=Path)
    args = parser.parse_args()
    runner = args.runner.resolve(strict=True)
    if runner.is_symlink() or not runner.is_file():
        raise RuntimeError("runner must be a regular non-symlink file")
    text = runner.read_text(encoding="utf-8")
    production_lines = [line for line in text.splitlines()
                        if line == PRODUCTION_LITERAL + " 2>&1 | tee sim.log"]
    if len(production_lines) != 1:
        raise RuntimeError("production timeout command line must occur exactly once")
    if sha256(TIMEOUT) != TIMEOUT_SHA256:
        raise RuntimeError("/usr/bin/timeout SHA drift")
    forbidden_env = ("SNPS_TELEMETRY", "VCS_TELEMETRY", "DISABLE_TELEMETRY",
                     "TELEMETRY_DISABLE")
    if any(token in text for token in forbidden_env):
        raise RuntimeError("unverified telemetry environment variable present")

    with tempfile.TemporaryDirectory(prefix="m849_r20_timeout_fake_simv.") as raw:
        root = Path(raw)

        fast = root / "fast_simv"
        make_fake(fast, """import sys
if sys.argv[1:] != ['-no_save']:
    raise SystemExit(9)
print('PASS_M533_M528_DW1RW_R8_DIRECTED_RANDOM_AND_ATTACKS fake=1')
raise SystemExit(0)""")
        fast_log = root / "fast.log"
        fast_rc = run_pipeline(fast, fast_log, timeout_seconds="1s", kill_after="0.2s")
        if fast_rc != (0, 0) or "PASS_M533" not in fast_log.read_text(encoding="utf-8"):
            raise RuntimeError("fast-pass pipeline mismatch: {}".format(fast_rc))

        term_marker = root / "term.marker"
        term_pid = root / "term.pid"
        term = root / "term_simv"
        make_fake(term, """import os, signal, sys, time
pid_path = {!r}; marker = {!r}
open(pid_path, 'w').write(str(os.getpid()))
def on_term(signum, frame):
    open(marker, 'w').write('TERM')
    raise SystemExit(143)
signal.signal(signal.SIGTERM, on_term)
while True: time.sleep(1)""".format(str(term_pid), str(term_marker)))
        term_log = root / "term.log"
        term_rc = run_pipeline(term, term_log)
        if term_rc != (124, 0) or term_marker.read_text(encoding="utf-8") != "TERM":
            raise RuntimeError("TERM timeout mismatch: {}".format(term_rc))
        term_pid_value = int(term_pid.read_text(encoding="utf-8"))
        if not process_absent(term_pid_value):
            raise RuntimeError("TERM fake-simv orphaned")

        kill_pid = root / "kill.pid"
        kill = root / "kill_simv"
        make_fake(kill, """import os, signal, time
open({!r}, 'w').write(str(os.getpid()))
signal.signal(signal.SIGTERM, signal.SIG_IGN)
while True: time.sleep(1)""".format(str(kill_pid)))
        kill_log = root / "kill.log"
        kill_rc = run_pipeline(kill, kill_log)
        if kill_rc != (137, 0):
            raise RuntimeError("TERM-to-KILL timeout mismatch: {}".format(kill_rc))
        kill_pid_value = int(kill_pid.read_text(encoding="utf-8"))
        for _ in range(20):
            if process_absent(kill_pid_value):
                break
            time.sleep(0.02)
        if not process_absent(kill_pid_value):
            raise RuntimeError("KILL fake-simv orphaned")

        bad_tee = root / "bad_tee"
        make_fake(bad_tee, """import sys
sys.stdin.read()
raise SystemExit(7)""")
        tee_rc = run_pipeline(fast, root / "unused.log", timeout_seconds="1s",
                              kill_after="0.2s", tee=bad_tee)
        if tee_rc != (0, 7):
            raise RuntimeError("tee rc propagation mismatch: {}".format(tee_rc))

        receipt_sha, manifest_sha, outer_sha = seal_timeout_receipt(
            root / "sealed" if False else root, "simv_timeout_124_tee_0")
        result = {
            "schema": "m849_r20_timeout_fake_simv_test_v1",
            "status": "PASS_SOURCE_ONLY_TIMEOUT_FAKE_SIMV",
            "production_literal_exact_once": True,
            "timeout_sha256": TIMEOUT_SHA256,
            "fast_pass_pipe_rc": list(fast_rc),
            "term_timeout_pipe_rc": list(term_rc),
            "term_process_absent": True,
            "kill_timeout_pipe_rc": list(kill_rc),
            "kill_process_absent": True,
            "tee_failure_pipe_rc": list(tee_rc),
            "timeout_pre_hdl_classification": "infrastructure_timeout_before_verilog_time",
            "receipt_double_seal_pass": True,
            "fake_receipt_sha256": receipt_sha,
            "fake_manifest_sha256": manifest_sha,
            "fake_outer_seal_file_sha256": outer_sha,
            "telemetry_environment_variables_added": 0,
            "vcs_runs": 0,
            "license_queries": 0,
            "eda_runs": 0,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
