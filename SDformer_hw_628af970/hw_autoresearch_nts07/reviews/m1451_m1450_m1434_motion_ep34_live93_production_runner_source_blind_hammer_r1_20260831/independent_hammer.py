#!/opt/anaconda3/bin/python
"""Local-only different-author blind hammer for the sealed M1450 runner.

No function in this program opens SSH, queries a real GPU, consumes a production
attempt, signals a process, or invokes EDA.  Runtime identities are attacked with
synthetic ``/proc`` trees and fake ``nvidia-smi`` subprocess results.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "scripts/run_m1450_m1434_motion_ep34_live93_production_one_shot.py"
TEST = HW / "tests/test_run_m1450_m1434_motion_ep34_live93_production_one_shot.py"
CONTRACT = HW / (
    "contracts/m1450_m1434_motion_ep34_live93_production_runner_source_"
    "contract_r1_20260831.json")
AUTHOR = HW / (
    "reviews/m1450_m1434_motion_ep34_live93_production_runner_source_author_"
    "r1_20260831")
RUNNER_SHA = "58b58876f9dd198f5c4b4c1b6bdaf3ef280d77dc87272aa6e38c4f4cedc7d098"
TEST_SHA = "97721136587650b5b82d2b44c361d9ad4385c462e61a50ab263a069d02f2d05c"
CONTRACT_SHA = "a79a1b16d9b1d3445daf04315fadeab255809d9f36c263b3c8e21014996ec000"
AUTHOR_REVIEW_SHA = "076027ea43add1dd6d6c5590705fc72929bc473826ab2be9b96e6bed7c349ec4"
AUTHOR_MANIFEST_SHA = "e61e35165fef114b79b423d05f239fc9c9e1d955b5c63c45031de9922c0cdd28"
AUTHOR_OUTER_SHA = "ba244e7e7dc3dafad1a60dc5c329693c9894ecb5b4b2e2acf58bf067e5327abf"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("import spec failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load("m1451_bound_m1450", SOURCE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed(stdout: str = "", returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def rejects(thunk) -> bool:
    try:
        thunk()
    except BaseException:
        return True
    return False


def controller_fixture(root: Path, *, pid: int = M.CONTROLLER_PID,
                       state: str = "T", ppid: int = 1,
                       start: int = M.CONTROLLER_START_TICKS,
                       argv=None, count: int = 1) -> list[Path]:
    roots = []
    argv = tuple(M.CONTROLLER_ARGV if argv is None else argv)
    for offset in range(count):
        item = root / str(pid + offset)
        item.mkdir()
        (item / "cmdline").write_bytes(b"\0".join(x.encode() for x in argv) + b"\0")
        fields = [state, str(ppid)] + ["0"] * 17 + [str(start + offset)]
        (item / "stat").write_text(
            f"{item.name} (python controller) " + " ".join(fields) + "\n",
            encoding="utf-8")
        (item / "cwd").touch()
        (item / "exe").touch()
        roots.append(item)
    return roots


def main() -> int:
    checks: list[dict[str, object]] = []
    attacks: list[dict[str, object]] = []

    def check(name: str, value: bool, category: str) -> None:
        checks.append({"check": name, "category": category, "pass": bool(value)})

    def attack(name: str, thunk, category: str) -> None:
        caught = rejects(thunk)
        attacks.append({"attack": name, "category": category,
                        "rejected": caught, "false_negative": not caught})

    # Exact source and all sealed predecessor identities.
    contract = M.strict_json(CONTRACT)
    check("runner_exact_sha", sha(SOURCE) == RUNNER_SHA, "identity")
    check("test_exact_sha", sha(TEST) == TEST_SHA, "identity")
    check("contract_exact_sha", sha(CONTRACT) == CONTRACT_SHA, "identity")
    check("m1434_source_exact", sha(M.M1434_SOURCE) == M.M1434_SOURCE_SHA256,
          "identity")
    check("m1434_test_exact", sha(M.M1434_TEST) == M.M1434_TEST_SHA256,
          "identity")
    check("m1434_contract_exact", sha(M.M1434_CONTRACT) == M.M1434_CONTRACT_SHA256,
          "identity")
    check("m1435_review_exact", sha(M.M1435_BLIND / "review.json") == M.M1435_REVIEW_SHA256,
          "identity")
    check("m1435_manifest_exact", sha(M.M1435_BLIND / "SHA256SUMS") == M.M1435_MANIFEST_SHA256,
          "identity")
    check("m1435_outer_exact", sha(M.M1435_BLIND / "SHA256SUMS.seal.sha256") == M.M1435_OUTER_SHA256,
          "identity")
    check("docs359_exact", sha(M.DOCS359) == M.DOCS359_SHA256, "identity")
    M.verify_prerequisites()
    check("predecessor_semantics", True, "identity")
    check("source_contract_semantics", M.validate_source_contract() == contract,
          "identity")

    author = M.strict_json(AUTHOR / "review.json")
    check("author_review_exact", sha(AUTHOR / "review.json") == AUTHOR_REVIEW_SHA,
          "author")
    check("author_manifest_exact", sha(AUTHOR / "SHA256SUMS") == AUTHOR_MANIFEST_SHA,
          "author")
    check("author_outer_exact", sha(AUTHOR / "SHA256SUMS.seal.sha256") == AUTHOR_OUTER_SHA,
          "author")
    check("author_runner_binding", author["bindings"]["runner_sha256"] == RUNNER_SHA,
          "author")
    check("author_test_binding", author["bindings"]["test_sha256"] == TEST_SHA,
          "author")
    check("author_contract_binding", author["bindings"]["contract_sha256"] == CONTRACT_SHA,
          "author")
    check("author_no_launch", author["authorization"]["launch"] is False and
          author["authorization"]["controller_restore"] is False,
          "author")

    # First-principles live graph: static 259/105 minus the exact 12 H60-dead
    # sn2_q nodes is live 247/93, hence exactly 9880 rows for forty samples.
    policy = M.M1434.R1.strict_json(M.M1434.R1.SOURCE_CONTRACT)
    static = M.M1434.R1.frozen_non_atlif_inventory(policy)
    static["atlif"] = list(M.M1434.M1349.EXPECTED_ATLIF_NAMES)
    live = M.M1434.expected_live93_inventory(static)
    dead = set(static["atlif"]) - set(live["atlif"])
    check("static_modules_259", sum(map(len, static.values())) == 259, "population")
    check("static_atlif_105", len(static["atlif"]) == 105, "population")
    check("live_modules_247", sum(map(len, live.values())) == 247, "population")
    check("live_atlif_93", len(live["atlif"]) == 93, "population")
    check("dead_exact_12", dead == set(M.M1434.DEAD_SN2_Q) and len(dead) == 12,
          "population")
    check("dead_digest", M.M1434.terminal_lf_digest(sorted(dead)) ==
          M.M1434.DEAD_SN2_Q_SHA256, "population")
    check("live_digest", M.M1434.terminal_lf_digest(live["atlif"]) ==
          M.M1434.LIVE_ATLIF_SHA256, "population")
    check("ordered_9880", 40 * sum(map(len, live.values())) == 9880 ==
          M.M1434.EXPECTED_ORDERED_RECORDS, "population")
    check("payload_640_attention_480", M.M1434.EXPECTED_PAYLOAD == 640 and
          M.M1434.EXPECTED_ATTENTION == 480, "population")
    replay = M.M1434.replay_sample0_forensic_summary()
    check("sample0_forensic_replay", replay == {
        "status": "PASS", "errors": [], "samples": 1,
        "live_modules_per_sample": 247, "records": 247,
        "expected_records": 247, "dead_modules": 12}, "population")

    rr = contract["remote_runtime"]
    exact_runtime = {
        "uid": 0, "repository": str(M.REMOTE_ROOT),
        "controller_pid": M.CONTROLLER_PID,
        "controller_start_ticks": M.CONTROLLER_START_TICKS,
        "controller_argv": list(M.CONTROLLER_ARGV),
        "controller_exe": M.CONTROLLER_EXE, "controller_ppid": 1,
        "controller_state": "T", "gpu_index": 0, "gpu_uuid": M.GPU_UUID,
        "gpu_name": M.GPU_NAME, "gpu_total_mib": M.GPU_TOTAL_MIB,
        "gpu_used_limit_mib": M.GPU_USED_LIMIT_MIB, "compute_apps": 0}
    check("remote_runtime_exact", rr == exact_runtime, "contract")
    check("one_shot_exact", contract["one_shot"] == {
        "result": str(M.CANONICAL_RESULT.relative_to(ROOT)),
        "attempt": str(M.CANONICAL_ATTEMPT.relative_to(ROOT)),
        "log": str(M.CANONICAL_LOG.relative_to(ROOT)),
        "attempt_create": "O_EXCL", "attempt_before_capture": True,
        "exclusive_gpu_lease": True, "runs": 1, "automatic_retry": False},
          "contract")
    check("author_stage_inert", contract["launch_authorized"] is False and
          all(value is False for value in contract["author_execution"].values()),
          "contract")

    # Current blind directory is expected to exist; later release/final and all
    # production namespaces must remain absent during this source-only review.
    check("future_release_absent", not os.path.lexists(str(M.FUTURE_RELEASE)), "freshness")
    check("future_final_absent", not os.path.lexists(str(M.FUTURE_FINAL)), "freshness")
    check("result_absent", not os.path.lexists(str(M.CANONICAL_RESULT)), "freshness")
    check("attempt_absent", not os.path.lexists(str(M.CANONICAL_ATTEMPT)), "freshness")
    check("log_absent", not os.path.lexists(str(M.CANONICAL_LOG)), "freshness")

    # Synthetic controller positive case and exhaustive identity mutations.
    def links(path, cwd=str(M.REMOTE_ROOT), exe=M.CONTROLLER_EXE):
        return exe if Path(path).name == "exe" else cwd

    with tempfile.TemporaryDirectory(prefix="m1451_proc_ok_") as raw:
        proc = Path(raw)
        controller_fixture(proc)
        with mock.patch.object(M.os, "readlink", side_effect=links):
            observed = M.inspect_controller(proc)
        check("controller_exact_accept", observed == {
            "pid": M.CONTROLLER_PID, "ppid": 1, "state": "T",
            "start_ticks": M.CONTROLLER_START_TICKS,
            "cwd": str(M.REMOTE_ROOT), "exe": M.CONTROLLER_EXE,
            "argv": list(M.CONTROLLER_ARGV)}, "controller")

    controller_mutations = [
        ("pid_low", {"pid": M.CONTROLLER_PID - 1}),
        ("pid_high", {"pid": M.CONTROLLER_PID + 1}),
        ("start_low", {"start": M.CONTROLLER_START_TICKS - 1}),
        ("start_high", {"start": M.CONTROLLER_START_TICKS + 1}),
        ("running_S", {"state": "S"}), ("running_R", {"state": "R"}),
        ("zombie_Z", {"state": "Z"}), ("ppid_zero", {"ppid": 0}),
        ("ppid_two", {"ppid": 2}), ("duplicate", {"count": 2}),
        ("empty_argv", {"argv": ()}),
        ("wrong_python", {"argv": ("/bin/python", "-u", M.CONTROLLER_SCRIPT)}),
        ("missing_dash_u", {"argv": (M.CONTROLLER_ARGV[0], M.CONTROLLER_SCRIPT)}),
        ("wrong_script", {"argv": (M.CONTROLLER_ARGV[0], "-u", "wrong.py")}),
        ("extra_arg", {"argv": M.CONTROLLER_ARGV + ("--extra",)}),
    ]
    for name, kwargs in controller_mutations:
        with tempfile.TemporaryDirectory(prefix="m1451_proc_bad_") as raw:
            proc = Path(raw)
            controller_fixture(proc, **kwargs)
            with mock.patch.object(M.os, "readlink", side_effect=links):
                attack("controller_" + name,
                       lambda proc=proc: M.inspect_controller(proc), "controller")
    for name, cwd, exe in (
            ("wrong_cwd", "/wrong/repo", M.CONTROLLER_EXE),
            ("wrong_exe", str(M.REMOTE_ROOT), "/wrong/python")):
        with tempfile.TemporaryDirectory(prefix="m1451_proc_link_") as raw:
            proc = Path(raw)
            controller_fixture(proc)
            with mock.patch.object(M.os, "readlink",
                                   side_effect=lambda p, c=cwd, e=exe: links(p, c, e)):
                attack("controller_" + name,
                       lambda proc=proc: M.inspect_controller(proc), "controller")

    # Fake-only A800 inspection and mutations.
    def gpu_runner(command, **_kwargs):
        if "--query-gpu=index,uuid,name,memory.used,memory.total" in command:
            return completed(f"0, {M.GPU_UUID}, {M.GPU_NAME}, 0, {M.GPU_TOTAL_MIB}\n")
        return completed("")

    gpu = M.inspect_gpu(gpu_runner)
    check("gpu_exact_accept", gpu["uuid"] == M.GPU_UUID and
          gpu["memory_total_mib"] == M.GPU_TOTAL_MIB and not gpu["compute_apps"],
          "gpu")

    def fake_gpu(row: str, apps: str = "", gpu_rc: int = 0, app_rc: int = 0):
        def runner(command, **_kwargs):
            if "--query-gpu=index,uuid,name,memory.used,memory.total" in command:
                return completed(row, gpu_rc)
            return completed(apps, app_rc)
        return runner

    gpu_rows = {
        "index": f"1, {M.GPU_UUID}, {M.GPU_NAME}, 0, {M.GPU_TOTAL_MIB}\n",
        "uuid": f"0, GPU-wrong, {M.GPU_NAME}, 0, {M.GPU_TOTAL_MIB}\n",
        "name": f"0, {M.GPU_UUID}, Other, 0, {M.GPU_TOTAL_MIB}\n",
        "used_65": f"0, {M.GPU_UUID}, {M.GPU_NAME}, 65, {M.GPU_TOTAL_MIB}\n",
        "used_negative": f"0, {M.GPU_UUID}, {M.GPU_NAME}, -1, {M.GPU_TOTAL_MIB}\n",
        "total": f"0, {M.GPU_UUID}, {M.GPU_NAME}, 0, 80000\n",
        "extra": (f"0, {M.GPU_UUID}, {M.GPU_NAME}, 0, {M.GPU_TOTAL_MIB}\n"
                  f"1, x, y, 0, 1\n"),
        "short": "0,x,y,0\n", "nonnumeric": f"0,{M.GPU_UUID},{M.GPU_NAME},x,y\n",
    }
    for name, row in gpu_rows.items():
        attack("gpu_" + name, lambda row=row: M.inspect_gpu(fake_gpu(row)), "gpu")
    valid_gpu = f"0, {M.GPU_UUID}, {M.GPU_NAME}, 0, {M.GPU_TOTAL_MIB}\n"
    attack("gpu_compute_app", lambda: M.inspect_gpu(fake_gpu(
        valid_gpu, f"99, {M.GPU_UUID}\n")), "gpu")
    attack("gpu_query_rc", lambda: M.inspect_gpu(fake_gpu("", gpu_rc=1)), "gpu")
    attack("gpu_app_query_rc", lambda: M.inspect_gpu(fake_gpu(valid_gpu, app_rc=1)), "gpu")

    # Each of eight external SHA gates sees eight malformed mutations.  These
    # attacks terminate before any referenced future file is read.
    malformed = ["", "0", "0" * 63, "0" * 65, "G" * 64,
                 "A" * 64, "z" * 64, "0" * 63 + "\n"]
    for variable in M.ENV_BINDINGS:
        for index, value in enumerate(malformed):
            environment = {name: "0" * 64 for name in M.ENV_BINDINGS}
            environment[variable] = value
            attack(f"external_sha_{variable}_{index}",
                   lambda env=environment: M.external_bindings(env), "external_sha")

    # Namespace and temporary-log mutations use temp paths only.
    for occupied in ("result", "attempt", "log"):
        with tempfile.TemporaryDirectory(prefix="m1451_ns_") as raw:
            root = Path(raw)
            paths = {name: root / name for name in ("result", "attempt", "log")}
            paths[occupied].touch()
            with mock.patch.object(M, "CANONICAL_RESULT", paths["result"]), \
                 mock.patch.object(M, "CANONICAL_ATTEMPT", paths["attempt"]), \
                 mock.patch.object(M, "CANONICAL_LOG", paths["log"]):
                attack("namespace_" + occupied, M.namespaces_fresh, "freshness")

    with tempfile.TemporaryDirectory(prefix="m1451_temp_") as raw:
        root = Path(raw)
        canonical = root / "production.log"
        bad_temps = [Path("relative.tmp"), root / "other.tmp.x",
                     canonical, root.parent / (canonical.name + ".tmp.x")]
        with mock.patch.object(M, "CANONICAL_LOG", canonical):
            for index, path in enumerate(bad_temps):
                attack(f"temporary_log_{index}", lambda p=path: M._temp_ok(p), "log")
            first = root / "production.log.tmp.first"
            M.publish_log(first, b"one\n")
            check("log_hardlink_publish", canonical.read_bytes() == b"one\n", "log")
            attack("log_no_replace", lambda: M.publish_log(
                root / "production.log.tmp.second", b"two\n"), "log")
            check("log_preserved_after_collision", canonical.read_bytes() == b"one\n", "log")

    # Strict JSON attacks.
    bad_json = [
        '{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '[]', 'null',
        '{"a":-Infinity}', '{broken}', '',
    ]
    for index, payload in enumerate(bad_json):
        with tempfile.TemporaryDirectory(prefix="m1451_json_") as raw:
            path = Path(raw) / "x.json"
            path.write_text(payload, encoding="utf-8")
            attack(f"strict_json_{index}", lambda p=path: M.strict_json(p), "json")

    # O_EXCL marker and payload.  No canonical path is touched.
    with tempfile.TemporaryDirectory(prefix="m1451_attempt_") as raw:
        marker = Path(raw) / "attempt"
        controller = {"pid": M.CONTROLLER_PID, "state": "T"}
        values = {"M1450_EXPECTED_RUNNER_SHA256": RUNNER_SHA}
        with mock.patch.object(M, "CANONICAL_ATTEMPT", marker):
            M.consume_attempt(controller, values)
            payload = M.strict_json(marker)
            check("attempt_mode_0400", marker.stat().st_mode & 0o777 == 0o400,
                  "attempt")
            check("attempt_runner_bound", payload["runner_sha256"] == RUNNER_SHA,
                  "attempt")
            check("attempt_no_retry_restore", payload["automatic_retry"] is False and
                  payload["controller_restore_permitted"] is False, "attempt")
            attack("attempt_O_EXCL_reuse", lambda: M.consume_attempt(controller, values),
                   "attempt")

    # Execute-order harness: lease -> rechecks -> O_EXCL -> capture; the result
    # must be double-seal checked only after the lease and before the PASS log.
    events: list[str] = []

    class Substrate:
        @contextlib.contextmanager
        def exclusive_gpu_lease(self, _path):
            events.append("lease_enter")
            yield
            events.append("lease_exit")

    runtime = {"contract_path": "x", "capture": {}, "cohort": {}, "output": {}}
    binding = {"checkpoint_path": "x", "config_path": "y"}
    controller = {"pid": M.CONTROLLER_PID, "state": "T"}
    values = {"M1450_EXPECTED_RUNNER_SHA256": RUNNER_SHA}
    with tempfile.TemporaryDirectory(prefix="m1451_execute_") as raw:
        root = Path(raw)
        canonical = root / "production.log"
        temp = root / "production.log.tmp.test"
        with mock.patch.object(M.os, "geteuid", return_value=0), \
             mock.patch.object(M, "CANONICAL_LOG", canonical), \
             mock.patch.object(M, "remote_preflight",
                               return_value=(runtime, binding, controller, values)), \
             mock.patch.object(M.M1434.R1, "load_substrate", return_value=Substrate()), \
             mock.patch.object(M, "namespaces_fresh", side_effect=lambda: events.append("fresh")), \
             mock.patch.object(M, "inspect_controller",
                               side_effect=lambda: events.append("controller") or controller), \
             mock.patch.object(M, "inspect_gpu",
                               side_effect=lambda: events.append("gpu") or {}), \
             mock.patch.object(M.M1434, "build_runtime",
                               side_effect=lambda: events.append("runtime") or (runtime, binding)), \
             mock.patch.object(M, "validate_bound_capture_files",
                               side_effect=lambda *_: events.append("files")), \
             mock.patch.object(M, "consume_attempt",
                               side_effect=lambda *_: events.append("attempt")), \
             mock.patch.object(M.M1434, "delegate_for_future_release",
                               side_effect=lambda *_a, **_k: events.append("capture") or root / "result"), \
             mock.patch.object(M.M1434.M1249.R1, "verify_double_seal",
                               side_effect=lambda *_: events.append("seal")), \
             mock.patch.object(M, "publish_log",
                               side_effect=lambda *_: events.append("log")):
            M.execute_once(temp)
    expected_order = ["lease_enter", "fresh", "controller", "gpu", "runtime",
                      "files", "attempt", "capture", "lease_exit", "seal", "log"]
    check("execute_exact_order", events == expected_order, "ordering")

    # Both pre-attempt and post-attempt failures fail closed.  The post-attempt
    # case leaves the separately named hidden staging/quarantine to the sealed
    # substrate and publishes a FAIL log that cannot authorize restore/retry.
    with tempfile.TemporaryDirectory(prefix="m1451_pre_fail_") as raw:
        root = Path(raw)
        canonical = root / "production.log"
        with mock.patch.object(M, "CANONICAL_LOG", canonical), \
             mock.patch.object(M, "remote_preflight", side_effect=M.M1450Error("pre")):
            attack("pre_attempt_failure", lambda: M.execute_once(
                root / "production.log.tmp.pre"), "failure")
        check("pre_attempt_no_log", not canonical.exists(), "failure")

    class FailSubstrate:
        @contextlib.contextmanager
        def exclusive_gpu_lease(self, _path):
            yield

    with tempfile.TemporaryDirectory(prefix="m1451_post_fail_") as raw:
        root = Path(raw)
        canonical = root / "production.log"
        temp = root / "production.log.tmp.fail"
        hidden_staging = root / ".m1434.staging"
        hidden_staging.mkdir()
        (hidden_staging / "FAILED.json").write_text(
            '{"status":"FAIL_CLOSED_NO_CANONICAL_RESULT"}\n', encoding="utf-8")
        with mock.patch.object(M.os, "geteuid", return_value=0), \
             mock.patch.object(M, "CANONICAL_LOG", canonical), \
             mock.patch.object(M, "remote_preflight",
                               return_value=(runtime, binding, controller, values)), \
             mock.patch.object(M.M1434.R1, "load_substrate", return_value=FailSubstrate()), \
             mock.patch.object(M, "namespaces_fresh"), \
             mock.patch.object(M, "inspect_controller", return_value=controller), \
             mock.patch.object(M, "inspect_gpu", return_value={}), \
             mock.patch.object(M.M1434, "build_runtime", return_value=(runtime, binding)), \
             mock.patch.object(M, "validate_bound_capture_files"), \
             mock.patch.object(M, "consume_attempt"), \
             mock.patch.object(M.M1434, "delegate_for_future_release",
                               side_effect=RuntimeError("capture failure")):
            attack("post_attempt_capture_failure", lambda: M.execute_once(temp), "failure")
        failed = M.strict_json(canonical)
        check("failure_log_quarantine", failed["status"] == "FAIL" and
              failed["failure_quarantine_required"] is True and
              failed["canonical_result_promotion_permitted"] is False,
              "failure")
        check("failure_no_retry_restore", failed["automatic_retry"] is False and
              failed["controller_restore_permitted"] is False and
              failed["controller_restored_by_runner"] is False, "failure")
        check("failure_staging_retained", hidden_staging.is_dir() and
              not (root / "result").exists(), "failure")

    # Source/AST inspection excludes every process-control primitive and checks
    # that no subprocess command is shell-interpreted.
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    forbidden = ("os.kill", "SIGCONT", "send_signal", "terminate(", "kill(",
                 "ssh ", "scp ", "rsync ", "dc_shell", "vcs ")
    for token in forbidden:
        check("source_forbids_" + token.strip().replace(" ", "_"), token not in text,
              "static")
    check("no_shell_true", all(not any(keyword.arg == "shell" and
          isinstance(keyword.value, ast.Constant) and keyword.value.value is True
          for keyword in node.keywords) for node in calls), "static")
    check("single_capture_call", text.count("M1434.delegate_for_future_release(") == 1,
          "static")
    check("single_attempt_call", text.count("consume_attempt(controller, values)") == 1,
          "static")
    check("double_seal_before_pass_log", text.index(
          "M1434.M1249.R1.verify_double_seal(output)") < text.index(
          'publish_log(temp, log_payload("PASS"'), "static")
    check("attempt_before_capture_static", text.index(
          "consume_attempt(controller, values)") < text.index(
          "M1434.delegate_for_future_release(runtime, binding, substrate)"), "static")

    # CLI is exercised with mocks only; neither real preflight nor production runs.
    with mock.patch.object(M, "remote_preflight") as preflight, \
         mock.patch.object(sys, "argv", [str(SOURCE), "--remote-preflight"]), \
         contextlib.redirect_stdout(io.StringIO()):
        rc = M.main()
    check("cli_preflight_dispatch", rc == 0 and preflight.call_count == 1, "cli")
    with tempfile.TemporaryDirectory(prefix="m1451_cli_") as raw:
        temp = Path(raw) / "production.log.tmp.mock"
        with mock.patch.object(M, "execute_once", return_value=Path("result")) as execute, \
             mock.patch.object(sys, "argv", [str(SOURCE), "--run", "--temporary-log", str(temp)]), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = M.main()
        check("cli_run_dispatch", rc == 0 and execute.call_count == 1, "cli")
    with mock.patch.object(sys, "argv", [str(SOURCE), "--run"]), \
         contextlib.redirect_stderr(io.StringIO()):
        attack("cli_run_requires_temp", M.main, "cli")

    false_negatives = [row["attack"] for row in attacks if row["false_negative"]]
    failed_checks = [row["check"] for row in checks if not row["pass"]]
    categories: dict[str, dict[str, int]] = {}
    for row in checks:
        item = categories.setdefault(str(row["category"]),
                                     {"checks": 0, "passed": 0, "failed": 0})
        item["checks"] += 1
        item["passed" if row["pass"] else "failed"] += 1
    attack_categories: dict[str, dict[str, int]] = {}
    for row in attacks:
        item = attack_categories.setdefault(str(row["category"]),
                                            {"attacks": 0, "rejected": 0,
                                             "false_negatives": 0})
        item["attacks"] += 1
        item["rejected" if row["rejected"] else "false_negatives"] += 1
    passed = not failed_checks and not false_negatives and len(attacks) >= 100
    output = {
        "schema": "m1451_m1450_m1434_live93_runner_blind_hammer_r1_v1",
        "status": "PASS" if passed else "FAIL_DO_NOT_CITE",
        "check_count": len(checks),
        "passed_count": len(checks) - len(failed_checks),
        "failed_count": len(failed_checks),
        "attack_count": len(attacks),
        "rejected_attack_count": len(attacks) - len(false_negatives),
        "false_negative_count": len(false_negatives),
        "failed_checks": failed_checks,
        "false_negatives": false_negatives,
        "categories": categories,
        "attack_categories": attack_categories,
        "checks": checks,
        "attacks": attacks,
        "execution": {"ssh": 0, "remote": 0, "gpu": 0, "capture": 0,
                      "attempts_consumed": 0, "controller_signals": 0,
                      "controller_restores": 0, "eda": 0},
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
