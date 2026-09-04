#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Local-only different-author review for M1400; never touches remote/GPU/capture."""
from __future__ import annotations

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
SOURCE = HW / "scripts/run_m1400_m1349_motion_ep34_live105_production_one_shot.py"
TEST = HW / "tests/test_run_m1400_m1349_motion_ep34_live105_production_one_shot.py"
CONTRACT = HW / "contracts/m1400_m1349_motion_ep34_live105_production_runner_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1400_m1349_motion_ep34_live105_production_runner_source_author_r1_20260831"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("import spec failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load("m1410_bound_m1400", SOURCE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed(stdout: str = "", returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def expect_reject(thunk) -> bool:
    try:
        thunk()
    except Exception:
        return True
    return False


def controller_fixture(root: Path, *, state: str = "T", ppid: int = 1,
                       argv=None, count: int = 1) -> None:
    argv = tuple(argv or M.CONTROLLER_ARGV)
    for offset in range(count):
        pid = root / str(700 + offset); pid.mkdir()
        (pid / "cmdline").write_bytes(b"\0".join(x.encode() for x in argv) + b"\0")
        fields = [state, str(ppid)] + ["0"] * 17 + [str(9000 + offset)]
        (pid / "stat").write_text(f"{pid.name} (python) " + " ".join(fields) + "\n")
        (pid / "cwd").touch(); (pid / "exe").touch()


def main() -> int:
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, category: str) -> None:
        checks.append({"check": name, "category": category, "pass": bool(condition)})

    # Frozen source, test, contract and predecessor pins.
    contract = M.strict_json(CONTRACT)
    check("runner_sha", sha(SOURCE) == "c9d7e0e3d6eca16c710b8bbcf44be3154f1891eb8b3b8452d3fda1a5094668be", "identity")
    check("test_sha", sha(TEST) == "7c90fc9c68fab5cef0c3430b1b06689f3c5dc93f55d0959a4d9d35a6bd52220e", "identity")
    check("contract_sha", sha(CONTRACT) == "a0f4e6661ab9709cd905ac77c17abce0fe00faae341c3c64e706a9eea23ed1ef", "identity")
    check("m1349_source_sha", sha(M.M1349_SOURCE) == M.M1349_SOURCE_SHA256, "identity")
    check("m1349_test_sha", sha(M.M1349_TEST) == M.M1349_TEST_SHA256, "identity")
    check("m1349_contract_sha", sha(M.M1349_CONTRACT) == M.M1349_CONTRACT_SHA256, "identity")
    check("m1347_inventory_sha", sha(M.M1347_INVENTORY) == M.M1347_INVENTORY_SHA256, "identity")
    check("docs359_sha", sha(M.DOCS359) == M.DOCS359_SHA256, "identity")
    M.verify_prerequisites()
    check("prerequisite_semantics", True, "identity")
    check("live105_count", M.M1349.EXPECTED_ATLIF_COUNT == 105, "identity")
    names = M.M1349.verify_m1347_failure()
    check("live105_inventory_count", len(names) == 105, "identity")
    check("live105_terminal_digest", M.M1349.terminal_lf_digest(list(names)) == M.M1349.EXPECTED_ATLIF_NAMES_SHA256, "identity")
    check("ordered_records", M.M1349.EXPECTED_ORDERED_RECORDS == 10360, "identity")
    check("payload", M.M1349.EXPECTED_PAYLOAD == 640, "identity")
    check("source_contract", M.validate_source_contract() == contract, "identity")

    # Author evidence has exact bindings and no launch authority.
    author_review = M.strict_json(AUTHOR / "review.json")
    check("author_runner_binding", author_review["bindings"]["runner_sha256"] == sha(SOURCE), "author")
    check("author_test_binding", author_review["bindings"]["test_sha256"] == sha(TEST), "author")
    check("author_contract_binding", author_review["bindings"]["contract_sha256"] == sha(CONTRACT), "author")
    check("author_no_launch", author_review["authorization"] == {
        "fresh_different_author_blind": True, "release_authoring": False,
        "launch": False, "remote": False, "gpu": False, "forward": False,
        "capture": False, "attempt": False, "controller_restore": False}, "author")
    check("author_claim_boundary", author_review["claim_boundary"] == {
        "source_and_tests_only": True, "production_result": False,
        "hardware_result": False, "cycles": False, "speedup": False,
        "energy": False, "ppa": False}, "author")

    # Contract binds exact controller and A800 identity plus one-shot namespaces.
    rr = contract["remote_runtime"]
    check("controller_argv_contract", tuple(rr["controller_argv"]) == M.CONTROLLER_ARGV, "contract")
    check("controller_exe_contract", rr["controller_exe"] == M.CONTROLLER_EXE, "contract")
    check("controller_stop_contract", rr["controller_ppid"] == 1 and rr["controller_state"] == "T", "contract")
    check("gpu_uuid_contract", rr["gpu_uuid"] == M.GPU_UUID, "contract")
    check("gpu_name_contract", rr["gpu_name"] == M.GPU_NAME, "contract")
    check("gpu_capacity_contract", rr["gpu_total_mib"] == M.GPU_TOTAL_MIB, "contract")
    check("gpu_idle_contract", rr["gpu_used_limit_mib"] == M.GPU_USED_LIMIT_MIB and rr["compute_apps"] == 0, "contract")
    one = contract["one_shot"]
    check("three_namespace_contract", (one["result"], one["attempt"], one["log"]) == tuple(
        str(p.relative_to(ROOT)) for p in (M.CANONICAL_RESULT, M.CANONICAL_ATTEMPT, M.CANONICAL_LOG)), "contract")
    check("one_shot_contract", one["attempt_create"] == "O_EXCL" and one["runs"] == 1 and one["automatic_retry"] is False, "contract")
    check("source_no_launch_contract", contract["launch_authorized"] is False and
          all(v is False for v in contract["author_execution"].values()), "contract")

    # Future release/final and production namespaces remain absent. M1410 is this review.
    check("m1412_absent", not os.path.lexists(str(M.FUTURE_RELEASE)), "freshness")
    check("m1430_absent", not os.path.lexists(str(M.FUTURE_FINAL)), "freshness")
    check("result_absent", not os.path.lexists(str(M.CANONICAL_RESULT)), "freshness")
    check("attempt_absent", not os.path.lexists(str(M.CANONICAL_ATTEMPT)), "freshness")
    check("log_absent", not os.path.lexists(str(M.CANONICAL_LOG)), "freshness")

    # Controller checks use synthetic /proc only.
    with tempfile.TemporaryDirectory(prefix="m1410_proc_") as raw:
        proc = Path(raw); controller_fixture(proc)
        def links(path):
            return M.CONTROLLER_EXE if Path(path).name == "exe" else str(M.REMOTE_ROOT)
        with mock.patch.object(M.os, "readlink", side_effect=links):
            exact = M.inspect_controller(proc)
        check("controller_exact_accept", exact["state"] == "T" and exact["ppid"] == 1 and exact["start_ticks"] == 9000, "controller")
    for name, kwargs in (("running", {"state": "S"}), ("wrong_ppid", {"ppid": 2}),
                         ("duplicate", {"count": 2}),
                         ("wrong_argv", {"argv": ("/bin/false",)})):
        with tempfile.TemporaryDirectory(prefix="m1410_proc_bad_") as raw:
            proc = Path(raw); controller_fixture(proc, **kwargs)
            with mock.patch.object(M.os, "readlink", side_effect=lambda p: M.CONTROLLER_EXE if Path(p).name == "exe" else str(M.REMOTE_ROOT)):
                check("controller_reject_" + name, expect_reject(lambda proc=proc: M.inspect_controller(proc)), "controller")
    with tempfile.TemporaryDirectory(prefix="m1410_proc_badlink_") as raw:
        proc = Path(raw); controller_fixture(proc)
        with mock.patch.object(M.os, "readlink", side_effect=lambda p: "/wrong" if Path(p).name == "cwd" else M.CONTROLLER_EXE):
            check("controller_reject_wrong_cwd", expect_reject(lambda: M.inspect_controller(proc)), "controller")
        with mock.patch.object(M.os, "readlink", side_effect=lambda p: "/wrong/python" if Path(p).name == "exe" else str(M.REMOTE_ROOT)):
            check("controller_reject_wrong_exe", expect_reject(lambda: M.inspect_controller(proc)), "controller")

    # GPU checks use a fake subprocess runner only.
    def gpu_runner(command, **_kwargs):
        if "--query-gpu=index,uuid,name,memory.used,memory.total" in command:
            return completed(f"0, {M.GPU_UUID}, {M.GPU_NAME}, 0, {M.GPU_TOTAL_MIB}\n")
        return completed("")
    gpu = M.inspect_gpu(gpu_runner)
    check("gpu_exact_accept", gpu["uuid"] == M.GPU_UUID and gpu["compute_apps"] == [], "gpu")
    def bad_gpu(row: str, apps: str = "", gpu_rc: int = 0, app_rc: int = 0):
        def runner(command, **_kwargs):
            if "--query-gpu=index,uuid,name,memory.used,memory.total" in command:
                return completed(row, gpu_rc)
            return completed(apps, app_rc)
        return runner
    bad_rows = {
        "index": f"1, {M.GPU_UUID}, {M.GPU_NAME}, 0, {M.GPU_TOTAL_MIB}\n",
        "uuid": f"0, GPU-wrong, {M.GPU_NAME}, 0, {M.GPU_TOTAL_MIB}\n",
        "name": f"0, {M.GPU_UUID}, Other GPU, 0, {M.GPU_TOTAL_MIB}\n",
        "total": f"0, {M.GPU_UUID}, {M.GPU_NAME}, 0, 1\n",
        "busy": f"0, {M.GPU_UUID}, {M.GPU_NAME}, 65, {M.GPU_TOTAL_MIB}\n",
        "extra_row": f"0, {M.GPU_UUID}, {M.GPU_NAME}, 0, {M.GPU_TOTAL_MIB}\n0, x, y, 0, 1\n",
        "malformed": "not,a,valid,row\n",
    }
    for name, row in bad_rows.items():
        check("gpu_reject_" + name, expect_reject(lambda row=row: M.inspect_gpu(bad_gpu(row))), "gpu")
    check("gpu_reject_app", expect_reject(lambda: M.inspect_gpu(bad_gpu(
        f"0, {M.GPU_UUID}, {M.GPU_NAME}, 0, {M.GPU_TOTAL_MIB}\n", f"99, {M.GPU_UUID}\n"))), "gpu")
    check("gpu_reject_query_failure", expect_reject(lambda: M.inspect_gpu(bad_gpu("", gpu_rc=1))), "gpu")
    check("gpu_reject_app_query_failure", expect_reject(lambda: M.inspect_gpu(bad_gpu(
        f"0, {M.GPU_UUID}, {M.GPU_NAME}, 0, {M.GPU_TOTAL_MIB}\n", app_rc=1))), "gpu")

    # O_EXCL attempt and three namespace collision gates, all in temp directories.
    with tempfile.TemporaryDirectory(prefix="m1410_attempt_") as raw:
        marker = Path(raw) / "attempt"
        values = {"M1400_EXPECTED_RUNNER_SHA256": sha(SOURCE)}
        controller = {"pid": 1, "state": "T"}
        with mock.patch.object(M, "CANONICAL_ATTEMPT", marker):
            M.consume_attempt(controller, values)
            attempt = M.strict_json(marker)
            check("attempt_mode_0400", marker.stat().st_mode & 0o777 == 0o400, "attempt")
            check("attempt_no_retry", attempt["automatic_retry"] is False, "attempt")
            check("attempt_no_restore", attempt["controller_restore_permitted"] is False, "attempt")
            check("attempt_runner_bound", attempt["runner_sha256"] == sha(SOURCE), "attempt")
            check("attempt_O_EXCL", expect_reject(lambda: M.consume_attempt(controller, values)), "attempt")
    for occupied in ("result", "attempt", "log"):
        with tempfile.TemporaryDirectory(prefix="m1410_ns_") as raw:
            root = Path(raw); paths = {name: root / name for name in ("result", "attempt", "log")}
            paths[occupied].touch()
            with mock.patch.object(M, "CANONICAL_RESULT", paths["result"]), \
                 mock.patch.object(M, "CANONICAL_ATTEMPT", paths["attempt"]), \
                 mock.patch.object(M, "CANONICAL_LOG", paths["log"]):
                check("namespace_reject_" + occupied, expect_reject(M.namespaces_fresh), "freshness")

    # Failure never restores; success only records permission for a later actor.
    controller = {"pid": 1, "state": "T"}
    failed = json.loads(M.log_payload("FAIL", controller, "failure"))
    passed = json.loads(M.log_payload("PASS", controller, "success"))
    check("failure_restore_forbidden", failed["controller_restore_permitted"] is False and
          failed["controller_restore_permitted_after_success"] is False, "restore")
    check("failure_runner_did_not_restore", failed["controller_restored_by_runner"] is False, "restore")
    check("success_later_restore_only", passed["controller_restore_permitted"] is True and
          passed["controller_restore_permitted_after_success"] is True and
          passed["controller_restored_by_runner"] is False, "restore")
    source_text = SOURCE.read_text()
    check("no_os_kill", "os.kill" not in source_text, "restore")
    check("no_sigcont", "SIGCONT" not in source_text, "restore")
    check("no_send_signal", "send_signal" not in source_text, "restore")

    # CLI dispatch is exercised only with mocks; no remote function executes.
    with mock.patch.object(M, "remote_preflight") as preflight, \
         mock.patch.object(sys, "argv", [str(SOURCE), "--remote-preflight"]):
        with contextlib.redirect_stdout(io.StringIO()):
            cli_preflight_rc = M.main()
        check("cli_preflight_dispatch", cli_preflight_rc == 0 and preflight.call_count == 1, "cli")
    with tempfile.TemporaryDirectory(prefix="m1410_cli_") as raw:
        temp = Path(raw) / "production.log.tmp.x"
        with mock.patch.object(M, "execute_once", return_value=Path("result")) as execute, \
             mock.patch.object(sys, "argv", [str(SOURCE), "--run", "--temporary-log", str(temp)]):
            with contextlib.redirect_stdout(io.StringIO()):
                cli_run_rc = M.main()
            check("cli_run_dispatch", cli_run_rc == 0 and execute.call_count == 1, "cli")
    with mock.patch.object(sys, "argv", [str(SOURCE), "--run"]):
        check("cli_run_requires_temp", expect_reject(M.main), "cli")
    check("production_claim_boundary", contract["launch_authorized"] is False and
          author_review["claim_boundary"]["production_result"] is False and
          author_review["claim_boundary"]["hardware_result"] is False, "claims")

    failed_checks = [row["check"] for row in checks if not row["pass"]]
    by_category: dict[str, dict[str, int]] = {}
    for row in checks:
        item = by_category.setdefault(str(row["category"]), {"checks": 0, "passed": 0, "failed": 0})
        item["checks"] += 1; item["passed" if row["pass"] else "failed"] += 1
    output = {
        "schema": "m1410_m1400_ep34_live105_runner_blind_hammer_r1_v1",
        "status": "PASS" if not failed_checks else "FAIL_DO_NOT_CITE",
        "check_count": len(checks),
        "passed_count": len(checks) - len(failed_checks),
        "failed_count": len(failed_checks),
        "failed_checks": failed_checks,
        "categories": by_category,
        "remote_runs": 0,
        "gpu_runs": 0,
        "capture_runs": 0,
        "attempts_consumed": 0,
        "controller_restores": 0,
        "checks": checks,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if not failed_checks else 1


if __name__ == "__main__":
    raise SystemExit(main())
