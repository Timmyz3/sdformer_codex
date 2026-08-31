#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Local-only different-author blind mutation hammer for sealed M1458.

This program uses only synthetic subprocess results, synthetic /proc trees and
temporary namespaces.  It does not open SSH, query a real GPU, consume a
production attempt, launch capture/EDA, or signal/restore a controller.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
REVIEW = Path(__file__).resolve().parent
SOURCE = HW / "scripts/run_m1458_m1434_motion_ep34_live93_production_one_shot.py"
TEST = HW / "tests/test_run_m1458_m1434_motion_ep34_live93_production_one_shot.py"
CONTRACT = HW / (
    "contracts/m1458_m1434_motion_ep34_live93_production_runner_source_"
    "contract_r1_20260831.json")
AUTHOR = HW / (
    "reviews/m1458_m1434_motion_ep34_live93_production_runner_source_author_"
    "r1_20260831")
RUNNER_SHA = "e81c20056dd261619f88884f2f097c9b594887927121d9e599a4f89185d33154"
TEST_SHA = "4a6039a203507fb952ea4cc803261299b69c0bbeab4f031eb937e55a7206ce63"
CONTRACT_SHA = "ae3fa89fe0517578e2ef475c675f1c26160d82fc6356e51b54f79e42960bc0b6"
AUTHOR_REVIEW_SHA = "435d6d075fef043b01d8793d7517d1aeb85fba09cd02ebd8520f258573bf1ebe"
AUTHOR_MANIFEST_SHA = "6690fec9c33c1754c54edfdf5cf2a64a94bc1ec1bb449b4dd9351961622bcbe0"
AUTHOR_OUTER_SHA = "833705c4f148d10950c0f66392248ab8ac93722237bd53dc0b67d87ad01a25cd"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("import spec failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load("m1461_bound_m1458", SOURCE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed(stdout: str = "", returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def rejected(thunk) -> bool:
    try:
        thunk()
    except BaseException:
        return True
    return False


def controller_fixture(root: Path, *, pid: int = M.CONTROLLER_PID,
                       state: str = "T", ppid: int = 1,
                       start: int = M.CONTROLLER_START_TICKS,
                       argv=None, count: int = 1) -> list[Path]:
    argv = tuple(M.CONTROLLER_ARGV if argv is None else argv)
    entries = []
    for offset in range(count):
        item = root / str(pid + offset)
        item.mkdir()
        (item / "cmdline").write_bytes(
            b"\0".join(part.encode() for part in argv) + b"\0")
        fields = [state, str(ppid)] + ["0"] * 17 + [str(start + offset)]
        (item / "stat").write_text(
            f"{item.name} (python controller) " + " ".join(fields) + "\n",
            encoding="utf-8")
        (item / "cwd").touch()
        (item / "exe").touch()
        entries.append(item)
    return entries


def main() -> int:
    checks: list[dict[str, object]] = []
    attacks: list[dict[str, object]] = []

    def check(name: str, value: bool, category: str) -> None:
        checks.append({"check": name, "category": category, "pass": bool(value)})

    def attack(name: str, thunk, category: str) -> None:
        caught = rejected(thunk)
        attacks.append({"attack": name, "category": category,
                        "rejected": caught, "false_negative": not caught})

    # Exact successor identities, immutable failed-predecessor pins, and author seal.
    policy = M.strict_json(CONTRACT)
    author = M.strict_json(AUTHOR / "review.json")
    for name, path, expected in (
            ("runner", SOURCE, RUNNER_SHA), ("test", TEST, TEST_SHA),
            ("contract", CONTRACT, CONTRACT_SHA),
            ("author_review", AUTHOR / "review.json", AUTHOR_REVIEW_SHA),
            ("author_manifest", AUTHOR / "SHA256SUMS", AUTHOR_MANIFEST_SHA),
            ("author_outer", AUTHOR / "SHA256SUMS.seal.sha256", AUTHOR_OUTER_SHA),
            ("m1450_source", M.M1450_SOURCE, M.M1450_SOURCE_SHA256),
            ("m1450_test", M.M1450_TEST, M.M1450_TEST_SHA256),
            ("m1450_contract", M.M1450_CONTRACT, M.M1450_CONTRACT_SHA256),
            ("m1451_fail_review", M.M1451_FAILURE / "review.json",
             M.M1451_FAILURE_REVIEW_SHA256),
            ("m1451_fail_manifest", M.M1451_FAILURE / "SHA256SUMS",
             M.M1451_FAILURE_MANIFEST_SHA256),
            ("m1451_fail_outer", M.M1451_FAILURE / "SHA256SUMS.seal.sha256",
             M.M1451_FAILURE_OUTER_SHA256),
            ("docs359", M.DOCS359, M.DOCS359_SHA256)):
        check(name + "_exact_sha", sha(path) == expected, "identity")
    M.verify_prerequisites()
    check("prerequisites_semantics", True, "identity")
    check("source_contract_semantics", M.validate_source_contract() == policy,
          "identity")
    check("author_exact_bindings", author.get("bindings", {}).get(
          "runner_sha256") == RUNNER_SHA and
          author.get("bindings", {}).get("test_sha256") == TEST_SHA and
          author.get("bindings", {}).get("contract_sha256") == CONTRACT_SHA,
          "author")
    check("author_boundary", author.get("authorization", {}).get("launch") is False and
          author.get("authorization", {}).get("m1461_different_author_blind") is True and
          author.get("authorization", {}).get("m1462_release_authoring") is False,
          "author")
    failed = M.strict_json(M.M1451_FAILURE / "review.json")
    check("m1451_exact_failure", failed.get("status") ==
          "FAIL_DO_NOT_CITE__M1450_GPU_NEGATIVE_MEMORY_FALSE_NEGATIVE" and
          failed.get("verification", {}).get("false_negative", {}).get(
              "minimal_repair") == "require 0 <= used <= GPU_USED_LIMIT_MIB" and
          failed.get("authorization", {}).get("launch") is False,
          "identity")

    # The future release/final and production namespaces must be fresh.  This
    # review directory itself necessarily exists while this hammer executes.
    check("future_release_absent", not os.path.lexists(str(M.FUTURE_RELEASE)),
          "freshness")
    check("future_final_absent", not os.path.lexists(str(M.FUTURE_FINAL)),
          "freshness")
    check("result_absent", not os.path.lexists(str(M.CANONICAL_RESULT)), "freshness")
    check("attempt_absent", not os.path.lexists(str(M.CANONICAL_ATTEMPT)), "freshness")
    check("log_absent", not os.path.lexists(str(M.CANONICAL_LOG)), "freshness")
    check("fresh_m1458_names", all("m1458_m1434_" in path.name for path in
          (M.CANONICAL_RESULT, M.CANONICAL_ATTEMPT, M.CANONICAL_LOG)), "freshness")

    # Direct exact-int closed-interval campaign.  Every non-int coercion and
    # every integer outside [0,64] is expected to reject.
    for value in (-2**63, -65536, -65, -64, -3, -2, -1, 65, 66, 1024, 2**63):
        attack("used_int_" + str(value), lambda value=value: M.validate_used_mib(value),
               "gpu_used_direct")
    for index, value in enumerate((True, False, -1.0, 0.0, 1.0, 64.0, 65.0,
                                   float("nan"), float("inf"), "-1", "0", "64",
                                   "65", b"0", None, [], {})):
        attack("used_coercion_" + str(index),
               lambda value=value: M.validate_used_mib(value), "gpu_used_direct")
    for value in range(65):
        check("used_accept_" + str(value), M.validate_used_mib(value) == value,
              "gpu_used_positive")

    invalid_text = (
        "", "-65536", "-65", "-2", "-1", "-0", "+0", "+1", "00", "01",
        "064", "65.0", "0.0", "1.0", "1e0", "True", "False", "None",
        "NaN", "Inf", "∞", "０", "١", " 0", "0 ", "\t0", "0\n")
    for index, value in enumerate(invalid_text):
        attack("used_text_" + str(index),
               lambda value=value: M.parse_decimal_int(value, "memory.used"),
               "gpu_used_text")
    for value in range(65):
        text = str(value)
        check("used_text_accept_" + text,
              M.parse_decimal_int(text, "memory.used") == value,
              "gpu_used_positive")
    check("used_text_65_parse_then_range_reject",
          M.parse_decimal_int("65", "memory.used") == 65 and
          rejected(lambda: M.validate_used_mib(
              M.parse_decimal_int("65", "memory.used"))), "gpu_used_positive")
    for index, value in enumerate((True, False, -1, 0, 1, 64, 65, 1.0, None, b"0")):
        attack("used_text_type_" + str(index),
               lambda value=value: M.parse_decimal_int(value, "memory.used"),
               "gpu_used_text")

    # Entire inspect_gpu path: fake nvidia-smi only.  This includes the original
    # -1 false negative, other negatives, type-like strings, 65, identity/total,
    # multiple/short rows, query failures and non-empty compute-app output.
    def fake_gpu(row: str, apps: str = "", gpu_rc: int = 0, app_rc: int = 0):
        def runner(command, **_kwargs):
            if "--query-gpu=index,uuid,name,memory.used,memory.total" in command:
                return completed(row, gpu_rc)
            return completed(apps, app_rc)
        return runner

    def row(used="0", *, index="0", uuid=M.GPU_UUID, name=M.GPU_NAME,
            total=str(M.GPU_TOTAL_MIB)):
        return f"{index}, {uuid}, {name}, {used}, {total}\n"

    for used in ("-65536", "-65", "-2", "-1", "-0", "+0", "+1", "00", "01",
                 "65", "65.0", "0.0", "True", "False", "None", "NaN", "Inf",
                 "1e0", "０", "١", ""):
        attack("gpu_row_used_" + repr(used),
               lambda used=used: M.inspect_gpu(fake_gpu(row(used))), "gpu")
    for used in ("0", "1", "32", "64"):
        observed = M.inspect_gpu(fake_gpu(row(used)))
        check("gpu_row_accept_" + used,
              observed["memory_used_mib"] == int(used) and
              observed["memory_total_mib"] == M.GPU_TOTAL_MIB and
              observed["compute_apps"] == [], "gpu_positive")
    gpu_attacks = (
        ("index", row(index="1"), "", 0, 0),
        ("index_leading_zero", row(index="00"), "", 0, 0),
        ("uuid", row(uuid="GPU-wrong"), "", 0, 0),
        ("name", row(name="Other"), "", 0, 0),
        ("total_low", row(total="81919"), "", 0, 0),
        ("total_high", row(total="81921"), "", 0, 0),
        ("total_negative", row(total="-1"), "", 0, 0),
        ("total_leading_zero", row(total="081920"), "", 0, 0),
        ("compute_app", row(), f"99, {M.GPU_UUID}\n", 0, 0),
        ("gpu_query_rc", "", "", 1, 0),
        ("app_query_rc", row(), "", 0, 1),
        ("short", "0,x,y,0\n", "", 0, 0),
        ("extra", row() + "1,x,y,0,1\n", "", 0, 0),
        ("empty", "", "", 0, 0),
    )
    for name, payload, apps, gpu_rc, app_rc in gpu_attacks:
        attack("gpu_" + name, lambda payload=payload, apps=apps, gpu_rc=gpu_rc,
               app_rc=app_rc: M.inspect_gpu(fake_gpu(payload, apps, gpu_rc, app_rc)),
               "gpu")

    # Synthetic controller positive and identity mutations.
    def links(path, cwd=str(M.REMOTE_ROOT), exe=M.CONTROLLER_EXE):
        return exe if Path(path).name == "exe" else cwd

    with tempfile.TemporaryDirectory(prefix="m1461_proc_ok_") as raw:
        proc = Path(raw); controller_fixture(proc)
        with mock.patch.object(M.os, "readlink", side_effect=links):
            observed = M.inspect_controller(proc)
        check("controller_exact_accept", observed["pid"] == M.CONTROLLER_PID and
              observed["state"] == "T" and observed["ppid"] == 1, "controller")
    mutations = (
        ("pid_low", {"pid": M.CONTROLLER_PID - 1}),
        ("pid_high", {"pid": M.CONTROLLER_PID + 1}),
        ("start_low", {"start": M.CONTROLLER_START_TICKS - 1}),
        ("start_high", {"start": M.CONTROLLER_START_TICKS + 1}),
        ("state_S", {"state": "S"}), ("state_R", {"state": "R"}),
        ("state_Z", {"state": "Z"}), ("ppid_0", {"ppid": 0}),
        ("ppid_2", {"ppid": 2}), ("duplicate", {"count": 2}),
        ("empty_argv", {"argv": ()}),
        ("wrong_python", {"argv": ("/bin/python", "-u", M.CONTROLLER_SCRIPT)}),
        ("missing_u", {"argv": (M.CONTROLLER_ARGV[0], M.CONTROLLER_SCRIPT)}),
        ("wrong_script", {"argv": (M.CONTROLLER_ARGV[0], "-u", "wrong.py")}),
        ("extra_arg", {"argv": M.CONTROLLER_ARGV + ("--extra",)}),
    )
    for name, kwargs in mutations:
        with tempfile.TemporaryDirectory(prefix="m1461_proc_bad_") as raw:
            proc = Path(raw); controller_fixture(proc, **kwargs)
            with mock.patch.object(M.os, "readlink", side_effect=links):
                attack("controller_" + name,
                       lambda proc=proc: M.inspect_controller(proc), "controller")
    for name, cwd, exe in (("cwd", "/wrong", M.CONTROLLER_EXE),
                           ("exe", str(M.REMOTE_ROOT), "/wrong/python")):
        with tempfile.TemporaryDirectory(prefix="m1461_proc_link_") as raw:
            proc = Path(raw); controller_fixture(proc)
            with mock.patch.object(M.os, "readlink",
                                   side_effect=lambda p, c=cwd, e=exe: links(p, c, e)):
                attack("controller_wrong_" + name,
                       lambda proc=proc: M.inspect_controller(proc), "controller")

    # External bindings fail before file reads for malformed values.
    malformed = ("", "0", "0" * 63, "0" * 65, "A" * 64, "G" * 64,
                 "z" * 64, "0" * 63 + "\n")
    for variable in M.ENV_BINDINGS:
        for index, value in enumerate(malformed):
            environment = {name: "0" * 64 for name in M.ENV_BINDINGS}
            environment[variable] = value
            attack("external_" + variable + "_" + str(index),
                   lambda env=environment: M.external_bindings(env), "external_sha")

    # O_EXCL attempt, atomic no-replace log and freshness collision gates.
    with tempfile.TemporaryDirectory(prefix="m1461_attempt_") as raw:
        marker = Path(raw) / "attempt"
        controller = {"pid": M.CONTROLLER_PID, "state": "T"}
        values = {"M1458_EXPECTED_RUNNER_SHA256": RUNNER_SHA}
        with mock.patch.object(M, "CANONICAL_ATTEMPT", marker):
            M.consume_attempt(controller, values)
            payload = M.strict_json(marker)
            check("attempt_mode_0400", marker.stat().st_mode & 0o777 == 0o400,
                  "attempt")
            check("attempt_exact_runner_no_retry", payload["runner_sha256"] == RUNNER_SHA and
                  payload["automatic_retry"] is False and
                  payload["controller_restore_permitted"] is False, "attempt")
            attack("attempt_O_EXCL_reuse", lambda: M.consume_attempt(controller, values),
                   "attempt")
    with tempfile.TemporaryDirectory(prefix="m1461_log_") as raw:
        root = Path(raw); canonical = root / "production.log"
        with mock.patch.object(M, "CANONICAL_LOG", canonical):
            M.publish_log(root / "production.log.tmp.first", b"one\n")
            check("atomic_log_first", canonical.read_bytes() == b"one\n", "log")
            attack("atomic_log_no_replace", lambda: M.publish_log(
                   root / "production.log.tmp.second", b"two\n"), "log")
            check("atomic_log_preserved", canonical.read_bytes() == b"one\n", "log")
    for occupied in ("result", "attempt", "log"):
        with tempfile.TemporaryDirectory(prefix="m1461_ns_") as raw:
            root = Path(raw); paths = {name: root / name for name in
                                      ("result", "attempt", "log")}
            paths[occupied].touch()
            with mock.patch.object(M, "CANONICAL_RESULT", paths["result"]), \
                 mock.patch.object(M, "CANONICAL_ATTEMPT", paths["attempt"]), \
                 mock.patch.object(M, "CANONICAL_LOG", paths["log"]):
                attack("namespace_" + occupied, M.namespaces_fresh, "freshness")

    # Exact execution order: GPU is rechecked inside the lease before O_EXCL;
    # O_EXCL precedes capture; double seal precedes atomic PASS publication.
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
    values = {"M1458_EXPECTED_RUNNER_SHA256": RUNNER_SHA}
    with tempfile.TemporaryDirectory(prefix="m1461_order_") as raw:
        root = Path(raw); canonical = root / "production.log"
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
             mock.patch.object(M, "build_runtime",
                               side_effect=lambda: events.append("runtime") or (runtime, binding)), \
             mock.patch.object(M, "validate_bound_capture_files",
                               side_effect=lambda *_: events.append("files")), \
             mock.patch.object(M, "consume_attempt",
                               side_effect=lambda *_: events.append("attempt")), \
             mock.patch.object(M, "delegate_capture",
                               side_effect=lambda *_a: events.append("capture") or root / "result"), \
             mock.patch.object(M.M1434.M1249.R1, "verify_double_seal",
                               side_effect=lambda *_: events.append("seal")), \
             mock.patch.object(M, "publish_log",
                               side_effect=lambda *_: events.append("log")):
            M.execute_once(root / "production.log.tmp.test")
    check("execution_exact_order", events == [
          "lease_enter", "fresh", "controller", "gpu", "runtime", "files",
          "attempt", "capture", "lease_exit", "seal", "log"], "ordering")

    # Post-attempt failure produces only FAIL/quarantine metadata, never retry,
    # canonical promotion or controller restore.  Pre-attempt failures write no log.
    with tempfile.TemporaryDirectory(prefix="m1461_pre_fail_") as raw:
        root = Path(raw); canonical = root / "production.log"
        with mock.patch.object(M, "CANONICAL_LOG", canonical), \
             mock.patch.object(M, "remote_preflight", side_effect=M.M1458Error("pre")):
            attack("pre_attempt_failure", lambda: M.execute_once(
                   root / "production.log.tmp.pre"), "failure")
        check("pre_attempt_no_log", not canonical.exists(), "failure")

    class FailSubstrate:
        @contextlib.contextmanager
        def exclusive_gpu_lease(self, _path):
            yield

    with tempfile.TemporaryDirectory(prefix="m1461_post_fail_") as raw:
        root = Path(raw); canonical = root / "production.log"
        hidden = root / ".capture.staging"; hidden.mkdir()
        with mock.patch.object(M.os, "geteuid", return_value=0), \
             mock.patch.object(M, "CANONICAL_LOG", canonical), \
             mock.patch.object(M, "remote_preflight",
                               return_value=(runtime, binding, controller, values)), \
             mock.patch.object(M.M1434.R1, "load_substrate", return_value=FailSubstrate()), \
             mock.patch.object(M, "namespaces_fresh"), \
             mock.patch.object(M, "inspect_controller", return_value=controller), \
             mock.patch.object(M, "inspect_gpu", return_value={}), \
             mock.patch.object(M, "build_runtime", return_value=(runtime, binding)), \
             mock.patch.object(M, "validate_bound_capture_files"), \
             mock.patch.object(M, "consume_attempt"), \
             mock.patch.object(M, "delegate_capture", side_effect=RuntimeError("capture")):
            attack("post_attempt_capture_failure", lambda: M.execute_once(
                   root / "production.log.tmp.fail"), "failure")
        failed_log = M.strict_json(canonical)
        check("failure_atomic_quarantine", failed_log["status"] == "FAIL" and
              failed_log["failure_quarantine_required"] is True and
              failed_log["canonical_result_promotion_permitted"] is False and
              hidden.is_dir(), "failure")
        check("failure_no_retry_restore", failed_log["automatic_retry"] is False and
              failed_log["controller_restore_permitted"] is False and
              failed_log["controller_restored_by_runner"] is False, "failure")

    # Static exclusions and exact single-use operations.
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    for token in ("os.kill", "SIGCONT", "send_signal", "terminate(", "kill(",
                  "ssh ", "scp ", "rsync ", "dc_shell", "vcs "):
        check("forbid_" + token.strip().replace(" ", "_"), token not in text, "static")
    check("no_shell_true", all(not any(keyword.arg == "shell" and
          isinstance(keyword.value, ast.Constant) and keyword.value.value is True
          for keyword in node.keywords) for node in calls), "static")
    check("single_attempt", text.count("consume_attempt(controller, values)") == 1,
          "static")
    check("single_capture", text.count("M1434.delegate_for_future_release(") == 1,
          "static")
    check("gpu_before_attempt_static", text.index("inspect_gpu()") <
          text.index("consume_attempt(controller, values)"), "static")
    execute_text = text[text.index("def execute_once("):text.index("\ndef main()")]
    check("attempt_before_capture_static", execute_text.index(
          "consume_attempt(controller, values)") < execute_text.index(
          "delegate_capture(runtime, binding, substrate)"), "static")
    check("seal_before_pass_static", text.index(
          "M1434.M1249.R1.verify_double_seal(output)") < text.index(
          'publish_log(temp, log_payload("PASS"'), "static")

    failed_checks = [str(row["check"]) for row in checks if not row["pass"]]
    false_negatives = [str(row["attack"]) for row in attacks if row["false_negative"]]
    categories: dict[str, dict[str, int]] = {}
    for item in checks:
        value = categories.setdefault(str(item["category"]),
                                      {"checks": 0, "passed": 0, "failed": 0})
        value["checks"] += 1
        value["passed" if item["pass"] else "failed"] += 1
    attack_categories: dict[str, dict[str, int]] = {}
    for item in attacks:
        value = attack_categories.setdefault(str(item["category"]),
                                             {"attacks": 0, "rejected": 0,
                                              "false_negatives": 0})
        value["attacks"] += 1
        value["rejected" if item["rejected"] else "false_negatives"] += 1
    passed = not failed_checks and not false_negatives and len(attacks) >= 150
    output = {
        "schema": "m1461_m1458_m1434_live93_runner_blind_hammer_r1_v1",
        "status": "PASS" if passed else "FAIL_DO_NOT_CITE",
        "check_count": len(checks), "passed_count": len(checks) - len(failed_checks),
        "failed_count": len(failed_checks), "attack_count": len(attacks),
        "rejected_attack_count": len(attacks) - len(false_negatives),
        "false_negative_count": len(false_negatives),
        "failed_checks": failed_checks, "false_negatives": false_negatives,
        "categories": categories, "attack_categories": attack_categories,
        "checks": checks, "attacks": attacks,
        "execution": {"ssh": 0, "remote": 0, "real_gpu": 0, "capture": 0,
                      "attempts_consumed": 0, "controller_signals": 0,
                      "controller_restores": 0, "eda": 0},
    }
    (REVIEW / "hammer_output.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in (
        "status", "check_count", "passed_count", "failed_count",
        "attack_count", "rejected_attack_count", "false_negative_count",
        "failed_checks", "false_negatives")}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
