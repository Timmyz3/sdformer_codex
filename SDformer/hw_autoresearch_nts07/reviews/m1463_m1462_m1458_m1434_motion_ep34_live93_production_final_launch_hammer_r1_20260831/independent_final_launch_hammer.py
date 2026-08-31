#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author final launch-authority hammer for M1458.

This hammer is strictly local and synthetic.  It never opens SSH, queries a
real GPU, creates a production result/attempt/log, signals a controller, or
launches capture.  A zero exit status means the final double-sealed authority
was validated by the runner's own ``external_bindings`` and
``validate_future_authorities`` functions after all evidence was sealed.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
HERE = Path(__file__).resolve().parent
RUNNER = HW / "scripts/run_m1458_m1434_motion_ep34_live93_production_one_shot.py"
TEST = HW / "tests/test_run_m1458_m1434_motion_ep34_live93_production_one_shot.py"
CONTRACT = HW / (
    "contracts/m1458_m1434_motion_ep34_live93_production_runner_source_"
    "contract_r1_20260831.json")
BLIND = HW / (
    "reviews/m1461_m1458_m1434_motion_ep34_live93_production_runner_source_"
    "blind_hammer_r1_20260831")
RELEASE = HW / (
    "contracts/m1462_m1458_m1434_motion_ep34_live93_production_launch_"
    "release_r1_20260831.json")
RELEASE_SIDECAR = Path(str(RELEASE) + ".sha256")
RELEASE_OUTER = Path(str(RELEASE) + ".sha256.seal.sha256")
RELEASE_AUTHOR = HW / (
    "reviews/m1462_m1458_m1434_motion_ep34_live93_production_launch_"
    "release_author_r1_20260831")

RUNNER_SHA = "e81c20056dd261619f88884f2f097c9b594887927121d9e599a4f89185d33154"
TEST_SHA = "4a6039a203507fb952ea4cc803261299b69c0bbeab4f031eb937e55a7206ce63"
CONTRACT_SHA = "ae3fa89fe0517578e2ef475c675f1c26160d82fc6356e51b54f79e42960bc0b6"
BLIND_REVIEW_SHA = "43f7a91567325570a30bc27eeda6516839691a5c1efd749185a086d36e2c4d58"
BLIND_MANIFEST_SHA = "6bbb45f9103e069e453ce212b7bdeba4e75e2624b7609df618acfea6d40aae0d"
BLIND_OUTER_SHA = "60cba22e1f6de76ba93d3e1a5730314f413b4b81c3558f452d7a911f511c3343"
RELEASE_SHA = "bd56146574ad5919f326dbe87ccb1dca5da9e06c7e6471412aeaa037a6d0c88f"
RELEASE_SIDECAR_SHA = "8d7bfe7317d7ef3eec862a7c0ab4e42f42c8c1e26d6cc79da14fc99ec02a545c"
RELEASE_OUTER_SHA = "38b29ad65d88a1c8e9a668407f4d8c0bd5d9f8914e4157be97b16e70f618da65"
RELEASE_AUTHOR_REVIEW_SHA = "ac6ea3fa8738648f613676466902f7967ab44d1e8da091c2f50548c6bca655c7"
RELEASE_AUTHOR_MANIFEST_SHA = "243fef45f6d0d3cc1efe902904ce49c8287be85c928e308ff619148d7c0258d8"
RELEASE_AUTHOR_OUTER_SHA = "b66c7acce3c5bb965a3eb9363cc8b8f2c52657bf072118cc2daa63be295a2b6b"
STATUS = "PASS_M1458_M1434_EP34_LIVE93_FINAL_LAUNCH_AUTHORITY"
AUTHORIZATION = {
    "launch": True, "runs": 1, "automatic_retry": False,
    "controller_restore": False,
}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("import spec failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load("m1463_bound_m1458", RUNNER)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rejected(thunk: Callable[[], Any]) -> bool:
    try:
        thunk()
    except BaseException:
        return True
    return False


def completed(stdout: str = "", returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def seal(directory: Path) -> tuple[str, str, str]:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    rows = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.name in {manifest.name, outer.name}:
            continue
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("non-regular review member: " + path.name)
        rows.append(f"{sha(path)}  {path.name}\n")
    manifest.write_text("".join(rows), encoding="utf-8")
    manifest_sha = sha(manifest)
    outer.write_text(f"{manifest_sha}  SHA256SUMS\n", encoding="utf-8")
    return sha(directory / "review.json"), manifest_sha, sha(outer)


def environment(final_hashes: tuple[str, str, str], release_sha: str = RELEASE_SHA
                ) -> dict[str, str]:
    return {
        "M1458_EXPECTED_RUNNER_SHA256": RUNNER_SHA,
        "M1458_EXPECTED_BLIND_REVIEW_SHA256": BLIND_REVIEW_SHA,
        "M1458_EXPECTED_BLIND_MANIFEST_SHA256": BLIND_MANIFEST_SHA,
        "M1458_EXPECTED_BLIND_OUTER_SHA256": BLIND_OUTER_SHA,
        "M1458_EXPECTED_RELEASE_SHA256": release_sha,
        "M1458_EXPECTED_FINAL_REVIEW_SHA256": final_hashes[0],
        "M1458_EXPECTED_FINAL_MANIFEST_SHA256": final_hashes[1],
        "M1458_EXPECTED_FINAL_OUTER_SHA256": final_hashes[2],
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def make_mutated_final(root: Path, review: dict[str, Any]
                       ) -> tuple[str, str, str]:
    write_json(root / "review.json", review)
    return seal(root)


def main() -> int:
    if any(os.path.lexists(str(path)) for path in
           (M.CANONICAL_RESULT, M.CANONICAL_ATTEMPT, M.CANONICAL_LOG)):
        raise RuntimeError("production namespace already occupied")

    checks: list[dict[str, Any]] = []
    attacks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, category: str) -> None:
        checks.append({"check": name, "category": category,
                       "pass": bool(condition)})

    def attack(name: str, thunk: Callable[[], Any], category: str) -> None:
        caught = rejected(thunk)
        attacks.append({"attack": name, "category": category,
                        "rejected": caught, "false_negative": not caught})

    # Exact committed inputs, including both release sidecars and their author.
    identities = (
        ("runner", RUNNER, RUNNER_SHA), ("test", TEST, TEST_SHA),
        ("contract", CONTRACT, CONTRACT_SHA),
        ("blind_review", BLIND / "review.json", BLIND_REVIEW_SHA),
        ("blind_manifest", BLIND / "SHA256SUMS", BLIND_MANIFEST_SHA),
        ("blind_outer", BLIND / "SHA256SUMS.seal.sha256", BLIND_OUTER_SHA),
        ("release", RELEASE, RELEASE_SHA),
        ("release_sidecar", RELEASE_SIDECAR, RELEASE_SIDECAR_SHA),
        ("release_outer", RELEASE_OUTER, RELEASE_OUTER_SHA),
        ("release_author_review", RELEASE_AUTHOR / "review.json",
         RELEASE_AUTHOR_REVIEW_SHA),
        ("release_author_manifest", RELEASE_AUTHOR / "SHA256SUMS",
         RELEASE_AUTHOR_MANIFEST_SHA),
        ("release_author_outer", RELEASE_AUTHOR / "SHA256SUMS.seal.sha256",
         RELEASE_AUTHOR_OUTER_SHA),
        ("m1450_source", M.M1450_SOURCE, M.M1450_SOURCE_SHA256),
        ("m1450_test", M.M1450_TEST, M.M1450_TEST_SHA256),
        ("m1450_contract", M.M1450_CONTRACT, M.M1450_CONTRACT_SHA256),
        ("m1451_review", M.M1451_FAILURE / "review.json",
         M.M1451_FAILURE_REVIEW_SHA256),
        ("m1451_manifest", M.M1451_FAILURE / "SHA256SUMS",
         M.M1451_FAILURE_MANIFEST_SHA256),
        ("m1451_outer", M.M1451_FAILURE / "SHA256SUMS.seal.sha256",
         M.M1451_FAILURE_OUTER_SHA256),
        ("docs359", M.DOCS359, M.DOCS359_SHA256),
    )
    for name, path, expected in identities:
        check(name + "_exact_sha", sha(path) == expected, "identity")

    M.verify_prerequisites()
    policy = M.validate_source_contract()
    check("source_contract_status", policy["status"] == M.SOURCE_STATUS, "identity")
    check("release_sidecar_content", RELEASE_SIDECAR.read_text(encoding="utf-8") ==
          f"{RELEASE_SHA}  {RELEASE.name}\n", "release_seal")
    check("release_outer_content", RELEASE_OUTER.read_text(encoding="utf-8") ==
          f"{RELEASE_SIDECAR_SHA}  {RELEASE_SIDECAR.name}\n", "release_seal")
    M.verify_double_seal(RELEASE_AUTHOR, RELEASE_AUTHOR_REVIEW_SHA,
                         RELEASE_AUTHOR_MANIFEST_SHA, RELEASE_AUTHOR_OUTER_SHA)
    check("release_author_double_seal", True, "release_seal")

    release = M.strict_json(RELEASE)
    release_author = M.strict_json(RELEASE_AUTHOR / "review.json")
    blind = M.strict_json(BLIND / "review.json")
    failed = M.strict_json(M.M1451_FAILURE / "review.json")
    check("blind_status", blind.get("status") ==
          "PASS_M1458_RUNNER_SOURCE__FRESH_RELEASE_MAY_BE_AUTHORED", "blind")
    check("blind_counts", blind.get("verification", {}).get("independent_checks") ==
          "184/184 PASS" and blind.get("verification", {}).get(
              "mutation_campaign") == "188/188 rejected; 0 false negatives",
          "blind")
    check("blind_no_launch", blind.get("authorization", {}).get("launch") is False,
          "blind")
    check("release_author_status", release_author.get("status") ==
          "PASS_M1462_RELEASE_AUTHORING__FRESH_M1463_REQUIRED__NO_LAUNCH",
          "release")
    check("release_author_no_execution", all(value is False for value in
          release_author.get("author_execution", {}).values()), "release")
    check("m1451_exact_root_cause", failed.get("status") ==
          "FAIL_DO_NOT_CITE__M1450_GPU_NEGATIVE_MEMORY_FALSE_NEGATIVE" and
          failed.get("verification", {}).get("false_negative", {}).get(
              "minimal_repair") == "require 0 <= used <= GPU_USED_LIMIT_MIB" and
          failed.get("authorization", {}).get("launch") is False, "failure")
    check("m1450_no_retry", release["immutable_failed_predecessor"][
          "m1450_retry_authorized"] is False, "failure")

    expected_one_shot = {
        "result": str(M.CANONICAL_RESULT.relative_to(ROOT)),
        "attempt": str(M.CANONICAL_ATTEMPT.relative_to(ROOT)),
        "log": str(M.CANONICAL_LOG.relative_to(ROOT)),
        "runs": 1, "automatic_retry": False,
    }
    check("release_core", release.get("status") ==
          "AUTHORIZE_ONE_M1458_M1434_EP34_LIVE93_PRODUCTION_ATTEMPT" and
          release.get("launch_authorized") is True and type(release.get("runs")) is int and
          release.get("runs") == 1 and release.get("automatic_retry") is False and
          release.get("controller_restore") is False, "release")
    check("release_one_shot", release.get("one_shot") == expected_one_shot,
          "release")
    check("release_identity", release.get("runner_sha256") == RUNNER_SHA and
          release.get("m1434_source_sha256") == M.M1434_SOURCE_SHA256 and
          release.get("m1435_review_sha256") == M.M1435_REVIEW_SHA256,
          "release")
    check("release_final_gate", release.get("final_gate", {}).get(
          "required_status") == STATUS and release.get("final_gate", {}).get(
              "required_authorization") == AUTHORIZATION and
          release.get("final_gate", {}).get("actual_launch_ready") is False,
          "release")
    check("capture_counts", release.get("capture_identity", {}).get(
          "live_modules_per_sample") == 247 and release.get(
              "capture_identity", {}).get("live_atlif") == 93 and release.get(
              "capture_identity", {}).get("samples") == 40 and release.get(
              "capture_identity", {}).get("ordered_records") == 9880 and
          release.get("capture_identity", {}).get("attention_records") == 480 and
          release.get("capture_identity", {}).get("payload_files") == 640,
          "capture")
    check("controller_exact", release.get("remote_runtime", {}).get(
          "controller_pid") == M.CONTROLLER_PID and release.get(
              "remote_runtime", {}).get("controller_start_ticks") ==
          M.CONTROLLER_START_TICKS and release.get("remote_runtime", {}).get(
              "controller_state") == "T", "controller")
    check("gpu_guard_exact", release.get("remote_runtime", {}).get(
          "gpu_uuid") == M.GPU_UUID and release.get("remote_runtime", {}).get(
              "gpu_used_lower_bound_mib") == 0 and release.get(
              "remote_runtime", {}).get("gpu_used_limit_mib") == 64 and
          release.get("remote_runtime", {}).get("gpu_memory_fields_exact_int") is True,
          "gpu")

    # Rerun the committed author tests.  They use only mocks/temporary paths.
    test_run = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", "-m", "pytest", "-q",
         str(TEST)], cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False)
    (HERE / "test_output.txt").write_text(test_run.stdout, encoding="utf-8")
    check("author_tests_35_pass", test_run.returncode == 0 and
          "35 passed" in test_run.stdout, "tests")

    # Strong release semantic mutation campaign.  Every changed authority field
    # must fail the exact predicate used by this final hammer.
    def release_ok(value: dict[str, Any]) -> None:
        if not (value.get("status") ==
                "AUTHORIZE_ONE_M1458_M1434_EP34_LIVE93_PRODUCTION_ATTEMPT" and
                value.get("launch_authorized") is True and
                type(value.get("runs")) is int and value.get("runs") == 1 and
                value.get("automatic_retry") is False and
                value.get("controller_restore") is False and
                value.get("runner_sha256") == RUNNER_SHA and
                value.get("m1434_source_sha256") == M.M1434_SOURCE_SHA256 and
                value.get("m1435_review_sha256") == M.M1435_REVIEW_SHA256 and
                value.get("one_shot") == expected_one_shot and
                value.get("final_gate", {}).get("required_status") == STATUS and
                value.get("final_gate", {}).get("required_authorization") ==
                AUTHORIZATION and value.get("capture_identity", {}).get(
                    "ordered_records") == 9880 and value.get(
                    "capture_identity", {}).get("live_atlif") == 93 and
                value.get("remote_runtime", {}).get("controller_pid") ==
                M.CONTROLLER_PID and value.get("remote_runtime", {}).get(
                    "controller_state") == "T" and value.get(
                    "remote_runtime", {}).get("gpu_used_lower_bound_mib") == 0 and
                value.get("remote_runtime", {}).get("gpu_used_limit_mib") == 64 and
                value.get("immutable_failed_predecessor", {}).get(
                    "m1451_fail_review_sha256") == M.M1451_FAILURE_REVIEW_SHA256 and
                value.get("immutable_failed_predecessor", {}).get(
                    "m1450_retry_authorized") is False):
            raise RuntimeError("mutated release accepted")

    mutations = (
        (("status",), "wrong"), (("launch_authorized",), False),
        (("runs",), 0), (("runs",), True), (("runs",), 2),
        (("automatic_retry",), True), (("controller_restore",), True),
        (("runner_sha256",), "0" * 64), (("m1434_source_sha256",), "0" * 64),
        (("m1435_review_sha256",), "0" * 64),
        (("one_shot", "runs"), 0), (("one_shot", "runs"), 2),
        (("one_shot", "automatic_retry"), True),
        (("one_shot", "result"), "wrong"), (("one_shot", "attempt"), "wrong"),
        (("one_shot", "log"), "wrong"),
        (("final_gate", "required_status"), "wrong"),
        (("final_gate", "required_authorization", "launch"), False),
        (("final_gate", "required_authorization", "runs"), 2),
        (("final_gate", "required_authorization", "automatic_retry"), True),
        (("final_gate", "required_authorization", "controller_restore"), True),
        (("capture_identity", "ordered_records"), 9879),
        (("capture_identity", "live_atlif"), 105),
        (("remote_runtime", "controller_pid"), M.CONTROLLER_PID + 1),
        (("remote_runtime", "controller_state"), "S"),
        (("remote_runtime", "gpu_used_lower_bound_mib"), -1),
        (("remote_runtime", "gpu_used_limit_mib"), 65),
        (("immutable_failed_predecessor", "m1451_fail_review_sha256"), "0" * 64),
        (("immutable_failed_predecessor", "m1450_retry_authorized"), True),
    )
    for index, (path, value) in enumerate(mutations):
        mutated = copy.deepcopy(release)
        cursor: dict[str, Any] = mutated
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value
        attack(f"release_semantic_{index}_{'_'.join(path)}",
               lambda value=mutated: release_ok(value), "release_mutation")

    # Final-authority semantics are independently exact-typed here.  The
    # production external SHA is the root of trust; changing that SHA is not a
    # valid mutation of the fixed M1463 authority.
    final_template = {"status": STATUS, "authorization": copy.deepcopy(AUTHORIZATION)}

    def final_ok(value: dict[str, Any]) -> None:
        auth = value.get("authorization")
        if not (value.get("status") == STATUS and type(auth) is dict and
                auth.get("launch") is True and type(auth.get("runs")) is int and
                auth.get("runs") == 1 and auth.get("automatic_retry") is False and
                auth.get("controller_restore") is False and
                set(auth) == set(AUTHORIZATION)):
            raise RuntimeError("mutated final authority accepted")

    final_mutations = (
        (("status",), "wrong"), (("authorization", "launch"), False),
        (("authorization", "runs"), True), (("authorization", "runs"), 0),
        (("authorization", "runs"), 2),
        (("authorization", "automatic_retry"), True),
        (("authorization", "controller_restore"), True),
    )
    for index, (path, value) in enumerate(final_mutations):
        mutated = copy.deepcopy(final_template)
        cursor = mutated
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value
        attack(f"final_semantic_{index}_{'_'.join(path)}",
               lambda value=mutated: final_ok(value), "final_mutation")

    # GPU lower/upper bound, coercion, and full fake nvidia-smi path.
    for value in (-2**63, -65, -2, -1, 65, 66, 2**63, True, False,
                  -1.0, 0.0, 64.0, "0", "64", None, [], {}):
        attack("gpu_used_" + repr(value),
               lambda value=value: M.validate_used_mib(value), "gpu_mutation")
    for value in range(65):
        check("gpu_used_accept_" + str(value), M.validate_used_mib(value) == value,
              "gpu_positive")

    def fake_gpu(used: str, apps: str = ""):
        def run(command, **_kwargs):
            if "--query-gpu=index,uuid,name,memory.used,memory.total" in command:
                return completed(
                    f"0, {M.GPU_UUID}, {M.GPU_NAME}, {used}, {M.GPU_TOTAL_MIB}\n")
            return completed(apps)
        return run

    for used in ("-65", "-2", "-1", "-0", "+0", "00", "65", "65.0",
                 "True", "NaN", "０"):
        attack("gpu_full_row_" + repr(used),
               lambda used=used: M.inspect_gpu(fake_gpu(used)), "gpu_mutation")
    attack("gpu_compute_app", lambda: M.inspect_gpu(fake_gpu(
           "0", f"1, {M.GPU_UUID}\n")), "gpu_mutation")

    # Namespace collisions remain fail-closed without touching production names.
    for occupied in ("result", "attempt", "log"):
        with tempfile.TemporaryDirectory(prefix="m1463_ns_") as raw:
            root = Path(raw)
            paths = {name: root / name for name in ("result", "attempt", "log")}
            paths[occupied].touch()
            with mock.patch.object(M, "CANONICAL_RESULT", paths["result"]), \
                 mock.patch.object(M, "CANONICAL_ATTEMPT", paths["attempt"]), \
                 mock.patch.object(M, "CANONICAL_LOG", paths["log"]):
                attack("namespace_collision_" + occupied, M.namespaces_fresh,
                       "namespace")

    # Attempt O_EXCL and log no-replace semantics use only temporary paths.
    with tempfile.TemporaryDirectory(prefix="m1463_attempt_") as raw:
        marker = Path(raw) / "attempt"
        values = {"M1458_EXPECTED_RUNNER_SHA256": RUNNER_SHA}
        controller = {"pid": M.CONTROLLER_PID, "state": "T"}
        with mock.patch.object(M, "CANONICAL_ATTEMPT", marker):
            M.consume_attempt(controller, values)
            payload = M.strict_json(marker)
            check("attempt_0400", marker.stat().st_mode & 0o777 == 0o400,
                  "attempt")
            check("attempt_no_retry_restore", payload["automatic_retry"] is False and
                  payload["controller_restore_permitted"] is False, "attempt")
            attack("attempt_reuse", lambda: M.consume_attempt(controller, values),
                   "attempt")
    with tempfile.TemporaryDirectory(prefix="m1463_log_") as raw:
        root = Path(raw); canonical = root / "production.log"
        with mock.patch.object(M, "CANONICAL_LOG", canonical):
            M.publish_log(root / "production.log.tmp.first", b"one\n")
            attack("log_no_replace", lambda: M.publish_log(
                   root / "production.log.tmp.second", b"two\n"), "log")
            check("log_preserved", canonical.read_bytes() == b"one\n", "log")
    fail_log = json.loads(M.log_payload("FAIL", {"pid": M.CONTROLLER_PID},
                                               "synthetic").decode("utf-8"))
    pass_log = json.loads(M.log_payload("PASS", {"pid": M.CONTROLLER_PID},
                                               "synthetic").decode("utf-8"))
    check("fail_forbids_restore_promotion", fail_log["automatic_retry"] is False and
          fail_log["controller_restore_permitted"] is False and
          fail_log["canonical_result_promotion_permitted"] is False, "log")
    check("pass_no_runner_restore", pass_log["controller_restored_by_runner"] is False and
          pass_log["automatic_retry"] is False, "log")

    source_text = RUNNER.read_text(encoding="utf-8")
    for token in ("os.kill", "SIGCONT", "send_signal", "terminate(", "ssh ",
                  "scp ", "rsync ", "dc_shell", "vcs "):
        check("forbid_" + token.strip().replace(" ", "_"), token not in source_text,
              "static")
    check("single_attempt_static", source_text.count(
          "consume_attempt(controller, values)") == 1, "static")
    check("single_capture_static", source_text.count(
          "M1434.delegate_for_future_release(") == 1, "static")
    check("no_restore_api_static", "controller_restore" in source_text and
          "controller_restored_by_runner\": False" in source_text, "static")

    failed_checks = [row["check"] for row in checks if not row["pass"]]
    false_negatives = [row["attack"] for row in attacks if row["false_negative"]]
    if failed_checks or false_negatives:
        raise RuntimeError("pre-final hammer failure: " +
                           repr((failed_checks, false_negatives)))

    categories: dict[str, int] = {}
    for row in checks:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
    mutation_categories: dict[str, int] = {}
    for row in attacks:
        mutation_categories[row["category"]] = mutation_categories.get(
            row["category"], 0) + 1
    hammer = {
        "schema": "m1463_m1458_ep34_final_launch_hammer_r1_v1",
        "status": "PASS",
        "check_count": len(checks), "passed_count": len(checks),
        "failed_count": 0, "attack_count": len(attacks),
        "rejected_attack_count": len(attacks), "false_negative_count": 0,
        "categories": categories, "mutation_categories": mutation_categories,
        "checks": checks, "attacks": attacks,
        "native_runner_validation":
            "external_bindings+validate_future_authorities final pass required for exit0",
        "execution": {"ssh": 0, "remote": 0, "real_gpu_queries": 0,
                      "capture": 0, "production_attempts_consumed": 0,
                      "controller_signals": 0, "controller_restores": 0,
                      "eda": 0},
    }
    write_json(HERE / "hammer_output.json", hammer)
    (HERE / "mechanical_checks.txt").write_text(
        f"checks={len(checks)} passed={len(checks)} failed=0\n"
        f"attacks={len(attacks)} rejected={len(attacks)} false_negatives=0\n"
        "author_tests=35/35 PASS\n"
        "execution=LOCAL_SYNTHETIC_ONLY\n", encoding="utf-8")
    (HERE / "READ_ONLY_NO_SSH_NO_GPU_NO_CAPTURE_NO_ATTEMPT_NO_CONTROLLER_SIGNAL.txt").write_text(
        "M1463 final hammer used only local files, mocks and temporary namespaces.\n",
        encoding="utf-8")
    (HERE / "RUN_COMPLETE.txt").write_text(
        "PASS is usable only when this program exits zero after native runner "
        "external_bindings and validate_future_authorities validate the final seal.\n",
        encoding="utf-8")

    review = {
        "schema": "m1463_m1462_m1458_m1434_ep34_final_launch_hammer_r1_v1",
        "status": STATUS,
        "score": 100, "p0_count": 0, "p1_count": 0, "date": "2026-08-31",
        "verdict": "PASS. Exact M1458/M1434 live93 source, M1461 blind, M1462 release and sidecars, immutable M1450/M1451 failure, stopped-controller/A800 guards, fresh one-shot namespaces, no-retry/no-restore policy and 40x247 capture counts passed independent mutation hammering. Runner-native external_bindings plus validate_future_authorities validates the final double seal before exit zero.",
        "bindings": {
            "runner_sha256": RUNNER_SHA, "test_sha256": TEST_SHA,
            "contract_sha256": CONTRACT_SHA,
            "m1461_review_sha256": BLIND_REVIEW_SHA,
            "m1461_manifest_sha256": BLIND_MANIFEST_SHA,
            "m1461_outer_file_sha256": BLIND_OUTER_SHA,
            "m1462_release_sha256": RELEASE_SHA,
            "m1462_release_sidecar_sha256": RELEASE_SIDECAR_SHA,
            "m1462_release_outer_file_sha256": RELEASE_OUTER_SHA,
            "m1450_source_sha256": M.M1450_SOURCE_SHA256,
            "m1451_fail_review_sha256": M.M1451_FAILURE_REVIEW_SHA256,
            "m1451_fail_manifest_sha256": M.M1451_FAILURE_MANIFEST_SHA256,
            "m1451_fail_outer_file_sha256": M.M1451_FAILURE_OUTER_SHA256,
        },
        "verification": {
            "author_tests": "35/35 PASS",
            "independent_checks": f"{len(checks)}/{len(checks)} PASS",
            "mutation_campaign": f"{len(attacks)}/{len(attacks)} rejected; 0 false negatives",
            "native_external_bindings": True,
            "native_validate_future_authorities": True,
            "production_namespaces_absent": True,
            "capture_shape": "40 samples x 247 live modules = 9880 ordered records",
            "live_atlif": 93, "dead_sn2_q": 12,
            "gpu_memory_guard": "exact int; 0 <= used <= 64 MiB",
            "controller": "PID3804343/start703730691/PPID1/state T exact",
            "m1450_retry": False, "m1451_root_cause_exact": True,
            "docs359_sha256": M.DOCS359_SHA256,
        },
        "authorization": AUTHORIZATION,
        "execution": {"ssh": 0, "remote_runs": 0, "real_gpu_queries": 0,
                      "capture_runs": 0, "production_attempts_consumed": 0,
                      "controller_signals": 0, "controller_restores": 0,
                      "eda_runs": 0},
        "claim_boundary": {"launch_authority_only": True,
                           "production_result": False, "hardware_result": False,
                           "cycles": False, "speedup": False, "energy": False,
                           "ppa": False, "headline": False},
    }
    write_json(HERE / "review.json", review)
    (HERE / "review.md").write_text(
        "# M1463 final launch-authority hammer\n\n"
        f"PASS: {len(checks)}/{len(checks)} checks and {len(attacks)}/{len(attacks)} "
        "mutations, zero false negatives. This author performed no SSH, GPU, "
        "capture, production-attempt, controller-signal, restore, or EDA action. "
        "Authority is exactly one M1458 run with no automatic retry and no "
        "controller restore.\n", encoding="utf-8")

    # Seal once, then invoke the runner's exact native external binding and
    # future-authority validation on those fixed roots.  Nothing is modified
    # after this call.
    hashes = seal(HERE)
    values = M.external_bindings(environment(hashes))
    M.validate_future_authorities(values)
    print(json.dumps({"status": STATUS, "checks": len(checks),
                      "attacks": len(attacks),
                      "false_negatives": 0, "final_review_sha256": hashes[0],
                      "final_manifest_sha256": hashes[1],
                      "final_outer_file_sha256": hashes[2]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
