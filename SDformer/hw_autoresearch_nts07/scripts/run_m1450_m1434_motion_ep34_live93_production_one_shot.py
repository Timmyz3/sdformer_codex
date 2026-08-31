#!/opt/conda/envs/sdformerflow/bin/python
"""Inert one-shot runner source for the sealed M1434 ep34 live-93 capture.

The source-author stage can only run ``--source-absent-self-check``.  Remote
preflight and production additionally require exact, external SHA bindings for
three future authorities.  This file never signals or restores the stopped
MVSEC controller.  A successful capture records permission for a later,
separately authorized restore; every failure records that restore is forbidden.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "hw_autoresearch_nts07"
REMOTE_ROOT = Path("/root/private_data/work/sdformer_codex/SDformer")
SOURCE = Path(__file__).resolve()
TEST = HW / "tests/test_run_m1450_m1434_motion_ep34_live93_production_one_shot.py"
SOURCE_CONTRACT = HW / (
    "contracts/m1450_m1434_motion_ep34_live93_production_runner_source_"
    "contract_r1_20260831.json")
M1434_SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1434_motion_ep34_live93_runtime_successor_r1.py")
M1434_SOURCE_SHA256 = "b28c8507f077b754048fc54afd9fe04900dac854b273df2ba1981fa5f892b6ed"
M1434_TEST = HW / "tests/test_m1434_motion_ep34_live93_runtime_successor.py"
M1434_TEST_SHA256 = "b05e122be8d3fb61b001be648ff8980a7341c2a19d29c401a9dc62ff5bafb8c2"
M1434_CONTRACT = HW / (
    "contracts/m1434_motion_ep34_live93_runtime_successor_source_"
    "contract_r1_20260831.json")
M1434_CONTRACT_SHA256 = "5e92af7c080f417fd94f190ce90c064a19fd70c02cfbd8fb6a2ad03d6f12e75e"
M1434_AUTHOR = HW / (
    "reviews/m1434_motion_ep34_live93_runtime_successor_source_author_"
    "r1_20260831")
M1434_AUTHOR_REVIEW_SHA256 = "7342c61780a02634caebb5e5e1e2576f86a2406c2811a52bbccf47205051fe38"
M1434_AUTHOR_MANIFEST_SHA256 = "396c13afb70dffbc98090ee90a6994fb859d5cad18e1c826695945a9acc1001c"
M1434_AUTHOR_OUTER_SHA256 = "bf5852329c3d1febd48d7c4478046be33a78a6b3398e3497fd23da5d4f73b2bc"
M1435_BLIND = HW / (
    "reviews/m1435_m1434_motion_ep34_live93_runtime_successor_source_"
    "blind_hammer_r1_20260831")
M1435_REVIEW_SHA256 = "6c32aa28e75dc98cec99a9bf777f3c3433330bbf77a4b3a4d645b11493697b59"
M1435_MANIFEST_SHA256 = "5c23d2870d8aa52689c8708be51d920ed1e3ba130010690e4b7c095cf8490f34"
M1435_OUTER_SHA256 = "513dea3956d5a67c7fcbd2a52997bd3a8a965bd86cc1fc0ff5b8435eeefc3b14"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

FUTURE_BLIND = HW / (
    "reviews/m1451_m1450_m1434_motion_ep34_live93_production_runner_source_"
    "blind_hammer_r1_20260831")
FUTURE_RELEASE = HW / (
    "contracts/m1452_m1450_m1434_motion_ep34_live93_production_launch_"
    "release_r1_20260831.json")
FUTURE_FINAL = HW / (
    "reviews/m1453_m1452_m1450_m1434_motion_ep34_live93_production_final_"
    "launch_hammer_r1_20260831")

CANONICAL_RESULT = HW / "results/m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831"
CANONICAL_ATTEMPT = HW / "results/.m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831.attempt_consumed"
CANONICAL_LOG = HW / "results/.m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831.production.log"
CONTROLLER_SCRIPT = (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "run_mvsec_strict_c00_continuation_20260830.py")
CONTROLLER_ARGV = (
    "/opt/conda/envs/sdformerflow/bin/python", "-u", CONTROLLER_SCRIPT)
CONTROLLER_EXE = "/opt/conda/envs/sdformerflow/bin/python3.10"
CONTROLLER_PID = 3804343
CONTROLLER_START_TICKS = 703730691
GPU_UUID = "GPU-499236d3-b46c-5d25-4a22-530d47ed5112"
GPU_NAME = "NVIDIA A800 80GB PCIe"
GPU_TOTAL_MIB = 81920
GPU_USED_LIMIT_MIB = 64
NVIDIA_SMI = "/usr/bin/nvidia-smi"
SOURCE_SCHEMA = "m1450_m1434_motion_ep34_live93_production_runner_source_r1_v1"
SOURCE_STATUS = "SOURCE_ONLY__M1434_M1435_BOUND__FRESH_BLIND_REQUIRED__NO_LAUNCH"
PASS_TOKEN = "PASS_M1450_SOURCE_ABSENT_SELF_CHECK__NO_REMOTE_NO_GPU_NO_ATTEMPT"


class M1450Error(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise M1450Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, expected: str, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise M1450Error("missing " + label) from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be a regular non-symlink")
    require(sha256(path) == expected, label + " SHA mismatch")


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    def reject(token: str):
        raise M1450Error("nonfinite JSON token: " + token)
    value = json.loads(path.read_text(encoding="utf-8"),
                       object_pairs_hook=pairs, parse_constant=reject)
    require(type(value) is dict, "JSON root must be object")
    return value


def load_m1434():
    regular_exact(M1434_SOURCE, M1434_SOURCE_SHA256, "M1434 source")
    spec = importlib.util.spec_from_file_location("m1450_sealed_m1434", M1434_SOURCE)
    require(spec is not None and spec.loader is not None, "cannot import M1434")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    regular_exact(M1434_SOURCE, M1434_SOURCE_SHA256, "M1434 source after import")
    return module


M1434 = load_m1434()


def verify_double_seal(root: Path, review_sha: str, manifest_sha: str,
                       outer_sha: str) -> dict[str, Any]:
    review = root / "review.json"
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    regular_exact(review, review_sha, str(root.name) + " review")
    regular_exact(manifest, manifest_sha, str(root.name) + " manifest")
    regular_exact(outer, outer_sha, str(root.name) + " outer")
    require(outer.read_text(encoding="utf-8") == manifest_sha + "  SHA256SUMS\n",
            str(root.name) + " outer content mismatch")
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        require(name not in rows, "duplicate seal member")
        rows[name] = digest
    require((rows.get(str(review.relative_to(ROOT))) == review_sha or
             rows.get("review.json") == review_sha),
            str(root.name) + " review not sealed")
    return strict_json(review)


def verify_prerequisites() -> None:
    regular_exact(M1434_TEST, M1434_TEST_SHA256, "M1434 test")
    regular_exact(M1434_CONTRACT, M1434_CONTRACT_SHA256, "M1434 contract")
    regular_exact(DOCS359, DOCS359_SHA256, "docs359")
    author = verify_double_seal(M1434_AUTHOR, M1434_AUTHOR_REVIEW_SHA256,
                                M1434_AUTHOR_MANIFEST_SHA256,
                                M1434_AUTHOR_OUTER_SHA256)
    require(author.get("status") ==
            "PASS_SOURCE_AUTHOR__DIFFERENT_AUTHOR_BLIND_REQUIRED" and
            author.get("claim_boundary", {}).get("source_and_tests_only") is True and
            author.get("authorization", {}).get("production_launch") is False,
            "M1434 author boundary mismatch")
    blind = verify_double_seal(M1435_BLIND, M1435_REVIEW_SHA256,
                               M1435_MANIFEST_SHA256, M1435_OUTER_SHA256)
    require(blind.get("status") ==
            "PASS_SOURCE__FRESH_RELEASE_AUTHOR_MAY_BE_AUTHORED" and
            blind.get("verification", {}).get("independent_checks") == "75/75 PASS" and
            blind.get("authorization", {}).get("production_launch") is False,
            "M1435 blind boundary mismatch")
    M1434.verify_predecessors()
    M1434.validate_source_policy()
    replay = M1434.replay_sample0_forensic_summary()
    require(replay == {
        "status": "PASS", "errors": [], "samples": 1,
        "live_modules_per_sample": 247, "records": 247,
        "expected_records": 247, "dead_modules": 12,
    }, "sealed sample0 live93 replay mismatch")
    static_policy = M1434.R1.strict_json(M1434.R1.SOURCE_CONTRACT)
    static_inventory = M1434.R1.frozen_non_atlif_inventory(static_policy)
    static_inventory["atlif"] = list(M1434.M1349.EXPECTED_ATLIF_NAMES)
    live_inventory = M1434.expected_live93_inventory(static_inventory)
    require(len(live_inventory["atlif"]) == 93 and
            sum(map(len, live_inventory.values())) == 247 and
            M1434.terminal_lf_digest(live_inventory["atlif"]) ==
            M1434.LIVE_ATLIF_SHA256 and
            M1434.CANONICAL_RESULT == CANONICAL_RESULT and
            M1434.CANONICAL_ATTEMPT == CANONICAL_ATTEMPT and
            M1434.CANONICAL_LOG == CANONICAL_LOG,
            "sealed live93 authority/namespace mismatch")


def source_commands() -> dict[str, str]:
    rel = str(SOURCE.relative_to(ROOT))
    return {
        "source_absent_self_check":
            f"/opt/conda/envs/sdformerflow/bin/python {rel} --source-absent-self-check",
        "remote_preflight":
            f"M1450_EXPECTED_RUNNER_SHA256=<sha> M1450_EXPECTED_BLIND_REVIEW_SHA256=<sha> "
            f"M1450_EXPECTED_BLIND_MANIFEST_SHA256=<sha> M1450_EXPECTED_BLIND_OUTER_SHA256=<sha> "
            f"M1450_EXPECTED_RELEASE_SHA256=<sha> M1450_EXPECTED_FINAL_REVIEW_SHA256=<sha> "
            f"M1450_EXPECTED_FINAL_MANIFEST_SHA256=<sha> M1450_EXPECTED_FINAL_OUTER_SHA256=<sha> "
            f"/opt/conda/envs/sdformerflow/bin/python {rel} --remote-preflight",
        "run":
            f"M1450_EXPECTED_RUNNER_SHA256=<sha> M1450_EXPECTED_BLIND_REVIEW_SHA256=<sha> "
            f"M1450_EXPECTED_BLIND_MANIFEST_SHA256=<sha> M1450_EXPECTED_BLIND_OUTER_SHA256=<sha> "
            f"M1450_EXPECTED_RELEASE_SHA256=<sha> M1450_EXPECTED_FINAL_REVIEW_SHA256=<sha> "
            f"M1450_EXPECTED_FINAL_MANIFEST_SHA256=<sha> M1450_EXPECTED_FINAL_OUTER_SHA256=<sha> "
            f"/opt/conda/envs/sdformerflow/bin/python {rel} --run --temporary-log "
            f"{CANONICAL_LOG.relative_to(ROOT)}.tmp.<nonce>",
    }


def validate_source_contract() -> dict[str, Any]:
    policy = strict_json(SOURCE_CONTRACT)
    require(policy.get("schema") == SOURCE_SCHEMA and
            policy.get("status") == SOURCE_STATUS, "source policy mismatch")
    require(policy.get("source") == {"path": str(SOURCE.relative_to(ROOT)),
                                      "sha256": sha256(SOURCE)},
            "runner identity mismatch")
    require(policy.get("test") == {"path": str(TEST.relative_to(ROOT)),
                                    "sha256": sha256(TEST)},
            "test identity mismatch")
    require(policy.get("commands") == source_commands(),
            "command-reference self-proof mismatch")
    require(policy.get("launch_authorized") is False and
            policy.get("author_execution") == {
                "remote": False, "gpu": False, "forward": False,
                "capture": False, "attempt_consumed": False,
            }, "source author boundary mismatch")
    return policy


def future_paths() -> tuple[Path, ...]:
    return (FUTURE_BLIND, FUTURE_RELEASE, FUTURE_FINAL)


def source_absent_self_check() -> None:
    verify_prerequisites()
    validate_source_contract()
    require(all(not os.path.lexists(str(path)) for path in future_paths()),
            "future blind/release/final authority must be absent")
    require(all(not os.path.lexists(str(path)) for path in
                (CANONICAL_RESULT, CANONICAL_ATTEMPT, CANONICAL_LOG)),
            "production namespaces must be absent")


ENV_BINDINGS = {
    "M1450_EXPECTED_RUNNER_SHA256": SOURCE,
    "M1450_EXPECTED_BLIND_REVIEW_SHA256": FUTURE_BLIND / "review.json",
    "M1450_EXPECTED_BLIND_MANIFEST_SHA256": FUTURE_BLIND / "SHA256SUMS",
    "M1450_EXPECTED_BLIND_OUTER_SHA256": FUTURE_BLIND / "SHA256SUMS.seal.sha256",
    "M1450_EXPECTED_RELEASE_SHA256": FUTURE_RELEASE,
    "M1450_EXPECTED_FINAL_REVIEW_SHA256": FUTURE_FINAL / "review.json",
    "M1450_EXPECTED_FINAL_MANIFEST_SHA256": FUTURE_FINAL / "SHA256SUMS",
    "M1450_EXPECTED_FINAL_OUTER_SHA256": FUTURE_FINAL / "SHA256SUMS.seal.sha256",
}


def external_bindings(environment: dict[str, str] | None = None) -> dict[str, str]:
    environment = os.environ if environment is None else environment
    values: dict[str, str] = {}
    for name, path in ENV_BINDINGS.items():
        value = environment.get(name, "")
        require(len(value) == 64 and all(ch in "0123456789abcdef" for ch in value),
                "missing/malformed external SHA: " + name)
        regular_exact(path, value, name)
        values[name] = value
    require(values["M1450_EXPECTED_RUNNER_SHA256"] == sha256(SOURCE),
            "external runner SHA mismatch")
    return values


def validate_future_authorities(values: dict[str, str]) -> None:
    blind = verify_double_seal(
        FUTURE_BLIND, values["M1450_EXPECTED_BLIND_REVIEW_SHA256"],
        values["M1450_EXPECTED_BLIND_MANIFEST_SHA256"],
        values["M1450_EXPECTED_BLIND_OUTER_SHA256"])
    require(blind.get("status") ==
            "PASS_M1450_RUNNER_SOURCE__FRESH_RELEASE_MAY_BE_AUTHORED" and
            blind.get("authorization", {}).get("launch") is False and
            blind.get("bindings") == {
                "runner_sha256": values["M1450_EXPECTED_RUNNER_SHA256"],
                "test_sha256": sha256(TEST),
                "contract_sha256": sha256(SOURCE_CONTRACT),
                "m1434_source_sha256": M1434_SOURCE_SHA256,
                "m1435_review_sha256": M1435_REVIEW_SHA256,
            },
            "future source blind mismatch")
    release = strict_json(FUTURE_RELEASE)
    require(release.get("status") ==
            "AUTHORIZE_ONE_M1450_M1434_EP34_LIVE93_PRODUCTION_ATTEMPT" and
            release.get("launch_authorized") is True and
            release.get("runs") == 1 and release.get("automatic_retry") is False and
            release.get("runner_sha256") == values["M1450_EXPECTED_RUNNER_SHA256"] and
            release.get("m1434_source_sha256") == M1434_SOURCE_SHA256 and
            release.get("m1435_review_sha256") == M1435_REVIEW_SHA256 and
            release.get("one_shot") == {
                "result": str(CANONICAL_RESULT.relative_to(ROOT)),
                "attempt": str(CANONICAL_ATTEMPT.relative_to(ROOT)),
                "log": str(CANONICAL_LOG.relative_to(ROOT)),
                "runs": 1, "automatic_retry": False,
            },
            "future release mismatch")
    final = verify_double_seal(
        FUTURE_FINAL, values["M1450_EXPECTED_FINAL_REVIEW_SHA256"],
        values["M1450_EXPECTED_FINAL_MANIFEST_SHA256"],
        values["M1450_EXPECTED_FINAL_OUTER_SHA256"])
    require(final.get("status") ==
            "PASS_M1450_M1434_EP34_LIVE93_FINAL_LAUNCH_AUTHORITY" and
            final.get("authorization") == {
                "launch": True, "runs": 1, "automatic_retry": False,
                "controller_restore": False},
            "future final hammer mismatch")


def namespaces_fresh() -> None:
    require(all(not os.path.lexists(str(path)) for path in
                (CANONICAL_RESULT, CANONICAL_ATTEMPT, CANONICAL_LOG)),
            "M1434 result/attempt/log namespace not fresh")


def _stat_fields(text: str) -> tuple[str, int, int]:
    close = text.rfind(")")
    require(close >= 0, "malformed proc stat")
    fields = text[close + 2:].split()
    require(len(fields) >= 20, "short proc stat")
    try:
        return fields[0], int(fields[1]), int(fields[19])
    except ValueError as exc:
        raise M1450Error("malformed proc stat numeric field") from exc


def inspect_controller(proc_root: Path = Path("/proc")) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = tuple(part.decode("utf-8") for part in
                         (entry / "cmdline").read_bytes().split(b"\0") if part)
        except (FileNotFoundError, ProcessLookupError, PermissionError, UnicodeDecodeError):
            continue
        if argv != CONTROLLER_ARGV:
            continue
        state, ppid, start = _stat_fields((entry / "stat").read_text(encoding="utf-8"))
        cwd = os.readlink(entry / "cwd")
        exe = os.readlink(entry / "exe")
        matches.append({"pid": int(entry.name), "ppid": ppid, "state": state,
                        "start_ticks": start, "cwd": cwd, "exe": exe,
                        "argv": list(argv)})
    require(len(matches) == 1, "exactly one MVSEC controller required")
    value = matches[0]
    require(value["pid"] == CONTROLLER_PID and
            value["start_ticks"] == CONTROLLER_START_TICKS and
            value["ppid"] == 1 and value["state"] == "T" and
            value["cwd"] == str(REMOTE_ROOT) and value["exe"] == CONTROLLER_EXE,
            "MVSEC controller is not the exact stopped remote controller identity")
    return value


Run = Callable[..., subprocess.CompletedProcess[str]]


def inspect_gpu(run: Run = subprocess.run) -> dict[str, Any]:
    common = {"text": True, "stdout": subprocess.PIPE,
              "stderr": subprocess.PIPE, "check": False}
    gpu = run([NVIDIA_SMI,
               "--query-gpu=index,uuid,name,memory.used,memory.total",
               "--format=csv,noheader,nounits"], **common)
    require(gpu.returncode == 0, "nvidia-smi GPU query failed")
    rows = list(csv.reader(gpu.stdout.splitlines()))
    require(len(rows) == 1 and len(rows[0]) == 5, "exactly one GPU row required")
    row = [item.strip() for item in rows[0]]
    try:
        used, total = int(row[3]), int(row[4])
    except ValueError as exc:
        raise M1450Error("malformed GPU memory") from exc
    require(row[:3] == ["0", GPU_UUID, GPU_NAME] and total == GPU_TOTAL_MIB and
            used <= GPU_USED_LIMIT_MIB, "A800 identity/idleness mismatch")
    apps = run([NVIDIA_SMI, "--query-compute-apps=pid,gpu_uuid",
                "--format=csv,noheader,nounits"], **common)
    require(apps.returncode == 0 and not apps.stdout.strip(),
            "GPU has compute applications or app query failed")
    return {"index": 0, "uuid": GPU_UUID, "name": GPU_NAME,
            "memory_used_mib": used, "memory_total_mib": total,
            "compute_apps": []}


def validate_bound_capture_files(binding: dict[str, Any]) -> None:
    regular_exact(Path(binding["checkpoint_path"]), M1434.CHECKPOINT_SHA256,
                  "ep34 checkpoint")
    regular_exact(Path(binding["config_path"]), M1434.CONFIG_SHA256, "ep34 config")
    regular_exact(ROOT / (
        "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
        "profile_nts11_hardware_p0.py"), M1434.PROFILE_SOURCE_SHA256,
        "profile source")
    regular_exact(ROOT / (
        "neuron_experiments/H9_bipolar_self_attention/overlay/models/"
        "STSwinNet_SNN/atlif_ternary_psn/atlif_ternary_psn.py"),
        M1434.ATLIF_OVERLAY_SOURCE_SHA256, "ATLIF overlay source")


def remote_preflight(environment: dict[str, str] | None = None,
                     proc_root: Path = Path("/proc"), run: Run = subprocess.run
                     ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    require(os.geteuid() == 0, "uid0 required")
    require(ROOT == REMOTE_ROOT and Path.cwd().resolve() == REMOTE_ROOT,
            "remote preflight requires exact repository cwd")
    verify_prerequisites()
    validate_source_contract()
    values = external_bindings(environment)
    validate_future_authorities(values)
    namespaces_fresh()
    controller = inspect_controller(proc_root)
    gpu = inspect_gpu(run)
    runtime, binding = M1434.build_runtime()
    validate_bound_capture_files(binding)
    namespaces_fresh()
    require(inspect_controller(proc_root) == controller, "controller changed during preflight")
    inspect_gpu(run)
    return runtime, binding, controller, values


def consume_attempt(controller: dict[str, Any], values: dict[str, str]) -> None:
    payload = {
        "schema": "m1450_m1434_ep34_live93_attempt_r1_v1",
        "status": "ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE",
        "runner_sha256": values["M1450_EXPECTED_RUNNER_SHA256"],
        "m1434_source_sha256": M1434_SOURCE_SHA256,
        "controller": controller,
        "gpu_uuid": GPU_UUID,
        "automatic_retry": False,
        "controller_restore_permitted": False,
    }
    descriptor = os.open(CANONICAL_ATTEMPT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        os.write(descriptor, (json.dumps(payload, sort_keys=True,
                                         separators=(",", ":")) + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def log_payload(status: str, controller: dict[str, Any], detail: str) -> bytes:
    success = status == "PASS"
    return (json.dumps({
        "schema": "m1450_m1434_ep34_live93_production_log_r1_v1",
        "status": status,
        "detail": detail,
        "automatic_retry": False,
        "controller": controller,
        "controller_restored_by_runner": False,
        "controller_restore_permitted_after_success": success,
        "controller_restore_permitted": success,
        "failure_quarantine_required": not success,
        "canonical_result_promotion_permitted": success,
    }, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _temp_ok(path: Path) -> None:
    require(path.is_absolute() and path.parent == CANONICAL_LOG.parent and
            path.name.startswith(CANONICAL_LOG.name + ".tmp.") and
            path != CANONICAL_LOG, "temporary log namespace mismatch")


def publish_log(temp: Path, payload: bytes) -> None:
    _temp_ok(temp)
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    require(not os.path.lexists(str(CANONICAL_LOG)), "canonical log occupied")
    os.link(temp, CANONICAL_LOG, follow_symlinks=False)
    temp.unlink()


def execute_once(temp: Path, environment: dict[str, str] | None = None) -> Path:
    require(os.geteuid() == 0, "uid0 required")
    _temp_ok(temp)
    runtime, binding, controller, values = remote_preflight(environment)
    substrate = M1434.R1.load_substrate()
    attempted = False
    try:
        with substrate.exclusive_gpu_lease(M1434.R1.CANONICAL_LEASE):
            namespaces_fresh()
            require(inspect_controller() == controller, "controller changed under lease")
            inspect_gpu()
            rebound_runtime, rebound_binding = M1434.build_runtime()
            require((rebound_runtime, rebound_binding) == (runtime, binding),
                    "capture binding changed under lease")
            validate_bound_capture_files(binding)
            consume_attempt(controller, values)
            attempted = True
            output = M1434.delegate_for_future_release(runtime, binding, substrate)
        M1434.M1249.R1.verify_double_seal(output)
        publish_log(temp, log_payload("PASS", controller,
                                      "result double seal verified; later restore only"))
        return Path(output)
    except BaseException as exc:
        # No signal/restore operation exists in this runner.  A consumed attempt
        # remains consumed and every failure log forbids controller restoration.
        if attempted and not os.path.lexists(str(CANONICAL_LOG)):
            publish_log(temp, log_payload("FAIL", controller,
                                          type(exc).__name__ + ": " + str(exc)))
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-absent-self-check", action="store_true")
    group.add_argument("--remote-preflight", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--temporary-log", type=Path)
    args = parser.parse_args()
    if args.source_absent_self_check:
        require(args.temporary_log is None, "source check cannot name a log")
        source_absent_self_check()
        print(PASS_TOKEN)
        return 0
    if args.remote_preflight:
        require(args.temporary_log is None, "preflight cannot name a log")
        remote_preflight()
        print("PASS_M1450_REMOTE_READ_ONLY_PREFLIGHT__NO_ATTEMPT")
        return 0
    require(args.temporary_log is not None, "run requires --temporary-log")
    execute_once(args.temporary_log.resolve())
    print("PASS_M1450_M1434_EP34_LIVE93_ONE_SHOT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
