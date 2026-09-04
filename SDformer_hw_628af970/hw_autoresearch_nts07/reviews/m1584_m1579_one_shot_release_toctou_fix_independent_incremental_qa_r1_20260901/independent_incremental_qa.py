#!/usr/bin/env python3
"""Independent incremental QA of commit 2b53c147 against M1581.

All execute() calls replace materialization/replay with tiny fixtures.  The
51.84M production ledger, GPU and EDA paths are never executed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/run_m1579_ep34_c1_same_ledger_cycle_model.py"
TEST = HW / "system_simulator/tests/test_m1579_ep34_c1_same_ledger_cycle_model.py"
M1581 = HW / "reviews/m1581_m1579_ep34_c1_same_ledger_cycle_model_independent_engineering_qa_r1_20260901"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
COMMIT = "2b53c14719a10eae8547fce23d809845733ada1d"

PINNED = {
    "source": "5fa210dcc551ceef649fadd0cc476d4baa701cc7c3a9f2551edfd5dfb6d86264",
    "test": "160dc31a7e8ece30a02842ae94139b6329983df2a4b2755f944075e725104782",
    "m1581_review": "924bcaec2b7b88c28c9fd32b5328f8ff3ddb4adf84ddf5440b527723b070412d",
    "m1581_manifest": "ce91a8d1e3628556d53e361f88fe68f7fe2bd67562313b5039fddefddbd2e273",
    "m1581_outer": "d940b5a66a627773ed260ffe42931111f58af0eaca3a8c4d64011f9b62827e4c",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, expected: str, label: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def load_source() -> Any:
    spec = importlib.util.spec_from_file_location("m1584_bound_m1579", SOURCE)
    require(spec is not None and spec.loader is not None, "cannot import M1579")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def expect_reject(function, label: str) -> str:
    try:
        function()
    except BaseException as error:
        require(not isinstance(error, (KeyboardInterrupt, SystemExit)),
                label + " escaped test process")
        return type(error).__name__
    raise RuntimeError(label + " did not fail closed")


def release_value(module: Any, output: Path, ledger: Path,
                  marker: Path) -> dict[str, Any]:
    return {
        "schema": module.RELEASE_SCHEMA,
        "status": module.RELEASE_STATUS,
        "source_sha256": sha256(SOURCE),
        "output": str(output),
        "ledger": str(ledger),
        "attempt_marker": str(marker),
        "cpu_runs": 1, "gpu_runs": 0, "eda_runs": 0,
        "maximum_workers": 3,
        "frozen_inputs": {"m1524": module.M1524_SHA256,
                          "m528": module.M528_SHA256,
                          "m505": module.M505_SHA256,
                          "m504": module.M504_SHA256,
                          "docs359": module.DOCS359_SHA256},
    }


def fake_m1524() -> Any:
    return SimpleNamespace(
        MODULES=("op0", "op1", "op2", "op3"),
        CHECKPOINT_SHA256="4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48",
        CAPTURE_MANIFEST_SHA256="3ab8431e3d7d17d6933c0b87da4a3405e87c97ccc302a27c78491b0a02491d6d",
        ORDERED_SHA256="5956085b196979848c3d283744396ea3b0a38a268fb21af0eaecb53e87fc6c9c")


def base_summary():
    return ({"aggregate_cycles": {"m468_strong_zero_cycles": 20,
                                   "m505_dead_write_only_1rw_cycles": 10,
                                   "speedup_vs_m468_strong_zero": 2.0},
             "ratio_semantics": "ratio_of_sums_over_ten_ep34_samples",
             "distribution": {}, "conservation": {}, "traffic": {},
             "capacity": {}},
            [{"sample": 0, "cycles": 10}],
            [{"sample": 0, "operator": 0, "cycles": 10}])


def install_tiny(module: Any, marker: Path, fail_materialize: bool = False,
                 mutate_release: Path | None = None) -> dict[str, Any]:
    original = {"M1524": module.M1524,
                "materialize_ledger": module.materialize_ledger,
                "replay": module.replay}
    module.M1524 = fake_m1524()

    def materialize(path):
        require(marker.is_file(), "attempt marker was not consumed before materialization")
        receipt = json.loads(marker.read_text(encoding="utf-8"))
        require(receipt["status"] ==
                "ATTEMPT_CONSUMED_BEFORE_LEDGER_MATERIALIZATION",
                "attempt receipt status drift")
        if fail_materialize:
            raise RuntimeError("synthetic materialization failure")
        path.write_bytes(b"tiny-ledger\n")
        return {"path": path.name, "sha256": sha256(path), "bytes": 12,
                "rows": 1, "line_format": "synthetic",
                "phase_order": "sample,operator,partition",
                "row_order": "timestep,output_y,output_x",
                "captured_input_active_values": 1,
                "captured_input_active_values_by_operator": [1, 0, 0, 0]}

    def replay(_ledger, _workers):
        if mutate_release is not None:
            changed = json.loads(mutate_release.read_text(encoding="utf-8"))
            changed["cpu_runs"] = 99
            mutate_release.write_text(json.dumps(changed), encoding="utf-8")
        return base_summary()

    module.materialize_ledger = materialize
    module.replay = replay
    return original


def restore(module: Any, original: dict[str, Any]) -> None:
    for key, value in original.items():
        setattr(module, key, value)


def verify_m1581() -> None:
    regular_exact(M1581 / "review.json", PINNED["m1581_review"], "M1581 review")
    regular_exact(M1581 / "SHA256SUMS", PINNED["m1581_manifest"],
                  "M1581 manifest")
    regular_exact(M1581 / "SHA256SUMS.seal.sha256", PINNED["m1581_outer"],
                  "M1581 outer")
    require((M1581 / "SHA256SUMS.seal.sha256").read_text(
                encoding="ascii").split() ==
            [PINNED["m1581_manifest"], "SHA256SUMS"],
            "M1581 outer content drift")


def secure_snapshot_hammer(module: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="m1584_snapshot.") as temp:
        base = Path(temp)
        output = base / "result"
        ledger = output / "rows.memh"
        marker = base / "attempt.json"
        release = base / "release.json"
        original = release_value(module, output, ledger, marker)
        payload = json.dumps(original, sort_keys=True).encode("utf-8")
        release.write_bytes(payload)
        parsed, digest = module.read_release_snapshot(release)
        require(parsed == original and digest == hashlib.sha256(payload).hexdigest(),
                "parsed release and digest are not from one snapshot")
        release.write_text(json.dumps(dict(original, cpu_runs=99)), encoding="utf-8")
        require(parsed["cpu_runs"] == 1 and digest != sha256(release),
                "snapshot changed with pathname")

        target = base / "target.json"
        target.write_bytes(payload)
        symlink = base / "release.symlink"
        symlink.symlink_to(target)
        symlink_error = expect_reject(
            lambda: module.read_release_snapshot(symlink), "release symlink")
        directory_error = expect_reject(
            lambda: module.read_release_snapshot(base), "release directory")

        source_text = SOURCE.read_text(encoding="utf-8")
        named_regular_checked_before_open = (
            "stat.S_ISREG(named.st_mode)" in source_text and
            source_text.index("stat.S_ISREG(named.st_mode)") <
            source_text.index("os.open(str(path)"))
        opened_regular_checked_before_read = (
            "stat.S_ISREG(opened.st_mode)" in source_text and
            source_text.index("stat.S_ISREG(opened.st_mode)") <
            source_text.index("os.read(descriptor"))
        fifo = base / "release.fifo"
        os.mkfifo(str(fifo))
        child_code = (
            "import importlib.util; from pathlib import Path; "
            "p=Path(" + repr(str(SOURCE)) + "); "
            "s=importlib.util.spec_from_file_location('m1584_fifo_child',p); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "m.read_release_snapshot(Path(" + repr(str(fifo)) + "))")
        process = subprocess.Popen([sys.executable, "-c", child_code],
                                   cwd=str(ROOT), stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        try:
            process.wait(timeout=0.5)
            fifo_blocked = False
        except subprocess.TimeoutExpired:
            fifo_blocked = True
            process.terminate()
            process.wait(timeout=2.0)
        return {"parsed_bytes_and_digest_same_snapshot": True,
                "pathname_mutation_does_not_change_snapshot": True,
                "symlink_rejected": symlink_error,
                "directory_rejected": directory_error,
                "named_regular_mode_checked_before_open":
                    named_regular_checked_before_open,
                "opened_regular_mode_checked_before_read":
                    opened_regular_checked_before_read,
                "fifo_blocks_before_json_or_attempt_gate": fifo_blocked}


def one_shot_hammer(module: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="m1584_oneshot.") as temp:
        base = Path(temp)
        output = base / "result"
        ledger = output / "rows.memh"
        marker = base / "attempt.json"
        release = base / "release.json"
        release.write_text(json.dumps(release_value(
            module, output, ledger, marker)), encoding="utf-8")
        release_sha = sha256(release)
        original = install_tiny(module, marker)
        try:
            result = module.execute(release, output, ledger, 3)
        finally:
            restore(module, original)
        receipt = json.loads(marker.read_text(encoding="utf-8"))
        require(receipt["release_sha256"] == release_sha and
                receipt["output"] == str(output.resolve()) and
                result["identity"]["release_sha256"] == release_sha,
                "attempt/result release binding drift")
        archived = base / "first_result"
        output.rename(archived)
        original = install_tiny(module, marker)
        try:
            second_error = expect_reject(
                lambda: module.execute(release, output, ledger, 3),
                "same release second execution")
        finally:
            restore(module, original)
        require(marker.is_file() and not output.exists(),
                "second attempt changed consumed state")
        return {"marker_exists_before_materialization": True,
                "marker_release_sha_matches_verified_snapshot": True,
                "result_release_sha_matches_verified_snapshot": True,
                "second_execution_rejected": second_error,
                "maximum_workers": 3}


def failure_consumption_hammer(module: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="m1584_failure.") as temp:
        base = Path(temp)
        output = base / "result"
        ledger = output / "rows.memh"
        marker = base / "attempt.json"
        release = base / "release.json"
        release.write_text(json.dumps(release_value(
            module, output, ledger, marker)), encoding="utf-8")
        original = install_tiny(module, marker, fail_materialize=True)
        try:
            first_error = expect_reject(
                lambda: module.execute(release, output, ledger, 2),
                "synthetic failure")
        finally:
            restore(module, original)
        require(marker.is_file() and not output.exists(),
                "failed attempt did not remain consumed")
        original = install_tiny(module, marker)
        try:
            second_error = expect_reject(
                lambda: module.execute(release, output, ledger, 2),
                "retry after failure")
        finally:
            restore(module, original)
        return {"first_failure": first_error,
                "marker_persists_after_failure": marker.is_file(),
                "retry_after_failure_rejected": second_error}


def toctou_result_hammer(module: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="m1584_toctou.") as temp:
        base = Path(temp)
        output = base / "result"
        ledger = output / "rows.memh"
        marker = base / "attempt.json"
        release = base / "release.json"
        release.write_text(json.dumps(release_value(
            module, output, ledger, marker)), encoding="utf-8")
        verified_sha = sha256(release)
        original = install_tiny(module, marker, mutate_release=release)
        try:
            result = module.execute(release, output, ledger, 1)
        finally:
            restore(module, original)
        current_sha = sha256(release)
        require(current_sha != verified_sha and
                result["identity"]["release_sha256"] == verified_sha and
                json.loads(marker.read_text())["release_sha256"] == verified_sha,
                "verified release SHA was replaced by pathname SHA")
        return {"release_mutated_during_replay": True,
                "current_path_sha_differs": True,
                "result_uses_verified_snapshot_sha": True,
                "attempt_uses_verified_snapshot_sha": True}


def path_binding_hammer(module: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="m1584_paths.") as temp:
        base = Path(temp)
        output = base / "result"
        ledger = output / "rows.memh"
        marker = base / "attempt.json"
        release = base / "release.json"
        value = release_value(module, output, ledger, marker)
        parsed = dict(value)
        expect_reject(lambda: module.verify_release_value(
            parsed, base / "other", ledger, 3), "output mismatch")
        expect_reject(lambda: module.verify_release_value(
            parsed, output, base / "other.memh", 3), "ledger mismatch")
        expect_reject(lambda: module.verify_release_value(
            parsed, output, ledger, 4), "workers above three")

        bad_ledger = base / "outside.memh"
        bad_value = release_value(module, output, bad_ledger, marker)
        release.write_text(json.dumps(bad_value), encoding="utf-8")
        original = install_tiny(module, marker)
        try:
            outside_error = expect_reject(
                lambda: module.execute(release, output, bad_ledger, 3),
                "ledger outside output")
        finally:
            restore(module, original)
        require(marker.is_file(), "invalid first attempt did not remain consumed")
        return {"output_mismatch_rejected": True,
                "ledger_mismatch_rejected": True,
                "workers_above_three_rejected": True,
                "ledger_outside_output_rejected": outside_error,
                "invalid_first_attempt_consumed": True}


def main() -> int:
    regular_exact(SOURCE, PINNED["source"], "M1579 successor source")
    regular_exact(TEST, PINNED["test"], "M1579 successor test")
    regular_exact(DOC359, PINNED["docs359"], "docs/359")
    verify_m1581()
    git_root = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"]).decode().strip()
    require(subprocess.check_output(
        ["git", "-C", git_root, "rev-parse", COMMIT]).decode().strip() == COMMIT,
        "successor commit drift")
    relative = str(SOURCE.relative_to(Path(git_root)))
    committed = subprocess.check_output(
        ["git", "-C", git_root, "show", COMMIT + ":" + relative])
    require(hashlib.sha256(committed).hexdigest() == PINNED["source"],
            "committed source byte drift")

    module = load_source()
    test = subprocess.run([sys.executable, "-m", "unittest", "-v",
        "hw_autoresearch_nts07.system_simulator.tests.test_m1579_ep34_c1_same_ledger_cycle_model"],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, check=False)
    require(test.returncode == 0 and "Ran 7 tests" in test.stderr and
            "OK" in test.stderr, "successor author tests failed")

    snapshot = secure_snapshot_hammer(module)
    one_shot = one_shot_hammer(module)
    failure = failure_consumption_hammer(module)
    toctou = toctou_result_hammer(module)
    paths = path_binding_hammer(module)
    secure_regular = (
        snapshot["named_regular_mode_checked_before_open"] and
        snapshot["opened_regular_mode_checked_before_read"] and
        not snapshot["fifo_blocks_before_json_or_attempt_gate"])
    status = ("PASS_M1584_M1579_ONE_SHOT_TOCTOU_INCREMENT__EXACTLY_ONE_CPU_PRODUCTION_AUTHORIZED" if
              secure_regular else
              "NO_GO_M1584_M1579_CPU_PRODUCTION__RELEASE_SNAPSHOT_LACKS_PRE_READ_REGULAR_FILE_GATE")
    result = {
        "schema": "m1584_m1579_one_shot_toctou_fix_independent_incremental_qa_r1_v1",
        "status": status,
        "commit": COMMIT,
        "pinned_inputs": {"source_sha256": sha256(SOURCE),
                          "test_sha256": sha256(TEST),
                          "m1581_review_sha256": sha256(M1581 / "review.json"),
                          "m1581_manifest_sha256": sha256(M1581 / "SHA256SUMS"),
                          "m1581_outer_sha256": sha256(M1581 / "SHA256SUMS.seal.sha256"),
                          "docs359_sha256": sha256(DOC359)},
        "passed": {"author_tests": "7/7", "secure_snapshot": snapshot,
                   "one_shot_success": one_shot,
                   "failed_attempt_consumption": failure,
                   "release_toctou_result_binding": toctou,
                   "output_ledger_worker_binding": paths,
                   "m1581_core_model_pass_inherited_not_reexecuted": True},
        "p0_finding": {
            "release_regular_file_checked_pre_open_and_post_open": secure_regular,
            "fifo_or_device_can_enter_blocking_read_before_json_rejection":
                not secure_regular,
            "required_fix": (None if secure_regular else
                "Before os.open require lstat regular non-symlink; after open require fstat regular and exact dev/inode/size identity before os.read. Retain O_NOFOLLOW and the exact snapshot digest.")},
        "authorization": {
            "regular_file_gate_successor_authoring": not secure_regular,
            "exactly_one_cpu_production_execution": secure_regular,
            "maximum_workers": 3 if secure_regular else 0,
            "post_result_independent_qa_required": True,
            "production_51m": secure_regular,
            "gpu": False, "rtl": False, "eda": False},
        "claim_boundary": {"incremental_static_and_tiny_synthetic_only": True,
                           "production_replay_executed": False,
                           "gpu": False, "rtl": False, "eda": False,
                           "paper_result": False}}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
