#!/usr/bin/env python3
"""Final tiny/static incremental QA of M1579 commit 842da3aa."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/run_m1579_ep34_c1_same_ledger_cycle_model.py"
TEST = HW / "system_simulator/tests/test_m1579_ep34_c1_same_ledger_cycle_model.py"
M1584 = HW / "reviews/m1584_m1579_one_shot_release_toctou_fix_independent_incremental_qa_r1_20260901"
M1584_HAMMER = M1584 / "independent_incremental_qa.py"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
COMMIT = "842da3aa7e9aa2441e23953b643a836198d568d0"

PINNED = {
    "source": "e0f09bd218af6733c17b50781ab9c3a4f13117821e24e14ea0eaa2864c1535b5",
    "test": "c2cc102be14496b79bde2cee57a892bf76383c3180dba74563f78162ae4dec89",
    "m1584_review": "9c5780f965008bee1593fb91483333baace5bd882032a404fc2e1bc6a931318a",
    "m1584_manifest": "4c1b58b4019983024accc459ff81e053eaf61aef2fda7649af9de3cb5c3e2d3c",
    "m1584_outer": "e5cb9561fe9e4ed16dc19490bfae4a336ba4d87b9d66b7bdcad073ad479b424c",
    "m1584_hammer": "d3ef0dabb0772e3b49cd404fe3841f2f876575aa40d37c66f7571eec95eb8787",
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


def load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def expect_reject(function, label: str) -> str:
    try:
        function()
    except BaseException as error:
        require(not isinstance(error, (KeyboardInterrupt, SystemExit)),
                label + " escaped process")
        return type(error).__name__
    raise RuntimeError(label + " did not fail closed")


def verify_m1584() -> int:
    regular_exact(M1584 / "review.json", PINNED["m1584_review"], "M1584 review")
    regular_exact(M1584 / "SHA256SUMS", PINNED["m1584_manifest"],
                  "M1584 manifest")
    regular_exact(M1584 / "SHA256SUMS.seal.sha256", PINNED["m1584_outer"],
                  "M1584 outer")
    regular_exact(M1584_HAMMER, PINNED["m1584_hammer"], "M1584 hammer")
    require((M1584 / "SHA256SUMS.seal.sha256").read_text(
                encoding="ascii").split() ==
            [PINNED["m1584_manifest"], "SHA256SUMS"],
            "M1584 outer content drift")
    expected = {}
    for line in (M1584 / "SHA256SUMS").read_text(
            encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        require(name not in expected and "/" not in name and ".." not in name,
                "M1584 manifest malformed")
        expected[name] = digest
        regular_exact(M1584 / name, digest, "M1584 member " + name)
    actual = set(path.name for path in M1584.iterdir() if path.is_file() and
                 path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    require(actual == set(expected), "M1584 seal coverage drift")
    return len(expected)


def source_order_static() -> dict[str, bool]:
    text = SOURCE.read_text(encoding="utf-8")
    positions = {
        "named_lstat": text.index("named = path.lstat()"),
        "named_regular": text.index("stat.S_ISREG(named.st_mode)"),
        "open": text.index("descriptor = os.open(str(path), flags)"),
        "opened_fstat": text.index("opened = os.fstat(descriptor)"),
        "opened_regular": text.index("stat.S_ISREG(opened.st_mode)"),
        "current_lstat": text.index("current = path.lstat()"),
        "triple_identity": text.index("(current.st_dev, current.st_ino, current.st_size) =="),
        "read": text.index("block = os.read(descriptor, 1 << 20)"),
    }
    return {
        "named_lstat_before_open":
            positions["named_lstat"] < positions["open"],
        "named_regular_non_symlink_before_open":
            positions["named_regular"] < positions["open"] and
            "and not path.is_symlink()" in text[
                positions["named_regular"]:positions["open"]],
        "opened_fstat_regular_before_read":
            positions["open"] < positions["opened_fstat"] <
            positions["opened_regular"] < positions["read"],
        "current_named_opened_identity_before_read":
            positions["current_lstat"] < positions["triple_identity"] <
            positions["read"],
        "O_NOFOLLOW_retained": "flags |= os.O_NOFOLLOW" in text,
    }


def nonregular_immediate_hammer(module: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="m1589_nonregular.") as temp:
        base = Path(temp)
        output = base / "result"
        ledger = output / "rows.memh"
        marker = base / "attempt.json"
        target = base / "target.json"
        target.write_text("{}", encoding="utf-8")
        symlink = base / "release.symlink"
        symlink.symlink_to(target)
        fifo = base / "release.fifo"
        os.mkfifo(str(fifo))
        errors = {
            "directory": expect_reject(lambda: module.execute(
                base, output, ledger, 1), "directory release"),
            "symlink": expect_reject(lambda: module.execute(
                symlink, output, ledger, 1), "symlink release"),
        }

        child_code = (
            "import importlib.util; from pathlib import Path; "
            "p=Path(" + repr(str(SOURCE)) + "); "
            "s=importlib.util.spec_from_file_location('m1589_fifo_child',p); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "m.execute(Path(" + repr(str(fifo)) + "),Path(" +
            repr(str(output)) + "),Path(" + repr(str(ledger)) + "),1)")
        process = subprocess.Popen([sys.executable, "-c", child_code],
                                   cwd=str(ROOT), stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        try:
            return_code = process.wait(timeout=1.0)
            fifo_immediate = return_code != 0
        except subprocess.TimeoutExpired:
            fifo_immediate = False
            process.terminate()
            process.wait(timeout=2.0)
        require(fifo_immediate and not marker.exists() and not output.exists(),
                "FIFO blocked or mutated attempt/output")
        return {"directory_rejected": errors["directory"],
                "symlink_rejected": errors["symlink"],
                "fifo_rejected_without_timeout": fifo_immediate,
                "attempt_marker_absent": not marker.exists(),
                "output_absent": not output.exists()}


def path_swap_hammer(module: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="m1589_swap.") as temp:
        base = Path(temp)
        release = base / "release.json"
        replacement = base / "replacement.json"
        release.write_text(json.dumps({"which": "named"}), encoding="utf-8")
        replacement.write_text(json.dumps({"which": "opened"}), encoding="utf-8")
        original_open = module.os.open
        original_replace = module.os.replace
        swapped = [False]

        def swap_then_open(path, flags, *args):
            if Path(path) == release and not swapped[0]:
                swapped[0] = True
                original_replace(str(replacement), str(release))
            return original_open(path, flags, *args)

        module.os.open = swap_then_open
        try:
            error = expect_reject(lambda: module.read_release_snapshot(release),
                                  "named/opened identity swap")
        finally:
            module.os.open = original_open
        return {"swap_performed": swapped[0], "identity_swap_rejected": error}


def main() -> int:
    regular_exact(SOURCE, PINNED["source"], "M1579 final source")
    regular_exact(TEST, PINNED["test"], "M1579 final test")
    regular_exact(DOC359, PINNED["docs359"], "docs/359")
    m1584_members = verify_m1584()
    git_root = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"]).decode().strip()
    require(subprocess.check_output(
        ["git", "-C", git_root, "rev-parse", COMMIT]).decode().strip() == COMMIT,
        "final commit drift")
    relative = str(SOURCE.relative_to(Path(git_root)))
    committed = subprocess.check_output(
        ["git", "-C", git_root, "show", COMMIT + ":" + relative])
    require(hashlib.sha256(committed).hexdigest() == PINNED["source"],
            "committed source byte drift")

    module = load(SOURCE, "m1589_bound_m1579")
    sealed_hammer = load(M1584_HAMMER, "m1589_bound_m1584_hammer")
    tests = subprocess.run([sys.executable, "-m", "unittest", "-v",
        "hw_autoresearch_nts07.system_simulator.tests.test_m1579_ep34_c1_same_ledger_cycle_model"],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, check=False)
    require(tests.returncode == 0 and "Ran 8 tests" in tests.stderr and
            "OK" in tests.stderr, "final author tests failed")

    static = source_order_static()
    require(all(static.values()), "regular-file gate ordering drift")
    nonregular = nonregular_immediate_hammer(module)
    swap = path_swap_hammer(module)
    snapshot = sealed_hammer.secure_snapshot_hammer(module)
    require(snapshot["named_regular_mode_checked_before_open"] and
            snapshot["opened_regular_mode_checked_before_read"] and
            not snapshot["fifo_blocks_before_json_or_attempt_gate"],
            "M1584 secure-snapshot P0 not closed")
    one_shot = sealed_hammer.one_shot_hammer(module)
    failure = sealed_hammer.failure_consumption_hammer(module)
    toctou = sealed_hammer.toctou_result_hammer(module)
    paths = sealed_hammer.path_binding_hammer(module)

    result = {
        "schema": "m1589_m1579_regular_file_gate_final_incremental_qa_r1_v1",
        "status": "PASS_M1589_M1579_FINAL_RELEASE_GATE__EXACTLY_ONE_51840000_ROW_CPU_PRODUCTION_AUTHORIZED_MAX3_WORKERS",
        "commit": COMMIT,
        "pinned_inputs": {"source_sha256": sha256(SOURCE),
                          "test_sha256": sha256(TEST),
                          "m1584_review_sha256": sha256(M1584 / "review.json"),
                          "m1584_manifest_sha256": sha256(M1584 / "SHA256SUMS"),
                          "m1584_outer_sha256": sha256(M1584 / "SHA256SUMS.seal.sha256"),
                          "docs359_sha256": sha256(DOC359)},
        "passed": {"m1584_sealed_members": m1584_members,
                   "author_tests": "8/8",
                   "regular_gate_static_order": static,
                   "nonregular_immediate_rejection": nonregular,
                   "named_opened_current_identity_swap": swap,
                   "opened_snapshot_regression": snapshot,
                   "O_EXCL_success_and_second_rejection": one_shot,
                   "failed_attempt_remains_consumed": failure,
                   "verified_snapshot_result_binding": toctou,
                   "output_ledger_worker_binding": paths,
                   "M1581_core_same_ledger_model_pass_inherited": True},
        "authorization": {
            "exactly_one_cpu_production_execution": True,
            "source_rows": 51840000,
            "maximum_workers": 3,
            "fresh_release_and_attempt_namespace_required": True,
            "post_result_independent_qa_required": True,
            "paper_citable_before_result_qa": False,
            "gpu": False, "rtl": False, "eda": False},
        "claim_boundary": {"incremental_static_and_tiny_synthetic_only": True,
                           "production_replay_executed_by_M1589": False,
                           "authorization_is_cycle_model_CPU_once": True,
                           "rtl_cycle": False, "full_network": False,
                           "system_speedup": False, "energy": False,
                           "paper_result": False}}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
