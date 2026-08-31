#!/usr/bin/python3.12
"""M1301 claim-only release-authority successor to frozen M1297.

No M1297 entity, policy, candidate, snapshot, selection or execution semantics
change.  The zero-argument production path first pins the exact M1297 triplet
and the blocking M1298 hammer, then validates the restored M1292 exact-false
claim map before delegating to the fd-bound M1297 executor.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
from types import ModuleType
from typing import Any, Callable, Sequence


M1297_SOURCE = Path(__file__).with_name(
    "run_m1297_m1292_motion_final_checkpoint_binder_interpreter_entity_successor.py")
M1297_SOURCE_SHA256 = "336195e40cf07cfa273be650f2edd0cf1c537c8c70b1c39e68515c087ca81899"
M1297_TEST = Path(__file__).parents[1] / "tests/test_run_m1297_m1292_motion_final_checkpoint_binder_interpreter_entity_successor.py"
M1297_TEST_SHA256 = "1dddabbc1334ed98633e556898cf5df74a4b49089f70c253cae2cf5e408563de"
M1297_CONTRACT = Path(__file__).parents[1] / "contracts/m1297_m1292_motion_final_checkpoint_binder_interpreter_entity_successor_source_contract_r1_20260830.json"
M1297_CONTRACT_SHA256 = "ace730ff38df4ba5025afb46edcd90e6913ef9806058570e2db2db04fdf35cb2"

M1298_REL = Path(
    "hw_autoresearch_nts07/reviews/m1298_m1297_interpreter_entity_fd_bound_receipt_blind_hammer_r1_20260830")
M1298_MANIFEST_SHA256 = "c0f556d43be76e10d1518de44c8d4820292defed860f5ad0a8475b4d5c36b3a1"
M1298_OUTER_SHA256 = "638cdf4a83e3b05e1752faae99ac74b93a35e7b100a09657d6bb3efd1689bca2"
M1298_SCHEMA = "m1298_m1297_interpreter_entity_fd_bound_receipt_blind_hammer_r1_v1"
M1298_STATUS = (
    "BLOCK_M1298_M1297_PRODUCTION__ENTITY_FD_BINDING_PASS__"
    "CONTRACT_CLAIM_KEYSET_DRIFT__ADDITIVE_SUCCESSOR_REQUIRED")

EXACT_CLAIM_BOUNDARY = {
    "checkpoint_selected_now": False,
    "hardware_rebind_authorized": False,
    "hardware_speedup": False,
    "system_speedup": False,
    "power_or_energy": False,
    "paper_metric": False,
    "remote_execution_authorized": False,
}
COMPLETE_TOKEN = "PASS_M1301_ONE_SHOT_SELECTION_RECEIPT__FRESH_RESULT_HAMMER_REQUIRED"


def _load_m1297() -> ModuleType:
    payload = M1297_SOURCE.read_bytes()
    if hashlib.sha256(payload).hexdigest() != M1297_SOURCE_SHA256:
        raise RuntimeError("frozen M1297 source SHA drift")
    module = ModuleType("m1301_frozen_m1297")
    module.__file__ = str(M1297_SOURCE); module.__package__ = ""
    sys.modules[module.__name__] = module
    exec(compile(payload, str(M1297_SOURCE), "exec"), module.__dict__)
    return module


M = _load_m1297()
B = M.M.M.B
PRODUCTION_POLICY = M.PRODUCTION_POLICY


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_claim_boundary(value: Any) -> dict[str, bool]:
    B.require(type(value) is dict and set(value) == set(EXACT_CLAIM_BOUNDARY),
              "claim boundary exact key drift")
    for key, expected in EXACT_CLAIM_BOUNDARY.items():
        B.require(type(value[key]) is bool and value[key] is expected,
                  "claim boundary {} must be exact false boolean".format(key))
    return dict(value)


def _verify_m1298(root: Path) -> None:
    B.directory(root, "M1298 review")
    manifest, manifest_payload = B.snapshot_file(root / B.MANIFEST, "M1298 manifest")
    outer, outer_payload = B.snapshot_file(root / B.OUTER, "M1298 outer")
    B.require(manifest.sha256 == M1298_MANIFEST_SHA256, "M1298 manifest SHA drift")
    B.require(outer.sha256 == M1298_OUTER_SHA256, "M1298 outer SHA drift")
    B.require(outer_payload.decode("utf-8").split() ==
              [M1298_MANIFEST_SHA256, B.MANIFEST], "M1298 outer content drift")
    seen = set(); review_payload = None
    for line in manifest_payload.decode("utf-8").splitlines():
        fields = line.split(None, 1)
        B.require(len(fields) == 2 and len(fields[0]) == 64,
                  "invalid M1298 manifest row")
        name = fields[1].lstrip("*")
        B.require(Path(name).name == name and name not in seen,
                  "invalid M1298 member")
        observed, payload = B.snapshot_file(root / name, "M1298 member " + name)
        B.require(observed.sha256 == fields[0], "M1298 member SHA drift: " + name)
        seen.add(name)
        if name == "review.json": review_payload = payload
    B.require(review_payload is not None, "M1298 review.json missing")
    review = B.strict_json_payload(review_payload, "M1298 review")
    B.require(review.get("schema") == M1298_SCHEMA and
              review.get("status") == M1298_STATUS,
              "M1298 schema/status drift")
    authority = review.get("authority")
    B.require(type(authority) is dict and
              authority.get("exact_reviewed_byte_transfer") == "GO" and
              authority.get("exactly_one_remote_production_execution") == "STOP" and
              authority.get("production_execution_authorized_now") is False and
              authority.get("attempt_may_be_consumed_now") is False and
              authority.get("checkpoint_selected_now") is False,
              "M1298 additive-successor authority drift")


def verify_frozen_authorities(repo: Path) -> None:
    for path, expected, label in (
        (M1297_SOURCE, M1297_SOURCE_SHA256, "M1297 source"),
        (M1297_TEST, M1297_TEST_SHA256, "M1297 test"),
        (M1297_CONTRACT, M1297_CONTRACT_SHA256, "M1297 contract"),
    ):
        mode = path.lstat().st_mode
        B.require(stat.S_ISREG(mode) and not path.is_symlink() and
                  _sha(path) == expected, label + " identity drift")
    _verify_m1298(repo / M1298_REL)
    validate_claim_boundary(EXACT_CLAIM_BOUNDARY)


Runner = Callable[[Sequence[str], Path, tuple[int, ...]], subprocess.CompletedProcess[str]]


def execute_once(policy: Any, cwd: Path, logical_path: Path, real_path: Path,
                 expected: dict[str, Any], runner: Runner = M.default_runner,
                 probe: M.Probe = M.probe_fd_runtime):
    verify_frozen_authorities(policy.base.repo)
    validate_claim_boundary(EXACT_CLAIM_BOUNDARY)
    return M.execute_once(policy, cwd, logical_path, real_path, expected,
                          runner=runner, probe=probe)


def main() -> int:
    B.require(len(sys.argv) == 1, "production M1301 release accepts zero arguments")
    completed = execute_once(PRODUCTION_POLICY, Path.cwd(), M.TARGET_LINK,
                             M.TARGET_REALPATH, dict(M.TARGET_ENTITY))
    sys.stdout.write(completed.stdout)
    if completed.stderr: sys.stderr.write(completed.stderr)
    print(COMPLETE_TOKEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
