#!/usr/bin/python3.12
"""M1306 additive inherited-authority successor to frozen M1301.

The only semantic repair is restoring frozen M1297.main's inherited M1257
authority preflight.  M1301 seals/claims are checked first; then the exact
M1297.M.verify_frozen_authorities() is called once before delegation to the
unchanged fd-bound M1297 executor.
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


M1301_SOURCE = Path(__file__).with_name(
    "run_m1301_m1297_motion_final_checkpoint_binder_claim_authority_successor.py")
M1301_SOURCE_SHA256 = "e8db73150c8d08ad52f4cf39d2013e1207c17db1192141fa002789b722203b4a"
M1301_TEST = Path(__file__).parents[1] / "tests/test_run_m1301_m1297_motion_final_checkpoint_binder_claim_authority_successor.py"
M1301_TEST_SHA256 = "de6381edde6f4722085c830c6960032fdd738e8f4fdb05fc76bf522927a48a30"
M1301_CONTRACT = Path(__file__).parents[1] / "contracts/m1301_m1297_motion_final_checkpoint_binder_claim_authority_successor_source_contract_r1_20260830.json"
M1301_CONTRACT_SHA256 = "4aec2a68ac47a76bbef7b9ac773568ed0465d2d70fda4701c8a5e37fca7413ae"

M1303_REL = Path(
    "hw_autoresearch_nts07/reviews/m1303_m1301_final_checkpoint_binder_claim_authority_blind_hammer_r1_20260830")
M1303_MANIFEST_SHA256 = "8d2a938ebd475bca3b2a7dc0adbdc51c4848604d3255f1706234b266ce788b04"
M1303_OUTER_SHA256 = "67294688d5285a0836e8e401525e2835d7d87ea4aad6fb013d1914e52e8c2ff5"
M1303_SCHEMA = "m1303_m1301_final_checkpoint_binder_claim_authority_blind_hammer_r1_v1"
M1303_STATUS = (
    "BLOCK_M1303_M1301_PRODUCTION__CLAIM_REPAIR_PASS__"
    "M1297_INHERITED_AUTHORITY_PREFLIGHT_OMITTED__ADDITIVE_SUCCESSOR_REQUIRED")
COMPLETE_TOKEN = "PASS_M1306_ONE_SHOT_SELECTION_RECEIPT__FRESH_RESULT_HAMMER_REQUIRED"


def _load_m1301() -> ModuleType:
    payload = M1301_SOURCE.read_bytes()
    if hashlib.sha256(payload).hexdigest() != M1301_SOURCE_SHA256:
        raise RuntimeError("frozen M1301 source SHA drift")
    module = ModuleType("m1306_frozen_m1301")
    module.__file__ = str(M1301_SOURCE); module.__package__ = ""
    sys.modules[module.__name__] = module
    exec(compile(payload, str(M1301_SOURCE), "exec"), module.__dict__)
    return module


M = _load_m1301()
B = M.B
PRODUCTION_POLICY = M.PRODUCTION_POLICY
EXACT_CLAIM_BOUNDARY = dict(M.EXACT_CLAIM_BOUNDARY)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_m1303(root: Path) -> None:
    B.directory(root, "M1303 review")
    manifest, manifest_payload = B.snapshot_file(root / B.MANIFEST, "M1303 manifest")
    outer, outer_payload = B.snapshot_file(root / B.OUTER, "M1303 outer")
    B.require(manifest.sha256 == M1303_MANIFEST_SHA256, "M1303 manifest SHA drift")
    B.require(outer.sha256 == M1303_OUTER_SHA256, "M1303 outer SHA drift")
    B.require(outer_payload.decode("utf-8").split() ==
              [M1303_MANIFEST_SHA256, B.MANIFEST], "M1303 outer content drift")
    seen = set(); review_payload = None
    for line in manifest_payload.decode("utf-8").splitlines():
        fields = line.split(None, 1)
        B.require(len(fields) == 2 and len(fields[0]) == 64,
                  "invalid M1303 manifest row")
        name = fields[1].lstrip("*")
        B.require(Path(name).name == name and name not in seen,
                  "invalid M1303 member")
        observed, payload = B.snapshot_file(root / name, "M1303 member " + name)
        B.require(observed.sha256 == fields[0], "M1303 member SHA drift: " + name)
        seen.add(name)
        if name == "review.json": review_payload = payload
    B.require(review_payload is not None, "M1303 review.json missing")
    review = B.strict_json_payload(review_payload, "M1303 review")
    B.require(review.get("schema") == M1303_SCHEMA and
              review.get("status") == M1303_STATUS,
              "M1303 schema/status drift")
    authority = review.get("authority")
    B.require(type(authority) is dict and
              authority.get("exactly_one_remote_production_execution") == "STOP" and
              authority.get("production_execution_authorized_now") is False and
              authority.get("attempt_may_be_consumed_now") is False and
              authority.get("checkpoint_selected_now") is False,
              "M1303 successor authority drift")


def verify_frozen_authorities(repo: Path) -> None:
    for path, expected, label in (
        (M1301_SOURCE, M1301_SOURCE_SHA256, "M1301 source"),
        (M1301_TEST, M1301_TEST_SHA256, "M1301 test"),
        (M1301_CONTRACT, M1301_CONTRACT_SHA256, "M1301 contract"),
    ):
        mode = path.lstat().st_mode
        B.require(stat.S_ISREG(mode) and not path.is_symlink() and
                  _sha(path) == expected, label + " identity drift")
    _verify_m1303(repo / M1303_REL)


Runner = Callable[[Sequence[str], Path, tuple[int, ...]], subprocess.CompletedProcess[str]]


def execute_once(policy: Any, cwd: Path, logical_path: Path, real_path: Path,
                 expected: dict[str, Any], runner: Runner = M.M.default_runner,
                 probe: M.M.Probe = M.M.probe_fd_runtime):
    # New M1306 seal, then frozen M1301 seal/claim gate.
    verify_frozen_authorities(policy.base.repo)
    M.verify_frozen_authorities(policy.base.repo)
    M.validate_claim_boundary(EXACT_CLAIM_BOUNDARY)
    # Sole repair: preserve the authority check previously performed by
    # frozen M1297.main before invoking M1297.execute_once.
    M.M.M.verify_frozen_authorities()
    return M.M.execute_once(policy, cwd, logical_path, real_path, expected,
                            runner=runner, probe=probe)


def main() -> int:
    B.require(len(sys.argv) == 1, "production M1306 release accepts zero arguments")
    completed = execute_once(PRODUCTION_POLICY, Path.cwd(), M.M.TARGET_LINK,
                             M.M.TARGET_REALPATH, dict(M.M.TARGET_ENTITY))
    sys.stdout.write(completed.stdout)
    if completed.stderr: sys.stderr.write(completed.stderr)
    print(COMPLETE_TOKEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
