#!/opt/conda/envs/sdformerflow/bin/python
"""Read-only remote preflight for the exact M1313/M1249 ep34 launch."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M1249_SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1249_motion_final_checkpoint_unified_hardware_one_shot_release_r1.py")
M1249_SOURCE_SHA256 = "5fbcc4d287f3ffd3b1c9994efa24245e5e3828927cdac925c1a35d8a88a19219"
CONTRACT = HW / (
    "contracts/m1313_motion_ep34_final_unified_capture_production_launch_r1_20260831.json")
CONTRACT_SHA256 = "eeb0a8380e51610652ec6cdf1c2bb58c22395c9d72608e98f6a88a18f5c6bbda"
EXPECTED = {
    "candidate_id": "resume_ep34",
    "epoch": 34,
    "checkpoint_sha256": "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48",
    "config_sha256": "630e735c8fe1d643b524ecd82ecf69d514df548d36380144cef442541daa4d39",
    "profile_sha256": "144ba2d94eeafd2b6549a7b0aa7d0c89d2b334fe814a7d45f71d6990670e379c",
}
PASS_TOKEN = "PASS_M1317_M1249_EP34_REMOTE_READ_ONLY_PREFLIGHT"


class PreflightError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise PreflightError(message)


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
        raise PreflightError("missing " + label) from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(), label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA mismatch")


def load_m1249():
    regular_exact(M1249_SOURCE, M1249_SOURCE_SHA256, "M1249 source")
    spec = importlib.util.spec_from_file_location("m1317_remote_sealed_m1249", str(M1249_SOURCE))
    require(spec is not None and spec.loader is not None, "cannot load M1249")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def preflight(contract_path: Path = CONTRACT) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    require(contract_path == CONTRACT, "only the canonical M1313 contract is allowed")
    regular_exact(CONTRACT, CONTRACT_SHA256, "M1313 production contract")
    module = load_m1249()
    namespaces = (module.CANONICAL_RESULT, module.CANONICAL_ATTEMPT, module.CANONICAL_LOG)
    require(all(not os.path.lexists(str(path)) for path in namespaces),
            "M1249 result/attempt/log namespace is not fresh before preflight")
    contract = module.strict_json(CONTRACT)
    binding = module.validate_production_launch(contract, CONTRACT)
    identity = binding["identity"]
    require(all(identity[key] == value for key, value in EXPECTED.items()),
            "final checkpoint identity mismatch")
    samples = binding["verified_samples"]
    require(isinstance(samples, list) and len(samples) == 40,
            "exact forty-sample cohort was not verified")
    require(all(not os.path.lexists(str(path)) for path in namespaces),
            "read-only preflight changed an M1249 namespace")
    return {
        **EXPECTED,
        "samples": 40,
        "result_fresh": True,
        "attempt_fresh": True,
        "canonical_log_fresh": True,
        "automatic_retry": False,
        "capture_executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    args = parser.parse_args()
    result = preflight(args.contract)
    print(PASS_TOKEN + " " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
