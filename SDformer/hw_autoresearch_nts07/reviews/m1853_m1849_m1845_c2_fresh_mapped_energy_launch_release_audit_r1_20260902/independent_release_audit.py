#!/usr/bin/env python3
"""Read-only M1853 audit of the M1849 -> M1845 launch authority."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "dc_handoff/scripts/run_m1845_c2_fresh_mapped_production_energy_one_shot.py"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    spec = importlib.util.spec_from_file_location("m1853_runner_audit", str(RUNNER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pins = {
        "M1845_EXPECTED_RUNNER_SHA256": sha(module.RUNNER),
        "M1845_EXPECTED_SOURCE_CONTRACT_SHA256": sha(module.CONTRACT),
        "M1845_EXPECTED_M1848_REVIEW_SHA256": sha(module.M1848_SOURCE_REVIEW / "review.json"),
        "M1845_EXPECTED_M1848_MANIFEST_SHA256": sha(module.M1848_SOURCE_REVIEW / "SHA256SUMS"),
        "M1845_EXPECTED_M1848_OUTER_FILE_SHA256": sha(module.M1848_SOURCE_REVIEW / "SHA256SUMS.seal.sha256"),
        "M1845_EXPECTED_M1849_RELEASE_SHA256": sha(module.M1849_RELEASE),
        "M1845_EXPECTED_M1849_SIDECAR_SHA256": sha(module.M1849_RELEASE_SIDECAR),
        "M1845_EXPECTED_M1849_OUTER_FILE_SHA256": sha(module.M1849_RELEASE_OUTER),
    }
    namespaces = (module.ATTEMPT, module.RESULT, module.FAILURE, module.PRIVATE)
    if any(os.path.lexists(str(path)) for path in namespaces):
        raise RuntimeError("M1845 namespace exists before launch audit")
    previous = {name: os.environ.get(name) for name in pins}
    os.environ.update(pins)
    try:
        mapped = module.verify_authority_and_canonical()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if any(os.path.lexists(str(path)) for path in namespaces):
        raise RuntimeError("static audit created an M1845 namespace")
    print(json.dumps({
        "status": "PASS",
        "verify_authority_and_canonical": True,
        "caller_pins": pins,
        "mapped_axes": sorted(mapped),
        "namespaces_absent": True,
        "license_queries": 0,
        "eda_runs": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
