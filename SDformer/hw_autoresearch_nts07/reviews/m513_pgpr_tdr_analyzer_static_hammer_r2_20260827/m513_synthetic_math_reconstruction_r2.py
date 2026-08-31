#!/usr/bin/env python3
"""Run the sealed M513 synthetic reconstruction against the r2 analyzer SHA.

The imported reconstruction opens only the analyzer source and generated small
NumPy arrays.  It never opens production capture, payload, contract or model data.
"""

import hashlib
import importlib.util
from pathlib import Path


HW_ROOT = Path(__file__).resolve().parents[2]
R1_RECONSTRUCTION = HW_ROOT / "reviews" / \
    "m513_pgpr_tdr_analyzer_static_hammer_r1_20260827" / \
    "m513_synthetic_math_reconstruction.py"
EXPECTED_R1_RECONSTRUCTION_SHA256 = \
    "ffe84dff30689e11383f506123b55370f6d765f493967c897b19a0b3271ce409"
EXPECTED_R2_ANALYZER_SHA256 = \
    "9790f62d7a3e8fa4ca0ab98947bc6bfb49ae4720bbfb075ec75cebcd3cf7e299"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    assert sha256(R1_RECONSTRUCTION) == EXPECTED_R1_RECONSTRUCTION_SHA256
    spec = importlib.util.spec_from_file_location(
        "m513_sealed_synthetic_reconstruction", str(R1_RECONSTRUCTION))
    reconstruction = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reconstruction)
    reconstruction.EXPECTED_ANALYZER_SHA256 = EXPECTED_R2_ANALYZER_SHA256
    reconstruction.main()


if __name__ == "__main__":
    main()
