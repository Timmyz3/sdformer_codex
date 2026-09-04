from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/report_local5_source_owned_same_port_ablation.py"
SPEC = importlib.util.spec_from_file_location("same_port", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_percentile_and_bad_marker_contract() -> None:
    assert MODULE.percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert MODULE.BAD_RE.search("MISMATCH")
    assert not MODULE.BAD_RE.search("PASS Local5")
