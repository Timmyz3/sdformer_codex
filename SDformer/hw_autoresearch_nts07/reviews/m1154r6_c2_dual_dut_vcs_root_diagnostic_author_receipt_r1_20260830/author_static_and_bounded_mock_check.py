#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Author check for M1154R6; static/bounded mock only, no VCS or DC."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "dc_handoff/scripts/run_m1154r6_c2_dual_dut_vcs_root_diagnostic_source_r1.py"
SOURCE_SHA = "5e39999037463a5b190f61c66c9d895f7ad1af93bb7d9d7503d737b8133350e4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if sha256(SOURCE) != SOURCE_SHA:
    raise SystemExit("source identity drift")
spec = importlib.util.spec_from_file_location("m1154r6_author_check_subject", SOURCE)
if spec is None or spec.loader is None:
    raise SystemExit("source import spec")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
value = module.source_bounded_mock_self_test()
if not (value["status"] == "PASS_SOURCE_AND_BOUNDED_MOCK__REAL_STABLE_TAP_GATE_STOP" and
        value["real_preflight"]["stable_tap_census"]["present_count"] == 5 and
        value["real_preflight"]["stable_tap_census"]["missing_count"] == 8 and
        value["attempt_created"] is False and value["vcs_calls"] == 0 and
        value["dc_calls"] == 0 and module.namespace_fresh()):
    raise SystemExit("bounded mock or real fail-closed gate drift")
print(json.dumps(value, sort_keys=True, allow_nan=False))
