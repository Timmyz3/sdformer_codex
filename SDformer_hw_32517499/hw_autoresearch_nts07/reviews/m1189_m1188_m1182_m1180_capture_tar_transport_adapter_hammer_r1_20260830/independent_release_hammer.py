#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Read-only independent semantic hammer for the M1188 transport adapter."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "scripts/run_m1188_m1182_m1180_capture_tar_transport_adapter_source.py"
TEST = HW / "tests/test_run_m1188_m1182_m1180_capture_tar_transport_adapter_source.py"
CONTRACT = HW / "contracts/m1188_m1182_m1180_capture_tar_transport_adapter_source_contract_r1_20260830.json"
M1184 = HW / "reviews/m1184_m1182_m1180_motion_ep29_unified_capture_launch_release_hammer_r1_20260830/review.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    spec = importlib.util.spec_from_file_location("m1188_hammered", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    contract = module.load_contract()
    module.verify_transport_contract(contract)
    members = module.exact_members(contract)
    tests = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", "-m", "unittest", "-v",
         str(TEST.relative_to(ROOT))], cwd=ROOT, shell=False, check=False,
        capture_output=True, text=True)
    if tests.returncode != 0 or "Ran 7 tests" not in tests.stderr or "OK" not in tests.stderr:
        raise RuntimeError("controlled tests failed")
    review = json.loads(M1184.read_text(encoding="utf-8"))
    required = contract["m1184_hammer"]["required_status"]
    semantic_match = review.get("status") == required
    output = {
        "status": "FAIL_CLOSED",
        "p0": 1,
        "controlled_tests_passed": 7,
        "exact_members": len(members),
        "original_exact42": sum(row["class"] == "ORIGINAL_EXACT42" for row in members),
        "m1184_exact9": sum(row["class"] == "M1184_EXACT_SEAL" for row in members),
        "m1184_actual_status": review.get("status"),
        "m1184_actual_verdict": review.get("verdict"),
        "m1188_required_status": required,
        "required_status_matches_actual_status": semantic_match,
        "source_parses_m1184_review_semantics": False,
        "docs359_sha256": sha(DOCS359),
        "execution": {"remote": False, "transfer": False, "gpu": False,
                      "capture": False, "checkpoint": False, "eda": False},
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 1 if not semantic_match else 0


if __name__ == "__main__":
    raise SystemExit(main())
