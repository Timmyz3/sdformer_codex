#!/usr/bin/env python3
"""Source-only static and executable pre-gate test for M1187/R4."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parent
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1187_m1168r3_m1162_c1_common_charge_protocol_exact_sha_r4.sh"
PRE_GATE = HERE / "validate_m1187_m1168r3_vcs_pre_attempt_gate_r4.py"
CONTRACT = HW / "contracts/m1187_m1168r3_m1162_c1_vcs_launcher_source_contract_r4_20260830.json"
RELEASE = HW / "contracts/m1187_m1168r3_m1162_c1_vcs_launch_release_r4_20260830.json"
SOURCE_HAMMER = HW / "reviews/m1182_m1181_m1168r3_m1162_c1_common_charge_protocol_vcs_source_hammer_r1_20260830"
SOURCE_AUTHOR = HW / "reviews/m1181_m1168r3_m1162_c1_common_charge_protocol_vcs_source_author_receipt_r1_20260830"
R2_Q = HW / "results/m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs_r2_20260830.failed_or_incomplete.3284331.quarantine"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
FUTURE_HAMMER = HW / "reviews/m1188_m1187_m1168r3_c1_vcs_release_hammer_r1_20260830"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(directory: Path) -> None:
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    manifest = directory / "SHA256SUMS"
    manifest.write_text("".join(f"{sha(p)}  {p.relative_to(directory).as_posix()}\n" for p in files))
    (directory / "SHA256SUMS.seal.sha256").write_text(f"{sha(manifest)}  SHA256SUMS\n")


def synthetic_gate(release_mutator=None, hammer_mutator=None) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory(prefix="m1187_static_gate.") as td:
        root = Path(td)
        release = json.loads(RELEASE.read_text())
        release_path = root / "release.json"
        hammer_dir = root / "release_hammer"
        hammer_dir.mkdir()
        release["fresh_release_hammer"]["path"] = str(hammer_dir)
        if release_mutator:
            release_mutator(release)
        release_path.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n")
        hammer = {
            "schema": "m1188_m1187_m1168r3_c1_vcs_release_hammer_review_r1_v1",
            "status": "PASS_M1188_M1187_C1_VCS_RELEASE_HAMMER__AUTHORIZE_ONE_LAUNCH",
            "verdict": "GO",
            "score": 99,
            "issue_counts": {"P0": 0, "P1": 0, "P2": 0},
            "identity": {"release_sha256": sha(release_path), "runner_sha256": sha(RUNNER),
                         "source_contract_sha256": sha(CONTRACT)},
            "execution_audit": {"vcs_compiles": 0, "simv_runs": 0, "all_eda_runs": 0},
            "authorization": {"vcs_compiles": 1, "simv_runs": 1, "all_other_eda_runs": 0}
        }
        if hammer_mutator:
            hammer_mutator(hammer)
        review = hammer_dir / "review.json"
        review.write_text(json.dumps(hammer, indent=2, sort_keys=True) + "\n")
        (hammer_dir / "NO_EDA.txt").write_text("synthetic static-gate fixture only\n")
        seal(hammer_dir)
        env = dict(os.environ)
        env.update({
            "M1187_EXPECTED_RELEASE_SHA256": sha(release_path),
            "M1187_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256": sha(SOURCE_HAMMER / "review.json"),
            "M1187_EXPECTED_SOURCE_HAMMER_OUTER_SHA256": sha(SOURCE_HAMMER / "SHA256SUMS.seal.sha256"),
            "M1187_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256": sha(review),
            "M1187_EXPECTED_RELEASE_HAMMER_OUTER_SHA256": sha(hammer_dir / "SHA256SUMS.seal.sha256")
        })
        return subprocess.run([str(PYTHON), "-I", str(PRE_GATE), str(CONTRACT), str(RUNNER),
                               str(SOURCE_HAMMER), str(release_path), str(hammer_dir),
                               str(SOURCE_AUTHOR), str(R2_Q)], text=True, capture_output=True, env=env)


def main() -> None:
    runner = RUNNER.read_text()
    contract = json.loads(CONTRACT.read_text())
    release = json.loads(RELEASE.read_text())
    assert sha(RUNNER) == "ea369894be301e4594f252e5edad3534b93363c8287310f62363b4d7fa8b8caa"
    assert sha(PRE_GATE) == "792563125a0711cd3e584f12c863d99fd5a9a3770347846b62c903ff154664d4"
    assert sha(CONTRACT) == "427421aaa4ee0919d81beb5a869866608d13a1e1baa3ca14edd8e0606e482651"
    assert sha(RELEASE) == "54460115fa9bfdc5f5bb1eae759177abe8ed49a5de3961ff9d1fa891ab2e43e5"
    assert contract["identity"]["runner_sha256"] == sha(RUNNER)
    assert release["identity"]["runner_sha256"] == sha(RUNNER)
    assert release["identity"]["source_contract_sha256"] == sha(CONTRACT)
    assert "contract_sha256" not in release["identity"]
    assert 'ident["source_contract_sha256"]' in PRE_GATE.read_text()
    assert "i['contract_sha256']" not in runner
    assert runner.count('"${VCS_BIN}" -full64') == 1
    assert runner.count("./simv -no_save") == 1
    assert "+define+UNIT_DELAY" in runner
    assert runner.index('"${PYTHON_BIN}" -I "${PRE_GATE}"') < runner.index('mkdir -- "${ATTEMPT}"')
    for name in ("M1187_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256",
                 "M1187_EXPECTED_RELEASE_HAMMER_OUTER_SHA256"):
        assert name in runner and name in release["required_environment"]
    assert "verify_recursive_seal \"${RELEASE_HAMMER}\"" in runner
    assert "EDA collision" in runner and "MemAvailable below 64 GiB" in runner
    assert 'seal_dir "${WORK}"' in runner and "failed_or_incomplete.$$.quarantine" in runner
    assert "legal_masks_clear=29" in runner and "service_assumption_attacks=2" in runner
    assert not FUTURE_HAMMER.exists()
    for path in (HW / "results/.m1168r3_m1162_c1_common_charge_protocol_vcs_r3_attempt_consumed",
                 HW / "results/m1168r3_m1162_c1_common_charge_protocol_unit_delay_vcs_r3_20260830",
                 HW / "results/.m1187_m1168r3_m1162_c1_common_charge_protocol_vcs_r4_attempt_consumed",
                 HW / "results/m1187_m1168r3_m1162_c1_common_charge_protocol_unit_delay_vcs_r4_20260830"):
        assert not path.exists(), path

    good = synthetic_gate()
    assert good.returncode == 0 and good.stdout.strip() == "PASS_M1187_R4_PRE_ATTEMPT_GATE", (good.stdout, good.stderr)
    mutations = [
        (None, lambda d: d.__setitem__("status", "NO_GO")),
        (None, lambda d: d.__setitem__("verdict", "NO_GO")),
        (None, lambda d: d.__setitem__("score", 94)),
        (None, lambda d: d["issue_counts"].__setitem__("P0", 1)),
        (None, lambda d: d["issue_counts"].__setitem__("P1", 1)),
        (None, lambda d: d["identity"].__setitem__("release_sha256", "0" * 64)),
        (None, lambda d: d["identity"].__setitem__("runner_sha256", "0" * 64)),
        (None, lambda d: d["identity"].__setitem__("source_contract_sha256", "0" * 64)),
        (None, lambda d: d["execution_audit"].__setitem__("vcs_compiles", 1)),
        (None, lambda d: d["authorization"].__setitem__("simv_runs", 2)),
        (lambda d: d["identity"].__setitem__("source_contract_sha256", "0" * 64), None),
        (lambda d: d["authorization"].__setitem__("vcs_compiles", 2), None)
    ]
    for release_mutator, hammer_mutator in mutations:
        bad = synthetic_gate(release_mutator, hammer_mutator)
        assert bad.returncode != 0

    print(json.dumps({
        "schema": "m1187_m1168r3_vcs_launcher_release_static_check_r4_v1",
        "status": "PASS_M1187_R4_SOURCE_AND_EXECUTABLE_PRE_ATTEMPT_GATE__FRESH_M1188_HAMMER_REQUIRED__NO_VCS_NO_EDA",
        "exact_pre_attempt_parse_passed": True,
        "runtime_key_error": False,
        "fresh_release_hammer_runtime_bound": True,
        "mutations_rejected": len(mutations),
        "vcs_compile_cardinality": 1,
        "simv_cardinality": 1,
        "unit_delay": True,
        "r3_namespace_absent": True,
        "r4_namespace_absent": True,
        "vcs_runs": 0,
        "simv_runs": 0,
        "all_eda_runs": 0,
        "license_queries": 0
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
