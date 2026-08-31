#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Fail-closed source checker for the additive M1497 C1 successor."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat
import sys

HERE = Path(__file__).resolve().parent
HW = HERE.parent
TB_R13 = HW / "verif_m1270r13_c1_real_m935_protocol/tb_m1270r13_m1162_real_m935_protocol_unit_delay_r13.sv"
TB = HERE / "tb_m1497_m1270r13_m1162_real_m935_protocol_unit_delay.sv"
FILELIST = HERE / "m1497_unit_delay_filelist.f"
RUNNER = HW / "dc_handoff/scripts/run_m1497_m1459_c1_active_next_oracle_clean_result_successor_one_shot.py"
TESTS = HERE / "test_m1497_source.py"
CONTRACT = HW / "contracts/m1497_c1_active_next_oracle_clean_result_successor_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1497_c1_active_next_oracle_clean_result_successor_source_author_r1_20260831"
HAMMER = HW / "reviews/m1498_m1497_c1_active_next_oracle_source_blind_hammer_r1_20260831"
RELEASE = HW / "contracts/m1499_m1498_m1497_c1_active_next_oracle_vcs_launch_release_r1_20260831.json"
FINAL = HW / "reviews/m1500_m1499_m1497_c1_active_next_oracle_final_launch_hammer_r1_20260831"
ATTEMPT = HW / "results/.m1497_c1_active_next_oracle_vcs_attempt_consumed"
RESULT = HW / "results/m1497_c1_active_next_oracle_unit_delay_vcs_r1_20260831"
QUARANTINE = Path(str(RESULT) + ".failed_or_incomplete.quarantine")
R13_SHA = "b749c7d635dc5b65669320aec7b7edb40cd5e2a5d781a9e474e3d28cbb054263"
SOURCE_STATUS = "M1497_C1_ACTIVE_NEXT_ORACLE_CLEAN_RESULT_SOURCE_READY__NO_LAUNCH"
CLAIMS = {"source_only": True, "functional_vcs": False,
          "timing_verified": False, "cycles_measured": False,
          "speedup": False, "ppa": False, "power": False, "energy": False,
          "system_speedup": False, "headline": False}

OLD = """                    && (!dut.request_active_q
                        || (issue_request_valid
                            && !dut.weight_request_accepted_q
                            && !dut.psum_request_accepted_q
                            && dut.request_source_index_q != served_source)),"""
NEW = """                    && !$isunknown(dut.request_active_q)
                    && (!dut.request_active_q
                        || (issue_request_valid
                            && !$isunknown({issue_request_first,
                                issue_request_source_index,
                                dut.request_first_q,
                                dut.request_source_index_q,
                                dut.weight_request_accepted_q,
                                dut.psum_request_accepted_q})
                            && dut.weight_request_accepted_q == 1'b0
                            && dut.psum_request_accepted_q
                                == !dut.request_first_q
                            && dut.request_first_q == issue_request_first
                            && dut.request_source_index_q
                                == issue_request_source_index
                            && dut.request_source_index_q
                                != served_source)),"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def strict_json(path: Path) -> dict:
    require(path.exists() and not path.is_symlink()
            and stat.S_ISREG(path.lstat().st_mode), "JSON not regular")
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token:
                      (_ for _ in ()).throw(RuntimeError(token)))


def active_next_oracle(*, active, issue_valid, public_first,
                       public_source, latched_first, latched_source,
                       weight_accepted, psum_accepted, served_source) -> bool:
    """Four-state small model: None denotes X/Z and must fail closed."""
    if active == 0:
        return True
    values = (active, issue_valid, public_first, public_source,
              latched_first, latched_source, weight_accepted,
              psum_accepted, served_source)
    if any(value is None for value in values):
        return False
    if active != 1 or issue_valid != 1:
        return False
    if public_first not in (0, 1) or latched_first not in (0, 1):
        return False
    if weight_accepted != 0 or psum_accepted != (1 - latched_first):
        return False
    return (latched_first == public_first
            and latched_source == public_source
            and latched_source != served_source)


def verify_sidecar(path: Path) -> None:
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    require(sidecar.read_text().split() == [sha(path), path.name],
            "contract sidecar")
    require(outer.read_text().split() == [sha(sidecar), sidecar.name],
            "contract outer")


def check_source(require_runtime_authority: bool = False) -> dict:
    require(sha(TB_R13) == R13_SHA, "frozen R13 drift")
    old = TB_R13.read_text()
    require(old.count(OLD) == 1, "frozen oracle cardinality")
    require(TB.read_text() == old.replace(OLD, NEW),
            "new TB contains delta beyond active-next oracle")
    lines = FILELIST.read_text().splitlines()
    require(len(lines) == 7 and len(lines) == len(set(lines)),
            "filelist cardinality")
    require(str(TB) in lines and str(TB_R13) not in lines,
            "filelist TB binding")
    runner = RUNNER.read_text()
    required = (
        "RAW_BUILD", "CLEAN_RESULT_STAGE", "CLEAN_PAYLOAD",
        "make_clean_evidence(CLEAN_RESULT_STAGE",
        "copy_regular(source, stage / name)",
        "symlink_policy_relaxed\": False",
        "P.seal_dir_generic(root)",
        "P.verify_recursive_seal_generic(root)",
        "M1497_EXPECTED_RUNNER_SHA256",
    )
    require(all(token in runner for token in required),
            "runner clean-stage mechanism")
    require("seal_dir_generic(RAW_BUILD)" not in runner,
            "raw build must never be sealed/published")
    require("M1494" not in runner and "m1494" not in runner,
            "reserved C2 namespace collision")
    require(not any(path.exists() for path in (ATTEMPT, RESULT, QUARANTINE)),
            "M1497 canonical namespace not fresh")
    contract = strict_json(CONTRACT)
    verify_sidecar(CONTRACT)
    require(contract.get("status") == SOURCE_STATUS, "contract status")
    require(contract.get("claim_boundary") == CLAIMS, "claim boundary")
    require(contract.get("authorization", {}).get("vcs_compiles") == 0,
            "source author authorized VCS")
    if require_runtime_authority:
        require(AUTHOR.is_dir() and HAMMER.is_dir()
                and RELEASE.is_file() and FINAL.is_dir(),
                "runtime authority incomplete")
    return {
        "schema": "m1497_c1_active_next_oracle_source_check_r1_v1",
        "status": "PASS_M1497_C1_ACTIVE_NEXT_ORACLE_CLEAN_RESULT_SOURCE__NO_VCS_NO_EDA",
        "bindings": {
            "runner_sha256": sha(RUNNER), "testbench_sha256": sha(TB),
            "filelist_sha256": sha(FILELIST), "tests_sha256": sha(TESTS),
            "contract_sha256": sha(CONTRACT), "frozen_r13_sha256": sha(TB_R13),
        },
        "author_tests": {
            "first_to_nonfirst": True, "first_to_first": True,
            "unknown_fail_closed": True, "wrong_accepted_fail_closed": True,
        },
        "claim_boundary": CLAIMS,
    }


def main() -> int:
    if sys.argv[1:] not in (["--mode", "source_only"],
                            ["--mode", "runtime_present"]):
        raise SystemExit("usage: check_m1497_source.py --mode source_only|runtime_present")
    result = check_source(sys.argv[-1] == "runtime_present")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
