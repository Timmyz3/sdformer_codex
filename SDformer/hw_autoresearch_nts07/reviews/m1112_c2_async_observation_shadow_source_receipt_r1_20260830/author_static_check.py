#!/usr/bin/env python3
"""Author-side static source check for M1112.  It never invokes EDA."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "rtl_m1112/m1112_c2_k1_async_observation_shadow_wrapper.sv"
TB = ROOT / "dc_handoff/tb/tb_m1112_c2_k1_async_observation_shadow_case0_short.sv"
FILELIST = ROOT / "dc_handoff/filelists/date_m1112_c2_k1_async_observation_shadow_logic_only_dc.f"
ENGINE = ROOT / "dc_handoff/scripts/m1112_c2_async_observation_authorized_engine_source_r1.py"
CONTRACT = ROOT / "contracts/m1112_c2_async_observation_shadow_source_contract_r1_20260830.json"
M1109 = ROOT / "reviews/m1109_m1091r3_c2_observation_mapped_x_failure_audit_r1_20260830"
DOCS359 = ROOT / "docs/359_DATE终局冻结_20260813.md"
EXPECTED_ENGINE = "d3aa6f25c43cf13df03493bc92b0d41b59273de8e7f9ae25b5e3f5ab64fbd125"
EXPECTED_M1109_OUTER = "5c7a1f667c6c800f84a0e8219ddf58574412090812cda5d8bdaf36265f43d52d"
EXPECTED_DOCS359 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit("FAIL_M1112_AUTHOR_STATIC_CHECK: " + message)


def regular(path: Path) -> bool:
    return path.exists() and not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode)


for path in (WRAPPER, TB, FILELIST, ENGINE, CONTRACT, DOCS359):
    require(regular(path), f"live source is absent/non-regular: {path}")

contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
require(contract["status"] == "M1112_ASYNC_OBSERVATION_SHADOW_SOURCE_ONLY__INDEPENDENT_HAMMER_REQUIRED__NO_EDA", "contract status")
require(contract["launch_now"] is False and contract["max_attempts_now"] == 0, "source attempt boundary")
require(sha(DOCS359) == EXPECTED_DOCS359, "docs359 drift")
require(sha(ENGINE) == EXPECTED_ENGINE, "engine drift")
require(sha(M1109 / "SHA256SUMS.seal.sha256") == EXPECTED_M1109_OUTER, "M1109 outer drift")

side = Path(str(CONTRACT) + ".sha256")
outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
require(regular(side) and regular(outer), "contract seal files")
contract_digest, contract_name = side.read_text(encoding="utf-8").split()
require(contract_digest == sha(CONTRACT), "contract digest")
require(contract_name == CONTRACT.relative_to(ROOT).as_posix(), "contract sidecar path")
outer_digest, outer_name = outer.read_text(encoding="utf-8").split()
require(outer_digest == sha(side), "contract outer digest")
require(outer_name == side.relative_to(ROOT).as_posix(), "contract outer path")

for table_name in ("source_sha256", "frozen_filelist_member_sha256"):
    for relative, expected in contract[table_name].items():
        path = ROOT / relative
        require(regular(path), f"pinned live input not regular: {relative}")
        require(sha(path) == expected, f"pinned live input drift: {relative}")

filelist_members = [line.strip() for line in FILELIST.read_text(encoding="utf-8").splitlines() if line.strip()]
require(filelist_members[-1] == "rtl_m1112/m1112_c2_k1_async_observation_shadow_wrapper.sv", "filelist top")
require(len(filelist_members) == len(set(filelist_members)), "filelist duplicate")
require(set(filelist_members[:-1]) == set(contract["frozen_filelist_member_sha256"]), "filelist frozen member coverage")

wrapper = WRAPPER.read_text(encoding="utf-8")
require(wrapper.count("always_ff @(posedge clk_core or posedge rst_core)") == 1, "single async shadow bank")
shadow_names = re.findall(r"logic\s+(?:\[[^;]+\]\s+)?(shadow_\w+_q)\s*;", wrapper)
require(len(shadow_names) == 13 and len(set(shadow_names)) == 13, "13 shadow registers")
for name in shadow_names:
    require(wrapper.count(name + " <= '0;") == 2, f"async+epoch reset coverage: {name}")
require("initial begin" not in wrapper and "$isunknown" not in wrapper and not re.search(r"#\s*\d", wrapper), "no RTL masking/initialization/delay")
instance = wrapper.split(") implementation (", 1)[1].split("));", 1)[0]
require("obs_" not in instance and "shadow_" not in instance, "no observation feedback into implementation")
require(instance.count("unused_frozen_debug_") == 13, "frozen debug outputs terminated")
observation = wrapper.split("always_comb begin", 2)[2]
require("unused_frozen_debug_" not in observation, "frozen sync debug not observed")
for name in shadow_names:
    require(name in observation, f"shadow not exported: {name}")
for functional in ("header_ready", "raw_ready", "mem_req_valid", "mem_rsp_ready", "result_valid", "result_accumulator", "token_done_valid"):
    require(not re.search(rf"{functional}\s*=\s*shadow_", wrapper), f"functional feedback: {functional}")

tb = TB.read_text(encoding="utf-8")
indices = sorted(int(value) for value in re.findall(r"sample_unknown_bitmap\[(\d+)\]=\$isunknown\(", tb))
require(indices == list(range(22)), "atomic 22-signal bitmap census")
sample_block = tb.split("always_comb begin", 2)[2].split("end", 1)[0]
require("$fatal" not in sample_block, "fatal inside atomic sample")
first_x_block = tb.split('if((sample_unknown_bitmap!=\'0)&&!first_x_seen)begin', 1)[1].split("end", 1)[0]
require("$fatal" not in first_x_block, "first-X short circuit")
require("unknown_union_bitmap|sample_unknown_bitmap" in tb, "later-X union")
require("if(window_cycle==128)" in tb, "complete window gate")
require("M1112_FIRST_X cycle=%0d bitmap=%06h" in tb, "first-X report")
require(contract["unknown_sampling_contract"]["pass_token"] in tb, "exact PASS token")

engine = ENGINE.read_text(encoding="utf-8")
ast.parse(engine)
require('sys.argv[1:] != ["--authorized-launch"]' in engine, "fixed engine argv")
require("len(argv) != 2" in engine, "zero-argument launcher parent")
require("verify_historical_quarantine()" in engine and "symlinks != 1" in engine, "historical exception boundary")
require("SHADOW_REGISTER_BITS = 337" in engine, "mapped shadow census")
require("structural_reset_gate(netlist)" in engine, "mapped resettable-cell gate")
require('print("M1112 failure: "' in engine and "M1091r2 failure:" not in engine, "stderr identity repair")
require(not (ROOT / "dc_handoff/scripts/run_m1112_c2_async_observation_authorized_launch_r1.py").exists(), "launcher must remain absent")
require(not (ROOT / "contracts/m1112_c2_async_observation_authorized_launch_receipt_r1_20260830.json").exists(), "launch receipt must remain absent")
require(not (ROOT / "results/.m1112_c2_async_observation_dc_mapped_vcs_attempt_consumed").exists(), "attempt must remain absent")
require(not (ROOT / "results/m1112_c2_async_observation_dc_mapped_vcs_r1_20260830").exists(), "result must remain absent")

print("PASS_M1112_AUTHOR_STATIC_SOURCE_RECEIPT checks=47 eda=0 attempt=0 launcher=0 docs359_unchanged=1")
