#!/usr/bin/env python3
"""Static, no-EDA audit for the M1090 C2 observation-only source release."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
CONTRACT = HW / "contracts/m1090_c2_k1_observation_dc_mapped_vcs_source_contract_r1_20260830.json"
RELEASE = HW / "contracts/m1090_c2_k1_observation_dc_mapped_vcs_release_r1_20260830.json"
WRAPPER = HW / "rtl_m1090/m1090_c2_k1_observation_wrapper.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1090_c2_k1_observation_logic_only_dc.f"
TB = HW / "dc_handoff/tb/tb_m1090_c2_k1_observation_mapped_case0_short.sv"
RUNNER = HW / "dc_handoff/scripts/run_m1091_m1090_c2_observation_dc_mapped_vcs_one_shot_r1.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_double(path: Path) -> str:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(side.read_text(encoding="utf-8").split() == [digest(path), path.relative_to(HW).as_posix()], f"bad sidecar: {path}")
    require(outer.read_text(encoding="utf-8").split() == [digest(side), side.relative_to(HW).as_posix()], f"bad outer seal: {path}")
    return digest(outer)


contract = load(CONTRACT)
release = load(RELEASE)
require(contract["status"] == "M1090_OBSERVATION_SOURCE_ONLY__M1092_REQUIRED__NO_EDA", "contract boundary")
require(release["status"] == "M1090_RELEASE_FROZEN__M1092_REQUIRED__NO_EDA", "release boundary")
require(contract["launch_now"] is False and contract["max_attempts_now"] == 0, "source must not launch")
require(release["launch_now"] is False and release["authorization"]["eda_now"] is False, "release must not launch")
require(contract["next_gate"] == {
    "different_author": "M1092",
    "required_status": "PASS_M1092_M1090_OBSERVATION_SOURCE_HAMMER__GO_ONE_M1091_ATTEMPT",
    "one_attempt_after_hammer": True,
}, "wrong next gate")

contract_outer = verify_double(CONTRACT)
release_outer = verify_double(RELEASE)
require(release["contract_sha256"] == digest(CONTRACT), "release contract pin")
require(release["contract_outer_seal_file_sha256"] == contract_outer, "release contract outer pin")
require(release["runner_sha256"] == digest(RUNNER), "release runner pin")
require(release["wrapper_sha256"] == digest(WRAPPER), "release wrapper pin")
require(release["mapped_tb_sha256"] == digest(TB), "release TB pin")
require(release["filelist_sha256"] == digest(FILELIST), "release filelist pin")

for relative, expected in contract["source_sha256"].items():
    path = HW / relative
    require(path.is_file() and not path.is_symlink(), f"missing/nonregular source: {relative}")
    require(digest(path) == expected, f"source drift: {relative}")

require(digest(DOCS359) == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4", "docs359 drift")
require(contract["docs359_sha256"] == digest(DOCS359), "contract docs359 pin")
require(contract["frozen_failure_authority"]["m1080_retry"] is False, "M1080 retry reopened")
require(release["m1080_do_not_retry"] is True, "release reopened M1080")

expected_filelist = [
    "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv",
    "rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv",
    "rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv",
    "rtl_m218/m218_fc2_tagged_slice_service_island.sv",
    "rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv",
    "rtl_m519/m519_fc2_k1_registered_release_service_island.sv",
    "rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv",
    "rtl_m519/m519_fc2_k1_registered_release_8bank_raw4_acc24.sv",
    "rtl_m519/m519_fc2_k1x8_registered_release_raw4_acc24.sv",
    "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv",
    "rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv",
    "rtl_m1058/m1058_fc2_k1_reset_hygiene_registered_release_service_island.sv",
    "rtl_m1058/m1058_fc2_reset_hygiene_registered_release_standalone_raw4_acc24.sv",
    "rtl_m1058/m1058_fc2_k1_reset_hygiene_registered_release_8bank_raw4_acc24.sv",
    "rtl_m1058/m1058_fc2_reset_hygiene_channel_split_registered_release_matched_8bank_raw4_acc24.sv",
    "rtl_m1090/m1090_c2_k1_observation_wrapper.sv",
]
filelist_lines = [line.strip() for line in FILELIST.read_text(encoding="utf-8").splitlines() if line.strip()]
require(filelist_lines == expected_filelist, "filelist order/set drift")
require(len(filelist_lines) == len(set(filelist_lines)), "duplicate filelist member")

wrapper = WRAPPER.read_text(encoding="utf-8")
require("module m1090_c2_k1_observation_wrapper" in wrapper, "wrong wrapper module")
require(wrapper.count("m1058_fc2_k1_reset_hygiene_registered_release_8bank_raw4_acc24") == 1, "wrong frozen implementation count")
instance_start = wrapper.index(") implementation (")
instance_end = wrapper.index("));", instance_start)
require("obs_" not in wrapper[instance_start:instance_end], "observation port feeds implementation")
obs_names = sorted(set(re.findall(r"\b(obs_[A-Za-z0-9_]+)\b", wrapper)))
require(len(obs_names) == 22, f"unexpected observation count: {len(obs_names)}")
for name in obs_names:
    require(len(re.findall(rf"\b{re.escape(name)}\s*=", wrapper)) == 1, f"{name} must have one fanout assignment")
require("always_ff" not in wrapper and "posedge" not in wrapper, "wrapper added sequential state")

tb = TB.read_text(encoding="utf-8")
require("module tb_m1090_c2_k1_observation_mapped_case0_short" in tb, "wrong TB module")
require(tb.count("`M1090_FAIL_X(") == 22, "not every observation is checked")
require("$isunknown(signal_name)" in tb and "M1090_FIRST_X" in tb, "first-X fail-closed missing")
require("M1090_STAGE" in tb and "window_cycle==128" in tb, "bounded per-cycle diagnostic missing")
require("PASS_M1090_OBSERVATION_SHORT_WINDOW cycles=128 raw_seen=1 no_unknown=1 diagnostic_only=1" in tb, "PASS token drift")
for forbidden in ("$toggle", "+vcs+initreg", ".saif", "$fsdb", "$dumpfile"):
    require(forbidden.lower() not in tb.lower(), f"forbidden TB activity/init feature: {forbidden}")

runner = RUNNER.read_text(encoding="utf-8")
ast.parse(runner, filename=RUNNER.as_posix())
require("M1092=HW/\"reviews/m1092_m1090_c2_observation_source_hammer_r1_20260830\"" in runner, "runner hammer path")
require("M1091_EXPECTED_M1092_OUTER_SHA256" in runner, "runner hammer pin")
require("PASS_M1092_M1090_OBSERVATION_SOURCE_HAMMER__GO_ONE_M1091_ATTEMPT" in runner, "runner hammer token")
require("results/.m1091_m1090_c2_observation_dc_mapped_vcs_attempt_consumed" in runner, "new attempt namespace missing")
require("results/m1091_m1090_c2_observation_dc_mapped_vcs_r1_20260830" in runner, "new result namespace missing")
require("FRESH_DC_M1090_OBSERVATION_TOP" in runner and "FRESH_MAPPED_VCS_CASE0_COMPILE" in runner, "fresh-DC/mapped order markers missing")
require(runner.index("FRESH_DC_M1090_OBSERVATION_TOP") < runner.index("FRESH_MAPPED_VCS_CASE0_COMPILE"), "mapped VCS precedes fresh DC")
for forbidden in ("-ucli", "-cm", "-debug_access", ".saif", "write_saif", "read_saif"):
    require(forbidden not in runner.lower(), f"forbidden runner activity/power invocation: {forbidden}")

old_exact = [
    HW / "rtl_m1089/m1089_c2_k1_observation_wrapper.sv",
    HW / "dc_handoff/filelists/date_m1089_c2_k1_observation_logic_only_dc.f",
    HW / "dc_handoff/tb/tb_m1089_c2_k1_observation_mapped_case0_short.sv",
    HW / "dc_handoff/scripts/run_m1091_m1089_c2_observation_dc_mapped_vcs_one_shot_r1.py",
    HW / "contracts/m1089_c2_k1_observation_dc_mapped_vcs_source_contract_r1_20260830.json",
    HW / "contracts/m1089_c2_k1_observation_dc_mapped_vcs_release_r1_20260830.json",
]
require(not any(path.exists() or path.is_symlink() for path in old_exact), "old M1089 C2 source remains")
for path in old_exact[-2:]:
    require(not Path(str(path) + ".sha256").exists(), "old M1089 sidecar remains")
    require(not Path(str(path) + ".sha256.seal.sha256").exists(), "old M1089 outer seal remains")

for path in (
    HW / "results/.m1091_m1090_c2_observation_dc_mapped_vcs_attempt_consumed",
    HW / "results/m1091_m1090_c2_observation_dc_mapped_vcs_r1_20260830",
):
    require(not path.exists() and not path.is_symlink(), f"EDA namespace already consumed: {path.name}")

print(json.dumps({
    "status": "PASS_M1090_STATIC_SOURCE_CHECK__NO_EDA",
    "checks": 50,
    "observation_outputs": len(obs_names),
    "filelist_members": len(filelist_lines),
    "contract_outer_seal_file_sha256": contract_outer,
    "release_outer_seal_file_sha256": release_outer,
    "eda_executed": False,
    "attempt_consumed": False,
    "docs359_sha256": digest(DOCS359),
}, indent=2, sort_keys=True))
