#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Author static and adversarial mutation checks for M1112r2; never EDA."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import stat
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "dc_handoff/scripts/m1112r2_c2_async_observation_authorized_engine_source_r1.py"
CONTRACT = ROOT / "contracts/m1112r2_c2_async_observation_shadow_source_contract_r1_20260830.json"
WRAPPER_ALIAS = ROOT / "rtl_m1112r2/m1112r2_c2_k1_async_observation_shadow_wrapper.sv"
TB_ALIAS = ROOT / "dc_handoff/tb/tb_m1112r2_c2_k1_async_observation_shadow_case0_short.sv"
WRAPPER = ROOT / "rtl_m1112/m1112_c2_k1_async_observation_shadow_wrapper.sv"
TB = ROOT / "dc_handoff/tb/tb_m1112_c2_k1_async_observation_shadow_case0_short.sv"
DOCS359 = ROOT / "docs/359_DATE终局冻结_20260813.md"
M1113 = ROOT / "reviews/m1113_m1112_c2_async_observation_engine_hammer_r1_20260830"
ENGINE_SHA = "cd4f3eb4d9c659b14fca143651b2e5a4c0d3147335469b9ec22063b1113980c4"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
M1113_OUTER = "ee665be8def8c598669566467a6d1e59dc021a3b0743e2faf43122ed0991da64"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit("FAIL_M1112R2_AUTHOR_CHECK: " + message)


def expect_reject(label: str, callback) -> None:
    try:
        callback()
    except engine.GateFailure:
        mutations[label] = "REJECTED"
        return
    raise SystemExit("FAIL_M1112R2_MUTATION_ACCEPTED: " + label)


for path in (ENGINE, CONTRACT, WRAPPER_ALIAS, TB_ALIAS, WRAPPER, TB, DOCS359):
    require(path.exists() and not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode), f"live input non-regular: {path}")
require(sha(ENGINE) == ENGINE_SHA, "engine identity")
require(sha(DOCS359) == DOCS359_SHA, "docs359 drift")
require(sha(M1113 / "SHA256SUMS.seal.sha256") == M1113_OUTER, "M1113 STOP outer")
ast.parse(ENGINE.read_text(encoding="utf-8"))
require("/bin/dc_shell" not in ENGINE.read_text(encoding="utf-8") and "os.readlink" not in ENGINE.read_text(encoding="utf-8"), "no live tool symlink")

spec = importlib.util.spec_from_file_location("m1112r2_engine", ENGINE)
require(spec is not None and spec.loader is not None, "engine import spec")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)

contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
require(contract["status"] == "M1112R2_RESET_PROVENANCE_AND_LIVE_SEAL_SOURCE_ONLY__INDEPENDENT_HAMMER_REQUIRED__NO_EDA", "contract status")
require(contract["launch_now"] is False and contract["max_attempts_now"] == 0, "source boundary")
for table in ("source_sha256", "frozen_filelist_member_sha256"):
    for relative, expected in contract[table].items():
        path = ROOT / relative
        require(path.exists() and not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode), f"pinned live input type: {relative}")
        require(sha(path) == expected, f"pinned live input digest: {relative}")

wrapper = WRAPPER.read_text(encoding="utf-8")
tb = TB.read_text(encoding="utf-8")
require(wrapper.count("always_ff @(posedge clk_core or posedge rst_core)") == 1, "13-counter async bank")
require(wrapper.count("unused_frozen_debug_") >= 26, "frozen debug isolation")
require(len(set(int(index) for index in __import__("re").findall(r"sample_unknown_bitmap\[(\d+)\]=\$isunknown", tb))) == 22, "22 unknown predicates")
require("unknown_union_bitmap|sample_unknown_bitmap" in tb and "if(window_cycle==128)" in tb, "union/window behavior")
require("`include \"rtl_m1112/m1112_c2_k1_async_observation_shadow_wrapper.sv\"" in WRAPPER_ALIAS.read_text(), "wrapper alias binding")
require("`include \"dc_handoff/tb/tb_m1112_c2_k1_async_observation_shadow_case0_short.sv\"" in TB_ALIAS.read_text(), "TB alias binding")

good = "module mapped(input rst_core);\nINVD1BWP35P140 reset_inv (.I(rst_core), .ZN(rst_core_n));\n"
good += "".join(
    f"DFCNQD1BWP35P140 shadow_service_group_count_q_reg_{index}_ (.D(d{index}), .CP(clk_core), .CDN(rst_core_n), .Q(q{index}));\n"
    for index in range(337)
)
good += "endmodule\n"
accepted = engine.structural_reset_gate_text(good)
require(accepted["shadow_register_bits"] == 337 and accepted["inversion_depth"] == 1, "good reset provenance")

mutations: dict[str, str] = {}
expect_reject("unrelated_fake_reset", lambda: engine.structural_reset_gate_text(good.replace(".I(rst_core)", ".I(unrelated_fake_reset)")))
expect_reject("direct_wrong_polarity", lambda: engine.structural_reset_gate_text(good.replace(".CDN(rst_core_n)", ".CDN(rst_core)")))
expect_reject("constant_clear", lambda: engine.structural_reset_gate_text(good.replace(".CDN(rst_core_n)", ".CDN(1'b0)")))
expect_reject("two_level_inversion", lambda: engine.structural_reset_gate_text(good.replace("INVD1BWP35P140 reset_inv (.I(rst_core), .ZN(rst_core_n));", "INVD1BWP35P140 reset_inv0 (.I(rst_core), .ZN(rst_mid));\nINVD1BWP35P140 reset_inv (.I(rst_mid), .ZN(rst_core_n));")))
expect_reject("reconvergent_gate", lambda: engine.structural_reset_gate_text(good.replace("INVD1BWP35P140 reset_inv (.I(rst_core), .ZN(rst_core_n));", "ND2D1BWP35P140 reset_gate (.A1(rst_core), .A2(other), .ZN(rst_core_n));")))
expect_reject("set_only_cell", lambda: engine.structural_reset_gate_text(good.replace("DFCNQD1BWP35P140", "DFSNQD1BWP35P140").replace(".CDN(rst_core_n)", ".SDN(rst_core_n)")))
last = "DFCNQD1BWP35P140 shadow_service_group_count_q_reg_336_ (.D(d336), .CP(clk_core), .CDN(rst_core_n), .Q(q336));\n"
expect_reject("shadow_census_336", lambda: engine.structural_reset_gate_text(good.replace(last, "")))

engine.verify_exact_flat(engine.M1113_STOP, engine.M1113_STOP_OUTER_SHA256)
engine.verify_exact_flat(engine.M1088, engine.M1088_OUTER_SHA256)
engine.verify_exact_flat(engine.M1080_ATTEMPT, engine.M1080_ATTEMPT_OUTER_SHA256)
engine.verify_historical_m1080()

receipt_dir = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix="m1112r2_mutation_", dir=receipt_dir) as temporary:
    temp = Path(temporary)
    primary = temp / "live.json"
    primary.write_text("{}\n", encoding="utf-8")
    side = Path(str(primary) + ".sha256")
    side.write_text(f"{sha(primary)}  {primary.relative_to(ROOT).as_posix()}\n", encoding="utf-8")
    outer = Path(str(primary) + ".sha256.seal.sha256")
    outer.write_text(f"{sha(side)}  {side.relative_to(ROOT).as_posix()}\n", encoding="utf-8")
    engine.verify_double(primary, sha(primary), sha(outer))
    side_real = temp / "side.real"
    side.rename(side_real)
    side.symlink_to(side_real.name)
    expect_reject("live_sidecar_symlink", lambda: engine.verify_double(primary, sha(primary), sha(outer)))
    side.unlink(); side_real.rename(side)
    outer_real = temp / "outer.real"
    outer.rename(outer_real); outer.symlink_to(outer_real.name)
    expect_reject("live_outer_symlink", lambda: engine.verify_double(primary, sha(primary), sha(outer_real)))
    outer.unlink(); outer_real.rename(outer)
    primary_real = temp / "primary.real"
    primary.rename(primary_real); primary.symlink_to(primary_real.name)
    expect_reject("live_primary_symlink", lambda: engine.verify_double(primary, sha(primary_real), sha(outer)))

with tempfile.TemporaryDirectory(prefix="m1112r2_manifest_", dir=receipt_dir) as temporary:
    sealed = Path(temporary)
    member = sealed / "member.txt"
    member.write_text("member\n", encoding="utf-8")
    manifest = sealed / "SHA256SUMS"
    manifest.write_text(f"{sha(member)}  member.txt\n", encoding="utf-8")
    outer = sealed / "SHA256SUMS.seal.sha256"
    outer.write_text(f"{sha(manifest)}  SHA256SUMS\n", encoding="utf-8")
    engine.verify_exact_flat(sealed, sha(outer))
    extra = sealed / "unlisted.extra"
    extra.write_text("extra\n", encoding="utf-8")
    expect_reject("unlisted_manifest_extra", lambda: engine.verify_exact_flat(sealed, sha(outer)))
    extra.unlink()
    manifest_real = sealed / "manifest.real"
    manifest.rename(manifest_real)
    manifest.symlink_to(manifest_real.name)
    expect_reject("live_manifest_symlink", lambda: engine.verify_exact_flat(sealed, sha(outer)))
    manifest.unlink(); manifest_real.rename(manifest)
    member_real = sealed / "member.real"
    member.rename(member_real); member.symlink_to(member_real.name)
    expect_reject("live_manifest_member_symlink", lambda: engine.verify_exact_flat(sealed, sha(outer)))

require(len(mutations) == 13 and all(value == "REJECTED" for value in mutations.values()), "mutation census")
require(not (ROOT / "dc_handoff/scripts/run_m1112r2_c2_async_observation_authorized_launch_r1.py").exists(), "launcher absent")
require(not (ROOT / "contracts/m1112r2_c2_async_observation_authorized_launch_receipt_r1_20260830.json").exists(), "launch receipt absent")
require(not (ROOT / "results/.m1112r2_c2_async_observation_dc_mapped_vcs_attempt_consumed").exists(), "attempt absent")
require(not (ROOT / "results/m1112r2_c2_async_observation_dc_mapped_vcs_r1_20260830").exists(), "result absent")

print("PASS_M1112R2_AUTHOR_STATIC_MUTATION checks=70 mutations=13/13 eda=0 attempt=0 launcher=0 docs359_unchanged=1")
