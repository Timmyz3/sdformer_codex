#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author M1112r2 source/engine hammer; strictly no EDA/launch."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import stat
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ENGINE = ROOT / "dc_handoff/scripts/m1112r2_c2_async_observation_authorized_engine_source_r1.py"
CONTRACT = ROOT / "contracts/m1112r2_c2_async_observation_shadow_source_contract_r1_20260830.json"
WRAPPER_ALIAS = ROOT / "rtl_m1112r2/m1112r2_c2_k1_async_observation_shadow_wrapper.sv"
TB_ALIAS = ROOT / "dc_handoff/tb/tb_m1112r2_c2_k1_async_observation_shadow_case0_short.sv"
WRAPPER = ROOT / "rtl_m1112/m1112_c2_k1_async_observation_shadow_wrapper.sv"
TB = ROOT / "dc_handoff/tb/tb_m1112_c2_k1_async_observation_shadow_case0_short.sv"
AUTHOR = ROOT / "reviews/m1112r2_c2_async_observation_source_receipt_r1_20260830"
M1113 = ROOT / "reviews/m1113_m1112_c2_async_observation_engine_hammer_r1_20260830"
M1080 = ROOT / "results/m1080_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_r1_20260830.failed_or_incomplete.2746017.quarantine"
DOCS359 = ROOT / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "engine": "cd4f3eb4d9c659b14fca143651b2e5a4c0d3147335469b9ec22063b1113980c4",
    "contract": "0f378e5d6100c2d9ae30fcc15a3e3cad53f2fb2d4aa51583c4e53935014b677d",
    "contract_outer": "b2670f2a1f4742235d013f8f7e954db84d80ae2de6d4f6a13e0273e6e10817fa",
    "author_outer": "bafe08fe786b7e51b8f064786ffeb02aa164af39e24674f48ccacaadc0ece2de",
    "m1113_outer": "ee665be8def8c598669566467a6d1e59dc021a3b0743e2faf43122ed0991da64",
    "wrapper_alias": "b1fccaa03b1e3c69205d440ed0e2af93beb0f6eca68e7f7291c67f56322e89f5",
    "tb_alias": "134c4a430d1daa257d73403612cdf41a2bb75369a4f16026413304d38e828d9c",
    "wrapper": "95c31bc70a7617c6653eaca2f77a54388119f744b814dfc909c75edad1c39218",
    "tb": "ff6bd371c3b1371c520b38680960ad0297a8c01eb92eb7b4a0f4d2e59fc861b6",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    return stat.S_ISREG(mode) and not path.is_symlink()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_flat_independent(directory: Path, expected_outer: str, allow_internal_symlink: bool = False) -> dict:
    require(directory.exists() and not directory.is_symlink() and stat.S_ISDIR(directory.lstat().st_mode), f"sealed directory {directory}")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(regular(manifest) and regular(outer), f"regular seal metadata {directory}")
    require(sha(outer) == expected_outer, f"outer identity {directory}")
    require(outer.read_text(encoding="utf-8").split() == [sha(manifest), "SHA256SUMS"], f"outer content {directory}")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]) is not None, "manifest syntax")
        name = fields[1].lstrip("*")
        rel = Path(name)
        require(name and not rel.is_absolute() and ".." not in rel.parts and rel.as_posix() == name, "safe manifest path")
        require(name not in expected, "unique manifest path")
        expected[name] = fields[0]
    actual = set()
    symlinks = []
    root = directory.resolve(strict=True)
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        if stat.S_ISLNK(mode):
            require(allow_internal_symlink, f"live symlink {name}")
            resolved = member.resolve(strict=True)
            require(resolved == root or root in resolved.parents, "historical symlink escape")
            require(regular(resolved), "historical target regular")
            symlinks.append(name)
            actual.add(name)
        elif stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), f"special member {name}")
    require(actual == set(expected), f"exact member coverage {directory}")
    for name, digest in expected.items():
        member = directory / name
        if not allow_internal_symlink:
            require(regular(member), f"live member regular {name}")
        require(sha(member) == digest, f"member digest {name}")
    return {"members": len(expected), "symlinks": symlinks, "outer_seal_file_sha256": sha(outer)}


def import_engine():
    spec = importlib.util.spec_from_file_location("m1112r2_independent_subject", ENGINE)
    require(spec is not None and spec.loader is not None, "engine import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def good_netlist(clear_pin: str = "CDN", inverter: str = "INVD1BWP35P140", bits: int = 337) -> str:
    text = f"module mapped(input rst_core);\n{inverter} reset_inv (.I(rst_core), .ZN(rst_core_n));\n"
    text += "".join(
        f"DFCNQD1BWP35P140 shadow_service_group_count_q_reg_{index}_ (.D(d{index}), .CP(clk_core), .{clear_pin}(rst_core_n), .Q(q{index}));\n"
        for index in range(bits)
    )
    return text + "endmodule\n"


def expect_reject(engine, label: str, text: str, results: dict[str, str]) -> None:
    try:
        engine.structural_reset_gate_text(text)
    except engine.GateFailure:
        results[label] = "REJECTED"
        return
    results[label] = "ACCEPTED"


def reset_attacks(engine) -> dict:
    base = good_netlist()
    accepted_cdn = engine.structural_reset_gate_text(base)
    accepted_cn = engine.structural_reset_gate_text(good_netlist(clear_pin="CN"))
    accepted_cknd = engine.structural_reset_gate_text(good_netlist(inverter="CKND1BWP35P140"))
    require(accepted_cdn["shadow_register_bits"] == accepted_cn["shadow_register_bits"] == accepted_cknd["shadow_register_bits"] == 337, "three legal reset shapes")
    mutations: dict[str, str] = {}
    expect_reject(engine, "unrelated_fake_reset", base.replace(".I(rst_core)", ".I(fake_reset)"), mutations)
    expect_reject(engine, "constant_clear_zero", base.replace(".CDN(rst_core_n)", ".CDN(1'b0)"), mutations)
    expect_reject(engine, "constant_clear_one", base.replace(".CDN(rst_core_n)", ".CDN(1'b1)"), mutations)
    expect_reject(engine, "direct_rst_wrong_polarity", base.replace(".CDN(rst_core_n)", ".CDN(rst_core)"), mutations)
    expect_reject(engine, "multilevel_inversion", base.replace("INVD1BWP35P140 reset_inv (.I(rst_core), .ZN(rst_core_n));", "INVD1BWP35P140 inv0 (.I(rst_core), .ZN(rst_mid));\nINVD1BWP35P140 reset_inv (.I(rst_mid), .ZN(rst_core_n));"), mutations)
    expect_reject(engine, "reconvergent_gate", base.replace("INVD1BWP35P140 reset_inv (.I(rst_core), .ZN(rst_core_n));", "ND2D1BWP35P140 reset_gate (.A1(rst_core), .A2(other), .ZN(rst_core_n));"), mutations)
    expect_reject(engine, "buffer_not_inverter", base.replace("INVD1BWP35P140", "BUFFD1BWP35P140", 1), mutations)
    expect_reject(engine, "set_only", base.replace("DFCNQD1BWP35P140", "DFSNQD1BWP35P140").replace(".CDN(rst_core_n)", ".SDN(rst_core_n)"), mutations)
    expect_reject(engine, "d_clock_only", base.replace(", .CDN(rst_core_n)", ""), mutations)
    expect_reject(engine, "shadow_census_336", good_netlist(bits=336), mutations)
    expect_reject(engine, "two_clear_drivers", base.replace("module mapped(input rst_core);", "module mapped(input rst_core);\nINVD2BWP35P140 reset_inv_duplicate (.I(rst_core), .ZN(rst_core_n));"), mutations)
    expect_reject(engine, "inverter_extra_pin", base.replace(".I(rst_core), .ZN(rst_core_n)", ".I(rst_core), .ZN(rst_core_n), .A(extra)", 1), mutations)
    expect_reject(engine, "active_set_not_inactive", base.replace(".CDN(rst_core_n), .Q", ".CDN(rst_core_n), .SDN(rst_core_n), .Q"), mutations)
    require(all(value == "REJECTED" for value in mutations.values()), f"reset mutation escaped {mutations}")
    return {
        "legal_cdn_single_inv_337": "ACCEPTED",
        "legal_cn_single_inv_337": "ACCEPTED",
        "legal_cknd_single_inv_337": "ACCEPTED",
        "mutations": mutations,
        "mutations_rejected": len(mutations),
    }


def expect_gate_failure(engine, label: str, callback, results: dict[str, str]) -> None:
    try:
        callback()
    except engine.GateFailure:
        results[label] = "REJECTED"
        return
    results[label] = "ACCEPTED"


def live_seal_attacks(engine) -> dict:
    mutations: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="m1114r2_double_", dir=ROOT / "reviews") as raw:
        temp = Path(raw)
        primary = temp / "receipt.json"
        primary.write_text("{}\n", encoding="utf-8")
        side = Path(str(primary) + ".sha256")
        side.write_text(f"{sha(primary)}  {primary.relative_to(ROOT).as_posix()}\n", encoding="utf-8")
        outer = Path(str(primary) + ".sha256.seal.sha256")
        outer.write_text(f"{sha(side)}  {side.relative_to(ROOT).as_posix()}\n", encoding="utf-8")
        engine.verify_double(primary, sha(primary), sha(outer))

        side_real = temp / "side.real"
        side.rename(side_real); side.symlink_to(side_real.name)
        expect_gate_failure(engine, "live_sidecar_symlink", lambda: engine.verify_double(primary, sha(primary), sha(outer)), mutations)
        side.unlink(); side_real.rename(side)
        outer_real = temp / "outer.real"
        outer.rename(outer_real); outer.symlink_to(outer_real.name)
        expect_gate_failure(engine, "live_outer_symlink", lambda: engine.verify_double(primary, sha(primary), sha(outer_real)), mutations)
        outer.unlink(); outer_real.rename(outer)
        primary_real = temp / "primary.real"
        primary.rename(primary_real); primary.symlink_to(primary_real.name)
        expect_gate_failure(engine, "live_primary_symlink", lambda: engine.verify_double(primary, sha(primary_real), sha(outer)), mutations)

    with tempfile.TemporaryDirectory(prefix="m1114r2_flat_", dir=ROOT / "reviews") as raw:
        sealed = Path(raw)
        member = sealed / "review.json"
        member.write_text("{}\n", encoding="utf-8")
        manifest = sealed / "SHA256SUMS"
        manifest.write_text(f"{sha(member)}  review.json\n", encoding="utf-8")
        outer = sealed / "SHA256SUMS.seal.sha256"
        outer.write_text(f"{sha(manifest)}  SHA256SUMS\n", encoding="utf-8")
        engine.verify_exact_flat(sealed, sha(outer))
        extra = sealed / "unlisted.extra"
        extra.write_text("extra\n", encoding="utf-8")
        expect_gate_failure(engine, "unlisted_extra", lambda: engine.verify_exact_flat(sealed, sha(outer)), mutations)
        extra.unlink()
        member_real = sealed / "member.real"
        member.rename(member_real); member.symlink_to(member_real.name)
        expect_gate_failure(engine, "live_member_symlink", lambda: engine.verify_exact_flat(sealed, sha(outer)), mutations)
        member.unlink(); member_real.rename(member)
        manifest_real = sealed / "manifest.real"
        manifest.rename(manifest_real); manifest.symlink_to(manifest_real.name)
        expect_gate_failure(engine, "live_manifest_symlink", lambda: engine.verify_exact_flat(sealed, sha(outer)), mutations)
        manifest.unlink(); manifest_real.rename(manifest)
        outer_real = sealed / "outer.real"
        outer.rename(outer_real); outer.symlink_to(outer_real.name)
        expect_gate_failure(engine, "live_manifest_outer_symlink", lambda: engine.verify_exact_flat(sealed, sha(outer_real)), mutations)
        outer.unlink(); outer_real.rename(outer)
        member.unlink()
        expect_gate_failure(engine, "missing_manifest_member", lambda: engine.verify_exact_flat(sealed, sha(outer)), mutations)

    require(all(value == "REJECTED" for value in mutations.values()), f"live seal mutation escaped {mutations}")
    return {"good_double": "ACCEPTED", "good_exact_flat": "ACCEPTED", "mutations": mutations, "mutations_rejected": len(mutations)}


def wrapper_tb_checks() -> dict:
    wrapper = WRAPPER.read_text(encoding="utf-8")
    tb = TB.read_text(encoding="utf-8")
    alias = WRAPPER_ALIAS.read_text(encoding="utf-8")
    tb_alias = TB_ALIAS.read_text(encoding="utf-8")
    require('`include "rtl_m1112/m1112_c2_k1_async_observation_shadow_wrapper.sv"' in alias, "wrapper alias")
    require('`include "dc_handoff/tb/tb_m1112_c2_k1_async_observation_shadow_case0_short.sv"' in tb_alias, "TB alias")
    widths = []
    names = []
    for width, name in re.findall(r"logic\s+(?:\[(\d+):0\]\s+)?(shadow_\w+_q)\s*;", wrapper):
        widths.append(int(width) + 1 if width else 1); names.append(name)
    require(len(names) == len(set(names)) == 13 and sum(widths) == 337, "13 counters / 337 bits")
    require(wrapper.count("always_ff @(posedge clk_core or posedge rst_core)") == 1, "async counter bank")
    instance = wrapper.split(") implementation (", 1)[1].split("));", 1)[0]
    require(instance.count("unused_frozen_debug_") == 13 and "obs_" not in instance and "shadow_" not in instance, "frozen debug and no feedback")
    export = wrapper.rsplit("always_comb begin", 1)[1]
    require("unused_frozen_debug_" not in export, "frozen debug excluded from observation")
    predicates = re.findall(r"sample_unknown_bitmap\[(\d+)\]=\$isunknown\((obs_\w+)\);", tb)
    require(sorted(int(index) for index, _ in predicates) == list(range(22)) and len({signal for _, signal in predicates}) == 22, "22 atomic bitmap predicates")
    require("window_unknown_bitmap=unknown_union_bitmap|sample_unknown_bitmap;" in tb, "first plus union")
    require("if(window_cycle==128)begin" in tb, "128 close")
    first = tb.split("if((sample_unknown_bitmap!='0)&&!first_x_seen)begin", 1)[1].split("end", 1)[0]
    require("$fatal" not in first, "no first-X short circuit")
    return {"shadow_counters": 13, "shadow_bits": 337, "unknown_predicates": 22, "functional_feedback": False, "frozen_sync_debug_observed": False}


def engine_boundary_checks(source: str, contract: dict) -> dict:
    ast.parse(source)
    require('sys.argv[1:] != ["--authorized-launch"]' in source, "fixed argv")
    require("len(argv) != 2" in source and "zero-argument fixed launcher parent required" in source, "zero-arg launcher parent")
    require(source.count("ATTEMPT.mkdir(); attempted = True") == 1, "single attempt consume")
    require(contract["launch_now"] is False and contract["max_attempts_now"] == 0, "no current attempt authority")
    require(not (ROOT / "dc_handoff/scripts/run_m1112r2_c2_async_observation_authorized_launch_r1.py").exists(), "launcher absent")
    require(not (ROOT / "contracts/m1112r2_c2_async_observation_authorized_launch_receipt_r1_20260830.json").exists(), "launch receipt absent")
    require(not (ROOT / "results/.m1112r2_c2_async_observation_dc_mapped_vcs_attempt_consumed").exists(), "attempt absent")
    require(not (ROOT / "results/m1112r2_c2_async_observation_dc_mapped_vcs_r1_20260830").exists(), "result absent")
    require("verify_exact_flat(M1113_STOP, M1113_STOP_OUTER_SHA256)" in source, "M1113 STOP bind")
    return {"python_ast": "PASS", "engine_argv": ["--authorized-launch"], "zero_argument_launcher_required": True, "attempt_consume_calls": 1, "launcher_absent": True, "attempt_absent": True}


def main() -> None:
    fixed_paths = {
        "engine": ENGINE, "contract": CONTRACT, "wrapper_alias": WRAPPER_ALIAS,
        "tb_alias": TB_ALIAS, "wrapper": WRAPPER, "tb": TB, "docs359": DOCS359,
    }
    for label, path in fixed_paths.items():
        require(regular(path), f"regular fixed input {label}")
        require(sha(path) == EXPECTED[label], f"fixed identity {label}")
    contract_outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    require(regular(contract_outer) and sha(contract_outer) == EXPECTED["contract_outer"], "contract outer")
    author = verify_flat_independent(AUTHOR, EXPECTED["author_outer"])
    stopped = verify_flat_independent(M1113, EXPECTED["m1113_outer"])
    require(json.loads(M1113.joinpath("review.json").read_text(encoding="utf-8"))["status"] == "FAIL_M1113_ENGINE_HAMMER__SOURCE_REPAIR_REQUIRED__NO_LAUNCHER_NO_EDA", "M1113 STOP status")

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    require(contract["status"] == "M1112R2_RESET_PROVENANCE_AND_LIVE_SEAL_SOURCE_ONLY__INDEPENDENT_HAMMER_REQUIRED__NO_EDA", "contract status")
    require(contract["m1113_stop_authority"]["outer_seal_file_sha256"] == EXPECTED["m1113_outer"], "contract M1113 bind")
    for table in ("source_sha256", "frozen_filelist_member_sha256"):
        for relative, digest in contract[table].items():
            path = ROOT / relative
            require(regular(path) and sha(path) == digest, f"pinned live input {relative}")

    source = ENGINE.read_text(encoding="utf-8")
    engine = import_engine()
    reset = reset_attacks(engine)
    live = live_seal_attacks(engine)
    engine.verify_historical_m1080()
    historical = verify_flat_independent(M1080, contract["historical_exception"]["outer_seal_file_sha256"], allow_internal_symlink=True)
    require(historical["symlinks"] == ["mapped_vcs/csrc/_2931510_archive_1.so"], "unique M1080 historical symlink")
    behavior = wrapper_tb_checks()
    boundary = engine_boundary_checks(source, contract)

    status = "PASS_M1114R2_M1112R2_ENGINE_HAMMER__AUTHOR_ZERO_ARG_LAUNCHER_ONLY__NO_EDA"
    checks = {
        "schema": "m1114r2_m1112r2_c2_async_observation_engine_hammer_mechanical_v1",
        "status": status,
        "checks": 83,
        "scope": {"static_and_mutation_only": True, "eda_invocations": 0, "launcher_invocations": 0, "attempts_consumed": 0},
        "identity": {**EXPECTED, "author_receipt": author, "m1113_stop": stopped},
        "reset_provenance": reset,
        "live_seals": live,
        "historical_m1080": historical,
        "preserved_behavior": behavior,
        "engine_boundary": boundary,
    }
    (OUT / "mechanical_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    review = {
        "schema": "m1114r2_m1112r2_c2_async_observation_engine_hammer_review_v1",
        "status": status,
        "verdict": "GO_DIFFERENT_AUTHOR_ZERO_ARGUMENT_LAUNCHER_AUTHORING_ONLY",
        "score": 99,
        "issue_counts": {"P0": 0, "P1": 0, "P2": 0},
        "closure": {
            "m1113_p0_reset_provenance": True,
            "m1113_p0_live_seal_metadata": True,
            "legal_reset_shapes_accepted": 3,
            "reset_mutations_rejected": reset["mutations_rejected"],
            "live_seal_mutations_rejected": live["mutations_rejected"],
            "historical_exception_path": contract["historical_exception"]["only_path"],
            "historical_symlink_count": len(historical["symlinks"]),
        },
        "preserved": behavior,
        "authorization": {
            "different_author_launcher_authoring": True,
            "launcher_execution": False,
            "attempt_creation": False,
            "eda": False,
            "next_required_gate": "A separately sealed zero-argument launcher/receipt and different-author M1115r2 launch hammer.",
        },
        "identity": {
            "engine_sha256": EXPECTED["engine"], "contract_sha256": EXPECTED["contract"],
            "contract_outer_seal_file_sha256": EXPECTED["contract_outer"],
            "author_receipt_outer_seal_file_sha256": EXPECTED["author_outer"],
            "m1113_stop_outer_seal_file_sha256": EXPECTED["m1113_outer"],
            "docs359_sha256": EXPECTED["docs359"],
        },
        "claim_boundary": {"source_launcher_admission_only": True, "mapped_functionality": False, "performance": False, "activity_or_power": False, "system_speedup": False, "paper_citable": False, "paper_ppa_ready": False},
    }
    (OUT / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "RUN_COMPLETE.txt").write_text(status + "\n", encoding="utf-8")
    (OUT / "READONLY_NO_LAUNCH.txt").write_text("M1114r2 source hammer only: no EDA, launcher, launch receipt, attempt, work, result, or quarantine was created.\n", encoding="utf-8")
    print(f"{status} reset_mutations={reset['mutations_rejected']} live_mutations={live['mutations_rejected']} eda=0 launcher=0 attempt=0")


if __name__ == "__main__":
    main()
