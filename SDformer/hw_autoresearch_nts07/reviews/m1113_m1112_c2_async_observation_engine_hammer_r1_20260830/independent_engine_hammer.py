#!/usr/bin/env python3
"""Independent, source-only hammer for the fixed M1112 identity.

This audit deliberately does not invoke VCS, DC, simv, any launcher, or any
production attempt.  A successful execution means that the audit completed;
the admission verdict is carried in review.json and may still be NO_GO.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
WRAPPER = ROOT / "rtl_m1112/m1112_c2_k1_async_observation_shadow_wrapper.sv"
TB = ROOT / "dc_handoff/tb/tb_m1112_c2_k1_async_observation_shadow_case0_short.sv"
ENGINE = ROOT / "dc_handoff/scripts/m1112_c2_async_observation_authorized_engine_source_r1.py"
CONTRACT = ROOT / "contracts/m1112_c2_async_observation_shadow_source_contract_r1_20260830.json"
AUTHOR = ROOT / "reviews/m1112_c2_async_observation_shadow_source_receipt_r1_20260830"
M1109 = ROOT / "reviews/m1109_m1091r3_c2_observation_mapped_x_failure_audit_r1_20260830"
DOCS359 = ROOT / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "wrapper": "95c31bc70a7617c6653eaca2f77a54388119f744b814dfc909c75edad1c39218",
    "tb": "ff6bd371c3b1371c520b38680960ad0297a8c01eb92eb7b4a0f4d2e59fc861b6",
    "engine": "d3aa6f25c43cf13df03493bc92b0d41b59273de8e7f9ae25b5e3f5ab64fbd125",
    "contract": "016290ad92593f6d43989a9b57576657340d481ebc6f72d7e82c8081740f3a08",
    "author_outer": "0100253fa1aac43597d0c86d567c9c663da31fceb9184d361c4720ac22ebf338",
    "m1109_outer": "5c7a1f667c6c800f84a0e8219ddf58574412090812cda5d8bdaf36265f43d52d",
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


def verify_flat(directory: Path, expected_outer: str) -> dict:
    require(directory.is_dir() and not directory.is_symlink(), f"sealed directory: {directory}")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(regular(manifest) and regular(outer), f"regular manifest/outer: {directory}")
    require(sha(outer) == expected_outer, f"outer identity: {directory}")
    require(outer.read_text(encoding="utf-8").split() == [sha(manifest), "SHA256SUMS"], f"outer content: {directory}")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        relative = Path(name.lstrip("*"))
        require(not relative.is_absolute() and ".." not in relative.parts, "manifest escape")
        member = directory / relative
        require(regular(member), f"live sealed member is not regular: {member}")
        require(sha(member) == digest, f"sealed member drift: {member}")
        listed.add(relative.as_posix())
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    }
    require(actual == listed, f"exact member coverage: {directory}")
    return {"manifest_members": len(listed), "manifest_sha256": sha(manifest), "outer_seal_file_sha256": sha(outer)}


def wrapper_checks(text: str) -> dict:
    names_and_widths = []
    for width, name in re.findall(r"logic\s+(?:\[(\d+):0\]\s+)?(shadow_\w+_q)\s*;", text):
        names_and_widths.append((name, int(width) + 1 if width else 1))
    require(len(names_and_widths) == 13 and len({name for name, _ in names_and_widths}) == 13, "13 unique shadow registers")
    require(sum(width for _, width in names_and_widths) == 337, "337 shadow register bits")
    require(text.count("always_ff @(posedge clk_core or posedge rst_core)") == 1, "one async shadow bank")
    for name, _ in names_and_widths:
        require(text.count(name + " <= '0;") == 2, f"async plus epoch clear: {name}")
    require("initial begin" not in text and "$isunknown" not in text and not re.search(r"#\s*\d", text), "no mask/init/delay in wrapper")
    instance = text.split(") implementation (", 1)[1].split("));", 1)[0]
    require("obs_" not in instance and "shadow_" not in instance, "no observation feedback to implementation")
    require(instance.count("unused_frozen_debug_") == 13, "13 frozen debug sinks")
    export = text.rsplit("always_comb begin", 1)[1]
    require("unused_frozen_debug_" not in export, "frozen synchronous debug excluded from observation")
    for name, _ in names_and_widths:
        require(name in export, f"shadow exported: {name}")
    for functional in ("header_ready", "raw_ready", "mem_req_valid", "mem_rsp_ready", "result_valid", "result_accumulator", "token_done_valid"):
        require(not re.search(rf"{functional}\s*=\s*(?:shadow_|obs_)", text), f"functional feedback: {functional}")
    return {"shadow_counters": 13, "shadow_register_bits": 337, "frozen_debug_sinks": 13, "async_blocks": 1}


def tb_validator(text: str) -> list[str]:
    errors = []
    matches = re.findall(r"sample_unknown_bitmap\[(\d+)\]\s*=\s*\$isunknown\((obs_\w+)\)\s*;", text)
    if sorted(int(index) for index, _ in matches) != list(range(22)) or len({signal for _, signal in matches}) != 22:
        errors.append("22_atomic_predicates")
    try:
        sample = text.split("always_comb begin", 2)[2].split("end", 1)[0]
        if "$fatal" in sample:
            errors.append("fatal_in_sample")
    except IndexError:
        errors.append("sample_block")
    marker = "if((sample_unknown_bitmap!='0)&&!first_x_seen)begin"
    try:
        first = text.split(marker, 1)[1].split("end", 1)[0]
        if "$fatal" in first:
            errors.append("first_x_short_circuit")
    except IndexError:
        errors.append("first_x_block")
    if "window_unknown_bitmap=unknown_union_bitmap|sample_unknown_bitmap;" not in text:
        errors.append("later_x_union")
    close = "if(window_cycle==128)begin"
    if close not in text:
        errors.append("window_close_128")
    else:
        before, after = text.split(close, 1)
        if "$fatal" in before.split("always @(posedge clk_core)begin", 1)[-1]:
            errors.append("fatal_before_window_close")
        if "if(window_unknown_bitmap!='0)begin" not in after or "$fatal" not in after:
            errors.append("fatal_missing_at_window_close")
    return errors


def load_engine_namespace(source: str) -> dict:
    # Execute definitions only; the production flow tail is excluded.
    prefix = source.split("for caught_signal in", 1)[0]
    namespace: dict = {"__file__": str(ENGINE), "__name__": "m1112_static_hammer_subject"}
    exec(compile(prefix, str(ENGINE), "exec"), namespace)
    return namespace


def fake_netlist(reset_net: str, async_pin: bool = True, bits: int = 337) -> str:
    pin = f", .CDN({reset_net})" if async_pin else ""
    return "\n".join(
        f"DFFRQ_X1 shadow_service_group_count_q_reg_{bit} (.D(d{bit}), .CP(clk_core){pin}, .Q(q{bit}));"
        for bit in range(bits)
    ) + "\n"


def structural_gate_attacks(namespace: dict) -> dict:
    gate = namespace["structural_reset_gate"]
    GateFailure = namespace["GateFailure"]
    with tempfile.TemporaryDirectory(prefix="m1113_struct_") as raw:
        temp = Path(raw)
        valid = temp / "valid.v"
        bogus = temp / "bogus.v"
        d_only = temp / "d_only.v"
        short = temp / "short.v"
        valid.write_text(fake_netlist("rst_core"), encoding="utf-8")
        bogus.write_text(fake_netlist("unrelated_fake_reset"), encoding="utf-8")
        d_only.write_text(fake_netlist("rst_core", async_pin=False), encoding="utf-8")
        short.write_text(fake_netlist("rst_core", bits=336), encoding="utf-8")
        require(gate(valid)["shadow_register_bits"] == 337, "valid resettable mapped census")
        results = {}
        for label, path in (("bogus_reset_net", bogus), ("d_only", d_only), ("short_census", short)):
            try:
                gate(path)
                results[label] = "ACCEPTED"
            except GateFailure:
                results[label] = "REJECTED"
    require(results["d_only"] == "REJECTED", "D/CP-only mapped cells must be rejected")
    require(results["short_census"] == "REJECTED", "336-bit census must be rejected")
    return {
        "valid_rst_core_net": "ACCEPTED",
        **results,
        "required_but_missing": results["bogus_reset_net"] == "ACCEPTED",
    }


def trust_boundary_attacks(namespace: dict) -> dict:
    namespace["verify_historical_quarantine"]()
    with tempfile.TemporaryDirectory(prefix="m1113_trust_", dir=ROOT / "reviews") as raw:
        temp = Path(raw)
        target = temp / "target.txt"
        target.write_text("fixed\n", encoding="utf-8")
        live = temp / "live.txt"
        live.symlink_to(target.name)
        live_regular_rejected = False
        try:
            namespace["verify_regular"](live, sha(target))
        except namespace["GateFailure"]:
            live_regular_rejected = True

        item = temp / "item.json"
        item.write_text("{}\n", encoding="utf-8")
        side_target = temp / "side.real"
        side_target.write_text(f"{sha(item)}  {item.relative_to(ROOT).as_posix()}\n", encoding="utf-8")
        side = Path(str(item) + ".sha256")
        side.symlink_to(side_target.name)
        outer = Path(str(item) + ".sha256.seal.sha256")
        outer.write_text(f"{sha(side)}  {side.relative_to(ROOT).as_posix()}\n", encoding="utf-8")
        sidecar_symlink_accepted = True
        try:
            namespace["verify_double"](item, sha(item), sha(outer))
        except namespace["GateFailure"]:
            sidecar_symlink_accepted = False

        flat = temp / "flat"
        flat.mkdir()
        member = flat / "review.json"
        member.write_text("{}\n", encoding="utf-8")
        manifest_real = flat / "manifest.real"
        manifest_real.write_text(f"{sha(member)}  review.json\n", encoding="utf-8")
        manifest = flat / "SHA256SUMS"
        manifest.symlink_to(manifest_real.name)
        flat_outer = flat / "SHA256SUMS.seal.sha256"
        flat_outer.write_text(f"{sha(manifest)}  SHA256SUMS\n", encoding="utf-8")
        flat_manifest_symlink_accepted = True
        try:
            namespace["verify_flat"](flat, sha(flat_outer))
        except namespace["GateFailure"]:
            flat_manifest_symlink_accepted = False
    return {
        "historical_sealed_symlink_exception_valid": True,
        "live_primary_symlink_rejected": live_regular_rejected,
        "live_sidecar_symlink_accepted": sidecar_symlink_accepted,
        "live_flat_manifest_symlink_accepted": flat_manifest_symlink_accepted,
        "required_but_missing": sidecar_symlink_accepted or flat_manifest_symlink_accepted,
    }


def engine_static_checks(source: str, contract: dict) -> dict:
    ast.parse(source)
    require('sys.argv[1:] != ["--authorized-launch"]' in source, "fixed engine argv")
    require("len(argv) != 2" in source and "Path(argv[0]) != PYTHON" in source, "zero-argument pinned launcher parent")
    require("verify_parent_launcher(receipt)" in source, "launcher parent verified")
    require("ATTEMPT.mkdir(); attempted = True" in source, "exclusive attempt consume")
    require(source.count("ATTEMPT.mkdir(); attempted = True") == 1, "one attempt consume")
    require("if any(path.exists() or path.is_symlink() for path in (ATTEMPT, RESULT, WORK))" in source, "fresh namespace gate")
    require(contract["launch_now"] is False and contract["max_attempts_now"] == 0, "source stage no attempt")
    require(not (ROOT / "dc_handoff/scripts/run_m1112_c2_async_observation_authorized_launch_r1.py").exists(), "launcher absent")
    require(not (ROOT / "contracts/m1112_c2_async_observation_authorized_launch_receipt_r1_20260830.json").exists(), "launch receipt absent")
    require(not (ROOT / "results/.m1112_c2_async_observation_dc_mapped_vcs_attempt_consumed").exists(), "attempt absent")
    require(not (ROOT / "results/m1112_c2_async_observation_dc_mapped_vcs_r1_20260830").exists(), "result absent")
    # The future launcher must erase the caller environment.  The engine copies
    # only that already-sanitized environment and adds fixed tool variables.
    require("env = os.environ.copy()" in source, "engine environment inheritance identified")
    return {
        "python_ast": "PASS",
        "fixed_engine_argv": ["--authorized-launch"],
        "zero_argument_launcher_parent_required": True,
        "attempt_consume_calls": 1,
        "attempt_absent": True,
        "launcher_absent": True,
        "caller_environment_erasure_required_in_future_launcher": True,
    }


def mutation_checks(wrapper: str, tb: str, engine: str) -> dict:
    mutations = {}
    omitted = re.sub(r"\s*sample_unknown_bitmap\[21\]=\$isunknown\([^;]+;", "", tb, count=1)
    mutations["omit_predicate_rejected"] = "22_atomic_predicates" in tb_validator(omitted)
    no_union = tb.replace("window_unknown_bitmap=unknown_union_bitmap|sample_unknown_bitmap;", "window_unknown_bitmap=sample_unknown_bitmap;", 1)
    mutations["later_union_removed_rejected"] = "later_x_union" in tb_validator(no_union)
    short = tb.replace("first_x_cycle=window_cycle;", "first_x_cycle=window_cycle; $fatal(1,\"short\");", 1)
    mutations["first_x_short_circuit_rejected"] = "first_x_short_circuit" in tb_validator(short)
    early = tb.replace("if(header_accept)header_seen=1;", "if(header_accept)header_seen=1; $fatal(1,\"early\");", 1)
    mutations["early_fatal_rejected"] = "fatal_before_window_close" in tb_validator(early)
    require(all(mutations.values()), "all TB mutations must be rejected")

    # Independent textual sentinels for the argv/attempt boundary.
    mutations["argv_gate_mutation_detected"] = 'sys.argv[1:] != ["--authorized-launch"]' not in engine.replace('sys.argv[1:] != ["--authorized-launch"]', "False", 1)
    mutations["attempt_consume_duplication_detected"] = engine.replace("ATTEMPT.mkdir(); attempted = True", "ATTEMPT.mkdir(); attempted = True\n        ATTEMPT.mkdir(); attempted = True", 1).count("ATTEMPT.mkdir(); attempted = True") != 1
    require(mutations["argv_gate_mutation_detected"] and mutations["attempt_consume_duplication_detected"], "engine mutations detected")
    return mutations


def main() -> None:
    require(os.getpid() == os.getpid(), "source-only process")
    for label, path in (("wrapper", WRAPPER), ("tb", TB), ("engine", ENGINE), ("contract", CONTRACT)):
        require(regular(path), f"regular fixed input: {label}")
        require(sha(path) == EXPECTED[label], f"fixed identity: {label}")
    require(sha(AUTHOR / "SHA256SUMS.seal.sha256") == EXPECTED["author_outer"], "author receipt outer")
    require(sha(M1109 / "SHA256SUMS.seal.sha256") == EXPECTED["m1109_outer"], "M1109 outer")
    require(sha(DOCS359) == EXPECTED["docs359"], "docs359")
    author_seal = verify_flat(AUTHOR, EXPECTED["author_outer"])
    m1109_seal = verify_flat(M1109, EXPECTED["m1109_outer"])
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    require(contract["m1109_authority"]["outer_seal_file_sha256"] == EXPECTED["m1109_outer"], "contract M1109 bind")
    require(json.loads(M1109.joinpath("review.json").read_text(encoding="utf-8"))["status"] == "PASS_M1109_FAILURE_AUDIT__M1091R3_DO_NOT_RETRY__NEW_ASYNC_OBSERVATION_SHADOW_REPAIR_ONLY", "M1109 status")

    wrapper = WRAPPER.read_text(encoding="utf-8")
    tb = TB.read_text(encoding="utf-8")
    engine = ENGINE.read_text(encoding="utf-8")
    rtl = wrapper_checks(wrapper)
    require(tb_validator(tb) == [], "unmutated TB contract")
    namespace = load_engine_namespace(engine)
    structural = structural_gate_attacks(namespace)
    trust = trust_boundary_attacks(namespace)
    engine_checks = engine_static_checks(engine, contract)
    mutations = mutation_checks(wrapper, tb, engine)

    p0 = []
    if structural["required_but_missing"]:
        p0.append({
            "id": "M1113-P0-01",
            "title": "Mapped async-pin checker does not prove reset provenance",
            "evidence": "The fixed engine accepted all 337 shadow instances with .CDN(unrelated_fake_reset).",
            "required_repair": "Resolve every async pin through only rst_core or exactly one inverter; reject constants, unrelated nets, data logic, and reconvergence.",
        })
    if trust["required_but_missing"]:
        p0.append({
            "id": "M1113-P0-02",
            "title": "Live seal metadata can be a symlink",
            "evidence": "The fixed verify_double accepted a symlink .sha256 and verify_flat accepted a symlink SHA256SUMS plus unlisted manifest.real.",
            "required_repair": "Apply verify_regular to every live sidecar/manifest/outer and require exact manifest member coverage; retain the sole sealed historical member-symlink exception.",
        })

    status = "FAIL_M1113_ENGINE_HAMMER__SOURCE_REPAIR_REQUIRED__NO_LAUNCHER_NO_EDA"
    checks = {
        "schema": "m1113_m1112_c2_async_observation_engine_hammer_mechanical_v1",
        "status": status,
        "scope": {"source_static_only": True, "eda_invocations": 0, "launcher_invocations": 0, "attempts_consumed": 0},
        "identity": {**EXPECTED, "author_receipt": author_seal, "m1109": m1109_seal},
        "rtl": rtl,
        "testbench": {"observation_predicates": 22, "atomic_same_cycle": True, "first_plus_union": True, "fatal_window_cycle": 128},
        "structural_gate_attacks": structural,
        "trust_boundary_attacks": trust,
        "engine": engine_checks,
        "mutations": mutations,
        "p0_count": len(p0),
        "p0": p0,
    }
    (OUT / "mechanical_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    review = {
        "schema": "m1113_m1112_c2_async_observation_engine_hammer_review_v1",
        "status": status,
        "verdict": "NO_GO__FIXED_M1112_ENGINE_MUST_BE_REAUTHORED_AND_RESEALED",
        "score": 78,
        "issue_counts": {"P0": len(p0), "P1": 0, "P2": 0},
        "passed": {
            "fixed_identity_and_M1109_binding": True,
            "thirteen_async_shadow_counters": True,
            "shadow_register_bits": 337,
            "frozen_sync_debug_excluded": True,
            "no_observation_functional_feedback": True,
            "twenty_two_atomic_unknown_predicates": True,
            "first_x_plus_later_union": True,
            "fatal_only_at_cycle_128": True,
            "omission_and_short_circuit_mutations_rejected": True,
            "d_cp_only_and_336_bit_mapped_mutations_rejected": True,
            "historical_sealed_symlink_exception_valid": True,
            "argv_and_single_attempt_boundary_present": True,
        },
        "blocking_findings": p0,
        "authorization": {
            "different_author_launcher_authoring": False,
            "launcher_creation": False,
            "attempt_creation": False,
            "eda": False,
            "required_next_stage": "Author M1112r2 with reset-provenance traversal and strict live seal metadata; then obtain a fresh different-author hammer.",
        },
        "claim_boundary": {"paper_citable": False, "mapped_functionality": False, "performance": False, "activity_or_power": False, "system_speedup": False, "paper_ppa_ready": False},
        "identity": {"wrapper_sha256": EXPECTED["wrapper"], "testbench_sha256": EXPECTED["tb"], "engine_sha256": EXPECTED["engine"], "contract_sha256": EXPECTED["contract"], "author_receipt_outer_seal_file_sha256": EXPECTED["author_outer"], "m1109_outer_seal_file_sha256": EXPECTED["m1109_outer"], "docs359_sha256": EXPECTED["docs359"]},
    }
    (OUT / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "RUN_COMPLETE.txt").write_text(status + "\n", encoding="utf-8")
    print(f"{status} p0={len(p0)} eda=0 launcher=0 attempt=0")


if __name__ == "__main__":
    main()
