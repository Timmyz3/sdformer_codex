#!/usr/bin/env python3
"""Fail-closed M1210/R8 source gate for random request-ready quiesce. No EDA."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HW = Path(__file__).resolve().parents[1]
TB = HW / "verif_m1210r8_c1_common_charge_protocol/tb_m1210r8_m1162_common_charge_protocol_unit_delay_r8.sv"
SVA = HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1210r8_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
WRAPPER = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
M1201 = HW / "reviews/m1201_m1198_c1_r7_source_gate_repair_hammer_r1_20260830/hammer_m1201.py"
M1207_ATTEMPT = HW / "results/.m1207_m1198r7_m1162_c1_common_charge_protocol_vcs_r7_attempt_consumed/identity.txt"
M1207_FAIL = HW / "results/m1207_m1198r7_m1162_c1_common_charge_protocol_unit_delay_vcs_r7_20260830.failed_or_incomplete.60228.quarantine"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    TB: "060ec9d5ae6085a0dd013160d22f63e21615730384ddaef342eb3fa77e17947b",
    FILELIST: "048253d22301df9fb84502ff35f5129459a5b43e4ff9e8d11ea62973f7047af6",
    SVA: "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    WRAPPER: "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    M935: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    M1201: "b59fcd476cfddda527dbd29fc06aad26e6699c1e5004cdc2f682c63939fb4113",
    M1207_ATTEMPT: "fada26ddb698b27453444b6a8b8718335673d5ff43c3d6aa28c149a517e11b28",
    M1207_FAIL / "RUN_FAILED_OR_INCOMPLETE.txt": "d128ff1f5cb6e1514ee4ee25e746cd96c2889dc4efaad0132bb5512dcf740012",
    M1207_FAIL / "compile.log": "1052919cfa448afed586f6fd90be61fa303440c648fd95ba43a60e0bab9c3411",
    M1207_FAIL / "sim.log": "4a758035d25c79b1af435a7499eccb6efdf976ab6a17bb78840001bf255cd765",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

checks = 0
mutations = 0


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise AssertionError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_m1201():
    spec = importlib.util.spec_from_file_location("m1201_independent", M1201)
    require(spec is not None and spec.loader is not None, "load M1201 validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M1201_MODULE = load_m1201()


def task_body(tb: str, name: str) -> str:
    found = M1201_MODULE.tasks(tb)
    require(name in found, "task present " + name)
    return M1201_MODULE.strip(found[name])


def validate(tb: str, sva: str) -> None:
    M1201_MODULE.independent_validate(tb, sva)
    require("module tb_m1210r8_m1162_common_charge_protocol_unit_delay_r8" in tb,
            "R8 module identity")
    require("M1207 compiled, elaborated, and linked" in tb, "R7 failure provenance")
    require("R8_RANDOM_REQUEST_READY_QUIESCE_BOUNDARY" in tb, "quiesce marker")
    random = task_body(tb, "random_legal_transaction")

    # The intended request fires must be observed before quiesce, and both
    # request-ready inputs must be low before the first response-valid branch.
    order = re.compile(
        r"wait\s*\(weight_fire_count\s*==\s*w0\s*\+\s*1\s*\)\s*;"
        r"\s*if\s*\(first\)\s*wait\s*\(psum_fire_count\s*==\s*p0\s*\+\s*1\s*\)\s*;"
        r"\s*@\s*\(negedge\s+clk_core\)\s*;"
        r"\s*weight_req_ready\s*=\s*1'b0\s*;"
        r"\s*psum_req_ready\s*=\s*1'b0\s*;"
        r"\s*random_request_window_active\s*=\s*1'b0\s*;"
        r".*?if\s*\(index\[0\]\)", re.S)
    require(len(order.findall(random)) == 1, "single pre-response request quiesce")
    require(random.count("weight_req_ready = 1'b0;") == 2,
            "initial stall plus one weight quiesce")
    require(random.count("psum_req_ready = 1'b0;") == 2,
            "initial stall plus one psum quiesce")
    require(random.count("random_request_window_active = 1'b1;") == 1,
            "single random window enable")
    require(random.count("random_request_window_active = 1'b0;") == 1,
            "single random window retire")
    require(random.count("random_weight_request_handshakes != 1") == 2,
            "initial and terminal exact weight handshake oracles")
    require(random.count("random_psum_request_handshakes != first") == 2,
            "initial and terminal exact psum handshake oracles")
    require("force dut.core_issue_data_ready = 1'b0;" in random
            and "repeat (hold_cycles) @(posedge clk_core);" in random
            and "force dut.core_issue_data_ready = 1'b1;" in random,
            "core-ready response stall preserved")

    # Counters are driven only inside the explicit legal random request window.
    require(tb.count("random_weight_request_handshakes =\n"
                     "                random_weight_request_handshakes + 1;") == 1,
            "one weight counter increment")
    require(tb.count("random_psum_request_handshakes =\n"
                     "                random_psum_request_handshakes + 1;") == 1,
            "one psum counter increment")
    require(tb.count("reset_n && random_request_window_active") == 2,
            "both counter increments window-gated")
    require("random_request_quiesce=24" in tb
            and "exactly_one_random_request_handshake=1" in tb,
            "PASS evidence tokens")
    require(sva.count("assert property") == 16, "16 frozen assertions")
    require(sva.count("cover property") == 6, "6 frozen covers")


def reject_mutation(tb: str, sva: str, old: str, new: str, label: str) -> None:
    global mutations
    require(old in tb, "mutation anchor " + label)
    changed = tb.replace(old, new, 1)
    try:
        validate(changed, sva)
    except AssertionError:
        mutations += 1
        return
    raise AssertionError("mutation accepted " + label)


def main() -> None:
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "identity drift " + str(path))
    require("random transaction 1 duplicated request" in
            (M1207_FAIL / "sim.log").read_text(), "exact M1207 failure")
    require("Compiler version V-2023.12-SP1_Full64" in
            (M1207_FAIL / "sim.log").read_text(), "M1207 VCS runtime provenance")
    require("Error-[" not in (M1207_FAIL / "compile.log").read_text(),
            "M1207 compile/elab/link passed")
    lines = [line.strip() for line in FILELIST.read_text().splitlines() if line.strip()]
    require(len(lines) == 6 and len(set(lines)) == 6, "R8 filelist cardinality")
    require(lines[-1] == str(TB), "R8 filelist exact TB tail")
    for member in lines:
        require(Path(member).is_file() and not Path(member).is_symlink(),
                "R8 filelist member " + member)
    tb, sva = TB.read_text(), SVA.read_text()
    validate(tb, sva)

    quiesce = ("            weight_req_ready = 1'b0;\n"
               "            psum_req_ready = 1'b0;\n"
               "            random_request_window_active = 1'b0;")
    reject_mutation(tb, sva, quiesce,
        "            weight_req_ready = 1'b1;\n"
        "            psum_req_ready = 1'b1;\n"
        "            random_request_window_active = 1'b0;",
        "remove_both_ready_quiesce")
    reject_mutation(tb, sva, "            weight_req_ready = 1'b0;\n"
        "            psum_req_ready = 1'b0;\n"
        "            random_request_window_active = 1'b0;\n"
        "            if (random_weight_request_handshakes != 1",
        "            random_request_window_active = 1'b0;\n"
        "            if (random_weight_request_handshakes != 1",
        "remove_quiesce_statements")
    reject_mutation(tb, sva, "random_weight_request_handshakes != 1",
                    "random_weight_request_handshakes < 1", "relax_weight_oracle")
    reject_mutation(tb, sva, "random_psum_request_handshakes != first",
                    "random_psum_request_handshakes < first", "relax_psum_oracle")
    reject_mutation(tb, sva, "random_request_window_active = 1'b0;",
                    "random_request_window_active = 1'b1;", "window_not_retired")
    reject_mutation(tb, sva,
        "            force dut.core_issue_data_ready = 1'b0;\n"
        "            repeat (hold_cycles) @(posedge clk_core);\n"
        "            force dut.core_issue_data_ready = 1'b1;",
        "            force dut.core_issue_data_ready = 1'b1;\n"
        "            repeat (hold_cycles) @(posedge clk_core);\n"
        "            force dut.core_issue_data_ready = 1'b1;",
        "remove_core_stall")
    require(mutations == 6, "all six R8 mutations rejected")
    print(json.dumps({
        "schema": "m1210r8_m1162_c1_random_request_quiesce_source_static_check_v1",
        "status": "PASS_R8_SOURCE_ONLY__M1207_RANDOM_DUPLICATE_TESTBENCH_RACE_CLOSED__FRESH_DIFFERENT_AUTHOR_HAMMER_REQUIRED__NO_EDA",
        "checks_passed": checks,
        "mutations_rejected": mutations,
        "m1207_compile_elab_link_passed": True,
        "m1207_sim_failure_reproduced": "random transaction 1 duplicated request",
        "request_ready_quiesced_before_responses": True,
        "random_request_handshakes_exactly_one": True,
        "core_ready_stall_preserved": True,
        "assertions": 16,
        "covers": 6,
        "protocol_attacks": 7,
        "service_assumption_attacks": 2,
        "random_legal_transactions": 24,
        "normal_m935_rows": 1,
        "normal_m935_tasks": 1,
        "ii": 2,
        "rtl_modified": False,
        "sva_modified": False,
        "vcs_runs": 0,
        "simv_runs": 0,
        "all_eda_runs": 0,
        "docs359_sha256": sha(DOCS359),
        "source_sha256": {str(path.relative_to(HW)): sha(path)
                          for path in (TB, SVA, FILELIST, WRAPPER, M935)},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
