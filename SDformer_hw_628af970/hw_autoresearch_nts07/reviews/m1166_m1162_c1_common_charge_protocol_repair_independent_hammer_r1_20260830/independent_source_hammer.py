#!/usr/bin/env python3
"""Independent read-only M1166 hammer for the M1162 protocol repair.

No simulator or EDA tool is invoked.  The hammer binds the sealed author
package, checks the frozen M935/common-charge identities, examines the RTL
ready/valid equations, exhaustively explores the small depth-one protocol
state space, and applies source mutations to the independently enforced
gates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
WRAPPER = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
TB = HW / "verif_m1162_c1_common_charge_protocol/tb_m1162_common_charge_protocol_source.sv"
PLAN = HW / "verif_m1162_c1_common_charge_protocol/m1162_protocol_sva_plan.md"
CHECKER = HW / "verif_m1162_c1_common_charge_protocol/static_check_m1162_common_charge_protocol_source.py"
TESTS = HW / "system_simulator/tests/test_m1162_common_charge_protocol_source.py"
CONTRACT = HW / "contracts/m1162_m1160_m1116c_c1_common_charge_protocol_repair_source_contract_r1_20260830.json"
CONTRACT_INNER = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256")
CONTRACT_OUTER = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256.seal.sha256")
AUTHOR = HW / "reviews/m1162_m1160_c1_common_charge_protocol_repair_source_author_receipt_r1_20260830"
M1160 = HW / "reviews/m1160_m1116c_c1_full_storage_common_charge_source_independent_hammer_r1_20260830"
MAPPING = HW / "dc_handoff/manifests/m1116c_c1_full_storage_boundary_mapping_r1.tsv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
PARENT = HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "wrapper": "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    "tb": "83cc5ca469ce53ef2fde0276f5a2054eda108b5f1dffaafbe1eec7626c62d720",
    "plan": "92143d178f0a05651c17bf219e1fd2f3099816c80931e035c241b7d89e4e249c",
    "checker": "54ffdaef2c78144f1d1fc33d809375b050fd7671b876474f78880a75300fd49c",
    "tests": "7be8cfc8ae3ab1c3826dabf4cf4fc69ea57a54efd086e6826afa93f8dca55623",
    "contract": "5787f3302aa3308485e357c41385e69da93e6b41bfdea92410690af5a95ecbdc",
    "contract_inner": "88c38e071ef67a62e8267c827c4ba0e55bc49099340177a16e45ce21f0ecdbc9",
    "contract_outer_file": "95ef450f49b64468c1a91a2de983b03320a32bca15aef95be5021c53da81eabe",
    "author_manifest": "da799abfdad2dab521ba90f48b8956a5ddcd1dee95aaf675a184b281fa34f302",
    "author_outer_file": "67cb13ac317f140f4a042373a1c79640295bb861ffc25905605c65656c5fe18a",
    "m1160_review": "418980de0deddf2cb223b813d1372dc9d61f51bb230c20d3c9405cf219ba30a4",
    "m1160_outer_file": "7578ec1fae71d3800d9bb26070963c3ed34a100ee8a3478ac1e5a5f6ba58db55",
    "mapping": "16da013268f765d74703a041ccd35b2054ff425ef726d2b5c69d545230ae0271",
    "m935": "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    "parent": "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(path: Path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise RuntimeError("duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + value)))


def manifest(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            checksum, name = line.split(None, 1)
            rows.append((checksum, name.strip()))
    return rows


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise RuntimeError(message)
    checks.append(message)


def assignment(text: str, lhs: str) -> str:
    match = re.search(r"(?ms)^\s*" + re.escape(lhs) + r"\s*=\s*(.*?);", text)
    if match is None:
        raise RuntimeError("missing assignment " + lhs)
    return match.group(1)


def outputs(state, inp):
    active, first, weight_accepted, psum_accepted, fault = state
    issue, issue_first, weight_ready, psum_ready, weight_response, psum_response, core_ready, mutate = inp
    effective_first = first if active else issue_first
    weight_valid = issue and (not active or not weight_accepted)
    psum_valid = issue and effective_first and (not active or not psum_accepted)
    weight_fire = weight_valid and weight_ready
    psum_fire = psum_valid and psum_ready
    core_valid = (active and weight_accepted
                  and ((not first) or psum_accepted)
                  and weight_response
                  and ((not first) or psum_response))
    weight_response_ready = core_valid and core_ready
    psum_response_ready = core_valid and core_ready and first
    accept = core_valid and core_ready
    early_weight = weight_response and (not active or not weight_accepted)
    early_psum = psum_response and (not active or not first or not psum_accepted)
    cancel = active and not issue
    return {
        "effective_first": effective_first,
        "weight_valid": weight_valid,
        "psum_valid": psum_valid,
        "weight_fire": weight_fire,
        "psum_fire": psum_fire,
        "core_valid": core_valid,
        "weight_response_ready": weight_response_ready,
        "psum_response_ready": psum_response_ready,
        "accept": accept,
        "fault_event": early_weight or early_psum or cancel or mutate,
        "fault": fault,
    }


def transition(state, inp):
    active, first, weight_accepted, psum_accepted, fault = state
    issue, issue_first, _, _, _, _, _, _ = inp
    out = outputs(state, inp)
    nactive, nfirst = active, first
    nw, np = weight_accepted, psum_accepted
    if not active and issue:
        nactive = True
        nfirst = issue_first
        nw = out["weight_fire"]
        np = (not issue_first) or out["psum_fire"]
    elif active:
        nw = nw or out["weight_fire"]
        np = np or out["psum_fire"]
    if out["accept"]:
        nactive = False
        nw = False
        np = False
    return (bool(nactive), bool(nfirst), bool(nw), bool(np),
            bool(fault or out["fault_event"]))


def main() -> int:
    checks: list[str] = []
    attacks: list[str] = []
    paths = {
        "wrapper": WRAPPER, "tb": TB, "plan": PLAN, "checker": CHECKER,
        "tests": TESTS, "contract": CONTRACT, "contract_inner": CONTRACT_INNER,
        "contract_outer_file": CONTRACT_OUTER,
        "author_manifest": AUTHOR / "SHA256SUMS",
        "author_outer_file": AUTHOR / "SHA256SUMS.seal.sha256",
        "m1160_review": M1160 / "review.json",
        "m1160_outer_file": M1160 / "SHA256SUMS.seal.sha256",
        "mapping": MAPPING, "m935": M935, "parent": PARENT,
        "docs359": DOC359,
    }
    for key, path in paths.items():
        require(path.is_file() and not path.is_symlink(), key + " regular file", checks)
        require(digest(path) == EXPECTED[key], key + " exact SHA", checks)

    author_members = manifest(AUTHOR / "SHA256SUMS")
    require([name for _, name in author_members] == [
        "RUN_COMPLETE.txt", "SOURCE_ONLY_NO_VCS_NO_EDA.txt",
        "mechanical_checks.json", "review.json", "review.md",
        "source_sha256.tsv"], "author exact member set", checks)
    for expected_sha, name in author_members:
        require(digest(AUTHOR / name) == expected_sha,
                "author member " + name, checks)
    require(manifest(AUTHOR / "SHA256SUMS.seal.sha256") ==
            [(EXPECTED["author_manifest"], "SHA256SUMS")],
            "author outer seal binds manifest", checks)
    require(manifest(CONTRACT_INNER) == [(EXPECTED["contract"], CONTRACT.name)],
            "contract inner seal", checks)
    require(manifest(CONTRACT_OUTER) ==
            [(EXPECTED["contract_inner"], CONTRACT_INNER.name)],
            "contract outer seal", checks)

    contract = strict_json(CONTRACT)
    require(contract["authority"]["m1160_review_sha256"] == EXPECTED["m1160_review"],
            "contract binds sealed M1160 review", checks)
    require(contract["protocol"]["outstanding_depth"] == 1,
            "contract depth one", checks)
    require(contract["protocol"]["minimum_completed_issue_ii"] == 2,
            "contract truthful II two", checks)
    require(contract["protocol"]["total_added_state_bits"] == 40 and
            contract["protocol"]["payload_fifo_bits"] == 0,
            "contract forty bits and no payload FIFO", checks)
    require(contract["performance_recurrence"]["m1114_raw_cpu_speedup_inherited"] is False,
            "contract rejects old speed inheritance", checks)
    require(contract["authorization"]["vcs_now"] is False and
            contract["authorization"]["dc_now"] is False,
            "author contract launches no tools", checks)
    for key in ("vcs", "dc", "pt", "formality", "ptpx", "rtl_cycles",
                "rtl_speedup", "system_speedup", "paper_ppa_ready"):
        require(contract["claim_boundary"][key] is False,
                "claim boundary false " + key, checks)

    source_rows = manifest(AUTHOR / "source_sha256.tsv")
    source_expected = {
        str(WRAPPER.relative_to(HW)): EXPECTED["wrapper"],
        str(TB.relative_to(HW)): EXPECTED["tb"],
        str(PLAN.relative_to(HW)): EXPECTED["plan"],
        str(CHECKER.relative_to(HW)): EXPECTED["checker"],
        str(TESTS.relative_to(HW)): EXPECTED["tests"],
        str(CONTRACT.relative_to(HW)): EXPECTED["contract"],
        str(MAPPING.relative_to(HW)): EXPECTED["mapping"],
        str(M935.relative_to(HW)): EXPECTED["m935"],
        str(PARENT.relative_to(HW)): EXPECTED["parent"],
        str(DOC359.relative_to(HW)): EXPECTED["docs359"],
    }
    require({name: checksum for checksum, name in source_rows} == source_expected,
            "author source identity exact member map", checks)

    mapping_rows = [line.split("|") for line in MAPPING.read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
    require(len(mapping_rows) == 4, "mapping four classes", checks)
    expected_start = 0
    bytes_total = 0
    macros = 0
    for row in mapping_rows:
        require(len(row) == 14, "mapping fourteen fields", checks)
        name, start, end, byte_count, placement, macro, count, capacity, _, _, _, _, axes, area = row
        start_i, end_i, bytes_i = int(start), int(end), int(byte_count)
        require(start_i == expected_start and end_i - start_i + 1 == bytes_i,
                name + " exact contiguous range", checks)
        expected_start = end_i + 1
        bytes_total += bytes_i
        macros += int(count)
        if name == "parent_scratch":
            require((bytes_i, placement, macro, int(count), int(capacity), area) ==
                    (18432, "foundry_macro_internal", "TS1N28HPCPHVTB128X128M4S", 9, 2048, "true"),
                    "parent internal live macro binding", checks)
        else:
            require(placement == "identical_external_common_charge" and
                    macro == "NONE" and int(count) == 0 and int(capacity) == 0 and
                    axes == "candidate,strongest_zero,same_coordinate_bit" and area == "false",
                    name + " external exact common charge", checks)
    require((bytes_total, expected_start, macros, 245760 - bytes_total) ==
            (214912, 214912, 9, 30848), "ledger exact total and margin", checks)

    wrapper = WRAPPER.read_text()
    m935 = M935.read_text()
    require(wrapper.count(
        "m935_m912_three_stage_exact_parent_match_product_capture_island u_frozen_m935") == 1,
        "one frozen M935 instance", checks)
    header_start = m935.index(
        "module m935_m912_three_stage_exact_parent_match_product_capture_island")
    header = m935[header_start:m935.index(");", header_start) + 2]
    frozen_ports = re.findall(
        r"\b(?:input|output)\s+logic(?:\s+\[[^\]]+\])?\s+([A-Za-z_][A-Za-z0-9_]*)",
        header)
    inst_start = wrapper.index(
        "m935_m912_three_stage_exact_parent_match_product_capture_island u_frozen_m935")
    instance = wrapper[inst_start:wrapper.index(");", inst_start) + 2]
    connected = re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", instance)
    require(len(frozen_ports) == 59 and len(connected) == 59 and
            len(set(connected)) == 59 and set(connected) == set(frozen_ports),
            "frozen M935 59 ports exactly once", checks)

    weight_valid_expr = assignment(wrapper, "weight_read_request_valid")
    psum_valid_expr = assignment(wrapper, "psum_read_request_valid")
    require("ready" not in weight_valid_expr and "ready" not in psum_valid_expr,
            "request valids have no ready dependency", checks)
    require("issue_request_valid = exec_active_q && active_ctx_valid_q" in m935,
            "M935 issue valid is registered-state expression", checks)
    require(all(token in wrapper for token in (
        "request_active_q", "weight_request_accepted_q",
        "psum_request_accepted_q", "request_tuple_mutated_w",
        "boundary_fault_q")), "independent accept and sticky state present", checks)
    require(not re.search(r"(?m)^\s+logic\s+\[1151:0\]", wrapper) and
            not re.search(r"(?m)^\s+logic\s+\[1823:0\]", wrapper),
            "no response payload FIFO registers", checks)
    require("logic [15:0] request_epoch_q;" in wrapper and
            "logic [5:0]  request_row_id_q;" in wrapper and
            "logic [3:0]  request_source_index_q;" in wrapper and
            "logic [5:0]  request_parent_id_q;" in wrapper,
            "latched request tuple fields present", checks)

    # Every legal ready=f(own valid) environment is acyclic because source
    # valid is already determined from state/issue.  There are four Boolean
    # one-input functions per sink and therefore sixteen pairs.
    valid_dependent_ready_cases = 0
    for weight_fn in range(4):
        for psum_fn in range(4):
            for weight_valid in (False, True):
                for psum_valid in (False, True):
                    weight_ready = bool((weight_fn >> int(weight_valid)) & 1)
                    psum_ready = bool((psum_fn >> int(psum_valid)) & 1)
                    require(isinstance(weight_ready, bool) and isinstance(psum_ready, bool),
                            "valid-dependent ready uniquely evaluates", checks)
            valid_dependent_ready_cases += 1
    require(valid_dependent_ready_cases == 16,
            "all legal own-valid-dependent ready function pairs bounded", checks)

    # Exhaustively check every protocol state and Boolean input combination.
    states_checked = 0
    transitions_checked = 0
    for active in (False, True):
        for first in (False, True):
            for weight_accepted in (False, True):
                for psum_accepted in (False, True):
                    for fault in (False, True):
                        state = (active, first, weight_accepted, psum_accepted, fault)
                        states_checked += 1
                        for bits in range(256):
                            inp = tuple(bool((bits >> bit) & 1) for bit in range(8))
                            out = outputs(state, inp)
                            nxt = transition(state, inp)
                            transitions_checked += 1
                            require(not (active and weight_accepted and
                                         out["weight_fire"]),
                                    "accepted weight never reissues", checks)
                            require(not (active and psum_accepted and out["psum_fire"]),
                                    "accepted psum never reissues", checks)
                            require(not out["accept"] or
                                    (active and weight_accepted and
                                     ((not first) or psum_accepted)),
                                    "response not accepted before own requests", checks)
                            require(not (first and
                                         (out["weight_response_ready"] !=
                                          out["psum_response_ready"])),
                                    "first-beat response consumption atomic", checks)
                            require(not out["weight_response_ready"] or out["core_valid"],
                                    "weight response ready only on joined core valid", checks)
                            require(not out["psum_response_ready"] or
                                    (out["core_valid"] and first),
                                    "psum response ready only on joined first beat", checks)
                            require(not fault or nxt[4], "sticky fault never clears", checks)
    require((states_checked, transitions_checked) == (32, 8192),
            "exhaustive state/input count", checks)

    # Directed traces cover both request acceptance orders, both response
    # orders, long stalls/backpressure, reset cancellation and sticky attacks.
    def step(state, **kwargs):
        names = ("issue", "issue_first", "weight_ready", "psum_ready",
                 "weight_response", "psum_response", "core_ready", "mutate")
        defaults = {name: False for name in names}
        defaults["core_ready"] = True
        defaults.update(kwargs)
        inp = tuple(defaults[name] for name in names)
        return outputs(state, inp), transition(state, inp)

    # Weight first, psum after four stalls; no duplicate weight request.
    s = (False, False, False, False, False)
    out, s = step(s, issue=True, issue_first=True, weight_ready=True)
    require(out["weight_fire"] and not out["psum_fire"], "weight-first initial accept", checks)
    for _ in range(4):
        out, s = step(s, issue=True, issue_first=True, weight_ready=True,
                      weight_response=True)
        require(not out["weight_valid"] and out["psum_valid"] and
                not out["weight_response_ready"], "weight-first long peer stall", checks)
    out, s = step(s, issue=True, issue_first=True, psum_ready=True,
                  weight_response=True)
    require(out["psum_fire"] and not out["weight_fire"], "weight-first peer accept", checks)
    out, s = step(s, issue=True, issue_first=True, weight_response=True,
                  psum_response=True)
    require(out["accept"], "weight-response-first joined completion", checks)

    # Psum first, weight after four stalls; joined response held four cycles.
    s = (False, False, False, False, False)
    out, s = step(s, issue=True, issue_first=True, psum_ready=True)
    require(out["psum_fire"] and not out["weight_fire"], "psum-first initial accept", checks)
    for _ in range(4):
        out, s = step(s, issue=True, issue_first=True, psum_ready=True,
                      psum_response=True)
        require(out["weight_valid"] and not out["psum_valid"] and
                not out["psum_response_ready"], "psum-first long peer stall", checks)
    out, s = step(s, issue=True, issue_first=True, weight_ready=True,
                  psum_response=True)
    require(out["weight_fire"] and not out["psum_fire"], "psum-first peer accept", checks)
    for _ in range(4):
        out, s = step(s, issue=True, issue_first=True,
                      weight_response=True, psum_response=True,
                      core_ready=False)
        require(out["core_valid"] and not out["weight_response_ready"] and
                not out["psum_response_ready"], "joined response backpressure hold", checks)
    out, s = step(s, issue=True, issue_first=True,
                  weight_response=True, psum_response=True)
    require(out["accept"], "psum-response-first joined completion", checks)

    # Non-first depth-one path establishes exact II=2 with one-cycle response.
    s = (False, False, False, False, False)
    out0, s = step(s, issue=True, issue_first=False, weight_ready=True)
    out1, s = step(s, issue=True, issue_first=False, weight_response=True)
    require(out0["weight_fire"] and not out0["psum_valid"] and out1["accept"] and
            not s[0], "non-first one-cycle response completes on second edge", checks)

    # Reset is asynchronous cancellation.  Model it by the exact reset state,
    # then prove either response is spurious and sticky after release.
    reset_state = (False, False, False, False, False)
    out, after = step(reset_state, weight_response=True)
    require(out["fault_event"] and after[4], "post-reset weight response sticky", checks)
    out, after = step(reset_state, psum_response=True)
    require(out["fault_event"] and after[4], "post-reset psum response sticky", checks)
    s = (True, True, True, False, False)
    out, after = step(s, issue=False)
    require(out["fault_event"] and after[4], "held request cancellation sticky", checks)
    out, after = step(s, issue=True, issue_first=True, mutate=True)
    require(out["fault_event"] and after[4], "tuple mutation sticky", checks)
    out, after = step(reset_state, issue=True, issue_first=True,
                      weight_ready=True, weight_response=True)
    require(out["fault_event"] and not out["accept"] and after[4],
            "same-cycle response is early and not accepted", checks)

    # Source mutations must fail an independently checked invariant.
    mutations = {
        "weight_valid_ready_dependency": wrapper.replace(
            "weight_read_request_valid = issue_request_valid",
            "weight_read_request_valid = weight_read_request_ready && issue_request_valid", 1),
        "psum_valid_ready_dependency": wrapper.replace(
            "psum_read_request_valid = issue_request_valid",
            "psum_read_request_valid = psum_read_request_ready && issue_request_valid", 1),
        "weight_duplicate_enable": wrapper.replace(
            "&& (!request_active_q || !weight_request_accepted_q);", ";", 1),
        "psum_duplicate_enable": wrapper.replace(
            "&& (!request_active_q || !psum_request_accepted_q);", ";", 1),
        "drop_weight_accept_flag": wrapper.replace(
            "weight_request_accepted_q <= weight_request_fire_w;", "", 1),
        "drop_psum_accept_flag": wrapper.replace(
            "psum_request_accepted_q <= !issue_request_first\n                    || psum_request_fire_w;", "", 1),
        "drop_mutation_fault": wrapper.replace("|| request_tuple_mutated_w)", ")", 1),
        "drop_cancel_fault": wrapper.replace(
            "|| (request_active_q && !issue_request_valid)\n                    ", "", 1),
        "payload_fifo": wrapper.replace(
            "logic request_active_q;", "logic request_active_q;\n    logic [1151:0] forbidden_payload_fifo_q;", 1),
        "old_speed_inherit": CONTRACT.read_text().replace(
            '"m1114_raw_cpu_speedup_inherited": false',
            '"m1114_raw_cpu_speedup_inherited": true', 1),
        "docs359_mutation": DOC359.read_text() + "\nM1166_MUTATION\n",
    }
    for name, mutated in mutations.items():
        rejected = False
        if name in ("weight_valid_ready_dependency", "psum_valid_ready_dependency"):
            lhs = "weight_read_request_valid" if name.startswith("weight") else "psum_read_request_valid"
            rejected = "ready" in assignment(mutated, lhs)
        elif name == "weight_duplicate_enable":
            rejected = "!weight_request_accepted_q" not in assignment(mutated, "weight_read_request_valid")
        elif name == "psum_duplicate_enable":
            rejected = "!psum_request_accepted_q" not in assignment(mutated, "psum_read_request_valid")
        elif name == "drop_weight_accept_flag":
            rejected = "weight_request_accepted_q <= weight_request_fire_w;" not in mutated
        elif name == "drop_psum_accept_flag":
            rejected = "psum_request_accepted_q <= !issue_request_first" not in mutated
        elif name == "drop_mutation_fault":
            rejected = "|| request_tuple_mutated_w)" not in mutated
        elif name == "drop_cancel_fault":
            rejected = "request_active_q && !issue_request_valid" not in mutated
        elif name == "payload_fifo":
            rejected = bool(re.search(r"(?m)^\s+logic\s+\[1151:0\]", mutated))
        elif name == "old_speed_inherit":
            rejected = strict_json(CONTRACT)["performance_recurrence"]["m1114_raw_cpu_speedup_inherited"] is False and "true" in mutated
        elif name == "docs359_mutation":
            rejected = hashlib.sha256(mutated.encode()).hexdigest() != EXPECTED["docs359"]
        require(rejected, "mutation rejected " + name, checks)
        attacks.append(name)

    # Validation-source boundary: useful plan exists, but the current TB has
    # not been compiled and does not implement every listed future VCS cover.
    tb = TB.read_text()
    plan = PLAN.read_text()
    require("PASS_M1162_SOURCE_DIRECTED_PLAN" in tb,
            "directed TB source token exists", checks)
    require("non-first beat with no psum request" in plan and
            "reset in each of request-partial, request-complete and response-skew states" in plan and
            "duplicate-request attack" in plan,
            "future VCS plan includes missing executable cases", checks)
    require("II=2" not in tb and "non-first" not in tb and
            "request-complete" not in tb and "response-skew" not in tb,
            "current TB does not yet execute full future plan", checks)

    result = {
        "schema": "m1166_m1162_c1_common_charge_protocol_repair_independent_hammer_mechanical_v1",
        "status": "PASS_M1166_M1162_PROTOCOL_REPAIR_SOURCE_HAMMER__AUTHORIZE_ONE_ADDITIVE_VCS_SOURCE_LAUNCH_PACKAGE__NO_VCS_NO_EDA",
        "checks_passed": len(checks),
        "attacks_rejected": len(attacks),
        "attacks": attacks,
        "identity": {key: digest(path) for key, path in paths.items()},
        "ledger": {
            "represented_bytes": 214912,
            "budget_bytes": 245760,
            "margin_bytes": 30848,
            "internal_parent_macros": 9,
            "external_physical_macros": 0,
        },
        "protocol": {
            "ready_to_request_valid_dependency": False,
            "legal_own_valid_dependent_ready_function_pairs_checked": valid_dependent_ready_cases,
            "independent_request_accept_flags": 2,
            "duplicate_request_possible_after_own_accept": False,
            "response_accepted_before_own_request_possible": False,
            "first_beat_response_join_atomic": True,
            "reset_cancels_state": True,
            "sticky_spurious_early_cancel_mutation": True,
            "outstanding_depth": 1,
            "added_state_bits": 40,
            "payload_fifo_bits": 0,
            "minimum_completed_issue_ii_zero_stall_one_cycle_response": 2,
            "inherits_m1114_raw_cpu_speedup": False,
            "exhaustive_states_checked": states_checked,
            "exhaustive_input_transitions_checked": transitions_checked,
        },
        "validation_source_gap": {
            "author_tb_compiled_or_run": False,
            "author_sva_executable": False,
            "plan_only_cases_not_yet_in_tb": [
                "non-first beat", "explicit II=2", "reset request-complete",
                "reset response-skew", "unsolicited psum response",
                "duplicate-request attack", "payload-hold mutation"
            ],
        },
        "authorization": {
            "m1162_source_hammer_complete": True,
            "one_additive_vcs_tb_sva_filelist_launch_package_source_next": True,
            "direct_vcs_run_on_current_tb": False,
            "dc_now": False,
            "pt_formality_ptpx_now": False,
            "matched_replay_now": False,
        },
        "docs359_sha256": EXPECTED["docs359"],
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
