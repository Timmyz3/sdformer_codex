#!/usr/bin/env python3
"""Fail-closed independent validator for the M64 hammer review.

This validator does not invoke any HDL simulator.  It verifies the sealed r1b
evidence, independently regenerates the deterministic campaign and elastic
schedule, and attacks claim/score fields in memory.
"""

from __future__ import print_function

import copy
import hashlib
import json
from pathlib import Path
import re
import sys


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
REVIEW = HERE / "m64_independent_hammer_review.json"
VALIDATION_RECEIPT = HERE / "m64_independent_hammer_validation_receipt.json"
RUN = HW / "dc_handoff/runs/m64_parent_selector_directed_vcs_r1b_20260823"
R1 = HW / "dc_handoff/runs/m64_parent_selector_directed_vcs_r1_20260823"

PATHS = {
    "rtl": HW / "rtl_m64/qfit_adaptive_parent_selector_p256.sv",
    "sva": HW / "verif_m64/qfit_adaptive_parent_selector_p256_assertions.sv",
    "tb": HW / "tb_m64/tb_qfit_adaptive_parent_selector_p256.sv",
    "contract": HW / "contracts/m64_online_adaptive_parent_selector_directed_vcs_contract_r1_20260823.json",
    "filelist": HW / "dc_handoff/filelists/date_m64_parent_selector_directed_vcs.f",
    "runner": HW / "dc_handoff/scripts/run_vcs_m64_parent_selector_directed_sva.sh",
    "builder": HW / "dc_handoff/scripts/build_m64_parent_selector_directed_vcs_receipt.py",
    "producer_validator": HW / "dc_handoff/scripts/validate_m64_parent_selector_directed_vcs.py",
    "r1b_receipt": RUN / "m64_directed_vcs_receipt_r1.json",
    "r1b_sim_log": RUN / "sim.raw.log",
    "r1b_output_manifest": RUN / "output.sha256",
    "r1b_simv": RUN / "simv",
    "m53_result": HW / "results/m53_adaptive_temporal_parent_k4_ctx16_dse_r1_20260823/m53_adaptive_temporal_parent_k4_ctx16_dse.json",
    "m57_s00_manifest": HW / "results/m57_h67_k4c16_temporal_vcs_r1_20260823/m57_s00_schedule_manifest.json",
    "m57_s00_replay": HW / "results/m57_h67_k4c16_temporal_vcs_s00_phase_safe_full_compact_r3_20260823/m57_s00_ledger_replay.json",
}

EXPECTED_SHA = {
    "rtl": "1178a0ae412a17059a2a2865025ff759b9fc351cbd7f20451f8621c92cce9fe8",
    "sva": "b037f722667d8600c47b16f47293563cd6c70a22ed8a0da1d3af2e3a0c1c5b27",
    "tb": "82d317c8952771c8adc4fd61679b798aaee540d017bf1caf20bda5064544ffaf",
    "contract": "c63d5265f56471a34cb5bc4c48b88260c4a61755fd14b5a89c669e9d5c81c5c0",
    "filelist": "561dab1f4d4e4d9d60633a79e437a683538ac01c1a8c375eeb3599f3bcc45591",
    "runner": "197c268e3c3ea64896201a46f346a88a8c6b99e6dc07b8bbd39d43b525ca0b7a",
    "builder": "7e1e087b17fef02de5833f9d0f0baf970fe729ac9048ec1e9dd95beee575a02a",
    "producer_validator": "6f47e6bbff55b8565608b211fcf5f61610a4baa68444bc4363bb11468f70325e",
    "r1b_receipt": "a4fca970d047382c13674761d49b3f3b1493ed9362814548644838921c482acc",
    "r1b_sim_log": "920ec8e29d341cc3f1882e1ea1f7b4aa06b3b94af17b001552eca7f6070cd48d",
    "r1b_output_manifest": "986ec621951fcb58eb3d785cd0f8bcba0e50aeac31b97342effbc68ab1792133",
    "r1b_simv": "812cb7a16f50b126e09334e468593c47487c2fb24dc17b7116756e54f34c7317",
    "m53_result": "344ae1f777e0640d46b19118f0b6d451465046350d68a9f33b1faae124747bb4",
    "m57_s00_manifest": "7e93928600e0ceeddf2e2103de66c7d065260e98a5845d44c0618d26c3c4c125",
    "m57_s00_replay": "6ff1f3101ae9d0c1a2331e428d133e17397005294ff54b2b16fc1caa31afec9b",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("non-standard JSON constant: " + value)))


def check_hashes():
    for name, path in PATHS.items():
        observed = sha256_path(path)
        require(observed == EXPECTED_SHA[name],
                "{} SHA drift {} != {}".format(
                    name, observed, EXPECTED_SHA[name]))


def validate_output_manifest():
    lines = PATHS["r1b_output_manifest"].read_text(
        encoding="utf-8").splitlines()
    require(len(lines) > 50, "r1b output manifest unexpectedly short")
    run_resolved = str(RUN.resolve()) + "/"
    seen = set()
    for line in lines:
        match = re.match(r"^([0-9a-f]{64})  (\./.+)$", line)
        require(match is not None, "malformed output manifest line")
        expected, relative = match.groups()
        target = (RUN / relative[2:]).resolve()
        require(str(target).startswith(run_resolved),
                "output manifest path escapes run")
        require(target.is_file(), "sealed output missing: " + relative)
        require(relative not in seen, "duplicate sealed output: " + relative)
        seen.add(relative)
        require(sha256_path(target) == expected,
                "sealed output SHA drift: " + relative)
    for required in ("./RUN_COMPLETE.txt", "./compile.raw.log",
                     "./sim.raw.log", "./m64_directed_vcs_receipt_r1.json",
                     "./validation.raw.log", "./simv"):
        require(required in seen, "required sealed output absent: " + required)


def xorshift32(value):
    value = (value ^ ((value << 13) & 0xffffffff)) & 0xffffffff
    value = (value ^ (value >> 17)) & 0xffffffff
    value = (value ^ ((value << 5) & 0xffffffff)) & 0xffffffff
    return value


def popcount(value):
    return bin(value).count("1")


def choose_parent(target, parents, valid):
    costs = [popcount(target ^ parent) for parent in parents]
    selected = 0
    tie_pairs = []
    for candidate in range(1, 4):
        if valid[candidate]:
            if costs[candidate] < costs[selected]:
                selected = candidate
            elif costs[candidate] == costs[selected]:
                tie_pairs.append((selected, candidate))
    mask = (1 << 256) - 1
    add_bits = target & ~parents[selected] & mask
    subtract_bits = parents[selected] & ~target & mask
    require((add_bits & subtract_bits) == 0, "oracle masks overlap")
    require(popcount(add_bits | subtract_bits) == costs[selected],
            "oracle source conservation")
    return selected, costs, add_bits, subtract_bits, tie_pairs


def regenerate_campaign():
    mask = (1 << 256) - 1
    rng = 0x64a52026
    hits = [0, 0, 0, 0]
    ties = {}
    invalid_better = [0, 0, 0, 0]
    max_selected = 0
    selected_256 = 0
    ledger = []
    for test_index in range(4096):
        vectors = [0, 0, 0, 0]
        for word in range(8):
            for vector_index in range(4):
                rng = xorshift32(rng)
                vectors[vector_index] |= rng << (word * 32)
        target, left, up, previous = vectors
        left_valid = (test_index % 7) != 0
        up_valid = (test_index % 11) != 0
        previous_valid = (test_index % 5) != 0
        if test_index == 0:
            target = 0
            left = up = previous = mask
        elif test_index == 1:
            left_valid = True
            target = left
        elif test_index == 2:
            up_valid = True
            target = up
        elif test_index == 3:
            previous_valid = True
            target = previous
        elif test_index == 4:
            left_valid = up_valid = previous_valid = True
            target = left
            up = left
            previous = left
        parents = [0, left, up, previous]
        valid = [True, left_valid, up_valid, previous_valid]
        selected, costs, add_bits, subtract_bits, row_ties = choose_parent(
            target, parents, valid)
        hits[selected] += 1
        max_selected = max(max_selected, costs[selected])
        selected_256 += int(costs[selected] == 256)
        for pair in row_ties:
            ties[pair] = ties.get(pair, 0) + 1
        for candidate in range(1, 4):
            if not valid[candidate] and costs[candidate] < costs[selected]:
                invalid_better[candidate] += 1
        ledger.append(
            "{},{},{},{:064x},{:064x},{}{}{}".format(
                test_index, selected, costs[selected], add_bits,
                subtract_bits, int(left_valid), int(up_valid),
                int(previous_valid)))
    supplemental = {}
    selected, costs, add_bits, subtract_bits, unused = choose_parent(
        mask, [0, mask, mask, mask], [True, False, False, False])
    supplemental["count_256_zero_parent"] = (
        selected == 0 and costs[selected] == 256 and add_bits == mask and
        subtract_bits == 0)
    pattern = int("f0" * 32, 16)
    selected, costs, unused_a, unused_s, unused_t = choose_parent(
        pattern, [0, pattern, 0, 0], [True, False, False, False])
    supplemental["invalid_identical_left_rejected"] = selected == 0
    selected, costs, unused_a, unused_s, unused_t = choose_parent(
        1, [0, 3, 0, 0], [True, True, False, False])
    supplemental["zero_over_left_tie_priority"] = (
        selected == 0 and costs[0] == costs[1] == 1)
    selected, costs, unused_a, unused_s, unused_t = choose_parent(
        15, [0, mask, 15, 15], [True, True, True, True])
    supplemental["up_over_previous_tie_priority"] = (
        selected == 2 and costs[2] == costs[3] == 0)
    require(all(supplemental.values()), "supplemental oracle case failed")
    return {
        "final_rng_state_hex": "{:08x}".format(rng),
        "ledger_sha256": hashlib.sha256(
            "\n".join(ledger).encode("ascii")).hexdigest(),
        "parent_hits": {
            "zero": hits[0], "left": hits[1], "up": hits[2],
            "previous_timestep": hits[3]},
        "tie_comparisons_preserving_earlier_parent": {
            "zero_left": ties.get((0, 1), 0),
            "zero_up": ties.get((0, 2), 0),
            "zero_previous": ties.get((0, 3), 0),
            "left_up": ties.get((1, 2), 0),
            "left_previous": ties.get((1, 3), 0),
            "up_previous": ties.get((2, 3), 0)},
        "total_tie_comparisons": sum(ties.values()),
        "invalid_candidate_strictly_better_than_selected_valid_parent": {
            "total": sum(invalid_better),
            "left": invalid_better[1], "up": invalid_better[2],
            "previous_timestep": invalid_better[3]},
        "maximum_selected_source_count": max_selected,
        "selected_source_count_256_cases": selected_256,
        "supplemental_software_boundary_cases": {
            name: "PASS" for name in sorted(supplemental)},
    }


def replay_elastic_schedule():
    tests = 4096
    s0_valid = False
    s1_valid = False
    holding_input = False
    next_offer = 2
    accepted = 0
    outputs = 0
    previous_accept = None
    consecutive_accepts = 0
    full_pop_advance_push = 0
    pop_push = 0
    output_blocked = 0
    input_blocked = 0
    maximum_occupancy = 0
    for posedge in range(1, 100000):
        out_ready = False if posedge == 1 else (
            ((posedge - 1) % 13) != 0 and ((posedge - 1) % 17) != 0)
        if (not holding_input and accepted < tests and
                posedge >= next_offer):
            holding_input = True
        s1_ready = (not s1_valid) or out_ready
        s0_ready = (not s0_valid) or s1_ready
        output_handshake = s1_valid and out_ready
        input_handshake = holding_input and s0_ready
        s0_advance = s0_valid and s1_ready
        output_blocked += int(s1_valid and not out_ready)
        input_blocked += int(holding_input and not s0_ready)
        full_pop_advance_push += int(
            output_handshake and s0_advance and input_handshake)
        pop_push += int(output_handshake and input_handshake)
        outputs += int(output_handshake)
        next_s1_valid = s0_valid if s1_ready else s1_valid
        next_s0_valid = holding_input if s0_ready else s0_valid
        s0_valid, s1_valid = next_s0_valid, next_s1_valid
        if input_handshake:
            consecutive_accepts += int(previous_accept == posedge - 1)
            previous_accept = posedge
            accepted += 1
            holding_input = False
            next_offer = posedge + 2
        maximum_occupancy = max(
            maximum_occupancy, int(s0_valid) + int(s1_valid))
        if outputs == tests:
            pattern_low = sum(
                1 for cycle in range(1, posedge)
                if cycle % 13 == 0 or cycle % 17 == 0)
            return {
                "terminal_posedge_count_after_reset": posedge,
                "backpressure_pattern_low_cycles": pattern_low,
                "actual_out_valid_and_not_ready_cycles": output_blocked,
                "simultaneous_output_pop_s0_advance_and_new_input_push":
                    full_pop_advance_push,
                "simultaneous_output_pop_and_input_push": pop_push,
                "input_backpressure_cycles": input_blocked,
                "consecutive_input_acceptance_pairs": consecutive_accepts,
                "maximum_pipeline_occupancy": maximum_occupancy,
            }
    raise ValueError("elastic replay failed to terminate")


def validate_receipt_core(receipt):
    require(receipt["status"] == "PASS_EXACT_SHA_DIRECTED_VCS_SVA",
            "producer receipt status drift")
    require(receipt["results"]["tests"] == 4096 and
            receipt["results"]["outputs"] == 4096 and
            receipt["results"]["functional_mismatches"] == 0 and
            receipt["results"]["assertion_failures"] == 0,
            "producer result counters drift")
    require(receipt["results"]["parent_hits"] == {
        "zero": 1271, "left": 974, "up": 988,
        "previous_timestep": 863}, "producer parent hits drift")
    require(receipt["results"]["output_stall_cycles"] == 1074,
            "producer stall counter drift")
    require(receipt["admission"] == {
        "all10_trace_admitted": False,
        "cycles_or_system_speedup_admitted": False,
        "dc_sta_formality_admitted": False,
        "directed_vcs_sva_admitted": True,
        "headline_admitted": False,
        "power_energy_ppa_admitted": False,
        "seed_sram_or_scheduler_admitted": False,
    }, "producer admission widened")


def validate_review_core(review, oracle, schedule):
    require(review["schema"] == "m64_independent_hammer_review_v1" and
            review["status"] ==
            "PASS_DIRECTED_FUNCTIONAL_AUDIT_NO_GO_DATE_HEADLINE",
            "review identity/status drift")
    scores = review["scores"]
    require(scores["date_prosperity_phi_completeness_score"] == 43 and
            sum(scores["subscores"].values()) == 43 and
            scores["directed_rtl_readiness_score"] == 82 and
            scores["novelty_potential_score"] == 48,
            "review score drift")
    require(review["issues"]["P0"] == [] and
            [item["id"] for item in review["issues"]["P1"]] == [
                "M64-P1-01-ALL10-M57-INTEGRATION",
                "M64-P1-02-SEED-SRAM-BANDWIDTH-COST",
                "M64-P1-03-THROUGHPUT-AND-SYNOPSYS-CLOSURE",
                "M64-P1-04-NOVELTY-PERFORMANCE-COMPARISON"] and
            [item["id"] for item in review["issues"]["P2"]] == [
                "M64-P2-01-COUNT256-ENDPOINT",
                "M64-P2-02-STALL-METRIC-NAME",
                "M64-P2-03-SVA-SEMANTIC-DEPTH",
                "M64-P2-04-PARAMETER-GUARD",
                "M64-P2-05-PRODUCER-VALIDATOR-INDEPENDENCE"],
            "review issue inventory drift")
    observed_oracle = dict(review["independent_oracle"])
    observed_oracle.pop("generator")
    observed_oracle.pop("seed_hex")
    require(observed_oracle == oracle, "independent oracle evidence drift")
    observed_schedule = dict(review["independent_elastic_schedule_replay"])
    observed_schedule.pop("interpretation")
    require(observed_schedule == schedule, "elastic schedule evidence drift")
    require(review["m57_integration_gate"] == {
        "m57_current_scope": "sample_id_0 streaming FIFO/tag arithmetic replay",
        "m57_status": "PASS_M57_STREAMING_FIFO_TAG_ARITHMETIC_REPLAY",
        "m57_rtl_cycles": 8791654,
        "m57_system_or_full_network_cycles_admitted": False,
        "m64_instantiated_in_m57": False,
        "m64_m53_all10_parent_choice_equivalence": False,
        "all10_cycle_replay_with_m64": False,
        "gate": "NO_GO_UNTIL_M64_IS_INSERTED_BEFORE_M57_AND_ALL10_PARENT_TAG_MASK_AND_CYCLE_LEDGER_MATCHES_M53",
    }, "M57 integration gate drift")
    require(review["novelty_and_story_judgment"][
        "current_performance_advantage"] == "UNPROVEN",
        "performance advantage improperly promoted")
    require(review["admission_gate_for_next_milestone"]["current_result"] ==
            "NO_GO_DATE_HEADLINE_GO_M64_INTEGRATED_M57_ALL10_AND_SYNOPSYS_NEXT",
            "next gate drift")


def run_attacks(receipt, review, oracle, schedule):
    attacks = []
    mutant = copy.deepcopy(receipt)
    mutant["results"]["parent_hits"]["zero"] += 1
    try:
        validate_receipt_core(mutant)
        rejected = False
    except ValueError:
        rejected = True
    require(rejected, "parent-hit tamper accepted")
    attacks.append({"name": "receipt_parent_hit_tamper", "rejected": True})

    mutant = copy.deepcopy(receipt)
    mutant["admission"]["cycles_or_system_speedup_admitted"] = True
    try:
        validate_receipt_core(mutant)
        rejected = False
    except ValueError:
        rejected = True
    require(rejected, "speedup admission tamper accepted")
    attacks.append({"name": "receipt_system_speedup_promotion",
                    "rejected": True})

    mutant = copy.deepcopy(review)
    mutant["scores"]["date_prosperity_phi_completeness_score"] = 93
    try:
        validate_review_core(mutant, oracle, schedule)
        rejected = False
    except ValueError:
        rejected = True
    require(rejected, "review score tamper accepted")
    attacks.append({"name": "review_score_promotion", "rejected": True})

    mutant = copy.deepcopy(review)
    mutant["m57_integration_gate"]["all10_cycle_replay_with_m64"] = True
    try:
        validate_review_core(mutant, oracle, schedule)
        rejected = False
    except ValueError:
        rejected = True
    require(rejected, "all10 integration tamper accepted")
    attacks.append({"name": "m57_all10_false_promotion", "rejected": True})
    return attacks


def validate_validation_receipt(receipt, attacks, oracle, schedule):
    require(receipt["schema"] ==
            "m64_independent_hammer_validation_receipt_v1" and
            receipt["status"] == "PASS_M64_INDEPENDENT_HAMMER_VALIDATED",
            "validation receipt status drift")
    require(receipt["review_sha256"] == sha256_path(REVIEW) and
            receipt["validator_sha256"] == sha256_path(Path(__file__)) and
            receipt["producer_r1b_receipt_sha256"] ==
            EXPECTED_SHA["r1b_receipt"] and
            receipt["producer_r1b_output_manifest_sha256"] ==
            EXPECTED_SHA["r1b_output_manifest"],
            "validation receipt identity drift")
    require(receipt["scores"] == {
        "date_prosperity_phi_completeness": 43,
        "directed_rtl_readiness": 82,
        "novelty_potential": 48}, "validation receipt score drift")
    require(receipt["severity_counts"] == {"P0": 0, "P1": 4, "P2": 5},
            "validation receipt severity drift")
    require(receipt["oracle_summary"] == {
        "final_rng_state_hex": oracle["final_rng_state_hex"],
        "ledger_sha256": oracle["ledger_sha256"],
        "parent_hits": oracle["parent_hits"],
        "maximum_selected_source_count": 142,
        "selected_source_count_256_cases": 0,
        "tie_comparisons": 390,
        "discriminating_invalid_candidates": 464,
    }, "validation receipt oracle drift")
    require(receipt["elastic_summary"] == {
        "terminal_posedges": schedule["terminal_posedge_count_after_reset"],
        "pattern_low_cycles": schedule["backpressure_pattern_low_cycles"],
        "actual_blocked_output_cycles":
            schedule["actual_out_valid_and_not_ready_cycles"],
        "full_pop_advance_push_cases": schedule[
            "simultaneous_output_pop_s0_advance_and_new_input_push"],
        "consecutive_input_acceptance_pairs": 0,
    }, "validation receipt elastic drift")
    require(receipt["negative_attacks"] == attacks,
            "validation receipt attacks drift")
    require(receipt["admission"] == {
        "real_all10_trace": False,
        "m57_m64_integration": False,
        "seed_sram_address_cost": False,
        "dc_sta_formality": False,
        "system_speedup": False,
        "date_headline": False,
    }, "validation receipt admission drift")


def main():
    check_hashes()
    require((RUN / "RUN_COMPLETE.txt").read_text(encoding="utf-8").splitlines() == [
        "status=PASS_EXACT_SHA_DIRECTED_VCS_SVA",
        "system_speedup_admitted=false",
        "dc_sta_formality_admitted=false",
        "paper_ppa_ready=false"], "r1b completion seal drift")
    require((R1 / "FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt").is_file(),
            "r1 exclusion marker missing")
    validate_output_manifest()
    receipt = strict_json(PATHS["r1b_receipt"])
    review = strict_json(REVIEW)
    validate_receipt_core(receipt)
    oracle = regenerate_campaign()
    schedule = replay_elastic_schedule()
    validate_review_core(review, oracle, schedule)
    m57 = strict_json(PATHS["m57_s00_replay"])
    require(m57["sample_id"] == 0 and
            m57["status"] == "PASS_M57_STREAMING_FIFO_TAG_ARITHMETIC_REPLAY" and
            m57["rtl_cycles"] == 8791654 and
            m57["system_or_full_network_cycles_admitted"] is False,
            "M57 source evidence drift")
    attacks = run_attacks(receipt, review, oracle, schedule)
    validation_receipt = strict_json(VALIDATION_RECEIPT)
    validate_validation_receipt(validation_receipt, attacks, oracle, schedule)
    print("PASS M64 independent hammer score=43 directed_readiness=82 "
          "P0=0 P1=4 P2=5 attacks={}/{} all10=false dc_sta_formality=false "
          "system_speedup=false".format(len(attacks), len(attacks)))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("FAIL M64 independent hammer: {}".format(error), file=sys.stderr)
        sys.exit(1)
