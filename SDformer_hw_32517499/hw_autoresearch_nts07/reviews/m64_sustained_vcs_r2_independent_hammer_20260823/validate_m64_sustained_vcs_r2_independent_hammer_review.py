#!/usr/bin/env python3
"""Fail-closed independent validator for sealed M64 sustained VCS r2."""

from __future__ import print_function

import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUN = HW / "dc_handoff/runs/m64_parent_selector_sustained_vcs_r2_sealed_20260823"
REVIEW = HERE / "m64_sustained_vcs_r2_independent_hammer_review.json"
RECEIPT = HERE / "m64_sustained_vcs_r2_independent_hammer_validation_receipt.json"
PRODUCER_RECEIPT = RUN / "m64_parent_selector_sustained_vcs_receipt_r2.json"

EXPECTED_SHA = {
    "snapshot": "f5c376e1f2f6b7971680c0683e047a6d76ccb3878272a01c90a1fa120f827e8f",
    "output": "5e80e00364a93478184629dbafd974462e2681d16a52e2763231f02e4708f8ae",
    "producer_receipt": "ebbd84dfa7a4d04d3661b7168f904c6c60769ded33b89b9799d06005ad81590d",
    "run_complete": "67b0188c15e3b460fae055bcaa21ad649dc81cd590aac5cd30d72841ca756c90",
    "compile": "f34a5a1b7bedb5221d120458e1bb064768128de90690726a7cb242e1af6b2a57",
    "simulation": "4dfa4a396a4cfdd6adc52f14c89768944a2a9177395725394febec56e3824090",
    "simv": "0a58d3808d777913257d9b59217c890a85740bedb46db34618bd6d760c2c6d5b",
    "rtl": "1178a0ae412a17059a2a2865025ff759b9fc351cbd7f20451f8621c92cce9fe8",
    "tb": "f12ee02c5cda96e0e79a374b4837bc13919ace046f4032bbfc0e4d077b8a7615",
    "sva": "adf139a9ab1c67dd6682c971601d6c917a91d984a36357baaec12a2b8a320313",
    "contract": "7f7bb86ea1694a615f10e547ae328492e5c8a28e334719e085285cb475a28b9f",
    "vcs": "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
}

PATHS = {
    "snapshot": RUN / "snapshot.sha256",
    "output": RUN / "output_manifest.sha256",
    "producer_receipt": PRODUCER_RECEIPT,
    "run_complete": RUN / "RUN_COMPLETE.txt",
    "compile": RUN / "compile.raw.log",
    "simulation": RUN / "sim.raw.log",
    "simv": RUN / "simv",
    "rtl": RUN / "snapshot/rtl_m64/qfit_adaptive_parent_selector_p256.sv",
    "tb": RUN / "snapshot/tb_m64/tb_qfit_adaptive_parent_selector_p256_sustained_r2.sv",
    "sva": RUN / "snapshot/verif_m64/qfit_adaptive_parent_selector_p256_sustained_assertions_r2.sv",
    "contract": RUN / "snapshot/contracts/m64_parent_selector_sustained_vcs_contract_r2_20260823.json",
    "vcs": Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"),
}

PASS_RE = re.compile(
    r"^PASS M64 R2 sustained tests=(\d+) inputs=(\d+) outputs=(\d+) "
    r"b2b_accepts=(\d+) full_cycles=(\d+) max_full_run=(\d+) "
    r"full_push_pop=(\d+) source256=(\d+) "
    r"parent_hits=(\d+),(\d+),(\d+),(\d+) ties=(\d+) "
    r"random_stalls=(\d+) output_stalls=(\d+) max_outstanding=(\d+) "
    r"valid_low=(\d+) mismatches=(\d+)$", re.MULTILINE)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def strict_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=no_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("non-standard JSON constant: " + value)))


def text(path):
    return Path(path).read_text(encoding="utf-8", errors="strict")


def validate_hashes():
    for name, expected in EXPECTED_SHA.items():
        observed = sha256_path(PATHS[name])
        require(observed == expected,
                "{} SHA drift {} != {}".format(name, observed, expected))


def validate_manifest(manifest, root, expected_count, required):
    lines = text(manifest).splitlines()
    require(len(lines) == expected_count,
            "manifest count drift {} != {}".format(
                len(lines), expected_count))
    root_string = str(root.resolve()) + "/"
    seen = set()
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (\./.+)", line)
        require(match is not None, "malformed manifest line")
        expected, relative = match.groups()
        target = (root / relative[2:]).resolve()
        require(str(target).startswith(root_string),
                "manifest path escape: " + relative)
        require(relative not in seen, "duplicate manifest path: " + relative)
        seen.add(relative)
        require(target.is_file(), "manifest target missing: " + relative)
        require(sha256_path(target) == expected,
                "manifest SHA drift: " + relative)
    for item in required:
        require(item in seen, "required manifest entry absent: " + item)


def validate_manifests():
    validate_manifest(
        PATHS["snapshot"], RUN / "snapshot", 11,
        {
            "./rtl_m64/qfit_adaptive_parent_selector_p256.sv",
            "./tb_m64/tb_qfit_adaptive_parent_selector_p256_sustained_r2.sv",
            "./verif_m64/qfit_adaptive_parent_selector_p256_sustained_assertions_r2.sv",
            "./contracts/m64_parent_selector_sustained_vcs_contract_r2_20260823.json",
            "./dc_handoff/scripts/run_vcs_m64_parent_selector_sustained_r2.sh",
            "./dc_handoff/scripts/build_m64_parent_selector_sustained_vcs_r2_receipt.py",
            "./dc_handoff/scripts/validate_m64_parent_selector_sustained_vcs_r2.py",
        })
    validate_manifest(
        PATHS["output"], RUN, 107,
        {
            "./RUN_COMPLETE.txt", "./compile.raw.log", "./compile.rc",
            "./sim.raw.log", "./sim.rc", "./simv",
            "./m64_parent_selector_sustained_vcs_receipt_r2.json",
            "./receipt.sha256", "./snapshot.sha256",
            "./validator.raw.log",
            "./snapshot/rtl_m64/qfit_adaptive_parent_selector_p256.sv",
            "./snapshot/tb_m64/tb_qfit_adaptive_parent_selector_p256_sustained_r2.sv",
            "./snapshot/verif_m64/qfit_adaptive_parent_selector_p256_sustained_assertions_r2.sv",
        })


def xorshift32(value):
    value = (value ^ ((value << 13) & 0xffffffff)) & 0xffffffff
    value = (value ^ (value >> 17)) & 0xffffffff
    value = (value ^ ((value << 5) & 0xffffffff)) & 0xffffffff
    return value


def popcount(value):
    return bin(value).count("1")


def reconstruct_oracle():
    mask = (1 << 256) - 1
    rng = 0x64a52026
    hits = [0, 0, 0, 0]
    ties = Counter()
    invalid_better = Counter()
    selected_counts = Counter()
    ledger = []
    for test_index in range(2048):
        vectors = [0, 0, 0, 0]
        for word in range(8):
            for vector_index in range(4):
                rng = xorshift32(rng)
                vectors[vector_index] |= rng << (32 * word)
        target, left, up, previous = vectors
        valid = [True, test_index % 7 != 0, test_index % 11 != 0,
                 test_index % 5 != 0]
        if test_index == 0:
            target = mask
            left = up = previous = 0
            valid = [True, False, False, False]
        elif test_index == 1:
            target = 0
            left = up = previous = mask
            valid = [True, True, True, True]
        elif test_index == 2:
            target = int("a5c3691e" * 8, 16)
            left = target
            up = (~target) & mask
            previous = int("3c96a55a" * 8, 16)
            valid = [True, True, True, True]
        elif test_index == 3:
            target = int("5a3c96e1" * 8, 16)
            left = (~target) & mask
            up = target
            previous = int("c3695aa5" * 8, 16)
            valid = [True, True, True, True]
        elif test_index == 4:
            target = int("0ff0c33c" * 8, 16)
            left = (~target) & mask
            up = int("f00f3cc3" * 8, 16)
            previous = target
            valid = [True, True, True, True]
        elif test_index == 5:
            target = left = up = previous = 0
            valid = [True, True, True, True]
        elif test_index == 6:
            target = int("96963cc3" * 8, 16)
            left = up = previous = target
            valid = [True, True, True, True]
        elif test_index == 7:
            target = int("69c3a55a" * 8, 16)
            left = (~target) & mask
            up = previous = target
            valid = [True, False, True, True]
        parents = [0, left, up, previous]
        costs = [popcount(target ^ parent) for parent in parents]
        selected = 0
        for candidate in range(1, 4):
            if valid[candidate] and costs[candidate] < costs[selected]:
                selected = candidate
            elif valid[candidate] and costs[candidate] == costs[selected]:
                ties[(selected, candidate)] += 1
        for candidate in range(1, 4):
            if not valid[candidate] and costs[candidate] < costs[selected]:
                invalid_better[candidate] += 1
        add_bits = target & (~parents[selected] & mask)
        subtract_bits = parents[selected] & (~target & mask)
        require((add_bits & subtract_bits) == 0, "oracle mask overlap")
        require(popcount(add_bits | subtract_bits) == costs[selected],
                "oracle source-count conservation failure")
        hits[selected] += 1
        selected_counts[costs[selected]] += 1
        ledger.append(
            "%d,%012x,%d,%d,%064x,%064x,%d%d%d" % (
                test_index, 0x640200000000 + test_index, selected,
                costs[selected], add_bits, subtract_bits, valid[1], valid[2],
                valid[3]))
    return {
        "final_rng": "%08x" % rng,
        "ledger_sha256": hashlib.sha256(
            "\n".join(ledger).encode("ascii")).hexdigest(),
        "hits": hits,
        "source_min": min(selected_counts),
        "source_max": max(selected_counts),
        "source256": selected_counts[256],
        "ties": {
            "zero_left": ties[(0, 1)], "zero_up": ties[(0, 2)],
            "zero_previous": ties[(0, 3)], "left_up": ties[(1, 2)],
            "left_previous": ties[(1, 3)], "up_previous": ties[(2, 3)],
        },
        "invalid": {
            "left": invalid_better[1], "up": invalid_better[2],
            "previous_timestep": invalid_better[3],
        },
    }


def consecutive_windows(vector, width):
    return sum(all(vector[index:index + width])
               for index in range(len(vector) - width + 1))


def maximum_run(vector):
    maximum = 0
    current = 0
    for value in vector:
        if value:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def reconstruct_schedule():
    writes = reads = 0
    s0_valid = s1_valid = False
    out_ready = True
    throughput_phase = True
    backpressure_rng = 0x64b25e11
    random_decisions = 0
    maximum_outstanding = 0
    input_accepts = []
    output_accepts = []
    full_events = []
    push_pop_events = []
    stalls = []
    while reads < 2048:
        in_valid = writes < 2048
        s1_ready = (not s1_valid) or out_ready
        s0_ready = (not s0_valid) or s1_ready
        input_accept = in_valid and s0_ready
        output_accept = s1_valid and out_ready
        push_pop = (input_accept and output_accept and s0_valid and s1_valid)
        full = throughput_phase and push_pop
        input_accepts.append(input_accept)
        output_accepts.append(output_accept)
        push_pop_events.append(push_pop)
        full_events.append(full)
        stalls.append(s1_valid and not out_ready)
        if output_accept:
            reads += 1
        if input_accept:
            writes += 1
        maximum_outstanding = max(maximum_outstanding, writes - reads)
        require(0 <= writes - reads <= 2, "replayed occupancy violation")
        if s1_ready:
            next_s1_valid = s0_valid
        else:
            next_s1_valid = s1_valid
        if s0_ready:
            next_s0_valid = in_valid
        else:
            next_s0_valid = s0_valid
        s0_valid, s1_valid = next_s0_valid, next_s1_valid
        if writes < 128:
            out_ready = True
            throughput_phase = True
        elif reads < 2048:
            backpressure_rng = xorshift32(backpressure_rng)
            out_ready = bool((backpressure_rng & 1) or
                             (backpressure_rng & 8))
            throughput_phase = False
            random_decisions += 1
        else:
            out_ready = True
            throughput_phase = False
    return {
        "cycles": len(input_accepts), "writes": writes, "reads": reads,
        "b2b": consecutive_windows(input_accepts, 2),
        "accept8": consecutive_windows(input_accepts, 8),
        "max_accept_run": maximum_run(input_accepts),
        "full_cycles": sum(full_events),
        "full8": consecutive_windows(full_events, 8),
        "max_full_run": maximum_run(full_events),
        "push_pop": sum(push_pop_events), "stalls": sum(stalls),
        "maximum_outstanding": maximum_outstanding,
        "random_decisions": random_decisions,
        "final_rng": "%08x" % backpressure_rng,
    }


def validate_sources():
    tb = text(PATHS["tb"])
    sva = text(PATHS["sva"])
    for token in (
            "out_tag !== expected_tag[reads]",
            "out_parent_id !== expected_parent[reads]",
            "out_add_bits !== expected_add[reads]",
            "out_subtract_bits !== expected_subtract[reads]",
            "out_source_count !== expected_count[reads][8:0]",
            "$fatal(1, \"M64-r2 oracle mismatch",
            "stimulus_rng = 32'h64a5_2026",
            "backpressure_rng = 32'h64b2_5e11",
            "in_valid = 1'b1", "source256_outputs < 1",
            "max_full_throughput_run < 32"):
        require(token in tb, "testbench semantic token absent: " + token)
    assert_labels = re.findall(r"^\s*(ap_[A-Za-z0-9_]+): assert property",
                               sva, flags=re.MULTILINE)
    cover_labels = re.findall(r"^\s*(cp_[A-Za-z0-9_]+): cover property",
                              sva, flags=re.MULTILINE)
    require(len(assert_labels) == 5 and len(set(assert_labels)) == 5,
            "SVA assertion inventory drift")
    require(len(cover_labels) == 11 and len(set(cover_labels)) == 11,
            "SVA cover inventory drift")


def validate_logs_and_receipt(oracle, schedule):
    compile_log = text(PATHS["compile"])
    sim_log = text(PATHS["simulation"])
    require(text(RUN / "compile.rc").strip() == "0", "compile RC nonzero")
    require(text(RUN / "sim.rc").strip() == "0", "simulation RC nonzero")
    require("V-2023.12-SP1_Full64" in compile_log and
            "V-2023.12-SP1_Full64" in sim_log, "VCS identity absent")
    require(re.search(r"Warning-\[|Error-\[|^Error", compile_log,
                      flags=re.IGNORECASE | re.MULTILINE) is None,
            "compile diagnostic signature found")
    require(re.search(r"Assertion failure|failed at|Offending|\bFatal\b|\bError-\[",
                      sim_log, flags=re.IGNORECASE) is None,
            "simulation failure signature found")
    require(sim_log.splitlines().count(
        "M64_R2_SUSTAINED_ASSERTION_MODULE_ACTIVE=1") == 1,
        "assertion module active line missing/duplicated")
    matches = PASS_RE.findall(sim_log)
    require(len(matches) == 1, "terminal PASS missing/duplicated")
    values = [int(value) for value in matches[0]]
    require(values == [
        2048, 2048, 2048, 1561, 126, 126, 2046, 1,
        628, 495, 486, 439, 3, 648, 648, 2, 0, 0,
    ], "terminal PASS metric drift")
    cover_matches = {}
    for name, attempts, hits in re.findall(
            r"r2_sva\.(cp_[A-Za-z0-9_]+),\s*(\d+) attempts,\s*(\d+) match",
            sim_log):
        require(name not in cover_matches, "duplicate cover line: " + name)
        require(int(attempts) == 2703, "cover attempts drift: " + name)
        cover_matches[name] = int(hits)
    expected_covers = {
        "cp_source_count_256": 1,
        "cp_parent_zero": oracle["hits"][0],
        "cp_parent_left": oracle["hits"][1],
        "cp_parent_up": oracle["hits"][2],
        "cp_parent_previous": oracle["hits"][3],
        "cp_forced_tie_accept": 3,
        "cp_random_output_backpressure": schedule["stalls"],
        "cp_pipeline_full_push_pop_same_cycle": schedule["push_pop"],
        "cp_back_to_back_input_accept": schedule["b2b"],
        "cp_sustained_accept_8": schedule["accept8"],
        "cp_full_throughput_8": schedule["full8"],
    }
    require(cover_matches == expected_covers,
            "cover match ledger differs from independent reconstruction")
    receipt = strict_json(PRODUCER_RECEIPT)
    require(receipt["status"] == "PASS_EXACT_SHA_SYNOPSYS_VCS_SUSTAINED_R2",
            "producer receipt state drift")
    results = receipt["results"]
    require(results["accepted_inputs"] == schedule["writes"] and
            results["accepted_outputs"] == schedule["reads"] and
            results["back_to_back_input_accepts"] == schedule["b2b"] and
            results["full_throughput_cycles"] == schedule["full_cycles"] and
            results["maximum_full_throughput_run"] == schedule["max_full_run"] and
            results["pipeline_full_push_pop_cycles"] == schedule["push_pop"] and
            results["random_output_stall_cycles"] == schedule["stalls"] and
            results["maximum_outstanding"] == schedule["maximum_outstanding"],
            "producer schedule receipt differs from independent replay")
    require(results["parent_hits"] == {
        "zero": oracle["hits"][0], "left": oracle["hits"][1],
        "up": oracle["hits"][2],
        "previous_timestep": oracle["hits"][3],
    }, "producer parent-hit receipt differs from independent oracle")
    require(results["source_count_256_outputs"] == oracle["source256"],
            "producer source256 count differs from independent oracle")
    claim = receipt["claim_boundary"]
    require(claim["sustained_directed_vcs_sva_admitted"] is True,
            "directed admission missing")
    for field in ("system_speedup_admitted", "headline_admitted",
                  "ppa_admitted", "power_energy_admitted",
                  "all10_or_full_network_admitted",
                  "random_or_formal_protocol_proof_admitted"):
        require(claim[field] is False, "producer claim promotion: " + field)


def validate_reconstruction(oracle, schedule):
    require(oracle == {
        "final_rng": "0c958f32",
        "ledger_sha256": "4bc8604da999b28b44b5efef29f5b673529815366b4b31754b961e452fcc1d2f",
        "hits": [628, 495, 486, 439],
        "source_min": 0, "source_max": 256, "source256": 1,
        "ties": {
            "zero_left": 72, "zero_up": 38, "zero_previous": 25,
            "left_up": 25, "left_previous": 14, "up_previous": 15,
        },
        "invalid": {"left": 77, "up": 54, "previous_timestep": 114},
    }, "independent oracle reconstruction drift")
    require(schedule == {
        "cycles": 2698, "writes": 2048, "reads": 2048,
        "b2b": 1561, "accept8": 379, "max_accept_run": 131,
        "full_cycles": 126, "full8": 119, "max_full_run": 126,
        "push_pop": 2046, "stalls": 648, "maximum_outstanding": 2,
        "random_decisions": 2570, "final_rng": "10fc87d1",
    }, "independent elastic schedule reconstruction drift")


def validate_review_and_attacks(oracle, schedule):
    review = strict_json(REVIEW)
    scores = review["scores"]
    require(scores["sustained_vcs_extension_quality_score"] == 90,
            "extension score drift")
    require(sum(scores["extension_subscores"].values()) == 90,
            "extension subscore arithmetic drift")
    require(scores["directed_rtl_readiness_score"] == 94,
            "directed readiness score drift")
    require(scores["combined_m64_date_prosperity_phi_completeness_score"] == 58,
            "DATE completeness score drift")
    require(sum(scores["combined_subscores"].values()) == 58,
            "DATE subscore arithmetic drift")
    require(len(review["issues"]["P0"]) == 0, "P0 count drift")
    require(len(review["issues"]["P1"]) == 4, "P1 count drift")
    require(len(review["issues"]["P2"]) == 5, "P2 count drift")
    independent = review["independent_oracle_reconstruction"]
    require(independent["expected_ledger_sha256"] == oracle["ledger_sha256"],
            "review oracle ledger drift")
    require(independent["actual_output_ledger_available"] is False,
            "false actual-ledger promotion")
    replay = review["independent_elastic_schedule_reconstruction"]
    require(replay["back_to_back_input_accepts"] == schedule["b2b"] and
            replay["full_throughput_cycles"] == schedule["full_cycles"] and
            replay["pipeline_full_push_pop_cycles"] == schedule["push_pop"] and
            replay["random_output_stall_cycles"] == schedule["stalls"],
            "review replay metric drift")
    claim = review["claim_boundary"]
    require(claim["sustained_directed_vcs_sva_admitted"] is True,
            "review directed admission missing")
    for field in ("system_speedup_admitted", "headline_admitted",
                  "ppa_admitted", "power_energy_admitted",
                  "all10_or_full_network_admitted",
                  "random_or_formal_protocol_proof_admitted"):
        require(claim[field] is False, "review claim promotion: " + field)
    attacks = []
    for name, mutate in (
        ("system_speedup_promotion",
         lambda data: data["claim_boundary"].__setitem__(
             "system_speedup_admitted", True)),
        ("formal_protocol_promotion",
         lambda data: data["claim_boundary"].__setitem__(
             "random_or_formal_protocol_proof_admitted", True)),
        ("actual_ledger_false_promotion",
         lambda data: data["independent_oracle_reconstruction"].__setitem__(
             "actual_output_ledger_available", True)),
        ("source256_tamper",
         lambda data: data["sealed_results"].__setitem__(
             "source_count_256_outputs", 0)),
        ("extension_score_promotion",
         lambda data: data["scores"].__setitem__(
             "sustained_vcs_extension_quality_score", 99)),
        ("full_throughput_tamper",
         lambda data: data["independent_elastic_schedule_reconstruction"].__setitem__(
             "full_throughput_cycles", 2048)),
    ):
        attacked = copy.deepcopy(review)
        mutate(attacked)
        rejected = False
        try:
            require(attacked["scores"]["sustained_vcs_extension_quality_score"] == 90,
                    "score")
            require(attacked["claim_boundary"]["system_speedup_admitted"] is False,
                    "speedup")
            require(attacked["claim_boundary"]["random_or_formal_protocol_proof_admitted"] is False,
                    "formal")
            require(attacked["independent_oracle_reconstruction"]["actual_output_ledger_available"] is False,
                    "ledger")
            require(attacked["sealed_results"]["source_count_256_outputs"] == 1,
                    "source256")
            require(attacked["independent_elastic_schedule_reconstruction"]["full_throughput_cycles"] == 126,
                    "throughput")
        except ValueError:
            rejected = True
        require(rejected, "negative attack not rejected: " + name)
        attacks.append({"name": name, "rejected": True})
    return review, attacks


def validate_receipt(attacks):
    receipt = strict_json(RECEIPT)
    require(receipt["status"] ==
            "PASS_M64_SUSTAINED_VCS_R2_INDEPENDENT_HAMMER_VALIDATED",
            "independent receipt status drift")
    require(receipt["review_sha256"] == sha256_path(REVIEW),
            "receipt review SHA drift")
    require(receipt["validator_sha256"] == sha256_path(Path(__file__)),
            "receipt validator SHA drift")
    require(receipt["producer_snapshot_manifest_sha256"] == EXPECTED_SHA["snapshot"],
            "receipt snapshot SHA drift")
    require(receipt["producer_output_manifest_sha256"] == EXPECTED_SHA["output"],
            "receipt output SHA drift")
    require(receipt["producer_receipt_sha256"] == EXPECTED_SHA["producer_receipt"],
            "receipt producer-receipt SHA drift")
    require(receipt["scores"] == {
        "sustained_vcs_extension_quality": 90,
        "directed_rtl_readiness": 94,
        "combined_m64_date_completeness": 58,
        "novelty_potential": 48,
    }, "receipt score drift")
    require(receipt["severity_counts"] == {"P0": 0, "P1": 4, "P2": 5},
            "receipt severity drift")
    require(receipt["negative_attacks"] == attacks,
            "receipt negative-attack inventory drift")
    require(receipt["admission"] == {
        "sustained_directed_rtl": True,
        "actual_output_file_ledger": False,
        "formal_protocol_proof": False,
        "seed_sram_parent_bandwidth": False,
        "m57_all10_system_speedup": False,
        "paper_performance_headline": False,
    }, "receipt admission drift")


def main():
    validate_hashes()
    validate_manifests()
    validate_sources()
    oracle = reconstruct_oracle()
    schedule = reconstruct_schedule()
    validate_reconstruction(oracle, schedule)
    validate_logs_and_receipt(oracle, schedule)
    unused_review, attacks = validate_review_and_attacks(oracle, schedule)
    validate_receipt(attacks)
    print("PASS_M64_SUSTAINED_VCS_R2_INDEPENDENT_HAMMER_VALIDATED")
    print("snapshot_sha256=" + EXPECTED_SHA["snapshot"])
    print("output_manifest_sha256=" + EXPECTED_SHA["output"])
    print("oracle_ledger_sha256=" + oracle["ledger_sha256"])
    print("b2b=1561 full=126 push_pop=2046 source256=1 stalls=648")
    print("scores=90/94/58 severities=P0:0,P1:4,P2:5")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("FAIL_M64_SUSTAINED_VCS_R2_INDEPENDENT_HAMMER: " + str(exc),
              file=sys.stderr)
        sys.exit(1)
