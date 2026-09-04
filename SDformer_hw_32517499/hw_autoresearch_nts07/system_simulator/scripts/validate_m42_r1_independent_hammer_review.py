#!/usr/bin/env python3
"""Fail-closed independent validator for the M42-r1 headroom-gate review.

The raw M40 bitmap replay and M39/M42 budget arithmetic are rebuilt here
without calling the M42 analyzer.  The candidate analyzer is imported only
for its regression/adversarial behavior after the independent facts close.
"""

from __future__ import print_function

import argparse
import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
DEFAULT_REVIEW = HW_ROOT / (
    "results/m42_real_work_headroom_gate_r1_20260823/"
    "m42_r1_independent_hammer_review.json")

ANCHORS = {
    "contract": (
        "contracts/m42_real_work_headroom_gate_contract_r1_20260823.json",
        "b197daa2f6881cc9489aa067a0bb3f110a38b0755ea6569fa3469bd8e5da76eb"),
    "analyzer": (
        "system_simulator/scripts/analyze_m42_real_work_headroom_gate.py",
        "40089996c4313ad5f5a8623ceedb19af18a37a25634b377c5072143d2c65ef57"),
    "regression": (
        "system_simulator/tests/test_m42_real_work_headroom_gate.py",
        "ee431d37d039447b598aa5e94485dfc4497a976144f5aa260a8ae59a699d710e"),
    "result": (
        "results/m42_real_work_headroom_gate_r1_20260823/"
        "m42_real_work_headroom_gate.json",
        "c0677ce56775996481ba500fc397191e7de407768f29c591ae731c69ed45cd13"),
    "specification": (
        "rtl_m42/M42_REAL_WORK_HEADROOM_GATE_R1.md",
        "f0c0966f8e3084642ec3a2d797d4feca1669592b1410cc2bd783aa346d459387"),
    "m39_result": (
        "results/m39_remaining_bottleneck_r3_20260822/"
        "m39_remaining_bottleneck.json",
        "8923bbf5b1e630ad8e940ffa967f18ae9e59176c3f2dd6b29af2c1d696fbdcbb"),
    "m40_result": (
        "results/m40_conflict_aware_event_schedule_r3_20260822/"
        "m40_conflict_aware_event_schedule.json",
        "419ea51faabda4c2f45b9fa535d1a0fa8142bb4c8b8258468e88a1dc99c310e7"),
    "m40_independent_review": (
        "results/m40_conflict_aware_event_schedule_r3_20260822/"
        "m40a_r3_independent_hammer_review.json",
        "b562d2b77ed5b3acb04ae6688c96f033b69c16faa3b73e984f8c29380c417abf"),
    "m40_packed_source_manifest": (
        "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822/"
        "m40_bottleneck_packed_source_manifest.json",
        "e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3"),
    "p8_l96_engine_rtl": (
        "rtl_qfit/qfit_local_banked_multisource_engine.sv",
        "4003637653110fe2407b646a9f82ca4b77d775e01c1151c3c4ce0a8c47c0b3dc"),
    "p8_l96_dc_tops_rtl": (
        "rtl_qfit/qfit_local_banked_multisource_l96_dc_tops.sv",
        "9656d79a87ce8057cd8f3926bb2f57f91fa241485a11cb59a9b7d3712ab0a019"),
}

TARGETS = (
    "sttmultires_unet.resblocks.0.conv1.0",
    "sttmultires_unet.resblocks.0.conv2.0",
    "sttmultires_unet.resblocks.1.conv1.0",
    "sttmultires_unet.resblocks.1.conv2.0",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_constant(raw):
    raise ValueError("non-standard JSON constant: {}".format(raw))


def read_json(path):
    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: {}".format(key))
            result[key] = value
        return result
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs_hook,
                         parse_constant=reject_constant)


def exact_keys(value, keys, label):
    require(type(value) is dict and set(value) == set(keys),
            "{} key population drift".format(label))


def fraction_json(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def contained_file(base, raw, label):
    base = Path(base).resolve()
    candidate = (base / raw).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise ValueError("{} escapes source directory".format(label))
    require(candidate.is_file() and not candidate.is_symlink(),
            "{} is not a regular non-symlink file".format(label))
    return candidate


def validate_anchors():
    observed = {}
    for name, pair in sorted(ANCHORS.items()):
        path = HW_ROOT / pair[0]
        require(path.is_file() and not path.is_symlink() and
                sha256(path) == pair[1], "anchor drift: {}".format(name))
        observed[name] = pair[1]
    observed["independent_validator"] = sha256(Path(__file__).resolve())
    return observed


def bitmap_tables():
    popcount = [bin(value).count("1") for value in range(256)]
    spatial = 15 * 20
    period = spatial // math.gcd(8, spatial)
    weighted = []
    for phase in range(period):
        base = (phase * 8) % spatial
        row = []
        for value in range(256):
            destinations = 0
            for bit in range(8):
                if value & (1 << bit):
                    y, x = divmod((base + bit) % spatial, 20)
                    destinations += ((2 if y in (0, 14) else 3) *
                                     (2 if x in (0, 19) else 3))
            row.append(destinations)
        weighted.append(row)
    return period, popcount, weighted


def audit_record_payload(record, packed, period, popcount, weighted):
    require(record["shape"] == [10, 1, 768, 15, 20] and
            record["output_shape"] == [10, 1, 768, 15, 20] and
            record["elements"] == 2304000, "record shape/extent drift")
    geometry = record["module_geometry"]
    require(geometry["kernel_size"] == [3, 3] and
            geometry["stride"] == [1, 1] and
            geometry["padding"] == [1, 1] and
            geometry["dilation"] == [1, 1] and
            geometry["groups"] == 1 and
            geometry["in_channels"] == 768 and
            geometry["out_channels"] == 768 and
            geometry["bias_present"] is False,
            "record Conv3x3 geometry drift")
    plane_bytes = record["positive_plane_bytes"]
    require(plane_bytes == 288000 and len(packed) == 3 * plane_bytes ==
            record["packed_file_bytes"], "packed plane extent drift")
    positive = packed[:plane_bytes]
    negative = packed[plane_bytes:2 * plane_bytes]
    changed = packed[2 * plane_bytes:]
    require(not any(negative), "negative activity population is nonzero")
    bytes_per_timestep = plane_bytes // 10
    local_sources = local_pairs = motion_sources = motion_pairs = 0
    changed_bits_checked = 0
    for timestep in range(10):
        start = timestep * bytes_per_timestep
        previous_start = (timestep - 1) * bytes_per_timestep
        local_timestep = motion_timestep = 0
        for offset, current in enumerate(
                positive[start:start + bytes_per_timestep]):
            previous = (0 if timestep == 0 else
                        positive[previous_start + offset])
            delta = current ^ previous
            require(changed[start + offset] == delta,
                    "changed plane is not adjacent-timestep XOR")
            phase = offset % period
            local_sources += popcount[current]
            local_pairs += weighted[phase][current]
            motion_sources += popcount[delta]
            motion_pairs += weighted[phase][delta]
            local_timestep += popcount[current]
            motion_timestep += popcount[delta]
            changed_bits_checked += 8
        require(local_timestep ==
                record["local_nonzero_count_by_timestep"][timestep] and
                motion_timestep ==
                record["motion_numeric_transition_count_by_timestep"][timestep],
                "per-timestep support count drift")
    require(local_sources == record["nonzero_count"] ==
            record["positive_count"], "Local source conservation drift")
    return {
        "local_sources": local_sources,
        "local_pairs": local_pairs,
        "motion_sources": motion_sources,
        "motion_pairs": motion_pairs,
        "changed_bits_checked": changed_bits_checked,
    }


def independent_trace_rebuild():
    manifest_path = HW_ROOT / ANCHORS["m40_packed_source_manifest"][0]
    source_dir = manifest_path.parent
    manifest = read_json(manifest_path)
    require(manifest["schema"] == "m40_bottleneck_packed_source_trace_v1" and
            manifest["cohort"]["samples"] == 10 and
            manifest["cohort"]["records"] == 40 and
            manifest["cohort"]["operators"] == list(TARGETS),
            "M40 r6 cohort drift")
    period, popcount, weighted = bitmap_tables()
    seen = set()
    local_work = [0] * 10
    motion_work = [0] * 10
    local_sources = [0] * 10
    motion_sources = [0] * 10
    local_pairs = [0] * 10
    motion_pairs = [0] * 10
    changed_bits = 0
    packed_files_verified = 0
    for record in manifest["records"]:
        key = (record["sample_id"], record["operator"])
        require(key not in seen and key[0] in range(10) and
                key[1] in TARGETS, "record identity/population drift")
        seen.add(key)
        path = contained_file(source_dir, record["packed_file"], "packed file")
        require(sha256(path) == record["packed_file_sha256"],
                "packed file SHA drift")
        packed = path.read_bytes()
        row = audit_record_payload(record, packed, period, popcount, weighted)
        sample = key[0]
        local_sources[sample] += row["local_sources"]
        motion_sources[sample] += row["motion_sources"]
        local_pairs[sample] += row["local_pairs"]
        motion_pairs[sample] += row["motion_pairs"]
        local_work[sample] += row["local_pairs"] * 8
        motion_work[sample] += row["motion_pairs"] * 8
        changed_bits += row["changed_bits_checked"]
        packed_files_verified += 1
    require(seen == set((sample, operator) for sample in range(10)
                        for operator in TARGETS), "40-record cohort incomplete")
    expected_local = [74514032, 73731376, 74254256, 73745520, 73417496,
                      73847856, 74995872, 73721232, 74040896, 74855240]
    expected_motion = [110550816, 109632536, 110434912, 109509648,
                       108971224, 109343896, 110962768, 109257448,
                       109652160, 110832144]
    require(local_work == expected_local and motion_work == expected_motion,
            "raw M40 work distribution drift")
    require(changed_bits == 92160000 and sum(local_sources) == 11010375 and
            sum(motion_sources) == 16343544 and
            sum(local_pairs) == 92640472 and
            sum(motion_pairs) == 137393444,
            "raw source/pair conservation drift")
    return {
        "records": 40,
        "unique_sample_operator_pairs": 40,
        "packed_file_shas_verified": packed_files_verified,
        "changed_plane_bits_recomputed": changed_bits,
        "changed_plane_bit_mismatches": 0,
        "local_active_sources_sum": sum(local_sources),
        "motion_active_sources_sum": sum(motion_sources),
        "local_source_destination_pairs_sum": sum(local_pairs),
        "motion_source_destination_pairs_sum": sum(motion_pairs),
        "local_product_count_div_96_work_quanta_per_sample": local_work,
        "motion_product_count_div_96_work_quanta_per_sample": motion_work,
        "local_work_quanta_sum": sum(local_work),
        "motion_work_quanta_sum": sum(motion_work),
        "local_mean_exact": fraction_json(Fraction(sum(local_work), 10)),
        "local_p95_nearest_rank": max(local_work),
        "local_p99_nearest_rank": max(local_work),
        "motion_mean_exact": fraction_json(Fraction(sum(motion_work), 10)),
        "motion_p95_nearest_rank": max(motion_work),
        "motion_over_local_exact": fraction_json(
            Fraction(sum(motion_work), sum(local_work))),
        "qualification": (
            "PRODUCT_COUNT_DIV_96_LOGICAL_WORK_QUANTA_NOT_EXECUTABLE_CYCLES"),
    }


def independent_p8_l96_geometry():
    engine = (HW_ROOT / ANCHORS["p8_l96_engine_rtl"][0]).read_text(
        encoding="utf-8")
    tops = (HW_ROOT / ANCHORS["p8_l96_dc_tops_rtl"][0]).read_text(
        encoding="utf-8")
    required_engine_fragments = (
        "logic [ISSUE_WIDTH-1:0]            weight_request_bank_valid",
        "logic [ISSUE_WIDTH*OUT_LANES*W_W-1:0] weight_response_data",
        "for (int bank = 0; bank < ISSUE_WIDTH; bank = bank + 1)",
        "for (int lane = 0; lane < OUT_LANES; lane = lane + 1)",
        "response_sum[lane] = response_sum[lane] + extend_weight(",
    )
    require(all(fragment in engine for fragment in required_engine_fragments),
            "P8-L96 engine issue/broadcast structure drift")
    require(".ISSUE_WIDTH(P), .OUT_LANES(96)" in tops and
            "qfit_local_banked_multisource_p8_l96_top, 8" in tops,
            "P8-L96 top binding drift")
    return {
        "issue_width_sources_per_cycle_maximum": 8,
        "output_lanes_per_source_broadcast": 96,
        "peak_int8_weight_additions_per_cycle": 768,
        "incorrect_total_96_products_per_cycle_interpretation_rejected": True,
        "geometry_is_peak_not_measured_utilization": True,
        "rtl_sha_bound_by_independent_review_not_m42_contract": True,
    }


def independent_budget_rebuild(trace):
    m39 = read_json(HW_ROOT / ANCHORS["m39_result"][0])
    require(m39["status"] ==
            "PASS_M39_R3_CURRENT_ANCHORS_CONDITIONAL_BOTTLENECK_DSE_ONLY" and
            m39["admission"]["system_speedup_admitted"] is False,
            "M39 model-only status drift")
    rows = [row for row in m39["conditional_dse"]["four_bottleneck_rows"]
            if row.get("line") == "Local" and
            row.get("late_scale_implementation") ==
            "M35_parallel_complement_CSD_sidecar"]
    require(len(rows) == 1, "M39 Local/M35 row population drift")
    row = rows[0]
    fixed = m39["conditional_dse"]["fixed_compute_reference_cycles"]
    before_model = row["m38_model_substituted_ideal_before_scope_cycles"]
    before_scope = row["before_cycles"]
    outside = before_model - before_scope
    late = row["replacement"]["late_scale_cycles"]
    frontend = row["replacement"]["proportional_frontend_control_cycles"]
    event = row["replacement"]["conditional_m4_projected_event_cycles"]
    replacement = row["replacement"]["total_cycles"]
    overhead = late + frontend
    require((fixed, before_model, before_scope, outside, late, frontend,
             event, replacement, overhead) ==
            (620868243, 268455448, 79630957, 188824491, 1152000,
             1484515, 13282496, 15919011, 2636515),
            "M39 frozen baseline/replacement arithmetic drift")
    require(replacement == event + overhead,
            "M39 replacement conservation drift")
    local_mean = Fraction(trace["local_work_quanta_sum"], 10)
    local_p95 = Fraction(trace["local_p95_nearest_rank"], 1)
    gates = []
    for target in (Fraction(5, 2), Fraction(27, 10), Fraction(3, 1)):
        total_ceiling = Fraction(fixed, 1) / target
        replacement_budget = total_ceiling - outside
        product_budget = replacement_budget - overhead
        required_mean = local_mean / product_budget
        required_p95 = local_p95 / product_budget
        gates.append({
            "target_compute_speedup": fraction_json(target),
            "maximum_executable_product_cycles_required": fraction_json(
                product_budget),
            "required_effective_source_issue_width_from_local_mean":
                fraction_json(required_mean),
            "required_effective_source_issue_width_from_local_p95":
                fraction_json(required_p95),
            "peak_issue_width_margin_from_local_mean": fraction_json(
                Fraction(8, 1) / required_mean),
            "target_crossing_admitted": False,
        })
    return {
        "fixed_compute_reference_cycles": fixed,
        "outside_four_bottleneck_model_cycles": outside,
        "fixed_late_scale_plus_frontend_cycles": overhead,
        "conditional_projected_event_work_quanta": event,
        "conditional_projected_replacement_total_cycles": replacement,
        "target_gates": gates,
        "required_width_is_parallel_source_issue_not_logical_work_reduction": True,
        "real_finite_bank_schedule_admitted": False,
        "system_speedup_admitted": False,
    }


def validate_candidate_result(result, trace, budget):
    exact_keys(result, {
        "schema", "status", "identity", "frozen_resource_model",
        "independently_reviewed_real_work",
        "non_executable_diagnostic_envelopes", "target_gates", "admission",
        "required_next_gate", "claim_boundary"}, "M42 result")
    require(result["schema"] == "m42_real_work_headroom_gate_result_v1" and
            result["status"] ==
            "PASS_M42_EXACT_PERFORMANCE_BUDGETS_ONLY_REAL_EXECUTABLE_SCHEDULE_PENDING",
            "M42 schema/status drift")
    require(result["identity"] == {
        "contract_sha256": ANCHORS["contract"][1],
        "m39_result_sha256": ANCHORS["m39_result"][1],
        "m40_result_sha256": ANCHORS["m40_result"][1],
        "m40_independent_review_sha256": ANCHORS["m40_independent_review"][1],
    }, "M42 identity chain drift")
    resource = result["frozen_resource_model"]
    require(resource["fixed_compute_reference_cycles"] ==
            budget["fixed_compute_reference_cycles"] and
            resource["outside_four_bottleneck_model_cycles"] ==
            budget["outside_four_bottleneck_model_cycles"] and
            resource["fixed_late_scale_plus_frontend_cycles"] ==
            budget["fixed_late_scale_plus_frontend_cycles"] and
            resource["conditional_projected_event_work_quanta"] ==
            budget["conditional_projected_event_work_quanta"] and
            resource["conditional_projected_replacement_total_cycles"] ==
            budget["conditional_projected_replacement_total_cycles"] and
            resource["event_engine_issue_width"] == 8 and
            resource["event_engine_output_lanes"] == 96 and
            resource["event_engine_peak_product_adds_per_cycle"] == 768,
            "M42 frozen resource geometry drift")
    real = result["independently_reviewed_real_work"]
    require(real["qualification"] ==
            "PRODUCT_COUNT_DIV_96_LOWER_BOUND_WORK_QUANTA_NOT_EXECUTABLE_CYCLES" and
            real["local_mean"] == trace["local_mean_exact"] and
            real["local_p95"] == fraction_json(Fraction(
                trace["local_p95_nearest_rank"], 1)) and
            real["local_p99"] == real["local_p95"] and
            real["pure_motion_is_worse_on_this_cohort"] is True,
            "M42 real-work qualification/distribution drift")
    require(len(result["target_gates"]) == 3, "M42 target population drift")
    for observed, expected in zip(result["target_gates"],
                                  budget["target_gates"]):
        for key in ("target_compute_speedup",
                    "maximum_executable_product_cycles_required",
                    "required_effective_source_issue_width_from_local_mean",
                    "required_effective_source_issue_width_from_local_p95",
                    "peak_issue_width_margin_from_local_mean",
                    "target_crossing_admitted"):
            require(observed[key] == expected[key],
                    "M42 target gate drift: {}".format(key))
        require(observed["issue_width_peak"] == 8 and
                observed["real_executable_schedule_admitted"] is False,
                "M42 target admission/peak drift")
    admission = result["admission"]
    require(admission["exact_budget_math_admitted"] is True and
            admission["m39_m40_identity_chain_admitted"] is True and
            all(admission[key] is False for key in (
                "real_executable_schedule_admitted",
                "target_2p5_crossing_admitted", "target_2p7_crossing_admitted",
                "target_3p0_crossing_admitted", "system_speedup_admitted",
                "rtl_synopsys_ppa_power_energy_admitted",
                "headline_or_best_paper_admitted")),
            "M42 forbidden admission opened")
    require(result["non_executable_diagnostic_envelopes"]
            ["executable_or_system_metric_admitted"] is False and
            "system or end-to-end speedup" in
            result["claim_boundary"]["forbidden"] and
            "calling product-count-div-96 work quanta executable cycles without a finite-bank issue schedule"
            in result["claim_boundary"]["forbidden"],
            "M42 diagnostic/claim boundary drift")


def load_candidate():
    path = HW_ROOT / ANCHORS["analyzer"][0]
    spec = importlib.util.spec_from_file_location("m42_hammer_candidate", str(path))
    require(spec is not None and spec.loader is not None,
            "candidate module import failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate_regression_and_writer():
    test_path = HW_ROOT / ANCHORS["regression"][0]
    run = subprocess.run(
        ["/usr/bin/python3.6", "-m", "unittest", str(test_path)],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = (run.stdout + run.stderr).decode("utf-8", "replace")
    require(run.returncode == 0 and "Ran 8 tests" in output and
            "OK" in output, "M42 Python3.6 regression failed")
    analyzer = HW_ROOT / ANCHORS["analyzer"][0]
    with tempfile.TemporaryDirectory() as directory:
        first = Path(directory) / "first.json"
        second = Path(directory) / "second.json"
        for output_path in (first, second):
            proc = subprocess.run(
                ["/usr/bin/python3.6", str(analyzer), "--output",
                 str(output_path)], cwd=str(ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            require(proc.returncode == 0, "candidate output rebuild failed")
        require(first.read_bytes() == second.read_bytes() and
                hashlib.sha256(first.read_bytes()).hexdigest() ==
                ANCHORS["result"][1], "candidate result is not byte deterministic")
        before = first.read_bytes()
        occupied = subprocess.run(
            ["/usr/bin/python3.6", str(analyzer), "--output", str(first)],
            cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        require(occupied.returncode == 2 and first.read_bytes() == before,
                "candidate occupied output was overwritten")
    return {
        "python36_regression": {"passed": 8, "failed": 0, "errors": 0},
        "byte_deterministic_rebuilds": 2,
        "rebuilt_result_sha256": ANCHORS["result"][1],
        "occupied_output_attempts": 1,
        "occupied_output_rejected_without_mutation": True,
    }


def adversarial_rebuild(trace, budget):
    module = load_candidate()
    contract = read_json(HW_ROOT / ANCHORS["contract"][0])
    canonical_result = read_json(HW_ROOT / ANCHORS["result"][0])

    forged = copy.deepcopy(contract)
    forged["identity"]["m40_result"]["sha256"] = "0" * 64
    trace_identity_rejected = False
    try:
        module.build_result(forged)
    except module.AuditError:
        trace_identity_rejected = True
    require(trace_identity_rejected, "forged M40 identity was accepted")

    forged = copy.deepcopy(contract)
    forged["frozen_model"]["fixed_compute_reference_cycles"] += 1
    baseline_rejected = False
    try:
        module.build_result(forged)
    except module.AuditError:
        baseline_rejected = True
    require(baseline_rejected, "forged fixed baseline was accepted")

    tampered = copy.deepcopy(canonical_result)
    tampered["frozen_resource_model"]["event_engine_issue_width"] = 96
    result_rejected = False
    try:
        validate_candidate_result(tampered, trace, budget)
    except ValueError:
        result_rejected = True
    require(result_rejected, "tampered result was accepted")

    promoted = copy.deepcopy(canonical_result)
    promoted["independently_reviewed_real_work"]["qualification"] = (
        "MEASURED_EXECUTABLE_SYSTEM_CYCLES")
    promoted["admission"]["system_speedup_admitted"] = True
    promoted["target_gates"][2]["target_crossing_admitted"] = True
    promotion_rejected = False
    try:
        validate_candidate_result(promoted, trace, budget)
    except ValueError:
        promotion_rejected = True
    require(promotion_rejected,
            "logical work factor was promoted to speedup without rejection")

    manifest_path = HW_ROOT / ANCHORS["m40_packed_source_manifest"][0]
    manifest = read_json(manifest_path)
    record = copy.deepcopy(manifest["records"][0])
    source_dir = manifest_path.parent
    packed = bytearray((source_dir / record["packed_file"]).read_bytes())
    packed[2 * record["positive_plane_bytes"]] ^= 1
    record["packed_file_sha256"] = hashlib.sha256(bytes(packed)).hexdigest()
    period, popcount, weighted = bitmap_tables()
    raw_trace_rejected = False
    try:
        audit_record_payload(record, bytes(packed), period, popcount, weighted)
    except ValueError:
        raw_trace_rejected = True
    require(raw_trace_rejected, "self-consistent-SHA changed-plane forgery accepted")

    broadened = copy.deepcopy(contract)
    broadened["claim_policy"]["admitted"].append("system speedup")
    broadened["claim_policy"]["forbidden"] = []
    broadened_result = module.build_result(broadened)
    require("system speedup" in broadened_result["claim_boundary"]["admitted"],
            "expected direct build_result policy weakness disappeared")
    pinned_policy_rejected = False
    try:
        require(hashlib.sha256(json.dumps(
            broadened, sort_keys=True).encode("utf-8")).hexdigest() ==
                ANCHORS["contract"][1], "canonical contract SHA mismatch")
    except ValueError:
        pinned_policy_rejected = True
    require(pinned_policy_rejected,
            "independent contract pin did not reject policy broadening")
    return {
        "m40_trace_identity_mutation": {"tested": 1, "rejected": 1},
        "raw_changed_plane_mutation_with_updated_payload_sha": {
            "tested": 1, "rejected": 1},
        "fixed_baseline_mutation": {"tested": 1, "rejected": 1},
        "tampered_result_geometry": {"tested": 1, "rejected": 1},
        "logical_work_factor_promoted_to_system_speedup": {
            "tested": 1, "rejected_by_independent_validator": 1},
        "claim_policy_broadening_direct_build_result": {
            "tested": 1, "accepted_by_candidate_build_result": 1,
            "rejected_by_independent_contract_sha_pin": 1},
    }


def validate_review(review_path=DEFAULT_REVIEW):
    anchors = validate_anchors()
    trace = independent_trace_rebuild()
    geometry = independent_p8_l96_geometry()
    budget = independent_budget_rebuild(trace)
    candidate = read_json(HW_ROOT / ANCHORS["result"][0])
    validate_candidate_result(candidate, trace, budget)
    module = load_candidate()
    require(module.build_result(read_json(HW_ROOT / ANCHORS["contract"][0])) ==
            candidate, "candidate in-memory result rebuild drift")
    regression = candidate_regression_and_writer()
    attacks = adversarial_rebuild(trace, budget)

    review = read_json(review_path)
    exact_keys(review, {
        "schema", "status", "date", "exact_anchors", "validator",
        "mandatory_rereview_passes", "findings", "review", "admitted",
        "claim_boundary", "next_gate"}, "review")
    require(review["schema"] == "m42_r1_independent_hammer_review_v1" and
            review["status"] == "GO_M42_R1_EXACT_HEADROOM_GATE_ONLY",
            "review schema/status drift")
    for name, pair in ANCHORS.items():
        require(review["exact_anchors"][name] == [pair[0], pair[1]],
                "review anchor drift: {}".format(name))
    require(review["validator"] == [
        str(Path(__file__).resolve().relative_to(ROOT)),
        anchors["independent_validator"]], "review validator identity drift")
    passes = review["mandatory_rereview_passes"]
    require(passes["independent_raw_m40_r6_rebuild"] == trace and
            passes["independent_m39_m42_budget_rebuild"] == budget and
            passes["independent_p8_l96_geometry_audit"] == geometry and
            passes["candidate_regression_and_writer"] == regression and
            passes["adversarial_matrix"] == attacks,
            "review independent pass receipt drift")
    require(review["review"] == {
        "decision": "GO_EXACT_HEADROOM_GATE_ONLY",
        "independent_of_m42_implementation_for_core_math": True,
        "score_0_to_100": 94,
        "p0": 0, "p1": 0, "p2": 4,
        "pass_admission_may_be_generated": True,
    }, "review decision/score drift")
    require(not review["findings"]["p0"] and
            not review["findings"]["p1"] and
            len(review["findings"]["p2"]) == 4,
            "review finding population drift")
    admitted = review["admitted"]
    require(admitted["exact_sha_bound_headroom_budget_math"] is True and
            admitted["p8_l96_peak_geometry"] is True and
            admitted["required_effective_source_issue_width_gates"] is True and
            all(admitted[key] is False for key in (
                "real_finite_bank_schedule", "measured_target_crossing",
                "system_or_end_to_end_speedup", "memory_timing_closure",
                "integrated_rtl_vcs_synopsys", "ppa_power_energy",
                "date_headline_or_best_paper")),
            "review claim admission drift")
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()
    review = validate_review(args.review.resolve())
    print("PASS {} score={}".format(
        review["review"]["decision"], review["review"]["score_0_to_100"]))


if __name__ == "__main__":
    main()
