#!/usr/bin/env python3
"""Seal M482 VCS evidence and apply its validated recurrence to M481.

The VCS workload is directed and frequency-compressed.  It is not a literal
391M-descriptor VCS replay.  Equivalence comes from the clean recurrence
asserted in the testbench for all 255 nonzero context masks and additional
pseudo-random orders: tile_cycles = bank_rounds + chunk_count + 39.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path


EXPECTED = {
    "m481_result": "2a7a1c917cb2f9aa1adb61092c7619de8d9b495aab5550f1fa41291188006578",
    "m481_seal": "fe323dc43a90b2fa33d23fb15c2eb55289b6685819da17c7b581ea340a846713",
    "contract": "26c703cf972765665a7cfd7ff290a7796452e048c9e87f3ec817e5e1eef95888",
    "m483_review": "eb60ea57fa065c73587ed2d2d3a315fcc0feb6c5563d1abb6abac282e1316a53",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
PASS_RE = re.compile(
    r"PASS M482 groups=(?P<groups>\d+) "
    r"all255_factor_cycles=(?P<factor_cycles>\d+) "
    r"all255_sparse_cycles=(?P<sparse_cycles>\d+) "
    r"all255_ratio=(?P<ratio>[0-9.]+) "
    r"factor_rounds=(?P<factor_rounds>\d+) "
    r"sparse_rounds=(?P<sparse_rounds>\d+) "
    r"attacks=(?P<attacks>\d+) empty=(?P<empty>\d+) "
    r"latency_checks=(?P<latency_checks>\d+) "
    r"stalls=(?P<factor_stalls>\d+),(?P<weight_stalls>\d+),"
    r"(?P<bank_stalls>\d+),(?P<commit_stalls>\d+) "
    r"commits=(?P<commits>\d+)")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ratio(numerator, denominator):
    require(float(denominator) > 0.0, "zero denominator")
    return float(numerator) / float(denominator)


def component_view(point, view):
    comp = point["cycle_components"]
    fixed = sum(value for key, value in comp.items()
                if key not in ("factor_fill", "weight_fill",
                               "accumulator_update_drain"))
    factor = int(comp["factor_fill"])
    weight = int(comp["weight_fill"])
    update = int(comp["accumulator_update_drain"])
    if view == "serial":
        cycles = fixed + factor + weight + update
    elif view == "factor_weight_parallel":
        cycles = fixed + max(factor, weight) + update
    elif view == "full_overlap_analytical":
        cycles = fixed + max(factor, weight, update)
    else:
        raise RuntimeError("bad view {}".format(view))
    return {"fixed_lifecycle_cycles": fixed, "factor_fill_cycles": factor,
            "weight_fill_cycles": weight, "update_drain_cycles": update,
            "cycles": cycles}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m481-result", required=True, type=Path)
    parser.add_argument("--m481-seal", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--m483-review", required=True, type=Path)
    parser.add_argument("--docs359", required=True, type=Path)
    parser.add_argument("--vcs-log", required=True, type=Path)
    parser.add_argument("--assert-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    identities = {
        "m481_result": sha256(args.m481_result),
        "m481_seal": sha256(args.m481_seal),
        "contract": sha256(args.contract),
        "m483_review": sha256(args.m483_review),
        "docs359": sha256(args.docs359),
    }
    require(identities == EXPECTED, "frozen input identity drift")
    m481 = json.loads(args.m481_result.read_text())
    log = args.vcs_log.read_text()
    matches = list(PASS_RE.finditer(log))
    require(len(matches) == 1, "missing or duplicate M482 PASS line")
    vcs = dict((key, int(value) if key != "ratio" else float(value))
               for key, value in matches[0].groupdict().items())
    require(vcs["groups"] == 8 and vcs["attacks"] == 1
            and vcs["empty"] == 1 and vcs["latency_checks"] == 3828,
            "directed VCS population drift")
    for key in ("factor_stalls", "weight_stalls", "bank_stalls",
                "commit_stalls"):
        require(vcs[key] > 0, "missing {} coverage".format(key))
    require(vcs["factor_rounds"] == 652
            and vcs["sparse_rounds"] == 1024,
            "all-255 round count drift")
    # all255 means one source for each mask 1..255, not an all-ones mask.
    require(vcs["factor_cycles"] - vcs["factor_rounds"] - 16 == 39,
            "candidate recurrence constant drift")
    require(vcs["sparse_cycles"] - vcs["sparse_rounds"] - 16 == 39,
            "baseline recurrence constant drift")

    report = args.assert_report.read_text()
    covers = ["cp_full_credit", "cp_dual_bank", "cp_factor_weight_overlap",
              "cp_weight_update_overlap", "cp_triple_overlap",
              "cp_factor_stall", "cp_weight_stall", "cp_bank_stall",
              "cp_commit_stall", "cp_same_bank_rdw",
              "cp_same_address_forward", "cp_conflict", "cp_fault",
              "cp_done", "cp_empty"]
    cover_matches = {}
    for cover in covers:
        found = re.search(r"{}.*,\s+\d+ attempts,\s+(\d+) match".format(cover),
                          report)
        require(found is not None and int(found.group(1)) > 0,
                "missing cover {}".format(cover))
        cover_matches[cover] = int(found.group(1))
    require(not re.search(r"failed at|Offending|^Error|Fatal:", report,
                          flags=re.MULTILINE), "assertion failure found")

    point = next(row for row in m481["points"]
                 if row["point_id"] == "L96_F2_C16_B2")
    views = {}
    for view in ("serial", "factor_weight_parallel",
                 "full_overlap_analytical"):
        baseline = component_view(point["baseline"], view)
        candidate = component_view(point["candidate"], view)
        views[view] = {"baseline": baseline, "candidate": candidate,
                       "same_resource_ratio": ratio(baseline["cycles"],
                                                     candidate["cycles"])}

    trace = m481["aggregate_trace"]
    nonempty = int(trace["nonempty_group_streams"])
    empty = int(trace["empty_group_streams"])
    directory = int(trace["chunks"]["16"]
                    ["directory_chunk_streams_nonempty_groups"])
    baseline_rounds = int(point["baseline"]["accumulator_update_issue_rounds"])
    candidate_rounds = int(point["candidate"]["accumulator_update_issue_rounds"])
    fixed = nonempty * 39 + empty * 2
    baseline_cycles = baseline_rounds + directory + fixed
    candidate_cycles = candidate_rounds + directory + fixed
    rtl_ratio = ratio(baseline_cycles, candidate_cycles)
    require(abs(rtl_ratio - 1.3598966734564581) < 1e-12,
            "exact recurrence result drift")

    scope = m481["scope_partition"]
    eligible = int(scope["eligible_binary_fc1_cycles"])
    fallback = int(scope["excluded_stage3_nonbinary_fc1_cycles"])
    envelope = int(scope["compute_envelope_cycles"])
    projected_eligible = float(eligible) / rtl_ratio
    ideal_envelope = ratio(envelope,
                           float(envelope - eligible) + projected_eligible)

    # Sensitivities only.  M481 has the exact F2/B4 and F4/B4 issue rounds,
    # but neither geometry is implemented in M482 RTL.
    f2b4 = next(row for row in m481["points"]
                if row["point_id"] == "L96_F2_C16_B4")
    f4b4 = next(row for row in m481["points"]
                if row["point_id"] == "L96_F4_C16_B4")
    p1 = {}
    for name, row in (("L96_F2_C16_B4", f2b4),
                      ("L96_F4_C16_B4", f4b4)):
        rounds = int(row["candidate"]["accumulator_update_issue_rounds"])
        cycles = rounds + directory + fixed
        p1[name] = {"candidate_cycles_if_same_recurrence": cycles,
                    "ratio_if_same_recurrence_not_admitted":
                        ratio(baseline_cycles, cycles),
                    "rtl_implemented": False}
    f2_candidate_cycles = candidate_cycles
    f4_candidate_cycles = p1["L96_F4_C16_B4"][
        "candidate_cycles_if_same_recurrence"]
    f4_local_gain = ratio(f2_candidate_cycles, f4_candidate_cycles)
    require(abs(f4_local_gain - 1.2478348246622157) < 1e-12,
            "F2 to F4 local gain drift")
    f4_area_prescreen = {
        "from": "L96_F2_C16_B2",
        "to": "L96_F4_C16_B4",
        "local_cycle_throughput_gain_if_same_recurrence_not_admitted":
            f4_local_gain,
        "lane_adders": {"from": 192, "to": 384,
                         "increase": "2.0x"},
        "accumulator_banks": {"from": 2, "to": 4,
                               "increase": "2.0x"},
        "accumulator_read_write_ports": {"from": "2R2W", "to": "4R4W",
                                           "increase": "2.0x"},
        "maximum_total_area_growth_for_non_decreasing_throughput_per_mm2":
            f4_local_gain,
        "linear_dominant_adder_port_area_proxy_not_physical_mm2":
            f4_local_gain / 2.0,
        "physical_throughput_per_mm2_available": False,
        "preaudit": "NO_GO_AUTOMATIC_F4_RTL_OR_DC",
        "reason": "Only 1.248x local cycle throughput is purchased with 2x lane adders and 2x Acc banks/ports; no physical area exists to rescue throughput/mm2.",
        "m483_bound": "M483 explicitly forbids opening F4/C64 RTL to rescue the headline after the compact point fails its physical/finite-queue gate.",
    }

    output = {
        "schema": "m482_fc1_l96_f2_c16_b2_full_overlap_vcs_recurrence_v1",
        "status": "PASS_VCS__P0_NO_GO_B2_BELOW_1P50__NO_PERFORMANCE_ADMISSION",
        "identity": identities,
        "vcs": {"tool": "Synopsys VCS V-2023.12-SP1",
                "directed": vcs, "cover_matches": cover_matches,
                "assertion_failures": 0, "numeric_mismatches": 0,
                "transaction_mismatches": 0,
                "response_contract": "fixed two-cycle, in-order",
                "legal_response_reordering_supported": False,
                "protocol_attack": "wrong identity is quarantined; no reorder coverage is claimed",
                "all255_definition": "one representative source for each nonzero mask 1..255; not 255 copies of mask 0xff",
                "directed_synthetic_not_literal_trace_replay": True},
        "cycle_views": views,
        "rtl_exact_recurrence": {
            "proof_domain": "all 255 nonzero masks plus deterministic pseudo-random mask orders, both factorized and expanded onehot baseline",
            "clean_nonempty_tile_formula": "bank_aware_issue_rounds + chunk_count + 39",
            "empty_tile_cycles": 2,
            "frozen_trace_source": "M481 exact aggregates from 100 H67 ep35 records",
            "baseline_issue_rounds": baseline_rounds,
            "candidate_issue_rounds": candidate_rounds,
            "chunk_directory_cycles": directory,
            "nonempty_fixed_cycles": nonempty * 39,
            "empty_cycles": empty * 2,
            "baseline_cycles": baseline_cycles,
            "candidate_cycles": candidate_cycles,
            "same_resource_ratio": rtl_ratio,
            "is_literal_full_trace_vcs": False,
            "is_frequency_compressed_equivalent_recurrence": True,
        },
        "scope_corrected_projection": {
            "eligible_binary_fc1_baseline_cycles": eligible,
            "eligible_binary_fc1_projected_cycles": projected_eligible,
            "stage3_fallback_cycles_unchanged": fallback,
            "all_fc1_projected_cycles": projected_eligible + fallback,
            "compute_envelope_cycles": envelope,
            "ideal_envelope_sensitivity_not_speedup": ideal_envelope,
        },
        "p1_sensitivity_not_admitted": p1,
        "f4_throughput_per_area_preaudit": f4_area_prescreen,
        "decision": {
            "p0_l96_f2_c16_b2_ratio_ge_1p50": rtl_ratio >= 1.50,
            "p0_ideal_envelope_ge_1p08": ideal_envelope >= 1.08,
            "verdict": "NO_GO_L96_F2_C16_B2_AS_PERFORMANCE_POINT",
            "reason": "With a fair 128-entry factor descriptor buffer, full overlap exposes bank-aware issue rounds as the bottleneck; B2 is below 1.50x even before physical SRAM stalls.",
            "p1": "Do not spend DC on B2. F2/B4 alone remains below 1.50x under the same recurrence; evaluate F4/B4 or a batched/hot-restart lifecycle only after a same-resource CPU gate.",
        },
        "admission": {
            "directed_fullwidth_vcs": True,
            "all_255_mask_classes": True,
            "trace_backed_exact_recurrence": True,
            "literal_full_trace_vcs": False,
            "physical_sram_macro": False,
            "dc_sta": False,
            "saif_ptpx": False,
            "measured_performance": False,
            "complete_fc1": False,
            "complete_ffn": False,
            "system_speedup": False,
            "headline": False,
            "paper_ppa_ready": False,
        },
        "claim_boundary": {
            "serial_2p0186x_is_not_rtl_result": True,
            "directed_all255_1p526x_is_not_trace_result": True,
            "rtl_exact_recurrence_1p3599x_is_not_system_speedup": True,
            "m230_m262_ratios_multiplied": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print("PASS M482 exact recurrence ratio={:.9f} envelope={:.9f} verdict={}".
          format(rtl_ratio, ideal_envelope, output["decision"]["verdict"]))


if __name__ == "__main__":
    main()
