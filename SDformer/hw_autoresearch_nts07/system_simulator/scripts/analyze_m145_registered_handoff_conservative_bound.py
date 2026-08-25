#!/usr/bin/env python3
"""Charge a conservative registered-control bound onto M143r2.

This audit does not claim a new cycle-exact engine model.  It starts from the
frozen M143r2 same-clock recurrence and serially charges every control edge
that the M142 independent hammer proved was absent: fill-to-PWP launch,
PWP completion-to-next-launch, PWP-to-correction launch, correction
completion-to-next-launch, zero-work endpoint floors, and explicit outer
barrier accept/commit-done handshakes.  Serial addition deliberately prevents
overlap from hiding any of these control costs, so the result is a conservative
candidate-cycle upper bound for this control recurrence and a speedup lower
bound against the same frozen baselines.  SRAM and engine arithmetic remain
outside the bound.
"""

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
PATHS = {
    "m143_result": HW / "results/m143r2_raw128_full_materialized_overlap_dse_r1_20260824/m143_raw128_full_materialized_overlap_dse.json",
    "m143_contract": HW / "contracts/m143r2_raw128_full_materialized_overlap_dse_contract_r1_20260824.json",
    "m142_review_manifest": HW / "results/m142_independent_hammer_review_r1_20260824/manifest.sha256",
    "m142_review_overlay": HW / "contracts/m142_independent_review_correction_overlay_r1_20260824.json",
    "m142_dc_receipt": HW / "dc_handoff/runs/m142_raw128_k4_bounded_overlap_controller_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt",
    "m144_vcs_contract": HW / "contracts/m144r2_sequence_fenced_raw128_overlap_wrapper_vcs_contract_r1_20260824.json",
    "m144_vcs_receipt": HW / "dc_handoff/runs/m144r2_sequence_fenced_raw128_overlap_wrapper_vcs_r1_sealed_20260824/RUN_COMPLETE.txt",
    "m144_dc_contract": HW / "contracts/m144r2_sequence_fenced_raw128_overlap_wrapper_logic_only_dc_contract_r1_20260824.json",
    "m144_dc_receipt": HW / "dc_handoff/runs/m144r2_sequence_fenced_raw128_overlap_wrapper_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt",
}
EXPECTED = {
    "m143_result": "8b5821d747e653ac9053a4cfe94fe9eb40c78ce0eaaca4c9af4fdf8073b5bd19",
    "m143_contract": "288f03c77556c3e9ea26bfeb18e457423e8f8d8c3dfac9bef070769436051413",
    "m142_review_manifest": "336b8b205e81344bb692948201565da9fe1e327b855fd652045e6f29ff756679",
    "m142_review_overlay": "9667c026b0dddd6eabfe6743087938d3855cdae98c6cfe16ef3a71ecb73ee929",
    "m142_dc_receipt": "3e6f6fabc2b4fdd686f54a57a6a724e451e402ede138486a3ed2ff87a6f0fef6",
    "m144_vcs_contract": "d6d807fe0f71da20bbb87d21975ffc1147dc59f6c9987ab80aa64ee79b34c40f",
    "m144_vcs_receipt": "a99295aa36f847a75ddef753fa66d3f3c08920bda3a6d22a9ce8ff15d187218b",
    "m144_dc_contract": "891be1fff7a76dc22a0e16da81433dcf234734e8afb9c0cffef1bb62e6cc2883",
    "m144_dc_receipt": "0b852e4061e90a383c670858a413cd493fc65c76cdd29b8f36cca8622b68c050",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: " + raw)

    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook,
                      parse_constant=reject)


def parse_receipt(path):
    result = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        key, value = raw.split("=", 1)
        require(key not in result, "duplicate receipt key: " + key)
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M145 output overwrite")
    script_sha = sha256(Path(__file__).resolve())
    observed = {label: sha256(path) for label, path in PATHS.items()}
    require(observed == EXPECTED, "M145 frozen input identity drift")

    m143 = strict_json(PATHS["m143_result"])
    contract = strict_json(PATHS["m143_contract"])
    m142_dc = parse_receipt(PATHS["m142_dc_receipt"])
    m144_vcs = parse_receipt(PATHS["m144_vcs_receipt"])
    m144_dc = parse_receipt(PATHS["m144_dc_receipt"])
    require(m144_vcs["status"].startswith("PASS_M144R2"),
            "M144r2 VCS receipt not admitted")
    require(m144_dc["status"].startswith("PASS_M144R2"),
            "M144r2 DC receipt not admitted")
    require(m144_dc["setup_worst_slack_ns"] == "0.0019",
            "M144r2 setup identity drift")

    b4 = m143["raw128_cycle_models"]["b4"]
    units = int(b4["descriptors"])
    zero_pwp = int(b4["zero_pwp_descriptors"])
    zero_correction = int(b4["zero_correction_descriptors"])
    barriers = int(b4["accumulator_pipeline_flush_cycles"])
    require((units, zero_pwp, zero_correction, barriers)
            == (69120, 1332, 300, 160), "M143 extent drift")

    charges = {
        "fill_complete_to_pwp_launch_edges": units,
        "pwp_completion_to_next_launch_edges": units,
        "pwp_to_correction_launch_edges": units,
        "correction_completion_to_next_launch_edges": units,
        "zero_pwp_minimum_service_cycles": zero_pwp,
        "zero_correction_minimum_service_cycles": zero_correction,
        "outer_barrier_accept_edges": barriers,
        "outer_commit_done_edges": barriers,
    }
    total_charge = sum(charges.values())
    require(total_charge == 278432, "registered control charge drift")

    bounded = {}
    for banks in (2, 3, 4):
        base = int(m143["raw128_cycle_models"]["b{}".format(banks)]
                   ["candidate_cycles"])
        bounded["b{}".format(banks)] = {
            "m143r2_base_cycles": base,
            "serial_registered_control_charge_cycles": total_charge,
            "conservative_candidate_cycle_upper_bound":
                base + total_charge,
        }
    compact = int(contract["cycle_results"]
                  ["m132_compact256_serial_cycles"])
    dualrow = int(contract["cycle_results"]
                  ["m132_dualrow512_serial_cycles"])
    fixed8 = int(b4["fair_fixed8_baseline_cycles"])
    b3_bound = bounded["b3"]["conservative_candidate_cycle_upper_bound"]
    b4_bound = bounded["b4"]["conservative_candidate_cycle_upper_bound"]
    comparisons = {
        "b3_speedup_lower_bound_vs_compact256": compact / b3_bound,
        "b4_speedup_lower_bound_vs_compact256": compact / b4_bound,
        "b4_speedup_lower_bound_vs_dualrow512": dualrow / b4_bound,
        "b4_same_clock_service_island_ratio_lower_bound_vs_fixed8":
            fixed8 / b4_bound,
        "b4_ratio_vs_b3_under_equal_serial_charge": b3_bound / b4_bound,
    }

    base_area = Decimal(m142_dc["cell_area_um2"])
    closure_area = Decimal(m144_dc["cell_area_um2"])
    resources = {
        "m142_cell_area_um2": str(base_area),
        "m144r2_integrated_cell_area_um2": str(closure_area),
        "sequence_barrier_closure_area_delta_um2":
            str(closure_area - base_area),
        "m142_cell_count": int(m142_dc["cell_count"]),
        "m144r2_integrated_cell_count": int(m144_dc["cell_count"]),
        "sequence_barrier_closure_cell_delta":
            int(m144_dc["cell_count"]) - int(m142_dc["cell_count"]),
        "m142_sequential_cells": int(m142_dc["sequential_cells"]),
        "m144r2_integrated_sequential_cells":
            int(m144_dc["sequential_cells"]),
        "sequence_barrier_closure_sequential_delta":
            int(m144_dc["sequential_cells"])
            - int(m142_dc["sequential_cells"]),
        "m144r2_setup_slack_ns": m144_dc["setup_worst_slack_ns"],
        "m144r2_hold_slack_ns": m144_dc["hold_worst_slack_ns"],
        "macro_count": int(m144_dc["macro_count"]),
    }

    payload = {
        "schema": "m145_registered_handoff_conservative_bound_v1",
        "status": "PASS_CONSERVATIVE_REGISTERED_CONTROL_BOUND",
        "identity": {
            "analyzer_start_end_sha256": script_sha,
            "frozen_inputs_sha256": observed,
        },
        "exact_extent": {
            "descriptor_units": units,
            "zero_pwp_units": zero_pwp,
            "zero_correction_units": zero_correction,
            "outer_barriers": barriers,
        },
        "serial_control_charges": charges,
        "serial_control_charge_total_cycles": total_charge,
        "cycle_bounds": bounded,
        "comparisons": comparisons,
        "synthesized_control_resources": resources,
        "bound_semantics": {
            "all_control_charges_forced_serial": True,
            "overlap_credit_for_new_control_edges": False,
            "candidate_cycles_are_upper_bound_over_m143r2_control_recurrence": True,
            "reported_speedups_are_lower_bounds_against_same_frozen_baselines": True,
        },
        "model_boundary": {
            "m144r2_vcs_and_dc_sealed": True,
            "registered_engine_arithmetic_cycle_exact": False,
            "descriptor_result_sram_macro": False,
            "matched_frequency_baseline": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
        "paper_safe_statement": (
            "Charging every missing registered handoff and barrier edge "
            "serially gives a B4 control-recurrence upper bound of {} cycles "
            "and same-baseline speedup lower bounds of {:.6f}x versus "
            "compact256 and {:.6f}x versus dualrow512; engines, SRAM, and "
            "matched physical baselines remain outside this bound."
        ).format(b4_bound,
                 comparisons["b4_speedup_lower_bound_vs_compact256"],
                 comparisons["b4_speedup_lower_bound_vs_dualrow512"]),
    }
    require(sha256(Path(__file__).resolve()) == script_sha,
            "M145 analyzer changed during execution")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "m145_registered_handoff_conservative_bound.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(
        "PASS M145 charge={} b2={} b3={} b4={} "
        "b4_vs_compact256={:.9f}x b4_vs_dualrow512={:.9f}x "
        "registered_control_bound=true engine_arithmetic=false "
        "sram_macro=false physical_speedup=false system_speedup=false "
        "headline=false".format(
            total_charge,
            bounded["b2"]["conservative_candidate_cycle_upper_bound"],
            b3_bound, b4_bound,
            comparisons["b4_speedup_lower_bound_vs_compact256"],
            comparisons["b4_speedup_lower_bound_vs_dualrow512"]),
        flush=True)


if __name__ == "__main__":
    main()
