#!/usr/bin/env python3
"""Reconcile the admitted M31/M37/M38 ATLIF module performance boundary."""

from __future__ import division

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def exact_fraction(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    contract = load_json(args.contract)
    require(contract.get("schema") ==
            "m243_atlif_decoupled_csd_module_performance_contract_v1",
            "contract schema drift")
    root = args.contract.resolve().parents[1]
    loaded = {}
    identities = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file(), "missing input: {}".format(path))
        observed = sha256(path)
        require(observed == spec["sha256"],
                "SHA drift for {}: {}".format(name, observed))
        identities[name] = {"path": spec["path"], "sha256": observed}
        if path.suffix == ".json":
            loaded[name] = load_json(path)

    m31 = loaded["m31_vcs_admission"]
    m37 = loaded["m37_synopsys_review"]
    m38 = loaded["m38_reachable_cycle_model"]
    m39 = loaded["m39_frozen_cycle_population"]

    require(m31["status"] == "PASS_EXACT_FROZEN_M31_R4_STATIC_PHASE_VCS_ONLY",
            "M31 status drift")
    require(m31["observed"]["conditional_t10_no_stall_accept_ii"] == 10,
            "M31 T10 II drift")
    require(m31["observed"]["sole_source_multiplier_pool_instances"] == 1 and
            m31["observed"]["source_multiplier_slots"] == 96,
            "M31 multiplier-pool identity drift")

    require(m37["status"] == "PASS_INDEPENDENT_HAMMER_STANDALONE_LOGIC_ONLY",
            "M37 status drift")
    require(m37["review_score_0_to_100"] == 94 and
            m37["p0_count"] == 0 and m37["p1_count"] == 0,
            "M37 independent review drift")
    dc = m37["independently_recomputed_dc_sta"]
    fm = m37["independently_recomputed_formality"]
    require(dc["clock_period_ns"] == 3.0 and
            dc["independent_physical_multiplier_hit_count"] == 0 and
            dc["macro_or_blackbox_cell_count"] == 0 and
            dc["setup_wns_ns"] >= 0 and dc["hold_wns_ns"] >= 0,
            "M37 DC/STA admission drift")
    require(fm["verification_succeeded_terminal_count"] == 1 and
            fm["failing_compare_points"] == 0 and
            fm["aborted_compare_points"] == 0 and
            fm["unverified_compare_points"] == 0 and
            fm["unmatched_compare_points"] == 0,
            "M37 Formality admission drift")

    require(m38["status"] ==
            "PASS_M38_R5_TYPE_STRICT_MATH_PROTOCOL_COMPLETE_REACHABLE_STATE_ONLY",
            "M38 status drift")
    theory = m38["conditional_theory"]
    require(theory["serialized_steady_ii"] == 10 and
            theory["parallel_steady_ii"] == 5 and
            theory["finite_n_ratio"] == "10*N/(5+5*N)",
            "M38 cycle recurrence drift")
    require(m38["finite_reachable_state_audit"]["reachable_states"] == 669 and
            m38["finite_reachable_state_audit"]["no_overflow_holds"] is True and
            m38["finite_reachable_state_audit"]["single_writer_holds"] is True,
            "M38 reachable-state proof drift")

    population = contract["frozen_population"]
    n_tiles = population["t10_tiles"]
    serial_cycles = population["serial_cycles_per_tile"] * n_tiles
    decoupled_cycles = (population["decoupled_startup_cycles"] +
                        population["decoupled_steady_cycles_per_tile"] * n_tiles)
    module_speedup = Fraction(serial_cycles, decoupled_cycles)
    fixed_cycles = population["fixed_compute_reference_cycles"]

    m39_sub = m39["conditional_dse"]["m38_r5_model_only_t10_substitution"]
    require(m39_sub["old_t10_cycles"] == serial_cycles and
            m39_sub["conditional_model_t10_ii"] == 5 and
            m39_sub["conditional_model_t10_cycles"] == 5 * n_tiles,
            "M39 frozen T10 population drift")
    require(m39["conditional_dse"]["fixed_compute_reference_cycles"] == fixed_cycles,
            "M39 fixed-compute reference drift")

    conditional_total_after = fixed_cycles - serial_cycles + decoupled_cycles
    conditional_fixed_compute_speedup = Fraction(fixed_cycles,
                                                 conditional_total_after)
    tile_share = Fraction(serial_cycles, fixed_cycles)

    result = {
        "schema": "m243_atlif_decoupled_csd_module_performance_v1",
        "status": "PASS_ATLIF_FINITE_POPULATION_MODULE_SPEEDUP_PENDING_MATCHED_AREA_BASELINE",
        "identity": identities,
        "architecture": {
            "serial_reference": "one shared 96-slot signed-INT8 multiplier pool; T10 reduction and reconstruction serialized",
            "candidate": "phase-decoupled T10 reduction plus zero-multiplier CSD4 reconstruction sidecar",
            "candidate_logic_only_area_um2": dc["total_cell_area_um2"],
            "candidate_setup_wns_ns": dc["setup_wns_ns"],
            "candidate_hold_wns_ns": dc["hold_wns_ns"],
            "candidate_mapped_multiplier_count": 0,
            "candidate_formality_passing_points": fm["passing_compare_points"]
        },
        "finite_population_cycles": {
            "t10_tiles": n_tiles,
            "serial_cycles": serial_cycles,
            "decoupled_cycles_including_startup": decoupled_cycles,
            "cycles_saved": serial_cycles - decoupled_cycles,
            "module_speedup_exact": exact_fraction(module_speedup),
            "module_speedup": float(module_speedup),
            "asymptotic_speedup": 2.0,
            "reachable_states_exhaustively_checked": 669
        },
        "conditional_cross_population_context_only": {
            "fixed_compute_reference_cycles": fixed_cycles,
            "t10_cycle_share_exact": exact_fraction(tile_share),
            "t10_cycle_share": float(tile_share),
            "fixed_compute_speedup_if_only_this_substitution_exact":
                exact_fraction(conditional_fixed_compute_speedup),
            "fixed_compute_speedup_if_only_this_substitution":
                float(conditional_fixed_compute_speedup),
            "system_speedup_admitted": False
        },
        "decision": {
            "atlif_performance_direction": "KEEP",
            "innovation_candidate": "multiplierless CSD reconstruction overlapped with the sole shared multiplier phase",
            "next_synopsys_gate": "synthesize an exact matched serial T10 baseline with the same T10 ports/state boundary, then report (cycles/s)/logic-area for serial versus decoupled",
            "do_not_add_more_datapath_rtl_before_gate": True
        },
        "admission": contract["claim_boundary"],
        "claim_boundary": "The nearly 2x result is a finite-population ATLIF T10 module cycle-model result backed by admitted VCS, complete finite-state recurrence, and standalone CSD-sidecar DC/STA/Formality. It is not trained accuracy, matched-area throughput, integrated RTL, power/energy, system speedup, paper PPA, or a headline result."
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "m243_atlif_decoupled_csd_module_performance_r1.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("M243_PASS module_speedup={:.9f} conditional_fixed={:.9f}".format(
        float(module_speedup), float(conditional_fixed_compute_speedup)))


if __name__ == "__main__":
    main()
