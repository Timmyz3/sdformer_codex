#!/usr/bin/env python3
"""Audit signed-INT8 CSD reconstruction and phase-decoupling sensitivity."""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT = ROOT / "hw_autoresearch_nts07/contracts/m37_phase_decoupled_csd_reconstruct_input_contract_r2_20260822.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(raw):
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_contract(path):
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    if contract.get("schema") != "m37_phase_decoupled_csd_reconstruct_input_contract_v2":
        raise ValueError("unexpected M37 contract")
    payloads = {}
    hashes = {}
    for name, spec in sorted(contract["inputs"].items()):
        source = resolve(spec["path"])
        actual = sha256(source)
        if actual != spec["sha256"]:
            raise ValueError("M37 input hash drift for {}".format(name))
        payloads[name] = json.loads(source.read_text(encoding="utf-8"))
        hashes[name] = actual
    return contract, payloads, hashes


def naf_terms(value):
    """Return a non-adjacent signed-power representation of signed INT8."""
    sign = -1 if int(value) < 0 else 1
    remaining = abs(int(value))
    terms = []
    shift = 0
    while remaining:
        if remaining & 1:
            coefficient = 2 - (remaining & 3)
            terms.append((sign * coefficient, shift))
            remaining -= coefficient
        remaining >>= 1
        shift += 1
    return terms


def reconstruct(terms):
    return sum(sign * (1 << shift) for sign, shift in terms)


def minimum_terms(value):
    powers = [(sign, shift) for shift in range(8) for sign in (-1, 1)]
    if value == 0:
        return 0
    for count in range(1, 5):
        for chosen in itertools.combinations(powers, count):
            if len({shift for _sign, shift in chosen}) != count:
                continue
            if reconstruct(chosen) == value:
                return count
    raise ValueError("signed INT8 value lacks a four-term descriptor: {}".format(value))


def find_m32_rows(m32):
    rows = m32["control_charged_cycle_sensitivity"]["rows"]
    selected = {}
    for row in rows:
        if row["variant"] == "byte12_arithmetic_lower_bound":
            selected[row["line"]] = row
    if set(selected) != {"local", "motion"}:
        raise ValueError("M37 M32 sensitivity row population drift")
    return selected


def build(contract_path=DEFAULT_CONTRACT):
    contract, payloads, hashes = load_contract(contract_path)
    m30, m32 = payloads["m30"], payloads["m32"]
    m31_contract = payloads["m31_vcs_contract"]
    m31_receipt = payloads["m31_vcs_receipt"]
    arch = contract["architecture"]
    if (m30["frozen_resources"]["signed_int8_multipliers"] != 96
            or m30["frozen_resources"]["t10_rank"] != 3
            or m30["frozen_resources"]["t10_product_issue_cycles_per_tile"] != 10):
        raise ValueError("M37 M30 exact96/rank3 resource identity drift")
    if not m31_receipt["status"].startswith("PASS_UNIFIED_T10_T2_EXACT_FIXED_POINT"):
        raise ValueError("M37 M31 VCS receipt is not admitted")
    if m31_receipt["contract"]["sha256"] != hashes["m31_vcs_contract"]:
        raise ValueError("M37 M31 receipt-to-contract hash drift")
    m31_schedule = m31_contract["t10_schedule"]
    if (m31_schedule["stage1_cycles_per_tile"] != 5
            or m31_schedule["stage2_cycles_per_tile"] != 5
            or m31_schedule["conditional_steady_ii_cycles"] != 10
            or m31_contract["rtl_resource_contract"]["signed_int8_multiplier_assignments_in_pool"] != 96):
        raise ValueError("M37 M31 VCS schedule/resource identity drift")

    distribution = {str(count): 0 for count in range(5)}
    coefficient_rows = []
    for value in range(-128, 128):
        terms = naf_terms(value)
        if reconstruct(terms) != value or len(terms) > 4:
            raise ValueError("M37 NAF construction failed")
        minimum = minimum_terms(value)
        if minimum != len(terms):
            raise ValueError("M37 NAF is not minimum for {}".format(value))
        distribution[str(len(terms))] += 1
        coefficient_rows.append({
            "value": value,
            "terms": [{"sign": sign, "shift": shift} for sign, shift in terms],
            "minimum_terms": minimum,
        })

    tiles = int(next(row for row in m30["port_candidates"]
        if row["name"] == "dual256b_independent_output_packed24")["t10_factor_tiles"])
    rows = int(arch["t10_rows"])
    reduction_cycles = int(math.ceil(rows / float(arch["pool_reduction_rows_per_cycle"])))
    reconstruction_cycles = int(math.ceil(rows / float(arch["csd_reconstruction_rows_per_cycle_target"])))
    serialized_per_tile = reduction_cycles + reconstruction_cycles
    overlapped_steady_ii = max(reduction_cycles, reconstruction_cycles)
    serialized_total = tiles * serialized_per_tile
    overlapped_total = reduction_cycles + tiles * overlapped_steady_ii
    if serialized_total != 73183500:
        raise ValueError("M37 source T10 cycle identity drift")
    saved = serialized_total - overlapped_total

    fixed = int(m32["control_charged_cycle_sensitivity"]["fixed_compute_cycles"])
    m32_rows = find_m32_rows(m32)
    sensitivity = []
    for line in ("local", "motion"):
        source = int(m32_rows[line]["control_charged_proposal_cycles_sensitivity"])
        proposal = source - saved
        sensitivity.append({
            "line": line,
            "m32_byte12_arithmetic_lower_bound_anchor": source,
            "phase_overlap_t10_cycles_saved_sensitivity": saved,
            "proposal_cycles_sensitivity": proposal,
            "compound_optimistic_speedup_vs_fixed_sensitivity": fixed / float(proposal),
            "crosses_3x_sensitivity": proposal * 3 < fixed,
            "cycles_margin_to_3x_sensitivity": fixed / 3.0 - proposal,
            "status": "UNIMPLEMENTED_PHASE_DECOUPLING_SENSITIVITY_NOT_SYSTEM_CYCLES",
        })

    return {
        "schema": "m37_phase_decoupled_csd_reconstruct_audit_v2",
        "status": "PASS_SIGNED_INT8_FULL_DOMAIN_FOUR_TERM_CSD_AND_PHASE_OVERLAP_SENSITIVITY_ONLY",
        "identity": {
            "contract": str(Path(contract_path).resolve()),
            "contract_sha256": sha256(contract_path),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "verified_input_sha256": hashes,
        },
        "signed_int8_coefficient_audit": {
            "domain": [-128, 127],
            "values": 256,
            "maximum_terms": max(int(key) for key, count in distribution.items() if count),
            "term_count_distribution": distribution,
            "all_constructive_identities_exact": True,
            "all_term_counts_minimum_over_distinct_shifts_0_to_7": True,
            "rows": coefficient_rows,
        },
        "phase_schedule_sensitivity": {
            "factor_tiles": tiles,
            "reduction_cycles_per_tile": reduction_cycles,
            "reconstruction_cycles_per_tile": reconstruction_cycles,
            "serialized_cycles_per_tile": serialized_per_tile,
            "overlapped_steady_state_tile_ii_target": overlapped_steady_ii,
            "serialized_t10_cycles": serialized_total,
            "overlapped_arithmetic_issue_cycles_with_ideal_phase_fill": overlapped_total,
            "cycles_saved_sensitivity": saved,
            "added_hardware_target": {
                "independent_csd_coefficient_ops_per_cycle": 96,
                "worst_case_signed_shift_add_terms_per_cycle": 384,
                "intermediate_ping_pong_storage_bits": 768,
                "average_result_beats_per_cycle_required_for_ii5": 1.0,
                "area_matched_baseline_required": "second_96_lane_signed_int8_multiplier_pool",
            },
            "rows": sensitivity,
            "unmodeled_nonzero_costs": [
                "programmable CSD reconstruction area and timing",
                "M31 bank handoff and simultaneous phase arbitration",
                "descriptor load bandwidth and CSD legality checking",
                "result FIFO contention, startup, tail, and backpressure",
                "48-bit result sink and 512-bit source bandwidth closure",
                "trained rank-3 INT8 accuracy and checkpoint descriptors",
                "SRAM/DRAM transactions, power, and physical routing",
            ],
        },
        "admission": {
            "signed_int8_full_domain_csd_math_admitted": True,
            "rtl_admitted": False,
            "integrated_phase_overlap_admitted": False,
            "trained_accuracy_admitted": False,
            "area_timing_power_admitted": False,
            "system_cycles_admitted": False,
            "speedup_admitted": False,
            "headline_admitted": False,
        },
        "claim_boundary": contract["claim_boundary"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.contract.resolve())
    write_output(args.output, result)
    print(args.output)


def write_output(output, result):
    output = Path(output)
    if output.exists():
        raise ValueError("refusing to overwrite M37 output")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
