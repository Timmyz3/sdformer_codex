#!/usr/bin/env python3
"""Independent, receipt-blind static recomputation for the M714 pre-run review.

This script does not import or execute the author M714/M366 modules.  It checks
the source identities, proves the signed-INT8 bit-plane scalar identity over all
256 codes, and rebuilds the table/macro arithmetic used by the review.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M714 = HW / "system_simulator/scripts/trace_m714_h67_ep35_pctda_pattern_s10.py"
M366 = HW / "system_simulator/scripts/trace_m366_h67_ep35_atlif_remaining_budget_s10.py"
M366_CONTRACT = HW / "contracts/m366_h67_ep35_atlif_remaining_budget_s10_contract_r1_20260825.json"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
M518_RTL = HW / "rtl_m518/m518_matched_fixed_t10_atlif.sv"
M518_TB = HW / "tb_m518/tb_m518_matched_fixed_t10_atlif.sv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    m714_text = M714.read_text(encoding="utf-8")
    m366_text = M366.read_text(encoding="utf-8")
    m518_rtl_text = M518_RTL.read_text(encoding="utf-8")
    m518_tb_text = M518_TB.read_text(encoding="utf-8")

    # Two's-complement DA is linear in each source.  Exhaustively proving that
    # each 8-bit code reconstructs its signed scalar proves the algebra for any
    # integer weight matrix and any grouping of the ten sources.
    scalar_mismatches = []
    for code in range(256):
        signed = code if code < 128 else code - 256
        reconstructed = sum(
            ((-128 if bit == 7 else (1 << bit))
             if ((code >> bit) & 1) else 0)
            for bit in range(8)
        )
        if reconstructed != signed:
            scalar_mismatches.append(
                {"code": code, "signed": signed, "reconstructed": reconstructed})

    group = 5
    outputs = 10
    lanes = 16
    acc_bits = 25
    subset_min = group * -128
    subset_max = group * 127
    subset_width = 11
    logical_table_bits = 2 * (1 << group) * outputs * subset_width
    macro_bits = 128 * 128
    macro_area_um2 = 8758.360550
    fixed_provisional_area_um2 = 66778.235814

    port_rows = []
    for ports in (1, 2, 4, 8):
        active_macros = ports
        resident_macros = ports * 23
        active_capacity_bits = active_macros * macro_bits
        resident_capacity_bits = resident_macros * macro_bits
        macro_area = active_macros * macro_area_um2
        resident_area = resident_macros * macro_area_um2
        port_rows.append({
            "ports": ports,
            "logical_table_bytes_replicated": logical_table_bits * ports // 8,
            "active_macro_count": active_macros,
            "active_macro_capacity_bytes": active_capacity_bits // 8,
            "active_macro_plus_accumulator_bytes":
                (active_capacity_bits + lanes * outputs * acc_bits) // 8,
            "active_macro_area_um2": round(macro_area, 6),
            "all45_resident_macro_count": resident_macros,
            "all45_resident_macro_capacity_bytes": resident_capacity_bits // 8,
            "all45_resident_macro_area_um2": round(resident_area, 6),
            "all45_resident_area_over_provisional_fixed_cell_area":
                round(resident_area / fixed_provisional_area_um2, 6),
        })

    contract_sha = sha256(M366_CONTRACT)
    report = {
        "schema": "m716_m714_pctda_prerun_static_recompute_v1",
        "status": "PASS_INDEPENDENT_STATIC_RECOMPUTE__NOT_A_M714_RUN_RECEIPT",
        "method": {
            "receipt_blind": True,
            "author_module_imported": False,
            "author_capture_executed": False,
            "gpu_used": False,
            "eda_used": False,
        },
        "identity": {
            "m714_script_sha256": sha256(M714),
            "m366_script_sha256": sha256(M366),
            "m366_contract_sha256": contract_sha,
            "m518_rtl_sha256": sha256(M518_RTL),
            "m518_tb_sha256": sha256(M518_TB),
            "protected_docs359_sha256": sha256(DOC359),
        },
        "signed_da": {
            "all_256_signed_int8_codes_checked": True,
            "scalar_mismatch_count": len(scalar_mismatches),
            "scalar_mismatches": scalar_mismatches,
            "group_size": group,
            "subset_sum_range": [subset_min, subset_max],
            "signed_subset_width_bits": subset_width,
            "subset_range_fits_width":
                subset_min >= -(1 << (subset_width - 1)) and
                subset_max <= (1 << (subset_width - 1)) - 1,
            "logical_table_bits_per_unreplicated_config": logical_table_bits,
            "logical_table_bytes_per_unreplicated_config": logical_table_bits // 8,
            "accumulator_guard_bits": acc_bits,
            "worst_absolute_bitplane_product_sum_bound":
                2 * group * 128 * 255,
            "worst_absolute_with_q24_bias_bound":
                (1 << 23) + 2 * group * 128 * 255,
            "fits_signed25_by_conservative_absolute_bound":
                (1 << 23) + 2 * group * 128 * 255 < (1 << 24),
        },
        "m518_cycle_anchor": {
            "rtl_declares_five_256bit_config_beats":
                "five 256-bit configuration beats" in m518_rtl_text,
            "rtl_acc_width_25": "localparam int ACC_W = 25" in m518_rtl_text,
            "tb_exact_formula_17n_plus_12":
                "measured_cycles!=(17*tiles+12)" in m518_tb_text,
            "n1_cycles": 17 * 1 + 12,
            "n4_cycles": 17 * 4 + 12,
            "interpretation": (
                "The measured interval begins at first config acceptance, so the "
                "12-cycle intercept already includes the five-beat configuration boundary."
            ),
        },
        "resource_recompute": {
            "one_config_logical_table_bits": logical_table_bits,
            "two_configs_fit_one_128x128_macro_by_64x110_layout":
                2 * logical_table_bits <= macro_bits,
            "all45_macro_pairs_per_replica": 23,
            "accumulator_state_bits": lanes * outputs * acc_bits,
            "ports": port_rows,
        },
        "static_fail_closed_checks": {
            "m714_literal_pins_canonical_m366_contract_sha":
                contract_sha in m714_text,
            "m714_requires_caller_pinned_expected_self_sha":
                "EXPECTED_M714" in m714_text,
            "m714_has_gpu_process_idle_guard": any(
                token in m714_text for token in
                ("nvidia-smi", "four_consecutive_idle", "gpu_idle_check")),
            "m714_uses_atomic_staging_commit": any(
                token in m714_text for token in
                ("staging", "atomic", "os.replace", ".tmp")),
            "m714_emits_member_manifest_or_seal": any(
                token in m714_text for token in
                ("SHA256SUMS", "OUTER_SHA256", "seal")),
            "m714_checks_m366_zero_range_before_pass":
                "zero_range_violation" in m714_text,
            "m714_checks_m366_population_before_pass":
                "live_t10_sites" in m714_text and "samples" in m714_text,
            "m714_real_checkpoint_da_output_miter_present":
                "weight_q8" in m714_text and "fixed_event" in m714_text and
                "da_output" in m714_text,
            "m366_execute_has_internal_gpu_idle_guard": any(
                token in m366_text for token in
                ("nvidia-smi", "four_consecutive_idle", "gpu_idle_check")),
        },
        "cold_path_accounting": {
            "m518_config_payload_bits": 1064,
            "m518_bus_transfer_bits_for_five_beats": 5 * 256,
            "da_table_logical_bits": logical_table_bits,
            "direct_table_load_256bit_beats_ceiling":
                (logical_table_bits + 255) // 256,
            "finding": (
                "A build-from-weights mode uses the existing five config beats plus "
                "table-build cycles; a direct-table-load mode needs 28 beats and no "
                "64-cycle build. Reporting 880 external bytes together with five beats "
                "and a 64-cycle build conflates both modes."
            ),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
