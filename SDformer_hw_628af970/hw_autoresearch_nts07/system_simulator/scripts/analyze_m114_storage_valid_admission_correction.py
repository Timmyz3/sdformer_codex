#!/usr/bin/env python3
"""Supersede M109 storage/admission metadata without changing its schedule.

M109-r2 intentionally excluded accumulator valid/epoch state.  M111/M112
implement one valid bit per (output block, window row), so the executable
lower bound must add W*8 bits.  This audit freezes every M109 work and cycle
field, adds only that state, and records the standalone W384 commercial-VCS
admissions.  It does not create an integrated or physical speedup claim.
"""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
INPUTS = {
    "m109_r2_result": HW / (
        "results/m109_r2_window_storage_dual_timeline_frontier_r1_20260824/"
        "m109_r2_window_storage_dual_timeline_frontier.json"),
    "m110_w384_controller_vcs": HW / (
        "dc_handoff/runs/m110_w384_full_capacity_vcs_r1_sealed_20260824/"
        "RUN_COMPLETE.txt"),
    "m111_w384_accumulator_vcs": HW / (
        "dc_handoff/runs/m111_w384_signed24_accumulator_vcs_r1_sealed_20260824/"
        "RUN_COMPLETE.txt"),
    "m112_w384_lane_adapter_vcs": HW / (
        "dc_handoff/runs/m112_w384_lane_sliced_accumulator_vcs_r1_sealed_20260824/"
        "RUN_COMPLETE.txt"),
    "m111_independent_review": HW / (
        "reviews/m111_w384_signed24_accumulator_independent_hammer_r1_20260824/"
        "m111_w384_signed24_accumulator_independent_hammer_review.json"),
}
EXPECTED_SHA256 = {
    "m109_r2_result":
        "ee61b90ee894c6e6c778b815a52f1d8b6edc9c877227bc4987e4b135aa16c321",
    "m110_w384_controller_vcs":
        "2b73e6e29fcd176ab17d479fa33c0d0d785d3e2b90719ec7047b9513f5acfef7",
    "m111_w384_accumulator_vcs":
        "9a10f6e25b4451d17ce6849624bdf205d64548e7085986db74b4e75694088bcc",
    "m112_w384_lane_adapter_vcs":
        "458dc8af156165bf726d36a57813d2d476ec25dded82ffdee077c186f63bba26",
    "m111_independent_review":
        "e4b5fbc45ccaf263b7b16393b5b54eb04c7fb9abab342e4eb50257025302204d",
}
OUTPUT_BLOCKS = 8


def require(condition, message):
    if not condition:
        raise ValueError(message)


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
        output = {}
        for key, value in pairs:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def require_receipt(label, required_lines):
    lines = set(INPUTS[label].read_text(encoding="utf-8").splitlines())
    for line in required_lines:
        require(line in lines, label + " missing receipt line: " + line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M114 output overwrite")
    script_start_sha = sha256(Path(__file__).resolve())

    for label, path in INPUTS.items():
        require(sha256(path) == EXPECTED_SHA256[label],
                "frozen input identity drift: " + label)
    require_receipt("m110_w384_controller_vcs", (
        "status=PASS_M110_W384_FULL_CAPACITY_DIRECTED_VCS_SVA",
        "scheduled_cycle_ratio=false",
        "physical_speedup=false",
        "system_speedup=false",
        "headline=false",
    ))
    require_receipt("m111_w384_accumulator_vcs", (
        "status=PASS_M111_W384_SIGNED24_ACCUMULATOR_DIRECTED_VCS_SVA",
        "exact_heldout_integrated_replay=false",
        "physical_speedup=false",
        "system_speedup=false",
        "headline=false",
    ))
    require_receipt("m112_w384_lane_adapter_vcs", (
        "status=PASS_M112_W384_LANE_SLICED_ACCUMULATOR_DIRECTED_VCS_SVA",
        "foundry_sram_macro=false",
        "exact_heldout_integrated_replay=false",
        "physical_speedup=false",
        "system_speedup=false",
        "headline=false",
    ))

    frozen = strict_json(INPUTS["m109_r2_result"])
    require(frozen["schema"] ==
            "m109_r2_window_storage_dual_timeline_frontier_result_v1",
            "unexpected M109 schema")
    corrected = []
    for old in frozen["frontier"]:
        window = int(old["window_rows"])
        old_storage = old["storage_lower_bound"]
        valid_bits = window * OUTPUT_BLOCKS
        old_bits = int(old_storage[
            "combined_bits_before_control_ecc_macro_rounding"])
        corrected_bits = old_bits + valid_bits
        corrected_bytes = (corrected_bits + 7) // 8
        require(corrected_bytes == int(old_storage[
            "combined_bytes_ceiling_before_control_ecc_macro_rounding"]) + window,
                "valid-byte correction invariant failed W{}".format(window))

        corrected.append({
            "window_rows": window,
            "windows_per_phase": old["windows_per_phase"],
            "exact_work": old["exact_work"],
            "dual_timeline_recurrence": old["dual_timeline_recurrence"],
            "storage_lower_bound_corrected": {
                "dual_bank_presence_plus_direction_bits": old_storage[
                    "dual_bank_presence_plus_direction_bits"],
                "descriptor_bank_metadata_bits_minimum": old_storage[
                    "descriptor_bank_metadata_bits_minimum"],
                "single_window_signed24_accumulator_bits": old_storage[
                    "single_window_signed24_accumulator_bits"],
                "single_window_accumulator_valid_bits": valid_bits,
                "single_window_accumulator_valid_bytes": window,
                "combined_bits_before_control_ecc_macro_rounding": corrected_bits,
                "combined_bytes_ceiling_before_control_ecc_macro_rounding":
                    corrected_bytes,
                "delta_bytes_vs_m109_r2": window,
            },
            "admission": {
                "same_clock_dual_timeline_projection": True,
                "exact_heldout_work": True,
                "controller_geometry_vcs":
                    bool(old["admission"]["controller_geometry_vcs"])
                    or window == 384,
                "full_lane_accumulator_vcs": window == 384,
                "lane_sliced_accumulator_adapter_vcs": window == 384,
                "integrated_controller_accumulator_vcs": False,
                "exact_heldout_integrated_replay": False,
                "foundry_sram_macro": False,
                "macro_inclusive_ppa": False,
                "physical_speedup": False,
                "system_speedup": False,
                "headline": False,
            },
        })

    old_by_w = {int(row["window_rows"]): row for row in frozen["frontier"]}
    new_by_w = {int(row["window_rows"]): row for row in corrected}
    require(new_by_w[384]["storage_lower_bound_corrected"][
        "combined_bytes_ceiling_before_control_ecc_macro_rounding"] == 909736,
            "W384 corrected storage drift")
    require(new_by_w[384]["dual_timeline_recurrence"] ==
            old_by_w[384]["dual_timeline_recurrence"],
            "M114 must not change W384 schedule")

    require(sha256(Path(__file__).resolve()) == script_start_sha,
            "M114 analyzer changed during execution")
    payload = {
        "schema": "m114_storage_valid_admission_correction_result_v1",
        "status": "PASS_M109_STORAGE_VALID_BITS_CORRECTED_SCHEDULE_FROZEN",
        "identity": {
            "analyzer_start_end_sha256": script_start_sha,
            "frozen_inputs_sha256": EXPECTED_SHA256,
        },
        "supersession": {
            "supersedes": "M109-r2 storage lower-bound and admission metadata only",
            "does_not_supersede": "M109-r2 exact work or dual-timeline cycles",
            "correction": "add one valid bit per output-block/window-row",
        },
        "frontier": corrected,
        "model_boundary": {
            "precompaction_schedule": False,
            "shared_weight_sram_arbitration": False,
            "integrated_controller_accumulator_vcs": False,
            "exact_heldout_integrated_replay": False,
            "foundry_sram_macro": False,
            "macro_inclusive_ppa": False,
            "equal_area": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / "m114_storage_valid_admission_correction.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS M114 " + " ".join(
        "W{}={}B/{:.9f}x".format(
            row["window_rows"],
            row["storage_lower_bound_corrected"][
                "combined_bytes_ceiling_before_control_ecc_macro_rounding"],
            row["dual_timeline_recurrence"][
                "same_clock_service_island_ratio"])
        for row in corrected), flush=True)


if __name__ == "__main__":
    main()
