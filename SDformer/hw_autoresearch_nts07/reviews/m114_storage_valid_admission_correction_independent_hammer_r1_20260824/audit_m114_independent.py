#!/usr/bin/env python3
"""Independent strict audit of M114 storage/admission correction."""

import argparse
import hashlib
import json
import re
from decimal import Decimal, getcontext
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
M109 = HW / (
    "results/m109_r2_window_storage_dual_timeline_frontier_r1_20260824/"
    "m109_r2_window_storage_dual_timeline_frontier.json")
M114_ANALYZER = HW / "system_simulator/scripts/analyze_m114_storage_valid_admission_correction.py"
M114_RESULT = HW / (
    "results/m114_storage_valid_admission_correction_r1_20260824/"
    "m114_storage_valid_admission_correction.json")
M114_CONTRACT = HW / "contracts/m114_storage_valid_admission_correction_contract_r1_20260824.json"
M114_MANIFEST = HW / "results/m114_storage_valid_admission_correction_r1_20260824/SHA256SUMS.txt"
M111_REVIEW = HW / (
    "reviews/m111_w384_signed24_accumulator_independent_hammer_r1_20260824/"
    "m111_w384_signed24_accumulator_independent_hammer_review.json")

RUN_DIRS = {
    "m110": HW / "dc_handoff/runs/m110_w384_full_capacity_vcs_r1_sealed_20260824",
    "m111": HW / "dc_handoff/runs/m111_w384_signed24_accumulator_vcs_r1_sealed_20260824",
    "m112": HW / "dc_handoff/runs/m112_w384_lane_sliced_accumulator_vcs_r1_sealed_20260824",
}

EXPECTED_SHA = {
    "m109": "ee61b90ee894c6e6c778b815a52f1d8b6edc9c877227bc4987e4b135aa16c321",
    "analyzer": "216be31a92ae22148462de21cdca61f58a2627414736e2f20ccd00501162555b",
    "result": "1559c65779fbc15026b3d744e3f1463bba8effd13c2efaa04e8562d4dbfb2226",
    "contract": "223af2980808282aa6fc0cd39d20a9bc6e7b2488e4807e9c8037e0c1e1f60c53",
    "m110_receipt": "2b73e6e29fcd176ab17d479fa33c0d0d785d3e2b90719ec7047b9513f5acfef7",
    "m111_receipt": "9a10f6e25b4451d17ce6849624bdf205d64548e7085986db74b4e75694088bcc",
    "m112_receipt": "458dc8af156165bf726d36a57813d2d476ec25dded82ffdee077c186f63bba26",
    "m111_review": "e4b5fbc45ccaf263b7b16393b5b54eb04c7fb9abab342e4eb50257025302204d",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

EXPECTED_WINDOWS = [43, 64, 96, 128, 192, 256, 294, 384, 512, 1024, 3000]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256(path):
    return sha256_bytes(Path(path).read_bytes())


def strict_loads(text):
    def reject(raw):
        raise ValueError("non-standard JSON constant: " + raw)

    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=pairs_hook, parse_constant=reject)


def strict_json(path):
    return strict_loads(Path(path).read_text(encoding="utf-8"))


def parse_manifest_text(text, base, allow_absolute):
    entries = []
    seen = set()
    for line_number, raw in enumerate(text.splitlines(), 1):
        require(raw != "", "blank manifest line {}".format(line_number))
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", raw)
        require(match is not None, "malformed manifest line {}".format(line_number))
        expected, raw_path = match.groups()
        path = Path(raw_path)
        require(allow_absolute or not path.is_absolute(), "absolute path forbidden")
        require(".." not in path.parts, "path traversal forbidden")
        key = str(path)
        require(key not in seen, "duplicate manifest path: " + key)
        seen.add(key)
        resolved = path if path.is_absolute() else Path(base) / path
        entries.append((expected, raw_path, resolved))
    require(entries, "empty manifest")
    return entries


def verify_manifest(path, base, allow_absolute=False):
    entries = parse_manifest_text(Path(path).read_text(encoding="utf-8"),
                                  base, allow_absolute)
    failed = []
    for expected, raw_path, resolved in entries:
        if not resolved.is_file() or sha256(resolved) != expected:
            failed.append(raw_path)
    return {
        "path": str(Path(path).relative_to(HW)),
        "sha256": sha256(path),
        "entries": len(entries),
        "failed": failed,
        "listed_paths": [raw_path for _, raw_path, _ in entries],
    }


def receipt_lines(path, required):
    lines = set(Path(path).read_text(encoding="utf-8").splitlines())
    missing = sorted(set(required) - lines)
    require(not missing, "receipt missing lines {}: {}".format(path, missing))
    return len(required)


def audit_upstream_run(label, required_lines):
    run = RUN_DIRS[label]
    receipt = run / "RUN_COMPLETE.txt"
    expected_receipt_sha = EXPECTED_SHA[label + "_receipt"]
    require(sha256(receipt) == expected_receipt_sha, label + " receipt SHA drift")
    required_count = receipt_lines(receipt, required_lines)
    input_manifest = verify_manifest(run / "input_sha256.txt", HW, allow_absolute=False)
    output_manifest = verify_manifest(run / "output_sha256.txt", run, allow_absolute=True)
    require(not input_manifest["failed"], label + " input manifest failure")
    require(not output_manifest["failed"], label + " output manifest failure")
    require((run / "compile.rc").read_text().strip() == "0", label + " compile RC")
    require((run / "sim.rc").read_text().strip() == "0", label + " sim RC")
    return {
        "receipt_sha256": expected_receipt_sha,
        "required_positive_and_boundary_lines_checked": required_count,
        "compile_return_code": 0,
        "simulation_return_code": 0,
        "input_manifest": input_manifest,
        "output_manifest": output_manifest,
    }


def expect_rejected(callable_obj, label):
    try:
        callable_obj()
    except (ValueError, json.JSONDecodeError):
        return True
    raise ValueError(label + " was not rejected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing independent audit overwrite")

    paths = {
        "m109": M109,
        "analyzer": M114_ANALYZER,
        "result": M114_RESULT,
        "contract": M114_CONTRACT,
        "m110_receipt": RUN_DIRS["m110"] / "RUN_COMPLETE.txt",
        "m111_receipt": RUN_DIRS["m111"] / "RUN_COMPLETE.txt",
        "m112_receipt": RUN_DIRS["m112"] / "RUN_COMPLETE.txt",
        "m111_review": M111_REVIEW,
        "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
    }
    observed_sha = {label: sha256(path) for label, path in paths.items()}
    require(observed_sha == EXPECTED_SHA, "exact SHA identity mismatch")

    frozen = strict_json(M109)
    result = strict_json(M114_RESULT)
    contract = strict_json(M114_CONTRACT)
    strict_json(M111_REVIEW)
    require(frozen["schema"] == "m109_r2_window_storage_dual_timeline_frontier_result_v1",
            "M109 schema drift")
    require(result["schema"] == "m114_storage_valid_admission_correction_result_v1",
            "M114 result schema drift")
    require(contract["schema"] == "m114_storage_valid_admission_correction_contract_v1",
            "M114 contract schema drift")

    strict_attacks = {
        "duplicate_json_key_rejected": expect_rejected(
            lambda: strict_loads('{"frontier":[],"frontier":[]}'), "duplicate JSON key"),
        "nan_rejected": expect_rejected(
            lambda: strict_loads('{"ratio":NaN}'), "NaN"),
        "positive_infinity_rejected": expect_rejected(
            lambda: strict_loads('{"ratio":Infinity}'), "Infinity"),
        "duplicate_manifest_path_rejected": expect_rejected(
            lambda: parse_manifest_text(
                "{}  a\n{}  a".format("0" * 64, "1" * 64), HW, False),
            "duplicate manifest path"),
        "malformed_manifest_hash_rejected": expect_rejected(
            lambda: parse_manifest_text("xyz  a", HW, False),
            "malformed manifest hash"),
        "manifest_path_traversal_rejected": expect_rejected(
            lambda: parse_manifest_text("{}  ../a".format("0" * 64), HW, False),
            "manifest path traversal"),
        "receipt_byte_mutation_changes_sha": (
            sha256_bytes(paths["m110_receipt"].read_bytes() + b"x")
            != EXPECTED_SHA["m110_receipt"]),
    }
    require(all(strict_attacks.values()), "strict parser/manifest attack failure")

    producer_manifest = verify_manifest(M114_MANIFEST, HW, allow_absolute=False)
    require(not producer_manifest["failed"], "M114 producer manifest verification failure")
    analyzer_inputs = {
        str(M109.relative_to(HW)),
        str((RUN_DIRS["m110"] / "RUN_COMPLETE.txt").relative_to(HW)),
        str((RUN_DIRS["m111"] / "RUN_COMPLETE.txt").relative_to(HW)),
        str((RUN_DIRS["m112"] / "RUN_COMPLETE.txt").relative_to(HW)),
        str(M111_REVIEW.relative_to(HW)),
    }
    listed = set(producer_manifest["listed_paths"])
    producer_manifest["analyzer_input_paths"] = sorted(analyzer_inputs)
    producer_manifest["missing_analyzer_input_paths"] = sorted(analyzer_inputs - listed)
    producer_manifest["covers_all_analyzer_inputs"] = not (analyzer_inputs - listed)

    upstream = {
        "m110": audit_upstream_run("m110", {
            "status=PASS_M110_W384_FULL_CAPACITY_DIRECTED_VCS_SVA",
            "exact_sha=true", "window_rows=384",
            "w384_controller_geometry_vcs=true",
            "accumulator_implemented=false",
            "actual_heldout_record_replay=false",
            "scheduled_cycle_ratio=false", "physical_speedup=false",
            "system_speedup=false", "headline=false",
        }),
        "m111": audit_upstream_run("m111", {
            "status=PASS_M111_W384_SIGNED24_ACCUMULATOR_DIRECTED_VCS_SVA",
            "exact_sha=true", "window_rows=384", "banks=8", "lanes=96",
            "logical_accumulator_bytes=884736", "lazy_valid_bits=3072",
            "behavioral_sync_1r1w_macro=true", "foundry_sram_macro=false",
            "full_lane_numeric_directed_miter=true",
            "exact_heldout_integrated_replay=false",
            "scheduled_cycle_ratio=false", "physical_speedup=false",
            "system_speedup=false", "headline=false",
        }),
        "m112": audit_upstream_run("m112", {
            "status=PASS_M112_W384_LANE_SLICED_ACCUMULATOR_DIRECTED_VCS_SVA",
            "exact_sha=true", "window_rows=384", "lane_macro_count=96",
            "lane_macro_depth=3072", "lane_macro_width_bits=24",
            "logical_accumulator_bytes=884736", "lazy_valid_bits=3072",
            "behavioral_sync_lane_sliced_1r1w_macro=true",
            "foundry_sram_macro=false", "full_lane_numeric_directed_miter=true",
            "exact_heldout_integrated_replay=false",
            "scheduled_cycle_ratio=false", "physical_speedup=false",
            "system_speedup=false", "headline=false",
        }),
    }

    old_rows = {int(row["window_rows"]): row for row in frozen["frontier"]}
    new_rows = {int(row["window_rows"]): row for row in result["frontier"]}
    require(sorted(old_rows) == EXPECTED_WINDOWS, "M109 window set drift")
    require(sorted(new_rows) == EXPECTED_WINDOWS, "M114 window set drift")

    getcontext().prec = 50
    storage_rows = []
    exact_work_fields_checked = 0
    recurrence_fields_checked = 0
    windows_per_phase_checked = 0
    max_ratio_recompute_error = Decimal(0)
    for window in EXPECTED_WINDOWS:
        old = old_rows[window]
        new = new_rows[window]
        require(new["windows_per_phase"] == old["windows_per_phase"],
                "windows_per_phase changed W{}".format(window))
        windows_per_phase_checked += 1
        require(new["exact_work"] == old["exact_work"],
                "exact work changed W{}".format(window))
        exact_work_fields_checked += len(old["exact_work"])
        require(new["dual_timeline_recurrence"] == old["dual_timeline_recurrence"],
                "dual timeline changed W{}".format(window))
        recurrence_fields_checked += len(old["dual_timeline_recurrence"])

        recurrence = old["dual_timeline_recurrence"]
        recomputed_ratio = (Decimal(recurrence["fair_fixed8_baseline_cycles"])
                            / Decimal(recurrence["candidate_cycles"]))
        serialized_ratio = Decimal(str(recurrence["same_clock_service_island_ratio"]))
        ratio_error = abs(recomputed_ratio - serialized_ratio)
        max_ratio_recompute_error = max(max_ratio_recompute_error, ratio_error)
        require(ratio_error < Decimal("5e-15"),
                "ratio division mismatch W{}".format(window))

        descriptor_bits = 2 * 128 * window * 2
        metadata_bits = 314
        accumulator_bits = window * 8 * 96 * 24
        old_bits = descriptor_bits + metadata_bits + accumulator_bits
        old_bytes = (old_bits + 7) // 8
        valid_bits = window * 8
        valid_bytes = (valid_bits + 7) // 8
        corrected_bits = old_bits + valid_bits
        corrected_bytes = (corrected_bits + 7) // 8
        old_storage = old["storage_lower_bound"]
        new_storage = new["storage_lower_bound_corrected"]
        expected_old = {
            "dual_bank_presence_plus_direction_bits": descriptor_bits,
            "descriptor_bank_metadata_bits_minimum": metadata_bits,
            "single_window_signed24_accumulator_bits": accumulator_bits,
            "combined_bits_before_control_ecc_macro_rounding": old_bits,
            "combined_bytes_ceiling_before_control_ecc_macro_rounding": old_bytes,
        }
        for key, expected in expected_old.items():
            require(old_storage[key] == expected,
                    "M109 storage arithmetic drift W{} {}".format(window, key))
        expected_new = dict(expected_old)
        expected_new.update({
            "single_window_accumulator_valid_bits": valid_bits,
            "single_window_accumulator_valid_bytes": valid_bytes,
            "combined_bits_before_control_ecc_macro_rounding": corrected_bits,
            "combined_bytes_ceiling_before_control_ecc_macro_rounding": corrected_bytes,
            "delta_bytes_vs_m109_r2": corrected_bytes - old_bytes,
        })
        for key, expected in expected_new.items():
            require(new_storage[key] == expected,
                    "M114 storage arithmetic drift W{} {}".format(window, key))
        require(valid_bytes == window, "valid bytes not W at W{}".format(window))
        require(corrected_bytes - old_bytes == window,
                "combined byte delta not W at W{}".format(window))

        old_geometry = bool(old["admission"]["controller_geometry_vcs"])
        admission = new["admission"]
        expected_admission = {
            "same_clock_dual_timeline_projection": True,
            "exact_heldout_work": True,
            "controller_geometry_vcs": old_geometry or window == 384,
            "full_lane_accumulator_vcs": window == 384,
            "lane_sliced_accumulator_adapter_vcs": window == 384,
            "integrated_controller_accumulator_vcs": False,
            "exact_heldout_integrated_replay": False,
            "foundry_sram_macro": False,
            "macro_inclusive_ppa": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        }
        require(admission == expected_admission,
                "admission mapping mismatch W{}".format(window))
        storage_rows.append({
            "window_rows": window,
            "valid_bits": valid_bits,
            "valid_bytes": valid_bytes,
            "m109_combined_bytes": old_bytes,
            "m114_corrected_combined_bytes": corrected_bytes,
            "delta_bytes": corrected_bytes - old_bytes,
            "candidate_cycles_unchanged": recurrence["candidate_cycles"],
            "baseline_cycles_unchanged": recurrence["fair_fixed8_baseline_cycles"],
            "ratio_unchanged": recurrence["same_clock_service_island_ratio"],
            "controller_geometry_vcs": admission["controller_geometry_vcs"],
            "full_lane_accumulator_vcs": admission["full_lane_accumulator_vcs"],
            "lane_sliced_adapter_vcs": admission["lane_sliced_accumulator_adapter_vcs"],
        })

    require(new_rows[64]["storage_lower_bound_corrected"]
            ["combined_bytes_ceiling_before_control_ecc_macro_rounding"] == 151656,
            "W64 headline correction mismatch")
    require(new_rows[384]["storage_lower_bound_corrected"]
            ["combined_bytes_ceiling_before_control_ecc_macro_rounding"] == 909736,
            "W384 headline correction mismatch")

    require(contract["frozen_identity"]["analyzer_sha256"] == EXPECTED_SHA["analyzer"],
            "contract analyzer pin mismatch")
    require(contract["frozen_identity"]["result_sha256"] == EXPECTED_SHA["result"],
            "contract result pin mismatch")
    require(contract["correction"]["w64_bytes_before"] == 151592
            and contract["correction"]["w64_bytes_corrected"] == 151656,
            "contract W64 correction mismatch")
    require(contract["correction"]["w384_bytes_before"] == 909352
            and contract["correction"]["w384_bytes_corrected"] == 909736,
            "contract W384 correction mismatch")
    w384_contract = contract["w384_frozen_observation"]
    require(w384_contract["candidate_cycles"] == 439708199
            and w384_contract["fair_fixed8_baseline_cycles"] == 1114863448
            and w384_contract["same_clock_precompacted_service_island_projection"]
                == 2.53546204172554,
            "contract W384 frozen schedule mismatch")
    require(w384_contract["standalone_controller_geometry_commercial_vcs"]
            and w384_contract["standalone_signed24_accumulator_commercial_vcs"]
            and w384_contract["standalone_lane_sliced_adapter_commercial_vcs"],
            "contract standalone admission missing")
    for key in ("integrated_controller_accumulator_vcs", "foundry_sram_macro",
                "macro_inclusive_ppa", "physical_speedup", "system_speedup", "headline"):
        require(w384_contract[key] is False, "contract over-admits " + key)

    prohibited = " ".join(contract["prohibited_claims"]).lower()
    for token in ("rtl measured", "physical", "equal-area", "full-network",
                  "system", "headline", "foundry", "integration"):
        require(token in prohibited, "missing prohibited-claim token: " + token)
    require("software projection" in contract["paper_safe_statement"].lower(),
            "paper-safe statement lost projection label")
    for key, value in result["model_boundary"].items():
        require(value is False, "result model boundary over-admits " + key)

    payload = {
        "schema": "m114_storage_valid_admission_correction_independent_audit_v1",
        "status": "PASS_STORAGE_AND_SCHEDULE_IDENTITY_ADMISSION_BOUNDED_MANIFEST_NOT_SELF_CONTAINED",
        "identity": observed_sha,
        "strict_attacks": strict_attacks,
        "producer_manifest": producer_manifest,
        "upstream_sealed_receipts": upstream,
        "field_identity": {
            "windows": len(EXPECTED_WINDOWS),
            "windows_per_phase_fields_checked": windows_per_phase_checked,
            "exact_work_fields_checked": exact_work_fields_checked,
            "dual_timeline_recurrence_fields_checked": recurrence_fields_checked,
            "candidate_fields_unchanged": len(EXPECTED_WINDOWS),
            "baseline_fields_unchanged": len(EXPECTED_WINDOWS),
            "serialized_ratio_fields_unchanged": len(EXPECTED_WINDOWS),
            "max_independent_ratio_division_error": str(max_ratio_recompute_error),
            "all_deep_equal": True,
        },
        "storage_recomputation": {
            "formula_bits": "2*128*W*2 + 314 + W*8*96*24 + W*8",
            "byte_rule": "ceil(total_bits/8)",
            "rows": storage_rows,
            "w64_before": 151592,
            "w64_after": 151656,
            "w384_before": 909352,
            "w384_after": 909736,
            "all_valid_bit_byte_and_combined_ceil_checks_pass": True,
        },
        "admission_audit": {
            "w64_controller_geometry_from_m109_m106_chain_only": True,
            "w384_controller_geometry_from_m110_receipt": True,
            "w384_full_lane_accumulator_from_m111_receipt": True,
            "w384_lane_sliced_adapter_from_m112_receipt": True,
            "non_w384_new_accumulator_or_adapter_admission": False,
            "integrated_controller_accumulator_vcs": False,
            "exact_heldout_integrated_replay": False,
            "foundry_sram_macro": False,
            "macro_inclusive_ppa": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
            "current_mapping_overreach_found": False,
            "producer_positive_line_checks_complete": False,
        },
        "claim_boundary": {
            "ratio": 2.53546204172554,
            "same_clock_precompacted_software_projection": True,
            "rtl_measured": False,
            "physical": False,
            "equal_area": False,
            "full_network": False,
            "system": False,
            "headline": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS M114 independent windows={} recurrence_fields={} W64={}B W384={}B manifest_missing={}".format(
        len(EXPECTED_WINDOWS), recurrence_fields_checked,
        payload["storage_recomputation"]["w64_after"],
        payload["storage_recomputation"]["w384_after"],
        len(producer_manifest["missing_analyzer_input_paths"])), flush=True)


if __name__ == "__main__":
    main()
