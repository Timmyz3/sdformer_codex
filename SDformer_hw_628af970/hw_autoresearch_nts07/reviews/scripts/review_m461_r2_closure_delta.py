#!/usr/bin/env python3
"""Independent delta hammer for the sealed M461 prereview R2 closure.

Only the R2 closure and the prior independent R1 review are accepted.  There
is deliberately no argument for M40, M453b, docs/359, an RTL tree or a catalog.
The script writes nothing and emits its audit as JSON on stdout.
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_double_seal(directory, expected_manifest_sha, expected_outer_sha):
    manifest = directory / "SHA256SUMS"
    seal = directory / "SHA256SUMS.seal.sha256"
    require(sha256(manifest) == expected_manifest_sha, "inner manifest SHA mismatch")
    require(sha256(seal) == expected_outer_sha, "outer seal-file SHA mismatch")
    sealed_manifest_sha, sealed_name = seal.read_text(encoding="utf-8").split()
    require(sealed_name == "SHA256SUMS", "outer seal target mismatch")
    require(sealed_manifest_sha == expected_manifest_sha,
            "outer seal does not bind expected manifest")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected_sha, name = line.split(None, 1)
        require(sha256(directory / name.strip()) == expected_sha,
                "manifest payload mismatch: " + name.strip())


def descriptor_legal(destination, original, center, stored_distance,
                     stored_use_pwp, descriptor_valid, reserved):
    distance = bin(original ^ center).count("1")
    use_pwp = int(1 + distance < bin(original).count("1"))
    return (descriptor_valid == 1 and reserved == 0 and
            0 <= destination <= 2999 and 0 < original < (1 << 16) and
            0 <= center < (1 << 16) and stored_distance == distance and
            stored_use_pwp == use_pwp)


def compact_mapping(bitmap):
    centers = [center for center in range(128) if (bitmap >> center) & 1]
    center_to_slot = dict((center, slot) for slot, center in enumerate(centers))
    slot_to_center = dict((slot, center) for slot, center in enumerate(centers))
    return centers, center_to_slot, slot_to_center


def mapping_legal(bitmap, center_to_slot, slot_to_center, pwp_valid):
    centers = [center for center in range(128) if (bitmap >> center) & 1]
    nused = len(centers)
    if set(center_to_slot) != set(centers):
        return False
    if set(center_to_slot.values()) != set(range(nused)):
        return False
    if set(slot_to_center) != set(range(nused)):
        return False
    for center, slot in center_to_slot.items():
        if slot_to_center.get(slot) != center:
            return False
        if pwp_valid.get(slot) != 0xff:
            return False
    return True


def role_switch_legal(assignment_sealed, remap_ready, pwp_ready,
                      weight_valid, config_valid, generator_idle,
                      no_next_write, old_replay_drained,
                      downstream_drained):
    return all((assignment_sealed, remap_ready, pwp_ready, weight_valid,
                config_valid, generator_idle, no_next_write,
                old_replay_drained, downstream_drained))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-dir", required=True)
    parser.add_argument("--prior-review-dir", required=True)
    parser.add_argument("--expected-subject-json-sha", required=True)
    parser.add_argument("--expected-subject-manifest-sha", required=True)
    parser.add_argument("--expected-subject-outer-seal-sha", required=True)
    parser.add_argument("--expected-prior-review-sha", required=True)
    parser.add_argument("--expected-prior-manifest-sha", required=True)
    parser.add_argument("--expected-prior-outer-seal-sha", required=True)
    args = parser.parse_args()

    subject_dir = Path(args.subject_dir)
    prior_dir = Path(args.prior_review_dir)
    subject_path = subject_dir / "m461_exact_derived_pwp_prereview_r2_closure.json"
    prior_path = prior_dir / "m461r2_m461_prereview_independent_hammer_review_r1.json"

    verify_double_seal(subject_dir, args.expected_subject_manifest_sha,
                       args.expected_subject_outer_seal_sha)
    verify_double_seal(prior_dir, args.expected_prior_manifest_sha,
                       args.expected_prior_outer_seal_sha)
    require(sha256(subject_path) == args.expected_subject_json_sha,
            "subject JSON SHA mismatch")
    require(sha256(prior_path) == args.expected_prior_review_sha,
            "prior review SHA mismatch")
    subject = json.loads(subject_path.read_text(encoding="utf-8"))
    prior = json.loads(prior_path.read_text(encoding="utf-8"))

    descriptor = subject["primary_48bit_descriptor_contract"]
    layout = descriptor["bit_layout_lsb0"]
    field_widths = {
        "destination_row_id": 12,
        "original16": 16,
        "global_center_id": 7,
        "hamming_distance": 5,
        "use_pwp": 1,
        "descriptor_valid": 1,
        "reserved": 6,
    }
    require(sum(field_widths.values()) == descriptor["word_bits"] == 48,
            "48-bit field sum mismatch")
    require(set(layout) == set(("bits_11_0", "bits_27_12", "bits_34_28",
                               "bits_39_35", "bit_40", "bit_41",
                               "bits_47_42")), "layout range gap/overlap")

    storage = subject["descriptor_storage_sensitivity"]
    independent_48 = 2 * 3000 * 48 // 8
    independent_64 = 2 * 3000 * 64 // 8
    require(storage["logical_two_banks_bytes"] == independent_48 == 36000,
            "48-bit bank bytes mismatch")
    require(storage["physical_64bit_macro_two_banks_bytes"] == independent_64 == 48000,
            "64-bit sensitivity bytes mismatch")
    require(storage["padding_delta_bytes"] == independent_64 - independent_48 == 12000,
            "padding delta mismatch")
    require(abs(storage["padding_delta_fraction_vs_48bit_logical"] - 1.0 / 3.0) < 1e-15,
            "padding fraction mismatch")

    descriptor_attacks = {}
    base = dict(destination=2999, original=0x00f3, center=0x00f1,
                stored_distance=1, stored_use_pwp=1,
                descriptor_valid=1, reserved=0)
    require(descriptor_legal(**base), "legal boundary descriptor rejected")
    for name, mutation in (
            ("valid_zero", {"descriptor_valid": 0}),
            ("reserved_nonzero", {"reserved": 1}),
            ("destination_3000", {"destination": 3000}),
            ("original_zero", {"original": 0}),
            ("distance_mismatch", {"stored_distance": 2}),
            ("use_pwp_mismatch", {"stored_use_pwp": 0})):
        attacked = dict(base)
        attacked.update(mutation)
        descriptor_attacks[name] = not descriptor_legal(**attacked)
    require(all(descriptor_attacks.values()), "descriptor attack survived")
    require(descriptor["fallback_semantics"].startswith("use_pwp=0 remains exact bit-sparse"),
            "fallback semantics drift")
    require("must not read remap or PWP storage" in descriptor["fallback_semantics"],
            "fallback can access PWP")

    sentinel = descriptor["valid_and_sentinel"]
    require(sentinel["no_extra_word_required"] and
            "addresses 0..2999" in sentinel["full_3000_active_rows"] and
            "pointer value 3000" in sentinel["full_3000_active_rows"],
            "full-3000 sentinel closure failed")
    require("sealed_active_count=0" in sentinel["empty_phase"] and
            "no descriptor/PWP SRAM request" in sentinel["empty_phase"],
            "empty sentinel closure failed")
    require("not fetched from descriptor SRAM" in sentinel["synthetic_end_sentinel"] and
            "never sent to the arithmetic backend" in sentinel["synthetic_end_sentinel"],
            "sentinel leak remains possible")

    mapping_vectors = [0, 1, 1 << 127, (1 << 128) - 1,
                       int("aa" * 16, 16), int("55" * 16, 16)]
    rng = random.Random(461)
    mapping_vectors.extend(rng.getrandbits(128) for _ in range(1024))
    mapping_checks = 0
    for bitmap in mapping_vectors:
        centers, forward, inverse = compact_mapping(bitmap)
        valid = dict((slot, 0xff) for slot in range(len(centers)))
        require(mapping_legal(bitmap, forward, inverse, valid),
                "legal compact mapping rejected")
        mapping_checks += 1
    bitmap = (1 << 2) | (1 << 9)
    centers, forward, inverse = compact_mapping(bitmap)
    valid = {0: 0xff, 1: 0xff}
    duplicate_forward = dict(forward)
    duplicate_forward[9] = 0
    bad_inverse = dict(inverse)
    bad_inverse[1] = 2
    missing_block = dict(valid)
    missing_block[1] = 0x7f
    mapping_attacks = {
        "duplicate_slot": not mapping_legal(bitmap, duplicate_forward, inverse, valid),
        "inverse_mismatch": not mapping_legal(bitmap, forward, bad_inverse, valid),
        "missing_pwp_block": not mapping_legal(bitmap, forward, inverse, missing_block),
        "unknown_fail_closed_stated": "any X/Z" in
            subject["center_to_compact_slot_contract"]["unknown_behavior"] and
            "fail-closed" in subject["center_to_compact_slot_contract"]["unknown_behavior"],
        "invalid_lookup_sticky_stated": "sticky protocol_error" in
            subject["center_to_compact_slot_contract"]["invalid_lookup_behavior"],
    }
    require(all(mapping_attacks.values()), "remap attack survived")

    role_attack_count = 0
    all_ready = [True] * 9
    require(role_switch_legal(*all_ready), "fully ready role switch rejected")
    for index in range(len(all_ready)):
        attacked = list(all_ready)
        attacked[index] = False
        require(not role_switch_legal(*attacked), "early role switch survived")
        role_attack_count += 1
    lifecycle = subject["assignment_bank_lifecycle"]
    require("same-bank read/write is forbidden" in lifecycle["roles"],
            "same-bank role attack not closed")
    require(any("bank_epoch_valid=0" in item for item in lifecycle["invalidate_next"]),
            "epoch invalidation not closed")
    require(any("matcher child-stage drain" in item for item in lifecycle["seal"]),
            "matcher drain before seal missing")
    require(any("old current replay plus downstream update_delta drain is complete" in item
                for item in lifecycle["role_switch"]),
            "downstream drain before switch missing")
    require("protocol_error is sticky and reset-only" in lifecycle["reload_and_recovery"],
            "sticky recovery closure missing")
    require("suppresses descriptor/remap/PWP/weight request accept" in
            subject["atomic_fail_closed_effect"] and
            "holds all externally visible outputs quiescent until reset" in
            subject["atomic_fail_closed_effect"], "atomic fail-closed missing")

    percentile = subject["percentile_closure"]
    nvalue = percentile["population_n"]
    nearest_index = int(math.ceil(0.95 * nvalue)) - 1
    legacy_index = int(math.floor(0.95 * (nvalue - 1)))
    require(nearest_index == 1641 and legacy_index == 1640,
            "percentile index arithmetic mismatch")
    require(percentile["standard_nearest_rank"]["zero_based_index"] == nearest_index and
            percentile["standard_nearest_rank"]["value"] == 1584 and
            percentile["standard_nearest_rank"]["canonical_p95_for_future_tables"],
            "canonical nearest-rank closure failed")
    require(percentile["legacy_floor_n_minus_1"]["zero_based_index"] == legacy_index and
            percentile["legacy_floor_n_minus_1"]["value"] == 1576 and
            not percentile["legacy_floor_n_minus_1"]["canonical_p95"],
            "legacy percentile closure failed")
    prior_structural = prior["independent_recompute"][
        "all_128_centers_times_8_weight_update_cycles_only"]
    require(prior_structural["p95_standard_nearest_rank"] == 1584 and
            prior_structural["p95_subject_floor_0p95_times_n_minus_1"] == 1576,
            "R2 percentile values drift from sealed prior review")

    config = subject["B_config_footprint_closure"]
    per_tile = 288 + 6144 + 20480 + 640
    require(per_tile == 27552 and
            config["per_tile_compatibility_address_footprint"]["bytes_per_tile"] == per_tile and
            config["per_tile_compatibility_address_footprint"]["two_tile_address_footprint_bytes"] == 2 * per_tile,
            "B address footprint mismatch")
    base_without_config = prior["independent_storage_arithmetic"][
        "B_known_lower_bound_bytes"] - 2 * 288
    shared_config = 2 * 288
    replicated_config = 2 * 2 * 288
    shared_total = base_without_config + shared_config
    replicated_total = base_without_config + replicated_config
    require(shared_total == 156896 and replicated_total == 157472,
            "independent B config totals mismatch")
    require(config["physical_options"]["shared_one_copy_per_phase_role"]
            ["B_known_lower_bound_bytes"] == shared_total and
            config["physical_options"]["physically_replicated_one_copy_per_tile_per_phase_role"]
            ["B_known_lower_bound_bytes"] == replicated_total,
            "subject B config totals mismatch")

    prior_findings = set()
    for severity in ("P1", "P2"):
        prior_findings.update(item["id"] for item in prior["findings"][severity])
    closures = subject["closure_of_independent_findings"]
    require(set(closures) == prior_findings,
            "R2 closure keys do not exactly cover prior P1/P2")
    require(all(value.startswith("CLOSED") for value in closures.values()),
            "one prior finding is not closed")

    priority = subject["priority_and_decision"]
    prior_decision = prior["decision"]
    require(priority["compact_used_center_original_order"] ==
            prior_decision["compact_used_center_original_order"] ==
            "UNIQUE_FIRST_FUTURE_MODEL_WAIT_FOR_SEALED_NMAX",
            "unique-first priority drift")
    require(priority["A_full_q128_cache"] == prior_decision["A_full_q128_cache"],
            "A priority drift")
    require(priority["C_small_original_order_child_cache"] ==
            prior_decision["C_small_original_order_child_cache"],
            "C priority drift")
    require(priority["true_group_replay"] == prior_decision["true_group_replay"],
            "group-replay priority drift")
    require(priority["integrated_fold"] == prior_decision["integrated_fold"],
            "fold priority drift")
    require("BACKUP" in priority["B_q32_parent_plus_child_scratch"] and
            "NO_RTL_NO_PERFORMANCE" in priority["B_q32_parent_plus_child_scratch"],
            "B priority drift")
    false_claim_fields = ("M40_read", "M453b_read_or_execution", "rtl_authorized",
                          "cycle_speedup", "system_speedup",
                          "resource_normalized_speedup", "power", "energy",
                          "paper_ppa_ready", "date_headline")
    require(all(priority[field] is False for field in false_claim_fields),
            "claim boundary drift")

    output = {
        "status": "PASS_M461_R2_DELTA_CLOSURE_INDEPENDENT_HAMMER",
        "identity": {
            "subject_json_sha256": sha256(subject_path),
            "subject_manifest_sha256": sha256(subject_dir / "SHA256SUMS"),
            "subject_outer_seal_sha256": sha256(subject_dir / "SHA256SUMS.seal.sha256"),
            "prior_review_sha256": sha256(prior_path),
            "prior_manifest_sha256": sha256(prior_dir / "SHA256SUMS"),
            "prior_outer_seal_sha256": sha256(prior_dir / "SHA256SUMS.seal.sha256"),
        },
        "descriptor": {
            "field_widths": field_widths,
            "field_sum": sum(field_widths.values()),
            "logical_two_bank_bytes": independent_48,
            "physical_64bit_sensitivity_bytes": independent_64,
            "padding_delta_bytes": independent_64 - independent_48,
            "descriptor_attacks": descriptor_attacks,
            "full_3000_sentinel": True,
            "empty_phase_sentinel": True,
        },
        "remap": {
            "legal_bitmap_vectors_checked": mapping_checks,
            "attacks": mapping_attacks,
        },
        "lifecycle": {
            "early_role_switch_attacks": role_attack_count,
            "assignment_seal_epoch_role_switch_and_drain": True,
            "atomic_fail_closed": True,
        },
        "percentile": {
            "n": nvalue,
            "nearest_rank_index": nearest_index,
            "nearest_rank_value": 1584,
            "legacy_floor_index": legacy_index,
            "legacy_floor_value": 1576,
        },
        "B_config": {
            "per_tile_address_footprint_bytes": per_tile,
            "two_tile_address_footprint_bytes": 2 * per_tile,
            "base_without_config_bytes": base_without_config,
            "shared_config_bytes": shared_config,
            "shared_lower_bound_bytes": shared_total,
            "replicated_config_bytes": replicated_config,
            "replicated_lower_bound_bytes": replicated_total,
        },
        "closure": {
            "prior_P1_P2_ids": sorted(prior_findings),
            "all_exactly_closed": True,
            "unique_first_and_claim_boundary_unchanged": True,
        },
        "scope": {
            "m40_argument_or_read": False,
            "m453b_argument_or_read_or_execution": False,
            "docs359_argument_or_read_or_write": False,
            "rtl_argument_or_read_or_write": False,
            "future_contract_event_model_only": True,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
