#!/usr/bin/env python3
from __future__ import print_function

import argparse
from decimal import Decimal, getcontext
import hashlib
import json
import os
import sys


getcontext().prec = 50

PINS = {
    "result": "adf0648fa3b9b1ac2d085d094fb060cfe57ed376bad49c808c6f8c5c717f2e60",
    "author_source": "4929659c5548fbf1156109d0e8f59eb804130d6b9d29a0f135a72189cef081f6",
    "author_test": "6baab4a7a1be742303ffe11bfd099bcaf883761ce820800071e69005ce4dacdc",
    "m1597_review": "bfa3414ebb69d4a3022182ef7a4989d738c8370a855dff3ce5232c320623c33f",
    "m1125c_review": "348e18ebdcf37f1740bcd8b977885ee86ea5b0a172232413866f2c739879d77c",
    "m1006_review": "d7b30ff3a82a099c080f3aa3dd32c13c1d2d5b5e278112eb9e3b1c24588809ea",
    "m623_review": "9681239182a27192f69bbc59ec48a2bf9f6336e9c8fc0575924964f69fde6b3a",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def reject_duplicate_pairs(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "duplicate JSON key: " + key)
        value[key] = item
    return value


def reject_constant(token):
    raise ValueError("non-finite JSON token: " + token)


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle, object_pairs_hook=reject_duplicate_pairs,
                         parse_constant=reject_constant)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def d(value):
    return Decimal(str(value))


def require_regular_pinned(path, digest, label):
    require(os.path.isfile(path) and not os.path.islink(path),
            label + " is absent, non-regular or symlinked")
    require(sha256(path) == digest, label + " SHA mismatch")


def build_audit(project_root):
    hw = os.path.join(project_root, "hw_autoresearch_nts07")
    paths = {
        "result": os.path.join(hw, "results",
            "m1607_ep34_c1_parent_partial_energy_model_r1_20260901",
            "m1607_ep34_c1_parent_partial_energy_model_result_r1.json"),
        "author_source": os.path.join(hw, "system_simulator", "scripts",
            "build_m1607_ep34_c1_parent_partial_energy_model.py"),
        "author_test": os.path.join(hw, "system_simulator", "tests",
            "test_m1607_ep34_c1_parent_partial_energy_model.py"),
        "m1597_review": os.path.join(hw, "reviews",
            "m1597_m1590_ep34_c1_same_ledger_cycle_model_result_hammer_r1_20260901",
            "review.json"),
        "m1125c_review": os.path.join(hw, "reviews",
            "m1125c_c1_path_c_105macro_common_model_first_principles_audit_r1_20260830",
            "review.json"),
        "m1006_review": os.path.join(hw, "reviews",
            "m1006_m993_m989_m962_recovered_c1_component_result_hammer_r1_20260829",
            "review.json"),
        "m623_review": os.path.join(hw, "reviews",
            "m623_m617_m597_m593_parent_scratch_energy_r5_result_hammer_r1_20260828",
            "review.json"),
        "docs359": os.path.join(hw, "docs", "359_DATE终局冻结_20260813.md")
    }
    for label, path in paths.items():
        require_regular_pinned(path, PINS[label], label)

    result = load_json(paths["result"])
    cycle = load_json(paths["m1597_review"])
    memory = load_json(paths["m1125c_review"])
    physical = load_json(paths["m1006_review"])
    m623 = load_json(paths["m623_review"])

    require(result["schema"] == "m1607_ep34_c1_parent_partial_energy_model_r1_v1",
            "M1607 schema mismatch")
    require(result["status"] ==
            "PASS_M1607_EP34_C1_PARENT_DYNAMIC_PLUS_CAPACITY_LEAKAGE_PARTIAL_MODEL",
            "M1607 status mismatch")
    require(cycle["ratio_of_sums_rederivation"]["candidate_cycles"] == 382848700,
            "candidate cycle authority mismatch")
    traffic = cycle["conservation_and_traffic"]
    require(traffic["traffic_scope"] ==
            "parent scratch only; not total SRAM or DRAM traffic",
            "traffic scope drift")
    read_bytes = int(traffic["parent_read_bytes_all_eight_blocks"])
    write_bytes = int(traffic["parent_write_bytes_all_eight_blocks"])
    require((read_bytes, write_bytes) == (16711429248, 10449510912),
            "parent byte authority mismatch")

    capacity = memory["capacity_equivalent_model"]
    coefficients = memory["energy_coefficients"]
    require(capacity["cell"] == "TS1N28HPCPHVTB128X128M4S" and
            capacity["word_bits"] == 128 and capacity["port"] == "1RW" and
            capacity["native_macro_equivalents"] == 105 and
            capacity["physical_integration"] is False,
            "capacity model identity/boundary mismatch")
    require(coefficients["source"] ==
            "M623 independently hammered generated-macro datasheet component model",
            "coefficient source mismatch")
    require(memory["authorities"]["m623_review_sha256"] == PINS["m623_review"] and
            m623["status"] == "PASS_M623_M617_R5_BOUNDED_GENERATED_MACRO_COMPONENT_RESULT",
            "M623 coefficient chain mismatch")
    require(physical["anchors"]["clock_period_ns"] == 3.0 and
            physical["anchors"]["setup_met"] is True and
            physical["claim_boundary"]["power"] is False,
            "3 ns setup-only coordinate mismatch")

    vector_bytes = 144
    native_word_bytes = 16
    native_macros_per_vector = 9
    require(read_bytes % vector_bytes == 0 and write_bytes % vector_bytes == 0,
            "parent bytes do not represent whole 144B vector accesses")
    read_vector_accesses = read_bytes // vector_bytes
    write_vector_accesses = write_bytes // vector_bytes
    read_activations = read_vector_accesses * native_macros_per_vector
    write_activations = write_vector_accesses * native_macros_per_vector
    require(read_activations == read_bytes // native_word_bytes and
            write_activations == write_bytes // native_word_bytes,
            "144B-vector and 16B-word activation derivations disagree")
    require((read_activations, write_activations) == (1044464328, 653094432),
            "macro activation count mismatch")

    read_pj = d(coefficients["native_read_pj_per_activated_macro"])
    write_pj = d(coefficients["native_write_pj_per_activated_macro"])
    native_leakage_mw = d(capacity["native_leakage_power_mw"])
    require(read_pj == d("10.50786") and write_pj == d("10.07307") and
            native_leakage_mw == d("0.06001047"),
            "energy coefficient drift")
    read_dynamic_pj = d(read_activations) * read_pj
    write_dynamic_pj = d(write_activations) * write_pj
    parent_dynamic_pj = read_dynamic_pj + write_dynamic_pj
    samples = d(10)
    aggregate_time_s = d(382848700) * d("3.0") * d("1e-9")
    parent_9macro_leakage_mj = native_leakage_mw * d(9) * aggregate_time_s
    full_105macro_leakage_mj = native_leakage_mw * d(105) * aggregate_time_s
    parent_dynamic_mj = parent_dynamic_pj / d("1e9")
    known_partial_mj = parent_dynamic_mj + full_105macro_leakage_mj

    expected = {
        "aggregate_modeled_time_s": aggregate_time_s,
        "parent_dynamic_mj_aggregate_10_samples": parent_dynamic_mj,
        "parent_dynamic_mj_per_sample": parent_dynamic_mj / samples,
        "parent_9macro_leakage_mj_per_sample": parent_9macro_leakage_mj / samples,
        "full_105macro_capacity_leakage_mj_per_sample":
            full_105macro_leakage_mj / samples,
        "known_partial_parent_dynamic_plus_full_capacity_leakage_mj_per_sample":
            known_partial_mj / samples
    }
    for key, value in expected.items():
        require(d(result["energy"][key]) == value,
                "M1607 energy mismatch: " + key)
    parent = result["parent_sram"]
    require(parent["vector_bytes"] == 144 and
            parent["native_macros_per_vector"] == 9 and
            parent["native_macro_word_bytes"] == 16 and
            parent["read_macro_activations"] == read_activations and
            parent["write_macro_activations"] == write_activations,
            "M1607 parent geometry/count mismatch")
    require(d(result["energy"]["known_partial_parent_dynamic_plus_full_capacity_leakage_mj_per_sample"]) ==
            d(result["energy"]["parent_dynamic_mj_per_sample"]) +
            d(result["energy"]["full_105macro_capacity_leakage_mj_per_sample"]),
            "known partial double-counted or omitted leakage")

    boundary = result["claim_boundary"]
    require(boundary["component_energy_model"] is True and
            boundary["capacity_equivalent_leakage_model"] is True,
            "model-positive boundary missing")
    for key in ["weight_dynamic", "psum_dynamic", "metadata_dynamic",
                "logic_dynamic_or_leakage", "dram_energy", "total_c1_energy",
                "energy_per_full_frame", "system_energy", "measured_power"]:
        require(boundary[key] is False, "illegal positive boundary: " + key)
    require(boundary["paper_citable_after_independent_review"] is False,
            "author result self-authorized before independent review")
    require(result["scope"] == {
        "checkpoint": "Motion C12 ep34 live93",
        "clock_period_ns": "3.0",
        "operators": "four bottleneck Conv3x3",
        "samples": 10,
        "sequence": "zurich_city_09_a"
    }, "scope mismatch")

    return {
        "schema": "m1608_m1607_c1_parent_partial_energy_independent_audit_r1_v1",
        "status": "PASS_INDEPENDENT_DECIMAL_RECOMPUTE__PARTIAL_COMPONENT_MODEL_ONLY",
        "date": "2026-09-01",
        "identity": dict((key + "_sha256", value) for key, value in PINS.items()),
        "activation_recompute": {
            "read_bytes": read_bytes,
            "write_bytes": write_bytes,
            "read_vector_accesses_144B": read_vector_accesses,
            "write_vector_accesses_144B": write_vector_accesses,
            "native_macros_per_vector": native_macros_per_vector,
            "native_word_bytes": native_word_bytes,
            "read_macro_activations": read_activations,
            "write_macro_activations": write_activations,
            "both_vector_and_word_derivations_agree": True
        },
        "coefficient_recompute": {
            "cell": capacity["cell"],
            "corner": capacity["corner"],
            "read_pj_per_activated_macro": str(read_pj),
            "write_pj_per_activated_macro": str(write_pj),
            "native_leakage_mw": str(native_leakage_mw),
            "source": coefficients["source"]
        },
        "energy_recompute": {
            "read_dynamic_pj_aggregate_10_samples": str(read_dynamic_pj),
            "write_dynamic_pj_aggregate_10_samples": str(write_dynamic_pj),
            "parent_dynamic_mj_aggregate_10_samples": str(parent_dynamic_mj),
            "parent_dynamic_mj_per_captured_sample": str(parent_dynamic_mj / samples),
            "aggregate_modeled_time_s": str(aggregate_time_s),
            "parent_9macro_leakage_mj_per_captured_sample":
                str(parent_9macro_leakage_mj / samples),
            "full_105macro_capacity_leakage_mj_per_captured_sample":
                str(full_105macro_leakage_mj / samples),
            "known_partial_mj_per_captured_sample": str(known_partial_mj / samples),
            "known_partial_uses_full_105macro_leakage_once": True,
            "parent_9macro_leakage_is_diagnostic_not_added_again": True
        },
        "review_findings": {
            "p0_count": 0,
            "p1_count": 1,
            "p1": "Exact paper label is mandatory because the author JSON does not carry the macro cell/corner and non-integrated-capacity caveat in its top-level scope. Cite only as a one-sequence ten-captured-sample four-Conv parent-dynamic plus 105-macro capacity-equivalent leakage component [model]."
        },
        "claim_boundary": {
            "component_partial_model_admitted_with_exact_label": True,
            "total_c1_energy": False,
            "energy_per_full_frame": False,
            "system_energy": False,
            "measured_power": False,
            "physical_105macro_integration": False,
            "weight_psum_metadata_logic_dram_dynamic_complete": False,
            "sample_is_camera_frame": False,
            "new_eda": False,
            "author_result_modified": False,
            "docs359_modified": False
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-frozen")
    args = parser.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    value = build_audit(project_root)
    if args.check_frozen:
        require(load_json(args.check_frozen) == value,
                "frozen audit differs from independent recompute")
        print("PASS_M1608_FROZEN_AUDIT_MATCH")
    else:
        print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
