#!/usr/bin/env python3
"""Build the final-ep34 C1 parent-SRAM partial energy model."""
from __future__ import annotations

import argparse
from decimal import Decimal, getcontext
import hashlib
import json
from pathlib import Path


getcontext().prec = 40
ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M1597 = HW / "reviews/m1597_m1590_ep34_c1_same_ledger_cycle_model_result_hammer_r1_20260901/review.json"
M1125C = HW / "reviews/m1125c_c1_path_c_105macro_common_model_first_principles_audit_r1_20260830/review.json"
M1006 = HW / "reviews/m1006_m993_m989_m962_recovered_c1_component_result_hammer_r1_20260829/review.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
PINS = {
    M1597: "bfa3414ebb69d4a3022182ef7a4989d738c8370a855dff3ce5232c320623c33f",
    M1125C: "348e18ebdcf37f1740bcd8b977885ee86ea5b0a172232413866f2c739879d77c",
    M1006: "d7b30ff3a82a099c080f3aa3dd32c13c1d2d5b5e278112eb9e3b1c24588809ea",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + token)))


def d(value) -> Decimal:
    return Decimal(str(value))


def build():
    for path, digest in PINS.items():
        require(path.is_file() and not path.is_symlink() and sha256(path) == digest,
                "identity drift: " + str(path))
    cycle = load(M1597)
    memory = load(M1125C)
    physical = load(M1006)
    require(cycle["status"] ==
            "PASS_M1597_M1590_EP34_C1_RESULT_HAMMER_WITH_CAPACITY_SUPERSESSION" and
            cycle["ratio_of_sums_rederivation"]["candidate_cycles"] == 382_848_700,
            "M1597 cycle authority drift")
    traffic = cycle["conservation_and_traffic"]
    read_bytes = int(traffic["parent_read_bytes_all_eight_blocks"])
    write_bytes = int(traffic["parent_write_bytes_all_eight_blocks"])
    require((read_bytes, write_bytes) == (16_711_429_248, 10_449_510_912) and
            traffic["traffic_scope"] == "parent scratch only; not total SRAM or DRAM traffic",
            "M1597 traffic drift")
    coefficients = memory["energy_coefficients"]
    capacity = memory["capacity_equivalent_model"]
    require(capacity["native_macro_bytes"] == 2048 and
            capacity["native_macro_equivalents"] == 105 and
            capacity["model_capacity_bytes"] == 215040 and
            coefficients["three_axis_totals_available_now"] is False,
            "M1125C boundary drift")
    require(physical["anchors"]["clock_period_ns"] == 3.0 and
            physical["anchors"]["setup_met"] is True,
            "M1006 3ns setup coordinate drift")

    bytes_per_native_activation = 16
    read_activations = read_bytes // bytes_per_native_activation
    write_activations = write_bytes // bytes_per_native_activation
    require(read_bytes % 16 == 0 and write_bytes % 16 == 0,
            "nonintegral native macro activation")
    read_pj = d(coefficients["native_read_pj_per_activated_macro"])
    write_pj = d(coefficients["native_write_pj_per_activated_macro"])
    parent_dynamic_pj = d(read_activations) * read_pj + d(write_activations) * write_pj
    samples = d(10)
    cycles = d(cycle["ratio_of_sums_rederivation"]["candidate_cycles"])
    clock_ns = d(physical["anchors"]["clock_period_ns"])
    aggregate_time_s = cycles * clock_ns * d("1e-9")
    parent_leakage_mw = d(capacity["native_leakage_power_mw"]) * d(9)
    full_storage_leakage_mw = d(capacity["common_leakage_power_model_mw"])
    parent_leakage_mj = parent_leakage_mw * aggregate_time_s
    full_storage_leakage_mj = full_storage_leakage_mw * aggregate_time_s
    parent_dynamic_mj = parent_dynamic_pj / d("1e9")
    known_partial_mj = parent_dynamic_mj + full_storage_leakage_mj

    return {
        "schema": "m1607_ep34_c1_parent_partial_energy_model_r1_v1",
        "status": "PASS_M1607_EP34_C1_PARENT_DYNAMIC_PLUS_CAPACITY_LEAKAGE_PARTIAL_MODEL",
        "scope": {"checkpoint": "Motion C12 ep34 live93",
                  "sequence": "zurich_city_09_a", "samples": 10,
                  "operators": "four bottleneck Conv3x3",
                  "clock_period_ns": str(clock_ns)},
        "parent_sram": {"vector_bytes": 144, "native_macros_per_vector": 9,
                        "native_macro_word_bytes": bytes_per_native_activation,
                        "read_bytes": read_bytes, "write_bytes": write_bytes,
                        "read_macro_activations": read_activations,
                        "write_macro_activations": write_activations,
                        "read_pj_per_activation": str(read_pj),
                        "write_pj_per_activation": str(write_pj)},
        "energy": {
            "parent_dynamic_mj_aggregate_10_samples": str(parent_dynamic_mj),
            "parent_dynamic_mj_per_sample": str(parent_dynamic_mj / samples),
            "parent_9macro_leakage_mj_per_sample": str(parent_leakage_mj / samples),
            "full_105macro_capacity_leakage_mj_per_sample": str(full_storage_leakage_mj / samples),
            "known_partial_parent_dynamic_plus_full_capacity_leakage_mj_per_sample":
                str(known_partial_mj / samples),
            "aggregate_modeled_time_s": str(aggregate_time_s)},
        "identity": {path.relative_to(ROOT).as_posix(): digest for path, digest in PINS.items()},
        "claim_boundary": {
            "component_energy_model": True,
            "parent_dynamic_complete_for_candidate": True,
            "capacity_equivalent_leakage_model": True,
            "weight_dynamic": False, "psum_dynamic": False,
            "metadata_dynamic": False, "logic_dynamic_or_leakage": False,
            "dram_energy": False, "total_c1_energy": False,
            "energy_per_full_frame": False, "system_energy": False,
            "measured_power": False, "paper_citable_after_independent_review": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    value = build()
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out is None:
        print(payload, end="")
    else:
        require(not args.out.exists(), "refuse overwrite")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
