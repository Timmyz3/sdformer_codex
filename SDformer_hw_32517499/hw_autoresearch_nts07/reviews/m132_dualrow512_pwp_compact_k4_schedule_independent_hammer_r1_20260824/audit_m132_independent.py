#!/usr/bin/env python3
import hashlib
import json
import math
from pathlib import Path


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parent.parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: " + raw)

    def hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=hook, parse_constant=reject)


def require(condition, message):
    if not condition:
        raise SystemExit("FAIL " + message)


paths = {
    "analyzer": HW / ("system_simulator/scripts/"
                       "analyze_m132_dualrow512_pwp_compact_k4_schedule.py"),
    "contract": HW / ("contracts/"
                       "m132_dualrow512_pwp_compact_k4_schedule_contract_r1_20260824.json"),
    "result": HW / ("results/"
                     "m132_dualrow512_pwp_compact_k4_schedule_r1_20260824/"
                     "m132_dualrow512_pwp_compact_k4_schedule.json"),
    "m122_script": HW / ("system_simulator/scripts/"
                          "analyze_m122_w384_row_synchronous_source_fold.py"),
    "m122_result": HW / ("results/"
                          "m122_w384_row_synchronous_source_fold_dse_r1_20260824/"
                          "m122_w384_row_synchronous_source_fold_dse.json"),
    "m129_result": HW / ("results/"
                          "m129_row_admission_bubble_and_descriptor_cost_r1_20260824/"
                          "m129_row_admission_bubble_and_descriptor_cost.json"),
    "m129_overlay": HW / ("contracts/"
                           "m129_r1_independent_review_identity_correction_r1_20260824.json"),
    "m129_review": HW / ("reviews/"
                          "m129_row_admission_bubble_and_descriptor_cost_independent_hammer_r1_20260824/"
                          "manifest.sha256"),
    "m131_receipt": HW / ("dc_handoff/runs/"
                           "m131_synthesis_safe_compact_canonical_k4_row_fold_vcs_r1_sealed_20260824/"
                           "RUN_COMPLETE.txt"),
    "m109_script": HW / ("system_simulator/scripts/"
                          "analyze_m109_r2_window_storage_dual_timeline_frontier.py"),
    "m109_result": HW / ("results/"
                          "m109_r2_window_storage_dual_timeline_frontier_r1_20260824/"
                          "m109_r2_window_storage_dual_timeline_frontier.json"),
    "m108_script": HW / ("system_simulator/scripts/"
                          "analyze_m108_w64_fused_pwp_accumulator_schedule.py"),
    "m105_script": HW / ("reviews/"
                          "m105_bounded_row_transpose_preflight_independent_hammer_r1_20260824/"
                          "audit_m105_bounded_row_transpose.py"),
    "m40_manifest": HW / ("results/"
                           "m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822/"
                           "m40_bottleneck_packed_source_manifest.json"),
    "m72_result": HW / ("results/"
                         "m72_phi_kmeans_k16q16_valid825_internal_screen_dev_r1_20260823/"
                         "m72_phi_kmeans_k16q16_valid825_internal_screen.json"),
    "m41_result": HW / ("results/"
                         "m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/"
                         "m41_h67_ep35_bottleneck_int8_bridge.json"),
    "m115r2_result": HW / ("results/"
                            "m115r2_pwp_prefix_coefficient_width_r1_20260824/"
                            "m115r2_pwp_prefix_coefficient_width.json"),
    "docs_359": HW / "docs/359_DATE终局冻结_20260813.md",
}
expected = {
    "analyzer": "f140b6b72559f04cdac374eaf696c3f6650b20d3b00bd580419b88494d89c952",
    "contract": "2e6033ea5adc27a692ec5588d6d52af11292ff95e984bdcc3ced939b92a7b7fd",
    "result": "f74444576ec487b9b1034aced7add0da868a9dea5d4185e0a62c1e33fe1ad755",
    "m122_script": "ecf2ae43e1282ac483b6832f5a21af6d1b6259c3595eb6150e840f0dc7a55cd3",
    "m122_result": "be11341211b92d85dc42cb7b79b98a826a782765a4780e1207e7bad5368d27b2",
    "m129_result": "2443a651675763c9e867a2186e83440c323cf20e381e7a49724d6cb0d9ab411e",
    "m129_overlay": "9b4073183c8ecd541758a693472b1b2c92f829de915d836428a2b9e5e7a9968d",
    "m129_review": "eeada044c1199099de574dc8ed131bc81c33e0063d1581cd312c9f4649bd284d",
    "m131_receipt": "e30e273ff791475d7f015ae4fb580a8c5fa0b018a432adf666519ffd44184316",
    "m109_script": "4eed1e1ef25cdbea0fdd40d1602d6b1eb7661b15b5ae47541c80e149fd060ada",
    "m109_result": "ee61b90ee894c6e6c778b815a52f1d8b6edc9c877227bc4987e4b135aa16c321",
    "m108_script": "4404e5825ece95fbf0a28dd580c03c7e9f34bcfa9ec12fa3b66d226a9042cbe2",
    "m105_script": "5e5c07631dd8c4bb328cd234da5c04fde8eb9800d1516b3fe462124b2b661ed5",
    "m40_manifest": "e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3",
    "m72_result": "e3f40697e1b1442d3b190c3aa2cc540ee5892a5db37366808d97d7c635250133",
    "m41_result": "20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb",
    "m115r2_result": "b0e7fbb0573473ad854ca856d5eab3eaf15af1ba79ea2ce3a958810575bc6708",
    "docs_359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
for label, path in paths.items():
    require(path.is_file() and sha256(path) == expected[label],
            label + " identity drift")

production = strict_json(paths["result"])
contract = strict_json(paths["contract"])
m129 = strict_json(paths["m129_result"])
independent = strict_json(REVIEW / "independent_result/m132_independent_recompute.json")
negative = strict_json(REVIEW / "m132_identity_negative_tests.json")
rerun = REVIEW / ("production_rerun/"
                  "m132_dualrow512_pwp_compact_k4_schedule.json")
require(sha256(rerun) == expected["result"], "production rerun byte compare")

uses = {8: 11164284, 9: 32360036, 10: 13936011, 11: 1509043}
beat256 = {8: 3, 9: 4, 10: 4, 11: 5}
beat512 = {8: 2, 9: 2, 10: 2, 11: 3}
tokens256 = sum(uses[width] * beat256[width] for width in uses)
tokens512 = sum(uses[width] * beat512[width] for width in uses)
require(tokens256 == 226222255 and tokens512 == 119447791,
        "token algebra")
require(independent["token_algebra"]["pwp256_tokens"] == tokens256
        and independent["token_algebra"]["pwp512_tokens"] == tokens512
        and independent["token_algebra"]["raw_trace_matches"],
        "independent token receipt")

c256 = independent["candidate_cycles"]["compact256"]
c512 = independent["candidate_cycles"]["dualrow512"]
ratio = independent["comparisons"]["dualrow512_vs_compact256"]
fixed8 = independent["comparisons"]["fixed8_same_clock_service_island"]
require(c256 == 351479358 and c512 == 245485910, "candidate cycles")
require(abs(ratio - 1.4317699863100086) < 1e-15, "cycle ratio")
require(abs(fixed8 - 4.541455955659533) < 1e-15, "fixed8 ratio")
require(independent["compact256_exactly_reproduces_m129"],
        "compact256 M129 receipt")
require(production["cycle_models"]["compact_k4_pwp256"]
        == m129["cycle_models"]["m128_descriptor_conservative_startup"],
        "production compact256 does not exactly reproduce M129")

boundary = production["model_boundary"]
for key in ("dualrow512_pwp_rtl", "bank_conflicts_modeled",
            "foundry_dualrow_or_16bank_macro", "macro_area_energy",
            "matched_dc_frequency", "physical_speedup", "system_speedup",
            "headline"):
    require(boundary[key] is False, "model boundary: " + key)
for key in ("dualrow512_pwp_rtl", "bank_conflict_freedom",
            "foundry_dualrow_or_16bank_macro", "macro_area_energy",
            "matched_dc_frequency", "physical_speedup", "system_speedup",
            "headline"):
    require(contract["admission"][key] is False,
            "contract admission: " + key)

m131_text = paths["m131_receipt"].read_text(encoding="utf-8")
for marker in ("complete_row_partition_losslessness=false",
               "descriptor_producer_implemented=false",
               "synopsys_dc_elaboration_clean=false",
               "physical_speedup=false", "system_speedup=false",
               "headline=false"):
    require(marker in m131_text, "M131 boundary: " + marker)
require(negative["direct_m129_result_drift"]["rejected"] is True,
        "direct identity drift")
require(negative["transitive_m109_result_drift"]["rejected"] is False,
        "M109 result identity fail-open")

vector_geometry = {}
for width in sorted(uses):
    bits = 96 * width
    vector_geometry[str(width)] = {
        "vector_bits": bits,
        "cycles_at_256": math.ceil(bits / 256),
        "cycles_at_512": math.ceil(bits / 512),
    }
require([vector_geometry[str(width)]["cycles_at_256"]
         for width in sorted(uses)] == [3, 4, 4, 5],
        "256 geometry")
require([vector_geometry[str(width)]["cycles_at_512"]
         for width in sorted(uses)] == [2, 2, 2, 3],
        "512 geometry")
break_even_frequency_ratio = 1 / ratio

audit = {
    "schema": "m132_dualrow512_pwp_compact_k4_schedule_independent_hammer_v1",
    "status": "PASS_CYCLE_DSE_PHYSICAL_PORT_UNADMITTED_IDENTITY_CORRECTION_REQUIRED",
    "date": "2026-08-24",
    "score": {"total": 87, "out_of": 100,
              "p0": 0, "p1": 2, "p2": 2},
    "frozen_identity": {label + "_sha256": value
                        for label, value in expected.items()},
    "production_rerun": {"status": "PASS_BYTE_IDENTICAL",
                         "result_sha256": sha256(rerun)},
    "independent_width_algebra": {
        "uses": uses,
        "beats_256": beat256,
        "beats_512": beat512,
        "pwp256_tokens": tokens256,
        "pwp512_tokens": tokens512,
        "token_reduction_fraction": 1 - tokens512 / tokens256,
        "raw_trace_match": True,
    },
    "independent_cycle_recurrence": {
        "implementation_imports_production_fold_schedule": False,
        "compact256_candidate_cycles": c256,
        "dualrow512_candidate_cycles": c512,
        "dualrow512_vs_compact256": ratio,
        "fixed8_same_clock_service_island_ratio": fixed8,
        "compact256_exactly_reproduces_m129": True,
    },
    "free_512bit_port_attack": {
        "vector_geometry": vector_geometry,
        "baseline_port_bits": 256,
        "candidate_port_bits": 512,
        "instantaneous_read_bandwidth_multiplier": 2,
        "candidate_requires_two_independent_256bit_rows_per_cycle": True,
        "baseline_32bit_words_per_cycle": 8,
        "candidate_32bit_words_per_cycle": 16,
        "candidate_requires_16_conflict_free_word_banks_or_equivalent": True,
        "address_to_bank_mapping_provided": False,
        "conflict_trace_or_arbiter_provided": False,
        "dualrow512_rtl": False,
        "foundry_macro": False,
        "area_energy_cost": False,
        "matched_frequency": False,
        "clock_ratio_needed_to_preserve_cycle_speedup":
            break_even_frequency_ratio,
        "interpretation": (
            "Physical throughput improves only if f512/f256 exceeds "
            "0.698436208. Wider read muxing, wires, banking, arbitration, "
            "periphery and switching energy are not free or modeled."
        ),
    },
    "identity_negative_tests": negative,
    "claim_boundary": {
        "scope": "frozen heldout W384 same-clock service-island cycle DSE",
        "dualrow512_rtl": False,
        "bank_conflict_freedom": False,
        "macro_area_energy": False,
        "frequency": False,
        "physical_speedup": False,
        "system_speedup": False,
        "headline": False,
    },
    "priorities": {
        "p0": [],
        "p1": [
            "The 512-bit candidate has no dual-row/16-bank RTL, address-to-bank mapping, conflict proof, foundry macro, or matched Synopsys evidence; physical promotion is blocked.",
            "M132 reads the M109 result for fixed-baseline service tokens but omits that result from frozen_paths; a semantic-no-op SHA drift is accepted and exact-SHA provenance remains open.",
        ],
        "p2": [
            "M131 only proves local compact descriptor VCS: complete row-partition losslessness, descriptor producer implementation and clean DC elaboration are all false.",
            "The same-clock cycle gain survives physically only if the 512-bit implementation retains at least 69.8436 percent of compact256 frequency; macro area and energy must be measured separately.",
        ],
    },
    "safe_statement": (
        "Independent width algebra and descriptor recurrence reproduce the "
        "frozen M132 result: two logical 256-bit rows reduce modeled PWP "
        "tokens from 226,222,255 to 119,447,791 and same-clock heldout "
        "service-island cycles from 351,479,358 to 245,485,910 (1.43177x). "
        "The 4.54146x fixed8 comparison is also same-clock service-island "
        "only. A conflict-free 512-bit read implementation, macro cost, "
        "frequency, energy, physical and system speedup are unadmitted."
    ),
}
output = REVIEW / "m132_dualrow512_pwp_compact_k4_schedule_independent_audit.json"
output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n",
                  encoding="utf-8")
print("PASS M132 independent audit sha256=" + sha256(output))
