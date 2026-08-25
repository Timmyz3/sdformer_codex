#!/usr/bin/env python3
import hashlib
import json
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
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key: " + key)
            value[key] = item
        return value

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=hook, parse_constant=reject)


def require(condition, message):
    if not condition:
        raise SystemExit("FAIL " + message)


paths = {
    "analyzer": HW / ("system_simulator/scripts/"
                       "analyze_m129_row_admission_bubble_and_descriptor_cost.py"),
    "contract": HW / ("contracts/"
                       "m129_row_admission_bubble_and_descriptor_cost_contract_r1_20260824.json"),
    "result": HW / ("results/"
                     "m129_row_admission_bubble_and_descriptor_cost_r1_20260824/"
                     "m129_row_admission_bubble_and_descriptor_cost.json"),
    "m122_script": HW / ("system_simulator/scripts/"
                          "analyze_m122_w384_row_synchronous_source_fold.py"),
    "m122_result": HW / ("results/"
                          "m122_w384_row_synchronous_source_fold_dse_r1_20260824/"
                          "m122_w384_row_synchronous_source_fold_dse.json"),
    "m128_receipt": HW / ("dc_handoff/runs/"
                           "m128_descriptor_streamed_k4_row_fold_vcs_r1_sealed_20260824/"
                           "RUN_COMPLETE.txt"),
    "m128_overlay": HW / ("contracts/"
                           "m128_r1_independent_review_correction_overlay_r1_20260824.json"),
    "docs_359": HW / "docs/359_DATE终局冻结_20260813.md",
}
expected = {
    "analyzer": "b755cc5492f6fabde359363454566265cdcd26d146c8984acc4a8e45764f66e1",
    "contract": "f7bb5038bfea128b87a9df899cd93cf1e66ef66a60226b2e537cd83384c9f777",
    "result": "2443a651675763c9e867a2186e83440c323cf20e381e7a49724d6cb0d9ab411e",
    "m122_script": "ecf2ae43e1282ac483b6832f5a21af6d1b6259c3595eb6150e840f0dc7a55cd3",
    "m122_result": "be11341211b92d85dc42cb7b79b98a826a782765a4780e1207e7bad5368d27b2",
    "m128_receipt": "d9e320092d381999ec158fa31d8aaf32be47c02283d50e3e7ba463cfd7751f28",
    "m128_overlay": "e646cc71cc62ce0d50c128c1a57db9a59221909948413ad3493bfc23cf3d44ec",
    "docs_359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
for name, path in paths.items():
    require(path.is_file(), name + " missing")
    require(sha256(path) == expected[name], name + " SHA drift")

production = strict_json(paths["result"])
contract = strict_json(paths["contract"])
m122 = strict_json(paths["m122_result"])
overlay = strict_json(paths["m128_overlay"])
independent = strict_json(
    REVIEW / "independent_result/m129_independent_recompute.json")
negative = strict_json(REVIEW / "m129_identity_negative_tests.json")
rerun = REVIEW / ("production_rerun/"
                  "m129_row_admission_bubble_and_descriptor_cost.json")
require(sha256(rerun) == expected["result"],
        "production rerun not byte-identical")

histogram = {int(key): int(value) for key, value in
             m122["same_row_source_count_histogram"].items()}
events = sum(count * occurrences
             for count, occurrences in histogram.items())
active = sum(occurrences for count, occurrences in histogram.items()
             if count != 0)
k4 = sum(((count + 3) // 4) * occurrences
         for count, occurrences in histogram.items())
nonempty = int(production["exact_work"]["active_correction_descriptors"])
require(events == 188148490, "independent events")
require(active == 94735083, "independent active row-blocks")
require(k4 == 99847888, "independent K4 descriptors")
require(nonempty == 68820, "nonempty partition windows")
require(independent["histogram_algebra"]["events"] == events
        and independent["histogram_algebra"]["active_row_blocks"] == active
        and independent["histogram_algebra"]["k4_descriptors"] == k4,
        "independent histogram receipt")
require(independent["raw_trace_work"]["active_row_blocks"] == active
        and independent["raw_trace_work"]["k4_descriptors"] == k4
        and independent["raw_trace_work"]["nonempty_partition_windows"]
        == nonempty, "independent raw trace receipt")

cycles = independent["candidate_cycles"]
require(cycles == {
    "m122_ideal": 351410711,
    "m125_m127_row_mask": 446132870,
    "m128_conservative_descriptor": 351479358,
}, "independent candidate cycles")
require(independent["cycle_charge_identities"] == {
    "m122_folded_event_cycles": k4,
    "row_mask_folded_event_cycles": k4 + active,
    "m128_folded_event_cycles": k4 + nonempty,
}, "cycle charge identities")

speedup = cycles["m125_m127_row_mask"] / cycles[
    "m128_conservative_descriptor"]
require(abs(speedup - 1.2693003439479367) < 1e-15, "speedup")
traffic = independent["descriptor_traffic"]
require(traffic["row_mask_bits_each"] == 44
        and traffic["m128_bits_each"] == 53
        and traffic["m130_proposed_bits_each"] == 35,
        "descriptor widths")
require(traffic["m128_total_bits"] > traffic["row_mask_total_bits"],
        "M128 traffic should increase")
require(abs(traffic["m128_total_fraction_vs_row_mask"]
            - 1.2695541696666233) < 1e-15, "M128 total traffic fraction")
require(abs(traffic["m130_proposed_total_fraction_vs_row_mask"]
            - 0.8383848290251286) < 1e-15,
        "M130 proposed total traffic fraction")

boundary = production["model_boundary"]
require(boundary[
    "m125_m127_charge_one_row_admission_cycle_per_active_row_block"],
    "row-mask admission boundary")
require(boundary[
    "m128_charge_one_pipeline_startup_cycle_per_nonempty_partition_window"],
    "M128 startup boundary")
for key in ("external_descriptor_generation_cycles",
            "descriptor_storage_memory_energy", "foundry_weight_macro",
            "physical_speedup", "system_speedup", "headline"):
    require(boundary[key] is False, "model boundary: " + key)
for key in ("physical_speedup", "system_speedup", "headline"):
    require(contract["admission"][key] is False,
            "contract admission: " + key)
require(contract["admission"]["m128_bandwidth_reduction"] is False,
        "contract bandwidth boundary")
require(contract["admission"]["m130_bandwidth_reduction"] is False,
        "contract proposed M130 boundary")
require(overlay["admission"]["external_descriptor_producer_implemented"]
        is False, "overlay producer boundary")

require(negative["direct_m122_result_drift"]["rejected"] is True,
        "direct identity negative")
require(negative["transitive_m109_script_drift"]["rejected"] is False,
        "transitive identity negative")
require(negative["m128_correction_overlay"]["pinned_by_analyzer"] is False,
        "overlay pin negative")

audit = {
    "schema": "m129_row_admission_bubble_and_descriptor_cost_independent_hammer_v1",
    "status": "PASS_NUMERIC_AND_BOUNDARY_WITH_IDENTITY_CORRECTIONS_REQUIRED",
    "date": "2026-08-24",
    "score": {"total": 88, "out_of": 100,
              "p0": 0, "p1": 2, "p2": 2},
    "frozen_identity": {name + "_sha256": digest
                        for name, digest in expected.items()},
    "production_rerun": {
        "status": "PASS_BYTE_IDENTICAL",
        "result_sha256": sha256(rerun),
    },
    "independent_exact_work": {
        "events": events,
        "active_row_blocks": active,
        "k4_descriptors": k4,
        "nonempty_partition_windows": nonempty,
        "histogram_and_raw_trace_match": True,
    },
    "cycle_models": {
        "m122_ideal_candidate_cycles": cycles["m122_ideal"],
        "m125_m127_row_mask_candidate_cycles":
            cycles["m125_m127_row_mask"],
        "m128_conservative_descriptor_candidate_cycles":
            cycles["m128_conservative_descriptor"],
        "m128_vs_row_mask_speedup": speedup,
        "scope": "heldout same-clock module-cycle model A/B only",
        "physical_speedup": False,
        "system_speedup": False,
        "headline": False,
    },
    "admission_charge_model": {
        "row_mask": {
            "charge": "one cycle per active (row, output-block)",
            "charges": active,
            "folded_event_cycles": k4 + active,
        },
        "m128": {
            "charge": "one startup cycle per nonempty (partition, window)",
            "charges": nonempty,
            "folded_event_cycles": k4 + nonempty,
            "external_descriptor_generation_cycles": False,
        },
    },
    "descriptor_traffic": {
        "row_mask": {"bits_each": 44,
                     "field_sum": "block3+row9+mask16+negate16",
                     "total_bits": traffic["row_mask_total_bits"]},
        "m128": {"bits_each": 53,
                 "field_sum": "block3+row9+valid4+ids16+negate4+selected16+last1",
                 "per_item_increase_fraction": 53 / 44 - 1,
                 "total_bits": traffic["m128_total_bits"],
                 "total_increase_fraction":
                     traffic["m128_total_fraction_vs_row_mask"] - 1,
                 "bandwidth_reduction": False},
        "m130": {"bits_each": 35,
                 "field_sum": "block3+row9+count2+ids16+negate4+last1",
                 "total_bits": traffic["m130_proposed_total_bits"],
                 "total_fraction_vs_row_mask":
                     traffic["m130_proposed_total_fraction_vs_row_mask"],
                 "m129_status": "proposed_only",
                 "m129_bound_implementation_or_verification": False},
        "payload_only_not_complete_interface_or_energy": True,
    },
    "identity_negative_tests": negative,
    "priorities": {
        "p0": [],
        "p1": [
            "M129 executes transitive M109/M108/M105 replay helpers without enforcing their embedded/frozen SHA identities; a semantic-no-op M109 SHA drift passes and emits the sealed numeric result.",
            "M129 does not pin the active M128 independent-review correction overlay, so descriptor-producer, canonicality, completion-token and ready/valid restrictions are not provenance-closed with the cycle claim.",
        ],
        "p2": [
            "The one-cycle row-mask and one-cycle nonempty-window charges are explicit model assumptions; external descriptor generation, storage energy, buffering and physical frequency are excluded.",
            "M128 is 53 vs 44 bits and increases heldout payload traffic by 26.9554%; the 35-bit successor is only a proposed M129 what-if and cannot repair M128 bandwidth claims.",
        ],
    },
    "safe_statement": (
        "Independent histogram algebra and raw-trace recurrence reproduce "
        "94,735,083 active row-blocks, 99,847,888 K4 descriptors and all "
        "three candidate-cycle totals. On this held-out same-clock module "
        "model only, charging one row-mask admission per active row-block "
        "versus one M128 startup per nonempty partition-window gives "
        "1.269300344x. M128's 53-bit payload increases modeled traffic "
        "versus the 44-bit row mask; the 35-bit successor remains proposed."
    ),
}
output = REVIEW / "m129_row_admission_bubble_and_descriptor_cost_independent_audit.json"
output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n",
                  encoding="utf-8")
print("PASS M129 independent audit sha256=" + sha256(output))
