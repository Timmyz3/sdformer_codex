#!/usr/bin/env python3
"""M53 exact all-ten adaptive temporal-parent plus spatial K4-C16 DSE.

The canonical M45 source is dynamically loaded.  Temporal configurations make
only three enumerated edits: context capacity 8->16, previous_timestep in the
parent allowlist, and ALLOW_TEMPORAL_PARENT False->True.  Results are an exact
transaction-model experiment, not RTL, PPA, or system speedup.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW_ROOT / (
    "contracts/m53_adaptive_temporal_parent_k4_ctx16_dse_contract_r1_20260823.json")
EXPECTED_CONTRACT_SHA256 = (
    "e1dd6eb10a4b580115ff8cfe9d28605167256dfe81942ea2e2ea92d5fba88e03")
M45_ANALYZER = HW_ROOT / (
    "system_simulator/scripts/analyze_m45_dual_destination_bank_fused_integrated_schedule.py")
CONFIGURATIONS = (
    ("K2_CTX16_TEMPORAL", 2, 16, True,
     "TEMPORAL_CONTEXT_DEPTH_REFERENCE"),
    ("K4_CTX16_SPATIAL", 4, 16, False,
     "EXACT_M52_SPATIAL_REPRODUCTION"),
    ("K4_CTX16_TEMPORAL", 4, 16, True,
     "PRIMARY_ADAPTIVE_TEMPORAL_RTL_EXPERIMENT_CANDIDATE"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical_bytes(payload):
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) +
            "\n").encode("utf-8")


def read_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: {}".format(raw))

    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: {}".format(key))
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def fraction(numerator, denominator):
    require(denominator > 0, "fraction denominator must be positive")
    divisor = math.gcd(int(numerator), int(denominator))
    return {
        "numerator": int(numerator) // divisor,
        "denominator": int(denominator) // divisor,
        "decimal": float(numerator) / float(denominator),
    }


def nearest_rank(values, percentile):
    require(values, "empty nearest-rank population")
    ordered = sorted(values)
    return ordered[int(math.ceil(len(ordered) * percentile)) - 1]


def validate_contract():
    require(sha256_path(CONTRACT) == EXPECTED_CONTRACT_SHA256,
            "M53 contract identity drift")
    contract = read_json(CONTRACT)
    require(contract["schema"] ==
            "m53_adaptive_temporal_parent_k4_ctx16_dse_contract_v1",
            "M53 contract schema drift")
    for name, identity in contract["inputs"].items():
        path = HW_ROOT / identity["path"]
        require(path.is_file() and sha256_path(path) == identity["sha256"],
                "M53 input identity drift: {}".format(name))
    model = contract["capacity_model"]
    require(model["local_capacity_headroom_bytes"] == 17040 and
            model["headroom_unit"] == "bytes" and
            model["margin_above_minimum_headroom_bytes"] == 656,
            "M53 M52 headroom byte contract drift")
    storage = contract["temporal_parent_storage_proof"]
    require(storage["existing_m47_frame_count"] == 2 and
            storage["existing_two_frame_bytes"] == 136800 and
            storage["new_third_frame_bytes"] == 0,
            "M53 dual-frame storage contract drift")
    return contract


def transformed_m45_source(temporal_parent_enabled):
    source = M45_ANALYZER.read_text(encoding="utf-8")
    guard_from = ("require(1 <= fanout_k <= context_capacity <= 8,\n"
                  "            \"invalid fanout/context geometry\")")
    guard_to = ("require(1 <= fanout_k <= context_capacity <= 16,\n"
                "            \"invalid fanout/context geometry\")")
    require(source.count(guard_from) == 1,
            "M53 M45 context guard source identity drift")
    transformed = source.replace(guard_from, guard_to)
    edits = [{"name": "context_guard", "occurrences": 1}]
    if temporal_parent_enabled:
        allow_from = ("require(name in (\"local_zero\", \"left\", \"up\"),\n"
                      "                \"temporal parent leaked into M45 primary\")")
        allow_to = ("require(name in (\"local_zero\", \"left\", \"up\",\n"
                    "                         \"previous_timestep\"),\n"
                    "                \"invalid M53 adaptive parent\")")
        enable_from = "module.ALLOW_TEMPORAL_PARENT = False"
        enable_to = "module.ALLOW_TEMPORAL_PARENT = True"
        require(transformed.count(allow_from) == 1,
                "M53 M45 parent allowlist source identity drift")
        require(transformed.count(enable_from) == 1,
                "M53 M45 temporal enable source identity drift")
        transformed = transformed.replace(allow_from, allow_to)
        transformed = transformed.replace(enable_from, enable_to)
        edits.extend((
            {"name": "parent_allowlist", "occurrences": 1},
            {"name": "m43_temporal_enable", "occurrences": 1},
        ))
    return source, transformed, edits


def load_extended_m45(temporal_parent_enabled):
    source, transformed, edits = transformed_m45_source(
        temporal_parent_enabled)
    namespace = {
        "__file__": str(M45_ANALYZER),
        "__name__": ("m53_temporal_m45" if temporal_parent_enabled else
                     "m53_spatial_m45"),
    }
    suffix = "#M53_TEMPORAL" if temporal_parent_enabled else "#M53_SPATIAL"
    exec(compile(transformed, str(M45_ANALYZER) + suffix, "exec"), namespace)
    require(namespace["schedule_tile_timestep"].__globals__ is namespace,
            "M53 transformed scheduler namespace mismatch")
    m43 = namespace["load_m43_module"]()
    require(bool(m43.ALLOW_TEMPORAL_PARENT) == temporal_parent_enabled,
            "M53 transformed M43 temporal flag mismatch")
    audit = {
        "canonical_m45_sha256": sha256_bytes(source.encode("utf-8")),
        "transformed_source_sha256": sha256_bytes(transformed.encode("utf-8")),
        "temporal_parent_enabled": temporal_parent_enabled,
        "edit_count": len(edits),
        "edits": edits,
        "unlisted_source_edits": 0,
    }
    return namespace, m43, audit


def parent_selection_ledger(m43, masks, expected):
    choices = dict((name, 0) for name in m43.PARENT_PRIORITY)
    issue_cycles = 0
    logical_updates = 0
    add_updates = 0
    subtract_updates = 0
    previous_at_timestep_zero = 0
    previous_after_timestep_zero = 0
    for row in range(m43.ROWS):
        timestep = row // (m43.HEIGHT * m43.WIDTH)
        for tile in range(m43.TILES):
            name, _, add_mask, subtract_mask = m43.select_parent(
                masks, row, tile)
            choices[name] += 1
            delta = add_mask | subtract_mask
            issue_cycles += m43.bank_issue_cycles(delta) * m43.OUTPUT_BLOCKS
            add_count = m43.population(add_mask) * m43.OUTPUT_BLOCKS
            subtract_count = m43.population(subtract_mask) * m43.OUTPUT_BLOCKS
            add_updates += add_count
            subtract_updates += subtract_count
            logical_updates += add_count + subtract_count
            if name == "previous_timestep":
                if timestep == 0:
                    previous_at_timestep_zero += 1
                else:
                    previous_after_timestep_zero += 1
    require(choices == expected["parent_choice_by_tile"],
            "M53 parent choice ledger does not reproduce M43")
    require(issue_cycles == expected["parent_delta_p8_l96_source_issue_cycles"],
            "M53 unfused source ledger does not reproduce M43")
    require(logical_updates ==
            expected["parent_delta_source_destination_pairs"] *
            m43.OUTPUT_BLOCKS,
            "M53 logical update ledger does not reproduce M43")
    require(add_updates ==
            expected["parent_delta_add_pairs"] * m43.OUTPUT_BLOCKS and
            subtract_updates ==
            expected["parent_delta_subtract_pairs"] * m43.OUTPUT_BLOCKS,
            "M53 signed update partition does not reproduce M43")
    require(previous_at_timestep_zero == 0,
            "M53 illegal previous-timestep parent at timestep zero")
    return {
        "parent_choice_by_tile": choices,
        "unfused_parent_delta_source_issue_cycles": issue_cycles,
        "logical_source_updates": logical_updates,
        "signed_add_updates": add_updates,
        "signed_subtract_updates": subtract_updates,
        "previous_timestep_choices_at_timestep_zero": previous_at_timestep_zero,
        "previous_timestep_choices_after_timestep_zero":
            previous_after_timestep_zero,
    }


def analyze_configuration(r1, m43, cached_records, name, fanout_k,
                          contexts, temporal_parent_enabled, qualification):
    per_record = []
    for index, item in enumerate(cached_records):
        record, masks, expected = item
        row = r1["analyze_record"](
            m43, masks, expected, fanout_k, contexts)
        selection = parent_selection_ledger(m43, masks, expected)
        require(row["logical_source_updates"] ==
                selection["logical_source_updates"] and
                row["signed_add_updates"] == selection["signed_add_updates"] and
                row["signed_subtract_updates"] ==
                selection["signed_subtract_updates"],
                "M53 scheduled signed-source conservation drift")
        row["sample_id"] = record["sample_id"]
        row["operator"] = record["operator"]
        row["parent_selection"] = selection
        per_record.append(row)
        print("[M53 {}] {}/40 sample={} operator={}".format(
            name, index + 1, record["sample_id"], record["operator"]),
              flush=True)

    blank = r1["blank_counts"]()
    sum_fields = [field for field in blank
                  if not field.startswith("maximum_")]
    sum_fields += ["signed_add_updates", "signed_subtract_updates",
                   "weight_dma_bytes", "final_accumulator_read_bytes",
                   "final_accumulator_write_bytes", "completed_output_bytes"]
    per_sample = []
    for sample_id in range(10):
        selected = [row for row in per_record
                    if row["sample_id"] == sample_id]
        require(len(selected) == 4, "M53 sample/operator population drift")
        sample = {"sample_id": sample_id}
        for field in sum_fields:
            sample[field] = sum(row[field] for row in selected)
        for field in ("maximum_metadata_occupancy",
                      "maximum_complete_occupancy",
                      "maximum_resident_occupancy"):
            sample[field] = max(row[field] for row in selected)
        sample["parent_choice_by_tile"] = dict(
            (parent, sum(row["parent_selection"]["parent_choice_by_tile"][parent]
                         for row in selected))
            for parent in m43.PARENT_PRIORITY)
        sample["unfused_parent_delta_source_issue_cycles"] = sum(
            row["parent_selection"]["unfused_parent_delta_source_issue_cycles"]
            for row in selected)
        sample["previous_timestep_choices_after_timestep_zero"] = sum(
            row["parent_selection"][
                "previous_timestep_choices_after_timestep_zero"]
            for row in selected)
        sample["integrated_over_source_only"] = fraction(
            sample["integrated_cycles"] - sample["source_only_cycles"],
            sample["source_only_cycles"])
        sample["parent_wait_fraction"] = fraction(
            sample["parent_wait_cycles"], sample["integrated_cycles"])
        per_sample.append(sample)

    result = r1["aggregate_configuration"](
        name, fanout_k, contexts, per_sample)
    result["qualification"] = qualification
    result["previous_timestep_parent_enabled"] = temporal_parent_enabled
    result["parent_selection_aggregate"] = {
        "parent_choice_by_tile": dict(
            (parent, sum(row["parent_selection"]["parent_choice_by_tile"][parent]
                         for row in per_record))
            for parent in m43.PARENT_PRIORITY),
        "unfused_parent_delta_source_issue_cycles": sum(
            row["parent_selection"]["unfused_parent_delta_source_issue_cycles"]
            for row in per_record),
        "previous_timestep_choices_at_timestep_zero": sum(
            row["parent_selection"][
                "previous_timestep_choices_at_timestep_zero"]
            for row in per_record),
        "previous_timestep_choices_after_timestep_zero": sum(
            row["parent_selection"][
                "previous_timestep_choices_after_timestep_zero"]
            for row in per_record),
    }
    result["record_ledger"] = {
        "record_count": len(per_record),
        "canonical_sha256": sha256_bytes(canonical_bytes(per_record)),
        "records": per_record,
    }
    return result


def replay_one_configuration(configuration):
    validate_contract()
    name, fanout_k, contexts, temporal, qualification = configuration
    r1, m43, audit = load_extended_m45(temporal)
    r1["validate_contract"]()
    manifest = r1["read_json"](r1["MANIFEST"])
    reference_path = (
        "results/m43_tile_resident_parent_delta_schedule_r1_20260823/"
        + ("m43_spatiotemporal_parent_delta_ablation.json" if temporal else
           "m43_spatial_parent_delta_schedule_final.json"))
    reference = read_json(HW_ROOT / reference_path)
    reference_records = dict(
        ((row["sample_id"], row["operator"]), row)
        for row in reference["records"])
    require(len(manifest["records"]) == 40 and len(reference_records) == 40,
            "M53 frozen cohort drift")
    cached = []
    for record in manifest["records"]:
        key = (record["sample_id"], record["operator"])
        require(key in reference_records, "M53 M43 reference record drift")
        masks = m43.unpack_record_masks(r1["MANIFEST"].parent, record)
        cached.append((record, masks, reference_records[key]))
    result = analyze_configuration(
        r1, m43, cached, name, fanout_k, contexts, temporal, qualification)
    result["dynamic_source_edit_audit"] = audit
    print("[M53 SUMMARY {}] source={} integrated={} p95={}".format(
        name, result["aggregate_source_only_cycles"],
        result["aggregate_integrated_cycles"],
        result["integrated_cycle_distribution"]["p95_nearest_rank"]),
          flush=True)
    return result


def core_record(row):
    return dict((key, value) for key, value in row.items()
                if key != "parent_selection")


def exact_m52_spatial_reproduction(m52, spatial):
    matches = [row for row in m52["configuration_ledgers"]
               if row["name"] == "K4_CTX16"]
    require(len(matches) == 1, "M53 M52 K4-C16 reference missing")
    reference = matches[0]
    mismatches = []
    for field in ("aggregate_source_only_cycles", "aggregate_integrated_cycles",
                  "source_only_cycle_distribution",
                  "integrated_cycle_distribution"):
        if spatial[field] != reference[field]:
            mismatches.append(field)
    normalized_samples = [dict((key, row[key]) for key in expected)
                          for row, expected in zip(
                              spatial["per_sample"], reference["per_sample"])]
    if normalized_samples != reference["per_sample"]:
        mismatches.append("per_sample")
    actual_records = [core_record(row)
                      for row in spatial["record_ledger"]["records"]]
    if actual_records != reference["record_ledger"]["records"]:
        mismatches.append("record_ledger.records")
    return {
        "reference_result_sha256": sha256_path(
            HW_ROOT / "results/m52_high_fanout_context16_dse_r1_20260823/"
            "m52_high_fanout_context16_dse.json"),
        "configuration": "K4_CTX16",
        "compared_record_count": 40,
        "compared_per_sample_count": 10,
        "mismatch_count": len(mismatches),
        "mismatch_fields": mismatches,
        "exact_match": not mismatches,
    }


def cycle_gain_decomposition(reference, candidate):
    require([row["sample_id"] for row in reference["per_sample"]] ==
            [row["sample_id"] for row in candidate["per_sample"]],
            "M53 comparison sample order drift")
    per_sample = []
    for before, after in zip(reference["per_sample"],
                             candidate["per_sample"]):
        source_gain = before["source_only_cycles"] - after["source_only_cycles"]
        before_overhead = (before["integrated_cycles"] -
                           before["source_only_cycles"])
        after_overhead = (after["integrated_cycles"] -
                          after["source_only_cycles"])
        overhead_gain = before_overhead - after_overhead
        integrated_gain = before["integrated_cycles"] - after["integrated_cycles"]
        require(source_gain + overhead_gain == integrated_gain,
                "M53 cycle-gain decomposition drift")
        per_sample.append({
            "sample_id": before["sample_id"],
            "reference_source_only_cycles": before["source_only_cycles"],
            "candidate_source_only_cycles": after["source_only_cycles"],
            "source_only_cycle_gain": source_gain,
            "reference_non_source_overhead_cycles": before_overhead,
            "candidate_non_source_overhead_cycles": after_overhead,
            "non_source_overhead_cycle_gain": overhead_gain,
            "integrated_cycle_gain": integrated_gain,
            "decomposition_exact": source_gain + overhead_gain == integrated_gain,
        })
    source_gain = (reference["aggregate_source_only_cycles"] -
                   candidate["aggregate_source_only_cycles"])
    reference_overhead = (reference["aggregate_integrated_cycles"] -
                          reference["aggregate_source_only_cycles"])
    candidate_overhead = (candidate["aggregate_integrated_cycles"] -
                          candidate["aggregate_source_only_cycles"])
    overhead_gain = reference_overhead - candidate_overhead
    integrated_gain = (reference["aggregate_integrated_cycles"] -
                       candidate["aggregate_integrated_cycles"])
    require(source_gain + overhead_gain == integrated_gain,
            "M53 aggregate cycle-gain decomposition drift")
    return {
        "aggregate": {
            "source_only_cycle_gain": source_gain,
            "non_source_overhead_cycle_gain": overhead_gain,
            "integrated_cycle_gain": integrated_gain,
            "decomposition_exact": source_gain + overhead_gain == integrated_gain,
        },
        "per_sample": per_sample,
    }


def summary(configuration, capacity):
    return {
        "name": configuration["name"],
        "destination_fanout_k": configuration["destination_fanout_k"],
        "resident_contexts": configuration["resident_contexts"],
        "previous_timestep_parent_enabled":
            configuration["previous_timestep_parent_enabled"],
        "qualification": configuration["qualification"],
        "aggregate_source_only_cycles":
            configuration["aggregate_source_only_cycles"],
        "aggregate_integrated_cycles":
            configuration["aggregate_integrated_cycles"],
        "source_only_cycle_distribution":
            configuration["source_only_cycle_distribution"],
        "integrated_cycle_distribution":
            configuration["integrated_cycle_distribution"],
        "parent_selection_aggregate":
            configuration["parent_selection_aggregate"],
        "capacity": capacity,
    }


def build():
    contract = validate_contract()
    with ProcessPoolExecutor(max_workers=len(CONFIGURATIONS)) as executor:
        configurations = list(executor.map(
            replay_one_configuration, CONFIGURATIONS))
    by_name = dict((row["name"], row) for row in configurations)
    k2_temporal = by_name["K2_CTX16_TEMPORAL"]
    k4_spatial = by_name["K4_CTX16_SPATIAL"]
    k4_temporal = by_name["K4_CTX16_TEMPORAL"]

    m52 = read_json(HW_ROOT / contract["inputs"]["m52_result"]["path"])
    m47 = read_json(HW_ROOT / contract["inputs"]["m47_dual_frame_result"]["path"])
    m43_spatial = read_json(HW_ROOT / contract["inputs"]["m43_spatial_result"]["path"])
    m43_temporal = read_json(HW_ROOT / contract["inputs"]["m43_temporal_ablation"]["path"])
    reproduction = exact_m52_spatial_reproduction(m52, k4_spatial)
    require(reproduction["exact_match"], "M53 does not reproduce M52 K4 spatial")

    require(k4_spatial["parent_selection_aggregate"][
                "unfused_parent_delta_source_issue_cycles"] ==
            m43_spatial["aggregate"]["parent_delta_p8_l96_source_issue_cycles"] ==
            116376872, "M53 spatial parent source reference drift")
    require(k4_temporal["parent_selection_aggregate"][
                "unfused_parent_delta_source_issue_cycles"] ==
            m43_temporal["aggregate"]["parent_delta_p8_l96_source_issue_cycles"] ==
            113347744, "M53 temporal parent source reference drift")

    m52_k4_capacity = [row["capacity"] for row in m52["configuration_summaries"]
                        if row["name"] == "K4_CTX16"]
    m52_k2_capacity = [row["capacity"] for row in m52["configuration_summaries"]
                        if row["name"] == "K2_CTX16"]
    require(len(m52_k4_capacity) == 1 and len(m52_k2_capacity) == 1,
            "M53 M52 capacity configuration drift")
    require(m52_k4_capacity[0]["local_capacity_headroom_bytes"] == 17040 and
            m52_k4_capacity[0]["combined_local_capacity_bytes"] == 176688,
            "M53 M52 K4 capacity byte drift")
    capacity = {
        "K2_CTX16_TEMPORAL": dict(m52_k2_capacity[0]),
        "K4_CTX16_SPATIAL": dict(m52_k4_capacity[0]),
        "K4_CTX16_TEMPORAL": dict(m52_k4_capacity[0]),
    }
    for name in capacity:
        capacity[name]["headroom_unit"] = "bytes"
        capacity[name]["margin_above_minimum_headroom_bytes"] = (
            capacity[name]["local_capacity_headroom_bytes"] -
            capacity[name]["minimum_headroom_bytes"])
    capacity["K4_CTX16_TEMPORAL"]["temporal_parent_third_frame_bytes"] = 0

    storage_contract = contract["temporal_parent_storage_proof"]
    require(m47["capacity"]["bit_tight_frame_bytes"] ==
            storage_contract["bit_tight_frame_bytes"] == 68400 and
            m47["capacity"]["components"]["two_bit_tight_frames_bytes"] ==
            storage_contract["existing_two_frame_bytes"] == 136800,
            "M53 M47 dual-frame identity drift")
    two_frame = {
        "source": "M47 exact bit-tight capacity ledger",
        "frame_bytes": 68400,
        "existing_frame_count": 2,
        "existing_two_frame_bytes": 136800,
        "current_timestep_mapping": "frame[t mod 2]",
        "previous_timestep_mapping": "frame[(t-1) mod 2]",
        "new_third_frame_bytes": 0,
        "combined_k4_ctx16_capacity_bytes": 176688,
        "local_capacity_headroom_bytes": 17040,
        "headroom_unit": "bytes",
        "margin_above_16kib_gate_bytes": 656,
        "storage_gate_pass": True,
        "rtl_fifo_feasibility_admitted": False,
    }

    canonical_source = M45_ANALYZER.read_text(encoding="utf-8")
    order_snippets = (
        "for timestep in range(T):\n        for tile in range(TILES):",
        "block_time = tile_start + scheduled[\"integrated_cycles\"]",
        "while committed < ROWS_PER_T:",
        "require(counts[\"descriptor_commands\"] == ROWS_PER_T and",
    )
    snippet_counts = dict((snippet, canonical_source.count(snippet))
                          for snippet in order_snippets)
    require(all(count == 1 for count in snippet_counts.values()),
            "M53 canonical timestep-then-tile proof source drift")
    order_proof = {
        "canonical_m45_sha256": sha256_path(M45_ANALYZER),
        "loop_order": ["timestep", "feature_tile"],
        "timesteps_per_record": 10,
        "feature_tiles_per_timestep": 27,
        "rows_committed_per_tile": 300,
        "modeled_previous_to_current_boundaries_per_record": 9,
        "modeled_previous_to_current_boundaries_all_40_records": 360,
        "output_block_expanded_boundaries": 2880,
        "source_snippet_occurrences": snippet_counts,
        "previous_timestep_complete_before_current_starts": True,
        "proof": ("schedule_tile_timestep returns only after all 300 rows for "
                  "one tile commit; block_time advances after every tile; the "
                  "nested timestep-then-tile loop therefore completes all 27 "
                  "tiles of t-1 before entering t"),
        "qualification": ("canonical transaction-order proof; M54 must prove "
                          "finite RTL tags, arithmetic-state mapping, ports and FIFO events"),
    }

    spatial_choices = k4_spatial["parent_selection_aggregate"][
        "parent_choice_by_tile"]
    temporal_choices = k4_temporal["parent_selection_aggregate"][
        "parent_choice_by_tile"]
    displaced = dict((name, spatial_choices[name] - temporal_choices[name])
                     for name in ("local_zero", "left", "up"))
    require(sum(displaced.values()) == temporal_choices["previous_timestep"] and
            spatial_choices["previous_timestep"] == 0,
            "M53 adaptive parent-choice displacement conservation drift")
    selection = {
        "spatial_parent_choice_by_tile": spatial_choices,
        "adaptive_temporal_parent_choice_by_tile": temporal_choices,
        "spatial_choices_displaced_by_previous_timestep": displaced,
        "displaced_choice_count": sum(displaced.values()),
        "previous_timestep_choice_count":
            temporal_choices["previous_timestep"],
        "choice_count_conserved": sum(displaced.values()) ==
            temporal_choices["previous_timestep"],
        "previous_timestep_at_timestep_zero":
            k4_temporal["parent_selection_aggregate"][
                "previous_timestep_choices_at_timestep_zero"],
        "unfused_source_issue_cycle_gain": (
            k4_spatial["parent_selection_aggregate"][
                "unfused_parent_delta_source_issue_cycles"] -
            k4_temporal["parent_selection_aggregate"][
                "unfused_parent_delta_source_issue_cycles"]),
    }

    temporal_gain = cycle_gain_decomposition(k4_spatial, k4_temporal)
    fanout_gain = cycle_gain_decomposition(k2_temporal, k4_temporal)
    gates_contract = contract["predeclared_gates"]
    source_gain = temporal_gain["aggregate"]["source_only_cycle_gain"]
    integrated_gain = temporal_gain["aggregate"]["integrated_cycle_gain"]
    spatial_p95 = k4_spatial["integrated_cycle_distribution"]["p95_nearest_rank"]
    temporal_p95 = k4_temporal["integrated_cycle_distribution"]["p95_nearest_rank"]
    k2_temporal_p95 = k2_temporal["integrated_cycle_distribution"][
        "p95_nearest_rank"]
    p95_gain = spatial_p95 - temporal_p95
    k4_over_k2_p95_gain = k2_temporal_p95 - temporal_p95

    def threshold_pass(gain, reference, threshold):
        return (gain * threshold["denominator"] >=
                reference * threshold["numerator"])

    per_sample_source_nonregression = all(
        after["source_only_cycles"] <= before["source_only_cycles"]
        for before, after in zip(k4_spatial["per_sample"],
                                 k4_temporal["per_sample"]))
    source_gate = threshold_pass(
        source_gain, k4_spatial["aggregate_source_only_cycles"],
        gates_contract[
            "k4_temporal_minimum_aggregate_source_improvement_vs_k4_spatial_fraction"])
    integrated_gate = threshold_pass(
        integrated_gain, k4_spatial["aggregate_integrated_cycles"],
        gates_contract[
            "k4_temporal_minimum_aggregate_integrated_improvement_vs_k4_spatial_fraction"])
    p95_gate = threshold_pass(
        p95_gain, spatial_p95,
        gates_contract[
            "k4_temporal_minimum_p95_integrated_improvement_vs_k4_spatial_fraction"])
    k4_over_k2_gate = threshold_pass(
        k4_over_k2_p95_gain, k2_temporal_p95,
        gates_contract[
            "k4_temporal_minimum_p95_integrated_improvement_vs_k2_temporal_fraction"])
    overhead_threshold = gates_contract[
        "maximum_integrated_over_source_fraction_each_sample"]
    overhead_gate = all(
        (row["integrated_cycles"] - row["source_only_cycles"]) *
        overhead_threshold["denominator"] <=
        row["source_only_cycles"] * overhead_threshold["numerator"]
        for row in k4_temporal["per_sample"])
    prior_exercised = selection["previous_timestep_choice_count"] > 0
    capacity_gate = (capacity["K4_CTX16_TEMPORAL"][
        "local_capacity_headroom_bytes"] >=
        gates_contract["minimum_capacity_headroom_bytes"])
    all_gates = all((per_sample_source_nonregression, source_gate,
                     integrated_gate, p95_gate, k4_over_k2_gate,
                     overhead_gate, prior_exercised, capacity_gate,
                     reproduction["exact_match"]))

    m52_contract = read_json(HW_ROOT / contract["inputs"]["m52_contract"]["path"])
    pair = m52_contract["conservative_pair_model"]
    pair_samples = []
    for sample in k4_temporal["per_sample"]:
        pair_samples.append({
            "sample_id": sample["sample_id"],
            "transaction_integrated_cycles": sample["integrated_cycles"],
            "serialized_weight_load_cycles_added":
                pair["serialized_single_buffer_weight_load_cycles_per_sample"],
            "conservative_pair_upper_bound_cycles":
                sample["integrated_cycles"] +
                pair["serialized_single_buffer_weight_load_cycles_per_sample"],
        })
    pair_values = [row["conservative_pair_upper_bound_cycles"]
                   for row in pair_samples]
    pair_p95 = nearest_rank(pair_values, 0.95)
    conditional_total = (pair["outside_four_bottleneck_model_cycles"] +
                         pair["fixed_late_scale_plus_frontend_cycles"] +
                         pair_p95)
    conditional = {
        "construction": pair["construction"],
        "per_sample": pair_samples,
        "aggregate_pair_upper_bound_cycles": sum(pair_values),
        "pair_p95_nearest_rank_cycles": pair_p95,
        "fixed_compute_reference_cycles": pair["fixed_compute_reference_cycles"],
        "conditional_total_cycles": conditional_total,
        "conditional_compute_ratio": fraction(
            pair["fixed_compute_reference_cycles"], conditional_total),
        "three_x_crossing_in_conditional_model":
            pair["fixed_compute_reference_cycles"] >= 3 * conditional_total,
        "address_timed_pair_replayed": False,
        "system_or_end_to_end_speedup_admitted": False,
        "qualification": pair["qualification"],
    }

    gates = {
        "m52_k4_ctx16_spatial_exact_reproduction": reproduction["exact_match"],
        "temporal_parent_source_cycles_nonincreasing_each_sample":
            per_sample_source_nonregression,
        "k4_temporal_aggregate_source_improvement_vs_k4_spatial_ge_1pct":
            source_gate,
        "k4_temporal_aggregate_integrated_improvement_vs_k4_spatial_ge_2pct":
            integrated_gate,
        "k4_temporal_p95_integrated_improvement_vs_k4_spatial_ge_2pct":
            p95_gate,
        "k4_temporal_p95_integrated_improvement_vs_k2_temporal_ge_10pct":
            k4_over_k2_gate,
        "k4_temporal_each_sample_integrated_over_source_le_20pct":
            overhead_gate,
        "previous_timestep_parent_exercised": prior_exercised,
        "previous_timestep_parent_never_used_at_timestep_zero":
            selection["previous_timestep_at_timestep_zero"] == 0,
        "previous_timestep_all_tiles_committed_before_use": True,
        "m47_two_frames_reused_without_third_frame":
            two_frame["new_third_frame_bytes"] == 0,
        "k4_ctx16_capacity_headroom_ge_16kib": capacity_gate,
        "m52_headroom_is_17040_bytes":
            capacity["K4_CTX16_TEMPORAL"]["local_capacity_headroom_bytes"] ==
            17040,
        "all_predeclared_transaction_and_capacity_gates_pass": all_gates,
    }

    return {
        "schema": "m53_adaptive_temporal_parent_k4_ctx16_dse_result_v1",
        "status": (
            "PASS_M53_K4_CTX16_TEMPORAL_TRANSACTION_DSE_M54_RTL_REQUIRED"
            if all_gates else
            "NO_GO_M53_ONE_OR_MORE_PREDECLARED_DSE_GATES_FAILED"),
        "identity": {
            "contract_sha256": sha256_path(CONTRACT),
            "analyzer_sha256": sha256_path(Path(__file__).resolve()),
            "inputs_sha256": dict((name, item["sha256"])
                                   for name, item in contract["inputs"].items()),
        },
        "population": {"samples": 10, "operators": 4, "records": 40},
        "m52_spatial_reproduction": reproduction,
        "configuration_summaries": [
            summary(k2_temporal, capacity["K2_CTX16_TEMPORAL"]),
            summary(k4_spatial, capacity["K4_CTX16_SPATIAL"]),
            summary(k4_temporal, capacity["K4_CTX16_TEMPORAL"]),
        ],
        "configuration_ledgers": configurations,
        "adaptive_parent_selection_decomposition": selection,
        "cycle_gain_decomposition": {
            "k4_temporal_vs_k4_spatial": temporal_gain,
            "k4_temporal_vs_k2_temporal": fanout_gain,
        },
        "performance_comparisons": {
            "k4_temporal_aggregate_source_improvement_vs_k4_spatial":
                fraction(source_gain,
                         k4_spatial["aggregate_source_only_cycles"]),
            "k4_temporal_aggregate_integrated_improvement_vs_k4_spatial":
                fraction(integrated_gain,
                         k4_spatial["aggregate_integrated_cycles"]),
            "k4_temporal_p95_integrated_improvement_vs_k4_spatial":
                fraction(p95_gain, spatial_p95),
            "k4_temporal_p95_integrated_improvement_vs_k2_temporal":
                fraction(k4_over_k2_p95_gain, k2_temporal_p95),
            "k4_temporal_transaction_speedup_vs_m52_k4_spatial_integrated":
                fraction(k4_spatial["aggregate_integrated_cycles"],
                         k4_temporal["aggregate_integrated_cycles"]),
            "k4_temporal_source_speedup_vs_m43_local_zero_source":
                fraction(m43_spatial["aggregate"][
                             "local_p8_l96_source_issue_cycles"],
                         k4_temporal["aggregate_source_only_cycles"]),
        },
        "timestep_then_tile_commit_proof": order_proof,
        "two_frame_capacity_ledger": two_frame,
        "predeclared_gates": gates,
        "conditional_frozen_compute_model": conditional,
        "admission": {
            "exact_all10_transaction_dse_admitted": all_gates,
            "adaptive_parent_selection_ledger_admitted": all_gates,
            "two_frame_zero_third_frame_byte_ledger_admitted": all_gates,
            "canonical_transaction_order_proof_admitted": all_gates,
            "k4_temporal_promoted_to_m54_rtl_experiment_only": all_gates,
            "adaptive_per_tile_temporal_parent_arithmetic_state_rtl_admitted": False,
            "finite_context_tag_allocation_admitted": False,
            "response_metadata_fifo_event_ledger_admitted": False,
            "new_configuration_vcs_or_synopsys_admitted": False,
            "sram_macro_port_feasibility_admitted": False,
            "full_network_or_system_speedup_admitted": False,
            "date_headline_or_best_paper_admitted": False,
        },
        "auditable_defects_and_boundaries": [
            "M52 headroom is 17040 bytes, but its margin above the 16-KiB gate is only 656 bytes",
            "maximum_metadata_occupancy remains a clamped ready-window proxy, not a response-FIFO enqueue/dequeue event ledger",
            "K4 response RMW and completion widths remain unsynthesized structural assumptions until M54",
            "adaptive per-tile temporal-parent arithmetic state and stale-tag behavior are not RTL-proved",
            "the conditional compute ratio is not an address-timed full-network or system speedup"
        ],
        "claim_policy": contract["claim_policy"],
    }


def write_output(path, payload):
    path = Path(path)
    require(not path.exists(), "refusing to overwrite M53 output")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_output(args.output, build())
    print(args.output)


if __name__ == "__main__":
    main()
