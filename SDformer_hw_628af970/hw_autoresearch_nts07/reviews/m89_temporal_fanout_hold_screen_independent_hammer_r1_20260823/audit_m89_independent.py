#!/usr/bin/env python3
"""Independent arithmetic/provenance audit of the M89 screening receipt.

The audit deliberately does not import or execute either M89 producer script.
It parses the ten selected raw remote logs, reads the two frozen baseline
receipts, and reconstructs policy ranking, fanout ranking, capacity, structural
width, comparison ratios, and the PAFT reduction needed by the stated
equal-overhead composition.
"""

from __future__ import print_function

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RESULT_DIR = HW / "results/m89_temporal_fanout_hold_screen_r1_20260823"
RECEIPT = RESULT_DIR / "m89_temporal_fanout_hold_screen_receipt.json"
BUILDER = HW / "system_simulator/scripts/build_m89_temporal_fanout_hold_screen.py"
PROBE = HW / "system_simulator/scripts/probe_m89_k4_nohold_temporal.py"
M53 = HW / (
    "results/m53_adaptive_temporal_parent_k4_ctx16_dse_r1_20260823/"
    "m53_adaptive_temporal_parent_k4_ctx16_dse.json")
M43 = HW / (
    "results/m43_tile_resident_parent_delta_schedule_r1_20260823/"
    "m43_spatial_parent_delta_schedule_final.json")
OUTPUT = HERE / "m89_independent_recompute.json"

EXPECTED = {
    "receipt": "afacec344ec8481dd27b667751e97d938655f46e5cced7b330460a530b92e9cf",
    "builder": "127fe4bc6591c8a10f6cfd8fb7a9bdafdc9cf0ed65d4e7db0dd9911d62942fa1",
    "probe": "9c443d62cef5ce4f6ba5509ebda04527b2ad200211a1e81f45e68f68598fab20",
    "m53": "344ae1f777e0640d46b19118f0b6d451465046350d68a9f33b1faae124747bb4",
    "m43": "70c52dfc8ef1b223391a1c0699f6ada8ff999d2079370bcd9d3917c198a1c329",
}

LOGS = {
    "K4_NOHOLD": (
        "m89_k4_nohold_r2_temporal_probe_20260823.log",
        "2589ef112c47fc9b9935b9c204936a90aaf7ace6c56a1f51f875353ac9a3f1fd"),
    "K4_ONLY_ONE": (
        "m89_k4_only_one_temporal_probe_20260823.log",
        "37f324b873e8cbc12c6fee9dc9beef1e222dc10984a6c09d540d086dfa3b410a"),
    "K4_ONLY_TWO": (
        "m89_k4_only_two_temporal_probe_20260823.log",
        "d2ae23a94a8c3862dc2470027aad8deab44bbe3ed47cd548c022072123a5908e"),
    "K4_ONLY_THREE": (
        "m89_k4_only_three_temporal_probe_20260823.log",
        "13bf6b32de990be3c6e067bfcfd96c94bb0b59388497cec7c72e8166a7db0a0a"),
    "K4_UP_TO_TWO": (
        "m89_k4_up_to_two_temporal_probe_20260823.log",
        "d532cebddd390ba356b31ce4f58dd65f5020c28004e2715a7257281e86a761cd"),
    "K4_TWO_OR_THREE": (
        "m89_k4_two_or_three_temporal_probe_20260823.log",
        "33b5fab4ef3ddc2717618fef7a492f28b77160ae94ff6651fd8d0bc38698549c"),
    "K5": (
        "m89_k5_ctx16_temporal_probe_20260823.log",
        "df216564b3c0c822ba64bdf7fa6fbdccfb778c51e3a86b9aff9862b186135de4"),
    "K6": (
        "m89_k6_ctx16_temporal_probe_20260823.log",
        "8f1c4751259526be7ff94e3859e969a881f90ded87c83043f445a8d29dca5955"),
    "K7": (
        "m89_k7_ctx16_temporal_probe_20260823.log",
        "06a68c6b8af62106e65dd65d82d2a23e397d663f58710ffa34cc09dcea70f353"),
    "K8": (
        "m89_k8_ctx16_temporal_probe_20260823.log",
        "02a1674243a407f4d9745007782d24ec99ab2972af7e3bb9e2aa5e4f1d7268a3"),
}

PROGRESS = re.compile(
    r"^\[M53 [^]]+\] ([0-9]+)/40 sample=([0-9]+) operator=(\S+)$")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json_text(text, label):
    def pairs(pairs_list):
        result = {}
        for key, value in pairs_list:
            require(key not in result, "duplicate key {} in {}".format(key, label))
            result[key] = value
        return result

    return json.loads(
        text, object_pairs_hook=pairs,
        parse_constant=lambda raw: (_ for _ in ()).throw(
            ValueError("non-standard JSON constant {} in {}".format(raw, label))))


def read_json(path):
    return strict_json_text(Path(path).read_text(encoding="utf-8"), str(path))


def compare(left, right, label="root"):
    if isinstance(left, dict) and isinstance(right, dict):
        require(set(left) == set(right), label + " key drift")
        for key in left:
            compare(left[key], right[key], label + "." + str(key))
    elif isinstance(left, list) and isinstance(right, list):
        require(len(left) == len(right), label + " length drift")
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, label + "[{}]".format(index))
    elif isinstance(left, float) or isinstance(right, float):
        require(abs(float(left) - float(right)) <=
                1e-12 * max(1.0, abs(float(right))), label + " float drift")
    else:
        require(left == right, "{} drift: {} != {}".format(label, left, right))


def exact_ratio(numerator, denominator):
    require(denominator > 0, "ratio denominator must be positive")
    value = Fraction(int(numerator), int(denominator))
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


def parse_log(path, name):
    text = Path(path).read_text(encoding="utf-8")
    progress = []
    for line in text.splitlines():
        match = PROGRESS.match(line)
        if match:
            progress.append((int(match.group(1)), int(match.group(2)),
                             match.group(3)))
    require([row[0] for row in progress] == list(range(1, 41)),
            name + " progress ordinals are not exactly 1..40")
    identities = [(row[1], row[2]) for row in progress]
    require(len(set(identities)) == 40, name + " duplicate record identity")
    require(set(row[0] for row in identities) == set(range(10)),
            name + " sample population drift")
    require(all(sum(1 for sample, _ in identities if sample == expected) == 4
                for expected in range(10)), name + " operator population drift")
    markers = [line for line in text.splitlines()
               if line.startswith("M89_") and "_PROBE=" in line]
    require(len(markers) == 1, name + " final marker population drift")
    payload = strict_json_text(markers[0].split("=", 1)[1], name + " marker")
    for field in ("source", "integrated", "p95", "max_complete"):
        require(field in payload, name + " missing " + field)
    max_meta = payload.get("max_metadata", payload.get("max_meta"))
    require(payload["integrated"] >= payload["source"] and
            payload["max_complete"] == 16 and max_meta == 16,
            name + " conservation/occupancy drift")
    if name.startswith("K4_"):
        require("per_sample" not in payload,
                name + " unexpectedly changed K4 evidence granularity")
    else:
        per_sample = payload.get("per_sample")
        require(isinstance(per_sample, list) and len(per_sample) == 10,
                name + " per-sample population drift")
        require([row["sample_id"] for row in per_sample] == list(range(10)),
                name + " per-sample order drift")
        require(sum(row["source"] for row in per_sample) == payload["source"] and
                sum(row["integrated"] for row in per_sample) ==
                payload["integrated"], name + " per-sample aggregation drift")
        require(max(row["integrated"] for row in per_sample) == payload["p95"],
                name + " ten-point nearest-rank p95 drift")
    return payload


def capacity(fanout):
    payload_bits = 21 * fanout + 8
    aligned_entry_bytes = ((payload_bits + 63) // 64) * 8
    fixed_bytes = 176432
    combined = fixed_bytes + 16 * aligned_entry_bytes
    available = 193728
    headroom = available - combined
    threshold = 16384
    return {
        "response_metadata_payload_bits": payload_bits,
        "response_metadata_aligned_bytes_per_entry": aligned_entry_bytes,
        "combined_local_capacity_bytes": combined,
        "local_capacity_headroom_bytes": headroom,
        "minimum_headroom_bytes": threshold,
        "headroom_gate_pass": headroom >= threshold,
    }


def width(fanout):
    return {
        "accumulator_paths": fanout * 96,
        "signed_bank_terms": fanout * 8 * 96,
        "atomic_complete_payload_bits_excluding_tags": fanout * 96 * 19,
        "relative_to_k4": exact_ratio(fanout, 4),
        "qualification": "structural width only; not synthesized area",
    }


def candidate(name, payload, fanout, local_source, legacy_integrated):
    source = payload["source"]
    integrated = payload["integrated"]
    overhead = integrated - source
    # Solve (local + overhead) / (source - reduction + overhead) >= 2.
    reduction = max(0, (2 * source + overhead - local_source + 1) // 2)
    return {
        "name": name,
        "fanout": fanout,
        "source_cycles": source,
        "integrated_cycles": integrated,
        "integrated_over_source_cycles": overhead,
        "p95_integrated_cycles": payload["p95"],
        "fusion_hold_cycles": payload.get("fusion_hold", 0),
        "maximum_complete_occupancy": payload["max_complete"],
        "maximum_metadata_occupancy":
            payload.get("max_metadata", payload.get("max_meta")),
        "per_sample": payload.get("per_sample"),
        "capacity": capacity(fanout),
        "structural_width": width(fanout),
        "performance_screen": {
            "vs_m53_k4_legacy_integrated":
                exact_ratio(legacy_integrated, integrated),
            "local_source_only_over_candidate_integrated":
                exact_ratio(local_source, integrated),
            "equal_candidate_overhead_composition":
                exact_ratio(local_source + overhead, integrated),
            "source_cycle_reduction_required_for_2x_equal_overhead_composition":
                reduction,
            "required_reduction_fraction_of_candidate_source":
                exact_ratio(reduction, source),
        },
    }


def main():
    paths = {
        "receipt": RECEIPT,
        "builder": BUILDER,
        "probe": PROBE,
        "m53": M53,
        "m43": M43,
    }
    for name, path in paths.items():
        require(path.is_file() and sha256(path) == EXPECTED[name],
                name + " SHA drift")
    stored = read_json(RECEIPT)
    m53 = read_json(M53)
    m43 = read_json(M43)
    k4_rows = [row for row in m53["configuration_ledgers"]
               if row["name"] == "K4_CTX16_TEMPORAL"]
    require(len(k4_rows) == 1, "M53 legacy K4 population drift")
    legacy = k4_rows[0]
    local_source = m43["aggregate"]["local_p8_l96_source_issue_cycles"]
    require(local_source == 141484880 and
            legacy["aggregate_source_only_cycles"] == 68847096 and
            legacy["aggregate_integrated_cycles"] == 79869808,
            "frozen baseline number drift")

    payloads = {}
    log_identity = {}
    for name, (filename, expected_hash) in LOGS.items():
        path = RESULT_DIR / "remote_logs" / filename
        require(path.is_file() and sha256(path) == expected_hash,
                name + " selected log SHA drift")
        payloads[name] = parse_log(path, name)
        log_identity[name] = {
            "path": str(path),
            "sha256": expected_hash,
            "bytes": path.stat().st_size,
        }
        compare(stored["identity"]["remote_log_manifest"][name]["sha256"],
                expected_hash, name + " receipt log SHA")

    names = ("K4_NOHOLD", "K4_ONLY_ONE", "K4_ONLY_TWO",
             "K4_ONLY_THREE", "K4_UP_TO_TWO", "K4_TWO_OR_THREE",
             "K5", "K6", "K7", "K8")
    rows = {}
    stored_rows = dict((row["name"], row)
                       for row in stored["configurations"])
    for name in names:
        fanout = 4 if name.startswith("K4_") else int(name[1:])
        rows[name] = candidate(
            name, payloads[name], fanout, local_source,
            legacy["aggregate_integrated_cycles"])
        compare(rows[name], stored_rows[name], "receipt." + name)

    ranking = sorted(names, key=lambda name: rows[name]["integrated_cycles"])
    compare(ranking, stored["ranking_by_integrated_cycles"], "global ranking")
    k4_ranking = sorted((name for name in names if name.startswith("K4_")),
                        key=lambda name: rows[name]["integrated_cycles"])
    fanout_ranking = sorted(("K5", "K6", "K7", "K8"),
                            key=lambda name: rows[name]["integrated_cycles"])
    k6 = rows["K6"]
    k8 = rows["K8"]
    integrated_delta = k6["integrated_cycles"] - k8["integrated_cycles"]
    require(integrated_delta == 339968, "K6-vs-K8 cycle delta drift")
    require(k6["structural_width"]["accumulator_paths"] * 4 ==
            k8["structural_width"]["accumulator_paths"] * 3,
            "K6-vs-K8 structural-width ratio drift")

    result = {
        "schema": "m89_temporal_fanout_hold_screen_independent_recompute_v1",
        "status": "PASS_EXACT_ARITHMETIC_AND_LOG_SCREEN_RECOMPUTE_ONLY",
        "independence": {
            "producer_builder_imported_or_executed": False,
            "producer_probe_imported_or_executed": False,
            "ten_selected_logs_parsed_directly": True,
            "production_files_modified": False,
            "simulator_event_decisions_replayed_from_source_traces": False,
        },
        "identity": {
            "inputs": dict((name, {
                "path": str(path), "sha256": EXPECTED[name],
                "bytes": path.stat().st_size}) for name, path in paths.items()),
            "selected_logs": log_identity,
        },
        "scope": stored["scope"],
        "baseline": {
            "local_zero_p8_l96_source_cycles": local_source,
            "m53_k4_temporal_source_cycles":
                legacy["aggregate_source_only_cycles"],
            "m53_k4_temporal_integrated_cycles":
                legacy["aggregate_integrated_cycles"],
        },
        "k4_policy_ranking": [
            {"name": name,
             "source_cycles": rows[name]["source_cycles"],
             "fusion_hold_cycles": rows[name]["fusion_hold_cycles"],
             "integrated_cycles": rows[name]["integrated_cycles"]}
            for name in k4_ranking],
        "k5_to_k8_ranking": [
            {"name": name,
             "source_cycles": rows[name]["source_cycles"],
             "integrated_over_source_cycles":
                 rows[name]["integrated_over_source_cycles"],
             "integrated_cycles": rows[name]["integrated_cycles"],
             "p95_integrated_cycles": rows[name]["p95_integrated_cycles"]}
            for name in fanout_ranking],
        "global_ranking_by_integrated_cycles": list(ranking),
        "capacity_and_width": dict((name, {
            "capacity": rows[name]["capacity"],
            "structural_width": rows[name]["structural_width"],
            "headroom_margin_beyond_gate_bytes":
                rows[name]["capacity"]["local_capacity_headroom_bytes"] -
                rows[name]["capacity"]["minimum_headroom_bytes"],
        }) for name in ("K4_NOHOLD", "K5", "K6", "K7", "K8")),
        "k6_vs_k8": {
            "k6_minus_k8_integrated_cycles": integrated_delta,
            "k8_cycle_advantage_fraction_of_k8":
                exact_ratio(integrated_delta, k8["integrated_cycles"]),
            "k8_speedup_over_k6":
                exact_ratio(k6["integrated_cycles"], k8["integrated_cycles"]),
            "k6_structural_width_fraction_of_k8": exact_ratio(3, 4),
            "k6_structural_width_reduction_fraction_vs_k8": exact_ratio(1, 4),
            "k6_minus_k8_source_cycles":
                k6["source_cycles"] - k8["source_cycles"],
            "k8_minus_k6_overhead_cycles":
                k8["integrated_over_source_cycles"] -
                k6["integrated_over_source_cycles"],
        },
        "comparison_compositions": dict((name, {
            "local_source_only_over_candidate_integrated":
                rows[name]["performance_screen"][
                    "local_source_only_over_candidate_integrated"],
            "equal_candidate_overhead_composition":
                rows[name]["performance_screen"][
                    "equal_candidate_overhead_composition"],
            "paft_source_cycle_reduction_required_for_2x_composition":
                rows[name]["performance_screen"][
                    "source_cycle_reduction_required_for_2x_equal_overhead_composition"],
            "paft_reduction_fraction_of_candidate_source":
                rows[name]["performance_screen"][
                    "required_reduction_fraction_of_candidate_source"],
        }) for name in ("K4_NOHOLD", "K5", "K6", "K7", "K8")),
        "evidence_limits": {
            "k4_policy_logs_have_per_sample_ledgers": False,
            "k5_to_k8_logs_have_per_sample_ledgers": True,
            "logs_embed_generator_sha_command_git_commit_or_environment": False,
            "ratios_are_executed_equal_area_baselines": False,
            "capacity_is_a_hardcoded_structural_ledger_not_a_macro_result": True,
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M89 independent K4_winner={} fanout_winner={} K6_K8_delta={}"
          .format(k4_ranking[0], fanout_ranking[0], integrated_delta))


if __name__ == "__main__":
    main()
