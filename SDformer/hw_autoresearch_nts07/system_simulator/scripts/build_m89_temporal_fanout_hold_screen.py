#!/usr/bin/env python3
"""Build a fail-closed, non-citable receipt for the M89 screening DSE."""

from __future__ import print_function

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
M53_RESULT = HW_ROOT / (
    "results/m53_adaptive_temporal_parent_k4_ctx16_dse_r1_20260823/"
    "m53_adaptive_temporal_parent_k4_ctx16_dse.json")
M43_RESULT = HW_ROOT / (
    "results/m43_tile_resident_parent_delta_schedule_r1_20260823/"
    "m43_spatial_parent_delta_schedule_final.json")
M53_ANALYZER = HW_ROOT / (
    "system_simulator/scripts/"
    "analyze_m53_adaptive_temporal_parent_k4_ctx16_dse.py")
PROBE = HW_ROOT / (
    "system_simulator/scripts/probe_m89_k4_nohold_temporal.py")

EXPECTED = {
    str(M53_RESULT.relative_to(HW_ROOT)):
        "344ae1f777e0640d46b19118f0b6d451465046350d68a9f33b1faae124747bb4",
    str(M43_RESULT.relative_to(HW_ROOT)):
        "70c52dfc8ef1b223391a1c0699f6ada8ff999d2079370bcd9d3917c198a1c329",
    str(M53_ANALYZER.relative_to(HW_ROOT)):
        "638809bd72ab7f66fc69b51f4cb726f2c0d1c7712f71188066b4ef04cbdda531",
    str(PROBE.relative_to(HW_ROOT)):
        "9c443d62cef5ce4f6ba5509ebda04527b2ad200211a1e81f45e68f68598fab20",
}

LOGS = {
    "K4_NOHOLD": "m89_k4_nohold_r2_temporal_probe_20260823.log",
    "K4_ONLY_ONE": "m89_k4_only_one_temporal_probe_20260823.log",
    "K4_ONLY_TWO": "m89_k4_only_two_temporal_probe_20260823.log",
    "K4_ONLY_THREE": "m89_k4_only_three_temporal_probe_20260823.log",
    "K4_UP_TO_TWO": "m89_k4_up_to_two_temporal_probe_20260823.log",
    "K4_TWO_OR_THREE": "m89_k4_two_or_three_temporal_probe_20260823.log",
    "K5": "m89_k5_ctx16_temporal_probe_20260823.log",
    "K6": "m89_k6_ctx16_temporal_probe_20260823.log",
    "K7": "m89_k7_ctx16_temporal_probe_20260823.log",
    "K8": "m89_k8_ctx16_temporal_probe_20260823.log",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: {}".format(raw))
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=reject)


def fraction(numerator, denominator):
    require(denominator > 0, "zero denominator")
    divisor = math.gcd(int(numerator), int(denominator))
    return {
        "numerator": int(numerator) // divisor,
        "denominator": int(denominator) // divisor,
        "decimal": float(numerator) / float(denominator),
    }


def parse_log(path):
    text = Path(path).read_text(encoding="utf-8")
    require(text.count("/40 sample=") == 40,
            "M89 log does not contain exactly 40 completed records: {}".format(path))
    markers = [line for line in text.splitlines()
               if line.startswith("M89_") and "_PROBE=" in line]
    require(len(markers) == 1,
            "M89 log final marker population drift: {}".format(path))
    payload = json.loads(markers[0].split("=", 1)[1])
    require(payload["integrated"] >= payload["source"] and
            payload.get("max_complete", 0) == 16 and
            payload.get("max_metadata", payload.get("max_meta", 0)) == 16,
            "M89 log conservation/occupancy drift: {}".format(path))
    if "per_sample" in payload:
        require(len(payload["per_sample"]) == 10 and
                sum(row["source"] for row in payload["per_sample"]) ==
                payload["source"] and
                sum(row["integrated"] for row in payload["per_sample"]) ==
                payload["integrated"],
                "M89 per-sample aggregation drift: {}".format(path))
    return payload


def capacity(fanout):
    payload_bits = 21 * fanout + 8
    entry_bytes = ((payload_bits + 63) // 64) * 8
    fixed = 176432
    combined = fixed + 16 * entry_bytes
    return {
        "response_metadata_payload_bits": payload_bits,
        "response_metadata_aligned_bytes_per_entry": entry_bytes,
        "combined_local_capacity_bytes": combined,
        "local_capacity_headroom_bytes": 193728 - combined,
        "minimum_headroom_bytes": 16384,
        "headroom_gate_pass": 193728 - combined >= 16384,
    }


def candidate_row(name, fanout, payload, local_source, k4_integrated):
    source = payload["source"]
    integrated = payload["integrated"]
    overhead = integrated - source
    required_numerator = max(0, 2 * source + overhead - local_source)
    required_reduction = (required_numerator + 1) // 2
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
        "structural_width": {
            "accumulator_paths": fanout * 96,
            "signed_bank_terms": fanout * 8 * 96,
            "atomic_complete_payload_bits_excluding_tags": fanout * 96 * 19,
            "relative_to_k4": fraction(fanout, 4),
            "qualification": "structural width only; not synthesized area",
        },
        "performance_screen": {
            "vs_m53_k4_legacy_integrated": fraction(k4_integrated, integrated),
            "local_source_only_over_candidate_integrated":
                fraction(local_source, integrated),
            "equal_candidate_overhead_composition":
                fraction(local_source + overhead, integrated),
            "source_cycle_reduction_required_for_2x_equal_overhead_composition":
                required_reduction,
            "required_reduction_fraction_of_candidate_source":
                fraction(required_reduction, source),
        },
    }


def build(log_dir):
    for relative, expected in EXPECTED.items():
        path = HW_ROOT / relative
        require(path.is_file() and sha256(path) == expected,
                "M89 frozen input drift: {}".format(relative))
    m53 = read_json(M53_RESULT)
    k4_matches = [row for row in m53["configuration_ledgers"]
                  if row["name"] == "K4_CTX16_TEMPORAL"]
    require(len(k4_matches) == 1, "M89 M53 K4 baseline population drift")
    k4 = k4_matches[0]
    local_source = read_json(M43_RESULT)["aggregate"][
        "local_p8_l96_source_issue_cycles"]
    require(local_source == 141484880 and
            k4["aggregate_source_only_cycles"] == 68847096 and
            k4["aggregate_integrated_cycles"] == 79869808,
            "M89 baseline numeric drift")

    payloads = {}
    log_manifest = {}
    for name, filename in LOGS.items():
        path = Path(log_dir) / filename
        require(path.is_file(), "M89 missing log: {}".format(filename))
        payloads[name] = parse_log(path)
        log_manifest[name] = {
            "file": "remote_logs/" + filename,
            "sha256": sha256(path),
            "record_completion_markers": 40,
        }

    configurations = [{
        "name": "K4_CTX16_TEMPORAL_LEGACY",
        "fanout": 4,
        "source_cycles": k4["aggregate_source_only_cycles"],
        "integrated_cycles": k4["aggregate_integrated_cycles"],
        "integrated_over_source_cycles":
            k4["aggregate_integrated_cycles"] -
            k4["aggregate_source_only_cycles"],
        "fusion_hold_cycles": sum(row["fusion_hold_wait_cycles"]
                                  for row in k4["per_sample"]),
        "source": "frozen M53 exact all10 result",
    }]
    for name in ("K4_NOHOLD", "K4_ONLY_ONE", "K4_ONLY_TWO",
                 "K4_ONLY_THREE", "K4_UP_TO_TWO", "K4_TWO_OR_THREE",
                 "K5", "K6", "K7", "K8"):
        fanout = 4 if name.startswith("K4_") else int(name[1:])
        configurations.append(candidate_row(
            name, fanout, payloads[name], local_source,
            k4["aggregate_integrated_cycles"]))

    ranked = sorted(configurations[1:], key=lambda row: row["integrated_cycles"])
    by_name = dict((row["name"], row) for row in configurations)
    require(by_name["K4_NOHOLD"]["integrated_cycles"] == 78803200 and
            by_name["K6"]["integrated_cycles"] == 76677320 and
            by_name["K8"]["integrated_cycles"] == 76337352,
            "M89 expected screening point drift")
    return {
        "schema": "m89_temporal_fanout_hold_screen_receipt_v1",
        "status": "PASS_NON_CITABLE_EXACT_ALL10_SCREENING_DSE",
        "scope": "frozen H67 ep35 four expensive Conv3x3 operators across ten valid825-internal windows",
        "identity": {
            "frozen_inputs": EXPECTED,
            "remote_log_manifest": log_manifest,
        },
        "baseline": {
            "local_zero_p8_l96_source_cycles": local_source,
            "m53_k4_temporal_source_cycles": k4["aggregate_source_only_cycles"],
            "m53_k4_temporal_integrated_cycles": k4["aggregate_integrated_cycles"],
        },
        "configurations": configurations,
        "ranking_by_integrated_cycles": [row["name"] for row in ranked],
        "decisions": {
            "k4_nohold_beats_all_selective_hold_policies": all(
                by_name["K4_NOHOLD"]["integrated_cycles"] <
                by_name[name]["integrated_cycles"]
                for name in ("K4_ONLY_ONE", "K4_ONLY_TWO", "K4_ONLY_THREE",
                             "K4_UP_TO_TWO", "K4_TWO_OR_THREE")),
            "k6_is_within_half_percent_of_k8":
                (by_name["K6"]["integrated_cycles"] -
                 by_name["K8"]["integrated_cycles"]) * 200 <=
                by_name["K8"]["integrated_cycles"],
            "k6_has_25_percent_less_structural_width_than_k8": True,
            "performance_candidate": "K6",
            "minimum_width_candidate": "K4_NOHOLD",
            "k7_k8_rtl_promotion": "KILL_DIMINISHING_RETURN",
        },
        "claim_policy": {
            "admitted": [
                "exact deterministic screening totals and K5-K8 per-sample totals in the copied logs",
                "structural K-scaled width and bit-tight response-metadata capacity ledgers",
                "derived source-only floor and equal-overhead composition ratios labeled as compositions",
            ],
            "forbidden": [
                "RTL cycle speedup for K5-K8 or selective hold policies",
                "equal-area, synthesized PPA, SRAM macro, energy, accuracy or full-system claim",
                "calling a composition an executed baseline",
                "DATE headline, best-paper or external accelerator superiority claim",
            ],
            "paper_ppa_ready": False,
            "rtl_cycle_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build(args.log_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M89 configurations={} winner={} performance_candidate={}".format(
        len(result["configurations"]),
        result["ranking_by_integrated_cycles"][0],
        result["decisions"]["performance_candidate"]))


if __name__ == "__main__":
    main()
