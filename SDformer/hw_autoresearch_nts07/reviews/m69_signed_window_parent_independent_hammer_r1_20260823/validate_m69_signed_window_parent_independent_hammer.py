#!/usr/bin/env python3
"""Independent audit and sample-0 raw replay for M69."""

from __future__ import print_function

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile


HW = Path(__file__).resolve().parents[2]
ANALYZER = HW / "system_simulator/scripts/analyze_m69_signed_window_parent_dse.py"
RESULT = HW / (
    "results/m69_signed_window_parent_source_dse_dev_r1_20260823/"
    "m69_signed_window_parent_source_dse.json")
MANIFEST = HW / (
    "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822/"
    "m40_bottleneck_packed_source_manifest.json")
M43 = HW / "system_simulator/scripts/analyze_m43_tile_resident_parent_delta_schedule.py"
M53 = HW / (
    "results/m53_adaptive_temporal_parent_k4_ctx16_dse_r1_20260823/"
    "m53_adaptive_temporal_parent_k4_ctx16_dse.json")
CPP = Path(__file__).with_name("recompute_m69_sample0.cpp")
WINDOWS = (1, 2, 4, 8, 16, 32, 64)

EXPECTED_SHA = {
    "analyzer": "01372065fab0d08106017d45e1a3854de3758af8590d3a22fa9e348850b0e9f1",
    "result": "ecc8afaaa5b01055ecb86d801ff94bbc4329a9f7cb5490bd03e84da221a96f7e",
    "manifest": "e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3",
    "m43": "a4ddebf4687b32c65735c591a6526f43b7274777ace4e3ca90d19a2d04adb1c3",
    "m53": "344ae1f777e0640d46b19118f0b6d451465046350d68a9f33b1faae124747bb4",
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


def pairs(pairs_list):
    result = {}
    for key, value in pairs_list:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError("non-standard JSON: " + value)))


def exact_identity():
    for name, path in (("analyzer", ANALYZER), ("result", RESULT),
                       ("manifest", MANIFEST), ("m43", M43), ("m53", M53)):
        require(path.is_file() and not path.is_symlink(), "missing identity " + name)
        require(sha256(path) == EXPECTED_SHA[name], "identity drift " + name)
    payload = load(RESULT)
    require(payload["identity"] == {
        "analyzer_sha256": EXPECTED_SHA["analyzer"],
        "inputs_sha256": {
            "m43": EXPECTED_SHA["m43"],
            "manifest": EXPECTED_SHA["manifest"],
            "m53": EXPECTED_SHA["m53"],
        },
    }, "result identity ledger drift")
    return payload


def aggregate_replay(payload):
    records = payload["records"]
    require(len(records) == 40, "record population drift")
    by_key = {}
    for record in records:
        key = (record["sample_id"], record["operator"])
        require(key not in by_key, "duplicate record identity")
        by_key[key] = record
        for window in WINDOWS:
            item = record["windows"][str(window)]
            require(item["signed_add_updates"] + item["signed_subtract_updates"] ==
                    item["logical_updates"], "record signed conservation drift")
            require(item["zero_parent_tiles"] + item["nonzero_parent_tiles"] ==
                    10 * 15 * 20 * 27, "record tile population drift")
            require(sum(item["origin"].values()) == 10 * 15 * 20 * 27,
                    "record origin population drift")
            require(item["source_or_matcher_lower_bound_cycles"] >=
                    item["source_issue_cycles"], "lower bound below source")
    require(len(by_key) == 40 and
            all(sum(1 for key in by_key if key[0] == sample) == 4
                for sample in range(10)), "cohort identity drift")

    fields = ("source_issue_cycles", "logical_updates", "signed_add_updates",
              "signed_subtract_updates", "exact_copy_tiles",
              "zero_parent_tiles", "nonzero_parent_tiles",
              "matcher_query_rows", "matcher_nominal_cycles",
              "source_or_matcher_lower_bound_cycles")
    derived_samples = []
    for sample in range(10):
        selected = [row for row in records if row["sample_id"] == sample]
        derived = {
            "sample_id": sample,
            "local_zero_source_issue_cycles": sum(
                row["local_zero_source_issue_cycles"] for row in selected),
            "canonical_m53_parent_source_issue_cycles": sum(
                row["canonical_m53_parent_source_issue_cycles"] for row in selected),
            "windows": {},
        }
        for window in WINDOWS:
            key = str(window)
            derived["windows"][key] = {
                field: sum(row["windows"][key][field] for row in selected)
                for field in fields
            }
        derived_samples.append(derived)
    require(derived_samples == payload["per_sample"], "per-sample aggregation drift")

    local = sum(row["local_zero_source_issue_cycles"] for row in derived_samples)
    canonical = sum(row["canonical_m53_parent_source_issue_cycles"]
                    for row in derived_samples)
    require(local == 141484880 and canonical == 113347744,
            "baseline aggregate identity drift")
    summary = payload["summary"]
    require(summary["local_zero_source_issue_cycles"] == local and
            summary["canonical_m53_parent_source_issue_cycles"] == canonical,
            "summary baseline drift")
    matcher_is_free = True
    for window in WINDOWS:
        key = str(window)
        source = sum(row["windows"][key]["source_issue_cycles"]
                     for row in derived_samples)
        lower = sum(row["windows"][key]["source_or_matcher_lower_bound_cycles"]
                    for row in derived_samples)
        logical = sum(row["windows"][key]["logical_updates"]
                      for row in derived_samples)
        add = sum(row["windows"][key]["signed_add_updates"]
                  for row in derived_samples)
        subtract = sum(row["windows"][key]["signed_subtract_updates"]
                       for row in derived_samples)
        item = summary["windows"][key]
        require((source, lower, logical, add, subtract) ==
                (item["source_issue_cycles"],
                 item["source_or_matcher_lower_bound_cycles"],
                 item["logical_updates"], item["signed_add_updates"],
                 item["signed_subtract_updates"]),
                "summary sum drift W{}".format(window))
        require(add + subtract == logical, "summary signed conservation drift")
        require(abs(item["local_zero_over_source_issue_speedup"] -
                    float(Fraction(local, source))) < 1e-15, "local ratio drift")
        require(abs(item["canonical_m53_over_source_issue_speedup"] -
                    float(Fraction(canonical, source))) < 1e-15, "M53 ratio drift")
        require(abs(item["local_zero_over_source_or_matcher_lower_bound"] -
                    float(Fraction(local, lower))) < 1e-15, "lower ratio drift")
        matcher_is_free = matcher_is_free and source == lower
    require(matcher_is_free, "expected matcher-free lower bound changed")
    return derived_samples


def parse_cpp(text):
    parsed = {"windows": {}}
    for line in text.splitlines():
        fields = line.split()
        if fields[0] == "LOCAL":
            parsed["local"] = int(fields[1])
        elif fields[0] == "CANONICAL":
            parsed["canonical"] = int(fields[1])
        elif fields[0] == "W":
            require(len(fields) == 18, "independent replay output arity drift")
            window = int(fields[1])
            values = list(map(int, fields[2:]))
            parsed["windows"][window] = {
                "source_issue_cycles": values[0],
                "logical_updates": values[1],
                "signed_add_updates": values[2],
                "signed_subtract_updates": values[3],
                "exact_copy_tiles": values[4],
                "zero_parent_tiles": values[5],
                "nonzero_parent_tiles": values[6],
                "matcher_query_rows": values[7],
                "matcher_nominal_cycles": values[8],
                "source_or_matcher_lower_bound_cycles": values[9],
                "origins": values[10:15],
                "maximum_window_distance": values[15],
            }
        else:
            raise ValueError("unexpected independent replay line")
    require(set(parsed["windows"]) == set(WINDOWS), "independent window set drift")
    return parsed


def raw_sample0_replay(payload):
    manifest = load(MANIFEST)
    records = [row for row in manifest["records"] if row["sample_id"] == 0]
    require(len(records) == 4, "sample0 manifest population drift")
    packed_paths = []
    for record in records:
        path = MANIFEST.parent / record["packed_file"]
        require(path.is_file() and not path.is_symlink(), "packed file missing")
        require(sha256(path) == record["packed_file_sha256"], "packed SHA drift")
        packed_paths.append(path)
    with tempfile.TemporaryDirectory(prefix="m69_sample0_hammer_") as temp_name:
        binary = Path(temp_name) / "recompute_m69_sample0"
        compile_result = subprocess.run(
            ["/usr/bin/g++", "-O3", "-std=c++17", "-Wall", "-Wextra",
             "-Werror", str(CPP), "-o", str(binary)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True)
        require(compile_result.returncode == 0,
                "independent C++ replay compile failed: " + compile_result.stdout)
        run = subprocess.run([str(binary)] + [str(path) for path in packed_paths],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True)
        require(run.returncode == 0,
                "independent C++ replay failed: " + run.stdout)
        replay = parse_cpp(run.stdout)

    expected_records = [row for row in payload["records"]
                        if row["sample_id"] == 0]
    require(replay["local"] == sum(row["local_zero_source_issue_cycles"]
                                   for row in expected_records),
            "sample0 raw local mismatch")
    require(replay["canonical"] == sum(
        row["canonical_m53_parent_source_issue_cycles"]
        for row in expected_records), "sample0 raw canonical mismatch")
    origin_order = ("zero", "left", "up", "previous_timestep", "window")
    for window in WINDOWS:
        expected = {}
        for field in ("source_issue_cycles", "logical_updates",
                      "signed_add_updates", "signed_subtract_updates",
                      "exact_copy_tiles", "zero_parent_tiles",
                      "nonzero_parent_tiles", "matcher_query_rows",
                      "matcher_nominal_cycles",
                      "source_or_matcher_lower_bound_cycles"):
            expected[field] = sum(row["windows"][str(window)][field]
                                  for row in expected_records)
        actual = replay["windows"][window]
        for field, value in expected.items():
            require(actual[field] == value,
                    "sample0 raw W{} {} mismatch".format(window, field))
        expected_origins = [sum(row["windows"][str(window)]["origin"].get(name, 0)
                                for row in expected_records)
                            for name in origin_order]
        require(actual["origins"] == expected_origins,
                "sample0 raw origin mismatch W{}".format(window))
        require(actual["maximum_window_distance"] <= window,
                "noncausal/out-of-window selected parent")
    return replay


def gate_and_model_audit(payload):
    best = payload["summary"]["windows"]["64"]
    local_ratio = Fraction(payload["summary"]["local_zero_source_issue_cycles"],
                           best["source_issue_cycles"])
    m53_ratio = Fraction(
        payload["summary"]["canonical_m53_parent_source_issue_cycles"],
        best["source_issue_cycles"])
    require(local_ratio < 3 and m53_ratio < Fraction(3, 2),
            "pre-registered gate unexpectedly passes")
    require(payload["admission"] == {
        "all10_frozen_trace_source_bank_opportunity": True,
        "date_headline": False,
        "dependency_aware_integrated_cycles": False,
        "exact_signed_arithmetic_identity": True,
        "full_network_or_system_speedup": False,
        "memory_feasible": False,
        "same_resource_rtl": False,
    }, "admission boundary drift")
    matcher_nominal = sum(
        row["windows"]["64"]["matcher_nominal_cycles"]
        for row in payload["records"])
    require(matcher_nominal == 40 * 10 * 27 * (300 + 6 + 3),
            "matcher nominal accounting drift")
    return {
        "best_window": 64,
        "local_zero_speedup": float(local_ratio),
        "m53_speedup": float(m53_ratio),
        "local_zero_gate": 3.0,
        "m53_gate": 1.5,
        "decision": "NO_GO_STOP_WINDOW_RTL",
        "matcher_nominal_cycles": matcher_nominal,
        "matcher_lower_bound_equals_source_for_every_window": True,
        "w64_parallel_candidate_source_bits_per_query": 64 * 256,
        "w64_candidate_source_bytes_per_query": 64 * 256 // 8,
    }


def main():
    payload = exact_identity()
    aggregate_replay(payload)
    sample0 = raw_sample0_replay(payload)
    gate = gate_and_model_audit(payload)
    print(json.dumps({
        "status": "PASS_M69_CURRENT_BYTES_NO_GO_WINDOW_RTL",
        "sample0_raw_replay": {
            "local": sample0["local"],
            "canonical": sample0["canonical"],
            "windows_recomputed": list(WINDOWS),
            "signed_conservation": "PASS",
            "bank_metric": "PASS",
            "causal_window": "PASS",
        },
        "all40_aggregate_identity": "PASS",
        "gate": gate,
        "p0_count": 0,
        "p1_count": 3,
        "p2_count": 2,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("FAIL M69 independent hammer: {}".format(error))
        raise SystemExit(1)
