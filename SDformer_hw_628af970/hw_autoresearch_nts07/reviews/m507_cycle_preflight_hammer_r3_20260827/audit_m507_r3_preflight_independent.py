#!/usr/bin/env python3
"""Independent static/mathematical audit for M507 r3.

This checker never imports or executes the production analyzer and never opens
the compressed trace payloads.  It checks frozen identities, distinguishes the
outer review-seal hash from the sealed-manifest hash, recomputes the 240-KiB
capacity/port arithmetic, and evaluates four hand-checkable boundary groups.
"""

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts/m507_h67_apec_g2_same_resource_cycle_fastkill_contract_r3_20260827.json"
ANALYZER = ROOT / "system_simulator/scripts/analyze_m507_h67_apec_g2_same_resource_cycle_fastkill_r3.py"
EXPECTED_CONTRACT_SHA = "34128e04e31742c857914b232eb8cecf9ff02b834230e9aa8b77fda602a4b88a"
EXPECTED_ANALYZER_SHA = "561d3e06dce9fac87e61bb1f18844f29630f881bbc4c5a8d51f30bc4f4552045"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            assert key not in result, "duplicate JSON key: " + key
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)


def check_manifest(manifest, base):
    missing = []
    mismatch = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(None, 1)
        path = base / name.strip()
        if not path.is_file():
            missing.append(name.strip())
        elif sha256(path) != expected:
            mismatch.append(name.strip())
    return {"missing": missing, "mismatch": mismatch}


def slot_terms(taps, model):
    channels = int(model["output_channels"])
    bits = int(model["accumulator_bits"])
    banks = int(model["destination_slot_banks"])
    bank_bw = int(model["destination_slot_bank_bytes_per_cycle"])
    logical_bytes = math.ceil(channels * taps * bits / 8)
    bytes_per_bank = math.ceil((channels // banks) * taps * bits / 8)
    cycles = math.ceil(bytes_per_bank / bank_bw)
    physical_bytes = cycles * banks * bank_bw
    return logical_bytes, cycles, physical_bytes


def service_terms(events, taps, model):
    operations = events * int(model["output_channels"]) * taps
    compute = events * math.ceil(
        int(model["output_channels"]) * taps / int(model["compute_lanes"]))
    weight = math.ceil(operations / int(model["weight_bytes_per_cycle"]))
    return compute, weight


def commit_terms(taps, model):
    logical, slot_cycles, physical = slot_terms(taps, model)
    sink_bw = int(model["output_banks"]) * int(model["output_bank_bytes_per_cycle"])
    sink_cycles = math.ceil(logical / sink_bw)
    tail = int(model["destination_slot_sync_read_latency_cycles"])
    current = max(slot_cycles + tail, sink_cycles)
    causal = max(slot_cycles, sink_cycles) + tail
    return {
        "logical_bytes": logical,
        "physical_slot_bytes": physical,
        "slot_cycles": slot_cycles,
        "sink_cycles": sink_cycles,
        "current_cycles": current,
        "causal_cycles": causal,
    }


def group_cycles(taps0, taps1, event0, event1, common, union_taps, model):
    startup = int(model["weight_startup_latency_cycles"])
    bitmap = int(model["bitmap_pair_read_cycles"])
    compare = int(model["exact_compare_cycles"])

    baseline_zero = sum(slot_terms(t, model)[1] for e, t in
                        ((event0, taps0), (event1, taps1)) if e)
    baseline_compute = sum(service_terms(e, t, model)[0] for e, t in
                           ((event0, taps0), (event1, taps1)))
    baseline_weight = sum(service_terms(e, t, model)[1] for e, t in
                          ((event0, taps0), (event1, taps1)))
    baseline_tails = int(bool(event0)) + int(bool(event1))
    baseline_exec = max(baseline_compute + baseline_tails,
                        baseline_weight + (startup if event0 + event1 else 0))
    baseline_commit = sum(commit_terms(t, model)["current_cycles"] for e, t in
                          ((event0, taps0), (event1, taps1)) if e)
    baseline = bitmap + baseline_zero + baseline_exec + baseline_commit

    residual0 = event0 - common
    residual1 = event1 - common
    candidate_zero = sum(slot_terms(t, model)[1] for e, t in
                         ((event0, taps0), (event1, taps1)) if e and not common)
    candidate_compute = sum(service_terms(e, t, model)[0] for e, t in
                            ((residual0, taps0), (residual1, taps1),
                             (common, union_taps)))
    candidate_weight = sum(service_terms(e, t, model)[1] for e, t in
                           ((residual0, taps0), (residual1, taps1),
                            (common, union_taps)))
    candidate_tails = int(bool(residual0)) + int(bool(residual1))
    candidate_exec = max(candidate_compute + candidate_tails,
                         candidate_weight +
                         (startup if residual0 + residual1 + common else 0))
    scratch = 0
    if common:
        scratch_bytes = math.ceil(
            int(model["output_channels"]) * union_taps *
            int(model["accumulator_bits"]) / 8)
        scratch_pass = math.ceil(scratch_bytes /
                                 int(model["scratch_bytes_per_cycle"]))
        scratch = 3 * scratch_pass + 2
    candidate_commit = sum(commit_terms(t, model)["current_cycles"] for e, t in
                           ((event0, taps0), (event1, taps1)) if e)
    candidate = (bitmap + compare + candidate_zero + candidate_exec +
                 scratch + candidate_commit)
    return {"baseline": baseline, "candidate": candidate,
            "ratio": baseline / candidate}


def main():
    assert sha256(CONTRACT) == EXPECTED_CONTRACT_SHA
    assert sha256(ANALYZER) == EXPECTED_ANALYZER_SHA
    contract = read_strict_json(CONTRACT)
    model = contract["cycle_model"]

    input_checks = {}
    for name, spec in contract["inputs"].items():
        path = ROOT / spec["path"]
        actual = sha256(path) if path.is_file() else "MISSING"
        input_checks[name] = {"expected": spec["sha256"], "actual": actual,
                              "match": actual == spec["sha256"]}
    assert all(row["match"] for row in input_checks.values())

    seal_checks = {}
    for key in ("m507_r1_preflight_review_seal",
                "m507_r2_preflight_review_seal"):
        spec = contract["inputs"][key]
        seal = ROOT / spec["path"]
        inner = seal.read_text(encoding="utf-8").split()[0]
        assert sha256(seal) == spec["sha256"]
        assert inner == spec["sealed_manifest_sha256"]
        manifest = seal.with_name("SHA256SUMS")
        assert sha256(manifest) == inner
        base = ROOT if key.startswith("m507_r1") else manifest.parent
        seal_checks[key] = {
            "outer_file_sha": sha256(seal),
            "inner_manifest_sha": inner,
            "current_manifest_members": check_manifest(manifest, base),
        }

    channels = int(model["input_channels"])
    outputs = int(model["output_channels"])
    acc_bits = int(model["accumulator_bits"])
    kernel = 1
    for value in model["kernel"]:
        kernel *= int(value)
    pair_bitmap = math.ceil(2 * channels / 8)
    overlap = math.ceil(outputs * kernel * acc_bits / 8)
    destinations = 2 * overlap
    payload = int(model["common_total_sram_bytes"]) - pair_bitmap - overlap - destinations
    capacity = {
        "pair_bitmap_bytes": pair_bitmap,
        "overlap_cache_bytes": overlap,
        "two_destination_vector_slots_bytes": destinations,
        "payload_and_weight_window_bytes": payload,
        "sum_bytes": pair_bitmap + overlap + destinations + payload,
    }
    assert capacity["sum_bytes"] == 240 * 1024
    assert destinations == 32832

    lane_demand_per_bank = math.ceil(
        (int(model["compute_lanes"]) // int(model["destination_slot_banks"])) *
        acc_bits / 8)
    assert lane_demand_per_bank == 29
    assert lane_demand_per_bank <= int(model["destination_slot_bank_bytes_per_cycle"])

    boundaries = {
        "empty_interior": group_cycles(9, 9, 0, 0, 0, 9, model),
        "one_each_full_overlap_interior": group_cycles(9, 9, 1, 1, 1, 9, model),
        "one_each_no_overlap_interior": group_cycles(9, 9, 1, 1, 0, 9, model),
        "one_each_full_overlap_top_left_pair": group_cycles(4, 6, 1, 1, 1, 6, model),
    }
    assert boundaries["empty_interior"]["baseline"] == 2
    assert boundaries["empty_interior"]["candidate"] == 3
    assert boundaries["one_each_full_overlap_interior"]["baseline"] == 536
    assert boundaries["one_each_full_overlap_interior"]["candidate"] == 722
    assert boundaries["one_each_no_overlap_interior"]["baseline"] == 536
    assert boundaries["one_each_no_overlap_interior"]["candidate"] == 537
    assert boundaries["one_each_full_overlap_top_left_pair"]["baseline"] == 299
    assert boundaries["one_each_full_overlap_top_left_pair"]["candidate"] == 454

    source = ANALYZER.read_text(encoding="utf-8")
    static_findings = {
        "common_service_exists":
            "ccommon = service_terms(common, union_taps, model)" in source,
        "common_destination_rmw_exists":
            "destination_rmw_terms(common" in source,
        "non_atomic_direct_output_mkdir":
            "args.output_dir.mkdir(parents=True)" in source,
        "run_complete_excluded_from_seal":
            ("write_seal(args.output_dir, [result_name, csv_name, seq_name, readme_name])" in source and
             "[result_name, csv_name, seq_name, readme_name, \"RUN_COMPLETE.txt\"]" not in source),
    }
    assert static_findings["common_service_exists"]
    assert not static_findings["common_destination_rmw_exists"]

    report = {
        "status": "NO_GO_REVISE_R4_BEFORE_ONE_SHOT_EXECUTION",
        "score": 72,
        "input_checks": input_checks,
        "seal_checks": seal_checks,
        "capacity": capacity,
        "destination_port": {
            "banks_per_slot": int(model["destination_slot_banks"]),
            "bytes_per_bank_per_cycle": int(model["destination_slot_bank_bytes_per_cycle"]),
            "lane_demand_bytes_per_bank_per_cycle": lane_demand_per_bank,
            "full_9tap_slot": commit_terms(9, model),
        },
        "boundary_groups": boundaries,
        "static_findings": static_findings,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
