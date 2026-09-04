#!/usr/bin/env python3
"""Independent static/mathematical preflight for M507 r2.

This checker deliberately does not import or execute the production analyzer,
does not decompress/replay raw activation payloads, and does not create an
M507 production result.  It checks frozen identities, manifest metadata,
M501 reconciliation coverage, source/AST obligations, the resource ledger,
and four independently implemented boundary-cycle examples.
"""

import ast
import hashlib
import json
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts/m507_h67_apec_g2_same_resource_cycle_fastkill_contract_r2_20260827.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def service(events: int, taps: int, model: dict) -> dict:
    """Independent service formula; no production source is imported."""
    output_channels = int(model["output_channels"])
    lanes = int(model["compute_lanes"])
    aggregate_weight_bw = int(model["weight_bytes_per_cycle"])
    banks = int(model["weight_banks"])
    bank_bw = int(model["weight_bank_bytes_per_cycle"])
    compute = events * math.ceil(output_channels * taps / lanes)
    weight_bytes = events * output_channels * taps
    weight = math.ceil(weight_bytes / aggregate_weight_bw)
    per_bank = math.ceil(
        events * taps * (output_channels // banks) / bank_bw
    )
    return {
        "compute": compute,
        "weight": weight,
        "per_bank_weight": per_bank,
        "mapping_equal": weight == per_bank,
    }


def commit(taps: int, model: dict) -> dict:
    channels = int(model["output_channels"])
    bits = int(model["accumulator_bits"])
    banks = int(model["output_banks"])
    bank_bw = int(model["output_bank_bytes_per_cycle"])
    per_bank_bytes = math.ceil((channels // banks) * taps * bits / 8)
    cycles = math.ceil(per_bank_bytes / bank_bw)
    aggregate_bytes = math.ceil(channels * taps * bits / 8)
    aggregate_cycles = math.ceil(aggregate_bytes / (banks * bank_bw))
    return {
        "cycles": cycles,
        "aggregate_cycles": aggregate_cycles,
        "mapping_equal": cycles == aggregate_cycles,
    }


def hand_case(name: str, count0: int, count1: int, common: int,
              taps0: int, taps1: int, union_taps: int, model: dict) -> dict:
    """Independent group-cycle calculation for one timestep/group."""
    assert 0 <= common <= min(count0, count1)
    startup = int(model["weight_startup_latency_cycles"])
    bitmap = int(model["bitmap_pair_read_cycles"])
    compare = int(model["exact_compare_cycles"])

    b0 = service(count0, taps0, model)
    b1 = service(count1, taps1, model)
    bcompute = b0["compute"] + b1["compute"]
    bweight = b0["weight"] + b1["weight"]
    bexec = max(bcompute, bweight + (startup if count0 + count1 else 0))

    c0 = service(count0 - common, taps0, model)
    c1 = service(count1 - common, taps1, model)
    cov = service(common, union_taps, model)
    ccompute = c0["compute"] + c1["compute"] + cov["compute"]
    cweight = c0["weight"] + c1["weight"] + cov["weight"]
    cexec = max(
        ccompute,
        cweight + (startup if count0 + count1 - common else 0),
    )

    out0 = commit(taps0, model)["cycles"] if count0 else 0
    out1 = commit(taps1, model)["cycles"] if count1 else 0
    common_commit = out0 + out1
    scratch_pass = 0
    scratch = 0
    if common:
        scratch_bytes = math.ceil(
            int(model["output_channels"]) * union_taps *
            int(model["accumulator_bits"]) / 8
        )
        scratch_pass = math.ceil(
            scratch_bytes / int(model["scratch_bytes_per_cycle"])
        )
        scratch = 3 * scratch_pass + 2

    baseline = bitmap + bexec + common_commit
    candidate = bitmap + compare + cexec + scratch + common_commit
    return {
        "case": name,
        "inputs": {
            "count0": count0,
            "count1": count1,
            "common": common,
            "taps0": taps0,
            "taps1": taps1,
            "union_taps": union_taps,
        },
        "baseline_execution_cycles": bexec,
        "candidate_execution_cycles": cexec,
        "destination_commit_cycles_each_arm": common_commit,
        "scratch_pass_cycles": scratch_pass,
        "candidate_scratch_cycles_including_two_read_tails": scratch,
        "baseline_cycles": baseline,
        "candidate_cycles": candidate,
        "cycle_ratio": baseline / candidate,
        "queue_backpressure_baseline": max(0, baseline - 1),
        "queue_backpressure_candidate": max(0, candidate - 1),
    }


def manifest_metadata_audit(manifest: dict, expectation: dict) -> dict:
    records = manifest["records"]
    operators = expectation["operators"]
    samples = int(expectation["samples"])
    expected_pairs = {(sample, operator)
                      for sample in range(samples) for operator in operators}
    observed_pairs = {(int(row["sample_id"]), row["operator"])
                      for row in records}
    codeword_errors = []
    geometry_errors = []
    for index, row in enumerate(records):
        operator = row["operator"]
        expected_word = expectation["operator_nonzero_codeword_bits"][operator]
        words = {entry["float32_bits_hex"]
                 for entry in row["value_bit_pattern_population"]["codebook"]}
        if (words != {"00000000", expected_word} or
                int(row["negative_count"]) != 0 or
                int(row["nonzero_count"]) != int(row["positive_count"])):
            codeword_errors.append(index)
        geometry = row["module_geometry"]
        if not (
            list(row["shape"]) == list(expectation["shape"]) and
            list(row["output_shape"]) == list(expectation["shape"]) and
            int(geometry["in_channels"]) == 768 and
            int(geometry["out_channels"]) == 768 and
            list(geometry["kernel_size"]) == [3, 3] and
            list(geometry["stride"]) == [1, 1] and
            list(geometry["padding"]) == [1, 1] and
            list(geometry["dilation"]) == [1, 1] and
            int(geometry["groups"]) == 1
        ):
            geometry_errors.append(index)
    return {
        "records": len(records),
        "record_count_matches": len(records) == int(expectation["records"]),
        "cartesian_pairs_expected": len(expected_pairs),
        "cartesian_pairs_observed": len(observed_pairs),
        "cartesian_complete_unique": (
            len(records) == len(observed_pairs) and
            observed_pairs == expected_pairs
        ),
        "codeword_metadata_errors": codeword_errors,
        "geometry_errors": geometry_errors,
    }


def m501_coverage_audit(m501: dict) -> dict:
    result = {}
    for cohort in m501["cohorts"]:
        name = cohort["cohort"]
        detail = [row for row in cohort["detailed"]
                  if row["axis"] == "horizontal" and
                  int(row["group_size"]) == 2]
        overall = [row for row in cohort["aggregate"]["overall"]
                   if row["axis"] == "horizontal" and
                   int(row["group_size"]) == 2]
        sequences = [row for row in cohort["aggregate"]["per_sequence"]
                     if row["axis"] == "horizontal" and
                     int(row["group_size"]) == 2]
        keys = {(int(row["sample_id"]), row["sample_key"], row["operator"])
                for row in detail}
        result[name] = {
            "detail_rows": len(detail),
            "unique_detail_keys": len(keys),
            "overall_rows": len(overall),
            "sequence_rows": len(sequences),
            "sequence_names": sorted({row["sequence"] for row in sequences}),
        }
    return result


def main() -> None:
    contract = read_json(CONTRACT)
    model = contract["cycle_model"]
    analyzer = ROOT / contract["inputs"]["analyzer"]["path"]
    source = analyzer.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = sorted(node.name for node in tree.body
                       if isinstance(node, ast.FunctionDef))

    input_hashes = {}
    for name, spec in contract["inputs"].items():
        path = ROOT / spec["path"]
        actual = sha256(path) if path.is_file() else None
        input_hashes[name] = {
            "path": spec["path"],
            "expected": spec["sha256"],
            "actual": actual,
            "match": actual == spec["sha256"],
        }

    # Independent 240-KiB component calculation.
    total = int(model["common_total_sram_bytes"])
    bitmap = math.ceil(2 * int(model["input_channels"]) / 8)
    overlap = math.ceil(
        int(model["output_channels"]) * 9 *
        int(model["accumulator_bits"]) / 8
    )
    destination = 2 * overlap
    payload = total - bitmap - overlap - destination
    independent_capacity = {
        "pair_bitmap_bytes": bitmap,
        "overlap_cache_bytes": overlap,
        "two_destination_vector_slots_bytes": destination,
        "payload_and_weight_window_bytes": payload,
    }

    m40 = read_json(ROOT / contract["inputs"]["m40_manifest"]["path"])
    m73 = read_json(ROOT / contract["inputs"]["m73_manifest"]["path"])
    m501 = read_json(ROOT / contract["inputs"]["m501_result"]["path"])

    exspike = Path("/home/zhumd/work/literature_artifacts/ExSpike")
    prior_commit = subprocess.run(
        ["git", "-C", str(exspike), "rev-parse", "HEAD"],
        check=True, universal_newlines=True, stdout=subprocess.PIPE,
    ).stdout.strip()

    output_ports_literal = model.get("output_banks") is not None
    result = {
        "scope": {
            "production_analyzer_imported": False,
            "production_main_executed": False,
            "raw_payload_replayed": False,
        },
        "identity": {
            "contract_sha256": sha256(CONTRACT),
            "analyzer_sha256": sha256(analyzer),
            "input_hashes": input_hashes,
            "all_contract_input_hashes_match": all(
                item["match"] for item in input_hashes.values()
            ),
            "exspike_commit": prior_commit,
        },
        "source_ast": {
            "functions": functions,
            "has_two_destination_commit_paths": (
                'destination_commit_cycles_baseline' in source and
                'destination_commit_cycles_candidate' in source and
                'bcommit' in source and 'ccommit' in source
            ),
            "has_two_synchronous_read_tails": (
                'scratch_sync_read_tail_cycles' in source and
                'scratch_cycles = 3 * scratch_pass + 2' in source
            ),
            "has_geometry_check": 'frozen Conv geometry drift' in source,
            "has_cartesian_check": 'Cartesian coverage drift' in source,
            "has_exact_codeword_check": (
                'decoded payload contains a third codeword' in source
            ),
            "has_m501_both_cohorts": (
                '"validation_s10"' in source and
                '"train_calibration_s32"' in source
            ),
            "has_per_record_per_sequence_reconciliation": (
                'per-record event ledger mismatch' in source and
                'per-sequence event ledger mismatch' in source
            ),
            "has_queue_backpressure_counter": (
                'group_queue_backpressure_cycles_baseline' in source and
                'group_queue_max_occupancy_candidate' in source
            ),
            "output_bank_model_present": output_ports_literal,
            "destination_accumulator_port_entry_present": (
                '"destination_accumulator_slots"' in source or
                '"destination_accumulator"' in source
            ),
        },
        "resource_math": {
            "independent_capacity": independent_capacity,
            "capacity_sum": sum(independent_capacity.values()),
            "capacity_equals_240_kib": (
                sum(independent_capacity.values()) == 240 * 1024
            ),
            "contract_components_match": {
                "bitmap": bitmap == int(model["pair_bitmap_buffer_bytes"]),
                "overlap": overlap == int(model["reserved_overlap_scratch_bytes"]),
                "destination": destination == int(model["destination_vector_slots_bytes"]),
                "payload": payload == int(model["payload_and_weight_window_bytes"]),
            },
            "weight_bank_mapping_one_event_full_kernel": service(1, 9, model),
            "output_bank_mapping_full_kernel": commit(9, model),
        },
        "manifest_metadata": {
            "m40": manifest_metadata_audit(
                m40, contract["cohort_expectations"]["m40"]
            ),
            "m73": manifest_metadata_audit(
                m73, contract["cohort_expectations"]["m73"]
            ),
        },
        "m501_coverage": m501_coverage_audit(m501),
        "independent_hand_cases": [
            hand_case("empty_interior", 0, 0, 0, 9, 9, 9, model),
            hand_case("full_overlap_interior", 1, 1, 1, 9, 9, 9, model),
            hand_case("no_overlap_interior", 1, 1, 0, 9, 9, 9, model),
            hand_case("full_overlap_top_left_pair", 1, 1, 1, 4, 6, 6, model),
        ],
        "blockers": [
            {
                "id": "P0_IDENTITY_SEAL_FILE_SHA",
                "detail": "The contract locks the hash written inside the old review seal, not the SHA256 of the seal file itself.",
            },
            {
                "id": "P0_DESTINATION_ACCUMULATOR_PORT_LEDGER",
                "detail": "The capacity ledger reserves two destination-vector accumulator slots, but build_resource_ledger has no named read/modify/write or readout ports for those slots; output_sink exposes write banks only.",
            },
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
