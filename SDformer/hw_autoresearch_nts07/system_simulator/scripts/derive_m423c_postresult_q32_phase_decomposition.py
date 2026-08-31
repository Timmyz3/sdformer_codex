#!/usr/bin/env python3
"""Derive per-phase M423 q32 categories after the frozen held-out result.

This pass does not select a catalog, compute a cycle speedup, or make a new
decision.  It only decomposes the already-admitted M423 population for a
subsequent fixed-formula combination audit.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=lambda value: (_ for _ in ()).throw(
                             RuntimeError("non-standard JSON number: " + value)))


def load_module(path):
    spec = importlib.util.spec_from_file_location("m423c_m43", str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import frozen M43 unpacker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M423c overwrite")
    source_start = sha256(Path(__file__).resolve())
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m423c_postresult_q32_phase_decomposition_contract_v1" and
            contract.get("status") == "FROZEN_READ_ONLY_DERIVATION",
            "M423c contract drift")
    root = args.contract.resolve().parents[1]
    paths = {}
    identities = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file() and sha256(path) == spec["sha256"],
                "M423c input SHA drift: " + name)
        paths[name] = path
        identities[name] = {"path": spec["path"],
                            "sha256": spec["sha256"]}
    require(paths["deriver"].resolve() == Path(__file__).resolve() and
            identities["deriver"]["sha256"] == source_start,
            "M423c deriver self-identity drift")
    catalog = strict_json(paths["m423a_catalog"])
    admitted = strict_json(paths["m423b_result"])
    trace = strict_json(paths["m40_trace"])
    require(admitted["decision"] == "GO_M423_Q32_CATALOG" and
            admitted["split_audit"]["completed_heldout_evaluation_count"] == 1 and
            admitted["split_audit"]["post_heldout_tuning"] is False and
            catalog["admission"]["exact_arithmetic_identity"] is True,
            "M423c predecessor admission drift")

    flags = [[[[False for _ in range(8)] for _ in range(32)]
              for _ in range(432)] for _ in range(4)]
    static_rows = 0
    with paths["static_codec_audit"].open("r", encoding="utf-8",
                                          newline="") as handle:
        for row in csv.DictReader(handle):
            operator = int(row["operator"])
            partition = int(row["partition"])
            center = int(row["center_id"])
            block = int(row["output_block"])
            require(0 <= operator < 4 and 0 <= partition < 432 and
                    0 <= center < 32 and 0 <= block < 8,
                    "M423c static codec index drift")
            flags[operator][partition][center][block] = bool(int(row["narrow"]))
            static_rows += 1
    require(static_rows == 442368, "M423c static codec extent drift")

    m43 = load_module(paths["m43_unpacker"])
    trace_dir = paths["m40_trace"].parent
    operators = tuple(trace["cohort"]["operators"])
    operator_index = {name: index for index, name in enumerate(operators)}
    histograms = defaultdict(Counter)
    payload_files = 0
    payload_bytes = 0
    for record_index, record in enumerate(trace["records"]):
        for key, sha_key in (("packed_file", "packed_file_sha256"),
                             ("value_payload_file", "value_payload_sha256")):
            path = trace_dir / record[key]
            require(path.is_file() and sha256(path) == record[sha_key],
                    "M423c M40 payload drift")
            payload_files += 1
            payload_bytes += path.stat().st_size
        masks = m43.unpack_record_masks(trace_dir, record)
        for source_row in range(m43.ROWS):
            base = source_row * m43.TILES
            for tile in range(m43.TILES):
                value256 = masks[base + tile]
                for subtile in range(16):
                    value = (value256 >> (subtile * 16)) & 0xffff
                    histograms[(int(record["sample_id"]),
                                operator_index[record["operator"]],
                                tile * 16 + subtile)][value] += 1
        print("[M423C HIST] {}/{}".format(
            record_index + 1, len(trace["records"])), flush=True)

    popcount = tuple(bin(value).count("1") for value in range(1 << 16))
    rows = []
    total = Counter()
    for sample in range(10):
        for operator in range(4):
            for partition in range(432):
                counter = histograms[(sample, operator, partition)]
                require(sum(counter.values()) == 3000,
                        "M423c phase extent drift")
                centers = [int(value, 16) for value in
                           catalog["operators"][operator]["partitions"]
                           [partition]["nested_patterns"]]
                phase = Counter()
                for original, count in counter.items():
                    population = popcount[original]
                    if original == 0:
                        continue
                    distance = [popcount[original ^ center]
                                for center in centers]
                    best_distance = min(distance)
                    best_index = distance.index(best_distance)
                    use_pwp = 1 + best_distance < population
                    if use_pwp:
                        phase["pwp_rows"] += count
                        if best_distance == 0:
                            phase["exact_pwp_rows"] += count
                        else:
                            phase["positive_residual_pwp_rows"] += count
                            phase["positive_residual_correction_ops"] += (
                                count * best_distance)
                        phase["correction_ops_per_block"] += (
                            count * best_distance)
                        narrow = sum(flags[operator][partition]
                                     [best_index])
                        phase["narrow_pwp_blocks"] += count * narrow
                        phase["wide_pwp_blocks"] += count * (8 - narrow)
                    else:
                        phase["fallback_rows"] += count
                        phase["fallback_source_ops"] += count * population
                        phase["correction_ops_per_block"] += count * population
                require(phase["pwp_rows"] == phase["exact_pwp_rows"] +
                        phase["positive_residual_pwp_rows"] and
                        phase["correction_ops_per_block"] ==
                        phase["positive_residual_correction_ops"] +
                        phase["fallback_source_ops"] and
                        phase["narrow_pwp_blocks"] + phase["wide_pwp_blocks"] ==
                        phase["pwp_rows"] * 8,
                        "M423c phase conservation failure")
                row = {"sample": sample, "operator": operator,
                       "partition": partition}
                for field in contract["output_fields"]:
                    row[field] = phase[field]
                rows.append(row)
                total.update(phase)
        print("[M423C PHASE] sample={}/10".format(sample + 1), flush=True)

    expected = contract["expected_totals"]
    for field, value in expected.items():
        require(total[field] == value,
                "M423c frozen total mismatch: " + field)
    require(len(rows) == 17280 and source_start ==
            sha256(Path(__file__).resolve()),
            "M423c extent/source drift")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = args.output_dir / "m423c_per_phase_q32_decomposition.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "sample", "operator", "partition"] + contract["output_fields"])
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "schema": "m423c_postresult_q32_phase_decomposition_v1",
        "status": "PASS_M423C_READ_ONLY_POSTRESULT_DECOMPOSITION",
        "identity": identities,
        "derivation_boundary": {
            "catalog_changed": False,
            "cycle_model_changed": False,
            "new_cycle_or_speedup_computed": False,
            "new_catalog_decision": False,
            "post_heldout_tuning": False,
            "payload_read_attempts_total": 3,
            "completed_heldout_evaluations_total": 1,
            "postresult_decomposition_passes": 1
        },
        "payload_audit": {"files_rehashed": payload_files,
                          "bytes_rehashed": payload_bytes,
                          "mismatches": 0},
        "phase_rows": len(rows),
        "totals": dict(total),
        "conservation": {
            "pwp_equals_exact_plus_positive_residual": True,
            "correction_equals_positive_residual_plus_fallback_source": True,
            "pwp_blocks_equal_narrow_plus_wide": True,
            "mismatches": 0
        },
        "admission": {
            "read_only_derived_population": True,
            "standalone_four_bottleneck_conv_only": True,
            "cycle_speedup": False,
            "energy": False,
            "system_speedup": False,
            "date_headline": False
        },
        "output_file": csv_path.name
    }
    (args.output_dir / "m423c_postresult_q32_phase_decomposition_r1.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("M423C_PASS phases={} pwp={} exact={} positive={} correction={}".format(
        len(rows), total["pwp_rows"], total["exact_pwp_rows"],
        total["positive_residual_pwp_rows"],
        total["correction_ops_per_block"]), flush=True)


if __name__ == "__main__":
    main()
