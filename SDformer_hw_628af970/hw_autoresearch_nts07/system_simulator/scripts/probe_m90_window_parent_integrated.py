#!/usr/bin/env python3
"""Non-citable M90 integrated-cycle probe for the M69 row-parent window."""

from __future__ import print_function

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
M45 = HW_ROOT / (
    "system_simulator/scripts/"
    "analyze_m45_dual_destination_bank_fused_integrated_schedule.py")
M43_RESULT = HW_ROOT / (
    "results/m43_tile_resident_parent_delta_schedule_r1_20260823/"
    "m43_spatiotemporal_parent_delta_ablation.json")
EXPECTED = {
    "m45": "c1e3610ce59753f786498db46cde7b330155fa2e3c836198be165aad3eb3f38f",
    "m43_result": "995fa9643ab2180d9b1480b4143959275dc3a04b4b346f8d7e22bed5266a639c",
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
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fraction(numerator, denominator):
    require(denominator > 0, "zero fraction denominator")
    divisor = math.gcd(int(numerator), int(denominator))
    return {
        "numerator": int(numerator) // divisor,
        "denominator": int(denominator) // divisor,
        "decimal": float(numerator) / float(denominator),
    }


def load_extended_m45():
    require(sha256(M45) == EXPECTED["m45"], "M90 M45 source drift")
    source = M45.read_text(encoding="utf-8")
    guard_from = ("require(1 <= fanout_k <= context_capacity <= 8,\n"
                  "            \"invalid fanout/context geometry\")")
    guard_to = ("require(1 <= fanout_k <= context_capacity <= 16,\n"
                "            \"invalid fanout/context geometry\")")
    allow_from = ("require(name in (\"local_zero\", \"left\", \"up\"),\n"
                  "                \"temporal parent leaked into M45 primary\")")
    allow_to = ("require(name in (\"local_zero\", \"left\", \"up\",\n"
                "                         \"previous_timestep\") or\n"
                "                name.startswith(\"window_\"),\n"
                "                \"invalid M90 window parent\")")
    require(source.count(guard_from) == 1 and source.count(allow_from) == 1,
            "M90 M45 transform anchor drift")
    transformed = source.replace(guard_from, guard_to).replace(
        allow_from, allow_to)
    namespace = {
        "__file__": str(M45),
        "__name__": "m90_window_parent_m45",
    }
    exec(compile(transformed, str(M45) + "#M90_WINDOW", "exec"), namespace)
    namespace["validate_contract"]()
    return namespace, source, transformed


def install_window_parent(namespace, window):
    m43 = namespace["load_m43_module"]()
    m43.ALLOW_TEMPORAL_PARENT = True
    origins = Counter()
    distances = Counter()

    def metric(current, parent, priority):
        delta = current ^ parent
        return (m43.bank_issue_cycles(delta), m43.population(delta),
                priority, parent, delta)

    def select_parent(masks, row, tile):
        index = row * m43.TILES + tile
        current = masks[index]
        timestep, spatial = divmod(row, m43.HEIGHT * m43.WIDTH)
        y, x = divmod(spatial, m43.WIDTH)
        candidates = [("local_zero", metric(current, 0, 0))]
        if x > 0:
            candidates.append(("left", metric(
                current, masks[index - m43.TILES], 1)))
        if y > 0:
            candidates.append(("up", metric(
                current, masks[index - m43.WIDTH * m43.TILES], 2)))
        if timestep > 0:
            candidates.append(("previous_timestep", metric(
                current,
                masks[index - m43.HEIGHT * m43.WIDTH * m43.TILES], 3)))
        maximum = min(window, spatial)
        for distance in range(1, maximum + 1):
            parent_index = ((timestep * m43.HEIGHT * m43.WIDTH +
                             spatial - distance) * m43.TILES + tile)
            candidates.append(("window_{}".format(distance), metric(
                current, masks[parent_index], 4 + distance)))
        name, selected = min(candidates, key=lambda item: item[1][0:3])
        _, _, _, parent, delta = selected
        add_mask = current & ~parent
        subtract_mask = parent & ~current
        require((add_mask & subtract_mask) == 0 and
                (add_mask | subtract_mask) == delta,
                "M90 signed correction partition drift")
        origins["window" if name.startswith("window_") else name] += 1
        if name.startswith("window_"):
            distances[int(name.split("_", 1)[1])] += 1
        return name, parent, add_mask, subtract_mask

    def build_structural_dag(selected_parent):
        indegree = [0] * namespace["ROWS_PER_T"]
        children = [[] for _ in range(namespace["ROWS_PER_T"])]
        for spatial, name in enumerate(selected_parent):
            y, x = divmod(spatial, namespace["W"])
            parents = set()
            # Preserve the canonical M45 conservative row-progress edge.
            if y > 0:
                parents.add(spatial - namespace["W"])
            if name == "left":
                require(x > 0, "M90 illegal left parent")
                parents.add(spatial - 1)
            elif name == "up":
                require(y > 0, "M90 illegal up parent")
                parents.add(spatial - namespace["W"])
            elif name.startswith("window_"):
                distance = int(name.split("_", 1)[1])
                require(1 <= distance <= window and distance <= spatial,
                        "M90 illegal window distance")
                parents.add(spatial - distance)
            else:
                require(name in ("local_zero", "previous_timestep"),
                        "M90 unknown parent name")
            for parent in sorted(parents):
                indegree[spatial] += 1
                children[parent].append(spatial)
        return indegree, children

    m43.select_parent = select_parent
    namespace["build_structural_dag"] = build_structural_dag
    return m43, origins, distances


def analyze(fanout, window):
    namespace, source, transformed = load_extended_m45()
    m43, origins, distances = install_window_parent(namespace, window)
    manifest = namespace["read_json"](namespace["MANIFEST"])
    reference = read_json(M43_RESULT)
    references = dict(((row["sample_id"], row["operator"]), row)
                      for row in reference["records"])
    require(len(manifest["records"]) == 40 and len(references) == 40,
            "M90 frozen cohort drift")
    per_record = []
    for index, record in enumerate(manifest["records"]):
        key = (record["sample_id"], record["operator"])
        require(key in references, "M90 missing M43 reference")
        masks = m43.unpack_record_masks(namespace["MANIFEST"].parent, record)
        row = namespace["analyze_record"](
            m43, masks, references[key], fanout, 16)
        row["sample_id"] = record["sample_id"]
        row["operator"] = record["operator"]
        per_record.append(row)
        print("[M90 K{} W{}] {}/40 sample={} operator={}".format(
            fanout, window, index + 1, record["sample_id"],
            record["operator"]), flush=True)

    blank = namespace["blank_counts"]()
    sum_fields = [name for name in blank if not name.startswith("maximum_")]
    sum_fields += ["signed_add_updates", "signed_subtract_updates",
                   "weight_dma_bytes", "final_accumulator_read_bytes",
                   "final_accumulator_write_bytes", "completed_output_bytes"]
    per_sample = []
    for sample_id in range(10):
        selected = [row for row in per_record if row["sample_id"] == sample_id]
        require(len(selected) == 4, "M90 sample/operator population drift")
        sample = {"sample_id": sample_id}
        for field in sum_fields:
            sample[field] = sum(row[field] for row in selected)
        for field in ("maximum_metadata_occupancy",
                      "maximum_complete_occupancy",
                      "maximum_resident_occupancy"):
            sample[field] = max(row[field] for row in selected)
        per_sample.append(sample)
    result = namespace["aggregate_configuration"](
        "K{}_CTX16_WINDOW{}_TEMPORAL".format(fanout, window),
        fanout, 16, per_sample)
    result["window_parent"] = {
        "window": window,
        "origin_counts": dict(origins),
        "distance_counts": dict(distances),
        "canonical_m45_sha256": hashlib.sha256(
            source.encode("utf-8")).hexdigest(),
        "transformed_m45_sha256": hashlib.sha256(
            transformed.encode("utf-8")).hexdigest(),
        "conservative_m45_row_progress_edge_preserved": True,
        "exact_selected_window_parent_dependency_edge_added": True,
    }
    result["claim_policy"] = {
        "status": "NON_CITABLE_SCREENING_ONLY",
        "paper_ppa_ready": False,
        "rtl_cycle_speedup": False,
        "system_speedup": False,
        "headline": False,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fanout", type=int, choices=(4, 5, 6, 7, 8),
                        required=True)
    parser.add_argument("--window", type=int, choices=(8, 16, 32, 64),
                        required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    require(sha256(M43_RESULT) == EXPECTED["m43_result"],
            "M90 M43 result drift")
    result = analyze(args.fanout, args.window)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    compact = {
        "fanout": args.fanout,
        "window": args.window,
        "source": result["aggregate_source_only_cycles"],
        "integrated": result["aggregate_integrated_cycles"],
        "p95": result["integrated_cycle_distribution"]["p95_nearest_rank"],
        "max_metadata": max(row["maximum_metadata_occupancy"]
                            for row in result["per_sample"]),
        "max_complete": max(row["maximum_complete_occupancy"]
                            for row in result["per_sample"]),
        "window_choices": result["window_parent"]["origin_counts"].get(
            "window", 0),
    }
    print("M90_WINDOW_PARENT_PROBE=" + json.dumps(compact, sort_keys=True))


if __name__ == "__main__":
    main()
