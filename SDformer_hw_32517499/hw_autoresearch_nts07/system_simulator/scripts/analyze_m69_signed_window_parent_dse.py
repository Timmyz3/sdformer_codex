#!/usr/bin/env python3
"""M69 exact frozen-trace opportunity DSE for a signed row-parent window.

This is deliberately a source-bank opportunity oracle, not an integrated-cycle
or system claim.  A parent is either zero, the canonical left/up/previous-time
candidate, or an already-produced row from the same timestep.  The identity

    W*x = W*p + W*(x-p)

is exact for binary x and p; the correction is represented by disjoint signed
add/subtract masks.  Each finite source bank issues at most one source per
cycle.  The selected window is promoted only after a dependency-aware
transaction simulator charges matcher, parent-read, context and memory costs.
"""

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M43_PATH = HW / "system_simulator/scripts/analyze_m43_tile_resident_parent_delta_schedule.py"
MANIFEST_PATH = HW / (
    "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822/"
    "m40_bottleneck_packed_source_manifest.json")
M53_PATH = HW / (
    "results/m53_adaptive_temporal_parent_k4_ctx16_dse_r1_20260823/"
    "m53_adaptive_temporal_parent_k4_ctx16_dse.json")
EXPECTED_SHA256 = {
    "m43": "a4ddebf4687b32c65735c591a6526f43b7274777ace4e3ca90d19a2d04adb1c3",
    "manifest": "e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3",
    "m53": "344ae1f777e0640d46b19118f0b6d451465046350d68a9f33b1faae124747bb4",
}
WINDOWS = (1, 2, 4, 8, 16, 32, 64)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: {}".format(raw))

    def pairs_hook(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: {}".format(key))
            value[key] = item
        return value

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def load_m43():
    spec = importlib.util.spec_from_file_location("m69_m43", M43_PATH)
    require(spec is not None and spec.loader is not None, "cannot load M43")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ALLOW_TEMPORAL_PARENT = True
    return module


def metric(m43, current, parent, priority):
    delta = current ^ parent
    return (m43.bank_issue_cycles(delta), m43.population(delta), priority,
            parent, delta)


def add_choice(m43, accumulator, current, selected, origin):
    _, _, _, parent, delta = selected
    add_mask = current & ~parent
    subtract_mask = parent & ~current
    require((add_mask & subtract_mask) == 0 and
            (add_mask | subtract_mask) == delta,
            "signed correction partition mismatch")
    accumulator["source_issue_cycles"] += m43.bank_issue_cycles(delta) * m43.OUTPUT_BLOCKS
    accumulator["logical_updates"] += m43.population(delta) * m43.OUTPUT_BLOCKS
    accumulator["signed_add_updates"] += m43.population(add_mask) * m43.OUTPUT_BLOCKS
    accumulator["signed_subtract_updates"] += m43.population(subtract_mask) * m43.OUTPUT_BLOCKS
    accumulator["exact_copy_tiles"] += int(delta == 0 and parent != 0)
    accumulator["zero_parent_tiles"] += int(parent == 0)
    accumulator["nonzero_parent_tiles"] += int(parent != 0)
    accumulator["origin"][origin] += 1


def analyze_record(m43, manifest_dir, record):
    masks = m43.unpack_record_masks(manifest_dir, record)
    totals = dict((window, {
        "source_issue_cycles": 0,
        "logical_updates": 0,
        "signed_add_updates": 0,
        "signed_subtract_updates": 0,
        "exact_copy_tiles": 0,
        "zero_parent_tiles": 0,
        "nonzero_parent_tiles": 0,
        "origin": defaultdict(int),
        "matcher_query_rows": 0,
        "matcher_nominal_cycles": 0,
        "source_or_matcher_lower_bound_cycles": 0,
    }) for window in WINDOWS)
    canonical = {
        "source_issue_cycles": 0,
        "logical_updates": 0,
        "signed_add_updates": 0,
        "signed_subtract_updates": 0,
        "exact_copy_tiles": 0,
        "zero_parent_tiles": 0,
        "nonzero_parent_tiles": 0,
        "origin": defaultdict(int),
    }
    local_zero_cycles = 0

    for timestep in range(m43.TIMESTEPS):
        row_base = timestep * m43.HEIGHT * m43.WIDTH
        for tile in range(m43.TILES):
            per_window_tile_cycles = dict((window, 0) for window in WINDOWS)
            for spatial in range(m43.HEIGHT * m43.WIDTH):
                row = row_base + spatial
                index = row * m43.TILES + tile
                current = masks[index]
                local_zero_cycles += m43.bank_issue_cycles(current) * m43.OUTPUT_BLOCKS
                y, x = divmod(spatial, m43.WIDTH)
                candidates = [("zero", metric(m43, current, 0, 0))]
                if x > 0:
                    candidates.append(("left", metric(
                        m43, current, masks[index - m43.TILES], 1)))
                if y > 0:
                    candidates.append(("up", metric(
                        m43, current,
                        masks[index - m43.WIDTH * m43.TILES], 2)))
                if timestep > 0:
                    candidates.append(("previous_timestep", metric(
                        m43, current,
                        masks[index - m43.HEIGHT * m43.WIDTH * m43.TILES], 3)))
                canonical_origin, canonical_best = min(
                    candidates, key=lambda item: item[1][0:3])
                add_choice(m43, canonical, current, canonical_best, canonical_origin)

                # The cumulative best for distance d is also the exact result
                # for every configured window whose extent is d or smaller.
                best_origin, best = canonical_origin, canonical_best
                distance_results = {}
                for distance in range(1, min(max(WINDOWS), spatial) + 1):
                    parent_spatial = spatial - distance
                    parent_index = (row_base + parent_spatial) * m43.TILES + tile
                    candidate = metric(m43, current, masks[parent_index],
                                       4 + distance)
                    if candidate[0:3] < best[0:3]:
                        best_origin, best = "window", candidate
                    if distance in WINDOWS:
                        distance_results[distance] = (best_origin, best)
                for window in WINDOWS:
                    if spatial == 0:
                        selected_origin, selected = canonical_origin, canonical_best
                    else:
                        extent = min(window, spatial)
                        if extent in distance_results:
                            selected_origin, selected = distance_results[extent]
                        else:
                            selected_origin, selected = best_origin, best
                    add_choice(m43, totals[window], current, selected, selected_origin)
                    per_window_tile_cycles[window] += selected[0] * m43.OUTPUT_BLOCKS

            for window in WINDOWS:
                # One row query per cycle plus a conservative logarithmic
                # registered reduction.  It is shared by all eight output
                # blocks and may overlap source execution after the first tile.
                matcher_cycles = (m43.HEIGHT * m43.WIDTH +
                                  int(math.ceil(math.log(window, 2))) + 3)
                totals[window]["matcher_query_rows"] += m43.HEIGHT * m43.WIDTH
                totals[window]["matcher_nominal_cycles"] += matcher_cycles
                totals[window]["source_or_matcher_lower_bound_cycles"] += max(
                    per_window_tile_cycles[window], matcher_cycles)

    require(canonical["source_issue_cycles"] + 0 > 0,
            "empty canonical record")
    result = {
        "sample_id": record["sample_id"],
        "operator": record["operator"],
        "local_zero_source_issue_cycles": local_zero_cycles,
        "canonical_m53_parent_source_issue_cycles": canonical["source_issue_cycles"],
        "canonical_m53_parent_logical_updates": canonical["logical_updates"],
        "windows": {},
    }
    for window in WINDOWS:
        item = totals[window]
        require(item["signed_add_updates"] + item["signed_subtract_updates"] ==
                item["logical_updates"], "signed total mismatch")
        result["windows"][str(window)] = dict(
            (key, dict(value) if key == "origin" else value)
            for key, value in item.items())
    return result


def aggregate(records):
    per_sample = []
    for sample_id in range(10):
        selected = [row for row in records if row["sample_id"] == sample_id]
        require(len(selected) == 4, "M69 sample population drift")
        row = {
            "sample_id": sample_id,
            "local_zero_source_issue_cycles": sum(
                item["local_zero_source_issue_cycles"] for item in selected),
            "canonical_m53_parent_source_issue_cycles": sum(
                item["canonical_m53_parent_source_issue_cycles"] for item in selected),
            "windows": {},
        }
        for window in WINDOWS:
            key = str(window)
            fields = ("source_issue_cycles", "logical_updates",
                      "signed_add_updates", "signed_subtract_updates",
                      "exact_copy_tiles", "zero_parent_tiles",
                      "nonzero_parent_tiles", "matcher_query_rows",
                      "matcher_nominal_cycles",
                      "source_or_matcher_lower_bound_cycles")
            row["windows"][key] = dict(
                (field, sum(item["windows"][key][field] for item in selected))
                for field in fields)
        per_sample.append(row)
    aggregate_row = {
        "local_zero_source_issue_cycles": sum(
            row["local_zero_source_issue_cycles"] for row in per_sample),
        "canonical_m53_parent_source_issue_cycles": sum(
            row["canonical_m53_parent_source_issue_cycles"] for row in per_sample),
        "windows": {},
    }
    for window in WINDOWS:
        key = str(window)
        source = sum(row["windows"][key]["source_issue_cycles"] for row in per_sample)
        lower = sum(row["windows"][key]["source_or_matcher_lower_bound_cycles"]
                    for row in per_sample)
        aggregate_row["windows"][key] = {
            "source_issue_cycles": source,
            "source_or_matcher_lower_bound_cycles": lower,
            "local_zero_over_source_issue_speedup": (
                aggregate_row["local_zero_source_issue_cycles"] / source),
            "canonical_m53_over_source_issue_speedup": (
                aggregate_row["canonical_m53_parent_source_issue_cycles"] / source),
            "local_zero_over_source_or_matcher_lower_bound": (
                aggregate_row["local_zero_source_issue_cycles"] / lower),
            "logical_updates": sum(
                row["windows"][key]["logical_updates"] for row in per_sample),
            "signed_add_updates": sum(
                row["windows"][key]["signed_add_updates"] for row in per_sample),
            "signed_subtract_updates": sum(
                row["windows"][key]["signed_subtract_updates"] for row in per_sample),
        }
    return per_sample, aggregate_row


def build(output):
    for name, path in (("m43", M43_PATH), ("manifest", MANIFEST_PATH),
                       ("m53", M53_PATH)):
        require(path.is_file() and sha256(path) == EXPECTED_SHA256[name],
                "M69 input SHA drift: {}".format(name))
    require(not output.exists(), "refusing M69 result overwrite")
    manifest = strict_json(MANIFEST_PATH)
    m53 = strict_json(M53_PATH)
    require(len(manifest["records"]) == 40, "M69 record population drift")
    canonical = [item for item in m53["configuration_summaries"]
                 if item["name"] == "K4_CTX16_TEMPORAL"]
    require(len(canonical) == 1, "M69 canonical M53 config missing")
    m43 = load_m43()
    records = []
    for index, record in enumerate(manifest["records"]):
        records.append(analyze_record(m43, MANIFEST_PATH.parent, record))
        print("[M69] {}/40 sample={} operator={}".format(
            index + 1, record["sample_id"], record["operator"]), flush=True)
    per_sample, summary = aggregate(records)
    require(summary["local_zero_source_issue_cycles"] == 141484880,
            "M69 local-zero source ledger does not reproduce M43")
    require(summary["canonical_m53_parent_source_issue_cycles"] == 113347744,
            "M69 canonical source ledger does not reproduce M53/M43")
    payload = {
        "schema": "m69_signed_window_parent_source_dse_v1",
        "status": "PASS_M69_FROZEN_TRACE_SOURCE_OPPORTUNITY_INTEGRATED_CYCLES_UNADMITTED",
        "identity": {
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "inputs_sha256": EXPECTED_SHA256,
        },
        "architecture": {
            "name": "SIGNED_PREVIOUS_ROW_WINDOW_PARENT",
            "source_banks": m43.ISSUE_WIDTH,
            "sources_per_bank_per_cycle": 1,
            "output_lanes": m43.OUTPUT_LANES,
            "candidate_windows": list(WINDOWS),
            "candidate_scope": "already-produced rows in same timestep and feature tile",
            "canonical_candidates_retained": [
                "zero", "left", "up", "same-position previous timestep"],
            "signed_identity": "W*x = W*p + W*(x-p)",
            "matcher_model": "one query row/cycle plus ceil(log2(window))+3 pipeline",
        },
        "population": {"samples": 10, "operators": 4, "records": 40},
        "summary": summary,
        "per_sample": per_sample,
        "records": records,
        "admission": {
            "exact_signed_arithmetic_identity": True,
            "all10_frozen_trace_source_bank_opportunity": True,
            "dependency_aware_integrated_cycles": False,
            "same_resource_rtl": False,
            "memory_feasible": False,
            "full_network_or_system_speedup": False,
            "date_headline": False,
        },
        "next_gate": (
            "promote one window only after exact dependency-aware K4-C16 schedule, "
            "single-weight-buffer correction, VCS and Synopsys PPA"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M69 windows={} best64_local={:.6f}x best64_m53={:.6f}x".format(
        WINDOWS,
        summary["windows"]["64"]["local_zero_over_source_issue_speedup"],
        summary["windows"]["64"]["canonical_m53_over_source_issue_speedup"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
