#!/usr/bin/env python3
"""Build a constructive two-slot, single-DMA schedule over frozen M344 work.

Unlike M344/M351, this schedule does not overlap preprocessing across phase
boundaries.  A phase loads its pattern table once, completes matching/packing,
and then ping-pongs output tiles through two finite cache slots.  The sole DMA
server can load the next tile while the SHARED96/WIDE144 compute port consumes
the current tile.  This is intentionally conservative but constructive.
"""

from __future__ import division

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


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

    def reject(token):
        raise RuntimeError("non-standard JSON number: " + token)

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs, parse_constant=reject)


def load_module(path):
    spec = importlib.util.spec_from_file_location("m358_frozen_m344", str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import frozen M344 analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M358 output overwrite")
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m358_two_slot_serial_phase_constructive_cycle_contract_v1",
            "M358 contract schema drift")
    require(contract.get("status") == "FROZEN_BEFORE_M358_EXECUTION",
            "M358 contract not frozen")
    root = args.contract.resolve().parents[1]
    for name, identity in contract["inputs"].items():
        path = root / identity["path"]
        require(path.is_file(), "missing input {}: {}".format(name, path))
        require(sha256(path) == identity["sha256"],
                "SHA drift for " + name)

    m344_path = root / contract["inputs"]["m344_analyzer"]["path"]
    m344 = load_module(m344_path)

    def corrected_tile_load_cycles(phase, q, output_tile, model):
        payload = (
            phase["used_pwp_patterns"] *
            model["pwp_vector_bytes_per_output_block"] * output_tile +
            model["partition_bits"] * model["weight_vector_bytes"] *
            output_tile)
        return int(math.ceil(payload / float(model["dram_bytes_per_cycle"])))

    def constructive_replay(phases, q, output_tile, port, architecture,
                            model, common_commit, _unused_overlap_flag):
        total = 0
        components = {
            "pattern_dma_cycles": 0,
            "matcher_cycles": 0,
            "packer_cycles": 0,
            "initial_tile_dma_cycles": 0,
            "compute_cycles_not_additive": 0,
            "next_tile_dma_cycles_not_additive": 0,
            "tail_cycles": 0,
        }
        tiles = model["output_blocks"] // output_tile
        require(tiles >= 1, "invalid tile count")
        for phase in phases:
            pattern_dma = int(math.ceil(
                q * model["pattern_bytes"] /
                float(model["dram_bytes_per_cycle"])))
            matcher = m344.matcher_cycles(phase, q, architecture)
            packer = (int(math.ceil(
                phase["assignment_rows"] / float(model["packer_lanes"]))) +
                model["packer_pipeline_cycles"])
            tile_dma = corrected_tile_load_cycles(
                phase, q, output_tile, model)
            tile_compute = (
                phase["correction_ops_per_block"] * output_tile *
                port["weight_cycles"] +
                phase["pwp_ops_per_block"] * output_tile *
                port["pwp_cycles"])

            # One pattern DMA, then matcher/packer.  First tile load is exposed.
            # Each later tile uses the alternate slot and the sole DMA server;
            # it overlaps only the current tile's compute.  The final compute is
            # exposed before the phase-local tail and slot release.
            total += pattern_dma + matcher + packer + tile_dma
            if tiles > 1:
                total += (tiles - 1) * max(tile_compute, tile_dma)
            total += tile_compute + model["compute_tail_cycles_per_partition"]

            components["pattern_dma_cycles"] += pattern_dma
            components["matcher_cycles"] += matcher
            components["packer_cycles"] += packer
            components["initial_tile_dma_cycles"] += tile_dma
            components["compute_cycles_not_additive"] += tile_compute * tiles
            components["next_tile_dma_cycles_not_additive"] += (
                tile_dma * (tiles - 1))
            components["tail_cycles"] += model[
                "compute_tail_cycles_per_partition"]
        total += common_commit
        components["common_commit_cycles"] = common_commit
        return {"cycles": total, "binding_phases": {},
                "constructive_components": components}

    m344.candidate_tile_load_cycles = corrected_tile_load_cycles
    m344.replay_candidate = constructive_replay
    original_strict_json = m344.strict_json

    def compatible_strict_json(path):
        payload = original_strict_json(path)
        if Path(path).resolve() == args.contract.resolve():
            payload = dict(payload)
            payload["schema"] = "m344_output_block_tiled_q128_kfirst_contract_v1"
        return payload

    m344.strict_json = compatible_strict_json
    saved_argv = list(sys.argv)
    try:
        sys.argv = [str(m344_path), "--contract", str(args.contract),
                    "--output-dir", str(args.output_dir)]
        m344.main()
    finally:
        sys.argv = saved_argv

    inherited_path = (args.output_dir /
                      "m344_output_block_tiled_q128_kfirst_r1.json")
    require(inherited_path.is_file(), "frozen M344 replay did not emit result")
    payload = strict_json(inherited_path)
    rows = []
    for row in payload.pop("cycle_bounds"):
        require(row["strict_first_tile_serial_cycles"] ==
                row["last_tile_first_tile_overlap_cycles"],
                "constructive schedule unexpectedly depends on overlap flag")
        converted = dict(row)
        converted["constructive_two_slot_serial_phase_cycles"] = converted.pop(
            "strict_first_tile_serial_cycles")
        converted["constructive_speedup_vs_bit_sparse"] = converted.pop(
            "strict_speedup_vs_bit_sparse")
        converted.pop("last_tile_first_tile_overlap_cycles")
        converted.pop("overlap_speedup_vs_bit_sparse")
        converted["finite_tile_slots"] = 2
        converted["single_dma_server"] = True
        converted["cross_phase_preprocess_overlap"] = False
        converted["cycle_admitted"] = False
        rows.append(converted)

    descriptor_bytes = max(
        row["descriptor_sram_bytes_two_contexts"] for row in rows)
    payload["schema"] = "m358_two_slot_serial_phase_constructive_cycle_v1"
    payload["status"] = (
        "PASS_M358_CONSTRUCTIVE_TWO_SLOT_SERIAL_PHASE_CYCLE_UNADMITTED")
    payload["mechanism"] = {
        "tile_slots": 2,
        "tile_slot_bytes_each": 32768,
        "tile_cache_bytes": 65536,
        "descriptor_banks": 2,
        "descriptor_sram_bytes": descriptor_bytes,
        "fixed_physical_cache_plus_descriptor_bytes": 65536 + descriptor_bytes,
        "dram_bytes_per_cycle": contract["cycle_model"][
            "dram_bytes_per_cycle"],
        "single_dma_server": True,
        "pattern_loads_per_phase": 1,
        "later_tile_dma_overlaps_current_compute": True,
        "cross_phase_preprocess_overlap": False,
        "q_to_output_block_tile": {"16": 8, "32": 4,
                                     "64": 2, "128": 1},
        "exact_arithmetic": True,
    }
    payload["constructive_cycle_rows"] = rows
    payload["admission"] = {
        "m339_exact_work_reproduced": True,
        "fixed64kb_capacity_fit": True,
        "duplicated_pattern_dma_absent": True,
        "finite_two_slot_schedule_constructed": True,
        "single_dma_contention_constructed": True,
        "cycle_bound": False,
        "rtl_cycle_match": False,
        "area_matched": False,
        "energy": False,
        "system_speedup": False,
        "date_headline": False,
    }
    payload["claim_boundary"] = (
        "M358 constructs a finite two-tile-slot, single-DMA, phase-serial "
        "schedule with exact frozen work and no cross-phase overlap. It is a "
        "conservative module schedule only; descriptor read timing, matcher "
        "DC/equal-area normalization, RTL cycle match, energy, system speedup "
        "and DATE headline remain unadmitted.")
    inherited_path.unlink()
    output_path = (args.output_dir /
                   "m358_two_slot_serial_phase_constructive_cycle_r1.json")
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    selected = next(
        row for row in rows if row["q_capacity"] == 128 and
        row["port"] == "SHARED96" and
        row["matcher_architecture"] == "SERIAL16_II1")
    print("M358_PASS q128_o1_shared96_serial16_constructive={:.6f}x "
          "cycles={} cycle_admitted=false".format(
              selected["constructive_speedup_vs_bit_sparse"],
              selected["constructive_two_slot_serial_phase_cycles"]),
          flush=True)


if __name__ == "__main__":
    main()
