#!/usr/bin/env python3
"""Correct M344's duplicated pattern DMA without mutating the sealed result.

This overlay deliberately reuses the frozen M344 trace/work replay.  It changes
only output-tile DMA service: the q-entry pattern table is loaded once in the
matcher/packer pre-stage, while each output tile transfers weight and the
selectively used PWP vectors.  The resulting numbers remain analytical
recurrences, not finite-context executable cycles.
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
    spec = importlib.util.spec_from_file_location("m351_frozen_m344", str(path))
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
    require(not args.output_dir.exists(), "refusing M351 output overwrite")
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m351_m344_pattern_dma_correction_overlay_contract_v1",
            "M351 contract schema drift")
    require(contract.get("status") == "FROZEN_BEFORE_M351_EXECUTION",
            "M351 contract is not frozen")
    root = args.contract.resolve().parents[1]
    for name, identity in contract["inputs"].items():
        path = root / identity["path"]
        require(path.is_file(), "missing input {}: {}".format(name, path))
        require(sha256(path) == identity["sha256"],
                "SHA drift for " + name)

    m344_path = root / contract["inputs"]["m344_analyzer"]["path"]
    m344 = load_module(m344_path)

    # M344 uses candidate_tile_bytes both for physical capacity and DMA.  Keep
    # its capacity function intact, but remove the already-resident pattern
    # table from each output-tile DMA transaction.
    def corrected_tile_load_cycles(phase, q, output_tile, model):
        payload = (
            phase["used_pwp_patterns"] *
            model["pwp_vector_bytes_per_output_block"] * output_tile +
            model["partition_bits"] * model["weight_vector_bytes"] *
            output_tile)
        return int(math.ceil(payload / float(model["dram_bytes_per_cycle"])))

    m344.candidate_tile_load_cycles = corrected_tile_load_cycles

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
    corrected_rows = []
    for row in payload.pop("cycle_bounds"):
        corrected = dict(row)
        corrected["analytical_serial_first_tile_cycles"] = corrected.pop(
            "strict_first_tile_serial_cycles")
        corrected["analytical_serial_first_tile_speedup"] = corrected.pop(
            "strict_speedup_vs_bit_sparse")
        corrected["analytical_last_first_overlap_cycles"] = corrected.pop(
            "last_tile_first_tile_overlap_cycles")
        corrected["analytical_last_first_overlap_speedup"] = corrected.pop(
            "overlap_speedup_vs_bit_sparse")
        corrected["analytical_recurrence_only"] = True
        corrected["cycle_admitted"] = False
        corrected_rows.append(corrected)

    descriptor_bytes = max(
        row["descriptor_sram_bytes_two_contexts"] for row in corrected_rows)
    payload["schema"] = "m351_m344_pattern_dma_correction_overlay_v1"
    payload["status"] = (
        "PASS_M351_PATTERN_DMA_CORRECTED_ANALYTICAL_RECURRENCE_UNADMITTED")
    payload["correction"] = {
        "sealed_parent_mutated": False,
        "pattern_loads_per_phase": 1,
        "pattern_bytes_removed_from_each_output_tile_dma": True,
        "pattern_bytes_retained_in_each_context_capacity": True,
        "weight_and_selective_pwp_bytes_retained_per_output_tile_dma": True,
        "review_trigger": "M347 P1 duplicated pattern DMA",
    }
    payload["mechanism"]["fixed_tile_cache_bytes"] = payload[
        "mechanism"].pop("fixed_total_cache_bytes")
    payload["mechanism"]["separate_two_context_descriptor_sram_bytes"] = (
        descriptor_bytes)
    payload["mechanism"]["fixed_physical_cache_plus_descriptor_bytes"] = (
        payload["mechanism"]["fixed_tile_cache_bytes"] + descriptor_bytes)
    payload["analytical_recurrences"] = corrected_rows
    payload["admission"]["cycle_bound"] = False
    payload["admission"]["pattern_dma_correction_applied"] = True
    payload["admission"]["finite_queue_executable_cycle"] = False
    payload["admission"]["date_headline"] = False
    payload["claim_boundary"] = (
        "M351 corrects only M344's duplicated pattern DMA charge and preserves "
        "the fixed-cache capacity proof. Both recurrences remain unadmitted: "
        "finite cache states, a single shared DMA server, descriptor ports, "
        "bank conflicts, RTL cycle match, area normalization, energy, system "
        "speedup and DATE headline are absent.")

    inherited_path.unlink()
    output_path = (args.output_dir /
                   "m351_m344_pattern_dma_correction_overlay_r1.json")
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    selected = next(
        row for row in corrected_rows
        if row["q_capacity"] == 128 and row["output_block_tile"] == 1 and
        row["port"] == "SHARED96" and
        row["matcher_architecture"] == "SERIAL16_II1")
    print("M351_PASS q128_o1_shared96_serial16_analytical={:.6f}x "
          "physical={}B cycle_admitted=false".format(
              selected["analytical_serial_first_tile_speedup"],
              payload["mechanism"][
                  "fixed_physical_cache_plus_descriptor_bytes"]),
          flush=True)


if __name__ == "__main__":
    main()
