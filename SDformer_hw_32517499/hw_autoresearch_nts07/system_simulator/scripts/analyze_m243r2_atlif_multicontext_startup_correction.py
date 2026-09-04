#!/usr/bin/env python3
"""Fail-closed M243 correction for 45 independently configured T10 contexts."""

from __future__ import division

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def exact_fraction(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    contract = load_json(args.contract)
    require(contract.get("schema") ==
            "m243r2_atlif_multicontext_startup_correction_contract_v1",
            "contract schema drift")
    root = args.contract.resolve().parents[1]
    identities = {}
    loaded = {}
    resolved = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file(), "missing input: {}".format(path))
        observed = sha256(path)
        require(observed == spec["sha256"],
                "SHA drift for {}: {}".format(name, observed))
        identities[name] = {"path": spec["path"], "sha256": observed}
        resolved[name] = path
        if path.suffix == ".json":
            loaded[name] = load_json(path)

    old = loaded["m243_r1_result"]
    review = loaded["m244_independent_review"]
    review_math = loaded["m244_independent_recompute"]
    m38 = loaded["m38_integration_ledger"]
    model = contract["finite_context_model"]

    require(old["finite_population_cycles"]["decoupled_cycles_including_startup"] ==
            contract["supersedes"]["old_decoupled_cycles"],
            "old M243 cycle identity drift")
    require(review["status"] ==
            "NOGO_M243_EXACT_POPULATION_ADMISSION_GO_CORRECTED_CONDITIONAL_DIRECTION" and
            review["severity_counts"]["P0"] == 1,
            "M244 verdict drift")
    require(review_math["corrected_multi_context_recompute"]["startup_domains_per_inference"] ==
            model["expected_t10_contexts_per_sample"],
            "M244 startup-domain identity drift")

    candidates = {item["name"]: item
                  for item in m38["integrated_theory_ledger"]["candidates"]}
    require(candidates["m37_csd4_parallel_normalized_integration_target"]
            ["conditional_n_tile_cycle_equation"] == "5 + 5*N",
            "M38 one-context recurrence drift")
    require("45 T10 modules" in m38["line_scope"]["motion_h67"],
            "M38 context-count scope drift")

    manifest_text = resolved["raw_trace_manifest"].read_text(encoding="utf-8")
    raw_sha = identities["raw_execution_trace"]["sha256"]
    require(raw_sha in manifest_text and "execution_trace.csv" in manifest_text,
            "raw trace is not directly bound by raw manifest")

    with resolved["raw_execution_trace"].open("r", encoding="utf-8",
                                               newline="") as handle:
        rows = list(csv.DictReader(handle))
    t10_rows = [row for row in rows
                if row["kind"] == "atlif" and
                int(row["temporal_steps"] or 0) == model["temporal_steps"]]
    by_sample = defaultdict(list)
    for row in t10_rows:
        by_sample[row["sample_id"]].append(row)
    require(len(by_sample) == model["expected_samples"],
            "raw sample count drift")

    sample_audit = []
    reference_names = None
    for sample_id in sorted(by_sample, key=lambda value: int(value)):
        sample_rows = by_sample[sample_id]
        names = [row["name"] for row in sample_rows]
        counts = Counter(names)
        name_set = set(names)
        require(len(name_set) == model["expected_t10_contexts_per_sample"],
                "T10 context count drift for sample {}".format(sample_id))
        require(all(value == 1 for value in counts.values()),
                "duplicate T10 context in sample {}".format(sample_id))
        if reference_names is None:
            reference_names = name_set
        require(name_set == reference_names,
                "T10 context-name set drift for sample {}".format(sample_id))

        tiles = 0
        for row in sample_rows:
            output_elements = int(row["output_elements"])
            denominator = (model["temporal_steps"] *
                           model["lanes_per_factor_tile"])
            require(output_elements % denominator == 0,
                    "nonintegral factor-tile count for {}".format(row["name"]))
            tiles += output_elements // denominator
        require(tiles == model["expected_t10_tiles_per_sample"],
                "T10 tile total drift for sample {}".format(sample_id))
        sample_audit.append({
            "sample_id": int(sample_id),
            "sample_key": sample_rows[0]["sample_key"],
            "t10_contexts": len(name_set),
            "t10_tiles": tiles
        })

    n_tiles = model["expected_t10_tiles_per_sample"]
    contexts = model["expected_t10_contexts_per_sample"]
    serial_cycles = model["serial_cycles_per_tile"] * n_tiles
    fill_cycles = model["candidate_fill_cycles_per_context"] * contexts
    candidate_cycles = model["candidate_cycles_per_tile"] * n_tiles + fill_cycles
    speedup = Fraction(serial_cycles, candidate_cycles)
    fixed_cycles = model["fixed_compute_reference_cycles"]
    conditional_total = fixed_cycles - serial_cycles + candidate_cycles
    conditional_speedup = Fraction(fixed_cycles, conditional_total)

    independent = review_math["corrected_multi_context_recompute"]
    require(candidate_cycles == independent["candidate_cycles"] and
            serial_cycles == independent["serial_cycles"] and
            speedup.numerator == independent["speedup_exact"]["numerator"] and
            speedup.denominator == independent["speedup_exact"]["denominator"],
            "independent M244 corrected arithmetic mismatch")

    result = {
        "schema": "m243r2_atlif_multicontext_startup_correction_v1",
        "status": "PASS_CORRECTED_45_CONTEXT_CONDITIONAL_MODULE_CYCLES",
        "identity": identities,
        "raw_population_audit": {
            "execution_records": len(rows),
            "t10_records": len(t10_rows),
            "profile_samples": len(by_sample),
            "t10_contexts_per_sample": contexts,
            "t10_context_name_set_identical_across_samples": True,
            "t10_tiles_per_sample": n_tiles,
            "samples": sample_audit
        },
        "revocation": {
            "old_m243_r1_exact_values_admitted": False,
            "old_candidate_cycles": contract["supersedes"]["old_decoupled_cycles"],
            "old_module_speedup": contract["supersedes"]["old_module_speedup"],
            "omitted_context_fills": contexts - 1,
            "omitted_cycles": model["candidate_fill_cycles_per_context"] *
                              (contexts - 1),
            "reason": contract["supersedes"]["reason"]
        },
        "corrected_conditional_module_cycles": {
            "formula": "sum_i(5 + 5*N_i) = 5*N + 5*S",
            "startup_domains_S": contexts,
            "serial_cycles": serial_cycles,
            "candidate_steady_cycles": model["candidate_cycles_per_tile"] * n_tiles,
            "candidate_fill_cycles": fill_cycles,
            "candidate_cycles": candidate_cycles,
            "cycles_saved": serial_cycles - candidate_cycles,
            "module_speedup_exact": exact_fraction(speedup),
            "module_speedup": float(speedup),
            "asymptotic_speedup": 2.0
        },
        "conditional_fixed_compute_context_only": {
            "fixed_compute_reference_cycles": fixed_cycles,
            "conditional_total_after": conditional_total,
            "speedup_exact": exact_fraction(conditional_speedup),
            "speedup": float(conditional_speedup),
            "system_speedup_admitted": False
        },
        "architecture_scope": {
            "candidate_innovation": "phase-decoupled multiplierless CSD4 T10 reconstruction",
            "m37_r10_frozen_rtl_sha256": identities["frozen_m37_r10_sidecar_rtl"]["sha256"],
            "m37_is_standalone_stage2_sidecar_only": True,
            "matched_full_candidate_area_available": False,
            "configuration_load_and_result_backpressure_in_cycle_formula": False
        },
        "admission": contract["claim_boundary"],
        "claim_boundary": "Corrected finite-population conditional no-stall ATLIF module cycle model for one inference and 45 independently configured T10 contexts. The near-2x ratio is not integrated RTL throughput, matched throughput per area, trained accuracy, energy, system speedup, paper PPA or a headline claim."
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "m243r2_atlif_multicontext_startup_correction_r1.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("M243R2_PASS contexts={} candidate_cycles={} module_speedup={:.9f} conditional_fixed={:.9f}".format(
        contexts, candidate_cycles, float(speedup), float(conditional_speedup)))


if __name__ == "__main__":
    main()
