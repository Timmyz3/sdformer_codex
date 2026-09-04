#!/usr/bin/env python3
"""Independent M427 recomputation of the frozen M426r3 replay.

This program intentionally does not import or execute the M401, M418, M419,
or M426 analyzers.  It decodes the fixed-width M410R2 row transport and the
M401 phase ledger directly, derives the three candidate recurrences, and then
compares every phase/component/timestamp against the sealed M426r3 CSV.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np


ROWS_PER_PHASE = 3000
PHASES = 17280
PHASES_PER_SAMPLE = 1728
PARTITIONS = 432
OUTPUT_BLOCKS_PER_TILE = 4
VARIANTS = (
    "m401_serial_low8_high4",
    "dualbank_parallel_low8_high4",
    "dualbank_seed_first_correction_fused",
)
M426_FIELDS = (
    "variant", "phase_global_index", "sample", "operator", "partition",
    "record_start", "record_end", "phase_cycles", "active_rows",
    "pwp_rows", "exact_pwp_rows", "positive_residual_pwp_rows",
    "correction_ops_per_block", "narrow_tile0", "narrow_tile1", "work0",
    "work1", "active_compute_work", "matcher_cycles", "tile_dma_cycles",
    "tile1_dma_exposed_cycles", "replay0_cycles", "replay1_cycles",
    "tail_cycles", "commit_cycles",
)


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
        raise RuntimeError("non-standard JSON token: " + token)

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def exact_int(value, label):
    require(value is not None and str(value).lstrip("-").isdigit(),
            "non-integer {}={}".format(label, value))
    return int(value)


def load_ledger(path):
    required = {
        "sample", "operator", "partition", "active_rows", "eligible_rows",
        "pwp_rows", "fallback_rows", "used_pwp_patterns",
        "used_center_runs", "narrow_tile0", "narrow_tile1", "early_matcher",
        "eligible_exact_q32",
    }
    result = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None and
                required.issubset(set(reader.fieldnames)),
                "M401 phase-ledger schema drift")
        for index, row in enumerate(reader):
            item = {key: exact_int(value, "ledger." + key)
                    for key, value in row.items()}
            within = index % PHASES_PER_SAMPLE
            expected = (index // PHASES_PER_SAMPLE,
                        within // PARTITIONS, within % PARTITIONS)
            observed = (item["sample"], item["operator"], item["partition"])
            require(observed == expected,
                    "ledger ordering drift at phase {}".format(index))
            result.append(item)
    require(len(result) == PHASES, "M401 phase-ledger extent drift")
    return result


def decode_phase(block, nibble_lut):
    require(len(block) == ROWS_PER_PHASE * 9,
            "M410R2 transport truncation")
    raw = np.frombuffer(block, dtype=np.uint8).reshape(ROWS_PER_PHASE, 9)
    require(bool(np.all(raw[:, 8] == 10)), "M410R2 newline/layout drift")
    digits = nibble_lut[raw[:, :8]]
    require(not bool(np.any(digits == 255)), "M410R2 non-hex digit")
    words = np.zeros(ROWS_PER_PHASE, dtype=np.uint32)
    for column in range(8):
        words = (words << np.uint32(4)) | digits[:, column].astype(np.uint32)
    require(not bool(np.any(words >> np.uint32(29))),
            "M410R2 reserved bits nonzero")
    return words


def population_from_words(words, pop16):
    originals = (words & np.uint32(0xffff)).astype(np.uint16)
    population = pop16[originals].astype(np.int64)
    distance = ((words >> np.uint32(21)) & np.uint32(0x1f)).astype(np.int64)
    use_pwp = ((words >> np.uint32(26)) & np.uint32(1)).astype(np.int64)
    pass1 = ((words >> np.uint32(27)) & np.uint32(1)).astype(np.int64)
    early = ((words >> np.uint32(28)) & np.uint32(1)).astype(np.int64)
    eligible_mask = population >= 2
    require(bool(np.all(use_pwp == ((distance + 1) < population))),
            "strict PWP predicate mismatch")
    require(not bool(np.any((pass1 != 0) & (early != 0))),
            "pass1/early overlap")
    require(int(pass1.sum()) + int(early.sum()) ==
            int(np.count_nonzero(eligible_mask)),
            "eligible/pass partition mismatch")
    pwp_rows = int(use_pwp.sum())
    exact_rows = int(np.count_nonzero((use_pwp != 0) & (distance == 0)))
    positive_rows = pwp_rows - exact_rows
    residual_corrections = int(
        np.where(use_pwp != 0, distance, np.int64(0)).sum())
    require(residual_corrections >= positive_rows,
            "positive residual cannot supply one first correction")
    return {
        "source_rows": ROWS_PER_PHASE,
        "zero_rows": int(np.count_nonzero(population == 0)),
        "pop1_rows": int(np.count_nonzero(population == 1)),
        "active_rows": int(np.count_nonzero(population)),
        "eligible_rows": int(np.count_nonzero(eligible_mask)),
        "pwp_rows": pwp_rows,
        "exact_pwp_rows": exact_rows,
        "positive_residual_pwp_rows": positive_rows,
        "fallback_rows": int(np.count_nonzero(population)) - pwp_rows,
        "correction_ops_per_block": int(
            np.where(use_pwp != 0, distance, population).sum()),
        "residual_correction_ops_per_block": residual_corrections,
        "bit_sparse_ops_per_block": int(population.sum()),
        "pass1_tasks": int(pass1.sum()),
        "early_matcher": ROWS_PER_PHASE + int(pass1.sum()) + 2,
    }


def candidate_phase(index, start, name, pop, phase):
    correction_tile = OUTPUT_BLOCKS_PER_TILE * pop["correction_ops_per_block"]
    if name == VARIANTS[0]:
        work0 = correction_tile + 8 * pop["pwp_rows"] - phase["narrow_tile0"]
        work1 = correction_tile + 8 * pop["pwp_rows"] - phase["narrow_tile1"]
    elif name == VARIANTS[1]:
        work0 = correction_tile + 4 * pop["pwp_rows"]
        work1 = correction_tile + 4 * pop["pwp_rows"]
    else:
        # d>0: d correction cycles, with PWP replacing old accumulator on
        # only the first one.  d=0: retain exactly one PWP seed cycle.
        work0 = correction_tile + 4 * pop["exact_pwp_rows"]
        work1 = correction_tile + 4 * pop["exact_pwp_rows"]
    require(work0 >= pop["active_rows"] and work1 >= pop["active_rows"],
            "descriptor issue underflow")
    matcher = pop["early_matcher"]
    time = start + 3 + 32 + matcher + 1
    tile_dma = exposed = replay0 = replay1 = 0
    if pop["active_rows"]:
        tile_bytes = 6144 + 640 * phase["used_pwp_patterns"]
        require(96 + tile_bytes <= 32768 and tile_bytes % 32 == 0,
                "tile slot/alignment drift")
        tile_dma = tile_bytes // 32 + 32 * (1 + phase["used_center_runs"])
        replay0 = work0 + 8
        replay1 = work1 + 8
        exposed = max(0, tile_dma - replay0)
        time += tile_dma + replay0 + exposed + replay1
    else:
        work0 = work1 = 0
    commit = 96000 if index % PHASES_PER_SAMPLE == PHASES_PER_SAMPLE - 1 else 0
    time += 2 + commit
    return time, {
        "variant": name,
        "phase_global_index": index,
        "sample": phase["sample"],
        "operator": phase["operator"],
        "partition": phase["partition"],
        "record_start": start,
        "record_end": time,
        "phase_cycles": time - start,
        "active_rows": pop["active_rows"],
        "pwp_rows": pop["pwp_rows"],
        "exact_pwp_rows": pop["exact_pwp_rows"],
        "positive_residual_pwp_rows": pop["positive_residual_pwp_rows"],
        "correction_ops_per_block": pop["correction_ops_per_block"],
        "narrow_tile0": phase["narrow_tile0"],
        "narrow_tile1": phase["narrow_tile1"],
        "work0": work0,
        "work1": work1,
        "active_compute_work": work0 + work1,
        "matcher_cycles": matcher,
        "tile_dma_cycles": tile_dma,
        "tile1_dma_exposed_cycles": exposed,
        "replay0_cycles": replay0,
        "replay1_cycles": replay1,
        "tail_cycles": 2,
        "commit_cycles": commit,
    }


def baseline_phase(start, compute, first, last):
    preprocess = max(ROWS_PER_PHASE + 5, 12288 // 32 + 32)
    initial = preprocess if first else 0
    exposed = 0 if last else max(0, preprocess - compute)
    end = start + initial + compute + exposed + 2 + (96000 if last else 0)
    return end, {"initial": initial, "compute": compute,
                 "exposed": exposed, "tail": 2,
                 "commit": 96000 if last else 0}


def signed12(raw):
    raw &= 0xfff
    return raw - 0x1000 if raw & 0x800 else raw


def prove_wide12():
    mismatch = narrow_mismatch = 0
    narrow_values = wide_values = 0
    for raw in range(4096):
        low = raw & 0xff
        high = (raw >> 8) & 0xf
        exact = signed12(raw)
        serial = signed12(low) + signed12(high << 8)
        direct = signed12((high << 8) | low)
        if serial != exact or direct != exact:
            mismatch += 1
        narrow = exact >= -128 and exact <= 127
        sign_extended_low = low - 256 if low & 0x80 else low
        if narrow:
            narrow_values += 1
            if sign_extended_low != exact:
                narrow_mismatch += 1
        else:
            wide_values += 1
    require(mismatch == 0 and narrow_mismatch == 0 and
            narrow_values == 256 and wide_values == 3840,
            "signed-12 exhaustive reconstruction failure")
    return {
        "signed12_values_checked": 4096,
        "wide_values_checked": wide_values,
        "narrow_values_checked": narrow_values,
        "serial_low_plus_signed_high_vs_direct_mismatches": mismatch,
        "narrow_sign_extension_mismatches": narrow_mismatch,
        "proof_scope": "scalar lane encoding; 96 lanes follow independently",
    }


def audit_recovery(paths):
    result = {}
    expected_fragments = {
        "r1_log": (
            "phase_population() takes 1 positional argument but 2 were given",
            "[M426] replayed",
        ),
        "r2_log": (
            "ufunc 'bitwise_and' not supported for the input types",
            "[M426] replayed",
        ),
    }
    for name, (failure, progress) in expected_fragments.items():
        text = paths[name].read_text(encoding="utf-8")
        directory_files = sorted(item.name for item in paths[name].parent.iterdir())
        require(failure in text and progress not in text,
                name + " is not a pre-phase failure")
        require(directory_files == ["RUN_FAILED_OR_INCOMPLETE.log",
                                    "RUN_FAILED_OR_INCOMPLETE.txt"],
                name + " leaked additional candidate artifacts")
        marker = paths[name.replace("log", "marker")].read_text(
            encoding="utf-8")
        require(marker ==
                "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=1\n",
                name + " fail-closed marker drift")
        result[name[:2]] = {
            "completed_phase_count": 0,
            "candidate_artifact_count": 0,
            "fail_closed_marker": True,
            "failure_signature": failure,
            "directory_files": directory_files,
        }
    r2_wrapper = paths["r2_wrapper"].read_text(encoding="utf-8")
    r3_wrapper = paths["r3_wrapper"].read_text(encoding="utf-8")
    require("return ORIGINAL_PHASE_POPULATION(words)" in r2_wrapper and
            "ORIGINAL_DECODE_WORDS(raw_block, nibble_lut)" in r3_wrapper and
            "ORIGINAL_PHASE_POPULATION(" in r3_wrapper,
            "recovery wrapper semantics drift")
    require("candidate_feedback_available_before_recovery\": false" in
            paths["r3_contract"].read_text(encoding="utf-8"),
            "r3 no-feedback declaration missing")
    result["r3"] = {
        "decode_only_adapter": True,
        "original_analyzer_sha_preserved": True,
        "predecessors_exposed_candidate_artifacts": False,
        "candidate_feedback_available_before_recovery": False,
        "evidence_limit": (
            "fail-closed directories and frozen pre-execution recovery "
            "contract; filesystem mtimes are not treated as cryptographic proof"
        ),
    }
    return result


def compare_m426_csv(path, expected_by_phase):
    mismatches = Counter()
    rows = 0
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == M426_FIELDS,
                "M426 CSV schema/order drift")
        for row in reader:
            phase = exact_int(row["phase_global_index"], "m426.phase")
            variant = row["variant"]
            require(0 <= phase < PHASES and variant in VARIANTS,
                    "M426 CSV key drift")
            expected = expected_by_phase[phase][variant]
            for key in M426_FIELDS:
                observed = row[key] if key == "variant" else exact_int(
                    row[key], "m426." + key)
                if observed != expected[key]:
                    mismatches[key] += 1
            rows += 1
    require(rows == PHASES * len(VARIANTS), "M426 CSV extent drift")
    require(not mismatches, "M426 CSV mismatches: " + str(dict(mismatches)))
    return {"rows_checked": rows, "phases_checked": PHASES,
            "field_comparisons": rows * len(M426_FIELDS),
            "mismatch_by_field": dict(mismatches)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M427 overwrite")
    contract = strict_json(args.contract)
    require(contract["schema"] ==
            "m427_m426r3_independent_hammer_contract_v1" and
            contract["status_before_execution"] ==
            "FROZEN_BEFORE_INDEPENDENT_RECOMPUTE",
            "M427 contract state drift")
    hw = args.contract.resolve().parents[1]
    require(sha256(Path(__file__).resolve()) ==
            contract["recompute_script"]["sha256"],
            "M427 recompute script SHA drift")
    paths = {}
    for name, identity in contract["inputs"].items():
        path = hw / identity["path"]
        require(path.is_file() and not path.is_symlink(),
                "missing/symlink input: " + name)
        require(sha256(path) == identity["sha256"],
                "M427 input SHA drift: " + name)
        paths[name] = path

    ledger = load_ledger(paths["m401_phase_ledger"])
    pop16 = np.asarray([value.bit_count() for value in range(1 << 16)],
                       dtype=np.uint8)
    nibble_lut = np.full(256, 255, dtype=np.uint8)
    for value, digit in zip(b"0123456789abcdef", range(16)):
        nibble_lut[value] = digit
    for value, digit in zip(b"ABCDEF", range(10, 16)):
        nibble_lut[value] = digit

    totals = Counter()
    component_totals = {name: Counter() for name in VARIANTS}
    times = {name: 0 for name in VARIANTS}
    dense_time = zero_time = 0
    expected_by_phase = []
    output_phase_rows = []
    min_zero_compute = None
    max_zero_compute = 0
    with paths["m410_rows"].open("rb") as handle:
        for index, phase in enumerate(ledger):
            words = decode_phase(handle.read(ROWS_PER_PHASE * 9), nibble_lut)
            pop = population_from_words(words, pop16)
            for key in ("active_rows", "eligible_rows", "pwp_rows",
                        "fallback_rows", "early_matcher"):
                require(pop[key] == phase[key],
                        "raw/ledger mismatch phase={} key={}".format(index, key))
            require(pop["exact_pwp_rows"] == phase["eligible_exact_q32"],
                    "exact PWP ledger mismatch phase={}".format(index))
            totals.update(pop)
            first = index % PHASES_PER_SAMPLE == 0
            last = index % PHASES_PER_SAMPLE == PHASES_PER_SAMPLE - 1
            dense_time, dense_parts = baseline_phase(
                dense_time, ROWS_PER_PHASE * 16 * 8, first, last)
            zero_compute = pop["bit_sparse_ops_per_block"] * 8
            min_zero_compute = (zero_compute if min_zero_compute is None else
                                min(min_zero_compute, zero_compute))
            max_zero_compute = max(max_zero_compute, zero_compute)
            zero_time, zero_parts = baseline_phase(
                zero_time, zero_compute, first, last)
            records = {}
            for name in VARIANTS:
                times[name], record = candidate_phase(
                    index, times[name], name, pop, phase)
                records[name] = record
                for key in ("phase_cycles", "active_compute_work",
                            "matcher_cycles", "tile_dma_cycles",
                            "tile1_dma_exposed_cycles", "replay0_cycles",
                            "replay1_cycles", "tail_cycles", "commit_cycles"):
                    component_totals[name][key] += record[key]
            expected_by_phase.append(records)
            output_phase_rows.append({
                "phase_global_index": index,
                "sample": phase["sample"],
                "operator": phase["operator"],
                "partition": phase["partition"],
                "exact_pwp_rows": pop["exact_pwp_rows"],
                "positive_residual_pwp_rows":
                    pop["positive_residual_pwp_rows"],
                "current_start": records[VARIANTS[0]]["record_start"],
                "current_end": records[VARIANTS[0]]["record_end"],
                "current_cycles": records[VARIANTS[0]]["phase_cycles"],
                "dual_start": records[VARIANTS[1]]["record_start"],
                "dual_end": records[VARIANTS[1]]["record_end"],
                "dual_cycles": records[VARIANTS[1]]["phase_cycles"],
                "fused_start": records[VARIANTS[2]]["record_start"],
                "fused_end": records[VARIANTS[2]]["record_end"],
                "fused_cycles": records[VARIANTS[2]]["phase_cycles"],
                "dense_end": dense_time,
                "zero_end": zero_time,
                "zero_compute": zero_parts["compute"],
            })
        require(handle.read(1) == b"", "M410R2 transport trailing bytes")

    expected_totals = {
        "source_rows": 51840000,
        "active_rows": 27305568,
        "eligible_rows": 19789148,
        "pwp_rows": 16971357,
        "exact_pwp_rows": 5350591,
        "positive_residual_pwp_rows": 11620766,
        "fallback_rows": 10334211,
        "correction_ops_per_block": 38690838,
        "bit_sparse_ops_per_block": 92640472,
    }
    for key, expected in expected_totals.items():
        require(totals[key] == expected,
                "population mismatch {} {} != {}".format(
                    key, totals[key], expected))
    require(dense_time == 6636544610 and zero_time == 742148386,
            "independent baseline reproduction drift")
    expected_cycles = dict(zip(VARIANTS, (641790704, 530606660, 437640532)))
    require(times == expected_cycles,
            "candidate cycle reproduction drift: " + str(times))

    wide_blocks = sum(8 * phase["pwp_rows"] - phase["narrow_tile0"] -
                      phase["narrow_tile1"] for phase in ledger)
    coread_saved = times[VARIANTS[0]] - times[VARIANTS[1]]
    fusion_saved = times[VARIANTS[1]] - times[VARIANTS[2]]
    require(wide_blocks == 111184044 and coread_saved == wide_blocks,
            "wide-block/coread conservation drift")
    require(fusion_saved == totals["positive_residual_pwp_rows"] * 8 ==
            92966128,
            "single-fusion conservation drift")
    require(times[VARIANTS[0]] - times[VARIANTS[2]] ==
            coread_saved + fusion_saved == 204150172,
            "total savings conservation drift")

    csv_audit = compare_m426_csv(paths["m426_csv"], expected_by_phase)
    m426 = strict_json(paths["m426_result"])
    require(m426["baselines"]["primary_comparison"] == "strong_zero_elided",
            "M426 primary baseline drift")
    require(m426["baselines"]["strong_zero_elided_cycles"] == zero_time and
            m426["baselines"]["weak_dense16_cycles"] == dense_time,
            "M426 baseline receipt mismatch")
    for name in VARIANTS:
        require(m426["variants"][name]["cycles"] == times[name],
                "M426 variant receipt mismatch: " + name)
        expected_speedup = zero_time / float(times[name])
        require(math.isclose(m426["variants"][name]
                             ["speedup_vs_strong_zero_elided"],
                             expected_speedup, rel_tol=0.0, abs_tol=1e-15),
                "M426 strong speedup mismatch: " + name)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    phase_csv = args.output_dir / "m427_independent_phase_timestamps_r1.csv"
    with phase_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_phase_rows[0]))
        writer.writeheader()
        writer.writerows(output_phase_rows)

    wide12 = prove_wide12()
    recovery = audit_recovery(paths)
    result = {
        "schema": "m427_m426r3_independent_recompute_receipt_v1",
        "status": "PASS_INDEPENDENT_FULL_RECOMPUTE_WITH_RESOURCE_CAVEAT",
        "independence": {
            "imported_or_executed_m401_m418_m419_m426_analyzer": False,
            "direct_inputs": ["M410R2 fixed-width row memh",
                              "M401 ordered phase ledger"],
            "comparison_only_after_independent_derivation":
                "sealed M426r3 JSON and CSV",
        },
        "population": {key: int(totals[key]) for key in expected_totals},
        "baselines": {
            "primary_strong_zero_elided_cycles": zero_time,
            "secondary_weak_dense16_cycles": dense_time,
            "minimum_phase_zero_compute_cycles": min_zero_compute,
            "maximum_phase_zero_compute_cycles": max_zero_compute,
        },
        "variants": {
            name: {
                "cycles": times[name],
                "speedup_vs_primary_strong_zero": zero_time / times[name],
                "speedup_vs_secondary_weak_dense16": dense_time / times[name],
                "components": dict(component_totals[name]),
            } for name in VARIANTS
        },
        "savings_conservation": {
            "wide_blocks": wide_blocks,
            "coread_saved_cycles": coread_saved,
            "positive_residual_pwp_rows":
                totals["positive_residual_pwp_rows"],
            "fusion_saved_cycles": fusion_saved,
            "fusion_equation": "8 * positive_residual_pwp_rows",
            "total_saved_vs_current": coread_saved + fusion_saved,
            "mismatches": 0,
        },
        "single_correction_cycle_semantics": {
            "d_zero_pwp_cycles_per_output_block": 1,
            "d_positive_pwp_cycles_per_output_block": "d",
            "first_d_positive_cycle":
                "PWP is left operand and exactly one signed correction is right operand",
            "later_d_positive_cycles": "one signed correction per cycle",
            "duplicate_first_correction_or_extra_subtraction": False,
            "fallback_cycles_per_output_block": "source population",
            "fallback_formula_changed": False,
        },
        "wide12_exhaustive_proof": wide12,
        "m426_csv_comparison": csv_audit,
        "recovery_chain_audit": recovery,
        "resource_fairness": {
            "low_pwp_bank_bytes_per_coread_cycle": 96,
            "high_pwp_bank_logical_bytes_per_coread_cycle": 48,
            "high_pwp_existing_m405_physical_port_bytes": 64,
            "aggregate_pwp_bytes_per_coread_cycle": 144,
            "aggregate_existing_m405_physical_input_port_bytes": 160,
            "correction_bank_bytes_per_fused_first_cycle": 96,
            "aggregate_first_fused_cycle_bytes": 240,
            "aggregate_first_fused_cycle_bits": 1920,
            "logical_capacity_change_claim": 0,
            "dram_payload_or_command_change_claim": 0,
            "m405_already_exposes_independent_low_and_high_input_ports": True,
            "new_bank_capacity_or_per_bank_port_count_proven": False,
            "existing_correction_rate": "one 96-byte signed vector per cycle",
            "correction_vectors_per_cycle_increased": False,
            "new_concurrency": (
                "same-cycle low/high PWP read, and on first positive-residual "
                "cycle same-cycle reconstructed PWP plus correction-vector read"
            ),
            "new_interconnect_or_critical_path": True,
            "capacity_equal_to_m401_by_contract": True,
            "resource_equal_to_strong_zero_baseline": False,
            "reason": (
                "M405 already has separate low/high ports but assembles/emits "
                "serially; M426 enables their concurrent use.  The correction "
                "bank preserves its existing one-vector/cycle rate, but its first "
                "residual access now coincides with PWP reconstruction.  Capacity "
                "may stay unchanged while aggregate enabled bandwidth, operand "
                "muxing, fanout, and the reconstruct-to-adder timing path change."
            ),
            "required_before_performance_admission": [
                "standalone RTL protocol and arithmetic VCS",
                "dual-bank plus correction-bank macro/port model",
                "DC/STA area and timing for reconstruct/mux/adder path",
                "Formality or an exact RTL reference miter",
            ],
        },
        "decision": {
            "standalone_rtl_vcs_dc_formality": "GO_CONDITIONAL",
            "direct_headline_or_system_admission": "NO_GO",
            "allowed_claim_now":
                "executable architectural replay for four frozen H67 bottleneck Conv3x3 operators",
        },
        "claim_boundary": {
            "exact_architectural_cycle_replay": True,
            "four_h67_bottleneck_conv_only": True,
            "rtl_measured_speedup": False,
            "macro_or_interconnect_ppa": False,
            "power_or_energy": False,
            "full_network_or_system_speedup": False,
            "paper_ppa_ready": False,
            "date_headline": False,
        },
        "output_files": {
            "phase_timestamps": phase_csv.name,
        },
    }
    receipt = args.output_dir / "m427_independent_recompute_receipt_r1.json"
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    shutil.copyfile(args.contract, args.output_dir / "contract.json")
    shutil.copyfile(Path(__file__).resolve(),
                    args.output_dir / "independent_recompute_m427_m426r3.py")
    print("M427_RECOMPUTE_PASS strong={} current={} dual={} fused={} "
          "csv_rows={} wide12_mismatch=0".format(
              zero_time, times[VARIANTS[0]], times[VARIANTS[1]],
              times[VARIANTS[2]], csv_audit["rows_checked"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
