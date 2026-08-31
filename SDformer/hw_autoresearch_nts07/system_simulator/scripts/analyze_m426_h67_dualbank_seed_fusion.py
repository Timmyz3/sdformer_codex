#!/usr/bin/env python3
"""Replay exact H67 dual-bank PWP co-read and seed/correction fusion.

The replay retains the frozen M401 q32/O4 assignments, payload, matcher,
DMA, descriptor latency and phase ordering.  It changes only the PWP issue
microarchitecture:

* dual-bank: read low8 and high4 PWP banks together and reconstruct one
  signed-12 vector contribution per output block;
* seed-fusion: when residual distance is nonzero, use that reconstructed PWP
  as the accumulator's left operand while the first signed correction is the
  right operand.  Exact-match PWP rows still consume one seed cycle.

This is an executable architectural cycle replay, not RTL-measured speed,
SRAM-macro PPA, power, energy, a full network, or a system result.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROWS_PER_PHASE = 3000
PHASES = 17280
PHASES_PER_SAMPLE = 1728
PARTITIONS = 432
OUTPUT_BLOCKS_PER_TILE = 4
POPCOUNT = np.asarray([bin(value).count("1") for value in range(1 << 16)],
                      dtype=np.uint8)
VAR_CURRENT = "m401_serial_low8_high4"
VAR_DUAL = "dualbank_parallel_low8_high4"
VAR_FUSED = "dualbank_seed_first_correction_fused"
VARIANTS = (VAR_CURRENT, VAR_DUAL, VAR_FUSED)


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
    number = float(value)
    require(math.isfinite(number) and number.is_integer(),
            "non-integral {}={}".format(label, value))
    return int(number)


def decode_words(block, nibble):
    require(len(block) == ROWS_PER_PHASE * 9,
            "row phase byte extent drift")
    raw = np.frombuffer(block, dtype=np.uint8).reshape(ROWS_PER_PHASE, 9)
    require(bool(np.all(raw[:, 8] == 10)), "row newline drift")
    digits = nibble[raw[:, :8]]
    require(not bool(np.any(digits == 255)), "row non-hex digit")
    words = np.zeros(ROWS_PER_PHASE, dtype=np.uint32)
    for column in range(8):
        words = words * np.uint32(16) + digits[:, column].astype(np.uint32)
    require(not bool(np.any(words >> np.uint32(29))),
            "reserved row bits nonzero")
    return words


def phase_population(words):
    original = np.bitwise_and(words, np.uint32(0xffff)).astype(np.uint16)
    population = POPCOUNT[original].astype(np.int64)
    distance = np.bitwise_and(words >> np.uint32(21),
                              np.uint32(0x1f)).astype(np.int64)
    use_pwp = np.bitwise_and(words >> np.uint32(26),
                             np.uint32(1)).astype(np.int64)
    pass1 = np.bitwise_and(words >> np.uint32(27),
                          np.uint32(1)).astype(np.int64)
    early = np.bitwise_and(words >> np.uint32(28),
                          np.uint32(1)).astype(np.int64)
    active = int(np.count_nonzero(population))
    eligible = int(np.count_nonzero(population >= 2))
    pwp = int(use_pwp.sum())
    exact_pwp = int(np.count_nonzero((use_pwp != 0) & (distance == 0)))
    positive_pwp = pwp - exact_pwp
    correction = int(np.where(use_pwp != 0, distance, population).sum())
    require(bool(np.all(use_pwp == ((1 + distance) < population))),
            "PWP predicate drift")
    require(not bool(np.any(pass1 & early)), "pass1/early overlap")
    require(int(pass1.sum()) + int(early.sum()) == eligible,
            "eligible matcher partition drift")
    require(int(np.where(use_pwp != 0, distance, 0).sum()) >= positive_pwp,
            "positive residual cannot supply first correction")
    return {
        "source_rows": ROWS_PER_PHASE,
        "active_rows": active,
        "eligible_rows": eligible,
        "pwp_rows": pwp,
        "exact_pwp_rows": exact_pwp,
        "positive_residual_pwp_rows": positive_pwp,
        "fallback_rows": active - pwp,
        "correction_ops_per_block": correction,
        "bit_sparse_ops_per_block": int(population.sum()),
        "early_matcher": ROWS_PER_PHASE + int(pass1.sum()) + 2,
    }


def load_phase_ledger(path):
    needed = {
        "sample", "operator", "partition", "active_rows", "eligible_rows",
        "pwp_rows", "fallback_rows", "used_pwp_patterns",
        "used_center_runs", "narrow_tile0", "narrow_tile1",
        "early_matcher", "eligible_exact_q32",
    }
    result = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None and needed.issubset(reader.fieldnames),
                "M401 phase ledger schema drift")
        for index, row in enumerate(reader):
            item = {key: exact_int(value, "ledger." + key)
                    for key, value in row.items()}
            within = index % PHASES_PER_SAMPLE
            require((item["sample"], item["operator"], item["partition"]) ==
                    (index // PHASES_PER_SAMPLE, within // PARTITIONS,
                     within % PARTITIONS),
                    "M401 phase ordering drift at {}".format(index))
            result.append(item)
    require(len(result) == PHASES, "M401 phase ledger extent drift")
    return result


def variant_work(variant, pop, phase):
    correction_tile = OUTPUT_BLOCKS_PER_TILE * pop["correction_ops_per_block"]
    if variant == VAR_CURRENT:
        work0 = correction_tile + 8 * pop["pwp_rows"] - phase["narrow_tile0"]
        work1 = correction_tile + 8 * pop["pwp_rows"] - phase["narrow_tile1"]
    elif variant == VAR_DUAL:
        work0 = correction_tile + 4 * pop["pwp_rows"]
        work1 = correction_tile + 4 * pop["pwp_rows"]
    elif variant == VAR_FUSED:
        work0 = correction_tile + 4 * pop["exact_pwp_rows"]
        work1 = correction_tile + 4 * pop["exact_pwp_rows"]
    else:
        raise RuntimeError("unknown variant")
    require(work0 >= pop["active_rows"] and work1 >= pop["active_rows"],
            "descriptor issue underflow: " + variant)
    return work0, work1


def replay_phase(index, start, variant, pop, phase):
    time = start
    config_data = 3
    config_command = 32
    matcher = pop["early_matcher"]
    seal = 1
    time += config_data + config_command + matcher + seal
    tile_dma = tile1_exposed = replay0 = replay1 = active_work = 0
    work0 = work1 = 0
    if pop["active_rows"]:
        tile_bytes = 6144 + phase["used_pwp_patterns"] * 640
        require(96 + tile_bytes <= 32768 and tile_bytes % 32 == 0,
                "frozen slot/DMA alignment drift")
        tile_dma = tile_bytes // 32 + (1 + phase["used_center_runs"]) * 32
        work0, work1 = variant_work(variant, pop, phase)
        replay0 = work0 + 8
        replay1 = work1 + 8
        time += tile_dma
        tile1_exposed = max(0, tile_dma - replay0)
        time += replay0 + tile1_exposed + replay1
        active_work = work0 + work1
    tail = 2
    commit = 96000 if index % PHASES_PER_SAMPLE == PHASES_PER_SAMPLE - 1 else 0
    time += tail + commit
    return time, {
        "variant": variant,
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
        "active_compute_work": active_work,
        "matcher_cycles": matcher,
        "tile_dma_cycles": tile_dma,
        "tile1_dma_exposed_cycles": tile1_exposed,
        "replay0_cycles": replay0,
        "replay1_cycles": replay1,
        "tail_cycles": tail,
        "commit_cycles": commit,
    }


def zero_baseline_phase(index, start, pop):
    first = index % PHASES_PER_SAMPLE == 0
    last = index % PHASES_PER_SAMPLE == PHASES_PER_SAMPLE - 1
    preprocess = max(ROWS_PER_PHASE + 5, 12288 // 32 + 32)
    compute = pop["bit_sparse_ops_per_block"] * 8
    initial = preprocess if first else 0
    next_preprocess = 0 if last else preprocess
    exposed = max(0, next_preprocess - compute)
    return start + initial + compute + exposed + 2 + (96000 if last else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hw-root", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    hw = args.hw_root.resolve()
    contract = strict_json(args.contract)
    require(contract["schema"] == "m426_h67_dualbank_seed_fusion_contract_v1" and
            contract["status_before_execution"] ==
            "FROZEN_BEFORE_M426_EXECUTABLE_REPLAY",
            "M426 contract state drift")
    require(not args.output_dir.exists(), "refusing M426 output overwrite")

    inputs = {}
    for name, identity in contract["inputs"].items():
        path = (hw / identity["path"] if not Path(identity["path"]).is_absolute()
                else Path(identity["path"]))
        require(path.is_file() and not path.is_symlink(),
                "missing/symlink input: " + name)
        require(sha256(path) == identity["sha256"], "SHA drift: " + name)
        inputs[name] = path
    require(sha256(Path(__file__).resolve()) ==
            contract["analyzer"]["sha256"], "analyzer SHA drift")

    m401 = strict_json(inputs["m401_result"])
    m418 = strict_json(inputs["m418_result"])
    m419 = strict_json(inputs["m419_review"])
    require(m401["status"] ==
            "PASS_M401_H67_Q32_EXACT_ELASTIC_PWP_FULL_REPLAY" and
            m419["score"] == 94 and m419["severity_counts"]["P0"] == 0,
            "upstream admission drift")
    ledger = load_phase_ledger(inputs["m401_phase_ledger"])
    args.output_dir.mkdir(parents=True, exist_ok=False)

    nibble = np.full(256, 255, dtype=np.uint8)
    nibble[ord("0"):ord("9") + 1] = np.arange(10, dtype=np.uint8)
    nibble[ord("a"):ord("f") + 1] = np.arange(10, 16, dtype=np.uint8)
    nibble[ord("A"):ord("F") + 1] = np.arange(10, 16, dtype=np.uint8)
    times = {variant: 0 for variant in VARIANTS}
    zero_time = 0
    totals = {variant: Counter() for variant in VARIANTS}
    population = Counter()
    rows_out = []
    with inputs["m410_rows"].open("rb") as handle:
        for index in range(PHASES):
            pop = phase_population(handle.read(ROWS_PER_PHASE * 9), nibble)
            phase = ledger[index]
            for key in ("active_rows", "eligible_rows", "pwp_rows",
                        "fallback_rows", "early_matcher"):
                require(pop[key] == phase[key],
                        "raw/M401 phase drift {} {}".format(index, key))
            require(pop["exact_pwp_rows"] == phase["eligible_exact_q32"],
                    "exact PWP phase drift {}".format(index))
            population.update(pop)
            zero_time = zero_baseline_phase(index, zero_time, pop)
            for variant in VARIANTS:
                times[variant], record = replay_phase(
                    index, times[variant], variant, pop, phase)
                rows_out.append(record)
                for key in ("phase_cycles", "active_compute_work",
                            "matcher_cycles", "tile_dma_cycles",
                            "tile1_dma_exposed_cycles", "replay0_cycles",
                            "replay1_cycles", "tail_cycles", "commit_cycles"):
                    totals[variant][key] += record[key]
            if (index + 1) % PHASES_PER_SAMPLE == 0:
                print("[M426] replayed {}/{} phases".format(index + 1, PHASES),
                      flush=True)
        require(handle.read(1) == b"", "row transport trailing bytes")

    require(zero_time == 742148386, "strong zero baseline reproduction drift")
    require(times[VAR_CURRENT] == 641790704,
            "M401 selected-cycle reproduction drift")
    require(m418["variants"]["dense16_same_resource"]["cycles"] ==
            6636544610 and
            m418["variants"]["zero_elided_bit_sparse_exact_reproduction"]
            ["cycles"] == zero_time,
            "M418 baseline identity drift")
    require(population["pwp_rows"] == 16971357 and
            population["exact_pwp_rows"] == 5350591 and
            population["positive_residual_pwp_rows"] == 11620766 and
            population["correction_ops_per_block"] == 38690838,
            "M426 population anchor drift")

    csv_path = args.output_dir / "m426_per_phase_three_candidate_replay.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)

    variants = {}
    for variant in VARIANTS:
        variants[variant] = {
            "cycles": times[variant],
            "speedup_vs_strong_zero_elided": zero_time / float(times[variant]),
            "speedup_vs_weak_dense16": 6636544610 / float(times[variant]),
            "components": dict(totals[variant]),
        }
    dual_saved = times[VAR_CURRENT] - times[VAR_DUAL]
    fusion_saved = times[VAR_DUAL] - times[VAR_FUSED]
    require(dual_saved == 111184044 and
            dual_saved == m401["runtime_elastic"]["wide_block_descriptors"],
            "dual-bank savings anchor drift")
    require(fusion_saved ==
            population["positive_residual_pwp_rows"] * 8,
            "seed-fusion savings conservation drift")
    decision = (
        "GO_DUALBANK_AND_SEED_FUSION_RTL"
        if variants[VAR_DUAL]["speedup_vs_strong_zero_elided"] >= 1.30 and
        variants[VAR_FUSED]["speedup_vs_strong_zero_elided"] >= 1.50
        else "NO_GO_DUALBANK_AND_SEED_FUSION_RTL"
    )
    result = {
        "schema": "m426_h67_dualbank_seed_fusion_executable_replay_v1",
        "status": "PASS_M426_H67_EXACT_DUALBANK_SEED_FUSION_REPLAY",
        "decision": decision,
        "scope": "four frozen H67 bottleneck Conv3x3 operators only",
        "population": dict(population),
        "baselines": {
            "strong_zero_elided_cycles": zero_time,
            "weak_dense16_cycles": 6636544610,
            "primary_comparison": "strong_zero_elided",
        },
        "variants": variants,
        "cycle_savings": {
            "parallel_low8_high4_coread_vs_m401": dual_saved,
            "seed_first_correction_fusion_vs_coread": fusion_saved,
            "total_vs_m401": times[VAR_CURRENT] - times[VAR_FUSED],
            "coread_conservation": "one saved issue cycle for every M401 wide PWP block",
            "fusion_conservation": "eight saved issue cycles for every positive-residual PWP row",
        },
        "hardware_contract": {
            "pwp_storage": "same low8 and high4 logical payload; independent 96-byte and 48-byte banks co-read in one cycle for wide blocks; high bank clock-gated for narrow blocks",
            "correction_storage": "existing separate 96-byte signed INT8 weight-vector bank",
            "fused_datapath": "one 96-lane signed accumulator adder; mux selects reconstructed PWP instead of prior accumulator for the first correction; no second correction is issued",
            "exactness": "for d>0, first cycle computes PWP + signed W_delta; for d=0, one PWP seed cycle; fallback remains exact bit-sparse",
            "unchanged": ["q32/O4 assignments", "checkpoint", "payload bytes", "DMA commands and data", "matcher", "descriptor latency", "phase order", "commit and tail"],
        },
        "execution_gates": {
            "phases": PHASES,
            "raw_row_words": PHASES * ROWS_PER_PHASE,
            "m401_cycle_reproduction_mismatch": 0,
            "strong_zero_cycle_reproduction_mismatch": 0,
            "population_mismatch": 0,
            "arithmetic_or_accuracy_change": 0,
        },
        "output_files": {"per_phase_csv": csv_path.name},
        "claim_boundary": {
            "executable_architectural_cycle_replay": True,
            "exact_arithmetic": True,
            "four_bottleneck_conv_only": True,
            "rtl_measured_speedup": False,
            "dualbank_macro_or_interconnect_ppa": False,
            "dc_sta_formality": False,
            "power_or_energy": False,
            "full_network_or_system_speedup": False,
            "paper_ppa_ready": False,
            "date_headline": False,
        },
    }
    output_json = args.output_dir / "m426_h67_dualbank_seed_fusion_replay_r1.json"
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("M426_PASS dual={:.9f}x fused={:.9f}x decision={}".format(
        variants[VAR_DUAL]["speedup_vs_strong_zero_elided"],
        variants[VAR_FUSED]["speedup_vs_strong_zero_elided"], decision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
