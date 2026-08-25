#!/usr/bin/env python3
"""Bound the M78 cap11 shared32 cycle model with the admitted M86 frontend.

This is deliberately a module cycle simulator, not a full-network scheduler.
It replays the exact M78 heldout phase work, replaces ideal PWP fetches with
M86's 3/4/4/5-beat synchronous-bank service, charges canonical M83 record
bytes, and schedules phase preparation through two finite 460-row buffers.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M78_PATH = HW / "system_simulator/scripts/analyze_m78_precision_elastic_pwp.py"
M78_RESULT = HW / (
    "results/m78_precision_elastic_pwp_valid825_internal_dev_r1_20260823/"
    "m78_precision_elastic_pwp.json")
M83_RECEIPT = HW / (
    "results/m83_canonical_cap11_pwp_records_r1_20260823/"
    "m83_canonical_cap11_pwp_records_receipt.json")
M86_RUN = HW / (
    "dc_handoff/runs/m86_sync_banked_guarded_pwp_vcs_r1_sealed_20260823/"
    "RUN_COMPLETE.txt")
M86_INPUTS = HW / (
    "dc_handoff/runs/m86_sync_banked_guarded_pwp_vcs_r1_sealed_20260823/"
    "input_sha256.txt")
M86_CONTRACT = HW / (
    "contracts/m86_sync_banked_guarded_pwp_frontend_vcs_contract_r1_20260823.json")
EXPECTED_SHA256 = {
    "m78_source": "9215c2eeff8ccbfa0ef7d27f48ed6100a56813c1881873013e9e23a2e149df6b",
    "m78_result": "00d2802eb8e4085fdf740f0183b23488ef2def5ca38f027c57ccba04f30064cc",
    "m83_receipt": "46893b0dc7499f3c163d4c3709560f5d208a2272bb49dd8ce709132062bb4303",
    "m86_contract": "d7bb4929abca9d3f9562c3a7d85bdaa769734a877e064c50eaa6173fc519578a",
}
M83_OFFSETS_SHA256 = (
    "1cddfc800ba18569b9c0d7f4c193f4b07ddf9046fa7688a1859f6c3e448bf30c")
CAP = 11
PHASES = 4 * 432
SAMPLES = tuple(range(5, 10))
ROWS = 460
METADATA_BYTES = 74
PWP_BUFFER_BYTES = ROWS * 8 * 4
WEIGHT_PHASE_BYTES = 16 * 8 * 96
DRAM_BYTES_PER_CYCLE = 32
PARSER_CYCLES = 128
METADATA_COMMIT_CYCLES = 1
M86_SYNC_PIPELINE_FILL_CYCLES = 2


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
        raise ValueError("non-standard JSON constant: " + raw)

    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def load_m78():
    spec = importlib.util.spec_from_file_location("m88_m78", str(M78_PATH))
    require(spec is not None and spec.loader is not None, "cannot load M78")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_offsets(path):
    raw = Path(path).read_bytes()
    require(len(raw) == (PHASES + 1) * 4, "M88 offset extent mismatch")
    offsets = list(struct.unpack("<{}I".format(PHASES + 1), raw))
    require(offsets[0] == 0 and all(a < b for a, b in zip(offsets, offsets[1:])),
            "M88 offsets are not strict phase boundaries")
    return offsets


def phase_prepare_cycles(record_bytes):
    # The weight phase and canonical PWP record share the 32-byte DRAM port.
    # SRAM row writes and the 128-entry metadata parser may proceed while the
    # transfer is in flight, but all three resources must finish before commit.
    shared_dram = int(math.ceil(
        (WEIGHT_PHASE_BYTES + record_bytes) / float(DRAM_BYTES_PER_CYCLE)))
    return max(shared_dram, ROWS, PARSER_CYCLES) + METADATA_COMMIT_CYCLES


def bounded_double_buffer(durations, preparations):
    require(len(durations) == len(preparations) == PHASES,
            "M88 bounded schedule extent mismatch")
    compute_ends = []
    dma_end = 0
    load_wait = 0
    midstream_stalls = 0
    maximum_ready_ahead = 0
    for phase, (duration, preparation) in enumerate(zip(durations, preparations)):
        slot_free = compute_ends[phase - 2] if phase >= 2 else 0
        load_start = max(dma_end, slot_free)
        load_end = load_start + preparation
        previous_compute_end = compute_ends[-1] if compute_ends else 0
        compute_start = max(previous_compute_end, load_end)
        wait = compute_start - previous_compute_end
        load_wait += wait
        if phase > 0 and wait > 0:
            midstream_stalls += 1
        maximum_ready_ahead = max(maximum_ready_ahead,
                                  max(0, previous_compute_end - load_end))
        compute_ends.append(compute_start + duration)
        dma_end = load_end
    return {
        "cycles": compute_ends[-1],
        "load_wait_cycles_including_startup": load_wait,
        "midstream_load_stall_phases": midstream_stalls,
        "maximum_phase_ready_ahead_cycles": maximum_ready_ahead,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m83-offsets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M88 output overwrite")
    source_start_sha = sha256(Path(__file__).resolve())
    for name, path in (("m78_source", M78_PATH), ("m78_result", M78_RESULT),
                       ("m83_receipt", M83_RECEIPT),
                       ("m86_contract", M86_CONTRACT)):
        require(sha256(path) == EXPECTED_SHA256[name], "M88 identity drift: " + name)
    require(sha256(args.m83_offsets) == M83_OFFSETS_SHA256,
            "M88 M83 offsets identity drift")
    require(M86_RUN.read_text(encoding="utf-8").splitlines()[0] ==
            "status=PASS_M86_SYNC_BANK_ACTUAL_RECORD_VCS_SVA",
            "M88 M86 VCS admission missing")
    require("bank_read_issues=835383" in M86_RUN.read_text(encoding="utf-8")
            and "compiled_sram_macro=false" in M86_RUN.read_text(encoding="utf-8"),
            "M88 M86 scope receipt mismatch")

    m78 = load_m78()
    stored = strict_json(M78_RESULT)
    m72 = m78.load_m72()
    m72_result = m78.strict_json(m78.M72_RESULT)
    m41_result = m78.strict_json(m78.M41_RESULT)
    width_catalog, _, _, _, _ = m78.build_width_catalog(m72_result, m41_result)
    manifest = m78.strict_json(m72.MANIFEST_PATH)
    operator_names = [item["operator"] for item in m72_result["operators"]]
    m43 = m72.load_m43()
    histograms = m78.collect_per_sample_histograms(
        m72, m43, manifest, operator_names)
    offsets = read_offsets(args.m83_offsets)
    record_lengths = [offsets[index + 1] - offsets[index]
                      for index in range(PHASES)]
    preparations = [phase_prepare_cycles(length) for length in record_lengths]
    baseline_preparations = [
        int(math.ceil(WEIGHT_PHASE_BYTES / float(DRAM_BYTES_PER_CYCLE)))] * PHASES

    stored_cap = next(row for row in stored["configurations"]
                      if row["signed_width_cap"] == CAP)
    stored_shared32 = next(row for row in stored_cap["cycle_simulations"]
                           if row["port"] == "SHARED_32B")
    results = []
    totals = Counter()
    minimum_compute_margin = None
    for sample in SAMPLES:
        candidate_durations = []
        baseline_durations = []
        m78_phases = []
        for op in range(4):
            for partition in range(432):
                centers = [item["center"] for item in width_catalog[op][partition]]
                widths = [item["blocks"] for item in width_catalog[op][partition]]
                base, caps = m78.phase_metrics(
                    histograms[(sample, op, partition)], centers, widths)
                row = caps[CAP]
                pwp_compute = sum(
                    uses * m78.pwp_service_cycles(width, m78.PORTS[2])
                    for width, uses in row["pwp_uses_by_width"].items())
                candidate_compute = (row["correction_ops_all_blocks"] *
                                     m78.PORTS[2]["weight_cycles"] + pwp_compute)
                matcher = base["matcher_rows"] + m78.MATCHER_PIPELINE_CYCLES
                packer = int(math.ceil(row["assignment_rows"] / 8.0)) + \
                    m78.PACKER_PIPELINE_CYCLES
                candidate_duration = (max(candidate_compute, matcher, packer) +
                                      m78.COMPUTE_TAIL_CYCLES +
                                      M86_SYNC_PIPELINE_FILL_CYCLES)
                baseline_compute = (base["baseline_ops_per_block"] *
                                    m78.OUTPUT_BLOCKS *
                                    m78.PORTS[2]["weight_cycles"])
                baseline_duration = baseline_compute + m78.COMPUTE_TAIL_CYCLES
                candidate_durations.append(candidate_duration)
                baseline_durations.append(baseline_duration)
                m78_phases.append({
                    "base": base,
                    "caps": caps,
                    "pwp_payload_bytes": {
                        CAP: m78.phase_pwp_payload_bytes(
                            width_catalog[op][partition], CAP)},
                })
                margin = candidate_duration - preparations[op*432 + partition]
                minimum_compute_margin = (margin if minimum_compute_margin is None
                                          else min(minimum_compute_margin, margin))
        legacy = m78.replay_sample(m78_phases, CAP, m78.PORTS[2])
        stored_sample = next(row for row in stored_shared32["per_sample"]
                             if row["sample_id"] == sample)
        require(legacy["candidate_cycles"] == stored_sample["candidate_cycles"]
                and legacy["bit_sparse_cycles"] == stored_sample["bit_sparse_cycles"],
                "M88 failed to reproduce frozen M78 sample")
        candidate = bounded_double_buffer(candidate_durations, preparations)
        baseline = bounded_double_buffer(baseline_durations,
                                         baseline_preparations)
        row = {
            "sample_id": sample,
            "m78_candidate_cycles": legacy["candidate_cycles"],
            "bounded_candidate_cycles": candidate["cycles"],
            "bounded_bit_sparse_cycles": baseline["cycles"],
            "speedup_vs_bit_sparse": (
                baseline["cycles"] / float(candidate["cycles"])),
            "candidate_schedule": candidate,
            "baseline_schedule": baseline,
        }
        results.append(row)
        totals["candidate"] += candidate["cycles"]
        totals["baseline"] += baseline["cycles"]
        totals["m78_candidate"] += legacy["candidate_cycles"]
        totals["candidate_midstream_stalls"] += candidate[
            "midstream_load_stall_phases"]

    require(totals["m78_candidate"] == stored_shared32["candidate_cycles"],
            "M88 aggregate M78 reproduction mismatch")
    require(sha256(Path(__file__).resolve()) == source_start_sha,
            "M88 analyzer source changed during run")
    response_fifo_bytes = int(math.ceil(4 * (256 + 4 + 3 + 3 + 32) / 8.0))
    local_storage = {
        "two_pwp_phase_buffers_bytes": 2 * (PWP_BUFFER_BYTES + METADATA_BYTES),
        "two_weight_phase_buffers_bytes": 2 * WEIGHT_PHASE_BYTES,
        "four_entry_response_fifo_bytes": response_fifo_bytes,
        "pattern_table_bytes": 4 * 432 * 16 * 2,
        "phase_offset_table_bytes": (PHASES + 1) * 4,
    }
    local_storage["total_bytes"] = sum(local_storage.values())
    payload = {
        "schema": "m88_bounded_sync_bank_double_buffer_cycle_sim_v1",
        "status": "PASS_M88_BOUNDED_MODULE_CYCLE_SIM_VALID825_INTERNAL_ONLY",
        "identity": {
            "analyzer_start_end_sha256": source_start_sha,
            "m78_result_sha256": sha256(M78_RESULT),
            "m83_offsets_sha256": sha256(args.m83_offsets),
            "m83_receipt_sha256": sha256(M83_RECEIPT),
            "m86_contract_sha256": sha256(M86_CONTRACT),
            "m86_run_complete_sha256": sha256(M86_RUN),
            "m86_input_manifest_sha256": sha256(M86_INPUTS),
        },
        "resource_model": {
            "dram_bytes_per_cycle": DRAM_BYTES_PER_CYCLE,
            "finite_phase_buffers": 2,
            "pwp_rows_per_buffer": ROWS,
            "pwp_banks": 8,
            "pwp_word_bits": 32,
            "metadata_bytes_per_phase": METADATA_BYTES,
            "metadata_parser_cycles": PARSER_CYCLES,
            "m86_sync_pipeline_fill_cycles_per_phase": (
                M86_SYNC_PIPELINE_FILL_CYCLES),
            "cap11_shared32_intrinsic_pwp_cycles": {
                "signed8": 3, "signed9": 4, "signed10": 4, "signed11": 5},
            "local_storage": local_storage,
        },
        "phase_preparation": {
            "canonical_record_bytes_min": min(record_lengths),
            "canonical_record_bytes_max": max(record_lengths),
            "prepare_cycles_min": min(preparations),
            "prepare_cycles_max": max(preparations),
            "minimum_compute_minus_prepare_margin_cycles": minimum_compute_margin,
        },
        "aggregate": {
            "bounded_candidate_cycles": totals["candidate"],
            "bounded_bit_sparse_cycles": totals["baseline"],
            "speedup_vs_bit_sparse": totals["baseline"] / float(totals["candidate"]),
            "frozen_m78_candidate_cycles": totals["m78_candidate"],
            "cycle_increase_vs_m78": totals["candidate"] - totals["m78_candidate"],
            "candidate_midstream_load_stall_phases": totals[
                "candidate_midstream_stalls"],
        },
        "per_sample": results,
        "admission": {
            "exact_m78_work_reproduced": True,
            "canonical_record_bytes_charged": True,
            "finite_double_buffer_schedule": True,
            "m86_sync_bank_service_charged": True,
            "isolated_module_cycle_simulator_estimate": True,
            "train_catalog": False,
            "accuracy": False,
            "physical_sram_macro_ppa": False,
            "full_network_or_system_speedup": False,
            "date_headline": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS M88 bounded_candidate={} bounded_bit_sparse={} speedup={:.9f}x midstream_stalls={}".format(
        totals["candidate"], totals["baseline"],
        payload["aggregate"]["speedup_vs_bit_sparse"],
        totals["candidate_midstream_stalls"]), flush=True)


if __name__ == "__main__":
    main()
