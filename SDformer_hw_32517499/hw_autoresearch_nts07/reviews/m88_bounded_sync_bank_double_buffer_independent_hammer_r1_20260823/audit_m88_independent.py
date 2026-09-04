#!/usr/bin/env python3
"""Independent M88 accounting and bounded-timeline reconstruction.

This review never imports the M88 or M78 production analyzers.  It starts from
the sealed independent M78 reconstruction, directly decodes every canonical M83
record/offset, proves the no-stall condition algebraically, and rebuilds all
per-sample and aggregate M88 headline numbers.
"""

from __future__ import print_function

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
M88_ANALYZER = HW / "system_simulator/scripts/analyze_m88_bounded_sync_bank_double_buffer.py"
M88_RESULT = HW / (
    "results/m88_bounded_sync_bank_double_buffer_valid825_internal_r1_20260823/"
    "m88_bounded_sync_bank_double_buffer.json")
M78_INDEPENDENT = HW / (
    "reviews/m78_precision_elastic_pwp_independent_hammer_r1_20260823/"
    "m78_independent_reconstruction.json")
M78_REVIEW = HW / (
    "reviews/m78_precision_elastic_pwp_independent_hammer_r1_20260823/"
    "m78_precision_elastic_pwp_independent_hammer_review.json")
M83_REVIEW = HW / (
    "reviews/m83_canonical_cap11_pwp_records_independent_hammer_r1_20260823/"
    "m83_independent_hammer_review.json")
M86_REVIEW = HW / (
    "reviews/m86_sync_banked_guarded_pwp_independent_hammer_r1_20260823/"
    "m86_sync_banked_guarded_pwp_independent_hammer_review.json")
M86_RUN = HW / (
    "dc_handoff/runs/m86_sync_banked_guarded_pwp_vcs_r1_sealed_20260823/"
    "RUN_COMPLETE.txt")
M86_CONTRACT = HW / (
    "contracts/m86_sync_banked_guarded_pwp_frontend_vcs_contract_r1_20260823.json")
M83_RECORDS = Path("/tmp/m85_inputs/m83_cap11_phase_records.bin")
M83_OFFSETS = Path("/tmp/m85_inputs/m83_cap11_phase_offsets_u32le.bin")

EXPECTED = {
    "m88_analyzer": "5b62d1f23555fba4bc00f1e1b427ae5861089e0a8ea5f8ae98c062acb071dfae",
    "m88_result": "36e9b0603422ccff7afd23e6e5e2309bc5d53b3c7e9898538095d6baa23da483",
    "m78_independent": "100f5803923caf1bdba27318819089308eb869e853376d011690116ca3c9dd36",
    "m78_review": "9661e4a3750de325e1f0b885ca9b6170e289b142ef75643ad13f3a05c50f7a94",
    "m83_review": "bfb5ed654e5eedfc7d88e7ee8f630b36379c594306333a44b42ce9e39f6c4100",
    "m86_review": "ed7ce836f20cd4f5d018741aa142e04f8db984b8471d9f9a0831f0d573c421ed",
    "m86_contract": "d7bb4929abca9d3f9562c3a7d85bdaa769734a877e064c50eaa6173fc519578a",
    "m86_run": "140a8094fa51a6e5cd024aa78148da22bfe1a91a57782a6d384718707d0314a6",
    "m83_records": "6de1521b2ee91281eadd5945d6f69b45df2cf5f1e2cc0c93834df4ec4c87190d",
    "m83_offsets": "1cddfc800ba18569b9c0d7f4c193f4b07ddf9046fa7688a1859f6c3e448bf30c",
}

PHASES = 1728
ROWS = 460
WEIGHT_PHASE_BYTES = 12288
DRAM_BYTES_PER_CYCLE = 32
SYNC_FILL_PER_PHASE = 2
METADATA_BYTES = 74
WORDS_BY_CODE = {0: 24, 1: 27, 2: 30, 3: 33, 4: 0}
BEATS_BY_CODE = {0: 3, 1: 4, 2: 4, 3: 5, 4: 1}


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
    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=lambda raw: (_ for _ in ()).throw(
            ValueError("non-standard JSON constant: " + raw)))


def compare(left, right, label="root"):
    if isinstance(left, dict) and isinstance(right, dict):
        require(set(left) == set(right), label + " keys drift")
        for key in left:
            compare(left[key], right[key], label + "." + str(key))
    elif isinstance(left, list) and isinstance(right, list):
        require(len(left) == len(right), label + " length drift")
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, label + "[{}]".format(index))
    elif isinstance(left, float) or isinstance(right, float):
        require(abs(float(left) - float(right)) <=
                1e-12 * max(1.0, abs(float(right))), label + " float drift")
    else:
        require(left == right, label + " drift: {} != {}".format(left, right))


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def decode_records():
    raw = M83_RECORDS.read_bytes()
    offset_raw = M83_OFFSETS.read_bytes()
    require(len(offset_raw) == (PHASES + 1) * 4, "offset extent drift")
    offsets = struct.unpack("<{}I".format(PHASES + 1), offset_raw)
    require(offsets[0] == 0 and offsets[-1] == len(raw),
            "offset terminal drift")
    require(all(a < b for a, b in zip(offsets, offsets[1:])),
            "offset ordering drift")

    code_histogram = Counter()
    record_lengths = []
    payload_bytes = []
    terminal_rows = []
    new_preparations = []
    old_preparations = []
    padding_bytes = 0
    beat_issues = 0
    for phase in range(PHASES):
        record = raw[offsets[phase]:offsets[phase + 1]]
        require(len(record) >= 48 and len(record) % 32 == 0,
                "record alignment drift")
        header = int.from_bytes(record[:48], "little")
        codes = [(header >> (3 * entry)) & 7 for entry in range(128)]
        require(all(code in WORDS_BY_CODE for code in codes),
                "reserved M83 code observed")
        code_histogram.update(codes)
        beat_issues += sum(BEATS_BY_CODE[code] for code in codes)
        words = sum(WORDS_BY_CODE[code] for code in codes)
        payload = words * 4
        expected_length = ceil_div(48 + payload, 32) * 32
        require(len(record) == expected_length, "record length/payload drift")
        require(not any(record[48 + payload:]), "nonzero phase padding")
        record_lengths.append(len(record))
        payload_bytes.append(payload)
        padding_bytes += len(record) - 48 - payload
        terminal_rows.append(ceil_div(words, 8))
        old_preparations.append(ceil_div(
            WEIGHT_PHASE_BYTES + payload, DRAM_BYTES_PER_CYCLE))
        shared_dram = ceil_div(
            WEIGHT_PHASE_BYTES + len(record), DRAM_BYTES_PER_CYCLE)
        new_preparations.append(max(shared_dram, ROWS, 128) + 1)

    require(code_histogram == Counter({
        0: 52248, 1: 128893, 2: 37144, 3: 2898, 4: 1}),
        "M83 width/escape histogram drift")
    require(beat_issues == 835383, "M86 beat ledger drift")
    require(sum(payload_bytes) == 23776068, "payload byte ledger drift")
    require(sum(record_lengths) == 23884000, "record byte ledger drift")
    require(padding_bytes == 24988, "padding byte ledger drift")
    require(sum(terminal_rows) == 743731, "terminal row ledger drift")
    require(max(terminal_rows) == ROWS, "maximum row bound drift")
    deltas = Counter(new - old for new, old in zip(
        new_preparations, old_preparations))
    require(deltas == Counter({2: 812, 3: 916}),
            "new-vs-M78 preparation delta drift")
    return {
        "record_lengths": record_lengths,
        "payload_bytes": payload_bytes,
        "terminal_rows": terminal_rows,
        "new_preparations": new_preparations,
        "old_preparations": old_preparations,
        "code_histogram": dict((str(key), value)
                               for key, value in sorted(code_histogram.items())),
        "beat_issues": beat_issues,
        "padding_bytes": padding_bytes,
        "preparation_delta_histogram": dict((str(key), value)
                                             for key, value in sorted(deltas.items())),
    }


def reconstruct_timeline(records, independent_m78, m78_review, stored):
    cap11 = next(row for row in independent_m78["configurations"]
                 if row["signed_width_cap"] == 11)
    shared32 = next(row for row in cap11["cycle_simulations"]
                    if row["port"] == "SHARED_32B")
    require(shared32["binding_phases"] == {"compute": 8640,
                                              "matcher": 0,
                                              "packer": 0,
                                              "dma": 0},
            "sealed independent M78 compute-bound proof drift")
    old_margin = m78_review["cycle_reconstruction"]["cap11"][
        "SHARED_32B"]["minimum_compute_margin_over_matcher_packer_dma"]
    maximum_extra_prepare = max(
        new - old for new, old in zip(records["new_preparations"],
                                      records["old_preparations"]))
    conservative_new_margin = old_margin - maximum_extra_prepare
    require(old_margin == 12637 and conservative_new_margin == 12634,
            "no-stall margin proof drift")
    # With two slots, phase i+2 may start loading when phase i releases its
    # slot.  The strictly positive bound above proves every next preparation
    # completes inside phase i+1 compute, so only phase-0 startup is exposed.
    require(conservative_new_margin > 0, "double-buffer midstream stall possible")

    first_prepare_delta = (records["new_preparations"][0] -
                           records["old_preparations"][0])
    require(first_prepare_delta == 2 and
            records["new_preparations"][0] == 838 and
            records["old_preparations"][0] == 836,
            "phase-0 preparation identity drift")
    per_sample = []
    for old, reported in zip(shared32["per_sample"], stored["per_sample"]):
        candidate = (old["candidate_cycles"] +
                     PHASES * SYNC_FILL_PER_PHASE + first_prepare_delta)
        baseline = old["bit_sparse_cycles"]
        row = {
            "sample_id": old["sample_id"],
            "bounded_candidate_cycles": candidate,
            "bounded_bit_sparse_cycles": baseline,
            "speedup_vs_bit_sparse": baseline / float(candidate),
            "m78_candidate_cycles": old["candidate_cycles"],
            "cycle_delta_vs_m78": candidate - old["candidate_cycles"],
            "midstream_load_stall_phases": 0,
        }
        compare(row["sample_id"], reported["sample_id"], "sample id")
        for key in ("bounded_candidate_cycles", "bounded_bit_sparse_cycles",
                    "speedup_vs_bit_sparse", "m78_candidate_cycles"):
            compare(row[key], reported[key],
                    "sample{}.{}".format(row["sample_id"], key))
        compare(reported["candidate_schedule"]["midstream_load_stall_phases"],
                0, "candidate stall count")
        compare(reported["baseline_schedule"]["midstream_load_stall_phases"],
                0, "baseline stall count")
        compare(reported["candidate_schedule"][
            "load_wait_cycles_including_startup"], 838,
            "candidate exposed startup")
        compare(reported["baseline_schedule"][
            "load_wait_cycles_including_startup"], 384,
            "baseline exposed startup")
        per_sample.append(row)

    aggregate_candidate = sum(row["bounded_candidate_cycles"] for row in per_sample)
    aggregate_baseline = sum(row["bounded_bit_sparse_cycles"] for row in per_sample)
    aggregate = {
        "bounded_candidate_cycles": aggregate_candidate,
        "bounded_bit_sparse_cycles": aggregate_baseline,
        "speedup_vs_bit_sparse": aggregate_baseline / float(aggregate_candidate),
        "frozen_m78_candidate_cycles": sum(
            row["m78_candidate_cycles"] for row in per_sample),
        "cycle_increase_vs_m78": sum(
            row["cycle_delta_vs_m78"] for row in per_sample),
        "candidate_midstream_load_stall_phases": 0,
    }
    compare(aggregate, stored["aggregate"], "aggregate")
    return {
        "proof": {
            "sealed_m78_minimum_compute_margin_cycles": old_margin,
            "maximum_extra_prepare_cycles_vs_m78": maximum_extra_prepare,
            "conservative_minimum_compute_minus_new_prepare_cycles": (
                conservative_new_margin),
            "phase0_old_prepare_cycles": records["old_preparations"][0],
            "phase0_new_prepare_cycles": records["new_preparations"][0],
            "sync_fill_cycles_per_phase": SYNC_FILL_PER_PHASE,
            "per_sample_cycle_delta_equation": "2*1728 + (838-836) = 3458",
            "two_slot_induction": (
                "slot(i+2) frees at compute_end(i); positive duration(i+1)-"
                "prepare(i+2) means load(i+2) completes before compute_end(i+1)"),
        },
        "per_sample": per_sample,
        "aggregate": aggregate,
    }


def reconstruct_storage(stored):
    values = {
        "two_pwp_phase_buffers_bytes": 2 * (ROWS * 8 * 4 + METADATA_BYTES),
        "two_weight_phase_buffers_bytes": 2 * WEIGHT_PHASE_BYTES,
        "four_entry_response_fifo_bytes": ceil_div(
            4 * (256 + 4 + 3 + 3 + 32), 8),
        "pattern_table_bytes": 4 * 432 * 16 * 2,
        "phase_offset_table_bytes": (PHASES + 1) * 4,
    }
    values["total_bytes"] = sum(values.values())
    compare(values, stored["resource_model"]["local_storage"], "local_storage")
    require(values["total_bytes"] == 116525, "storage total drift")
    return dict(values, **{
        "total_kib": values["total_bytes"] / 1024.0,
        "scope": "listed frontend/catalog structures only, not complete accelerator module",
    })


def main():
    paths = {
        "m88_analyzer": M88_ANALYZER,
        "m88_result": M88_RESULT,
        "m78_independent": M78_INDEPENDENT,
        "m78_review": M78_REVIEW,
        "m83_review": M83_REVIEW,
        "m86_review": M86_REVIEW,
        "m86_contract": M86_CONTRACT,
        "m86_run": M86_RUN,
        "m83_records": M83_RECORDS,
        "m83_offsets": M83_OFFSETS,
    }
    for name, path in paths.items():
        require(path.is_file() and sha256(path) == EXPECTED[name],
                name + " SHA drift")
    identity = dict((name, {"path": str(path), "sha256": sha256(path),
                                 "bytes": path.stat().st_size})
                    for name, path in paths.items())
    stored = strict_json(M88_RESULT)
    independent_m78 = strict_json(M78_INDEPENDENT)
    m78_review = strict_json(M78_REVIEW)
    m83_review = strict_json(M83_REVIEW)
    m86_review = strict_json(M86_REVIEW)
    require(independent_m78["status"] ==
            "PASS_M78_INDEPENDENT_M41_M72_M40_RECONSTRUCTION",
            "M78 independent admission drift")
    require(m83_review["status"] == "GO_M83_CLOSES_M80_SERIALIZATION_P0",
            "M83 independent admission drift")
    require(m86_review["status"] ==
            "SCOPED_GO_REGISTERED_BANK_FIFO_FUNCTIONAL_ONLY_NO_SPEEDUP_READMISSION",
            "M86 independent scope drift")
    require(stored["status"] ==
            "PASS_M88_BOUNDED_MODULE_CYCLE_SIM_VALID825_INTERNAL_ONLY",
            "M88 scope status drift")

    records = decode_records()
    compare(stored["phase_preparation"]["canonical_record_bytes_min"],
            min(records["record_lengths"]), "record length min")
    compare(stored["phase_preparation"]["canonical_record_bytes_max"],
            max(records["record_lengths"]), "record length max")
    compare(stored["phase_preparation"]["prepare_cycles_min"],
            min(records["new_preparations"]), "prepare cycles min")
    compare(stored["phase_preparation"]["prepare_cycles_max"],
            max(records["new_preparations"]), "prepare cycles max")
    timeline = reconstruct_timeline(records, independent_m78, m78_review, stored)
    storage = reconstruct_storage(stored)
    require(stored["admission"]["isolated_module_cycle_simulator_estimate"] is True
            and all(stored["admission"][key] is False for key in (
                "train_catalog", "accuracy", "physical_sram_macro_ppa",
                "full_network_or_system_speedup", "date_headline")),
            "M88 claim boundary widened")
    accounting = {
        "per_catalog_pass": {
            "weight_phase_bytes": PHASES * WEIGHT_PHASE_BYTES,
            "pwp_numeric_payload_bytes": sum(records["payload_bytes"]),
            "canonical_record_header_bytes": PHASES * 48,
            "canonical_record_padding_bytes": records["padding_bytes"],
            "canonical_record_total_bytes": sum(records["record_lengths"]),
            "candidate_shared_dram_bytes": (
                PHASES * WEIGHT_PHASE_BYTES + sum(records["record_lengths"])),
            "bit_sparse_shared_dram_bytes": PHASES * WEIGHT_PHASE_BYTES,
            "mandatory_full_row_writes": PHASES * ROWS,
            "minimum_terminal_rows": sum(records["terminal_rows"]),
            "explicit_zero_tail_row_writes": (
                PHASES * ROWS - sum(records["terminal_rows"])),
        },
        "five_sample": {
            "candidate_shared_dram_bytes": 5 * (
                PHASES * WEIGHT_PHASE_BYTES + sum(records["record_lengths"])),
            "bit_sparse_shared_dram_bytes": 5 * PHASES * WEIGHT_PHASE_BYTES,
            "candidate_full_row_writes": 5 * PHASES * ROWS,
        },
        "double_count_audit": {
            "record_bytes": (
                "charged once per phase on shared DRAM; includes 48B header, "
                "numeric payload and zero alignment padding"),
            "weight_bytes": (
                "charged once per phase on DRAM preparation; per-event weight "
                "service later is an on-chip SRAM/datapath cycle, not the same transfer"),
            "metadata_74B": (
                "48B code header is already inside the record; 26B pattern-base "
                "portion is derived by the 128-entry parser and held per buffer"),
            "row_writes": (
                "all 460 rows per phase are charged on an independent one-row/cycle "
                "writer through max(DRAM,row-write,parser), not added serially"),
            "assessment": "NO_CORE_LEDGER_DOUBLE_COUNT_OR_OMISSION_FOUND",
        },
    }
    require(accounting["five_sample"]["candidate_shared_dram_bytes"] == 225588320,
            "candidate DRAM byte denominator drift")
    require(accounting["five_sample"]["bit_sparse_shared_dram_bytes"] == 106168320,
            "baseline DRAM byte denominator drift")

    findings = [
        {
            "severity": "P1",
            "id": "M88-P1-01-DOUBLE-BUFFER-DMA-PARSER-WRITER-NOT-RTL",
            "finding": (
                "The max(DRAM,460-row-writer,128-parser)+commit preparation is "
                "causally feasible and fully charged, but M86 contains one bank image; "
                "its testbench supplies preformatted 256-bit rows and external 74B metadata."),
            "impact": (
                "M88 closes the bounded cycle-model gap, not the physical ping-pong DMA/"
                "record-parser/unpacker implementation or its timing/backpressure."),
        },
        {
            "severity": "P1",
            "id": "M88-P1-02-ALWAYS-READY-AND-ESCAPE-CONSUMER-ASSUMPTION",
            "finding": (
                "The 3/4/4/5 service is applied to all heldout work as an always-ready "
                "throughput model. M86's 128 backpressured phases do not admit those "
                "intervals, and its sole escape still returns a zero placeholder."),
            "impact": (
                "Zero midstream_load_stall_phases means zero ping-pong refill stalls, "
                "not zero FIFO/output/accumulator/correction/fallback stalls."),
        },
        {
            "severity": "P1",
            "id": "M88-P1-03-BANDWIDTH-FAIR-NOT-EQUAL-AREA-BASELINE",
            "finding": (
                "Candidate and bit-sparse baseline use the same 32B/cycle DRAM and "
                "three-cycle 96B weight service, and both reload every phase. Candidate "
                "additionally receives 113.8KiB listed local structures plus matcher, "
                "packer and frontend logic."),
            "impact": (
                "The 1.409375695x comparison is workload/bandwidth-fair, but not yet an "
                "equal-area, equal-power, or same-macro Synopsys comparison."),
        },
        {
            "severity": "P1",
            "id": "M88-P1-04-113P8KIB-IS-PARTIAL-STORAGE-NOT-MODULE-AREA",
            "finding": (
                "116,525 bytes exactly covers two PWP images+metadata, two weight images, "
                "one response FIFO, pattern table and offset table. It excludes activation/"
                "descriptor queues, matcher/packer state, accumulators, correction/fallback "
                "routing, parser logic, ECC, macro padding and bank-select control."),
            "impact": "113.8KiB is a scoped storage subtotal, not total module SRAM/area.",
        },
        {
            "severity": "P1",
            "id": "M88-P1-05-VALID825-INTERNAL-NO-ACCURACY-OR-SYSTEM-COMPOSITION",
            "finding": (
                "Centers use valid825 samples 0-4 and replay uses samples 5-9 of the same "
                "internal cohort; no train-only catalog, sequence-disjoint test, hardware-"
                "order accuracy, or full-network composition is present."),
            "impact": (
                "The scoped module estimate is reproducible, but cannot support an "
                "accuracy-preserving, full-network, DATE, or best-paper speedup claim."),
        },
        {
            "severity": "P2",
            "id": "M88-P2-01-COLD-PRELOAD-AND-READ-PORTS-UNMODELED",
            "finding": (
                "Pattern and phase-offset tables are counted in storage but their cold "
                "preload, matcher table read ports, arbitration and energy are not charged."),
            "impact": "Warm-resident cycle estimates remain valid; cold latency/energy does not.",
        },
        {
            "severity": "P2",
            "id": "M88-P2-02-FULL-460-ROW-TAIL-OVERHEAD",
            "finding": (
                "The model correctly charges 794,880 row writes per catalog pass although "
                "only 743,731 rows contain record words; 51,149 writes (6.88%) are explicit "
                "zero tail rows."),
            "impact": "Correct but avoidable loader work; terminal-aware implicit zero could save it.",
        },
        {
            "severity": "P2",
            "id": "M88-P2-03-M86-RUN-INPUT-MANIFEST-NOT-EXPECTED-SHA-GATED-IN-ANALYZER",
            "finding": (
                "M88 pins the M86 contract but only status/count strings from RUN_COMPLETE; "
                "it records, rather than pre-gates, M86 run/input-manifest SHA values."),
            "impact": (
                "The current result freezes the observed hashes, but future reruns could "
                "accept a different similarly worded sealed run unless externally gated."),
        },
    ]

    review = {
        "schema": "m88_bounded_sync_bank_double_buffer_independent_hammer_v1",
        "status": "GO_SCOPED_VALID825_INTERNAL_MODULE_CYCLE_ESTIMATE_NO_GO_SYSTEM_DATE_HEADLINE",
        "identity": identity,
        "independence": {
            "production_m88_imported": False,
            "production_m78_imported": False,
            "m83_binary_records_directly_decoded": True,
            "m78_work_source": "sealed independent M41/M72/M40 reconstruction",
            "producer_files_modified": False,
        },
        "record_and_frontend_reconstruction": {
            "phases": PHASES,
            "codes": records["code_histogram"],
            "bank_read_issues_including_escape": records["beat_issues"],
            "record_bytes": sum(records["record_lengths"]),
            "numeric_payload_bytes": sum(records["payload_bytes"]),
            "header_bytes": PHASES * 48,
            "padding_bytes": records["padding_bytes"],
            "record_length_min": min(records["record_lengths"]),
            "record_length_max": max(records["record_lengths"]),
            "prepare_cycles_min": min(records["new_preparations"]),
            "prepare_cycles_max": max(records["new_preparations"]),
            "terminal_rows": sum(records["terminal_rows"]),
            "full_row_writes": PHASES * ROWS,
            "zero_tail_row_writes": PHASES * ROWS - sum(records["terminal_rows"]),
            "preparation_delta_vs_m78_histogram": (
                records["preparation_delta_histogram"]),
            "mismatches": 0,
        },
        "bounded_timeline_reconstruction": timeline,
        "traffic_and_double_count_audit": accounting,
        "storage_reconstruction": storage,
        "baseline_fairness": {
            "same_heldout_work": True,
            "same_shared_dram_bytes_per_cycle": 32,
            "same_weight_vector_service_cycles": 3,
            "same_phase_weight_reload_policy": True,
            "candidate_record_bytes_charged": True,
            "finite_two_slot_schedule_applied_to_both": True,
            "equal_area_or_power": False,
            "assessment": "FAIR_FOR_SAME_BANDWIDTH_MODULE_CYCLE_DSE_ONLY",
        },
        "findings": findings,
        "severity_counts": {
            "P0": sum(item["severity"] == "P0" for item in findings),
            "P1": sum(item["severity"] == "P1" for item in findings),
            "P2": sum(item["severity"] == "P2" for item in findings),
        },
        "scores": {
            "hardware_innovation": 68,
            "performance_advantage": 73,
            "evidence_quality": 91,
            "scoped_milestone_completeness": 92,
            "date_paper_completeness": 58,
            "overall_scoped_milestone": 84,
        },
        "claim_boundary": {
            "go": [
                "Exact valid825-internal five-sample isolated module cycle estimate",
                "1.409375695x aggregate versus the same-bandwidth bit-sparse model",
                "zero midstream phase-refill stalls under the frozen always-ready compute model",
                "116525-byte (113.793945KiB) listed local-storage subtotal",
            ],
            "no_go": [
                "zero total datapath/FIFO/backpressure stalls",
                "implemented RTL double-buffer DMA/parser/row-writer",
                "equal-area/equal-power speedup or physical SRAM macro PPA/energy",
                "train-catalog, accuracy-preserving, full-network/system speedup",
                "DATE or best-paper headline",
            ],
        },
        "verdict": {
            "m88_scoped_module_cycle_estimate": "GO",
            "m78_1p409x_readmission_scope": "GO_VALID825_INTERNAL_MODEL_ONLY",
            "rtl_executable_module_speedup": "NO_GO",
            "system_or_paper_performance": "NO_GO",
        },
    }
    output = HERE / "m88_independent_hammer_review.json"
    output.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M88 independent candidate={} baseline={} speedup={:.9f}x "
          "stalls=0 storage={} P0={} P1={} P2={}".format(
              timeline["aggregate"]["bounded_candidate_cycles"],
              timeline["aggregate"]["bounded_bit_sparse_cycles"],
              timeline["aggregate"]["speedup_vs_bit_sparse"],
              storage["total_bytes"], review["severity_counts"]["P0"],
              review["severity_counts"]["P1"],
              review["severity_counts"]["P2"]), flush=True)


if __name__ == "__main__":
    main()
