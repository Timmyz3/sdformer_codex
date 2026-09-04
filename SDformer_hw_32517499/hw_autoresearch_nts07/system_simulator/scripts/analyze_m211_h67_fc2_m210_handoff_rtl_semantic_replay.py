#!/usr/bin/env python3
"""Replay every frozen H67 FC2 token with the VCS-calibrated M210 recurrence."""

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPECTED_MANIFEST = "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e"
EXPECTED_M172 = "ae633daee1e07f16396570c1ef757c11bac7f1f72be108f4411d75f6dcb5f6d9"
EXPECTED_M192 = "39715b64890d75be7c60587d639f3e75b51e5bf38642e6b8640c761fa512f24b"
EXPECTED_M209_RESULT = "c87030245d7fccf444e6403a68f9098d7b37701f8d8ec96834129b41057ed576"
EXPECTED_M210_MODEL = "40a43c1a86f67ca58c6e59440e4c2a54d066d7f44e5546be312a4144560e387d"
EXPECTED_M202_RTL = "eb9f42ffd4286a4f5c83436acdad30568ddd6e7d90510e725d210a9a35677354"
EXPECTED_M210_SINK_RTL = "69ad410b3860ece667fdf3ed4c32584c3149633579c65ba6eb3ea5155eeaa929"
EXPECTED_M210_TOP_RTL = "e0e5ad5667f133344671bbd88e4da3d40abd6752a135c54788e93e3aac4fb721"
EXPECTED_DOCS359 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
EXPECTED_RECORDS = 120
EXPECTED_TOKENS = 5580000
EXPECTED_EVENTS = 143894510
EXPECTED_RAW_BEATS = 36480000
EXPECTED_DESCRIPTORS = 18869376
EXPECTED_WINDOWS = 6523707

M172 = None
M192 = None
M210 = None
PAYLOAD_ROOT = None
CHUNK_TOKENS = None


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path, expected, name):
    require(sha256(path) == expected, name + " identity drift")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fraction(numerator, denominator):
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "float": float(numerator) / float(denominator),
    }


def empty_ledger():
    return {
        "records": 0, "tokens": 0, "events": 0, "raw96_beats": 0,
        "nonzero96_descriptors": 0, "windows": 0, "zero_tokens": 0,
        "full_final_tokens": 0, "partial_final_tokens": 0,
        "m210_rtl_semantic_cycles": 0, "terminal_collapses": 0,
        "tokens_with_terminal_collapse": 0,
        "maximum_m202_queue": 0, "maximum_descriptor_hold": 0,
    }


def merge(target, source):
    for key, value in source.items():
        if key in ("maximum_m202_queue", "maximum_descriptor_hold"):
            target[key] = max(target[key], value)
        else:
            target[key] += value


def audit_record(task):
    ordinal, record = task
    shape = [int(value) for value in record["input_shape"]]
    output_shape = [int(value) for value in record["output_shape"]]
    stage = int(record["name"].split(".layers.")[1].split(".")[0])
    input_width, output_width, depth = M192.STAGE_GEOMETRY[stage]
    require((shape[-1], output_shape[-1]) == (input_width, output_width),
            "FC2 geometry drift")
    output_blocks = output_width // 96
    beats_per_token = input_width // 96
    bytes_per_token = input_width // 8
    tokens = int(np.prod(shape[:-1], dtype=np.int64))
    payload = PAYLOAD_ROOT / record["relative_path"]
    require(payload.is_file() and payload.stat().st_size == record["packed_bytes"],
            "payload extent drift")
    require(sha256(payload) == record["file_sha256"], "payload SHA drift")
    raw = np.memmap(payload, dtype=np.uint8, mode="r").reshape(
        tokens, beats_per_token, bytes_per_token // beats_per_token)
    ledger = empty_ledger()
    ledger["records"] = 1
    ledger["tokens"] = tokens
    ledger["raw96_beats"] = tokens * beats_per_token
    zero_cycles = beats_per_token // 4 + 2

    for start in range(0, tokens, CHUNK_TOKENS):
        stop = min(tokens, start + CHUNK_TOKENS)
        byte_bits = M172.BYTE_BITS[np.asarray(raw[start:stop])]
        bank_counts = byte_bits.sum(axis=2, dtype=np.int16)
        beat_events = bank_counts.sum(axis=2, dtype=np.int16)
        descriptor_counts = (beat_events != 0).sum(axis=1, dtype=np.int16)
        ledger["events"] += int(beat_events.sum(dtype=np.int64))
        ledger["nonzero96_descriptors"] += int(
            descriptor_counts.sum(dtype=np.int64))
        ledger["windows"] += int(
            ((descriptor_counts.astype(np.int64) + depth - 1) // depth)
            .sum(dtype=np.int64))
        for token_offset, count_value in enumerate(descriptor_counts):
            descriptor_count = int(count_value)
            if descriptor_count == 0:
                ledger["zero_tokens"] += 1
                ledger["m210_rtl_semantic_cycles"] += zero_cycles
                continue
            if descriptor_count % depth:
                ledger["partial_final_tokens"] += 1
            else:
                ledger["full_final_tokens"] += 1
            loads = [
                tuple(int(value) for value in bank_counts[token_offset, beat])
                if beat_events[token_offset, beat] else None
                for beat in range(beats_per_token)
            ]
            measured = M210.simulate_m210_bank_loads(
                loads, depth, output_blocks)
            ledger["m210_rtl_semantic_cycles"] += measured["cycles"]
            ledger["terminal_collapses"] += measured["terminal_collapses"]
            ledger["tokens_with_terminal_collapse"] += int(
                measured["terminal_collapses"] != 0)
            ledger["maximum_m202_queue"] = max(
                ledger["maximum_m202_queue"], measured["maximum_queue"])
            ledger["maximum_descriptor_hold"] = max(
                ledger["maximum_descriptor_hold"],
                measured["maximum_descriptor_hold"])
    require(ledger["events"] == int(record["active_elements"]),
            "payload popcount drift")
    return ordinal, stage, ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--m172-analyzer", required=True, type=Path)
    parser.add_argument("--m192-analyzer", required=True, type=Path)
    parser.add_argument("--m209-result", required=True, type=Path)
    parser.add_argument("--m210-model", required=True, type=Path)
    parser.add_argument("--m210-validation", required=True, type=Path)
    parser.add_argument("--m202-rtl", required=True, type=Path)
    parser.add_argument("--m210-sink-rtl", required=True, type=Path)
    parser.add_argument("--m210-top-rtl", required=True, type=Path)
    parser.add_argument("--docs359", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-tokens", type=int, default=8192)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite output")
    script_start = sha256(Path(__file__).resolve())
    require(sha256(args.manifest) == EXPECTED_MANIFEST, "manifest drift")
    require(sha256(args.m209_result) == EXPECTED_M209_RESULT, "M209 drift")
    require(sha256(args.m202_rtl) == EXPECTED_M202_RTL, "M202 RTL drift")
    require(sha256(args.m210_sink_rtl) == EXPECTED_M210_SINK_RTL,
            "M210 sink RTL drift")
    require(sha256(args.m210_top_rtl) == EXPECTED_M210_TOP_RTL,
            "M210 top RTL drift")
    require(sha256(args.docs359) == EXPECTED_DOCS359, "docs359 drift")
    with args.m210_validation.open("r", encoding="utf-8") as handle:
        validation = json.load(handle)
    require(validation["status"] == "PASS_EXACT_256_CASE_VCS"
            and validation["cases"] == 256
            and validation["mismatches"] == 0,
            "M210 VCS calibration not admitted")

    global M172, M192, M210, PAYLOAD_ROOT, CHUNK_TOKENS
    M172 = load_module(args.m172_analyzer, EXPECTED_M172, "m172_pinned_m211")
    M192 = load_module(args.m192_analyzer, EXPECTED_M192, "m192_pinned_m211")
    M210 = load_module(args.m210_model, EXPECTED_M210_MODEL, "m210_pinned_m211")
    PAYLOAD_ROOT = args.payload_root
    CHUNK_TOKENS = args.chunk_tokens
    with args.manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    records = [record for record in manifest["records"]
               if record["operator"] == "Linear"
               and ".mlp.fc2" in record["name"]]
    require(len(records) == EXPECTED_RECORDS, "FC2 record count drift")

    tasks = list(enumerate(records))
    if args.workers == 1:
        audited = [audit_record(task) for task in tasks]
    else:
        context = multiprocessing.get_context("fork")
        with context.Pool(processes=args.workers) as pool:
            audited = list(pool.imap_unordered(audit_record, tasks))
    audited.sort(key=lambda item: item[0])
    aggregate = empty_ledger()
    per_stage = defaultdict(empty_ledger)
    for count, (_ordinal, stage, ledger) in enumerate(audited, start=1):
        merge(aggregate, ledger)
        merge(per_stage[stage], ledger)
        print("[M211] {}/120".format(count), flush=True)

    require(aggregate["tokens"] == EXPECTED_TOKENS, "token identity drift")
    require(aggregate["events"] == EXPECTED_EVENTS, "event identity drift")
    require(aggregate["raw96_beats"] == EXPECTED_RAW_BEATS,
            "raw identity drift")
    require(aggregate["nonzero96_descriptors"] == EXPECTED_DESCRIPTORS,
            "descriptor identity drift")
    require(aggregate["windows"] == EXPECTED_WINDOWS, "window identity drift")
    require(aggregate["zero_tokens"] + aggregate["full_final_tokens"]
            + aggregate["partial_final_tokens"] == EXPECTED_TOKENS,
            "tail census drift")
    with args.m209_result.open("r", encoding="utf-8") as handle:
        m209 = json.load(handle)
    analytic = m209["comparison"]["m203_analytic_stage_aware_cycles"]
    m209_cycles = m209["aggregate"]["m207_rtl_semantic_cycles"]
    legacy_baseline = m209["comparison"][
        "legacy_s1_f1_w1_analytic_cycles"]
    exact_cycles = aggregate["m210_rtl_semantic_cycles"]
    result = {
        "schema": "m211_h67_fc2_m210_rtl_semantic_replay_v1",
        "status": "PASS_EXACT_FROZEN_PAYLOAD_VCS_CALIBRATED_M210_RECURRENCE",
        "identity": {
            "analyzer_start_sha256": script_start,
            "manifest_sha256": EXPECTED_MANIFEST,
            "m172_analyzer_sha256": EXPECTED_M172,
            "m192_analyzer_sha256": EXPECTED_M192,
            "m209_result_sha256": EXPECTED_M209_RESULT,
            "m210_model_sha256": EXPECTED_M210_MODEL,
            "m210_validation_sha256": sha256(args.m210_validation),
            "m202_rtl_sha256": EXPECTED_M202_RTL,
            "m210_sink_rtl_sha256": EXPECTED_M210_SINK_RTL,
            "m210_top_rtl_sha256": EXPECTED_M210_TOP_RTL,
            "docs359_sha256": EXPECTED_DOCS359,
        },
        "architecture": {
            "raw_scan_width": 4,
            "descriptor_emit_width": 4,
            "descriptor_queue_depth": 8,
            "window_buffers": 2,
            "paired_windows_for_stages_1_to_3": True,
            "stage0_single_window": True,
            "stage0_adjacent_window_handoff_prefetch": True,
            "fixed_output_banks": 8,
            "terminal_group_token_collapse": True,
            "token_header_chain_vcs_proven": True,
            "isolated_token_ready_outputs_high": True,
        },
        "aggregate": aggregate,
        "per_stage": {str(key): value
                      for key, value in sorted(per_stage.items())},
        "comparison": {
            "m203_analytic_stage_aware_cycles": analytic,
            "m210_rtl_semantic_cycles": exact_cycles,
            "m209_m207_rtl_semantic_cycles": m209_cycles,
            "cycles_saved_vs_m209": m209_cycles - exact_cycles,
            "m210_speed_vs_m209": fraction(m209_cycles, exact_cycles),
            "m210_over_m203_analytic_cycle_factor": fraction(
                exact_cycles, analytic),
            "m203_analytic_over_m210_factor": fraction(
                analytic, exact_cycles),
            "legacy_s1_f1_w1_analytic_cycles": legacy_baseline,
            "legacy_analytic_baseline_over_m210_factor": fraction(
                legacy_baseline, exact_cycles),
        },
        "claim_boundary": {
            "exact_payload_identity": True,
            "synopsys_vcs_calibrated_control_recurrence": True,
            "all_frozen_h67_fc2_tokens_replayed": True,
            "isolated_token_cycle_model": True,
            "matched_rtl_baseline_speedup": False,
            "complete_fc2": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    require(sha256(Path(__file__).resolve()) == script_start,
            "analyzer mutated during run")
    args.output.parent.mkdir(parents=True, exist_ok=False)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result["comparison"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
