#!/usr/bin/env python3
"""Census M205 terminal control bubbles on the frozen H67 FC2 payload."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPECTED_MANIFEST = "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e"
EXPECTED_M203_RESULT = "fd8d03eebc760cdba125e4d008a95a9eb8e0ee41abd6c665df8ca357761120df"
EXPECTED_M205_RTL = "17dd8458bcdd4f888e46a9425cdec4b52988c6b1931e10a639f299d162ead467"
EXPECTED_M204_DC = "cb7ecbf6e9a3e171c3fd4ccb262c089d34f921c2b8126e9c9d32e82409aaa4dd"
EXPECTED_M205_VCS = "f19fa092d9d39448505335b6eaef04f1bfe6a2e1947a94fa64a170379cb43c27"
EXPECTED_DOCS359 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
EXPECTED_RECORDS = 120
EXPECTED_TOKENS = 5_580_000
EXPECTED_DESCRIPTORS = 18_869_376
EXPECTED_WINDOWS = 6_523_707
STAGE_DEPTH = {0: 2, 1: 4, 2: 8, 3: 8}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fraction(numerator, denominator):
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "float": float(numerator) / float(denominator),
    }


def empty_ledger():
    return {
        "records": 0,
        "tokens": 0,
        "zero_tokens": 0,
        "nonzero_full_final_window_tokens": 0,
        "nonzero_partial_final_window_tokens": 0,
        "nonzero_descriptors": 0,
        "compact_windows": 0,
        "calibrated_terminal_control_cycles": 0,
    }


def merge(target, source):
    for key, value in source.items():
        target[key] += int(value)


def audit_record(record, payload_root, chunk_tokens):
    stage = int(record["name"].split(".layers.")[1].split(".")[0])
    depth = STAGE_DEPTH[stage]
    shape = [int(value) for value in record["input_shape"]]
    tokens = int(np.prod(shape[:-1], dtype=np.int64))
    beats = shape[-1] // 96
    payload = payload_root / record["relative_path"]
    require(payload.is_file(), "payload missing")
    require(payload.stat().st_size == int(record["packed_bytes"]),
            "payload extent drift")
    require(sha256(payload) == record["file_sha256"], "payload SHA drift")
    raw = np.memmap(payload, dtype=np.uint8, mode="r").reshape(
        tokens, beats, 12
    )
    ledger = empty_ledger()
    ledger["records"] = 1
    ledger["tokens"] = tokens
    for start in range(0, tokens, chunk_tokens):
        stop = min(tokens, start + chunk_tokens)
        nonzero = np.any(np.asarray(raw[start:stop]) != 0, axis=2)
        descriptor_count = nonzero.sum(axis=1, dtype=np.int16)
        zero = descriptor_count == 0
        full = (descriptor_count > 0) & (descriptor_count % depth == 0)
        partial = (descriptor_count > 0) & (descriptor_count % depth != 0)
        zero_count = int(np.count_nonzero(zero))
        full_count = int(np.count_nonzero(full))
        partial_count = int(np.count_nonzero(partial))
        ledger["zero_tokens"] += zero_count
        ledger["nonzero_full_final_window_tokens"] += full_count
        ledger["nonzero_partial_final_window_tokens"] += partial_count
        ledger["nonzero_descriptors"] += int(
            descriptor_count.sum(dtype=np.int64)
        )
        ledger["compact_windows"] += int(
            ((descriptor_count.astype(np.int64) + depth - 1) // depth).sum()
        )
        # Continuous-source VCS calibration of the current M205 composition:
        # a naturally full terminal window is one cycle beyond finite_wall;
        # an upstream_done-closed partial tail is two cycles beyond it; a zero
        # token already matches finite_wall.  This is an optimization census,
        # not yet an admitted RTL speed result.
        ledger["calibrated_terminal_control_cycles"] += (
            full_count + 2 * partial_count
        )
    return stage, ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--m203-result", required=True, type=Path)
    parser.add_argument("--m205-rtl", required=True, type=Path)
    parser.add_argument("--m204-dc-run-complete", required=True, type=Path)
    parser.add_argument("--m205-vcs-run-complete", required=True, type=Path)
    parser.add_argument("--docs359", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-tokens", type=int, default=65536)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite output")
    script_start = sha256(Path(__file__).resolve())
    require(sha256(args.manifest) == EXPECTED_MANIFEST, "manifest drift")
    require(sha256(args.m203_result) == EXPECTED_M203_RESULT, "M203 drift")
    require(sha256(args.m205_rtl) == EXPECTED_M205_RTL, "M205 RTL drift")
    require(sha256(args.m204_dc_run_complete) == EXPECTED_M204_DC,
            "M204 DC drift")
    require(sha256(args.m205_vcs_run_complete) == EXPECTED_M205_VCS,
            "M205 VCS drift")
    require(sha256(args.docs359) == EXPECTED_DOCS359, "docs359 drift")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    m203 = json.loads(args.m203_result.read_text(encoding="utf-8"))
    records = [record for record in manifest["records"]
               if record["operator"] == "Linear"
               and ".mlp.fc2" in record["name"]]
    require(len(records) == EXPECTED_RECORDS, "FC2 record extent drift")
    aggregate = empty_ledger()
    per_stage = defaultdict(empty_ledger)
    for ordinal, record in enumerate(records):
        stage, ledger = audit_record(record, args.payload_root,
                                     args.chunk_tokens)
        merge(aggregate, ledger)
        merge(per_stage[stage], ledger)
        print("[M206] {}/{}".format(ordinal + 1, EXPECTED_RECORDS),
              flush=True)
    require(aggregate["tokens"] == EXPECTED_TOKENS, "token drift")
    require(aggregate["nonzero_descriptors"] == EXPECTED_DESCRIPTORS,
            "descriptor drift")
    require(aggregate["compact_windows"] == EXPECTED_WINDOWS,
            "window drift")
    require(aggregate["zero_tokens"]
            == m203["aggregate"]["zero_tokens"], "zero-token crosscheck")
    for stage, ledger in per_stage.items():
        old = m203["per_stage"][str(stage)]
        require(ledger["tokens"] == old["tokens"], "stage token drift")
        require(ledger["zero_tokens"] == old["zero_tokens"],
                "stage zero drift")
        require(ledger["nonzero_descriptors"]
                == old["nonzero96_descriptors"], "stage descriptor drift")
        require(ledger["compact_windows"] == old["windows"],
                "stage window drift")
    correction = aggregate["calibrated_terminal_control_cycles"]
    analytic = m203["comparison"]["m202_stage_aware_cycles"]
    baseline = m203["comparison"]["baseline_s1_f1_w1_cycles"]
    corrected = analytic + correction
    matched_baseline = baseline + correction
    result = {
        "schema": "m206_h67_fc2_m205_terminal_control_bubble_census_v1",
        "status": "PASS_EXACT_PAYLOAD_CONTROL_OPPORTUNITY__ADMISSION_PENDING",
        "identity": {
            "analyzer_start_sha256": script_start,
            "manifest_sha256": EXPECTED_MANIFEST,
            "m203_result_sha256": EXPECTED_M203_RESULT,
            "m205_rtl_sha256": EXPECTED_M205_RTL,
            "m204_dc_run_complete_sha256": EXPECTED_M204_DC,
            "m205_vcs_run_complete_sha256": EXPECTED_M205_VCS,
            "docs359_sha256": EXPECTED_DOCS359,
        },
        "calibration_hypothesis": {
            "continuous_source_and_always_ready_group_sink": True,
            "full_final_window_cycles_beyond_m203_finite_wall": 1,
            "partial_final_window_cycles_beyond_m203_finite_wall": 2,
            "zero_token_cycles_beyond_m203_finite_wall": 0,
            "independent_exhaustive_validation_pending": True,
        },
        "aggregate": aggregate,
        "per_stage": {str(stage): ledger
                      for stage, ledger in sorted(per_stage.items())},
        "comparison": {
            "m203_analytic_stage_aware_cycles": analytic,
            "calibrated_m205_control_cycles": correction,
            "calibrated_m205_stage_aware_cycles": corrected,
            "legacy_abstract_baseline_cycles": baseline,
            "matched_control_baseline_cycles": matched_baseline,
            "matched_control_speed": fraction(matched_baseline, corrected),
            "conservative_legacy_baseline_over_calibrated_m205":
                fraction(baseline, corrected),
            "recoverable_to_m203_if_terminal_control_collapses": correction,
        },
        "optimization_target": {
            "same_cycle_partial_tail_close": True,
            "same_cycle_final_group_token_done": True,
            "same_cycle_next_header_chain": True,
            "zero_token_terminal_bypass": "not_yet_credited",
        },
        "claim_boundary": {
            "exact_payload_token_class_census": True,
            "calibration_rule_independently_admitted": False,
            "measured_frozen_payload_rtl_cycles": False,
            "physical_speedup": False,
            "complete_fc2": False,
            "ffn_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(result["comparison"], sort_keys=True))


if __name__ == "__main__":
    main()
