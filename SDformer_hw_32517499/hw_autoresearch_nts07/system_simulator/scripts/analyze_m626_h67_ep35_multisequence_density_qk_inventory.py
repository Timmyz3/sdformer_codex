#!/usr/bin/env python3
"""M626: fail-closed H67 ep35 multi-sequence density/QK evidence inventory.

This is an evidence availability and stratification audit.  It does not model
cycles, energy, PPA, or full-network speedup.  The only raw tensor replay here
is exact CPU verification of the 1,200 packaged Q/K attention NPZ files.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
BASE = REPO / "hw_autoresearch_nts07"
HANDOFF = (
    BASE
    / "system_handoff/received/h67_ep35_system_trace_handoff_20260821"
    / "h67_ep35_system_trace_handoff_20260821"
)
CONTRACT = BASE / "contracts/m626_h67_ep35_multisequence_density_qk_cpu_replay_contract_r1_20260828.json"
DOCS359 = BASE / "docs/359_DATE终局冻结_20260813.md"
TRACE_MANIFEST = HANDOFF / "trace_qk_100sample_12block/manifest.relocated.json"
HANDOFF_MANIFEST = HANDOFF / "handoff_manifest.json"
SAMPLE_WORKLOAD = HANDOFF / "profile100/sample_workload.csv"
ACTIVATION_RECORDS = HANDOFF / "profile100/activation_records.csv"
M51_MANIFEST = BASE / "system_handoff/incoming/m51_capture_bundle_r2_20260823/manifest.json"
M460_RESULT = BASE / "results/m460r5_h67_g8_one_shot_s10_r1_20260826/capture_payload/m460_h67_g8_ffn_token_residual_s10_capture.json"
M511_CONTRACT = BASE / "contracts/m511_h67_ep35_convtranspose_binary_input_capture_contract_r1_20260827.json"
M515_RESULT = BASE / "results/m515_atlif_state_boundary_audit_r2_20260827/m515_atlif_state_boundary_audit_r2.json"
ORDERED_TRACE = BASE / "results/h67_ep35_full_network_ordered_trace_s10_20260821/execution_trace.csv"

EXPECTED_DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
EXPECTED_CHECKPOINT_SHA256 = "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158"
EXPECTED_CONFIG_SHA256 = "8be3f7bbffd75c4356d3abf5935679d80e15c1caefd307c19a727729659e6c49"
EXPECTED_TRACE_MANIFEST_SHA256 = "e178e7caa494926c4c1232ac1f2551be665be485cbeb6ed1683007fea3545f87"
EXPECTED_SAMPLE_WORKLOAD_SHA256 = "68da0e8e1e46e6196ecec2bc2467a664d4dad8b6894e3e4f4e95dfe737178cf2"
EXPECTED_HANDOFF_MANIFEST_SHA256 = "c888bb28ca5a5b8324d22fefd052b2d850a22d1b7f164684606351a8a62a87b1"

# Fixed bins, chosen before reading sample values.  The last interval is closed
# at one; all other intervals are left-closed/right-open.
DENSITY_BINS = [
    ("D0_[0.00,0.20)", 0.00, 0.20),
    ("D1_[0.20,0.25)", 0.20, 0.25),
    ("D2_[0.25,0.30)", 0.25, 0.30),
    ("D3_[0.30,0.35)", 0.30, 0.35),
    ("D4_[0.35,0.40)", 0.35, 0.40),
    ("D5_[0.40,1.00]", 0.40, 1.0000000001),
]


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite_mean(values):
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def density_bin(value):
    for label, lo, hi in DENSITY_BINS:
        if lo <= value < hi:
            return label
    raise RuntimeError("density outside frozen bins: {}".format(value))


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def verify_static_identities(contract):
    expected = {
        DOCS359: EXPECTED_DOCS359_SHA256,
        HANDOFF_MANIFEST: EXPECTED_HANDOFF_MANIFEST_SHA256,
        TRACE_MANIFEST: EXPECTED_TRACE_MANIFEST_SHA256,
        SAMPLE_WORKLOAD: EXPECTED_SAMPLE_WORKLOAD_SHA256,
        HANDOFF / "checkpoint/checkpoint_epoch35.pth": EXPECTED_CHECKPOINT_SHA256,
        HANDOFF / "config/deploy_q7q17.yml": EXPECTED_CONFIG_SHA256,
        SCRIPT: contract["inputs"]["analyzer"]["sha256"],
    }
    receipt = []
    for path, wanted in expected.items():
        require(path.is_file(), "missing required identity: {}".format(path))
        got = sha256_file(path)
        require(got == wanted, "SHA mismatch {}: {} != {}".format(path, got, wanted))
        receipt.append({"path": str(path.relative_to(REPO)), "sha256": got, "bytes": path.stat().st_size})
    return receipt


def verify_entire_handoff(handoff_manifest):
    files = handoff_manifest["files"]
    require(len(files) == 1230, "handoff member count changed")
    total = 0
    verified = []
    for idx, item in enumerate(files):
        path = HANDOFF / item["path"]
        require(path.is_file(), "handoff member missing: {}".format(item["path"]))
        size = path.stat().st_size
        require(size == int(item["size"]), "size mismatch: {}".format(item["path"]))
        got = sha256_file(path)
        require(got == item["sha256"], "handoff SHA mismatch: {}".format(item["path"]))
        total += size
        if idx < 8 or item["path"].endswith("manifest.relocated.json"):
            verified.append({"path": item["path"], "sha256": got, "bytes": size})
    require(total == int(handoff_manifest["total_bytes"]), "handoff total_bytes mismatch")
    return {"members_verified": len(files), "bytes_verified": total, "identity_sample": verified}


def popcount_packed(array, logical_bits):
    require(array.dtype == np.uint8, "packed tensor must be uint8")
    require(array.ndim == 1, "packed tensor must be flat")
    require(array.size * 8 >= logical_bits, "packed tensor shorter than logical shape")
    require(array.size * 8 - logical_bits < 8, "packed tensor has more than one byte of padding")
    table = np.asarray([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    if logical_bits % 8 == 0:
        return int(table[array].sum(dtype=np.uint64))
    bits = np.unpackbits(array, bitorder="little")[:logical_bits]
    return int(bits.sum(dtype=np.uint64))


def raw_payload_coverage(samples):
    sample_sets = defaultdict(set)
    details = {}

    # M51 exact binary Conv/FC inputs: manifest says 310 records, but this local
    # incoming directory is intentionally checked for physically present bytes.
    m51 = read_json(M51_MANIFEST)
    m51_root = M51_MANIFEST.parent
    present = []
    missing = []
    present_by_op = Counter()
    missing_by_op = Counter()
    for rec in m51["records"]:
        path = m51_root / rec["relative_path"]
        if path.is_file():
            present.append(rec)
            present_by_op[rec["operator"]] += 1
            sample_sets["raw_conv_fc_bitpack"].add(rec["sample_key"])
        else:
            missing.append(rec)
            missing_by_op[rec["operator"]] += 1
    details["raw_conv_fc_bitpack"] = {
        "manifest_records": len(m51["records"]),
        "physical_records": len(present),
        "missing_records": len(missing),
        "physical_by_operator": dict(present_by_op),
        "missing_by_operator": dict(missing_by_op),
        "sequences": sorted(set(r["sequence_key"] for r in present)),
        "claim": m51["claim_boundary"],
    }

    m460 = read_json(M460_RESULT)
    details["raw_ffn_postcompute_npz"] = {
        "manifest_npz_payloads": int(m460["files"]["npz_payloads"]),
        "samples": int(m460["population"]["samples"]),
        "modules": int(m460["population"]["ffn_modules"]),
        "sequences": ["zurich_city_09_a"],
        "claim": m460["claim_boundary"],
    }
    for s in samples:
        if s["sequence_key"] == "zurich_city_09_a" and int(s["sample_id"]) < 10:
            sample_sets["raw_ffn_postcompute_npz"].add(s["sample_key"])

    m511 = read_json(M511_CONTRACT)
    details["raw_decoder_convtranspose_bitpack"] = {
        "capture_contract_present": True,
        "captured_result_present": False,
        "contract_samples": len(m511["samples"]),
        "contract_sequences": sorted(set(s["sequence_key"] for s in m511["samples"])),
        "claim": "contract only; no M511 output payload found locally",
    }

    m515 = read_json(M515_RESULT)
    details["raw_atlif_state_or_io_payload"] = {
        "state_boundary_audit_present": True,
        "raw_multisequence_payload_present": False,
        "trace_samples": int(m515["trace_population"]["samples"]),
        "trace_records": int(m515["trace_population"]["atlif_records"]),
        "sequences": ["zurich_city_09_a"],
        "claim": "state/accounting audit, not raw ATLIF I/O payload",
    }

    ordered_rows = read_csv(ORDERED_TRACE)
    ordered_sample_keys = set(r.get("sample_key", "") for r in ordered_rows if r.get("sample_key", ""))
    sample_sets["ordered_full_network_trace"].update(ordered_sample_keys)
    details["ordered_full_network_trace"] = {
        "records": len(ordered_rows),
        "sample_keys_with_explicit_identity": len(ordered_sample_keys),
        "sequences": sorted(set(r.get("sequence_key", "") for r in ordered_rows if r.get("sequence_key", ""))),
        "claim": "ordered operator/source-work trace; not address-timed cycles",
    }

    rows = []
    all_sequences = sorted(set(s["sequence_key"] for s in samples))
    by_sequence = Counter(s["sequence_key"] for s in samples)
    for seq in all_sequences:
        rows.extend([
            {"sequence_key": seq, "evidence": "event_density_and_sample_aee_summary", "samples_present": by_sequence[seq], "samples_expected": by_sequence[seq], "raw_payload": False, "direct_cpu_replay": False, "paper_use": "density/AEE stratification only"},
            {"sequence_key": seq, "evidence": "raw_attention_qk_npz", "samples_present": by_sequence[seq], "samples_expected": by_sequence[seq], "raw_payload": True, "direct_cpu_replay": True, "paper_use": "multi-sequence attention sparsity"},
            {"sequence_key": seq, "evidence": "profile_activation_summary_34_per_sample", "samples_present": by_sequence[seq], "samples_expected": by_sequence[seq], "raw_payload": False, "direct_cpu_replay": False, "paper_use": "summary-only; 26/34 rows lack explicit sample identity"},
        ])
        for evidence in ["raw_conv_fc_bitpack", "raw_ffn_postcompute_npz", "ordered_full_network_trace"]:
            present = sum(1 for s in samples if s["sequence_key"] == seq and s["sample_key"] in sample_sets[evidence])
            rows.append({"sequence_key": seq, "evidence": evidence, "samples_present": present, "samples_expected": by_sequence[seq], "raw_payload": evidence != "ordered_full_network_trace", "direct_cpu_replay": False, "paper_use": "single-sequence only" if present else "missing"})
        rows.append({"sequence_key": seq, "evidence": "raw_decoder_convtranspose_bitpack", "samples_present": 0, "samples_expected": by_sequence[seq], "raw_payload": True, "direct_cpu_replay": False, "paper_use": "missing; contract only"})
        rows.append({"sequence_key": seq, "evidence": "raw_atlif_state_or_io_payload", "samples_present": 0, "samples_expected": by_sequence[seq], "raw_payload": True, "direct_cpu_replay": False, "paper_use": "missing"})
    return rows, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    require(not output.exists(), "output already exists: {}".format(output))
    require(CONTRACT.is_file(), "M626 contract missing")
    contract = read_json(CONTRACT)
    identities = verify_static_identities(contract)

    handoff_manifest = read_json(HANDOFF_MANIFEST)
    require(handoff_manifest["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256, "handoff checkpoint identity drift")
    require(handoff_manifest["config_sha256"] == EXPECTED_CONFIG_SHA256, "handoff config identity drift")
    handoff_audit = verify_entire_handoff(handoff_manifest)

    trace = read_json(TRACE_MANIFEST)
    samples = read_csv(SAMPLE_WORKLOAD)
    activations = read_csv(ACTIVATION_RECORDS)
    require(len(samples) == 100, "sample workload must contain exactly 100 rows")
    require(len(trace["records"]) == 1200, "QK trace must contain exactly 1200 records")
    require(len(activations) == 3400, "activation summary must contain exactly 3400 rows")
    require(trace["run_context"]["artifact_identity"]["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256, "trace checkpoint drift")
    require(trace["run_context"]["artifact_identity"]["config_sha256"] == EXPECTED_CONFIG_SHA256, "trace config drift")
    load = trace["run_context"]["checkpoint_load_audit"]
    for key in ["missing_count", "unexpected_count", "overlay_missing_count", "overlay_unexpected_count"]:
        require(int(load[key]) == 0, "checkpoint load audit is not exact: {}".format(key))

    sample_by_id = {}
    for row in samples:
        sid = int(row["sample_id"])
        require(sid not in sample_by_id, "duplicate sample_id")
        row = dict(row)
        row["sample_id"] = sid
        row["input_event_density"] = float(row["input_event_density"])
        row["sample_aee"] = float(row["sample_aee"])
        row["density_bin"] = density_bin(row["input_event_density"])
        sample_by_id[sid] = row
    require(set(sample_by_id) == set(range(100)), "sample ids are not exactly 0..99")

    records_by_sample = defaultdict(list)
    input_rows = []
    qk_stat_mismatch = 0
    qk_file_sha_mismatch = 0
    block_names = set()
    for rec in trace["records"]:
        sid = int(rec["sample_id"])
        require(sid in sample_by_id, "trace sample_id absent from workload")
        sample = sample_by_id[sid]
        require(rec["sample_key"] == sample["sample_key"], "sample key mismatch at {}".format(sid))
        path = HANDOFF / rec["file"]
        got_sha = sha256_file(path)
        if got_sha != rec["sha256"]:
            qk_file_sha_mismatch += 1
        with np.load(str(path), allow_pickle=False) as z:
            required_keys = {"q_shape", "q_bits_packed", "k_shape", "k_bits_packed", "gate_q17"}
            require(required_keys.issubset(set(z.files)), "NPZ keys missing: {}".format(rec["file"]))
            q_bits = int(np.prod(z["q_shape"], dtype=np.int64))
            k_bits = int(np.prod(z["k_shape"], dtype=np.int64))
            q_active = popcount_packed(z["q_bits_packed"], q_bits)
            k_active = popcount_packed(z["k_bits_packed"], k_bits)
            gate = z["gate_q17"]
            gate_nonzero = int(np.count_nonzero(gate))
            if q_active != int(rec["q_active_bits"]) or k_active != int(rec["k_active_bits"]) or gate_nonzero != int(rec["gate_nonzero"]):
                qk_stat_mismatch += 1
        block_names.add(rec["name"])
        enriched = dict(rec)
        enriched.update({
            "sequence_key": sample["sequence_key"],
            "input_event_density": sample["input_event_density"],
            "sample_aee": sample["sample_aee"],
            "density_bin": sample["density_bin"],
            "q_logical_bits": q_bits,
            "k_logical_bits": k_bits,
            "q_active_bits_replayed": q_active,
            "k_active_bits_replayed": k_active,
            "gate_nonzero_replayed": gate_nonzero,
            "sha256_replayed": got_sha,
        })
        records_by_sample[sid].append(enriched)
        input_rows.append(enriched)
    require(qk_file_sha_mismatch == 0, "QK NPZ SHA mismatch count {}".format(qk_file_sha_mismatch))
    require(qk_stat_mismatch == 0, "QK replay statistic mismatch count {}".format(qk_stat_mismatch))
    require(len(block_names) == 12, "expected exactly 12 attention blocks")
    for sid in range(100):
        require(len(records_by_sample[sid]) == 12, "sample {} does not have 12 QK records".format(sid))
        require(len(set(r["name"] for r in records_by_sample[sid])) == 12, "sample {} block names not unique".format(sid))

    sample_rows = []
    for sid in range(100):
        sample = sample_by_id[sid]
        rr = records_by_sample[sid]
        q_active = sum(int(r["q_active_bits_replayed"]) for r in rr)
        k_active = sum(int(r["k_active_bits_replayed"]) for r in rr)
        q_bits = sum(int(r["q_logical_bits"]) for r in rr)
        k_bits = sum(int(r["k_logical_bits"]) for r in rr)
        gate_nonzero = sum(int(r["gate_nonzero_replayed"]) for r in rr)
        gate_elements = sum(int(r["heads"]) * int(r["temporal_tokens"]) for r in rr)
        sample_rows.append({
            "sample_id": sid,
            "sample_key": sample["sample_key"],
            "sequence_key": sample["sequence_key"],
            "density_bin": sample["density_bin"],
            "input_event_density": sample["input_event_density"],
            "input_events": int(sample["input_events"]),
            "input_active_pixel_ratio": float(sample["input_active_pixel_ratio"]),
            "sample_aee": sample["sample_aee"],
            "token_kzero_ratio": float(sample["token_kzero_ratio"]),
            "q_active_bits": q_active,
            "q_logical_bits": q_bits,
            "q_active_ratio": q_active / float(q_bits),
            "k_active_bits": k_active,
            "k_logical_bits": k_bits,
            "k_active_ratio": k_active / float(k_bits),
            "gate_nonzero": gate_nonzero,
            "gate_elements": gate_elements,
            "gate_nonzero_ratio": gate_nonzero / float(gate_elements),
            "qk_npz_records": len(rr),
        })

    density_rows = []
    for seq in ["ALL"] + sorted(set(r["sequence_key"] for r in sample_rows)):
        seq_rows = sample_rows if seq == "ALL" else [r for r in sample_rows if r["sequence_key"] == seq]
        for label, lo, hi in DENSITY_BINS:
            rr = [r for r in seq_rows if r["density_bin"] == label]
            q_active = sum(r["q_active_bits"] for r in rr)
            q_bits = sum(r["q_logical_bits"] for r in rr)
            k_active = sum(r["k_active_bits"] for r in rr)
            k_bits = sum(r["k_logical_bits"] for r in rr)
            density_rows.append({
                "sequence_key": seq,
                "density_bin": label,
                "lower_inclusive": lo,
                "upper_exclusive_except_last": hi,
                "samples": len(rr),
                "mean_input_event_density": finite_mean([r["input_event_density"] for r in rr]),
                "mean_sample_aee": finite_mean([r["sample_aee"] for r in rr]),
                "q_active_ratio_weighted": q_active / float(q_bits) if q_bits else None,
                "k_active_ratio_weighted": k_active / float(k_bits) if k_bits else None,
                "mean_token_kzero_ratio": finite_mean([r["token_kzero_ratio"] for r in rr]),
            })

    sequence_rows = []
    for seq in sorted(set(r["sequence_key"] for r in sample_rows)):
        rr = [r for r in sample_rows if r["sequence_key"] == seq]
        q_active = sum(r["q_active_bits"] for r in rr)
        q_bits = sum(r["q_logical_bits"] for r in rr)
        k_active = sum(r["k_active_bits"] for r in rr)
        k_bits = sum(r["k_logical_bits"] for r in rr)
        sequence_rows.append({
            "sequence_key": seq,
            "samples": len(rr),
            "density_min": min(r["input_event_density"] for r in rr),
            "density_mean": finite_mean([r["input_event_density"] for r in rr]),
            "density_max": max(r["input_event_density"] for r in rr),
            "sample_aee_mean": finite_mean([r["sample_aee"] for r in rr]),
            "q_active_ratio_weighted": q_active / float(q_bits),
            "k_active_ratio_weighted": k_active / float(k_bits),
            "token_kzero_ratio_mean": finite_mean([r["token_kzero_ratio"] for r in rr]),
            "qk_npz_records": sum(r["qk_npz_records"] for r in rr),
        })
    require(len(sequence_rows) == 3, "expected exactly three sequences")
    require(sum(1 for r in sequence_rows if r["sequence_key"] != "zurich_city_09_a") >= 2, "fewer than two additional sequences")

    # Activation CSV is summary-only.  It is deterministically position-grouped
    # 34 rows per sample, but only 8 rows/sample carry explicit identity.
    activation_explicit = sum(1 for r in activations if r["sample_id"] != "")
    require(activation_explicit == 800, "activation explicit identity population drift")
    for sid in range(100):
        block = activations[sid * 34:(sid + 1) * 34]
        explicit = [r for r in block if r["sample_id"] != ""]
        require(len(explicit) == 8, "activation group explicit identity drift")
        require(all(int(r["sample_id"]) == sid for r in explicit), "activation group/sample mismatch")
        require(all(r["sample_key"] == sample_by_id[sid]["sample_key"] for r in explicit), "activation sample key mismatch")

    coverage_rows, payload_details = raw_payload_coverage([sample_by_id[i] for i in range(100)])

    evaluator_rel = "source/third_party/SDformerFlow/eval_DSEC_flow_SNN.py"
    handoff_files = {x["path"]: x for x in handoff_manifest["files"]}
    require(evaluator_rel in handoff_files, "evaluator source missing from handoff inventory")

    summary = {
        "schema": "m626_h67_ep35_multisequence_density_qk_inventory_v1",
        "status": "PASS_MULTI_SEQUENCE_QK_CPU_REPLAY__RAW_NONATTENTION_COVERAGE_REMAINS_SINGLE_SEQUENCE_OR_MISSING",
        "claim_boundary": "H67 ep35 evidence availability, density stratification, and exact CPU verification of packaged Q/K bit tensors only; no cycles, speedup, energy, PPA, full-network simulator, or new accuracy run.",
        "identity": {
            "contract_path": str(CONTRACT.relative_to(REPO)),
            "contract_sha256": sha256_file(CONTRACT),
            "static_inputs": identities,
            "evaluator": {"path": evaluator_rel, "sha256": handoff_files[evaluator_rel]["sha256"]},
            "checkpoint_load_audit": load,
            "eval_protocol": trace["run_context"]["eval_protocol"],
        },
        "handoff_integrity": handoff_audit,
        "population": {
            "sequences": len(sequence_rows),
            "additional_sequences_beyond_zurich_city_09_a": 2,
            "samples": len(sample_rows),
            "attention_blocks_per_sample": 12,
            "qk_npz_records": len(input_rows),
            "activation_summary_records": len(activations),
            "activation_summary_records_with_explicit_sample_identity": activation_explicit,
        },
        "qk_exact_cpu_replay": {
            "files_sha_verified": len(input_rows),
            "file_sha_mismatches": qk_file_sha_mismatch,
            "manifest_stat_mismatches": qk_stat_mismatch,
            "npz_required_keys_checked": ["q_shape", "q_bits_packed", "k_shape", "k_bits_packed", "gate_q17"],
            "projection_weight_metadata_used_for_claim": False,
        },
        "sequence_summary": sequence_rows,
        "raw_payload_coverage_details": payload_details,
        "gaps": [
            "No multi-sequence raw Conv/FC bitpacks: local M51 payload is limited to zurich_city_09_a S10 and is physically incomplete relative to its 310-record manifest.",
            "No captured M511 ConvTranspose payload exists locally; only a zurich_city_09_a S10 capture contract exists.",
            "No raw multi-sequence ATLIF I/O/state payload exists; M515 is a standalone state-boundary/accounting audit on the S10 ordered trace.",
            "M460 FFN payload is a zurich_city_09_a S10 post-compute oracle, not a pre-compute certificate and not multi-sequence.",
            "The 100-sample activation_records.csv is summary statistics, not raw tensor payload; 26/34 rows per sample omit explicit identity and are only positionally groupable.",
            "The 100-sample Q/K package captures one window per attention block; it is attention-complete only under that stated window boundary, not a full-window/full-network transaction trace.",
            "Sample AEE values are frozen profiler outputs; this audit does not rerun the evaluator because the raw DSEC event/GT/mask dataset is not packaged locally for all 100 samples.",
        ],
        "paper_safe_use": [
            "Use the three-sequence table and fixed event-density bins to show that the available attention sparsity evidence is not a single-sequence anecdote.",
            "Do not use this artifact to claim multi-sequence Conv/FC/ATLIF/decoder speedup; those raw payloads remain single-sequence or absent.",
            "Do not convert Q/K activity ratios into system speedup without a separately admitted same-resource cycle model.",
        ],
        "admission": {
            "multi_sequence_density_stratification": True,
            "multi_sequence_attention_qk_raw_replay": True,
            "multi_sequence_nonattention_raw_payload": False,
            "cycles": False,
            "speedup": False,
            "energy": False,
            "ppa": False,
            "system_speedup": False,
            "date_headline": False,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".m626_staging_", dir=str(output.parent)))
    try:
        result_json = staging / "m626_h67_ep35_multisequence_density_qk_inventory_r1.json"
        result_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_csv(staging / "m626_sequence_summary.csv", list(sequence_rows[0].keys()), sequence_rows)
        write_csv(staging / "m626_sample_density_qk.csv", list(sample_rows[0].keys()), sample_rows)
        write_csv(staging / "m626_density_bins.csv", list(density_rows[0].keys()), density_rows)
        write_csv(staging / "m626_qk_cpu_replay_input_inventory.csv", [
            "sample_id", "sample_key", "sequence_key", "density_bin", "input_event_density", "sample_aee",
            "name", "file", "sha256", "sha256_replayed", "q_logical_bits", "q_active_bits_replayed",
            "k_logical_bits", "k_active_bits_replayed", "gate_nonzero_replayed", "heads", "temporal_tokens",
        ], input_rows)
        write_csv(staging / "m626_evidence_coverage_matrix.csv", [
            "sequence_key", "evidence", "samples_present", "samples_expected", "raw_payload", "direct_cpu_replay", "paper_use",
        ], coverage_rows)
        md = [
            "# M626 H67 ep35 multi-sequence evidence inventory",
            "",
            "Status: `{}`".format(summary["status"]),
            "",
            "Exact CPU replay verified all 1,200 packaged Q/K NPZ files with zero SHA or manifest-stat mismatches. The frozen population spans 100 samples across three DSEC sequences; two are additional to `zurich_city_09_a`.",
            "",
            "| sequence | samples | density min/mean/max | mean AEE | Q active | K active | mean K-zero |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in sequence_rows:
            md.append("| {sequence_key} | {samples} | {density_min:.6f}/{density_mean:.6f}/{density_max:.6f} | {sample_aee_mean:.6f} | {q_active_ratio_weighted:.6%} | {k_active_ratio_weighted:.6%} | {token_kzero_ratio_mean:.6%} |".format(**r))
        md.extend([
            "",
            "This closes multi-sequence availability for attention Q/K density evidence only. Conv/FC, FFN, ATLIF and decoder raw payloads are still single-sequence or missing; see the coverage matrix and JSON gaps. No cycle, speedup, energy, PPA, full-network, or headline claim is admitted.",
            "",
        ])
        (staging / "README.md").write_text("\n".join(md), encoding="utf-8")

        members = sorted(p for p in staging.iterdir() if p.is_file())
        with (staging / "SHA256SUMS").open("w", encoding="utf-8") as f:
            for p in members:
                f.write("{}  {}\n".format(sha256_file(p), p.name))
        seal = sha256_file(staging / "SHA256SUMS")
        (staging / "SHA256SUMS.seal.sha256").write_text("{}  SHA256SUMS\n".format(seal), encoding="utf-8")
        os.rename(str(staging), str(output))
    except Exception:
        shutil.rmtree(str(staging), ignore_errors=True)
        raise
    print(json.dumps({"status": summary["status"], "output": str(output), "sequences": len(sequence_rows), "samples": len(sample_rows), "qk_npz": len(input_rows)}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("M626 FAIL_CLOSED: {}".format(exc), file=sys.stderr)
        sys.exit(2)
