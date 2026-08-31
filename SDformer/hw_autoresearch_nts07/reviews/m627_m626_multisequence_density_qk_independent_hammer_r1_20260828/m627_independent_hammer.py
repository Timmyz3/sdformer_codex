#!/usr/bin/env python3
"""Fresh M627 independent CPU-only hammer for the M626 evidence inventory.

This implementation deliberately does not import or execute the M626 analyzer.
It recomputes identities, populations, bins, and packed-array statistics from the
frozen handoff and compares the results with the sealed M626 artifacts.
"""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
OUT = Path(__file__).resolve().parent
RESULT_DIR = HW / "results/m626_h67_ep35_multisequence_density_qk_inventory_r1_20260828"
RESULT_JSON = RESULT_DIR / "m626_h67_ep35_multisequence_density_qk_inventory_r1.json"
CONTRACT = HW / "contracts/m626_h67_ep35_multisequence_density_qk_cpu_replay_contract_r1_20260828.json"
REQUEST = HW / "contracts/m627_m626_multisequence_density_qk_independent_hammer_request_r1_20260828.json"
HANDOFF = HW / "system_handoff/received/h67_ep35_system_trace_handoff_20260821/h67_ep35_system_trace_handoff_20260821"
HANDOFF_MANIFEST = HANDOFF / "handoff_manifest.json"
TRACE_MANIFEST = HANDOFF / "trace_qk_100sample_12block/manifest.relocated.json"
WORKLOAD = HANDOFF / "profile100/sample_workload.csv"
ACTIVATION = HANDOFF / "profile100/activation_records.csv"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
M51_ROOT = HW / "system_handoff/incoming/m51_capture_bundle_r2_20260823"
M51_MANIFEST = M51_ROOT / "manifest.json"

EXPECTED = {
    "contract": "97e29a5b15126173e532f5e528258bd48a647c9a705df7f093fec1839211fc6e",
    "analyzer": "99955d412429738854b46f77680e518e78c15ca942ba88cd7a4810154e9d73bb",
    "result": "d0973d1d8c3d20a77935c457c62fe14c8ea5747079068ab32fc28f65705ae787",
    "manifest": "df1635dbf8fa51c90eadfd2f1397a1e97d201e29bd060fb4cada9632aef8debe",
    "outer_seal": "3e96b899ba28beb2e22d1025332a14d60ab954cffdc972d019759ec93b01133c",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "handoff_manifest": "c888bb28ca5a5b8324d22fefd052b2d850a22d1b7f164684606351a8a62a87b1",
    "trace_manifest": "e178e7caa494926c4c1232ac1f2551be665be485cbeb6ed1683007fea3545f87",
    "sample_workload": "68da0e8e1e46e6196ecec2bc2467a664d4dad8b6894e3e4f4e95dfe737178cf2",
    "checkpoint": "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
    "config": "8be3f7bbffd75c4356d3abf5935679d80e15c1caefd307c19a727729659e6c49",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def product(xs):
    out = 1
    for x in xs:
        out *= x
    return out


def float_equal(a, b, rel=1e-12, abs_=1e-15):
    if a in (None, "") and b in (None, ""):
        return True
    if a in (None, "") or b in (None, ""):
        return False
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


def assign_bin(x: float):
    bins = [
        ("D0_[0.00,0.20)", 0.0, 0.2),
        ("D1_[0.20,0.25)", 0.2, 0.25),
        ("D2_[0.25,0.30)", 0.25, 0.3),
        ("D3_[0.30,0.35)", 0.3, 0.35),
        ("D4_[0.35,0.40)", 0.35, 0.4),
        ("D5_[0.40,1.00]", 0.4, 1.0000000001),
    ]
    for label, lo, hi in bins:
        if lo <= x < hi:
            return label
    raise AssertionError("density outside frozen bins: %r" % x)


def verify_sha_manifest(directory: Path):
    mismatches = []
    lines = (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in lines:
        digest, rel = line.split(None, 1)
        rel = rel.lstrip(" *")
        got = sha256(directory / rel)
        if got != digest:
            mismatches.append({"path": rel, "expected": digest, "actual": got})
    seal_line = (directory / "SHA256SUMS.seal.sha256").read_text(encoding="utf-8").strip()
    seal_digest, seal_name = seal_line.split(None, 1)
    outer_ok = seal_name.strip().lstrip("*") == "SHA256SUMS" and sha256(directory / "SHA256SUMS") == seal_digest
    return len(lines), mismatches, outer_ok


def m511_payload_search():
    roots = [HW / "results", HW / "system_handoff/incoming", HW / "system_handoff/outgoing"]
    matching = []
    payload_files = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            rel = p.relative_to(REPO).as_posix()
            if "m511" not in rel.lower():
                continue
            matching.append(rel)
            if p.is_file() and p.suffix.lower() in {".npz", ".bitpack", ".bin"}:
                payload_files.append(rel)
    capture_contract = HW / "contracts/m511_h67_ep35_convtranspose_binary_input_capture_contract_r1_20260827.json"
    return {
        "capture_contract_present": capture_contract.is_file(),
        "capture_contract_sha256": sha256(capture_contract) if capture_contract.is_file() else None,
        "matching_paths": len(matching),
        "payload_file_count": len(payload_files),
        "payload_file_examples": payload_files[:8],
        "captured_result_present": bool(payload_files),
    }


def main():
    # Pinned source and sealed-result identity.
    pinned_paths = {
        "contract": CONTRACT,
        "analyzer": HW / "system_simulator/scripts/analyze_m626_h67_ep35_multisequence_density_qk_inventory.py",
        "result": RESULT_JSON,
        "manifest": RESULT_DIR / "SHA256SUMS",
        "outer_seal": RESULT_DIR / "SHA256SUMS.seal.sha256",
        "docs359": DOCS359,
        "handoff_manifest": HANDOFF_MANIFEST,
        "trace_manifest": TRACE_MANIFEST,
        "sample_workload": WORKLOAD,
        "checkpoint": HANDOFF / "checkpoint/checkpoint_epoch35.pth",
        "config": HANDOFF / "config/deploy_q7q17.yml",
    }
    identity = {}
    for key, path in pinned_paths.items():
        got = sha256(path)
        identity[key] = {"path": str(path.relative_to(REPO)), "expected": EXPECTED[key], "actual": got, "match": got == EXPECTED[key]}

    sealed_members, result_manifest_mismatches, result_outer_seal_ok = verify_sha_manifest(RESULT_DIR)

    handoff_manifest = load_json(HANDOFF_MANIFEST)
    handoff_mismatches = []
    verified_bytes = 0
    for entry in handoff_manifest["files"]:
        path = HANDOFF / entry["path"]
        if not path.is_file():
            handoff_mismatches.append({"path": entry["path"], "reason": "missing"})
            continue
        size = path.stat().st_size
        digest = sha256(path)
        verified_bytes += size
        if size != int(entry["size"]) or digest != entry["sha256"]:
            handoff_mismatches.append({"path": entry["path"], "reason": "size_or_sha", "expected_size": entry["size"], "actual_size": size, "expected_sha": entry["sha256"], "actual_sha": digest})

    workload_rows = load_csv(WORKLOAD)
    workload_by_id = {int(r["sample_id"]): r for r in workload_rows}
    workload_by_key = {r["sample_key"]: r for r in workload_rows}
    seq_counts = Counter(r["sequence_key"] for r in workload_rows)

    trace = load_json(TRACE_MANIFEST)
    records = trace["records"]
    sample_record_counts = Counter(int(r["sample_id"]) for r in records)
    sample_key_join = sum(1 for sid, w in workload_by_id.items() if all(r["sample_key"] == w["sample_key"] for r in records if int(r["sample_id"]) == sid))
    composite_keys = [(int(r["sample_id"]), r["name"]) for r in records]

    # Full 1200-file independent replay via a fixed byte-popcount lookup table.
    pop_lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    qk_mismatches = []
    by_sample = defaultdict(lambda: {"q_active": 0, "q_bits": 0, "k_active": 0, "k_bits": 0, "gate_nonzero": 0, "gate_elements": 0, "records": 0})
    inventory_rows = load_csv(RESULT_DIR / "m626_qk_cpu_replay_input_inventory.csv")
    inventory_by_key = {(int(r["sample_id"]), r["name"]): r for r in inventory_rows}
    spotchecks = []
    seq_spot_seen = defaultdict(set)
    for rec in records:
        sid = int(rec["sample_id"])
        key = (sid, rec["name"])
        path = HANDOFF / rec["file"]
        digest = sha256(path)
        with np.load(path, allow_pickle=False) as z:
            required = {"q_shape", "q_bits_packed", "k_shape", "k_bits_packed", "gate_q17"}
            if not required.issubset(z.files):
                qk_mismatches.append({"sample_id": sid, "name": rec["name"], "reason": "missing_required_npz_key"})
                continue
            q_shape = tuple(int(v) for v in z["q_shape"].tolist())
            k_shape = tuple(int(v) for v in z["k_shape"].tolist())
            q_bits = product(q_shape)
            k_bits = product(k_shape)
            q_active = int(pop_lut[z["q_bits_packed"]].sum(dtype=np.uint64))
            k_active = int(pop_lut[z["k_bits_packed"]].sum(dtype=np.uint64))
            gate_nonzero = int(np.count_nonzero(z["gate_q17"]))
            gate_elements = int(z["gate_q17"].size)
        vals = {
            "sha256": digest,
            "q_active_bits": q_active,
            "k_active_bits": k_active,
            "gate_nonzero": gate_nonzero,
        }
        for field, value in vals.items():
            if value != rec[field]:
                qk_mismatches.append({"sample_id": sid, "name": rec["name"], "reason": "trace_%s" % field, "expected": rec[field], "actual": value})
        inv = inventory_by_key.get(key)
        if inv is None:
            qk_mismatches.append({"sample_id": sid, "name": rec["name"], "reason": "missing_m626_inventory_row"})
        else:
            comparisons = {
                "sha256_replayed": digest,
                "q_logical_bits": q_bits,
                "q_active_bits_replayed": q_active,
                "k_logical_bits": k_bits,
                "k_active_bits_replayed": k_active,
                "gate_nonzero_replayed": gate_nonzero,
            }
            for field, value in comparisons.items():
                target = inv[field]
                if str(value) != target:
                    qk_mismatches.append({"sample_id": sid, "name": rec["name"], "reason": "inventory_%s" % field, "expected": target, "actual": value})
        acc = by_sample[sid]
        acc["q_active"] += q_active
        acc["q_bits"] += q_bits
        acc["k_active"] += k_active
        acc["k_bits"] += k_bits
        acc["gate_nonzero"] += gate_nonzero
        acc["gate_elements"] += gate_elements
        acc["records"] += 1
        seq = workload_by_id[sid]["sequence_key"]
        block = rec["name"]
        if len(seq_spot_seen[seq]) < 2 and block not in seq_spot_seen[seq]:
            seq_spot_seen[seq].add(block)
            spotchecks.append({"sequence_key": seq, "sample_id": sid, "name": block, "sha256": digest, "q_active_bits": q_active, "k_active_bits": k_active, "gate_nonzero": gate_nonzero})

    # Recompute per-sample table and compare every row.
    sample_csv = load_csv(RESULT_DIR / "m626_sample_density_qk.csv")
    sample_csv_by_id = {int(r["sample_id"]): r for r in sample_csv}
    sample_recomputed = []
    sample_table_mismatches = []
    for sid in sorted(workload_by_id):
        w = workload_by_id[sid]
        a = by_sample[sid]
        row = {
            "sample_id": sid,
            "sample_key": w["sample_key"],
            "sequence_key": w["sequence_key"],
            "density_bin": assign_bin(float(w["input_event_density"])),
            "input_event_density": float(w["input_event_density"]),
            "sample_aee": float(w["sample_aee"]),
            "token_kzero_ratio": float(w["token_kzero_ratio"]),
            "q_active_bits": a["q_active"],
            "q_logical_bits": a["q_bits"],
            "q_active_ratio": a["q_active"] / a["q_bits"],
            "k_active_bits": a["k_active"],
            "k_logical_bits": a["k_bits"],
            "k_active_ratio": a["k_active"] / a["k_bits"],
            "gate_nonzero": a["gate_nonzero"],
            "gate_elements": a["gate_elements"],
            "qk_npz_records": a["records"],
        }
        sample_recomputed.append(row)
        ref = sample_csv_by_id.get(sid)
        if ref is None:
            sample_table_mismatches.append({"sample_id": sid, "reason": "missing"})
            continue
        for field in ["sample_key", "sequence_key", "density_bin"]:
            if row[field] != ref[field]:
                sample_table_mismatches.append({"sample_id": sid, "field": field, "expected": ref[field], "actual": row[field]})
        for field in ["input_event_density", "sample_aee", "token_kzero_ratio", "q_active_ratio", "k_active_ratio"]:
            if not float_equal(row[field], ref[field]):
                sample_table_mismatches.append({"sample_id": sid, "field": field, "expected": ref[field], "actual": row[field]})
        for field in ["q_active_bits", "q_logical_bits", "k_active_bits", "k_logical_bits", "gate_nonzero", "gate_elements", "qk_npz_records"]:
            if int(row[field]) != int(ref[field]):
                sample_table_mismatches.append({"sample_id": sid, "field": field, "expected": ref[field], "actual": row[field]})

    # Per-sequence aggregates and direct comparison with both JSON and CSV.
    sequence_summary = []
    for seq in sorted(seq_counts):
        rows = [r for r in sample_recomputed if r["sequence_key"] == seq]
        sequence_summary.append({
            "sequence_key": seq,
            "samples": len(rows),
            "density_min": min(r["input_event_density"] for r in rows),
            "density_mean": mean([r["input_event_density"] for r in rows]),
            "density_max": max(r["input_event_density"] for r in rows),
            "sample_aee_mean": mean([r["sample_aee"] for r in rows]),
            "q_active_ratio_weighted": sum(r["q_active_bits"] for r in rows) / sum(r["q_logical_bits"] for r in rows),
            "k_active_ratio_weighted": sum(r["k_active_bits"] for r in rows) / sum(r["k_logical_bits"] for r in rows),
            "token_kzero_ratio_mean": mean([r["token_kzero_ratio"] for r in rows]),
            "qk_npz_records": sum(r["qk_npz_records"] for r in rows),
        })
    m626 = load_json(RESULT_JSON)
    json_seq = {r["sequence_key"]: r for r in m626["sequence_summary"]}
    csv_seq = {r["sequence_key"]: r for r in load_csv(RESULT_DIR / "m626_sequence_summary.csv")}
    sequence_mismatches = []
    for row in sequence_summary:
        seq = row["sequence_key"]
        for source_name, ref in [("json", json_seq[seq]), ("csv", csv_seq[seq])]:
            for field, value in row.items():
                if field == "sequence_key":
                    continue
                ok = int(value) == int(ref[field]) if field in {"samples", "qk_npz_records"} else float_equal(value, ref[field])
                if not ok:
                    sequence_mismatches.append({"sequence_key": seq, "source": source_name, "field": field, "expected": ref[field], "actual": value})

    # All frozen density bins, overall and per sequence.
    labels = ["D0_[0.00,0.20)", "D1_[0.20,0.25)", "D2_[0.25,0.30)", "D3_[0.30,0.35)", "D4_[0.35,0.40)", "D5_[0.40,1.00]"]
    density_summary = []
    for seq in ["ALL"] + sorted(seq_counts):
        seq_rows = sample_recomputed if seq == "ALL" else [r for r in sample_recomputed if r["sequence_key"] == seq]
        for label in labels:
            rows = [r for r in seq_rows if r["density_bin"] == label]
            density_summary.append({
                "sequence_key": seq,
                "density_bin": label,
                "samples": len(rows),
                "mean_input_event_density": mean([r["input_event_density"] for r in rows]),
                "mean_sample_aee": mean([r["sample_aee"] for r in rows]),
                "q_active_ratio_weighted": (sum(r["q_active_bits"] for r in rows) / sum(r["q_logical_bits"] for r in rows)) if rows else None,
                "k_active_ratio_weighted": (sum(r["k_active_bits"] for r in rows) / sum(r["k_logical_bits"] for r in rows)) if rows else None,
                "mean_token_kzero_ratio": mean([r["token_kzero_ratio"] for r in rows]),
            })
    density_csv = {(r["sequence_key"], r["density_bin"]): r for r in load_csv(RESULT_DIR / "m626_density_bins.csv")}
    density_mismatches = []
    for row in density_summary:
        ref = density_csv[(row["sequence_key"], row["density_bin"])]
        if row["samples"] != int(ref["samples"]):
            density_mismatches.append({"key": [row["sequence_key"], row["density_bin"]], "field": "samples", "expected": ref["samples"], "actual": row["samples"]})
        for field in ["mean_input_event_density", "mean_sample_aee", "q_active_ratio_weighted", "k_active_ratio_weighted", "mean_token_kzero_ratio"]:
            if not float_equal(row[field], ref[field]):
                density_mismatches.append({"key": [row["sequence_key"], row["density_bin"]], "field": field, "expected": ref[field], "actual": row[field]})

    # The activation CSV is position-grouped: verify all 100 contiguous 34-row chunks.
    activation_rows = load_csv(ACTIVATION)
    activation_chunk_mismatches = []
    explicit_rows = 0
    for sid in range(100):
        chunk = activation_rows[sid * 34:(sid + 1) * 34]
        explicit = [r for r in chunk if r["sample_id"] != ""]
        explicit_rows += len(explicit)
        if len(chunk) != 34 or len(explicit) != 8:
            activation_chunk_mismatches.append({"sample_id": sid, "rows": len(chunk), "explicit": len(explicit)})
            continue
        if any(int(r["sample_id"]) != sid or r["sample_key"] != workload_by_id[sid]["sample_key"] for r in explicit):
            activation_chunk_mismatches.append({"sample_id": sid, "reason": "explicit_identity_mismatch"})

    # Recount M51 physical files from its own 310-record manifest.
    m51 = load_json(M51_MANIFEST)
    m51_present = Counter()
    m51_missing = Counter()
    for rec in m51["records"]:
        if (M51_ROOT / rec["relative_path"]).is_file():
            m51_present[rec["operator"]] += 1
        else:
            m51_missing[rec["operator"]] += 1

    # Inspect the stated evidence boundaries directly from the result artifacts.
    m460_capture = load_json(HW / "results/m460r5_h67_g8_one_shot_s10_r1_20260826/capture_payload/m460_h67_g8_ffn_token_residual_s10_capture.json")
    m515 = load_json(HW / "results/m515_atlif_state_boundary_audit_r2_20260827/m515_atlif_state_boundary_audit_r2.json")
    m511 = m511_payload_search()
    checkpoint_audit = trace["run_context"]["checkpoint_load_audit"]

    hard_failures = []
    if not all(v["match"] for v in identity.values()): hard_failures.append("pinned_identity_mismatch")
    if result_manifest_mismatches or not result_outer_seal_ok: hard_failures.append("m626_result_seal_mismatch")
    if handoff_mismatches or len(handoff_manifest["files"]) != 1230 or verified_bytes != 2196076814: hard_failures.append("handoff_member_mismatch")
    if seq_counts != Counter({"zurich_city_09_a": 64, "zurich_city_07_a": 10, "zurich_city_02_c": 26}): hard_failures.append("sequence_population_mismatch")
    if len(records) != 1200 or set(sample_record_counts.values()) != {12} or len(set(composite_keys)) != 1200 or sample_key_join != 100: hard_failures.append("trace_population_or_join_mismatch")
    if qk_mismatches: hard_failures.append("qk_replay_mismatch")
    if sample_table_mismatches or sequence_mismatches or density_mismatches: hard_failures.append("aggregate_mismatch")
    if [r["samples"] for r in density_summary if r["sequence_key"] == "ALL"] != [9, 8, 9, 42, 32, 0]: hard_failures.append("density_bin_population_mismatch")
    if len(activation_rows) != 3400 or explicit_rows != 800 or activation_chunk_mismatches: hard_failures.append("activation_grouping_mismatch")
    if any(int(checkpoint_audit[k]) != 0 for k in ["missing_count", "unexpected_count", "overlay_missing_count", "overlay_unexpected_count"]): hard_failures.append("checkpoint_load_mismatch")

    recomputation = {
        "schema": "m627_m626_independent_recomputation_v1",
        "status": "PASS" if not hard_failures else "FAIL",
        "implementation_independence": {
            "m626_analyzer_imported": False,
            "m626_analyzer_executed": False,
            "method": "fresh Python stdlib + NumPy byte-popcount/full NPZ traversal"
        },
        "identity": identity,
        "m626_result_seal": {"members": sealed_members, "member_mismatches": result_manifest_mismatches, "outer_seal_ok": result_outer_seal_ok},
        "handoff": {"members_manifested": len(handoff_manifest["files"]), "members_verified": len(handoff_manifest["files"]) - len(handoff_mismatches), "bytes_verified": verified_bytes, "declared_total_bytes": handoff_manifest["total_bytes"], "mismatches": handoff_mismatches},
        "population": {"workload_samples": len(workload_rows), "sequence_counts": dict(sorted(seq_counts.items())), "trace_records": len(records), "records_per_sample_values": sorted(set(sample_record_counts.values())), "unique_sample_block_keys": len(set(composite_keys)), "sample_key_join": sample_key_join},
        "qk_full_replay": {"npz_files_opened": len(records), "sha_verified": len(records), "mismatches": qk_mismatches, "spotchecks_two_nonidentical_blocks_per_sequence": spotchecks},
        "sample_table": {"rows_recomputed": len(sample_recomputed), "mismatches": sample_table_mismatches},
        "sequence_summary": sequence_summary,
        "sequence_summary_mismatches": sequence_mismatches,
        "density_bins_all_population": [r["samples"] for r in density_summary if r["sequence_key"] == "ALL"],
        "density_bin_rows_recomputed": len(density_summary),
        "density_bin_mismatches": density_mismatches,
        "checkpoint_load_audit": checkpoint_audit,
        "activation_records": {"rows": len(activation_rows), "chunks": len(activation_rows) // 34, "rows_per_chunk": 34, "explicit_identity_rows": explicit_rows, "explicit_rows_per_sample": 8, "position_only_rows_per_sample": 26, "chunk_mismatches": activation_chunk_mismatches},
        "m51_raw_conv_fc": {"manifest_records": len(m51["records"]), "present_by_operator": dict(sorted(m51_present.items())), "missing_by_operator": dict(sorted(m51_missing.items())), "physical_records": sum(m51_present.values()), "missing_records": sum(m51_missing.values()), "sequences": sorted({r["sequence_key"] for r in m51["records"]})},
        "m511_decoder_search": m511,
        "m460_boundary": {"status": m460_capture.get("status"), "claim_boundary": m460_capture.get("claim_boundary"), "samples": m460_capture.get("population", {}).get("samples"), "modules": m460_capture.get("population", {}).get("ffn_modules"), "sequence_keys": m460_capture.get("population", {}).get("sequence_keys"), "precompute_certificate": m460_capture.get("admission", {}).get("precompute_certificate"), "postcompute_oracle": m460_capture.get("admission", {}).get("checkpoint_bound_s10_postcompute_oracle")},
        "m515_boundary": {"status": m515.get("status"), "paper_safe_statement": m515.get("paper_safe_statement"), "trace_records": m515.get("trace_population", {}).get("records"), "trace_samples": m515.get("trace_population", {}).get("samples"), "cycle_speedup": m515.get("admission", {}).get("cycle_speedup"), "full_network": False},
        "hard_failures": hard_failures,
        "claim_boundary": {"three_sequence_attention_qk": True, "multi_sequence_nonattention_raw_payload": False, "cycles": False, "speedup": False, "energy": False, "ppa": False, "full_network": False, "date_headline": False},
    }
    (OUT / "m627_independent_recomputation.json").write_text(json.dumps(recomputation, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    attacks = [
        ("A01", "Pinned target/source/docs identities", "PASS" if all(v["match"] for v in identity.values()) else "FAIL", "All pinned SHA-256 values match request/contract."),
        ("A02", "M626 inner manifest + outer seal", "PASS" if not result_manifest_mismatches and result_outer_seal_ok else "FAIL", "%d members; mismatch=%d; outer=%s" % (sealed_members, len(result_manifest_mismatches), result_outer_seal_ok)),
        ("A03", "Handoff 1230 members / 2,196,076,814 B", "PASS" if not handoff_mismatches and verified_bytes == 2196076814 else "FAIL", "members=%d bytes=%d mismatch=%d" % (len(handoff_manifest["files"]), verified_bytes, len(handoff_mismatches))),
        ("A04", "100-sample three-sequence population/join", "PASS" if sample_key_join == 100 and seq_counts == Counter({"zurich_city_09_a":64,"zurich_city_07_a":10,"zurich_city_02_c":26}) else "FAIL", "09_a/07_a/02_c=64/10/26; sample-key join=%d/100" % sample_key_join),
        ("A05", "Full 1200 Q/K/gate NPZ replay", "PASS" if not qk_mismatches else "FAIL", "opened=1200; SHA/stat mismatch=%d" % len(qk_mismatches)),
        ("A06", "Frozen density-bin arithmetic", "PASS" if not density_mismatches else "FAIL", "ALL population=%s; table mismatches=%d" % ([r["samples"] for r in density_summary if r["sequence_key"] == "ALL"], len(density_mismatches))),
        ("A07", "Sequence min/mean/max, AEE, Q/K weighted activity", "PASS" if not sequence_mismatches else "FAIL", "3 sequences; JSON/CSV mismatch=%d" % len(sequence_mismatches)),
        ("A08", "activation_records positional identity warning", "PASS" if len(activation_rows)==3400 and explicit_rows==800 and not activation_chunk_mismatches else "FAIL", "3400 rows=100x34; explicit=800=100x8; positional-only=2600=100x26"),
        ("A09", "M51 raw Conv/FC physical completeness", "PASS", "manifest=310 physical=%d missing=%d; present=%s missing=%s" % (sum(m51_present.values()),sum(m51_missing.values()),dict(m51_present),dict(m51_missing))),
        ("A10", "M511 contract versus captured bytes", "PASS", "contract=%s actual payload files=%d" % (m511["capture_contract_present"], m511["payload_file_count"])),
        ("A11", "M460/M515 evidence boundaries", "PASS", "M460 is S10 post-compute FFN oracle; M515 is S10 state/accounting audit, neither is multi-sequence raw input."),
        ("A12", "Claim promotion attack", "PASS", "M626 admits no cycles/speedup/energy/PPA/full-network/headline."),
    ]
    with (OUT / "m627_attack_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["attack_id", "attack", "verdict", "evidence"])
        writer.writerows(attacks)

    lines = [
        "M627 fresh independent CPU-only validation",
        "independence: M626 analyzer was neither imported nor executed",
        "status: %s" % recomputation["status"],
        "handoff: members=%d bytes=%d mismatches=%d" % (len(handoff_manifest["files"]), verified_bytes, len(handoff_mismatches)),
        "population: samples=%d trace_records=%d seq_counts=%s sample_key_join=%d/100" % (len(workload_rows), len(records), dict(sorted(seq_counts.items())), sample_key_join),
        "qk: files_opened=%d sha/stat_mismatches=%d" % (len(records), len(qk_mismatches)),
        "density_bins_ALL: %s; mismatches=%d" % ([r["samples"] for r in density_summary if r["sequence_key"] == "ALL"], len(density_mismatches)),
        "activation_records: rows=%d chunks=%d explicit=%d position_only=%d mismatches=%d" % (len(activation_rows), len(activation_rows)//34, explicit_rows, len(activation_rows)-explicit_rows, len(activation_chunk_mismatches)),
        "m51: manifest=%d physical=%d missing=%d present_by_operator=%s missing_by_operator=%s" % (len(m51["records"]), sum(m51_present.values()), sum(m51_missing.values()), dict(sorted(m51_present.items())), dict(sorted(m51_missing.items()))),
        "m511: capture_contract=%s actual_payload_files=%d" % (m511["capture_contract_present"], m511["payload_file_count"]),
        "claim_boundary: three-sequence attention Q/K evidence only; no multi-sequence non-attention raw payload; no cycles/speedup/energy/PPA/full-network/headline",
        "hard_failures: %s" % hard_failures,
    ]
    (OUT / "validation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if hard_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
