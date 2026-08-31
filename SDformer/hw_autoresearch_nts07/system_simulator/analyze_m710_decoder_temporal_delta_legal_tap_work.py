#!/usr/bin/env python3
"""M710 exact decoder temporal-delta legal-tap product-work audit.

This is a CPU-only workload counter.  It does not model cycles, memory, state,
accuracy, RTL, or a decoder numerical bridge.  D1 is counted only as its
admitted exact {0, runtime-theta} mask identity.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import struct

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def strict_json(path):
    def pairs(values):
        out = {}
        for key, value in values:
            if key in out:
                raise ValueError("duplicate JSON key: %s" % key)
            out[key] = value
        return out

    def bad_constant(value):
        raise ValueError("non-finite JSON value: %s" % value)

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=bad_constant,
    )


def canonical_json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def confined(root, relative):
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("unconfined path: %s" % relative)
    root_real = root.resolve()
    target = (root / rel).resolve()
    try:
        target.relative_to(root_real)
    except ValueError:
        raise ValueError("path escapes root: %s" % relative)
    return target


def parse_sha_manifest(path):
    rows = []
    seen = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        if len(line) < 67 or line[64:66] != "  ":
            raise ValueError("malformed SHA256SUMS row")
        digest, rel = line[:64], line[66:]
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("invalid SHA256 digest")
        if rel in seen:
            raise ValueError("duplicate SHA256SUMS member: %s" % rel)
        seen.add(rel)
        rows.append((digest, rel))
    return rows


def verify_double_sealed_tree(root, expected_manifest_sha, expected_outer_file_sha):
    root = Path(root)
    manifest_path = root / "SHA256SUMS"
    outer_path = root / "SHA256SUMS.seal.sha256"
    if sha256(manifest_path) != expected_manifest_sha:
        raise ValueError("manifest SHA mismatch: %s" % root)
    if sha256(outer_path) != expected_outer_file_sha:
        raise ValueError("outer seal file SHA mismatch: %s" % root)
    outer_rows = parse_sha_manifest(outer_path)
    if outer_rows != [(expected_manifest_sha, "SHA256SUMS")]:
        raise ValueError("outer seal content mismatch: %s" % root)
    members = parse_sha_manifest(manifest_path)
    for digest, relative in members:
        target = confined(root, relative)
        if not target.is_file() or target.is_symlink():
            raise ValueError("sealed member not a regular file: %s" % target)
        if sha256(target) != digest:
            raise ValueError("sealed member SHA mismatch: %s" % target)
    return members


def legal_tap_weights(height, width):
    if height <= 0 or width <= 0:
        raise ValueError("nonpositive geometry")
    y = np.full((height,), 3, dtype=np.uint8)
    x = np.full((width,), 3, dtype=np.uint8)
    y[0] = 2
    x[0] = 2
    return y[:, None] * x[None, :]


def count_mask(mask, tap_weights):
    # mask is [T,B,C,H,W], uint8 0/1.
    if mask.ndim != 5 or mask.shape[0] != 10 or mask.shape[1] != 1:
        raise ValueError("unexpected mask shape: %r" % (mask.shape,))
    if mask.shape[-2:] != tap_weights.shape:
        raise ValueError("tap shape mismatch")
    if np.any((mask != 0) & (mask != 1)):
        raise ValueError("mask is not binary")

    delta = np.empty_like(mask)
    delta[0] = mask[0]
    delta[1:] = np.bitwise_xor(mask[1:], mask[:-1])

    full_by_tap = {}
    delta_by_tap = {}
    for multiplicity in (4, 6, 9):
        spatial = tap_weights == multiplicity
        full_by_tap[str(multiplicity)] = int(mask[..., spatial].sum(dtype=np.int64))
        delta_by_tap[str(multiplicity)] = int(delta[..., spatial].sum(dtype=np.int64))

    full_sources = int(mask.sum(dtype=np.int64))
    delta_sources = int(delta.sum(dtype=np.int64))
    t0_sources = int(mask[0].sum(dtype=np.int64))
    transition_sources = int(delta[1:].sum(dtype=np.int64))
    full_legal_taps = sum(int(key) * value for key, value in full_by_tap.items())
    delta_legal_taps = sum(int(key) * value for key, value in delta_by_tap.items())
    return {
        "full_active_sources": full_sources,
        "delta_initial_active_sources": t0_sources,
        "delta_transition_sources": transition_sources,
        "delta_sources": delta_sources,
        "full_sources_by_legal_tap_multiplicity": full_by_tap,
        "delta_sources_by_legal_tap_multiplicity": delta_by_tap,
        "full_active_legal_tap_events": full_legal_taps,
        "delta_initial_plus_xor_legal_tap_events": delta_legal_taps,
    }


def ratio(numerator, denominator):
    if denominator <= 0:
        raise ValueError("ratio denominator is not positive")
    return float(numerator) / float(denominator)


def aggregate(rows, keys):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        item = groups.setdefault(key, {
            "records": 0,
            "full_product_work": 0,
            "delta_product_work": 0,
            "full_active_legal_tap_events": 0,
            "delta_initial_plus_xor_legal_tap_events": 0,
        })
        item["records"] += 1
        for name in (
            "full_product_work",
            "delta_product_work",
            "full_active_legal_tap_events",
            "delta_initial_plus_xor_legal_tap_events",
        ):
            item[name] += int(row[name])
    out = []
    for key in sorted(groups):
        item = groups[key]
        entry = {name: value for name, value in zip(keys, key)}
        entry.update(item)
        entry["delta_over_full_product_work"] = ratio(
            item["delta_product_work"], item["full_product_work"])
        out.append(entry)
    return out


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    contract_path = Path(args.contract).resolve()
    contract = strict_json(contract_path)
    if contract.get("schema") != "m710_decoder_temporal_delta_legal_tap_work_contract_v1":
        raise ValueError("wrong contract schema")
    if contract.get("status") != "FROZEN_CPU_PRODUCT_WORK_AUDIT_ONLY":
        raise ValueError("wrong contract status")

    repo = Path(__file__).resolve().parents[2]
    hw = repo / "hw_autoresearch_nts07"
    output = Path(args.output_dir).resolve()
    allowed_output = (hw / contract["output"]["relative_path"]).resolve()
    if output != allowed_output:
        raise ValueError("output path is not contract-pinned")
    if output.exists():
        raise FileExistsError("immutable output already exists")

    analyzer_rel = contract["identity"]["analyzer"]["path"]
    analyzer_path = confined(hw, analyzer_rel)
    if analyzer_path.resolve() != Path(__file__).resolve():
        raise ValueError("analyzer path identity mismatch")
    if sha256(analyzer_path) != contract["identity"]["analyzer"]["sha256"]:
        raise ValueError("analyzer SHA mismatch")
    test_path = confined(hw, contract["identity"]["tests"]["path"])
    if sha256(test_path) != contract["identity"]["tests"]["sha256"]:
        raise ValueError("tests SHA mismatch")
    if sha256(contract_path) != contract["identity"]["contract_sha256_excluding_self"]:
        # The contract stores the canonical payload SHA with this one field set
        # to 64 zeroes, avoiding a circular self-hash.
        normalized = json.loads(json.dumps(contract))
        normalized["identity"]["contract_sha256_excluding_self"] = "0" * 64
        if hashlib.sha256(canonical_json_bytes(normalized)).hexdigest() != contract["identity"]["contract_sha256_excluding_self"]:
            raise ValueError("contract canonical payload SHA mismatch")

    upstream = contract["upstream"]
    m699_root = confined(hw, upstream["m699_root"])
    m699_members = verify_double_sealed_tree(
        m699_root,
        upstream["m699_manifest_file_sha256"],
        upstream["m699_outer_seal_file_sha256"],
    )
    m705_root = confined(hw, upstream["m705_root"])
    verify_double_sealed_tree(
        m705_root,
        upstream["m705_manifest_file_sha256"],
        upstream["m705_outer_seal_file_sha256"],
    )
    m705_review_path = confined(m705_root, "review.json")
    if sha256(m705_review_path) != upstream["m705_review_sha256"]:
        raise ValueError("M705 review SHA mismatch")
    m705 = strict_json(m705_review_path)
    if not m705.get("go") or m705["severity"]["p0"] != 0 or m705["severity"]["p1"] != 0:
        raise ValueError("M705 is not payload-admitted")
    if m705["frozen_result"]["manifest_sha256"] != upstream["m699_payload_manifest_sha256"]:
        raise ValueError("M705/M699 payload identity mismatch")

    docs359 = confined(hw, upstream["docs359_path"])
    if sha256(docs359) != upstream["docs359_sha256"]:
        raise ValueError("docs359 SHA mismatch")

    manifest_path = m699_root / "manifest.json"
    if sha256(manifest_path) != upstream["m699_payload_manifest_sha256"]:
        raise ValueError("M699 payload manifest SHA mismatch")
    manifest = strict_json(manifest_path)
    if manifest.get("schema") != "m699_h67_ep35_multisequence_decoder_payload_v1":
        raise ValueError("wrong M699 schema")
    if len(manifest.get("records", [])) != 120:
        raise ValueError("M699 record count is not 120")
    if manifest["population"] != contract["population"]:
        raise ValueError("population mismatch")

    theta = manifest["d1_runtime_threshold_identity"]
    expected_theta = contract["d1_runtime_threshold_identity"]
    for key in ("value", "ieee754_uint32", "ieee754_le_hex", "content_sha256"):
        if theta.get(key) != expected_theta.get(key):
            raise ValueError("D1 theta identity mismatch: %s" % key)
    if struct.unpack("<I", struct.pack("<f", float(theta["value"])))[0] != theta["ieee754_uint32"]:
        raise ValueError("D1 theta bit identity mismatch")

    expected_member_names = {relative for _, relative in m699_members}
    expected_member_names.update(("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    actual_names = set()
    for path in m699_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("M699 symlink is forbidden")
        if path.is_file():
            actual_names.add(path.relative_to(m699_root).as_posix())
    if actual_names != expected_member_names:
        raise ValueError("M699 topology differs from sealed topology")

    modules = contract["modules"]
    rows = []
    seen = set()
    for record in manifest["records"]:
        key = (record["global_sample_id"], record["module_index"])
        if key in seen:
            raise ValueError("duplicate record key: %r" % (key,))
        seen.add(key)
        module_id = "D%d" % record["module_index"]
        module = modules[module_id]
        expected_route = "EXACT_SCALED_BINARY_BITPACK" if module_id == "D1" else "EXACT_BINARY_BITPACK"
        if record["route"] != expected_route:
            raise ValueError("route mismatch for %s" % module_id)
        expected_shape = [10, 1, module["cin"], module["height"], module["width"]]
        if record["input_shape"] != expected_shape:
            raise ValueError("shape mismatch for %r" % (key,))
        relative = record["relative_path"]
        payload = confined(m699_root, relative)
        if relative not in expected_member_names:
            raise ValueError("payload is not sealed: %s" % relative)
        if expected_route == "EXACT_SCALED_BINARY_BITPACK":
            payload_stats = record["statistics"]["scaled_binary_audit"]
            active_count = payload_stats["theta_count"]
            if not payload_stats.get("theta_gate_pass"):
                raise ValueError("D1 theta gate is not passed")
        else:
            payload_stats = record["statistics"]
            active_count = payload_stats["one_count"]
        if sha256(payload) != payload_stats["packed_sha256"]:
            raise ValueError("payload SHA mismatch")
        elements = int(np.prod(expected_shape, dtype=np.int64))
        expected_bytes = (elements + 7) // 8
        raw = payload.read_bytes()
        if len(raw) != expected_bytes or len(raw) != payload_stats["packed_bytes"]:
            raise ValueError("payload size mismatch")
        unpacked = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
        if unpacked.size > elements and np.any(unpacked[elements:]):
            raise ValueError("nonzero packed tail")
        mask = unpacked[:elements].reshape(expected_shape)
        if int(mask.sum(dtype=np.int64)) != active_count:
            raise ValueError("active-count mismatch")
        counts = count_mask(mask, legal_tap_weights(module["height"], module["width"]))
        full_work = counts["full_active_legal_tap_events"] * module["cout"]
        delta_work = counts["delta_initial_plus_xor_legal_tap_events"] * module["cout"]
        row = {
            "global_sample_id": record["global_sample_id"],
            "sequence": record["sequence"],
            "sequence_sample_id": record["sequence_sample_id"],
            "module": module_id,
            "module_index": record["module_index"],
            "route": record["route"],
            "cin": module["cin"],
            "cout": module["cout"],
            "height": module["height"],
            "width": module["width"],
            "packed_sha256": payload_stats["packed_sha256"],
            **counts,
            "full_product_work": full_work,
            "delta_product_work": delta_work,
            "delta_over_full_product_work": ratio(delta_work, full_work),
        }
        if counts["delta_sources"] != counts["delta_initial_active_sources"] + counts["delta_transition_sources"]:
            raise ValueError("delta source conservation failure")
        if full_work != counts["full_active_legal_tap_events"] * module["cout"]:
            raise ValueError("full product conservation failure")
        if delta_work != counts["delta_initial_plus_xor_legal_tap_events"] * module["cout"]:
            raise ValueError("delta product conservation failure")
        rows.append(row)

    if len(seen) != 120:
        raise ValueError("record lattice is incomplete")
    expected_lattice = {(sample, module) for sample in range(30) for module in range(4)}
    if seen != expected_lattice:
        raise ValueError("record lattice mismatch")

    by_sequence = aggregate(rows, ["sequence"])
    by_module = aggregate(rows, ["module"])
    by_sample = aggregate(rows, ["global_sample_id", "sequence", "sequence_sample_id"])
    overall = aggregate(rows, [])[0]
    record_ratios = [row["delta_over_full_product_work"] for row in rows]
    sample_ratios = [row["delta_over_full_product_work"] for row in by_sample]
    module_ratios = [row["delta_over_full_product_work"] for row in by_module]
    gate_threshold = float(contract["fast_kill_gate"]["maximum_delta_over_full_product_work"])
    gate_pass = overall["delta_over_full_product_work"] < gate_threshold
    all_modules_regress = all(value > 1.0 for value in module_ratios)
    verdict = "GO_MEASUREMENT_ONLY" if gate_pass else "KILL_N2_NO_RTL"

    summary = {
        "schema": "m710_decoder_temporal_delta_legal_tap_product_work_result_v1",
        "status": "PASS_CPU_PRODUCT_WORK_AUDIT__FRESH_REVIEW_REQUIRED",
        "verdict": verdict,
        "identity": {
            "contract_path": contract_path.as_posix(),
            "contract_sha256": sha256(contract_path),
            "analyzer_path": analyzer_path.as_posix(),
            "analyzer_sha256": sha256(analyzer_path),
            "tests_path": test_path.as_posix(),
            "tests_sha256": sha256(test_path),
            "m699_payload_manifest_sha256": sha256(manifest_path),
            "m699_manifest_file_sha256": sha256(m699_root / "SHA256SUMS"),
            "m699_outer_seal_file_sha256": sha256(m699_root / "SHA256SUMS.seal.sha256"),
            "m705_review_sha256": sha256(m705_review_path),
            "m705_manifest_file_sha256": sha256(m705_root / "SHA256SUMS"),
            "m705_outer_seal_file_sha256": sha256(m705_root / "SHA256SUMS.seal.sha256"),
            "docs359_sha256": sha256(docs359),
        },
        "population": contract["population"],
        "geometry": {
            "operator": "ConvTranspose2d",
            "kernel": [3, 3],
            "stride": [2, 2],
            "padding": [1, 1],
            "output_padding": [1, 1],
            "legal_tap_multiplicities": [4, 6, 9],
            "timestep_policy": "full=sum_t active(t); delta=active(t0)+sum_t>=1 XOR(active(t),active(t-1))",
        },
        "d1_boundary": {
            "route": "EXACT_SCALED_BINARY_BITPACK",
            "runtime_theta": theta["value"],
            "runtime_theta_ieee754_uint32": theta["ieee754_uint32"],
            "runtime_theta_content_sha256": theta["content_sha256"],
            "mask_product_work_counted": True,
            "folded_weight_deployment_admitted": False,
            "decoder_numeric_equivalence_admitted": False,
        },
        "overall_ratio_of_sums": overall,
        "distribution": {
            "per_record_delta_over_full_min": min(record_ratios),
            "per_record_delta_over_full_max": max(record_ratios),
            "per_sample_delta_over_full_min": min(sample_ratios),
            "per_sample_delta_over_full_max": max(sample_ratios),
            "per_module_delta_over_full_min": min(module_ratios),
            "per_module_delta_over_full_max": max(module_ratios),
        },
        "per_sequence": by_sequence,
        "per_module": by_module,
        "fast_kill_gate": {
            "metric": "delta_product_work / full_active_product_work",
            "maximum_strict": gate_threshold,
            "pass": gate_pass,
            "all_four_modules_regress_above_one": all_modules_regress,
            "decision": verdict,
        },
        "conservation": {
            "records": 120,
            "record_lattice_complete": True,
            "all_product_work_equals_legal_tap_events_times_cout": True,
            "all_delta_sources_equal_t0_plus_transitions": True,
            "ratio_of_sums_not_mean_of_ratios": True,
        },
        "claim_boundary": {
            "product_work_regression": True,
            "payload_identity": True,
            "cycles": False,
            "speedup": False,
            "system_speedup": False,
            "accuracy": False,
            "numeric_bridge": False,
            "rtl": False,
            "vcs": False,
            "eda": False,
            "dc": False,
            "formality": False,
            "ptpx": False,
            "energy": False,
            "ppa": False,
            "date_headline": False,
        },
    }

    staging = output.with_name(output.name + ".staging.%d" % os.getpid())
    if staging.exists():
        raise FileExistsError("staging path already exists")
    staging.mkdir(parents=False)
    try:
        (staging / "summary.json").write_bytes(canonical_json_bytes(summary))
        record_fields = [
            "global_sample_id", "sequence", "sequence_sample_id", "module", "module_index", "route",
            "cin", "cout", "height", "width", "packed_sha256", "full_active_sources",
            "delta_initial_active_sources", "delta_transition_sources", "delta_sources",
            "full_active_legal_tap_events", "delta_initial_plus_xor_legal_tap_events",
            "full_product_work", "delta_product_work", "delta_over_full_product_work",
        ]
        aggregate_fields = [
            "global_sample_id", "sequence", "sequence_sample_id", "module", "records",
            "full_active_legal_tap_events", "delta_initial_plus_xor_legal_tap_events",
            "full_product_work", "delta_product_work", "delta_over_full_product_work",
        ]
        write_csv(staging / "per_record.csv", rows, record_fields)
        write_csv(staging / "per_sequence.csv", by_sequence, aggregate_fields)
        write_csv(staging / "per_module.csv", by_module, aggregate_fields)
        write_csv(staging / "per_sample.csv", by_sample, aggregate_fields)
        complete = {
            "status": "PASS_M710_CPU_PRODUCT_WORK_AUDIT__FRESH_REVIEW_REQUIRED",
            "verdict": verdict,
            "delta_over_full_product_work": overall["delta_over_full_product_work"],
            "claims": "product-work regression only; no cycles/speedup/system/accuracy/numeric-bridge/RTL/EDA",
        }
        (staging / "RUN_COMPLETE.txt").write_bytes(canonical_json_bytes(complete))
        members = ["RUN_COMPLETE.txt", "per_module.csv", "per_record.csv", "per_sample.csv", "per_sequence.csv", "summary.json"]
        manifest_lines = ["%s  %s" % (sha256(staging / name), name) for name in members]
        (staging / "SHA256SUMS").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        manifest_sha = sha256(staging / "SHA256SUMS")
        (staging / "SHA256SUMS.seal.sha256").write_text(
            "%s  SHA256SUMS\n" % manifest_sha, encoding="utf-8")
        os.rename(str(staging), str(output))
    except Exception:
        # Leave staging for forensic inspection; never publish a partial canonical result.
        raise

    print("M710_PASS verdict=%s delta_over_full=%.12f" % (
        verdict, overall["delta_over_full_product_work"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
