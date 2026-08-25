#!/usr/bin/env python3
"""Prove M157 source-major reordering exact in the frozen INT8/Acc19 domain.

The proof has two independent parts.  First, every signed event mask is split
into disjoint low/high destination halves and reconstructed over the complete
20-record heldout population.  Second, the frozen INT8 checkpoint weights are
reduced to per-output-channel sum(abs(weight)) bounds.  If every bound fits a
signed19 accumulator, every prefix under any event/sign permutation also fits;
integer addition is then associative and the source-major permutation is exact.
"""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M157_SCRIPT = HW / (
    "system_simulator/scripts/analyze_m157_source_major_row_interleave_cache_dse.py")
M41_DIR = HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823"
M150_AUDIT = HW / (
    "results/m150_independent_hammer_review_r1_20260824/"
    "audit_m150_independent.py")
PATHS = {
    "m157_script": M157_SCRIPT,
    "m157_result": HW / (
        "results/m157_source_major_row_interleave_cache_dse_r2_20260824/"
        "m157_source_major_row_interleave_cache_dse.json"),
    "m157_manifest": HW / (
        "results/m157_source_major_row_interleave_cache_dse_r2_20260824/manifest.sha256"),
    "m41_result": M41_DIR / "m41_h67_ep35_bottleneck_int8_bridge.json",
    "m150_independent_audit": M150_AUDIT,
    "weight_o0": M41_DIR / "o0_weight_i_ky_kx_o_s8.bin",
    "weight_o1": M41_DIR / "o1_weight_i_ky_kx_o_s8.bin",
    "weight_o2": M41_DIR / "o2_weight_i_ky_kx_o_s8.bin",
    "weight_o3": M41_DIR / "o3_weight_i_ky_kx_o_s8.bin",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}
EXPECTED = {
    "m157_script": "de0a7d608e4ba25b25d9de09bc1576279b08ce72cab43f971700bc8c488d5a65",
    "m157_result": "b225e18d4a2014fc572ca334fdb37eba3e77813092fdc9175ac23dbac0ef514d",
    "m157_manifest": "fd23b9d0fc5f9384c4f1ca7729855471af174948245807cbcfea65b4b347bf60",
    "m41_result": "20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb",
    "m150_independent_audit": "3f975e74178a51d709f502a135792cebdac7f8422a61ff8277acf401c8433d9a",
    "weight_o0": "1197b961e08f4ca8f156c301280e7e3c630aea3b3bf68b0e78ee0f701e2e9f31",
    "weight_o1": "f0b8ed22f4fbefc7753e9eff12bec6880d7c199db6a78ccf7f2f6d1343e890d9",
    "weight_o2": "c2a5f5b2489dadc7b46892d40e12fd960f6ca0bd595ef238cdf9915bcb5f5c8a",
    "weight_o3": "f3d7f2587d2b72518d945dfb6e6b954d8b2d9627e491b74b879a36a5d031c6e1",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
ACC_BITS = 19
ACC_MIN = -(1 << (ACC_BITS - 1))
ACC_MAX = (1 << (ACC_BITS - 1)) - 1
FEATURES = 768 * 3 * 3
CHANNELS = 768


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(label, path):
    spec = importlib.util.spec_from_file_location(label, path)
    require(spec is not None and spec.loader is not None,
            "cannot import " + label)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(token):
        raise RuntimeError("non-standard JSON number: " + token)

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=reject)


def weight_bounds(m41):
    results = []
    for operator, (label, path) in zip(
            [row["operator"] for row in m41["layers"]],
            [("weight_o0", PATHS["weight_o0"]),
             ("weight_o1", PATHS["weight_o1"]),
             ("weight_o2", PATHS["weight_o2"]),
             ("weight_o3", PATHS["weight_o3"])]) :
        weight = np.fromfile(str(path), dtype=np.int8)
        require(weight.size == FEATURES * CHANNELS,
                "weight payload size drift")
        weight = weight.reshape(FEATURES, CHANNELS)
        sum_abs = np.abs(weight.astype(np.int16)).sum(axis=0, dtype=np.int64)
        layer = next(row for row in m41["layers"]
                     if row["operator"] == operator)
        claimed = layer["accumulator_bound"]
        require(int(sum_abs.max())
                == int(claimed["per_channel_sum_abs_q_maximum"]),
                "M41 maximum sum(abs) drift")
        require(int(sum_abs.min())
                == int(claimed["per_channel_sum_abs_q_minimum"]),
                "M41 minimum sum(abs) drift")
        require(bool(np.all(sum_abs <= ACC_MAX)),
                "frozen checkpoint exceeds signed19 positive bound")
        results.append({
            "operator": operator,
            "payload_label": label,
            "output_channels": CHANNELS,
            "features_per_output": FEATURES,
            "sum_abs_minimum": int(sum_abs.min()),
            "sum_abs_mean": float(sum_abs.mean()),
            "sum_abs_maximum": int(sum_abs.max()),
            "signed19_minimum": ACC_MIN,
            "signed19_maximum": ACC_MAX,
            "worst_positive_headroom": ACC_MAX - int(sum_abs.max()),
            "all_prefix_orders_fit_acc19": True,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), "refusing to overwrite M158 output")
    script_start = sha256(Path(__file__).resolve())
    observed = {label: sha256(path) for label, path in PATHS.items()}
    require(observed == EXPECTED, "M158 frozen input identity drift")

    m157 = load_module("m158_frozen_m157", M157_SCRIPT)
    m152 = m157.load_module("m158_frozen_m152", m157.M152_SCRIPT)
    m150 = m152.load_module("m158_frozen_m150", m152.M150_SCRIPT)
    m147 = m152.load_module("m158_frozen_m147", m150.M147_SCRIPT)
    observed_m147 = {label: sha256(path)
                     for label, path in m147.PATHS.items()}
    require(observed_m147 == m147.EXPECTED,
            "M158 inherited M147 input identity drift")
    audit = m147.load_module("m158_frozen_audit",
                             m147.PATHS["m141_audit"])
    m150_audit = load_module("m158_frozen_m150_audit", M150_AUDIT)
    m132 = m147.load_module("m158_frozen_m132", m147.PATHS["m132_script"])
    m105 = m147.load_module("m158_frozen_m105", m132.M105_SCRIPT)

    manifest = audit.strict_json(m147.PATHS["m40_manifest"])
    m72 = audit.strict_json(m147.PATHS["m72_result"])
    m41_for_widths = audit.strict_json(m147.PATHS["m41_result"])
    m41 = strict_json(PATHS["m41_result"])
    m157_result = strict_json(PATHS["m157_result"])
    heldout = sorted(
        (row for row in manifest["records"]
         if row["sample_id"] in range(5, 10)),
        key=lambda row: (row["sample_id"], row["operator_index"]))
    require(len(heldout) == 20, "M158 heldout extent drift")

    popcount = np.fromiter((bin(value).count("1")
                            for value in range(1 << 16)),
                           dtype=np.uint8, count=1 << 16)
    popcount8 = np.fromiter((bin(value).count("1")
                             for value in range(1 << 8)),
                            dtype=np.uint8, count=1 << 8)
    centers = m105.centers_array(m72)
    widths, _, _ = m105.build_width_catalog(m72, m41_for_widths)
    totals = Counter()
    per_record = []

    for record_index, record in enumerate(heldout):
        masks = m105.decode_natural_partition_masks(record, popcount)
        event_masks, negative_masks, _, _ = m150_audit.select_events(
            masks, record["operator_index"], centers, widths, popcount)
        record_totals = Counter()
        for source in range(16):
            event = ((event_masks >> source) & 1).astype(np.uint8,
                                                              copy=False)
            negative_source = ((negative_masks >> source) & 1).astype(
                np.uint8, copy=False)
            event_destination = np.zeros(event.shape[:2], dtype=np.uint8)
            for destination in range(8):
                event_destination |= event[:, :, destination] << destination
            negative_destination = np.zeros(event.shape[:2], dtype=np.uint8)
            for destination in range(8):
                negative_destination |= (
                    negative_source[:, :, destination] << destination)
            record_totals["source_keys"] += event_destination.size
            record_totals["active_source_keys"] += int(
                np.count_nonzero(event_destination))
            record_totals["negative_not_event_mismatches"] += int(
                np.count_nonzero(negative_destination
                                 & np.bitwise_not(event_destination)))
            event_reconstructed = ((event_destination & np.uint8(0x0f))
                                   | (event_destination & np.uint8(0xf0)))
            negative_reconstructed = (
                (negative_destination & np.uint8(0x0f))
                | (negative_destination & np.uint8(0xf0)))
            record_totals["event_half_reconstruction_mismatches"] += int(
                np.count_nonzero(event_reconstructed != event_destination))
            record_totals["negative_half_reconstruction_mismatches"] += int(
                np.count_nonzero(
                    negative_reconstructed != negative_destination))
            event_count = popcount8[event_destination]
            negative_count = popcount8[negative_destination]
            record_totals["source_events"] += int(event_count.sum())
            record_totals["negative_source_events"] += int(
                negative_count.sum())
            record_totals["positive_source_events"] += int(
                (event_count - negative_count).sum())
            record_totals["mixed_sign_source_keys"] += int(
                np.count_nonzero((negative_count != 0)
                                 & (event_count != negative_count)))
        per_record.append({
            "sample_id": int(record["sample_id"]),
            "operator_index": int(record["operator_index"]),
            **dict(record_totals),
        })
        totals.update(record_totals)
        print("[M158 RECORD] {}/20 sample={} op={} events={} neg={}".format(
            record_index + 1, record["sample_id"],
            record["operator_index"], record_totals["source_events"],
            record_totals["negative_source_events"]), flush=True)

    require(totals["source_keys"] == 20 * 432 * 3000 * 16,
            "M158 signed source-key population drift")
    require(totals["active_source_keys"]
            == m157_result["exact_work"]["source_active_keys"],
            "M158 active source-key conservation drift")
    require(totals["source_events"]
            == m157_result["exact_work"]["source_events"],
            "M158 source-event conservation drift")
    require(totals["positive_source_events"]
            + totals["negative_source_events"] == totals["source_events"],
            "M158 signed event partition drift")
    for key in ("negative_not_event_mismatches",
                "event_half_reconstruction_mismatches",
                "negative_half_reconstruction_mismatches"):
        require(totals[key] == 0, key + " is nonzero")

    bounds = weight_bounds(m41)
    maximum = max(row["sum_abs_maximum"] for row in bounds)
    require(maximum == 218338, "global frozen Acc19 bound drift")
    payload = {
        "schema": "m158_source_major_acc19_reorder_exactness_proof_v1",
        "status": "PASS_FROZEN_SIGNED_TUPLE_AND_ACC19_REORDER_PROOF",
        "identity": {
            "analyzer_start_end_sha256": script_start,
            "direct_inputs_sha256": observed,
            "inherited_m147_inputs_sha256": observed_m147,
        },
        "extent": {
            "lineage": "H67/Motion ep35 heldout sample IDs 5..9",
            "records": 20,
            "operators": 4,
            "signed_source_keys": totals["source_keys"],
        },
        "signed_tuple_proof": dict(totals),
        "per_record": per_record,
        "accumulator_bound_proof": {
            "weight_layout": "I_KY_KX_O_C_ORDER",
            "int8_weight_payloads": 4,
            "accumulator_bits": ACC_BITS,
            "accumulator_minimum": ACC_MIN,
            "accumulator_maximum": ACC_MAX,
            "global_maximum_per_channel_sum_abs_weight": maximum,
            "global_minimum_positive_headroom": ACC_MAX - maximum,
            "layers": bounds,
            "proof":
                "For every output channel, every reordered prefix magnitude is bounded by the sum of absolute frozen INT8 weights. Since that bound is <=218338<262143, no signed19 prefix overflows for any event subset or sign permutation; integer addition is therefore associative and commutative in the admitted domain.",
        },
        "admission": {
            "heldout_signed_tuple_half_split_exact": True,
            "frozen_int8_weight_identity": True,
            "all_reordered_prefixes_fit_acc19": True,
            "source_major_integer_reorder_exactness": True,
            "rtl_trace_miter": False,
            "cache_rtl": False,
            "integrated_accumulator_rtl": False,
            "runtime_overflow_detector_required_for_frozen_domain": False,
            "runtime_overflow_detector_required_for_untrusted_inputs": True,
            "physical_speedup": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
        "paper_safe_statement":
            "Across all 414,720,000 heldout source keys, source-major destination-half splitting preserves every positive/negative tuple exactly. Four frozen INT8 Conv payloads independently satisfy a maximum per-channel sum(abs(weight)) of 218,338, so every reordered prefix fits signed19 and the integer accumulation permutation is exact within the frozen checkpoint domain. RTL, macro PPA and speedup remain unadmitted.",
    }
    require(sha256(Path(__file__).resolve()) == script_start,
            "M158 analyzer changed during execution")
    output.mkdir(parents=True, exist_ok=False)
    (output / "m158_source_major_acc19_reorder_exactness.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(
        "PASS M158 source_keys={} events={} positive={} negative={} mixed={} "
        "split_mismatch=0 max_sum_abs={} acc19_headroom={} "
        "reorder_exact=true runtime_overflow_guard_frozen=false rtl=false "
        "physical_speedup=false system_speedup=false headline=false".format(
            totals["source_keys"], totals["source_events"],
            totals["positive_source_events"],
            totals["negative_source_events"],
            totals["mixed_sign_source_keys"], maximum, ACC_MAX - maximum),
        flush=True)


if __name__ == "__main__":
    main()
