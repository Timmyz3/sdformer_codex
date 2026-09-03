#!/usr/bin/env python3
"""Read-only ep34 TSBG B2/B4/B8 same-I/O/cache quick-kill.

This analysis deliberately strengthens the TSBG baseline.  Both arms use the
same captured FC1+FC2 token order, an ordinary persistent LRU weight-row
buffer of exactly B rows, eight 16-byte/cycle banks (128 byte/cycle aggregate),
one 96-output INT8 weight row per cache entry, the same eight-source compute
service, and the same Acc24 commit work.  The baseline is therefore *not* an
uncached K1 stream.  TSBG only changes traversal order: B consecutive token
contexts share a fetched weight row, but their signed values and Acc24 state
remain private; products are never shared or dropped.

The candidate's complete token/context buffering is priced explicitly and a
conservative serialized cycle column adds one setup cycle per bundle.  This is
still a CPU premodel, not RTL timing, full-area equality, energy, or a paper
result.  Captured int8 codewords are diagnostic coordinates and do not grant a
model-bit-exact quantization claim.
"""
from __future__ import print_function

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = Path(__file__).resolve()
PREDECESSOR = HW / (
    "system_simulator/scripts/"
    "analyze_m1763_m1707_ep34_tsbg_layer_private_s2_witness_source.py")
CAPTURE = HW / (
    "results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_"
    "binary_capture_s40_r1_20260901")
M1763_RESULT = HW / (
    "results/m1763_m1707_ep34_tsbg_layer_private_s2_witness_r1_20260902")
CONTRACT = HW / (
    "contracts/tsbg_ep34_same_io_b2_b4_b8_quickkill_contract_r1_20260902.json")
RESULT = HW / "results/tsbg_ep34_same_io_b2_b4_b8_quickkill_r1_20260902"
WORK = HW / "results/.tsbg_ep34_same_io_b2_b4_b8_quickkill_r1_20260902.work"

PREDECESSOR_SHA256 = (
    "f86bbb02da1e259626539e83969c9e42cf9b34b7ee84c5072ffb9f6c3f70646c")
CAPTURE_MANIFEST_SHA256 = (
    "be0b89f9b8084baf0c2cd959805530a4e4f41c437a446335142a65bd73960a8f")
CAPTURE_OUTER_SHA256 = (
    "8d63d1054452377836c333c9848f771a6fc964e4ed4b00ed9a22f1537bd73c85")
M1763_DECISION_SHA256 = (
    "722aa302c983b63eae4e40816cffd123d0da34b09df56b872d52502d18cee961")
BUNDLES = (2, 4, 8)
GROUP_WIDTH = 16
OUTPUT_TILE = 96
SOURCES_PER_CYCLE = 8
ACC_BYTES = 3
WEIGHT_BYTES_PER_ELEMENT = 1
WEIGHT_ROW_BYTES = GROUP_WIDTH * OUTPUT_TILE * WEIGHT_BYTES_PER_ELEMENT
WEIGHT_BANKS = 8
WEIGHT_BANK_BYTES_PER_CYCLE = 16
WEIGHT_BYTES_PER_CYCLE = WEIGHT_BANKS * WEIGHT_BANK_BYTES_PER_CYCLE
CYCLE_GATE = 1.15
SEQUENCE_GATE = 1.05
WEIGHT_REDUCTION_GATE = 0.30
MAX_CYCLE_REGRESSION = 0.05


class QuickKillError(RuntimeError):
    pass


def need(value, message):
    if not value:
        raise QuickKillError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def verify_capture_tree():
    manifest = CAPTURE / "SHA256SUMS"
    outer = CAPTURE / "SHA256SUMS.seal.sha256"
    need(CAPTURE.is_dir() and manifest.is_file() and outer.is_file(),
         "capture or seals missing")
    need(sha256(manifest) == CAPTURE_MANIFEST_SHA256 and
         sha256(outer) == CAPTURE_OUTER_SHA256,
         "capture seal identity drift")
    need(outer.read_text(encoding="ascii").split() ==
         [CAPTURE_MANIFEST_SHA256, manifest.name], "capture outer seal drift")
    names = []
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, name = line.split(None, 1)
        name = name.strip().lstrip("*")
        need(name not in names and not Path(name).is_absolute() and
             ".." not in Path(name).parts, "unsafe capture manifest")
        need(sha256(CAPTURE / name) == digest,
             "capture member SHA drift: " + name)
        names.append(name)
    return names


need(sha256(PREDECESSOR) == PREDECESSOR_SHA256,
     "sealed M1763 predecessor drift")
_SPEC = importlib.util.spec_from_file_location("tsbg_exact_m1763", str(PREDECESSOR))
need(_SPEC is not None and _SPEC.loader is not None, "cannot import M1763")
M1763 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(M1763)
need(sha256(PREDECESSOR) == PREDECESSOR_SHA256,
     "sealed M1763 predecessor drift after import")
BASE = M1763.BASE


def ceil_div(a, b):
    return (int(a) + int(b) - 1) // int(b)


def add_metric(table, key, row):
    dst = table.setdefault(key, {})
    for name, value in row.items():
        if isinstance(value, list):
            current = dst.setdefault(name, [0] * len(value))
            need(len(current) == len(value), "metric vector width drift")
            dst[name] = [int(a) + int(b) for a, b in zip(current, value)]
        elif isinstance(value, bool):
            dst[name] = bool(dst.get(name, True) and value)
        elif isinstance(value, (int, float)):
            dst[name] = dst.get(name, 0) + value


def weight_service(misses_by_key, base_row, np):
    misses = np.asarray(misses_by_key, dtype=np.int64)
    keys = np.arange(misses.size, dtype=np.int64) + int(base_row)
    bank_misses = np.bincount((keys % WEIGHT_BANKS).astype(np.int32),
                              weights=misses, minlength=WEIGHT_BANKS)
    bank_bytes = [int(round(value)) * WEIGHT_ROW_BYTES
                  for value in bank_misses.tolist()]
    total_bytes = int(misses.sum()) * WEIGHT_ROW_BYTES
    aggregate_cycles = ceil_div(total_bytes, WEIGHT_BYTES_PER_CYCLE)
    bank_cycles = max([ceil_div(value, WEIGHT_BANK_BYTES_PER_CYCLE)
                       for value in bank_bytes] or [0])
    return total_bytes, bank_bytes, max(aggregate_cycles, bank_cycles)


def state_account(channels, source_groups, bundle):
    bitmap_bytes_one = ceil_div(source_groups, 8)
    baseline = {
        "row_buffer_bytes": bundle * WEIGHT_ROW_BYTES,
        "source_value_fifo_bytes": channels,
        "active_bitmap_bytes": bitmap_bytes_one,
        "acc24_context_bytes": OUTPUT_TILE * ACC_BYTES,
        "context_tag_bytes": 4,
    }
    candidate = {
        "row_buffer_bytes": bundle * WEIGHT_ROW_BYTES,
        "source_value_fifo_bytes": bundle * channels,
        "active_bitmap_bytes": bundle * bitmap_bytes_one,
        "acc24_context_bytes": bundle * OUTPUT_TILE * ACC_BYTES,
        "context_tag_bytes": bundle * 4,
    }
    baseline["total_explicit_state_bytes"] = sum(baseline.values())
    candidate["total_explicit_state_bytes"] = sum(candidate.values())
    return baseline, candidate


def pair_metrics(active, nnz, output_tiles, base_row, bundle, channels, np):
    baseline = BASE.exact_lru_entity_stats(active, output_tiles, bundle, np)
    bundled = BASE._bundle_active(active, bundle, np)
    candidate = BASE.exact_lru_entity_stats(bundled, output_tiles, bundle, np)
    b_bytes, b_banks, b_weight = weight_service(
        baseline["misses_by_key"], base_row, np)
    c_bytes, c_banks, c_weight = weight_service(
        candidate["misses_by_key"], base_row, np)
    compute = int(((nnz + SOURCES_PER_CYCLE - 1) //
                   SOURCES_PER_CYCLE).sum()) * int(output_tiles)
    commit = int(active.shape[0]) * int(output_tiles)
    b_schedule = int(baseline["accesses"])
    c_schedule = int(candidate["accesses"])
    bundle_setup = int(bundled.shape[0])
    b_roof = max(compute, commit, b_weight, b_schedule)
    c_roof = max(compute, commit, c_weight, c_schedule + bundle_setup)
    b_serial = compute + commit + b_weight + b_schedule
    c_serial = compute + commit + c_weight + c_schedule + bundle_setup
    b_state, c_state = state_account(channels, int(active.shape[1]), bundle)
    return {
        "tokens": int(active.shape[0]),
        "bundles": int(bundled.shape[0]),
        "baseline_weight_row_accesses": int(baseline["accesses"]),
        "baseline_weight_row_hits": int(baseline["hits"]),
        "baseline_weight_row_fetches": int(baseline["misses"]),
        "candidate_weight_row_accesses": int(candidate["accesses"]),
        "candidate_weight_row_hits": int(candidate["hits"]),
        "candidate_weight_row_fetches": int(candidate["misses"]),
        "baseline_weight_bytes": b_bytes,
        "candidate_weight_bytes": c_bytes,
        "baseline_weight_bank_bytes": b_banks,
        "candidate_weight_bank_bytes": c_banks,
        "baseline_weight_cycles": b_weight,
        "candidate_weight_cycles": c_weight,
        "compute_issue_cycles": compute,
        "commit_cycles": commit,
        "baseline_schedule_cycles": b_schedule,
        "candidate_schedule_cycles": c_schedule,
        "candidate_bundle_setup_cycles": bundle_setup,
        "baseline_roofline_cycles": b_roof,
        "candidate_roofline_cycles": c_roof,
        "baseline_serialized_cycles": b_serial,
        "candidate_serialized_cycles": c_serial,
        "baseline_explicit_state_byte_sum": b_state["total_explicit_state_bytes"],
        "candidate_explicit_state_byte_sum": c_state["total_explicit_state_bytes"],
        "max_baseline_explicit_state_bytes": b_state["total_explicit_state_bytes"],
        "max_candidate_explicit_state_bytes": c_state["total_explicit_state_bytes"],
        "max_incremental_state_bytes":
            c_state["total_explicit_state_bytes"] -
            b_state["total_explicit_state_bytes"],
        "same_cache_rows": True,
        "same_bank_and_bandwidth": True,
        "compute_work_changed": False,
    }


class Accumulator(object):
    def __init__(self, layers, samples, np):
        self.layers = dict((int(row["layer_id"]), row) for row in layers)
        self.samples = dict((int(row["global_sample_id"]), row)
                            for row in samples)
        self.np = np
        self.rows = {}
        self.pairs = 0
        self.tokens = 0

    def consume_pair(self, sample_id, layer_id, codes):
        np = self.np
        layer = self.layers[int(layer_id)]
        target = layer["target"]
        need(target in ("FC1", "FC2"), "unexpected target")
        value = np.asarray(codes, dtype=np.int8)
        channels = int(layer["input_channels"])
        need(value.ndim == 2 and value.shape[0] == int(layer["tokens_per_call"]) and
             value.shape[1] == channels, "captured frame shape drift")
        padded = ceil_div(channels, GROUP_WIDTH) * GROUP_WIDTH
        if padded != channels:
            value = np.pad(value, ((0, 0), (0, padded - channels)),
                           mode="constant")
        shaped = value.reshape(value.shape[0], -1, GROUP_WIDTH)
        nnz = (shaped != 0).sum(axis=2).astype(np.int16)
        active = nnz > 0
        layout = layer["weight_layout"]
        need(int(layout["source_group_count"]) == active.shape[1] and
             int(layout["bank_count"]) == WEIGHT_BANKS and
             int(layout["row_bytes"]) == GROUP_WIDTH * OUTPUT_TILE * 4,
             "captured weight layout drift")
        output_tiles = ceil_div(int(layer["output_channels"]), OUTPUT_TILE)
        base_row = int(layout["base_address"]) // int(layout["row_bytes"])
        sequence = self.samples[int(sample_id)]["sequence"]
        for bundle in BUNDLES:
            metric = pair_metrics(active, nnz, output_tiles, base_row, bundle,
                                  channels, np)
            for scope_type, scope in (("all", "FC1_FC2"),
                                      ("sequence", sequence),
                                      ("family", target),
                                      ("layer", layer["module_name"])):
                key = (bundle, scope_type, scope)
                # Max fields are maxima; all other numeric fields are sums.
                max_values = {name: metric.pop(name) for name in list(metric)
                              if name.startswith("max_")}
                add_metric(self.rows, key, metric)
                for name, item in max_values.items():
                    self.rows[key][name] = max(
                        int(self.rows[key].get(name, 0)), int(item))
                metric.update(max_values)
        self.pairs += 1
        self.tokens += int(value.shape[0])

    def finish(self):
        result = []
        for key in sorted(self.rows, key=lambda item: (item[0], item[1], item[2])):
            bundle, scope_type, scope = key
            row = dict(self.rows[key])
            b_bytes = int(row["baseline_weight_bytes"])
            c_bytes = int(row["candidate_weight_bytes"])
            b_roof = int(row["baseline_roofline_cycles"])
            c_roof = int(row["candidate_roofline_cycles"])
            b_serial = int(row["baseline_serialized_cycles"])
            c_serial = int(row["candidate_serialized_cycles"])
            row.update({
                "bundle": bundle,
                "scope_type": scope_type,
                "scope": scope,
                "ordinary_lru_capacity_rows": bundle,
                "row_buffer_bytes_each_arm": bundle * WEIGHT_ROW_BYTES,
                "weight_fetch_ratio": float(b_bytes) / float(c_bytes),
                "weight_byte_reduction": 1.0 - float(c_bytes) / float(b_bytes),
                "roofline_speedup": float(b_roof) / float(c_roof),
                "conservative_serialized_speedup":
                    float(b_serial) / float(c_serial),
                "cycle_gate_ge_1p15":
                    float(b_serial) / float(c_serial) >= CYCLE_GATE,
                "energy_branch_weight_reduction_ge_30pct":
                    1.0 - float(c_bytes) / float(b_bytes) >= WEIGHT_REDUCTION_GATE,
                "energy_branch_cycle_regression_le_5pct":
                    float(b_serial) / float(c_serial) >=
                    1.0 / (1.0 + MAX_CYCLE_REGRESSION),
            })
            result.append(row)
        return result


def seal(root):
    members = sorted(path.relative_to(root).as_posix()
                     for path in root.rglob("*") if path.is_file() and
                     path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join("{}  {}\n".format(sha256(root / name), name)
                                for name in members), encoding="ascii")
    (root / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(manifest)), encoding="ascii")


def run():
    need(not RESULT.exists() and not WORK.exists(), "fresh result required")
    verify_capture_tree()
    need(sha256(M1763_RESULT / "decision.json") == M1763_DECISION_SHA256,
         "M1763 decision drift")
    need(CONTRACT.is_file(), "quick-kill contract missing")
    manifest, receipt, sample_order, layer_document, tree = \
        BASE.verify_capture_identity(CAPTURE)
    del receipt
    m1558 = BASE.load_m1558()
    layers = layer_document["layers"]
    need(m1558.canonical_sha(m1558.frozen_layer_specs()) ==
         layer_document["inventory_sha256"], "layer inventory drift")
    try:
        import numpy as np
    except ImportError as error:
        raise QuickKillError("NumPy required") from error
    accumulator = Accumulator(layers, sample_order["samples"], np)
    frames = BASE.replay_capture(CAPTURE, accumulator, m1558, layers)
    need(frames == int(manifest["population"]["fc_frames"]) and
         accumulator.tokens == int(manifest["population"]["fc_tokens"]),
         "capture replay population drift")
    rows = accumulator.finish()
    all_rows = [row for row in rows if row["scope_type"] == "all"]
    sequence_rows = [row for row in rows if row["scope_type"] == "sequence"]
    need([row["bundle"] for row in all_rows] == list(BUNDLES),
         "aggregate B2/B4/B8 rows missing")
    decisions = []
    for row in all_rows:
        bundle = row["bundle"]
        seq = [item for item in sequence_rows if item["bundle"] == bundle]
        cycle_go = (row["conservative_serialized_speedup"] >= CYCLE_GATE and
                    all(item["conservative_serialized_speedup"] >=
                        SEQUENCE_GATE for item in seq))
        energy_go = (row["energy_branch_weight_reduction_ge_30pct"] and
                     row["energy_branch_cycle_regression_le_5pct"])
        decisions.append({
            "bundle": bundle,
            "cycle_gate": cycle_go,
            "energy_branch_gate": energy_go,
            "recommended_disposition":
                "GO_CPU_PREMODEL_ONLY__RTL_STILL_REQUIRES_SEPARATE_GATE"
                if cycle_go else
                ("ENERGY_ONLY_CPU_PREMODEL" if energy_go else "NO_GO"),
        })
    result = {
        "schema": "tsbg_ep34_same_io_b2_b4_b8_quickkill_r1_v1",
        "status": "CPU_PREMODEL_ONLY__NO_RTL_NO_EDA_NO_PAPER_ADMISSION",
        "date_cst": "2026-09-02",
        "identity": {
            "source_sha256": sha256(SOURCE),
            "contract_sha256": sha256(CONTRACT),
            "m1763_predecessor_source_sha256": PREDECESSOR_SHA256,
            "m1763_decision_sha256": M1763_DECISION_SHA256,
            "capture_manifest_sha256": CAPTURE_MANIFEST_SHA256,
            "capture_outer_seal_file_sha256": CAPTURE_OUTER_SHA256,
            "capture_tree_manifest_sha256": tree["manifest_sha256"],
            "checkpoint_sha256": manifest["identity"]["checkpoint_sha256"],
            "sample_order_sha256": sha256(CAPTURE / "sample_order.json"),
        },
        "population": {
            "samples": 40,
            "layers": len(layers),
            "fc_pairs": accumulator.pairs,
            "fc_frames": frames,
            "fc_tokens": accumulator.tokens,
        },
        "fair_baseline": {
            "name": "ordinary persistent same-capacity LRU-B weight-row buffer",
            "not_uncached_k1": True,
            "same_trace": True,
            "same_weight_row_cache_capacity": True,
            "same_weight_ports": True,
            "same_weight_bandwidth": True,
            "same_eight_source_compute_service": True,
            "same_acc24_commit_work": True,
            "candidate_context_state_priced": True,
            "candidate_context_state_equal_area": False,
            "full_same_area_claim": False,
        },
        "service": {
            "weight_element_bytes": WEIGHT_BYTES_PER_ELEMENT,
            "weight_row_bytes": WEIGHT_ROW_BYTES,
            "banks": WEIGHT_BANKS,
            "bank_bytes_per_cycle": WEIGHT_BANK_BYTES_PER_CYCLE,
            "aggregate_bytes_per_cycle": WEIGHT_BYTES_PER_CYCLE,
            "sources_per_cycle": SOURCES_PER_CYCLE,
            "output_tile": OUTPUT_TILE,
            "accumulator": "signed Acc24, private per token context",
            "product_sharing": False,
            "pruning_or_approximation": False,
        },
        "rows": rows,
        "decisions": decisions,
        "claim_boundary": {
            "same_io_and_cache_capacity_cpu_premodel": True,
            "same_area": False,
            "captured_codeword_model_bit_exact": False,
            "hardware_weight_quantization_authority": False,
            "rtl": False,
            "vcs": False,
            "eda": False,
            "energy": False,
            "component_speedup_admitted": False,
            "system_speedup": False,
            "paper_result": False,
        },
    }
    WORK.mkdir(parents=False)
    (WORK / "result.json").write_bytes(canonical(result))
    fields = [
        "bundle", "scope_type", "scope", "tokens", "bundles",
        "ordinary_lru_capacity_rows", "row_buffer_bytes_each_arm",
        "baseline_weight_row_fetches", "candidate_weight_row_fetches",
        "baseline_weight_bytes", "candidate_weight_bytes",
        "weight_byte_reduction", "baseline_roofline_cycles",
        "candidate_roofline_cycles", "roofline_speedup",
        "baseline_serialized_cycles", "candidate_serialized_cycles",
        "conservative_serialized_speedup", "max_incremental_state_bytes",
        "cycle_gate_ge_1p15"]
    with (WORK / "rows.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (WORK / "RUN_COMPLETE.txt").write_text(
        "PASS TSBG ep34 same-I/O/cache B2/B4/B8 CPU quick-kill\n",
        encoding="ascii")
    seal(WORK)
    os.replace(str(WORK), str(RESULT))
    return result


def self_check():
    verify_capture_tree()
    need(CONTRACT.is_file(), "quick-kill contract missing")
    return {
        "status": "PASS_SOURCE_SELF_CHECK__NO_ANALYSIS",
        "bundles": list(BUNDLES),
        "baseline": "ordinary persistent same-capacity LRU-B",
        "capture_present_and_sealed": True,
        "gpu_runs": 0,
        "eda_runs": 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-self-check", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    value = self_check() if args.source_self_check else run()
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
