#!/usr/bin/env python3
"""M1721 source for a post-M1707 TSBG/S2 fail-closed decision replay.

The production mode is deliberately unavailable until the exact M1707 result
tree exists.  When invoked later it will verify the M1709 authorization, the
complete M1707 double seal, the child receipt, sample order, frame order,
zlib/CRC extents and the selected ep34 checkpoint before producing a fresh,
double-sealed M1721 *decision* directory.

TSBG is evaluated only at B4 and B8.  Its strong comparator is an ordinary,
persistent, same-capacity LRU-B weight-row buffer.  Weight fetch bytes, exact
captured-codeword compute issue, and a finite-bandwidth roofline cycle model
are reported separately; a fetch ratio is never called cycle speedup.

S2 is restricted to FC1 and PATCH and inherits M1713's fixed FC2 NO-GO.  FC1
uses retained signed codewords and checkpoint-derived ceil(abs(weight)) bounds
to make 16x16 block keep/drop decisions.  PATCH has no retained token/block
values in the exact M1558/M1707 format, so it is reported as unavailable rather
than reconstructed from its histogram.  No paired AEE exists in this source;
all fields needed by a later paired forty-sample replay are emitted explicitly.

This file is CPython-3.6 syntax compatible.  NumPy and torch are imported only
inside the future production path.  Source authoring runs neither capture nor
analysis and touches no GPU, RTL or EDA tool.
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
import stat
import struct
import sys
import zlib


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = Path(__file__).resolve()
TEST = HW / (
    "system_simulator/tests/test_m1721_m1707_ep34_tsbg_b4_b8_s2_"
    "fc1_patch_decision_source.py")
CONTRACT = HW / (
    "contracts/m1721_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_"
    "source_contract_r1_20260901.json")

M1558_SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1558_motion_ep34_s2_tsbg_reduced_binary_source_r1.py")
M1707_SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1707_motion_ep34_s2_tsbg_deployment_complete_successor_r1.py")
M1707_CONTRACT = HW / (
    "contracts/m1707_motion_ep34_s2_tsbg_deployment_complete_"
    "successor_source_contract_r1_20260901.json")
M1709_RELEASE = HW / (
    "contracts/m1709_m1708_m1707_motion_ep34_s2_tsbg_deployment_"
    "complete_capture_release_r1_20260901.json")
M1713_SOURCE = HW / (
    "system_simulator/scripts/analyze_m1713_ep34_s2_fc_patch_zero_cost_"
    "upper_bound_fastkill.py")
M1713_RESULT = HW / (
    "results/m1713_ep34_s2_fc_patch_zero_cost_upper_bound_fastkill_"
    "r1_20260901")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
CHECKPOINT = HW / (
    "system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth")
CAPTURE = HW / (
    "results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_"
    "binary_capture_s40_r1_20260901")
RESULT = HW / (
    "results/m1721_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_"
    "r1_20260901")
WORK = HW / (
    "results/.m1721_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_"
    "r1_20260901.work")

EXPECTED = {
    M1558_SOURCE:
        "e6686564064ae3acda2bfcfc8c2d75061eb9cb591bc739d090bc03911469b089",
    M1707_SOURCE:
        "cd135c1d8936fcb973335d1710cc7f422cf2e27648c76e6d45d7bc23bf5f72f2",
    M1707_CONTRACT:
        "79b77c8c4d7671b4235635f649770baa9c0b79b2410198611c21061a6b2181ce",
    M1709_RELEASE:
        "015c40cfc6c288ad9f6f89da8e21bdff344d71335cc1474d9ec781f95bc962c3",
    Path(str(M1709_RELEASE) + ".sha256"):
        "b1b0386717e3fe53744afd9d2d2a99f6fce9e9b7592125436850a0f328086f7a",
    Path(str(M1709_RELEASE) + ".sha256.seal.sha256"):
        "9cda57dc7e29ba4701c54b5795fac854404b711dd1dc63869c24dfb605c253ff",
    M1713_SOURCE:
        "ab83f55e8f80aa3acafe4a8c11d7ff6a0a09e1be18f0a7e3d2ff1ce419fbcc39",
    M1713_RESULT / "result.json":
        "31fad91c3f5450c733b51211e108297aaaf7f5d3a9fcfb3a0ef2d943fa30c63e",
    M1713_RESULT / "SHA256SUMS":
        "4c545098414445ab48cee85eecf50b4ec7c3bf77a73abc20963549726858b216",
    M1713_RESULT / "SHA256SUMS.seal.sha256":
        "7d54277522600f50c47e0659d81f3d2700cdde66ddfac3a8d0f7f49072de30c4",
    CHECKPOINT:
        "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48",
    DOCS359:
        "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

SCHEMA = "m1721_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_r1_v1"
STATUS = "DECISION_ONLY__TSBG_B4_B8__S2_FC1_REAL_VALUES__PATCH_BLOCKED__FC2_FIXED_NO_GO__NO_PAPER_RESULT"
BUNDLES = (4, 8)
S2_EPSILON_RATIO = (0.0, 0.01, 0.02, 0.05, 0.10)
GROUP_WIDTH = 16
TSBG_OUTPUT_TILE = 96
S2_OUTPUT_TILE = 16
SOURCES_PER_CYCLE = 8
WEIGHT_BYTES_PER_ELEMENT = 4
WEIGHT_BYTES_PER_CYCLE = 128
WEIGHT_BANKS = 8
WEIGHT_BANK_BYTES_PER_CYCLE = 16
ACC_BYTES = 3
TSBG_AGGREGATE_SPEEDUP_MIN = 1.15
TSBG_SEQUENCE_SPEEDUP_MIN = 1.05
TSBG_ENERGY_BRANCH_MAX_CYCLE_REGRESSION = 0.05
TSBG_WEIGHT_BYTE_REDUCTION_MIN = 0.30
S2_METADATA_BYTES = 2
S2_LOCAL_SPEEDUP_MIN = 1.15
S2_OVERALL_AEE_DELTA_MAX = 0.02
S2_SEQUENCE_AEE_DELTA_MAX = 0.03
S2_METADATA_RATIO_MAX = 0.02
CHECKPOINT_SHA256 = EXPECTED[CHECKPOINT]
M1707_RECEIPT = "m1707_clean_child_receipt.json"


class M1721Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1721Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise M1721Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA mismatch")


def strict_json(path, root_type=dict):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            M1721Error("nonfinite JSON: " + token)))
    require(type(value) is root_type, "JSON root type mismatch")
    return value


def canonical_json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def canonical_sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def ceil_div(numerator, denominator):
    return (int(numerator) + int(denominator) - 1) // int(denominator)


def verify_static_authorities():
    for path, expected in EXPECTED.items():
        regular_exact(path, expected, str(path.relative_to(ROOT)))
    release = strict_json(M1709_RELEASE)
    require(release.get("status") ==
            "AUTHORIZE_ONE_M1707_EP34_S2_TSBG_DEPLOYMENT_COMPLETE_CAPTURE" and
            release.get("identity", {}).get("source_sha256") ==
                EXPECTED[M1707_SOURCE] and
            release.get("identity", {}).get("checkpoint_sha256") ==
                CHECKPOINT_SHA256 and
            release.get("authorization") == {
                "parent_calls": 1, "clean_child_processes": 1,
                "gpu_runs": 1, "production_captures": 1,
                "automatic_retry": False, "all_other_runs": 0} and
            release.get("claim_boundary", {}).get("paper_result") is False,
            "M1709 release identity/budget drift")
    sidecar = Path(str(M1709_RELEASE) + ".sha256")
    outer = Path(str(M1709_RELEASE) + ".sha256.seal.sha256")
    require(sidecar.read_text(encoding="ascii").split() ==
            [EXPECTED[M1709_RELEASE], M1709_RELEASE.name],
            "M1709 release digest sidecar drift")
    require(outer.read_text(encoding="ascii").split() ==
            [EXPECTED[sidecar], sidecar.name],
            "M1709 release outer seal drift")
    m1713 = strict_json(M1713_RESULT / "result.json")
    fc2 = [row for row in m1713.get("family_upper_bounds", [])
           if row.get("object") == "fc2"]
    require(len(fc2) == 1 and fc2[0].get("direct_no_go_below_1p15") is True and
            fc2[0].get("decision") == "NO_GO_AS_S2_TARGET" and
            m1713.get("decision", {}).get("fc2") ==
                "DIRECT_NO_GO_EVEN_IF_ALL_REMAINING_FC2_WORK_IS_FREE",
            "M1713 FC2 mathematical NO-GO drift")
    return release


def load_m1558():
    regular_exact(M1558_SOURCE, EXPECTED[M1558_SOURCE], "exact M1558 source")
    spec = importlib.util.spec_from_file_location("m1721_exact_m1558",
                                                  str(M1558_SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot import exact M1558")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    regular_exact(M1558_SOURCE, EXPECTED[M1558_SOURCE],
                  "exact M1558 source after import")
    return module


def verify_tree(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(),
            "capture root must be a directory non-symlink")
    sums = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(sums.is_file() and not sums.is_symlink() and
            outer.is_file() and not outer.is_symlink(),
            "capture inner/outer seal missing")
    require(outer.read_text(encoding="ascii").split() ==
            [sha256(sums), "SHA256SUMS"], "capture outer seal drift")
    names = []
    for line in sums.read_text(encoding="ascii").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and len(fields[0]) == 64,
                "malformed capture SHA manifest")
        digest, name = fields[0], fields[1].strip()
        require(name and name not in names and
                Path(name).as_posix() == name and
                not Path(name).is_absolute() and ".." not in Path(name).parts,
                "unsafe/duplicate capture member")
        member = root / name
        require(member.is_file() and not member.is_symlink() and
                sha256(member) == digest,
                "capture member SHA drift: " + name)
        names.append(name)
    actual = sorted(path.relative_to(root).as_posix()
                    for path in root.rglob("*") if path.is_file() and
                    path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    require(sorted(names) == actual, "capture SHA manifest coverage drift")
    required = {"RUN_COMPLETE.txt", "capture_manifest.json", "fc_frames.bin",
                "layers.json", "patch_s1_histogram_debt.jsonl.zlib",
                "preload_permit_receipt.json", "sample_order.json",
                M1707_RECEIPT}
    require(required.issubset(set(names)), "M1707 result members incomplete")
    return {"manifest_sha256": sha256(sums),
            "outer_seal_file_sha256": sha256(outer),
            "members": names}


def verify_capture_identity(root):
    verify_static_authorities()
    tree = verify_tree(root)
    manifest = strict_json(Path(root) / "capture_manifest.json")
    receipt = strict_json(Path(root) / M1707_RECEIPT)
    samples = strict_json(Path(root) / "sample_order.json")
    layers = strict_json(Path(root) / "layers.json")
    require(manifest.get("schema") ==
            "m1558_reduced_binary_capture_manifest_r1_v1" and
            manifest.get("status") ==
                "PRODUCTION_PAYLOAD_REQUIRES_INDEPENDENT_RELEASE_AND_HAMMER" and
            manifest.get("identity", {}).get("checkpoint_sha256") ==
                CHECKPOINT_SHA256 and
            manifest.get("population", {}).get("samples") == 40 and
            manifest.get("population", {}).get("layers") == 32 and
            manifest.get("encoding", {}).get("patch_per_token_payload") is False and
            manifest.get("claim_boundary", {}).get(
                "hardware_quantization_authority") is False and
            manifest.get("claim_boundary", {}).get("model_bit_exact") is False,
            "M1707/M1558 capture manifest drift")
    require(receipt.get("schema") ==
            "m1707_ep34_s2_tsbg_deployment_complete_receipt_r1_v1" and
            receipt.get("status") ==
                "PAYLOAD_COMPLETE__FRESH_DIFFERENT_AUTHOR_RESULT_HAMMER_REQUIRED" and
            receipt.get("identity", {}).get("source_sha256") ==
                EXPECTED[M1707_SOURCE] and
            receipt.get("identity", {}).get("source_contract_sha256") ==
                EXPECTED[M1707_CONTRACT] and
            receipt.get("identity", {}).get("release_sha256") ==
                EXPECTED[M1709_RELEASE] and
            receipt.get("identity", {}).get("checkpoint_sha256") ==
                CHECKPOINT_SHA256 and
            receipt.get("checkpoint_load") == {
                "missing_count": 0, "unexpected_count": 0,
                "overlay_missing_count": 0, "overlay_unexpected_count": 0} and
            receipt.get("population", {}).get("samples") == 40 and
            receipt.get("claim_boundary", {}).get("paper_result") is False,
            "M1707 child receipt drift")
    require(samples.get("schema") == "m1544_ep34_sample_order_r1_v1" and
            len(samples.get("samples", [])) == 40 and
            [row.get("global_sample_id") for row in samples["samples"]] ==
                list(range(40)) and
            samples.get("identity", {}).get("checkpoint_sha256") ==
                CHECKPOINT_SHA256,
            "M1707 sample order drift")
    require(layers.get("schema") == "m1558_static_layers_r1_v1" and
            len(layers.get("layers", [])) == 32 and
            layers.get("inventory_sha256") ==
                manifest.get("identity", {}).get("inventory_sha256"),
            "M1707 static layer inventory drift")
    return manifest, receipt, samples, layers, tree


def _reference_lru(accesses, capacity, initial=()):
    require(type(capacity) is int and capacity > 0, "LRU capacity invalid")
    cache = list(initial)
    require(len(cache) <= capacity and len(set(cache)) == len(cache),
            "initial LRU state invalid")
    misses = 0
    hits = []
    for key in accesses:
        key = int(key)
        if key in cache:
            cache.remove(key)
            hits.append(key)
        else:
            misses += 1
            if len(cache) == capacity:
                cache.pop(0)
        cache.append(key)
    return misses, tuple(cache), hits


def _prefix_suffix(access_mask, capacity, np):
    rows, width = access_mask.shape
    require(width >= capacity, "access universe smaller than LRU capacity")
    positions = np.arange(width, dtype=np.int32)[None, :]
    low = np.where(access_mask, positions, width)
    prefix = np.partition(low, capacity - 1, axis=1)[:, :capacity]
    prefix.sort(axis=1)
    high = np.where(access_mask, positions, -1)
    suffix = np.partition(high, width - capacity, axis=1)[:, -capacity:]
    suffix.sort(axis=1)
    return prefix.astype(np.int32, copy=False), suffix.astype(np.int32, copy=False)


def _vector_lru_hits(initial_cache, current_prefix, universe, np):
    """Exact C-step LRU transitions for rows with >=C distinct accesses."""
    cache = initial_cache.copy()
    capacity = int(cache.shape[1])
    hits_by_key = np.zeros(int(universe), dtype=np.int64)
    hit_total = 0
    columns = np.arange(capacity - 1, dtype=np.int32)[None, :]
    for step in range(capacity):
        key = current_prefix[:, step]
        matches = cache == key[:, None]
        hit = matches.any(axis=1)
        hit_total += int(hit.sum())
        if bool(hit.any()):
            hits_by_key += np.bincount(key[hit], minlength=int(universe))
        index = matches.argmax(axis=1).astype(np.int32, copy=False)
        gather = columns + (columns >= index[:, None]).astype(np.int32)
        removed = np.take_along_axis(cache, gather, axis=1)
        hit_cache = np.concatenate((removed, key[:, None]), axis=1)
        miss_cache = np.concatenate((cache[:, 1:], key[:, None]), axis=1)
        cache = np.where(hit[:, None], hit_cache, miss_cache)
    return hit_total, hits_by_key


def exact_lru_entity_stats(active_groups, output_tiles, capacity, np):
    """Exact persistent LRU statistics without expanding nonzero products.

    Each entity has a distinct canonical access list: output tile major, then
    ascending active source group.  Long-to-long boundaries are evaluated in
    vectorized batches; only entities with fewer than C accesses use the small
    scalar fallback.  This makes runtime scale with decoded frames and sparse
    boundary cases, not with signed product count.
    """
    active = np.asarray(active_groups, dtype=np.bool_)
    require(active.ndim == 2 and active.shape[0] > 0 and
            active.shape[1] > 0 and output_tiles > 0 and capacity > 0,
            "invalid LRU entity matrix")
    access_mask = np.tile(active, (1, int(output_tiles)))
    counts = access_mask.sum(axis=1).astype(np.int64)
    universe = int(access_mask.shape[1])
    effective_capacity = min(int(capacity), universe)
    accesses_by_key = access_mask.sum(axis=0).astype(np.int64)
    total_accesses = int(counts.sum())
    prefix, suffix = _prefix_suffix(access_mask, effective_capacity, np)
    long_rows = counts >= effective_capacity
    safe_indices = np.nonzero(long_rows[:-1] & long_rows[1:])[0] + 1
    hit_total = 0
    hits_by_key = np.zeros(universe, dtype=np.int64)
    if int(safe_indices.size):
        safe_hits, safe_keys = _vector_lru_hits(
            suffix[safe_indices - 1], prefix[safe_indices], universe, np)
        hit_total += safe_hits
        hits_by_key += safe_keys

    unsafe = np.nonzero(np.concatenate((np.array([True], dtype=np.bool_),
        ~(long_rows[:-1] & long_rows[1:]))))[0]
    cache = []
    previous_index = -2
    for raw_index in unsafe.tolist():
        index = int(raw_index)
        if index == 0:
            cache = []
        elif previous_index != index - 1:
            cache = [int(value) for value in suffix[index - 1].tolist()]
        keys = np.flatnonzero(access_mask[index]).tolist()
        misses, final_cache, hit_keys = _reference_lru(
            keys, effective_capacity, cache)
        del misses
        cache = list(final_cache)
        hit_total += len(hit_keys)
        if hit_keys:
            hits_by_key += np.bincount(np.asarray(hit_keys, dtype=np.int32),
                                       minlength=universe)
        previous_index = index
    misses_by_key = accesses_by_key - hits_by_key
    require(bool((misses_by_key >= 0).all()) and
            int(misses_by_key.sum()) == total_accesses - hit_total,
            "LRU miss accounting drift")
    return {"accesses": total_accesses, "hits": hit_total,
            "misses": total_accesses - hit_total,
            "accesses_by_key": accesses_by_key,
            "misses_by_key": misses_by_key}


def _bundle_active(active_groups, bundle, np):
    active = np.asarray(active_groups, dtype=np.bool_)
    tail = (-int(active.shape[0])) % int(bundle)
    if tail:
        active = np.pad(active, ((0, tail), (0, 0)), mode="constant")
    return active.reshape(-1, int(bundle), active.shape[1]).any(axis=1)


def weight_cycles(misses_by_key, row_bytes, base_row, np):
    misses = np.asarray(misses_by_key, dtype=np.int64)
    total_bytes = int(misses.sum()) * int(row_bytes)
    keys = np.arange(misses.size, dtype=np.int64) + int(base_row)
    bank_misses = np.bincount((keys % WEIGHT_BANKS).astype(np.int32),
                              weights=misses, minlength=WEIGHT_BANKS)
    bank_bytes = [int(round(value)) * int(row_bytes)
                  for value in bank_misses.tolist()]
    aggregate = ceil_div(total_bytes, WEIGHT_BYTES_PER_CYCLE)
    bank_limited = max([ceil_div(value, WEIGHT_BANK_BYTES_PER_CYCLE)
                        for value in bank_bytes] or [0])
    return total_bytes, bank_bytes, max(aggregate, bank_limited)


def tsbg_pair_metrics(active_groups, nnz_by_group, output_tiles, row_bytes,
                      base_row, bundle, np):
    active = np.asarray(active_groups, dtype=np.bool_)
    nnz = np.asarray(nnz_by_group, dtype=np.int16)
    require(active.shape == nnz.shape and
            bool((nnz >= 0).all()) and bool((nnz <= GROUP_WIDTH).all()) and
            bool((active == (nnz > 0)).all()), "TSBG group payload drift")
    baseline = exact_lru_entity_stats(active, output_tiles, bundle, np)
    bundled_active = _bundle_active(active, bundle, np)
    candidate = exact_lru_entity_stats(
        bundled_active, output_tiles, bundle, np)
    baseline_bytes, baseline_banks, baseline_weight_cycles = weight_cycles(
        baseline["misses_by_key"], row_bytes, base_row, np)
    candidate_bytes, candidate_banks, candidate_weight_cycles = weight_cycles(
        candidate["misses_by_key"], row_bytes, base_row, np)
    compute_cycles = int(((nnz + SOURCES_PER_CYCLE - 1) //
                          SOURCES_PER_CYCLE).sum()) * int(output_tiles)
    commit_cycles = int(active.shape[0]) * int(output_tiles)
    baseline_schedule = int(baseline["accesses"])
    candidate_schedule = int(candidate["accesses"])
    baseline_roof = max(compute_cycles, commit_cycles,
                        baseline_weight_cycles, baseline_schedule)
    candidate_roof = max(compute_cycles, commit_cycles,
                         candidate_weight_cycles, candidate_schedule)
    return {
        "tokens": int(active.shape[0]), "bundle_count": int(bundled_active.shape[0]),
        "ordinary_lru_capacity_rows": int(bundle),
        "baseline_weight_row_accesses": int(baseline["accesses"]),
        "baseline_weight_row_hits": int(baseline["hits"]),
        "baseline_weight_row_fetches": int(baseline["misses"]),
        "candidate_weight_row_accesses": int(candidate["accesses"]),
        "candidate_weight_row_hits": int(candidate["hits"]),
        "candidate_weight_row_fetches": int(candidate["misses"]),
        "baseline_weight_fetch_bytes": baseline_bytes,
        "candidate_weight_fetch_bytes": candidate_bytes,
        "candidate_fetch_not_greater_than_baseline":
            candidate_bytes <= baseline_bytes,
        "baseline_weight_bank_bytes": baseline_banks,
        "candidate_weight_bank_bytes": candidate_banks,
        "compute_issue_cycles": compute_cycles,
        "commit_cycles": commit_cycles,
        "baseline_weight_cycles": baseline_weight_cycles,
        "candidate_weight_cycles": candidate_weight_cycles,
        "baseline_schedule_cycles": baseline_schedule,
        "candidate_schedule_cycles": candidate_schedule,
        "baseline_roofline_cycles": baseline_roof,
        "candidate_roofline_cycles": candidate_roof,
    }


def s2_fc1_pair_metrics(active_groups, nnz_by_group, abs_sum_by_group,
                        output_channels, beta_by_output_block, epsilon, np):
    active = np.asarray(active_groups, dtype=np.bool_)
    nnz = np.asarray(nnz_by_group, dtype=np.int16)
    magnitude = np.asarray(abs_sum_by_group, dtype=np.int32)
    betas = np.asarray(beta_by_output_block, dtype=np.int32)
    output_blocks = ceil_div(output_channels, S2_OUTPUT_TILE)
    require(active.shape == nnz.shape == magnitude.shape and
            betas.ndim == 1 and int(betas.size) == output_blocks and
            bool((betas > 0).all()) and bool((magnitude >= 0).all()) and
            bool((active == (nnz > 0)).all()), "S2 FC1 retained-value drift")
    eps = float(epsilon)
    require(eps in S2_EPSILON_RATIO, "S2 epsilon axis drift")
    threshold_abs_sum = int(math.floor(eps * GROUP_WIDTH * 127.0 + 1.0e-12))
    drop_group = active & (magnitude <= threshold_abs_sum)
    if eps == 0.0:
        drop_group[:] = False
    active_groups_count = int(active.sum())
    dropped_groups = int(drop_group.sum())
    baseline_blocks = active_groups_count * output_blocks
    dropped_blocks = dropped_groups * output_blocks
    baseline_products = int(nnz.sum()) * int(output_channels)
    saved_products = int(nnz[drop_group].sum()) * int(output_channels)
    block_weight_bytes = GROUP_WIDTH * S2_OUTPUT_TILE * WEIGHT_BYTES_PER_ELEMENT
    metadata_bytes = 0 if eps == 0.0 else baseline_blocks * S2_METADATA_BYTES
    baseline_weight_bytes = baseline_blocks * block_weight_bytes
    drop_abs_per_token = (magnitude * drop_group.astype(np.int32)).sum(axis=1)
    max_accumulated = (0 if not int(drop_abs_per_token.size) else
                       int(drop_abs_per_token.max()) * int(betas.max()))
    sum_debt = int((magnitude[drop_group]).sum()) * int(betas.sum())
    max_block_debt = (0 if dropped_groups == 0 else
                      int(magnitude[drop_group].max()) * int(betas.max()))
    drop_seen = drop_group.any(axis=0)
    keep_seen = (active & ~drop_group).any(axis=0)
    packed = np.packbits(drop_group, axis=1, bitorder="little").tobytes()
    return {
        "epsilon_ratio": eps, "threshold_abs_code_sum": threshold_abs_sum,
        "tokens": int(active.shape[0]), "baseline_nonzero_blocks": baseline_blocks,
        "kept_blocks": baseline_blocks - dropped_blocks,
        "dropped_blocks": dropped_blocks,
        "metadata_bytes": metadata_bytes,
        "baseline_weight_bytes": baseline_weight_bytes,
        "saved_weight_bytes": dropped_blocks * block_weight_bytes,
        "baseline_nonzero_products": baseline_products,
        "saved_nonzero_products": saved_products,
        "saved_psum_update_events": dropped_blocks,
        "max_dropped_block_abs_output_code_debt": max_block_debt,
        "max_accumulated_abs_output_code_debt_per_token": max_accumulated,
        "sum_abs_output_code_debt": sum_debt,
        "drop_seen_by_source_group": drop_seen,
        "keep_seen_by_source_group": keep_seen,
        "decision_payload": packed,
    }


def checkpoint_fc1_betas(layer_rows):
    regular_exact(CHECKPOINT, CHECKPOINT_SHA256, "selected ep34 checkpoint")
    try:
        import torch
    except ImportError as error:
        raise M1721Error("future production analysis requires CPU torch") from error
    value = torch.load(str(CHECKPOINT), map_location="cpu")
    require(type(value) is dict and type(value.get("model_state_dict")) is not type(None),
            "checkpoint model_state_dict missing")
    state = value["model_state_dict"]
    result = {}
    for row in layer_rows:
        if row.get("target") != "FC1":
            continue
        key = row["module_name"] + ".weight"
        require(key in state, "FC1 checkpoint weight missing: " + key)
        tensor = state[key].detach().cpu()
        require(len(tensor.shape) == 2 and
                int(tensor.shape[0]) == int(row["output_channels"]) and
                int(tensor.shape[1]) == int(row["input_channels"]),
                "FC1 checkpoint weight shape drift")
        betas = []
        for begin in range(0, int(tensor.shape[0]), S2_OUTPUT_TILE):
            maximum = float(tensor[begin:begin + S2_OUTPUT_TILE].abs().max().item())
            require(math.isfinite(maximum), "nonfinite FC1 checkpoint weight")
            betas.append(max(1, int(math.ceil(maximum))))
        require(max(betas) * GROUP_WIDTH * 127 <= 65535,
                "uint16 CCBS bound cannot represent worst block debt")
        result[int(row["layer_id"])] = betas
    require(len(result) == 12, "FC1 checkpoint beta inventory drift")
    del value
    return result


def _new_sum_metric():
    return {}


def _add_sum_metric(store, key, metric, max_fields=()):
    row = store.setdefault(key, {})
    for name, value in metric.items():
        if name in ("ordinary_lru_capacity_rows", "epsilon_ratio",
                    "threshold_abs_code_sum"):
            if name in row:
                require(row[name] == value, "aggregate coordinate drift")
            else:
                row[name] = value
        elif name in max_fields:
            row[name] = max(row.get(name, 0), int(value))
        elif type(value) is list:
            prior = row.get(name, [0] * len(value))
            require(len(prior) == len(value), "aggregate vector drift")
            row[name] = [int(a) + int(b) for a, b in zip(prior, value)]
        elif type(value) in (int, float) and not isinstance(value, bool):
            row[name] = row.get(name, 0) + value
    return row


def _ratio(numerator, denominator):
    return None if int(denominator) == 0 else float(numerator) / float(denominator)


class DecisionAccumulator(object):
    def __init__(self, layer_rows, sample_rows, betas, np):
        self.layers = dict((int(row["layer_id"]), row) for row in layer_rows)
        self.samples = dict((int(row["global_sample_id"]), row)
                            for row in sample_rows)
        self.betas = betas
        self.np = np
        self.tsbg = {}
        self.s2 = {}
        self.s2_seen = {}
        self.s2_hash = dict((epsilon, hashlib.sha256())
                            for epsilon in S2_EPSILON_RATIO)
        self.pairs = 0
        self.frames = 0
        self.tokens = 0
        self.nonzero_codes = 0

    def consume_pair(self, sample_id, layer_id, codes):
        np = self.np
        row = self.layers[int(layer_id)]
        sample = self.samples[int(sample_id)]
        target = row["target"]
        require(target in ("FC1", "FC2"), "binary frame target drift")
        value = np.asarray(codes, dtype=np.int8)
        channels = int(row["input_channels"])
        require(value.ndim == 2 and int(value.shape[1]) == channels and
                int(value.shape[0]) == int(row["tokens_per_call"]),
                "pair code matrix shape drift")
        padded = ceil_div(channels, GROUP_WIDTH) * GROUP_WIDTH
        if padded != channels:
            value = np.pad(value, ((0, 0), (0, padded - channels)),
                           mode="constant")
        shaped = value.reshape(value.shape[0], -1, GROUP_WIDTH)
        nnz = (shaped != 0).sum(axis=2).astype(np.int16)
        active = nnz > 0
        magnitude = np.abs(shaped.astype(np.int16)).sum(axis=2).astype(np.int32)
        output_tiles = ceil_div(int(row["output_channels"]), TSBG_OUTPUT_TILE)
        layout = row["weight_layout"]
        row_bytes = int(layout["row_bytes"])
        base = int(layout["base_address"])
        require(row_bytes == GROUP_WIDTH * TSBG_OUTPUT_TILE * 4 and
                base % row_bytes == 0 and
                int(layout["source_group_count"]) == int(active.shape[1]) and
                int(layout["output_tile_count"]) == output_tiles and
                int(layout["bank_count"]) == WEIGHT_BANKS,
                "static weight layout drift")
        sequence = sample["sequence"]
        for bundle in BUNDLES:
            metric = tsbg_pair_metrics(active, nnz, output_tiles, row_bytes,
                                       base // row_bytes, bundle, np)
            for scope_type, scope in (("all", "FC1_FC2"),
                                      ("sequence", sequence),
                                      ("family", target),
                                      ("layer", row["module_name"])):
                _add_sum_metric(self.tsbg, (bundle, scope_type, scope), metric)

        if target == "FC1":
            for epsilon in S2_EPSILON_RATIO:
                metric = s2_fc1_pair_metrics(
                    active, nnz, magnitude, int(row["output_channels"]),
                    self.betas[int(layer_id)], epsilon, np)
                self.s2_hash[epsilon].update(struct.pack(
                    "<IId", int(sample_id), int(layer_id), float(epsilon)))
                self.s2_hash[epsilon].update(metric.pop("decision_payload"))
                drop_seen = metric.pop("drop_seen_by_source_group")
                keep_seen = metric.pop("keep_seen_by_source_group")
                for scope_type, scope in (("all", "FC1"),
                                          ("sequence", sequence),
                                          ("layer", row["module_name"])):
                    key = (epsilon, scope_type, scope)
                    _add_sum_metric(self.s2, key, metric, max_fields=(
                        "max_dropped_block_abs_output_code_debt",
                        "max_accumulated_abs_output_code_debt_per_token"))
                    seen = self.s2_seen.setdefault(key, {
                        "drop": np.zeros(drop_seen.shape, dtype=np.bool_),
                        "keep": np.zeros(keep_seen.shape, dtype=np.bool_),
                        "output_blocks": ceil_div(int(row["output_channels"]),
                                                  S2_OUTPUT_TILE)})
                    require(seen["drop"].shape == drop_seen.shape,
                            "S2 witness group shape drift")
                    seen["drop"] |= drop_seen
                    seen["keep"] |= keep_seen
        self.pairs += 1
        self.tokens += int(value.shape[0])
        self.nonzero_codes += int(nnz.sum())

    def finalize_tsbg_rows(self):
        rows = []
        for key in sorted(self.tsbg, key=lambda x: (x[0], x[1], x[2])):
            bundle, scope_type, scope = key
            metric = dict(self.tsbg[key])
            baseline_bytes = metric["baseline_weight_fetch_bytes"]
            candidate_bytes = metric["candidate_weight_fetch_bytes"]
            baseline_cycles = metric["baseline_roofline_cycles"]
            candidate_cycles = metric["candidate_roofline_cycles"]
            fetch_ratio = _ratio(baseline_bytes, candidate_bytes)
            cycle_ratio = _ratio(baseline_cycles, candidate_cycles)
            reduction = (0.0 if baseline_bytes == 0 else
                         1.0 - float(candidate_bytes) / float(baseline_bytes))
            metric.update({
                "bundle": bundle, "scope_type": scope_type, "scope": scope,
                "weight_fetch_ratio": fetch_ratio,
                "weight_fetch_reduction": reduction,
                "roofline_cycle_speedup": cycle_ratio,
                "fetch_ratio_is_cycle_speedup": False,
                "candidate_fetch_not_greater_than_baseline":
                    candidate_bytes <= baseline_bytes,
                "compute_work_changed": False,
                "same_capacity_ordinary_lru_baseline": True})
            rows.append(metric)
        all_rows = [row for row in rows if row["scope_type"] == "all"]
        sequence_rows = [row for row in rows if row["scope_type"] == "sequence"]
        require(len(all_rows) == len(BUNDLES) and
                len(sequence_rows) >= len(BUNDLES) * 2,
                "TSBG aggregate/sequence population drift")
        for row in rows:
            if row["scope_type"] == "all":
                row["aggregate_cycle_gate_ge_1p15"] = (
                    row["roofline_cycle_speedup"] >= TSBG_AGGREGATE_SPEEDUP_MIN)
            if row["scope_type"] == "sequence":
                row["sequence_cycle_gate_ge_1p05"] = (
                    row["roofline_cycle_speedup"] >= TSBG_SEQUENCE_SPEEDUP_MIN)
            row["energy_branch_weight_reduction_ge_30pct"] = (
                row["weight_fetch_reduction"] >= TSBG_WEIGHT_BYTE_REDUCTION_MIN)
            row["energy_branch_cycle_regression_le_5pct"] = (
                row["roofline_cycle_speedup"] >=
                1.0 / (1.0 + TSBG_ENERGY_BRANCH_MAX_CYCLE_REGRESSION))
        return rows

    def finalize_s2_rows(self):
        rows = []
        for key in sorted(self.s2, key=lambda x: (x[0], x[1], x[2])):
            epsilon, scope_type, scope = key
            metric = dict(self.s2[key])
            seen = self.s2_seen[key]
            witness = int((seen["drop"] & seen["keep"]).sum()) * int(
                seen["output_blocks"])
            baseline_blocks = metric["baseline_nonzero_blocks"]
            baseline_weights = metric["baseline_weight_bytes"]
            metric.update({
                "epsilon_ratio": epsilon, "scope_type": scope_type,
                "scope": scope,
                "drop_fraction_of_remaining_nonzero_blocks":
                    _ratio(metric["dropped_blocks"], baseline_blocks),
                "extra_nonzero_product_reduction":
                    _ratio(metric["saved_nonzero_products"],
                           metric["baseline_nonzero_products"]),
                "metadata_to_baseline_weight_bytes":
                    _ratio(metric["metadata_bytes"], baseline_weights),
                "dynamic_same_block_keep_drop_witness_count": witness,
                "paired_aee_present": False,
                "overall_delta_aee": None,
                "max_sequence_delta_aee": None,
                "same_resource_cycle_speedup": None,
                "passes_fixed_gate": False,
                "paper_admission": False})
            rows.append(metric)
        return rows


def _read_exact(stream, size):
    value = stream.read(int(size))
    require(len(value) == int(size), "truncated FC binary frame")
    return value


def replay_capture(capture_root, accumulator, m1558, layers):
    np = accumulator.np
    specs = [row for row in layers if row["target"] in ("FC1", "FC2")]
    expected_pairs = [(sample, int(row["layer_id"]))
                      for sample in range(40) for row in specs]
    by_id = dict((int(row["layer_id"]), row) for row in specs)
    pair_index = 0
    pair_frame = 0
    pair_token = 0
    chunks = []
    frames = 0
    with (Path(capture_root) / "fc_frames.bin").open("rb") as stream:
        while True:
            prefix = stream.read(m1558.FRAME_HEADER.size)
            if not prefix:
                break
            require(len(prefix) == m1558.FRAME_HEADER.size,
                    "truncated FC frame header")
            values = m1558.FRAME_HEADER.unpack(prefix)
            (magic, version, header_size, layer_id, sample_id, frame_index,
             token_start, token_count, channels, bitrow, nnz_total,
             raw_bytes, compressed_bytes, crc32) = values
            require(magic == m1558.FRAME_MAGIC and
                    version == m1558.FRAME_VERSION and
                    header_size == m1558.FRAME_HEADER.size and
                    pair_index < len(expected_pairs) and
                    (sample_id, layer_id) == expected_pairs[pair_index],
                    "FC frame identity/order drift")
            row = by_id[int(layer_id)]
            require(frame_index == pair_frame and token_start == pair_token and
                    channels == int(row["input_channels"]) and
                    0 < token_count <= m1558.FRAME_TOKENS and
                    bitrow == ceil_div(channels, 8) and
                    0 <= nnz_total <= token_count * channels and
                    0 < compressed_bytes < m1558.MAX_RUNTIME_BYTES,
                    "FC frame dimensions/count drift")
            compressed = _read_exact(stream, compressed_bytes)
            decoder = zlib.decompressobj()
            raw = decoder.decompress(compressed) + decoder.flush()
            require(decoder.eof and not decoder.unused_data and
                    not decoder.unconsumed_tail and len(raw) == raw_bytes and
                    (zlib.crc32(raw) & 0xffffffff) == crc32,
                    "FC frame zlib/CRC/extent drift")
            decoded = m1558.decode_frame_payload(
                raw, token_count, channels, bitrow, nnz_total,
                return_codes=True)
            chunks.append(decoded["codes"])
            frames += 1
            pair_frame += 1
            pair_token += token_count
            expected_tokens = int(row["tokens_per_call"])
            require(pair_token <= expected_tokens, "FC pair token overflow")
            if pair_token == expected_tokens:
                codes = np.concatenate(chunks, axis=0)
                accumulator.consume_pair(sample_id, layer_id, codes)
                pair_index += 1
                pair_frame = 0
                pair_token = 0
                chunks = []
    require(pair_index == len(expected_pairs) and pair_token == 0 and not chunks,
            "FC binary population incomplete")
    accumulator.frames = frames
    return frames


def validate_patch_histogram_only(capture_root, m1558, layers):
    expected = [(sample, int(row["layer_id"]), tile)
                for sample in range(40) for row in layers
                if row["target"] == "PATCH"
                for tile in range(ceil_div(int(row["output_channels"]),
                                           TSBG_OUTPUT_TILE))]
    count = 0
    decoder = zlib.decompressobj()
    buffer = b""
    with (Path(capture_root) /
          "patch_s1_histogram_debt.jsonl.zlib").open("rb") as stream:
        while True:
            chunk = stream.read(1 << 16)
            if not chunk:
                buffer += decoder.flush()
                break
            buffer += decoder.decompress(chunk)
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line:
                    continue
                row = json.loads(line.decode("utf-8"))
                require(count < len(expected) and
                        (int(row["sample_global_id"]), int(row["layer_id"]),
                         int(row["output_tile_id"])) == expected[count] and
                        row.get("per_token_payload_emitted") is False and
                        sum(int(value) for value in
                            row["count_by_magnitude_bin"]) ==
                            int(row["nonzero_source_count"]),
                        "PATCH histogram-only stream drift")
                count += 1
    require(decoder.eof and not decoder.unused_data and
            not decoder.unconsumed_tail and buffer == b"" and
            count == len(expected), "PATCH histogram stream incomplete")
    return count


def _write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _seal_result(root):
    members = sorted(path.relative_to(root).as_posix()
                     for path in root.rglob("*") if path.is_file() and
                     path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    sums = root / "SHA256SUMS"
    sums.write_text("".join("{}  {}\n".format(sha256(root / name), name)
                            for name in members), encoding="ascii")
    (root / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(sums)), encoding="ascii")


def run_analysis():
    require(not os.path.lexists(str(RESULT)) and
            not os.path.lexists(str(WORK)),
            "fresh M1721 result/work namespace required")
    manifest, receipt, sample_order, layer_document, tree = \
        verify_capture_identity(CAPTURE)
    m1558 = load_m1558()
    specs = m1558.frozen_layer_specs()
    layer_rows = layer_document["layers"]
    require(m1558.canonical_sha(specs) == layer_document["inventory_sha256"],
            "M1707/M1558 inventory SHA drift")
    static_fields = ("layer_id", "target", "module_name", "operator",
                     "operator_order", "channel_axis", "input_channels",
                     "output_channels", "tokens_per_call", "tokens_s40")
    for emitted, expected in zip(layer_rows, specs):
        require(all(emitted[field] == expected[field] for field in static_fields),
                "M1707/M1558 emitted layer drift")
    try:
        import numpy as np
    except ImportError as error:
        raise M1721Error("production analysis requires NumPy") from error
    betas = checkpoint_fc1_betas(layer_rows)
    accumulator = DecisionAccumulator(
        layer_rows, sample_order["samples"], betas, np)
    frames = replay_capture(CAPTURE, accumulator, m1558, layer_rows)
    patch_rows = validate_patch_histogram_only(CAPTURE, m1558, layer_rows)
    require(frames == int(manifest["population"]["fc_frames"]) and
            accumulator.tokens == int(manifest["population"]["fc_tokens"]) and
            patch_rows == int(manifest["population"]["patch_histogram_rows"]),
            "M1707 manifest population mismatch after replay")
    tsbg_rows = accumulator.finalize_tsbg_rows()
    s2_rows = accumulator.finalize_s2_rows()
    tsbg_all = [row for row in tsbg_rows if row["scope_type"] == "all"]
    sequence_gate = dict((bundle, all(
        row.get("sequence_cycle_gate_ge_1p05") is True for row in tsbg_rows
        if row["bundle"] == bundle and row["scope_type"] == "sequence"))
        for bundle in BUNDLES)
    tsbg_decisions = []
    for row in tsbg_all:
        admitted = (row.get("aggregate_cycle_gate_ge_1p15") is True and
                    sequence_gate[row["bundle"]])
        energy_only = (not admitted and
            row["energy_branch_weight_reduction_ge_30pct"] and
            row["energy_branch_cycle_regression_le_5pct"])
        tsbg_decisions.append({"bundle": row["bundle"],
            "cycle_path_admitted": admitted,
            "energy_only_path_eligible": energy_only,
            "ordinary_lru_same_capacity": True,
            "weight_fetch_ratio_is_not_cycle_speedup": True})

    identity = {
        "analyzer_sha256": sha256(SOURCE),
        "m1707_source_sha256": EXPECTED[M1707_SOURCE],
        "m1707_contract_sha256": EXPECTED[M1707_CONTRACT],
        "m1709_release_sha256": EXPECTED[M1709_RELEASE],
        "m1707_capture_manifest_sha256": sha256(CAPTURE / "capture_manifest.json"),
        "m1707_capture_inner_manifest_sha256": tree["manifest_sha256"],
        "m1707_capture_outer_seal_file_sha256": tree["outer_seal_file_sha256"],
        "m1707_receipt_sha256": sha256(CAPTURE / M1707_RECEIPT),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "m1713_result_sha256": EXPECTED[M1713_RESULT / "result.json"],
        "docs359_sha256": EXPECTED[DOCS359],
        "sample_order_sha256": sha256(CAPTURE / "sample_order.json"),
        "fc1_beta_sha256": canonical_sha(betas),
    }
    result = {
        "schema": SCHEMA, "status": STATUS, "date_cst": "2026-09-01",
        "identity": identity,
        "population": {"samples": 40, "layers": 32,
            "fc_pairs": accumulator.pairs, "fc_frames": frames,
            "fc_tokens": accumulator.tokens,
            "captured_nonzero_codes": accumulator.nonzero_codes,
            "patch_histogram_rows": patch_rows},
        "tsbg": {"bundles": list(BUNDLES),
            "baseline": "ordinary persistent same-capacity LRU-B weight-row buffer",
            "separate_axes": ["weight_fetch", "compute", "roofline_cycle"],
            "rows": tsbg_rows, "decisions": tsbg_decisions,
            "component_speedup_multiplication_allowed": False},
        "s2": {"geometry": "16x16", "epsilon_ratio_axis": list(S2_EPSILON_RATIO),
            "fc1_rows": s2_rows,
            "fc1_decision_sha256": dict((str(epsilon),
                accumulator.s2_hash[epsilon].hexdigest())
                for epsilon in S2_EPSILON_RATIO),
            "patch": {"status":
                "BLOCKED_M1707_PATCH_IS_HISTOGRAM_ONLY__NO_BLOCK_KEEP_DROP",
                "real_histogram_rows_read": patch_rows,
                "retained_token_or_block_values": False,
                "keep_drop_claim": False, "paired_aee_claim": False},
            "fc2": {"status": "FIXED_NO_GO_FROM_M1713_ZERO_COST_UPPER_BOUND",
                "evaluated_again": False},
            "paired_aee_present": False, "paper_admission": False},
        "claim_boundary": {"decision_only": True,
            "captured_codeword_and_contributor_scope_only": True,
            "hardware_quantization_authority": False,
            "model_bit_exact": False, "paired_aee": False,
            "rtl": False, "vcs": False, "eda": False,
            "energy": False, "system_speedup": False,
            "paper_result": False}}

    WORK.mkdir()
    (WORK / "decision.json").write_bytes(canonical_json_bytes(result))
    tsbg_fields = ["bundle", "scope_type", "scope", "tokens", "bundle_count",
        "ordinary_lru_capacity_rows", "baseline_weight_row_fetches",
        "candidate_weight_row_fetches", "baseline_weight_fetch_bytes",
        "candidate_weight_fetch_bytes", "weight_fetch_reduction",
        "weight_fetch_ratio", "compute_issue_cycles", "commit_cycles",
        "baseline_weight_cycles", "candidate_weight_cycles",
        "baseline_roofline_cycles", "candidate_roofline_cycles",
        "roofline_cycle_speedup", "fetch_ratio_is_cycle_speedup"]
    _write_csv(WORK / "tsbg_b4_b8_rows.csv", tsbg_rows, tsbg_fields)
    s2_fields = ["epsilon_ratio", "scope_type", "scope", "tokens",
        "baseline_nonzero_blocks", "kept_blocks", "dropped_blocks",
        "drop_fraction_of_remaining_nonzero_blocks", "metadata_bytes",
        "baseline_weight_bytes", "saved_weight_bytes",
        "baseline_nonzero_products", "saved_nonzero_products",
        "extra_nonzero_product_reduction", "saved_psum_update_events",
        "max_dropped_block_abs_output_code_debt",
        "max_accumulated_abs_output_code_debt_per_token",
        "sum_abs_output_code_debt", "dynamic_same_block_keep_drop_witness_count",
        "paired_aee_present", "overall_delta_aee", "max_sequence_delta_aee",
        "same_resource_cycle_speedup", "paper_admission"]
    _write_csv(WORK / "s2_fc1_rows.csv", s2_rows, s2_fields)
    paired = {"schema": "m1721_s2_paired_aee_required_fields_r1_v1",
        "status": "INPUT_REQUIRED__NO_AEE_RESULT",
        "identity_required": identity,
        "cohort_required": {"samples": 40,
            "exact_sample_order_sha256": identity["sample_order_sha256"]},
        "per_sample_required_fields": ["global_sample_id", "sequence",
            "sequence_sample_id", "epsilon_ratio", "baseline_aee",
            "candidate_aee", "baseline_prediction_sha256",
            "candidate_prediction_sha256", "ground_truth_sha256"],
        "aggregate_required_fields": ["ratio_of_sums_same_resource_cycles",
            "overall_mean_delta_aee", "per_sequence_mean_delta_aee"],
        "gates": {"overall_delta_aee_max": S2_OVERALL_AEE_DELTA_MAX,
            "per_sequence_delta_aee_max": S2_SEQUENCE_AEE_DELTA_MAX,
            "metadata_to_weight_bytes_max": S2_METADATA_RATIO_MAX,
            "local_same_resource_speedup_min": S2_LOCAL_SPEEDUP_MIN},
        "component_speedup_multiplication_allowed": False,
        "paper_admission": False}
    (WORK / "s2_paired_aee_required_fields.json").write_bytes(
        canonical_json_bytes(paired))
    (WORK / "RUN_COMPLETE.txt").write_text(
        "M1721_DECISION_ONLY_COMPLETE__NO_PAPER_RESULT\n", encoding="ascii")
    _seal_result(WORK)
    os.rename(str(WORK), str(RESULT))
    return result


def validate_source_contract():
    value = strict_json(CONTRACT)
    require(value.get("schema") ==
            "m1721_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_source_contract_r1_v1" and
            value.get("source", {}).get("path") ==
                str(SOURCE.relative_to(ROOT)) and
            value.get("source", {}).get("sha256") == sha256(SOURCE) and
            value.get("test", {}).get("path") == str(TEST.relative_to(ROOT)) and
            value.get("test", {}).get("sha256") == sha256(TEST) and
            value.get("authorization", {}).get("analysis_run") is False and
            value.get("claim_boundary", {}).get("paper_result") is False,
            "M1721 source contract drift")
    return value


def source_self_check():
    verify_static_authorities()
    validate_source_contract()
    require(BUNDLES == (4, 8) and S2_EPSILON_RATIO[0] == 0.0 and
            RESULT != CAPTURE and WORK != RESULT and
            not os.path.lexists(str(RESULT)) and
            not os.path.lexists(str(WORK)), "M1721 coordinate/namespace drift")
    misses, cache, hits = _reference_lru([0, 1, 2, 3, 0, 4], 4)
    require(misses == 5 and hits == [0] and cache == (1, 2, 3, 0, 4)[-4:],
            "ordinary LRU reference drift")
    return {"status": "PASS_M1721_SOURCE_SELF_CHECK__NO_CAPTURE_NO_ANALYSIS",
        "bundles": list(BUNDLES),
        "tsbg_strong_baseline": "ordinary_persistent_same_capacity_LRU_B",
        "tsbg_axes_separate": ["weight_fetch", "compute", "roofline_cycle"],
        "s2_targets": {"FC1": "REAL_RETAINED_VALUES",
            "PATCH": "BLOCKED_NO_RETAINED_BLOCK_VALUES",
            "FC2": "FIXED_NO_GO_M1713"},
        "fresh_result_namespace": str(RESULT.relative_to(ROOT)),
        "capture_present": os.path.lexists(str(CAPTURE)),
        "analysis_executed": False, "gpu_runs": 0, "eda_runs": 0,
        "claim_boundary": {"source_only": True, "cycles": False,
            "aee": False, "speedup": False, "energy": False,
            "rtl": False, "eda": False, "paper_result": False}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source-self-check", action="store_true")
    mode.add_argument("--run-analysis", action="store_true")
    args = parser.parse_args(argv)
    result = source_self_check() if args.source_self_check else run_analysis()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
