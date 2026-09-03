#!/opt/anaconda3/bin/python
"""M2064 CPU/source quick gate for exact G48 continuation tiling.

This script does not run RTL or EDA.  It extends the already VCS-calibrated
M2051 G48 service semantics to the eight ep34 FC2 layers whose source-group
count is greater than 48.  Every source-group interval is partitioned into
contiguous chunks of at most 48 groups.  The global group identity is retained
for weight addressing, Acc24 state is retained across chunks, and only the
last chunk retires the four-token bundle.

The ordinary and TSBG axes pay identical descriptor-preload, continuation and
final-retirement costs.  The source-cycle equation is calibrated against all
1,920 M2057 VCS workloads and remains explicitly a model, not an RTL result.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M2051_BUILDER = HW / (
    "system_simulator/scripts/build_m2051_ep34_tsbg_full40_fixture.py"
)
M2051_META = HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.json"
M2057 = HW / (
    "results/m2057_m2053_ep34_tsbg_full40_missing3_vcs_r1_20260903"
)
CAPTURE = HW / (
    "results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_"
    "binary_capture_s40_r1_20260901"
)
CONTRACT = HW / (
    "contracts/m2064_ep34_fc2_exact_continuation_quick_gate_contract_"
    "r1_20260903.json"
)
OUT = HW / (
    "results/m2064_ep34_fc2_exact_continuation_quick_gate_r1_20260903"
)

M2051_BUILDER_SHA = (
    "3a8642914ccad60df89dfdad1b78c375c6d4e4609435c5731357f294d9acf8cf"
)
M2057_MANIFEST_SHA = (
    "f00ab87e69043ed1eaa15980728c3858001122e47e5ff621dcf238eb5aeba971"
)
M2057_OUTER_SHA = (
    "3bd2d119e72792f75c636ca82856305151f08d02d15418b675139f504fb51df2"
)
M2057_RESULT_SHA = (
    "b4ee4f9cf4d55a4f722f1487ba4bc23948bc3f6a096178fa835d9ed18b50fe2a"
)
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOC359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

CONTEXTS = 4
MAX_GROUPS = 48
SOURCES = 16
SLICES = 6
LANES = 16
CACHE_ROWS = 4
ACC24_MAX = (1 << 23) - 1
ACC24_MIN = -(1 << 23)
TARGET_LAYER_IDS = (17, 19, 21, 23, 25, 27, 29, 31)
TARGET_SAMPLES = tuple(range(40))
TOKEN_ROLES = ("first", "middle", "last")

# The M2051 load task accepts one G48 descriptor every two core cycles.  Each
# output tile therefore pays the same 4-context x 48-group preload on both
# axes.  Intermediate chunks replace the 24 commits plus final retirement with
# one explicit continuation transition while retaining Acc24 state.
PRELOAD_CYCLES_PER_CHUNK = 2 * CONTEXTS * MAX_GROUPS
FINAL_RETIRE_CYCLES = 27
CONTINUATION_CYCLES = 2

# Frozen quick-gate thresholds.  These are admission thresholds for a later
# continuation RTL experiment, never headline claims by themselves.
MIN_NEW_FC2_RATIO_OF_SUMS = 1.20
MAX_CALIBRATION_ABS_RESIDUAL = 5

PASS_RE = re.compile(
    r"PASS_M2051_EP34_TSBG_FULL40_CYCLE .*?"
    r"workload_slot=(?P<slot>\d+) .*?rows=(?P<rows>\d+) "
    r"issues=(?P<issues>\d+) products=(?P<products>\d+) "
    r"commits=(?P<commits>\d+) base_cycles=(?P<base_cycles>\d+) "
    r"tsbg_cycles=(?P<tsbg_cycles>\d+) bundles_base=(?P<bundles_base>\d+) "
    r"bundles_tsbg=(?P<bundles_tsbg>\d+)"
)


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_python(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_m2057_and_calibrate() -> dict:
    manifest = M2057 / "SHA256SUMS"
    outer = M2057 / "SHA256SUMS.seal.sha256"
    result = M2057 / "result.json"
    need(sha256(manifest) == M2057_MANIFEST_SHA, "M2057 manifest SHA drift")
    need(sha256(outer) == M2057_OUTER_SHA, "M2057 outer SHA drift")
    need(sha256(result) == M2057_RESULT_SHA, "M2057 result SHA drift")
    need(outer.read_text(encoding="ascii").split() ==
         [M2057_MANIFEST_SHA, "SHA256SUMS"], "M2057 outer contents drift")
    rows = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, name = line.split(None, 1)
        name = name.strip().lstrip("*")
        path = M2057 / name
        need(path.is_file() and sha256(path) == digest,
             f"M2057 member drift: {name}")
        if not name.startswith("sim_slot") or not name.endswith(".log"):
            continue
        match = PASS_RE.search(path.read_text(encoding="utf-8", errors="replace"))
        need(match is not None, f"M2057 PASS token absent: {name}")
        row = {key: int(value) for key, value in match.groupdict().items()}
        need(row["slot"] not in rows, "duplicate M2057 workload slot")
        rows[row["slot"]] = row
    need(sorted(rows) == list(range(1920)), "M2057 calibration population drift")

    residuals = []
    for row in rows.values():
        for mode in ("base", "tsbg"):
            predicted = source_execute_cycles(
                row["issues"], row[f"bundles_{mode}"]
            )
            residuals.append(predicted - row[f"{mode}_cycles"])
    need(min(residuals) >= -MAX_CALIBRATION_ABS_RESIDUAL and
         max(residuals) <= MAX_CALIBRATION_ABS_RESIDUAL,
         "M2057 source-cycle calibration residual escaped frozen envelope")
    return {
        "workloads": len(rows),
        "observations": len(residuals),
        "equation": "27 + (7/6)*issues + (21/2)*weight_bundle_beats",
        "residual_definition": "source_model_minus_VCS_execute_cycles",
        "residual_min_cycles": min(residuals),
        "residual_max_cycles": max(residuals),
        "residual_abs_max_cycles": max(abs(value) for value in residuals),
        "residual_abs_mean_cycles": sum(abs(value) for value in residuals) /
        len(residuals),
        "residual_histogram": dict(sorted(Counter(residuals).items())),
        "vcs_result_weighted_speedup": json.loads(
            result.read_text(encoding="utf-8")
        )["aggregate"]["weighted_cycle_speedup"],
    }


def source_execute_cycles(issues: int, bundles: int) -> int:
    """VCS-calibrated G48 source model including one final retirement."""
    need(issues % SLICES == 0, "issue count is not six-slice integral")
    need(bundles % (2 * SLICES) == 0,
         "weight bundles are not full two-half/six-slice fills")
    return FINAL_RETIRE_CYCLES + (7 * issues) // 6 + (21 * bundles) // 2


def global_cache_counts(accesses: list[int], builder) -> tuple[int, int, int]:
    # Equality and LRU order, rather than the numeric group label, determine
    # cache behavior.  Still pass global IDs so accidental chunk aliasing is
    # visible in the evidence and weight-address audit.
    return builder.cache_counts(accesses)


def describe_chunk(codes: np.ndarray, group_base: int, builder) -> dict:
    need(codes.shape == (CONTEXTS, MAX_GROUPS, SOURCES),
         "continuation chunk is not physical G48")
    active = codes != 0
    live = active.any(axis=2)
    ordinary_accesses = [group_base + group
                         for context in range(CONTEXTS)
                         for group in range(MAX_GROUPS)
                         if live[context, group]]
    tsbg_accesses = [group_base + group
                     for group in range(MAX_GROUPS)
                     for context in range(CONTEXTS)
                     if live[context, group]]
    ordinary = global_cache_counts(ordinary_accesses, builder)
    tsbg = global_cache_counts(tsbg_accesses, builder)
    half_live = active.reshape(CONTEXTS, MAX_GROUPS, 2, 8).any(axis=3)
    issues = int(half_live.sum()) * SLICES
    return {
        "group_base": group_base,
        "group_limit": group_base + MAX_GROUPS,
        "live_rows": int(live.sum()),
        "nonzero_codes": int(active.sum()),
        "negative_codes": int((codes < 0).sum()),
        "issues": issues,
        "products": int(active.sum()) * SLICES * LANES,
        "ordinary_misses": ordinary[0],
        "ordinary_hits": ordinary[1],
        "ordinary_evictions": ordinary[2],
        "tsbg_misses": tsbg[0],
        "tsbg_hits": tsbg[1],
        "tsbg_evictions": tsbg[2],
        "ordinary_bundles": ordinary[0] * 2 * SLICES,
        "tsbg_bundles": tsbg[0] * 2 * SLICES,
    }


def directed_weights(groups: int, output_tiles: int) -> np.ndarray:
    """Deterministic signed INT8 weights with global group/tile identity."""
    weights = np.empty(
        (groups, SOURCES, output_tiles, SLICES, LANES), dtype=np.int16
    )
    for group in range(groups):
        for source in range(SOURCES):
            half = source // 8
            bank = source % 8
            for output_tile in range(output_tiles):
                for output_slice in range(SLICES):
                    for lane in range(LANES):
                        raw = (group * 17 + half * 11 +
                               (output_tile * SLICES + output_slice) * 7 +
                               bank * 5 + lane * 3) % 255 - 127
                        if (group == 0 and half == 0 and output_tile == 0 and
                                output_slice == 0 and bank == 0 and lane == 0):
                            raw = -128
                        weights[group, source, output_tile,
                                output_slice, lane] = raw
    need(int(weights.min()) == -128 and int(weights.max()) <= 127,
         "directed INT8 weight construction drift")
    return weights


def arithmetic_check(codes: np.ndarray, weights: np.ndarray) -> dict:
    groups = codes.shape[1]
    direct = np.einsum("cgs,gstul->ctul", codes.astype(np.int64),
                       weights.astype(np.int64), optimize=True)
    continuation = np.zeros_like(direct)
    max_intermediate_abs = 0
    overflow_count = 0
    for group_base in range(0, groups, MAX_GROUPS):
        group_limit = min(group_base + MAX_GROUPS, groups)
        continuation += np.einsum(
            "cgs,gstul->ctul",
            codes[:, group_base:group_limit, :].astype(np.int64),
            weights[group_base:group_limit].astype(np.int64),
            optimize=True,
        )
        chunk_abs = np.abs(continuation)
        max_intermediate_abs = max(max_intermediate_abs, int(chunk_abs.max()))
        overflow_count += int(np.count_nonzero(
            (continuation < ACC24_MIN) | (continuation > ACC24_MAX)
        ))
    mismatch = int(np.count_nonzero(direct != continuation))
    return {
        "integer_checks": int(direct.size),
        "integer_mismatches": mismatch,
        "acc24_overflow_observations": overflow_count,
        "max_intermediate_abs_accumulator": max_intermediate_abs,
        "max_final_abs_accumulator": int(np.abs(direct).max()),
    }


def logical_workload_cycles(chunk_rows: list[dict], output_tiles: int) -> dict:
    chunks = len(chunk_rows)
    need(chunks in (2, 4), "unexpected continuation chunk count")
    common = output_tiles * (
        chunks * PRELOAD_CYCLES_PER_CHUNK +
        (chunks - 1) * CONTINUATION_CYCLES + FINAL_RETIRE_CYCLES
    )
    # source_execute_cycles has already included a final retirement per chunk;
    # retain only its data-dependent calibrated service term here.
    ordinary_data = sum(
        source_execute_cycles(row["issues"], row["ordinary_bundles"])
        - FINAL_RETIRE_CYCLES for row in chunk_rows
    ) * output_tiles
    tsbg_data = sum(
        source_execute_cycles(row["issues"], row["tsbg_bundles"])
        - FINAL_RETIRE_CYCLES for row in chunk_rows
    ) * output_tiles
    ordinary = common + ordinary_data
    tsbg = common + tsbg_data
    calibration_margin = MAX_CALIBRATION_ABS_RESIDUAL * chunks * output_tiles
    return {
        "output_tiles": output_tiles,
        "chunks": chunks,
        "common_cycles": common,
        "descriptor_preload_cycles": output_tiles * chunks *
        PRELOAD_CYCLES_PER_CHUNK,
        "continuation_cycles": output_tiles * (chunks - 1) *
        CONTINUATION_CYCLES,
        "final_retire_cycles": output_tiles * FINAL_RETIRE_CYCLES,
        "ordinary_cycles_nominal": ordinary,
        "tsbg_cycles_nominal": tsbg,
        "ordinary_cycles_pessimistic_lower": max(1, ordinary - calibration_margin),
        "tsbg_cycles_pessimistic_upper": tsbg + calibration_margin,
    }


def aggregate(rows: list[dict]) -> dict:
    ordinary = sum(row["ordinary_cycles_nominal"] for row in rows)
    tsbg = sum(row["tsbg_cycles_nominal"] for row in rows)
    ordinary_lower = sum(
        row["ordinary_cycles_pessimistic_lower"] for row in rows
    )
    tsbg_upper = sum(row["tsbg_cycles_pessimistic_upper"] for row in rows)
    return {
        "workloads": len(rows),
        "ordinary_cycles_nominal": ordinary,
        "tsbg_cycles_nominal": tsbg,
        "ratio_of_sums_nominal": ordinary / tsbg,
        "time_reduction_fraction_nominal": 1.0 - tsbg / ordinary,
        "ordinary_cycles_pessimistic_lower": ordinary_lower,
        "tsbg_cycles_pessimistic_upper": tsbg_upper,
        "ratio_of_sums_pessimistic": ordinary_lower / tsbg_upper,
        "aggregate_non_regression": tsbg <= ordinary,
        "workload_regressions_nominal": sum(
            row["tsbg_cycles_nominal"] > row["ordinary_cycles_nominal"]
            for row in rows
        ),
    }


def supported_full_fc_model(builder, layer_by_id: dict[int, dict]) -> dict:
    payload = json.loads(M2051_META.read_text(encoding="utf-8"))
    rows = []
    for source in payload["rows"]:
        layer = layer_by_id[int(source["layer_id"])]
        output_tiles = int(layer["weight_layout"]["output_tile_count"])
        ordinary_bundles = int(source["base_misses"]) * 2 * SLICES
        tsbg_bundles = int(source["tsbg_misses"]) * 2 * SLICES
        common = output_tiles * (
            PRELOAD_CYCLES_PER_CHUNK + FINAL_RETIRE_CYCLES
        )
        ordinary = common + output_tiles * (
            source_execute_cycles(int(source["issues"]), ordinary_bundles)
            - FINAL_RETIRE_CYCLES
        )
        tsbg = common + output_tiles * (
            source_execute_cycles(int(source["issues"]), tsbg_bundles)
            - FINAL_RETIRE_CYCLES
        )
        margin = MAX_CALIBRATION_ABS_RESIDUAL * output_tiles
        rows.append({
            "ordinary_cycles_nominal": ordinary,
            "tsbg_cycles_nominal": tsbg,
            "ordinary_cycles_pessimistic_lower": max(1, ordinary - margin),
            "tsbg_cycles_pessimistic_upper": tsbg + margin,
        })
    need(len(rows) == 1920, "supported full-FC model population drift")
    return aggregate(rows)


def breakdown(rows: list[dict], field: str) -> dict:
    buckets = defaultdict(list)
    for row in rows:
        buckets[str(row[field])].append(row)
    return {key: aggregate(value) for key, value in sorted(buckets.items())}


def seal_output() -> None:
    members = [path for path in OUT.iterdir()
               if path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")]
    manifest = OUT / "SHA256SUMS"
    manifest.write_text("".join(
        f"{sha256(path)}  {path.name}\n" for path in sorted(members)
    ), encoding="ascii")
    (OUT / "SHA256SUMS.seal.sha256").write_text(
        f"{sha256(manifest)}  SHA256SUMS\n", encoding="ascii"
    )


def main() -> int:
    need(sha256(DOC359) == DOC359_SHA, "docs/359 identity drift before run")
    need(sha256(M2051_BUILDER) == M2051_BUILDER_SHA,
         "M2051 builder source SHA drift")
    need(CONTRACT.is_file(), "M2064 source contract absent")
    calibration = verify_m2057_and_calibrate()
    builder = load_python(M2051_BUILDER, "m2064_frozen_m2051_builder")
    builder.verify_capture()

    layers_payload = json.loads((CAPTURE / "layers.json").read_text())
    sample_payload = json.loads((CAPTURE / "sample_order.json").read_text())
    all_layers = {int(row["layer_id"]): row for row in layers_payload["layers"]}
    layers = [all_layers[layer_id] for layer_id in TARGET_LAYER_IDS]
    need(all(row["target"] == "FC2" for row in layers),
         "target inventory includes a non-FC2 layer")
    need([int(row["weight_layout"]["source_group_count"]) for row in layers] ==
         [96, 96, 96, 96, 96, 96, 192, 192],
         "target G96/G192 geometry drift")
    samples_by_id = {int(row["global_sample_id"]): row
                     for row in sample_payload["samples"]}
    samples = [samples_by_id[index] for index in TARGET_SAMPLES]
    need(len(samples) == 40 and
         sorted(Counter(row["sequence"] for row in samples).values()) ==
         [10, 10, 10, 10],
         "four-sequence sample cohort drift")

    capture_module = builder.load_module()
    found = builder.extract(capture_module, layers)
    weights_by_layer = {
        int(layer["layer_id"]): directed_weights(
            int(layer["weight_layout"]["source_group_count"]),
            int(layer["weight_layout"]["output_tile_count"]),
        ) for layer in layers
    }

    rows = []
    total_integer_checks = total_mismatches = total_overflows = 0
    max_intermediate_abs = max_final_abs = 0
    for sample in samples:
        sample_id = int(sample["global_sample_id"])
        for layer in layers:
            layer_id = int(layer["layer_id"])
            groups = int(layer["weight_layout"]["source_group_count"])
            output_tiles = int(layer["weight_layout"]["output_tile_count"])
            for role, start in zip(
                    TOKEN_ROLES,
                    builder.token_starts(int(layer["tokens_per_call"]))):
                codes = found[(sample_id, layer_id, start)]
                need(codes.shape == (CONTEXTS, groups * SOURCES),
                     "logical quartet geometry drift")
                codes = codes.reshape(CONTEXTS, groups, SOURCES)
                need(set(int(value) for value in np.unique(codes)).issubset(
                    {-1, 0, 1}), "source code escaped ternary domain")
                chunks = []
                for group_base in range(0, groups, MAX_GROUPS):
                    physical = np.zeros(
                        (CONTEXTS, MAX_GROUPS, SOURCES), dtype=np.int8
                    )
                    logical = codes[:, group_base:group_base + MAX_GROUPS, :]
                    physical[:, :logical.shape[1], :] = logical
                    chunks.append(describe_chunk(physical, group_base, builder))
                arithmetic = arithmetic_check(codes, weights_by_layer[layer_id])
                cycles = logical_workload_cycles(chunks, output_tiles)
                row = {
                    "slot": len(rows),
                    "sample_id": sample_id,
                    "sequence": sample["sequence"],
                    "layer_id": layer_id,
                    "token_role": role,
                    "token_start": start,
                    "source_groups": groups,
                    "output_tiles": output_tiles,
                    "chunks": chunks,
                    **cycles,
                    **arithmetic,
                }
                rows.append(row)
                total_integer_checks += arithmetic["integer_checks"]
                total_mismatches += arithmetic["integer_mismatches"]
                total_overflows += arithmetic["acc24_overflow_observations"]
                max_intermediate_abs = max(
                    max_intermediate_abs,
                    arithmetic["max_intermediate_abs_accumulator"],
                )
                max_final_abs = max(
                    max_final_abs, arithmetic["max_final_abs_accumulator"]
                )
    need(len(rows) == 960, "G>48 FC2 workload population incomplete")

    new_fc2 = aggregate(rows)
    supported = supported_full_fc_model(builder, all_layers)
    combined_rows = rows + [{
        "ordinary_cycles_nominal": supported["ordinary_cycles_nominal"],
        "tsbg_cycles_nominal": supported["tsbg_cycles_nominal"],
        "ordinary_cycles_pessimistic_lower":
            supported["ordinary_cycles_pessimistic_lower"],
        "tsbg_cycles_pessimistic_upper":
            supported["tsbg_cycles_pessimistic_upper"],
    }]
    combined = aggregate(combined_rows)
    # The synthetic aggregate row above represents 1,920 supported workloads.
    combined["workloads"] = 2880
    combined["workload_regressions_nominal"] = (
        new_fc2["workload_regressions_nominal"] +
        supported["workload_regressions_nominal"]
    )

    gates = {
        "workloads_exactly_960": len(rows) == 960,
        "integer_mismatches_zero": total_mismatches == 0,
        "acc24_overflow_zero": total_overflows == 0,
        "new_fc2_ratio_of_sums_at_least_1p20":
            new_fc2["ratio_of_sums_nominal"] >= MIN_NEW_FC2_RATIO_OF_SUMS,
        "new_fc2_pessimistic_ratio_at_least_1p20":
            new_fc2["ratio_of_sums_pessimistic"] >=
            MIN_NEW_FC2_RATIO_OF_SUMS,
        "new_fc2_aggregate_non_regression":
            new_fc2["aggregate_non_regression"],
        "combined_full_fc_aggregate_non_regression":
            combined["aggregate_non_regression"],
        "docs359_unchanged": sha256(DOC359) == DOC359_SHA,
    }
    decision = "GO_TO_INDEPENDENT_HAMMER_ONLY" if all(gates.values()) else "NO_GO_RTL"

    result = {
        "schema": "m2064_ep34_fc2_exact_continuation_quick_gate_r1_v1",
        "status": "PASS" if all(gates.values()) else "FAIL_CLOSED",
        "decision": decision,
        "evidence_level": "CPU/source model calibrated to M2057 VCS; not RTL cycle",
        "input_identity": {
            "contract_sha256": sha256(CONTRACT),
            "m2051_builder_sha256": M2051_BUILDER_SHA,
            "m2057_manifest_sha256": M2057_MANIFEST_SHA,
            "m2057_outer_sha256": M2057_OUTER_SHA,
            "m2057_result_sha256": M2057_RESULT_SHA,
            "capture_manifest_sha256": builder.CAPTURE_MANIFEST_SHA,
            "fc_frames_sha256": builder.FC_FRAMES_SHA,
            "layers_sha256": builder.LAYERS_SHA,
            "sample_order_sha256": builder.SAMPLE_ORDER_SHA,
            "checkpoint_sha256":
                "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48",
            "docs359_sha256": DOC359_SHA,
        },
        "mechanism": {
            "physical_engine_source_groups": MAX_GROUPS,
            "partition": "contiguous global source groups in chunks <=48",
            "global_group_base_preserved_for_weight_address": True,
            "acc24_retained_per_token_across_chunks": True,
            "intermediate_chunk_commit": False,
            "only_final_chunk_terminal_and_commit": True,
            "ordinary_and_tsbg_share_all_fixed_costs": True,
            "preload_cycles_per_chunk_per_output_tile":
                PRELOAD_CYCLES_PER_CHUNK,
            "continuation_cycles_per_intermediate_chunk_per_output_tile":
                CONTINUATION_CYCLES,
            "final_retire_cycles_per_output_tile": FINAL_RETIRE_CYCLES,
        },
        "geometry": {
            "sequences": 4,
            "samples": 40,
            "target_fc2_layers": 8,
            "g96_layers": 6,
            "g192_layers": 2,
            "quartets_per_layer_sample": 3,
            "new_workloads": len(rows),
            "supported_workloads_reaccounted": 1920,
            "combined_full_fc_workloads": 2880,
        },
        "calibration": calibration,
        "arithmetic": {
            "integer_checks": total_integer_checks,
            "integer_mismatches": total_mismatches,
            "acc24_overflow_observations": total_overflows,
            "max_intermediate_abs_accumulator": max_intermediate_abs,
            "max_final_abs_accumulator": max_final_abs,
            "acc24_range": [ACC24_MIN, ACC24_MAX],
            "hardware_weight_values": False,
            "directed_weight_identity_includes_global_group_and_output_tile": True,
        },
        "aggregate_new_greater_than_g48_fc2": new_fc2,
        "aggregate_existing_g_le_48_fc_model_same_fees": supported,
        "aggregate_combined_all_24_fc_layers": combined,
        "breakdown": {
            "layer_id": breakdown(rows, "layer_id"),
            "sequence": breakdown(rows, "sequence"),
            "source_groups": breakdown(rows, "source_groups"),
        },
        "gates": gates,
        "claim_boundary": {
            "real_ep34_activity_and_sign_descriptors": True,
            "all_fc2_layers_covered_after_continuation": True,
            "exact_directed_integer_continuation": True,
            "vcs_calibrated_source_cycle_model": True,
            "new_continuation_rtl": False,
            "new_vcs": False,
            "new_eda": False,
            "hardware_weight_values": False,
            "full_network_speedup": False,
            "system_speedup": False,
            "paper_admission": False,
            "independent_hammer_required_before_upgrade": True,
        },
        "rows": rows,
    }

    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "status": result["status"],
        "decision": decision,
        "workloads": len(rows),
        "integer_checks": total_integer_checks,
        "integer_mismatches": total_mismatches,
        "acc24_overflow_observations": total_overflows,
        "new_fc2_ratio_of_sums_nominal":
            new_fc2["ratio_of_sums_nominal"],
        "new_fc2_ratio_of_sums_pessimistic":
            new_fc2["ratio_of_sums_pessimistic"],
        "combined_full_fc_ratio_of_sums_nominal":
            combined["ratio_of_sums_nominal"],
        "docs359_sha256": sha256(DOC359),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    seal_output()
    print(json.dumps(summary, sort_keys=True))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
