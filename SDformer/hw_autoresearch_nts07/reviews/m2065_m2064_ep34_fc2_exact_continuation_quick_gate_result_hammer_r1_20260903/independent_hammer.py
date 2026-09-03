#!/usr/bin/env python3
"""Read-only independent hammer for the M2064 CPU/source quick gate.

The hammer deliberately does not import the M2064 analyzer.  It reuses only
the frozen M2051 capture decoder, independently reconstructs the G48 chunking,
LRU/cache ledger, cycle equations, directed integer arithmetic, and an alias
mutation.  It writes nothing and launches no simulator, EDA tool, or GPU job.
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
CONTRACT = HW / "contracts/m2064_ep34_fc2_exact_continuation_quick_gate_contract_r1_20260903.json"
SOURCE = HW / "system_simulator/scripts/analyze_m2064_ep34_fc2_exact_continuation_quick_gate.py"
RESULT_DIR = HW / "results/m2064_ep34_fc2_exact_continuation_quick_gate_r1_20260903"
CAPTURE = HW / "results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901"
BUILDER = HW / "system_simulator/scripts/build_m2051_ep34_tsbg_full40_fixture.py"
META = HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.json"
M2057 = HW / "results/m2057_m2053_ep34_tsbg_full40_missing3_vcs_r1_20260903"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED_CONTRACT_SHA = "34fc9f878004e593285dbfdafa22e34e288ea03007819451c9d8a78d92e6d9fc"
EXPECTED_SOURCE_SHA = "58c0589178b23ab31826a0dd9e329bab977333829cefb8518553629a18af4161"
EXPECTED_RESULT_SHA = "8f6e39b52a08042965b12a4e646ac2959567e032952426296a01194a8c8295d9"
EXPECTED_SUMMARY_SHA = "ebc508f3ac4407b30bc1fd3790a90c14ab18b6ca51ae44dcb3eb2876f535da03"
EXPECTED_CAPTURE_MANIFEST_SHA = "be0b89f9b8084baf0c2cd959805530a4e4f41c437a446335142a65bd73960a8f"
EXPECTED_CAPTURE_OUTER_SHA = "8d63d1054452377836c333c9848f771a6fc964e4ed4b00ed9a22f1537bd73c85"
EXPECTED_BUILDER_SHA = "3a8642914ccad60df89dfdad1b78c375c6d4e4609435c5731357f294d9acf8cf"
EXPECTED_META_SHA = "3ac7048f0a97aeea0ac91627d303f4eea06b8a48bab816468825acfee180ccc5"
EXPECTED_M2057_MANIFEST_SHA = "f00ab87e69043ed1eaa15980728c3858001122e47e5ff621dcf238eb5aeba971"
EXPECTED_M2057_OUTER_SHA = "3bd2d119e72792f75c636ca82856305151f08d02d15418b675139f504fb51df2"
EXPECTED_M2057_RESULT_SHA = "b4ee4f9cf4d55a4f722f1487ba4bc23948bc3f6a096178fa835d9ed18b50fe2a"
EXPECTED_DOC359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

CONTEXTS = 4
GROUPS_PER_CHUNK = 48
SOURCES = 16
SLICES = 6
LANES = 16
CACHE_ROWS = 4
PRELOAD = 384
CONTINUATION = 2
RETIRE = 27
TARGET_LAYERS = (17, 19, 21, 23, 25, 27, 29, 31)
ROLES = ("first", "middle", "last")
PASS_RE = re.compile(
    r"PASS_M2051_EP34_TSBG_FULL40_CYCLE .*?"
    r"workload_slot=(?P<slot>\d+) .*?rows=(?P<rows>\d+) "
    r"issues=(?P<issues>\d+) products=(?P<products>\d+) "
    r"commits=(?P<commits>\d+) base_cycles=(?P<base_cycles>\d+) "
    r"tsbg_cycles=(?P<tsbg_cycles>\d+) bundles_base=(?P<bundles_base>\d+) "
    r"bundles_tsbg=(?P<bundles_tsbg>\d+)"
)


def need(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(path: Path):
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, f"duplicate JSON key in {path}: {key}")
            out[key] = value
        return out

    def bad(value):
        raise AssertionError(f"non-finite JSON value in {path}: {value}")

    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=bad)


def verify_manifest(directory: Path, expected_manifest: str | None = None,
                    expected_outer: str | None = None, full: bool = True) -> int:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    if expected_manifest:
        need(sha(manifest) == expected_manifest, f"manifest drift: {directory}")
    if expected_outer:
        need(sha(outer) == expected_outer, f"outer seal drift: {directory}")
    words = outer.read_text(encoding="ascii").split()
    need(words == [sha(manifest), "SHA256SUMS"], f"outer contents drift: {directory}")
    count = 0
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, name = line.split(None, 1)
        member = directory / name.strip().lstrip("*")
        need(member.is_file() and not member.is_symlink(), f"bad member: {member}")
        if full:
            need(sha(member) == digest, f"member drift: {member}")
        count += 1
    return count


def import_builder():
    need(sha(BUILDER) == EXPECTED_BUILDER_SHA, "builder drift")
    spec = importlib.util.spec_from_file_location("m2065_frozen_builder", BUILDER)
    need(spec is not None and spec.loader is not None, "cannot load builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    need(sha(BUILDER) == EXPECTED_BUILDER_SHA, "builder changed during import")
    return module


def cache_counts(accesses: list[int]) -> tuple[int, int, int]:
    valid = [False] * CACHE_ROWS
    labels = [0] * CACHE_ROWS
    age = [0] * CACHE_ROWS
    clock = hits = misses = evictions = 0
    for group in accesses:
        hit = next((i for i in range(CACHE_ROWS)
                    if valid[i] and labels[i] == group), None)
        if hit is not None:
            hits += 1
            age[hit] = clock
        else:
            misses += 1
            invalid = next((i for i in range(CACHE_ROWS) if not valid[i]), None)
            if invalid is None:
                victim = min(range(CACHE_ROWS), key=lambda i: (age[i], i))
                evictions += 1
            else:
                victim = invalid
            labels[victim] = group
            valid[victim] = True
            age[victim] = clock + 1
        clock += 1
    return misses, hits, evictions


def service(issues: int, misses: int) -> int:
    bundles = misses * 2 * SLICES
    need(issues % SLICES == 0 and bundles % (2 * SLICES) == 0,
         "non-integral service")
    return (7 * issues) // 6 + (21 * bundles) // 2


def directed_weights(groups: int, tiles: int) -> np.ndarray:
    weights = np.empty((groups, SOURCES, tiles, SLICES, LANES), dtype=np.int16)
    for g in range(groups):
        for source in range(SOURCES):
            half = source // 8
            bank = source % 8
            for tile in range(tiles):
                for slc in range(SLICES):
                    for lane in range(LANES):
                        value = (g * 17 + half * 11 +
                                 (tile * SLICES + slc) * 7 +
                                 bank * 5 + lane * 3) % 255 - 127
                        if (g, half, tile, slc, bank, lane) == (0, 0, 0, 0, 0, 0):
                            value = -128
                        weights[g, source, tile, slc, lane] = value
    return weights


def aggregate(rows: list[dict]) -> dict:
    ordinary = sum(row["ordinary_cycles_nominal"] for row in rows)
    tsbg = sum(row["tsbg_cycles_nominal"] for row in rows)
    ordinary_lower = sum(row["ordinary_cycles_pessimistic_lower"] for row in rows)
    tsbg_upper = sum(row["tsbg_cycles_pessimistic_upper"] for row in rows)
    return {
        "workloads": len(rows),
        "ordinary_cycles_nominal": ordinary,
        "tsbg_cycles_nominal": tsbg,
        "ratio_of_sums_nominal": ordinary / tsbg,
        "ordinary_cycles_pessimistic_lower": ordinary_lower,
        "tsbg_cycles_pessimistic_upper": tsbg_upper,
        "ratio_of_sums_pessimistic": ordinary_lower / tsbg_upper,
        "workload_regressions_nominal": sum(
            row["tsbg_cycles_nominal"] > row["ordinary_cycles_nominal"]
            for row in rows),
    }


def compare_number(actual, expected, label: str) -> None:
    if isinstance(expected, float):
        need(math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-12),
             f"{label}: {actual} != {expected}")
    else:
        need(actual == expected, f"{label}: {actual} != {expected}")


def main() -> int:
    doc_before = sha(DOC359)
    need(doc_before == EXPECTED_DOC359_SHA, "docs359 drift")
    need(sha(CONTRACT) == EXPECTED_CONTRACT_SHA, "contract drift")
    need(sha(SOURCE) == EXPECTED_SOURCE_SHA, "source drift")
    need(sha(RESULT_DIR / "result.json") == EXPECTED_RESULT_SHA, "result drift")
    need(sha(RESULT_DIR / "summary.json") == EXPECTED_SUMMARY_SHA, "summary drift")

    # Contract sidecars and canonical result double seal.
    contract_digest = CONTRACT.with_name(CONTRACT.name + ".sha256")
    contract_outer = CONTRACT.with_name(CONTRACT.name + ".sha256.seal.sha256")
    need(contract_digest.read_text(encoding="ascii").split() ==
         [EXPECTED_CONTRACT_SHA, CONTRACT.name], "contract digest contents")
    need(contract_outer.read_text(encoding="ascii").split() ==
         [sha(contract_digest), contract_digest.name], "contract outer contents")
    result_members = verify_manifest(RESULT_DIR, full=True)
    need(result_members == 2, "M2064 result member count")
    need({p.name for p in RESULT_DIR.iterdir()} ==
         {"result.json", "summary.json", "SHA256SUMS", "SHA256SUMS.seal.sha256"},
         "M2064 result extra/missing member")

    contract = strict_json(CONTRACT)
    result = strict_json(RESULT_DIR / "result.json")
    summary = strict_json(RESULT_DIR / "summary.json")
    need(contract["source"]["sha256"] == EXPECTED_SOURCE_SHA, "source not pinned")
    need(result["input_identity"]["contract_sha256"] == EXPECTED_CONTRACT_SHA,
         "result contract binding")

    # Reopen M1707 input identity. Full manifest content is hashed, then the
    # three consumed payloads are checked against both builder and result pins.
    capture_members = verify_manifest(
        CAPTURE, EXPECTED_CAPTURE_MANIFEST_SHA, EXPECTED_CAPTURE_OUTER_SHA,
        full=True)
    builder = import_builder()
    for path, expected in (
        (CAPTURE / "fc_frames.bin", builder.FC_FRAMES_SHA),
        (CAPTURE / "layers.json", builder.LAYERS_SHA),
        (CAPTURE / "sample_order.json", builder.SAMPLE_ORDER_SHA),
    ):
        need(sha(path) == expected, f"capture payload drift: {path.name}")
    need(result["input_identity"]["capture_manifest_sha256"] ==
         EXPECTED_CAPTURE_MANIFEST_SHA, "result capture binding")

    # Reopen M2057 and independently recover the calibration residuals.
    m2057_members = verify_manifest(
        M2057, EXPECTED_M2057_MANIFEST_SHA, EXPECTED_M2057_OUTER_SHA,
        full=True)
    need(sha(M2057 / "result.json") == EXPECTED_M2057_RESULT_SHA,
         "M2057 result drift")
    m2057_json = strict_json(M2057 / "result.json")
    need(sha(META) == EXPECTED_META_SHA ==
         m2057_json["identity"]["fixture_json_sha256"],
         "M2051 metadata is not the M2057 fixture identity")
    residuals = []
    slots = set()
    for path in sorted(M2057.glob("sim_slot*.log")):
        match = PASS_RE.search(path.read_text(encoding="utf-8", errors="replace"))
        need(match is not None, f"missing M2057 PASS: {path.name}")
        row = {key: int(value) for key, value in match.groupdict().items()}
        need(row["slot"] not in slots, "duplicate M2057 slot")
        slots.add(row["slot"])
        for mode in ("base", "tsbg"):
            predicted = RETIRE + (7 * row["issues"]) // 6 + \
                (21 * row[f"bundles_{mode}"]) // 2
            residuals.append(predicted - row[f"{mode}_cycles"])
    need(slots == set(range(1920)), "M2057 population mismatch")
    need((min(residuals), max(residuals), max(map(abs, residuals))) == (-5, 1, 5),
         "M2057 calibration envelope drift")

    # Decode the target cohort through the frozen capture decoder, then use an
    # independent chunk/cache/arithmetic implementation.
    builder.verify_capture()
    decoder = builder.load_module()
    layer_payload = strict_json(CAPTURE / "layers.json")
    sample_payload = strict_json(CAPTURE / "sample_order.json")
    layer_by_id = {int(row["layer_id"]): row for row in layer_payload["layers"]}
    layers = [layer_by_id[i] for i in TARGET_LAYERS]
    need([int(row["weight_layout"]["source_group_count"]) for row in layers] ==
         [96, 96, 96, 96, 96, 96, 192, 192], "G96/G192 geometry")
    sample_by_id = {int(row["global_sample_id"]): row
                    for row in sample_payload["samples"]}
    samples = [sample_by_id[i] for i in range(40)]
    need(Counter(row["sequence"] for row in samples) == {
        "interlaken_01_a": 10, "thun_01_b": 10,
        "zurich_city_09_a": 10, "zurich_city_12_a": 10},
        "sample sequence distribution")
    found = builder.extract(decoder, layers)

    evidence_rows = result["rows"]
    need(len(evidence_rows) == 40 * 8 * 3 == 960, "960 workload identity")
    independent_rows = []
    integer_checks = mismatches = overflows = 0
    max_final = max_intermediate = 0
    alias_mismatches = alias_sensitive_workloads = 0
    group_base_sets = defaultdict(set)
    fee_mismatches = row_mismatches = 0

    for sample in samples:
        sample_id = int(sample["global_sample_id"])
        for layer in layers:
            layer_id = int(layer["layer_id"])
            groups = int(layer["weight_layout"]["source_group_count"])
            tiles = int(layer["weight_layout"]["output_tile_count"])
            weights = directed_weights(groups, tiles)
            starts = builder.token_starts(int(layer["tokens_per_call"]))
            for role, start in zip(ROLES, starts):
                slot = len(independent_rows)
                observed = evidence_rows[slot]
                codes = found[(sample_id, layer_id, start)].reshape(
                    CONTEXTS, groups, SOURCES).astype(np.int64)
                direct = np.einsum("cgs,gstul->ctul", codes,
                                   weights.astype(np.int64), optimize=True)
                continued = np.zeros_like(direct)
                aliased = np.zeros_like(direct)
                chunk_service_base = chunk_service_tsbg = 0
                chunks = groups // GROUPS_PER_CHUNK
                workload_max_intermediate = 0
                for base in range(0, groups, GROUPS_PER_CHUNK):
                    limit = base + GROUPS_PER_CHUNK
                    group_base_sets[groups].add(base)
                    part = codes[:, base:limit, :]
                    continued += np.einsum(
                        "cgs,gstul->ctul", part,
                        weights[base:limit].astype(np.int64), optimize=True)
                    # Mutation: incorrectly drop global_group_base from every
                    # chunk after the first. Real data must detect this alias.
                    aliased += np.einsum(
                        "cgs,gstul->ctul", part,
                        weights[:GROUPS_PER_CHUNK].astype(np.int64), optimize=True)
                    workload_max_intermediate = max(
                        workload_max_intermediate, int(np.abs(continued).max()))
                    max_intermediate = max(max_intermediate,
                                           workload_max_intermediate)
                    active = part != 0
                    live = active.any(axis=2)
                    base_accesses = [base + group
                                     for context in range(CONTEXTS)
                                     for group in range(GROUPS_PER_CHUNK)
                                     if live[context, group]]
                    tsbg_accesses = [base + group
                                     for group in range(GROUPS_PER_CHUNK)
                                     for context in range(CONTEXTS)
                                     if live[context, group]]
                    half_live = active.reshape(
                        CONTEXTS, GROUPS_PER_CHUNK, 2, 8).any(axis=3)
                    issues = int(half_live.sum()) * SLICES
                    chunk_service_base += service(issues, cache_counts(base_accesses)[0])
                    chunk_service_tsbg += service(issues, cache_counts(tsbg_accesses)[0])

                mismatch = int(np.count_nonzero(direct != continued))
                alias_bad = int(np.count_nonzero(direct != aliased))
                checks = int(direct.size)
                bound = groups * SOURCES * 128
                need(bound < (1 << 23), "formal Acc24 bound failed")
                overflow = int(np.count_nonzero(
                    (continued < -(1 << 23)) | (continued > (1 << 23) - 1)))
                integer_checks += checks
                mismatches += mismatch
                overflows += overflow
                max_final = max(max_final, int(np.abs(direct).max()))
                alias_mismatches += alias_bad
                alias_sensitive_workloads += alias_bad > 0

                preload = tiles * chunks * PRELOAD
                cont = tiles * (chunks - 1) * CONTINUATION
                retire = tiles * RETIRE
                common = preload + cont + retire
                ordinary = common + tiles * chunk_service_base
                tsbg = common + tiles * chunk_service_tsbg
                margin = 5 * chunks * tiles
                row = {
                    "ordinary_cycles_nominal": ordinary,
                    "tsbg_cycles_nominal": tsbg,
                    "ordinary_cycles_pessimistic_lower": ordinary - margin,
                    "tsbg_cycles_pessimistic_upper": tsbg + margin,
                    "sequence": sample["sequence"],
                    "layer_id": layer_id,
                    "source_groups": groups,
                }
                independent_rows.append(row)
                expected = {
                    "slot": slot, "sample_id": sample_id,
                    "sequence": sample["sequence"], "layer_id": layer_id,
                    "token_role": role, "token_start": start,
                    "source_groups": groups, "output_tiles": tiles,
                    "chunks": chunks, "common_cycles": common,
                    "descriptor_preload_cycles": preload,
                    "continuation_cycles": cont,
                    "final_retire_cycles": retire,
                    "ordinary_cycles_nominal": ordinary,
                    "tsbg_cycles_nominal": tsbg,
                    "ordinary_cycles_pessimistic_lower": ordinary - margin,
                    "tsbg_cycles_pessimistic_upper": tsbg + margin,
                    "integer_checks": checks,
                    "integer_mismatches": mismatch,
                    "acc24_overflow_observations": overflow,
                    "max_intermediate_abs_accumulator": workload_max_intermediate,
                    "max_final_abs_accumulator": int(np.abs(direct).max()),
                }
                for key, value in expected.items():
                    if observed.get(key) != value:
                        row_mismatches += 1
                fee_mismatches += not (
                    observed["descriptor_preload_cycles"] == tiles * chunks * 384 and
                    observed["continuation_cycles"] == tiles * (chunks - 1) * 2 and
                    observed["final_retire_cycles"] == tiles * 27 and
                    observed["common_cycles"] ==
                    observed["descriptor_preload_cycles"] +
                    observed["continuation_cycles"] +
                    observed["final_retire_cycles"])

    need(group_base_sets[96] == {0, 48}, "G96 global bases")
    need(group_base_sets[192] == {0, 48, 96, 144}, "G192 global bases")
    need(alias_sensitive_workloads > 0 and alias_mismatches > 0,
         "global-group alias mutation was not detected")
    need(row_mismatches == fee_mismatches == 0, "row or fee mismatch")
    need(integer_checks == 1843200 and mismatches == 0 and overflows == 0,
         "integer audit mismatch")
    need(all(row["tsbg_cycles_nominal"] <= row["ordinary_cycles_nominal"]
             for row in independent_rows), "per-workload regression")

    new_aggregate = aggregate(independent_rows)
    for key, value in new_aggregate.items():
        compare_number(value, result["aggregate_new_greater_than_g48_fc2"][key],
                       f"new aggregate {key}")

    # Reaccount the 1,920 already supported workloads with identical full-FC
    # fees, independently of the new rows.
    meta = strict_json(META)
    supported_rows = []
    for source in meta["rows"]:
        layer = layer_by_id[int(source["layer_id"])]
        tiles = int(layer["weight_layout"]["output_tile_count"])
        common = tiles * (PRELOAD + RETIRE)
        issues = int(source["issues"])
        ordinary = common + tiles * service(issues, int(source["base_misses"]))
        tsbg = common + tiles * service(issues, int(source["tsbg_misses"]))
        margin = 5 * tiles
        supported_rows.append({
            "ordinary_cycles_nominal": ordinary,
            "tsbg_cycles_nominal": tsbg,
            "ordinary_cycles_pessimistic_lower": ordinary - margin,
            "tsbg_cycles_pessimistic_upper": tsbg + margin,
        })
    need(len(supported_rows) == 1920, "supported population")
    supported = aggregate(supported_rows)
    for key, value in supported.items():
        compare_number(value,
                       result["aggregate_existing_g_le_48_fc_model_same_fees"][key],
                       f"supported aggregate {key}")

    combined = {
        "workloads": 2880,
        "ordinary_cycles_nominal":
            supported["ordinary_cycles_nominal"] + new_aggregate["ordinary_cycles_nominal"],
        "tsbg_cycles_nominal":
            supported["tsbg_cycles_nominal"] + new_aggregate["tsbg_cycles_nominal"],
        "ordinary_cycles_pessimistic_lower":
            supported["ordinary_cycles_pessimistic_lower"] +
            new_aggregate["ordinary_cycles_pessimistic_lower"],
        "tsbg_cycles_pessimistic_upper":
            supported["tsbg_cycles_pessimistic_upper"] +
            new_aggregate["tsbg_cycles_pessimistic_upper"],
        "workload_regressions_nominal":
            supported["workload_regressions_nominal"] +
            new_aggregate["workload_regressions_nominal"],
    }
    combined["ratio_of_sums_nominal"] = (
        combined["ordinary_cycles_nominal"] / combined["tsbg_cycles_nominal"])
    combined["ratio_of_sums_pessimistic"] = (
        combined["ordinary_cycles_pessimistic_lower"] /
        combined["tsbg_cycles_pessimistic_upper"])
    for key, value in combined.items():
        compare_number(value, result["aggregate_combined_all_24_fc_layers"][key],
                       f"combined aggregate {key}")

    # Check result summary, source-level claim boundary, and no paper upgrade.
    need(summary["workloads"] == 960 and summary["integer_checks"] == 1843200,
         "summary cardinality")
    need(math.isclose(summary["new_fc2_ratio_of_sums_nominal"],
                      1.8690072353457743, rel_tol=0, abs_tol=1e-12),
         "nominal ratio")
    need(math.isclose(summary["new_fc2_ratio_of_sums_pessimistic"],
                      1.8641944934268204, rel_tol=0, abs_tol=1e-12),
         "pessimistic ratio")
    need(math.isclose(summary["combined_full_fc_ratio_of_sums_nominal"],
                      2.230259079328016, rel_tol=0, abs_tol=1e-12),
         "combined ratio")
    need(result["decision"] == "GO_TO_INDEPENDENT_HAMMER_ONLY", "decision")
    need(result["claim_boundary"] == {
        "all_fc2_layers_covered_after_continuation": True,
        "exact_directed_integer_continuation": True,
        "full_network_speedup": False,
        "hardware_weight_values": False,
        "independent_hammer_required_before_upgrade": True,
        "new_continuation_rtl": False,
        "new_eda": False,
        "new_vcs": False,
        "paper_admission": False,
        "real_ep34_activity_and_sign_descriptors": True,
        "system_speedup": False,
        "vcs_calibrated_source_cycle_model": True,
    }, "claim boundary drift")
    need(sha(DOC359) == doc_before, "docs359 changed during review")

    output = {
        "status": "PASS_M2065_M2064_INDEPENDENT_CPU_SOURCE_RESULT_HAMMER",
        "score_over_100": 98,
        "severity_counts": {"P0": 0, "P1": 0, "P2": 2},
        "contract_double_seal": True,
        "result_double_seal": True,
        "capture_manifest_members_verified": capture_members,
        "m2057_manifest_members_verified": m2057_members,
        "m2051_metadata_sha256": sha(META),
        "calibration": {
            "observations": len(residuals),
            "residual_min": min(residuals),
            "residual_max": max(residuals),
            "residual_abs_max": max(map(abs, residuals)),
            "residual_histogram": dict(sorted(Counter(residuals).items())),
        },
        "geometry": {
            "workloads": len(independent_rows),
            "g96_layers": 6,
            "g192_layers": 2,
            "g96_bases": sorted(group_base_sets[96]),
            "g192_bases": sorted(group_base_sets[192]),
            "combined_workloads": 2880,
        },
        "global_group_alias_attack": {
            "mutation": "map every later chunk to weight groups 0..47",
            "sensitive_workloads": alias_sensitive_workloads,
            "mismatching_output_values": alias_mismatches,
            "rejected": True,
        },
        "arithmetic": {
            "integer_checks": integer_checks,
            "integer_mismatches": mismatches,
            "overflow_observations": overflows,
            "formal_max_abs_bound_g192": 192 * 16 * 128,
            "max_intermediate_abs": max_intermediate,
            "max_final_abs": max_final,
        },
        "fairness": {
            "row_field_mismatches": row_mismatches,
            "fee_mismatches": fee_mismatches,
            "preload_per_chunk_per_output_tile": 384,
            "continuation_per_intermediate_chunk_per_output_tile": 2,
            "final_unique_retire_per_output_tile": 27,
            "per_workload_regressions": new_aggregate["workload_regressions_nominal"],
        },
        "new_fc2": new_aggregate,
        "supported_fc": supported,
        "combined": combined,
        "authorization": {
            "next_vcs_source_only": True,
            "vcs_launch": False,
            "eda": False,
            "paper_admission": False,
            "system_speedup": False,
        },
        "p2": [
            "M2064 reads M2051 fixture metadata without directly pinning its SHA in its own contract/source; the current file exactly matches the M2057-pinned identity.",
            "M2064 row serialization overwrites per-chunk detail with the chunk count, so global_group_base is reconstructed by this hammer rather than visible in result rows."
        ],
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
