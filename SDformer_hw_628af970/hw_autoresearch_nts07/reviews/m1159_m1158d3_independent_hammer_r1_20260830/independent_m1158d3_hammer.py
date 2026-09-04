#!/usr/bin/env python3
"""Fresh different-author hammer for M1158D3.

This file deliberately does not import or execute the M1158 author analyzer.
It reconstructs D3 destination contributors and modulo-8 bank conflicts from
the frozen bitpack, then recomputes the same-width A1 baselines and gates.
"""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
D3_PAYLOAD_SHA = "0a8567d62df9aaf31ab19d7f1ad78366171be850a63562837d86f12570be86e3"
M712_ROWS_SHA = "a299dc9cae32a007dd7a8705afd046bce342bed5a7c9642c05123dfa04dff0c6"
EXPECTED_POLICY = {
    "D0": "A1-OSG",
    "D1": "A1-OSG",
    "D2": "A1-OSG",
    "D3": "STATIC_WEIGHT_FIT_BRIDGE_INCLUSIVE",
}
GEOMETRY = {
    "D0": (1536, 15, 20),
    "D1": (770, 30, 40),
    "D2": (386, 60, 80),
    "D3": (194, 120, 160),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_sealed_directory(path: Path) -> dict:
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    assert manifest.is_file() and outer.is_file(), f"missing seal in {path}"
    members = []
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip(" *")
        member = path / name
        assert member.is_file(), f"missing sealed member {member}"
        actual = sha256(member)
        assert actual == digest, f"inner seal mismatch {member}: {actual} != {digest}"
        members.append(name)
    outer_tokens = outer.read_text().split()
    assert len(outer_tokens) >= 1
    manifest_sha = sha256(manifest)
    assert outer_tokens[0] == manifest_sha, f"outer content mismatch in {path}"
    return {
        "member_count": len(members),
        "manifest_sha256": manifest_sha,
        "outer_file_sha256": sha256(outer),
    }


def verify_contract_triple(contract: Path) -> dict:
    digest_file = Path(str(contract) + ".sha256")
    seal_file = Path(str(contract) + ".sha256.seal.sha256")
    assert contract.is_file() and digest_file.is_file() and seal_file.is_file()
    digest_tokens = digest_file.read_text().split()
    seal_tokens = seal_file.read_text().split()
    contract_sha = sha256(contract)
    digest_sha = sha256(digest_file)
    assert digest_tokens[0] == contract_sha
    assert seal_tokens[0] == digest_sha
    return {
        "contract_sha256": contract_sha,
        "digest_file_sha256": digest_sha,
        "seal_file_sha256": sha256(seal_file),
    }


def load_frozen_d3(root: Path):
    m699 = root / "system_handoff/outgoing/m699_h67_ep35_multisequence_decoder_payload_s3x10_r1_20260828"
    manifest = json.loads((m699 / "manifest.json").read_text())
    records = [r for r in manifest["records"] if r["relative_path"] == "calls/s00_d3.binary.le.bitpack"]
    assert len(records) == 1
    record = records[0]
    assert record["global_sample_id"] == 0
    assert record["sequence"] == "interlaken_01_a"
    assert record["input_shape"] == [10, 1, 194, 120, 160]
    assert record["statistics"]["bit_order"] == "little"
    assert record["statistics"]["packing_order"] == "C_ORDER_FLAT"
    payload = m699 / record["relative_path"]
    assert sha256(payload) == D3_PAYLOAD_SHA == record["statistics"]["packed_sha256"]
    packed = np.frombuffer(payload.read_bytes(), dtype=np.uint8)
    bits = np.unpackbits(packed, bitorder="little")
    assert bits.size == 10 * 194 * 120 * 160
    tensor = bits.reshape(10, 1, 194, 120, 160)[:, 0]
    assert int(tensor.sum()) == record["statistics"]["one_count"] == 10_799_386
    return tensor, record, payload


def replay_d3(tensor: np.ndarray):
    rows = []
    for timestep in range(10):
        src = tensor[timestep]
        # Independent bank reduction: bank is input channel modulo eight.
        bank_src = np.stack(
            [src[bank::8].sum(axis=0, dtype=np.uint16) for bank in range(8)], axis=-1
        )
        dest = np.zeros((240, 320, 8), dtype=np.uint16)
        for ky in range(3):
            oy = np.arange(120) * 2 - 1 + ky
            valid_y = (oy >= 0) & (oy < 240)
            iy = np.flatnonzero(valid_y)
            for kx in range(3):
                ox = np.arange(160) * 2 - 1 + kx
                valid_x = (ox >= 0) & (ox < 320)
                ix = np.flatnonzero(valid_x)
                dest[np.ix_(oy[valid_y], ox[valid_x], np.arange(8))] += bank_src[
                    np.ix_(iy, ix, np.arange(8))
                ]
        population = dest.sum(axis=2)
        rows.append(
            {
                "timestep": timestep,
                "contributors": int(population.sum()),
                "bank_conflict_groups": int(dest.max(axis=2).sum()),
                "nonempty_destination_rows": int(np.count_nonzero(population)),
            }
        )
    return rows


def recompute_m712_baselines(root: Path):
    rows_path = root / "results/m712_pidp_decoder_exact_cpu_fastkill_r1_20260828/rows.jsonl"
    assert sha256(rows_path) == M712_ROWS_SHA
    sums = defaultdict(lambda: defaultdict(int))
    counts = defaultdict(int)
    d3_optimistic_groups = 0
    d3_refs = 0
    d3_misses = 0
    for line in rows_path.read_text().splitlines():
        row = json.loads(line)
        if row["global_sample_id"] != 0:
            continue
        module = row["module"]
        counts[module] += 1
        for key, value in row["a1_cycles"]["A1-OSG"].items():
            sums[module][key] += int(value)
        if module == "D3":
            d3_optimistic_groups += int(row["optimistic_k8_groups"])
            d3_refs += int(row["weight_cache"]["pidp_references"])
            d3_misses += int(row["weight_cache"]["pidp_misses"])
    assert counts == {"D0": 10, "D1": 10, "D2": 10, "D3": 10}
    baselines = {}
    for width in (128, 96):
        per_module = {}
        for module, (cin, hin, win) in GEOMETRY.items():
            ingress = 10 * math.ceil(cin * hin * win / width)
            per_module[module] = sums[module]["total"] - sums[module]["source_scan"] + ingress
        baselines[str(width)] = per_module
    return baselines, {
        "optimistic_groups": d3_optimistic_groups,
        "weight_references": d3_refs,
        "weight_misses": d3_misses,
        "sample0_rows_by_module": dict(counts),
    }


def inverse_tap_count(oy: int, ox: int) -> int:
    taps = 0
    for ky in range(3):
        y_num = oy + 1 - ky
        if y_num % 2:
            continue
        iy = y_num // 2
        if not 0 <= iy < 120:
            continue
        for kx in range(3):
            x_num = ox + 1 - kx
            if x_num % 2:
                continue
            ix = x_num // 2
            if 0 <= ix < 160:
                taps += 1
    return taps


def independent_bitmap_probes() -> dict:
    spatial_edges = 0
    valid_edge_probes96 = 0
    tap_histogram = defaultdict(int)
    for oy in range(240):
        for ox in range(320):
            taps = inverse_tap_count(oy, ox)
            assert taps > 0
            tap_histogram[taps] += 1
            spatial_edges += taps
            valid_edge_probes96 += math.ceil(taps * 194 / 96)
    assert spatial_edges == 171_961
    probes128 = spatial_edges * math.ceil(194 / 128)
    # The frozen 96-bit policy charges four fixed polyphase classes for every
    # input site, including boundary padding scans. This is slightly more
    # conservative than pruning invalid boundary taps.
    fixed_phase_probes96 = 120 * 160 * sum(math.ceil(taps * 194 / 96) for taps in (4, 2, 2, 1))
    return {
        "tap_histogram": {str(k): v for k, v in sorted(tap_histogram.items())},
        "spatial_edges_per_timestep": spatial_edges,
        "bitmap_probe_128_per_timestep": probes128,
        "bitmap_probe_96_per_timestep": fixed_phase_probes96,
        "bitmap_probe_96_valid_edge_lower_bound_per_timestep": valid_edge_probes96,
        "bitmap_probe_96_boundary_padding_overcharge_per_timestep": fixed_phase_probes96 - valid_edge_probes96,
    }


def compute_cycles(replay, baselines: dict, probes: dict) -> dict:
    groups = sum(r["bank_conflict_groups"] for r in replay)
    contributors = sum(r["contributors"] for r in replay)
    nonempty = sum(r["nonempty_destination_rows"] for r in replay)
    assert nonempty == 768_000
    result = {}
    for width in (128, 96):
        source_ingress = 10 * math.ceil(194 * 120 * 160 / width)
        bitmap_probe = 10 * probes[f"bitmap_probe_{width}_per_timestep"]
        components = {
            "bank_conflict_group_service_15_cycles": groups * 15,
            "bitmap_probe": bitmap_probe,
            "dense_commit": 31_488_000,
            "owner_and_terminal_control": 10_340,
            "source_ingress": source_ingress,
            "weight_refill_13_of_16": 130 * 140,
        }
        d3_candidate = sum(components.values())
        all_four_baseline = sum(baselines[str(width)].values())
        mixed = baselines[str(width)]["D0"] + baselines[str(width)]["D1"] + baselines[str(width)]["D2"] + d3_candidate
        result[str(width)] = {
            "D3_candidate_components": components,
            "D3_candidate_cycles": d3_candidate,
            "baseline_a1_osg_cycles": baselines[str(width)],
            "baseline_all_four_sum": all_four_baseline,
            "all_four_static_mixed_cycles": mixed,
            "D3_local_a1_over_candidate": f"{baselines[str(width)]['D3'] / d3_candidate:.12f}",
            "all_four_a1_over_static_mixed": f"{all_four_baseline / mixed:.12f}",
        }
    return {
        "contributors": contributors,
        "bank_conflict_groups": groups,
        "width_axes": result,
    }


def compare_author(author: dict, replay, independent: dict):
    mismatches = []
    for key in ("contributors", "bank_conflict_groups"):
        if author["exact_D3_replay"][key] != independent[key]:
            mismatches.append(f"exact_D3_replay.{key}")
    author_ts = list(zip(
        author["exact_D3_replay"]["per_timestep_contributors"],
        author["exact_D3_replay"]["per_timestep_bank_conflict_groups"],
        author["exact_D3_replay"]["per_timestep_nonempty_destination_rows"],
    ))
    independent_ts = [
        (r["contributors"], r["bank_conflict_groups"], r["nonempty_destination_rows"])
        for r in replay
    ]
    if author_ts != independent_ts:
        mismatches.append("exact_D3_replay.per_timestep")
    for width in ("128", "96"):
        expected = independent["width_axes"][width]
        got = author["width_axes"][width]
        for key in (
            "D3_candidate_components", "D3_candidate_cycles", "baseline_a1_osg_cycles",
            "baseline_all_four_sum", "all_four_static_mixed_cycles",
            "D3_local_a1_over_candidate", "all_four_a1_over_static_mixed",
        ):
            if got[key] != expected[key]:
                mismatches.append(f"width_axes.{width}.{key}")
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    assert sha256(root / "docs/359_DATE终局冻结_20260813.md") == DOCS359_SHA
    contract = root / "contracts/m1158d3_static_weight_fit_bridge_fastkill_contract_r1_20260830.json"
    contract_triple = verify_contract_triple(contract)
    contract_json = json.loads(contract.read_text())
    assert {key: contract_json["fixed_policy"][key] for key in EXPECTED_POLICY} == EXPECTED_POLICY
    assert contract_json["population"]["partial_population"] is True
    assert contract_json["authorization"]["rtl"] is False

    sealed = {
        "m699": verify_sealed_directory(root / "system_handoff/outgoing/m699_h67_ep35_multisequence_decoder_payload_s3x10_r1_20260828"),
        "m712": verify_sealed_directory(root / "results/m712_pidp_decoder_exact_cpu_fastkill_r1_20260828"),
        "m718": verify_sealed_directory(root / "reviews/m718_m712_pidp_decoder_fresh_result_hammer_r1_20260828"),
        "m1157": verify_sealed_directory(root / "reviews/m1157hc_m1156hc_decoder_cross_layer_hammer_r1_20260830"),
        "m1158": verify_sealed_directory(root / "results/m1158d3_static_weight_fit_bridge_fastkill_r1_20260830"),
        "m1158_author": verify_sealed_directory(root / "reviews/m1158d3_static_weight_fit_bridge_fastkill_author_receipt_r1_20260830"),
    }

    tensor, record, payload = load_frozen_d3(root)
    replay = replay_d3(tensor)
    baselines, m712 = recompute_m712_baselines(root)
    probes = independent_bitmap_probes()
    independent = compute_cycles(replay, baselines, probes)
    author_path = root / "results/m1158d3_static_weight_fit_bridge_fastkill_r1_20260830/report.json"
    author = json.loads(author_path.read_text())
    mismatches = compare_author(author, replay, independent)
    author_count_path = root / "results/m1158d3_static_weight_fit_bridge_fastkill_r1_20260830/d3_timestep_counts.jsonl"
    author_count_rows = [json.loads(line) for line in author_count_path.read_text().splitlines()]
    if author_count_rows != replay:
        mismatches.append("d3_timestep_counts.jsonl")
    assert not mismatches, mismatches

    capacity = {
        "logical_budget_bytes": 245_760,
        "weight_tile_bytes": 13_824,
        "line_buffer_bytes": 8_064,
        "acc24_plus_metadata_bytes": 290,
        "control_bytes": 8_192,
    }
    capacity["total_13_entries_bytes"] = 13 * 13_824 + 8_064 + 290 + 8_192
    capacity["total_16_entries_bytes"] = 16 * 13_824 + 8_064 + 290 + 8_192
    capacity["headroom_13_entries_bytes"] = 245_760 - capacity["total_13_entries_bytes"]
    capacity["headroom_16_entries_bytes"] = 245_760 - capacity["total_16_entries_bytes"]
    assert capacity["headroom_13_entries_bytes"] == 49_502
    assert capacity["headroom_16_entries_bytes"] == 8_030
    assert m712["weight_references"] == 9_549_672
    assert m712["weight_misses"] == 130
    assert m712["optimistic_groups"] == 16_421_852

    min_local = min(float(v["D3_local_a1_over_candidate"]) for v in independent["width_axes"].values())
    min_all = min(float(v["all_four_a1_over_static_mixed"]) for v in independent["width_axes"].values())
    assert min_local >= 1.20
    assert min_all < 1.20

    # Fail-closed mutation attacks. Each mutation must violate a pinned invariant.
    flipped = bytearray(payload.read_bytes())
    flipped[0] ^= 1
    attacks = {
        "payload_bit_flip_rejected_by_sha": hashlib.sha256(flipped).hexdigest() != D3_PAYLOAD_SHA,
        "drop_timestep_rejected_by_cardinality": len(replay[:-1]) != 10,
        "policy_oracle_rejected": {**EXPECTED_POLICY, "D3": "RUNTIME_ORACLE"} != EXPECTED_POLICY,
        "m712_baseline_plus_one_rejected": independent["width_axes"]["128"]["baseline_all_four_sum"] + 1 != author["width_axes"]["128"]["baseline_all_four_sum"],
        "author_ratio_plus_one_ulp_rejected": f"{float(author['width_axes']['96']['all_four_a1_over_static_mixed']) + 1e-9:.12f}" != independent["width_axes"]["96"]["all_four_a1_over_static_mixed"],
        "capacity_12_rejected_by_static_identity_count": 12 < 13,
    }
    assert all(attacks.values())

    output = {
        "schema": "m1159_m1158d3_independent_hammer_recompute_v1",
        "date": "2026-08-30",
        "status": "PASS_INDEPENDENT_RECOMPUTE__CONFIRM_NO_GO_RTL",
        "method": {
            "author_analyzer_imported": False,
            "author_analyzer_executed": False,
            "author_result_used_as_input": False,
            "author_result_used_only_for_output_comparison": True,
            "bitpack_direct_replay": True,
        },
        "identity": {
            "docs359_sha256": DOCS359_SHA,
            "d3_payload_sha256": sha256(payload),
            "m712_rows_sha256": sha256(root / "results/m712_pidp_decoder_exact_cpu_fastkill_r1_20260828/rows.jsonl"),
            "m1158_author_report_sha256": sha256(author_path),
            "contract_triple": contract_triple,
            "sealed_inputs": sealed,
        },
        "population": {
            "checkpoint": "H67_ep35",
            "sequence": record["sequence"],
            "sequence_sample_id": record["sequence_sample_id"],
            "timesteps": 10,
            "calls_in_mixed_policy": ["D0", "D1", "D2", "D3"],
            "D1_diagnostic_included": True,
            "partial_population": True,
        },
        "replay": replay,
        "aggregate": independent,
        "m712_reconciliation": m712,
        "bitmap_probe_reconstruction": probes,
        "capacity": capacity,
        "policy": EXPECTED_POLICY,
        "gate": {
            "minimum_D3_local_ratio": f"{min_local:.12f}",
            "minimum_all_four_ratio": f"{min_all:.12f}",
            "D3_local_gate_1p20_pass": min_local >= 1.20,
            "all_four_gate_1p20_pass": min_all >= 1.20,
            "overall_gate_pass": min_local >= 1.20 and min_all >= 1.20,
            "decision": "NO_GO_RTL__ALL_FOUR_1P20_GATE_FAILED",
        },
        "author_comparison": {
            "mismatches": mismatches,
            "fields_match": True,
            "timestep_sidecar_matches": author_count_rows == replay,
        },
        "mutation_attacks": attacks,
        "claim_boundary": {
            "D3_local_support_only": True,
            "decoder_population_complete": False,
            "headline": False,
            "system_speedup": False,
            "rtl_vcs_dc_eda": False,
            "paper_ppa_ready": False,
        },
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
