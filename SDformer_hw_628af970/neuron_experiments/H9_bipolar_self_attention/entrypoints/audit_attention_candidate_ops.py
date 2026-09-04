"""Audit incremental attention operations for unified H60-family candidates.

This report is separate from ``spike_profile.json``. The latter measures spike
activity and a spike-energy proxy but does not include overlay attention
control, popcount trees, context reductions, or fixed reciprocal operations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


E_LOGIC_PJ = 0.1
E_ADD_PJ = 0.9
E_MAC_PJ = 4.6


def is_power_of_two(value: float) -> bool:
    if value <= 0 or not float(value).is_integer():
        return False
    integer = int(value)
    return integer & (integer - 1) == 0


def counts_for_record(record: dict[str, Any], attention: dict[str, Any]) -> dict[str, int]:
    windows = int(record.get("batch_windows", 0))
    heads = int(record.get("num_heads", 0))
    tokens = int(record.get("tokens", 0))
    lanes = int(record.get("head_dim", 0))
    token_heads = windows * heads * tokens
    lane_entries = token_heads * lanes
    window_head_channels = windows * heads * lanes
    window_heads = windows * heads
    pair_scores = token_heads * tokens

    counts = {
        "tx_lane_logic": lane_entries,
        "tx_popcount_add": token_heads * max(lanes - 1, 0),
        "shiftmax_tokens": token_heads,
        "gate_k_entries": lane_entries,
        "incremental_logic": 0,
        "incremental_add": 0,
        "incremental_mac": 0,
    }
    if float(attention.get("binary_motion_xor_alpha", 0.0) or 0.0) != 0.0:
        counts["incremental_logic"] += lane_entries + token_heads  # XOR lanes + dyadic shift
        counts["incremental_add"] += token_heads * max(lanes - 1, 0) + token_heads

    score_scale = float(attention.get("score_scale", 1.0) or 1.0)
    if score_scale != 1.0 and is_power_of_two(score_scale):
        counts["incremental_logic"] += token_heads

    if bool(attention.get("event_temperature_enabled", False)):
        counts["incremental_logic"] += lane_entries  # Q/K OR
        counts["incremental_add"] += token_heads * max(lanes - 1, 0)  # popcount tree
        counts["incremental_logic"] += 2 * token_heads  # leading-one + variable shift

    if bool(attention.get("context_broadcast_enabled", False)):
        counts["incremental_add"] += window_head_channels * max(tokens - 1, 0)
        counts["incremental_mac"] += window_head_channels  # exact fixed 1/N reciprocal
        counts["incremental_add"] += lane_entries  # token + context
        counts["incremental_logic"] += lane_entries  # final divide by two

    mode = str(attention.get("mode", "h60"))
    if mode in {"binary_alpha_xnor_matrix_shiftmax", "strict_binary_alpha_xnor_shiftmax"}:
        # N Q/K comparisons and N weighted K values per query, compared with
        # H60's one score and one gated K value per token.
        counts["incremental_logic"] += max(pair_scores * lanes - lane_entries, 0)
        counts["incremental_add"] += max(
            pair_scores * max(lanes - 1, 0) - token_heads * max(lanes - 1, 0), 0
        )
        counts["incremental_logic"] += max(pair_scores - token_heads, 0)
        counts["incremental_mac"] += max(pair_scores * lanes - lane_entries, 0)
    elif mode in {"binary_axnor_temporal_pair_shiftmax", "tp_ttx", "h66_tp"}:
        neighbors = 2
        counts["incremental_logic"] += (neighbors - 1) * (lane_entries + token_heads)
        counts["incremental_add"] += (neighbors - 1) * (
            token_heads * max(lanes - 1, 0) + lane_entries
        )
        counts["incremental_mac"] += (neighbors - 1) * lane_entries
        if float(attention.get("matrix_diag_bias", 0.0) or 0.0) != 0.0:
            counts["incremental_add"] += token_heads
    elif mode in {"binary_axnor_local5_shiftmax", "lr_ttx", "h66_lr"}:
        neighbors = 5
        counts["incremental_logic"] += (neighbors - 1) * (lane_entries + token_heads)
        counts["incremental_add"] += (neighbors - 1) * (
            token_heads * max(lanes - 1, 0) + lane_entries
        )
        counts["incremental_mac"] += (neighbors - 1) * lane_entries
    elif mode in {
        "binary_axnor_local5_tp_shiftmax",
        "local5_tp",
        "h66f_local5_tp",
        "h66f",
    }:
        # self + temporal peer + 4-axial spatial neighbors
        neighbors = 6
        counts["incremental_logic"] += (neighbors - 1) * (lane_entries + token_heads)
        counts["incremental_add"] += (neighbors - 1) * (
            token_heads * max(lanes - 1, 0) + lane_entries
        )
        counts["incremental_mac"] += (neighbors - 1) * lane_entries
    elif mode in {
        "binary_axnor_local5_motion_shiftmax",
        "local5_motion",
        "h66g_local5_motion",
        "h66g",
    }:
        neighbors = 5
        counts["incremental_logic"] += (neighbors - 1) * (lane_entries + token_heads)
        counts["incremental_add"] += (neighbors - 1) * (
            token_heads * max(lanes - 1, 0) + lane_entries
        )
        counts["incremental_mac"] += (neighbors - 1) * lane_entries
        # H67-style motion XOR on self lane (already counted if alpha!=0 above,
        # but local5_motion defaults alpha=0.25 even when config omits the field).
        if float(attention.get("binary_motion_xor_alpha", 0.0) or 0.0) == 0.0:
            counts["incremental_logic"] += lane_entries + token_heads
            counts["incremental_add"] += token_heads * max(lanes - 1, 0) + token_heads
    elif mode in {"hamming_binary_direct", "spikevideoformer_hamming", "h21a"}:
        # K^T V followed by Q(K^T V). Q/K are sign bits, so a dedicated
        # implementation uses add/sub accumulation instead of generic MACs.
        kv_adds = window_heads * lanes * lanes * max(tokens - 1, 0)
        qkv_adds = token_heads * lanes * max(lanes - 1, 0)
        counts["incremental_logic"] += 2 * lane_entries
        counts["incremental_add"] += max(
            kv_adds + qkv_adds - token_heads * max(lanes - 1, 0), 0
        )
        counts["incremental_logic"] += token_heads * lanes
    elif mode in {"binary_de9_match_code", "de9_match_code"}:
        # Nine cross-time offsets retain event-event and silence-silence as
        # separate evidence, then project the 18-entry descriptor through a
        # static per-head codebook. Counts are incremental over one H60 score.
        neighbors = 9
        descriptors = 18
        comparisons = 2 * neighbors
        counts["incremental_logic"] += max(comparisons * lane_entries - lane_entries, 0)
        counts["incremental_add"] += max(
            comparisons * token_heads * max(lanes - 1, 0)
            - token_heads * max(lanes - 1, 0),
            0,
        )
        counts["incremental_logic"] += max(descriptors * token_heads - token_heads, 0)
        counts["incremental_mac"] += token_heads * descriptors * lanes
    elif mode in {"binary_mc49_match_code", "mc49_match_code"}:
        # Exact EEMFlow-style 49-offset cross-time XNOR descriptor followed by
        # a static per-head codebook; there is no dynamic weights@K carrier.
        descriptors = 49
        counts["incremental_logic"] += max(descriptors * lane_entries - lane_entries, 0)
        counts["incremental_add"] += max(
            descriptors * token_heads * max(lanes - 1, 0)
            - token_heads * max(lanes - 1, 0),
            0,
        )
        counts["incremental_logic"] += max(descriptors * token_heads - token_heads, 0)
        counts["incremental_mac"] += token_heads * descriptors * lanes
    elif mode in {"binary_ax17_match_code", "ax17_match_code"}:
        # Horizontal/vertical radius-4 cross-time descriptor (center shared),
        # inspired by Flow1D's orthogonal factorization but without its dynamic
        # attention/value carrier.
        descriptors = 17
        counts["incremental_logic"] += max(descriptors * lane_entries - lane_entries, 0)
        counts["incremental_add"] += max(
            descriptors * token_heads * max(lanes - 1, 0)
            - token_heads * max(lanes - 1, 0),
            0,
        )
        counts["incremental_logic"] += max(descriptors * token_heads - token_heads, 0)
        counts["incremental_mac"] += token_heads * descriptors * lanes
    elif mode in {"binary_pc9_patch_match_code", "pc9_patch_match_code", "h76_pc9"}:
        # Nine alpha-XNOR planes are reused by a fixed 3x3 4/2/1 dyadic
        # corresponding-patch filter. Boundary normalization uses a static ROM.
        neighbors = 9
        descriptors = 9
        comparisons = 2 * neighbors
        counts["incremental_logic"] += max(comparisons * lane_entries - lane_entries, 0)
        counts["incremental_add"] += max(
            comparisons * token_heads * max(lanes - 1, 0)
            - token_heads * max(lanes - 1, 0),
            0,
        )
        counts["incremental_logic"] += token_heads * neighbors * neighbors
        counts["incremental_add"] += token_heads * neighbors * (neighbors - 1)
        counts["incremental_mac"] += token_heads * descriptors * lanes
    elif mode in {"binary_lc4_match_code", "lc4_match_code", "h77_lc4"}:
        # One AND-popcount plus Q/K population counts derives all four
        # contingencies; four dyadic terms produce each of nine scores.
        neighbors = 9
        descriptors = 9
        counts["incremental_logic"] += max(neighbors * lane_entries - lane_entries, 0)
        counts["incremental_add"] += max(
            neighbors * token_heads * max(lanes - 1, 0)
            - token_heads * max(lanes - 1, 0),
            0,
        )
        counts["incremental_add"] += token_heads * neighbors * 6
        counts["incremental_logic"] += token_heads * neighbors * 4
        counts["incremental_mac"] += token_heads * descriptors * lanes
    elif mode in {"binary_g4_match_code", "g4_match_code", "h78_g4"}:
        # Four byte-sliced groups retain four Shiftmax9 distributions. The
        # number of compared bits is unchanged from scalar Omega9 matching.
        neighbors = 9
        groups = 4
        descriptors = neighbors * groups
        comparisons = 2 * neighbors
        counts["incremental_logic"] += max(comparisons * lane_entries - lane_entries, 0)
        counts["incremental_add"] += max(
            comparisons * token_heads * max(lanes - groups, 0)
            - token_heads * max(lanes - 1, 0),
            0,
        )
        counts["incremental_logic"] += max(descriptors * token_heads - token_heads, 0)
        counts["incremental_mac"] += token_heads * descriptors * lanes
    elif mode in {"binary_cf10_match_code", "cf10_match_code", "h79_cf10"}:
        # Nine local XNOR scores plus a top-2/activity null score. The tenth
        # codeword is hard zero, so only nine rows reach the static projection.
        neighbors = 9
        comparisons = 2 * neighbors
        counts["incremental_logic"] += max(comparisons * lane_entries - lane_entries, 0)
        counts["incremental_add"] += max(
            comparisons * token_heads * max(lanes - 1, 0)
            - token_heads * max(lanes - 1, 0),
            0,
        )
        counts["incremental_logic"] += token_heads * 15  # conservative top-2 comparator net
        counts["incremental_add"] += token_heads * (max(lanes - 1, 0) + 4)
        counts["incremental_mac"] += token_heads * (neighbors * lanes + 2)
    elif mode in {"binary_dn9_match_code", "dn9_match_code", "h80_dn9"}:
        # The same nine local XNOR scores feed row and incoming-destination
        # Shiftmax9. Every valid edge then performs one Q1.7 gate product.
        neighbors = 9
        comparisons = 2 * neighbors
        counts["incremental_logic"] += max(comparisons * lane_entries - lane_entries, 0)
        counts["incremental_add"] += max(
            comparisons * token_heads * max(lanes - 1, 0)
            - token_heads * max(lanes - 1, 0),
            0,
        )
        counts["incremental_logic"] += max(2 * neighbors * token_heads - token_heads, 0)
        counts["incremental_mac"] += token_heads * (neighbors * lanes + neighbors)
    return counts


def summarize(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    if not records:
        raise ValueError("hardware profile contains no H60 records")
    if any(int(record.get("batch_windows", 0)) <= 0 for record in records):
        raise ValueError("hardware profile predates batch_windows operation-audit support")
    attention = config.get("bsa_attention", {})
    keys = (
        "tx_lane_logic", "tx_popcount_add", "shiftmax_tokens", "gate_k_entries",
        "incremental_logic", "incremental_add", "incremental_mac",
    )
    totals = {key: 0 for key in keys}
    for record in records:
        counts = counts_for_record(record, attention)
        for key in keys:
            totals[key] += counts[key]
    samples = int(config.get("_profile_samples", 1) or 1)
    per_sample = {key: value / samples for key, value in totals.items()}
    base_proxy_pj = (
        per_sample["tx_lane_logic"] * E_LOGIC_PJ
        + per_sample["tx_popcount_add"] * E_ADD_PJ
    )
    incremental_proxy_pj = (
        per_sample["incremental_logic"] * E_LOGIC_PJ
        + per_sample["incremental_add"] * E_ADD_PJ
        + per_sample["incremental_mac"] * E_MAC_PJ
    )
    return {
        "experiment": config.get("experiment", "unknown"),
        "profile_samples": samples,
        "counts_total": totals,
        "counts_per_sample": per_sample,
        "base_tx_score_proxy_uj": base_proxy_pj / 1e6,
        "incremental_attention_proxy_uj": incremental_proxy_pj / 1e6,
        "incremental_vs_base_tx_score_pct": 100.0 * incremental_proxy_pj / base_proxy_pj if base_proxy_pj else 0.0,
        "proxy_constants_pj": {"logic": E_LOGIC_PJ, "add": E_ADD_PJ, "mac": E_MAC_PJ},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    profile = json.loads(args.profile_json.read_text(encoding="utf-8"))
    records = profile["summary"]["h60_records"]
    samples = int(profile.get("samples", 1) or 1)
    rows = []
    for config_path in args.config:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        config["_profile_samples"] = samples
        row = summarize(records, config)
        row["config"] = str(config_path)
        rows.append(row)

    result = {
        "source_profile": str(args.profile_json),
        "scope": "attention datapath only; excludes SRAM/NoC and shared Shiftmax implementation energy",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = args.output.with_suffix(".md")
    lines = [
        "# Unified H60 Candidate Attention Operation Audit", "",
        "`spike_profile.json` energy does not include these incremental attention operations.", "",
        "| candidate | logic ops/sample | add ops/sample | fixed MAC/sample | incremental proxy (uJ) | vs base TX-score proxy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        counts = row["counts_per_sample"]
        lines.append(
            f"| {row['experiment']} | {counts['incremental_logic']:.0f} | "
            f"{counts['incremental_add']:.0f} | {counts['incremental_mac']:.0f} | "
            f"{row['incremental_attention_proxy_uj']:.3f} | "
            f"{row['incremental_vs_base_tx_score_pct']:.2f}% |"
        )
    lines += [
        "", "Proxy constants: logic 0.1 pJ, add 0.9 pJ, MAC 4.6 pJ (45 nm literature proxy).",
        "This is an operation audit, not post-layout energy and not an SRAM/NoC model.",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
