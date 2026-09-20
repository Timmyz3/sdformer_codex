#!/usr/bin/env python3
"""CPU-only read-only replay of the C1 cross-pair statistic-plane calibration.

Reads the sealed post-G0 ordered traces (ep29 proxy-only / ep44 real-Q) from
hw_autoresearch_nts07/results/, recomputes the pre-registered C1 statistics
(G3/G4/G5/G6 proxy AND real-Q side), and writes a JSON + MD receipt into this
review directory.  Writes nothing outside this directory.  No GPU.  No dump.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/private_data/work/sdformer_codex/SDformer")
OUT = Path(__file__).resolve().parent

EP29 = ROOT / ("hw_autoresearch_nts07/results/"
               "local5_fullres_postg0_qfsa_profile100_20260730/ordered_term_items.npz")
EP44 = ROOT / ("hw_autoresearch_nts07/results/"
               "local5_ep44_hardware_rebind_20260815_profile100/ordered_term_items.npz")
SEALED_EP29_FULL = 0.7936   # docs/CLAUDE_LOCAL5_QLABEL_DUMP_SPEC_20260818 §4.3
SEALED_EP44_FULL = 0.7859762898038378  # dump script SEALED_EP44_FULL_MATCH

M64 = np.uint64(0x5555555555555555)
M32 = np.uint64(0x3333333333333333)
M16 = np.uint64(0x0F0F0F0F0F0F0F0F)
M8 = np.uint64(0x0101010101010101)


def popcount64(x: np.ndarray) -> np.ndarray:
    """Vectorised SWAR popcount for uint64 arrays."""
    x = x.astype(np.uint64, copy=False)
    x = x - ((x >> np.uint64(1)) & M64)
    x = (x & M32) + ((x >> np.uint64(2)) & M32)
    x = (x + (x >> np.uint64(4))) & M16
    return ((x * M8) >> np.uint64(56)).astype(np.uint8)


def boundary_pairs(offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    """(p_prev, p_next, n_within, n_cross) for the group-aware descriptor stream."""
    within_prev: list[np.ndarray] = []
    cross_prev: list[int] = []
    for s, e in zip(offsets[:-1], offsets[1:]):
        if e - s >= 2:
            within_prev.append(np.arange(int(s), int(e) - 1, dtype=np.int64))
        if e - s >= 1:
            cross_prev.append(int(e) - 1)
    w_prev = np.concatenate(within_prev) if within_prev else np.zeros(0, np.int64)
    c_prev = np.asarray(cross_prev[:-1], dtype=np.int64)  # no successor for last group
    prev = np.concatenate([w_prev, c_prev])
    nxt = np.concatenate([w_prev + 1, c_prev + 1])
    order = np.argsort(prev, kind="stable")
    return prev[order], nxt[order], int(w_prev.size), int(c_prev.size)


def stat_block(name: str, kpop: np.ndarray, qpop: np.ndarray, qvec: np.ndarray | None,
               kvec: np.ndarray, prev: np.ndarray, nxt: np.ndarray,
               tag: str) -> dict:
    hit = qpop[nxt] == kpop[prev]
    bz = (kpop[prev] == 0) & (qpop[nxt] == 0)
    nz = (kpop[prev] > 0) & (qpop[nxt] > 0)
    n_hit = int(hit.sum())
    out = {
        "tag": tag,
        "boundaries": int(prev.size),
        "full_match_rate": float(hit.mean()) if prev.size else float("nan"),
        "hits": n_hit,
        "both_zero_share_of_hits": float((hit & bz).sum() / max(1, n_hit)),
        "nz_nz_boundaries": int(nz.sum()),
        "nz_nz_match_rate": float((hit & nz).sum() / max(1, int(nz.sum()))),
        "k_prev_zero_frac": float((kpop[prev] == 0).mean()) if prev.size else float("nan"),
        "q_next_zero_frac": float((qpop[nxt] == 0).mean()) if prev.size else float("nan"),
    }
    if qvec is not None:
        out["vector_identity_rate"] = float(np.equal(qvec[nxt], kvec[prev]).mean())
        out["vector_identity_rate_nz_nz"] = float(
            np.equal(qvec[nxt][nz], kvec[prev][nz]).mean()) if nz.any() else float("nan")
        out["vector_identity_rate_nz_nz_hits"] = float(
            np.equal(qvec[nxt][nz & hit], kvec[prev][nz & hit]).mean()
        ) if (nz & hit).any() else float("nan")
    return out


def replay(name: str, path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    kpop = z["source_k_popcount"].astype(np.int64)
    pc_k = popcount64(z["descriptor_k_bitmap"]).astype(np.int64)
    has_q = "descriptor_q_bitmap" in z.files
    qpop_true = popcount64(z["descriptor_q_bitmap"]).astype(np.int64) if has_q else None
    offsets = z["descriptor_group_offsets"].astype(np.int64)
    n = kpop.size

    flat_prev = np.arange(0, n - 1, dtype=np.int64)
    flat_next = flat_prev + 1
    g_prev, g_next, n_within, n_cross = boundary_pairs(offsets)

    res: dict = {
        "name": name,
        "npz": str(path.relative_to(ROOT)),
        "descriptors": int(n),
        "groups": int(offsets.size - 1),
        "has_descriptor_q_bitmap": bool(has_q),
        "kpop_equals_pc_kbitmap": bool(np.array_equal(kpop, pc_k)),
        "flat_boundaries": int(flat_prev.size),
        "group_aware_boundaries": int(g_prev.size),
        "group_aware_within": n_within,
        "group_aware_cross": n_cross,
        "kpop_zero_frac": float((kpop == 0).mean()),
    }
    # proxy (k-bitmap-side) statistics, both adjacency conventions
    res["proxy_flat"] = stat_block(name, kpop, pc_k, None, z["descriptor_k_bitmap"],
                                   flat_prev, flat_next, "proxy_flat")
    res["proxy_group_aware"] = stat_block(name, kpop, pc_k, None, z["descriptor_k_bitmap"],
                                          g_prev, g_next, "proxy_group_aware")
    if has_q:
        qvec = z["descriptor_q_bitmap"]
        res["trueQ_flat"] = stat_block(name, kpop, qpop_true, qvec, z["descriptor_k_bitmap"],
                                       flat_prev, flat_next, "trueQ_flat")
        res["trueQ_group_aware"] = stat_block(name, kpop, qpop_true, qvec,
                                              z["descriptor_k_bitmap"],
                                              g_prev, g_next, "trueQ_group_aware")
        # G6 same-token Q-event vs K-event (sealed closure of the dump's G6)
        res["G6_sealed_same_token_pop_eq"] = float(np.equal(qpop_true, kpop).mean())
        res["G6_sealed_same_token_vec_eq"] = float(np.equal(qvec, z["descriptor_k_bitmap"]).mean())
        act = (kpop > 0) | (qpop_true > 0)
        res["G6_sealed_pop_eq_tokens_with_activity"] = float(
            np.equal(qpop_true[act], kpop[act]).mean())
        res["G6_sealed_vec_eq_tokens_with_activity"] = float(
            np.equal(qvec[act], z["descriptor_k_bitmap"][act]).mean())
        res["tokens_with_activity"] = int(act.sum())
        res["q1_true_zero_frac"] = float((qpop_true == 0).mean())
    return res


def g7_score_tuple_replay(name: str, path: Path) -> dict:
    """G7 (389 defence) approximation on the sealed ep44 layout.

    Recomputes the 5-lane Q7 score code per destination from the real Q/K
    bitmaps with the L3856-3874 stencil geometry (self,N,S,E,W, clamp) and
    measures score-tuple equality rate over statistic-preserving boundaries.
    """
    z = np.load(path, allow_pickle=True)
    if "descriptor_q_bitmap" not in z.files:
        return {"name": name, "skipped": "no descriptor_q_bitmap"}
    kpop = z["source_k_popcount"].astype(np.int64)
    qv = z["descriptor_q_bitmap"].astype(np.uint64)
    kv = z["descriptor_k_bitmap"].astype(np.uint64)
    offsets = z["descriptor_group_offsets"].astype(np.int64)
    n = qv.size
    side, plane_tokens = 15, 225

    # canonical 450-token layout is identical inside every group
    n = qv.size
    dest_local = np.arange(n, dtype=np.int64) % 450
    group_start = np.repeat(offsets[:-1], 450)
    plane = dest_local // plane_tokens
    pos = dest_local % plane_tokens
    yy = pos // side
    xx = pos % side
    offs = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

    codes = np.zeros((n, 5), dtype=np.int64)
    lane_valid = 0
    lane_nbr_k0 = 0
    lane_dst_q0 = 0
    for lane, (dy, dx) in enumerate(offs):
        ny = np.clip(yy + dy, 0, side - 1)
        nx = np.clip(xx + dx, 0, side - 1)
        nbr = group_start + plane * plane_tokens + ny * side + nx
        kb_all = kv[nbr]
        k1_all = popcount64(kb_all).astype(np.int64)
        q1_all = popcount64(qv).astype(np.int64)
        valid = ((yy + dy >= 0) & (yy + dy < side)
                 & (xx + dx >= 0) & (xx + dx < side)) if lane > 0 else np.ones(n, bool)
        lane_valid += int(valid.sum())
        lane_nbr_k0 += int((valid & (k1_all == 0)).sum())
        lane_dst_q0 += int((valid & (q1_all == 0)).sum())
        qb = qv  # destination own Q event vector (per destination)
        kb = kb_all
        n11 = popcount64(qb & kb).astype(np.int64)
        q1 = popcount64(qb).astype(np.int64)
        k1 = k1_all
        codes[:, lane] = np.rint((65 * n11 + 32 - q1 - k1) / 16.0).astype(np.int64)

    g_prev, g_next, n_within, n_cross = boundary_pairs(offsets)
    within = g_prev < g_next  # group-aware arrays are already ordered
    qpop = popcount64(qv).astype(np.int64)
    hit = qpop[g_next] == kpop[g_prev]
    nz = (kpop[g_prev] > 0) & (qpop[g_next] > 0)
    tup_eq = np.all(codes[g_prev] == codes[g_next], axis=1)
    res = {
        "name": name,
        "scope": "group-aware boundaries; raw codes (no invalid clamp)",
        "boundaries": int(g_prev.size),
        "hits": int(hit.sum()),
        "score_tuple_eq_on_hits": float(tup_eq[hit].mean()) if hit.any() else float("nan"),
        "score_tuple_eq_all": float(tup_eq.mean()),
        "score_tuple_eq_within_group_hits": float(
            tup_eq[hit & within].mean()) if (hit & within).any() else float("nan"),
        "score_tuple_eq_nz_nz_hits": float(
            tup_eq[hit & nz].mean()) if (hit & nz).any() else float("nan"),
        "score_tuple_eq_nz_nz_all": float(
            tup_eq[nz].mean()) if nz.any() else float("nan"),
        "nz_nz_boundaries": int(nz.sum()),
        "hits_nz_nz": int((hit & nz).sum()),
        "lane_edges_valid": lane_valid,
        "lane_edges_neighbor_K_zero_share": lane_nbr_k0 / max(1, lane_valid),
        "lane_edges_dest_Q_zero_share": lane_dst_q0 / max(1, lane_valid),
        "n_within": int(n_within),
    }
    del n_cross
    return res


def main() -> int:
    results = {
        "script": Path(__file__).name,
        "mode": "CPU-only read-only replay; no GPU; no dump; no writes outside review dir",
        "expected": {
            "ep29_proxy_flat_full_match": 0.7936,
            "ep44_proxy_flat_full_match": SEALED_EP44_FULL,
            "ep44_proxy_both_zero_share": 0.9492,
            "ep44_proxy_nz_nz": 0.2252,
            "ep44_k_zero_frac": 0.7843,
        },
        "runs": [],
    }
    for name, path in (("ep29", EP29), ("ep44", EP44)):
        r = replay(name, path)
        r["g7_replay"] = g7_score_tuple_replay(name, path)
        results["runs"].append(r)
        print(json.dumps(r, indent=2, ensure_ascii=False), flush=True)

    (OUT / "cpu_replay_local5_c1_statistics_20260917.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[replay] wrote cpu_replay_local5_c1_statistics_20260917.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
