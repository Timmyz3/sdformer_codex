#!/usr/bin/env python3
"""Faithful BitL *control-path* port: hybrid row/column lookup with dynamic pivot.

BitL (MICRO'25, 10.1145/3725843.3756044) is a bit-wise MAC unit that
switches between horizontal (row) and vertical (column) lookups on a
bit sub-tile so the critical path is not stuck in unidirectional
MSB->LSB bit-serial. It is NOT 'replace a dot-product with two 5-bit
LUTs' (that was Claude T10).

This script counts *cycles to consume all 1-bits* of a 10 x 24 two's
complement Y bit-matrix under:

  seq_msb     : 24 cycles (all 10 lanes in parallel, C1 supply w/o cert)
  v_skip      : #planes that contain at least one 1  (column-only skip)
  h_skip      : max popcount over the 10 rows        (row-serial skip)
  bitl_pivot  : greedy dynamic pivot on 8-bit sub-tiles
                one cycle either clears one remaining row (H LUT of that
                row-slice) or one remaining column (V LUT of 10-bit plane)

A is not in the cycle model: BitL's claim is datapath throughput of the
bit-tile, independent of the certificate. Overlaying exact-interval
lock on top is the X step, not the A step.

Self-contained. Reads bn_state traces only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HW = ROOT.parent
TRACES = sorted((HW / "bn_state").glob("trace_*.npz"))
G = 8000
SEED = 20260916
ROWS, COLS = 10, 24
TILE_W = 8


def load_Yq(trace, g=G):
    z = np.load(trace)
    C = int(z["W"].shape[1])
    S = np.unpackbits(z["source_packed"], axis=1, bitorder="little")[:, :C].astype(np.float64)
    T, P = 10, S.shape[0] // 10
    H = z["W"].shape[0]
    rng = np.random.default_rng(SEED)
    ps = rng.integers(0, P, g)
    hs = rng.integers(0, H, g)
    Ssub = S.reshape(T, P, C)[:, ps, :]
    Wsub = z["W"].astype(np.float64)[hs]
    Ysub = np.einsum("tgc,gc->tg", Ssub, Wsub)
    Yq = np.clip(np.rint(Ysub * (1 << 14)), -(1 << 23), (1 << 23) - 1).astype(np.int64).T
    return Yq, trace.name


def bit_matrix(Yq):
    """(g, 10, 24) bits, column 0 = bit23 (MSB)."""
    bits = np.stack([((Yq >> j) & 1) for j in range(23, -1, -1)], axis=-1)
    return bits.astype(np.uint8)


def bitl_pivot_cycles(tile):
    """tile: (R, W) 0/1. Greedy: pick densest remaining row or col each cycle."""
    M = tile.copy()
    cyc = 0
    # cap: never worse than R+W
    for _ in range(M.shape[0] + M.shape[1] + 1):
        if M.sum() == 0:
            return cyc
        row_n = M.sum(1)
        col_n = M.sum(0)
        if int(row_n.max()) >= int(col_n.max()):
            r = int(np.argmax(row_n))
            M[r, :] = 0
        else:
            c = int(np.argmax(col_n))
            M[:, c] = 0
        cyc += 1
    return cyc


def one_trace(Yq, name):
    B = bit_matrix(Yq)  # (g,10,24)
    g = B.shape[0]
    seq = np.full(g, COLS, np.int16)
    v_skip = B.any(1).sum(1)  # planes with a 1
    h_skip = B.sum(2).max(1)  # busiest row popcount
    # three 10x8 subtiles along the bit axis
    pivot = np.zeros(g, np.int16)
    for t0 in range(0, COLS, TILE_W):
        sl = B[:, :, t0:t0 + TILE_W]
        for gi in range(g):
            pivot[gi] += bitl_pivot_cycles(sl[gi])
    return {
        "trace": name,
        "seq_msb": float(seq.mean()),
        "v_skip": float(v_skip.mean()),
        "h_skip": float(h_skip.mean()),
        "bitl_pivot": float(pivot.mean()),
        "vs_seq": float(pivot.mean() / COLS),
        "ones_per_group": float(B.sum((1, 2)).mean()),
    }


def main():
    rows = []
    for tr in TRACES:
        Yq, name = load_Yq(tr)
        row = one_trace(Yq, name)
        rows.append(row)
        print("%s  seq=%.1f  V-skip=%.2f  H-skip=%.2f  BitL-pivot=%.2f  (%.1f%% of 24)"
              % (name, row["seq_msb"], row["v_skip"], row["h_skip"],
                 row["bitl_pivot"], 100 * row["vs_seq"]))
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "bitl_copy.json").write_text(json.dumps(rows, indent=2))
    print("wrote", out / "bitl_copy.json")


if __name__ == "__main__":
    main()
