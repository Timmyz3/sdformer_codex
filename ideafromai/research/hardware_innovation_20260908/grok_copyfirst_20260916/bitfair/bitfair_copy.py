#!/usr/bin/env python3
"""Faithful BitFair algorithm port onto the fc1 gate (not Claude T1b).

BitFair (arXiv:2607.05445, JETCAS'26):
  - serialize magnitude bits of one operand across ALL inputs, then next bit
  - each output PE terminates independently when partial sum P_k <= theta
    (predict ReLU(final)=0)
  - theta initialized from fused BN (eq.5) then optionally learned
  - greedy adaptive bit order (ABO, Alg.1) on a calibration split

Mapping to this network (MS MLP fc1->bn1->sn2):
  Y[s] (s=0..9) plays BitFair's *weights* (sign-magnitude, bit-serial)
  A[t,s] plays BitFair's *activations* (bit-parallel)
  output t's analog of ReLU-zero is (V_t < thr_t) for D=+1, i.e. 'not fire'
  Independent PE = each of 10 time-steps may stop at a different bit
  Group-lock (C1) = wait for the slowest of 10  <-- NOT BitFair; reported as control

This script is inference-only A: BN-init theta (thr itself) + ABO on traces.
Learnable theta + temperature annealing needs sd5ai torch; flagged, not skipped
as a claim of completeness.

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
# Paper Table I: 8-bit sign-magnitude weights (1 sign + 7 mag). Using 24-bit mag
# makes the first MSB-first step all leading zeros, P=0, and theta=thr fires
# on every PE (etr=1). That is not BitFair; it is a format bug. Quantize Y to
# 8-bit SM like the paper, scale thr into the same integer MAC.
MAG_BITS = 7
Y8_SHIFT = 16  # signed24 f14 -> keep top 8 bits (sign+7 mag) via >>16


def to_signed(v, bits):
    out = np.asarray(v, np.int64)
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if np.any(out < lo) or np.any(out > hi):
        raise OverflowError("signed%d" % bits)
    return out


def load_groups(trace, g=G):
    z = np.load(trace)
    C = int(z["W"].shape[1])
    S = np.unpackbits(z["source_packed"], axis=1, bitorder="little")[:, :C].astype(np.float64)
    T, P = 10, S.shape[0] // 10
    H = z["W"].shape[0]
    W = z["W"].astype(np.float64)
    A, gamma, beta = z["A"], z["gamma"], z["beta"]
    bias, center = z["bias"], z["center"]
    theta_neu = 1.0
    R = A.sum(1).reshape(T, 1)
    direction = np.sign(gamma)

    rng = np.random.default_rng(SEED)
    ps = rng.integers(0, P, g)
    hs = rng.integers(0, H, g)

    tau = np.zeros((T, g))
    Ssub = S.reshape(T, P, C)[:, ps, :]
    Wsub = W[hs]
    Ysub = np.einsum("tgc,gc->tg", Ssub, Wsub)  # (T,g)
    for t in range(T):
        Yt = Ysub[t]
        mu = Yt.mean()
        var = ((Yt - mu) ** 2).mean()
        tau[t] = (mu * R[t, 0] + np.sqrt(var + 1e-5) / gamma[hs]
                  * (theta_neu + center[t, 0] - bias[t, 0] - beta[hs] * R[t, 0])) * direction[hs]

    A_q = to_signed(np.rint(A.astype(np.float64) * 4096), 16)
    Yq = np.clip(np.rint(Ysub * (1 << 14)), -(1 << 23), (1 << 23) - 1).astype(np.int64)
    Yq = Yq.T  # (g, 10)  Y[s]
    tau_q = np.rint(tau * (1 << 14)).astype(np.int64)  # (T,g)
    thr = np.where(direction[hs][None, :] < 0,
                   -(tau_q << 12) + 1, tau_q << 12)
    thr = to_signed(thr.T, 48)  # (g, T)
    Dflag = (direction[hs] > 0).astype(np.int64)  # (g,)
    Vfull = np.einsum("gs,ts->gt", Yq, A_q)
    raw = Vfull >= thr
    dec = np.where(Dflag[:, None] > 0, raw, ~raw)
    # BitFair 8-bit SM (paper Table I): stretch Y onto [-127,127] so the 7 mag
    # bits are actually occupied. Truncating signed24 by >>16 leaves |Y8|≪64,
    # so MSB-first step is all zeros and P<=thr fires on every PE (etr=1).
    y_abs = float(np.max(np.abs(Yq)))
    scale = 127.0 / max(y_abs, 1.0)
    Y8 = np.clip(np.rint(Yq.astype(np.float64) * scale), -127, 127).astype(np.int64)
    thr8 = np.rint(thr.astype(np.float64) * scale).astype(np.int64)
    return dict(A_q=A_q, Yq=Yq, Y8=Y8, thr=thr, thr8=thr8, Dflag=Dflag,
                Vfull=Vfull, dec=dec, name=trace.name)


def signmag(Yq):
    """Y as BitFair sign-magnitude: sign in {0,1}, mag bits [g,10,23] LSB-first."""
    sign = (Yq < 0).astype(np.int64)
    mag = np.abs(Yq)
    bits = np.stack([(mag >> j) & 1 for j in range(MAG_BITS)], axis=-1)
    return sign, bits


def partial_after_prefix(A_q, Yq, sign, bits, order, k):
    """P_k after first k mag bits in `order` (length MAG_BITS permutation). (g,T)"""
    g, Tdim = Yq.shape[0], 10
    P = np.zeros((g, Tdim), np.int64)
    for j in order[:k]:
        contrib = ((1 - 2 * sign) * bits[:, :, j] * (1 << j)).astype(np.int64)  # (g,10)
        P += contrib @ A_q.T
    return P


def bitfair_stop(A_q, Yq, thr, Dflag, order, dec_ref):
    """Per-t independent BitFair stop: first k with P_k <= thr (D=+1 not-fire analog).

    For D=-1 the comparison direction is flipped to match 'predict not-fire'.
    Remaining mag bits after stop are skipped (approximate).
    `dec_ref` is the deployment integer gate (f14/48bit), used only for flip rate.
    """
    sign, bits = signmag(Yq)
    g = Yq.shape[0]
    stop = np.full((g, 10), MAG_BITS, np.int16)
    live = np.ones((g, 10), bool)
    P = np.zeros((g, 10), np.int64)
    for step, j in enumerate(order, start=1):
        contrib = ((1 - 2 * sign) * bits[:, :, j] * (1 << j)).astype(np.int64)
        P += contrib @ A_q.T
        pred_neg = np.where(Dflag[:, None] > 0, P <= thr, P >= thr)
        newly = live & pred_neg
        stop[newly] = step
        live &= ~pred_neg
        if not live.any():
            break
    V8 = np.einsum("gs,ts->gt", Yq, A_q)
    raw = V8 >= thr
    dec8 = np.where(Dflag[:, None] > 0, raw, ~raw)
    approx = dec8.copy()
    approx[stop < MAG_BITS] = False
    flips = int((approx != dec_ref).sum())
    return stop, approx, flips, dec8


def etr_and_loss(A_q, Yq, thr, Dflag, order, dec_full):
    stop, approx, flips, _ = bitfair_stop(A_q, Yq, thr, Dflag, order, dec_full)
    etr = float((stop < MAG_BITS).mean())
    indep = float(stop.mean())  # bit-steps / (group,t)
    group = float(stop.max(1).mean())
    acc_loss = flips / max(dec_full.size, 1)
    return dict(etr=etr, indep_bits=indep, group_bits=group, flip_rate=acc_loss, flips=flips)


def greedy_abo(A_q, Yq, thr, Dflag, dec_full, cal_n):
    """Alg.1 on first cal_n groups. Remaining slots filled MSB-first."""
    Yc, tc, Dc, dc = Yq[:cal_n], thr[:cal_n], Dflag[:cal_n], dec_full[:cal_n]
    selected = []
    remaining = list(range(MAG_BITS - 1, -1, -1))  # MSB-first candidates
    msb_rest = lambda rem: sorted(rem, reverse=True)
    for _slot in range(MAG_BITS):
        best, best_sc = None, -1.0
        for b in remaining:
            test = selected + [b] + msb_rest([x for x in remaining if x != b])
            st = etr_and_loss(A_q, Yc, tc, Dc, test, dc)
            sc = st["etr"] / (st["flip_rate"] + 1e-9)
            if sc > best_sc:
                best_sc, best = sc, b
        selected.append(best)
        remaining.remove(best)
    return selected


def exact_interval_stop(A_q, Yq, thr):
    """C1-style exact lock, per-t, MSB-first two's complement planes bit23..0."""
    g = Yq.shape[0]
    P_t = A_q.clip(min=0).sum(1)
    N_t = A_q.clip(max=0).sum(1)
    stop = np.full((g, 10), 24, np.int16)
    frozen = np.zeros((g, 10), bool)
    for j in range(23, -1, -1):  # remaining m=j bits after consuming bit j
        Vtop = (Yq >> j) @ A_q.T
        m = j
        vmin = (Vtop << m) + N_t * ((1 << m) - 1)
        vmax = (Vtop << m) + P_t * ((1 << m) - 1)
        lock = (vmin >= thr) | (vmax < thr)
        newly = lock & ~frozen
        stop[newly] = 24 - j  # planes consumed including this one (plus would count sop separately)
        frozen |= lock
        if frozen.all():
            break
    return stop


def main():
    msb_order = list(range(MAG_BITS - 1, -1, -1))
    rows = []
    for tr in TRACES:
        d = load_groups(tr)
        A_q, Yq, thr, Dflag, dec = d["A_q"], d["Yq"], d["thr"], d["Dflag"], d["dec"]
        Y8, thr8 = d["Y8"], d["thr8"]
        cal = min(2000, Yq.shape[0] // 4)
        msb = etr_and_loss(A_q, Y8, thr8, Dflag, msb_order, dec)
        abo = greedy_abo(A_q, Y8, thr8, Dflag, dec, cal)
        abo_st = etr_and_loss(A_q, Y8, thr8, Dflag, abo, dec)
        ex = exact_interval_stop(A_q, Yq, thr)
        row = {
            "trace": d["name"],
            "bitfair_msb": msb,
            "abo_order": abo,
            "bitfair_abo": abo_st,
            "exact_indep_planes": float(ex.mean()),
            "exact_group_planes": float(ex.max(1).mean()),
            "fx24_ratio_bitfair_indep": msb["indep_bits"] / 24.0,
            "fx24_ratio_bitfair_group": msb["group_bits"] / 24.0,
            "fx24_ratio_exact_indep": float(ex.mean()) / 24.0,
            "fx24_ratio_exact_group": float(ex.max(1).mean()) / 24.0,
        }
        rows.append(row)
        print("%s  BF-msb indep=%.2f group=%.2f flip=%.4f etr=%.3f | ABO indep=%.2f flip=%.4f | exact indep=%.2f group=%.2f"
              % (d["name"], msb["indep_bits"], msb["group_bits"], msb["flip_rate"], msb["etr"],
                 abo_st["indep_bits"], abo_st["flip_rate"], row["exact_indep_planes"], row["exact_group_planes"]))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "bitfair_copy.json").write_text(json.dumps(rows, indent=2))
    print("wrote", out / "bitfair_copy.json")


if __name__ == "__main__":
    main()
