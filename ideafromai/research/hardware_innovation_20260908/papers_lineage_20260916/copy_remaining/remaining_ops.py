"""Fair leftover extras vs nnz(S)*Cout word-adds and vs leftover PSN T×T MACs.

KEEP bar is extra_save_after_tax >= 0.15 on the named remaining op.
BitWave vs word-add cycles is 1 - bit_ops/base (negative if bit-serial costs more).
CGNet/SnaPEA extra is skipped/total over ALL Cout, not skipped/processed.
"""
from __future__ import annotations

import torch

KEEP = 0.15
WBITS = 7
TAX_RUNTIME = 1.0 / 32.0


def keep_kill(extra, _name=''):
    return 'KEEP' if extra >= KEEP else 'KILL'


def q8_abs(w):
    a = w.detach().float().abs()
    s = a.amax().clamp_min(1e-8) / 127.0
    return (a / s).round().clamp(0, 127).to(torch.int32)


def word_stats(S, W):
    """S: (N,Cin) 0/1; W: (Cout,Cin). Remaining word-adds = nnz(S)*Cout."""
    S = (S.detach() != 0)
    W = W.detach().float()
    N, Cin = S.shape
    Cout = W.shape[0]
    nnz = int(S.sum().item())
    base = nnz * Cout
    w_col_nz = (W.abs() >= 1e-12).sum(dim=0)
    dual = int((S.float() @ w_col_nz.float()).sum().item())
    extra_dual = 1.0 - dual / max(base, 1)
    Wq = q8_abs(W)
    dual_q8 = int((S.float() @ (Wq != 0).sum(0).float()).sum().item())
    extra_dual_q8 = 1.0 - dual_q8 / max(base, 1)
    bit_col = torch.zeros(Cin, dtype=torch.float64, device=W.device)
    for b in range(WBITS):
        bit_col += ((Wq >> b) & 1).sum(dim=0).double()
    bit_ops = float((S.double() @ bit_col).sum().item())
    extra_bitwave_vs_word = 1.0 - bit_ops / max(base, 1)
    extra_bitwave_vs_7serial = 1.0 - bit_ops / max(base * WBITS, 1)
    extra_unique = 0.0
    unique_nz = 0
    n_nz_rows = int(S.any(dim=1).sum().item())
    if 0 < n_nz_rows <= 250000:
        packed = S[S.any(dim=1)].to(torch.uint8).cpu()
        u = torch.unique(packed, dim=0)
        unique_nz = int(u.shape[0])
        extra_unique = 1.0 - unique_nz / n_nz_rows
    density = float(bit_ops / max(base * WBITS, 1))
    return {
        'N': int(N), 'Cin': int(Cin), 'Cout': int(Cout), 'nnz_S': nnz,
        'elem_nnz': nnz / max(N * Cin, 1),
        'zero_row_frac': float((~S.any(dim=1)).float().mean()),
        'n_nz_rows': n_nz_rows, 'unique_nz_rows': unique_nz,
        'baseline_aac_word_adds': base,
        'extra_dual': extra_dual,
        'extra_dual_q8': extra_dual_q8,
        'extra_unique': extra_unique,
        'bit_ops': bit_ops,
        'W_bit_density': density,
        'extra_bitwave_vs_word_add_cycles': extra_bitwave_vs_word,
        'extra_bitwave_vs_7serial': extra_bitwave_vs_7serial,
        'inspect_tax_runtime': TAX_RUNTIME,
    }


def prefix_nnz_extra(S, keep_idx):
    """Fraction of AAC word-adds removed by keeping only keep_idx input channels."""
    S = (S.detach() != 0)
    nnz = int(S.sum().item())
    if nnz == 0:
        return 0.0
    kept = int(S[:, keep_idx].sum().item())
    return 1.0 - kept / nnz


def k_keep_for_nnz_extra(S, order, target=KEEP):
    """Largest number of mass-ordered channels to keep while nnz-extra >= target."""
    S = (S.detach() != 0)
    nnz = float(S.sum().item())
    Cin = S.shape[1]
    if nnz <= 0:
        return Cin, 0.0
    best_k, best_e = Cin, 0.0
    for k in range(1, Cin + 1):
        extra = 1.0 - float(S[:, order[:k]].sum().item()) / nnz
        if extra >= target:
            best_k, best_e = k, extra
        else:
            break
    return int(best_k), float(best_e)


def sna_pea_extra(S, W, y_th, max_tokens=512):
    """skipped/total word-adds over ALL Cout. Sampled tokens. y_th is Cout in Y-space."""
    S = (S.detach() != 0)
    W = W.detach().float()
    N, Cin = S.shape
    Cout = W.shape[0]
    step = max(1, N // max_tokens)
    Sb = S[::step]
    y_th = y_th.detach().float().reshape(-1).to(device=W.device, dtype=W.dtype)
    skipped = 0.0
    total = 0.0
    for o in range(Cout):
        order = torch.argsort(W[o].abs(), descending=True)
        Ss = Sb[:, order].float()
        wv = W[o, order]
        av = wv.abs()
        contrib = Ss * wv
        psum = contrib.cumsum(1)
        used_abs = (Ss * av).cumsum(1)
        tot_abs = used_abs[:, -1:]
        psum_before = psum - contrib
        rest_before = tot_abs - (used_abs - Ss * av)
        th = y_th[o]
        cannot = (psum_before + rest_before) < th
        must = (psum_before - rest_before) >= th
        decided = cannot | must
        has = decided.any(dim=1)
        first = decided.long().argmax(dim=1)
        first = torch.where(has, first, torch.full_like(first, Cin))
        rng = torch.arange(Cin, device=Ss.device)
        after = (rng.unsqueeze(0) >= first.unsqueeze(1)) & (Ss > 0)
        skipped += float(after.sum().item())
        total += float(Ss.sum().item())
    extra = skipped / max(total, 1.0)
    return extra, int(Sb.shape[0]), skipped, total


def psn_zero_t_extra(S_tpc):
    """Skip PSN mix columns where the token's raw Y_t is 0 because S_t is all-zero.

    S_tpc: (T, P, Cin) bool. Baseline leftover PSN = n_active * T * T * (hidden later).
    extra = 1 - mean(nz_timesteps)/T on spatially-active tokens.
    """
    S = S_tpc.detach() != 0
    occ = S.any(dim=-1)
    active = occ.any(dim=0)
    if int(active.sum().item()) == 0:
        return 0.0
    return float(1.0 - occ[:, active].float().mean().item())


def rate_matched_theta(U, shared_theta):
    """Per-channel θ so fire rate under θ_h matches fire rate under shared θ.

    U: (..., H). Returns theta (H,), fire_shared (H,).
    Official AT-LIF still o_h = θ_h H(m_h - θ_h); absorb θ_h into next W column.
    """
    u = U.detach().float().reshape(-1, U.shape[-1])
    H = u.shape[1]
    shared = float(shared_theta)
    fire = (u >= shared).float().mean(0)
    theta = torch.empty(H, dtype=u.dtype, device=u.device)
    for h in range(H):
        r = float(fire[h].item())
        col = u[:, h]
        if r <= 0.0:
            theta[h] = col.max() + 1.0
        elif r >= 1.0:
            theta[h] = col.min() - 1.0
        else:
            theta[h] = torch.quantile(col, 1.0 - r)
    return theta, fire


def scale_theta_for_rate(U, fire_target):
    """θ_h = quantile(U_h, 1 - fire_target_h). fire_target: scalar or (H,)."""
    u = U.detach().float().reshape(-1, U.shape[-1])
    H = u.shape[1]
    if not torch.is_tensor(fire_target):
        fire_target = torch.full((H,), float(fire_target), dtype=u.dtype, device=u.device)
    else:
        fire_target = fire_target.detach().float().reshape(-1).to(device=u.device)
    theta = torch.empty(H, dtype=u.dtype, device=u.device)
    for h in range(H):
        r = float(fire_target[h].clamp(0, 1).item())
        col = u[:, h]
        if r <= 0.0:
            theta[h] = col.max() + 1.0
        elif r >= 1.0:
            theta[h] = col.min() - 1.0
        else:
            theta[h] = torch.quantile(col, 1.0 - r)
    return theta


def block_pool_theta(theta_h, block, how='mean'):
    """Share one θ per block of channels (hardware tile broadcast)."""
    t = theta_h.detach().float().reshape(-1)
    H = t.numel()
    block = int(block)
    if block <= 1:
        return t.clone()
    out = t.clone()
    for i in range(0, H, block):
        sl = t[i:i + block]
        if how == 'median':
            v = sl.median()
        else:
            v = sl.mean()
        out[i:i + block] = v
    return out


def psn_lowrank_mac_extra(T, rank):
    """Compile-time factor extra vs dense T×T. 0 if 2*T*r >= T*T (no reduction)."""
    dense = T * T
    fact = 2 * T * rank
    if fact >= dense or rank <= 0:
        return 0.0
    return 1.0 - fact / dense


def psn_scrooge_inspect_tax(T, bound='l1_maxabs'):
    """Inspect tax vs leftover T×T mix MACs.

    l1_maxabs: one max|Y| over T (shared by T outputs) → T / T² = 1/T.
    tot_oracle: rest=Σ|A||Y| is itself T muls per output = T×T leftover → tax 1.
    Do not charge tot_oracle as 1/T: that undercharges by a factor of T.
    """
    if T <= 0:
        return 1.0
    if bound == 'tot_oracle':
        return 1.0
    if bound == 'l1_maxabs':
        return 1.0 / T
    raise ValueError(bound)


def psn_scrooge_extra(Y, A, tau, bound='l1_maxabs'):
    """Certificate early-stop on leftover PSN mix.

    Y: (T, N, H) PSN inputs. A: (T, T). tau: (T, H). A@Y >= tau.
    Baseline remaining = N * H * T * T mix MACs.
    extra = skipped mix terms / baseline.

    bound='l1_maxabs': rest = ||A_rest||_1 * max|Y|. Does not MAC skipped Y.
    bound='tot_oracle': rest = Σ_rest |A||Y| — that tot is leftover work (KILL).
    """
    if bound not in ('l1_maxabs', 'tot_oracle'):
        raise ValueError(bound)
    Y = Y.detach().float()
    A = A.detach().float()
    tau = tau.detach().float()
    T, N, H = Y.shape
    total = float(T * N * H * T)
    if total <= 0:
        return 0.0
    ymax = Y.abs().amax(dim=0)
    skipped = 0.0
    for t in range(T):
        order = torch.argsort(A[t].abs(), descending=True)
        Ys = Y.index_select(0, order)
        a = A[t].index_select(0, order)
        a_abs = a.abs()
        th = tau[t].reshape(1, H)
        if bound == 'l1_maxabs':
            rest0 = a_abs.sum() * ymax
        else:
            rest0 = (Ys.abs() * a_abs[:, None, None]).sum(0)
        cannot0 = rest0 < th
        must0 = (-rest0) >= th
        decided0 = cannot0 | must0
        contrib = Ys * a[:, None, None]
        psum = contrib.cumsum(0)
        if bound == 'l1_maxabs':
            suffix_after = a_abs.flip(0).cumsum(0).flip(0) - a_abs
            rest = suffix_after[:, None, None] * ymax
        else:
            used = (Ys.abs() * a_abs[:, None, None]).cumsum(0)
            rest = used[-1].unsqueeze(0) - used
        cannot = (psum + rest) < th.unsqueeze(0)
        must = (psum - rest) >= th.unsqueeze(0)
        decided = cannot | must
        has = decided.any(dim=0)
        first = decided.long().argmax(dim=0)
        used_terms = torch.where(has, first + 1, torch.full_like(first, T))
        used_terms = torch.where(decided0, torch.zeros_like(used_terms), used_terms)
        skipped += float((T - used_terms).clamp(min=0).sum().item())
    return skipped / total


def support_code_compute_extra(n_nz_rows, K):
    """LUT compute extra vs one matvec per nz row, ignoring encode tax."""
    if n_nz_rows <= 0:
        return 0.0
    return 1.0 - min(K, n_nz_rows) / n_nz_rows


def lut_gemm_group_extra(S, group=4):
    """LUT-GEMM/Platinum: one table-add per nonempty G-bit group vs nnz word-adds.

    Fair leftover: each nonempty group costs one word-add (table lookup) per Cout,
    baseline is nnz(S)*Cout. extra = 1 - n_nonempty_groups / nnz(S).
    Lossless for binary S. Inspect tax = 1/group if runtime bitmap, 0 if packed.
    """
    S = (S.detach() != 0)
    N, Cin = S.shape
    pad = (group - (Cin % group)) % group
    if pad:
        S = torch.nn.functional.pad(S, (0, pad))
    G = S.shape[1] // group
    groups = S.reshape(N, G, group).any(dim=-1)
    n_grp = int(groups.sum().item())
    nnz = int(S[:, :Cin].sum().item()) if pad else int(S.sum().item())
    if nnz <= 0:
        return 0.0, 0, 0
    extra = 1.0 - n_grp / nnz
    return extra, n_grp, nnz


def support_code_encode_tax(n_nz_rows, K, cin, baseline_word_adds):
    """Hamming-distance encode tax in word-ops / leftover word-adds.

    Charge one word-op per 32-bit popcount limb per code per nz row.
    """
    if baseline_word_adds <= 0:
        return 1.0
    limbs = (cin + 31) // 32
    tax_ops = n_nz_rows * K * limbs
    return tax_ops / baseline_word_adds


def emit_copy(mech, paper, op, extra, tax, lossless, note='', baseline='nnz(S)*Cout word-adds after skip S==0'):
    extra_tax = extra - tax
    return {
        'mechanism': mech,
        'paper': paper,
        'target_op': op,
        'fair_baseline': baseline,
        'extra_save': extra,
        'inspect_tax': tax,
        'extra_save_after_tax': extra_tax,
        'keep_kill': keep_kill(extra_tax),
        'lossless': lossless,
        'denominator': 'post-skip-zero remaining of ' + op,
        'note': note,
    }
