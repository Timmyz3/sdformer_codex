"""Leftover PSN / certificate / support-code extra on existing stage0 traces.

Reads bn_state/trace_s{0,10}_stage0.npz only. No network, no GPU AEE.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BN = HERE.parents[1] / 'bn_state'
OUT_JSON = Path('/tmp/grok-goal-1aea184f5e81/implementer/psn_probe.json')
OUT_MD = Path('/tmp/grok-goal-1aea184f5e81/implementer/psn_probe.md')
KEEP_BAR = 0.15
INSPECT_TAX = 1.0 / 32.0
N_SPATIAL_SAMPLE = 512
K_LIST = (16, 32, 64, 128, 256)
THETA_TRACE = 1.0
EPS = 1e-12
POPCNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def py(x):
    if isinstance(x, dict):
        return {k: py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [py(v) for v in x]
    if isinstance(x, np.ndarray):
        return py(x.tolist())
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def keep_label(extra_after_tax):
    return 'KEEP_CANDIDATE' if extra_after_tax >= KEEP_BAR else 'KILL'


def load_trace(path):
    z = np.load(path)
    S = np.unpackbits(z['source_packed'], axis=1, bitorder='little')[:, :96].astype(np.uint8)
    return dict(
        S=S,
        W=z['W'].astype(np.float64),
        A=z['A'].astype(np.float64),
        G=z['G'].astype(np.float64),
        k=z['k'].astype(np.float64),
        gamma=z['gamma'].astype(np.float64),
        beta=z['beta'].astype(np.float64),
        bias=z['bias'].astype(np.float64).reshape(-1),
        center=z['center'].astype(np.float64).reshape(-1),
        source_packed=z['source_packed'],
    )


def moments_from_gram(W, k, G, N):
    mu = (W @ k) / N
    var = np.maximum(np.sum((W @ G) * W, axis=1) / N - mu * mu, 0.0)
    return mu, var


def bn_folded_u_th(mu, var, gamma, beta, A, bias, center, theta):
    """Threshold on U = A @ Y (bias/center/BN folded). gamma>0 => fire iff U >= u_th.

    From build_traces.py:
      R = A.sum(1)
      sigma = sqrt(var+1e-5)
      direction = sign(gamma)
      tau_V = (mu*R + sigma/gamma * (theta+center-bias-beta*R)) * direction
      fire iff (U * direction) >= tau_V
    """
    T = A.shape[0]
    R = A.sum(axis=1)
    sigma = np.sqrt(var + 1e-5)
    direction = np.sign(gamma)
    direction = np.where(direction == 0, 1.0, direction)
    u_th = mu[None, :] * R[:, None] + (sigma / gamma)[None, :] * (
        theta + center[:, None] - bias[:, None] - beta[None, :] * R[:, None]
    )
    tau_V = u_th * direction[None, :]
    y_th_bn = mu + (sigma / gamma) * (theta - beta)
    y_th_tau = np.divide(u_th, R[:, None], out=np.full_like(u_th, np.nan), where=np.abs(R)[:, None] > 1e-8)
    return dict(
        R=R, sigma=sigma, direction=direction, u_th=u_th, tau_V=tau_V,
        y_th_bn=y_th_bn, y_th_tau=y_th_tau,
        formula=(
            'U = A @ Y  (FC1 bias=False, Y=S@W.T, theta_source=1). '
            'sigma=sqrt(var+1e-5); R=A.sum(1); direction=sign(gamma). '
            'u_th[t,h] = mu[h]*R[t] + (sigma[h]/gamma[h])*(theta + center[t] - bias[t] - beta[h]*R[t]). '
            'fire iff direction[h]*U[t,h] >= u_th[t,h]*direction[h]. '
            'theta=1.0 from numeric_s*_stage0.json (checkpoint sn2 thresh), not 0.1. '
            'y_th_bn[h] = mu + (sigma/gamma)*(theta-beta). '
            'y_th_tau[t,h] = u_th[t,h]/R[t] (constant-Y inversion of the same tau; R may be negative).'
        ),
        theta=theta,
        T=T,
    )


def svd_lowrank(A):
    T = A.shape[0]
    _, s, _ = np.linalg.svd(A, full_matrices=False)
    U, _, Vt = np.linalg.svd(A, full_matrices=False)
    fn = np.linalg.norm(A, 'fro')
    rows = []
    for r in range(1, 6):
        Ar = (U[:, :r] * s[:r]) @ Vt[:r]
        rel = float(np.linalg.norm(A - Ar, 'fro') / max(fn, EPS))
        extra = 1.0 - (2.0 * T * r) / (T * T)
        rows.append(dict(
            r=r, extra_mac=extra, extra_save_after_tax=extra, inspect_tax=0.0,
            rel_frobenius=rel, singular_value_tail=float(s[r:].sum() if r < len(s) else 0.0),
            keep_kill=keep_label(extra),
        ))
    return dict(singular_values=s.tolist(), ranks=rows, A_rank=int(np.linalg.matrix_rank(A)))


def scrooge_psn(Y, A, u_th, direction):
    """Y (n,T,H), order |A[t,j]| desc, skip rest when U bound cannot cross u_th."""
    n, T, H = Y.shape
    skipped = 0
    skipped_nz = 0
    leftover_nz = 0
    decided_out = 0
    processed_mults = 0
    first_hist = np.zeros(T + 1, dtype=np.int64)
    k0_cannot = 0
    k0_must = 0
    cert_err = 0
    Utrue = np.einsum('tj,njh->nth', A, Y)
    fire_true = (Utrue * direction[None, None, :]) >= (u_th[None, :, :] * direction[None, None, :])
    fire_rate = float(fire_true.mean())
    for t in range(T):
        order = np.argsort(-np.abs(A[t]))
        a = A[t, order]
        Yord = Y[:, order, :]
        absY = np.abs(Yord)
        nz = absY > EPS
        leftover_nz += int(nz.sum())
        contrib = a[None, :, None] * Yord
        psum = np.cumsum(contrib, axis=1)
        abs_c = np.abs(a)[None, :, None] * absY
        used = np.cumsum(abs_c, axis=1)
        tot = used[:, -1:, :]
        psum_before = psum - contrib
        rest_before = tot - (used - abs_c)
        ge = direction[None, None, :] >= 0
        th = u_th[t][None, None, :]
        cannot_ge = (psum_before + rest_before) < th
        must_ge = (psum_before - rest_before) >= th
        cannot_le = (psum_before - rest_before) > th
        must_le = (psum_before + rest_before) <= th
        decided = np.where(ge, cannot_ge | must_ge, cannot_le | must_le)
        has = decided.any(axis=1)
        first = np.where(has, decided.argmax(axis=1), T)
        decided_out += int(has.sum())
        k = np.arange(T)[None, :, None]
        skip = k >= first[:, None, :]
        skipped += int(skip.sum())
        skipped_nz += int((skip & nz).sum())
        processed_mults += int((~skip).sum())
        for v, c in zip(*np.unique(first, return_counts=True)):
            first_hist[int(v)] += int(c)
        k0_cannot += int((decided[:, 0] & np.where(ge[:, 0], cannot_ge[:, 0], cannot_le[:, 0])).sum())
        k0_must += int((decided[:, 0] & np.where(ge[:, 0], must_ge[:, 0], must_le[:, 0])).sum())
        fi = np.minimum(first, T - 1)
        ii = np.arange(n)[:, None]
        hh = np.arange(H)[None, :]
        pred_must = np.where(ge[:, 0], must_ge, must_le)[ii, fi, hh]
        pred_cannot = np.where(ge[:, 0], cannot_ge, cannot_le)[ii, fi, hh]
        fire_t = fire_true[:, t, :]
        cert_err += int((has & pred_cannot & fire_t).sum()) + int((has & pred_must & (~fire_t)).sum())
    dense = n * H * T * T
    extra_dense = skipped / max(dense, 1)
    extra_nz = skipped_nz / max(leftover_nz, 1)
    processed = dense - skipped
    bound_tax = 1.0 / T
    return dict(
        n_spatial=n, H=H, T=T,
        dense_macs=dense,
        skipped_A_mults=skipped,
        skipped_nz_A_mults=skipped_nz,
        leftover_nz_A_mults=leftover_nz,
        processed_A_mults=processed_mults,
        decided_outputs=decided_out,
        n_outputs=n * H * T,
        extra_vs_dense=extra_dense,
        extra_vs_leftover_nzY=extra_nz,
        skip_over_total=extra_dense,
        skip_over_processed=(skipped / max(processed, 1)),
        first_hist=first_hist.tolist(),
        k0_cannot=k0_cannot,
        k0_must=k0_must,
        fire_rate=fire_rate,
        certificate_errors=cert_err,
        bound_dot_tax=bound_tax,
        extra_after_bound_tax=extra_dense - bound_tax,
        extra_after_inspect_tax=extra_dense - INSPECT_TAX,
    )


def snapea_fc1(S, W, y_th, ge=True):
    """S (Ns,C) 0/1, W (H,C). extra = skipped/(nnz*H) over ALL H."""
    S = (S != 0).astype(np.float64)
    Ns, C = S.shape
    H = W.shape[0]
    nnz = float(S.sum())
    total = nnz * H
    skipped = 0.0
    y2d = y_th.ndim == 2
    ge_arr = np.asarray(ge)
    ge_vec = ge_arr.ndim == 1
    for h in range(H):
        order = np.argsort(-np.abs(W[h]))
        Ss = S[:, order]
        wv = W[h, order]
        av = np.abs(wv)
        contrib = Ss * wv
        psum = np.cumsum(contrib, axis=1)
        used = np.cumsum(Ss * av, axis=1)
        tot = used[:, -1:]
        psum_before = psum - contrib
        rest_before = tot - (used - Ss * av)
        th = y_th[:, h][:, None] if y2d else y_th[h]
        if ge_vec:
            g = ge_arr[:, None]
            cannot = np.where(g, (psum_before + rest_before) < th, (psum_before - rest_before) > th)
            must = np.where(g, (psum_before - rest_before) >= th, (psum_before + rest_before) <= th)
        elif ge:
            cannot = (psum_before + rest_before) < th
            must = (psum_before - rest_before) >= th
        else:
            cannot = (psum_before - rest_before) > th
            must = (psum_before + rest_before) <= th
        decided = cannot | must
        has = decided.any(axis=1)
        first = np.where(has, decided.argmax(axis=1), C)
        after = (np.arange(C)[None, :] >= first[:, None]) & (Ss > 0)
        skipped += float(after.sum())
    processed = total - skipped
    extra_total = skipped / max(total, 1.0)
    return dict(
        n_tokens=int(Ns), H=int(H),
        nnz_sampled=int(nnz),
        skipped_adds=skipped,
        total_adds=total,
        processed_adds=processed,
        extra_skip_over_total=extra_total,
        extra_skip_over_processed=skipped / max(processed, 1.0),
    )


def hamming_to_codes(rows, codes, chunk=4096):
    n = rows.shape[0]
    out = np.empty(n, dtype=np.int16)
    for lo in range(0, n, chunk):
        sl = rows[lo:lo + chunk]
        xor = np.bitwise_xor(sl[:, None, :], codes[None, :, :])
        dist = POPCNT[xor].sum(axis=2)
        out[lo:lo + chunk] = dist.min(axis=1)
    return out


def probe_one(name, tr, rng):
    S = tr['S']
    W = tr['W']
    A = tr['A']
    N, C = S.shape
    H = W.shape[0]
    T = A.shape[0]
    P = N // T
    ST = S.reshape(T, P, C)
    row_nz = S.any(axis=1)
    spatial_any = ST.any(axis=2)
    active = spatial_any.any(axis=0)
    n_active = int(active.sum())
    nz_count = spatial_any[:, active].sum(axis=0).astype(np.float64) if n_active else np.zeros(0)
    mean_nz_t = float(nz_count.mean()) if n_active else 0.0
    extra_skip_zero_yt = 1.0 - mean_nz_t / T
    dense_psn = n_active * H * T * T
    nnz = int(S.sum())
    n_nz_rows = int(row_nz.sum())
    packed_nz = tr['source_packed'][row_nz]
    unique_nz = int(np.unique(packed_nz, axis=0).shape[0])
    extra_unique = 1.0 - unique_nz / max(n_nz_rows, 1)

    mu, var = moments_from_gram(W, tr['k'], tr['G'], N)
    fold = bn_folded_u_th(mu, var, tr['gamma'], tr['beta'], A, tr['bias'], tr['center'], THETA_TRACE)
    lowrank = svd_lowrank(A)

    idx_pool = np.flatnonzero(active)
    n_s = int(min(N_SPATIAL_SAMPLE, idx_pool.size))
    idx = np.sort(rng.choice(idx_pool, size=n_s, replace=False))
    Ss = ST[:, idx, :].astype(np.float64)
    Y = np.tensordot(Ss, W.T, axes=([2], [0]))  # (T,n,H)
    Yn = np.transpose(Y, (1, 0, 2))
    sc = scrooge_psn(Yn, A, fold['u_th'], fold['direction'])

    x = tr['gamma'][None, None, :] * (Y - mu[None, None, :]) / fold['sigma'][None, None, :] + tr['beta'][None, None, :]
    ordered = np.tensordot(A, x, axes=([1], [0])) + tr['bias'][:, None, None] - tr['center'][:, None, None]
    U = np.tensordot(A, Y, axes=([1], [0]))
    fire_u = (U * fold['direction'][None, None, :]) >= fold['tau_V'][:, None, :]
    fire_ref = ordered >= THETA_TRACE
    tau_match = float(np.mean(fire_u == fire_ref))

    S_rows = Ss.reshape(T * n_s, C)
    times = np.repeat(np.arange(T), n_s)
    y_th_tau_tok = fold['y_th_tau'][times]
    ge_tok = fold['R'][times] > 0
    sna_tau = snapea_fc1(S_rows, W, y_th_tau_tok, ge=ge_tok)
    sna_bn = snapea_fc1(S_rows, W, fold['y_th_bn'], ge=True)
    fold01 = bn_folded_u_th(mu, var, tr['gamma'], tr['beta'], A, tr['bias'], tr['center'], 0.1)
    sna_bn_01 = snapea_fc1(S_rows, W, fold01['y_th_bn'], ge=True)

    sample_nz = np.flatnonzero(row_nz)
    n_ham = int(min(4096, sample_nz.size))
    ham_idx = rng.choice(sample_nz, size=n_ham, replace=False)
    ham_rows = tr['source_packed'][ham_idx]
    uniq_all = np.unique(packed_nz, axis=0)
    codebook = {}
    for K in K_LIST:
        extra_lut = 1.0 - K / max(n_nz_rows, 1)
        enc_pop = n_nz_rows * K
        enc_bits = enc_pop * 96
        word_adds = nnz * H
        tax_pop = enc_pop / max(word_adds, 1)
        tax_bits = enc_bits / max(word_adds, 1)
        tax_3word = (enc_pop * 3) / max(word_adds, 1)
        cidx = rng.choice(uniq_all.shape[0], size=min(K, uniq_all.shape[0]), replace=False)
        codes = uniq_all[cidx]
        dist = hamming_to_codes(ham_rows, codes)
        codebook[str(K)] = dict(
            K=K,
            extra_lut_ignore_tax=extra_lut,
            encoding_popcounts=enc_pop,
            encoding_bitops=enc_bits,
            word_adds_nnz_H=word_adds,
            tax_popcount_eq_word=tax_pop,
            tax_popcount_as_3word=tax_3word,
            tax_bitops_eq_word=tax_bits,
            extra_after_tax_popcount=extra_lut - tax_pop,
            extra_after_tax_3word=extra_lut - tax_3word,
            extra_after_tax_bitops=extra_lut - tax_bits,
            sample_mean_min_hamming=float(dist.mean()),
            sample_exact_hit=float(np.mean(dist == 0)),
            codebook='random unique-row subsample',
        )

    y_nz_frac = float((np.abs(Yn) > EPS).mean())
    return dict(
        trace=name,
        N=N, P=P, C=C, H=H, T=T,
        nnz=nnz, n_nz_rows=n_nz_rows, unique_nz_rows=unique_nz,
        n_active_spatial=n_active,
        mean_nz_timesteps=mean_nz_t,
        extra_skip_zero_YT=extra_skip_zero_yt,
        dense_psn_macs=dense_psn,
        extra_unique=extra_unique,
        elem_nnz=nnz / max(N * C, 1),
        A_rank=lowrank['A_rank'],
        A_nonzero=int(np.count_nonzero(A)),
        W_true_zeros=int(np.count_nonzero(np.abs(W) < 1e-12)),
        gamma_min=float(tr['gamma'].min()),
        gamma_max=float(tr['gamma'].max()),
        mu_mean=float(mu.mean()),
        var_mean=float(var.mean()),
        R=fold['R'].tolist(),
        tau_formula=fold['formula'],
        theta=THETA_TRACE,
        tau_vs_bn_psn_match=tau_match,
        sample_spatial=n_s,
        sample_y_nonzero_frac=y_nz_frac,
        lowrank=lowrank,
        scrooge=sc,
        snapea_tau_mapped=sna_tau,
        snapea_bn_theta1=sna_bn,
        snapea_bn_theta01_sensitivity=sna_bn_01,
        codebook=codebook,
        y_th_bn_mean=float(fold['y_th_bn'].mean()),
        y_th_tau_mean=float(np.nanmean(fold['y_th_tau'])),
    )


def mechanism_rows(agg):
    rows = []
    def add(mech, paper, extra, tax, lossless, baseline, note, extra_after=None):
        after = extra - tax if extra_after is None else extra_after
        rows.append(dict(
            mechanism=mech, paper=paper,
            extra_save=extra, inspect_tax=tax, extra_save_after_tax=after,
            keep_kill=keep_label(after), lossless=lossless,
            fair_baseline=baseline, note=note,
        ))

    base_psn = (
        'leftover dense PSN mix = n_active * H * T * T on spatial tokens with at least one spike across T '
        '(after skip fully-zero spatial; FC1 leftover skip-S=0 is a different denominator)'
    )
    add('skip_zero_Y_T', 'skip zero PSN timesteps on leftover mix',
        agg['extra_skip_zero_YT'], 0.0, True, base_psn,
        'extra = 1 - mean(nz_timesteps)/T on active spatial tokens. Source-row any-spike as Y_T=0 iff S_t=0 (W has no true zeros).')
    for r in agg['lowrank_ranks']:
        add(f'lowrank_PSN_r{r["r"]}', 'compile-time SVD of A, tax=0',
            r['extra_mac'], 0.0, False, base_psn,
            f'extra_mac=1-2*T*r/(T*T); rel Frobenius={r["rel_frobenius"]:.4f}; A is rank 10 so this is approximate, not ep34 lossless.')
    sc = agg['scrooge']
    add('scrooge_certificate_PSN_mix', 'Scrooge DATE26 / SnaPEA-style |A| prefix vs BN-folded tau on A@Y',
        sc['extra_vs_dense'], INSPECT_TAX, True,
        'leftover dense PSN mix n*H*T*T on sampled active spatial, all H=384; extra=skipped_A_mults/(n*H*T*T)',
        f'skip/total={sc["extra_vs_dense"]:.6f}; skip/processed={sc["skip_over_processed"]:.6f}; '
        f'extra vs leftover nzY={sc["extra_vs_leftover_nzY"]:.6f}; '
        f'decided_outputs={sc["decided_outputs"]}/{sc["n_outputs"]}; fire_rate={sc["fire_rate"]:.4f}; '
        f'certificate_errors={sc["certificate_errors"]}. '
        f'Primary tax 1/32 inspect; alternate bound-dot tax 1/T={sc["bound_dot_tax"]:.2f} '
        f'gives extra_after_tax={sc["extra_after_bound_tax"]:.4f}. |A| order compile-time. '
        f'Most skip is k=0 cannot-fire (U typically below tau).')
    add('support_exact_unique_96bit', 'Prosperity/Phi identical 96-bit row merge on leftover FC1',
        agg['extra_unique'], INSPECT_TAX, True,
        'leftover FC1 row-matvecs n_nz_rows after skip S=0; extra=1-unique_nz_rows/n_nz_rows',
        'teacher exact dictionary on this ep34 trace. Do not treat forced_code 37.9% as this student; that is a different trained student. FC1+PSN chain teacher LUT16 was ~4.6%.')
    for K, cb in agg['codebook'].items():
        add(f'support_codebook_K{K}', f'Hamming random-subsample codebook K={K} (k-medoids extra formula identical)',
            cb['extra_lut_ignore_tax'], cb['tax_popcount_eq_word'], False,
            'leftover FC1 n_nz_rows after skip S=0; LUT compute extra ignoring encoding tax = 1-K/n_nz_rows; encoding tax = 96-bit popcount*K per nz row vs word-adds nnz*384',
            f'lossy. sample mean min Hamming={cb["sample_mean_min_hamming"]:.2f}, exact hit={cb["sample_exact_hit"]:.4f}. '
            f'tax_popcount_eq_word={cb["tax_popcount_eq_word"]:.4f}, tax_3word={cb["tax_popcount_as_3word"]:.4f}, '
            f'tax_bitops_eq_word={cb["tax_bitops_eq_word"]:.4f}. extra_after_tax_3word={cb["extra_after_tax_3word"]:.4f}. '
            f'NOT forced_code 37.9% and NOT ep34 lossless.',
            extra_after=cb['extra_after_tax_popcount'])
    sna = agg['snapea_tau_mapped']
    add('frozenBN_SnaPEA_FC1_tau_mapped', 'CGNet MICRO19 / SnaPEA ISCA18 |W| prefix vs y_th from BN-folded tau mapped to Y-space',
        sna['extra_skip_over_total'], INSPECT_TAX, True,
        'leftover FC1 word-adds nnz_sampled*H after skip S=0, ALL H=384; extra=skipped_adds/(nnz_sampled*H) (skip/total, not skip/processed)',
        f'skip/total={sna["extra_skip_over_total"]:.6f}; skip/processed={sna["extra_skip_over_processed"]:.6f}; '
        f'y_th[t,h]=u_th[t,h]/R[t]; comparison flips if R[t]<0. Frozen Gram moments, theta=1.0. Not applied to a forward AEE.')
    sna1 = agg['snapea_bn_theta1']
    add('frozenBN_SnaPEA_FC1_pointwise_theta1', 'SnaPEA |W| prefix vs BN-only y_th (ignore PSN mix), theta=1.0',
        sna1['extra_skip_over_total'], INSPECT_TAX, True,
        'leftover FC1 word-adds nnz_sampled*H after skip S=0, ALL H=384; extra=skipped_adds/(nnz_sampled*H)',
        f'skip/total={sna1["extra_skip_over_total"]:.6f}; skip/processed={sna1["extra_skip_over_processed"]:.6f}; '
        f'y_th=mu+(sigma/gamma)*(theta-beta), theta=1.0 from traces. Compare to legacy v_th=0.1 proxy.')
    sna01 = agg['snapea_bn_theta01']
    add('frozenBN_SnaPEA_FC1_pointwise_theta01_sensitivity', 'same SnaPEA with theta=0.1 (legacy remaining_copies v_th, NOT this checkpoint)',
        sna01['extra_skip_over_total'], INSPECT_TAX, True,
        'leftover FC1 word-adds nnz_sampled*H after skip S=0, ALL H=384; extra=skipped_adds/(nnz_sampled*H)',
        f'sensitivity only. skip/total={sna01["extra_skip_over_total"]:.6f}. Do not replace theta=1.0.')
    return rows


def mean_pair(a, b, keys):
    out = {}
    for k in keys:
        out[k] = 0.5 * (a[k] + b[k])
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    traces = []
    for sid in (0, 10):
        name = f's{sid}_stage0'
        print('TRACE', name, flush=True)
        traces.append(probe_one(name, load_trace(BN / f'trace_s{sid}_stage0.npz'), rng))

    def avg_scrooge():
        keys = ['extra_vs_dense', 'extra_vs_leftover_nzY', 'skip_over_processed',
                'skipped_A_mults', 'dense_macs', 'decided_outputs', 'n_outputs',
                'leftover_nz_A_mults', 'skipped_nz_A_mults', 'k0_cannot', 'k0_must',
                'fire_rate', 'certificate_errors', 'bound_dot_tax',
                'extra_after_bound_tax', 'extra_after_inspect_tax']
        s0, s1 = traces[0]['scrooge'], traces[1]['scrooge']
        out = {k: 0.5 * (s0[k] + s1[k]) for k in keys}
        out['first_hist'] = [0.5 * (a + b) for a, b in zip(s0['first_hist'], s1['first_hist'])]
        return out

    def avg_sna(field):
        keys = ['extra_skip_over_total', 'extra_skip_over_processed', 'skipped_adds', 'total_adds', 'nnz_sampled', 'n_tokens']
        return {k: 0.5 * (traces[0][field][k] + traces[1][field][k]) for k in keys}

    ranks = []
    for i, r in enumerate(traces[0]['lowrank']['ranks']):
        r1 = traces[1]['lowrank']['ranks'][i]
        ranks.append(dict(
            r=r['r'], extra_mac=r['extra_mac'], rel_frobenius=0.5 * (r['rel_frobenius'] + r1['rel_frobenius']),
        ))
    codebook = {}
    for K in K_LIST:
        k = str(K)
        c0, c1 = traces[0]['codebook'][k], traces[1]['codebook'][k]
        codebook[k] = {kk: 0.5 * (c0[kk] + c1[kk]) if isinstance(c0[kk], (int, float)) else c0[kk]
                       for kk in c0 if kk != 'codebook'}
        codebook[k]['codebook'] = c0['codebook']
        codebook[k]['K'] = K

    agg = dict(
        extra_skip_zero_YT=0.5 * (traces[0]['extra_skip_zero_YT'] + traces[1]['extra_skip_zero_YT']),
        extra_unique=0.5 * (traces[0]['extra_unique'] + traces[1]['extra_unique']),
        n_active_spatial=0.5 * (traces[0]['n_active_spatial'] + traces[1]['n_active_spatial']),
        mean_nz_timesteps=0.5 * (traces[0]['mean_nz_timesteps'] + traces[1]['mean_nz_timesteps']),
        dense_psn_macs=0.5 * (traces[0]['dense_psn_macs'] + traces[1]['dense_psn_macs']),
        n_nz_rows=0.5 * (traces[0]['n_nz_rows'] + traces[1]['n_nz_rows']),
        unique_nz_rows=0.5 * (traces[0]['unique_nz_rows'] + traces[1]['unique_nz_rows']),
        nnz=0.5 * (traces[0]['nnz'] + traces[1]['nnz']),
        lowrank_ranks=ranks,
        scrooge=avg_scrooge(),
        snapea_tau_mapped=avg_sna('snapea_tau_mapped'),
        snapea_bn_theta1=avg_sna('snapea_bn_theta1'),
        snapea_bn_theta01=avg_sna('snapea_bn_theta01_sensitivity'),
        codebook=codebook,
        tau_vs_bn_psn_match=0.5 * (traces[0]['tau_vs_bn_psn_match'] + traces[1]['tau_vs_bn_psn_match']),
    )
    mechs = mechanism_rows(agg)
    keepers = [m for m in mechs if m['keep_kill'] == 'KEEP_CANDIDATE']
    out = dict(
        scope=(
            'Leftover extra-save on existing bn_state stage0 traces (sample0, sample10). '
            'No GPU AEE, no valid825, no RTL/PPA, no claim that forced_code 37.9% is ep34 teacher lossless.'
        ),
        no_gpu_aee_claim=True,
        KEEP_threshold=KEEP_BAR,
        inspect_tax_runtime=INSPECT_TAX,
        theta_used=THETA_TRACE,
        theta_note='checkpoint sn2/output_theta=1.0 in numeric_s*_stage0.json; 0.1 is only a SnaPEA sensitivity.',
        tau_formula=traces[0]['tau_formula'],
        unpack='S=unpackbits(source_packed)[:, :96]; W(384,96); A(10,10); Y=S@W.T; FC1 bias=False; N=192000=T10*P19200; C=96; H=384',
        support_code_disclaimer=(
            'forced_code 37.9% is a DIFFERENT trained student. Teacher exact dictionary on FC1+PSN chain is ~4.6% '
            '(support_service_notes.md). Unique-row extra on these traces is the lossless support-code number.'
        ),
        traces=traces,
        aggregate=agg,
        mechanisms=mechs,
        KEEP_CANDIDATE=[m['mechanism'] for m in keepers],
        wall_seconds=time.time() - t0,
    )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(py(out), indent=2) + '\n')
    md = render_md(out)
    OUT_MD.write_text(md)
    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / 'psn_probe.json').write_text(json.dumps(py(out), indent=2) + '\n')
    print('WROTE', OUT_JSON, OUT_MD, flush=True)
    print('KEEP_CANDIDATE', out['KEEP_CANDIDATE'], flush=True)
    for m in mechs:
        print(f"{m['keep_kill']:16s} {m['mechanism']:44s} extra={m['extra_save']:.4f} after_tax={m['extra_save_after_tax']:.4f}", flush=True)


def render_md(out):
    agg = out['aggregate']
    lines = [
        '# PSN / certificate / support-code leftover extra (stage0 traces)',
        '',
        'Existing `bn_state/trace_s{0,10}_stage0.npz` only. **No GPU AEE.** '
        'forced_code 37.9% is a different trained student; teacher exact dictionary on the FC1+PSN chain is ~4.6%.',
        '',
        '## Tau (BN-folded, used for Scrooge and Y-space map)',
        '',
        '```',
        out['tau_formula'],
        '```',
        '',
        f"theta = {out['theta_used']} from traces. tau vs BN-then-PSN fire match on samples: {agg['tau_vs_bn_psn_match']:.6f}.",
        '',
        '## 1. PSN baseline leftover + skip-zero-Y-T',
        '',
        f"- n_active (mean two traces) = {agg['n_active_spatial']:.1f} / 19200",
        f"- dense A mix = n_active * H * T * T = {agg['dense_psn_macs']:.0f}",
        f"- mean(nz_timesteps) on active = {agg['mean_nz_timesteps']:.4f} / T=10",
        f"- skip-zero-Y-T extra = 1 - mean(nz_timesteps)/T = {agg['extra_skip_zero_YT']:.6f} (tax=0) → **{keep_label(agg['extra_skip_zero_YT'])}**",
        '',
        '## 2. Low-rank PSN (SVD of A, compile-time tax=0)',
        '',
        '| r | extra_mac=1-2Tr/T² | rel ||A-Ar||_F | after tax | flag |',
        '|---:|---:|---:|---:|---|',
    ]
    for r in agg['lowrank_ranks']:
        lines.append(
            f"| {r['r']} | {r['extra_mac']:.4f} | {r['rel_frobenius']:.4f} | {r['extra_mac']:.4f} | {keep_label(r['extra_mac'])} |"
        )
    sc = agg['scrooge']
    sc_after = sc['extra_vs_dense'] - INSPECT_TAX
    lines += [
        '',
        'A is rank 10; r≤5 is approximate. extra≥0.15 is MAC count only, not lossless.',
        '',
        '## 3. Scrooge / certificate on PSN mix (512 spatial, all H=384)',
        '',
        f"- extra vs dense T×T = {sc['extra_vs_dense']:.6f}",
        f"- extra vs leftover nzY = {sc['extra_vs_leftover_nzY']:.6f}",
        f"- skip/processed = {sc['skip_over_processed']:.6f}",
        f"- decided outputs = {sc['decided_outputs']:.1f} / {sc['n_outputs']:.0f}",
        f"- fire rate = {sc['fire_rate']:.4f}; k0 cannot/must = {sc['k0_cannot']:.0f}/{sc['k0_must']:.0f}; certificate errors = {sc['certificate_errors']:.0f}",
        f"- after 1/32 inspect tax = {sc_after:.6f} → **{keep_label(sc_after)}**; after 1/T bound-dot tax = {sc['extra_after_bound_tax']:.6f} → **{keep_label(sc['extra_after_bound_tax'])}**",
        '',
        'Fair baseline: leftover dense PSN mix n*H*T*T on sampled active spatial tokens; extra = skipped_A_mults / (n*H*T*T).',
        '',
        '## 4. Support-code (exact unique + Hamming codebook)',
        '',
        f"- n_nz_rows = {agg['n_nz_rows']:.1f}, unique = {agg['unique_nz_rows']:.1f}",
        f"- exact unique extra = {agg['extra_unique']:.6f}; after 1/32 tax = {agg['extra_unique']-INSPECT_TAX:.6f} → **{keep_label(agg['extra_unique']-INSPECT_TAX)}**",
        '',
        '| K | extra=1-K/n_nz (ignore tax) | tax popcount=1 word | extra after that tax | tax 96-bit=word | mean min Hamming | exact hit |',
        '|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for K in K_LIST:
        cb = agg['codebook'][str(K)]
        lines.append(
            f"| {K} | {cb['extra_lut_ignore_tax']:.6f} | {cb['tax_popcount_eq_word']:.4f} | "
            f"{cb['extra_after_tax_popcount']:.4f} | {cb['tax_bitops_eq_word']:.3f} | "
            f"{cb['sample_mean_min_hamming']:.2f} | {cb['sample_exact_hit']:.4f} |"
        )
    sna = agg['snapea_tau_mapped']
    sna1 = agg['snapea_bn_theta1']
    sna01 = agg['snapea_bn_theta01']
    lines += [
        '',
        'LUT extra = 1-K/n_nz_rows ignores encoding. Encoding tax = 96-bit popcount × K per nz row vs word-adds nnz×384. '
        'Random unique-row subsample, not a trained forced_code student.',
        '',
        '## 5. Frozen-BN SnaPEA-style on FC1 (sampled tokens, all H=384)',
        '',
        f"- tau-mapped y_th skip/total = {sna['extra_skip_over_total']:.6f}; skip/processed = {sna['extra_skip_over_processed']:.6f}; after tax = {sna['extra_skip_over_total']-INSPECT_TAX:.6f} → **{keep_label(sna['extra_skip_over_total']-INSPECT_TAX)}**",
        f"- BN-only theta=1.0 skip/total = {sna1['extra_skip_over_total']:.6f}; skip/processed = {sna1['extra_skip_over_processed']:.6f}; after tax = {sna1['extra_skip_over_total']-INSPECT_TAX:.6f} → **{keep_label(sna1['extra_skip_over_total']-INSPECT_TAX)}**",
        f"- sensitivity theta=0.1 skip/total = {sna01['extra_skip_over_total']:.6f} (not this checkpoint)",
        '',
        'Fair baseline: leftover FC1 word-adds nnz_sampled×H after skip S=0, all H; extra = skipped_adds / (nnz_sampled×H).',
        '',
        '## KEEP_CANDIDATE (extra≥0.15 after documented tax)',
        '',
    ]
    keepers = [m for m in out['mechanisms'] if m['keep_kill'] == 'KEEP_CANDIDATE']
    if not keepers:
        lines.append('None.')
    else:
        for m in keepers:
            lines += [
                f"### {m['mechanism']}  extra_after_tax={m['extra_save_after_tax']:.4f}",
                '',
                f"**Baseline:** {m['fair_baseline']}",
                '',
                f"{m['note']}",
                '',
            ]
    lines += [
        '## Boundary',
        '',
        '- Not GPU AEE, not valid825, not nts07/tcasii production edits.',
        '- Live BN cannot drop FC1 tails in the real net; frozen-BN SnaPEA is a frozen-stat proxy.',
        '- Receipt: `/tmp/grok-goal-1aea184f5e81/implementer/psn_probe.json`.',
        '',
    ]
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    main()
