"""Small, executable CFMP-style producer/consumer reference (NumPy only).

This concretizes the published ISSCC 2025 Fig.23.2.5, not an author training
recipe or a bank-cycle simulator. The same TC list produces compact Z and
selects TR slices of V. A single original convolution is factored; its outer
BN, temporal neuron and second convolution are not algebraically merged.
"""
from __future__ import annotations

import numpy as np


STRUCTURES = ('shared', 'grouped', 'hybrid', 'shared_compact', 'grouped_compact')


def svd_factors(weight, rank):
    left, singular, right = np.linalg.svd(np.asarray(weight, np.float64), full_matrices=False)
    root = np.sqrt(singular[:rank])
    return left[:, :rank] * root, root[:, None] * right[:rank]


def initialize(weight, structure, pool=96, output_tile=8, latent_tile=4, regions=8):
    """Fixed R96/active48; compact controls have R48/all-active.

    All expanded axes reserve identical U/V slots. Structural V zeros in
    grouped/hybrid are reported, not secretly exchanged for extra U capacity.
    """
    weight = np.asarray(weight, np.float64)  # [K,H], original *unfolded* W
    k, h = weight.shape
    groups = h // output_tile
    if h % output_tile or pool % (2 * groups) or pool % (2 * latent_tile):
        raise ValueError('This fixed comparison needs aligned H8/J4 group widths.')
    if structure not in STRUCTURES:
        raise ValueError(structure)
    compact = structure.endswith('_compact')
    r = pool // 2 if compact else pool
    base = structure.removesuffix('_compact')
    u = np.zeros((k, r), np.float64)
    v = np.zeros((r, h), np.float64)
    connectivity = np.zeros_like(v, dtype=bool)
    if base == 'shared':
        u[:], v[:] = svd_factors(weight, r)
        connectivity[:] = True
    elif base == 'grouped':
        width = r // groups
        for g in range(groups):
            hs, js = slice(g*output_tile, (g+1)*output_tile), slice(g*width, (g+1)*width)
            u[:, js], v[js, hs] = svd_factors(weight[:, hs], width)
            connectivity[js, hs] = True
    else:
        shared = r // 2
        u[:, :shared], v[:shared] = svd_factors(weight, shared)
        connectivity[:shared] = True
        residual = weight - u[:, :shared] @ v[:shared]
        width = (r-shared) // groups
        for g in range(groups):
            hs = slice(g*output_tile, (g+1)*output_tile)
            js = slice(shared+g*width, shared+(g+1)*width)
            u[:, js], v[js, hs] = svd_factors(residual[:, hs], width)
            connectivity[js, hs] = True
    if r % latent_tile:
        raise ValueError('R must be divisible by the latent tile.')
    # Weight-only initialization, before observing train or validation. This
    # is explicitly our recipe. Training can move all permitted factors.
    scalar_energy = np.square(u).sum(0) * np.square(v).sum(1)
    tile_energy = scalar_energy.reshape(-1, latent_tile).sum(1)
    logs = np.log(tile_energy + max(float(tile_energy.max()), 1.) * 1e-12)
    logs = logs-logs.mean()
    logits = np.broadcast_to(logs, (regions, len(logs))).copy()
    active_tiles = r//latent_tile if compact else pool//(2*latent_tile)
    return dict(u=u, v=v, connectivity=connectivity, logits=logits,
                active_tiles=active_tiles, latent_tile=latent_tile,
                structure=structure, reserved_pool=pool)


def hard_masks(logits, keep):
    # Stable ties favor original latent indices, not validation outcomes.
    order = np.argsort(-np.asarray(logits), axis=-1, kind='stable')
    masks = np.zeros_like(logits, dtype=bool)
    np.put_along_axis(masks, order[:, :keep], True, axis=-1)
    return masks


def spatial_regions(group_ids, height=240, width=320, positions=4, regions=8):
    """Eight fixed horizontal bands; every unobserved pixel has a definition."""
    y = np.asarray(group_ids, np.int64) // (width // positions)
    return np.minimum(y * regions // height, regions-1)


def decode_tc(mask, word_bits=8):
    """FMS mask split/flatten/valid-count termination, in increasing TC ID."""
    mask = np.asarray(mask, bool)
    offsets, words = [], 0
    for begin in range(0, len(mask), word_bits):
        bits = sum(int(value) << i for i, value in enumerate(mask[begin:begin+word_bits]))
        words += 1
        valid = bits.bit_count()
        while valid:
            low = bits & -bits
            offsets.append(begin + low.bit_length()-1)
            bits ^= low
            valid -= 1
    return offsets, words


def store_v_column_banks(v, banks=8):
    """Explicit TC-oriented column-major storage; TR needs bank/row slices."""
    flat = np.asarray(v).T.reshape(-1)
    memory = np.zeros((banks, (len(flat)+banks-1)//banks), dtype=flat.dtype)
    addr = np.arange(len(flat))
    memory[addr % banks, addr // banks] = flat
    return memory


def read_tr(memory, js, hs, rank):
    addr = np.asarray(hs)[None, :] * rank + np.asarray(js)[:, None]
    return memory[addr % len(memory), addr // len(memory)]


def sparse_forward(x, u, v, mask_by_group, latent_tile=4, output_tile=8, banks=8):
    """x[G,T,P,K] -> raw Y[G,T,P,H]; no omitted TC dot is evaluated.

    A packet's stored TC offset is exactly its consumer TR row range. This
    reference deliberately exposes compact storage and dense recovery, but
    does not claim one packet, bank read, or matrix multiply is one cycle.
    """
    x, u, v = map(np.asarray, (x, u, v))
    groups, times, positions, _ = x.shape
    r, h = v.shape
    memory = store_v_column_banks(v, banks)
    out = np.zeros((groups, times, positions, h), dtype=np.result_type(x, u, v))
    stats = dict(mask_words=0, decoded_tc=0, compact_z_values=0,
                 u_dense_scalar_products=0, v_dense_scalar_products=0,
                 u_nonzero_scalar_products=0, v_nonzero_scalar_products=0,
                 tr_scalar_addresses=0, packets=0)
    for g in range(groups):
        offsets, mask_words = decode_tc(mask_by_group[g])
        stats['mask_words'] += mask_words
        stats['decoded_tc'] += len(offsets)
        packets = []
        for tc in offsets:
            js = np.arange(tc*latent_tile, min((tc+1)*latent_tile, r))
            # Selecting these columns happens *before* producing Z.
            z = x[g] @ u[:, js]
            packets.append((tc, z))
            stats['packets'] += 1
            stats['compact_z_values'] += z.size
            stats['u_dense_scalar_products'] += times*positions*u[:, js].size
            stats['u_nonzero_scalar_products'] += int(np.einsum(
                'tpk,kj->', x[g].astype(bool).astype(np.int64),
                u[:, js].astype(bool).astype(np.int64)))
        for tc, z in packets:
            js = np.arange(tc*latent_tile, min((tc+1)*latent_tile, r))
            for first in range(0, h, output_tile):
                hs = np.arange(first, min(first+output_tile, h))
                values = read_tr(memory, js, hs, r)
                out[g, ..., hs[0]:hs[-1]+1] += z @ values
                stats['tr_scalar_addresses'] += values.size
                stats['v_dense_scalar_products'] += times*positions*values.size
                stats['v_nonzero_scalar_products'] += int(np.einsum(
                    'tpj,jh->', z.astype(bool).astype(np.int64),
                    values.astype(bool).astype(np.int64)))
    return out, stats


def latent_demands(unresolved, temporal_support, connectivity, mask_by_group, latent_tile=4):
    """Exact dependency transform; no predictor, no child answers are selected.

    unresolved[G,t,H,P] -> needed[G,s,P,J]. E and V structure are static;
    callers must obtain unresolved from their own already-computed values.
    """
    unresolved = np.asarray(unresolved, bool)
    g, _, _, p = unresolved.shape
    r = len(connectivity)
    needed = np.zeros((g, temporal_support.shape[1], p, r), bool)
    for s in range(temporal_support.shape[1]):
        ts = np.flatnonzero(temporal_support[:, s])
        for j in range(r):
            hs = np.flatnonzero(connectivity[j])
            if len(ts) and len(hs):
                needed[:, s, :, j] = unresolved[:, ts][:, :, hs].any((1, 2))
    mask = np.repeat(mask_by_group, latent_tile, axis=1)[:, :r]
    return needed & mask[:, None, None, :]


def storage_summary(params, positions=4, times=10, scalar_bytes=4):
    u, v = params['u'], params['v']
    active = params['active_tiles'] * params['latent_tile']
    return dict(rank_pool=v.shape[0], active_latents_per_region=active,
        actual_factor_slots=int(u.size+v.size),
        nonzero_factor_slots=int(np.count_nonzero(u)+np.count_nonzero(v)),
        allowed_v_slots=int(params['connectivity'].sum()),
        factor_fp32_bytes=int(scalar_bytes*(u.size+v.size)),
        common_expanded_reservation_fp32_bytes=int(scalar_bytes*params['reserved_pool']*(u.shape[0]+v.shape[1])),
        mask_bytes=0 if params['structure'].endswith('_compact') else int((params['logits'].size+7)//8),
        compact_Z_one_time_fp32_bytes=positions*active*scalar_bytes,
        compact_Z_all_T_fp32_bytes=times*positions*active*scalar_bytes,
        dense_Y_all_T_fp32_bytes=times*positions*v.shape[1]*scalar_bytes,
        temporal_state_note='capacity only; no peak liveness, ports, cycles, or equal-2KiB claim')
