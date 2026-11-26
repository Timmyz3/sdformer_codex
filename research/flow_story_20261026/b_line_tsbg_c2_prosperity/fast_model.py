#!/usr/bin/env python3
"""Vectorised re-implementation of official Prosperity run_fc cycle accounting (issue_type=2),
validated layer-by-layer against the unmodified official run (runs_official/). Used to evaluate
TSBG mappings that would be too slow in the official Python loop (tile_M=4 etc.).

Official per (m,n,k) tile:  compute += sum_rows_nonzero max(residual_i, 1)
   residual_i = nnz_i - max{nnz_j : row_j subset of row_i, not (row_j == row_i and j >= i)} if nnz_i>=2 and max>=1
   preprocess += #(nnz_i > 1) + curM // num_popcnt(8)
Layer: compute = max(compute, preprocess); mem_stall = init + max(0, middle - compute).
"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import common as cm

POP16 = np.array([bin(i).count('1') for i in range(1 << 16)], dtype=np.int16)

# name: (N lanes, tile_M, tile_K, row_order, intra_tile_permutation)
CONFIGS = {
    'base_ps':            (96, 64, 16, 'official', None),     # = official matched_ps
    'tsbg_perm_ps':       (96, 64, 16, 'official', 'tsbg'),   # TSBG group-priority issue order inside each tile
    'rand_perm_ps':       (96, 64, 16, 'official', 'rand'),
    'canon_ps':           (96, 64, 16, 'canonical', None),    # TSBG capture token order (t,h,w), tile 64
    'tsbg_b4_canon_ps':   (96, 4, 16, 'canonical', None),     # literal TSBG-B4: aligned quartets of canonical tokens
    'tsbg_b4_official_ps': (96, 4, 16, 'official', None),
}


def tile_costs(a):
    """a: [B, m] uint16 rows of one (m,k) tile each. Returns (cost[B], searched[B])."""
    pop = POP16[a]                                    # [B,m]
    B, m = a.shape
    ai = a[:, :, None]; aj = a[:, None, :]
    sub = (aj & ai) == aj                             # row j subset of row i
    eq = aj == ai
    jge = (np.arange(m)[None, :] >= np.arange(m)[:, None])[None]
    real = sub & ~(eq & jge)
    popj = np.broadcast_to(pop[:, None, :], (B, m, m))
    best = np.where(real, popj, -1).max(axis=2)       # [B,m]
    search = pop >= 2
    use = search & (best >= 1)
    resid = np.where(use, pop - best, pop)
    cost = np.where(pop > 0, np.maximum(resid, 1), 0).sum(axis=1)
    return cost.astype(np.int64), search.sum(axis=1).astype(np.int64)


def perm_rows(rows_tile, mode, rng):
    # rows_tile: [nM, tM, nK] ; TSBG group-priority = issue source-group-major over the tile's tokens.
    # Prosperity's tile cost is a function of the row *set*, so any permutation must give identical cycles.
    nM, tM, nK = rows_tile.shape
    if mode == 'tsbg':
        key = (POP16[rows_tile].sum(axis=2) == 0).astype(np.int64)  # live tokens first, stable
        order = np.argsort(key, axis=1, kind='stable')[:, ::-1]
        return np.take_along_axis(rows_tile, order[:, :, None], axis=1)
    order = np.argsort(rng.random((nM, tM)), axis=1)
    return np.take_along_axis(rows_tile, order[:, :, None], axis=1)


def layer_model(S, info, cfg, acc_cls, FC_cls, rng):
    nt, tM, tK, order, perm = CONFIGS[cfg]
    T, HWn, K, N = info['T'], info['HW'], info['C'], info['N']
    X = S.reshape(T, HWn, K)
    if order == 'official':
        X = X.transpose(1, 0, 2)
    X = X.reshape(-1, K)
    M = X.shape[0]
    nK = K // tK; assert K % tK == 0
    nM = -(-M // tM); nN = -(-N // nt)
    pad = nM * tM - M
    if pad:
        X = np.concatenate([X, np.zeros((pad, K), bool)], 0)
    packed = np.packbits(X.reshape(-1, nK, 16), axis=-1, bitorder='little')   # [Mp,nK,2]
    vals = packed.view('<u2').reshape(nM, tM, nK)
    if perm:
        vals = perm_rows(vals, perm, rng)
    tiles = vals.transpose(0, 2, 1).reshape(-1, tM)                            # (m,k) major
    cost = np.empty(tiles.shape[0], np.int64); srch = np.empty_like(cost)
    B = max(1, (1 << 22) // (tM * tM))
    for b in range(0, tiles.shape[0], B):
        cost[b:b + B], srch[b:b + B] = tile_costs(tiles[b:b + B])
    curM = np.full(nM, tM); curM[-1] = tM - pad
    pre = srch.reshape(nM, nK).sum(1) + (curM // 8) * nK
    compute_raw = int(cost.sum()) * nN
    preprocess = int(pre.sum()) * nN
    # official DRAM accounting (buffer state logic copied from run_fc)
    acc = acc_cls('Prosperity', nt, 32, tM, tK, product_sparsity=True, dense=False, issue_type=2)
    wbits = 8
    act_tile = tM * tK; wgt_tile = tK * nt * wbits
    sa = acc.sram_size['act']; sw = acc.sram_size['wgt']
    st_act = 'all' if M * K <= sa else ('row' if act_tile * nK <= sa else 'tile')
    st_wgt = 'all' if K * N * wbits <= sw else ('col' if wgt_tile * nK <= sw else 'tile')
    act_bits = M * K * (nN if st_act in ('row', 'tile') else 1)
    w_bits_total = K * N * wbits * (nM if st_wgt == 'tile' else 1)
    dram = act_bits + w_bits_total
    init = min(tK, K) * min(nt, N) * wbits + min(tK, K) * min(tM, M)
    compute = max(compute_raw, preprocess)
    mem_stall = init // acc.mem_if_width + max(0, (dram - init) // acc.mem_if_width - compute)
    return dict(compute_raw=compute_raw, preprocess=preprocess, compute_cycles=compute,
                preprocess_stall_cycles=max(0, preprocess - compute_raw), mem_stall_cycles=mem_stall,
                total_cycles=compute + mem_stall, dram_reads_bits=dram, buffer_state=[st_act, st_wgt],
                sram_size_bits=dict(acc.sram_size), tiles_mk=int(tiles.shape[0]), nN=nN,
                bit_sparse_compute=int(X.sum()) * nN)


def run_sample(args):
    sample, cfgs = args
    torch, simmod, FC, Accelerator = cm.import_official()
    idx = cm.get_index(); rng = np.random.default_rng(1234 + sample)
    out = {}
    for layer in sorted(cm.layers()):
        S, info = cm.load_codes_support(sample, layer, idx)
        out[layer] = dict(info=info)
        for c in cfgs:
            t0 = time.time()
            out[layer][c] = layer_model(S, info, c, Accelerator, FC, rng)
            out[layer][c]['sec'] = round(time.time() - t0, 2)
    p = cm.OUT / 'runs_fast' / f's{sample:02d}.json'
    p.write_text(json.dumps(out, indent=1))
    return sample


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', default=list(CONFIGS))
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--samples', nargs='+', type=int, default=list(range(40)))
    a = ap.parse_args()
    cm.get_index()
    (cm.OUT / 'runs_fast').mkdir(exist_ok=True)
    tasks = [(s, a.configs) for s in a.samples if not (cm.OUT / 'runs_fast' / f's{s:02d}.json').exists()]
    print('TASKS', len(tasks), flush=True)
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for s in pool.imap_unordered(run_sample, tasks):
            print('DONE', s, time.strftime('%H:%M:%S'), flush=True)
    print('ALL_DONE', flush=True)
