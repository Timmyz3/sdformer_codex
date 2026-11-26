#!/usr/bin/env python3
"""Feed the sealed C1 ep34 Conv3x3 input spikes to the OFFICIAL Prosperity simulator (CPU).

Only an input adapter is local. Cycles come from the unmodified official
Simulator.run_fc (snapshot of simulator.py.official_bak, rev 6ee1c6f).
"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import argparse, contextlib, hashlib, io, json, time, types
from pathlib import Path
import numpy as np

HOME = Path.home()
OUT = HOME / 'work/sdformer_codex/research/flow_story_20261026/c1_prosperity_denominator'
SNAP = OUT / 'official_sim_snapshot'
HW = HOME / 'work/sdformer_codex/SDformer/hw_autoresearch_nts07'
CAPTURE = HW / 'results/m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831'
LEDGER = HW / 'results/m1590_ep34_c1_same_ledger_cycle_model_r1_20260901/ep34_c1_support16_rows.memh'
MODULES = ('sttmultires_unet.resblocks.0.conv1.0', 'sttmultires_unet.resblocks.0.conv2.0',
           'sttmultires_unet.resblocks.1.conv1.0', 'sttmultires_unet.resblocks.1.conv2.0')
T, C, H, W = 10, 768, 15, 20
COUT = 768

# name: (adder_array_size=N tile, tile_M, tile_K, product_sparsity, dense, row_order)
CONFIGS = {
    'matched_ps':        (96, 64, 16, True, False, 'official'),
    'matched_bit':       (96, 64, 16, False, False, 'official'),
    'matched_dense':     (96, 64, 16, False, True, 'official'),
    'matched_ps_c1order': (96, 64, 16, True, False, 'c1'),
    'default_ps':        (128, 256, 16, True, False, 'official'),
    'default_bit':       (128, 256, 16, False, False, 'official'),
    'default_dense':     (128, 256, 16, False, True, 'official'),
}


def records():
    out = []
    with (CAPTURE / 'unified_ordered_records.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            if r.get('cohort') == 'c1' and r.get('category') == 'c1_conv3x3' and r['payload'].get('retained'):
                out.append(r)
    out.sort(key=lambda r: (r['global_sample_id'], r['global_order']))
    assert len(out) == 40, len(out)
    for i, r in enumerate(out):
        assert r['name'] == MODULES[i % 4] and r['global_sample_id'] == i // 4, (i, r['name'])
        assert r['input']['shape'] == [T, 1, C, H, W]
    return out


def decode_support(r):
    p = r['payload']
    raw = (CAPTURE / p['support_sign']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == p['support_sign_sha256']
    n = int(p['positive_plane_bytes'])
    pos, neg = raw[:n], raw[n:]
    assert len(neg) == int(p['negative_plane_bytes']) and not any(neg)
    bits = np.unpackbits(np.frombuffer(pos, dtype=np.uint8), bitorder='little')
    assert bits.size == T * C * H * W and int(bits.sum()) == r['input']['active']
    return bits.astype(np.bool_).reshape(T, C, H, W)


def import_official():
    # prosparsity_engine is the CUDA extension, used only by run_fc_cuda; GPU is forbidden here.
    sys.modules.setdefault('prosparsity_engine', types.ModuleType('prosparsity_engine'))
    sys.path.insert(0, str(SNAP))
    import torch
    torch.set_num_threads(1)
    import simulator as simmod
    from networks import FC
    from accelerator import Accelerator
    from utils import img2col
    return torch, simmod, FC, Accelerator, img2col


def im2col_rows(torch, img2col, support):
    x = torch.from_numpy(support.astype(np.float32))          # [T,C,H,W]; batch dim = T
    cols = img2col(x, 3, 1, 1)                                 # official util -> [T, H*W, C*9]
    assert tuple(cols.shape) == (T, H * W, C * 9)
    return cols


def run_task(args):
    cfg, idx = args
    torch, simmod, FC, Accelerator, img2col = import_official()
    recs = records()
    r = recs[idx]
    support = decode_support(r)
    cols = im2col_rows(torch, img2col, support)
    n_tile, mt, kt, ps, dense, order = CONFIGS[cfg]
    name = r['name'] + '_kernel3_fc'   # official conv2d_2_fc naming, so on-chip img2col DRAM rule applies
    if order == 'official':
        op = FC(name, C * 9, COUT, H * W, 1, T)            # run_fc permutes [T,N,K]->[N,T,K]
        op.activation_tensor.sparse_map = cols.contiguous()
    else:
        op = FC(name, C * 9, COUT, T * H * W, 1, 1)        # keep C1 ledger order (t,y,x)
        op.activation_tensor.sparse_map = cols.reshape(1, T * H * W, C * 9).contiguous()
    acc = Accelerator('Prosperity', n_tile, 32, mt, kt, product_sparsity=ps, dense=dense, issue_type=2)
    sim = simmod.Simulator(acc, [op], 'c1_ep34_native', use_cuda=False)
    simmod.clear_global_stats()
    buf = io.StringIO(); t0 = time.time()
    with contextlib.redirect_stdout(buf):
        s = sim.run_fc(op)
    res = dict(config=cfg, index=idx, sample=idx // 4, operator=idx % 4, module=r['name'],
               accelerator=dict(adder_array_size=n_tile, tile_M=mt, tile_K=kt, product_sparsity=ps, dense=dense,
                                issue_type=2, mem_if_width=acc.mem_if_width, sram_size_bits=dict(acc.sram_size),
                                row_order=order),
               input_active=int(support.sum()), im2col_nnz=int((cols != 0).sum().item()),
               total_cycles=int(s.total_cycles), compute_cycles=int(s.compute_cycles),
               preprocess_stall_cycles=int(s.preprocess_stall_cycles), mem_stall_cycles=int(s.mem_stall_cycles),
               num_ops=int(s.num_ops), reads={k: int(v) for k, v in s.reads.items()},
               writes={k: int(v) for k, v in s.writes.items()}, official_stdout=buf.getvalue(),
               host_seconds=time.time() - t0)
    if ps:
        res['global_counters'] = dict(ori_nnz=int(sum(simmod.ori_nnzs)), processed_nnz=int(sum(simmod.processed_nnzs)),
                                      ops_nnz=int(sum(simmod.ops_nnzs)), EM_rows=int(sum(simmod.num_EM_row)),
                                      PM_rows=int(sum(simmod.num_PM_row)), all_zero_rows=int(sum(simmod.num_all_zero_row)),
                                      prefix_edges=int(sum(simmod.rank_one_prefix)),
                                      note='appended once per (m,n,k) tile: divide by tile_num_N for per-matrix counts')
    p = OUT / 'runs' / f'{cfg}__s{idx // 4:02d}_o{idx % 4}.json'
    p.write_text(json.dumps(res, indent=1))
    return cfg, idx, res['total_cycles'], res['host_seconds']


def verify_ledger():
    """Bit-exact check: official img2col of the captured spikes == C1 ledger support16 rows."""
    torch, simmod, FC, Accelerator, img2col = import_official()
    recs = records()
    raw = np.fromfile(LEDGER, dtype=np.uint8).reshape(-1, 9)
    assert raw.shape[0] == 51_840_000 and (raw[:, 8] == 10).all()
    lut = np.zeros(256, np.uint16)
    for i, ch in enumerate(b'0123456789abcdef'):
        lut[ch] = i
    hexd = lut[raw[:, 4:8]]
    led = (hexd[:, 0] << 12) | (hexd[:, 1] << 8) | (hexd[:, 2] << 4) | hexd[:, 3]
    led = led.reshape(10, 4, 432, 3000)
    weights = (1 << np.arange(16, dtype=np.uint32))
    out = []
    for idx, r in enumerate(recs):
        cols = im2col_rows(torch, img2col, decode_support(r)).numpy().astype(np.uint32)  # [T,HW,K]
        rows = cols.reshape(T * H * W, 432, 16)                     # (t,y,x) rows
        mine = (rows * weights).sum(-1).astype(np.uint16).T          # [432,3000]
        eq = bool(np.array_equal(mine, led[idx // 4, idx % 4]))
        out.append(dict(index=idx, equal=eq, im2col_nnz=int(cols.sum())))
        assert eq, idx
    tot = sum(o['im2col_nnz'] for o in out)
    return dict(all_40_layers_bit_exact=True, im2col_nnz_total=tot, per_layer=out)


def timing_probe():
    torch, simmod, FC, Accelerator, img2col = import_official()
    r = records()[0]
    cols = im2col_rows(torch, img2col, decode_support(r))
    a = cols.permute(1, 0, 2).reshape(-1, C * 9)
    res = {}
    for mt in (64, 256):
        t0 = time.time(); n = 0
        for k in range(0, 40 * 16, 16):
            simmod.find_product_sparsity(a[0:mt, k:k + 16]); n += 1
        res[mt] = (time.time() - t0) / n
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['probe', 'verify', 'run'])
    ap.add_argument('--configs', nargs='+', default=list(CONFIGS))
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--indices', nargs='+', type=int, default=list(range(40)))
    a = ap.parse_args()
    if a.mode == 'probe':
        print(json.dumps(timing_probe()))
    elif a.mode == 'verify':
        v = verify_ledger(); (OUT / 'LEDGER_IDENTITY.json').write_text(json.dumps(v, indent=1))
        print('VERIFY_OK', v['im2col_nnz_total'])
    else:
        import multiprocessing as mp
        (OUT / 'runs').mkdir(exist_ok=True)
        tasks = [(c, i) for c in a.configs for i in a.indices
                 if not (OUT / 'runs' / f'{c}__s{i // 4:02d}_o{i % 4}.json').exists()]
        # heavy product-sparsity tasks first
        tasks.sort(key=lambda t: (not CONFIGS[t[0]][3], t[0]))
        print('TASKS', len(tasks), flush=True)
        with mp.get_context('spawn').Pool(a.workers) as pool:
            for cfg, idx, cyc, sec in pool.imap_unordered(run_task, tasks):
                print('DONE', cfg, idx, cyc, round(sec, 1), time.strftime('%H:%M:%S'), flush=True)
        print('ALL_DONE', flush=True)
