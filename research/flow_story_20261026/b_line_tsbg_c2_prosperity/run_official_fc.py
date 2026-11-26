#!/usr/bin/env python3
"""Official (unmodified) Prosperity Simulator.run_fc on ALL 24 ep34 FC1/FC2 layers x 40 samples (CPU)."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import argparse, contextlib, io, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import common as cm

CONFIGS = {'matched_ps': (96, 64, 16, True, False), 'matched_bit': (96, 64, 16, False, False)}


def run_task(args):
    cfg, sample, layer = args
    torch, simmod, FC, Accelerator = cm.import_official()
    S, info = cm.load_codes_support(sample, layer)
    T, HWn, K, N = info['T'], info['HW'], info['C'], info['N']
    nt, mt, kt, ps, dense = CONFIGS[cfg]
    op = FC(info['name'], K, N, HWn, 1, T)          # run_fc permutes [T,HW,K] -> [HW,T,K]
    op.activation_tensor.sparse_map = torch.from_numpy(S.reshape(T, HWn, K).astype(np.float32))
    acc = Accelerator('Prosperity', nt, 32, mt, kt, product_sparsity=ps, dense=dense, issue_type=2)
    sim = simmod.Simulator(acc, [op], 'ep34_fc_native', use_cuda=False)
    simmod.clear_global_stats()
    buf = io.StringIO(); t0 = time.time()
    with contextlib.redirect_stdout(buf):
        s = sim.run_fc(op)
    res = dict(config=cfg, sample=sample, layer=layer, info=info,
               accelerator=dict(adder_array_size=nt, tile_M=mt, tile_K=kt, product_sparsity=ps,
                                sram_size_bits=dict(acc.sram_size), mem_if_width=acc.mem_if_width),
               weight_nbits=int(op.weight_tensor.nbits), output_nbits=int(op.output_tensor.nbits),
               total_cycles=int(s.total_cycles), compute_cycles=int(s.compute_cycles),
               preprocess_stall_cycles=int(s.preprocess_stall_cycles), mem_stall_cycles=int(s.mem_stall_cycles),
               reads={k: int(v) for k, v in s.reads.items()}, writes={k: int(v) for k, v in s.writes.items()},
               official_stdout=buf.getvalue()[-400:], host_seconds=time.time() - t0)
    p = cm.OUT / 'runs_official' / f'{cfg}__s{sample:02d}_L{layer:02d}.json'
    p.write_text(json.dumps(res, indent=1))
    return cfg, sample, layer, res['total_cycles'], res['host_seconds']


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', default=['matched_ps'])
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--samples', nargs='+', type=int, default=list(range(40)))
    ap.add_argument('--layers', nargs='+', type=int, default=sorted(cm.layers()))
    a = ap.parse_args()
    cm.get_index()
    (cm.OUT / 'runs_official').mkdir(exist_ok=True)
    tasks = [(c, s, l) for c in a.configs for s in a.samples for l in a.layers
             if not (cm.OUT / 'runs_official' / f'{c}__s{s:02d}_L{l:02d}.json').exists()]
    big = {8, 9, 10, 11}
    tasks.sort(key=lambda t: (t[2] not in big, t[1]))
    print('TASKS', len(tasks), flush=True)
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for cfg, s, l, cyc, sec in pool.imap_unordered(run_task, tasks):
            print('DONE', cfg, s, l, cyc, round(sec, 1), time.strftime('%H:%M:%S'), flush=True)
    print('ALL_DONE', flush=True)
