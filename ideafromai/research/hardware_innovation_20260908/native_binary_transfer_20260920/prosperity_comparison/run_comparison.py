"""Same archived ep34 input: ProSparsity vs G4/NRV opportunities.

Only the explicitly labelled official run_fc results are model cycles.
No CUDA execution, new quantization, network inference, or production writes.
"""
import sys
sys.dont_write_bytecode = True
assert sys.version_info[:2] == (3, 12)
from pathlib import Path
from collections import Counter
import contextlib
import io
import json
import struct
import time
import zlib
import csv
import numpy as np
import torch  # Load the working local environment before legacy wrapper path insertion.

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
REPO = BASE.parents[2]
sys.path.insert(0, str(BASE.parent / 'complete_transfer_20260907'))
import c1_full_layer as prior


def captures():
    for sid in (0, 10):
        for stage in (0, 3):
            path = BASE / 'bn_state' / f'trace_s{sid}_stage{stage}.npz'
            with np.load(path) as z:
                c, h = z['W'].shape[1], z['W'].shape[0]
                s = np.unpackbits(z['source_packed'], axis=1, bitorder='little')[:, :c]
                s = s.reshape(10, -1, c).astype(bool)
            yield dict(sample=sid, stage=stage, block=0, H=h,
                       source=str(path), module=f'layers.{stage}.swin_blocks.0.mlp.fc1'), s
    cap = REPO / 'SDformer/hw_autoresearch_nts07/results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901'
    specs = {r['layer_id']: r for r in json.loads((cap / 'layers.json').read_text())['layers']
             if 'layers.2.' in r['module_name'] and '.mlp.fc1' in r['module_name']}
    chunks = {(sid, lid): [] for sid in (0, 10) for lid in specs}
    header = struct.Struct('<8sHH11I')
    with (cap / 'fc_frames.bin').open('rb') as f:
        while raw := f.read(header.size):
            magic, version, hs, lid, sid, fi, start, n, c, br, nnz, rb, cb, unused = header.unpack(raw)
            if sid > 10:
                break
            if (sid, lid) not in chunks:
                f.seek(cb, 1)
                continue
            data = zlib.decompress(f.read(cb))
            bits = np.unpackbits(np.frombuffer(data[:n*br], np.uint8).reshape(n, br),
                                 axis=1, bitorder='little')[:, :c]
            assert start == sum(len(x) for x in chunks[sid, lid])
            chunks[sid, lid].append(bits)
    for (sid, lid), parts in chunks.items():
        spec = specs[lid]
        s = np.concatenate(parts).reshape(10, 1200, 384).astype(bool)
        yield dict(sample=sid, stage=2,
                   block=int(spec['module_name'].split('swin_blocks.')[1].split('.')[0]),
                   H=1536, source=str(cap / 'fc_frames.bin'),
                   module=spec['module_name'], layer_id=lid), s


def official_run(Accelerator, FC, Simulator, s, h, product, relation):
    t, p, c = s.shape
    accel = Accelerator('Prosperity', 128, 32, 256, 16,
                        product_sparsity=product, issue_type=2, mem_if_width=1024)
    # This is FC, not the old kernel3 im2col wrapper: no divide-by-nine traffic heuristic.
    op = FC('ep34_native_mlp_fc1', c, h, p, 1, t)
    op.activation_tensor.sparse_map = torch.from_numpy(s.copy())
    sim = Simulator(accel, [op], benchmark_name='ep34_native_fc', use_cuda=False)
    scope = Simulator.run_fc.__globals__
    original = scope['find_product_sparsity']
    scope['clear_global_stats']()
    if relation is not None:
        scope['find_product_sparsity'] = relation
    stdout = io.StringIO()
    start = time.monotonic()
    try:
        with contextlib.redirect_stdout(stdout):
            got = sim.run_fc(op)
    finally:
        scope['find_product_sparsity'] = original
    ordered = s.transpose(1, 0, 2).reshape(t*p, c)
    assert np.array_equal(op.activation_tensor.sparse_map.reshape(t*p, c).numpy(), ordered)
    return prior.fields(got), stdout.getvalue(), time.monotonic()-start, accel


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    Accelerator, FC, Simulator, _ = prior.load_official_api()
    original = Simulator.run_fc.__globals__['find_product_sparsity']
    rng = np.random.default_rng(20260920)
    checks = []
    directed = np.array([0, 1, 1, 3, 7, 11, 3, 15], np.uint16)
    arrays = [((directed[:, None] >> np.arange(16)) & 1).astype(bool)]
    arrays += [rng.random((m, k)) < .23 for m, k in ((9, 5), (256, 16), (224, 16))]
    for a in arrays:
        rel = prior.Relation()
        x, px = original(torch.from_numpy(a.copy()))
        y, py = rel(a)
        assert torch.equal(x, y) and torch.equal(px, py)
        checks.append(dict(kind='directed_or_random', shape=list(a.shape), matched=True))

    # Original relation + original outer versus validated callback, on a non-divisible multi-K/N shape.
    small = rng.random((10, 29, 35)) < .19
    outer_checks = []
    for product in (False, True):
        plain, _, _, _ = official_run(Accelerator, FC, Simulator, small, 197, product, None)
        fast, _, _, acc = official_run(Accelerator, FC, Simulator, small, 197, product, prior.Relation())
        ref, _, _ = prior.reference(small.transpose(1, 0, 2).reshape(290, 35), acc,
                                    product, prior.Relation(), N=197, conv=False)
        assert plain == fast == ref
        outer_checks.append(dict(product=product, shape=[10, 29, 35, 197], exact_fields=plain))

    report = dict(date='2026-09-20', scope='16 archived ep34 FC1 source domains, full source population in each domain; statistical opportunities plus 3 separately labelled full official CPU run_fc comparisons.',
                  official_tile_MKN=[256, 16, 128], official_row_order='P-major then T within each P; K unchanged',
                  official_weights_outputs_bits=8, official_mem_if_bits=1024,
                  parent_rule='Only child popcount>=2; nonempty subset; exact-equal parent index must be smaller; max parent popcount then smallest original index (official CPU torch.argmax).',
                  parent_join_note='All parent relations require parent value availability. Official issue2 adds one explicit copy cycle only for nonzero exact matches (EM); partial-match parent load is not separately timed. We report all joins and EM separately, not a fabricated sum-cycle model.',
                  limits=['Statistics exclude matcher, parent cache/ports, scheduling, table construction, alignment, NRV creation, BN and full PSN.',
                          'Official CPU model has ideal issue2 sorting and aggregate preprocessing overlap; default SRAM bits act4096/wgt16384/out262144, not paper SRAM sizes or mapped hardware.',
                          'No trained W numerics or AEE run; binary support identities are retained but fp8 counter model is not ep34 FP32 deployment.',
                          'Different G4/NRV/parent operations cannot be compared as equal-cost cycles. No hardware winner asserted.'],
                  relation_checks=checks, original_outer_checks=outer_checks, domains=[], official_runs=[])
    tile_rows = []
    old = json.loads((BASE / 'handoff_review_20260919/GUSTAV_NRV_OPPORTUNITY.json').read_text())
    old_by_key = {(x['sample'], x['stage'], x.get('block', 0)): x for x in old['traces']}
    for meta, s in captures():
        start = time.monotonic()
        t, p, c = s.shape
        ordered = np.ascontiguousarray(s.transpose(1, 0, 2).reshape(t*p, c))
        relation = prior.Relation()
        cnt = Counter()
        for m in range(0, t*p, 256):
            for k in range(0, c, 16):
                a = ordered[m:m+256, k:k+16]
                delta_t, parents_t, counts = relation.entry(a)
                delta = delta_t.numpy()
                parent = parents_t.numpy().astype(np.int64)
                pc = a.sum(1)
                live = parent >= 0
                equal = live & (delta.sum(1) == 0)
                idx = np.arange(len(a))
                parent_values = np.zeros_like(a)
                parent_values[live] = a[parent[live]]
                assert not (delta & parent_values).any()
                assert np.array_equal(delta | parent_values, a)
                assert np.all((pc[parent[live]] < pc[live]) |
                              ((pc[parent[live]] == pc[live]) & (parent[live] < idx[live])))
                row = dict(meta, m=m, k=k, rows=len(a), original_nnz=int(a.sum()),
                           root_nnz=int(delta[~live].sum()), child_delta_nnz=int(delta[live].sum()),
                           nonempty_roots=int(((pc > 0) & ~live).sum()), parent_joins=int(live.sum()),
                           exact_match_copies=int(equal.sum()), partial_match_joins=int((live & ~equal).sum()),
                           original_zero_rows=int((pc == 0).sum()), preprocessing_queries=int((pc > 1).sum()),
                           popcount_batches=len(a)//8)
                tile_rows.append(row)
                cnt.update({key: row[key] for key in ('original_nnz', 'root_nnz', 'child_delta_nnz',
                           'nonempty_roots', 'parent_joins', 'exact_match_copies', 'partial_match_joins',
                           'original_zero_rows', 'preprocessing_queries', 'popcount_batches')})
                cnt['tiles'] += 1
                cnt['tile_rows'] += len(a)
        # A real tile in each domain is compared with the unchanged official relation.
        a = ordered[:256, :16]
        x, px = original(torch.from_numpy(a.copy()))
        y, py = relation(a)
        assert torch.equal(x, y) and torch.equal(px, py)
        checks.append(dict(kind='real_domain_tile', sample=meta['sample'], stage=meta['stage'], block=meta['block'], matched=True))
        w = rng.integers(-128, 128, (16, 8), dtype=np.int64)
        direct = a.astype(np.int64) @ w
        emitted = np.zeros_like(direct)
        parent = py.numpy().astype(int)
        for i in np.lexsort((np.arange(len(a)), a.sum(1))):
            emitted[i] = y[i].numpy().astype(np.int64) @ w
            if parent[i] >= 0:
                emitted[i] += emitted[parent[i]]
        assert np.array_equal(direct, emitted)
        g4 = int(s.reshape(t, p, c//4, 4).any(axis=-1).sum())
        groups = (p+7)//8
        padded = np.pad(s, ((0,0),(0,groups*8-p),(0,0)))
        nrv_live = padded.reshape(t, groups, 8, c).any(axis=2)
        nrv = int(nrv_live.sum())
        nrv_payload_bits = nrv*(8 + int(np.ceil(np.log2(c))))
        old_nrv = old_by_key[meta['sample'],meta['stage'],meta['block']]['spatial_partitions'][1]
        assert nrv == old_nrv['nrvs']
        assert nrv_payload_bits == old_nrv['nrv_min_index_and_payload_bits']
        residual = cnt['root_nnz'] + cnt['child_delta_nnz']
        domain = dict(meta, T=t, positions=p, C=c, source_bits=int(s.size), source_nnz=int(s.sum()),
                      spike_density=float(s.mean()), prosperity=dict(cnt),
                      prosperity_residual_nnz=residual,
                      prosperity_residual_reduction=1-residual/cnt['original_nnz'],
                      prosperity_residual_plus_all_parent_references=residual+cnt['parent_joins'],
                      prosperity_official_data_issue_proxy=residual+cnt['exact_match_copies'],
                      g4_nonempty_queries=g4, g4_combination_reduction=1-g4/int(s.sum()),
                      gustav_P8_nrv_rows=nrv, gustav_P8_spike_weight_scatter_contributions=int(s.sum()),
                      gustav_P8_index_payload_ratio=nrv_payload_bits/s.size,
                      gustav_P8_per_tick_empty_fraction=float((~nrv_live).mean()),
                      gustav_P8_T10_union_empty_fraction=float((~nrv_live.any(axis=0)).mean()),
                      host_seconds_not_cycles=time.monotonic()-start)
        report['domains'].append(domain)
        print(f"s{meta['sample']} S{meta['stage']} B{meta['block']} nnz={cnt['original_nnz']} root+delta={residual} "
              f"parents={cnt['parent_joins']} EM={cnt['exact_match_copies']} G4={g4} NRV={nrv}", flush=True)
        if (meta['sample'],meta['stage'],meta['block']) in ((0,0,0),(0,2,0),(0,2,5)):
            modes = {}
            for product in (False, True):
                got, stdout, elapsed, acc = official_run(Accelerator, FC, Simulator, s, meta['H'], product, relation)
                expected, _, residency = prior.reference(ordered, acc, product, relation, N=meta['H'], conv=False)
                assert got == expected, (meta, product, got, expected)
                modes['product' if product else 'bit'] = dict(official_outer_counts=got, stdout=stdout,
                     host_seconds_not_cycles=elapsed, artifact_residency=residency,
                     counters_bits_not_bytes=True, default_sram_bits=acc.sram_size)
                print('  official run_fc', 'product' if product else 'bit', got['total_cycles'], flush=True)
            report['official_runs'].append(dict(meta, T=t, positions=p, C=c, M=t*p,
                  modes=modes, model_bit_over_product=modes['bit']['official_outer_counts']['total_cycles']/modes['product']['official_outer_counts']['total_cycles']))
        (HERE / 'results.json').write_text(json.dumps(report, indent=2)+'\n')
    with (HERE / 'tile_counts.csv').open('w', newline='') as f:
        fields = sorted(set().union(*(r.keys() for r in tile_rows)))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(tile_rows)
    report['completed_domains'] = len(report['domains'])
    report['completed_tiles'] = len(tile_rows)
    report['all_tile_support_reconstruction_and_acyclic_parent_checks'] = 'PASS'
    report['official_relation_checks'] = len(checks)
    report['representative_random_integer_N8_numeric_checks'] = 16*256*8
    (HERE / 'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print('COMPLETE', len(tile_rows), 'tiles,', len(report['domains']), 'domains', flush=True)


if __name__ == '__main__':
    main()
