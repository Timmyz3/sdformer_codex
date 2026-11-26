"""Count Gustav-style spatial NRV opportunities on existing ep34 captures.

No cycle model: index/payload bit counts exclude padding/alignment/descriptors.
Temporal flattening follows bn_state/build_traces.py: [T, position, channel].
"""
from pathlib import Path
import json
import struct
import zlib
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
rows = []


def add_partitions(record, s):
    positions, c = s.shape[1:]
    for p in (4, 8, 16, 32):
        groups = (positions + p - 1) // p
        padded = np.pad(s, ((0, 0), (0, groups*p-positions), (0, 0)))
        live = padded.reshape(10, groups, p, c).any(axis=2)
        nrv = int(live.sum())
        indices = int(np.ceil(np.log2(c)))
        record['spatial_partitions'].append(dict(
            P=p, groups=groups, nrvs=nrv,
            per_tick_empty_row_fraction=float((~live).mean()),
            all_T_empty_group_row_fraction=float((~live.any(axis=0)).mean()),
            nrv_min_index_and_payload_bits=nrv*(indices+p),
            original_bitmap_bits=int(s.size),
            nrv_payload_ratio_vs_bitmap=nrv*(indices+p)/s.size,
            layout='adjacent flattened spatial positions; trailing padding counted in payload'))
    rows.append(record)


for sid in (0, 10):
    for stage in (0, 3):
        path = BASE / 'bn_state' / f'trace_s{sid}_stage{stage}.npz'
        with np.load(path) as z:
            c = z['W'].shape[1]
            h = z['W'].shape[0]
            flat = np.unpackbits(z['source_packed'], axis=1, bitorder='little')[:, :c]
            s = flat.reshape(10, -1, c).astype(bool)
            positions = s.shape[1]
            record = dict(sample=sid, stage=stage, source=str(path.relative_to(BASE)),
                          T=10, positions=positions, C=c, H=h,
                          spike_density=float(s.mean()),
                          fp32_weight_zero_fraction=float((z['W'] == 0).mean()),
                          all_T_silent_source_fraction=float((~s.any(axis=0)).mean()),
                          spatial_partitions=[])
        add_partitions(record, s)

# Include all six S2 FC1 blocks, since S2 dominates FFN work.
cap = BASE.parents[2] / 'SDformer/hw_autoresearch_nts07/results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901'
specs = {r['layer_id']: r for r in json.loads((cap/'layers.json').read_text())['layers']
         if 'layers.2.' in r['module_name'] and '.mlp.fc1' in r['module_name']}
chunks = {(sid, lid): [] for sid in (0, 10) for lid in specs}
header = struct.Struct('<8sHH11I')
with (cap/'fc_frames.bin').open('rb') as f:
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
    record = dict(sample=sid, stage=2, block=int(spec['module_name'].split('swin_blocks.')[1].split('.')[0]),
                  module=spec['module_name'], source=str(cap/'fc_frames.bin'),
                  T=10, positions=1200, C=384, H=1536, spike_density=float(s.mean()),
                  fp32_weight_zero_fraction=None,
                  all_T_silent_source_fraction=float((~s.any(axis=0)).mean()), spatial_partitions=[])
    add_partitions(record, s)
out = dict(scope='16 archived ep34 FC1 input domains: S0/S3 block0 plus all six S2 blocks, samples 0 and 10; statistics only, no RTL/cycle/energy result.',
           excludes='NRV generation, grouping transpose, descriptors, alignment, weight/BN/PSN traffic, bank conflicts; cannot infer speedup.',
           traces=rows)
(ROOT / 'GUSTAV_NRV_OPPORTUNITY.json').write_text(json.dumps(out, indent=2)+'\n')
for row in rows:
    r = row['spatial_partitions'][1]
    print(f"s{row['sample']} S{row['stage']} B{row.get('block',0)} density={row['spike_density']:.4%} "
          f"P8 zero-row={r['per_tick_empty_row_fraction']:.4%} "
          f"T10-P8 zero={r['all_T_empty_group_row_fraction']:.4%} "
          f"NRV/bitmap bits={r['nrv_payload_ratio_vs_bitmap']:.4f}")
