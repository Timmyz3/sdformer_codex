"""Native ep34 support + checkpoint W -> a new, explicit signed8 linear contract.

Selected T10/P8/H8 slices, not full-layer RTL or an accuracy evaluation.
"""
from pathlib import Path
import json
import struct
import sys
import zlib
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = BASE.parents[2]
sys.path.insert(0, str(BASE.parent / 'mechanism_rebuild_gh_20260906/scripts'))
from checkpoint_numpy import read_checkpoint

checkpoint = REPO / 'SDformer/hw_autoresearch_nts07/system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth'
state = read_checkpoint(checkpoint)['model_state_dict']
domains = []
for sid in (0, 10):
    path = BASE / 'bn_state' / f'trace_s{sid}_stage0.npz'
    with np.load(path) as z:
        s = np.unpackbits(z['source_packed'], axis=1, bitorder='little').reshape(10, -1, 96)
    domains.append((sid, 0, 0, s))

cap = REPO / 'SDformer/hw_autoresearch_nts07/results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901'
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
        bits = np.unpackbits(np.frombuffer(data[:n*br], np.uint8).reshape(n, br), axis=1, bitorder='little')[:, :c]
        assert start == sum(len(x) for x in chunks[sid, lid])
        chunks[sid, lid].append(bits)
for (sid, lid), parts in chunks.items():
    block = int(specs[lid]['module_name'].split('swin_blocks.')[1].split('.')[0])
    domains.append((sid, 2, block, np.concatenate(parts).reshape(10, 1200, 384)))

cases, meta = [], []
for sid, stage, block, s in domains:
    prefix = f'sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.{block}.mlp.'
    w = state[prefix+'fc1.weight'].astype(np.float64)
    theta = float(state[prefix+'sn1.spiking_neuron.thresh'])
    w *= theta
    for hbase in (0, w.shape[0]//2):
        wf = w[hbase:hbase+8]
        scale = np.max(np.abs(wf), axis=1)/127
        q = np.clip(np.rint(wf/scale[:, None]), -128, 127).astype(np.int32).T.copy()
        # Eight evenly spaced complete P8 groups include interior as well as edges.
        for pbase in np.linspace(0, s.shape[1]//8-1, 8, dtype=int)*8:
            pbase = int(pbase)
            src = s[:, pbase:pbase+8, :].reshape(80, -1).copy()
            label = f'ep34_s{sid}_S{stage}B{block}_h{hbase}_p{pbase}'
            cases.append((label, src, q))
            meta.append(dict(label=label, sample=sid, stage=stage, block=block, hbase=hbase,
                             pbase=pbase, C=src.shape[1], theta=theta, scale=scale.tolist(),
                             nnz=int(src.sum()), real=True))

rng = np.random.default_rng(20260920)
for c in (96, 384):
    q = rng.integers(-128, 128, size=(c, 8), dtype=np.int32)
    q[:, 0] = -128
    q[:, 1] = 127
    q[:, 2] = np.where(np.arange(c)%2, -127, 127)
    for pattern in ('zero', 'full', 'alternating', 'boundary'):
        src = np.zeros((80, c), np.uint8)
        if pattern == 'full':
            src[:] = 1
        elif pattern == 'alternating':
            src[:] = (np.arange(c)[None, :] + np.arange(80)[:, None])%2
        elif pattern == 'boundary':
            for r in range(80):
                src[r, [r%c, c-1, min(127, c-1), min(128, c-1)]] = 1
        label = f'diagnostic_C{c}_{pattern}'
        cases.append((label, src, q))
        meta.append(dict(label=label, C=c, nnz=int(src.sum()), real=False))

with (HERE/'cases.txt').open('w') as f:
    f.write(f'{len(cases)}\n')
    for label, s, w in cases:
        c = s.shape[1]
        words = (c+127)//128
        padded = np.pad(s, ((0, 0), (0, words*128-c)))
        packed = np.packbits(padded, axis=1, bitorder='little').copy().view('<u4')
        y = s.astype(np.int64) @ w.astype(np.int64)
        f.write(f'{label} {c}\n')
        for row in packed:
            f.write(' '.join(f'{int(x):08x}' for x in row)+'\n')
        for row in w:
            f.write(' '.join(map(str, row))+'\n')
        for row in y:
            f.write(' '.join(map(str, row))+'\n')
(HERE/'case_metadata.json').write_text(json.dumps(dict(
    scope=f'{sum(m["real"] for m in meta)} real slices from S0 B0 and all six S2 FC1 blocks, samples 0/10; eight evenly spaced spatial P8 groups and two H8 groups per domain; 8 diagnostic slices',
    quantization='Per-output-channel round-even Q8 of checkpoint W*theta, maxabs/127. New linear integer contract; no FP32 equivalence or AEE claim.',
    checkpoint=str(checkpoint), cases=meta), indent=2)+'\n')
print(f'Prepared {len(cases)} cases, {sum(m["real"] for m in meta)} real; 80x8 outputs each.')
