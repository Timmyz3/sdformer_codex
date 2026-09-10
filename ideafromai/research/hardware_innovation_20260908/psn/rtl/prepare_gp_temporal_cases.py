"""Bounded real-code/W test cases; NumPy int64 dot is the numeric reference."""
from pathlib import Path
import json
import struct
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'bn_state'))
from support_service_model import read_torch


def make_case(name, codes, weight, decode, p4f2, real=False, splits=None, meta=None):
    codes = np.asarray(codes, dtype=np.uint8)  # P,C; actual 3-bit source codes
    weight = np.asarray(weight, dtype=np.int8)  # F,C, real theta-folded W
    decode = np.asarray(decode, dtype=np.uint8)  # R,8, each element 0/1
    P, C = codes.shape
    F = 2 if p4f2 else 1
    R = decode.shape[0]
    assert P == (4 if p4f2 else 8) and weight.shape == (F, C)
    assert 1 <= R <= 7 and decode.shape[1] == 8
    assert codes.max(initial=0) < 8 and np.all((decode == 0) | (decode == 1))
    # A dense, non-NRV, int64 matrix product per output f, independent of
    # packet grouping, member masks and RTL scheduling.
    gates = decode[:, codes].transpose(1, 2, 0).astype(np.int64)  # P,C,R
    values = np.einsum('pcr,fc->fpr', gates, weight.astype(np.int64))
    touched = np.einsum('pcr,fc->fpr', gates,
                        (weight != 0).astype(np.int64)) != 0
    expected = np.zeros(56, dtype=np.int32)
    valid = np.zeros(56, dtype=np.uint8)
    for f in range(F):
        for p in range(P):
            expected[(f*P+p)*7:(f*P+p)*7+R] = values[f, p]
            valid[(f*P+p)*7:(f*P+p)*7+R] = touched[f, p]
    # The static bound applies to ALL binary support subsets and all source
    # prefixes, including regrouping. It does not assume measured cancellation.
    lower = np.minimum(weight.astype(np.int64), 0).sum(1)
    upper = np.maximum(weight.astype(np.int64), 0).sum(1)
    assert lower.min() >= -16384 and upper.max() <= 16383
    if splits is None:
        splits = [min(4, C-i) for i in range(0, C, 4)]
    assert sum(splits) == C and all(0 <= n <= 4 for n in splits)
    packets = []
    first = 0
    for n in splits:
        # Nonzero poison in inactive fields ensures count/P/F controls matter.
        cs = np.full((4, 8), 7, dtype=np.uint8)
        ws = np.full((2, 4), -128, dtype=np.int8)
        cs[:n, :P] = codes[:, first:first+n].T
        ws[:F, :n] = weight[:, first:first+n]
        packets.append((n, cs, ws))
        first += n
    return dict(name=name, p4f2=p4f2, real=real, rank=R, codes=codes,
                weight=weight, decode=decode, packets=packets,
                expected=expected, valid=valid,
                meta=dict(meta or {}, source_count=C, static_S15_range=[int(lower.min()), int(upper.max())]))


def main():
    params = read_torch(ROOT/'algorithm/stage2_temporal_codes/integer_parameters.pt')
    books = np.load(ROOT/'algorithm/stage2_temporal_codes/codebooks.npz')
    folder = ROOT/'algorithm/direct_code_integer/deployment/capture10'
    paths = sorted(p for p in folder.glob('*.npz') if p.name.startswith(('v000_', 'v006_')))
    assert len(paths) == 12
    cases = []
    for index, path in enumerate(paths):
        block = int(path.stem[-1])
        q = params[f'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.{block}.mlp.']
        W = np.asarray(q['weight_int8'], dtype=np.int8)
        D = np.asarray(books[f's2b{block}_dictionary'], dtype=np.uint8)
        with np.load(path) as z:
            full_codes = z['codes']
            assert full_codes.shape == (1200, 384)
        unique, seen = [], set()
        for t in range(10):
            key = tuple(D[:, t].tolist())
            if any(key) and key not in seen:
                unique.append(t)
                seen.add(key)
        routes = dict(onehot7=np.eye(8, dtype=np.uint8)[1:],
                      time_rows=D[:, unique].T)
        base = (173*block+29*(index//6)) % 1192
        base -= base % 4
        h = (211*block+97*(index//6)) % 1535
        for p4f2 in (False, True):
            P, F = (4, 2) if p4f2 else (8, 1)
            for route, decode in routes.items():
                name = f'{path.stem}_{route}_P{P}F{F}_p{base}_h{h}'
                meta = dict(capture=path.name, block=block, positions=list(range(base, base+P)),
                            output_channels=list(range(h, h+F)), route=route,
                            theta_source=float(np.asarray(q['theta_source'])),
                            theta_folded_into_real_weight=True, unique_time_rows=unique)
                cases.append(make_case(name, full_codes[base:base+P], W[h:h+F],
                                       decode, p4f2, True, meta=meta))
    onehot = np.eye(8, dtype=np.uint8)[1:]
    for p4f2 in (False, True):
        P, F = (4, 2) if p4f2 else (8, 1)
        suffix = f'_P{P}F{F}'
        cases.append(make_case('directed_empty'+suffix, np.zeros((P, 9), dtype=np.uint8),
                               np.full((F, 9), -128, dtype=np.int8), onehot, p4f2,
                               splits=[0, 4, 0, 4, 1, 0]))
        codes = np.ones((P, 5), dtype=np.uint8)
        w = np.tile([127, -127, -128, 127, 1], (F, 1))
        if F == 2: w[1] = [-128, 127, 1, -127, 127]
        cases.append(make_case('directed_valid_zero'+suffix, codes, w, onehot, p4f2,
                               splits=[4, 0, 1, 0]))
        multi = np.array([[(c >> r) & 1 for c in range(8)] for r in range(3)], dtype=np.uint8)
        codes = np.array([[(3*p+2*c)%8 for c in range(13)] for p in range(P)], dtype=np.uint8)
        w = np.array([[(-128, 127, 0, -1, 1)[(c+f)%5] for c in range(13)] for f in range(F)])
        cases.append(make_case('directed_tail_multirow'+suffix, codes, w, multi, p4f2,
                               splits=[3, 1, 4, 0, 2, 3]))
        rank1 = np.array([[0, 1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)
        cases.append(make_case('directed_S15_positive_boundary'+suffix,
                               np.ones((P, 129), dtype=np.uint8),
                               np.full((F, 129), 127, dtype=np.int8), rank1, p4f2))
        cases.append(make_case('directed_S15_negative_boundary'+suffix,
                               np.ones((P, 128), dtype=np.uint8),
                               np.full((F, 128), -128, dtype=np.int8), rank1, p4f2))
    with (HERE/'gp_temporal_cases.bin').open('wb') as out:
        out.write(b'GPT1')
        out.write(struct.pack('<I', len(cases)))
        for c in cases:
            name = c['name'].encode()
            out.write(struct.pack('<H', len(name))); out.write(name)
            out.write(bytes([c['p4f2'], c['rank'], c['real']]))
            table = (c['decode'].T.astype(np.uint64) << np.arange(c['rank'], dtype=np.uint64)).sum(1).astype(np.uint8)
            out.write(table.tobytes())
            out.write(struct.pack('<I', len(c['packets'])))
            for i, (n, cs, ws) in enumerate(c['packets']):
                out.write(bytes([n, i == len(c['packets'])-1]))
                out.write(cs.tobytes()); out.write(ws.tobytes())
            out.write(c['expected'].astype('<i4').tobytes())
            out.write(c['valid'].tobytes())
    summary = dict(reference='NumPy int64 dense FC1 dot per output f; independent of packet member grouping',
                   real_capture_count=len(paths), real_cases=sum(c['real'] for c in cases),
                   directed_cases=sum(not c['real'] for c in cases),
                   complete_C384_real_tiles=True,
                   state_address='(f*P+p)*7+r, rank<7 leaves explicit invalid holes',
                   precision='actual deployment W_int8 after source theta folding; not exact original FP32 ep34',
                   cases=[dict(name=c['name'], p4f2=c['p4f2'], rank=c['rank'], real=c['real'],
                               packet_count=len(c['packets']), **c['meta']) for c in cases])
    (HERE/'gp_temporal_cases.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'cases'}, indent=2))


if __name__ == '__main__': main()
