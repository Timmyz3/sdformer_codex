"""Actual FC1 -> temporal-consumer tiles for the isolated shared-adder RTL.

Both representations implement the same trained function. Configuration and
weight delivery are exposed inputs, not a simulated SRAM/DRAM subsystem.
"""
from pathlib import Path
import json
import struct
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'bn_state'))
from support_service_model import read_torch


def digits(value):
    sign = 1 if value >= 0 else -1
    n, shift, result = abs(int(value)), 0, []
    while n:
        if n & 1:
            d = 2-(n & 3)
            result.append((shift, sign*d))
            n -= d
        n >>= 1
        shift += 1
    assert sum(sign*(1 << shift) for shift, sign in result) == value
    return result


def program(coeff):
    instructions = []
    for t, row in enumerate(coeff):
        terms = [(r, shift, sign) for r, value in enumerate(row)
                 for shift, sign in digits(value)]
        if not terms:
            instructions.append((1 << 15)|(t << 11)|(1 << 10)|(1 << 9))
        for i, (r, shift, sign) in enumerate(terms):
            assert shift <= 7
            instructions.append(r|(shift << 3)|((sign < 0) << 8)|
                                ((i == 0) << 9)|((i == len(terms)-1) << 10)|(t << 11))
    assert len(instructions) <= 256
    return np.asarray(instructions, dtype='<u2')


def case(name, codes, weight, decode, coeff, tau, metadata):
    # Reference builds every continuous state from actual source contributions.
    source = np.einsum('rpc,hc->rph', decode[:, codes].astype(np.int64),
                       weight.astype(np.int64), optimize=True)
    assert source.min() >= -(1 << 14) and source.max() < (1 << 14)
    value = np.einsum('tr,rph->tph', coeff.astype(np.int64), source, optimize=True)
    expected = (value >= tau[:, None, :]).transpose(1, 0, 2).astype(np.uint8)
    residual = -tau[:, None, :]+np.zeros((10, len(codes), 96), dtype=np.int64)
    for t, row in enumerate(coeff):
        for r, c in enumerate(row):
            for shift, sign in digits(c):
                residual[t] += sign*(source[r] << shift)
                assert residual[t].min() >= -(1 << 23) and residual[t].max() < (1 << 23)
    assert np.array_equal(residual >= 0, expected.transpose(1, 0, 2))
    decode_packed = np.asarray([sum((int(decode[r, k]) & 3) << (r*2)
                                    for r in range(len(decode))) for k in range(8)], dtype='<u2')
    return dict(name=name, codes=codes, weight=weight, decode=decode_packed,
                uops=program(coeff), tau=tau, expected=expected,
                metadata=dict(**metadata, source_min=int(source.min()),
                              source_max=int(source.max()), gate_ones=int(expected.sum())))


def main():
    folder = ROOT/'algorithm/stage2_class_shift'
    consumers = json.loads((folder/'consumers.json').read_text())
    params = read_torch(ROOT/'algorithm/stage2_temporal_codes/integer_parameters.pt')
    books = np.load(ROOT/'algorithm/stage2_temporal_codes/codebooks.npz')
    basis = np.load(ROOT/'algorithm/stage2_temporal_codes/signed_basis.npz')
    cases = []
    # Four sequences, three H96 stripes, a partial final P16 tile included.
    selections = [(0, 0, 0), (1, 32, 7), (2, 1184, 15), (3, 576, 3)]
    for variant in ('integer_trained', 'power2_trained'):
        for block in range(6):
            tag = f's2b{block}'
            prefix = f'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.{block}.mlp.'
            q = params[prefix]
            B = np.asarray(consumers[variant][tag]['B_int8'], dtype=np.int64)
            tau = np.asarray(consumers[variant][tag]['tau_int32'], dtype=np.int64)
            E = basis[tag+'_coordinates_int8'].astype(np.int64).T
            D = books[tag+'_dictionary'].astype(np.int64)
            rows, seen = [], set()
            for t in range(10):
                key = tuple(D[:, t])
                if any(key) and key not in seen:
                    rows.append(t)
                    seen.add(key)
            matrix = D[basis[tag+'_selected_code_indices']].T[rows]
            inverse = np.rint(np.linalg.inv(matrix)).astype(np.int64)
            Y, A = D[:, rows].T, B @ inverse
            assert np.array_equal(A @ Y, B @ E)
            for frame, position, hidden_block in selections:
                path, = (folder/(variant+'_capture')).glob(f'v{frame:03}_*_{tag}.npz')
                codes = np.load(path)['codes'][position:position+32]
                hs = slice(hidden_block*96, (hidden_block+1)*96)
                W = q['weight_int8'][hs]
                same_function = []
                onehot = np.eye(8, dtype=np.int64)[1:]
                B7 = (B @ E)[:, 1:]
                for route, decode, coeff in [('signed', E, B), ('exact_row', Y, A),
                                              ('onehot7', onehot, B7)]:
                    name = f'{variant}/{tag}/v{frame}/p{position}/h{hidden_block}/{route}'
                    item = case(name, codes, W, decode, coeff, tau[:, hs],
                                dict(variant=variant, module=tag, route=route,
                                     capture=path.name, position_start=position,
                                     hidden_block=hidden_block, real_network=True))
                    cases.append(item)
                    same_function.append(item['expected'])
                assert all(np.array_equal(same_function[0], other)
                           for other in same_function[1:])
    # Ordinary numerical/handshake corner cases: signed -128, exact equality,
    # zero coefficient rows, zero tiles after populated tiles and single p.
    decode = np.zeros((6, 8), dtype=np.int64)
    decode[:, 1:7] = np.eye(6, dtype=np.int64)
    decode[:, 7] = [1, -1, 0, 1, 0, -1]
    coeff = np.asarray([[0]*6, [3,-64,5,1,0,-7], [-1,0,2,3,-5,0],
                        [64,0,0,0,0,0], [0,-64,0,0,0,0]]*2, dtype=np.int64)
    W = np.resize(np.asarray([-128,127,0,1,-1,7,-11], dtype=np.int8), (96,17))
    for n, zero in [(32, False), (16, True), (1, False)]:
        codes = (np.arange(n*17).reshape(n,17)%8).astype(np.uint8)
        if zero:
            codes.fill(0)
        S = np.einsum('rpc,hc->rph', decode[:, codes], W.astype(np.int64))
        U = np.einsum('tr,rph->tph', coeff, S)
        tau = U[:, 0]+np.resize(np.asarray([-1,0,1]), (10,96))
        cases.append(case(f'directed/p{n}/zero{int(zero)}', codes, W, decode,
                          coeff, tau, dict(real_network=False)))
    output = Path(__file__).resolve().parent
    with (output/'cases.bin').open('wb') as f:
        f.write(struct.pack('<I', len(cases)))
        for item in cases:
            name = item['name'].encode()
            f.write(struct.pack('<I', len(name)))
            f.write(name)
            f.write(struct.pack('<IIII', len(item['codes']), item['weight'].shape[1],
                                96, len(item['uops'])))
            for data, dtype in [(item['decode'], '<u2'), (item['uops'], '<u2'),
                                (item['tau'], '<i4'), (item['weight'].T, 'i1'),
                                (item['codes'].T, 'u1'), (item['expected'], 'u1')]:
                f.write(np.asarray(data, dtype=dtype).tobytes())
    records = [dict(name=item['name'], positions=len(item['codes']),
                    input_channels=item['weight'].shape[1], uops=len(item['uops']),
                    **item['metadata']) for item in cases]
    (output/'cases.json').write_text(json.dumps(records, indent=2)+'\n')
    print(f'{len(cases)} cases, {sum(item["expected"].size for item in cases):,} gate decisions')


if __name__ == '__main__':
    main()
