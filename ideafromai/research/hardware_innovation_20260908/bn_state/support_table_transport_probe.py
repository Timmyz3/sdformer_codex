"""Actual forced-code table and broadcast requests: exact format controls.

No training, pruning, cycles or energy result. All formats reconstruct all table
entries. Counts are for the same original H96 coefficient rows and B32/T10 tile.
"""
import json
from pathlib import Path

import numpy as np

from support_service_model import CAP, read_torch

HERE = Path(__file__).resolve().parent


def signed_width(values):
    low, high = int(values.min()), int(values.max())
    return max(1, high.bit_length()+1 if high >= 0 else 1,
               (-low-1).bit_length()+1 if low < 0 else 1)


def pack_fields(fields):
    payload, shift = 0, 0
    for value, width in fields:
        payload |= (int(value) & ((1 << width)-1)) << shift
        shift += width
    return payload, shift


def decode_signed(payload, width, count):
    mask = (1 << width)-1
    values = [(payload >> (i*width)) & mask for i in range(count)]
    return np.array([v-(1 << width) if v & (1 << (width-1)) else v for v in values])


def encode_row(row, mode):
    if mode.startswith('fixed'):
        width = int(mode[5:]); data, bits = pack_fields([(v, width) for v in row])
        recovered = decode_signed(data, width, 96)
    elif mode == 'low8_exception2':
        # The mask marks values that cannot use ordinary sign extension.
        # Each exception supplies the actual two high bits of signed INT10.
        low = row & 255
        exception = (row < -128) | (row > 127)
        mask = sum(int(v) << i for i, v in enumerate(exception))
        fields = [(v, 8) for v in low] + [(mask, 96)]
        fields += [((int(v) & 1023) >> 8, 2) for v, yes in zip(row, exception) if yes]
        data, bits = pack_fields(fields)
        recovered = np.where(low < 128, low, low-256).copy()
        cursor = 864
        for i in np.flatnonzero(exception):
            top = (data >> cursor) & 3; cursor += 2
            value = int(low[i]) | (top << 8)
            recovered[i] = value-1024 if value & 512 else value
    else:
        block = int(mode.removeprefix('block'))
        blocks = row.reshape(-1, block)
        widths = [signed_width(x) for x in blocks]
        fields = [(w, 4) for w in widths]
        fields += [(v, w) for x, w in zip(blocks, widths) for v in x]
        data, bits = pack_fields(fields)
        cursor = 4*len(widths); pieces = []
        for w in widths:
            pieces.append(decode_signed(data >> cursor, w, block))
            cursor += w*block
        recovered = np.concatenate(pieces)
    if not np.array_equal(recovered, row):
        raise RuntimeError(f'{mode} failed exact table reconstruction')
    return bits, (bits+127)//128


def main():
    dictionary = np.load(CAP/'dictionary.npy').astype(np.int64)
    weight = read_torch(CAP/'forced_code_weight_int8.pt').astype(np.int64)
    values = np.einsum('gkc,hgc->gkh', dictionary, weight.reshape(384, 6, 16))
    codes = [(g, k) for g in range(6) for k in range(16) if dictionary[g, k].any()]
    table = np.stack([values[g, k] for g, k in codes]).reshape(90, 4, 96)
    modes = ['fixed16', 'fixed12', 'fixed10', 'low8_exception2', 'block8', 'block16']
    row_bits, row_words = {}, {}
    for mode in modes:
        encoded = np.array([encode_row(r, mode) for r in table.reshape(-1, 96)])
        row_bits[mode] = encoded[:, 0].reshape(90, 4)
        row_words[mode] = encoded[:, 1].reshape(90, 4)
    dict_words = (dictionary*(1 << np.arange(16))).sum(-1)
    lookup = np.full((6, 65536), -1, dtype=np.int16)
    for g in range(6):
        lookup[g, dict_words[g]] = np.arange(16)
    run = json.loads((CAP/'run.json').read_text())
    result = dict(
        scope='one existing forced-code S0 FC1 student, actual INT8 W and train32 support dictionary; ten complete captured input frames',
        arithmetic='L[g,k,h]=sum_c D[g,k,c]*W[h,g,c], W already carries source theta and dyadic scale; no entry or gate changed',
        D_rank=[int(np.linalg.matrix_rank(g)) for g in dictionary],
        D_active_columns=[int(g.any(0).sum()) for g in dictionary],
        L_nonzero_code_shape=[90, 384], L_min=int(table.min()), L_max=int(table.max()),
        scalar_zero_fraction=float((table == 0).mean()),
        H8_zero_word_fraction=float((table.reshape(90, 48, 8) == 0).all(-1).mean()),
        INT8_scalar_fit_fraction=float(((table >= -128) & (table < 128)).mean()),
        exact_reconstruction_values_per_format=int(table.size),
        table_layout={m:dict(payload_bits=int(row_bits[m].sum()),
                            independently_aligned_128bit_bytes=int(row_words[m].sum())*16,
                            row_words_hist={str(w):int((row_words[m] == w).sum()) for w in np.unique(row_words[m])}) for m in modes},
        metadata='variable-length formats additionally need 360 row offsets/lengths, conservatively4 bytes each =1440 B; per-request uncached directory128 word is reported separately',
        exclusions=['row payload transfers are not a completion schedule, throughput or energy measurement',
                    'variable packing needs address/length metadata, extraction and exception/width decode; none is free',
                    'generic exact compression is a baseline, not a proposed novelty',
                    'zero-word opportunity is measured on current W; no L-domain pruning or representability-changing training was performed'],
        frames=[])
    all_presence = np.zeros((6, 16), dtype=np.int64)
    all_tokens = np.zeros((6, 16), dtype=np.int64)
    for file in run['validation_files'][:10]:
        path = CAP/('forced_code_'+Path(file).stem+'_source.npz')
        with np.load(path) as sample:
            words = np.ascontiguousarray(sample['gate_bits']).view('<u2').reshape(10, 19200, 6)
        ids = np.stack([lookup[g, words[:, :, g]] for g in range(6)], axis=-1)
        if np.any(ids < 0):
            raise RuntimeError('The forced-code trace contains a source outside its dictionary')
        counts = np.zeros((6, 16), dtype=np.int64)
        token_counts = np.zeros((6, 16), dtype=np.int64)
        tiles = ids.reshape(10, 600, 32, 6)
        for g in range(6):
            for k in range(16):
                counts[g, k] = (tiles[:, :, :, g] == k).any(axis=(0, 2)).sum()
                token_counts[g, k] = (ids[:, :, g] == k).sum()
        freq = np.array([counts[g, k] for g, k in codes])[:, None]
        requests = int(freq.sum())*4
        metrics = {m:dict(payload_128bit_read_bytes=int((freq*row_words[m]).sum())*16,
                          with_uncached_row_directory_bytes=int((freq*row_words[m]).sum())*16+(requests*16 if not m.startswith('fixed') else 0)) for m in modes}
        result['frames'].append(dict(file=file, H96_broadcast_row_requests=requests, formats=metrics))
        all_presence += counts; all_tokens += token_counts
    result['code_broadcast_tile_presence'] = all_presence.tolist()
    result['code_token_frequency'] = all_tokens.tolist()
    result['mean_per_frame'] = {m:{k:sum(f['formats'][m][k] for f in result['frames'])/10 for k in result['frames'][0]['formats'][m]} for m in modes}
    result['mean_H96_broadcast_row_requests'] = sum(f['H96_broadcast_row_requests'] for f in result['frames'])/10
    (HERE/'support_table_transport_probe.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['D_rank','D_active_columns','H8_zero_word_fraction','mean_per_frame']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
