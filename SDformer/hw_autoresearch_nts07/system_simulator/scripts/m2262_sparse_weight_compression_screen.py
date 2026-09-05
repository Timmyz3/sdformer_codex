#!/usr/bin/env python3
"""Opportunity screen, not an EBPC implementation or a hardware result.

Keep the existing INT8 arrays unchanged. Profile ordinary signed-width and
minimum-plus-offset packing at bank-local 16/32/64/128-byte blocks. Include
4-byte random-access directory entries, byte packing and 128-bit read rows.
The FC codewords remain the M2251 candidate, not an accuracy-admitted model.
No training, quantization search, EDA, or paper edits are performed.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

HW = Path(__file__).resolve().parents[2]
BLOCKS = (16, 32, 64, 128)


def pack_layout(bank_words, n):
    """One independent stream per bank; never merge values across banks."""
    x = bank_words.astype(np.int16).reshape(-1, n)
    lo, hi = x.min(axis=1), x.max(axis=1)
    sb = np.zeros(len(x), dtype=np.int16)
    for b in range(1, 9):
        need = (lo < -(1 << (b-1))) | (hi >= (1 << (b-1)))
        sb[need] = b + 1
    sb[(lo != 0) | (hi != 0)] = np.maximum(1, sb[(lo != 0) | (hi != 0)])
    ub = np.ceil(np.log2((hi-lo).astype(np.float64)+1)).astype(np.int16)
    signed_bytes = (n*sb+7)//8
    base_bytes = 1+(n*ub+7)//8
    mode = base_bytes < signed_bytes
    width = np.where(mode, ub, sb)
    base = np.where(mode, lo, 0)
    size = np.where(mode, base_bytes, signed_bytes).astype(np.int64)
    offset = np.cumsum(size)-size
    if len(offset) and offset[-1] >= 1 << 24:
        raise ValueError('24-bit byte directory offset insufficient for bank')
    # Verify every value through a separately expressed integer decoder.
    mask = (1 << width[:, None])-1
    encoded = (x-base[:, None]) & mask
    signed = encoded - (((encoded >> np.maximum(width[:, None]-1, 0)) & 1)
                        << width[:, None])
    decoded = np.where(mode[:, None], encoded+base[:, None], signed)
    assert np.array_equal(decoded, x)
    # Actual bitstream round trips, separate from the vectorized width check.
    tested = np.unique(np.linspace(0, len(x)-1, min(64, len(x)), dtype=int))
    for i in tested:
        b = int(width[i]); minimum = int(base[i]); isbase = bool(mode[i])
        payload = sum(((int(v)-minimum) & ((1 << b)-1)) << (j*b)
                      for j, v in enumerate(x[i]))
        serialized = (bytes([minimum & 255]) if isbase else b'') + payload.to_bytes(
            (n*b+7)//8, 'little')
        assert len(serialized) == size[i]
        prefix = int(mode[i])
        payload = int.from_bytes(serialized[prefix:], 'little')
        restored = []
        for j in range(n):
            v = (payload >> (j*b)) & ((1 << b)-1)
            if isbase:
                v += int.from_bytes(serialized[:1], 'little', signed=True)
            elif b and v & (1 << (b-1)):
                v -= 1 << b
            restored.append(v)
        assert restored == x[i].tolist()
    return dict(width=width, base=base, mode=mode, size=size, offset=offset,
                blocks=len(x), roundtrip_values=x.size,
                bitstream_roundtrip_values=len(tested)*n)


def code_arrays():
    exported = HW/'results/m2042_ep34_s40_eight_operator_int8_export_r1_20260902'
    manifest = json.loads((exported/'result.json').read_text())
    for row in manifest['layers']:
        path = exported/row['hardware_code_file']
        x = np.load(path)
        yield f"export_{row['global_ordinal']}", row['family'], str(path), x.reshape(-1, x.shape[-1])
    import torch
    torch.set_num_threads(1)
    checkpoint = HW/'system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)['model_state_dict']
    layers = json.loads((HW/'results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901/layers.json').read_text())['layers']
    for row in layers:
        if row['operator'] != 'Linear':
            continue
        key = row['module_name']+'.weight'
        full = state[key].float()
        maxima = full.abs().amax(dim=1)
        scale = torch.pow(2.0, torch.ceil(torch.log2(torch.clamp(maxima/127, min=2.0**-30))))
        codes = torch.round(full/scale[:, None]).clamp(-127, 127).to(torch.int8).numpy()
        assert codes.shape == (row['output_channels'], row['input_channels'])
        # Verify compatibility with all three already exported power tiles.
        for window, lid in (('low', 15), ('median', 30), ('high', 13)):
            if row['layer_id'] == lid:
                old = np.array([int(v, 16) for v in (HW/f'results/m2251_fc_power_weight_inputs/{window}_weights.memh').read_text().split()], dtype=np.uint8).view(np.int8)
                packed = codes[:96].T.reshape(48, 2, 8, 6, 16).transpose(0, 1, 3, 2, 4).ravel()
                assert np.array_equal(packed, old)
        yield str(row['layer_id']), 'fc_candidate_'+key.rsplit('.', 2)[-2], str(checkpoint)+'::'+key, codes.T


def bank_stream(code, bank, layout):
    x = code[bank::8]
    if layout == 'tile-major':
        assert x.shape[1] % 96 == 0
        x = x.reshape(len(x), -1, 96).transpose(1, 0, 2)
    return np.ascontiguousarray(x).ravel()


def profile(layer_id, family, source, code, layout_name='source-major'):
    assert code.dtype == np.int8 and code.shape[1] % 16 == 0
    hist = np.bincount(code.view(np.uint8).ravel(), minlength=256)
    p = hist[hist > 0]/code.size
    entry = dict(layer=layer_id, family=family, source=source, shape_source_output=list(code.shape),
                 raw_bytes=code.size, zero_fraction=float((code == 0).mean()),
                 marginal_entropy_bits=float(-(p*np.log2(p)).sum()), configurations=[])
    layouts = {}
    for n in BLOCKS:
        point = Counter()
        widths = Counter()
        banks = []
        for bank in range(8):
            raw = bank_stream(code, bank, layout_name)
            padding = (-len(raw)) % n
            point['bank_tail_padding_values'] += padding
            raw = np.pad(raw, (0, padding))
            layout = pack_layout(raw, n)
            point.update({k: layout[k] for k in ('blocks', 'roundtrip_values', 'bitstream_roundtrip_values')})
            point['payload_bytes'] += int(layout['size'].sum())
            point['directory_bytes'] += 4*layout['blocks']
            point['dense_payload_128bit_rows'] += (int(layout['size'].sum())+15)//16
            point['independently_aligned_payload_bytes'] += int(((layout['size']+15)//16*16).sum())
            point['base_offset_blocks'] += int(layout['mode'].sum())
            widths.update({str(i): int(v) for i,v in enumerate(np.bincount(layout['width'], minlength=9))})
            banks.append(layout)
        point = dict(point)
        point.update(block_values=n, width_histogram=dict(widths),
            dense_indexed_bytes=point['payload_bytes']+point['directory_bytes'],
            aligned_indexed_bytes=point['independently_aligned_payload_bytes']+point['directory_bytes'])
        point['dense_indexed_fraction'] = point['dense_indexed_bytes']/code.size
        point['aligned_indexed_fraction'] = point['aligned_indexed_bytes']/code.size
        # Whole-layer raw bypass is available; the directory is not free.
        point['with_layer_raw_bypass_fraction'] = min(1., point['dense_indexed_fraction'])
        entry['configurations'].append(point)
        layouts[n] = banks
    return entry, layouts


def request_counts():
    """Exact source address incl. FC2 continuation base, output tile zero only.

    The existing chunks() drops global_group_base, so reconstruct it here.
    Count each group's B4 union once, equivalent to mask-aware group-demand
    and cofill at the original row cache. No cross-chunk warm-cache inference.
    """
    fix = HW/'tb_m2018/fixtures'
    counts = {}; chunks = 0
    for prefix, extent in (('m2051_ep34_tsbg_full40_s1920', 48), ('m2067_ep34_fc2_exact_continuation_s960', 192)):
        meta = json.loads((fix/(prefix+'.json')).read_text())
        words = np.array([int(x, 16) & 65535 for x in (fix/(prefix+'.memh')).read_text().split()], dtype=np.uint16)
        words = words.reshape(len(meta['rows']), 4, extent)
        for row in meta['rows']:
            hist = counts.setdefault(str(row['layer_id']), Counter())
            for part in row.get('chunk_rows', [{'global_group_base': 0}]):
                begin = part['global_group_base']; chunks += 1
                union = np.bitwise_or.reduce(words[row['slot'], :, begin:begin+48], axis=0)
                for group, mask in enumerate(union):
                    for source in range(16):
                        if int(mask) >> source & 1:
                            hist[(begin+group)*16+source] += 1
    return counts, chunks


def access_profile(code, layouts, counts, layout_name='source-major'):
    """128-bit payload reads without an added compressed-row reuse buffer.

    Header cache perfect vs absent are stated bounds, not free real metadata.
    Byte-packed blocks may cross 128-bit rows; even one requested vector can
    require two payload reads. A base byte is read if not in those rows.
    """
    nout = code.shape[1] if layout_name == 'source-major' else 96
    result = []
    baseline = sum(counts.values())*6
    for n, banks in layouts.items():
        payload_reads = directory_reads = base_reads = raw_blocks = 0
        for source, count in counts.items():
            assert source < code.shape[0]
            bank = source % 8; d = banks[bank]
            for slice_id in range(6):
                element = (source//8)*nout + slice_id*16
                block, pos = divmod(element, n)
                b = int(d['width'][block]); start = int(d['offset'][block])*8
                firstbit = start+int(d['mode'][block])*8+pos*b
                addresses = set(range(firstbit//128, (firstbit+16*b-1)//128+1)) if b else set()
                payload_reads += count*len(addresses)
                if d['mode'][block] and start//128 not in addresses:
                    base_reads += count
                directory_reads += count
                raw_blocks += count*int(b == 8 and not d['mode'][block])
        data = payload_reads+base_reads
        result.append(dict(block_values=n, uncompressed_128bit_reads=baseline,
            packed_payload_128bit_reads=data, directory_lookups=directory_reads,
            perfect_header_cache_read_ratio=data/baseline,
            absent_header_cache_total_read_ratio=(data+directory_reads)/baseline,
            requests_in_raw_width8_blocks=raw_blocks))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--layout', choices=('source-major', 'tile-major'), default='source-major')
    args = ap.parse_args()
    # Signed edges, constant blocks, and full dynamic range.
    for n in BLOCKS:
        for row in (np.zeros(n, dtype=np.int8), np.full(n, -128, dtype=np.int8),
                    np.resize(np.array([-128, 127, -1, 0, 1], dtype=np.int8), n)):
            pack_layout(row, n)
    counts, chunk_count = request_counts()
    rows = []; access = []; groups = {}
    for lid, family, source, code in code_arrays():
        row, layouts = profile(lid, family, source, code, args.layout)
        rows.append(row)
        if lid in counts:
            access.append(dict(layer=lid, rows=access_profile(code, layouts, counts[lid], args.layout)))
        aggregate = groups.setdefault(family, {str(n): Counter() for n in BLOCKS})
        for point in row['configurations']:
            out = aggregate[str(point['block_values'])]
            out['raw_bytes'] += code.size
            for metric in ('dense_indexed_bytes', 'aligned_indexed_bytes', 'payload_bytes', 'directory_bytes', 'roundtrip_values', 'bitstream_roundtrip_values'):
                out[metric] += point[metric]
        print(json.dumps(dict(layer=lid, family=family, bytes=code.size,
            entropy=row['marginal_entropy_bits'], indexed_fraction=[round(p['dense_indexed_fraction'], 4) for p in row['configurations']])), flush=True)
    for family in groups.values():
        for point in family.values():
            point['dense_indexed_fraction'] = point['dense_indexed_bytes']/point['raw_bytes']
            point['aligned_indexed_fraction'] = point['aligned_indexed_bytes']/point['raw_bytes']
    access_totals = {}
    for n in BLOCKS:
        fields = ('uncompressed_128bit_reads', 'packed_payload_128bit_reads', 'directory_lookups', 'requests_in_raw_width8_blocks')
        totals = {k: sum(p[k] for a in access for p in a['rows'] if p['block_values'] == n) for k in fields}
        totals['perfect_header_cache_read_ratio'] = totals['packed_payload_128bit_reads']/totals['uncompressed_128bit_reads']
        totals['absent_header_cache_total_read_ratio'] = (totals['packed_payload_128bit_reads']+totals['directory_lookups'])/totals['uncompressed_128bit_reads']
        access_totals[str(n)] = totals
    assert chunk_count == 4320
    assert set(counts) <= {r['layer'] for r in access}
    result = dict(scope='Bank-local fixed INT8 codewords; no model modification, no RTL/cycle/energy result',
        quantization_scope={'conv_decoder': 'Existing M2042 code arrays, unmodified; consult M2045 for accuracy boundary',
            'FC': 'Same M2251 power-of-two per-output candidate INT8 rule; full FC AEE not admitted'},
        formats='Signed width or min+unsigned offset; lossless elementary baselines, NOT EBPC',
        directory='Per bank/block 24-bit byte offset + mode/width within 32 bits; base byte in payload',
        memory='128-bit logical read rows; no foundry capacity/energy projection or decoder timing',
        layout=args.layout,
        access_scope='4320 cold G48 chunks with global FC2 continuation offsets; output tile zero only, mask-aware B4 union baseline',
        aggregate=groups, access_totals=access_totals, layers=rows, accesses=access,
        limitations=['Dense packing needs address translation and possibly two reads per vector',
            'Header cache bounds do not establish same-area or same-total-SRAM performance',
            'No compressed-row reuse buffer is modeled; one could coalesce adjacent demands',
            'Marginal entropy is a statistic, not a universal compression lower bound',
            'Failing these two codecs does not kill entropy/bitplane codecs or the research direction'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(aggregate=groups, access_totals=access_totals), indent=2))


if __name__ == '__main__':
    main()
