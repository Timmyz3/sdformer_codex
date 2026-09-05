#!/usr/bin/env python3
"""A stronger software codec control, not a proposed novel Huffman circuit.

Canonical per-layer INT8 Huffman, independent restart every 32/96/384 values,
whole-block raw fallback, 4-byte offset/mode directory and 256-byte codebook.
96 weights are the existing one-bank/one-source/one-output-tile demand.
Profile only; no multi-symbol decoder throughput/area/energy is assumed.
"""
import argparse
from collections import Counter
import heapq
import json
from pathlib import Path

import numpy as np
from m2262_sparse_weight_compression_screen import HW, bank_stream, code_arrays
from m2263_compressed_row_cache_probe import workload_sources, misses


def canonical_table(hist):
    heap = [(int(count), i, [i]) for i,count in enumerate(hist) if count]
    lengths = [0]*256
    heapq.heapify(heap); tie = 256
    if len(heap) == 1:
        lengths[heap[0][2][0]] = 1
    while len(heap) > 1:
        a, _, sa = heapq.heappop(heap); b, _, sb = heapq.heappop(heap)
        for i in sa+sb:
            lengths[i] += 1
        heapq.heappush(heap, (a+b, tie, sa+sb)); tie += 1
    code = 0; previous = 0; table = {}
    for bits, symbol in sorted((v,i) for i,v in enumerate(lengths) if v):
        code <<= bits-previous
        table[symbol] = (code, bits)
        code += 1; previous = bits
    assert code <= 1 << previous
    return np.array(lengths, dtype=np.int16), table


def roundtrip(values, table):
    stream = 0; size = 0
    for v in values:
        code, bits = table[int(v)]; stream = (stream << bits) | code; size += bits
    pad = (-size) % 8
    payload = (stream << pad).to_bytes((size+7)//8, 'big')
    inverse = {(bits,code):symbol for symbol,(code,bits) in table.items()}
    acc = 0; bits = 0; output = []
    for byte in payload:
        for shift in range(7, -1, -1):
            acc = (acc << 1) | ((byte >> shift) & 1); bits += 1
            if (bits,acc) in inverse:
                output.append(inverse[(bits,acc)]); acc = bits = 0
                if len(output) == len(values):
                    assert output == values.tolist()
                    return len(payload)
    raise AssertionError('Incomplete Huffman packet')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    workloads = workload_sources(); layers = []; aggregates = {}
    requests = {}
    for lid, chunks in workloads.items():
        requests[lid] = Counter(s for chunk in chunks for s in chunk)
    for lid, family, source, code in code_arrays():
        hist = np.bincount(code.view(np.uint8).ravel(), minlength=256)
        lengths, table = canonical_table(hist)
        rows = []
        cache_rows = []
        for n in (32, 96, 384):
            out = Counter(raw_bytes=code.size, codebook_bytes=256)
            demand_map = {}
            for bank in range(8):
                values = bank_stream(code, bank, 'tile-major').view(np.uint8)
                padding = (-len(values)) % n
                values = np.pad(values, (0,padding)).reshape(-1, n)
                # If a zero-padding symbol was absent, append the final block
                # as raw. All real values remain representable by the codebook.
                sizes = (lengths[values].sum(axis=1)+7)//8
                if padding and lengths[0] == 0:
                    sizes[-1] = n
                encoded = np.minimum(sizes, n)
                offset = np.cumsum(encoded)-encoded
                prefix_bits = np.cumsum(lengths[values], axis=1)
                assert int(offset[-1]) < 1 << 31
                out['payload_bytes'] += int(encoded.sum())
                out['directory_bytes'] += len(values)*4
                out['block_raw_fallbacks'] += int((sizes >= n).sum())
                out['blocks'] += len(values)
                for i in np.unique(np.linspace(0, len(values)-1, min(8,len(values)), dtype=int)):
                    if sizes[i] < n:
                        assert roundtrip(values[i], table) == sizes[i]
                        out['bitstream_roundtrip_values'] += n
                # A decoded 96-value request intersects whole restart packets.
                # No cache/reuse: this diagnoses granularity, not lower bounds.
                for s, count in requests.get(lid, {}).items():
                    if s % 8 != bank:
                        continue
                    first = (s//8)*96
                    blocks = range(first//n, (first+95)//n+1)
                    out['demanded_values'] += count*96
                    out['decoded_values_no_reuse'] += count*n*len(blocks)
                    data_rows = set(); dir_rows = set(); prefix_rows = set()
                    for b in blocks:
                        start = int(offset[b]); size = int(encoded[b])
                        data_rows.update(range(start//16, (start+size-1)//16+1))
                        dir_rows.add(b//4)
                        stop = min(n, first+96-b*n)
                        begin = max(0, first-b*n)
                        if sizes[b] >= n:
                            # Raw fallback needs neither full packet decoding
                            # nor scanning a prefix before the demanded range.
                            ps, pe = start+begin, start+stop
                            decoded_values = stop-begin
                        else:
                            ps, pe = start, start+(int(prefix_bits[b,stop-1])+7)//8
                            decoded_values = stop
                        prefix_rows.update(range(ps//16, (pe-1)//16+1))
                        out['prefix_stop_decoded_values_no_reuse'] += count*decoded_values
                    out['raw_128bit_reads'] += count*6
                    out['packet_payload_reads_no_reuse'] += count*len(data_rows)
                    out['packet_directory_reads_no_reuse'] += count*len(dir_rows)
                    out['prefix_stop_payload_reads_no_reuse'] += count*len(prefix_rows)
                    if n == 96:
                        demand_map[s] = [[('directory', v) for v in sorted(dir_rows)] +
                                         [('payload', v) for v in sorted(prefix_rows)]]
            if n == 96 and lid in workloads:
                raw_map = {s: [[('payload', (s//8)*6+j) for j in range(6)]] for s in demand_map}
                for capacity in (4,16):
                    raw = misses(workloads[lid], raw_map, capacity)
                    packed = misses(workloads[lid], demand_map, capacity)
                    assert raw['delivered_vectors'] == packed['delivered_vectors']
                    cache_rows.append(dict(cache_words_per_bank=capacity, raw=raw, compressed=packed,
                        read_ratio=packed['physical_128bit_reads']/raw['physical_128bit_reads']))
            out = dict(out)
            out['indexed_bytes'] = out['payload_bytes']+out['directory_bytes']+out['codebook_bytes']
            out['indexed_fraction'] = out['indexed_bytes']/code.size
            out['restart_values'] = n
            rows.append(out)
            total = aggregates.setdefault((family,n), Counter())
            total.update({k:v for k,v in out.items() if k not in ('restart_values','indexed_fraction')})
        layers.append(dict(layer=lid,family=family,source=source,configs=rows, restart96_shared_word_cache=cache_rows))
        print(json.dumps(dict(layer=lid,indexed_fractions=[round(r['indexed_fraction'],4) for r in rows])), flush=True)
    summary = []
    for (family,n), counts in aggregates.items():
        r = dict(counts); r.update(family=family, restart_values=n, indexed_fraction=r['indexed_bytes']/r['raw_bytes'])
        if r.get('demanded_values'):
            r['decode_amplification_no_reuse'] = r['decoded_values_no_reuse']/r['demanded_values']
            r['read_ratio_no_reuse'] = (r['packet_payload_reads_no_reuse']+r['packet_directory_reads_no_reuse'])/r['raw_128bit_reads']
            r['prefix_stop_decode_amplification_no_reuse'] = r['prefix_stop_decoded_values_no_reuse']/r['demanded_values']
            r['prefix_stop_read_ratio_no_reuse'] = (r['prefix_stop_payload_reads_no_reuse']+r['packet_directory_reads_no_reuse'])/r['raw_128bit_reads']
        summary.append(r)
    cache_summary = []
    for capacity in (4,16):
        points = [p for layer in layers for p in layer['restart96_shared_word_cache'] if p['cache_words_per_bank'] == capacity]
        row = dict(cache_words_per_bank=capacity)
        for axis in ('raw','compressed'):
            row[axis] = dict(sum((Counter(p[axis]) for p in points), Counter()))
        row['read_ratio'] = row['compressed']['physical_128bit_reads']/row['raw']['physical_128bit_reads']
        cache_summary.append(row)
    result = dict(scope='Software reference codec control; bank-local tile-major frozen code arrays; FC quantization remains candidate',
        codec='Per-layer canonical Huffman, separate 32/96/384-value restart packets, block raw fallback',
        overhead='4-byte packet offset/mode; byte-aligned payload; 256 code-length bytes per layer; unpack lookup expansion not included',
        access='Aggregate: cold no-decoded-packet-reuse sensitivity, output tile0; separate restart96 shared-word LRU replay; neither is hardware speedup',
        limitations=['Serial Huffman has variable decode work and cannot be presumed to deliver 16 weights/bank/cycle',
            'Lookup table replication, queueing, physical cache and foundry macro cost remain unmeasured',
            'Full packet raw fallback still occupies decoded width and directory entries',
            'Marginal entropy/profile does not establish novel circuitry'],
        restart96_shared_word_cache_scope='Same 4/16 128-bit compressed-payload-or-directory words per bank; no decoded buffer, decoder cycles or tags area',
        restart96_shared_word_cache=cache_summary,aggregate=summary,layers=layers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
