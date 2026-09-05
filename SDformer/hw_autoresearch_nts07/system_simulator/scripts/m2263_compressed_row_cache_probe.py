#!/usr/bin/env python3
"""M2262 follow-up: count physical reads with a finite shared word cache.

Raw and compressed axes have the same number of 128-bit cache words per bank.
Compression directory rows compete with payload rows, rather than using a
free perfect metadata cache. No cycle/PPA inference or decoder implementation.
"""
import argparse
from collections import Counter, OrderedDict
import json
from pathlib import Path

import numpy as np
from m2262_sparse_weight_compression_screen import HW, BLOCKS, code_arrays, pack_layout, bank_stream


def workload_sources():
    fix = HW/'tb_m2018/fixtures'
    result = {}
    for prefix, extent in (('m2051_ep34_tsbg_full40_s1920', 48), ('m2067_ep34_fc2_exact_continuation_s960', 192)):
        meta = json.loads((fix/(prefix+'.json')).read_text())
        words = np.array([int(x, 16) & 65535 for x in (fix/(prefix+'.memh')).read_text().split()], dtype=np.uint16)
        words = words.reshape(len(meta['rows']), 4, extent)
        for row in meta['rows']:
            for part in row.get('chunk_rows', [{'global_group_base': 0}]):
                begin = part['global_group_base']
                union = np.bitwise_or.reduce(words[row['slot'], :, begin:begin+48], axis=0)
                sources = [(begin+g)*16+b for g,m in enumerate(union)
                           for b in range(16) if int(m) >> b & 1]
                result.setdefault(str(row['layer_id']), []).append(sources)
    assert sum(map(len, result.values())) == 4320
    return result


def payload_addresses(d, element, n):
    block, pos = divmod(element, n)
    b = int(d['width'][block]); start = int(d['offset'][block])*8
    first = start+int(d['mode'][block])*8+pos*b
    rows = list(range(first//128, (first+16*b-1)//128+1)) if b else []
    if d['mode'][block] and start//128 not in rows:
        rows.insert(0, start//128)
    return [('directory', block//4)] + [('payload', v) for v in rows]


def misses(chunks, address_map, capacity):
    total = Counter()
    for sources in chunks:
        caches = [OrderedDict() for _ in range(8)]
        for source in sources:
            cache = caches[source % 8]
            for vector in address_map[source]:
                total['delivered_vectors'] += 1
                for key in vector:
                    total['lookups'] += 1
                    if key in cache:
                        total['hits'] += 1
                        cache.move_to_end(key)
                    else:
                        total['physical_128bit_reads'] += 1
                        total[key[0]+'_reads'] += 1
                        if len(cache) == capacity:
                            cache.popitem(last=False)
                        cache[key] = None
    return dict(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--layout', choices=('source-major', 'tile-major'), default='source-major')
    args = ap.parse_args()
    workloads = workload_sources(); layers = []
    for lid, family, _, code in code_arrays():
        if lid not in workloads:
            continue
        chunks = workloads[lid]
        sources = sorted(set(s for chunk in chunks for s in chunk))
        assert sources[-1] < code.shape[0]
        nout = code.shape[1] if args.layout == 'source-major' else 96
        raw_map = {s: [[('payload', (s//8)*(nout//16)+j)] for j in range(6)] for s in sources}
        raw = {cap: misses(chunks, raw_map, cap) for cap in (4, 16)}
        points = []
        for n in BLOCKS:
            banks = []; storage = 0
            for bank in range(8):
                values = bank_stream(code, bank, args.layout)
                values = np.pad(values, (0, (-len(values)) % n))
                d = pack_layout(values, n)
                banks.append(d)
                storage += int(d['size'].sum())+4*d['blocks']
            packed_map = {s: [payload_addresses(banks[s % 8], (s//8)*nout+j*16, n)
                              for j in range(6)] for s in sources}
            for cap in (4, 16):
                compressed = misses(chunks, packed_map, cap)
                assert compressed['delivered_vectors'] == raw[cap]['delivered_vectors']
                # Storage-only selection uses weights, not validation speedup.
                bypass = storage >= code.size
                selected = raw[cap] if bypass else compressed
                points.append(dict(block_values=n, cache_words_per_bank=cap,
                    raw=raw[cap], compressed=compressed,
                    packed_indexed_bytes=storage, raw_bytes=code.size,
                    whole_layer_raw_bypass=bypass, storage_selected=selected,
                    forced_compressed_read_ratio=compressed['physical_128bit_reads']/raw[cap]['physical_128bit_reads'],
                    storage_selected_read_ratio=selected['physical_128bit_reads']/raw[cap]['physical_128bit_reads']))
        layers.append(dict(layer=lid, family=family, chunks=len(chunks), points=points))
        print(json.dumps(dict(layer=lid, read_ratios=[round(p['forced_compressed_read_ratio'], 4) for p in points])), flush=True)
    aggregate = []
    for n in BLOCKS:
        for cap in (4, 16):
            points = [p for layer in layers for p in layer['points'] if p['block_values'] == n and p['cache_words_per_bank'] == cap]
            r = dict(block_values=n, cache_words_per_bank=cap)
            for axis in ('raw', 'compressed', 'storage_selected'):
                r[axis] = dict(sum((Counter(p[axis]) for p in points), Counter()))
            for axis in ('compressed', 'storage_selected'):
                r[axis+'_read_ratio'] = r[axis]['physical_128bit_reads']/r['raw']['physical_128bit_reads']
            aggregate.append(r)
    result = dict(scope='CPU physical-read traffic, output tile zero, same 4320 cold G48 chunks',
        layout=args.layout,
        storage_budget='4 or 16 shared 128-bit data+directory words per bank for both axes; cache tags/control unpriced',
        ordering='Original group/source bank/six-output-slice order; no task regrouping or future knowledge',
        raw_bypass='Per whole layer when indexed storage >= raw, selected from weights only; no access-profile oracle',
        limitations=['Same payload budget, not same physical area; cache tags and directory type discrimination remain',
            'Cache hits still consume access energy; decoder and cache lookup latency unmeasured',
            'Original 4-row C2 cache and pipeline costs remain unchanged and are not newly counted here',
            'No macro reduction or energy/speedup claim from physical-read traffic'],
        layers=layers, aggregate=aggregate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
