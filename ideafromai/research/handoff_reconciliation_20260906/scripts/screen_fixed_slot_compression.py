#!/usr/bin/env python3.12
"""LCP-style fixed-slot control versus dense restart-Huffman and a cache control.

CPU byte identity, storage and read-traffic screen only. Not an RTL mechanism,
cycle/PPA estimate or accuracy admission for the existing FC INT8 candidate.
"""
import argparse
from collections import Counter, OrderedDict
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path('/home/zhumd/work')
HW = ROOT / 'sdformer_codex/SDformer/hw_autoresearch_nts07'
sys.path.insert(0, str(HW / 'system_simulator/scripts'))
sys.path.insert(0, str(ROOT / 'ideafromai/research/mechanism_rebuild_gh_20260906/scripts'))
from checkpoint_numpy import read_checkpoint
from m2262_sparse_weight_compression_screen import bank_stream
from m2263_compressed_row_cache_probe import workload_sources, misses
from m2264_restart_huffman_screen import canonical_table


def fc_codes():
    checkpoint = HW / 'system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth'
    state = read_checkpoint(checkpoint)['model_state_dict']
    meta = json.loads((HW / 'results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901/layers.json').read_text())['layers']
    for row in meta:
        if row['operator'] != 'Linear':
            continue
        key = row['module_name'] + '.weight'
        full = state[key].astype(np.float32)
        maxima = np.abs(full).max(axis=1)
        scale = np.power(np.float32(2), np.ceil(np.log2(np.maximum(
            maxima / np.float32(127), np.float32(2.0**-30)))))
        codes = np.rint(full / scale[:, None]).clip(-127, 127).astype(np.int8)
        assert codes.shape == (row['output_channels'], row['input_channels'])
        for window, lid in (('low', 15), ('median', 30), ('high', 13)):
            if row['layer_id'] == lid:
                old = np.array([int(v, 16) for v in (HW / f'results/m2251_fc_power_weight_inputs/{window}_weights.memh').read_text().split()], dtype=np.uint8).view(np.int8)
                packed = codes[:96].T.reshape(48, 2, 8, 6, 16).transpose(0, 1, 3, 2, 4).ravel()
                assert np.array_equal(packed, old), ('exported tile mismatch', lid)
        yield str(row['layer_id']), key, codes.T


def decoder_tree(serialized_lengths):
    """Build a separate prefix decoder using only the serialized code lengths."""
    nodes = [[-1, -1, -1]]
    code = previous = 0
    for bits, symbol in sorted((b, s) for s, b in enumerate(serialized_lengths) if b):
        code <<= bits - previous
        node = 0
        for shift in range(bits - 1, -1, -1):
            branch = (code >> shift) & 1
            if nodes[node][branch] == -1:
                nodes[node][branch] = len(nodes)
                nodes.append([-1, -1, -1])
            node = nodes[node][branch]
        assert nodes[node] == [-1, -1, -1]
        nodes[node][2] = symbol
        code += 1
        previous = bits
    return nodes


def decode_packet(main, pool, packet, tree):
    off = packet * 80
    header = int.from_bytes(main[off:off + 4], 'little')
    if header >> 31:
        pointer = header & 0x7fffffff
        if pointer % 16 or pointer + 96 > len(pool):
            raise ValueError('invalid exception pointer')
        return bytes(pool[pointer:pointer + 96])
    if header:
        raise ValueError('nonzero inline header')
    output = bytearray()
    node = 0
    for byte in main[off + 4:off + 80]:
        for shift in range(7, -1, -1):
            node = tree[node][(byte >> shift) & 1]
            if node < 0:
                raise ValueError('invalid prefix')
            if tree[node][2] >= 0:
                output.append(tree[node][2])
                if len(output) == 96:
                    return bytes(output)
                node = 0
    raise ValueError('short inline packet')


def fixed_layout(packets, sizes, table, tree):
    main = bytearray(len(packets) * 80)
    pool = bytearray()
    pointers = np.full(len(packets), -1, dtype=np.int64)
    for q, values in enumerate(packets):
        off = q * 80
        if sizes[q] > 76:
            pointer = len(pool)
            assert pointer % 16 == 0 and pointer < 2**31
            pointers[q] = pointer
            main[off:off + 4] = (0x80000000 | pointer).to_bytes(4, 'little')
            pool.extend(values.tobytes())
        else:
            word = total = 0
            for v in values:
                code, bits = table[int(v)]
                word = (word << bits) | code
                total += bits
            payload = (word << ((-total) % 8)).to_bytes((total + 7) // 8, 'big')
            assert len(payload) == sizes[q]
            main[off + 4:off + 4 + len(payload)] = payload
    # Actual serialized bytes, all packets, all output tiles, separately decoded.
    for q, expected in enumerate(packets):
        assert decode_packet(main, pool, q, tree) == expected.tobytes(), q
    return len(main), len(pool), pointers


def reserved_directory_misses(chunks, address_map, capacity):
    """One reserved directory word plus capacity-1 payload words, per bank."""
    total = Counter()
    for sources in chunks:
        caches = [[OrderedDict(), OrderedDict()] for _ in range(8)]
        for source in sources:
            for vector in address_map[source]:
                total['delivered_vectors'] += 1
                for key in vector:
                    is_directory = key[0] == 'directory'
                    cache = caches[source % 8][int(is_directory)]
                    limit = 1 if is_directory else capacity - 1
                    total['lookups'] += 1
                    if key in cache:
                        total['hits'] += 1
                        cache.move_to_end(key)
                    else:
                        total['physical_128bit_reads'] += 1
                        total[key[0] + '_reads'] += 1
                        if len(cache) == limit:
                            cache.popitem(last=False)
                        cache[key] = None
    return dict(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    started = time.monotonic()
    workloads = workload_sources()
    sealed_path = HW / 'results/m2264_restart_huffman_screen/result.json'
    sealed = {r['layer']: r for r in json.loads(sealed_path.read_text())['layers']}
    layers = []
    # Exercise both representations even if real packets all happen to overflow.
    test_values = np.tile(np.arange(4, dtype=np.uint8), 24).reshape(1, 96)
    test_lengths, test_table = canonical_table(np.bincount(test_values.ravel(), minlength=256))
    test_tree = decoder_tree(bytes(test_lengths.astype(np.uint8)))
    fixed_layout(test_values, np.array([24]), test_table, test_tree)
    fixed_layout(test_values, np.array([96]), test_table, test_tree)
    malformed = bytearray(80)
    for header in (1, 0x80000001, 0x80000060):
        malformed[:4] = header.to_bytes(4, 'little')
        try:
            decode_packet(malformed, bytearray(96), 0, test_tree)
        except ValueError:
            pass
        else:
            raise AssertionError('malformed header accepted')
    for lid, key, code in fc_codes():
        counts = Counter(s for chunk in workloads[lid] for s in chunk)
        lengths, table = canonical_table(np.bincount(code.view(np.uint8).ravel(), minlength=256))
        tree = decoder_tree(bytes(lengths.astype(np.uint8)))
        dense_map = {}
        fixed_map = {}
        stats = Counter(raw_bytes=int(code.size), common_layer_descriptor_bytes=128,
                        codebook_bytes=256)
        size_hist = Counter()
        for bank in range(8):
            packets = bank_stream(code, bank, 'tile-major').view(np.uint8).reshape(-1, 96)
            sizes = (lengths[packets].sum(axis=1) + 7) // 8
            encoded = np.minimum(sizes, 96)
            offsets = np.cumsum(encoded) - encoded
            main_bytes, pool_bytes, pointers = fixed_layout(packets, sizes, table, tree)
            stats.update(blocks=len(packets), exception_blocks=int((pointers >= 0).sum()),
                         payload_bytes=int(encoded.sum()), directory_bytes=len(packets) * 4,
                         block_raw_fallbacks=int((sizes >= 96).sum()),
                         fixed_main_bytes=main_bytes, fixed_pool_bytes=pool_bytes,
                         dense_alignment_bytes=int((-int(encoded.sum())) % 16),
                         serialized_readback_bytes=int(packets.size))
            size_hist.update(int(s) for s in sizes)
            for source, times in counts.items():
                if source % 8 != bank:
                    continue
                q = source // 8
                start = int(offsets[q])
                size = int(encoded[q])
                dense_map[source] = [[('directory', q // 4)] + [
                    ('payload', r) for r in range(start // 16, (start + size - 1) // 16 + 1)]]
                if pointers[q] >= 0:
                    pointer = int(pointers[q])
                    fixed_map[source] = [[('main', q * 5)] + [
                        ('pool', pointer // 16 + j) for j in range(6)]]
                    stats['exception_requests'] += times
                else:
                    fixed_map[source] = [[('main', q * 5 + j) for j in range(5)]]
                stats['requests'] += times
                stats['requested_huffman_bytes'] += times * int(sizes[q])
        old = next(p for p in sealed[lid]['configs'] if p['restart_values'] == 96)
        for field in ('raw_bytes', 'payload_bytes', 'directory_bytes', 'blocks', 'block_raw_fallbacks'):
            assert stats[field] == old[field], ('sealed layout mismatch', lid, field, stats[field], old[field])
        raw_map = {s: [[('payload', (s // 8) * 6 + j) for j in range(6)]] for s in counts}
        raw_storage = stats['raw_bytes'] + 128
        dense_storage = stats['payload_bytes'] + stats['directory_bytes'] + stats['dense_alignment_bytes'] + 256 + 128
        fixed_storage = stats['fixed_main_bytes'] + stats['fixed_pool_bytes'] + 256 + 128
        bypass = fixed_storage >= raw_storage
        traffic = []
        for cap in (4, 16):
            raw = misses(workloads[lid], raw_map, cap)
            dense = misses(workloads[lid], dense_map, cap)
            old_cache = next(p for p in sealed[lid]['restart96_shared_word_cache'] if p['cache_words_per_bank'] == cap)
            assert raw == old_cache['raw'] and dense == old_cache['compressed'], ('sealed traffic mismatch', lid, cap)
            reserved = reserved_directory_misses(workloads[lid], dense_map, cap)
            fixed = misses(workloads[lid], fixed_map, cap)
            selected = raw if bypass else fixed
            traffic.append(dict(cache_words_per_bank=cap, raw=raw, dense_shared=dense,
                                dense_reserved_directory=reserved, fixed_forced=fixed,
                                fixed_storage_selected=selected))
        row = dict(layer=lid, weight=key, chunks=len(workloads[lid]), counts=dict(stats),
                   huffman_packet_size_histogram=dict(sorted(size_hist.items())),
                   storage=dict(raw=raw_storage, dense=dense_storage, fixed_forced=fixed_storage,
                                fixed_storage_selected=raw_storage if bypass else fixed_storage),
                   fixed_whole_layer_raw_bypass=bool(bypass), traffic=traffic)
        layers.append(row)
        print(json.dumps(dict(layer=lid, exceptions=round(stats['exception_blocks'] / stats['blocks'], 5),
                              fixed_capacity_ratio=round(fixed_storage / raw_storage, 5),
                              reserved_read_ratios=[round(p['dense_reserved_directory']['physical_128bit_reads'] / p['raw']['physical_128bit_reads'], 5) for p in traffic],
                              elapsed_s=round(time.monotonic() - started, 1))), flush=True)
    total_counts = sum((Counter(r['counts']) for r in layers), Counter())
    total_storage = sum((Counter(r['storage']) for r in layers), Counter())
    aggregate = []
    axes = ('raw', 'dense_shared', 'dense_reserved_directory', 'fixed_forced', 'fixed_storage_selected')
    for cap in (4, 16):
        points = [p for r in layers for p in r['traffic'] if p['cache_words_per_bank'] == cap]
        aggregate.append(dict(cache_words_per_bank=cap, **{
            axis: dict(sum((Counter(p[axis]) for p in points), Counter())) for axis in axes}))
    assert len(layers) == 24 and total_counts['requests'] == 267405
    result = dict(status='CPU_STORAGE_AND_128BIT_READ_COUNTS_ONLY',
        inputs=dict(checkpoint='motion_c12_ep34_live93_checkpoint_epoch34.pth', sealed_control=str(sealed_path),
                    fc_precision='Existing M2251 INT8 deployment candidate, not frozen FP software admission',
                    scope='24 FC weight tensors, all tiles for storage; 4320 cold G48 chunks, output tile0 for traffic'),
        format=dict(slot_bytes=80, inline_header_bytes=4, max_inline_payload_bytes=76,
                    exception_bytes=96, exception_alignment=16, codebook_bytes_per_layer=256,
                    common_descriptor_bytes_per_layer=128,
                    header='bit31 exception; lower31 bank-local exception byte offset; inline header zero',
                    normal_reads='all five slot words; no free variable-prefix stopping',
                    exception_reads='one main/header word then six pool words; slot hole remains allocated',
                    selection='Whole-layer raw bypass based only on complete weight-storage bytes'),
        fairness='Same 4/16 128-bit data words per bank; reserved control partitions 1 directory + 3/15 payload words',
        validation=dict(sealed_fc_layouts_and_cache_results_reproduced=24, exported_int8_tiles_matched=3,
                        all_fixed_serialized_values_read_back=int(total_counts['serialized_readback_bytes']),
                        malformed_headers_rejected=3),
        limitations=['Not same physical area: cache tags, decoder tables/control and muxes remain unpriced',
                     'Descriptor/codebook loads are common setup or codec setup, outside the inherited packet replay; their storage is charged',
                     'Fixed-slot/exception addressing is inherited from LCP; this is a control, not an originality claim',
                     'Serial variable-length Huffman decode has no proven 16-weight-per-bank-per-cycle throughput',
                     'CPU physical-read counts are neither cycles nor memory energy, no RTL/EDA performed'],
        counts=dict(total_counts), storage=dict(total_storage),
        raw_bypass_layers=sum(r['fixed_whole_layer_raw_bypass'] for r in layers),
        aggregate=aggregate, layers=layers, elapsed_s=time.monotonic() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('validation', 'counts', 'storage', 'raw_bypass_layers', 'aggregate', 'elapsed_s')}, indent=2))


if __name__ == '__main__':
    main()
