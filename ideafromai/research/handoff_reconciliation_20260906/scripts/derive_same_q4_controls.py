"""Derive necessary same-organization controls from saved repair traces.

No numerical forward pass. Same q4 static layout, 24 spatial issue positions,
one packet source cache; choose fine4, coarse96 expanded into q4, or full replay.
These are logical event counts, not cycle or physical-area comparisons.
"""
from pathlib import Path
import json
import numpy as np

BASE = Path(__file__).resolve().parents[1]
TRACE = BASE / 'records/repair_sector_sample0_r1'


def count(mask, nc, sizes, C, P):
    sectors = mask.sum(1, dtype=np.int64)
    active = (nc > 0).sum(1, dtype=np.int64)
    ones = nc.sum(1, dtype=np.int64)
    colwords = (sizes * 10 + 127) // 128
    compact_per_sector = ((nc.astype(np.int64) + 23) // 24).sum(1)
    ingress = 0
    for b in np.flatnonzero(sectors):
        for t in range(10):
            start = (t * P + int(b) * 32) * C
            stop = start + int(sizes[b]) * C
            ingress += (stop + 127) // 128 - start // 128
    return dict(sector_packet_count=int(sectors.sum()),
                replay_FC1_active_scalar_terms=int(np.dot(sectors, 4 * ones)),
                PSN_scalar_terms=int(np.dot(sectors, 4 * sizes * 100)),
                weight_128bit_word_reads=int(np.dot(sectors, active)),
                weight_bank_read_beats=int(np.dot(sectors, active)),
                source_scratch_column_128bit_reads=int(np.dot(sectors, active * colwords)),
                source_nonzero_summary_scan_128bit_reads=int(sectors.sum()) * ((C + 127) // 128),
                compacted_spatial_issue_batches=int(np.dot(sectors, compact_per_sector)),
                fixed_spatial_scan_issue_batches=int(np.dot(sectors, active * ((sizes * 10 + 23) // 24))),
                source_ingress_128bit_reads=ingress)


def main():
    original = json.loads((TRACE / 'result.json').read_text())
    layers = []
    for layer in original['layers']:
        nb, H = layer['bitmap_shape']
        nc = np.load(TRACE / layer['active_position_counts_file'], allow_pickle=False)
        bits = np.fromfile(TRACE / layer['anyT_failure_bitmap'], dtype=np.uint8).reshape(nb, H // 8)
        failure = np.unpackbits(bits, axis=1, bitorder='little')[:, :H].astype(bool)
        sizes = np.minimum(32, layer['P'] - np.arange(nb) * 32).astype(np.int64)
        fine = failure.reshape(nb, H // 4, 4).any(2)
        coarse = np.repeat(failure.reshape(nb, H // 96, 96).any(2), 24, axis=1)
        full = np.ones_like(fine)
        views = {name: count(mask, nc, sizes, layer['C'], layer['P'])
                 for name, mask in (('fine4', fine), ('coarse96_on_same_q4', coarse), ('full_on_same_q4', full))}
        previous = next(r for r in layer['service'] if r['q'] == 4)
        for k, v in views['fine4'].items():
            if k != 'source_ingress_128bit_reads':
                assert v == previous[k], (layer['stage'], k)
        assert views['fine4']['source_ingress_128bit_reads'] == layer['source_ingress']['source_128bit_word_reads']
        assert views['coarse96_on_same_q4']['replay_FC1_active_scalar_terms'] == layer['exact_GH_counter_matches']['replay_FC1_terms_fixed96']
        assert views['full_on_same_q4']['replay_FC1_active_scalar_terms'] == layer['exact_GH_counter_matches']['full_FC1_active_scalar_terms']
        assert views['full_on_same_q4']['PSN_scalar_terms'] == layer['exact_GH_counter_matches']['full_PSN_scalar_terms']
        # A storage/recompute alternative has two FC1 passes but only one PSN:
        # its first pass only produces current BN moments, without PSN outputs.
        F = layer['exact_GH_counter_matches']['full_FC1_active_scalar_terms']
        A = layer['exact_GH_counter_matches']['full_PSN_scalar_terms']
        work = dict(full_save=dict(FC1_terms=F, PSN_terms=A),
                    full_recompute=dict(FC1_terms=2*F, PSN_terms=A),
                    late_threshold_q4=dict(FC1_terms=F+views['fine4']['replay_FC1_active_scalar_terms'],
                                           PSN_terms=A+views['fine4']['PSN_scalar_terms']))
        layers.append(dict(stage=layer['stage'], views=views, charged_first_work=work,
                           source_backing_bytes=layer['N']*layer['C']//8,
                           note='Full source backing persists to statistics seal; no assumed on-chip residency'))
    result = dict(status='DERIVED_FROM_SAVED_FAILURE_BITMAPS_AND_SOURCE_COUNTS_NO_FORWARD_RERUN',
                  scope='Same two-layer sample0 B32K4; same q4 mapping for all three mask controls',
                  reuse='No cross-packet weight cache; retained packet source rescanned per q4 sector',
                  limitations=['Same logical organization only, no mapped circuit or timed selector/RMW.',
                               'Full-save/full-recompute/late first-work table is FC1 and PSN arithmetic only; sums must not be called cycles.',
                               'Normalization, moments, certificate creation, encode/decode, transpose, Y ports and commit remain unpriced.',
                               'Source scratch read and selector scan refer to the same scan, not two independent physical reads.',
                               'FP32 weight word layout is not proof of FP32 Y/U arithmetic or frozen model equivalence.'],
                  layers=layers)
    output = BASE / 'records/repair_same_q4_controls.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
