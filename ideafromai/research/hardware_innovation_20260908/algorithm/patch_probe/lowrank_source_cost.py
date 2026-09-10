"""Count the ordinary R-before-Conv control on real source time patterns.

R and the shared spatial convolution commute in real arithmetic. Fixed BN
constants must also be transformed. This does not measure a reordered GPU
forward or equate continuous MACs to the original theta-g weighted adds.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent/'dependency'


def main():
    source = json.loads((HERE/'source_element_histogram.json').read_text())
    p4 = json.loads((HERE/'source_batch_histogram.json').read_text())
    controls = json.loads((HERE/'lowrank_controls.json').read_text())['variants']
    bits = np.array([[(pattern >> t) & 1 for pattern in range(1024)]
                     for t in range(10)], dtype=np.float64)
    popcounts = bits.sum(0).astype(np.int64)
    result = dict(
        source='same fixed-four-patch-BN parent r1.conv1 theta-g source, 42 real frames',
        algebra='R(a_h Conv_W(theta*g)+b_h*1) = a_h Conv_W(R(theta*g)) + b_h*(R*1); then apply L and exported low-rank bias b_r, with unchanged center/threshold',
        scope='exact pattern counts with stored FP32 R interpreted as real coefficients; not an implemented reordered forward',
        variants={},
        excluded=['R lookup/formation and its storage/precision',
                  'transformed-source lifetime, halo line buffers and source traffic',
                  'actual Y/latent widths, bank accesses, MAC area and energy',
                  'remaining L transform, gates, conv2, and full chain cycles'],
    )
    for name, control in controls.items():
        right = np.asarray(control['right_factor_fp32'], dtype=np.float64)
        values = right @ bits
        nonzero = np.count_nonzero(values, axis=0)
        negative = np.count_nonzero(values < 0, axis=0)
        all_nonempty_survive = bool(np.all(nonzero[1:] != 0))
        row = dict(rank=control['rank'],
            nonempty_patterns_annihilated=int(np.count_nonzero(nonzero[1:] == 0)),
            theoretical_nonzero_amplitude_min=float(values[values != 0].min()),
            theoretical_nonzero_amplitude_max=float(values[values != 0].max()),
            splits={})
        for split in ('train', 'valid'):
            frames = [f for f in source['frames'] if f['split'] == split]
            # All selected source thresholds/amplitudes have been measured,
            # rather than assuming that every ATLIF layer has theta=1.
            assert all(f['theta'] == f['nonzero_amplitude_min'] ==
                       f['nonzero_amplitude_max'] == 1. for f in frames)
            hist = np.asarray(source[split+'_hist'], dtype=np.int64)
            old = int(hist @ popcounts)*96/len(frames)
            new = int(hist @ nonzero)*96/len(frames)
            neg = int(hist @ negative)*96/len(frames)
            common = np.asarray(p4[split+'_hist'], dtype=np.int64)
            row['splits'][split] = dict(
                frames=len(frames), original_theta_g_weighted_add_terms_per_frame=old,
                transformed_continuous_MAC_terms_per_frame=new,
                transformed_negative_source_MAC_terms_per_frame=neg,
                continuous_MAC_count_over_original_weighted_add_count=new/old,
                logical_W_vectors_per_P4_output_group_per_frame=
                    int(common[1:].sum())/len(frames) if all_nonempty_survive else None,
                issue_count_boundary='one coefficient vector reused across all latent outputs within each fixed P4 context; logical uses, not SRAM transactions',
            )
        result['variants'][name] = row
    (HERE/'lowrank_source_cost.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    for name, row in result['variants'].items():
        print(name, json.dumps(row['splits']['valid'], ensure_ascii=False))


if __name__ == '__main__':
    main()
