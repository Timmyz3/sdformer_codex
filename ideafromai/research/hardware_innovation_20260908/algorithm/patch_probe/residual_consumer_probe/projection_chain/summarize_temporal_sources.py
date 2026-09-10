"""Weight-zero-aware source arithmetic and H8 logical-vector accounting.

These are counters over the existing ten complete real frames. They do not
schedule memory, charge continuous MACs as AACs, or predict clock cycles.
"""
from pathlib import Path
import argparse
import json
import numpy as np


HERE = Path(__file__).resolve().parent
AXES = {
    'ordinary_independent_A': ('shared_temporal_recovery128x256',
        'shared_temporal_recovery128x256_single_q_sources_diverse10'),
    'shared_assigned_affine_A': ('shared_temporal_recovery128x256',
        'shared_temporal_recovery128x256_single_q_sources_diverse10'),
    'identity_permuted_base': ('temporal_structured_recovery/stage128x256',
        'temporal_structured_recovery/raw_factorized_sources_diverse10'),
    'identity_permuted_joint_r2': ('temporal_structured_recovery/stage128x256',
        'temporal_structured_recovery/raw_factorized_sources_diverse10'),
}


def count_weighted(source, weight):
    nonzero = weight.reshape(weight.shape[0], -1) != 0
    by_k = nonzero.sum(0).astype(np.int64)
    h8_by_k = np.asarray([nonzero[i:i+8].any(0) for i in range(0, len(nonzero), 8)]).sum(0)
    return dict(source_occurrences=source['source_occurrences'],
        logical_nrv_rows=source['nrv_rows'],
        actual_weight_nonzero=int(nonzero.sum()), weight_elements=int(nonzero.size),
        theta_weighted_terms=int(np.asarray(source['active_columns'], np.int64)@by_k),
        h8_logical_vectors=int(np.asarray(source['nrv_rows_by_k'], np.int64)@h8_by_k))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--projection-parameters', type=Path, required=True,
        help='Actual TemporalConsumerCapture parameters.npz containing spike_conv_weight.')
    args = parser.parse_args()
    preview = HERE.parents[1]/'factor_completion_20260909/latent_stage_train16/flow_recovery64/preview_only/shared48_u8_vq5.npz'
    with np.load(preview) as values:
        live = np.flatnonzero(np.any(values['v'] != 0, axis=1))
        preview_u = values['u'][:, live].T
    with np.load(args.projection_parameters) as values:
        projection_weight = values['spike_conv_weight'].copy()
    result = dict(scope='Ten complete diverse frames per saved student; actual theta*g, exact stored weight zeros, T10/P2 logical source layout.',
        projection_weight_source=str(args.projection_parameters),
        definition='theta_weighted_terms=sum_k active_occurrences[k]*nnz(W[:,k]); h8_logical_vectors=sum_k nrv_rows[k]*number_of_H8_groups_with_any_nonzero_weight_at_k.',
        boundaries='Includes first assignments and halo repetitions. H8 vectors are logical coefficient-group demands with within-H8 zeros permitted; no physical SRAM transactions, bank scheduling, cycles, continuous products or PPA. Conv2 weights are the actual floating R16 first factor; future coefficient quantization must be recounted.',
        axes={})
    for axis, (parameters, counters) in AXES.items():
        source = json.loads((HERE/counters/(axis+'_sources.json')).read_text())
        with np.load(HERE/parameters/(axis+'.npz')) as values:
            conv2_u = values['conv2_U_R16'].copy()
        weights = dict(sn1_preview_conv1=preview_u, sn2_anchor_conv2=conv2_u,
                       proj_sn_spike_conv=projection_weight)
        row = {name: count_weighted(source['totals'][name], weight)
               for name, weight in weights.items()}
        totals = {key: sum(v[key] for v in row.values())
                  for key in ('source_occurrences', 'logical_nrv_rows', 'theta_weighted_terms', 'h8_logical_vectors')}
        result['axes'][axis] = dict(frames=len(source['frames']), sources=row, totals=totals)
    shared = result['axes']['shared_assigned_affine_A']['totals']
    baseline = result['axes']['identity_permuted_base']['totals']
    result['shared_minus_raw_diagonal'] = {key: dict(difference=shared[key]-baseline[key],
        percent=100*(shared[key]/baseline[key]-1)) for key in shared}
    (HERE/'shared_temporal_source_cost.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({axis: row['totals'] for axis,row in result['axes'].items()}, indent=2))
    print(json.dumps(result['shared_minus_raw_diagonal'], indent=2))


if __name__ == '__main__':
    main()
