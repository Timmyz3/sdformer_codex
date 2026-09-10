"""Count the actual fixed-lifting sources and coefficients; no cycle model."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
CHAIN = HERE.parent
sys.path.insert(0, str(CHAIN))
from summarize_temporal_sources import count_weighted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation', type=Path, nargs='?', default=HERE/'fixed_lifting_diverse10')
    args = parser.parse_args()
    directory = args.evaluation.resolve()
    run = json.loads((directory/'result.json').read_text())
    preview = CHAIN.parents[1]/'factor_completion_20260909/latent_stage_train16/flow_recovery64/preview_only/shared48_u8_vq5.npz'
    projection = CHAIN/'shared_temporal_recovery128x256_single_q_consumers_train4/shared_assigned_affine_A_capture/parameters.npz'
    with np.load(preview) as values:
        live = np.flatnonzero(np.any(values['v'] != 0, axis=1))
        preview_u = values['u'][:, live].T.copy()
    with np.load(projection) as values:
        projection_weight = values['spike_conv_weight'].copy()
    compiled = json.loads((HERE/'constant_compilation_result.json').read_text())
    result = dict(
        scope='Actual fixed-lifting evaluation, complete frames; source theta*g kept. No GPU execution or re-evaluation here.',
        preview_parameters=str(preview), projection_parameters=str(projection),
        weight_scope='Preview U8 and unchanged FP32 PED spike-conv from the common parent; Conv2 first factor uses the actual exported theta-weighted q16 coefficients, including quantized zeros.',
        counting='AAC includes first assignment/halo occurrences. H8 is a logical coefficient vector after T10/P2 source sharing, not a physical word or scheduled request. Static zero omission is available to both axes.',
        boundaries='AAC, general products, add/sub, writes and logical vectors remain distinct. No latency, energy, PPA or memory-halving inference. Per-half-step RNE forbids replacing the entire factor chain by its real dense product.',
        axes={})
    for axis, evaluation in run['axes'].items():
        sources = json.loads((directory/(axis+'_sources.json')).read_text())
        metadata = json.loads((directory/(axis+'_fixed_metadata.json')).read_text())
        with np.load(directory/(axis+'_fixed_constants.npz')) as values:
            weights = dict(sn1_preview_conv1=preview_u,
                sn2_anchor_conv2=values['U_conv2_theta_q16'].copy(),
                proj_sn_spike_conv=projection_weight)
            halfsteps = int(np.count_nonzero(values['lifting_q12']))
        frames = len(sources['frames'])
        parts = {name: count_weighted(sources['totals'][name], weight)
                 for name, weight in weights.items()}
        totals = {key: sum(row[key] for row in parts.values()) for key in
                  ('source_occurrences', 'logical_nrv_rows', 'theta_weighted_terms', 'h8_logical_vectors')}
        anchors, full_positions, ticks = 120*160, 240*320, 10
        shared = axis == 'fast_shared'
        channel_products = {name: int(metadata['matrices'][name]['integer_nonzero'])*ticks*anchors
                            for name in ('F', 'U_ped', 'V_ped')}
        lifting = dict(source_products=halfsteps*96*full_positions,
            source_add_sub=40*96*full_positions, source_halfstep_writes=40*96*full_positions,
            extra_BZ_products=halfsteps*16*anchors if shared else 0,
            extra_inverse_products=halfsteps*32*anchors if shared else 0,
            extra_BZ_and_inverse_add_sub=40*(16+32)*anchors if shared else 0,
            extra_BZ_and_inverse_halfstep_writes=40*(16+32)*anchors if shared else 0)
        directions = compiled['models'][axis]['directions']
        forward = directions['forward']['per_T10_vector']['whole_five_pairs_da4ml']
        inverse = directions['inverse']['per_T10_vector']['whole_five_pairs_da4ml']
        compiled_counts = dict(
            source_add_sub=forward['add_sub']*96*full_positions,
            extra_BZ_add_sub=forward['add_sub']*16*anchors if shared else 0,
            extra_inverse_add_sub=inverse['add_sub']*32*anchors if shared else 0,
            source_internal_RNE=(40 if shared else 35)*96*full_positions,
            extra_RNE=40*(16+32)*anchors if shared else 0,
            source_arithmetic_bit_sum=forward['full_carry_bits']*96*full_positions,
            extra_arithmetic_bit_sum=(forward['full_carry_bits']*16+inverse['full_carry_bits']*32)*anchors if shared else 0,
            note='Ordinary compiled numerators include aligned identity adds; do not add40 again. Raw final five source RNE operations may fold into exact gate cutoffs; shared needs all Q states. Comparator/round/saturate/port logic is not in the arithmetic bit sum. These are executed operation demands, not gates, cycles or energy.')
        result['axes'][axis] = dict(frames=frames, AEE_frame_mean=evaluation['evaluation']['AEE_frame_mean'],
            parts=parts, totals=totals, per_frame={k: v/frames for k,v in totals.items()},
            continuous_products_per_frame=channel_products, lifting_per_frame=lifting,
            compiled_lifting_per_frame=compiled_counts,
            complete_live_time_vector_bits=10*96*24,
            stored_BN_constant_bytes=(10*96 if shared else 96)*3,
            lifting_coefficient_bytes=40*2,
            state_note='Raw retains one I and shared one Q; 960x24 bits excludes temporary Z/U, ports and buffers. Shared Bc storage is the full rounded 10x96 table; raw stores96 entries. No free factorization across RNE.')
    raw, shared = (result['axes'][axis]['per_frame'] for axis in ('fast_raw_diagonal', 'fast_shared'))
    result['shared_minus_raw_per_frame'] = {k: dict(difference=shared[k]-raw[k], percent=100*(shared[k]/raw[k]-1)) for k in raw}
    old = json.loads((CHAIN/'fixed_temporal_source_cost.json').read_text())
    if run['evaluation_files'] == old['files']:
        baseline = old['axes']['identity_permuted_base']['source_per_frame_mean']
        result['paired_old_dense_source_control'] = dict(
            reference='../fixed_temporal_source_cost.json; identity_permuted_base',
            same_ordered_frames=True, source_per_frame=baseline,
            interpretation='Different trained/numeric students with the same recovery budget; not an isolated causal effect of factorization or a hardware speed ratio.',
            differences={axis: {k: dict(difference=row['per_frame'][k]-baseline[k],
                percent=100*(row['per_frame'][k]/baseline[k]-1)) for k in baseline}
                for axis,row in result['axes'].items()})
    (directory/'source_cost.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({axis: row['per_frame'] for axis,row in result['axes'].items()}, indent=2))
    print(json.dumps(result['shared_minus_raw_per_frame'], indent=2))


if __name__ == '__main__':
    main()
