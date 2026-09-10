"""Actual native-P4 Conv2 rank controls: work and coefficient-word opportunities.

This is not a cycle model.  AAC updates, continuous products and word requests
remain distinct.  Factors are the companion probe's single SVD export.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def word_counts(active_by_input_group, coefficient_nonzero, coefficient_bytes=4):
    # active: [input,G]; coefficient: [output,input].  Every word is reused
    # across all legal T/P uses in this native P4, and discarded at next G.
    activity = (active_by_input_group.T[:, :, None]
                & coefficient_nonzero.T[None])
    # Keys retain the FP32 H2/H8 port names. FP64 K has H1/H4 on those ports.
    return {str(width): dict(
        requests=int(activity.reshape(activity.shape[0], activity.shape[1], -1, width*4//coefficient_bytes).any(-1).sum()),
        bytes=int(activity.reshape(activity.shape[0], activity.shape[1], -1, width*4//coefficient_bytes).any(-1).sum())*width*4,
        useful_coefficient_lanes=int(activity.sum())) for width in (2, 8)}


def binary_producer(source, route, coefficients, coefficient_bytes=4):
    """source[T,K,G,P] and output-by-K static coefficients, same theta*g."""
    live_source = source & route[None, None]
    nz = coefficients != 0
    counts_k = live_source.sum((0, 2, 3), dtype=np.int64)
    updates = int(counts_k @ nz.sum(0, dtype=np.int64))
    if nz.all():
        active = np.broadcast_to(live_source.any(1)[None],
                                 (len(coefficients), source.shape[0], source.shape[2], source.shape[3]))
    else:
        # Static coefficient zeros are a legal source-support control.
        flat = live_source.transpose(1, 0, 2, 3).reshape(source.shape[1], -1)
        active = (nz.astype(np.int32) @ flat.astype(np.int32) > 0).reshape(
            len(coefficients), source.shape[0], source.shape[2], source.shape[3])
    seeds = int(active.sum())
    return dict(sparse_AAC_updates=updates, seed_loads=seeds,
                arithmetic_adds_after_first_seed=updates-seeds,
                source_bit_contributions=int(live_source.sum()),
                coefficient_words=word_counts(live_source.any((0, 3)), nz, coefficient_bytes)), active


def continuous_consumer(active_latents, coefficients):
    """Skip only Z known empty from source/static-U support, never future Z value."""
    nz = coefficients != 0
    per_r = active_latents.sum((1, 2, 3), dtype=np.int64)
    products = int(per_r @ nz.sum(0, dtype=np.int64))
    if nz.all():
        seeds = int(active_latents.any(0).sum()) * len(coefficients)
    else:
        flat = active_latents.reshape(len(active_latents), -1)
        seeds = int((nz.astype(np.int32) @ flat.astype(np.int32) > 0).sum())
    return dict(continuous_products=products, product_seeds=seeds,
                continuous_accumulation_adds_after_first_seed=products-seeds,
                nonempty_Z_values=int(active_latents.sum()),
                coefficient_words=word_counts(active_latents.any((1, 3)), nz))


def execute_axis(source, anchor, weight, first, second, projection, kernel64, name, rank):
    all_positions = np.ones(anchor.shape, bool)
    zero_positions = np.zeros(anchor.shape, bool)
    mixed = name == 'nonanchor16_anchororiginal'
    branch_zero = name == 'nonanchor_norm2_branch_zero'
    direct_route = anchor if mixed or branch_zero else (all_positions if rank == 0 else zero_positions)
    lowrank_route = zero_positions if rank == 0 else (~anchor if mixed else all_positions)
    direct, _ = binary_producer(source, direct_route, weight)
    if rank:
        producer, active = binary_producer(source, lowrank_route, first[:rank])
        consumer = continuous_consumer(active, second[:, :rank])
    else:
        empty_words = {width: dict(requests=0, bytes=0, useful_coefficient_lanes=0) for width in ('2', '8')}
        producer = dict(sparse_AAC_updates=0, seed_loads=0, arithmetic_adds_after_first_seed=0,
                        source_bit_contributions=0, coefficient_words=empty_words)
        consumer = dict(continuous_products=0, product_seeds=0,
                        continuous_accumulation_adds_after_first_seed=0,
                        nonempty_Z_values=0, coefficient_words=empty_words)
    # The original single conv_res(identity+branch) already includes identity;
    # do not charge it a second identity projection. K routes instead project x.
    ped_active = np.broadcast_to(anchor[None, None], (96, 10, *anchor.shape))
    ped = continuous_consumer(ped_active, projection)
    has_kernel = '_plus_K_' in name
    kernel_bytes = 8 if name.endswith('_fp64') else 4
    if has_kernel:
        kernel_coeff = kernel64 if kernel_bytes == 8 else kernel64.astype(np.float32)
        kernel, _ = binary_producer(source, anchor, kernel_coeff, kernel_bytes)
        # K contributions merge into the identity accumulator. A private K
        # first-seed saving requires a final merge add, cancelling that saving.
        kernel['arithmetic_adds_after_first_seed'] = kernel['sparse_AAC_updates']
    else:
        kernel = dict(sparse_AAC_updates=0, seed_loads=0, arithmetic_adds_after_first_seed=0,
                      source_bit_contributions=0,
                      coefficient_words={width: dict(requests=0, bytes=0, useful_coefficient_lanes=0) for width in ('2', '8')})
    words = {width: {key: sum(stage['coefficient_words'][width][key]
                    for stage in (direct, producer, consumer, kernel, ped))
                    for key in ('requests', 'bytes', 'useful_coefficient_lanes')}
             for width in ('2', '8')}
    return dict(axis=name, rank=rank, direct_original=direct, U=producer, V=consumer, K=kernel, PED=ped,
        sparse_AAC_updates=direct['sparse_AAC_updates']+producer['sparse_AAC_updates']+kernel['sparse_AAC_updates'],
        sparse_adds_after_seed=direct['arithmetic_adds_after_first_seed']+producer['arithmetic_adds_after_first_seed']+kernel['arithmetic_adds_after_first_seed'],
        continuous_V_products=consumer['continuous_products'],
        continuous_V_adds_after_seed=consumer['continuous_accumulation_adds_after_first_seed'],
        continuous_PED_products=ped['continuous_products'],
        continuous_PED_adds_after_seed=ped['continuous_accumulation_adds_after_first_seed'],
        fused_K_constant_initializations=int(anchor.sum())*10*96 if has_kernel else 0,
        coefficient_words=words,
        direct_positions=int(direct_route.sum()), lowrank_positions=int(lowrank_route.sum()))


def sum_axis(rows, name):
    values = [row['axes'][name] for row in rows]
    flat_keys = ('sparse_AAC_updates', 'sparse_adds_after_seed', 'continuous_V_products',
                 'continuous_V_adds_after_seed', 'continuous_PED_products', 'continuous_PED_adds_after_seed',
                 'fused_K_constant_initializations', 'direct_positions', 'lowrank_positions')
    result = {key: sum(value[key] for value in values) for key in flat_keys}
    result['coefficient_words'] = {width: {key: sum(v['coefficient_words'][width][key] for v in values)
        for key in ('requests', 'bytes', 'useful_coefficient_lanes')} for width in ('2', '8')}
    for stage in ('direct_original', 'U', 'V', 'K', 'PED'):
        result[stage] = {key: sum(value[stage][key] for value in values)
                         for key in values[0][stage] if key != 'coefficient_words'}
        result[stage]['coefficient_words'] = {width: {key: sum(v[stage]['coefficient_words'][width][key] for v in values)
            for key in ('requests', 'bytes', 'useful_coefficient_lanes')} for width in ('2', '8')}
    return result


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, default=here/'capture_train4_ped')
    parser.add_argument('--parameters', type=Path, default=here/'rank_control_parameters.npz')
    parser.add_argument('--output', type=Path, default=here/'rank_service_opportunity.json')
    args = parser.parse_args()
    params = np.load(args.parameters)
    print('PARAMETER_FIELDS', params.files, flush=True)
    weight = params['W2'].reshape(96, 864)
    first, second = params['first_factor32'], params['second_factor32']
    reference = np.load(args.capture/'bound_parameters.npz')
    assert np.array_equal(weight, reference['W2'].reshape(96, 864))
    projection_parameters = np.load(args.capture/'projection_parameters.npz')
    projection = projection_parameters['conv_res_weight'].reshape(96, 96)
    gain = reference['bn2_gain'].astype(np.float64)
    kernel64 = (projection.astype(np.float64)*gain[None]) @ weight.astype(np.float64)
    kernel_constant64 = projection.astype(np.float64) @ (
        reference['bn2_offset'].astype(np.float64)+gain*reference['conv2_bias'].astype(np.float64))
    kernel_constant64 += projection_parameters['conv_res_bias'].astype(np.float64)
    theta = float(reference['sn2_theta'])
    # The current export is literal theta=1.  Do not silently redefine a future
    # arbitrary-amplitude producer as unit gates without compiled coefficients.
    assert theta == 1.0
    axes = dict(original=0, nonanchor_norm2_branch_zero=0,
                uniform16=16, uniform32=32, nonanchor16_anchororiginal=16,
                exact_plus_K_fp32_scenario=0, rank16_gate_plus_K_fp32_scenario=16,
                exact_plus_K_fp64=0, rank16_gate_plus_K_fp64=16)
    rows = []
    for path in sorted(args.capture.glob('[0-9][0-9]_*.npz')):
        d = np.load(path)
        assert float(d['conv2_source_theta_g_max_abs']) == 0
        words = d['conv2_source_gate_words']
        source = ((words.transpose(1, 0, 2)[None] >> np.arange(10)[:, None, None, None]) & 1).astype(bool)
        anchor = d['anchor_mask'].astype(bool)
        # Actual exact-control projection inputs are dense at sampled anchors.
        # Approximate UV combined-value routes use that dense issue upper bound;
        # no future cancellation is assumed.
        assert np.all(d['identity'][:, :, anchor] != 0)
        assert np.all(d['r1out'][:, :, anchor] != 0)
        row = dict(file=path.name, groups=len(anchor), positions=int(anchor.size),
            anchor_positions=int(anchor.sum()), source_active_bits=int(source.sum()),
            source_density=float(source.mean()), nonempty_time_positions=int(source.any(1).sum()),
            source_descriptor_scans=int(words.shape[0]*words.shape[1]),
            source_nonempty_descriptors=int(source.any((0, 3)).sum()),
            source_40bit_dense_payload_bytes=int(words.shape[0]*words.shape[1]*5),
            source_padded64_dense_read_bytes=int(words.shape[0]*words.shape[1]*8), axes={})
        for name, rank in axes.items():
            row['axes'][name] = execute_axis(source, anchor, weight, first, second, projection, kernel64, name, rank)
        # Closed-form check independent of the routed coefficient-word loops,
        # applicable to these actual all-nonzero matrices.
        assert np.all(weight != 0) and np.all(first != 0) and np.all(second != 0)
        assert row['axes']['original']['sparse_AAC_updates'] == int(source.sum())*96
        assert row['axes']['nonanchor_norm2_branch_zero']['sparse_AAC_updates'] == int((source & anchor[None, None]).sum())*96
        for rank in (16, 32):
            metrics = row['axes']['uniform'+str(rank)]
            assert metrics['U']['sparse_AAC_updates'] == int(source.sum())*rank
            assert metrics['V']['continuous_products'] == int(source.any(1).sum())*rank*96
        for width in ('2', '8'):
            assert row['axes']['original']['direct_original']['coefficient_words'][width]['requests'] == int(source.any((0, 3)).sum())*96//int(width)
            assert row['axes']['exact_plus_K_fp64']['K']['coefficient_words'][width]['requests'] == 2*row['axes']['exact_plus_K_fp32_scenario']['K']['coefficient_words'][width]['requests']
        rows.append(row)
        print(path.name, {name: (v['sparse_AAC_updates'], v['continuous_V_products'], v['coefficient_words']['8']['requests'])
                         for name, v in row['axes'].items()}, flush=True)
    assert len(rows) == 4
    aggregate = {name: sum_axis(rows, name) for name in axes}
    baseline = aggregate['original']
    for name, value in aggregate.items():
        value['AAC_update_ratio_vs_original'] = value['sparse_AAC_updates']/baseline['sparse_AAC_updates']
        value['word_request_ratio_vs_original'] = {width: value['coefficient_words'][width]['requests']/baseline['coefficient_words'][width]['requests']
                                                    for width in ('2', '8')}
        if value['continuous_V_products'] and not name.endswith('_fp64'):
            value['break_even_MAC_cost_in_AAC_update_units_ignoring_all_other_cost'] = (
                baseline['sparse_AAC_updates']-value['sparse_AAC_updates'])/value['continuous_V_products']
            value['break_even_multiply_cost_in_add_units_with_first_seed_ignoring_all_other_cost'] = (
                baseline['sparse_adds_after_seed']-value['sparse_adds_after_seed']-value['continuous_V_adds_after_seed'])/value['continuous_V_products']
        strong = aggregate['nonanchor_norm2_branch_zero']
        value['word_request_ratio_vs_branch_zero'] = {width: value['coefficient_words'][width]['requests']/strong['coefficient_words'][width]['requests']
                                                     for width in ('2', '8')}
        if value['continuous_V_products'] and not name.endswith('_fp64'):
            value['break_even_MAC_cost_in_AAC_update_units_vs_branch_zero_ignoring_all_other_cost'] = (
                strong['sparse_AAC_updates']-value['sparse_AAC_updates'])/value['continuous_V_products']
    payload = {}
    for name, rank in axes.items():
        original_bytes = weight.nbytes if rank == 0 or name.startswith('nonanchor') else 0
        u_bytes = first[:rank].nbytes if rank else 0
        v_bytes = second[:, :rank].nbytes if rank else 0
        k_bytes = kernel64.nbytes if name.endswith('_fp64') else kernel64.size*4
        if '_plus_K_' not in name:
            k_bytes = 0
        payload[name] = dict(original_W2_bytes=original_bytes, U_bytes=u_bytes, V_bytes=v_bytes,
            K_bytes=k_bytes, common_PED_projection_FP32_bytes=projection.nbytes,
            total_coefficient_payload_bytes=original_bytes+u_bytes+v_bytes+k_bytes+projection.nbytes,
            compiled_K_constant_FP64_bytes=kernel_constant64.nbytes if k_bytes else 0,
            Z_FP32_peak_bytes=rank*10*4*4,
            identity_FP32_P4_T10_H96_bytes=10*4*96*4,
            Conv2_output_FP32_P4_T10_H96_bytes=10*4*96*4,
            Conv2_logically_live_output_FP32_peak_bytes=10*(2 if name == 'nonanchor_norm2_branch_zero' else 4)*96*4,
            combined_Z_identity_output_FP32_bytes=(rank+96+96)*10*4*4,
            anchor_PED_value_FP32_peak_bytes=10*2*96*4,
            optional_per_accumulator_valid_bits=(rank+96)*10*4,
            current_source_descriptor_register_bits=40)
    result = dict(complete=True, parameter_file=str(args.parameters), capture=str(args.capture),
        scope='Same four captured train frames, original64 horizontal P4 per frame; not full-frame inference. Fixed untrained SVD controls plus the separately validated ordinary nonanchor whole-BN-branch-zero control. Conv2 and continuous PED branch arithmetic/local coefficient words; not cycles, latency, energy or PPA.',
        numeric='Actual source is theta*g with captured theta=1; both factors retain FP32 continuous coefficients. No INT8/Acc24 assumption. Different ranks are different approximate functions, accuracy evaluated separately.',
        arithmetic='sparse_AAC_updates includes the first seed/write; arithmetic_adds_after_first_seed gives that ordinary control. V continuous products and its accumulation additions are independent counts. No cancellation of numerically computed Z is used to suppress V; only source/static-U support proves zero.',
        schedule='Bounded native P4: hold identity and all output partials; scan K once, retain current source40 while updating all applicable routes. UV holds complete T/P4 Z and consumes V afterwards. Mixed uses original W2 only at anchor and UV only elsewhere. K routes share the source scan but add an independent kernel update into value accumulators. A common 35840B naked FP32 Z/identity/output arena plus 7680B anchor value payload is sufficient for this accounting order, not a macro/port/whole-projector pipeline claim. FP64 K accumulation precision/state needs are not validated or included in that FP32 state scenario.',
        addresses={'original_W2':'separate aligned base+(k*96+h)*4',
                   'U':'separate aligned base+(k*R+r)*4; export first_factor32[r,k] transposed in storage',
                   'V':'separate aligned base+(r*96+h)*4; export second_factor32[h,r] transposed in storage',
                   'K':'separate aligned base+(k*96+o)*coefficient_bytes; coefficient_bytes=8 for computed FP64 K, 4 only for unvalidated FP32 storage scenario',
                   'PED':'separate aligned base+(h*96+o)*4; current word reused over T/P4 anchor positions',
                   'service':'H2=64bit, H8=256bit, independently evaluated. One current coefficient word reused across all legal T/P4, no inter-P4 coefficient cache. The same actual nonzero mask is granted to each axis.'},
        excluded='No Conv2/BN2/proj execution precision claim, no source/zero-mask index memory, no bank stalls, no Z/identity traffic timing, no remaining BN2/shortcut/proj.sn/spiking-proj.conv timing, no cold DMA attribution. Coefficient payload is unique storage demand; local word uses are not off-chip transfers. Ordinary parity regroup and cross-P4 caching are unmeasured.',
        K_control='K[o,k]=sum_h Wres[o,h]*BNgain[h]*W2[h,k], formed once in FP64; bias/BN offset becomes a static value constant. Real-algebra exactness is not native FP32 equivalence. FP32 K is only a generous unvalidated storage/arithmetic scenario; FP64 K word traffic is also reported. The native original does one Wres(identity+branch), already including identity; K routes still require one Wres(identity), never remove a fictitious second original projection.',
        branch_zero_control='Nonanchor entire normalized branch is zero: no Conv2/U/V/BN2 or branch-add there; proj.sn consumes identity. Anchor retains original Conv2/BN2/residual and continuous PED value. sn2 source is unchanged, so the existing captured support is directly reusable. It is ordinary spatial pruning, not a new mechanism or an exact-original-function skip.',
        rows=rows, aggregate=aggregate, payload=payload,
        parameters=dict(W2_nonzero=int(np.count_nonzero(weight)), W2_elements=weight.size,
            U16_nonzero=int(np.count_nonzero(first[:16])), U32_nonzero=int(np.count_nonzero(first)),
            V16_nonzero=int(np.count_nonzero(second[:, :16])), V32_nonzero=int(np.count_nonzero(second)),
            K64_nonzero=int(np.count_nonzero(kernel64)), K32_nonzero=int(np.count_nonzero(kernel64.astype(np.float32))),
            K_max_abs_FP32_cast_error=float(np.max(np.abs(kernel64-kernel64.astype(np.float32))))))
    result['checks'] = dict(actual_original_weight_equal_export=True,
        actual_source_theta_g_residual_zero=True, actual_anchor_identity_and_original_r1out_all_nonzero=True,
        dense_closed_form_AAC_V_products_and_original_word_counts='all four frames match',
        FP64_K_vs_FP32_K_word_width_relation='all four frames match',
        no_rank_function_accuracy_test_here=True)
    zero_summary = args.capture.parent/'branch_control_valid825/nonanchor_norm2_branch_zero_summary.json'
    result['branch_zero_accuracy_external'] = dict(file=str(zero_summary), **json.loads(zero_summary.read_text()))
    static_groups = sum(int(np.load(path)['anchor_mask'].any(-1).sum())
                        for path in sorted(args.capture.glob('[0-9][0-9]_*.npz')))
    result['branch_zero_descriptor_boundary'] = dict(
        fixed_full_P4_scans_reported=sum(row['source_descriptor_scans'] for row in rows),
        static_anchor_containing_P4=static_groups,
        potential_scans_if_whole_no_anchor_P4_skipped=static_groups*864,
        potential_padded64_read_bytes_if_whole_no_anchor_P4_skipped=static_groups*864*8,
        scope='Static no-anchor-group omission is legal but not scheduled here; no 75-percent-position-to-cycle extrapolation. Per-anchor halos still need actual upstream sn2 support.')
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
