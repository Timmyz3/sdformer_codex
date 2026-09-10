"""Untrained magnitude masks on real W: request opportunities, never cycles/AEE."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'bn_state'))
from support_service_model import read_torch


def top2_mask(scores, count=2):
    # Stable tie break: lower source-column index wins.
    order = np.argsort(-scores, axis=-1, kind='stable')[..., :count]
    mask = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(mask, order, True, axis=-1)
    return mask


def count_mask(weight, mask, codes):
    H, C = weight.shape
    P = 4
    G = H//8
    support = (weight != 0) & mask
    group = support.reshape(G, 8, C)
    union = group.any(axis=1)
    intersection = group.all(axis=1)
    active = (codes.reshape(-1, P, C) != 0).any(axis=1)
    active_by_c = active.sum(axis=0, dtype=np.int64)
    w_per_c = support.sum(axis=0, dtype=np.int64)
    union_per_c = union.sum(axis=0, dtype=np.int64)
    stripe_union = union.reshape(G//16, 16, C).any(axis=1)

    # Logical P4 source row is 12 bits; count touched 64-bit words, not rows.
    source_words = (P*3*C+63)//64
    word_needed = np.zeros((G, source_words), dtype=bool)
    stripe_words = np.zeros((G//16, source_words), dtype=bool)
    for c in range(C):
        lo = (P*3*c)//64
        hi = (P*3*(c+1)-1)//64
        word_needed[:, lo:hi+1] |= union[:, c:c+1]
        stripe_words[:, lo:hi+1] |= stripe_union[:, c:c+1]

    w2 = weight.astype(np.int64)**2

    counts = {
        'weight_elements': int(weight.size),
        'mask_kept_slots': int(mask.sum()),
        'nonzero_weights': int(support.sum()),
        'original_W_squared_sum': int(w2.sum()),
        'retained_W_squared_sum': int((w2*mask).sum()),
        'group_columns': int(union.size),
        'union_nonzero_group_columns': int(union.sum()),
        'all_eight_nonzero_group_columns': int(intersection.sum()),
        'C4_group_blocks': int(G*C//4),
        'all_eight_zero_C4_blocks': int((~union.reshape(G, C//4, 4).any(axis=-1)).sum()),
        'C16_group_blocks': int(G*C//16),
        'all_eight_zero_C16_blocks': int((~union.reshape(G, C//16, 16).any(axis=-1)).sum()),
        'all_H_zero_source_columns': int((~support.any(axis=0)).sum()),
        'all_C_zero_output_rows': int((~support.any(axis=1)).sum()),
        'static_source64_words_per_spatial_P4_group_all_Hgroups': int(word_needed.sum()),
        'dense_source64_words_per_spatial_P4_group_all_Hgroups': int(G*source_words),
        'F_cache16_stripes': int(G//16),
        'F_cache16_stripes_covering_all_C': int(stripe_union.all(axis=1).sum()),
        'F_cache16_stripe_columns': int(stripe_union.size),
        'F_cache16_union_live_columns': int(stripe_union.sum()),
        'F_cache16_C16_blocks': int(stripe_union.size//16),
        'F_cache16_all_zero_C16_blocks': int((~stripe_union.reshape(G//16, C//16, 16).any(axis=-1)).sum()),
        'F_cache16_static_source64_words_per_spatial_P4_group': int(stripe_words.sum()),
        'F_cache16_dense_source64_words_per_spatial_P4_group': int((G//16)*source_words),
        'v000_source_spatial_P4_groups': int(len(active)),
        'v000_active_source_rows_one_Hgroup': int(active.sum()),
        'v000_source_active_rows_before_weight_union_all_Hgroups': int(active.sum())*G,
        'v000_source_active_rows_after_weight_union_all_Hgroups': int(active_by_c @ union_per_c),
        'v000_W_scalar_reads_with_source_and_weight_metadata': int(active_by_c @ w_per_c),
        'v000_W_scalar_reads_dense_fanout_before_weight_metadata': int(active.sum())*H,
        'v000_static_source64_reads_with_weight_union_only': int(word_needed.sum())*len(active),
        'v000_dense_source64_reads_all_Hgroups': int(G*source_words)*len(active),
        'v000_nonzero_class_events': int((codes != 0).sum()),
        'v000_class_contributions_after_Wzero': int((codes != 0).sum(axis=0, dtype=np.int64) @ w_per_c),
    }
    return counts


def with_rates(counts):
    q = dict(counts)
    q.update(
        weight_zero_rate=1-counts['nonzero_weights']/counts['weight_elements'],
        W_squared_energy_retained=counts['retained_W_squared_sum']/counts['original_W_squared_sum'],
        mask_kept_fraction=counts['mask_kept_slots']/counts['weight_elements'],
        source_column_union_rate=counts['union_nonzero_group_columns']/counts['group_columns'],
        source_column_all_eight_nonzero_rate=counts['all_eight_nonzero_group_columns']/counts['group_columns'],
        all_eight_zero_C4_block_rate=counts['all_eight_zero_C4_blocks']/counts['C4_group_blocks'],
        all_eight_zero_C16_block_rate=counts['all_eight_zero_C16_blocks']/counts['C16_group_blocks'],
        v000_logical_source_rows_retained=counts['v000_source_active_rows_after_weight_union_all_Hgroups']/counts['v000_source_active_rows_before_weight_union_all_Hgroups'],
        static_source64_words_retained=counts['static_source64_words_per_spatial_P4_group_all_Hgroups']/counts['dense_source64_words_per_spatial_P4_group_all_Hgroups'],
        F_cache16_union_column_rate=counts['F_cache16_union_live_columns']/counts['F_cache16_stripe_columns'],
        F_cache16_source64_words_retained=counts['F_cache16_static_source64_words_per_spatial_P4_group']/counts['F_cache16_dense_source64_words_per_spatial_P4_group'],
    )
    return q


def capacity_table():
    # Conservative local copies: each W tile stores its own mask and tau.
    modes = {'original': (384, 0), 'row_2of4': (192, 36),
             'broadcast8_shared_2of4': (192, 36),
             'broadcast8_shared_C16_half': (192, 3),
             'hidden_H_half_compacted': (384, 0)}
    result = {}
    for name, (wbytes, mbytes) in modes.items():
        per = wbytes+mbytes+30
        max_rows = 8192//per
        power2 = 1 << (max_rows.bit_length()-1)
        result[name] = {'W_bytes_per_row': wbytes, 'mask_bytes_per_row': mbytes,
            'tau_bytes_per_row': 30, 'total_bytes_per_row': per,
            'capacity_only_max_rows': max_rows, 'largest_power2_rows': power2,
            'F16_total_bytes_per_tile': 16*per, 'F32_total_bytes_per_tile': 32*per}
    result['shared_mask_once_across_eight_W_tiles_capacity_only'] = {
        'shared_2of4_F32_total_W_tau_mask_bytes': 8*32*(192+30)+32*36,
        'shared_2of4_F32_average_bytes_per_tile': (8*32*(192+30)+32*36)//8,
        'shared_C16_F32_total_W_tau_mask_bytes': 8*32*(192+30)+32*3,
        'warning': 'Requires explicit distributed/shared mask storage and its ports/broadcast. Not a free metadata pool or modeled schedule.'}
    return result


def main():
    params = read_torch(ROOT/'algorithm/stage2_temporal_codes/integer_parameters.pt')
    records = []
    sums = {}
    channel_sums = {}
    channel_original_energy = 0
    for b in range(6):
        prefix = f'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.{b}.mlp.'
        w = np.asarray(params[prefix]['weight_int8']).astype(np.int16)
        H, C = w.shape
        capture = next((ROOT/'algorithm/direct_code_integer/deployment/capture10').glob(f'v000_*_s2b{b}.npz'))
        codes = np.load(capture)['codes']
        assert w.shape == (1536, 384) and codes.shape == (1200, 384)
        row_mask = top2_mask(np.abs(w).reshape(H, C//4, 4)).reshape(H, C)
        scores = np.abs(w).reshape(H//8, 8, C//4, 4).sum(axis=1)
        shared = np.repeat(top2_mask(scores)[:, None], 8, axis=1).reshape(H, C)
        w2 = w.astype(np.int64)**2
        block_scores = w2.reshape(H//8, 8, C//16, 16).sum(axis=(1, 3))
        coarse = top2_mask(block_scores, C//32)
        coarse = np.broadcast_to(coarse[:, None, :, None], (H//8, 8, C//16, 16)).reshape(H, C)
        masks = {'original': np.ones_like(w, dtype=bool),
                 'row_2of4': row_mask, 'broadcast8_shared_2of4': shared,
                 'broadcast8_shared_C16_half': coarse}
        variants = {}
        for name, mask in masks.items():
            counts = count_mask(w, mask, codes)
            variants[name] = with_rates(counts)
            if name not in sums:
                sums[name] = {k: 0 for k in counts}
            for key, value in counts.items():
                sums[name][key] += value
        # Delete whole hidden channels, rather than leaving tau-induced gates.
        selected_h = np.argsort(-w2.sum(axis=1), kind='stable')[:H//2]
        selected_h.sort()
        packed_w = w[selected_h]
        channel = count_mask(packed_w, np.ones_like(packed_w, dtype=bool), codes)
        for key, value in channel.items():
            channel_sums[key] = channel_sums.get(key, 0)+value
        channel_original_energy += int(w2.sum())
        records.append({'block': b, 'weight_shape': [H, C], 'capture': capture.name,
                        'variants': variants, 'hidden_H_half_control': {
                            'packed_weight_shape': list(packed_w.shape),
                            'selected_h': selected_h.tolist(),
                            'W_squared_energy_retained_vs_original': float(w2[selected_h].sum()/w2.sum()),
                            'packed_counts': with_rates(channel)}})
    aggregate = {k: with_rates(v) for k, v in sums.items()}
    channel_aggregate = with_rates(channel_sums)
    channel_aggregate['W_squared_energy_retained_vs_original'] = channel_sums['retained_W_squared_sum']/channel_original_energy
    channel_aggregate['original_FC1_slots_retained'] = 0.5
    for q in list(aggregate.values())+[channel_aggregate]:
        for key in ('v000_source_active_rows_after_weight_union_all_Hgroups',
                    'v000_W_scalar_reads_with_source_and_weight_metadata',
                    'v000_static_source64_reads_with_weight_union_only'):
            q[key+'_ratio_vs_original'] = q[key]/aggregate['original'][key]
    result = {
        'kind': 'UNTRAINED_WEIGHT_MASK_AND_METADATA_OPPORTUNITY',
        'scope': 'six S2 FC1 real integer W; v000 direct integer source codes held fixed',
        'mask_rule': 'row C4 top2 abs; shared-eight C4 top2 sumabs; shared-eight C16 retain12 of24 by sum W^2; lower index wins ties',
        'layout': 'h = first + f*8 + tile; each broadcast group is eight consecutive H rows',
        'budget': 'all three pruning masks retain exactly 50% coefficient slots; retained original zero entries can make actual nnz differ',
        'physical_source_read': 'P4, 12 bits per c, packed 64-bit source bank; unique words needed by static W-union per group. Zero codes cannot be known without source read or paid metadata.',
        'dynamic_counts': 'ideal logical source/weight intersection using actual v000 codes; not an implementation of mask fetch/decoder, NR4 scheduling or ports',
        'unchanged_costs': ['full source DMA per F_cache stripe unless its complete union is separately used',
                            'full noncausal T10 consumer for each retained h',
                            'source producer unless all H weights of c are zero',
                            'FC2/BN2/shortcut and final dense optical-flow output'],
        'limits': ['no pruning training or AEE evaluation', 'fixed captured codes are not a pruned-network forward',
                   'no measured or modeled cycle speedup',
                   'W^2 retention is an algebraic metric, not retained accuracy or signal energy',
                   'capacity arithmetic includes mask/tau, but their port/decode schedule is unmodeled'],
        'aggregate': aggregate, 'records': records,
        'hidden_H_half_control': {'aggregate': channel_aggregate,
            'rule': 'per layer keep highest W^2 768 of1536 output rows, compact them, delete corresponding full T10 neurons and FC2 input columns',
            'saved_per_layer': {'FC1_weight_slots': 768*384, 'full_T10_hidden_neurons_per_position': 768,
                                'FC2_input_columns': 768, 'FC2_weight_slots_if_output_C384': 768*384},
            'unchanged': ['all384 direct source producers', 'C384 packed-source width', 'remaining noncausal T10 definition', 'FC2 output shape, dynamic BN2 and shortcut'],
            'not_equivalent_to': 'zeroing FC1 rows and retaining tau-induced constant hidden outputs'},
        'W8KiB_capacity': capacity_table(),
    }
    out = ROOT/'psn/pruning_broadcast_probe.json'
    out.write_text(json.dumps(result, indent=2)+'\n')
    for name, q in aggregate.items():
        print(name, json.dumps({k: q[k] for k in ('weight_zero_rate', 'source_column_union_rate',
            'all_eight_zero_C4_block_rate', 'all_eight_zero_C16_block_rate',
            'v000_logical_source_rows_retained', 'static_source64_words_retained', 'W_squared_energy_retained',
            'F_cache16_union_column_rate', 'F_cache16_source64_words_retained',
            'v000_W_scalar_reads_with_source_and_weight_metadata')}))
    print('hidden_H_half', json.dumps({k: channel_aggregate[k] for k in (
        'W_squared_energy_retained_vs_original', 'v000_source_active_rows_after_weight_union_all_Hgroups_ratio_vs_original',
        'v000_W_scalar_reads_with_source_and_weight_metadata_ratio_vs_original',
        'v000_static_source64_reads_with_weight_union_only_ratio_vs_original')}))
    print(out)


if __name__ == '__main__':
    main()
