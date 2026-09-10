"""One fixed local-mask experiment; selection on train32, no GT/AEE claim."""
from pathlib import Path
import itertools
import json
import time
import numpy as np
import torch
from probe_weight_support import ALG, OUT, FRAMES, POP


def forest_cost(raw, masks):
    """Full single-parent subset forest for each mask; one read per parent."""
    n = len(raw)
    e = masks[:, None] & raw[None, :]
    pc = POP[e]
    legal = ((e[:, None, :] & e[:, :, None]) == e[:, None, :]) & (
        (pc[:, None, :] < pc[:, :, None]) |
        ((pc[:, None, :] == pc[:, :, None]) &
         (np.arange(n)[None, :] < np.arange(n)[:, None])))
    best_parent_gain = np.maximum(np.where(legal, pc[:, None, :] - 1, -1).max(-1), 0)
    parent_reads = (best_parent_gain > 0).sum(-1)
    cost = (pc - best_parent_gain).sum(-1)
    return cost, cost-parent_reads, parent_reads, pc.sum(-1)


def pack(gates):
    return (gates.astype(np.uint32) * (1 << np.arange(16, dtype=np.uint32))).sum(-1).astype(np.uint16)


def main():
    start = time.time()
    train = torch.load(ALG / 'nrv_cost_probe/s2b3_train.pt', map_location='cpu', weights_only=False)
    train_codes = train['start_codes'].numpy()
    flat = train_codes.reshape(-1, 384)
    dictionary = np.load(ALG / 'stage2_temporal_codes/codebooks.npz')['s2b3_dictionary']
    params = np.load(ALG / 'group_pruning_probe/controls/original_trained.npz')
    w = params['weight_int8'].astype(np.float64)
    B, E = params['B_int8'].astype(np.float64), params['E_int8'].astype(np.float64)
    tau = params['threshold_int32'].astype(np.float64)
    pairs = [sum(1 << i for i in c) for c in itertools.combinations(range(4), 2)]
    mask_words = np.array([sum(q << (4*i) for i, q in enumerate(parts))
                           for parts in itertools.product(pairs, repeat=4)], dtype=np.uint16)
    keep = ((mask_words[:, None].astype(np.uint32) >> np.arange(16)) & 1).astype(bool)
    drop = (~keep).astype(np.float64)
    drop_outer = np.einsum('bi,bj->bij', drop, drop).reshape(len(keep), 256)
    selected = {name: np.zeros((192, 384), dtype=bool)
                for name in ['shared_magnitude', 'shared_membrane', 'shared_activity_cost', 'shared_product_cost']}
    selected_ids = {name: [] for name in selected}
    block_records = []
    total_local_error = {name: 0.0 for name in selected}
    total_train_cost = {name: 0 for name in selected}

    for kblock in range(24):
        sl = slice(16*kblock, 16*(kblock+1))
        # All train32 source positions, all T10, full actual B/E transform.
        z = np.einsum('tr,pcr->ptc', B, E[flat[:, sl]], optimize=True).reshape(-1, 16)
        covariance = z.T @ z / len(z)
        wg = w[:, sl].reshape(192, 8, 16)
        gram_w = np.einsum('ghi,ghj->gij', wg, wg, optimize=True)
        errors = (gram_w * covariance).reshape(192, 256) @ drop_outer.T
        errors = np.maximum(errors, 0)
        magnitude = np.einsum('ghi,bi->gb', wg*wg, drop, optimize=True)
        graph_cost = np.zeros(len(mask_words), dtype=np.int64)
        activity_cost = np.zeros(len(mask_words), dtype=np.int64)
        # All captured native P4 groups, complete T10: local M40 forests.
        for frame_codes in train_codes:
            for group_codes in frame_codes:
                gates = dictionary[group_codes[:, sl]].transpose(0, 2, 1).reshape(40, 16)
                counts = forest_cost(pack(gates), mask_words)
                graph_cost += counts[0]
                activity_cost += counts[3]
        best_error = errors.min(-1)
        ordinary_id = errors.argmin(-1)
        magnitude_id = magnitude.argmin(-1)
        eligible = errors <= best_error[:, None] * 1.05 + 1e-9
        # Within the fixed local-error budget choose cost, break ties on error.
        min_cost = np.where(eligible, graph_cost[None, :], np.iinfo(np.int64).max).min(-1)
        chosen_id = np.where(eligible & (graph_cost[None, :] == min_cost[:, None]), errors, np.inf).argmin(-1)
        min_activity = np.where(eligible, activity_cost[None, :], np.iinfo(np.int64).max).min(-1)
        activity_id = np.where(eligible & (activity_cost[None, :] == min_activity[:, None]), errors, np.inf).argmin(-1)
        ids = {'shared_magnitude': magnitude_id, 'shared_membrane': ordinary_id,
               'shared_activity_cost': activity_id, 'shared_product_cost': chosen_id}
        for name, index in ids.items():
            selected[name][:, sl] = keep[index]
            selected_ids[name].append(index.tolist())
            total_local_error[name] += float(errors[np.arange(192), index].sum())
            total_train_cost[name] += int(graph_cost[index].sum())
        block_records.append({'kblock': kblock,
                              'changed_groups_vs_membrane': int(np.count_nonzero(chosen_id != ordinary_id)),
                              'eligible_masks_median': float(np.median(eligible.sum(-1)))})
        if kblock % 6 == 5:
            print(json.dumps({'kblock': kblock+1, 'elapsed_s': round(time.time()-start, 1)}), flush=True)

    axis = {name: {'local_train_membrane_squared_error': total_local_error[name],
                   'train_logical_access_proxy': total_train_cost[name],
                   'kept_mask_entries': int(selected[name].sum()*8),
                   'actual_weight_nonzeros': int(np.count_nonzero(w * selected[name].repeat(8, axis=0))),
                   'heldout': []} for name in selected}
    for frame in FRAMES:
        codes = np.load(ALG / 'group_pruning_probe/controls/codes/row_2of4_trained' / frame / 's2b3.npz')['codes']
        positions = np.linspace(0, len(codes)-1, 32, dtype=int)
        c = codes[positions]
        gates = dictionary[c].transpose(0, 2, 1).reshape(-1, 384)
        z = np.einsum('tr,pcr->ptc', B, E[c], optimize=True).reshape(-1, 384)
        reference = z @ w.T
        threshold = np.tile(tau, (len(positions), 1))
        for name, mask in selected.items():
            student = z @ (w * mask.repeat(8, axis=0)).T
            counts = np.zeros(4, dtype=np.int64)
            for kblock in range(24):
                sl = slice(kblock*16, (kblock+1)*16)
                mw = pack(mask[:, sl])
                for pos in range(0, len(gates), 40):
                    counts += np.array([x.sum() for x in forest_cost(pack(gates[pos:pos+40, sl]), mw)])
            axis[name]['heldout'].append({'frame': frame,
                'logical_access_proxy': int(counts[0]),
                'coefficient_vector_uses': int(counts[1]),
                'parent_vector_reads': int(counts[2]),
                'direct_activity_vector_uses': int(counts[3]),
                'full_C_membrane_squared_error': float(np.square(student-reference).sum()),
                'teacher_membrane_squared_norm': float(np.square(reference).sum()),
                'gate_mismatches': int(np.count_nonzero((student >= threshold) != (reference >= threshold))),
                'gate_false_positive': int(np.count_nonzero((student >= threshold) & (reference < threshold))),
                'gate_false_negative': int(np.count_nonzero((student < threshold) & (reference >= threshold))),
                'gates': int(reference.size)})

    result = {
        'kind': 'OFFLINE_MASK_SELECTION_ONLY_NO_TRAINING_NO_AEE',
        'identity': 'Same existing integer bits3 s2b3 student. Source theta is in W; B/E implement actual noncausal T10; tau is distinct. No GT recovery or new network evaluation.',
        'selection': 'Fixed H8 contiguous groups and K16 blocks, each K4 retains two shared source positions: 1296 masks. Four axes share candidate set, source, W and downstream transform. Activity and product-cost axes have exactly the same 1.05 local-error feasible set.',
        'candidate_budget': 'Local K16 membrane squared error <=1.05 times its best shared-mask error. This is not an AEE tolerance.',
        'train_error_scope': 'All 32 training frames, all 32 native P4 groups per frame, all T10. Sum of K16 errors ignores cross-block error correlation; full-C heldout error is measured separately.',
        'train_cost_scope': 'All 32 native P4 groups per each of train32 frames, full T10 M40; effective-mask Prosperity subset forest, stable popcount/index. Same graph optimizer for all axes.',
        'heldout_scope': 'Four fixed distinct frames, 32 uniformly sampled positions/frame, full T10/C384/H1536; groups of four sampled positions for local M40 statistics, not a spatially contiguous hardware schedule.',
        'limits': ['Logical coefficient H8 vector uses plus parent reads; no physical packing/caching, detector, parent writes, complete K/PSN schedule or service cycles.',
                   'Four evaluation frames were previously used for research problem-finding; held out from mask selection here, not a new blinded test set.',
                   'Ordinary shared-membrane control is block-local and does not include full OBS/HiNM/PRAP-PIM joint optimization, task recovery or common-node CSE.',
                   'Masks have equal retained slots; exact W8 zero counts can differ and are reported.',
                   'This diagnostic block can be cheaply narrowed/deleted in prior task tests; results do not select it as a publication target.'],
        'train_frames': train['frames'], 'heldout_frames': FRAMES, 'axes': axis,
        'block_records': block_records, 'elapsed_s': time.time()-start}
    (OUT / 'shared_support_selection_result.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(OUT / 'shared_support_selected_masks.npz', **selected)
    print(json.dumps({'complete': True, 'elapsed_s': round(time.time()-start,1)}), flush=True)


if __name__ == '__main__':
    main()
