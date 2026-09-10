"""Measured P4/H8 opportunity for uncertain-output-directed Conv production.

This is a lossy statistical decision experiment, not a cycle model. All axes
use the same row34 student, five Y columns plus one working U, true zero-source
columns, and train-selected time order. The ordinary per-output predicate plus
dependency short circuit is exactly the proposed semantic control; novelty must
come from an efficient implementation, not the Boolean dependency closure.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'bn_state'))
from support_service_model import read_torch

ORDER = [0, 2, 5, 7, 8, 1, 3, 6, 4, 9]
POPCOUNT = np.array([i.bit_count() for i in range(1024)])
BITS = 1 << np.arange(10)


def residual_tables(a, mean, cov):
    tables_mean, tables_sd = [], []
    for seen in range(1024):
        remaining = a * ((seen & BITS) == 0)[None, :]
        tables_mean.append(remaining @ mean)
        tables_sd.append(np.sqrt(np.maximum(0.,
            np.einsum('ij,jk,ik->i', remaining, cov, remaining))))
    return np.array(tables_mean), np.array(tables_sd)


def measure(sample, a, bias, theta, tm, ts, mode, gamma, capacity=5, first_batch_size=None):
    # One shared context is four adjacent x locations and eight output channels.
    y = sample['Y'].reshape(10, 12, 8, 64, 4).transpose(3, 1, 0, 2, 4)
    y = y.reshape(768, 10, 32).astype(np.float64)
    words = np.repeat(sample['source_words'][:, None, :], 12, 1).reshape(768, 864)
    activity = np.repeat(sample['source_active_terms'].sum(1)[:, None, :], 12, 1)
    activity = activity.reshape(768, 10).astype(np.int64) * 8
    nonzero_columns = sample['source_active_terms'].transpose(0, 2, 1) != 0
    nonzero_columns = np.broadcast_to(nonzero_columns[:, None, :, None, :],
        (64, 12, 10, 8, 4)).reshape(768, 10, 32)
    lane_activity = np.broadcast_to(
        sample['source_active_terms'].transpose(0, 2, 1)[:, None, :, None, :],
        (64, 12, 10, 8, 4)).reshape(768, 10, 32)
    support = a != 0
    deps = (support * BITS).sum(1).astype(np.int64)
    full_h = np.einsum('ts,nsq->ntq', a, y) + bias[None, :, None]
    teacher = full_h >= theta
    # Entire source window empty across the true P4 is a known BN constant,
    # available without a Conv or allocated Y. Grant this to every control.
    zero = np.zeros(768, dtype=np.int64)
    for t in range(10):
        zero |= (np.count_nonzero(words & (1 << t), axis=1) == 0).astype(np.int64) << t
    seen = zero.copy()
    unresolved = np.ones_like(teacher)
    answer = np.zeros_like(teacher)
    produced = np.zeros(768, dtype=np.int64)
    issued_batches = np.zeros(768, dtype=np.int64)
    sums = dict(conv1_active_terms=0, logical_W_vector_uses=0,
                lane_gated_conv1_active_terms=0,
                PSN_nonzero_coefficient_evaluations=0, confidence_comparisons=0,
                PSN_evaluations_excluding_known_raw_zero_columns=0,
                peak_allocated_Y_columns=0, completed_with_prior_only=0)
    # Reconstructing from retained Y is the same one-working-U family as the
    # strong baseline. Recomputing partial predicates is charged explicitly.
    for stage in range(11):
        live_context = unresolved.any(axis=(1, 2))
        if not np.any(live_context):
            break
        for state in np.unique(seen[live_context]):
            ix = np.flatnonzero(live_context & (seen == state))
            observed = (state & BITS) != 0
            partial = np.einsum('ts,nsq->ntq', a[:, observed], y[ix][:, observed])
            partial += bias[None, :, None]
            exact = (deps & ~int(state)) == 0
            if mode == 'exact':
                confident = np.broadcast_to(exact[None, :, None], partial.shape)
            else:
                pred = partial + tm[state][None, :, None]
                sd = ts[state][None, :, None]
                confident = ((pred - gamma*sd >= theta) |
                             (pred + gamma*sd < theta))
                confident[:, exact, :] = True
                sums['confidence_comparisons'] += int((unresolved[ix] & ~exact[None, :, None]).sum())*2
                if mode == 'whole_word':
                    # Each (p,h) T10 word retires independently and remains
                    # retired even while another word shares its W request.
                    all_ok = np.all(confident | ~unresolved[ix], axis=1)
                    confident = np.broadcast_to(exact[None, :, None], partial.shape).copy()
                    confident |= all_ok[:, None, :]
            eval_rows = (unresolved[ix] & (exact[None, :, None] if mode == 'exact' else True))
            sums['PSN_nonzero_coefficient_evaluations'] += int(
                (eval_rows * support[:, observed].sum(1)[None, :, None]).sum())
            raw_nonzero_terms = np.einsum('ts,nsq->ntq',
                support[:, observed].astype(np.int32),
                nonzero_columns[ix][:, observed].astype(np.int32))
            sums['PSN_evaluations_excluding_known_raw_zero_columns'] += int(
                (eval_rows * raw_nonzero_terms).sum())
            predicted = partial + tm[state][None, :, None] >= theta
            decide = unresolved[ix] & confident
            answer[ix] = np.where(decide, predicted, answer[ix])
            unresolved[ix] &= ~decide
        if stage == 0:
            sums['completed_with_prior_only'] = int((~unresolved.any(axis=(1, 2))).sum())
        remaining_context = np.flatnonzero(unresolved.any(axis=(1, 2)))
        for i in remaining_context:
            needed_rows = unresolved[i].any(axis=1)
            dependency = int(np.bitwise_or.reduce(deps[needed_rows]))
            retained = dependency & int(seen[i]) & ~int(zero[i])
            needed = dependency & ~int(seen[i])
            free = capacity-int(POPCOUNT[retained])
            if free <= 0:
                raise RuntimeError('time order requires more Y than the shared capacity')
            if mode == 'individual_no_growth' and issued_batches[i] > 0:
                free = min(free, 1)
            if first_batch_size is not None and issued_batches[i] == 0:
                free = min(free, first_batch_size)
            columns = [t for t in ORDER if needed & (1 << t)][:free]
            batch = sum(1 << t for t in columns)
            if batch == 0:
                raise RuntimeError('unresolved gate without remaining dependencies')
            sums['peak_allocated_Y_columns'] = max(sums['peak_allocated_Y_columns'],
                int(POPCOUNT[retained])+len(columns))
            sums['conv1_active_terms'] += int(activity[i, columns].sum())
            # Ordinary SIMD lane enables may suppress a p/h accumulator whose
            # consumers have all retired, even while the same W vector is read.
            wanted_lanes = np.einsum('ts,tq->sq', support.astype(np.int32),
                unresolved[i].astype(np.int32)) > 0
            sums['lane_gated_conv1_active_terms'] += int(
                (lane_activity[i, columns] * wanted_lanes[columns]).sum())
            sums['logical_W_vector_uses'] += int(np.count_nonzero(words[i] & batch))
            seen[i] |= batch
            produced[i] |= batch
            issued_batches[i] += 1
    if np.any(unresolved):
        raise RuntimeError('incomplete decision experiment')
    wrong = answer != teacher
    false_negative = teacher & ~answer
    sums.update(
        contexts=768, gates=int(teacher.size), teacher_spikes=int(teacher.sum()),
        predicted_spikes=int(answer.sum()), wrong_gates=int(wrong.sum()),
        false_negative_gates=int(false_negative.sum()),
        contexts_with_any_gate_error=int(wrong.any(axis=(1, 2)).sum()),
        produced_time_columns=int(POPCOUNT[produced].sum()),
        issued_batches=int(issued_batches.sum()),
        selected_W_uses_if_all_selected_times_shared_at_once=int(np.count_nonzero(words & produced[:, None])),
        dense_time_batched_W_uses=int(np.count_nonzero(words)),
        full_conv1_active_terms=int(activity.sum()),
    )
    return sums


def main():
    captured = read_torch(HERE/'capture.pt')
    v = json.loads((HERE.parent/'dependency/fit.json').read_text())['variants']['row34']
    a, bias = np.array(v['weight']), np.array(v['bias']).reshape(10)
    stats = np.load(HERE.parent/'dependency/train_moments.npz')
    theta = float(captured['metadata']['neuron_theta'])
    tm, ts = residual_tables(a, stats['mean'], stats['covariance'])
    axes = [('exact', None)] + [(mode, gamma)
        for gamma in (4., 3., 2.5, 2.)
        for mode in ('whole_word', 'individual_no_growth', 'individual')]
    axes += [('whole_word', 1.5), ('whole_word', 1.)]
    out = dict(
        scope='20 real frames, 64 adjacent P4 groups/frame, full C96/T10; not complete-layer AEE or RTL',
        student='same row34, fixed four patch BNs; Float64 partial evaluation of captured FP32 inputs',
        prediction='statistical remaining-PSN mean +/- gamma SD, trained from train32 full moments; not certified intervals',
        selection='gamma is an exploratory staged-validation sweep; 1.5/1 whole-word controls added after the initial 4/3/2.5/2 screen; moment and order fitting remain train-only',
        method='same 5 Y columns plus 1 working U; true empty source columns granted free to all controls',
        time_order=ORDER, threshold=theta,
        prior_boundary='individual mode is also ordinary per-output prediction plus sparse dependency short-circuit; no independent semantic novelty claim',
        parameters=dict(distinct_row_subset_table_entries=sum(2**int(n) for n in (a != 0).sum(1)),
                        learned_predictor_multiplications_on_stored_BN_Y=0,
                        note='two residual constants per entry on stored BN Y; folding BN and raw-zero compensation may require additional per-channel constants or arithmetic, not included in this table size'),
        exclusions=['SRAM banking/cache residency, input rereads and controller latency',
                    'physical arithmetic/precision/area/energy',
                    'complete Conv2/BN2/shortcut and full optical-flow accuracy',
                    'BN compensation/table traffic for the separately shown raw-zero PSN arithmetic control',
                    'full spatial coverage; sampled locations are fixed before capture'],
        axes={},
    )
    for mode, gamma in axes:
        name = mode if gamma is None else f'{mode}_g{gamma:g}'
        frame_rows = []
        for sample in captured['samples']:
            row = measure(sample, a, bias, theta, tm, ts, mode, gamma)
            frame_rows.append(dict(file=sample['file'], split=sample['split'], **row))
        axis = dict(frames=frame_rows, splits={})
        for split in ('train', 'valid'):
            rows = [r for r in frame_rows if r['split'] == split]
            sums = {k: (max(r[k] for r in rows) if k == 'peak_allocated_Y_columns' else sum(r[k] for r in rows))
                    for k in rows[0] if k not in ('file', 'split')}
            sums['gate_error_fraction'] = sums['wrong_gates']/sums['gates']
            sums['false_negative_fraction_of_active'] = sums['false_negative_gates']/sums['teacher_spikes']
            sums['conv1_active_fraction'] = sums['conv1_active_terms']/sums['full_conv1_active_terms']
            sums['lane_gated_conv1_active_fraction'] = sums['lane_gated_conv1_active_terms']/sums['full_conv1_active_terms']
            if mode != 'exact':
                base = out['axes']['exact']['splits'][split]
                sums['W_uses_over_same_capacity_exact'] = sums['logical_W_vector_uses']/base['logical_W_vector_uses']
                sums['PSN_evaluations_over_exact'] = sums['PSN_nonzero_coefficient_evaluations']/base['PSN_nonzero_coefficient_evaluations']
                sums['PSN_raw_nonzero_evaluations_over_exact'] = sums['PSN_evaluations_excluding_known_raw_zero_columns']/base['PSN_evaluations_excluding_known_raw_zero_columns']
            sums['frame_count'] = len(rows)
            axis['splits'][split] = sums
        out['axes'][name] = axis
        print(name, json.dumps(axis['splits']['valid']), flush=True)
    (HERE/'probe_results.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
