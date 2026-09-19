"""Summarize completed paired inference and successful-update receipts."""
import json
from pathlib import Path
import re
from statistics import median

HERE = Path(__file__).resolve().parent


def main():
    training = json.loads((HERE / 'RECOVERY.json').read_text())
    paired = json.loads((HERE / 'PAIRED_VALID825.json').read_text())
    assert training['complete'] and paired['complete']
    reference = paired['arms']['ep34']['rows']
    assert len(reference) == 825 and len({r['file'] for r in reference}) == 825
    sequence_ids = [re.search(r'(?:zurich_city_\d+_[a-z]|interlaken_\d+_[a-z]|thun_\d+_[a-z])', r['file']).group()
                    for r in reference]
    summary = dict(frames=len(reference), sequences=len(set(sequence_ids)),
                   valid_pixels=sum(r['pixels'] for r in reference),
                   trainable_parameters=training['trainable_parameters'],
                   baseline_aee=paired['arms']['ep34']['aee_frame_mean'],
                   baseline_firing=paired['arms']['ep34']['element_weighted_firing'],
                   baseline_zero_firing_modules=paired['arms']['ep34']['zero_firing_modules_observed'],
                   baseline_observed_modules=len(paired['arms']['ep34']['activity']),
                   arms=[])
    baseline_zero = set(summary['baseline_zero_firing_modules'])
    for arm in training['arms']:
        name = arm['rule']
        result = paired['arms'][name]
        assert len(result['rows']) == len(reference)
        by_sequence = {}
        for seq, r0, r1 in zip(sequence_ids, reference, result['rows']):
            assert r0['file'] == r1['file'] and r0['pixels'] == r1['pixels']
            entry = by_sequence.setdefault(seq, dict(frames=0, baseline_sum=0., candidate_sum=0.))
            entry['frames'] += 1
            entry['baseline_sum'] += r0['aee']; entry['candidate_sum'] += r1['aee']
        sequences = [dict(sequence=seq, frames=v['frames'],
                          baseline_aee=v['baseline_sum'] / v['frames'],
                          candidate_aee=v['candidate_sum'] / v['frames'],
                          delta=(v['candidate_sum'] - v['baseline_sum']) / v['frames'])
                     for seq, v in sorted(by_sequence.items())]
        frame_deltas = [dict(file=r1['file'], baseline_aee=r0['aee'],
                             candidate_aee=r1['aee'], delta=r1['aee'] - r0['aee'])
                        for r0, r1 in zip(reference, result['rows'])]
        best_frame = min(frame_deltas, key=lambda r: r['delta'])
        worst_frame = max(frame_deltas, key=lambda r: r['delta'])
        deltas = [r['delta'] for r in frame_deltas]
        row = dict(rule=name, successful_updates=arm['successful_updates'],
                   overflow_retries=arm['overflow_attempts'],
                   first_batch_loss=arm['updates'][0]['attempts'][-1]['loss'],
                   ten_frame_aee=arm['diagnostics'][-1]['validation']['aee_frame_mean'],
                   valid825_aee=result['aee_frame_mean'],
                   delta_aee=result['aee_frame_mean'] - summary['baseline_aee'],
                   pixel_weighted_aee=result['aee_pixel_weighted'],
                   changed_theta=sum(v['before'] != v['after'] for v in arm['theta'].values()),
                   changed_q_theta=sum(v['before'] != v['after'] for n, v in arm['theta'].items() if '.sn_q.' in n),
                   max_theta_change=max(abs(v['before'] - v['after']) for v in arm['theta'].values()),
                   final_q_nonzero_theta_gradient=arm['updates'][-1]['q_nonzero_theta_grad'],
                   q_gradient_steps=sum(r['q_nonzero_theta_grad'] > 0 for r in arm['updates']),
                   changed_psn_matrices=sum(v > 0 for n, v in arm['parameter_max_change'].items() if n.endswith('.weight')),
                   element_weighted_firing=result['element_weighted_firing'],
                   observed_modules=len(result['activity']),
                   min_nonzero_frames_per_module=min(r['nonzero_frames'] for r in result['activity'].values()),
                   zero_firing_modules=result['zero_firing_modules_observed'],
                   newly_active_modules=sorted(baseline_zero - set(result['zero_firing_modules_observed'])),
                   newly_zero_modules=sorted(set(result['zero_firing_modules_observed']) - baseline_zero),
                   improved_sequences=sum(r['delta'] < 0 for r in sequences),
                   worsened_sequences=sum(r['delta'] > 0 for r in sequences),
                   paired_frame_median_delta=median(deltas),
                   improved_frames=sum(d < 0 for d in deltas),
                   worsened_frames=sum(d > 0 for d in deltas),
                   best_improvement_frame=best_frame, worst_regression_frame=worst_frame,
                   posthoc_leave_best_improvement_frame_out_delta=
                       (sum(deltas) - best_frame['delta']) / (len(deltas) - 1),
                   sequences=sequences)
        summary['arms'].append(row)
    (HERE / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({**summary, 'arms': [{k: v for k, v in a.items() if k not in {'sequences', 'zero_firing_modules'}}
                                        for a in summary['arms']]}, indent=2))


if __name__ == '__main__':
    main()
