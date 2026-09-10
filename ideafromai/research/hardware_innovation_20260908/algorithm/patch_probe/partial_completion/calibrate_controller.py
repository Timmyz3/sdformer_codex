"""Bounded controller calibration on the existing real P4/H8 capture.

Both objectives use exactly probe.measure(mode='individual', capacity=5), the
fixed row34 A/bias/neuron theta, the same train32 residual-moment tables, and
the same ten per-output gamma parameters. This is CPU coordinate calibration,
not network gradient training, a cycle model, or an optical-flow AEE result.

Use all 16 pre-existing training frames and all 64 adjacent P4 groups in each.
Uniform gamma=3 defines joint wrong-gate and false-negative upper limits.
Start at gamma=4. In each round, evaluate reducing exactly
one coordinate by one step in [4, 3.5, 3, 2.5, 2]. Accept the feasible candidate
with the smallest objective only if it strictly improves the current value;
ties use the original output-t index, with no secondary cost objective.
The initial eight-round run is retained as history. The authorized continuation
stops at no feasible strict improvement or at forty total accepted steps per
objective (all ten coordinates at their lower bounds), not a global optimum.

The first objective minimizes raw-nonzero PSN coefficient evaluations; the
second minimizes logical shared W-vector uses. Repeated configurations share
their training evaluation. All four validation frames are evaluated only after
both searches finish. Their results never select a parameter or a candidate.
The raw-zero arithmetic count inherits probe's explicit BN-compensation and
table-traffic exclusions; neither objective represents complete service cost.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import json
import argparse
from pathlib import Path
import time

import numpy as np

import probe


HERE = Path(__file__).resolve().parent
LEVELS = (4.0, 3.5, 3.0, 2.5, 2.0)
MAX_ROUNDS = 40
OBJECTIVES = {
    'independent_neuron_cost': 'PSN_evaluations_excluding_known_raw_zero_columns',
    'shared_production_cost': 'logical_W_vector_uses',
}


def aggregate(rows):
    total = {
        key: (max(row[key] for row in rows)
              if key == 'peak_allocated_Y_columns'
              else sum(row[key] for row in rows))
        for key in rows[0]
    }
    total['frame_count'] = len(rows)
    total['gate_error_fraction'] = total['wrong_gates'] / total['gates']
    total['false_negative_fraction_of_active'] = (
        total['false_negative_gates'] / total['teacher_spikes'])
    total['conv1_active_fraction'] = (
        total['conv1_active_terms'] / total['full_conv1_active_terms'])
    total['lane_gated_conv1_active_fraction'] = (
        total['lane_gated_conv1_active_terms'] / total['full_conv1_active_terms'])
    return total


def compare_costs(ordinary, shared):
    return {
        'shared_over_independent_W_uses': (
            shared['logical_W_vector_uses'] / ordinary['logical_W_vector_uses']),
        'shared_over_independent_PSN_raw_evaluations': (
            shared['PSN_evaluations_excluding_known_raw_zero_columns'] /
            ordinary['PSN_evaluations_excluding_known_raw_zero_columns']),
        'shared_over_independent_Conv1_terms': (
            shared['conv1_active_terms'] / ordinary['conv1_active_terms']),
        'shared_over_independent_lane_gated_Conv1_terms': (
            shared['lane_gated_conv1_active_terms'] /
            ordinary['lane_gated_conv1_active_terms']),
        'shared_over_independent_confidence_comparisons': (
            shared['confidence_comparisons'] / ordinary['confidence_comparisons']),
        'shared_minus_independent_wrong_gates': (
            shared['wrong_gates'] - ordinary['wrong_gates']),
        'shared_minus_independent_false_negatives': (
            shared['false_negative_gates'] - ordinary['false_negative_gates']),
    }


def strongest_available_ordinary(final, budget):
    # The uniform gamma3 used to define the error budget is itself a feasible
    # simple control. A local coordinate path must not artificially weaken it.
    choices = ['uniform_gamma4', 'uniform_gamma3_budget', 'independent_neuron_cost']
    feasible = [name for name in choices if all(
        final[name]['train'][key] <= limit for key, limit in budget.items())]
    chosen = min(feasible, key=lambda name:
                 final[name]['train']['PSN_evaluations_excluding_known_raw_zero_columns'])
    return {
        'selection': 'smallest training PSN raw-nonzero count among already evaluated ordinary controls, subject to the same training error limits',
        'candidate_controls': choices,
        'selected_control': chosen,
        'reason': 'finite coordinate search is not a global optimum; do not omit the feasible uniform gamma3 control',
        'shared_vs_selected': {
            split: compare_costs(final[chosen][split], final['shared_production_cost'][split])
            for split in ('train', 'valid')
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true',
                        help='reuse all saved training configurations and continue each path')
    args = parser.parse_args()
    started = time.perf_counter()
    output_path = HERE / 'controller_calibration.json'
    previous = json.loads(output_path.read_text()) if args.resume else None
    captured = probe.read_torch(HERE / 'capture.pt')
    fit_path = HERE.parent / 'dependency/fit.json'
    moments_path = HERE.parent / 'dependency/train_moments.npz'
    variant = json.loads(fit_path.read_text())['variants']['row34']
    a = np.array(variant['weight'])
    bias = np.array(variant['bias']).reshape(10)
    theta = float(captured['metadata']['neuron_theta'])
    moments = np.load(moments_path)
    tm, ts = probe.residual_tables(a, moments['mean'], moments['covariance'])
    train = [sample for sample in captured['samples'] if sample['split'] == 'train']
    valid = [sample for sample in captured['samples'] if sample['split'] == 'valid']

    def run_frames(samples, gamma, mode='individual'):
        shaped = None if gamma is None else np.array(gamma).reshape(1, 10, 1)
        rows = [probe.measure(sample, a, bias, theta, tm, ts,
                              mode, shaped, capacity=5) for sample in samples]
        return rows, aggregate(rows)

    first_row = None
    if previous is None:
        benchmark_start = time.perf_counter()
        first_row = probe.measure(train[0], a, bias, theta, tm, ts,
                                  'individual', np.full((1, 10, 1), 4.), capacity=5)
        single_frame_seconds = time.perf_counter() - benchmark_start
        benchmark = {
            'single_train_frame_seconds': single_frame_seconds,
            'estimated_160_config_train16_seconds': single_frame_seconds * 160 * 16,
            'decision': 'use all 16 predetermined training frames',
        }
        print('benchmark', json.dumps(benchmark), flush=True)
    else:
        benchmark = previous['benchmark']

    cache = {}
    records = []
    history = []
    if previous is not None:
        saved_rows = {
            tuple(control['gamma_by_output_t']): [
                {key: value for key, value in row.items() if key != 'file'}
                for row in control['train_frames']]
            for control in previous['final_controls'].values()
            if control.get('train_frames') is not None
        }
        records = previous['training_configurations']
        for record in records:
            key = tuple(record['gamma_by_output_t'])
            cache[key] = (record, saved_rows.get(key))
        if previous.get('history'):
            history = previous['history']
        else:
            history = [{
                'stage': 'initial_eight_rounds',
                'searches': previous['searches'],
                'final_controls': previous['final_controls'],
                'shared_vs_independent': previous['shared_vs_independent'],
                'strong_ordinary_reference': strongest_available_ordinary(
                    previous['final_controls'], previous['protocol']['error_budget']),
                'unique_training_configuration_count': len(records),
                'elapsed_seconds': previous['elapsed_seconds'],
            }]
        print('resume_cached_training_configurations', len(records), flush=True)
    initial_cache_count = len(records)

    def evaluate_train(gamma):
        key = tuple(gamma)
        if key in cache:
            return cache[key]
        begin = time.perf_counter()
        if key == (4.,) * 10 and first_row is not None:
            other_rows, _ = run_frames(train[1:], gamma)
            rows = [first_row] + other_rows
            totals = aggregate(rows)
        else:
            rows, totals = run_frames(train, gamma)
        record = {
            'configuration': f'c{len(records):03d}',
            'gamma_by_output_t': list(gamma),
            'train': totals,
            'evaluation_seconds': time.perf_counter() - begin,
        }
        cache[key] = (record, rows)
        records.append(record)
        return record, rows

    start_record, _ = evaluate_train((4.,) * 10)
    budget_record, _ = evaluate_train((3.,) * 10)
    budget = {key: budget_record['train'][key]
              for key in ('wrong_gates', 'false_negative_gates')}

    def feasible(record):
        return all(record['train'][key] <= limit for key, limit in budget.items())

    if not feasible(start_record):
        raise RuntimeError('Predeclared uniform gamma=4 start is infeasible')
    print('budget', json.dumps(budget), flush=True)

    searches = {}
    for name, objective in OBJECTIVES.items():
        if previous is None:
            gamma = [4.] * 10
            current = start_record
            path = []
        else:
            old = previous['searches'][name]
            gamma = old['gamma_by_output_t'].copy()
            current, _ = evaluate_train(gamma)
            path = old['path'].copy()
        stop = 'maximum_forty_total_accepted_steps'
        accepted_before = sum(step['accepted_configuration'] is not None for step in path)
        for round_index in range(accepted_before, MAX_ROUNDS):
            candidate_rows = []
            eligible = []
            for t in range(10):
                level = LEVELS.index(gamma[t])
                if level == len(LEVELS) - 1:
                    continue
                candidate = gamma.copy()
                candidate[t] = LEVELS[level + 1]
                record, _ = evaluate_train(candidate)
                allowed = feasible(record)
                improves = record['train'][objective] < current['train'][objective]
                candidate_rows.append({
                    'output_t': t,
                    'configuration': record['configuration'],
                    'feasible': allowed,
                    'strict_objective_improvement': improves,
                })
                if allowed and improves:
                    eligible.append((record['train'][objective], t, record))
            step = {
                'round': round_index + 1,
                'from_configuration': current['configuration'],
                'candidates': candidate_rows,
            }
            if not eligible:
                step['accepted_configuration'] = None
                path.append(step)
                stop = 'no_feasible_strict_improvement'
                print(name, 'stop', round_index + 1, stop, flush=True)
                break
            _, t, chosen = min(eligible, key=lambda item: (item[0], item[1]))
            gamma = chosen['gamma_by_output_t'].copy()
            current = chosen
            step['accepted_output_t'] = t
            step['accepted_configuration'] = current['configuration']
            step['accepted_gamma'] = gamma
            path.append(step)
            print(name, 'round', round_index + 1,
                  json.dumps({'gamma': gamma, 'objective': current['train'][objective],
                              **{key: current['train'][key] for key in budget}}),
                  flush=True)
        searches[name] = {
            'objective': objective,
            'start_configuration': start_record['configuration'],
            'final_configuration': current['configuration'],
            'gamma_by_output_t': gamma,
            'path': path,
            'stop_reason': stop,
        }

    # Held-out validation is first used here, after both optimization paths end.
    final_gammas = {
        'uniform_gamma4': [4.] * 10,
        'uniform_gamma3_budget': [3.] * 10,
        **{name: search['gamma_by_output_t'] for name, search in searches.items()},
    }
    final = {}
    valid_cache = {}
    if previous is not None:
        for control in previous['final_controls'].values():
            rows = [{key: value for key, value in row.items() if key != 'file'}
                    for row in control['valid_frames']]
            valid_cache[tuple(control['gamma_by_output_t'])] = (rows, control['valid'])
    for name, gamma in final_gammas.items():
        record, train_rows = evaluate_train(gamma)
        key = tuple(gamma)
        if key not in valid_cache:
            valid_cache[key] = run_frames(valid, gamma)
        valid_rows, valid_total = valid_cache[key]
        final[name] = {
            'configuration': record['configuration'],
            'gamma_by_output_t': gamma,
            'train': record['train'],
            'valid': valid_total,
            'train_frames': (None if train_rows is None else [
                dict(file=sample['file'], **row) for sample, row in zip(train, train_rows)]),
            'train_frame_detail_note': ('aggregates reused from existing configuration without recomputation; per-frame details were not retained for this candidate'
                                        if train_rows is None else 'per-frame details available'),
            'valid_frames': [dict(file=sample['file'], **row)
                             for sample, row in zip(valid, valid_rows)],
        }

    if previous is None:
        exact = {}
        for split, samples in (('train', train), ('valid', valid)):
            _, exact[split] = run_frames(samples, None, 'exact')
    else:
        exact = previous['same_capacity_exact']

    comparison = {}
    for split in ('train', 'valid'):
        ordinary = final['independent_neuron_cost'][split]
        shared = final['shared_production_cost'][split]
        comparison[split] = compare_costs(ordinary, shared)
        for values in final.values():
            row = values[split]
            row['W_uses_over_exact'] = (
                row['logical_W_vector_uses'] / exact[split]['logical_W_vector_uses'])
            row['PSN_raw_evaluations_over_exact'] = (
                row['PSN_evaluations_excluding_known_raw_zero_columns'] /
                exact[split]['PSN_evaluations_excluding_known_raw_zero_columns'])

    out = {
        'scope': 'bounded CPU controller calibration, not AEE or cycle/RTL performance',
        'sources': {
            'capture': str(HERE / 'capture.pt'),
            'measure': str(HERE / 'probe.py'),
            'fixed_row34': str(fit_path),
            'fixed_residual_moments': str(moments_path),
        },
        'protocol': {
            'mode': 'individual', 'Y_capacity': 5, 'working_U_columns': 1,
            'P': 4, 'H': 8, 'groups_per_frame': 64,
            'time_order': probe.ORDER, 'gamma_levels': LEVELS,
            'max_total_accepted_steps_per_objective': MAX_ROUNDS,
            'max_one_step_candidate_proposals': 800,
            'tie_break': 'minimum original output_t, no secondary objective',
            'error_budget_source': 'uniform gamma3, same train16',
            'error_budget': budget,
            'training_files': [sample['file'] for sample in train],
            'validation_files': [sample['file'] for sample in valid],
            'validation_policy': 'evaluate only final controls; never used in search',
            'moments_scope': 'pre-existing train32 full moments; not cross-fit',
        },
        'fixed_parameters': {
            'A': a.tolist(), 'bias': bias.tolist(), 'theta': theta,
            'residual_input_mean': moments['mean'].tolist(),
            'residual_input_covariance': moments['covariance'].tolist(),
        },
        'benchmark': benchmark,
        'history': history,
        'resume_reused_training_configurations': initial_cache_count,
        'searches': searches,
        'training_configurations': records,
        'unique_training_configuration_count': len(records),
        'final_controls': final,
        'same_capacity_exact': exact,
        'shared_vs_independent': comparison,
        'strong_ordinary_reference': strongest_available_ordinary(final, budget),
        'exclusions': [
            'statistical prediction is lossy, not a certified bound',
            'fixed Float64 local reference over captured FP32 Y, not full-network equivalence',
            'logical W-vector uses are not physical SRAM requests or service cycles',
            'raw-zero PSN arithmetic omits BN compensation and table-traffic costs',
            'input metadata scanning, source rereads, cache/bank and control latency',
            'full Conv2, BN2, shortcut, optical-flow AEE, RTL/PPA',
        ],
        'elapsed_seconds': time.perf_counter() - started,
    }
    output_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print('finished', json.dumps({'seconds': out['elapsed_seconds'],
                                 'unique_configs': len(records),
                                 'comparison': comparison}), flush=True)


if __name__ == '__main__':
    main()
