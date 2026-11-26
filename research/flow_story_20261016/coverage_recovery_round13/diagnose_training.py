#!/usr/bin/env python3
"""Post-round CPU-only optimization diagnostics; never selects quality/checkpoints."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ORIGINAL_SHA = 'd5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.size and np.isfinite(values).all()
    return dict(zip(('min', 'p25', 'median', 'p75', 'max'),
                    [float(x) for x in np.quantile(values, [0, .25, .5, .75, 1])]))


def param_summary(before, after):
    before = before.detach().cpu().double()
    after = after.detach().cpu().double()
    assert before.shape == after.shape
    assert torch.isfinite(before).all() and torch.isfinite(after).all()
    delta = after - before
    n2, d2 = float((before * before).sum()), float((delta * delta).sum())
    return {'elements': before.numel(), 'initial_squared_L2': n2,
            'delta_squared_L2': d2, 'delta_max_abs': float(delta.abs().max()),
            'changed_elements': int((delta != 0).sum()),
            'relative_L2': (d2 / n2) ** .5 if n2 else None,
            'initial_zero_norm': n2 == 0}


def combine(items):
    result = {k: sum(x[k] for x in items) for k in
              ('elements', 'initial_squared_L2', 'delta_squared_L2', 'changed_elements')}
    result['delta_max_abs'] = max(x['delta_max_abs'] for x in items)
    n2, d2 = result['initial_squared_L2'], result['delta_squared_L2']
    result['relative_L2'] = (d2 / n2) ** .5 if n2 else None
    result['initial_zero_norm'] = n2 == 0
    return result


def category_of(suffix):
    allowed = {'FC': {'fc1.weight', 'fc2.weight'},
               'BN_affine': {a + '.norm_layer.' + b for a in ('bn1', 'bn2') for b in ('weight', 'bias')},
               'PSN': {a + '.spiking_neuron.' + b for a in ('sn1', 'sn2') for b in ('weight', 'bias')}}
    matches = [category for category, names in allowed.items() if suffix in names]
    assert len(matches) == 1, suffix
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    # Admission is completion only; quality is neither read nor used in this diagnostic.
    sources = []
    for condition in ('reference', 'diversity'):
        for arm in ('native', 'FF', 'CC', 'fine_s2b2'):
            path = HERE / 'raw/experiments_v1' / condition / 'recover' / arm / 'results.json'
            r = json.loads(path.read_text())
            assert r['status'] == 'ROUND13_COVERAGE_RECOVERY_COMPLETE'
            assert r['completed_optimizer_steps'] == 200 and len(r['steps']) == 200
            assert r['condition'] == condition and r['arm'] == arm
            sources.append((path, r))
    original = HERE / 'source_snapshot/checkpoint_step_000135930.pt'
    assert sha(original) == ORIGINAL_SHA
    payload = torch.load(original, map_location='cpu', weights_only=True)
    initial = payload['model_state_dict']
    del payload
    summaries = []
    for path, r in sources:
        weights = path.parent / Path(r['weights_file']).name
        assert sha(weights) == r['weights_sha256']
        payload = torch.load(weights, map_location='cpu', weights_only=True)
        state, names = payload['model_state_dict'], r['allowed_parameter_names']
        assert names == payload['trainable_names'] and len(names) == 80
        assert sum(state[n].numel() for n in names) == 7252064
        parameters, layers, categories = {}, defaultdict(list), defaultdict(list)
        for name in names:
            p = param_summary(initial[name], state[name])
            parameters[name] = p
            layer, suffix = name.split('.mlp.', 1)
            category = category_of(suffix)
            layers[layer + '.mlp'].append(p)
            categories[category].append(p)
        assert {k: len(v) for k, v in categories.items()} == {'FC': 16, 'BN_affine': 32, 'PSN': 32}
        gradients, traces, repeats = [], [], defaultdict(list)
        for step in r['steps']:
            g = step['gradient_audit']
            assert g['all_allowed_gradients_present_finite'] and g['frozen_gradients_None']
            reconstructed = sum(g['category_squared_L2'].values()) ** .5
            actual = step['preclip_grad_norm']
            assert np.isclose(reconstructed, actual, rtol=2e-5, atol=1e-8)
            gradients.append(actual)
            row = {k: step[k] for k in ('step', 'sequence', 'target_id', 'task_loss',
                                        'valid_GT_pixels', 'preclip_grad_norm')}
            traces.append(row)
            repeats[step['target_id']].append(row)
        repeat_pairs = []
        if r['condition'] == 'reference':
            assert len(repeats) == 40 and {len(x) for x in repeats.values()} == {5}
            for target, rows in sorted(repeats.items()):
                repeat_pairs.append({'target_id': target, 'first_online_loss': rows[0]['task_loss'],
                                     'fifth_online_loss': rows[-1]['task_loss'],
                                     'fifth_minus_first': rows[-1]['task_loss'] - rows[0]['task_loss']})
        else:
            assert len(repeats) == 200 and {len(x) for x in repeats.values()} == {1}
        summaries.append({'condition': r['condition'], 'arm': r['arm'],
                          'result_sha256': sha(path), 'weights_sha256': r['weights_sha256'],
                          'parameter_delta': parameters,
                          'layer_delta': {k: combine(v) for k, v in layers.items()},
                          'category_delta': {k: combine(v) for k, v in categories.items()},
                          'total_delta': combine(list(parameters.values())),
                          'preclip_gradient_norm': stats(gradients),
                          'preclip_norm_above100_steps': sum(x > 100 for x in gradients),
                          'online_loss_distribution': stats([s['task_loss'] for s in r['steps']]),
                          'reference_repeated_member_loss_changes': repeat_pairs,
                          'online_trace': traces, 'gradient_coverage': r['gradient_coverage'],
                          'training_seconds': r['training_seconds'], 'wall_seconds': r['wall_seconds'],
                          'max_allocated_bytes': r['max_allocated_bytes'],
                          'full_entry_peak_bytes': r['full_entry_peak_bytes']})
        del payload, state
    output = {'schema': 'flow_r13_optimization_diagnostic_v1', 'status': 'COMPLETE_CPU_ONLY',
              'script_sha256': sha(__file__), 'original_checkpoint_sha256': ORIGINAL_SHA,
              'scope': 'All eight completed arms. No quality, inference, checkpoint or recipe selection.',
              'limitations': ['Online loss uses changing weights and samples; not fixed-state endpoint training AEE.',
                             'Reference same-ID visit1/5 is an online descriptive comparison, not an independent validation.',
                             'Parameter displacement and raw gradient norms do not establish a causal learning-rate mechanism.',
                             'Relative L2 is null for initially zero-norm tensors; inspect absolute displacement.',
                             'Independent execution audit remains the authority for complete state/BN/optimizer integrity.'],
              'runs': summaries}
    assert not torch.cuda.is_initialized()
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'sha256': sha(args.output), 'runs': len(summaries)}))


if __name__ == '__main__':
    main()
