#!/usr/bin/env python3
"""Compare real native pilot restart with an independently trained control."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(directory, torch):
    pointer = json.loads((directory / 'latest.json').read_text())
    filename = pointer['checkpoint_file']
    if Path(filename).name != filename:
        raise ValueError('Unsafe checkpoint pointer')
    path = directory / filename
    side = json.loads(path.with_suffix('.json').read_text())
    if side['schema_version'] != 'flow_a_native_encoding_v2_checkpoint_v1':
        raise ValueError('Not a new-native checkpoint')
    if sha(path) != side['checkpoint_sha256']:
        raise ValueError('Checkpoint hash differs')
    return torch.load(path, map_location='cpu', weights_only=True), side


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resumed', type=Path, required=True)
    p.add_argument('--direct', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import torch
    torch.set_num_threads(1)
    left, lside = load(a.resumed, torch)
    right, rside = load(a.direct, torch)
    differences = []
    counts = {'tensors_compared': 0, 'scalar_leaves_compared': 0}

    def compare(x, y, path):
        if type(x) is not type(y):
            differences.append(path + ': type'); return
        if isinstance(x, torch.Tensor):
            counts['tensors_compared'] += 1
            if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y):
                differences.append(path + ': tensor')
        elif isinstance(x, dict):
            if x.keys() != y.keys():
                differences.append(path + ': keys'); return
            for k in x: compare(x[k], y[k], path + '/' + str(k))
        elif isinstance(x, (list, tuple)):
            if len(x) != len(y):
                differences.append(path + ': length'); return
            for i, (xx, yy) in enumerate(zip(x, y)): compare(xx, yy, path + '/' + str(i))
        else:
            counts['scalar_leaves_compared'] += 1
            if x != y: differences.append(path + ': value')

    for name in ('schema_version', 'contract', 'contract_sha256', 'model_state_dict',
                 'optimizer_state_dict', 'scheduler_state_dict', 'rng_state', 'epoch_order'):
        compare(left[name], right[name], name)
    for name in ('completed_optimizer_steps', 'epoch', 'next_index', 'zero_valid_crops'):
        compare(left['training'][name], right['training'][name], 'training/' + name)
    expected_steps = left['training']['completed_optimizer_steps'] == right['training']['completed_optimizer_steps'] == 36
    result = {'passed': not differences and expected_steps, 'expected_steps': 36,
              'model_tensor_count': len(left['model_state_dict']), **counts,
              'differences': differences, 'resumed_checkpoint': lside, 'direct_checkpoint': rside,
              'excluded_fields': ['training_seconds', 'stop_reason', 'checkpoint file bytes/timestamps'],
              'scope': 'same source/data/recipe/runtime; real 32+4 versus uninterrupted36; not AEE evidence',
              'cuda_initialized': torch.cuda.is_initialized()}
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('passed', 'expected_steps', 'model_tensor_count', 'tensors_compared', 'differences')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
