#!/usr/bin/env python3
"""Bind completed local R13 artifacts for the frozen analyzer; no metric access."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def binding(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def main():
    quality_freeze = json.loads((HERE / 'QUALITY_FREEZE_V1.json').read_text())
    for relative, expected in quality_freeze['source_sha256'].items():
        assert binding(HERE / relative)['sha256'] == expected, relative
    common = HERE.parent.parent / 'flow_story_20261011/preflight/data_protocol/artifacts'
    inputs = {'schema': 'flow_round13_analysis_inputs_v1'}
    for key, path in {
        'membership': HERE / 'data_protocol/manifest_v1.json',
        'runtime_manifest': HERE / 'runtime_v1.json',
        'bn_receipt': HERE / 'preparation/bn_receipt.json',
        'bn_buffers': HERE / 'preparation/bn_buffers.npz',
        'regions': HERE / 'preparation/regions_v1.json',
        'canonical': common / 'common_targets.json',
        'roles': common / 'train_auxiliary_roles.csv',
    }.items():
        inputs[key] = binding(path)
    results = []
    for condition in ('reference', 'diversity'):
        for arm in ('native', 'FF', 'CC', 'fine_s2b2'):
            path = HERE / 'raw/experiments_v1' / condition / 'recover' / arm / 'results.json'
            result = json.loads(path.read_text())
            assert result['status'] == 'ROUND13_COVERAGE_RECOVERY_COMPLETE', (condition, arm, result['status'])
            assert result['completed_optimizer_steps'] == 200
            item = binding(path)
            item.update(condition=condition, arm=arm)
            results.append(item)
    inputs['results'] = results
    output = HERE / 'analysis_inputs_v1.json'
    with output.open('x') as stream:
        json.dump(inputs, stream, indent=2)
        stream.write('\n')
    print(json.dumps(binding(output)))


if __name__ == '__main__':
    main()
