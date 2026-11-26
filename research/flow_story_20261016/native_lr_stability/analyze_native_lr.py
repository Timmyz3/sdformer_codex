#!/usr/bin/env python3
"""Prespecified native LR comparison with all regions and two deployment layouts."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R13 = HERE.parent / 'coverage_recovery_round13'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    math_path = HERE.parent / 'ordinary_recovery_round12/analyze_recovery_v2.py'
    assert sha(math_path) == '525ba2821d1e671b581f63e80c4bce02ef3e524c3e6af7d143a723914b001cd8'
    spec = importlib.util.spec_from_file_location('_lr_frozen_math', str(math_path))
    math = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math)
    paths = {'new': HERE / 'recover_v1/results.json',
             'old_native': R13 / 'raw/experiments_v1/diversity/recover/native/results.json',
             'old_FF_initial_source': R13 / 'raw/experiments_v1/diversity/recover/FF/results.json'}
    raw = {k: json.loads(p.read_text()) for k, p in paths.items()}
    new, old = raw['new'], raw['old_native']
    assert new['status'] == 'NATIVE_LR_RECOVERY_COMPLETE'
    assert new['completed_optimizer_steps'] == 200 and new['full_forward_count'] == 228
    assert new['paired_initial_state_exact'] and new['paired_initial_all_eight_native_predictions_exact']
    assert new['binding']['baseline_comparison']['result_sha256'] == sha(paths['old_native'])
    assert new['initial_model_state_sha256'] == old['initial_model_state_sha256'] == raw['old_FF_initial_source']['initial_model_state_sha256']
    assert {k for k in new['binding']['recipe'] if new['binding']['recipe'][k] != old['binding']['recipe'][k]} == {'lr'}
    assert new['actual_optimizer_hyperparameters']['lr'] == 2.5e-5
    ids = old['binding']['check_ids']
    assert len(ids) == 8 and new['binding']['check_ids'] == ids
    cells, support = {}, {}
    selected = [('initial/native', 'new', 'initial', 'native'),
                ('initial/FF', 'old_FF_initial_source', 'initial', 'FF'),
                ('lr_2.5e-4/native', 'old_native', 'endpoint', 'native'),
                ('lr_2.5e-4/FF', 'old_native', 'endpoint', 'FF'),
                ('lr_2.5e-5/native', 'new', 'endpoint', 'native'),
                ('lr_2.5e-5/FF', 'new', 'endpoint', 'FF')]
    for key, source, phase, layout in selected:
        frames = raw[source]['evaluations'][phase]['frames']
        assert [f['target_id'] for f in frames] == ids
        cells[key] = []
        for f in frames:
            archive = paths[source].parent / f['prediction_file']
            assert sha(archive) == f['prediction_file_sha256']
            with np.load(archive, allow_pickle=False) as values:
                prediction = values[layout]
                assert prediction.dtype == np.float32 and prediction.shape == (2, 480, 640)
                assert np.isfinite(prediction).all()
                assert hashlib.sha256(prediction.tobytes()).hexdigest() == f['arms'][layout]['prediction_sha256']
            regions = f['arms'][layout]['metrics']['regions']
            assert set(regions) == set(math.REGIONS)
            for region, value in regions.items():
                index = (f['target_id'], region)
                support.setdefault(index, value['valid_pixels'])
                assert value['valid_pixels'] == support[index]
            cells[key].append({'target_id': f['target_id'], 'sequence': f['sequence'], 'regions': regions})
    sequences = sorted({f['sequence'] for f in cells['initial/native']})
    assert len(sequences) == 2
    summaries = {}
    for key, frames in cells.items():
        scopes = {'overall': frames}
        scopes.update({s: [f for f in frames if f['sequence'] == s] for s in sequences})
        assert all(len(scopes[s]) == 4 for s in sequences)
        summaries[key] = {s: {region: math.aggregate([f['regions'][region] for f in frames])
                             for region in math.REGIONS} for s, frames in scopes.items()}
    comparisons = {}
    for layout in ('native', 'FF'):
        comparisons['new_vs_initial/' + layout] = math.compare(summaries['lr_2.5e-5/' + layout], summaries['initial/' + layout])
        comparisons['old_vs_initial/' + layout] = math.compare(summaries['lr_2.5e-4/' + layout], summaries['initial/' + layout])
        comparisons['new_vs_old/' + layout] = math.compare(summaries['lr_2.5e-5/' + layout], summaries['lr_2.5e-4/' + layout])
    output = {'schema': 'flow_native_lr_primary_analysis_v1',
              'sources': {k: {'path': str(p), 'sha256': sha(p)} for k, p in paths.items()},
              'script_sha256': sha(__file__), 'math_source_sha256': sha(math_path),
              'regions': list(math.REGIONS), 'all_endpoints': cells,
              'summaries': summaries, 'comparisons': comparisons,
              'native_stability_gate': comparisons['new_vs_initial/native']['pass'],
              'same_native_weights_FF_preservation_gate': comparisons['new_vs_initial/FF']['pass'],
              'scope': 'One native-trained LR arm, two deployment layouts. FF is not an independently FF-trained arm. '
                       'Reused development8, no official test, flow increment, four-arm qualification or hardware claim. '
                       'Primary aggregation uses retained metrics and verified prediction bytes; original-GT recomputation is independent.'}
    with (HERE / 'primary_analysis_v1.json').open('x') as f:
        json.dump(output, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: output[k] for k in ('native_stability_gate', 'same_native_weights_FF_preservation_gate')}))
    for key, value in summaries.items():
        print(key, json.dumps({region: value['overall'][region]['frame_mean_AEE'] for region in ('all_valid', 'B', 'L')}))


if __name__ == '__main__':
    main()
