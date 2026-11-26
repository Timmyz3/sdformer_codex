#!/usr/bin/env python3
"""Compare every LR primary cell and gate to independent new/previous raw reductions."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LR = HERE.parent.parent / 'native_lr_stability'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text())
math_path = HERE / 'reduce_raw_quality.py'
assert sha(math_path) == '26a2e5c97b988f10685e1c440a11506354d9591cb9b36a567292abb6d0348088'
spec = importlib.util.spec_from_file_location('independent_math', math_path)
math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math)
old_path = HERE / 'actual_reduction_v1/independent_reduction.json'
assert sha(old_path) == 'aaf5fae835475903b01f7ca4656884b1c1d3ba51de0c8fc03d72ddabd8f98a14'
old = read(old_path)['conditions']['diversity']
new_path = HERE / 'native_lr_actual_v1/independent_reduction.json'
new = read(new_path)
primary_path = LR / 'primary_analysis_v1.json'
primary = read(primary_path)
assert new['status'] == 'PASS' and primary['schema'] == 'flow_native_lr_primary_analysis_v1'
cells, summaries = {}, {}
for label, source, key in (
        ('initial/native', new, 'initial/native'),
        ('initial/FF', old, 'initial/FF'),
        ('lr_2.5e-4/native', old, 'native/native'),
        ('lr_2.5e-4/FF', old, 'native/FF'),
        ('lr_2.5e-5/native', new, 'endpoint/native'),
        ('lr_2.5e-5/FF', new, 'endpoint/FF')):
    cells[label] = source['all_endpoints'][key]
    summaries[label] = source['summaries'][key]
assert set(cells) == set(primary['all_endpoints']) == set(primary['summaries'])
numeric = math.NumericAudit()
exact_count = 0


def compare(actual, report, location):
    global exact_count
    if isinstance(actual, dict):
        assert set(actual) == set(report), location
        for key in actual:
            compare(actual[key], report[key], location + '/' + key)
    elif isinstance(actual, list):
        assert len(actual) == len(report), location
        for idx, (x, y) in enumerate(zip(actual, report)):
            compare(x, y, location + '/' + str(idx))
    elif isinstance(actual, bool) or isinstance(actual, str):
        exact_count += 1
        assert actual == report, location
    else:
        numeric.check(actual, report, location)


compare(cells, primary['all_endpoints'], 'all_endpoints')
for key, scopes in summaries.items():
    for scope, regions in scopes.items():
        for reg, values in regions.items():
            actual = {k: v for k, v in values.items() if k != 'empty_region_frames'}
            compare(actual, primary['summaries'][key][scope][reg], '/'.join((key, scope, reg)))
            assert values['empty_region_frames'] == (8 if scope == 'overall' else 4) - actual['frames_with_region']
gates = {}
for mode in ('native', 'FF'):
    gates['new_vs_initial/' + mode] = math.compare(summaries['lr_2.5e-5/' + mode], summaries['initial/' + mode])
    gates['old_vs_initial/' + mode] = math.compare(summaries['lr_2.5e-4/' + mode], summaries['initial/' + mode])
    gates['new_vs_old/' + mode] = math.compare(summaries['lr_2.5e-5/' + mode], summaries['lr_2.5e-4/' + mode])
compare(gates, primary['comparisons'], 'comparisons')
assert primary['native_stability_gate'] == gates['new_vs_initial/native']['pass']
assert primary['same_native_weights_FF_preservation_gate'] == gates['new_vs_initial/FF']['pass']
result = {
    'schema': 'flow_native_lr_independent_primary_agreement_v1', 'status': 'PASS',
    'script_sha256': sha(__file__), 'primary_sha256': sha(primary_path),
    'new_independent_reduction_sha256': sha(new_path), 'R13_independent_reduction_sha256': sha(old_path),
    'state_layouts': 6, 'frame_region_cells': 768, 'scope_region_cells': 288,
    'comparisons': 6, 'individual_gate_cells': 54,
    'numeric_comparisons': numeric.count, 'exact_string_boolean_comparisons': exact_count,
    'maximum_absolute_difference': numeric.maximum_absolute,
    'maximum_relative_difference': numeric.maximum_relative,
    'atol': math.ATOL, 'rtol': math.RTOL,
    'native_stability_gate': gates['new_vs_initial/native']['pass'],
    'same_native_weights_FF_preservation_gate': gates['new_vs_initial/FF']['pass'],
    'scope': 'All six primary state-layouts; three freshly reproduced raw states and three already independently audited R13 states. Missing primary empty-region counts derived from fixed 8/4 frames; all five primary summary fields compared directly. One native-trained arm only.',
}
output = HERE / 'native_lr_actual_v1/primary_agreement_v1.json'
with output.open('x') as f:
    json.dump(result, f, indent=2, sort_keys=True)
    f.write('\n')
print(json.dumps(result, indent=2))
