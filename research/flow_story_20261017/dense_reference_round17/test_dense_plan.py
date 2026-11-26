"""No neural forwards: fixed plan, geometry, and numerical rejection checks."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import torch
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('dense_reference_tested', HERE/'run_dense_reference.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def main():
    rows = [{'target_id': str(i)} for i in range(8)]
    items = p.plan(rows, 'dense120')
    assert len(items) == 120 and sum(x['execution'] == 'native' for x in items) == 8
    assert sum(x['execution'] == 'expanded' for x in items) == 112
    assert sum(p.expected_rows(x) for x in items) == 66960000
    assert 78*len(items) == 9360
    for row in rows:
        assert [x['arm'] for x in items if x['target_id'] == row['target_id']] == list(p.ARMS)
    for arm in p.ARMS[1:]:
        mask = p.mask_for(arm, torch)
        assert mask.shape == (4, 5)
        assert int(mask.sum()) == (20 if arm == 'FF' else 0 if arm == 'CC' else 1)
        if arm.startswith('fine_'):
            assert not mask[3, :].any() and not mask[:, 4].any()
    parsed = ast.parse((HERE/'run_dense_reference.py').read_text())
    calls = [n.func.attr for n in ast.walk(parsed) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert 'expanded_mlp' in calls and 'compact_mlp' not in calls
    x = {'u': torch.zeros(2), 'branch': torch.zeros(2), 'sn1': torch.zeros(2), 'sn2': torch.zeros(2)}
    a = ([torch.zeros(2) for _ in range(4)], x, {})
    b = ([torch.zeros(2) for _ in range(4)], {k: v.clone() for k, v in x.items()}, {})
    assert p.numerical_pair(a, b, torch)['pass']
    b[1]['sn1'][0] = 1
    assert not p.numerical_pair(a, b, torch)['pass']
    b[1]['sn1'][0] = 0
    b[0][3][0] = 1e-3
    assert not p.numerical_pair(a, b, torch)['pass']
    try:
        p.compare(torch.tensor([float('nan')]), torch.zeros(1), torch)
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite must fail')
    assert not torch.cuda.is_initialized()
    out = {'schema': 'flow_r17_orchestration_cpu_v1', 'status': 'PASS',
           'source_sha256': {n: hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('run_dense_reference.py', 'test_dense_plan.py')},
           'full_forward_plan': 120, 'native': 8, 'expanded': 112, 'compact': 0,
           'expected_rows_per_FC': 66960000, 'expected_BN_calls': 9360,
           'checks': ['full_H8_plan', 'real_tails', 'dense_rows', 'no_compact_call_AST', 'spike_reject', 'head_reject', 'finite_reject'],
           'neural_forwards': 0, 'GPU_used': False, 'GT_read': False}
    dest = Path(sys.argv[1])
    assert not dest.exists()
    dest.write_text(json.dumps(out, indent=2)+'\n')
    print('PASS: fixed120 dense-only plan and numerical rejection checks; no model/GT/GPU')


if __name__ == '__main__':
    main()
