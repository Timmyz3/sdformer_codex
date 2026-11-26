"""CPU-only orchestration checks; these do not qualify primitive arithmetic."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('r16_pilot_tested', HERE/'run_spatial_pilot.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def main():
    rows = [{'target_id': 'fixed%d' % i} for i in range(8)]
    full = p.plan(rows, 'pilot123')
    seven = p.plan(rows, 'diagnostic7')
    assert len(full) == 123 and seven == full[:7]
    assert len([r for r in full if r['execution'] != 'expanded']) == 120
    assert len([r for r in full if r['execution'] == 'expanded']) == 3
    for row in rows:
        actual = [r for r in full if r['target_id'] == row['target_id'] and r['execution'] != 'expanded']
        assert [r['arm'] for r in actual] == list(p.ARMS)
    # Independent real-cell count, including eight partial tiles.
    all_native = sum(depth*10*h*w for depth, h, w in ((2, 120, 160), (2, 60, 80), (6, 30, 40), (2, 15, 20)))
    assert all_native == 558000
    row_checks = []
    for arm in p.ARMS[1:]:
        mask = p.mask_for(arm, torch)
        cells, area, tails, fine = 0, 0, 0, 0
        for y in range(0, 30, 9):
            for x in range(0, 40, 9):
                h, w = min(9, 30-y), min(9, 40-x)
                area += h*w
                tails += h != 9 or w != 9
                if bool(mask[y//9, x//9]):
                    cells += h*w
                    fine += 1
                else:
                    cells += ((h+2)//3)*((w+2)//3)
        independent = all_native-12000+10*cells
        assert tails == 8 and area == 1200
        assert fine == (20 if arm == 'FF' else 0 if arm == 'CC' else 1)
        assert p.expected_rows({'arm': arm, 'execution': 'compact'}) == independent
        row_checks.append({'arm': arm, 'representatives': cells, 'whole_network_fc_rows_each': independent})
    # Near-zero output error cannot exempt a single spike mismatch.
    base = {'u': torch.zeros(2), 'branch': torch.zeros(2), 'sn1': torch.zeros(2), 'sn2': torch.zeros(2)}
    actual = ([torch.zeros(2) for _ in range(4)], base, {})
    same = ([torch.zeros(2) for _ in range(4)], {k: v.clone() for k, v in base.items()}, {})
    assert p.numerical_pair(actual, same, torch)['pass']
    same[1]['sn2'][0] = 1
    assert not p.numerical_pair(actual, same, torch)['pass']
    same[1]['sn2'][0] = 0
    same[0][3][0] = 1e-3
    assert not p.numerical_pair(actual, same, torch)['pass']
    same[0][3][0] = 0
    same[1]['u'][0] = 1e-9
    assert not p.numerical_pair(actual, same, torch)['pass']
    try:
        p.compare(torch.tensor([float('nan')]), torch.zeros(1), torch)
    except ValueError:
        pass
    else:
        raise AssertionError('NaN must be rejected')
    assert not torch.cuda.is_initialized()
    out = {'schema': 'flow_r16_pilot_orchestration_cpu_v1', 'status': 'PASS',
           'source_sha256': {n: hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('run_spatial_pilot.py', 'test_pilot_cpu.py')},
           'fixed_plan_counts': {'pilot': 123, 'actual': 120, 'expanded': 3, 'diagnostic': 7},
           'row_checks': row_checks, 'negative_numerical_checks': ['one_spike', 'head_outside_tolerance', 'prefix_not_exact', 'nonfinite'],
           'CUDA_initialized': False, 'full_model_forwards': 0,
           'scope': 'Plan/geometry/comparison orchestration only; primitive/source/real-model closure are separate requirements.'}
    output = Path(sys.argv[1])
    assert not output.exists()
    output.write_text(json.dumps(out, indent=2)+'\n')
    print('PASS: fixed plans, true tails/rows, strict numerical rejection; no model or CUDA')


if __name__ == '__main__':
    main()
