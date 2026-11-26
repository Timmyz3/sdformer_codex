#!/usr/bin/env python3
"""Small CPU checks of bridge math and real fixed-mask coordinate interface."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load('_independent_final_bridge', HERE/'replay_r30_export.py')
core = load('_frozen_mixed_core', ROOT/'implementation/mixed_integer_core.py')
sys.path.insert(0, str(ROOT/'implementation'))
masks = load('_frozen_masks', ROOT/'implementation/mask_contract.py')
contract_path = ROOT/'hardware/MASK_CONTRACT_V1.json'
contract = audit.read(contract_path)
selected, _ = masks.materialize(contract, 'full9')
checks = []
unique = set()
for name, kind, shape, h, w in audit.TARGETS:
    group = name.rsplit('.', 1)[0] if kind == 'linear' else name
    locs = audit.expected_locations(contract, name, h, w)
    for category, y, x, mode in locs:
        require_high = y >= h//9*9 or x >= w//9*9 or bool(selected[group][y, x])
        assert require_high == (mode == 'Q8')
        assert 0 <= y < h and 0 <= x < w
        for t in (0, 9):
            for start in (0, shape[0]-16):
                identity = (name, t, y, x, start)
                assert identity not in unique and start+16 <= shape[0]
                unique.add(identity)
assert len(unique) == 32
checks.append('real frozen mask produces 32 unique valid class/time/output slices')
rng = np.random.default_rng(20261020)
for shape in ((16, 384), (16, 96, 3, 3)):
    w = rng.normal(size=shape).astype(np.float32)
    w[0] = 0
    w[1] = np.float32(-1)
    q, scale = audit.quantize_numpy(w)
    tq, ts = core.quantize(torch.from_numpy(w))
    assert np.array_equal(q, tq.numpy()) and np.array_equal(scale, ts.numpy())
checks.append('independent NumPy q8/scale equals frozen producer, including zero channels')
q = np.arange(-127, 128, dtype=np.int16)
twos = ((q.astype(np.uint8) >> 4) & 7).astype(np.int64)*16 - (q < 0).astype(np.int64)*128
assert np.array_equal((q.astype(np.int64)//16)*16, twos)
checks.append('all 255 q8 values: independent signed floor equals retained top four planes')
assert not torch.cuda.is_initialized()
destination = Path(sys.argv[1])
assert not destination.exists()
destination.write_text(json.dumps(dict(
    schema='flow_r30_final_bridge_CPU_preparation_v1', status='PASS', checks=checks,
    full_model_forwards=0, integer_affine_forwards=0, GPU_used=False, GT_values_read=False,
    sources={str(p.relative_to(ROOT)): audit.sha(p) for p in (
        Path(__file__), HERE/'replay_r30_export.py', contract_path,
        ROOT/'implementation/mixed_integer_core.py', ROOT/'implementation/mask_contract.py')}
), indent=2)+'\n')
print(json.dumps({'status': 'PASS', 'checks': len(checks)}))
