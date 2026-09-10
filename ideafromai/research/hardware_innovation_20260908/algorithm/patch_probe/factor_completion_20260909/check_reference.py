"""Run real-weight and directed TC/TR checks; no torch, training or GPU."""
from pathlib import Path
import json
import time

import numpy as np

from factor_reference import (STRUCTURES, initialize, hard_masks, spatial_regions,
    sparse_forward, latent_demands, store_v_column_banks, read_tr, storage_summary)

HERE = Path(__file__).resolve().parent
PARTIAL = HERE.parent/'partial_completion'


def compare(x, u, v, masks, latent_tile, output_tile):
    sparse, counts = sparse_forward(x, u, v, masks, latent_tile, output_tile)
    full_mask = np.repeat(masks, latent_tile, axis=1)
    dense = ((x @ u)*full_mask[:, None, None, :]) @ v
    error = float(np.max(np.abs(sparse-dense)))
    assert np.allclose(sparse, dense, atol=1e-10, rtol=1e-10), error
    return dict(max_abs_error=error, values=int(sparse.size), counts=counts)


def main():
    started = time.monotonic()
    rng = np.random.default_rng(909)
    # Signed factors, non-unit theta, different spatial masks, and zero mask.
    x = (rng.random((3, 3, 4, 5)) < .3).astype(np.float64)*2.5
    u = rng.integers(-3, 4, (5, 8)).astype(np.float64)
    v = rng.integers(-3, 4, (8, 6)).astype(np.float64)
    masks = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 0]], bool)
    directed = compare(x, u, v, masks, 2, 3)
    memory = store_v_column_banks(v, 7)
    restored = read_tr(memory, np.arange(8), np.arange(6), 8)
    assert np.array_equal(restored, v)
    # Two output groups: private, shared, private latent connections. Once
    # H0 is done, only the H0-private latent may disappear.
    unresolved = np.zeros((1, 2, 4, 2), bool)
    unresolved[:, :, 2:, :] = True
    e = np.array([[1, 0], [1, 1]], bool)
    connection = np.array([[1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 1, 1]], bool)
    demand = latent_demands(unresolved, e, connection, np.ones((1, 3), bool), 1)
    assert not demand[..., 0].any() and demand[..., 1:].all()
    with np.load(PARTIAL/'shared_column_deployment_source.npz') as operator:
        weight = operator['weight'].astype(np.float64).reshape(96, 864).T
        theta = float(operator['source_theta'])
    # Two real native P4 groups from an existing validation capture. This is
    # numerical verification only; no factor/mask is selected from validation.
    with np.load(PARTIAL/'integer_valid10/capture_00.npz') as capture:
        words = capture['source_gate_words'][:2]
        gids = capture['group_ids'][:2]
    x = (((words[..., None] >> np.arange(10)) & 1).transpose(0, 3, 2, 1)).astype(np.float64)*theta
    axes = {}
    for structure in STRUCTURES:
        params = initialize(weight, structure)
        all_masks = hard_masks(params['logits'], params['active_tiles'])
        masks = all_masks[spatial_regions(gids)]
        value = compare(x, params['u'], params['v'], masks, 4, 8)
        difference = params['u'] @ params['v']-weight
        if structure in ('shared', 'grouped'):
            assert np.max(np.abs(difference)) < 1e-10
        value.update(storage=storage_summary(params),
                     unmasked_weight_mse=float(np.mean(difference**2)),
                     masks=all_masks.astype(int).tolist())
        axes[structure] = value
    result = dict(complete=True,
        scope='NumPy Float64 directed case plus two actual native P4 source groups and original FP32 W; no training or accuracy estimate',
        directed=directed, bank_TC_to_TR_exact=True, dependency_private_shared_example=True,
        axes=axes, elapsed_seconds=time.monotonic()-started,
        not_measured='no cycles/ports/peak state, no network AEE, no official recipe reproduction')
    (HERE/'reference_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(complete=True, directed_max_abs_error=directed['max_abs_error'],
        actual_axes={k:v['max_abs_error'] for k,v in axes.items()},
        wall_seconds=result['elapsed_seconds']), indent=2), flush=True)


if __name__ == '__main__':
    main()
