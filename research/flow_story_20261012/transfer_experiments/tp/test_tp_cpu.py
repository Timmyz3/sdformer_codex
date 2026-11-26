#!/usr/bin/env python3
"""Synthetic CPU mechanism checks only; no data, checkpoint, training or GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import sys
sys.dont_write_bytecode = True
import ast
import copy
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
import torch.nn.functional as F
from tp_score import official_sdt_scores, select_spatial, TPTilePolicy, mask_dense_branch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
SOURCE = PROJECT/'SDformer/third_party/SDformerFlow'
PREFLIGHT = PROJECT/'research/flow_story_20261011/preflight/cpu_numerics'
DEPLOYMENT = HERE.parents[1]/'deployment_contract'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_numpy(x):
    """Independent scalar neighborhood/cosine, FP64; no torch scorer operators."""
    a = x.numpy().astype(np.float64)
    t, b, c, h, w = a.shape
    temporal_raw = np.abs(np.diff(a.mean(axis=2), axis=0)).mean(axis=0)
    mu = a.mean(axis=0)
    spatial_raw = np.empty((b, h, w), dtype=np.float64)
    for i in range(b):
        for y in range(h):
            for z in range(w):
                local_sum = 0.0
                for yy in range(max(0, y-1), min(h, y+2)):
                    for zz in range(max(0, z-1), min(w, z+2)):
                        for ch in range(c):
                            local_sum += mu[i, ch, yy, zz]/9.0
                v = mu[i, :, y, z]
                v2 = np.full(c, local_sum)
                cosine = np.dot(v, v2)/(max(float(np.linalg.norm(v)), 1e-6)
                                       * max(float(np.linalg.norm(v2)), 1e-6))
                spatial_raw[i, y, z] = -cosine
    def softmax(a):
        q = a.reshape(b, -1)
        e = np.exp(q-q.max(axis=1, keepdims=True))
        return (e/e.sum(axis=1, keepdims=True)).reshape(b, h, w)
    return {'temporal': softmax(temporal_raw), 'spatial': softmax(spatial_raw)}


def upstream_classes():
    manifest = json.loads((HERE/'source_manifest.json').read_text())
    file = HERE/manifest['local_copy']
    expected = 'e4a688393b34e9bf2c15a4ee783cabe8427ff1cfab22429bad95a1cc8e91049d'
    assert manifest['source_sha256'] == sha(file) == expected
    parsed = ast.parse(file.read_text())
    names = {'TemporalChangeScorer', 'LocalSpatialSimilarity'}
    nodes = [node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name in names]
    assert {node.name for node in nodes} == names
    # Only the two reviewed scorer definitions, no upstream imports/model code.
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    space = {'torch': torch, 'nn': torch.nn, 'F': F}
    exec(compile(module, str(file)+'#two_reviewed_scorers_only', 'exec'), space)
    return space['TemporalChangeScorer'](), space['LocalSpatialSimilarity']()


def raises(fn, error=ValueError):
    try:
        fn()
    except error:
        return
    raise AssertionError('Expected ' + error.__name__)


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(1012)
    started = time.perf_counter()
    checks = []
    def check(name, fn):
        extra = fn()
        checks.append(dict(name=name, passed=True, **(extra or {})))

    ut, us = upstream_classes()
    def score_case(shape, zeros=False):
        x = torch.zeros(shape) if zeros else torch.randn(shape)
        out = official_sdt_scores(x)
        ref = reference_numpy(x)
        official = {'temporal': ut(x), 'spatial': us(x)}
        errors = {}
        for key in ('temporal', 'spatial'):
            assert torch.equal(out[key], official[key]), (shape, key)
            err = float(np.abs(out[key].numpy()-ref[key]).max())
            assert err < 2e-6, (shape, key, err)
            errors[key] = err
        assert torch.allclose(out['total'].sum((1, 2)), torch.full((shape[1],), 2.0))
        return {'shape_TBCHW': list(shape), 'official_equal': True, 'numpy_max_error': errors}
    for shape in ((10, 1, 1, 1, 1), (10, 2, 3, 2, 4), (10, 1, 3, 10, 11)):
        check('score_reference_'+str(shape), lambda shape=shape: score_case(shape))
    check('zero_input_softmax_finite', lambda: score_case((10, 1, 2, 3, 4), True))

    def cancellation():
        x = torch.zeros(10, 1, 2, 2, 3)
        x[:, 0, 0, 0, 0] = torch.arange(10)
        x[:, 0, 1, 0, 0] = -torch.arange(10)
        score = official_sdt_scores(x)
        assert torch.equal(score['temporal'], torch.full((1, 2, 3), 1/6))
        assert x.diff(dim=0).abs().mean((0, 2)).max() > 0
    check('channel_cancellation_preserved_not_abs_before_channel_mean', cancellation)

    def tail_case():
        scores = torch.ones(1, 10, 11)
        scores[:, 9:, 9:] = 20
        selection = select_spatial(scores, 'tile9', keep_count=1)
        assert selection.valid_pixels_per_unit.tolist() == [[81, 18], [9, 2]]
        assert selection.unit_scores.tolist() == [[[1, 1], [1, 20]]]
        assert selection.unit_keep.tolist() == [[[False, False], [False, True]]]
        assert selection.spatial_keep.sum().item() == 2
        full = selection.complete_t_mask()
        assert tuple(full.shape) == (10, 1, 10, 11, 1)
        assert all(torch.equal(full[0], full[t]) for t in range(10))
        return selection.json_record()
    check('tile_tail_mean_and_complete_T_support', tail_case)

    def select_case(mode, count):
        scores = torch.arange(110, dtype=torch.float32).reshape(1, 10, 11)
        selection = select_spatial(scores, mode, keep_count=count)
        assert selection.unit_keep.sum().item() == count
        if count == 0: assert not selection.spatial_keep.any()
        if count == selection.metadata['candidate_units_per_image']: assert selection.spatial_keep.all()
        if mode == 'token' and count:
            assert selection.spatial_keep.flatten()[-count:].all()
        return {'mode': mode, 'count': count, 'real_pixels': selection.metadata['kept_real_spatial_pixels']}
    for mode, counts in (('token', [0, 13, 110]), ('tile9', [0, 2, 4])):
        for n in counts:
            check(mode+'_keep_'+str(n), lambda mode=mode, n=n: select_case(mode, n))

    def ties():
        score = torch.ones(2, 2, 3)
        selected = select_spatial(score, 'token', keep_count=2, tie_policy='stable_index')
        assert selected.spatial_keep.reshape(2, -1).tolist() == [[True, True, False, False, False, False]]*2
        faithful = select_spatial(score, 'token', keep_count=2)
        expected = torch.zeros(2, 6, dtype=torch.bool).scatter_(1, torch.topk(score.reshape(2, 6), 2, dim=1).indices, True)
        assert torch.equal(faithful.spatial_keep.reshape(2, -1), expected)
    check('explicit_tie_policies_and_batched_selection', ties)

    def token_diagnostic():
        selection = select_spatial(torch.arange(6.0).reshape(1, 2, 3), 'token', keep_count=2)
        calls = []
        branch = torch.ones(10, 1, 2, 3, 4)
        calls.append('full_MLP_already_executed')
        out = mask_dense_branch(branch, selection)
        assert out.count_nonzero().item() == 10*2*4 and len(calls) == 1
        assert selection.metadata['scope'] == 'token diagnostic only'
    check('token_mask_is_postproducer_diagnostic', token_diagnostic)

    invalid = [
        ('T1', lambda: official_sdt_scores(torch.ones(1, 1, 1, 2, 2))),
        ('dtype', lambda: official_sdt_scores(torch.ones(10, 1, 1, 2, 2).double())),
        ('nan', lambda: official_sdt_scores(torch.full((10, 1, 1, 2, 2), float('nan')))),
        ('overflow', lambda: official_sdt_scores(torch.full((10, 1, 2, 2, 2), 3e38))),
        ('negative_keep', lambda: select_spatial(torch.ones(1, 2, 2), 'tile9', keep_count=-1)),
        ('too_many', lambda: select_spatial(torch.ones(1, 2, 2), 'tile9', keep_count=2)),
        ('ambiguous_budget', lambda: select_spatial(torch.ones(1, 2, 2), 'token', keep_count=1, keep_fraction=.5)),
        ('nan_budget', lambda: select_spatial(torch.ones(1, 2, 2), 'token', keep_fraction=float('nan'))),
        ('nonfinite_score', lambda: select_spatial(torch.full((1, 2, 2), float('inf')), 'token', keep_count=0)),
        ('policy_T2', lambda: TPTilePolicy(keep_count=0)('block', torch.ones(2, 1, 2, 2, 4))),
        ('policy_B2', lambda: TPTilePolicy(keep_count=0)('block', torch.ones(10, 2, 2, 2, 4))),
    ]
    for name, fn in invalid:
        with torch.inference_mode(): check('reject_'+name, lambda fn=fn: raises(fn))
    check('reject_grad_enabled_policy', lambda: raises(lambda: TPTilePolicy(keep_count=0)(
        'block', torch.ones(10, 1, 2, 2, 4))))

    sys.path[:0] = [str(PREFLIGHT/'deps'), str(SOURCE), str(DEPLOYMENT), str(PREFLIGHT)]
    from native_mlp_probe import make_mlp
    from tile_mlp import TileMLP
    model = make_mlp(4, torch.float32)
    u0 = torch.randn(10, 1, 10, 11, 4)
    def integration(count):
        outputs, policies, wrappers = {}, {}, {}
        producer_calls = {'dense_masked': 0, 'actual_skip': 0}
        with torch.inference_mode():
            for mode in ('dense_masked', 'actual_skip'):
                # Synthetic producer stands for already completed attention+residual.
                producer_calls[mode] += 1
                u = u0*1.05 + 0.1
                policy = TPTilePolicy(keep_count=count)
                wrapper = TileMLP(copy.deepcopy(model), 'block.mlp', mode, policy)
                branch = wrapper(u)
                outputs[mode] = (u+branch, branch)
                policies[mode], wrappers[mode] = policy, wrapper
            assert torch.equal(policies['dense_masked'].last_selection.spatial_keep,
                               policies['actual_skip'].last_selection.spatial_keep)
            for a, b in zip(outputs['dense_masked'], outputs['actual_skip']):
                assert torch.allclose(a, b, atol=2e-5, rtol=2e-5)
            if count == 0:
                assert outputs['actual_skip'][1].count_nonzero() == 0
                assert torch.equal(outputs['actual_skip'][0], u0*1.05+0.1)
            else:
                assert outputs['actual_skip'][1].abs().max() > .01
            record = wrappers['actual_skip'].records[0]
            assert record['observed']['fc1_rows'] == 10*record['kept_real_pixels']
            assert record['observed']['fc2_rows'] == record['observed']['fc1_rows']
            assert producer_calls == {'dense_masked': 1, 'actual_skip': 1}
            assert policies['actual_skip'].records[0]['score_cost']['official_dense_conv_mac_including_padded_taps'] > 0
            return {'count': count, 'producer_calls': producer_calls,
                'branch_max_error': float((outputs['dense_masked'][1]-outputs['actual_skip'][1]).abs().max()),
                'execution': record, 'policy': policies['actual_skip'].records[0]}
    for count in (0, 1, 3, 4):
        check('native_TileMLP_integration_keep_'+str(count), lambda count=count: integration(count))

    def bn_guard():
        wrong = copy.deepcopy(model)
        wrong.bn1.norm_layer.running_mean = None
        wrong.bn1.norm_layer.running_var = None
        wrong.bn1.norm_layer.track_running_stats = False
        raises(lambda: TileMLP(wrong, 'block.mlp', 'actual_skip', TPTilePolicy(keep_count=0)))
    check('current_stat_BN_rejected_by_existing_executor', bn_guard)
    check('fraction_floor_explicit', lambda: select_case('tile9', select_spatial(
        torch.ones(1, 10, 11), 'tile9', keep_fraction=.74).metadata['keep_count_per_image']))
    assert not torch.cuda.is_initialized()
    result = {'schema': 'flow_tp_cpu_checks_v1', 'status': 'PASS', 'device': 'cpu',
        'cuda_initialized': torch.cuda.is_initialized(), 'python': platform.python_version(),
        'torch': str(torch.__version__), 'numpy': str(np.__version__), 'threads': 2,
        'checkpoint_used': False, 'dataset_used': False, 'training_performed': False,
        'scope': 'synthetic mechanism/reference/native MLP integration; not AEE, GPU throughput or PE cycles',
        'check_count': len(checks), 'checks': checks, 'elapsed_seconds': time.perf_counter()-started,
        'files_sha256': {str(p.relative_to(HERE)): sha(p) for p in
            (HERE/'tp_score.py', HERE/'test_tp_cpu.py', HERE/'source_manifest.json', HERE/'upstream/sdt_drop.py')},
        'external_source_sha256': {str(p): sha(p) for p in (DEPLOYMENT/'tile_mlp.py',
            PREFLIGHT/'native_mlp_probe.py', SOURCE/'models/STSwinNet_SNN/Spiking_swin_transformer3D.py',
            SOURCE/'models/STSwinNet_SNN/Spiking_submodules.py')}}
    (HERE/'cpu_checks.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': 'PASS', 'checks': len(checks), 'cuda_initialized': False,
                      'output': str(HERE/'cpu_checks.json')}))


if __name__ == '__main__':
    main()
