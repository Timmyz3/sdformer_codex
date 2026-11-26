#!/usr/bin/env python3
"""Four dense-only BN-scope diagnostics, with the frozen native evaluator loop."""
import json
from pathlib import Path
import types

import evaluate_breadth as base
import evaluate_breadth_extended as extended

EXTENDED_SHA = 'c49f2979c435242c641619cc4d8433e36dbb6c4e13ec5461ca8e8b78b3e2fdf0'
SCOPES = ('P_paper', 'MLP_original', 'MLP_CMA176', 'nonMLP_CMA176')
CASES = [{'id': 'dense_'+scope, 'method': 'dense', 'bn': 'P_paper', 'bn_scope': scope} for scope in SCOPES]


def bn_inventory(model, torch, expected_mlp_blocks=12, expected_total_bn=78):
    all_bn = {name: m for name, m in model.named_modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}
    blocks = {name: m for name, m in model.named_modules() if m.__class__.__name__ == 'MS_Spiking_Mlp'}
    base.require(len(blocks) == expected_mlp_blocks and len(all_bn) == expected_total_bn, 'Unexpected native BN/MLP count')
    selected = set()
    for name, block in blocks.items():
        base.require(block.norm_layer == 'BN', 'Only native BN MLP supported')
        found = {n for n in all_bn if n.startswith(name+'.')}
        legal = {n for n in found if n.startswith(name+'.bn1.') or n.startswith(name+'.bn2.')}
        base.require(found == legal and len(legal) == 2, 'MLP normalization closure is not exactly bn1/bn2')
        selected.update(legal)
    return all_bn, selected


def apply_scope(model, scope, artifact, torch, expected_mlp_blocks=12, expected_total_bn=78):
    base.require(scope in SCOPES, 'Unknown BN scope')
    all_bn, mlp_bn = bn_inventory(model, torch, expected_mlp_blocks, expected_total_bn)
    non_bn_before = extended.non_bn_digest(model, all_bn)
    model.eval()
    load_record = None
    if scope.endswith('CMA176'):
        base.require(artifact is not None, 'CMA artifact required')
        # Validate/install the complete dense artifact first. Current-stat domains below discard
        # those buffers explicitly; the retained subset alone contributes running statistics.
        load_record = extended.apply_bn_buffers(model, artifact, torch)
    frozen = set() if scope == 'P_paper' else set(all_bn)-mlp_bn if scope == 'nonMLP_CMA176' else mlp_bn
    for name, layer in all_bn.items():
        if name in frozen:
            base.require(layer.track_running_stats and layer.running_mean is not None and layer.running_var is not None
                         and layer.num_batches_tracked is not None and layer.num_batches_tracked.item() > 0
                         and torch.isfinite(layer.running_mean).all().item() and torch.isfinite(layer.running_var).all().item()
                         and not (layer.running_var < 0).any().item(), 'Frozen BN has invalid statistics: '+name)
        else:
            layer.track_running_stats = False
            layer.running_mean = layer.running_var = layer.num_batches_tracked = None
    base.require(all(not m.training for m in model.modules()), 'Dropout/training module enabled')
    base.require(extended.non_bn_digest(model, all_bn) == non_bn_before, 'BN scope changed non-buffer model state')
    return {'scope': scope, 'BN_count': len(all_bn), 'MLP_BN_count': len(mlp_bn),
            'frozen_BN_names': sorted(frozen), 'current_stat_BN_names': sorted(set(all_bn)-frozen),
            'MLP_BN_names': sorted(mlp_bn), 'non_bn_state_unchanged': True,
            'non_bn_state_sha256': non_bn_before, 'dense_CMA_artifact_checked_before_subset_selection': load_record,
            'BN_signature_before_inference': base.bn_signature(model, torch),
            'execution': 'Dense native full-grid forward in every module; no pruning, tile wrappers or parameter updates'}


def borrowed_inference(scope_callback):
    """Reuse the exact pinned function code without mutating the imported module globals."""
    base.require(base.sha(base.__file__) == extended.BASE_SHA, 'Frozen base source changed')
    namespace = dict(base.run_inference.__globals__)
    namespace['apply_bn'] = scope_callback
    return types.FunctionType(base.run_inference.__code__, namespace, 'run_bn_scope_frozen_loop',
                              base.run_inference.__defaults__, base.run_inference.__closure__)


def scope_preflight(a):
    base.require(base.sha(extended.__file__) == EXTENDED_SHA, 'Frozen extension changed')
    base.require(a.geometry == 'full9' and a.actual_pair_count == 0 and a.wall_minutes <= 15,
                 'Scope diagnostic requires full9, actual-pair-count0 and a wall cap <=15 minutes')
    base.require(a.bn_artifact is not None and a.svs_artifact is None, 'Only the fixed CMA176 artifact is accepted')
    context = extended.extended_preflight(a)
    result = context[-1]
    base.require(result['cases'] == CASES, 'Only the declared four dense BN-scope cases may run')
    # The original validator sees dense P-paper placeholders. The diagnostic output and actual
    # callback use explicit mixed-scope names, never label these graphs original P-code/D.
    for case in result['cases']:
        case['bn'] = case['bn_scope']
    result.update(schema='flow_bn_scope_diagnostic_v1', bn_scope_runner_sha256=base.sha(__file__),
                  borrowed_evaluator_sha256=extended.BASE_SHA, bn_artifact_validator_sha256=EXTENDED_SHA,
                  BN_labels={
                      'P_paper': 'All78 BN current-stat, eval/dropout-off, unchanged P-paper dense graph',
                      'MLP_original': 'Only24 MLP BN use checkpoint running stats; other54 current-stat',
                      'MLP_CMA176': 'Only24 MLP BN use common dense CMA176 stats; other54 current-stat',
                      'nonMLP_CMA176': 'Only54 non-MLP BN use common dense CMA176 stats; MLP24 current-stat'},
                  diagnostic_scope='Dense BN-conversion diagnosis only. No compression/recovery, skip-equivalence, flow-specific novelty or hardware-benefit claim.')
    return context


def main(argv=None):
    a = extended.parse_extended(argv)
    context = scope_preflight(a)
    result = context[-1]
    def callback(model, scope, torch):
        active = [entry for entry in result['arms'].values() if entry['status'] == 'RUNNING']
        base.require(len(active) == 1 and active[0]['case']['bn_scope'] == scope, 'BN callback/case scope mismatch')
        record = apply_scope(model, scope, a.bn_payload, torch)
        active[0]['BN_scope_application'] = record
        return record['BN_count']
    try:
        if a.run_inference:
            borrowed_inference(callback)(a, context)
    except BaseException as exc:
        result.update(status='FAILED_GLOBAL_NO_COMPLETE_CONCLUSION', error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        base.write_result(a.output, result)
    print(json.dumps({'status': result['status'], 'output': str(a.output), 'complete_cases': result.get('complete_cases', 0)}))
    if a.run_inference and result.get('complete_cases') != 4:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
