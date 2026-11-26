#!/usr/bin/env python3
"""Nine fixed MLP_CMA176 hybrid-BN actual-skip diagnostic cases."""
import json
from pathlib import Path
import types

import evaluate_breadth as base
import evaluate_breadth_extended as extended
import run_bn_scope as scope
import hybrid_tile_context as hybrid

POLICIES = ('uniform', 'seeded_random', 'temporal_change', 'tp_factored')
CASES = [{'id': 'dense_MLP_CMA176', 'method': 'dense', 'bn': 'D', 'bn_scope': 'MLP_CMA176'}] + [
    {'id': policy+'_keep'+str(round(ratio*100)), 'method': 'tile', 'bn': 'D', 'bn_scope': 'MLP_CMA176',
     'policy': policy, 'keep_fraction': ratio, 'seed': 17, 'execution': 'actual_skip'}
    for ratio in (.75, .5) for policy in POLICIES]


def hybrid_preflight(a):
    base.require(base.sha(scope.__file__) == hybrid.SCOPE_SHA and base.sha(extended.__file__) == scope.EXTENDED_SHA,
                 'Frozen dependency changed')
    base.require(a.geometry == 'full9' and a.actual_pair_count == 0 and a.wall_minutes <= 15,
                 'Requires full9, actual-pair-count0 and <=15minute wall cap')
    base.require(a.bn_artifact is not None and a.svs_artifact is None, 'Fixed CMA176 artifact is mandatory; SVS not used')
    context = extended.extended_preflight(a)
    result = context[-1]
    base.require(result['cases'] == CASES, 'Only the fixed dense + eight actual-skip cases may run')
    for case in result['cases']:
        case['bn'] = 'MLP_CMA176'
    result.update(schema='flow_hybrid_tile_diagnostic_v1', hybrid_runner_sha256=base.sha(__file__),
                  hybrid_context_sha256=base.sha(hybrid.__file__), bn_scope_helper_sha256=hybrid.SCOPE_SHA,
                  old_tile_sha256=hybrid.OLD_TILE_SHA, borrowed_evaluator_sha256=extended.BASE_SHA,
                  BN_labels={'MLP_CMA176': 'Only24 MLP BN common-dense CMA176 frozen; other54 current-stat on complete native grids'},
                  diagnostic_scope='Actual tile execution quality on interim train32. No inheritance of dense-masked AEE, recovery, equal-service or hardware speedup claim.')
    return context


def borrowed_inference(a, context):
    result = context[-1]
    expected_tile = (a.prior_root/'deployment_contract/tile_mlp.py').resolve()
    def import_override(name, path):
        path = Path(path).resolve()
        if name == 'breadth_tile_executor' or path == expected_tile:
            base.require(name == 'breadth_tile_executor' and path == expected_tile, 'Unexpected tile import alias/path')
            base.require(base.sha(path) == hybrid.OLD_TILE_SHA, 'Original tile source changed at execution')
            base.require(base.sha(hybrid.__file__) == result['hybrid_context_sha256'], 'Hybrid context changed after preflight')
            return hybrid.make_module(a.prior_root)
        return base.import_file(name, path)
    def bn_override(model, policy, torch):
        base.require(policy == 'MLP_CMA176', 'Unexpected graph BN scope')
        active = [v for v in result['arms'].values() if v['status'] == 'RUNNING']
        base.require(len(active) == 1 and active[0]['case']['bn_scope'] == policy, 'BN callback/case mismatch')
        record = scope.apply_scope(model, policy, a.bn_payload, torch)
        active[0]['BN_scope_application'] = record
        return record['BN_count']
    base.require(base.sha(base.__file__) == extended.BASE_SHA, 'Frozen inference loop changed')
    namespace = dict(base.run_inference.__globals__)
    namespace.update(apply_bn=bn_override, import_file=import_override)
    return types.FunctionType(base.run_inference.__code__, namespace, 'run_hybrid_frozen_loop',
                              base.run_inference.__defaults__, base.run_inference.__closure__)


def main(argv=None):
    a = extended.parse_extended(argv)
    context = hybrid_preflight(a)
    result = context[-1]
    try:
        if a.run_inference:
            borrowed_inference(a, context)(a, context)
    except BaseException as exc:
        result.update(status='FAILED_GLOBAL_NO_COMPLETE_CONCLUSION', error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        base.write_result(a.output, result)
    print(json.dumps({'status': result['status'], 'output': str(a.output), 'complete_cases': result.get('complete_cases', 0)}))
    if a.run_inference and result.get('complete_cases') != 9:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
