#!/usr/bin/env python3
"""Explicit initial or fixed-endpoint evaluation of one saved recovery arm."""
import argparse
import copy
import inspect
import json
import os
from pathlib import Path
import sys
import traceback

BASE_SHA = '1cacbaeea4d9eaf19a304c3faaa13f5278f0212b6cdc92210c4619447d571628'
START_SHA = 'd5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30'
ORDER_SHA = '44976c1e0e145b6ca374fa8697c43e06aaeea5dd0fd2f0c1d36f370ae601b15d'
CHECK64_SHA = '310d7c55fe5dc625d7b974bd03482513d46227254105501c088c17e580997a92'


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--breadth-root', type=Path, required=True)
    parser.add_argument('--candidate-checkpoint', type=Path, required=True)
    parser.add_argument('--candidate-sha256', required=True)
    parser.add_argument('--model-api-sha256', required=True)
    parser.add_argument('--evaluation-core-sha256', required=True)
    parser.add_argument('--recovery-manifest', type=Path, required=True)
    parser.add_argument('--check64-manifest', type=Path, required=True)
    parser.add_argument('--stage', choices=('initial', 'endpoint'), required=True)
    custom, remaining = parser.parse_known_args(argv)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(custom.breadth_root.resolve()))
    import evaluate_breadth as e
    e.require(e.sha(e.__file__) == BASE_SHA, 'Frozen evaluator changed')
    import model_api as api
    import evaluation_core as evaluator
    e.require(e.sha(api.__file__) == custom.model_api_sha256 and
              e.sha(evaluator.__file__) == custom.evaluation_core_sha256, 'Explicit recovery implementation pins differ')
    a = e.parse_args(remaining)
    for k, v in vars(custom).items(): setattr(a, k, v.resolve() if isinstance(v, Path) else v)
    e.require(a.checkpoint_sha256 == START_SHA and a.geometry == 'full9' and
              a.actual_pair_count == 0 and a.wall_minutes <= 15, 'Wrong bounded endpoint evaluation protocol')
    context = e.preflight(a)
    native, trainer, metrics, old_rows, hashes, config, original_run, original_side, regions, plan = context
    e.require(original_side['completed_optimizer_steps'] == 135930 and regions is not None,
              'Common checkpoint or complete regions absent')
    order = e.read_pin(a.recovery_manifest, ORDER_SHA)
    check = e.read_pin(a.check64_manifest, CHECK64_SHA)
    payload, side = api.read_checkpoint(a.candidate_checkpoint, a.candidate_sha256)
    contract = payload['contract']
    e.require(contract.get('start_checkpoint_sha256') == START_SHA and contract.get('start_step') == 135930 and
              contract.get('data_contract') == plan['data_contract'] and contract.get('source_sha256') == plan['source_sha256'] and
              contract.get('source_manifest_sha256') == e.sha(a.source_manifest), 'Recovery checkpoint native provenance differs')
    e.require(contract.get('recovery_manifest_sha256') == ORDER_SHA and contract.get('training_target_ids') == order['target_ids'] and
              contract.get('training_target_ids_sha256') == e.digest(order['target_ids']) and
              contract.get('canonical_manifest_sha256') == e.sha(a.canonical) and
              contract.get('roles_sha256') == e.sha(a.roles), 'Recovery population/source differs')
    e.require(contract['implementation_sha256']['model_api.py'] == a.model_api_sha256 and
              contract['architecture_sha256'] == api.digest(payload['architecture']) and
              contract['arm'] == payload['architecture']['arm'], 'Recovery graph/implementation differs')
    e.require(contract.get('one_pass') is True and contract.get('max_target_visits') == 4573 and
              contract.get('no_validation_or_best_checkpoint_selection') is True and
              payload.get('provenance', {}).get('validation_used') is False, 'Endpoint selection contract differs')
    visited = payload['training']['visited_targets']
    e.require(visited == (0 if a.stage == 'initial' else 4573), 'Only saved step0 or fixed full-pass endpoint may be evaluated')
    e.require(payload['training']['completed_optimizer_steps']+payload['training']['zero_valid_crops'] == visited,
              'Recovery update/empty-target accounting differs')
    e.require(check['canonical_sha256'] == e.sha(a.canonical) and check['roles_sha256'] == e.sha(a.roles),
              'Check64 source differs')
    populations = {'gate_dev32': [r['target_id'] for r in old_rows]}
    if a.stage == 'endpoint': populations['check64'] = check['target_ids']
    all_rows = metrics.canonical_rows(a.canonical); roles = metrics.auxiliary_roles(a.roles, all_rows)
    e.require(len(check['target_ids']) == len(set(check['target_ids'])) == 64 and
              not set(check['target_ids']) & set(populations['gate_dev32']) and
              not set(check['target_ids']) & set(order['target_ids']) and
              all(roles.get(t) == 'gate_development' for t in check['target_ids']), 'Check64 membership leaked')
    identity = {'candidate_checkpoint_sha256': a.candidate_sha256, 'candidate_sidecar': side,
                'source_checkpoint_sha256': START_SHA, 'recovery_contract_sha256': payload['contract_sha256'],
                'architecture_sha256': payload['architecture_sha256'], 'arm': contract['arm'],
                'recovery_training': payload['training'], 'SVS_binding': contract['svs_binding'],
                'check64_manifest_sha256': CHECK64_SHA, 'model_api_sha256': a.model_api_sha256,
                'wrapper_sha256': e.sha(__file__), 'final_valid_used': False}
    if not a.run_inference:
        e.write_result(a.output, {'status': 'RECOVERY_EVAL_PREFLIGHT_PASS_NO_INFERENCE', 'identity': identity,
                                 'populations': populations, 'native_preflight': plan})
        return 0
    import torch
    os.environ['SDFORMER_SNN_BACKEND'] = 'torch'
    sys.path.insert(0, str(a.source_root))
    from spikingjelly.activation_based import functional
    from models.STSwinNet_SNN.Spiking_submodules import PSN
    from utils.runtime_backend import configure_snn_backend
    for item in (PSN, configure_snn_backend):
        e.require(Path(inspect.getfile(item)).resolve().is_relative_to(a.source_root), 'Unpinned model backend')
    full_config = copy.deepcopy(payload['effective_config'])
    full_config['swin_transformer']['input_size'] = [480, 640]
    def factory():
        model = api.construct_model(full_config, payload['architecture'], source_root=a.source_root,
                                    prior_root=a.prior_root, breadth_root=a.breadth_root)
        model.load_state_dict(payload['model_state_dict'], strict=True)
        configure_snn_backend(model, torch.device(a.device), full_config, PSN)
        return model
    try:
        result = evaluator.run_evaluation(a, context, factory, identity, populations, e, torch, functional)
    except BaseException as exc:
        e.write_result(a.output.with_suffix(a.output.suffix+'.failure.json'), {
            'status': 'CANDIDATE_EVALUATION_INCOMPLETE', 'identity': identity,
            'error': type(exc).__name__+': '+str(exc), 'traceback': traceback.format_exc()})
        raise
    print(json.dumps({'status': result['status'], 'arm': contract['arm'], 'stage': a.stage,
                      'groups': len(result['evaluations']), 'output': str(a.output)}))
    return 0


if __name__ == '__main__': raise SystemExit(main())
