#!/usr/bin/env python3
"""Bind deployment arithmetic to actual frozen heads; no data/quality/timing run."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np

P = Path(__file__).resolve().parent
B = P.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def audit_ridge(r):
    i, o = r['input_width'], r['output_width']
    shapes = {'coef': (i, o), 'mean': (i,), 'scale': (i,), 'intercept': (o,)}
    arrays = {k: np.asarray(r[k], dtype=np.float64) for k in shapes}
    for k, a in arrays.items():
        assert a.shape == shapes[k] and np.isfinite(a).all(), k
    assert (arrays['scale'] > 0).all()
    assert r['alpha'] == 0.1 and r['fit_rows'] == 64
    return dict(input_width=i, output_width=o,
                parameter_shapes={k: list(a.shape) for k, a in arrays.items()},
                actual_parameter_values=sum(a.size for a in arrays.values()),
                FP64_parameter_bytes=sum(a.nbytes for a in arrays.values()),
                dense_product_terms=arrays['coef'].size,
                mathematical_dot_adds=o * (i-1), bias_adds=o,
                center_subtractions=i, scale_divisions=i,
                real_algebra_folded_FP64_bytes_unverified=(i*o+o)*8,
                finite_parameters=True, positive_scales=True,
                actual_FP64_numpy_parameters_read=True)


def one_head(module, model, x, name):
    """Audit-only single-head extraction; no changes to the frozen source."""
    if name == 'direct_raw':
        return module.ridge_predict(model['models'][name], x)
    grid = int(name[-1])
    aux = module.ridge_predict(model['auxiliary']['g'+str(grid)], x)
    # The residual path can share this actual deterministic product array.
    products = module.component_products(aux, grid)
    if name.startswith('signed_'):
        return products.sum(axis=2)
    aug = np.concatenate((x, aux, products.reshape(len(x), -1)), axis=1)
    output = module.ridge_predict(model['models'][name], aug)
    if name.startswith('direct_residual_'):
        output = products.sum(axis=2) + output
    return output


branches = {}
for branch, source_rel, ledger_name, prefix in [
        ('voxel128_v1', 'algorithm/predictors.py', 'PREDICTOR_COST_COUNTS_V1.json', ''),
        ('directional160_v1', 'directional_branch/predictors.py', 'DIRECTIONAL_FEATURE_COST_SCOPE_V1.json', 'DIRECTIONAL_')]:
    folder = B/'fitted'/branch
    model_path, receipt_path = folder/'MODEL_V1.json', folder/'FIT_RECEIPT_V1.json'
    model, receipt, freeze = read(model_path), read(receipt_path), read(folder/'HEADS_FREEZE_V1.json')
    source = B/source_rel
    assert sha(model_path) == receipt['model_sha256'] == freeze['model_sha256']
    assert sha(receipt_path) == freeze['fit_receipt_sha256']
    assert sha(source) == receipt['predictor_source_sha256'] == model['metadata']['predictor_source_sha256']
    # The confirmation archive's frozen copy is read only as a model binding,
    # never a confirmation pack or output.
    assert sha(B/'confirmation_freeze_v1'/(prefix+'MODEL_V1.json')) == sha(model_path)
    assert receipt['identity_count'] == 64 and receipt['alpha'] == 0.1
    assert not receipt['tune_or_confirm_read'] and not receipt['GPU_used']
    assert model['metadata']['no_upgrade_threshold'] == 0
    assert model['metadata']['higher_is_better'] is False
    spec = importlib.util.spec_from_file_location('deploy_audit_'+branch, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded = module.load_model(model_path)
    width = model['metadata']['input_width']
    # Four artificial vectors, including the actual fitted feature mean.
    # This checks extraction equivalence, not real feature or action quality.
    x = np.stack((np.zeros(width), np.ones(width),
                  np.sin(np.arange(width, dtype=np.float64)),
                  loaded['models']['direct_raw']['mean']))
    suite = module.predict_suite(loaded, x)
    blocks = {}
    for section in ('models', 'auxiliary'):
        for name, r in model[section].items():
            blocks[section+'/'+name] = audit_ridge(r)
    heads = {}
    ledger = read(P/ledger_name)
    for name in model['methods']:
        if name == 'direct_raw':
            names, grid = ['models/'+name], 0
        else:
            grid = int(name[-1])
            names = ['auxiliary/g'+str(grid)]
            if not name.startswith('signed_'):
                names.append('models/'+name)
        keys = ['actual_parameter_values', 'FP64_parameter_bytes',
                'dense_product_terms', 'mathematical_dot_adds', 'bias_adds',
                'center_subtractions', 'scale_divisions',
                'real_algebra_folded_FP64_bytes_unverified']
        item = {k: sum(blocks[n][k] for n in names) for k in keys}
        item.update(required_ridge_blocks=names,
                    component_products_once=12*grid*grid if grid else 0,
                    signed_reduction_adds=6*(2*grid*grid-1) if grid and (name.startswith('signed_') or 'residual' in name) else 0,
                    residual_score_adds=6 if 'residual' in name else 0,
                    six_way_argmin_logical_compares=5,
                    optional_no_upgrade_compare=1,
                    deployed_feature_width=width,
                    feature_arithmetic_excluded_from_head_counts=True)
        actual = one_head(module, loaded, x, name)
        assert np.array_equal(actual, suite['scores'][name]), name
        for allow in (False, True):
            assert np.array_equal(module.choose_actions(actual, allow),
                                  module.choose_actions(suite['scores'][name], allow))
        item['synthetic_four_vector_FP64_scores_bitwise_equal_to_suite'] = True
        item['synthetic_fixed6_and_allowQ4_actions_equal_to_suite'] = True
        expected = ledger['single_candidate_heads'][name]
        assert item['dense_product_terms'] == expected['dense_MAC_equivalent_terms']
        pkey = 'FP64_deployment_parameter_bytes' if width == 128 else 'FP64_parameter_bytes'
        assert item['FP64_parameter_bytes'] == expected[pkey]
        assert item['component_products_once'] == expected['component_products_once_if_shared']
        assert item['signed_reduction_adds'] == expected['signed_reduction_adds']
        item['previous_analytic_ledger_matches_actual_fitted_shapes'] = True
        heads[name] = item
    suite_cost = {k: sum(v[k] for v in blocks.values()) for k in [
        'actual_parameter_values', 'FP64_parameter_bytes', 'dense_product_terms',
        'mathematical_dot_adds', 'bias_adds', 'center_subtractions', 'scale_divisions']}
    suite_cost.update(unique_ridge_blocks=7, method_outputs=7,
                      literal_component_products=408,
                      component_products_if_shared=204,
                      signed_reduction_adds=192, residual_score_adds=12,
                      actual_predict_suite_runs_all_seven=True,
                      this_is_not_single_head_deployment_cost=True)
    branches[branch] = dict(model_sha256=sha(model_path),
        fit_receipt_sha256=sha(receipt_path), source_sha256=sha(source),
        heads_freeze_sha256=sha(folder/'HEADS_FREEZE_V1.json'),
        theoretical_ledger_sha256=sha(P/ledger_name),
        fit_only_receipt_and_confirmation_frozen_model_binding_PASS=True,
        actual_ridge_blocks=blocks, single_deployed_heads=heads,
        research_all_seven_suite=suite_cost,
        synthetic_score_equivalence_scope='four synthetic feature rows, actual fitted FP64 arrays, original operation ordering; no actual data, FP32, fixedpoint or folding check')

summary = read(P/'results/SUMMARY_V1.json')
background = summary['candidates'][6]['dynamic_affine_BN_ticks']
upgrade = []
for r in summary['candidates'][:6]:
    delta = r['dynamic_affine_BN_ticks'] - background
    upgrade.append(dict(action=r['action'],
        same_model_Q4_complete_tailQ8_background_ticks=background,
        active_dynamic_affine_BN_ticks=r['dynamic_affine_BN_ticks'],
        incremental_ticks_over_background_without_predictor=delta,
        relative_operator_cost_increase_without_predictor=delta/background,
        extra_core_ticks_per_operator=20548080,
        four_isolated_operator_increment_sum_without_predictor=4*delta,
        dynamic_minus_same_static_ticks=170,
        whole_network_latency=False))
freeze = read(P/'FREEZE_V1.json')
unchanged = all(sha(P/name) == wanted for name, wanted in freeze['files'].items())
assert unchanged
out = dict(schema='round33_actual_fitted_deployability_audit_v1',
    status='PASS_MODEL_BINDINGS_SHAPES_SYNTHETIC_SINGLE_HEAD_EXTRACTION',
    branches=branches, same_model_upgrade_vs_Q4_cost=upgrade,
    frozen_controller_files_verified=len(freeze['files']),
    frozen_controller_unchanged=unchanged,
    feature_cost_sources=['PREDICTOR_COST_COUNTS_V1.json', 'DIRECTIONAL_FEATURE_COST_SCOPE_V1.json'],
    runtime_selected_head_not_yet_frozen=True,
    deployed_single_head_API_or_RTL_implemented=False,
    real_input_or_quality_data_read=False, confirmation_results_read=False,
    actual_fitted_arrays_read=True, new_RTL=False, GPU_used=False,
    predictor_cycles_measured=False, full_network_benefit_proven=False,
    coefficients_quantized_or_folded=False)
(P/'FINAL_DEPLOYABILITY_AUDIT_V1.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
print(json.dumps(dict(status=out['status'], branches={b: {
    'model_sha256':r['model_sha256'],
    'single_heads':len(r['single_deployed_heads']),
    'suite':r['research_all_seven_suite']} for b, r in branches.items()},
    controller_unchanged=unchanged, controller_files=len(freeze['files'])), indent=2))
