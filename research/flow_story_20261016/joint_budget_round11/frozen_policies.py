"""Inference-only R10 coefficients and deterministic full rankings; no fitting/GT API."""
import json
import numpy as np
from source_contract import HERE, load_source, require, sha, source_path

TILES = tuple((r, c) for r in range(3) for c in range(4))
BUDGETS = (2, 4)
UNIFORM_ORDER = (3, 8, 1, 10, 5, 6, 0, 11, 2, 9, 4, 7)
POLICIES = ('B5_task', 'B5_MSE', 'rich37_task', 'rich37_MSE', 'PCA11_task', 'PCA11_MSE',
            'flow11_task', 'flow11_MSE', 'B5_small4_task', 'B5_small4_MSE',
            'fit_position_mean', 'fixed_uniform_index3', 'amplitude_abs_mean',
            'pooling_residual_mse', 'psn1_activity', 'psn1_spatial_neighbor_hamming')
pm = load_source('predictor_models', '_r11_frozen_numpy_predictors')


def array_model(model):
    out = dict(model)
    for key in ('mean', 'std', 'beta', 'intercept'):
        out[key] = np.asarray(out[key], dtype=np.float64)
    return out


def load_models():
    source = HERE/'inference_models_v1.json'
    require(sha(source) == 'c56af012cf4ed212a9c50131ee3e82d046604ff1debbac9afcf16445b253edbd', 'Compact coefficient bytes differ')
    result = json.loads(source.read_text())
    require(result['schema'] == 'flow_round11_frozen_inference_only_v1', 'Unknown compact coefficient source')
    require(result['origin_sha256'] == sha(source_path('r10_decisions')), 'Original coefficient artifact differs')
    require(tuple(result['policy_names']) == POLICIES and tuple(result['uniform_order']) == UNIFORM_ORDER,
            'Frozen policies or complete uniform order differ')
    require(set(result['final_heads']) == {'flow', 'PCA'} and len(result['selector_models']) == 10,
            'Fixed 2 projections and 10 selector models required')
    return result


def ranking(scores):
    scores = np.asarray(scores)
    require(scores.shape == (12,) and np.isfinite(scores).all(), 'Twelve finite scores required')
    return sorted(range(12), key=lambda i: (-float(scores[i]), TILES[i]))


def infer(models, cells, ordinary):
    """No frame, GT, endpoint, branch-MSE or file argument is accepted."""
    cells = np.asarray(cells, dtype=np.float32)
    require(cells.shape == (10, 14, 384) and np.isfinite(cells).all(), 'Complete finite CC descriptor required')
    require(len(ordinary) == 12 and {tuple(x['tile']) for x in ordinary} == set(TILES), 'Twelve ordinary tiles required')
    heads = {name: array_model(model) for name, model in models['final_heads'].items()}
    flow = pm.predict_fp32(heads['flow'], cells)
    pca = pm.predict_fp32(heads['PCA'], cells)
    features = pm.feature_families(ordinary, cells, flow, pca)
    scores = {}
    for name, model in models['selector_models'].items():
        family = 'B5' if name.startswith('B5_small4_') else name.rsplit('_', 1)[0]
        scores[name] = pm.predict_fp32(array_model(model), features[family])
    scores['fit_position_mean'] = np.asarray(models['position_mean_gain'], dtype=np.float64)
    scores['fixed_uniform_index3'] = np.asarray([12-UNIFORM_ORDER.index(i) for i in range(12)], dtype=np.int64)
    ordinary_by_tile = {tuple(item['tile']): item for item in ordinary}
    for key in POLICIES[12:]:
        scores[key] = np.asarray([ordinary_by_tile[tile][key] for tile in TILES])
    decisions = {}
    for name in POLICIES:
        order = ranking(scores[name])
        require(set(order) == set(range(12)), 'Full ordering required')
        decisions[name] = {'ranking_indices': order, 'ranking_tiles': [list(TILES[i]) for i in order],
                           'scores': scores[name].tolist(),
                           'score_dtype': str(scores[name].dtype),
                           'selected': {str(k): [list(TILES[i]) for i in sorted(order[:k])] for k in BUDGETS}}
    require(tuple(decisions['fixed_uniform_index3']['ranking_indices']) == UNIFORM_ORDER, 'Uniform order differs')
    return {'schema': 'flow_round11_no_GT_full_rankings_v1', 'decisions': decisions,
            'features': pm.serializable(features), 'flow_projection': flow.tolist(), 'PCA_projection': pca.tolist(),
            'GT_used': False, 'fitting_performed': False,
            'compact_models_sha256': sha(HERE/'inference_models_v1.json')}


def arm_name(policy, k):
    require(policy in POLICIES and k in BUDGETS, 'Unknown frozen arm')
    return policy+'_k'+str(k)


def joint_plan(decisions):
    """Dedup within one frame only; all entries retain canonical row/column mask."""
    seen, result = {}, []
    for policy in POLICIES:
        for k in BUDGETS:
            selected = tuple(tuple(t) for t in decisions[policy]['selected'][str(k)])
            require(len(selected) == len(set(selected)) == k and set(selected) <= set(TILES), 'Illegal fixed joint mask')
            require(tuple(sorted(selected)) == selected, 'Canonical row-column ordering required')
            key = ('actual', 'stage02_target_block2', selected)
            name = arm_name(policy, k)
            result.append({'arm': name, 'policy': policy, 'k': k, 'selected_tiles': [list(t) for t in selected],
                           'alias_of': seen.get(key), 'execution_mode': 'actual', 'scope': 'stage02_target_block2'})
            if key not in seen:
                seen[key] = name
    require(len(result) == 32, 'All16 policies at both budgets required')
    return result
