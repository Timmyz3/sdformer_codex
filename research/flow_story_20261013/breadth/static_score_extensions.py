"""Weight-only criterion components; new CPU graphs, no training or data access.

FPGM: sum of Euclidean distances between FC1 hidden weight rows, no norm-first
removal. Low distance sum is redundant and is removed first; high is kept.
BN control: abs(BN1.gamma), without Network Slimming's sparsity training.
Neither is a full-paper reproduction. No frozen breadth module is modified.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
STATIC_PATH = HERE / 'static_methods.py'
STATIC_SHA = '5b1938b2815a8201066db01c9a1d3920aa523116a30b22a6f4883fc86c477aab'
_PRIOR_ROOT = None
METHODS = ('fpgm', 'bn1_abs_gamma')
FORMULAE = {
    'fpgm': {
        'formula': 'score[i] = sum_j ||FC1.weight[i,:] - FC1.weight[j,:]||_2',
        'source': 'https://arxiv.org/abs/1811.00250',
        'official_code': 'https://github.com/he-y/filter-pruning-geometric-median/blob/master/pruning_cifar10.py',
        'location': 'Mask.get_filter_similar: scipy distance.cdist, similar_sum, similar_small_index',
        'scope': 'FPGM distance criterion component; convolution filters mapped to FC1 hidden rows',
        'norm_first_removal': False,
        'includes_bias_or_BN_folding': False,
        'self_distance': 'included and zero',
        'paper_training_or_recovery_implemented': False,
    },
    'bn1_abs_gamma': {
        'formula': 'score[i] = abs(BN1.weight[i])',
        'source': 'https://arxiv.org/abs/1708.06519',
        'official_code': 'https://github.com/Eric-mingjie/network-slimming',
        'scope': 'ordinary BN-scale control; NOT Network Slimming reproduction',
        'sparsity_regularization_or_recovery_implemented': False,
    },
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()


def _static():
    require(STATIC_PATH.is_file() and hashlib.sha256(STATIC_PATH.read_bytes()).hexdigest() == STATIC_SHA,
            'Frozen static_methods dependency is missing or changed')
    spec = importlib.util.spec_from_file_location('flow_breadth_static_pinned_extension', STATIC_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if _PRIOR_ROOT is not None:
        module.configure_prior_root(_PRIOR_ROOT)
    return module


def configure_prior_root(path):
    """Relocate the pinned prior SVS surgery, preserving its exact source check."""
    global _PRIOR_ROOT
    module = _static()
    result = module.configure_prior_root(path)
    _PRIOR_ROOT = str(Path(path).resolve())
    return result


def model_state_sha256(model):
    """Bind actual CPU tensor state, including BN buffers; not a checkpoint SHA.

    No storage aliases or serialization metadata enter this stable byte digest.
    The evaluator separately binds its actual checkpoint, native data and BN
    protocol. This factory does not assert those caller-owned facts.
    """
    sha = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu', 'CPU tensor state required')
        require(value.layout == torch.strided and not value.is_complex(), 'Dense real tensor state required')
        require(not value.is_floating_point() or bool(torch.isfinite(value).all()), 'Nonfinite model state: ' + name)
        header = {'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype)}
        sha.update(json.dumps(header, sort_keys=True, separators=(',', ':')).encode('utf-8') + b'\0')
        sha.update(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes() + b'\0')
    return sha.hexdigest()


def _score_input(mlp):
    shape = _static()._svs().validate_native_mlp(mlp)
    weight = mlp.fc1.weight
    require(weight.device.type == 'cpu' and weight.dtype in (torch.float32, torch.float64),
            'Score CPU FP32/FP64 native weights; move the candidate afterward')
    require(bool(torch.isfinite(weight).all()), 'Nonfinite FC1 weights')
    return shape


def fpgm_score(mlp, *, distance_row_chunk=64):
    """CPU float64 [hidden F]; keep HIGH, delete LOW (more redundant).

    Literal Euclidean distance, not squared distance, cosine, mean distance,
    geometric-median coordinates, nor mean distance only to kept channels.
    Chunking bounds the materialized distance matrix to min(chunk,F)*F;
    cdist's direct mode avoids a squared-norm cancellation shortcut. Full FC1
    weights remain resident. This is offline O(F^2*C) work, never free runtime
    computation. A different chunk/backend may change floating-point ties.
    """
    shape = _score_input(mlp)
    require(type(distance_row_chunk) is int and distance_row_chunk > 0, 'Positive integer row chunk required')
    weight = mlp.fc1.weight.detach().double().contiguous()
    width = shape['hidden_channels']
    scores = torch.empty(width, dtype=torch.float64, device='cpu')
    with torch.no_grad():
        for start in range(0, width, distance_row_chunk):
            distance = torch.cdist(weight[start:start+distance_row_chunk], weight, p=2,
                                   compute_mode='donot_use_mm_for_euclid_dist')
            scores[start:start+distance_row_chunk] = distance.sum(dim=1)
    require(bool(torch.isfinite(scores).all()), 'Nonfinite FPGM distance sum')
    return scores


def bn1_abs_gamma_score(mlp):
    """Ordinary control on existing BN1 affine scales; no sparsity training."""
    shape = _score_input(mlp)
    bn = mlp.bn1.norm_layer
    require(bn.affine and bn.weight is not None and tuple(bn.weight.shape) == (shape['hidden_channels'],),
            'BN1 affine gamma required')
    require(bn.weight.device.type == 'cpu' and bn.weight.dtype in (torch.float32, torch.float64),
            'CPU FP32/FP64 BN gamma required')
    scores = bn.weight.detach().double().abs().clone()
    require(bool(torch.isfinite(scores).all()), 'Nonfinite BN gamma')
    return scores


def build_static_score_candidate(model, method, mlp_paths=None, *, keep_ratio,
                                 pe_group_width=1, distance_row_chunk=64):
    """Return (independent narrow CPU model, JSON report), no old-state mutation.

    Same ratio rounding, descending selection, physical packing and dependency
    closure as frozen magnitude/static methods. Scores are computed afresh
    from the loaded model and included with state/source hashes in the report;
    no SVS envelope, synthetic fit IDs or unbound score replay is accepted.
    """
    require(method in METHODS, 'Unknown criterion component')
    require(type(pe_group_width) is int and pe_group_width > 0, 'Positive integer PE group required')
    require(type(distance_row_chunk) is int and distance_row_chunk > 0, 'Positive integer row chunk required')
    static = _static()
    ratio = static._ratio(keep_ratio)
    svs = static._svs()
    paths, shapes = static._paths_and_closure(model, mlp_paths, svs)
    state_sha = model_state_sha256(model)
    modules = dict(model.named_modules())
    # All paths/widths validated before any construction; source remains intact
    # even if the last target has an invalid ratio or score.
    widths = {p: math.floor(shapes[p]['hidden_channels']*ratio/pe_group_width)*pe_group_width for p in paths}
    require(all(widths.values()), 'Requested ratio/group leaves no hidden channel')
    require(paths == [''] or '' not in paths, 'Root MLP cannot combine with nested targets')
    report = {
        'schema': 'flow_breadth_static_score_extension_v1', 'method': method,
        'formula': dict(FORMULAE[method]), 'mlp_paths': paths, 'blocks': {},
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'static_dependency_sha256': STATIC_SHA, 'svs_dependency_sha256': static.SVS_SHA,
        'source_model_state_sha256': state_sha, 'requested_keep_ratio': ratio,
        'pe_group_width': pe_group_width, 'parameters_before': sum(p.numel() for p in model.parameters()),
        'collection': 'weight_only_no_data', 'fit_ids': None, 'CPU_construction': True,
        'checkpoint_loaded_here': False, 'training_executed': False, 'recovery_executed': False,
        'optimizer_state_reusable': False, 'spatially_dynamic': False,
        'required_reconstruction': 'strict dense load -> same structural factory -> strict candidate state_dict load',
        'scope': 'criterion component transfer on current weights; no full-paper/AEE/cycle/energy conclusion',
    }
    plan = {}
    for path in paths:
        mlp = modules[path]
        scores = fpgm_score(mlp, distance_row_chunk=distance_row_chunk) if method == 'fpgm' else bn1_abs_gamma_score(mlp)
        selection = svs.select_hidden_channels(scores, widths[path], pe_group_width)
        plan[path] = selection['keep_indices']
        shape = shapes[path]
        block = {
            'before': shape, 'selection': selection, 'scores': scores.tolist(),
            'scores_sha256': digest(scores.tolist()), 'source_mlp_state_sha256': model_state_sha256(mlp),
            'actual_hidden_ratio': widths[path]/shape['hidden_channels'],
            'rounding': 'floor(F*ratio/group)*group',
            'direction': 'higher score retained; lower removed; exact ties retain lower original index',
            'per_input_row_nominal_weighted_accumulates_before': 2*shape['input_channels']*shape['hidden_channels'],
            'per_input_row_nominal_weighted_accumulates_after': 2*shape['input_channels']*widths[path],
            'service_scope': 'two spike-input FC nominal weighted accumulates only; not PE cycles or real-valued PSN MACs',
            'runtime_score_computation': False,
        }
        if method == 'fpgm':
            block['offline_score_cost'] = {
                'pair_distances_evaluated_including_symmetry_and_self': shape['hidden_channels']**2,
                'coordinates_per_pair': shape['input_channels'],
                'distance_row_chunk': distance_row_chunk,
                'peak_materialized_distance_elements_bound': min(distance_row_chunk, shape['hidden_channels'])*shape['hidden_channels'],
                'arithmetic_dtype': 'torch.float64', 'complexity': 'O(F^2*C)',
            }
        report['blocks'][path] = block
    if paths == ['']:
        new, transformation = svs.narrow_mlp_deepcopy(model, plan[''], pe_group_width)
    else:
        new, transformation = svs.narrow_model_deepcopy(model, plan, pe_group_width)
    report['reused_surgery_report'] = transformation
    report['surgery_attribution'] = 'Reuse of physical FC1/BN1/FC2 slicing; its SVS-original report does not identify this scoring method'
    report['parameters_after'] = sum(p.numel() for p in new.parameters())
    report['parameter_reduction'] = report['parameters_before'] - report['parameters_after']
    report['parameter_ratio_actual'] = report['parameters_after']/report['parameters_before']
    report['width_and_score_plan_sha256'] = digest(report['blocks'])
    report['candidate_model_state_sha256'] = model_state_sha256(new)
    return new, report
