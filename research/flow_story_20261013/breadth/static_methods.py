"""CPU construction of ordinary static controls for native complete-T MLPs.

Fresh deep copies only. No checkpoint loading, dataset access, GPU, optimizer,
training, or hardware claims. Apply AFTER strict dense checkpoint loading and
BEFORE moving the new graph to GPU. Save the explicit structure report with its
state_dict; the old dense graph cannot load factored/narrowed keys unchanged.
"""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
SVS_PATH = HERE.parents[1] / 'flow_story_20261012/transfer_experiments/svs/svs_mlp.py'
SVS_SHA = '001cdf3e49e63a2eee04671c1752ad43a9b7b08c43e67ef02f04351f595a8273'
PROVENANCE_KEYS = ('checkpoint_sha256', 'data_contract_sha256', 'source_sha256',
                   'bn_policy_sha256', 'geometry_sha256', 'fit_ids_sha256')


def configure_prior_root(path):
    """Relocate a deployed prior research root without weakening the source pin."""
    global SVS_PATH
    candidate = Path(path).resolve() / 'transfer_experiments/svs/svs_mlp.py'
    if not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).hexdigest() != SVS_SHA:
        raise ValueError('Relocated SVS dependency is absent or its pinned SHA differs')
    SVS_PATH = candidate
    return str(SVS_PATH)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()


def _svs():
    require(hashlib.sha256(SVS_PATH.read_bytes()).hexdigest() == SVS_SHA, 'SVS dependency changed')
    spec = importlib.util.spec_from_file_location('flow_breadth_pinned_svs', SVS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ratio(value):
    require(not isinstance(value, bool) and isinstance(value, (int, float)) and
            math.isfinite(value) and 0 < value <= 1, 'keep_ratio must be finite in (0,1]')
    return float(value)


def _valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


class FactoredLinear(nn.Module):
    """x @ down.T @ up.T + original bias; arbitrary nonempty leading axes.

    No activation/norm between factors. The first factor can accept binary
    spikes; its output is signed real, so the SECOND factor is a real MAC.
    nn.Linear leaves are deliberate: generic execution hooks can see both.
    """
    def __init__(self, in_features, out_features, rank, bias=True, *, dtype=None, device=None):
        super().__init__()
        require(all(type(x) is int and x > 0 for x in (in_features, out_features, rank)) and
                rank <= min(in_features, out_features), 'Invalid factor dimensions')
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        self.down = nn.Linear(in_features, rank, bias=False, dtype=dtype, device=device)
        self.up = nn.Linear(rank, out_features, bias=bias, dtype=dtype, device=device)

    def forward(self, x):
        return self.up(self.down(x))

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}'


def choose_rank(linear, keep_ratio=None, rank=None, rank_basis='rank_fraction'):
    """Explicit rounding, including the possibility that full rank costs MORE.

    rank_fraction: floor(ratio*min(I,O)). parameter_ratio: the largest integer
    r whose r*(I+O)+original_bias_count <= ratio*(I*O+original_bias_count).
    A too-small target is rejected rather than silently forcing a larger rank.
    """
    require(isinstance(linear, nn.Linear), 'Expected an unfactored nn.Linear')
    require(rank_basis in ('rank_fraction', 'parameter_ratio'), 'Unknown rank_basis')
    require((keep_ratio is None) != (rank is None), 'Provide exactly one rank or keep_ratio')
    i, o = linear.in_features, linear.out_features
    if rank is None:
        ratio = _ratio(keep_ratio)
        if rank_basis == 'rank_fraction':
            rank = math.floor(ratio * min(i, o))
        else:
            bias_count = o if linear.bias is not None else 0
            rank = min(min(i, o), math.floor((ratio * (i*o + bias_count) - bias_count)/(i+o)))
    require(type(rank) is int and 1 <= rank <= min(i, o), 'Rank target gives no legal positive rank')
    return rank


def factor_linear(linear, rank):
    """Truncated CPU float64 SVD, then cast once to original FP32/FP64 dtype.

    The split is W_r=(U_r sqrt(S_r)) (sqrt(S_r) Vh_r). Bias remains only in up.
    Returns a new module and measured parameter/weight-error metadata, not AEE.
    """
    rank = choose_rank(linear, rank=rank)
    require(linear.weight.device.type == 'cpu' and linear.weight.dtype in (torch.float32, torch.float64),
            'Construct factors on CPU from FP32/FP64 weights before .to(GPU)')
    require(not any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in linear.modules()),
            'Remove observation hooks before factor construction')
    require(bool(torch.isfinite(linear.weight).all()) and
            (linear.bias is None or bool(torch.isfinite(linear.bias).all())), 'Nonfinite linear parameter')
    # Constructor RNG is isolated: choosing a compression control must not alter
    # subsequent recovery/dropout RNG. SVD is deterministic on this CPU backend.
    with torch.random.fork_rng(devices=[]):
        new = FactoredLinear(linear.in_features, linear.out_features, rank,
                             linear.bias is not None, dtype=linear.weight.dtype, device='cpu')
    with torch.no_grad():
        weight = linear.weight.detach().double()
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        root = s[:rank].sqrt()
        new.down.weight.copy_(root[:, None] * vh[:rank])
        new.up.weight.copy_(u[:, :rank] * root[None, :])
        new.down.weight.requires_grad_(linear.weight.requires_grad)
        new.up.weight.requires_grad_(linear.weight.requires_grad)
        if linear.bias is not None:
            new.up.bias.copy_(linear.bias)
            new.up.bias.requires_grad_(linear.bias.requires_grad)
        reconstructed = new.up.weight.double() @ new.down.weight.double()
        absolute = torch.linalg.vector_norm(weight-reconstructed).item()
        norm = torch.linalg.vector_norm(weight).item()
    new.train(linear.training)
    before = sum(p.numel() for p in linear.parameters())
    after = sum(p.numel() for p in new.parameters())
    i, o = linear.in_features, linear.out_features
    report = {'rank': rank, 'max_rank': min(i, o), 'rank_fraction_actual': rank/min(i, o),
              'in_features': i, 'out_features': o, 'factor_shapes': {'down': [rank, i], 'up': [o, rank]},
              'original_bias_preserved_in': 'up.bias' if linear.bias is not None else None,
              'parameters_before': before, 'parameters_after': after, 'parameter_ratio_actual': after/before,
              'parameter_reduction': before-after, 'is_parameter_compression': after < before,
              'weight_frobenius_error_after_dtype_cast': absolute,
              'weight_relative_frobenius_error_after_dtype_cast': absolute/norm if norm else 0.,
              'source': 'ordinary control: classical truncated matrix SVD; not a new compression method',
              'numeric_contract': 'CPU float64 SVD; factor dtype matches original; no quantization',
              'per_input_row_services': {'original_binary_weighted_accumulates_if_spike_input': i*o,
                  'new_first_factor_binary_weighted_accumulates_if_spike_input': i*rank,
                  'new_second_factor_real_scalar_products': rank*o,
                  'new_intermediate_real_elements': rank},
              'cycle_or_energy_equivalence_claimed': False}
    return new, report


def fake_quantize_weight_(linear, bits):
    """On an OWNED COPY only: symmetric per-output-channel round+dequant.

    No calibration, clipping search, activation quantization or integer kernel.
    Zero rows use scale=1. Bias and parameter dtype/storage stay unchanged.
    """
    require(isinstance(linear, nn.Linear) and type(bits) is int and bits in (4, 6, 8), 'Expected Linear and bits=4/6/8')
    require(linear.weight.device.type == 'cpu' and linear.weight.dtype in (torch.float32, torch.float64), 'CPU FP32/FP64 weights required')
    require(bool(torch.isfinite(linear.weight).all()), 'Nonfinite FC weight')
    with torch.no_grad():
        original = linear.weight.detach().clone()
        qmax = 2**(bits-1)-1
        peak = original.abs().amax(dim=1, keepdim=True)
        scale = torch.where(peak > 0, peak/qmax, torch.ones_like(peak))
        require(bool(torch.isfinite(scale).all()) and bool((scale > 0).all()), 'Quantization scale underflow/nonfinite')
        integer = (original/scale).round().clamp(-qmax, qmax)
        linear.weight.copy_(integer*scale)
        error = torch.linalg.vector_norm((linear.weight-original).double()).item()
        norm = torch.linalg.vector_norm(original.double()).item()
    return {'method': 'ordinary per-output-channel symmetric weight-only fake quantization',
            'bits': bits, 'qmin': -qmax, 'qmax': qmax, 'axis': 'output channel (weight dimension0)',
            'rounding': 'torch.round, nearest with ties to even', 'zero_row_scale': 1.,
            'scale_values': scale.flatten().tolist(), 'scale_count': linear.out_features,
            'weight_dtype_after_dequant': str(linear.weight.dtype), 'bias_changed': False,
            'weight_frobenius_error': error, 'weight_relative_frobenius_error': error/norm if norm else 0.,
            'storage_is_still_floating_point': True, 'arithmetic_is_still_floating_point': True,
            'full_PTQ_or_integer_hardware_claimed': False}


def _paths_and_closure(model, mlp_paths, svs):
    require(isinstance(model, nn.Module), 'Expected torch module')
    require(all(p.device.type == 'cpu' for p in model.parameters()) and
            all(b.device.type == 'cpu' for b in model.buffers()), 'CPU factory only; load dense checkpoint first')
    require(not any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in model.modules()),
            'Remove observation hooks before copying model')
    modules = dict(model.named_modules())
    paths = sorted(p for p, m in modules.items() if type(m).__name__ == 'MS_Spiking_Mlp') if mlp_paths is None else list(mlp_paths)
    require(paths and all(isinstance(p, str) for p in paths) and len(set(paths)) == len(paths), 'Unique explicit MLP paths required')
    paths = sorted(paths)
    all_paths = list(model.named_modules(remove_duplicate=False))
    shapes = {}
    for path in paths:
        require(path in modules, 'Unknown MLP path: ' + path)
        module = modules[path]
        require(sum(m is module for _, m in all_paths) == 1, 'Shared MLP needs a separate dependency audit')
        shapes[path] = svs.validate_native_mlp(module)
        owned = {id(p) for p in module.parameters()}
        names = [n for n, p in model.named_parameters(remove_duplicate=False) if id(p) in owned]
        require(all(path == '' or n.startswith(path + '.') for n in names), 'MLP shares external parameters')
    return paths, shapes


def validate_svs_replay(envelope, expected, paths, shapes, svs):
    """Metadata binding, not an assertion that an arbitrary caller's hash is trusted.

    Evaluation must provide hashes of its ACTUAL checkpoint/data/BN/geometry.
    A missing train-only collection is pending; random scores are never a fallback.
    """
    require(isinstance(envelope, dict) and envelope.get('schema') == 'flow_breadth_svs_replay_v1', 'Explicit SVS replay envelope required')
    require(isinstance(expected, dict) and all(_valid_sha(expected.get(k)) for k in PROVENANCE_KEYS)
            and expected.get('split') == 'train', 'Bind current checkpoint/data/source/BN/geometry/train-fit hashes explicitly')
    provenance = envelope.get('provenance', {})
    require(all(provenance.get(k) == expected[k] for k in (*PROVENANCE_KEYS, 'split')), 'SVS provenance differs from evaluation contract')
    ids = envelope.get('fit_ids')
    require(isinstance(ids, list) and ids and all(isinstance(i, str) and i for i in ids) and len(set(ids)) == len(ids), 'SVS fit IDs missing/duplicate')
    require(digest(ids) == expected['fit_ids_sha256'], 'Ordered train fit-ID digest differs')
    blocks = envelope.get('blocks', {})
    require(isinstance(blocks, dict) and set(paths) <= set(blocks), 'SVS scores missing selected MLP paths')
    scores = {}
    for path in paths:
        block = blocks[path]; width = shapes[path]['hidden_channels']
        require(block.get('schema') == 'flow_svs_score_v1' and block.get('formula') == svs.FORMULA,
                'SVS formula/site/numerical contract changed')
        require(block.get('sample_ids') == ids and block.get('sample_count') == len(ids) and
                block.get('hidden_width') == width, 'SVS fit membership/width differs')
        batches = block.get('batch_sizes', [])
        require(batches and all(type(b) is int and b > 0 for b in batches) and sum(batches) == len(ids), 'SVS sample accounting differs')
        sums = block.get('rank_sums', [])
        require(len(sums) == width and all(type(v) is int and v >= 0 for v in sums), 'Invalid accumulated SVS ranks')
        value = torch.as_tensor(block.get('scores'), dtype=torch.float64, device='cpu')
        require(value.shape == (width,) and bool(torch.isfinite(value).all()) and
                torch.allclose(value, torch.tensor(sums, dtype=torch.float64)/len(ids), atol=1e-12, rtol=0), 'SVS scores disagree with sample-weighted ranks')
        scores[path] = value
    return scores


def build_static_candidate(model, method, mlp_paths=None, *, keep_ratio=None, ranks=None,
                           factor_targets=('fc1', 'fc2'), seed=0, svs_scores=None,
                           svs_expected_provenance=None, pe_group_width=1, rank_basis='rank_fraction', bits=None):
    """Return (independent model, JSON-safe report); no optimizer or recovery.

    method: magnitude | random | svs | lowrank | weight_fakequant. paths=None means all native MLPs,
    including '' when model itself is an MLP. ranks is an explicit mapping
    {mlp_path: {fc1: rank, fc2: rank}} for exactly the targeted layers/factors.
    For lowrank provide ranks OR keep_ratio; pruning uses keep_ratio only.
    """
    require(method in ('magnitude', 'random', 'svs', 'lowrank', 'weight_fakequant'), 'Unknown static method')
    require(bits is None or method == 'weight_fakequant', 'bits only applies to weight_fakequant')
    require(type(seed) is int and 0 <= seed < 2**63, 'Invalid random seed')
    require(type(pe_group_width) is int and pe_group_width > 0, 'Invalid PE group width')
    svs = _svs()
    paths, shapes = _paths_and_closure(model, mlp_paths, svs)
    modules = dict(model.named_modules())
    report = {'schema': 'flow_breadth_static_candidate_v1', 'method': method, 'mlp_paths': paths,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'svs_dependency_sha256': SVS_SHA, 'requested_keep_ratio': keep_ratio,
              'parameters_before': sum(p.numel() for p in model.parameters()),
              'blocks': {}, 'CPU_construction': True, 'training_executed': False,
              'checkpoint_loaded_here': False, 'optimizer_state_reusable': False,
              'required_reconstruction': 'strict dense load -> same structural factory -> strict candidate state_dict load',
              'scope': 'ordinary static breadth control; no AEE/latency/energy claim'}
    if method != 'svs':
        require(svs_scores is None and svs_expected_provenance is None, 'SVS inputs supplied to a different method')
    if method == 'weight_fakequant':
        require(keep_ratio is None and ranks is None and pe_group_width == 1 and rank_basis == 'rank_fraction',
                'Weight fake quantization has no pruning/rank ratio')
        targets = tuple(factor_targets)
        require(targets and len(set(targets)) == len(targets) and set(targets) <= {'fc1', 'fc2'}, 'Only unique fc1/fc2 targets supported')
        new = copy.deepcopy(model)
        report.update(bits=bits, factor_targets=list(targets))
        for path in paths:
            target = new.get_submodule(path)
            for dropout in (target.drop1, target.drop2):
                if hasattr(dropout, 'reset'): dropout.reset()
            report['blocks'][path] = {'before': shapes[path], 'after': dict(shapes[path]),
                'weights': {name: fake_quantize_weight_(getattr(target, name), bits) for name in targets},
                'preserved': ['PSN1/PSN2', 'BN1/BN2', 'all bias', 'external C', 'hidden F', 'complete T']}
    elif method == 'lowrank':
        require((ranks is None) != (keep_ratio is None), 'Lowrank requires exactly one ratio or explicit ranks')
        targets = tuple(factor_targets)
        require(targets and len(set(targets)) == len(targets) and set(targets) <= {'fc1', 'fc2'}, 'Only unique fc1/fc2 targets supported')
        require(pe_group_width == 1, 'Factor rank alignment is explicit through ranks; hidden PE group rounding does not apply')
        if ranks is not None:
            require(isinstance(ranks, dict) and set(ranks) == set(paths) and
                    all(isinstance(ranks[p], dict) and set(ranks[p]) == set(targets) for p in paths), 'Explicit ranks must exactly cover selected paths/factors')
        new = copy.deepcopy(model)
        report.update(rank_basis=rank_basis, factor_targets=list(targets))
        for path in paths:
            source, target = modules[path], new.get_submodule(path)
            for dropout in (target.drop1, target.drop2):
                if hasattr(dropout, 'reset'): dropout.reset()
            block = {'before': shapes[path], 'after': dict(shapes[path]), 'factors': {},
                     'preserved': ['external C', 'hidden F', 'complete T', 'PSN1/PSN2', 'BN1/BN2', 'residual interface'],
                     'new_nonlinearity_between_factors': False}
            for name in targets:
                original = getattr(source, name)
                rank = choose_rank(original, keep_ratio=keep_ratio,
                                   rank=ranks[path][name] if ranks is not None else None, rank_basis=rank_basis)
                factor, info = factor_linear(original, rank)
                target.add_module(name, factor); block['factors'][name] = info
            report['blocks'][path] = block
    else:
        require(ranks is None and rank_basis == 'rank_fraction' and tuple(factor_targets) == ('fc1', 'fc2'),
                'Lowrank-specific options supplied to hidden-channel pruning')
        ratio = _ratio(keep_ratio)
        replay = validate_svs_replay(svs_scores, svs_expected_provenance, paths, shapes, svs) if method == 'svs' else None
        plan = {}
        for path in paths:
            width = shapes[path]['hidden_channels']
            # Explicit downward group rounding; never force a nonzero result.
            keep = math.floor(width*ratio/pe_group_width)*pe_group_width
            require(keep > 0, 'Requested ratio/group leaves no hidden channel')
            if method == 'magnitude':
                scores = svs.magnitude_score(modules[path]); source = 'ordinary control: mean absolute FC1 row weight, no BN folding'
            elif method == 'random':
                local_seed = int(hashlib.sha256((str(seed)+'\0'+path).encode()).hexdigest()[:16], 16) % (2**63)
                generator = torch.Generator(device='cpu').manual_seed(local_seed)
                ordering = torch.randperm(width, generator=generator)
                scores = torch.empty(width, dtype=torch.float64)
                scores[ordering] = torch.arange(width, 0, -1, dtype=torch.float64)
                source = 'ordinary control: uniform channel permutation; per-path deterministic private generator'
            else:
                scores = replay[path]; source = 'QP-paper SVS-only hidden-channel adaptation; replay of caller-bound train-only scores'
            selection = svs.select_hidden_channels(scores, keep, pe_group_width)
            plan[path] = selection['keep_indices']
            report['blocks'][path] = {'selection': selection, 'score_source': source,
                'actual_hidden_ratio': keep/width, 'rounding': 'floor(F*ratio/group)*group',
                'scores_sha256': digest(scores.tolist())}
            if method == 'random': report['blocks'][path]['local_seed'] = local_seed
        if paths == ['']:
            new, transformation = svs.narrow_mlp_deepcopy(model, plan[''], pe_group_width)
        else:
            require('' not in paths, 'Root MLP cannot be combined with nested MLP targets')
            new, transformation = svs.narrow_model_deepcopy(model, plan, pe_group_width)
        report['narrowing'] = transformation
        report['pe_group_width'] = pe_group_width
        if method == 'svs':
            report['svs_replay_canonical_digest'] = digest(svs_scores)
            report['svs_provenance'] = copy.deepcopy(svs_expected_provenance)
    report['parameters_after'] = sum(p.numel() for p in new.parameters())
    report['parameter_reduction'] = report['parameters_before'] - report['parameters_after']
    report['parameter_ratio_actual'] = report['parameters_after']/report['parameters_before']
    report['width_and_factor_plan_sha256'] = digest(report['blocks'])
    return new, report
