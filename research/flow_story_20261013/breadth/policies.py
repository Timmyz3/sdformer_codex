"""Ordinary breadth controls for complete 9x9 x T10 MLP service selection.

Inference-only policy(layer_name, u) -> bool[ceil(H/9),ceil(W/9)].
No checkpoint/GT/data access, recovery training, or hardware-cycle estimates.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
TP_PATH = HERE.parents[1]/'flow_story_20261012/transfer_experiments/tp/tp_score.py'
TP_SHA256 = '7d0db322ac276cfaa459cb6ece4b41722b85714bd25feb4f53ddd4529093108a'
_tp = None


def configure_tp_path(path):
    """Set an explicit local/read-only prior TP file before making policies."""
    global TP_PATH, _tp
    path = Path(path).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != TP_SHA256:
        raise ValueError('Pinned TP adaptation changed; review before updating breadth dependency')
    TP_PATH, _tp = path, None


def configure_prior_root(path):
    configure_tp_path(Path(path)/'transfer_experiments/tp/tp_score.py')


def _load_tp():
    global _tp
    if _tp is None:
        if hashlib.sha256(TP_PATH.read_bytes()).hexdigest() != TP_SHA256:
            raise ValueError('Pinned TP adaptation changed; review before updating breadth dependency')
        spec = importlib.util.spec_from_file_location('_flow_breadth_pinned_tp', TP_PATH)
        _tp = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = _tp
        spec.loader.exec_module(_tp)
    return _tp

POLICY_NAMES = ('uniform', 'seeded_random', 'u_meanabs', 'u_l2', 'temporal_change',
                'tp_literal', 'tp_factored')
DEFAULT_KEEP_FRACTIONS = (0.75, 0.5, 0.25)


def _validate_u(u):
    if (not isinstance(u, torch.Tensor) or u.dtype != torch.float32 or u.ndim != 5
            or tuple(u.shape[:2]) != (10, 1) or min(u.shape) < 1):
        raise ValueError('Expected FP32 complete [10,1,H,W,C] incoming u')
    if not bool(torch.isfinite(u).all()):
        raise ValueError('Nonfinite incoming u')


def _tile_means(score):
    """[1,H,W] -> [ny,nx], means over actual positions including tail."""
    _, h, w = score.shape
    values = []
    for y in range(0, h, 9):
        values.append(torch.stack([score[:, y:y+9, x:x+9].mean() for x in range(0, w, 9)]))
    return torch.stack(values)


@torch.no_grad()
def tp_factored_scores(u):
    """Same real-arithmetic SDT score; reduced summation changes FP rounding.

    Every literal [C,C,3,3] kernel output channel is identical. Sum C once,
    apply a single [1,1,3,3] convolution, then broadcast for the original cosine.
    This is ordinary algebra, not a new score or a bitwise equivalence promise.
    """
    _validate_u(u)
    t, b, h, w, c = u.shape
    x = u.permute(0, 1, 4, 2, 3).contiguous()
    cm = x.mean(dim=2)
    temporal = F.softmax((cm[1:]-cm[:-1]).abs().mean(0).reshape(b, -1), dim=1).reshape(b, h, w)
    mu = x.mean(0)
    summed = mu.sum(dim=1, keepdim=True)
    neighbor = F.conv2d(summed, torch.ones(1, 1, 3, 3, device=u.device)/9.0, padding=1)
    cosine = F.cosine_similarity(mu.reshape(b, c, -1),
        neighbor.expand(-1, c, -1, -1).reshape(b, c, -1), dim=1, eps=1e-6)
    spatial = F.softmax(-cosine, dim=1).reshape(b, h, w)
    total = temporal+spatial
    if not bool(torch.isfinite(total).all()):
        raise ValueError('Nonfinite factored score')
    return {'temporal': temporal, 'spatial': spatial, 'total': total,
        'cost': {'factored_channel_sum_additions': b*h*w*(c-1),
            'single_channel_conv_nominal_MAC_including_padding': 9*b*h*w,
            'kernel_elements': 9, 'cosine_pairs': b*h*w, 'cosine_vector_length': c,
            'softmax_calls': 2, 'softmax_elements_per_call': b*h*w,
            'full_T_u_feature_elements': u.numel(),
            'numeric_scope': 'real-arithmetic equivalence only; FP score/rank/mask may differ',
            'unpriced': ['temporal/spatial means', 'cosine', 'softmax', 'selection', 'validation', 'recording'],
            'cycles': None, 'traffic': None}}


def _rank(units, tie_policy):
    flat = units.flatten()
    if tie_policy == 'stable_index':
        return torch.argsort(flat, descending=True, stable=True)
    return torch.topk(flat, flat.numel()).indices


@torch.no_grad()
def compare_tp_factorization(u, keep_fractions=DEFAULT_KEEP_FRACTIONS, tie_policy='stable_index'):
    """Explicit comparison helper; not silently run in optimized inference."""
    _validate_u(u)
    if tie_policy not in ('stable_index', 'torch_topk'):
        raise ValueError('Unknown tie policy')
    _load_tp()
    original = _tp.official_sdt_scores(u.permute(0, 1, 4, 2, 3).contiguous())
    factored = tp_factored_scores(u)
    a, b = _tile_means(original['total']), _tile_means(factored['total'])
    result = {'shape_TBHWC': list(u.shape), 'tie_policy': tie_policy,
        'score_max_abs': float((original['total']-factored['total']).abs().max()),
        'score_mean_abs': float((original['total']-factored['total']).abs().mean()),
        'score_bitwise_equal': bool(torch.equal(original['total'], factored['total'])),
        'score_allclose_atol2e_6_rtol2e_5': bool(torch.allclose(original['total'], factored['total'], atol=2e-6, rtol=2e-5)),
        'tile_score_max_abs': float((a-b).abs().max()),
        'tile_rank_different_positions': int((_rank(a, tie_policy) != _rank(b, tie_policy)).sum()),
        'literal_cost': original['cost'], 'factored_cost': factored['cost'], 'masks': []}
    for ratio in keep_fractions:
        aa = _tp.select_spatial(original['total'], 'tile9', keep_fraction=ratio, tie_policy=tie_policy)
        bb = _tp.select_spatial(factored['total'], 'tile9', keep_fraction=ratio, tie_policy=tie_policy)
        result['masks'].append({'keep_fraction': ratio,
            'different_tiles': int((aa.unit_keep != bb.unit_keep).sum()),
            'different_real_spatial_pixels': int((aa.spatial_keep != bb.spatial_keep).sum()),
            'literal_mask': aa.unit_keep[0].cpu().tolist(), 'factored_mask': bb.unit_keep[0].cpu().tolist()})
    return result


class BreadthPolicy:
    def __init__(self, name, keep_fraction, *, seed=0, target_id, hidden_channels=None,
                 tie_policy='stable_index'):
        aliases = {'random': 'seeded_random', 'meanabs': 'u_meanabs', 'l2': 'u_l2', 'tp': 'tp_literal'}
        self.name = aliases.get(name, name)
        if self.name not in POLICY_NAMES:
            raise ValueError('Unknown policy: ' + str(name))
        if (isinstance(keep_fraction, bool) or not isinstance(keep_fraction, (int, float))
                or not math.isfinite(keep_fraction) or not 0 <= keep_fraction <= 1):
            raise ValueError('Invalid keep fraction')
        if not isinstance(target_id, str) or not target_id:
            raise ValueError('Explicit target_id is required for reproducible records/seeding')
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if tie_policy not in ('stable_index', 'torch_topk'):
            raise ValueError('Unknown tie policy')
        if hidden_channels is not None and (not isinstance(hidden_channels, dict) or any(
                not isinstance(k, str) or isinstance(v, bool) or not isinstance(v, int) or v < 1
                for k, v in hidden_channels.items())):
            raise ValueError('hidden_channels must map module names to positive integer widths')
        self.keep_fraction, self.seed, self.target_id = float(keep_fraction), seed, target_id
        self.hidden_channels = {} if hidden_channels is None else dict(hidden_channels)
        self.tie_policy, self.records, self.last_selection = tie_policy, [], None

    def __call__(self, layer_name, u):
        if torch.is_grad_enabled():
            raise ValueError('Inference-only policy; use inference_mode/no_grad')
        _validate_u(u)
        if not isinstance(layer_name, str) or not layer_name:
            raise ValueError('Layer name is required')
        t, _, h, w, c = u.shape
        ny, nx = math.ceil(h/9), math.ceil(w/9)
        n, k = ny*nx, math.floor(ny*nx*self.keep_fraction)
        pixels = torch.tensor([[min(9, h-9*y)*min(9, w-9*x) for x in range(nx)]
                               for y in range(ny)], dtype=torch.int64, device=u.device)
        score_cost = {'scope': 'operator counts/volumes only; not cycles, traffic or peak memory',
            'feature_elements_for_score': 0, 'finite_guard_elements': u.numel(),
            'selection_candidates': n, 'cycles': None}
        if self.name in ('uniform', 'seeded_random'):
            if self.name == 'uniform':
                # Deterministic evenly spaced row-major centers, without input statistics.
                indexes = [((2*i+1)*n)//(2*k) for i in range(k)] if k else []
                units = torch.zeros((ny, nx), dtype=torch.float32, device=u.device)
                mode_detail = 'evenly spaced row-major tile centers; independent of input and target'
            else:
                keys = [hashlib.sha256(json.dumps([self.seed, self.target_id, layer_name, h, w, i],
                         separators=(',', ':'), ensure_ascii=True).encode()).digest() for i in range(n)]
                order = sorted(range(n), key=lambda i: (keys[i], i))
                indexes = order[:k]
                # Display a strictly ordered ordinal score, not floating hash truncation.
                ranks = [0]*n
                for rank, i in enumerate(order): ranks[i] = n-rank
                units = torch.tensor(ranks, dtype=torch.float32, device=u.device).reshape(ny, nx)
                mode_detail = 'SHA256 order(seed,target_id,layer,H,W,tile_index); no global RNG mutation'
                score_cost.update(cpu_sha256_items=n, sha256_seed_components=['seed','target_id','layer','H','W','tile_index'])
            keep = torch.zeros(n, dtype=torch.bool, device=u.device)
            if indexes: keep[torch.tensor(indexes, device=u.device)] = True
            keep = keep.reshape(ny, nx)
        else:
            if self.name == 'u_meanabs':
                point = u.abs().mean(dim=(0, 4))
                units = _tile_means(point)
                mode_detail = 'mean(abs(u)) over full T, C and real tile positions'
                score_cost.update(feature_elements_for_score=u.numel(), abs_elements=u.numel())
            elif self.name == 'u_l2':
                point = u.square().mean(dim=(0, 4))
                units = _tile_means(point).sqrt()
                mode_detail = 'tile RMS=sqrt(mean(u^2)) over T,C,real positions; size-normalized L2 control'
                score_cost.update(feature_elements_for_score=u.numel(), squares=u.numel(), square_roots=n)
            elif self.name == 'temporal_change':
                point = (u[1:]-u[:-1]).abs().mean(dim=(0, 4))
                units = _tile_means(point)
                mode_detail = 'mean(abs(u[t]-u[t-1])) over t=1..9,C,real positions; ordinary control, not TP temporal term'
                score_cost.update(feature_elements_for_score=u.numel(), difference_and_abs_elements_each=9*h*w*c)
            else:
                _load_tp()
                scored = (_tp.official_sdt_scores(u.permute(0,1,4,2,3).contiguous())
                          if self.name == 'tp_literal' else tp_factored_scores(u))
                units = _tile_means(scored['total'])
                mode_detail = ('official SDT score at already produced u; mean over each tile; '
                    + ('literal channel-mixing dense convolution' if self.name == 'tp_literal'
                       else 'ordinary mathematically equivalent factoring; FP equivalence not assumed'))
                score_cost = scored['cost']
            if not bool(torch.isfinite(units).all()):
                raise ValueError('Nonfinite policy score')
            keep = torch.zeros(n, dtype=torch.bool, device=u.device)
            if k == n: keep.fill_(True)
            elif k:
                indexes = (_rank(units, self.tie_policy)[:k] if self.tie_policy == 'stable_index'
                           else torch.topk(units.flatten(), k).indices)
                keep[indexes] = True
            keep = keep.reshape(ny, nx)
        spatial = keep.repeat_interleave(9, 0).repeat_interleave(9, 1)[:h, :w]
        kept = int((pixels*keep).sum())
        d = self.hidden_channels.get(layer_name)
        record = {'schema': 'flow_breadth_tile_policy_v1', 'policy': self.name, 'target_id': self.target_id,
            'seed': self.seed, 'layer_name': layer_name, 'shape_TBHWC': list(u.shape),
            'policy_detail': mode_detail, 'ordinary_control_not_new_paper_method': True,
            'input': 'already-produced possibly signed real-valued attention residual u; no GT/final flow',
            'cancel_scope': 'subsequent MLP only; producer attention and u already executed',
            'tile_hw': [9, 9], 'full_T_shared': True, 'keep_fraction_requested': self.keep_fraction,
            'budget_rule': 'floor(tile_count*keep_fraction); unit count, NOT equal PE service or global frame budget',
            'tie_policy': self.tie_policy, 'tile_count': n, 'kept_tiles': k,
            'unit_keep': keep.cpu().tolist(), 'spatial_keep': spatial.cpu().tolist(),
            'unit_scores': units.cpu().tolist(), 'valid_pixels_per_unit': pixels.cpu().tolist(),
            'kept_real_pixels': kept, 'dense_real_pixels': h*w,
            'candidate_dense_tile_services': n, 'candidate_kept_tile_services': k,
            'candidate_cancelled_tile_services': n-k,
            'candidate_fc_rows_per_layer': t*kept, 'dense_fc_rows_per_layer': t*h*w,
            'candidate_cancelled_fc_rows_per_layer': t*(h*w-kept),
            'hidden_channels': d, 'candidate_fc_MACs_both_layers': None if d is None else 2*t*kept*c*d,
            'score_cost': score_cost, 'observed_execution': None,
            'scope': 'intermediate-checkpoint exploration; insufficient recovery cannot reject method',
            'upstream_tp_revision': _tp.UPSTREAM_REVISION if self.name.startswith('tp_') else None,
            'tp_adaptation_sha256': TP_SHA256 if self.name.startswith('tp_') else None,
            'factorization_runtime_equivalence_checked': False if self.name == 'tp_factored' else None}
        self.records.append(record)
        self.last_selection = {'unit_keep': keep, 'spatial_keep': spatial, 'unit_scores': units,
                               'valid_pixels_per_unit': pixels}
        return keep


def make_policy(name, keep_fraction, *, seed=0, target_id, hidden_channels=None, tie_policy='stable_index'):
    return BreadthPolicy(name, keep_fraction, seed=seed, target_id=target_id,
                         hidden_channels=hidden_channels, tie_policy=tie_policy)
