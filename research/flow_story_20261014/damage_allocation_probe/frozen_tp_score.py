"""TP-Spikformer SDT score -> native v2 MLP-only selection (inference only).

The score deliberately preserves upstream's channel-mixing 3x3 convolution.
Selection/grouping is our named adaptation; this is not a full TP reproduction.
No model/checkpoint/data access and no claim of hardware cycles or GPU speedup.
"""
from dataclasses import dataclass
import math
import time

import torch
import torch.nn.functional as F

UPSTREAM_REVISION = 'bcc915ad62254efa45800245d8abcf40905618fc'
UPSTREAM_FILE_SHA256 = 'e4a688393b34e9bf2c15a4ee783cabe8427ff1cfab22429bad95a1cc8e91049d'
ADAPTATION_VERSION = 'flow_tp_sdt_score_mlp_only_v1'


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + ' must be a positive integer')


def scoring_cost(shape, element_bytes=4):
    """Operator volumes of the implemented score, not measured traffic/cycles."""
    t, b, c, h, w = shape
    n = b*h*w
    return {
        'scope': 'nominal operator volumes; not PE cycles, DRAM traffic or peak memory',
        'input_shape_TBCHW': list(shape),
        'full_input_elements': t*n*c,
        'v2_layout_copy_bytes_if_materialized': t*n*c*element_bytes,
        'temporal_channel_mean_output_elements': t*n,
        'temporal_differences_and_abs_elements_each': (t-1)*n,
        'spatial_time_mean_output_elements': n*c,
        'official_dense_conv_mac_including_padded_taps': n*c*c*9,
        'official_conv_kernel_elements': c*c*9,
        'official_conv_kernel_allocated_bytes': c*c*9*element_bytes,
        'cosine_pairs': n,
        'cosine_vector_length': c,
        'cosine_eps': 1e-6,
        'softmax_calls': 2,
        'softmax_elements_per_call': n,
        'softmax_axis_length': h*w,
        'spatial_temporal_score_additions': n,
        'score_outputs_retained_elements': 3*n,
        'feature_ready_requirement': 'complete T and whole layer spatial map',
        'already_executed': ['attention', 'attention residual producing u'],
        'candidate_cancel_scope': 'subsequent MLP branch only; never u producer',
        'unpriced_operations': ['finite guards', 'cosine arithmetic', 'softmax exp/reductions',
            'selection comparisons/indices', 'mask materialization', 'CPU diagnostic copies'],
    }


@torch.no_grad()
def official_sdt_scores(x):
    """x: FP32 [T,B,C,H,W]; faithful upstream no-previous-mask score.

    T=1 upstream averages an empty tensor; reject it instead of inventing a score.
    Upstream creates its convolution kernel in default FP32, so FP32 is explicit.
    """
    if not isinstance(x, torch.Tensor) or x.ndim != 5 or x.dtype != torch.float32:
        raise ValueError('Expected FP32 [T,B,C,H,W] tensor')
    t, b, c, h, w = x.shape
    if t < 2 or min(b, c, h, w) < 1:
        raise ValueError('Score requires T >= 2 and nonempty axes')
    if not bool(torch.isfinite(x).all()):
        raise ValueError('Nonfinite score input')
    channel_mean = x.mean(dim=2)
    change = (channel_mean[1:] - channel_mean[:-1]).abs().mean(dim=0)
    temporal = F.softmax(change.reshape(b, -1), dim=1).reshape(b, h, w)
    time_mean = x.mean(dim=0)
    # Faithful to upstream: NOT groups=C and NOT division by C.
    kernel = torch.ones(c, c, 3, 3, device=x.device, dtype=torch.float32)/9.0
    neighborhood = F.conv2d(time_mean, kernel, padding=1)
    cosine = F.cosine_similarity(time_mean.reshape(b, c, -1),
                                 neighborhood.reshape(b, c, -1), dim=1, eps=1e-6)
    spatial = F.softmax(-cosine, dim=1).reshape(b, h, w)
    total = temporal + spatial
    if not bool(torch.isfinite(total).all()):
        raise ValueError('Nonfinite score output (possible input overflow)')
    return {'temporal': temporal, 'spatial': spatial, 'total': total,
            'cost': scoring_cost(tuple(x.shape))}


@dataclass
class Selection:
    unit_scores: torch.Tensor
    unit_keep: torch.Tensor
    spatial_keep: torch.Tensor
    valid_pixels_per_unit: torch.Tensor
    metadata: dict

    def complete_t_mask(self, t=10):
        """[T,B,H,W,1], shared support; channel axis broadcasts intentionally."""
        _positive_int(t, 't')
        return self.spatial_keep[None, :, :, :, None].expand(t, -1, -1, -1, 1)

    def json_record(self):
        # Explicit diagnostic materialization. This is not a hardware implementation.
        return dict(self.metadata, unit_scores=self.unit_scores.cpu().tolist(),
                    unit_keep=self.unit_keep.cpu().tolist(),
                    spatial_keep=self.spatial_keep.cpu().tolist(),
                    valid_pixels_per_unit=self.valid_pixels_per_unit.cpu().tolist())


@torch.no_grad()
def select_spatial(total, mode, *, keep_count=None, keep_fraction=None, tie_policy='torch_topk'):
    """Per-image/layer exact COUNT selection. Counts do not imply service budgets.

    mode token: individual H*W positions; diagnostic only for current TileMLP.
    mode tile9: mean over real positions of each 9x9 (including truncated tails).
    Default topk uses upstream tie behavior; optional stable ties are an adaptation.
    """
    if (not isinstance(total, torch.Tensor) or total.ndim != 3
            or not total.is_floating_point() or min(total.shape) < 1
            or not bool(torch.isfinite(total).all())):
        raise ValueError('Expected finite nonempty [B,H,W] scores')
    if mode not in ('token', 'tile9') or tie_policy not in ('torch_topk', 'stable_index'):
        raise ValueError('Unsupported mode or tie policy')
    if (keep_count is None) == (keep_fraction is None):
        raise ValueError('Declare exactly one of keep_count/keep_fraction')
    b, h, w = total.shape
    tile = 1 if mode == 'token' else 9
    ny, nx = math.ceil(h/tile), math.ceil(w/tile)
    pixels = torch.empty((ny, nx), device=total.device, dtype=torch.int64)
    if mode == 'token':
        units = total.clone()
        pixels.fill_(1)
    else:
        units = torch.empty((b, ny, nx), device=total.device, dtype=total.dtype)
        for y in range(ny):
            for x in range(nx):
                part = total[:, y*tile:min((y+1)*tile, h), x*tile:min((x+1)*tile, w)]
                units[:, y, x] = part.mean(dim=(1, 2))
                pixels[y, x] = part.shape[1]*part.shape[2]
    n = ny*nx
    if keep_fraction is not None:
        if isinstance(keep_fraction, bool) or not isinstance(keep_fraction, (float, int)):
            raise ValueError('keep_fraction must be numeric')
        if not math.isfinite(keep_fraction) or not 0 <= keep_fraction <= 1:
            raise ValueError('keep_fraction out of range')
        count = math.floor(n*keep_fraction)
    else:
        if isinstance(keep_count, bool) or not isinstance(keep_count, int) or not 0 <= keep_count <= n:
            raise ValueError('keep_count out of range')
        count = keep_count
    keep = torch.zeros((b, n), device=total.device, dtype=torch.bool)
    if count == n:
        keep.fill_(True)
    elif count:
        flat = units.reshape(b, n)
        if tie_policy == 'torch_topk':
            index = torch.topk(flat, count, dim=1).indices
        else:
            index = torch.argsort(flat, dim=1, descending=True, stable=True)[:, :count]
        keep.scatter_(1, index, True)
    keep = keep.reshape(b, ny, nx)
    expanded = keep.repeat_interleave(tile, 1).repeat_interleave(tile, 2)[:, :h, :w]
    real = expanded.sum(dim=(1, 2)).cpu().tolist()
    metadata = {'mode': mode, 'selection_rule': 'largest mean score per unit',
        'selection_budget': 'exact unit COUNT per image/layer; NOT exact PE service or frame-wide quota',
        'keep_count_per_image': count, 'candidate_units_per_image': n,
        'keep_fraction_requested': keep_fraction, 'count_rounding': 'floor for fraction',
        'tile_hw': [tile, tile], 'shape_BHW': [b, h, w],
        'tie_policy': tie_policy, 'tie_portability': 'torch_topk equal-score indices are device/runtime dependent',
        'full_T_shared': True, 'kept_real_spatial_pixels': real,
        'spatial_keep_bytes': b*h*w, 'unit_keep_bytes': b*n,
        'aggregation': 'mean of real spatial positions; no padded-zero score entries',
        'tile_aggregation_elements': b*h*w if mode == 'tile9' else 0,
        'topk_requested_k': count, 'topk_axis_length': n,
        'selection_cost_cycles': None, 'hardware_speedup': None,
        'scope': 'whole-tile executor compatible' if mode == 'tile9' else 'token diagnostic only'}
    return Selection(units, keep, expanded, pixels, metadata)


class TPTilePolicy:
    """Callable compatible with deployment_contract.tile_mlp.TileMLP policy.

    Call receives already-produced u [10,1,H,W,C]. Returns [ceil(H/9),ceil(W/9)].
    It records scoring/masks, while TileMLP remains responsible for execution/BN
    guards and observed FC rows. No calibration or checkpoint loading is done here.
    """
    def __init__(self, *, keep_fraction=None, keep_count=None, tie_policy='torch_topk'):
        if (keep_count is None) == (keep_fraction is None):
            raise ValueError('Declare exactly one count/fraction')
        self.keep_fraction, self.keep_count, self.tie_policy = keep_fraction, keep_count, tie_policy
        self.records = []
        self.last_selection = None

    def __call__(self, layer_name, u):
        if torch.is_grad_enabled():
            raise ValueError('Inference-only policy: use inference_mode/no_grad')
        if not isinstance(u, torch.Tensor) or u.ndim != 5 or tuple(u.shape[:2]) != (10, 1):
            raise ValueError('Expected complete [10,1,H,W,C] u')
        if u.device.type != 'cpu':
            # CUDA work is asynchronous; do not misreport enqueue time as scorer latency.
            elapsed = None
        else:
            elapsed = time.perf_counter()
        scored = official_sdt_scores(u.permute(0, 1, 4, 2, 3).contiguous())
        selected = select_spatial(scored['total'], 'tile9', keep_count=self.keep_count,
                                  keep_fraction=self.keep_fraction, tie_policy=self.tie_policy)
        if elapsed is not None:
            elapsed = time.perf_counter() - elapsed
        self.last_selection = selected
        record = selected.json_record()
        h, w = u.shape[2:4]
        kept = selected.metadata['kept_real_spatial_pixels'][0]
        record.update(layer_name=layer_name, shape_TBHWC=list(u.shape), score_cost=scored['cost'],
            adaptation_version=ADAPTATION_VERSION, upstream_revision=UPSTREAM_REVISION,
            upstream_file_sha256=UPSTREAM_FILE_SHA256,
            scoring_and_selection_cpu_wall_seconds=elapsed,
            cpu_wall_scope='includes validation/layout/score/select; excludes final JSON copies; not PE cycles',
            score_input='already-produced attention residual u; no GT or final-flow input',
            score_input_domain='real-valued possibly signed u; not asserted binary spike tensor',
            dense_fc_rows_per_layer=10*h*w, candidate_cancelled_fc_rows_per_layer=10*(h*w-kept),
            executed_fc_rows=None, evidence='score/mask only; obtain executed rows from TileMLP.records')
        self.records.append(record)
        return selected.unit_keep[0]


@torch.no_grad()
def mask_dense_branch(branch, selection):
    """Token diagnostic: masks an ALREADY executed MLP; claims no work saved."""
    if branch.ndim != 5 or tuple(branch.shape[1:4]) != tuple(selection.spatial_keep.shape):
        raise ValueError('Dense branch and selection shapes differ')
    return branch * selection.complete_t_mask(branch.shape[0]).to(branch.device)
