"""Untrained runtime coarse-flow head and explicit training-only labels.

Inference reads only the already-produced MLP input u=[10,1,H,W,C]. It first
averages the complete T axis, then the real points of each nonoverlapping 9x9
tile (including smaller tails). A shared C->16->2 MLP predicts dx,dy directly
in original-image pixels. There is no final-flow/GT input, warp or state.

The caller supplies displacement direction/time and feature-to-image geometry.
This file does not establish those DSEC/native-stage conventions. The model
is an interface/prototype, NOT a DSEC-trained predictor. Full T aggregation
requires u to exist before this head can decide; its producer is not canceled.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _contract(value):
    _require(isinstance(value, dict), 'Explicit flow_contract dict required')
    _require(set(value) == {'direction', 'time_interval', 'units'},
             'flow_contract needs exactly direction, time_interval and units')
    _require(all(isinstance(value[k], str) and value[k].strip() for k in value),
             'Flow contract fields must be nonempty strings')
    _require(value['units'] == 'original_image_pixels', 'Flow must use original-image pixel displacement')
    return dict(value)


def _positive_pair(value, name, integer=False):
    _require(isinstance(value, (tuple, list)) and len(value) == 2, name + ' needs two values')
    for item in value:
        good = type(item) is int if integer else type(item) in (int, float)
        _require(good and math.isfinite(item) and item > 0, name + ' must be positive and finite')
    return tuple(value)


def _shape(shape, in_channels=None):
    _require(len(shape) == 5 and all(type(v) is int and v > 0 for v in shape),
             'Expected positive integer [T,B,H,W,C] shape')
    t, b, h, w, c = shape
    _require(t == 10 and b == 1, 'This prototype requires complete T10 and B1')
    _require(in_channels is None or c == in_channels, 'Input channel count differs')
    return t, b, h, w, c


def shape_operation_counts(input_shape):
    """Nominal real-valued operations, NOT timing, ASIC cycles or energy.

    Pooling counts describe the declared reduction, not PyTorch's exact kernel
    implementation or rounding order. A division can be reciprocal multiply.
    GELU internals and memory-system traffic must be measured/modelled later.
    """
    t, b, h, w, c = _shape(tuple(input_shape))
    ny, nx = (h + 8) // 9, (w + 8) // 9
    n = b * ny * nx
    return {
        'schema': 'flow_runtime_coarse_head_cost_v1',
        'input_shape_TBHWC': list(input_shape), 'output_shape_BtilesYtilesX2': [b, ny, nx, 2],
        'tile_size_feature_yx': [9, 9], 'tile_count': n,
        'spatial_points_per_tile': [[min(9, h - y * 9) * min(9, w - x * 9)
                                     for x in range(nx)] for y in range(ny)],
        'temporal_reduction_additions': (t - 1) * b * h * w * c,
        'temporal_mean_divisions': b * h * w * c,
        'spatial_reduction_additions': b * c * (h * w - ny * nx),
        'spatial_mean_divisions': n * c,
        'linear_real_MACs': n * (c * 16 + 16 * 2),
        'linear_bias_additions': n * (16 + 2), 'GELU_scalar_evaluations': n * 16,
        'trainable_parameters': c * 16 + 16 + 16 * 2 + 2,
        'u_elements_required': t * b * h * w * c,
        'temporal_mean_intermediate_elements': b * h * w * c,
        'pooled_descriptor_elements': n * c, 'prediction_elements': n * 2,
        'u_producer_already_required': True, 'costs_are_measured_cycles': False,
        'omitted_from_counts': ['GELU internal implementation', 'kernel dispatch',
                                'memory transfers/caches', 'grouping/budget selection',
                                'fast/slow MLP branches', 'training/label construction'],
    }


class RuntimeCoarseFlowHead(nn.Module):
    """Shared 9x9xT10 pooled C->16 GELU->2 head; raw output is pixel dx,dy."""
    def __init__(self, in_channels, *, flow_contract):
        super().__init__()
        _require(type(in_channels) is int and in_channels > 0, 'Positive channel count required')
        self.in_channels = in_channels
        self.flow_contract = _contract(flow_contract)
        self.fc1 = nn.Linear(in_channels, 16)
        self.activation = nn.GELU(approximate='none')
        self.fc2 = nn.Linear(16, 2)

    def aggregate(self, u):
        _shape(tuple(u.shape), self.in_channels)
        _require(u.is_floating_point(), 'u must be floating point')
        # ceil_mode includes smaller real tails; count_include_pad=False keeps
        # their denominator equal to their actual spatial point count.
        temporal_mean = u.mean(dim=0).permute(0, 3, 1, 2)
        pooled = F.avg_pool2d(temporal_mean, kernel_size=9, stride=9,
                              ceil_mode=True, count_include_pad=False)
        return pooled.permute(0, 2, 3, 1)

    def forward(self, u):
        return self.fc2(self.activation(self.fc1(self.aggregate(u))))


def make_tile_targets(gt_2hw, valid_mask, *, feature_hw, feature_stride_pixels,
                      flow_contract, image_origin_xy=(0., 0.)):
    """Training-only valid-GT means over each tile's physical image support.

    feature_hw=(H,W); feature_stride_pixels=(sx,sy); image_origin_xy gives the
    continuous image boundary of feature coordinate (0,0), before any shifts.
    A feature cell occupies [x*sx,(x+1)*sx) by [y*sy,(y+1)*sy), translated by
    origin. GT pixel (x,y) has center (x+.5,y+.5); half-open supports are clipped
    to the GT image. This explicit support convention is NOT automatically a
    receptive-field or shifted-window correspondence. Caller must supply it.

    GT displacement is already in original-image pixels: labels are averaged,
    never divided by stride/tile size. Invalid GT can contain nonfinite values;
    a nonfinite value marked valid is an error. Empty targets are NaN with
    valid=False, serializable as null via targets_as_records; never a zero label.
    """
    contract = _contract(flow_contract)
    h, w = _positive_pair(feature_hw, 'feature_hw', integer=True)
    sx, sy = _positive_pair(feature_stride_pixels, 'feature_stride_pixels')
    _require(isinstance(image_origin_xy, (tuple, list)) and len(image_origin_xy) == 2 and
             all(type(v) in (int, float) and math.isfinite(v) for v in image_origin_xy),
             'Finite image_origin_xy required')
    ox, oy = image_origin_xy
    _require(isinstance(gt_2hw, torch.Tensor) and gt_2hw.ndim == 3 and gt_2hw.shape[0] == 2 and
             gt_2hw.is_floating_point(), 'GT must be floating [2,Himage,Wimage]')
    gh, gw = gt_2hw.shape[1:]
    _require(gh > 0 and gw > 0 and isinstance(valid_mask, torch.Tensor) and
             tuple(valid_mask.shape) == (gh, gw), 'GT mask/image shape mismatch')
    _require(valid_mask.device == gt_2hw.device, 'GT and mask must use the same device')
    _require(bool(((valid_mask == 0) | (valid_mask == 1)).all()), 'Valid mask must be binary')
    valid = valid_mask.bool()
    _require(bool(torch.isfinite(gt_2hw[:, valid]).all()), 'Valid GT contains nonfinite displacement')
    ny, nx = (h + 8) // 9, (w + 8) // 9
    labels = gt_2hw.new_full((1, ny, nx, 2), float('nan'))
    counts = torch.zeros((1, ny, nx), dtype=torch.long, device=gt_2hw.device)
    support = torch.zeros_like(counts)
    bounds = []
    with torch.no_grad():
        for y in range(ny):
            row = []
            for x in range(nx):
                left, right = ox + x * 9 * sx, ox + min((x + 1) * 9, w) * sx
                top, bottom = oy + y * 9 * sy, oy + min((y + 1) * 9, h) * sy
                x0, x1 = max(0, min(gw, math.ceil(left - .5))), max(0, min(gw, math.ceil(right - .5)))
                y0, y1 = max(0, min(gh, math.ceil(top - .5))), max(0, min(gh, math.ceil(bottom - .5)))
                row.append([left, top, right, bottom])
                patch_valid = valid[y0:y1, x0:x1]
                count = int(patch_valid.sum())
                counts[0, y, x] = count
                support[0, y, x] = (y1 - y0) * (x1 - x0)
                if count:
                    labels[0, y, x] = gt_2hw[:, y0:y1, x0:x1][:, patch_valid].mean(dim=1)
            bounds.append(row)
    return {
        'schema': 'flow_runtime_coarse_targets_v1', 'flow': labels.detach(),
        'valid': counts > 0, 'valid_pixel_count': counts, 'support_pixel_count': support,
        'flow_contract': contract, 'feature_hw': [h, w],
        'feature_stride_pixels_xy': [sx, sy], 'image_origin_xy': [ox, oy],
        'physical_bounds_xyxy': bounds,
        'support_convention': 'half-open image support; pixel centers x+.5,y+.5',
        'label_aggregation': 'valid-only arithmetic mean; no flow-unit rescaling',
    }


def targets_as_records(targets):
    """Small inspectable CPU/JSON representation; empty target is actual None."""
    _require(targets.get('schema') == 'flow_runtime_coarse_targets_v1', 'Foreign target schema')
    valid = targets['valid'].detach().cpu()
    values = targets['flow'].detach().cpu()
    counts = targets['valid_pixel_count'].detach().cpu()
    support = targets['support_pixel_count'].detach().cpu()
    return [{'row': y, 'col': x, 'valid': bool(valid[0, y, x]),
             'flow_dxdy': values[0, y, x].tolist() if bool(valid[0, y, x]) else None,
             'valid_pixel_count': int(counts[0, y, x]), 'support_pixel_count': int(support[0, y, x])}
            for y in range(valid.shape[1]) for x in range(valid.shape[2])]


def coarse_supervised_loss(prediction, targets, *, flow_contract, weighting='tile'):
    """Mean component-L1 over valid tile means; optional GT-pixel-count weights.

    weighting='tile': every valid tile has equal weight. 'valid_pixels': each
    tile mean error is weighted by its valid GT count; this is NOT pixelwise
    dense-flow L1. Empty-label tiles are excluded before arithmetic. With no
    valid tiles, loss=None and skip_optimizer=True; no fabricated zero target.
    No region/GT data is required by RuntimeCoarseFlowHead.forward.
    """
    _require(targets.get('schema') == 'flow_runtime_coarse_targets_v1', 'Foreign target schema')
    _require(_contract(flow_contract) == targets['flow_contract'], 'Prediction/GT direction or time contract differs')
    _require(weighting in ('tile', 'valid_pixels'), 'Unknown supervision weighting')
    truth, valid, counts = targets['flow'], targets['valid'], targets['valid_pixel_count']
    _require(isinstance(prediction, torch.Tensor) and prediction.is_floating_point() and
             tuple(prediction.shape) == tuple(truth.shape), 'Prediction/target shape mismatch')
    _require(prediction.device == truth.device and valid.device == truth.device and counts.device == truth.device,
             'Prediction/targets must share a device')
    _require(valid.dtype == torch.bool and tuple(valid.shape) == tuple(prediction.shape[:-1]) and
             tuple(counts.shape) == tuple(valid.shape) and bool((counts >= 0).all()) and
             bool((valid == (counts > 0)).all()), 'Inconsistent target validity/counts')
    n = int(valid.sum())
    if not n:
        return {'loss': None, 'valid_tiles': 0, 'weight_sum': 0, 'weighting': weighting,
                'skip_optimizer': True, 'status': 'no_valid_GT_tiles'}
    selected_prediction, selected_truth = prediction[valid], truth[valid]
    _require(bool(torch.isfinite(selected_prediction).all()) and bool(torch.isfinite(selected_truth).all()),
             'Nonfinite supervised prediction/target')
    per_tile = (selected_prediction - selected_truth).abs().mean(dim=-1)
    weights = counts[valid].to(per_tile.dtype) if weighting == 'valid_pixels' else torch.ones_like(per_tile)
    return {'loss': (per_tile * weights).sum() / weights.sum(),
            'valid_tiles': n, 'weight_sum': int(counts[valid].sum()) if weighting == 'valid_pixels' else n,
            'weighting': weighting, 'skip_optimizer': False,
            'status': 'valid_tile_mean_component_L1'}
