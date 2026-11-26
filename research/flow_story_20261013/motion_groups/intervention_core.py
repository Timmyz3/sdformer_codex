"""Paired omission diagnostics, not a deployable selector or speed measurement."""
import hashlib
import math


def require(ok, message):
    if not ok:
        raise ValueError(message)


def choose_targets(rows, count=8):
    """Round robin the existing ordered development pool by sequence; no GT use."""
    by_sequence = {}
    for row in rows:
        by_sequence.setdefault(row['sequence'], []).append(row)
    selected = []
    for rank in range(max(map(len, by_sequence.values()))):
        for sequence in sorted(by_sequence):
            if rank < len(by_sequence[sequence]):
                selected.append(by_sequence[sequence][rank])
                if len(selected) == count:
                    return selected
    raise ValueError('Not enough declared development targets')


def comparable_pairs(flow, height, width, *, target, layer, count, quantize):
    """Same anchor and full9x9 endpoints for three one-hop direction controls.

    Flow is pooled dense-teacher dx/dy in output-pixel units. Its use is an
    offline diagnostic, never a prediction available before this MLP.
    All candidate anchors, including ineligible ones, are counted.
    """
    ny, nx = math.ceil(height / 9), math.ceil(width / 9)
    require(len(flow) == ny and all(len(r) == nx for r in flow), 'Flow grid differs')
    candidates = []
    rejected = {'tail_anchor': 0, 'no_direction': 0, 'endpoint_boundary_or_tail': 0}
    for gy in range(ny):
        for gx in range(nx):
            if 9*(gy+1) > height or 9*(gx+1) > width:
                rejected['tail_anchor'] += 1
                continue
            dx, dy = flow[gy][gx]
            offset = quantize(dx, dy)
            if offset is None:
                rejected['no_direction'] += 1
                continue
            ox, oy = offset
            offsets = {'motion': (ox, oy), 'rotated90': (-oy, ox), 'horizontal_local': (1, 0)}
            endpoints = {name: (gy+y, gx+x) for name, (x, y) in offsets.items()}
            if any(y < 0 or x < 0 or 9*(y+1) > height or 9*(x+1) > width
                   for y, x in endpoints.values()):
                rejected['endpoint_boundary_or_tail'] += 1
                continue
            key = hashlib.sha256(f'motion-pair-v1:{target}:{layer}:{gy}:{gx}'.encode()).hexdigest()
            candidates.append((key, {'anchor_yx': [gy, gx], 'flow_dxdy_output_pixels': [dx, dy],
                                     'endpoints_yx': {k: list(v) for k, v in endpoints.items()},
                                     'squared_distance_in_tiles': {k: x*x+y*y for k, (x, y) in offsets.items()},
                                     'distance_control': 'rotated90 preserves distance; horizontal is equal-work local control only',
                                     'each_member_valid_feature_points': 81}))
    chosen = [row for _, row in sorted(candidates)[:count]]
    return chosen, {'all_anchor_tiles': ny*nx, 'eligible_common_anchors': len(candidates),
                    'selected_anchors': len(chosen), 'requested_anchors': count, 'rejected': rejected,
                    'selection': 'SHA rank on common eligible anchors; no GT/error/activity ranking',
                    'scope': 'Full tiles, one-hop directional neighborhood; not an endpoint correspondence.'}


def pair_statistics(base_prediction, a_prediction, b_prediction, ab_prediction, gt, masks):
    """Separate EPE norm nonlinearity from nonlinear network response."""
    import numpy as np
    p0, pa, pb, pab, gt = [np.asarray(x, dtype=np.float64)
                          for x in (base_prediction, a_prediction, b_prediction, ab_prediction, gt)]
    require(gt.ndim == 3 and gt.shape[0] == 2 and np.isfinite(gt).all(), 'GT must be finite [2,H,W]')
    require(all(x.shape == gt.shape and np.isfinite(x).all() for x in (p0, pa, pb, pab)),
            'Prediction shape/finite mismatch')
    additive = pa + pb - p0
    endpoint = lambda pred: np.sqrt(np.square(pred-gt).sum(axis=0))
    errors = {name: endpoint(x) for name, x in (('base', p0), ('a', pa), ('b', pb), ('ab', pab), ('additive', additive))}
    residual = np.sqrt(np.square(pab-additive).sum(axis=0))
    pa_change = np.sqrt(np.square(pa-p0).sum(axis=0))
    pb_change = np.sqrt(np.square(pb-p0).sum(axis=0))
    output = {}
    for name, mask in masks.items():
        mask = np.asarray(mask)
        require(mask.dtype == np.bool_ and mask.shape == gt.shape[1:] and
                not np.any(mask & ~masks['all_valid']), 'Invalid region')
        n = int(mask.sum())
        if not n:
            output[name] = {'valid_pixels': 0, 'AEE': None, 'interaction': None}
            continue
        means = {k: float(v[mask].mean()) for k, v in errors.items()}
        da, db, dab = (means[k]-means['base'] for k in ('a', 'b', 'ab'))
        denom = float((pa_change+pb_change)[mask].mean())
        output[name] = {'valid_pixels': n, 'AEE': means,
                        'delta_a': da, 'delta_b': db, 'delta_ab': dab,
                        'interaction': dab-da-db,
                        'metric_only_interaction': means['additive']-means['a']-means['b']+means['base'],
                        'nonlinear_network_AEE_excess': means['ab']-means['additive'],
                        'flow_superposition_residual_mean': float(residual[mask].mean()),
                        'relative_superposition_residual': float(residual[mask].mean())/denom if denom else None}
    return output


def masked_branch(module, dropped_yx):
    """Context manager for exact full-shape output intervention, no producer skip."""
    import contextlib
    @contextlib.contextmanager
    def context():
        import torch
        dropped = tuple(tuple(x) for x in dropped_yx)
        require(len(set(dropped)) == len(dropped), 'Duplicate intervention tile')
        calls = []
        def hook(_module, inputs, output):
            require(not torch.is_grad_enabled() and not module.training, 'Inference only')
            require(output.ndim == 5 and tuple(output.shape[:2]) == (10, 1), 'Expected complete T10 B1')
            require(tuple(output.shape) == tuple(inputs[0].shape), 'Unexpected MLP branch layout')
            result = output.clone()
            points = 0
            for gy, gx in dropped:
                require(gy >= 0 and gx >= 0 and gy*9 < output.shape[2] and gx*9 < output.shape[3], 'Tile outside grid')
                ys, xs = slice(gy*9, min((gy+1)*9, output.shape[2])), slice(gx*9, min((gx+1)*9, output.shape[3]))
                result[:, :, ys, xs, :] = 0
                points += (ys.stop-ys.start)*(xs.stop-xs.start)
            calls.append({'shape_TBHWC': list(output.shape), 'masked_feature_points': points,
                          'logical_omitted_fc_rows_each': 10*points, 'producer_calls_cancelled': 0,
                          'actual_PE_cycles_saved': None})
            return result
        handle = module.register_forward_hook(hook)
        try:
            yield calls
            require(len(calls) == 1, 'Selected MLP must execute once per forward')
        finally:
            handle.remove()
    return context()
