"""CPU geometry only: four fixed two-rectangle covers of one complete 3x3 cell.

No model/checkpoint/GT, fitting, GPU, native BN, or frozen executor imports.
Float64 scoring is a stated prototype contract, not an eventual FP32 kernel.
All leading axes and all channels are preserved by pool/expand.
"""
import numpy as np

CANDIDATES = (
    ('H1', ((0, 1, 0, 3), (1, 3, 0, 3))),
    ('H2', ((0, 2, 0, 3), (2, 3, 0, 3))),
    ('V1', ((0, 3, 0, 1), (0, 3, 1, 3))),
    ('V2', ((0, 3, 0, 2), (0, 3, 2, 3))),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(values):
    values = np.asarray(values)
    require(values.ndim >= 3 and values.shape[-3:-1] == (3, 3)
            and values.shape[-1] > 0, 'One complete 3x3 cell, with channels last')
    require(values.dtype in (np.dtype('float32'), np.dtype('float64')),
            'Explicit float32 or float64 input')
    require(bool(np.isfinite(values).all()), 'Finite input')
    return values.astype(np.float64, copy=False)


def candidate_rectangles(index):
    require(isinstance(index, (int, np.integer)) and 0 <= index < 4,
            'Fixed candidate index 0..3')
    return CANDIDATES[index][1]


def coverage(index):
    """The two rectangular masks, each with 3 or 6 actual spatial positions."""
    masks = np.zeros((2, 3, 3), dtype=np.bool_)
    for i, (y0, y1, x0, x1) in enumerate(candidate_rectangles(index)):
        masks[i, y0:y1, x0:x1] = True
    return masks


def pool_and_expand(values, index):
    x = checked(values)
    representatives = []
    expanded = np.empty_like(x)
    areas = []
    for y0, y1, x0, x1 in candidate_rectangles(index):
        block = x[..., y0:y1, x0:x1, :]
        representative = block.mean(axis=(-3, -2), dtype=np.float64)
        representatives.append(representative)
        expanded[..., y0:y1, x0:x1, :] = representative[..., None, None, :]
        areas.append((y1 - y0) * (x1 - x0))
    return {
        'name': CANDIDATES[index][0],
        'index': int(index),
        'areas': np.asarray(areas, dtype=np.int64),
        'representatives': np.stack(representatives, axis=-2),
        'expanded': expanded,
    }


def distortion_scores(representation):
    """Same objective for q(C=2) and generic features(C=384): scalar MSE.

    The denominator is all positions, channels and any leading axes. No
    per-channel normalization or fitting is silently applied. Scores from
    different representations have different units and are not compared.
    """
    x = checked(representation)
    return np.asarray([
        np.mean(np.square(x - pool_and_expand(x, i)['expanded']), dtype=np.float64)
        for i in range(4)
    ], dtype=np.float64)


def choose_partition(representation):
    scores = distortion_scores(representation)
    # Exact equal scores use the fixed H1,H2,V1,V2 order; no tolerance tie band.
    index = int(np.argmin(scores))
    return {'index': index, 'name': CANDIDATES[index][0], 'scores': scores}


def producer_geometry(time_steps=10, batch_size=1):
    require(isinstance(time_steps, int) and time_steps > 0
            and isinstance(batch_size, int) and batch_size > 0,
            'Positive complete T and B')
    return {
        'spatial_support': 9,
        'spatial_representatives': 2,
        'native_rows_per_FC': 9 * time_steps * batch_size,
        'selected_rows_per_FC': 2 * time_steps * batch_size,
        'candidate_count': 4,
        'selected_candidate_count': 1,
        'full_T_preserved': True,
        'all_channels_preserved': True,
        'measured_model_FC_calls': 0,
        'cycle_or_energy_claim': False,
    }
