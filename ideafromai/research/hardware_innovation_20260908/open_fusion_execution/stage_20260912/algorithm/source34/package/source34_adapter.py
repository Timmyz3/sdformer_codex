"""One fixed contiguous3/3/4 source control on ordinary diag/raw.

Only the already-quantized source As changes; exponent, thresholds and all
consumers remain the actual current parent values. No requantization/fitting.
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def support():
    mask = np.zeros((10, 10), bool)
    for group in ([0, 1, 2], [3, 4, 5], [6, 7, 8, 9]):
        mask[np.ix_(group, group)] = True
    return mask


def install(helper, parameters=None):
    import torch
    parameters = dict(np.load(parameters or HERE/'deployed_constants.npz'))
    assert helper.kind == 'diag', 'This control is only ordinary raw/diag, never shared coordinates.'
    original = helper.matrices['As']
    q = parameters['As_q16'].astype(np.int64)
    exponent = int(parameters['As_exponent'])
    assert exponent == original['exponent']
    assert np.array_equal(q, original['q_numpy']*support())
    pos, neg = np.maximum(q, 0).sum(1), np.minimum(q, 0).sum(1)
    low, high = -(1 << 23), (1 << 23)-1
    lower, upper = pos*low+neg*high, pos*high+neg*low
    bound = helper.prove48('As_contiguous334', lower, upper)
    helper.matrices['As'] = dict(q=torch.as_tensor(q, dtype=torch.float64, device=helper.device),
        q_numpy=q, exponent=exponent, lower=lower, upper=upper)
    helper.matrix_metadata['As'] = dict(shape=[10, 10], exponent=exponent,
        input_domain='signed24 [-8388608,8388607]', integer_nonzero=int(np.count_nonzero(q)),
        original_nonzero=int(np.count_nonzero(original['q_numpy'])), dot_abs_bound=bound, fits_signed48=True,
        intervention='Retain current quantized coefficients only inside contiguous3/3/4 support; no refit.')
    helper.constants['As_q16'] = q.astype(np.int16)
    helper.constants['As_exponent'] = np.asarray(exponent)
    return dict(parent='ordinary original_ordered24 + onepass', source='contiguous3/3/4,34 coefficients',
        changed_fields=['As_q16'], preserved_exponent=exponent,
        source_comparisons_unchanged=True, consumers_unchanged=True)


def reference(i24, parameters):
    matrix = parameters['As_q16'].astype(np.int64)
    acc = matrix@i24.reshape(10, -1).astype(np.int64)
    divisor = 1 << int(parameters['As_exponent'])
    q, remainder = np.divmod(acc, divisor)
    q += ((2*remainder > divisor) | ((2*remainder == divisor) & ((q & 1) != 0)))
    q = q.clip(-(1 << 23), (1 << 23)-1)
    threshold = parameters['source_threshold'][:, None]
    direction = parameters['source_direction'][:, None]
    constant = parameters['source_constant'][:, None]
    gate = np.where(constant >= 0, constant.astype(bool), np.where(direction > 0, q >= threshold, q <= threshold))
    return q.reshape(i24.shape).astype(np.int32), gate.reshape(i24.shape)
