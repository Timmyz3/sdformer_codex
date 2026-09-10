"""Fixed T10 four-factor orthogonal basis and independent neuron readouts.

Each pair maps (x,y) -> (x+y, sigma*(x-y)); the reverse factor uses
(u,v) -> (u+sigma*v, u-sigma*v), in reverse layer order. Hard sigma is ±1.
F^T F=16I and B=F/4, so canonical Q=B I is independent of both readout gains.
A zero source gain makes that source readout constant; it never singularizes B.
All pairings and the number of factors are fixed. Initial signs are all+1.
Optional20 sign logits use exact hard-forward signs and identity STE backward;
this is a surrogate gradient, not the derivative of a discrete sign choice.
The inverse factors use a different local continuation: 1/sigma equals sigma
at either hard sign, but has derivative -1 there. Reverse therefore evaluates
2*sigma.detach()-sigma. Fixed-sign forward values and input gradients stay
unchanged, while the sign derivatives of inverse(forward(X)) cancel. Applying
the forward +STE to the transposed sign would give that identity path a false
sign gradient. The reverse is the exact transpose at hard signs; its parameter
backward follows the local inverse, not the transpose's linear extension.

INTEGER CONTRACT: raw factor numerators use exact signed add/subtract/negate.
Reserve four guard bits for each four-layer pass; this CPU module promotes
integer operands to int64. No stage floor, RNE, clipping or lifting is applied.
Canonical forward and inverse each divide the real raw result by4 only at the
end. An integer code at unchanged fractional precision can do that exactly
only when its numerator is divisible by4. Otherwise retain two fractional
bits (store N=F X as the code for Q), or accept a separately measured rounding
function. With stored N, F^T N=16X and recovery at X's original scale divides
by16. Mid-stage floor lifting is nonlinear and does not obey these identities.

This module provides only structure and closed-form initialization from
caller-supplied moments. It reads no dataset, trains nothing, selects no signs,
and makes no AEE, cycle, fixed24-bit invertibility or novelty claim.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


MATCHINGS = (
    ((0,1), (2,3), (4,5), (6,7), (8,9)),
    ((0,2), (1,3), (4,6), (5,8), (7,9)),
    ((0,4), (1,5), (2,7), (3,8), (6,9)),
    ((0,6), (1,7), (2,8), (3,9), (4,5)),
)


class _HardSignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        return torch.where(logits >= 0, torch.ones_like(logits), -torch.ones_like(logits))

    @staticmethod
    def backward(ctx, gradient):
        return gradient


class FastTemporalBasis(nn.Module):
    def __init__(self, learnable_signs=False, *, dtype=torch.float32, device=None):
        super().__init__()
        logits = torch.ones(4, 5, dtype=dtype, device=device)
        if learnable_signs:
            self.sign_logits = nn.Parameter(logits)
        else:
            self.register_buffer('sign_logits', logits)
        self.register_buffer('matchings', torch.tensor(MATCHINGS, dtype=torch.long, device=device))

    def signs(self):
        return _HardSignSTE.apply(self.sign_logits)

    def _pass(self, value, reverse=False, normalized=True, return_stages=False):
        # The leading axis is always the ten original time coordinates.
        if value.shape[0] != 10:
            raise ValueError('The fixed temporal basis has leading dimension T10.')
        result = value if value.is_floating_point() else value.to(torch.int64)
        signs = self.signs().to(device=result.device, dtype=result.dtype)
        stages = []
        for layer in (range(3, -1, -1) if reverse else range(4)):
            pairs = self.matchings[layer]
            first, second = pairs[:, 0], pairs[:, 1]
            x, y = result.index_select(0, first), result.index_select(0, second)
            sigma = signs[layer].reshape(5, *([1]*(result.ndim-1)))
            if reverse:
                reciprocal_sign = 2*sigma.detach()-sigma
                signed = reciprocal_sign*y
                u, v = x+signed, x-signed
            else:
                u, v = x+y, sigma*(x-y)
            result = torch.empty_like(result).index_copy(0, first, u).index_copy(0, second, v)
            if return_stages:
                stages.append(result)
        if normalized:
            # For integer inputs return the exact quarter-valued Float64 result,
            # rather than silently invoking int-to-FP32 true division or floor.
            result = (result if result.is_floating_point() else result.double())/4
        return (result, stages) if return_stages else result

    def forward_factors(self, value, *, normalized=True, return_stages=False):
        return self._pass(value, normalized=normalized, return_stages=return_stages)

    def reverse_factors(self, value, *, normalized=True, return_stages=False):
        return self._pass(value, reverse=True, normalized=normalized, return_stages=return_stages)

    def forward(self, value):
        return self.forward_factors(value)

    def reverse(self, canonical_q):
        return self.reverse_factors(canonical_q)

    def dense_matrix(self, *, normalized=True, inverse=False):
        identity = torch.eye(10, dtype=self.sign_logits.dtype, device=self.sign_logits.device)
        return self._pass(identity, reverse=inverse, normalized=normalized)

    def specification(self):
        return dict(T=10, layers=4, matchings=MATCHINGS, hard_signs=self.signs().detach().cpu().tolist(),
            learnable_signs=self.sign_logits.requires_grad, sign_parameters=20 if self.sign_logits.requires_grad else 0,
            canonical_state='Q=B I, B=F/4; readout gains/permutations are outside B',
            inverse='B^T=F^T/4; reverse factors use transposed pairs in reversed layer order',
            per_pass_add_subtracts=40, conditional_sign_negations=20,
            integer_guard_bits_per_raw_pass=4, final_scale='divide4 only after all factors',
            scope='Arithmetic structure only; no free permutation wiring, port, state, rounding or cycle assertion.')


class TemporalReadout(nn.Module):
    """Instantiate separately for source and consumer; neither changes B.

    membrane(Q) is differentiable and can feed the caller's native surrogate.
    forward(Q) defaults to the exact hard theta-valued output. An optional
    activation(membrane, theta) callback can retain that native training gate;
    no alternative surrogate or threshold update rule is introduced here.
    """
    def __init__(self, row_gain=None, row_permutation=None, bias=None, theta=1., center=0.,
                 *, trainable=True, dtype=torch.float32, device=None):
        super().__init__()
        gain = torch.ones(10, dtype=dtype, device=device) if row_gain is None else torch.as_tensor(row_gain, dtype=dtype, device=device).reshape(10)
        bias = torch.zeros(10, dtype=dtype, device=device) if bias is None else torch.as_tensor(bias, dtype=dtype, device=device).reshape(-1).expand(10)
        permutation = torch.arange(10, device=device) if row_permutation is None else torch.as_tensor(row_permutation, dtype=torch.long, device=device)
        self.row_gain = nn.Parameter(gain.clone(), requires_grad=trainable)
        self.bias = nn.Parameter(bias.clone(), requires_grad=trainable)
        self.register_buffer('row_permutation', permutation.reshape(10).clone())
        self.register_buffer('theta', torch.as_tensor(theta, dtype=dtype, device=device).clone())
        self.register_buffer('center', torch.as_tensor(center, dtype=dtype, device=device).reshape(-1).expand(10).clone())

    def coordinates(self, canonical_q):
        shape = (10,)+((1,)*(canonical_q.ndim-1))
        return canonical_q.index_select(0, self.row_permutation)*self.row_gain.to(canonical_q).reshape(shape)

    def membrane(self, canonical_q):
        shape = (10,)+((1,)*(canonical_q.ndim-1))
        return self.coordinates(canonical_q)+self.bias.to(canonical_q).reshape(shape)-self.center.to(canonical_q).reshape(shape)

    def forward(self, canonical_q, activation=None):
        value = self.membrane(canonical_q)
        theta = self.theta.to(value)
        if theta.numel() != 1:
            theta = theta.reshape(10, *((1,)*(value.ndim-1)))
        return activation(value, theta) if activation is not None else (value >= theta).to(value.dtype)*theta


def exact_row_assignment(cost):
    """Exact one-to-one assignment; ten rows use1024 subset states."""
    cost = np.asarray(cost, np.float64)
    n = cost.shape[0]
    best = np.full(1 << n, np.inf)
    previous, choice = np.full(1 << n, -1, np.int64), np.full(1 << n, -1, np.int64)
    best[0] = 0.
    for mask in range((1 << n)-1):
        row = mask.bit_count()
        for column in range(n):
            if mask & (1 << column):
                continue
            nxt = mask | (1 << column)
            value = best[mask]+cost[row, column]
            if value < best[nxt]:
                best[nxt], previous[nxt], choice[nxt] = value, mask, column
    permutation = np.empty(n, np.int64)
    mask = (1 << n)-1
    for row in range(n-1, -1, -1):
        permutation[row] = choice[mask]
        mask = int(previous[mask])
    return permutation, float(best[-1])


def fit_readout(mean, covariance, teacher_A, teacher_bias, basis_matrix, row_permutation=None):
    """Float64 centered scalar LS plus exact row assignment, from given moments.

    The caller supplies the moments/domain and fixed B. This routine does not
    read data, search signs or consult validation. If P is supplied it only fits
    those pairs; otherwise it solves the ten-row assignment once. A zero-variance
    canonical row gets gain0 and a constant mean-matching bias. Teacher/local
    center and theta stay unchanged, so they need not enter the MSE fit.
    """
    def array(value):
        return value.detach().double().cpu().numpy() if torch.is_tensor(value) else np.asarray(value, np.float64)
    mean, covariance = array(mean).reshape(10), array(covariance)
    teacher, bias, basis = array(teacher_A), array(teacher_bias).reshape(10), array(basis_matrix)
    mu_q, mu_y = basis@mean, teacher@mean+bias
    variance_q = np.maximum(np.einsum('ti,ij,tj->t', basis, covariance, basis), 0.)
    variance_y = np.maximum(np.einsum('ti,ij,tj->t', teacher, covariance, teacher), 0.)
    cross = teacher@covariance@basis.T
    gains = np.divide(cross, variance_q[None], out=np.zeros_like(cross), where=variance_q[None] > 0)
    cost = np.maximum(variance_y[:, None]+gains*gains*variance_q[None]-2*gains*cross, 0.)
    if row_permutation is None:
        permutation, total = exact_row_assignment(cost)
        selection = 'One exact1024-state assignment on supplied scalar-LS errors; no sign search.'
    else:
        permutation = np.asarray(row_permutation, np.int64).reshape(10)
        total = float(cost[np.arange(10), permutation].sum())
        selection = 'Given fixed permutation; only scalar gain/bias fitting.'
    gain = gains[np.arange(10), permutation]
    fitted_bias = mu_y-gain*mu_q[permutation]
    fitted_A = gain[:, None]*basis[permutation]
    return dict(row_gain=gain, row_permutation=permutation, bias=fitted_bias,
        A=fitted_A, per_row_MSE=cost[np.arange(10), permutation], sum_MSE=total,
        mean_error=teacher@mean+bias-(fitted_A@mean+fitted_bias),
        cost_matrix=cost, zero_variance_basis_rows=variance_q == 0,
        rule=selection, scope='Moment-domain least squares only; no gate/AEE guarantee. Source and consumer must each fit their own teacher and keep their own bias/center/theta.')
