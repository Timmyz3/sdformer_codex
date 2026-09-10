"""Four fixed T10 matching layers with forty continuous lifting coefficients.

Each pair executes y0=x0+a*x1, then y1=x1+b*y0. Its inverse executes
x1=y1-b*y0, then x0=y0-a*x1; reverse the four layer orders as well. Both
directions use the same live parameters and ordinary autograd, with no detach,
discrete-sign STE, dense forward or runtime matrix inverse. There is no /4
normalization. Every real pair has determinant1, but B is not orthogonal and
its condition number and intermediate ranges are not bounded by that fact.

The FP32 implementation defines a new numeric function. Finite precision can
prevent exact round trips, particularly for ill-conditioned fitted B. No
integer lifting/floor, saturation, guard-bit choice or quantization is applied
here. Exported real linear L1 bounds are bounds per unit input infinity norm,
not measured activation ranges or proof of a deployable fixed-width inverse.
"""
from __future__ import annotations

import torch
from torch import nn

from fast_temporal_basis import MATCHINGS


class LiftingTemporalBasis(nn.Module):
    def __init__(self, coefficients, *, dtype=torch.float32, device=None):
        super().__init__()
        values = torch.as_tensor(coefficients, device=device, dtype=dtype).reshape(4, 5, 2)
        self.lifting = nn.Parameter(values.clone())
        self.register_buffer('matchings', torch.tensor(MATCHINGS, dtype=torch.long, device=device))

    def _pass(self, value, reverse=False, return_stages=False):
        if value.shape[0] != 10:
            raise ValueError('The fixed lifting basis has leading dimension T10.')
        result = value if value.is_floating_point() else value.to(self.lifting.dtype)
        coefficients = self.lifting.to(dtype=result.dtype)
        stages = []
        for layer in (range(3, -1, -1) if reverse else range(4)):
            first, second = self.matchings[layer, :, 0], self.matchings[layer, :, 1]
            x0, x1 = result.index_select(0, first), result.index_select(0, second)
            a = coefficients[layer, :, 0].reshape(5, *([1]*(result.ndim-1)))
            b = coefficients[layer, :, 1].reshape(5, *([1]*(result.ndim-1)))
            if reverse:
                recovered1 = x1-b*x0
                recovered0 = x0-a*recovered1
                half = result.index_copy(0, second, recovered1)
                result = half.index_copy(0, first, recovered0)
            else:
                y0 = x0+a*x1
                y1 = x1+b*y0
                half = result.index_copy(0, first, y0)
                result = half.index_copy(0, second, y1)
            if return_stages:
                stages.extend((half, result))
        return (result, stages) if return_stages else result

    def forward_factors(self, value, *, return_stages=False):
        return self._pass(value, return_stages=return_stages)

    def reverse_factors(self, value, *, return_stages=False):
        return self._pass(value, reverse=True, return_stages=return_stages)

    def forward(self, value):
        return self.forward_factors(value)

    def reverse(self, canonical_q):
        return self.reverse_factors(canonical_q)

    def dense_matrix(self, *, inverse=False, dtype=None):
        # Reference/export only. Actual data forward never calls this method.
        eye = torch.eye(10, dtype=dtype or self.lifting.dtype, device=self.lifting.device)
        return self._pass(eye, reverse=inverse)

    @torch.no_grad()
    def conditioning(self):
        # Current coefficient values are promoted to F64; not an empirical
        # peak measurement and not a claim of FP32-dense execution equality.
        eye = torch.eye(10, dtype=torch.float64, device=self.lifting.device)
        basis, forward = self._pass(eye, return_stages=True)
        inverse, backward = self._pass(eye, reverse=True, return_stages=True)
        forward_bounds = [float(m.abs().sum(1).max()) for m in forward]
        inverse_bounds = [float(m.abs().sum(1).max()) for m in backward]
        return dict(basis_rank=int(torch.linalg.matrix_rank(basis)),
            basis_condition_number=float(torch.linalg.cond(basis)),
            basis_inverse_product_max_abs=float((inverse@basis-eye).abs().max()),
            basis_max_row_L1=float(basis.abs().sum(1).max()),
            inverse_basis_max_row_L1=float(inverse.abs().sum(1).max()),
            forward_halfstage_max_row_L1=forward_bounds,
            inverse_halfstage_max_row_L1=inverse_bounds,
            forward_intermediate_bound_per_unit_Linf=max([1.]+forward_bounds),
            inverse_intermediate_bound_per_unit_Linf=max([1.]+inverse_bounds),
            coefficient_max_abs=float(self.lifting.abs().max()),
            scope='Real linear row-L1 bounds for unit input infinity norm using current coefficients promoted to F64. Not observed activation ranges or a guard/quantization guarantee.')

    def specification(self):
        return dict(kind='lifting40', T=10, layers=4, matchings=MATCHINGS,
            coefficient_parameters=40, per_pass_products=40, per_pass_add_subtracts=40,
            canonical_state='Q=B I; B excludes both readout gains/permutations',
            forward='y0=x0+a*x1; y1=x1+b*y0',
            inverse='Reverse layers: x1=y1-b*y0; x0=y0-a*x1; same live a/b, no dense inverse',
            normalization='None; B is not orthogonal',
            derivative='Ordinary autograd through both lifting and inverse arithmetic, no detach/STE',
            numeric='FP32 multiply then add/subtract at each lifting step; F64 only for caller-requested references/conditioning.',
            scope='Arithmetic structure only. Condition, quantization, state widths, ports and complete-network AEE need separate measurement.')
