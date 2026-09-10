"""Exact-range integer factor evaluation inside the existing dense network.

This is a distinct deployed student: folded U8, aligned dyadic V, Aq14 and
compiled thresholds. CUDA provides a numerical reference, not sparse speed.
The displayed Conv/BN tensors are ordinary affine views; the neuron consumes
the stored integer sums and the compiled BN/threshold, so it never applies
the displayed BN a second time.
"""
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F


class IntegerLatentPair:
    def __init__(self, arrays, device, conditional=True):
        self.arrays = arrays
        self.source_theta = float(arrays['theta_source'])
        self.theta = float(arrays['theta_output'])
        self.shared_rank = int(arrays['shared_rank'])
        self.conditional = conditional
        self.u = torch.as_tensor(arrays['u_int8'], device=device).float().T.reshape(-1, 96, 3, 3)
        self.v = torch.as_tensor(arrays['integer_v_align_coeff'], device=device).double()
        self.a = torch.as_tensor(arrays['integer_a_q14'], device=device).double()
        self.y_scale = torch.exp2(torch.as_tensor(arrays['integer_y_exponent'], device=device).double())
        self.full_threshold = torch.as_tensor(arrays['integer_full_threshold'], device=device).double()
        self.indices = [torch.as_tensor(arrays[f'integer_empty_indices_t{t}'], device=device).long() for t in range(10)]
        self.positive = [torch.as_tensor(arrays[f'integer_threshold_pos_t{t}'], device=device).double() for t in range(10)]
        self.negative = [torch.as_tensor(arrays[f'integer_threshold_neg_t{t}'], device=device).double() for t in range(10)]
        self.full_y = self.shared_y = self.empty = None
        self.last_counts = {}
        self.numeric_checks = {}
        self.table_metadata = dict(
            entries_per_row=[len(x) for x in self.positive],
            conservative_INT48_pair_bytes=sum(len(x) for x in self.positive)*96*12,
            scope='compiled source-empty positive/negative integer thresholds; physical cache and ports are separate')
        if 'integer_predictor_compact_metadata_json' in arrays:
            # The numerical adapter still indexes expanded rows. A constant
            # control need not physically replicate those identical entries.
            self.table_metadata.update(json.loads(str(arrays['integer_predictor_compact_metadata_json'])))

    def conv_forward(self, x):
        if x.shape[:3] != (10, 1, 96) or x.shape[-1] % 4:
            raise ValueError('Expected native T10/B1/C96 with complete horizontal P4 groups.')
        gate = x.ne(0)
        # Explicitly use the folded theta*U code, with the gate as operand.
        previous_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        try:
            z = F.conv2d(gate.flatten(0, 1).to(x.dtype), self.u, padding=1)
        finally:
            torch.backends.cudnn.allow_tf32 = previous_tf32
        if not self.numeric_checks:
            residual = (z-z.round()).abs().max()
            source_error = torch.where(gate, (x-self.source_theta).abs(), x.abs()).max()
            self.numeric_checks.update(source_theta_residual=float(source_error),
                Z_integer_residual=float(residual), Z_observed_min=float(z.min()), Z_observed_max=float(z.max()),
                arithmetic='FP32 exact-range integer gate*U8 convolution; FP64 aligned V and Aq14 integer sums')
            if residual != 0 or source_error != 0:
                raise ValueError('Input amplitude or exact integer convolution differs from the compiled student.')
        # The compiled static domains are below the exact-integer range of
        # FP64. No rounding between shared/tail, V and A is introduced.
        values = z.permute(0, 2, 3, 1).double()
        shared = values[..., :self.shared_rank] @ self.v[:self.shared_rank]
        tail = values[..., self.shared_rank:] @ self.v[self.shared_rank:]
        self.shared_y = shared.permute(0, 3, 1, 2).unsqueeze(1)
        self.full_y = (shared+tail).permute(0, 3, 1, 2).unsqueeze(1)
        activity = gate.any(2).float()
        self.empty = F.max_pool2d(activity, 3, stride=1, padding=1).eq(0)[:, 0]
        return (self.full_y*self.y_scale[None, None, :, None, None]).to(x.dtype)

    def forward_groups(self, full_y, shared_y, empty):
        """All group tensors use G,T,P,H and source-empty uses G,T,P."""
        full_u = torch.einsum('ts,gsph->gtph', self.a, full_y)
        full_gate = full_u >= self.full_threshold[None, :, None, :]
        if not self.conditional:
            return full_gate, torch.zeros_like(full_gate)
        shared_u = torch.einsum('ts,gsph->gtph', self.a, shared_y)
        positive, negative = [], []
        for t, indices in enumerate(self.indices):
            bits = empty[:, indices].permute(0, 2, 1).long()
            code = (bits*(1 << torch.arange(len(indices), device=bits.device))).sum(-1)
            positive.append(shared_u[:, t] >= self.positive[t][code])
            negative.append(shared_u[:, t] <= self.negative[t][code])
        pos, neg = torch.stack(positive, 1), torch.stack(negative, 1)
        accepted = pos | neg
        # In the zero-radius tie both predicates may hold: >= zero spikes.
        return torch.where(accepted, pos, full_gate), accepted

    def neuron_forward(self, displayed_bn_y):
        if self.full_y is None:
            raise RuntimeError('Integer factor state belongs to this Conv1 call.')
        t, batch, channels, height, width = self.full_y.shape
        output = torch.empty_like(displayed_bn_y)
        accepted_count = total = 0
        for first in range(0, height, 8):
            last = min(first+8, height)
            def group(values):
                return values[:, 0, :, first:last].reshape(10, 96, last-first, width//4, 4).permute(2, 3, 0, 4, 1).reshape(-1, 10, 4, 96)
            empty = self.empty[:, first:last].reshape(10, last-first, width//4, 4).permute(1, 2, 0, 3).reshape(-1, 10, 4)
            gate, accepted = self.forward_groups(group(self.full_y), group(self.shared_y), empty)
            value = (gate.to(output.dtype)*self.theta).reshape(last-first, width//4, 10, 4, 96)
            output[:, 0, :, first:last] = value.permute(2, 4, 0, 1, 3).reshape(10, 96, last-first, width)
            total += gate.numel()
            accepted_count += int(accepted.sum())
        self.last_counts = dict(gates=total, accepted=accepted_count, failed=total-accepted_count,
            numeric='folded U8/dyadic V/Aq14 integer comparisons; dense GPU numerical execution')
        self.full_y = self.shared_y = self.empty = None
        return output


def install_integer_factor(conv1, neuron, parameter_file, conditional=True):
    with np.load(Path(parameter_file)) as data:
        arrays = {key: data[key].copy() for key in data.files}
    pair = IntegerLatentPair(arrays, conv1.weight.device, conditional)
    originals = dict(conv1_forward=conv1.forward, neuron_forward=neuron.forward)
    conv1.forward, neuron.forward = pair.conv_forward, pair.neuron_forward
    return pair, originals
