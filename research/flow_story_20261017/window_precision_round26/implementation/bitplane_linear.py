"""R26 integer weight bitplanes on binary spikes, full spatial/T/channel axes.

No spatial compact, floating GEMM fallback, correction offset, training or GT.
CUDA uses the same torch._int_mm int8 x int8 -> int32 operator as CPU.
"""
import hashlib
import torch
from torch import nn

MODES = ('Q8ref', 'Q8serial', 'Q6', 'Q4')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def quantize_source(linear):
    """Once per checkpoint, explicit CPU FP32 per-output maxabs/127 quantization."""
    w = linear.weight.detach().cpu()
    require(w.dtype == torch.float32 and w.ndim == 2 and bool(torch.isfinite(w).all()), 'Finite FP32 original linear weight')
    scale = w.abs().amax(1)/127.0
    denominator = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(w/denominator[:, None]).clamp(-127, 127).to(torch.int8)
    bias = None if linear.bias is None else linear.bias.detach().cpu().clone()
    if bias is not None:
        require(bias.dtype == torch.float32 and bool(torch.isfinite(bias).all()), 'Finite original FP32 bias')
    return {'q8': q, 'scale': scale, 'bias': bias,
            'source_weight_sha256': digest(w), 'source_bias_sha256': None if bias is None else digest(bias),
            'q8_sha256': digest(q), 'scale_sha256': digest(scale),
            'zero_output_channels': int((scale == 0).sum()), 'source_quantization_device': 'cpu',
            'rounding': 'torch.round_nearest_ties_to_even', 'quantized_range': [-127, 127],
            'scale_rule': 'output_channel_max_abs_div_127; zero_channel_scale0_q0',
            'low_precision_rule': 'signed_arithmetic_right_shift_then_left_shift_no_reclamp_no_correction'}


def place_spec(spec, device):
    q = spec['q8'].to(device=device)
    out = dict(spec, q8=q, scale=spec['scale'].to(device=device),
               bias=None if spec['bias'] is None else spec['bias'].to(device=device))
    # int8 is the actual software carrier, not a one-bit-packed storage claim.
    out['q8_transpose'] = q.T.contiguous()
    out['planes'] = [((q.to(torch.int16) >> bit) & 1).to(torch.int8).T.contiguous() for bit in range(8)]
    return out


def full_precision_rows(shape, device):
    require(len(shape) == 5 and shape[0] == 10 and shape[2:4] == (30, 40), 'Complete T10/B/30x40/channel geometry')
    mask = torch.ones((30, 40), dtype=torch.bool, device=device)
    mask[:27, :36] = False
    return mask[None, None].expand(shape[0], shape[1], 30, 40).reshape(-1).nonzero().flatten()


def integer_mm(a, b):
    require(a.dtype == b.dtype == torch.int8 and a.ndim == b.ndim == 2 and
            a.shape[1] == b.shape[0] and a.device == b.device, 'Actual int8 contraction')
    if a.device.type == 'cuda':
        require(a.shape[0] > 16 and a.shape[1] % 8 == 0 and b.shape[1] % 8 == 0,
                'PyTorch2.5 CUDA int_mm native shape; no user padding')
    result = torch._int_mm(a.contiguous(), b.contiguous())
    require(result.dtype == torch.int32 and result.shape == (a.shape[0], b.shape[1]), 'Actual int32 output')
    return result


class QuantLinear(nn.Module):
    def __init__(self, source, spec, mode):
        super().__init__()
        require(mode in MODES, 'One fixed quantized mode')
        self.source, self.spec, self.mode = source, spec, mode
        self.in_features, self.out_features = source.in_features, source.out_features
        self.last_audit, self.last_integer = None, None

    def forward(self, spikes):
        require(spikes.dtype == torch.float32 and spikes.shape[-1] == self.in_features
                and bool(torch.isfinite(spikes).all()) and bool(((spikes == 0) | (spikes == 1)).all()),
                'Only finite binary full-precision spikes; no activation quantization')
        shape = tuple(spikes.shape)
        tail = full_precision_rows(shape, spikes.device)
        x = spikes.reshape(-1, self.in_features).to(torch.int8).contiguous()
        require(x.shape[0] == 12000*shape[1] and len(tail) == 2280*shape[1], 'Actual full/tail row inventory')
        q, scale, bias = (self.spec[k] for k in ('q8', 'scale', 'bias'))
        require(q.device == spikes.device and scale.dtype == torch.float32, 'Prepared same-device quantization')
        records = []
        if self.mode == 'Q8ref':
            integer = integer_mm(x, self.spec['q8_transpose'])
            records.append({'kind': 'integer_dense_reference', 'bit': None, 'coefficient': None,
                'rows': x.shape[0], 'Cin': self.in_features, 'Cout': self.out_features,
                'input_dtype': 'torch.int8', 'weight_dtype': 'torch.int8', 'output_dtype': 'torch.int32',
                'input_spike_ones': int(x.sum(dtype=torch.int64)), 'weight_shape': list(self.spec['q8_transpose'].shape),
                'explicit_padding_rows': 0, 'explicit_padding_channels': 0})
        else:
            drop = {'Q8serial': 0, 'Q6': 2, 'Q4': 4}[self.mode]
            integer = torch.zeros((len(x), self.out_features), dtype=torch.int32, device=x.device)
            tail_x = x.index_select(0, tail) if drop else None
            # Full signed two's complement; only low planes skip complete tiles.
            for bit in range(8):
                use_tail = bit < drop
                actual = tail_x if use_tail else x
                coefficient = -128 if bit == 7 else 1 << bit
                partial = integer_mm(actual, self.spec['planes'][bit])
                contribution = partial * coefficient
                if use_tail:
                    integer.index_add_(0, tail, contribution)
                else:
                    integer.add_(contribution)
                records.append({'kind': 'binary_weight_plane', 'bit': bit, 'coefficient': coefficient,
                    'row_scope': 'tail_only_Q8' if use_tail else 'all_spatial_rows',
                    'rows': len(actual), 'Cin': self.in_features, 'Cout': self.out_features,
                    'input_dtype': 'torch.int8', 'weight_dtype': 'torch.int8', 'output_dtype': 'torch.int32',
                    'input_spike_ones': int(actual.sum(dtype=torch.int64)), 'weight_shape': list(self.spec['planes'][bit].shape),
                    'explicit_padding_rows': 0, 'explicit_padding_channels': 0})
        # All integer accumulation completes before this one channel-wise scale.
        output = integer.to(torch.float32)*scale
        if bias is not None:
            output = output+bias
        require(bool(torch.isfinite(output).all()), 'Finite scaled affine output')
        self.last_integer = integer.reshape(*shape[:-1], self.out_features)
        self.last_audit = {'mode': self.mode, 'actual_input_shape': list(shape), 'input_rows': len(x),
            'full_tile_rows': 9720*shape[1], 'tail_rows': len(tail), 'original_linear_forward_calls': 0,
            'int_mm_calls': records, 'int_mm_call_count': len(records),
            'row_plane_work': sum(r['rows'] for r in records) if self.mode != 'Q8ref' else 0,
            'integer_dense_rows': len(x) if self.mode == 'Q8ref' else 0,
            'binary_input_verified': True, 'full_T_B_C_spatial_positions_retained': True,
            'scale_applications_per_output_element': 1, 'bias_applications_per_output_element': 0 if bias is None else 1,
            'q8_sha256': self.spec['q8_sha256'], 'scale_sha256': self.spec['scale_sha256'],
            'integer_output_shape': list(self.last_integer.shape), 'integer_output_sha256': digest(self.last_integer),
            'scaled_output_sha256': digest(output.reshape(*shape[:-1], self.out_features)),
            'software_plane_storage_dtype': 'torch.int8', 'user_padding': False,
            'internal_kernel_padding': 'not_observed_not_a_cycle_claim', 'float_matmul_fallback': False}
        return output.reshape(*shape[:-1], self.out_features)


def backend_shape_smoke(device):
    """Four explicitly budgeted operator tests, zero model/data forward."""
    records = []
    for cin, cout in ((384, 1536), (1536, 384)):
        for rows in (12000, 2280):
            a = torch.ones((rows, cin), dtype=torch.int8, device=device)
            b = ((torch.arange(cin*cout, device=device).reshape(cin, cout) % 3)-1).to(torch.int8)
            result = integer_mm(a, b)
            expected = b.to(torch.int32).sum(0, dtype=torch.int32)
            require(bool((result == expected[None]).all()), 'Backend shape/signed exact operator test')
            records.append({'rows': rows, 'Cin': cin, 'Cout': cout, 'input_dtype': str(a.dtype),
                'output_dtype': str(result.dtype), 'explicit_padding_rows': 0,
                'passed_exact_column_sum': True, 'model_forward_count': 0})
    return records
