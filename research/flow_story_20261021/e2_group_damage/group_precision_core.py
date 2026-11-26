"""E2 inference-only Q6 background / complete-tile Q4 actions / Q8 true tails.

Numeric source: frozen R30 per-output maxabs/127 RNE q8 and integer plane
primitive. This file does not train, load a checkpoint, read GT, select a
family, or start CUDA. Every selected action omits planes in the producer.
"""
import contextlib
import hashlib
import torch
from torch import nn
from torch.nn import functional as F

PREFIX = 'sttmultires_unet.encoders.swin3d.'
LINEARS = tuple(PREFIX + f'layers.{s}.swin_blocks.{b}.mlp.{fc}'
                for s, depth in enumerate((2, 2, 6, 2))
                for b in range(depth) for fc in ('fc1', 'fc2'))
CONVS = tuple(PREFIX + f'patch_embed.residual_encoding.resblocks.{b}.conv{j}.0'
              for b in (0, 1) for j in (1, 2))
TARGETS = LINEARS + CONVS
MAX_ROWS_PER_KERNEL = 65536
MAX_OUTPUT_ELEMENTS = 8 * 1024 * 1024


def require(ok, why):
    if not ok:
        raise ValueError(why)


def digest(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def quantize(weight):
    w = weight.detach().reshape(weight.shape[0], -1).cpu()
    require(w.dtype == torch.float32 and bool(torch.isfinite(w).all()),
            'Finite FP32 master weights')
    scale = w.abs().amax(1) / 127.
    divisor = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(w / divisor[:, None]).clamp(-127, 127).to(torch.int8)
    return q.to(weight.device), scale.to(weight.device)


def row_domains(t, b, h, w, device, selected):
    """selected is a CPU HxW bool mask of complete output9x9 tiles."""
    require(t == 10 and b >= 1 and h > 0 and w > 0, 'Full T10/native spatial graph')
    require(selected is not None and selected.dtype == torch.bool
            and selected.device.type == 'cpu' and tuple(selected.shape) == (h, w),
            'Explicit complete-tile CPU mask, including empty background mask')
    hh, ww = h // 9 * 9, w // 9 * 9
    require(not bool(selected[hh:, :].any()) and not bool(selected[:, ww:].any()),
            'True tails cannot become Q4')
    counts = selected[:hh, :ww].reshape(h // 9, 9, w // 9, 9).permute(0, 2, 1, 3).sum((2, 3))
    require(bool(((counts == 0) | (counts == 81)).all()), 'Whole output9x9 tiles only')
    tail2d = torch.ones(h, w, dtype=torch.bool, device=device)
    tail2d[:hh, :ww] = False
    tail = tail2d[None, None].expand(t, b, h, w).reshape(-1)
    q4 = selected.to(device)[None, None].expand(t, b, h, w).reshape(-1)
    q6 = ~tail & ~q4
    require(not bool((tail & q4).any()), 'Domains mutually exclusive')
    return tail, q6, q4, dict(
        geometric_shape=[t, b, h, w], row_order='T,B,output_y,output_x',
        selected_complete_tiles=torch.nonzero(counts == 81).tolist(),
        selected_spatial_mask_sha256=digest(selected),
        tail_rows=int(tail.sum()), Q6_rows=int(q6.sum()), Q4_rows=int(q4.sum()),
        tail_rows_sha256=digest(tail), Q6_rows_sha256=digest(q6), Q4_rows_sha256=digest(q4))


def kernel_mm(a, b):
    require(a.ndim == b.ndim == 2 and a.dtype == b.dtype == torch.int8
            and a.shape[1] == b.shape[0], 'Integer operand contract')
    n = len(a)
    padding = 0
    if a.device.type == 'cuda':
        require(a.shape[1] % 8 == 0 and b.shape[1] % 8 == 0, 'Native K/N multiple8')
        issued = max(24, ((n + 7) // 8) * 8)
        padding = issued - n
        if padding:
            a = F.pad(a, (0, 0, 0, padding))
    output = torch._int_mm(a.contiguous(), b.contiguous())
    require(output.dtype == torch.int32 and output.shape == (n + padding, b.shape[1]),
            'True int32 output')
    return output[:n], padding


def integer_affine(x, q, scale, tail, q6, q4, *, reference=False):
    require(not torch.is_grad_enabled(), 'E2 is inference-only')
    require(x.ndim == 2 and x.dtype == torch.float32 and bool(torch.isfinite(x).all())
            and bool(((x == 0) | (x == 1)).all()), 'Actual binary FP32 activation')
    require(q.dtype == torch.int8 and q.ndim == 2 and x.shape[1] == q.shape[1]
            and scale.dtype == torch.float32 and tuple(scale.shape) == (q.shape[0],),
            'Single shared q8/scale')
    require(all(a.dtype == torch.bool and a.shape == (len(x),) for a in (tail, q6, q4)),
            'Actual row-domain masks')
    require(bool((tail | q6 | q4).all()) and not bool(((tail & q6) | (tail & q4) | (q6 & q4)).any()),
            'Exclusive exhaustive Q8/Q6/Q4 domains')
    binary = x.to(torch.int8)
    limit = max(24, min(MAX_ROWS_PER_KERNEL, MAX_OUTPUT_ELEMENTS // q.shape[0]))
    limit = limit // 8 * 8
    integer = torch.zeros(len(x), q.shape[0], device=x.device, dtype=torch.int32)
    calls = []
    if reference:
        for label, mask, drop in (('tail8', tail, 0), ('complete6', q6, 2), ('selected4', q4, 4)):
            indices = torch.nonzero(mask).flatten()
            effective = ((q.to(torch.int16) >> drop) << drop).to(torch.int8)
            for start in range(0, len(indices), limit):
                ids = indices[start:start + limit]
                part, pad = kernel_mm(binary.index_select(0, ids), effective.T)
                integer.index_copy_(0, ids, part)
                calls.append(dict(kind='dense_integer_reference', bit=None, scope=label,
                                  rows=len(ids), padding_rows=pad, coefficient=None))
    else:
        tail_indices = torch.nonzero(tail).flatten()
        non_q4_indices = torch.nonzero(~q4).flatten()
        for bit in range(8):
            indices = tail_indices if bit < 2 else non_q4_indices if bit < 4 else None
            scope = 'tail8' if bit < 2 else 'non_q4' if bit < 4 else 'all'
            n = len(indices) if indices is not None else len(x)
            plane = ((q.to(torch.int16) >> bit) & 1).to(torch.int8).T.contiguous()
            coefficient = -128 if bit == 7 else 1 << bit
            for start in range(0, n, limit):
                ids = indices[start:start + limit] if indices is not None else None
                operand = binary.index_select(0, ids) if ids is not None else binary[start:start + limit]
                part, pad = kernel_mm(operand, plane)
                part *= coefficient
                if ids is None:
                    integer[start:start + len(part)].add_(part)
                else:
                    integer.index_add_(0, ids, part)
                calls.append(dict(kind='binary_weight_plane', bit=bit, scope=scope,
                                  rows=len(part), padding_rows=pad, coefficient=coefficient))
    audit = dict(
        mode='Q6_background_Q4_selected_Q8_true_tail', reference=reference,
        rows=len(x), Cin=q.shape[1], Cout=q.shape[0],
        tail_rows=int(tail.sum()), Q6_rows=int(q6.sum()), Q4_rows=int(q4.sum()),
        complete_rows=int((~tail).sum()), binary_verified=True, int32_accumulation=True,
        weight_q8_sha256=digest(q), weight_scale_sha256=digest(scale),
        logical_row_plane_work=sum(z['rows'] for z in calls) if not reference else None,
        actual_integer_kernel_calls=len(calls), explicit_padded_rows=sum(z['padding_rows'] for z in calls),
        kernel_calls=calls, scale_applications=1, shared_q8_scale=True,
        low_bit_rule='tail_drop0_complete_drop2_selected_complete_drop4',
        actual_low2_plane_cancellation=True, canonical_scatter_before_BN=True,
        spatial_rows_retained=True, full_T_and_channels=True,
        software_plane_dtype='int8', quantization_device='cpu', cycle_claim=False)
    return integer.float() * scale, integer, audit


class GroupAffine(nn.Module):
    def __init__(self, source, kind, selected, *, reference=False, expected_geometry=None):
        super().__init__()
        self.weight, self.bias = source.weight, source.bias
        self.kind, self.selected, self.reference = kind, selected, reference
        self.expected_geometry = expected_geometry
        self.capture_integer = False
        self.last_integer = self.last_audit = None
        if kind == 'linear':
            self.in_features, self.out_features = source.in_features, source.out_features
        else:
            require(tuple(source.kernel_size) == (3, 3) and tuple(source.stride) == (1, 1)
                    and tuple(source.padding) == (1, 1) and tuple(source.dilation) == (1, 1)
                    and source.groups == 1, 'Only original patch stride1/pad1 conv')
            self.in_channels, self.out_channels = source.in_channels, source.out_channels

    def forward(self, x):
        require(not torch.is_grad_enabled(), 'No E2 STE or training')
        q, scale = quantize(self.weight)
        if self.kind == 'linear':
            t, b, h, w, cin = x.shape
            rows = x.reshape(-1, cin)
        else:
            t, b, cin, h, w = x.shape
            rows = F.unfold(x.reshape(t*b, cin, h, w), 3, padding=1).transpose(1, 2).reshape(-1, cin*9)
        require(b == 1, 'One identity per actual current-BN forward; no cross-identity batching')
        if self.expected_geometry is not None:
            g = self.expected_geometry
            require((t, h, w, rows.shape[1], q.shape[0]) ==
                    (g['T'], g['H'], g['W'], g['K'], g['Cout']),
                    'Actual affine geometry matches selected macro-action contract')
        tail, q6, q4, dom = row_domains(t, b, h, w, x.device, self.selected)
        y, integer, audit = integer_affine(rows, q, scale, tail, q6, q4, reference=self.reference)
        audit.update(dom)
        audit['actual_input_shape'] = list(x.shape)
        if self.kind == 'conv':
            audit.update(operation='Conv2d', kernel=[3, 3], stride=[1, 1], padding=[1, 1],
                         groups=1, dilation=[1, 1], unfold_elements=rows.numel(),
                         padded_zero_input_taps_included=True)
        if self.bias is not None:
            y = y + self.bias
        self.last_integer = integer.detach() if self.capture_integer else None
        self.last_audit = audit
        y = y.reshape(t, b, h, w, q.shape[0])
        return y if self.kind == 'linear' else y.permute(0, 1, 4, 2, 3).contiguous()


@contextlib.contextmanager
def installed(model, mode, *, masks, reference=False, ste=False, geometries=None):
    require(mode in ('Q6', 'Q6_Q4'), 'Only the fixed Q6 background and Q4 actions')
    require(not ste, 'Inference-only integer graph; forward additionally requires no_grad')
    require(set(masks) == set(TARGETS), 'All28 explicit masks; unselected affines remain Q6')
    if mode == 'Q6':
        require(all(not bool(mask.any()) for mask in masks.values()), 'Background has no selected Q4 rows')
    if geometries is not None:
        require(set(geometries) == set(TARGETS), 'Actual full geometry contract')
    originals = {name: model.get_submodule(name) for name in TARGETS}
    require(all(m.bias is None for m in originals.values()), 'Exactly28 original no-bias affines')
    wrappers = {}
    try:
        for name, source in originals.items():
            wrap = GroupAffine(source, 'linear' if name in LINEARS else 'conv', masks[name],
                               reference=reference, expected_geometry=geometries[name] if geometries else None)
            parent, attr = name.rsplit('.', 1)
            setattr(model.get_submodule(parent), attr, wrap)
            wrappers[name] = wrap
        yield wrappers
    finally:
        for name, source in originals.items():
            parent, attr = name.rsplit('.', 1)
            setattr(model.get_submodule(parent), attr, source)
        require(all(model.get_submodule(n) is m for n, m in originals.items()), 'Original affine objects restored')
