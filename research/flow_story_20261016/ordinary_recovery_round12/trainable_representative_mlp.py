"""Differentiable fixed layouts; shared native parameters and complete-T tile paths.

No gate, teacher detach, hidden loss or BN update. Unlike the old inference
wrapper this path permits u and native parameters to require gradients.
"""
import contextlib
import math
import torch
from source_contract import TARGET, require

ARMS = ('native', 'FF', 'CC', 'fine_s2b2')


def is_selected_layer(name):
    return any('.layers.{}.'.format(s) in name for s in (0, 2))


def fine_layer(arm, name):
    return arm == 'FF' or (arm == 'fine_s2b2' and name == TARGET)


def validate_native(native):
    require(type(native).__name__ == 'MS_Spiking_Mlp' and native.norm_layer == 'BN', 'Native BN MLP required')
    require(all(not m.training for m in native.modules()), 'Eval/dropout-off native required')
    for sn in (native.sn1.spiking_neuron, native.sn2.spiking_neuron):
        require(type(sn).__name__ == 'PSN' and sn.T == 10 and tuple(sn.weight.shape) == (10, 10), 'Complete-T10 PSN required')
    for bn in (native.bn1.norm_layer, native.bn2.norm_layer):
        require(not bn.training and bn.track_running_stats and bn.running_mean is not None and bn.running_var is not None,
                'Private BN must keep fixed running statistics')
    require(native.fc1.in_features == native.fc2.out_features and native.fc1.out_features == native.fc2.in_features,
            'Native FC widths differ')


def spatial_mean(x):
    # Same scalar reductions, order and real-area division as frozen source.
    h, w = x.shape[2:4]
    spans = [[(y, min(y+3, h), z, min(z+3, w)) for z in range(0, w, 3)] for y in range(0, h, 3)]
    rows = []
    for row in spans:
        cols = [x[:, :, y0:y1, x0:x1, :].sum((2, 3))/((y1-y0)*(x1-x0)) for y0, y1, x0, x1 in row]
        rows.append(torch.stack(cols, dim=2))
    return torch.stack(rows, dim=2).contiguous(), spans


def repeat_branch(x, spans):
    # A gradient-preserving dense assembly; no detached CopySlices buffer.
    rows = []
    for gy, row in enumerate(spans):
        cols = [x[:, :, gy:gy+1, gx:gx+1, :].expand(-1, -1, y1-y0, x1-x0, -1)
                for gx, (y0, y1, x0, x1) in enumerate(row)]
        rows.append(torch.cat(cols, dim=3))
    return torch.cat(rows, dim=2).contiguous()


def tile_core(native, u, fine, mode='actual'):
    require(mode in ('actual', 'dual_reference'), 'Unknown producer mode')
    validate_native(native)
    require(u.ndim == 5 and tuple(u.shape[:2]) == (10, 1) and u.shape[-1] == native.fc1.in_features,
            'Complete T10/B1/native C required')
    require(u.dtype == native.fc1.weight.dtype and u.dtype in (torch.float32, torch.float64), 'Input dtype differs')
    require(not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('cuda'), 'No autocast')
    require(bool(torch.isfinite(u).all()), 'Nonfinite input')
    require(type(fine) is bool, 'Whole fixed layer decision required')
    rows, calls, fcrows = [], 0, 0
    for y in range(0, u.shape[2], 9):
        cols = []
        for x in range(0, u.shape[3], 9):
            local = u[:, :, y:y+9, x:x+9, :].contiguous()
            full = coarse = None
            if fine or mode == 'dual_reference':
                full = native(local); calls += 1; fcrows += local.numel()//local.shape[-1]
            if not fine or mode == 'dual_reference':
                pooled, spans = spatial_mean(local)
                representatives = native(pooled); calls += 1; fcrows += pooled.numel()//pooled.shape[-1]
                coarse = repeat_branch(representatives, spans)
            cols.append(full if fine else coarse)
        rows.append(torch.cat(cols, dim=3))
    out = torch.cat(rows, dim=2).contiguous()
    require(out.shape == u.shape and bool(torch.isfinite(out).all()), 'Invalid assembled branch')
    return out, {'mode': mode, 'shape_TBHWC': list(u.shape), 'fine_mask': [[fine]*math.ceil(u.shape[3]/9)
            for _ in range(math.ceil(u.shape[2]/9))], 'native_mlp_calls': calls,
            'executed_fc_rows_per_layer': fcrows, 'temporal_pooling': False, 'external_residual_added_here': False,
            'input_requires_grad': u.requires_grad, 'output_requires_grad': out.requires_grad,
            'scope': 'Actual differentiable producer service; no cycle/speedup claim'}


class TrainableRepresentativeMLP(torch.nn.Module):
    def __init__(self, native, fine, shape, mode):
        super().__init__(); self.native = native
        self.fine, self.shape, self.mode, self.records = fine, tuple(shape), mode, []
        self.eval()

    def forward(self, u):
        require(tuple(u.shape) == self.shape and not self.records, 'One declared-shape call per wrapper required')
        out, record = tile_core(self.native, u, self.fine, self.mode)
        self.records.append(record)
        return out


@contextlib.contextmanager
def wrapped(model, targets, shapes, arm, mode='actual'):
    require(arm in ARMS and all(not m.training for m in model.modules()), 'Fixed eval layout required')
    wrappers, originals = {}, []
    try:
        if arm != 'native':
            for name, native in targets.items():
                if not is_selected_layer(name): continue
                parent_name, child = name.rsplit('.', 1); parent = model.get_submodule(parent_name)
                require(getattr(parent, 'cnf', None) == 'ADD' and not getattr(parent, 'use_checkpoint', False),
                        'Unchanged external ADD/no-checkpoint block required')
                wrapper = TrainableRepresentativeMLP(native, fine_layer(arm, name), shapes[name], mode)
                originals.append((parent, child, native)); setattr(parent, child, wrapper); wrappers[name] = wrapper
        yield wrappers
    finally:
        for parent, child, native in reversed(originals): setattr(parent, child, native)
