"""Inference-only whole-spatial-tile omission for the pinned native v2 MLP.

This is a numerical/execution probe, not a GPU speed implementation or a learned
gate. The outer native block adds its residual u; this module returns ONLY the
MLP branch. Removed blocks return zero, so the original block emits u.
"""
import contextlib
import math
import torch


def require_frozen(module):
    for name, layer in module.named_modules():
        if layer.training:
            raise ValueError('Inference-only contract: training module ' + name)
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
            if not layer.track_running_stats or layer.running_mean is None or layer.running_var is None:
                raise ValueError('Spatial current-stat BN cannot be skipped: ' + name)
            if not torch.isfinite(layer.running_mean).all() or not torch.isfinite(layer.running_var).all():
                raise ValueError('Invalid frozen BN statistics: ' + name)
            if (layer.running_var < 0).any():
                raise ValueError('Negative frozen BN variance: ' + name)


def pattern_mask(name, height, width, tile=9, device='cpu'):
    ny, nx = math.ceil(height/tile), math.ceil(width/tile)
    y, x = torch.meshgrid(torch.arange(ny, device=device), torch.arange(nx, device=device), indexing='ij')
    if name == 'all_keep': return torch.ones((ny, nx), dtype=torch.bool, device=device)
    if name == 'all_skip': return torch.zeros((ny, nx), dtype=torch.bool, device=device)
    if name == 'checkerboard': return (y+x) % 2 == 0
    if name == 'last_edge_only': return (y == ny-1) & (x == nx-1)
    raise ValueError('Unknown diagnostic mask: ' + name)


class TileMLP(torch.nn.Module):
    """Use whole 9x9 spatial regions across all T=10, B=1 and all channels.

mode=dense_masked always executes the full native producer then masks its output.
mode=actual_skip calls the native producer only for retained whole spatial tiles.
Each observed FC row is counted using actual forward hooks, not a mask estimate.
"""
    def __init__(self, native, name, mode, policy, tile=9):
        super().__init__()
        if native.__class__.__name__ != 'MS_Spiking_Mlp' or native.norm_layer != 'BN':
            raise ValueError('Only pinned MS_Spiking_Mlp is supported')
        if mode not in ('dense_masked', 'actual_skip') or tile != 9:
            raise ValueError('Unsupported execution mode/tile contract')
        for sn in (native.sn1.spiking_neuron, native.sn2.spiking_neuron):
            if sn.__class__.__name__ != 'PSN' or sn.T != 10:
                raise ValueError('Only stateless complete-T10 PSN is supported')
        require_frozen(native)
        self.native, self.layer_name, self.mode = native, name, mode
        self.policy, self.tile = policy, tile
        self.records = []
        self.eval()

    def forward(self, x):
        if torch.is_grad_enabled():
            raise ValueError('Tile probe requires inference_mode/no_grad; no recovery-training claim')
        require_frozen(self.native)
        if x.ndim != 5 or x.shape[0] != 10 or x.shape[1] != 1:
            raise ValueError('Expected complete [10,1,H,W,C] MLP input')
        t, b, h, w, c = x.shape
        keep = self.policy(self.layer_name, x)
        if not isinstance(keep, torch.Tensor) or keep.dtype != torch.bool or tuple(keep.shape) != (math.ceil(h/9), math.ceil(w/9)):
            raise ValueError('Policy must return a boolean whole-tile grid')
        keep = keep.to(device=x.device)
        decisions = keep.detach().cpu().tolist()
        expanded = keep.repeat_interleave(9, 0).repeat_interleave(9, 1)[:h, :w]
        kept_pixels = int(expanded.sum().item())
        observed = {'fc1_rows': 0, 'fc2_rows': 0, 'native_mlp_calls': 0}
        handles = []
        for key in ('fc1', 'fc2'):
            def count(layer, inputs, label=key):
                observed[label+'_rows'] += inputs[0].numel() // inputs[0].shape[-1]
            handles.append(getattr(self.native, key).register_forward_pre_hook(count))
        try:
            if self.mode == 'dense_masked':
                out = self.native(x) * expanded[None, None, :, :, None]
                observed['native_mlp_calls'] = 1
            elif kept_pixels == h*w:
                # Preserve native dense arithmetic/shape for the all-keep control.
                out = self.native(x)
                observed['native_mlp_calls'] = 1
            else:
                out = torch.zeros_like(x)
                for gy, row in enumerate(decisions):
                    for gx, accept in enumerate(row):
                        if not accept: continue
                        ys, xs = slice(9*gy, min(9*(gy+1), h)), slice(9*gx, min(9*(gx+1), w))
                        out[:, :, ys, xs, :] = self.native(x[:, :, ys, xs, :].contiguous())
                        observed['native_mlp_calls'] += 1
        finally:
            for handle in handles: handle.remove()
        expected = t*(h*w if self.mode == 'dense_masked' else kept_pixels)
        if observed['fc1_rows'] != expected or observed['fc2_rows'] != expected:
            raise RuntimeError('Actual FC execution does not match declared whole-tile contract')
        self.records.append({'layer_name': self.layer_name, 'shape_TBHWC': list(x.shape),
            'hidden_channels': self.native.fc1.out_features, 'tile_hw': [9, 9],
            'spans_complete_T': True, 'mode': self.mode, 'keep': decisions,
            'kept_real_pixels': kept_pixels, 'dense_fc_rows_per_layer': t*h*w,
            'candidate_cancelled_fc_rows_per_layer': t*(h*w-kept_pixels),
            'observed': observed, 'evidence': 'executed FC rows, not cycles or traffic'})
        return out


@contextlib.contextmanager
def tiled_mlp_model(model, mode, policy, selected_names=None):
    """Temporarily wrap selected native blocks, then restore original objects.

The entire model must be frozen/eval, including attention and decoder BN. This
prevents claiming whole-network equivalence while another spatial norm changes.
"""
    require_frozen(model)
    targets = [(n, m) for n, m in model.named_modules() if m.__class__.__name__ == 'MS_Spiking_Mlp']
    if selected_names is not None:
        wanted = set(selected_names)
        if not wanted.issubset({n for n, m in targets}):
            raise ValueError('Unknown selected MLP name')
        targets = [(n, m) for n, m in targets if n in wanted]
    if not targets: raise ValueError('No native MLP targets')
    replacements, wrappers = [], []
    try:
        for name, original in targets:
            parent_name, child_name = name.rsplit('.', 1)
            parent = model.get_submodule(parent_name)
            if getattr(parent, 'cnf', None) != 'ADD' or getattr(parent, 'use_checkpoint', False):
                raise ValueError('Native outer block must use ADD residual and no activation checkpoint')
            wrapped = TileMLP(original, name, mode, policy)
            replacements.append((parent, child_name, original))
            setattr(parent, child_name, wrapped)
            wrappers.append(wrapped)
        yield wrappers
    finally:
        for parent, child_name, original in reversed(replacements):
            setattr(parent, child_name, original)
