"""Complete-T10, valid9x9 tile slow/learned-fast reference and actual execution.

Fast topology/weights are supplied by the pinned DynamicSwin adaptation. The
selected branch excludes the native block's external residual. No gate training
or whole-network recovery is implemented here.
"""
import contextlib
import hashlib
import math

import torch

LAYER = 'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.0.mlp'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class FastSlowTileMLP(torch.nn.Module):
    def __init__(self, native, fast, mean, mode, branch, policy, tile_module):
        super().__init__()
        require(type(native).__name__ == 'MS_Spiking_Mlp' and native.norm_layer == 'BN', 'Native BN MLP required')
        require(mode in ('actual', 'reference') and branch in ('zero', 'mean', 'trained'), 'Unknown execution')
        require(type(fast).__name__ == 'FastBranch' and not fast.training and
                all(not p.requires_grad for p in fast.parameters()), 'Frozen inherited FastBranch required')
        require(tuple(mean.shape) == (384,) and mean.device.type == 'cpu' and bool(torch.isfinite(mean).all()), 'Mean shape differs')
        require(native.fc1.in_features == 384 and native.fc1.out_features == 1536 and
                native.fc2.in_features == 1536 and native.fc2.out_features == 384, 'Stage2 widths differ')
        for sn in (native.sn1.spiking_neuron, native.sn2.spiking_neuron):
            require(type(sn).__name__ == 'PSN' and sn.T == 10, 'Complete-T10 PSN required')
        tile_module.frozen(native)
        self.native, self.fast = native, fast
        self.register_buffer('mean', mean.detach().clone())
        self.mode, self.branch, self.policy = mode, branch, policy
        self.tile_module, self.records = tile_module, []
        self.eval()

    def forward(self, u):
        require(not torch.is_grad_enabled() and tuple(u.shape) == (10, 1, 30, 40, 384) and
                u.device.type == 'cpu', 'Only fixed CPU stage2 inference supported')
        self.tile_module.frozen(self.native)
        keep = self.policy(LAYER, u)
        require(isinstance(keep, torch.Tensor) and keep.dtype == torch.bool and tuple(keep.shape) == (4, 5), 'Invalid mask')
        observed = {'slow_fc1_rows': 0, 'slow_fc2_rows': 0, 'fast_fc_rows': 0,
                    'slow_calls': 0, 'fast_calls': 0, 'mean_fill_elements': 0}
        handles = []
        for module, key in ((self.native.fc1, 'slow_fc1_rows'), (self.native.fc2, 'slow_fc2_rows'),
                            (self.fast.fc1[1], 'fast_fc_rows')):
            def count(m, inputs, k=key):
                observed[k] += inputs[0].numel()//inputs[0].shape[-1]
            handles.append(module.register_forward_pre_hook(count))
        out = torch.zeros_like(u, memory_format=torch.contiguous_format)
        slow_points = 0; slow_tiles = 0
        try:
            for gy in range(4):
                for gx in range(5):
                    ys = slice(9*gy, min(9*(gy+1), 30)); xs = slice(9*gx, min(9*(gx+1), 40))
                    selected = bool(keep[gy, gx]); points = (ys.stop-ys.start)*(xs.stop-xs.start)
                    slow_points += points*selected; slow_tiles += int(selected)
                    local = u[:, :, ys, xs, :].contiguous()
                    slow = fast = None
                    if self.mode == 'reference' or selected:
                        slow = self.native(local); observed['slow_calls'] += 1
                    if self.mode == 'reference' or not selected:
                        if self.branch == 'trained':
                            fast = self.fast(local); observed['fast_calls'] += 1
                        elif self.branch == 'mean':
                            fast = self.mean.expand_as(local)
                            observed['mean_fill_elements'] += local.numel()
                    chosen = slow if selected else fast
                    if chosen is not None:
                        require(chosen.shape == local.shape and bool(torch.isfinite(chosen).all()), 'Invalid branch')
                        out[:, :, ys, xs, :] = chosen
        finally:
            for handle in handles:
                handle.remove()
        expected_slow_rows = 12000 if self.mode == 'reference' else 10*slow_points
        expected_fast_rows = (12000 if self.mode == 'reference' else 10*(1200-slow_points)) if self.branch == 'trained' else 0
        require(observed['slow_fc1_rows'] == expected_slow_rows and observed['slow_fc2_rows'] == expected_slow_rows and
                observed['fast_fc_rows'] == expected_fast_rows, 'Actual FC rows differ')
        require(observed['slow_calls'] == (20 if self.mode == 'reference' else slow_tiles), 'Slow calls differ')
        require(observed['fast_calls'] == ((20 if self.mode == 'reference' else 20-slow_tiles) if self.branch == 'trained' else 0), 'Fast calls differ')
        self.records.append({'layer_name': LAYER, 'shape_TBHWC': list(u.shape), 'keep': keep.tolist(),
            'mode': self.mode, 'branch': self.branch, 'slow_points': slow_points, 'fast_region_points': 1200-slow_points,
            'slow_tiles': slow_tiles, 'observed': observed, 'input_sha256': tensor_sha(u),
            'output_sha256': tensor_sha(out), 'input_stride': list(u.stride()), 'output_stride': list(out.stride()),
            'fast_real_MACs_observed_rows_times_C2': observed['fast_fc_rows']*384*384,
            'fast_LN_rows': observed['fast_fc_rows'], 'all_keep_shortcut': False,
            'mean_fill_elements_scope': 'logical mean candidates; reference may discard them; not measured memory writes',
            'scope': 'actual FC rows/calls; not PE cycles; fast is real LN+Linear, not spike AC'})
        return out


@contextlib.contextmanager
def replace_one(model, fast, mean, mode, branch, policy, tile_module):
    require(all(not m.training for m in model.modules()), 'Whole model must be eval')
    parent_name, child = LAYER.rsplit('.', 1)
    parent = model.get_submodule(parent_name); original = getattr(parent, child)
    require(getattr(parent, 'cnf', None) == 'ADD' and not getattr(parent, 'use_checkpoint', False), 'ADD/no-checkpoint parent required')
    wrapper = FastSlowTileMLP(original, fast, mean, mode, branch, policy, tile_module)
    try:
        setattr(parent, child, wrapper)
        yield wrapper
    finally:
        setattr(parent, child, original)
