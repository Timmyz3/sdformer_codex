"""Independent inference reference: all tiles then mask versus kept tiles only.

Both execute complete T10/all-C valid-size tiles; neither has an all-keep
full-shape shortcut. This is a numerical/execution check, not a speed engine.
"""
import contextlib
import hashlib
import math
import torch


def require(ok, message):
    if not ok: raise ValueError(message)


def frozen(module):
    for name, layer in module.named_modules():
        require(not layer.training, 'Inference/eval required: '+name)
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
            require(layer.track_running_stats and layer.running_mean is not None and layer.running_var is not None,
                    'Current-stat BN prohibited: '+name)
            require(bool(torch.isfinite(layer.running_mean).all()) and bool(torch.isfinite(layer.running_var).all())
                    and bool((layer.running_var >= 0).all()), 'Invalid BN buffers: '+name)


def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class ReplayPolicy:
    def __init__(self, records):
        self.masks = {r['layer_name']: r for r in records}
        require(len(self.masks) == len(records), 'Duplicate replay layer')
        self.visited = []

    def __call__(self, name, x):
        require(name in self.masks and name not in self.visited, 'Missing/duplicate replay call: '+name)
        record = self.masks[name]
        require(list(x.shape) == record['shape_TBHWC'], 'Replay geometry differs')
        self.visited.append(name)
        return torch.tensor(record['keep'], dtype=torch.bool, device=x.device)

    def complete(self):
        require(set(self.visited) == set(self.masks), 'Incomplete mask replay')


class SameShapeTileMLP(torch.nn.Module):
    def __init__(self, native, name, mode, policy, audit_hashes=False):
        super().__init__()
        require(type(native).__name__ == 'MS_Spiking_Mlp' and native.norm_layer == 'BN', 'Native BN MLP required')
        require(mode in ('tiled_all_then_mask', 'tiled_kept_only'), 'Unknown mode')
        for sn in (native.sn1.spiking_neuron, native.sn2.spiking_neuron):
            require(type(sn).__name__ == 'PSN' and sn.T == 10, 'Complete-T10 stateless PSN required')
        frozen(native)
        self.native, self.layer_name, self.mode, self.policy = native, name, mode, policy
        self.audit_hashes, self.records = audit_hashes, []
        self.eval()

    def forward(self, x):
        require(not torch.is_grad_enabled(), 'Inference only')
        frozen(self.native)
        require(x.ndim == 5 and tuple(x.shape[:2]) == (10, 1), 'Expected T10/B1/H/W/C')
        t, b, h, w, c = x.shape
        keep = self.policy(self.layer_name, x)
        require(isinstance(keep, torch.Tensor) and keep.dtype == torch.bool and
                tuple(keep.shape) == (math.ceil(h/9), math.ceil(w/9)), 'Invalid whole-tile mask')
        decisions = keep.detach().cpu().tolist()
        observed = {'fc1_rows': 0, 'fc2_rows': 0, 'native_mlp_calls': 0}
        handles = []
        for name in ('fc1', 'fc2'):
            def count(module, inputs, key=name):
                observed[key+'_rows'] += inputs[0].numel() // inputs[0].shape[-1]
            handles.append(getattr(self.native, name).register_forward_pre_hook(count))
        # Explicit contiguous allocation prevents mode-dependent returned layout.
        out = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
        kept_pixels, kept_tiles = 0, 0
        try:
            for gy, row in enumerate(decisions):
                for gx, accept in enumerate(row):
                    y0, y1, x0, x1 = 9*gy, min(9*(gy+1), h), 9*gx, min(9*(gx+1), w)
                    if accept:
                        kept_pixels += (y1-y0)*(x1-x0); kept_tiles += 1
                    if self.mode == 'tiled_kept_only' and not accept: continue
                    local = x[:, :, y0:y1, x0:x1, :].contiguous()
                    branch = self.native(local)
                    require(tuple(branch.shape) == tuple(local.shape), 'Native branch external shape changed')
                    observed['native_mlp_calls'] += 1
                    if accept: out[:, :, y0:y1, x0:x1, :] = branch
        finally:
            for handle in handles: handle.remove()
        expected_rows = t*(h*w if self.mode == 'tiled_all_then_mask' else kept_pixels)
        require(observed['fc1_rows'] == expected_rows and observed['fc2_rows'] == expected_rows, 'FC service differs')
        expected_calls = math.ceil(h/9)*math.ceil(w/9) if self.mode == 'tiled_all_then_mask' else kept_tiles
        require(observed['native_mlp_calls'] == expected_calls, 'Tile call count differs')
        record = {'layer_name': self.layer_name, 'mode': self.mode, 'shape_TBHWC': list(x.shape),
            'input_stride': list(x.stride()), 'output_stride': list(out.stride()), 'keep': decisions,
            'kept_real_pixels': kept_pixels, 'kept_tiles': kept_tiles, 'observed': observed,
            'expected_fc_rows_per_layer': expected_rows, 'all_keep_shortcut': False,
            'tail_rule': 'valid spatial extent, no zero padding; full T10 and channels',
            'scope': 'observed FC rows and calls, not PE cycles/traffic'}
        if self.audit_hashes: record.update(input_sha256=tensor_sha(x), output_sha256=tensor_sha(out))
        self.records.append(record)
        return out


@contextlib.contextmanager
def same_shape_mlp_model(model, mode, policy, audit_hashes=False):
    frozen(model)
    targets = [(n, m) for n, m in model.named_modules() if type(m).__name__ == 'MS_Spiking_Mlp']
    require(bool(targets), 'No native MLPs')
    replacements, wrappers = [], []
    try:
        for name, original in targets:
            parent_name, child = name.rsplit('.', 1)
            parent = model.get_submodule(parent_name)
            require(getattr(parent, 'cnf', None) == 'ADD' and not getattr(parent, 'use_checkpoint', False),
                    'ADD residual and no activation checkpoint required')
            wrapper = SameShapeTileMLP(original, name, mode, policy, audit_hashes)
            replacements.append((parent, child, original)); setattr(parent, child, wrapper); wrappers.append(wrapper)
        yield wrappers
    finally:
        for parent, child, original in reversed(replacements): setattr(parent, child, original)
