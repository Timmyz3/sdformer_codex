"""Capture two fixed, preselected full-chain windows of existing students.

Inference only. Reuses the exact Stage B builder and exported fixed constants;
no new quantization, training, full activation archive, or GPU speed claim.
"""
from pathlib import Path
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import capture_inputs as base

# Output coordinates fixed before observing either model's values.
OUTPUTS = {'corner': (0, 0), 'interior': (60, 80)}


def geometry(origin, shape=(240, 320)):
    y, x = origin
    gate_lo = (max(0, 2*y-1), max(0, 2*x-1))
    gate_hi = (min(shape[0], 2*(y+3)+2), min(shape[1], 2*(x+3)+2))
    source_lo = (max(0, gate_lo[0]-1), max(0, gate_lo[1]-1))
    source_hi = (min(shape[0], gate_hi[0]+1), min(shape[1], gate_hi[1]+1))
    return dict(output_origin=list(origin), output_shape=[4, 4],
                gate_origin=list(gate_lo), gate_shape=[gate_hi[i]-gate_lo[i] for i in (0, 1)],
                source_origin=list(source_lo), source_shape=[source_hi[i]-source_lo[i] for i in (0, 1)])


def crop(value, origin, shape):
    # Actual producer tensors are either T,B,C,H,W with B1 or TB,C,H,W.
    if value.ndim == 5:
        assert value.shape[1] == 1
        value = value[:, 0]
    y, x = origin
    h, w = shape
    return value[:, :, y:y+h, x:x+w]


class WindowCapture(base.OrderedCapture):
    def __init__(self, model, modules, helper, theta_sn2, names, directory, continuous_only=False):
        from capture import SOURCE_SN, BLOCK, CONSUMER_SN, PROJECT
        super().__init__(model, modules, helper, theta_sn2, names, directory, True)
        self.window_geometry = {name: geometry(origin) for name, origin in OUTPUTS.items()}
        self.pair = modules[BLOCK+'.conv1.0'].forward.__self__
        self.previous_callback = self.pair.capture_callback
        self.pair.capture_callback = self.preview_values
        self.handles.extend([
            modules[SOURCE_SN].register_forward_hook(self.source_values),
            modules[BLOCK+'.norm1'].register_forward_hook(self.bn1_values),
            modules[BLOCK+'.sn2.spiking_neuron'].register_forward_hook(self.sn2_values),
            modules[CONSUMER_SN].register_forward_hook(self.proj_gate_values),
            modules[PROJECT+'.conv'].register_forward_hook(self.output_values('proj_conv_fp32')),
            modules[PROJECT+'.norm_layer'].register_forward_hook(self.output_values('proj_norm_fp32')),
            modules[PROJECT].register_forward_hook(self.output_values('ped_output_fp32', final=True)),
        ])
        parameters = dict(
            preview_u=self.pair.u.detach().reshape(self.pair.u.shape[0], -1).T.cpu().numpy(),
            preview_v=self.pair.v.detach()[:, :, 0, 0].T.cpu().numpy(),
            preview_shared_rank=np.array(self.pair.shared_rank),
            preview_A=self.pair.temporal.a.detach().cpu().numpy(),
            preview_b=self.pair.temporal.b.detach().cpu().numpy(),
            preview_theta_source=np.array(self.pair.source_theta),
            preview_theta_output=np.array(self.pair.temporal.theta),
            proj_theta_output=np.array(float(modules[CONSUMER_SN].thresh)),
            source_module=np.array(SOURCE_SN), proj_module=np.array(PROJECT))
        for prefix, path in [('bn1', BLOCK+'.norm1.norm_layer'), ('proj_bn', PROJECT+'.norm_layer')]:
            bn = modules[path]
            assert not bn.training
            parameters[prefix+'_name'] = np.array(path)
            parameters[prefix+'_eps'] = np.array(bn.eps)
            parameters[prefix+'_training'] = np.array(bn.training)
            parameters[prefix+'_track_running_stats'] = np.array(bn.track_running_stats)
            for name, attr in [('gamma','weight'), ('beta','bias'), ('mean','running_mean'), ('var','running_var')]:
                value = getattr(bn, attr)
                parameters[prefix+'_'+name] = (np.empty(0, np.float32) if value is None else value.detach().cpu().numpy())
            print('LIVE_BN_MODE', prefix, 'training', bn.training,
                  'track_running_stats', bn.track_running_stats,
                  'uses_batch_statistics', bn.running_mean is None or bn.running_var is None,
                  flush=True)
        conv = modules[PROJECT+'.conv']
        parameters.update(proj_weight_fp32=conv.weight.detach().cpu().numpy(),
            proj_bias_fp32=(np.empty(0, np.float32) if conv.bias is None else conv.bias.detach().cpu().numpy()),
            proj_has_bias=np.array(conv.bias is not None))
        for name in ('stride', 'padding', 'dilation', 'kernel_size', 'groups'):
            parameters['proj_'+name] = np.array(getattr(conv, name))
        np.savez_compressed(directory/'live_parameters.npz', **parameters)

    def begin(self, module, inputs):
        super().begin(module, inputs)
        self.arrays['window_geometry_json'] = np.array(json.dumps(self.window_geometry))

    def save_value(self, name, value, domain='gate', dtype=None):
        for label, geo in self.window_geometry.items():
            local = crop(value, geo[domain+'_origin'], geo[domain+'_shape'])
            array = local.detach().cpu().numpy()
            self.arrays[label+'_'+name] = array if dtype is None else array.astype(dtype)

    def save_gate(self, name, output, theta, domain='gate'):
        import torch
        for label, geo in self.window_geometry.items():
            value = crop(output, geo[domain+'_origin'], geo[domain+'_shape'])
            gate = value.ne(0)
            error = float(torch.where(gate, value-float(theta), value).abs().max())
            assert error == 0
            self.arrays[label+'_'+name] = gate.detach().cpu().numpy()
            self.stats[label+'_'+name] = dict(elements=gate.numel(), theta=float(theta), theta_g_error=error)

    def source_values(self, module, inputs, output):
        self.order.append('source_from_I24')
        self.save_value('I24', self.helper.i, 'source', np.int32)
        self.save_gate('sn1_gate', output, float(module.thresh), 'source')

    def preview_values(self, kind, **values):
        if kind != 'conv':
            return
        self.order.append('FP32_preview_U_V')
        self.save_value('preview_Z_shared', values['z'][:, :self.pair.shared_rank])
        self.save_value('preview_shared_raw', values['shared_raw'])
        self.save_value('preview_tail_raw', values['tail_raw'])

    def bn1_values(self, module, inputs, output):
        self.order.append('fixed_BN1')
        self.save_value('preview_BN1_Y', output)

    def sn2_values(self, module, inputs, output):
        self.order.append('full_T10_sn2')
        self.save_gate('sn2_gate', output, self.theta_sn2)

    def observe_continuous(self, module, inputs, output):
        import torch
        assert getattr(module.forward, '__self__', None) is self.helper and self.helper.ready
        self.order.append('fixed_PED_continuous')
        for label, geo in self.window_geometry.items():
            local = crop(output, geo['output_origin'], geo['output_shape']).double()*(1 << 14)
            expected = crop(self.helper.continuous, geo['output_origin'], geo['output_shape'])
            assert torch.equal(local, expected) and torch.equal(local, local.round())
            self.arrays[label+'_continuous_q24'] = local.to(torch.int32).cpu().numpy()
        self.save_value('updated_I24', self.helper.updated, dtype=np.int32)

    def proj_gate_values(self, module, inputs, output):
        self.order.append('fixed_proj_gate')
        self.save_gate('proj_gate', output, float(module.thresh))

    def output_values(self, name, final=False):
        def hook(module, inputs, output):
            self.order.append(name)
            if name == 'proj_norm_fp32':
                import torch
                value = inputs[0]
                self.arrays['proj_bn_full_input_shape'] = np.array(value.shape)
                self.arrays['proj_bn_uses_batch_statistics'] = np.array(module.running_mean is None or module.running_var is None)
                if module.running_mean is None or module.running_var is None:
                    # Observe, never replace, the actual full-domain BN. These
                    # statistics are verification data, NOT free constants in
                    # a local hardware tile or permission to skip the domain.
                    variance, mean = torch.var_mean(value, dim=(0, 2, 3), correction=0)
                    self.arrays['proj_bn_actual_domain_mean'] = mean.detach().cpu().numpy()
                    self.arrays['proj_bn_actual_domain_var'] = variance.detach().cpu().numpy()
                    self.arrays['proj_bn_full_input_fp32'] = value.detach().cpu().numpy()
                    self.arrays['proj_bn_full_output_fp32'] = output.detach().cpu().numpy()
            self.save_value(name, output, 'output')
            if final:
                self.save_frame()
        return hook

    def restore(self):
        self.pair.capture_callback = self.previous_callback
        super().restore()


def main():
    output = HERE/'capture'
    if '--output' not in sys.argv:
        sys.argv += ['--output', str(output)]
    else:
        output = Path(sys.argv[sys.argv.index('--output')+1])
    # Select the parent's observer-only route; this subclass saves all required
    # local stages and deliberately avoids its full-frame gate archive.
    if '--continuous-only' not in sys.argv:
        sys.argv += ['--continuous-only']
    base.OrderedCapture = WindowCapture
    base.main()
    record = json.loads((output/'result.json').read_text())
    record.update(scope='Two preselected local full-chain windows of unchanged fixed students; actual live FP32 preview/native projection and fixed integer consumers.',
        window_geometry={name: geometry(origin) for name, origin in OUTPUTS.items()},
        continuous_only=False, capture_scope='Local I24, three local gates, preview Z/shared/tail/BN1, integer continuous/updated I, native projection convolution/BN/final addition; no new quantization or training.',
        layout='Each tensor T,C,H,W; per-window global origins are explicit in window_geometry_json.')
    base.save_json(output/'result.json', record)


if __name__ == '__main__':
    main()
