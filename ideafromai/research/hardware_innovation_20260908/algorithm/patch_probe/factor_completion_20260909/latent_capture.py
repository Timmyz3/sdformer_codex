"""Bounded real-frame capture for disjoint latent-stage factor students.

Only gates, source-empty bits and tail-J2 demands cover the full image.
All real-valued tensors are five shared 5x5 windows. Tail demand uses only
accepted flags, source emptiness and static A/V connectivity, never Z values.
Run --self-check for a CPU layout/independent nested-OR check, not network AEE.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'joint_completion_20260909'))
from evaluate_network import BLOCK, TARGET, FrameCapture, window_snapshot, save_consumer
from latent_stage_train16.adapter import LatentTemporal, LatentPair


def tail_need_z(accepted, source_empty, a, v, shared_rank, latent_word=2):
    """Bool [G,S,P,J2_tail], no private values or future gates as inputs.

    Need a latent pair if any unresolved (t,h) uses source time s through
    nonzero A[t,s] and either V[r,h] in that pair. An empty source receptive
    field contributes no Z, even when a constant BN output remains live.
    """
    support = v[shared_rank:].ne(0).reshape(-1, latent_word, v.shape[1]).any(1)
    e = a.ne(0)
    need_y = ((~accepted).permute(0, 2, 3, 1).float() @ e.float()).gt(0)
    need_y = need_y.permute(0, 3, 1, 2) & (~source_empty)[..., None]
    return (need_y.float() @ support.float().T).gt(0)


def save_latent_consumer(modules, fixed, output):
    """Real W2/BN/shortcut parameters; original W1 is only a reference."""
    save_consumer(modules, fixed, output)
    file = Path(output)/'parameters.npz'
    with np.load(file) as data:
        arrays = {key: data[key].copy() for key in data.files}
    arrays['unfactored_reference_W1'] = arrays.pop('W1')
    arrays['unfactored_reference_conv1_bias'] = arrays.pop('conv1_bias')
    arrays['actual_conv1_weights'] = np.asarray(
        'Each axis/student_parameters.npz: u/v and any u_int8/u_dyadic_scale/v_sign/v_shift/v_nonzero; original W1 is not executed.')
    np.savez(file, **arrays)


class LatentFrameCapture(FrameCapture):
    def __init__(self, modules, output, arrays, conditional):
        super().__init__(modules, output, float(arrays['theta_source']),
                         float(arrays['theta_output']), dense=True)
        self.conditional = bool(conditional)
        self.shared_rank = int(arrays['shared_rank'])
        self.rank = arrays['v'].shape[0]
        self.latent_word = 2
        if (self.rank-self.shared_rank) % 2:
            raise ValueError('Tail coefficient words must contain two latent lanes.')
        device = modules[BLOCK+'.conv1.0'].weight.device
        self.a = torch.as_tensor(arrays['a'], device=device).float()
        self.v = torch.as_tensor(arrays['v'], device=device).float()
        # Full-mode preview windows are optional diagnostic arithmetic on
        # CPU. They are not used by the full network or its demand masks.
        self.preview = LatentTemporal(arrays).cpu().eval()
        self.output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.output/'student_parameters.npz', **arrays)
        self.latent_record = {}
        self.accepted_packed = self.need_packed = None

    def begin(self, index, name):
        super().begin(index, name)
        self.latent_record = {}
        self.accepted_packed = self.need_packed = None

    def __call__(self, event, **values):
        if not self.active:
            return
        if event == 'conv':
            self._conv_parts(**values)
        elif event == 'neuron_groups':
            self._decisions(**values)
        else:
            raise ValueError(event)

    def _conv_parts(self, z, shared_raw, tail_raw, source_empty):
        windows, origins = window_snapshot(z[:, None])
        self.latent_record.update(Z_shared=windows[:, :, :self.shared_rank],
                                  Z_tail=windows[:, :, self.shared_rank:],
                                  window_origins_yx=origins)
        self.latent_record['shared_raw'], _ = window_snapshot(shared_raw[:, None])
        self.latent_record['tail_raw'], _ = window_snapshot(tail_raw[:, None])
        self.latent_record['source_empty'], _ = window_snapshot(source_empty[:, None, None])
        bits = source_empty.detach().cpu().numpy()
        self.record['source_empty_bits'] = np.packbits(bits.reshape(-1), bitorder='little')
        self.record['source_empty_shape'] = np.asarray(bits.shape, dtype=np.int32)

    def _window_groups(self, key, values, first, last, width):
        """Gather only requested pixels from [G,T,P,H], without full reorder."""
        origins = self.latent_record['window_origins_yx']
        channels = values.shape[-1]
        if key not in self.latent_record:
            self.latent_record[key] = np.empty((len(origins), 10, channels, 5, 5), np.float32)
        for i, (y0, x0) in enumerate(origins):
            lo, hi = max(int(y0), first), min(int(y0)+5, last)
            if hi <= lo:
                continue
            yy = torch.arange(lo, hi, device=values.device)[:, None]
            xx = torch.arange(int(x0), int(x0)+5, device=values.device)[None]
            groups = (yy-first)*(width//4)+xx//4
            p = (xx % 4).expand(hi-lo, -1)
            small = values[groups, :, p].permute(2, 3, 0, 1)
            self.latent_record[key][i, :, :, lo-int(y0):hi-int(y0)] = small.detach().cpu().numpy()

    def _decisions(self, first, last, height, width, empty, details, conditional):
        if width % 8:
            raise ValueError('Native capture width must be divisible by eight for direct gate bit packing.')
        accepted = details['accepted']
        need = tail_need_z(accepted, empty, self.a, self.v, self.shared_rank)
        count = height*(width//4)
        tail_words = (self.rank-self.shared_rank)//2
        if self.accepted_packed is None:
            self.accepted_packed = np.empty((10, 96, height, width//8), np.uint8)
            self.need_packed = np.empty((count, 10*4*tail_words//8), np.uint8)
        native = accepted.reshape(last-first, width//4, 10, 4, 96)
        native = native.permute(2, 4, 0, 1, 3).reshape(10, 96, last-first, width)
        packed = np.packbits(native.detach().cpu().numpy(), axis=-1, bitorder='little')
        self.accepted_packed[:, :, first:last] = packed
        self.need_packed[first*(width//4):last*(width//4)] = np.packbits(
            need.detach().cpu().numpy().reshape(-1, 10*4*tail_words), axis=-1, bitorder='little')
        self._window_groups('full_margin', details['full_margin'], first, last, width)
        if conditional:
            for source, target in [('shared_margin', 'shared_margin'),
                                   ('predicted', 'predicted_margin'), ('radius', 'radius')]:
                self._window_groups(target, details[source], first, last, width)

    def _full_preview_windows(self):
        """Only shared raw and source-empty metadata; no private or full Y."""
        shared = torch.from_numpy(self.latent_record['shared_raw'])
        shared = shared.permute(0, 1, 3, 4, 2).reshape(-1, 10, 25, 96)
        empty = torch.from_numpy(self.latent_record['source_empty']).reshape(-1, 10, 25).bool()
        temporal = self.preview
        with torch.no_grad():
            y = shared*temporal.bn_scale+temporal.bn_bias
            base = torch.einsum('ts,gsph->gtph', temporal.a, y)
            base = base+temporal.b[None, :, None, None]-temporal.theta
            offsets, radii = [], []
            for t in range(10):
                indices = getattr(temporal, 'indices_'+str(t))
                local = empty[:, indices].permute(0, 2, 1).long()
                code = (local*(1 << torch.arange(len(indices)))).sum(-1)
                offsets.append(getattr(temporal, 'mean_'+str(t))[code])
                radii.append(getattr(temporal, 'radius_'+str(t))[code])
            for key, values in [('shared_margin', base),
                                ('predicted_margin', base+torch.stack(offsets, 1)),
                                ('radius', torch.stack(radii, 1))]:
                self.latent_record[key] = values.reshape(-1, 10, 5, 5, 96).permute(0, 1, 4, 2, 3).numpy()

    def finish(self):
        if not self.active:
            return
        directory = self.output/(f'{self.index:03d}_'+Path(self.name).stem)
        directory.mkdir(parents=True, exist_ok=True)
        keys = ('source_gate_bits', 'source_gate_shape', 'output_gate_bits',
                'output_gate_shape', 'source_empty_bits', 'source_empty_shape')
        gates = {key: self.record.pop(key) for key in keys}
        if self.accepted_packed is None or self.need_packed is None:
            raise RuntimeError('No latent decision callback was received for this frame.')
        height, width = gates['source_gate_shape'][-2:]
        gates.update(accepted_gate_bits=self.accepted_packed.reshape(-1),
            accepted_gate_shape=gates['output_gate_shape'], need_Z_bits=self.need_packed.reshape(-1),
            need_Z_shape=np.asarray((height*(width//4), 10, 4, (self.rank-self.shared_rank)//2), np.int32),
            theta_source=np.asarray(self.source_theta), theta_output=np.asarray(self.target_theta),
            frame_name=np.asarray(self.name), bitorder=np.asarray('little'),
            shared_rank=np.asarray(self.shared_rank), latent_word=np.asarray(2),
            conditional=np.asarray(self.conditional),
            preview_acceptance_evaluated=np.asarray(self.conditional),
            gate_layout=np.asarray('flat T,C,H,W C order'),
            source_empty_semantics=np.asarray('T,H,W; exact OR over original source C and zero-padded 3x3 receptive field'),
            need_Z_layout=np.asarray('G,S,P,J2_tail C order; G=y*(W/4)+x//4, P=x%4, latent=shared_rank+2*j+[0,1]'),
            need_Z_semantics=np.asarray('OR_(t,h,r in J2) !accepted[g,t,p,h] & (A[t,s]!=0) & (V[r,h]!=0) & !source_empty[g,s,p]; no private Z used; full accepted=0'))
        np.savez_compressed(directory/'gates.npz', **gates)
        np.savez_compressed(directory/'consumer_windows.npz', **self.record,
            frame_name=np.asarray(self.name), windows_layout=np.asarray('window,T,C,local_y,local_x'),
            sn2_gate_semantics=np.asarray('Boolean; actual payload=theta_output*g'),
            identity_semantics=np.asarray('real block input; retained shortcut'),
            output_semantics=np.asarray('real norm2(conv2(theta_output*g))+identity'))
        if not self.conditional:
            self._full_preview_windows()
        np.savez_compressed(directory/'latent_windows.npz', **self.latent_record,
            frame_name=np.asarray(self.name), windows_layout=np.asarray('window,T,channel,local_y,local_x'),
            Z_semantics=np.asarray('continuous factor outputs from this forward; split by shared_rank'),
            raw_semantics=np.asarray('separate V outputs before original fixed norm1; conv1_raw=shared_raw+tail_raw'),
            full_margin_semantics=np.asarray('actual A@norm1_Y + temporal_bias - theta_output'),
            prediction_semantics=np.asarray('actual same-forward preview/radius' if self.conditional else
                'capture-only CPU preview from saved shared_raw/source_empty; not evaluated for full-mode decisions'),
            radius_semantics=np.asarray('train-calibrated statistical residual radius; not a strict certificate'),
            source_empty_layout=np.asarray('window,T,1,local_y,local_x'))
        self.record.clear()
        self.latent_record.clear()
        self.accepted_packed = self.need_packed = None
        self.active = self.dense_active = False


def self_check(parameter_file):
    """A bounded synthetic image tests layout; an independent loop tests OR."""
    from torch import nn
    import torch.nn.functional as F
    torch.set_num_threads(4)
    with np.load(parameter_file) as data:
        arrays = {key: data[key].copy() for key in data.files}
    generator = torch.Generator().manual_seed(913)
    accepted = torch.rand((2, 10, 4, 96), generator=generator) < .94
    empty = torch.rand((2, 10, 4), generator=generator) < .2
    a, v = torch.from_numpy(arrays['a']), torch.from_numpy(arrays['v'])
    shared = int(arrays['shared_rank'])
    observed = tail_need_z(accepted, empty, a, v, shared)
    expected = torch.zeros_like(observed)
    e, support = a.ne(0).numpy(), v.ne(0).numpy()
    aa, ee = accepted.numpy(), empty.numpy()
    for g in range(2):
        for s in range(10):
            for p in range(4):
                for j in range((len(v)-shared)//2):
                    expected[g, s, p, j] = not ee[g, s, p] and any(
                        e[t, s] and not aa[g, t, p, h] and
                        (support[shared+2*j, h] or support[shared+2*j+1, h])
                        for t in range(10) for h in range(96))
    assert torch.equal(observed, expected)
    x = (torch.rand((10, 1, 96, 11, 16), generator=generator) < .04).float()
    x[:, :, :, :4, :4] = 0
    x *= float(arrays['theta_source'])
    scale = torch.from_numpy(arrays['bn_scale']).float()[None, None, :, None, None]
    bias = torch.from_numpy(arrays['bn_bias']).float()[None, None, :, None, None]

    class Example(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(96, 96, 3, padding=1, bias=False)
            self.neuron = nn.Identity()
            self.conv2 = nn.Identity()

        def forward(self, value):
            raw = self.conv1(value)
            gate = self.neuron(raw*scale+bias)
            return self.conv2(gate)+value

    results = {}
    with tempfile.TemporaryDirectory(prefix='latent_capture_') as temporary, torch.no_grad():
        for mode in (False, True):
            reference = LatentPair(arrays, 'cpu', conditional=mode)
            raw = reference.conv_forward(x)
            expected_gate = reference.neuron_forward(raw*scale+bias)
            block = Example()
            modules = {BLOCK: block, BLOCK+'.conv1.0': block.conv1,
                       BLOCK+'.conv2.0': block.conv2, TARGET: block.neuron}
            capture = LatentFrameCapture(modules, Path(temporary)/str(mode), arrays, mode)
            pair = LatentPair(arrays, 'cpu', conditional=mode, capture_callback=capture)
            block.conv1.forward, block.neuron.forward = pair.conv_forward, pair.neuron_forward
            capture.begin(0, 'synthetic_layout_only.npz')
            output = block(x)
            capture.finish()
            capture.close()
            assert torch.equal(output, expected_gate+x)
            directory = Path(temporary)/str(mode)/'000_synthetic_layout_only'
            with np.load(directory/'gates.npz') as data:
                decode = lambda key: np.unpackbits(data[key+'_bits'], bitorder='little')[:int(np.prod(data[key+'_shape']))].reshape(data[key+'_shape']).astype(bool)
                actual_gate, actual_source = decode('output_gate'), decode('source_gate')
                native_accept, native_empty, packed_need = decode('accepted_gate'), decode('source_empty'), decode('need_Z')
            assert np.array_equal(actual_gate, expected_gate[:, 0].ne(0).numpy())
            assert np.array_equal(actual_source, x[:, 0].ne(0).numpy())
            empty_reference = np.empty_like(native_empty)
            for yy in range(x.shape[-2]):
                for xx in range(x.shape[-1]):
                    neighborhood = actual_source[:, :, max(0, yy-1):yy+2, max(0, xx-1):xx+2]
                    empty_reference[:, yy, xx] = ~neighborhood.any((1, 2, 3))
            assert np.array_equal(native_empty, empty_reference)
            accepted_groups = torch.from_numpy(native_accept).reshape(10, 96, 11, 4, 4).permute(2, 3, 0, 4, 1).reshape(-1, 10, 4, 96)
            empty_groups = torch.from_numpy(native_empty).reshape(10, 11, 4, 4).permute(1, 2, 0, 3).reshape(-1, 10, 4)
            assert np.array_equal(packed_need, tail_need_z(accepted_groups, empty_groups, a, v, shared).numpy())
            if not mode:
                assert not native_accept.any()
            with np.load(directory/'latent_windows.npz') as data, np.load(directory/'consumer_windows.npz') as consumer:
                assert np.array_equal(data['window_origins_yx'], consumer['window_origins_yx'])
                assert np.array_equal(data['shared_raw']+data['tail_raw'], consumer['conv1_raw'])
                full_windows = window_snapshot(expected_gate)[0].astype(bool)
                assert np.array_equal(full_windows, consumer['sn2_gate'])
                assert all(np.isfinite(data[key]).all() for key in ('full_margin', 'predicted_margin', 'radius'))
                direct_margin = torch.einsum('ts,sbchw->tbchw', a.float(), raw*scale+bias)
                direct_margin += torch.from_numpy(arrays['temporal_bias']).float().reshape(10, 1, 1, 1, 1)
                direct_margin -= float(arrays['theta_output'])
                reference_windows = window_snapshot(direct_margin)[0]
                margin_error = float(np.max(np.abs(data['full_margin']-reference_windows)))
                assert np.allclose(data['full_margin'], reference_windows, rtol=2e-6, atol=2e-6)
            with np.load(directory.parent/'student_parameters.npz') as exported:
                assert set(exported.files) == set(arrays)
                assert all(np.array_equal(exported[key], value) for key, value in arrays.items())
            results['conditional' if mode else 'full'] = dict(gates=expected_gate.numel(),
                default_forward_differences=0, packed_gate_differences=0,
                packed_need_Z_differences=0, windows_same_coordinates=True,
                source_empty_receptive_field_differences=0, student_fields_copied_exactly=True,
                native_margin_window_max_abs_error=margin_error,
                premature_accepts_full_mode=int(native_accept.sum()) if not mode else None)
    return dict(scope='CPU synthetic image/layout and independently nested Boolean OR; no real network/AEE/GPU',
                nested_OR_entries=observed.numel(), nested_OR_differences=0, modes=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', type=Path, required=True, metavar='STUDENT_NPZ')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = self_check(args.self_check)
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
