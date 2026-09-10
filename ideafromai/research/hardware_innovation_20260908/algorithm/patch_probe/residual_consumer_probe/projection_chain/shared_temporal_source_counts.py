"""Actual theta*g source counts for the shared-temporal control family.

Usage during an existing evaluation (does not change any forward):
    counts = SharedTemporalSourceCounts(modules, sn2_theta=pair.temporal.theta).install()
    counts.reset(axis)
    # Run exactly one ordinary network forward for each name.
    result = counts.report(names)
    counts.restore()

The caller installs the common nonanchor whole-BN2-branch deletion. Conv2
therefore needs only even/even outputs; their 3x3 input-neighbourhood union
still requires every sn2 position and hence full-spatial preview Conv1.
Counts include halo repetitions, not coefficient-zero filtering, physical
memory requests, cycles, or dense second-factor/BN/PSN work. No GPU is launched
by this file. --self-test uses small CPU tensors only.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from count_projection_sources import count_source, explicit_padded_reference


PATCH = 'sttmultires_unet.encoders.swin3d.patch_embed'
BLOCK = PATCH + '.residual_encoding.resblocks.1'
PROJECT = PATCH + '.proj'
SPECS = {
    'sn1_preview_conv1': (BLOCK + '.sn1.spiking_neuron', BLOCK + '.conv1.0', 1, 1),
    'sn2_anchor_conv2': (BLOCK + '.sn2.spiking_neuron', BLOCK + '.conv2.0', 1, 2),
    'proj_sn_spike_conv': (PROJECT + '.sn.spiking_neuron', PROJECT + '.conv', 2, 2),
}


def geometry(module):
    return dict(kernel=list(module.kernel_size), stride=list(module.stride),
                padding=list(module.padding), dilation=list(module.dilation),
                groups=int(module.groups), weight_shape=list(module.weight.shape))


def static_dimensions(ticks=10, channels=96, height=240, width=320):
    result = {}
    for label, (_, _, _, stride) in SPECS.items():
        oh, ow = (height + stride - 1) // stride, (width + stride - 1) // stride
        pairs = oh * ((ow + 1) // 2)
        result[label] = dict(source_shape=[ticks, channels, height, width],
            required_output_shape=[oh, ow], required_output_positions=oh * ow,
            paired_output_groups=pairs, k_vectors=channels * 9,
            source_slots_including_padding=ticks * channels * 9 * oh * ow,
            logical_rows_including_empty=channels * 9 * pairs)
    return result


class SharedTemporalSourceCounts:
    """Observe producer outputs and their actual convolution inputs.

    sn2_theta must come from the installed preview pair, not the replaced
    native sn2 leaf's potentially stale threshold. A zero-argument callable
    is also accepted when the caller replaces that pair between axes.
    """
    def __init__(self, modules, sn2_theta):
        self.modules, self.sn2_theta = modules, sn2_theta
        self.handles = []
        self.identities = {}
        for label, (producer, consumer, native_stride, stride) in SPECS.items():
            conv = modules[consumer]
            g = geometry(conv)
            if (g['kernel'], g['stride'], g['padding'], g['dilation'], g['groups']) != (
                    [3, 3], [native_stride] * 2, [1, 1], [1, 1], 1):
                raise ValueError('Fixed nine-offset counter does not match ' + consumer)
            self.identities[label] = dict(producer=producer, consumer=consumer,
                producer_class=type(modules[producer]).__name__, native_geometry=g,
                required_output_stride_in_source=stride,
                coefficient_scope=('First 3x3 factor only; current compiled preview has 32 useful latent rows.'
                    if label == 'sn1_preview_conv1' else
                    'First 3x3 factor only; current Conv2 has 16 latent rows, even/even outputs only.'
                    if label == 'sn2_anchor_conv2' else
                    'Actual PED spike conv; output-channel count is in native_geometry.weight_shape.'))
        continuous = geometry(modules[PROJECT + '.conv_res'])
        if (continuous['kernel'], continuous['stride'], continuous['padding'],
                continuous['dilation'], continuous['groups']) != ([1, 1], [2, 2], [0, 0], [1, 1], 1):
            raise ValueError('The declared even/even continuous PED anchors do not match conv_res')
        self.continuous_geometry = continuous
        self.reset(None)

    def reset(self, axis):
        self.axis = axis
        self.frames, self.pending, self.producers = [], {}, {}

    def _producer(self, label):
        def observe(module, inputs, output):
            theta = (self.sn2_theta() if callable(self.sn2_theta) else self.sn2_theta
                     ) if label == 'sn2_anchor_conv2' else module.thresh
            self.producers[label] = dict(pointer=output.data_ptr(),
                native_shape=list(output.shape), theta=float(theta))
        return observe

    def _consumer(self, label):
        @torch.no_grad()
        def observe(module, inputs):
            x = inputs[0]
            if x.ndim == 5 and tuple(x.shape[:2]) == (10, 1):
                local = x[:, 0]
            elif x.ndim == 4 and x.shape[0] == 10:
                local = x
            else:
                raise ValueError('Expected actual T10/B1 theta*g, got ' + str(tuple(x.shape)))
            producer = self.producers.pop(label)
            stride = SPECS[label][3]
            value = count_source(local, stride, producer['theta'])
            value.update(native_producer_shape=producer['native_shape'],
                native_consumer_input_shape=list(x.shape),
                consumer_input_same_storage_as_producer=x.data_ptr() == producer['pointer'],
                required_output_shape=value['anchor_shape'],
                source_slots_including_padding=int(local.shape[0] * local.shape[1] * 9 * value['anchors']),
                required_output_scope=('Complete output lattice' if stride == SPECS[label][2]
                                       else 'Only even/even Conv2 outputs retained by common branch deletion'))
            if label in self.pending:
                raise RuntimeError('A declared source executed twice in one frame: ' + label)
            self.pending[label] = value
            if len(self.pending) == len(SPECS):
                self.frames.append(self.pending)
                self.pending = {}
        return observe

    def install(self):
        if self.handles:
            raise RuntimeError('Source hooks are already installed')
        for label, (producer, consumer, _, _) in SPECS.items():
            self.handles.append(self.modules[producer].register_forward_hook(self._producer(label)))
            self.handles.append(self.modules[consumer].register_forward_pre_hook(self._consumer(label)))
        return self

    def restore(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.producers.clear()

    def report(self, names):
        names = list(names)
        if self.pending or self.producers or len(names) != len(self.frames):
            raise ValueError('Report requires one completed three-source frame for each name')
        totals = {}
        for label in SPECS:
            rows = [frame[label] for frame in self.frames]
            if not rows:
                continue
            fields = ('source_occurrences', 'nrv_rows', 'source_slots_including_padding',
                      'logical_rows_with_empty_rows', 'T_times_anchors', 'paired_anchor_groups')
            totals[label] = {key: sum(row[key] for row in rows) for key in fields}
            for key in ('active_columns', 'nrv_rows_by_k'):
                totals[label][key] = np.asarray([row[key] for row in rows], dtype=np.int64).sum(0).tolist()
            totals[label]['theta_g_referenced_max_abs'] = max(row['theta_g_referenced_max_abs'] for row in rows)
            totals[label]['consumer_inputs_same_storage_as_producer'] = all(
                row['consumer_input_same_storage_as_producer'] for row in rows)
        return dict(axis=self.axis, frames=[dict(file=name, sources=frame)
                    for name, frame in zip(names, self.frames)], totals=totals,
            source_identity=self.identities, continuous_PED_geometry=self.continuous_geometry,
            layout='k=((c*3)+kh)*3+kw. Pair=(output_y,floor(output_x/2)); OR T10 and the two adjacent required output positions, never different k or pairs. Padding contributes zero and does not shift global pair parity.',
            dependency='The common student deletes the entire normalized Conv2 branch at non-even/even positions. The 3x3 halo of retained even/even Conv2 outputs covers every sn2 spatial position, so the complete T10 sn2 output still requires full-spatial preview Conv1. No further weight-zero or time-dependency pruning is assumed.',
            boundaries='Observed actual theta*g input values; halo occurrences and logical T10/P2 rows only. Ranks/output lanes are not multiplied into these counts. Continuous latent V products, BN, PSNs, continuous PED conv_res, source production, code encoding, physical words/broadcast/ports and cycles are outside this helper.',
            caller_contract='Use with the existing full-network student and its nonanchor whole-BN2-branch deletion, with the PED spike branch retained. If a caller deletes that branch too, these observed software counts are diagnostic and cannot be charged as required production.',
            theta='Source output amplitude is retained and checked independently of temporal decision thresholds. sn2 amplitude is supplied by the installed preview pair.',
            static_target_dimensions=static_dimensions())


def self_test():
    rng = np.random.default_rng(60910)
    cases = 0
    for stride in (1, 2):
        for height, width in ((1, 1), (5, 7), (6, 8)):
            for kind, theta in (('zero', 1.375), ('ones', -0.625), ('random', 1.375)):
                shape = (10, 3, height, width)
                gates = (np.zeros(shape, bool) if kind == 'zero' else np.ones(shape, bool)
                         if kind == 'ones' else rng.random(shape) < 0.13)
                x = torch.from_numpy(gates.astype(np.float32) * theta)
                got = count_source(x, stride, theta)
                activity, rows = explicit_padded_reference(x, stride)
                assert np.array_equal(got['active_columns'], activity)
                assert np.array_equal(got['nrv_rows_by_k'], rows)
                assert got['theta_g_referenced_max_abs'] == 0
                cases += 1
    # The union of the retained Conv2 anchors' 3x3 source neighbourhoods.
    for height, width in ((1, 1), (5, 7), (6, 8), (240, 320)):
        needed = np.zeros((height, width), bool)
        for y in range(0, height, 2):
            for x in range(0, width, 2):
                needed[max(0, y-1):y+2, max(0, x-1):x+2] = True
        assert needed.all()
    # Exercise hooks/reset/report/restore with native 5D and flattened 4D inputs.
    modules = {}
    for _, (producer, consumer, stride, _) in SPECS.items():
        neuron = torch.nn.Identity()
        neuron.thresh = torch.tensor(1.375)
        modules[producer] = neuron
        conv = torch.nn.Conv2d(3, 3, 3, stride=stride, padding=1)
        conv.forward = lambda x: x  # Numerical convolution is outside the observer.
        modules[consumer] = conv
    modules[PROJECT + '.conv_res'] = torch.nn.Conv2d(3, 3, 1, stride=2)
    counter = SharedTemporalSourceCounts(modules, sn2_theta=-0.625).install()
    for axis in ('first', 'second'):
        counter.reset(axis)
        for _ in range(2):
            for label, (producer, consumer, _, stride) in SPECS.items():
                theta = -0.625 if label == 'sn2_anchor_conv2' else 1.375
                x = torch.from_numpy((rng.random((10, 1, 3, 5, 7)) < .13).astype(np.float32) * theta)
                value = modules[producer](x)
                modules[consumer](value.flatten(0, 1) if label == 'proj_sn_spike_conv' else value)
                got = (counter.frames[-1] if label == 'proj_sn_spike_conv' else counter.pending)[label]
                activity, rows = explicit_padded_reference(x[:, 0], stride)
                assert got['active_columns'] == activity.tolist()
                assert got['nrv_rows_by_k'] == rows.tolist()
        report = counter.report(['frame0', 'frame1'])
        assert report['axis'] == axis and len(report['frames']) == 2
        assert all(row['consumer_inputs_same_storage_as_producer'] for row in report['totals'].values())
        assert all(row['theta_g_referenced_max_abs'] == 0 for row in report['totals'].values())
    counter.restore()
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in modules.values())
    print(json.dumps(dict(cpu_self_test='PASS', scalar_reference_cases=cases,
        hook_frames=4, checks='k order, T10/P2 OR, padding and odd last pair, nonunit signed theta, dependency coverage, reset/restore, 5D/4D consumer inputs',
        static_dimensions=static_dimensions())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        parser.print_help()
