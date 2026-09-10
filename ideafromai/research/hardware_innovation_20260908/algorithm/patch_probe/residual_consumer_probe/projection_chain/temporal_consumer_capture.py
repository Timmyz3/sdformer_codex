"""Small actual-consumer capture for the fixed one-check reconstruction probe.

Select 64 uniformly spaced native P2 groups, two even/even anchors per group.
Capture only values already produced by the chosen coordinate function and
the real consumer gates. No forward arithmetic or sampling based on gates.
Full-frame ranges remain in the evaluator's separate coordinate range file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class TemporalConsumerCapture:
    def __init__(self, controller, coordinates, modules, directory):
        self.c, self.coordinates = controller, coordinates
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rows, self.current = [], {}
        groups = np.linspace(0, 120*80-1, 64, dtype=np.int64)
        self.y = np.repeat(2*(groups//80), 2)
        self.x = (4*(groups%80)[:, None]+np.array([0, 2])).reshape(-1)
        self.locations = np.stack((self.y, self.x), axis=-1).reshape(64, 2, 2)
        coordinates.capture_sink = self.observe
        self.handle = controller.consumer.register_forward_hook(self.finish_frame)

        bn = controller.bn
        gain = bn.weight.detach()/torch.sqrt(bn.running_var+bn.eps)
        offset = bn.bias.detach()-gain*bn.running_mean
        conv_bias = controller.rank.bias
        folded = offset if conv_bias is None else offset+gain*conv_bias
        values = dict(As=controller.source.weight, Ap_reference=controller.consumer.weight,
            F_fp32=gain[:, None]*controller.conv_v[:, :, 0, 0], BN_offset_fp32=folded,
            U=controller.u[:, :, 0, 0], V=controller.v[:, :, 0, 0],
            conv2_U=controller.conv_u, conv2_V=controller.conv_v,
            source_bias=controller.source.bias, consumer_bias=controller.consumer.bias,
            source_theta=controller.source.thresh, consumer_theta=controller.consumer.thresh,
            source_center=controller.source.center, consumer_center=controller.consumer.center,
            projection_bias_delta=controller.projection_bias_delta)
        if controller.original['projection_bias'] is not None:
            values['projection_base_bias'] = controller.original['projection_bias']
        if controller.shared_parameterization is not None:
            p = controller.shared_parameterization
            values.update(shared_d=p.d, permutation=p.permutation)
        if controller.raw_parameterization is not None:
            p = controller.raw_parameterization
            values.update(raw_e=p.e, permutation=p.permutation)
            if p.residual:
                values.update(raw_L=p.left, raw_R=p.right)
        project = 'sttmultires_unet.encoders.swin3d.patch_embed.proj'
        values['spike_conv_weight'] = modules[project+'.conv'].weight
        self.parameters = {key: value.detach().cpu().numpy() if torch.is_tensor(value)
                           else np.asarray(value) for key, value in values.items()}
        self.metadata = dict(mode=coordinates.metadata['mode'],
            source_center_mode=controller.source.center_mode,
            consumer_center_mode=controller.consumer.center_mode,
            selection='64 linearly spaced native T10/P2 anchor groups out of9600; two adjacent required outputs per group; fixed before observing values.',
            shape='Captured tensors [T10,C,group64,P2], locations[group64,P2,(source_y,source_x)].',
            scope='Actual values and gates from this FP32 coordinate function. Any new rearranged or fixed-point sum and stopping rule still require an explicit numeric definition and fresh function checks; this is not a cycle trace.')

    def observe(self, name, value):
        value = value.detach()
        if value.ndim == 5:
            value = value[:, 0]
        if value.ndim == 2:
            value = value.reshape(10, 96, 240, 320)
        # The coordinate helper reports either full240x320 or anchor120x160.
        scale = 240//value.shape[-2]
        y = torch.as_tensor(self.y//scale, device=value.device)
        x = torch.as_tensor(self.x//scale, device=value.device)
        sampled = value[:, :, y, x].reshape(10, value.shape[1], 64, 2)
        self.current[name] = sampled.cpu().numpy().copy()

    def finish_frame(self, module, inputs, output):
        self.observe('actual_consumer_input', inputs[0])
        self.observe('actual_consumer_theta_g', output)
        self.rows.append(self.current)
        self.current = {}

    def save(self, names):
        np.savez_compressed(self.directory/'parameters.npz', **self.parameters,
                            locations=self.locations)
        files = []
        for i, (name, row) in enumerate(zip(names, self.rows)):
            filename = f'{i:02d}_{Path(name).stem}.npz'
            np.savez_compressed(self.directory/filename, **row)
            files.append(dict(frame=name, capture=filename, fields=list(row)))
        (self.directory/'capture.json').write_text(json.dumps(
            dict(**self.metadata, frames=files), indent=2)+'\n')

    def restore(self):
        self.coordinates.capture_sink = None
        self.handle.remove()
