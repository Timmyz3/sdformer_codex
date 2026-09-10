"""Paired full825 evaluation at the supervised penultimate readout.

Compare the integer shallow teacher with a saved temporal-code producer.
Both executions stop before the last decoder.  The student directly consumes
seven class partial sums with the compiled temporal matrix, rather than
materializing the original ten FC1 outputs.
"""
import argparse
import json
from pathlib import Path
import types

import numpy as np
import torch
import torch.nn.functional as F

from run_bn_probe import build_model, input_frame, read_names, save_json, set_bn_mode, MLPS, SHALLOW
from run_integer_fc1_probe import install_integer_consumers
from train_temporal_codes import TemporalStudent
from evaluate_coarse_readout import CoarseReady


def main():
    parser = argparse.ArgumentParser()
    for name in ('code-root', 'config', 'checkpoint', 'data', 'calibration', 'student-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--variant', default='untrained_code8')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = read_names(args.data, 'valid')
    rows = []
    for mode in ('integer_teacher', args.variant):
        model, _, _, _ = build_model(args)
        from spikingjelly.activation_based import functional
        stats = torch.load(args.calibration, map_location='cpu', weights_only=False)
        set_bn_mode(model, stats, SHALLOW)
        params, _ = install_integer_consumers(model, stats)
        modules = dict(model.named_modules())
        if mode != 'integer_teacher':
            prefix = MLPS[0]
            source, fc = modules[prefix+'sn1.spiking_neuron'], modules[prefix+'fc1']
            record = params[prefix]
            dictionary = torch.from_numpy(np.load(args.student_root/'dictionary.npy')).float()
            student = TemporalStudent(source, fc, record, dictionary).cuda().eval()
            student.load_state_dict(torch.load(args.student_root/(args.variant+'_parameters.pt'),
                                              map_location='cuda', weights_only=False))
            qw = student.weight(False).detach()
            transformed = record['temporal_int16'].cuda().double() @ student.dictionary.double().T
            tau = record['threshold_int64'].cuda().double()
            positive = record['positive_gain'].cuda()
            variable = ~record['constant_channels'].cuda()
            fixed = record['constant_gate'].cuda()
            theta_out = record['theta_output'].cuda().float()

            def forward(self, x):
                _, code, _ = student.source(x, False)
                membership = F.one_hot(code.reshape(-1, 96), num_classes=8)[:, :, 1:].float().permute(0, 2, 1)
                partials = membership @ qw.T
                u = torch.einsum('tk,pkh->tph', transformed[:, 1:], partials.double())
                gate = torch.where(positive[None, None, :], u >= tau[:, None, :], u <= tau[:, None, :])
                gate = torch.where(variable[None, None, :], gate, fixed[:, None, :])
                hidden = gate.reshape(*x.shape[:-1], 384).float()*theta_out
                out = self.fc2(hidden)
                return self.bn2(out.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)

            mlp = modules[prefix.rstrip('.')]
            mlp.forward = types.MethodType(forward, mlp)
        current = {}

        def coarse_hook(module, inputs, output):
            current['flow'] = output.detach().sum(0)
            raise CoarseReady()

        handle = modules['sttmultires_unet.preds.2'].register_forward_hook(coarse_hook)
        with torch.no_grad():
            for i, name in enumerate(names):
                functional.reset_net(model)
                x, label, mask = input_frame(args.data, name)
                try:
                    model(x)
                except CoarseReady:
                    pred = F.interpolate(current.pop('flow'), size=(480, 640),
                                         mode='bilinear', align_corners=False)
                error = (pred-label).square().sum(1).sqrt()
                count, total = int(mask.sum()), float(error[mask].sum())
                rows.append({'mode': mode, 'file': name, 'valid_pixels': count,
                             'AEE': total/count, 'sum': total})
                if (i+1) % 50 == 0:
                    print('VALID', mode, i+1, float(np.mean([r['AEE'] for r in rows if r['mode']==mode])), flush=True)
                    save_json(args.output/'frames.json', rows)
                del x, label, mask, pred, error
        handle.remove()
        save_json(args.output/'frames.json', rows)
        del model, modules
        torch.cuda.empty_cache()
    summary = {'frames_per_mode': len(names),
               'valid_pixels_per_mode': {mode: sum(r['valid_pixels'] for r in rows if r['mode']==mode)
                                         for mode in ('integer_teacher', args.variant)},
               'AEE': {mode: float(np.mean([r['AEE'] for r in rows if r['mode']==mode]))
                       for mode in ('integer_teacher', args.variant)},
               'model_change': 'both use two shallow integer FC1 and supervised preds.2 time sum plus bilinear; temporal student additionally uses train-only eight source codes in block0 and direct compiled PSN consumption',
               'claim': 'new-model full825 AEE; no RTL timing, original-model equivalence or PPA'}
    save_json(args.output/'summary.json', summary)
    print('COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
