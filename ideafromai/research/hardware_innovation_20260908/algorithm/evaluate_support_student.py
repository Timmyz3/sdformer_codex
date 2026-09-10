"""Full official validation of one locally distilled support student."""
import argparse
import json
from pathlib import Path
import types

from run_bn_probe import build_model, input_frame, read_names, save_json, set_bn_mode, MLPS, SHALLOW
from run_integer_fc1_probe import install_integer_consumers
from train_exact_support_probe import Student
import numpy as np
import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser()
    for name in ('code-root', 'config', 'checkpoint', 'data', 'calibration', 'student-root', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--variant', default='forced_code')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model, _, _, _ = build_model(args)
    from spikingjelly.activation_based import functional
    stats = torch.load(args.calibration, map_location='cpu', weights_only=False)
    set_bn_mode(model, stats, SHALLOW)
    integer, _ = install_integer_consumers(model, stats)
    modules = dict(model.named_modules())
    prefix = MLPS[0]
    source, fc = modules[prefix+'sn1.spiking_neuron'], modules[prefix+'fc1']
    dictionary = torch.from_numpy(np.load(args.student_root/'dictionary.npy')).float()
    student = Student(source, fc, integer[prefix], dictionary, args.variant).cuda().eval()
    student.load_state_dict(torch.load(args.student_root/(args.variant+'_parameters.pt'),
                                      map_location='cuda', weights_only=False))
    qw = student.quantized_weight(False).detach()

    def source_forward(self, x):
        emitted, _, _ = student.source(x, differentiable=False)
        return emitted*student.theta

    def linear(self, x):
        return F.linear(x/student.theta, qw)

    source.forward = types.MethodType(source_forward, source)
    fc.forward = types.MethodType(linear, fc)
    rows = []
    with torch.no_grad():
        for i, name in enumerate(read_names(args.data, 'valid')):
            functional.reset_net(model)
            x, label, mask = input_frame(args.data, name)
            prediction = model(x)['flow'][-1]
            error = (prediction-label).square().sum(1).sqrt()
            n = int(mask.sum())
            total = float(error[mask].sum())
            row = {'file': name, 'valid_pixels': n, 'AEE': total/n, 'aee_sum': total}
            rows.append(row)
            if (i+1) % 50 == 0:
                save_json(args.output/'frames.json', rows)
                print('VALID', i+1, float(np.mean([r['AEE'] for r in rows])), flush=True)
            del x, label, mask, prediction, error
    save_json(args.output/'frames.json', rows)
    summary = {'variant': args.variant, 'frames': len(rows),
               'valid_pixels': sum(r['valid_pixels'] for r in rows),
               'AEE': float(np.mean([r['AEE'] for r in rows])),
               'student_root': str(args.student_root),
               'training': '32 training frames, 512 spatial samples per frame, 32 local FC1 teacher-distillation updates',
               'claim': 'new-model algorithm result; not frozen FP32 equivalence or RTL performance'}
    save_json(args.output/'summary.json', summary)
    print('COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
