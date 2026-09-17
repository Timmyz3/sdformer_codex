"""Optional inference-only recapture at the first forced-student source boundary.

Requires the original checkpoint/code/data. This script does not run in prepare,
does not train, and aborts each frame immediately after its source capture.
"""
import argparse
import json
from pathlib import Path
import sys

if sys.version_info[:2] != (3, 12):
    raise RuntimeError('Validation recapture requires Python 3.12 with its existing Torch environment.')
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent / 'algorithm'))
from run_bn_probe import build_model, input_frame, set_bn_mode, MLPS, SHALLOW
from run_integer_fc1_probe import install_integer_consumers
from train_exact_support_probe import Student, CaptureComplete


def main():
    p = argparse.ArgumentParser()
    for name in ['code-root', 'config', 'checkpoint', 'data', 'calibration', 'student-root']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--output', type=Path, default=HERE / 'validation_recapture.npz')
    args = p.parse_args()
    names = json.loads((args.student_root / 'run.json').read_text())['validation_files'][:2]
    model, _, _, _ = build_model(args)
    from spikingjelly.activation_based import functional
    calibration = torch.load(args.calibration, map_location='cpu', weights_only=False)
    set_bn_mode(model, calibration, SHALLOW)
    integer, _ = install_integer_consumers(model, calibration)
    modules = dict(model.named_modules())
    prefix = MLPS[0]
    source = modules[prefix + 'sn1.spiking_neuron']
    fc = modules[prefix + 'fc1']
    D = torch.from_numpy(np.load(args.student_root / 'dictionary.npy')).float()
    student = Student(source, fc, integer[prefix], D, 'forced_code').cuda().eval()
    student.load_state_dict(torch.load(args.student_root / 'forced_code_parameters.pt',
                                      map_location='cuda', weights_only=False))
    records = []
    def hook(module, inputs):
        xx = inputs[0].detach()
        projected, raw, _ = student.source(xx, differentiable=False)
        records.append(dict(X_fp32=xx.reshape(10, -1, 96).cpu().numpy(),
                            raw_g_fp32=raw.reshape(10, -1, 96).cpu().numpy().astype(np.uint8),
                            projected_g_fp32=projected.reshape(10, -1, 96).cpu().numpy().astype(np.uint8)))
        raise CaptureComplete()
    handle = source.register_forward_pre_hook(hook)
    comparisons = []
    with torch.no_grad():
        for name in names:
            functional.reset_net(model)
            x, _, _ = input_frame(args.data, name, targets=False)
            count = len(records)
            try:
                model(x)
            except CaptureComplete:
                pass
            assert len(records) == count + 1
            old = np.load(args.student_root / ('forced_code_' + name[:-4] + '_source.npz'))
            old_gate = np.unpackbits(old['gate_bits'], axis=-1, bitorder='little', count=96)
            differences = int((old_gate != records[-1]['projected_g_fp32']).sum())
            comparisons.append(dict(file=name, original_projected_gate_differences=differences,
                                    gate_values=int(old_gate.size)))
            print('CAPTURE', name, records[-1]['X_fp32'].shape, differences, flush=True)
            del x
    handle.remove()
    fields = {k: np.stack([r[k] for r in records]) for k in records[0]}
    fields.update(frame_file=np.array(names),
        **{k + '_fp32': getattr(student, k).detach().cpu().numpy()
           for k in ['A', 'bias', 'center', 'theta']},
        D=student.dictionary.detach().cpu().numpy().astype(np.uint8))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **fields)
    args.output.with_suffix('.json').write_text(json.dumps(dict(
        module=prefix + 'sn1.spiking_neuron', scope='inference-only source capture; no AEE',
        torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        comparisons=comparisons), indent=2) + '\n')


if __name__ == '__main__':
    main()
