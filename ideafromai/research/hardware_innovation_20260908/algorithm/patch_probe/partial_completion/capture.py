"""Capture native horizontal P4 groups at fixed-BN patch r1, without fitting.

The network stops at sn2's input after computing the real conv1 and fixed BN.
Source masks retain real 3x3 offsets; no separated positions are called a group.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_patch_probe import probe, RES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    system = probe.load_system(args)
    from run_bn_probe import input_frame
    from spikingjelly.activation_based import functional
    model, modules, _, current, sources, _, _, _ = system
    probe.install_sources(system, sources)
    current['count_codes'] = False
    patch = args.root/'algorithm/patch_probe'
    out = patch/'partial_completion'
    out.mkdir(parents=True, exist_ok=True)
    fixed = torch.load(patch/'patch_train_calibration.pt', map_location='cpu', weights_only=False)
    for name, values in fixed.items():
        module = modules[name]
        module.track_running_stats = True
        module.running_mean = values['mean'].to(module.weight)
        module.running_var = values['var'].to(module.weight)
    train = json.loads((args.root/'algorithm/direct_code_integer/run.json').read_text())['train'][:16]
    valid = json.loads((args.root/'algorithm/samples.json').read_text())['valid'][:4]
    # Raster group-ID is y*80 + x//4.  All four positions remain native neighbors.
    group_ids = torch.linspace(0, 240*80-1, 64, device='cuda').round().long()
    ys = group_ids//80
    xs = (group_ids%80)*4
    positions = ys[:, None]*320+xs[:, None]+torch.arange(4, device='cuda')[None]
    yy = ys[:, None].expand(64, 4)
    xx = xs[:, None]+torch.arange(4, device='cuda')[None]
    pending, rows = {}, []
    theta = float(modules[RES+'1.sn1.spiking_neuron'].thresh)

    class Captured(Exception):
        pass

    def source_hook(module, inputs):
        x = inputs[0].detach()
        assert tuple(x.shape) == (10, 1, 96, 240, 320)
        g = x[:, 0].ne(0)
        padded = F.pad(g, (1, 1, 1, 1))
        counts = torch.zeros((10, 64, 4), device=x.device, dtype=torch.int32)
        masks = torch.zeros((64, 96, 3, 3), device=x.device, dtype=torch.int32)
        for kh in range(3):
            for kw in range(3):
                support = padded[:, :, yy+kh, xx+kw]  # [T,C,G,P]
                counts.add_(support.sum(1, dtype=torch.int32))
                word = torch.zeros((96, 64), device=x.device, dtype=torch.int32)
                for t in range(10):
                    word.bitwise_or_(support[t].any(-1).int() << t)
                masks[:, :, kh, kw] = word.T
        # Check actual theta*g rather than silently replacing arbitrary payloads.
        sampled = x[:, 0].flatten(2)[:, :, positions]
        residual = (sampled-theta*sampled.ne(0)).abs().max()
        pending.update(source_active_terms=counts.permute(1, 2, 0).cpu().short(),
                       source_words=masks.reshape(64, 864).cpu().short(),
                       sampled_source_theta_residual=float(residual))

    def neuron_input_hook(module, inputs):
        x = inputs[0].detach()
        assert tuple(x.shape) == (10, 1, 96, 240, 320)
        pending['Y'] = x[:, 0].flatten(2)[:, :, positions].cpu()
        raise Captured()

    h1 = modules[RES+'1.conv1.0'].register_forward_pre_hook(source_hook)
    h2 = modules[RES+'1.sn2.spiking_neuron'].register_forward_pre_hook(neuron_input_hook)
    started = time.monotonic()
    with torch.no_grad():
        for split, names in [('train', train), ('valid', valid)]:
            for i, name in enumerate(names):
                pending.clear()
                functional.reset_net(model)
                x, _, _ = input_frame(args.data, name, targets=False)
                try:
                    model(x)
                except Captured:
                    rows.append(dict(file=name, split=split, **pending))
                del x
                print('CAPTURE', split, i+1, '/', len(names), 'seconds',
                      round(time.monotonic()-started, 2), flush=True)
    h1.remove()
    h2.remove()
    neuron = modules[RES+'1.sn2.spiking_neuron']
    metadata = dict(complete=True, parent='saved integer-source/coarse student with all four patch BN fixed to train32 statistics',
                    operation='read-only parent forward; stop before r1.sn2; no AEE or learned prediction',
                    train=train, valid=valid, source_theta=theta,
                    Y_shape=[10, 96, 64, 4], Y_semantics='real norm1 output before r1.sn2; T,C,group,within-P4',
                    source_words_shape=[64, 864],
                    source_words_order='k=((c*3)+kh)*3+kw; bit t = any source at the four output positions at input time t',
                    source_coordinates='input (y+kh-1,x+kw-1), zero outside 240x320',
                    active_count_shape=[64, 4, 10], active_count_unit='active source (c,kh,kw) terms for one output channel; multiply by output-channel count when applicable',
                    source_storage_dtype='int16, nonnegative values 0..1023',
                    neuron_theta=float(neuron.thresh), center_mode=neuron.center_mode,
                    threshold_mode=neuron.threshold_mode, output_mode=neuron.output_mode,
                    tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                    tf32_cudnn=torch.backends.cudnn.allow_tf32,
                    wall_seconds=time.monotonic()-started)
    artifact = dict(metadata=metadata, groups=group_ids.cpu(), positions=positions.cpu(),
                    group_y=ys.cpu(), group_x_start=xs.cpu(), samples=rows,
                    neuron_state={k:v.detach().cpu() for k,v in neuron.state_dict().items()})
    torch.save(artifact, out/'capture.pt')
    metadata['file_bytes'] = (out/'capture.pt').stat().st_size
    metadata['sampled_source_theta_residual_max'] = max(r['sampled_source_theta_residual'] for r in rows)
    probe.save_json(out/'capture.json', metadata)
    print('DONE', json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
