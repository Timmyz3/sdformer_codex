"""Evaluate saved plain row34 / continuous334 students; no fit or demand mask.

Each requested axis first repeats the original diverse10. --full825 then runs
the official full validation list for every requested axis, regardless of rank
or the other axis's result. Support-count instrumentation is deliberately absent.
"""
import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from run_patch_probe import probe, RES


def source_histogram(args, system, directory, per_element=False):
    """Exact single-element or P4-OR support counts; no source tensor is saved."""
    from run_bn_probe import input_frame
    from spikingjelly.activation_based import functional
    model, modules, _, _, _, _, _, _ = system
    train = json.loads((args.root/'algorithm/direct_code_integer/run.json').read_text())['train']
    valid = json.loads((args.root/'algorithm/samples.json').read_text())['valid'][:10]
    theta = float(modules[RES+'1.sn1.spiking_neuron'].thresh)
    frames = []
    class Captured(Exception):
        pass
    def before(module, inputs):
        x = inputs[0].detach()
        t,b,c,h,w = x.shape
        assert b == 1 and (t,c,h,w) == (10,96,240,320)
        live = x.ne(0)
        word = torch.zeros((c,h,w),device=x.device,dtype=torch.int32)
        for tick in range(t):
            word.bitwise_or_(live[tick,0].to(torch.int32) << tick)
        word = F.pad(word,(1,1,1,1))
        hist = torch.zeros(1024,device=x.device,dtype=torch.int64)
        for dy in range(3):
            for dx in range(3):
                local = word[:,dy:dy+h,dx:dx+w]
                if not per_element:
                    local = local.reshape(c,h,w//4,4)
                    local = local[...,0] | local[...,1] | local[...,2] | local[...,3]
                hist.add_(torch.bincount(local.flatten().long(),minlength=1024))
        nz = x[live]
        count = c*9*h*(w if per_element else w//4)
        assert int(hist.sum()) == count
        frames.append(dict(shape=list(x.shape),theta=theta,cell_count=count,hist=hist.cpu().tolist(),
            nonzero_amplitude_min=float(nz.min()) if nz.numel() else None,
            nonzero_amplitude_max=float(nz.max()) if nz.numel() else None))
        raise Captured()
    handle = modules[RES+'1.conv1.0'].register_forward_pre_hook(before)
    for split,names in [('train',train),('valid',valid)]:
        for i,name in enumerate(names):
            functional.reset_net(model)
            x,_,_ = input_frame(args.data,name,targets=False)
            try:
                model(x)
            except Captured:
                frames[-1].update(file=name,split=split)
            if (i+1)%8 == 0 or i+1 == len(names):
                print('SOURCE_ELEMENT_HISTOGRAM' if per_element else 'SOURCE_HISTOGRAM',split,i+1,flush=True)
    handle.remove()
    totals = {split:[sum(row['hist'][k] for row in frames if row['split']==split) for k in range(1024)]
              for split in ('train','valid')}
    filename = 'source_element_histogram.json' if per_element else 'source_batch_histogram.json'
    probe.save_json(directory/filename,dict(complete=True,frames=frames,
        train_hist=totals['train'],valid_hist=totals['valid'],pattern_bit='bit t is PSN input time t',
        cell=('one input channel, one 3x3 kernel offset, one output y/x; no P4 OR' if per_element else
              'one input channel, one 3x3 kernel offset, one output y, one aligned P4 output group; OR of four source supports'),
        padding='zero source outside 240x320; padded cells included in per-frame cell_count',
        parent='same fixed-four-patch-BN parent; source captured before r1.conv1; original sn2 restored',
        claim=('sum(hist[p] * popcount(p)) * 96 reconstructs conv1 active terms; no transformed-source network or AEE evaluation' if per_element else
               'logical weight-vector request count for a time batch B is sum(hist[p] for p if p & B); not cycles or bank transactions')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--axis', choices=('row34', 'continuous334', 'both'), default='both')
    parser.add_argument('--full825', action='store_true')
    parser.add_argument('--external-json', type=Path, nargs='+',
        help='additional saved A/bias controls, evaluated in their original time labels')
    parser.add_argument('--source-histogram', action='store_true')
    parser.add_argument('--source-element-only', action='store_true',
        help='only collect train32+valid10 single-position source patterns; do not evaluate AEE')
    args = parser.parse_args()
    axes = ['row34', 'continuous334'] if args.axis == 'both' else [args.axis]
    system = probe.load_system(args)
    from run_bn_probe import input_frame, read_names
    from evaluate_stage2_deployment import CoarseReady, summarize
    from spikingjelly.activation_based import functional
    model, modules, _, current, sources, _, _, _ = system
    probe.install_sources(system, sources)
    current['count_codes'] = False
    patch = args.root/'algorithm/patch_probe'
    source = patch/'dependency'
    out = source/'full825'
    if args.external_json:
        out = out/'external_controls'
    out.mkdir(parents=True, exist_ok=True)
    fixed = torch.load(patch/'patch_train_calibration.pt', map_location='cpu', weights_only=False)
    for name, values in fixed.items():
        module = modules[name]
        module.track_running_stats = True
        module.running_mean = values['mean'].to(module.weight)
        module.running_var = values['var'].to(module.weight)
    if args.source_element_only:
        with torch.no_grad():
            source_histogram(args,system,source,per_element=True)
        return
    saved = torch.load(source/'parameters.pt', map_location='cpu', weights_only=False)
    fit = json.loads((source/'fit.json').read_text())['variants']
    if args.external_json:
        saved, fit = {}, {}
        for path in args.external_json:
            document = json.loads(path.read_text())
            variants = document['variants'] if 'variants' in document else {document['name']:document}
            for name, variant in variants.items():
                if 'left_factor_fp32' in variant:
                    saved[name] = dict(weight=torch.tensor(variant['reconstructed_weight_fp32']),
                        bias=torch.tensor(variant['bias_fp32']),left=torch.tensor(variant['left_factor_fp32']),
                        right=torch.tensor(variant['right_factor_fp32']),rank=variant['rank'])
                else:
                    saved[name] = {key:torch.tensor(variant[key], dtype=torch.float32) for key in ('weight','bias')}
                fit[name] = variant
        axes = list(saved)
    original10 = json.loads((source/'valid10_summary.json').read_text())
    parent = json.loads((patch/'fixed_bn_valid825_summary.json').read_text())
    neuron = modules[RES+'1.sn2.spiking_neuron']
    native_weight, native_bias = neuron.weight.detach().clone(), neuron.bias.detach().clone()
    diverse10 = json.loads((args.root/'algorithm/samples.json').read_text())['valid'][:10]
    full = read_names(args.data, 'valid') if args.full825 else []
    run = dict(axes=axes, requested_full825=args.full825, fixed_BN=list(fixed),
        parameter_file=[str(p) for p in args.external_json] if args.external_json else str(source/'parameters.pt'),
        target=RES+'1.sn2.spiking_neuron',
        theta=float(neuron.thresh), output_mode=neuron.output_mode,
        threshold_mode=neuron.threshold_mode, center_mode=neuron.center_mode,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,
        diverse10=diverse10, fitting='none; saved train32 moment-fit A/bias only',
        masks='none; plain complete T10 outputs and all real conv2/BN2/shortcut consumers',
        parent_full825_AEE=parent['AEE_frame_mean'], claim='algorithm AEE only, no hardware timing',
        external_time_labels='JSON A rows/columns and bias already use original t; scheduling input_order is not reapplied')
    run_name = 'external_controls' if args.external_json else args.axis
    probe.save_json(out/('run_'+run_name+'.json'),run)
    results = {}
    with torch.no_grad():
        for split, names in [('valid10', diverse10)]+([('valid825',full)] if full else []):
            for axis in axes:
                neuron.temporal_factor_rank = saved[axis].get('rank',0)
                if neuron.temporal_factor_rank:
                    neuron.temporal_factor_left = torch.nn.Parameter(saved[axis]['left'].to(neuron.weight),requires_grad=False)
                    neuron.temporal_factor_right = torch.nn.Parameter(saved[axis]['right'].to(neuron.weight),requires_grad=False)
                neuron.weight.copy_(saved[axis]['weight'].to(neuron.weight))
                neuron.bias.copy_(saved[axis]['bias'].reshape_as(neuron.bias).to(neuron.bias))
                rows, started = [], time.monotonic()
                for i,name in enumerate(names):
                    functional.reset_net(model)
                    x,label,mask = input_frame(args.data,name)
                    try:
                        model(x)
                    except CoarseReady:
                        pred = F.interpolate(current.pop('flow'),(480,640),mode='bilinear',align_corners=False)
                    error = torch.linalg.vector_norm(pred.permute(0,2,3,1)[mask]
                        -label.permute(0,2,3,1)[mask],dim=1)
                    total,pixels = float(error.double().sum()),error.numel()
                    rows.append(dict(file=name, valid_pixels=pixels,aee_sum=total,AEE=total/pixels))
                    if (i+1)%50 == 0 or i+1 == len(names):
                        summary = dict(**summarize(rows,split=='valid825' and i+1==len(names)),
                            axis=axis, split=split, evaluation_complete=i+1==len(names),
                            wall_seconds=time.monotonic()-started,
                            matrix_rank=fit[axis].get('matrix_rank',fit[axis].get('rank')),
                            actual_connections=fit[axis].get('actual_connections'),
                            factor_coefficients=fit[axis].get('factor_coefficients'),
                            numerical_execution='R@x then L@latent+bias' if neuron.temporal_factor_rank else 'sparse A stored in dense GPU reference',
                            claim='new FP32 student; complete T10 output; no demand mask or sparse timing')
                        if i+1 == len(names):
                            if split == 'valid10' and axis+'_plain' in original10:
                                old = original10[axis+'_plain']['AEE_frame_mean']
                                summary['previous_valid10_AEE'] = old
                                summary['delta_vs_previous_valid10'] = summary['AEE_frame_mean']-old
                            elif split == 'valid825':
                                summary['parent_fixed_patch_BN_AEE'] = parent['AEE_frame_mean']
                                summary['delta_vs_fixed_patch_BN_parent'] = summary['AEE_frame_mean']-parent['AEE_frame_mean']
                        probe.save_json(out/(axis+'_'+split+'_summary.json'),summary)
                        probe.save_json(out/(axis+'_'+split+'_frames.json'),rows)
                        print('PROGRESS',axis,split,i+1,json.dumps(summary),flush=True)
                    del x,label,mask,pred,error
                results[axis+'_'+split] = summary
        if args.source_histogram:
            neuron.temporal_factor_rank = 0
            neuron.weight.copy_(native_weight)
            neuron.bias.copy_(native_bias)
            source_histogram(args,system,source)
    run.update(complete=True,results=results)
    probe.save_json(out/('run_'+run_name+'.json'),run)


if __name__ == '__main__':
    main()
