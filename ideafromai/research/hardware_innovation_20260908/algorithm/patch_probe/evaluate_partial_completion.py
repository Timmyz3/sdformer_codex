"""Dense GPU semantic replay of selected completion policies; no timing claim.

Conv1 is physically dense here. Future columns are masked out of the decision
calculation; the separate CPU probe accounts for hypothetical sparse production.
Only the selected patch neuron changes, with the real conv2/shortcut/coarse head.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_patch_probe import probe, RES

ORDER = [0, 2, 5, 7, 8, 1, 3, 6, 4, 9]
AXES = ('exact', 'stats_individual_g4', 'stats_individual_g3', 'stats_individual_g2',
        'stats_whole_g2', 'stats_whole_g1p5', 'stats_whole_g1',
        'clip_extrema_full', 'clip_extrema_individual', 'clip99_individual')


def replay(y, zero, a, bias, theta, tm, ts, mode, gamma,
           order=None, first_batch_size=None):
    """All quantities are Float64; y[N,T,32] has intact P4/H8 contexts."""
    n = len(y)
    bits = 1 << torch.arange(10, device=y.device)
    support = a.ne(0)
    deps = (support.long()*bits).sum(1)
    lutmask = torch.arange(1024, device=y.device)[:, None].bitwise_and(bits).ne(0)
    popcount = lutmask.sum(1)
    need_lut = torch.zeros(1024,device=y.device,dtype=torch.long)
    for t in range(10):
        need_lut |= lutmask[:,t].long()*deps[t]
    full = (torch.matmul(a, y)+bias[None, :, None]) >= theta
    if mode == 'full':
        return full, 0
    seen = zero.clone()
    issued = torch.zeros(n, device=y.device, dtype=torch.bool) if first_batch_size is not None else None
    unresolved = torch.ones((n, 10, 32), device=y.device, dtype=torch.bool)
    answer = torch.zeros_like(unresolved)
    for _ in range(11):
        if not bool(unresolved.any()):
            break
        observed = seen[:, None].bitwise_and(bits).ne(0)
        partial = torch.matmul(a, y*observed[:, :, None])+bias[None, :, None]
        complete_row = deps[None].bitwise_and(~seen[:, None]).eq(0)
        predicted_margin = partial+tm[seen, :, None]-theta
        radius = gamma*ts[seen, :, None]
        confident = (predicted_margin-radius >= 0) | (predicted_margin+radius < 0)
        confident |= complete_row[:, :, None]
        if mode == 'whole':
            # Strong ordinary control: independent whole-T10 (p,h) words,
            # preserving retired words while other words share the source.
            context_done = (confident | ~unresolved).all(1)
            confident = complete_row[:, :, None].expand_as(confident).clone()
            confident |= context_done[:, None, :]
        decide = unresolved & confident
        answer = torch.where(decide, predicted_margin >= 0, answer)
        unresolved &= ~decide
        active_rows = unresolved.any(2).long().mul(bits).sum(1)
        dependency = need_lut[active_rows]
        retained = dependency.bitwise_and(seen).bitwise_and(~zero)
        needed = dependency.bitwise_and(~seen)
        free = 5-popcount[retained]
        if bool(((free <= 0) & needed.ne(0)).any()):
            raise RuntimeError('policy exceeded common five-Y capacity')
        if first_batch_size is not None:
            free = torch.where(issued, free, free.clamp(max=first_batch_size))
        batch = torch.zeros_like(seen)
        for t in (ORDER if order is None else order):
            take = needed.bitwise_and(1 << t).ne(0) & (free > 0)
            batch |= take.long() << t
            free -= take.long()
        seen |= batch
        if issued is not None:
            issued |= batch.ne(0)
    if bool(unresolved.any()):
        raise RuntimeError('incomplete replay')
    return answer, int((answer != full).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--axes', nargs='+', choices=AXES, default=['exact'])
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--full825', action='store_true',help='use the official complete valid list, ignoring short-list limit')
    parser.add_argument('--parameter-file', type=Path, help='optional JSON containing weight/bias; default is dependency/fit.json')
    parser.add_argument('--parameter-key', default='variants.row34', help='dot-separated JSON key')
    parser.add_argument('--output-directory', type=Path, help='optional isolated result directory')
    parser.add_argument('--order', nargs=10, type=int, help='fixed train-selected permutation of time columns')
    parser.add_argument('--first-batch-size', type=int, choices=range(1,6), help='first live issue per P4/H8 context; later issues use five-Y capacity')
    parser.add_argument('--deployment-source', type=Path, help='optional NPZ of actual conv/BN/neuron parameters; read-only export')
    args = parser.parse_args()
    if args.order is not None and sorted(args.order) != list(range(10)):
        parser.error('--order must be a permutation of 0..9')
    system = probe.load_system(args)
    from run_bn_probe import input_frame, read_names
    from evaluate_stage2_deployment import CoarseReady, summarize
    from spikingjelly.activation_based import functional
    model, modules, _, current, sources, _, _, _ = system
    probe.install_sources(system, sources)
    current['count_codes'] = False
    patch = args.root/'algorithm/patch_probe'
    fixed = torch.load(patch/'patch_train_calibration.pt', map_location='cpu', weights_only=False)
    for name, values in fixed.items():
        module = modules[name]
        module.track_running_stats = True
        module.running_mean = values['mean'].to(module.weight)
        module.running_var = values['var'].to(module.weight)
    neuron = modules[RES+'1.sn2.spiking_neuron']
    assert neuron.center_mode == 'zero'
    theta = float(neuron.thresh)
    parameter_file=args.parameter_file or patch/'dependency/fit.json'
    v=json.loads(parameter_file.read_text())
    for key in args.parameter_key.split('.'):
        v=v[key]
    parameter_label=args.parameter_key.split('.')[-1]
    a_np, bias_np = np.array(v['weight']), np.array(v['bias']).reshape(10)
    a, bias = torch.tensor(a_np, device='cuda'), torch.tensor(bias_np, device='cuda')
    if args.deployment_source is not None:
        conv=modules[RES+'1.conv1.0']
        bn=modules[RES+'1.norm1.norm_layer']
        source_neuron=modules[RES+'1.sn1.spiking_neuron']
        gamma=bn.weight.detach().cpu().double().numpy()
        beta=bn.bias.detach().cpu().double().numpy()
        running_mean=bn.running_mean.detach().cpu().double().numpy()
        running_var=bn.running_var.detach().cpu().double().numpy()
        bn_scale=gamma/np.sqrt(running_var+bn.eps)
        bn_bias=beta-bn_scale*running_mean
        args.deployment_source.parent.mkdir(parents=True,exist_ok=True)
        np.savez(args.deployment_source,
            weight=conv.weight.detach().cpu().numpy(),
            source_theta=np.array(float(source_neuron.thresh),dtype=np.float32),
            output_theta=np.array(theta,dtype=np.float32),
            bn_scale=bn_scale,bn_bias=bn_bias,temporal_bias=bias_np,
            bn_epsilon=np.array(bn.eps,dtype=np.float64),
            bn_affine_computation_dtype=np.array('float64 derived from loaded FP32 parameters; not FP32 BN bit-equivalence'),
            parameter_key=np.array(args.parameter_key))
        print('DEPLOYMENT_SOURCE',str(args.deployment_source),flush=True)
    stats = np.load(patch/'dependency/train_moments.npz')
    means, scales = [], []
    bits_np = 1 << np.arange(10)
    for state in range(1024):
        remain = a_np*((state & bits_np) == 0)[None]
        means.append(remain @ stats['mean'])
        scales.append(np.sqrt(np.maximum(0.,np.einsum('ij,jk,ik->i',remain,stats['covariance'],remain))))
    tm = torch.tensor(np.array(means),device='cuda')
    ts = torch.tensor(np.array(scales),device='cuda')
    bounded = json.loads((patch/'trainable_fusion/bounded_clip_result.json').read_text())['axes']
    clip_parameters={}
    for key in ('train_extrema_clip','percentile99_clip'):
        lower, upper = np.array(bounded[key]['lower']), np.array(bounded[key]['upper'])
        bmean, bradius = [], []
        for seen in range(1024):
            remain = a_np*((seen & bits_np) == 0)[None]
            lo=np.maximum(remain,0)@lower+np.minimum(remain,0)@upper
            hi=np.maximum(remain,0)@upper+np.minimum(remain,0)@lower
            bmean.append((lo+hi)*.5)
            bradius.append((hi-lo)*.5+1e-10*(1+np.maximum(np.abs(lo),np.abs(hi))))
        clip_parameters[key]=(torch.tensor(np.array(bmean),device='cuda'),
            torch.tensor(np.array(bradius),device='cuda'),
            torch.tensor(lower,device='cuda')[None,:,None],
            torch.tensor(upper,device='cuda')[None,:,None])
    state = {}

    def source_hook(module, inputs):
        spatial=inputs[0][:,0].ne(0).any(1)
        halo=F.max_pool2d(spatial[:,None].float(),3,stride=1,padding=1).bool()[:,0]
        empty=~halo.reshape(10,240,80,4).any(-1)
        word=(empty.long()*(1 << torch.arange(10,device=empty.device))[:,None,None]).sum(0)
        state['zero']=word[:,:,None].expand(240,80,12).reshape(-1)

    modules[RES+'1.conv1.0'].register_forward_pre_hook(source_hook)

    def run_neuron(self,x):
        assert tuple(x.shape)==(10,1,96,240,320)
        # Group order is native output y, native P4 x-group, consecutive H8.
        y=x[:,0].reshape(10,12,8,240,80,4).permute(3,4,1,0,2,5).reshape(-1,10,32).double()
        axis=state['axis']
        clipped=axis.startswith('clip')
        if clipped:
            btm,bts,clip_lo,clip_hi=clip_parameters['percentile99_clip' if axis.startswith('clip99') else 'train_extrema_clip']
            changed=(y<clip_lo)|(y>clip_hi)
            state['clipped_inputs']=int(changed.sum())
            y=torch.maximum(clip_lo,torch.minimum(clip_hi,y))
        mode='full' if axis in ('exact','clip_extrema_full') else ('whole' if 'whole' in axis else 'individual')
        gamma=(float(axis.rsplit('_g',1)[1].replace('p','.')) if axis.startswith('stats_') else 1.)
        gate,wrong=replay(y,state['zero'],a,bias,theta,btm if clipped else tm,bts if clipped else ts,mode,gamma,
            order=args.order,first_batch_size=args.first_batch_size)
        state['gate_difference_vs_same_full_function']=wrong
        if clipped and wrong:
            raise RuntimeError('bounded replay disagrees with complete clipped function')
        out=gate.reshape(240,80,12,10,8,4).permute(3,2,4,0,1,5).reshape_as(x)
        state['conv2_source_active_scalars']=int(out.sum())
        # Integer counts include real 3x3 boundary fanout, without FP32 sum rounding.
        fanout=state['fanout']
        state['conv2_active_terms']=int((out.sum((0,1,2),dtype=torch.int64)*fanout).sum())*96
        return out.to(x.dtype)*theta

    neuron.forward=types.MethodType(run_neuron,neuron)
    state['fanout']=F.conv2d(torch.ones(1,1,240,320,device='cuda'),
        torch.ones(1,1,3,3,device='cuda'),padding=1)[0,0].long()
    names=(read_names(args.data,'valid') if args.full825 else
           json.loads((args.root/'algorithm/samples.json').read_text())['valid'][:args.limit])
    out=args.output_directory or patch/'partial_completion'/('network_valid825' if args.full825 else 'network_valid10')
    out.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        for axis in args.axes:
            rows=[]
            started=time.monotonic()
            for i,name in enumerate(names):
                state.update(axis=axis,clipped_inputs=0)
                functional.reset_net(model)
                x,label,mask=input_frame(args.data,name)
                try:
                    model(x)
                except CoarseReady:
                    pred=F.interpolate(current.pop('flow'),(480,640),mode='bilinear',align_corners=False)
                error=torch.linalg.vector_norm(pred.permute(0,2,3,1)[mask]-label.permute(0,2,3,1)[mask],dim=1)
                total,pixels=float(error.double().sum()),error.numel()
                rows.append(dict(file=name,valid_pixels=pixels,aee_sum=total,AEE=total/pixels,
                    gate_difference_vs_same_full_function=state['gate_difference_vs_same_full_function'],
                    conv2_source_active_scalars=state['conv2_source_active_scalars'],
                    conv2_active_terms=state['conv2_active_terms'],
                    clipped_inputs=state['clipped_inputs']))
                if not args.full825 or (i+1)%50==0 or i+1==len(names):
                    progress=dict(summarize(rows,args.full825 and i+1==len(names)),axis=axis,
                        evaluation_complete=i+1==len(names),wall_seconds=time.monotonic()-started)
                    probe.save_json(out/(axis+'_summary.json'),progress)
                    probe.save_json(out/(axis+'_frames.json'),rows)
                    print('REPLAY',axis,i+1,len(names),json.dumps(progress),flush=True)
            result=dict(summarize(rows,args.full825),axis=axis,complete=True,theta=theta,
                wall_seconds=time.monotonic()-started,
                target_numeric=parameter_label+' Float64 computation from real FP32 fixed-BN input; theta*g returns FP32',
                parameter_file=str(parameter_file),parameter_key=args.parameter_key,
                time_order=ORDER if args.order is None else args.order,first_batch_size=args.first_batch_size,
                actual_connections=int(np.count_nonzero(a_np)),matrix_rank=int(np.linalg.matrix_rank(a_np)),
                target='r1.sn2 only; true conv2/norm2/shortcut and coarse full network',
                claim='dense GPU semantic/AEE reference, no sparse timing; clipping is a different model',
                clipping_fit=('train16 only, per-time 0.5%/99.5% quantiles' if axis.startswith('clip99') else
                              'train16 only, per-time extrema' if axis.startswith('clip_') else 'none'),
                statistical_fit='mu/sigma and time order use train32 only; gamma axes are selected through staged validation screening',
                selection_scope='gamma/axis selection uses diverse10 validation; full825 is exploratory validation, not an independent held-out test',
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32)
            result['conv2_active_terms_mean']=sum(r['conv2_active_terms'] for r in rows)/len(rows)
            result['conv2_source_active_scalars_mean']=sum(r['conv2_source_active_scalars'] for r in rows)/len(rows)
            if axis=='exact' and len(rows)==10 and parameter_label=='row34':
                old=json.loads((patch/'dependency/valid10_summary.json').read_text())['row34_plain']['AEE_frame_mean']
                result.update(previous_row34_FP32_AEE=old,delta_vs_previous_row34_FP32=result['AEE_frame_mean']-old)
            probe.save_json(out/(axis+'_summary.json'),result)
            probe.save_json(out/(axis+'_frames.json'),rows)
            print('SUMMARY',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
