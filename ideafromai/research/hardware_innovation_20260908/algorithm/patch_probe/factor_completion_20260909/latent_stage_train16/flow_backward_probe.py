"""One real-frame, full-network flow-gradient diagnostic for latent factors.

Default: no parameter update. Both fixed R56 layouts use the first existing
diverse validation frame; this is a gradient/forward check, not a training or
held-out result. Root launches CUDA. --self-check uses a small CPU tensor.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from adapter import LatentPair,fp32_matmul


class HardThetaGate(torch.autograd.Function):
    """Same theta-valued hard forward; existing ATLIF-style input surrogate."""
    @staticmethod
    def forward(ctx,margin,theta):
        ctx.save_for_backward(margin)
        ctx.theta=float(theta)
        return margin.ge(0).to(margin.dtype)*ctx.theta

    @staticmethod
    def backward(ctx,grad):
        (margin,)=ctx.saved_tensors
        slope=(1-(margin/max(abs(ctx.theta),1e-12)).abs()).clamp_min(0.)
        return grad*slope,None


class TrainableLatentPair(LatentPair):
    def __init__(self,arrays,device):
        super().__init__(arrays,device,conditional=False)
        self.u=nn.Parameter(self.u.detach().clone())
        self.v=nn.Parameter(self.v.detach().clone())
        self.connectivity=torch.as_tensor(arrays['connectivity'],device=device).T[:,:,None,None].bool()
        # Keep the forward Conv operands identical to the numeric adapter.
        # Group-external V slots receive no update, without an extra multiply
        # or a different forward weight layout.
        self.v.register_hook(lambda gradient: gradient*self.connectivity)

    def neuron_forward(self,y):
        t,b,c,height,width=y.shape
        if (t,b,c)!=(10,1,96) or width%4:
            raise ValueError('Expected complete T10/C96 and native P4 width.')
        out=torch.empty_like(y)
        with fp32_matmul():
            for first in range(0,height,8):
                last=min(first+8,height)
                local=y[:,0,:,first:last].reshape(10,96,last-first,width//4,4)
                local=local.permute(2,3,0,4,1).reshape(-1,10,4,96)
                margin=torch.einsum('ts,gsph->gtph',self.temporal.a,local)
                margin=margin+self.temporal.b[None,:,None,None]-self.temporal.theta
                value=HardThetaGate.apply(margin,self.temporal.theta)
                value=value.reshape(last-first,width//4,10,4,96)
                out[:,0,:,first:last]=value.permute(2,4,0,1,3).reshape(10,96,last-first,width)
        self.last_counts=dict(gates=out.numel(),mode='full hard forward / local ATLIF triangular backward')
        self.shared_raw=self.empty=None
        return out

    def parameters(self):
        return [self.u,self.v]

    def export(self,arrays):
        saved={key:value.copy() for key,value in arrays.items()}
        saved['u']=self.u.detach().reshape(self.u.shape[0],-1).T.cpu().numpy()
        saved['v']=self.v.detach()[:,:,0,0].T.cpu().numpy()
        saved['update_scope']=np.array('one diagnostic SGD step on the first diverse validation frame; not training-set recovery')
        saved['completion_statistics_valid']=np.array(False)
        saved['allowed_evaluation']=np.array('full only; conditional statistics require train recalibration after the update')
        return saved


def gradient_stats(parameter,allowed=None):
    gradient=parameter.grad
    if gradient is None:
        return dict(present=False,elements=parameter.numel(),nonzero=0)
    selected=gradient if allowed is None else gradient[allowed]
    return dict(present=True,elements=parameter.numel(),allowed_elements=selected.numel(),
        nonzero=int(torch.count_nonzero(selected)),all_finite=bool(torch.isfinite(selected).all()),
        l2=float(torch.linalg.vector_norm(selected.double())),max_abs=float(selected.abs().max()),
        forbidden_nonzero=(int(torch.count_nonzero(gradient[~allowed])) if allowed is not None else 0))


def read_arrays(filename):
    with np.load(filename) as data:
        return {key:data[key].copy() for key in data.files}


def save_json(filename,value):
    Path(filename).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def self_check():
    torch.set_num_threads(4)
    generator=torch.Generator().manual_seed(911)
    result={}
    for name in ('shared56_lambda01','shared32_private2_lambda01'):
        arrays=read_arrays(HERE/(name+'.npz'))
        reference=LatentPair(arrays,'cpu',conditional=False)
        student=TrainableLatentPair(arrays,'cpu')
        x=(torch.rand((10,1,96,4,32),generator=generator)<.12).float()*float(arrays['theta_source'])
        scale=torch.tensor(arrays['bn_scale']).float()[None,None,:,None,None]
        bias=torch.tensor(arrays['bn_bias']).float()[None,None,:,None,None]
        with torch.no_grad():
            expected=reference.neuron_forward(reference.conv_forward(x)*scale+bias)
        actual=student.neuron_forward(student.conv_forward(x)*scale+bias)
        assert torch.equal(actual,expected)
        actual.sum().backward()
        row=dict(gates=actual.numel(),forward_differences=int((actual!=expected).sum()),
            U=gradient_stats(student.u),V=gradient_stats(student.v,student.connectivity))
        assert row['U']['nonzero'] and row['V']['nonzero'] and row['U']['all_finite'] and row['V']['all_finite']
        result[name]=row
    print('CPU_SELF_CHECK',json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path)
    p.add_argument('--model-files',type=Path,nargs='+',default=[HERE/'shared56_lambda01.npz',HERE/'shared32_private2_lambda01.npz'])
    p.add_argument('--steps',type=int,choices=(0,1),default=0)
    p.add_argument('--output',type=Path,default=HERE/'flow_backward_diagnostic')
    p.add_argument('--self-check',action='store_true')
    args=p.parse_args()
    if args.self_check:
        self_check();return
    if args.root is None:
        p.error('--root is required for the real CUDA-frame diagnostic')
    sys.path.insert(0,str(args.root/'algorithm/nrv_cost_probe'))
    sys.path.insert(0,str(args.root/'algorithm'))
    sys.path.insert(0,str(args.root/'algorithm/patch_probe/joint_completion_20260909'))
    import run_probe as probe
    from run_bn_probe import input_frame
    from evaluate_stage2_deployment import CoarseReady
    from evaluate_network import BLOCK,TARGET
    from spikingjelly.activation_based import functional

    args.output.mkdir(parents=True,exist_ok=True)
    system=probe.load_system(args)
    model,modules,_,current,sources,_,_,_=system
    probe.install_sources(system,sources)
    current['count_codes']=False
    calibration=torch.load(args.root/'algorithm/patch_probe/patch_train_calibration.pt',map_location='cpu',weights_only=False)
    for name,values in calibration.items():
        bn=modules[name];bn.track_running_stats=True
        bn.running_mean=values['mean'].to(bn.weight)
        bn.running_var=values['var'].to(bn.weight)
    model.eval()
    assert not any(parameter.requires_grad for parameter in model.parameters())
    conv,neuron=modules[BLOCK+'.conv1.0'],modules[TARGET]
    original_conv,original_neuron=conv.forward,neuron.forward
    filename=json.loads((args.root/'algorithm/samples.json').read_text())['valid'][0]
    x,label,valid=input_frame(args.data,filename)
    run=dict(complete=False,file=filename,model_files=[str(f) for f in args.model_files],steps=args.steps,
        numeric='same FP32 full latent forward; only r1.sn2 backward uses an ATLIF triangular surrogate',
        parent='saved integer-S2-source/coarse-head student and the same four fixed patch BNs',
        trainable='standalone factor U/V only; group-external V gradients zero; all original network parameters frozen',
        unchanged_branches='integer source/class/compiled-neuron derivatives are not restored; downstream residual paths and original surrogates may propagate input gradients',
        sample_scope='first fixed diverse validation frame; gradient diagnosis, not a recovery-training or held-out test result',
        optional_update='steps1 is one SGD step at 1e-5 on this same diagnostic frame; saved NPZ full-only until train statistics are refitted',
        loss='mean sqrt(sum_xy(flow-GT)^2+1e-6) over the real valid pixel mask',
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,
        temporal_and_backward_matmul_TF32=False,checkpointing=False,axes={})
    save_json(args.output/'result.json',run)

    def preserve_graph(module,inputs,output):
        # The existing later hook still performs CoarseReady early exit.
        current['differentiable_flow']=output.sum(0)
    handle=modules['sttmultires_unet.preds.2'].register_forward_hook(preserve_graph,prepend=True)
    def forward_flow():
        functional.reset_net(model)
        try:
            model(x)
        except CoarseReady:
            flow=current.pop('differentiable_flow')
            current.pop('flow',None)
            return F.interpolate(flow,(480,640),mode='bilinear',align_corners=False)
        raise RuntimeError('Expected existing preds.2 coarse early exit')
    try:
        for path in args.model_files:
            started=time.monotonic()
            name=path.stem
            arrays=read_arrays(path)
            assert float(arrays['theta_source'])==float(modules[BLOCK+'.sn1.spiking_neuron'].thresh)
            assert float(arrays['theta_output'])==float(neuron.thresh)
            reference=LatentPair(arrays,conv.weight.device,conditional=False)
            conv.forward,neuron.forward=reference.conv_forward,reference.neuron_forward
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                expected=forward_flow()
            torch.cuda.synchronize()
            reference_peak=int(torch.cuda.max_memory_allocated())
            del reference
            student=TrainableLatentPair(arrays,conv.weight.device)
            conv.forward,neuron.forward=student.conv_forward,student.neuron_forward
            torch.cuda.reset_peak_memory_stats()
            allocated_start=int(torch.cuda.memory_allocated())
            predicted=forward_flow()
            torch.cuda.synchronize()
            forward_peak=int(torch.cuda.max_memory_allocated())
            difference=(predicted.detach()-expected).abs()
            row=dict(forward_flow_elements=predicted.numel(),forward_flow_differences=int(torch.count_nonzero(difference)),
                forward_flow_max_abs=float(difference.max()),forward_flow_l2=float(torch.linalg.vector_norm(difference.double())),
                reference_peak_allocated_bytes=reference_peak,training_start_allocated_bytes=allocated_start,
                train_forward_peak_allocated_bytes=forward_peak,prediction_requires_grad=predicted.requires_grad)
            error=predicted.permute(0,2,3,1)[valid]-label.permute(0,2,3,1)[valid]
            loss=torch.sqrt(error.square().sum(-1)+1e-6).mean()
            row.update(valid_pixels=int(valid.sum()),AEE=float(torch.linalg.vector_norm(error.detach(),dim=-1).double().mean()),
                robust_EPE_loss=float(loss.detach()),loss_requires_grad=loss.requires_grad)
            run['axes'][name]=row;save_json(args.output/'result.json',run)
            print('FORWARD',name,json.dumps(row),flush=True)
            if loss.requires_grad:
                with fp32_matmul():
                    loss.backward()
                torch.cuda.synchronize()
                row.update(U_gradient=gradient_stats(student.u),V_gradient=gradient_stats(student.v,student.connectivity),
                    backward_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()))
            else:
                row['backward_status']='loss has no graph; reported rather than assumed'
            if args.steps:
                if row['forward_flow_differences']:
                    row['update_status']='not applied: gradient-enabled forward differs from reference'
                elif not (student.u.grad is not None and student.v.grad is not None and
                          torch.isfinite(student.u.grad).all() and torch.isfinite(student.v.grad).all()):
                    row['update_status']='not applied: missing or nonfinite gradient'
                else:
                    with torch.no_grad():
                        student.u.add_(student.u.grad,alpha=-1e-5)
                        student.v.add_(student.v.grad,alpha=-1e-5)
                        student.v.mul_(student.connectivity)
                    destination=args.output/(name+'_diagnostic_step1.npz')
                    np.savez_compressed(destination,**student.export(arrays))
                    row['update_status']='one diagnostic SGD step applied; no post-step AEE claim'
                    row['updated_full_only_parameters']=str(destination)
            row['wall_seconds']=time.monotonic()-started
            save_json(args.output/'result.json',run)
            print('BACKWARD',name,json.dumps(row),flush=True)
            conv.forward,neuron.forward=original_conv,original_neuron
            del student,expected,predicted,difference,error,loss
            current.pop('differentiable_flow',None);current.pop('flow',None)
            functional.reset_net(model)
            gc.collect();torch.cuda.empty_cache()
        run['complete']=True
        save_json(args.output/'result.json',run)
    except Exception as exc:
        run['exception']=dict(type=type(exc).__name__,message=str(exc))
        run['peak_allocated_bytes_on_exception']=int(torch.cuda.max_memory_allocated())
        save_json(args.output/'result.json',run)
        raise
    finally:
        handle.remove()
        conv.forward,neuron.forward=original_conv,original_neuron
    print('DONE',json.dumps(run['axes']),flush=True)


if __name__=='__main__':
    main()
