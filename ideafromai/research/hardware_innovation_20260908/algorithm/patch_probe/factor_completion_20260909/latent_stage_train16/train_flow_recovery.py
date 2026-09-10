"""Six fixed real-flow recovery controls; root launches CUDA separately.

shared56/private56/shared48 x FP32/(dyadic-U8 + DeepShift-Q5 V).
Exactly64 Adam updates at1e-4, same explicit training-frame schedule, U/V only.
No validation run, Y/gate distillation, request loss, rank/LR/step sweep, or
restoration of the frozen integer branches' derivatives.
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
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
FACTOR=HERE.parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(FACTOR/'shift_consumer_train16'))
from flow_backward_probe import TrainableLatentPair,read_arrays,gradient_stats,save_json
from adapter import LatentPair,fp32_matmul
from train_shift_consumer import Pow2STE

STEPS=64
LR=1e-4
INITIAL=(('shared56','shared56_lambda01.npz',512),
         ('private56','shared32_private2_lambda01.npz',512),
         ('shared48','shared_compact48.npz',256))


class DyadicRoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,step):
        return (value/step).round().clamp(-127,127)*step

    @staticmethod
    def backward(ctx,upstream):
        return upstream,None


class CalibrationReady(Exception):
    pass


class RecoveryPair(TrainableLatentPair):
    def __init__(self,arrays,device,quantized):
        super().__init__(arrays,device)
        self.quantized=bool(quantized)
        self.statistics=None
        if self.source_theta<=0:
            raise ValueError('This saved source theta is positive and cannot be silently omitted.')
        self.project()

    @torch.no_grad()
    def project(self):
        self.v.mul_(self.connectivity)
        if self.quantized:
            # Same already-written DeepShift-Q math; exact structural zero
            # stays zero instead of being replaced with a minimum magnitude.
            self.v.copy_(self.v.sign()*self.v.abs().clamp(2**-15,1.))

    def effective(self):
        if not self.quantized:
            return self.u,self.v
        folded=self.u*self.source_theta
        maximum=folded.detach().abs().flatten(1).amax(1)
        # Weight-only scale, recomputed from current shadow with no gradient
        # through exponent selection. Zero latent columns get scale1.
        exponent=torch.ceil(torch.log2((maximum/127).clamp_min(2**-40)))
        step=torch.where(maximum>0,torch.exp2(exponent),torch.ones_like(maximum))
        effective=DyadicRoundSTE.apply(folded,step[:,None,None,None])
        # Existing input remains theta*g. This restores its amplitude
        # convention; an integer implementation instead consumes g and the
        # separately exported folded U codes/scales.
        raw=effective/self.source_theta
        return raw,Pow2STE.apply(self.v)

    @torch.no_grad()
    def start_statistics(self):
        self.statistics=dict(count=0,
            sum=torch.zeros(10,96,dtype=torch.float64,device=self.u.device),
            cross=torch.zeros(96,10,10,dtype=torch.float64,device=self.u.device))

    @torch.no_grad()
    def add_statistics(self,tail):
        # Full spatial domain, not the earlier64 sampled P4 groups. Only
        # private post-BN contribution is uncertain; no BN bias here.
        scale=self.temporal.bn_scale
        for h in range(0,96,8):
            local=(tail[:,h:h+8]*scale[h:h+8][None,:,None,None]).double().flatten(2)
            self.statistics['sum'][:,h:h+8]+=local.sum(-1)
            values=local.permute(1,0,2)
            self.statistics['cross'][h:h+8]+=values@values.transpose(1,2)
        self.statistics['count']+=tail.shape[-2]*tail.shape[-1]

    def conv_forward(self,x):
        if x.shape[:3]!=(10,1,96):
            raise ValueError('Expected full T10,B1,C96 patch source.')
        u,v=self.effective()
        # During final train-only calibration, the upstream network is
        # unchanged and this private linear branch fully determines its
        # moments. Stop here instead of needlessly running all successors.
        if self.statistics is not None:
            # Keep the identical all-R Conv call/layout used in the full
            # student; only the unneeded shared V and successors are omitted.
            z=F.conv2d(x.flatten(0,1),u,padding=1)
            tail=F.conv2d(z[:,self.shared_rank:],v[:,self.shared_rank:])
            self.add_statistics(tail)
            raise CalibrationReady()
        z=F.conv2d(x.flatten(0,1),u,padding=1)
        shared=F.conv2d(z[:,:self.shared_rank],v[:,:self.shared_rank])
        tail=F.conv2d(z[:,self.shared_rank:],v[:,self.shared_rank:])
        self.shared_raw=shared.reshape(10,1,96,*shared.shape[-2:])
        # Full execution does not consult the empty-source predictor.
        self.empty=None
        return (shared+tail).reshape_as(self.shared_raw)

    @torch.no_grad()
    def finish_statistics(self):
        count=self.statistics['count']
        mean=self.statistics['sum']/count
        covariance=self.statistics['cross']/count-mean.T[:,:,None]*mean.T[:,None,:]
        self.statistics=None
        return dict(mean=mean.float().cpu().numpy(),covariance=covariance.float().cpu().numpy(),count=count)

    @torch.no_grad()
    def export_recovery(self,arrays,moments,names):
        result={key:value.copy() for key,value in arrays.items()}
        u,v=self.effective()
        result.update(u=u.reshape(u.shape[0],-1).T.cpu().numpy(),v=v[:,:,0,0].T.cpu().numpy(),
            u_shadow=self.u.reshape(self.u.shape[0],-1).T.cpu().numpy(),
            v_shadow=self.v[:,:,0,0].T.cpu().numpy(),
            completion_mean=moments['mean'],completion_covariance=moments['covariance'],
            completion_statistics_valid=np.array(True),completion_train_files=np.array(names),
            completion_spatial_observations=np.array(moments['count']),
            recovery_updates=np.array(STEPS),recovery_lr=np.array(LR),
            recovery_loss=np.array('actual full-network GT robust EPE; no Y/gate/request loss'),
            u_weight_bits=np.array(8 if self.quantized else 32),v_weight_bits=np.array(5 if self.quantized else 32),
            numeric_scope=np.array('FP32 dequantized factor network; no integer activation/accumulator deployment claim'))
        if self.quantized:
            folded=self.u*self.source_theta
            maximum=folded.abs().flatten(1).amax(1)
            exponent=torch.ceil(torch.log2((maximum/127).clamp_min(2**-40)))
            exponent=torch.where(maximum>0,exponent,torch.zeros_like(exponent))
            scale=torch.exp2(exponent)
            code=(folded/scale[:,None,None,None]).round().clamp(-127,127).to(torch.int8)
            vv=result['v'];nz=vv!=0
            shift=np.zeros(vv.shape,np.int8)
            shift[nz]=np.rint(np.log2(np.abs(vv[nz]))).astype(np.int8)
            result.update(u_int8=code.reshape(code.shape[0],-1).T.cpu().numpy(),
                u_dyadic_scale=scale.cpu().numpy(),u_scale_exponent=exponent.to(torch.int16).cpu().numpy(),
                u_quantization=np.array('RNE(theta_source*U/scale), symmetric INT8[-127,127]; per-latent positive dyadic scale; u field restores /theta_source'),
                v_shift=shift,v_sign=np.sign(vv).astype(np.int8),v_nonzero=nz,
                v_quantization=np.array('existing DeepShift-Q5 weight-only math: sign*2^round(log(abs(V))/log(2)), p=-15..0; structural zeros fixed'))
        return result


def read_train_list(path):
    text=Path(path).read_text()
    if Path(path).suffix.lower()=='.json':
        names=json.loads(text)
        if isinstance(names,dict):
            names=names['train']
    else:
        names=[line.strip() for line in text.splitlines() if line.strip()]
    if not isinstance(names,list) or not names or any(not isinstance(name,str) for name in names):
        raise ValueError('Explicit train list: JSON string list, JSON {train:[...]}, or one filename per line.')
    if len(set(names))!=len(names):
        raise ValueError('Train-list entries must be unique; the fixed64-step schedule repeats them explicitly.')
    return names


def self_check():
    torch.set_num_threads(4)
    generator=torch.Generator().manual_seed(912)
    result={}
    for name,file,_ in INITIAL:
        arrays=read_arrays(HERE/file)
        x=(torch.rand((10,1,96,4,32),generator=generator)<.12).float()*float(arrays['theta_source'])
        scale=torch.tensor(arrays['bn_scale']).float()[None,None,:,None,None]
        bias=torch.tensor(arrays['bn_bias']).float()[None,None,:,None,None]
        for quantized in (False,True):
            pair=RecoveryPair(arrays,'cpu',quantized)
            dummy=dict(mean=arrays['completion_mean'],covariance=arrays['completion_covariance'],count=1)
            exported=pair.export_recovery(arrays,dummy,['CPU_SELF_CHECK_ONLY'])
            reference=LatentPair(exported,'cpu',conditional=False)
            with torch.no_grad():
                expected=reference.neuron_forward(reference.conv_forward(x)*scale+bias)
            actual=pair.neuron_forward(pair.conv_forward(x)*scale+bias)
            assert torch.equal(actual,expected)
            actual.sum().backward()
            gu,gv=gradient_stats(pair.u),gradient_stats(pair.v,pair.connectivity)
            assert gu['nonzero'] and gv['nonzero'] and gu['all_finite'] and gv['all_finite']
            record=dict(gates=actual.numel(),exported_forward_gate_differences=int((actual!=expected).sum()),
                U_gradient=gu,V_gradient=gv,forbidden_V_nonzeros=int(np.count_nonzero(exported['v'][~arrays['connectivity']])))
            if quantized:
                reconstructed=exported['u_int8'].astype(np.float32)*exported['u_dyadic_scale'][None,:]/float(arrays['theta_source'])
                record.update(U_export_exact=bool(np.array_equal(reconstructed,exported['u'])),
                    V_dyadic_exact=bool(np.array_equal(np.ldexp(exported['v_sign'].astype(np.float32),exported['v_shift']),exported['v'])))
                assert record['U_export_exact'] and record['V_dyadic_exact']
            result[name+('_u8_vq5' if quantized else '_fp32')]=record
    print('CPU_SELF_CHECK',json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path)
    p.add_argument('--train-list',type=Path)
    p.add_argument('--output',type=Path,default=HERE/'flow_recovery64')
    p.add_argument('--self-check',action='store_true')
    args=p.parse_args()
    if args.self_check:
        self_check();return
    if args.root is None or args.train_list is None:
        p.error('--root and an explicit --train-list are required; there is no validation-data default')
    names=read_train_list(args.train_list)
    sys.path.insert(0,str(args.root/'algorithm/nrv_cost_probe'))
    sys.path.insert(0,str(args.root/'algorithm'))
    sys.path.insert(0,str(args.root/'algorithm/patch_probe/joint_completion_20260909'))
    import run_probe as probe
    from run_bn_probe import input_frame,read_names
    from evaluate_stage2_deployment import CoarseReady
    from evaluate_network import BLOCK,TARGET
    from spikingjelly.activation_based import functional
    system=probe.load_system(args)
    model,modules,_,current,sources,_,_,_=system
    probe.install_sources(system,sources)
    current['count_codes']=False
    official_valid=set(read_names(args.data,'valid'))
    overlap=sorted(set(names)&official_valid)
    if overlap:
        raise ValueError('The supplied recovery list overlaps official validation filenames: '+', '.join(overlap))
    fixed=torch.load(args.root/'algorithm/patch_probe/patch_train_calibration.pt',map_location='cpu',weights_only=False)
    for name,values in fixed.items():
        bn=modules[name];bn.track_running_stats=True
        bn.running_mean=values['mean'].to(bn.weight);bn.running_var=values['var'].to(bn.weight)
    model.eval()
    assert not any(parameter.requires_grad for parameter in model.parameters())
    conv,neuron=modules[BLOCK+'.conv1.0'],modules[TARGET]
    original_conv,original_neuron=conv.forward,neuron.forward
    generator=np.random.default_rng(912)
    order=[]
    while len(order)<STEPS:
        order.extend(generator.permutation(len(names)).tolist())
    schedule=[names[i] for i in order[:STEPS]]
    args.output.mkdir(parents=True,exist_ok=True)
    run=dict(complete=False,train_list=str(args.train_list),train_files=names,train_schedule=schedule,
        official_valid_filename_overlap=overlap,
        fixed=dict(updates=STEPS,optimizer='Adam',lr=LR,seed=912,batch_frames=1,
            loss='mean sqrt(sum_xy(flow-GT)^2+1e-6) at real valid GT pixels',
            trainable='U/V only; connectivity projected, common3 A/b/theta and all other weights fixed',
            request_loss=False,local_Y_or_gate_loss=False,global_model_mode='eval with the same four fixed patch BNs'),
        quantization=dict(U='per-latent RNE symmetric INT8[-127,127] of theta_source*U; detached positive dyadic max scale from current shadow, restored /theta for existing x input',
            V='existing independently written DeepShift-Q5: nonzero magnitude projection[2^-15,1], round(log(abs)/log2), sign; identity STE; structural zeros fixed',
            Z='FP32 dequantized numeric reference, no accumulator/activation integer claim',
            scale_updates='weight-only scale recomputed per forward; no validation input calibration'),
        backward='only r1.sn2 adds existing ATLIF-style triangular surrogate; downstream original integer branches remain nondifferentiable, residual/native paths carry gradients',
        statistics='after update64 only supplied train frames, complete spatial-domain private post-BN residual moments; float64 accumulation, FP32 export; no validation fitting',
        evaluation='no validation run or final AEE claim in this trainer',
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,
        temporal_and_backward_matmul_TF32=False,checkpointing=False,axes={})
    save_json(args.output/'result.json',run)
    def preserve_graph(module,inputs,output):
        current['differentiable_flow']=output.sum(0)
    handle=modules['sttmultires_unet.preds.2'].register_forward_hook(preserve_graph,prepend=True)
    def forward_flow(x):
        functional.reset_net(model)
        try:
            model(x)
        except CoarseReady:
            flow=current.pop('differentiable_flow');current.pop('flow',None)
            return F.interpolate(flow,(480,640),mode='bilinear',align_corners=False)
        raise RuntimeError('Expected existing coarse early exit')
    try:
        for name,file,previous_updates in INITIAL:
            arrays=read_arrays(HERE/file)
            assert float(arrays['theta_source'])==float(modules[BLOCK+'.sn1.spiking_neuron'].thresh)
            assert float(arrays['theta_output'])==float(neuron.thresh)
            for quantized in (False,True):
                axis=name+('_u8_vq5' if quantized else '_fp32')
                started=time.monotonic()
                pair=RecoveryPair(arrays,conv.weight.device,quantized)
                conv.forward,neuron.forward=pair.conv_forward,pair.neuron_forward
                optimizer=torch.optim.Adam(pair.parameters(),lr=LR)
                torch.cuda.reset_peak_memory_stats()
                row=dict(complete=False,initial_file=str(HERE/file),initial_local_updates=previous_updates,
                    recovery_updates=0,total_updates_after_recovery=previous_updates+STEPS,
                    initial_scope='legacy local-fit budget; recovery-stage budget64 is matched, cumulative budgets differ',history=[])
                run['axes'][axis]=row;save_json(args.output/'result.json',run)
                for step,filename in enumerate(schedule):
                    x,label,valid=input_frame(args.data,filename)
                    optimizer.zero_grad(set_to_none=True)
                    prediction=forward_flow(x)
                    error=prediction.permute(0,2,3,1)[valid]-label.permute(0,2,3,1)[valid]
                    loss=torch.sqrt(error.square().sum(-1)+1e-6).mean()
                    if not loss.requires_grad or not torch.isfinite(loss):
                        raise RuntimeError('Real training loss has no finite graph: '+axis+' '+filename)
                    with fp32_matmul():
                        loss.backward()
                    if pair.u.grad is None or pair.v.grad is None or not torch.isfinite(pair.u.grad).all() or not torch.isfinite(pair.v.grad).all():
                        raise RuntimeError('Missing/nonfinite U/V training gradient: '+axis+' '+filename)
                    entry=dict(step=step+1,file=filename,loss_before_update=float(loss.detach()),
                        frame_AEE_before_update=float(torch.linalg.vector_norm(error.detach(),dim=-1).double().mean()),
                        valid_pixels=int(valid.sum()))
                    if step==0 or step==STEPS-1:
                        entry.update(U_gradient=gradient_stats(pair.u),V_gradient=gradient_stats(pair.v,pair.connectivity))
                    optimizer.step();pair.project()
                    row['history'].append(entry);row['recovery_updates']=step+1
                    if (step+1)%8==0 or step==0:
                        row['peak_allocated_bytes']=int(torch.cuda.max_memory_allocated())
                        save_json(args.output/'result.json',run)
                        print('FLOW_RECOVERY',axis,step+1,json.dumps(entry),flush=True)
                    del prediction,error,loss,x,label,valid
                    # Optional activity-regularization diagnostics are not in
                    # this loss; keep values but release their old graph refs.
                    for module in modules.values():
                        value=getattr(module,'act_value',None)
                        if torch.is_tensor(value): module.act_value=value.detach()
                row['training_peak_allocated_bytes']=int(torch.cuda.max_memory_allocated())
                pair.start_statistics()
                with torch.no_grad():
                    for index,filename in enumerate(names):
                        functional.reset_net(model)
                        x,_,_=input_frame(args.data,filename,targets=False)
                        try:
                            model(x)
                        except CalibrationReady:
                            pass
                        else:
                            raise RuntimeError('Expected the private-residual statistics stop')
                        del x
                        if (index+1)%4==0 or index+1==len(names):
                            print('TRAIN_STATISTICS',axis,index+1,len(names),flush=True)
                moments=pair.finish_statistics()
                exported=pair.export_recovery(arrays,moments,names)
                exported['initial_local_updates']=np.array(previous_updates)
                exported['total_updates_after_recovery']=np.array(previous_updates+STEPS)
                destination=args.output/(axis+'.npz')
                np.savez_compressed(destination,**exported)
                row.update(complete=True,parameters=str(destination),statistical_observations_per_time_channel=moments['count'],
                    forbidden_V_nonzeros=int(np.count_nonzero(exported['v'][~arrays['connectivity']])),
                    training_trajectory_mean_loss=float(np.mean([x['loss_before_update'] for x in row['history']])),
                    wall_seconds=time.monotonic()-started)
                save_json(args.output/'result.json',run)
                print('AXIS_DONE',axis,json.dumps({key:value for key,value in row.items() if key!='history'}),flush=True)
                conv.forward,neuron.forward=original_conv,original_neuron
                del pair,optimizer
                functional.reset_net(model);gc.collect();torch.cuda.empty_cache()
        run['complete']=True;save_json(args.output/'result.json',run)
    except Exception as exc:
        run['exception']=dict(type=type(exc).__name__,message=str(exc))
        run['peak_allocated_bytes_on_exception']=int(torch.cuda.max_memory_allocated())
        save_json(args.output/'result.json',run)
        raise
    finally:
        handle.remove();conv.forward,neuron.forward=original_conv,original_neuron
    print('DONE',str(args.output/'result.json'),flush=True)


if __name__=='__main__':
    main()
