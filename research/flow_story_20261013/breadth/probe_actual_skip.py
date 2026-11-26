#!/usr/bin/env python3
"""Same-u local numerical diagnosis of full-shape versus actual tile MLP calls.

No algorithm change, recovery, or hardware claim. Uses frozen evaluator's full
native provenance preflight. FP64 is a local diagnostic, never a new accuracy
baseline. Default preflight does not initialize a GPU.
"""
import argparse
import contextlib
import copy
import fcntl
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode=True

HERE=Path(__file__).resolve().parent
EVALUATOR_SHA='1cacbaeea4d9eaf19a304c3faaa13f5278f0212b6cdc92210c4619447d571628'
STAGES=('sn1_h','sn1','fc1','bn1','sn2_h','sn2','fc2','bn2')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def evaluator():
    path=HERE/'evaluate_breadth.py';require(sha(path)==EVALUATOR_SHA,'Frozen evaluator source differs')
    spec=importlib.util.spec_from_file_location('skip_diag_evaluator',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def tile_views(h,w,count=3):
    candidates=[(0,0),(math.ceil(h/9)//2,math.ceil(w/9)//2),((h-1)//9,(w-1)//9)]
    out=[];seen=set()
    for gy,gx in candidates:
        if (gy,gx) in seen:continue
        seen.add((gy,gx));out.append((str(gy)+','+str(gx),9*gy,min(9*(gy+1),h),9*gx,min(9*(gx+1),w)))
        if len(out)==count:break
    return out


def trace_mlp(mlp,u,views):
    """Run the UNMODIFIED module once; only selected output slices reach CPU.

    Surrogate pre-hooks observe addmm's real membrane before the binary step.
    BN wrapper outputs are T,B,C,H,W and are explicitly permuted for comparison.
    No hook edits input/output. All temporary hooks are removed even on errors.
    """
    import torch
    from spikingjelly.activation_based import functional
    require(not torch.is_grad_enabled() and not mlp.training,'Inference/eval required')
    require(tuple(u.shape[:2])==(10,1),'Complete T10/B1 required')
    t,b,h,w,c=u.shape;f=mlp.fc1.out_features
    records={stage:{} for stage in STAGES};handles=[]
    def save(stage,value):
        require(value.ndim==5 and tuple(value.shape[:4])==(t,b,h,w),'Stage layout differs:'+stage)
        require(bool(torch.isfinite(value).all()),'Nonfinite stage:'+stage)
        for key,y0,y1,x0,x1 in views:
            records[stage][key]=value[:,:,y0:y1,x0:x1,:].detach().to(device='cpu',copy=True).contiguous()
    for stage in ('sn1','fc1','sn2','fc2'):
        handles.append(getattr(mlp,stage).register_forward_hook(lambda m,inp,out,s=stage:save(s,out)))
    for stage in ('bn1','bn2'):
        handles.append(getattr(mlp,stage).register_forward_hook(lambda m,inp,out,s=stage:save(s,out.permute(0,1,3,4,2))))
    for name,width in (('sn1',c),('sn2',f)):
        psn=getattr(mlp,name).spiking_neuron
        require(type(psn).__name__=='PSN' and psn.T==10,'Expected native PSN')
        handles.append(psn.surrogate_function.register_forward_pre_hook(
            lambda m,inp,s=name+'_h',channels=width:save(s,inp[0].reshape(t,b,h,w,channels))))
    try:
        functional.reset_net(mlp);result=mlp(u)
        require(tuple(result.shape)==tuple(u.shape),'MLP external shape changed')
    finally:
        for handle in handles:handle.remove()
    require(all(set(records[s])=={v[0] for v in views} for s in STAGES),'Missing stage observation')
    return records


def near_zero(x):
    import torch
    return {str(eps):int((x.abs()<=eps).sum()) for eps in (0.,1e-8,1e-7,1e-6,1e-5,1e-4)}


def compare_stage(a,b,stage,origin=(0,0)):
    import torch
    require(a.shape==b.shape,'Compared stage shapes differ')
    a,b=a.double(),b.double();delta=(a-b).abs();different=delta!=0
    dense_norm=float(torch.linalg.vector_norm(a))
    error_norm=float(torch.linalg.vector_norm(a-b))
    result={'elements':a.numel(),'numerically_equal':torch.equal(a,b),'different_elements':int(different.sum()),
        'max_abs_error':float(delta.max()),'mean_abs_error':float(delta.mean()),
        'dense_L2_norm':dense_norm,'error_L2_norm':error_norm,
        'relative_L2_error_to_dense':error_norm/dense_norm if dense_norm else None,
        'relative_norm_undefined_for_zero_reference':dense_norm==0,
        'allclose_atol1e_5_rtol1e_5':bool(torch.allclose(a,b,atol=1e-5,rtol=1e-5)),
        'dense_min':float(a.min()),'dense_max':float(a.max()),'tile_min':float(b.min()),'tile_max':float(b.max())}
    if stage in ('sn1','sn2'):
        require(bool(((a==0)|(a==1)).all()) and bool(((b==0)|(b==1)).all()),'Nonbinary PSN output')
        result['spike_disagreements']=int(different.sum())
        result['spike_total_elements']=a.numel()
        result['spike_disagreement_fraction']=int(different.sum())/a.numel()
    if stage.endswith('_h'):
        flip=(a>=0)!=(b>=0)
        result.update(dense_near_zero=near_zero(a),tile_near_zero=near_zero(b),threshold_sign_disagreements=int(flip.sum()),
                      threshold_sign_disagreement_fraction=int(flip.sum())/a.numel(),
                      threshold_rule='native ATan surrogate forward uses heaviside(h>=0)',first_sign_disagreements=[])
        for index in torch.nonzero(flip)[:8].tolist():
            idx=tuple(index);global_index=index.copy();global_index[2]+=origin[0];global_index[3]+=origin[1]
            result['first_sign_disagreements'].append({'global_TBHWC_index':global_index,'dense_h':float(a[idx]),'tile_h':float(b[idx])})
    return result


def local_probe(mlp,u,views):
    """Same full u; every local slice has the same values, T and channel axis."""
    import torch
    records=trace_mlp(mlp,u,views);results=[]
    for key,y0,y1,x0,x1 in views:
        local=u[:,:,y0:y1,x0:x1,:].contiguous()
        trace=trace_mlp(mlp,local,[(key,0,y1-y0,0,x1-x0)])
        stages={s:compare_stage(records[s][key],trace[s][key],s,(y0,x0)) for s in STAGES}
        results.append({'tile_key':key,'valid_bounds_yx':[y0,y1,x0,x1],
            'full_input_shape':list(u.shape),'tile_input_shape':list(local.shape),'stages':stages,
            'first_nonzero_stage':next((s for s in STAGES if stages[s]['different_elements']),None),
            'first_spike_disagreement':next((s for s in ('sn1','sn2') if stages[s]['spike_disagreements']),None)})
    return results,records


def run_probe(a,context,ev):
    native,trainer,metrics,rows,hashes,config,run,side,regions,original_plan=context
    import torch
    torch.set_num_threads(a.cpu_threads);torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    device=torch.device(a.device);require(device.type in ('cpu','cuda'),'Unsupported device')
    require(device.type!='cuda' or torch.cuda.is_available(),'CUDA unavailable')
    if device.type=='cuda':torch.cuda.set_device(device)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    require(saved.get('schema_version')==ev.SCHEMA and saved.get('contract')==run['contract'] and
        saved.get('contract_sha256')==side['contract_sha256'] and saved.get('provenance')==run['provenance'] and
        saved.get('effective_config')==run['effective_config'] and
        saved['training']['completed_optimizer_steps']==side['completed_optimizer_steps'],'Checkpoint payload differs')
    state=saved['model_state_dict'];require(all(isinstance(v,torch.Tensor) and
        (not v.is_floating_point() or bool(torch.isfinite(v).all())) for v in state.values()),'Invalid checkpoint tensor')
    sys.path.insert(0,str(a.source_root));os.environ['SDFORMER_SNN_BACKEND']='torch'
    from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4
    from models.STSwinNet_SNN.Spiking_submodules import PSN
    from utils.runtime_backend import configure_snn_backend
    from spikingjelly.activation_based import functional
    for obj in (MS_SpikingformerFlowNet_en4,PSN,configure_snn_backend):
        require(Path(inspect.getfile(obj)).resolve().is_relative_to(a.source_root),'Unpinned native import')
    config=copy.deepcopy(config);config['swin_transformer']['input_size']=[288,384] if a.geometry=='crop_center9' else [480,640]
    model=MS_SpikingformerFlowNet_en4(copy.deepcopy(config['model']),copy.deepcopy(config['swin_transformer']))
    model.load_state_dict(state,strict=True);del state,saved
    model.to(device);functional.set_step_mode(model,'m');configure_snn_backend(model,device,config,PSN)
    ev.apply_bn(model,'D',torch);before=ev.bn_signature(model,torch)
    mlps=[(n,m) for n,m in model.named_modules() if type(m).__name__=='MS_Spiking_Mlp']
    require(mlps,'No native MLPs');ys,xs=metrics.crop_slices(a.geometry);began=time.monotonic()
    result={'schema':'flow_actual_skip_local_numeric_probe_v1','status':'RUNNING','probe_sha256':sha(__file__),
        'evaluator_sha256':EVALUATOR_SHA,'checkpoint_sha256':a.checkpoint_sha256,
        'preflight':original_plan,'source_graph':'original dense D, all MLPs unmasked; frozen checkpoint BN',
        'targets':[r['target_id'] for r in rows[:a.probe_targets]],'precision':'FP32 TF32off plus local-only FP64 cloned diagnostic',
        'test_tile_rule':'first/center/last valid9x9 tile, duplicates removed; same true u',
        'scope':'numerical diagnosis only; no changed algorithm/BN/source, no AEE/hardware claim',
        'all_keep_caveat':'existing TileMLP all-keep calls original full-shape native; zero error does not test partial-tile arithmetic',
        'layers':[],'input_sha256':{},'GPU_inference_executed':device.type=='cuda'}
    ev.write_result(a.output,result)
    for row in rows[:a.probe_targets]:
        require(time.monotonic()-began<a.wall_minutes*60,'Probe wall budget exhausted before next target')
        evidence={};expected={}
        for name,rel in native.target_files(row).items():
            native.checked_path(a.data_root,rel);expected['events' if name=='input' else name]=hashes[rel]['sha256']
        arrays,_=metrics.load_arrays(a.data_root/'saved_flow_data',row,evidence=evidence,expected_hashes=expected)
        result['input_sha256'][row['target_id']]=evidence
        captured={};handles=[]
        for name,module in mlps:
            def capture(mod,inputs,n=name):
                require(n not in captured,'MLP input observed more than once')
                captured[n]=inputs[0].detach().to(device='cpu',copy=True).contiguous()
            handles.append(module.register_forward_pre_hook(capture))
        try:
            with torch.inference_mode():
                functional.reset_net(model);full=model(ev.prepare_input(arrays['events'][:,ys,xs],device,torch))
                require(all(bool(torch.isfinite(x).all()) for x in full['flow']),'Nonfinite dense producer')
                if device.type=='cuda':torch.cuda.synchronize(device)
                del full
        finally:
            for h in handles:h.remove()
        require(set(captured)=={n for n,m in mlps},'Incomplete dense u capture')
        for index,(name,module) in enumerate(mlps):
            require(time.monotonic()-began<a.wall_minutes*60,'Probe wall budget exhausted before next layer')
            cpu_u=captured.pop(name);views=tile_views(cpu_u.shape[2],cpu_u.shape[3],a.probe_tiles)
            entry={'target_id':row['target_id'],'layer_name':name,'u_shape':list(cpu_u.shape),
                   'u_dtype':str(cpu_u.dtype),'u_tensor_sha256':hashlib.sha256(cpu_u.numpy().tobytes()).hexdigest()}
            with torch.inference_mode():
                u=cpu_u.to(device);entry['FP32'],trace32=local_probe(module,u,views);del u
                if index<a.fp64_layer_count:
                    clone=copy.deepcopy(module).to(device=device,dtype=torch.float64).eval()
                    entry['FP64'],trace64=local_probe(clone,cpu_u.to(device=device,dtype=torch.float64),views)
                    entry['FP32_dense_vs_FP64_dense']={key:{s:compare_stage(trace32[s][key],trace64[s][key],s,(y0,x0))
                        for s in STAGES} for key,y0,y1,x0,x1 in views}
                    del clone,trace64
                else:entry['FP64']='NOT_REQUESTED_FOR_THIS_LAYER'
                del trace32
            result['layers'].append(entry);del cpu_u;gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
            ev.write_result(a.output,result)
            print(json.dumps({'target':row['target_id'],'layer':name,
                'FP32_first_differences':[r['first_nonzero_stage'] for r in entry['FP32']],
                'FP32_first_spike_flips':[r['first_spike_disagreement'] for r in entry['FP32']]}),flush=True)
    result['BN_buffers_unchanged']=before==ev.bn_signature(model,torch)
    require(result['BN_buffers_unchanged'],'Original BN mutated')
    result.update(status='PROBE_COMPLETE',wall_seconds=time.monotonic()-began,
                  PSN_source_sha256=sha(Path(inspect.getfile(PSN))))
    ev.write_result(a.output,result);return result


def main(argv=None):
    custom=argparse.ArgumentParser(add_help=False)
    custom.add_argument('--probe-targets',type=int,choices=(1,2),default=1)
    custom.add_argument('--probe-tiles',type=int,choices=(1,2,3),default=3)
    custom.add_argument('--fp64-layer-count',type=int,default=1)
    custom.add_argument('--gpu-lock',type=Path)
    extra,remaining=custom.parse_known_args(argv)
    require(0<=extra.fp64_layer_count<=12,'FP64 layer count must be 0..12')
    ev=evaluator();a=ev.parse_args(remaining)
    for k,v in vars(extra).items():setattr(a,k,v)
    require(not (a.run_inference and a.device.startswith('cuda')) or a.gpu_lock is not None,'Explicit shared GPU lock required')
    context=ev.preflight(a)
    if not a.run_inference:
        ev.write_result(a.output,{'status':'PROBE_PREFLIGHT_PASS_NO_INFERENCE','native_preflight':context[-1],
            'probe_targets':a.probe_targets,'probe_tiles':a.probe_tiles,'fp64_layer_count':a.fp64_layer_count,
            'probe_sha256':sha(__file__)});return 0
    lock=a.gpu_lock.open('a') if a.gpu_lock else contextlib.nullcontext()
    try:
        with lock as handle:
            if a.gpu_lock:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run_probe(a,context,ev)
    except BaseException as exc:
        path=a.output.with_suffix(a.output.suffix+'.failure.json')
        ev.write_result(path,{'status':'LOCAL_PROBE_INCOMPLETE_NOT_ALGORITHM_FAILURE','error':str(exc),
            'type':type(exc).__name__,'traceback':traceback.format_exc()});raise
    return 0


if __name__=='__main__':raise SystemExit(main())
