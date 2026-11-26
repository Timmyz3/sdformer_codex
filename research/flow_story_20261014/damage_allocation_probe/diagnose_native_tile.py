#!/usr/bin/env python3
"""Two fixed dev inputs: isolate native MLP vs tile arithmetic on identical u.

Loads pinned ROUND4 experiment infrastructure, does not edit its sources/model.
No task-method conclusion. First two fixed dev8 frames; stage0/block0 and
stage2/block0, each observed under native prefix. No gradients/recovery.
"""
import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent/'structured_merge/scope_extension_v1/run_scope_extension.py'
PARENT_SHA = '109f535f29fa6cbad5e37258a44ec3b7a1f0626303a85f4afdcd47ace4665402'


def main():
    if hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError('Pinned ROUND4 loader changed')
    spec = importlib.util.spec_from_file_location('frozen_scope_diag', PARENT)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('parent-contract', 'identity-bank', 'prior-screen-results'):
        parser.add_argument('--'+key, type=Path, required=True)
        parser.add_argument('--'+key+'-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--module-checks', type=Path, default=PARENT.parent.parent/'cpu_results.json')
    parser.add_argument('--runner-checks', type=Path, default=PARENT.parent/'runner_cpu_results.json')
    parser.add_argument('--wall-minutes', type=float, default=15)
    parser.add_argument('--preflight-output', type=Path)
    parser.add_argument('--execute', action='store_true')
    a = parser.parse_args()
    prepared = parent.prepare(a)
    result = {'schema':'flow_native_tile_numerics_v1','status':'PREFLIGHT_PASS_NO_INFERENCE',
              'parent_binding':prepared[-1], 'script_sha256':parent.sha(__file__),
              'targets':prepared[-1]['dev_ids'][:2], 'layers':['stage0/block0','stage2/block0'],
              'inference_executed':False,'training_performed':False}
    if not a.execute:
        parent.require(a.preflight_output and not a.preflight_output.exists(), 'New preflight output required')
        parent.write(a.preflight_output,result); print(result['status']); return
    with parent.GPU_LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        a.output_dir.mkdir(parents=True,exist_ok=False)
        previous = signal.getsignal(signal.SIGALRM)
        def alarm(*_): raise TimeoutError('Fixed diagnostic wall cap')
        signal.signal(signal.SIGALRM,alarm); signal.setitimer(signal.ITIMER_REAL,a.wall_minutes*60)
        try:
            execute(a,prepared,result,parent)
        except BaseException as e:
            result.update(status='INCOMPLETE_NO_CONCLUSION',error=str(e),traceback=traceback.format_exc()); raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous)
            parent.write(a.output_dir/'results.json',result)


def execute(a,prepared,result,parent):
    import torch
    parent.require(not torch.cuda.is_initialized(),'No pre-lock CUDA')
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(20261014)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    device=torch.device('cuda:0'); torch.cuda.set_device(device)
    _,b,context,base,scope,loader,dev,binding=prepared
    native,trainer,metrics,_,hashes,config,saved,side,regions,plan=context
    model,functional=loader.load_model(b,context,base,torch,device)
    bn_info=scope.apply_scope(model,'MLP_CMA176',b.bn_payload,torch)
    model.eval().requires_grad_(False)
    targets,shapes=parent.target_layers(model)
    selected=[n for n in targets if '.swin_blocks.0.' in n and any(f'.layers.{s}.' in n for s in (0,2))]
    before=parent.source_digest(model); bn_before=base.bn_signature(model,torch)
    result.update(status='RUNNING',inference_executed=True,frames=[],BN_scope=bn_info,
                  original_state_sha256=before,runtime={'torch':str(torch.__version__),'TF32':False,'AMP':False})
    start=time.monotonic()

    def capture_modules(mlp,x):
        captured={}; handles=[]
        try:
            for key in ('sn1','fc1','bn1','sn2','fc2','bn2'):
                def hook(m,inp,out,key=key):
                    val=out.permute(0,1,3,4,2) if key.startswith('bn') else out
                    captured[key]=val.detach().clone()
                handles.append(getattr(mlp,key).register_forward_hook(hook))
            y=mlp(x)
        finally:
            for h in handles: h.remove()
        parent.require(tuple(captured)==('sn1','fc1','bn1','sn2','fc2','bn2'),'Native atomic order differs')
        return y,captured

    with torch.inference_mode():
        for row in dev[:2]:
            expected={}; evidence={}
            for key,rel in native.target_files(row).items():
                native.checked_path(b.data_root,rel); expected['events' if key=='input' else key]=hashes[rel]['sha256']
            arrays,_=metrics.load_arrays(b.data_root/'saved_flow_data',row,evidence=evidence,expected_hashes=expected)
            chunk=base.prepare_input(arrays['events'],device,torch)
            captured_inputs={}; captured_outputs={}; handles=[]
            try:
                for name in selected:
                    def hook(m,inp,out,name=name):
                        captured_inputs[name]=inp[0].detach().clone()
                        captured_outputs[name]=out.detach().clone()
                    handles.append(targets[name].register_forward_hook(hook))
                functional.reset_net(model); flows=model(chunk)['flow']
            finally:
                for h in handles: h.remove()
            frame={'target_id':row['target_id'],'input_evidence':evidence,'layers':{}}
            for name in selected:
                mlp=targets[name]; u=captured_inputs[name]
                full,full_ops=capture_modules(mlp,u)
                parent.require(torch.equal(full,captured_outputs[name]),'Isolated full MLP is not repeated native result')
                h,w=u.shape[2:4]
                totals={k:{'elements':0,'different_elements':0,'sum_abs_error':0.,'max_abs_error':0.,'tiles_with_difference':0}
                        for k in full_ops}
                first_diff=None
                for y0 in range(0,h,9):
                    for x0 in range(0,w,9):
                        local=u[:,:,y0:min(y0+9,h),x0:min(x0+9,w),:].contiguous()
                        tile,tile_ops=capture_modules(mlp,local)
                        for key,val in tile_ops.items():
                            ref=full_ops[key][:,:,y0:min(y0+9,h),x0:min(x0+9,w),:]
                            parent.require(ref.shape==val.shape,'Atomic shape mismatch')
                            d=(ref.double()-val.double()).abs(); count=int((d!=0).sum()); z=totals[key]
                            z['elements']+=d.numel(); z['different_elements']+=count
                            z['sum_abs_error']+=float(d.sum()); z['max_abs_error']=max(z['max_abs_error'],float(d.max()))
                            z['tiles_with_difference']+=int(count>0)
                            if count and first_diff is None:
                                first_diff={'tile_yx':[y0//9,x0//9],'operator':key,'max_abs_error':float(d.max())}
                        del tile,tile_ops
                for z in totals.values(): z['mean_abs_error']=z['sum_abs_error']/z['elements']
                frame['layers'][name]={'shape':list(u.shape),'stride':list(u.stride()),'input_sha256':parent.load_frozen_merge().tensor_sha(u),
                    'isolated_full_exact_native_branch':True,'atomic_comparisons':totals,'first_difference_in_tile_traversal':first_diff,
                    'interpretation':'Same fixed u, complete T; shape-dependent arithmetic may propagate through thresholds. No full-flow causal attribution.'}
                del full,full_ops
            result['frames'].append(frame); parent.write(a.output_dir/'results.json',result)
            print(json.dumps({'target_id':row['target_id'],'completed':len(result['frames'])}),flush=True)
            del chunk,flows,captured_inputs,captured_outputs
    parent.require(parent.source_digest(model)==before and base.bn_signature(model,torch)==bn_before,'Native state/BN changed')
    result.update(status='NATIVE_TILE_ATOMIC_DIAGNOSTIC_COMPLETE',full_network_forwards=2,training_performed=False,
                  original_model_unchanged=True,BN_buffers_unchanged=True,wall_seconds=time.monotonic()-start,
                  scope='2 fixed dev8 frames, identical native-prefix u at two MLPs; isolated arithmetic localization, not task quality or hardware benefit')


if __name__=='__main__': main()
