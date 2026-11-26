#!/usr/bin/env python3
"""Fixed271482 native CMA176 + check8 native/FF + first FF dual =193; no training."""
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time
import traceback
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
H0DIR=HERE.parent/'coverage_recovery_round13/next_action_headroom/implementation_h0'
CHECKPOINT_SHA='1e72cca3946f40e24a1575f3348f9a596d744460808ad910a616ba713b5e4977'
CHECKPOINT_BYTES=660107890
CONTRACT_SHA='961f1700e76f3427c74d4e9af9799a684433ff00449b8386be6c1f9cdd75e934'
RUN_CONTRACT_SHA='864c949018707b3329d3c7b3f23f4f36fa8d17b80a30150710689b991beeb3c6'
SIDECAR_SHA='3967ce0f858b76699c1fb815d8ca2c4b443296eafbe9e1293c285a69d0358b41'
MEMBER_SHA='eba4544fe99c082d7d099f90d03470d7abad22b01b68d3eb4d022e61e78151d1'
REGION_SHA='2e4934f6d82663f16ca1ae6c3aea238a31e699d4f49916052f6ea6b5e9c1d41b'
SOURCE_STATE_SHA='ad14a7d45091aa75d64df83e1a71bd89b35dfbbf49c9ab84861d5ae0aa270560'
SOURCE_NON_BN_SHA='b721031729a9a5f69d03037be53e62ebf8c1536dad0ad8a8f8e1601fe9fc2798'
BN_SCOPE='MLP_271482_OWN_CMA176'
SCHEMA='flow_later_checkpoint_deployment_v1'
OWN=('run_deployment.py','deployment_core.py','test_deployment_cpu.py','source_pins.json','PROTOCOL.md')
import deployment_core as core
require=core.require


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load_runtime():
    require(sha(H0DIR/'h0_core.py')=='48fccdd60cbc468c381d3caf6a316a6427dd68f09933731290befd729b9c58da','Frozen H0 math differs')
    h0=load(H0DIR/'h0_core.py','_later_checkpoint_math_only')
    for relative,digest in json.loads((HERE/'source_pins.json').read_text())['sources'].items():
        require(sha(h0.dependency_path(relative))==digest,'Frozen source differs: '+relative)
    return h0,h0.load_frozen()


def check_payload(payload,run,side,torch,expected_count=711,expected_elements=55001588):
    require(payload.get('schema_version')=='flow_a_native_encoding_v2_checkpoint_v1'
            and payload.get('contract_sha256')==CONTRACT_SHA==side['contract_sha256'],'Later native checkpoint contract differs')
    require(all(payload.get(k)==run[k] for k in ('contract','provenance','effective_config')),'Later CPU payload/source/encoding contract differs')
    require(payload['training']['completed_optimizer_steps']==271482==side['completed_optimizer_steps'],'Only exact271482 checkpoint')
    state=payload['model_state_dict']
    require(all(isinstance(v,torch.Tensor) and v.device.type=='cpu' and bool(torch.isfinite(v).all()) for v in state.values()),'Finite CPU later model tensors required')
    require(len(state)==expected_count and sum(v.numel() for v in state.values())==expected_elements,'Later model tensor inventory differs')
    require(payload.get('stop_reason')=='wall_budget','Later checkpoint stopping provenance differs')
    return {'checkpoint_step':271482,'model_tensors':len(state),'model_elements':sum(v.numel() for v in state.values()),
            'all_model_tensors_CPU_finite':True,'contract_sha256':CONTRACT_SHA,'optimizer_or_scheduler_restored':False,
            'stop_reason':payload['stop_reason']}


def prepare(a,h0,runtime):
    r13,collector,probe,parent,r9,r8=runtime
    require(0<a.wall_minutes<=30 and not a.output_dir.exists(),'Fresh output and at most30min required')
    require(sha(a.checkpoint)==a.checkpoint_sha256==CHECKPOINT_SHA and a.checkpoint.stat().st_size==CHECKPOINT_BYTES,'Exact actual271482 checkpoint bytes required')
    require(sha(a.checkpoint.with_suffix('.json'))==SIDECAR_SHA and sha(a.checkpoint.parent/'run_contract.json')==RUN_CONTRACT_SHA,'Later sidecar/run contract bytes differ')
    require(sha(a.runtime_manifest)==a.runtime_manifest_sha256 and sha(a.membership)==a.membership_sha256==MEMBER_SHA,'Explicit runtime/member SHA differs')
    rt=json.loads(a.runtime_manifest.read_text());manifest=json.loads(a.membership.read_text())
    require(rt['schema']=='flow_round13_runtime_v1' and rt['region_sha256']==REGION_SHA,'Inherited runtime/region differs')
    for path,digest in rt['source_pins'].items():require(sha(path)==digest,'Runtime source differs: '+path)
    args=list(rt['base_args'])
    require(not any(x.startswith('--bn-') or x.startswith('--svs-') or x.startswith('--regions') or x=='--run-inference' for x in args),'Imported BN/calibration artifacts forbidden')
    for flag,value in (('--checkpoint',str(a.checkpoint)),('--checkpoint-sha256',CHECKPOINT_SHA),('--output',str(a.output_dir/'results.json'))):
        require(args.count(flag)==1,'Exactly one native argument '+flag);args[args.index(flag)+1]=value
    sys.path[:0]=[rt['breadth_root'],rt['motion_root']]
    base=load(Path(rt['breadth_root'])/'evaluate_breadth.py','_later_native_source_preflight')
    b=base.parse_args(args);b.device='cuda:0';b.wall_minutes=a.wall_minutes
    require(b.geometry=='full9' and not b.run_inference,'Full9/no implicit inference required')
    context=list(base.preflight(b)) # verifies native source63, encoding and actual accepted data.
    native,trainer,metrics,_,hashes,config,run,side,_,plan=context
    canonical=metrics.canonical_rows(b.canonical);roles=metrics.auxiliary_roles(b.roles,canonical)
    membership_audit=r13.validate_members(manifest,canonical,roles)
    require(manifest['source_sha256']['canonical']==sha(b.canonical) and manifest['source_sha256']['roles']==sha(b.roles),'Member source bytes differ')
    region_path=Path(rt['region_path']);require(sha(region_path)==REGION_SHA,'Training-side region source differs')
    regions=json.loads(region_path.read_text());metrics.validate_region_contract(regions,canonical,roles,require_complete_fit=False)
    require(regions['fit_target_ids']==[r['target_id'] for r in manifest['train_members_diversity']],'Region training identity order differs')
    by_id={r['target_id']:r for r in canonical}
    require([r['target_id'] for r in regions['data_evidence']]==regions['fit_target_ids'],'Region GT evidence missing')
    for e in regions['data_evidence']:
        files=native.target_files(by_id[e['target_id']]);require(all(e[k+'_sha256']==hashes[files[k]]['sha256'] for k in ('gt','mask')),'Region GT/mask source differs')
    context[8]=regions
    evidence=[]
    for phase,key in (('calibration','bn_members'),('evaluation','check_members')):
        for row in manifest[key]:
            files=native.target_files(row);keys=('input',) if phase=='calibration' else ('input','gt','mask');entry={}
            for kind in keys:
                relative=files[kind];require(sha(native.checked_path(b.data_root,relative))==hashes[relative]['sha256'],'Native source bytes differ')
                entry[kind]=hashes[relative]['sha256']
            evidence.append({'phase':phase,'target_id':row['target_id'],'sequence':row['sequence'],'files':entry})
    cpu_sha=parent.checked_cpu_receipt(a.cpu_checks,[HERE/n for n in OWN])
    import torch
    require(not torch.cuda.is_initialized(),'Preflight must remain CPU')
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    checkpoint_audit=check_payload(payload,run,side,torch);checkpoint_audit['full_source_state_sha256']=r13.state_digest(types_state(payload['model_state_dict']))
    require(checkpoint_audit['full_source_state_sha256']==SOURCE_STATE_SHA,'Later source state differs from independent actual CPU audit')
    del payload
    binding={'schema':'flow_later_checkpoint_binding_v1','source_sha256':{n:sha(HERE/n) for n in OWN},'cpu_checks_sha256':cpu_sha,
             'checkpoint_sha256':CHECKPOINT_SHA,'checkpoint_path':str(a.checkpoint),'checkpoint_step':271482,'contract_sha256':CONTRACT_SHA,
             'run_contract_sha256':RUN_CONTRACT_SHA,'sidecar_sha256':SIDECAR_SHA,'runtime_manifest_sha256':a.runtime_manifest_sha256,
             'membership_sha256':a.membership_sha256,'membership_origin_checkpoint_sha256':manifest['checkpoint_sha256'],
             'region_sha256':REGION_SHA,'bn_ids':[r['target_id'] for r in manifest['bn_members']],
             'check_ids':[r['target_id'] for r in manifest['check_members']],'BN_scope':BN_SCOPE,'forward_plan':core.plan(manifest),
             'calibration_forwards':176,'evaluation_actual_forwards':16,'dual_forwards':1,'total_full_forwards':193,'optimizer_steps':0,
             'calibration_GT_used':False,'external_BN_current_stat':54,'private_BN_frozen':24,'old_BN_loaded':False,
             'wall_minutes':a.wall_minutes,'GPU_lock':str(parent.GPU_LOCK),'precision':'FP32/noAMP/noTF32/T10/allC/full9',
             'scope':'Later checkpoint plus its own BN deployment package. Check8 development only; no training, recovery, H0, flow increment or hardware claim.'}
    return {'binding':binding,'manifest':manifest,'runtime_manifest':rt,'b':b,'context':tuple(context),'base':base,
            'checkpoint_cpu_audit':checkpoint_audit,'membership_audit':membership_audit,'data_evidence':evidence}


def types_state(state):
    class Holder:
        def state_dict(self):return state
    return Holder()


def execute(a,p,h0,runtime,result):
    import numpy as np
    import torch
    r13,collector,probe,parent,r9,r8=runtime
    require(not torch.cuda.is_initialized(),'CUDA initialized before shared lock')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261016)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.manual_seed_all(20261016)
    started=time.monotonic()
    def deadline():require(time.monotonic()-started<a.wall_minutes*60,'Fixed193 diagnostic wall cap exceeded')
    rt=p['runtime_manifest'];base=p['base'];b=p['b'];context=p['context'];members=p['manifest']
    root=Path(rt['breadth_root']);loader=load(root/'run_same_shape_reference.py','_later_frozen_loader')
    scope=load(root/'run_bn_scope.py','_later_bn_inventory');cma=load(root/'collect_bn_cma.py','_later_cma_functions_only')
    native,trainer,metrics,_,hashes,config,run,side,regions,plan=context
    model,functional=loader.load_model(b,context,base,torch,device);model.requires_grad_(False)
    source_state=r13.state_digest(model)
    require(source_state==p['checkpoint_cpu_audit']['full_source_state_sha256'],'GPU source state differs from CPU checkpoint')
    targets,shapes=parent.target_layers(model);merge=parent.load_frozen_merge();all_bn,mlp_bn=scope.bn_inventory(model,torch)
    cma.reset_bn_cma(model,torch);before=cma.non_bn_digest(model,all_bn)
    require(before==SOURCE_NON_BN_SHA,'Later non-BN state differs from independent actual CPU audit')
    reset_signature=base.bn_signature(model,torch)
    result.update(status='RUNNING',inference_executed=True,source_checkpoint_state_sha256=source_state,full_forward_count=0,
                  forward_records=[],evaluations={},runtime={'torch':str(torch.__version__),'numpy':np.__version__,'cuda':torch.version.cuda,
                  'device':torch.cuda.get_device_name(device),'TF32':False,'AMP':False},BN_scope={'scope':BN_SCOPE,'BN_count':78,'MLP_BN_count':24})
    def record(item,execution):
        expected=p['binding']['forward_plan'][result['full_forward_count']]
        require(all(item[k]==v for k,v in expected.items() if k!='forward_id'),'Fixed forward order differs')
        result['full_forward_count']+=1;result['forward_records'].append(dict(expected,execution=execution))
        require(result['full_forward_count']<=193,'No additional model forward allowed')
    receipt={'schema':'flow_later_checkpoint_own_bn_v1','status':'RUNNING','checkpoint_sha256':CHECKPOINT_SHA,'checkpoint_step':271482,
             'membership_sha256':a.membership_sha256,'runtime_manifest_sha256':a.runtime_manifest_sha256,
             'source_sha256':p['binding']['source_sha256'],'bn_target_ids':p['binding']['bn_ids'],'BN_scope':BN_SCOPE,
             'calibration_definition':'all78_reset_CMA_train_nonBN_eval','calibration_forwards':0,'calibration_bn_count':78,
             'retained_mlp_bn_count':24,'old_CMA_loaded':False,'GT_used':False,'optimizer_steps':0,'input_evidence':[],
             'source_checkpoint_state_sha256':source_state,'non_bn_state_before_sha256':before,
             'all78_reset_signature':reset_signature,'all78_reset_counters':{n:int(v.num_batches_tracked) for n,v in all_bn.items()}}
    require(set(receipt['all78_reset_counters'].values())=={0},'All78 reset counters must be zero')
    calibration_started=time.monotonic()
    for row in members['bn_members']:
        deadline();relative=native.target_files(row)['input'];path=native.checked_path(b.data_root,relative)
        require(sha(path)==hashes[relative]['sha256'],'Calibration event changed after preflight')
        events=np.load(path,allow_pickle=False);require(events.shape==(10,480,640) and events.dtype==np.float32 and np.isfinite(events).all(),'Calibration event geometry/encoding differs')
        chunk=base.prepare_input(events,device,torch);ex=core.native_calibration_forward(model,functional,targets,shapes,chunk,torch)
        ex['all78_num_batches_tracked']={n:int(v.num_batches_tracked) for n,v in all_bn.items()}
        require(set(ex['all78_num_batches_tracked'].values())=={receipt['calibration_forwards']+1},'Every native calibration call must update all78 once')
        record({'phase':'calibration','target_id':row['target_id'],'layout':'native','mode':'actual'},ex);del events,chunk
        receipt['calibration_forwards']+=1;receipt['input_evidence'].append({'target_id':row['target_id'],'relative':relative,'sha256':hashes[relative]['sha256']})
        if receipt['calibration_forwards']%16==0:
            write(a.output_dir/'bn_receipt.json',receipt);write(a.output_dir/'results.json',result)
            print(json.dumps({'BN_completed':receipt['calibration_forwards'],'of':176}),flush=True)
    buffers=core.finish_calibration(model,all_bn,mlp_bn,cma,before,torch)
    bn_path=a.output_dir/'bn_buffers.npz';np.savez_compressed(bn_path,**buffers)
    receipt.update(status='OWN_BN_COMPLETE',buffers_sha256=sha(bn_path),non_bn_state_after_sha256=cma.non_bn_digest(model,all_bn),
                   non_bn_state_unchanged=True,retained_names=sorted(mlp_bn))
    write(a.output_dir/'bn_receipt.json',receipt)
    result['calibration_seconds']=time.monotonic()-calibration_started
    result.update(own_BN_receipt={'path':'bn_receipt.json','sha256':sha(a.output_dir/'bn_receipt.json')},
                  own_BN_buffers={'path':'bn_buffers.npz','sha256':sha(bn_path)},calibration_completed=176,
                  deployment_initial_state_sha256=r13.state_digest(model))
    bn_before=base.bn_signature(model,torch);state_before=r13.state_digest(model)
    result['BN_scope'].update(frozen_BN_names=sorted(mlp_bn),current_stat_BN_names=sorted(set(all_bn)-mlp_bn),
                              BN_signature_before_inference=bn_before,old_CMA_loaded=False)
    frames=[];meters={name:metrics.MetricAccumulator() for name in ('native','FF')};evaluation_started=time.monotonic()
    with torch.inference_mode():
        for index,row in enumerate(members['check_members']):
            deadline();expected={'events' if k=='input' else k:hashes[v]['sha256'] for k,v in native.target_files(row).items()};evidence={}
            arrays,_=metrics.load_arrays(b.data_root/'saved_flow_data',row,evidence=evidence,expected_hashes=expected)
            masks=metrics.partitions(arrays['gt'],arrays['mask'],arrays['events'],regions);chunk=base.prepare_input(arrays['events'],device,torch)
            frame={'target_id':row['target_id'],'sequence':row['sequence'],'input_evidence':evidence,'arms':{}};predictions={}
            for layout in ('native','FF'):
                deadline()
                if index==0 and layout=='FF':
                    actual,raw_inputs=core.forward_with_raw_inputs(h0,runtime,model,functional,targets,shapes,chunk,merge,torch,'actual',deadline)
                    raw_actual=core.save_raw_pair_side(a.output_dir/'first_FF_actual_raw.npz',actual,raw_inputs,np,sha);del raw_inputs
                else:actual=h0.forward(runtime,model,functional,targets,shapes,chunk,layout,merge,torch,deadline=deadline)
                flows,execution,_,_=actual;require(len(flows)==4,'Four native output heads required')
                record({'phase':'evaluation','target_id':row['target_id'],'layout':layout,'mode':'actual'},execution)
                prediction=flows[-1][0].numpy().copy();predictions[layout]=prediction
                frame['arms'][layout]={'metrics':meters[layout].update(row['target_id'],row['sequence'],prediction,arrays['gt'],masks),
                                      'prediction_sha256':hashlib.sha256(prediction.tobytes()).hexdigest(),'execution':execution}
                if index==0 and layout=='FF':
                    # Flush actual raw arrays and release their CPU branch copies
                    # before dual. Only its final prediction/metadata is retained.
                    del actual,flows
                    reference,raw_inputs=core.forward_with_raw_inputs(h0,runtime,model,functional,targets,shapes,chunk,merge,torch,'dual_reference',deadline)
                    raw_dual=core.save_raw_pair_side(a.output_dir/'first_FF_dual_raw.npz',reference,raw_inputs,np,sha);del raw_inputs
                    record({'phase':'evaluation','target_id':row['target_id'],'layout':'FF','mode':'dual_reference'},reference[1])
                    require(raw_actual['arrays']==raw_dual['arrays'] and raw_actual['branch_names_in_index_order']==raw_dual['branch_names_in_index_order'],
                            'First FF four heads/eight inputs/eight branches raw actual/dual mismatch')
                    result['first_frame_FF_dual']={'actual_execution':execution,'dual_execution':reference[1],
                        'actual_raw':raw_actual,'dual_raw':raw_dual,'all4_flow_heads_exact':True,'all8_inputs_exact':True,'all8_branches_exact':True}
                    del reference
                else:del actual,flows
            path=a.output_dir/('endpoint_'+row['target_id']+'.npz');np.savez_compressed(path,**predictions)
            frame.update(prediction_file=path.name,prediction_file_sha256=sha(path));frames.append(frame);del chunk,predictions,arrays
            core.check_closure(model,targets,state_before,r13.state_digest,bn_before,base.bn_signature,torch)
            print(json.dumps({'check_completed':index+1,'of':8}),flush=True)
    require(result['full_forward_count']==193 and receipt['calibration_forwards']==176 and len(frames)==8,'Incomplete193 diagnostic')
    require(cma.non_bn_digest(model,all_bn)==before,'Final non-BN state changed')
    core.check_closure(model,targets,state_before,r13.state_digest,bn_before,base.bn_signature,torch)
    result.update(status='LATER_CHECKPOINT_193_FORWARD_COMPLETE',evaluations={'endpoint':{'frames':frames,'summaries':{n:m.summary() for n,m in meters.items()}}},
                  deployment_final_state_sha256=r13.state_digest(model),non_bn_state_unchanged=True,BN_buffers_unchanged=True,
                  original_modules_restored=True,all_hooks_removed=True,wall_seconds=time.monotonic()-started,
                  evaluation_seconds=time.monotonic()-evaluation_started,
                  max_allocated_bytes=torch.cuda.max_memory_allocated(device),max_reserved_bytes=torch.cuda.max_memory_reserved(device))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime-manifest','membership','checkpoint'):
        parser.add_argument('--'+name,type=Path,required=True);parser.add_argument('--'+name+'-sha256',required=True)
    parser.add_argument('--cpu-checks',type=Path,default=HERE/'cpu_results.json')
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--preflight-output',type=Path)
    parser.add_argument('--wall-minutes',type=float,default=30);parser.add_argument('--execute',action='store_true')
    a=parser.parse_args();h0,runtime=load_runtime();p=prepare(a,h0,runtime)
    result={'schema':SCHEMA,'status':'PREFLIGHT_PASS_NO_INFERENCE','binding':p['binding'],'inference_executed':False,
            'training_performed':False,'optimizer_steps':0,'checkpoint_cpu_audit':p['checkpoint_cpu_audit'],
            'membership_audit':p['membership_audit'],'data_evidence':p['data_evidence'],'hardware_benefit_claim':False,'flow_specific_increment_claim':False}
    if not a.execute:
        require(a.preflight_output and not a.preflight_output.exists(),'Fresh explicit preflight output required')
        require(not a.preflight_output.resolve().is_relative_to(a.output_dir.resolve()),'Preflight receipt must be outside future execution directory')
        write(a.preflight_output,result);print(result['status']);return
    parent=runtime[3];require(parent.GPU_LOCK.parent.is_dir(),'Existing shared GPU0 lock required')
    with parent.GPU_LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);a.output_dir.mkdir(parents=True,exist_ok=False)
        previous=signal.getsignal(signal.SIGALRM)
        def timeout(signum,frame):raise TimeoutError('Later deployment fixed30min cap exhausted')
        signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,a.wall_minutes*60)
        try:execute(a,p,h0,runtime,result)
        except BaseException as exc:
            result.update(status='INCOMPLETE_NO_SCIENTIFIC_CONCLUSION',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous);write(a.output_dir/'results.json',result)


if __name__=='__main__':main()
