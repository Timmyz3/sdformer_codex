#!/usr/bin/env python3
"""Two fixed original checkpoints x check8 native all78current; exactly16 calls."""
import argparse
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
KDIR=HERE.parent/'later_checkpoint_deployment'
SCHEMA='flow_bn_conversion_round14_v1'
OWN=('run_conversion_v2.py','conversion_core.py','test_conversion_v2_cpu.py','source_pins.json','PROTOCOL.md')
import conversion_core as core
require=core.require


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def runtime():
    require(sha(KDIR/'run_deployment.py')=='93ffc0ed503ea62d67a1d4f25e26c47800b122c0e5d0f5578ff35eb4069fa140','Frozen K helper changed')
    sys.path.insert(0,str(KDIR));k=load(KDIR/'run_deployment.py','_r14_source_helpers_only')
    h0,modules=k.load_runtime() # No checkpoint/BN load, prepare, execute or forward.
    for relative,digest in read(HERE/'source_pins.json')['sources'].items():
        require(sha(h0.dependency_path(relative))==digest,'Frozen source changed '+relative)
    return k,h0,modules


def asset(spec):
    require(set(spec)=={'path','sha256'} and len(spec['sha256'])==64,'Explicit path/SHA asset required')
    path=Path(spec['path']);require(path.is_file() and sha(path)==spec['sha256'],'Actual asset bytes differ: '+str(path));return path


def validate_manifest_sources(members,canonical,roles):
    # The frozen manifest also binds eight historical provenance files. Those
    # keys are legitimate; validate the two actual data sources required here.
    require(members['source_sha256']['canonical']==sha(canonical) and
            members['source_sha256']['roles']==sha(roles),'Canonical/role source bytes')


def validate_checkpoint(path,step,context,h0,torch):
    payload=torch.load(path,map_location='cpu',weights_only=True);run,side=context[6:8]
    require(payload['schema_version']=='flow_a_native_encoding_v2_checkpoint_v1' and payload['contract_sha256']==side['contract_sha256']=='961f1700e76f3427c74d4e9af9799a684433ff00449b8386be6c1f9cdd75e934','Fixed native contract required')
    require(all(payload[k]==run[k] for k in ('contract','provenance','effective_config')),'Source/encoding/config differs')
    require(payload['training']['completed_optimizer_steps']==int(step)==side['completed_optimizer_steps'],'Wrong original checkpoint step')
    state=payload['model_state_dict'];require(all(isinstance(v,torch.Tensor) and v.device.type=='cpu' and bool(torch.isfinite(v).all()) for v in state.values()),'Finite CPU state required')
    require(sum(v.numel() for v in state.values())==55001588,'Original state element count')
    audit=core.strip_bn_state(state,h0.state_dict_digest,core.SOURCE_STATE_SHA[step],core.CURRENT_STATE_SHA[step])
    audit.update(checkpoint_sha256=sha(path),checkpoint_step=int(step),contract_sha256=payload['contract_sha256'],
                 sidecar_sha256=sha(path.with_suffix('.json')),run_contract_sha256=sha(path.parent/'run_contract.json'),optimizer_restored=False,
                 external_BN_artifact_loaded=False,source_BN_buffers_removed_before_inference=True)
    return audit


def verify_cache_arrays(root,descriptor,np):
    for f in descriptor['cached_native_predictions']:
        name=f['prediction_file'];require(name==Path(name).name,'Cache artifact must be sibling basename')
        path=root/name;require(sha(path)==f['prediction_file_sha256'],'Cache actual prediction archive SHA')
        with np.load(path,allow_pickle=False) as values:
            a=values['native'];require(a.shape==(2,480,640) and a.dtype==np.float32 and np.isfinite(a).all(),'Cache native prediction shape/finite')
            require(hashlib.sha256(a.tobytes()).hexdigest()==f['prediction_sha256'],'Cache actual native prediction payload SHA')


def prepare(a,k,h0,modules):
    import numpy as np
    import torch
    require(not torch.cuda.is_initialized(),'Preflight CPU only')
    require(0<a.wall_minutes<=15 and not a.output_dir.exists(),'Fresh output/max15min required')
    require(sha(a.inputs_manifest)==a.inputs_manifest_sha256,'Input manifest source differs')
    inputs=read(a.inputs_manifest);require(inputs['schema']=='flow_bn_conversion_round14_inputs_v1','Inputs schema')
    require(set(inputs['checkpoints'])==set(core.CHECKPOINTS) and set(inputs['references'])==set(core.CHECKPOINTS),'Only fixed2 original sources and caches')
    rtpath=asset(inputs['runtime_manifest']);memberpath=asset(inputs['membership']);rt=read(rtpath);members=read(memberpath)
    require(rt['schema']=='flow_round13_runtime_v1' and rt['region_sha256']==k.REGION_SHA and sha(memberpath)==k.MEMBER_SHA,'Fixed R13 cohort/runtime')
    for path,digest in rt['source_pins'].items():require(sha(path)==digest,'Runtime dependency source differs')
    sys.path[:0]=[rt['breadth_root'],rt['motion_root']]
    base=load(Path(rt['breadth_root'])/'evaluate_breadth.py','_r14_native_source_preflight')
    r13,collector,probe,parent,r9,r8=modules
    cpu_sha=parent.checked_cpu_receipt(a.cpu_checks,[HERE/n for n in OWN])
    common={'membership_sha256':sha(memberpath),'runtime_manifest_sha256':sha(rtpath),'region_sha256':k.REGION_SHA}
    prepared={};reference_bindings={};data_evidence=[]
    for step in core.CHECKPOINTS:
        cp=asset(inputs['checkpoints'][step]);require(sha(cp)==core.CHECKPOINT_SHA[step],'Only specified actual original checkpoint SHA')
        args=list(rt['base_args']);require(not any(x.startswith('--bn-') or x.startswith('--svs-') or x.startswith('--regions') or x=='--run-inference' for x in args),'No imported BN/calibration/inference args')
        for flag,value in (('--checkpoint',str(cp)),('--checkpoint-sha256',core.CHECKPOINT_SHA[step]),('--output',str(a.output_dir/'results.json'))):
            require(args.count(flag)==1,'One native argument '+flag);args[args.index(flag)+1]=value
        b=base.parse_args(args);b.device='cuda:0';b.wall_minutes=a.wall_minutes
        require(b.geometry=='full9' and not b.run_inference,'Fixed native full9/no implicit execution')
        context=list(base.preflight(b));native,trainer,metrics,_,hashes,config,run,side,_,unused_plan=context
        canonical=metrics.canonical_rows(b.canonical);roles=metrics.auxiliary_roles(b.roles,canonical)
        r13.validate_members(members,canonical,roles)
        validate_manifest_sources(members,b.canonical,b.roles)
        regionpath=Path(rt['region_path']);require(sha(regionpath)==k.REGION_SHA,'Frozen training region bytes')
        regions=read(regionpath);metrics.validate_region_contract(regions,canonical,roles,require_complete_fit=False)
        require(regions['fit_target_ids']==[r['target_id'] for r in members['train_members_diversity']],'D200 region source order')
        byid={r['target_id']:r for r in canonical}
        require([r['target_id'] for r in regions['data_evidence']]==regions['fit_target_ids'],'Complete frozen region source evidence')
        for e in regions['data_evidence']:
            files=native.target_files(byid[e['target_id']]);require(all(e[q+'_sha256']==hashes[files[q]]['sha256'] for q in ('gt','mask')),'Region source GT/mask')
        context[8]=regions
        refspec=inputs['references'][step];require(set(refspec)=={'result','execution_audit'},'Exact reference metadata assets')
        refpath=asset(refspec['result']);auditpath=asset(refspec['execution_audit'])
        require(sha(refpath)==core.CACHE_SHA[step] and sha(auditpath)==core.AUDIT_SHA[step],'Only actual frozen same-W cache and independent audit')
        ref=read(refpath);audit=read(auditpath)
        descriptor=core.reference_contract(step,ref,sha(refpath),audit,members,common)
        verify_cache_arrays(refpath.parent,descriptor,np)
        descriptor.update(cache_result_path=str(refpath),execution_audit_sha256=sha(auditpath),execution_audit_path=str(auditpath))
        for row,f in zip(members['check_members'],descriptor['cached_native_predictions']):
            files=native.target_files(row);evidence={('events' if key=='input' else key):hashes[value]['sha256'] for key,value in files.items()}
            require(evidence==f['input_evidence'],'Cached native actual input/GT/mask differs')
            require(all(sha(native.checked_path(b.data_root,v))==hashes[v]['sha256'] for v in files.values()),'Actual current check source bytes')
            data_evidence.append({'checkpoint_step':int(step),'target_id':row['target_id'],'sequence':row['sequence'],'input_evidence':evidence})
        sourceaudit=validate_checkpoint(cp,step,context,h0,torch)
        prepared[step]={'b':b,'context':tuple(context),'checkpoint_cpu_audit':sourceaudit,'cache_runtime':ref['runtime']}
        reference_bindings[step]=descriptor
    binding={'schema':'flow_bn_conversion_round14_binding_v1','source_sha256':{n:sha(HERE/n) for n in OWN},'cpu_checks_sha256':cpu_sha,
             'inputs_manifest_sha256':a.inputs_manifest_sha256,**common,'check_ids':[r['target_id'] for r in members['check_members']],
             'checkpoint_cpu_audits':{step:prepared[step]['checkpoint_cpu_audit'] for step in core.CHECKPOINTS},'references':reference_bindings,
             'forward_plan':core.plan(members),'total_full_forwards':16,'optimizer_steps':0,'calibration_forwards':0,'dual_forwards':0,
             'BN_scope':'ALL78_CURRENT','precision':'FP32/noAMP/noTF32/T10/B1/full9','GPU_lock':str(parent.GPU_LOCK),'wall_minutes':a.wall_minutes,
             'scope':'Two fixed original weights, samecheck8 native current-stat vs audited cached private24 native. No quality-based checkpoint selection.'}
    return {'binding':binding,'members':members,'prepared':prepared,'runtime_manifest':rt,'base':base,'data_evidence':data_evidence}


def execute(a,p,k,h0,modules,result):
    import numpy as np
    import torch
    require(not torch.cuda.is_initialized(),'CUDA only after shared lock')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261016)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.manual_seed_all(20261016)
    started=time.monotonic()
    def deadline():require(time.monotonic()-started<a.wall_minutes*60,'Fixed16 wall cap exhausted')
    r13,collector,probe,parent,r9,r8=modules;rt=p['runtime_manifest'];base=p['base']
    loader=load(Path(rt['breadth_root'])/'run_same_shape_reference.py','_r14_frozen_native_loader')
    scope=load(Path(rt['breadth_root'])/'run_bn_scope.py','_r14_frozen_inventory')
    result.update(status='RUNNING',inference_executed=True,full_forward_count=0,forward_records=[],checkpoints={},
                  runtime={'torch':str(torch.__version__),'numpy':np.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
                           'device':torch.cuda.get_device_name(device),'TF32':False,'AMP':False,'threads':2})
    for step in core.CHECKPOINTS:
        deadline();case=p['prepared'][step];b=case['b'];context=case['context'];cached=case['cache_runtime']
        require(cached['torch']==result['runtime']['torch'] and cached['cuda']==result['runtime']['cuda']
                and cached.get('numpy',cached.get('NumPy'))==np.__version__ and cached.get('device',cached.get('device_name'))==result['runtime']['device'],
                'Same cached native runtime/device required')
        if 'cudnn' in cached:require(cached['cudnn']==result['runtime']['cudnn'],'Same cached cuDNN required')
        native,trainer,metrics,_,hashes,config,run,side,regions,unused_plan=context
        model,functional=loader.load_model(b,context,base,torch,device);model.requires_grad_(False)
        require(r13.state_digest(model)==core.SOURCE_STATE_SHA[step],'Actual source model changed after CPU preflight')
        targets,shapes=parent.target_layers(model);merge=parent.load_frozen_merge()
        bn=core.switch_allcurrent(model,base,scope,r13.state_digest,torch,core.CURRENT_STATE_SHA[step])
        before=r13.state_digest(model);signature=base.bn_signature(model,torch)
        item={'checkpoint_step':int(step),'checkpoint_sha256':core.CHECKPOINT_SHA[step],'source_state_sha256':core.SOURCE_STATE_SHA[step],
              'initial_allcurrent_state_sha256':before,'BN_scope':bn,'evaluations':{'endpoint':{'frames':[]}},'full_forward_count':0}
        result['checkpoints'][step]=item;meter=metrics.MetricAccumulator();case_started=time.monotonic()
        with torch.inference_mode():
            for index,row in enumerate(p['members']['check_members']):
                deadline();expected={'events' if q=='input' else q:hashes[v]['sha256'] for q,v in native.target_files(row).items()};evidence={}
                arrays,_=metrics.load_arrays(b.data_root/'saved_flow_data',row,evidence=evidence,expected_hashes=expected)
                cached_frame=p['binding']['references'][step]['cached_native_predictions'][index]
                require(evidence==cached_frame['input_evidence'],'Actual inference input differs from same-W cached native')
                masks=metrics.partitions(arrays['gt'],arrays['mask'],arrays['events'],regions);chunk=base.prepare_input(arrays['events'],device,torch)
                flows,execution,branches,inputs=h0.forward(modules,model,functional,targets,shapes,chunk,'native',merge,torch,deadline=deadline)
                require(len(flows)==4 and execution['layers']=={} and execution['summed_fc1_rows']==execution['summed_fc2_rows']==558000,'Only true full native forward')
                prediction=flows[-1][0].numpy().copy();name='current_'+step+'_'+row['target_id']+'.npz';path=a.output_dir/name
                np.savez_compressed(path,native=prediction)
                frame={'target_id':row['target_id'],'sequence':row['sequence'],'input_evidence':evidence,'prediction_file':name,'prediction_file_sha256':sha(path),
                       'arms':{'native':{'metrics':meter.update(row['target_id'],row['sequence'],prediction,arrays['gt'],masks),
                                        'prediction_sha256':hashlib.sha256(prediction.tobytes()).hexdigest(),'execution':execution}},
                       'paired_cached_native':cached_frame}
                item['evaluations']['endpoint']['frames'].append(frame)
                planned=p['binding']['forward_plan'][result['full_forward_count']]
                require(planned['checkpoint_step']==int(step) and planned['target_id']==row['target_id'],'Fixed16 order changed')
                result['forward_records'].append(dict(planned,execution=execution));result['full_forward_count']+=1;item['full_forward_count']+=1
                k.core.check_closure(model,targets,before,r13.state_digest,signature,base.bn_signature,torch)
                del arrays,chunk,flows,branches,inputs,prediction
                print(json.dumps({'checkpoint':step,'completed':index+1,'total_forward_count':result['full_forward_count']}),flush=True)
        item['evaluations']['endpoint']['summaries']={'native':meter.summary()}
        item.update(final_allcurrent_state_sha256=r13.state_digest(model),state_unchanged=True,BN_buffers_unchanged=True,
                    original_modules_restored=True,all_hooks_removed=True,wall_seconds=time.monotonic()-case_started)
        require(item['full_forward_count']==8 and item['final_allcurrent_state_sha256']==before,'Complete unchanged same-W8 required')
        del model,targets,functional;torch.cuda.empty_cache();write(a.output_dir/'results.json',result)
    require(result['full_forward_count']==16 and set(result['checkpoints'])==set(core.CHECKPOINTS),'Complete two-source16 only')
    result.update(status='ROUND14_BN_CONVERSION_COMPLETE',wall_seconds=time.monotonic()-started,
                  max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved(),
                  all_states_unchanged=True,all_BN_buffers_removed=True,all_hooks_removed=True,original_modules_restored=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-manifest',type=Path,required=True);parser.add_argument('--inputs-manifest-sha256',required=True)
    parser.add_argument('--cpu-checks',type=Path,default=HERE/'cpu_results_v2.json');parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--preflight-output',type=Path);parser.add_argument('--wall-minutes',type=float,default=15);parser.add_argument('--execute',action='store_true')
    a=parser.parse_args();k,h0,modules=runtime();p=prepare(a,k,h0,modules)
    result={'schema':SCHEMA,'status':'PREFLIGHT_PASS_NO_INFERENCE','binding':p['binding'],'data_evidence':p['data_evidence'],
            'inference_executed':False,'training_performed':False,'optimizer_steps':0,'calibration_forwards':0,'dual_forwards':0,
            'flow_specific_increment_claim':False,'hardware_benefit_claim':False}
    if not a.execute:
        require(a.preflight_output and not a.preflight_output.exists(),'Fresh explicit CPU preflight receipt')
        require(not a.preflight_output.resolve().is_relative_to(a.output_dir.resolve()),'Preflight outside future run directory')
        write(a.preflight_output,result);print(result['status']);return
    lockpath=modules[3].GPU_LOCK;require(lockpath.parent.is_dir(),'Existing shared GPU lock directory')
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);a.output_dir.mkdir(parents=True,exist_ok=False)
        previous=signal.getsignal(signal.SIGALRM)
        def timeout(signum,frame):raise TimeoutError('Fixed16 conversion15min wall exhausted')
        signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,a.wall_minutes*60)
        try:execute(a,p,k,h0,modules,result)
        except BaseException as exc:
            result.update(status='INCOMPLETE_NO_SCIENTIFIC_CONCLUSION',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous);write(a.output_dir/'results.json',result)


if __name__=='__main__':main()
