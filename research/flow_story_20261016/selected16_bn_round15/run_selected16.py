#!/usr/bin/env python3
"""R15 S8: original271482 + own selected16 BN; eight native forwards only."""
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
R14DIR=HERE.parent/'bn_conversion_round14'
OWN=('run_selected16.py','selected_core.py','source_state_cpu.py','test_selected16_cpu.py','expected_state.json','source_pins.json','PROTOCOL.md')
import selected_core as core
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
    require(sha(R14DIR/'run_conversion_v2.py')=='622a7536ffa9b887f11c4294e3a96d53573701c678fb37f8edb36745a7bf813b','Frozen R14 V2 helpers required')
    sys.path.insert(0,str(R14DIR));old=load(R14DIR/'run_conversion_v2.py','_r15_readonly_source_helpers')
    k,h0,modules=old.runtime() # No prepare/execute/checkpoint/BN/inference call.
    for relative,digest in read(HERE/'source_pins.json')['sources'].items():
        require(sha(h0.dependency_path(relative))==digest,'Frozen inherited helper changed '+relative)
    return old,k,h0,modules


def prepare(a,old,k,h0,modules):
    import numpy as np
    import torch
    require(not torch.cuda.is_initialized(),'CPU preflight only')
    torch.set_num_threads(2)
    require(0<a.wall_minutes<=15 and not a.output_dir.exists(),'Fresh output/max15min required')
    require(sha(a.inputs_manifest)==a.inputs_manifest_sha256,'Input manifest bytes differ')
    inputs=read(a.inputs_manifest);require(inputs['schema']=='flow_selected16_bn_round15_inputs_v1','R15 inputs schema')
    require(set(inputs)=={'schema','runtime_manifest','membership','checkpoint','own_BN_buffers','own_BN_receipt','references'},'Exact S8 inputs only')
    require(set(inputs['references'])=={'private24','current78'},'Both fixed same-W caches required')
    rtpath=old.asset(inputs['runtime_manifest']);memberpath=old.asset(inputs['membership']);rt=read(rtpath);members=read(memberpath)
    require(rt['schema']=='flow_round13_runtime_v1' and rt['region_sha256']==k.REGION_SHA and sha(memberpath)==k.MEMBER_SHA,'Fixed common R13 input contract')
    for path,digest in rt['source_pins'].items():require(sha(path)==digest,'Runtime dependency changed')
    sys.path[:0]=[rt['breadth_root'],rt['motion_root']]
    base=load(Path(rt['breadth_root'])/'evaluate_breadth.py','_r15_native_source_preflight')
    r13,collector,probe,parent,r9,r8=modules
    cpu_sha=parent.checked_cpu_receipt(a.cpu_checks,[HERE/n for n in OWN])
    common={'membership_sha256':sha(memberpath),'runtime_manifest_sha256':sha(rtpath),'region_sha256':k.REGION_SHA}
    cp=old.asset(inputs['checkpoint']);require(sha(cp)==core.CHECKPOINT_SHA,'Only original271482')
    args=list(rt['base_args']);require(not any(x.startswith('--bn-') or x.startswith('--svs-') or x.startswith('--regions') or x=='--run-inference' for x in args),'No implicit BN/calibration/inference args')
    for flag,value in (('--checkpoint',str(cp)),('--checkpoint-sha256',core.CHECKPOINT_SHA),('--output',str(a.output_dir/'results.json'))):
        require(args.count(flag)==1,'Single native argument '+flag);args[args.index(flag)+1]=value
    b=base.parse_args(args);b.device='cuda:0';b.wall_minutes=a.wall_minutes
    require(b.geometry=='full9' and not b.run_inference,'Full9/no implicit execution')
    context=list(base.preflight(b));native,trainer,metrics,_,hashes,config,run,side,_,unused_plan=context
    canonical=metrics.canonical_rows(b.canonical);roles=metrics.auxiliary_roles(b.roles,canonical)
    r13.validate_members(members,canonical,roles);old.validate_manifest_sources(members,b.canonical,b.roles)
    regionpath=Path(rt['region_path']);require(sha(regionpath)==k.REGION_SHA,'Frozen D200 region bytes')
    regions=read(regionpath);metrics.validate_region_contract(regions,canonical,roles,require_complete_fit=False)
    require(regions['fit_target_ids']==[r['target_id'] for r in members['train_members_diversity']],'D200 region source order')
    byid={r['target_id']:r for r in canonical}
    require([e['target_id'] for e in regions['data_evidence']]==regions['fit_target_ids'],'Full region evidence order')
    for e in regions['data_evidence']:
        files=native.target_files(byid[e['target_id']]);require(all(e[q+'_sha256']==hashes[files[q]]['sha256'] for q in ('gt','mask')),'Region source GT/mask')
    context[8]=regions
    references={};runtimes={}
    for label,resultsha,auditsha in (('private24',core.K_RESULT_SHA,core.K_AUDIT_SHA),('current78',core.CURRENT_RESULT_SHA,core.CURRENT_AUDIT_SHA)):
        spec=inputs['references'][label];require(set(spec)=={'result','execution_audit'},'Exact reference assets')
        rp=old.asset(spec['result']);ap=old.asset(spec['execution_audit'])
        require(sha(rp)==resultsha and sha(ap)==auditsha,'Fixed actual same-W cache/audit bytes')
        value=read(rp);audit=read(ap)
        desc=old.core.reference_contract(core.STEP,value,resultsha,audit,members,common) if label=='private24' else core.current_reference(value,audit,members,common)
        old.verify_cache_arrays(rp.parent,desc,np)
        desc.update(cache_result_path=str(rp),execution_audit_path=str(ap),execution_audit_sha256=auditsha)
        references[label]=desc;runtimes[label]=value['runtime']
    data_evidence=[]
    for index,row in enumerate(members['check_members']):
        files=native.target_files(row);evidence={'events' if q=='input' else q:hashes[v]['sha256'] for q,v in files.items()}
        require(all(evidence==references[label]['cached_native_predictions'][index]['input_evidence'] for label in references),'Same actual inputs in both caches')
        require(all(sha(native.checked_path(b.data_root,v))==hashes[v]['sha256'] for v in files.values()),'Current input actual bytes')
        data_evidence.append({'checkpoint_step':271482,'target_id':row['target_id'],'sequence':row['sequence'],'input_evidence':evidence})
    sourceaudit=old.validate_checkpoint(cp,core.STEP,context,h0,torch)
    bnpath=old.asset(inputs['own_BN_buffers']);receiptpath=old.asset(inputs['own_BN_receipt'])
    require(sha(bnpath)==core.OWN_BN_SHA and sha(receiptpath)==core.OWN_RECEIPT_SHA,'Original K own72/receipt bytes')
    core.validate_own_receipt(read(receiptpath),members,common)
    with np.load(bnpath,allow_pickle=False) as z:values={n:z[n].copy() for n in z.files}
    payload=torch.load(cp,map_location='cpu',weights_only=True)
    state,chosen,stateaudit=core.construct_state(payload['model_state_dict'],values,h0.state_dict_digest,torch,np)
    expected=read(HERE/'expected_state.json')
    require(expected['schema']=='flow_selected16_expected_state_v1' and expected['status']=='PASS' and stateaudit==expected['selected_state'],'Actual CPU reconstructed525 differs from frozen source preparation')
    require(stateaudit['selected16_tensor_count']==525,'Actual525 state required')
    del payload,state
    binding={'schema':'flow_selected16_bn_round15_binding_v1','source_sha256':{n:sha(HERE/n) for n in OWN},'cpu_checks_sha256':cpu_sha,
             'inputs_manifest_sha256':a.inputs_manifest_sha256,**common,'checkpoint_sha256':core.CHECKPOINT_SHA,'checkpoint_step':271482,
             'checkpoint_cpu_audit':sourceaudit,'selected_state_cpu_audit':stateaudit,
             'own_BN_buffers_sha256':sha(bnpath),'own_BN_receipt_sha256':sha(receiptpath),'references':references,
             'check_ids':[r['target_id'] for r in members['check_members']],'forward_plan':core.plan(members),'total_full_forwards':8,
             'optimizer_steps':0,'calibration_forwards':0,'dual_forwards':0,'BN_scope':core.BN_SCOPE,
             'precision':'FP32/noAMP/noTF32/T10/B1/full9','GPU_lock':str(parent.GPU_LOCK),'wall_minutes':a.wall_minutes,
             'native_forward_observer':{'frozen_h0_core_sha256':sha(h0.__file__),
                 'adaptation':'Private forward globals; only exact Unremoved forward hooks guard accepts the known78 observer pre-hook ids. All other guards unchanged; finally removes every observer.',
                 'historical_module_globals_modified':False,'extra_full_forwards':0},
             'scope':'Preselected271482 own-CMA selected16/native8 diagnostic; no FF, training, recalibration, mask selection or automatic next-stage admission.'}
    return {'binding':binding,'members':members,'b':b,'context':tuple(context),'runtime_manifest':rt,'base':base,
            'data_evidence':data_evidence,'cache_runtimes':runtimes,'own_values':values,'selected_values':chosen}


def execute(a,p,old,k,h0,modules,result):
    import numpy as np
    import torch
    require(not torch.cuda.is_initialized(),'CUDA only after shared lock')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261016)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.manual_seed_all(20261016)
    started=time.monotonic()
    def deadline():require(time.monotonic()-started<a.wall_minutes*60,'Fixed8 wall cap exhausted')
    r13,collector,probe,parent,r9,r8=modules;rt=p['runtime_manifest'];base=p['base'];b=p['b'];context=p['context']
    result.update(status='RUNNING',inference_executed=True,full_forward_count=0,forward_records=[],
                  runtime={'torch':str(torch.__version__),'numpy':np.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
                           'device':torch.cuda.get_device_name(device),'TF32':False,'AMP':False,'threads':2})
    for cached in p['cache_runtimes'].values():
        require(cached['torch']==result['runtime']['torch'] and cached['cuda']==result['runtime']['cuda']
                and cached.get('numpy',cached.get('NumPy'))==np.__version__ and cached.get('device',cached.get('device_name'))==result['runtime']['device'],'Same cached runtime/device')
        if 'cudnn' in cached:require(cached['cudnn']==result['runtime']['cudnn'],'Same cached cuDNN')
    loader=load(Path(rt['breadth_root'])/'run_same_shape_reference.py','_r15_frozen_native_loader')
    scope=load(Path(rt['breadth_root'])/'run_bn_scope.py','_r15_frozen_inventory')
    native,trainer,metrics,_,hashes,config,run,side,regions,unused_plan=context
    model,functional=loader.load_model(b,context,base,torch,device);model.requires_grad_(False)
    require(r13.state_digest(model)==core.SOURCE_SHA,'Actual original W source digest')
    targets,shapes=parent.target_layers(model);merge=parent.load_frozen_merge()
    expected=p['binding']['selected_state_cpu_audit']['selected16_state_sha256']
    bn,chosen,stateaudit=core.install(model,p['own_values'],scope,base,h0.state_dict_digest,torch,np,expected)
    require(stateaudit==p['binding']['selected_state_cpu_audit'],'Actual installation differs from CPU reconstruction')
    before=r13.state_digest(model);signature=base.bn_signature(model,torch)
    path=a.output_dir/'selected_bn_buffers.npz';np.savez_compressed(path,**chosen)
    result['retained_BN_buffers']={'file':path.name,'sha256':sha(path),'keys':sorted(chosen),'buffer_tensor_sha256':stateaudit['retained_buffer_tensor_sha256'],
                                   'source_own72_sha256':core.OWN_BN_SHA,'source_own_receipt_sha256':core.OWN_RECEIPT_SHA}
    item={'checkpoint_step':271482,'checkpoint_sha256':core.CHECKPOINT_SHA,'source_state_sha256':core.SOURCE_SHA,
          'initial_selected16_state_sha256':before,'BN_scope':bn,'evaluations':{'endpoint':{'frames':[]}},'full_forward_count':0}
    result['checkpoints']={core.STEP:item};meter=metrics.MetricAccumulator()
    with torch.inference_mode():
        for index,row in enumerate(p['members']['check_members']):
            deadline();expected_hashes={'events' if q=='input' else q:hashes[v]['sha256'] for q,v in native.target_files(row).items()};evidence={}
            arrays,_=metrics.load_arrays(b.data_root/'saved_flow_data',row,evidence=evidence,expected_hashes=expected_hashes)
            cached_frames={label:desc['cached_native_predictions'][index] for label,desc in p['binding']['references'].items()}
            require(all(evidence==f['input_evidence'] for f in cached_frames.values()),'Actual same-W cached inputs')
            masks=metrics.partitions(arrays['gt'],arrays['mask'],arrays['events'],regions);chunk=base.prepare_input(arrays['events'],device,torch)
            forward,bn_calls=core.native_with_bn_records(lambda native_call:native_call(modules,model,functional,targets,shapes,chunk,'native',merge,torch,deadline=deadline),model,torch,h0)
            flows,execution,branches,inputs=forward
            require(len(flows)==4 and execution['layers']=={} and execution['summed_fc1_rows']==execution['summed_fc2_rows']==558000,'True full native call only')
            execution['all78_BN_observations']=bn_calls
            prediction=flows[-1][0].numpy().copy();name='selected16_'+core.STEP+'_'+row['target_id']+'.npz';path=a.output_dir/name
            np.savez_compressed(path,native=prediction)
            frame={'target_id':row['target_id'],'sequence':row['sequence'],'input_evidence':evidence,'prediction_file':name,'prediction_file_sha256':sha(path),
                   'arms':{'native':{'metrics':meter.update(row['target_id'],row['sequence'],prediction,arrays['gt'],masks),
                                    'prediction_sha256':hashlib.sha256(prediction.tobytes()).hexdigest(),'execution':execution}},'paired_cached_native':cached_frames}
            item['evaluations']['endpoint']['frames'].append(frame)
            planned=p['binding']['forward_plan'][index]
            require(planned['target_id']==row['target_id'],'Fixed8 order changed')
            result['forward_records'].append(dict(planned,execution=execution));result['full_forward_count']+=1;item['full_forward_count']+=1
            k.core.check_closure(model,targets,before,r13.state_digest,signature,base.bn_signature,torch)
            del arrays,chunk,forward,flows,branches,inputs,prediction
            print(json.dumps({'completed':index+1,'total_forward_count':result['full_forward_count'],'BN_scope':core.BN_SCOPE}),flush=True)
    item['evaluations']['endpoint']['summaries']={'native':meter.summary()}
    item.update(final_selected16_state_sha256=r13.state_digest(model),state_unchanged=True,BN_buffers_unchanged=True,original_modules_restored=True,all_hooks_removed=True)
    require(item['full_forward_count']==result['full_forward_count']==8 and item['final_selected16_state_sha256']==before,'Complete unchanged fixed8')
    result.update(status='ROUND15_SELECTED16_S8_COMPLETE',wall_seconds=time.monotonic()-started,
                  max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved(),
                  all_states_unchanged=True,BN_buffers_unchanged=True,all_hooks_removed=True,original_modules_restored=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-manifest',type=Path,required=True);parser.add_argument('--inputs-manifest-sha256',required=True)
    parser.add_argument('--cpu-checks',type=Path,default=HERE/'cpu_results.json');parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--preflight-output',type=Path);parser.add_argument('--wall-minutes',type=float,default=15);parser.add_argument('--execute',action='store_true')
    a=parser.parse_args();old,k,h0,modules=runtime();p=prepare(a,old,k,h0,modules)
    result={'schema':'flow_selected16_bn_round15_v1','status':'PREFLIGHT_PASS_NO_INFERENCE','binding':p['binding'],'data_evidence':p['data_evidence'],
            'inference_executed':False,'training_performed':False,'optimizer_steps':0,'calibration_forwards':0,'dual_forwards':0,
            'flow_specific_increment_claim':False,'hardware_benefit_claim':False,'automatic_FF_H0_admission':False}
    if not a.execute:
        require(a.preflight_output and not a.preflight_output.exists(),'Fresh explicit CPU preflight receipt')
        require(not a.preflight_output.resolve().is_relative_to(a.output_dir.resolve()),'Preflight outside future run directory')
        write(a.preflight_output,result);print(result['status']);return
    lockpath=modules[3].GPU_LOCK;require(lockpath.parent.is_dir(),'Existing shared GPU lock directory')
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);a.output_dir.mkdir(parents=True,exist_ok=False)
        previous=signal.getsignal(signal.SIGALRM)
        def timeout(signum,frame):raise TimeoutError('Fixed8 selected16 15min wall exhausted')
        signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,a.wall_minutes*60)
        try:execute(a,p,old,k,h0,modules,result)
        except BaseException as exc:
            result.update(status='INCOMPLETE_NO_SCIENTIFIC_CONCLUSION',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous);write(a.output_dir/'results.json',result)


if __name__=='__main__':main()
