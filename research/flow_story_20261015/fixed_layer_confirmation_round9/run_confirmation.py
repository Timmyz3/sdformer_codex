#!/usr/bin/env python3
"""Frozen single-layer ordinary baseline on 16 identity-new members, 69 forwards."""
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
R8_PATH=HERE.parent/'layer_repair_round8/run_layer_repair.py'
R8_SHA='93c8e846be3f28cdfab916e240ac251941be56e6cb5d3471c143d3d8291b64bc'
ARMS=('native','FF','CC','fine_s2b2')
ROWS={'native':558000,'FF':558000,'CC':153600,'fine_s2b2':164200}
BUDGET=69


def require(ok,msg):
    if not ok:raise ValueError(msg)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def load_r8():
    require(sha(R8_PATH)==R8_SHA,'Frozen R8 runner differs')
    spec=importlib.util.spec_from_file_location('_r9_frozen_r8',R8_PATH)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def forward_arm(r8,parent,model,functional,targets,shapes,chunk,arm,merge,torch,**kwargs):
    require(arm in ARMS,'Fixed four arms only')
    if arm=='native':return parent.forward_probe(model,functional,targets,shapes,chunk,'native',(),merge,torch,**kwargs)
    if arm=='FF':return parent.forward_probe(model,functional,targets,shapes,chunk,'fine',(0,2),merge,torch,**kwargs)
    return r8.mixed_forward(parent,model,functional,targets,shapes,chunk,() if arm=='CC' else (2,),merge,torch,**kwargs)


def prepare(a,r8,collector,probe,parent):
    import select_members
    require(0<a.wall_minutes<=10,'At most10 minutes')
    inherited=argparse.Namespace(**vars(a));inherited.runner_checks=a.collector_runner_checks;inherited.group='check'
    first=collector.prepare(inherited,probe,parent)
    inherited.selection_manifest=a.second_manifest;inherited.selection_manifest_sha256=a.second_manifest_sha256
    second=collector.prepare(inherited,probe,parent)
    require(sha(a.prior_round6_selection)==a.prior_round6_selection_sha256=='c20684f58702034887f6454073f993b6309182d99bea763398fa16ba39b6541a','Prior selection differs')
    require(sha(a.membership)==a.membership_sha256,'R9 identity manifest differs')
    prior=json.loads(a.prior_round6_selection.read_text());membership=json.loads(a.membership.read_text())
    c,b,context,base,scope,loader,chosen,oldbinding,oldhead=first
    metrics=context[2];canonical=metrics.canonical_rows(b.canonical);roles=metrics.auxiliary_roles(b.roles,canonical)
    expected=select_members.select(canonical,roles,prior)
    require([r['target_id'] for r in expected]==membership['target_ids'],'Identity selection does not replay')
    actual=chosen+second[6];require(len(actual)==16 and len({r['target_id'] for g,r in actual})==16,'Duplicate or missing new members')
    byid={r['target_id']:(g,r) for g,r in actual}
    require(set(byid)==set(membership['target_ids']),'Preflight batches differ from fixed union16')
    chosen=[byid[tid] for tid in membership['target_ids']]
    audit=oldbinding['new_member_data_preflight']+second[7]['new_member_data_preflight']
    check=parent.checked_cpu_receipt(a.runner_checks,[Path(__file__),HERE/'test_confirmation_cpu.py',HERE/'select_members.py'])
    require(sha(a.reference_r8)==a.reference_r8_sha256,'R8 reference differs')
    ref=json.loads(a.reference_r8.read_text());require(ref['status']=='ROUND8_SINGLE_LAYER_DIAGNOSTIC_COMPLETE','R8 incomplete')
    anchor=ref['frames'][0];anchor_row=next(r for r in canonical if r['target_id']==anchor['target_id'])
    require(anchor_row['target_id'] not in byid,'Replay member overlaps new quality cohort')
    binding={'runner_sha256':sha(__file__),'runner_checks_sha256':check,'R8_source_sha256':R8_SHA,
             'inherited_batch_a':oldbinding,'inherited_batch_b':second[7],
             'membership':membership,'membership_sha256':sha(a.membership),'reference_r8_sha256':sha(a.reference_r8),
             'new_member_data_preflight':audit,'arms':ARMS,'full_forward_budget':BUDGET,'GPU_lock':str(parent.GPU_LOCK),
             'scope':'Fixed ordinary static layer chosen only on R8. New members in old backbone-seen development sequences; no score fitting, official test or hardware benefit'}
    return (c,b,context,base,scope,loader,chosen,oldhead,ref,anchor_row),binding


def execute(a,prepared,result,r8,parent):
    import numpy as np
    import torch
    inherited,binding=prepared;c,b,context,base,scope,loader,chosen,oldhead,reference,anchor_row=inherited
    require(not torch.cuda.is_initialized(),'CUDA before lock')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261014)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.manual_seed_all(20261014)
    started=time.monotonic()
    def deadline():require(time.monotonic()-started<a.wall_minutes*60,'Wall budget exhausted')
    native,trainer,metrics,dev32,hashes,config,saved,side,regions,plan=context
    model,functional=loader.load_model(b,context,base,torch,device)
    bn_info=scope.apply_scope(model,'MLP_CMA176',b.bn_payload,torch);model.eval().requires_grad_(False)
    targets,shapes=parent.target_layers(model);merge=parent.load_frozen_merge()
    before,bn_before=parent.source_digest(model),base.bn_signature(model,torch)
    result.update(status='RUNNING',original_state_sha256=before,BN_scope=bn_info,frames=[],full_forward_count=0,
                  anchor_replays={},dual_checks={},runtime={'torch':str(torch.__version__),'cuda':torch.version.cuda,
                  'device_name':torch.cuda.get_device_name(device),'TF32':False,'AMP':False,'threads':2})
    require(before==reference['original_state_sha256'],'R8 model numerical state differs')
    meters={arm:metrics.MetricAccumulator() for arm in ARMS}
    times={x['target_id']:x['timestamps'] for x in binding['new_member_data_preflight']}
    def load(row):
        expected={};evidence={}
        for key,relative in native.target_files(row).items():
            native.checked_path(b.data_root,relative);expected['events' if key=='input' else key]=hashes[relative]['sha256']
        arrays,_=metrics.load_arrays(b.data_root/'saved_flow_data',row,evidence=evidence,expected_hashes=expected)
        return arrays,evidence,base.prepare_input(arrays['events'],device,torch)
    def forward(chunk,arm,**kwargs):
        out=forward_arm(r8,parent,model,functional,targets,shapes,chunk,arm,merge,torch,deadline=deadline,**kwargs)
        result['full_forward_count']+=1;require(result['full_forward_count']<=BUDGET,'Forward cap exceeded')
        if kwargs.get('mode','actual')=='actual':require(out[1]['summed_fc1_rows']==out[1]['summed_fc2_rows']==ROWS[arm],'Real service differs')
        return out
    with torch.inference_mode():
        arrays,evidence,chunk=load(anchor_row);archived=reference['frames'][0]
        require(evidence==archived['input_evidence'],'Archived input differs')
        for arm,expected in [('CC',reference['anchor_replays']['coarse_stage02']['prediction_sha256']),
                             ('fine_s2b2',archived['arms']['fine_s2b2']['prediction_sha256'])]:
            flows,ex,_=forward(chunk,arm);digest=hashlib.sha256(flows[-1][0].numpy().tobytes()).hexdigest()
            require(digest==expected,'Archive replay not exact')
            result['anchor_replays'][arm]={'target_id':anchor_row['target_id'],'prediction_sha256':digest,'exact_archived_prediction':True,'execution':ex}
        del arrays,chunk,flows
        for index,(group,row) in enumerate(chosen):
            deadline();tid=row['target_id'];arrays,evidence,chunk=load(row)
            masks=metrics.partitions(arrays['gt'],arrays['mask'],arrays['events'],regions)
            frame={'target_id':tid,'sequence':row['sequence'],'group':'fixed_ordinary_member_confirmation',
                   'input_evidence':evidence,'timestamps':times[tid],'arms':{}};predictions={};cc=None
            for arm in ARMS:
                flows,ex,branches=forward(chunk,arm,capture=index==0 and arm!='native')
                if arm=='CC':cc=ex
                if arm=='fine_s2b2':
                    prefix=['sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.'+str(j)+'.mlp' for j in range(2)]
                    prefix += [r8.PREFIX+str(j)+'.mlp' for j in range(3)]
                    require(all(ex['layers'][k]['input_sha256']==cc['layers'][k]['input_sha256'] for k in prefix),'Unchanged prefix differs')
                prediction=flows[-1][0].numpy().copy();predictions[arm]=prediction
                frame['arms'][arm]={'metrics':meters[arm].update(tid,row['sequence'],prediction,arrays['gt'],masks),
                                   'prediction_sha256':hashlib.sha256(prediction.tobytes()).hexdigest(),'execution':ex}
                if index==0 and arm!='native':
                    dual,dx,db=forward(chunk,arm,mode='dual_reference',capture=True)
                    require(len(flows)==len(dual) and all(torch.equal(x,y) for x,y in zip(flows,dual)),'Actual/dual flow differs')
                    require(branches.keys()==db.keys() and all(torch.equal(branches[k],db[k]) for k in branches),'Actual/dual branch differs')
                    result['dual_checks'][arm]={'flow_exact':True,'all8_branches_exact':True,'actual_execution':ex,'dual_execution':dx}
            path=a.output_dir/(tid+'.npz');np.savez_compressed(path,**predictions)
            frame.update(prediction_file=path.name,prediction_file_sha256=sha(path));result['frames'].append(frame)
            result['summaries']={arm:m.summary() for arm,m in meters.items()};parent.write(a.output_dir/'results.json',result)
            print(json.dumps({'target_id':tid,'completed_targets':len(result['frames']),'full_forwards':result['full_forward_count']}),flush=True)
            del chunk,flows,prediction
    require(parent.source_digest(model)==before and base.bn_signature(model,torch)==bn_before,'Model/BN changed')
    require(all(model.get_submodule(name) is m for name,m in targets.items()),'Modules not restored')
    require(result['full_forward_count']==BUDGET and len(result['frames'])==16,'Fixed count differs')
    result.update(status='ROUND9_FIXED_LAYER_CONFIRMATION_COMPLETE',original_model_unchanged=True,BN_buffers_unchanged=True,
                  original_modules_restored=True,wall_seconds=time.monotonic()-started,hardware_benefit_claim=False)


def main():
    r8=load_r8();collector=r8.load_collector();probe=collector.load_probe();parent=probe.load_scope_runner()
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-contract','identity-bank','prior-screen-results','prior-scope-results','selection-manifest','fit-membership','prior-check-membership','second-manifest','prior-round6-selection','membership','reference-r8'):
        parser.add_argument('--'+name,type=Path,required=True);parser.add_argument('--'+name+'-sha256',required=True)
    for name,default in [('module-checks',probe.BASE/'cpu_results.json'),('scope-runner-checks',probe.BASE/'scope_extension_v1/runner_cpu_results.json'),
                         ('probe-runner-checks',probe.HERE/'runner_cpu_results.json'),('collector-runner-checks',r8.OLD/'damage_predictor_round6/collector_cpu_results.json'),('runner-checks',HERE/'runner_cpu_results.json')]:
        parser.add_argument('--'+name,type=Path,default=default)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--preflight-output',type=Path)
    parser.add_argument('--wall-minutes',type=float,default=10);parser.add_argument('--execute',action='store_true')
    a=parser.parse_args();prepared=prepare(a,r8,collector,probe,parent)
    result={'schema':'flow_round9_fixed_layer_v1','status':'PREFLIGHT_PASS_NO_INFERENCE','binding':prepared[1],
            'inference_executed':False,'training_performed':False}
    if not a.execute:
        require(a.preflight_output and not a.preflight_output.exists(),'New preflight required')
        parent.write(a.preflight_output,result);print(result['status']);return
    with parent.GPU_LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);a.output_dir.mkdir(parents=True,exist_ok=False)
        previous=signal.getsignal(signal.SIGALRM)
        def timeout(signum,frame):raise TimeoutError('R9 wall budget exhausted')
        signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,a.wall_minutes*60)
        try:result['inference_executed']=True;execute(a,prepared,result,r8,parent)
        except BaseException as exc:
            result.update(status='INCOMPLETE_NO_SCIENTIFIC_CONCLUSION',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous);parent.write(a.output_dir/'results.json',result)


if __name__=='__main__':main()
