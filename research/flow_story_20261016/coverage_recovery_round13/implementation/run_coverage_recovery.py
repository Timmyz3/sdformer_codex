#!/usr/bin/env python3
"""R13 sequence-held recovery coverage test; preflight by default, new BN only."""
import argparse
import collections
import csv
import contextlib
import fcntl
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import types
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
R12 = HERE.parents[1]/'ordinary_recovery_round12'
CHECKPOINT_SHA = 'd5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30'
SCHEMA = 'flow_round13_coverage_recovery_v1'
BN_SCOPE = 'MLP_R13_COMMON_CMA176'
OWN = ('run_coverage_recovery.py','execute_recovery.py','test_coverage_cpu.py','inherited_pins.json')


def require(ok, message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def load_file(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def load_inherited():
    pins=json.loads((HERE/'inherited_pins.json').read_text())['r12_sources']
    for name,digest in pins.items():require(sha(R12/name)==digest,'Frozen R12 source differs: '+name)
    sys.path.insert(0,str(R12))
    old=load_file(R12/'run_recovery.py','_r13_frozen_r12_helpers')
    require(Path(old.tile.__file__).resolve()==R12/'trainable_representative_mlp.py','Wrong tile import')
    return old


OLD=load_inherited()
tile=OLD.tile
ARMS=OLD.ARMS
LOSS_SHA=OLD.LOSS_SHA
PARAMETERS=OLD.PARAMETERS
RECIPE=dict(OLD.RECIPE)
RECIPE.update(fit_targets={'reference':40,'diversity':200},fit_order='manifest fixed200; per-step sequence matched; reference40 each5, diversity200 each1',
              BN='new common training-sequence CMA176 private24 fixed; external54 current-stat',
              check_used_for_training_or_checkpoint_selection=False)
for _name in ('state_digest','select_parameters','expected_rows','forward_train','frozen_parameters',
              'reference_forward','compare_all','gradient_audit','gradient_coverage'):
    globals()[_name]=getattr(OLD,_name)


def load_runtime():
    # No OLD.prepare/load_runtime: their legacy membership and CMA payload paths are not used.
    collector=OLD.load_source('collector','_r13_collector_helpers')
    probe=collector.load_probe();parent=probe.load_scope_runner()
    r9=OLD.load_source('r9_runner','_r13_anchor_adapter');r8=r9.load_r8()
    require(sha(r8.__file__)==sha(OLD.source_path('r8_runner')),'R8 helper differs')
    return collector,probe,parent,r9,r8


def validate_members(manifest,canonical,roles):
    require(manifest.get('schema')=='flow_round13_coverage_identity_v1','Unknown R13 member schema')
    require(manifest.get('checkpoint_sha256')==CHECKPOINT_SHA,'Wrong membership checkpoint')
    by_id={row['target_id']:row for row in canonical}
    require(len(by_id)==len(canonical),'Duplicate canonical')
    names=('train_members_reference','train_members_diversity','check_members','bn_members','selector_members')
    ids={};seqs={}
    for name in names:
        rows=manifest[name];require(isinstance(rows,list) and rows,'Nonempty member lists required')
        ids[name]=[r['target_id'] for r in rows]
        require(len(ids[name])==len(set(ids[name])),'Duplicate members: '+name)
        for row in rows:
            require(row['target_id'] in by_id and row==by_id[row['target_id']],'Canonical member row differs')
            role='bn_calibration' if name=='bn_members' else 'gate_fit'
            require(row['protocol_split']=='train' and roles.get(row['target_id'])==role,'Member role/split differs: '+name)
        seqs[name]={r['sequence'] for r in rows}
    train=set(manifest['split_sequences']['train']);check=set(manifest['split_sequences']['check'])
    require(len(train)==8 and len(check)==2 and not train&check,'8/2 disjoint sequence split required')
    require(seqs['check_members']==check,'Check sequences differ')
    require(all(seqs[n]==train for n in names if n!='check_members'),'Training-side sequence coverage differs')
    require(len(ids['train_members_reference'])==40 and len(ids['train_members_diversity'])==200,'40/200 coverage required')
    require(set(ids['train_members_reference'])<=set(ids['train_members_diversity']),'Reference not nested')
    require(len(ids['bn_members'])==176,'Exactly176 new BN members required')
    others=['train_members_diversity','check_members','bn_members','selector_members']
    for i,name in enumerate(others):
        for other in others[i+1:]:require(not set(ids[name])&set(ids[other]),'Identity roles overlap')
    for condition,n,repeat in (('reference',40,5),('diversity',200,1)):
        ordered=manifest['train_order_'+condition]
        require(len(ordered)==200 and [r['step'] for r in ordered]==list(range(1,201)),'Exactly fixed200 ordered steps')
        ordered_ids=[];visits=collections.Counter()
        for row in ordered:
            pure={k:v for k,v in row.items() if k not in ('step','sequence_visit_index')}
            require(pure==by_id.get(row['target_id']),'Training order canonical row differs')
            visits[row['sequence']]+=1
            require(row['sequence_visit_index']==visits[row['sequence']],'Sequence visit index differs')
            ordered_ids.append(row['target_id'])
        require(collections.Counter(ordered_ids)==collections.Counter({x:repeat for x in ids['train_members_'+condition]}),'Training repetition differs')
        require(len(set(ordered_ids))==n and set(visits)==train and set(visits.values())=={25},'Equal sequence visitation differs')
    require([r['sequence'] for r in manifest['train_order_reference']]==[r['sequence'] for r in manifest['train_order_diversity']],
            'Per-step sequence visitation differs between conditions')
    require(collections.Counter(r['sequence'] for r in manifest['bn_members'])==collections.Counter({s:22 for s in train}),'BN must be22 per training sequence')
    return {'train_sequences':sorted(train),'check_sequences':sorted(check),'member_counts':{n:len(v) for n,v in ids.items()},
            'recovery_and_BN_sequence_held':True,'backbone_unseen_claim':False,'per_step_sequence_matched':True}


def validate_bn_receipt(receipt,buffers,manifest,membership_sha):
    require(receipt.get('schema')=='flow_round13_common_bn_v1' and receipt.get('status')=='COMMON_BN_COMPLETE','Completed new BN receipt required')
    require(receipt.get('checkpoint_sha256')==CHECKPOINT_SHA and receipt.get('membership_sha256')==membership_sha,'New BN source differs')
    require(receipt.get('bn_target_ids')==[r['target_id'] for r in manifest['bn_members']],'New BN identities/order differs')
    require(receipt.get('buffers_sha256')==sha(buffers),'New BN array SHA differs')
    require(receipt.get('calibration_forwards')==176 and receipt.get('calibration_bn_count')==78 and receipt.get('retained_mlp_bn_count')==24,'New BN calibration count differs')
    require(receipt.get('BN_scope')==BN_SCOPE and receipt.get('calibration_definition')=='all78_reset_CMA_train_nonBN_eval','New BN semantics differs')
    require(receipt.get('old_CMA_loaded') is False and receipt.get('GT_used') is False,'Legacy BN/GT calibration forbidden')


def apply_fresh_bn(model,buffers_path,receipt_path,manifest,membership_sha,scope,base,torch,
                   expected_mlp_blocks=12,expected_total_bn=78,validate_receipt=True):
    import numpy as np
    if validate_receipt:validate_bn_receipt(json.loads(Path(receipt_path).read_text()),buffers_path,manifest,membership_sha)
    all_bn,mlp_bn=scope.bn_inventory(model,torch,expected_mlp_blocks,expected_total_bn)
    non_bn_before=scope.extended.non_bn_digest(model,all_bn);model.eval()
    expected={n+'.'+key for n in mlp_bn for key in ('running_mean','running_var','num_batches_tracked')}
    with np.load(buffers_path,allow_pickle=False) as values:
        require(set(values.files)==expected,'New BN keyset differs')
        validated={}
        for name in sorted(mlp_bn):
            layer=all_bn[name]
            require(layer.track_running_stats,'Original checkpoint BN must still track before fresh installation')
            for key in ('running_mean','running_var','num_batches_tracked'):
                value=values[name+'.'+key];ref=getattr(layer,key)
                require(value.shape==tuple(ref.shape),'New BN buffer shape differs')
                require(value.dtype==np.dtype('int64' if key=='num_batches_tracked' else 'float32'),'New BN buffer dtype differs')
                require(np.isfinite(value).all(),'Nonfinite new BN buffer')
                require((value>=0).all() if key=='running_var' else int(value)==176 if key=='num_batches_tracked' else True,'Invalid new BN variance/count')
                validated[(name,key)]=value.copy()
        # Entire payload is validated before any mutation.
        with torch.no_grad():
            for (name,key),value in validated.items():getattr(all_bn[name],key).copy_(torch.from_numpy(value).to(getattr(all_bn[name],key)))
    for name,layer in all_bn.items():
        if name not in mlp_bn:layer.track_running_stats=False;layer.running_mean=layer.running_var=layer.num_batches_tracked=None
    require(scope.extended.non_bn_digest(model,all_bn)==non_bn_before,'New BN changed non-buffer state')
    require(all(not m.training for m in model.modules()),'New BN left training modules')
    return {'scope':BN_SCOPE,'BN_count':len(all_bn),'MLP_BN_count':len(mlp_bn),'frozen_BN_names':sorted(mlp_bn),
            'current_stat_BN_names':sorted(set(all_bn)-mlp_bn),'MLP_BN_names':sorted(mlp_bn),'non_bn_state_unchanged':True,
            'non_bn_state_sha256':non_bn_before,'new_buffers_sha256':sha(buffers_path),'new_receipt_sha256':sha(receipt_path),
            'old_CMA_loaded':False,'BN_signature_before_inference':base.bn_signature(model,torch)}


def validate_smokes(paths,binding):
    require(len(paths)==4,'Four new-condition smoke receipts required')
    seen=set();out={}
    fields=('runner_sha256','execute_sha256','wrapper_sha256','runner_checks_sha256','source_pins_sha256','checkpoint_sha256',
            'loss_sha256','membership_sha256','runtime_manifest_sha256','BN_buffers_sha256','BN_receipt_sha256','allowed_parameter_count','condition')
    for path in paths:
        value=json.loads(path.read_text());arm=value.get('arm')
        require(arm in ARMS and arm not in seen,'Duplicate/unknown smoke arm');seen.add(arm)
        require(value.get('schema')==SCHEMA and value.get('status')=='ROUND13_SMOKE_COMPLETE' and
                value.get('completed_optimizer_steps')==5 and value.get('training_performed') is True,'Incomplete R13 smoke')
        require(all(value['binding'][k]==binding[k] for k in fields) and value['binding']['recipe']==RECIPE and
                value['binding']['arm']==arm and value['binding']['mode']=='smoke','R13 smoke binding differs')
        require(all(value[k] for k in ('frozen_state_unchanged','BN_buffers_unchanged','original_modules_restored','all_hooks_removed','smoke_weights_discarded')),'Smoke closure missing')
        require(value['initial_numerical_pair']['all_flow_outputs_exact'] and value['final_numerical_pair']['all_flow_outputs_exact'] and value['gradient_coverage']['complete'],'Smoke numeric/gradient audit missing')
        out[arm]=sha(path)
    return out


def prepare(a,collector,probe,parent):
    require(0<a.wall_minutes<=30 and not a.output_dir.exists(),'Fresh output and at most30min required')
    for name in ('runtime_manifest','membership','bn_buffers','bn_receipt'):
        require(sha(getattr(a,name))==getattr(a,name+'_sha256'),'Explicit input SHA differs: '+name)
    runtime=json.loads(a.runtime_manifest.read_text());manifest=json.loads(a.membership.read_text())
    require(runtime.get('schema')=='flow_round13_runtime_v1','Unknown runtime manifest')
    for path,digest in runtime['source_pins'].items():require(sha(path)==digest,'Runtime source differs: '+path)
    sys.path[:0]=[runtime['breadth_root'],runtime['motion_root']]
    import evaluate_breadth as base
    import run_bn_scope as scope
    import run_same_shape_reference as loader
    expected_sources={'evaluate_breadth.py':'1cacbaeea4d9eaf19a304c3faaa13f5278f0212b6cdc92210c4619447d571628',
                      'run_bn_scope.py':'2b5c61e3abedcc02ba09adadb432dfc0ff2cd2fcd2b7b3e3a2475c63d7a11840',
                      'run_same_shape_reference.py':'6b6c2f93d24bac9fa1dd3a99fde0db05aa180d05d897482d13a028208340774b'}
    for module in (base,scope,loader):require(sha(module.__file__)==expected_sources[Path(module.__file__).name],'Pinned native helper differs')
    args=list(runtime['base_args'])
    require(not any(x.startswith('--bn-') or x.startswith('--svs-') or x.startswith('--regions') or x=='--run-inference' for x in args),
            'Legacy BN/SVS/region/inference arguments forbidden')
    require(args.count('--output')==1,'One base output required');args[args.index('--output')+1]=str(a.output_dir/'results.json')
    b=base.parse_args(args);b.device='cuda:0';b.wall_minutes=a.wall_minutes
    require(b.geometry=='full9' and b.checkpoint_sha256==CHECKPOINT_SHA and not b.run_inference,'Pinned native full9 source required')
    context=list(base.preflight(b))
    native,trainer,metrics,dev32,hashes,config,saved,side,_,plan=context
    require(side['completed_optimizer_steps']==135930,'Wrong original checkpoint step')
    canonical=metrics.canonical_rows(b.canonical);roles=metrics.auxiliary_roles(b.roles,canonical)
    require(manifest['source_sha256']['canonical']==sha(b.canonical) and manifest['source_sha256']['roles']==sha(b.roles),'Member source manifests differ')
    membership_audit=validate_members(manifest,canonical,roles)
    regions_path=Path(runtime['region_path']);require(sha(regions_path)==runtime['region_sha256'],'New training-only region SHA differs')
    regions=json.loads(regions_path.read_text());metrics.validate_region_contract(regions,canonical,roles,require_complete_fit=False)
    require(regions['fit_target_ids']==[r['target_id'] for r in manifest['train_members_diversity']],'Region fit must use frozen diversity200 order')
    evidence=regions.get('data_evidence',[]);require([r['target_id'] for r in evidence]==regions['fit_target_ids'],'Region evidence coverage differs')
    by_id={r['target_id']:r for r in canonical}
    for item in evidence:
        paths=native.target_files(by_id[item['target_id']])
        require(all(item.get(k+'_sha256')==hashes[paths[k]]['sha256'] for k in ('gt','mask')),'Region GT source differs')
    context[8]=regions
    validate_bn_receipt(json.loads(a.bn_receipt.read_text()),a.bn_buffers,manifest,a.membership_sha256)
    ordered=manifest['train_order_'+a.condition]
    fit=[by_id[r['target_id']] for r in ordered]
    checks=[] if a.mode=='smoke' else manifest['check_members']
    chosen=[('fit',r) for r in fit]+[('check',r) for r in checks]
    timing=collector.load_pinned(Path(runtime['motion_root'])/'collect_coarse_cache.py',collector.TIMESTAMP_SHA,'_r13_timestamp_binding')
    acceptance=json.loads(b.acceptance.read_text());inventory_path=b.acceptance.parent/'native_deployment_manifest.csv'
    require(sha(inventory_path)==acceptance['deployment_csv_sha256'],'Deployment inventory differs')
    with inventory_path.open(newline='') as stream:inventory={row['relative_path']:row for row in csv.DictReader(stream)}
    data_evidence=[];seen=set()
    for group,row in chosen:
        if row['target_id'] in seen:continue
        seen.add(row['target_id']);files={}
        for key,relative in native.target_files(row).items():
            require(relative in hashes and sha(native.checked_path(b.data_root,relative))==hashes[relative]['sha256'],'Accepted input source differs')
            files[key]=hashes[relative]['sha256']
        data_evidence.append({'target_id':row['target_id'],'group':group,'files':files,
            'timestamps':timing.timestamp_binding(b.data_root,row,native.target_files(row),hashes,inventory,acceptance)})
    source=json.loads(b.source_manifest.read_text());require(source['loss/flow_supervised.py']==LOSS_SHA and sha(b.source_root/'loss/flow_supervised.py')==LOSS_SHA,'Loss source differs')
    receipt=parent.checked_cpu_receipt(a.runner_checks,[HERE/n for n in OWN])
    binding={'runner_sha256':sha(__file__),'execute_sha256':sha(HERE/'execute_recovery.py'),'wrapper_sha256':sha(tile.__file__),
             'runner_checks_sha256':receipt,'source_pins_sha256':sha(HERE/'inherited_pins.json'),'checkpoint_sha256':CHECKPOINT_SHA,
             'membership_sha256':a.membership_sha256,'runtime_manifest_sha256':a.runtime_manifest_sha256,
             'BN_buffers_sha256':a.bn_buffers_sha256,'BN_receipt_sha256':a.bn_receipt_sha256,'recipe':RECIPE,'loss_sha256':LOSS_SHA,
             'allowed_parameter_count':PARAMETERS,'arm':a.arm,'mode':a.mode,'condition':a.condition,'fit_ids':[r['target_id'] for r in fit],
             'check_ids':[r['target_id'] for r in checks],'membership_audit':membership_audit,'data_evidence':data_evidence,
             'region_sha256':runtime['region_sha256'],'GPU_lock':str(parent.GPU_LOCK),'wall_minutes':a.wall_minutes,
             'expected_training_forwards':5 if a.mode=='smoke' else 200,'expected_audit_forwards':4 if a.arm=='native' else 6,
             'expected_evaluation_forwards':0 if a.mode=='smoke' else 5*len(checks),
             'scope':'Recovery/BN sequence-held train-side coverage comparison; original backbone has seen all training sequences. No flow-specific increment or hardware claim.'}
    if a.mode=='recover':binding['smoke_receipts_sha256']=validate_smokes(a.smoke_receipt,binding)
    import torch
    require(not torch.cuda.is_initialized(),'CUDA before shared lock')
    inherited=(runtime,b,tuple(context),base,scope,loader,chosen,binding,None)
    return inherited,fit,checks,binding,manifest


def execute(a,prepared,result,collector,probe,parent,r9,r8):
    module=load_file(HERE/'execute_recovery.py','_r13_adapted_training_loop')
    namespace=dict(module.execute.__globals__)
    for key in ('require','RECIPE','BN_SCOPE','ARMS','LOSS_SHA','tile','apply_fresh_bn','state_digest','select_parameters',
                'forward_train','reference_forward','compare_all','gradient_audit','gradient_coverage','sha'):
        namespace[key]=globals()[key]
    function=types.FunctionType(module.execute.__code__,namespace,module.execute.__name__,module.execute.__defaults__)
    return function(a,prepared,result,collector,probe,parent,r9,r8)


def main(argv=None):
    collector,probe,parent,r9,r8=load_runtime()
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime-manifest','membership','bn-buffers','bn-receipt'):
        parser.add_argument('--'+name,type=Path,required=True);parser.add_argument('--'+name+'-sha256',required=True)
    parser.add_argument('--condition',choices=('reference','diversity'),required=True)
    parser.add_argument('--mode',choices=('smoke','recover'),required=True);parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--smoke-receipt',type=Path,action='append',default=[])
    parser.add_argument('--runner-checks',type=Path,default=HERE/'runner_cpu_results.json')
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--preflight-output',type=Path)
    parser.add_argument('--wall-minutes',type=float,default=30);parser.add_argument('--execute',action='store_true')
    a=parser.parse_args(argv);prepared=prepare(a,collector,probe,parent)
    result={'schema':SCHEMA,'status':'PREFLIGHT_PASS_NO_INFERENCE','binding':prepared[3],'training_performed':False,'inference_executed':False}
    if not a.execute:
        require(a.preflight_output and not a.preflight_output.exists(),'Fresh preflight output required')
        parent.write(a.preflight_output,result);print(result['status']);return
    require(parent.GPU_LOCK.parent.is_dir(),'Existing shared lock directory required')
    with parent.GPU_LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);a.output_dir.mkdir(parents=True,exist_ok=False)
        previous=signal.getsignal(signal.SIGALRM)
        def timeout(signum,frame):raise TimeoutError('R13 fixed30minute arm cap exhausted')
        signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,a.wall_minutes*60)
        try:
            result['inference_executed']=True;execute(a,prepared,result,collector,probe,parent,r9,r8)
        except BaseException as exc:
            result.update(status='INCOMPLETE_NO_SCIENTIFIC_CONCLUSION',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous);parent.write(a.output_dir/'results.json',result)


if __name__=='__main__':main()
