#!/usr/bin/env python3
"""R22 native representation collector: fixed fit16/dev8, 26 forwards, no training."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import sys
import time
import traceback
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROLE_SHA='e6f216f6b017fc2cbba262ec51c70474f342e593e1549058a2200970949e10a1'
NATIVE_HELPER_SHA='7e650140ebd50a1af50aed0f46189a3c218683ad8c1e814e7c7a131326532411'
GRAPH_ID='r22_W271482_all78current_native_fit16dev8'
TARGET='sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.2.mlp'


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def checked(spec):
    require(set(spec)=={'path','sha256'},'Explicit asset path/SHA')
    path=Path(spec['path']);require(path.is_file() and sha(path)==spec['sha256'],'Actual asset bytes '+str(path))
    return path


def plan(rows,role_map):
    require(len(rows)==len(role_map)==24 and {r['target_id'] for r in rows}==set(role_map),'Exact distinct allowed24')
    require([role_map[r['target_id']] for r in rows]==['fit']*16+['development']*8,'Fixed fit then development ordering')
    result=[]
    for i,row in enumerate(rows):
        result.append(dict(target_id=row['target_id'],selector_role=role_map[row['target_id']],arm='native',
                           execution='native',quality_endpoint=True,endpoint_arm='native'))
        if i==0:
            for repeat in (1,2):
                result.append(dict(target_id=row['target_id'],selector_role='fit',arm='native_repeat%d'%repeat,
                                   execution='native',quality_endpoint=False,endpoint_arm='native'))
    require(len(result)==26,'Exactly26 native forwards')
    return [dict(v,forward_id=i+1) for i,v in enumerate(result)]


def replay(actual,reference,torch):
    af,ac,_=actual;rf,rc,_=reference
    require(len(af)==len(rf)==4 and set(ac)==set(rc)=={'u','branch','sn1','sn2'},'Eight native raw arrays')
    aa=dict(ac,**{'head%d'%i:v for i,v in enumerate(af)})
    rr=dict(rc,**{'head%d'%i:v for i,v in enumerate(rf)})
    checks={}
    for key,a in aa.items():
        b=rr[key];require(a.shape==b.shape and a.dtype==b.dtype,'Replay shape/dtype '+key)
        av=a.contiguous().view(torch.uint8);bv=b.contiguous().view(torch.uint8)
        checks[key]={'bitwise_equal':bool(torch.equal(av,bv)),'different_bytes':int((av!=bv).sum()),
                     'shape':list(a.shape),'dtype':str(a.dtype),
                     'actual_sha256':hashlib.sha256(a.contiguous().numpy().tobytes()).hexdigest(),
                     'reference_sha256':hashlib.sha256(b.contiguous().numpy().tobytes()).hexdigest()}
    return {'pass':all(v['bitwise_equal'] for v in checks.values()),'array_checks':checks}


def selector_members(selector, canonical, hashes, native):
    """Only inspect frozen identity/acceptance metadata; never visit array paths."""
    require(selector['schema'] == 'flow_r20_next_selector_identity_roles_v1' and
            selector['status'] == 'FROZEN_METADATA_ONLY', 'Frozen selector metadata schema')
    groups = selector['groups']
    require(set(groups) == {'fit', 'development', 'sealed_confirmation'} and
            selector['group_counts'] == {'fit':16, 'development':8, 'sealed_confirmation':8}, 'Exact role population')
    require(all(len(groups[k]) == selector['group_counts'][k] for k in groups), 'Actual role counts')
    all_ids = [r['target_id'] for group in groups.values() for r in group]
    require(len(set(all_ids)) == len(all_ids) == 32, 'Roles must be pairwise disjoint')
    selected = groups['fit'] + groups['development']
    role_map = {r['target_id']:role for role in ('fit','development') for r in groups[role]}
    records = {r['target_id']:r for r in selector['records']}
    require(len(records) == len(selector['records']) == 32 and set(records) == set(all_ids), 'Complete unique role records')
    by_id = {r['target_id']:r for r in canonical}
    for row in selected:
        tid = row['target_id']; record = records[tid]
        require(row == by_id[tid] and row['protocol_split'] == 'train' and
                record['canonical'] == row and record['role'] == role_map[tid], 'Canonical selected identity/role')
        proof = {v['category']:v for v in record['accepted_array_metadata']}
        require(len(record['accepted_array_metadata']) == len(proof) == 3 and set(proof) == {'input','gt','mask'}, 'Three accepted array metadata entries')
        for kind, relative in native.target_files(row).items():
            item = proof[kind]
            require(item['relative_path'] == relative and item['target'] == tid and item['split'] == 'train' and
                    item['sequence'] == row['sequence'] and item['sha256'] == hashes[relative]['sha256'] and
                    int(item['bytes']) == hashes[relative]['bytes'], 'Selected acceptance identity/path/SHA/size')
    return selected, role_map



def verify_selected_files(selected, native, root, hashes, role_map):
    require(len(selected) == len(role_map) == 24 and set(role_map) == {r['target_id'] for r in selected}, 'Only allowed24 file scope')
    evidence = []
    for row in selected:
        entry = {}
        for kind, relative in native.target_files(row).items():
            path = native.checked_path(root, relative)
            require(path.is_file() and path.stat().st_size == hashes[relative]['bytes'] and
                    sha(path) == hashes[relative]['sha256'], 'Actual allowed24 input/GT/mask size/SHA')
            entry['events' if kind == 'input' else kind] = hashes[relative]['sha256']
        require(set(entry) == {'events','gt','mask'}, 'All three selected source files')
        evidence.append(dict(target_id=row['target_id'], selector_role=role_map[row['target_id']], input_evidence=entry))
    return evidence




def prepare(a):
    import torch
    require(not torch.cuda.is_initialized(), 'CPU preflight only')
    require(not a.output.exists() and 0 < a.wall_minutes <= 15, 'Fresh output, bounded inference')
    require(sha(a.inputs) == a.inputs_sha256 and sha(a.freeze) == a.freeze_sha256, 'Explicit manifest/freeze bytes')
    inp, freeze = read(a.inputs), read(a.freeze)
    require(inp['schema'] == 'flow_r22_representation_inputs_v1', 'Explicit R22 input schema')
    require(freeze['schema'] == 'flow_r22_representation_source_freeze_v1', 'Frozen source schema')
    for spec in freeze['files']:
        checked(spec)
    require(any(Path(s['path']).resolve() == Path(__file__).resolve() for s in freeze['files']), 'Actual entrypoint must be frozen')
    admission_path = checked(inp['independent_admission'])
    helper_path = checked(inp['native_forward_helper'])
    require(sha(helper_path) == NATIVE_HELPER_SHA, 'Frozen native source/forward helper')
    helper = load(helper_path, '_r22_frozen_native_helpers')
    require(inp['BN_mode'] == 'NATIVE_DENSE_CURRENT', 'Original all78 native current-BN contract')
    admission = read(admission_path)
    require(admission['schema'] == 'flow_r22_representation_independent_admission_v1' and admission['status'] == 'ADMIT'
            and a.plan in admission['permitted_plans'], 'Independent source admission')
    require(admission['runner_sha256'] == sha(__file__) and admission['native_helper_sha256'] == sha(helper_path)
            and admission['BN_mode'] == inp['BN_mode'], 'Admission binds actual native runner/helper')
    require(any(Path(s['path']).resolve() == helper_path.resolve() for s in freeze['files']), 'Actual native helper must be frozen')
    old = Path(inp['old_root'])
    conversion = old/'bn_conversion_round14'
    sys.path.insert(0, str(conversion))
    r14 = load(conversion/'run_conversion_v2.py', '_r22_original_source_helpers')
    k, h0, modules = r14.runtime()
    rtpath, memberpath = checked(inp['runtime']), checked(inp['membership'])
    rolepath, regionpath, cp = checked(inp['selector_roles']), checked(inp['regions']), checked(inp['checkpoint'])
    require(sha(rolepath) == ROLE_SHA, 'Frozen identity-only fit/development/confirmation roles')
    rt, members, selector, regions = map(read, (rtpath, memberpath, rolepath, regionpath))
    require(sha(memberpath) == k.MEMBER_SHA and sha(regionpath) == k.REGION_SHA, 'Fixed membership/regions')
    require(sha(cp) == k.CHECKPOINT_SHA, 'Only original W271482')
    require(sha(cp.with_suffix('.json')) == k.SIDECAR_SHA and sha(cp.parent/'run_contract.json') == k.RUN_CONTRACT_SHA, 'Checkpoint sidecar/run source')
    for path, digest in rt['source_pins'].items():
        require(sha(path) == digest, 'Frozen native dependency '+path)
    sys.path[:0] = [rt['breadth_root'], rt['motion_root']]
    base = load(Path(rt['breadth_root'])/'evaluate_breadth.py', '_r22_native_preflight')
    args = list(rt['base_args'])
    require(not any(x.startswith('--bn-') or x.startswith('--svs-') or x.startswith('--regions') or x == '--run-inference' for x in args), 'No implicit legacy experiment')
    for flag, value in (('--checkpoint', str(cp)), ('--checkpoint-sha256', k.CHECKPOINT_SHA), ('--output', str(a.output/'results.json'))):
        require(args.count(flag) == 1, 'Single native argument '+flag)
        args[args.index(flag)+1] = value
    b = base.parse_args(args)
    b.device = 'cuda:0'
    context = list(helper.source_preflight(b, base))
    require(context[9].pop('allowed8_current_files_checked_separately') is True,
            'Frozen helper did source metadata only; own24 check follows')
    context[9]['allowed24_current_files_checked_separately'] = True
    native, trainer, metrics, _, hashes, config, run, side, _, _ = context
    canonical = metrics.canonical_rows(b.canonical)
    roles = metrics.auxiliary_roles(b.roles, canonical)
    modules[0].validate_members(members, canonical, roles)
    r14.validate_manifest_sources(members, b.canonical, b.roles)
    metrics.validate_region_contract(regions, canonical, roles, require_complete_fit=False)
    require(regions['fit_target_ids'] == [r['target_id'] for r in members['train_members_diversity']], 'Fixed training region fit')
    context[8] = regions
    selected, role_map = selector_members(selector, canonical, hashes, native)
    context[3] = selected
    cp_audit = r14.validate_checkpoint(cp, '271482', tuple(context), h0, torch)
    evidence = verify_selected_files(selected, native, b.data_root, hashes, role_map)
    context[9]['allowed24_current_files_SHA_checked'] = 72
    binding = {'graph_id': GRAPH_ID, 'plan': a.plan, 'forward_plan': plan(selected, role_map), 'checkpoint_cpu_audit': cp_audit,
               'selector_roles_sha256': sha(rolepath), 'selected_target_ids': [r['target_id'] for r in selected], 'selector_role_map': role_map, 'confirmation_array_accesses': 0, 'membership_sha256': sha(memberpath), 'region_sha256': sha(regionpath),
               'inputs_sha256': a.inputs_sha256, 'source_freeze_sha256': a.freeze_sha256,
               'independent_admission_sha256': sha(admission_path), 'other_11_MLPs': 'native',
               'hardware_mapping_status': 'Native-only representation collection; no dynamic action or hardware benefit claim',
               'compact_forwards': 0, 'native_forwards': 26, 'quality_endpoints': 24, 'determinism_replays': 2, 'target': TARGET,
               'BN': 'all78_current', 'quality_interpretation': a.plan == 'native26', 'optimizer_steps': 0,
               'native_helper_sha256': sha(helper_path), 'BN_mode': inp['BN_mode'],
               'primitive_calls': 0, 'expected_FC_rows_each': 14508000, 'expected_BN_calls': 2028,
               'data_preflight': context[9], 'calibration_forwards': 0, 'precision': 'FP32; no AMP/TF32; T10/B1/full9', 'wall_minutes': a.wall_minutes}
    return dict(binding=binding, data_evidence=evidence, selected=selected, role_map=role_map, context=tuple(context), base=base, b=b,
                rt=rt, r14=r14, k=k, h0=h0, modules=modules, helper=helper)


def execute(a, p, result):
    import numpy as np
    import torch
    require(not torch.cuda.is_initialized(), 'CUDA begins only inside global lock')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261017)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.manual_seed_all(20261017)
    begin=time.monotonic()
    rt,base,b,context=p['rt'],p['base'],p['b'],p['context']
    native,trainer,metrics,_,hashes,config,run,side,regions,_=context
    r13,collector,probe,parent,r9,r8=p['modules']
    loader=load(Path(rt['breadth_root'])/'run_same_shape_reference.py','_r22_native_loader')
    scope=load(Path(rt['breadth_root'])/'run_bn_scope.py','_r22_native_inventory')
    model,functional=loader.load_model(b,context,base,torch,device)
    require(r13.state_digest(model)==p['k'].SOURCE_STATE_SHA,'Actual original model state')
    bn=p['r14'].core.switch_allcurrent(model,base,scope,r13.state_digest,torch,p['k'].SOURCE_NON_BN_SHA)
    targets,shapes=parent.target_layers(model)
    require(tuple(shapes[TARGET])==(10,1,30,40,384),'Original target feature geometry')
    before,signature=r13.state_digest(model),base.bn_signature(model,torch)
    result.update(status='RUNNING',inference_executed=True,
        runtime={'torch':str(torch.__version__),'numpy':np.__version__,'device':torch.cuda.get_device_name(device),
                 'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'TF32':False,'AMP':False},
        BN_scope=bn,initial_allcurrent_state_sha256=before,forward_records=[],numerical_checks=[])
    rows={r['target_id']:r for r in p['selected']};last_id=None;first_output=None
    for item in p['binding']['forward_plan']:
        require(time.monotonic()-begin<a.wall_minutes*60,'Native26 wall bound')
        tid=item['target_id']
        if tid!=last_id:
            row=rows[tid];files=native.target_files(row)
            evidence={'events' if k=='input' else k:hashes[v]['sha256'] for k,v in files.items()}
            eventpath=native.checked_path(b.data_root,files['input'])
            require(sha(eventpath)==evidence['events'],'Actual selected event bytes')
            events=np.load(eventpath,allow_pickle=False)
            require(events.shape==(10,480,640) and events.dtype==np.float32 and np.isfinite(events).all(),'Finite original event tensor')
            chunk=base.prepare_input(events,device,torch);del events;last_id=tid
        require(item['execution']=='native','No compact/expanded entry is possible')
        result['full_forward_attempts']+=1;write(a.output/'results.json',result)
        output=p['helper'].forward(model,functional,targets,shapes,chunk,item,None,'NATIVE_DENSE_CURRENT',torch)
        flows,captures,record=output
        require(record['target_execution']=={} and record['summed_fc1_rows']==record['summed_fc2_rows']==558000,'Pure native full compute')
        require(set(captures)=={'u','branch','sn1','sn2'} and tuple(captures['u'].shape)==(10,1,30,40,384),'Full native T/B/H/W/C capture')
        record['input_evidence']=evidence
        name='%03d_%s_%s_native.npz'%(item['forward_id'],tid,item['arm'])
        arrays={'head%d'%i:v.numpy() for i,v in enumerate(flows)}
        arrays.update({k:v.numpy() for k,v in captures.items()})
        require(len(arrays)==8,'Complete eight native raw arrays')
        np.savez_compressed(a.output/name,**arrays)
        record.update(prediction_file=name,prediction_file_sha256=sha(a.output/name),saved_payload_keys=sorted(arrays))
        result['forward_records'].append(record);result['full_forward_count']+=1
        check=None
        if item['forward_id']==1:
            require(item['arm']=='native','First real native reference');first_output=output
        elif item['arm'].startswith('native_repeat'):
            require(first_output is not None and tid==first_output[2]['target_id'] and first_output[2]['forward_id']==1,'Own first identity replay')
            check=dict(kind='native_exact_replay',target_id=tid,arm=item['arm'],forward_id=item['forward_id'],
                       reference_forward_id=1,**replay(output,first_output,torch))
            result['numerical_checks'].append(check)
        p['k'].core.check_closure(model,targets,before,r13.state_digest,signature,base.bn_signature,torch)
        record['state_and_BN_unchanged']=True;record['original_modules_restored_and_hooks_removed']=True
        write(a.output/'results.json',result)
        print(json.dumps({'forward':item['forward_id'],'target_id':tid,'selector_role':item['selector_role'],
                          'arm':item['arm'],'numeric_pass':check['pass'] if check else None}),flush=True)
        require(check is None or check['pass'],'Native replay failed; no quality/training interpretation')
    require(result['full_forward_count']==result['full_forward_attempts']==26 and len(result['numerical_checks'])==2,'Complete26 and two replay gates')
    require(sum(x['quality_endpoint'] for x in result['forward_records'])==24,'Exactly24 quality endpoints')
    require(all(sum(r['summed_'+fc+'_rows'] for r in result['forward_records'])==14508000 for fc in ('fc1','fc2')),'Native FC closure')
    result.update(status='REPRESENTATION_NATIVE26_COMPLETE',final_state_sha256=r13.state_digest(model),
                  all_states_unchanged=True,all_hooks_removed=True,original_modules_restored=True,
                  wall_seconds=time.monotonic()-begin,max_allocated_bytes=torch.cuda.max_memory_allocated(),
                  max_reserved_bytes=torch.cuda.max_memory_reserved())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--inputs-sha256',required=True)
    parser.add_argument('--freeze',type=Path,required=True);parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--plan',choices=('native26',),required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--wall-minutes',type=float,default=15)
    parser.add_argument('--preflight-output',type=Path);parser.add_argument('--execute',action='store_true')
    a=parser.parse_args();p=prepare(a)
    result=dict(schema='flow_r22_representation_v1',graph_id=GRAPH_ID,status='PREFLIGHT_PASS_NO_INFERENCE',binding=p['binding'],
                data_evidence=p['data_evidence'],inference_executed=False,full_forward_count=0,full_forward_attempts=0,
                training_performed=False,optimizer_steps=0,calibration_forwards=0,selector_training_performed=False,
                flow_specific_increment_claim=False,hardware_benefit_claim=False)
    if not a.execute:
        require(a.preflight_output and not a.preflight_output.exists(),'Fresh metadata preflight receipt')
        write(a.preflight_output,result);print(result['status']);return
    lock=p['modules'][3].GPU_LOCK.open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        a.output.mkdir(parents=True,exist_ok=False);write(a.output/'results.json',result)
        def alarm(signum,frame):raise TimeoutError('R22 hard wall cap')
        signal.signal(signal.SIGALRM,alarm);signal.alarm(int(a.wall_minutes*60))
        try:execute(a,p,result)
        except BaseException as exc:
            result.update(status='INCOMPLETE_OR_NUMERICAL_FAILURE',error=str(exc),traceback=traceback.format_exc())
            write(a.output/'results.json',result);raise
        finally:signal.alarm(0)
        write(a.output/'results.json',result)
    finally:lock.close()


if __name__=='__main__':main()


