#!/usr/bin/env python3
"""R13 train-only region fit or common native-CMA BN preparation. No optimizer."""
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
sys.dont_write_bytecode = True


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write(p, data):
    Path(p).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def prepare(a):
    require(sha(a.runtime) == a.runtime_sha256, 'Runtime changed')
    require(sha(a.membership) == a.membership_sha256, 'Membership changed')
    runtime = json.loads(a.runtime.read_text())
    members = json.loads(a.membership.read_text())
    require(runtime['schema'] == 'flow_round13_preparation_runtime_v1', 'Runtime schema')
    require(members['schema'] == 'flow_round13_coverage_identity_v1', 'Membership schema')
    forbidden = {'--regions','--regions-sha256','--bn-artifact','--bn-artifact-sha256','--bn-schedule'}
    require(not forbidden.intersection(runtime['base_args']), 'Old region/BN forbidden')
    for p, expected in runtime['source_pins'].items():
        require(sha(p) == expected, 'Inherited source changed: '+p)
    root = Path(runtime['breadth_root'])
    sys.path.insert(0, str(root))
    base = module('evaluate_breadth', root/'evaluate_breadth.py')
    b = base.parse_args(runtime['base_args'])
    context = base.preflight(b)
    native, trainer, metrics, _, hashes, config, saved, side, regions, plan = context
    canonical = metrics.canonical_rows(b.canonical)
    roles = metrics.auxiliary_roles(b.roles, canonical)
    lookup = {r['target_id']:r for r in canonical}
    train = set(members['split_sequences']['train'])
    check = set(members['split_sequences']['check'])
    require(len(train)==8 and len(check)==2 and not train & check, 'Sequence isolation')
    require(b.checkpoint_sha256 == 'd5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30', 'Checkpoint changed')
    for key, count, role in [('bn_members',176,'bn_calibration'),('train_members_diversity',200,'gate_fit')]:
        rows = members[key]
        require(len(rows)==len({r['target_id'] for r in rows})==count, 'Fixed preparation count')
        for row in rows:
            require(row==lookup[row['target_id']] and row['sequence'] in train and
                    row['protocol_split']=='train' and roles[row['target_id']]==role, 'Role/canonical mismatch')
    return runtime, members, base, b, context, roles


def fit_regions(a, prepared):
    import numpy as np
    runtime, members, base, b, context, roles = prepared
    native, trainer, metrics, _, hashes, config, saved, side, _, plan = context
    rows = members['train_members_diversity']
    values = {'boundary':[], 'large':[]}
    evidence = []
    started = time.monotonic()
    for row in rows:
        expected = {key:hashes[relative]['sha256'] for key,relative in native.target_files(row).items() if key!='input'}
        observed = {}
        arrays, _ = metrics.load_arrays(b.data_root/'saved_flow_data', row, include_events=False,
                                       evidence=observed, expected_hashes=expected)
        mag,jump,eligible = metrics.region_features(arrays['gt'], arrays['mask'])
        for key, candidate in [('boundary',jump[eligible]),('large',mag[arrays['mask']])]:
            seed = int.from_bytes(hashlib.sha256(('flow-a-regions-v1:'+key+':'+row['target_id']).encode()).digest()[:8], 'little')
            if len(candidate)>4096:
                indices = np.random.Generator(np.random.PCG64(seed)).choice(len(candidate),4096,replace=False)
                candidate = candidate[np.sort(indices)]
            values[key].append(candidate)
        evidence.append({'target_id':row['target_id'],'valid_pixels':int(arrays['mask'].sum()),
                         'boundary_eligible_pixels':int(eligible.sum()),'gt_sha256':observed['gt'], 'mask_sha256':observed['mask']})
    pooled = {key:np.concatenate(parts) for key,parts in values.items()}
    require(all(len(x) for x in pooled.values()), 'No region samples')
    ids = [r['target_id'] for r in rows]
    result = {'schema':metrics.REGION_SCHEMA, 'definition':metrics.REGION_DEFINITION,
        'source':'R13 training-only diagnostic flow-jump proxy; not semantic boundaries or final-valid protocol',
        'canonical_manifest_sha256':metrics.CANONICAL_SHA,'auxiliary_roles_sha256':metrics.AUX_ROLES_SHA,
        'fit_target_ids':ids,'fit_target_ids_sha256':metrics.ids_sha(ids),
        'fit_scope':'explicit_gate_fit_subset_exploration_only','fit_uses_final_valid':False,
        'boundary_quantile':.95,'large_quantile':.90,
        'boundary_threshold_pixels':float(np.quantile(pooled['boundary'],.95,method='linear')),
        'large_displacement_threshold_pixels':float(np.quantile(pooled['large'],.90,method='linear')),
        'comparison':'strict_greater_than',
        'sampling':{'per_target_feature_cap':4096,'seed_prefix':'flow-a-regions-v1:',
                    'generator':'NumPy PCG64 SHA-derived64bit; uniform without replacement','quantile_method':'linear'},
        'sample_counts':{k:len(v) for k,v in pooled.items()},
        'sample_value_sha256':{k:hashlib.sha256(v.astype('<f8').tobytes()).hexdigest() for k,v in pooled.items()},
        'empty_boundary_candidate_frames':sum(e['boundary_eligible_pixels']==0 for e in evidence),
        'data_evidence':evidence,'wall_seconds':time.monotonic()-started,'numpy_version':np.__version__,
        'script_sha256':sha(__file__),'contract_module_sha256':sha(metrics.__file__),
        'membership_sha256':a.membership_sha256,'runtime_sha256':a.runtime_sha256,
        'thresholds_fixed_before_validation':True,'GPU_executed':False,
        'scope':'Same training-only thresholds for all arms; D200 GT fit is evaluation definition, not reference model updates'}
    metrics.validate_region_contract(result, metrics.canonical_rows(b.canonical), roles, require_complete_fit=False)
    write(a.output_dir/'regions.json', result)
    print(json.dumps({'status':'REGIONS_COMPLETE','frames':len(rows),'sha256':sha(a.output_dir/'regions.json')}),flush=True)


def calibrate(a, prepared):
    import numpy as np
    import torch
    runtime,members,base,b,context,roles = prepared
    root=Path(runtime['breadth_root'])
    loader=module('r13_native_loader',root/'run_same_shape_reference.py')
    cma=module('r13_original_cma',root/'collect_bn_cma.py')
    scope=module('r13_bn_inventory',root/'run_bn_scope.py')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.manual_seed(20261016)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.manual_seed_all(20261016)
    native,trainer,metrics,_,hashes,config,saved,side,_,plan=context
    result={'schema':'flow_round13_common_bn_v1','status':'RUNNING','checkpoint_sha256':b.checkpoint_sha256,
            'membership_sha256':a.membership_sha256,'runtime_sha256':a.runtime_sha256,
            'bn_target_ids':[r['target_id'] for r in members['bn_members']],
            'calibration_forwards':0,'calibration_bn_count':78,'retained_mlp_bn_count':24,
            'BN_scope':'MLP_R13_COMMON_CMA176','calibration_definition':'all78_reset_CMA_train_nonBN_eval',
            'old_CMA_loaded':False,'GT_used':False,'optimizer_steps':0,'input_evidence':[],
            'source_sha256':{str(p):sha(p) for p in [Path(__file__),Path(base.__file__),Path(loader.__file__),Path(cma.__file__),Path(scope.__file__)]},
            'runtime':{'torch':str(torch.__version__),'numpy':np.__version__,'cuda':torch.version.cuda,
                       'device':torch.cuda.get_device_name(0),'precision':'FP32 noAMP TF32off'}}
    started=time.monotonic()
    try:
        model,functional=loader.load_model(b,context,base,torch,device)
        all_bn,mlp_bn=scope.bn_inventory(model,torch)
        cma.reset_bn_cma(model,torch)
        before=cma.non_bn_digest(model,all_bn)
        result['non_bn_state_before_sha256']=before
        with torch.inference_mode():
            for row in members['bn_members']:
                relative=native.target_files(row)['input'];path=native.checked_path(b.data_root,relative)
                expected=hashes[relative]['sha256'];require(sha(path)==expected,'BN event bytes differ')
                events=np.load(path,allow_pickle=False)
                require(events.shape==(10,480,640) and events.dtype==np.float32 and np.isfinite(events).all(),'BN events invalid')
                chunk=base.prepare_input(events,device,torch);functional.reset_net(model)
                flows=model(chunk)['flow']
                require(len(flows)==4 and all(bool(torch.isfinite(x).all()) for x in flows),'Nonfinite native flows')
                del chunk,events,flows
                result['calibration_forwards']+=1
                result['input_evidence'].append({'target_id':row['target_id'],'relative':relative,'sha256':expected})
                if result['calibration_forwards']%16==0:
                    print(json.dumps({'BN_completed':result['calibration_forwards'],'of':176}),flush=True)
                    write(a.output_dir/'bn_receipt.json',result)
        after=cma.non_bn_digest(model,all_bn);require(after==before,'Non-buffer state changed')
        buffers={}
        for name,bn in all_bn.items():
            require(int(bn.num_batches_tracked)==176 and bool(torch.isfinite(bn.running_mean).all()) and
                    bool(torch.isfinite(bn.running_var).all()) and bool((bn.running_var>=0).all()),'Invalid/incomplete CMA')
            if name in mlp_bn:
                for field in ['running_mean','running_var','num_batches_tracked']:
                    buffers[name+'.'+field]=getattr(bn,field).detach().cpu().numpy().copy()
        require(len(buffers)==72,'Retained buffer count')
        path=a.output_dir/'bn_buffers.npz';np.savez_compressed(path,**buffers)
        result.update(status='COMMON_BN_COMPLETE',buffers_sha256=sha(path),non_bn_state_unchanged=True,
                      non_bn_state_after_sha256=after,retained_names=sorted(mlp_bn),
                      max_allocated_bytes=torch.cuda.max_memory_allocated(0),max_reserved_bytes=torch.cuda.max_memory_reserved(0))
    except BaseException as exc:
        result.update(status='INCOMPLETE_NO_SCIENTIFIC_CONCLUSION',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        raise
    finally:
        result['wall_seconds']=time.monotonic()-started
        write(a.output_dir/'bn_receipt.json',result)
    print(json.dumps({'status':result['status'],'forwards':result['calibration_forwards'],'receipt_sha256':sha(a.output_dir/'bn_receipt.json')}),flush=True)


def main():
    p=argparse.ArgumentParser()
    for key in ['runtime','membership']:
        p.add_argument('--'+key,type=Path,required=True);p.add_argument('--'+key+'-sha256',required=True)
    p.add_argument('--phase',choices=['regions','bn'],required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();prepared=prepare(a)
    if not a.execute:
        print(json.dumps({'status':'PREPARATION_PREFLIGHT_PASS','phase':a.phase,'no_GPU':True}));return
    a.output_dir.mkdir(parents=True,exist_ok=False)
    if a.phase=='regions':fit_regions(a,prepared);return
    lock_path=Path('/data/flow_a/runs/.flow_a_gpu0.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def timeout(signum,frame):raise TimeoutError('R13 BN 15 minute cap')
        signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,900)
        try:calibrate(a,prepared)
        finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':main()
