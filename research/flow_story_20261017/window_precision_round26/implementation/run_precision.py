#!/usr/bin/env python3
"""R26 full-space precision feasibility. Data preflight only reused; no compact path."""
import argparse
import contextlib
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
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
TARGET = 'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.2.mlp'
ARMS = ('native', 'Q8ref', 'Q8serial', 'Q6', 'Q4')
PRIMITIVE_SHA = '759eaa6917ad6f02b25793f11c26c961bf375eaf62249c6299c7b014dd78b916'
DATA_HELPER_SHA = '4677fe4c1fb7f296ffea62887381ada7bcab46b8e641e14cbd8f31de86abde66'
GRAPH_ID = 'r26_W271482_all78current_s2b2_fullspatial_weightplanes_H8'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def checked(spec):
    require(set(spec) == {'path', 'sha256'}, 'Path/SHA asset required')
    p = Path(spec['path'])
    require(p.is_file() and sha(p) == spec['sha256'], 'Actual asset differs: '+str(p))
    return p


def plan(rows, mode):
    require(mode == 'precision42' and len(rows) == 8, 'Fixed H8 precision42 only')
    items = []
    for i, row in enumerate(rows):
        for arm in ARMS:
            items.append(dict(target_id=row['target_id'], arm=arm, endpoint_arm=arm,
                              execution='native' if arm == 'native' else 'integer_quantized', quality_endpoint=True))
            if i == 0 and arm == 'Q6':
                for j in (1,2):
                    items.append(dict(target_id=row['target_id'], arm='Q6_repeat%d'%j, endpoint_arm='Q6',
                                      execution='integer_quantized', quality_endpoint=False))
    return [dict(item, forward_id=i+1) for i,item in enumerate(items)]


def expected_counts():
    return {'model_forwards':42, 'native_forwards':8, 'integer_reference_forwards':8,
            'bitplane_forwards':26, 'expected_native_BN_calls':3276,
            'expected_full_grid_logical_FC_rows_each':23436000,
            'expected_original_FC_rows_each':23028000,
            'expected_target_native_FC_rows_each':96000,
            'expected_target_integer_dense_rows_each':96000,
            'expected_target_row_plane_work_each':1990560,
            'expected_model_int_mm_calls_each_targetFC':216,
            'backend_operator_smoke_calls':4, 'backend_smoke_model_forwards':0,
            'expected_saved_arrays':314}


def prepare(a):
    import torch
    require(not torch.cuda.is_initialized(), 'CPU preflight only')
    require(not a.output.exists() and 0 < a.wall_minutes <= 20, 'Fresh output, bounded inference')
    require(sha(a.inputs) == a.inputs_sha256 and sha(a.freeze) == a.freeze_sha256, 'Explicit manifest/freeze bytes')
    inp, freeze = read(a.inputs), read(a.freeze)
    require(inp['schema'] == 'flow_r26_window_precision_inputs_v1', 'Explicit R24 input schema')
    require(freeze['schema'] == 'flow_r26_window_precision_source_freeze_v1', 'Frozen source schema')
    for spec in freeze['files']:
        checked(spec)
    require(any(Path(s['path']).resolve() == Path(__file__).resolve() for s in freeze['files']), 'Actual entrypoint must be frozen')
    admission_path = checked(inp['independent_admission'])
    primitive_path = checked(inp['primitive'])
    data_helper_path = checked(inp['data_helper'])
    require(inp['BN_mode'] == 'ALL78_NATIVE_CURRENT', 'All78 native current BN')
    require(sha(primitive_path) == PRIMITIVE_SHA and sha(data_helper_path) == DATA_HELPER_SHA, 'Explicit new integer and old data-only source')
    helper = load(data_helper_path, '_r26_data_only_helper')
    admission = read(admission_path)
    require(admission['schema'] == 'flow_r26_precision_independent_admission_v1' and admission['status'] == 'ADMIT'
            and a.plan in admission['permitted_plans'], 'Independent CPU source admission')
    require(admission['runner_sha256'] == sha(__file__) and admission['core_sha256'] == sha(primitive_path), 'Admission source bytes')
    for actual in (primitive_path, data_helper_path):
        require(any(Path(s['path']).resolve() == actual.resolve() for s in freeze['files']), 'Actual source must be frozen')
    old = Path(inp['old_root'])
    conversion = old/'bn_conversion_round14'
    sys.path.insert(0, str(conversion))
    r14 = load(conversion/'run_conversion_v2.py', '_r26_original_source_helpers')
    k, h0, modules = r14.runtime()
    rtpath, memberpath = checked(inp['runtime']), checked(inp['membership'])
    h8path, regionpath, cp = checked(inp['h8']), checked(inp['regions']), checked(inp['checkpoint'])
    rt, members, h8, regions = map(read, (rtpath, memberpath, h8path, regionpath))
    require(sha(memberpath) == k.MEMBER_SHA and sha(regionpath) == k.REGION_SHA, 'Fixed membership/regions')
    require(h8 == h0.h8_manifest(members, sha(memberpath)), 'Exact identity-only H8 selector')
    require(sha(cp) == k.CHECKPOINT_SHA, 'Only original W271482')
    require(sha(cp.with_suffix('.json')) == k.SIDECAR_SHA and sha(cp.parent/'run_contract.json') == k.RUN_CONTRACT_SHA, 'Checkpoint sidecar/run source')
    for path, digest in rt['source_pins'].items():
        require(sha(path) == digest, 'Frozen native dependency '+path)
    sys.path[:0] = [rt['breadth_root'], rt['motion_root']]
    base = load(Path(rt['breadth_root'])/'evaluate_breadth.py', '_r26_native_preflight')
    args = list(rt['base_args'])
    require(not any(x.startswith('--bn-') or x.startswith('--svs-') or x.startswith('--regions') or x == '--run-inference' for x in args), 'No implicit legacy experiment')
    for flag, value in (('--checkpoint', str(cp)), ('--checkpoint-sha256', k.CHECKPOINT_SHA), ('--output', str(a.output/'results.json'))):
        require(args.count(flag) == 1, 'Single native argument '+flag)
        args[args.index(flag)+1] = value
    b = base.parse_args(args)
    b.device = 'cuda:0'
    context = list(helper.source_preflight(b, base))
    native, trainer, metrics, _, hashes, config, run, side, _, _ = context
    canonical = metrics.canonical_rows(b.canonical)
    roles = metrics.auxiliary_roles(b.roles, canonical)
    modules[0].validate_members(members, canonical, roles)
    r14.validate_manifest_sources(members, b.canonical, b.roles)
    metrics.validate_region_contract(regions, canonical, roles, require_complete_fit=False)
    require(regions['fit_target_ids'] == [r['target_id'] for r in members['train_members_diversity']], 'Fixed training region fit')
    context[8] = regions
    cp_audit = r14.validate_checkpoint(cp, '271482', tuple(context), h0, torch)
    evidence = []
    for row in h8['members']:
        files = native.target_files(row)
        entry = {}
        for kind, relative in files.items():
            require(sha(native.checked_path(b.data_root, relative)) == hashes[relative]['sha256'], 'Actual H8 input bytes')
            entry['events' if kind == 'input' else kind] = hashes[relative]['sha256']
        evidence.append(dict(target_id=row['target_id'], input_evidence=entry))
    context[9]['allowed8_current_files_SHA_checked'] = 24
    binding = {'graph_id': GRAPH_ID, 'plan': a.plan, 'forward_plan': plan(h8['members'], a.plan),
               'checkpoint_cpu_audit': cp_audit, 'h8_sha256': sha(h8path), 'membership_sha256': sha(memberpath),
               'region_sha256': sha(regionpath), 'inputs_sha256': a.inputs_sha256, 'source_freeze_sha256': a.freeze_sha256,
               'independent_admission_sha256': sha(admission_path), 'other_11_MLPs': 'native',
               'target': TARGET, 'BN': 'all78_current', 'BN_mode': inp['BN_mode'],
               'quality_endpoints': 40, 'determinism_replays': 2, 'optimizer_steps': 0, 'calibration_forwards': 0,
               'primitive_sha256': sha(primitive_path), 'data_only_helper_sha256': sha(data_helper_path),
               'old_spatial_compact_called': False, 'source_quantization_device': 'cpu',
               'precision': 'FP32 activations/currentBN, exact int8 x int8 -> int32 FC; no AMP/TF32; T10/B1/full9',
               'data_preflight': context[9], 'wall_minutes': a.wall_minutes, **expected_counts()}
    return dict(binding=binding, data_evidence=evidence, h8=h8, context=tuple(context), base=base, b=b,
                rt=rt, r14=r14, k=k, h0=h0, modules=modules, primitive_path=primitive_path)


def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def descriptor(x):
    return {'shape':list(x.shape), 'dtype':str(x.dtype), 'sha256':tensor_sha(x)}


def forward(model, functional, targets, shapes, chunk, item, primitive, specs, torch):
    """Native model/MLP/PSN/BN, replacing only the two target FC callables."""
    functional.reset_net(model)
    target = targets[TARGET]
    originals = {fc:getattr(target,fc) for fc in ('fc1','fc2')}
    mode = item['endpoint_arm']
    require(mode in ARMS, 'One frozen arm')
    active = originals if mode == 'native' else {fc:primitive.QuantLinear(originals[fc], specs[fc], mode).eval() for fc in originals}
    handles, captured = [], {}
    original_rows = {name:{'fc1_rows':0,'fc2_rows':0,'fc1_calls':0,'fc2_calls':0} for name in targets}
    active_rows = {fc:0 for fc in originals}
    bn_calls, bn_inputs = {}, {}
    def store_target(module, ins, out):
        require('u' not in captured, 'One target MLP call')
        captured['u'], captured['branch'] = ins[0].detach().cpu().clone(), out.detach().cpu().clone()
    try:
        for name,module in targets.items():
            for fc in originals:
                def observe(module, ins, name=name, fc=fc):
                    original_rows[name][fc+'_rows'] += ins[0].numel()//ins[0].shape[-1]
                    original_rows[name][fc+'_calls'] += 1
                handles.append(getattr(module,fc).register_forward_pre_hook(observe))
        for name,module in model.named_modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                bn_calls[name], bn_inputs[name] = 0, []
                def bn_observe(module, ins, name=name):
                    bn_calls[name] += 1
                    bn_inputs[name].append({'shape':list(ins[0].shape),'dtype':str(ins[0].dtype),'stride':list(ins[0].stride())})
                handles.append(module.register_forward_pre_hook(bn_observe))
        for sn in ('sn1','sn2'):
            def capture_sn(module, ins, out, sn=sn):
                require(sn not in captured, 'One actual PSN call')
                captured[sn] = out.detach().cpu().clone()
            handles.append(getattr(target,sn).register_forward_hook(capture_sn))
        handles.append(target.register_forward_hook(store_target))
        for fc,module in active.items():
            if mode != 'native':
                def count_active(module, ins, fc=fc):
                    active_rows[fc] += ins[0].numel()//ins[0].shape[-1]
                handles.append(module.register_forward_pre_hook(count_active))
                setattr(target,fc,module)
        with torch.inference_mode():
            output = model(chunk)
        flows = [x.detach().cpu().clone() for x in output['flow']]
        require(len(flows)==4 and all(bool(torch.isfinite(x).all()) for x in flows), 'Four finite flow heads')
        require(set(captured)=={'u','branch','sn1','sn2'} and all(bool(torch.isfinite(x).all()) for x in captured.values()), 'Finite native full-grid captures')
        require(all(list(captured[sn].shape[:-1])==list(captured['u'].shape[:-1]) for sn in ('sn1','sn2')), 'No spatial/T/branch compression')
        target_execution = {'mode':mode,'full_spatial_grid_retained':True, 'old_compact_called':False,
                            'BN_mode':'ALL78_NATIVE_CURRENT','FC':{}}
        for name,counts in original_rows.items():
            expected = 0 if name==TARGET and mode!='native' else math.prod(shapes[name][:-1])
            require(counts['fc1_rows']==counts['fc2_rows']==expected and
                    counts['fc1_calls']==counts['fc2_calls']==(0 if expected==0 else 1), 'True original Linear calls/rows '+name)
        if mode != 'native':
            for fc,module in active.items():
                require(active_rows[fc] == math.prod(shapes[TARGET][:-1]), 'Full-grid active wrapper row count')
                target_execution['FC'][fc] = module.last_audit
                captured[fc+'_integer'] = module.last_integer.detach().cpu().clone()
        require(len(bn_calls)==78 and all(n==1 for n in bn_calls.values()), 'Exactly all78 native BN modules once')
        logical_rows = {name:{'fc1_rows':math.prod(shape[:-1]),'fc2_rows':math.prod(shape[:-1])} for name,shape in shapes.items()}
        return flows,captured,dict(item, all12_original_fc_observations=original_rows,
                all12_full_grid_logical_fc_rows=logical_rows, target_active_quantized_rows=active_rows,
                native_BN_calls=bn_calls,native_BN_inputs=bn_inputs,target_execution=target_execution,
                summed_original_fc1_rows=sum(v['fc1_rows'] for v in original_rows.values()),
                summed_original_fc2_rows=sum(v['fc2_rows'] for v in original_rows.values()),
                summed_full_grid_logical_fc1_rows=sum(v['fc1_rows'] for v in logical_rows.values()),
                summed_full_grid_logical_fc2_rows=sum(v['fc2_rows'] for v in logical_rows.values()),
                tensor_records={k:descriptor(v) for k,v in captured.items()},head_records=[descriptor(v) for v in flows])
    finally:
        for fc,module in originals.items(): setattr(target,fc,module)
        for handle in handles: handle.remove()
        require(all(model.get_submodule(name) is module for name,module in targets.items()), 'Original MLP identities restored')
        require(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()), 'All hooks removed')


def exact_pair(actual, reference, torch):
    af,ac,_=actual; rf,rc,_=reference
    required={'u','branch','sn1','sn2','fc1_integer','fc2_integer'}
    require(set(ac)==set(rc)==required and len(af)==len(rf)==4, 'All ten actual reference arrays')
    aa=dict(ac,**{'head%d'%i:x for i,x in enumerate(af)})
    rr=dict(rc,**{'head%d'%i:x for i,x in enumerate(rf)})
    checks={}
    for name,x in aa.items():
        y=rr[name]
        require(x.shape==y.shape and x.dtype==y.dtype, 'Exact pair shape/dtype')
        xb,yb=x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)
        checks[name]={'bitwise_equal':bool(torch.equal(xb,yb)), 'different_bytes':int((xb!=yb).sum()),
                      'shape':list(x.shape),'dtype':str(x.dtype),'actual_sha256':tensor_sha(x),'reference_sha256':tensor_sha(y)}
    return {'pass':all(v['bitwise_equal'] for v in checks.values()),'array_checks':checks}


def keep_captures(item, first_id):
    return item['arm'] in ('native','Q8ref','Q8serial') or (item['target_id']==first_id and item['endpoint_arm']=='Q6')


def execute(a,p,result):
    import numpy as np
    import torch
    require(not torch.cuda.is_initialized(), 'CUDA starts only inside global lock')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.manual_seed(20261017)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.manual_seed_all(20261017)
    begin=time.monotonic()
    primitive=load(p['primitive_path'],'_r26_integer_weight_planes')
    result.update(status='BACKEND_SMOKE_RUNNING',inference_executed=False,backend_operator_smoke=[],
                  runtime={'torch':str(torch.__version__),'numpy':np.__version__,'device':torch.cuda.get_device_name(device),
                           'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'TF32':False,'AMP':False})
    write(a.output/'results.json',result)
    # Four explicit true-shape integer operator checks; these are not model forwards.
    result['backend_operator_smoke']=primitive.backend_shape_smoke(device)
    require(len(result['backend_operator_smoke'])==4,'Four backend integer calls')
    write(a.output/'results.json',result)
    rt,base,b,context=p['rt'],p['base'],p['b'],p['context']
    native,trainer,metrics,_,hashes,config,run,side,regions,_=context
    r13,collector,probe,parent,r9,r8=p['modules']
    loader=load(Path(rt['breadth_root'])/'run_same_shape_reference.py','_r26_native_loader')
    scope=load(Path(rt['breadth_root'])/'run_bn_scope.py','_r26_BN_inventory')
    model,functional=loader.load_model(b,context,base,torch,device)
    require(r13.state_digest(model)==p['k'].SOURCE_STATE_SHA,'Actual original W271482')
    bn=p['r14'].core.switch_allcurrent(model,base,scope,r13.state_digest,torch,p['k'].SOURCE_NON_BN_SHA)
    targets,shapes=parent.target_layers(model)
    require(tuple(shapes[TARGET])==(10,1,30,40,384),'Fixed target geometry')
    require(targets[TARGET].fc1.bias is None and targets[TARGET].fc2.bias is None,'Original two FC biases absent')
    before,signature=r13.state_digest(model),base.bn_signature(model,torch)
    cpu_specs={fc:primitive.quantize_source(getattr(targets[TARGET],fc)) for fc in ('fc1','fc2')}
    quant_arrays={fc+'__'+key:spec[key].numpy() for fc,spec in cpu_specs.items() for key in ('q8','scale')}
    np.savez_compressed(a.output/'quantization.npz',**quant_arrays)
    quant_receipt={fc:{k:v for k,v in spec.items() if k not in ('q8','scale','bias')} for fc,spec in cpu_specs.items()}
    write(a.output/'quantization.json',dict(schema='flow_r26_weight_quantization_v1',checkpoint_sha256=p['k'].CHECKPOINT_SHA,
        original_allcurrent_state_sha256=before,FC=quant_receipt,arrays='quantization.npz',arrays_sha256=sha(a.output/'quantization.npz'),
        arrays_records={name:{'shape':list(arr.shape),'dtype':str(arr.dtype),'sha256':hashlib.sha256(arr.tobytes()).hexdigest()} for name,arr in quant_arrays.items()}))
    specs={fc:primitive.place_spec(spec,device) for fc,spec in cpu_specs.items()}
    result.update(status='RUNNING',inference_executed=True,BN_scope=bn,initial_allcurrent_state_sha256=before,
                  forward_records=[],numerical_checks=[],quantization_receipt={'path':'quantization.json','sha256':sha(a.output/'quantization.json')},
                  quantization_archive={'path':'quantization.npz','sha256':sha(a.output/'quantization.npz')})
    rows={r['target_id']:r for r in p['h8']['members']};first_id=p['h8']['members'][0]['target_id']
    last_id=None;native_out=q8_reference=q6_reference=None
    for item in p['binding']['forward_plan']:
        require(time.monotonic()-begin<a.wall_minutes*60,'Fixed wall cap')
        tid=item['target_id']
        if tid!=last_id:
            files=native.target_files(rows[tid]);evidence={'events' if q=='input' else q:hashes[v]['sha256'] for q,v in files.items()}
            path=native.checked_path(b.data_root,files['input']);require(sha(path)==evidence['events'],'Actual event bytes')
            events=np.load(path,allow_pickle=False)
            require(events.shape==(10,480,640) and events.dtype==np.float32 and np.isfinite(events).all(),'Original event input')
            chunk=base.prepare_input(events,device,torch);del events
            last_id=tid;native_out=q8_reference=None
        result['full_forward_attempts']+=1;write(a.output/'results.json',result)
        output=forward(model,functional,targets,shapes,chunk,item,primitive,specs,torch)
        flows,captures,record=output;record['input_evidence']=evidence
        filename='%03d_%s_%s.npz'%(item['forward_id'],tid,item['arm'])
        arrays={'head%d'%i:x.numpy() for i,x in enumerate(flows)}
        if keep_captures(item,first_id):arrays.update({k:v.numpy() for k,v in captures.items()})
        np.savez_compressed(a.output/filename,**arrays)
        record.update(prediction_file=filename,prediction_file_sha256=sha(a.output/filename),saved_array_names=list(arrays))
        result['forward_records'].append(record);result['full_forward_count']+=1
        check=None
        if item['arm']=='native':native_out=output
        else:require(native_out is not None and tensor_sha(captures['u'])==tensor_sha(native_out[1]['u']),'Exact common predecision input')
        if item['arm']=='Q8ref':q8_reference=output
        elif item['arm']=='Q8serial':
            require(q8_reference is not None,'Actual dense integer reference')
            check=dict(kind='Q8serial_vs_Q8ref_exact',reference_forward_id=q8_reference[2]['forward_id'],**exact_pair(output,q8_reference,torch))
        elif item['arm']=='Q6' and tid==first_id:q6_reference=output
        elif item['arm'].startswith('Q6_repeat'):
            require(q6_reference is not None,'Actual first Q6 reference')
            check=dict(kind='Q6_exact_replay',reference_forward_id=q6_reference[2]['forward_id'],**exact_pair(output,q6_reference,torch))
        if check:result['numerical_checks'].append(dict(target_id=tid,forward_id=item['forward_id'],arm=item['arm'],**check))
        p['k'].core.check_closure(model,targets,before,r13.state_digest,signature,base.bn_signature,torch)
        record.update(state_and_BN_unchanged=True,original_modules_restored_and_hooks_removed=True)
        write(a.output/'results.json',result)
        print(json.dumps({'forward':item['forward_id'],'arm':item['arm'],'numeric_pass':None if check is None else check['pass']}),flush=True)
        require(check is None or check['pass'],'Predeclared exact integer graph contract failed')
    require(result['full_forward_count']==42 and sum(r['quality_endpoint'] for r in result['forward_records'])==40 and len(result['numerical_checks'])==10,'Complete42/40/10 closure')
    require(sum(len(r['saved_array_names']) for r in result['forward_records'])==314,'Fixed raw payload inventory')
    result.update(status='WINDOW_PRECISION42_COMPLETE',final_state_sha256=r13.state_digest(model),
                  all_states_unchanged=True,all_hooks_removed=True,original_modules_restored=True,
                  wall_seconds=time.monotonic()-begin,max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--inputs-sha256',required=True)
    parser.add_argument('--freeze',type=Path,required=True);parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--plan',choices=('precision42',),required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--wall-minutes',type=float,default=20)
    parser.add_argument('--preflight-output',type=Path);parser.add_argument('--execute',action='store_true')
    a=parser.parse_args();p=prepare(a)
    result=dict(schema='flow_r26_window_precision_v1',graph_id=GRAPH_ID,status='PREFLIGHT_PASS_NO_INFERENCE',binding=p['binding'],
                data_evidence=p['data_evidence'],inference_executed=False,full_forward_count=0,full_forward_attempts=0,
                training_performed=False,optimizer_steps=0,calibration_forwards=0,flow_specific_increment_claim=False,hardware_benefit_claim=False)
    if not a.execute:
        require(a.preflight_output and not a.preflight_output.exists(),'Fresh preflight receipt');write(a.preflight_output,result);print(result['status']);return
    lock=p['modules'][3].GPU_LOCK.open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);a.output.mkdir(parents=True,exist_ok=False);write(a.output/'results.json',result)
        def alarm(signum,frame):raise TimeoutError('R26 bounded42 hard wall cap')
        signal.signal(signal.SIGALRM,alarm);signal.alarm(int(a.wall_minutes*60)+60)
        try:execute(a,p,result)
        except BaseException as exc:
            result.update(status='INCOMPLETE_OR_NUMERICAL_FAILURE',error=str(exc),traceback=traceback.format_exc());write(a.output/'results.json',result);raise
        finally:signal.alarm(0)
        write(a.output/'results.json',result)
    finally:lock.close()


if __name__=='__main__':main()
