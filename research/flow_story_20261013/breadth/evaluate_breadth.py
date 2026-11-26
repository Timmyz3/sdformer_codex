#!/usr/bin/env python3
"""Bounded TRAIN-side breadth exploration on a pinned native-v1 checkpoint.

Default: CPU provenance/data preflight. Explicit --run-inference uses the named
device. No final-valid data, training, checkpoint mutation, or hardware kill gate.
"""
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SUPPORT_SHA = 'db426409fa805b960f6d9cede548341e55add7f599f38a8655f02104c85ff70f'
SAMPLE_SHA = '41309684f927a10ef24b479178ec73a473da9d2ee31880011646f5f682d80509'
SCHEMA = 'flow_a_native_encoding_v2_checkpoint_v1'
CREATOR = 'flow_a_20261012_native_encoding_baseline'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(ok, message):
    if not ok: raise ValueError(message)


def read_pin(path, expected):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'Artifact SHA differs: '+str(path))
    return json.loads(raw)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def support_modules(prior):
    pins = read_pin(HERE/'support_source_pins.json', SUPPORT_SHA)
    for relative, expected in pins.items(): require(sha(prior/relative) == expected, 'Prior implementation changed: '+relative)
    sys.path.insert(0, str(prior/'baseline'))
    native = import_file('native_data_acceptance', prior/'baseline/native_data_acceptance.py')
    trainer = import_file('breadth_native_trainer', prior/'baseline/train_native_baseline.py')
    metrics = import_file('breadth_metrics', prior/'evaluation/flow_eval_contract.py')
    return native, trainer, metrics, pins


def selected_rows(canonical, roles, metrics):
    records = metrics.canonical_rows(canonical); role_map = metrics.auxiliary_roles(roles, records)
    sample = read_pin(HERE/'gate_dev32.json', SAMPLE_SHA)
    require(sample['canonical_sha256'] == metrics.CANONICAL_SHA and sample['roles_sha256'] == metrics.AUX_ROLES_SHA,
            'Frozen breadth sample source differs')
    allowed = [r for r in records if role_map.get(r['target_id']) == 'gate_development']
    ordered = sorted(allowed, key=lambda r: (hashlib.sha256((sample['seed']+r['target_id']).encode()).hexdigest(),r['target_id']))[:32]
    require([r['target_id'] for r in ordered] == sample['target_ids'] and len(ordered) == 32 and
            metrics.ids_sha(sample['target_ids']) == sample['target_ids_sha256'], 'Frozen32 IDs/order differ')
    require(all(r['protocol_split'] == 'train' for r in ordered), 'Final-valid prohibited in breadth')
    return ordered, records, role_map, sample


def validate_cases(value):
    require(value.get('schema') == 'flow_breadth_cases_v1', 'Unknown case manifest')
    cases = value['cases']; require(1 <= len(cases) <= 48, 'Expected1..48 explicitly declared cases')
    require(len({c['id'] for c in cases}) == len(cases), 'Duplicate case ID')
    for c in cases:
        require(c['id'] and all(ch.isalnum() or ch in '_-' for ch in c['id']), 'Unsafe case ID')
        require(c['bn'] in ('P_paper','P_code','D') and c['method'] in ('dense','tile','static'), 'Unsupported BN/method')
        if c['method'] == 'tile':
            require(c['bn'] != 'P_paper', 'Tile execution needs frozen BN; P-paper remains a dense diagnostic')
            require(c.get('execution','actual_skip') in ('actual_skip','dense_masked'), 'Unknown tile execution')
            require(type(c.get('keep_fraction')) in (int,float) and 0 <= c['keep_fraction'] <= 1, 'Bad tile keep fraction')
    return cases


def region_source_binding(regions,records,roles,hashes,native,metrics):
    metrics.validate_region_contract(regions,records,roles,require_complete_fit=True)
    require(regions.get('fit_uses_final_valid') is False and regions.get('thresholds_fixed_before_validation') is True,
            'Region fit must precede final validation and use training only')
    evidence=regions.get('data_evidence',[]);ids=regions['fit_target_ids']
    require([r['target_id'] for r in evidence]==ids,'Region data evidence order/coverage differs')
    by_id={r['target_id']:r for r in records}
    for item in evidence:
        paths=native.target_files(by_id[item['target_id']])
        require(all(item.get(k+'_sha256')==hashes[paths[k]]['sha256'] for k in ('gt','mask')),
                'Region fit GT/mask differs from native acceptance inventory')
    return {'fit_targets':len(ids),'GT_mask_native_source_hash_checks':2*len(ids),'passed':True}


def validate_contract(run, side, data_binding, sources, trainer, train_rows, hashes):
    contract = run['contract']; provenance = run['provenance']
    require(side.get('schema_version') == contract.get('schema_version') == SCHEMA and
            side.get('created_by') == provenance.get('created_by') == CREATOR and
            side.get('training_scope') == contract.get('scope') == 'canonical6000', 'Old/engineering checkpoint rejected')
    require(side.get('contract_sha256') == run.get('contract_sha256') == digest(contract), 'Training contract digest differs')
    require(provenance.get('random_init') is True and provenance.get('old_checkpoint_loaded') is False and
            provenance.get('training_protocol_split') == 'train' and provenance.get('BN_policy') == 'native_train_tracked',
            'Checkpoint initialization/BN provenance differs')
    require(contract.get('canonical_manifest_sha256') == provenance.get('canonical_manifest_sha256') == data_binding['canonical_manifest_sha256'],
            'Checkpoint canonical SHA differs')
    require(contract.get('source_sha256') == provenance.get('source_sha256') == sources and
            contract.get('source_manifest_sha256') == trainer.load_core().SOURCE_MANIFEST_SHA and
            contract.get('trainer_sha256') == trainer.CORE_SHA, 'Checkpoint source/core differs')
    ids = [r['target_id'] for r in train_rows]
    require(contract.get('training_target_ids_sha256') == provenance.get('training_target_ids_sha256') == digest(ids) and
            contract.get('training_files_sha256') == provenance.get('training_manifest_sha256') == digest(hashes),
            'Checkpoint training membership/content differs')
    config = copy.deepcopy(run['effective_config']); normalized = copy.deepcopy(config)
    normalized['data']['path'] = '<verified_DSEC>/saved_flow_data'; normalized['loader']['gpu'] = '<logical-device>'
    require(normalized == contract.get('effective_config') and config.get('data_contract') == data_binding,
            'Checkpoint effective config/native encoding contract differs')
    require(provenance.get('window_size') == [2,9,9] and config['swin_transformer']['window_size'] == [2,9,9] and
            config['data']['num_frames'] == 10 and config['loader']['batch_size'] == 1 and
            config['loader']['polarity'] is True and config['model']['norm_input'] == 'minmax' and
            config['model']['name'] == 'MS_SpikingformerFlowNet_en4' and config['metrics']['flow_scaling'] == 1 and
            config['optimizer']['use_amp'] is False, 'Only declared native T10 B1 FP32 W9 supported')
    require(type(side.get('completed_optimizer_steps')) is int and side['completed_optimizer_steps'] > 0,
            'Untrained initialization is not this exploration checkpoint')
    return config


def preflight(a):
    native, trainer, metrics, pins = support_modules(a.prior_root)
    rows, canonical, roles, sample = selected_rows(a.canonical,a.roles,metrics)
    train, hashes, binding, data_audit = native.validate_native_bundle(a.data_root,a.canonical,a.acceptance,
        a.acceptance_sha256,a.archive_audit,check_current_files=True)
    binding.update(entrypoint_sha256=pins['baseline/train_native_baseline.py'],adapter_sha256=pins['baseline/native_data_acceptance.py'],
                   training_core_sha256=trainer.CORE_SHA,checkpoint_schema=SCHEMA,checkpoint_creator=CREATOR)
    core = trainer.load_core(); sources = native.pinned_json(a.source_manifest,core.SOURCE_MANIFEST_SHA)
    require(len(sources)==63,'Expected63 native source files')
    for relative, expected in sources.items(): require(sha(native.checked_path(a.source_root,relative))==expected,'Source differs:'+relative)
    require(sha(a.checkpoint)==a.checkpoint_sha256,'Explicit frozen checkpoint SHA differs')
    side = json.loads(a.checkpoint.with_suffix('.json').read_text())
    require(side.get('checkpoint_sha256')==a.checkpoint_sha256 and side.get('checkpoint_file')==a.checkpoint.name,'Checkpoint sidecar differs')
    run = json.loads((a.checkpoint.parent/'run_contract.json').read_text())
    config = validate_contract(run,side,binding,sources,trainer,train,hashes)
    cases = validate_cases(read_pin(a.cases,a.cases_sha256))
    regions = None;region_evidence=None
    if a.regions:
        regions = metrics.validate_region_contract(read_pin(a.regions,a.regions_sha256),canonical,roles,require_complete_fit=True)
        region_evidence=region_source_binding(regions,canonical,roles,hashes,native,metrics)
    plan = {'schema':'flow_breadth_exploration_v1','status':'PREFLIGHT_PASS_NO_INFERENCE','inference_executed':False,
        'scientific_reproduction_claim':False,'method_rejection_claim':False,'checkpoint_sha256':a.checkpoint_sha256,
        'checkpoint_sidecar':side,'checkpoint_run_contract_sha256':sha(a.checkpoint.parent/'run_contract.json'),
        'training_progress_label':'Interim canonical6000 native training checkpoint; adequacy and paper AEE reproduction unproven',
        'population':sample,'geometry':a.geometry,'precision':'FP32, TF32 off, no autocast','data_contract':binding,
        'data_preflight':data_audit,'source_sha256':sources,'support_implementation_sha256':pins,
        'script_sha256':sha(__file__),'cases_sha256':a.cases_sha256,'cases':cases,
        'region_status':'COMPLETE4573_GATE_FIT_THRESHOLDS' if regions else 'PENDING_COMPLETE_TRAIN_FIT; all-valid AEE still evaluated',
        'regions_sha256':a.regions_sha256,'region_native_source_binding':region_evidence,
        'actual_skip_paired_targets_per_policy':a.actual_pair_count,'BN_labels':{'P_paper':'eval/dropout-off; current-batch spatial BN',
            'P_code':'eval; original checkpoint tracked running BN','D':'frozen same checkpoint running BN; no recalibration yet, numerical policy equals P_code'},
        'hardware_kill_gate':None,'time_budget_minutes':a.wall_minutes,'arms':{},'training_performed':False}
    return native,trainer,metrics,rows,hashes,config,run,side,regions,plan


def apply_bn(model,policy,torch):
    model.eval(); count=0
    for layer in model.modules():
        require(not layer.training,'Dropout/training module enabled')
        if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):
            count+=1
            if policy=='P_paper':
                layer.track_running_stats=False;layer.running_mean=layer.running_var=layer.num_batches_tracked=None
            else:
                require(layer.track_running_stats and layer.running_mean is not None and layer.running_var is not None and
                    layer.num_batches_tracked is not None and layer.num_batches_tracked.item()>0 and
                    torch.isfinite(layer.running_mean).all().item() and torch.isfinite(layer.running_var).all().item() and
                    not (layer.running_var<0).any().item(),'Frozen BN lacks populated finite checkpoint statistics')
    require(count>0,'No BN modules discovered');return count


def bn_signature(model,torch):
    return {name:{'training':m.training,'tracking':m.track_running_stats,**{key:None if getattr(m,key) is None else
        hashlib.sha256(getattr(m,key).detach().cpu().numpy().tobytes()).hexdigest()
        for key in ('running_mean','running_var','num_batches_tracked')}}
        for name,m in model.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)}


def prepare_input(events,device,torch):
    chunk=torch.from_numpy(np.ascontiguousarray(events,dtype=np.float32)).unsqueeze(0).to(device)
    chunk=torch.cat((torch.relu(chunk).unsqueeze(2),torch.relu(-chunk).unsqueeze(2)),dim=2)
    nz=chunk[chunk!=0];require(nz.numel()>0,'Empty native minmax input; no resampling')
    low,high=nz.min(),nz.max()
    if low!=high:chunk[chunk!=0]=(chunk[chunk!=0]-low)/(high-low)
    return chunk


def compact_policy_records(records):
    """Keep exact whole-tile masks; omit only the redundant spatial expansion."""
    result=[]
    for row in records:
        out=dict(row)
        if 'spatial_keep' in out:
            spatial=np.asarray(out.pop('spatial_keep'),dtype=np.bool_)
            grid=np.asarray(out['unit_keep'],dtype=np.bool_)
            reconstructed=np.repeat(np.repeat(grid,9,axis=0),9,axis=1)[:spatial.shape[0],:spatial.shape[1]]
            require(np.array_equal(spatial,reconstructed),'Whole-tile/spatial mask differs')
            out['spatial_mask_encoding']={'shape':list(spatial.shape),'reconstruct':'repeat unit_keep9 on each spatial axis, then crop to H,W',
                'packed_little_bitorder_sha256':hashlib.sha256(np.packbits(spatial,bitorder='little').tobytes()).hexdigest()}
        result.append(out)
    return result


@contextlib.contextmanager
def linear_observer(model,torch):
    observed={};handles=[]
    for name,layer in model.named_modules():
        if isinstance(layer,torch.nn.Linear):
            def hook(module,inputs,label=name):
                entry=observed.setdefault(label,{'calls':0,'rows':0,'in_features':module.in_features,'out_features':module.out_features})
                entry['calls']+=1;entry['rows']+=inputs[0].numel()//module.in_features
            handles.append(layer.register_forward_pre_hook(hook))
    try:yield observed
    finally:
        for handle in handles:handle.remove()


def run_inference(a, context):
    native,trainer,metrics,rows,hashes,config,run,side,regions,result=context
    import torch
    torch.set_num_threads(a.cpu_threads)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    device=torch.device(a.device);require(device.type in ('cpu','cuda'),'Unsupported device')
    require(device.type!='cuda' or torch.cuda.is_available(),'Requested CUDA unavailable; no silent fallback')
    if device.type=='cuda':torch.cuda.set_device(device)
    # Single safe deserialization; no unrestricted-pickle fallback.
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    require(checkpoint.get('schema_version')==SCHEMA and checkpoint.get('contract')==run['contract'] and
        checkpoint.get('contract_sha256')==side['contract_sha256'] and checkpoint.get('provenance')==run['provenance'] and
        checkpoint.get('effective_config')==run['effective_config'] and
        checkpoint['training']['completed_optimizer_steps']==side['completed_optimizer_steps'],'Checkpoint payload/metadata differs')
    state=checkpoint['model_state_dict'];require(state and all(isinstance(v,torch.Tensor) and
        (not v.is_floating_point() or torch.isfinite(v).all().item()) for v in state.values()),'Nonfinite/non-tensor state')
    result['training_progress']=checkpoint['training'];del checkpoint
    sys.path.insert(0,str(a.source_root));os.environ['SDFORMER_SNN_BACKEND']='torch'
    from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4
    from models.STSwinNet_SNN.Spiking_submodules import PSN
    from utils.runtime_backend import configure_snn_backend
    from spikingjelly.activation_based import functional
    for obj in (MS_SpikingformerFlowNet_en4,PSN,configure_snn_backend):
        require(Path(inspect.getfile(obj)).resolve().is_relative_to(a.source_root),'Unpinned native import')
    tile=import_file('breadth_tile_executor',a.prior_root/'deployment_contract/tile_mlp.py')
    policies=import_file('breadth_policies',HERE/'policies.py')
    policies.configure_prior_root(a.prior_root)
    result['method_implementation_sha256']={'policies.py':sha(HERE/'policies.py')}
    config=copy.deepcopy(config);config['swin_transformer']['input_size']=[288,384] if a.geometry=='crop_center9' else [480,640]
    ys,xs=metrics.crop_slices(a.geometry)
    result['inference_executed']=True;result['runtime']={'torch':str(torch.__version__),'device':str(device),'cuda':torch.version.cuda,
        'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu'}
    began=time.perf_counter();data_evidence={};budget_stop=False;probed_methods=set();tp_compared=False
    sync=lambda:torch.cuda.synchronize(device) if device.type=='cuda' else None
    for case in result['cases']:
        if time.perf_counter()-began>=a.wall_minutes*60:budget_stop=True;break
        entry={'case':case,'status':'RUNNING','frames':[],'region_status':result['region_status']}
        result['arms'][case['id']]=entry;model=None;started=time.perf_counter();meter=metrics.MetricAccumulator()
        try:
            model=MS_SpikingformerFlowNet_en4(copy.deepcopy(config['model']),copy.deepcopy(config['swin_transformer']))
            model.load_state_dict(state,strict=True)
            if case['method']=='static':
                if case['static_method']=='svs':
                    entry['status']='PENDING_RELIABLE_TRAIN_FIT_SVS_ARTIFACT';continue
                methods=import_file('breadth_static_methods',HERE/'static_methods.py')
                result['method_implementation_sha256']['static_methods.py']=sha(HERE/'static_methods.py')
                names=[n for n,m in model.named_modules() if m.__class__.__name__=='MS_Spiking_Mlp']
                kwargs={k:case[k] for k in ('keep_ratio','ranks','factor_targets','seed','pe_group_width','rank_basis','bits') if k in case}
                if hasattr(methods,'configure_prior_root'):methods.configure_prior_root(a.prior_root)
                model,entry['static_report']=methods.build_static_candidate(model,case['static_method'],names,**kwargs)
            model.to(device);functional.set_step_mode(model,'m');configure_snn_backend(model,device,config,PSN)
            entry['BN_count']=apply_bn(model,case['bn'],torch);before=bn_signature(model,torch)
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            entry['parameter_count']=sum(p.numel() for p in model.parameters())
            with torch.inference_mode():
                for index,row in enumerate(rows):
                    if time.perf_counter()-began>=a.wall_minutes*60:budget_stop=True;break
                    per_started=time.perf_counter();expected={};evidence={}
                    for name,rel in native.target_files(row).items():
                        native.checked_path(a.data_root,rel)
                        expected['events' if name=='input' else name]=hashes[rel]['sha256']
                    arrays,_=metrics.load_arrays(a.data_root/'saved_flow_data',row,evidence=evidence,expected_hashes=expected)
                    if row['target_id'] in data_evidence:require(data_evidence[row['target_id']]==evidence,'Data changed between cases')
                    data_evidence[row['target_id']]=evidence
                    masks=metrics.partitions(arrays['gt'],arrays['mask'],arrays['events'],regions) if regions else {'all_valid':arrays['mask']}
                    masks={k:v[ys,xs] for k,v in masks.items()};gt=arrays['gt'][:,ys,xs]
                    chunk=prepare_input(arrays['events'][:,ys,xs],device,torch)
                    policy=None;execution=contextlib.nullcontext([]);tp_comparisons=[]
                    if case['method']=='tile':
                        policy=policies.make_policy(case['policy'],case['keep_fraction'],seed=case.get('seed',0),target_id=row['target_id'])
                        if case['policy'] in ('tp_literal','tp_factored') and index<2 and not tp_compared:
                            base_policy=policy
                            def audited_policy(layer_name,u):
                                sync();t0=time.perf_counter()
                                compare=policies.compare_tp_factorization(u)
                                sync();tp_comparisons.append({'layer_name':layer_name,'seconds':time.perf_counter()-t0,
                                    'comparison':compare,'scope':'Extra explicit source-formula check; included in this frame timing, not normal selector cost'})
                                return base_policy(layer_name,u)
                            audited_policy.records=base_policy.records;policy=audited_policy
                        execution=tile.tiled_mlp_model(model,case.get('execution','actual_skip'),policy)
                        if index==0:
                            functional.reset_net(model);dense=model(chunk)['flow'][-1]
                            all_policy=policies.make_policy('uniform',1.0,seed=0,target_id=row['target_id'])
                            with tile.tiled_mlp_model(model,'actual_skip',all_policy) as controls:
                                functional.reset_net(model);all_keep=model(chunk)['flow'][-1]
                                error=float((dense-all_keep).abs().max().item())
                                entry['all_keep_equivalence']={'target_id':row['target_id'],'max_abs_error':error,
                                    'bitwise_equal':torch.equal(dense,all_keep),'scope':'First fixed target, same model/BN FP32',
                                    'actual_MLP_rows':[record for wrap in controls for record in wrap.records]}
                            require(error<=1e-6,'All-keep implementation check failed')
                            del dense,all_keep
                    functional.reset_net(model);sync();forward_start=time.perf_counter()
                    with execution as wrappers,linear_observer(model,torch) as observed:
                        output=model(chunk)['flow'][-1];sync()
                        require(tuple(output.shape)==(1,2)+gt.shape[1:],'Unexpected final flow shape')
                        prediction=output[0].detach().cpu().numpy()
                        observed_tiles=[record for wrap in wrappers for record in wrap.records]
                    forward_seconds=time.perf_counter()-forward_start
                    frame=meter.update(row['target_id'],row['sequence'],prediction,gt,masks)
                    frame.update(forward_seconds_including_policy_and_hooks=forward_seconds,
                        regular_path_wall_seconds_including_data_and_metrics=time.perf_counter()-per_started,
                        tile_execution=observed_tiles,linear_execution=observed,
                        policy_records=compact_policy_records(policy.records) if policy else [],tp_factorization_probe=tp_comparisons)
                    if (case['method']=='tile' and case.get('execution','actual_skip')=='dense_masked' and
                            index<a.actual_pair_count and case['policy'] not in probed_methods):
                        paired=policies.make_policy(case['policy'],case['keep_fraction'],seed=case.get('seed',0),target_id=row['target_id'])
                        functional.reset_net(model);sync();pair_start=time.perf_counter()
                        with tile.tiled_mlp_model(model,'actual_skip',paired) as actual,linear_observer(model,torch) as actual_linear:
                            actual_out=model(chunk)['flow'][-1];sync()
                            actual_tiles=[r for wrapper in actual for r in wrapper.records]
                        frame['actual_skip_pair']={'scope':'First2 fixed targets of first declared case for this policy; execution check only',
                            'seconds':time.perf_counter()-pair_start,'max_abs_error':float((actual_out-output).abs().max().item()),
                            'allclose_atol1e-5_rtol1e-5':torch.allclose(actual_out,output,atol=1e-5,rtol=1e-5),
                            'finite':torch.isfinite(actual_out).all().item(),
                            'same_tile_masks':[r['keep'] for r in actual_tiles]==[r['keep'] for r in observed_tiles],
                            'tile_execution':actual_tiles,'linear_execution':actual_linear,'policy_records':compact_policy_records(paired.records),
                            'quality_measurement_uses':'dense_masked regular result; paired GPU loop time is not accelerator speed'}
                        del actual_out
                    entry['frames']=meter.frames
                    del output,chunk,prediction
                    if (index+1)%8==0:print(json.dumps({'case':case['id'],'completed':index+1,'of':len(rows)}),flush=True)
            entry['BN_buffers_unchanged']=before==bn_signature(model,torch)
            require(entry['BN_buffers_unchanged'],'BN mutated during evaluation')
            entry['status']='PASS' if len(meter.frames)==len(rows) else 'BUDGET_PARTIAL_NOT_COMPLETE_METRIC'
            if case['method']=='tile' and len(meter.frames)>=a.actual_pair_count:probed_methods.add(case['policy'])
            if case['method']=='tile' and case['policy'] in ('tp_literal','tp_factored') and len(meter.frames)>=2:tp_compared=True
        except Exception as exc:
            entry.update(status='IMPLEMENTATION_OR_CAPACITY_ERROR_NOT_METHOD_FAILURE',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        finally:
            sync();entry['wall_seconds_including_build_data_metrics']=time.perf_counter()-started
            entry['summary']=meter.summary()
            entry['max_allocated_bytes']=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
            entry['max_reserved_bytes']=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None
            result['input_file_sha256']=data_evidence
            if model is not None:functional.reset_net(model)
            del model
            if device.type=='cuda':torch.cuda.empty_cache()
            write_result(a.output,result)
    result['status']='BREADTH_BUDGET_STOP' if budget_stop else 'BREADTH_CASES_FINISHED'
    result['complete_cases']=sum(x['status']=='PASS' for x in result['arms'].values())
    result['elapsed_seconds']=time.perf_counter()-began


def write_result(path,result):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('prior-root','source-root','data-root','source-manifest','canonical','roles','archive-audit','acceptance','checkpoint','cases','output'):
        p.add_argument('--'+key,type=Path,required=True)
    for key in ('acceptance-sha256','checkpoint-sha256','cases-sha256'):p.add_argument('--'+key,required=True)
    p.add_argument('--geometry',choices=('full9','crop_center9'),default='full9')
    p.add_argument('--regions',type=Path);p.add_argument('--regions-sha256')
    p.add_argument('--device',default='cuda:0');p.add_argument('--cpu-threads',type=int,default=2)
    p.add_argument('--actual-pair-count',type=int,choices=(0,1,2),default=2,
                   help='First0..2 fixed targets of first dense_masked case per policy also execute actual_skip')
    p.add_argument('--wall-minutes',type=float,default=60);p.add_argument('--run-inference',action='store_true')
    a=p.parse_args(argv)
    if bool(a.regions)!=bool(a.regions_sha256):p.error('regions path and SHA must be paired')
    if not 1<=a.cpu_threads<=4 or not 0<a.wall_minutes<=60:p.error('CPUthreads1..4 and one bounded0..60minute call required')
    for key,v in vars(a).items():
        if isinstance(v,Path):setattr(a,key,v.resolve())
    if a.output.exists():p.error('Output exists; preserve prior evidence')
    return a


def main(argv=None):
    a=parse_args(argv);context=preflight(a);result=context[-1]
    try:
        if a.run_inference:run_inference(a,context)
    except BaseException as exc:
        result.update(status='FAILED_GLOBAL_NO_COMPLETE_CONCLUSION',error=type(exc).__name__+': '+str(exc));raise
    finally:write_result(a.output,result)
    print(json.dumps({'status':result['status'],'output':str(a.output),'complete_cases':result.get('complete_cases',0)}))


if __name__=='__main__':main()
