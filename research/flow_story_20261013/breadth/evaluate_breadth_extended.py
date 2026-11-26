#!/usr/bin/env python3
"""Explicit BN-buffer and static-score extensions; frozen first-pool code unchanged.

The inference function below is a reviewable copy of the SHA-pinned original;
only the strict-load -> BN-buffer -> static-candidate construction section is
changed. All data/metric/selection/execution checks retain their original order.
"""
import evaluate_breadth as base
from evaluate_breadth import *
BASE_SHA='1cacbaeea4d9eaf19a304c3faaa13f5278f0212b6cdc92210c4619447d571628'
SCHEDULE_SHA='2f2b374eddf68d13dce6d6aead93218f9568f3c5143a59a25cd779a806289659'
BN_COLLECTOR_SHA='a364c6a08816a7df1c1eee4ce249f44ea9cf0631c172053c3e40eaf2df74b00f'
STATIC_SHA='5b1938b2815a8201066db01c9a1d3920aa523116a30b22a6f4883fc86c477aab'
SVS_FIT_SHA='9a908ea2da02812c771ff42e9a04f96e14388f27526a73660e4e6abb489e4f6f'
SVS_COLLECTOR_SHA='49b0668626d4be21f861efc587a397e13b17a17bf4fe0d382478656560fac831'
STATIC_EXTENSION_SHA='cf23efdfa144b3629fa5a5bc90bfd4aa234832fcdbc5bb973be4b07f8ba97ba9'
require(sha(base.__file__)==BASE_SHA,'Frozen breadth base changed')


def validate_bn_artifact(artifact,a,context):
    native,trainer,metrics,eval_rows,hashes,config,run,side,regions,plan=context
    schedule=read_pin(a.bn_schedule,SCHEDULE_SHA)
    ids=schedule['stages']['initial']['target_ids']
    records=metrics.canonical_rows(a.canonical);roles=metrics.auxiliary_roles(a.roles,records)
    require(len(ids)==len(set(ids))==176 and all(roles.get(t)=='bn_calibration' for t in ids) and
            not set(ids)&{r['target_id'] for r in eval_rows},'BN fixed176 roles/disjointness differs')
    require(artifact.get('schema')=='flow_breadth_bn_calibration_v1' and artifact.get('status')=='COMPLETE' and
            artifact.get('method')=='cma_native_batch_stats_reset' and artifact.get('non_bn_state_unchanged') is True and
            artifact.get('fit_uses_final_valid') is False and artifact.get('optimizer_steps')==0 and artifact.get('GPU_executed') is True,
            'Incomplete/foreign BN artifact')
    require(artifact.get('checkpoint_sha256')==a.checkpoint_sha256 and
            artifact.get('data_contract_sha256')==digest(config['data_contract']) and
            artifact.get('source_sha256')==plan['source_sha256'] and artifact.get('geometry')==a.geometry,
            'BN checkpoint/data/source/geometry binding differs')
    require(artifact.get('script_sha256')==BN_COLLECTOR_SHA and artifact.get('evaluator_sha256')==BASE_SHA and
            artifact.get('schedule_sha256')==SCHEDULE_SHA and artifact.get('fit_stage')=='initial' and
            artifact.get('fit_role')=='bn_calibration' and artifact.get('fit_target_ids')==ids and
            artifact.get('completed_ids')==ids and artifact.get('fit_target_ids_sha256')==digest(ids),
            'BN source/schedule/completion binding differs')
    before=artifact.get('source_non_bn_state_sha256')
    require(native.valid_sha(before) and before==artifact.get('after_non_bn_state_sha256'),'BN non-buffer state proof differs')
    evidence=artifact.get('input_evidence',[]);require([r['target_id'] for r in evidence]==ids,'BN input evidence incomplete')
    by_id={r['target_id']:r for r in records}
    for item in evidence:
        rel=native.target_files(by_id[item['target_id']])['input']
        require(item.get('relative')==rel and item.get('sha256')==hashes[rel]['sha256'],'BN input differs from native inventory')
    require(isinstance(artifact.get('buffers'),dict) and artifact['buffers'],'Missing BN buffers')
    for name,fields in artifact['buffers'].items():
        require(set(fields)=={'running_mean','running_var','num_batches_tracked'} and
                type(fields['num_batches_tracked']) is int and fields['num_batches_tracked']==176,'BN buffer fields/count differ')
        mu=np.asarray(fields['running_mean']);var=np.asarray(fields['running_var'])
        require(mu.ndim==var.ndim==1 and mu.shape==var.shape and mu.size>0 and
                np.isfinite(mu).all() and np.isfinite(var).all() and (var>=0).all(),'BN shape/finite/nonnegative variance fails')
    return artifact


def non_bn_digest(model,names):
    excluded={name+'.'+field for name in names for field in ('running_mean','running_var','num_batches_tracked')}
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        if name in excluded:continue
        value=tensor.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.numpy().tobytes())
    return h.hexdigest()


def apply_bn_buffers(model,artifact,torch):
    layers={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)}
    require(set(layers)==set(artifact['buffers']),'BN buffer names differ from dense model')
    before=non_bn_digest(model,layers)
    require(before==artifact['source_non_bn_state_sha256']==artifact['after_non_bn_state_sha256'],
            'BN artifact was collected from different non-buffer model state')
    prepared={}
    for name,layer in layers.items():
        fields=artifact['buffers'][name]
        require(layer.track_running_stats and layer.running_mean is not None and layer.running_var is not None and
                layer.num_batches_tracked is not None,'BN model must have original tracked buffers')
        mu=torch.as_tensor(fields['running_mean'],dtype=layer.running_mean.dtype,device=layer.running_mean.device)
        var=torch.as_tensor(fields['running_var'],dtype=layer.running_var.dtype,device=layer.running_var.device)
        require(mu.shape==layer.running_mean.shape and var.shape==layer.running_var.shape and
                torch.isfinite(mu).all().item() and torch.isfinite(var).all().item() and not (var<0).any().item() and
                type(fields['num_batches_tracked']) is int and fields['num_batches_tracked']==176,'BN tensor shape/value/count differs')
        prepared[name]=(mu,var)
    with torch.no_grad():
        for name,layer in layers.items():
            mu,var=prepared[name];layer.running_mean.copy_(mu);layer.running_var.copy_(var);layer.num_batches_tracked.fill_(176)
    after=non_bn_digest(model,layers);require(after==before,'Non-BN state changed while loading buffers')
    return {'BN_count':len(layers),'count_per_BN':176,'non_bn_state_unchanged':True,'non_bn_state_sha256':after,
            'order':'Dense checkpoint -> common dense BN artifact -> optional static compression; no candidate-specific recalibration'}


def build_static_extended(model,case,a,context,torch):
    methods=import_file('breadth_static_methods',HERE/'static_methods.py')
    require(sha(HERE/'static_methods.py')==STATIC_SHA,'Static module changed');methods.configure_prior_root(a.prior_root)
    paths=[n for n,m in model.named_modules() if m.__class__.__name__=='MS_Spiking_Mlp']
    name=case['static_method']
    if name in ('fpgm','bn1_abs_gamma'):
        require(a.static_extension_sha256 and sha(HERE/'static_score_extensions.py')==a.static_extension_sha256,
                'Explicit static-score extension SHA required')
        extension=import_file('breadth_static_extensions',HERE/'static_score_extensions.py');extension.configure_prior_root(a.prior_root)
        kwargs={k:case[k] for k in ('keep_ratio','pe_group_width','distance_row_chunk') if k in case}
        return extension.build_static_score_candidate(model,name,paths,**kwargs)
    kwargs={k:case[k] for k in ('keep_ratio','ranks','factor_targets','seed','pe_group_width','rank_basis','bits') if k in case}
    if name=='svs':
        require(a.svs_payload is not None,'SVS artifact absent')
        native,trainer,metrics,rows,hashes,config,run,side,regions,plan=context
        sample=read_pin(HERE/'svs_gate_fit2.json',SVS_FIT_SHA)
        ids=sample['target_ids'];require(a.svs_payload.get('fit_ids')==ids and len(ids)==2,'SVS fixed fit IDs differ')
        all_rows=metrics.canonical_rows(a.canonical);roles=metrics.auxiliary_roles(a.roles,all_rows)
        require(all(roles.get(t)=='gate_fit' for t in ids) and not set(ids)&{r['target_id'] for r in rows},'SVS fit role/leakage differs')
        apply_bn(model,case['bn'],torch)
        bn_contract={'policy':case['bn'],'eval_dropout_off':True,'original_dense_bn_signature':bn_signature(model,torch),
                     'recalibration_performed':a.bn_payload is not None and case['bn']!='P_paper'}
        geometry={'name':a.geometry,'input_hw':[288,384] if a.geometry=='crop_center9' else [480,640],
                  'window':[2,9,9],'T':10,'B':1,'precision':'FP32 TF32off noautocast','input_preprocess':'pinned evaluate_breadth.prepare_input'}
        expected={'checkpoint_sha256':a.checkpoint_sha256,'data_contract_sha256':digest(config['data_contract']),
                  'source_sha256':digest(plan['source_sha256']),'bn_policy_sha256':digest(bn_contract),
                  'geometry_sha256':digest(geometry),'fit_ids_sha256':digest(ids),'split':'train'}
        ev=a.svs_payload.get('evidence',{})
        require(ev.get('evaluator_sha256')==BASE_SHA and ev.get('static_methods_sha256')==STATIC_SHA and
                ev.get('collector_sha256')==a.svs_collector_sha256 and
                ev.get('CPU_only_SVD') is True and ev.get('cuda_initialized') is False and ev.get('sample_count')==2 and
                ev.get('source_site')=='sn2 binary output before drop2/fc2' and
                ev.get('score_input')=='lossless uint8 counts; float64/10; no spatial approximation',
                'SVS source/formula/complete-T evidence differs')
        kwargs.update(svs_scores=a.svs_payload,svs_expected_provenance=expected)
    return methods.build_static_candidate(model,name,paths,**kwargs)

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
            if a.bn_payload is not None and case['bn'] != 'P_paper':
                entry['BN_buffer_load']=apply_bn_buffers(model,a.bn_payload,torch)
            if case['method']=='static':
                if case['static_method']=='svs' and a.svs_payload is None:
                    entry['status']='PENDING_RELIABLE_TRAIN_FIT_SVS_ARTIFACT';continue
                model,entry['static_report']=build_static_extended(model,case,a,context,torch)
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


def parse_extended(argv=None):
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--bn-artifact',type=Path);p.add_argument('--bn-artifact-sha256');p.add_argument('--bn-schedule',type=Path)
    p.add_argument('--svs-artifact',type=Path);p.add_argument('--svs-artifact-sha256');p.add_argument('--svs-collector-sha256')
    p.add_argument('--static-extension-sha256',default=STATIC_EXTENSION_SHA)
    extra,rest=p.parse_known_args(argv);a=base.parse_args(rest)
    require(bool(extra.bn_artifact)==bool(extra.bn_artifact_sha256)==bool(extra.bn_schedule),'BN artifact/SHA/schedule must be supplied together')
    require(bool(extra.svs_artifact)==bool(extra.svs_artifact_sha256)==bool(extra.svs_collector_sha256),'SVS artifact/SHA/collector SHA must be supplied together')
    for key,value in vars(extra).items():setattr(a,key,value.resolve() if isinstance(value,Path) else value)
    return a


def extended_preflight(a):
    context=base.preflight(a);plan=context[-1]
    a.bn_payload=read_pin(a.bn_artifact,a.bn_artifact_sha256) if a.bn_artifact else None
    a.svs_payload=read_pin(a.svs_artifact,a.svs_artifact_sha256) if a.svs_artifact else None
    plan['extended_entry_sha256']=sha(__file__);plan['frozen_base_sha256']=BASE_SHA
    plan['extended_implementation_sha256']={'static_methods.py':STATIC_SHA}
    if any(c['method']=='static' and c['static_method'] in ('fpgm','bn1_abs_gamma') for c in plan['cases']):
        require(a.static_extension_sha256==STATIC_EXTENSION_SHA and sha(HERE/'static_score_extensions.py')==STATIC_EXTENSION_SHA,
                'Unreviewed static-score extension')
        plan['extended_implementation_sha256']['static_score_extensions.py']=STATIC_EXTENSION_SHA
    if a.bn_payload is not None:
        validate_bn_artifact(a.bn_payload,a,context)
        plan['BN_artifact']={'path':str(a.bn_artifact),'sha256':a.bn_artifact_sha256,
            'metadata':{k:v for k,v in a.bn_payload.items() if k not in ('buffers','input_evidence')},
            'buffers_digest':digest(a.bn_payload['buffers']),'buffer_names':sorted(a.bn_payload['buffers']),
            'input_evidence_digest':digest(a.bn_payload['input_evidence'])}
        for bn in ('D','P_code'):plan['BN_labels'][bn]='Fixed176 input-only dense-model CMA artifact; frozen during evaluation. Common calibration precedes compression, no candidate-specific recovery.'
    if a.svs_payload is not None:
        require(a.svs_collector_sha256==SVS_COLLECTOR_SHA,'Reviewed SVS collector SHA required')
        require(a.svs_payload.get('schema')=='flow_breadth_svs_replay_v1','Unknown SVS replay schema')
        plan['SVS_artifact']={'path':str(a.svs_artifact),'sha256':a.svs_artifact_sha256,
            'provenance':a.svs_payload.get('provenance'),'evidence':a.svs_payload.get('evidence'),
            'fit_ids':a.svs_payload.get('fit_ids'),'current_BN_signature_binding':'checked on dense model before narrowing'}
    return context


def main(argv=None):
    a=parse_extended(argv);context=extended_preflight(a);result=context[-1]
    try:
        if a.run_inference:run_inference(a,context)
    except BaseException as exc:
        result.update(status='FAILED_GLOBAL_NO_COMPLETE_CONCLUSION',error=type(exc).__name__+': '+str(exc));raise
    finally:write_result(a.output,result)
    print(json.dumps({'status':result['status'],'output':str(a.output),'complete_cases':result.get('complete_cases',0)}))


if __name__=='__main__':main()
