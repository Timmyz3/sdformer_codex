#!/usr/bin/env python3
"""Explicit recovered-state H0 collector. Default preflight, no training code.

Requires a root-prepared admission/inputs manifest after the complete R13 audit.
Imports frozen numerical helpers, never calls any old prepare/execute entry.
"""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import signal
import sys
import time
import traceback
sys.dont_write_bytecode = True
import h0_core as core
HERE = core.HERE
OWN = ('h0_core.py','run_h0.py','test_h0_cpu.py','source_pins.json','h8_manifest.json')
require,sha = core.require,core.sha


def asset(value):
    require(set(value) == {'path','sha256'}, 'Asset requires exact path/SHA fields')
    path = Path(value['path']);require(path.is_absolute() and path.is_file(), 'Existing absolute input file required')
    require(sha(path) == value['sha256'], 'Explicit source SHA differs: '+str(path))
    return path


def validate_admission(admission, condition, w_arm):
    require(admission.get('schema') == 'flow_h0_admission_v1' and admission.get('status') == 'READY_AFTER_R13_ALL_EIGHT_AUDIT',
            'Separate completed all-eight scientific audit required')
    require(admission.get('condition') == condition and admission.get('W_arm') == w_arm, 'Admission selected state differs')
    require(admission.get('audited_conditions_arms') == [[c,a] for c in ('reference','diversity') for a in ('native','FF','CC','fine_s2b2')],
            'Complete eight-arm audit coverage required')
    require(admission.get('ordinary_N_F_stable') is True, 'N/F instability must be diagnosed before H0')
    scope = admission.get('scope')
    require(scope in ('qualified_ordinary_base','action_only_screening'), 'Explicit scientific scope required')
    if scope == 'qualified_ordinary_base':
        require(w_arm in ('CC','fine_s2b2') and admission.get('selected_base_passes_required_gates') is True
                and admission.get('shared_fine_protection_passes') is True, 'Qualified base gates not affirmed')
    else:
        require(w_arm == 'FF' and admission.get('selected_base_passes_required_gates') is False,
                'Action-only diagnostic uses matched F state without pretending deployable gates passed')
    require(isinstance(admission.get('rationale'),str) and admission['rationale'], 'Audit rationale required')
    return scope


def recovered_descriptor(result, descriptor, role, condition, w_arm, common, r13):
    require(result.get('schema') == r13.SCHEMA and result.get('status') == 'ROUND13_COVERAGE_RECOVERY_COMPLETE'
            and result.get('completed_optimizer_steps') == 200 and result.get('training_performed') is True, 'Incomplete R13 recovery')
    arm = w_arm if role == 'W' else 'native' if role == 'N' else 'FF'
    binding = result['binding']
    require(binding.get('mode') == 'recover' and binding.get('condition') == condition and binding.get('arm') == arm,
            'Matched condition/arm differs')
    for key,expected in common.items():
        require(binding.get(key) == expected, 'Recovered common binding differs: '+key)
    require(binding['runner_sha256'] == core.R13_SHA and binding['execute_sha256'] ==
            'da4f460c9582dbe244218dd70f8b5928c6ede1495527d6cf355d1cbc01955541', 'Unfrozen recovery execution source')
    require(binding['recipe'] == r13.RECIPE and binding['allowed_parameter_count'] == r13.PARAMETERS, 'Recovery recipe/parameters differ')
    for key in ('frozen_state_unchanged','BN_buffers_unchanged','original_modules_restored','all_hooks_removed'):
        require(result.get(key) is True, 'Recovery closure audit missing: '+key)
    require(result.get('weights_sha256') == descriptor['weights']['sha256'] and result.get('weights_file') == Path(descriptor['weights']['path']).name,
            'Named recovered artifact differs')
    return {'binding':binding,'initial_state_sha256':result['initial_model_state_sha256'],
            'final_state_sha256':result['final_state_sha256'],'trainable_names':result['allowed_parameter_names']}


def validate_completed_analysis(analysis, inputs, admission, common):
    """Bind admission to the frozen all-eight analysis, not a self-asserted flag."""
    require(analysis.get('schema') == 'flow_round13_coverage_quality_analysis_v1' and analysis.get('script_sha256') ==
            'b1997b55deea309088388a011b83272bafe438bedc43db5ee4440655bc5034ad', 'Completed frozen R13 analysis required')
    require(analysis.get('analysis_protocol_sha256') == '5ec51175fbf18b9d2f4fab329247b011700f4cf109b50bf73e905708698c2d7b'
            and analysis.get('R12_V2_source_sha256') == '525ba2821d1e671b581f63e80c4bce02ef3e524c3e6af7d143a723914b001cd8', 'Analysis protocol/math provenance differs')
    require((analysis.get('state_layout_count'),analysis.get('evaluation_endpoint_count'),analysis.get('region_count'),
             analysis.get('full_forward_count')) == (40,320,16,1964), 'Incomplete all-eight analysis counts')
    require(analysis.get('initial_state_all_eight_exact') is True and
            analysis.get('initial_same_layout_prediction_SHA_and_metrics_cross_condition_exact') is True, 'Initial state/prediction audit missing')
    keys={c+'/'+a for c in ('reference','diversity') for a in ('native','FF','CC','fine_s2b2')}
    require(set(analysis.get('inputs',{})) == set(analysis.get('prediction_artifact_audits',{})) == keys, 'All-eight artifact coverage required')
    require(set(inputs.get('all_eight_results',{})) == keys, 'Explicit current paths for all eight audited results required')
    require(all(inputs['all_eight_results'][k]['sha256'] == analysis['inputs'][k]['sha256'] for k in keys), 'Relocated result bytes differ from frozen analysis')
    require(all(item == {'prediction_files':16,'prediction_payloads':40} for item in analysis['prediction_artifact_audits'].values()), 'Missing retained prediction audit')
    require(all(analysis['common_contract'].get(k) == value for k,value in common.items()), 'All-eight analysis common state differs')
    condition=inputs['condition'];w_arm=inputs['W_arm']
    require(analysis['ordinary_reference_normality'].get(condition) is True, 'Audited ordinary N/F are not stable')
    for role,item in inputs['states'].items():
        arm=w_arm if role == 'W' else 'native' if role == 'N' else 'FF'
        require(item['result'] == inputs['all_eight_results'][condition+'/'+arm], 'Chosen recovered state does not belong to audited all-eight result set')
    if admission['scope'] == 'qualified_ordinary_base':
        gate=analysis['candidate_gates'][condition][w_arm]
        require(gate['ordinary_base_quality_pass'] is True and all(v['pass'] is True for v in gate['shared_fine_preserved'].values()),
                'Qualified admission disagrees with frozen quality gates')
    return {'complete_eight_sources_bound':True,'ordinary_N_F_stability_verified':True,
            'source_analysis_schema':analysis['schema'],'scope':admission['scope']}


def validate_independent_execution(audit, analysis, inputs, common):
    require(audit.get('schema') == 'flow_round13_independent_execution_audit_v1' and audit.get('status') == 'PASS' and
            audit.get('auditor_sha256') == 'dd57f9f315e1c5f38961208496a37f46db577c47b8507029a55b5bc216d8c8e8', 'Frozen independent execution PASS required')
    scope=audit['scope']
    require(set(scope['conditions']) == {'reference','diversity'} and set(scope['modes']) == {'smoke','recover'} and scope['runs'] == 16,
            'Both conditions and all16 smoke/recovery runs required')
    require((audit.get('weights_count'),audit.get('prediction_count'),audit.get('archive_count'),audit.get('full_forwards'),
             audit.get('optimizer_updates')) == (8,320,128,2048,1640), 'Independent artifact/update counts differ')
    require(audit.get('paired_initial_predictions_equal') is True and audit.get('common_initial_model_sha256') == analysis['common_contract']['initial_state_sha256']
            and audit.get('original_checkpoint_sha256') == common['checkpoint_sha256'] and audit.get('new_BN_actual_sha256') == common['BN_buffers_sha256'],
            'Independent initial/BN/checkpoint state differs')
    rows={(r['condition'],r['mode'],r['arm']):r for r in audit['results']}
    expected={(c,m,a) for c in ('reference','diversity') for m in ('smoke','recover') for a in ('native','FF','CC','fine_s2b2')}
    require(len(audit['results']) == 16 and set(rows) == expected, 'Independent per-run audit coverage differs')
    for condition,mode,arm in expected:
        row=rows[(condition,mode,arm)]
        if mode == 'recover':
            require(row['result_sha256'] == analysis['inputs'][condition+'/'+arm]['sha256'], 'Independent result differs from quality analysis')
            weights=row['weights']
            require(all(weights.get(k) is True for k in ('final_state_independently_recomputed','frozen_complement_independently_recomputed',
                                                         'initial_state_reconstructed','BN72_actual_values_match')), 'Independent weight audit incomplete')
    for role,item in inputs['states'].items():
        arm=inputs['W_arm'] if role == 'W' else 'native' if role == 'N' else 'FF'
        row=rows[(inputs['condition'],'recover',arm)]
        require(item['weights']['sha256'] == row['weights']['sha256'], 'Chosen weights differ from independent audit')
    return {'complete16_independent_runs_bound':True,'selected_weights_match_independent_actual_state_audit':True,
            'path_identity':'condition/mode/arm plus actual bytes SHA; local and remote paths may differ'}


def prepare(a,runtime):
    r13,collector,probe,parent,r9,r8 = runtime
    require(0 < a.wall_minutes <= 30 and not a.output_dir.exists(), 'Fresh output and <=30minute stage required')
    require(sha(a.inputs) == a.inputs_sha256, 'Explicit H0 input manifest SHA differs')
    inputs = json.loads(a.inputs.read_text());require(inputs.get('schema') == 'flow_h0_inputs_v1', 'Unknown H0 input manifest')
    files = {k:asset(inputs[k]) for k in ('admission','runtime_manifest','membership','bn_buffers','bn_receipt','h8_manifest')}
    require(inputs['membership']['sha256'] == core.MEMBER_SHA, 'H0 R13 members differ')
    require(inputs['h8_manifest']['sha256'] == sha(HERE/'h8_manifest.json'), 'H8 frozen generated members differ')
    condition,w_arm = inputs['condition'],inputs['W_arm']
    require(condition in ('reference','diversity'), 'Unknown recovered condition')
    admission = json.loads(files['admission'].read_text());scope_name = validate_admission(admission,condition,w_arm)
    audit_path = asset(admission['all_eight_audit_receipt'])
    independent_path = asset(admission['independent_execution_audit'])
    require(audit_path != files['admission'], 'Admission cannot attest to itself as completed audit')
    rt = json.loads(files['runtime_manifest'].read_text());manifest = json.loads(files['membership'].read_text())
    require(rt.get('schema') == 'flow_round13_runtime_v1', 'Frozen R13 runtime required')
    h8 = json.loads(files['h8_manifest'].read_text())
    require(h8 == core.h8_manifest(manifest,core.MEMBER_SHA), 'Reconstructed H8 identities/order differ')
    for path,digest in rt['source_pins'].items():require(sha(path) == digest,'Runtime source differs: '+path)
    sys.path[:0] = [rt['breadth_root'],rt['motion_root']]
    import evaluate_breadth as base
    import run_bn_scope as scope
    import run_same_shape_reference as loader
    expected = {'evaluate_breadth.py':'1cacbaeea4d9eaf19a304c3faaa13f5278f0212b6cdc92210c4619447d571628',
                'run_bn_scope.py':'2b5c61e3abedcc02ba09adadb432dfc0ff2cd2fcd2b7b3e3a2475c63d7a11840',
                'run_same_shape_reference.py':'6b6c2f93d24bac9fa1dd3a99fde0db05aa180d05d897482d13a028208340774b'}
    for module in (base,scope,loader):require(sha(module.__file__) == expected[Path(module.__file__).name], 'Native helper SHA differs')
    args = list(rt['base_args'])
    require(not any(x.startswith(('--bn-','--svs-','--regions')) or x == '--run-inference' for x in args), 'Legacy BN/region/inference arguments forbidden')
    require(args.count('--output') == 1, 'One original base output required')
    args[args.index('--output')+1] = str(a.output_dir/'results.json')
    b = base.parse_args(args);b.device='cuda:0';b.wall_minutes=a.wall_minutes
    require(b.geometry == 'full9' and b.checkpoint_sha256 == r13.CHECKPOINT_SHA and not b.run_inference, 'Original native constructor source required')
    context = list(base.preflight(b))
    native,trainer,metrics,dev32,hashes,config,saved,side,_,plan = context
    require(side['completed_optimizer_steps'] == 135930, 'Wrong original architecture source')
    canonical = metrics.canonical_rows(b.canonical);roles=metrics.auxiliary_roles(b.roles,canonical)
    require(manifest['source_sha256']['canonical'] == sha(b.canonical) and manifest['source_sha256']['roles'] == sha(b.roles), 'Canonical/role sources differ')
    membership_audit = r13.validate_members(manifest,canonical,roles)
    region_path = Path(rt['region_path']);require(sha(region_path) == rt['region_sha256'], 'Training-only region SHA differs')
    regions = json.loads(region_path.read_text());metrics.validate_region_contract(regions,canonical,roles,require_complete_fit=False)
    require(regions['fit_target_ids'] == [r['target_id'] for r in manifest['train_members_diversity']], 'Only D200 region fit is legal')
    by_id = {r['target_id']:r for r in canonical};evidence=regions.get('data_evidence',[])
    require([r['target_id'] for r in evidence] == regions['fit_target_ids'], 'Region fit evidence differs')
    for item in evidence:
        paths=native.target_files(by_id[item['target_id']])
        require(all(item.get(k+'_sha256') == hashes[paths[k]]['sha256'] for k in ('gt','mask')), 'Region source hashes differ')
    context[8]=regions
    r13.validate_bn_receipt(json.loads(files['bn_receipt'].read_text()),files['bn_buffers'],manifest,core.MEMBER_SHA)
    timing = collector.load_pinned(Path(rt['motion_root'])/'collect_coarse_cache.py',collector.TIMESTAMP_SHA,'_h0_timestamp_binding')
    acceptance=json.loads(b.acceptance.read_text());inventory_path=b.acceptance.parent/'native_deployment_manifest.csv'
    require(sha(inventory_path) == acceptance['deployment_csv_sha256'], 'Data deployment inventory differs')
    with inventory_path.open(newline='') as stream:inventory={row['relative_path']:row for row in csv.DictReader(stream)}
    data_evidence=[]
    for row in h8['members']:
        require(row == by_id[row['target_id']] and roles[row['target_id']] == 'gate_fit', 'H8 canonical/role mismatch')
        files_for_row={}
        for key,relative in native.target_files(row).items():
            require(relative in hashes and sha(native.checked_path(b.data_root,relative)) == hashes[relative]['sha256'], 'H8 accepted input source differs')
            files_for_row[key]=hashes[relative]['sha256']
        data_evidence.append({'target_id':row['target_id'],'files':files_for_row,
                              'timestamps':timing.timestamp_binding(b.data_root,row,native.target_files(row),hashes,inventory,acceptance)})
    common={'checkpoint_sha256':r13.CHECKPOINT_SHA,'membership_sha256':core.MEMBER_SHA,
            'runtime_manifest_sha256':inputs['runtime_manifest']['sha256'],'BN_buffers_sha256':inputs['bn_buffers']['sha256'],
            'BN_receipt_sha256':inputs['bn_receipt']['sha256']}
    require(set(inputs['states']) == {'W','N','F'}, 'Exactly named W, matched N and F states required')
    analysis=json.loads(audit_path.read_text())
    admission_audit=validate_completed_analysis(analysis,inputs,admission,common)
    execution_audit=validate_independent_execution(json.loads(independent_path.read_text()),analysis,inputs,common)
    # SHA-check all eight bound raw records; selected weights are separately loaded strictly.
    for item in inputs['all_eight_results'].values():asset(item)
    descriptors={}
    for role,item in inputs['states'].items():
        result_path=asset(item['result']);asset(item['weights'])
        require(Path(item['weights']['path']).parent == result_path.parent, 'Weights must accompany bound R13 result')
        descriptors[role]=recovered_descriptor(json.loads(result_path.read_text()),item,role,condition,w_arm,common,r13)
    require(len({d['initial_state_sha256'] for d in descriptors.values()}) == 1, 'Recovered arms initial states differ')
    require(next(iter(descriptors.values()))['initial_state_sha256'] == analysis['common_contract']['initial_state_sha256'], 'Recovered initial state differs from all-eight audit')
    cpu_sha=parent.checked_cpu_receipt(a.cpu_checks,[HERE/n for n in OWN])
    binding={'schema':'flow_h0_binding_v1','inputs_sha256':a.inputs_sha256,'cpu_checks_sha256':cpu_sha,
             'source_sha256':{name:sha(HERE/name) for name in OWN},'common':common,'H8_ids':[r['target_id'] for r in h8['members']],
             'condition':condition,'W_arm':w_arm,'scientific_scope':scope_name,'all_eight_audit_receipt':admission['all_eight_audit_receipt'],
             'completed_analysis_admission_audit':admission_audit,
             'independent_execution_audit':admission['independent_execution_audit'],'independent_execution_admission_audit':execution_audit,
             'region_sha256':rt['region_sha256'],'region_fit_used_for_model_update':False,'forward_plan':core.forward_plan(h8['members']),
             'total_full_forwards':69,'actual_anchors':64,'dual_forwards':5,'training_forwards':0,'new_BN_calibration_forwards':0,
             'GT_used_for_actions':False,'wall_minutes':a.wall_minutes,'GPU_lock':str(parent.GPU_LOCK),
             'flow_specific_increment_claim':False,'hardware_benefit_claim':False}
    return {'inputs':inputs,'files':files,'manifest':manifest,'rows':h8['members'],'descriptors':descriptors,'binding':binding,
            'b':b,'context':context,'base':base,'scope':scope,'loader':loader,'data_evidence':data_evidence,'membership_audit':membership_audit}


def execute(a,p,result,runtime):
    import numpy as np
    import torch
    r13,collector,probe,parent,r9,r8=runtime
    require(not torch.cuda.is_initialized(), 'CUDA initialized before shared lock')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261016)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.manual_seed_all(20261016)
    started=time.monotonic()
    def deadline():require(time.monotonic()-started < a.wall_minutes*60, 'H0 bounded stage elapsed')
    b,context,base,scope,loader=p['b'],p['context'],p['base'],p['scope'],p['loader']
    native,trainer,metrics,dev32,hashes,config,saved,side,regions,old_plan=context
    # Original checkpoint initializes architecture only; no original-state inference.
    model,functional=loader.load_model(b,context,base,torch,device)
    bn=r13.apply_fresh_bn(model,p['files']['bn_buffers'],p['files']['bn_receipt'],p['manifest'],core.MEMBER_SHA,scope,base,torch)
    common_buffers={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if n.endswith(core.BUFFERS)}
    require(len(common_buffers) == 72 and bn['BN_count'] == 78 and bn['MLP_BN_count'] == 24, 'Exact R13 BN scope required')
    bn_signature=base.bn_signature(model,torch);targets,shapes=parent.target_layers(model);merge=parent.load_frozen_merge()
    require(tuple(shapes[core.TARGET]) == (10,1,30,40,384), 'Production target geometry differs')
    cache={};frames={};predictions={};meters={key:metrics.MetricAccumulator() for key in core.ANCHORS}
    for row in p['rows']:
        deadline();expected={'events' if k=='input' else k:hashes[rel]['sha256'] for k,rel in native.target_files(row).items()}
        evidence={};arrays,_=metrics.load_arrays(b.data_root/'saved_flow_data',row,evidence=evidence,expected_hashes=expected)
        cache[row['target_id']]=arrays
        frames[row['target_id']]={'target_id':row['target_id'],'sequence':row['sequence'],'input_evidence':evidence,'arms':{}}
        predictions[row['target_id']]={}
    rows={r['target_id']:r for r in p['rows']};state_info={};current=[None];state_digest=[None]
    result.update(status='H0_RUNNING',full_forward_count=0,training_performed=False,new_BN_calibration_performed=False,
                  runtime={'torch':str(torch.__version__),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
                           'device_name':torch.cuda.get_device_name(device),'TF32':False,'AMP':False,'precision':'float32'},BN_scope=bn)
    def closure():
        if current[0] is None:return
        require(r13.state_digest(model) == state_digest[0], 'Recovered weights mutated during H0')
        require(base.bn_signature(model,torch) == bn_signature, 'Common/private or current-stat BN mutated')
        require(all(model.get_submodule(n) is m for n,m in targets.items()), 'Original module not restored')
        require(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()), 'Hooks remain')
    def switch_state(role):
        deadline();closure()
        item=p['inputs']['states'][role];path=asset(item['weights'])
        payload=torch.load(path,map_location='cpu',weights_only=True)
        state_info[role]=core.install_recovered(model,payload,p['descriptors'][role],common_buffers,torch)
        del payload
        state_digest[0]=p['descriptors'][role]['final_state_sha256'];current[0]=role
        require(base.bn_signature(model,torch) == bn_signature, 'Recovered state changed BN signature')
        functional.reset_net(model)
    def infer(item):
        deadline();require(current[0] == item['state'], 'Wrong recovered state for plan item')
        chunk=base.prepare_input(cache[item['target_id']]['events'],device,torch)
        output=core.forward(runtime,model,functional,targets,shapes,chunk,item['anchor'],merge,torch,item['mode'],deadline)
        del chunk
        require(len(output[0]) == 4 and all(tuple(f.shape) == (1,2,480,640) and f.dtype == torch.float32 for f in output[0]), 'Four native full-resolution FP32 heads required')
        result['full_forward_count']+=1
        require(result['full_forward_count'] == item['forward_id'], 'Count/order mismatch')
        result['last_completed_forward']=dict(item)
        parent.write(a.output_dir/'results.json',result)
        return output
    def save_actual(item,output):
        row=rows[item['target_id']];arrays=cache[item['target_id']]
        # GT first enters evaluation here; static actions consume identities/geometry only.
        masks=metrics.partitions(arrays['gt'],arrays['mask'],arrays['events'],regions)
        require(len(masks) == 16, 'All16 fixed valid-GT regions required')
        prediction=output[0][-1][0].numpy().copy();predictions[row['target_id']][item['anchor']]=prediction
        frames[row['target_id']]['arms'][item['anchor']]={
            'metrics':meters[item['anchor']].update(row['target_id'],row['sequence'],prediction,arrays['gt'],masks),
            'prediction_sha256':merge.tensor_sha(output[0][-1][0]),'execution':output[1],'state':item['state']}
    records,pairs=core.drive_plan(p['binding']['forward_plan'],switch_state,infer,save_actual,lambda x,y:core.check_pair(x,y,torch))
    closure()
    require(result['full_forward_count'] == 69 and set(state_info) == {'W','N','F'}, 'Incomplete H0')
    for row in p['rows']:
        identity=row['target_id'];require(set(predictions[identity]) == set(core.ANCHORS), 'Missing same-frame anchors')
        path=a.output_dir/('H0_'+identity+'.npz');np.savez_compressed(path,**predictions[identity])
        frames[identity].update(prediction_file=path.name,prediction_file_sha256=sha(path))
    result.update(status='H0_69_FORWARD_COMPLETE',completed_targets=8,forward_records=records,first_frame_dual_audits=pairs,
                  recovered_state_loads=state_info,frames=[frames[r['target_id']] for r in p['rows']],
                  summaries={k:m.summary() for k,m in meters.items()},all_recovered_states_unchanged=True,BN_buffers_unchanged=True,
                  original_modules_restored=True,all_hooks_removed=True,wall_seconds=time.monotonic()-started,
                  max_allocated_bytes=torch.cuda.max_memory_allocated(device),max_reserved_bytes=torch.cuda.max_memory_reserved(device),
                  flow_specific_increment_claim=False,hardware_benefit_claim=False,
                  interpretation='Training-side action-space diagnostic only; every region retained. No H1, predictor, or quality gate decision computed here.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--inputs-sha256',required=True)
    parser.add_argument('--cpu-checks',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--wall-minutes',type=float,default=30);parser.add_argument('--run-inference',action='store_true')
    a=parser.parse_args();runtime=core.load_frozen();parent=runtime[3];p=prepare(a,runtime)
    result={'schema':'flow_h0_action_screen_v1','status':'H0_PREFLIGHT_COMPLETE','binding':p['binding'],
            'data_evidence':p['data_evidence'],'membership_audit':p['membership_audit'],'inference_executed':False}
    a.output_dir.mkdir(parents=True,exist_ok=False)
    if not a.run_inference:parent.write(a.output_dir/'results.json',result);return
    parent.GPU_LOCK.parent.mkdir(parents=True,exist_ok=True)
    with parent.GPU_LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=signal.getsignal(signal.SIGALRM)
        def timeout(signum,frame):raise TimeoutError('H0 <=30minute fixed stage cap exhausted')
        signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,a.wall_minutes*60)
        try:
            result['inference_executed']=True;execute(a,p,result,runtime)
        except BaseException as exc:
            result.update(status='INCOMPLETE_NO_SCIENTIFIC_CONCLUSION',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous);parent.write(a.output_dir/'results.json',result)


if __name__ == '__main__':main()
