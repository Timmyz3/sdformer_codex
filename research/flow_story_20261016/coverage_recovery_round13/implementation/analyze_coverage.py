#!/usr/bin/env python3
"""Analyze all eight completed R13 arms; pinned R12 V2 arithmetic and gates."""
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
R12_V2=HERE.parents[1]/'ordinary_recovery_round12/analyze_recovery_v2.py'
R12_V2_SHA='525ba2821d1e671b581f63e80c4bce02ef3e524c3e6af7d143a723914b001cd8'
PROTOCOL=HERE.parent/'ANALYSIS_PROTOCOL.md'
PROTOCOL_SHA='5ec51175fbf18b9d2f4fab329247b011700f4cf109b50bf73e905708698c2d7b'
CONDITIONS=('reference','diversity')
CHECKPOINT_SHA='d5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30'
COMMON_FIELDS=('runner_sha256','execute_sha256','wrapper_sha256','runner_checks_sha256','source_pins_sha256','checkpoint_sha256',
               'membership_sha256','runtime_manifest_sha256','BN_buffers_sha256','BN_receipt_sha256','recipe','loss_sha256',
               'allowed_parameter_count','region_sha256','membership_audit')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


require(sha(R12_V2)==R12_V2_SHA,'Frozen R12 V2 analysis differs')
spec=importlib.util.spec_from_file_location('_r13_frozen_v2_quality',R12_V2)
V2=importlib.util.module_from_spec(spec);spec.loader.exec_module(V2)
ARMS=V2.ARMS;REGIONS=V2.REGIONS
aggregate=V2.aggregate;compare=V2.compare;gates=V2.gates


def read_pin(item):
    p=Path(item['path']);require(sha(p)==item['sha256'],'Input bytes differ: '+str(p))
    return p


def validate_input_protocol(inputs):
    """Identity/source checks only; this runs before reading eight result files."""
    require(inputs.get('schema')=='flow_round13_analysis_inputs_v1','Unknown analysis input schema')
    require(sha(PROTOCOL)==PROTOCOL_SHA,'R13 preregistered analysis protocol differs')
    paths={k:read_pin(inputs[k]) for k in ('membership','runtime_manifest','bn_receipt','bn_buffers','regions','canonical','roles')}
    m=json.loads(paths['membership'].read_text());runtime=json.loads(paths['runtime_manifest'].read_text())
    bn=json.loads(paths['bn_receipt'].read_text());regions=json.loads(paths['regions'].read_text())
    require(m.get('schema')=='flow_round13_coverage_identity_v1' and m['checkpoint_sha256']==CHECKPOINT_SHA,'R13 member contract differs')
    require(runtime.get('schema')=='flow_round13_runtime_v1','Runtime schema differs')
    require(m['source_sha256']['canonical']==inputs['canonical']['sha256'] and m['source_sha256']['roles']==inputs['roles']['sha256'],'Identity source pins differ')
    with paths['roles'].open(newline='') as stream:roles={r['target_id']:r['auxiliary_role'] for r in csv.DictReader(stream)}
    # Canonical JSON is either rows list or the archived dict carrying targets.
    canonical=json.loads(paths['canonical'].read_text())
    rows=canonical if isinstance(canonical,list) else canonical['records']
    by_id={r['target_id']:r for r in rows}
    require(len(by_id)==len(rows),'Duplicate canonical identity')
    train=set(m['split_sequences']['train']);check=set(m['split_sequences']['check'])
    require(len(train)==8 and len(check)==2 and not train&check,'Sequence split differs')
    ids={}
    for name in ('train_members_reference','train_members_diversity','check_members','bn_members','selector_members'):
        members=m[name];ids[name]=[r['target_id'] for r in members]
        require(len(ids[name])==len(set(ids[name])) and members,'Empty/duplicate membership')
        expected_role='bn_calibration' if name=='bn_members' else 'gate_fit'
        require(all(r==by_id.get(r['target_id']) and r['protocol_split']=='train' and roles.get(r['target_id'])==expected_role for r in members),'Canonical/role differs')
        require({r['sequence'] for r in members}==(check if name=='check_members' else train),'Member sequence role differs')
    require(len(ids['check_members'])==8 and collections.Counter(r['sequence'] for r in m['check_members'])==collections.Counter({s:4 for s in check}),'Check8 must be4 per sequence')
    require(len(ids['train_members_reference'])==40 and len(ids['train_members_diversity'])==200 and set(ids['train_members_reference'])<=set(ids['train_members_diversity']),'Nested40/200 differs')
    require(len(ids['bn_members'])==176 and collections.Counter(r['sequence'] for r in m['bn_members'])==collections.Counter({s:22 for s in train}),'BN176 differs')
    names=('train_members_diversity','check_members','bn_members','selector_members')
    for i,name in enumerate(names):
        for other in names[i+1:]:require(not set(ids[name])&set(ids[other]),'Identity role overlap')
    for condition,repeats in (('reference',5),('diversity',1)):
        order=m['train_order_'+condition]
        require([r['step'] for r in order]==list(range(1,201)),'Training step order differs')
        require(collections.Counter(r['target_id'] for r in order)==collections.Counter({x:repeats for x in ids['train_members_'+condition]}),'Training repetition differs')
        seen=collections.Counter()
        for r in order:
            seen[r['sequence']]+=1
            require(r['sequence_visit_index']==seen[r['sequence']] and {k:v for k,v in r.items() if k not in ('step','sequence_visit_index')}==by_id.get(r['target_id']),'Ordered canonical row differs')
        require(seen==collections.Counter({s:25 for s in train}),'Training sequence counts differ')
    require([r['sequence'] for r in m['train_order_reference']]==[r['sequence'] for r in m['train_order_diversity']],'Unmatched sequence schedule')
    require(bn.get('schema')=='flow_round13_common_bn_v1' and bn.get('status')=='COMMON_BN_COMPLETE' and bn.get('checkpoint_sha256')==CHECKPOINT_SHA,'BN source/status differs')
    require(bn.get('membership_sha256')==inputs['membership']['sha256'] and bn.get('buffers_sha256')==inputs['bn_buffers']['sha256'] and bn.get('bn_target_ids')==ids['bn_members'],'BN identity/source differs')
    require(bn.get('old_CMA_loaded') is False and bn.get('GT_used') is False and bn.get('calibration_forwards')==176 and
            bn.get('calibration_bn_count')==78 and bn.get('retained_mlp_bn_count')==24 and bn.get('BN_scope')=='MLP_R13_COMMON_CMA176' and
            bn.get('calibration_definition')=='all78_reset_CMA_train_nonBN_eval','BN semantics differs')
    require(runtime['region_sha256']==inputs['regions']['sha256'] and regions['fit_target_ids']==ids['train_members_diversity'],'Training-only region source differs')
    require(regions['canonical_manifest_sha256']==inputs['canonical']['sha256'] and regions['auxiliary_roles_sha256']==inputs['roles']['sha256'],'Region identity sources differ')
    require(regions['boundary_quantile']==.95 and regions['large_quantile']==.90,'Region quantiles differ')
    require(all(math.isfinite(regions[k]) and regions[k]>=0 for k in ('boundary_threshold_pixels','large_displacement_threshold_pixels')),'Region thresholds invalid')
    return m,bn


def validate_completion(raw,member,expected_binding=None):
    """Validate all eight headers/contracts before inspecting quality endpoints."""
    expected={(c,a) for c in CONDITIONS for a in ARMS}
    require(set(raw)==expected,'All eight conditions/arms must be supplied before quality analysis')
    common={};initial_state=None;bn_scope=None;full=0
    ids=[r['target_id'] for r in member['check_members']]
    for condition in CONDITIONS:
        for arm in ARMS:
            z=raw[(condition,arm)];binding=z['binding']
            require(z.get('schema')=='flow_round13_coverage_recovery_v1' and z.get('status')=='ROUND13_COVERAGE_RECOVERY_COMPLETE','All eight arms must be complete before analysis')
            require(z['arm']==binding['arm']==arm and z['condition']==binding['condition']==condition and z['mode']==binding['mode']=='recover','Condition/arm/mode mismatch')
            require(z['training_performed'] is True and z['completed_optimizer_steps']==200 and len(z['steps'])==200,'Formal200 updates missing')
            require([r['step'] for r in z['steps']]==list(range(1,201)) and [r['target_id'] for r in z['steps']]==[r['target_id'] for r in member['train_order_'+condition]],'Actual update identity order differs')
            require([r['sequence'] for r in z['steps']]==[r['sequence'] for r in member['train_order_'+condition]],'Actual update sequence order differs')
            require(binding['fit_ids']==[r['target_id'] for r in member['train_order_'+condition]] and binding['check_ids']==ids,'Bound identity order differs')
            require(all(z[k] is True for k in ('frozen_state_unchanged','BN_buffers_unchanged','original_modules_restored','all_hooks_removed')),'Execution closure missing')
            require(z['smoke_weights_discarded'] is False and z['weights_file']=='weights.pt' and bool(re.fullmatch('[0-9a-f]{64}',z['weights_sha256'])),'Formal weight artifact record missing')
            require(z['initial_numerical_pair']['all_flow_outputs_exact'] and z['final_numerical_pair']['all_flow_outputs_exact'],'Initial/final exact numerical audit missing')
            if arm!='native':require(z['initial_numerical_pair']['actual_dual_all_flows_branches_exact'] and z['final_numerical_pair']['actual_dual_all_flows_branches_exact'],'Actual/dual audit missing')
            budget=244 if arm=='native' else 246
            require(z['full_forward_count']==budget and len(z['forward_records'])==budget,'Forward accounting differs')
            require(binding['expected_training_forwards']==200 and binding['expected_audit_forwards']==(4 if arm=='native' else 6) and binding['expected_evaluation_forwards']==40,'Fixed budget binding differs')
            require(sum(r['purpose']=='training' for r in z['forward_records'])==200 and [r['forward_id'] for r in z['forward_records']]==list(range(1,budget+1)),'Training/forward sequence differs')
            require(z['hardware_benefit_claim'] is False and z['flow_specific_increment_claim'] is False,'Unsupported scientific claims in raw record')
            for k in COMMON_FIELDS:
                common.setdefault(k,binding[k]);require(common[k]==binding[k],'Mixed common binding: '+k)
                if expected_binding and k in expected_binding:require(binding[k]==expected_binding[k],'Explicit analysis source differs: '+k)
            if initial_state is None:initial_state=z['initial_model_state_sha256']
            require(initial_state==z['initial_model_state_sha256'],'All eight original common-BN initial states must be identical')
            if bn_scope is None:bn_scope=z['BN_scope']
            require(z['BN_scope']==bn_scope and bn_scope['scope']=='MLP_R13_COMMON_CMA176' and bn_scope['BN_count']==78 and
                    bn_scope['MLP_BN_count']==24 and bn_scope['old_CMA_loaded'] is False,'Common BN state/semantics differ')
            require(len(bn_scope['frozen_BN_names'])==24 and len(bn_scope['current_stat_BN_names'])==54 and
                    set(bn_scope['frozen_BN_names'])==set(bn_scope['MLP_BN_names']) and
                    not set(bn_scope['frozen_BN_names'])&set(bn_scope['current_stat_BN_names']),'BN scope name closure differs')
            require(bn_scope['new_buffers_sha256']==binding['BN_buffers_sha256'] and bn_scope['new_receipt_sha256']==binding['BN_receipt_sha256'],'BN state differs from common inputs')
            full+=budget
    require(full==1964,'Two-condition formal budget differs')
    common['initial_state_sha256']=initial_state
    return common


def validate_predictions(path,z):
    """SHA/shape checks of every retained NPZ; called only after all arms complete."""
    import numpy as np
    files=payloads=0
    for label,expected in (('initial',(z['arm'],)),('endpoint',ARMS)):
        for frame in z['evaluations'][label]['frames']:
            name=frame['prediction_file'];require(Path(name).name==name,'Unsafe prediction filename')
            p=path.parent/name;require(sha(p)==frame['prediction_file_sha256'],'Prediction NPZ differs')
            with np.load(p,allow_pickle=False) as arrays:
                require(set(arrays.files)==set(expected),'Prediction NPZ layout keyset differs')
                for mode in expected:
                    prediction=arrays[mode]
                    require(prediction.dtype==np.float32 and prediction.shape==(2,480,640) and bool(np.isfinite(prediction).all()),'Prediction shape/dtype/finite differs')
                    require(hashlib.sha256(prediction.tobytes()).hexdigest()==frame['arms'][mode]['prediction_sha256'],'Prediction payload SHA differs')
                    payloads+=1
            files+=1
    return {'prediction_files':files,'prediction_payloads':payloads}


def summarize_frames(frames,sequences):
    scopes={'overall':frames};scopes.update({s:[f for f in frames if f['sequence']==s] for s in sequences})
    require(all(len(scopes[s])==4 for s in sequences),'Per-sequence check membership differs')
    return {s:{r:aggregate([f['regions'][r] for f in fs]) for r in REGIONS} for s,fs in scopes.items()}


def differences(candidate,reference):
    result={}
    for scope,regions in candidate.items():
        result[scope]={}
        for region in REGIONS:
            c=regions[region];r=reference[scope][region]
            require(c['valid_pixels']==r['valid_pixels'] and c['frames_with_region']==r['frames_with_region'],'Paired region support differs')
            values={'candidate':c,'reference':r,'evidence':c['frame_mean_AEE'] is not None}
            for metric in ('frame_mean_AEE','pixel_weighted_AEE','error_sum'):
                x,y=c[metric],r[metric]
                require((x is None)==(y is None),'Paired empty region differs')
                values[metric+'_difference']=None if x is None else x-y
                values[metric+'_relative_change']=None if y in (None,0) else x/y-1
            result[scope][region]=values
    return result


def member_changes(candidate,reference):
    rows=[]
    for c,r in zip(candidate,reference):
        require((c['target_id'],c['sequence'])==(r['target_id'],r['sequence']),'Unpaired members')
        initial=r['regions']['all_valid']['AEE'];post=c['regions']['all_valid']['AEE']
        gain=initial-post
        rows.append({'target_id':c['target_id'],'sequence':c['sequence'],'reference_AEE':initial,'candidate_AEE':post,
                     'candidate_minus_reference_AEE':post-initial,'initial_minus_post_AEE':gain})
    total=sum(r['initial_minus_post_AEE'] for r in rows)
    for r in rows:r['signed_gain_share_percent']=100*r['initial_minus_post_AEE']/total if total else None
    return {'per_member':rows,'signed_total_gain':total,'wins':sum(r['initial_minus_post_AEE']>0 for r in rows),
            'ties':sum(r['initial_minus_post_AEE']==0 for r in rows),'losses':sum(r['initial_minus_post_AEE']<0 for r in rows),
            'share_warning':'Signed contribution may exceed100percent when other members offset it; no member is dropped.'}


def analyze_completed(raw,member,common):
    # Caller must first run validate_completion for all eight records.
    validate_completion(raw,member)
    check=member['check_members'];ids=[r['target_id'] for r in check];sequences=sorted({r['sequence'] for r in check})
    by_id={r['target_id']:r for r in check};counts={};input_evidence={};cells={c:{} for c in CONDITIONS}
    initial_sha={};predictions=0
    for condition in CONDITIONS:
        for arm in ARMS:
            z=raw[(condition,arm)]
            require(set(z['evaluations'])=={'initial','endpoint'},'Extra/missing evaluation state')
            for label,expected in (('initial',(arm,)),('endpoint',ARMS)):
                frames=z['evaluations'][label]['frames']
                require([f['target_id'] for f in frames]==ids,'Evaluation member order differs')
                for mode in expected:
                    key=('initial' if label=='initial' else arm)+'/'+mode
                    require(key not in cells[condition],'Duplicate state/layout')
                    records=[]
                    for f in frames:
                        target=f['target_id'];require(f['sequence']==by_id[target]['sequence'] and set(f['arms'])==set(expected),'Evaluation sequence/layout differs')
                        input_evidence.setdefault(target,f['input_evidence']);require(input_evidence[target]==f['input_evidence'],'Evaluation input bytes differ across states')
                        value=f['arms'][mode];metric=value['metrics'];regions=metric['regions']
                        require(metric['target_id']==target and metric['sequence']==f['sequence'] and set(regions)==set(REGIONS),'Metric identity/regions differ')
                        require(bool(re.fullmatch('[0-9a-f]{64}',value['prediction_sha256'])),'Invalid prediction SHA')
                        for region,v in regions.items():
                            count_key=(target,region);counts.setdefault(count_key,v['valid_pixels'])
                            require(type(v['valid_pixels']) is int and v['valid_pixels']>=0 and counts[count_key]==v['valid_pixels'],'Region support differs')
                            require(math.isfinite(v['error_sum']) and v['error_sum']>=0,'Invalid region error sum')
                            if v['valid_pixels']:
                                require(v['AEE'] is not None and math.isfinite(v['AEE']) and v['AEE']>=0 and
                                        math.isclose(v['AEE'],v['error_sum']/v['valid_pixels'],rel_tol=1e-10,abs_tol=1e-10),'Invalid region AEE')
                            else:require(v['AEE'] is None and v['error_sum']==0,'Empty region must have no metric evidence')
                        require(regions['all_valid']['valid_pixels']>0,'No empty all-GT sample')
                        require(regions['all_valid']['valid_pixels']==regions['voxel_nonzero']['valid_pixels']+regions['voxel_zero']['valid_pixels'],'Voxel support does not conserve all valid GT')
                        if label=='initial':
                            sk=(arm,target);item=(value['prediction_sha256'],regions)
                            initial_sha.setdefault(sk,item);require(initial_sha[sk]==item,'Initial predictions/metrics differ between coverage conditions')
                        records.append({'target_id':target,'sequence':f['sequence'],'regions':regions,'prediction_sha256':value['prediction_sha256']})
                        predictions+=1
                    cells[condition][key]=records
    require(all(len(v)==20 for v in cells.values()) and predictions==320,'Require40 state/layout cells and320 payloads')
    summaries={condition:{k:summarize_frames(v,sequences) for k,v in cell.items()} for condition,cell in cells.items()}
    own={};candidate_gates={};paired={}
    for condition in CONDITIONS:
        candidate_gates[condition]=gates(summaries[condition]);own[condition]={}
        for arm in ARMS:
            key=arm+'/'+arm
            own[condition][arm]={'change':differences(summaries[condition][key],summaries[condition]['initial/'+arm]),
                                  'members':member_changes(cells[condition][key],cells[condition]['initial/'+arm])}
    for key in summaries['reference']:
        paired[key]={'change':differences(summaries['diversity'][key],summaries['reference'][key]),
                     'members':member_changes(cells['diversity'][key],cells['reference'][key])}
    normal={c:all(compare(summaries[c][a+'/'+a],summaries[c]['initial/'+a])['pass'] for a in ('native','FF')) for c in CONDITIONS}
    if normal['diversity'] and not normal['reference']:
        interpretation='Supports combined greater-identity-coverage/lower-repetition effect within this fixed state, recipe and check set; does not separate their causal effects.'
    elif all(normal.values()):interpretation='Both ordinary references pass; evaluate fixed coarse-base gates. Diversity is not assumed necessary.'
    elif not normal['diversity']:interpretation='Coverage axis has not established a stable diversity ordinary reference; optimizer, parameter scope, BN and original checkpoint remain competing explanations.'
    else:raise AssertionError('Unreachable normality pattern')
    return {'schema':'flow_round13_coverage_quality_analysis_v1','script_sha256':sha(__file__),'R12_V2_source_sha256':R12_V2_SHA,
            'analysis_protocol_sha256':PROTOCOL_SHA,'common_contract':common,'initial_state_all_eight_exact':True,
            'initial_same_layout_prediction_SHA_and_metrics_cross_condition_exact':True,'summaries':summaries,
            'candidate_gates':candidate_gates,'ordinary_reference_normality':normal,'own_layout_recovery':own,
            'diversity_minus_reference_all_states':paired,'all_endpoints':cells,'full_forward_count':1964,
            'state_layout_count':40,'evaluation_endpoint_count':320,'region_count':16,'scopes':['overall']+sequences,
            'interpretation':interpretation,'flow_specific_increment_claim':False,'hardware_benefit_claim':False,
            'scope':'Sequence-held for this recovery/BN/selector split; original backbone trained on all train sequences. Not official DSEC, unseen-backbone, significance, dynamic-selector, flow-novelty or net-cycle evidence.'}


def render_markdown(result):
    def fmt(x):return 'empty' if x is None else '%.9f'%x
    lines=['# R13 全部覆盖条件、状态与布局','',result['scope'],'',result['interpretation'],'',
           '四类质量门为合取：allGT≤+3%、B/L≤+5%，总体和两序列同时检查。空区不构成改善证据。',
           '', '| 条件/自身布局 | 初始allGT | 恢复后allGT | ΔAEE | 相对变化 | 胜/平/负 |','|---|---:|---:|---:|---:|---|']
    for condition in CONDITIONS:
        for arm in ARMS:
            v=result['own_layout_recovery'][condition][arm];r=v['change']['overall']['all_valid'];w=v['members']
            lines.append('| %s/%s | %s | %s | %s | %s | %d/%d/%d |'%(condition,arm,fmt(r['reference']['frame_mean_AEE']),fmt(r['candidate']['frame_mean_AEE']),fmt(r['frame_mean_AEE_difference']),fmt(r['frame_mean_AEE_relative_change']),w['wins'],w['ties'],w['losses']))
    lines+=['','相对变化为比例（0.01=1%）；D−R负值表示diversity AEE更低。下表保留全部状态/布局；初始行必须为零。','',
            '| 状态/布局 | reference allGT | diversity allGT | D−R allGT | D−R B | D−R L |','|---|---:|---:|---:|---:|---:|']
    for key,value in result['diversity_minus_reference_all_states'].items():
        row=value['change']['overall'];v=row['all_valid']
        lines.append('| %s | %s | %s | %s | %s | %s |'%(key,fmt(v['reference']['frame_mean_AEE']),fmt(v['candidate']['frame_mean_AEE']),fmt(v['frame_mean_AEE_difference']),fmt(row['B']['frame_mean_AEE_difference']),fmt(row['L']['frame_mean_AEE_difference'])))
    for condition in CONDITIONS:
        for arm,value in result['candidate_gates'][condition].items():lines+=['',condition+'/'+arm+': '+value['classification']]
    lines+=['','全部16区域、总体与每序列、每帧、有符号损伤贡献、像素加权值及四类门细节见JSON。贡献率可能超过100%，不能据此删除困难帧。',
            '本轮没有光流特征评分增量或硬件收益主张。']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--inputs-sha256',required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--cpu-receipt',type=Path,default=HERE/'analysis_cpu_results.json');a=p.parse_args()
    require(sys.flags.optimize==0,'Pinned V2 asserts require normal Python mode')
    cpu=json.loads(a.cpu_receipt.read_text());require(cpu.get('passed') is True and cpu.get('GPU_initialized') is False,'Analysis CPU checks must pass')
    require(cpu.get('R12_V2_source_sha256')==R12_V2_SHA and cpu.get('analysis_protocol_sha256')==PROTOCOL_SHA,'Analysis CPU provenance differs')
    for name in ('analyze_coverage.py','test_analyze_coverage.py'):
        require(cpu['source_sha256'][name]==sha(HERE/name),'Stale analysis CPU receipt: '+name)
    require(not a.output_dir.exists(),'Fresh analysis directory required');require(sha(a.inputs)==a.inputs_sha256,'Analysis input manifest SHA differs')
    inputs=json.loads(a.inputs.read_text());member,bn=validate_input_protocol(inputs)
    require(len(inputs['results'])==8,'All eight result sources required')
    raw={};paths={};evidence={}
    for item in inputs['results']:
        path=read_pin(item);key=(item['condition'],item['arm']);require(key not in raw,'Duplicate result source')
        raw[key]=json.loads(path.read_text());paths[key]=path;evidence['/'.join(key)]={'path':str(path),'sha256':item['sha256']}
    expected={'runner_sha256':sha(HERE/'run_coverage_recovery.py'),'execute_sha256':sha(HERE/'execute_recovery.py'),
              'runner_checks_sha256':sha(HERE/'runner_cpu_results.json'),'source_pins_sha256':sha(HERE/'inherited_pins.json'),
              'wrapper_sha256':sha(HERE.parents[1]/'ordinary_recovery_round12/trainable_representative_mlp.py'),
              'allowed_parameter_count':7252064,'loss_sha256':'92aa1451469690ad89c16b6954c972d3c460689f5cb765662b4eff7fc7dec122',
              'membership_sha256':inputs['membership']['sha256'],'runtime_manifest_sha256':inputs['runtime_manifest']['sha256'],
              'BN_buffers_sha256':inputs['bn_buffers']['sha256'],'BN_receipt_sha256':inputs['bn_receipt']['sha256'],
              'region_sha256':inputs['regions']['sha256'],'checkpoint_sha256':CHECKPOINT_SHA}
    common=validate_completion(raw,member,expected)
    # There is no quality aggregation before all eight completed contracts pass.
    audits={condition+'/'+arm:validate_predictions(paths[(condition,arm)],raw[(condition,arm)]) for condition in CONDITIONS for arm in ARMS}
    require(sum(v['prediction_files'] for v in audits.values())==128 and sum(v['prediction_payloads'] for v in audits.values())==320,'All retained predictions required')
    result=analyze_completed(raw,member,common);result.update(inputs=evidence,analysis_inputs_sha256=a.inputs_sha256,analysis_cpu_receipt_sha256=sha(a.cpu_receipt),prediction_artifact_audits=audits)
    a.output_dir.mkdir(parents=True);(a.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (a.output_dir/'ANALYSIS.md').write_text(render_markdown(result))
    print(json.dumps({'status':'ALL_EIGHT_COMPLETE_ANALYSIS_WRITTEN','state_layout_count':40,'evaluation_endpoint_count':320}))


if __name__=='__main__':main()
