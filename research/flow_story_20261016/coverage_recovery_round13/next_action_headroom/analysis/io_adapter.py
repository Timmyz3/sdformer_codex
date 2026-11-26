#!/usr/bin/env python3
"""CPU-only H0/H1 artifact binding and independent raw metric recomputation.

The frozen pure core is unchanged. No implementation analyzer is imported.
Ordinary scores are source/u/payload bound, not independently re-executed PSN.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import analysis_core as core

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
R13 = HERE.parents[1]
TARGET = 'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.2.mlp'
BRANCHES = {'sttmultires_unet.encoders.swin3d.layers.%d.swin_blocks.%d.mlp' % (s,b):
            [10,1,120,160,96] if s == 0 else [10,1,30,40,384]
            for s, depth in ((0,2),(2,6)) for b in range(depth)}
PREFIX = tuple(list(BRANCHES)[:5])
H0_OWN = ('h0_core.py','run_h0.py','test_h0_cpu.py','source_pins.json','h8_manifest.json')
H1_OWN = ('h1_core.py','run_h1.py','finalize_h1.py','test_h1_cpu.py','source_pins.json')
MEMBER_SHA = 'eba4544fe99c082d7d099f90d03470d7abad22b01b68d3eb4d022e61e78151d1'
REGION_SHA = '2e4934f6d82663f16ca1ae6c3aea238a31e699d4f49916052f6ea6b5e9c1d41b'
RAW_MATH_SHA = '26a2e5c97b988f10685e1c440a11506354d9591cb9b36a567292abb6d0348088'
require = core.require


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def digest_ok(value):
    return isinstance(value,str) and len(value)==64 and set(value)<=set('0123456789abcdef')


class Assets:
    """Relocate by expected bytes, never silently substitute a similarly named file."""
    def __init__(self, relocations):
        self.relocations = relocations
        self.checked = {}

    def path(self, descriptor):
        require(set(descriptor)=={'path','sha256'} and digest_ok(descriptor['sha256']), 'Explicit path/SHA asset required')
        digest = descriptor['sha256']
        path = Path(self.relocations.get(digest,descriptor['path']))
        require(path.is_absolute() and path.is_file(), 'Existing absolute asset required: '+str(path))
        require(sha(path)==digest, 'Asset bytes differ: '+str(path))
        self.checked[str(path)] = digest
        return path

    def read(self, descriptor):
        return json.loads(self.path(descriptor).read_text())


def load_raw_math():
    path = R13/'independent_quality_review/reduce_raw_quality.py'
    require(sha(path)==RAW_MATH_SHA,'Independent raw-math source changed')
    spec = importlib.util.spec_from_file_location('_headroom_independent_raw_math',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_preflight():
    io_freeze=json.loads((HERE/'io_freeze_v1.json').read_text())
    for name,digest in io_freeze['files'].items():
        require(sha(HERE/name)==digest,'Frozen I/O analysis changed: '+name)
    pure = json.loads((HERE/'freeze_v1.json').read_text())
    for name,digest in pure['files'].items():
        require(sha(HERE/name)==digest,'Frozen pure analysis changed: '+name)
    pins = json.loads((HERE/'io_source_pins_v1.json').read_text())
    for relative,digest in pins['sources'].items():
        require(sha(REPO/relative)==digest,'Pinned I/O dependency changed: '+relative)
    return pins


def h0_plan(members):
    out=[]
    for index,row in enumerate(members):
        for anchor in ('native','CC','FF','P','F12','T8'):
            item={'state':'W','anchor':anchor,'mode':'actual','target_id':row['target_id']}
            out.append(item)
            if index==0 and anchor!='native':out.append(dict(item,mode='dual_reference'))
    for role,anchor in (('N','matched_N'),('F','matched_F')):
        out.extend({'state':role,'anchor':anchor,'mode':'actual','target_id':r['target_id']} for r in members)
    return [dict(item,forward_id=i+1) for i,item in enumerate(out)]


def h1_plan(members):
    out=[]
    masks=[(i,) for i in range(12)]+list(core.PAIRS)
    counts={}
    for index,row in enumerate(members):
        segment=index//2+1
        for mask in masks:
            item={'action':core.action(mask),'tiles':[list(core.TILES[i]) for i in mask],'k':len(mask),
                  'target_id':row['target_id'],'sequence':row['sequence'],'segment':segment,'mode':'actual'}
            out.append(item)
            if index==0 and mask==(0,1):out.append(dict(item,mode='dual_reference'))
    for index,item in enumerate(out):
        segment=item['segment'];counts[segment]=counts.get(segment,0)+1
        item.update(global_forward_id=index+1,segment_forward_id=counts[segment])
    return out


def execution(ex,rows):
    require(ex['summed_fc1_rows']==ex['summed_fc2_rows']==rows,'Actual producer rows differ')
    require(len(ex['head_records'])==4 and all(h['shape']==[1,2,480,640] and h['dtype']=='torch.float32'
            and h['finite'] is True and digest_ok(h['sha256']) for h in ex['head_records']), 'Four complete final heads required')
    require(set(ex['branch_records'])==set(BRANCHES),'Eight branch records required')
    for name,shape in BRANCHES.items():
        record=ex['branch_records'][name]
        require(record['shape']==shape and all(digest_ok(record[k]) for k in ('input_sha256','output_sha256')), 'Branch geometry/SHA differs')


def prefix(ex,cc):
    for name in PREFIX:
        require(ex['branch_records'][name]['input_sha256']==cc['branch_records'][name]['input_sha256'], 'Changed causal-prefix input')
        if name!=TARGET:
            require(ex['branch_records'][name]['output_sha256']==cc['branch_records'][name]['output_sha256'], 'Changed causal-prefix branch')


def dual(actual,reference):
    require(all(actual[k]==reference[k] for k in ('head_records','branch_records')), 'Actual/dual full heads or branches differ')


def validate_h0(value,members,expected_sources):
    require(value.get('schema')=='flow_h0_action_screen_v1' and value.get('status')=='H0_69_FORWARD_COMPLETE'
            and value.get('inference_executed') is True,'Complete actual H0 required')
    require(value['full_forward_count']==69 and value['completed_targets']==8 and len(value['forward_records'])==69,'H0 count differs')
    require(value.get('training_performed') is False and value.get('new_BN_calibration_performed') is False,'H0 changed training/BN')
    require(all(value.get(k) is True for k in ('all_recovered_states_unchanged','BN_buffers_unchanged','original_modules_restored','all_hooks_removed')),'H0 closure missing')
    binding=value['binding']
    require(binding['source_sha256']==expected_sources and binding['H8_ids']==[m['target_id'] for m in members], 'H0 source/member differs')
    require(binding['cpu_checks_sha256']==sha(HERE.parent/'implementation_h0/cpu_results.json'),'H0 CPU receipt source differs')
    require(binding['common']['membership_sha256']==MEMBER_SHA and binding['region_sha256']==REGION_SHA,'H0 region/member binding differs')
    require(binding['GT_used_for_actions'] is False and binding['total_full_forwards']==69,'H0 action provenance differs')
    require([f['target_id'] for f in value['frames']]==binding['H8_ids'],'H0 frame ordering differs')
    require([f['target_id'] for f in value['data_evidence']]==binding['H8_ids'],'H0 source evidence coverage differs')
    by_id={f['target_id']:f for f in value['frames']};actual={}
    totals={'native':558000,'matched_N':558000,'FF':558000,'matched_F':558000,'CC':153600,'P':164200,'F12':162240,'T8':155560}
    plan=h0_plan(members)
    require(binding['forward_plan']==plan,'H0 bound plan differs')
    for record,item in zip(value['forward_records'],plan):
        require(all(record[k]==v for k,v in item.items()),'H0 ledger order/identity differs')
        ex=record['execution'];execution(ex,609600 if item['mode']=='dual_reference' else totals[item['anchor']])
        key=(item['target_id'],item['anchor'])
        if item['mode']=='actual':
            actual[key]=ex
            arm=by_id[item['target_id']]['arms'][item['anchor']]
            require(arm['execution']==ex and arm['state']==item['state'] and arm['prediction_sha256']==ex['head_records'][-1]['sha256'],'H0 endpoint/state/forward mismatch')
            if item['anchor'] in ('P','F12','T8'):prefix(ex,actual[(item['target_id'],'CC')])
        else:dual(actual[key],ex)
    require(set(value['first_frame_dual_audits'])=={'CC','FF','P','F12','T8'},'H0 dual receipt coverage differs')
    for frame,member in zip(value['frames'],members):
        require(frame['sequence']==member['sequence'] and set(frame['arms'])==set(core.ANCHORS),'H0 endpoint/sequence coverage differs')
    return by_id


def validate_h1(value,segments,h0,members,expected_sources,h0_sha):
    require(value.get('schema')=='flow_h1_625_forward_complete_v1' and value.get('status')=='H1_625_FORWARD_COMPLETE','Complete actual H1 required')
    require((value['full_forward_count'],value['actual_forwards'],value['dual_forwards'],value['completed_targets'])==(625,624,1,8),'H1 count differs')
    binding=value['common_binding']
    require(binding['source_sha256']==expected_sources and binding['h0_binding']==h0['binding'] and binding['h0_result_sha256']==h0_sha,'H1/H0/source binding differs')
    require(binding['cpu_checks_sha256']==sha(HERE.parent/'implementation_h1/cpu_results.json') and
            binding['h0_cpu_checks_sha256']==h0['binding']['cpu_checks_sha256'] and binding['h0_inputs_sha256']==h0['binding']['inputs_sha256'],'H1 CPU/input source differs')
    require(binding['H8_ids']==[m['target_id'] for m in members] and binding['H8_sequences']==[m['sequence'] for m in members],'H1 member order differs')
    require(binding['BN_buffers_sha256']==h0['binding']['common']['BN_buffers_sha256'],'H1 BN differs')
    require(value['GT_used_to_generate_masks'] is False and binding['GT_used_to_generate_masks'] is False,'H1 GT/action contamination')
    require([f['target_id'] for f in value['frames']]==binding['H8_ids'],'H1 frames incomplete')
    require(len(segments)==4 and {s['binding']['segment'] for s in segments}=={1,2,3,4},'Four original complete segments required')
    differing={'segment','segment_ids','segment_forward_budget','forward_plan','wall_minutes'}
    concatenated=[];original_frames=[]
    for segment in sorted(segments,key=lambda s:s['binding']['segment']):
        require(segment['schema']=='flow_h1_finite_action_segment_v1' and segment['status']=='H1_SEGMENT_COMPLETE'
                and segment['inference_executed'] is True,'Incomplete actual segment')
        require({k:v for k,v in segment['binding'].items() if k not in differing}==binding,'Mixed H1 segments')
        require(all(segment.get(k) is True for k in ('recovered_W_unchanged','BN_buffers_unchanged','original_modules_restored','all_hooks_removed')),'H1 segment closure missing')
        require(segment['training_performed'] is False and segment['new_BN_calibration_performed'] is False,'H1 unexpected update')
        loaded=segment['recovered_state_load']
        require(loaded==h0['recovered_state_loads']['W'] and loaded['state_sha256']==binding['W_descriptor']['final_state_sha256'],'H1 recovered W differs')
        require(segment['BN_scope']['BN_signature_before_inference']==h0['BN_scope']['BN_signature_before_inference'],'H1 actual BN signature differs')
        require(segment['full_forward_count']==len(segment['forward_records'])==(157 if segment['binding']['segment']==1 else 156),'Incomplete segment ledger')
        require(segment['binding']['forward_plan']==[p for p in h1_plan(members) if p['segment']==segment['binding']['segment']],'H1 segment plan differs')
        concatenated.extend(segment['forward_records']);original_frames.extend(segment['frames'])
    require(value['forward_records']==concatenated,'Aggregated H1 ledger differs from raw segments')
    require(len(original_frames)==8 and [{k:v for k,v in f.items() if k not in ('prediction_source','predecision_u_source')} for f in value['frames']]==original_frames,'Aggregated H1 frames differ from raw segments')
    by_id={f['target_id']:f for f in value['frames']};coarse={f['target_id']:f['arms']['CC'] for f in h0['frames']};actual={}
    expected=h1_plan(members)
    require(len(value['forward_records'])==len(expected)==625,'H1 full ledger missing')
    for record,item in zip(value['forward_records'],expected):
        require(all(record[k]==v for k,v in item.items()),'H1 fixed enumeration/order differs')
        ex=record['execution'];execution(ex,609600 if item['mode']=='dual_reference' else 153600+720*item['k'])
        require(ex['action']==item['action'] and ex['tiles']==item['tiles'] and ex['mode']==item['mode'],'H1 execution action differs')
        prefix(ex,coarse[item['target_id']]['execution'])
        key=(item['target_id'],item['action'])
        if item['mode']=='actual':
            actual[key]=ex;arm=by_id[item['target_id']]['arms'][item['action']]
            require(arm['execution']==ex and arm['prediction_sha256']==ex['head_records'][-1]['sha256'],'H1 endpoint/forward mismatch')
        else:dual(actual[key],ex)
    for frame,member in zip(value['frames'],members):
        require(frame['sequence']==member['sequence'] and set(frame['arms'])=={core.action((i,)) for i in range(12)}|{core.action(p) for p in core.PAIRS},'H1 action support differs')
        cc=frame['H0_CC_reference']
        require(cc['result_sha256']==h0_sha and cc['execution']==coarse[member['target_id']]['execution'] and cc['prediction_sha256']==coarse[member['target_id']]['prediction_sha256'],'Saved H0 CC reference changed')


def validate_prior_audits(quality,execution_audit,common):
    require(quality['schema']=='flow_round13_coverage_quality_analysis_v1' and quality['script_sha256']=='b1997b55deea309088388a011b83272bafe438bedc43db5ee4440655bc5034ad','Unbound R13 analysis source')
    require(quality['analysis_protocol_sha256']=='5ec51175fbf18b9d2f4fab329247b011700f4cf109b50bf73e905708698c2d7b' and
            (quality['state_layout_count'],quality['evaluation_endpoint_count'],quality['region_count'],quality['full_forward_count'])==(40,320,16,1964),'R13 complete quality coverage differs')
    require(quality['initial_state_all_eight_exact'] is True and quality['initial_same_layout_prediction_SHA_and_metrics_cross_condition_exact'] is True,'R13 initial pairing failed')
    expected={(c,m,a) for c in ('reference','diversity') for m in ('smoke','recover') for a in ('native','FF','CC','fine_s2b2')}
    keys={c+'/'+a for c,m,a in expected if m=='recover'}
    require(set(quality['inputs'])==set(quality['prediction_artifact_audits'])==keys and
            all(v=={'prediction_files':16,'prediction_payloads':40} for v in quality['prediction_artifact_audits'].values()),'R13 complete prediction artifact coverage differs')
    require(execution_audit['schema']=='flow_round13_independent_execution_audit_v1' and execution_audit['status']=='PASS','Independent R13 execution audit required')
    require(execution_audit['auditor_sha256']=='dd57f9f315e1c5f38961208496a37f46db577c47b8507029a55b5bc216d8c8e8' and
            (execution_audit['weights_count'],execution_audit['prediction_count'],execution_audit['archive_count'],execution_audit['full_forwards'],execution_audit['optimizer_updates'])==(8,320,128,2048,1640),'Independent execution source/count differs')
    scope=execution_audit['scope']
    require(set(scope['conditions'])=={'reference','diversity'} and set(scope['modes'])=={'smoke','recover'} and scope['runs']==16,'Independent execution scope differs')
    rows=execution_audit['results']
    require(len(rows)==16 and {(r['condition'],r['mode'],r['arm']) for r in rows}==expected,'Independent execution run membership duplicates/missing')
    require(execution_audit['paired_initial_predictions_equal'] is True and
            execution_audit['common_initial_model_sha256']==quality['common_contract']['initial_state_sha256'] and
            execution_audit['original_checkpoint_sha256']==common['checkpoint_sha256'] and
            execution_audit['new_BN_actual_sha256']==common['BN_buffers_sha256'],'Independent source/initial/BN pairing differs')
    require(all(quality['common_contract'][k]==v for k,v in common.items()),'Quality audit common source differs')


def bind_admission(inputs,h0,assets):
    require(inputs['schema']=='flow_h0_inputs_v1' and set(inputs['states'])=={'W','N','F'},'H0 named input states required')
    binding=h0['binding'];common=binding['common']
    require(inputs['condition']==binding['condition'] and inputs['W_arm']==binding['W_arm'],'Selected condition/state differs')
    require(inputs['h8_manifest']['sha256']==sha(HERE.parent/'implementation_h0/h8_manifest.json'),'H8 original input manifest differs')
    assets.path(inputs['h8_manifest'])
    for key in ('runtime_manifest','membership','bn_buffers','bn_receipt'):
        require(common[{'membership':'membership_sha256','runtime_manifest':'runtime_manifest_sha256','bn_buffers':'BN_buffers_sha256','bn_receipt':'BN_receipt_sha256'}[key]]==inputs[key]['sha256'],'Common source asset differs')
        assets.path(inputs[key])
    admission=assets.read(inputs['admission'])
    require(admission['schema']=='flow_h0_admission_v1' and admission['status']=='READY_AFTER_R13_ALL_EIGHT_AUDIT','Completed R13 admission required')
    quality=assets.read(admission['all_eight_audit_receipt']);execution_audit=assets.read(admission['independent_execution_audit'])
    require(admission['all_eight_audit_receipt']==binding['all_eight_audit_receipt'] and admission['independent_execution_audit']==binding['independent_execution_audit'],'Audited source assets differ')
    validate_prior_audits(quality,execution_audit,common)
    require(quality['ordinary_reference_normality'][inputs['condition']] is True,'No stable R13 ordinary references')
    scope=binding['scientific_scope']
    require(scope==admission['scope'] and scope in ('qualified_ordinary_base','action_only_screening'),'Admission scope differs')
    if scope=='qualified_ordinary_base':
        require(inputs['W_arm'] in ('CC','fine_s2b2') and quality['candidate_gates'][inputs['condition']][inputs['W_arm']]['ordinary_base_quality_pass'] is True,'Qualified base not demonstrated')
    else:require(inputs['W_arm']=='FF','Action-only W must be matched F')
    require(set(inputs['all_eight_results'])=={c+'/'+a for c in ('reference','diversity') for a in ('native','FF','CC','fine_s2b2')},'Incomplete R13 sources')
    audited={(r['condition'],r['arm']):r for r in execution_audit['results'] if r['mode']=='recover'}
    results={}
    for key,asset in inputs['all_eight_results'].items():
        value=assets.read(asset)
        require(value['status']=='ROUND13_COVERAGE_RECOVERY_COMPLETE' and value['completed_optimizer_steps']==200,'Incomplete recovered source')
        require(all(value['binding'][k]==v for k,v in common.items()),'Recovered common W/BN/source binding differs')
        require(asset['sha256']==quality['inputs'][key]['sha256']==audited[tuple(key.split('/'))]['result_sha256'],'R13 audit/source mismatch')
        results[key]=value
    for role,item in inputs['states'].items():
        arm=inputs['W_arm'] if role=='W' else 'native' if role=='N' else 'FF';key=inputs['condition']+'/'+arm
        require(item['result']==inputs['all_eight_results'][key],'Chosen state outside complete audited set')
        result=results[key];assets.path(item['weights'])
        require(item['weights']['sha256']==result['weights_sha256']==audited[(inputs['condition'],arm)]['weights']['sha256'],'Chosen weight bytes differ')
        weight_audit=audited[(inputs['condition'],arm)]['weights']
        require(all(weight_audit[k] is True for k in ('final_state_independently_recomputed','frozen_complement_independently_recomputed','initial_state_reconstructed','BN72_actual_values_match')),'Incomplete independent actual-weight audit')
        require(h0['recovered_state_loads'][role]['state_sha256']==result['final_state_sha256'] and
                h0['recovered_state_loads'][role]['strict_full_state_load'] is True and h0['recovered_state_loads'][role]['common_BN_exact'] is True,'H0 actual loaded state differs')
    return {'scientific_scope':scope,'complete_R13_audits_bound':True,'weights_bytes_bound_to_independent_execution_audit':True,
            'weight_tensors_recomputed_here':False}


def read_prediction_archive(path,frame,names,raw,arrays,masks,numeric,shape=(2,480,640)):
    require(sha(path)==frame['prediction_file_sha256'],'Prediction file bytes differ')
    output={}
    with np.load(path,allow_pickle=False) as archive:
        require(set(archive.files)==set(names),'Prediction archive endpoint set differs')
        for name in names:
            pred=archive[name]
            require(pred.shape==shape and pred.dtype==np.float32 and np.isfinite(pred).all(),'Prediction shape/dtype/finite differs')
            require(raw.payload_sha(pred)==frame['arms'][name]['prediction_sha256'],'Prediction payload SHA differs')
            measured=raw.measure(pred,arrays['gt'],masks)
            metric=frame['arms'][name]['metrics']
            require(metric['target_id']==frame['target_id'] and metric['sequence']==frame['sequence'] and set(metric['regions'])==set(core.REGIONS),'Metric identity/regions differ')
            for region in core.REGIONS:
                for field in ('valid_pixels','error_sum','AEE'):
                    numeric.check(measured[region][field],metric['regions'][region][field],'/'.join((frame['target_id'],name,region,field)))
            output[name]=measured
    return output


def bind_descriptors(frame,assets,h0frame):
    record=frame['predecision_u'];path=assets.path(frame['predecision_u_source'])
    require(sha(path)==record['file_sha256'] and frame['predecision_u_source']['sha256']==record['file_sha256'],'u source bytes differ')
    with np.load(path,allow_pickle=False) as archive:
        require(archive.files==['u'],'Single u array required');u=archive['u']
        require(u.shape==(10,1,30,40,384) and u.dtype==np.float32 and np.isfinite(u).all(),'u shape/dtype/finite differs')
        digest=hashlib.sha256(u.tobytes()).hexdigest()
    require(digest==record['payload_sha256']==h0frame['arms']['CC']['execution']['branch_records'][TARGET]['input_sha256'],'u not bound to original H0 CC prefix')
    require(record['source_action']=='k1_r0c0' and record['full_forward_reused'] is True and record['dtype']=='float32'
            and record['shape_TBHWC']==[10,1,30,40,384],'u capture provenance differs')
    require(json_sha(frame['ordinary_scores'])==frame['ordinary_scores_sha256'],'Ordinary score payload changed')
    cost,audit=frame['ordinary_score_cost'],frame['ordinary_score_actual_audit']
    require(cost['diagnostic_only_extra_psn1_calls']==audit['observed_extra_psn1_calls']==12 and
            cost['diagnostic_only_extra_psn1_elements']==audit['observed_extra_psn1_elements']==3732480 and
            cost['diagnostic_only_extra_psn1_temporal_MACs_nominal']==37324800 and cost['extra_fc_calls']==0,'Ordinary descriptor service differs')
    require(audit['full_model_forwards']==0 and audit['temporary_hook_removed'] is True and audit['call_shapes']==[[10,1,9,9,384]]*12,'Ordinary descriptor hook audit differs')
    require(cost['score_reductions_and_feature_reads_counted_in_cycles'] is False,'Unproved descriptor cycle claim')
    return {'u_file_sha256':sha(path),'u_payload_sha256':digest,'scores_sha256':frame['ordinary_scores_sha256'],
            'scores_recomputed_here':False,'scope':'Scores bound to frozen source, exact same-W predecision u, actual descriptor work; no independent PSN replay.'}


def run(args):
    require(not args.output_dir.exists(),'Fresh output required')
    require(sha(args.inputs)==args.inputs_sha256,'Explicit adapter input SHA differs')
    config=json.loads(args.inputs.read_text());require(config['schema']=='flow_headroom_analysis_io_inputs_v1' and config['stage'] in ('H0','H1'),'Explicit H0 or H1 stage required')
    source_preflight();raw=load_raw_math();assets=Assets(config.get('local_by_sha256',{}))
    members=json.loads((HERE.parent/'implementation_h0/h8_manifest.json').read_text())['members']
    h0_path=assets.path(config['h0_result']);h0=json.loads(h0_path.read_text())
    sources0={n:sha(HERE.parent/'implementation_h0'/n) for n in H0_OWN}
    by_id=validate_h0(h0,members,sources0)
    inputs=assets.read(config['h0_inputs'])
    require(config['h0_inputs']['sha256']==h0['binding']['inputs_sha256'],'H0 input manifest differs')
    admission=bind_admission(inputs,h0,assets)
    h1=None
    if config['stage']=='H1':
        h1=assets.read(config['h1_result']);segments=[assets.read(s) for s in h1['segment_sources']]
        validate_h1(h1,segments,h0,members,{n:sha(HERE.parent/'implementation_h1'/n) for n in H1_OWN},config['h0_result']['sha256'])
        require(h1['finalizer_sha256']==sha(HERE.parent/'implementation_h1/finalize_h1.py'),'H1 finalizer source differs')
        require(h1['common_binding']['W_weights_sha256']==inputs['states']['W']['weights']['sha256'],'H1 W artifact differs')
    # Only after all requested endpoint/segment metadata and source/state closure is complete:
    region_path=assets.path(config['regions']);require(sha(region_path)==REGION_SHA,'Region fit source changed')
    regions=json.loads(region_path.read_text());snapshot=assets.path(config['snapshot_receipt'])
    snapshot_metadata=json.loads(snapshot.read_text())
    require(snapshot_metadata['schema']=='flow_h8_input_snapshot_v1' and
            snapshot_metadata['h8_sha256']==sha(HERE.parent/'implementation_h0/h8_manifest.json'),'H8 raw snapshot identity differs')
    arrays,evidence=raw.load_snapshot(Path(config['snapshot_root']),snapshot,[m['target_id'] for m in members])
    masks={tid:raw.masks_from_original(a['gt'],a['mask'],a['events'],regions['boundary_threshold_pixels'],regions['large_displacement_threshold_pixels']) for tid,a in arrays.items()}
    numeric=raw.NumericAudit();normalized0=[];normalized1=[];descriptors={}
    for frame in h0['frames']:
        tid=frame['target_id'];require(frame['input_evidence']==evidence[tid],'H0 original array SHA mismatch')
        preflight=next(r for r in h0['data_evidence'] if r['target_id']==tid)
        require({'events' if k=='input' else k:v for k,v in preflight['files'].items()}==evidence[tid],'H0 preflight data source differs')
        require(Path(frame['prediction_file']).name==frame['prediction_file'],'H0 prediction basename required')
        path=assets.path({'path':str(h0_path.parent/frame['prediction_file']),'sha256':frame['prediction_file_sha256']})
        arms=read_prediction_archive(path,frame,core.ANCHORS,raw,arrays[tid],masks[tid],numeric)
        normalized0.append({'target_id':tid,'sequence':frame['sequence'],'arms':arms})
    analysis0=core.analyze_h0(normalized0,members)
    for name,summary in analysis0['summaries'].items():
        for scope,metrics in summary.items():
            recorded=h0['summaries'][name] if scope=='overall' else h0['summaries'][name]['per_sequence'][scope]
            for region,fields in metrics.items():
                for field,value in fields.items():numeric.check(value,recorded['regions'][region][field],'/'.join(('H0-summary',name,scope,region,field)))
    analysis1=None
    if h1 is not None:
        for frame in h1['frames']:
            tid=frame['target_id'];require(frame['input_evidence']==evidence[tid],'H1 original array SHA mismatch')
            path=assets.path(frame['prediction_source'])
            require(frame['prediction_source']['sha256']==frame['prediction_file_sha256'],'H1 archive source mismatch')
            arms=read_prediction_archive(path,frame,[core.action((i,)) for i in range(12)]+[core.action(p) for p in core.PAIRS],raw,arrays[tid],masks[tid],numeric)
            descriptors[tid]=bind_descriptors(frame,assets,by_id[tid])
            normalized1.append({'target_id':tid,'sequence':frame['sequence'],'arms':arms,'ordinary_scores':frame['ordinary_scores']})
        analysis1=core.analyze_h1(normalized0,normalized1,members)
    report={'schema':'flow_headroom_independent_artifact_metric_analysis_v1','status':'PASS','stage':config['stage'],
            'adapter_sha256':sha(__file__),'inputs_sha256':args.inputs_sha256,'pure_freeze_sha256':sha(HERE/'freeze_v1.json'),
            'io_freeze_sha256':sha(HERE/'io_freeze_v1.json'),
            'admission':admission,'source_array_sha256':evidence,'checked_assets':assets.checked,'descriptors':descriptors,
            'metric_reproduction':{'atol':raw.ATOL,'rtol':raw.RTOL,'comparisons':numeric.count,'maximum_absolute_difference':numeric.maximum_absolute,
                                   'maximum_relative_difference':numeric.maximum_relative,'maximum_absolute_location':numeric.maximum_location},
            'counts':{'frames':8,'original_arrays':24,'prediction_payloads':688 if h1 else 64,'full_forward_ledger':694 if h1 else 69,
                      'u_payloads':8 if h1 else 0,'extra_PSN1_calls':96 if h1 else 0},
            'H0':analysis0,'H1':analysis1,
            'scope':'Actual source/ledger/artifact binding and independent raw metric recomputation. Source-bound ordinary scores not replayed; tensors/RTL/cycles not independently measured here.'}
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'independent_analysis.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'status':'PASS','counts':report['counts'],'metric_reproduction':report['metric_reproduction']},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--inputs-sha256',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args())


if __name__=='__main__':main()
