#!/usr/bin/env python3
"""Verify four complete H1 segments; no forward, fit, ranking or oracle claim."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import h1_core as core
require,sha=core.require,core.sha


def validate_segments(values,h0_result,h0):
    require(len(values)==4,'All4 complete segments required')
    by_segment={v['binding']['segment']:v for v in values}
    require(set(by_segment)=={1,2,3,4},'Duplicate/missing H1 segment')
    baseline=by_segment[1]['binding']
    different={'segment','segment_ids','segment_forward_budget','forward_plan','wall_minutes'}
    stable={k:v for k,v in baseline.items() if k not in different}
    rows=[{'target_id':identity,'sequence':seq} for identity,seq in zip(baseline['H8_ids'],baseline['H8_sequences'])]
    require(len(rows)==8 and len({r['sequence'] for r in rows})==8,'Fixed eight one-per-sequence H8 required')
    require(h0_result.get('schema')=='flow_h0_action_screen_v1' and h0_result.get('status')=='H0_69_FORWARD_COMPLETE' and
            h0_result['binding']==baseline['h0_binding'],'Actual original H0 receipt/binding differs')
    h0_frames={f['target_id']:f for f in h0_result['frames']}
    require(set(h0_frames)==set(baseline['H8_ids']),'Actual H0 member coverage differs')
    branches={'sttmultires_unet.encoders.swin3d.layers.%d.swin_blocks.%d.mlp' % (stage,block):
              [10,1,120,160,96] if stage==0 else [10,1,30,40,384] for stage,depth in ((0,2),(2,6)) for block in range(depth)}
    records=[];frames=[]
    for segment in (1,2,3,4):
        value=by_segment[segment];binding=value['binding'];expected=core.plan(rows,segment)
        require(value.get('schema')=='flow_h1_finite_action_segment_v1' and value.get('status')=='H1_SEGMENT_COMPLETE'
                and value.get('inference_executed') is True,'Incomplete/unexecuted H1 segment')
        require({k:v for k,v in binding.items() if k not in different}==stable,'Segment common source/W/BN/H8/H0 differs')
        require(binding['forward_plan']==expected and binding['segment_forward_budget']==len(expected),'Segment fixed enumeration differs')
        require(value.get('full_forward_count')==len(expected) and value.get('completed_targets')==2,'Segment counts differ')
        require(value.get('training_performed') is False and value.get('new_BN_calibration_performed') is False,'Unexpected training/calibration')
        require(all(value.get(k) is True for k in ('recovered_W_unchanged','BN_buffers_unchanged','original_modules_restored','all_hooks_removed')),
                'Segment closure failed')
        require(value['recovered_state_load']['state_sha256']==binding['W_descriptor']['final_state_sha256'] and
                value['recovered_state_load']['strict_full_state_load'] is True and value['recovered_state_load']['common_BN_exact'] is True,
                'Segment actual recovered W differs')
        require(len(value['forward_records'])==len(expected),'Forward ledger incomplete')
        for observed,item in zip(value['forward_records'],expected):
            require(all(observed[k]==v for k,v in item.items()),'Forward tuple/order/identity differs')
            ex=observed['execution'];rows_expected=609600 if item['mode']=='dual_reference' else 153600+720*item['k']
            require(ex['summed_fc1_rows']==ex['summed_fc2_rows']==rows_expected,'Actual complete-tile producer rows differ')
            require(ex['tiles']==item['tiles'] and ex['action']==item['action'] and ex['mode']==item['mode'],'Execution action differs')
            require(len(ex['head_records'])==4 and all(h['finite'] is True and h['shape']==[1,2,480,640] and h['dtype']=='torch.float32' for h in ex['head_records']),
                    'Missing finite full-resolution head audit')
            require(isinstance(ex.get('branch_records'),dict) and set(ex['branch_records'])==set(branches) and all(ex['branch_records'][n]['shape']==shape for n,shape in branches.items()),
                    'All eight native-shape branch records required')
            for record in ex['branch_records'].values():
                require(all(isinstance(record[key],str) and len(record[key])==64 and set(record[key])<=set('0123456789abcdef') for key in ('input_sha256','output_sha256')),
                        'Valid complete branch SHA fields required')
            recomputed=h0.check_prefix(ex,h0_frames[item['target_id']]['arms']['CC']['execution'])
            require(ex['same_H0_CC_prefix_audit']['same_target_input'] is True and ex['same_H0_CC_prefix_audit']['preceding4_selected_MLP_branches_exact'] is True,
                    'Same saved H0 CC prefix not established')
        pair=value['first_pair_dual_audit']
        if segment==1:
            require(pair and pair.get('head_count')==4 and all(pair.get(k) is True for k in ('all_flow_heads_exact','all8_branch_inputs_exact','all8_branches_exact')),
                    'First pair full-head/branch dual audit failed')
            dual_record=next(r for r in value['forward_records'] if r['mode']=='dual_reference')
            actual_record=next(r for r in value['forward_records'] if r['mode']=='actual' and r['target_id']==dual_record['target_id'] and r['action']==dual_record['action'])
            require(all(actual_record['execution'][key]==dual_record['execution'][key] for key in ('head_records','branch_records')),
                    'First pair recorded full heads/branches differ across actual/dual')
        else:require(pair is None,'Unexpected extra dual')
        wanted=rows[2*(segment-1):2*segment]
        require([f['target_id'] for f in value['frames']]==[r['target_id'] for r in wanted]==binding['segment_ids'],'Fixed two-frame coverage differs')
        for frame,row in zip(value['frames'],wanted):
            require(frame['sequence']==row['sequence'] and set(frame['arms'])=={a['action'] for a in core.actions()},'Missing actual finite action result')
            require(frame['H0_CC_reference']['result_sha256']==binding['h0_result_sha256'],'Baseline CC reference differs')
            original=h0_frames[row['target_id']]['arms']['CC']
            require(frame['H0_CC_reference']['execution']==original['execution'] and frame['H0_CC_reference']['prediction_sha256']==original['prediction_sha256'],
                    'Saved CC reference differs from actual bound H0 source')
            for name,arm in frame['arms'].items():
                execution=[r['execution'] for r in value['forward_records'] if r['target_id']==row['target_id'] and r['action']==name and r['mode']=='actual']
                require(execution==[arm['execution']],'Retained prediction/execution differs')
                require(arm['prediction_sha256']==arm['execution']['head_records'][-1]['sha256'],'Final prediction differs from recorded full forward head')
                require(len(arm['metrics']['regions'])==16,'All16 region results must be retained')
        records.extend(value['forward_records']);frames.extend(value['frames'])
    require(len(records)==625 and [r['global_forward_id'] for r in records]==list(range(1,626)),'Complete625 ledger required')
    return {'schema':'flow_h1_625_forward_complete_v1','status':'H1_625_FORWARD_COMPLETE','common_binding':stable,
            'full_forward_count':625,'actual_forwards':624,'dual_forwards':1,'completed_targets':8,'frames':frames,
            'forward_records':records,'first_pair_dual_audit':by_segment[1]['first_pair_dual_audit'],
            'flow_specific_increment_claim':False,'hardware_benefit_claim':False,'GT_used_to_generate_masks':False,
            'scope':'All twelve singleton and66 actual pair interventions per H8 frame, conditional on one W/BN/CC background. No mask ranking, additive oracle, global optimum, fitted predictor or held-out claim.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segments-manifest',type=Path,required=True);parser.add_argument('--segments-manifest-sha256',required=True)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--cpu-checks',type=Path,required=True)
    parser.add_argument('--h0-result',type=Path,required=True);parser.add_argument('--h0-result-sha256',required=True)
    a=parser.parse_args();require(not a.output_dir.exists(),'Fresh aggregate output required')
    require(sha(a.segments_manifest)==a.segments_manifest_sha256,'Explicit segment manifest SHA differs')
    manifest=json.loads(a.segments_manifest.read_text());require(manifest.get('schema')=='flow_h1_four_segments_v1' and len(manifest['segments'])==4,'Four explicit segment sources required')
    h0,h0runner,runtime=core.load_h0()
    import run_h1
    cpu_sha=runtime[3].checked_cpu_receipt(a.cpu_checks,[core.HERE/n for n in run_h1.OWN])
    values=[];sources=[];artifacts=[]
    for item in manifest['segments']:
        path=h0runner.asset(item);values.append(json.loads(path.read_text()));sources.append(item)
    require(sha(a.h0_result)==a.h0_result_sha256,'Actual H0 source bytes SHA differs')
    h0_result=json.loads(a.h0_result.read_text())
    require(all(v['binding']['h0_result_sha256']==a.h0_result_sha256 for v in values),'Segments refer to different H0 result')
    result=validate_segments(values,h0_result,h0)
    h0_artifacts=core.validate_prediction_artifacts(a.h0_result,h0_result,h0.ANCHORS)
    require(result['common_binding']['source_sha256']=={n:sha(core.HERE/n) for n in run_h1.OWN} and
            result['common_binding']['cpu_checks_sha256']==cpu_sha,'Frozen current H1 implementation differs from segments')
    descriptor_artifacts=[]
    for item,value in zip(sources,values):
        artifacts.extend(core.validate_prediction_artifacts(Path(item['path']),value,[a['action'] for a in core.actions()]))
        descriptor_artifacts.extend(core.validate_descriptor_artifacts(Path(item['path']),value,h0.TARGET))
    require(sum(a['predictions'] for a in artifacts)==624 and len(artifacts)==8,'All624 retained prediction payloads required')
    by_id={item['target_id']:item for item in artifacts}
    for frame in result['frames']:frame['prediction_source']={'path':by_id[frame['target_id']]['path'],'sha256':by_id[frame['target_id']]['sha256']}
    descriptors_by_id={item['target_id']:item for item in descriptor_artifacts}
    require(len(descriptors_by_id)==len(descriptor_artifacts)==8,'All8 predecision u/ordinary descriptor artifacts required')
    for frame in result['frames']:frame['predecision_u_source']={'path':descriptors_by_id[frame['target_id']]['path'],'sha256':descriptors_by_id[frame['target_id']]['sha256']}
    result.update(segment_sources=sources,prediction_artifact_audits=artifacts,segments_manifest_sha256=a.segments_manifest_sha256,
                  descriptor_artifact_audits=descriptor_artifacts,h0_prediction_artifact_audits=h0_artifacts,
                  h0_result_source={'path':str(a.h0_result),'sha256':a.h0_result_sha256},
                  additional_descriptor_work={'observed_extra_PSN1_calls':96,'observed_extra_PSN1_elements':29859840,
                      'PSN1_temporal_MACs_nominal':298598400,'full_model_forwards':0,'cycle_or_runtime_benefit_claim':False},
                  finalizer_sha256=sha(__file__),cpu_checks_sha256=cpu_sha)
    a.output_dir.mkdir(parents=True,exist_ok=False);runtime[3].write(a.output_dir/'results.json',result)
    print(json.dumps({'status':result['status'],'full_forward_count':625,'prediction_payloads':624}),flush=True)


if __name__=='__main__':main()
