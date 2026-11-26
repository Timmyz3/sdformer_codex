#!/usr/bin/env python3
"""Only generated metadata and temporary synthetic NPZ; never real endpoints."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import analysis_core as core
import io_adapter as io

FAKE='1'*64


def members():
    return [{'target_id':'synthetic_%d'%i,'sequence':'seq_%d'%i} for i in range(8)]


def ex(rows):
    return {'summed_fc1_rows':rows,'summed_fc2_rows':rows,
            'head_records':[{'shape':[1,2,480,640],'dtype':'torch.float32','finite':True,'sha256':FAKE} for _ in range(4)],
            'branch_records':{n:{'shape':shape,'input_sha256':FAKE,'output_sha256':FAKE} for n,shape in io.BRANCHES.items()}}


def h0fixture():
    rows=members();plan=io.h0_plan(rows)
    frames={r['target_id']:dict(r,arms={}) for r in rows};records=[]
    totals={'native':558000,'FF':558000,'CC':153600,'P':164200,'F12':162240,'T8':155560,'matched_N':558000,'matched_F':558000}
    for item in plan:
        execution=ex(609600 if item['mode']=='dual_reference' else totals[item['anchor']])
        records.append(dict(item,execution=execution))
        if item['mode']=='actual':frames[item['target_id']]['arms'][item['anchor']]={'execution':execution,'state':item['state'],'prediction_sha256':FAKE}
    value={'schema':'flow_h0_action_screen_v1','status':'H0_69_FORWARD_COMPLETE','inference_executed':True,
           'full_forward_count':69,'completed_targets':8,'forward_records':records,'frames':list(frames.values()),
           'training_performed':False,'new_BN_calibration_performed':False,
           'all_recovered_states_unchanged':True,'BN_buffers_unchanged':True,'original_modules_restored':True,'all_hooks_removed':True,
           'binding':{'source_sha256':{},'H8_ids':[r['target_id'] for r in rows],'inputs_sha256':FAKE,
             'cpu_checks_sha256':io.sha(io.HERE.parent/'implementation_h0/cpu_results.json'),
             'common':{'membership_sha256':io.MEMBER_SHA,'BN_buffers_sha256':FAKE},'region_sha256':io.REGION_SHA,
             'GT_used_for_actions':False,'total_full_forwards':69,'forward_plan':plan},
           'first_frame_dual_audits':{n:{} for n in ('CC','FF','P','F12','T8')},
           'data_evidence':[{'target_id':r['target_id']} for r in rows],
           'recovered_state_loads':{'W':{'state_sha256':FAKE,'strict_full_state_load':True,'common_BN_exact':True}},
           'BN_scope':{'BN_signature_before_inference':FAKE}}
    return rows,value


def h1fixture():
    rows,h0=h0fixture();plan=io.h1_plan(rows);segments=[]
    by_id={f['target_id']:f for f in h0['frames']}
    stable={'source_sha256':{},'h0_binding':h0['binding'],'h0_result_sha256':FAKE,
            'H8_ids':[r['target_id'] for r in rows],'H8_sequences':[r['sequence'] for r in rows],
            'BN_buffers_sha256':FAKE,'GT_used_to_generate_masks':False,
            'cpu_checks_sha256':io.sha(io.HERE.parent/'implementation_h1/cpu_results.json'),
            'h0_cpu_checks_sha256':h0['binding']['cpu_checks_sha256'],'h0_inputs_sha256':FAKE,
            'W_descriptor':{'final_state_sha256':FAKE}}
    for s in range(1,5):
        part=[p for p in plan if p['segment']==s];rs=rows[2*(s-1):2*s]
        frames={r['target_id']:dict(r,arms={},H0_CC_reference={'result_sha256':FAKE,
            'execution':by_id[r['target_id']]['arms']['CC']['execution'],'prediction_sha256':FAKE}) for r in rs};records=[]
        for item in part:
            execution=ex(609600 if item['mode']=='dual_reference' else 153600+720*item['k'])
            execution.update(action=item['action'],tiles=item['tiles'],mode=item['mode'])
            records.append(dict(item,execution=execution))
            if item['mode']=='actual':frames[item['target_id']]['arms'][item['action']]={'execution':execution,'prediction_sha256':FAKE}
        segments.append({'schema':'flow_h1_finite_action_segment_v1','status':'H1_SEGMENT_COMPLETE','inference_executed':True,
            'binding':dict(stable,segment=s,segment_ids=[r['target_id'] for r in rs],segment_forward_budget=len(part),forward_plan=part,wall_minutes=30),
            'recovered_W_unchanged':True,'BN_buffers_unchanged':True,'original_modules_restored':True,'all_hooks_removed':True,
            'training_performed':False,'new_BN_calibration_performed':False,'recovered_state_load':h0['recovered_state_loads']['W'],
            'BN_scope':h0['BN_scope'],'full_forward_count':len(part),'forward_records':records,'frames':list(frames.values())})
    value={'schema':'flow_h1_625_forward_complete_v1','status':'H1_625_FORWARD_COMPLETE','full_forward_count':625,
           'actual_forwards':624,'dual_forwards':1,'completed_targets':8,'common_binding':stable,'GT_used_to_generate_masks':False,
           'frames':[f for s in segments for f in s['frames']],'forward_records':[r for s in segments for r in s['forward_records']]}
    return rows,h0,value,segments


class IOChecks(unittest.TestCase):
    def test_prior_audit_full_scope_and_initial_pairing(self):
        common={'checkpoint_sha256':FAKE,'BN_buffers_sha256':FAKE}
        rows=[{'condition':c,'mode':m,'arm':a} for c in ('reference','diversity') for m in ('smoke','recover') for a in ('native','FF','CC','fine_s2b2')]
        keys=[r['condition']+'/'+r['arm'] for r in rows if r['mode']=='recover']
        quality={'schema':'flow_round13_coverage_quality_analysis_v1','script_sha256':'b1997b55deea309088388a011b83272bafe438bedc43db5ee4440655bc5034ad',
            'analysis_protocol_sha256':'5ec51175fbf18b9d2f4fab329247b011700f4cf109b50bf73e905708698c2d7b',
            'state_layout_count':40,'evaluation_endpoint_count':320,'region_count':16,'full_forward_count':1964,
            'initial_state_all_eight_exact':True,'initial_same_layout_prediction_SHA_and_metrics_cross_condition_exact':True,
            'inputs':{k:{} for k in keys},'prediction_artifact_audits':{k:{'prediction_files':16,'prediction_payloads':40} for k in keys},
            'common_contract':dict(common,initial_state_sha256=FAKE)}
        execution={'schema':'flow_round13_independent_execution_audit_v1','status':'PASS',
            'auditor_sha256':'dd57f9f315e1c5f38961208496a37f46db577c47b8507029a55b5bc216d8c8e8',
            'weights_count':8,'prediction_count':320,'archive_count':128,'full_forwards':2048,'optimizer_updates':1640,
            'scope':{'conditions':['reference','diversity'],'modes':['smoke','recover'],'runs':16},'results':rows,
            'paired_initial_predictions_equal':True,'common_initial_model_sha256':FAKE,'original_checkpoint_sha256':FAKE,'new_BN_actual_sha256':FAKE}
        io.validate_prior_audits(quality,execution,common)
        for change in ('duplicate','scope','count','initial','missing'):
            q,e=copy.deepcopy((quality,execution))
            if change=='duplicate':e['results'][0]=copy.deepcopy(e['results'][1])
            elif change=='scope':e['scope']['modes']=['recover']
            elif change=='count':q['state_layout_count']=39
            elif change=='initial':q['initial_state_all_eight_exact']=False
            else:del q['prediction_artifact_audits'][keys[0]]
            with self.assertRaises(ValueError):io.validate_prior_audits(q,e,common)

    def test_h0_complete_ledger_and_tail_counts(self):
        rows,value=h0fixture();io.validate_h0(value,rows,{})
        self.assertEqual(len(io.h0_plan(rows)),69)
        for mutator in (
            lambda x:x.update(status='H0_RUNNING'),
            lambda x:x['forward_records'][0]['execution'].update(summed_fc1_rows=1),
            lambda x:x['frames'][0]['arms']['P'].update(state='N'),
            lambda x:x['forward_records'][2]['execution']['head_records'][0].update(sha256='2'*64)):
            bad=copy.deepcopy(value);mutator(bad)
            with self.assertRaises(ValueError):io.validate_h0(bad,rows,{})

    def test_h1_full_actual_enumeration_and_admission(self):
        rows,h0,value,segments=h1fixture();io.validate_h1(value,segments,h0,rows,{},FAKE)
        self.assertEqual(len(io.h1_plan(rows)),625)
        self.assertEqual(sum(x['mode']=='actual' for x in io.h1_plan(rows)),624)
        for change in ('prefix','source','state','rows','aggregate'):
            v,s=copy.deepcopy((value,segments))
            if change=='prefix':s[0]['forward_records'][0]['execution']['branch_records'][io.TARGET]['input_sha256']='2'*64
            elif change=='source':v['common_binding']['source_sha256']={'wrong':'2'*64}
            elif change=='state':s[0]['recovered_state_load']['state_sha256']='2'*64
            elif change=='rows':s[0]['forward_records'][0]['execution']['summed_fc2_rows']=0
            else:v['frames']=v['frames'][:7]
            with self.assertRaises(ValueError):io.validate_h1(v,s,h0,rows,{},FAKE)

    def test_prefix_accepts_downstream_change_only(self):
        cc=ex(153600);changed=copy.deepcopy(cc)
        last=list(io.BRANCHES)[-1];changed['branch_records'][last]['input_sha256']='2'*64
        io.prefix(changed,cc)
        changed['branch_records'][io.PREFIX[0]]['output_sha256']='2'*64
        with self.assertRaises(ValueError):io.prefix(changed,cc)

    def test_relocation_requires_exact_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'asset';p.write_bytes(b'synthetic');digest=io.sha(p)
            assets=io.Assets({digest:str(p)})
            self.assertEqual(assets.path({'path':'/unavailable/old','sha256':digest}),p)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):assets.path({'path':str(p),'sha256':digest})

    def test_raw_prediction_recompute_and_tamper(self):
        raw=io.load_raw_math();gt=np.zeros((2,5,6),dtype=np.float32);pred=gt.copy();pred[0]=1
        events=np.zeros((10,5,6),dtype=np.float32);valid=np.ones((5,6),dtype=bool)
        masks=raw.masks_from_original(gt,valid,events,1.,1.)
        measured=raw.measure(pred,gt,masks)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'synthetic.npz';np.savez(p,CC=pred)
            frame={'target_id':'synthetic','sequence':'synthetic','prediction_file_sha256':io.sha(p),
                   'arms':{'CC':{'prediction_sha256':raw.payload_sha(pred),'metrics':{'target_id':'synthetic','sequence':'synthetic','regions':measured}}}}
            out=io.read_prediction_archive(p,frame,['CC'],raw,{'gt':gt},masks,raw.NumericAudit(),shape=(2,5,6))
            self.assertEqual(out['CC']['all_valid']['AEE'],1.)
            frame['arms']['CC']['metrics']['regions']['all_valid']['AEE']=2.
            with self.assertRaises(ValueError):io.read_prediction_archive(p,frame,['CC'],raw,{'gt':gt},masks,raw.NumericAudit(),shape=(2,5,6))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IOChecks))
    print(json.dumps({'schema':'flow_headroom_io_synthetic_checks_v1','synthetic_only':True,'tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'pass':result.wasSuccessful(),
        'real_quality_read':False,'source_sha256':{n:io.sha(io.HERE/n) for n in ('io_adapter.py','test_io_cpu.py')}},indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
