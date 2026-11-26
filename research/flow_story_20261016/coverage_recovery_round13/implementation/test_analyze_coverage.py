#!/usr/bin/env python3
"""Synthetic completed-result tests; never reads new check predictions or quality."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import analyze_coverage as a
HERE=a.HERE
MEMBER_PATH=HERE.parent/'data_protocol/manifest_v1.json'
MEMBER=json.loads(MEMBER_PATH.read_text())
MEASUREMENTS=[]


def metric(target,sequence,value):
    counts={region:2 for region in a.REGIONS}
    counts.update(all_valid=4,voxel_nonzero=3,voxel_zero=1,L_voxel_zero=0)
    return {'target_id':target,'sequence':sequence,'regions':{
        region:{'valid_pixels':n,'error_sum':value*n,'AEE':value if n else None} for region,n in counts.items()}}


def synthetic():
    common={k:'pin-'+k for k in a.COMMON_FIELDS}
    common.update(recipe={'synthetic':True},membership_audit={'synthetic':True},allowed_parameter_count=7252064)
    bn={'scope':'MLP_R13_COMMON_CMA176','BN_count':78,'MLP_BN_count':24,'old_CMA_loaded':False,
        'frozen_BN_names':['private'+str(i) for i in range(24)],'MLP_BN_names':['private'+str(i) for i in range(24)],
        'current_stat_BN_names':['external'+str(i) for i in range(54)],'new_buffers_sha256':common['BN_buffers_sha256'],
        'new_receipt_sha256':common['BN_receipt_sha256']}
    raw={}
    for condition in a.CONDITIONS:
        for arm in a.ARMS:
            budget=244 if arm=='native' else 246
            binding=dict(common,condition=condition,arm=arm,mode='recover',
                fit_ids=[r['target_id'] for r in MEMBER['train_order_'+condition]],check_ids=[r['target_id'] for r in MEMBER['check_members']],
                expected_training_forwards=200,expected_audit_forwards=4 if arm=='native' else 6,expected_evaluation_forwards=40)
            z={'schema':'flow_round13_coverage_recovery_v1','status':'ROUND13_COVERAGE_RECOVERY_COMPLETE','condition':condition,'arm':arm,'mode':'recover',
               'training_performed':True,'completed_optimizer_steps':200,'binding':binding,
               'steps':[{k:r[k] for k in ('step','sequence','target_id')} for r in MEMBER['train_order_'+condition]],
               'frozen_state_unchanged':True,'BN_buffers_unchanged':True,'original_modules_restored':True,'all_hooks_removed':True,
               'smoke_weights_discarded':False,'weights_file':'weights.pt','weights_sha256':'a'*64,
               'initial_numerical_pair':{'all_flow_outputs_exact':True,'actual_dual_all_flows_branches_exact':True},
               'final_numerical_pair':{'all_flow_outputs_exact':True,'actual_dual_all_flows_branches_exact':True},
               'initial_model_state_sha256':'b'*64,'BN_scope':copy.deepcopy(bn),'full_forward_count':budget,
               'forward_records':[{'forward_id':i+1,'purpose':'training' if i<200 else 'audit_or_evaluation'} for i in range(budget)],
               'hardware_benefit_claim':False,'flow_specific_increment_claim':False,'evaluations':{}}
            for label,modes in (('initial',(arm,)),('endpoint',a.ARMS)):
                frames=[]
                for i,r in enumerate(MEMBER['check_members']):
                    value=1.+i*.05
                    if label=='endpoint':value*=1.01 if condition=='reference' else .99
                    frame={'target_id':r['target_id'],'sequence':r['sequence'],'input_evidence':{'events':'input-'+r['target_id']},'arms':{}}
                    for mode in modes:
                        signature=hashlib.sha256((label+mode+r['target_id']+('' if label=='initial' else condition+arm)).encode()).hexdigest()
                        frame['arms'][mode]={'prediction_sha256':signature,'metrics':metric(r['target_id'],r['sequence'],value)}
                    frames.append(frame)
                z['evaluations'][label]={'frames':frames}
            raw[(condition,arm)]=z
    return raw


class Checks(unittest.TestCase):
    def test_all40_states_all320_payloads_four_families_and_paired_differences(self):
        raw=synthetic();common=a.validate_completion(raw,MEMBER);result=a.analyze_completed(raw,MEMBER,common)
        self.assertEqual((result['state_layout_count'],result['evaluation_endpoint_count'],result['region_count']),(40,320,16))
        self.assertEqual(result['full_forward_count'],1964)
        self.assertTrue(all(g['ordinary_base_quality_pass'] for c in result['candidate_gates'].values() for g in c.values()))
        self.assertTrue(all(result['ordinary_reference_normality'].values()))
        self.assertEqual(len(result['diversity_minus_reference_all_states']),20)
        row=result['diversity_minus_reference_all_states']['CC/FF']['change']['overall']['all_valid']
        self.assertAlmostEqual(row['frame_mean_AEE_difference'],-.0235)
        self.assertAlmostEqual(row['frame_mean_AEE_relative_change'],.99/1.01-1)
        self.assertEqual(result['own_layout_recovery']['diversity']['CC']['members']['wins'],8)
        self.assertEqual(result['own_layout_recovery']['reference']['native']['members']['losses'],8)
        self.assertIsNone(result['summaries']['reference']['CC/CC']['overall']['L_voxel_zero']['frame_mean_AEE'])
        self.assertIn('D−R',a.render_markdown(result))
        MEASUREMENTS.append({'states':40,'predictions':320,'regions':16,'formal_forwards':1964,'all_condition_gates':True,'paired_all20_states':True})

    def test_incomplete_missing_common_bn_state_and_identity_mismatches_rejected_before_quality(self):
        for mutation in ('missing','incomplete','state','bn','runtime','order','forward'):
            raw=synthetic();z=raw[('diversity','CC')]
            if mutation=='missing':raw.pop(('diversity','CC'))
            elif mutation=='incomplete':z['status']='RUNNING'
            elif mutation=='state':z['initial_model_state_sha256']='c'*64
            elif mutation=='bn':z['BN_scope']['new_buffers_sha256']='wrong'
            elif mutation=='runtime':z['binding']['runtime_manifest_sha256']='wrong'
            elif mutation=='order':z['steps'][0]['target_id']='wrong'
            else:z['full_forward_count']-=1
            # Remove every quality section: rejection must precede quality access.
            for value in raw.values():value.pop('evaluations')
            with self.assertRaises(ValueError):a.validate_completion(raw,MEMBER)

    def test_initial_prediction_same_layout_and_support_rejection(self):
        for mutation in ('sha','count','input','negative','region','initial_metric'):
            raw=synthetic();z=raw[('diversity','CC')]
            f=z['evaluations']['initial']['frames'][0];v=f['arms']['CC']
            if mutation=='sha':v['prediction_sha256']='d'*64
            elif mutation=='count':v['metrics']['regions']['B']['valid_pixels']+=1
            elif mutation=='input':f['input_evidence']['events']='different'
            elif mutation=='negative':v['metrics']['regions']['B']['error_sum']=-1
            elif mutation=='region':v['metrics']['regions'].pop('B_unknown')
            else:
                v['metrics']['regions']['B']['error_sum']+=.2
                v['metrics']['regions']['B']['AEE']+=.1
            with self.assertRaises(ValueError):a.analyze_completed(raw,MEMBER,a.validate_completion(raw,MEMBER))

    def test_v2_preservation_gate_and_sequence_specific_failure_not_hidden(self):
        raw=synthetic();z=raw[('diversity','CC')]
        for f in z['evaluations']['endpoint']['frames']:
            if f['sequence']==MEMBER['check_members'][0]['sequence']:
                for region,v in f['arms']['FF']['metrics']['regions'].items():
                    if v['valid_pixels']:v['AEE']*=2;v['error_sum']*=2
        result=a.analyze_completed(raw,MEMBER,a.validate_completion(raw,MEMBER))
        gate=result['candidate_gates']['diversity']['CC']
        self.assertTrue(all(v['pass'] for v in gate['matched_recovered_references'].values()))
        self.assertFalse(gate['shared_fine_preserved']['FF']['pass'])
        self.assertFalse(gate['ordinary_base_quality_pass'])
        self.assertTrue(result['candidate_gates']['reference']['CC']['ordinary_base_quality_pass'])

    def test_signed_damage_concentration_and_frame_vs_pixel_weighting(self):
        reference=[{'target_id':str(i),'sequence':'s','regions':{'all_valid':{'AEE':1.}}} for i in range(3)]
        candidate=copy.deepcopy(reference)
        for f,v in zip(candidate,(.5,.5,3.)):f['regions']['all_valid']['AEE']=v
        value=a.member_changes(candidate,reference)
        self.assertEqual((value['wins'],value['losses']),(2,1))
        self.assertEqual(value['per_member'][2]['signed_gain_share_percent'],200.)
        result=a.aggregate([{'valid_pixels':1,'error_sum':2.,'AEE':2.},{'valid_pixels':9,'error_sum':36.,'AEE':4.}])
        self.assertEqual((result['frame_mean_AEE'],result['pixel_weighted_AEE']),(3.,3.8))

    def test_prediction_artifact_exact_shape_dtype_sha_and_tamper(self):
        # Synthetic dense arrays; no dataset or actual model predictions.
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory);frames={};count=0
            for label,modes in (('initial',('native',)),('endpoint',a.ARMS)):
                frame={'prediction_file':label+'.npz','arms':{}}
                arrays={mode:np.full((2,480,640),i,dtype=np.float32) for i,mode in enumerate(modes)}
                p=directory/frame['prediction_file'];np.savez_compressed(p,**arrays);frame['prediction_file_sha256']=a.sha(p)
                for mode,value in arrays.items():frame['arms'][mode]={'prediction_sha256':hashlib.sha256(value.tobytes()).hexdigest()}
                frames[label]={'frames':[frame]}
            z={'arm':'native','evaluations':frames}
            self.assertEqual(a.validate_predictions(directory/'results.json',z),{'prediction_files':2,'prediction_payloads':5})
            frames['endpoint']['frames'][0]['arms']['CC']['prediction_sha256']='f'*64
            with self.assertRaisesRegex(ValueError,'payload SHA'):a.validate_predictions(directory/'results.json',z)

    def test_analysis_protocol_and_canonical_roles_source_validation(self):
        root=HERE.parents[3]
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory);bp=directory/'bn.npz';bp.write_bytes(b'synthetic source binding only')
            members=[r['target_id'] for r in MEMBER['bn_members']]
            bn={'schema':'flow_round13_common_bn_v1','status':'COMMON_BN_COMPLETE','checkpoint_sha256':a.CHECKPOINT_SHA,
                'membership_sha256':a.sha(MEMBER_PATH),'buffers_sha256':a.sha(bp),'bn_target_ids':members,
                'old_CMA_loaded':False,'GT_used':False,'calibration_forwards':176,'calibration_bn_count':78,
                'retained_mlp_bn_count':24,'BN_scope':'MLP_R13_COMMON_CMA176','calibration_definition':'all78_reset_CMA_train_nonBN_eval'}
            region={'fit_target_ids':[r['target_id'] for r in MEMBER['train_members_diversity']],
                    'canonical_manifest_sha256':MEMBER['source_sha256']['canonical'],'auxiliary_roles_sha256':MEMBER['source_sha256']['roles'],
                    'boundary_quantile':.95,'large_quantile':.90,'boundary_threshold_pixels':1.,'large_displacement_threshold_pixels':10.}
            bnp=directory/'bn.json';bnp.write_text(json.dumps(bn));rp=directory/'regions.json';rp.write_text(json.dumps(region))
            runtime={'schema':'flow_round13_runtime_v1','region_sha256':a.sha(rp)};rt=directory/'runtime.json';rt.write_text(json.dumps(runtime))
            paths={'membership':MEMBER_PATH,'runtime_manifest':rt,'bn_receipt':bnp,'bn_buffers':bp,'regions':rp,
                   'canonical':root/'research/flow_story_20261011/preflight/data_protocol/artifacts/common_targets.json',
                   'roles':root/'research/flow_story_20261011/preflight/data_protocol/artifacts/train_auxiliary_roles.csv'}
            inputs={'schema':'flow_round13_analysis_inputs_v1',**{k:{'path':str(p),'sha256':a.sha(p)} for k,p in paths.items()}}
            self.assertEqual(a.validate_input_protocol(inputs)[0],MEMBER)
            bn['GT_used']=True;bnp.write_text(json.dumps(bn));inputs['bn_receipt']['sha256']=a.sha(bnp)
            with self.assertRaisesRegex(ValueError,'BN semantics'):a.validate_input_protocol(inputs)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    receipt={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'source_sha256':{n:a.sha(HERE/n) for n in ('analyze_coverage.py','test_analyze_coverage.py')},
             'R12_V2_source_sha256':a.R12_V2_SHA,'analysis_protocol_sha256':a.PROTOCOL_SHA,'measurements':MEASUREMENTS,
             'errors':[str(x) for x in result.errors],'failures':[str(x) for x in result.failures],
             'GPU_initialized':False,'scope':'Synthetic metadata, AEE values and prediction arrays only; no real new check predictions/quality, GPU, model or SSH.'}
    (HERE/'analysis_cpu_results.json').write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
