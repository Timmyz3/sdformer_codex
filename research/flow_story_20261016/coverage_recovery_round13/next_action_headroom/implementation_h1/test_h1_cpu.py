#!/usr/bin/env python3
"""H1 synthetic execution/identity/receipt checks; no real data or weights."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import ast
import builtins
import copy
import dis
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
import numpy as np
import torch
import h1_core as h
import run_h1 as runner
import finalize_h1 as finalizer
H0,H0RUN,RUNTIME=h.load_h0();R13,COLLECTOR,PROBE,PARENT,R9,R8=RUNTIME
H0TEST=H0.load_file(H0.HERE/'test_h0_cpu.py','_h1_accepted_h0_toy_helpers')
MERGE=H0TEST.MERGE
ROWS=json.loads((H0.HERE/'h8_manifest.json').read_text())['members']
MEASUREMENTS=[]


def heads():return [{'shape':[1,2,480,640],'dtype':'torch.float32','finite':True,'sha256':'synthetic'} for _ in range(4)]


def branch_records():
    return {'sttmultires_unet.encoders.swin3d.layers.%d.swin_blocks.%d.mlp' % (s,b):
            {'input_sha256':hashlib.sha256(('input_%d_%d' % (s,b)).encode()).hexdigest(),
             'output_sha256':hashlib.sha256(('output_%d_%d' % (s,b)).encode()).hexdigest(),
             'shape':[10,1,120,160,96] if s==0 else [10,1,30,40,384]} for s,depth in ((0,2),(2,6)) for b in range(depth)}


def fake_h0():
    stable=('inputs_sha256','cpu_checks_sha256','source_sha256','common','H8_ids','condition','W_arm','scientific_scope','region_sha256',
            'all_eight_audit_receipt','independent_execution_audit','completed_analysis_admission_audit','independent_execution_admission_audit')
    binding={k:'synthetic_'+k for k in stable};descriptors={k:{'final_state_sha256':'state_'+k} for k in ('W','N','F')}
    prepared={'binding':binding,'descriptors':descriptors,'rows':ROWS,'data_evidence':['synthetic_input_binding']}
    totals={'native':558000,'matched_N':558000,'FF':558000,'matched_F':558000,'CC':153600,'P':164200,'F12':162240,'T8':155560}
    records=[];frames={r['target_id']:{'target_id':r['target_id'],'sequence':r['sequence'],'arms':{}} for r in ROWS}
    for item in H0.forward_plan(ROWS):
        total=609600 if item['mode']=='dual_reference' else totals[item['anchor']]
        ex={'head_records':heads(),'branch_records':branch_records(),'summed_fc1_rows':total,'summed_fc2_rows':total}
        records.append(dict(item,execution=ex))
        if item['mode']=='actual':frames[item['target_id']]['arms'][item['anchor']]={'execution':ex,'state':item['state'],'prediction_sha256':'a'*64}
    pairs={name:{'head_count':4,'all_flow_heads_exact':True,'all8_branch_inputs_exact':True,'all8_branches_exact':True} for name in H0.DUALS}
    value={'schema':'flow_h0_action_screen_v1','status':'H0_69_FORWARD_COMPLETE','full_forward_count':69,'completed_targets':8,'inference_executed':True,
           'binding':copy.deepcopy(binding),'data_evidence':['synthetic_input_binding'],'recovered_state_loads':{
               role:{'state_sha256':d['final_state_sha256'],'strict_full_state_load':True,'common_BN_exact':True} for role,d in descriptors.items()},
           'forward_records':records,'first_frame_dual_audits':pairs,'frames':[frames[r['target_id']] for r in ROWS],
           'training_performed':False,'new_BN_calibration_performed':False,
           **{k:True for k in ('all_recovered_states_unchanged','BN_buffers_unchanged','original_modules_restored','all_hooks_removed')}}
    return value,prepared


def fake_segments():
    values=[];h0value,_=fake_h0();by_id={f['target_id']:f for f in h0value['frames']}
    for segment in (1,2,3,4):
        plan=h.plan(ROWS,segment);selected=ROWS[(segment-1)*2:segment*2]
        frames={r['target_id']:{'target_id':r['target_id'],'sequence':r['sequence'],'arms':{},
                              'H0_CC_reference':{'result_sha256':'H0','execution':by_id[r['target_id']]['arms']['CC']['execution'],'prediction_sha256':'a'*64}} for r in selected}
        records=[]
        for item in plan:
            total=609600 if item['mode']=='dual_reference' else 153600+720*item['k']
            ex={'summed_fc1_rows':total,'summed_fc2_rows':total,'tiles':item['tiles'],'action':item['action'],'mode':item['mode'],
                'head_records':heads(),'branch_records':branch_records(),
                'same_H0_CC_prefix_audit':{'same_target_input':True,'preceding4_selected_MLP_branches_exact':True}}
            records.append(dict(item,execution=ex))
            if item['mode']=='actual':frames[item['target_id']]['arms'][item['action']]={'execution':ex,'prediction_sha256':'synthetic','metrics':{'regions':{str(i):{} for i in range(16)}}}
        binding={'segment':segment,'segment_ids':[r['target_id'] for r in selected],'segment_forward_budget':len(plan),'forward_plan':plan,'wall_minutes':30,
                 'H8_ids':[r['target_id'] for r in ROWS],'H8_sequences':[r['sequence'] for r in ROWS],'W_descriptor':{'final_state_sha256':'W'},
                 'h0_result_sha256':'H0','h0_binding':h0value['binding']}
        value={'schema':'flow_h1_finite_action_segment_v1','status':'H1_SEGMENT_COMPLETE','inference_executed':True,'binding':binding,
               'full_forward_count':len(plan),'completed_targets':2,'training_performed':False,'new_BN_calibration_performed':False,
               'recovered_state_load':{'state_sha256':'W','strict_full_state_load':True,'common_BN_exact':True},'forward_records':records,
               'first_pair_dual_audit':{'head_count':4,'all_flow_heads_exact':True,'all8_branch_inputs_exact':True,'all8_branches_exact':True} if segment==1 else None,
               'frames':[frames[r['target_id']] for r in selected],
               **{k:True for k in ('recovered_W_unchanged','BN_buffers_unchanged','original_modules_restored','all_hooks_removed')}}
        values.append(value)
    return values,h0value


class Tests(unittest.TestCase):
    def test_exact_enumeration_segments_and_independent_budget(self):
        plan=h.plan(ROWS);self.assertEqual(len(plan),625)
        self.assertEqual([len(h.plan(ROWS,s)) for s in (1,2,3,4)],[157,156,156,156])
        masks={tuple(tuple(t) for t in a['tiles']) for a in h.actions()}
        independent={(p,) for p in h.TILES}|{tuple(sorted((p,q))) for p in h.TILES for q in h.TILES if p<q}
        self.assertEqual(masks,independent);self.assertEqual(sum(len(m)==1 for m in masks),12);self.assertEqual(sum(len(m)==2 for m in masks),66)
        self.assertEqual(sum(p['mode']=='dual_reference' for p in plan),1)
        for row in ROWS:self.assertEqual({p['action'] for p in plan if p['target_id']==row['target_id'] and p['mode']=='actual'},{a['action'] for a in h.actions()})
        total=sum(609600 if p['mode']=='dual_reference' else 153600+720*p['k'] for p in plan)
        self.assertEqual(total,96*154320+528*155040+609600)
        MEASUREMENTS.append({'forwards':625,'actual':624,'dual':1,'segments':[157,156,156,156],'rows_per_FC_all625':total,
                             'uncompressed_final_prediction_bytes':624*2*480*640*4,'H0_CC_repeated':0,'GT_mask_generation':False})

    def test_two_real_math_masks_actual_dual_and_exact_saved_CC_prefix(self):
        model,targets,shapes,x,functional=H0TEST.toy()
        shapes={name:(10,1,11,23,4) for name in shapes};x=torch.randn(10,1,11,23,4)
        initial=R13.state_digest(model);bn=H0TEST.base.bn_signature(model,torch)
        cc=H0.forward(RUNTIME,model,functional,targets,shapes,x,'CC',MERGE,torch)
        prior_wrap=PROBE.wrap;prior_target=PROBE.TARGET
        for tiles in (((0,0),),((0,0),(0,1))):
            capture={}
            with patch.object(model,'forward',wraps=model.forward) as model_call:
                actual=h.forward(H0,RUNTIME,model,functional,targets,shapes,x,tiles,MERGE,torch,capture_sink=capture)
                self.assertEqual(MERGE.tensor_sha(capture['u']),cc[3][H0.TARGET])
                scores,cost,audit=h.ordinary_descriptors(capture['u'],targets[H0.TARGET],PROBE,MERGE,torch)
                self.assertEqual(model_call.call_count,1)
                self.assertEqual(len(scores),2);self.assertEqual(cost['diagnostic_only_extra_psn1_calls'],2)
                self.assertEqual(audit['observed_extra_psn1_elements'],2*10*81*4)
                self.assertEqual(audit['full_model_forwards'],0)
            dual=h.forward(H0,RUNTIME,model,functional,targets,shapes,x,tiles,MERGE,torch,mode='dual_reference')
            self.assertTrue(H0.check_pair(actual,dual,torch)['all_flow_heads_exact'])
            self.assertTrue(H0.check_prefix(actual[1],cc[1])['same_target_input'])
            target_mask=actual[1]['layers'][H0.TARGET]['fine_mask']
            self.assertEqual(sum(sum(row) for row in target_mask),len(tiles));self.assertTrue(all(not v for v in target_mask[-1]))
            self.assertFalse(target_mask[0][-1])
            self.assertEqual(actual[1]['summed_fc1_rows'],H0.expected_rows(shapes,'CC',torch)+720*len(tiles))
        self.assertIs(PROBE.wrap,prior_wrap);self.assertEqual(PROBE.TARGET,prior_target)
        self.assertEqual(R13.state_digest(model),initial);self.assertEqual(H0TEST.base.bn_signature(model,torch),bn)
        self.assertTrue(all(model.get_submodule(n) is m for n,m in targets.items()))
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        for tiles in ((),((0,0),(0,0)),((0,1),(0,0)),((1,0),),((0,2),),((0,0),(0,1),(0,2))):
            with self.assertRaises(ValueError):h.validate_tiles(tiles,shapes[H0.TARGET])
        MEASUREMENTS.append({'singleton_pair_actual_dual_exact':True,'same_saved_CC_prefix_exact':True,'tails_stay_coarse':True,
                             'actual_capture_u_matches_CC':True,'descriptor_PSN1_hook_count_verified':True,'no_extra_model_forward_for_scores':True,
                             'state_BN_hooks_historical_globals_unchanged':True,'scope':'Synthetic12MLP fixture,11x23,C4,two synthetic heads, not a real flow model.'})

    def test_mock_production_driver_complete625_and_single_pair_audit(self):
        names=['sttmultires_unet.encoders.swin3d.layers.%d.swin_blocks.%d.mlp' % (s,b) for s,depth in ((0,2),(2,6)) for b in range(depth)]
        ex={'branch_records':{n:{'input_sha256':'i_'+n,'output_sha256':'o_'+n} for n in names}}
        coarse={r['target_id']:copy.deepcopy(ex) for r in ROWS};calls=[];saved=[];pairs=[]
        def infer(item):
            calls.append(item)
            return [torch.zeros(1) for _ in range(4)],copy.deepcopy(ex),{n:torch.zeros(1) for n in names},{n:'i_'+n for n in names}
        for segment in (1,2,3,4):
            records,pair=h.drive(h.plan(ROWS,segment),coarse,infer,lambda item,out:saved.append(item),lambda x,y:H0.check_pair(x,y,torch),H0.check_prefix)
            self.assertTrue(all(r['execution']['same_H0_CC_prefix_audit']['same_target_input'] for r in records))
            if pair:pairs.append(pair)
        self.assertEqual((len(calls),len(saved),len(pairs)),(625,624,1))
        self.assertEqual([r['global_forward_id'] for r in calls],list(range(1,626)))
        MEASUREMENTS.append({'actual_production_driver_mock_calls':625,'saved_actual':624,'dual_pairs':1,'prefix_checks':625})

    def test_completed_h0_inheritance_rejects_changed_W_BN_inputs_ledger(self):
        value,p=fake_h0();self.assertTrue(h.validate_h0_receipt(value,p,H0)['completed_H0_69_bound'])
        for mutation in ('status','W','binding','input','order','rows','dual','frame','closure'):
            bad=copy.deepcopy(value)
            if mutation=='status':bad['status']='H0_RUNNING'
            elif mutation=='W':bad['recovered_state_loads']['W']['state_sha256']='changed'
            elif mutation=='binding':bad['binding']['common']='changed BN/runtime'
            elif mutation=='input':bad['data_evidence']=['other input']
            elif mutation=='order':bad['forward_records'][0]['forward_id']=2
            elif mutation=='rows':bad['forward_records'][0]['execution']['summed_fc1_rows']=1
            elif mutation=='dual':bad['first_frame_dual_audits']['CC']['all_flow_heads_exact']=False
            elif mutation=='frame':bad['frames'][0]['arms']['CC']['state']='N'
            else:bad['BN_buffers_unchanged']=False
            with self.assertRaises(ValueError):h.validate_h0_receipt(bad,p,H0)
        MEASUREMENTS.append({'completed_H0_binding_synthetic_check':True,'H0_mutation_rejections':9,'real_H0_or_R13_results_not_read':True})

    def test_all4_segments_required_no_duplicate_or_wrong_state_or_row_budget(self):
        values,h0value=fake_segments();out=finalizer.validate_segments(values,h0value,H0)
        self.assertEqual((out['full_forward_count'],len(out['frames'])),(625,8))
        for mutation in ('missing','duplicate','W','plan','rows','pair','prefix','mask','region','branch_missing','all_branches_deleted','raw_prefix','CC_reference','dual_output'):
            bad=copy.deepcopy(values)
            if mutation=='missing':bad.pop()
            elif mutation=='duplicate':bad[3]=copy.deepcopy(bad[2])
            elif mutation=='W':bad[1]['binding']['W_descriptor']['final_state_sha256']='other'
            elif mutation=='plan':bad[1]['forward_records'][0]['tiles']=[[0,1]]
            elif mutation=='rows':bad[1]['forward_records'][0]['execution']['summed_fc1_rows']=153600
            elif mutation=='pair':bad[0]['first_pair_dual_audit']=None
            elif mutation=='prefix':bad[2]['forward_records'][0]['execution']['same_H0_CC_prefix_audit']['same_target_input']=False
            elif mutation=='mask':bad[2]['frames'][0]['arms'].pop(next(iter(bad[2]['frames'][0]['arms'])))
            elif mutation=='region':next(iter(bad[2]['frames'][0]['arms'].values()))['metrics']['regions'].pop('15')
            elif mutation=='branch_missing':bad[2]['forward_records'][0]['execution']['branch_records'].pop(H0.TARGET)
            elif mutation=='all_branches_deleted':
                for segment in bad:
                    for record in segment['forward_records']:record['execution'].pop('branch_records')
            elif mutation=='raw_prefix':bad[2]['forward_records'][0]['execution']['branch_records'][H0.TARGET]['input_sha256']='f'*64
            elif mutation=='CC_reference':bad[2]['frames'][0]['H0_CC_reference']['prediction_sha256']='changed'
            else:next(r for r in bad[0]['forward_records'] if r['mode']=='dual_reference')['execution']['branch_records'][H0.TARGET]['output_sha256']='f'*64
            with self.assertRaises(ValueError):finalizer.validate_segments(bad,h0value,H0)
        MEASUREMENTS.append({'all4_complete_segment_validation':True,'mutation_rejections':14,'all625_branch_records_deleted_rejected':True,'raw8branch_and_H0_prefix_recomputed':True,
                             'no_best_mask_or_quality_gate_computed':True})

    def test_actual_npz_source_payload_and_shape_binding(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'pred.npz';pred=np.zeros((2,480,640),dtype=np.float32);np.savez_compressed(path,a=pred,b=pred)
            digest=hashlib.sha256(pred.tobytes()).hexdigest()
            value={'frames':[{'target_id':'synthetic','prediction_file':path.name,'prediction_file_sha256':h.sha(path),
                              'arms':{key:{'prediction_sha256':digest} for key in ('a','b')}}]}
            records=h.validate_prediction_artifacts(Path(d)/'results.json',value,['a','b'])
            self.assertEqual(records[0]['predictions'],2)
            bad=copy.deepcopy(value);bad['frames'][0]['arms']['b']['prediction_sha256']='wrong'
            with self.assertRaises(ValueError):h.validate_prediction_artifacts(Path(d)/'results.json',bad,['a','b'])
            with self.assertRaises(ValueError):h.validate_prediction_artifacts(Path(d)/'results.json',value,['a'])
            pred=np.zeros((2,2,2),dtype=np.float32);np.savez_compressed(path,a=pred,b=pred);value['frames'][0]['prediction_file_sha256']=h.sha(path)
            with self.assertRaises(ValueError):h.validate_prediction_artifacts(Path(d)/'results.json',value,['a','b'])

    def test_u_and_ordinary_descriptor_artifact_integrity_and_cost(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'u.npz';u=np.zeros((10,1,30,40,384),dtype=np.float32);np.savez_compressed(path,u=u)
            digest=hashlib.sha256(u.tobytes()).hexdigest()
            scores=[dict(tile=list(tile),**{name:0. for name in PROBE.SCORE_NAMES}) for tile in h.TILES]
            frame={'target_id':'synthetic','predecision_u':{'file':path.name,'file_sha256':h.sha(path),'shape_TBHWC':list(u.shape),'dtype':'float32',
                   'payload_sha256':digest,'source_action':h.mask_name(((0,0),)),'full_forward_reused':True},
                   'H0_CC_reference':{'execution':{'branch_records':{H0.TARGET:{'input_sha256':digest}}}},
                   'ordinary_scores':scores,'ordinary_scores_sha256':h.json_sha(scores),
                   'ordinary_score_cost':{'diagnostic_only_extra_psn1_calls':12,'diagnostic_only_extra_psn1_elements':3732480,
                      'diagnostic_only_extra_psn1_temporal_MACs_nominal':37324800,'extra_fc_calls':0,'score_reductions_and_feature_reads_counted_in_cycles':False},
                   'ordinary_score_actual_audit':{'observed_extra_psn1_calls':12,'observed_extra_psn1_elements':3732480,
                      'call_shapes':[[10,1,9,9,384]]*12,'full_model_forwards':0,'temporary_hook_removed':True}}
            value={'frames':[frame]}
            self.assertEqual(len(h.validate_descriptor_artifacts(Path(d)/'results.json',value,H0.TARGET)),1)
            for mutation in ('u_payload','u_file','prefix','score_payload','score_tile','score_field','cost','observed','source'):
                bad=copy.deepcopy(value);f=bad['frames'][0]
                if mutation=='u_payload':f['predecision_u']['payload_sha256']='bad'
                elif mutation=='u_file':f['predecision_u']['file_sha256']='bad'
                elif mutation=='prefix':f['H0_CC_reference']['execution']['branch_records'][H0.TARGET]['input_sha256']='bad'
                elif mutation=='score_payload':f['ordinary_scores_sha256']='bad'
                elif mutation=='score_tile':f['ordinary_scores'][0]['tile']=[3,4]
                elif mutation=='score_field':f['ordinary_scores'][0]['psn1_activity']=float('nan')
                elif mutation=='cost':f['ordinary_score_cost']['diagnostic_only_extra_psn1_calls']=0
                elif mutation=='observed':f['ordinary_score_actual_audit']['observed_extra_psn1_calls']=11
                else:f['predecision_u']['source_action']='different'
                with self.assertRaises(ValueError):h.validate_descriptor_artifacts(Path(d)/'results.json',bad,H0.TARGET)
            MEASUREMENTS.append({'u_original_shape_dtype_SHA_and_H0_prefix_bound':True,'all12x7_ordinary_scores_and_actual_PSN1_work_bound':True,
                                 'descriptor_artifact_mutation_rejections':9,'nominal_extra_PSN1_calls_all8':96,'fullforward_budget_unchanged':625})

    def test_global_dependencies_and_only_accepted_prepare_no_training(self):
        for module in (h,runner,finalizer):
            for name,fn in vars(module).items():
                if not isinstance(fn,types.FunctionType) or fn.__module__!=module.__name__:continue
                dependencies=set()
                def scan(code):
                    dependencies.update(i.argval for i in dis.get_instructions(code) if i.opname=='LOAD_GLOBAL')
                    for obj in code.co_consts:
                        if isinstance(obj,types.CodeType):scan(obj)
                scan(fn.__code__)
                self.assertFalse(dependencies-set(fn.__globals__)-set(vars(builtins)),name)
        source=(h.HERE/'run_h1.py').read_text();tree=ast.parse(source)
        calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertEqual([c for c in calls if c.endswith('.prepare')],['h0runner.prepare'])
        self.assertFalse(any(c.endswith('.execute') or c.endswith('.backward') or 'AdamW' in c for c in calls))
        self.assertIn('r13.apply_fresh_bn',calls);self.assertIn('h0.install_recovered',calls)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261016)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    receipt={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'GPU_initialized':torch.cuda.is_initialized(),'torch':str(torch.__version__),
             'source_sha256':{str(h.HERE/n):h.sha(h.HERE/n) for n in runner.OWN},'accepted_H0_source_pins':json.loads((h.HERE/'source_pins.json').read_text()),
             'measurements':MEASUREMENTS,'errors':[str(x) for x in result.errors],'failures':[str(x) for x in result.failures],
             'scope':'CPU synthetic12MLPs/metadata/NPZ and full625 mock; no real H0/R13 result, weights, event or GT arrays read. No GPU/SSH/full model.'}
    (h.HERE/'cpu_results.json').write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1)
