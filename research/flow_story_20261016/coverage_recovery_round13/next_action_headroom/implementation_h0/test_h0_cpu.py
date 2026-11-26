#!/usr/bin/env python3
"""Bounded H0 CPU fixture checks; never full model, data arrays, SSH or GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import ast
import builtins
import copy
import dis
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
import h0_core as h
import run_h0 as runner
RUNTIME=h.load_frozen();R13,COLLECTOR,PROBE,PARENT,R9,R8=RUNTIME
FIXPATH=R13.OLD.source_path('fixture');sys.path.insert(0,str(FIXPATH.parent))
FIX=h.load_file(FIXPATH,'_h0_native_fixture')
sys.path.insert(0,str(h.REPO/'research/flow_story_20261013/breadth'))
import evaluate_breadth as base
import run_bn_scope as scope
MERGE=PARENT.load_frozen_merge()
MEASUREMENTS=[]


def toy():
    model,targets,shapes,x,functional=FIX.build()
    model.external=torch.nn.ModuleList([torch.nn.BatchNorm1d(4) for _ in range(54)]);model.eval()
    all_bn,mlp=scope.bn_inventory(model,torch);values={}
    for name in mlp:
        values[name+'.running_mean']=np.linspace(-.1,.1,all_bn[name].num_features,dtype=np.float32)
        values[name+'.running_var']=np.linspace(.7,1.3,all_bn[name].num_features,dtype=np.float32)
        values[name+'.num_batches_tracked']=np.array(176,dtype=np.int64)
    with tempfile.TemporaryDirectory() as directory:
        p=Path(directory)/'bn.npz';receipt=Path(directory)/'receipt.json';receipt.write_text('{}');np.savez(p,**values)
        R13.apply_fresh_bn(model,p,receipt,{},'',scope,base,torch,validate_receipt=False)
    model.eval().requires_grad_(False)
    return model,targets,shapes,x,functional


class Tests(unittest.TestCase):
    def test_h8_only_fixed_identity_selection_and_disjoint_roles(self):
        path=h.HERE.parents[1]/'data_protocol/manifest_v1.json'
        m=json.loads(path.read_text());out=h.h8_manifest(m,h.sha(path))
        self.assertEqual(out,json.loads((h.HERE/'h8_manifest.json').read_text()))
        self.assertEqual([r['sequence'] for r in out['members']],m['split_sequences']['train'])
        self.assertEqual(len({r['target_id'] for r in out['members']}),8)
        bad=copy.deepcopy(m);bad['selector_members'].append(bad['selector_members'][0])
        with self.assertRaises(ValueError):h.h8_manifest(bad,h.MEMBER_SHA)
        with self.assertRaises(ValueError):h.h8_manifest(m,'wrong')
        MEASUREMENTS.append({'H8_ids':[r['target_id'] for r in out['members']],'selection_uses_identity_only':True})

    def test_production_true_tail_masks_and_exact_row_budget(self):
        shapes={n:tuple(PARENT.STAGE_SHAPES[int(n.split('.layers.')[1].split('.')[0])]) for n in toy()[1]}
        expected={'native':558000,'FF':558000,'CC':153600,'P':164200,'F12':162240,'T8':155560}
        for anchor,n in expected.items():self.assertEqual(h.expected_rows(shapes,anchor,torch),n)
        for anchor in h.DUALS:self.assertEqual(h.expected_rows(shapes,anchor,torch,dual=True),609600)
        geometry=h.tile_geometry(shapes[h.TARGET]);complete=[r for r in geometry if r['complete']];tails=[r for r in geometry if not r['complete']]
        self.assertEqual((len(complete),len(tails)),(12,8))
        self.assertEqual([r['fine_extra_fc_rows'] for r in tails],[300,300,300,240,240,240,240,100])
        self.assertEqual(sum(r['fine_extra_fc_rows'] for r in tails),1960)
        f=h.mask_for(shapes[h.TARGET],h.TARGET,'F12',torch);t=h.mask_for(shapes[h.TARGET],h.TARGET,'T8',torch)
        self.assertEqual((int(f.sum()),int(t.sum())),(12,8));self.assertTrue(torch.equal(~f,t))
        MEASUREMENTS.append({'production_FC_rows_each':expected,'all5_dual_FC_rows_each':609600,'target_tail_geometry':tails})

    def test_five_real_math_actual_dual_pairs_and_prefix_restore(self):
        model,targets,shapes,x,functional=toy();initial=R13.state_digest(model);bn=base.bn_signature(model,torch)
        coarse=None
        old_wrap=PROBE.wrap;old_target=PROBE.TARGET
        for anchor in ('CC','FF','P','F12','T8'):
            actual=h.forward(RUNTIME,model,functional,targets,shapes,x,anchor,MERGE,torch)
            dual=h.forward(RUNTIME,model,functional,targets,shapes,x,anchor,MERGE,torch,mode='dual_reference')
            self.assertTrue(h.check_pair(actual,dual,torch)['all8_branches_exact'])
            if anchor == 'CC':coarse=actual[1]
            if anchor in ('P','F12','T8'):self.assertTrue(h.check_prefix(actual[1],coarse)['same_target_input'])
            self.assertEqual(R13.state_digest(model),initial);self.assertEqual(base.bn_signature(model,torch),bn)
            self.assertTrue(all(model.get_submodule(n) is m for n,m in targets.items()))
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        self.assertIs(PROBE.wrap,old_wrap);self.assertEqual(PROBE.TARGET,old_target)
        native=h.forward(RUNTIME,model,functional,targets,shapes,x,'native',MERGE,torch)
        self.assertEqual(len(native[2]),8)
        for anchor,legacy in (('native','native'),('FF','FF'),('CC','CC'),('P','fine_s2b2')):
            new=h.forward(RUNTIME,model,functional,targets,shapes,x,anchor,MERGE,torch)
            old=R13.reference_forward(R9,R8,PARENT,model,functional,targets,shapes,x,legacy,MERGE,torch)
            self.assertTrue(h.check_pair(new,old,torch)['all_flow_heads_exact'])
        MEASUREMENTS.append({'all5_actual_dual_pairs_exact':True,'4_existing_anchor_references_exact':True,
                             'F12_T8_real_tail_fixture_hw':[11,14],'target_prefix3_anchors_exact':True,
                             'historical_probe_globals_unchanged':True,'all_state_BN_hooks_modules_restored':True,
                             'scope':'Two synthetic output heads, not full-model flow quality.'})

    def test_failure_cleanup(self):
        model,targets,shapes,x,functional=toy();before=R13.state_digest(model)
        with patch.object(targets[h.TARGET].fc2,'forward',side_effect=RuntimeError('injected H0 failure')):
            with self.assertRaisesRegex(RuntimeError,'injected H0'):
                h.forward(RUNTIME,model,functional,targets,shapes,x,'T8',MERGE,torch)
        self.assertEqual(R13.state_digest(model),before)
        self.assertTrue(all(model.get_submodule(n) is m for n,m in targets.items()))
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_strict_recovered_loading_w_n_f_and_atomic_corruption_reject(self):
        model,targets,shapes,x,functional=toy();original=copy.deepcopy(model.state_dict());initial=R13.state_digest(model)
        common={n:v.clone() for n,v in original.items() if n.endswith(h.BUFFERS)}
        names=[n for n,_ in model.named_parameters() if '.mlp.' in n and h.selected_layer(n)]
        key=next(n for n in names if '.fc1.weight' in n)
        for index,arm in enumerate(('CC','native','FF')):
            state=copy.deepcopy(original);state[key]+=float(index+1)/1000
            binding={'condition':'diversity','arm':arm,'mode':'recover'}
            descriptor={'binding':binding,'initial_state_sha256':initial,'final_state_sha256':h.state_dict_digest(state),'trainable_names':names}
            payload={'schema':'flow_round13_recovered_shared_native_state_v1','completed_optimizer_steps':200,
                     'BN_scope':'MLP_R13_COMMON_CMA176','BN_buffers_unchanged':True,'model_state_dict':state,**descriptor}
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'weights.pt';torch.save(payload,path);loaded=torch.load(path,weights_only=True,map_location='cpu')
            info=h.install_recovered(model,loaded,descriptor,common,torch)
            self.assertEqual(info['state_sha256'],descriptor['final_state_sha256']);self.assertNotEqual(info['state_sha256'],initial)
            before=R13.state_digest(model)
            for mutation in ('key','dtype','BN','digest','binding','steps','names'):
                bad=copy.deepcopy(loaded)
                if mutation == 'key':bad['model_state_dict'].pop(key)
                elif mutation == 'dtype':bad['model_state_dict'][key]=bad['model_state_dict'][key].double()
                elif mutation == 'BN':bad['model_state_dict'][next(iter(common))]+=1
                elif mutation == 'digest':bad['model_state_dict'][key]+=1
                elif mutation == 'binding':bad['binding']['condition']='reference'
                elif mutation == 'steps':bad['completed_optimizer_steps']=199
                else:bad['trainable_names']=names[:-1]
                with self.assertRaises(ValueError):h.install_recovered(model,bad,descriptor,common,torch)
                self.assertEqual(R13.state_digest(model),before)
        self.assertTrue(all(not m.track_running_stats and m.running_mean is None for m in model.external))
        MEASUREMENTS.append({'recovered_states_strictly_switched':3,'atomic_rejection_cases_per_state':7,
                             'common24_BN_exact_external54_current_stat':True,'original_state_never_used_for_H0_inference':True})

    def test_mock_production_orchestration_69_calls_and_no_extra_work(self):
        rows=json.loads((h.HERE/'h8_manifest.json').read_text())['members'];plan=h.forward_plan(rows)
        states=[];calls=[];saved=[]
        names=['sttmultires_unet.encoders.swin3d.layers.%d.swin_blocks.%d.mlp' % (s,b) for s,depth in ((0,2),(2,6)) for b in range(depth)]
        def infer(item):
            calls.append(item)
            inputs={n:'same_'+n for n in names};branches={n:torch.zeros(1) for n in names}
            ex={'branch_records':{n:{'input_sha256':inputs[n],'output_sha256':'out_'+n} for n in names}}
            return [torch.zeros(1,2,1,1) for _ in range(4)],ex,branches,inputs
        rec,pairs=h.drive_plan(plan,states.append,infer,lambda item,out:saved.append(item),lambda x,y:h.check_pair(x,y,torch))
        self.assertEqual(states,['W','N','F']);self.assertEqual((len(calls),len(saved),len(rec),len(pairs)),(69,64,69,5))
        self.assertEqual({key:sum(x['anchor']==key for x in saved) for key in h.ANCHORS},{key:8 for key in h.ANCHORS})
        self.assertEqual(sum('prefix_audit' in x['execution'] for x in rec),24)
        with self.assertRaises(ValueError):h.drive_plan(plan[:-1],states.append,infer,lambda *_:None,lambda *_:None)
        MEASUREMENTS.append({'production_orchestration_mock_calls':69,'actual8x8':64,'first_frame5_dual':5,
                             'state_switch_order':['W','N','F'],'prefix_audits':24,'GT_action_input_absent':True})

    def test_admission_declines_unfinished_or_unstable_recovery(self):
        admission={'schema':'flow_h0_admission_v1','status':'READY_AFTER_R13_ALL_EIGHT_AUDIT','condition':'diversity','W_arm':'CC',
                   'audited_conditions_arms':[[c,a] for c in ('reference','diversity') for a in ('native','FF','CC','fine_s2b2')],
                   'ordinary_N_F_stable':True,'scope':'qualified_ordinary_base','selected_base_passes_required_gates':True,
                   'shared_fine_protection_passes':True,'rationale':'Synthetic admission fixture, no real quality inspected'}
        self.assertEqual(runner.validate_admission(admission,'diversity','CC'),'qualified_ordinary_base')
        for key,val in (('ordinary_N_F_stable',False),('status','RUNNING'),('shared_fine_protection_passes',False)):
            bad=copy.deepcopy(admission);bad[key]=val
            with self.assertRaises(ValueError):runner.validate_admission(bad,'diversity','CC')
        admission.update(scope='action_only_screening',W_arm='FF',selected_base_passes_required_gates=False)
        self.assertEqual(runner.validate_admission(admission,'diversity','FF'),'action_only_screening')

    def test_entry_global_dependencies_and_forbidden_legacy_calls(self):
        for module in (h,runner):
            for name,function in vars(module).items():
                if not isinstance(function,types.FunctionType) or function.__module__ != module.__name__:continue
                dependencies=set()
                def scan(code):
                    dependencies.update(i.argval for i in dis.get_instructions(code) if i.opname == 'LOAD_GLOBAL')
                    for obj in code.co_consts:
                        if isinstance(obj,types.CodeType):scan(obj)
                scan(function.__code__)
                self.assertFalse(dependencies-set(function.__globals__)-set(vars(builtins)),name)
        tree=ast.parse((h.HERE/'run_h0.py').read_text())
        calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(c.endswith(('.prepare','.execute')) for c in calls))
        self.assertIn('r13.apply_fresh_bn',calls);self.assertIn('core.install_recovered',calls)
        self.assertFalse(torch.cuda.is_initialized())

    def test_completed_analysis_is_bound_to_all8_sources_and_selected_states(self):
        common={'membership_sha256':h.MEMBER_SHA,'BN_buffers_sha256':'synthetic','checkpoint_sha256':'original'}
        records={c+'/'+a:{'path':'/synthetic/'+c+'/'+a+'/results.json','sha256':c+'_'+a} for c in ('reference','diversity') for a in ('native','FF','CC','fine_s2b2')}
        inputs={'condition':'diversity','W_arm':'CC','all_eight_results':copy.deepcopy(records),
                'states':{role:{'result':records['diversity/'+arm],'weights':{'path':'/remote/'+arm+'/weights.pt','sha256':'weights_'+arm}} for role,arm in (('W','CC'),('N','native'),('F','FF'))}}
        admission={'scope':'qualified_ordinary_base'}
        analysis={'schema':'flow_round13_coverage_quality_analysis_v1','script_sha256':'b1997b55deea309088388a011b83272bafe438bedc43db5ee4440655bc5034ad',
                  'analysis_protocol_sha256':'5ec51175fbf18b9d2f4fab329247b011700f4cf109b50bf73e905708698c2d7b',
                  'R12_V2_source_sha256':'525ba2821d1e671b581f63e80c4bce02ef3e524c3e6af7d143a723914b001cd8',
                  'state_layout_count':40,'evaluation_endpoint_count':320,'region_count':16,'full_forward_count':1964,
                  'initial_state_all_eight_exact':True,'initial_same_layout_prediction_SHA_and_metrics_cross_condition_exact':True,
                  'inputs':records,'prediction_artifact_audits':{k:{'prediction_files':16,'prediction_payloads':40} for k in records},
                  'common_contract':dict(common,initial_state_sha256='initial'),'ordinary_reference_normality':{'diversity':True},
                  'candidate_gates':{'diversity':{'CC':{'ordinary_base_quality_pass':True,'shared_fine_preserved':{k:{'pass':True} for k in ('native','FF')}}}}}
        self.assertTrue(runner.validate_completed_analysis(analysis,inputs,admission,common)['complete_eight_sources_bound'])
        for mutation in ('schema','script','coverage','state_source','BN','ordinary','qualified','audit'):
            bad=copy.deepcopy(analysis)
            if mutation == 'schema':bad['schema']='arbitrary_file'
            elif mutation == 'script':bad['script_sha256']='unfrozen'
            elif mutation == 'coverage':bad['inputs'].pop('reference/CC')
            elif mutation == 'state_source':bad['inputs']['diversity/CC']['sha256']='different'
            elif mutation == 'BN':bad['common_contract']['BN_buffers_sha256']='wrong'
            elif mutation == 'ordinary':bad['ordinary_reference_normality']['diversity']=False
            elif mutation == 'qualified':bad['candidate_gates']['diversity']['CC']['ordinary_base_quality_pass']=False
            else:bad['prediction_artifact_audits']['reference/FF']['prediction_payloads']=39
            with self.assertRaises(ValueError):runner.validate_completed_analysis(bad,inputs,admission,common)
        MEASUREMENTS.append({'all_eight_analysis_and_selected_states_binding':True,'synthetic_wrong_admission_rejections':8,
                             'real_R13_analysis_not_read':True})
        rows=[]
        for condition in ('reference','diversity'):
            for mode in ('smoke','recover'):
                for arm in ('native','FF','CC','fine_s2b2'):
                    rows.append({'condition':condition,'mode':mode,'arm':arm,'result_sha256':records[condition+'/'+arm]['sha256'],
                                 'weights':{'sha256':'weights_'+arm,'final_state_independently_recomputed':True,
                                            'frozen_complement_independently_recomputed':True,'initial_state_reconstructed':True,
                                            'BN72_actual_values_match':True} if mode == 'recover' else None})
        audit={'schema':'flow_round13_independent_execution_audit_v1','status':'PASS',
               'auditor_sha256':'dd57f9f315e1c5f38961208496a37f46db577c47b8507029a55b5bc216d8c8e8',
               'scope':{'conditions':['reference','diversity'],'modes':['smoke','recover'],'runs':16},
               'weights_count':8,'prediction_count':320,'archive_count':128,'full_forwards':2048,'optimizer_updates':1640,
               'paired_initial_predictions_equal':True,'common_initial_model_sha256':'initial','original_checkpoint_sha256':'original',
               'new_BN_actual_sha256':'synthetic','results':rows}
        self.assertTrue(runner.validate_independent_execution(audit,analysis,inputs,common)['complete16_independent_runs_bound'])
        for mutation in ('status','scope','BN','weight','result','count'):
            bad=copy.deepcopy(audit)
            if mutation == 'status':bad['status']='RUNNING'
            elif mutation == 'scope':bad['scope']['conditions']=['diversity']
            elif mutation == 'BN':bad['new_BN_actual_sha256']='other'
            elif mutation == 'weight':bad['results'][-2]['weights']['sha256']='other'
            elif mutation == 'result':bad['results'][-2]['result_sha256']='other'
            else:bad['weights_count']=7
            with self.assertRaises(ValueError):runner.validate_independent_execution(bad,analysis,inputs,common)
        relocated=copy.deepcopy(inputs)
        for key,item in relocated['all_eight_results'].items():item['path']='/remote/'+key+'/results.json'
        for role,arm in (('W','CC'),('N','native'),('F','FF')):relocated['states'][role]['result']=relocated['all_eight_results']['diversity/'+arm]
        self.assertTrue(runner.validate_completed_analysis(analysis,relocated,admission,common)['complete_eight_sources_bound'])
        self.assertTrue(runner.validate_independent_execution(audit,analysis,relocated,common)['complete16_independent_runs_bound'])
        MEASUREMENTS.append({'all16_independent_execution_audit_bound':True,'rejections':6,'explicit_cross_host_SHA_relocation_passes':True})


if __name__ == '__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261016)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    receipt={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'GPU_initialized':torch.cuda.is_initialized(),
             'torch':str(torch.__version__),'source_sha256':{str(h.HERE/n):h.sha(h.HERE/n) for n in runner.OWN},
             'measurements':MEASUREMENTS,'errors':[str(x) for x in result.errors],'failures':[str(x) for x in result.failures],
             'scope':'CPU synthetic native MLPs, mock orchestration and identity metadata. No model/data/quality/GPU/SSH access; no real recovered artifact read.'}
    (h.HERE/'cpu_results.json').write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1)
