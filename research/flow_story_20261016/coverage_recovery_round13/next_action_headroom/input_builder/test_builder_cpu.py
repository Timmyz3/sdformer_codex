#!/usr/bin/env python3
"""Pure stdlib synthetic gate/mapping tests; no actual result or tensor read."""
import copy
import dis
import builtins
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
sys.dont_write_bytecode=True
import build_h0_inputs as b
H0,RUN=b.load_helpers()
MEASUREMENTS=[]


def gate_table(normal,qualified):
    families=('normal_references','matched_recovered_references','own_checkpoint_references','shared_fine_preserved')
    tables={}
    for condition in b.CONDITIONS:
        tables[condition]={}
        for arm in ('CC','fine_s2b2'):
            gate={f:{a:{'pass':normal[condition] if f=='normal_references' else (condition,arm) in qualified} for a in ('native','FF')} for f in families}
            gate['ordinary_base_quality_pass']=all(v['pass'] for f in families for v in gate[f].values())
            tables[condition][arm]=gate
    return {'ordinary_reference_normality':dict(normal),'candidate_gates':tables}


def fixture(directory,normal=None,qualified=None):
    root=Path(directory)
    if normal is None:normal={c:True for c in b.CONDITIONS}
    if qualified is None:qualified={('reference','CC')}
    def write(name,value):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,indent=2)+'\n');return path
    def asset(path):return {'local_path':str(path),'remote_path':'/remote/evidence/'+path.name,'sha256':b.sha(path)}
    membership_path=H0.HERE.parents[1]/'data_protocol/manifest_v1.json';membership=json.loads(membership_path.read_text())
    h8_path=H0.HERE/'h8_manifest.json'
    runtime=write('runtime.json',{'schema':'flow_round13_runtime_v1','region_sha256':'r'*64})
    buffers=root/'synthetic_bn.npz';buffers.write_bytes(b'synthetic source bytes, not real arrays')
    receipt=write('bn_receipt.json',{'schema':'flow_round13_common_bn_v1','status':'COMMON_BN_COMPLETE','BN_scope':'MLP_R13_COMMON_CMA176',
        'checkpoint_sha256':b.CHECKPOINT_SHA,'membership_sha256':H0.MEMBER_SHA,'buffers_sha256':b.sha(buffers),
        'bn_target_ids':[r['target_id'] for r in membership['bn_members']],'old_CMA_loaded':False,'GT_used':False,
        'calibration_forwards':176,'calibration_bn_count':78,'retained_mlp_bn_count':24,'calibration_definition':'all78_reset_CMA_train_nonBN_eval'})
    common={'checkpoint_sha256':b.CHECKPOINT_SHA,'membership_sha256':H0.MEMBER_SHA,'runtime_manifest_sha256':b.sha(runtime),
            'BN_buffers_sha256':b.sha(buffers),'BN_receipt_sha256':b.sha(receipt),'region_sha256':'r'*64,
            'runner_sha256':H0.R13_SHA,'execute_sha256':'da4f460c9582dbe244218dd70f8b5928c6ede1495527d6cf355d1cbc01955541'}
    result_map={};analysis_inputs={};audit_rows=[]
    for condition in b.CONDITIONS:
        for arm in b.ARMS:
            key=condition+'/'+arm;folder=root/condition/arm;folder.mkdir(parents=True)
            weights=folder/'weights.pt';weights.write_bytes(('synthetic_'+key).encode());weight_sha=b.sha(weights)
            result={'schema':'flow_round13_coverage_recovery_v1','status':'ROUND13_COVERAGE_RECOVERY_COMPLETE','completed_optimizer_steps':200,'training_performed':True,
                    'binding':dict(common,condition=condition,arm=arm,mode='recover'),'initial_model_state_sha256':'1'*64,'final_state_sha256':hashlib.sha256(key.encode()).hexdigest(),
                    'weights_file':'weights.pt','weights_sha256':weight_sha,
                    **{k:True for k in ('frozen_state_unchanged','BN_buffers_unchanged','original_modules_restored','all_hooks_removed')}}
            path=write(condition+'/'+arm+'/results.json',result)
            result_map[key]={'local_result':str(path),'remote_result':'/remote/runs/'+key+'/results.json','result_sha256':b.sha(path),
                             'remote_weights':'/remote/runs/'+key+'/weights.pt','weights_sha256':weight_sha,'local_weights':str(weights)}
            analysis_inputs[key]={'path':'/different/original/evidence/'+key+'/results.json','sha256':b.sha(path)}
            for mode in ('smoke','recover'):
                audit_rows.append({'condition':condition,'arm':arm,'mode':mode,'result_sha256':b.sha(path),'final_state_sha256':result['final_state_sha256'],
                    'weights':{'sha256':weight_sha,**{k:True for k in ('final_state_independently_recomputed','frozen_complement_independently_recomputed',
                                                               'initial_state_reconstructed','BN72_actual_values_match')}} if mode=='recover' else None})
    analysis={'schema':'flow_round13_coverage_quality_analysis_v1','script_sha256':'b1997b55deea309088388a011b83272bafe438bedc43db5ee4440655bc5034ad',
              'analysis_protocol_sha256':'5ec51175fbf18b9d2f4fab329247b011700f4cf109b50bf73e905708698c2d7b',
              'R12_V2_source_sha256':'525ba2821d1e671b581f63e80c4bce02ef3e524c3e6af7d143a723914b001cd8',
              'state_layout_count':40,'evaluation_endpoint_count':320,'region_count':16,'full_forward_count':1964,
              'initial_state_all_eight_exact':True,'initial_same_layout_prediction_SHA_and_metrics_cross_condition_exact':True,
              'inputs':analysis_inputs,'prediction_artifact_audits':{k:{'prediction_files':16,'prediction_payloads':40} for k in b.KEYS},
              'common_contract':dict(common,initial_state_sha256='1'*64),**gate_table(normal,qualified)}
    analysis_path=write('analysis.json',analysis)
    audit={'schema':'flow_round13_independent_execution_audit_v1','status':'PASS','auditor_sha256':'dd57f9f315e1c5f38961208496a37f46db577c47b8507029a55b5bc216d8c8e8',
           'scope':{'conditions':list(b.CONDITIONS),'modes':['smoke','recover'],'runs':16},'weights_count':8,'prediction_count':320,'archive_count':128,
           'full_forwards':2048,'optimizer_updates':1640,'paired_initial_predictions_equal':True,'common_initial_model_sha256':'1'*64,
           'original_checkpoint_sha256':b.CHECKPOINT_SHA,'new_BN_actual_sha256':b.sha(buffers),'results':audit_rows}
    audit_path=write('audit.json',audit)
    return {'schema':'flow_h0_remote_mapping_v1','output_remote_dir':'/remote/new_bundle','results':result_map,
            'assets':{k:asset(p) for k,p in (('analysis',analysis_path),('independent_execution_audit',audit_path),('runtime_manifest',runtime),
                       ('membership',membership_path),('h8_manifest',h8_path),('bn_buffers',buffers),('bn_receipt',receipt))}}


class Tests(unittest.TestCase):
    def test_fixed_priority_and_stable_FF_fallback_only(self):
        cases=[({'reference':True,'diversity':True},{('reference','CC'),('reference','fine_s2b2'),('diversity','CC')},('reference','CC','qualified_ordinary_base')),
               ({'reference':True,'diversity':True},{('reference','fine_s2b2'),('diversity','CC')},('reference','fine_s2b2','qualified_ordinary_base')),
               ({'reference':True,'diversity':True},{('diversity','CC')},('diversity','CC','qualified_ordinary_base')),
               ({'reference':True,'diversity':True},{('diversity','fine_s2b2')},('diversity','fine_s2b2','qualified_ordinary_base')),
               ({'reference':True,'diversity':True},set(),('reference','FF','action_only_screening')),
               ({'reference':False,'diversity':True},set(),('diversity','FF','action_only_screening')),
               ({'reference':False,'diversity':False},set(),None)]
        for normal,qualified,expected in cases:
            value=b.select_state(gate_table(normal,qualified))
            self.assertEqual(None if value is None else (value['condition'],value['W_arm'],value['scope']),expected)
        bad=gate_table({'reference':True,'diversity':True},set());bad['candidate_gates']['reference']['CC']['ordinary_base_quality_pass']=True
        with self.assertRaises(ValueError):b.select_state(bad)
        MEASUREMENTS.append({'synthetic_priority_cases':7,'inconsistent_gate_conjunction_rejected':True,'no_actual_condition_selected':True})

    def test_complete_mapping_and_generated_inputs_follow_H0_template(self):
        with tempfile.TemporaryDirectory() as d:
            mapping=fixture(d);inputs,admission,receipt=b.prepare(mapping,H0,RUN)
            out=Path(d)/'bundle';built=b.write_bundle(out,mapping['output_remote_dir'],inputs,admission,receipt)
            value=json.loads((out/'inputs.json').read_text());admit=json.loads((out/'admission.json').read_text())
            self.assertEqual(set(value),set(json.loads((H0.HERE/'inputs.template.json').read_text())))
            self.assertEqual(set(admit),set(json.loads((H0.HERE/'admission.template.json').read_text())))
            self.assertEqual(value['admission'],{'path':'/remote/new_bundle/admission.json','sha256':b.sha(out/'admission.json')})
            self.assertEqual(value['states']['W']['result'],value['all_eight_results']['reference/CC'])
            self.assertTrue(all(receipt['local_weight_bytes_checked'].values()));self.assertFalse(built['experiment_started'])
            self.assertFalse(built['remote_path_existence_or_bytes_checked'])
            with self.assertRaises(ValueError):b.write_bundle(out,mapping['output_remote_dir'],inputs,admission,receipt)
        MEASUREMENTS.append({'complete_synthetic_bundle_matches_exact_H0_template_keys':True,'cross_host_paths_sha_bound':True,
                             'fresh_output_only':True,'remote_bytes_not_claimed':True})

    def test_no_qualifying_state_writes_no_runnable_files(self):
        with tempfile.TemporaryDirectory() as d:
            mapping=fixture(d,normal={c:False for c in b.CONDITIONS},qualified=set())
            inputs,admission,receipt=b.prepare(mapping,H0,RUN);self.assertIsNone(inputs);self.assertIsNone(admission)
            out=Path(d)/'no_qualified';b.write_bundle(out,mapping['output_remote_dir'],inputs,admission,receipt)
            self.assertEqual([p.name for p in out.iterdir()],['NO_QUALIFYING_STATE.json'])

    def test_incomplete_or_wrong_source_and_remote_pairing_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            mapping=fixture(d)
            for mutation in ('source_sha','missing_result','weights_sha','remote_pair','optional_weight','collision'):
                bad=copy.deepcopy(mapping)
                if mutation=='source_sha':bad['assets']['analysis']['sha256']='wrong'
                elif mutation=='missing_result':bad['results'].pop('diversity/CC')
                elif mutation=='weights_sha':bad['results']['reference/CC']['weights_sha256']='wrong'
                elif mutation=='remote_pair':bad['results']['reference/CC']['remote_weights']='/unpaired/weights.pt'
                elif mutation=='optional_weight':bad['results']['reference/CC']['local_weights']=str(Path(d)/'not-present.pt')
                else:bad['assets']['analysis']['remote_path']='/remote/new_bundle/admission.json'
                with self.assertRaises(ValueError):b.prepare(bad,H0,RUN)
            p=Path(mapping['assets']['independent_execution_audit']['local_path']);value=json.loads(p.read_text());value['scope']['runs']=8
            p.write_text(json.dumps(value));mapping['assets']['independent_execution_audit']['sha256']=b.sha(p)
            with self.assertRaises(ValueError):b.prepare(mapping,H0,RUN)
        MEASUREMENTS.append({'synthetic_source_mapping_rejections':7})

    def test_all_entry_globals_bound_no_Torch_import(self):
        for name,fn in vars(b).items():
            if not isinstance(fn,types.FunctionType) or fn.__module__!=b.__name__:continue
            dependencies=set()
            def scan(code):
                dependencies.update(i.argval for i in dis.get_instructions(code) if i.opname=='LOAD_GLOBAL')
                for obj in code.co_consts:
                    if isinstance(obj,types.CodeType):scan(obj)
            scan(fn.__code__)
            self.assertFalse(dependencies-set(fn.__globals__)-set(vars(builtins)),name)
        self.assertNotIn('torch',sys.modules)

    def test_wrong_new_BN_calibration_definition_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            mapping=fixture(d)
            p=Path(mapping['assets']['bn_receipt']['local_path']);value=json.loads(p.read_text())
            value['calibration_bn_count']=24;p.write_text(json.dumps(value))
            mapping['assets']['bn_receipt']['sha256']=b.sha(p)
            with self.assertRaisesRegex(ValueError,'calibration definition/counts'):b.prepare(mapping,H0,RUN)
        MEASUREMENTS.append({'incomplete_or_wrong_BN_calibration_rejected':True})


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    receipt={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'GPU_initialized':False,'torch_imported':'torch' in sys.modules,
             'source_sha256':{name:b.sha(b.HERE/name) for name in b.OWN},'measurements':MEASUREMENTS,
             'errors':[str(x) for x in result.errors],'failures':[str(x) for x in result.failures],
             'scope':'Stdlib synthetic quality/audit/result/weight byte fixtures and existing identity metadata only. No real formal quality, recovered weights, events, GT, SSH, GPU or model import.'}
    (b.HERE/'cpu_results.json').write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() and not receipt['torch_imported'] else 1)
