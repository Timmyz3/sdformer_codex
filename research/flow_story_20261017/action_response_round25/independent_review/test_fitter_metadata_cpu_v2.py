"""Synthetic admission/role attacks, before any model-feature or GT array load."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
DATE=HERE.parents[1]
FITTER=HERE.parent/'design/fit_response_screen.py'
sys.path.insert(0,str(FITTER.parent))
spec=importlib.util.spec_from_file_location('_r25_review_fitter',FITTER)
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
ROLE=DATE/'action_domain_round20/independent_review/SELECTOR_ROLES_V1.json'


class Boundary(Exception):pass


def save(path,value):
    path.write_text(json.dumps(value)+'\n')
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


class Checks(unittest.TestCase):
    def fixture(self,root):
        roles=json.loads(ROLE.read_text());rows=[];records=[];rolemap={}
        for role in ('fit','development'):
            for c in roles['groups'][role]:
                tid=c['target_id'];rolemap[tid]=role
                row={k:c[k] for k in ('target_id','sequence')};row['role']=role
                actionpins=[]
                for arm in ('native','FF','T2')+tuple('fine_%d_%d'%t for t in f.m.TILES):
                    path=root/(tid+'_'+arm+'.npz');path.write_bytes(b'synthetic archive bytes; never loaded')
                    pin={'path':str(path),'sha256':f.sha(path)}
                    records.append({'target_id':tid,'arm':arm,'selector_role':role,'quality_endpoint':True,
                        'execution':'native' if arm=='native' else 'compact','prediction_file':path.name,
                        'prediction_file_sha256':pin['sha256']})
                    if arm=='native':row['native_archive']=pin
                    elif arm=='T2' and role=='fit':row['background_archive']=pin
                    elif arm.startswith('fine_') and role=='fit':actionpins.append(pin)
                if role=='fit':row['action_archives']=actionpins
                rows.append(row)
        records.extend([dict(records[2],quality_endpoint=False,arm='T2_repeat'+str(i)) for i in (1,2)])
        execution={'schema':'flow_r25_action_response_collection_v1','graph_id':f.GRAPH,
            'status':'ACTION_RESPONSE_COLLECTION362_COMPLETE','full_forward_count':362,
            'binding':{'selector_roles_sha256':f.ROLE_SHA,'selector_role_map':rolemap,'graph_id':f.GRAPH},'forward_records':records}
        result=save(root/'result.json',execution)
        audit={'schema':'flow_r25_action_response_independent_execution_audit_v1','status':'PASS',
            'quality_analysis_allowed':True,'source_result_sha256':result['sha256']}
        auditpin=save(root/'audit.json',audit)
        labels={'schema':'flow_r25_fit_gain_labels_v1','graph_id':f.GRAPH,'background':'T2',
            'source_result_sha256':result['sha256'],'execution_audit_sha256':auditpin['sha256'],
            'tile_order':[list(t) for t in f.m.TILES],'normalization':'max(native_allGT_AEE,1e-6)',
            'signed_relative_gain_by_id':{r['target_id']:[0.]*12 for r in rows[:16]}}
        gp=DATE/'ordinary_controls_round18/implementation/g0_core.py'
        self.assertEqual(f.sha(gp),f.G0_SHA)
        inp={'schema':'flow_r25_fit_only_inputs_v1','source_pins':[
            {'path':str(path),'sha256':f.sha(path)} for path in (FITTER,Path(f.m.__file__))],
            'roles':{'path':str(ROLE),'sha256':f.sha(ROLE)},'execution_result':result,'execution_audit':auditpin,
            'members':rows,'fit_labels':save(root/'labels.json',labels),'ordinary_score_source':{'path':str(gp),'sha256':f.sha(gp)}}
        return inp

    def run_case(self,mutation=None):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inp=self.fixture(root)
            if mutation:mutation(inp,root)
            pin=save(root/'inputs.json',inp)
            argv=['fit_response_screen.py','--inputs',pin['path'],'--inputs-sha256',pin['sha256'],'--output',str(root/'new-output')]
            with mock.patch.object(sys,'argv',argv),mock.patch.object(f.np,'load',side_effect=Boundary('ARRAY_BOUNDARY')) as loads,\
                 mock.patch.object(f.torch,'set_num_interop_threads'):
                if mutation:
                    with self.assertRaises(ValueError):f.main()
                    self.assertEqual(loads.call_count,0)
                else:
                    with self.assertRaises(Boundary):f.main()
                    self.assertEqual(loads.call_count,1)
            self.assertFalse((root/'new-output').exists())

    def test_valid_metadata_reaches_first_native_array_only(self):self.run_case()

    def test_malicious_roles_and_sources_rejected_before_arrays(self):
        def swap(inp,root):
            a=inp['members'][0];b=next(r for r in inp['members'][:16] if r['sequence']!=a['sequence'])
            a['sequence'],b['sequence']=b['sequence'],a['sequence']
        def dev_action(inp,root):inp['members'][16]['action_archives']=[]
        def dev_gt(inp,root):inp['members'][16]['gt']={'path':'never-read-devGT','sha256':'0'*64}
        def wrong_role(inp,root):inp['members'][0]['role']='development'
        def wrong_label_source(inp,root):
            labels=json.loads(Path(inp['fit_labels']['path']).read_text());labels['source_result_sha256']='0'*64
            inp['fit_labels']=save(root/'badlabels.json',labels)
        def dev_label(inp,root):
            labels=json.loads(Path(inp['fit_labels']['path']).read_text())
            labels['signed_relative_gain_by_id'][inp['members'][16]['target_id']]=[0.]*12
            inp['fit_labels']=save(root/'badlabels.json',labels)
        for mutation in (swap,dev_action,dev_gt,wrong_role,wrong_label_source,dev_label):
            with self.subTest(mutation=mutation.__name__):self.run_case(mutation)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    out={'schema':'flow_r25_fitter_metadata_independent_cpu_v2','status':'PASS' if result.wasSuccessful() else 'FAIL',
        'tests_run':result.testsRun,'negative_cases':6,'failures':len(result.failures),'errors':len(result.errors),
        'source_sha256':{'fitter':f.sha(FITTER),'math':f.sha(Path(f.m.__file__)),'review':f.sha(__file__)},
        'real_arrays_or_GT_read':False,'GPU_used':False,'actual_training':False,
        'valid_metadata_reaches_first_native_array':result.wasSuccessful(),'malformed_metadata_array_loads':0}
    path=HERE/'FITTER_METADATA_CPU_V2.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out));raise SystemExit(0 if result.wasSuccessful() else 1)
