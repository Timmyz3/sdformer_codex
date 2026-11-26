#!/usr/bin/env python3
"""Exercise actual historical service schema and adversarial corruption only.

No R13 result or quality field is inspected. Old result dictionaries are parsed
solely to check explicit forward/service records; no runner module is imported.
"""
import copy
import json
from pathlib import Path
import unittest
import audit_execution as a

class AuditChecks(unittest.TestCase):
    def test_geometry_against_actual_prior_hook_records(self):
        count=0;rows=0
        for arm in a.ARMS:
            d=a.read(a.R12/('smoke_'+arm+'_v1')/'results.json')
            for f in d['forward_records']:
                v=a.check_execution(f['execution'],arm,'old_dual' in f['purpose'],f['purpose'])
                rows+=v['FC1_rows'];count+=1
        self.assertEqual(count,42)
        self.__class__.actual_schema_checks={'old_smoke_forward_records':count,'rows_each_FC':rows,'R13_new_results_read':False}

    def test_changed_hook_rejected(self):
        d=a.read(a.R12/'smoke_CC_v1/results.json');x=copy.deepcopy(d['forward_records'][0]['execution'])
        next(iter(x['all12_observed_fc_rows'].values()))['fc1_rows']+=1
        with self.assertRaises(ValueError):a.check_execution(x,'CC',False,'audit_new_initial')

    def test_changed_tail_rejected(self):
        d=a.read(a.R12/'smoke_CC_v1/results.json');x=copy.deepcopy(d['forward_records'][1]['execution'])
        layer=next(iter(x['layers'].values()));layer['observed']['native_input_shapes']['coarse'][-1][2]=3
        with self.assertRaises(ValueError):a.check_execution(x,'CC',False,'audit_old_actual_initial')

    def test_membership_course_corruption_rejected(self):
        m=a.read(a.ROOT/'data_protocol/manifest_v1.json');a.memberships(m)
        m['train_order_reference'][0],m['train_order_reference'][1]=m['train_order_reference'][1],m['train_order_reference'][0]
        with self.assertRaises(ValueError):a.memberships(m)

    def test_common_state_actual_reconstruction(self):
        m=a.read(a.ROOT/'data_protocol/manifest_v1.json');inputs=a.read(a.ROOT/'inputs_v1.json')
        bn,receipt=a.bn_payload(a.ROOT/'preparation/bn_buffers.npz',m,inputs)
        state=a.reconstruct_initial(a.ROOT/'source_snapshot/checkpoint_step_000135930.pt',bn)
        self.assertEqual(len(state),549)
        self.assertEqual(sum(state[n].numel() for n in a.ALLOWED),7252064)
        self.__class__.initial={'original_sha256':a.ORIGINAL_SHA,'new_BN_sha256':inputs['bn_buffers_sha256'],
                               'state_tensor_count':len(state),'state_sha256':a.tensor_digest(state),
                               'frozen_complement_sha256':a.tensor_digest(state,a.ALLOWED),
                               'actual_independent_CPU_reconstruction':True}

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AuditChecks))
    path=Path(__file__).with_name('pre_result_cpu_checks_v1.json');assert not path.exists()
    path.write_text(json.dumps({'tests_run':result.testsRun,'passed':result.wasSuccessful(),
                               'auditor_sha256':a.sha(a.__file__),'test_sha256':a.sha(__file__),
                               'prior_schema':getattr(AuditChecks,'actual_schema_checks',None),
                               'initial_state':getattr(AuditChecks,'initial',None)},indent=2,sort_keys=True)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
