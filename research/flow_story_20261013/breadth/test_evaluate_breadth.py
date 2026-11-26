#!/usr/bin/env python3
"""CPU provenance/metric/BN checks; actual remote metadata, no remote weights."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
sys.dont_write_bytecode=True
import copy
import json
from pathlib import Path
import unittest
import numpy as np
import torch
import evaluate_breadth as e

HERE=Path(__file__).resolve().parent
RESEARCH=HERE.parents[1]
PRIOR=RESEARCH/'flow_story_20261012'
ART=RESEARCH/'flow_story_20261011/preflight/data_protocol/artifacts'


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native,cls.trainer,cls.metrics,cls.pins=e.support_modules(PRIOR)
        cls.rows,cls.records,cls.roles,cls.sample=e.selected_rows(ART/'common_targets.json',ART/'train_auxiliary_roles.csv',cls.metrics)
        cls.train,cls.hashes,cls.binding,_=cls.native.validate_native_bundle('/data/flow_a/data/DSEC_native_v1',
            ART/'common_targets.json',PRIOR/'baseline/native_setup/acceptance.json',
            'b1d87de8fd48b710fe5a49dd37d6d64427d2fe111df249efa73ba2299f454540',
            ART.parent/'archive_data_audit.json',False)
        cls.binding.update(entrypoint_sha256=cls.pins['baseline/train_native_baseline.py'],
            adapter_sha256=cls.pins['baseline/native_data_acceptance.py'],training_core_sha256=cls.trainer.CORE_SHA,
            checkpoint_schema=e.SCHEMA,checkpoint_creator=e.CREATOR)
        cls.snapshot_run=json.loads((HERE/'remote_snapshot/run_contract.json').read_text())
        cls.side=json.loads((HERE/'remote_snapshot/checkpoint_step_000135930.json').read_text())
        cls.sources=cls.snapshot_run['contract']['source_sha256']

    def validate(self,run=None,side=None,binding=None):
        return e.validate_contract(run or self.snapshot_run,side or self.side,binding or self.binding,
                                   self.sources,self.trainer,self.train,self.hashes)

    def test_actual_native_interim_contract_metadata(self):
        config=self.validate()
        self.assertEqual(config['data_contract']['encoding_id'],self.native.ENCODING_ID)
        self.assertEqual(self.side['completed_optimizer_steps'],135930)
        self.assertEqual(self.side['checkpoint_sha256'],'d5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30')

    def test_legacy_checkpoint_and_encoding_rejected(self):
        side=copy.deepcopy(self.side);side['schema_version']='flow_a_native_v2_checkpoint_v1'
        with self.assertRaisesRegex(ValueError,'Old/engineering'):self.validate(side=side)
        binding=copy.deepcopy(self.binding);binding['encoding_id']='old_absolute_timestamp_fp32'
        with self.assertRaisesRegex(ValueError,'encoding'):self.validate(binding=binding)
        run=copy.deepcopy(self.snapshot_run);run['contract']['training_target_ids_sha256']='0'*64
        run['contract_sha256']=e.digest(run['contract']);side=copy.deepcopy(self.side);side['contract_sha256']=run['contract_sha256']
        with self.assertRaisesRegex(ValueError,'membership'):self.validate(run,side)

    def test_fixed32_train_gate_dev_no_valid(self):
        self.assertEqual(len(self.rows),32)
        self.assertEqual(len({r['target_id'] for r in self.rows}),32)
        self.assertTrue(all(self.roles[r['target_id']]=='gate_development' and r['protocol_split']=='train' for r in self.rows))
        self.assertEqual(self.sample['sequence_counts'],{'zurich_city_01_a':7,'zurich_city_05_a':25})

    def test_complete_regions_bind_native_GT_and_mask(self):
        regions=e.read_pin(HERE/'remote_snapshot/regions_gatefit4573.json',
            '8d64b18d3c69adb6c2a1bc146f55c254a2f6bbf02e50366b069f1d1d57fa8d26')
        result=e.region_source_binding(regions,self.records,self.roles,self.hashes,self.native,self.metrics)
        self.assertEqual(result['GT_mask_native_source_hash_checks'],9146)
        bad=copy.deepcopy(regions);bad['data_evidence'][0]['gt_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'native acceptance'):e.region_source_binding(bad,self.records,self.roles,self.hashes,self.native,self.metrics)
        bad=copy.deepcopy(regions);bad['fit_target_ids']=bad['fit_target_ids'][:4]
        with self.assertRaises(ValueError):e.region_source_binding(bad,self.records,self.roles,self.hashes,self.native,self.metrics)

    def test_case_no_current_stat_tile_and_no_duplicate(self):
        c={'id':'dense','method':'dense','bn':'P_paper'}
        self.assertEqual(len(e.validate_cases({'schema':'flow_breadth_cases_v1','cases':[c]})),1)
        with self.assertRaisesRegex(ValueError,'Duplicate'):e.validate_cases({'schema':'flow_breadth_cases_v1','cases':[c,c]})
        with self.assertRaisesRegex(ValueError,'frozen BN'):e.validate_cases({'schema':'flow_breadth_cases_v1','cases':[dict(c,method='tile',keep_fraction=.5)]})

    def test_all_valid_metric_keeps_zero_event_pixels(self):
        m=self.metrics.MetricAccumulator();gt=np.zeros((2,4,5));prediction=gt.copy();prediction[0]=3;prediction[1]=4
        valid=np.ones((4,5),dtype=bool);valid[0,0]=False
        m.update('test','test',prediction,gt,{'all_valid':valid})
        summary=m.summary()['regions']['all_valid']
        self.assertEqual(summary['valid_pixels'],19);self.assertEqual(summary['frame_mean_AEE'],5.)

    def test_BN_running_D_equal_current_stat_no_mutation(self):
        model=torch.nn.Sequential(torch.nn.BatchNorm2d(2),torch.nn.Dropout(.5))
        with torch.no_grad():model[0].num_batches_tracked.fill_(3);model[0].running_mean.fill_(2)
        x=torch.randn(1,2,3,4)
        code=copy.deepcopy(model);deploy=copy.deepcopy(model);paper=copy.deepcopy(model)
        e.apply_bn(code,'P_code',torch);e.apply_bn(deploy,'D',torch);e.apply_bn(paper,'P_paper',torch)
        signature=e.bn_signature(code,torch)
        with torch.inference_mode():self.assertTrue(torch.equal(code(x),deploy(x)));paper(x)
        self.assertEqual(signature,e.bn_signature(code,torch));self.assertIsNone(paper[0].running_mean)
        self.assertFalse(paper[1].training)

    def test_input_polarity_minmax_and_empty_guard(self):
        events=np.arange(10*3*4,dtype=np.float32).reshape(10,3,4)-50
        x=e.prepare_input(events,torch.device('cpu'),torch)
        self.assertEqual(tuple(x.shape),(1,10,2,3,4));self.assertTrue(torch.isfinite(x).all())
        self.assertTrue((x>=0).all());self.assertLessEqual(x.max(),1)
        with self.assertRaisesRegex(ValueError,'Empty'):e.prepare_input(np.zeros_like(events),torch.device('cpu'),torch)

    def test_linear_observer_counts_leaf_factors_once(self):
        model=torch.nn.Sequential(torch.nn.Linear(4,2,bias=False),torch.nn.Linear(2,6,bias=False))
        with torch.inference_mode(),e.linear_observer(model,torch) as records:model(torch.ones(10,1,3,5,4))
        self.assertEqual(records['0']['rows'],150);self.assertEqual(records['1']['rows'],150)
        self.assertEqual(len(records),2)

    def test_compact_actual_mask_is_exact_and_rejects_mismatch(self):
        grid=np.array([[True,False],[False,True]])
        spatial=np.repeat(np.repeat(grid,9,axis=0),9,axis=1)[:11,:13]
        raw={'unit_keep':grid.tolist(),'spatial_keep':spatial.tolist()}
        packed=e.compact_policy_records([raw])[0]
        self.assertEqual(packed['unit_keep'],raw['unit_keep']);self.assertNotIn('spatial_keep',packed)
        self.assertIn('spatial_keep',raw);self.assertEqual(packed['spatial_mask_encoding']['shape'],[11,13])
        raw['spatial_keep'][0][0]=False
        with self.assertRaisesRegex(ValueError,'mask differs'):e.compact_policy_records([raw])


if __name__=='__main__':
    torch.set_num_threads(1)
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    out={'schema':'flow_breadth_evaluator_cpu_checks_v1','passed':r.wasSuccessful(),'tests_run':r.testsRun,
         'errors':len(r.errors),'failures':len(r.failures),'GPU_initialized':torch.cuda.is_initialized(),
         'entry_sha256':e.sha(HERE/'evaluate_breadth.py'),'sample_sha256':e.sha(HERE/'gate_dev32.json'),
         'support_manifest_sha256':e.sha(HERE/'support_source_pins.json'),
         'scope':'Real native acceptance/ep22.655 metadata and region9146-source hashes; synthetic CPU metrics/BN/input/hooks. No local checkpoint tensor load, full model inference or GPU.'}
    (HERE/'evaluator_cpu_checks.json').write_text(json.dumps(out,indent=2)+'\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
