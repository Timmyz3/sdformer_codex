"""CPU checks of new control wiring/guards; no checkpoint or dataset arrays."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import unittest

HERE = Path(__file__).resolve().parent
FAST = HERE.parent/'fast_warmstart'
MOTION = HERE.parents[1]/'flow_story_20261013/motion_groups'
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(MOTION))
import run_selection_controls as runner


def prior():
    return json.loads((FAST/'runtime_v1/results.json').read_text())


def current_fixture():
    old = prior()['frames'][0]
    f = copy.deepcopy(old)
    f['arms'] = {n: f['arms'][n] for n in runner.ANCHORS}
    for name, branch, policy in runner.ARM_SPECS:
        if name in f['arms']: continue
        reference = 'independent_mean' if branch == 'mean' else 'independent_trained'
        rec = copy.deepcopy(old['arms'][reference])
        rec['runtime_selector']['policy'] = policy
        if policy == 'motion':
            rec['execution'][0]['keep'] = copy.deepcopy(old['arms']['motion_trained']['execution'][0]['keep'])
        f['arms'][name] = rec
    return f, old


class ControlsChecks(unittest.TestCase):
    def test_seven_fixed_arms_and_forward_budget(self):
        self.assertEqual(len(runner.ARM_SPECS), 7)
        self.assertEqual(len(runner.ARM_SPECS)*len(runner.TARGETS)+sum(p is not None for _, _, p in runner.ARM_SPECS), 33)
        self.assertEqual({(b,p) for n,b,p in runner.ARM_SPECS if n not in runner.ANCHORS},
                         {('mean','motion'),('trained','horizontal_local'),('trained','rotated90')})

    def test_actual_prior_contract_and_incomplete_or_changed_population_rejected(self):
        c = json.loads((FAST/'runtime_contract_v1.json').read_text()); old = prior()
        self.assertEqual(runner.sha(FAST/'runtime_contract_v1.json'), runner.PARENT_CONTRACT_SHA)
        self.assertEqual(runner.sha(FAST/'runtime_v1/results.json'), runner.PRIOR_RESULTS_SHA)
        runner.bind_prior(c, old)
        for key, value in [('status','RUNNING'), ('complete_frames',3), ('targets',list(reversed(runner.TARGETS))),
                           ('contract_sha256','0'*64), ('BN_buffers_unchanged',False)]:
            bad=copy.deepcopy(old);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): runner.bind_prior(c,bad)

    def test_same_information_cost_and_branch_identity_guards(self):
        f,_ = current_fixture();self.assertTrue(runner.verify_frame_service(f))
        mutations = [
            lambda r:r['execution'][0].update(slow_points=899),
            lambda r:r['execution'][0]['observed'].update(fast_fc_rows=2999),
            lambda r:r['runtime_selector'].update(policy='motion'),
            lambda r:r['runtime_selector'].update(scores=[[0]*5]*4),
            lambda r:r['runtime_selector'].update(GT_or_final_flow_used=True)]
        for i,change in enumerate(mutations):
            bad=copy.deepcopy(f);change(bad['arms']['rotated90_trained'])
            with self.subTest(i=i),self.assertRaises(ValueError):runner.verify_frame_service(bad)
        bad=copy.deepcopy(f);bad['arms']['motion_mean']['execution'][0]['keep'][0][0] ^= True
        with self.assertRaisesRegex(ValueError,'Mean replacement'):runner.verify_frame_service(bad)

    def test_prior_prediction_metric_and_mask_guards(self):
        f,old=current_fixture();self.assertTrue(runner.verify_prior_frame(f,old)['all_four_anchor_predictions_and_metrics_exact'])
        for kind in ('prediction','GT_count','mask'):
            bad=copy.deepcopy(f)
            if kind=='prediction':bad['arms']['independent_trained']['final_flow_sha256']='0'*64
            if kind=='GT_count':bad['arms']['native_dense']['metrics']['regions']['all_valid']['valid_pixels']-=1
            if kind=='mask':bad['arms']['motion_mean']['execution'][0]['keep'][0][0]^=True
            with self.subTest(kind=kind),self.assertRaises(ValueError):runner.verify_prior_frame(bad,old)

    def test_contrast_null_coverage_and_sign(self):
        arms=copy.deepcopy(prior()['arms'])
        arms['motion_mean']=copy.deepcopy(arms['motion_trained'])
        for name in ('horizontal_local_trained','rotated90_trained'):
            arms[name]=copy.deepcopy(arms['independent_trained'])
        r=runner.paired_contrasts(arms)
        self.assertEqual(r['motion_trained_minus_motion_mean']['all_valid']['delta_frame_mean_AEE'],0)
        self.assertIsNone(r['motion_trained_minus_motion_mean']['direction_abstain']['delta_frame_mean_AEE'])
        self.assertLess(r['motion_trained_minus_independent_trained']['L']['delta_frame_mean_AEE'],0)
        arms['motion_mean']['summary']['regions']['L']['valid_pixels']-=1
        with self.assertRaisesRegex(ValueError,'population'):runner.paired_contrasts(arms)

    def test_real_four_policy_interface_keeps_tail_cost_and_common_information(self):
        import torch
        import coarse_head
        import runtime_group_adapter as adapter
        torch.set_num_threads(2)
        h=coarse_head.RuntimeCoarseFlowHead(384,flow_contract={
            'direction':'synthetic forward','time_interval':'synthetic T10','units':'original_image_pixels'})
        with torch.no_grad():
            for p in h.parameters():p.zero_()
            h.fc2.bias.copy_(torch.tensor([4.,3.]))
        h.eval().requires_grad_(False)
        u=torch.arange(10,dtype=torch.float32).view(10,1,1,1,1).expand(adapter.SHAPE).contiguous()/10
        records=[]
        with torch.inference_mode():
            for name in adapter.POLICIES:
                policy=adapter.RuntimeGroupPolicy(h,name)
                mask=policy(adapter.LAYER,u);rec=policy.records[0];records.append(rec)
                self.assertEqual(tuple(mask.shape),(4,5));self.assertEqual(mask.dtype,torch.bool)
                actual=sum(rec['points_per_tile'][y][x] for y in range(4) for x in range(5) if mask[y,x])
                self.assertEqual(actual,900);self.assertEqual(rec['selection']['slow_service_cost'],900)
                self.assertEqual(rec['points_per_tile'][-1],[27,27,27,27,12])
                for group in rec['proposal']['groups']:
                    self.assertEqual(len({bool(mask[i//5,i%5]) for i in group['members']}),1)
        for rec in records[1:]:
            for key in ('input_sha256','descriptor_sha256','flow_dxdy_pixels','scores','points_per_tile'):
                self.assertEqual(rec[key],records[0][key])
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':
    started=time.monotonic()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ControlsChecks)
    outcome=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'schema':'flow_selection_controls_CPUchecks_v1','status':'PASS' if outcome.wasSuccessful() else 'FAIL',
            'tests_run':outcome.testsRun,'failures':len(outcome.failures),'errors':len(outcome.errors),
            'wall_seconds':time.monotonic()-started,'scope':'Synthetic selector plus published JSON/contract guards; no full model, GT arrays, SSH or GPU',
            'source_sha256':runner.sha(runner.__file__),'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE/'cpu_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if outcome.wasSuccessful() else 1)
