"""CPU synthetic numerical/boundary tests; never load real data or a backbone."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from fit_coarse_cache import (aggregate_metrics, descriptor_forward, evaluate_head, fit_head,
                             fit_normalizer, fold_normalizer, frame_metrics, load_verified_cache,
                             sha_file, train_and_evaluate, validate_cache)
from coarse_head import RuntimeCoarseFlowHead


CONTRACT={'direction':'forward','time_interval':'synthetic same-target interval','units':'original_image_pixels'}


def target(values,valid=None):
    values=torch.tensor(values,dtype=torch.float32).reshape(1,1,-1,2)
    valid=torch.ones(values.shape[:-1],dtype=torch.bool) if valid is None else torch.tensor(valid,dtype=torch.bool).reshape(values.shape[:-1])
    values[~valid]=float('nan')
    return {'schema':'flow_runtime_coarse_targets_v1','flow_contract':dict(CONTRACT),'flow':values,
            'valid':valid,'valid_pixel_count':valid.long()*10}


def cache_fixture():
    frames=[]
    for role,seq,index,x,y in [
        ('fit','fit_scene',1,[[1.,2.,0.],[3.,4.,0.]],[[1.,0.],[3.,2.]]),
        ('fit','fit_scene',2,[[-1.,0.,0.],[2.,3.,0.]],[[0.,1.],[2.,1.]]),
        ('dev','dev_scene',1,[[2.,5.,0.],[0.,1.,0.]],[[2.,3.],[0.,1.]])]:
        frames.append({'target_id':seq+'_'+str(index),'sequence':seq,'role':role,
                       'stages':{'0':{'descriptor':torch.tensor(x).reshape(1,1,2,3),'targets':target(y)}}})
    return {'schema':'flow_coarse_feature_cache_v1','flow_contract':dict(CONTRACT),'frames':frames}


class FitChecks(unittest.TestCase):
    def test_normalization_fit_only_including_empty_GT_and_population_std(self):
        c=cache_fixture()
        c['frames'][0]['stages']['0']['targets']=target([[0,0],[0,0]],valid=[True,False])
        n=fit_normalizer(c['frames'],'0')
        torch.testing.assert_close(n['mean'],torch.tensor([1.25,2.25,0.]))
        torch.testing.assert_close(n['population_std'],torch.tensor([2.1875,2.1875,0.]).sqrt())
        self.assertEqual(n['descriptor_count'],4)
        self.assertEqual(n['divisor'][2].item(),1.)
        changed=copy.deepcopy(c); changed['frames'][-1]['stages']['0']['descriptor'].fill_(1e6)
        n2=fit_normalizer(changed['frames'],'0')
        torch.testing.assert_close(n['mean'],n2['mean'],rtol=0,atol=0)
        torch.testing.assert_close(n['divisor'],n2['divisor'],rtol=0,atol=0)

    def test_dev_changes_do_not_change_fitted_parameters_or_constant(self):
        c=cache_fixture(); d=copy.deepcopy(c)
        d['frames'][-1]['stages']['0']['descriptor']*=8
        d['frames'][-1]['stages']['0']['targets']['flow'].fill_(-100.)
        n=fit_normalizer(c['frames'],'0')
        for method in ('constant_component_median','linear','C16_GELU2'):
            a,_=fit_head(c['frames'],'0',method,CONTRACT,n,steps=40)
            b,_=fit_head(d['frames'],'0',method,CONTRACT,n,steps=40)
            if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
            else:
                for name,value in a.state_dict().items():
                    torch.testing.assert_close(value,b.state_dict()[name],rtol=0,atol=0)

    def test_fold_agrees_on_unseen_inputs_for_both_heads(self):
        torch.manual_seed(777)
        n={'mean':torch.tensor([2.,-3.,.5]),'divisor':torch.tensor([.25,4.,1.])}
        raw=torch.randn(19,3)
        for method,model in [('linear',torch.nn.Linear(3,2)),('C16_GELU2',RuntimeCoarseFlowHead(3,flow_contract=CONTRACT))]:
            folded=fold_normalizer(model,method,n)
            reference=descriptor_forward(model,method,(raw-n['mean'])/n['divisor'])
            torch.testing.assert_close(descriptor_forward(folded,method,raw),reference,atol=2e-6,rtol=2e-6)

    def test_fixed_1000_steps_finite_results_and_safe_small_artifact(self):
        c=cache_fixture()
        # This fixture has an exactly representable, modest affine target;
        # arbitrary labels need not be fitted by the linear control.
        for frame in c['frames']:
            item=frame['stages']['0']; x=item['descriptor']
            item['targets']['flow']=torch.stack((.4*x[...,0]+.2,.3*x[...,1]+.1),dim=-1)
        result,artifact=train_and_evaluate(c,steps=1000)
        self.assertEqual(result['status'],'COMPLETE')
        for method in ('linear','C16_GELU2'):
            item=result['stages']['0']['methods'][method]
            self.assertEqual(item['training']['optimized_steps'],1000)
            self.assertTrue(item['training']['fold_pass'])
            self.assertLess(item['evaluation']['fit']['summary']['frame_equal_component_L1'],.1)
        json.dumps(result,allow_nan=False)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'weights.pt';torch.save(artifact,p)
            restored=torch.load(p,map_location='cpu',weights_only=True)
            self.assertEqual(restored['schema'],'flow_coarse_folded_head_weights_v1')


class MetricsAndGuards(unittest.TestCase):
    def test_abstention_stays_in_GT_eligible_denominator(self):
        labels=target([[3,0],[0,4],[.5,0]])
        p=torch.tensor([[[[0.,0.],[0.,3.],[5.,0.]]]])
        r=frame_metrics(p,labels)
        self.assertEqual(r['gt_motion_eligible_tiles'],2)
        self.assertEqual(r['direction_correct'],1)
        self.assertEqual(r['direction_abstain_on_GT_eligible'],1)
        self.assertEqual(r['direction_accuracy_GT_eligible'],.5)
        self.assertAlmostEqual(r['pred_motion_fraction_valid_tiles'],2/3)

    def test_empty_tiles_frames_and_frame_equal_not_pooled(self):
        a=frame_metrics(torch.tensor([[[[2.,2.]]]]),target([[0,0]]))
        b=frame_metrics(torch.tensor([[[[0.,0.],[0.,0.],[0.,0.]]]]),target([[0,0],[0,0],[0,0]]))
        empty=frame_metrics(torch.zeros(1,1,1,2),target([[0,0]],valid=[False]))
        summary=aggregate_metrics([a,b,empty])
        self.assertEqual(summary['frame_equal_component_L1'],1.)
        self.assertEqual(summary['pooled_component_L1'],.5)
        self.assertEqual(summary['frames_without_valid_GT_tiles'],1)
        self.assertEqual(summary['empty_tiles'],1)
        self.assertIsNone(summary['direction_accuracy_GT_eligible'])
        self.assertIsNone(aggregate_metrics([empty])['frame_equal_EPE'])
        json.dumps(summary,allow_nan=False)

    def test_zero_valid_fit_refuses_training_and_empty_dev_is_null(self):
        c=cache_fixture();c['frames'][-1]['stages']['0']['targets']=target([[0,0],[0,0]],valid=[False,False])
        r,_=train_and_evaluate(c,steps=5)
        self.assertIsNone(r['stages']['0']['methods']['linear']['evaluation']['dev']['summary']['frame_equal_EPE'])
        for f in c['frames']:
            if f['role']=='fit':f['stages']['0']['targets']=target([[0,0],[0,0]],valid=[False,False])
        with self.assertRaises(ValueError):train_and_evaluate(c,steps=5)

    def test_identity_role_schema_and_hash_guards(self):
        c=cache_fixture()
        for mode in ('duplicate','sequence_overlap','schema','contract','counts'):
            d=copy.deepcopy(c)
            if mode=='duplicate':d['frames'][-1]['target_id']=d['frames'][0]['target_id']
            elif mode=='sequence_overlap':d['frames'][-1].update(sequence='fit_scene',target_id='fit_scene_3')
            elif mode=='schema':d['schema']='foreign'
            elif mode=='contract':d['frames'][0]['stages']['0']['targets']['flow_contract']['direction']='backward'
            elif mode=='counts':d['frames'][0]['stages']['0']['targets']['valid_pixel_count'].zero_()
            with self.assertRaises(ValueError):validate_cache(d)
        with self.assertRaises(ValueError):validate_cache(c,strict_population=True)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'cache.pt';torch.save(c,p)
            loaded,_=load_verified_cache(p,sha_file(p),strict_population=False)
            self.assertEqual(len(loaded['frames']),3)
            with self.assertRaises(ValueError):load_verified_cache(p,'0'*64,strict_population=False)

    def test_real_CLI_artifacts_and_existing_output_rejected(self):
        base=cache_fixture(); frames=[]
        for role,sequence_count,per_sequence in (('fit',10,2),('dev',2,4)):
            for sequence in range(sequence_count):
                for index in range(per_sequence):
                    frame=copy.deepcopy(base['frames'][index%2 if role=='fit' else 2])
                    name=role+'_scene'+str(sequence)
                    frame.update(role=role,sequence=name,target_id=name+'_'+str(index))
                    frame['stages']['2']=copy.deepcopy(frame['stages']['0'])
                    frames.append(frame)
        base['frames']=frames
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory)/'cache.pt';out=Path(directory)/'new_run'
            torch.save(base,cache)
            cmd=[sys.executable,'-B',str(Path(__file__).with_name('fit_coarse_cache.py')),
                 '--cache',str(cache),'--cache-sha256',sha_file(cache),'--output-dir',str(out)]
            run=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
            self.assertEqual(run.returncode,0,run.stderr)
            result=json.loads((out/'results.json').read_text())
            self.assertEqual(result['status'],'COMPLETE')
            self.assertEqual(result['recipe']['steps_per_trainable_head'],1000)
            self.assertEqual(result['weights_sha256'],sha_file(out/'weights.pt'))
            weights=torch.load(out/'weights.pt',map_location='cpu',weights_only=True)
            self.assertEqual(set(weights['stages']),{'0','2'})
            before=sha_file(out/'results.json')
            rerun=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=15)
            self.assertNotEqual(rerun.returncode,0)
            self.assertIn('refusing overwrite',rerun.stderr)
            self.assertEqual(sha_file(out/'results.json'),before)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
