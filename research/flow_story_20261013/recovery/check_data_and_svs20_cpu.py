#!/usr/bin/env python3
"""Synthetic CPU guards and sealed metadata checks; no dataset arrays loaded."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
sys.dont_write_bytecode=True
import numpy as np
import torch
import build_data_contract as data
import collect_svs20 as collector
import compare_svs_2_to_20 as stability

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[1]/'flow_story_20261011/preflight/data_protocol/artifacts'
BREADTH=HERE.parent/'breadth'
MEASUREMENTS=[]


def make_payloads():
    return data.make_payloads(SOURCE/'common_targets.json',SOURCE/'train_auxiliary_roles.csv',
                              BREADTH/'svs_gate_fit2.json',BREADTH/'gate_dev32.json')


def synthetic_counts(directory):
    fit=collector.fixed_fit_manifest();ids=fit['target_ids'];paths=['synthetic_mlp_'+str(i) for i in range(12)]
    root=Path(directory)/'counts';root.mkdir()
    records=[];rng=np.random.RandomState(74)
    for target in ids:
        for path in paths:
            value=rng.randint(0,11,(1,4,5,64)).astype(np.uint8)
            destination=root/(collector.digest([target,path])+'.npy');np.save(destination,value,allow_pickle=False)
            records.append({'target_id':target,'mlp_path':path,'file':destination.name,'bytes':destination.stat().st_size,
                'sha256':collector.sha(destination),'shape_BHWF':list(value.shape),'dtype':'uint8',
                'complete_T':10,'binary_guard_passed':True})
    bindings={'data':{'synthetic_fixture':True},'source':{'synthetic_fixture':True},
              'BN':{'policy':'P_paper'},'geometry':{'name':'full9'}}
    provenance={'checkpoint_sha256':collector.CHECKPOINT_SHA,'fit_ids_sha256':collector.digest(ids),'split':'train'}
    for key,part in (('data_contract_sha256','data'),('source_sha256','source'),('bn_policy_sha256','BN'),('geometry_sha256','geometry')):
        provenance[key]=collector.digest(bindings[part])
    m={'schema':'flow_svs_counts_v1','status':'ALL_NATIVE_MLP_COUNTS_COMPLETE',
       'collector_sha256':collector.sha(collector.__file__),'evaluator_sha256':collector.EVALUATOR_SHA,
       'static_methods_sha256':collector.STATIC_SHA,'fit_manifest_sha256':collector.FIT_SHA,
       'upstream_collector_sha256':collector.UPSTREAM_COLLECTOR_SHA,'data_manifest_contract_sha256':collector.DATA_CONTRACT_SHA,
       'prior_root':str(HERE.parents[1]/'flow_story_20261012'),'breadth_root':str(BREADTH),
       'counts_root':str(root),'fit_ids':ids,'mlp_paths':paths,'files':records,'provenance':provenance,
       'shapes':{p:{'input_channels':16,'hidden_channels':64,'output_channels':16,'T':10} for p in paths},
       'binding_contracts':bindings,'BN_buffers_unchanged':True,'training_performed':False,'synthetic_fixture':True}
    path=Path(directory)/'counts_manifest.json';collector.atomic(path,m)
    return path,collector.sha(path),m


class Checks(unittest.TestCase):
    def test_sealed_members_reconstruct_exactly(self):
        payloads=make_payloads()
        self.assertEqual(data.publish(payloads,HERE/'manifests',True)['status'],'VERIFIED')
        m=json.loads(payloads['data_contract.json'])
        self.assertEqual(m['counts'],{'fit20':20,'check64':64,'recovery4573':4573,'BN703':703,'historical_dev32':32})
        self.assertEqual(m['target_intersections']['fit20__recovery4573'],20)
        self.assertTrue(all(v==0 for k,v in m['target_intersections'].items() if k!='fit20__recovery4573'))
        self.assertFalse(m['image_or_GT_array_read'])

    def test_wrong_metadata_pin_and_immutable_publication(self):
        payloads=make_payloads()
        with tempfile.TemporaryDirectory() as tmp:
            bad=Path(tmp)/'roles.csv';bad.write_text('not original roles')
            with self.assertRaises(ValueError):data.make_payloads(SOURCE/'common_targets.json',bad,BREADTH/'svs_gate_fit2.json',BREADTH/'gate_dev32.json')
            out=Path(tmp)/'sealed';data.publish(payloads,out)
            (out/'svs_fit20.json').write_text('changed')
            with self.assertRaises(ValueError):data.publish(payloads,out)

    def test_core_formula_helpers_AST_unchanged(self):
        old=ast.parse((BREADTH/'collect_svs.py').read_text());new=ast.parse(Path(collector.__file__).read_text())
        names=('binary_time_counts','ranks_from_counts','CountHooks','bounded_worker')
        for name in names:
            a=next(x for x in old.body if getattr(x,'name',None)==name)
            b=next(x for x in new.body if getattr(x,'name',None)==name)
            self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))
        MEASUREMENTS.append({'test':'core_AST_unchanged','functions_or_classes':list(names),'matched':True})

    def test_fit20_reconstructed_via_original_metric_interface(self):
        import csv
        rows=list(csv.DictReader((SOURCE/'train_auxiliary_roles.csv').open()))
        role_map={r['target_id']:r['auxiliary_role'] for r in rows}
        metrics=types.SimpleNamespace(CANONICAL_SHA=data.CANONICAL_SHA,AUX_ROLES_SHA=data.ROLES_SHA,
            canonical_rows=lambda unused:[dict(r,protocol_split='train') for r in rows],
            auxiliary_roles=lambda path,rows:role_map)
        selected,sample=collector.fixed_fit_rows(None,None,metrics)
        self.assertEqual(len(selected),20);self.assertEqual([r['target_id'] for r in selected],sample['target_ids'])

    def test_preflight_rejects_other_checkpoint_BN_geometry_before_access(self):
        for override in ({'checkpoint_sha256':'0'*64},{'bn':'D'},{'geometry':'crop_center9'}):
            values={'checkpoint_sha256':collector.CHECKPOINT_SHA,'bn':'P_paper','geometry':'full9'};values.update(override)
            with self.assertRaises(ValueError):collector.preflight(types.SimpleNamespace(**values))

    def test_complete_T_counts_vs_original_binary_reference(self):
        generator=torch.Generator().manual_seed(761)
        spikes=torch.randint(0,2,(10,1,5,6,8),generator=generator).double()
        counts=collector.binary_time_counts(spikes,8)
        actual=collector.ranks_from_counts(counts,3)
        svs=collector.static_module(HERE.parents[1]/'flow_story_20261012')._svs()
        expected=svs.svs_ranks(spikes,channel_chunk=3)
        self.assertTrue(torch.equal(actual,expected))
        for bad in (spikes[:-1],spikes+.1):
            with self.assertRaises(ValueError):collector.binary_time_counts(bad,8)
        bad=counts.clone();bad[0,0,0,0]=11
        with self.assertRaises(ValueError):collector.ranks_from_counts(bad)

    def test_manifest_complete_coverage_and_BN_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,digest,m=synthetic_counts(tmp)
            self.assertEqual(len(collector.load_counts_manifest(path,digest)['files']),240)
            for change in ('missing','duplicate','BN','checkpoint','membership'):
                bad=copy.deepcopy(m)
                if change=='missing':bad['files'].pop()
                elif change=='duplicate':bad['files'][-1]=bad['files'][0]
                elif change=='BN':bad['binding_contracts']['BN']['policy']='D'
                elif change=='checkpoint':bad['provenance']['checkpoint_sha256']='0'*64
                else:bad['fit_ids']=bad['fit_ids'][:-1]
                collector.atomic(path,bad)
                with self.assertRaises(ValueError):collector.load_counts_manifest(path,collector.sha(path))

    def test_actual_bounded_CPU_worker_20_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,digest,m=synthetic_counts(tmp)
            out=Path(tmp)/'out';out.mkdir()
            result=collector.run_scoring(path,digest,out,60,8)
            self.assertEqual(result['status'],'SVS_REPLAY_READY');self.assertEqual(result['sample_count'],20)
            artifact=json.loads((out/'svs_replay.json').read_text())
            self.assertEqual(len(artifact['blocks']),12)
            self.assertTrue(all(b['sample_count']==20 for b in artifact['blocks'].values()))
            self.assertFalse(artifact['evidence']['cuda_initialized'])
            MEASUREMENTS.append({'test':'synthetic20_bounded_worker','samples':20,'MLPs':12,'count_files':240,
                                 'status':result['status'],'cuda_initialized':False})

    def test_correlation_ties_constant_and_overlap(self):
        self.assertEqual(stability.average_ranks([3,1,1,2]),[4,1.5,1.5,3])
        self.assertIsNone(stability.correlation([1,1],[2,3]))
        self.assertAlmostEqual(stability.correlation([1,2,3],[3,2,1]),-1.)
        fit=collector.fixed_fit_manifest();oldids=fit['nested_fit2_target_ids'];newids=fit['target_ids']
        svs=collector.static_module(HERE.parents[1]/'flow_story_20261012')._svs()
        def artifact(ids):
            values=[i%5 for i in range(64)]
            blocks={str(i):{'schema':'flow_svs_score_v1','hidden_width':64,'sample_ids':ids,'sample_count':len(ids),
                    'batch_sizes':[1]*len(ids),'rank_sums':[v*len(ids) for v in values],'scores':values,'formula':svs.FORMULA} for i in range(12)}
            prov={'checkpoint_sha256':collector.CHECKPOINT_SHA,'fit_ids_sha256':stability.digest(ids),'split':'train','bn':'same'}
            return {'schema':'flow_breadth_svs_replay_v1','fit_ids':ids,'provenance':prov,'blocks':blocks,
                    'evidence':{'sample_count':len(ids),'CPU_only_SVD':True,'cuda_initialized':False,'formula':svs.FORMULA}}
        old,new=artifact(oldids),artifact(newids);out=stability.compare(old,new,fit)
        self.assertTrue(all(v['channel_micro_kept_overlap']==1 for v in out['aggregate_channel_overlap_not_PE_cost'].values()))
        new['provenance']['bn']='different'
        with self.assertRaises(ValueError):stability.compare(old,new,fit)


if __name__=='__main__':
    torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    output={'schema':'flow_recovery_data_svs20_cpu_checks_v1','passed':result.wasSuccessful(),'tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'measurements':MEASUREMENTS,
        'real_event_GT_check64_or_valid_arrays_read':False,'real_model_forward_or_training_executed':False,
        'CUDA_initialized':torch.cuda.is_initialized(),'torch':str(torch.__version__),
        'source_sha256':{name:data.sha(HERE/name) for name in
            ('build_data_contract.py','collect_svs20.py','compare_svs_2_to_20.py','check_data_and_svs20_cpu.py')}}
    (HERE/'data_svs20_cpu_checks.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
