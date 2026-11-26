#!/usr/bin/env python3
"""CPU synthetic unit checks; no real dataset/checkpoint/GPU collection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='2'
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
sys.dont_write_bytecode=True
import numpy as np
import torch
from torch import nn
import collect_svs as c

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parents[1]/'flow_story_20261012'
PROBE=HERE.parents[1]/'flow_story_20261011/preflight/cpu_numerics/native_mlp_probe.py'
spec=importlib.util.spec_from_file_location('collect_native_probe',PROBE)
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def fixture(directory):
    """Disposable metadata only: synthetic counts, no real collected artifact."""
    root=Path(directory);counts_root=root/'counts';counts_root.mkdir()
    sample=json.loads((HERE/'svs_gate_fit2.json').read_text());ids=sample['target_ids'];paths=['a','b']
    shapes={p:{'hidden_channels':4,'input_channels':2,'output_channels':2,'T':10} for p in paths}
    bindings={'data':{'fixture':True},'source':{'fixture':True},'BN':{'policy':'D','fixture':True},'geometry':{'fixture':True}}
    provenance={'checkpoint_sha256':'1'*64,'fit_ids_sha256':c.digest(ids),'split':'train'}
    for k,v in (('data_contract_sha256','data'),('source_sha256','source'),('bn_policy_sha256','BN'),('geometry_sha256','geometry')):
        provenance[k]=c.digest(bindings[v])
    files=[];expected={}
    svs=c.static_module(PRIOR)._svs()
    for p in paths:
        accumulator=svs.SVSAccumulator(ids,2)
        for index,target in enumerate(ids):
            generator=torch.Generator().manual_seed(index+len(p)+12)
            spikes=torch.randint(0,2,(10,1,3,5,4),generator=generator).float()
            accumulator.update(spikes,[target])
            counts=c.binary_time_counts(spikes)
            name=c.digest([target,p])+'.npy';file=counts_root/name
            with file.open('wb') as f:np.save(f,counts.numpy(),allow_pickle=False)
            files.append({'target_id':target,'mlp_path':p,'file':name,'bytes':file.stat().st_size,'sha256':c.sha(file),
                'shape_BHWF':list(counts.shape),'dtype':'uint8','complete_T':10,'binary_guard_passed':True})
        expected[p]=accumulator.result()
    m={'schema':'flow_svs_counts_v1','status':'ALL_NATIVE_MLP_COUNTS_COMPLETE','collector_sha256':c.sha(c.__file__),
       'evaluator_sha256':c.EVALUATOR_SHA,'static_methods_sha256':c.STATIC_SHA,'fit_manifest_sha256':c.FIT_SHA,
       'prior_root':str(PRIOR),'counts_root':str(counts_root),'fit_ids':ids,'mlp_paths':paths,'shapes':shapes,'files':files,
       'provenance':provenance,'binding_contracts':bindings,'BN_buffers_unchanged':True,'training_performed':False,
       'synthetic_test_fixture':True}
    path=root/'counts_manifest.json';c.atomic(path,m)
    return path,c.sha(path),m,expected


class Tests(unittest.TestCase):
    def test_counts_exact_mean_and_original_rank_definition(self):
        svs=c.static_module(PRIOR)._svs()
        for shape in ((10,1,3,5,4),(10,1,9,11,8)):
            spikes=torch.randint(0,2,shape).float()
            counts=c.binary_time_counts(spikes)
            self.assertTrue(torch.equal(counts.double()/10,spikes.double().mean(0)))
            self.assertTrue(torch.equal(c.ranks_from_counts(counts,3),svs.svs_ranks(spikes)))
        zero=torch.zeros(10,1,3,5,4)
        self.assertEqual(c.ranks_from_counts(c.binary_time_counts(zero)).tolist(),[[0,0,0,0]])

    def test_binary_and_complete_time_guards(self):
        spikes=torch.zeros(10,1,3,5,4)
        for bad in (spikes[:9],spikes.expand(-1,2,-1,-1,-1),spikes+0.5,spikes+float('nan')):
            with self.assertRaises(ValueError):c.binary_time_counts(bad)
        with self.assertRaises(ValueError):c.ranks_from_counts(torch.full((1,3,5,4),11,dtype=torch.uint8))

    def test_actual_native_hooks_each_sn2_exactly_once(self):
        class Pair(nn.Module):
            def __init__(self):
                super().__init__();self.a=probe.make_mlp(8,torch.float64);self.b=probe.make_mlp(8,torch.float64)
            def forward(self,x):return self.b(self.a(x))
        model=Pair().eval();svs=c.static_module(PRIOR)._svs();paths=['a','b']
        shapes={p:svs.validate_native_mlp(model.get_submodule(p)) for p in paths}
        before=copy.deepcopy(model.state_dict())
        with tempfile.TemporaryDirectory() as temp:
            with torch.inference_mode(),c.CountHooks(model,Path(temp),paths,shapes) as hooks:
                for target in ('synthetic0','synthetic1'):
                    hooks.begin(target);model(torch.randn(10,1,3,5,8,dtype=torch.float64));hooks.end()
                self.assertEqual(len(hooks.records),4)
                for r in hooks.records:
                    self.assertEqual(r['shape_BHWF'],[1,3,5,32])
                    self.assertEqual(c.sha(Path(temp)/r['file']),r['sha256'])
            self.assertTrue(all(not m._forward_hooks for m in model.modules()))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))

    def test_fixed_real_membership_is_train_fit_not_dev(self):
        evaluation=c.import_pinned('collect_test_eval','evaluate_breadth.py',c.EVALUATOR_SHA)
        spec=importlib.util.spec_from_file_location('collect_test_metrics',PRIOR/'evaluation/flow_eval_contract.py')
        metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
        artifacts=HERE.parents[1]/'flow_story_20261011/preflight/data_protocol/artifacts'
        rows,sample=c.fixed_fit_rows(artifacts/'common_targets.json',artifacts/'train_auxiliary_roles.csv',metrics)
        self.assertEqual(len(rows),2);self.assertEqual(len({r['sequence'] for r in rows}),2)
        self.assertTrue(all(r['protocol_split']=='train' for r in rows))

    def test_cpu_worker_produces_exact_replay_and_is_consumable(self):
        with tempfile.TemporaryDirectory() as temp:
            path,checksum,m,expected=fixture(temp);out=Path(temp)/'score';out.mkdir()
            status=c.run_scoring(path,checksum,out,30,3)
            self.assertEqual(status['status'],'SVS_REPLAY_READY')
            artifact=json.loads((out/'svs_replay.json').read_text())
            self.assertEqual(artifact['blocks'],expected)
            methods=c.static_module(PRIOR)
            methods.validate_svs_replay(artifact,m['provenance'],m['mlp_paths'],m['shapes'],methods._svs())
            self.assertFalse(artifact['evidence']['cuda_initialized'])

    def test_worker_error_or_timeout_never_publishes_usable_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            path,checksum,m,_=fixture(temp);out=Path(temp)/'score';out.mkdir()
            # Corrupt one byte after the count receipt was bound.
            raw=Path(m['counts_root'])/m['files'][0]['file'];raw.write_bytes(raw.read_bytes()+b'x')
            status=c.run_scoring(path,checksum,out,30,3)
            self.assertEqual(status['status'],'CPU_SVD_ERROR_NOT_METHOD_FAILURE')
            self.assertFalse((out/'svs_replay.json').exists())
            out2=Path(temp)/'timeout';out2.mkdir()
            status=c.run_scoring(path,checksum,out2,.001,3)
            self.assertEqual(status['status'],'PENDING_CPU_SVD_TIME_LIMIT')
            self.assertFalse((out2/'svs_replay.json').exists())

    def test_deadline_kills_cpu_descendants(self):
        with tempfile.TemporaryDirectory() as temp:
            pidfile=Path(temp)/'pids.json'
            child='import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'
            script='import subprocess,sys,os,time,json;from pathlib import Path;p=subprocess.Popen([sys.executable,"-c",'+repr(child)+']);Path('+repr(str(pidfile))+').write_text(json.dumps([os.getpid(),p.pid]));time.sleep(60)'
            result=c.bounded_worker([sys.executable,'-c',script],.5,Path(temp)/'log')
            self.assertTrue(result['timed_out']);ids=json.loads(pidfile.read_text());time.sleep(.05)
            for pid in ids:
                path=Path('/proc')/str(pid)/'stat'
                if path.exists():self.assertEqual(path.read_text().split()[2],'Z')


def main():
    torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    out={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'errors':[str(x) for x in result.errors],
         'failures':[str(x) for x in result.failures],'scope':'CPU synthetic module/counts/metadata only; no real checkpoint or data forward',
         'cuda_initialized':torch.cuda.is_initialized(),'optimizer_steps':0,'torch':str(torch.__version__),
         'source_sha256':{p.name:c.sha(p) for p in (Path(c.__file__),Path(__file__),HERE/'svs_gate_fit2.json')}}
    c.atomic(HERE/'collect_svs_cpu_results.json',out)
    return 0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1


if __name__=='__main__':raise SystemExit(main())
