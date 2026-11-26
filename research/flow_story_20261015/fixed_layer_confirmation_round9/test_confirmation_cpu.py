#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch
import run_confirmation as runner
import select_members

HERE=Path(__file__).resolve().parent
R8=runner.load_r8();C=R8.load_collector();PROBE=C.load_probe();PARENT=PROBE.load_scope_runner()
FP=PROBE.BASE/'scope_extension_v1/test_scope_extension_cpu.py';sys.path.insert(0,str(FP.parent))
spec=importlib.util.spec_from_file_location('_r9_fixture',FP);FIXTURE=importlib.util.module_from_spec(spec);spec.loader.exec_module(FIXTURE)
MERGE=PARENT.load_frozen_merge()


class Tests(unittest.TestCase):
    def test_four_arms_and_dual_services(self):
        model,targets,shapes,x,functional=FIXTURE.build();before=PARENT.source_digest(model)
        for arm,rows in [('native',18480),('FF',18480),('CC',7760),('fine_s2b2',9100)]:
            actual,ex,branch=runner.forward_arm(R8,PARENT,model,functional,targets,shapes,x,arm,MERGE,torch,capture=True)
            self.assertEqual(ex['summed_fc1_rows'],rows);self.assertEqual(ex['summed_fc2_rows'],rows)
            if arm=='native':self.assertFalse(ex['layers']);self.assertFalse(branch)
            else:
                dual,dx,db=runner.forward_arm(R8,PARENT,model,functional,targets,shapes,x,arm,MERGE,torch,mode='dual_reference',capture=True)
                self.assertEqual(len(actual),len(dual));self.assertTrue(all(torch.equal(a,b) for a,b in zip(actual,dual)))
                self.assertEqual(branch.keys(),db.keys());self.assertTrue(all(torch.equal(branch[k],db[k]) for k in branch))
                self.assertEqual(dx['summed_fc1_rows'],20080)
        self.assertEqual(before,PARENT.source_digest(model))

    def test_failure_cleanup(self):
        model,targets,shapes,x,functional=FIXTURE.build();adapter=PARENT.wrapped_model
        with patch.object(model,'forward',side_effect=RuntimeError('synthetic failure')):
            with self.assertRaisesRegex(RuntimeError,'synthetic failure'):
                runner.forward_arm(R8,PARENT,model,functional,targets,shapes,x,'fine_s2b2',MERGE,torch)
        self.assertIs(PARENT.wrapped_model,adapter)
        self.assertTrue(all(model.get_submodule(k) is v for k,v in targets.items()))
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_identity_rule_replays_and_excludes(self):
        import csv
        prior=json.loads(select_members.OLD.read_text());canonical=json.loads(select_members.CANON.read_text())['records']
        with select_members.ROLES.open() as f:roles={r['target_id']:r['auxiliary_role'] for r in csv.DictReader(f)}
        selected=select_members.select(canonical,roles,prior)
        manifest=json.loads((HERE/'membership_v1.json').read_text())
        self.assertEqual([r['target_id'] for r in selected],manifest['target_ids'])
        self.assertFalse(set(manifest['target_ids']) & set(manifest['excluded_ids']))
        for seq in prior['check_sequences']:
            indices=[int(r['target_id'].rsplit('_',1)[1]) for r in selected if r['sequence']==seq]
            self.assertEqual(len(indices),8)
            self.assertTrue(all(abs(x-y)>=3 for i,x in enumerate(indices) for y in indices[i+1:]))


if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261014)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    sources=[Path(__file__),Path(runner.__file__),Path(select_members.__file__),runner.R8_PATH,FP]
    receipt={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'GPU_initialized':torch.cuda.is_initialized(),
             'torch':str(torch.__version__),'source_sha256':{str(p):runner.sha(p) for p in sources},
             'errors':[str(x) for x in result.errors],'failures':[str(x) for x in result.failures],
             'scope':'Toy four-arm service/dual, cleanup, identity-only member rule. No full network or quality.'}
    (HERE/'runner_cpu_results.json').write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1)
