"""Local temporary-file orchestration checks; no network, model, or GPU."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PIPELINE = Path(__file__).with_name('run_native_cpu_pipeline.py')


class PipelineTests(unittest.TestCase):
    def build(self, root, fail=False):
        setup=root/'setup'; rec=setup/'raw_recovery_20261012'; pilot=rec/'pilot01_a'
        pilot.mkdir(parents=True)
        (root/'envs/v2/bin').mkdir(parents=True)
        (root/'envs/v2/bin/python').symlink_to(sys.executable)
        data=root/'data/DSEC_native_v1'; data.mkdir(parents=True)
        arrays=[]
        for i in range(2):
            name='sample'+str(i);(data/name).write_bytes(bytes([i]))
            arrays.append(dict(relative_path=name,sha256=hashlib.sha256(bytes([i])).hexdigest()))
        gen=setup/'generate_native_inputs.py'
        gen.write_text('import os,sys,json\nfrom pathlib import Path\nassert os.environ["CUDA_VISIBLE_DEVICES"]==""\na=sys.argv\n'
                       'i=int(a[a.index("--shard-index")+1])\n'
                       + ('if i==1:sys.exit(7)\n' if fail else '')
                       + 'Path(a[a.index("--report")+1]).write_text(json.dumps({"status":"SHARD_COMPLETE","i":i}))\n')
        val=setup/'verify_native_dataset.py'
        val.write_text('import sys,os,json,hashlib\nfrom pathlib import Path\nassert os.environ["CUDA_VISIBLE_DEVICES"]==""\na=sys.argv\n'
                       'r=[Path(a[i+1]) for i,x in enumerate(a) if x=="--shard-report"]\nassert len(r)==4 and all(p.exists() for p in r)\n'
                       'o=Path(a[a.index("--output-dir")+1]);o.mkdir();(o/"mock_pass.json").write_text("{}")\n'
                       '(o/"acceptance.json").write_text(json.dumps(dict(status="VERIFIED_NATIVE_DATASET",passed=True,root=a[a.index("--root")+1],validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),counts=dict(generated_inputs=8152,files=24514),training_started=False)))\n')
        gsha=hashlib.sha256(gen.read_bytes()).hexdigest();vsha=hashlib.sha256(val.read_bytes()).hexdigest()
        (pilot/'generation.json').write_text(json.dumps(dict(status='BOUNDED_SUBSET_COMPLETE',script_sha256=gsha,records=arrays)))
        (rec/'recovery_status.json').write_text(json.dumps(dict(status='raw_and_GT_verified_ready_for_separate_CPU_generation')))
        cross=setup/'cross.json';cross.write_text(json.dumps(dict(status='PASS',comparison='exact_npy_sha256',targets=['a','b'])))
        (setup/'common_targets.json').write_text('{}')
        return [sys.executable,str(PIPELINE),'--root',str(root),'--generator-sha256',gsha,
                '--validator-sha256',vsha,'--cross-version-evidence',str(cross),'--run']

    def test_success_and_failed_shard_do_not_train(self):
        for fail in (False,True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);args=self.build(root,fail)
                result=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=50)
                state=json.loads((root/'setup/native_cpu_pipeline_20261012/status.json').read_text())
                self.assertEqual(result.returncode==0,not fail)
                self.assertFalse(state['GPU_training_started'])
                self.assertEqual(state['status'],'FAILED_NO_TRAINING' if fail else 'CPU_PREPARATION_COMPLETE_NO_TRAINING')
                self.assertEqual((root/'setup/native_cpu_pipeline_20261012/acceptance/mock_pass.json').exists(),not fail)

    def test_changed_pilot_rejected_before_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args=self.build(root)
            (root/'data/DSEC_native_v1/sample0').write_bytes(b'bad')
            result=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=5)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((root/'setup/native_cpu_pipeline_20261012').exists())


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PipelineTests))
    Path(__file__).with_name('cpu_pipeline_checks.json').write_text(json.dumps(dict(
        passed=result.wasSuccessful(),tests_run=result.testsRun,scope='Temporary stub subprocesses only; no GPU/network/data generation',
        pipeline_sha256=hashlib.sha256(PIPELINE.read_bytes()).hexdigest()),indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
