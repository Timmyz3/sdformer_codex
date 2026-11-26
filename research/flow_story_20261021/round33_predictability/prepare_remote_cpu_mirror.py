"""Run on dedicated server CPU; preserve exact repository-relative audit paths."""
from pathlib import Path
import tarfile,json,hashlib,os,subprocess
r=json.loads(Path('/data/flow_a/round33_mirror_inputs_v1.json').read_text())
base=Path(r['root']);base.mkdir(exist_ok=False)
with tarfile.open('/data/flow_a/round33_mirror_inputs_v1.tar.gz') as t:t.extractall(base)
for f,h in r['files'].items():assert hashlib.sha256((base/f).read_bytes()).hexdigest()==h
links={'research/flow_story_20261021/actual/e0_recovery_v1':'/data/flow_a/runs/round32_20261021/e0_recovery_v1',
       'research/flow_story_20261021/actual/e3_rne_probe_v1':'/data/flow_a/runs/round32_20261021/e3_rne_probe_v1',
       'research/flow_story_20261021/round33_predictability/actual/fit_tune_v1':'/data/flow_a/runs/round33_20261021/fit_tune_v1'}
for dest,source in links.items():
    p=base/dest;p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(source,target_is_directory=True)
b=base/'research/flow_story_20261021/round33_predictability'
a=['/data/flow_a/envs/v2/bin/python','-B',str(b/'independent_review/audit_collector_actual_v1.py'),'--run',str(b/'actual/fit_tune_v1'),'--source','/data/flow_a/data/DSEC_native_v1','--output',str(b/'independent_review/FIT_TUNE_REMOTE_EXECUTION_AUDIT_V1.json')]
e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
completed=subprocess.run(a,env=e,check=False,capture_output=True,text=True)
r.update(symlinks=links,audit_args=a,audit_returncode=completed.returncode,audit_stdout=completed.stdout,audit_stderr=completed.stderr,
         setup_failure='Prior inline helper had a string-substitution SyntaxError before any remote code executed; corrected standalone helper, no GPU or data-result change.')
(b/'REMOTE_CPU_EXECUTION_RECEIPT_V1.json').write_text(json.dumps(r,indent=2)+'\n')
print(completed.stdout);print(completed.stderr);assert completed.returncode==0
