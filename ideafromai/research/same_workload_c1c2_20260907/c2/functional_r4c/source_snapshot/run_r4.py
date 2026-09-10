import sys
assert sys.version_info[:2]==(3,12)
import subprocess,json,hashlib,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
HW=Path('/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
attempt=sys.argv[1] if len(sys.argv)==2 else 'functional_r4'
assert attempt.startswith('functional_r') and '/' not in attempt
OUT=ROOT/attempt;OUT.mkdir(exist_ok=False)
plan=json.loads((ROOT/'plan_r4.json').read_text())
prod=HW/'rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv'
adapter=HW/'rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv'
assert sha(prod)==plan['production']['top_sha256']
assert sha(adapter)==plan['production']['adapter_sha256']
for name,digest in plan['fixture_sha256'].items():assert sha(ROOT/name)==digest,name
inputs=[ROOT/x for x in ['plan_r4.json','c2_ftp_bank_parallel.sv','scoreboard_r4.cpp','run_r4.py','make_fixture.py','fixture.json','support.txt','weights.txt','golden.txt']]+[prod,adapter]
identities={str(p):sha(p) for p in inputs}
(OUT/'input_sha256.json').write_text(json.dumps(identities,indent=2)+'\n')
snapshot=OUT/'source_snapshot';snapshot.mkdir()
for p in inputs:
 shutil.copyfile(p,snapshot/p.name)
history=[];start=time.time()
def run(cmd,where,label,timeout=240):
 record={'label':label,'argv':[str(x) for x in cmd],'cwd':str(where)}
 history.append(record);(OUT/'commands.json').write_text(json.dumps(history,indent=2)+'\n')
 with (OUT/(label+'.log')).open('xb') as log:
  r=subprocess.run([str(x) for x in cmd],cwd=where,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 record['returncode']=r.returncode;(OUT/'commands.json').write_text(json.dumps(history,indent=2)+'\n')
 if r.returncode:raise RuntimeError(f'{label} failed, log retained')
 print(label,'PASS',flush=True)
try:
 for production in [1,0]:
  kind='production' if production else 'ftp';obj=OUT/(kind+'_obj');obj.mkdir()
  top=plan['production']['top'] if production else 'c2_ftp_bank_parallel'
  rtl=[adapter,prod] if production else [ROOT/'c2_ftp_bank_parallel.sv']
  cmd=['/usr/bin/verilator','--cc','--exe','--top-module',top,'--prefix','Vdut','--Mdir',obj,'-Wno-fatal',
       '-CFLAGS',f'-std=c++17 -O2 -DPRODUCTION={production}', '-o','c2_sim',*rtl,ROOT/'scoreboard_r4.cpp']
  if production:cmd.insert(1,'-GSCHEDULE_MODE=1')
  else:cmd[1:1]=['--unroll-count','1024','--unroll-stmts','100000']
  run(cmd,ROOT,kind+'_verilate',180)
  run(['/usr/bin/make','-C',obj,'-f','Vdut.mk','-j1'],ROOT,kind+'_compile',300)
  modes=[0] if production else [0,1]
  for mode in modes:
   axis='production' if production else 'ftp_shared' if mode else 'ftp_direct'
   run([obj/'c2_sim',ROOT,str(mode),OUT/(axis+'.json')],ROOT,axis+'_simulate',300)
 for p in inputs:assert sha(p)==identities[str(p)],str(p)+' changed during campaign'
 outputs={name:json.loads((OUT/(name+'.json')).read_text()) for name in ['production','ftp_direct','ftp_shared']}
 summary={'status':'PASS_BOUNDED_RTL_BEHAVIORAL_IO_NO_PPA','elapsed_seconds':time.time()-start,'axes':outputs,
  'input_sha256':identities,'plan':plan,'claim_boundary':plan['claim_boundary']}
 (OUT/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps({k:v['end_to_end_cycle_sum'] for k,v in outputs.items()}),flush=True)
except Exception as e:
 (OUT/'FAILED.json').write_text(json.dumps({'status':'FAILED_DO_NOT_CITE','error':repr(e),'elapsed_seconds':time.time()-start,'inputs':identities},indent=2)+'\n')
 raise
