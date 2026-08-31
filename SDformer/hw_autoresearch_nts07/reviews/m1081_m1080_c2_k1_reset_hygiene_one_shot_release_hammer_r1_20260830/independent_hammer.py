#!/usr/bin/env python3
"""Receipt-blind, non-EDA M1081 hammer for exact M1080."""
from __future__ import annotations

import ast, fcntl, hashlib, json, os, re, shutil, stat, subprocess, tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent; HW=HERE.parent.parent
PY=Path('/opt/anaconda3/envs/pytorch310/bin/python3.10')
RUNNER=Path('dc_handoff/scripts/run_m1080_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_one_shot_r1.py')
CONTRACT=Path('contracts/m1080_c2_k1_reset_hygiene_dc_mapped_vcs_source_contract_r1_20260830.json')
RELEASE=Path('contracts/m1080_c2_k1_reset_hygiene_dc_mapped_vcs_one_shot_release_r1_20260830.json')
REVIEW=Path('reviews/m1081_m1080_c2_k1_reset_hygiene_one_shot_release_hammer_r1_20260830')
ATTEMPT=Path('results/.m1080_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_attempt_consumed')
RESULT=Path('results/m1080_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_r1_20260830')
RSH='6ca208ed899337b8acf0433e8d28af7da1afe9855f779bebb24e0b5da1735836'
CSH='b77cbbd958ef737291dffa7861e1411377a96a5d51a5612461366bddcf3cbf67'
COUT='a568aaadc0021839c54977c1b9dccc055fd32c3aab733874418b02ad85900fea'
LSH='478097925d21093dfd66e5fac5d799daaac415f0229e7c8013ebfcc06b1028ee'
LOUT='831023d2f5e96cbd50a1f51e1167da3fa717b33cda2e6971b262450df8cf612e'
DOC='dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4'
ANCHORS=[259,737,3153,7569,14]
UP={
Path('reviews/m1059_m1058_c2_k1_reset_hygiene_source_release_hammer_r1_20260830'):'c22d41a87f82f939487637155b35d11496234850631b5894d159ff41e41fb4b3',
Path('results/m1058_c2_k1_reset_hygiene_rtl_vcs_r1_20260830'):'f22a55c33fadf74749060546e877fc10f892649aa31f3fa0da2d3fd164b70787',
Path('reviews/m1050_m1046_c2_mapped_gate_watchdog_failure_audit_r1_20260829'):'bc239844a71b5c017002ea1f6a756143d3c58b5ebf39d6a5499c76228da188bb',
Path('results/m1046_m1001_c2_three_axis_mapped_gate_saif_r5_20260829.failed_or_incomplete.2027456.quarantine'):'cb6f6b69e2cb51d60556f5bcb8a7748865f72ee2bdbe2f178925a624d9e9d705',
Path('reviews/m1069_m1068_c2_k1_reset_hygiene_one_shot_release_hammer_r1_20260830'):'ece91f960cf98892b12879bdf19d57f1d408cb971b4ba2a249b1a89a72853a9a',
Path('reviews/m1071_m1070_c2_k1_reset_hygiene_one_shot_release_hammer_r1_20260830'):'812a1543dc9c198ca504768cf7e4bfd5ef3941094438a1ebc8cc32cd709f3725'}
SIDES={
Path('contracts/m1058_c2_k1_reset_hygiene_source_only_contract_r1_20260830.json'):'1d06a6bdda5b15e404c758e5571498d026cb23e586fc7ba1d929f1c064518b44',
Path('contracts/m1058_c2_k1_reset_hygiene_dc_mapped_vcs_launch_candidate_r1_20260830.json'):'12c131029fc6f049e2f2a58082dcb6e4f72c4056a9bc68cde9006d585b2c7f82',
CONTRACT:COUT,RELEASE:LOUT}

def req(x,m):
 if not x: raise RuntimeError(m)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def js(p):
 def pairs(a):
  d={}
  for k,v in a:req(k not in d,'duplicate key '+k);d[k]=v
  return d
 return json.loads(Path(p).read_text(),object_pairs_hook=pairs)
def seal_dir(p):
 p=Path(p); ms=sorted(x for x in p.rglob('*') if x.is_file() and x.name not in {'SHA256SUMS','SHA256SUMS.seal.sha256'})
 man=p/'SHA256SUMS';man.write_text(''.join(f'{sha(x)}  {x.relative_to(p).as_posix()}\n' for x in ms))
 inner=p/'SHA256SUMS.seal.sha256';inner.write_text(f'{sha(man)}  SHA256SUMS\n');return sha(inner)
def verify_dir(p,o):
 p=Path(p);req(p.is_dir() and not p.is_symlink(),'dir absent/symlink')
 man=p/'SHA256SUMS'
 for line in man.read_text().splitlines():
  d,r=line.split(None,1);x=p/r.strip().lstrip('*');req(x.is_file() and sha(x)==d,'member mismatch '+str(x))
 req((p/'SHA256SUMS.seal.sha256').read_text().split()==[sha(man),'SHA256SUMS'],'inner mismatch')
 req(sha(p/'SHA256SUMS.seal.sha256')==o,'outer mismatch '+str(p))
def verify_side(p,o,root=HW):
 p=Path(p);s=Path(str(p)+'.sha256');z=Path(str(p)+'.sha256.seal.sha256');n=p.relative_to(root).as_posix()
 req(s.read_text().split()==[sha(p),n],'primary token mismatch')
 req(z.read_text().split()==[sha(s),n+'.sha256'],'outer token mismatch');req(sha(z)==o,'side outer mismatch')
def cpfile(a,b): Path(b).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)
def cptree(a,b): shutil.copytree(a,b)
def reseal_side(p,name=None,oname=None,extra=''):
 p=Path(p);root=p.parents[1];n=p.relative_to(root).as_posix();s=Path(str(p)+'.sha256');z=Path(str(p)+'.sha256.seal.sha256')
 s.write_text(f'{sha(p)}  {name or n}{extra}\n');z.write_text(f'{sha(s)}  {oname or n+".sha256"}\n');return sha(z)
def fake_review(root,release_outer,status='PASS_M1081_M1080_C2_K1_RESET_HYGIENE_ONE_SHOT_RELEASE_HAMMER',auth=True):
 p=root/REVIEW
 if p.exists():shutil.rmtree(p)
 p.mkdir(parents=True);(p/'review.json').write_text(json.dumps({'status':status,'identity':{'release_outer_seal_sha256':release_outer},'authorization':{'one_m1080_dc_then_mapped_vcs_attempt':auth}},sort_keys=True)+'\n')
 return seal_dir(p)
def mock_source():
 return r'''import fcntl,os,pathlib,subprocess
mode=os.environ.get("M1081_MODE","full"); rr=pathlib.Path.read_text; rf=fcntl.flock
def read(self,*a,**k):
 if str(self)=="/proc/meminfo":
  return "MemAvailable: 1 kB\nCommitLimit: 2 kB\nCommitted_AS: 2 kB\n" if mode=="resource" else "MemAvailable: 67108864 kB\nCommitLimit: 67108864 kB\nCommitted_AS: 1 kB\n"
 return rr(self,*a,**k)
pathlib.Path.read_text=read
def flock(fd,op):
 if mode=="flock":raise BlockingIOError("busy")
 return rf(fd,op)
fcntl.flock=flock
def done(a,rc=0):return subprocess.CompletedProcess(a,rc,stdout="",stderr="")
def run(a,*x,**k):
 w=[str(q) for q in a];e=w[0]
 if e=="/usr/bin/pgrep":return done(a,0 if mode=="eda" else 1)
 if e.endswith("/lmutil"):return done(a,0)
 if e.endswith("/dc_shell"):
  open(os.environ["M1081_EVENTS"],"a").write("DC\n")
  if mode=="dc_fail":return done(a,9)
  o=pathlib.Path(k["env"]["OUTPUT_DIR"]);(o/"reports").mkdir(parents=True);(o/"netlist").mkdir()
  (o/"TCL_PASS_TERMINAL.txt").write_text("PASS\n");(o/"reports/precompile_loop_gate.rpt").write_text("TIM-209=0\nOPT-150=0\nstatus=PASS_PRECOMPILE_LOOP_GATE\n")
  (o/"reports/area.rpt").write_text("Total cell area: 123.0\n");(o/"reports/timing_setup.rpt").write_text("slack (MET) 0.001\n")
  (o/f'netlist/{k["env"]["DESIGN_NAME"]}_mapped.v').write_text("module mock;endmodule\n");return done(a)
 if e.endswith("/vcs"):
  open(os.environ["M1081_EVENTS"],"a").write("VCS_COMPILE\n");s=pathlib.Path(w[w.index("-o")+1]);s.write_text("#!/bin/sh\nexit 0\n");s.chmod(0o755);return done(a)
 if pathlib.Path(e).name=="simv":
  c=int(next(q.split("=",1)[1] for q in w if q.startswith("+M979_CASE=")));A=[259,737,3153,7569,14];open(os.environ["M1081_EVENTS"],"a").write(f"CASE{c}\n")
  o=k["stdout"];o.write(f"PASS M979 mapped replay axis=K1 case={c} events=1 cycles={A[c]} saif_duration_ns={A[c]*3} numeric_mismatches=0 tuple_mismatches=0 weight_mismatches=0 accepted_unknowns=0 protocol_errors=0\n");o.flush();return done(a)
 raise RuntimeError(repr(w))
subprocess.run=run
'''
def prepare(root):
 files={RUNNER,CONTRACT,RELEASE,Path('docs/359_DATE终局冻结_20260813.md'),Path('dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl'),Path('dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc'),Path('dc_handoff/filelists/date_m1058_c2_k1_reset_hygiene_logic_only_dc.f'),Path('dc_handoff/tb/tb_m1058_c2_k1_reset_hygiene_mapped_gate_case.sv'),Path('tb_m349/m349_fc2_scalar_bank_memory_model.sv')}
 files|=set(SIDES)
 for p in list(SIDES):files|={Path(str(p)+'.sha256'),Path(str(p)+'.sha256.seal.sha256')}
 for line in (HW/'dc_handoff/filelists/date_m1058_c2_k1_reset_hygiene_logic_only_dc.f').read_text().splitlines():
  if line.strip() and not line.startswith('#'):files.add(Path(line.strip()))
 for p in files:cpfile(HW/p,root/p)
 for p in UP:cptree(HW/p,root/p)
 (root/'results').mkdir(exist_ok=True);(root/'reviews').mkdir(exist_ok=True);site=root/'mock';site.mkdir();(site/'sitecustomize.py').write_text(mock_source())
 return site,fake_review(root,LOUT)
def run(root,site,outer,mode='full',pin=RSH,lic=True):
 env={'PATH':'/usr/bin:/bin','PYTHONPATH':str(site),'M1081_MODE':mode,'M1081_EVENTS':str(root/'events'),'M1080_EXPECTED_RUNNER_SHA256':pin,'M1080_EXPECTED_M1081_OUTER_SHA256':outer}
 if lic:env['LM_LICENSE_FILE']='mock'
 return subprocess.run([str(PY),str(root/RUNNER)],env=env,text=True,capture_output=True,timeout=30)
def refresh(root,co=None):
 p=root/RELEASE;d=js(p)
 if co:d['identity']['contract_outer_seal_sha256']=co
 p.unlink();p.write_text(json.dumps(d,sort_keys=True,indent=2)+'\n');return fake_review(root,reseal_side(p))
def pre(name,mut=None,mode='full',pin=RSH,lic=True,pre_result=False):
 with tempfile.TemporaryDirectory(prefix='m1081_') as raw:
  root=Path(raw)/'hw_autoresearch_nts07';root.mkdir();site,o=prepare(root)
  if mut:o=mut(root,o)
  q=run(root,site,o,mode,pin,lic);req(q.returncode==3,name+' escaped');req(not(root/ATTEMPT).exists(),name+' consumed');req((root/RESULT).exists()==pre_result,name+' result boundary')
 return 'REJECTED_BEFORE_ATTEMPT'
def side_attack(kind):
 def f(root,o):
  p=root/CONTRACT;n=CONTRACT.as_posix()
  if kind=='base':co=reseal_side(p,name=p.name)
  elif kind=='suffix':co=reseal_side(p,name=n+'.x')
  elif kind=='trav':co=reseal_side(p,name='contracts/../'+p.name)
  elif kind=='extra':co=reseal_side(p,extra=' attacker')
  else:co=reseal_side(p,oname=p.name+'.sha256')
  return refresh(root,co)
 return f
def mutate_status(rel,key='status'):
 def f(root,o):
  p=root/rel;d=js(p);d[key]='ATTACK';p.unlink();p.write_text(json.dumps(d,sort_keys=True)+'\n');reseal_side(p);return o
 return f
def symlink_unit():
 text=(HW/RUNNER).read_text();tree=ast.parse(text);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='expect_exact_symlink_payload');src=ast.get_source_segment(text,fn);req(src is not None,'function absent')
 ns={'os':os,'stat':stat,'Path':Path,'sha':sha,'GateFailure':RuntimeError,'fail':lambda m:(_ for _ in ()).throw(RuntimeError(m))};exec(src,ns);f=ns['expect_exact_symlink_payload'];out={}
 with tempfile.TemporaryDirectory(prefix='m1081_symlink_') as raw:
  r=Path(raw);t=r/'snps_shell';t.write_text('payload');link=r/'dc_shell';link.symlink_to('snps_shell');digest=sha(t)
  f(link,'snps_shell',t,digest);out['exact']='ACCEPTED'
  attacks={'regular':lambda:(link.unlink(),link.write_text('payload')),
   'wrong_readlink':lambda:(link.unlink(),link.symlink_to('wrong')),
   'broken':lambda:(link.unlink(),link.symlink_to('missing')),
   'wrong_exact_target':lambda:None,
   'payload_sha':lambda:t.write_text('drift')}
  for k,m in attacks.items():
   if link.exists() or link.is_symlink():link.unlink()
   t.write_text('payload');link.symlink_to('snps_shell');m()
   try:f(link,'snps_shell',(r/'other') if k=='wrong_exact_target' else t,digest)
   except (RuntimeError,OSError):out[k]='REJECTED'
   else:raise RuntimeError('symlink attack escaped '+k)
 return out
def static():
 req(sha(HW/RUNNER)==RSH and sha(HW/CONTRACT)==CSH and sha(HW/RELEASE)==LSH,'source identity')
 req(sha(HW/Path(str(CONTRACT)+'.sha256.seal.sha256'))==COUT and sha(HW/Path(str(RELEASE)+'.sha256.seal.sha256'))==LOUT,'outer identity')
 req(sha(HW/'docs/359_DATE终局冻结_20260813.md')==DOC,'docs');[verify_side(HW/p,o) for p,o in SIDES.items()];[verify_dir(HW/p,o) for p,o in UP.items()]
 c=js(HW/CONTRACT);l=js(HW/RELEASE);req(c['status'].startswith('PASS_M1080_ADDITIVE_SOURCE_ONLY') and l['status']=='PASS_M1080_C2_K1_RESET_HYGIENE_DC_MAPPED_VCS_ONE_SHOT_RELEASE_SOURCE','status')
 req(c['pinned_evidence']['m1071_attempt_authorized'] is False,'M1071 stop auth');req(c['sole_additive_repair']['exact_readlink']=='snps_shell','repair')
 s=(HW/RUNNER).read_text();req('+vcs+initreg' not in s and 'power enable' not in s.lower(),'production pollution');req('expect_exact_symlink_payload(' in s and 'os.lstat(link)' in s and 'os.readlink(link)' in s,'symlink implementation')
 return {'runner':RSH,'contract_outer':COUT,'release_outer':LOUT,'symlink':symlink_unit()}
def dynamic():
 def relstat(root,o):
  p=root/RELEASE;d=js(p);d['status']='ATTACK';p.unlink();p.write_text(json.dumps(d,sort_keys=True)+'\n');return fake_review(root,reseal_side(p))
 def relseal(root,o):p=root/Path(str(RELEASE)+'.sha256');p.unlink();p.write_text('0'*64+'  '+RELEASE.as_posix()+'\n');return o
 def rvstat(root,o):return fake_review(root,LOUT,'ATTACK')
 def upstat(root,o):p=root/Path('reviews/m1071_m1070_c2_k1_reset_hygiene_one_shot_release_hammer_r1_20260830/review.json');d=js(p);d['status']='ATTACK';p.unlink();p.write_text(json.dumps(d)+'\n');return o
 def ns(root,o):(root/RESULT).mkdir(parents=True);return o
 a={'runner':pre('runner',pin='0'*64),'contract':pre('contract',mut=mutate_status(CONTRACT)),'release_status':pre('release_status',mut=relstat),'release_seal':pre('release_seal',mut=relseal),'review_status':pre('review_status',mut=rvstat),'upstream':pre('upstream',mut=upstat),'namespace':pre('namespace',mut=ns,pre_result=True),'flock':pre('flock',mode='flock'),'eda':pre('eda',mode='eda'),'resource':pre('resource',mode='resource'),'license':pre('license',lic=False)}
 for k in ('base','suffix','trav','extra','outer'):a['side_'+k]=pre('side_'+k,mut=side_attack(k))
 with tempfile.TemporaryDirectory(prefix='m1081_fail_') as raw:
  root=Path(raw)/'hw_autoresearch_nts07';root.mkdir();site,o=prepare(root);q=run(root,site,o,'dc_fail');req(q.returncode==3 and (root/ATTEMPT).is_dir(),'dc fail attempt');qs=list((root/'results').glob('m1080_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_r1_20260830.failed_or_incomplete.*.quarantine'));req(len(qs)==1 and js(qs[0]/'failure.json')['phase']=='FRESH_DC_ARCH_MODE0','quarantine');req((root/'events').read_text().splitlines()==['DC'],'compile after fail')
 with tempfile.TemporaryDirectory(prefix='m1081_full_') as raw:
  root=Path(raw)/'hw_autoresearch_nts07';root.mkdir();site,o=prepare(root);q=run(root,site,o);req(q.returncode==0,'full '+q.stderr);events=(root/'events').read_text().splitlines();req(events==['DC','VCS_COMPILE','CASE0','CASE1','CASE2','CASE3','CASE4'],'order');r=js(root/RESULT/'m1080_dc_mapped_vcs_receipt_r1.json');req(r['anchors']==ANCHORS and r['mapped_cases']==5 and not r['random_register_initialization_used'] and r['saif_files']==r['ptpx_runs']==0,'receipt')
 return a,events
def publish(st,a,e):
 d={'schema':'m1081_m1080_c2_k1_reset_hygiene_one_shot_release_hammer_r1','milestone':'M1081','date':'2026-08-30','status':'PASS_M1081_M1080_C2_K1_RESET_HYGIENE_ONE_SHOT_RELEASE_HAMMER','verdict':'GO_EXACTLY_ONE_M1080_FRESH_DC_THEN_FIVE_CASE_MAPPED_VCS_ATTEMPT','identity':{'runner_sha256':RSH,'contract_outer_seal_sha256':COUT,'release_outer_seal_sha256':LOUT,'docs359_sha256':DOC},'receipt_blind':True,'real_eda_launched':False,'real_m1080_attempt_consumed':False,'static_audit':st,'fault_injections':a,'redirected_mock_order':e,'authorization':{'one_m1080_dc_then_mapped_vcs_attempt':True},'caller':{'required_runner_sha256':RSH,'required_m1081_outer_seal_sha256':'PIN_THIS_DIRECTORY_SHA256SUMS_SEAL_SHA256'},'claim_boundary':{'saif_authorized':False,'ptpx_authorized':False,'power_admitted':False,'system_speedup_admitted':False,'paper_ppa_ready':False}}
 (HERE/'review.json').write_text(json.dumps(d,sort_keys=True,indent=2)+'\n');(HERE/'mechanical_checks.txt').write_text('PASS exact source/release/upstream identities\nPASS exact DC shell symlink plus five attacks\nPASS 11 inherited plus five sidecar pre-attempt attacks\nPASS DC failure quarantine\nPASS DC -> compile -> five cases mock\nPASS no real EDA/attempt\n');(HERE/'review.md').write_text('# M1081 independent release hammer\n\n**GO：仅授权一次 M1080 fresh DC → 五案例 mapped VCS。**\n\n独立核验精确 symlink/path/payload、sidecar、上游 STOP、16 类 pre-attempt 攻击、失败隔离和完整重定向顺序。未运行真实 EDA，未消费真实 attempt；不授权 SAIF/PTPX/功耗/系统倍速。\n');(HERE/'RUN_COMPLETE.txt').write_text('PASS_M1081_M1080_C2_K1_RESET_HYGIENE_ONE_SHOT_RELEASE_HAMMER\n');print('M1081_OUTER='+seal_dir(HERE))
def main():st=static();a,e=dynamic();publish(st,a,e)
if __name__=='__main__':main()
