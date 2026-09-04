#!/usr/bin/env python3
"""Receipt-blind M1075 source hammer; never opens canonical rows or runs full."""
from __future__ import annotations

import ast, contextlib, copy, hashlib, importlib.util, inspect, io, json, os
from pathlib import Path
import subprocess, sys, tempfile, unittest

HERE=Path(__file__).resolve().parent; HW=HERE.parent.parent
ENGINE=HW/'system_simulator/scripts/execute_m1074_m1072_c1_full_exact_1rw_one_shot.py'
RUNNER=HW/'system_simulator/scripts/run_m1074_m1072_c1_full_exact_1rw_one_shot.sh'
CHECKER=HW/'system_simulator/scripts/check_m1074_c1_full_exact_1rw_one_shot_source.py'
TESTS=HW/'system_simulator/tests/test_m1074_c1_full_exact_1rw_one_shot_source.py'
CONTRACT=HW/'contracts/m1074_m1073_m1072_c1_full_exact_1rw_one_shot_source_contract_r1_20260830.json'
M1072=HW/'system_simulator/scripts/run_m1072_c1_row_provenance_exact_1rw_source.py'
M1073=HW/'reviews/m1073_m1072_c1_row_provenance_exact_1rw_source_hammer_r1_20260830'
M1074=HW/'reviews/m1074_m1073_c1_full_exact_1rw_one_shot_source_receipt_r1_20260830'
ROWS=HW/'results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/m410r2_h67_q32_runtime_rows_32.memh'
DOCS=HW/'docs/359_DATE终局冻结_20260813.md'
EXPECTED={'engine':'90ead8cb4a0196114dbb6c51f4fe9e042fee1bf2816855687327221c8c3274e5','runner':'cec9da5f0faaef281c705f46b41020fe6572be0f98317f6f8ab29f5e1a090812','checker':'3dfe09b8e3c71055a47a03ef9d6cb34c35503d5f734f7c2c90f358dadef2c880','tests':'6ad691f33962500bd1fd35aaf71040359dae95100384544fc63f7d726d526f4b','contract':'5d385afe4c0b5875568b19f903d1ed56a224d79790c206a62a28fdeefb967a67','contract_outer':'b2892273abf602787f8d857d97ef9d9a5c9282fa380ba8787fbd9e55c15214aa','m1072':'879712a59785acc79776990236884582431adea81103a222d5415905199a1e4c','m1073_outer':'0a0457481fda030275205cb8c3b59938b66d86e1ce2cac63b0e2572b2de75e70','m1074_outer':'9845b8d6845c09fda4833e2e73f1024e54c87babbe411460e1ac55e4ed4f92d5','docs':'dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4'}

def req(v,m):
 if not v: raise RuntimeError(m)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def strict_json(p):
 def pairs(xs):
  d={}
  for k,v in xs:req(k not in d,'duplicate key '+k);d[k]=v
  return d
 return json.loads(Path(p).read_text(),object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in()).throw(RuntimeError('nonfinite '+x)))
def verify_dir(p,outer):
 p=Path(p);req(p.is_dir() and not p.is_symlink(),'sealed dir absent/symlink')
 man=p/'SHA256SUMS';seen=set()
 for line in man.read_text().splitlines():
  d,r=line.split(None,1);r=r.strip().lstrip('*');req(r not in seen,'duplicate member');seen.add(r);x=p/r;req(x.is_file() and not x.is_symlink() and sha(x)==d,'sealed member '+r)
 inner=p/'SHA256SUMS.seal.sha256';req(inner.read_text().split()==[sha(man),'SHA256SUMS'],'inner seal');req(sha(inner)==outer,'outer seal')
def seal_dir(p):
 p=Path(p);xs=sorted(x for x in p.rglob('*') if x.is_file() and x.name not in {'SHA256SUMS','SHA256SUMS.seal.sha256'});man=p/'SHA256SUMS';man.write_text(''.join(f'{sha(x)}  {x.relative_to(p).as_posix()}\n' for x in xs));inner=p/'SHA256SUMS.seal.sha256';inner.write_text(f'{sha(man)}  SHA256SUMS\n');return sha(inner)

class RowGuard:
 def __init__(self):self.attempts=[]
 def __enter__(self):
  self.po=Path.open;self.oo=os.open;self.ol=os.lstat;self.pr=os.pread;target=ROWS.absolute()
  def same(p):
   try:return Path(os.fspath(p)).absolute()==target
   except (TypeError,ValueError):return False
  def popen(p,*a,**k):
   if same(p):self.attempts.append('Path.open');raise RuntimeError('canonical rows access forbidden')
   return self.po(p,*a,**k)
  def oopen(p,*a,**k):
   if same(p):self.attempts.append('os.open');raise RuntimeError('canonical rows access forbidden')
   return self.oo(p,*a,**k)
  def lstat(p,*a,**k):
   if same(p):self.attempts.append('os.lstat');raise RuntimeError('canonical rows access forbidden')
   return self.ol(p,*a,**k)
  def pread(*a,**k):self.attempts.append('os.pread');raise RuntimeError('pread forbidden in source hammer')
  Path.open=popen;os.open=oopen;os.lstat=lstat;os.pread=pread;return self
 def __exit__(self,*x):Path.open=self.po;os.open=self.oo;os.lstat=self.ol;os.pread=self.pr

def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);req(s and s.loader,'load '+name);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def static_audit(M):
 for p,k in ((ENGINE,'engine'),(RUNNER,'runner'),(CHECKER,'checker'),(TESTS,'tests'),(CONTRACT,'contract'),(M1072,'m1072'),(DOCS,'docs')):req(p.is_file() and not p.is_symlink() and sha(p)==EXPECTED[k],k+' identity')
 side=Path(str(CONTRACT)+'.sha256');outer=Path(str(CONTRACT)+'.sha256.seal.sha256');req(side.read_text().split()==[EXPECTED['contract'],CONTRACT.name],'contract sidecar');req(outer.read_text().split()==[sha(side),side.name] and sha(outer)==EXPECTED['contract_outer'],'contract outer')
 verify_dir(M1073,EXPECTED['m1073_outer']);verify_dir(M1074,EXPECTED['m1074_outer'])
 c=strict_json(CONTRACT);req(c['status']=='PASS_M1074_ONE_SHOT_SOURCE_CONTRACT__M1075_REQUIRED_NO_LAUNCH' and c['launch_now'] is False and c['max_attempts_now']==0,'contract boundary')
 ar=strict_json(M1074/'review.json');req(ar['status']=='PASS_M1074_SOURCE_ONLY__M1075_REQUIRED_NO_LAUNCH' and ar['identity']['m1074_engine_sha256']==EXPECTED['engine'] and ar['identity']['m1074_runner_sha256']==EXPECTED['runner'],'author receipt identity')
 rr=strict_json(M1073/'review.json');req(rr['status']=='PASS_M1073_M1072_C1_ROW_PROVENANCE_EXACT_1RW_SOURCE_HAMMER','M1073 status')
 rt=RUNNER.read_text();et=ENGINE.read_text();req('403922' not in rt+et,'forbidden +403922');req('[[ "$#" -eq 0 ]]' in rt,'runner args not zero')
 envs=set(__import__('re').findall(r'M1074_EXPECTED_[A-Z0-9_]+',rt));req(envs=={'M1074_EXPECTED_RUNNER_SHA256','M1074_EXPECTED_M1075_REVIEW_SHA256','M1074_EXPECTED_M1075_MANIFEST_SHA256','M1074_EXPECTED_M1075_OUTER_SHA256'},'caller env injection surface')
 forbidden=('--cycles','--preprocess','--capacity','--coverage','--rows','--records','--sample','--work-cycles')
 req(all(x not in rt+et for x in forbidden),'caller metric/work injection')
 consume_pos=rt.index('--consume-attempt')
 order=[rt.index('--validate-source'),rt.index('--validate-authority'),rt.index('m1074_process_gate\n'),rt.index('m1074_resource_gate\n'),consume_pos,rt.index('[[ -d "${m1074_attempt}" ]]',consume_pos),rt.index('--execute-full'),rt.index('--publish')];req(order==sorted(order),'runner attempt/rows order')
 fn=inspect.getsource(M.execute_full);tree=ast.parse(fn);calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='iter_canonical_full_replay_results'];req(len(calls)==1 and not calls[0].args and not calls[0].keywords,'production iterator not unique zero-arg')
 req(fn.index('validate_attempt(authority)')<fn.index('iter_canonical_full_replay_results()'),'engine attempt/iterator order')
 req(inspect.isgeneratorfunction(M.M1072.iter_canonical_full_replay_results) and len(inspect.signature(M.M1072.iter_canonical_full_replay_results).parameters)==0,'M1072 production iterator signature')
 req(M.M1072.TASKS==812160 and M.M1072.M1064.RAW_ROWS==51840000 and M.M1072.SAMPLES==10 and list(M.M1072.DESIGNS)==['candidate','strongest_zero','same_coordinate_bit'],'population/design boundary')
 req(M.M1072.M1064.derive_physical_capacity()['derived_total_bytes']==214912,'capacity')
 return {'exact_identities':9,'runner_order':order,'caller_environment':sorted(envs),'production_iterator_calls':1,'production_iterator_arguments':0,'tasks':812160,'rows':51840000,'samples':10,'designs':list(M.M1072.DESIGNS),'capacity_bytes':214912}

def run_directed_tests():
 T=load(TESTS,'m1075_imported_m1074_tests');suite=unittest.defaultTestLoader.loadTestsFromModule(T);stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=1).run(suite);req(r.wasSuccessful() and r.testsRun==15,'frozen 15 tests failed '+stream.getvalue());return {'tests':15,'failures':0,'errors':0}

def dynamic_audit(M):
 out={};authority={'m1075_outer_seal_file_sha256':'b'*64}
 with tempfile.TemporaryDirectory(prefix='m1075_attempt_') as raw:
  p=Path(raw);first=M.consume_attempt(authority,p);req(first['receipt']['canonical_rows_opened_or_hashed_before_attempt'] is False,'attempt rows boundary')
  try:M.consume_attempt(authority,p)
  except RuntimeError:out['attempt_unique']='REJECTED_SECOND_CONSUME'
  else:raise RuntimeError('second attempt accepted')
  req(M.verify_atomic_seal(p/M.ATTEMPT.name)==first['seal'],'attempt seal')
 raw=M.synthetic_raw_result();norm=M.normalize_full_result(raw);req(len(norm['sample_boundaries'])==10 and set(norm['aggregate'])==set(M.M1072.DESIGNS) and norm['capacity']['derived_total_bytes']==214912,'normalization')
 out['ten_sample_three_design_214912B']='PASS'
 attacks=[]
 for kind in ('partial','duplicate','reordered','extra'):
  x=copy.deepcopy(raw)
  if kind=='partial':x['samples'].pop()
  elif kind=='duplicate':x['samples'][1]=x['samples'][0]
  elif kind=='reordered':x['samples'][0],x['samples'][1]=x['samples'][1],x['samples'][0]
  else:x['samples'].append(copy.deepcopy(x['samples'][-1]))
  try:M.normalize_full_result(x)
  except RuntimeError:attacks.append(kind)
 req(len(attacks)==4,'sample attacks');out['sample_attacks']=attacks
 for field in ('cycles_after_commit','delayed_accesses','nominal_excess_accesses'):
  x=copy.deepcopy(raw);x['samples'][0]['designs']['candidate'][field]=True
  try:M.normalize_full_result(x)
  except RuntimeError:pass
  else:raise RuntimeError('boolean metric accepted '+field)
 out['caller_metric_boolean_attacks']='REJECTED'
 x=copy.deepcopy(raw);x['coverage']['execution_provenance_digest_sha256']='bad'
 try:M.normalize_full_result(x)
 except RuntimeError:out['provenance_forgery']='REJECTED'
 else:raise RuntimeError('provenance forgery')
 x=copy.deepcopy(raw);x['coverage']['parent']['candidate']['reads']+=1
 try:M.normalize_full_result(x)
 except RuntimeError:out['parent_forgery']='REJECTED'
 else:raise RuntimeError('parent forgery')
 x=copy.deepcopy(raw);x['capacity']['derived_total_bytes']=1
 try:M.normalize_full_result(x)
 except RuntimeError:out['capacity_forgery']='REJECTED'
 else:raise RuntimeError('capacity forgery')
 cascade=M.M1072.M1064.small_oracle();req(cascade['m1056_cascade_kernel_preserved'] is True and cascade['capacity']['derived_total_bytes']==214912,'cascade oracle');out['one_1rw_cascade']='PASS_20_TO_22_FROZEN_ORACLE'
 with tempfile.TemporaryDirectory(prefix='m1075_partial_') as rawdir:
  p=Path(rawdir);w=p/(M.WORK_PREFIX+'partial');w.mkdir();(w/'payload').write_text('x')
  for fault in ('after_manifest','before_rename'):
   try:M.atomic_seal(w,inject_fault=fault)
   except RuntimeError:pass
   else:raise RuntimeError('seal fault escaped '+fault)
  q=p/(M.FAILURE_PREFIX+'partial');v=M.quarantine_work(w,q,130,'PARTIAL_SEAL',p);req(v['status']=='PASS_M1074_SEALED_FAILURE_QUARANTINE' and q.is_dir(),'partial quarantine');M.verify_atomic_seal(q);out['partial_seal_quarantine']='PASS'
 with tempfile.TemporaryDirectory(prefix='m1075_rename_') as rawdir:
  p=Path(rawdir);w=p/(M.WORK_PREFIX+'rename');w.mkdir();(w/'payload').write_text('x');M.atomic_seal(w);dest=p/'collision';dest.mkdir()
  try:M.rename_noreplace(w,dest)
  except RuntimeError:pass
  else:raise RuntimeError('rename collision escaped')
  q=p/(M.FAILURE_PREFIX+'rename');M.quarantine_work(w,q,17,'RENAMEAT2_COLLISION',p);M.verify_atomic_seal(q);out['renameat2_failure_quarantine']='PASS'
 before=(M.ATTEMPT.exists(),M.RESULT.exists());q=subprocess.run([str(RUNNER),'--cycles','1'],text=True,capture_output=True);req(q.returncode==2 and (M.ATTEMPT.exists(),M.RESULT.exists())==before,'runner arg injection');out['runner_argument_injection']='REJECTED_BEFORE_ATTEMPT'
 req(not M.ATTEMPT.exists() and not M.RESULT.exists() and not any((HW/'results').glob(M.WORK_PREFIX+'*')) and not any((HW/'results').glob(M.FAILURE_PREFIX+'*')),'canonical namespace pollution')
 out['canonical_namespaces']='ABSENT'
 return out

def publish(st,tests,dyn,guard):
 review={'schema':'m1075_m1074_c1_full_exact_1rw_one_shot_source_hammer_review_v1','status':'PASS_M1075_M1074_C1_FULL_EXACT_1RW_ONE_SHOT_SOURCE_HAMMER','verdict':'GO_ONE_M1074_CPU_FULL_REPLAY_ONLY','score':100,'p0_count':0,'p1_count':0,'receipt_blind':True,'identity':{'m1074_engine_sha256':EXPECTED['engine'],'m1074_runner_sha256':EXPECTED['runner'],'m1074_checker_sha256':EXPECTED['checker'],'m1074_tests_sha256':EXPECTED['tests'],'m1074_contract_sha256':EXPECTED['contract'],'m1074_contract_outer_seal_file_sha256':EXPECTED['contract_outer'],'m1074_author_receipt_outer_seal_file_sha256':EXPECTED['m1074_outer'],'m1072_source_sha256':EXPECTED['m1072'],'m1073_outer_seal_file_sha256':EXPECTED['m1073_outer'],'docs359_sha256':EXPECTED['docs']},'static_audit':st,'directed_tests':tests,'dynamic_attacks':dyn,'canonical_rows_guard':{'lstat_open_pread_attempts':guard.attempts,'canonical_rows_opened_or_hashed':False},'claim_boundary':{'launch_now':True,'max_attempts':1,'automatic_retry':False,'cpu_only':True,'eda_gpu_remote':False,'full_replay_executed_by_m1075':False,'raw_result_created':False,'capacity_only_214912B_admitted':False,'matched_cycles_admitted':False,'speedup_admitted':False,'rtl_cycles':False,'paper_ppa_ready':False},'caller':{'required_runner_sha256':EXPECTED['runner'],'required_m1075_review_sha256':'PIN_REVIEW_JSON_SHA256','required_m1075_manifest_sha256':'PIN_SHA256SUMS_SHA256','required_m1075_outer_sha256':'PIN_SHA256SUMS_SEAL_SHA256'}}
 (HERE/'review.json').write_text(json.dumps(review,sort_keys=True,indent=2)+'\n');(HERE/'mechanical_checks.json').write_text(json.dumps({'status':'PASS','static':st,'tests':tests,'dynamic':dyn,'rows_guard_attempts':guard.attempts},sort_keys=True,indent=2)+'\n');(HERE/'review.md').write_text('# M1075 independent source hammer\n\n**GO：仅授权一次 M1074 CPU full replay。**\n\n审计绑定 M1074 engine/runner/checker/tests/contract、M1072、M1073 和作者 receipt。15 个源测试、attempt 原子唯一性、零参数 iterator、10-sample/三设计/1RW cascade/214912B/provenance、partial seal 与 renameat2 失败隔离均通过。canonical rows 访问硬拦截计数为 0；未执行 full、EDA、GPU、remote。结果仍需独立 result hammer。\n');(HERE/'RUN_COMPLETE.txt').write_text('PASS_M1075_M1074_C1_FULL_EXACT_1RW_ONE_SHOT_SOURCE_HAMMER\n');outer=seal_dir(HERE);print('M1075_REVIEW_SHA='+sha(HERE/'review.json'));print('M1075_MANIFEST_SHA='+sha(HERE/'SHA256SUMS'));print('M1075_OUTER_SHA='+outer)

def main():
 with RowGuard() as guard:
  M=load(ENGINE,'m1075_engine');st=static_audit(M);tests=run_directed_tests();dyn=dynamic_audit(M)
 req(guard.attempts==[],'canonical rows access attempted')
 publish(st,tests,dyn,guard)
if __name__=='__main__':main()
