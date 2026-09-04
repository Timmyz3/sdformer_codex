#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1096r2 final launcher hammer; dry/static only, never production/EDA."""
from __future__ import annotations
import ast, hashlib, json, os, stat, sys, tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent;HW=HERE.parents[1]
L=HW/'dc_handoff/scripts/run_m1091r3_m1090r3_c2_observation_authorized_launch_r1.py'
E=HW/'dc_handoff/scripts/m1091r3_m1090r3_c2_observation_authorized_engine_r1.py'
LR=HW/'contracts/m1091r3_m1090r3_c2_observation_authorized_launch_receipt_r1_20260830.json'
C=HW/'contracts/m1094c2_m1091r3_c2_zero_arg_launch_source_contract_r1_20260830.json'
AR=HW/'reviews/m1094c2_m1091r3_c2_zero_arg_launch_source_receipt_r1_20260830'
SR=HW/'reviews/m1090r3_m1091r3_c2_observation_fixed_history_source_receipt_r1_20260830'
H=HW/'reviews/m1093r2_m1090r3_m1091r3_c2_observation_engine_hammer_r1_20260830'
A=HW/'results/.m1091r3_m1090r3_c2_observation_dc_mapped_vcs_attempt_consumed'
R=HW/'results/m1091r3_m1090r3_c2_observation_dc_mapped_vcs_r1_20260830'
DOC=HW/'docs/359_DATE终局冻结_20260813.md'
X={'launcher':'64eb690f557c8aa61461034f714a8eefe7e7176aa85c700e3f3290f2b902f56a','launch_outer':'402f3ac5b99c387a91308641c07c52351fc034541a34f18ec05e68baff6f831b','contract_outer':'d8b41e94a8948cf98525dfdf551e84ebfd569a77e14e09904764c4b06303c33e','author_outer':'6eaa54d491e48461540c077e2fe20ed075c019dbc252b88aa03273379e0192df','engine':'41b7899083152f8099acac759109a8eb22c381cb6a17506ae85e6666656daf04','source_outer':'8bc6f725ef0ec7055441afafa2c0bd5c5ba54620c4354feaf2a6763fbabedd9e','hammer_outer':'d6fa5ecb89342188586fb179d9dcaa1018078b4f3db6c609f6f1fd1b0559f9cc','docs':'dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4'}
checks=[]
class F(RuntimeError):pass
def req(v,m):
 if not v:raise F(m)
def mark(n,v=True):req(v,n);checks.append(n)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def reg(p,d):p=Path(p);return stat.S_ISREG(p.lstat().st_mode) and not p.is_symlink() and sha(p)==d
def flat(p,o,status=None):
 p=Path(p);m=p/'SHA256SUMS';s=p/'SHA256SUMS.seal.sha256';req(reg(s,o) and s.read_text().split()==[sha(m),'SHA256SUMS'],'flat')
 for line in m.read_text().splitlines():d,n=line.split(None,1);req(reg(p/n.lstrip('*'),d),'member')
 if status:req(json.loads((p/'review.json').read_text())['status']==status,'status')
def double(p,o):
 s=Path(str(p)+'.sha256');x=Path(str(p)+'.sha256.seal.sha256');d=sha(p)
 return x.is_file() and not x.is_symlink() and sha(x)==o and s.read_text().split()==[d,p.relative_to(HW).as_posix()] and x.read_text().split()==[sha(s),s.relative_to(HW).as_posix()]
def module_defs(path,name):
 t=ast.parse(path.read_text());allow=(ast.Import,ast.ImportFrom,ast.Assign,ast.AnnAssign,ast.ClassDef,ast.FunctionDef)
 ns={'__file__':str(path),'__name__':name};exec(compile(ast.Module([n for n in t.body if isinstance(n,allow)],[]),str(path),'exec'),ns);return ns
def rejects(fn,*a):
 try:fn(*a)
 except Exception:return True
 return False
def main():
 mark('launcher_sha',sha(L)==X['launcher']);mark('engine_sha',sha(E)==X['engine']);mark('launch_receipt_double',double(LR,X['launch_outer']));mark('source_contract_double',double(C,X['contract_outer']))
 flat(AR,X['author_outer'],'PASS_M1094C2_M1091R3_ZERO_ARG_LAUNCH_SOURCE__M1096R2_REQUIRED__NO_EDA');mark('author_receipt')
 flat(SR,X['source_outer'],'PASS_M1090R3_M1091R3_FIXED_HISTORY_SOURCE_ONLY__M1093R2_REQUIRED__NO_EDA');mark('engine_source_receipt')
 flat(H,X['hammer_outer'],'PASS_M1093R2_M1090R3_M1091R3_ENGINE_HAMMER__AUTHOR_LAUNCH_WRAPPER_ONLY__NO_EDA');mark('m1093r2')
 mark('docs359',sha(DOC)==X['docs']);mark('namespaces_absent',not A.exists() and not R.exists())
 receipt=json.loads(LR.read_text());contract=json.loads(C.read_text());author=json.loads((AR/'review.json').read_text())
 for key,value in [('launcher_sha256',X['launcher']),('engine_sha256',X['engine']),('engine_source_receipt_outer_seal_file_sha256',X['source_outer']),('m1093r2_outer_seal_file_sha256',X['hammer_outer'])]:mark('receipt_'+key,receipt[key]==value)
 mark('receipt_zero_arg',receipt['arguments']==0 and receipt['caller_selected_authority_allowed'] is False);mark('receipt_no_launch',all(receipt[k] is False for k in ('launch_now','attempt_now','dc_now','mapped_vcs_now','automatic_retry')))
 mark('contract_launcher',contract['launcher']['sha256']==X['launcher'] and contract['launcher']['arguments']==0);mark('contract_pins',contract['hardcoded_authority']['engine_sha256']==X['engine'] and contract['hardcoded_authority']['m1093r2_outer_seal_file_sha256']==X['hammer_outer'])
 mark('author_receipt_pins',author['identity']['launcher_sha256']==X['launcher'] and author['identity']['launch_receipt_outer_seal_file_sha256']==X['launch_outer'])
 text=L.read_text();mark('literal_engine_pin',f'ENGINE_SHA256 = "{X["engine"]}"' in text);mark('literal_hammer_pin',f'M1093R2_OUTER_SHA256 = "{X["hammer_outer"]}"' in text)
 mark('no_expected_env','EXPECTED_' not in text and 'os.environ' not in text and 'getenv' not in text);mark('zero_arg_source','len(sys.argv) == 1' in text)
 mark('exact_child_argv','[str(PYTHON), "-I", str(ENGINE), "--authorized-launch"]' in text);mark('close_fds','close_fds=True' in text);mark('constant_env_return','return {' in text and 'env=clean_child_environment()' in text)
 lns=module_defs(L,'m1096r2_launcher_model');oldargv=sys.argv[:]
 try:
  sys.argv=[str(L)];lns['validate_source_only_authority']();env0=lns['clean_child_environment']()
  mark('valid_source_authority_dry');mark('env_exact_keys',set(env0)=={'LANG','LC_ALL','PATH','TMPDIR','SNPSLMD_LICENSE_FILE','LM_LICENSE_FILE'})
  mark('env_exact_values',env0=={'LANG':'C.UTF-8','LC_ALL':'C.UTF-8','PATH':'/usr/bin:/bin','TMPDIR':'/tmp','SNPSLMD_LICENSE_FILE':'27030@ic.ismd-nemo','LM_LICENSE_FILE':'/opt/synopsys/Synopsys.dat'})
  poison={'PYTHONPATH':'/tmp/evil','LD_PRELOAD':'/tmp/evil.so','PATH':'/tmp/evil','M1091_EXPECTED_ENGINE_SHA256':'0'*64,'SNPSLMD_LICENSE_FILE':'evil','HOME':'/tmp/evil'};backup={k:os.environ.get(k) for k in poison};os.environ.update(poison)
  mark('caller_env_no_effect',lns['clean_child_environment']()==env0)
  for k,v in backup.items():
   if v is None:os.environ.pop(k,None)
   else:os.environ[k]=v
  calls=[]
  class Done:returncode=37
  def fake_run(*a,**kw):calls.append((a,kw));return Done()
  lns['subprocess'].run=fake_run;mark('main_dry_return',lns['main']()==37);mark('one_child_call',len(calls)==1)
  argv=calls[0][0][0];kw=calls[0][1];mark('dry_child_argv',argv==[str(lns['PYTHON']),'-I',str(lns['ENGINE']),'--authorized-launch']);mark('dry_child_cwd',kw['cwd']==str(HW));mark('dry_child_env',kw['env']==env0 and kw['close_fds'] is True and kw['check'] is False)
  sys.argv=[str(L),'x'];before=len(calls);mark('argv_attack_reject',rejects(lns['main']) and len(calls)==before)
 finally:sys.argv=oldargv
 with tempfile.TemporaryDirectory(prefix='m1096r2_') as t:
  p=Path(t);good=p/'good';good.write_bytes(b'abc');link=p/'link';link.symlink_to(good);bad=p/'bad';bad.write_bytes(b'abd')
  mark('launcher_verify_symlink_reject',rejects(lns['verify_regular'],link,sha(good)));mark('launcher_verify_bytes_reject',rejects(lns['verify_regular'],bad,sha(good)))
  mutated=p/'launcher.py';mutated.write_bytes(L.read_bytes()+b'\n#mut\n');mark('launcher_byte_attack_external_pin_reject',sha(mutated)!=X['launcher'])
  forged=dict(receipt);forged['engine_sha256']='0'*64;mark('receipt_engine_attack_reject',forged['engine_sha256']!=X['engine'])
  forged=dict(receipt);forged['m1093r2_outer_seal_file_sha256']='8188a86aa07856217223d6d939f7b3cd8c84ee3b10d7bacc62dee777d8e2e2ac';mark('old_hammer_seal_reject',forged['m1093r2_outer_seal_file_sha256']!=X['hammer_outer'])
 ens=module_defs(E,'m1096r2_engine_model');flow=text=E.read_text();fseg=flow[flow.index('def flow()'):flow.index('def quarantine(')]
 mark('attempt_collision_before_lock',fseg.index('if any(path.exists()')<fseg.index('LOCK.parent.mkdir'));mark('attempt_collision_before_consume',fseg.index('if any(path.exists()')<fseg.index('ATTEMPT.mkdir()'))
 with tempfile.TemporaryDirectory(prefix='m1096r2_attempt_') as t:
  p=Path(t);fakea=p/'attempt';fakea.mkdir();ens.update({'ATTEMPT':fakea,'RESULT':p/'result','WORK':p/'work','LOCK':p/'lock','static_gate':lambda:{}})
  mark('preset_attempt_attack_reject',rejects(ens['flow']) and fakea.is_dir() and not (p/'result').exists())
 # Parent argv contract: exact root command executes pinned Python with launcher
 # as argv[1]; the launcher remains alive while its subprocess runs.
 command=contract['launcher']['unique_future_command'];mark('env_i_command_prefix',command.startswith('/usr/bin/env -i LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/usr/bin:/bin TMPDIR=/tmp '));mark('env_i_pinned_python_launcher',f'/opt/anaconda3/envs/pytorch310/bin/python3.10 {L}' in command)
 mark('parent_argv_shape',str(lns['PYTHON'])=='/opt/anaconda3/envs/pytorch310/bin/python3.10' and L.is_absolute());mark('no_launcher_or_engine_executed',not A.exists() and not R.exists())
 result={'schema':'m1096r2_final_launch_hammer_mechanical_v1','status':'PASS_M1096R2_M1091R3_AUTHORIZED_LAUNCH_HAMMER__GO_ONE_ATTEMPT','verdict':'GO_ROOT_EXECUTE_ONE_EXACT_COMMAND','checks_passed':len(checks),'checks':checks,'identity':{'launcher_sha256':sha(L),'launch_receipt_outer_seal_file_sha256':sha(Path(str(LR)+'.sha256.seal.sha256')),'source_contract_outer_seal_file_sha256':sha(Path(str(C)+'.sha256.seal.sha256')),'author_receipt_outer_seal_file_sha256':sha(AR/'SHA256SUMS.seal.sha256'),'engine_sha256':sha(E),'m1093r2_outer_seal_file_sha256':sha(H/'SHA256SUMS.seal.sha256')},'execution':{'launcher_executed':False,'engine_executed':False,'eda':False,'attempt_consumed':False,'result_created':False},'exact_command':command}
 (HERE/'mechanical_checks.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
