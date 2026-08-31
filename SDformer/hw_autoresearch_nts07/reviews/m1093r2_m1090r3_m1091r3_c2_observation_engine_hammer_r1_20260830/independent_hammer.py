#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent bounded M1093r2 hammer; never launches EDA or consumes attempt."""
from __future__ import annotations
import ast, hashlib, json, os, re, stat, subprocess, tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent; HW=HERE.parents[1]
ENGINE=HW/'dc_handoff/scripts/m1091r3_m1090r3_c2_observation_authorized_engine_r1.py'
CONTRACT=HW/'contracts/m1090r3_c2_k1_observation_fixed_history_source_contract_r1_20260830.json'
RELEASE=HW/'contracts/m1090r3_c2_k1_observation_fixed_history_release_r1_20260830.json'
RECEIPT=HW/'reviews/m1090r3_m1091r3_c2_observation_fixed_history_source_receipt_r1_20260830'
M1093=HW/'reviews/m1093_m1090r2_m1091r2_c2_observation_engine_hammer_r1_20260830'
M1092=HW/'reviews/m1092_m1090_c2_observation_source_hammer_r1_20260830'
M1088=HW/'reviews/m1088_m1080_c2_mapped_gate_failure_audit_r1_20260830'
M1080A=HW/'results/.m1080_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_attempt_consumed'
M1080F=HW/'results/m1080_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_r1_20260830.failed_or_incomplete.2746017.quarantine'
ATTEMPT=HW/'results/.m1091r3_m1090r3_c2_observation_dc_mapped_vcs_attempt_consumed'
RESULT=HW/'results/m1091r3_m1090r3_c2_observation_dc_mapped_vcs_r1_20260830'
LAUNCHER=HW/'dc_handoff/scripts/run_m1091r3_m1090r3_c2_observation_authorized_launch_r1.py'
LAUNCH_RECEIPT=HW/'contracts/m1091r3_m1090r3_c2_observation_authorized_launch_receipt_r1_20260830.json'
DOCS=HW/'docs/359_DATE终局冻结_20260813.md'
E={'receipt':'8bc6f725ef0ec7055441afafa2c0bd5c5ba54620c4354feaf2a6763fbabedd9e',
'engine':'41b7899083152f8099acac759109a8eb22c381cb6a17506ae85e6666656daf04',
'contract':'bdb443003de0e26b7dcb6e29838eec8e024e843f90ce033aac2203330287e808',
'contract_outer':'d2e5d49d9e5cc11f1927ad75bc621b59f341c56352c58242b7f2dbd84db82c0d',
'release':'15f40b39b3f96b06978b9d9966c9bfeedfcbff7c018651101c1d926f8f7df954',
'release_outer':'fc6bb48800c7d595203aee21ccba140753f38fd04bc28f93425a9dd74dc9c853',
'm1093':'8188a86aa07856217223d6d939f7b3cd8c84ee3b10d7bacc62dee777d8e2e2ac',
'm1092':'f55dc0afde8d350d1ff028c30e511eb15b2670f3ad1ee2f5643759406ca8ccb4',
'm1088':'fb3f208dc704c7663769422ad9f27b17851cc86b11826727fe0c0c795260bd5f',
'm1080a':'21944247a673bda71a1d3f8cce2cf567b91e51a661b88d5028ed89b70d3a8f7c',
'm1080f':'2e3367c239cda08987027a55a01f65b0cbebbd1c0dd907a9a945aa12f5cea89d',
'docs':'dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4'}
checks=[]
class F(RuntimeError):pass
def req(v,m):
    if not v: raise F(m)
def mark(n,v=True): req(v,n);checks.append(n)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def regular(p,d):
    p=Path(p);return stat.S_ISREG(p.lstat().st_mode) and not p.is_symlink() and sha(p)==d
def flat(p,outer):
    p=Path(p);m=p/'SHA256SUMS';o=p/'SHA256SUMS.seal.sha256'
    req(p.is_dir() and not p.is_symlink() and regular(o,outer),'flat outer')
    for line in m.read_text().splitlines():
        d,n=line.split(None,1);req(regular(p/n.lstrip('*'),d),'flat member')
    req(o.read_text().split()==[sha(m),'SHA256SUMS'],'flat content')
def history(p,outer):
    p=Path(p);m=p/'SHA256SUMS';o=p/'SHA256SUMS.seal.sha256';root=p.resolve();syms=0
    req(regular(o,outer) and o.read_text().split()==[sha(m),'SHA256SUMS'],'history seal')
    for line in m.read_text().splitlines():
        d,n=line.split(None,1);r=Path(n.lstrip('*'));req(not r.is_absolute() and '..' not in r.parts,'escape')
        q=p/r;mode=q.lstat().st_mode
        if stat.S_ISLNK(mode):
            syms+=1;t=q.resolve(strict=True);req((t==root or root in t.parents) and stat.S_ISREG(t.lstat().st_mode) and not t.is_symlink(),'symlink')
        else:req(stat.S_ISREG(mode),'member kind')
        req(sha(q)==d,'followed hash')
    return syms
def double(p,d,o):
    s=Path(str(p)+'.sha256');x=Path(str(p)+'.sha256.seal.sha256')
    return regular(p,d) and s.read_text().split()==[d,p.relative_to(HW).as_posix()] and sha(x)==o and x.read_text().split()==[sha(s),s.relative_to(HW).as_posix()]
def defs():
    t=ast.parse(ENGINE.read_text());allowed=(ast.Import,ast.ImportFrom,ast.Assign,ast.AnnAssign,ast.ClassDef,ast.FunctionDef)
    ns={'__file__':str(ENGINE),'__name__':'m1093r2_model'};exec(compile(ast.Module([n for n in t.body if isinstance(n,allowed)],[]),str(ENGINE),'exec'),ns);return ns
def seal_temp(root):
    members=sorted(q for q in root.rglob('*') if q.is_file() and q.name not in {'SHA256SUMS','SHA256SUMS.seal.sha256'})
    m=root/'SHA256SUMS';m.write_text(''.join(f'{sha(q)}  {q.relative_to(root)}\n' for q in members));o=root/'SHA256SUMS.seal.sha256';o.write_text(f'{sha(m)}  SHA256SUMS\n');return sha(o)
def rejects(fn,*args):
    try:fn(*args)
    except Exception:return True
    return False
def main():
    m=RECEIPT/'SHA256SUMS';o=RECEIPT/'SHA256SUMS.seal.sha256'
    mark('01_receipt_outer',sha(o)==E['receipt']);mark('02_receipt_outer_content',o.read_text().split()==[sha(m),'SHA256SUMS'])
    listed=[]
    for line in m.read_text().splitlines():d,n=line.split(None,1);req(regular(RECEIPT/n,d),'receipt member');listed.append(n)
    mark('03_receipt_members',len(listed)==6);mark('04_receipt_coverage',set(listed)=={q.name for q in RECEIPT.iterdir() if q.is_file() and q.name not in {'SHA256SUMS','SHA256SUMS.seal.sha256'}})
    mark('05_contract_double',double(CONTRACT,E['contract'],E['contract_outer']));mark('06_release_double',double(RELEASE,E['release'],E['release_outer']))
    mark('07_engine_sha',sha(ENGINE)==E['engine']);mark('08_docs',sha(DOCS)==E['docs'])
    flat(M1093,E['m1093']);mark('09_m1093_stop_seal');mark('10_m1093_stop_status',load(M1093/'review.json')['status'].startswith('STOP_M1093_'))
    flat(M1092,E['m1092']);mark('11_m1092_seal');mark('12_m1092_stop',load(M1092/'review.json')['status'].startswith('STOP_M1092_'))
    flat(M1088,E['m1088']);mark('13_m1088_seal');mark('14_m1088_status',load(M1088/'review.json')['status']=='PASS_M1088_M1080_FAILURE_AUDIT__M1080_DO_NOT_RETRY')
    flat(M1080A,E['m1080a']);mark('15_m1080_attempt_seal');mark('16_real_history_valid',history(M1080F,E['m1080f'])==1)
    mark('17_new_namespaces_absent',not ATTEMPT.exists() and not RESULT.exists());mark('18_launcher_absent',not LAUNCHER.exists() and not LAUNCH_RECEIPT.exists())
    c=load(CONTRACT);r=load(RELEASE);mark('19_contract_boundary',c['launch_now'] is False and c['max_attempts_now']==0);mark('20_release_boundary',r['launch_now'] is False)
    pins=c['source_sha256'];req(len(pins)==21,'pin count')
    for i,(p,d) in enumerate(sorted(pins.items()),21):mark(f'{i:02d}_source_{Path(p).name}',regular(HW/p,d))
    regs=[(Path(p),v['sha256']) for p,v in c['external_identity'].items() if v['kind']=='regular'];req(len(regs)==7,'ext count')
    for i,(p,d) in enumerate(regs,42):mark(f'{i:02d}_external_{p.name}',regular(p,d))
    dc=Path('/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell');mark('49_dc_symlink',dc.is_symlink());mark('50_dc_readlink',os.readlink(dc)=='snps_shell')
    text=ENGINE.read_text();flow=text[text.index('def flow()'):text.index('def quarantine(')]
    mark('51_fixed_argv','sys.argv[1:] != ["--authorized-launch"]' in text);mark('52_no_expected_env','M1091_EXPECTED' not in text)
    mark('53_history_exact_path','directory != M1080_FAILURE' in text);mark('54_history_inside_check','root not in resolved.parents' in text)
    mark('55_history_regular_target','historical symlink target is not regular' in text);mark('56_history_followed_hash','historical followed-byte digest drift' in text)
    mark('57_history_only_call',text.count('verify_frozen_history_flat(')==2);mark('58_live_regular_policy',text.count('verify_regular(')>10)
    mark('59_static_before_attempt',flow.index('static_gate()')<flow.index('ATTEMPT.mkdir()'));mark('60_lock_resource_license_before_attempt',all(flow.index(x)<flow.index('ATTEMPT.mkdir()') for x in ('flock(','collision_gate()','resource_gate()','license_gate()')))
    mark('61_one_dc',flow.count('str(DC_SHELL)')==1);mark('62_one_vcs',flow.count('str(VCS)')==1);mark('63_one_sim',flow.count('run([str(simv), "-no_save"]')==1)
    mark('64_no_saif_initreg','SAIF' not in flow and 'initreg' not in flow);mark('65_diagnostic_only','"paper_citable": False' in flow)
    w=(HW/'rtl_m1090r3/m1090r3_c2_k1_observation_wrapper.sv').read_text();tb=(HW/'dc_handoff/tb/tb_m1090r3_c2_k1_observation_mapped_case0_short.sv').read_text()
    obs=set(re.findall(r'\bobs_[A-Za-z0-9_]+\b',w[w.index('module '):w.index(');')]))
    mark('66_obs22',len(obs)==22);mark('67_fanout',all(len(re.findall(rf'\b{x}\b',w))==2 for x in obs));mark('68_xchecks',tb.count('`M1090R3_FAIL_X(')==22);mark('69_window',all(x in tb for x in ('window_cycle==128','wait_cycles<16','wait_cycles<32','M1090R3_STAGE','#1000 $fatal')))
    before=(ATTEMPT.exists(),RESULT.exists());env=os.environ.copy();env['M1091_EXPECTED_ENGINE_SHA256']='0'*64
    runs=[]
    for argv in ([],['--authorized-launch','x'],['--authorized-launch']):runs.append(subprocess.run(['/opt/anaconda3/envs/pytorch310/bin/python3.10',str(ENGINE),*argv],capture_output=True,text=True,env=env,timeout=30))
    mark('70_noargv_stop',runs[0].returncode==3 and 'fixed argv required' in runs[0].stderr);mark('71_extraargv_stop',runs[1].returncode==3 and 'fixed argv required' in runs[1].stderr)
    mark('72_legal_preflight_reaches_launcher_boundary',runs[2].returncode==3 and 'fixed launch wrapper/receipt absent' in runs[2].stderr and before==(ATTEMPT.exists(),RESULT.exists()))
    ns=defs()
    with tempfile.TemporaryDirectory(prefix='m1093r2_hist_') as t:
        root=Path(t)/'hist';root.mkdir();target=root/'target';target.write_bytes(b'abc');link=root/'link';link.symlink_to('target');outer=seal_temp(root);ns['M1080_FAILURE']=root
        mark('73_synthetic_valid_history',ns['verify_frozen_history_flat'](root,outer)==1)
        other=Path(t)/'other';other.mkdir();mark('74_nonm1080_path_reject',rejects(ns['verify_frozen_history_flat'],other,outer))
        outside=Path(t)/'outside';outside.write_bytes(b'abc');link.unlink();link.symlink_to(outside);outer=seal_temp(root);mark('75_escape_target_reject',rejects(ns['verify_frozen_history_flat'],root,outer))
        link.unlink();link.symlink_to('target');target.write_bytes(b'changed');mark('76_followed_hash_drift_reject',rejects(ns['verify_frozen_history_flat'],root,outer))
        target.write_bytes(b'abc');same=root/'same';same.write_bytes(b'abc');link.unlink();link.symlink_to('same');outer=seal_temp(root);mark('77_same_bytes_internal_target_allowed',ns['verify_frozen_history_flat'](root,outer)==1)
        manifest=root/'SHA256SUMS';manifest.write_text(f'{sha(target)}  ../target\n');(root/'SHA256SUMS.seal.sha256').write_text(f'{sha(manifest)}  SHA256SUMS\n');mark('78_manifest_escape_reject',rejects(ns['verify_frozen_history_flat'],root,sha(root/'SHA256SUMS.seal.sha256')))
        rootlink=Path(t)/'rootlink';rootlink.symlink_to(root);mark('79_history_directory_symlink_reject',rejects(ns['verify_frozen_history_flat'],rootlink,sha(root/'SHA256SUMS.seal.sha256')))
        live=Path(t)/'live';live.write_bytes(b'live');live_link=Path(t)/'live_link';live_link.symlink_to(live)
        mark('80_live_symlink_reject',rejects(ns['verify_regular'],live_link,sha(live)));mark('81_live_byte_swap_reject',rejects(ns['verify_regular'],live,'0'*64))
        extlink=Path(t)/'external_lib';extlink.symlink_to('/opt/anaconda3/envs/pytorch310/bin/python3.10');mark('82_external_tool_symlink_reject',rejects(ns['verify_regular'],extlink,sha(Path('/opt/anaconda3/envs/pytorch310/bin/python3.10'))))
    req(len(checks)==82,f'count {len(checks)}')
    result={'status':'PASS_M1093R2_M1090R3_M1091R3_ENGINE_HAMMER__AUTHOR_LAUNCH_WRAPPER_ONLY__NO_EDA','checks_passed':82,'checks':checks,'identity':{'receipt_outer_seal_file_sha256':sha(o),'engine_sha256':sha(ENGINE),'contract_outer_seal_file_sha256':sha(Path(str(CONTRACT)+'.sha256.seal.sha256')),'release_outer_seal_file_sha256':sha(Path(str(RELEASE)+'.sha256.seal.sha256'))},'attacks':{'historical_escape':'REJECT','historical_non_exact_path':'REJECT','historical_target_byte_drift':'REJECT','historical_same_bytes_internal_retarget':'ALLOW_BY_FOLLOWED_BYTE_SEMANTICS','live_symlink':'REJECT','live_byte_swap':'REJECT','caller_env':'NO_AUTHORITY_EFFECT','caller_argv':'REJECT'},'authorization':{'author_zero_argument_launcher':True,'launch_now':False,'attempt_now':False,'eda_now':False,'m1096r2_required':True}}
    (HERE/'mechanical_checks.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
