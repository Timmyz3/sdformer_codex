"""R22 identity/source admission, no automatic GT-value access."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;RUN=HERE.parent;DATE=RUN.parent;REPO=HERE.parents[3]
ROLES=DATE/'action_domain_round20/independent_review/SELECTOR_ROLES_V1.json'
ROLE_SHA='e6f216f6b017fc2cbba262ec51c70474f342e593e1549058a2200970949e10a1'
REGIONS=REPO/'research/flow_story_20261016/coverage_recovery_round13/preparation/regions_v1.json'
REGION_SHA='2e4934f6d82663f16ca1ae6c3aea238a31e699d4f49916052f6ea6b5e9c1d41b'
MATH=REPO/'research/flow_story_20261016/coverage_recovery_round13/independent_quality_review/reduce_raw_quality.py'
MATH_SHA='26a2e5c97b988f10685e1c440a11506354d9591cb9b36a567292abb6d0348088'
ANALYSIS_SHA='e24e7de47c27f0740a3b10ac84e2ed8b5b0816ac97d64be2cb074e7c97fc84ca'
GRAPH='r22_W271482_all78current_native_fit16dev8'
def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def payload(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(path,obj):
    p=Path(path);require(not p.exists(),'Fresh artifact');p.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def preflight(path,digest):
    require(sha(path)==digest,'Fixed IO source manifest');f=read(path)
    require(f['schema']=='flow_r22_independent_quality_io_freeze_v1','IO freeze schema')
    for n,h in f['own_files'].items():require(sha(HERE/n)==h,'Own frozen source '+n)
    for n,v in f['external_sources'].items():require(sha(REPO/v['relative_path'])==v['sha256'],'External source '+n)
    require(sha(ROLES)==ROLE_SHA and sha(REGIONS)==REGION_SHA and sha(MATH)==MATH_SHA,'Frozen roles/regions/original metric')
    require(sha(HERE/'ANALYSIS_FREEZE_V1.json')==ANALYSIS_SHA,'Fixed science rules')
    for n,h in read(HERE/'ANALYSIS_FREEZE_V1.json')['files'].items():require(sha(HERE/n)==h,'Frozen science source '+n)
    return f

def plan(members):
    out=[]
    for i,row in enumerate(members):
        role='fit' if i<16 else 'development'
        out.append((row['target_id'],role,'native',True))
        if i==0:out.extend((row['target_id'],role,f'native_repeat{j}',False) for j in (1,2))
    require(len(out)==26 and sum(x[3] for x in out)==24,'Native26 fixed roles/replays')
    return out

def source_admission(freeze,result_dir,input_root):
    """Hash source/archives only; never np.load a GT, event or prediction array."""
    ext=freeze['external_sources'];rp=Path(result_dir)/'results.json';result=read(rp)
    ap=REPO/freeze['actual_execution_audit_relative_path'];audit=read(ap)
    require(result['schema']=='flow_r22_representation_v1' and result['status']=='REPRESENTATION_NATIVE26_COMPLETE','Complete actual native26')
    require(result['graph_id']==result['binding']['graph_id']==GRAPH,'Fixed native graph')
    require(audit['schema']=='flow_r22_representation_independent_execution_audit_v1' and audit['status']=='PASS' and audit['representation_artifacts_allowed'] is True and audit['quality_analysis_allowed'] is False,'Native artifacts admission preserves dev GT seal')
    require(audit['admission']=='NATIVE24_ARTIFACTS_ONLY_DEV_GT_REQUIRES_FROZEN_PREDICTIONS','No automatic dev quality')
    require(audit['source_result_sha256']==sha(rp),'Actual audited result')
    require(audit['auditor_sha256']==ext['execution_auditor']['sha256'] and audit['pre_result_freeze_sha256']==ext['execution_audit_freeze']['sha256'],'Frozen auditor')
    require(audit['source_freeze_sha256']==ext['runner_source_freeze']['sha256']==result['binding']['source_freeze_sha256'],'Frozen native execution')
    b=result['binding'];require(b['inputs_sha256']==ext['runner_inputs']['sha256'] and b['independent_admission_sha256']==ext['source_admission']['sha256'],'Exact inputs/admission')
    require(b['plan']=='native26' and b['selector_roles_sha256']==ROLE_SHA and b['region_sha256']==REGION_SHA and b['BN']=='all78_current' and b['primitive_calls']==b['confirmation_array_accesses']==0,'Fixed state/roles/no primitive/sealed access')
    required=dict(full_forwards=26,native=26,compact=0,expanded=0,primitive_calls=0,quality_endpoints=24,determinism_replays=2,FC1_rows=14508000,FC2_rows=14508000,BN_calls=2028,archives=26,head_payloads=104,capture_payloads=104,optimizer_steps=0,calibration_forwards=0)
    require(all(audit['counts'][k]==v for k,v in required.items()),'Native26 complete independent actual counts')
    require(len(audit['numerical_checks'])==2 and all(x['kind']=='native_exact_replay' and x['reference_forward_id']==1 and x['pass'] is True for x in audit['numerical_checks']),'Two native exact replays')
    require(audit['sealed_array_accesses']==0 and audit['GT_array_values_loaded'] is False and audit['source_file_count']==72,'Actual source access scope')
    require(result['full_forward_attempts']==result['full_forward_count']==26 and result['training_performed'] is False and result['selector_training_performed'] is False,'No hidden attempts/training')
    roles=read(ROLES);members=roles['groups']['fit']+roles['groups']['development'];meta={r['target_id']:r for r in roles['records']}
    require(len(members)==24 and len({r['target_id'] for r in members})==24,'Fixed selected identities')
    evidence={};paths={}
    for row in members:
        tid=row['target_id'];evidence[tid]={};paths[tid]={}
        for item in meta[tid]['accepted_array_metadata']:
            cat='events' if item['category']=='input' else item['category'];relative=Path(item['relative_path'])
            require(not relative.is_absolute() and '..' not in relative.parts,'Safe source path')
            path=Path(input_root)/relative
            require(path.stat().st_size==int(item['bytes']) and sha(path)==item['sha256'],'Original source bytes '+tid+'/'+cat)
            evidence[tid][cat]=item['sha256'];paths[tid][cat]=path
    expect=plan(members);require(len(result['forward_records'])==len(audit['predictions'])==26,'All26 native records')
    native={}
    for fid,(r,ir,want) in enumerate(zip(result['forward_records'],audit['predictions'],expect),1):
        keys=('target_id','selector_role','arm','quality_endpoint')
        require(tuple(r[k] for k in keys)==tuple(ir[k] for k in keys)==want and r['forward_id']==ir['forward_id']==fid,'Exact role/order/replay membership')
        require(r['execution']==ir['execution']=='native' and r['endpoint_arm']==ir['endpoint_arm']=='native' and r['target_execution']=={},'Pure native')
        require(r['input_evidence']==evidence[r['target_id']],'Record input provenance')
        name=Path(r['prediction_file']);require(len(name.parts)==1 and name.suffix=='.npz' and name.name==Path(ir['file']).name,'Actual archive basename')
        path=Path(result_dir)/name;require(sha(path)==r['prediction_file_sha256']==ir['file_sha256'],'Full original native archive bytes')
        require(ir['u_sha256']==r['tensor_records']['u']['sha256'] and ir['head3_sha256']==r['head_records'][3]['sha256'],'Audited actual payload metadata')
        if r['quality_endpoint']:native[r['target_id']]=(path,r)
    require(set(native)==set(evidence),'All24 nonduplicate native endpoints')
    return dict(result=result,audit=audit,result_path=rp,audit_path=ap,members=members,native=native,paths=paths,evidence=evidence)
