#!/usr/bin/env python3
"""Only four new R30 states; old reference scalars and GT masks explicitly reused."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
import analysis_joint as core
import audit_schedule_v1 as schedule
import raw_quality_math as raw

HERE=Path(__file__).resolve().parent
R27=core.R27
R28=R27.parents[1]/'round28/independent_review'
REF27=R27/'actual_quality_v1/ACTUAL_RAW_QUALITY_V1.json'
REF28=R28/'actual_quality_v1/ACTUAL_RAW_QUALITY_V1.json'
REF27_SHA='2855b04808568ce34afc1fe097c14739011f73a2542e8331e71f777a37908a0e'
REF28_SHA='1d4e43b0224c38bb2e7d7d389c8ea711cd37b7f747d298eb0f817c63e0ffaabd'
MEMBERS_SHA=schedule.MEMBERS_SHA
ORIGINAL='b721031729a9a5f69d03037be53e62ebf8c1536dad0ad8a8f8e1601fe9fc2798'
GRAPH='flow_r30_joint_Q4_Q8_M_all12mlp_patch4_currentBN_v1'
AUDIT_SCHEMA='flow_r30_independent_actual_execution_audit_v1'
STAGES={'initial_M':('initial_M','initial_M'),
        'endpoint_Q4':('joint_Q4','recovered_joint_Q4'),
        'endpoint_Q8':('joint_Q8','recovered_joint_Q8'),
        'endpoint_M':('joint_M','recovered_joint_M')}
require=core.require

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def descriptor(d):
    p=Path(d['path'])
    require(sha(p)==d['sha256'],'Bound source bytes '+str(p))
    return p
def safe(root,rel):
    p=Path(rel)
    require(not p.is_absolute() and '..' not in p.parts,'Safe actual relative artifact')
    return Path(root)/p
def preflight(path):
    f=read(path)
    require(f['schema']=='flow_r30_joint_quality_io_freeze_v1','Independent frozen R30 IO')
    for n,h in f['files'].items():require(sha(HERE/n)==h,'Own frozen source '+n)
    for d in f['dependencies'].values():descriptor(d)
    ap=descriptor(f['auditor']);af=descriptor(f['auditor_freeze'])
    frozen=read(af)
    # Root audit is frozen in its own directory; some immutable old helpers
    # may be external. These are exact predeclared source paths, not input data.
    for n,h in frozen['files'].items():require(sha(ap.parent/n)==h,'Actual auditor frozen source '+n)
    descriptor(f['mask_contract'])
    require(Path(raw.__file__).resolve()==R27/'raw_quality_math.py','Frozen raw metric module')
    return f

def validate_audit(a,f,ids,result_sha,mask_sha):
    require(a['schema']==AUDIT_SCHEMA and a['status']=='PASS'
            and a['quality_analysis_allowed'] is True and a['engineering_smoke'] is False,
            'Complete formal actual execution only')
    require(a['auditor_sha256']==f['auditor']['sha256']
            and a['auditor_freeze_sha256']==f['auditor_freeze']['sha256'],
            'Final independent execution source/freeze')
    require(a['graph_id']==GRAPH and a['source_result_sha256']==result_sha
            and a['source_mask_contract_sha256']==mask_sha,'Same actual graph/result/quota')
    for k,n in dict(optimizer_updates=512,quality_predictions=256,formal_model_forwards=1289).items():
        require(a['counts'][k]==n,'Fixed completed count '+k)
    require(a['student_mode_counts']==dict(Q4=256,Q8=128,M=128),'Actual student distribution')
    rows=a['predictions'];by={(x['state'],x['target_id']):x for x in rows}
    labels={v[1] for v in STAGES.values()}
    require(len(rows)==len(by)==256 and set(by)=={(s,i) for s in labels for i in ids},
            'All four unique actual64 states, no parity/smoke quality')
    return by

def historical_sources(q27,q28,ids):
    """Byte-check old artifacts, explicitly reusing their already audited metrics."""
    result_descriptors={**q27['source_results'],'Q4':q28['source_result']}
    audit27=read(descriptor(q27['execution_audit']))
    audit28=read(descriptor(q28['execution_audit']))
    require(audit27['status']==audit28['status']=='PASS'
            and audit27['quality_analysis_allowed'] is audit28['quality_analysis_allowed'] is True,
            'Prior actual complete audits')
    results={};checked=0
    for arm,d in result_descriptors.items():
        rp=descriptor(d);r=read(rp);a=audit28 if arm=='Q4' else audit27
        require(a['source_results'][arm]==sha(rp) and r['arm']==arm
                and r['initial_state_sha256']==ORIGINAL and r['sources']['members']==MEMBERS_SHA,
                'Actual original-W matched ordinary source')
        by={(x['state'],x['target_id']):x for x in a['predictions']}
        for phase in ('initial','endpoint'):
            state=('original_native' if arm=='native' else 'initial_'+arm) if phase=='initial' else 'recovered_'+arm
            ev=r['evaluations'][phase]
            require([x['target_id'] for x in ev['frames']]==ids,'Old exact64 order')
            for row in ev['frames']:
                old=by[(state,row['target_id'])];p=safe(rp.parent,row['prediction_file'])
                require(p.resolve()==Path(old['file']).resolve()
                        and sha(p)==row['prediction_file_sha256']==old['file_sha256']
                        and row['prediction_sha256']==old['head3_sha256'],
                        'Old original prediction archive provenance')
                checked+=1
        results[arm]=(rp,r,q28['execution_audit'] if arm=='Q4' else q27['execution_audit'])
    require(checked==512,'Explicit eight prior64-state archives')
    return results,checked

def bind(inp,f):
    require(inp['schema']=='flow_r30_joint_quality_inputs_v1','R30 new256 input contract')
    require(sha(REF27)==REF27_SHA and sha(REF28)==REF28_SHA
            and sha(R27/'membership_v1.json')==MEMBERS_SHA,'Immutable references/membership')
    q27,q28=read(REF27),read(REF28);m=read(R27/'membership_v1.json')
    ids=[x['target_id'] for x in m['dev_members']]
    require(len(ids)==len(set(ids))==64 and [x['target_id'] for x in q27['frames']]
            ==[x['target_id'] for x in q28['frames']]==ids,'Same original development64')
    require(not set(ids)&set(m['sealed_ids']+m['H8_historical_ids']),'No sealed/H8 inputs')
    rp=descriptor(inp['run']);r=read(rp);ap=descriptor(inp['execution_audit']);a=read(ap)
    contract_path=descriptor(f['mask_contract']);contract=read(contract_path);mask_sha=sha(contract_path)
    by=validate_audit(a,f,ids,sha(rp),mask_sha)
    require(r['schema']=='flow_r30_joint_recovery_v1' and r['graph_id']==GRAPH
            and r['status']=='JOINT512_AND256_DEVELOPMENT_PREDICTIONS_COMPLETE'
            and r['arm']=='joint' and r['engineering_smoke'] is False,'Declared complete joint recovery')
    require(r['model_forwards']==r['expected_model_forwards']==1289
            and r['completed_optimizer_steps']==512 and r['training_performed'] is True,
            'Fixed ordinary training opportunity')
    require(all(r[k] is False for k in ('development_GT_read','quality_computed',
                'quality_selection','formal_resume_from_smoke')),'No dev selection or smoke reuse')
    require(r['initial_state_sha256']==r['teacher_initial_state_sha256']
            ==r['teacher_final_state_sha256']==ORIGINAL,'Original same teacher/student source')
    require(r['final_state_sha256']==r['training_state_sha256']==a['final_state_sha256']
            and r['final_state_sha256']!=ORIGINAL,'One genuinely updated final state')
    require(r['sources']['members']==MEMBERS_SHA and r['sources']['mask_contract']==mask_sha
            and r['joint_contract']['mask_contract']==contract
            and r['joint_contract']['mask_contract_sha256']==mask_sha,
            'Predeclared mask independent of quality')
    require(r['joint_contract']['mode_counts']==r['actual_student_mode_counts']
            ==dict(Q4=256,Q8=128,M=128),'Actual fixed mode counts')
    expected=schedule.audit(m)['schedule']
    lp=safe(rp.parent,r['training_ledger_file'])
    require(sha(lp)==r['training_ledger_sha256'],'Actual512 training ledger bytes')
    ledger=[json.loads(line) for line in lp.read_text().splitlines()]
    require(len(ledger)==512,'All512 updates')
    for l,e in zip(ledger,expected):
        require(all(l[k]==e[k] for k in ('step','target_id','sequence','pass_index','crop','mode')),
                'Independently reconstructed mode/crop/identity schedule')
    weights=safe(rp.parent,r['weights_file'])
    require(sha(weights)==r['weights_sha256']==a['checkpoint_sha256'],'Actual final checkpoint bytes')
    old_runs,old_count=historical_sources(q27,q28,ids)
    require(r['recipe']==old_runs['native'][1]['recipe'],'Matched ordinary recovery recipe')
    require(set(r['original_initial_references'])=={'initial_Q4','initial_Q8'},'Only original uniform references')
    for state,d in r['original_initial_references'].items():
        arm=state.split('_')[1];oldp,oldr,olda=old_runs[arm]
        require(d['result_sha256']==sha(oldp) and d['audit_sha256']==olda['sha256']
                and d['state_sha256']==oldr['evaluations']['initial']['state_sha256']==ORIGINAL
                and d['new_forwards']==0 and d['model_forwards_reused']==64,
                'Old original uniform observations explicitly reused')
    records={x['target_id']:x for x in m['records'] if x['role']=='development'}
    snapshots={x['target_id']:x for x in q27['shared_source_artifacts']}
    require(set(records)==set(snapshots)==set(ids),'Original64 common source snapshots')
    shared={};evidence={}
    for row in m['dev_members']:
        tid=row['target_id'];s=snapshots[tid]
        evidence[tid]={{'input':'events','gt':'gt','mask':'mask'}[x['category']]:
            dict(relative_path=x['relative_path'],sha256=x['sha256']) for x in records[tid]['source_files']}
        require(s['sequence']==row['sequence'] and s['input_evidence']==evidence[tid],
                'Same accepted source population')
        p=Path(s['file']);p=p if p.is_absolute() else safe(R27,p)
        require(sha(p)==s['file_sha256'],'Original common GT/mask archive bytes')
        shared[tid]=dict(path=p,record=s)
    require(set(r['evaluations'])==set(STAGES),'Only initial M and three final modes')
    bound={s:{} for s in core.NEW_STATES}
    for stage,(state,label) in STAGES.items():
        ev=r['evaluations'][stage]
        require(ev['model_forwards']==64 and ev['GT_arrays_read'] is False
                and ev['state_sha256']==(ORIGINAL if stage=='initial_M' else r['final_state_sha256']),
                'All endpoint modes use the same final W')
        if state in ('initial_M','joint_M'):
            require(ev['mask_layout']['geometry']=='full9' and ev['mask_layout']['step'] is None
                    and ev['mask_layout']['selected_tiles']==contract['evaluation_tiles'],
                    'Same predeclared M coordinates before/after recovery')
        else:require(ev['mask_layout'] is None,'Uniform mode no hidden mask')
        require([(x['target_id'],x['sequence']) for x in ev['frames']]
                ==[(x['target_id'],x['sequence']) for x in m['dev_members']],'New exact64 order')
        for row in ev['frames']:
            tid=row['target_id'];audit=by[(label,tid)]
            require(audit['sequence']==row['sequence']
                    and audit['input_evidence']==row['input_evidence']==evidence[tid],'New source identity')
            p=safe(rp.parent,row['prediction_file'])
            require(p.resolve()==Path(audit['file']).resolve()
                    and sha(p)==row['prediction_file_sha256']==audit['file_sha256'],'New actual archive')
            with np.load(p,allow_pickle=False) as z:
                require(z.files==['head3'],'Only one genuine quality payload')
                v=z['head3']
                require(v.dtype==np.float32 and v.shape==(2,480,640) and np.isfinite(v).all(),
                        'Finite pixel-unit full-resolution prediction')
                require(raw.payload_sha(v)==row['prediction_sha256']==audit['head3_sha256'],
                        'New actual payload bytes')
            bound[state][tid]=dict(path=p,sha256=audit['file_sha256'],payload_sha256=audit['head3_sha256'])
    return q27,q28,m,shared,bound,rp,ap,old_count,mask_sha

def run(args):
    require(not args.output.exists(),'Fresh independent quality output')
    f=preflight(args.freeze);inp=read(args.inputs)
    q27,q28,m,shared,bound,rp,ap,old_count,mask_sha=bind(inp,f)
    # No numeric GT read until complete actual audit and all new256 payloads close.
    frames=[]
    for row in m['dev_members']:
        tid=row['target_id'];s=shared[tid]
        require(sha(s['path'])==s['record']['file_sha256'],'Common source unchanged')
        with np.load(s['path'],allow_pickle=False) as z:
            require(set(z.files)==set(s['record']['payloads'])
                    =={'gt','mask','voxel_nonzero_bins'}|{'region__'+r for r in core.old.REGIONS},
                    'All19 original source arrays')
            arrays={k:z[k] for k in z.files}
        for k,v in arrays.items():
            d=s['record']['payloads'][k]
            require(list(v.shape)==d['shape'] and str(v.dtype)==d['dtype']
                    and raw.payload_sha(v)==d['sha256'],'Actual original source payload '+k)
        gt=arrays['gt'];valid=arrays['mask'];masks={r:arrays['region__'+r] for r in core.old.REGIONS}
        require(gt.dtype==np.float32 and gt.shape==(2,480,640) and np.isfinite(gt).all()
                and valid.dtype==np.bool_ and valid.shape==(480,640),'Original GT format')
        require(np.array_equal(masks['all_valid'],valid)
                and all(v.dtype==np.bool_ and v.shape==valid.shape and not np.any(v&~valid)
                        for v in masks.values()),'Complete16 fixed same-support subsets')
        states={}
        for state in core.NEW_STATES:
            d=bound[state][tid]
            require(sha(d['path'])==d['sha256'],'New archive unchanged')
            with np.load(d['path'],allow_pickle=False) as z:
                pred=z['head3']
                require(raw.payload_sha(pred)==d['payload_sha256'],'New payload unchanged')
                states[state]=raw.measure(pred,gt,masks)
        frames.append(dict(target_id=tid,sequence=row['sequence'],states=states))
    out=core.analyze(q27,q28,frames)
    out.update(schema='flow_r30_independent_raw_quality_v1',status='RAW_METRICS_COMPLETE',
        graph_id=GRAPH,source_result=dict(path=str(rp),sha256=sha(rp)),
        execution_audit=dict(path=str(ap),sha256=sha(ap)),source_mask_contract_sha256=mask_sha,
        reference_qualities={'R27':dict(path=str(REF27),sha256=REF27_SHA),
                             'R28':dict(path=str(REF28),sha256=REF28_SHA)},
        source_inputs_manifest_sha256=sha(args.inputs),io_freeze_sha256=sha(args.freeze),
        independently_recomputed_new_predictions=256,reused_GT_snapshot_archives=64,
        reused_reference_archives_byte_checked=old_count)
    args.output.mkdir(parents=True);p=args.output/'ACTUAL_RAW_QUALITY_V1.json';write(p,out)
    print(json.dumps(dict(status=out['status'],quality_sha256=sha(p),
        M_native_dual_reference_pass=out['M_native_dual_reference_pass'],
        configuration_decision=out['configuration_decision'],new_predictions=256)))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--freeze',type=Path,default=HERE/'IO_FREEZE_V1.json')
    p.add_argument('--source-preflight-only',action='store_true');a=p.parse_args()
    if a.source_preflight_only:preflight(a.freeze);print('R30_QUALITY_SOURCE_PREFLIGHT_PASS')
    else:run(a)
