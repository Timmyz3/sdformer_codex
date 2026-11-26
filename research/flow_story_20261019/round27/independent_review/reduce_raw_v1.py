#!/usr/bin/env python3
"""R27 new64 x six states: original-array quality after complete execution audit.

No runner metrics are imported. No training, model, SSH, GPU or sealed/H8 array
access. The independent execution auditor covers the actual optimizer/kernel
trace; this program independently binds its384 artifacts and remeasures GT.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import analysis_core as core
import raw_quality_math as raw

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
MEMBERS_SHA='40eba2514b56fd33a910fda4e37d4f86e80431ffe8149efc5f79c18827991e7b'
REGIONS_SHA='2e4934f6d82663f16ca1ae6c3aea238a31e699d4f49916052f6ea6b5e9c1d41b'
ANALYSIS_SHA='7b9bc437cb42901fc794bc2caf519b600c9a809be380c8dac82cf9f3733730af'
REGIONS_PATH=REPO/'research/flow_story_20261016/coverage_recovery_round13/preparation/regions_v1.json'
GRAPH='flow_r27_all12mlp_patch4conv_integer_Q6_complete9_tailQ8_all78current_v1'
AUDIT_SCHEMA='flow_r27_independent_actual_execution_audit_v1'
ARMS=('native','Q8','Q6')
STATE={('native','initial'):'original_native',('native','endpoint'):'recovered_native',
       ('Q8','initial'):'initial_Q8',('Q8','endpoint'):'recovered_Q8',
       ('Q6','initial'):'initial_Q6',('Q6','endpoint'):'recovered_Q6'}

require=core.require

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def safe_child(root,relative):
    r=Path(relative)
    require(not r.is_absolute() and '..' not in r.parts,'Safe relative source/artifact path')
    return Path(root)/r

def checked_descriptor(d):
    p=Path(d['path'])
    require(sha(p)==d['sha256'],'Actual bound file bytes: '+str(p))
    return p

def source_preflight(freeze_path):
    freeze=read(freeze_path)
    require(freeze['schema']=='flow_r27_raw_quality_io_freeze_v1','Independent raw IO freeze')
    for name,h in freeze['files'].items():require(sha(HERE/name)==h,'Frozen independent source '+name)
    require(sha(HERE/'ANALYSIS_FREEZE_V1.json')==ANALYSIS_SHA,'Original prospective rule freeze')
    pure=read(HERE/'ANALYSIS_FREEZE_V1.json')
    for name,h in pure['files'].items():require(sha(HERE/name)==h,'Original prospective math '+name)
    for key in ('auditor','audit_freeze'):
        checked_descriptor(freeze[key])
    audit_freeze_path=Path(freeze['audit_freeze']['path'])
    for name,h in read(audit_freeze_path)['files'].items():
        require(sha(safe_child(audit_freeze_path.parent,name))==h,'Actual auditor frozen dependency '+name)
    return freeze

def bind_artifacts(inp,freeze):
    """All metadata, source bytes and384 original predictions before GT values."""
    require(inp['schema']=='flow_r27_raw_quality_inputs_v1','R27 input schema')
    require(sha(HERE/'membership_v1.json')==MEMBERS_SHA and sha(REGIONS_PATH)==REGIONS_SHA,'Fixed roles/regions')
    members=read(HERE/'membership_v1.json');regions=read(REGIONS_PATH)
    require(regions['fit_uses_final_valid'] is False,'No development threshold fit')
    ids=[r['target_id'] for r in members['dev_members']]
    canonical={r['target_id']:r for r in members['dev_members']}
    records={r['target_id']:r for r in members['records'] if r['role']=='development'}
    require(len(ids)==len(set(ids))==64 and set(records)==set(ids),'Exact64 development sources')
    require(not set(ids)&set(regions['fit_target_ids']),'No region fit/dev identity reuse')
    require(not set(ids)&set(members['sealed_ids']+members['H8_historical_ids']),'No sealed/H8 source access')
    audit_path=checked_descriptor(inp['execution_audit']);audit=read(audit_path)
    require(audit['schema']==AUDIT_SCHEMA and audit['status']=='PASS' and audit['quality_analysis_allowed'] is True,
            'Complete actual independent execution required')
    require(audit['auditor_sha256']==freeze['auditor']['sha256'] and
            audit['auditor_freeze_sha256']==freeze['audit_freeze']['sha256'] and
            audit['pre_result_freeze_sha256']==ANALYSIS_SHA,'Actual auditor and prospective rule binding')
    require(audit['counts']['optimizer_updates']==1536 and audit['counts']['quality_predictions']==384 and
            audit['counts']['quality_archives']==384,'Full three-arm fixed endpoint population')
    require(set(inp['runs'])==set(ARMS) and set(audit['source_results'])==set(ARMS),'All3 run results')
    predictions=audit['predictions']
    by_key={(x['state'],x['target_id']):x for x in predictions}
    require(len(predictions)==len(by_key)==384 and set(by_key)=={(s,t) for s in core.STATES for t in ids},'Audit complete6x64 unique keys')
    source_paths={};source_evidence={}
    for tid in ids:
        rec=records[tid]
        require(rec['canonical']==canonical[tid] and len(rec['source_files'])==3,'Original canonical source member')
        source_paths[tid]={};source_evidence[tid]={}
        for item in rec['source_files']:
            kind={'input':'events','gt':'gt','mask':'mask'}[item['category']]
            require(kind not in source_paths[tid],'Unique event/GT/mask')
            path=safe_child(inp['source_input_root'],item['relative_path'])
            require(sha(path)==item['sha256'] and path.stat().st_size==int(item['bytes']),'Accepted dev source bytes')
            source_paths[tid][kind]=path
            source_evidence[tid][kind]=dict(relative_path=item['relative_path'],sha256=item['sha256'])
        require(set(source_paths[tid])=={'events','gt','mask'},'All3 source arrays')
    bound={};result_sources={};common={}
    for arm in ARMS:
        result_path=checked_descriptor(inp['runs'][arm]);result=read(result_path)
        require(audit['source_results'][arm]==inp['runs'][arm]['sha256'],'Audit actual result byte binding')
        require(result['schema']=='flow_r27_bounded_recovery_v1' and result['graph_id']==GRAPH and result['arm']==arm and
                result['status']=='FIXED512_AND128_DEVELOPMENT_PREDICTIONS_COMPLETE',
                'Declared actual ordinary graph/arm')
        # The actual status, all512 updates, model states, step order and runtime
        # closure are independently checked by the pinned execution auditor.
        ledger_path=safe_child(result_path.parent,result['training_ledger_file'])
        require(sha(ledger_path)==result['training_ledger_sha256'],'Actual completed training ledger bytes')
        steps=[json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
        require(result['completed_optimizer_steps']==512 and len(steps)==512,'Fixed optimizer endpoint')
        for actual,expected in zip(steps,members['train_order']):
            require(all(actual[k]==expected[k] for k in ('step','target_id','sequence','pass_index','crop')),
                    'Same full512 fixed training order/crops')
        require(result['development_GT_read'] is False and result['quality_computed'] is False and
                result['quality_selection'] is False and result['formal_resume_from_smoke'] is False,
                'Fixed endpoint; development GT not used by training/evaluation runner')
        for key in ('initial_state_sha256','teacher_initial_state_sha256'):
            common.setdefault(key,result[key]);require(result[key]==common[key],'Same original state/teacher acrossarms')
        common.setdefault('recipe',result['recipe']);require(result['recipe']==common['recipe'],'Matched ordinary training opportunity')
        require(result['sources']['members']==MEMBERS_SHA,'Current fixed320 roles')
        for phase in ('initial','endpoint'):
            rows=result['evaluations'][phase]['frames']
            require([(r['target_id'],r['sequence']) for r in rows]==[(r['target_id'],r['sequence']) for r in members['dev_members']],
                    'Exact64 order/sequence for every state')
            state=STATE[(arm,phase)]
            for row in rows:
                tid=row['target_id'];a=by_key[(state,tid)]
                require(row['input_evidence']==source_evidence[tid]==a['input_evidence'],'Same actual original source bytes')
                require(a['sequence']==row['sequence'],'Audit sequence mapping')
                path=safe_child(result_path.parent,row['prediction_file'])
                require(path.resolve()==Path(a['file']).resolve(),'Audited actual artifact path')
                require(sha(path)==row['prediction_file_sha256']==a['file_sha256'],'Actual prediction archive')
                with np.load(path,allow_pickle=False) as z:
                    require(z.files==['head3'],'Exactly one quality head3; no diagnostic mixing')
                    pred=z['head3']
                    require(pred.shape==(2,480,640) and pred.dtype==np.float32 and np.isfinite(pred).all(),'Full-grid pixel float32 quality')
                    h=raw.payload_sha(pred)
                    require(h==row['prediction_sha256']==a['head3_sha256'],'Independent original prediction payload')
                bound[(state,tid)]=dict(path=path,archive_sha256=a['file_sha256'],payload_sha256=h)
        result_sources[arm]=dict(path=str(result_path),sha256=sha(result_path))
    return members,regions,source_paths,source_evidence,bound,result_sources,audit_path

def run(a):
    require(not a.output.exists(),'Fresh independent output directory')
    freeze=source_preflight(a.freeze);inp=read(a.inputs)
    members,regions,source_paths,source_evidence,bound,results,audit_path=bind_artifacts(inp,freeze)
    # First GT numerical access is below: full audit +192 input SHA +384 real
    # predictions closed. Sealed/H8/fit GT paths are absent from source_paths.
    a.output.mkdir(parents=True)
    snapshot=a.output/'shared_development_arrays';snapshot.mkdir()
    frames=[];saved=[]
    for row in members['dev_members']:
        tid=row['target_id'];original={k:np.load(p,allow_pickle=False) for k,p in source_paths[tid].items()}
        require(original['events'].shape==(10,480,640) and original['gt'].shape==(2,480,640) and original['mask'].shape==(480,640),'Original full input geometry')
        masks=raw.region_masks(original['gt'],original['mask'],original['events'],
              regions['boundary_threshold_pixels'],regions['large_displacement_threshold_pixels'])
        states={}
        for state in core.STATES:
            spec=bound[(state,tid)]
            require(sha(spec['path'])==spec['archive_sha256'],'Artifact unchanged since admission')
            with np.load(spec['path'],allow_pickle=False) as z:
                pred=z['head3'];require(raw.payload_sha(pred)==spec['payload_sha256'],'Payload unchanged')
                states[state]=raw.measure(pred,original['gt'],masks)
        path=snapshot/(tid+'.npz')
        arrays=raw.artifact_arrays(original['gt'],original['mask'],original['events'],masks)
        np.savez_compressed(path,**arrays)
        saved.append(dict(target_id=tid,sequence=row['sequence'],file=str(path),file_sha256=sha(path),
                          input_evidence=source_evidence[tid],payloads={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=raw.payload_sha(v)) for k,v in arrays.items()}))
        frames.append(dict(target_id=tid,sequence=row['sequence'],states=states))
    out=core.analyze(frames)
    out.update(schema='flow_r27_independent_raw_quality_v1',status='RAW_METRICS_COMPLETE',
               graph_id=GRAPH,source_results=results,source_inputs_manifest_sha256=sha(a.inputs),
               execution_audit=dict(path=str(audit_path),sha256=sha(audit_path)),
               analysis_freeze_sha256=sha(HERE/'ANALYSIS_FREEZE_V1.json'),io_freeze_sha256=sha(a.freeze),
               membership_sha256=MEMBERS_SHA,region_sha256=REGIONS_SHA,
               original_array_count=192,independently_recomputed_predictions=384,
               shared_source_artifacts=saved,
               event_quantity_note='voxel_nonzero_bins counts occupied signed voxel bins, not original raw events; active= count>0.',
               source_scope='All64 source arrays and all384 real head3 payloads checked. Execution/optimizer/STE verification belongs to the separately pinned actual execution auditor. No runner quality metrics reused.')
    write(a.output/'ACTUAL_RAW_QUALITY_V1.json',out)
    print(json.dumps(dict(status=out['status'],resource_decision=out['resource_decision'],
         quality_sha256=sha(a.output/'ACTUAL_RAW_QUALITY_V1.json'),quality_predictions=384,
         review_flags={k:dict(high_support_count=len(v['high_support_tail_review']),
                            sequence_flagged=[r for r,x in v['sequence_balanced_review'].items() if x['review_flag']])
                       for k,v in out['comparisons'].items()})))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',required=True,type=Path)
    p.add_argument('--freeze',type=Path,default=HERE/'IO_FREEZE_V1.json')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--source-preflight-only',action='store_true')
    a=p.parse_args()
    if a.source_preflight_only:
        source_preflight(a.freeze);print('R27_RAW_QUALITY_SOURCE_PREFLIGHT_PASS')
    else:run(a)
