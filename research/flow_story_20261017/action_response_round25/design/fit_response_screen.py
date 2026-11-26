#!/usr/bin/env python3
"""Fit16-only action-response/gain pilot; freeze dev decisions before dev quality."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch
import response_models as m

ROLE_SHA='e6f216f6b017fc2cbba262ec51c70474f342e593e1549058a2200970949e10a1'
GRAPH='r25_W271482_all78current_s2b2_fixedH1_two_slot_T2_fit16dev8'
G0_SHA='01d54124dfecec70cffa9efaa2a8e88579ecb14477ae891d46643b140f15216d'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def checked(spec):
    m.require(set(spec)=={'path','sha256'},'Pinned path/SHA')
    p=Path(spec['path']);m.require(p.is_file() and sha(p)==spec['sha256'],'Actual source mismatch '+str(p));return p

def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:serial(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [serial(x) for x in v]
    return v

def payload(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--inputs-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    m.require(not args.output.exists() and sha(args.inputs)==args.inputs_sha256,'Fresh output, exact inputs')
    inp=json.loads(args.inputs.read_text());m.require(inp['schema']=='flow_r25_fit_only_inputs_v1','Fit-only schema')
    for pin in inp['source_pins']:checked(pin)
    for source in (Path(__file__).resolve(),Path(m.__file__).resolve()):
        m.require(any(Path(s['path']).resolve()==source for s in inp['source_pins']),'Freeze actual source '+str(source))
    rolespath=checked(inp['roles']);m.require(sha(rolespath)==ROLE_SHA,'Original32 role bytes')
    roles=json.loads(rolespath.read_text())
    resultpath=checked(inp['execution_result']);execution=json.loads(resultpath.read_text());result_sha=sha(resultpath)
    auditpath=checked(inp['execution_audit']);audit=json.loads(auditpath.read_text())
    m.require(audit['schema']=='flow_r25_action_response_independent_execution_audit_v1' and
              audit['status']=='PASS' and audit['quality_analysis_allowed'] and audit['source_result_sha256']==result_sha,
              'Full independent collection audit')
    m.require(execution['schema']=='flow_r25_action_response_collection_v1' and execution['graph_id']==GRAPH and
              execution['status']=='ACTION_RESPONSE_COLLECTION362_COMPLETE' and execution['full_forward_count']==362 and
              execution['binding']['selector_roles_sha256']==ROLE_SHA and execution['binding']['graph_id']==GRAPH,
              'Complete actual response graph')
    rows=inp['members'];fit=[r for r in rows if r['role']=='fit'];dev=[r for r in rows if r['role']=='development']
    m.require(len(rows)==24 and len(fit)==16 and len(dev)==8 and rows==fit+dev,'Fixed fit16/dev8 order')
    for role,subset in (('fit',fit),('development',dev)):
        m.require([r['target_id'] for r in subset]==[r['target_id'] for r in roles['groups'][role]],'Canonical identities')
        for r,c in zip(subset,roles['groups'][role]):
            m.require(r['sequence']==c['sequence'],'Canonical sequence for LOO')
            expected={'target_id','sequence','role','native_archive'}|({'background_archive','action_archives'} if role=='fit' else set())
            m.require(set(r)==expected,'Strict allowed fields, no dev actions/GT/mask/gain')
    labels=json.loads(checked(inp['fit_labels']).read_text())
    m.require(labels['schema']=='flow_r25_fit_gain_labels_v1' and labels['graph_id']==GRAPH and labels['background']=='T2' and
              labels['source_result_sha256']==result_sha and labels['execution_audit_sha256']==sha(auditpath) and
              labels['tile_order']==[list(t) for t in m.TILES] and labels['normalization']=='max(native_allGT_AEE,1e-6)',
              'Own actual graph, exact label direction/normalization')
    m.require(set(labels['signed_relative_gain_by_id'])=={r['target_id'] for r in fit},'Fit-only gain labels')
    records=execution['forward_records'];endpoints={(r['target_id'],r['arm']):r for r in records if r['quality_endpoint']}
    m.require(len(records)==362 and len(endpoints)==360,'Unique complete actual endpoints')
    def endpoint(pathspec,tid,arm):
        path=checked(pathspec);r=endpoints[(tid,arm)]
        role=next(v['role'] for v in rows if v['target_id']==tid)
        m.require(path.name==r['prediction_file'] and sha(path)==r['prediction_file_sha256'] and
                  r['selector_role']==role==execution['binding']['selector_role_map'][tid], 'Actual endpoint role/archive')
        m.require(r['execution']==('native' if arm=='native' else 'compact'),'Actual native/compact source')
        return path,r
    def head(pathspec,tid,arm):
        path,r=endpoint(pathspec,tid,arm)
        with np.load(path,allow_pickle=False) as z:
            a=z['head3'];m.require(a.shape==(1,2,480,640) and a.dtype==np.float32 and np.isfinite(a).all() and
                                 payload(a)==r['head_records'][3]['sha256'],'Actual full head3 payload');return a[0].copy()
    gp=checked(inp['ordinary_score_source']);m.require(sha(gp)==G0_SHA,'Frozen original seven ordinary scores')
    spec=importlib.util.spec_from_file_location('_r25_ordinary',gp)
    g0=importlib.util.module_from_spec(spec);spec.loader.exec_module(g0)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    m.require(not torch.cuda.is_initialized(),'CPU learning only')
    begin=time.monotonic();prefix={};response={};ordinary={};source_evidence={}
    for row in rows:
        tid=row['target_id'];path,rec=endpoint(row['native_archive'],tid,'native')
        with np.load(path,allow_pickle=False) as z:
            u,sn=z['u'],z['sn1']
            m.require(u.shape==sn.shape==(10,1,30,40,384) and u.dtype==sn.dtype==np.float32 and
                      np.isfinite(u).all() and np.isfinite(sn).all(),'Actual finite prefix/spike tensor')
            m.require(payload(u)==rec['tensor_records']['u']['sha256'] and payload(sn)==rec['tensor_records']['sn1']['sha256'],'Predecision source hash')
            prefix[tid]=u[:,0].mean(0,dtype=np.float32);ordinary[tid]=g0.score_frame(u,sn)['scores']
        source_evidence[tid]={'native_archive_sha256':sha(path),'prefix_mean_sha256':payload(prefix[tid])}
        if row['role']=='fit':
            background=head(row['background_archive'],tid,'T2')
            m.require(len(row['action_archives'])==12,'Exactly12 actual action labels')
            response[tid]=np.stack([m.response_grid(head(pin,tid,'fine_%d_%d'%tile),background)
                                    for pin,tile in zip(row['action_archives'],m.TILES)])
            source_evidence[tid]['response_label_sha256']=payload(response[tid])
    fit_ids=[r['target_id'] for r in fit];dev_ids=[r['target_id'] for r in dev]
    sequence={r['target_id']:r['sequence'] for r in fit};sequences=sorted(set(sequence.values()))
    m.require(len(sequences)==8 and all(sum(s==v for v in sequence.values())==2 for s in sequences),'Fixed8x2 OOF')
    x={t:m.features(prefix[t][None]) for t in fit_ids+dev_ids}
    def train(ids):return m.fit_response(np.concatenate([x[t] for t in ids]),np.concatenate([response[t] for t in ids]))
    oof={};folds=[]
    for seq in sequences:
        train_ids=[t for t in fit_ids if sequence[t]!=seq];held=[t for t in fit_ids if sequence[t]==seq]
        m.require(len(train_ids)==14 and len(held)==2,'14-to2 same-sequence holdout')
        fitted=train(train_ids)
        for tid in held:oof[tid]=m.predict_response(fitted,x[tid])
        folds.append(dict(sequence=seq,train_ids=train_ids,held_ids=held,model=fitted))
    final=train(fit_ids);dev_response={t:m.predict_response(final,x[t]) for t in dev_ids}
    X=np.concatenate([x[t] for t in fit_ids]);D=np.concatenate([oof[t] for t in fit_ids])
    Y=np.asarray([labels['signed_relative_gain_by_id'][t] for t in fit_ids],np.float32)
    m.require(Y.shape==(16,12) and np.isfinite(Y).all(),'Signed fit gains')
    xd=np.concatenate([x[t] for t in dev_ids]);dd=np.concatenate([dev_response[t] for t in dev_ids])
    args.output.mkdir(parents=True,exist_ok=False);scores={};models={};shared=[]
    for kind in ('direct','generic_response','structured_response'):
        fitted=m.fit_head(kind,X,D,Y.reshape(-1));pred=m.infer_head(fitted,xd,dd).reshape(8,12)
        scores[kind]={t:pred[i] for i,t in enumerate(dev_ids)}
        path=args.output/(kind+'_state.pt');torch.save(fitted['model'].state_dict(),path)
        models[kind]={k:v for k,v in fitted.items() if k!='model'}
        models[kind].update(state_file=path.name,state_file_sha256=sha(path),state_payload_sha256=m.state_sha(fitted['model']))
        shared.append((fitted['x_scaler'],fitted['d_scaler']))
        print(json.dumps({'head':kind,'steps':fitted['steps'],'parameters':fitted['parameter_count'],'fit_loss':fitted['loss_record']}),flush=True)
    for a,b in zip(shared,shared[1:]):
        m.require(all(np.array_equal(a[i][k],b[i][k]) for i in (0,1) for k in ('mean','std')),'Identical shared input and response scales')
    for name in g0.SCORES:scores['G0_'+name]={t:np.asarray([r[name] for r in ordinary[t]]) for t in dev_ids}
    scores['H1_feature_residual']={t:np.square(x[t][:,768:1152]).mean(-1) for t in dev_ids}
    scores['predicted_response_magnitude']={t:np.linalg.norm(dev_response[t],axis=-1).mean((1,2)) for t in dev_ids}
    scores['fit_position_prior']={t:Y.mean(0) for t in dev_ids}
    decisions={name:{tid:int(np.argmax(v)) for tid,v in values.items()} for name,values in scores.items()}
    response_files=[]
    for role,ids,values in (('fit',fit_ids,oof),('development',dev_ids,dev_response)):
        for tid in ids:
            path=args.output/(tid+'_predicted_response.npz');np.savez_compressed(path,delta=values[tid])
            response_files.append(dict(target_id=tid,role=role,file=path.name,sha256=sha(path),payload_sha256=payload(values[tid])))
    out=dict(schema='flow_r25_frozen_development_predictions_v1',status='FROZEN_BEFORE_DEVELOPMENT_QUALITY',
             graph_id=GRAPH,background='T2',inputs_sha256=args.inputs_sha256,source_result_sha256=result_sha,
             execution_audit_sha256=sha(auditpath),fit_ids=fit_ids,development_ids=dev_ids,response_folds=folds,
             final_response_model=final,head_training=models,scores=scores,decisions=decisions,
             response_files=response_files,source_evidence=source_evidence,exact_tie_rule='first row-major maximum',
             CUDA_initialized=torch.cuda.is_initialized(),wall_seconds=time.monotonic()-begin,
             development_action_outputs_loaded=False,development_GT_paths_present=False,
             scope='Development information screening at equal k, not equal net cycles; no quality or hardware advantage asserted')
    path=args.output/'predictions.json';path.write_text(json.dumps(serial(out),indent=2,allow_nan=False)+'\n')
    print('FROZEN '+sha(path),flush=True)

if __name__=='__main__':main()
