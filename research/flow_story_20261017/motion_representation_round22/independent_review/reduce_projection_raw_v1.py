#!/usr/bin/env python3
"""R22 original-array proxy quality only after all projection predictions are frozen."""
import argparse
from pathlib import Path
import numpy as np
import io_common as c
import analysis_core as core
import projection_audit_math as audit_math
import release_fit_inputs_v1 as release


def source_arrays(state,tid,*,fit_only=False):
    result={}
    wanted=('gt','mask') if fit_only else ('events','gt','mask')
    kinds={'events':((10,480,640),np.dtype('float32')),'gt':((2,480,640),np.dtype('float32')),'mask':((480,640),np.dtype('bool'))}
    for kind in wanted:
        p=state['paths'][tid][kind];c.require(c.sha(p)==state['evidence'][tid][kind],'Unchanged source bytes before loading')
        x=np.load(p,allow_pickle=False);shape,dtype=kinds[kind]
        c.require(x.shape==shape and x.dtype==dtype and np.isfinite(x).all(),'Finite original source shape/dtype '+kind)
        result[kind]=x
    c.require(result['mask'].any(),'Positive all-GT support')
    return result


def bind_projection(a,freeze,state):
    c.require(c.sha(a.predictions)==a.predictions_sha256,'Projection manifest frozen before dev GT')
    p=c.read(a.predictions);fit=c.read(a.fit_inputs)
    c.require(p['schema']=='flow_r22_motion_projection_predictions_v1' and p['status']=='FROZEN_BEFORE_DEVELOPMENT_QUALITY','Projection freeze status')
    c.require(fit['schema']=='flow_r22_motion_projection_fit_inputs_v1' and p['inputs_sha256']==c.sha(a.fit_inputs),'Exact actual fit input contract')
    c.require(p['source_result_sha256']==c.sha(state['result_path']) and p['execution_audit_sha256']==c.sha(state['audit_path']) and p['graph_id']==fit['graph_id']==c.GRAPH,'Same audited native sources')
    expected=release.release(state,'not-used-for-members')['members']
    c.require(fit['members']==expected,'Exact independent16fit/8dev release including canonical sequences')
    c.require(fit['roles']['sha256']==c.ROLE_SHA and p['fit_ids']==[v['target_id'] for v in expected[:16]] and p['development_ids']==[v['target_id'] for v in expected[16:]],'No role or identity changes')
    c.require(p['selector_training'] is False and p['original_model_training'] is False and p['CUDA_initialized'] is False and p['device']=='CPU','Projection-only CPU fitting')
    for pin in fit['source_pins']:
        c.require(c.sha(pin['path'])==pin['sha256'],'Actually fitted source bytes')
    for name in ('projection_fitter','projection_core'):
        ext=freeze['external_sources'][name];path=(c.REPO/ext['relative_path']).resolve()
        c.require(any(Path(v['path']).resolve()==path and v['sha256']==ext['sha256'] for v in fit['source_pins']),'Fixed projection math imported')
    c.require(len(p['predictions'])==24 and [v['target_id'] for v in p['predictions']]==[v['target_id'] for v in expected],'All24 prediction files, fixed order')
    values={}
    for row,record in zip(expected,p['predictions']):
        tid=row['target_id'];c.require(record['role']==row['role'],'Prediction role unchanged')
        name=Path(record['file']);c.require(len(name.parts)==1 and name.suffix=='.npz','Safe projection artifact')
        path=a.predictions.parent/name;c.require(c.sha(path)==record['sha256'],'Frozen complete proxy archive')
        with np.load(path,allow_pickle=False) as z:
            c.require(set(z.files)==set(core.METHODS)==set(record['payloads']),'All four prespecified proxies')
            values[tid]={}
            for method in core.METHODS:
                x=z[method];c.require(x.shape==(30,40,2) and x.dtype==np.float32 and np.isfinite(x).all() and c.payload(x)==record['payloads'][method],'Frozen finite original-grid proxy')
                values[tid][method]=x.copy()
        audit_math.close(values[tid]['q_pool_expand3'],core.pool3(values[tid]['q']))
    return p,values


def independent_projection_check(state,projection,values,prefix):
    rows=state['members'];fit_ids=[r['target_id'] for r in rows[:16]];dev_ids=[r['target_id'] for r in rows[16:]]
    seq={r['target_id']:r['sequence'] for r in rows};truth={};counts={}
    # Only fit GT loaded here; dev remains unopened during source/OOF checks.
    for tid in fit_ids:
        data=source_arrays(state,tid,fit_only=True);truth[tid],counts[tid]=audit_math.truth_grid(data['gt'],data['mask'])
    c.require(projection['fit_valid_pixels']=={t:int(counts[t].sum()) for t in fit_ids},'All fit support')
    c.require(len(projection['q_folds'])==8 and [f['sequence'] for f in projection['q_folds']]==sorted(set(seq[t] for t in fit_ids)),'Exactly8 prespecified sequence folds')
    def refit(ids):return audit_math.independent_ridge(np.stack([prefix[t] for t in ids]),np.stack([truth[t] for t in ids]),np.stack([counts[t] for t in ids]))
    def reference_priors(ids):return audit_math.priors(np.stack([truth[t] for t in ids]),np.stack([counts[t] for t in ids]))
    receipts=[]
    for fold in projection['q_folds']:
        train=[t for t in fit_ids if seq[t]!=fold['sequence']];held=[t for t in fit_ids if seq[t]==fold['sequence']]
        c.require(fold['train_ids']==train and fold['held_ids']==held and len(train)==14 and len(held)==2,'No same-sequence supervised q fitting')
        expected=refit(train);prior=reference_priors(train);checks=audit_math.verify_model(fold['projection'],expected)
        endpoints={}
        for tid in held:
            endpoints[tid]={'q_reported_model':audit_math.close(values[tid]['q'],audit_math.project(fold['projection'],prefix[tid])),
              'q_independent_refit':audit_math.close(values[tid]['q'],audit_math.project(expected,prefix[tid])),
              **{k:audit_math.close(values[tid][k],prior[k]) for k in ('global_vector','position_prior')}}
        receipts.append(dict(held_sequence=fold['sequence'],train_ids=train,held_ids=held,model_checks=checks,prediction_checks=endpoints))
    expected=refit(fit_ids);prior=reference_priors(fit_ids);checks=audit_math.verify_model(projection['final_q_projection'],expected)
    audit_math.close(projection['final_ordinary_priors']['global_vector'],prior['global_vector'][0,0]);audit_math.close(projection['final_ordinary_priors']['position'],prior['position_prior'])
    endpoints={}
    for tid in dev_ids:
        endpoints[tid]={'q_reported_model':audit_math.close(values[tid]['q'],audit_math.project(projection['final_q_projection'],prefix[tid])),
          'q_independent_refit':audit_math.close(values[tid]['q'],audit_math.project(expected,prefix[tid])),
          **{k:audit_math.close(values[tid][k],prior[k]) for k in ('global_vector','position_prior')}}
    return dict(status='PASS',independent_equation='sum1 probability weights; ridge regularizer10/fit_cell_count; output folded FP32',
      fold_checks=receipts,final_fit16=dict(train_ids=fit_ids,held_ids=dev_ids,model_checks=checks,prediction_checks=endpoints),
      fit_GT_identity_count=16,development_GT_values_read_during_projection_source_audit=False)


def reduce(a):
    freeze=c.preflight(a.freeze,a.freeze_sha256);state=c.source_admission(freeze,a.result_dir,a.input_root)
    projection,values=bind_projection(a,freeze,state)
    prefix={};native_predictions={}
    for row in state['members']:
        tid=row['target_id'];path,r=state['native'][tid]
        with np.load(path,allow_pickle=False) as z:
            u=z['u'];head=z['head3'];c.require(u.shape==(10,1,30,40,384) and u.dtype==np.float32 and np.isfinite(u).all() and c.payload(u)==r['tensor_records']['u']['sha256'],'Actual predecision native u')
            c.require(head.shape==(1,2,480,640) and head.dtype==np.float32 and np.isfinite(head).all() and c.payload(head)==r['head_records'][3]['sha256'],'Actual final native head')
            prefix[tid]=u[:,0].mean(0,dtype=np.float32);native_predictions[tid]=head[0].copy()
        evidence=projection['source_evidence'][tid]
        c.require(evidence['native_archive_sha256']==c.sha(path) and evidence['prefix_mean_payload_sha256']==c.payload(prefix[tid]),'Actual time mean prefix producer')
    independent=independent_projection_check(state,projection,values,prefix)
    metric=c.load(c.MATH,'r22_original_regions_and_native_metrics');regions=c.read(c.REGIONS)
    c.require(set(metric.REGIONS)==set(core.REGIONS),'All original16 regions')
    c.require(regions['fit_uses_final_valid'] is False and len(regions['fit_target_ids'])==200,'Fixed training-only regions')
    # Development GT is first loaded only after all source/projection checks above.
    frames=[]
    for i,row in enumerate(state['members']):
        tid=row['target_id'];data=source_arrays(state,tid)
        masks=metric.masks_from_original(data['gt'],data['mask'],data['events'],regions['boundary_threshold_pixels'],regions['large_displacement_threshold_pixels'])
        frames.append(dict(target_id=tid,sequence=row['sequence'],role='fit' if i<16 else 'development',
          native_head3=metric.measure(native_predictions[tid],data['gt'],masks),representations={method:core.measure(values[tid][method],data['gt'],masks) for method in core.METHODS}))
    return dict(schema='flow_r22_independent_motion_representation_quality_v1',status='COMPLETE_ORIGINAL_ARRAY_REPRESENTATION_REDUCTION',
      graph_id=c.GRAPH,source_result_sha256=c.sha(state['result_path']),execution_audit_sha256=c.sha(state['audit_path']),projection_predictions_sha256=a.predictions_sha256,
      fit_inputs_sha256=c.sha(a.fit_inputs),IO_freeze_sha256=a.freeze_sha256,source_input_arrays=72,native_quality_endpoints=24,replays_excluded=2,
      independent_projection_audit=independent,**core.analyze(frames))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--result-dir',type=Path);p.add_argument('--input-root',type=Path);p.add_argument('--predictions',type=Path);p.add_argument('--predictions-sha256');p.add_argument('--fit-inputs',type=Path);p.add_argument('--output',type=Path);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    if a.preflight_only:c.preflight(a.freeze,a.freeze_sha256);print('R22_QUALITY_SOURCE_PREFLIGHT_PASS')
    else:
        c.require(all((a.result_dir,a.input_root,a.predictions,a.predictions_sha256,a.fit_inputs,a.output)),'All fixed actual bindings');c.write(a.output,reduce(a));print('R22_REPRESENTATION_REDUCED '+c.sha(a.output))
