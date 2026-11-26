#!/usr/bin/env python3
"""Bounded descriptive CPU audit after R12; no fitting, model, GPU or new members."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ARMS=('native','FF','CC','fine_s2b2')
PINS={
 'check_receipt':'9b6c98313f610abc53b374335f4e8fa336cb02e48ffd7641f95f13902e387894',
 'fit_receipt':'0f551a53d204e9c3f886b65a46a693547218868731a25871f05d2c7c269b9099',
 'raw_audit':'0a68e1ab58f39038b716242e1a958a5beda2fe182556151bb596072d4f706965',
 'auditor':'e5c51e9a8f4068f9d66069e4a2a783712657d46886967da98accd72231d85dff',
 'membership':'cfc84c52158213599717d1aaafb19c73506ff08edbf54831d2b839bfa9da75d5',
 'regions':'8d64b18d3c69adb6c2a1bc146f55c254a2f6bbf02e50366b069f1d1d57fa8d26'}
QUANTILES=(0.,.5,.9,.95,.99,1.)


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()


def distribution(values):
    values=np.asarray(values,dtype=np.float64).reshape(-1)
    require(np.isfinite(values).all(),'Nonfinite descriptive input')
    if not values.size:return {'count':0,'mean':None,'quantiles':{str(q):None for q in QUANTILES}}
    return {'count':int(values.size),'mean':float(values.mean()),
            'quantiles':dict(zip(map(str,QUANTILES),map(float,np.quantile(values,QUANTILES,method='linear'))))}


def gt_geometry(gt,valid,boundary_threshold,large_threshold):
    g=gt.astype(np.float64);mag=np.sqrt(np.square(g).sum(0))
    known=np.zeros_like(valid)
    known[1:-1,1:-1]=valid[1:-1,1:-1]&valid[:-2,1:-1]&valid[2:,1:-1]&valid[1:-1,:-2]&valid[1:-1,2:]
    jump=np.zeros(valid.shape,dtype=np.float64);center=g[:,1:-1,1:-1]
    for other in (g[:,:-2,1:-1],g[:,2:,1:-1],g[:,1:-1,:-2],g[:,1:-1,2:]):
        jump[1:-1,1:-1]=np.maximum(jump[1:-1,1:-1],np.sqrt(np.square(center-other).sum(0)))
    b=known&(jump>boundary_threshold);large=valid&(mag>large_threshold)
    masks={'all_valid':valid,'B':b,'L':large,'B_and_L':b&large,'B_only':b&~large,
           'L_only_B_known':large&known&~b,'neither_B_known':known&~b&~large,'B_unknown':valid&~known}
    require(np.array_equal(sum(masks[k].astype(np.int8) for k in ('B_and_L','B_only','L_only_B_known','neither_B_known','B_unknown')),valid.astype(np.int8)), 'GT atoms fail')
    return mag,masks


def gt_summary(gt,valid,thresholds):
    mag,masks=gt_geometry(gt,valid,*thresholds)
    vec=gt.astype(np.float64)[:,valid].mean(1);mean=float(mag[valid].mean())
    return {'valid_pixels':int(valid.sum()),'image_pixels':int(valid.size),'valid_fraction':float(valid.mean()),
            'magnitude_pixels':distribution(mag[valid]),'mean_vector_xy_pixels':vec.tolist(),
            'mean_vector_norm_divided_by_mean_magnitude':float(np.linalg.norm(vec)/mean) if mean else None,
            'GT_region_counts':{k:int(v.sum()) for k,v in masks.items()},
            'GT_large_fraction_among_valid':float(masks['L'].sum()/valid.sum())},mag,masks


def snapshot(directory,which,ids,kinds,accepted):
    path=directory/'receipt.json';require(sha(path)==PINS[which+'_receipt'],'Snapshot receipt differs')
    receipt=json.loads(path.read_text());arrays={t:{} for t in ids};proof={t:{} for t in ids}
    for item in receipt['all_files']:
        t,k=item['target_id'],item['kind'];require(t in arrays and k in kinds and k not in arrays[t],'Unexpected/duplicate input')
        p=directory/item['relative_path'];require(directory.resolve() in p.resolve().parents,'Input path escape')
        checksum=sha(p);require(checksum==item['sha256']==item['expected_sha256']==accepted[t][k],'Input source binding differs')
        require(p.stat().st_size==item['bytes'],'Input byte length differs')
        value=np.load(p,allow_pickle=False)
        shape,dtype={'gt':((2,480,640),np.float32),'mask':((480,640),np.bool_),
                     'events':((10,480,640),np.float32)}[k]
        require(value.shape==shape and value.dtype==dtype and np.isfinite(value).all(),'Array shape/dtype/finite')
        arrays[t][k]=value;proof[t][k]={'sha256':checksum,'bytes':item['bytes'],'relative_path':item['relative_path']}
    require(len(receipt['all_files'])==len(ids)*len(kinds) and all(set(v)==set(kinds) for v in arrays.values()),'Snapshot incomplete')
    return arrays,proof


def predict(directory,frame):
    path=directory/frame['prediction_file'];require(path.name==frame['prediction_file'],'Prediction path escape')
    require(sha(path)==frame['prediction_file_sha256'],'NPZ SHA differs')
    with np.load(path,allow_pickle=False) as payload:value=payload['native'].copy()
    require(value.shape==(2,480,640) and value.dtype==np.float32 and np.isfinite(value).all(),'Prediction shape/dtype')
    require(hashlib.sha256(value.tobytes()).hexdigest()==frame['arms']['native']['prediction_sha256'],'Prediction payload SHA')
    return value,{'file':frame['prediction_file'],'NPZ_sha256':sha(path),'payload_sha256':frame['arms']['native']['prediction_sha256']}


def prediction_description(pred,gt,mask):
    p,g=pred.astype(np.float64),gt.astype(np.float64)
    pm=np.sqrt(np.square(p).sum(0));gm=np.sqrt(np.square(g).sum(0))
    denom=pm*gm;ok=mask&(denom>0)
    cosine=(p*g).sum(0)[ok]/denom[ok]
    return {'magnitude_pixels':distribution(pm[mask]),'mean_vector_xy_pixels':p[:,mask].mean(1).tolist(),
        'mean_component_error_xy_pixels':(p-g)[:,mask].mean(1).tolist(),
        'mean_predicted_minus_GT_magnitude_pixels':float((pm-gm)[mask].mean()),
        'direction_cosine_nonzero_vector_pixels':int(ok.sum()),
        'mean_direction_cosine_nonzero_vectors':float(cosine.mean()) if ok.any() else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require(not args.output.exists(),'Fresh diagnostic output required')
    auditor_path=HERE.parent/'reviews/audit_round12_raw_gt.py'
    audit_path=auditor_path.with_name('ROUND12_INDEPENDENT_RAW_GT_AUDIT.json')
    require(sha(auditor_path)==PINS['auditor'] and sha(audit_path)==PINS['raw_audit'],'Independent metric evidence differs')
    spec=importlib.util.spec_from_file_location('_frozen_independent_metric_only',auditor_path)
    metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)
    audit=json.loads(audit_path.read_text());require(audit['status']=='PASS' and all(v==0 for v in audit['max_absolute_differences'].values()),'Exact independent audit required')
    member_path=HERE/'check_membership_v1.json';require(sha(member_path)==PINS['membership'],'Membership differs')
    members=json.loads(member_path.read_text());checkids,fitids=members['groups']['check'],members['groups']['fit']
    raw={}
    for arm in ARMS:
        p=HERE/('recover_'+arm+'_v1/results.json');require(sha(p)==audit['inputs'][arm]['sha256'],'Complete raw SHA differs')
        raw[arm]=json.loads(p.read_text());require(raw[arm]['status']=='ROUND12_FAIR_RECOVERY_COMPLETE','Incomplete recovery')
    native=raw['native'];fit_evidence=native['training_input_evidence']
    require(list(fit_evidence)==fitids and all(raw[a]['training_input_evidence']==fit_evidence for a in ARMS),'Four-arm training evidence differs')
    initial={f['target_id']:f for f in native['evaluations']['initial']['frames']}
    endpoint={f['target_id']:f for f in native['evaluations']['endpoint']['frames']}
    check_evidence={t:initial[t]['input_evidence'] for t in checkids}
    check,check_proof=snapshot(HERE/'metric_input_snapshot_v1','check',checkids,('gt','mask','events'),check_evidence)
    fit,fit_proof=snapshot(HERE/'fit_gt_snapshot_v1','fit',fitids,('gt','mask'),fit_evidence)
    region_path=HERE/'metric_input_snapshot_v1/regions_gatefit4573.json';require(sha(region_path)==PINS['regions'],'Region contract differs')
    regions=json.loads(region_path.read_text());thresholds=[regions['boundary_threshold_pixels'],regions['large_displacement_threshold_pixels']]
    timing={r['target_id']:r['timestamps'] for r in native['binding']['inherited_data_and_numerical_contract']['new_member_data_preflight']}
    check_rows=[];fit_rows=[];check_pool=[];fit_pool=[]
    for t in checkids:
        a=check[t];summary,mag,base_masks=gt_summary(a['gt'],a['mask'],thresholds)
        masks=metric.region_masks(a['gt'],a['mask'],a['events'],*thresholds)
        require(all(np.array_equal(base_masks[k],masks[k]) for k in base_masks),'Independent GT region mismatch')
        activity=np.any(a['events']!=0,axis=0);valid=a['mask'];count=int(valid.sum())
        row={'target_id':t,'sequence':t.rsplit('_',1)[0],'GT':summary,'source_arrays':check_proof[t],
            'timestamps':timing[t],'voxel_support':{
                'nonzero_bin_entries':int(np.count_nonzero(a['events'])), 'bin_entries':int(a['events'].size),
                'nonzero_bin_fraction':float(np.count_nonzero(a['events'])/a['events'].size),
                'pixels_with_any_nonzero_bin':int(activity.sum()),'any_nonzero_fraction_all_pixels':float(activity.mean()),
                'valid_pixels_with_any_nonzero_bin':int((valid&activity).sum()),
                'any_nonzero_fraction_within_valid_GT':float(activity[valid].mean()),
                'valid_pixels_with_zero_voxel':int((valid&~activity).sum()),
                'scope':'Encoded signed voxel any!=0 proxy; not raw-event density, no proof of absent model context'},
             'predictions':{},'regions':{}}
        metrics={}
        for name,frame in [('initial_native',initial[t]),('recovered_N_native',endpoint[t])]:
            pred,proof=predict(HERE/'recover_native_v1',frame)
            m=metric.frame_metrics(pred,a['gt'],masks)
            require(m==frame['arms']['native']['metrics']['regions'],'Description AEE differs from frozen raw')
            row['predictions'][name]={'evidence':proof,'valid_GT_support':prediction_description(pred,a['gt'],valid)}
            metrics[name]=m
        for region,mask in masks.items():
            n=int(mask.sum());before=metrics['initial_native'][region];after=metrics['recovered_N_native'][region]
            row['regions'][region]={'valid_pixels':n,'fraction_of_all_valid':n/count,
                'GT_magnitude_pixels':distribution(mag[mask]),'initial_native':before,'recovered_N_native':after,
                'delta_AEE':after['AEE']-before['AEE'] if n else None,
                'delta_error_sum':after['error_sum']-before['error_sum']}
        check_rows.append(row);check_pool.append(mag[valid])
    for t in fitids:
        a=fit[t];summary,mag,_=gt_summary(a['gt'],a['mask'],thresholds)
        counts={arm:[s['valid_GT_pixels'] for s in raw[arm]['steps'] if s['target_id']==t] for arm in ARMS}
        require(all(len(v)==10 and set(v)=={summary['valid_pixels']} for v in counts.values()),'Training valid support/visit count differs')
        fit_rows.append({'target_id':t,'sequence':t.rsplit('_',1)[0],'GT':summary,'source_arrays':fit_proof[t],
            'training_array_SHA_all_four_arms':fit_evidence[t],'training_visits_each_arm':10,
            'valid_count_matches_all_40_training_visits':True,'timestamps':timing[t],
            'fit_events_downloaded':False,'voxel_support':None,'fit_predictions_available_in_this_snapshot':False})
        fit_pool.append(mag[a['mask']])
    delta=[r['regions']['all_valid']['delta_AEE'] for r in check_rows];net=sum(delta)
    initial_sum=sum(r['regions']['all_valid']['initial_native']['AEE'] for r in check_rows)
    for row,d in zip(check_rows,delta):
        row['signed_contribution_to_sum_frame_AEE_change']=d/net if net else None
        row['share_of_initial_sum_frame_AEE']=row['regions']['all_valid']['initial_native']['AEE']/initial_sum
    compare={}
    for key,selector in [('GT_mean',lambda r:r['GT']['magnitude_pixels']['mean']),
                         ('GT_median',lambda r:r['GT']['magnitude_pixels']['quantiles']['0.5']),
                         ('GT_p95',lambda r:r['GT']['magnitude_pixels']['quantiles']['0.95']),
                         ('GT_p99',lambda r:r['GT']['magnitude_pixels']['quantiles']['0.99']),
                         ('GT_max',lambda r:r['GT']['magnitude_pixels']['quantiles']['1.0'])]:
        values=[selector(r) for r in fit_rows]
        compare[key]={'fit20_frame_min':min(values),'fit20_frame_max':max(values),
            'check8_per_frame':{r['target_id']:selector(r) for r in check_rows}}
    result={'schema':'flow_round12_posthoc_input_diagnostic_v1','status':'COMPLETE_DESCRIPTIVE_ONLY',
        'scope':'Posthoc on fixed all8 check and original fit20. N initial native versus N200 native; not causal, not new validation, no fitting/training/quality gate change.',
        'script_sha256':sha(__file__),'numpy':np.__version__,'pins':PINS,
        'complete_result_SHA':{a:audit['inputs'][a]['sha256'] for a in ARMS},
        'member_order_check8':checkids,'member_order_fit20':fitids,
        'definitions':{'GT_support':'all valid original GT; pixel units unchanged; all8 retained',
            'quantiles':list(QUANTILES),'quantile_method':'NumPy linear, FP64 magnitudes',
            'thresholds':dict(zip(('boundary_pixels','large_displacement_pixels'),thresholds)),
            'direction':'mean per-pixel cosine where both GT and prediction vector magnitudes >0; FP64; supplemental posthoc descriptor',
            'coherence':'norm(mean GT vector)/mean(GT magnitude), valid support only; descriptive, not learned score'},
        'check8':check_rows,'fit20':fit_rows,'fit20_vs_check8_same_support':compare,
        'pooled_valid_pixel_magnitude_descriptive':{'fit20':distribution(np.concatenate(fit_pool)),
            'check8':distribution(np.concatenate(check_pool)), 'scope':'Pixel-weighted descriptive distributions; spatial/temporal samples are not independent observations'},
        'native_frame_equal_summary':{'initial_AEE':initial_sum/8,'endpoint_AEE':(initial_sum+net)/8,
            'relative_change_percent':100*net/initial_sum,'sum_signed_frame_AEE_change':net,
            'wins':sum(d<0 for d in delta),'ties':sum(d==0 for d in delta),'losses':sum(d>0 for d in delta)},
        'all_sixteen_regions_recomputed_exact_to_frozen_raw':True,'fit_event_support_comparison_available':False,
        'limits':['No new member/threshold/parameter choice','Fit20 events and saved dense predictions not downloaded in fit snapshot',
                  'A distribution contrast does not identify coverage as a causal explanation','Voxel occupancy is not raw event information content',
                  'Native-only descriptive comparison does not establish compression or flow-specific gain']}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'output_sha256':sha(args.output),'check_frames':len(check_rows),'fit_frames':len(fit_rows),
        'native_frame_equal_summary':result['native_frame_equal_summary']},ensure_ascii=False))


if __name__=='__main__':main()
