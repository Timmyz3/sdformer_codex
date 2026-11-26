#!/usr/bin/env python3
"""Independent local result audit; no imports from R11 implementation or analysis."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

POLS=('B5_task','B5_MSE','rich37_task','rich37_MSE','PCA11_task','PCA11_MSE','flow11_task','flow11_MSE','B5_small4_task','B5_small4_MSE','fit_position_mean','fixed_uniform_index3','amplitude_abs_mean','pooling_residual_mse','psn1_activity','psn1_spatial_neighbor_hamming')
BASIC=('amplitude_abs_mean','feature_rms','pooling_residual_mse','spatial_neighbor_abs_difference','temporal_neighbor_abs_difference')
TILES=[(r,c) for r in range(3) for c in range(4)]
UNIFORM=[3,8,1,10,5,6,0,11,2,9,4,7]
SEQ_CONTROLS=('B5_task','PCA11_task','psn1_activity','psn1_spatial_neighbor_hamming','flow11_MSE')
ANCHORS=('native','FF','CC','fine_s2b2')
REGIONS=('all_valid','B','L','B_and_L','B_only','L_only_B_known','neither_B_known','B_unknown','voxel_nonzero','voxel_zero','B_voxel_nonzero','B_voxel_zero','L_voxel_nonzero','L_voxel_zero','B_unknown_voxel_nonzero','B_unknown_voxel_zero')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4194304),b''):h.update(chunk)
    return h.hexdigest()

def require(ok,msg):
    if not ok:raise AssertionError(msg)

def proj(m,x):
    std=np.array(m['std'],dtype=np.float64);beta=np.array(m['beta'],dtype=np.float64)
    weight=beta/(std[:,None] if beta.ndim==2 else std)
    bias=np.array(m['intercept'],dtype=np.float64)-np.array(m['mean'],dtype=np.float64)@weight
    return np.array(x,dtype=np.float32)@weight.astype(np.float32)+np.asarray(bias,dtype=np.float32)

def vector(v):
    v=np.asarray(v,dtype=np.float32)
    mag=np.sqrt(np.sum(v*v,axis=-1))
    edge=np.concatenate([(v[1:]-v[:-1]).reshape(-1,2),(v[:,1:]-v[:,:-1]).reshape(-1,2)])
    variation=np.sqrt(np.sum(edge*edge,axis=-1));avg=v.mean((0,1))
    return np.array([mag.mean(),mag.max(),mag.std(),variation.mean(),variation.max(),np.sqrt(np.sum(avg*avg))/(mag.mean()+np.float32(1e-6))],dtype=np.float32)

def features(cells,ordinary,heads):
    flow=proj(heads['flow'],cells);pca=proj(heads['PCA'],cells)
    by={tuple(x['tile']):x for x in ordinary};b=np.array([[by[t][k] for k in BASIC] for t in TILES],dtype=np.float32)
    rich=[];fs=[];ps=[]
    for r,c in TILES:
        patch=cells[r*3:r*3+3,c*3:c*3+3]
        grouped=patch.reshape(3,3,16,24).mean(-1)
        edges=np.concatenate([np.abs(grouped[1:]-grouped[:-1]).reshape(-1,16),np.abs(grouped[:,1:]-grouped[:,:-1]).reshape(-1,16)])
        rich.append(np.concatenate([grouped.mean((0,1)),edges.mean(0)]))
        fs.append(vector(flow[r*3:r*3+3,c*3:c*3+3]));ps.append(vector(pca[r*3:r*3+3,c*3:c*3+3]))
    return {'B5':b,'rich37':np.concatenate([b,np.array(rich)],axis=1),'flow11':np.concatenate([b,np.array(fs)],axis=1),'PCA11':np.concatenate([b,np.array(ps)],axis=1)},flow,pca

def ranks(scores):return sorted(range(12),key=lambda i:(-float(scores[i]),i))
def allclose(x,y):return np.allclose(np.asarray(x),np.asarray(y),rtol=5e-5,atol=1e-4)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require(not args.output.exists(),'Refuse output overwrite')
    base=args.raw.parent;work=Path(__file__).resolve().parent.parent/'joint_budget_round11'
    raw=json.loads(args.raw.read_text());analysis=json.loads(args.analysis.read_text());model=json.loads((work/'inference_models_v1.json').read_text())
    old=json.loads((work.parent.parent/'flow_story_20261015/block2_allocation_round10/analysis_v1/predictions_before_check_evaluation.json').read_text())
    for key in ('final_heads','selector_models','position_mean_gain'):require(model[key]==old[key],'Frozen model changed '+key)
    require(raw['status']=='ROUND11_FROZEN_JOINT_BUDGET_COMPLETE','Incomplete raw')
    require(analysis['input_sha256']==sha(args.raw),'Analysis binds wrong raw')
    require(raw['completed_targets']==8 and all(raw[k] for k in ('original_model_unchanged','BN_buffers_unchanged','original_modules_restored','all_hooks_removed')),'State closure')
    members=json.loads((work/'membership_v1.json').read_text())['groups']['check']
    require([f['target_id'] for f in raw['frames']]==members,'Ordered members')
    names=list(ANCHORS)+[p+'_k'+str(k) for k in (2,4) for p in POLS]
    npz_files=0;decision_files=0;predictions=0;replayed_rankings=0;max_score_error=0.;max_feature_error=0.;alias_count=0;actual_ids=[]
    mask_matches={str(k):{p:0 for p in POLS if p!='flow11_task'} for k in (2,4)}
    for frame in raw['frames']:
        tid=frame['target_id'];require(set(frame['arms'])==set(names),'Complete named arms '+tid)
        decision_path=base/frame['decision_file'];require(sha(decision_path)==frame['decision_file_sha256'],'Decision file SHA')
        d=json.loads(decision_path.read_text());decision_files+=1
        require(d['GT_used'] is False and d['fitting_performed'] is False,'NoGT contract')
        require(d['target_id']==tid and d['decisions']==frame['decisions'],'Decision identity/content')
        require(frame['decision_archived_before_GT_load'] is True,'Archive order receipt')
        require(d['compact_models_sha256']==sha(work/'inference_models_v1.json'),'Compact hash')
        input_path=base/frame['predecision_input_file'];require(sha(input_path)==frame['predecision_input_file_sha256'],'InputNPZ SHA');npz_files+=1
        with np.load(input_path,allow_pickle=False) as z:
            require(set(z.files)=={'predecision_u','no_GT_cell_descriptor','candidate_tiles'},'Only noGT input cache')
            u=z['predecision_u'];cells=z['no_GT_cell_descriptor'][0]
            require(str(u.dtype)=='float32' and np.isfinite(u).all(),'u dtype/finite')
            require(hashlib.sha256(u.tobytes()).hexdigest()==frame['predecision_u_sha256'],'u SHA')
            require(cells.shape==(10,14,384) and str(cells.dtype)=='float32' and np.isfinite(cells).all(),'cells shape/dtype')
            require(np.array_equal(z['candidate_tiles'],np.array(TILES)),'Full tiles')
        ordinary=frame['ordinary_scores'];require(ordinary==d['ordinary_scores'],'Ordinary scores')
        fts,flow,pca=features(cells,ordinary,model['final_heads'])
        for field,reconstructed in [('flow_projection',flow),('PCA_projection',pca)]:require(allclose(reconstructed,d[field]),field+' mismatch')
        for fam,value in fts.items():
            require(allclose(value,d['features'][fam]),'Feature mismatch '+fam)
            max_feature_error=max(max_feature_error,float(np.max(np.abs(value-np.array(d['features'][fam])))))
        scores={}
        for p,m in model['selector_models'].items():scores[p]=proj(m,fts['B5' if p.startswith('B5_small4_') else p.rsplit('_',1)[0]])
        scores['fit_position_mean']=np.array(model['position_mean_gain'])
        scores['fixed_uniform_index3']=np.array([12-UNIFORM.index(i) for i in range(12)])
        ordinary_map={tuple(x['tile']):x for x in ordinary}
        for p in POLS[12:]:scores[p]=np.array([ordinary_map[t][p] for t in TILES])
        seen={}
        for p in POLS:
            stored=d['decisions'][p];order=ranks(scores[p]);require(order==stored['ranking_indices'],'Ranking differs '+tid+' '+p)
            require([list(TILES[i]) for i in order]==stored['ranking_tiles'],'Rank tile/index')
            require(allclose(scores[p],stored['scores']),'Score mismatch '+p)
            max_score_error=max(max_score_error,float(np.max(np.abs(scores[p]-np.array(stored['scores'])))));replayed_rankings+=1
            for k in (2,4):
                arm=p+'_k'+str(k);selected=[list(TILES[i]) for i in sorted(order[:k])]
                require(stored['selected'][str(k)]==selected==frame['arms'][arm]['selected_tiles'],'Selected joint mask')
                key=tuple(tuple(t) for t in selected);expected_alias=seen.get(key)
                require(frame['arms'][arm]['alias_of']==expected_alias,'Alias first actual arm')
                if expected_alias is None:seen[key]=arm
                else:
                    alias_count+=1
                    require(frame['arms'][arm]['actual_forward_id']==frame['arms'][expected_alias]['actual_forward_id'],'Alias forward id')
                    require(frame['arms'][arm]['prediction_sha256']==frame['arms'][expected_alias]['prediction_sha256'],'Alias prediction')
                if p!='flow11_task' and selected==frame['arms']['flow11_task_k'+str(k)]['selected_tiles']:mask_matches[str(k)][p]+=1
        predpath=base/frame['prediction_file'];require(sha(predpath)==frame['prediction_file_sha256'],'Prediction NPZ SHA');npz_files+=1
        with np.load(predpath,allow_pickle=False) as z:
            require(set(z.files)==set(names),'All36 prediction arrays')
            for name in names:
                value=z[name];require(value.shape==(2,480,640) and str(value.dtype)=='float32' and np.isfinite(value).all(),'Prediction shape/dtype')
                require(hashlib.sha256(value.tobytes()).hexdigest()==frame['arms'][name]['prediction_sha256'],'Prediction array SHA')
                predictions+=1
        for name in names:
            entry=frame['arms'][name];regions=entry['metrics']['regions'];require(set(regions)==set(REGIONS),'Regions')
            if entry['alias_of'] is None:actual_ids.append(entry['actual_forward_id'])
            for region,v in regions.items():
                require(v['valid_pixels']==frame['arms']['native']['metrics']['regions'][region]['valid_pixels'],'Paired supports')
                require(v['valid_pixels']>=0 and v['error_sum']>=0 and np.isfinite(v['error_sum']),'Count/sum')
                if v['valid_pixels']:require(np.isclose(v['error_sum']/v['valid_pixels'],v['AEE'],rtol=1e-10,atol=1e-10),'Metric reduction')
                else:require(v['AEE'] is None and v['error_sum']==0,'Empty metric')
            atoms=['B_and_L','B_only','L_only_B_known','neither_B_known','B_unknown']
            require(sum(regions[x]['valid_pixels'] for x in atoms)==regions['all_valid']['valid_pixels'],'Partition count closure')
            require(np.isclose(sum(regions[x]['error_sum'] for x in atoms),regions['all_valid']['error_sum'],rtol=1e-10,atol=1e-10),'Partition error closure')
    require(len(actual_ids)==len(set(actual_ids))==raw['main_actual_forward_count'],'Unique mainforward IDs')
    require(alias_count==raw['alias_logical_arm_count'] and len(actual_ids)+alias_count==288,'Logical count')
    require(raw['full_forward_count']==len(actual_ids)+5==len(raw['forward_records']),'Exact5audits count')
    scopes={'overall':raw['frames']}
    scopes.update({seq:[f for f in raw['frames'] if f['sequence']==seq] for seq in sorted({f['sequence'] for f in raw['frames']})})
    means={};max_summary_error=0.
    for name in names:
        means[name]={}
        for scope,frames in scopes.items():
            means[name][scope]={}
            for region in REGIONS:
                values=[f['arms'][name]['metrics']['regions'][region] for f in frames]
                n=sum(v['valid_pixels'] for v in values);err=sum(v['error_sum'] for v in values);nonempty=[v['AEE'] for v in values if v['valid_pixels']]
                calc={'frames_with_region':len(nonempty),'valid_pixels':n,'error_sum':err,'frame_mean_AEE':sum(nonempty)/len(nonempty) if nonempty else None,'pixel_weighted_AEE':err/n if n else None}
                means[name][scope][region]=calc['frame_mean_AEE'];saved=analysis['summaries'][name]['scopes'][scope][region]
                for key,value in calc.items():
                    if value is None:require(saved[key] is None,'Empty summary')
                    else:
                        diff=abs(value-saved[key]);max_summary_error=max(max_summary_error,diff)
                        require(np.isclose(value,saved[key],rtol=1e-12,atol=1e-12),'Summary differs '+name+' '+scope+' '+region+' '+key)
    gates={}
    for k in (2,4):
        cname='flow11_task_k'+str(k);others=[p for p in POLS if p!='flow11_task'];best=min(others,key=lambda p:means[p+'_k'+str(k)]['overall']['all_valid'])
        candidate=means[cname];base=means[best+'_k'+str(k)];v=candidate['overall']['all_valid'];b=base['overall']['all_valid']
        def guarded(value,ref,mult):
            if value is None or ref is None:require(value is None and ref is None,'Support mismatch');return True
            return value<=ref*mult
        native=all(guarded(candidate['overall'][r],means['native']['overall'][r],m) for r,m in [('all_valid',1.03),('B',1.05),('L',1.05)])
        strict=all(v<means[p+'_k'+str(k)]['overall']['all_valid'] for p in others)
        cc=v<means['CC']['overall']['all_valid']
        seq=all(candidate[s]['all_valid']<=means[p+'_k'+str(k)][s]['all_valid'] for s in scopes if s!='overall' for p in SEQ_CONTROLS)
        regions=all(guarded(candidate[s][r],means['CC'][s][r],1.01) for s in scopes for r in ('B','L'))
        pilot=native and strict and cc and seq and regions
        margin=(b-v)/b if b>0 else None;bestreg=all(guarded(candidate['overall'][r],base['overall'][r],1.) for r in ('B','L'))
        main=pilot and margin is not None and margin>=.01 and bestreg
        got=analysis['per_budget_gates'][str(k)];require(pilot==got['pilot_gate_passed'] and main==got['main_quality_increment_gate_passed'],'Fixed gate differs')
        require(best==got['best_comparator_by_overall_allGT'],'Best comparator differs')
        gates[str(k)]={'native_overall':native,'strict15':strict,'beats_CC':cc,'sequence_nonreverse':seq,'CC_regions':regions,'best_comparator':best,'margin_fraction':margin,'best_regions':bestreg,'pilot':pilot,'main':main}
    report={'schema':'round11_independent_local_review_audit_v1','raw_sha256':sha(args.raw),'analysis_sha256':sha(args.analysis),'script_sha256':sha(__file__),'npz_files':npz_files,'decision_files':decision_files,'final_prediction_arrays':predictions,'replayed_policy_rankings':replayed_rankings,'max_score_absolute_error':max_score_error,'max_feature_absolute_error':max_feature_error,'max_summary_absolute_error':max_summary_error,'logical_endpoints':288,'unique_main_forwards':len(actual_ids),'alias_arms':alias_count,'full_forwards':raw['full_forward_count'],'mask_matches_to_primary':mask_matches,'independent_gates':gates,'scope':'All noGT decisions/ranks and saved raw metric reductions independently recomputed. No import of implementation/analyzer; no original remote GT re-evaluation, new fit, fullmodel or GPU. Producer services independently audited separately.'}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
