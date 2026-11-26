#!/usr/bin/env python3
"""CPU/metadata-only score stability, never reads event/GT/check/valid arrays."""
import argparse
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
FIT2_ARTIFACT_SHA='15da3e3de35e402ddd01efae1f6f80f36ef2a5223f3652141acdb1ff78071982'
FIT20_MANIFEST_SHA='29ec7ba9c9faca74c62c3e170c9d3c70855a4605a7eab31f9331c519f56f8915'
CHECKPOINT_SHA='d5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30'


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()


def average_ranks(values):
    """Ascending midranks; ties are not arbitrarily broken for Spearman."""
    order=sorted(range(len(values)),key=lambda i:(values[i],i)); ranks=[0.]*len(values); start=0
    while start<len(order):
        end=start+1
        while end<len(order) and values[order[end]]==values[order[start]]: end+=1
        value=(start+end-1)/2.+1
        for j in range(start,end): ranks[order[j]]=value
        start=end
    return ranks


def correlation(a,b):
    ma,mb=sum(a)/len(a),sum(b)/len(b)
    xa=[x-ma for x in a];xb=[x-mb for x in b]
    da=sum(x*x for x in xa);db=sum(x*x for x in xb)
    return None if da==0 or db==0 else sum(x*y for x,y in zip(xa,xb))/math.sqrt(da*db)


def block_scores(artifact,path,ids):
    b=artifact['blocks'][path];n=len(ids);width=b['hidden_width'];scores=b['scores'];ranks=b['rank_sums']
    require(b['schema']=='flow_svs_score_v1' and b['sample_ids']==ids and b['sample_count']==n and
            b['batch_sizes']==[1]*n and len(scores)==len(ranks)==width and width>0,'SVS sample accounting differs')
    require(all(type(r) is int and r>=0 for r in ranks) and all(isinstance(v,(int,float)) and math.isfinite(v) for v in scores) and
            all(abs(v-r/n)<=1e-12 for v,r in zip(scores,ranks)),'Scores do not equal sample-weighted ranks')
    return scores


def compare(reference,candidate,fit_manifest,group=16):
    require(type(group) is int and group>0,'Positive group required')
    require(reference['schema']==candidate['schema']=='flow_breadth_svs_replay_v1','Foreign replay schema')
    old_ids=reference['fit_ids'];ids=candidate['fit_ids']
    require(len(old_ids)==2 and old_ids==fit_manifest['nested_fit2_target_ids'] and
            ids==fit_manifest['target_ids'] and len(ids)==20 and set(old_ids)<=set(ids),'Fit2/20 nesting or order differs')
    for obj in (reference,candidate):
        p=obj['provenance']
        require(p['checkpoint_sha256']==CHECKPOINT_SHA and p['split']=='train' and
                p['fit_ids_sha256']==digest(obj['fit_ids']),'Checkpoint or train member binding differs')
        require(obj['evidence']['sample_count']==len(obj['fit_ids']) and obj['evidence']['CPU_only_SVD'] is True and
                obj['evidence']['cuda_initialized'] is False,'Foreign scoring evidence')
    p,q=reference['provenance'],candidate['provenance']
    require(set(p)==set(q) and all(p[k]==q[k] for k in p if k!='fit_ids_sha256'),
            'Checkpoint/data/source/BN/geometry changed; not a sampling-only comparison')
    paths=sorted(reference['blocks'])
    require(paths==sorted(candidate['blocks']) and len(paths)==12,'Native MLP coverage differs')
    out={'schema':'flow_recovery_svs_stability_v1','population':'train-fit20 contains original train-fit2',
         'sampling_only_comparison':True,'formula_unchanged':True,'pe_group_width':group,'blocks':{},
         'comparison_definitions':{'Pearson':'centered raw SVS scores across hidden channels',
            'Spearman':'Pearson of ascending average ranks; constant-vector correlation is null',
            'selection':'keep floor(F*ratio/group)*group highest scores; stable original-index tiebreak; physical order ascending',
            'kept_overlap_fraction':'size(keep2 intersect keep20)/keep_width',
            'Jaccard':'size(intersection)/size(union)'},
         'AEE_or_hardware_measurement':False,'check64_or_final_valid_read':False,'GT_or_event_array_read':False}
    totals={str(r):{'keep_width_sum':0,'intersection_sum':0,'union_sum':0} for r in (.75,.5,.25)}
    for path in paths:
        b,c=reference['blocks'][path],candidate['blocks'][path]
        require(b['formula']==c['formula']==reference['evidence']['formula']==candidate['evidence']['formula'],
                'SVS formula/site/epsilon differs')
        x,y=block_scores(reference,path,old_ids),block_scores(candidate,path,ids)
        require(len(x)==len(y),'Hidden width changed')
        row={'hidden_width':len(x),'score_max_abs_difference':max(abs(a-b) for a,b in zip(x,y)),
             'score_mean_abs_difference':sum(abs(a-b) for a,b in zip(x,y))/len(x),
             'score_pearson':correlation(x,y),'score_spearman_midranks':correlation(average_ranks(x),average_ranks(y)),
             'unique_scores_fit2':len(set(x)),'unique_scores_fit20':len(set(y)),'selections':{}}
        for ratio in (.75,.5,.25):
            k=math.floor(len(x)*ratio/group)*group;require(k>0,'Requested group leaves no channels')
            ix=sorted(sorted(range(len(x)),key=lambda i:(-x[i],i))[:k]);iy=sorted(sorted(range(len(y)),key=lambda i:(-y[i],i))[:k])
            inter=len(set(ix)&set(iy));union=len(set(ix)|set(iy))
            row['selections'][str(ratio)]={'keep_width':k,'fit2_keep_indices':ix,'fit20_keep_indices':iy,
                'intersection':inter,'union':union,'kept_overlap_fraction':inter/k,'jaccard':inter/union,
                'mask_changed_channels':2*(k-inter),'fit2_mask_sha256':digest(ix),'fit20_mask_sha256':digest(iy)}
            totals[str(ratio)]['keep_width_sum']+=k;totals[str(ratio)]['intersection_sum']+=inter;totals[str(ratio)]['union_sum']+=union
        out['blocks'][path]=row
    for value in totals.values():
        value['channel_micro_kept_overlap']=value['intersection_sum']/value['keep_width_sum']
        value['channel_micro_jaccard']=value['intersection_sum']/value['union_sum']
    out['aggregate_channel_overlap_not_PE_cost']=totals
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference2',type=Path,required=True)
    p.add_argument('--candidate20',type=Path,required=True);p.add_argument('--candidate-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--pe-group-width',type=int,default=16)
    a=p.parse_args();manifest=HERE/'manifests/svs_fit20.json'
    require(sha(a.reference2)==FIT2_ARTIFACT_SHA and sha(a.candidate20)==a.candidate_sha256 and
            sha(manifest)==FIT20_MANIFEST_SHA,'Artifact or fit manifest SHA differs')
    old=json.loads(a.reference2.read_text());new=json.loads(a.candidate20.read_text())
    require(new['evidence'].get('collector_sha256')==sha(HERE/'collect_svs20.py') and
            new['evidence'].get('fit_manifest_sha256')==FIT20_MANIFEST_SHA,'Candidate20 collector/fit binding differs')
    result=compare(old,new,json.loads(manifest.read_text()),a.pe_group_width)
    result['source_sha256']={str(a.reference2):sha(a.reference2),str(a.candidate20):sha(a.candidate20),
                           'svs_fit20.json':sha(manifest),'comparison_script':sha(__file__)}
    require(not a.output.exists(),'Stability report is immutable')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'STABILITY_REPORTED','output':str(a.output),'AEE_executed':False}))
