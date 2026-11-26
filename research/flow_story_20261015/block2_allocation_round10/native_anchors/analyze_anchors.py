#!/usr/bin/env python3
"""Same-check diagnostic anchors, not a new selector or a reopened R10 gate."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--results-sha256',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    require(sha(args.results)==args.results_sha256,'Anchor result SHA differs')
    require(not args.output_dir.exists(),'No output overwrite')
    raw=json.loads(args.results.read_text())
    parent_path=HERE.parent/'collect_v1/results.json'
    analysis_path=HERE.parent/'analysis_v1/analysis.json'
    parent=json.loads(parent_path.read_text());analysis=json.loads(analysis_path.read_text())
    require(sha(parent_path)=='c31d18f5eca5d4343c02f347d0fbed0c9604120dfb874524f8caf6404b1b005d','Frozen parent differs')
    require(raw['status']=='ROUND10_SAME_CHECK_NATIVE_ANCHORS_COMPLETE' and raw['full_forward_count']==27 and raw['completed_targets']==8,'Incomplete anchors')
    require(all(raw[k] for k in ('original_model_unchanged','BN_buffers_unchanged','original_modules_restored')),'State closure differs')
    require(raw['original_state_sha256']==parent['original_state_sha256'],'Numerical state differs')
    require(raw['anchor_replays']['CC']['exact_archived_prediction'] and raw['anchor_replays']['CC']['all8_input_output_SHA_exact'],'CC archive differs')
    require(set(raw['dual_checks'])=={'FF','fine_s2b2'} and all(all(v[k] for k in ('all_flows_exact','all8_inputs_exact','all8_branches_exact')) for v in raw['dual_checks'].values()),'Dual differs')
    old=[f for f in parent['frames'] if f['group']=='check'];frames=raw['frames']
    require([f['target_id'] for f in frames]==[f['target_id'] for f in old],'Original ordered check8 differs')
    names=['native','FF','fine_s2b2']+list(analysis['check_summaries'])
    rows=[]
    for f,prior in zip(frames,old):
        tid=f['target_id'];require(f['input_evidence']==prior['input_evidence'] and f['timestamps']==prior['timestamps'],'Data provenance differs')
        path=args.results.parent/f['prediction_file'];require(sha(path)==f['prediction_file_sha256'],'Prediction file differs')
        with np.load(path,allow_pickle=False) as z:
            require(set(z.files)=={'native','FF','fine_s2b2'},'Anchor arrays differ')
            for name in z.files:
                require(np.isfinite(z[name]).all() and hashlib.sha256(z[name].tobytes()).hexdigest()==f['arms'][name]['prediction_sha256'],'Prediction payload differs')
        metrics={n:f['arms'][n]['metrics']['regions'] for n in ('native','FF','fine_s2b2')}
        for n,s in analysis['check_summaries'].items():
            metrics[n]=prior['arms'][s['selected_arms'][tid]]['metrics']['regions']
        regions=list(metrics['native'])
        require(all(set(m)==set(regions) for m in metrics.values()),'Region coverage differs')
        for region in regions:
            require(all(m[region]['valid_pixels']==metrics['native'][region]['valid_pixels'] for m in metrics.values()),'Region count differs')
        rows.append({'target_id':tid,'sequence':f['sequence'],'metrics':metrics})
    scopes={'overall':rows}
    scopes.update({s:[r for r in rows if r['sequence']==s] for s in sorted({r['sequence'] for r in rows})})
    summaries={}
    for scope,members in scopes.items():
        summaries[scope]={}
        for region in regions:
            valid=[r for r in members if r['metrics']['native'][region]['valid_pixels']]
            count=sum(r['metrics']['native'][region]['valid_pixels'] for r in valid)
            values={n:float(np.mean([r['metrics'][n][region]['AEE'] for r in valid])) if valid else None for n in names}
            weighted={n:sum(r['metrics'][n][region]['AEE']*r['metrics'][n][region]['valid_pixels'] for r in valid)/count if count else None for n in names}
            summaries[scope][region]={'frames_with_region':len(valid),'valid_pixels':count,'frame_mean_AEE':values,'pixel_weighted_AEE':weighted,
                'relative_native_percent':{n:100*(v/values['native']-1) if v is not None and values['native'] else None for n,v in values.items()}}
    influence=[]
    for r in rows:
        e={n:r['metrics'][n]['all_valid']['AEE'] for n in names}
        influence.append({'target_id':r['target_id'],'allGT_AEE':e,
            'FF_minus_native':e['FF']-e['native'],'CC_minus_FF':e['coarse_baseline']-e['FF'],
            'CC_minus_full_block2':e['coarse_baseline']-e['fine_s2b2'],
            'full_block2_minus_FF':e['fine_s2b2']-e['FF'],
            'posthoc_k1_best_minus_full_block2':e['posthoc_GT_best']-e['fine_s2b2']})
    gates={n:all(summaries['overall'][r]['relative_native_percent'][n] is not None and summaries['overall'][r]['relative_native_percent'][n]<=limit for r,limit in [('all_valid',3),('B',5),('L',5)]) for n in names}
    result={'schema':'flow_round10_same_check_anchor_analysis_v1','source_sha256':{str(p):sha(p) for p in (args.results,parent_path,analysis_path,Path(__file__))},
        'summaries':summaries,'per_frame':rows,'allGT_decomposition':influence,'overall_native_quality_gate':gates,
        'original_R10_advancement':analysis['advancement'],'original_failed_candidate_not_reopened':True,
        'scope':'Outcome-triggered diagnostic on original check8. Full-layer protection is higher cost than k1. Native and FF remain separate numerical graphs. No new validation, significance, causal identification of feature failure, or hardware gain.'}
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# R10 同成员 native/FF/整层保护锚点','',result['scope'],'','| 规则 | allGT | 对native | B 对native | L 对native | 总体质量门 |','|---|---:|---:|---:|---:|---|']
    for n in names:
        s=summaries['overall'];v=s['all_valid']['frame_mean_AEE'][n]
        rel=[s[r]['relative_native_percent'][n] for r in ('all_valid','B','L')]
        lines.append('| '+n+' | %.9f | '%v+' | '.join('%+.6f%%'%v if v is not None else 'empty' for v in rel)+' | '+str(gates[n])+' |')
    lines+=['','所有成员、所有分区、像素加权与分序列对照见 analysis.json。此表不改变 R10 主候选失败，不把整层成本与 k1 互换。']
    (args.output_dir/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'ANCHOR_ANALYSIS_COMPLETE','output':str(args.output_dir),'original_candidate_passed':analysis['advancement']['passed']}))

if __name__=='__main__':main()
