#!/usr/bin/env python3
"""R38 independent action-specific pixel/row/bit-plane golden; no GPU/GT."""
import argparse,collections,hashlib,importlib.util,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R38=P.parent;OLD=P.parents[2]/'flow_story_20261021'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def calls(high,low,cout):
    limit=max(24,min(65536,8*1024*1024//cout))//8*8;out=[]
    for scope,n,bits in [('high8',high,range(8)),('low4',low,range(4,8))]:
        for start in range(0,n,limit):
            count=min(limit,n-start);padding=max(24,(count+7)//8*8)-count
            for bit in bits:out.append(dict(kind='binary_weight_plane',bit=bit,rows=count,padding_rows=padding,
                scope=scope,coefficient=-128 if bit==7 else 1<<bit))
    return out

def golden(label,geometries):
    mode='Q4' if label=='background_Q4' else 'M'
    if label=='background_Q4':ids=[]
    elif label=='static360':ids=list(range(209))+list(range(302,453))
    else:
        action=int(label.split('_')[1]);assert 0<=action<6;ids=list(range(action*151,(action+1)*151))
    out={}
    for name,g in geometries.items():
        t,h,w,k,co=[g[x] for x in ['T','H','W','K','Cout']];yy,xx=np.indices((h,w))
        tail=(yy>=h//9*9)|(xx>=w//9*9);selected=np.zeros((h,w),bool)
        isconv='.patch_embed.' in name
        if isconv:
            for tid in ids:
                y,x=divmod(tid,35);selected[y*9:y*9+9,x*9:x*9+9]=True
        hi=np.broadcast_to(tail|selected,(t,1,h,w)).copy().ravel();rows=t*h*w;high=int(hi.sum());low=rows-high
        spatial_sha=None if mode=='Q4' else hashlib.sha256(selected.tobytes()).hexdigest()
        coords=[list(divmod(i,35)) for i in sorted(ids)] if isconv else []
        ks=calls(high,low,co);planes=8*high+4*low
        out[name]=dict(mode=mode,rows=rows,Cin=k,Cout=co,tail_rows=int(tail.sum())*t,
            complete_rows=int((~tail).sum())*t,restored_complete_rows=int(selected.sum())*t,
            high8_rows=high,low4_rows=low,high8_rows_sha256=hashlib.sha256(hi.tobytes()).hexdigest(),
            selected_spatial_mask_sha256=spatial_sha,selected_complete_tiles=coords,geometric_shape=[10,1,h,w],
            row_order='T,B,output_y,output_x',logical_row_plane_work=planes,kernel_calls=ks,
            actual_integer_kernel_calls=len(ks),explicit_padded_rows=sum(c['padding_rows'] for c in ks),
            accepted_bitplane128_beats=planes*math.ceil(k/128)*math.ceil(co/16))
    return out

def main():
    a=argparse.ArgumentParser();a.add_argument('actual',type=Path);args=a.parse_args()
    res=read(args.actual/'results.json');plan=read(R38/'ACTION_PLAN_V1.json');contract=read(OLD/'round35_static_neighbor_probe/ACTION_CONTRACT_V1.json')
    assert res['status']=='R38_COMPLETE' and res['source_freeze_sha256']==sha(R38/'SOURCE_FREEZE_V1.json')
    assert res['runner_sha256']==sha(R38/'collect.py') and res['action_plan_sha256']==sha(R38/'ACTION_PLAN_V1.json')
    assert res['membership_sha256']==sha(R38/'data/PROPOSED_SOURCE_MEMBERSHIP_V1.json')
    assert res['final_state_sha256']==contract['endpoint']['state_sha256'] and res['endpoint']==contract['endpoint']
    assert (res['model_forwards'],res['forward_attempts'],res['formal_model_forwards'],res['engineering_model_forwards'])==(73,73,71,2)
    assert res['completed_optimizer_steps']==0 and not res['training_performed'] and not res['GT_arrays_read']
    assert res['wall_seconds']<900 and plan['expected_formal_forwards']==71
    source=read(R38/'SOURCE_FREEZE_V1.json')
    for n,h in source['files'].items():assert sha(R38/n)==h,n
    expected_pairs={(p['target_id'],label) for p in plan['records'] for label in p['unique_actions']};assert len(expected_pairs)==71
    expected_labels={label for _,label in expected_pairs};assert set(res['first_work_by_action'])==expected_labels
    summaries={};representation={};work_sha={};all_frames=[]
    dynamic=read(OLD/'round33_predictability/hardware/results/SUMMARY_V1.json')['candidates']
    service={int(r['action']):4*int(r['dynamic_affine_BN_ticks']) for r in dynamic}
    for label,work in res['first_work_by_action'].items():
        gold=golden(label,contract['geometries']);assert set(work)==set(gold) and len(gold)==28
        count=0 if label=='background_Q4' else 360 if label=='static360' else 151
        total=0;targets={}
        for name,g in gold.items():
            w=work[name]
            for key,value in g.items():
                if key!='accepted_bitplane128_beats':assert w[key]==value,(label,name,key)
            for key in ['binary_verified','int32_accumulation','spatial_rows_retained','full_T_and_channels','canonical_scatter_before_BN','independent_domain_execution']:
                assert w[key] is True,(label,name,key)
            assert w['reference'] is False and w['cycle_claim'] is False
            assert w['scale_variants']==2 and w['scale_applications']==w['scale_applications_per_output']==1
            assert w['midpoint_correction']['rows']==w['midpoint_correction']['offset']==0
            r=w['representation'];assert r['variant']=='rne4' and r['q8_shape']==[g['Cout'],g['Cin']]
            assert r['high_scale']=='per_output_maxabs/127' and r['low_scale']=='per_output_maxabs/7/16'
            assert r['low_rule']=='independent_rne4_maxabs7_clamp_minus8_plus7' and r['rounding']=='ties_to_even'
            assert r['low_encoding']=='signed_q4_left_shift4' and r['low_offset_per_active_input']==0
            if name in representation:assert r==representation[name],(label,name,'same frozen representations across all actions')
            else:representation[name]=r
            if '.patch_embed.' in name:assert g['restored_complete_rows']==count*810 and g['tail_rows']==30900
            else:assert g['restored_complete_rows']==0
            total+=g['accepted_bitplane128_beats'];targets[name]={k:g[k] for k in ['rows','tail_rows','restored_complete_rows','high8_rows','low4_rows','accepted_bitplane128_beats']}
        assert total==960645600+count*544320
        work_sha[label]=hashlib.sha256(json.dumps(work,sort_keys=True).encode()).hexdigest()
        cost=1032538860 if label=='static360' else service[6 if label=='background_Q4' else int(label.split('_')[1])]
        summaries[label]=dict(upgraded_complete_tiles_per_conv=count,all28_accepted_bitplane128_beats=total,
            added_four_conv_core_beats_vs_Q4=count*544320,four_isolated_conv_service_ticks=cost,
            first_full_work_sha256=work_sha[label],targets=targets)
    pairs=[]
    for label,evaluation in res['evaluations'].items():
        for f in evaluation['frames']:
            tid=f['target_id'];assert f['action']==label and f['work_sha256']==work_sha[label]
            assert set(f['actual_affine_call_counts'])==set(contract['geometries']) and set(f['actual_affine_call_counts'].values())=={1}
            for k,n in [('actual_BN_calls',78),('actual_PSN_calls',105)]:assert len(f[k])==n and set(f[k].values())=={1}
            assert read(args.actual/'records'/(tid+'__'+label+'.json'))==f
            p=next(p for p in plan['records'] if p['target_id']==tid)
            assert f['policy_users']==[n for n,x in p['policy_actions'].items() if x==label]
            pairs.append((tid,label));all_frames.append(f)
    assert len(pairs)==71 and set(pairs)==expected_pairs
    for label,f in res['engineering'].items():
        assert label in ('background_Q4','static360') and f['work_sha256']==work_sha[label]
        assert f['all_four_head_hashes_exact'] and len(f['actual_affine_call_counts'])==28 and set(f['actual_affine_call_counts'].values())=={1}
    prefix=read(R38/'hardware_budget/actual_v2/ACTUAL_PREFIX_BUDGET_V1.json');byid={r['target_id']:r for r in prefix['rows']}
    policies=[]
    for p in plan['records']:
        tid=p['target_id'];r=byid[tid]
        for policy,key in [('folded128_fp32_fixed6','folded128'),('ordinary128_fp32_fixed6','ordinary128')]:
            action=p['policy_actions'][policy];m=r['models'][key];assert summaries[action]['four_isolated_conv_service_ticks']==m['dynamic_four_isolated_ticks']
            assert m['static360_four_isolated_ticks']==summaries['static360']['four_isolated_conv_service_ticks'] and m['static360_remaining_ticks']>0
            policies.append(dict(target_id=tid,policy=policy,action=action,actual_work_sha256=work_sha[action],
                actual_prefix_ticks=m['prefix_ticks'],four_isolated_conv_service_ticks=m['dynamic_four_isolated_ticks'],
                static360_margin=m['static360_remaining_ticks']))
    out=dict(status='PASS',schema='r38_independent_action_specific_actual_work_v1',formal_frames=71,engineering_frames=2,
        formal_affine_receipts=71*28,all_realized_actions=sorted(expected_labels),action_golden=summaries,
        actual_policy_service_bindings=policies,actual_work_hash_constant_within_action=True,
        all24_MLP_unchanged=True,true_tail_Q8=True,whole_9x9_tiles_only=True,all_actions_same_weight_and_scale_representations=True,
        bindings={str(p):sha(p) for p in [Path(__file__),args.actual/'results.json',R38/'SOURCE_FREEZE_V1.json',
            R38/'ACTION_PLAN_V1.json',R38/'hardware_budget/actual_v2/ACTUAL_PREFIX_BUDGET_V1.json']},
        scope=['Independent pixel/row masks, every high/low kernel chunk/plane/coefficient/padding and weight representation metadata checked for each observed action exemplar.',
               'Every formal/engineering receipt binds that action exemplar work SHA; full work is not redundantly stored per identity.',
               'Existing operator service model is associated only after actual geometry/precision action matches; core work is not relabeled as latency.',
               'Prefix plus four isolated conv services exclude shared stem and full-network execution.',
               'Prediction payload integrity and AEE are independently owned by the scientific reviewer, not recomputed here.',
               'No new GPU inference, RTL, synthesis, training or GT decoding by this reviewer.'])
    output=P/'hardware_actual_work_audit_v1.json';assert not output.exists();output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('action_golden','actual_policy_service_bindings','bindings')},indent=2))
if __name__=='__main__':main()
