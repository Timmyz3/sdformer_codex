#!/usr/bin/env python3
"""CPU arithmetic opportunity screen, never a cycle/RTL/quality result.

Two separate objects: source-compatible operation vectors and a labeled old
issue-template sensitivity with current-BN mathematical repair. This code does
not read experimental quality, import a model, or use R13 serial-BN cycles.
"""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
RESEARCH=HERE.parents[2]
OLD=RESEARCH/'flow_story_20261011/preflight/execution_contract/execution_services_results.json'
BN=RESEARCH/'flow_story_20261014/damage_allocation_probe/hardware_contract_complete_bn_audit.json'
SOURCES=[OLD,BN,HERE.parents[1]/'ordinary_recovery_round12/DEPLOYMENT_COST_BOUNDARY.md']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ceildiv(a,b):return (a+b-1)//b

def ledger(arm):
    counts=Counter();layers=[]
    for stage,(depth,h,w,c) in enumerate([(2,120,160,96),(2,60,80,192),(6,30,40,384),(2,15,20,768)]):
        # Represent each real 9x9 tail tile by ceil(h/3)*ceil(w/3) cells.
        reps=sum(ceildiv(min(9,h-y),3)*ceildiv(min(9,w-x),3) for y in range(0,h,9) for x in range(0,w,9))
        for block in range(depth):
            coarse=arm!='FF' and stage in (0,2) and not(arm=='P' and stage==2 and block==2)
            p=reps if coarse else h*w; q=h*w; t=10; f=4*c
            row=dict(FC1_rows=t*p,FC2_rows=t*p,FC_weighted_terms=2*t*p*c*f,
                     PSN_real_temporal_MACs=t*t*p*(c+f),threshold_elements=t*p*(c+f),
                     private_BN_elements=t*p*(c+f),residual_add_elements=t*q*c,
                     pool_add= t*c*(q-p) if coarse else 0,
                     pool_real_area_divide=t*c*p if coarse else 0,
                     pool_input_elements=t*c*q if coarse else 0,
                     restored_output_elements=t*c*q if coarse else 0)
            counts.update(row)
            # Same old specified issue packing, all these dimensions divide16.
            # PSN has M=T,K=T,N=P*C, so 16 reduction lanes cannot be assumed full.
            fc=t*p*ceildiv(c,16)*ceildiv(f,16)*2
            psn=t*ceildiv(t,16)*(ceildiv(p*c,16)+ceildiv(p*f,16))
            threshold=ceildiv(t*p*c,16)+ceildiv(t*p*f,16)
            layers.append(dict(stage=stage,block=block,shape=[10,1,h,w,c],coarse=coarse,
                               actual_spatial_points=p,counts=row,old_issue_components=dict(FC=fc,PSN=psn,threshold=threshold)))
    issue={k:sum(x['old_issue_components'][k] for x in layers) for k in ('FC','PSN','threshold')}
    return dict(operation_vector=dict(counts),layers=layers,old_template_private_producer_issue=sum(issue.values()),
                old_template_issue_components=issue)


def run():
    assert sha(OLD)=='3cc67c3c331dc8475cd1647003a684bb6a36843e8b6ce00faa5b41f1f30968e8'
    assert sha(BN)=='9a92bd764878d36b5ca67af8ef0a39806830e755dd0f9a91217c81e165e27567'
    old=json.loads(OLD.read_text()); current=json.loads(BN.read_text())
    base=next(r for r in old['configurations'] if r['name']=='full_480x640_window9_current_config')
    assert all(old['source_sha256'][k]==v for k,v in current['native_source_sha256'].items())
    bn=current['external_BN'];e=sum(r['elements'] for r in bn['records']);c=sum(r['channels'] for r in bn['records'])
    assert (e,c,len(bn['records']))==(976584960,19056,54)
    # Independent centered two-pass mathematical counters, not old aggregate only.
    bn_add=sum(4*r['elements'] for r in bn['records'])
    bn_mul=sum(2*r['elements']+4*r['channels'] for r in bn['records'])
    bn_issue=sum(ceildiv(4*r['elements'],16)+ceildiv(2*r['elements']+4*r['channels'],16) for r in bn['records'])
    assert bn_issue==current['conditional_service_denominator']['declared_BN_vector16_issue_excluding_rsqrt']==366224124
    padding=sum(r['depth']*(r['new_packed_attention_issue_per_block']-r['old_packed_attention_issue_per_block']) for r in current['attention_padding_repair'])
    assert padding==26902080
    old_total=base['cross_group_packed_dense_baseline_issue_steps'];assert old_total==3045155774
    models={arm:ledger(arm) for arm in ('FF','CC','P')};fine=models['FF']['operation_vector']
    assert [models[a]['operation_vector']['FC1_rows'] for a in ('FF','CC','P')]==[558000,153600,164200]
    assert fine['FC_weighted_terms']==169869312000 and fine['PSN_real_temporal_MACs']==4377600000
    rows=[]
    for private_bn_ops in (0,2):
        private_issue=private_bn_ops*fine['private_BN_elements']//16
        subtotal=old_total+padding+bn_issue+private_issue
        for arm in ('CC','P'):
            actual=models[arm]['operation_vector'];delta={k:fine[k]-actual[k] for k in ('FC1_rows','FC2_rows','FC_weighted_terms','PSN_real_temporal_MACs','threshold_elements','private_BN_elements')}
            issue_delta=models['FF']['old_template_private_producer_issue']-models[arm]['old_template_private_producer_issue']
            issue_delta+=private_bn_ops*delta['private_BN_elements']//16
            # Optimistic arithmetic screen assigns division/real-area scaling
            # same one vector16 issue as multiplication; exact divider unverified.
            pool=(actual['pool_add']+actual['pool_real_area_divide'])//16
            require_divisible=(actual['pool_add']+actual['pool_real_area_divide'])%16==0;assert require_divisible
            net=issue_delta-pool
            # Let X be omitted common issue, Y net incremental candidate issue.
            # (delta-pool-Y)/(subtotal+X)>=.1 iff X+10Y <=10net-subtotal.
            cap=10*net-subtotal
            rows.append(dict(arm=arm,private_BN_ops_per_element=private_bn_ops,
                private_BN_assumption='old folded-private-BN assumption, unverified for native FP32' if private_bn_ops==0 else 'two vector affine ops per element; FP32 native equivalence unverified',
                known_baseline_issue_subtotal=subtotal,cancelled_private_producer_issue=issue_delta,
                optimistic_pool_issue=pool,net_known_issue_reduction=net,
                conditional_known_subtotal_reduction_percent=float(Fraction(100*net,subtotal)),
                combined_missing_issue_budget_X_plus_10Y=cap,
                if_Y_zero_max_common_X=cap,if_X_zero_max_added_Y_integer=cap//10,
                necessary_inequality='X + 10Y <= combined_missing_issue_budget_X_plus_10Y; X,Y >=0 within this fixed issue template',
                operation_vector_delta=delta,
                interpretation='conditional arithmetic-screen opportunity only; neither real PE-cycle upper bound nor current complete network denominator'))
    return dict(schema='round13_MLP_headroom_screen_v1',status='ARITHMETIC_SCREEN_ONLY_NO_GATE_PASS',
                source_sha256={str(p):sha(p) for p in SOURCES},common_native_source_overlap_matches=True,
                source_compatible_MLP_vectors=models,
                shared_current_BN=dict(count=54,channels=c,elements=e,centered_two_pass_add_sub=bn_add,
                                       centered_two_pass_mul=bn_mul,rsqrt=c,vector16_arithmetic_issue_excluding_rsqrt=bn_issue),
                old_template=dict(whole_folded_issue=old_total,padded_attention_restoration_issue=padding,
                    already_included_shared_conv_bottleneck_decoder_patch_service=base['retained_summary']['static_issue_steps'],
                    issue_parameters=base['engine'],meaning='historical same-source shape template, not a calibrated PE machine'),
                conditional_issue_table=rows,
                actual_PE_cycles=None,actual_whole_network_cycles=None,actual_10percent_gate=None,actual_latency_gate=None,
                serial_R13_BN_cycles_used=False,experimental_quality_read=False,GPU_used=False,
                decision='Do not kill the action solely from this arithmetic screen; do not declare its actual 10% gate feasible/passed. Complete common-machine denominator and predictor cost remain required.')

if __name__=='__main__':
    out=HERE/'HEADROOM_TABLE_V1.json';assert not out.exists()
    out.write_text(json.dumps(run(),indent=2,sort_keys=True)+'\n')
    print('Wrote',out.name)
