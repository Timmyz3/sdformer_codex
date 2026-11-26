#!/usr/bin/env python3
"""Predeclared four-layout recovery analysis; all endpoints and scopes retained."""
import argparse
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
ARMS=('native','FF','CC','fine_s2b2')
REGIONS=('all_valid','B','L','B_and_L','B_only','L_only_B_known','neither_B_known','B_unknown',
         'voxel_nonzero','voxel_zero','B_voxel_nonzero','B_voxel_zero','L_voxel_nonzero',
         'L_voxel_zero','B_unknown_voxel_nonzero','B_unknown_voxel_zero')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(values):
    active=[v for v in values if v['valid_pixels']>0]
    n=sum(v['valid_pixels'] for v in values)
    error=sum(v['error_sum'] for v in values)
    return {'frames_with_region':len(active),'valid_pixels':n,'error_sum':error,
            'frame_mean_AEE':sum(v['AEE'] for v in active)/len(active) if active else None,
            'pixel_weighted_AEE':error/n if n else None}


def compare(candidate,reference):
    out={}
    for scope,regions in candidate.items():
        out[scope]={}
        for region,limit in [('all_valid',.03),('B',.05),('L',.05)]:
            x=regions[region]['frame_mean_AEE'];y=reference[scope][region]['frame_mean_AEE']
            assert (x is None)==(y is None)
            out[scope][region]={'candidate':x,'reference':y,'limit':limit,
                'relative_change':x/y-1 if y is not None and y!=0 else None,
                'evidence':x is not None,'pass':x is None or x<=y*(1+limit)}
    return {'scopes':out,'pass':all(v['pass'] for row in out.values() for v in row.values())}


def gates(summaries):
    normal={'native':compare(summaries['native/native'],summaries['initial/native']),
            'FF':compare(summaries['FF/FF'],summaries['initial/FF'])}
    result={}
    for arm in ('CC','fine_s2b2'):
        candidate=summaries[arm+'/'+arm]
        matches={ref:compare(candidate,summaries[ref+'/'+ref]) for ref in ('native','FF')}
        shared={ref:compare(candidate,summaries[arm+'/'+ref]) for ref in ('native','FF')}
        preserved={ref:compare(summaries[arm+'/'+ref],summaries[ref+'/'+ref]) for ref in ('native','FF')}
        passed=all(v['pass'] for family in (normal,matches,shared,preserved) for v in family.values())
        result[arm]={'normal_references':normal,'matched_recovered_references':matches,
                     'own_checkpoint_references':shared,'shared_fine_preserved':preserved,'ordinary_base_quality_pass':passed,
                     'classification':'limited_ordinary_quality_base' if passed else 'failed_fixed_ordinary_quality_gate',
                     'flow_specific_increment':False,'hardware_gate_evaluated':False}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,action='append',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();assert len(a.result)==4 and not a.output_dir.exists()
    member=json.loads((HERE/'check_membership_v1.json').read_text());ids=member['groups']['check']
    raw={};evidence={};common={};cells={};input_counts={};full=0
    for path in a.result:
        z=json.loads(path.read_text());arm=z['arm'];assert arm in ARMS and arm not in raw
        assert z['status']=='ROUND12_FAIR_RECOVERY_COMPLETE' and z['completed_optimizer_steps']==200
        assert all(z[k] for k in ('frozen_state_unchanged','BN_buffers_unchanged','original_modules_restored','all_hooks_removed'))
        assert z['binding']['check_ids']==ids and z['binding']['fit_ids']==member['groups']['fit']
        raw[arm]=z;evidence[arm]={'path':str(path),'sha256':sha(path)};full+=z['full_forward_count']
        for k in ('runner_sha256','wrapper_sha256','runner_checks_sha256','source_pins_sha256','checkpoint_sha256',
                  'membership_sha256','selection_sha256','recipe','loss_sha256','allowed_parameter_count'):
            if k not in common:common[k]=z['binding'][k]
            assert common[k]==z['binding'][k], 'Mixed contracts: '+k
        if 'initial_state' not in common:common['initial_state']=z['initial_model_state_sha256']
        assert common['initial_state']==z['initial_model_state_sha256']
        for label,expected in [('initial',(arm,)),('endpoint',ARMS)]:
            frames=z['evaluations'][label]['frames'];assert [f['target_id'] for f in frames]==ids
            for mode in expected:
                key=('initial' if label=='initial' else arm)+'/'+mode
                assert key not in cells;cells[key]=[]
                for f in frames:
                    assert set(f['arms'])==set(expected)
                    m=f['arms'][mode]['metrics']['regions'];assert set(m)==set(REGIONS)
                    for region,v in m.items():
                        count_key=(f['target_id'],region)
                        input_counts.setdefault(count_key,v['valid_pixels']);assert input_counts[count_key]==v['valid_pixels']
                        assert v['valid_pixels']>=0 and math.isfinite(v['error_sum']) and v['error_sum']>=0
                        if v['valid_pixels']:
                            assert v['AEE'] is not None and math.isfinite(v['AEE'])
                            assert math.isclose(v['AEE'],v['error_sum']/v['valid_pixels'],rel_tol=1e-10,abs_tol=1e-10)
                        else:assert v['AEE'] is None and v['error_sum']==0
                    cells[key].append({'target_id':f['target_id'],'sequence':f['sequence'],'regions':m})
    assert set(raw)==set(ARMS) and len(cells)==20 and full==982
    sequences=sorted({f['sequence'] for f in next(iter(cells.values()))})
    assert len(sequences)==2
    summaries={}
    for key,frames in cells.items():
        scopes={'overall':frames};scopes.update({s:[f for f in frames if f['sequence']==s] for s in sequences})
        assert all(len(scopes[s])==4 for s in sequences)
        summaries[key]={s:{r:aggregate([f['regions'][r] for f in fs]) for r in REGIONS} for s,fs in scopes.items()}
    changes={}
    for arm in ARMS:
        initial=cells['initial/'+arm];post=cells[arm+'/'+arm]
        rows=[{'target_id':f['target_id'],'initial_minus_post_AEE':f['regions']['all_valid']['AEE']-g['regions']['all_valid']['AEE']}
              for f,g in zip(initial,post)]
        total=sum(r['initial_minus_post_AEE'] for r in rows)
        for r in rows:r['signed_gain_share_percent']=100*r['initial_minus_post_AEE']/total if total else None
        changes[arm]={'per_member':rows,'signed_total_gain':total,'wins':sum(r['initial_minus_post_AEE']>0 for r in rows),
                      'ties':sum(r['initial_minus_post_AEE']==0 for r in rows),'losses':sum(r['initial_minus_post_AEE']<0 for r in rows)}
    result={'schema':'flow_round12_fixed_quality_analysis_v2','inputs':evidence,'script_sha256':sha(__file__),
            'common_contract':common,'summaries':summaries,'candidate_gates':gates(summaries),'own_layout_recovery':changes,
            'all_endpoints':cells,'full_forward_count':full,'evaluation_endpoint_count':160,
            'scope':'Finite ordinary recovery pilot only. All20 checkpoint/layout combinations and all16 regions retained; no official test, independence, significance, flow novelty or net-cycle claim.'}
    a.output_dir.mkdir(parents=True)
    (a.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# R12 全部初始与交叉布局质量','',result['scope'],'',
           '| 训练状态/评价布局 | allGT | B | L |','|---|---:|---:|---:|']
    for key,s in summaries.items():
        values=[s['overall'][r]['frame_mean_AEE'] for r in ('all_valid','B','L')]
        lines.append('| '+key+' | '+' | '.join('%.9f'%v if v is not None else 'empty' for v in values)+' |')
    for name,g in result['candidate_gates'].items():lines+=['',name+': '+g['classification']+'。完整分序列/双参考/同checkpoint门见JSON。']
    (a.output_dir/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v['classification'] for k,v in result['candidate_gates'].items()}))


if __name__=='__main__':main()
