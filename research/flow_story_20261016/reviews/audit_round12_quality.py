#!/usr/bin/env python3
"""Independent R12 V2 quality reductions/gates; no implementation imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ARMS=('native','FF','CC','fine_s2b2')
REGIONS=('all_valid','B','L','B_and_L','B_only','L_only_B_known','neither_B_known','B_unknown','voxel_nonzero','voxel_zero','B_voxel_nonzero','B_voxel_zero','L_voxel_nonzero','L_voxel_zero','B_unknown_voxel_nonzero','B_unknown_voxel_zero')
LIMITS={'all_valid':1.03,'B':1.05,'L':1.05}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise AssertionError(message)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,action='append',required=True)
    p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();require(len(args.result)==4 and not args.output.exists(),'Four results and fresh output required')
    here=Path(__file__).resolve().parent.parent/'ordinary_recovery_round12'
    members=json.loads((here/'check_membership_v1.json').read_text());ids=members['groups']['check']
    analysis=json.loads(args.analysis.read_text());require(analysis['schema']=='flow_round12_fixed_quality_analysis_v2','V2 analysis required')
    raw={};cells={};sources={};support={};state_hash=None;total=0;contract=None
    for path in args.result:
        z=json.loads(path.read_text());arm=z['arm'];require(arm in ARMS and arm not in raw,'Arm identity')
        require(z['status']=='ROUND12_FAIR_RECOVERY_COMPLETE' and z['mode']=='recover' and z['completed_optimizer_steps']==200,'Complete fixed endpoint')
        require(all(z[k] for k in ('frozen_state_unchanged','BN_buffers_unchanged','original_modules_restored','all_hooks_removed')),'State contract')
        require(z['full_forward_count']==(244 if arm=='native' else 246),'Per-arm forward count')
        require(z['binding']['fit_ids']==members['groups']['fit'] and z['binding']['check_ids']==ids,'Membership')
        require(len(z['steps'])==200 and [s['target_id'] for s in z['steps']]==members['groups']['fit']*10,'Frozen200 training order')
        require(z['allowed_parameter_count']==7252064 and len(z['allowed_parameter_names'])==80,'Allowed parameter scope')
        raw[arm]=z;sources[arm]={'path':str(path),'sha256':sha(path)};total+=z['full_forward_count']
        require(analysis['inputs'][arm]['sha256']==sha(path),'Analysis input SHA')
        if state_hash is None:state_hash=z['initial_model_state_sha256']
        require(state_hash==z['initial_model_state_sha256'],'Same initial model')
        stable={k:z['binding'][k] for k in ('runner_sha256','wrapper_sha256','runner_checks_sha256','source_pins_sha256','checkpoint_sha256','membership_sha256','selection_sha256','recipe','loss_sha256','allowed_parameter_count')}
        if contract is None:contract=stable
        require(stable==contract,'Four identical numerical/data/optimizer contracts')
        for label,modes in [('initial',(arm,)),('endpoint',ARMS)]:
            frames=z['evaluations'][label]['frames'];require([f['target_id'] for f in frames]==ids,'Ordered evaluation members')
            for mode in modes:
                key=('initial' if label=='initial' else arm)+'/'+mode;require(key not in cells,'Unique state/layout');cells[key]=[]
                for f in frames:
                    tid=f['target_id'];require(f['sequence']==tid.rsplit('_',1)[0],'Canonical sequence')
                    require(set(f['arms'])==set(modes),'All four cross-layout modes')
                    regions=f['arms'][mode]['metrics']['regions'];require(set(regions)==set(REGIONS),'All16regions')
                    for region,v in regions.items():
                        count=v['valid_pixels'];error=v['error_sum'];require(isinstance(count,int) and count>=0 and math.isfinite(error) and error>=0,'Valid region count/sum')
                        ref=(tid,region);support.setdefault(ref,count);require(support[ref]==count,'Same paired valid support')
                        if count:require(v['AEE'] is not None and math.isfinite(v['AEE']) and math.isclose(v['AEE'],error/count,rel_tol=1e-10,abs_tol=1e-10),'AEE from sum/count')
                        else:require(v['AEE'] is None and error==0,'Empty region')
                    atoms=('B_and_L','B_only','L_only_B_known','neither_B_known','B_unknown')
                    require(sum(regions[r]['valid_pixels'] for r in atoms)==regions['all_valid']['valid_pixels'],'Exclusive region count closure')
                    require(math.isclose(sum(regions[r]['error_sum'] for r in atoms),regions['all_valid']['error_sum'],rel_tol=1e-10,abs_tol=1e-10),'Exclusive error closure')
                    cells[key].append({'target_id':tid,'sequence':f['sequence'],'regions':regions})
    require(total==982 and len(cells)==20,'All982 forwards/all20state-layouts')
    sequences=sorted({t.rsplit('_',1)[0] for t in ids});require(len(sequences)==2,'Two sequences')
    summaries={};max_error=0.
    for key,frames in cells.items():
        groups={'overall':frames};groups.update({s:[f for f in frames if f['sequence']==s] for s in sequences})
        require(all(len(groups[s])==4 for s in sequences),'4+4support')
        summaries[key]={}
        for scope,rows in groups.items():
            summaries[key][scope]={}
            for region in REGIONS:
                values=[f['regions'][region] for f in rows];nonempty=[v for v in values if v['valid_pixels']]
                count=sum(v['valid_pixels'] for v in values);error=sum(v['error_sum'] for v in values)
                actual={'frames_with_region':len(nonempty),'valid_pixels':count,'error_sum':error,'frame_mean_AEE':sum(v['AEE'] for v in nonempty)/len(nonempty) if nonempty else None,'pixel_weighted_AEE':error/count if count else None}
                summaries[key][scope][region]=actual
                for k,v in actual.items():
                    got=analysis['summaries'][key][scope][region][k]
                    if v is None:require(got is None,'Empty summary')
                    else:
                        max_error=max(max_error,abs(v-got));require(math.isclose(v,got,rel_tol=1e-12,abs_tol=1e-12),'Summary numeric mismatch')
    def comparison(candidate,reference):
        flags={}
        for scope in ['overall']+sequences:
            flags[scope]={}
            for region,multiplier in LIMITS.items():
                x=summaries[candidate][scope][region]['frame_mean_AEE'];y=summaries[reference][scope][region]['frame_mean_AEE']
                require((x is None)==(y is None),'Comparison support')
                flags[scope][region]=(x is None or x<=y*multiplier)
        return {'pass':all(v for row in flags.values() for v in row.values()),'scope_region_flags':flags}
    normal={ref:comparison(ref+'/'+ref,'initial/'+ref) for ref in ('native','FF')}
    gates={}
    for candidate in ('CC','fine_s2b2'):
        cs=candidate+'/'+candidate
        families={'normal_references':normal,
                  'matched_recovered_references':{ref:comparison(cs,ref+'/'+ref) for ref in ('native','FF')},
                  'own_checkpoint_references':{ref:comparison(cs,candidate+'/'+ref) for ref in ('native','FF')},
                  'shared_fine_preserved':{ref:comparison(candidate+'/'+ref,ref+'/'+ref) for ref in ('native','FF')}}
        passed=all(v['pass'] for family in families.values() for v in family.values())
        got=analysis['candidate_gates'][candidate];require(got['ordinary_base_quality_pass']==passed,'V2 combined gate')
        for fname,family in families.items():
            for ref,comp in family.items():
                require(comp['pass']==got[fname][ref]['pass'],'Individual gate')
                for scope,regions in comp['scope_region_flags'].items():
                    for r,value in regions.items():require(value==got[fname][ref]['scopes'][scope][r]['pass'],'Individual scope-region gate')
        gates[candidate]={'families':families,'ordinary_base_quality_pass':passed}
    changes={}
    for arm in ARMS:
        vals=[old['regions']['all_valid']['AEE']-new['regions']['all_valid']['AEE'] for old,new in zip(cells['initial/'+arm],cells[arm+'/'+arm])]
        total_gain=sum(vals);got=analysis['own_layout_recovery'][arm]
        require(math.isclose(total_gain,got['signed_total_gain'],rel_tol=1e-12,abs_tol=1e-12),'Signed total gain')
        counts={'wins':sum(v>0 for v in vals),'ties':sum(v==0 for v in vals),'losses':sum(v<0 for v in vals)}
        require(all(got[k]==v for k,v in counts.items()),'OwnlayoutW/T/L')
        for i,v in enumerate(vals):
            require(math.isclose(got['per_member'][i]['initial_minus_post_AEE'],v,rel_tol=1e-12,abs_tol=1e-12),'Ownlayout perframe difference')
        changes[arm]={'signed_total_gain':total_gain,**counts}
    report={'schema':'round12_independent_quality_audit_v2','script_sha256':sha(__file__),'inputs':sources,'analysis_sha256':sha(args.analysis),'all160_endpoint_metrics_checked':True,'state_layouts':20,'regions':16,'scopes':['overall']+sequences,'full_forward_count':982,'max_summary_absolute_error':max_error,'independent_candidate_gates':gates,'own_layout_recovery':changes,'scope':'Independent saved-raw metric reductions and all four V2 gate families. No imports of collector/wrapper/analyzer, new fitting, quality tuning, GPU or original remote GT re-evaluation. Prediction/weights payload and service audits separately reported.'}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'audit':'PASS','output':str(args.output),'max_summary_error':max_error,'gates':{k:g['ordinary_base_quality_pass'] for k,g in gates.items()}}))

if __name__=='__main__':main()
