"""R35 actual static360 quality; immutable R34 policies, no fitting or selection."""
from pathlib import Path
import sys,os,time,hashlib,json,copy
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;R35=HERE.parent;R34=R35.parent/'round34_deployability';R33=R35.parent/'round33_predictability'
sys.path.insert(0,str(R34/'bias_control'))
import run_bias as bc
u=bc.u

def main():
    start=time.time();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):assert os.environ.get(k)=='2'
    freeze=u.read(HERE/'SOURCE_FREEZE_V1.json')
    assert u.sha(__file__)==freeze['files']['reduce_static.py']
    for bind in freeze['bound'].values():assert u.sha(bind['path'])==bind['sha256']
    out=HERE/'actual_v1';assert not out.exists();out.mkdir()
    ev=u.imported('r35_eval_frozen',R33/'evaluate.py');red=u.imported('r35_raw_frozen',R35.parent/'independent_review/reduce_e1_quality_v1.py');raw=red.raw
    rp=R33.parents[1]/'flow_story_20261016/coverage_recovery_round13/preparation/regions_v1.json';assert u.sha(rp)==red.REGIONS_SHA;regions=u.read(rp)
    run=R35/'actual/static360_v1';result=u.read(run/'results.json');contract=u.read(R35/'ACTION_CONTRACT_V1.json')
    assert result['status']=='R35_STATIC360_COMPLETE' and result['formal_model_forwards']==120 and result['engineering_model_forwards']==4
    assert result['model_forwards']==result['forward_attempts']==124 and result['completed_optimizer_steps']==0
    assert result['source_freeze_sha256']==u.sha(R35/'SOURCE_FREEZE_V1.json') and result['action_contract_sha256']==u.sha(R35/'ACTION_CONTRACT_V1.json')
    assert result['final_state_sha256']==contract['endpoint']['state_sha256']
    assert all(result['parity']['Q4_'+role]['all_four_head_hashes_exact'] for role in ('fit','tune','confirm'))
    assert result['parity']['static360_dense']['comparison']['pass']
    members=u.read(R33/'data/MEMBERSHIP_V1.json')['records'];rows=result['evaluations']['static360']['frames'];assert [r['target_id'] for r in rows]==[r['target_id'] for r in members]
    sources=u.read(R34/'bias_control/actual_v1/SOURCE_BINDINGS_V1.json')['records']
    bmodel=R34/'bias_control/actual_v1/BIAS_MODEL_V1.json';b=np.asarray(u.read(bmodel)['bias'],dtype=np.float64);bindings={};quality=[];zero=0
    for rec,row in zip(members,rows):
        tid=rec['target_id'];assert row['role']==rec['role'] and row['sequence']==rec['sequence'];src={};srcbind={}
        for cat in ('input','gt','mask'):
            ref=sources[tid][cat];expected=next(f for f in rec['source_files'] if f['category']==cat);assert ref['sha256']==expected['sha256'] and u.sha(ref['path'])==ref['sha256']
            src[cat]=np.load(ref['path'],allow_pickle=False);srcbind[cat]=ref
        path=red.prediction(run,row)
        with np.load(path,allow_pickle=False) as z:p=z['head3'].copy()
        masks=raw.region_masks(src['gt'],src['mask'],src['input'],regions['boundary_threshold_pixels'],regions['large_displacement_threshold_pixels'])
        original=raw.measure(p,src['gt'],masks);assert original==bc.measure64(p.astype(np.float64),src['gt'],masks);zero+=len(original)
        corrected=p.astype(np.float64)-b[:,None,None];changed=bc.measure64(corrected,src['gt'],masks)
        quality.append(dict(target_id=tid,sequence=rec['sequence'],role=rec['role'],original=original,same_bias_corrected=changed,
            corrected_payload_sha256=hashlib.sha256(np.ascontiguousarray(corrected).tobytes()).hexdigest()))
        bindings[tid]=dict(source=srcbind,prediction=dict(path=str(path),sha256=u.sha(path),head3_sha256=row['prediction_sha256']))
    u.write(out/'QUALITY_V1.json',dict(schema='flow_r35_static360_raw_quality_v1',frames=quality,bias_model_sha256=u.sha(bmodel),region_source_sha256=u.sha(rp),
        all16regions=True,zero_bias_exact_region_comparisons=zero,no_policy_or_model_fit=True))
    u.write(out/'BINDINGS_V1.json',dict(source_freeze_sha256=u.sha(HERE/'SOURCE_FREEZE_V1.json'),collector_result_sha256=u.sha(run/'results.json'),records=bindings))
    oldroot=R34/'bias_control/extension_v1/actual_v1';role_summary={};byid={f['target_id']:f for f in quality};oldbindings={}
    important=['Q4','X128__folded_constant_v__six','X128__folded_constant_v__allow_Q4','X160__signed_g1__six','X128__calibrated_ordinary__six','X160__calibrated_ordinary__six','fit_static__six','uniform_six_expectation','oracle__six']
    for role in ('fit','tune','confirm'):
        role_summary[role]={};ids=[r['target_id'] for r in members if r['role']==role]
        for condition,oldcondition in [('original','uncorrected'),('same_bias_corrected','same_bias_corrected')]:
            oldpath=oldroot/f'{role}_{oldcondition}_EVALUATION_V1.json';old=u.read(oldpath);oldbindings[str(oldpath)]=u.sha(oldpath)
            assert [f['target_id'] for f in old['frames']]==ids
            frames=[]
            for f in old['frames']:
                q=byid[f['target_id']];assert q['sequence']==f['sequence'];states={name:v['regions'] for name,v in f['policies'].items()};states['static360']=q[condition]
                assert all(v['all_valid']['valid_pixels']==q[condition]['all_valid']['valid_pixels'] for v in states.values())
                frames.append(dict(target_id=f['target_id'],sequence=f['sequence'],role=role,states=states))
            seqs=list(dict.fromkeys(f['sequence'] for f in frames));summary={}
            for policy in frames[0]['states']:
                summary[policy]={scope:{reg:ev.mc.aggregate([f['states'][policy][reg] for f in frames if scope=='overall' or f['sequence']==scope]) for reg in ev.REGIONS} for scope in ['overall']+seqs}
                if policy!='static360':assert summary[policy]==old['summaries'][policy]
            comparisons={name+'__vs_Q4':ev.comparison(frames,summary,name,'Q4',seqs) for name in summary if name!='Q4'}
            for name in summary:
                if name!='static360':comparisons['static360__vs_'+name]=ev.comparison(frames,summary,'static360',name,seqs)
            for name in important:
                if name!='Q4':comparisons[name+'__vs_static360']=ev.comparison(frames,summary,name,'static360',seqs)
            leave={}
            for other in important:
                leave[other]={}
                for dropped in seqs:
                    selected=[f for f in frames if f['sequence']!=dropped];leave[other][dropped]={}
                    for reg in ('all_valid','B','L'):
                        x=ev.mc.aggregate([f['states']['static360'][reg] for f in selected]);y=ev.mc.aggregate([f['states'][other][reg] for f in selected]);xa,ya=x['pixel_weighted_AEE'],y['pixel_weighted_AEE']
                        leave[other][dropped][reg]=dict(static360=xa,reference=ya,delta=ev.mc.difference(xa,ya),relative=ev.mc.ratio(xa,ya),valid_pixels=x['valid_pixels'])
            details=[dict(target_id=f['target_id'],sequence=f['sequence'],role=role,policies={n:dict(regions=v) for n,v in f['states'].items()}) for f in frames]
            res=dict(schema='flow_r35_static_neighbor_evaluation_v1',role=role,condition=condition,summaries=summary,comparisons=comparisons,frames=details,
                static360_leave_one_sequence_vs_reference=leave,original_policy_choices=old['unchanged_observable_policy_choices'],
                uniform_six_actual_tail_risk=old['uniform_six_actual_tail_risk'],oracle_scope='Original six or seven actions only; static360 excluded from oracle candidate set',
                data_previously_exposed=True,new_head_or_bias_fits=0,old_quality_exact=True,postprocess_cost=old['postprocess_cost'],
                static360_cost=dict(four_isolated_convolution_service_ticks=1032538860,selector=False,complete_Q8_tiles_each_patch=360,
                    hardware_contract_sha256=u.sha(R35/'HARDWARE_STATIC360_V2.json'),budget_scope='R34 V2 serialized prefix plus four isolated convolution services; not network latency',
                    old_policy_operator_costs_not_upgraded_here=True))
            u.write(out/f'{role}_{condition}_EVALUATION_V1.json',res)
            role_summary[role][condition]={name:dict(AEE=vals['overall']['all_valid']['pixel_weighted_AEE'],frame_mean_AEE=vals['overall']['all_valid']['frame_mean_AEE'],
                B=vals['overall']['B']['pixel_weighted_AEE'],L=vals['overall']['L']['pixel_weighted_AEE'],
                relative_Q4=ev.mc.ratio(vals['overall']['all_valid']['pixel_weighted_AEE'],summary['Q4']['overall']['all_valid']['pixel_weighted_AEE']),
                relative_static360=ev.mc.ratio(vals['overall']['all_valid']['pixel_weighted_AEE'],summary['static360']['overall']['all_valid']['pixel_weighted_AEE']),
                tails=0 if name=='Q4' else len(comparisons[name+'__vs_Q4']['high_support_tail_review'])) for name,vals in summary.items()}
        print('ROLE_COMPLETE',role,flush=True)
    u.write(out/'SUMMARY_V1.json',dict(status='COMPLETE',results=role_summary,old_evaluation_bindings=oldbindings,CPU_seconds=time.time()-start,GPU_used=False,
        static360_formal_forwards=120,engineering_forwards=4,all_data_previously_exposed=True,fullnet_cycle_claim=False,zero_bias_exact_region_comparisons=zero,
        bias_model_sha256=u.sha(bmodel),source_freeze_sha256=u.sha(HERE/'SOURCE_FREEZE_V1.json')))
    print('R35_QUALITY_COMPLETE',round(time.time()-start,3),flush=True)
if __name__=='__main__':main()
