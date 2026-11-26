"""Same frozen b on every actual action, with original choices unchanged."""
from pathlib import Path
import sys,os,time,hashlib,copy
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import run_bias as biascode
u=biascode.u

def choice_record(role):
    choices={};bindings={};ids=None
    for width in (128,160):
        path=u.HERE/'actual_v1'/f'{role}_{width}'/'PREDICTIONS_V1.json';v=u.read(path);bindings[str(path)]=u.sha(path)
        if ids is None:ids=v['ids']
        assert ids==v['ids']
        for mode,field in [('six','fixed_six_choices'),('allow_Q4','allow_Q4_choices')]:
            for method,action in v['results']['fp64'][field].items():choices[f'X{width}__{method}__{mode}']=action
        path=u.HERE/'ordinary_extension_actual_v1'/f'{role}_{width}_ARRAYS_V1.npz';bindings[str(path)]=u.sha(path)
        with np.load(path,allow_pickle=False) as z:
            assert ids==z['ids'].astype(str).tolist()
            choices[f'X{width}__calibrated_ordinary__six']=z['results__fp64__fixed_six'].tolist()
            choices[f'X{width}__calibrated_ordinary__allow_Q4']=z['results__fp64__allow_Q4'].tolist()
    path=u.HERE/'folded_constant_v/actual_v1'/f'{role}_ARRAYS_V1.npz';bindings[str(path)]=u.sha(path)
    with np.load(path,allow_pickle=False) as z:
        assert ids==z['ids'].astype(str).tolist()
        choices['X128__folded_constant_v__six']=z['choices__unfolded_fp64__fixed_six'].tolist()
        choices['X128__folded_constant_v__allow_Q4']=z['choices__unfolded_fp64__allow_Q4'].tolist()
    mp=u.R33/'fitted/voxel128_v1/MODEL_V1.json';m=u.read(mp);bindings[str(mp)]=u.sha(mp)
    choices['Q4']=[-1]*len(ids);choices['fit_static__six']=[m['metadata']['fit_static_six_action']]*len(ids)
    choices['fit_static__allow_Q4']=[m['metadata']['fit_static_allow_Q4_action']]*len(ids)
    return dict(ids=ids,choices=choices,source_hashes=bindings,no_bias_or_corrected_quality_used=True)

def policy_set(choices,delta,ev):
    n=len(delta);policies={}
    def add(name,action,regime,observable='frozen_pre_bias_choices',control='dynamic'):
        a=np.asarray(action,dtype=np.int64);policies[name]=dict(weights=ev.policy_weights(a),choices=a,observable=observable,
            regime=regime,operator_control=control,label_source=None,stochastic_expectation=False)
    for name,a in choices.items():
        regime='Q4_only' if name=='Q4' else ('fixed_six' if name.endswith('__six') else 'allow_Q4')
        add(name,a,regime,control='static' if name=='Q4' or name.startswith('fit_static') else 'dynamic')
    policies['uniform_six_expectation']=dict(weights=np.tile(np.r_[0,np.ones(6)/6],(n,1)),choices=None,
        observable='fixed_uniform',regime='fixed_six',operator_control='dynamic',label_source=None,stochastic_expectation=True)
    add('oracle__six',np.argmin(delta,axis=1),'fixed_six','offline_GT_counterfactual_diagnostic')
    best=np.argmin(delta,axis=1);best[np.min(delta,axis=1)>=0]=-1
    add('oracle__allow_Q4',best,'allow_Q4','offline_GT_counterfactual_diagnostic')
    return policies

def leave_one(frames,summary,seqs,ev):
    out={}
    for policy in summary:
        out[policy]={}
        for dropped in seqs:
            selected=[f for f in frames if f['sequence']!=dropped];out[policy][dropped]={}
            for region in ('all_valid','B','L'):
                c=ev.mc.aggregate([f['states'][policy][region] for f in selected]);r=ev.mc.aggregate([f['states']['Q4'][region] for f in selected])
                ca,ra=c['pixel_weighted_AEE'],r['pixel_weighted_AEE']
                out[policy][dropped][region]=dict(candidate_AEE=ca,Q4_AEE=ra,delta=ev.mc.difference(ca,ra),relative=ev.mc.ratio(ca,ra),
                    valid_pixels=c['valid_pixels'])
    return out

def main():
    start=time.time();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):assert os.environ.get(k)=='2'
    freeze=u.check_freeze(HERE/'SOURCE_FREEZE_V1.json')
    for value in freeze['bound'].values():assert u.sha(value['path'])==value['sha256']
    out=HERE/'actual_v1';assert not out.exists();out.mkdir()
    bmodel=HERE.parent/'actual_v1/BIAS_MODEL_V1.json';b=np.asarray(u.read(bmodel)['bias'],dtype=np.float64)
    memberpath=u.R33/'data/MEMBERSHIP_V1.json';assert u.sha(memberpath)==u.MEMBERSHIP_SHA;members=u.read(memberpath)['records']
    frozen_choices={role:choice_record(role) for role in ('fit','tune','confirm')}
    u.write(out/'FROZEN_POLICY_CHOICES_V1.json',dict(roles=frozen_choices,bias_model_sha256=u.sha(bmodel),corrected_quality_computed=False))
    reducer=u.imported('r34_biasext_source',u.R33.parent/'independent_review/reduce_e1_quality_v1.py');raw=reducer.raw
    ev=u.imported('r34_biasext_eval',u.R33/'evaluate.py');hardware=u.read(u.R33/'hardware/results/SUMMARY_V1.json')
    rp=u.R33.parents[1]/'flow_story_20261016/coverage_recovery_round13/preparation/regions_v1.json';assert u.sha(rp)==reducer.REGIONS_SHA;regions=u.read(rp)
    sources=u.read(HERE.parent/'actual_v1/SOURCE_BINDINGS_V1.json')['records'];predpaths={};bindings={}
    for phase in ('fit_tune','confirm'):
        run=u.R33/'actual'/f'{phase}_v1';result=u.read(run/'results.json')
        for state in ev.STATES:
            for row in result['evaluations'][state]['frames']:
                path=reducer.prediction(run,row);predpaths[row['target_id'],state]=path
                bindings[row['target_id']+'/'+state]=dict(path=str(path),sha256=u.sha(path))
    for tid,record in sources.items():
        for field in ('input','gt','mask'):assert u.sha(record[field]['path'])==record[field]['sha256']
    u.write(out/'ALL_ACTION_INPUT_BINDINGS_V1.json',dict(predictions=bindings,source_binding_sha256=u.sha(HERE.parent/'actual_v1/SOURCE_BINDINGS_V1.json')))
    roles={};zero_comparisons=0
    for role in ('fit','tune','confirm'):
        phase='confirm' if role=='confirm' else 'fit_tune';packdir=u.R33/'packs'/f'{phase}_128_v1'
        with np.load(packdir/f'{role}.npz',allow_pickle=False) as z:pack={k:z[k].copy() for k in z.files}
        ids=pack['ids'].astype(str).tolist();assert ids==frozen_choices[role]['ids']
        cached={f['target_id']:f for f in u.read(packdir/'QUALITY_V1.json')['frames']};corrected=[];uncorrected=[];hashes=[]
        for tid,group in zip(ids,pack['groups']):
            src={k:np.load(sources[tid][k]['path'],allow_pickle=False) for k in ('input','gt','mask')}
            masks=raw.region_masks(src['gt'],src['mask'],src['input'],regions['boundary_threshold_pixels'],regions['large_displacement_threshold_pixels'])
            states={};payload={}
            for state in ev.STATES:
                with np.load(predpaths[tid,state],allow_pickle=False) as z:prediction=z['head3']
                zero=biascode.measure64(prediction.astype(np.float64),src['gt'],masks)
                assert zero==cached[tid]['states'][state];zero_comparisons+=len(zero)
                changed=prediction.astype(np.float64)-b[:,None,None]
                states[state]=biascode.measure64(changed,src['gt'],masks)
                payload[state]=hashlib.sha256(np.ascontiguousarray(changed).tobytes()).hexdigest()
            corrected.append(dict(target_id=tid,sequence=str(group),role=role,states=states));uncorrected.append(cached[tid]);hashes.append(dict(target_id=tid,corrected_payload_sha256=payload))
        u.write(out/f'{role}_ALL_ACTION_QUALITY_V1.json',dict(bias_model_sha256=u.sha(bmodel),corrected=corrected,uncorrected=uncorrected,
            corrected_payloads=hashes,old_arrays_zero_bias_exact=True,all16_GT_regions=True))
        role_result={}
        for condition,frames in [('uncorrected',uncorrected),('same_bias_corrected',corrected)]:
            localpack=dict(pack);localpack['delta_aee']=np.asarray([[f['states'][s]['all_valid']['AEE']-f['states']['background_Q4']['all_valid']['AEE'] for s in ev.STATES[1:]] for f in frames])
            policies=policy_set(frozen_choices[role]['choices'],localpack['delta_aee'],ev)
            allframes,summary,decision,details,seqs=ev.summarize_policies(frames,localpack,policies,hardware)
            comparisons={name+'__vs_Q4':ev.comparison(allframes,summary,name,'Q4',seqs) for name in policies if name!='Q4'}
            for mode in ('six','allow_Q4'):
                for width,candidate in [(128,f'X128__folded_constant_v__{mode}'),(160,f'X160__signed_g1__{mode}')]:
                    others=[f'X{width}__{m}__{mode}' for m in ('direct_raw','direct_aug_g1','direct_aug_g4','direct_residual_g1','direct_residual_g4','calibrated_ordinary')]
                    others += [f'fit_static__{mode}',f'X{288-width}__calibrated_ordinary__{mode}']
                    for other in others:comparisons[candidate+'__vs_'+other]=ev.comparison(allframes,summary,candidate,other,seqs)
            result=dict(schema='flow_r34_matched_bias_policy_evaluation_v1',role=role,condition=condition,
                summaries=summary,decisions=decision,frames=details,comparisons=comparisons,
                uniform_six_actual_tail_risk=ev.uniform_tail_risk(frames),leave_one_sequence=leave_one(allframes,summary,seqs,ev),
                unchanged_observable_policy_choices=frozen_choices[role]['choices'],oracle_is_offline_only=True,
                postprocess_cost=dict(component_adds_per_identity=614400 if condition=='same_bias_corrected' else 0,
                    constants_bytes=16 if condition=='same_bias_corrected' else 0,measured_cycles=False,common_to_all_policies=True,
                    output_traffic_and_feature_head_cost_not_in_operator_service=True),data_already_exposed=True)
            u.write(out/f'{role}_{condition}_EVALUATION_V1.json',result)
            role_result[condition]={name:dict(AEE=vals['overall']['all_valid']['pixel_weighted_AEE'],
                B=vals['overall']['B']['pixel_weighted_AEE'],L=vals['overall']['L']['pixel_weighted_AEE'],
                delta_Q4=ev.mc.difference(vals['overall']['all_valid']['pixel_weighted_AEE'],summary['Q4']['overall']['all_valid']['pixel_weighted_AEE']),
                relative_Q4=ev.mc.ratio(vals['overall']['all_valid']['pixel_weighted_AEE'],summary['Q4']['overall']['all_valid']['pixel_weighted_AEE']),
                tails=0 if name=='Q4' else len(comparisons[name+'__vs_Q4']['high_support_tail_review'])) for name,vals in summary.items()}
        roles[role]=role_result;print('MATCHED_BIAS_ROLE_COMPLETE',role,flush=True)
    u.write(out/'SUMMARY_V1.json',dict(status='COMPLETE',results=roles,bias=b,bias_model_sha256=u.sha(bmodel),
        zero_bias_frame_action_region_exact=zero_comparisons,all_observable_policy_choices_unchanged=True,
        new_head_fits=0,new_bias_fits=0,new_flow_forwards=0,GPU_used=False,CPU_seconds=time.time()-start))
    print('MATCHED_BIAS_COMPLETE',round(time.time()-start,3),flush=True)

if __name__=='__main__':main()
