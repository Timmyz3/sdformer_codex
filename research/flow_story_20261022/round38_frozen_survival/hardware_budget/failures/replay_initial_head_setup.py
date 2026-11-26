#!/usr/bin/env python3
"""New-input replay of unmodified frozen R34 stream/head; CPU only, no GT."""
import concurrent.futures,hashlib,importlib.util,json,subprocess
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R38=P.parent;O=P/'actual_v1'
R34=P.parents[2]/'flow_story_20261021/round34_deployability'
R33=R34.parent/'round33_predictability'
H=R34/'hardware'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    frozen=read(H/'FREEZE_V2.json')['files']
    for f in ['stream_features_v2.cpp','head_prefix_v2.py']:
        assert sha(H/f)==frozen[f],f
    records=read(R38/'data/LOCAL_INPUT_RECEIPT_V1.json')['records'];assert len(records)==20
    plan=read(R38/'ACTION_PLAN_V1.json');reference=np.load(R38/'FEATURES_AND_SCORES_V1.npz')
    assert [r['target_id'] for r in records]==list(reference['ids'])==[r['target_id'] for r in plan['records']]
    def stream(item):
        index,r=item;tid=r['target_id'];path=Path(r['path']);assert sha(path)==r['sha256']
        d=O/tid;d.mkdir(exist_ok=False);f=d/'feature.f32'
        subprocess.run([str(P/'stream_frozen_rebuilt'),str(path),str(f),str(d/'feature_service.json'),str(d/'full.trace') if index==0 else '-','rowband','0'],check=True)
        x=np.fromfile(f,dtype=np.float32);memory=np.fromfile(str(f)+'.ios',dtype=np.float32)
        assert x.shape==(128,) and memory.shape==(524288//4,) and np.array_equal(x,memory[100000:100128])
        # Literal filename in the frozen Head constructor is redirected only by
        # its directory parameter. A symlink aliases the actual same IOS image;
        # it performs no modeled copy and never substitutes the old input bytes.
        (d/'zurich_city_07_a_0010.f32.ios').symlink_to('feature.f32.ios')
        service=read(d/'feature_service.json');zero=int(np.sum((x[::8]==0)&(x[1::8]==0)))
        assert service['feature_ready_tick']==113938624-32*zero
        return dict(target_id=tid,input_path=str(path),input_sha256=r['sha256'],feature_service=service,
                    zero_mass_cells=zero,stream_feature_max_abs_error=float(np.max(np.abs(x-reference['x128_fp32'][index]))),x=x,directory=d)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(stream,enumerate(records)))
    spec=importlib.util.spec_from_file_location('r38_unmodified_frozen_head',H/'head_prefix_v2.py');head=importlib.util.module_from_spec(spec);spec.loader.exec_module(head)
    folded_path=R38/'model_sources/FOLDED_MODEL_V1.json';ordinary_path=R38/'model_sources/ORDINARY_MODEL_128_V1.json'
    assert sha(folded_path)==sha(R34/'numeric/folded_constant_v/actual_v1/FOLDED_MODEL_V1.json')
    assert sha(ordinary_path)==sha(R34/'ordinary/actual_v1/MODEL_128_V1.json')
    folded=read(folded_path)['ridge'];ordinary=read(ordinary_path)
    dyn={int(x['action']):4*int(x['dynamic_affine_BN_ticks']) for x in read(R33/'hardware/results/SUMMARY_V1.json')['candidates']}
    static=4*read(H/'actual_v2/STATIC_360_V1.json')['result']['affine_plus_BN_completion_ticks'];assert static==1032538860
    snapshots=[]
    for i,r in enumerate(rows):
        head.O=r['directory'];x=r.pop('x');models={}
        for name in ['folded128','ordinary128']:
            if name=='folded128':rec,params=head.execute(None,x,'X128_folded_constant_v',folded=folded)
            else:rec,params=head.execute(None,x,'direct_residual_g1',calibrated=ordinary)
            scores=np.asarray(rec['scores'],np.float32);action=rec['fixed_action']
            primary=int(reference['choice__'+name+'_fp32'][i]);assert action==primary,(r['target_id'],name,'stream action differs from primary')
            assert plan['records'][i]['policy_actions'][name+'_fp32_fixed6']=='upgrade_'+str(action)
            # A separate explicit numeric readout verifies actual IOS-fed head.
            if name=='folded128':expected,_=head.num.affine_fp32(folded,x[None])
            else:
                aux,_=head.num.affine_fp32(ordinary['family']['auxiliary'],x[None]);products=head.num.products(aux,1)
                aug=np.concatenate((x[None],aux,products.reshape(1,-1)),axis=1)
                residual,_=head.num.affine_fp32(ordinary['family']['residual'][str(ordinary['selection']['alpha'])],aug)
                expected=head.num.signed(aux,1)+np.float32(ordinary['selection']['shrink'])*residual
            assert np.array_equal(scores,expected[0])
            events=rec.pop('service_events');assert next(e for e in events if e['space']=='IOS' and not e['write'])['address']==400000
            (r['directory']/(name+'_head_events.json')).write_text(json.dumps(events,indent=2)+'\n')
            (r['directory']/(name+'_parameters.bin')).write_bytes(params)
            prefix=r['feature_service']['feature_ready_tick']+rec['action_ready_tick'];budget=prefix+dyn[action]
            assert budget>=static
            models[name]=dict(actual_head=rec,primary_action=primary,stream_action_equal_primary=True,
                score_max_abs_error_to_primary=float(np.max(np.abs(scores-reference[name+'_fp32'][i]))),
                prefix_ticks=prefix,dynamic_four_isolated_ticks=dyn[action],scoped_total=budget,
                static360_four_isolated_ticks=static,static360_remaining_ticks=budget-static,
                parameter_sha256=hashlib.sha256(params).hexdigest())
        snapshots.append(x);r['directory']=str(r['directory']);r['models']=models
    # A separate, universally valid guard bound does not rely on observed inputs.
    universal=(113938624-32*16)+8599+min(dyn[a] for a in range(6))-static
    assert universal==184511
    output=dict(status='PASS_20_ACTUAL_FROZEN_PREFIX_REPLAYS',rows=rows,
        min_folded_static_margin=min(r['models']['folded128']['static360_remaining_ticks'] for r in rows),
        max_folded_static_margin=max(r['models']['folded128']['static360_remaining_ticks'] for r in rows),
        all_possible_16_zero_mass_cells_and_six_action_lower_margin=universal,
        CPU_only=True,GT_read=False,new_fits=False,old_sources_modified=False,
        bindings={str(p):sha(p) for p in [Path(__file__),H/'FREEZE_V2.json',P/'stream_frozen_rebuilt',H/'stream_features_v2.cpp',H/'head_prefix_v2.py',
           folded_path,ordinary_path,R38/'ACTION_PLAN_V1.json',R38/'FEATURES_AND_SCORES_V1.npz',R38/'data/LOCAL_INPUT_RECEIPT_V1.json']},
        scope=['Actual per-identity frozen X128 feature service, same IOS byte image fed to unmodified head, no old mean substituted.',
               'FP32 streaming reduction may differ from NumPy primary values; actual selected actions checked for both policies.',
               'Prefix paid once plus four isolated conv services; no stem/whole-network latency or free overlap claimed.',
               'The original scalar RF/routing/FPU timing assumptions remain unchanged; no new RTL or DC claimed for X128.',
               'No static360 quality inferred; new GPU quality collection is separately owned.'])
    np.savez(O/'STREAM_FEATURES_V1.npz',ids=np.asarray([r['target_id'] for r in rows]),x128=np.stack(snapshots))
    (O/'ACTUAL_PREFIX_BUDGET_V1.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('rows','bindings')},indent=2))
if __name__=='__main__':main()
