"""Replay new fixed masks on the previously completed fastest mode3 RTL."""
from pathlib import Path
import importlib.util
import json
import subprocess
import numpy as np
from cost import features, measure

HERE=Path(__file__).resolve().parent
NEW=HERE.parent
PREV=NEW.parent/'r0_execution_trials_20260913'


def main():
    spec=importlib.util.spec_from_file_location('native_fixture',PREV/'native_sparse/run.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    helper.HERE=HERE
    cap=np.load(PREV/'data_and_quality/r0_contiguous_t10.npz')
    masks={'dense':np.ones((12,24),dtype=bool)}
    for arm,path in [
        ('block_magnitude25',NEW/'data/block_magnitude25.npz'),
        ('cin_magnitude25',NEW/'data/cin_magnitude25.npz'),
        ('cin_fullcost25',NEW/'data/cin_fullcost25.npz'),
        ('mixed_retirement25',HERE/'mixed_retirement25_q16.npz')]:
        z=np.load(path);masks[arm]=z['live']
        assert np.array_equal(z['weight_q16'],cap['weight_q16']*z['live'].repeat(8,0).repeat(4,1)[:,:,None,None])
    fixtures=[];estimates={}
    for arm,live in masks.items():
        for tile,source in enumerate(cap['source_bits']):
            name=f'{arm}_tile{tile}'
            path=helper.fixture(name,source,cap['weight_q16'],live,
                {'arm':arm,'tile':tile,'training_frame':'thun_00_a_0002.npy','evaluation_frame':str(cap['frame'])},
                origin=cap['input_origin_yx'][tile])
            fixtures.append((arm,tile,name,path))
            estimates[name]=measure(live,features(source[None],cap['input_origin_yx'][tile:tile+1]))
    print('FIXTURES_READY',len(fixtures),flush=True)
    rows=[]
    binary=PREV/'consumer_enumeration/obj_dir/Vnative_sparse'
    for arm,tile,name,path in fixtures:
        for stall in (0,1):
            result=subprocess.run([str(binary),str(path),'3',str(stall)],capture_output=True,text=True,check=True)
            for line in result.stdout.splitlines():
                row=json.loads(line);row.update(arm=arm,tile=tile,fixture=name)
                pred=estimates[name]
                for key,value in pred.items():
                    if key=='live_source_groups':continue
                    actual=row[key]
                    if key=='cycles':actual-=row['source_stalls']+row['weight_stalls']+row['output_stalls']
                    assert actual==value,(name,key,actual,value)
                rows.append(row)
        print('RTL_PASS',name,flush=True)
    (HERE/'reference_results.json').write_text(json.dumps(rows,indent=2)+'\n')
    summary={}
    for arm in masks:
        rs=[r for r in rows if r['arm']==arm and r['stall']==0 and r['command']==0]
        keys=('cycles','source_words','weight_words','psum_reads','psum_writes','sum_issues','update_issues')
        s={k:sum(r[k] for r in rs) for k in keys}
        s['cycles_with_fresh_source_origin']=s['cycles']+1537*len(rs)
        s['retired_C4']=np.where(~masks[arm].any(axis=0))[0].tolist()
        summary[arm]=s
    report=dict(complete=True,runs=len(rows),output_comparisons=sum(r['outputs'] for r in rows),
        cycle_source='actual Verilator previously compiled mode3; same unchanged source and TB',
        matching_core_model=True,summary=summary,
        scope='8 evaluation tiles; calibration separated in another train sequence; not full layer/PPA',
        first_static_weight_mask_load=10656)
    (HERE/'reference_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
