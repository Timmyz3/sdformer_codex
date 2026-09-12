"""Two predeclared signed-PoT source approximations, fresh diverse10 only."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from parent_network import ParentNetwork,arrays
from train_matched import dump
HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();out=HERE/'source_constant_aee';out.mkdir(parents=True,exist_ok=True);args.output=out
    net=ParentNetwork(args)
    from fixed_structure import LiteralForward
    from evaluate_branch_control import evaluate_axis
    full=net.lift/'schedule_compare_same_port/full_chain';sys.path.insert(0,str(full));sys.path.insert(0,str(full.parent))
    from capture_reference import geometry,OUTPUTS
    from endpoint_observer import SmallCapture
    names=json.loads((args.root/'algorithm/samples.json').read_text())['valid'][:10]
    run=dict(complete=False,new_training=False,full_valid825=False,rule='Fixed nearest at most two signed powers of two per source coefficient, same exponent/cutoff/RNE/consumers.',files=names,axes={})
    dump(out/'run.json',run)
    try:
        for structure in ['dense','lifting40']:
            package=HERE.parent/'source_constant_probe'/structure;params=arrays(package/'deployed_constants.npz');gold=arrays(package/'cpu_gold.npz')
            net.install('ordinary');net.helper.restore();helper=LiteralForward(net.controller,net.pair.temporal.theta,params,structure);net.helper=helper
            exports=out/structure/'hardware_exports';exports.mkdir(parents=True,exist_ok=True)
            latest={};original_statistics=net.math.statistics
            def statistics(x,gamma,beta,eps):
                st=original_statistics(x,gamma,beta,eps);latest['stats']=st.copy();return st
            net.math.statistics=statistics
            observer=SmallCapture(net.model,net.modules,helper,net.pair.temporal.theta,names[:1],exports,True,
                structure=structure,latest=latest,function_label=structure+'_source_twopot',stop_after_first=True)
            checks={};calls=[0]
            def source(module,inputs,output):
                calls[0]+=1
                if calls[0]!=1:return
                for label,origin in OUTPUTS.items():
                    geo=geometry(origin);y,x=geo['source_origin'];h,w=geo['source_shape']
                    i=helper.i[:,:,y:y+h,x:x+w].detach().cpu().numpy()
                    g=output[:,0,:,y:y+h,x:x+w].detach().ne(0).cpu().numpy()
                    row=dict(values=g.size,I24_differences=int(np.count_nonzero(i!=gold[label+'_I24'])),source_gate_differences=int(np.count_nonzero(g!=gold[label+'_source_gate'])))
                    checks[label]=row
                    assert row['I24_differences']==row['source_gate_differences']==0
            hook=net.modules[net.SOURCE_SN].register_forward_hook(source)
            try:
                args.output=out/structure;args.output.mkdir(parents=True,exist_ok=True);args.split='diverse';helper.frames.clear()
                summary=evaluate_axis(args,net.model,net.current,names,structure,progress_tag='SOURCE_CONSTANT_AEE')
                measured=json.loads((args.output/(structure+'_frames.json')).read_text())
                parent=json.loads((HERE/'matched_training'/structure/'stage320'/(structure+'_frames.json')).read_text())
                assert all(a['file']==b['file'] and a['valid_pixels']==b['valid_pixels'] for a,b in zip(measured,parent))
                h9=float(np.mean([r['AEE'] for r in measured[1:]]));p10=float(np.mean([r['AEE'] for r in parent]));p9=float(np.mean([r['AEE'] for r in parent[1:]]))
                result=dict(complete=True,summary=summary,AEE_holdout9=h9,delta_parent_diverse10=summary['AEE_frame_mean']-p10,delta_parent_holdout9=h9-p9,
                    NB0_diverse10=1.45460286107,NB0_holdout9=1.446425661411,better_than_NB0=summary['AEE_frame_mean']<1.45460286107 and h9<1.446425661411,
                    GPU_source_vs_CPU=checks,parameters=str(package/'deployed_constants.npz'),fresh_function=True)
                np.savez_compressed(exports/'deployed_constants.npz',**params)
                dump(exports/'index.json',dict(complete=True,rows=observer.rows,structure=structure,function_label=structure+'_source_twopot'))
                result['hardware_exports']='hardware_exports'
                dump(args.output/'quality.json',result);dump(args.output/'activity_ranges.json',helper.range_report(names));run['axes'][structure]=result;dump(out/'run.json',run)
                print('SOURCE_CONSTANT_DONE',structure,json.dumps(result),flush=True)
            finally:hook.remove();observer.restore();net.math.statistics=original_statistics;net.release_axis()
        run['complete']=True;dump(out/'run.json',run)
    finally:net.close()


if __name__=='__main__':main()
