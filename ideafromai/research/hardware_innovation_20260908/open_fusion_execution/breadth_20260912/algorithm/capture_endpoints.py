"""Two original halos of the three final fixed students; no new fitting."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from parent_network import ParentNetwork,arrays
from train_matched import dump
HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();args.output=HERE/'hardware_exports';args.output.mkdir(parents=True,exist_ok=True);out=args.output
    net=ParentNetwork(args)
    import torch
    from fixed_structure import LiteralForward
    from evaluate_branch_control import evaluate_axis
    full=net.lift/'schedule_compare_same_port/full_chain'
    sys.path.insert(0,str(full));sys.path.insert(0,str(full.parent))
    from endpoint_observer import SmallCapture
    name=json.loads((args.root/'algorithm/samples.json').read_text())['valid'][0]
    index=dict(complete=False,file=name,scope='Two original corner/interior halos, actual post-training literal fixed forward; no new AEE population or parameters.',
        raw_I24='Copied at actual SOURCE_SN hook before any residual update; not the legacy train_capture extra source_I24 field.',axes={})
    try:
        for structure in ['dense','contiguous34','lifting40']:
            directory=out/structure;directory.mkdir(parents=True,exist_ok=True)
            constants=arrays(HERE/'matched_training'/structure/'stage320/deployed_constants.npz')
            net.install('ordinary');net.helper.restore();helper=LiteralForward(net.controller,net.pair.temporal.theta,constants,structure);net.helper=helper
            latest={};original_statistics=net.math.statistics
            def observe_statistics(x,gamma,beta,eps):
                st=original_statistics(x,gamma,beta,eps);latest['stats']=st.copy();return st
            net.math.statistics=observe_statistics
            observer=SmallCapture(net.model,net.modules,helper,net.pair.temporal.theta,[name],directory,True,structure=structure,latest=latest)
            try:
                args.output=directory;args.split='diverse'
                summary=evaluate_axis(args,net.model,net.current,[name],structure,progress_tag='FINAL_CAPTURE')
                gold=json.loads((HERE/'matched_training'/structure/'stage320'/(structure+'_frames.json')).read_text())[0]
                observed=json.loads((directory/(structure+'_frames.json')).read_text())[0]
                assert observed['file']==gold['file'] and observed['valid_pixels']==gold['valid_pixels'] and observed['aee_sum']==gold['aee_sum']
                np.savez_compressed(directory/'deployed_constants.npz',**constants)
                np.savez_compressed(directory/'BN_parameters.npz',gamma=net.bn_gamma,beta=net.bn_beta,eps=np.asarray(net.bn.eps))
                dump(directory/'deployment_metadata.json',helper.metadata)
                index['axes'][structure]=dict(complete=True,capture=observer.rows,actual_first_frame_AEE=summary['AEE_frame_mean'],
                    exact_same_trained_endpoint_AEE=True,raw_source_capture_at_source_hook=True)
                dump(out/'index.json',index)
            finally:observer.restore();net.math.statistics=original_statistics;net.release_axis()
        index['complete']=True;dump(out/'index.json',index)
    finally:net.close()
    print('FINAL_ENDPOINT_CAPTURES_DONE',flush=True)


if __name__=='__main__':main()
