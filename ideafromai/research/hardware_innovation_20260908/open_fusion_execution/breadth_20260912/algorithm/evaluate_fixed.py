"""Fresh deployment evaluation of selected matched stage320 students."""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
from parent_network import ParentNetwork,arrays
from train_matched import dump
HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--structures',nargs='+',required=True,choices=['dense','contiguous34','lifting40'])
    p.add_argument('--split',choices=['valid','diverse'],default='valid')
    args=p.parse_args();selected_split=args.split
    out=HERE/('valid825' if selected_split=='valid' else 'reload_diverse10');out.mkdir(parents=True,exist_ok=True)
    args.output=out;net=ParentNetwork(args)
    import torch
    from fixed_structure import LiteralForward
    from run_bn_probe import read_names
    from evaluate_branch_control import evaluate_axis
    sys.path.insert(0,str(net.op/'stage_20260912/algorithm'))
    from run_combinations import Activity
    nbpath=net.op/'accuracy_baseline/source_nb0_valid825.csv'
    if not nbpath.exists():
        # The previous stage already carries this identical825-row source.
        nbpath=net.op/'stage_20260912/algorithm/source_nb0_valid825.csv'
    with nbpath.open() as f:nb0={r['file']:r for r in csv.DictReader(f)}
    names=(read_names(args.data,'valid') if selected_split=='valid' else json.loads((args.root/'algorithm/samples.json').read_text())['valid'][:10])
    if selected_split=='valid':assert len(names)==825 and set(names)==set(nb0)
    args.split=selected_split;activity=Activity(net.model,net.modules)
    run=dict(complete=False,new_training=False,split=selected_split,frames=names,axes={},
        parent='Every selected student inherits the identical ordinary R24+onepass parent, common1024 TRAIN-moment updates, common320 newGT updates.',
        numeric='Reload actual trained signed16/literal-cutoff/signed24-RNE parameters; actual onepass BN and real coarse head.',
        precision_rule='Strictly better same-file/valid-pixel NB0; no +0.005 gate',
        NB0_source=str(nbpath),upstream_checkpoint=str(args.checkpoint),config=str(args.config),
        output_head='Actual coarse preds.2 summed overT, bilinear480x640 align_corners=False; NB0 is the local upstream reproduction with its original final head, not an official-author checkpoint.',
        hardware_cycles=False,old_valid825_not_inherited=True)
    dump(out/'run.json',run)
    try:
        for structure in args.structures:
            stage=HERE/'matched_training'/structure/'stage320'
            params=arrays(stage/'deployed_constants.npz');net.install('ordinary');net.helper.restore()
            helper=LiteralForward(net.controller,net.pair.temporal.theta,params,structure);net.helper=helper
            bn_calls=[];bn_forward=net.bn.forward
            def count_bn(x):
                bn_calls.append(int(x.shape[0]*x.shape[2]*x.shape[3]));return bn_forward(x)
            net.bn.forward=count_bn
            args.output=out/structure;args.output.mkdir(parents=True,exist_ok=True)
            helper.frames.clear();activity.start(names)
            summary=evaluate_axis(args,net.model,net.current,names,structure,progress_tag='MATCHED_FULL '+structure)
            measured=json.loads((args.output/(structure+'_frames.json')).read_text())
            assert len(measured)==len(helper.frames)==len(names)
            assert all(r['valid_pixels']==int(float(nb0[r['file']]['valid_pixels'])) for r in measured)
            baseline=float(np.mean([float(nb0[r['file']]['AEE']) for r in measured]))
            paired=[dict(file=r['file'],valid_pixels=r['valid_pixels'],AEE=r['AEE'],NB0_AEE=float(nb0[r['file']]['AEE']),delta=r['AEE']-float(nb0[r['file']]['AEE'])) for r in measured]
            dump(args.output/'paired_NB0.json',paired)
            actual=activity.finish(helper.frames,[]);actual['onepass_calls']=len(bn_calls)
            actual['onepass_observed_domains']=sorted(set(bn_calls))
            actual['onepass_min_variance']=None;actual['BN_note']='Actual frozen gamma/beta plus full current-domain onepass statistics; min variance not separately captured in this runner.'
            dump(args.output/'activity_summary.json',actual)
            result=dict(complete=True,summary=summary,NB0_AEE=baseline,delta_NB0=summary['AEE_frame_mean']-baseline,
                better_than_NB0=summary['AEE_frame_mean']<baseline,same_frame_set=True,same_per_frame_valid_pixels=True,
                trained_parameters=str(stage/'deployed_constants.npz'),new_GT_steps=320,common_prior_GT_steps=320,
                activity='activity_summary.json',fresh_inference=True)
            run['axes'][structure]=result;dump(args.output/'quality.json',result);dump(out/'run.json',run)
            print('MATCHED_FULL_DONE',structure,json.dumps(result),flush=True)
            activity.active=False;net.release_axis()
        run['complete']=True;dump(out/'run.json',run)
    finally:activity.restore();net.close()


if __name__=='__main__':main()
