"""One native projection W8 case on the old ordinary R24+onepass parent."""
from pathlib import Path
import argparse,json
import numpy as np
from parent_network import ParentNetwork,arrays
from train_matched import dump
HERE=Path(__file__).resolve().parent
MODULE='sttmultires_unet.encoders.swin3d.patch_embed.proj.conv'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--capture-only',action='store_true',help='Replay only the first already evaluated frame to save local endpoints.')
    args=p.parse_args();args.output=HERE/'native_w8_aee';args.output.mkdir(parents=True,exist_ok=True);out=args.output
    net=ParentNetwork(args)
    import torch
    from evaluate_branch_control import evaluate_axis
    names=json.loads((args.root/'algorithm/samples.json').read_text())['valid'][:10]
    package=HERE.parent/'hardware/native_bn_join/native_w8/native_w8.npz';parameters=arrays(package)
    helper=net.install('ordinary');module=net.modules[MODULE];old_weight=module.weight.detach().clone();sample={};call=[0]
    expected=arrays(net.op/'stage_20260912/algorithm/hardware_exports/ordinary/deployed_constants.npz')
    assert all(np.array_equal(helper.export_constants()[k],expected[k]) for k in expected)
    with torch.no_grad():module.weight.copy_(torch.as_tensor(parameters['expanded_weight_fp32'],device=module.weight.device))
    original_stats=net.math.statistics
    def statistics(x,gamma,beta,eps):
        st=original_stats(x,gamma,beta,eps)
        if call[0]<=1:sample['proj_bn_onepass_statistics']=st.copy()
        return st
    net.math.statistics=statistics
    def capture(key,first=False):
        def hook(module,inputs,output):
            if first:call[0]+=1
            if call[0]!=1:return
            if output.ndim==5:output=output[:,0]
            for label,(y,x) in {'corner':(0,0),'interior':(60,80)}.items():sample[label+'_'+key]=output[:,:,y:y+4,x:x+4].detach().cpu().numpy()
        return hook
    hooks=[module.register_forward_hook(capture('proj_conv_fp32',True)),
        net.modules[net.PROJECT+'.norm_layer'].register_forward_hook(capture('proj_norm_fp32')),
        net.modules[net.PROJECT].register_forward_hook(capture('ped_output_fp32'))]
    try:
        args.split='diverse';helper.frames.clear()
        if args.capture_only:
            evaluate_axis(args,net.model,net.current,names[:1],'capture_only',progress_tag='NATIVE_W8_CAPTURE')
            observed=json.loads((out/'capture_only_frames.json').read_text())[0]
            prior=json.loads((out/'native_w8_frames.json').read_text())[0]
            assert observed==prior
            sample['input_file']=np.asarray(names[0])
            np.savez_compressed(out/'first_frame_local_outputs.npz',**sample)
            print('NATIVE_W8_CAPTURE_DONE',flush=True)
            return
        summary=evaluate_axis(args,net.model,net.current,names,'native_w8',progress_tag='NATIVE_W8_AEE')
        measured=json.loads((out/'native_w8_frames.json').read_text())
        parent=json.loads((net.op/'stage_20260912/algorithm/combinations/ordinary/diverse10/combo_frames.json').read_text())
        assert all(a['file']==b['file'] and a['valid_pixels']==b['valid_pixels'] for a,b in zip(measured,parent))
        h9=float(np.mean([r['AEE'] for r in measured[1:]]));p10=float(np.mean([r['AEE'] for r in parent]));p9=float(np.mean([r['AEE'] for r in parent[1:]]))
        result=dict(complete=True,training=False,full_valid825=False,parent='Old ordinary original_ordered24 + onepass; not matched stage320',
            changed_module=MODULE+'.weight',parameters=str(package),source_PED_constants_same_old_parent=True,
            actual_weight_matches_expanded=bool(np.array_equal(module.weight.detach().cpu().numpy(),parameters['expanded_weight_fp32'])),
            summary=summary,AEE_holdout9=h9,delta_parent_diverse10=summary['AEE_frame_mean']-p10,delta_parent_holdout9=h9-p9,
            NB0_diverse10=1.45460286107,NB0_holdout9=1.446425661411,better_than_NB0=summary['AEE_frame_mean']<1.45460286107 and h9<1.446425661411,
            GPU_CPU_reduction_equivalence='Not asserted; first-frame actual local outputs and full-domain onepass statistics provided for followup binding.')
        dump(out/'quality.json',result);dump(out/'activity_ranges.json',helper.range_report(names))
        sample['input_file']=np.asarray(names[0]);np.savez_compressed(out/'first_frame_local_outputs.npz',**sample)
        print('NATIVE_W8_DONE',json.dumps(result),flush=True)
    finally:
        for h in hooks:h.remove()
        with torch.no_grad():module.weight.copy_(old_weight)
        net.math.statistics=original_stats;net.close()


if __name__=='__main__':main()
