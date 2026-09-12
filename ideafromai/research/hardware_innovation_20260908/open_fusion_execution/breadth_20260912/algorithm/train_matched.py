"""Matched fixed-forward source training; only the backward is surrogate."""
from pathlib import Path
import argparse,gc,json,time
import numpy as np
from parent_network import ParentNetwork,arrays
HERE=Path(__file__).resolve().parent
SEED=912


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--smoke-only',action='store_true');p.add_argument('--structures',nargs='+',default=['dense','contiguous34','lifting40'])
    args=p.parse_args();args.output=HERE/('smoke' if args.smoke_only else 'matched_training');args.output.mkdir(parents=True,exist_ok=True)
    output=args.output;net=ParentNetwork(args)
    import torch
    import torch.nn.functional as F
    from fixed_structure import LiteralForward,QATForward,OnepassSTE
    from train_consumer_recovery import read_train16,release_model_graphs
    from run_bn_probe import input_frame
    from evaluate_stage2_deployment import CoarseReady
    from evaluate_branch_control import evaluate_axis
    from spikingjelly.activation_based import functional
    from adapter import fp32_matmul
    train=read_train16(net.latent/'flow_train_list.json');rng=np.random.default_rng(SEED)
    schedules=[(64,[train[int(i)] for i in np.concatenate([rng.permutation(16) for _ in range(4)])]),
        (320,json.loads((net.chain/'shared_temporal_recovery128x256/train128.json').read_text())['train_schedule'])]
    names=json.loads((args.root/'algorithm/samples.json').read_text())['valid'][:10]
    run=dict(complete=False,common_parent='ordinary original_ordered24 + onepass; prior320 GT updates inherited by every arm',
        initialization=str(HERE/'initialization/run.json'),training=dict(seed=SEED,stages=[64,256],
        optimizer='Fresh Adam1e-4 each stage; final endpoint, no validation selection',
        schedules={str(n):s for n,s in schedules},loss='Real valid-pixel coarse-head GT sqrt(sum(error^2)+1e-6); no teacher'),
        arithmetic='Signed16 frozen grids; signed24/f14 RNE/saturate; exact dyadic dot products. Same hard forward as literal deployment; STE backward only.',
        evaluation_files=names,axes={})
    dump(output/'run.json',run)
    def preserve(module,inputs,result):net.current['differentiable_flow']=result.sum(0)
    hook=net.modules['sttmultires_unet.preds.2'].register_forward_hook(preserve,prepend=True)
    def release():
        if net.controller is not None:
            release_model_graphs(net.modules,net.current,net.controller)
        net.current.pop('differentiable_flow',None)
        functional.reset_net(net.model)
    def flow(x):
        functional.reset_net(net.model);net.current.pop('flow',None);net.current.pop('differentiable_flow',None)
        try:net.model(x)
        except CoarseReady:
            result=net.current.pop('differentiable_flow');net.current.pop('flow',None)
            return F.interpolate(result,(480,640),mode='bilinear',align_corners=False)
        raise RuntimeError('Missing actual coarse-head exit')
    def gradients(q):
        return {k:dict(present=v.grad is not None,finite=bool(torch.isfinite(v.grad).all()) if v.grad is not None else False,
            nonzero=int(v.grad.count_nonzero()) if v.grad is not None else 0,
            maximum=float(v.grad.abs().max()) if v.grad is not None else None) for k,v in q.parameters.items()}
    def qat_bn(x):return OnepassSTE.apply(x,net.math,net.bn_gamma,net.bn_beta,net.bn.eps)
    def replace(cls,constants,structure):
        net.helper.restore();net.helper=cls(net.controller,net.pair.temporal.theta,constants,structure);return net.helper
    def measure_hard(constants,structure,x):
        helper=replace(LiteralForward,constants,structure);values={}
        def bits(name):
            def capture(module,inputs,result):values[name]=result.detach().ne(0).cpu()
            return capture
        def ped(module,inputs,result):values['PED24']=(result.detach().double()*16384).to(torch.int32).cpu()
        hooks=[net.modules[net.SOURCE_SN].register_forward_hook(bits('source_g')),
            net.sn2.register_forward_hook(bits('preview_g')),
            net.modules[net.CONSUMER_SN].register_forward_hook(bits('consumer_g')),
            net.modules[net.PROJECT+'.conv_res'].register_forward_hook(ped)]
        try:
            with torch.no_grad():result=flow(x).detach()
        finally:
            for h in hooks:h.remove()
        release();return result,values
    def smoke(constants,structure):
        x,label,valid=input_frame(args.data,train[0]);original_bn=net.bn.forward
        parent_prediction=None
        if structure=='dense':
            with torch.no_grad():parent_prediction=flow(x).detach()
            release()
        expected,expected_values=measure_hard(constants,structure,x)
        parent_delta=float((expected-parent_prediction).abs().max()) if parent_prediction is not None else None
        del parent_prediction
        q=replace(QATForward,constants,structure);net.bn.forward=qat_bn
        differences={}
        def compare(name,is_bits=True):
            def capture(module,inputs,result):
                actual=(result.detach().ne(0) if is_bits else (result.detach().double()*16384).to(torch.int32)).cpu()
                differences[name]=dict(values=actual.numel(),differences=int((actual!=expected_values[name]).sum()))
            return capture
        hooks=[net.modules[net.SOURCE_SN].register_forward_hook(compare('source_g')),
            net.sn2.register_forward_hook(compare('preview_g')),
            net.modules[net.CONSUMER_SN].register_forward_hook(compare('consumer_g')),
            net.modules[net.PROJECT+'.conv_res'].register_forward_hook(compare('PED24',False))]
        torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        try:
            prediction=flow(x);error=prediction.permute(0,2,3,1)[valid]-label.permute(0,2,3,1)[valid]
            loss=torch.sqrt(error.square().sum(-1)+1e-6).mean()
            with fp32_matmul():loss.backward()
            torch.cuda.synchronize();g=gradients(q)
            row=dict(file=train[0],optimizer_updates=0,flow_max_abs=float((prediction.detach()-expected).abs().max()),
                dense_literal_vs_original_R24_helper_max_abs=parent_delta,
                intermediate_comparisons=differences,loss=float(loss.detach()),gradients=g,
                seconds_forward_backward=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                parameter_counts={k:v.numel() for k,v in q.parameters.items()})
            row['pass']=row['flow_max_abs']==0 and (parent_delta is None or parent_delta==0) and all(v['differences']==0 for v in differences.values()) and all(v['present'] and v['finite'] and v['nonzero']>0 for v in g.values())
        finally:
            for h in hooks:h.remove()
        for v in q.parameters.values():v.grad=None
        release();del x,label,valid,prediction,error,loss,expected,expected_values
        return q,row,original_bn
    def evaluate(constants,structure,directory,original_bn):
        helper=replace(LiteralForward,constants,structure);net.bn.forward=original_bn
        args.output=directory;directory.mkdir(parents=True,exist_ok=True)
        helper.frames.clear();summary=evaluate_axis(args,net.model,net.current,names,structure,progress_tag='MATCHED_AEE')
        rows=json.loads((directory/(structure+'_frames.json')).read_text());holdout=rows[1:]
        quality=dict(diverse10_AEE=summary['AEE_frame_mean'],holdout9_AEE=float(np.mean([r['AEE'] for r in holdout])),
            diverse10_pixels=sum(r['valid_pixels'] for r in rows),holdout9_pixels=sum(r['valid_pixels'] for r in holdout),
            NB0_diverse10=1.45460286107,NB0_holdout9=1.446425661411,
            pass_diverse10=summary['AEE_frame_mean']<1.45460286107,pass_holdout9=np.mean([r['AEE'] for r in holdout])<1.446425661411)
        dump(directory/'quality.json',quality);dump(directory/'activity_ranges.json',helper.range_report(names))
        dump(directory/'deployment_metadata.json',helper.metadata);release();args.output=output
        return quality
    try:
        for structure in args.structures:
            torch.manual_seed(SEED);np.random.seed(SEED)
            net.install('ordinary');constants=arrays(HERE/'initialization'/(structure+'_constants.npz'))
            q,check,original_bn=smoke(constants,structure)
            row=dict(check=check,stages={},complete=False);run['axes'][structure]=row;dump(output/'run.json',run)
            print('MATCHED_CHECK',structure,json.dumps(check),flush=True)
            if not check['pass']:raise RuntimeError('Literal/QAT hard function or active gradient mismatch: '+structure)
            if args.smoke_only:
                row['complete']=True;net.bn.forward=original_bn;net.release_axis();continue
            for cumulative,schedule in schedules:
                directory=output/structure/('stage'+str(cumulative));directory.mkdir(parents=True,exist_ok=True)
                optimizer=torch.optim.Adam(list(q.parameters.values()),lr=1e-4);history=[];started=time.monotonic()
                stage=dict(complete=False,history=history);row['stages'][str(cumulative)]=stage
                for step,name in enumerate(schedule,1):
                    x,label,valid=input_frame(args.data,name);optimizer.zero_grad(set_to_none=True)
                    prediction=flow(x);error=prediction.permute(0,2,3,1)[valid]-label.permute(0,2,3,1)[valid]
                    loss=torch.sqrt(error.square().sum(-1)+1e-6).mean()
                    with fp32_matmul():loss.backward()
                    g=gradients(q)
                    if not bool(torch.isfinite(loss)) or not all(v['present'] and v['finite'] for v in g.values()):raise RuntimeError('Missing/nonfinite gradient')
                    entry=dict(step=step,file=name,loss=float(loss.detach()),valid_pixels=int(valid.sum()))
                    optimizer.step();history.append(entry);del x,label,valid,prediction,error,loss;release()
                    if step==1 or step%8==0:
                        print('MATCHED_TRAIN',structure,cumulative,step,entry['loss'],flush=True);dump(output/'run.json',run)
                stage['training_wall_seconds']=time.monotonic()-started
                constants=q.literal_constants();np.savez_compressed(directory/'deployed_constants.npz',**constants)
                torch.save({k:v.detach().cpu() for k,v in q.parameters.items()},directory/'master_parameters.pt')
                # Preserve the same optimizer-free master values across stages.
                masters={k:v.detach().clone() for k,v in q.parameters.items()}
                x,_,_=input_frame(args.data,train[0],targets=False)
                with torch.no_grad():trained_hard=flow(x).detach()
                release();net.bn.forward=original_bn
                deployed_hard,_=measure_hard(constants,structure,x)
                stage['trained_QAT_vs_reloaded_literal_flow_max_abs']=float((trained_hard-deployed_hard).abs().max())
                if stage['trained_QAT_vs_reloaded_literal_flow_max_abs']!=0:raise RuntimeError('Trained/exported fixed hard functions differ')
                del x,trained_hard,deployed_hard
                stage['quality']=evaluate(constants,structure,directory,original_bn)
                q=replace(QATForward,constants,structure)
                with torch.no_grad():
                    for k,v in masters.items():q.parameters[k].copy_(v)
                net.bn.forward=qat_bn;stage['complete']=True;dump(output/'run.json',run);del optimizer,masters
            row['complete']=True;net.bn.forward=original_bn;net.release_axis();gc.collect();torch.cuda.empty_cache()
        run['complete']=True;dump(output/'run.json',run)
    finally:hook.remove();net.close()
    print('MATCHED_DONE',flush=True)


if __name__=='__main__':main()
