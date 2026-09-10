"""Fixed second-stage request-aware recovery of all five factor controls.

Predeclared once: same rank/masks/256 updates; gamma=3; train-only moments
refit every64 updates. No lambda sweep, forced private activation, AEE, or
latency claim. First records the unchanged stage-one factor students.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from factor_reference import STRUCTURES, initialize
from train_factors import load_data, FactorModel, batch, evaluate, HERE, PARTIAL
from demand_completion import calibrate, completion, request_loss, measure, PREFIX, GAMMA


def restore(weight,filename):
    with np.load(filename) as saved:
        structure = str(saved['structure'].item())
        initial = initialize(weight,structure)
        initial.update(u=saved['u'].copy(),v=saved['v'].copy())
        model = FactorModel(initial)
        with torch.no_grad():
            model.logits.copy_(torch.from_numpy(saved['logits']))
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--capture',type=Path,default=PARTIAL/'capture.pt')
    p.add_argument('--valid-source',type=Path,default=PARTIAL/'integer_valid10')
    p.add_argument('--operator',type=Path,default=PARTIAL/'shared_column_deployment_source.npz')
    p.add_argument('--temporal',choices=['native','row34','common3'],default='common3')
    p.add_argument('--initial',type=Path,default=HERE/'fit_train16')
    p.add_argument('--steps',type=int,default=256)
    p.add_argument('--structures',nargs='+',choices=STRUCTURES,default=list(STRUCTURES))
    p.add_argument('--request-weight',type=float,default=0.10)
    p.add_argument('--batch-groups',type=int,default=8)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',default='cpu')
    p.add_argument('--measure-only',action='store_true')
    p.add_argument('--output',type=Path,default=HERE/'demand_train16')
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(910)
    data,op,a,b,theta,yscale,mscale,rate,groups = load_data(args)
    constants = dict(a=a.to(args.device),b=b.to(args.device),theta=theta,
        bn_scale=torch.tensor(op['bn_scale'],dtype=torch.float32,device=args.device),
        bn_bias=torch.tensor(op['bn_bias'],dtype=torch.float32,device=args.device),
        y_scale=yscale.to(args.device),margin_scale=mscale.to(args.device),rate=rate.to(args.device))
    source_theta = float(op['source_theta'])
    weight = op['weight'].reshape(96,864).T
    args.output.mkdir(parents=True,exist_ok=True)
    generator = torch.Generator().manual_seed(910)
    batches = torch.randint(len(data['train']['y']),(args.steps,args.batch_groups),generator=generator)
    result = dict(scope='local train16/valid4, same factor student and same common3 full temporal function',
        policy=dict(prefix=list(PREFIX),gamma=GAMMA,kind='one-shot per-gate statistical completion',
            moments='per-axis current train16 post-BN Y means/covariance; refit at 0/64/128/192 and after256',
            empty_source='known Y=original fixed BN bias; same compile simplification for every axis',
            truth='each axis own full A@newY+b-theta; original captured Y only distillation target',
            no_guarantee='statistical radius, not a strict bound'),
        training=dict(steps=args.steps,batch_native_P4=args.batch_groups,optimizer='Adam',lr=0.002,seed=910,
            fixed_loss=f'normalized Y MSE + .125 balanced full BCE + .125 balanced mixed-completion BCE + {args.request_weight} U_J4_request/full_one_scan',
            request_weight=args.request_weight,
            derivative='hard-topK mask STE; sigmoid acceptance, max over source T/P and actual V support',
            budget='same R96/active48 and rank48 compact controls as stage1',
            moment_gradient='detached train-only statistics; no validation calibration',
            trainable='U,V,allowed spatial mask only; no A/bias/theta/gamma training'),
        numeric='CPU FP32 factor/BN/PSN reference, not prior INT8 or frozen FP32 proof',
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        train=data['train']['files'],valid=data['valid']['files'],axes={})
    (args.output/'definition.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    started = time.monotonic()
    baseline = dict(scope='stage-one factors, before adding the demand loss',policy=result['policy'],axes={})
    # All five baseline measurements are completed and written before any
    # second-stage optimizer runs, so these controls cannot be overwritten.
    for axis in args.structures:
        model = restore(weight,args.initial/(axis+'.npz')).to(args.device)
        moments = calibrate(model,data['train'],constants,args.device,source_theta)
        baseline['axes'][axis] = measure(model,data['valid'],constants,moments,args.device,source_theta)
        (args.output/'stage1_completion.json').write_text(json.dumps(baseline,ensure_ascii=False,indent=2)+'\n')
        print('STAGE1',axis,json.dumps(baseline['axes'][axis]),flush=True)
    if args.measure_only:
        return
    for axis in args.structures:
        model = restore(weight,args.initial/(axis+'.npz')).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(),lr=0.002)
        history = []
        for step,ids in enumerate(batches):
            if step%64==0:
                moments = calibrate(model,data['train'],constants,args.device,source_theta)
            model.train()
            item = batch(data['train'],ids,args.device,source_theta)
            state = completion(model,item,constants,moments,soft=True)
            y_loss = ((state['y']-item['y'])/constants['y_scale']).square().mean()
            expected = item['target'].ge(0)
            weights = torch.where(expected,0.5/constants['rate'],0.5/(1-constants['rate']))
            temp = (0.25*constants['margin_scale']).clamp_min(.025)
            full_probability = torch.sigmoid(state['full']/temp)
            pred_probability = torch.sigmoid(state['predicted']/temp)
            accept = state['accept_probability']
            mixed_probability = accept*pred_probability+(1-accept)*full_probability
            full_bce = (F.binary_cross_entropy_with_logits(state['full']/temp,expected.float(),reduction='none')*weights).mean()
            mixed_bce = (F.binary_cross_entropy(mixed_probability.clamp(1e-6,1-1e-6),expected.float(),reduction='none')*weights).mean()
            work = request_loss(model,item,state)
            loss = y_loss+.125*full_bce+.125*mixed_bce+args.request_weight*work
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            with torch.no_grad():
                model.v.mul_(model.connectivity)
            if step%64==0 or step==args.steps-1:
                row=dict(step=step+1,total=float(loss.detach()),y_mse=float(y_loss.detach()),
                    full_bce=float(full_bce.detach()),mixed_bce=float(mixed_bce.detach()),U_J4_soft_requests=float(work.detach()))
                history.append(row)
                print('TRAIN_DEMAND',axis,json.dumps(row),flush=True)
        moments = calibrate(model,data['train'],constants,args.device,source_theta)
        train_full = evaluate(model,data['train'],constants,args.device,source_theta)
        valid_full = evaluate(model,data['valid'],constants,args.device,source_theta)
        valid_completion = measure(model,data['valid'],constants,moments,args.device,source_theta)
        arrays = model.export()
        arrays.update(a=a.numpy(),temporal_bias=b.numpy(),theta_source=np.array(source_theta),
            theta_output=np.array(theta),bn_scale=op['bn_scale'],bn_bias=op['bn_bias'],
            completion_mean=moments['mean'].cpu().numpy(),completion_covariance=moments['covariance'].cpu().numpy(),
            prefix=np.array(PREFIX),gamma=np.array(GAMMA))
        np.savez_compressed(args.output/(axis+'.npz'),**arrays)
        result['axes'][axis] = dict(train_full=train_full,valid_full=valid_full,
            valid_completion=valid_completion,history=history,
            masks=arrays['masks'].astype(int).tolist(),
            distinct_masks=int(np.unique(arrays['masks'],axis=0).shape[0]),
            active_private_tiles_by_region=arrays['masks'][:,12:].sum(1).tolist() if axis=='hybrid' else None)
        (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print('STAGE2',axis,json.dumps(valid_completion),flush=True)
    result['complete'] = True
    result['wall_seconds'] = time.monotonic()-started
    (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('DONE',result['wall_seconds'],flush=True)


if __name__=='__main__':
    main()
