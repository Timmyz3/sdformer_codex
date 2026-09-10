"""Train a matched 2x2 on saved real FP32 membranes; no network AEE claim.

Prediction sees only three prefix columns. Full Y is a training label and a
reference for fallback. Group-column work is a P4-OR upper envelope, not a
cycle model. The shared-W-stage request count is reported separately.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PATCH = HERE.parent
PREFIX = (2, 3, 7)
BITS = 1 << np.arange(10)


def load_inputs():
    cap = torch.load(PATCH/'partial_completion/capture.pt', map_location='cpu', weights_only=False)
    pars = torch.load(PATCH/'dependency/parameters.pt', map_location='cpu', weights_only=False)
    shared = json.loads((PATCH/'partial_completion/shared_column_result.json').read_text())
    entry = shared['selected']['common3_diagonal_34']
    variants = {
        'row34': {k: torch.as_tensor(v).float() for k, v in pars['row34'].items()},
        'common3': {k: torch.as_tensor(entry[k]).float() for k in ('weight','bias')},
    }
    splits = {}
    for split in ('train', 'valid'):
        rows = [r for r in cap['samples'] if r['split'] == split]
        y = torch.stack([r['Y'] for r in rows]).permute(0,3,2,4,1).contiguous()
        # frame, native group, channel, within-P4, T -> context,H8,P4,T
        y = y.reshape(len(rows),64,12,8,4,10).reshape(-1,8,4,10)
        words = torch.stack([r['source_words'] for r in rows]).long().reshape(-1,864)
        hist = torch.zeros(len(words),1024)
        hist.scatter_add_(1, words, torch.ones_like(words,dtype=torch.float32))
        hist = hist[:,None].expand(-1,12,-1).reshape(-1,1024).contiguous()
        bit_table = torch.tensor(((np.arange(1024)[:,None]&BITS)!=0).astype(np.float32))
        column_visits = hist @ bit_table
        h = (torch.arange(len(y))%12)[:,None]*8+torch.arange(8)[None]
        splits[split] = dict(y=y, hist=hist, visits=column_visits, channels=h,
                             files=[r['file'] for r in rows])
    train_y = splits['train']['y'].reshape(-1,12,8,4,10).permute(1,2,0,3,4).reshape(96,-1,10)
    mean = train_y.mean(1)
    centered = train_y-mean[:,None]
    cov = centered.transpose(1,2) @ centered / train_y.shape[1]
    native = cap['neuron_state']
    theta = float(native['thresh'])
    native_a = native['weight'].float()
    native_b = native['bias'].float().reshape(10)
    for data in splits.values():
        data['teacher_margin'] = data['y'] @ native_a.T + native_b-theta
        data['teacher_gate'] = data['teacher_margin'] >= 0
    return variants, splits, mean, cov, theta, float(cap['metadata']['source_theta'])


def ste_round(x):
    return x+(x.round()-x).detach()


class Completion(nn.Module):
    def __init__(self, params, mean, covariance, theta, prefix=PREFIX):
        super().__init__()
        self.a = nn.Parameter(params['weight'].clone())
        self.b = nn.Parameter(params['bias'].reshape(10).clone())
        self.log_gamma = nn.Parameter(torch.full((10,),float(np.log(3.0))))
        self.register_buffer('mask',params['weight'].ne(0))
        known = torch.zeros(10)
        known[list(prefix)] = 1
        self.register_buffer('known',known)
        self.register_buffer('mean',mean)
        self.register_buffer('covariance',covariance)
        self.register_buffer('word_bits',torch.tensor(((np.arange(1024)[:,None]&BITS)!=0).astype(np.float32)))
        self.theta=theta

    def coefficients(self):
        return ste_round(self.a*16384)/16384*self.mask

    def forward(self, y, channels):
        a=self.coefficients()
        remaining=a*(1-self.known)
        full=y@a.T+self.b-self.theta
        offset=self.mean@remaining.T+self.b-self.theta
        predicted=(y*self.known)@a.T+offset[channels][:,:,None,:]
        variance=torch.einsum('ts,csu,tu->ct',remaining,self.covariance,remaining)
        sigma=variance.clamp_min(1e-6).sqrt()
        radius=sigma[channels][:,:,None,:]*self.log_gamma.exp()
        temperature=(sigma[channels][:,:,None,:]*0.25).clamp_min(0.025)
        accept_soft=torch.sigmoid((predicted.abs()-radius)/temperature)
        accept=predicted.abs()>=radius
        return full,predicted,accept_soft,accept

    def demands(self, continuation):
        # [B,H8,P4,t,s]; maximum is a deterministic OR relaxation.
        by_lane=(continuation[...,None]*self.mask).amax(-2)
        by_lane=by_lane*(1-self.known)+self.known
        return by_lane.mean((1,2)),by_lane.amax((1,2))


def batch(data, ids):
    return {k:data[k][ids] for k in ['y','hist','visits','channels','teacher_margin','teacher_gate']}


def loss_for(model, item, objective):
    full,predicted,soft,_=model(item['y'],item['channels'])
    individual,group=model.demands(1-soft)
    if objective=='packed_word':
        tail=group*(1-model.known)
        word_need=(tail[:,None,:]*model.word_bits[None]).amax(-1)
        prefix_need=(model.word_bits*model.known).amax(-1)
        work=(item['hist']*(prefix_need+word_need)).sum()/item['hist'][:,1:].sum().clamp_min(1)
    else:
        need=individual if objective=='individual' else group
        work=(need*item['visits']).sum()/item['visits'].sum().clamp_min(1)
    teacher=item['teacher_gate'].float()
    # Equal positive/negative contributions prevent a sparse all-zero predictor
    # from looking accurate through its overall gate error alone.
    rate=teacher.mean().detach().clamp(0.005,0.995)
    weights=torch.where(teacher.bool(),0.5/rate,0.5/(1-rate))
    p_full=torch.sigmoid(full/0.25)
    p_pred=torch.sigmoid(predicted/0.25)
    p_mix=soft*p_pred+(1-soft)*p_full
    gate_loss=(F.binary_cross_entropy(p_mix.clamp(1e-6,1-1e-6),teacher,reduction='none')*weights).mean()
    complete_loss=F.mse_loss(full,item['teacher_margin'])
    # Prediction and complete function are both trained against the same native
    # teacher; a free change of the target function cannot hide early errors.
    loss=complete_loss+0.25*gate_loss+0.10*work
    return loss,dict(total=float(loss.detach()),complete_mse=float(complete_loss.detach()),
        balanced_gate_loss=float(gate_loss.detach()),soft_work=float(work.detach()))


@torch.no_grad()
def evaluate(model,data):
    totals=dict(gates=0,teacher_positive=0,full_fp=0,full_fn=0,early_fp=0,early_fn=0,
                changed_from_own_full=0,accepted=0,accepted_wrong=0,
                complete_mse_sum=0.,column_visits_full=0.,column_visits_early=0.,
                shared_w_full=0.,shared_w_prefix_tail=0.,full_psn_terms=0,early_psn_terms=0)
    need_rows=[]
    gates_rows=[]
    bits=torch.tensor(((np.arange(1024)[:,None]&BITS)!=0))
    prefix=model.known.nonzero().flatten().tolist()
    prefix_words=bits[:,prefix].any(1)
    for start in range(0,len(data['y']),128):
        sl=slice(start,start+128)
        full,predicted,_,accept=model(data['y'][sl],data['channels'][sl])
        gate=torch.where(accept,predicted>=0,full>=0)
        truth=data['teacher_gate'][sl]
        _,need=model.demands((~accept).float())
        need=need.bool()
        tail=need.clone();tail[:,prefix]=False
        hist=data['hist'][sl]
        tail_words=(tail[:,None,:]&bits[None]).any(-1)
        totals['gates']+=gate.numel()
        totals['teacher_positive']+=int(truth.sum())
        totals['full_fp']+=int(((full>=0)&~truth).sum())
        totals['full_fn']+=int(((full<0)&truth).sum())
        totals['early_fp']+=int((gate&~truth).sum())
        totals['early_fn']+=int((~gate&truth).sum())
        totals['changed_from_own_full']+=int((gate!=(full>=0)).sum())
        totals['accepted']+=int(accept.sum())
        totals['accepted_wrong']+=int((accept&(gate!=truth)).sum())
        totals['complete_mse_sum']+=float((full-data['teacher_margin'][sl]).square().sum())
        totals['column_visits_full']+=float(data['visits'][sl].sum())
        totals['column_visits_early']+=float((data['visits'][sl]*need).sum())
        totals['shared_w_full']+=float(hist[:,1:].sum())
        totals['shared_w_prefix_tail']+=float((hist*prefix_words).sum()+(hist*tail_words).sum())
        prefix_n=model.mask[:,prefix].sum(-1)
        tail_n=model.mask.sum(-1)-prefix_n
        totals['full_psn_terms']+=gate.numel()//10*int(model.mask.sum())
        totals['early_psn_terms']+=gate.numel()//10*int(prefix_n.sum())+int(((~accept)*tail_n).sum())
        need_rows.append(need.numpy())
        gates_rows.append(gate.numpy())
    n=totals['gates'];pos=totals['teacher_positive'];neg=n-pos
    totals.update(full_gate_error=(totals['full_fp']+totals['full_fn'])/n,
        early_gate_error=(totals['early_fp']+totals['early_fn'])/n,
        full_false_negative_rate=totals['full_fn']/max(pos,1),
        early_false_negative_rate=totals['early_fn']/max(pos,1),
        early_false_positive_rate=totals['early_fp']/max(neg,1),
        early_vs_own_full=totals['changed_from_own_full']/n,
        accepted_fraction=totals['accepted']/n,
        complete_membrane_mse=totals['complete_mse_sum']/n,
        column_visit_ratio=totals['column_visits_early']/totals['column_visits_full'],
        shared_w_ratio=totals['shared_w_prefix_tail']/totals['shared_w_full'],
        psn_term_ratio=totals['early_psn_terms']/totals['full_psn_terms'])
    return totals,np.concatenate(need_rows),np.concatenate(gates_rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=400)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--objectives',nargs='+',choices=['individual','group','packed_word'],default=['individual','group'])
    p.add_argument('--structures',nargs='+',choices=['row34','common3'],default=['row34','common3'])
    p.add_argument('--prefixes-json',type=Path)
    p.add_argument('--output',type=Path,default=HERE/'local_train16')
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(909)
    variants,splits,mean,cov,theta,source_theta=load_inputs()
    selected_prefixes=json.loads(args.prefixes_json.read_text())['selected'] if args.prefixes_json else {}
    args.output.mkdir(parents=True,exist_ok=True)
    generator=torch.Generator().manual_seed(909)
    batches=torch.randint(len(splits['train']['y']),(args.steps,64),generator=generator)
    result=dict(scope='local train16/valid4, 64 native P4 per frame, C96/T10; FP32 norm1 input capture',
        missing='no network AEE; no integer Conv1 train capture; no full layer service/PPA',
        train=splits['train']['files'],valid=splits['valid']['files'],
        prefixes={s:selected_prefixes.get(s,{}).get('prefix',list(PREFIX)) for s in args.structures},
        theta_output=theta,theta_source=source_theta,
        numeric='temporal A Q14 STE; captured FP32 Y; floating bias/radius; not the prior integer student',
        predictor='prefix partial plus train residual mean; train covariance radius and learned gamma[t]',
        objective='full native membrane MSE + 0.25 balanced mixed gate BCE + 0.10 normalized continuation work',
        cost='max unresolved P4/H8 dependency, P4-OR source upper envelope; staged shared W counted separately',
        training=dict(steps=args.steps,batch=64,lr=0.002,seed=909,validation_selection=False),axes={})
    started=time.monotonic()
    for structure,params in variants.items():
        if structure not in args.structures:
            continue
        for objective in args.objectives:
            name=structure+'_'+objective
            prefix=selected_prefixes.get(structure,{}).get('prefix',list(PREFIX))
            model=Completion(params,mean,cov,theta,prefix=prefix)
            initial,_,_=evaluate(model,splits['train'])
            optim=torch.optim.Adam(model.parameters(),lr=0.002)
            history=[]
            for step,ids in enumerate(batches):
                optim.zero_grad(set_to_none=True)
                loss,metrics=loss_for(model,batch(splits['train'],ids),objective)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                optim.step()
                with torch.no_grad():
                    model.a.mul_(model.mask)
                    model.log_gamma.clamp_(np.log(0.5),np.log(8.))
                if step%100==0 or step==args.steps-1:
                    history.append(dict(step=step+1,**metrics))
                    print(name,step+1,json.dumps(metrics),flush=True)
            train,_,_=evaluate(model,splits['train'])
            valid,need,gates=evaluate(model,splits['valid'])
            a=model.coefficients().detach().numpy()
            arrays=dict(temporal_q14=np.rint(a*16384).astype(np.int16),
                temporal_bias=model.b.detach().numpy(),gamma=model.log_gamma.exp().detach().numpy(),
                mean=mean.numpy(),covariance=cov.numpy(),support=model.mask.numpy(),
                theta_output=np.array(theta),theta_source=np.array(source_theta),prefix=np.array(prefix))
            np.savez_compressed(args.output/(name+'.npz'),**arrays)
            np.savez_compressed(args.output/(name+'_valid_decisions.npz'),need_column=need,gate=gates)
            result['axes'][name]=dict(initial_train=initial,train=train,valid=valid,
                actual_q14_nonzeros=int(np.count_nonzero(a)),rank=int(np.linalg.matrix_rank(a)),
                gamma=model.log_gamma.exp().detach().tolist(),history=history,prefix=list(prefix),
                parameters='34 temporal A values,10 biases,10 gamma; shared train moments compiled per t/h')
            (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            print('VALID',name,json.dumps(valid),flush=True)
    result['wall_seconds']=time.monotonic()-started
    result['complete']=True
    (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('DONE',result['wall_seconds'],flush=True)


if __name__=='__main__':
    main()
