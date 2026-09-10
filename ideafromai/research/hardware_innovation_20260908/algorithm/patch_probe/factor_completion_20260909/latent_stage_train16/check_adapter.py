"""Check saved compact-table adapter on all four captured validation frames."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from latent_stage import (HERE,FACTOR,PARTIAL,load_data,batch,read_model,completion)
from adapter import LatentTemporal,LatentPair


def main():
    torch.set_num_threads(4)
    args=argparse.Namespace(source=FACTOR.parent/'joint_completion_20260909/full_capture4/capture',
        capture=PARTIAL/'capture.pt',valid_source=PARTIAL/'integer_valid10',
        operator=PARTIAL/'shared_column_deployment_source.npz',temporal='common3')
    data,op,a,b,theta,ys,ms,rate,groups=load_data(args)
    constants=dict(a=a,b=b,theta=theta,bn_scale=torch.tensor(op['bn_scale'],dtype=torch.float32),
        bn_bias=torch.tensor(op['bn_bias'],dtype=torch.float32),y_scale=ys,margin_scale=ms,rate=rate)
    result=json.loads((HERE/'result.json').read_text())
    output=dict(scope='CPU FP32 compact-table and missing-private-Z replay, all valid4 sampled native P4; no GPU or RTL',axes={})
    names=[name for stage in ('reference','stage1','stage2') for name in result[stage]]
    with torch.no_grad():
        for name in names:
            with np.load(HERE/(name+'.npz')) as p:
                arrays={key:p[key].copy() for key in p.files}
            model=read_model(HERE/(name+'.npz'))
            temporal=LatentTemporal(arrays)
            moments=dict(mean=torch.from_numpy(arrays['completion_mean']),covariance=torch.from_numpy(arrays['completion_covariance']))
            counts=dict(gates=0,adapter_gate_differences=0,adapter_accept_differences=0,
                missing_private_gate_differences=0,A_V_FP32_gate_differences=0,
                predicted_max_difference=0.)
            for start in range(0,len(data['valid']['y']),8):
                item=batch(data['valid'],torch.arange(start,min(start+8,len(data['valid']['y']))),'cpu',float(op['source_theta']))
                state=completion(model,item,constants,moments)
                gate,detail=temporal.forward_groups(state['y'],state['shared'],state['empty'],return_details=True)
                counts['gates']+=gate.numel()
                counts['adapter_gate_differences']+=int((gate!=state['gate']).sum())
                counts['adapter_accept_differences']+=int((detail['accepted']!=state['accept']).sum())
                counts['predicted_max_difference']=max(counts['predicted_max_difference'],float((detail['predicted']-state['predicted']).abs().max()))
                sparse_tail=state['z'][...,32:]*state['need_z'].repeat_interleave(2,-1)
                raw=state['shared']+sparse_tail@(model.v*model.connectivity)[32:]
                sparse_y=raw*constants['bn_scale']+constants['bn_bias']
                full=torch.einsum('ts,gsph->gtph',a,sparse_y)+b[None,:,None,None]-theta
                sparse_gate=torch.where(state['accept'],state['predicted'].ge(0),full.ge(0))
                counts['missing_private_gate_differences']+=int((sparse_gate!=state['gate']).sum())
                q=torch.einsum('ts,gspr->gtpr',a,state['z'])
                av=(q@(model.v*model.connectivity))*constants['bn_scale']
                av+=(a.sum(1)[:,None]*constants['bn_bias'][None,:]+b[:,None]-theta)[None,:,None,:]
                counts['A_V_FP32_gate_differences']+=int((av.ge(0)!=state['full'].ge(0)).sum())
            # A real-valued mini-image assembled only to test T/C/P4 layout;
            # it is not described as adjacent physical capture positions.
            item=batch(data['valid'],torch.arange(32),'cpu',float(op['source_theta']))
            state=completion(model,item,constants,moments)
            def image(values):
                return values.reshape(4,8,10,4,96).permute(2,4,0,1,3).reshape(10,1,96,4,32)
            pair=LatentPair(arrays,'cpu')
            pair.shared_raw=image(state['shared'])
            pair.empty=state['empty'].reshape(4,8,10,4).permute(2,0,1,3).reshape(10,4,32)
            actual=pair.neuron_forward(image(state['y']))
            expected=image(state['gate'].float()*theta)
            counts['network_layout_gate_differences']=int((actual!=expected).sum())
            output['axes'][name]=counts
            if any(counts[key] for key in ('adapter_gate_differences','adapter_accept_differences',
                'missing_private_gate_differences','network_layout_gate_differences')):
                raise AssertionError(name+': '+json.dumps(counts))
            print('CHECK',name,json.dumps(counts),flush=True)
    # Original unfactored complete-T source/W-zero baseline, with H8 vector
    # requests converted to bytes rather than compared to J2 as equal words.
    w=torch.from_numpy(op['weight'].reshape(96,864).T).ne(0).float()
    word=w.T.reshape(12,8,864).any(1)
    original=dict(Conv1_active_scalar_adds=0,W_H8_requests=0,PSN_data_terms=0,source_live_P4_T10_words=0)
    with torch.no_grad():
        for start in range(0,len(data['valid']['y']),8):
            item=batch(data['valid'],torch.arange(start,min(start+8,len(data['valid']['y']))),'cpu',float(op['source_theta']))
            src=item['source'].ne(0)
            original['Conv1_active_scalar_adds']+=int((src.float()@w).sum(dtype=torch.float64))
            live=src.any((1,2))
            original['W_H8_requests']+=int((live[:,None,:]*word[None]).sum())
            original['source_live_P4_T10_words']+=int(live.sum())
            original['PSN_data_terms']+=int(((~src.eq(0).all(-1))*a.ne(0).sum(0)[None,:,None]).sum())*96
    original['W_request_bytes']=original['W_H8_requests']*8*4
    original['scope']='same valid4 sampled source theta*g and original W; all T10 in one scan, source/W-zero skipped; fixed BN constants compile into thresholds; not an AEE-matched service ratio'
    output['unfactored_same_A']=original
    (HERE/'adapter_checks.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print('ORIGINAL',json.dumps(original),flush=True)


if __name__=='__main__':
    main()
