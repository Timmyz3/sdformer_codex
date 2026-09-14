"""One fixed frame prefix only, requested for real IEEE FP32 identity RTL ingress."""
import json
import numpy as np
from model_access import HERE,BASE,REPO,load_parent
import sys
sys.path.insert(0,str(BASE/'algorithm'))
import torch
from run_bn_probe import input_frame
from spikingjelly.activation_based import functional

class Captured(Exception):pass

def main():
    args,net=load_parent(HERE/'identity_capture')
    r0=net.modules[net.BLOCK.rsplit('.',1)[0]+'.0']
    ref=np.load(HERE/'consumer_first8.npz')
    oldJ=np.load(HERE/'identity_q20_full.npy',mmap_mode='r')
    def pre(m,inputs):
        x=inputs[0];x=x[:,0] if x.ndim==5 else x
        identity=x.detach().cpu().numpy()
        tiles=identity.reshape(10,96,120,2,160,2).transpose(2,4,0,1,3,5).reshape(19200,10,96,2,2)
        assert tiles.dtype==np.float32
        J=np.clip(np.rint(tiles.astype(np.float64)*2**20),-2**31,2**31-1).astype(np.int32)
        assert np.array_equal(J,oldJ)
        for i,(oy,ox) in enumerate(ref['output_origin_yx']):
            tid=int(oy//2*160+ox//2);assert np.array_equal(tiles[tid],ref['identity_fp32'][i])
        np.save(HERE/'identity_fp32_full.npy',tiles)
        np.save(HERE/'identity_fp32_first64.npy',tiles[128:192])
        report=dict(complete=True,frame='zurich_city_09_a_0001.npy',shape=list(tiles.shape),dtype=str(tiles.dtype),
            capture='single frame model prefix stops at actual r0 pre-hook; no r0/new flow quality computation',
            new_model_or_format=False,Q20_reference_all_equal=True,Q20_values_checked=J.size,small8_FP32_all_equal=True,
            identity_range=[float(tiles.min()),float(tiles.max())])
        (HERE/'identity_fp32_contract.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
        raise Captured()
    h=r0.register_forward_pre_hook(pre)
    try:
        functional.reset_net(net.model)
        with torch.no_grad():
            x,_,_=input_frame(args.data,'zurich_city_09_a_0001.npy',targets=False)
            try:net.model(x)
            except Captured:pass
            else:raise AssertionError('required hook not called')
    finally:h.remove();net.close()
if __name__=='__main__':main()
