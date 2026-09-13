"""Exact integer linear leaf, computed through exact-range FP64 convolution."""
import json,time
import numpy as np
import torch
import torch.nn.functional as F
from model_access import HERE,BASE
ARMS=('dense_q16','block_magnitude25','cin_fullcost25')

def checked_int(x):
    assert torch.equal(x,x.round()),'Unexpected noninteger FP64 sum'
    assert x.abs().max()<2**31
    return x.cpu().numpy().astype('<i4')

def main():
    bits=np.load(HERE/'source_bits.npy',mmap_mode='r')
    padded=np.pad(bits,((0,0),(0,0),(1,1),(1,1)))
    ids=np.arange(128,192)
    tiles=np.stack([padded[:,:,2*(i//160):2*(i//160)+4,2*(i%160):2*(i%160)+4] for i in ids])
    x=torch.as_tensor(tiles.reshape(-1,96,4,4),device='cuda',dtype=torch.float64)
    report=dict(complete=False,arithmetic='FP64 convolution of exact binary and int16 values, all intermediate absolute sums <=864*32768<2^53; integer exact then checked before int32 cast',first64={})
    for arm in ARMS:
        w=np.load(HERE/(arm+'_weights.npz'))['weight_q16']
        wt=torch.as_tensor(w,device='cuda',dtype=torch.float64)
        g=checked_int(F.conv2d(x,wt)).reshape(64,10,96,2,2)
        np.save(HERE/('gold64_'+arm+'.npy'),g)
        report['first64'][arm]='gold64_'+arm+'.npy'
    (HERE/'gold_progress.json').write_text(json.dumps(report,indent=2)+'\n')
    print('GOLD64_READY',flush=True)
    del x
    started=time.monotonic()
    x=torch.as_tensor(np.array(bits),device='cuda',dtype=torch.float64)
    for arm in ARMS:
        arm_start=time.monotonic()
        wt=torch.as_tensor(np.load(HERE/(arm+'_weights.npz'))['weight_q16'],device='cuda',dtype=torch.float64)
        # T slices bound workspace. No partial K/tap/channel truncation.
        gold=np.lib.format.open_memmap(HERE/('gold_'+arm+'.npy'),mode='w+',dtype='<i4',shape=(19200,10,96,2,2))
        for t in range(10):
            y=checked_int(F.conv2d(x[t:t+1],wt,padding=1))[0]
            gold[:,t]=y.reshape(96,120,2,160,2).transpose(1,3,0,2,4).reshape(19200,96,2,2)
        gold.flush()
        assert np.array_equal(gold[ids],np.load(HERE/('gold64_'+arm+'.npy')))
        old=np.load(BASE/'r0_source_retirement_20260913/data'/(arm+'.npz'))
        old_ids=old['output_origin_yx'][:,0]//2*160+old['output_origin_yx'][:,1]//2
        assert np.array_equal(gold[old_ids],old['golden_accum'])
        report[arm]=dict(path='gold_'+arm+'.npy',shape=list(gold.shape),old8_integer_gold_equal=True,first64_equal=True,wall_seconds=time.monotonic()-arm_start,min=int(gold.min()),max=int(gold.max()))
        (HERE/'gold_progress.json').write_text(json.dumps(report,indent=2)+'\n')
        print('FULL_GOLD_READY',arm,report[arm]['wall_seconds'],flush=True)
    report.update(complete=True,total_wall_seconds=time.monotonic()-started)
    (HERE/'gold_progress.json').write_text(json.dumps(report,indent=2)+'\n')
    manifest=json.loads((HERE/'manifest.json').read_text());manifest['gold_complete']=True
    manifest['gold_report']='gold_progress.json'
    for arm in ARMS:manifest['arms'][arm]['gold']=report[arm]['path']
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
