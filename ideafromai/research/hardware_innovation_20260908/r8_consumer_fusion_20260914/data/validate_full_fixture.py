"""Independent bounded int64 consumer oracle, without model or RTL calls."""
from pathlib import Path
import json
import numpy as np
HERE=Path(__file__).resolve().parent

def main():
    coeff=np.load(HERE/'consumer_coefficients.npz')
    a=coeff['a_q40'].astype(np.int64)[None,None,:,None,None]
    b=coeff['b_q20'].astype(np.int64)[None,None,:,None,None]
    p=np.load(HERE/'raw_p_full.npy',mmap_mode='r')
    j=np.load(HERE/'identity_q20_full.npy',mmap_mode='r')
    gold=np.load(HERE/'i24_new_full.npy',mmap_mode='r')
    assert all(x.shape==(19200,10,96,2,2) and x.dtype==np.int32 for x in (p,j,gold))
    assert int(np.max(np.abs(a)))*679477248+(2**31+int(np.max(np.abs(b))))*2**20<2**63
    for x,name in ((p,'raw_p'),(j,'identity_q20'),(gold,'i24_new')):
        assert np.array_equal(x[128:192],np.load(HERE/(name+'_first64.npy')))
    lo=2**63-1;hi=-2**63;count=0
    for start in range(0,19200,64):
        wide=p[start:start+64].astype(np.int64)*a+((j[start:start+64].astype(np.int64)+b)<<20)
        lo=min(lo,int(wide.min()));hi=max(hi,int(wide.max()))
        q=wide//(2**26);rem=wide-q*2**26
        q+=((rem>2**25)|((rem==2**25)&((q&1)!=0)))
        q=np.clip(q,-2**23,2**23-1).astype(np.int32)
        assert np.array_equal(q,gold[start:start+64]);count+=q.size
    src=np.load(HERE/'first_source_words.npy',mmap_mode='r')
    small=np.load(HERE/'consumer_integer_first8.npz')
    assert src.shape==(96,240,320) and src.dtype==np.uint16 and int(src.max())<=1023
    for i,(oy,ox) in enumerate(small['output_origin_yx']):
        tile=int(oy//2*160+ox//2)
        for x,k in ((p,'p_int'),(j,'identity_q20'),(gold,'i24_new')):assert np.array_equal(x[tile],small[k][i])
        expected=np.zeros((10,96,4,4),np.uint8)
        for yy in range(4):
            for xx in range(4):
                y=int(oy)-1+yy;x=int(ox)-1+xx
                if 0<=y<240 and 0<=x<320:expected[:,:,yy,xx]=(src[:,y,x][None]>>np.arange(10)[:,None])&1
        assert np.array_equal(expected,small['source_bits'][i])
    result=dict(complete=True,GPU_or_RTL_rerun=False,full_consumer_values=count,all_integer_gold_equal=True,
        wide_range=[lo,hi],small8_and_first64_equal_full=True,tile_order='row-major output2x2: tile_id=(output_y//2)*160+output_x//2',
        array_shape=[19200,10,96,2,2],array_dtype='int32',source_shape=[96,240,320],source_dtype='uint16 low10bits',
        frame='zurich_city_09_a_0001.npy',capture_device='same A800/env312 as quality',
        source_file='first_source_words.npy',p_file='raw_p_full.npy',J_file='identity_q20_full.npy',I24_file='i24_new_full.npy')
    (HERE/'full_fixture_contract.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
