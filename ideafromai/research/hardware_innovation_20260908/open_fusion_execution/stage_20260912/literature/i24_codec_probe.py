"""Exact signed24/f14 byte opportunity on captured continuous PED.

Raw baseline is packed 24, never NumPy int32 storage. All codec modes pay a
mode/width byte, block padding, and bases/defaults. No cycle/PPA model.
"""
import json
import struct
import time
from pathlib import Path
import numpy as np
from exact_codec_probe import pack, unpack, HERE, CAP, B


def signed_width(a):
    # Signed range [-2**(w-1),2**(w-1)-1].
    m=np.maximum(a.max(axis=1),-a.min(axis=1)-1)
    return (np.floor(np.log2(np.maximum(m,1)))+2).astype(np.int64)-(m==0)


def enc(a,mode,default):
    if mode==0:return bytes([0])+pack(a&0xffffff,24)
    if mode==1:
        width=int(signed_width(a[None,:])[0]);v=a&((1<<width)-1)
        return bytes([1,width])+pack(v,width)
    if mode in [2,3]:
        v=a[1:]-a[0] if mode==2 else (a[1:]&0xffffff)^(int(a[0])&0xffffff)
        width=int(signed_width(v[None,:])[0]) if mode==2 else int(v.max(initial=0)).bit_length()
        return bytes([mode,width])+pack(a[:1]&0xffffff,24)+pack(v&((1<<width)-1),width)
    neq=a!=default
    return bytes([4])+pack(neq,1)+pack(a[neq]&0xffffff,24)


def sign(v,width):
    if width==0:return v.astype(np.int64)
    return ((v^(1<<(width-1))).astype(np.int64)-(1<<(width-1)))


def dec(data,default):
    mode=data[0]
    if mode==0:return sign(unpack(data[1:],24,B),24)
    if mode==1:return sign(unpack(data[2:],data[1],B),data[1])
    if mode in [2,3]:
        width=data[1];first=sign(unpack(data[2:5],24,1),24)[0]
        v=unpack(data[5:],width,B-1)
        tail=sign(v,width)+first if mode==2 else sign(v^(int(first)&0xffffff),24)
        return np.concatenate([[first],tail])
    neq=unpack(data[1:9],1,B).astype(bool);out=np.full(B,default,dtype=np.int64)
    out[neq]=sign(unpack(data[9:],24,int(neq.sum())),24)
    return out


def probe(q,label,layout):
    assert q.min()>=-2**23 and q.max()<2**23
    # Existing producer spill is spatial,T,C; alternative native capture layout
    # is reported separately and must pay a conversion if selected in deployment.
    v=(q.transpose(2,3,0,1) if layout=='spatial_T_C' else q).copy().reshape(-1).astype(np.int64)
    pad=(-v.size)%B
    a=np.pad(v,(0,pad)).reshape(-1,B)
    # 0 is a fixed dictionary value, not a same-frame learnt default.
    default=0
    width=signed_width(a)
    delta=a[:,1:]-a[:,:1]
    dw=signed_width(delta)
    xo=(a[:,1:]&0xffffff)^(a[:,:1]&0xffffff)
    xm=xo.max(axis=1)
    xw=np.where(xm>0,np.floor(np.log2(np.maximum(xm,1)))+1,0).astype(np.int64)
    raw=np.full(len(a),1+B*3,dtype=np.int64)
    sw=2+(B*width+7)//8
    ds=2+3+((B-1)*dw+7)//8
    xs=2+3+((B-1)*xw+7)//8
    zf=1+8+(a!=0).sum(axis=1)*3
    costs=np.stack([raw,sw,ds,xs,zf])
    modes=costs.argmin(axis=0)
    checks=0;mismatch=0;emitted=0
    for bi in np.unique(np.linspace(0,len(a)-1,min(512,len(a)),dtype=np.int64)):
        for mode in range(5):
            blob=enc(a[bi],mode,default)
            assert len(blob)==costs[mode,bi]
            rebuilt=dec(blob,default)
            mismatch+=int(np.count_nonzero(rebuilt!=a[bi]));checks+=1;emitted+=len(blob)
    nums={'packed24_framed':int(raw.sum()+16),'block_signed_width':int(np.minimum(raw,sw).sum()+16),'base_signed_delta':int(np.minimum(raw,ds).sum()+16),'base_bitwise_xor':int(np.minimum(raw,xs).sum()+16),'zero_mask':int(np.minimum(raw,zf).sum()+16),'ordinary_all_modes':int(costs.min(axis=0).sum()+16)}
    return {'label':label,'layout':layout,'shape':list(q.shape),'elements':int(q.size),'raw_packed24_bytes':int(q.size*3),'range':[int(q.min()),int(q.max())],'codec_bytes':nums,'ratio_over_packed24':{k:n/(q.size*3) for k,n in nums.items()},'selected_blocks':dict(zip(['raw','signed_width','signed_delta','xor','zero'],map(int,np.bincount(modes,minlength=5)))),'actual_serialized_blocks_tested':checks,'bit_mismatches':mismatch,'sample_bytes_emitted':emitted,'padded_values':pad}


def main():
    start=time.time();rows=[]
    for arm in ['ordinary','lifting_raw']:
        p=CAP.parent/'capture_full_producers'/arm/'000_zurich_city_09_a_0001.npz'
        with np.load(p,allow_pickle=False) as f:
            q=f['full_continuous_q24']
            for layout in ['spatial_T_C','native_T_C_spatial']:
                r=probe(q,arm+'/full_continuous_q24',layout);r['source_path']=str(p);rows.append(r)
                print(r['label'],layout,r['ratio_over_packed24'],flush=True)
            for key in ['corner_I24','interior_I24']:
                r=probe(f[key],arm+'/'+key,'native_T_C_spatial');r['source_path']=str(p);rows.append(r)
    (HERE/'i24_codec_results.json').write_text(json.dumps({'scope':'Actual captured signed24/f14 continuous PED full192000x96, plus corner/interior source windows; no new quantization.','format':'signed24/f14; packed24 baseline; producer layout spatial,T,C','limitations':['Reference simple codecs, not full EBPC/BDI implementation.','Per-block selection/histograms/codec cycles and buffers not modeled.','Native layout conversion costs not modeled; compare within layout only.','No accuracy rerun: no arithmetic changed, samples round-trip exactly.'],'elapsed_wall_s':time.time()-start,'rows':rows},indent=2)+'\n')


if __name__=='__main__':main()
