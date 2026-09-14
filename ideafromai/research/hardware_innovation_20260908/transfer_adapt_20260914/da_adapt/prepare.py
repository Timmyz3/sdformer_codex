"""Prepare exclusively from native source and static Q1/Q2; no latent inputs."""
from pathlib import Path
import json,shutil
import numpy as np
H=Path(__file__).resolve().parent
OLD=H.parents[1]/'fusion_ten_trials_20260914/decompositions/q2_da'
RNG=np.random.default_rng(914241)
def readhex(p):
    return np.array([int(s,16) for s in p.read_text().split()],np.uint32).view(np.int32).astype(np.int64)
def writehex(p,a):
    p.write_text(''.join(f'{int(v)&0xffffffff:08x}\n' for v in np.asarray(a).flat))
def native_gold(words,q1,q2,origin):
    # Independent direct convolution indexing, unlike old im2col factor preparation.
    z=np.zeros((4,10,8),np.int64)
    for p in range(4):
        for ky in range(3):
            for kx in range(3):
                y,x=p//2+ky,p%2+kx
                if 0<=int(origin[0])+y<240 and 0<=int(origin[1])+x<320:
                    for c in range(96):
                        for t in range(10):
                            if (int(words[c,y,x])>>t)&1:
                                z[p,t]+=q1[:,c*9+ky*3+kx]
    out=np.empty((480,8),np.int64)
    for og in range(12):
        for p in range(4):
            for t in range(10):
                # Python integer dot avoids copying a fixed-width accumulator overflow.
                out[og*40+p*10+t]=[sum(int(q2[og*8+l,r])*int(z[p,t,r]) for r in range(8)) for l in range(8)]
    assert np.max(z)<=4095 and np.min(z)>=-4096
    assert np.max(out)<2**31 and np.min(out)>=-2**31
    return z,out
cases=[]
def emit(name,words,q1,q2,origin,kind):
    d=H/'fixtures'/name;d.mkdir(parents=True,exist_ok=True)
    z,gold=native_gold(words,q1,q2,origin)
    for file,a in [('source',words),('q1',q1.T),('q2',q2.reshape(12,8,8).transpose(0,2,1)),('origin',origin),('k_live',np.any(q1,axis=0)),('gold',gold)]:writehex(d/(file+'.hex'),a)
    cases.append(dict(name=name,kind=kind,origin=list(map(int,origin)),z_min=int(z.min()),z_max=int(z.max()),nonzero_z=int(np.count_nonzero(z)),raw_min=int(gold.min()),raw_max=int(gold.max())))
    return gold
def main():
    oldmeta=json.loads((OLD/'definition.json').read_text())
    for c in oldmeta['fixtures']:
        d=OLD/'fixtures'/c['name']
        w=readhex(d/'source.hex').reshape(96,4,4)
        q1=readhex(d/'q1.hex').reshape(864,8).T
        q2=readhex(d/'q2.hex').reshape(12,8,8).transpose(0,2,1).reshape(96,8)
        gold=emit(c['name'],w,q1,q2,readhex(d/'origin.hex'),'original')
        assert np.array_equal(gold.ravel(),readhex(d/'gold.hex'))
    full=np.full((96,4,4),1023,np.int64)
    qreal=readhex(OLD/'fixtures/real_0/q1.hex').reshape(864,8).T
    vreal=readhex(OLD/'fixtures/real_0/q2.hex').reshape(12,8,8).transpose(0,2,1).reshape(96,8)
    qsmall=np.zeros((8,864),np.int64);qsmall[:,0]=1
    vext=RNG.choice([-32768,-32767,-1,1,32767],size=(96,8))
    emit('q2_all_zero',full,qreal,np.zeros_like(vreal),[80,120],'zero_Q2')
    vblocks=vreal.copy();vblocks[8:24]=0
    emit('q2_empty_blocks',full,qsmall,vblocks,[80,120],'partial_zero_Q2')
    emit('small_dense_positive',full,qsmall,vreal,[80,120],'synthetic_small_z')
    emit('small_dense_negative',full,-qsmall,vreal,[80,120],'synthetic_small_z')
    qm=qsmall.copy();qm[::2]*=-1
    emit('small_dense_mixed_extreme_q2',full,qm,vext,[80,120],'synthetic_small_z')
    for nr in [1,2,3]:
        q=qsmall.copy();q[nr:]=0
        emit(f'rank{nr}_small',full,q,vreal,[80,120],'selector_corner')
    vcancel=np.zeros_like(vreal);vcancel[:,::2]=-32768;vcancel[:,1::2]=32767
    # Exact cancelling full-vector LUT entries, while individual rank weights stay live.
    vcancel[:,::2]=-32767
    emit('small_q2_cancellation',full,qsmall,vcancel,[80,120],'selector_corner')
    emit('signed3_min',full,np.full_like(qreal,-4),np.full_like(vreal,-32768),[80,120],'signed_extreme')
    wr=RNG.integers(0,1024,size=full.shape,dtype=np.int64)
    qr=RNG.integers(-4,4,size=qreal.shape,dtype=np.int64)
    vr=vext.copy();vr[8:16]=0;vr[:,6:]=0
    emit('padding_poison',wr,qr,vr,[237,317],'padding_and_sign')
    meta=dict(oldmeta,fixtures=cases,modes={'11':'paid hybrid, lazy build, remaining-work DA<rank selector','14':'original MAC','15':'original fixed DA','13':'paid hybrid, nonempty eager build','12':'paid hybrid, lazy first evaluation build'})
    (H/'definition.json').write_text(json.dumps(meta,indent=2)+'\n')
    (H/'gold_audit.json').write_text(json.dumps(dict(fixtures=len(cases),values=len(cases)*3840,original_gold_values_verified=14*3840,source='raw source/Q1/Q2, independent convolution index and Python-integer output dot',latent_input=False),indent=2)+'\n')
    print(json.dumps({'fixtures':len(cases),'gold_values':len(cases)*3840}))

if __name__=="__main__": main()
