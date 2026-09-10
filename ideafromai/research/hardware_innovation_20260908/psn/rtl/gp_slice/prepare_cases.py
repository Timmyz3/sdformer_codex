"""Full-C384 source/W memories and independent NumPy int64 slice references."""
from pathlib import Path
import json
import struct
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'psn'))
from gustavsnn_reference import read_torch,identities,specialize_routes


def digits(v):
    n=abs(int(v)); sign=1 if v>=0 else -1; shift=0; out=[]
    while n:
        if n&1:
            d=2-(n&3);out.append((shift,sign*d));n-=d
        n>>=1;shift+=1
    return out


def make_program(coeff):
    program=[]
    for t,row in enumerate(coeff):
        terms=[(r,s,d) for r,c in enumerate(row) for s,d in digits(c)]
        program.append((1<<15)|(1<<9)|(t<<11)|((not terms)<<10))
        for i,(r,s,d) in enumerate(terms):
            assert s<=8
            program.append(r|(s<<3)|((d<0)<<8)|(t<<11)|((i==len(terms)-1)<<10))
    assert len(program)<=256
    return np.asarray(program,dtype='<u2')


def make(name,codes,W,decode,coeff,tau,theta,meta):
    codes=np.asarray(codes,dtype=np.uint8).reshape(4,4,384)
    W=np.asarray(W,dtype=np.int8).reshape(2,384)
    decode=np.asarray(decode,dtype=np.int64);coeff=np.asarray(coeff,dtype=np.int64)
    tau=np.asarray(tau,dtype=np.int64).reshape(10,2)
    R=len(decode)
    assert 1<=R<=7 and not np.any((decode!=0)&(decode!=1)) and not decode[:,0].any()
    assert coeff.shape==(10,R) and int(codes.max())<8
    wp=np.maximum(W.astype(np.int64),0).sum(1);wn=np.minimum(W.astype(np.int64),0).sum(1)
    assert wn.min()>=-16384 and wp.max()<=16383
    expanded=decode[:,codes] # R,K,P,C: independent dense support expansion.
    S=np.einsum('rkpc,mc->mkpr',expanded,W.astype(np.int64),optimize=True)
    touched=np.einsum('rkpc,mc->mkpr',expanded,(W!=0).astype(np.int64),optimize=True)!=0
    U=np.einsum('tr,mkpr->mkpt',coeff,S,optimize=True)
    golden=U>=tau.T[:,None,None,:]
    # All-code static bound at EACH CSD substep, not just final measured sums.
    lo=0;hi=0;operand_bound=0
    for t,row in enumerate(coeff):
        effective=np.zeros(8,dtype=np.int64)
        lo=min(lo,int((-tau[t]).min()));hi=max(hi,int((-tau[t]).max()))
        for r,c in enumerate(row):
            for shift,sign in digits(c):
                operand_bound=max(operand_bound,int(max(wp.max(),-wn.min()))*(1<<shift))
                effective+=sign*(decode[r]<<shift)
                lower=wp*effective.min()+wn*effective.max()-tau[t]
                upper=wp*effective.max()+wn*effective.min()-tau[t]
                lo=min(lo,int(lower.min()));hi=max(hi,int(upper.max()))
    assert -(1<<23)<=lo<=hi<(1<<23) and operand_bound<(1<<23)
    padded=np.zeros((2,4,4,7),dtype=np.int32);padded[...,:R]=S
    present=np.zeros_like(padded,dtype=np.uint8);present[...,:R]=touched
    coeff_pad=np.zeros((10,7),dtype=np.int32);coeff_pad[:,:R]=coeff
    gate_words=np.sum(golden.astype(np.uint16)<<np.arange(10,dtype=np.uint16),axis=-1,dtype=np.uint16)
    return dict(name=name,codes=codes,W=W,decode=decode,coeff=coeff_pad,tau=tau.T,
                theta=np.asarray(theta,dtype='<f4').view('<u4'),program=make_program(coeff),
                S=padded,present=present,gate_words=gate_words,
                metadata=dict(meta,rank=R,source_shape=[4,4,384],output_rows=2,
                              static_S15_range=[int(wn.min()),int(wp.max())],
                              static_Acc24_all_CSD_prefix_range=[lo,hi],
                              shift_operand_abs_bound=operand_bound,gate_ones=int(golden.sum()),
                              theta_output=list(map(float,theta))))


def main():
    params=read_torch(ROOT/'algorithm/stage2_temporal_codes/integer_parameters.pt')
    books=np.load(ROOT/'algorithm/stage2_temporal_codes/codebooks.npz')
    basis=np.load(ROOT/'algorithm/stage2_temporal_codes/signed_basis.npz')
    consumers=json.loads((ROOT/'algorithm/stage2_class_shift/consumers.json').read_text())
    producers=read_torch(ROOT/'algorithm/direct_code_stage2/producers.pt')
    paths=sorted(p for p in (ROOT/'algorithm/direct_code_integer/deployment/capture10').glob('*.npz')
                 if p.name.startswith(('v000_','v006_')))
    assert len(paths)==12
    cases=[]
    for i,path in enumerate(paths):
        b=int(path.stem[-1]);W,routes,identity=identities(b,params,books,basis,consumers)
        routes,identity,costs,tau=specialize_routes('integer',b,W,routes,identity,producers,consumers)
        q=params[f'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.{b}.mlp.']
        assert np.all(q['positive_gain']) and not np.any(q['constant_channels'])
        p=((137*b+53*(i//6))%1184)//4*4
        hs=[(211*b+97*(i//6))%1536,(211*b+97*(i//6)+733)%1536]
        with np.load(path) as z:codes=z['codes'][p:p+16]
        same=[]
        for route in ('packed_class','packed_time'):
            decode,coeff=routes[route]
            c=make(f'{path.stem}_p{p}_h{hs[0]}_{hs[1]}_{route}',codes,W[hs],decode,coeff,tau[:,hs],
                   [float(q['theta_output'])]*2,
                   dict(real=True,capture=path.name,block=b,route=route,positions=list(range(p,p+16)),
                        hidden_rows=hs,theta_source=float(q['theta_source']),
                        identity='Actual integer direct-code student; same trained B/tau, all T10; not ep34 FP32 equivalence'))
            cases.append(c);same.append(c['gate_words'])
        assert np.array_equal(*same)
    D=np.asarray([[(c>>r)&1 for c in range(8)] for r in range(3)],dtype=np.int64)
    A=np.asarray([[0,0,0],[1,-2,3],[-4,1,0],[2,3,-1],[0,0,0],
                  [-1,1,1],[4,-4,2],[1,0,0],[0,-1,0],[1,2,-3]],dtype=np.int64)
    ordinary=np.resize(np.array([-7,0,3,-2,1,5],dtype=np.int8),(2,384))
    dense=(np.arange(16*384).reshape(16,384)*3+np.arange(16)[:,None])%8
    directed=[]
    directed.append(('empty_source',np.zeros((16,384),dtype=np.uint8),ordinary))
    directed.append(('zero_weights',dense,np.zeros((2,384),dtype=np.int8)))
    directed.append(('full_nrv_backpressure',np.full((16,384),7,dtype=np.uint8),
                     np.resize(np.array([-7,3,5,-2],dtype=np.int8),(2,384))))
    asymmetric=np.zeros((2,384),dtype=np.int8)
    asymmetric[0,[0,5,91,177,383]]=[127,-127,-128,127,1]
    asymmetric[1,[0,3,6,55,129,251,383]]=[-128,127,1,31,-31,-7,7]
    directed.append(('asymmetric_tail_cancel',np.ones((16,384),dtype=np.uint8),asymmetric))
    limits=np.zeros((2,384),dtype=np.int8);limits[0,:129]=127;limits[1,:128]=-128
    directed.append(('S15_limits',np.ones((16,384),dtype=np.uint8),limits))
    # Exact rational theta-source compilation, scale=1/2. No invented checkpoint.
    base=np.resize(np.array([-7,-3,0,1,2,5,7],dtype=np.int64),(2,384))
    theta_twice=np.resize(np.array([1,3,4],dtype=np.int64),384)
    folded=(base*theta_twice).astype(np.int8)
    directed.append(('nonunit_theta_rational',dense,folded))
    for name,codes,weight in directed:
        S=np.einsum('rpc,mc->mpr',D[:,codes],weight.astype(np.int64))
        U=np.einsum('tr,mpr->mpt',A,S)
        tau=U[:,0,:].T+np.resize(np.array([-1,0,1],dtype=np.int64),(10,2))
        if name=='S15_limits':tau=np.resize(np.array([0,16383,-16384],dtype=np.int64),(10,2))
        if name=='nonunit_theta_rational':
            # Twice the physical real-valued dot: theta_c * baseW, with code D.
            physical_twice=np.einsum('rpc,mc,c->mpr',D[:,codes],base,theta_twice)
            assert np.array_equal(physical_twice,S)
        same=[]
        for route,dec,co in [('packed_class',np.eye(8,dtype=np.int64)[1:],(A@D)[:,1:]),
                             ('packed_time',D,A)]:
            meta=dict(real=False,route=route,directed=name,
                      identity='Legal finite integer/rational interface diagnostic, not checkpoint evidence')
            if name=='nonunit_theta_rational':meta.update(theta_source_values=[.5,1.5,2.0],
                weight_quantum=.5,tau_physical_quantum=.5,folded_weight='Wq=2*theta_c*baseW; theta_output independent')
            c=make(name+'_'+route,codes,weight,dec,co,tau,[1.25,.75],meta)
            cases.append(c);same.append(c['gate_words'])
        assert np.array_equal(*same)
    with (HERE/'cases.bin').open('wb') as f:
        f.write(b'GPS1');f.write(struct.pack('<I',len(cases)))
        for c in cases:
            n=c['name'].encode();f.write(struct.pack('<H',len(n)));f.write(n)
            f.write(bytes([len(c['decode']),c['metadata']['real'],c['metadata']['route']=='packed_time']))
            f.write(c['theta'].tobytes())
            table=np.sum(c['decode'].T.astype(np.uint64)<<np.arange(len(c['decode']),dtype=np.uint64),axis=1).astype(np.uint8)
            f.write(table.tobytes());f.write(struct.pack('<H',len(c['program'])));f.write(c['program'].tobytes())
            for key,dtype in [('coeff','<i4'),('tau','<i4'),('W','i1'),('codes','u1'),
                              ('S','<i4'),('present','u1'),('gate_words','<u2')]:
                f.write(np.asarray(c[key],dtype=dtype).tobytes())
    summary=dict(real_captures=len(paths),total_cases=len(cases),real_cases=sum(c['metadata']['real'] for c in cases),
                 reference='NumPy int64 dense C384 dot -> exact coefficient matrix -> actual tau, independent of NR4/ports; C++ repeats dense dot independently',
                 complete_extent='16 positions x 2 hidden rows x C384 x T10 per case',
                 cases=[dict(name=c['name'],program_length=len(c['program']),**c['metadata']) for c in cases])
    (HERE/'cases.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'{len(cases)} cases, {len(paths)} actual captures, same-student class/time equality verified')


if __name__=='__main__':main()
