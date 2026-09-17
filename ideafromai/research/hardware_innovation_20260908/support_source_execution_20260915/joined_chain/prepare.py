"""Export only X and static parameters; C++ independently derives all gold."""
from pathlib import Path
import argparse,json,struct
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent
PREV=SOURCE.parent/'support_lut_execution_20260915'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--adapt',action='store_true');args=ap.parse_args()
    src=np.load(SOURCE/'source_cases.npz')
    back=np.load(PREV/'cases.npz')
    graph=dict(np.load(SOURCE/'prefix_tables_entropy.npz'))
    D=src['D'].astype(np.uint8)
    assert np.array_equal(D,back['D'])
    ids=[int(np.flatnonzero((back['hblock']==h)&back['is_real'])[0]) for h in range(4)]
    A0=src['A_q12'].astype(np.int16);tau0=src['threshold_q28'].astype(np.int64)
    A1=back['A'].astype(np.int16)
    W=np.concatenate([back['W'][i] for i in ids],axis=1).astype(np.int8)
    Wp=np.load(PREV/'response_class_W.npy').T.astype(np.int8)
    if args.adapt:
        adapted=np.load(SOURCE/'source_class_adapt/adapt_tables.npz')
        Wp=np.load(SOURCE/'source_class_adapt/adapt_W.npy').T.astype(np.int8)
        graph['response_canonical_code']=adapted['canonical']
        graph['class_nodes64']=adapted['class_nodes_entropy']
        graph['roots']=graph['roots'].copy();graph['roots'][1]=adapted['roots'][1]
    tau=np.concatenate([back['tau'][i] for i in ids],axis=1).astype(np.int64)
    pos=np.concatenate([back['positive_gain'][i] for i in ids]).astype(np.uint8)
    con=np.concatenate([back['constant_channels'][i] for i in ids]).astype(np.uint8)
    cg=np.concatenate([back['constant_gate'][i] for i in ids],axis=1).astype(np.uint8)
    assert W.shape==Wp.shape==(96,384) and tau.shape==(10,384)
    rank=np.full((6,16),15,dtype=np.uint8)
    for g in range(6):rank[g,graph[f'variables_g{g}']]=np.arange(len(graph[f'variables_g{g}']))
    bounds={}
    for name,w in [('original',W),('response_class',Wp)]:
        w=w.astype(np.int64);a=A1.astype(np.int64)
        lo=np.minimum(w,0).sum(0);hi=np.maximum(w,0).sum(0)
        ulo=np.minimum(a[:,:,None]*lo[None,None,:],a[:,:,None]*hi[None,None,:]).sum(1)
        uhi=np.maximum(a[:,:,None]*lo[None,None,:],a[:,:,None]*hi[None,None,:]).sum(1)
        L=np.einsum('gkc,gch->gkh',D.astype(np.int64),w.reshape(6,16,384))
        assert lo.min()>=-(1<<23) and hi.max()<(1<<23)
        assert ulo.min()>=-(1<<47) and uhi.max()<(1<<47)
        assert L.min()>=-512 and L.max()<=511
        if name=='response_class':
            for g in range(6):assert np.array_equal(L[g],L[g,graph['response_canonical_code'][g]])
        bounds[name]={'Y_any_spike_prefix':[int(lo.min()),int(hi.max())],
                      'U_any_spike_prefix':[int(ulo.min()),int(uhi.max())],
                      'L_INT10':[int(L.min()),int(L.max())]}
    src_bound=int(np.abs(A0.astype(np.int64)).sum(1).max())*(1<<23)
    assert src_bound<(1<<47)
    cases=[]
    for i in range(2):cases.append((f'train{i}',1,src['X_q16'][i].transpose(1,2,0).astype(np.int32)))
    cases.append(('diagnostic_zero',0,np.zeros((32,96,10),dtype=np.int32)))
    p,c,t=np.indices((32,96,10))
    cases.append(('diagnostic_signed_extreme',0,np.where((p+c+t)%2,(1<<23)-1,-(1<<23)).astype(np.int32)))
    stem='inputs_adapt' if args.adapt else 'inputs'
    with (HERE/(stem+'.bin')).open('wb') as f:
        f.write(b'JOIN0001')
        for a,dtype in [(D,'u1'),(A0,'<i2'),(tau0,'<i8'),(A1,'<i2'),(tau,'<i8'),(pos,'u1'),(con,'u1'),(cg,'u1'),(W,'i1'),(Wp,'i1')]:
            f.write(a.astype(dtype).tobytes())
        for key in ['code_nodes64','class_nodes64']:
            a=graph[key];f.write(struct.pack('<I',len(a)));f.write(a.astype('<u8').tobytes())
        f.write(graph['roots'].astype('<u2').tobytes());f.write(rank.tobytes())
        f.write(graph['response_canonical_code'].astype('u1').tobytes())
        f.write(struct.pack('<I',len(cases)))
        for name,real,x in cases:
            assert x.shape==(32,96,10) and x.min()>=-(1<<23) and x.max()<(1<<23)
            s=name.encode();f.write(struct.pack('<II',real,len(s)));f.write(s);f.write(x.astype('<i4').tobytes())
    stats={'function_variant':'adapt_response_class' if args.adapt else 'response_class',
           'source_npz':str(SOURCE/'source_cases.npz'),'source_real_frames':src['frame_file'].tolist(),
           'source_split':'first two training frames, same 32 sampled positions; not held out',
           'backend_fixed_integer_source':str(PREV/'cases.npz'),'backend_records':[str(back['case_name'][i]) for i in ids],
           'backend_threshold_contract':'existing fixed A14/tau/sign/constant integer function; no new joined-chain AEE established',
           'D_shape':list(D.shape),'W_shape':list(W.shape),'cases':[n for n,_,_ in cases],
           'D_read_via_RTL':True,'DUT_dynamic_input':'X_q16 only; code/projected-g/Y/U/gold are never supplied',
           'source_abs_accumulator_bound':src_bound,'source_threshold_range':[int(tau0.min()),int(tau0.max())],
           'backend_threshold_range':[int(tau.min()),int(tau.max())],'bounds':bounds,
           'graph_nodes':{m:len(graph[f'{m}_nodes64']) for m in ['code','class']},
           'physical_bank_pool_bytes':262144,'bridge_gate_bytes':3840,'bridge_dictionary_bytes':192,
           'mac_units':{'source':10,'backend':96,'total_distinct_units':106,'concurrent_execution':False}}
    (HERE/(stem+'.json')).write_text(json.dumps(stats,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps(stats,ensure_ascii=False))

if __name__=='__main__':main()
