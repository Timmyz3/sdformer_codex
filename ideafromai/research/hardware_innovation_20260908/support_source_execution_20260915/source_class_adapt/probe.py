"""One pair/group: price response equivalence by actual T10 producer jobs.

Python3.12, no training. Frame0 chooses; frame1 is an unselected training probe.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[k]='1'
import sys,io,struct,json,csv
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
P=HERE.parent
B=P.parent
sys.path[:0]=[str(P),str(B/'support_lut_execution_20260915')]
from build_prefix_tables import Diagram,eval_packet_dag,eval_dag_all
from probe_response_projection import read_torch,bounded_zero_sum


def encode(nodes):
    x=np.asarray(nodes,dtype=np.uint64)
    ids=np.arange(len(x),dtype=np.uint64)
    term=ids<16
    return (x[:,1]|(x[:,2]<<16)|((x[:,0]&15)<<32)|(term.astype(np.uint64)<<36)
            |(np.where(term,ids,0)<<37))


def canonical(d,w):
    l=np.einsum('gkc,hgc->gkh',d,w.reshape(384,6,16))
    q=np.zeros((6,16),dtype=np.uint8)
    for g in range(6):
        for k in range(16):q[g,k]=np.flatnonzero(np.all(l[g]==l[g,k],axis=1))[0]
    return q,l


def main():
    src=np.load(P/'source_cases.npz');d=src['D'].astype(np.int64)
    w=read_torch(B/'algorithm/support_training/forced_code_weight_int8.pt').astype(np.int64)
    old=np.load(B/'support_lut_execution_20260915/response_class_W.npy').astype(np.int64)
    raw=src['raw_g_int'].transpose(0,2,1,3).reshape(64,10,6,16)
    words=(raw*(1<<np.arange(16))).sum(-1).astype(np.uint16)
    raw16=np.arange(65536,dtype=np.uint32)
    dbits=((raw16[:,None]>>np.arange(16))&1).astype(np.int16)
    profiles=[np.load(P/'prefix_tables.npz'),np.load(P/'prefix_tables_entropy.npz')]
    labels=[];expanded=[];variables=[];base=[];allrows=[];chosen=[];adapt=w.copy();integer_nearest=w.copy();integer_pairs=[]
    for g in range(6):
        lab=(dbits.sum(1)[:,None]+d[g].sum(1)[None,:]-2*(dbits@d[g].T)).argmin(1).astype(np.uint8)
        var=profiles[1][f'variables_g{g}']
        exp=(((np.arange(1<<len(var))[:,None]>>np.arange(len(var)))&1)*(1<<var.astype(np.uint32))).sum(1)
        dg=Diagram();root=dg.build(lab[exp],var);nodes=np.asarray(dg.nodes,dtype=np.uint32)
        _,bc=eval_packet_dag(nodes,root,words[:,:,g],var)
        labels.append(lab);expanded.append(exp);variables.append(var);base.append(bc)
        options=[]
        for a in range(1,16):
            for b in range(a+1,16):
                mapping=np.arange(16,dtype=np.uint8);mapping[b]=a
                gr=Diagram();rt=gr.build(mapping[lab[exp]],var)
                nd=np.asarray(gr.nodes,dtype=np.uint32)
                _,cost=eval_packet_dag(nd,rt,words[:,:,g],var)
                save=int((bc[:32].astype(int)-cost[:32]).sum())
                delta=d[g,a]-d[g,b];cols=np.flatnonzero(delta);sign=delta[cols]
                gw=w[:,g*16:(g+1)*16].copy()
                for h in range(384):gw[h,cols]=bounded_zero_sum(gw[h,cols]*sign)[1]*sign
                assert np.all(gw@delta==0) and gw.min()>=-128 and gw.max()<=127
                sq=int(np.square(gw-w[:,g*16:(g+1)*16]).sum())
                row=dict(g=g,a=a,b=b,cal_jobs_saved=save,probe_jobs_saved=int((bc[32:].astype(int)-cost[32:]).sum()),
                         weight_squared_change=sq,score=None if save<=0 else sq/save,nodes=len(nd))
                allrows.append(row)
                if save>0:options.append((sq/save,sq,a,b,gw,row))
        if options:
            best=min(options,key=lambda x:x[:4]);adapt[:,g*16:(g+1)*16]=best[4];chosen.append(best[5])
        else:chosen.append(dict(g=g,selection='unchanged: no candidate saves calibration T10 jobs'))
        nearest=min((r for r in allrows if r['g']==g),key=lambda r:(r['weight_squared_change'],r['a'],r['b']))
        integer_pairs.append(nearest)
        delta=d[g,nearest['a']]-d[g,nearest['b']];cols=np.flatnonzero(delta);sign=delta[cols]
        for h in range(384):integer_nearest[h,g*16+cols]=bounded_zero_sum(w[h,g*16+cols]*sign)[1]*sign
        print('selected',chosen[-1],flush=True)
    np.save(HERE/'adapt_W.npy',adapt.astype(np.int8))
    with (HERE/'pair_choices.csv').open('w') as f:
        wr=csv.DictWriter(f,fieldnames=list(allrows[0]));wr.writeheader();wr.writerows(allrows)
    p=(src['projected_g_int'].transpose(0,2,1,3)).astype(np.int64)
    fixed=np.load(B/'support_lut_execution_20260915/cases.npz');A=fixed['A'].astype(np.int64)
    tau=np.concatenate([fixed['tau'][h] for h in range(4)],axis=-1)
    pos=np.concatenate([fixed['positive_gain'][h] for h in range(4)])
    const=np.concatenate([fixed['constant_channels'][h] for h in range(4)])
    cgate=np.concatenate([fixed['constant_gate'][h] for h in range(4)],axis=-1)
    Y0=np.einsum('fptc,hc->fpth',p,w);U0=np.einsum('ts,fpsh->fpth',A,Y0)
    gate=lambda u:np.where(const[None,None,None,:],cgate[None,None],np.where(pos[None,None,None,:],u>=tau[None,None],u<=tau[None,None]))
    G0=gate(U0);variants={};tables={}
    for name,wt in [('original',w),('nearest_response',old),('integer_nearest',integer_nearest),('producer_cost',adapt)]:
        canon,L=canonical(d,wt);Y=np.einsum('fptc,hc->fpth',p,wt);U=np.einsum('ts,fpsh->fpth',A,Y)
        requests=[];class_nodes=[];roots=[]
        for prof in profiles:
            graph=Diagram();rootlist=[];counts=[]
            for g in range(6):
                var=prof[f'variables_g{g}'];exp=(((np.arange(1<<len(var))[:,None]>>np.arange(len(var)))&1)*(1<<var.astype(np.uint32))).sum(1)
                rootlist.append(graph.build(canon[g,labels[g][exp]],var))
            nodes=np.asarray(graph.nodes,dtype=np.uint32)
            for g in range(6):
                assert np.array_equal(eval_dag_all(nodes,rootlist[g],raw16),canon[g,labels[g]])
                _,ct=eval_packet_dag(nodes,rootlist[g],words[:,:,g],prof[f'variables_g{g}']);counts.append(ct)
            requests.append(np.asarray(counts).sum(0).reshape(2,32).sum(1).tolist())
            class_nodes.append(encode(nodes));roots.append(rootlist)
        variants[name]=dict(weight_squared_change=int(np.square(wt-w).sum()),weight_RMS_change=float(np.sqrt(np.square(wt-w).mean())),
            local_Y_relative_RMSE=[float(np.sqrt(np.square(Y[i]-Y0[i]).sum()/max(1,np.square(Y0[i]).sum()))) for i in range(2)],
            local_gate_flips=[int(np.count_nonzero(gate(U)[i]!=G0[i])) for i in range(2)],local_gate_denominator_each=int(G0[0].size),
            class_jobs_natural_and_entropy_by_frame=requests,distinct_responses_per_g=[len(np.unique(canon[g])) for g in range(6)],
            graph_nodes_by_order=list(map(len,class_nodes)),L_min=int(L.min()),L_max=int(L.max()))
        tables[name]=(canon,class_nodes,roots)
    # Rewrite static class tables only; X/A/tau/D and code DAG remain unchanged.
    # C++ recalculates raw source MAC/Hamming and follows canonical labels.
    original=(P/'source.bin').read_bytes();stream=io.BytesIO(original)
    header=stream.read(200+80+1536);op=[]
    for order in range(2):
        n=struct.unpack('<I',stream.read(4))[0];code=stream.read(n*8)
        nc=struct.unpack('<I',stream.read(4))[0];stream.read(nc*8)
        root=np.frombuffer(stream.read(24),dtype='<u2').copy();stream.read(96);rank=stream.read(96)
        op.append((n,code,root,rank))
    cases=stream.read()
    for name,path in [('producer_cost',HERE/'source.bin'),('integer_nearest',HERE/'integer_control'/'source.bin')]:
        path.parent.mkdir(exist_ok=True)
        canon,nodepack,rootpack=tables[name]
        with path.open('wb') as f:
            f.write(header)
            for order,(n,code,root,rank) in enumerate(op):
                assert len(nodepack[order])<=2180
                f.write(struct.pack('<I',n));f.write(code)
                f.write(struct.pack('<I',len(nodepack[order])));f.write(nodepack[order].astype('<u8').tobytes())
                root[6:]=rootpack[order];f.write(root.tobytes());f.write(canon.tobytes());f.write(rank)
            f.write(cases)
    canon,nodepack,rootpack=tables['producer_cost']
    np.savez(HERE/'adapt_tables.npz',canonical=canon,class_nodes_natural=nodepack[0],class_nodes_entropy=nodepack[1],roots=np.asarray(rootpack,dtype=np.uint16))
    assert np.array_equal(integer_nearest,old)
    assert (HERE/'integer_control/source.bin').read_bytes()==original
    out=dict(method='one nonzero pair/group, min squared integer W change per saved T10 source job on training frame0',
             selected=chosen,integer_nearest_pairs=integer_pairs,variants=variants,raw_source='2 training frames, 32 fixed sampled P each; frame1 not used for selection',
             local_tau='first validation tile H0..3 fixed integer tau/gain contract, NOT dynamic training-frame BN or AEE',
             new_AEE=None,RTL_cycles=None,
             integer_L2_control={'parameters_graphs_inputs':'Entire export identical to existing nearest-response fixture',
                                 'new_RTL_run':False,'reuse':'existing root replay, identical full stimulus and static metadata'})
    (HERE/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(variants,indent=2),flush=True)

if __name__=='__main__':main()
