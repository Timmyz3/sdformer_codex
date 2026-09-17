"""Add exactly 90 zero-pair candidates; reuse 630 old scores unchanged."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,csv,json,io,struct
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent
BASE=SOURCE.parent
sys.path[:0]=[str(SOURCE),str(BASE/'support_lut_execution_20260915')]
from build_prefix_tables import Diagram,eval_packet_dag,eval_dag_all
from probe_response_projection import read_torch,bounded_zero_sum

def encode(nodes):
    n=np.asarray(nodes,dtype=np.uint64);ids=np.arange(len(n),dtype=np.uint64);term=ids<16
    return n[:,1]|(n[:,2]<<16)|((n[:,0]&15)<<32)|(term.astype(np.uint64)<<36)|(np.where(term,ids,0)<<37)

def project(w,d,g,a,b):
    delta=d[g,a]-d[g,b];cols=np.flatnonzero(delta);sign=delta[cols]
    wg=w[:,g*16:(g+1)*16].copy()
    for h in range(384):wg[h,cols]=bounded_zero_sum(wg[h,cols]*sign)[1]*sign
    assert np.all(wg@delta==0) and wg.min()>=-127 and wg.max()<=127
    return wg

def main():
    src=np.load(SOURCE/'source_cases.npz');d=src['D'].astype(np.int64)
    w=read_torch(BASE/'algorithm/support_training/forced_code_weight_int8.pt').astype(np.int64)
    profiles=[np.load(SOURCE/'prefix_tables.npz'),np.load(SOURCE/'prefix_tables_entropy.npz')]
    raw=src['raw_g_int'].transpose(0,2,1,3).reshape(64,10,6,16)
    words=(raw*(1<<np.arange(16))).sum(-1).astype(np.uint16)
    allraw=np.arange(65536,dtype=np.uint32);bits=((allraw[:,None]>>np.arange(16))&1).astype(np.int16)
    original_rows=[]
    for x in csv.DictReader((SOURCE/'source_class_adapt/pair_choices.csv').open()):
        row={k:None if v=='' else float(v) if k=='score' else int(v) for k,v in x.items()};original_rows.append(row)
    assert len(original_rows)==630 and len({(r['g'],r['a'],r['b']) for r in original_rows})==630
    added=[];labels=[];base=[];selected=[];l2pairs=[];wz=w.copy();wl2=w.copy()
    for g in range(6):
        lab=(bits.sum(1)[:,None]+d[g].sum(1)[None,:]-2*bits@d[g].T).argmin(1).astype(np.uint8);labels.append(lab)
        var=profiles[1][f'variables_g{g}'];exp=(((np.arange(1<<len(var))[:,None]>>np.arange(len(var)))&1)*(1<<var.astype(np.uint32))).sum(1)
        graph=Diagram();root=graph.build(lab[exp],var);nodes=np.asarray(graph.nodes,dtype=np.uint32)
        _,bc=eval_packet_dag(nodes,root,words[:,:,g],var);base.append(bc)
        for k in range(1,16):
            mapping=np.arange(16,dtype=np.uint8);mapping[k]=0
            graph=Diagram();root=graph.build(mapping[lab[exp]],var);nodes=np.asarray(graph.nodes,dtype=np.uint32)
            assert np.array_equal(eval_dag_all(nodes,root,allraw),mapping[lab])
            _,ct=eval_packet_dag(nodes,root,words[:,:,g],var)
            saved=int((bc[:32].astype(int)-ct[:32]).sum())
            wg=project(w,d,g,0,k);sq=int(np.square(wg-w[:,g*16:(g+1)*16]).sum())
            added.append(dict(g=g,a=0,b=k,cal_jobs_saved=saved,probe_jobs_saved=int((bc[32:].astype(int)-ct[32:]).sum()),
                              weight_squared_change=sq,score=None if saved<=0 else sq/saved,nodes=len(nodes)))
        all_options=[r for r in original_rows+added if r['g']==g]
        options=[r for r in all_options if r['cal_jobs_saved']>0]
        if options:
            choice=min(options,key=lambda r:(r['score'],r['weight_squared_change'],r['a'],r['b']))
            selected.append(choice);wz[:,g*16:(g+1)*16]=project(w,d,g,choice['a'],choice['b'])
        else:selected.append(dict(g=g,selection='unchanged: no calibration T10 source gain'))
        nearest=min(all_options,key=lambda r:(r['weight_squared_change'],r['a'],r['b']));l2pairs.append(nearest)
        wl2[:,g*16:(g+1)*16]=project(w,d,g,nearest['a'],nearest['b'])
        print(json.dumps({'g':g,'selected':selected[-1],'ordinary_integer_L2':nearest}),flush=True)
    assert len(added)==90
    with (HERE/'zero_pair_choices.csv').open('w') as f:
        out=csv.DictWriter(f,fieldnames=list(added[0]));out.writeheader();out.writerows(added)
    np.save(HERE/'Wz.npy',wz.astype(np.int8));np.save(HERE/'integer_l2_W.npy',wl2.astype(np.int8))
    # Selection has ended. Frame1 and the wider training range are reporting only.
    extended=np.load(SOURCE/'source_class_adapt/expanded_sources/source_cases.npz')
    assert np.array_equal(extended['raw_g_int'][:2],src['raw_g_int'])
    extwords=(extended['raw_g_int'].transpose(0,2,1,3).reshape(1024,10,6,16)*(1<<np.arange(16))).sum(-1).astype(np.uint16)
    projected=src['projected_g_int'].transpose(0,2,1,3).astype(np.int64)
    back=np.load(BASE/'support_lut_execution_20260915/cases.npz');A=back['A'].astype(np.int64)
    ids=[int(np.flatnonzero((back['hblock']==h)&back['is_real'])[0]) for h in range(4)]
    tau=np.concatenate([back['tau'][i] for i in ids],axis=1);positive=np.concatenate([back['positive_gain'][i] for i in ids])
    constant=np.concatenate([back['constant_channels'][i] for i in ids]);cgate=np.concatenate([back['constant_gate'][i] for i in ids],axis=1)
    gate=lambda U:np.where(constant[None,None,None,:],cgate[None,None],np.where(positive[None,None,None,:],U>=tau[None,None],U<=tau[None,None]))
    Y0=np.einsum('fptc,hc->fpth',projected,w);U0=np.einsum('ts,fpsh->fpth',A,Y0);G0=gate(U0)
    weights=[('original',w),('previous_producer_cost',np.load(SOURCE/'source_class_adapt/adapt_W.npy').astype(np.int64)),
             ('zero_inclusive_producer_cost',wz),('zero_inclusive_integer_L2',wl2)]
    variants={}
    for name,wt in weights:
        L=np.einsum('gkc,hgc->gkh',d,wt.reshape(384,6,16));canonical=np.zeros((6,16),dtype=np.uint8)
        for g in range(6):
            for k in range(16):canonical[g,k]=np.flatnonzero(np.all(L[g]==L[g,k],axis=1))[0]
        tables=[];roots=[];counts=[]
        for order,prof in enumerate(profiles):
            graph=Diagram();rr=[]
            for g in range(6):
                var=prof[f'variables_g{g}'];exp=(((np.arange(1<<len(var))[:,None]>>np.arange(len(var)))&1)*(1<<var.astype(np.uint32))).sum(1)
                rr.append(graph.build(canonical[g,labels[g][exp]],var))
            nodes=np.asarray(graph.nodes,dtype=np.uint32);ct=[]
            for g in range(6):
                assert np.array_equal(eval_dag_all(nodes,rr[g],allraw),canonical[g,labels[g]])
                _,q=eval_packet_dag(nodes,rr[g],extwords[:,:,g],prof[f'variables_g{g}']);ct.append(q)
            counts.append(np.asarray(ct).sum(0).reshape(32,32).sum(1).tolist());tables.append(encode(nodes));roots.append(rr)
        if name=='zero_inclusive_producer_cost':stem='zero_tables'
        elif name=='zero_inclusive_integer_L2':stem='integer_l2_tables'
        else:stem=None
        if stem:np.savez(HERE/(stem+'.npz'),canonical=canonical,class_nodes_natural=tables[0],class_nodes_entropy=tables[1],roots=np.asarray(roots,dtype=np.uint16))
        Y=np.einsum('fptc,hc->fpth',projected,wt);U=np.einsum('ts,fpsh->fpth',A,Y);G=gate(U)
        variants[name]={'weight_squared_change':int(np.square(wt-w).sum()),'weight_range':[int(wt.min()),int(wt.max())],
            'Y_relative_RMSE_by_first_two_training_frames':[float(np.sqrt(np.square(Y[i]-Y0[i]).sum()/max(1,np.square(Y0[i]).sum()))) for i in range(2)],
            'U_relative_RMSE_by_first_two_training_frames':[float(np.sqrt(np.square(U[i].astype(np.float64)-U0[i]).sum()/max(1,np.square(U0[i].astype(np.float64)).sum()))) for i in range(2)],
            'local_gate_flips_by_first_two_training_frames':[int(np.count_nonzero(G[i]!=G0[i])) for i in range(2)],'gate_denominator_per_frame':122880,
            'source_jobs_by_order_then_training_frame':counts,'nodes_by_order':list(map(len,tables)),
            'classes_per_g':[len(np.unique(canonical[g])) for g in range(6)],'zero_class_members':[np.flatnonzero(canonical[g]==0).tolist() for g in range(6)],
            'L_range':[int(L.min()),int(L.max())]}
    zero_selected=[r for r in selected if r.get('a')==0]
    out={'B':'old zero-directory backend negative left source production untested; previous selector excluded zero',
         'selection':'reuse630 + exactly90 zero pairs; original-W symmetric integer projection; frame0 score/ties unchanged',
         'denominator':'explicit single-pair quotient complete-T10 source jobs, matching reused CSV; incidental full-W aliases counted only after selection',
         'selected':selected,'selected_zero_pairs':zero_selected,'ordinary_integer_L2_selected':l2pairs,'variants':variants,
         'exhaustive_65536_per_group_per_final_graph':'PASS','selection_frames':[0],'report_only_training_frames':list(range(1,32)),
         'AEE':None,'local_tau':'same fixed joined integer contract, not source-frame BN recalibration',
         'conditional_RTL_admission':bool(zero_selected) and sum(variants['zero_inclusive_producer_cost']['source_jobs_by_order_then_training_frame'][1][:2])<sum(variants['original']['source_jobs_by_order_then_training_frame'][1][:2])}
    (HERE/'probe.json').write_text(json.dumps(out,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps({'selected_zero_pairs':zero_selected,'variants':variants,'RTL_admission':out['conditional_RTL_admission']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
