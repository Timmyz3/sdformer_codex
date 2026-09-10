"""Ordinary same-W time-address packing opportunities, no new service model."""
from pathlib import Path
import json
import sys
sys.dont_write_bytecode=True
import numpy as np
from finite_frame_service import words_from_gates,accepted_and_needs
from full_frame_requests import unpack

HERE=Path(__file__).resolve().parent;JOINT=HERE.parent
POPC=np.array([i.bit_count() for i in range(1024)],np.uint8)


def count(words,need):
    out=dict(fixed_T_issue=0,one_per_Pbank_issue=0,cyclic_bank_issue=0,arbitrary_crossbar_lower_bound=0,
             active_P_H8_updates=0,W_vectors=0)
    for start in range(0,len(words),256):
        z=words[start:start+256]&need[start:start+256,None]
        tp=[((z>>(p*10))&1023).astype(np.uint16) for p in range(4)]
        n=np.stack([POPC[a] for a in tp])
        union=tp[0]|tp[1]|tp[2]|tp[3]
        out['fixed_T_issue']+=int(POPC[union].sum(dtype=np.int64))
        out['one_per_Pbank_issue']+=int(n.max(0).sum(dtype=np.int64))
        # Same four banks, fixed bijectionbank=(p+t)%4,row=10*hctx+t.
        # Eachbank has exactlyone logicalp for a givent, so its10-bit L1D
        # needs neither an additionalreadport nor an arbitrary4x4 crossbar.
        cyclic=[]
        for bank in range(4):
            mask=np.zeros_like(union)
            for p in range(4):
                tm=sum(1<<t for t in range(10) if (p+t)%4==bank)
                mask|=tp[p]&tm
            cyclic.append(POPC[mask])
        out['cyclic_bank_issue']+=int(np.maximum.reduce(cyclic).sum(dtype=np.int64))
        total=n.sum(0,dtype=np.int64)
        out['arbitrary_crossbar_lower_bound']+=int(((total+3)//4).sum())
        out['active_P_H8_updates']+=int(total.sum())
        out['W_vectors']+=int((union!=0).sum())
    return out


def main():
    first='000_zurich_city_09_a_0001'
    source_path=JOINT/'full_capture4/capture/01_local_train16_common3_group_exact'/first/'gates.npz'
    source=unpack(np.load(source_path),'source_gate_bits','source_gate_shape')
    words=words_from_gates(source);G=len(words)
    full=np.full(G,(1<<40)-1,np.uint64)
    ordinary=count(words,full);ordinary={k:12*v for k,v in ordinary.items()}
    header=np.bitwise_or.reduce(words,axis=1)
    empty=np.stack([(((header>>(p*10))&1023)[:,None]&(np.uint64(1)<<np.arange(10,dtype=np.uint64)))==0 for p in range(4)],1)
    result=dict(scope='One full240x320/T10/C96 source frame; actualall-nonzeroFP32W. Counts are32-lane issue lowerbounds,not service time.',
                mechanism='OneH8 Wvector broadcasts to thefourphysicalPbanks. EachP independentlyselects itsnextactiveT. No crossbar or extraYread/writeport is required by the declaredPbank mapping;4x10-bitL1D,timeindices andRAWforwarding are still hardwarecosts.',
                limit='PackingfromdifferentP intoonePbank is illegalwithout an additionalport or relocation; ceil(total/4) is only a lowerbound. Also count ordinaryfixedbank=(p+t)%4 androw=10*hctx+t: samefourports, fixed40-bit permutation, four10-bit selectors. OrdinaryfullT andconditional receivebothcontrols.',
                ordinary=ordinary,axes={},source_empty={})
    for name,folder,axis in [('row34','optimized_prefix_train16','row34_packed_word'),('common3','local_train16','common3_group')]:
        pred=np.load(JOINT/folder/(axis+'.npz'));prefix=pred['prefix'].tolist();a=pred['temporal_q14']!=0
        cp=JOINT/'accepted_capture1/capture'/('00_optimized_prefix_train16_row34_packed_word_conditional' if name=='row34' else '01_local_train16_common3_group_conditional')/first
        z=np.load(cp/'gates.npz');needs,_,review=accepted_and_needs(z,None,a,prefix,'conditional')
        pm=sum(1<<(p*10+t) for p in range(4) for t in prefix)
        pre=count(words,np.full(G,pm,np.uint64));pre={k:12*v for k,v in pre.items()}
        tail={k:0 for k in pre}
        for h in range(12):
            q=count(words,needs[:,h]&np.uint64(((1<<40)-1)^pm))
            for k,v in q.items():tail[k]+=v
        total={k:pre[k]+tail[k] for k in pre}
        total['packing_reduces_issue_fraction']=1-total['one_per_Pbank_issue']/total['fixed_T_issue']
        total['vs_packed_ordinary']=total['one_per_Pbank_issue']/ordinary['one_per_Pbank_issue']
        result['axes'][name]=dict(prefix=pre,tail=tail,total=total,accepted_review=review)
        zero_products=int((empty.sum((0,1),dtype=np.int64)*a.sum(0)).sum())*96
        all_empty_rows=np.stack([empty[:,:,a[t]].all(-1) for t in range(10)],-1)
        result['source_empty'][name]=dict(rawY_empty_scalar_values=int(empty.sum())*96,
            related_A_products_with_known_constantY=zero_products,
            scalar_output_rows_whose_allY_are_constant=int(all_empty_rows.sum())*96,
            complete_nonzero_A_scalar_products=int(240*320*96*a.sum()),
            maximum_removable_A_products_fraction=zero_products/int(240*320*96*a.sum()),
            note='KnownrawY=0 meansnorm1Y=BNoffset,notnumericalzero. Fractionis anoptimistic constant-compilation bound: partialemptysubsets requirechargedperH constants orsharedfactor arithmetic; fullyconstantrows canuse aprecompiledgate.')
    result['ordinary']['packing_reduces_issue_fraction']=1-ordinary['one_per_Pbank_issue']/ordinary['fixed_T_issue']
    (HERE/'simd_issue_bound_frame0.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
