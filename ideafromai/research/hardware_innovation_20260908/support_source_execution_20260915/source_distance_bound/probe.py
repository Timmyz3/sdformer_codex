"""Exact dominance tree proof and frozen-input source opportunity; no RTL helper."""
from pathlib import Path
from functools import lru_cache
import sys,json
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;SRC=HERE.parent
sys.path.insert(0,str(SRC/'zero_response_reopen/frontier_retained'))
from profile import frontier_work
POP=np.array([i.bit_count() for i in range(65536)],dtype=np.int16)

def tree(dictionary,canonical,order):
    pairxor=dictionary[:,None]^dictionary[None,:]
    tie=np.arange(16)[None,:]<np.arange(16)[:,None]
    nodes=[(0,0,0)]*16
    @lru_cache(None)
    def build(known,word):
        d=POP[(dictionary^word)&known];m=POP[pairxor&(~known&65535)]
        diff=d[:,None]-d[None,:]
        alive=np.flatnonzero(~((diff>m)|((diff==m)&tie)).any(1))
        assert len(alive)>0
        labels=canonical[alive]
        if np.all(labels==labels[0]):return int(labels[0])
        bits=int(np.bitwise_or.reduce(dictionary[alive])^np.bitwise_and.reduce(dictionary[alive]))&~known&65535
        v=next(int(v) for v in order if bits&(1<<int(v)))
        lo=build(known|(1<<v),word);hi=build(known|(1<<v),word|(1<<v))
        # Deliberately no lo==hi elimination or node coalescing: these are
        # actual paid queries of the proposed dominance algorithm.
        idx=len(nodes);nodes.append((v,lo,hi));return idx
    root=build(0,0)
    return np.asarray(nodes,dtype=np.uint32),root

def evaluate(nodes,root,words):
    state=np.full(words.shape,root,dtype=np.int64);count=np.zeros(words.shape,dtype=np.int64)
    for _ in range(17):
        live=state>=16
        if not np.any(live):return state.astype(np.uint8),count
        a=nodes[state[live]];count[live]+=1
        state[live]=np.where((words[live]>>a[:,0])&1,a[:,2],a[:,1])
    raise AssertionError('more than16 distinct known-bit queries')

def main():
    q=np.load(SRC/'prefix_tables_entropy.npz');raw=np.load(SRC/'source_class_adapt/expanded_sources/source_cases.npz')
    d=raw['D'].astype(np.int64);dw=(d*(1<<np.arange(16))).sum(-1).astype(np.uint16)
    canon={'code':np.tile(np.arange(16,dtype=np.uint8),(6,1)),
           'old_class':q['response_canonical_code'],
           'adapt_class':np.load(SRC/'source_class_adapt/adapt_tables.npz')['canonical'],
           'zero_class':np.load(SRC/'zero_response_reopen/zero_tables.npz')['canonical']}
    originals={'old_class':np.load(SRC.parent/'support_lut_execution_20260915/response_class_W.npy'),
               'adapt_class':np.load(SRC/'source_class_adapt/adapt_W.npy'),
               'zero_class':np.load(SRC/'zero_response_reopen/Wz.npy')}
    for name,w in originals.items():
        L=np.einsum('gkc,hgc->gkh',d,w.astype(np.int64).reshape(384,6,16))
        for g in range(6):assert np.array_equal(L[g],L[g,canon[name][g]])
    words=np.arange(65536,dtype=np.uint16);truth=[];out={};storage={}
    for g in range(6):truth.append(POP[words[:,None]^dw[g][None,:]].argmin(1))
    rank=np.full((6,16),15,dtype=int)
    for g in range(6):rank[g,q[f'variables_g{g}']]=np.arange(len(q[f'variables_g{g}']))
    bits=raw['raw_g_int'].transpose(0,2,1,3)
    for name,canonical in canon.items():
        allnodes=[(0,0,0)]*16;roots=[];nodecount=[]
        for g in range(6):
            n,r=tree(dw[g],canonical[g],q[f'variables_g{g}'])
            got,ct=evaluate(n,r,words);assert np.array_equal(got,canonical[g,truth[g]])
            offset=len(allnodes)-16
            for var,lo,hi in n[16:]:allnodes.append((int(var),int(lo)+(offset if lo>=16 else 0),int(hi)+(offset if hi>=16 else 0)))
            roots.append(r+offset if r>=16 else r);nodecount.append(len(n))
        n=np.asarray(allnodes,dtype=np.uint64);enc=n[:,1]|(n[:,2]<<16)|(n[:,0]<<32)
        assert len(n)<65536
        perframe=[]
        for frame in range(32):perframe.append(frontier_work(bits[frame],enc,np.array(roots),rank))
        out[name]=dict(proof_inputs=6*65536,proof='PASS; lowest-index Hamming tie',algorithm_tree_nodes_by_group=nodecount,
                       frame_work=perframe,first_two={k:sum(x[k] for x in perframe[:2]) for k in perframe[0]},
                       all32={k:sum(x[k] for x in perframe) for k in perframe[0]},graph_words_required=0)
        storage[name+'_nodes']=n.astype(np.uint32);storage[name+'_roots']=np.array(roots,dtype=np.uint16);storage[name+'_canonical']=canonical
        print(json.dumps(dict(name=name,first_two=out[name]['first_two'],all32=out[name]['all32'],proof='PASS')),flush=True)
    np.savez(HERE/'cpu_reference_trees.npz',**storage)
    result=dict(bound='delete k iff dk-dj>m or equality and j<k; all16 j are witnesses',
                tree_usage='CPU proof/opportunity only; forbidden as RTL candidate oracle',selection='fixed D-only existing order, no refit',
                scope='32 training frames P32, real A0*X source gate captures; no AEE',variants=out)
    (HERE/'probe.json').write_text(json.dumps(result,separators=(',',':'))+'\n')

if __name__=='__main__':main()
