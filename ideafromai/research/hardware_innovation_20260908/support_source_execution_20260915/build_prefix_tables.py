"""Exact fixed-order classifiers for real D, code and whole-response class.

Raw words use bit c for local channel c. One physical source request produces
all T10 bits of one channel. Only static informative channels are baseline work.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent


def prefix_table(labels,n):
    cert=[]
    for depth in range(n+1):
        completions=labels.reshape(-1,1<<depth)
        low=completions.min(0);high=completions.max(0)
        cert.append(np.where(low==high,low,255).astype(np.uint8))
    flat=np.concatenate(cert)
    first=np.full(len(labels),n,dtype=np.uint8)
    raw=np.arange(len(labels),dtype=np.uint32)
    unresolved=np.ones(len(labels),dtype=bool)
    for depth in range(n+1):
        value=cert[depth][raw&((1<<depth)-1)]
        lock=(value!=255)&unresolved
        assert np.array_equal(value[lock],labels[lock])
        first[lock]=depth;unresolved[lock]=False
    assert not unresolved.any()
    return flat,first


class Diagram:
    def __init__(self):
        self.nodes=[[255,i,i] for i in range(16)]
        self.unique={}
    def build(self,labels,variables):
        current=labels.astype(np.uint32)
        for pos in range(len(variables)-1,-1,-1):
            half=len(current)//2
            a,b=current[:half],current[half:]
            parent=[]
            for low,high in zip(a,b):
                low,high=int(low),int(high)
                if low==high:parent.append(low);continue
                key=(int(variables[pos]),low,high)
                if key not in self.unique:
                    self.unique[key]=len(self.nodes);self.nodes.append(list(key))
                parent.append(self.unique[key])
            current=np.asarray(parent,dtype=np.uint32)
        return int(current[0])


def eval_dag_all(nodes,root,raw):
    state=np.full(len(raw),root,dtype=np.uint32)
    for _ in range(16):
        live=state>=16
        if not live.any():break
        row=nodes[state[live]]
        bit=(raw[live]>>row[:,0])&1
        state[live]=np.where(bit,row[:,2],row[:,1])
    assert (state<16).all()
    return state.astype(np.uint8)


def eval_packet_dag(nodes,root,words,variables):
    """words[N,T10]. Request the earliest live variable in the fixed order.

    All 10 gates are produced on that channel; only matching nodes advance.
    Counters are logical source requests, not checker RAM latency.
    """
    assert words.ndim==2 and words.shape[1]==10
    state=np.full(words.shape,root,dtype=np.uint32)
    count=np.zeros(len(words),dtype=np.uint8)
    rank=np.full(256,255,dtype=np.uint8)
    rank[variables]=np.arange(len(variables),dtype=np.uint8)
    for _ in range(16):
        live=state>=16
        if not live.any():break
        var=np.where(live,nodes[state,0],255)
        next_rank=rank[var].min(1)
        active=next_rank!=255
        count+=active.astype(np.uint8)
        use=live&(rank[var]==next_rank[:,None])
        row=nodes[state[use]]
        bits=(words[use]>>row[:,0])&1
        state[use]=np.where(bits,row[:,2],row[:,1])
    assert (state<16).all()
    return state.astype(np.uint8),count


def packet_stats(words,leaf,depth,nodes,roots,variables):
    # Input shape [packets,10,6], independent of any distribution assumption.
    result={}
    for mode in ('code','class'):
        prefix=[];dag=[]
        for g in range(6):
            raw=words[:,:,g]
            expected=leaf[mode][g,raw]
            got,counts=eval_packet_dag(nodes[mode],roots[mode][g],raw,variables[g])
            assert np.array_equal(got,expected)
            p=depth[mode][g,raw].max(1)
            assert (counts<=p).all()
            prefix.append(p);dag.append(counts)
        p=np.stack(prefix,axis=1);q=np.stack(dag,axis=1)
        result[mode]={'prefix_source_requests':int(p.sum()),'dag_source_requests':int(q.sum()),
                      'prefix_mean_per_packet_all6g':float(p.sum(1).mean()),
                      'dag_mean_per_packet_all6g':float(q.sum(1).mean()),
                      'prefix_mean_per_g':p.mean(0).tolist(),'dag_mean_per_g':q.mean(0).tolist(),
                      'dag_max_per_g':q.max(0).tolist()}
    result['packets']=len(words)
    result['static_full_requests']=len(words)*sum(map(len,variables))
    result['incremental_class_vs_code_prefix']=result['code']['prefix_source_requests']-result['class']['prefix_source_requests']
    result['incremental_class_vs_code_dag']=result['code']['dag_source_requests']-result['class']['dag_source_requests']
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw-words',type=Path,action='append',default=[],
                    help='Optional real NPZ: raw_words uint16 [10,P,6], before nearest-code projection.')
    ap.add_argument('--source-cases',type=Path,action='append',default=[],
                    help='Actual source PSN fixture: raw_g_int [frames,10,P,96].')
    ap.add_argument('--order',choices=('natural','entropy'),default='natural',
                    help='One permitted adaptation: decreasing binary D-column entropy, physical-index ties.')
    args=ap.parse_args()
    d=np.load(BASE/'algorithm/support_training/dictionary.npy').astype(np.int64)
    w=np.load(BASE/'support_lut_execution_20260915/response_class_W.npy').astype(np.int64)
    dictionary_words=(d*(1<<np.arange(16))).sum(-1).astype(np.uint16)
    raw=np.arange(1<<16,dtype=np.uint32)
    bits=((raw[:,None]>>np.arange(16))&1).astype(np.int16)
    pop=bits.sum(1).astype(np.uint8)
    response=np.einsum('gkc,hgc->gkh',d,w.reshape(384,6,16))
    canonical=np.zeros((6,16),dtype=np.uint8)
    for g in range(6):
        for k in range(16):
            same=np.all(response[g]==response[g,k],axis=1)
            canonical[g,k]=np.flatnonzero(same)[0]
    variables=[np.flatnonzero(np.any(d[g]!=d[g,0],axis=0)).astype(np.uint8) for g in range(6)]
    assert list(map(len,variables))==[9,13,15,8,11,8]
    if args.order=='entropy':
        # h2(n_ones/16) decreases with |n_ones-8|. This exact integer key
        # avoids floating-point ties and uses D only, never source activity.
        variables=[np.asarray(sorted(var.tolist(),key=lambda c:(abs(int(d[g,:,c].sum())-8),c)),dtype=np.uint8)
                   for g,var in enumerate(variables)]
    leaf={m:np.zeros((6,1<<16),dtype=np.uint8) for m in ('code','class')}
    depth={m:np.zeros((6,1<<16),dtype=np.uint8) for m in ('code','class')}
    diagrams={m:Diagram() for m in ('code','class')};roots={m:[] for m in diagrams}
    arrays={'dictionary_words':dictionary_words,'response_canonical_code':canonical}
    group_stats=[]
    for g in range(6):
        distances=pop[np.bitwise_xor(raw[:,None],dictionary_words[g])]
        labels=distances.argmin(1).astype(np.uint8)
        # Independent original nearest-code formula, preserving first-index tie.
        distance2=bits.sum(1)[:,None]+d[g].sum(1)[None,:]-2*(bits@d[g].T)
        assert np.array_equal(distances,distance2)
        assert np.array_equal(labels,distance2.argmin(1))
        leaf['code'][g]=labels;leaf['class'][g]=canonical[g,labels]
        var=variables[g];n=len(var)
        compact_bits=((np.arange(1<<n,dtype=np.uint32)[:,None]>>np.arange(n))&1)
        expanded=(compact_bits*(1<<var.astype(np.uint32))).sum(1).astype(np.uint32)
        compact_from_raw=(bits[:,var]*(1<<np.arange(n))).sum(1).astype(np.uint32)
        assert np.array_equal(labels,labels[expanded][compact_from_raw])
        item={'g':g,'informative_columns':var.tolist(),'constant_columns':np.setdiff1d(np.arange(16),var).tolist(),
              'raw_inputs':len(raw),'ties':int((np.count_nonzero(distances==distances.min(1)[:,None],axis=1)>1).sum()),
              'prefix_nodes':(1<<(n+1))-1,'modes':{}}
        arrays[f'variables_g{g}']=var
        for mode in ('code','class'):
            compact=leaf[mode][g,expanded]
            cert,first=prefix_table(compact,n)
            arrays[f'{mode}_prefix_g{g}']=cert
            arrays[f'{mode}_compact_leaf_g{g}']=compact
            depth[mode][g]=first[compact_from_raw]
            roots[mode].append(diagrams[mode].build(compact,var))
            hist=np.bincount(first,minlength=n+1)
            cdf=np.cumsum(hist)/len(first)
            item['modes'][mode]={'single_uniform_mean_informative_channels':float(first.mean()),
                                 'single_uniform_stop_histogram':hist.tolist(),
                                 'T10_iid_uniform_prefix_mean_channels':float(np.sum(1-cdf[:n]**10)),
                                 'worst_informative_channels':int(first.max()),
                                 'zero_input_stop':int(depth[mode][g,0]),
                                 'onehot_input_stops':depth[mode][g,1<<np.arange(16)].tolist()}
        group_stats.append(item)
    nodes={m:np.asarray(diagrams[m].nodes,dtype=np.uint32) for m in diagrams}
    roots={m:np.asarray(roots[m],dtype=np.uint32) for m in roots}
    pointer_bits=max(len(nodes[m])-1 for m in nodes).bit_length()
    node_bits=4+2*pointer_bits
    word_bits=32 if node_bits<=32 else 64
    for mode in nodes:
        for g in range(6):
            assert np.array_equal(eval_dag_all(nodes[mode],roots[mode][g],raw),leaf[mode][g])
        arrays[f'{mode}_dag_nodes']=nodes[mode]
        arrays[f'{mode}_dag_roots']=roots[mode]
        records=nodes[mode][16:].astype(np.uint64)
        packed=(records[:,0]<<(2*pointer_bits))|(records[:,2]<<pointer_bits)|records[:,1]
        arrays[f'{mode}_dag_packed']=packed.astype(np.uint32 if word_bits==32 else np.uint64)
        # Root RTL's fixed schema; both arms admit the same physical capacity.
        assert len(nodes[mode])<65536
        n64=nodes[mode].astype(np.uint64)
        terminal=np.arange(len(n64))<16
        fixed64=(n64[:,1]|(n64[:,2]<<16)|((n64[:,0]&15)<<32)
                 |(terminal.astype(np.uint64)<<36)
                 |(np.where(terminal,np.arange(len(n64)),0).astype(np.uint64)<<37))
        arrays[f'{mode}_nodes64']=fixed64
        assert np.array_equal(fixed64&65535,n64[:,1])
        assert np.array_equal((fixed64>>16)&65535,n64[:,2])
        assert np.array_equal((fixed64[16:]>>32)&15,n64[16:,0])
        assert np.array_equal((fixed64>>36)&1,terminal)
        assert np.array_equal((fixed64[:16]>>37)&15,np.arange(16))
        arrays[f'{mode}_full_leaf']=leaf[mode]
        arrays[f'{mode}_full_stop_depth']=depth[mode]
    arrays['common_pointer_bits']=np.asarray(pointer_bits,dtype=np.uint8)
    arrays['common_record_bits']=np.asarray(word_bits,dtype=np.uint8)
    arrays['roots']=np.stack([roots['code'],roots['class']]).astype(np.uint16)
    rng=np.random.default_rng(20260915)
    synth=rng.integers(0,65536,size=(4096,10,6),dtype=np.uint16)
    # These are synthetic scheduling checks, never labelled as captured source.
    directed=np.stack([np.zeros((10,6),dtype=np.uint16),np.full((10,6),65535,dtype=np.uint16),
                       np.tile((1<<np.arange(10,dtype=np.uint16))[:,None],(1,6)),
                       np.tile(np.asarray([0,65535]*5,dtype=np.uint16)[:,None],(1,6))])
    result={'function':'argmin_k popcount(raw16 xor D[g,k]); ties choose smallest k; class=canonical_whole_H384_response[k]',
            'source_request':'one local physical channel returns all ten temporal gate bits; common fixed-order evaluation',
            'variable_order':args.order,
            'order_rule':'natural ascending physical channel' if args.order=='natural' else 'D-only descending h2(column one-count/16), equivalent ascending abs(one-count-8); physical-index ties; no fitted data or order search',
            'D_column_one_counts':d.sum(1).tolist(),
            'static_baseline_channels_per_g':list(map(len,variables)),'static_baseline_total':64,
            'static_deleted_common_Hamming_columns':32,'groups':group_stats,
            'exhaustive_correctness':{'groups':6,'raw_inputs_per_group':65536,'code_and_class_dag_and_prefix':'PASS',
                                      'distance_formula_and_tie':'PASS','static_common_term_deletion':'PASS'},
            'class_earlier_prefix_inputs_per_g':[(depth['class'][g]<depth['code'][g]).sum().item() for g in range(6)],
            'table_layout':{'prefix_index':'(1<<known_informative_depth)-1 + little_endian_prefix_bits',
                            'prefix_unresolved':255,'prefix_terminal':'0..15 actual code or canonical code, uint8 storage; logical 5-bit encoding uses 16 for unresolved',
                            'dag_terminal_ids':'0..15, no record read required',
                            'dag_record_address':'node_id - 16',
                            'dag_record_fields':'[var_local4 | high_child_Pbits | low_child_Pbits]',
                            'common_pointer_bits':pointer_bits,'common_record_bits':word_bits,
                            'prefix_nodes_per_arm':sum(x['prefix_nodes'] for x in group_stats),
                            'prefix_logical_5bit_bytes_per_arm':(sum(x['prefix_nodes'] for x in group_stats)*5+7)//8,
                            'dag_arms':{m:{'nodes_including_16_terminals':len(nodes[m]),
                                           'nonterminal_records':len(nodes[m])-16,
                                           'record_bytes':(len(nodes[m])-16)*(word_bits//8),
                                           'root_pointer_bytes_if_packed':(6*pointer_bits+7)//8} for m in nodes},
                            'T10_node_id_holding_bits':10*pointer_bits,
                            'root_RTL_fixed64_schema':'lo[15:0],hi[31:16],physical_var[35:32],terminal[36],label[40:37]; terminal var ignored',
                            'fixed64_nodes_capacity_each_arm':((max(map(len,nodes.values()))+1)//2)*2,
                            'fixed64_roots_bytes_each_arm':12,'fixed64_T10_node_id_holding_bytes':20,
                            'entropy_physical_to_order_rank_table_if_used':'96x4 bits = 48 bytes, plus configuration; not included in node memory',
                            'not_included':'source PSN, checker read/grant latency, current-record cache, proof output, configuration and backpressure'},
            'uniform_T10_analytic_prefix_total':{m:sum(x['modes'][m]['T10_iid_uniform_prefix_mean_channels'] for x in group_stats) for m in nodes},
            'synthetic_uniform_4096_T10_packets':packet_stats(synth,leaf,depth,nodes,roots,variables),
            'directed_T10_packets':packet_stats(directed,leaf,depth,nodes,roots,variables),
            'real_raw_captures':[]}
    for path in args.raw_words:
        with np.load(path) as z:raw_words=np.asarray(z['raw_words'])
        assert raw_words.ndim==3 and raw_words.shape[0]==10 and raw_words.shape[2]==6
        assert raw_words.min()>=0 and raw_words.max()<65536
        words=np.transpose(raw_words,(1,0,2)).astype(np.uint16)
        s=packet_stats(words,leaf,depth,nodes,roots,variables)
        s['path']=str(path);result['real_raw_captures'].append(s)
    for path in args.source_cases:
        with np.load(path) as z:
            gate=np.asarray(z['raw_g_int'])
            expected_code=np.asarray(z['code_index_int'])
            assert np.array_equal(z['D'],d)
        assert gate.ndim==4 and gate.shape[1]==10 and gate.shape[-1]==96
        assert np.isin(gate,[0,1]).all()
        frames,_,positions,_=gate.shape
        raw_words=(gate.reshape(frames,10,positions,6,16).astype(np.uint32)*(1<<np.arange(16))).sum(-1)
        words=raw_words.transpose(0,2,1,3).reshape(-1,10,6).astype(np.uint16)
        expected_code=expected_code.reshape(-1,10,6)
        for g in range(6):assert np.array_equal(leaf['code'][g,words[:,:,g]],expected_code[:,:,g])
        s=packet_stats(words,leaf,depth,nodes,roots,variables)
        s.update(path=str(path),frames=frames,positions_per_frame=positions,
                 source_field='raw_g_int before projection; actual A16Q12/X24Q16/thresholdQ28 source PSN',
                 split='first 2 training frames x 32 positions; not held out; no AEE claim',
                 code_index_int_crosscheck='PASS')
        result['real_raw_captures'].append(s)
    stem='prefix_tables' if args.order=='natural' else 'prefix_tables_entropy'
    np.savez_compressed(HERE/(stem+'.npz'),**arrays)
    (HERE/(stem+'.json')).write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps({k:result[k] for k in ('table_layout','uniform_T10_analytic_prefix_total',
                      'synthetic_uniform_4096_T10_packets','real_raw_captures','exhaustive_correctness')},ensure_ascii=False))


if __name__=='__main__':main()
