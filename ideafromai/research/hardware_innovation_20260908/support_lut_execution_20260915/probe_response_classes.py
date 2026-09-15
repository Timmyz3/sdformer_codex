"""Exact response classes, then one fixed approximate level (one pair/K16).

Weights-only pair selection. No training, threshold/sparsity sweep, RTL or AEE.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='1'
import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
import numpy as np
from probe_response_projection import read_torch, bounded_zero_sum, table, gate

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
CAP=BASE/'algorithm/support_training'
WIDTHS=(12,96,384)


def classes(values,width):
    """Return actual content classes, including the all-zero response class."""
    out=[]
    for g in range(6):
        for h in range(0,384,width):
            members={}
            for k in range(values.shape[1]):
                key=tuple(map(int,values[g,k,h:h+width]))
                members.setdefault(key,[]).append(k)
            out.append({'g':g,'h':h,'classes':[
                {'codes':ids,'zero':not any(v)} for v,ids in members.items()]})
    return out


def class_summary(entries,ncodes):
    hist={}
    for x in entries:
        n=len(x['classes']);hist[str(n)]=hist.get(str(n),0)+1
    return {'blocks':len(entries),'codes_per_block':ncodes,'class_count_histogram':hist,
            'merged_classes':sum(len(y['codes'])>1 for x in entries for y in x['classes']),
            'aliased_code_slots':sum(ncodes-len(x['classes']) for x in entries),
            'examples':[x for x in entries if len(x['classes'])<ncodes][:12]}


def requests(entries,seen,width):
    words=updates=0
    # seen[g,tile,code] is real B32*T10 code presence, before dedup.
    for item in entries:
        for c in item['classes']:
            if c['zero']:continue
            live=seen[item['g']][:,c['codes']].any(axis=1)
            words+=int(live.sum())*(width//12)
            updates+=int(live.sum())
    return words,updates


def main():
    d=np.load(CAP/'dictionary.npy').astype(np.int64)
    w=read_torch(CAP/'forced_code_weight_int8.pt').astype(np.int64)
    key='sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.'
    params=read_torch(BASE/'algorithm/integer_s0_valid825/integer_parameters.pt')[key]
    a=params['temporal_int16'].astype(np.int64)
    eye=np.repeat(np.eye(16,dtype=np.int64)[None],6,axis=0)
    zoh=np.concatenate([np.zeros((6,1,16),dtype=np.int64),eye],axis=1)
    alphabets={'support16':d,'zero_onehot17':zoh,'union32':np.concatenate([d,eye],axis=1)}
    exact={}
    for name,alphabet in alphabets.items():
        values=table(alphabet,w)
        exact[name]={str(width):class_summary(classes(values,width),len(alphabet[0])) for width in WIDTHS}
        for width in WIDTHS:
            payloads=values.reshape(-1,width)
            nonzero=payloads[np.any(payloads!=0,axis=1)]
            distinct=len(np.unique(nonzero,axis=0))
            exact[name][str(width)]['global_nonzero_payload_rows']=len(nonzero)
            exact[name][str(width)]['global_distinct_nonzero_payloads']=distinct
            exact[name][str(width)]['global_nonzero_payload_duplicates']=len(nonzero)-distinct
    # Only if the original full rows have no exact aliases: one approximate level.
    assert exact['support16']['384']['aliased_code_slots']==0
    pairs=[]
    wf=w.astype(float).copy();wi=w.copy();wn=w.copy();ws=w.copy()
    for g in range(6):
        choices=[]
        wg=w[:,16*g:16*(g+1)]
        for k in range(1,16):
            for l in range(k+1,16):
                delta=d[g,k]-d[g,l]
                r=wg@delta
                cost=float(r@r)/int(delta@delta)
                choices.append((cost,k,l))
        cost,k,l=min(choices)
        delta=d[g,k]-d[g,l]
        cols=np.flatnonzero(delta);sign=delta[cols]
        for h in range(384):
            f,i=bounded_zero_sum(wg[h,cols]*sign)
            wf[h,16*g+cols]=f*sign
            wi[h,16*g+cols]=i*sign
        ws[:,16*g+cols]=0
        active=np.flatnonzero(d[g].any(0))
        native_col=int(active[np.argmin(np.square(wg[:,active].astype(float)).sum(0))])
        wn[:,16*g+native_col]=0
        pairs.append({'g':g,'k':k,'l':l,'delta':delta.tolist(),
                      'delta_support':cols.tolist(),'original_projection_squared_cost':cost,
                      'ordinary_one_column':native_col})
    weights={'original':w,'projected_independent_RNE':np.clip(np.rint(wf),-127,127).astype(np.int64),
             'integer_pair_projection':wi,'W_one_column_equal_rank':wn,'W_same_pair_support':ws}
    # Requested executable artifact: the same fixed integer candidate, no refit.
    np.save(HERE/'response_class_W.npy',wi.astype(np.int8))
    tables={k:table(d,v) for k,v in weights.items()}
    free=tables['original'].copy()
    for p in pairs:
        g,k,l=p['g'],p['k'],p['l']
        avg=np.rint((free[g,k].astype(float)+free[g,l])/2).astype(np.int64)
        free[g,k]=avg;free[g,l]=avg
        assert np.array_equal(tables['integer_pair_projection'][g,k],tables['integer_pair_projection'][g,l])
    tables['free_L_pair_mean']=free
    variants={};classmaps={}
    for label,values in tables.items():
        classmaps[label]={width:classes(values,width) for width in WIDTHS}
        mismatches=[int(np.count_nonzero(values[p['g'],p['k']]-values[p['g'],p['l']])) for p in pairs]
        m={'L_range':[int(values.min()),int(values.max())],
           'INT10_fit':bool(values.min()>=-512 and values.max()<=511),
           'target_pair_different_scalars_per_g':mismatches,
           'classes':{str(width):class_summary(classmaps[label][width],16) for width in WIDTHS},
           'H12_words_no_dedup':0,'words_content_dedup':{str(width):0 for width in WIDTHS},
           'H12_vector_updates':0,'gate_mismatches':0,'local_scalar_outputs':0,
           'Y_error_square':0.,'Y_reference_square':0.,'Y_max_error':0,
           'U_error_square':0.,'U_reference_square':0.,'U_max_error':0}
        if label in weights:
            dw=weights[label]-w
            m.update(W_relative_fro=float(np.linalg.norm(dw)/np.linalg.norm(w)),
                     W_max_abs_change=int(np.abs(dw).max()),W_changed_scalars=int(np.count_nonzero(dw)),
                     W_zero_scalars=int(np.count_nonzero(weights[label]==0)))
        variants[label]=m
    result={'scope':'Exact classes in real D/W; one approximate level: one nonzero support-code pair per g, tied over all H384.',
            'inputs':'old forced-code post-projection source captures; raw source gate/partial-decision histories unavailable',
            'exact_original_alphabets':exact,'fixed_approximate_pairs':pairs,'variants':variants,
            'class_members_H384':{k:classmaps[k][384] for k in classmaps},
            'frames':[],'local_tile_indices':[0,199,399,599],
            'consumer_rank':int(np.linalg.matrix_rank(a)),
            'state_accounting':{'canonical_code_map_6x16x4_bits_bytes':48,'canonical_map_128bit_cold_words':3,
                               'real_tile_presence_bits':6*16,'destination_mask_bits_per_code':320,
                               'comment':'48 B mapping only; does not include 320-bit destination-mask reads/OR, live entries, holding, class proof, or their service.'},
            'limitations':['All class payload reductions are request opportunities, not RTL cycles.',
                           'Ordinary content dedup has exactly the same class maps and coalescing rights.',
                           'H12 classes do not finish a source group while any requested H word still distinguishes the codes.',
                           'Even H384 equivalence does not prove raw-gate prefix sufficiency or earlier PSN completion.',
                           'No training/AEE; local gates compare the fixed original integer sn2 threshold.',
                           'Ordinary W rank-matched control matches six deleted input columns; support-matched control deletes all differing columns.',
                           'Pair selection minimizes original unconstrained Euclidean projection score; subsequent INT8 projection is box constrained. No validation selection.']}
    packed_d=(d*(1<<np.arange(16))).sum(-1)
    lookup=np.full((6,65536),-1,dtype=np.int16)
    for g in range(6):lookup[g,packed_d[g]]=np.arange(16)
    files=json.loads((CAP/'run.json').read_text())['validation_files'][:10]
    old=json.loads((BASE/'bn_state/support_table_transport_probe.json').read_text())
    co_presence=np.zeros(6,dtype=np.int64)
    for file,oldf in zip(files,old['frames']):
        with np.load(CAP/('forced_code_'+Path(file).stem+'_source.npz')) as z:
            words=np.ascontiguousarray(z['gate_bits']).view('<u2').reshape(10,19200,6)
        ids=np.stack([lookup[g,words[:,:,g]] for g in range(6)],axis=-1)
        assert (ids>=0).all()
        tileids=ids.reshape(10,600,32,6)
        seen=np.zeros((6,600,16),dtype=bool)
        for g in range(6):
            for k in range(16):seen[g,:,k]=(tileids[:,:,:,g]==k).any(axis=(0,2))
        baseline=int(seen[:,:,1:].sum())*32
        assert baseline*16==oldf['formats']['fixed10']['payload_128bit_read_bytes']
        for p in pairs:co_presence[p['g']]+=np.count_nonzero(seen[p['g'],:,p['k']]&seen[p['g'],:,p['l']])
        selected=tileids[:,result['local_tile_indices'],:,:].reshape(10,-1,6)
        ys={label:sum(v[g,selected[:,:,g],:] for g in range(6)) for label,v in tables.items()}
        native=np.concatenate([d[g,selected[:,:,g]] for g in range(6)],axis=-1)
        assert np.array_equal(native@w.T,ys['original'])
        us={label:np.einsum('ts,sph->tph',a,y) for label,y in ys.items()}
        ref_gate=gate(us['original'],params)
        frame={'file':file,'original_words':baseline,'variants':{}}
        for label,v in tables.items():
            m=variants[label]
            # No dedup still skips any exact zero word, same ordinary permission.
            live=(v.reshape(6,16,32,12)!=0).any(-1)
            no_dedup=int((seen.sum(1)[:,:,None]*live).sum())
            m['H12_words_no_dedup']+=no_dedup
            wr={}
            for width in WIDTHS:
                n,_=requests(classmaps[label][width],seen,width)
                m['words_content_dedup'][str(width)]+=n;wr[str(width)]=n
            # Nonzero-code destinations are disjoint, so class OR does not
            # eliminate any per-token accumulator update absent a zero response.
            for g in range(6):
                freq=np.bincount(ids[:,:,g].ravel(),minlength=16)
                m['H12_vector_updates']+=int((freq[:,None]*live[g]).sum())
            dy=ys[label]-ys['original'];du=us[label]-us['original']
            m['Y_error_square']+=float(np.square(dy.astype(float)).sum())
            m['Y_reference_square']+=float(np.square(ys['original'].astype(float)).sum())
            m['U_error_square']+=float(np.square(du.astype(float)).sum())
            m['U_reference_square']+=float(np.square(us['original'].astype(float)).sum())
            m['Y_max_error']=max(m['Y_max_error'],int(np.abs(dy).max()))
            m['U_max_error']=max(m['U_max_error'],int(np.abs(du).max()))
            mismatch=int(np.count_nonzero(gate(us[label],params)!=ref_gate))
            m['gate_mismatches']+=mismatch;m['local_scalar_outputs']+=int(dy.size)
            frame['variants'][label]={'words_no_dedup':no_dedup,'dedup_words':wr,'gate_mismatches':mismatch}
        result['frames'].append(frame)
    result['fixed_pairs_tile_copresence_per_g']=co_presence.tolist()
    reference=variants['original']['words_content_dedup']['12']
    for label,m in variants.items():
        m['Y_relative_rmse']=float(np.sqrt(m.pop('Y_error_square')/m.pop('Y_reference_square')))
        m['U_relative_rmse']=float(np.sqrt(m.pop('U_error_square')/m.pop('U_reference_square')))
        m['gate_flip_fraction']=m['gate_mismatches']/m['local_scalar_outputs']
        m['request_reduction_fraction']=1-m['words_content_dedup']['12']/reference
        m['payload_MB_per_frame']=m['words_content_dedup']['12']*16/10/1e6
        m['independent_payload_words_saved_beyond_equal_rights_content_dedup']=0
        m['producer_partial_gate_service_saving']=None
    result['exact_original_request_words']=reference
    result['exact_original_independent_saving_beyond_content_dedup']=0
    (HERE/'probe_response_classes.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps({'pairs':pairs,'copresence':co_presence.tolist(),'variants':{
        k:{n:v[n] for n in ('target_pair_different_scalars_per_g','request_reduction_fraction','payload_MB_per_frame',
                            'Y_relative_rmse','gate_flip_fraction','W_relative_fro') if n in v}
        for k,v in variants.items()}},ensure_ascii=False))


if __name__=='__main__':main()
