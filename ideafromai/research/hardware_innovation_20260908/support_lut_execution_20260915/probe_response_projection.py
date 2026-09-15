"""Bounded real-D/W projection probe. No training, RTL, new captures or AEE.

One fixed target: dictionary code 1 in each K16 group, all H12 words.
The dictionary's ordering was fixed by the old train32 run. No validation fit.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE/'bn_state'))
from support_service_model import read_torch
CAP = BASE/'algorithm/support_training'


def bounded_zero_sum(w):
    """Euclidean box/hyperplane projection, then exact balanced integer round.

    Bisection solves one convex multiplier, not a model/threshold sweep.
    Integer floor plus lowest marginal-cost increments solves sum(q)=0.
    """
    w = np.asarray(w, dtype=np.float64)
    lo, hi = float(w.min()-127), float(w.max()+127)
    for _ in range(80):
        mid = (lo+hi)/2
        if np.clip(w-mid, -127, 127).sum() > 0:
            lo = mid
        else:
            hi = mid
    projected = np.clip(w-(lo+hi)/2, -127, 127)
    integer = np.floor(projected+1e-10).astype(np.int64)
    remaining = -int(integer.sum())
    assert 0 <= remaining <= len(w)
    cost = (integer+1-w)**2-(integer-w)**2
    cost[integer >= 127] = np.inf
    integer[np.argsort(cost, kind='stable')[:remaining]] += 1
    assert integer.sum() == 0 and integer.min() >= -127 and integer.max() <= 127
    return projected, integer


def table(d, w):
    return np.einsum('gkc,hgc->gkh', d, w.reshape(384, 6, 16))


def gate(u, params):
    tau = params['threshold_int64'][:, None, :]
    result = np.where(params['positive_gain'][None, None, :], u >= tau, u <= tau)
    return np.where(params['constant_channels'][None, None, :],
                    params['constant_gate'][:, None, :], result)


def main():
    d = np.load(CAP/'dictionary.npy').astype(np.int64)
    w = read_torch(CAP/'forced_code_weight_int8.pt').astype(np.int64)
    student = read_torch(CAP/'forced_code_parameters.pt')
    pp = read_torch(BASE/'algorithm/integer_s0_valid825/integer_parameters.pt')
    params = pp['sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.']
    a = params['temporal_int16'].astype(np.int64)
    assert d.shape == (6, 16, 16) and w.shape == (384, 96)
    scale_relative_difference=float(np.max(np.abs(student['scale']/params['weight_row_scale']-1)))
    assert scale_relative_difference < 1e-14
    target = np.zeros((6, 16, 32), dtype=bool)
    target[:, 1, :] = True
    target_columns = [np.flatnonzero(d[g, 1]) for g in range(6)]
    wf, wi, wp, ws = w.astype(float).copy(), w.copy(), w.copy(), w.copy()
    wm=w.copy()
    ordinary_selected_columns = []
    ordinary_response_matched_codes=[]
    for g in range(6):
        active = np.flatnonzero(d[g].any(0))
        cols = 16*g + target_columns[g]
        for h in range(384):
            wf[h, cols], wi[h, cols] = bounded_zero_sum(w[h, cols])
        ws[:, cols] = 0
        # Same 12 scalar constraints per H12 block, minimum original W energy.
        chosen = []
        for h0 in range(0, 384, 12):
            energy = (w[h0:h0+12, 16*g+active].astype(float)**2).sum(0)
            col = int(active[np.argmin(energy)])
            wp[h0:h0+12, 16*g+col] = 0
            chosen.append(col)
        ordinary_selected_columns.append(chosen)
        # Strong ordinary control: zero W groups that remove exactly one of
        # the 15 nonzero response rows, minimum W energy per H12 block.
        candidates=[]
        for k in range(1,16):
            contained=np.all(d[g,1:]<=d[g,k],axis=1)
            if contained.sum()==1:candidates.append(k)
        assert candidates
        selected=[]
        for h0 in range(0,384,12):
            energy=[float(np.square(w[h0:h0+12,16*g+np.flatnonzero(d[g,k])].astype(float)).sum())
                    for k in candidates]
            k=candidates[int(np.argmin(energy))]
            wm[h0:h0+12,16*g+np.flatnonzero(d[g,k])]=0
            selected.append(k)
        ordinary_response_matched_codes.append(selected)
    wr = np.clip(np.rint(wf), -127, 127).astype(np.int64)
    weights = dict(original=w, euclidean_rne=wr, integer_balanced=wi,
                   W_group_equal_rank=wp, W_group_same_support=ws,
                   W_group_equal_response_sparsity=wm)
    tables = {k:table(d,v) for k,v in weights.items()}
    free = tables['original'].copy()
    free[:, 1, :] = 0
    tables['free_L_same_targets'] = free
    original = tables['original']
    assert np.max(np.abs(table(d,wf)[:, 1])) < 1e-8
    assert (tables['integer_balanced'][:, 1] == 0).all()
    # Re-exporting the integer lattice on the original dyadic grid is exact.
    roundtrip = np.rint((wi*student['scale'][:,None]/float(student['theta']))
                       * float(student['theta'])/student['scale'][:,None]).astype(np.int64)
    assert np.array_equal(roundtrip, wi)
    valid = np.ones((6,16), dtype=bool); valid[:,0] = False
    zero_words = {k:(v.reshape(6,16,32,12)==0).all(-1) for k,v in tables.items()}
    assert int(zero_words['W_group_equal_response_sparsity'][valid].sum())==int(target.sum())
    representability = []
    for g in range(6):
        recovered = d[g] @ (np.linalg.pinv(d[g].astype(float)) @ free[g])
        representability.append(float(np.linalg.norm(recovered-free[g])))
    result = {
        'scope':'fixed code1, one linear constraint per output and K16 group; no parameter sweep',
        'source_scope':'old forced_code S0 block0; post-nearest-code captured gate bits, not raw source producer',
        'shape':{'D':list(d.shape),'W':list(w.shape),'physical_word':'H12 signed INT10, 120 payload bits + 8 pad bits; 8 words/H96'},
        'target_words':int(target.sum()),'nonzero_code_words':int(valid.sum()*32),
        'scale_storage_relative_difference':scale_relative_difference,
        'D_rank':[int(np.linalg.matrix_rank(x)) for x in d],
        'PSN_A_rank':int(np.linalg.matrix_rank(a)),
        'target_code':1,'target_code_popcounts':[len(x) for x in target_columns],
        'linear_constraints':2304,'W_scalar_parameters':int(w.size),
        'independent_RNE_target_nonzero_scalars':int(np.count_nonzero(tables['euclidean_rne'][:,1])),
        'independent_RNE_target_zero_words':int(zero_words['euclidean_rne'][:,1].sum()),
        'integer_balanced_target_zero_words':int(zero_words['integer_balanced'][:,1].sum()),
        'float_projection_target_max_abs':float(np.abs(table(d,wf)[:,1]).max()),
        'free_L_nonrepresentable_residual_fro_per_g':representability,
        'W_group_equal_rank_selected_columns_per_g_H12':ordinary_selected_columns,
        'W_group_equal_response_sparsity_selected_codes_per_g_H12':ordinary_response_matched_codes,
        'variants':{},'frames':[],
        'metadata':{'dense_L_INT10_bytes':90*32*16,
                    'nonzero_code_word_valid_bits':90*32,
                    'all_code_word_valid_bits':6*16*32,
                    'static_valid_bitmap_bytes':360,
                    'cold_bitmap_128bit_words':23,
                    'uncached_H96_bitmap_read_words_per_row':1,
                    'note':'Generic dense-address zero-word bitmap; packed addresses/popcount, request scheduling and decoder are not implemented.'},
        'exclusions':['All request reductions are payload opportunities, not measured cycles/energy.',
                      'Local raw-Y/U/gate errors are against the old forced student, not AEE.',
                      'The ordinary equal-rank arm matches number of scalar constraints, not response sparsity or W modifications.',
                      'Ordinary same-support pruning guarantees the same target zeros but may create additional response zeros.',
                      'Free L is a separate nonlinear code-LUT function, not necessarily D times one W.',
                      'No recovery training; this result cannot reject a training family.']}
    for k,v in tables.items():
        metrics = {'response_zero_words':int(zero_words[k][valid].sum()),
                   'response_zero_fraction':float(zero_words[k][valid].mean()),
                   'L_min':int(v.min()),'L_max':int(v.max()),
                   'INT10_fit':bool(v.min()>=-512 and v.max()<=511),
                   'target_zero_words':int(zero_words[k][:,1].sum()),
                   'requested_128bit_words':0,'skipped_payload_words':0,
                   'Y_squared_error':0,'Y_squared_reference':0,'Y_abs_error':0,'Y_max_error':0,
                   'U_squared_error':0,'U_squared_reference':0,'U_max_error':0,
                   'gate_mismatches':0,'local_scalar_outputs':0}
        if k in weights:
            delta=weights[k]-w
            metrics.update(W_relative_fro=float(np.linalg.norm(delta)/np.linalg.norm(w)),
                           W_max_abs_change=int(np.abs(delta).max()),
                           W_changed_scalars=int(np.count_nonzero(delta)),
                           W_zero_scalars=int((weights[k]==0).sum()))
        result['variants'][k]=metrics
    dictionary_words=(d*(1<<np.arange(16))).sum(-1)
    lookup=np.full((6,65536),-1,dtype=np.int16)
    for g in range(6):lookup[g,dictionary_words[g]]=np.arange(16)
    files=json.loads((CAP/'run.json').read_text())['validation_files'][:10]
    previous=json.loads((BASE/'bn_state/support_table_transport_probe.json').read_text())
    result['local_tile_indices']=[0,199,399,599]
    presence_sum=np.zeros((6,16),dtype=np.int64)
    token_sum=np.zeros((6,16),dtype=np.int64)
    for file,old in zip(files,previous['frames']):
        with np.load(CAP/('forced_code_'+Path(file).stem+'_source.npz')) as z:
            words=np.ascontiguousarray(z['gate_bits']).view('<u2').reshape(10,19200,6)
        ids=np.stack([lookup[g,words[:,:,g]] for g in range(6)],axis=-1)
        assert (ids>=0).all()
        tiles=ids.reshape(10,600,32,6)
        presence=np.zeros((6,16),dtype=np.int64)
        for g in range(6):
            token_sum[g]+=np.bincount(ids[:,:,g].ravel(),minlength=16)
            for c in range(16):presence[g,c]=(tiles[:,:,:,g]==c).any(axis=(0,2)).sum()
        presence_sum+=presence
        baseline_words=int(presence[valid].sum())*32
        assert baseline_words*16==old['formats']['fixed10']['payload_128bit_read_bytes']
        frame={'file':file,'baseline_H12_words':baseline_words,'variants':{}}
        selected=tiles[:,result['local_tile_indices'],:,:].reshape(10,-1,6)
        ys={k:sum(v[g,selected[:,:,g],:] for g in range(6)) for k,v in tables.items()}
        # Table arithmetic independently matches the original native binary dot.
        native=np.concatenate([d[g,selected[:,:,g],:] for g in range(6)],axis=-1)
        assert np.array_equal(native@w.T,ys['original'])
        us={k:np.einsum('ts,sph->tph',a,y) for k,y in ys.items()}
        reference_gate=gate(us['original'],params)
        for k in tables:
            zeros=zero_words[k].copy();zeros[:,0]=False
            saved=int((presence[:,:,None]*zeros).sum())
            metric=result['variants'][k]
            metric['requested_128bit_words']+=baseline_words-saved
            metric['skipped_payload_words']+=saved
            dy=ys[k]-ys['original'];du=us[k]-us['original']
            # Float64 squared sums avoid int64 overflow; all dot products above remain exact int64.
            metric['Y_squared_error']+=float(np.square(dy.astype(float)).sum())
            metric['Y_squared_reference']+=float(np.square(ys['original'].astype(float)).sum())
            metric['Y_abs_error']+=int(np.abs(dy).sum())
            metric['Y_max_error']=max(metric['Y_max_error'],int(np.abs(dy).max()))
            metric['U_squared_error']+=float(np.square(du.astype(float)).sum())
            metric['U_squared_reference']+=float(np.square(us['original'].astype(float)).sum())
            metric['U_max_error']=max(metric['U_max_error'],int(np.abs(du).max()))
            mismatch=int(np.count_nonzero(gate(us[k],params)!=reference_gate))
            metric['gate_mismatches']+=mismatch
            metric['local_scalar_outputs']+=int(dy.size)
            frame['variants'][k]={'payload_words':baseline_words-saved,'gate_mismatches':mismatch}
        result['frames'].append(frame)
    result['tile_presence_per_g_code']=presence_sum.tolist()
    result['token_frequency_per_g_code']=token_sum.tolist()
    baseline=result['variants']['original']['requested_128bit_words']
    row_requests=int(presence_sum[valid].sum())*4
    result['H96_row_requests_10frames']=row_requests
    for k,m in result['variants'].items():
        m['payload_read_bytes_per_frame']=m['requested_128bit_words']*16/len(files)
        m['payload_reduction_fraction']=m['skipped_payload_words']/baseline
        m['payload_plus_uncached_bitmap_bytes_per_frame']=(m['requested_128bit_words']+row_requests)*16/len(files)
        m['Y_relative_rmse']=float(np.sqrt(m.pop('Y_squared_error')/m.pop('Y_squared_reference')))
        m['U_relative_rmse']=float(np.sqrt(m.pop('U_squared_error')/m.pop('U_squared_reference')))
        m['Y_mae']=m.pop('Y_abs_error')/m['local_scalar_outputs']
        m['gate_flip_fraction']=m['gate_mismatches']/m['local_scalar_outputs']
    (HERE/'probe_response_projection.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps({k:result[k] for k in ('target_code_popcounts','independent_RNE_target_nonzero_scalars',
          'independent_RNE_target_zero_words','integer_balanced_target_zero_words','variants')},ensure_ascii=False))


if __name__=='__main__':main()
