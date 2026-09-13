"""Independent chained-convolution verification and actual sampled AAC ledger."""
from pathlib import Path
import json
import numpy as np
from prototype import HERE,load_capture,reconstruct,error

def conv(x,w):
    kh,kw=w.shape[-2:]
    padded=np.pad(x,((0,0),(0,0),(kh//2,kh//2),(kw//2,kw//2)))
    patches=np.lib.stride_tricks.sliding_window_view(padded,(kh,kw),axis=(-2,-1))
    return np.einsum('ncyxhw,ochw->noyx',patches,w,optimize=True)

def main():
    w,theta,bias,frames=load_capture()
    p=HERE/'results';report=json.loads((p/'summary.json').read_text())
    g=np.concatenate([f[1]!=0 for f in frames]);active=g.mean(0)
    # Per input channel/tap activity is sampled from the actual theta*g words.
    rate_ch=active.reshape(96,3,3).mean((1,2))
    rate_ch_h=active.reshape(96,3,3).mean(2)
    rng=np.random.default_rng(1313);x=(rng.random((2,96,7,9))<.04)*theta
    checks=[]
    report['rows']=[r for r in report['rows'] if not r['name'].endswith('of4_w8')]
    for row in report['rows']:
        with np.load(p/(row['name']+'.npz')) as z:
            f={k:z[k].astype(np.float64) for k in ['first','core','second','residual','weight'] if k in z}
        ww=f['weight'] if 'weight' in f else reconstruct(f)
        vals=[dict(file=n,**error(xx@ww.reshape(96,-1).T+bias,y)) for n,xx,y in frames]
        row.update(frames=vals,holdout2_relative_l2_mean=float(np.mean([r['relative_l2'] for r in vals[2:]])))
        bits=row['coefficient_bits']
        if 'spatial_streaming_line_buffer_values_per_row' in row['cost']:
            row['cost']['extra_horizontal_continuous_shift_values']=row['cost'].pop('spatial_streaming_line_buffer_values_per_row')
        if 'weight' in f:
            aac=float(active@np.count_nonzero(f['weight'],axis=0).reshape(-1))
            n=int(row['name'].split('_')[1][0]);idx_bits=4 if n==2 else 2
            row['coefficient_storage_bytes']=w.size*n/4*bits/8+w.size/4*idx_bits/8+96*4
            row['actual_sample_AAC_per_position']=aac;continue
        first=f['first'];second=f['second'];spatial=second.shape[-1]>1
        if 'core' in f:
            aac=float(rate_ch@np.count_nonzero(first[:,:,0,0],axis=0))
            qualifier='Input 1x1 activity inferred by averaging 3x3 captured taps; full streaming reuse assumed.'
        elif spatial:
            aac=float(np.sum(rate_ch_h*np.count_nonzero(first[:,:,:,0],axis=0)))
            qualifier='First vertical-stage activity averaged across captured horizontal taps; full streaming reuse assumed.'
        else:
            aac=float(active@np.count_nonzero(first,axis=0).reshape(-1))
            qualifier='Exact mean first-stage active coefficient terms for sampled output positions.'
        if 'residual' in f:aac+=float(active@np.count_nonzero(f['residual'],axis=0).reshape(-1))
        row['actual_sample_AAC_per_position']=aac
        row['coefficient_storage_bytes']=sum(v.size for k,v in f.items() if k!='residual')*bits/8
        if 'residual' in f:row['coefficient_storage_bytes']+=w.size*bits/16+w.size/4*4/8
        row['coefficient_storage_bytes']+=96*4
        if bits==8:row['coefficient_storage_bytes']+=sum(v.shape[0] for v in f.values())*4
        row['AAC_scope']=qualifier
        row['MAC_to_AAC_unit_ratios']={str(r):aac+r*row['cost']['continuous_MAC_per_position']+row['cost']['residual_merge_adds_per_position'] for r in [1,2,4,8]}
        if row['coefficient_bits']==32:
            y=conv(x,first)
            if 'core' in f:y=conv(y,f['core'])
            y=conv(y,second)
            if 'residual' in f:y=y+conv(x,f['residual'])
            ref=conv(x,reconstruct(f))
            maximum=float(np.max(np.abs(y-ref)))
            if maximum>1e-10:raise AssertionError((row['name'],maximum))
            checks.append(dict(name=row['name'],max_abs=maximum))
    # W8 pure structure gets the same coefficient permission as all factors.
    for row in list(report['rows']):
        if not row['name'].startswith('structure_'):continue
        with np.load(p/(row['name']+'.npz')) as z:ww=z['weight'].astype(np.float64)
        scale=np.max(np.abs(ww),axis=(1,2,3),keepdims=True)/127
        qw=np.rint(ww/scale)*scale
        vals=[dict(file=n,**error(xx@qw.reshape(96,-1).T+bias,y)) for n,xx,y in frames]
        new=dict(row);new.update(name=row['name']+'_w8',coefficient_bits=8,frames=vals,
            weight_relative_l2=float(np.linalg.norm(qw-w)/np.linalg.norm(w)),holdout2_relative_l2_mean=float(np.mean([r['relative_l2'] for r in vals[2:]])),
            actual_sample_AAC_per_position=float(active@np.count_nonzero(qw,axis=0).reshape(-1)))
        n=int(row['name'].split('_')[1][0]);idx_bits=4 if n==2 else 2
        new['coefficient_storage_bytes']=w.size*n/4+w.size/4*idx_bits/8+96*8
        report['rows'].append(new)
        np.savez_compressed(p/(new['name']+'.npz'),weight=qw.astype(np.float32),bias=bias.astype(np.float32),coefficient_bits=np.array(8))
    dense_aac=float(np.sum(active)*96)
    comparisons=[]
    for bits in [32,8]:
        for limit in [.2,.12,.10,.08,.06,.04]:
            rr=[r for r in report['rows'] if r['coefficient_bits']==bits and r['holdout2_relative_l2_mean']<=limit]
            def best(kind):
                group=[r for r in rr if kind(r['name'])]
                if not group:return None
                r=min(group,key=lambda r:r['actual_sample_AAC_per_position']+r['cost']['continuous_MAC_per_position']+r['cost'].get('residual_merge_adds_per_position',0))
                return dict(name=r['name'],error=r['holdout2_relative_l2_mean'],AAC=r['actual_sample_AAC_per_position'],MAC=r['cost']['continuous_MAC_per_position'],
                    merge_adds=r['cost'].get('residual_merge_adds_per_position',0))
            comparisons.append(dict(bits=bits,max_local_relative_l2=limit,ordinary_svd=best(lambda n:n.startswith(('flat_svd','activation_svd'))),
                structured_zero=best(lambda n:n.startswith('structure_')),ordinary_hybrid=best(lambda n:n.startswith('flat_nm')),
                new_spatial_hybrid=best(lambda n:n.startswith('spatial_nm')),spatial=best(lambda n:n.startswith('spatial_r')),tucker=best(lambda n:n.startswith('tucker'))))
    report.update(dense_sample_AAC_per_position=dense_aac,finite_padding_checks=checks,same_local_error_comparisons=comparisons,
        coefficient_storage_scope='Dense first/core/second tensors; compressed N:M residual plus 4 bits per 2:4 group, 2 bits per 1:4/3:4 group; bias and W8 FP32 row scales included. Alignment, addresses, memory banking and burst waste excluded.',
        cost_boundary='AAC and continuous MAC reported separately. Ratios are sensitivity assumptions, never hardware cycles. Exact sampled direct AAC; factor streaming AAC projections exclude prologue/epilogue, padding savings, metadata and ports.')
    (p/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('CHECKED',len(checks),'dense AAC/position',dense_aac)
    for r in comparisons:
        if r['bits']==32:print(json.dumps(r))

if __name__=='__main__':main()
