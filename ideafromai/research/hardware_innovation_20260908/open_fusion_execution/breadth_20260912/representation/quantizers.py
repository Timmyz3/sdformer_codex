"""Train-only fixed quantizer family on true T10 I24/projection-g pairs."""
from pathlib import Path
import json
import numpy as np

HERE=Path(__file__).resolve().parent
MODES=('fixed_q8','center_q8','affine_q8','diagonal_g_q8','full_g_q8')
STEP=2048


def predict(x,g,D,c):
    return (D@g.reshape(10,-1)+c[:,None]).reshape(x.shape)


def reconstruct(x,g,parameters):
    D,c,step=(parameters[k].astype(np.int64) for k in ('D','c','step'))
    b=predict(x,g,D,c)
    shape=(10,)+(1,)*(x.ndim-1)
    ss=step.reshape(shape)
    q,rem=np.divmod(x-b,ss)
    q+=((2*rem>ss)|((2*rem==ss)&((q&1)!=0)))
    code=np.clip(q,-128,127)
    raw=b+code*ss
    estimate=np.clip(raw,-(1<<23),(1<<23)-1)
    return estimate,code,dict(quant_clipped=int(np.count_nonzero((q< -128)|(q>127))),
        reconstructed_sat24=int(np.count_nonzero(raw!=estimate)))


def signed_bits(x):
    a=np.asarray(x,dtype=np.int64)
    return max(1,int(max(int(a.max(initial=0)),int(-a.min(initial=0)-1))).bit_length()+1)


def fit(x,g):
    """All modes get the same samples. No validation-dependent selection."""
    x=x.reshape(10,-1).astype(np.float64)
    g=g.reshape(10,-1).astype(np.float64)
    mean=x.mean(1)
    designs={}
    def pack(mode,D,c,step):
        designs[mode]=dict(D=np.rint(D).astype(np.int64),c=np.rint(c).astype(np.int64),
            step=np.asarray(step,dtype=np.int64))
    zeros=np.zeros((10,10));same=np.full(10,STEP,dtype=np.int64)
    pack('fixed_q8',zeros,np.zeros(10),same)
    pack('center_q8',zeros,mean,same)
    lo,hi=x.min(1),x.max(1)
    exponent=np.maximum(0,np.ceil(np.log2(np.maximum((hi-lo)/255,1)))).astype(np.int64)
    step=np.left_shift(np.ones(10,dtype=np.int64),exponent)
    pack('affine_q8',zeros,(lo+hi+step)/2,step)
    dg=np.zeros(10);c=mean.copy()
    for t in range(10):
        active=g[t].astype(bool)
        if active.any() and (~active).any():
            c[t]=x[t,~active].mean();dg[t]=x[t,active].mean()-c[t]
    pack('diagonal_g_q8',np.diag(dg),c,same)
    augmented=np.concatenate((g,np.ones((1,g.shape[1]))),axis=0)
    gram=augmented@augmented.T/augmented.shape[1]
    cross=x@augmented.T/augmented.shape[1]
    coefficients=cross@np.linalg.pinv(gram,rcond=1e-12)
    pack('full_g_q8',coefficients[:,:10],coefficients[:,10],same)
    metadata=dict(elements=int(x.size),positions=int(x.shape[1]//96),
        full_regression_rank=int(np.linalg.matrix_rank(gram)),
        full_regression='Minimum-norm least squares with intercept; Gram rcond1e-12, one fit.',
        affine='Per-time training min/max dyadic range with signed8 asymmetric-center correction; no validation tuning.',
        x_min=lo.tolist(),x_max=hi.tolist(),g_density=g.mean(1).tolist())
    return designs,metadata


def parameter_cost(parameters):
    D,c,step=(parameters[k] for k in ('D','c','step'))
    nonzero=np.count_nonzero(D,axis=1)
    # A future circuit must decide packed or expanded constants. Both are
    # reported; this is not an SRAM port execution model.
    bitsD=signed_bits(D);bitsC=signed_bits(c)
    return dict(D_nonzero=int(nonzero.sum()),D_signed_bits=bitsD,c_signed_bits=bitsC,
        expanded_D_c_bytes=int((D.size+c.size)*4),
        compact_nonzero_D_bits=int(nonzero.sum())*(bitsD+4+4),
        c_bits=int(np.count_nonzero(c))*bitsC,
        step_exponents=np.log2(step).astype(int).tolist(),
        predicted_additions_per_C_P=int(np.maximum(nonzero-1,0).sum()+np.count_nonzero(c)),
        gate_support_terms_per_C_P=int(nonzero.sum()),
        scope='Metadata/storage and scalar work only. Gate availability, dynamic zero skips, packing, decoder state and issue/ports unmeasured.')


def main():
    package=HERE/'parameters';package.mkdir(exist_ok=True)
    report=dict(training=False,calibration_split='train',modes=list(MODES),axes={})
    for axis in ('ordinary','lifting_raw'):
        source=HERE/'train_captures'/(axis+'.npz')
        with np.load(source) as z:
            x=z['updated_I24'];g=z['projection_g']
        designs,meta=fit(x,g)
        np.savez_compressed(package/(axis+'.npz'),
            **{m+'_'+k:v for m,d in designs.items() for k,v in d.items()})
        rows={}
        for mode,d in designs.items():
            estimate,code,counts=reconstruct(x.astype(np.int64),g.astype(np.int64),d)
            counts.update(parameter_cost(d));counts.update(source_MAE=float(np.mean(np.abs(estimate-x))),
                source_RMSE=float(np.sqrt(np.mean((estimate-x).astype(float)**2))),
                scalar_zero_fraction=float(np.mean(code==0)))
            rows[mode]=counts
        report['axes'][axis]=dict(calibration=meta,source=str(source.relative_to(HERE)),models=rows)
    (HERE/'fit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({a:{m:dict(RMSE=r['source_RMSE'],clipped=r['quant_clipped']) for m,r in q['models'].items()} for a,q in report['axes'].items()}))


if __name__=='__main__':main()
