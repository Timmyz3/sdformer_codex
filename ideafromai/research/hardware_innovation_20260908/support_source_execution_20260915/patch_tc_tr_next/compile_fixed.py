"""One fixed deployment; NumPy/Python 3.12, no fit, network run, or cycles."""
from pathlib import Path
from fractions import Fraction
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
FACTOR = BASE / 'algorithm/patch_probe/factor_completion_20260909'
sys.path.insert(0, str(FACTOR / 'latent_stage_train16'))
from compile_integer_factors import domains, ceil_fraction, fraction, power2, signed_width
sys.path.insert(0, str(FACTOR))
from factor_reference import spatial_regions, sparse_forward


def load(path):
    with np.load(path) as d:
        return {k: d[k].copy() for k in d.files}


def compile_one(name, original):
    if name == 'ordinary_r32':
        live = np.flatnonzero(original['v_nonzero'].any(1))
        assert live.tolist() == list(range(32))
        u = original['u_int8'][:, live].astype(np.int64)
        ue = original['u_scale_exponent'][live].astype(np.int64)
        vs = original['v_sign'][live].astype(np.int64)
        ve = original['v_shift'][live].astype(np.int64)
        vn = original['v_nonzero'][live].astype(bool)
        masks = np.ones((8, 8), bool)
        fp_u, fp_v = original['u'][:, live], original['v'][live]
    else:
        fp_u, fp_v = original['u'], original['v']
        folded = fp_u.astype(np.float64) * float(original['theta_source'])
        maximum = abs(folded).max(0)
        ue = np.where(maximum > 0, np.ceil(np.log2(np.maximum(maximum/127, 2.**-40))), 0).astype(np.int64)
        u = np.rint(folded/np.exp2(ue)).clip(-127, 127).astype(np.int64)
        vn = original['connectivity'].astype(bool) & (fp_v != 0)
        vs = np.sign(fp_v).astype(np.int64)
        ve = np.rint(np.log2(np.clip(abs(fp_v.astype(np.float64)), 2.**-15, 1))).astype(np.int64)
        masks = original['masks'].astype(bool)
    ex = ue[:, None] + ve
    eh = np.where(vn, ex, 1000).min(0)
    assert vn.any(0).all()
    shift = np.where(vn, ex-eh, 0)
    v = vs * (np.ones_like(vs) << shift) * vn
    a = np.rint(original['a'].astype(np.float64) * 16384).astype(np.int64)
    gain, bias = original['bn_scale'], original['bn_bias']
    assert (gain > 0).all() and abs(a).max() < 32768
    offset = [[fraction(bias[h])*Fraction(int(a[t].sum()),16384)
        + fraction(original['temporal_bias'][t])-fraction(original['theta_output'])
        for h in range(96)] for t in range(10)]
    kappa = [fraction(gain[h])*power2(int(eh[h])-14) for h in range(96)]
    tau = np.array([[ceil_fraction(-offset[t][h]/kappa[h]) for h in range(96)] for t in range(10)], np.int64)
    rank_mask = masks.repeat(4, axis=1)
    bounds = []
    for row in rank_mask:
        d, _ = domains(u[:, row], v[row], a, int(row.sum()))
        assert d['INT48_U_any_reduction_order']
        bounds.append(d)
    arrays = dict(u=u.astype(np.int8), v=v, v_sign=vs.astype(np.int8),
        v_nonzero=vn, v_shift=ve.astype(np.int8), aligned_v_shift=shift.astype(np.uint8),
        a=a.astype(np.int16), tau=tau, masks=masks, y_exponent=eh.astype(np.int16),
        u_exponent=ue.astype(np.int16), numeric_scope=np.array('new exact integer function; not inherited AEE'),
        fp_u=fp_u, fp_v=fp_v, fp_a=original['a'], bn_scale=gain, bn_bias=bias,
        temporal_bias=original['temporal_bias'], theta_source=original['theta_source'],
        theta_output=original['theta_output'])
    np.savez(HERE / (name+'.npz'), **arrays)
    report = dict(rank=int(u.shape[1]), active_rank_per_region=rank_mask.sum(1).tolist(),
        unique_masks=int(np.unique(masks,axis=0).shape[0]), u_exponent=[int(ue.min()),int(ue.max())],
        aligned_v_shift=[int(shift[vn].min()),int(shift[vn].max())],
        tau_signed_bits=signed_width(tau.min(),tau.max()), domains_by_region=bounds,
        U8_bytes=u.size, V_code5_nonzero_bits=int(vn.sum())*5,
        V_nonzero_bitmap_bytes=(vn.size+7)//8, mask_bytes=(masks.size+7)//8,
        A_bytes=200, tau_dense48_bytes=5760,
        source_scope='four existing validation source captures only; new downstream AEE unmeasured')
    return arrays, report


def run_sources(arrays):
    source_dir = BASE / 'algorithm/patch_probe/partial_completion/integer_valid10'
    rows, words_out, ids_out = [], [], []
    u, v, a = arrays['u'].astype(np.int64), arrays['v'], arrays['a'].astype(np.int64)
    # Integer independent expansion; no old Yi or gate is used as an answer.
    for path in sorted(source_dir.glob('capture_*.npz'))[:4]:
        cap = load(path)
        words = cap['source_gate_words'].astype(np.int64)
        ids = cap['group_ids'].astype(np.int64)
        x = ((words[...,None] >> np.arange(10)) & 1).transpose(0,3,2,1)
        region = spatial_regions(ids)
        mask = arrays['masks'][region]
        scalar_mask = mask.repeat(4, axis=1)
        z = (x @ u) * scalar_mask[:,None,None,:]
        y = z @ v
        uv = np.einsum('ts,gsph->gtph',a,y)
        q = np.einsum('ts,gspr->gtpr',a,z)
        ua = q @ v
        np.testing.assert_array_equal(ua,uv)
        # Actual TC decode/compact producer/TR address reference, no dense Y helper.
        ys, stats = sparse_forward(x,u,v,mask,latent_tile=4,output_tile=8,banks=8)
        np.testing.assert_array_equal(ys,y)
        gate = uv >= arrays['tau'][None,:,None,:]
        # FP64 reference to the saved FP32 weights, mathematical comparison only.
        fz = (x @ (arrays['fp_u'].astype(np.float64)*float(arrays['theta_source']))) * scalar_mask[:,None,None,:]
        fy = fz @ arrays['fp_v'].astype(np.float64)
        fmargin = np.einsum('ts,gsph->gtph',arrays['fp_a'].astype(np.float64),
            fy*arrays['bn_scale']+arrays['bn_bias']) + arrays['temporal_bias'][None,:,None,None]-float(arrays['theta_output'])
        fg = fmargin >= 0
        qy = np.ldexp(y.astype(np.float64), arrays['y_exponent'])
        dr = dict(file=str(cap['file']), groups=len(ids), region_counts=np.bincount(region,minlength=8).tolist(),
            TC_TR_Y_equal=True, A_V_V_A_equal=True, Y_values=y.size, U_gate_values=uv.size,
            quantized_vs_saved_weight_gate_flips=int((gate!=fg).sum()),
            saved_weight_gate_nonzeros=int(fg.sum()), integer_gate_nonzeros=int(gate.sum()),
            Y_relative_RMSE=float(np.sqrt(np.mean((qy-fy)**2)/max(np.mean(fy**2),1e-30))),
            observed_Z_abs=int(abs(z).max()), observed_Y_abs=int(abs(y).max()),
            observed_Q_abs=int(abs(q).max()), observed_U_abs=int(abs(uv).max()),
            TC_TR_reference_counts={k:int(val) for k,val in stats.items()},
            A_before_V_dense_MACs=int(10*10*4*scalar_mask.sum()),
            V_before_A_dense_MACs=int(len(ids)*10*10*4*96))
        rows.append(dr)
        words_out.append(words.astype(np.int16)); ids_out.append(ids)
    # Source-only artifact plus model arrays regenerate all gold; no validation-derived tuning.
    return rows


def main():
    paths = dict(ordinary_r32=FACTOR/'latent_stage_train16/flow_recovery64/preview_only/shared48_u8_vq5.npz',
        regional_r96_a48=FACTOR/'demand_train16/shared.npz')
    report = dict(scope='One fixed quantization, no fitting, no network AEE, no RTL cycle claim',models={})
    for name,path in paths.items():
        arrays, row = compile_one(name,load(path))
        row['input'] = str(path)
        row['source_checks'] = run_sources(arrays)
        report['models'][name] = row
        print(name,json.dumps(dict(rank=row['rank'],active=row['active_rank_per_region'],
            gate_flips=sum(x['quantized_vs_saved_weight_gate_flips'] for x in row['source_checks']),
            gate_values=sum(x['U_gate_values'] for x in row['source_checks']),
            y_relative_rmse=[x['Y_relative_RMSE'] for x in row['source_checks']])))
    (HERE/'INTEGER_RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
