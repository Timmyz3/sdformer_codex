"""Bounded audit: existing affine parameters, W4 parent, and recorded ports.

No refitting, GPU runs, or new candidate selection. Reads the already-executed
receipts and independently verifies the exported affine range and endpoints.
"""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent
REP = HERE.parent / 'representation'
OLD = HERE.parents[1] / 'stage_20260912'


def main():
    report = dict(scope='Independent control/identity/recorded-port audit only; no new AEE or cycle simulation.', axes={})
    for axis in ('ordinary', 'lifting_raw'):
        p = dict(np.load(REP/'parameters'/f'{axis}.npz'))
        z = np.load(REP/'train_captures'/f'{axis}.npz')
        x = z['updated_I24'].reshape(10, -1).astype(np.int64)
        lo, hi = x.min(1), x.max(1)
        # Integer range proof, independent of quantizers.fit's log2 formula.
        steps = []
        for width in hi-lo:
            s = 1
            while 255*s < int(width):
                s *= 2
            steps.append(s)
        step = np.asarray(steps, np.int64)
        center = np.rint((lo+hi+step)/2).astype(np.int64)
        assert np.array_equal(step, p['affine_q8_step'])
        assert np.array_equal(center, p['affine_q8_c'])
        assert not np.any(p['affine_q8_D'])
        lower, upper = center-128*step, center+127*step
        assert np.all(lower <= lo) and np.all(upper >= hi)
        # These magnitudes and powers of two are exact in float64. This check
        # is independent of the quotient/remainder integer implementation.
        code = np.rint((x-center[:, None])/step[:, None]).astype(np.int64)
        assert np.all((code >= -128) & (code <= 127))
        reconstruction = center[:, None]+code*step[:, None]
        assert np.all(np.abs(reconstruction-x)*2 <= step[:, None])
        affine = dict(training_values=int(x.size), independent_steps_equal=True,
            independent_center_equal=True, D_nonzero=0, train_codes_clipped=0,
            step=step.tolist(), center=center.tolist(),
            minimum_endpoint_margin=min(int((lo-lower).min()), int((upper-hi).min())),
            max_error=int(np.abs(reconstruction-x).max()),
            fixed_q8_steps=p['fixed_q8_step'].tolist(),
            same_step_as_fixed=bool(np.array_equal(step,p['fixed_q8_step'])),
            inference='Ordinary per-time signed8 affine/dyadic range; not an equal-step Dg-only ablation.')

        lowbit = dict(np.load(OLD/'weight_compensation'/f'{axis}_lowbit_gpu_parameters.npz'))
        new = dict(np.load(OLD/'algorithm/hardware_exports'/axis/'deployed_constants.npz'))
        weight = {}
        for mode in ('W8', 'W4'):
            old = dict(np.load(OLD/'algorithm/weight_controls/aee'/axis/mode/'deployed_constants.npz'))
            expanded = lowbit[mode+'_code'].astype(np.int64)*lowbit[mode+'_scale_q16'][:, None].astype(np.int64)
            assert np.array_equal(expanded, old['U_ped_q16'])
            assert np.array_equal(lowbit[mode+'_V'], old['V_ped_q16'])
            weight[mode] = dict(code_times_scale_equal_actual_deployed_U=True,
                original_U_shape=list(old['U_ped_q16'].shape), original_V_shape=list(old['V_ped_q16'].shape),
                U_exponent=int(old['U_ped_exponent']), V_exponent=int(old['V_ped_exponent']),
                current_R24_U_shape=list(new['U_ped_q16'].shape), current_R24_V_shape=list(new['V_ped_q16'].shape),
                same_rank_parent=False,
                parent='Original R32 + CUDA BN; new representation uses current R24 + onepass BN.')
        report['axes'][axis] = dict(affine=affine, weight_parent=weight)

    ports = []
    for name in ('ordinary_corner', 'ordinary_interior', 'lifting_raw_corner',
                 'lifting_raw_interior', 'ordinary_interior_stress'):
        record = json.loads((HERE/(name+'.json')).read_text())
        assert record['complete']
        for mode in ('W8','W4'):
            expanded = next(r for r in record['rows'] if r['mode']==mode+'_expanded16')
            packed = next(r for r in record['rows'] if r['mode']==mode+'_packed')
            delta = {p:packed['physical_port_bytes'][p]-expanded['physical_port_bytes'][p]
                     for p in ('SR64','SW64','CR256','CW256')}
            assert delta['SR64']==delta['SW64']==0
            assert delta['CR256']==(-23808 if mode=='W8' else -36096)
            assert delta['CW256']==(-2976 if mode=='W8' else -4512)
            assert packed['same_machine_handoff'] and packed['literal_packed_code_in_memory']
            assert not packed['free_third_RF_port']
            ports.append(dict(case=name,mode=mode,port_delta=delta,
                same_function_delta_slots=packed['service_slots']-expanded['service_slots']))
    report['ports'] = ports
    report['complete'] = True
    (HERE/'independent_control_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
