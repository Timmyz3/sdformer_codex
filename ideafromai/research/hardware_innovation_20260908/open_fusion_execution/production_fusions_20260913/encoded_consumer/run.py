"""Fixed actual-source encoded-consumer experiment; no training or fitting."""
from pathlib import Path
import argparse
import copy
import json
import sys
import numpy as np
import binding
import kernel
import predictor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(binding.REP))
from quantizers import reconstruct

ARMS = {'raw': (None, False), 'fixed_expanded': ('fixed_q8', False),
        'fixed_code8': ('fixed_q8', True), 'affine_expanded': ('affine_q8', False),
        'affine_code8': ('affine_q8', True), 'full_expanded': ('full_g_q8', False),
        'full_code8': ('full_g_q8', True)}


def rne_sat(x, shift):
    if shift > 0:
        a, b = np.divmod(x, 1 << shift)
        x = a+((2*b > 1 << shift) | ((2*b == 1 << shift) & ((a & 1) != 0)))
    else:
        x = x << -shift
    return np.clip(x, -(1 << 23), (1 << 23)-1)


def oracle(data, q, label, p):
    geo = json.loads(str(data['window_geometry_json']))[label]
    dy, dx = [2*a-b for a, b in zip(geo['output_origin'], geo['gate_origin'])]
    x = data[label+'_updated_I24'][:, :, dy:dy+8:2, dx:dx+8:2].astype(np.int64)
    g = data[label+'_proj_gate'][:, :, dy:dy+8:2, dx:dx+8:2].astype(np.int64)
    estimate, code, counts = reconstruct(x, g, p)
    acc = np.einsum('rk,tkij->trij', q['U_ped_q16'].astype(np.int64), estimate, dtype=np.int64)
    u = rne_sat(acc, int(q['U_ped_exponent']))
    acc_v = np.einsum('hr,trij->thij', q['V_ped_q16'].astype(np.int64), u, dtype=np.int64)
    ped = rne_sat(rne_sat(acc_v, int(q['V_ped_exponent']))+q['PED_bias_q24'][None, :, None, None], 0)
    return dict(continuous=ped, U_ped=u), dict(quantizer_counts=counts,
        quantized_values=int(estimate.size), code_nonzero=int(np.count_nonzero(code)),
        PED_vs_unquantized=binding.difference(ped, data[label+'_continuous_q24']))


def run(axis, label, arms, stress=False, suffix='final'):
    data, live, q, params = binding.load(axis, label)
    print('PREFIX_START', axis, label, stress, flush=True)
    _, producer, prefix = binding.run_prefix(kernel.QuantMachine, data, live, axis, label, stress)
    print('PREFIX_DONE', prefix.time, flush=True)
    result = dict(axis=axis, window=label, stress=stress, requested_arms=arms,
        producer=producer, rows=[], complete=False,
        scope='Same actual Machine I24 source -> preview/sn2 -> full K864/Conv2/merge/projection gate -> real quantizer -> U/V -> actual PED and gate egress. No native/globalBN/join.',
        evidence='CPU payload single-issue/port schedule, not RTL/PPA.',
        resources=dict(RF_vectors=96, lanes=8, RF_bits=48, state_bytes=131072,
                       coefficient_bytes=131072, SR64=1, SW64=1, CR256=1,
                       common_staging_bytes=64, source_ROM_bytes=8192),
        fixed_point_parent='Existing old R24/onepass ordinary or lifting_raw export; not new matched320 or two-term source.',
        common_source_supply='H8_T10_RF60_69',
        RF_live_bounds=dict(full_U=73, full_CSE_with_paid_spill=96, V=90),
        new_AEE=False, new_training=False, production_modified=False)
    path = HERE/f'{axis}_{label}_{suffix}{"_stress" if stress else ""}.json'
    for arm in arms:
        mode, encoded = ARMS[arm]
        p = params[mode] if mode else None
        cse_plan = None
        if arm == 'full_code8':
            import latent_cse
            cse_plan = latent_cse.get_plan(axis, p['D'])
        gold, delta = oracle(data, q, label, p) if p is not None else (None, {})
        audit = predictor.audit_integer_interface(p['D'], p['c'], p['step'], q['U_ped_q16']) if p is not None else None
        if audit:
            assert audit['all_code_reconstruction_signed24_safe'] and audit['complete_U_accumulator_signed48_safe']
        m = copy.deepcopy(prefix)
        callback = kernel.make_callback(p, encoded, cse_plan)
        print('ARM_START', axis, label, arm, stress, flush=True)
        value, report = binding.consumer_run(data, q, label, callback, machine=m, stress=stress, gold=gold)
        assert sum(m.stages.values()) == m.time
        counts = dict(m.count)
        row = dict(arm=arm, quantizer=mode, encoded=encoded, service_slots=m.time,
            consumer_service_slots=report['service_slots'], producer_end=prefix.time,
            same_executed_prefix_copied=True, consumer=report, counts=counts,
            stages=dict(m.stages), precision_delta=delta, arithmetic_bounds=audit,
            port_bytes=dict(SR64=8*counts['SR64_reads'], SW64=8*counts['SW64_writes'],
                            CR256=32*counts['CR256_reads'], CW256=32*counts['CW256_writes']),
            scratch=dict(code8=[kernel.CODE, kernel.CODE+960], expanded=[kernel.RECON,kernel.RECON+2880],
                         scopes='Scratch per P1; full real updated/projection remain. Inactive arm scratch is not read/written.'),
            all_U_and_PED_compared=True, original_U_V_RNE_bias_preserved=True,
            expanded_fuses_generated_code_into_I24_without_intermediate_Q8_spill=not encoded and mode is not None,
            whole_layer=False, whole_network=False)
        if cse_plan is not None:
            row['latent_CSE_plan'] = {k:v for k,v in cse_plan.items() if k not in ('instructions','nodes','matrix')}
        result['rows'].append(row)
        path.write_text(json.dumps(result, indent=2, default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
        print('ARM_DONE', arm, m.time, report['service_slots'], flush=True)
    result['complete'] = True
    path.write_text(json.dumps(result, indent=2, default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
    print('BATCH_DONE', path.name, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('axis', choices=('ordinary','lifting_raw'))
    ap.add_argument('label', choices=('corner','interior'))
    ap.add_argument('--arms', default=','.join(ARMS))
    ap.add_argument('--stress', action='store_true')
    ap.add_argument('--suffix', default='final')
    args = ap.parse_args()
    run(args.axis, args.label, args.arms.split(','), args.stress, args.suffix)
