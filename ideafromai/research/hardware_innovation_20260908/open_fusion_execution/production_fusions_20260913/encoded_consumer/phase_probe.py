"""Fixed integer census of full-D Q8 -> signed9 z + phase11 rewriting.

No new fit, issue model, kernel change, GPU, EDA, or performance estimate.
Saved updated/gate arrays are census inputs, not claimed hardware suppliers.
"""
from pathlib import Path
import json
import sys
import numpy as np
import binding

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(binding.REP))
from quantizers import reconstruct, signed_bits

STEP = 2048


def rne_div(value, step):
    quo, rem = np.divmod(value, step)
    return quo + ((2*rem > step) | ((2*rem == step) & ((quo & 1) != 0)))


def rne_sat(value, exponent):
    value = rne_div(value, 1 << exponent) if exponent > 0 else value << -exponent
    return np.clip(value, -(1 << 23), (1 << 23)-1)


def phase_encode(x, a, phase):
    """Exact translated ties-to-even and translated original q8 clipping."""
    k, rem = np.divmod(x-phase, STEP)
    # The original rounded q must be even on ties. Since z=q+a, the
    # parity target for z is a&1, not necessarily zero.
    z = k + ((2*rem > STEP) | ((2*rem == STEP) & (((k-a) & 1) != 0)))
    return np.clip(z, a-128, a+127)


def integer_range(value):
    return dict(min=int(np.min(value)), max=int(np.max(value)), signed_bits=signed_bits(value))


def universal(p, u):
    assert np.all(p['step'] == STEP)
    gates = ((np.arange(1024)[:, None] >> np.arange(10)) & 1).T
    b = p['D'] @ gates + p['c'][:, None]
    a, phase = np.divmod(b, STEP)
    baseline_phase = np.remainder(p['c'], STEP)
    delta = phase-baseline_phase[:, None]
    lo, hi = a-128, a+127
    z_bits = signed_bits(np.stack((lo, hi)))
    assert z_bits == 9
    raw_min, raw_max = int(np.min(b-128*STEP)), int(np.max(b+127*STEP))
    assert -(1 << 23) <= raw_min <= raw_max < (1 << 23)
    max_u_l1 = int(np.abs(u.astype(np.int64)).sum(1).max())
    triangle48 = max_u_l1*(STEP*max(abs(int(lo.min())), abs(int(hi.max())))
                          + int(np.abs(delta).max())+int(np.abs(baseline_phase).max()))
    assert triangle48 < (1 << 47)
    # Enumerate all gword/time/q8 values, not just observed gate classes.
    exhaustive = 0
    for code in range(-128, 128):
        z = code+a
        assert np.array_equal(STEP*z+phase, STEP*code+b)
        assert np.all((-256 <= z) & (z <= 255))
        exhaustive += z.size
    # Rounding/clipping adversarial boundaries, including far clipped codes.
    # General exactness follows from the quotient/remainder proof in the MD;
    # this is not enumeration of every signed24 x.
    tested = clipped = naive_differences = 0
    first_naive_counterexample = None
    ns = (-4096, -130, -129, -128, -1, 0, 1, 126, 127, 128, 129, 4095)
    residues = (0, 1, 1023, 1024, 1025, 2047)
    for n in ns:
        for residue in residues:
            x = b+STEP*n+residue
            legal = (x >= -(1 << 23)) & (x < (1 << 23))
            unclipped = rne_div(x-b, STEP)
            q = np.clip(unclipped, -128, 127)
            z = phase_encode(x, a, phase)
            expected = q+a
            assert np.array_equal(z[legal], expected[legal])
            naive = np.clip(rne_div(x-phase, STEP), a-128, a+127)
            wrong = legal & (naive != expected)
            if first_naive_counterexample is None and np.any(wrong):
                t, word = np.argwhere(wrong)[0]
                first_naive_counterexample = dict(time=int(t), gword=int(word),
                    x=int(x[t, word]), b=int(b[t, word]), a=int(a[t, word]),
                    phase=int(phase[t, word]), original_q=int(q[t, word]),
                    exact_z=int(expected[t, word]), naive_even_z=int(naive[t, word]))
            naive_differences += int(np.count_nonzero(wrong))
            clipped += int(np.count_nonzero(legal & (q != unclipped)))
            tested += int(np.count_nonzero(legal))
    assert naive_differences > 0
    return dict(gwords=1024, time_rows=10, q8_codes=256,
                exhaustive_decode_values=int(exhaustive), decode_differences=0,
                b_range=integer_range(b), a_range=integer_range(a),
                D_signed_bits=signed_bits(p['D']),
                z_universal=dict(min=int(lo.min()), max=int(hi.max()), signed_bits=z_bits),
                z_per_time_min=lo.min(1).tolist(), z_per_time_max=hi.max(1).tolist(),
                phase_range=integer_range(phase), phase_storage_unsigned_bits=11,
                phase_delta_range=integer_range(delta), phase_at_gword0=baseline_phase.tolist(),
                universal_nonzero_phase_delta_fraction=float(np.mean(delta != 0)),
                distinct_phase_vectors=int(np.unique(phase.T, axis=0).shape[0]),
                distinct_a_phase_vectors=int(np.unique(np.concatenate((a, phase), axis=0).T, axis=0).shape[0]),
                reconstructed_min=raw_min, reconstructed_max=raw_max, universal_no_sat24=True,
                phase_factor_triangle_accumulator_abs_bound=triangle48,
                phase_factor_universal_signed48_pass=True,
                encoder_boundary_values=tested, encoder_clipped_values=clipped,
                encoder_exact_differences=0, naive_even_z_differences=naive_differences,
                first_naive_even_z_counterexample=first_naive_counterexample)


def anchors(data, label):
    geo = json.loads(str(data['window_geometry_json']))[label]
    y, x = geo['gate_origin']; oy, ox = geo['output_origin']
    dy, dx = 2*oy-y, 2*ox-x
    take = lambda field: data[label+'_'+field][:, :, dy:dy+8:2, dx:dx+8:2].astype(np.int64)
    return take('updated_I24'), take('proj_gate')


def window_case(axis, label, data, q, params):
    x, gates = anchors(data, label)
    p = params['full_g_q8']
    estimate, code, counts = reconstruct(x, gates, p)
    b = (p['D'] @ gates.reshape(10, -1)+p['c'][:, None]).reshape(x.shape)
    a, phase = np.divmod(b, STEP)
    z = code+a
    phase0 = np.remainder(p['c'], STEP)
    delta = phase-phase0[:, None, None, None]
    assert np.array_equal(z, phase_encode(x, a, phase))
    assert np.array_equal(estimate, np.clip(STEP*z+phase, -(1 << 23), (1 << 23)-1))
    assert counts['reconstructed_sat24'] == 0
    u, v = q['U_ped_q16'].astype(np.int64), q['V_ped_q16'].astype(np.int64)
    uz = np.einsum('rk,tkij->trij', u, z, dtype=np.int64)
    ud = np.einsum('rk,tkij->trij', u, delta, dtype=np.int64)
    base = phase0[:, None]*u.sum(1)[None, :]
    factored = STEP*uz+ud+base[:, :, None, None]
    direct = np.einsum('rk,tkij->trij', u, estimate, dtype=np.int64)
    assert np.array_equal(factored, direct)
    assert max(np.abs(uz).max(), np.abs(ud).max(), np.abs(factored).max()) < (1 << 47)
    latent = rne_sat(factored, int(q['U_ped_exponent']))
    latent_ref = rne_sat(direct, int(q['U_ped_exponent']))
    ped = rne_sat(rne_sat(np.einsum('hr,trij->thij', v, latent, dtype=np.int64),
                         int(q['V_ped_exponent']))+q['PED_bias_q24'][None, :, None, None], 0)
    ped_ref = rne_sat(rne_sat(np.einsum('hr,trij->thij', v, latent_ref, dtype=np.int64),
                             int(q['V_ped_exponent']))+q['PED_bias_q24'][None, :, None, None], 0)
    assert np.array_equal(latent, latent_ref) and np.array_equal(ped, ped_ref)
    words = np.sum(gates*(1 << np.arange(10))[:, None, None, None], axis=0)
    keys, amounts = np.unique(words, return_counts=True)
    word_groups = words.reshape(12, 8, 4, 4).transpose(0, 2, 3, 1).reshape(-1, 8)
    unique_per_h8 = np.array([np.unique(row).size for row in word_groups])
    h8_zero = np.all(delta.reshape(10, 12, 8, 4, 4) == 0, axis=2)
    t10_h8_zero = np.all(h8_zero, axis=0)
    ordinary = {}
    for mode in ('fixed_q8', 'affine_q8'):
        ref, qc, qc_counts = reconstruct(x, gates, params[mode])
        ordinary[mode] = dict(code_range=integer_range(qc), reconstructed_range=integer_range(ref),
                             quantizer_counts=qc_counts, code8_dense_bytes=int(qc.size),
                             code16_dense_bytes=int(qc.size*2), has_gate_predictor=False,
                             U_V_coefficients='Same original Q16 matrices as full_g_q8.',
                             common_permissions='P1 keep-Z, source RF, exact dyadic folding, Uc, CSE, legal zero bypass.')
    return dict(axis=axis, window=label, positions=16, scalar_values=int(x.size),
                code_range=integer_range(code), z_range=integer_range(z),
                z_per_time_signed_bits=[signed_bits(row) for row in z],
                b_range=integer_range(b), a_range=integer_range(a),
                phase_range=integer_range(phase), phase_delta_range=integer_range(delta),
                quantizer_counts=counts, full_g_reconstruction_differences=0,
                pre_U_values=int(direct.size), pre_U_differences=0,
                U_values=int(latent.size), U_differences=0, PED_values=int(ped.size), PED_differences=0,
                U_accumulator_range=integer_range(factored),
                gword_values=int(words.size), gword_classes=int(keys.size),
                gword_histogram={str(int(k)): int(n) for k, n in zip(keys, amounts)},
                gword_zero_count=int(np.count_nonzero(words == 0)),
                gword_zero_fraction=float(np.mean(words == 0)),
                phase_nonzero_count=int(np.count_nonzero(phase)),
                phase_delta_nonzero_count=int(np.count_nonzero(delta)),
                phase_delta_nonzero_fraction=float(np.mean(delta != 0)),
                phase_delta_nonzero_per_time=np.count_nonzero(delta.reshape(10, -1), axis=1).tolist(),
                T_H8_groups=int(h8_zero.size), T_H8_allzero_groups=int(np.count_nonzero(h8_zero)),
                T_H8_allzero_fraction=float(np.mean(h8_zero)),
                T10_H8_groups=int(t10_h8_zero.size), T10_H8_allzero_groups=int(np.count_nonzero(t10_h8_zero)),
                T10_H8_allzero_fraction=float(np.mean(t10_h8_zero)),
                gword_classes_per_H8_histogram={str(i): int(np.count_nonzero(unique_per_h8 == i)) for i in range(1, 9)},
                logical_payload_bytes=dict(original_q8=int(code.size), z_packed9=int((z.size*9+7)//8),
                    z_expanded16=int(z.size*2), phase_packed11=int((phase.size*11+7)//8),
                    phase_expanded16=int(phase.size*2), existing_gate_words=int(words.size*2),
                    expanded_I24=int(estimate.size*3)),
                unscheduled_obligations=dict(
                    all_input_I24_scalars_to_encode=int(x.size), gword_selectors_without_reuse=int(words.size),
                    nonzero_gword_selectors_after_constant_g0_bypass=int(np.count_nonzero(words)),
                    possible_phase_delta_MAC8_if_scalar_zero_bypass=int(np.count_nonzero(delta))*3,
                    original_gate_Ug_AAC8_before_CSE=int(np.count_nonzero(gates))*3,
                    original_dense_D_times_Ug_vector_terms_before_CSE=int(np.count_nonzero(p['D']))*16*3,
                    warning='Counts are obligations, not cycles. Different terms/op widths cannot be summed or compared as service. Both controls retain complete CSE.'),
                ordinary_controls=ordinary)


def main():
    result = dict(scope=__doc__, fixed_step=STEP, parents={}, rows=[],
        exact_parent='Old stage_20260912 ordinary/lifting_raw R24+onepass; not matched320 or W8.',
        changed_parameters=False, training=False, GPU=False, EDA=False, issue_model=False,
        new_AEE=False, complete=False,
        table_obligations=dict(gwords=1024, time_rows=10, a_signed_bits=9, phase_unsigned_bits=11,
            phase_only_dense_bits_bytes=14080, complete_a_phase_dense_bits_bytes=25600,
            packed_32B_per_gword_image_bytes=32768, packed_CR256_responses_per_gword=1,
            expanded32_per_entry_bytes=40960, expanded64B_per_gword_image_bytes=65536,
            packed_decoder='Ten 20bit fields need real selectors, unsigned11 phase extraction and signed9 a extension; one CR response is not ten free scalar loads.',
            g0_bypass='a0/phase0 are constant; detect actual zero word with paid gate read/selection. Nonzero-word table remains complete1024.',
            code_spill='z needs signed9, so either paid9bit packing or16bit storage; phase needs lookup reuse or a separately charged spill/reread.',
            multiply='U16*z9 plus U16*delta_phase12 fit existing16x24 lanes. Avoids explicitD19*Ug only when all added encode/select/phase operations are paid.',
            baseline='Ordinary fixed/affine with sameQ16 U/V, same16bit code permission if used, completeCSE; original fullD also gets completeCSE.'))
    for axis in ('ordinary', 'lifting_raw'):
        for label in ('corner', 'interior'):
            data, _, q, params = binding.load(axis, label)
            if axis not in result['parents']:
                result['parents'][axis] = universal(params['full_g_q8'], q['U_ped_q16'])
            row = window_case(axis, label, data, q, params)
            result['rows'].append(row)
            print(axis, label, 'z', row['z_range'], 'phase_delta_nonzero',
                  row['phase_delta_nonzero_fraction'], 'H8_skip', row['T_H8_allzero_fraction'],
                  'gclasses', row['gword_classes'], 'qclip', row['quantizer_counts'], flush=True)
    result['complete'] = True
    result['totals'] = dict(cases=len(result['rows']),
        exhaustive_decode_values=sum(r['exhaustive_decode_values'] for r in result['parents'].values()),
        encoder_boundary_values=sum(r['encoder_boundary_values'] for r in result['parents'].values()),
        scalar_reconstruction_values=sum(r['scalar_values'] for r in result['rows']),
        pre_U_values=sum(r['pre_U_values'] for r in result['rows']),
        U_values=sum(r['U_values'] for r in result['rows']), PED_values=sum(r['PED_values'] for r in result['rows']),
        all_exact_differences=0)
    path = HERE/'phase_results.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print('PHASE_PROBE_COMPLETE', json.dumps(result['totals']), flush=True)


if __name__ == '__main__':
    main()
