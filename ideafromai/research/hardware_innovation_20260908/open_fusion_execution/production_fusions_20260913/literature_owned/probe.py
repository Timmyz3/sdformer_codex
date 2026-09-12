"""Fixed Q8 consumer-entry probe; no fitting, GPU, EDA, or production writes.

Numerical scope: all five existing train-only quantizers, old R24 parents,
four 4x4 anchor windows, complete original U/V/RNE/bias. Timing scope: the
three no-gate quantizers, ready source representation -> stored U24 only.
"""
from pathlib import Path
from collections import Counter
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
OPEN = HERE.parents[1]
REP = OPEN / 'breadth_20260912/representation'
EXPORT = OPEN / 'stage_20260912/algorithm/hardware_exports'
sys.path.insert(0, str(REP))
from quantizers import MODES, reconstruct, signed_bits
sys.path.insert(0, str(OPEN / 'breadth_20260912/hardware'))
import packed_weights as shared

INPUT, OUTPUT = 0, 65536
LIMIT = 1 << 47


def pack_signed(x, size):
    return b''.join(int(v).to_bytes(size, 'little', signed=True)
                    for v in np.asarray(x).flat)


def rne_sat(x, exponent):
    x = np.asarray(x, dtype=np.int64)
    if exponent > 0:
        q, r = np.divmod(x, 1 << exponent)
        x = q + ((2*r > 1 << exponent) | ((2*r == 1 << exponent) & ((q & 1) != 0)))
    else:
        x = x << -exponent
    return np.clip(x, -(1 << 23), (1 << 23)-1)


def bounds(p, u):
    D, c, s = [p[k].astype(np.int64) for k in ('D', 'c', 'step')]
    lo = c + np.minimum(D, 0).sum(1) - 128*s
    hi = c + np.maximum(D, 0).sum(1) + 127*s
    universal_no_sat = bool(np.all(lo >= -(1 << 23)) and np.all(hi < 1 << 23))
    abs_u = np.abs(u).sum(1)
    # Triangle bounds deliberately include independently impossible extrema.
    q_scaled = int(np.max(abs_u) * 128 * np.max(s))
    gate_latent = int(np.max(abs_u))
    gate_decoded = int(np.max(np.sum(np.abs(D), axis=1)) * gate_latent)
    bias = int(np.max(np.abs(c)) * np.max(abs_u))
    total = q_scaled + gate_decoded + bias
    assert total < LIMIT, 'This factoring cannot use the common signed48 accumulator.'
    return dict(all_binary_gate_patterns=1024, all_signed8_codes=True,
                reconstructed_min=lo.tolist(), reconstructed_max=hi.tolist(),
                universal_no_sat24=universal_no_sat,
                gate_latent_abs_bound=gate_latent,
                residual_scaled_abs_bound=q_scaled,
                decoded_gate_abs_bound=gate_decoded,
                bias_abs_bound=bias, triangle_accumulator_abs_bound=total,
                signed48_bound_pass=True)


def numeric_case(x, g, p, u, v, bias, eu, ev):
    """Independent integer tensor contraction; no intermediate RNE added."""
    estimate, code, quant_counts = reconstruct(x, g, p)
    D, c, s = [p[k].astype(np.int64) for k in ('D', 'c', 'step')]
    t, k, height, width = x.shape
    z = height*width
    q = code.reshape(t, k, z)
    gate = g.reshape(t, k, z)
    raw = (D @ gate.reshape(t, -1)).reshape(t, k, z) + c[:, None, None] + s[:, None, None]*q
    correction = np.clip(raw, -(1 << 23), (1 << 23)-1)-raw
    uq = np.einsum('rk,tkp->trp', u, q, dtype=np.int64)
    ug = np.einsum('rk,tkp->trp', u, gate, dtype=np.int64)
    decoded = np.einsum('tj,jrp->trp', D, ug, dtype=np.int64)
    uc = c[:, None]*u.sum(1)[None, :]
    ue = np.einsum('rk,tkp->trp', u, correction, dtype=np.int64)
    acc = uq*s[:, None, None] + decoded + uc[:, :, None] + ue
    direct = np.einsum('rk,tkp->trp', u, estimate.reshape(t, k, z), dtype=np.int64)
    assert np.array_equal(acc, direct)
    assert max(int(np.abs(a).max()) for a in (uq, ug, decoded, ue, acc)) < LIMIT
    latent = rne_sat(acc, eu)
    latent_ref = rne_sat(direct, eu)
    ped_acc = np.einsum('hr,trp->thp', v, latent, dtype=np.int64)
    ped = rne_sat(rne_sat(ped_acc, ev)+bias[None, :, None], 0)
    ref_acc = np.einsum('hr,trp->thp', v, latent_ref, dtype=np.int64)
    ref = rne_sat(rne_sat(ref_acc, ev)+bias[None, :, None], 0)
    assert np.array_equal(ped, ref)
    rank = u.shape[0]
    groups = rank//8
    # The lower counts below are mechanism obligations, not scheduled cycles.
    active_channels = np.count_nonzero(gate, axis=1)
    incoming_gate_aac = int(active_channels.sum())*groups if np.any(D) else 0
    d_terms = int(np.count_nonzero(D))*z*groups
    row = dict(U_accumulator_values=int(acc.size), U_values=int(latent.size),
               PED_values=int(ped.size), pre_RNE_differences=0,
               U_differences=0, PED_differences=0,
               correction_nonzero=int(np.count_nonzero(correction)),
               quantizer_counts=quant_counts,
               code_signed_bits=signed_bits(code),
               Uq_signed_bits=signed_bits(uq), Ug_signed_bits=signed_bits(ug),
               decoded_gate_signed_bits=signed_bits(decoded),
               final_accumulator_signed_bits=signed_bits(acc),
               required_extra_gate_projection_AAC8=incoming_gate_aac,
               required_nonzero_D_times_Ug_vector_products=d_terms,
               D_signed_bits=signed_bits(D),
               D_product_warning='A signed19 D times Ug cannot be assumed one existing signed16x24 product; split/shift/add or constant CSE must be paid.',
               code_bytes=int(code.size), reconstructed_I24_bytes=int(code.size)*3,
               prediction_materialization_bytes_avoided_if_factored=int(code.size)*3,
               constant_latent_base_bytes=int(uc.size)*6,
               output_U24_bytes=int(latent.size)*3,
               output_PED24_bytes=int(ped.size)*3)
    return row, estimate, code, latent


class EntryMachine(shared.PackedMachine):
    def collect_code8(self, address):
        # Same scalar collector permission as ordinary collect_i24. No
        # free SIMD gather: read actual SR64 response, then ILOAD below.
        raw = self.read_word(address)
        self.scalar_collector = int.from_bytes(raw[address % 8:address % 8+1],
                                              'little', signed=True)
        self.count['code8_scalar_collections'] += 1

    def cache48(self, reg, address):
        payload = bytearray()
        remaining = 48
        while remaining:
            self.coefficient(address)
            n = min(remaining, 32-address % 32)
            payload += self.cword[address % 32:address % 32+n]
            remaining -= n
            address += n
        assert len(payload) == 48  # Fits the existing common64B staging.
        vals = [int.from_bytes(payload[i:i+6], 'little', signed=True)
                for i in range(0, 48, 6)]
        self.wait_reg(reg)
        self.advance(op=('ILOAD', reg, vals), tag='constant48_RF_load')
        self.wait_reg(reg)


def execute_entry(estimate, code, p, u, eu, mode):
    """Actual payload issue/port model, 16 P1 contexts, same SRAM layout.

    expanded_I24 starts with decoded xhat available free at this boundary.
    factored_code24 and factored_code8 use exactly the same factor placement.
    """
    assert mode in ('expanded_I24', 'factored_code24', 'factored_code8')
    assert not np.any(p['D'])
    assert bounds(p, u)['universal_no_sat24']
    rank, k_count = u.shape
    groups = rank//8
    assert rank == 24
    to_spatial = lambda a: a.transpose(2, 3, 0, 1).reshape(16, 10, 96)
    x = to_spatial(estimate)
    q = to_spatial(code)
    factor = mode != 'expanded_I24'
    width = 1 if mode == 'factored_code8' else 3
    operand = q if factor else x
    image = bytearray(u.T.astype('<i2').tobytes())
    base_address = len(image)
    uc = p['c'].astype(np.int64)[:, None]*u.sum(1)[None, :]
    has_base = bool(np.any(uc))
    if factor and has_base:
        image += pack_signed(uc, 6)
    m = EntryMachine(False)
    # Common already-ready representation boundary; initial writes and the
    # encoder do not enter either endpoint. Coefficient cold fill does.
    raw = pack_signed(operand, width)
    m.state[INPUT:INPUT+len(raw)] = raw
    m.phase = 'coefficient_cold_fill'
    m.dma_input(bytes(image), 0, True)
    if factor and has_base:
        m.phase = 'precomputed_Uc_RF_cache'
        for t in range(10):
            for hg in range(groups):
                m.cache48(40+t*groups+hg, base_address+(t*rank+hg*8)*6)
    setup = m.time
    exact_acc = np.empty((16, 10, rank), dtype=np.int64)
    for pos in range(16):
        m.phase = 'U_accumulator_clear'
        for reg in range(10*groups):
            m.advance(op=('clear', reg, None), tag='output_clear')
        for k in range(96):
            m.phase = 'source_representation_gather'
            for t0 in (0, 8):
                vals = []
                for t in range(t0, min(10, t0+8)):
                    address = INPUT+((pos*10+t)*96+k)*width
                    if width == 1:
                        m.collect_code8(address)
                    else:
                        m.collect_i24(address)
                    vals.append(m.scalar_collector)
                vals += [0]*(8-len(vals))
                reg = 88+t0//8
                m.wait_reg(reg)
                m.advance(op=('ILOAD', reg, vals), tag='common_gather_RF_write')
            m.drain()
            m.phase = 'U_resident_MAC'
            for hg in range(groups):
                address = (k*rank+hg*8)*2
                m.coefficient(address)
                weights = np.frombuffer(m.cword, '<i2', 8, address % 32)
                if not np.any(weights):
                    continue
                for t in range(10):
                    src, lane = 88+t//8, t % 8
                    dst = t*groups+hg
                    if m.rf[src, lane] == 0:
                        m.advance(tag='ordinary_zero_bypass')
                    else:
                        m.wait_reg(dst)
                        m.advance(op=('IMAC_INDEX', dst, (address % 32, src, lane)),
                                  tag='ordinary_MAC')
        m.drain()
        for t in range(10):
            for hg in range(groups):
                dst = t*groups+hg
                completed_exponent = eu
                folded_scale = False
                if factor:
                    m.phase = 'factored_scale_and_base'
                    exponent = int(p['step'][t]).bit_length()-1
                    assert 1 << exponent == p['step'][t]
                    if not has_base:
                        # Strong ordinary compiler permission: cancel the
                        # exact power of two against the original U divisor.
                        # Reference accumulator is recorded in original units,
                        # but this multiplication is not executed by hardware.
                        completed_exponent = eu-exponent
                        folded_scale = True
                    else:
                        m.wait_reg(dst)
                        # Existing IRNE with negative shift is a signed48 shift.
                        m.advance(op=('IRNE', dst, -exponent), tag='dyadic_scale_shift48')
                        m.wait_reg(dst)
                        m.advance(op=('IADD', dst, 40+t*groups+hg), tag='pre_RNE_Uc_add')
                        m.wait_reg(dst)
                exact_acc[pos, t, hg*8:hg*8+8] = m.rf[dst].astype(np.int64)*(int(p['step'][t]) if folded_scale else 1)
                m.phase = 'original_U_RNE_sat_and_output'
                shared.integer.complete(m, dst, completed_exponent)
                m.store_i24(dst, OUTPUT+((pos*10+t)*rank+hg*8)*3)
    actual = shared.integer.read24(m, OUTPUT, (16, 10, rank))
    oracle = np.einsum('ptk,rk->ptr', x, u, dtype=np.int64)
    assert np.array_equal(exact_acc, oracle), mode
    assert np.array_equal(actual, rne_sat(oracle, eu)), mode
    assert sum(m.stages.values()) == m.time
    return dict(mode=mode, service_slots=m.time, cold_fill_and_cache_slots=setup,
                after_setup_slots=m.time-setup, stages=dict(m.stages), counts=dict(m.count),
                input_payload_bytes=len(raw), coefficient_payload_bytes=len(image),
                state_high_water=OUTPUT+int(actual.size)*3,
                live_RF_vectors=30+2+(30 if factor and has_base else 0),
                highest_RF_index=89, existing_common_staging_bytes=64,
                port_bytes=dict(SR64=m.count['SR64_reads']*8,
                                SW64=m.count['SW64_writes']*8,
                                CR256=m.count['CR256_reads']*32,
                                CW256=m.count['CW256_writes']*32),
                checked_accumulators=int(oracle.size), checked_U_values=int(actual.size),
                pre_RNE_differences=0, U_differences=0)


def correction_guard_check():
    # This adversarial check tests the mechanism, not an empirical hit rate.
    raw = np.asarray([[-(1 << 23)-1, (1 << 23), 3]], dtype=np.int64)
    u = np.asarray([[7, -3, 2], [-5, 4, 1]], dtype=np.int64)
    x = np.clip(raw, -(1 << 23), (1 << 23)-1)
    e = x-raw
    direct = u@x.T
    corrected = u@raw.T+u@e.T
    assert np.array_equal(direct, corrected)
    assert not np.array_equal(direct, u@raw.T)
    return dict(synthetic_values=3, corrected_accumulator_differences=0,
                omitted_correction_differences=int(np.count_nonzero(direct != u@raw.T)),
                empirical_interpretation='none; deliberate illegal-range guard test')


def main():
    report = dict(scope=__doc__, numeric_rows=[], execution_rows=[], range_proofs={},
                  correction_guard_check=correction_guard_check(),
                  resources=dict(RF_vectors=96, lanes=8, bits=48, state_bytes=131072,
                                 coefficients_bytes=131072, ROM_bytes=8192,
                                 SR64=1, SW64=1, CR256=1, issue=1),
                  timing_excludes=['producer encoding', 'projection gate generation',
                                   'V continuation', 'native projection', 'global BN', 'join'],
                  exact_parent='stage_20260912 old ordinary/lifting_raw R24 + onepass; not new matched320/source-two-term students',
                  new_training=False, new_AEE=False, GPU=False, EDA=False)
    for axis in ('ordinary', 'lifting_raw'):
        with np.load(EXPORT/axis/'000_zurich_city_09_a_0001.npz') as z:
            data = {k: z[k] for k in z.files}
        with np.load(EXPORT/axis/'deployed_constants.npz') as z:
            constants = {k: z[k] for k in z.files}
        with np.load(REP/'parameters'/(axis+'.npz')) as z:
            params = {m: {k: z[m+'_'+k] for k in ('D', 'c', 'step')} for m in MODES}
        u, v = [constants[k].astype(np.int64) for k in ('U_ped_q16', 'V_ped_q16')]
        bias = constants['PED_bias_q24'].astype(np.int64)
        eu, ev = [int(constants[k]) for k in ('U_ped_exponent', 'V_ped_exponent')]
        report['range_proofs'][axis] = {m: bounds(p, u) for m, p in params.items()}
        geometry = json.loads(str(data['window_geometry_json']))
        for label in ('corner', 'interior'):
            geo = geometry[label]
            y, x = geo['gate_origin']; oy, ox = geo['output_origin']
            dy, dx = 2*oy-y, 2*ox-x
            actual = data[label+'_updated_I24'][:, :, dy:dy+8:2, dx:dx+8:2].astype(np.int64)
            gates = data[label+'_proj_gate'][:, :, dy:dy+8:2, dx:dx+8:2].astype(np.int64)
            # Establish original U/V scale and bias against actual capture.
            a0 = np.einsum('rk,tkij->trij', u, actual, dtype=np.int64)
            l0 = rne_sat(a0, eu)
            p0 = np.einsum('hr,trij->thij', v, l0, dtype=np.int64)
            p0 = rne_sat(rne_sat(p0, ev)+bias[None, :, None, None], 0)
            assert np.array_equal(p0, data[label+'_continuous_q24'])
            for mode in MODES:
                row, estimate, code, latent = numeric_case(actual, gates, params[mode], u, v, bias, eu, ev)
                row.update(axis=axis, window=label, mode=mode, original_capture_differences=0)
                report['numeric_rows'].append(row)
                if not np.any(params[mode]['D']):
                    local = []
                    for method in ('expanded_I24', 'factored_code24', 'factored_code8'):
                        ex = execute_entry(estimate, code, params[mode], u, eu, method)
                        ex.update(axis=axis, window=label, quantizer=mode)
                        local.append(ex)
                        report['execution_rows'].append(ex)
                    base = local[0]['service_slots']
                    for ex in local:
                        ex['change_vs_free_ready_expanded_I24'] = ex['service_slots']/base-1
                    print(axis, label, mode, [(e['mode'], e['service_slots']) for e in local], flush=True)
                (HERE/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    report['complete'] = True
    report['totals'] = dict(numeric_cases=len(report['numeric_rows']),
                           timed_U_entry_cases=len(report['execution_rows']),
                           numeric_pre_RNE_values=sum(r['U_accumulator_values'] for r in report['numeric_rows']),
                           numeric_U_values=sum(r['U_values'] for r in report['numeric_rows']),
                           numeric_PED_values=sum(r['PED_values'] for r in report['numeric_rows']),
                           timed_accumulators=sum(r['checked_accumulators'] for r in report['execution_rows']),
                           timed_U_values=sum(r['checked_U_values'] for r in report['execution_rows']))
    (HERE/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['totals']), flush=True)


if __name__ == '__main__':
    main()
