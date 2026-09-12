"""Actual I24/gate -> Q8 or reconstructed I24 -> retained U -> V -> PED.

No quantizer code or decoded source enters this kernel from the oracle.
All new integer operations use the common single issue/two-slot RF model.
"""
import numpy as np
import binding
import predictor

shared, integer = binding.shared, binding.integer
CODE, RECON = 114688, 110592
D_BASE, C_BASE, UC_BASE = 65536, 65952, 66016


class QuantMachine(shared.PackedMachine):
    def integer_op(self, kind, dst, args):
        if kind == 'ISUB_REG':
            assert self.ready[args] <= self.time
            value = self.rf[dst].astype(np.int64)-self.rf[args].astype(np.int64)
        elif kind == 'ICLIP8':
            x = self.rf[dst].astype(np.int64)
            self.count['quantizer_clipped_lanes'] += int(np.count_nonzero((x < -128) | (x > 127)))
            value = np.clip(x, -128, 127)
        elif kind == 'IGATE_TICK_VECTOR':
            src, lane, start = args
            assert self.ready[src] <= self.time
            word = int(self.rf[src, lane])
            value = np.asarray([(word >> t) & 1 if t < 10 else 0 for t in range(start, start+8)], np.int64)
        else:
            return super().integer_op(kind, dst, args)
        assert np.all(value >= -(1 << 47)) and np.all(value < (1 << 47)), kind
        return value.astype(np.float64)

    def cache48(self, reg, address):
        payload = bytearray()
        remaining = 48
        while remaining:
            self.coefficient(address)
            take = min(remaining, 32-address % 32)
            payload += self.cword[address % 32:address % 32+take]
            remaining -= take; address += take
        values = [int.from_bytes(payload[j:j+6], 'little', signed=True) for j in range(0, 48, 6)]
        self.wait_reg(reg)
        self.advance(op=('ILOAD', reg, values), tag='Uc48_response_RF_load')
        self.wait_reg(reg)


predictor.install(QuantMachine)


def pack48(values):
    return b''.join(int(v).to_bytes(6, 'little', signed=True) for v in np.asarray(values).flat)


def load_constants(m, p, u, encoded, cse_plan=None):
    if getattr(m, 'quantizer_constants_loaded', False):
        return
    m.phase = 'quantizer_constant_cold_fill'
    db, cb = predictor.pack_constants(p['D'], p['c'])
    image = bytearray()
    if np.any(p['D']):
        image += db
    if np.any(p['c']) or np.any(p['D']):
        image += bytes(C_BASE-D_BASE-len(image)); image += cb
    uc = p['c'][:, None]*u.sum(1)[None, :]
    if encoded and np.any(uc):
        image += bytes(UC_BASE-D_BASE-len(image)); image += pack48(uc)
    if image:
        m.dma_input(bytes(image), D_BASE, True)
    if np.any(p['D']) or np.any(p['c']):
        predictor.bind_predictor(m, D_BASE, C_BASE, p['D'], c=p['c'])
    if cse_plan is not None:
        import latent_cse
        m.phase = 'latent_CSE_control_cold_fill'
        m.dma_input(latent_cse.control_image(cse_plan), 73728, True)
        latent_cse.bind_control(m, cse_plan, 73728, spill_base=98304)
    m.quantizer_constants_loaded = True


def encode(m, input_base, gate_address, p, encoded):
    """Store actual code8 or same-function reconstructed I24 for one pixel."""
    start = m.time
    has_prediction = bool(np.any(p['D']) or np.any(p['c']))
    for h in range(0, 96, 8):
        m.phase = 'quantizer_predictor'
        if has_prediction:
            predictor.predict_h8(m, gate_address+2*h, D_BASE, C_BASE)
        for t in range(10):
            m.phase = 'quantizer_actual_I24_read'
            m.load_i24(t, input_base+(t*96+h)*3)
            m.phase = 'quantizer_RNE_clip8'
            if has_prediction:
                m.wait_reg(40+t)
                m.advance(op=('ISUB_REG', t, 40+t), tag='quantizer_predictor_subtract')
                m.wait_reg(t)
            exponent = int(p['step'][t]).bit_length()-1
            assert 1 << exponent == int(p['step'][t])
            m.advance(op=('IRNE', t, exponent), tag='quantizer_RNE_to_code')
            m.wait_reg(t)
            m.advance(op=('ICLIP8', t, None), tag='quantizer_clip8')
            m.wait_reg(t)
            if encoded:
                m.phase = 'quantizer_code8_materialize'
                payload = m.rf[t].astype('i1').tobytes()
                m.advance(write=(CODE+t*96+h, payload), tag='Q8_vector_store')
            else:
                m.phase = 'quantizer_reconstruct_I24'
                m.advance(op=('IRNE', t, -exponent), tag='quantizer_reconstruct_scale')
                m.wait_reg(t)
                if has_prediction:
                    m.advance(op=('IADD', t, 40+t), tag='quantizer_reconstruct_base')
                    m.wait_reg(t)
                m.advance(op=('ISAT', t, None), tag='quantizer_reconstruct_sat24')
                m.wait_reg(t)
                m.store_i24(t, RECON+(t*96+h)*3)
    return m.time-start


def gather_h8(m, input_base, k, width):
    """Common ordinary control: hold all T10 rows of one channel H8 in RF."""
    assert k % 8 == 0
    for t in range(10):
        address = input_base+(t*96+k)*width
        reg = 60+t
        m.wait_reg(reg)
        if width == 3:
            m.load_i24(reg, address)
        else:
            word = m.read_word(address)
            values = np.frombuffer(word, 'i1').astype(np.int64)
            m.advance(op=('ILOAD', reg, values), tag='common_Q8_H8_source_RF_load')
            m.wait_reg(reg)
    m.drain()


def gate_for_k(m, gate_base, k):
    if k % 4 == 0:
        word = m.read_word(gate_base+2*k)
        vals = list(np.frombuffer(word, '<u2').astype(np.int64))+[0]*4
        m.wait_reg(95)
        m.advance(op=('ILOAD', 95, vals), tag='Ug_H4_gate_word_load')
        m.wait_reg(95)
    m.advance(tag='Ug_source_gate_test')
    live = bool(int(m.rf[95, k % 4]) & 1023)
    if live:
        for t0 in (0, 8):
            reg = 92+t0//8
            m.wait_reg(reg)
            m.advance(op=('IGATE_TICK_VECTOR', reg, (95, k % 4, t0)), tag='Ug_T10_unpack')
        m.drain()
    return live


def u_stage(m, base, q, p, encoded, input_base, gate_address, cse_plan):
    full = encoded and bool(np.any(p['D']))
    m.phase = 'retained_U_clear'
    for r in range(60 if full else 30):
        m.wait_reg(r); m.advance(op=('clear', r, None), tag='Uq_Ug_clear')
    for k in range(96):
        m.phase = 'U_actual_source_gather'
        if k % 8 == 0:
            gather_h8(m, input_base, k, 1 if encoded else 3)
        live_gate = gate_for_k(m, gate_address, k) if full else False
        m.phase = 'Uq_and_Ug_weight_supply'
        for hg in range(3):
            address = base['U_ped']+(k*24+hg*8)*2
            m.coefficient(address)
            if not np.frombuffer(m.cword, '<i2', 8, address % 32).any():
                continue
            for t in range(10):
                dst = t*3+hg; src, lane = 60+t, k % 8
                if m.rf[src, lane] == 0:
                    m.advance(tag='ordinary_zero_bypass')
                else:
                    m.wait_reg(dst)
                    m.advance(op=('IMAC_INDEX', dst, (address % 32, src, lane)), tag='U_resident_MAC')
            if live_gate:
                for t in range(10):
                    m.advance(tag='Ug_tick_select')
                    if m.rf[92+t//8, t % 8] == 0:
                        continue
                    dst = 30+t*3+hg; m.wait_reg(dst)
                    m.advance(op=('IAAC', dst, (address % 32, None)), tag='Ug_active_AAC')
    m.drain()
    has_uc = encoded and bool(np.any(p['c'][:, None]*q['U_ped_q16'].sum(1)[None, :]))
    folded = encoded and not has_uc and not full
    # Finish all Uc accesses contiguously, preserving adjacent CR-word reuse.
    for t in range(10):
        for hg in range(3):
            dst = t*3+hg
            if encoded and not folded:
                m.phase = 'U_scale_and_Uc_before_original_RNE'
                exponent = int(p['step'][t]).bit_length()-1
                m.wait_reg(dst)
                m.advance(op=('IRNE', dst, -exponent), tag='U_residual_dyadic_scale')
                m.wait_reg(dst)
                if has_uc:
                    m.cache48(94, UC_BASE+(t*24+hg*8)*6)
                    m.advance(op=('IADD', dst, 94), tag='U_pre_RNE_Uc_add')
                    m.wait_reg(dst)
    if full:
        import latent_cse
        m.phase = 'U_full_D_existing_shift_add_CSE'
        for hg in range(3):
            latent_cse.execute_latent_cse(m, cse_plan, hg)
    m.phase = 'original_U_RNE_sat'
    for t in range(10):
        exponent = int(q['U_ped_exponent'])
        if folded:
            exponent -= int(p['step'][t]).bit_length()-1
        for hg in range(3):
            integer.complete(m, t*3+hg, exponent)
    return m.rf[:30].astype(np.int64).copy().reshape(10, 24)


def v_stage(m, base, q, output_base):
    for h0 in (0, 48):
        m.phase = 'retained_P1_H48_V'
        for r in range(30, 90):
            m.wait_reg(r); m.advance(op=('clear', r, None), tag='V_output_clear')
        for k in range(24):
            for hg in range(6):
                address = base['V_ped']+(k*96+h0+hg*8)*2
                m.coefficient(address)
                if not np.frombuffer(m.cword, '<i2', 8, address % 32).any():
                    continue
                for t in range(10):
                    src, lane = t*3+k//8, k % 8
                    if m.rf[src, lane] == 0:
                        m.advance(tag='ordinary_zero_bypass'); continue
                    dst = 30+t*6+hg; m.wait_reg(dst)
                    m.advance(op=('IMAC_INDEX', dst, (address % 32, src, lane)), tag='V_resident_MAC')
        m.drain()
        for t in range(10):
            for hg in range(6):
                dst = 30+t*6+hg
                integer.complete(m, dst, int(q['V_ped_exponent']))
                m.coefficient(base['PED_bias']+(h0+hg*8)*4)
                m.advance(op=('IADD_COEF', dst, None), tag='V_original_bias')
                m.wait_reg(dst)
                m.advance(op=('ISAT', dst, None), tag='V_original_bias_sat24')
                m.store_i24(dst, output_base+((t*96+h0+hg*8)*3))


def make_callback(p, encoded, cse_plan=None):
    def callback(m, base, q, positions, geo):
        if p is not None:
            load_constants(m, p, q['U_ped_q16'].astype(np.int64), encoded, cse_plan)
        us, peds = [], []
        for ip, (y, x) in enumerate(positions):
            input_base = binding.integer.UPDATED+ip*10*96*3
            gate_address = binding.integer.PROJ+((y-geo['gate_origin'][0])*geo['gate_shape'][1]+x-geo['gate_origin'][1])*192
            if p is not None:
                # External observer only; these copies never feed encode/U/V.
                observed_x = integer.read24(m, input_base, (10, 96))
                observed_words = np.frombuffer(m.state, '<u2', count=96, offset=gate_address).astype(np.int64)
                encode(m, input_base, gate_address, p, encoded)
                from quantizers import reconstruct
                observed_g = np.stack([(observed_words >> t) & 1 for t in range(10)])
                expected_i24, expected_code, _ = reconstruct(observed_x, observed_g, p)
                if encoded:
                    actual_repr = np.frombuffer(m.state, 'i1', count=960, offset=CODE).reshape(10, 96)
                    assert np.array_equal(actual_repr, expected_code)
                else:
                    actual_repr = integer.read24(m, RECON, (10, 96))
                    assert np.array_equal(actual_repr, expected_i24)
                m.count['observed_quantizer_values_compared'] += 960
                source = CODE if encoded else RECON
            else:
                source = input_base
            actual_u = u_stage(m, base, q, p, encoded, source, gate_address, cse_plan)
            m.encoded_u_ready = m.time
            us.append(actual_u)
            out = binding.integer.PED_V+ip*10*96*3
            v_stage(m, base, q, out)
            peds.append(integer.read24(m, out, (10, 96)))
        return np.asarray(us), np.asarray(peds)
    return callback
