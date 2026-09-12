"""Paid H8 Dg+c kernel. No host prediction, encoder, RNE, or saturation.

Constant layout: D.T contiguous little-endian signed32, c contiguous signed32.
Register contract: predictions RF40..49, packed T10 gate words RF50.
The temporary 16-byte collector reuses common_staging[0:16], within the
existing 64-byte staging allocation. No writes touch [16:64].
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np

PREDICTION_REGS = tuple(range(40, 50))
GATE_REG = 50
SIGNED48_LIMIT = 1 << 47


def pack_constants(D, c):
    """Return two unpadded payloads, 400 and 40 bytes; caller owns cold fill."""
    D, c = np.asarray(D, np.int64), np.asarray(c, np.int64)
    assert D.shape == (10, 10) and c.shape == (10,)
    assert np.all(D >= -(1 << 31)) and np.all(D < (1 << 31))
    assert np.all(c >= -(1 << 31)) and np.all(c < (1 << 31))
    return D.T.astype('<i4').tobytes(), c.astype('<i4').tobytes()


def bind_predictor(m, D_base, c_base, D, *, c=None):
    """Bind static nonzero instruction layout after caller's real coef fill.

    D supplies static coordinates only. It never supplies arithmetic operands
    to RF. The direct memory read here is an assertion against the filled
    parameter image, not part of the predictor datapath. Rebind if this region
    is overwritten. c is optional parameter-image validation only.
    """
    D = np.asarray(D, np.int64)
    assert D.shape == (10, 10)
    assert D_base % 32 == c_base % 32 == 0
    assert 0 <= D_base <= len(m.coef)-400 and 0 <= c_base <= len(m.coef)-40
    assert D_base+400 <= c_base or c_base+40 <= D_base
    assert np.all(D >= -(1 << 18)) and np.all(D < (1 << 18)), 'D exceeds signed19 interface'
    assert bytes(m.coef[D_base:D_base+400]) == D.T.astype('<i4').tobytes(), 'D plan/image mismatch'
    if c is not None:
        c = np.asarray(c, np.int64)
        assert c.shape == (10,)
        assert np.all(c >= -(1 << 31)) and np.all(c < (1 << 31))
        assert bytes(m.coef[c_base:c_base+40]) == c.astype('<i4').tobytes(), 'c image mismatch'
    columns = tuple(tuple(int(t) for t in np.flatnonzero(D[:, j])) for j in range(10))
    if not hasattr(m, '_predictor_layouts'):
        m._predictor_layouts = {}
    m._predictor_layouts[(int(D_base), int(c_base))] = columns
    return dict(nonzero_D=sum(map(len, columns)), active_columns=sum(bool(x) for x in columns),
                gate_input_required=any(columns), static_values_not_retained=True,
                D_layout='signed32[j,t], address=D_base+4*(10*j+t)',
                c_layout='signed32[t], address=c_base+4*t')


def _conditional_add(m, dst, args):
    gate_reg, bit, coefficient_address = map(int, args)
    assert dst in PREDICTION_REGS and gate_reg == GATE_REG
    assert 0 <= bit < 10 and coefficient_address % 4 == 0
    assert m.ready[dst] <= m.time and m.ready[gate_reg] <= m.time
    assert m.caddr == coefficient_address//32*32
    offset = coefficient_address % 32
    constant = int.from_bytes(m.cword[offset:offset+4], 'little', signed=True)
    assert -(1 << 18) <= constant < (1 << 18)
    words = m.rf[gate_reg].astype(np.int64)
    assert np.all(words >= 0) and np.all(words <= 1023)
    # Two RF operands: existing accumulator and gate vector. D is the actual
    # CR256 response's selected signed32 scalar; no coefficient RF is read.
    active = ((words >> bit) & 1) != 0
    value = m.rf[dst].astype(np.int64) + np.where(active, constant, 0)
    assert np.all(value >= -SIGNED48_LIMIT) and np.all(value < SIGNED48_LIMIT)
    m.count['predictor_conditional_RF_vector_reads'] += 2
    m.count['predictor_conditioned_active_lanes'] += int(active.sum())
    return value.astype(np.float64)


def install(machine_class):
    """Install this one op on a caller-owned Machine class; idempotent.

    Call after defining any caller integer_op override. Existing operations,
    advance(), arbitration, and two-slot integer writeback are unchanged.
    """
    if machine_class.__dict__.get('_paid_predictor_installed', False):
        return machine_class
    original = machine_class.integer_op

    def integer_op(self, kind, dst, args):
        if kind == 'IADD_GATE_CONSTANT':
            return _conditional_add(self, dst, args)
        return original(self, kind, dst, args)

    machine_class.integer_op = integer_op
    machine_class._paid_predictor_installed = True
    return machine_class


def _load_gate_h8(m, gate_address):
    assert isinstance(gate_address, (int, np.integer)) and gate_address % 8 == 0
    assert 0 <= gate_address <= len(m.state)-16
    assert len(m.common_staging) == 64, 'reuse the existing 64-byte staging'
    held = bytes(m.common_staging[16:64])
    # Copy only returned SR64 payloads. read_word supplies the original
    # request/response cost and the existing one-word cache permission.
    for offset in (0, 8):
        m.common_staging[offset:offset+8] = m.read_word(int(gate_address)+offset)
    m.advance(tag='predictor_gate_collect16')  # explicit bounded collector issue
    words = np.frombuffer(m.common_staging, '<u2', count=8).astype(np.int64)
    assert np.all(words <= 1023), 'input is an actual packed T10 gate word'
    m.wait_reg(GATE_REG)
    m.advance(op=('ILOAD', GATE_REG, words), tag='predictor_gate_RF_load')
    m.wait_reg(GATE_REG)
    assert bytes(m.common_staging[16:64]) == held
    m.count['predictor_gate_H8_collections'] += 1


def predict_h8(m, gate_address, D_base, c_base):
    """Return RF40..49 after exact Dg+c is ready for all 10x8 outputs.

    gate_address = pixel_gate_base + 2*h0, h0 in {0,8,...,88}; each pixel has
    96 little-endian uint16 words. The caller provides already-produced gates
    and paid coefficient cold fill. For a bound D=0 plan, gate_address may be
    None: no gate address, SRAM read, collector, gate load, or test is used.

    Only the static nonzero coordinates are read from the binding. Every c
    and D arithmetic value is selected from a real CR256 response. This
    function retains no values across calls and does not alter m.phase.
    """
    key = (int(D_base), int(c_base))
    assert hasattr(m, '_predictor_layouts') and key in m._predictor_layouts, 'bind filled case constants first'
    columns = m._predictor_layouts[key]
    # Initial c broadcasts also travel through CR256 and an existing ILOAD.
    # The scalar response selection/broadcast belongs to that charged issue.
    for t, dst in enumerate(PREDICTION_REGS):
        address = c_base+4*t
        m.coefficient(address)
        assert m.caddr == address//32*32
        offset = address % 32
        c = int.from_bytes(m.cword[offset:offset+4], 'little', signed=True)
        m.wait_reg(dst)
        m.advance(op=('ILOAD', dst, np.full(8, c, np.int64)), tag='predictor_c_response_broadcast')
    if any(columns):
        _load_gate_h8(m, gate_address)
        for j, outputs in enumerate(columns):
            if not outputs:
                continue
            m.wait_reg(GATE_REG)
            any_active = bool(np.any((m.rf[GATE_REG].astype(np.int64) >> j) & 1))
            m.advance(tag='predictor_gate_bit_test')  # one RF read and reduction
            m.count['predictor_gate_predicate_RF_vector_reads'] += 1
            if not any_active:
                m.count['predictor_empty_gate_columns'] += 1
                m.count['predictor_D_adds_bypassed_after_paid_test'] += len(outputs)
                continue
            for t in outputs:
                dst = PREDICTION_REGS[t]
                address = D_base+4*(10*j+t)
                m.coefficient(address)
                m.wait_reg(dst)
                issued = m.time
                m.advance(op=('IADD_GATE_CONSTANT', dst, (GATE_REG, j, address)),
                          tag='predictor_D_gate_conditional_add')
                # Existing machine may serialize a shared WB collision before
                # issue; its ready timestamp must still be issue+two slots.
                assert m.ready[dst] == m.time+1 and m.time >= issued+1
    for dst in PREDICTION_REGS:
        m.wait_reg(dst)
    m.count['predictor_H8_calls'] += 1
    return PREDICTION_REGS


def audit_integer_interface(D, c, step, u):
    """Static bounds only; no plan values are ever used to supply predictions."""
    D, c, step, u = (np.asarray(a, np.int64) for a in (D, c, step, u))
    assert D.shape == (10, 10) and c.shape == step.shape == (10,) and u.ndim == 2 and u.shape[1] == 96
    assert np.all(step > 0) and np.all((step & (step-1)) == 0)
    pred_low = c + np.minimum(D, 0).sum(axis=1)
    pred_high = c + np.maximum(D, 0).sum(axis=1)
    # Bounds every g in {0,1}^10, every signed8 residual, each source lane.
    raw_low, raw_high = pred_low-128*step, pred_high+127*step
    max_abs_u_row = int(np.abs(u).sum(axis=1).max())
    worst_decoded_latent = int(np.abs(D).sum(axis=1).max())*max_abs_u_row
    worst_residual = max_abs_u_row*128*int(step.max())
    worst_base = max_abs_u_row*int(np.abs(c).max())
    total = worst_decoded_latent+worst_residual+worst_base
    return dict(D_nonzero=int(np.count_nonzero(D)), D_min=int(D.min()), D_max=int(D.max()),
                c_min=int(c.min()), c_max=int(c.max()), step=step.tolist(),
                prediction_min=pred_low.tolist(), prediction_max=pred_high.tolist(),
                prediction_signed24_safe=bool(np.all(pred_low >= -(1<<23)) and np.all(pred_high < (1<<23))),
                all_code_reconstruction_signed24_safe=bool(np.all(raw_low >= -(1<<23)) and np.all(raw_high < (1<<23))),
                U_shape=list(u.shape), Ug_absolute_bound=max_abs_u_row,
                D_Ug_absolute_bound=worst_decoded_latent,
                D_Ug_signed48_safe=worst_decoded_latent < SIGNED48_LIMIT,
                complete_U_accumulator_abs_bound=total,
                complete_U_accumulator_signed48_safe=total < SIGNED48_LIMIT,
                latent_D_Ug_is_not_this_helper=True)


def smoke():
    """Two real parameter parents, directed gate words, no file artifacts."""
    import json
    sys.dont_write_bytecode = True
    here = Path(__file__).resolve().parent
    root_open = here.parents[1]
    sys.path.insert(0, str(root_open/'breadth_20260912/hardware'))
    import packed_weights as shared

    class SmokeMachine(shared.PackedMachine):
        pass
    install(SmokeMachine)
    D_base, c_base, gate_base = 65536, 65952, 4096
    words = np.asarray([[0]*8, [1023]*8, [1<<j for j in range(8)],
                        [256, 512, 768, 1, 2, 341, 682, 0]], dtype='<u2')
    rows = []
    for axis in ('ordinary', 'lifting_raw'):
        p = dict(np.load(root_open/'breadth_20260912/representation/parameters'/f'{axis}.npz'))
        with np.load(root_open/'stage_20260912/algorithm/hardware_exports'/axis/'deployed_constants.npz') as z:
            u = z['U_ped_q16']
        for mode in ('fixed_q8', 'affine_q8', 'full_g_q8'):
            D, c, step = [p[mode+'_'+k] for k in ('D', 'c', 'step')]
            m = SmokeMachine(False)
            m.phase = 'smoke_input_and_constant_cold_fill'
            dbytes, cbytes = pack_constants(D, c)
            m.dma_input(dbytes, D_base, True)
            m.dma_input(cbytes, c_base, True)
            m.dma_input(words.tobytes(), gate_base, False)
            binding = bind_predictor(m, D_base, c_base, D, c=c)
            m.phase = 'predictor_smoke'
            before, start = Counter(m.count), m.time
            outputs = []
            m.common_staging[16:64] = bytes(range(48))
            held = bytes(m.common_staging[16:64])
            for i, group in enumerate(words):
                regs = predict_h8(m, gate_base+16*i if np.any(D) else None, D_base, c_base)
                actual = m.rf[list(regs)].astype(np.int64)
                bits = ((group.astype(np.int64)[None, :] >> np.arange(10)[:, None]) & 1)
                expected = D.astype(np.int64) @ bits + c[:, None]
                assert np.array_equal(actual, expected)
                assert bytes(m.common_staging[16:64]) == held
                outputs.append(actual.copy())
            counts = dict(Counter(m.count)-before)
            if not np.any(D):
                assert counts.get('SR64_reads', 0) == counts.get('predictor_gate_RF_load', 0) == counts.get('predictor_gate_bit_test', 0) == 0
            else:
                assert counts.get('SR64_reads', 0) == 8
                assert counts['predictor_gate_collect16'] == counts['predictor_gate_RF_load'] == 4
            assert sum(m.stages.values()) == m.time
            rows.append(dict(axis=axis, mode=mode, binding=binding,
                             checked_values=int(np.asarray(outputs).size), differences=0,
                             post_fill_slots=m.time-start, counts=counts,
                             bounds=audit_integer_interface(D, c, step, u)))
    result = dict(scope='Only two real existing quantizer parents and four directed H8 gate groups per mode',
                  rows=rows, input_gate_production_not_timed=True, encoder_and_UV_not_timed=True,
                  new_function=False, GPU=False, EDA=False)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    smoke()
