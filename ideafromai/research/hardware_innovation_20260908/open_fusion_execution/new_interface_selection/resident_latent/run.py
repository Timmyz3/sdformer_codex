"""Paid P1 RF-resident latent U/V on the existing ProbeMachine.

Compare each rank with its P2 source-resident/materialized-Z baseline and
grant R32 the same legal P1 retention privilege. Not a new scalar read port,
not an AEE trial, and not evidence that R24 capacity alone is novel.
"""
from pathlib import Path
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from rebase_service import ped, FULL
from rebase_probe import execute

INPUT, LATENT, OUTPUT = 0, 8192, 16384
T = 10


class CountMachine(ped.ProbeMachine):
    def __init__(self, stress, rank):
        super().__init__(stress)
        self.rank = rank
        self.u_bytes = 96*rank*2
        self.v_end = 2*self.u_bytes

    def advance(self, read=None, write=None, coef=None, cwrite=None, op=None, tag=''):
        # Counting only; all real operand data, latency, port readiness and
        # writeback conflicts still execute through the untouched Machine.
        if read is not None and LATENT <= read < LATENT+20*self.rank*3:
            self.count['latent_SR64_reads'] += 1
        if write is not None and LATENT <= write[0] < LATENT+20*self.rank*3:
            self.count['latent_SW64_writes'] += 1
        if coef is not None:
            self.count['U_CR256_reads' if coef < self.u_bytes else
                       'V_CR256_reads' if coef < self.v_end else 'bias_CR256_reads'] += 1
        if tag == 'ordinary_zero_bypass':
            self.count['paid_RF_scalar_selection_zero_slots'] += 1
        super().advance(read=read, write=write, coef=coef, cwrite=cwrite, op=op, tag=tag)

    def integer_op(self, kind, dst, args):
        if kind == 'IMAC_INDEX':
            self.count['paid_RF_scalar_selection_MAC_slots'] += 1
            self.count['MAC_source_vector_RF_reads'] += 1
            self.count['MAC_accumulator_vector_RF_reads'] += 1
        return super().integer_op(kind, dst, args)


def keep_u(m, base, rank, ip):
    groups = rank//8
    start = m.time
    m.phase = 'resident_Z_U_P1'
    for r in range(T*groups):
        m.wait_reg(r)
        m.advance(op=('clear', r, None), tag='U_output_clear')
    for k in range(96):
        # Existing paid gather into88/89; these source registers are dead
        # before V reuses those RF numbers as part of the H48 output tile.
        ped.gather_k(m, k, INPUT+ip*T*96*3, 96, 1)
        for hg in range(groups):
            address = base['U_ped']+(k*rank+hg*8)*2
            m.coefficient(address)
            if not np.frombuffer(m.cword, '<i2', count=8, offset=address%32).any():
                continue
            for t in range(T):
                src, lane = 88+t//8, t%8
                if m.rf[src, lane] == 0:
                    m.advance(tag='ordinary_zero_bypass')
                    continue
                dst = t*groups+hg
                m.wait_reg(dst)
                m.advance(op=('IMAC_INDEX', dst, (address%32, src, lane)), tag='ordinary_MAC')
    m.drain()
    m.phase = 'resident_Z_original_U_RNE_sat'
    for t in range(T):
        for hg in range(groups):
            ped.complete(m, t*groups+hg, 16)
    # Verification only; later V operands are read from this actual RF.
    z = m.rf[:T*groups].astype(np.int64).reshape(T, rank).copy()
    m.mark('resident_Z_U_P1_and_original_RNE', start)
    return z


def consume_z(m, base, rank, ip, hblock):
    zgroups = rank//8
    ybase = T*zgroups
    assert ybase+T*(hblock//8)+6 <= 96
    assert ybase+T*(hblock//8) <= 90
    start = m.time
    for h0 in range(0, 96, hblock):
        groups = min(hblock, 96-h0)//8
        begin = m.time
        m.phase = 'resident_Z_V_P1'
        for r in range(ybase, ybase+T*groups):
            m.wait_reg(r)
            m.advance(op=('clear', r, None), tag='V_output_clear')
        for k in range(rank):
            for hg in range(groups):
                address = base['V_ped']+(k*96+h0+hg*8)*2
                m.coefficient(address)
                if not np.frombuffer(m.cword, '<i2', count=8, offset=address%32).any():
                    continue
                for t in range(T):
                    src, lane = t*zgroups+k//8, k%8
                    m.wait_reg(src)
                    if m.rf[src, lane] == 0:
                        m.advance(tag='ordinary_zero_bypass')
                        continue
                    dst = ybase+t*groups+hg
                    m.wait_reg(dst)
                    # One existing indexed MAC reads one Z vector and one
                    # accumulator vector. No free RF-to-scalar staging call.
                    m.advance(op=('IMAC_INDEX', dst, (address%32, src, lane)), tag='ordinary_MAC')
        m.drain()
        m.phase = 'resident_Z_V_RNE_bias_store'
        for t in range(T):
            for hg in range(groups):
                dst = ybase+t*groups+hg
                ped.complete(m, dst, 15)
                m.coefficient(base['PED_bias']+(h0+hg*8)*4)
                m.advance(op=('IADD_COEF', dst, None), tag='V_ped_bias')
                m.wait_reg(dst)
                m.advance(op=('ISAT', dst, None), tag='V_ped_bias_sat24')
                m.store_i24(dst, OUTPUT+((ip*T+t)*96+h0+hg*8)*3)
        m.mark(f'resident_Z_V_P1_H{h0}_N{groups*8}', begin)
    m.mark('resident_Z_V_P1_complete', start)


def run_case(x, u, v, bias, rank, mode, stress):
    m = CountMachine(stress, rank)
    packed = u.T.astype('<i2').tobytes()+v.T.astype('<i2').tobytes()+bias.astype('<i4').tobytes()
    base = {'U_ped': 0, 'V_ped': u.size*2, 'PED_bias': (u.size+v.size)*2}
    m.phase = 'common_input_and_coefficient_fill'
    m.dma_input(ped.pack24(x), INPUT)
    m.dma_input(packed, 0, True)
    cold = m.time
    z_expected = ped.rne_sat(np.einsum('ptk,hk->pth', x, u, dtype=np.int64), 16)
    z_differences = 0
    if mode == 'P2_materialized_Z':
        ped.resident_mac(m, base, 'U_ped', 96, rank, INPUT, LATENT, 2, 16)
        actual_z = ped.read24(m, LATENT, (2, T, rank))
        z_differences = int(np.count_nonzero(actual_z != z_expected))
        ped.resident_mac(m, base, 'V_ped', rank, 96, LATENT, OUTPUT, 2, 15, bias='PED_bias')
        latent_vectors, y_vectors, hblock = 0, 80, 32
    else:
        hblock = 48 if rank == 24 else 32
        for ip in range(2):
            z = keep_u(m, base, rank, ip)
            z_differences += int(np.count_nonzero(z != z_expected[ip]))
            consume_z(m, base, rank, ip, hblock)
        latent_vectors, y_vectors = T*(rank//8), T*(hblock//8)
    actual = ped.read24(m, OUTPUT, (2,T,96))
    ref, bounds = execute(u, v, x.transpose(2,0,1).reshape(96,-1), bias)
    expected = ref.reshape(96,2,T).transpose(1,2,0)
    assert z_differences == 0 and np.array_equal(actual, expected)
    m.phase = 'common_output_egress'
    for off in range(0, 5760, 32):
        payload = b''.join(m.read_word(OUTPUT+off+j) for j in (0,8,16,24))
        assert len(payload) == 32
        for _ in range(5):
            m.advance(tag='DMA_output_slots')
    return actual, dict(mode=mode, rank=rank, stress=stress, service_slots=m.time,
        cold_fill_slots=cold, body_and_output_slots=m.time-cold,
        phases=dict(m.stages), counts=dict(m.count), timeline=m.timeline,
        input_bytes=x.size*3, coefficient_bytes=len(packed), output_bytes=5760,
        output_values=actual.size, output_differences=0, latent_values=z_expected.size,
        original_U_RNE_latent_differences=z_differences, arithmetic_bounds=bounds,
        scalar_selection='Existing paid IMAC_INDEX per MAC; one Z/source RF vector read plus one accumulator vector read; no extra read port. Zero tests retain a paid bypass slot.',
        state=dict(RF_capacity_vectors=96, state_capacity_bytes=131072, coefficient_capacity_bytes=131072,
            latent_resident_vectors=latent_vectors, V_output_vectors=y_vectors,
            U_accumulator_vectors=(2 if mode == 'P2_materialized_Z' else 1)*T*(rank//8),
            peak_Z_plus_Y_vectors=latent_vectors+y_vectors,
            gather_source_vectors=4 if mode == 'P2_materialized_Z' else 2,
            extra_new_RF_vectors=0, max_MAC_RF_vector_reads_per_issue=2,
            latent_materialized_bytes=2*T*rank*3 if mode == 'P2_materialized_Z' else 0,
            latent_physical_SR_bytes=8*m.count['latent_SR64_reads'],
            latent_physical_SW_bytes=8*m.count['latent_SW64_writes'],
            state_high_water_bytes=OUTPUT+5760,
            V_output_block_H=hblock, V_blocks_per_P1=(96+hblock-1)//hblock,
            coefficient_cr_repeated_for_P1=mode != 'P2_materialized_Z',
            RF_lifetime='P1 U gather88/89 is dead before V clear may overwrite88/89; Z prefix0..29/39 retained through every V block, outputs stored before next P1 reuses registers.'))


def main():
    result = dict(scope=__doc__, evidence='CPU actual-payload finite issue/port model; not AEE, RTL/PPA or full-chain/network service.',
        fixed_positions=[[120,160],[120,162]], location='original interior P2, same two real positions in each student',
        comparison='Same-rank P2 resident-MAC vs P1 RF-retained Z; both R24 and R32 get the same representation, coefficient cache, indexed-MAC and blocking privileges.',
        rank24_identity='Saved original_ordered_U[:24], original_ordered_V[:,:24]; no new SVD/whitening/training.',
        R32_H32_reason='H32 and H40 both require three blocks; H32 avoids the coefficient word split at the H40 boundary and uses40Z+40Y RF.',
        profile='ready plus the original fixed SR-last8/SW-last4 of32 pressure pattern',
        axes={})
    for axis in ('ordinary', 'lifting_raw'):
        with np.load(FULL/'capture'/axis/'parameters.npz') as z:
            q = {k:z[k] for k in z.files}
        with np.load(HERE.parent/(axis+'_rebase_parameters.npz')) as z:
            a = {k:z[k] for k in z.files}
        with np.load(FULL/'capture_full_producers'/axis/'000_zurich_city_09_a_0001.npz') as z:
            full = z['full_updated_I24']
            x = np.stack([full[:,:,120,160], full[:,:,120,162]]).astype(np.int64)
            del full
            original_gold = z['full_continuous_q24'][:,:,60,80:82].transpose(2,0,1).copy()
        rows = []
        for stress in (False, True):
            for rank, u, v in [(24, a['original_ordered_U'][:24], a['original_ordered_V'][:,:24]),
                               (32, q['U_ped_q16'], q['V_ped_q16'])]:
                for mode in ('P2_materialized_Z', 'P1_resident_Z'):
                    values, report = run_case(x, u, v, q['PED_bias_q24'], rank, mode, stress)
                    if rank == 32:
                        report['original_capture_differences'] = int(np.count_nonzero(values != original_gold))
                        assert report['original_capture_differences'] == 0
                    baseline = report if mode == 'P2_materialized_Z' else rows[-1]
                    report['same_rank_service_reduction'] = 1-report['service_slots']/baseline['service_slots']
                    report['same_rank_body_reduction'] = 1-report['body_and_output_slots']/baseline['body_and_output_slots']
                    rows.append(report)
                    print(axis, stress, rank, mode, report['service_slots'], report['same_rank_service_reduction'], 'PAYLOAD_PASS', flush=True)
        result['axes'][axis] = dict(input_range=[int(x.min()),int(x.max())], rows=rows)
        (HERE/'results.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
