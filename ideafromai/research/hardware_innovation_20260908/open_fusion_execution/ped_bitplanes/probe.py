"""Real PED_U payload probe on the existing finite 8-lane machine.

The only new primitive is an explicitly charged T10 bit-word decoder.
This is an exploratory CPU service model, not a LoAS/BitVert reproduction.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
FULL = BASE / 'algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40/schedule_compare_same_port/full_chain'
sys.path.insert(0, str(FULL / 'preview_sn2_chain'))
from machine import Machine
from integer_chain import pack24, read24, complete, dense

INPUT, OUTPUT = 0, 8192
T, K, H, P = 10, 96, 32, 2


class ProbeMachine(Machine):
    def integer_op(self, kind, dst, args):
        if kind == 'IMAC_INDEX':
            offset, src, lane = args
            assert self.ready[src] <= self.time
            value = self.rf[dst].astype(np.int64) + np.frombuffer(
                self.cword, '<i2', count=8, offset=offset).astype(np.int64) * int(self.rf[src, lane])
        elif kind in ('IWIDTH_T10', 'IBITWORD_T10'):
            ip, bit = args
            regs = (88 + 2 * ip, 89 + 2 * ip)
            assert all(self.ready[r] <= self.time for r in regs)
            x = np.concatenate([self.rf[regs[0]], self.rf[regs[1]][:2]]).astype(np.int64)
            if kind == 'IWIDTH_T10':
                width = max(1, max((int(v) if v >= 0 else ~int(v)).bit_length() + 1 for v in x))
                value = np.asarray([width, int(np.count_nonzero(x)), 0, 0, 0, 0, 0, 0], np.int64)
            else:
                word = sum(((int(v) >> bit) & 1) << t for t, v in enumerate(x))
                population = word.bit_count()
                default = int(population > 5)
                correction = word ^ (1023 if default else 0)
                value = np.asarray([word, population, default, correction, 0, 0, 0, 0], np.int64)
        elif kind == 'IBIT_COEF':
            offset, bit, sign = args
            weight = np.frombuffer(self.cword, '<i2', count=8, offset=offset).astype(np.int64)
            value = self.rf[dst].astype(np.int64) + weight * (sign * (1 << bit))
        else:
            return super().integer_op(kind, dst, args)
        assert np.all(value >= -(1 << 47)) and np.all(value < (1 << 47)), kind
        return np.asarray(value, np.float64)


def gather_k(m, k, input_base=INPUT, k_count=K, positions=P):
    # Four ordinary RF vectors hold all 20 real scalars. A bounded 24B
    # gather staging buffer and the existing 3B collector are common to
    # all new arms. Every byte is obtained through actual SR64 responses.
    for ip in range(positions):
        for t0 in (0, 8):
            values = []
            for t in range(t0, min(t0 + 8, T)):
                m.collect_i24(input_base + ((ip*T+t)*k_count+k)*3)
                values.append(m.scalar_collector)
            values += [0] * (8-len(values))
            r = 88 + 2*ip + t0//8
            m.wait_reg(r)
            m.advance(op=('ILOAD', r, values), tag='common_gather_RF_write')
    m.drain()


def resident_mac(m, base, name, k_count, h_count, input_base,
                 output_base, positions, shift, bias=None):
    """Drop-in dense continuation for a ProbeMachine, U32 or V96.

    Same calling fields as integer_chain.dense; this early variant needs
    IMAC_INDEX's packed RF scalar selector. Positions must be at most two.
    RF 0..79 = output accumulators, 88..91 = current k's T10 sources.
    Source last use is after all current H32 groups, before the next k.
    Wider H outputs use ordinary H32 tiling and re-read each source tile.
    """
    assert 1 <= positions <= 2
    start=m.time; m.phase='resident_'+name
    for h0 in range(0,h_count,32):
        groups=min(32,h_count-h0)//8
        for r in range(positions*T*groups):
            m.advance(op=('clear',r,None),tag='output_clear')
        for k in range(k_count):
            gather_k(m,k,input_base,k_count,positions)
            for hg in range(groups):
                address=base[name]+(k*h_count+h0+hg*8)*2
                m.coefficient(address)
                weights=np.frombuffer(m.cword,'<i2',count=8,offset=address%32)
                if not weights.any():
                    continue
                for ip in range(positions):
                    for t in range(T):
                        src,lane=88+2*ip+t//8,t%8
                        if m.rf[src,lane]==0:
                            m.advance(tag='ordinary_zero_bypass')
                            continue
                        dst=(ip*T+t)*groups+hg
                        m.wait_reg(dst)
                        m.advance(op=('IMAC_INDEX',dst,(address%32,src,lane)),tag='ordinary_MAC')
        m.drain()
        for ip in range(positions):
            for t in range(T):
                for hg in range(groups):
                    dst=(ip*T+t)*groups+hg
                    complete(m,dst,shift)
                    if bias is not None:
                        m.coefficient(base[bias]+(h0+hg*8)*4)
                        m.advance(op=('IADD_COEF',dst,None),tag=name+'_bias')
                        m.wait_reg(dst)
                        m.advance(op=('ISAT',dst,None),tag=name+'_bias_sat24')
                    m.store_i24(dst,output_base+((ip*T+t)*h_count+h0+hg*8)*3)
    m.mark(m.phase,start)


def execute(m, w, shift, mode):
    for r in range(P*T*4):
        m.advance(op=('clear', r, None), tag='output_clear')
    if mode == 'temporal_default':
        for r in range(80, 88):
            m.advance(op=('clear', r, None), tag='shared_base_clear')
    for k in range(K):
        gather_k(m, k)
        if mode == 'strong_mac':
            for hg in range(4):
                address = (k*H+hg*8)*2
                m.coefficient(address)
                weights = np.frombuffer(m.cword, '<i2', count=8, offset=address%32)
                if not weights.any():
                    continue
                for ip in range(P):
                    for t in range(T):
                        src, lane = 88 + 2*ip + t//8, t%8
                        # The stronger ordinary control gets scalar zero
                        # bypass after paid collection; no ideal zero index.
                        if m.rf[src, lane] == 0:
                            m.advance(tag='ordinary_zero_bypass')
                            continue
                        dst = (ip*T+t)*4+hg
                        m.wait_reg(dst)
                        m.advance(op=('IMAC_INDEX', dst, (address%32, src, lane)), tag='ordinary_MAC')
        else:
            for ip in range(P):
                m.advance(op=('IWIDTH_T10', 92, (ip, 0)), tag='signed_width_decode')
                m.wait_reg(92)
                width = int(m.rf[92, 0])
                m.count['effective_bitplanes'] += width
                for bit in range(width):
                    m.advance(op=('IBITWORD_T10', 92, (ip, bit)), tag='bit_transpose_popcount_decode')
                    m.wait_reg(92)
                    word, population, default, correction = map(int, m.rf[92, :4])
                    signed_bit = -1 if bit == width-1 else 1
                    if mode == 'plain_bitserial':
                        default, correction = 0, word
                    m.count['logical_default_terms'] += default
                    m.count['logical_correction_terms'] += correction.bit_count()
                    if not default and not correction:
                        m.advance(tag='empty_bitword_bypass')
                        continue
                    for hg in range(4):
                        address = (k*H+hg*8)*2
                        m.coefficient(address)
                        weights = np.frombuffer(m.cword, '<i2', count=8, offset=address%32)
                        if not weights.any():
                            continue
                        if default:
                            dst = 80+ip*4+hg
                            m.wait_reg(dst)
                            m.advance(op=('IBIT_COEF', dst, (address%32, bit, signed_bit)), tag='shared_default_shift_add')
                        for t in range(T):
                            if correction & (1 << t):
                                dst = (ip*T+t)*4+hg
                                m.wait_reg(dst)
                                sign = signed_bit * (-1 if default else 1)
                                m.advance(op=('IBIT_COEF', dst, (address%32, bit, sign)), tag='bit_correction_shift_add')
    m.drain()
    if mode == 'temporal_default':
        for ip in range(P):
            for t in range(T):
                for hg in range(4):
                    dst = (ip*T+t)*4+hg
                    m.advance(op=('IADD', dst, 80+ip*4+hg), tag='final_default_merge')
        m.drain()
    # Numerical reference reads the exact accumulator only for comparison.
    accum = m.rf[:80].astype(np.int64).reshape(P,T,H).copy()
    for ip in range(P):
        for t in range(T):
            for hg in range(4):
                dst = (ip*T+t)*4+hg
                complete(m, dst, shift)
                m.store_i24(dst, OUTPUT + ((ip*T+t)*H+hg*8)*3)
    return accum


def rne_sat(x, shift):
    if shift > 0:
        d = 1 << shift
        q, r = np.divmod(x, d)
        x = q + ((2*r > d) | ((2*r == d) & ((q&1) != 0)))
    else:
        x = x << -shift
    return np.clip(x, -(1 << 23), (1 << 23)-1)


def run_one(x, w, shift, mode, stress):
    m = ProbeMachine(stress)
    m.phase = 'cold_input_and_coefficients'
    m.dma_input(pack24(x), INPUT)
    m.dma_input(w.T.astype('<i2').tobytes(), 0, True)
    setup = m.time
    m.phase = mode
    if mode == 'existing_mac':
        m.forward_i24 = True
        dense(m, {'U_ped': 0}, 'U_ped', K, H, INPUT, OUTPUT, P, shift)
        accum = None
    else:
        accum = execute(m, w, shift, mode)
    reference_accum = np.einsum('ptk,hk->pth', x, w, dtype=np.int64)
    expected = rne_sat(reference_accum, shift)
    actual = read24(m, OUTPUT, (P,T,H))
    assert np.array_equal(actual, expected), (mode, int(np.count_nonzero(actual != expected)))
    if accum is not None:
        assert np.array_equal(accum, reference_accum), mode
    m.phase = 'common_output_egress'
    for off in range(0, P*T*H*3, 32):
        payload = b''.join(m.read_word(OUTPUT+off+j) for j in (0,8,16,24))
        assert len(payload) == 32
        for _ in range(5):
            m.advance(tag='DMA_output_slots')
    return dict(service_slots=m.time, cold_fill_slots=setup,
                phases=dict(m.stages), counts=dict(m.count),
                input_bytes=P*T*K*3, coefficient_bytes=H*K*2,
                output_bytes=P*T*H*3, state_high_water_bytes=OUTPUT+P*T*H*3,
                RF_peak_vectors=93 if 'bit' in mode or mode=='temporal_default' else (92 if mode=='strong_mac' else 86),
                value_differences=int(np.count_nonzero(actual != expected)),
                checked_values=actual.size,
                exact_dot_differences=0 if accum is not None else None,
                integer_shift=shift, timeline=m.timeline)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stress', action='store_true')
    args = parser.parse_args()
    result = dict(scope='Two fixed original P2 anchor tiles; one frame, both saved students; PED_U only.',
                  evidence='CPU payload-executing issue/port model; no RTL, frequency, power, area, AEE or full-chain claim.',
                  new_training=False, new_quantization=False, stress=args.stress,
                  resource=dict(lanes=8, accumulator_bits=48, RF_vectors=96, RF_vector_lanes=8,
                                integer_latency=2, shared_issue_per_slot=1,
                                SR_bytes_per_slot=8, SW_bytes_per_slot=8, coefficient_read_bytes_per_slot=32,
                                state_bytes=131072, coefficient_bytes=131072,
                                extra_control_hypothesis='One charged T10 bit-word+popcount decode from two RF reads; no PPA established.'),
                  axes={})
    for axis in ('ordinary', 'lifting_raw'):
        path = FULL/'capture_full_producers'/axis
        with np.load(path/'parameters.npz') as q:
            w=q['U_ped_q16'].astype(np.int64); shift=int(q['U_ped_exponent'])
        with np.load(path/'000_zurich_city_09_a_0001.npz') as d:
            geometry=json.loads(str(d['window_geometry_json']))
            full=d['full_updated_I24']
            tiles={}
            for label in ('corner','interior'):
                y,x=geometry[label]['output_origin']; positions=[(2*y,2*x),(2*y,2*x+2)]
                tiles[label]=(positions,np.stack([full[:,:,yy,xx] for yy,xx in positions]).astype(np.int64))
            del full
        result['axes'][axis]={}
        for label,(positions,x) in tiles.items():
            rows={}
            for mode in ('existing_mac','strong_mac','plain_bitserial','temporal_default'):
                rows[mode]=run_one(x,w,shift,mode,args.stress)
                print(axis,label,mode,rows[mode]['service_slots'],'PASS',flush=True)
            ref=rows['strong_mac']['service_slots']
            for row in rows.values():
                row['service_change_vs_strong_mac_pct']=(row['service_slots']/ref-1)*100
            result['axes'][axis][label]=dict(positions=positions,actual_input_range=[int(x.min()),int(x.max())],arms=rows)
    result['verdict']='PASS_FUNCTION; inspect performance versus strong_mac; negative result stops this serialized bitplane layout only.'
    (HERE/('results_stress.json' if args.stress else 'results.json')).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
