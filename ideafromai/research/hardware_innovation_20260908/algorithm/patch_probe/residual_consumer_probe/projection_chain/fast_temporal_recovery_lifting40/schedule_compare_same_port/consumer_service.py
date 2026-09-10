"""Finite raw integer consumer slice, actual sn2 source ready -> two PED consumers.

Top-left 4x4 continuous PED outputs plus all 8x8 proj gate words. Full K864,
real padding and retained I24 input reads. FP preview and native proj.conv
are outside this endpoint. Host NumPy arrays are only the numerical oracle.
"""
from pathlib import Path
from collections import Counter
import argparse
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent
T, C, SIDE = 10, 96, 8
SOURCE_BASE, GATE_BASE, I_BASE, WORK_BASE = 0, 12288, 24576, 30336
LIMIT = 1 << 23


def read_npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def unpack(data, label):
    if label + '_gate_tile_bits' in data:
        shape = tuple(map(int, data[label + '_tile_shape']))
        return np.unpackbits(data[label + '_gate_tile_bits'], axis=1, bitorder='little',
                             count=math.prod(shape[1:])).reshape(shape).astype(bool)
    shape = tuple(map(int, data[label + '_shape']))
    return np.unpackbits(data[label + '_gate_bits'], axis=1, bitorder='little',
                         count=math.prod(shape[1:])).reshape(shape).astype(bool)


def write_fixture(path, data, q, gold):
    # Debug packing is reduced, while original full-frame shape/address metadata
    # remains unchanged. The service model still supplies every modeled byte.
    content = dict(q)
    for key in ('frame_name', 'sn2_shape', 'proj_shape', 'sn2_theta'):
        content[key] = data[key]
    content['I24_halo'] = data['I24_halo'][:, :, :SIDE, :SIDE]
    for label in ('sn2', 'proj'):
        tile = unpack(data, label)[:, :, :SIDE, :SIDE]
        content[label+'_tile_shape'] = np.array(tile.shape)
        content[label+'_gate_tile_bits'] = np.packbits(tile.reshape(T, -1), axis=1, bitorder='little')
    content.update(gold)
    content['fixture_scope'] = np.array('Corner8x8 gates and I24, continuous4x4, original full-frame address metadata; no change of service layout.')
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, **content)


def rne24(values, shift):
    values = np.asarray(values, np.int64)
    if shift > 0:
        d = 1 << int(shift)
        q, r = np.divmod(values, d)
        values = q + ((2*r > d) | ((2*r == d) & ((q & 1) != 0)))
    elif shift < 0:
        values = values << -int(shift)
    return np.clip(values, -LIMIT, LIMIT-1)


def patch(gates, y, x):
    result = np.zeros((864, T), np.int64)
    for c in range(C):
        for kh in range(3):
            for kw in range(3):
                sy, sx = y+kh-1, x+kw-1
                if 0 <= sy < gates.shape[2] and 0 <= sx < gates.shape[3]:
                    result[(c*3+kh)*3+kw] = gates[:, c, sy, sx]
    return result


def numerical_reference(data, q, gates, continuous_gold=None):
    # These full arrays are the CPU checker, not extra modeled physical state.
    identity = data['I24_halo'][:, :, :SIDE, :SIDE].astype(np.int64)
    updated = identity.copy()
    output = np.empty((T, C, 4, 4), np.int64)
    extrema = Counter()
    source_terms, source_empty = 0, 0
    for y in range(0, SIDE, 2):
        for x in range(0, SIDE, 2):
            g = patch(gates, y, x)
            source_terms += int((np.count_nonzero(q['U_conv2_theta_q16'], axis=0)[:, None]*g).sum())
            source_empty += int(np.count_nonzero(~g.any(axis=0)))
            accum = q['U_conv2_theta_q16'].astype(np.int64) @ g
            extrema['U16_acc_maxabs'] = max(extrema['U16_acc_maxabs'], int(np.abs(accum).max()))
            z = rne24(accum, int(q['U_conv2_theta_exponent'])-14)
            accum = q['F_q16'].astype(np.int64) @ z
            extrema['F_acc_maxabs'] = max(extrema['F_acc_maxabs'], int(np.abs(accum).max()))
            branch = rne24(accum, int(q['F_exponent']))
            merged = identity[:, :, y, x].T + branch + q['BN2_constant_q24'][:, None]
            updated[:, :, y, x] = rne24(merged, 0).T
            accum = q['U_ped_q16'].astype(np.int64) @ updated[:, :, y, x].T
            extrema['PED_U_acc_maxabs'] = max(extrema['PED_U_acc_maxabs'], int(np.abs(accum).max()))
            u = rne24(accum, int(q['U_ped_exponent']))
            accum = q['V_ped_q16'].astype(np.int64) @ u
            extrema['PED_V_acc_maxabs'] = max(extrema['PED_V_acc_maxabs'], int(np.abs(accum).max()))
            v = rne24(accum, int(q['V_ped_exponent']))
            output[:, :, y//2, x//2] = rne24(v+q['PED_bias_q24'][:, None], 0).T
    permutation = q['consumer_permutation'].astype(int)
    value = updated[permutation]
    threshold = q['consumer_threshold'][:, None, None, None]
    direction = q['consumer_direction'][:, None, None, None]
    const = q['consumer_constant'][:, None, None, None]
    actual = np.where(const >= 0, const.astype(bool),
                      np.where(direction > 0, value >= threshold, value <= threshold))
    expected = unpack(data, 'proj')[:, :, :SIDE, :SIDE]
    expected_products = dict(
        F_nonzero_products=(160-source_empty)*int(np.count_nonzero(q['F_q16'])),
        PED_U_nonzero_products=160*int(np.count_nonzero(q['U_ped_q16'])),
        PED_V_nonzero_products=160*int(np.count_nonzero(q['V_ped_q16'])))
    legal_bounds, zero_vector_groups = {}, {}
    for name in ('U_conv2_theta', 'F', 'U_ped', 'V_ped'):
        a = q[name+'_q16'].astype(np.int64)
        zero_vector_groups[name] = int(np.count_nonzero(~a.reshape(-1, 8, a.shape[1]).any(axis=1)))
        positive, negative = np.maximum(a, 0).sum(1), np.minimum(a, 0).sum(1)
        if name == 'U_conv2_theta':
            lo, hi = negative, positive  # exact binary source domain 0/1
        else:
            lo = -LIMIT*positive+(LIMIT-1)*negative
            hi = (LIMIT-1)*positive-LIMIT*negative
        legal_bounds[name] = [int(lo.min()), int(hi.max())]
        assert lo.min() >= -(1 << 47) and hi.max() < (1 << 47)
    continuous_check = dict(native_available=continuous_gold is not None)
    if continuous_gold is not None:
        assert str(continuous_gold['frame_name']) == str(data['frame_name'])
        assert int(continuous_gold['proj_conv_res_state_frac']) == 14
        assert continuous_gold['proj_conv_res_origin_yx'].tolist() == [0, 0]
        assert bool(continuous_gold['proj_conv_res_helper_binding'])
        difference = output-continuous_gold['proj_conv_res_q24'].astype(np.int64)
        continuous_check.update(values=int(difference.size), differences=int(np.count_nonzero(difference)),
                                max_abs_difference=int(np.abs(difference).max()))
    return dict(gates=int(actual.size), gate_differences=int(np.count_nonzero(actual != expected)),
                U_actual_nonzero_AAC=source_terms, source_empty_anchor_time_pairs=source_empty,
                expected_nonzero_products=expected_products, complete_accumulator_legal_bounds=legal_bounds,
                static_zero_H8_coefficient_groups=zero_vector_groups,
                source_theta=float(data['sn2_theta']), output_theta=float(q['consumer_theta']),
                continuous_values=int(output.size), continuous_min=int(output.min()),
                continuous_max=int(output.max()), intermediate_maxabs=dict(extrema),
                continuous_reference='Independent NumPy int64 implements every saved RNE/sat boundary; native gold is the real fixed helper proj.conv_res return times2^14.',
                continuous_native_check=continuous_check,
                service_address_payload_execution_verified=False,
                complete_FP_chain_verified=False)


class Engine:
    """Blocking, finite instruction trace with independent operand-port overlap.

    Each call emits an actual word request or an eight-lane operation. A whole
    stage never becomes MAC_count/lanes. One accumulator vector is retained.
    """
    def __init__(self):
        self.now = 0
        self.free = Counter()
        self.busy = Counter()
        self.count = Counter()
        self.stage_steps = Counter()
        self.stage = 'setup'
        self.w_cache = None
        self.s_cache = None
        self.events = []

    def issue(self, resource, count=1, latency=1, ready=None, label=None):
        ready = self.now if ready is None else ready
        start = max(ready, self.free[resource])
        self.free[resource] = start+count
        self.busy[resource] += count
        self.count[label or resource] += count
        end = start+count-1+latency
        if len(self.events) < 24:
            self.events.append([self.stage, resource, int(start), int(end), label or resource])
        return end

    def wait(self, *times):
        target = max(self.now, *times)
        self.stage_steps[self.stage] += target-self.now
        self.now = target

    @staticmethod
    def words(address, size):
        return range(address//8, (address+size-1)//8+1)

    def read(self, address, size, label='state_read64', cache=False):
        end = self.now
        for word in self.words(address, size):
            if cache and self.s_cache == word:
                continue
            end = self.issue('SR', label=label)
            self.s_cache = word if cache else None
        return end

    def write(self, address, size, label='state_write64'):
        end = self.now
        for _ in self.words(address, size):
            end = self.issue('SW', label=label)
        self.s_cache = None
        self.wait(end)

    def weight(self, address):
        word = address//32
        if self.w_cache != word:
            self.wait(self.issue('W', label='coefficient_read256'))
            self.w_cache = word

    def alu(self, name, latency=1, ready=None):
        self.wait(self.issue('ALU', latency=latency, ready=ready, label=name))

    def complete24(self, name, ready=None):
        self.alu(name+'_RNE', ready=ready)
        self.alu(name+'_sat24')

    def dma(self, byte_count, label):
        # Complete 32-byte transactions, five abstract slots per transaction.
        beats = math.ceil(byte_count/32)
        self.count[label+'_transactions32'] += beats
        self.wait(self.issue('DMA', count=5*beats, label=label+'_bus_slots'))


def input_supply(e, data, layout='natural_T10_word'):
    e.stage = 'sn2_halo_input_and_repack'
    if layout == 'natural_T10_word':
        # Ordinary source kernels can emit one uint16 T10 word, six pad bits.
        # All 8x8x96 words are real input payload, not a free source-ready tile.
        for offset in range(0, 12288, 32):
            e.dma(32, 'sn2_T10word_input')
            e.write(SOURCE_BASE+offset, 32, 'sn2_cache_write64')
        return
    # Native capture layout: T,C,240,320 bitstream. For x=0..7 a single
    # actual 32-byte aligned transaction covers each (t,c,y) eight-bit row.
    shape = tuple(map(int, data['sn2_shape']))
    addresses = set()
    for c in range(C):
        for y in range(SIDE):
            for t in range(T):
                bit_address = ((t*C+c)*shape[2]+y)*shape[3]
                addresses.add((bit_address//8)//32)
                e.dma(32, 'sn2_native_input')
                e.wait(e.issue('DECODE', label='sn2_row_bit_extract'))
            # One 128-bit collector transposes 10x8 bits into eight uint16
            # words (six padding bits each). Each c,y row has
            # 16-byte allocation, so no free unaligned SRAM read-modify-write.
            e.write(SOURCE_BASE+(c*SIDE+y)*16, 16, 'sn2_cache_write64')
    e.count['sn2_distinct_external_transactions32'] = len(addresses)


def source_header(e, gates, positions, k):
    c, rem = divmod(k, 9)
    kh, kw = divmod(rem, 3)
    masks = np.zeros((len(positions), T), bool)
    end = e.now
    for p, (y, x) in enumerate(positions):
        sy, sx = y+kh-1, x+kw-1
        if 0 <= sy < SIDE and 0 <= sx < SIDE:
            first_bit = sx*16
            base = SOURCE_BASE+(c*SIDE+sy)*16
            end = max(end, e.read(base+first_bit//8,
                                 (first_bit%8+T+7)//8, 'source_word_read64', cache=True))
            masks[p] = gates[:, c, sy, sx]
    e.wait(end)
    e.wait(e.issue('DECODE', label='K_descriptor_scan'))
    return masks


def matrices(q):
    # All matrices have physical k-major, H-contiguous signed16 coefficients.
    # A 256-bit word holds sixteen H coefficients, including zero coefficients.
    values, bases, cursor = {}, {}, 0
    for name in ('U_conv2_theta', 'F', 'U_ped', 'V_ped'):
        a = q[name+'_q16'].astype(np.int64)
        values[name], bases[name] = a, cursor
        cursor += a.shape[1]*math.ceil(a.shape[0]/16)*32
    bases['BN2_constant'], cursor = cursor, cursor+96*4
    bases['PED_bias'], cursor = cursor, cursor+96*4
    bases['compare'], cursor = cursor, cursor+320
    return values, bases, (cursor+31)//32*32


def waddr(base, h_count, k, h):
    return base+(k*math.ceil(h_count/16)+h//16)*32


def sparse_u(e, gates, positions, a, base, order):
    e.stage = 'Conv2_U16_'+order
    source_nonempty = np.zeros((2, T), bool)
    if order == 'K_major':
        e.write(WORK_BASE, 1920, 'U_partial_zero_write64')
        for k in range(864):
            active = source_header(e, gates, positions, k)
            source_nonempty |= active
            for h in range(0, 16, 8):
                nz = int(np.count_nonzero(a[h:h+8, k]))
                if not nz or not active.any():
                    continue
                e.weight(waddr(base, 16, k, h))
                for p, t in np.argwhere(active):
                    address = WORK_BASE+((int(p)*T+int(t))*16+h)*6
                    ready = e.read(address, 48, 'U_partial_read64')
                    e.alu('U_AAC8', ready=ready)
                    e.count['U_actual_nonzero_AAC'] += nz
                    e.write(address, 48, 'U_partial_write64')
        for p in range(2):
            for t in range(T):
                for h in (0, 8):
                    address = WORK_BASE+((p*T+t)*16+h)*6
                    e.complete24('U', ready=e.read(address, 48))
                    e.write(address, 48, 'Z24_padded48_write64')
    else:
        for p in range(2):
            for t in range(T):
                for h in (0, 8):
                    e.alu('U_accumulator_zero')
                    for k in range(864):
                        active = source_header(e, gates, positions, k)
                        source_nonempty |= active
                        nz = int(np.count_nonzero(a[h:h+8, k]))
                        if active[p, t] and nz:
                            e.weight(waddr(base, 16, k, h))
                            e.alu('U_AAC8')
                            e.count['U_actual_nonzero_AAC'] += nz
                    e.complete24('U')
                    e.write(WORK_BASE+((p*T+t)*16+h)*6, 48, 'Z24_padded48_write64')
    return source_nonempty


def dense_output(e, a, base, input_base, input_stride, p, t, h, name):
    e.alu(name+'_accumulator_zero')
    for k in range(a.shape[1]):
        nz = int(np.count_nonzero(a[h:h+8, k]))
        if not nz:
            continue
        # Only one 64-bit input-word register: no whole latent row is free.
        ready = e.read(input_base+((p*T+t)*a.shape[1]+k)*input_stride,
                       3, name+'_input_read64', cache=True)
        e.weight(waddr(base, a.shape[0], k, h))
        e.alu(name+'_MAC8', latency=2, ready=max(ready, e.now))
        e.count[name+'_nonzero_products'] += nz
    e.complete24(name)


def gates_out(e, positions, q):
    e.stage = 'proj_gate_cutoffs'
    for h in range(0, C, 8):
        for p, (y, x) in enumerate(positions):
            for t in range(T):
                e.weight(e.bases['compare']+t*32)
                if int(q['consumer_constant'][t]) >= 0:
                    e.alu('constant_gate8')
                else:
                    s = int(q['consumer_permutation'][t])
                    ready = e.read(I_BASE+((p*T+s)*C+h)*3, 24, 'gate_I_read64')
                    e.alu('signed_cutoff_compare8', ready=ready)
            # H8 x ten gates collect in 128 bits; exact T-word cache layout.
            e.write(GATE_BASE+((y*SIDE+x)*C+h)*2, 16, 'proj_gate_word_write64')


def run_service(data, q, gates, order):
    e = Engine()
    weights, e.bases, pool_bytes = matrices(q)
    e.stage = 'coefficient_cold_fill'
    for offset in range(0, pool_bytes, 32):
        e.dma(32, 'coefficient_fill')
        e.wait(e.issue('CW', label='coefficient_local_write256'))
    input_supply(e, data)
    anchor_pairs = [[(y, x), (y, x+2)] for y in range(0, SIDE, 2) for x in (0, 4)]
    nonanchors = [(y, x) for y in range(SIDE) for x in range(SIDE) if y%2 or x%2]
    for positions in anchor_pairs+[nonanchors[i:i+2] for i in range(0, len(nonanchors), 2)]:
        e.stage = 'raw_I24_external_input'
        # Blocked per-position/T/H signed24 external layout: each complete
        # position is 2880 bytes (90 aligned transactions). No host-resident I.
        for offset in range(0, 5760, 32):
            e.dma(32, 'I24_input')
            e.write(I_BASE+offset, 32, 'I24_local_write64')
        anchor = positions[0][0]%2 == 0 and positions[0][1]%2 == 0
        if anchor:
            source_nonempty = sparse_u(e, gates, positions, weights['U_conv2_theta'], e.bases['U_conv2_theta'], order)
            e.stage = 'F_and_BN2_I24_merge'
            for p in range(2):
                for t in range(T):
                    for h in range(0, C, 8):
                        if source_nonempty[p, t]:
                            dense_output(e, weights['F'], e.bases['F'], WORK_BASE, 6, p, t, h, 'F')
                        else:
                            e.alu('F_source_empty_zero8')
                        e.weight(e.bases['BN2_constant']+h*4)
                        e.alu('BN_constant_add8')
                        address = I_BASE+((p*T+t)*C+h)*3
                        e.alu('I_branch_add8', ready=e.read(address, 24))
                        e.alu('anchor_sat24')
                        e.write(address, 24, 'updated_I24_write64')
            # Z last-used by F; the same 1920 bytes now hold P2/T10/R32 U24.
            e.stage = 'PED_U32'
            for p in range(2):
                for t in range(T):
                    for h in range(0, 32, 8):
                        dense_output(e, weights['U_ped'], e.bases['U_ped'], I_BASE, 3, p, t, h, 'PED_U')
                        e.write(WORK_BASE+((p*T+t)*32+h)*3, 24, 'PED_U24_write64')
            e.stage = 'PED_V32_bias_continuous_output'
            for p in range(2):
                for t in range(T):
                    for h in range(0, C, 8):
                        dense_output(e, weights['V_ped'], e.bases['V_ped'], WORK_BASE, 3, p, t, h, 'PED_V')
                        e.weight(e.bases['PED_bias']+h*4)
                        e.alu('PED_bias_add8')
                        e.alu('PED_bias_sat24')
                        e.alu('exact24f14_to_FP32_8')
                        # Native consumer output is FP32: one H8 vector fills
                        # the 32-byte egress slot and blocks until accepted.
                        e.dma(32, 'continuous_output')
        gates_out(e, positions, q)
    e.stage = 'proj_gate_cache_egress'
    for offset in range(0, 12288, 32):
        e.wait(e.read(GATE_BASE+offset, 32, 'proj_gate_output_read64'))
        e.dma(32, 'proj_gate_output')
    assert pool_bytes <= 131072
    return dict(order=order, service_slots=int(e.now), stage_slots=dict(e.stage_steps),
                resource_issue_slots=dict(e.busy), operations_and_word_requests=dict(e.count),
                coefficient_pool_used_bytes=pool_bytes, first_events=e.events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, default=HERE/'capture_inputs')
    parser.add_argument('--continuous-capture', type=Path, default=HERE/'capture_continuous')
    parser.add_argument('--fixtures', type=Path, help='Read compact per-axis NPZ while retaining original address metadata.')
    parser.add_argument('--write-fixtures', action='store_true')
    args = parser.parse_args()
    result = dict(scope='Source-ready sn2 + external raw I24 -> complete 4x4 PED continuous outputs and 8x8 proj gate words; top-left real boundary. No source-DAG, FP preview, native spike Conv, final PED sum or full-frame speed claim.',
        units='Declared CPU service slots; 8-lane MAC latency2/AAC-round-sat-compare each latency1, no clock/area mapping.',
        resource=dict(state_pool_bytes=32768, source_cache_bytes=12288, gate_cache_bytes=12288,
            P2_I24_bytes=5760, shared_Z_or_U_workspace_bytes=1920,
            registers_and_staging_bytes=320, peak_allocated_bytes=32576,
            register_allocation_bytes=dict(accumulator8x48=48, operand_assembly8x48=48,
                coefficient_word=32, state_word=8, DMA_or_output_staging=64,
                source_transpose_collector=16, gate_word_collector=16,
                source_nonempty_and_descriptor_bits=8, counters_addresses_and_configuration=80),
            workspace='U16 full-K partial48 and padded Z24 occupy1920B, released after F; PED U32 P2/T10/state24 then occupies the same1920B. No simultaneous Z/U allocation.',
            coefficient_pool_bytes=131072, coefficient_port='256bit1R1W, latency1; one256bit current word register',
            state_port='One64bit read and one64bit write port, latency1; source and accumulator accesses share it.',
            arithmetic='8 shared lanes supporting signed48 AAC, signed16x24->48 MAC, separate RNE and sat24, compare and exact24f14->FP32 conversion. MAC latency2, other operations latency1; additional capability versus the source-only add core, not same area.',
            global_bus='One32byte/5slot transaction stream, blocking ingress/egress, 64B staging. Actual source row extraction +128bit repack collector is separately charged.',
            schedule='Single P2 live context, blocking accumulator recurrence; W and operand reads may overlap. No full latent vector kept in an uncharged register. K-major and time-major are two complete fixed trajectories, no per-stage minimum splice.'),
        boundaries=['Input sn2 gates and I24 are actual captured values; sn2 producer/FP32 frozen preview is not included in this standalone subchain.',
                    'Actual FP Conv2/BN/residual shadows are excluded because fixed consumers do not read them.',
                    'NumPy int64 replays fullK and all intermediate RNE/sat boundaries; captured proj gates and actual fixed helper continuous return are checked independently. This is numerical validation of the bounded subchain, not an RTL or full-FP-chain PASS.',
                    'Engine traces finite requests/dependencies but does not execute a byte-addressed SRAM/accumulator payload. Numerical zero-difference is an independent formula check, not validation of instruction-by-instruction address/payload correspondence.',
                    'Not the strongest possible CMVM/Gustav implementation: no constant-matrix CSE, across-P2 state reuse, multi-accumulator pipelining, global transpose cache or NRV task queue. Both students get identical limitations.',
                    'The resident 8x8 source/gate caches are charged. The 184320-byte complete I tile is read externally in P2 groups, never simultaneously resident.',
                    'AllT sharing keeps a current W word until all corresponding source events finish in K-major. Time-major retains one accumulator but pays repeated source scans and coefficient reads. Whole source-empty (p,t) discovered by the mandatory K scan skips F for both students; no future value-zero oracle.',
                    'Main ingress is natural T10-word uint16 supplied by an ordinary temporal source kernel. Native debug TCHW bit-plane repacking is a separate input-only reference, never the headline denominator.',
                    'All DMA is drained or filled in 32B chunks with 64B staging; gate egress also charges its state SRAM reads. No complete input tile waits in an unmodeled host buffer.'], axes={})
    for axis in ('ordinary', 'lifting_raw'):
        if args.fixtures:
            data = q = gold = read_npz(args.fixtures/(axis+'.npz'))
        else:
            directory = args.capture/axis
            data = read_npz(directory/'000_zurich_city_09_a_0001.npz')
            q = read_npz(directory/'parameters.npz')
            gold_path = args.continuous_capture/axis/'000_zurich_city_09_a_0001.npz'
            gold = read_npz(gold_path) if gold_path.exists() else None
        if args.write_fixtures and gold is not None:
            write_fixture(HERE/'consumer_fixtures'/(axis+'.npz'), data, q, gold)
        gates = unpack(data, 'sn2')[:, :, :SIDE, :SIDE]
        numeric = numerical_reference(data, q, gates, gold)
        assert numeric['gate_differences'] == 0
        if gold is not None:
            assert numeric['continuous_native_check']['differences'] == 0
        ingress = {}
        for layout in ('natural_T10_word', 'debug_TCHW_bitplane'):
            engine = Engine()
            input_supply(engine, data, layout)
            ingress[layout] = dict(service_slots=engine.now, operations_and_word_requests=dict(engine.count))
        result['axes'][axis] = dict(numeric=numeric,
            ingress_controls=ingress,
            services={order: run_service(data, q, gates, order) for order in ('K_major', 'time_major')})
        for service in result['axes'][axis]['services'].values():
            assert service['operations_and_word_requests']['U_actual_nonzero_AAC'] == numeric['U_actual_nonzero_AAC']
            for name, value in numeric['expected_nonzero_products'].items():
                assert service['operations_and_word_requests'][name] == value
        print(axis, json.dumps({k:v['service_slots'] for k,v in result['axes'][axis]['services'].items()}), flush=True)
    (HERE/'consumer_service_result.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
