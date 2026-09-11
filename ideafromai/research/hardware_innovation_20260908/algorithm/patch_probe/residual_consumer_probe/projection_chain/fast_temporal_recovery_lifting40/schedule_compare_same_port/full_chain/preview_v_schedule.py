"""Payload-executing common preview V microkernel, not a whole-chain result.

Eight FP32 FMA lanes, latency four, one issue/slot, RF 96x8 words,
state SRAM 1R64/1W64, coefficient SRAM 1R256 with one-slot read latency.
One 64-byte staging area is reused by DMA, compute operands and writeback.
The kernel has full K32/H96/P2/T10; selected real Z values, actual V weights.
No training, quantization, FPGA/ASIC PPA or complete Gustav implementation.
"""
from pathlib import Path
from collections import Counter
import ctypes
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from numerical_reference import difference, tf32_round
sys.path.insert(0, str(HERE.parent))
from consumer_service import read_npz

fma = ctypes.CDLL('libm.so.6').fmaf
fma.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
fma.restype = ctypes.c_float


def vector_fma(a, scalar, c):
    return np.asarray([fma(float(x), float(scalar), float(y)) for x, y in zip(a, c)], np.float32)


class Machine:
    def __init__(self, data, coefficients, transposed, backpressure=False):
        self.time = 0
        self.rf = np.zeros((96, 8), np.float32)
        self.ready = np.zeros(96, np.int64)
        self.pending = {}
        self.sram = bytearray(131072)
        self.coefficients = bytearray(131072)
        # Compile-time K,H8,lane layout: one real 256-bit address supplies
        # each coefficient vector. No runtime eight-row coefficient gather.
        packed_coefficients = coefficients.reshape(12, 8, 32).transpose(2, 0, 1).astype('<f4').tobytes()
        self.coefficients[:len(packed_coefficients)] = packed_coefficients
        self.transposed = transposed
        self.backpressure = backpressure
        self.count = Counter()
        self.timeline = []
        self.input_word = None
        self.input_address = None
        self.input_response = None
        self.coefficient_word = None
        self.coefficient_response = None
        self.max_pending = 0
        self.max_staging_bytes = 0
        self.input_base = 4096 if transposed else 0
        # Explicit, deliberately serialized input DMA and state writes.
        payload = data.astype('<f4').tobytes()
        for start in range(0, len(payload), 32):
            for _ in range(5):
                self.tick(tag='input_DMA')
            source = np.frombuffer(payload[start:start+32], '<f4').copy()
            self.tick(issue=('TF32_round', 80, source), tag='input_conversion')
            while self.pending:
                self.tick(tag='input_conversion_drain')
            converted = self.rf[80].astype('<f4').tobytes()
            for j in range(0, 32, 8):
                self.tick(write=(start+j, converted[j:j+8]), tag='input_stage')
        if transposed:
            # A real scratch transpose. Each destination word gathers two
            # TP rows with two 64-bit reads, followed by one 64-bit write.
            for k in range(32):
                for tp in range(0, 20, 2):
                    first, second = (tp*32+k)*4, ((tp+1)*32+k)*4
                    self.tick(read=first//8*8, tag='transpose_read')
                    self.tick(read=second//8*8, tag='transpose_read')
                    first_value = bytes(self.input_word[first % 8:first % 8+4])
                    self.tick(tag='transpose_write', transpose=(first_value, second % 8, 4096+(k*20+tp)*4))
        self.timeline.append(dict(phase='input_and_layout', end=self.time))

    def tick(self, read=None, write=None, issue=None, coef=None, tag='', transpose=None):
        # One fixed synthetic stress trace, shared by both axes/layouts. It
        # checks that stalls preserve operands and in-flight writes; it is
        # not a captured downstream workload or a second resource point.
        while self.backpressure and ((read is not None and self.time % 32 >= 24)
                or ((write is not None or transpose is not None) and self.time % 32 >= 28)):
            self.tick(tag='state_backpressure')
        # All responses and writebacks become visible at the start of slot.
        if self.time in self.pending:
            dest, value = self.pending.pop(self.time)
            self.rf[dest] = value
            self.count['RF_writebacks'] += 1
        if self.input_response is not None:
            self.input_address, self.input_word = self.input_response
            self.input_response = None
        if self.coefficient_response is not None:
            self.coefficient_word = self.coefficient_response
            self.coefficient_response = None
        if transpose is not None:
            first, offset, address = transpose
            write = (address, first+bytes(self.input_word[offset:offset+4]))
        if read is not None:
            assert read % 8 == 0 and 0 <= read <= len(self.sram)-8
            self.input_response = (read, bytes(self.sram[read:read+8]))
            self.count['state_read_words'] += 1
        if write is not None:
            address, payload = write
            assert address % 8 == 0 and len(payload) == 8
            self.sram[address:address+8] = payload
            self.count['state_write_words'] += 1
        if coef is not None:
            h, k = coef
            address = (k*12+h//8)*32
            self.coefficient_response = np.frombuffer(self.coefficients, '<f4', count=8, offset=address).copy()
            self.count['coefficient_read_words_256'] += 1
        if issue is not None:
            kind, dest, source_address = issue
            assert self.ready[dest] <= self.time, ('RAW', self.time, dest, self.ready[dest])
            if kind == 'clear':
                value = np.zeros(8, np.float32)
            elif kind == 'TF32_round':
                value = tf32_round(source_address)
            else:
                assert self.input_address == source_address//8*8, (self.time, self.input_address, source_address)
                offset = source_address % 8
                scalar = np.frombuffer(self.input_word[offset:offset+4], '<f4')[0]
                value = vector_fma(self.coefficient_word, scalar, self.rf[dest])
            ready = self.time+4
            assert ready not in self.pending, ('one_RF_write_port', ready)
            self.pending[ready] = (dest, value)
            self.ready[dest] = ready
            self.count[kind+'_issue'] += 1
        self.max_pending = max(self.max_pending, len(self.pending))
        # Compute holds a 32B coefficient vector, current and next 8B source
        # words. DMA and output drain do not run concurrently with compute.
        self.max_staging_bytes = max(self.max_staging_bytes, 48 if issue is not None else 32)
        self.count[tag+'_slots'] += 1
        self.time += 1

    def address(self, tp, k):
        return self.input_base+4*(k*20+tp if self.transposed else tp*32+k)

    def execute(self):
        out = np.empty((20, 96), np.float32)
        for h0 in range(0, 96, 32):
            start = self.time
            for r in range(80):
                self.tick(issue=('clear', r, 0), tag='clear')
            # The 20 TP destinations separate accumulator revisits by at
            # least 20 slots. Coefficient/source reads overlap issued FMAs.
            for k in range(32):
                for hg in range(4):
                    self.tick(read=self.address(0, k)//8*8, coef=(h0+8*hg, k), tag='prefetch')
                    for tp in range(20):
                        next_read = self.address(tp+1, k)//8*8 if tp < 19 else None
                        if next_read == self.address(tp, k)//8*8:
                            next_read = None
                        self.tick(read=next_read, issue=('FMA', tp*4+hg, self.address(tp, k)), tag='FMA')
            while self.pending:
                self.tick(tag='drain')
            for tp in range(20):
                for hg in range(4):
                    value = self.rf[tp*4+hg].copy()
                    out[tp, h0+hg*8:h0+(hg+1)*8] = value
                    payload = value.astype('<f4').tobytes()
                    address = 8192+(tp*96+h0+hg*8)*4
                    for j in range(0, 32, 8):
                        self.tick(write=(address+j, payload[j:j+8]), tag='result_write')
            self.timeline.append(dict(phase='H32_complete', h0=h0, start=start, end=self.time))
        actual = np.frombuffer(self.sram, '<f4', count=20*96, offset=8192).reshape(20, 96).copy()
        assert np.array_equal(out, actual)
        return actual


def run(data, weight, transposed, backpressure=False):
    m = Machine(data, tf32_round(weight), transposed, backpressure)
    output = m.execute()
    expected = np.zeros_like(output)
    for k in range(32):
        for tp in range(20):
            for h in range(0, 96, 8):
                expected[tp, h:h+8] = vector_fma(tf32_round(weight[h:h+8, k]),
                    tf32_round(data[tp, k]), expected[tp, h:h+8])
    comparison = difference(output, expected)
    assert comparison['differences'] == 0
    return output, dict(service_slots=m.time, counts=dict(m.count), timeline=m.timeline,
        ordered_FP32_FMA_reference=comparison, allocated_RF_words_per_lane=96,
        live_accumulator_words_per_lane=80, allocated_state_bytes=131072,
        state_high_water_address=8192+20*96*4, staging_bytes=64,
        backpressure='fixed synthetic: R blocked at slots 24..31 mod32, W at28..31' if backpressure else 'always_ready',
        max_staging_bytes=m.max_staging_bytes, max_inflight_vector_results=m.max_pending,
        no_RAW_or_RF_write_collision=True,
        coefficients_already_resident_bytes=96*32*4,
        completion='H96 output written in state SRAM; downstream BN and external output DMA are outside this microkernel.',
        input_conversion='One explicitly issued eight-value TF32 RNE conversion per 32B input chunk, four-slot result latency. Later physical implementation still needs timing/area verification.')


def main():
    result = dict(scope=__doc__, axes={}, common_control=True,
        whole_frame_service_closed=False, native_BN_statistics_closed=False,
        note='FMA multiplicands are rounded to TF32 to diagnose the enabled cuDNN path. Ordered fmaf equivalence is tested; CUDA convolution reduction-order identity remains open.')
    for axis in ('ordinary', 'lifting_raw'):
        q = read_npz(HERE/'capture'/axis/'live_parameters.npz')
        x = read_npz(HERE/'capture'/axis/'000_zurich_city_09_a_0001.npz')
        rows = {}
        for label in ('corner', 'interior'):
            z = x[label+'_preview_Z_shared'][:, :, 0, :2].transpose(0, 2, 1).reshape(20, 32)
            gold = x[label+'_preview_shared_raw'][:, :, 0, :2].transpose(0, 2, 1).reshape(20, 96)
            a, direct = run(z, q['preview_v'][:32].T, False)
            b, transposed = run(z, q['preview_v'][:32].T, True)
            c, direct_stall = run(z, q['preview_v'][:32].T, False, True)
            d, transpose_stall = run(z, q['preview_v'][:32].T, True, True)
            assert np.array_equal(a, b)
            assert np.array_equal(a, c) and np.array_equal(a, d)
            rows[label] = dict(direct=direct, transpose=transposed, versus_actual_CUDA=difference(a, gold),
                direct_fixed_stress=direct_stall, transpose_fixed_stress=transpose_stall,
                layout_value_differences=0)
        result['axes'][axis] = rows
        print(axis, [(k, v['direct']['service_slots'], v['transpose']['service_slots']) for k,v in rows.items()], flush=True)
    result['decision'] = 'Keep common direct prefetch; transpose adds layout work at this resource point. This stops the scratch-transpose endpoint only; no lifting-family decision.'
    (HERE/'preview_v_schedule_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
