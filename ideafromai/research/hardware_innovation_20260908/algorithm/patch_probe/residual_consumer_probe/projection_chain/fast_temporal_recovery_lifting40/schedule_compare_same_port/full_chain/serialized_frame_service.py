"""One fixed, finite, serialized-layer service construction, not optimal RTL.

Actual full-frame spike words determine complete-K sparse row work. A row's
pipeline drains before the next row; stages spill between layers. This makes
the reservation count composable without pretending to have arbitrary overlap.
No cycle-level memory payload simulation or complete Gustav reproduction.
"""
from pathlib import Path
from collections import Counter
import json
import math
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from consumer_service import read_npz, unpack
sys.path.insert(0, str(HERE.parent/'two_stage_writeback'))
import run_compare as source_program

T, C, H, W = 10, 96, 240, 320
POP = np.asarray([int(i).bit_count() for i in range(1024)], np.uint8)


class Ledger:
    def __init__(self):
        self.now = 0
        self.phases = []
        self.counts = Counter()

    def add(self, name, items, state_bytes, detail):
        assert state_bytes <= 131072, (name, state_bytes)
        counts = {k: int(v) for k, v in items.items()}
        slots = sum(v for k, v in counts.items() if k.endswith('_slots'))
        self.phases.append(dict(name=name, start=self.now, end=self.now+slots,
                               service_slots=slots, state_bytes=state_bytes,
                               counts=counts, detail=detail))
        self.now += slots
        self.counts.update(counts)


def dma(items, label, size, state_io=None):
    n = math.ceil(int(size)/32)
    items[label+'_bytes'] += 32*n
    items[label+'_slots'] += 5*n
    if state_io is not None:
        items[label+'_state_'+state_io+'_slots'] += math.ceil(int(size)/8)


def words(gate):
    out = np.zeros(gate.shape[1:], np.uint16)
    for t in range(T):
        out |= gate[t].astype(np.uint16) << t
    return out


def sparse(w, word, stride, latency, out_bytes, label):
    """P2/T10, H8 groups, at most H32 live. Actual scalar-row masks.

    Each complete K row is scanned and encoded before NRV execution. Two
    explicit source-word reads + pack slot are paid; no free directory. One
    coefficient vector per live H8 and one issued AAC per live T/P/H8.
    """
    channels, height, width = word.shape
    oy = np.arange(0, height, stride)
    ox = np.arange(0, width, 2*stride)
    groups = len(oy)*len(ox)
    all_live = np.zeros((len(oy), len(ox), 2), np.uint16)
    # All output channels are split into resident H32 tiles for native conv.
    items = Counter()
    for h0 in range(0, w.shape[0], 32):
        local = w[h0:h0+32]
        hg = local.shape[0]//8
        coefficient_groups = np.any(local.reshape(hg, 8, -1) != 0, axis=1).sum(0)
        rows = np.zeros((len(oy), len(ox)), np.int64)
        for kh in range(3):
            sy = oy+kh-1
            for kw in range(3):
                sx0, sx1 = ox+kw-1, ox+stride+kw-1
                valid_y = (sy >= 0) & (sy < height)
                a = word[:, np.clip(sy, 0, height-1)[:, None], np.clip(sx0, 0, width-1)[None, :]].copy()
                b = word[:, np.clip(sy, 0, height-1)[:, None], np.clip(sx1, 0, width-1)[None, :]].copy()
                a[:, ~valid_y, :] = 0; b[:, ~valid_y, :] = 0
                a[:, :, (sx0 < 0) | (sx0 >= width)] = 0
                b[:, :, (sx1 < 0) | (sx1 >= width)] = 0
                all_live[:, :, 0] |= np.bitwise_or.reduce(a, axis=0)
                all_live[:, :, 1] |= np.bitwise_or.reduce(b, axis=0)
                for c in range(channels):
                    k = (c*3+kh)*3+kw
                    n_h = int(coefficient_groups[k])
                    active = POP[a[c]].astype(np.int64)+POP[b[c]]
                    if h0 == 0:
                        items['logical_source_occurrences'] += int(active.sum())
                    if n_h == 0:
                        continue
                    live = (a[c] | b[c]) != 0
                    nr = int(np.count_nonzero(live))
                    rows += live
                    items['source_scan_slots'] += 3*groups
                    items['source_scan_read_bytes'] += 16*groups
                    items['NRV_decode_slots'] += 2*nr
                    items['coefficient_read_slots'] += n_h*nr
                    items['coefficient_read_bytes'] += 32*n_h*nr
                    items['AAC_issue_slots'] += n_h*int(active.sum())
                    items['AAC_drain_slots'] += (latency-1)*nr
                    items['nonempty_K_rows'] += nr
        # Two encoded K10/P2T20 records per 64-bit word. Packed directory is
        # real scratch, built/read before this tile's arithmetic, then reused.
        nr_words = int(((rows+1)//2).sum())
        items['NRV_write_slots'] += nr_words
        items['NRV_read_slots'] += nr_words
        items['NRV_state_bytes_transferred'] += 16*nr_words
        items['accumulator_clear_slots'] += groups*hg*20+groups*(latency-1)
        items['result_write_slots'] += groups*local.shape[0]*20*out_bytes//8
        items['result_write_bytes'] += groups*local.shape[0]*20*out_bytes
        if out_bytes == 3:
            items['RNE_sat_issue_slots'] += groups*hg*20
            items['RNE_sat_drain_slots'] += groups
    return items, all_live


def dense(items, w, vectors, latency, out_bytes, label, active_tiles=None):
    """H32/P2T10 blocks; full scalar input, no unproved zero-payload skip."""
    active_h_per_k = np.any(w.reshape(-1, 8, w.shape[1]) != 0, axis=1).sum(0)
    vec = int(vectors)
    tiles = math.ceil(vec/20) if active_tiles is None else int(active_tiles)
    input_bytes = 3 if out_bytes == 3 else 4
    items[label+'_MAC_issue_slots'] += int(active_h_per_k.sum())*vec
    items[label+'_MAC_drain_slots'] += (latency-1)*w.shape[1]*tiles*math.ceil(w.shape[0]/32)
    items[label+'_coefficient_read_slots'] += int(active_h_per_k.sum())*tiles
    items[label+'_coefficient_read_bytes'] += 32*int(active_h_per_k.sum())*tiles
    # Results stay in TP,K layout for a contiguous eight-lane write. K-major
    # consumers gather one scalar from each TP row: volume/64 would grant a
    # free transpose or an unallocated twenty-line scalar cache.
    scalar_reads = sum(1+((k*input_bytes)%8+input_bytes > 8) for k in range(w.shape[1]))
    items[label+'_input_read_slots'] += scalar_reads*vec*math.ceil(w.shape[0]/32)
    items[label+'_input_read_bytes'] += 8*scalar_reads*vec*math.ceil(w.shape[0]/32)
    items[label+'_accumulator_clear_slots'] += (w.shape[0]//8)*vec
    items[label+'_result_write_slots'] += math.ceil(w.shape[0]*vec*out_bytes/8)
    items[label+'_result_write_bytes'] += w.shape[0]*vec*out_bytes
    if out_bytes == 3:
        items[label+'_RNE_sat_slots'] += (w.shape[0]//8)*vec+math.ceil(vec/20)


def frame(axis):
    path = HERE/'capture'/axis
    q, p = read_npz(path/'parameters.npz'), read_npz(path/'live_parameters.npz')
    original = read_npz(HERE.parent/'capture_inputs'/axis/'000_zurich_city_09_a_0001.npz')
    old_counts = json.loads((HERE.parent/'capture_inputs'/axis/'capture.json').read_text())['source_checks'][0]['sources']
    gates = {key: words(unpack(original, key)) for key in ('sn1', 'sn2', 'proj')}
    assert all(float(original[k+'_theta']) == 1.0 for k in ('sn1', 'sn2', 'proj'))
    ledger = Ledger()
    nodes, _, _ = source_program.graph_variant(axis, True)
    choices = [source_program.compile_program(nodes, policy) for policy in ('official_ready', 'last_use_pressure')]
    program, info = min(choices, key=lambda v: (v[1]['instructions'], v[1]['peak_live_words']))
    batches = C*H*W//8
    items = Counter(source_program_slots=batches*(len(program)+7),
                    source_LOAD_read_stall_slots=batches*20,
                    source_gate_write_slots=C*H*W*2//8,
                    raw_I_staging_write_slots=T*C*H*W*3//8)
    dma(items, 'raw_I_input', T*C*H*W*3)
    dma(items, 'sn1_output', C*H*W*2, 'read')
    ledger.add('source_complete_T10', items, 32*C*T*3+32*C*2,
        dict(program=info, physical_source_tile='one y row, 32 x positions; lanes traverse C,x; no complete I frame retained',
             LOAD='Each eight-lane signed24 input uses three 64-bit reads; two added stalls beyond the already charged LOAD. Gate commits are eight one-word FIFO transfers.'))

    rank = int(p['preview_shared_rank'])
    assert rank == 32 and not np.any(p['preview_v'][rank:])
    u, v = p['preview_u'][:, :rank].T, p['preview_v'][:rank].T
    items, live = sparse(u, gates['sn1'], 1, 4, 4, 'preview_U')
    assert items['logical_source_occurrences'] == old_counts['sn1']['source_occurrences']
    dense(items, v, H*W*T, 4, 4, 'preview_V')
    # Ordinary epilogue fusion: keep V's already rounded FP32 result in RF,
    # apply the actual fixed-BN affine operation, then write Y once.
    del items['preview_V_result_write_slots']
    del items['preview_V_result_write_bytes']
    # Actual TensorFloat multiplicands require a rounded input when V sees Z.
    items['preview_Z_TF32_round_slots'] += H*W*T*rank//8
    items['BN1_FMA_slots'] += H*W*T*C//8+3*(H*W//2)
    items['BN1_write_slots'] += H*W*T*C*4//8
    # SIMD is across channels here, not across ten output times padded to H16.
    # Per P2/C8: twenty source vectors, twenty psums, coefficient word cache.
    temporal_tiles = H*W*C//16
    items['sn2_full_temporal_FMA_slots'] += int(np.count_nonzero(p['preview_A']))*H*W*C//8
    items['sn2_temporal_drain_slots'] += 3*T*temporal_tiles
    items['sn2_temporal_coefficient_read_slots'] += math.ceil(p['preview_A'].nbytes/32)*temporal_tiles
    items['sn2_temporal_source_read_slots'] += H*W*T*C*4//8
    items['sn2_temporal_accumulator_clear_slots'] += H*W*T*C//8
    items['sn2_bias_threshold_gate_slots'] += 3*H*W*T*C//8
    items['sn2_gate_word_write_slots'] += H*W*C*2//8
    # x-tiles contain 32 output positions and a two-column halo; no full gate
    # plane fits on-chip. Explicit duplicated halo reads are charged.
    dma(items, 'sn1_halo_input', H*(W//32)*3*34*C*2, 'write')
    dma(items, 'preview_coefficients', u.nbytes+v.nbytes+p['preview_A'].nbytes+4*C*4)
    items['coefficient_refill_write_slots'] += math.ceil((u.nbytes+v.nbytes+p['preview_A'].nbytes+4*C*4)/32)
    dma(items, 'sn2_output', H*W*C*2, 'read')
    ledger.add('preview_and_noncausal_sn2', items, 3*34*C*2+864*4+2*T*C*4+2*T*rank*4,
        dict(live_latents=rank, private_tail='exact zero V columns omitted for both',
             note='FP32/TF32 primitive service uses actual CUDA source trace; numerical replay has separately reported rounding differences.'))

    items, live = sparse(q['U_conv2_theta_q16'], gates['sn2'], 2, 2, 3, 'conv2_U16')
    assert items['logical_source_occurrences'] == old_counts['sn2']['source_occurrences']
    nonempty_time = int(POP[live].sum())
    # F is skipped only for a provably empty entire K vector. The common
    # constant and retained I are still merged for every output time.
    dense(items, q['F_q16'], nonempty_time, 2, 3, 'F_BN2',
          active_tiles=np.count_nonzero(np.any(live != 0, axis=2)))
    # F's exact RNE/sat remains. Its following bias/raw-I merge is an ordinary
    # RF epilogue; no standalone F tensor needs a SRAM write.
    del items['F_BN2_result_write_slots']
    del items['F_BN2_result_write_bytes']
    anchors = H*W//4
    items['raw_residual_read_slots'] += T*C*H*W*3//8
    items['BN2_bias_and_residual_slots'] += 2*anchors*T*C//8+anchors*T*C//8
    items['updated_I_write_slots'] += anchors*T*C*3//8
    dense(items, q['U_ped_q16'], anchors*T, 2, 3, 'PED_U32')
    dense(items, q['V_ped_q16'], anchors*T, 2, 3, 'PED_V32')
    items['PED_bias_sat_slots'] += 2*anchors*T*C//8
    items['proj_gate_compare_slots'] += H*W*T*C//8
    items['proj_gate_word_write_slots'] += H*W*C*2//8
    dma(items, 'raw_residual_input', T*C*H*W*3, 'write')
    dma(items, 'sn2_halo_input', (H//2)*(W//64)*3*66*C*2, 'write')
    integer_weights = sum(q[k+'_q16'].nbytes for k in ('U_conv2_theta', 'F', 'U_ped', 'V_ped'))+C*3*2
    dma(items, 'integer_coefficients', integer_weights)
    items['integer_coefficient_refill_write_slots'] += math.ceil(integer_weights/32)
    dma(items, 'proj_gate_output', H*W*C*2, 'read')
    dma(items, 'continuous_PED_output', anchors*T*C*3, 'read')
    ledger.add('integer_residual_and_both_consumers', items, 3*66*C*2+32*C*T*3//2+2*T*C*3+864*4,
        dict(nonempty_complete_K_time_vectors=nonempty_time, full_K=864,
             note='All nonanchors keep raw I. Two consumers are charged, including the stored continuous branch that must survive dynamic projection BN.'))

    items, _ = sparse(p['proj_weight_fp32'].reshape(C, -1), gates['proj'], 2, 4, 4, 'native_projection')
    assert items['logical_source_occurrences'] == old_counts['proj']['source_occurrences']
    dma(items, 'native_weights', p['proj_weight_fp32'].nbytes)
    items['native_coefficient_refill_write_slots'] += math.ceil(p['proj_weight_fp32'].nbytes/32)
    dma(items, 'proj_gate_halo_input', 3*(H//2)*(W//64)*3*66*C*2, 'write')
    raw_bytes = anchors*T*C*4
    dma(items, 'native_projection_store', raw_bytes, 'read')
    ledger.add('native_projection_before_global_BN', items, 3*66*C*2+2*T*32*4+864*4,
        dict(resident_weight_tile_bytes=864*32*4, coefficient_tiles=3,
             complete_BN_domain_values=anchors*T*C, can_normalize_local_tile=False))

    # A scalar data-transfer reservation at the global completion barrier.
    # The actual statistics controller and reduction order remain OPEN.
    items = Counter()
    dma(items, 'BN_statistics_read', raw_bytes)
    # Two separately stored accumulators/channel; sum and square-sum only
    # serve a resource estimate here, not an admitted numerical replacement.
    values = anchors*T*C
    items['BN_stats_sum_square_FMA_slots'] += 2*values//8
    items['BN_stats_finalize_slots'] += C*32
    dma(items, 'BN_normalization_reread', raw_bytes)
    dma(items, 'continuous_PED_reread', anchors*T*C*3)
    items['BN_norm_and_residual_issue_slots'] += 2*values//8
    items['BN_convert_q24_slots'] += values//8
    dma(items, 'complete_PED_output', raw_bytes)
    ledger.add('full_domain_BN_then_final_add', items, 2*C*4+64+2*8*4,
        dict(statistics_domain=[T,C,H//2,W//2],
             numerical_reduction_closed=False,
             note='This explicit materialize/replay schedule is a cost construction, NOT a lower bound: fusion, early statistic accumulation or recompute may improve it. Sum/square primitive charge is not an approved BN replacement.'))
    return dict(service_slots=ledger.now, phases=ledger.phases, totals=dict(ledger.counts),
                source_program=info, state_capacity=131072, coefficient_capacity=131072,
                full_K_source_occurrences_match_earlier_independent_capture=True,
                complete_numerical_hardware_equivalence=False, complete_Gustav_or_CMVM_baseline=False)


def main():
    result = dict(scope=__doc__, resource_contract='resource_contract.json', axes={})
    for axis in ('ordinary', 'lifting_raw'):
        result['axes'][axis] = frame(axis)
        print(axis, result['axes'][axis]['service_slots'], flush=True)
    a, b = [result['axes'][k]['service_slots'] for k in ('ordinary','lifting_raw')]
    result['same_construction_service_reduction'] = 1-b/a
    result['Stage_B_admission'] = 'NOT_CLOSED: scalar reservation construction; full statistics arithmetic and stronger merged baseline remain open'
    (HERE/'serialized_frame_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
