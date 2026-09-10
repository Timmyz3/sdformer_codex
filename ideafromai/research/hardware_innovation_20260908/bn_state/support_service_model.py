"""Finite-resource FC1+PSN service comparison on real trained support captures.

No Torch, network forward, capture regeneration, RTL or EDA. The numerical
reference is the same student's integer operator, not the original FP32 model.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
from collections import OrderedDict, Counter
import io
import json
import math
import pickle
import time
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
ALG = HERE.parent/'algorithm'
CAP = ALG/'support_training'
T, P, C, H, B, LANES = 10, 19200, 96, 384, 32, 96


def read_torch(path):
    """Read only the tensor/dict subset emitted by these local torch saves."""
    with zipfile.ZipFile(path) as archive:
        prefix = next(n[:-8] for n in archive.namelist() if n.endswith('data.pkl'))
        dtypes = {name+'Storage': np.dtype(dt) for name, dt in
                  [('Char', 'i1'), ('Short', '<i2'), ('Int', '<i4'),
                   ('Long', '<i8'), ('Double', '<f8'), ('Float', '<f4'), ('Bool', '?')]}
        def rebuild(storage, offset, size, stride, *unused):
            dt, raw = storage
            return np.ndarray(size, dtype=dt, buffer=raw, offset=offset*dt.itemsize,
                              strides=tuple(x*dt.itemsize for x in stride))
        class Reader(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch' and name in dtypes:
                    return dtypes[name]
                if module == 'torch._utils' and name in ('_rebuild_tensor', '_rebuild_tensor_v2'):
                    return rebuild
                if (module, name) == ('collections', 'OrderedDict'):
                    return OrderedDict
                raise ValueError((module, name))
            def persistent_load(self, pid):
                _, dt, key, _, count = pid
                return dt, archive.read(prefix+'data/'+key)
        return Reader(io.BytesIO(archive.read(prefix+'data.pkl'))).load()


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def pool_layout(n_lut, lut_bits, forced, use_lut):
    parts = dict(W=C*H, thresholds=T*H*6, A_padded=208,
                 zero_Y_gate_templates=T*H//8, theta_and_context_padded=16,
                 dictionary=192 if (forced or use_lut) else 0,
                 L=n_lut*H*lut_bits//8 if use_lut else 0)
    raw = sum(parts.values())
    return dict(parts=parts, resident_bytes=raw, slack_bytes=131072-raw,
                cold_fill_beats=math.ceil(raw/128))


def jobs_from_counts(raw_counts, match_counts, valid_codes, min_pop, popcounts):
    # Integer IDs: W 0..95; L 96..185. A group is visited in source order,
    # then code order. Identical ordinary mask/column scheduling in both modes.
    jobs = []
    for g in range(6):
        for c in range(g*16, (g+1)*16):
            if raw_counts[c]:
                jobs.append((c, int(raw_counts[c])))
        if match_counts is not None:
            for k in range(16):
                if popcounts[g, k] >= min_pop and match_counts[g, k]:
                    jobs.append((C+valid_codes[g, k], int(match_counts[g, k])))
    return jobs


def schedule_fc1(jobs, lut_bits=16, hidden_block=0, n_lut=90):
    """Clocked finite descriptor/prefetch model; never divide total adds by96.

    Every job is one same-tile coefficient broadcast. The active row feeds one
    consumer vector/beat, while at most384 bytes/four rows are prefetched.
    Each bank returns one16-byte word/beat after one beat. Physical word indices
    set banks, so a six-word W row can share a bank beat with its successor.
    """
    if not jobs:
        return dict(beats=0, updates=0, memory_words=0, memory_active_beats=0,
                    bank_reads=[0]*8, prefetch_peak_bytes=0, descriptor_peak=0,
                    core_wait_beats=0, mask_reads=0)
    row_words = LANES*lut_bits//128
    pending = []
    next_job, now, active_left, occupied = 0, 0, 0, 0
    peak_bytes, peak_desc, words, mem_beats, updates = 0, 0, 0, 0, 0
    bank_reads = [0]*8
    while next_job < len(jobs) or pending or active_left:
        if not active_left and pending and pending[0]['ready'] <= now:
            item = pending.pop(0)
            occupied -= item['bytes']
            active_left = item['count']
        if active_left:
            active_left -= 1
            updates += 1
        # One mask/descriptor is read per beat; new descriptors may fetch only
        # on the next beat. The active coefficient has its own192-byte register.
        if next_job < len(jobs) and len(pending) < 4:
            ident, count = jobs[next_job]
            nw = 6 if ident < C else row_words
            size = nw*16
            if occupied+size <= 384:
                base = (hidden_block*C*6+ident*6 if ident < C else
                        C*H//16+hidden_block*n_lut*row_words+(ident-C)*row_words)
                remaining = [0]*8
                for w in range(base, base+nw):
                    remaining[w % 8] += 1
                pending.append(dict(count=count, bytes=size, remaining=remaining,
                                    earliest=now+1, ready=10**20))
                occupied += size
                peak_bytes = max(peak_bytes, occupied)
                peak_desc = max(peak_desc, len(pending))
                next_job += 1
        issued = 0
        for bank in range(8):
            for item in pending:
                if item['earliest'] <= now and item['remaining'][bank]:
                    item['remaining'][bank] -= 1
                    bank_reads[bank] += 1
                    issued += 1
                    if not any(item['remaining']):
                        item['ready'] = now+1
                    break
        words += issued
        mem_beats += bool(issued)
        now += 1
        # Jump only over cycles with the current broadcast active and no legal
        # descriptor or memory work. This preserves all finite queue stalls.
        all_ready = all(item['ready'] <= now for item in pending)
        if next_job == len(jobs):
            cannot_enqueue = True
        else:
            ident = jobs[next_job][0]
            next_size = 96 if ident < C else row_words*16
            cannot_enqueue = len(pending) >= 4 or occupied+next_size > 384
        if active_left and all_ready and cannot_enqueue:
            now += active_left
            updates += active_left
            active_left = 0
    # Last issue occupied [now-1,now); latency2 completes at edge(now+1).
    beats = now+1
    assert updates == sum(n for _, n in jobs)
    assert words == sum((6 if i < C else row_words) for i, _ in jobs)
    return dict(beats=beats, updates=updates, memory_words=words,
                memory_active_beats=mem_beats, bank_reads=bank_reads,
                prefetch_peak_bytes=peak_bytes, descriptor_peak=peak_desc,
                core_wait_beats=beats-updates, mask_reads=len(jobs))


def psn_schedule(A):
    """Per-spatial-position service for each10-bit known-Y-live pattern.

    Two Y operand registers prefetch source rows; each s feeds all nonzero t.
    One10x96 U register bank is retained. Final comparators stream in t order;
    the next position begins after this packet. Output transport is explicit.
    """
    out = []
    for pattern in range(1 << T):
        if pattern == 0:
            out.append(dict(beats=1, vector_MACs=0, Y_reads=0, compares=0,
                            known_zero_template_output=1))
            continue
        now, last = 1, [-100]*T  # first one-beat Y prefetch
        macs, reads = 0, 0
        for s in range(T):
            targets = np.flatnonzero(A[:, s])
            if not (pattern >> s & 1) or not len(targets):
                continue
            reads += 1
            for t in targets:
                t = int(t)
                now = max(now, last[t]+4)
                last[t] = now
                now += 1
                macs += 1
        # A separate96-lane comparator may consume completed U while the MAC
        # tail retires. Values with no contributions compare implicit zero.
        compare_time = 0
        for t in range(T):
            compare_time = max(compare_time, last[t]+4, 0)+1
        # One comparison output register, then one128-byte gate/theta packet.
        finish = max(now, compare_time)+1
        out.append(dict(beats=finish, vector_MACs=macs, Y_reads=reads,
                        compares=T, known_zero_template_output=0))
    return out


def stream_frontend(rows, interval, tail, output_bus_beats):
    """A256-byte FIFO,128-byte input transactions, and fixed encoder issue.

    A transaction returns after one beat. Output packets own the bus on their
    actual scheduled beats. Decoder accepts one96-bit source row each interval;
    all16 Hamming candidates are charged for forced encoding, including zeros.
    """
    total_words = math.ceil(rows*12/128)
    fifo, returns_at, read_words, started, ready, now = 0, -1, 0, 0, 0, 0
    peak, blocked = 0, 0
    while started < rows:
        if returns_at == now:
            fifo += 128
            returns_at = -1
            peak = max(peak, fifo)
        if now >= ready and fifo >= 12:
            fifo -= 12
            started += 1
            ready = now+interval
            if started == rows:
                return dict(beats=ready+tail, input_words=read_words,
                            fifo_peak_bytes=peak, bus_conflicts=blocked)
        if read_words < total_words and returns_at < 0 and fifo+128 <= 256:
            if now in output_bus_beats:
                blocked += 1
            else:
                returns_at = now+1
                read_words += 1
        now += 1
        if returns_at < 0 and now < ready and (read_words == total_words or fifo+128 > 256):
            now = ready
    raise AssertionError('unreachable')


def extend_pipeline(state, compute_beats, release, outputs, rows, interval, tail, double_buffer):
    if not state:
        frontend_start, blocked = 0, set()
        previous_end = 0
    else:
        frontend_start = state['compute_start']+(0 if double_buffer else state['release'])
        blocked = {state['compute_start']+o-frontend_start for o in state['outputs']
                   if state['compute_start']+o >= frontend_start}
        previous_end = state['end']
    front = stream_frontend(rows, interval, tail, blocked)
    compute_start = max(previous_end, frontend_start+front['beats'])
    return dict(compute_start=compute_start, end=compute_start+compute_beats,
                release=release, outputs=outputs,
                compute_idle=state.get('compute_idle', 0)+compute_start-previous_end,
                frontend_service=state.get('frontend_service', 0)+front['beats'],
                input_bus_conflicts=state.get('input_bus_conflicts', 0)+front['bus_conflicts'],
                input_words=state.get('input_words', 0)+front['input_words'],
                fifo_peak_bytes=max(state.get('fifo_peak_bytes', 0), front['fifo_peak_bytes']))


def classify(group_words, dict_words, popcounts, min_pop):
    matches = group_words[..., None] == dict_words[None, :, :]
    good = matches & (popcounts[None, :, :] >= min_pop)
    keep = ~good.any(-1)
    # Reconstruct every source support from the original escape bits and the
    # selected dictionary row, independent of actual weight values.
    recovered = np.where(keep, group_words, 0).astype(np.uint16)
    recovered |= np.bitwise_or.reduce(np.where(good, dict_words[None, :, :], 0), -1)
    assert np.array_equal(recovered, group_words)
    return keep, good


def numeric_check(bits, words, W, D, params):
    positions = np.array([0, 1, 31, 32, 127, P-1])
    x = bits[:, positions].reshape(-1, C).astype(np.int64)
    gw = words[:, positions].reshape(-1, 6)
    pc = D.sum(-1)
    dw = (D.astype(np.uint32)*(1 << np.arange(16))).sum(-1).astype(np.uint16)
    keep, good = classify(gw, dw, pc, 2)
    residual = (x.reshape(-1, 6, 16)*keep[..., None]).reshape(-1, C)
    y_ref = x @ W.astype(np.int64).T
    y_lut = residual @ W.astype(np.int64).T
    values = np.einsum('gkc,hgc->gkh', D.astype(np.int64),
                       W.astype(np.int64).reshape(H, 6, 16))
    y_lut += np.einsum('ngk,gkh->nh', good.astype(np.int64), values)
    A = params['temporal_int16'].astype(np.int64)
    u_ref = np.einsum('ts,sph->tph', A, y_ref.reshape(T, -1, H))
    u_lut = np.einsum('ts,sph->tph', A, y_lut.reshape(T, -1, H))
    tau = params['threshold_int64'][:, None, :]
    pos = params['positive_gain'][None, None, :]
    const = params['constant_channels'][None, None, :]
    const_gate = params['constant_gate'][:, None, :]
    g_ref = np.where(const, const_gate, np.where(pos, u_ref >= tau, u_ref <= tau))
    g_lut = np.where(const, const_gate, np.where(pos, u_lut >= tau, u_lut <= tau))
    bound_y = np.abs(W.astype(np.int64)).sum(1)
    bound_u = np.abs(A).sum(1)[:, None]*bound_y[None]
    return dict(spatial_positions=positions.tolist(), Y_values=int(y_ref.size),
                Y_mismatches=int(np.count_nonzero(y_lut != y_ref)),
                U_mismatches=int(np.count_nonzero(u_lut != u_ref)),
                final_gate_mismatches=int(np.count_nonzero(g_lut != g_ref)),
                actual_L_min=int(values.min()), actual_L_max=int(values.max()),
                INT12_L_safe=bool(values.min() >= -2048 and values.max() <= 2047),
                parameter_Y_absolute_bound=int(bound_y.max()),
                parameter_U_absolute_bound=int(bound_u.max()),
                INT24_Y_safe=bool(bound_y.max() < 1 << 23),
                INT48_U_safe=bool(bound_u.max() < 1 << 47),
                theta_source=float(params['theta_source']),
                theta_output=float(params['theta_output']),
                semantics='Wq already contains this student source theta and row scale; output is theta*g. This comparison is not against native FP32 capture.')


def evaluate_frame(variant, stem, dictionary, params, W, psn):
    with np.load(CAP/f'{variant}_{stem}_source.npz') as data:
        shape = tuple(data['shape'])
        packed = data['gate_bits']
    assert shape == (T, P, C)
    bits = np.unpackbits(packed, axis=-1, bitorder='little').astype(bool)
    words = np.ascontiguousarray(packed).view('<u2').reshape(T, P, 6)
    pc = dictionary.sum(-1)
    dw = (dictionary.astype(np.uint32)*(1 << np.arange(16))).sum(-1).astype(np.uint16)
    valid_list = list(zip(*np.nonzero(pc >= 2)))
    valid_ids = np.full((6, 16), -1, np.int32)
    for i, (g, k) in enumerate(valid_list):
        valid_ids[g, k] = i
    n_lut = len(valid_list)
    forced = variant == 'forced_code'
    names = ('bit_broadcast', 'exact_L16', 'exact_L12', 'exact_L16_minpop3')
    counters = {name: Counter() for name in names}
    pipeline = {(name, width): {} for name in names for width in ('main', 'parallel')}
    bank_totals = {name: np.zeros(8, np.int64) for name in names}
    peaks = {name: dict(prefetch_bytes=0, descriptors=0) for name in names}
    common = Counter()
    pophist = np.zeros(17, np.int64)
    num = numeric_check(bits, words, W, dictionary, params)
    assert num['INT12_L_safe'] and num['INT24_Y_safe'] and num['INT48_U_safe']
    assert num['Y_mismatches'] == num['U_mismatches'] == num['final_gate_mismatches'] == 0
    for start in range(0, P, B):
        n = min(B, P-start)
        tile = bits[:, start:start+n].reshape(T*n, C)
        gw = words[:, start:start+n].reshape(T*n, 6)
        raw_counts = tile.sum(0)
        live = tile.any(1)
        live_pattern = (live.reshape(T, n).T*(1 << np.arange(T))).sum(1)
        common['tiles'] += 1
        common['source_active_bits'] += int(tile.sum())
        common['source_groups'] += T*n*6
        common['source_input_bytes'] += T*n*C//8
        common['source_transfer_beats'] += math.ceil(T*n*C/8/128)
        common['Y_live_rows'] += int(live.sum())*4
        common['output_packet_bytes'] += n*128*4
        common['PSN_parameter_read_beats'] += (45+1)*4
        # All source PSN and forced-code projection outputs exist before FC1.
        # Source transport, transpose/encoder, and execution phases serialize;
        # a fixed double-buffer frontend overlap sensitivity is reported later.
        common['frontend_main_beats'] += T*n*(16 if forced else 1)+(2 if forced else 1)
        common['frontend_parallel_beats'] += T*n+(4 if forced else 1)
        if forced:
            common['forced_Hamming_16bit_evaluations'] += T*n*6*16
            common['forced_argmin_updates'] += T*n*6*16
        for pattern in live_pattern:
            spec = psn[int(pattern)]
            for key, value in spec.items():
                common['PSN_'+key] += value*4
        psn_tile = [psn[int(pattern)] for pattern in live_pattern]
        keep2, good2 = classify(gw, dw, pc, 2)
        keep3, good3 = classify(gw, dw, pc, 3)
        hitpc = (good2*pc[None]).sum(-1)
        pophist += np.bincount(hitpc[hitpc > 0], minlength=17)[:17]
        common['useful_exact_groups'] += int(good2.sum())
        common['unpriced_saved_vector_updates'] += int((good2*(pc-1)[None]).sum())
        exact_any = (gw[..., None] == dw[None]).any(-1)
        common['all_exact_groups_including_zero'] += int(exact_any.sum())
        for name in names:
            use_lut = name != 'bit_broadcast'
            min_pop = 3 if name.endswith('minpop3') else 2
            lut_bits = 12 if name == 'exact_L12' else 16
            keep, good = (keep3, good3) if min_pop == 3 else (keep2, good2)
            residual = tile if not use_lut else (tile.reshape(-1, 6, 16)*keep[..., None]).reshape(-1, C)
            counts = raw_counts if not use_lut else residual.sum(0)
            matches = good.sum(0) if use_lut else None
            jobs = jobs_from_counts(counts, matches, valid_ids, min_pop, pc)
            updates = sum(count for _, count in jobs)
            ctr = counters[name]
            ctr['vector_updates'] += updates*4
            ctr['first_vector_assignments'] += int(live.sum())*4
            ctr['vector_adds_after_first'] += (updates-int(live.sum()))*4
            ctr['escape_source_bit_updates'] += int(counts.sum())*4
            ctr['LUT_vector_updates'] += (int(matches.sum()) if use_lut else 0)*4
            ctr['Y_write_bytes'] += updates*4*LANES*3
            ctr['Y_read_for_add_bytes'] += (updates-int(live.sum()))*4*LANES*3
            ctr['mask_bit_writes'] += updates
            # Classification is shared by four h blocks. Forced encoding
            # already produces an exact class index; both variants can reuse it.
            ctr['classifier_extra_beats'] += int(use_lut and not forced)
            if use_lut and not forced:
                ctr['equality_16bit_evaluations'] += T*n*6*16
            if use_lut and lut_bits == 12:
                scheduled = [schedule_fc1(jobs, lut_bits, q, n_lut) for q in range(4)]
            else:
                r = schedule_fc1(jobs, lut_bits, 0, n_lut)
                scheduled = [r]*4  # all W/L16 q strides are multiples of8words
            for r in scheduled:
                ctr['FC1_beats'] += r['beats']
                ctr['coefficient_read_words'] += r['memory_words']
                ctr['coefficient_active_beats'] += r['memory_active_beats']
                ctr['FC1_core_wait_beats'] += r['core_wait_beats']
                ctr['mask_column_reads'] += r['mask_reads']
                bank_totals[name] += r['bank_reads']
                peaks[name]['prefetch_bytes'] = max(peaks[name]['prefetch_bytes'], r['prefetch_peak_bytes'])
                peaks[name]['descriptors'] = max(peaks[name]['descriptors'], r['descriptor_peak'])
            elapsed, release, outputs = 0, 0, []
            for q, r in enumerate(scheduled):
                elapsed += r['beats']
                if q == 3:
                    release = elapsed
                elapsed += 46  # current(t,h) thresholds and zero-Y gate template
                for spec in psn_tile:
                    elapsed += spec['beats']
                    outputs.append(elapsed-1)
            if not use_lut:
                route_columns = C
            elif forced:
                # Static support alphabet, independent of validation frequency.
                escaped_columns = int((dictionary & (pc < min_pop)[..., None]).any(1).sum())
                route_columns = escaped_columns+int((pc >= min_pop).sum())
            else:
                route_columns = C+n_lut
            double_buffer = route_columns <= 96
            for width in ('main', 'parallel'):
                interval = 16 if forced and width == 'main' else 1
                tail = ((2 if width == 'main' else 4) if forced else (2 if use_lut else 1))
                key = name, width
                pipeline[key] = extend_pipeline(pipeline[key], elapsed, release, outputs,
                                                T*n, interval, tail, double_buffer)
            ctr['route_columns'] = route_columns
            ctr['ordinary_double_source_buffer'] = int(double_buffer)
    result = dict(variant=variant, frame=stem, common=dict(common), numeric=num,
                  exact_useful_hit_by_popcount=pophist.tolist(), modes={})
    for name in names:
        use_lut = name != 'bit_broadcast'
        lut_bits = 12 if name == 'exact_L12' else 16
        layout = pool_layout(n_lut, lut_bits, forced, use_lut)
        assert layout['resident_bytes'] <= 131072
        v = dict(counters[name])
        v['coefficient_bank_reads'] = bank_totals[name].tolist()
        v['queue_peaks'] = peaks[name]
        v['resident'] = layout
        v['PSN_beats_including_output'] = common['PSN_beats']
        v['PSN_MAC_scalar_issues'] = common['PSN_vector_MACs']*LANES
        v['PSN_Y_read_bytes'] = common['PSN_Y_reads']*LANES*3
        v['threshold_and_zero_template_read_beats'] = common['PSN_parameter_read_beats']
        fixed = common['source_transfer_beats']+common['PSN_beats']+common['PSN_parameter_read_beats']
        v['serialized_warm_chain_beats'] = v['FC1_beats']+fixed+common['frontend_main_beats']+v['classifier_extra_beats']
        v['warm_chain_beats'] = pipeline[name, 'main']['end']
        v['cold_chain_beats'] = v['warm_chain_beats']+layout['cold_fill_beats']
        v['parallel_forced_encoder_warm_chain_beats'] = pipeline[name, 'parallel']['end']
        v['pipeline'] = {kind: {k: value for k, value in pipeline[name, kind].items()
                                if k not in ('outputs', 'release', 'compute_start')}
                         for kind in ('main', 'parallel')}
        assert v['pipeline']['main']['input_words'] == common['source_transfer_beats']
        assert v['pipeline']['main']['fifo_peak_bytes'] <= 256
        result['modes'][name] = v
    base = result['modes']['bit_broadcast']
    for name, v in result['modes'].items():
        v['same_student_FC1_reduction'] = 1-v['FC1_beats']/base['FC1_beats']
        v['same_student_warm_chain_reduction'] = 1-v['warm_chain_beats']/base['warm_chain_beats']
        v['same_student_cold_chain_reduction'] = 1-v['cold_chain_beats']/base['cold_chain_beats']
        v['same_student_parallel_encoder_chain_reduction'] = (1-v['parallel_forced_encoder_warm_chain_beats']
                                                               /base['parallel_forced_encoder_warm_chain_beats'])
    return result


def aggregate(frames):
    out = {}
    for variant in sorted({r['variant'] for r in frames}):
        rows = [r for r in frames if r['variant'] == variant]
        modes = {}
        base = sum(r['modes']['bit_broadcast']['warm_chain_beats'] for r in rows)
        cold_base = sum(r['modes']['bit_broadcast']['cold_chain_beats'] for r in rows)
        fbase = sum(r['modes']['bit_broadcast']['FC1_beats'] for r in rows)
        pbase = sum(r['modes']['bit_broadcast']['parallel_forced_encoder_warm_chain_beats'] for r in rows)
        for name in rows[0]['modes']:
            keys = ('FC1_beats', 'FC1_core_wait_beats', 'vector_updates', 'vector_adds_after_first',
                    'warm_chain_beats', 'cold_chain_beats', 'coefficient_read_words',
                    'PSN_beats_including_output', 'PSN_MAC_scalar_issues',
                    'Y_write_bytes', 'Y_read_for_add_bytes', 'PSN_Y_read_bytes',
                    'parallel_forced_encoder_warm_chain_beats')
            values = {k: sum(r['modes'][name][k] for r in rows) for k in keys}
            values['same_student_warm_chain_reduction'] = 1-values['warm_chain_beats']/base
            values['same_student_cold_chain_reduction'] = 1-values['cold_chain_beats']/cold_base
            values['same_student_FC1_reduction'] = 1-values['FC1_beats']/fbase
            values['same_student_parallel_encoder_chain_reduction'] = 1-values['parallel_forced_encoder_warm_chain_beats']/pbase
            values['per_frame_chain_reduction_min'] = min(r['modes'][name]['same_student_warm_chain_reduction'] for r in rows)
            values['per_frame_chain_reduction_max'] = max(r['modes'][name]['same_student_warm_chain_reduction'] for r in rows)
            modes[name] = values
        out[variant] = dict(frames=len(rows), source_active_bits=sum(r['common']['source_active_bits'] for r in rows),
                            useful_exact_groups=sum(r['common']['useful_exact_groups'] for r in rows),
                            source_groups=sum(r['common']['source_groups'] for r in rows),
                            unpriced_saved_vector_updates=sum(r['common']['unpriced_saved_vector_updates'] for r in rows),
                            modes=modes)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='+', default=['teacher', 'sparse', 'exact_dictionary', 'forced_code'])
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--output', default='support_service_result.json')
    args = parser.parse_args()
    start = time.monotonic()
    D = np.load(CAP/'dictionary.npy')
    assert D.shape == (6, 16, 16)
    for group in D:
        assert len(np.unique(group, axis=0)) == 16
    saved = read_torch(ALG/'integer_s0_valid825/integer_parameters.pt')
    prefix = 'sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.'
    params = saved[prefix]
    assert int(params['temporal_fractional_bits']) == 14
    psn = psn_schedule(params['temporal_int16'])
    run = json.loads((CAP/'run.json').read_text())
    files = [Path(p).stem for p in run['validation_files'][:args.frames]]
    frames = []
    for variant in args.variants:
        W = (params['weight_int8'] if variant == 'teacher' else
             read_torch(CAP/f'{variant}_weight_int8.pt'))
        assert W.shape == (H, C)
        for stem in files:
            row = evaluate_frame(variant, stem, D, params, W, psn)
            frames.append(row)
            quick = {k: round(v['same_student_warm_chain_reduction']*100, 4)
                     for k, v in row['modes'].items()}
            print(variant, stem, quick, flush=True)
            result = dict(kind='finite-resource service model; no RTL/PPA claim',
                          resources='support_service_resources.json',
                          boundary='source PSN ready through optional forced encoder, FC1, Q14 PSN and theta-g output; upstream source PSN excluded identically within each student',
                          data='train32 dictionary; four real students; fixed first10 validation frames from parent experiment',
                          arithmetic='Same trained integer student reference; source-theta-folded W, Yi24, Aq16 Q14, U48 and signed integer thresholds',
                          thresholds='native FP32 identity is not claimed; parent measures the new model AEE separately',
                          static_table_formation_vector_adds=int(np.maximum(D.sum(-1).astype(int)-1, 0).sum()),
                          static_table_formation_scalar_adds=int(np.maximum(D.sum(-1).astype(int)-1, 0).sum()*H),
                          frames=frames, aggregate=aggregate(frames), elapsed_wall_s=time.monotonic()-start)
            (HERE/args.output).write_text(json.dumps(serializable(result), ensure_ascii=False, indent=2)+'\n')
    print('FINISHED', len(frames), time.monotonic()-start, flush=True)


if __name__ == '__main__':
    main()
