"""Finite FC1/PSN services for the same six-S2 temporal-code student.

Read existing captures only. Service counters and external-bus lower bounds
are deliberately separate: this is not a complete accelerator cycle model.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import json
import math
import time
import numpy as np
import support_service_model as svc
from temporal_basis_service_model import psn_lookup

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
T, TILE, LANES, BANKS = 10, 32, 96, 7


def routes_for(tag, books, basis):
    D = books[tag+'_dictionary'].astype(np.int64)
    A = books[tag+'_A_int16'].astype(np.int64)
    groups, seen = [], {}
    for t in range(T):
        key = tuple(D[:, t])
        if not any(key):
            continue
        if key not in seen:
            seen[key] = len(groups)
            groups.append([])
        groups[seen[key]].append(t)
    dec = D[:, [g[0] for g in groups]].T
    folded = np.stack([A[:, g].sum(1) for g in groups], axis=1)
    routes = {
        'fold6': (dec, folded),
        'onehot7': (np.eye(8, dtype=np.int64)[1:], (A@D.T)[:, 1:]),
        'signed6': (basis[tag+'_coordinates_int8'].astype(np.int64).T,
                    basis[tag+'_B_int32'].astype(np.int64)),
    }
    for decode, coeff in routes.values():
        assert np.array_equal(coeff@decode, A@D.T)
        assert coeff.min() >= -32768 and coeff.max() <= 32767
        assert np.all(np.isin(decode, [-1, 0, 1]))
    return routes, groups


def add_metrics(dst, src, factor=1):
    for key, value in src.items():
        if key.startswith('peak_'):
            dst[key] = max(dst[key], int(value))
        else:
            dst[key] += int(value)*factor


def evaluate(codes, route, hidden_blocks, platforms=('L1', 'L7_plane', 'L7_striped')):
    decode, coeff = route
    R = decode.shape[0]
    lookup = psn_lookup(coeff)
    totals = {name: Counter() for name in platforms}
    psn_windows = []
    live_hist = np.zeros(1 << R, dtype=np.int64)
    for start in range(0, len(codes), TILE):
        tile = codes[start:start+TILE]
        n = len(tile)
        features = decode[:, tile]
        active = features != 0
        live = active.any(axis=2)                 # [logical slot,p]
        patterns = (live.T.astype(np.int64)*(1 << np.arange(R))).sum(1)
        hist = np.bincount(patterns, minlength=1 << R)
        live_hist += hist
        psn = Counter()
        for pat, count in enumerate(hist):
            if not count:
                continue
            q = lookup[pat]
            psn['psn_local_service_beats'] += int(count)*(q['beats']-1)
            psn['psn_vector_MACs'] += int(count)*q['vector_MACs']
            psn['psn_state_vector_reads'] += int(count)*q['slot_reads']
            psn['psn_vector_compares'] += int(count)*q['comparisons']
            psn['zero_template_positions'] += int(count)*(pat == 0)
        # Last output packet is removed from lookup service; all packets use
        # the shared external bus below. U may be held until that bus accepts.
        psn_windows.append(dict(positions=n, **psn))
        flat = active.reshape(R*n, -1)
        total_by_c = flat.sum(0)
        plane = active.sum(1)
        logical_addr = np.arange(R)[:, None]*TILE+np.arange(n)[None, :]
        striped_bank = (logical_addr % BANKS).reshape(-1)
        stripe = np.stack([flat[striped_bank == b].sum(0) for b in range(BANKS)])
        tick_counts = {
            'L1': total_by_c,
            'L7_plane': plane.max(0),
            'L7_striped': stripe.max(0),
        }
        vec_updates = int(total_by_c.sum())
        first_writes = int(live.sum())
        for name, ticks in tick_counts.items():
            if name not in totals:
                continue
            jobs = [(int(c), int(ticks[c])) for c in np.flatnonzero(ticks)]
            # All six actual W matrices have no all-zero96-lane vectors.
            # Both42,752-byte stripe bases align to an eight-bank boundary,
            # so all16 H96 blocks have this identical physical word service.
            service = svc.schedule_fc1(jobs, hidden_block=0)
            m = dict(psn)
            m.update(
                fc1_bank_service_beats=service['beats']+1,  # valid-bit clear
                fc1_broadcast_issue_beats=service['updates'],
                fc1_bank_wait_and_tail_beats=service['core_wait_beats']+1,
                coefficient_word_reads=service['memory_words'],
                coefficient_bank_active_beats=service['memory_active_beats'],
                coefficient_vector_requests=service['mask_reads'],
                fc1_state_vector_writes=vec_updates,
                fc1_state_vector_reads=vec_updates-first_writes,
                fc1_cold_vector_writes=first_writes,
                fc1_signed_vector_updates=int((features < 0).sum()),
                source_code_column_reads=service['mask_reads'],
                tau_and_template_bank_read_beats=46,
                peak_prefetch_bytes=service['prefetch_peak_bytes'],
                peak_pending_descriptors=service['descriptor_peak'],
                peak_live_state_vectors=first_writes,
            )
            for b, count in enumerate(service['bank_reads']):
                m['coefficient_bank'+str(b)+'_word_reads'] = count
            m['serialized_local_service_beats'] = (
                m['fc1_bank_service_beats']+46+m['psn_local_service_beats'])
            add_metrics(totals[name], m, hidden_blocks)
    return totals, psn_windows, live_hist


def front_and_bus(P, C, hidden_blocks, source_rows, psn_windows, code_bits=3, rom=True):
    sizes = [min(TILE, P-start) for start in range(0, P, TILE)]
    raw_bytes = sum(n*C*source_rows//8 for n in sizes)
    raw_beats = sum(math.ceil(n*C*source_rows/8/128) for n in sizes)
    fills = len(sizes)*hidden_blocks
    output_packets = P*hidden_blocks             # T10*96bits +8B header
    # This is an opportunity bound only, not a simulated shared-bus schedule.
    # Current PSN uses register coefficients after46-bank-beat tau preload.
    # Each window must also accommodate current outputs; a new spatial tile
    # requests its raw source only at the last H stripe of the previous tile.
    fits, transitions, min_slack = 0, 0, None
    for tile_idx, (n, window) in enumerate(zip(sizes, psn_windows)):
        for h in range(hidden_blocks):
            last = tile_idx == len(sizes)-1 and h == hidden_blocks-1
            if last:
                continue
            next_raw = (math.ceil(sizes[tile_idx+1]*C*source_rows/8/128)
                        if h == hidden_blocks-1 else 0)
            slack = window['psn_local_service_beats']-n-334-next_raw
            min_slack = slack if min_slack is None else min(min_slack, slack)
            fits += slack >= 0
            transitions += 1
    return dict(
        source_encoder_issue_beats=len(sizes)*C if rom else 0,
        source_column_store_issue_beats=len(sizes)*C,
        peak_source_encoder_32lane_rom_bytes=32*((1 << source_rows)*3//8) if rom else 0,
        source_raw_required_gate_bytes=raw_bytes,
        source_raw_bus_beats=raw_beats,
        compact_source_array_bytes=P*C*code_bits//8,
        coefficient_stripe_fills=fills,
        coefficient_fill_bytes=fills*42752,
        coefficient_fill_bus_beats=fills*334,
        output_bus_packets=output_packets,
        output_bus_bytes=output_packets*128,
        external_bus_beats_lower_bound=fills*334+raw_beats+output_packets,
        first_cold_stripe_bus_beats=334,
        next_stripe_psn_windows_fitting_bus_lower_bound=fits,
        next_stripe_psn_windows=transitions,
        minimum_psn_window_bus_slack_beats=min_slack,
    )


def time_row_control(args):
    """Separate model competitor: retain actual source time rows, no K8 map."""
    started = time.monotonic()
    files = sorted(args.capture.glob('*.npz'))
    assert len(files) == 60
    params = svc.read_torch(args.parameters)
    books = np.load(ROOT/'algorithm/stage2_temporal_codes/codebooks.npz')
    totals = defaultdict(Counter)
    modules = defaultdict(lambda: defaultdict(Counter))
    frames, definitions = [], {}
    bus_total = Counter()
    for index, path in enumerate(files):
        tag = path.stem.rsplit('_', 1)[1]
        block = int(tag[-1])
        prefix = f'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.{block}.mlp.'
        q = params[prefix]
        W, A = q['weight_int8'], q['temporal_int16'].astype(np.int64)
        H, C = W.shape
        assert (C, H) == (384, 1536)
        assert np.all(q['positive_gain']) and not np.any(q['constant_channels'])
        assert not np.any(~np.any(W.reshape(16, LANES, C), axis=1))
        svc.C, svc.H, svc.LANES = C, H, LANES
        with np.load(path) as capture:
            shape = tuple(capture['shape'])
            assert shape == (10, 1200, 384)
            gates = np.unpackbits(capture['gate_bits'], axis=-1, count=C, bitorder='little')
            selected = np.flatnonzero(capture['active_rows'])
        dictionary = books[tag+'_dictionary']
        expected = np.flatnonzero(np.any(dictionary != dictionary[0], axis=0))
        assert np.array_equal(selected, expected)
        assert len(selected) == (7 if block == 5 else 6)
        assert not np.any(gates[np.setdiff1d(np.arange(10), selected)])
        R = len(selected)
        # Packed R-bit masks are only a software representation of the direct
        # emitted gate rows. No projection, nearest-code search or ROM here.
        signatures = np.zeros(shape[1:], dtype=np.uint8)
        for r, t in enumerate(selected):
            signatures |= gates[t] << r
        decode = ((np.arange(1 << R)[None, :] >> np.arange(R)[:, None]) & 1)
        coeff = A[:, selected]
        assert coeff.min() >= -32768 and coeff.max() <= 32767
        result, windows, live_hist = evaluate(signatures, (decode, coeff), 16,
                                              platforms=('L1', 'L7_striped'))
        bus = front_and_bus(1200, C, 16, R, windows, code_bits=R, rom=False)
        add_metrics(bus_total, {k: v for k, v in bus.items()
                               if k != 'minimum_psn_window_bus_slack_beats'})
        w_bound = int(np.abs(W.astype(np.int64)).sum(1).max())
        assert w_bound < (1 << 23)
        assert int(np.abs(coeff).sum(1).max())*w_bound < (1 << 47)
        definitions[tag] = dict(
            source_time_rows=selected.tolist(), logical_slots=R,
            source_theta=float(q['theta_source']), output_theta=float(q['theta_output']),
            compact_source_bits_per_coordinate=R, compact_source_frame_bytes=1200*C*R//8,
            compact_source_double_tile_bytes=2*TILE*C*R//8,
            extra_double_tile_bytes_vs_3bit_code=2*TILE*C*(R-3)//8,
            source_column_read_bits=TILE*R, source_encoder_ROM_bytes=0,
            active_wide_state_bytes=R*TILE*96*3, allocated_wide_state_bytes=64512,
            negative_gain_channels=0, constant_gain_channels=0,
            INT24_abs_bound=w_bound, INT48_abs_bound=int(np.abs(coeff).sum(1).max())*w_bound,
        )
        record = dict(capture=path.name, source_active_gates=int(gates.sum()),
                      live_slot_histogram=live_hist.tolist(), bus=bus,
                      services={k: dict(v) for k, v in result.items()})
        for platform, metrics in result.items():
            add_metrics(totals[platform], metrics)
            add_metrics(modules[tag][platform], metrics)
        frames.append(record)
        print(f'row-control {index+1}/60 {path.name} elapsed={time.monotonic()-started:.1f}s', flush=True)
    report = dict(
        scope='Separate ordinary source-time-row-pruned model competitor, not the K8 student workload. Same local resource/service assumptions, real captured theta*g supports, original saved A columns and tau.',
        capture_directory=str(args.capture), captures=len(files), definitions=definitions,
        shared_resource_model=str(HERE/'stage2_temporal_service_result.json'),
        resource_differences={
            'source': 'No encoder ROM or code projection. Store actual retained6/7 gate rows in a6/7-bit coordinate array,192/224-bit column read per beat. Double-buffer source storage18,432/21,504B, versus9,216B for3-bit codes. Column-store throughput is charged, producer/transpose remains outside boundary.',
            'wide_state': 'Same seven physical32-word96-lane INT24 banks. First five modules use6 logical slots; last uses7 and does not merge original t0/t7. L1 and L7 striped remain separate platforms.',
            'PSN': 'Original full-rank A, only deleted source columns absent. All ten output rows, original tau and theta_output preserved; real FC2/BN2/shortcut are unchanged by this hardware representation and outside service model.',
            'coefficients': 'Same two42,752B H96 stripes and128KiB pool,8x16B1RW bank ports; no extra full-W cache.',
        },
        aggregate_services={k: dict(v) for k, v in totals.items()},
        aggregate_bus_and_frontend=dict(bus_total),
        modules={k: {s: dict(m) for s, m in v.items()} for k, v in modules.items()},
        frames=frames, runtime_seconds=time.monotonic()-started,
        interpretation='Pair AEE with these component-service/state counts. Cross-model ratios are design tradeoffs, not a same-workload acceleration or full network cycle result.')
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print('wrote '+str(args.output), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('code-student', 'time-row-control'), default='code-student')
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--parameters', type=Path, default=ROOT/'algorithm/stage2_temporal_codes/integer_parameters.pt')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.model == 'time-row-control':
        args.capture = args.capture or ROOT/'algorithm/time_row_control_valid825/capture'
        args.output = args.output or HERE/'stage2_time_row_control_service_result.json'
        return time_row_control(args)
    args.capture = args.capture or ROOT/'algorithm/stage2_deployment_diverse_capture/capture'
    args.output = args.output or HERE/'stage2_temporal_service_result.json'
    started = time.monotonic()
    files = sorted(args.capture.glob('*.npz'))
    assert len(files) == 60
    books = np.load(ROOT/'algorithm/stage2_temporal_codes/codebooks.npz')
    basis = np.load(ROOT/'algorithm/stage2_temporal_codes/signed_basis.npz')
    params = svc.read_torch(args.parameters)
    compiled, definitions = {}, {}
    for block in range(6):
        tag = f's2b{block}'
        prefix = f'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.{block}.mlp.'
        q = params[prefix]
        W = q['weight_int8']
        H, C = W.shape
        assert (C, H) == (384, 1536)
        assert not np.any(~np.any(W.reshape(H//LANES, LANES, C), axis=1))
        # Actual saved six-module compiler output, not an assumption about
        # ATLIF in general. No runtime direction/constant-mask fetch needed.
        assert np.all(q['positive_gain']) and not np.any(q['constant_channels'])
        svc.C, svc.H, svc.LANES = C, H, LANES
        routes, groups = routes_for(tag, books, basis)
        w_bound = int(np.abs(W.astype(np.int64)).sum(1).max())
        state_bound = {r: w_bound for r in routes}
        u_bound = {r: int(np.abs(coef).sum(1).max())*w_bound
                   for r, (_, coef) in routes.items()}
        assert max(state_bound.values()) < (1 << 23)
        assert max(u_bound.values()) < (1 << 47)
        compiled[tag] = routes
        definitions[tag] = dict(
            C=C, H=H, H96_blocks=H//LANES,
            folded_original_time_groups=groups,
            source_encoder_required_time_rows=basis[tag+'_source_PSN_rows'].tolist(),
            source_theta=float(q['theta_source']), output_theta=float(q['theta_output']),
            negative_gain_channels=0, constant_gain_channels=0,
            all_zero_W96_vectors=0,
            INT24_abs_bound=state_bound, INT48_abs_bound=u_bound,
            route_coefficients={r: coef.tolist() for r, (_, coef) in routes.items()},
        )
    totals, modules, frames = defaultdict(Counter), defaultdict(lambda: defaultdict(Counter)), []
    bus_totals = defaultdict(Counter)
    for index, path in enumerate(files):
        tag = path.stem.rsplit('_', 1)[1]
        with np.load(path) as capture:
            codes = capture['codes']
            assert codes.shape == (1200, 384)
            assert np.array_equal(capture['dictionary_words'], books[tag+'_words'])
            raw_hist = capture['raw_source_signature_histogram']
        record = dict(capture=path.name, P=len(codes), code_histogram=np.bincount(codes.ravel(), minlength=8).tolist(),
                      raw_zero_signature_count=int(raw_hist[0]), services={}, common_bus={})
        for route, definition in compiled[tag].items():
            result, windows, live_hist = evaluate(codes, definition, 16)
            front = front_and_bus(len(codes), 384, 16,
                                 len(definitions[tag]['source_encoder_required_time_rows']), windows)
            add_metrics(bus_totals[route], {k: v for k, v in front.items()
                                          if k != 'minimum_psn_window_bus_slack_beats'})
            record['common_bus'][route] = front
            record.setdefault('live_slot_histograms', {})[route] = live_hist.tolist()
            for platform, metrics in result.items():
                key = route+'/'+platform
                add_metrics(totals[key], metrics)
                add_metrics(modules[tag][key], metrics)
                record['services'][key] = dict(metrics)
        frames.append(record)
        print(f'{index+1}/60 {path.name} elapsed={time.monotonic()-started:.1f}s', flush=True)
    comparisons = {}
    for platform in ('L1', 'L7_plane', 'L7_striped'):
        base = totals['fold6/'+platform]
        comparisons[platform] = {}
        for route in ('onehot7', 'signed6'):
            cur = totals[route+'/'+platform]
            comparisons[platform][route] = {
                key+'_ratio_to_fold6': cur[key]/base[key] for key in
                ('fc1_bank_service_beats', 'psn_local_service_beats',
                 'serialized_local_service_beats', 'fc1_state_vector_writes')}
    report = dict(
        scope='Existing same-student four-sequence ten-frame captures, all six S2 blocks. Finite local services and shared-bus lower bounds; not full cycles, RTL, PPA or system speed.',
        capture_directory=str(args.capture), captures=len(files), definitions=definitions,
        resources=dict(
            geometry=dict(T=10, P=1200, C=384, H=1536, spatial_tile=32, hidden_tile=96, hidden_tiles=16, spatial_tiles=38),
            shared_wide_state=dict(physical_banks=7, words_per_bank=32, word_lanes=96, lane_bits=24, allocated_bytes=64512,
                                  plane='bank=slot,row=p', striped='addr=slot*32+p;bank=addr%7;row=addr//7',
                                  L1='One shared96-lane update port and adder; layout does not change service.',
                                  L7='Seven independent96-lane state read/write/update ports and adders; common W broadcast, independent destinations. Plane and striped are fixed layouts for all routes; seven ports are a different resource platform from L1.',
                                  update='Each enabled bank has one96-lane read and one96-lane write per beat. II1 with two-beat completion and forwarding; per-slot valid bits avoid wide zero initialization. First writes still pay service. Negative signed coordinates use subtraction at the same service rate. These are RF/register state ports, not claimed1RW SRAM ports.'),
            coefficients=dict(pool_bytes=131072, stripe_count=2, stripe_bytes=42752, total_stripe_bytes=85504,
                              W_bytes_per_stripe=36864, tau_bytes_per_stripe=5760, zero_template_bytes_per_stripe=128,
                              banks=8, bank_word_bytes=16, port='one1RW port per bank', read_latency_beats=1,
                              stripe_schedule='P32 outer,H96 inner; all16 stripes refilled for every spatial tile. No full-W residence.',
                              prefetch_bytes=384, pending_descriptors=4, active_W_register_bytes=96,
                              pending_and_active_mask_max_bytes=280,
                              mask_hardware='32 parallel3-bit decoders, fixed layout wiring and seven32-bit priority queues; same provisioning for every route. Decoder logic and fanout are not physically mapped.'),
            source=dict(double_code_tile_bytes=9216, common_zero_column_bitmaps_bytes=96,
                        raw_input_fifo_bytes=256, shared_max_encoder_ROM_bytes=1536,
                        encoder='32 parallel ROMs, one C-column of at most32 coordinates per beat; required raw6/7 time decisions. ROM replication is charged. Original PSN producer and its layout cost lie outside this boundary.'),
            PSN=dict(mac_lanes=96, operand_bits=[16, 24], accumulation_bits=48, initiation_interval=1, latency=4,
                     U_register_bytes=5760, tau_register_bytes=5760, operand_register_bytes=576,
                     transform_coefficient_register_bytes=144, zero_template_register_bytes=128,
                     output_packet_bytes=128, output_fifo_packets=2,
                     service='Common single vector MAC/comparator pipeline for all platforms. Static nonzero coefficient schedule and support-live slots; no speculative numeric cancellation.'),
            external_bus=dict(bytes_per_beat=128, shared_by=['coefficient refills', 'raw next source', 'T10 gate output packets'],
                              hiding='Next stripe writes may overlap only current PSN register service after tau preload. Window slack is a necessary bandwidth opportunity, not a complete arbitration proof.')),
        metric_notes=dict(
            serialized_local_service_beats='FC1 finite-bank service +46 tau/template bank beats per stripe +PSN local service. Stages serialized; excludes external-bus arbitration, source encode and producer. Comparisons are component service ratios only.',
            psn_local_service_beats='Existing PSN schedule minus final external packet beat. May stall on a full output FIFO; that global stall is not simulated.',
            fc1_state_vector_writes='Actual96-lane state updates; L7 broadcast issue beats count max per-bank destinations for each W, never total updates/7.',
            valid_clear='One beat clears224 support-valid bits per spatial/hidden tile; no free copy of a wide tensor.',
            theta='Source theta was folded by the saved integer compiler; final theta*g preserved. This same trained integer student is not frozen FP32 ep34 equivalence.',
            comparator='Actual six modules have zero negative-gain and zero constant-gain channels, so static comparison direction is positive throughout. This is checked from saved parameters; no general positive-gain assumption.',
            source='All three routes use the same encoded3-bit source. Frontend input is required source decisions in column/tile order; actual producer reordering/transpose is not modeled.',
            finite_width='INT24 and INT48 range bounds checked from actual W and coefficients; no source floating tensor or intermediate fullY is silently cached.',
            omitted='No all-stage overlap timeline, physical timing/area/energy, encoder/queue circuit delay, source PSN production, real FC2/BN2 or shortcut service. No RTL or network speed claim.'),
        aggregate_services={k: dict(v) for k, v in totals.items()},
        aggregate_bus_and_encoder={k: dict(v) for k, v in bus_totals.items()},
        fixed_layout_comparisons=comparisons,
        modules={m: {k: dict(v) for k, v in rows.items()} for m, rows in modules.items()},
        frames=frames, runtime_seconds=time.monotonic()-started)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(comparisons, ensure_ascii=False), flush=True)
    print('wrote '+str(args.output), flush=True)


if __name__ == '__main__':
    main()
