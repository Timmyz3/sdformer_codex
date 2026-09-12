"""Old R24 exports bound to actual source/preview and dual-consumer execution.

Only the completed anchor U/V block is replaced by a caller's callback.
Candidate gold is inspected after execution, never passed to the callback.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
OPEN = HERE.parents[1]
EXPORT = OPEN / 'stage_20260912/algorithm/hardware_exports'
REP = OPEN / 'breadth_20260912/representation'
sys.path.insert(0, str(OPEN / 'breadth_20260912/hardware'))
import packed_weights as shared

stage, consumer, integer = shared.stage, shared.consumer, shared.integer
I, DIR, LAT16, UPDATED = integer.I, integer.DIR, integer.LAT16, integer.UPDATED
PED_U, PED_V, PROJ, GATE = integer.PED_U, integer.PED_V, integer.PROJ, integer.GATE
read24, pack24, difference = integer.read24, integer.pack24, integer.difference
MODES = ('fixed_q8', 'affine_q8', 'full_g_q8')
PARENT_BINDINGS = {}


def read_npz(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def validate_parent(axis, label, data, live, q):
    """Read-only provenance check; these old arrays never drive execution."""
    old = stage.common.FULL / 'capture' / axis
    old_live = read_npz(old / 'live_parameters.npz')
    old_q = read_npz(old / 'parameters.npz')
    old_data = read_npz(old / '000_zurich_city_09_a_0001.npz')
    assert live.keys() == old_live.keys()
    assert all(np.array_equal(live[k], old_live[k]) for k in live)
    assert q.keys() == old_q.keys()
    changed = sorted(k for k in q if not np.array_equal(q[k], old_q[k]))
    assert changed == ['U_ped_q16', 'V_ped_q16'], changed
    fields = ('I24', 'sn1_gate', 'preview_Z_shared', 'preview_shared_raw',
              'preview_BN1_Y', 'sn2_gate', 'updated_I24', 'proj_gate')
    assert np.array_equal(data['window_geometry_json'], old_data['window_geometry_json'])
    for field in fields:
        assert np.array_equal(data[label+'_'+field], old_data[label+'_'+field]), field
    assert q['U_ped_q16'].shape == (24, 96) and q['V_ped_q16'].shape == (96, 24)
    blob, base, _ = shared.ordinary_coeffs(q, False)
    assert len(blob) == 41024
    report = dict(axis=axis, window=label, old_R24_onepass_export=str(EXPORT/axis),
                  original_source_preview_capture=str(old), live_parameters_equal=len(live),
                  changed_deployed_constant_keys=changed,
                  identical_source_preview_integer_capture_fields=list(fields),
                  preview_rank=32, PED_rank=24,
                  original_coefficient_image_bytes=len(blob), original_coefficient_base=base,
                  source_program='Existing axis_fused_program.json from two_stage_writeback; actual I24 loads, arithmetic and gate SW64 remain executed.',
                  source_constants_and_preview_live_parameters_exact=True)
    PARENT_BINDINGS[(axis, label)] = report
    return report


def load(axis, label):
    """Return (data, live, q, params), params[mode] contains integer D/c/step."""
    assert axis in ('ordinary', 'lifting_raw') and label in ('corner', 'interior')
    folder = EXPORT / axis
    data = read_npz(folder / '000_zurich_city_09_a_0001.npz')
    live = read_npz(folder / 'live_parameters.npz')
    q = read_npz(folder / 'deployed_constants.npz')
    archive = read_npz(REP / 'parameters' / (axis+'.npz'))
    params = {mode: {key: archive[mode+'_'+key].astype(np.int64)
                     for key in ('D', 'c', 'step')} for mode in MODES}
    validate_parent(axis, label, data, live, q)
    return data, live, q, params


def run_prefix(machine_class, data, live, axis, label, stress=False):
    """Execute true I24 -> source -> K864 preview/sn2; return (value, report, m)."""
    runner = stage.run_windows
    old_machine, old_directory = runner.Machine, runner.build_nrv
    try:
        runner.Machine = machine_class
        runner.build_nrv = stage.preview_directory
        value, report, m = runner.window(data, live, label, False, stress, True, axis)
    finally:
        runner.Machine, runner.build_nrv = old_machine, old_directory
    archived = json.loads((stage.common.FULL/'preview_sn2_chain/windows.json').read_text())['axes'][axis][label]['expanded_fp32']['checks']
    assert report['checks'].keys() == archived.keys()
    compatibility = []
    for endpoint, fields in archived.items():
        assert report['checks'][endpoint].keys() == fields.keys()
        for field, expected in fields.items():
            actual = report['checks'][endpoint][field]
            if actual == expected:
                continue
            allowed = bool(field == 'rms' and abs(actual-expected) <= 2*abs(np.spacing(expected)))
            assert allowed, (endpoint, field, actual, expected)
            compatibility.append(dict(endpoint=endpoint, field=field, actual=actual,
                                      archived=expected, accepted_summary_only_2ULP=True))
    assert report['checks']['Z']['differences'] == 0
    assert report['checks']['sn2']['differences'] == 0
    assert report['source_program']['checks']['differences'] == 0
    assert 'sn1_halo_input' not in m.stages and 'sn2_gate_egress' not in m.stages
    report['aggregate_compatibility'] = compatibility
    report['export_binding'] = PARENT_BINDINGS.get((axis, label))
    m.forward_i24 = True
    return value, report, m


def _original_u_gold(data, q, label, geo):
    # Called only after consumer execution; never an input supplier.
    oy, ox = geo['gate_origin']; out_y, out_x = geo['output_origin']
    dy, dx = 2*out_y-oy, 2*out_x-ox
    raw = data[label+'_updated_I24'][:, :, dy:dy+8:2, dx:dx+8:2].astype(np.int64)
    acc = np.einsum('rk,tkij->trij', q['U_ped_q16'].astype(np.int64), raw, dtype=np.int64)
    shift = int(q['U_ped_exponent'])
    if shift > 0:
        quo, rem = np.divmod(acc, 1 << shift)
        acc = quo + ((2*rem > 1 << shift) | ((2*rem == 1 << shift) & ((quo & 1) != 0)))
    else:
        acc = acc << -shift
    return np.clip(acc, -(1 << 23), (1 << 23)-1)


def consumer_run(data, q, label, callback, *, machine, stress=False, gold=None,
                 handoff_native=False):
    """Clone consumer_ranked.run with only anchor U/V replaced.

    callback(m, base, q, positions, geo) returns U/PED integer arrays in
    (count,10,24)/(count,10,96). It must also store actual PED at PED_V.
    gold may contain only final 'U_ped' and/or 'continuous' check arrays.
    """
    assert machine is not None and q['U_ped_q16'].shape == (24, 96)
    if gold is not None:
        assert set(gold).issubset({'U_ped', 'continuous'}) and 'continuous' in gold
    geo = json.loads(str(data['window_geometry_json']))[label]
    h, w = geo['gate_shape']; oy, ox = geo['gate_origin']; out_y, out_x = geo['output_origin']
    m = machine; begin = m.time
    before = dict(m.count); prior_stages = dict(m.stages)
    # Keep original addresses and full old image; callback may lazily cold-fill
    # its independent high-address constants after this replacement.
    blob, base, bounds = shared.ordinary_coeffs(q, False)
    m.phase = 'integer_coefficient_replace'; m.dma_input(blob, 0, True)
    sy, sx = geo['source_origin']
    updated = data[label+'_I24'][:, :, oy-sy:oy-sy+h, ox-sx:ox-sx+w].astype(np.int64).copy()
    continuous = np.empty((10, 96, 4, 4), np.int64)
    u_output = np.empty((10, 24, 4, 4), np.int64)
    spill = bytearray(16*10*96*3)
    anchors = {(2*(out_y+dy), 2*(out_x+dx)) for dy in range(4) for dx in range(4)}
    rows = []
    for y in range(oy, oy+h):
        for anchor in (True, False):
            xs = [x for x in range(ox, ox+w) if ((y, x) in anchors) == anchor]
            for i in range(0, len(xs), 2):
                positions = [(y, x) for x in xs[i:i+2]]; count = len(positions); start = m.time
                integer.feed_identity(m, data, label, geo, positions)
                if anchor:
                    n, live = stage.addresses.directory(m, geo, positions)
                    integer.sparse_u(m, n, count, base, q)
                    shared.ordinary_dense(m, base, 'F', 16, 96, LAT16, UPDATED, count,
                                          int(q['F_exponent']), live)
                    integer.merge(m, base, count)
                input_base = UPDATED if anchor else I
                actual_updated = read24(m, input_base, (count, 10, 96))
                integer.projection_gates(m, base, geo, positions, input_base)
                gate_ready = m.time
                u_ready = None
                if anchor:
                    # No data/gold/expected U reaches the callback. Its only
                    # incoming numerical operands are the real machine state.
                    m.encoded_u_ready = None
                    actual_u, actual_ped = callback(m, base, q, positions, geo)
                    assert np.asarray(actual_u).shape == (count, 10, 24)
                    assert np.asarray(actual_ped).shape == (count, 10, 96)
                    assert np.issubdtype(np.asarray(actual_u).dtype, np.integer)
                    assert np.issubdtype(np.asarray(actual_ped).dtype, np.integer)
                    assert np.array_equal(actual_ped, read24(m, PED_V, (count, 10, 96)))
                    u_ready = m.encoded_u_ready
                    m.phase = 'continuous_PED_egress'
                    tile_payload = bytearray()
                    for off in range(0, count*10*96*3, 32):
                        payload = b''.join(m.read_word(PED_V+off+j) for j in range(0, 32, 8))
                        assert len(payload) == 32
                        tile_payload.extend(payload)
                        for _ in range(5): m.advance(tag='PED_DMA_output_slots')
                for ip, (yy, xx) in enumerate(positions):
                    updated[:, :, yy-oy, xx-ox] = actual_updated[ip]
                    if anchor:
                        dy, dx = yy//2-out_y, xx//2-out_x
                        continuous[:, :, dy, dx] = actual_ped[ip]
                        u_output[:, :, dy, dx] = actual_u[ip]
                        stride = 10*96*3; offset = (dy*4+dx)*stride
                        spill[offset:offset+stride] = tile_payload[ip*stride:(ip+1)*stride]
                rows.append(dict(positions=positions, anchor=anchor, start=start,
                                 gate_ready=gate_ready, U_ready=u_ready, end=m.time))
    if not handoff_native:
        m.phase = 'projection_gate_egress'
        for off in range(0, h*w*192, 32):
            payload = b''.join(m.read_word(PROJ+off+j) for j in range(0, 32, 8))
            assert len(payload) == 32
            for _ in range(5): m.advance(tag='gate_DMA_output_slots')
    words = np.frombuffer(m.state, '<u2', count=h*w*96, offset=PROJ).reshape(h, w, 96).copy()
    gate = np.stack([(words >> t) & 1 for t in range(10)]).transpose(0, 3, 1, 2).astype(bool)
    # Final-only oracle selection: projection/updated always remain tied to
    # the original export, while lossy representation may change U/PED.
    checks = dict(updated=difference(updated, data[label+'_updated_I24']),
                  projection_gate=difference(gate, data[label+'_proj_gate']),
                  PED=difference(continuous, data[label+'_continuous_q24'] if gold is None else gold['continuous']))
    if gold is None or 'U_ped' in gold:
        expected_u = _original_u_gold(data, q, label, geo) if gold is None else gold['U_ped']
        checks['U_ped'] = difference(u_output, expected_u)
    assert all(v['differences'] == 0 for v in checks.values()), checks
    assert bytes(spill) == pack24(continuous.transpose(2, 3, 0, 1))
    report = dict(PED_rank=24, service_slots=m.time-begin,
                  stages={k: v-prior_stages.get(k, 0) for k, v in m.stages.items() if v-prior_stages.get(k, 0)},
                  counts={k: v-before.get(k, 0) for k, v in m.count.items() if v-before.get(k, 0)},
                  checks=checks, coefficients_bytes=len(blob), coefficient_base=base,
                  legal_accumulator48_bounds=bounds, state_high_water=PROJ+h*w*192,
                  state_high_water_scope='Original wrapper buffers only; callback reports its own scratch maximum.',
                  rows=rows, PED_spill_bytes=len(spill), PED_spill_width=96,
                  input_contract='Each raw I24 pixel is 2880 contiguous bytes, spatial/T/C order; raw input DMA remains paid.',
                  common_baseline='Original source, complete K864/Conv2/merge and projection gates; callback owns only U/V.',
                  V_coefficients_retained_across_native=False, full_chain_closed=False,
                  final_gold_only=True, actual_PED_SRAM_egress_checked=True,
                  U_ready_contract='Optional callback m.encoded_u_ready; null when no aggregate U completion is declared.')
    return dict(updated=updated, continuous=continuous, gate=gate, U_ped=u_output,
                PED_spill=np.frombuffer(spill, np.uint8).copy()), report


def original_p1_callback(m, base, q, positions, geo):
    """Binding smoke only: original resident dense, one position at a time.

    Materializes U, so this is NOT the later optimized P1 keep-Z denominator.
    """
    count = len(positions)
    for ip in range(count):
        shared.ordinary_dense(m, base, 'U_ped', 96, 24, UPDATED+ip*2880,
                              PED_U+ip*720, 1, int(q['U_ped_exponent']))
        if ip == count-1: m.encoded_u_ready = m.time
        shared.ordinary_dense(m, base, 'V_ped', 24, 96, PED_U+ip*720,
                              PED_V+ip*2880, 1, int(q['V_ped_exponent']), bias='PED_bias')
    return read24(m, PED_U, (count, 10, 24)), read24(m, PED_V, (count, 10, 96))


def smoke():
    axis, label = 'ordinary', 'interior'
    data, live, q, params = load(axis, label)
    # Read-only provenance validation of the other supported parent also.
    load('lifting_raw', label)
    print('BINDING_PREFIX_START', axis, label, flush=True)
    _, producer, m = run_prefix(shared.PackedMachine, data, live, axis, label, False)
    boundary = m.time
    print('BINDING_PREFIX_DONE', boundary, producer['checks'], flush=True)
    _, report = consumer_run(data, q, label, original_p1_callback, machine=m)
    assert sum(m.stages.values()) == m.time
    assert report['service_slots'] == m.time-boundary
    assert not m.count.get('DMA_output_slots', 0)
    result = dict(axis=axis, window=label, stress=False, producer=producer, consumer=report,
                  service_slots=m.time, producer_end=boundary, same_machine_handoff=True,
                  counts=dict(m.count), stages=dict(m.stages),
                  port_bytes=dict(SR64=8*m.count['SR64_reads'], SW64=8*m.count['SW64_writes'],
                                  CR256=32*m.count['CR256_reads'], CW256=32*m.count['CW256_writes']),
                  binding_checks=list(PARENT_BINDINGS.values()), quantizer_modes=list(params),
                  smoke_only_materialized_P1_not_keep_Z_denominator=True,
                  exact_integer_endpoints=True, no_native_or_global_BN=True, complete=True)
    # Output is stdout only: root owns stage result files; this agent owns
    # binding.py and BINDING.md and may redirect its smoke log separately.
    summary = {key: result[key] for key in ('axis', 'window', 'stress', 'service_slots',
               'producer_end', 'same_machine_handoff', 'port_bytes', 'quantizer_modes',
               'exact_integer_endpoints', 'no_native_or_global_BN', 'complete')}
    summary['consumer_service_slots'] = report['service_slots']
    summary['integer_checks'] = report['checks']
    summary['source_checks'] = producer['source_program']['checks']
    summary['preview_checks'] = producer['checks']
    summary['aggregate_compatibility'] = producer['aggregate_compatibility']
    print('BINDING_SMOKE_JSON '+json.dumps(summary), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', required=True)
    parser.parse_args()
    smoke()
