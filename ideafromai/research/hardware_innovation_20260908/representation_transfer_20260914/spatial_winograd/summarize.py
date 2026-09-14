from pathlib import Path
import json

H = Path(__file__).resolve().parent
raw = json.loads((H / 'raw_verification.json').read_text())
consumer = json.loads((H / 'consumer_verification.json').read_text())
profiles = json.loads((H / 'profiles.json').read_text())
metadata = json.loads((H / 'sequence_fixture_metadata.json').read_text())
os_root = H.parent / 'spatial_r16_direct/q11'
os_summary = json.loads((os_root / 'CONSUMER_SUMMARY.json').read_text())
assert raw['passed'] and consumer['passed'] and os_summary['passed']

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

comparison = dict(passed=True, scope='cycle RTL; fixed Q11 function; sameframe64 and 18-sequence first-frame probes', sets={})
for stage in ['sequences', 'held', 'disjoint', 'short', 'small']:
    names = (H / f'{stage}.txt').read_text().splitlines()
    table = []
    for stall in [0, 1]:
        for repeat in [0, 1]:
            tag = f's{stall}_r{repeat}'
            a = consumer['sets'][stage]['m0_' + tag]
            b = consumer['sets'][stage]['m1_' + tag]
            gain = a['service'] - b['service']
            record = dict(stall=stall, model_resident=bool(repeat),
                          ordinary=a['service'], winograd=b['service'],
                          saved_cycles=gain, saved_percent=100*gain/a['service'],
                          raw_ordinary=raw['sets'][stage]['m0_' + tag]['service'],
                          raw_winograd=raw['sets'][stage]['m1_' + tag]['service'])
            if stage in ['held', 'disjoint', 'sequences']:
                os_stage = 'sequence' if stage == 'sequences' else stage
                os_set = next(r for r in os_summary['sets'][os_stage]['summaries'] if r['stall'] == stall)
                record['direct_os'] = os_set['warm_stream_service' if repeat else 'cold_stream_service']
            table.append(record)
    a = consumer['sets'][stage]['m0_s0_r0']
    b = consumer['sets'][stage]['m1_s0_r0']
    predicted = a['q2_issues'] - b['q2_issues'] - 2832*len(names) - 192
    assert predicted == a['service'] - b['service']
    comparison['sets'][stage] = dict(tiles=len(names), services=table,
        ordinary_Q2_vector_MACs=a['q2_issues'], winograd_Q2_vector_MACs=b['q2_issues'],
        avoided_Q2_vector_MACs=a['q2_issues']-b['q2_issues'],
        added_fixed_cycles=2832*len(names), extra_cold_model_cycles=192,
        ordinary_weight_vectors=a['q2_words'], winograd_weight_vectors=b['q2_words'])

# Match the actual native inputs and all output oracles used by the independent
# expanded-W OS arm. This is a function/workload check, not a hash archive.
same_function_fixtures = 0
for stage in ['held', 'disjoint', 'sequences']:
    os_stage = 'sequence' if stage == 'sequences' else stage
    ours = (H / f'{stage}.txt').read_text().splitlines()
    theirs = (os_root / f'{os_stage}.txt').read_text().splitlines()
    assert len(ours) == len(theirs)
    for a, b in zip(ours, theirs):
        for name in ['source', 'origin', 'gold', 'identity', 'j', 'wide', 'i24']:
            av = [int(v, 16) for v in (Path(a) / f'{name}.hex').read_text().split()]
            bv = [int(v, 16) for v in (Path(b) / f'{name}.hex').read_text().split()]
            assert av == bv, (stage, a, b, name)
        same_function_fixtures += 1
comparison['same_function_direct_control_fixture_matches'] = same_function_fixtures

sequence_rows = []
base = {stall: rows(H / f'consumer_sequences_m0_s{stall}.jsonl')[:36] for stall in [0, 1]}
cand = {stall: rows(H / f'consumer_sequences_m1_s{stall}.jsonl')[:36] for stall in [0, 1]}
os_rows = {stall: rows(os_root / f'consumer_sequence_{stall}.jsonl')[:36] for stall in [0, 1]}
for i, meta in enumerate(metadata):
    e = profiles[meta['fixture']]
    gain = e['q2_issues'] - e['winograd_mac'] - 2832
    assert gain == base[0][i]['c_cycles'] - cand[0][i]['c_cycles']
    sequence_rows.append(dict(**meta, ordinary_MACs=e['q2_issues'],
        winograd_MACs=e['winograd_mac'], fixed_added_cycles=2832,
        ready_saved_cycles=gain,
        BP_saved_cycles=base[1][i]['c_cycles']-cand[1][i]['c_cycles'],
        ordinary_ready=base[0][i]['c_cycles'], winograd_ready=cand[0][i]['c_cycles'],
        direct_OS_ready=os_rows[0][i]['c_cycles']))
sequence_pairs = []
for i in range(18):
    pair = sequence_rows[2*i:2*i+2]
    sequence_pairs.append(dict(sequence=i, file=pair[0]['file'],
        edge_saved_cycles=pair[0]['ready_saved_cycles'],
        interior_saved_cycles=pair[1]['ready_saved_cycles'],
        ready_saved_cycles=sum(r['ready_saved_cycles'] for r in pair),
        BP_saved_cycles=sum(r['BP_saved_cycles'] for r in pair)))
comparison['sequence_breakdown'] = dict(
    tile_wins=sum(r['ready_saved_cycles'] > 0 for r in sequence_rows),
    tile_losses=sum(r['ready_saved_cycles'] < 0 for r in sequence_rows),
    sequence_pair_wins=sum(r['ready_saved_cycles'] > 0 for r in sequence_pairs),
    sequence_pair_losses=sum(r['ready_saved_cycles'] < 0 for r in sequence_pairs),
    edge_saved_cycles=sum(r['ready_saved_cycles'] for r in sequence_rows[::2]),
    interior_saved_cycles=sum(r['ready_saved_cycles'] for r in sequence_rows[1::2]),
    per_pair=sequence_pairs,
    convention='start-to-I24 intrinsic/BP differences, excludes cold model delta192 and equal source/start fees')
(H / 'sequence_results.jsonl').write_text(''.join(json.dumps(r, separators=(',', ':'))+'\n' for r in sequence_rows))
(H / 'comparison.json').write_text(json.dumps(comparison, separators=(',', ':'))+'\n')

contract = json.loads((H.parent / 'spatial_r16_rtl/resource_contract.json').read_text())
producer = contract['producer']
producer['modes'] = {'0': 'ordinary factor Q11', '1': 'Q1 unchanged; F(2,3) only on continuous Q2'}
producer['z_scalar_payload_bits_by_mode'] = {'0': 15, '1': 16}
producer['storage_bytes'].update(q2_static=9984, q1_q2_static_live=168,
    q2_cache=416, q2_block_live=4, q2_remaining=4,
    transform_tail=32, M1_M2_M3=96)
producer['added_storage_bytes_vs_original_factor'] = dict(
    q2_static_capacity=2496, q2_cache=104, q2_live=24,
    block_live=1, remaining=1, transform_tail=32, M1_M2_M3=96,
    listed_data_and_support_total=2754)
producer['additional_control'] = 'tx integer register32bit, mode_q latch, expanded support selection/address/subtraction muxes; debug counters/monitors/assertions excluded from functional storage claim'
producer['z_port'] = 'one common address; Q1/ZSCAN/XD reads 256bit all banks, BASE_MAC reads selected32bit bank only; XD reads both original words before two ALU-backed whole-word writes; no separate D array'
producer['psum_port'] = 'one common address, exclusive read/write; stripe0 stores all480 rows; stripe1 actual read then separate shared-ALU add then write; drain only after960 stores'
producer['local_access'] = 'original native window/gate masks; position_live uses up to four distributed selected-position lookups, replacing three; one selected104bit qcache vector per BASE_MAC'
producer['resident_model_lifecycle'] = 'same mode/model may serve no-reset changing source after I24 done; both modes run in separate reset/configuration processes with actual Q2 layout loads. Cross-mode no-reset reload not exercised; no online model versioning'
producer['fairness'] = 'both modes instantiate the union data/holding capacity; mode0 leaves extra capacity idle. This is common execution resources/capacity, not an equal-area claim for separately optimized arms'
producer['coefficient_layout'] = 'one Q2 table: original576 vectors OR transformed768 vectors. Original g is not also resident in Winograd mode; coefficients are fixed model constants loaded over the common configuration port'
admission = json.loads((H / 'admission.json').read_text())
stats = json.loads((H.parent / 'spatial_winograd_inputs/stats.json').read_text())
contract['model_admission'].update(q1_min=admission['q1_range'][0], q1_max=admission['q1_range'][1],
    q2_min=admission['q2_range'][0], q2_max=admission['q2_range'][1], q2_quantization_signed_bits=11,
    z_signed15_bound=[min(admission['z_lower']), max(admission['z_upper'])],
    p_any_prefix_abs_bound=max(admission['p_any_prefix_abs']),
    transformed_D_abs_bound=stats['transformed_Z_abs'],
    transformed_coefficient_range=stats['transformed_q2_range'],
    M_any_prefix_abs_bound=stats['M_prefix_abs'],
    reconstruction_abs_bound=stats['output_reconstruction_abs'],
    reconstruction='each stripe inverse sum even; exact arithmetic shift1; no intermediate RNE')
contract['service_convention']['factor_configuration_beats_per_model'] = {'0': 1152, '1': 1344}
contract['service_convention']['full_consumer_configuration_beats_per_model'] = {'0': 1176, '1': 1368}
contract['service_convention']['cold_stream'] = 'one actual model load, then every tile source/origin/start plus raw or last-I24 service; 36/64 tile stream, not per-tile cold'
contract['service_convention']['resident_stream'] = 'second no-reset traversal; same model resident; every tile source/origin/start still paid'
contract['service_convention']['BP'] = 'same per-command calendar: source denies n%11==3; W denies n%7 in{2,3}; consumer raw denies n%13==4; identity denies n%17 in{4,5,6}; output denies n%5 in{1,2}'
contract['verification_scope'] = 'per-value native Z, transformed D, raw/J20/wide64/I24; M update path static review and signed prefix assertions, no per-M RTL oracle; ordered output and BP hold; I24 done before next source/model config'
(H / 'resource_contract.json').write_text(json.dumps(contract, indent=2)+'\n')

verification = dict(passed=True, raw_commands=raw['commands'], consumer_commands=consumer['commands'],
    total_commands=raw['commands']+consumer['commands'],
    raw_values_total=raw['raw_values']+consumer['raw_J20_wide_I24_each'],
    J20_wide64_I24_values_each=consumer['raw_J20_wide_I24_each'],
    Z_values_total=raw['z_values']+consumer['z_values'],
    D_values_total=raw['d_values']+consumer['d_values'],
    counter_state_repeat_checks=raw['checks']+consumer['checks'],
    independent_oracle_fixtures=len(profiles), same_function_direct_control_fixture_matches=same_function_fixtures,
    cross_sequence_tiles=len(metadata),
    scope=contract['verification_scope'])
(H / 'verification.json').write_text(json.dumps(verification, separators=(',', ':'))+'\n')
print(json.dumps(verification))
for name, v in comparison['sets'].items():
    print(json.dumps(dict(stage=name, services=v['services'])))
