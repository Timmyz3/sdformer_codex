from pathlib import Path
import json

H = Path(__file__).resolve().parent
NEW = H.parent.parent
metadata = json.loads((H.parent/'sequence_fixture_metadata.json').read_text())
comparison = dict(passed=True, arms={}, cross_function_comparison='different functions; assess with separate network quality, never as bit-identical acceleration')
verification = dict(passed=True, raw_commands=0, consumer_commands=0,
                    raw_values_total=0, J20_wide64_I24_each=0,
                    Z_values_total=0, D_values_total=0, counter_state_checks=0)
for arm in ['moment','native_tap']:
    root = H/arm
    raw = json.loads((root/'raw_verification.json').read_text())
    con = json.loads((root/'consumer_verification.json').read_text())
    admission = json.loads((root/'admission.json').read_text())
    stats = json.loads((NEW/'spatial_winograd_pruning'/arm/'stats.json').read_text())
    assert raw['passed'] and con['passed'] and admission['passed']
    result = dict(sets={}, model_bounds=stats['bounds'],
                  native_zero_N8_groups=stats['native_zero_N8_groups'],
                  winograd_zero_N8_groups=stats['winograd_zero_N8_groups'])
    for stage in ['sequences','held','disjoint','small']:
        names = (root/f'{stage}.txt').read_text().splitlines()
        services = []
        for stall in [0,1]:
            for repeat in [0,1]:
                a=con['sets'][stage][f'm0_s{stall}_r{repeat}']
                b=con['sets'][stage][f'm1_s{stall}_r{repeat}']
                gain=a['service']-b['service']
                services.append(dict(stall=stall,model_resident=bool(repeat),ordinary=a['service'],
                    general_winograd=b['service'],saved_cycles=gain,saved_percent=100*gain/a['service'],
                    raw_ordinary=raw['sets'][stage][f'm0_s{stall}_r{repeat}']['service'],
                    raw_general_winograd=raw['sets'][stage][f'm1_s{stall}_r{repeat}']['service']))
        a,b=[con['sets'][stage][f'm{m}_s0_r0'] for m in [0,1]]
        assert a['service']-b['service']==a['q2_issues']-b['q2_issues']-2832*len(names)-192
        result['sets'][stage]=dict(tiles=len(names),services=services,
            ordinary_Q2_MACs=a['q2_issues'],general_winograd_Q2_MACs=b['q2_issues'],
            avoided_Q2_MACs=a['q2_issues']-b['q2_issues'],fixed_added_cycles=2832*len(names),
            ordinary_Q2_weight_vectors=a['q2_words'],general_winograd_Q2_weight_vectors=b['q2_words'],
            extra_cold_configuration_cycles=192)
    seq={}
    for m in [0,1]:
        for s in [0,1]:
            seq[m,s]=[json.loads(l) for l in (root/f'consumer_sequences_m{m}_s{s}.jsonl').read_text().splitlines()][:36]
    per_tile=[]
    for i,meta in enumerate(metadata):
        per_tile.append(dict(index=i,file=meta['file'],frame_index=meta['frame_index'],tile_id=meta['tile_id'],
            output_origin_yx=meta['output_origin_yx'],
            ready_saved_cycles=seq[0,0][i]['c_cycles']-seq[1,0][i]['c_cycles'],
            BP_saved_cycles=seq[0,1][i]['c_cycles']-seq[1,1][i]['c_cycles']))
    result['sequence_tile_wins']=sum(r['ready_saved_cycles']>0 for r in per_tile)
    result['sequence_tile_losses']=sum(r['ready_saved_cycles']<0 for r in per_tile)
    result['sequence_edge_saved_cycles']=sum(r['ready_saved_cycles'] for r in per_tile[::2])
    result['sequence_interior_saved_cycles']=sum(r['ready_saved_cycles'] for r in per_tile[1::2])
    (root/'sequence_results.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in per_tile))
    quality=NEW/'quality'/arm/'deployed_diverse/spatial_integer_summary.json'
    if quality.exists():
        q=json.loads(quality.read_text())
        result['independent_network_diverse']=dict(source=str(quality),frames=q['frames'],AEE_frame_mean=q['AEE_frame_mean'])
    comparison['arms'][arm]=result
    verification['raw_commands']+=raw['commands']
    verification['consumer_commands']+=con['commands']
    verification['raw_values_total']+=raw['raw_values']+con['raw_J20_wide_I24_each']
    verification['J20_wide64_I24_each']+=con['raw_J20_wide_I24_each']
    verification['Z_values_total']+=raw['z_values']+con['z_values']
    verification['D_values_total']+=raw['d_values']+con['d_values']
    verification['counter_state_checks']+=raw['checks']+con['checks']
verification['total_commands']=verification['raw_commands']+verification['consumer_commands']
verification['independent_function_fixtures']=356
verification['scope']='unchanged generic RTL: actual source/Z/D/raw/FP-J20/wide/I24 and BP; M prefix assertions plus static path, no M monitor; no-reset source changes, mode changes require separately configured processes'
(H/'verification.json').write_text(json.dumps(verification,separators=(',',':'))+'\n')
(H/'comparison.json').write_text(json.dumps(comparison,separators=(',',':'))+'\n')
print(json.dumps(verification))
for arm,x in comparison['arms'].items():
    print(arm,'sequence wins/losses',x['sequence_tile_wins'],x['sequence_tile_losses'])
    for stage,v in x['sets'].items():
        print(json.dumps(dict(arm=arm,stage=stage,services=v['services'],ordinary_MAC=v['ordinary_Q2_MACs'],general_MAC=v['general_winograd_Q2_MACs'])))
