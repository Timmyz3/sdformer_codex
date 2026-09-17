from pathlib import Path
from collections import defaultdict
import csv,json
HERE=Path(__file__).resolve().parent
FIELDS=['cycles_to_last_gate','cycles_to_done','first_X_request_to_last_gate','words','bytes',
        'dictionary_load_cycles','source_phase_cycles','backend_phase_cycles','backend_boot_cycles','backend_fc_cycles','backend_psn_cycles',
        'dictionary_words','source_config_words','source_X_words','source_graph_words','source_prefetch_words','backend_config_words','backend_coeff_words',
        'source_produced_pairs','source_channel_refs','source_scalar_mac','backend_vector_mac','backend_vector_updates','bridge_writes','bridge_reads','req_stall','out_stall']

def main():
    data={}
    for file,n in [('small.csv',32),('expanded.csv',136)]:
        rows=[{k:v if k in ('case','function') else int(v) for k,v in r.items()} for r in csv.DictReader((HERE/file).open())]
        assert len(rows)==n
        assert all(r['frontier']==1 and r['backend_mode']==4 and r['hblocks']==4 for r in rows)
        data[file]=rows
    small=data['small.csv'];expanded=data['expanded.csv']
    ref={(r['case'],r['mode'],r['bp']):r for r in expanded}
    for row in small:
        assert all(row[k]==ref[(row['case'],row['mode'],row['bp'])][k] for k in FIELDS)
    sets=[('first_two_training_frames',[0,1]),('selection_frame0',[0]),('unselected_training_frames',list(range(1,32))),('all_32_training_frames',list(range(32)))]
    summaries=[]
    for name,frames in sets:
        for bp in (0,1):
            count={}
            for mode in (2,3):
                rr=[r for r in expanded if r['real'] and int(r['case'][5:]) in frames and r['bp']==bp and r['mode']==mode]
                assert len(rr)==len(frames)
                count[mode]={k:sum(r[k] for r in rr) for k in FIELDS}
            a,b=count[2],count[3]
            for k in ['backend_config_words','backend_coeff_words','backend_vector_mac','backend_vector_updates','source_config_words','bridge_writes','bridge_reads']:assert a[k]==b[k]
            summaries.append(dict(scope=name,frames=frames,bp=bp,code=a,response_class=b,
                                  cycles_saved=a['cycles_to_last_gate']-b['cycles_to_last_gate'],
                                  cycle_savings_percent=100*(1-b['cycles_to_last_gate']/a['cycles_to_last_gate']),
                                  bytes_saved=a['bytes']-b['bytes']))
    prof=json.loads((HERE/'PROFILE.json').read_text());assert prof['status']=='PASS' and prof['records_verified']==168
    result=dict(status='PASS',scope='Wz fixed integer function, full H384, P32/T10, first2+2diagnostics repeated; expanded32+2diagnostics continuous, no network AEE',
                strong_controls=dict(frontier=1,resident_X_bytes=128,graph_cache_bytes=128,active_prefetch=1,first_P_cold_remaining_31_reuse_config=True,
                                     X_and_graph_dynamic_state_cleared_each_P=True,root_word6174_both_modes=True,zero_aware_backend4_both_modes=True),
                physical_shared_pool_bytes=262144,source_backend_distinct_multipliers=106,initial_pool_external_fill_modeled=False,
                period='accepted top start to accepted last gate inclusive; no multiplication of leaf speedups',
                validation=dict(full_H384_commands=168,Y_U_gate_scalar_positions=168*320*384,small_repeats_and_overlap_with_expanded_equal=True,
                                independent_semantic_frontier_reference_X_and_backend_work=True,single_initial_reset_each_executable_launch=True),
                summaries=summaries)
    (HERE/'SUMMARY.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps(dict(status='PASS',summaries=[{k:v for k,v in x.items() if k not in ('code','response_class','frames')} for x in summaries])))

if __name__=='__main__':main()
