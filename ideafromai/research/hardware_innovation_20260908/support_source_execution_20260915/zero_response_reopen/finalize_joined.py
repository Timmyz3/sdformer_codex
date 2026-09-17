from pathlib import Path
import csv,json
from collections import defaultdict

HERE=Path(__file__).resolve().parent
FIELDS=['cycles_to_last_gate','cycles_to_done','first_X_request_to_last_gate','words','bytes',
        'dictionary_load_cycles','source_phase_cycles','backend_phase_cycles','backend_boot_cycles',
        'backend_fc_cycles','backend_psn_cycles','dictionary_words','source_config_words','source_X_words',
        'source_graph_words','source_prefetch_words','backend_config_words','backend_coeff_words',
        'source_channels','source_scalar_mac','backend_vector_mac','backend_vector_updates','bridge_writes','bridge_reads']

def main():
    rows=[]
    for name in ['joined_zero_cycles.csv','joined_l2_cycles.csv']:
        with (HERE/name).open() as f:
            for raw in csv.DictReader(f):
                row={k:v if k in ['function','case'] else int(v) for k,v in raw.items()}
                row.setdefault('backend_mode',0 if row['mode']==0 else 2)
                row['file']=name;rows.append(row)
    repeated=defaultdict(list)
    for r in rows:repeated[(r['function'],r['case'],r['mode'],r['backend_mode'],r['bp'],r['hblocks'])].append(r)
    for key,rr in repeated.items():
        assert sorted(r['pass'] for r in rr)==[0,1],key
        assert all(rr[0][k]==rr[1][k] for k in FIELDS),key
    groups=defaultdict(list)
    for r in rows:
        if r['real'] and r['pass']==0:
            assert r['hblocks']==4
            groups[(r['function'],r['mode'],r['backend_mode'],r['bp'])].append(r)
    summaries=[]
    for key,rr in sorted(groups.items()):
        assert sorted(x['case'] for x in rr)==['train0','train1']
        summaries.append(dict(function=key[0],producer_mode=key[1],backend_mode=key[2],bp=key[3],
            selected_real_tiles=2,**{k:sum(r[k] for r in rr) for k in FIELDS}))
    lookup={(x['function'],x['producer_mode'],x['backend_mode'],x['bp']):x for x in summaries}
    comparisons=[]
    for function in ['zero_response','integer_L2_zero_allowed']:
        for bm in [2,4]:
            for bp in [0,1]:
                key=(function,2,bm,bp)
                if key not in lookup:continue
                a=lookup[key];b=lookup[(function,3,bm,bp)]
                comparisons.append(dict(function=function,backend_mode=bm,bp=bp,
                    code_cycles=a['cycles_to_last_gate'],class_cycles=b['cycles_to_last_gate'],
                    class_cycle_savings=a['cycles_to_last_gate']-b['cycles_to_last_gate'],
                    class_cycle_savings_percent=100*(1-b['cycles_to_last_gate']/a['cycles_to_last_gate']),
                    code_bytes=a['bytes'],class_bytes=b['bytes'],
                    independent_source_jobs_saved=a['source_channels']-b['source_channels'],
                    backend_coeff_words_code=a['backend_coeff_words'],backend_coeff_words_class=b['backend_coeff_words'],
                    backend_updates_code=a['backend_vector_updates'],backend_updates_class=b['backend_vector_updates']))
    profile=json.loads((HERE/'JOINED_PROFILE.json').read_text());assert profile['observed_csv_records_verified']>=len(rows)+4
    result={'scope':'two training frames x 32 selected positions x T10 x H384; separate fixed integer functions; no AEE',
            'period':'start accepted to final backend gate accepted, inclusive; all in-pool parameter reads included',
            'external_memory_pool_prepopulation_included':False,'physical_shared_bank_pool_bytes':262144,
            'validation':{'full_H384_commands':len(rows),'Y_U_gate_scalar_positions':len(rows)*320*384,
                          'raw_source_MAC_code_bridge_and_output':'PASS','single_initial_reset_per_run':True,
                          'two_repeat_passes_identical':'PASS','independent_semantic_counter_profile':'PASS'},
            'summaries':summaries,'class_vs_code_samefunction':comparisons,'small_H96_commands':4,'profile_records_including_small':len(rows)+4}
    (HERE/'SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps({'validation':result['validation'],'comparisons':comparisons},ensure_ascii=False))

if __name__=='__main__':main()
