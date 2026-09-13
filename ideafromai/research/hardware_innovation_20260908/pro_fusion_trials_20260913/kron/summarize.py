from pathlib import Path
import json,csv
import numpy as np
HERE=Path(__file__).resolve().parent
f=np.load(HERE/'fixture.npz');fit=json.load((HERE/'fit_results.json').open());rtl=json.load((HERE/'rtl_results.json').open())
bounds={}
for n in (1,2):
    a=f[f'k{n}_a_q7'];b=f[f'k{n}_b_q8'];ib=abs(a).sum(axis=2)*2**23
    ob=np.einsum('rbv,ra->ab',abs(b),ib)
    assert ib.max()<2**31 and ob.max()<2**47
    bounds[f'kron{n}']=dict(all_signed24_intermediate_abs_bound=int(ib.max()),
        all_signed24_accumulator_abs_bound=int(ob.max()),
        a_range=[int(a.min()),int(a.max())],b_range=[int(b.min()),int(b.max())],
        expanded_w_range=[int(f[f'k{n}_expanded_wq15'].min()),int(f[f'k{n}_expanded_wq15'].max())])
directbound=int(abs(f['original_wq15']).sum(axis=1).max()*2**23)
assert directbound<2**47
resource=dict(shared_modes=True,MACs=8,MAC='signed32 * signed16 -> signed48 product, signed48 accumulator',
    CR_read='One packed128-bit word per MAC cycle',
    source_read='Up to4 distinct32-bit local words per MAC cycle, broadcast to8 lanes; same mux/read budget in direct mode',
    external_input='192bits =8 signed24 words, ready/valid',external_output='192bits =8 signed24 words, ready/valid',
    local_state_bits={'coefficients':288*128,'coefficient_word_live_mask':288,'bias':96*24,'source':24*32,'latent':16*32,'output_accumulators':96*48,'MAC_accumulators':8*48},
    whole_word_skip='288bit live mask generated from cfg_data OR in the same configuration beat; common bounded24bit next-live selector,6/4 populated entries for factor loops; ascending nonzero CR words only',
    coefficient_config='One128-bit write per cycle; bias one24-bit write per cycle in same IDLE phase',
    data_write='8 lanes per COMMIT, either latent32 or output48; src8 lanes on input beat; no same-cycle conflicts',
    implementation='Synthesizable flop/mux arrays, asynchronous CR reads. SRAM mapping, MAC clock period and PPA unmeasured.',
    bounds=bounds,direct_all_signed24_abs_bound=directbound)
resource['local_state_bits_total_without_control']=sum(resource['local_state_bits'].values())
(HERE/'resource_contract.json').write_text(json.dumps(resource,ensure_ascii=False,indent=2)+'\n')
static_words={}
for name in ['original','expanded_k1','kron1','expanded_k2','kron2']:
    words=np.loadtxt(HERE/(name+'_coeff.txt'),dtype=np.int64).reshape(-1,8)
    zero=np.where(~words.any(axis=1))[0]
    static_words[name]=dict(loaded_CR_words=int(len(words)),zero_CR_words=int(len(zero)),zero_word_indices=zero.tolist())
(HERE/'static_zero_words.json').write_text(json.dumps(static_words,indent=2)+'\n')
rows=[]
for r in rtl['cases']:
    assert sum(r[k] for k in ['IDLE_config_start','LOAD','CLEAR','MAC','COMMIT','OUTPUT'])==r['total_cycles']
    assert r['mismatches']==0 and r['checked_values']==30720
    assert r['coefficient_read_words']==r['MAC']
    base=next(x for x in rtl['cases'] if x['mode']=='original' and x['stress']==r['stress'])
    same_mode={'kron1':'expanded_k1','kron2':'expanded_k2'}.get(r['mode'],r['mode'])
    same=next(x for x in rtl['cases'] if x['mode']==same_mode and x['stress']==r['stress'])
    row=dict(r);row['reduction_vs_direct']=1-r['total_cycles']/base['total_cycles'];row['speedup_vs_direct']=base['total_cycles']/r['total_cycles']
    row['same_function_direct_cycles']=same['total_cycles'];row['reduction_vs_same_function_direct']=1-r['total_cycles']/same['total_cycles'];row['speedup_vs_same_function_direct']=same['total_cycles']/r['total_cycles'];rows.append(row)
with (HERE/'benefits.csv').open('w',newline='') as fcsv:
    w=csv.DictWriter(fcsv,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
summary=dict(RTL_functional_mismatches=0,RTL_checked_values=sum(x['checked_values'] for x in rows),
    actual_input_vectors=320,weight_fits='one fixed layout,1/2 terms, ordinary rank1/2 same parameter count',
    tests_include=['original V','expanded integer Kron1','factorized integer Kron1','expanded integer Kron2','factorized integer Kron2','each with and without deterministic backpressure'],
    function_change=True,AEE_evaluated=False,PPA_evaluated=False,source_capture_verified=fit['source_replay'],
    total_state_bits_without_control=resource['local_state_bits_total_without_control'],
    static_zero_word_control=static_words,old_control='old_control/rtl_results.json; direct did not skip whole zero CR words',
    candidate_status='Fast complete local RTL; large local distortion; Kron2 weaker than equal-parameter rank2 on these inputs; A-based port, X=0 not new architecture claim',cases=rows)
(HERE/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'verified_values':summary['RTL_checked_values'],'state_bits':summary['total_state_bits_without_control'],'cases':[(x['mode'],x['stress'],x['total_cycles']) for x in rows]},indent=2))
