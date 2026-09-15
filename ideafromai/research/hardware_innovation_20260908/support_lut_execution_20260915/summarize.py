import csv
import json
from pathlib import Path

p=Path(__file__).resolve().parent
rows=list(csv.DictReader((p/'rtl_cycles.csv').open()))
names=['direct_bit','LUT16','LUT10','LUT10_zero_directory','LUT10_content_dedup']
fields=list(rows[0])[4:]
out={'label':'Verilator integer RTL; isolated P32 T10 H96, post-projection source',
     'tasks':len(rows),'values_per_kind':len(rows)*320*96,'real_cases':32,
     'independent_source_tiles':8,'frames':2,'aggregate_real':[], 'modified_or_diagnostic':[]}
for bp in range(2):
    aggregate=[]
    for mode in range(5):
        r=[x for x in rows if x['real']=='1' and int(x['mode'])==mode and int(x['bp'])==bp]
        assert len(r)==32
        a={'mode':names[mode],'bp':bp,**{k:sum(int(x[k]) for x in r) for k in fields}}
        # These two are maximum live capacity, not additive across tasks.
        for k in ['peak_words','peak_desc']:a[k]=max(int(x[k]) for x in r)
        aggregate.append(a)
    for a in aggregate:
        a['cycle_reduction_vs_direct']=1-a['cycles']/aggregate[0]['cycles']
        a['cycle_ratio_vs_direct']=aggregate[0]['cycles']/a['cycles']
        a['coefficient_traffic_vs_direct']=a['coeff_words']/aggregate[0]['coeff_words']
    out['aggregate_real']+=aggregate
out['modified_or_diagnostic']=[{k:(v if k=='case' else int(v)) for k,v in x.items()}
                               for x in rows if x['real']=='0']
assert len(rows)==410
assert all(int(x['peak_words'])<=24 and int(x['peak_desc'])<=4 for x in rows)
(p/'rtl_summary.json').write_text(json.dumps(out,indent=2)+'\n')
for x in out['aggregate_real']:
    print(x['bp'],x['mode'],x['cycles'],f"{x['cycle_reduction_vs_direct']:.4%}",x['coeff_words'])
