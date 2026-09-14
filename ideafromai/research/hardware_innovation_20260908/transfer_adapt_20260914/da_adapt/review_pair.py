"""Independent bounded review of sibling's packed arithmetic and remaining service delta."""
from pathlib import Path
import json
H=Path(__file__).resolve().parent
P=H.parent/'pair_sparse'
rows=json.loads((P/'results.json').read_text())
lookup={(r['fixture'],r['mode'],r['stall'],r['command']):r for r in rows}
checks=0
for r in rows:
    if r['mode']!=19:continue
    b=lookup[r['fixture'],18,r['stall'],r['command']]
    delta_issue=r['aux_events']-b['aux_events']
    stalls=sum(r[k]-b[k] for k in ['source_stalls','weight_stalls','output_stalls'])
    assert r['cycles']-b['cycles']==3*delta_issue+stalls
    for k in ['first_issues','z_vector_reads','z_writes']:assert r[k]-b[k]==delta_issue
    for k in ['metadata_reads','count_checks','count_bank_reads','count_bank_writes','source_words','weight_words','mac_issues','local_source_reads']:assert r[k]==b[k]
    checks+=1
# Ordinary Python integers/divmod: independent of implementation's bit-slice arithmetic.
packed_pairs=0
for coeff in range(-4,4):
    for low in range(256):
        for high in range(128):
            packed=low+2048*high
            assert 0<=packed<=262143
            product=packed*coeff
            high_floor,low_unsigned=divmod(product,2048)
            low_signed=low_unsigned if low_unsigned<1024 else low_unsigned-2048
            corrected_high=high_floor+int(coeff<0 and low>0)
            assert low_signed==coeff*low
            assert corrected_high==coeff*high
            packed_pairs+=1
sv=(P/'decomp_core.sv').read_text()
assert sv.count('count_mem[i][count_rd_addr[i]]')==1
assert 'count_rd_data[i]=count_rd_enable[i]?count_mem[i][count_rd_addr[i]]:20\'d0;' in sv
result=dict(passed=True,independent_packed_product_cases=packed_pairs,mode18_to19_exact_service_pairs=checks,math='divmod((low+2048*high)*q,2048); signed11 low and high carry-in (q<0 && low!=0) reproduce both products',read_port='one count_mem read expression per bank, C_CHECK/C_READ/G_READ mutually exclusive; C_ADD writes later',first_touch='count_live[g] resets on every start; invalid per-class block reads return zero before first C_ADD sets validity; C_ADD never reads count_mem directly',scope='arithmetic proof and measured per-state incremental service only; no new RTL simulation or root file edits')
(H/'pair_readonly_review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
