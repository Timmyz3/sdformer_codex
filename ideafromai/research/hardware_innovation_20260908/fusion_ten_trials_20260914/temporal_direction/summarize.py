import json
from pathlib import Path
H=Path(__file__).resolve().parent;r=json.loads((H/'results.json').read_text());v=json.loads((H/'verification.json').read_text())
real={}
for m in range(3):
 real[m]={}
 for stall in range(2):
  real[m][stall]={}
  for command in range(2):
   a=[x for x in r if x['fixture'].startswith('real_') and x['mode']==m and x['stall']==stall and x['command']==command]
   real[m][stall][command]={k:sum(x[k] for x in a) for k in a[0] if isinstance(a[0][k],int) and k not in ['mode','stall','command']}
summary=dict(complete=True,verification=v,real=real,scope='19fixtures including 8real, full nativeK864/R8/N96/T10 through realIEEE32identity/I24; no fullframe or quality rerun',modes={0:'full',1:'full_or_previous_delta_or_anchor_delta',2:'same_plus_anchor_minus1_plus2_minus2'},main_delta_cycles=real[2][0][1]['total_cycles']-real[1][0][1]['total_cycles'])
(H/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({m:{'cold':real[m][0][0]['total_cycles'],'warm':real[m][0][1]['total_cycles'],'encoder':real[m][0][1]['core_encoder_cycles'],'MAC':real[m][0][1]['core_mac_issues']} for m in real},indent=2))
