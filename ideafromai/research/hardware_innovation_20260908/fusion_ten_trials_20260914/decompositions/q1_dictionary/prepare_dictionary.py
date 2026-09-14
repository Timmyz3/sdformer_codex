from pathlib import Path
from collections import defaultdict
import json,numpy as np
H=Path(__file__).resolve().parent;meta=json.loads((H/'definition.json').read_text());info=[]
for c in meta['fixtures']:
 p=H/'fixtures'/c['name'];q=np.array([int(x,16)&7 for x in (p/'q1.hex').read_text().split()]).reshape(864,8)
 groups=defaultdict(list)
 for k,row in enumerate(q):
  code=sum(int(v)<<(3*i) for i,v in enumerate(row))
  if code:groups[code].append(k)
 chosen=sorted([(code,ks) for code,ks in groups.items() if len(ks)>=2],key=lambda x:(-len(x[1]),x[0]))[:32]
 cls=np.zeros(864,int);rep=np.zeros(32,int)
 for g,(code,ks) in enumerate(chosen):
  cls[ks]=g+1;rep[g]=ks[0]
 (p/'class.hex').write_text(''.join(f'{int(x):08x}\n' for x in cls));(p/'representative.hex').write_text(''.join(f'{int(x):08x}\n' for x in rep));(p/'ngroups.hex').write_text(f'{len(chosen):08x}\n')
 info.append({'fixture':c['name'],'nonzero_unique_columns':len(groups),'dictionary_groups':len(chosen),'covered_k':int(np.count_nonzero(cls)),'class_k_sizes':[len(ks) for _,ks in chosen]})
(H/'dictionary_manifest.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
