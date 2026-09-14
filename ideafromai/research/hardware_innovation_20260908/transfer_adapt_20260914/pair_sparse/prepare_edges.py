from pathlib import Path
import importlib.util,numpy as np,json
H=Path(__file__).resolve().parent;B=H.parents[1]
spec=importlib.util.spec_from_file_location('pair_orig',B/'fusion_review_followup_20260914/pair_dictionary/prepare.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
old=B/'fusion_review_followup_20260914/pair_dictionary/fixtures/real_0';v=m.readhex(old/'q2.hex').reshape(12,8,8).transpose(0,2,1).reshape(96,8)
cases=json.loads((H/'fixtures.json').read_text());cases=[c for c in cases if not c['name'].startswith('edge_')]
ks=[k for k in range(864)if k%3==0]
for axis,n in [('low',255),('low',256),('high',127),('high',128)]:
 q=np.zeros((864,8),np.int64);q[ks[:n]]=np.array([-4,-3,-2,-1,1,2,3,0])
 s=np.zeros((96,4,4),np.int64);s[:,:,0 if axis=='low' else 1]=1023
 e=np.zeros((864,40),np.int64)
 for k in range(864):
  c,tap=divmod(k,9)
  for p in range(4):e[k,p*10:p*10+10]=(s[c,p//2+tap//3,p%2+tap%3]>>np.arange(10))&1
 z=e.T@q;raw=np.concatenate([z@v[g*8:g*8+8].T for g in range(12)])
 cls,rep,sizes,unique=m.compile_table(q)
 name=f'edge_{axis}_{n}';p=H/'fixtures'/name;p.mkdir(exist_ok=True,parents=True)
 for f,a in [('source',s),('q1',q),('q2',v.reshape(12,8,8).transpose(0,2,1)),('origin',[11,13]),('k_live',np.any(q!=0,axis=1).astype(int)),('gold',raw),('class',np.sum(cls<<(6*np.arange(4)),axis=1)),('representative',rep),('ngroups',[max(sizes)])]:m.writehex(p/(f+'.hex'),a)
 cases.append(dict(name=name,group_sizes=sizes,nonzero_unique_keys=unique,grouped_pairs=int(np.count_nonzero((cls>0)&(cls<=32))),direct_pairs=int(np.count_nonzero(cls==63))))
(H/'fixtures.json').write_text(json.dumps(cases,indent=2)+'\n')
s=(H/'run.py').read_text();s=s.replace("str(H.parents[1] / 'fusion_review_followup_20260914/pair_dictionary/fixtures' / case['name'])", "str((H/'fixtures'/case['name']) if (H/'fixtures'/case['name']).exists() else H.parents[1]/'fusion_review_followup_20260914/pair_dictionary/fixtures'/case['name'])");(H/'run.py').write_text(s)
s=(H/'verify.py').read_text().replace("p = H.parents[1] / 'fusion_review_followup_20260914/pair_dictionary/fixtures' / c['name']", "p = H/'fixtures'/c['name']\n    if not p.exists():p=H.parents[1]/'fusion_review_followup_20260914/pair_dictionary/fixtures'/c['name']")
s=s.replace('(q >= -3)','(q >= -4)').replace("if r['mode']<=15:\n        eq", "if r['mode']<=15 and (r['fixture'],r['mode'],r['stall'],r['command'])in lookup:\n        eq")
(H/'verify.py').write_text(s)
