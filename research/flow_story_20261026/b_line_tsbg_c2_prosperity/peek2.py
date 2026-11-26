import json,os,collections
D=os.path.expanduser('~/work/sdformer_codex/SDformer/hw_autoresearch_nts07/results/m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831/')
cnt=collections.Counter(); ret=collections.Counter(); ex={}
names=collections.OrderedDict()
for l in open(D+'unified_ordered_records.jsonl'):
    r=json.loads(l)
    k=r['category']; cnt[k]+=1
    if r['payload'].get('retained'): ret[k]+=1; ex.setdefault(k,r)
    if r['global_sample_id']==0: names[r['name']]=(k,r['input']['shape'],r['input']['active'],r['input']['negative'],r['payload'].get('retained'))
print(cnt); print(ret)
for k,r in ex.items(): print(k, json.dumps(r)[:700])
print(len(names))
for n,v in names.items(): print(n,v)
print(os.listdir(D+'payloads')[:10], len(os.listdir(D+'payloads')))
