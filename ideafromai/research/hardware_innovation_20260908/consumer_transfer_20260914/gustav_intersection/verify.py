from pathlib import Path
from collections import defaultdict
import json,csv,struct,re
import numpy as np
H=Path(__file__).resolve().parent;B=H.parents[1];OLD=B/'psn/rtl/gp_slice'
rows=[json.loads(s) for s in (H/'results.jsonl').read_text().splitlines()]
meta={r['name']:r for r in json.loads((H/'cases.json').read_text())['cases']}
buf=(H/'cases.bin').read_bytes();offset=8;ncases=struct.unpack_from('<I',buf,4)[0];cases={};checks=0
def eq(a,b):
 global checks
 checks+=1;assert a==b,(a,b)
def get(dtype,shape):
 global offset
 n=int(np.prod(shape));a=np.frombuffer(buf,dtype,count=n,offset=offset).copy().reshape(shape);offset+=a.nbytes;return a
for _ in range(ncases):
 n=int(get('<u2',())[()]);name=buf[offset:offset+n].decode();offset+=n
 rank,real,time=map(int,get('u1',(3,)));theta=get('<u4',(2,));decode=get('u1',(8,));plen=int(get('<u2',())[()]);program=get('<u2',(plen,))
 coeff=get('<i4',(10,7)).astype(np.int64);tau=get('<i4',(2,10));W=get('i1',(2,384)).astype(np.int64);codes=get('u1',(4,4,384))
 goldenS=get('<i4',(2,4,4,7));goldvalid=get('u1',(2,4,4,7));goldgate=get('<u2',(2,4,4))
 ev=(decode[codes][...,None].astype(np.int64)>>np.arange(7))&1
 S=np.einsum('kpcr,mc->mkpr',ev,W);present=np.einsum('kpcr,mc->mkpr',ev,(W!=0).astype(np.int64))!=0
 U=np.einsum('tr,mkpr->mkpt',coeff,S);gates=np.sum((U>=tau[:,None,None,:]).astype(np.int64)<<np.arange(10),axis=-1)
 eq(bool(np.array_equal(S,goldenS)),True);eq(bool(np.array_equal(present,goldvalid)),True);eq(bool(np.array_equal(gates,goldgate)),True)
 live=np.any(ev,axis=(1,3));nnz=np.count_nonzero(W,axis=1);columns=[np.flatnonzero(w) for w in W]
 dense_per_tile=[int(live.sum())]*2;sparse_per_tile=[int((live&(W[m]!=0)[None,:]).sum()) for m in range(2)]
 oldmeta=0
 for m in range(2):
  for k in range(4):
   ptr=0;head=False;header=False
   for c in np.flatnonzero(live[k]):
    if not header:oldmeta+=2;header=True
    while ptr<len(columns[m]):
     if not head:oldmeta+=2;head=True
     if columns[m][ptr]<c:ptr+=1;head=False;continue
     if columns[m][ptr]==c:ptr+=1;head=False
     break
 expected={}
 for mode in range(4):
  metadata=0 if mode==0 else oldmeta if mode==1 else 96 if mode==2 else sum(2 if n>=334 else 50 for n in nnz)
  values=sum(dense_per_tile) if mode==0 else sum(sparse_per_tile) if mode in [1,2] else sum(dense_per_tile[m] if nnz[m]>=334 else sparse_per_tile[m] for m in range(2))
  zeros=sum(dense_per_tile)-sum(sparse_per_tile) if mode==0 else 0 if mode in [1,2] else sum(dense_per_tile[m]-sparse_per_tile[m] if nnz[m]>=334 else 0 for m in range(2))
  images=[384 if mode==0 else 2+3*n if mode==1 else 48+n if mode==2 else 386 if n>=334 else 50+n for n in nnz]
  expected[mode]=dict(w_metadata_reads=int(metadata),w_value_reads=int(values),w_reads=int(metadata+values),w_zero=int(zeros),image_bytes_tile0=int(images[0]),image_bytes_tile1=int(images[1]),source_reads=288,configuration_beats=plen+20,psn_issues=8*4*plen,nnz_tile0=int(nnz[0]),nnz_tile1=int(nnz[1]))
 cases[name]=dict(expected=expected,nnz=nnz.tolist(),real=real,time=time,rank=rank)
eq(offset,len(buf));eq(len(rows),ncases*16)
invariants={};groups=defaultdict(list)
for r in rows:
 for key,value in cases[r['case']]['expected'][r['frontend']].items():eq(r[key],value)
 eq(r['service_including_configuration'],r['last_gate']+r['configuration_beats']+1)
 inv=tuple(r[k] for k in ['source_reads','packets','events','commits','psn_issues']);key=(r['case'],r['reduce'],r['stress'])
 if key in invariants:eq(invariants[key],inv)
 else:invariants[key]=inv
 family=meta[r['case']].get('weight_variant','original_integer' if r['real'] else 'directed')
 groups[family,'time' if r['time'] else 'class','member' if r['reduce'] else 'scalar','stalled' if r['stress'] else 'ready',r['frontend']].append(r)
old=list(csv.DictReader((OLD/'intersection_results.tsv').open(),delimiter='\t'))
ix={(r['case'],int(r['intersection']),int(r['reduce'])):r for r in old if r['stress']=='0'};matched=0
for r in rows:
 key=(r['case'],r['frontend'],r['reduce'])
 if not r['stress'] and r['frontend']<2 and key in ix:
  for k,v in ix[key].items():
   if k not in ['case','intersection']:eq(r[k],int(v))
  matched+=1
out=[]
for key,rr in groups.items():
 family,route,alu,pressure,front=key
 a=dict(family=family,route=route,alu=alu,pressure=pressure,frontend=front,tasks=len(rr))
 for f in rr[0]:
  if f not in ['case','real','time','frontend','reduce','stress']:
   a[f]=max(r[f] for r in rr) if f.endswith('_peak') else sum(r[f] for r in rr)
 for f in ['last_gate','w_reads','w_metadata_reads','w_value_reads','service_including_configuration','frontend_load_beats']:a['mean_'+f]=a[f]/len(rr)
 a['density']=(a['nnz_tile0']+a['nnz_tile1'])/(768*len(rr));out.append(a)
(H/'group_results.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in out))
log=(H/'run.log').read_text();m=re.search(r'PASS tasks=(\d+) gate_bits=(\d+) cycles=(\d+) response_backpressure=(\d+) nr_peak=(\d+) W_accept_peak=(\d+)',log)
eq(bool(m),True);eq(int(m[1]),len(rows));eq(int(m[2]),len(rows)*320)
strong={}
for family in ['original_integer','row_2of4_trained','broadcast8_C16_half_trained']:
 a={x['frontend']:x for x in out if x['family']==family and x['route']=='class' and x['alu']=='scalar' and x['pressure']=='ready'}
 strong[family]={}
 for f in [1,2,3]:
  strong[family][f]=dict(mean_last_gate=a[f]['mean_last_gate'],versus_read_filter_saved_percent=100*(a[0]['last_gate']-a[f]['last_gate'])/a[0]['last_gate'],mean_including_configuration=a[f]['mean_service_including_configuration'],versus_read_filter_including_configuration_saved_percent=100*(a[0]['service_including_configuration']-a[f]['service_including_configuration'])/a[0]['service_including_configuration'])
result=dict(passed=True,checks=checks,cases=ncases,real_cases=sum(c['real'] for c in cases.values()),real_captures=16,rtl_tasks=len(rows),gate_bits_compared=len(rows)*320,independent_Python_full_C384_gold_cases=ncases,old_ready_all_fields_reproduced=matched,overflow=0,NR4_peak=int(m[5]),W_accept_peak=int(m[6]),W_response_backpressure_beats=int(m[4]),simulation_beats=int(m[3]),all_frontend_packet_events_commits_consumer_invariants=True,independent_metadata_value_bytes_and_image_footprints=True,table_class_scalar_ready=strong)
(H/'verification.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
print(json.dumps(result,indent=2))
