from pathlib import Path
import importlib.util,json
import numpy as np
H=Path(__file__).resolve().parent;B=H.parents[1]
old=B/'fusion_ten_trials_20260914/phase_borrow'
spec=importlib.util.spec_from_file_location('table',B/'transfer_adapt_20260914/pair_psum_overlay/prepare.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
order=[8,2,6,3,9,4,1,5,7,0]
def metadata(d):
 q=np.fromfile(d/'param4.bin','<i4').reshape(864,8)
 cls,rep,sizes,bounds,excluded=m.compile_table(q)
 data=np.zeros((864,8),'<u4');data[:,0]=np.sum(cls<<(6*np.arange(4)),axis=1);data.tofile(d/'param8.bin')
 rep.astype('<i4').tofile(d/'param9.bin')
 data=np.zeros((1,8),'<u4');data[0,0]=max(sizes);data.tofile(d/'param10.bin')
 packed=sum(v<<(4*t) for t,v in enumerate(order));data[0,0]=packed&0xffffffff;data[0,1]=packed>>32;data.tofile(d/'param11.bin')
 return dict(groups=max(sizes),max_class=max([0]+sum(bounds,[])),over255=sum(excluded))
cases=[]
for r in json.loads((old/'fixtures.json').read_text()):
 d=H/'fixtures'/r['name'];d.mkdir(parents=True,exist_ok=True)
 for p in (old/'fixtures'/r['name']).glob('*.bin'):
  if not (d/p.name).exists():(d/p.name).symlink_to(p)
 cases.append(dict(r,path=str(d),**metadata(d)))
# Add old count boundary/source-order fixtures using native full raw and the actual consumer coefficients.
extra=B/'transfer_adapt_20260914/pair_psum_overlay/fixtures'
for name,p in [(p.name,p) for p in extra.iterdir() if p.is_dir() and (p.name.startswith('edge_') or p.name in ['mixed_class_255_256','q2_zero_overlay'])]+[('inverse_time_unique',B/'transfer_adapt_20260914/pair_psum_overlay/temporal_pairing/fixtures/t_unique')]:
 if not p.exists():continue
 d=H/'fixtures'/('count_'+name);d.mkdir(parents=True,exist_ok=True)
 q=m.readhex(p/'q1.hex').reshape(864,8);v=m.readhex(p/'q2.hex').reshape(96,8)
 q.astype('<i4').tofile(d/'param4.bin');v.astype('<i4').tofile(d/'param5.bin')
 np.repeat(np.any(q!=0,axis=1)[:,None],8,axis=1).astype('<u4').tofile(d/'param6.bin')
 if not (d/'param7.bin').exists():(d/'param7.bin').symlink_to(old/'fixtures/real_0/param7.bin')
 source=m.readhex(p/'source.hex').astype('<u2');source.tofile(d/'source.bin')
 raw=m.readhex(p/'gold.hex');raw.astype('<i4').tofile(d/'raw.bin')
 np.zeros(3840,'<i4').tofile(d/'identity.bin');np.zeros(3840,'<f4').tofile(d/'identity_fp32.bin')
 ab=np.fromfile(d/'param7.bin','<i4').reshape(12,2,8).astype(np.int64)
 raw=raw.reshape(12,40,8);wide=raw*ab[:,0,None,:]+(ab[:,1,None,:]<<20)
 low=wide&((1<<26)-1);rounded=(wide>>26)+((low>(1<<25))|((low==(1<<25))&((wide>>26)&1)))
 np.clip(rounded,-(1<<23),(1<<23)-1).astype('<i4').tofile(d/'gold.bin')
 oy,ox=m.readhex(p/'origin.hex');tile=int((oy+1)//2*160+(ox+1)//2)
 cases.append(dict(name='count_'+name,tile_id=tile,input_origin=[int(oy),int(ox)],path=str(d),**metadata(d)))
(H/'fixtures.json').write_text(json.dumps(cases,indent=2)+'\n')
print(json.dumps({'fixtures':len(cases),'new_metadata_tables':len(cases),'count_edges':[r['name'] for r in cases if r['name'].startswith('count_')]}))
