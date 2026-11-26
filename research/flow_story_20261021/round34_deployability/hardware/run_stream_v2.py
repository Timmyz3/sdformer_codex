#!/usr/bin/env python3
from pathlib import Path
import concurrent.futures,hashlib,importlib.util,json,subprocess
import numpy as np
P=Path(__file__).resolve().parent; R33=P.parent.parent/'round33_predictability'; OUT=P/'actual_v2';OUT.mkdir(exist_ok=True)
member=json.loads((R33/'data/MEMBERSHIP_V1.json').read_text()); records=member['records']
reference=np.load(P.parent/'numeric/actual_v1/FEATURES_FP32_V1.npz')
jobs=[]
for i,r in enumerate(records):
 s=next(s for s in r['source_files'] if s['category']=='input');folder='confirm_source_snapshot' if r['role']=='confirm' else 'fit_tune_source_snapshot';source=R33/'data'/folder/s['relative_path']
 assert hashlib.sha256(source.read_bytes()).hexdigest()==s['sha256']
 jobs.append((i,r,source))
def job(item):
 i,r,source=item;name=r['target_id'];out=OUT/name
 trace=str(OUT/'rowband160.trace') if i==0 else '-'
 command=[str(P/'stream_features_v2'),str(source),str(out)+'.f32',str(out)+'.json',trace,'rowband','1']
 result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 if result.returncode: raise RuntimeError((name,result.stderr))
 x=np.fromfile(str(out)+'.f32',dtype=np.float32);memory=np.fromfile(str(out)+'.f32.ios',dtype=np.float32);assert memory.size==524288//4;assert np.array_equal(x,memory[400000//4:400000//4+160]);role=r['role'];refi=list(reference[role+'_ids']).index(name);expected=reference[role+'_x160'][refi]
 return dict(id=name,role=role,input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),input=str(source),feature_file=out.name+'.f32',service_file=out.name+'.json',max_abs_feature_error=float(np.max(np.abs(x-expected))),different_FP32_values=int(np.count_nonzero(x!=expected)),features=x)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 rows=list(pool.map(job,jobs))
x=np.stack([r.pop('features') for r in rows]);np.savez(OUT/'STREAM_FEATURES_V1.npz',ids=np.asarray([r['id'] for r in rows]),roles=np.asarray([r['role'] for r in rows]),x160=x)
first=jobs[0][2]
for label,layout,direction in [('rowband128','rowband','0'),('cell160','cell','1')]:
 subprocess.run([str(P/'stream_features_v2'),str(first),str(OUT/(label+'.f32')),str(OUT/(label+'.json')),str(OUT/(label+'.trace')),layout,direction],check=True)
assert np.array_equal(np.fromfile(OUT/'cell160.f32',dtype=np.float32),x[0])
assert np.array_equal(np.fromfile(OUT/'rowband128.f32',dtype=np.float32),x[0,:128])
report=dict(schema='r34_real_full_stream_feature_execution_v1',source_sha256=hashlib.sha256((P/'stream_features_v2.cpp').read_bytes()).hexdigest(),identity_count=len(rows),rows=rows,max_abs_feature_error=max(r['max_abs_feature_error'] for r in rows),feature_reference='numeric/actual_v1/FEATURES_FP32_V1.npz',reference_sha256=hashlib.sha256((P.parent/'numeric/actual_v1/FEATURES_FP32_V1.npz').read_bytes()).hexdigest(),countercondition_features_bitwise_equal=True,X128_prefix_bitwise_equal=True,model_ticks_not_RTL=True,GPU_used=False)
(OUT/'STREAM_EXECUTION_V1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
