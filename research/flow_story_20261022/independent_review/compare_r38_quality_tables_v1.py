"""Bind author's frozen R38 tables to independent original-array recomputation."""
from pathlib import Path
import hashlib,json,os
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];B=ROOT/'research/flow_story_20261022/round38_frozen_survival';A=B/'quality/actual_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):
 if a is None or b is None:assert a is None and b is None,(a,b)
 else:np.testing.assert_allclose(a,b,rtol=3e-13,atol=3e-12)
def main():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='';out=HERE/'R38_QUALITY_TABLES_INDEPENDENT_V1.json';assert not out.exists()
 ownpath=HERE/'R38_RAW_QUALITY_INDEPENDENT_V1.json';own=read(ownpath);assert own['status']=='PASS'
 freeze=read(B/'quality/SOURCE_FREEZE_V1.json')
 for file,h in freeze['files'].items():assert sha(B/'quality'/file)==h
 for spec in freeze['bound'].values():assert sha(spec['path'])==spec['sha256']
 summary=read(A/'SUMMARY_V1.json');bindings=read(A/'BINDINGS_V1.json');quality=read(A/'QUALITY_V1.json')
 assert summary['status']=='COMPLETE' and summary['source_freeze_sha256']==sha(B/'quality/SOURCE_FREEZE_V1.json')==bindings['source_freeze_sha256']
 assert summary['collector_result_sha256']==bindings['collector_result_sha256']==own['raw_result_sha256']
 assert bindings['action_plan_sha256']==own['action_plan_sha256'] and quality['bias_model_sha256']==own['bias_model_sha256']
 assert summary['zero_bias_exact_region_comparisons']==quality['zero_bias_exact_region_comparisons']==71*16
 assert summary['new_model_fits']==0 and summary['no_oracle_or_regret'] and not summary['GPU_used']
 rows=own['raw_frames'];assert [r['target_id'] for r in rows]==[r['target_id'] for r in quality['frames']]
 regs=own['region_names'];policies=list(rows[0]['policy_actions']);counts=dict(raw_policy_frame_regions=0,aggregate_regions=0,leave_one_regions=0,comparisons=0)
 for a,b in zip(rows,quality['frames']):
  assert a['sequence']==b['sequence'] and a['policy_actions']==b['policy_actions']
  for cond in ['original','same_bias_corrected']:
   for p in policies:
    action=a['policy_actions'][p]
    for j,r in enumerate(regs):
     q=b[cond][p][r];n=a['counts'][j];e=a['action_errors'][action][cond][j]
     assert q['valid_pixels']==n;close(q['error_sum'],e);close(q['AEE'],e/n if n else None);counts['raw_policy_frame_regions']+=1
 files={};key={}
 for cond in ['original','same_bias_corrected']:
  path=A/(cond+'_EVALUATION_V1.json');ev=read(path);files[path.name]=sha(path);key[cond]={}
  assert ev['no_oracle_or_regret'] and ev['new_model_fits']==0 and ev['primary_precision']=='frozen_FP32'
  for p in policies:
   ours=own['independent_quality'][cond][p]['versus_Q4'];stored=ev['summaries'][p]
   for scope,rr in ours['summaries'].items():
    for rn,q in rr.items():
     m=stored[scope][rn];assert q['valid_pixels']==m['valid_pixels'];close(q['error_sum'],m['error_sum']);close(q['AEE'],m['pixel_weighted_AEE']);close(q['frame_mean_AEE'],m['frame_mean_AEE']);counts['aggregate_regions']+=1
   s=summary['results'][cond][p];q=ours['summaries']['overall']
   close(s['AEE'],q['all_valid']['AEE']);close(s['frame_mean_AEE'],q['all_valid']['frame_mean_AEE']);close(s['B'],q['B']['AEE']);close(s['L'],q['L']['AEE']);close(s['relative_Q4'],q['all_valid']['relative']);assert s['tails']==len(ours['high_support_tail_review'])
   key[cond][p]={k:s[k] for k in ['AEE','frame_mean_AEE','B','L','relative_Q4','relative_static360','tails']}
   for ref in policies:
    if p==ref:continue
    independent=own['independent_quality'][cond][p]['versus_'+ref];cmp=ev['comparisons'][p+'__vs_'+ref]
    assert [(t['target_id'],t['region']) for t in cmp['high_support_tail_review']]==[(t['target_id'],t['region']) for t in independent['high_support_tail_review']]
    assert cmp['sequence_gain_counts']['improved']==independent['sequences_improved_all_valid'];counts['comparisons']+=1
    for drop,rr in independent['leave_one_sequence'].items():
     for rn,q in rr.items():
      m=ev['leave_one_sequence'][p+'__vs_'+ref][drop][rn]
      for metric,ck,rk in [('pixel_weighted_AEE','AEE','reference_AEE'),('frame_mean_AEE','frame_mean_AEE','reference_frame_mean_AEE')]:
       close(m[metric]['candidate'],q[ck]);close(m[metric]['reference'],q[rk]);close(m[metric]['delta'],q[ck]-q[rk] if q[ck] is not None and q[rk] is not None else None)
      counts['leave_one_regions']+=1
 report=dict(status='PASS',schema='flow_r38_quality_table_independent_v1',auditor_sha256=sha(__file__),independent_raw_audit_sha256=sha(ownpath),quality_source_freeze_sha256=sha(B/'quality/SOURCE_FREEZE_V1.json'),quality_sha256=sha(A/'QUALITY_V1.json'),summary_sha256=sha(A/'SUMMARY_V1.json'),evaluation_sha256=files,counts=counts,key=key,GPU_calls=0,new_fits=0)
 out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':'PASS','counts':counts,'sha256':sha(out)}))
if __name__=='__main__':main()
