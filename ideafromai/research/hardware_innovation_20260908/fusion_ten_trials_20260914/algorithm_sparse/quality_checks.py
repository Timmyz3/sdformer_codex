from pathlib import Path
import json,csv,numpy as np
H=Path(__file__).resolve().parent;old=list(csv.DictReader((H.parents[1]/'r8_consumer_fusion_20260914/data/quality_paired825.csv').open()));names=[r['file'] for r in old];base={r['file']:r for r in old}
checks=[];arms={}
for m,name in [(1,'group_drop'),(2,'rank_drop'),(5,'temporal_group_hold'),(6,'temporal_rank_deadband')]:
 p=H/f'quality_valid_{m}.json';rp=H/'valid_frames'/f'{name}_frames.json'
 if not p.exists() or not rp.exists():continue
 d=json.loads(p.read_text());rr=json.loads(rp.read_text())
 if not d.get('complete'):continue
 assert len(rr)==825 and [r['file'] for r in rr]==names and len(d['per_frame_integer_checks'])==825
 assert len(set(r['file'] for r in rr))==825
 for r in rr:assert r['valid_pixels']==int(base[r['file']]['valid_pixels'])
 avg=float(np.mean([r['AEE'] for r in rr]));pixels=sum(r['valid_pixels'] for r in rr);pixavg=sum(r['aee_sum'] for r in rr)/pixels
 assert np.isclose(avg,d['result']['AEE_frame_mean'],rtol=0,atol=1e-14) and np.isclose(pixavg,d['result']['AEE_pixel_mean'],rtol=0,atol=1e-14)
 assert pixels==48152523 and not any(r['identity_saturations'] or r['I24_saturations'] for r in d['per_frame_integer_checks'])
 by={r['file']:r for r in rr};ten=json.loads((H/f'quality_diverse_{m}.json').read_text());tenmean=float(np.mean([by[n]['AEE'] for n in ten['frames']]))
 arms[str(m)]={r['file']:r['AEE'] for r in rr};delta=np.array([r['AEE']-float(base[r['file']]['deployed_I24']) for r in rr]);nbo=np.array([r['AEE']-float(base[r['file']]['NB0_same_A800']) for r in rr])
 checks.append(dict(mode=m,complete=True,frames=825,valid_pixels=pixels,AEE=avg,pixel_AEE=pixavg,NB0=1.447936665574317,below_NB0=avg<1.447936665574317,mean_delta_exact=float(delta.mean()),frames_better_exact=int((delta<0).sum()),mean_abs_delta_exact=float(np.abs(delta).mean()),max_abs_delta_exact=float(np.abs(delta).max()),frames_better_NB0=int((nbo<0).sum()),original_diverse10=ten['result']['AEE_frame_mean'],same_files_from825=tenmean,diverse_mean_equal=tenmean==ten['result']['AEE_frame_mean'],integer_checks=825,identity_saturations=0,I24_saturations=0))
with (H/'paired825.csv').open('w') as out:
 w=csv.writer(out);w.writerow(['file','valid_pixels','exact_R8_I24','NB0_same_A800']+[f'mode{m}' for m in arms])
 for r in old:w.writerow([r['file'],r['valid_pixels'],r['deployed_I24'],r['NB0_same_A800']]+[arms[m][r['file']] for m in arms])
(H/'quality_checks.json').write_text(json.dumps(dict(complete=len(checks)==4,completed_arms=len(checks),arms=checks),indent=2)+'\n');print(json.dumps(checks,indent=2))
