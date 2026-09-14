from pathlib import Path
import json,csv,statistics,collections
H=Path(__file__).resolve().parent;O=H.parents[1]/'fusion_ten_trials_20260914/algorithm_sparse'
rows=list(csv.DictReader((O/'paired825.csv').open()));assert len(rows)==825 and len({r['file'] for r in rows})==825
pixels=sum(int(r['valid_pixels']) for r in rows);seqs=len({r['file'].rsplit('_',1)[0] for r in rows})
reports={}
for mode,name in [(1,'group_drop'),(2,'rank_drop'),(5,'temporal_group_hold'),(6,'temporal_rank_deadband')]:
 r=json.loads((O/'valid_frames'/f'{name}_frames.json').read_text());q=json.loads((O/f'quality_valid_{mode}.json').read_text())
 assert len(r)==825 and [x['file'] for x in r]==[x['file'] for x in rows]
 assert all(x['valid_pixels']==int(y['valid_pixels']) and x['AEE']==float(y[f'mode{mode}']) for x,y in zip(r,rows))
 frame=statistics.mean(x['AEE'] for x in r);pixel=sum(x['aee_sum'] for x in r)/pixels
 assert abs(frame-q['result']['AEE_frame_mean'])<1e-14 and abs(pixel-q['result']['AEE_pixel_mean'])<1e-14
 reports[str(mode)]=dict(frame_AEE=frame,pixel_AEE=pixel,saturation_frames=sum(bool(x['identity_saturations'] or x['I24_saturations']) for x in q['per_frame_integer_checks']))
result=dict(frames=len(rows),sequences=seqs,pixels=pixels,NB0=statistics.mean(float(x['NB0_same_A800']) for x in rows),R8_I24=statistics.mean(float(x['exact_R8_I24']) for x in rows),modes=reports,reran_network=False)
(H/'quality_recomputed.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
