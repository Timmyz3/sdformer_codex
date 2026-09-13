"""Validate completed exported records only; never runs a model."""
import csv,json
import numpy as np
from model_access import HERE

def main():
    r=json.loads((HERE/'valid825.json').read_text());assert r['complete']
    names=r['frames'];assert len(names)==len(set(names))==825
    baseline=r['historical_NB0_valid825'];assert baseline==1.44535253468097
    reference=None;table=[];allrows={}
    for arm in ('dense_q16','block_magnitude25','cin_fullcost25'):
        rows=json.loads((HERE/'quality825'/(arm+'_frames.json')).read_text())
        s=json.loads((HERE/'quality825'/(arm+'_summary.json')).read_text())
        assert len(rows)==825 and [v['file'] for v in rows]==names
        assert s['complete'] and s['frames']==825 and r['results'][arm]['complete']
        counts=[v['valid_pixels'] for v in rows]
        assert min(counts)>0
        if reference is None:reference=counts
        else:assert counts==reference
        mean=float(np.mean([v['aee_sum']/v['valid_pixels'] for v in rows]))
        weighted=sum(v['aee_sum'] for v in rows)/sum(counts)
        assert np.isfinite(mean) and abs(mean-s['AEE_frame_mean'])<1e-12
        assert abs(weighted-s['AEE_pixel_mean'])<1e-12
        assert abs(mean-r['results'][arm]['AEE_frame_mean'])<1e-12
        assert sum(counts)==s['valid_pixels']==48152523
        table.append(dict(arm=arm,frames=825,valid_pixels=sum(counts),AEE_frame_mean=mean,AEE_pixel_mean=weighted,
            historical_NB0=baseline,below_historical_NB0=mean<baseline,wall_seconds_including_any_priority_pause=s['wall_seconds']))
        allrows[arm]=rows
    probe=json.loads((HERE/'quality825/dense_protocol10_frames.json').read_text())
    dense={v['file']:v for v in allrows['dense_q16']}
    compare=[dict(file=v['file'],valid_pixels_equal=v['valid_pixels']==dense[v['file']]['valid_pixels'],
        AEE_delta=v['AEE']-dense[v['file']]['AEE']) for v in probe]
    with (HERE/'quality825_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n');w.writeheader();w.writerows(table)
    # Every paired frame remains inspectable in one table, in official order.
    with (HERE/'quality825_paired_frames.csv').open('w',newline='') as f:
        fields=['file','valid_pixels']+[a+'_AEE' for a in allrows]
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for i,n in enumerate(names):w.writerow(dict(file=n,valid_pixels=reference[i],**{a+'_AEE':v[i]['AEE'] for a,v in allrows.items()}))
    out=dict(complete=True,model_run=False,frames_per_arm=825,paired_frames=825,arms=3,
        all_frame_names_and_pixel_counts_equal=True,valid_pixels_per_arm=sum(reference),summary_recomputed=True,
        dense_protocol10_vs_official_same_frame=compare,results=table)
    (HERE/'quality825_validation.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
