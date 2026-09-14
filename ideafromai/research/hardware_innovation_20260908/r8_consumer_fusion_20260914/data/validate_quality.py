"""Complete per-frame pairing and source arithmetic audit; no inference."""
from pathlib import Path
import csv,json,math
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parent.parent

def main():
    raw=json.loads((HERE/'r8_valid825.json').read_text());new=json.loads((HERE/'deployed_valid.json').read_text());nb=json.loads((HERE/'nb0_valid825.json').read_text());ten=json.loads((HERE/'deployed_diverse.json').read_text())
    assert all(x['complete'] for x in (raw,new,nb,ten))
    assert len(raw['integer_checks'])==len(new['per_frame_integer_checks'])==825
    assert raw['frames']==new['frames']==nb['frames'] and len(raw['frames'])==len(set(raw['frames']))==825
    for field in ('python','torch','gpu','TF32_matmul','TF32_cudnn'):assert raw[field]==new[field]==nb[field],field
    rows={'raw_R8':json.loads((HERE/'quality/integer_r8_frames.json').read_text()),
          'deployed_I24':json.loads((HERE/'deployed_valid/integer_consumer_frames.json').read_text()),'NB0_same_A800':nb['rows']}
    hist=list(csv.DictReader((BASE/'open_fusion_execution/accuracy_baseline/source_nb0_valid825.csv').open()));history={r['file']:r for r in hist}
    summaries={}
    for label,rr in rows.items():
        assert [r['file'] for r in rr]==raw['frames']
        assert all(math.isfinite(r['AEE']) and abs(r['AEE']-r['aee_sum']/r['valid_pixels'])<1e-12 for r in rr)
        summary=dict(frames=len(rr),valid_pixels=sum(r['valid_pixels'] for r in rr),AEE_frame_mean=float(np.mean([r['AEE'] for r in rr])),AEE_pixel_mean=sum(r['aee_sum'] for r in rr)/sum(r['valid_pixels'] for r in rr))
        assert summary['valid_pixels']==48152523
        report=raw['result'] if label=='raw_R8' else new['result'] if label=='deployed_I24' else nb
        assert abs(summary['AEE_frame_mean']-report['AEE_frame_mean'])<1e-12
        assert abs(summary['AEE_pixel_mean']-report['AEE_pixel_mean'])<1e-12
        summaries[label]=summary
    paired=[]
    for a,b,c in zip(*rows.values()):
        h=history[a['file']];assert a['valid_pixels']==b['valid_pixels']==c['valid_pixels']==int(float(h['valid_pixels']))
        paired.append(dict(file=a['file'],valid_pixels=a['valid_pixels'],raw_R8=a['AEE'],deployed_I24=b['AEE'],NB0_same_A800=c['AEE'],NB0_historical=float(h['AEE']),new_minus_raw=b['AEE']-a['AEE'],new_minus_NB0=b['AEE']-c['AEE']))
    tenrows=json.loads((HERE/'deployed_diverse/integer_consumer_frames.json').read_text());lookup={r['file']:r for r in paired}
    assert len(tenrows)==10 and sum(r['valid_pixels'] for r in tenrows)==516735
    assert all(lookup[r['file']]['deployed_I24']==r['AEE'] and lookup[r['file']]['valid_pixels']==r['valid_pixels'] for r in tenrows)
    result=dict(complete=True,protocol='same official local valid825 population, same per-frame GT mask; FP32 EPE then FP64 sum; equal frame mean',fullnet_bittrue=False,
        summaries=summaries,historical_NB0=1.44535253468097,all825_per_frame_population_match=True,diverse10_exact_subset_match=True,
        same_environment_NB0_identity='original local upstream PSN/SDSA NB0 ep29 full flow[-1], remap v1, 78 BN no_running; zero ATLIF/Shiftmax; not author checkpoint',
        student_readout='existing matched-dense stage320 fixed helper/coarse preds2 flow readout; not same network as NB0',
        raw_diverse10_same825=float(np.mean([lookup[r['file']]['raw_R8'] for r in tenrows])),deployed_diverse10=ten['result']['AEE_frame_mean'],
        NB0_same_environment_diverse10=float(np.mean([lookup[r['file']]['NB0_same_A800'] for r in tenrows])),
        new_minus_raw=float(np.mean([r['new_minus_raw'] for r in paired])),new_minus_same_environment_NB0=float(np.mean([r['new_minus_NB0'] for r in paired])),
        new_beats_same_environment_NB0=summaries['deployed_I24']['AEE_frame_mean']<summaries['NB0_same_A800']['AEE_frame_mean'],
        new_beats_historical_NB0=summaries['deployed_I24']['AEE_frame_mean']<1.44535253468097,
        new_vs_raw_mean_absolute_frame_delta=float(np.mean([abs(r['new_minus_raw']) for r in paired])),
        new_vs_raw_max_absolute_frame_delta=float(max(abs(r['new_minus_raw']) for r in paired)),
        new_vs_raw_worse_frames=sum(r['new_minus_raw']>0 for r in paired),new_vs_raw_better_frames=sum(r['new_minus_raw']<0 for r in paired),
        NB0_new_minus_historical=summaries['NB0_same_A800']['AEE_frame_mean']-1.44535253468097,
        new_J_saturations=sum(r['identity_saturations'] for r in new['per_frame_integer_checks']),
        new_I24_saturations=sum(r['I24_low']+r['I24_high'] for r in new['per_frame_integer_checks']),
        new_identity_range=[min(r['identity_min'] for r in new['per_frame_integer_checks']),max(r['identity_max'] for r in new['per_frame_integer_checks'])],
        new_wide_range=[min(r['wide_min'] for r in new['per_frame_integer_checks']),max(r['wide_max'] for r in new['per_frame_integer_checks'])])
    with (HERE/'quality_paired825.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,lineterminator='\n',fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    (HERE/'quality_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
