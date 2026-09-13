"""Fixed 72-block budget: single-block pruning or close a whole source group.

The calibration Gram matrix preserves cross terms of removed contributions.
Service comes from the already measured mode3, not from unpriced SOP counts.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from cost import features, measure

HERE = Path(__file__).resolve().parent


def gram_matrix(source, weight):
    # O8 groups have disjoint outputs; retain all C4 contribution cross terms.
    grams = np.empty((12,24,24), dtype=np.float64)
    for og in range(12):
        vectors = []
        for cg in range(24):
            s = source[:,:,cg*4:(cg+1)*4].astype(np.int64)
            w = weight[og*8:(og+1)*8,cg*4:(cg+1)*4].astype(np.int64)
            parts = [np.einsum('ftcij,ncij->ftn', s[:,:,:,py:py+3,px:px+3], w,
                               optimize=True).reshape(-1) for py in range(2) for px in range(2)]
            vectors.append(np.concatenate(parts).astype(np.float64))
        y = np.stack(vectors)
        grams[og] = y @ y.T
    return grams


def sse(removed, grams):
    return float(np.einsum('gc,gcd,gd->',removed,grams,removed,optimize=True))


def choose(grams, feat):
    live = np.ones((12,24), dtype=bool)
    history = []
    remaining = 72
    while remaining:
        gone = (~live).astype(np.float64)
        current_error = sse(gone,grams)
        current = measure(live,feat)
        g = live.sum(axis=0)
        marginal_single = 2*np.einsum('gc,gcd->gd',gone,grams)+np.diagonal(grams,axis1=1,axis2=2)
        actions = []
        for og,cg in zip(*np.nonzero(live)):
            saved = int(feat['execution'][cg]) + (158*feat['tiles'] if g[cg] == 1 else 0)
            delta = float(marginal_single[og,cg])
            actions.append((delta/saved if saved else float('inf'),1,int(og),int(cg),delta,saved))
        for cg in range(24):
            count = int(g[cg])
            if count < 2 or count > remaining: continue
            delta = float(marginal_single[live[:,cg],cg].sum())
            saved = count*int(feat['execution'][cg])+158*feat['tiles']
            actions.append((delta/saved,count,12,cg,delta,saved))
        # Stable tie break favours the smaller action, then original O/C order.
        ratio,count,og,cg,delta,saved = min(actions)
        previous = live.copy()
        if og == 12: live[:,cg] = False
        else: live[og,cg] = False
        after = measure(live,feat)
        actual_error = sse((~live).astype(np.float64),grams)
        assert int((previous & ~live).sum()) == count
        assert current['cycles']-after['cycles'] == saved
        assert np.isclose(actual_error-current_error,delta,rtol=1e-9,atol=.1)
        remaining -= count
        history.append(dict(action='whole_remaining_C4' if og==12 else 'O8_C4_block',
            O8=None if og==12 else og,C4=cg,blocks_removed=count,remaining_budget=remaining,
            calibration_delta_SSE=delta,mode3_core_saved=saved,score=ratio if np.isfinite(ratio) else None,
            calibration_SSE=actual_error,mode3_remaining_core=after['cycles'],
            live_source_groups=after['live_source_groups']))
    return live,history


def export(live, evaluation, output):
    s = evaluation['source_bits'].astype(np.int64)
    raw = evaluation['weight_q16'].astype(np.int64)
    w = raw * live.repeat(8,axis=0).repeat(4,axis=1)[:,:,None,None]
    golden = np.empty((len(s),10,96,2,2),dtype=np.int64)
    for py in range(2):
        for px in range(2):
            golden[:,:,:,py,px] = np.einsum('ftcij,ncij->ftn',s[:,:,:,py:py+3,px:px+3],w,optimize=True)
    shared = {k:evaluation[k] for k in ('source_bits','source_valid_yx','weight_fp32','theta',
              'weight_exponent','output_origin_yx','input_origin_yx','frame')}
    np.savez_compressed(output,**shared,weight_q16=w.astype(np.int16),live=live,golden_accum=golden)
    return golden


def main():
    p=argparse.ArgumentParser();p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,required=True);args=p.parse_args()
    cal=np.load(args.calibration);ev=np.load(args.evaluation)
    assert str(cal['frame']) != str(ev['frame'])
    assert np.array_equal(cal['weight_q16'],ev['weight_q16'])
    grams=gram_matrix(cal['source_bits'],cal['weight_q16'])
    feat=features(cal['source_bits'],cal['input_origin_yx'])
    live,history=choose(grams,feat)
    gold=export(live,ev,HERE/'mixed_retirement25_q16.npz')
    ef=features(ev['source_bits'],ev['input_origin_yx'])
    report=dict(complete=True,calibration=str(args.calibration),calibration_frame=str(cal['frame']),
        calibration_tiles=len(cal['source_bits']),calibration_spikes=int(cal['source_bits'].sum()),
        evaluation_frame=str(ev['frame']),evaluation_tiles=len(ev['source_bits']),
        dropped_blocks=int((~live).sum()),fully_retired_C4=np.where(~live.any(axis=0))[0].tolist(),
        live_mask=live.astype(int).tolist(),history=history,
        calibration_core_model=measure(live,feat),evaluation_core_model=measure(live,ef),
        calibration_SSE=sse((~live).astype(float),grams),
        evaluation_SSE=float(np.square((gold-ev['golden_accum']).astype(np.float64)).sum()),
        training=False,sweep=False,RTL_measured_here=False,
        rule='Greedy exact cumulative output-SSE marginal / mode3 cycle marginal; single-block or whole-remaining-C4 action; fixed72blocks; stable ratio,count,O,C ordering',
        limitations='Hardware-aware mixed structured pruning candidate; no novelty or AEE/RTL result inferred from calibration.')
    np.savez_compressed(HERE/'calibration_gram.npz',gram=grams,**feat)
    (HERE/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('live_mask','history')},indent=2))


if __name__ == '__main__':
    main()
