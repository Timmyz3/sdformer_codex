"""Three predeclared g1 readouts, no refit or new neural forward."""
from pathlib import Path
import importlib.util
import json
import hashlib
import numpy as np

B=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('bounded_calibration',B/'run_calibration_v1.py')
H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
M=H.M


def main():
    out=B/'direction_ablation_v1';out.mkdir(exist_ok=False)
    hardware=H.read(H.R33/'hardware/results/SUMMARY_V1.json')
    sources={};predictions={};data={}
    for dim,branch in [(128,'voxel128_v1'),(160,'directional160_v1')]:
        mp=H.R33/f'fitted/{branch}/MODEL_V1.json';model=H.read(mp);a=model['auxiliary']['g1']
        a={k:np.asarray(v) if k in ['mean','scale','coef','intercept'] else v for k,v in a.items()}
        const=a['intercept'][:2];sources[f'model_{dim}']=H.sha(mp)
        for role,folder in [('tune','fit_tune'),('old_confirm','confirm')]:
            p=H.R33/f'packs/{folder}_{dim}_v1/{"confirm" if role=="old_confirm" else role}.npz'
            d=H.arrays(p);aux=M.ridge_predict(a,d['x']);v,response=M.split_aux(aux,1)
            scores={'predicted_direction':M.signed_scores(aux,1),
                    'fit_constant_direction':(const[None,None,:]*response).sum(axis=2),
                    'predicted_response_squared_norm':(response*response).sum(axis=2)}
            old=H.read(H.R33/f'evaluations/{"confirm" if role=="old_confirm" else role}_{dim}_v1/PREDICTIONS_V1.json')
            assert np.array_equal(scores['predicted_direction'],np.asarray(old['results']['scores']['signed_g1']))
            key=f'{dim}_{role}';data[key]=(d,H.R33/f'packs/{folder}_{dim}_v1/QUALITY_V1.json')
            predictions[key]=dict(ids=d['ids'].tolist(),constant_direction=const.tolist(),scores={k:s.tolist() for k,s in scores.items()})
            sources[key+'_pack']=H.sha(p)
    H.write(out/'PREDICTIONS_BEFORE_QUALITY_V1.json',dict(source_sha256=H.sha(__file__),predictions=predictions,sources=sources,no_refit=True))
    results={}
    for key,p in predictions.items():
        d,qp=data[key];raw=H.read(qp)['frames'];sources[key+'_quality']=H.sha(qp)
        results[key]={}
        for name,scores in p['scores'].items():
            # Summarizer also calculates allow-Q4 internally; it is not retained
            # or compared because the norm proxy has no compatible zero threshold.
            results[key][name]=H.summarize(np.asarray(scores),d,raw,hardware)['fixed_six']
    H.write(out/'RESULTS_V1.json',dict(status='COMPLETE',schema='r34_direction_ablation_v1',source_sha256=H.sha(__file__),
        summarizer_source_sha256=H.sha(B/'run_calibration_v1.py'),sources=sources,results=results,
        new_fits=0,GPU_used=False,new_model_forwards=0,old_confirm_exposed=True,only_fixed_six=True))
    print('DIRECTION_ABLATION_COMPLETE')


if __name__=='__main__':main()
