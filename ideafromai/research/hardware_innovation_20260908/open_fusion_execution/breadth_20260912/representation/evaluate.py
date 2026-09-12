"""Replay all fitted source quantizers through actual R24 integer PED windows."""
from pathlib import Path
import json
import sys
import numpy as np
from quantizers import MODES,reconstruct,parameter_cost

HERE=Path(__file__).resolve().parent
OPEN=HERE.parents[1]
sys.path.insert(0,str(OPEN/'new_interface_selection'))
from rebase_probe import execute


def main():
    report=dict(scope='Four actual source windows, same final R24+onepass parents; CPU integer function/representation measurements, not cycles/AEE.',
        calibration_split='train',axes={})
    for axis in ('ordinary','lifting_raw'):
        capture=OPEN/'stage_20260912/algorithm/hardware_exports'/axis
        with np.load(capture/'000_zurich_city_09_a_0001.npz') as z:data={k:z[k] for k in z.files}
        with np.load(capture/'deployed_constants.npz') as z:q={k:z[k] for k in z.files}
        with np.load(HERE/'parameters'/(axis+'.npz')) as z:params={m:{k:z[m+'_'+k] for k in ('D','c','step')} for m in MODES}
        geometry=json.loads(str(data['window_geometry_json']))
        ar=dict(windows={},parameters={m:parameter_cost(p) for m,p in params.items()})
        for label,geo in geometry.items():
            y,x=geo['gate_origin'];oy,ox=geo['output_origin'];dy,dx=2*oy-y,2*ox-x
            actual=data[label+'_updated_I24'][:,:,dy:dy+8:2,dx:dx+8:2].astype(np.int64)
            gates=data[label+'_proj_gate'][:,:,dy:dy+8:2,dx:dx+8:2].astype(np.int64)
            flat=lambda v:v.transpose(1,0,2,3).reshape(96,-1)
            gold,baseline=execute(q['U_ped_q16'],q['V_ped_q16'],flat(actual),q['PED_bias_q24'])
            original=flat(data[label+'_continuous_q24'])
            assert np.array_equal(gold,original)
            modes={}
            for mode in MODES:
                estimated,code,counts=reconstruct(actual,gates,params[mode])
                out,clips=execute(q['U_ped_q16'],q['V_ped_q16'],flat(estimated),q['PED_bias_q24'])
                error=out.astype(float)-gold
                grouped=code.reshape(10,96,4,2,2)
                physical_empty=np.count_nonzero(~np.any(grouped,axis=(0,4)))
                row=dict(output_values=int(out.size),changed_values=int(np.count_nonzero(error)),
                    source_physical_MAE=float(np.mean(np.abs(estimated-actual)))/(1<<14),
                    PED_physical_RMSE=float(np.sqrt(np.mean(error**2)))/(1<<14),
                    PED_NRMSE=float(np.linalg.norm(error)/np.linalg.norm(gold)),
                    scalar_zero_fraction=float(np.mean(code==0)),
                    P2_T10_empty_words=int(physical_empty),P2_T10_words=96*4*2,
                    original_packed24_bytes=int(actual.size)*3,code8_bytes=int(code.size),
                    representation_counts=counts,arithmetic=clips)
                modes[mode]=row
            ar['windows'][label]=dict(original_R24_capture_differences=0,baseline_arithmetic=baseline,modes=modes)
        report['axes'][axis]=ar
    (HERE/'windows.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({a:{w:{m:round(v['PED_NRMSE'],6) for m,v in q['modes'].items()} for w,q in ar['windows'].items()} for a,ar in report['axes'].items()}))


if __name__=='__main__':main()
