"""Pay the P2-aligned row-phase mask in the unchanged producer emitter."""
from pathlib import Path
import json
import numpy as np
from execute import FULL,read_npz,run,HERE


def main():
    masks=json.loads((HERE/'paired_phase_masks.json').read_text())
    prior=json.loads((HERE/'execution.json').read_text())
    old_masks=json.loads((HERE.parent/'review/phase_group8_masks.json').read_text())
    result=dict(scope='Same real source/preview/PSN producer and same H8 emitter; row parity shares masks across horizontal P2. CPU payload slots, not integrated chain or AEE.',axes={})
    for axis in ['ordinary','lifting_raw']:
        # The exhaustive 66-pair global control selected the identical mask,
        # so its existing exact same emitter result is reusable, not rerun.
        assert masks[axis]['global_joint_pair']['mask_uint8']==old_masks[axis]['global_group2']['mask_uint8']
        path=FULL/'capture'/axis
        data=read_npz(path/'000_zurich_city_09_a_0001.npz');p=read_npz(path/'live_parameters.npz')
        drop=np.array(masks[axis]['row_phase_joint_pair']['mask_uint8'],bool)
        result['axes'][axis]={}
        for label in ['corner','interior']:
            gate,report=run(data,p,label,axis,drop,False)
            previous=prior['axes'][axis][label]
            report['prior_global_service']=previous['global_group2']['service_slots']
            report['prior_four_phase_service']=previous['phase_joint']['service_slots']
            report['net_reduction_vs_unpruned']=1-report['service_slots']/previous['unpruned']['service_slots']
            report['global_control_mask_identical']=True
            result['axes'][axis][label]=report
            np.savez_compressed(HERE/f'{axis}_{label}_row_phase_joint_pair.npz',gate=gate)
            print(axis,label,report['service_slots'],report['prior_global_service'],report['prior_four_phase_service'],flush=True)
    (HERE/'paired_phase_execution.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
