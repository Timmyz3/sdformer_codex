"""Pair the new fixed-deployment AEE against both original no-training controls."""
import json
from pathlib import Path
import statistics
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
STAGE = HERE/'stage64'


def main():
    run = json.loads((STAGE/'run.json').read_text())
    before = json.loads((HERE.parent/'aee_results/run.json').read_text())['axes']['ordinary']['stages']['diverse10']
    original = before['unpruned']['frames']
    calibration = run['mask_calibration_frame']
    source = BASE/'algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40/schedule_compare_same_port/full_chain/capture_full_producers/ordinary/parameters.npz'
    with np.load(source) as z:
        old_constants = {k:z[k] for k in z.files}
    result = dict(scope='Fixed64-step paired recovery, actual original fixed helper deployment, diverse10 only; no full825/no hardware timing.',
        unpruned_AEE_frame_mean=before['unpruned']['summary']['AEE_frame_mean'], axes={})
    for mode, axis in run['axes'].items():
        rows = json.loads((STAGE/mode/(mode+'_frames.json')).read_text())
        prior = {r['file']:r for r in before[mode]['frames']}
        baseline = {r['file']:r for r in original}
        paired = [dict(file=r['file'], valid_pixels=r['valid_pixels'],
            AEE=r['AEE'], unpruned_AEE=baseline[r['file']]['AEE'],
            untrained_mask_AEE=prior[r['file']]['AEE'],
            delta_unpruned=r['AEE']-baseline[r['file']]['AEE'],
            delta_untrained_mask=r['AEE']-prior[r['file']]['AEE']) for r in rows]
        held = [r for r in paired if r['file'] != calibration]
        with np.load(STAGE/(mode+'_deployed_constants.npz')) as z:
            changed = [k for k in z.files if k not in old_constants or not np.array_equal(z[k],old_constants[k])]
        if set(changed)-{'U_conv2_theta_q16','F_q16'}:
            raise ValueError('Unexpected changed deployed constant: '+repr(changed))
        result['axes'][mode] = dict(AEE_frame_mean=axis['evaluation']['AEE_frame_mean'],
            AEE_pixel_mean=axis['evaluation']['AEE_pixel_mean'],
            delta_unpruned=statistics.mean(r['delta_unpruned'] for r in paired),
            delta_untrained_mask=statistics.mean(r['delta_untrained_mask'] for r in paired),
            holdout9_delta_unpruned=statistics.mean(r['delta_unpruned'] for r in held),
            holdout9_delta_untrained_mask=statistics.mean(r['delta_untrained_mask'] for r in held),
            changed_deployed_constant_fields=changed,
            gradient_smoke=axis['gradient_smoke'], deployment_coefficient_check=axis['deployment_coefficient_check'],
            final_QAT_vs_deployed_differences=axis['final_QAT_vs_deployed_differences'],
            updates=axis['updates'], paired=paired)
    result['phase_minus_global_AEE_frame_mean'] = (result['axes']['phase_joint']['AEE_frame_mean']-
        result['axes']['global_group2']['AEE_frame_mean'])
    (HERE/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{m:v for m,v in a.items() if m not in ['paired','gradient_smoke']}
                      for k,a in result['axes'].items()},indent=2))


if __name__ == '__main__':
    main()
