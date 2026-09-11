"""New row-phase masks versus remeasured global and original unpruned."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    run = json.loads((HERE/'run.json').read_text())
    old = json.loads((HERE.parent/'aee_results/run.json').read_text())
    result = dict(complete=run['complete'], scope='No-training diverse10 on the two original fixed students; horizontal-P2 equal mask, two vertical phases.',
                  axes={})
    for axis, value in run['axes'].items():
        stages = value['stages']['diverse10']
        old_global = {r['file']:r for r in old['axes'][axis]['stages']['diverse10']['global_group2']['frames']}
        baseline = {r['file']:r for r in old['axes'][axis]['stages']['diverse10']['unpruned']['frames']}
        new_global = stages['global_joint_pair']['frames']
        difference = max(abs(r['AEE']-old_global[r['file']]['AEE']) for r in new_global)
        result['axes'][axis] = dict(remeasured_global_max_abs_AEE_difference_from_old=difference, modes={})
        if difference:
            raise ValueError('Unchanged global mask did not reproduce its original AEE.')
        for mode, row in stages.items():
            frames = row['frames']
            if any(r['sn2_activity']['nonzero_before'] != baseline[r['file']]['sn2_activity']['nonzero_before'] for r in frames):
                raise ValueError('Pre-mask sn2 producer unexpectedly changed.')
            result['axes'][axis]['modes'][mode] = dict(AEE_frame_mean=row['summary']['AEE_frame_mean'],
                AEE_pixel_mean=row['summary']['AEE_pixel_mean'], delta_unpruned=row['delta_frame_mean'],
                delta_global=row['delta_first_mode_frame_mean'],
                holdout9_delta_unpruned=row['holdout_delta_frame_mean'],
                holdout9_delta_global=row['holdout_delta_first_mode_frame_mean'],
                removed_nonzero_fraction=sum(r['sn2_activity']['nonzero_removed'] for r in frames)/
                    sum(r['sn2_activity']['nonzero_before'] for r in frames), drop_groups=row['drop_groups'])
    (HERE/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
