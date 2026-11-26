#!/usr/bin/env python3
"""Replay saved selector proposals only: no model, arrays, GT fitting or GPU."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import select_joint as joint

HERE=Path(__file__).resolve().parent
OLD_SHA='9d7ece29230b891a961fe00855a92bfde0d149862de60d1db97a571a848ee519'
ARMS={'independent':'independent_trained','motion':'motion_trained',
      'rotated90':'rotated90_trained','horizontal_local':'horizontal_local_trained'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True);parser.add_argument('--results-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--old-dp',type=Path,default=HERE.parents[1]/'flow_story_20261013/motion_groups/exact_budget_policy.py')
    args=parser.parse_args()
    joint.require(not args.output.exists(),'New replay output required')
    joint.require(sha(args.results)==args.results_sha256 and sha(args.old_dp)==OLD_SHA,'Source hash differs')
    source=json.loads(args.results.read_text())
    joint.require(source.get('schema')=='flow_selection_controls_results_v1' and
                  source.get('status')=='SELECTION_CONTROLS_COMPLETE' and source.get('complete_frames')==4,
                  'Complete four-target fixed control result required')
    spec=importlib.util.spec_from_file_location('joint_replay_old_dp',args.old_dp)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    started=time.monotonic();frames=[]
    aggregate={'free_vs_old_independent_exact_checks':0,'legacy_vs_old_topology_exact_checks':0,
               'non_independent_topology_cases':12,'at_least_one_mask_changed_from_free':0,
               'all_or_none_tile_sum_mask_changed_from_legacy_max':0,'infeasible_cases':[]}
    for frame in source['frames']:
        base=frame['arms'][ARMS['independent']]['runtime_selector']['proposal']
        old_independent=old.select_exact(base,900)
        item={'target_id':frame['target_id'],'sequence':frame['sequence'],'topologies':{}}
        for policy,arm in ARMS.items():
            selector=frame['arms'][arm]['runtime_selector'];p=selector['proposal']
            signature=lambda v:[(t['tile_id'],t['valid_points'],t['slow_service_cost'],t['score']) for t in v['tiles']]
            joint.require(signature(p)==signature(base),'Saved topology changed scores/costs')
            decisions={rule:joint.select_exact(p,900,rule) for rule in joint.RULES}
            for key in ('feasible','slow_mask','selected_tile_ids','objective_score_exact','slow_service_cost'):
                joint.require(decisions['free'][key]==old_independent[key],'Free differs from original independent: '+key)
            aggregate['free_vs_old_independent_exact_checks']+=1
            legacy=old.select_exact(p,900)
            for key in ('feasible','slow_mask','selected_tile_ids','objective_score_exact','slow_service_cost','selected_group_ids'):
                joint.require(decisions['legacy_group_max'][key]==legacy[key],'Legacy differs from old topology: '+key)
            joint.require(legacy['slow_mask']==frame['arms'][arm]['execution'][0]['keep'],
                          'Original old DP does not reproduce saved deployed mask')
            aggregate['legacy_vs_old_topology_exact_checks']+=1
            new,free,oldscore=decisions['at_least_one'],decisions['free'],decisions['legacy_group_max']
            changed=lambda x,y:None if not x['feasible'] or not y['feasible'] else len(set(x['selected_tile_ids'])^set(y['selected_tile_ids']))
            diagnostics={'free_vs_old_independent_exact':True,'legacy_vs_saved_exact':True,
                'free_pair00_count':free['pair_state_counts']['00'],
                'at_least_one_symmetric_difference_from_free':changed(new,free),
                'all_or_none_tile_sum_symmetric_difference_from_legacy_max':changed(decisions['all_or_none'],oldscore),
                'meaning':'Mask/constraint activity only; no new prediction, AEE, recovery or PE measurement'}
            if policy!='independent':
                aggregate['at_least_one_mask_changed_from_free']+=int(bool(diagnostics['at_least_one_symmetric_difference_from_free']))
                aggregate['all_or_none_tile_sum_mask_changed_from_legacy_max']+=int(bool(diagnostics['all_or_none_tile_sum_symmetric_difference_from_legacy_max']))
            for rule,selection in decisions.items():
                if not selection['feasible']:aggregate['infeasible_cases'].append({'target_id':frame['target_id'],'topology':policy,'rule':rule,'target_cost':900})
            item['topologies'][policy]={'diagnostics':diagnostics,'selections':decisions}
        frames.append(item)
    result={'schema':'flow_joint_admission_saved_proposal_replay_v1','status':'REPLAY_COMPLETE',
        'source_results':str(args.results.resolve()),'source_results_sha256':args.results_sha256,
        'module_sha256':sha(joint.__file__),'replay_sha256':sha(__file__),'old_DP_sha256':OLD_SHA,
        'target_cost':900,'source_info':'Original saved runtime coarse-flow magnitude, unchanged proposal topology and true point costs',
        'new_model_or_GPU_work':False,'GT_used_by_new_rule':False,'quality_or_hardware_claim':False,
        'scope':'Same four repeatedly explored, backbone-seen dev targets; only selector mechanism replay, not recovery/quality evidence',
        'summary':aggregate,'frames':frames,'wall_seconds':time.monotonic()-started}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'summary':aggregate,'wall_seconds':result['wall_seconds']}))


if __name__=='__main__':main()
