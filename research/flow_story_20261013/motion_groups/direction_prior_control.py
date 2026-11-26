#!/usr/bin/env python3
"""Post-hoc ordinary direction-prior control; fit-only labels, no head training.

The deployment predictor is a fixed class per stage with no runtime feature or
GT input. Only the fit split chooses its class; dev labels score it afterward.
This is a direction control, not an optical-flow field or an EPE-optimal model.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'

import argparse
import json
from pathlib import Path
import sys

import torch

from fit_coarse_cache import load_verified_cache, sha_file, validate_cache
from group_policy import quantize_offset


DIRECTIONS = ((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1))
NAMES = ('E','SE','S','SW','W','NW','N','NE')
EXPECTED_FIT_SHA = 'd93869c693faa507923e8b85a8f2cff89cc0a4b502645ec52afa7608bf8f37a3'


def label_counts(targets):
    valid = targets['valid']
    counts = [0] * 8
    for dx,dy in targets['flow'][valid].tolist():
        direction = quantize_offset(dx,dy,tile_size_pixels=(1.,1.),
                                    association_mode='direction_one_hop',min_motion_pixels=1.)
        if direction is not None:
            counts[DIRECTIONS.index(direction)] += 1
    return {'direction_class_counts': counts, 'GT_eligible_tiles': sum(counts),
            'valid_GT_tiles': int(valid.sum()), 'empty_GT_tiles': valid.numel()-int(valid.sum())}


def modal_direction(fit_counts):
    if not sum(fit_counts):
        return None
    # First maximum implements the explicitly frozen eight-class tie order.
    return max(range(8),key=lambda index:fit_counts[index])


def summarize_frames(rows, selected):
    scored=[]
    for row in rows:
        row=dict(row)
        row['correct_tiles'] = row['direction_class_counts'][selected] if selected is not None else 0
        row['direction_accuracy_GT_eligible'] = (row['correct_tiles']/row['GT_eligible_tiles']
                                                if row['GT_eligible_tiles'] else None)
        scored.append(row)
    eligible_frames=[r for r in scored if r['GT_eligible_tiles']]
    eligible=sum(r['GT_eligible_tiles'] for r in scored)
    valid=sum(r['valid_GT_tiles'] for r in scored)
    summary={
        'frame_count':len(scored), 'frames_with_GT_eligible_tiles':len(eligible_frames),
        'frames_without_GT_eligible_tiles':len(scored)-len(eligible_frames),
        'valid_GT_tiles':valid, 'empty_GT_tiles':sum(r['empty_GT_tiles'] for r in scored),
        'GT_eligible_tiles':eligible,
        'GT_eligible_fraction_valid_tiles':eligible/valid if valid else None,
        'direction_class_counts':[sum(r['direction_class_counts'][i] for r in scored) for i in range(8)],
        'macro_frame_direction_accuracy':(sum(r['direction_accuracy_GT_eligible'] for r in eligible_frames)/len(eligible_frames)
                                          if eligible_frames else None),
        'pooled_tile_direction_accuracy':sum(r['correct_tiles'] for r in scored)/eligible if eligible else None,
        'abstained_GT_eligible_tiles':eligible if selected is None else 0,
    }
    return {'summary':summary,'frames':scored}


def compute_control(cache, strict_population=False):
    membership=validate_cache(cache,strict_population=strict_population)
    output={'schema':'flow_coarse_direction_prior_v1','status':'COMPLETE',
        'post_hoc_control':True,
        'declaration':'Ordinary stronger direction control added after initial head results; not originally preregistered.',
        'rule':'Per-stage mode of fit-only eligible GT tile-mean directions; pooled tile votes; fixed first-class tie.',
        'runtime_inputs':[], 'dev_used_to_choose_direction':False,
        'tie_order':list(NAMES),'direction_offsets_xy':[list(d) for d in DIRECTIONS],
        'eligibility':'Valid GT tile mean displacement norm>=1 original-image pixel; same quantize_offset direction_one_hop',
        'accuracy_denominator':'All GT-eligible tiles, including all errors; no predicted-motion filtering.',
        'macro_definition':'Mean of per-frame accuracy over frames with at least one GT-eligible tile; empty frames counted.',
        'scope':'Backbone-seen training-side fit20/dev8 diagnostic; no new flow model, full-model AEE or hardware benefit.',
        'flow_contract':cache['flow_contract'],'membership':membership,'stages':{}}
    for stage in membership['stages']:
        roles={'fit':[],'dev':[]}
        for frame in cache['frames']:
            row=label_counts(frame['stages'][stage]['targets'])
            row.update(target_id=frame['target_id'],sequence=frame['sequence'])
            roles[frame['role']].append(row)
        counts=[sum(row['direction_class_counts'][i] for row in roles['fit']) for i in range(8)]
        selected=modal_direction(counts)
        output['stages'][stage]={
            'selected_direction':NAMES[selected] if selected is not None else None,
            'selected_class_index':selected,
            'fixed_direction_offset_xy':list(DIRECTIONS[selected]) if selected is not None else None,
            'selection_status':'FIT_MODE_SELECTED' if selected is not None else 'NO_FIT_ELIGIBLE_TILES_ABSTAIN',
            'fit_class_counts_for_selection':counts,
            'evaluation':{role:summarize_frames(rows,selected) for role,rows in roles.items()}}
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--cache-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Output exists; refusing overwrite')
    here=Path(__file__).resolve().parent
    if sha_file(here/'fit_coarse_cache.py') != EXPECTED_FIT_SHA:
        raise ValueError('Frozen cache validation helper changed')
    torch.set_num_threads(2)
    cache,_=load_verified_cache(args.cache,args.cache_sha256,strict_population=True)
    result=compute_control(cache,strict_population=True)
    result['provenance']={'cache_path':str(args.cache.resolve()),'cache_sha256':args.cache_sha256,
        'source_sha256':{name:sha_file(here/name) for name in
                        ('direction_prior_control.py','fit_coarse_cache.py','coarse_head.py','group_policy.py')},
        'torch':str(torch.__version__),'python':sys.version.split()[0],'device':'cpu','GPU_used':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':result['status'],'output':str(args.output),
        'dev':{stage:{'direction':item['selected_direction'],
                     'macro':item['evaluation']['dev']['summary']['macro_frame_direction_accuracy'],
                     'pooled':item['evaluation']['dev']['summary']['pooled_tile_direction_accuracy']}
               for stage,item in result['stages'].items()}}))


if __name__=='__main__':
    main()
