"""Small independent direction-prior boundary checks; CPU only."""
import copy
import json
import unittest

import torch

from direction_prior_control import compute_control, label_counts, modal_direction, summarize_frames


CONTRACT={'direction':'forward','time_interval':'synthetic pair','units':'original_image_pixels'}


def targets(values,valid=None):
    y=torch.tensor(values,dtype=torch.float32).reshape(1,1,-1,2)
    v=torch.ones(y.shape[:-1],dtype=torch.bool) if valid is None else torch.tensor(valid,dtype=torch.bool).reshape(y.shape[:-1])
    y[~v]=float('nan')
    return {'schema':'flow_runtime_coarse_targets_v1','flow_contract':CONTRACT,
            'flow':y,'valid':v,'valid_pixel_count':v.long()}


class DirectionPriorChecks(unittest.TestCase):
    def test_fixed_tie_and_all_eight_direction_classes(self):
        self.assertEqual(modal_direction([3,3,1,1,0,0,0,0]),0)
        self.assertEqual(modal_direction([0,3,3,1,0,0,0,0]),1)
        self.assertIsNone(modal_direction([0]*8))
        t=targets([[2,0],[2,2],[0,2],[-2,2],[-2,0],[-2,-2],[0,-2],[2,-2]])
        self.assertEqual(label_counts(t)['direction_class_counts'],[1]*8)

    def test_low_motion_empty_and_all_eligible_denominator(self):
        t=targets([[0,0],[.5,0],[1,0],[0,2],[9,9]],valid=[True,True,True,True,False])
        r=label_counts(t)
        self.assertEqual(r['valid_GT_tiles'],4);self.assertEqual(r['empty_GT_tiles'],1)
        self.assertEqual(r['GT_eligible_tiles'],2)
        self.assertEqual(summarize_frames([r],0)['summary']['pooled_tile_direction_accuracy'],.5)
        empty=label_counts(targets([[0,0]],valid=[False]))
        summary=summarize_frames([empty],None)['summary']
        self.assertIsNone(summary['macro_frame_direction_accuracy'])
        self.assertIsNone(summary['pooled_tile_direction_accuracy'])
        json.dumps(summary,allow_nan=False)

    def test_macro_is_frame_equal_and_empty_frames_do_not_fake_zero(self):
        a=label_counts(targets([[2,0]]))
        b=label_counts(targets([[0,2],[0,2],[0,2]]))
        empty=label_counts(targets([[0,0]]))
        r=summarize_frames([a,b,empty],0)['summary']
        self.assertEqual(r['macro_frame_direction_accuracy'],.5)
        self.assertEqual(r['pooled_tile_direction_accuracy'],.25)
        self.assertEqual(r['frames_without_GT_eligible_tiles'],1)

    def test_development_class_distribution_cannot_choose_mode(self):
        frames=[]
        for role,values in [('fit',[[2,0],[0,2],[2,0]]),('dev',[[0,-2],[0,-2],[0,-2]])]:
            frames.append({'target_id':role+'_scene_1','sequence':role+'_scene','role':role,
                'stages':{'0':{'descriptor':torch.zeros(1,1,3,2),'targets':targets(values)}}})
        cache={'schema':'flow_coarse_feature_cache_v1','flow_contract':CONTRACT,'frames':frames}
        a=compute_control(cache)
        changed=copy.deepcopy(cache)
        changed['frames'][1]['stages']['0']['targets']=targets([[-2,0],[-2,0],[-2,0]])
        b=compute_control(changed)
        self.assertEqual(a['stages']['0']['selected_direction'],'E')
        self.assertEqual(a['stages']['0']['selected_direction'],b['stages']['0']['selected_direction'])
        self.assertEqual(a['stages']['0']['fit_class_counts_for_selection'],b['stages']['0']['fit_class_counts_for_selection'])
        self.assertEqual(a['stages']['0']['evaluation']['dev']['summary']['pooled_tile_direction_accuracy'],0.)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
