import importlib.util
import unittest
from pathlib import Path
import numpy as np
import raw_quality_math as r

REPO=Path(__file__).resolve().parents[4]
OLD=REPO/'research/flow_story_20261016/coverage_recovery_round13/independent_quality_review/reduce_raw_quality.py'

class Tests(unittest.TestCase):
    def test_frozen_region_and_metric_agreement(self):
        self.assertEqual(r.hashlib.sha256(OLD.read_bytes()).hexdigest(),'26a2e5c97b988f10685e1c440a11506354d9591cb9b36a567292abb6d0348088')
        spec=importlib.util.spec_from_file_location('_r27_frozen_region_only',OLD)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        rng=np.random.default_rng(20261019)
        for h,w in ((3,4),(9,11),(17,20)):
            gt=rng.normal(size=(2,h,w)).astype(np.float32)*9
            mask=rng.random((h,w))>.25;mask[0,0]=True
            events=rng.integers(-1,2,size=(10,h,w)).astype(np.float32)
            events[:,:,:2]=0
            pred=gt+rng.normal(size=gt.shape).astype(np.float32)
            a=r.region_masks(gt,mask,events,7.,12.);b=old.masks_from_original(gt,mask,events,7.,12.)
            self.assertTrue(all(np.array_equal(a[k],b[k]) for k in r.REGIONS))
            self.assertEqual(r.measure(pred,gt,a),old.measure(pred,gt,b))
    def test_all_gt_and_voxel_counts(self):
        gt=np.zeros((2,5,6),np.float32);gt[0,2,3]=500
        mask=np.ones((5,6),bool);events=np.zeros((10,5,6),np.float32)
        events[0,2,3]=-3.;events[3,2,3]=10.
        masks=r.region_masks(gt,mask,events,500.,500.)
        self.assertEqual(int(masks['all_valid'].sum()),30)
        self.assertEqual(int(masks['B'].sum()),0) # strictly greater threshold
        self.assertEqual(int(masks['L'].sum()),0)
        self.assertEqual(r.measure(np.zeros_like(gt),gt,masks)['all_valid']['error_sum'],500.)
        stored=r.artifact_arrays(gt,mask,events,masks)
        self.assertEqual(stored['voxel_nonzero_bins'][2,3],2) # not 13 or -3+10
        self.assertEqual(len(stored),19)
    def test_invalid_source_and_prediction_rejected(self):
        gt=np.zeros((2,3,4),np.float32);mask=np.ones((3,4),bool);events=np.zeros((10,3,4),np.float32)
        with self.assertRaises(ValueError):r.region_masks(gt,mask,np.zeros((9,3,4),np.float32),1.,1.)
        with self.assertRaises(ValueError):r.region_masks(gt,np.zeros_like(mask),events,1.,1.)
        masks=r.region_masks(gt,mask,events,1.,1.);pred=gt.copy();pred[0,0,0]=np.nan
        with self.assertRaises(ValueError):r.measure(pred,gt,masks)

if __name__=='__main__':unittest.main()
