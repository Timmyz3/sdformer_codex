#!/usr/bin/env python3
import copy
import hashlib
import json
from pathlib import Path
import unittest
import analyze_recovery as a


def summaries():
    return {state+'/'+mode:{scope:{region:{'frame_mean_AEE':1.0} for region in a.REGIONS}
                           for scope in ('overall','seq1','seq2')}
            for state in ('initial',)+a.ARMS for mode in a.ARMS}


class QualityChecks(unittest.TestCase):
    def test_full_conjunction_and_both_candidates(self):
        s=summaries()
        self.assertTrue(all(g['ordinary_base_quality_pass'] for g in a.gates(s).values()))
        s['CC/CC']['seq2']['all_valid']['frame_mean_AEE']=1.04
        z=a.gates(s)
        self.assertFalse(z['CC']['ordinary_base_quality_pass'])
        self.assertTrue(z['fine_s2b2']['ordinary_base_quality_pass'])

    def test_degraded_native_reference_cannot_hide_failure(self):
        s=summaries()
        s['native/native']['overall']['all_valid']['frame_mean_AEE']=1.1
        z=a.gates(s)
        self.assertTrue(z['CC']['matched_recovered_references']['native']['pass'])
        self.assertFalse(z['CC']['normal_references']['native']['pass'])
        self.assertFalse(z['CC']['ordinary_base_quality_pass'])

    def test_shared_weight_fine_reference_is_enforced(self):
        s=summaries()
        s['CC/FF']['seq1']['B']['frame_mean_AEE']=.9
        z=a.gates(s)['CC']
        self.assertTrue(all(v['pass'] for v in z['matched_recovered_references'].values()))
        self.assertFalse(z['own_checkpoint_references']['FF']['pass'])
        self.assertFalse(z['ordinary_base_quality_pass'])

    def test_empty_and_zero_reference(self):
        s=summaries()
        for v in s.values():v['seq1']['L']['frame_mean_AEE']=None
        z=a.gates(s)['CC']
        self.assertTrue(z['ordinary_base_quality_pass'])
        self.assertFalse(z['matched_recovered_references']['native']['scopes']['seq1']['L']['evidence'])
        for v in s.values():v['seq1']['L']['frame_mean_AEE']=0.
        s['CC/CC']['seq1']['L']['frame_mean_AEE']=.001
        self.assertFalse(a.gates(s)['CC']['ordinary_base_quality_pass'])

    def test_frame_mean_is_not_pixel_weighted(self):
        v=a.aggregate([{'valid_pixels':1,'error_sum':2.,'AEE':2.},
                       {'valid_pixels':9,'error_sum':36.,'AEE':4.},
                       {'valid_pixels':0,'error_sum':0.,'AEE':None}])
        self.assertEqual(v['frame_mean_AEE'],3.)
        self.assertEqual(v['pixel_weighted_AEE'],3.8)
        self.assertEqual(v['frames_with_region'],2)


if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(QualityChecks))
    root=Path(__file__).resolve().parent
    out=root/'quality_cpu_results.json'
    assert not out.exists()
    out.write_text(json.dumps({'passed':r.wasSuccessful(),'tests_run':r.testsRun,
        'source_sha256':{n:hashlib.sha256((root/n).read_bytes()).hexdigest()
                         for n in ('analyze_recovery.py','test_quality.py')},
        'scope':'Synthetic gate/aggregation checks only; no R12 quality or GPU data read.'},indent=2)+'\n')
    raise SystemExit(not r.wasSuccessful())
