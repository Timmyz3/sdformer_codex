import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import reduce_dense_raw_v1 as m

class Tests(unittest.TestCase):
    def test_exact_new_plan(self):
        rows=[{'target_id':str(i)} for i in range(8)]
        p=m.plan(rows)
        self.assertEqual(len(p),120)
        self.assertEqual(sum(x[2]=='native' for x in p),8)
        self.assertEqual(sum(x[2]=='expanded' for x in p),112)
        self.assertFalse(any(x[2]=='compact' for x in p))
        self.assertEqual(p[14],('0','fine_2_3','expanded'))
        self.assertEqual(p[15],('1','native','native'))

    def reject_before_GT(self,audit_status,result_status):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);result=root/'results.json'
            result.write_text(json.dumps({'schema':'flow_r17_dense_reference_v1','status':result_status}))
            audit={'schema':'flow_r17_dense_reference_independent_execution_audit_v1','status':audit_status,
                   'quality_analysis_allowed':audit_status=='PASS',
                   'source_result_sha256':m.sha(result),'auditor_sha256':'a','pre_result_freeze_sha256':'b','source_freeze_sha256':'c',
                   'counts':{'full_forwards':120,'native':8,'expanded':112,'compact':0,'FC1_rows':66960000,'FC2_rows':66960000,
                             'BN_calls':9360,'archives':120,'head_payloads':480,'capture_payloads':64,
                             'optimizer_steps':0,'calibration_forwards':0,'observed_FC_modules':2880,
                             'compact_target_calls':0,'BN_input_elements':1},'numerical_checks':[{'pass':True} for _ in range(8)]}
            (root/'audit.json').write_text(json.dumps(audit))
            freeze={'actual_audit_relative_path':'audit.json','external_sources':{
                'execution_auditor':{'sha256':'a'},'execution_audit_freeze':{'sha256':'b'},'pilot_source_freeze':{'sha256':'c'}}}
            args=argparse.Namespace(freeze=Path('unused'),freeze_sha256='unused',result_dir=root)
            with patch.object(m,'REPO',root),patch.object(m,'source_preflight',return_value=freeze),patch.object(m,'load_math') as metric:
                with self.assertRaises(ValueError):m.reduce(args)
                metric.assert_not_called()

    def test_prefix_audit_cannot_reach_GT(self):
        self.reject_before_GT('PASS_PREFIX_EVIDENCE_ONLY','DENSE_REFERENCE120_COMPLETE')

    def test_incomplete_result_cannot_reach_GT(self):
        self.reject_before_GT('PASS','INCOMPLETE_OR_NUMERICAL_FAILURE')

if __name__=='__main__':unittest.main()
