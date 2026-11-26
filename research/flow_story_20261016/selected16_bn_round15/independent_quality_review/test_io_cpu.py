"""Synthetic admission and retained-BN provenance checks; no real quality."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import reduce_raw as io


def fixture():
    ids = ['s_' + str(i) for i in range(8)]
    pins = {'execution_auditor_sha256': 'a'*64, 'execution_audit_freeze_sha256': 'b'*64}
    case = {'checkpoint_step':271482, 'checkpoint_sha256':io.CHECKPOINT_SHA, 'source_state_sha256':io.SOURCE_SHA,
            'initial_selected16_state_sha256':io.SELECTED_SHA, 'final_selected16_state_sha256':io.SELECTED_SHA,
            'full_forward_count':8, 'state_unchanged':True, 'BN_buffers_unchanged':True, 'all_hooks_removed':True, 'original_modules_restored':True}
    result = {'binding':{'BN_scope':io.BN_SCOPE,'checkpoint_sha256':io.CHECKPOINT_SHA}, 'checkpoints':{'271482':case}}
    audit = {'schema':'flow_selected16_round15_independent_execution_audit_v1','status':'PASS','source_result_sha256':'c'*64,
        'auditor_sha256':'a'*64,'audit_freeze_sha256':'b'*64,'full_forwards':8,'optimizer_updates':0,'calibration_forwards':0,
        'dual_forwards':0,'FF_forwards':0,'quality_prediction_count':8,'quality_archive_count':8,'retained_BN_archive_count':1,
        'state_tensors':525,'state_elements':54944214,'fixed_BN_count':16,'current_BN_count':62,'retained_BN_buffer_count':48,
        'executed_FC1_rows':4464000,'executed_FC2_rows':4464000,'BN_scope':io.BN_SCOPE,'checkpoint_sha256':io.CHECKPOINT_SHA,
        'source_state_sha256':io.SOURCE_SHA,'non_BN_state_sha256':io.NON_BN_SHA,'selected16_state_sha256':io.SELECTED_SHA,
        'retained_BN_values_match_own_source':True,'references':{'current78':{},'private24':{}},
        'quality_predictions':[{'checkpoint_step':271482,'target_id':tid,'layout':'native','array_sha256':'d'*64} for tid in ids]}
    for key in ('all_states_unchanged','BN_buffers_unchanged','all_hooks_removed','original_modules_restored'):
        result[key]=audit[key]=True
    return result,audit,pins,ids


class Admission(unittest.TestCase):
    def test_complete_native8(self):
        r,a,p,i=fixture();self.assertEqual(len(io.bind_execution(r,a,'c'*64,p,i)),8)

    def test_bad_admission_rejected(self):
        mutations=[
            lambda r,a:a.update(status='PENDING'),lambda r,a:a.update(source_result_sha256='x'),
            lambda r,a:a.update(auditor_sha256='x'),lambda r,a:a.update(audit_freeze_sha256='x'),
            lambda r,a:a.update(full_forwards=9),lambda r,a:a.update(optimizer_updates=1),
            lambda r,a:a.update(calibration_forwards=176),lambda r,a:a.update(dual_forwards=1),lambda r,a:a.update(FF_forwards=8),
            lambda r,a:a.update(quality_prediction_count=7),lambda r,a:a.update(quality_archive_count=7),
            lambda r,a:a.update(retained_BN_archive_count=0),lambda r,a:a.update(fixed_BN_count=24),
            lambda r,a:a.update(current_BN_count=54),lambda r,a:a.update(retained_BN_buffer_count=72),
            lambda r,a:a.update(state_tensors=477),lambda r,a:a.update(state_elements=54919238),
            lambda r,a:a.update(executed_FC1_rows=0),lambda r,a:a.update(BN_scope='ALL78_CURRENT'),
            lambda r,a:a.update(checkpoint_sha256='x'),lambda r,a:a.update(source_state_sha256='x'),
            lambda r,a:a.update(non_BN_state_sha256='x'),lambda r,a:a.update(retained_BN_values_match_own_source=False),
            lambda r,a:r['checkpoints']['271482'].update(final_selected16_state_sha256='x'),
            lambda r,a:a['quality_predictions'].__setitem__(1,copy.deepcopy(a['quality_predictions'][0])),
            lambda r,a:a['quality_predictions'][0].update(layout='FF'),
        ]
        for index,mutate in enumerate(mutations):
            with self.subTest(index=index):
                r,a,p,i=fixture();mutate(r,a)
                with self.assertRaises(ValueError):io.bind_execution(r,a,'c'*64,p,i)

    def test_actual_own_bn_subset_and_value_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); k=root/'K'; (k/'run_v1').mkdir(parents=True)
            values={}
            for stage,depth in ((0,2),(1,2),(2,6),(3,2)):
                for block in range(depth):
                    for bn in (1,2):
                        prefix=f'sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.{block}.mlp.bn{bn}.norm_layer'
                        for suffix in ('running_mean','running_var','num_batches_tracked'):
                            values[prefix+'.'+suffix]=np.array(176,dtype=np.int64) if suffix=='num_batches_tracked' else np.ones(3,dtype=np.float32)
            selected={key:v.copy() for key,v in values.items() if '.layers.0.' in key or '.layers.2.' in key}
            source=k/'run_v1/bn_buffers.npz';np.savez(source,**values)
            receipt=k/'run_v1/bn_receipt.json';receipt.write_text('{}\n')
            dest=root/'selected.npz';np.savez(dest,**selected)
            bnsha,recsha=io.sha(source),io.sha(receipt)
            r={'retained_BN_buffers':{'file':dest.name,'sha256':io.sha(dest),'source_own72_sha256':bnsha,
                'source_own_receipt_sha256':recsha,'keys':sorted(selected)},
                'binding':{'own_BN_buffers_sha256':bnsha,'own_BN_receipt_sha256':recsha}}
            with patch.multiple(io,K=k,OWN_BN_SHA=bnsha,OWN_RECEIPT_SHA=recsha):
                self.assertEqual(io.verify_retained_BN(r,root)['retained_buffers'],48)
                key=next(k for k in selected if k.endswith('running_mean'))
                selected[key][0]+=1;np.savez(dest,**selected);r['retained_BN_buffers']['sha256']=io.sha(dest)
                with self.assertRaises(ValueError):io.verify_retained_BN(r,root)
                selected[key][0]-=1
                moved=key.replace('.layers.0.','.layers.1.');selected[moved]=selected.pop(key)
                np.savez(dest,**selected);r['retained_BN_buffers'].update(sha256=io.sha(dest),keys=sorted(selected))
                with self.assertRaises(ValueError):io.verify_retained_BN(r,root)

    def test_sibling_file_names(self):
        self.assertEqual(io.sibling(Path('/tmp/x'),'a.npz'),Path('/tmp/x/a.npz'))
        for name in ('','..','../a.npz','/tmp/a.npz','a/b'):
            with self.subTest(name=name),self.assertRaises(ValueError):io.sibling(Path('/tmp/x'),name)


if __name__=='__main__':unittest.main()
