"""Bounded CPU geometry/provenance tests; optional real cache check, no training."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import torch
import derive_matched_support_cache as d
from coarse_head import make_tile_targets

FLOW = {'direction': 'forward', 'time_interval': 'fixture', 'units': 'original_image_pixels'}
REAL_SHA = '918c78b37883043a826f79f84a5aec1a63864bc34ebd2b784a7f7a3203b4a5c3'


def fixture():
    frames = []
    gt = torch.ones(2,480,640); valid = torch.ones(480,640,dtype=torch.bool)
    stages = {}
    for stage, hw, stride, channels in (('0',(120,160),4,96), ('2',(30,40),16,384)):
        counts = d.real_tile_counts(hw)
        stages[stage] = {'descriptor': torch.arange(counts.numel(),dtype=torch.float32).reshape(1,*counts.shape,1).expand(1,*counts.shape,channels).clone(),
            'targets': make_tile_targets(gt,valid,feature_hw=hw,feature_stride_pixels=(stride,stride),flow_contract=FLOW)}
    for role, n in (('fit',20), ('dev',8)):
        for i in range(n):
            frames.append({'target_id':role+'_seq_'+str(i), 'sequence':role+'_seq', 'role':role, 'stages':copy.deepcopy(stages)})
    return {'schema':'flow_coarse_feature_cache_v1', 'flow_contract':FLOW, 'frames':frames,
        'BN_scope':'MLP_CMA176', 'backbone_seen':True, 'label_geometry':'logical_cell_support_origin0_v1',
        'checkpoint_sha256':'a'*64, 'contract_sha256':'b'*64}


class PoolTests(unittest.TestCase):
    def test_manual_tail_weights(self):
        # H5 W7, 2x2 source cells: rows2,2,1; cols2,2,2,1.
        x = torch.arange(12,dtype=torch.float32).reshape(1,3,4,1)
        out, count = d.coarsen_descriptor(x,(5,7),tile_size=2,group_size=2)
        expected = torch.tensor([[(0+1+4+5)/4,(2*4+3*2+6*4+7*2)/12],
                                 [(8*2+9*2)/4,(10*2+11)/3]])
        torch.testing.assert_close(out[0,:,:,0], expected)
        self.assertEqual(count.tolist(),[[16,12],[4,3]])
        self.assertNotEqual(float(out[0,1,1,0]),10.5)

    def test_native_coverage_and_tail(self):
        source = d.real_tile_counts((120,160))
        self.assertEqual(source.shape,torch.Size([14,18])); self.assertEqual(int(source[-1,-1]),21)
        out, count = d.coarsen_descriptor(torch.ones(1,14,18,96))
        self.assertEqual(tuple(out.shape),(1,4,5,96))
        self.assertEqual(count.tolist(),[[1296]*4+[576]]*3+[[432]*4+[192]])
        self.assertEqual(int(count.sum()),19200)
        self.assertTrue(torch.equal(out,torch.ones_like(out)))
        expected_support=d.real_tile_counts((30,40))*16*16
        self.assertTrue(torch.equal(count*4*4,expected_support))

    def test_each_source_group_exactly_once(self):
        n=14*18; x=torch.eye(n).reshape(1,14,18,n)
        out,group=d.coarsen_descriptor(x)
        reconstructed=(out.double()*group[None,:,:,None]).sum((0,1,2))
        torch.testing.assert_close(reconstructed,d.real_tile_counts((120,160)).reshape(-1).double(),rtol=1e-7,atol=1e-5)

    def test_invalid_input(self):
        for value in (torch.ones(1,13,18,96),torch.ones(1,14,18,96,dtype=torch.int32),torch.full((1,14,18,96),float('nan'))):
            with self.assertRaises(ValueError): d.coarsen_descriptor(value)


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.source=fixture()

    def test_clone_contract_membership_and_no_GT_weight(self):
        source=copy.deepcopy(self.source)
        output,contract,raw=d.derive_cache(source,'c'*64,'d'*64)
        self.assertEqual(output['contract_sha256'],hashlib.sha256(raw).hexdigest())
        self.assertNotEqual(output['contract_sha256'],source['contract_sha256'])
        self.assertEqual(json.loads(raw),contract)
        self.assertEqual(len(output['frames']),28)
        for original, derived in zip(source['frames'],output['frames']):
            self.assertEqual(original['target_id'],derived['target_id'])
            self.assertTrue(torch.equal(original['stages']['2']['descriptor'],derived['stages']['2']['descriptor']))
            self.assertTrue(torch.equal(original['stages']['2']['targets']['flow'],derived['stages']['0']['targets']['flow']))
            self.assertNotEqual(original['stages']['2']['targets']['flow'].data_ptr(),derived['stages']['0']['targets']['flow'].data_ptr())
        changed=copy.deepcopy(source)
        for f in changed['frames']:
            for item in f['stages'].values():
                item['targets']['flow'].fill_(999.)
                item['targets']['valid_pixel_count'].zero_();item['targets']['valid'].zero_()
        changed_output,_,_=d.derive_cache(changed,'e'*64,'d'*64)
        self.assertTrue(torch.equal(output['frames'][0]['stages']['0']['descriptor'],changed_output['frames'][0]['stages']['0']['descriptor']))
        self.assertEqual(tuple(source['frames'][0]['stages']['0']['descriptor'].shape),(1,14,18,96))

    def test_weights_only_compatible(self):
        output,_,_=d.derive_cache(self.source,'c'*64,'d'*64)
        buf=io.BytesIO();torch.save(output,buf);buf.seek(0)
        loaded=torch.load(buf,weights_only=True,map_location='cpu')
        from fit_coarse_cache import validate_cache
        self.assertEqual(len(validate_cache(loaded,strict_population=True)['target_ids']['fit']),20)

    def test_rederive_and_bad_geometry_rejected(self):
        output,_,_=d.derive_cache(self.source,'c'*64,'d'*64)
        with self.assertRaises(ValueError): d.derive_cache(output,'e'*64,'d'*64)
        changed=copy.deepcopy(self.source)
        changed['frames'][0]['stages']['2']['targets']['image_origin_xy']=[6.,6.]
        with self.assertRaises(ValueError): d.derive_cache(changed,'c'*64,'d'*64)

    def test_real_cache_if_available(self):
        path=Path(__file__).parent/'coarse_cpu_pilot_v2/features.pt'
        if not path.exists(): self.skipTest('Real cache not present; synthetic tests remain sufficient for unit scope')
        self.assertEqual(d.sha(path),REAL_SHA)
        source=torch.load(path,map_location='cpu',weights_only=True)
        output,contract,_=d.derive_cache(source,REAL_SHA,d.sha(d.__file__))
        self.assertEqual(len(output['frames']),28)
        self.assertEqual(contract['membership']['channels'],{'0':96,'2':384})
        for original,derived in zip(source['frames'],output['frames']):
            torch.testing.assert_close(derived['stages']['0']['targets']['flow'],original['stages']['2']['targets']['flow'],rtol=0,atol=0,equal_nan=True)
            torch.testing.assert_close(derived['stages']['2']['descriptor'],original['stages']['2']['descriptor'],rtol=0,atol=0)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    unittest.main(verbosity=2)
