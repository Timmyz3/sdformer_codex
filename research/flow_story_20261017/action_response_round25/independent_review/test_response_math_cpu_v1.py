"""Independent synthetic-only review; no source data/GT or executor imports."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import unittest
from unittest import mock
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'design/response_models.py'
spec = importlib.util.spec_from_file_location('_r25_review_math', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
MEASUREMENTS = []


class Checks(unittest.TestCase):
    def test_response_signed_axes_and_full_pixel_support(self):
        yy, xx = np.indices((480, 640), dtype=np.float32)
        a = np.stack((1000*yy+xx, -2*yy+3*xx))
        b = np.empty_like(a);b[0] = 40;b[1] = -2
        grid = m.response_grid(a, b)
        cy, cx = np.indices((30, 40), dtype=np.float32)
        cy, cx = 16*cy+7.5, 16*cx+7.5
        expected = np.stack((1000*cy+cx-40, -2*cy+3*cx+2), -1)
        np.testing.assert_array_equal(grid, expected)
        np.testing.assert_array_equal(m.response_grid(b, a), -grid)
        impulse = np.zeros((2, 480, 640), np.float32);impulse[1, 479, 639] = -256
        response = m.response_grid(impulse, np.zeros_like(impulse))
        self.assertEqual(response[29, 39, 1], -1)
        self.assertEqual(np.count_nonzero(response), 1)

    def test_shared_1186_features_and_h1_residual(self):
        channel = np.arange(384, dtype=np.float32)
        x = np.broadcast_to(channel, (1, 30, 40, 384)).copy()
        features = m.features(x)
        self.assertEqual(features.shape, (12, 1186))
        np.testing.assert_array_equal(features[:, :384], np.broadcast_to(channel, (12, 384)))
        np.testing.assert_array_equal(features[:, 384:1152], np.zeros((12, 768)))
        grouped = channel.reshape(16, 24).mean(1)
        np.testing.assert_array_equal(features[:, 1152:1168], np.broadcast_to(grouped, (12, 16)))
        np.testing.assert_array_equal(features[:, 1168:1184], np.zeros((12, 16)))
        np.testing.assert_allclose(features[:, -2:], np.array([(r/2,c/3) for r,c in m.TILES], np.float32), atol=0, rtol=0)
        tile = np.arange(9*9*2,dtype=np.float32).reshape(9,9,2)
        expanded = m.h1_expand(tile)
        for y in range(0,9,3):
            for x in range(0,9,3):
                for low, high in ((y,y+1),(y+1,y+3)):
                    block = tile[low:high,x:x+3]
                    ref = block.reshape(-1,2).sum(0)/len(block.reshape(-1,2))
                    np.testing.assert_array_equal(expanded[low:high,x:x+3], np.broadcast_to(ref,block.shape))

    def test_pca_ridge_against_independent_primal_lowrank_response(self):
        rng = np.random.default_rng(71)
        n = 24
        x = np.zeros((n,1186),np.float32);x[:,:5] = rng.normal(size=(n,5))
        spatial = rng.normal(size=(5,2400));constant = rng.normal(size=2400)
        d = (x[:,:5].astype(np.float64)@spatial + constant).reshape(n,30,40,2)
        model = m.fit_response(x,d,rank=8,alpha=10.)
        test = np.zeros((7,1186),np.float32);test[:,:5] = rng.normal(size=(7,5))
        mean,std = x[:,:5].astype(np.float64).mean(0),x[:,:5].astype(np.float64).std(0)
        z = (x[:,:5]-mean)/std
        target = d.reshape(n,-1);ym=target.mean(0)
        # Independent 5x5 primal system, without fitted PCA coefficients or dual solve.
        beta = np.linalg.solve(z.T@z+10*np.eye(5),z.T@(target-ym))
        reference = ((test[:,:5]-mean)/std)@beta+ym
        actual = m.predict_response(model,test).reshape(7,-1)
        np.testing.assert_allclose(actual,reference,atol=8e-6,rtol=2e-6)
        np.testing.assert_allclose(model['basis']@model['basis'].T,np.eye(8),atol=4e-7,rtol=0)
        for row in model['basis']:
            self.assertGreaterEqual(row[np.argmax(abs(row))],0)
        MEASUREMENTS.append({'independent_primal_prediction_max_abs':float(abs(actual-reference).max()),'synthetic_fit_rows':24,'rank':8})

    def test_twenty_real_tiles_edge_direction_and_scalar_scale(self):
        body=m.StructuredResponse()
        expected=[];counts=[0]*20
        for y in range(30):
            for x in range(40):
                for dy,dx in ((0,1),(1,0)):
                    if y+dy<30 and x+dx<40:
                        expected.append((y*40+x,(y+dy)*40+x+dx,dx,dy));counts[(y//9)*5+x//9]+=1
        self.assertEqual(len(expected),2330)
        self.assertEqual(list(zip(body.src.tolist(),body.dst.tolist(),*body.direction.T.tolist())),expected)
        self.assertEqual([len(getattr(body,'group_'+str(i))) for i in range(20)],counts)
        self.assertEqual(sum(counts),2330);self.assertTrue(all(c>0 for c in counts))
        self.assertEqual(counts[-1],17)
        # Right-neighbor edge from x8 to9 is retained and assigned by its start.
        cross=expected.index((8,9,1,0));self.assertIn(cross,body.group_0.tolist())
        x=torch.arange(30*40,dtype=torch.float32).reshape(1,1,30,40)
        pooled=m.TilePool()(x)
        self.assertEqual(tuple(pooled.shape),(1,20))
        self.assertEqual(float(pooled[0,19]),float(x[:,:,27:30,36:40].mean()))
        rng=np.random.default_rng(4);d=rng.normal(size=(6,30,40,2)).astype(np.float32)
        s=m.motion_scaler(d);self.assertEqual(np.asarray(s['std']).shape,())
        scaled=m.scale(d,s)
        np.testing.assert_allclose(scaled[1]-scaled[0],(d[1]-d[0])/s['std'],atol=8e-7,rtol=2e-6)

    def test_all_three_heads_real_finite_gradients_and_updates(self):
        rng=np.random.default_rng(9)
        x=rng.normal(size=(12,1186)).astype(np.float32)
        d=rng.normal(size=(12,30,40,2)).astype(np.float32)
        target=np.linspace(-2,3,12,dtype=np.float32)
        original_step=torch.optim.AdamW.step
        for kind in ('direct','generic_response','structured_response'):
            torch.manual_seed(m.SEED)
            initial=m.GainHead(kind)
            before={n:p.detach().clone() for n,p in initial.named_parameters()}
            observed=[]
            def reviewed_step(opt,*args,**kwargs):
                gradients=[p.grad for group in opt.param_groups for p in group['params']]
                self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients))
                self.assertTrue(all(bool((g!=0).any()) for g in gradients))
                observed.append(len(gradients))
                return original_step(opt,*args,**kwargs)
            with mock.patch.object(torch.optim.AdamW,'step',reviewed_step):
                fitted=m.fit_head(kind,x,d,target,steps=3)
            self.assertEqual(len(observed),3)
            for name,parameter in fitted['model'].named_parameters():
                self.assertFalse(torch.equal(parameter,before[name]))
                self.assertTrue(bool(torch.isfinite(parameter).all()))
            self.assertTrue(np.isfinite(m.infer_head(fitted,x,d)).all())
            MEASUREMENTS.append({'head':kind,'synthetic_steps':3,'parameter_count':fitted['parameter_count'],'every_parameter_has_finite_nonzero_gradient':True})


if __name__=='__main__':
    torch.set_num_threads(2)
    start=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    receipt={'schema':'flow_r25_response_math_independent_cpu_v1','status':'PASS' if result.wasSuccessful() else 'FAIL',
       'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
       'source_sha256':{'response_models.py':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'protocol':hashlib.sha256((SOURCE.parent/'PROTOCOL_PRE_LABELS_V1.md').read_bytes()).hexdigest(),
          'review_script':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
       'measurements':MEASUREMENTS,'actual_data_or_GT_read':False,'GPU_used':False,
       'synthetic_optimization_steps':9,'wall_seconds':time.monotonic()-start}
    output=HERE/'MATH_CPU_REVIEW_V1.json';assert not output.exists()
    output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
    raise SystemExit(0 if result.wasSuccessful() else 1)
