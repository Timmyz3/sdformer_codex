"""CPU-only service/layout unit tests; not real native PSN or trained accuracy.

The slow fixture exposes real forward calls and native-shaped FC interfaces but
uses cheap expanded arithmetic. FastBranch is the actual fixed12Oct LN+Linear
class, extracted from its pinned source. No checkpoint/data/backbone/GPU runs.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import ast
import copy
import hashlib
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import unittest
import torch
from torch import nn
import fast_tile_adapter as adapter

HERE=Path(__file__).resolve().parent
RESEARCH=HERE.parents[1]
sys.path.insert(0,str(RESEARCH/'flow_story_20261013/breadth'))
import same_shape_tile_mlp as tile


def fast_class():
    path=RESEARCH/'flow_story_20261012/transfer_experiments/dynamicswin/adapter.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='75c97bb1e35e31a4e7b0fb97adefe434e6346dfa156ee6e9c297547efdbe30d7'
    tree=ast.parse(path.read_text())
    nodes=[x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='FastBranch']
    assert len(nodes)==1
    env={'nn':nn}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return env['FastBranch']


FastBranch=fast_class()


class PSN(nn.Module):
    """Synthetic complete-T guard and dependency, NOT the native PSN formula."""
    def __init__(self):
        super().__init__();self.T=10;self.shapes=[]
    def forward(self,x):
        if x.shape[0]!=10: raise ValueError('Synthetic PSN lost completeT10')
        self.shapes.append(tuple(x.shape))
        # Nonlocal in time but channel0-only for bounded CPU work.
        return (.125*x[:1,...,:1]+.25*x[-1:,...,:1]).expand_as(x)


class SpikeWrap(nn.Module):
    def __init__(self):
        super().__init__();self.spiking_neuron=PSN()
    def forward(self,x):return self.spiking_neuron(x)


class CheapLinear(nn.Module):
    """Shapes/hooks are real; no claim of native dense matrix arithmetic."""
    def __init__(self,i,o):
        super().__init__();self.in_features=i;self.out_features=o;self.shapes=[]
    def forward(self,x):
        assert x.shape[-1]==self.in_features
        self.shapes.append(tuple(x.shape))
        return (x[...,:1]*.5-.25).expand(*x.shape[:-1],self.out_features)


class MS_Spiking_Mlp(nn.Module):
    def __init__(self):
        super().__init__();self.norm_layer='BN'
        self.sn1=SpikeWrap();self.sn2=SpikeWrap()
        self.fc1=CheapLinear(384,1536);self.fc2=CheapLinear(1536,384)
        self.bn1=nn.BatchNorm1d(1536);self.bn2=nn.BatchNorm1d(384)
        self.eval()
    def affine(self,x,bn):
        # True frozen-BN buffer contract, cheap channel0 arithmetic only.
        y=(x[...,:1]-bn.running_mean[0])/torch.sqrt(bn.running_var[0]+bn.eps)
        return (y*bn.weight[0]+bn.bias[0]).expand_as(x)
    def forward(self,x):
        x=self.affine(self.fc1(self.sn1(x)),self.bn1)
        return self.affine(self.fc2(self.sn2(x)),self.bn2)


def mask900():
    # Removed300=81+81+36+36+27+27+12; all three tail types exercised.
    mask=torch.ones(4,5,dtype=torch.bool)
    for y,x in ((0,0),(0,1),(0,4),(1,4),(3,0),(3,1),(3,4)):mask[y,x]=False
    return mask


def points(mask):
    return sum(min(9,30-9*y)*min(9,40-9*x) for y in range(4) for x in range(5) if mask[y,x])


def make_fixture():
    native=MS_Spiking_Mlp()
    fast=FastBranch(384).eval().requires_grad_(False)
    mean=torch.linspace(-.5,.5,384)
    return native,fast,mean


class FastTileChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(20261014)
        cls.u=torch.randn(10,1,40,30,384).transpose(2,3)
        assert not cls.u.is_contiguous() and tuple(cls.u.shape)==(10,1,30,40,384)

    def pair(self,branch,mask):
        native,fast,mean=make_fixture();before={k:v.clone() for k,v in native.state_dict().items()}
        policy=lambda name,u:mask.clone()
        actual=adapter.FastSlowTileMLP(native,fast,mean,'actual',branch,policy,tile)
        with torch.no_grad():output=actual(self.u)
        replay=tile.ReplayPolicy(actual.records)
        reference=adapter.FastSlowTileMLP(native,fast,mean,'reference',branch,replay,tile)
        with torch.no_grad():gold=reference(self.u)
        replay.complete()
        self.assertTrue(torch.equal(output,gold));self.assertTrue(output.is_contiguous())
        ar,rr=actual.records[0],reference.records[0]
        for key in ('shape_TBHWC','keep','input_sha256','output_sha256','input_stride','output_stride'):
            self.assertEqual(ar[key],rr[key],key)
        self.assertEqual(ar['input_stride'],list(self.u.stride()))
        self.assertEqual(ar['output_stride'],list(torch.empty(tuple(self.u.shape)).stride()))
        self.assertFalse(ar['all_keep_shortcut']);self.assertFalse(rr['all_keep_shortcut'])
        for k,v in native.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
        for layer in (native.fc1,native.fc2,fast.fc1[1]):self.assertEqual(len(layer._forward_pre_hooks),0)
        self.assertTrue(all(shape[0]==10 for sn in (native.sn1,native.sn2) for shape in sn.spiking_neuron.shapes))
        self.assertEqual({(s[2],s[3]) for s in native.fc1.shapes}, {(9,9),(9,4),(3,9),(3,4)})
        return output,ar,rr,native,fast,mean

    def test_trained900_actual_and_same_shape_reference_exact(self):
        self.assertEqual(points(mask900()),900)
        output,a,r,_,_,_=self.pair('trained',mask900())
        self.assertEqual(a['observed']['slow_fc1_rows'],9000)
        self.assertEqual(a['observed']['slow_fc2_rows'],9000)
        self.assertEqual(a['observed']['fast_fc_rows'],3000)
        self.assertEqual(a['observed']['slow_calls'],13);self.assertEqual(a['observed']['fast_calls'],7)
        self.assertEqual(a['fast_LN_rows'],3000)
        self.assertEqual(a['fast_real_MACs_observed_rows_times_C2'],3000*384*384)
        self.assertEqual(r['observed']['slow_fc1_rows'],12000)
        self.assertEqual(r['observed']['fast_fc_rows'],12000)
        self.assertTrue(bool((output<0).any()))

    def test_zero_and_mean_no_fake_fast_FC(self):
        mask=mask900();expanded=mask.repeat_interleave(9,0).repeat_interleave(9,1)[:30,:40]
        for branch in ('zero','mean'):
            out,a,r,_,_,mean=self.pair(branch,mask)
            self.assertEqual(a['observed']['slow_fc1_rows'],9000)
            self.assertEqual(a['observed']['fast_fc_rows'],0);self.assertEqual(r['observed']['fast_fc_rows'],0)
            self.assertEqual(a['fast_real_MACs_observed_rows_times_C2'],0)
            expected=torch.zeros(384) if branch=='zero' else mean
            self.assertTrue(torch.equal(out[:,:,~expanded,:],expected.expand_as(out[:,:,~expanded,:])))
            self.assertEqual(a['observed']['mean_fill_elements'],0 if branch=='zero' else 3000*384)
            self.assertEqual(r['observed']['mean_fill_elements'],0 if branch=='zero' else 12000*384)

    def test_all_keep_and_all_fast_never_use_whole_shape_shortcut(self):
        for keep in (True,False):
            _,a,r,_,_,_=self.pair('trained',torch.full((4,5),keep,dtype=torch.bool))
            self.assertEqual(a['observed']['slow_fc1_rows'],12000 if keep else 0)
            self.assertEqual(a['observed']['fast_fc_rows'],0 if keep else 12000)
            self.assertEqual(a['observed']['slow_calls'],20 if keep else 0)
            self.assertEqual(a['observed']['fast_calls'],0 if keep else 20)

    def test_inference_guards(self):
        native,fast,mean=make_fixture();policy=lambda name,u:mask900()
        wrapper=adapter.FastSlowTileMLP(native,fast,mean,'actual','trained',policy,tile)
        with self.assertRaisesRegex(ValueError,'Only fixed CPU'):wrapper(self.u)
        with torch.no_grad():
            with self.assertRaisesRegex(ValueError,'Only fixed CPU'):wrapper(self.u[:9])
        for kind in ('training_fast','grad_fast','foreign_fast','invalid_mean','width','PSNT','currentBN'):
            n,f,m=make_fixture()
            if kind=='training_fast':f.train()
            elif kind=='grad_fast':f.requires_grad_(True)
            elif kind=='foreign_fast':f=nn.Identity().eval()
            elif kind=='invalid_mean':m=torch.zeros(383)
            elif kind=='width':n.fc1.out_features=1024
            elif kind=='PSNT':n.sn2.spiking_neuron.T=9
            else:n.bn1.track_running_stats=False;n.bn1.running_mean=None;n.bn1.running_var=None
            with self.assertRaises(ValueError):adapter.FastSlowTileMLP(n,f,m,'actual','trained',policy,tile)
        for bad_mask in (torch.ones(4,5),torch.ones(5,4,dtype=torch.bool)):
            n,f,m=make_fixture();w=adapter.FastSlowTileMLP(n,f,m,'actual','trained',lambda name,u:bad_mask,tile)
            with torch.no_grad():
                with self.assertRaisesRegex(ValueError,'Invalid mask'):w(self.u)

    def test_forward_error_removes_hooks_and_context_restores_original(self):
        native,fast,mean=make_fixture()
        model=nn.Module();parent=model
        components=adapter.LAYER.split('.')
        for name in components[:-1]:
            child=nn.Module();parent.add_module(name,child);parent=child
        parent.add_module(components[-1],native);parent.cnf='ADD';parent.use_checkpoint=False;model.eval()
        class IntentionalFailure(Exception):pass
        def fail(x):raise IntentionalFailure('fixture failure after hooks installed')
        native.forward=fail
        with self.assertRaises(IntentionalFailure):
            with adapter.replace_one(model,fast,mean,'actual','trained',lambda name,u:mask900(),tile) as wrapper:
                self.assertIs(model.get_submodule(adapter.LAYER),wrapper)
                with torch.no_grad():wrapper(self.u)
        self.assertIs(model.get_submodule(adapter.LAYER),native)
        self.assertTrue(all(not m.training for m in model.modules()))
        for layer in (native.fc1,native.fc2,fast.fc1[1]):self.assertEqual(len(layer._forward_pre_hooks),0)


if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    unittest.main(verbosity=2)
