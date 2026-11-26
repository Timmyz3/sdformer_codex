#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import time
import unittest
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO/'research/flow_story_20261011/preflight/cpu_numerics/deps'),
               str(REPO/'SDformer/third_party/SDformerFlow')]
import torch
from torch.nn import functional as F
import group_precision_core as c
import group_contract as gc
RECORDS = []


def mask(h=12, w=19):
    m = torch.zeros(h, w, dtype=torch.bool)
    m[:9, :9] = True
    return m


def load_old():
    path = REPO/'research/flow_story_20261019/round27/implementation/recovery_integer_core.py'
    spec = importlib.util.spec_from_file_location('_e2_original_q6', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, hashlib.sha256(path.read_bytes()).hexdigest()


class Tests(unittest.TestCase):
    def test_01_domains_and_invalid_actions(self):
        tail, q6, q4, rec = c.row_domains(10, 2, 12, 19, 'cpu', mask())
        self.assertEqual((int(tail.sum()), int(q6.sum()), int(q4.sum())),
                         ((12*19-2*81)*20, 81*20, 81*20))
        self.assertEqual(rec['selected_complete_tiles'], [[0, 0]])
        bad = mask(); bad[0, 0] = False
        for candidate in (bad, torch.ones(12, 19, dtype=torch.bool), mask().float()):
            with self.assertRaises(ValueError):
                c.row_domains(10, 1, 12, 19, 'cpu', candidate)
        with self.assertRaises(ValueError):
            c.row_domains(9, 1, 12, 19, 'cpu', mask())

    def test_02_independent_int64_actual_plane_cancellation(self):
        torch.manual_seed(20261021)
        x = torch.randint(0, 2, (10*12*19, 8)).float()
        q = torch.tensor([[-128, -127, -1, 0, 1, 3, 15, 127]], dtype=torch.int8).repeat(16, 1)
        scale = torch.linspace(.01, .16, 16)
        for chosen in (torch.zeros_like(mask()), mask(), mask() | torch.roll(mask(), 9, 1)):
            # A rolled mask can touch the true right tail; create only two complete tiles.
            chosen[:, 18:] = False
            tail, q6, q4, rec = c.row_domains(10, 1, 12, 19, 'cpu', chosen)
            with torch.no_grad():
                y, actual, audit = c.integer_affine(x, q, scale, tail, q6, q4)
                yd, dense, _ = c.integer_affine(x, q, scale, tail, q6, q4, reference=True)
            expected = x.to(torch.int64) @ q.to(torch.int64).T
            expected[q6] = x[q6].to(torch.int64) @ (((q.to(torch.int64) >> 2) << 2).T)
            expected[q4] = x[q4].to(torch.int64) @ (((q.to(torch.int64) >> 4) << 4).T)
            self.assertTrue(torch.equal(actual.to(torch.int64), expected))
            self.assertTrue(torch.equal(actual, dense))
            self.assertTrue(torch.equal(y.view(torch.uint8), yd.view(torch.uint8)))
            expected_rows = [int(tail.sum())]*2 + [int((~q4).sum())]*2 + [len(x)]*4
            self.assertEqual([sum(z['rows'] for z in audit['kernel_calls'] if z['bit'] == bit)
                              for bit in range(8)], expected_rows)
            self.assertEqual(audit['logical_row_plane_work'],
                             8*int(tail.sum())+6*int(q6.sum())+4*int(q4.sum()))
            self.assertTrue(all(z['coefficient'] == -128 for z in audit['kernel_calls'] if z['bit'] == 7))
        with self.assertRaises(ValueError):
            c.integer_affine(x, q, scale, tail, q6, q4)

    def test_03_original_Q6_background_byte_equivalence(self):
        old, source_sha = load_old()
        torch.manual_seed(40)
        x = torch.randint(0, 2, (10*12*19, 8)).float()
        w = torch.randn(16, 8)
        q, scale = c.quantize(w)
        tail, q6, q4, _ = c.row_domains(10, 1, 12, 19, 'cpu', torch.zeros_like(mask()))
        with torch.no_grad():
            y, actual, _ = c.integer_affine(x, q, scale, tail, q6, q4)
            yo, _, io, _ = old.integer_affine(x, q, scale, tail, 'Q6')
        self.assertTrue(torch.equal(actual, io))
        self.assertTrue(torch.equal(y.view(torch.uint8), yo.view(torch.uint8)))
        RECORDS.append(dict(original_Q6_source_sha256=source_sha, background_byte_equivalence=True))

    def test_04_linear_conv_and_halo_integer_values(self):
        torch.manual_seed(41)
        for kind in ('linear', 'conv'):
            source = torch.nn.Linear(8, 16, bias=True) if kind == 'linear' else torch.nn.Conv2d(8, 16, 3, padding=1, bias=True)
            shape = (10, 1, 12, 19, 8) if kind == 'linear' else (10, 1, 8, 12, 19)
            x = torch.randint(0, 2, shape).float()
            op = c.GroupAffine(source, kind, mask()); op.capture_integer = True
            with torch.no_grad():
                y = op(x); actual = op.last_integer.clone()
                op.reference = True; yr = op(x)
            self.assertTrue(torch.equal(y.contiguous().view(torch.uint8), yr.contiguous().view(torch.uint8)))
            q, scale = c.quantize(source.weight)
            rows = x.reshape(-1, 8) if kind == 'linear' else F.unfold(x.reshape(10, 8, 12, 19), 3, padding=1).transpose(1, 2).reshape(-1, 72)
            tail, q6, q4, _ = c.row_domains(10, 1, 12, 19, 'cpu', mask())
            expected = rows.to(torch.int64) @ q.to(torch.int64).T
            expected[q6] = rows[q6].to(torch.int64) @ (((q.to(torch.int64) >> 2) << 2).T)
            expected[q4] = rows[q4].to(torch.int64) @ (((q.to(torch.int64) >> 4) << 4).T)
            self.assertTrue(torch.equal(actual.to(torch.int64), expected))
            with self.assertRaises(ValueError):
                op(x)

    def test_05_real_inventory_equal_disjoint_macro_groups(self):
        path = REPO/'research/flow_story_20261019/round27/hardware/UNIFIED_SERVICE_V1.json'
        geom = gc.geometry_from_inventory(path)
        for family in (list(c.CONVS), [n for n in c.LINEARS if n.endswith('fc1')]):
            contract = gc.build(geom, family, source_state_sha256='CPU_TEST_ONLY_NOT_AN_EXECUTION_CONTRACT')
            self.assertEqual(len(gc.actions()), 22)
            self.assertEqual(len(set(g['cancelled_kernel_ticks'] for g in contract['groups'])), 1)
            single = contract['single_cancelled_kernel_ticks']
            for action in ((), (0,), (1, 5)):
                masks, rec = gc.materialize(contract, action)
                self.assertEqual(set(masks), set(c.TARGETS))
                actual_ticks = sum(int(m.sum())//81*gc.tile_ticks(geom[n]) for n, m in masks.items())
                self.assertEqual(actual_ticks, len(action)*single)
            bad = copy.deepcopy(contract)
            name = next(iter(bad['groups'][0]['tiles']))
            bad['groups'][1]['tiles'][name][0] = bad['groups'][0]['tiles'][name][0]
            with self.assertRaises(ValueError):
                gc.validate(bad)
        late = next(n for n in c.LINEARS if '.layers.3.' in n)
        with self.assertRaises(ValueError):
            gc.build(geom, [late], source_state_sha256='CPU_TEST_ONLY')
        RECORDS.append(dict(metadata_families_for_CPU_tests_only=True,
                            family_selected_for_experiment=False, GT_read=False))

    def test_06_real_native_MLP_resblock_currentBN(self):
        from spikingjelly.activation_based import functional
        from models.STSwinNet_SNN.Spiking_modules import MS_ResBlock
        from models.STSwinNet_SNN.Spiking_swin_transformer3D import MS_Spiking_Mlp
        torch.manual_seed(42)
        kw = dict(num_steps=10, spike_norm='BN', neuron_type='psn', v_th=.1, v_reset=None,
                  surrogate_fun='surrogate.ATan()', tau=2., detach_reset=True)
        mlp = MS_Spiking_Mlp(8, hidden_features=16, norm_layer='BN', drop=0., **kw)
        block = MS_ResBlock(8, 8, **kw)
        for mod in (mlp, block):
            functional.set_step_mode(mod, 'm'); mod.eval()
            for bn in mod.modules():
                if isinstance(bn, torch.nn.modules.batchnorm._BatchNorm):
                    bn.track_running_stats = False
                    bn.running_mean = bn.running_var = bn.num_batches_tracked = None
        entries = [(mlp, 'fc1', 'linear'), (mlp, 'fc2', 'linear'),
                   (block.conv1, '0', 'conv'), (block.conv2, '0', 'conv')]
        saved = []
        for parent, name, kind in entries:
            old = getattr(parent, name); saved.append((parent, name, old))
            setattr(parent, name, c.GroupAffine(old, kind, mask()))
        try:
            u, v = torch.randn(10, 1, 12, 19, 8), torch.randn(10, 1, 8, 12, 19)
            outputs = []
            for reference in (False, True):
                for parent, name, _ in saved:
                    getattr(parent, name).reference = reference
                functional.reset_net(mlp); functional.reset_net(block)
                with torch.no_grad():
                    outputs.append((mlp(u), block(v)))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(*outputs)))
            self.assertTrue(all(bn.running_mean is None for mod in (mlp, block) for bn in mod.modules()
                                if isinstance(bn, torch.nn.modules.batchnorm._BatchNorm)))
        finally:
            for parent, name, old in saved:
                setattr(parent, name, old)
        RECORDS.append(dict(real_original_micro_forwards=4, model_forwards=0, GT_read=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    a = parser.parse_args()
    assert not a.output.exists()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    start = time.monotonic()
    r = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    obj = dict(schema='flow_e2_group_precision_cpu_v1', status='PASS' if r.wasSuccessful() else 'FAIL',
               tests=r.testsRun, failures=len(r.failures), errors=len(r.errors), records=RECORDS,
               sources={n: gc.sha(HERE/n) for n in ('group_precision_core.py', 'group_contract.py', 'test_cpu.py')},
               GPU_used=False, GT_read=False, model_forwards=0, family_selected=False,
               wall_seconds=time.monotonic()-start)
    a.output.write_text(json.dumps(obj, indent=2)+'\n')
    print(json.dumps(obj))
    assert not torch.cuda.is_initialized()
    raise SystemExit(0 if r.wasSuccessful() else 1)
