"""CPU synthetic API checks, not native checkpoint accuracy/execution evidence."""
import importlib.util
import inspect
from pathlib import Path
import unittest

import torch
from torch import nn

import runtime_group_adapter as adapter
from coarse_head import RuntimeCoarseFlowHead

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('runtime_adapter_test_tile', HERE.parent/'breadth/same_shape_tile_mlp.py')
tile = importlib.util.module_from_spec(spec); spec.loader.exec_module(tile)
FLOW = {'direction': 'forward', 'time_interval': 'synthetic fixed interval', 'units': 'original_image_pixels'}


def head():
    module = RuntimeCoarseFlowHead(384, flow_contract=FLOW)
    # Deterministic nonzero direction enables real pairing in every policy.
    with torch.no_grad():
        for p in module.parameters(): p.zero_()
        module.fc2.bias.copy_(torch.tensor([4., 3.]))
    module.eval(); module.requires_grad_(False)
    return module


class PSN(nn.Module):
    """Lightweight complete-T synthetic coupling, NOT native spike arithmetic."""
    def __init__(self):
        super().__init__(); self.T = 10

    def forward(self, x):
        return x + x.mean(dim=0, keepdim=True)*.125


class SpikeWrapper(nn.Module):
    def __init__(self):
        super().__init__(); self.spiking_neuron = PSN()

    def forward(self, x):
        return self.spiking_neuron(x)


class MS_Spiking_Mlp(nn.Module):
    """Only type/API/shape-compatible; hidden2 keeps CPU checks lightweight."""
    def __init__(self):
        super().__init__(); self.norm_layer = 'BN'
        self.sn1, self.sn2 = SpikeWrapper(), SpikeWrapper()
        self.fc1, self.fc2 = nn.Linear(384, 2), nn.Linear(2, 384)
        self.bn1, self.bn2 = nn.BatchNorm1d(2), nn.BatchNorm1d(384)

    def forward(self, x):
        y = self.fc1(self.sn1(x))
        y = self.bn1(y.reshape(-1, 2)).reshape(y.shape)
        y = self.fc2(self.sn2(y))
        return self.bn2(y.reshape(-1, 384)).reshape(y.shape)


def model_fixture():
    model = nn.Module(); parent = model
    for part in adapter.LAYER.split('.')[:-1]:
        child = nn.Module(); parent.add_module(part, child); parent = child
    parent.cnf = 'ADD'; parent.use_checkpoint = False
    parent.add_module('mlp', MS_Spiking_Mlp())
    model.add_module('untouched_mlp', MS_Spiking_Mlp())
    model.add_module('external_current_bn', nn.BatchNorm1d(4, track_running_stats=False))
    model.eval()
    return model, parent


class RuntimeAdapterChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        # Distinct time lanes exercise complete-T coupling in the synthetic MLP.
        cls.u = (torch.arange(10).reshape(10, 1, 1, 1, 1).expand(adapter.SHAPE).float()/10).contiguous()

    def test_all_four_online_policies_share_real_cost_and_whole_time_mask(self):
        records = []
        self.assertEqual(list(inspect.signature(adapter.RuntimeGroupPolicy.__call__).parameters), ['self', 'name', 'u'])
        with torch.inference_mode():
            for name in adapter.POLICIES:
                policy = adapter.RuntimeGroupPolicy(head(), name)
                mask = policy(adapter.LAYER, self.u)
                self.assertEqual(tuple(mask.shape), (4, 5)); self.assertEqual(mask.dtype, torch.bool)
                rec = policy.records[0]; records.append(rec)
                true_cost = sum(rec['points_per_tile'][r][c] for r in range(4) for c in range(5) if mask[r, c])
                self.assertEqual(true_cost, rec['selection']['slow_service_cost'])
                self.assertEqual(true_cost, rec['common_budget']['target_cost'])
                self.assertLessEqual(true_cost, 900); self.assertGreater(true_cost, 0)
                self.assertEqual(sum(map(sum, rec['points_per_tile'])), 1200)
                self.assertEqual(rec['points_per_tile'][-1], [27, 27, 27, 27, 12])
                self.assertFalse(rec['GT_or_final_flow_used'])
                self.assertTrue(rec['all_four_proposals_computed_for_cost_matching'])
                self.assertEqual(set(rec['common_budget']['per_policy_reachability']), set(adapter.POLICIES))
                self.assertTrue(rec['selection']['counters']['state_visits'] > 0)
        self.assertEqual(len({r['selection']['slow_service_cost'] for r in records}), 1)
        self.assertEqual(len({r['input_sha256'] for r in records}), 1)
        self.assertTrue(all(r['flow_dxdy_pixels'] == records[0]['flow_dxdy_pixels'] for r in records))
        self.assertTrue(all(r['scores'] == records[0]['scores'] for r in records))

    def test_invalid_phase_shape_layer_cap_or_unfrozen_head_rejected(self):
        h = head(); h.train()
        with self.assertRaisesRegex(ValueError, 'Freeze head'): adapter.RuntimeGroupPolicy(h, 'motion')
        h = head(); h.fc1.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'Freeze head'): adapter.RuntimeGroupPolicy(h, 'motion')
        with self.assertRaises(ValueError): adapter.RuntimeGroupPolicy(head(), 'motion', 899)
        p = adapter.RuntimeGroupPolicy(head(), 'motion')
        with torch.enable_grad():
            with self.assertRaises(ValueError): p(adapter.LAYER, self.u)
        with torch.inference_mode():
            with self.assertRaises(ValueError): p('other.mlp', self.u)
            with self.assertRaises(ValueError): p(adapter.LAYER, self.u[:9])

    def test_single_replacement_exception_restores_objects_and_external_BN(self):
        model, parent = model_fixture()
        original, other, outside = parent.mlp, model.untouched_mlp, model.external_current_bn
        with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
            with adapter.replace_one_mlp(model, tile, 'tiled_kept_only', lambda name, u: torch.ones(4, 5, dtype=torch.bool)) as wrapper:
                self.assertIs(parent.mlp, wrapper); self.assertIs(wrapper.native, original)
                self.assertEqual(sum(isinstance(m, tile.SameShapeTileMLP) for m in model.modules()), 1)
                self.assertIs(model.untouched_mlp, other)
                self.assertIs(model.external_current_bn, outside)
                self.assertFalse(outside.track_running_stats); self.assertIsNone(outside.running_mean)
                raise RuntimeError('synthetic failure')
        self.assertIs(parent.mlp, original)
        self.assertEqual(sum(isinstance(m, tile.SameShapeTileMLP) for m in model.modules()), 0)
        self.assertFalse(outside.track_running_stats); self.assertIsNone(outside.running_var)

    def test_target_current_BN_training_nonADD_checkpoint_guards(self):
        for kind in ('target_current', 'training', 'not_add', 'checkpoint'):
            model, parent = model_fixture()
            if kind == 'target_current':
                parent.mlp.bn1.track_running_stats = False
                parent.mlp.bn1.running_mean = parent.mlp.bn1.running_var = None
            elif kind == 'training': model.external_current_bn.train()
            elif kind == 'not_add': parent.cnf = 'AND'
            else: parent.use_checkpoint = True
            original = parent.mlp
            with self.assertRaises(ValueError):
                with adapter.replace_one_mlp(model, tile, 'tiled_kept_only', lambda name, u: torch.ones(4, 5, dtype=torch.bool)):
                    self.fail('Guard did not reject')
            self.assertIs(parent.mlp, original)

    def test_actual_and_replay_same_shape_match_and_head_rows_are_separate(self):
        model, parent = model_fixture(); h = head()
        policy = adapter.RuntimeGroupPolicy(h, 'motion')
        head_rows = {'fc1': 0, 'fc2': 0}; handles = []
        for name in head_rows:
            def count(m, inputs, key=name): head_rows[key] += inputs[0].numel()//inputs[0].shape[-1]
            handles.append(getattr(h, name).register_forward_pre_hook(count))
        try:
            with torch.inference_mode():
                with adapter.replace_one_mlp(model, tile, 'tiled_kept_only', policy) as wrapper:
                    actual = wrapper(self.u); actual_record = wrapper.records[0]
                replay = tile.ReplayPolicy([actual_record])
                with adapter.replace_one_mlp(model, tile, 'tiled_all_then_mask', replay) as wrapper:
                    reference = wrapper(self.u); reference_record = wrapper.records[0]
                replay.complete()
        finally:
            for handle in handles: handle.remove()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        self.assertEqual(actual_record['input_sha256'], reference_record['input_sha256'])
        self.assertEqual(actual_record['output_sha256'], reference_record['output_sha256'])
        self.assertEqual(actual_record['output_stride'], reference_record['output_stride'])
        self.assertEqual(head_rows, {'fc1': 20, 'fc2': 20})
        selected = policy.records[0]['selection']
        for name in ('fc1_rows', 'fc2_rows'):
            self.assertEqual(actual_record['observed'][name], 10*selected['slow_service_cost'])
            self.assertEqual(reference_record['observed'][name], 12000)
        self.assertEqual(actual_record['observed']['native_mlp_calls'], selected['selected_tile_count'])
        self.assertEqual(reference_record['observed']['native_mlp_calls'], 20)
        keep = torch.tensor(actual_record['keep'])
        dense_keep = keep.repeat_interleave(9, 0).repeat_interleave(9, 1)[:30, :40]
        self.assertTrue((actual[:, 0, ~dense_keep, :] == 0).all())
        self.assertFalse(actual_record['all_keep_shortcut'])
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main(verbosity=2)
