"""CPU tests independent of CUDA, datasets and the installed SNN backend."""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.nn.utils import parametrize

EXP = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("refresh", EXP / "overlay/models/STSwinNet_SNN/refresh_training.py")
refresh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(refresh)


class RefreshTests(unittest.TestCase):
    def test_delay_initialization_and_gradient(self):
        model = nn.Linear(4, 4, bias=False)
        original = model.weight.detach().clone()
        parametrize.register_parametrization(model, "weight", refresh.SharedDelayResidual(original))
        self.assertTrue(torch.equal(model.weight, original))
        model(torch.ones(2, 4)).sum().backward()
        self.assertTrue(torch.all(model.parametrizations.weight[0].coefficients.grad != 0))

    def test_delay_export_after_optimizer_update(self):
        model = nn.Linear(4, 4, bias=False)
        parametrize.register_parametrization(model, "weight", refresh.SharedDelayResidual(model.weight))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randn(3, 4)
        model(x).square().sum().backward()
        opt.step()
        state = refresh.dense_export(model)
        self.assertEqual(set(state), {"weight"})
        dense = nn.Linear(4, 4, bias=False)
        dense.load_state_dict(state, strict=True)
        self.assertTrue(torch.equal(model(x), dense(x)))

    def test_drop_shape_values_polarity_and_nonmutation(self):
        x = torch.ones(4, 10, 2, 16, 16)
        y, selected = refresh.corrupt_voxel_view(x, torch.Generator().manual_seed(4), .2, 1)
        self.assertTrue(torch.all(x == 1))
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.equal(y[:, :, 0], y[:, :, 1]))
        self.assertTrue(selected.all())
        self.assertTrue(torch.all((y == 0) | (y == x)))

    def test_zero_drop_is_identity(self):
        x = torch.randn(2, 10, 2, 3, 3)
        y, _ = refresh.corrupt_voxel_view(x, torch.Generator(), 0, 1)
        self.assertTrue(torch.equal(x, y))

    def test_isolated_rng(self):
        x = torch.ones(2, 10, 2, 4, 4)
        before = torch.get_rng_state()
        a, _ = refresh.corrupt_voxel_view(x, torch.Generator().manual_seed(8), .2, .5)
        b, _ = refresh.corrupt_voxel_view(x, torch.Generator().manual_seed(8), .2, .5)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.equal(a, b))

    def test_loss_units_and_detached_teacher(self):
        student = torch.ones(1, 2, 2, 2, requires_grad=True)
        teacher = torch.zeros_like(student, requires_grad=True)
        gt = torch.zeros_like(student)
        loss, count, _ = refresh.confidence_distillation(student, teacher, gt,
            torch.ones(1, 1, 2, 2), torch.ones(1, dtype=torch.bool), 2, 1)
        self.assertEqual(count, 4)
        self.assertAlmostEqual(loss.item(), (8 + 1e-6)**.5 - .001, places=5)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(float(student.grad.abs().sum()), 0)

    def test_confidence_and_corrupted_sample_mask(self):
        s = torch.ones(2, 2, 2, 2, requires_grad=True)
        t = torch.zeros_like(s)
        t[1] = 100
        loss, count, _ = refresh.confidence_distillation(s, t, torch.zeros_like(s),
            torch.ones(2, 1, 2, 2), torch.tensor([False, True]), 1, 1)
        self.assertEqual(count, 0)
        self.assertEqual(float(loss), 0)
        loss.backward()
        self.assertEqual(float(s.grad.abs().sum()), 0)

    def test_invalid_gt_is_masked_without_nan(self):
        s = torch.ones(1, 2, 2, 2, requires_grad=True)
        gt = torch.full_like(s, float("nan"))
        loss, count, _ = refresh.confidence_distillation(s, torch.zeros_like(s), gt,
            torch.ones(1, 1, 2, 2), torch.tensor([True]), 1, 1)
        self.assertEqual(count, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_patch_compiles_and_old_entrypoint_unchanged(self):
        sys.path.insert(0, str(EXP / "entrypoints"))
        import train_algorithm_refresh as entry
        source = EXP.parents[1] / "third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py"
        patched = entry.patch_source(source.read_text(), source)
        compile(patched, str(source), "exec")
        self.assertIn('dense_export(model)', patched)
        self.assertNotIn('refresh', entry.h9._patch_source.__name__)


if __name__ == "__main__":
    unittest.main()
