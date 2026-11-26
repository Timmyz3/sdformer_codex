#!/usr/bin/env python3
"""CPU/synthetic native MLP contract checks; no checkpoint, full model or GPU."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
sys.dont_write_bytecode = True
import torch
import structured_merge_mlp as merge

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FACTORY = ROOT / "research/flow_story_20261011/preflight/cpu_numerics/native_mlp_probe.py"
spec = importlib.util.spec_from_file_location("structured_merge_native_fixture", FACTORY)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
MEASUREMENTS = []


def native(c=4, dtype=torch.float32):
    return fixture.make_mlp(c, dtype).requires_grad_(False)


def hand_reference(mlp, u, mask, stride):
    """Independent per-cell reference, deliberately using one native call/cell."""
    result = torch.empty_like(u)
    for gy in range(mask.shape[0]):
        for gx in range(mask.shape[1]):
            sy, ey = 9*gy, min(9*(gy+1), u.shape[2])
            sx, ex = 9*gx, min(9*(gx+1), u.shape[3])
            if mask[gy, gx]:
                result[:, :, sy:ey, sx:ex] = mlp(u[:, :, sy:ey, sx:ex].contiguous())
                continue
            for y in range(sy, ey, stride):
                for x in range(sx, ex, stride):
                    y1, x1 = min(y+stride, ey), min(x+stride, ex)
                    rep = u[:, :, y:y1, x:x1].mean((2, 3), keepdim=True).contiguous()
                    result[:, :, y:y1, x:x1] = mlp(rep)
    return result


def threshold_native():
    mlp = native(c=1, dtype=torch.float64)
    with torch.no_grad():
        for sn in (mlp.sn1.spiking_neuron, mlp.sn2.spiking_neuron):
            sn.weight.copy_(torch.eye(10, dtype=torch.float64))
            sn.bias.fill_(-0.5)
        mlp.fc1.weight.zero_(); mlp.fc1.weight[0, 0] = 1
        mlp.fc2.weight.zero_(); mlp.fc2.weight[0, 0] = 1
        for bn in (mlp.bn1.norm_layer, mlp.bn2.norm_layer):
            bn.running_mean.zero_(); bn.running_var.fill_(1-bn.eps)
            bn.weight.fill_(1); bn.bias.zero_()
    return mlp


class Tests(unittest.TestCase):
    def test_actual_dual_all_fine_all_coarse_mixed_and_tails(self):
        masks = {"all_fine": torch.ones(2, 2, dtype=torch.bool),
                 "all_coarse": torch.zeros(2, 2, dtype=torch.bool),
                 "mixed": torch.tensor([[True, False], [False, True]]),
                 "coarse_tail_only": torch.tensor([[True, True], [True, False]])}
        for dtype in (torch.float32, torch.float64):
            mlp = native(dtype=dtype)
            state = copy.deepcopy(mlp.state_dict())
            u = torch.randn(10, 1, 11, 28, 4, dtype=dtype)[:, :, :, ::2]
            saved = u.clone()
            self.assertFalse(u.is_contiguous())
            for stride in (3, 9):
                for name, mask in masks.items():
                    with torch.inference_mode():
                        actual, ar = merge.tile_core(mlp, u, mask, stride, "actual", True)
                        dual, dr = merge.tile_core(mlp, u, mask, stride, "dual_reference", True)
                        independent = hand_reference(mlp, u, mask, stride)
                        if name == "all_fine":
                            torch.testing.assert_close(actual, mlp(u), atol=2e-6, rtol=2e-6)
                    self.assertTrue(torch.equal(actual, dual))
                    self.assertEqual(actual.stride(), dual.stride())
                    tolerance = 2e-6 if dtype == torch.float32 else 1e-11
                    torch.testing.assert_close(actual, independent, atol=tolerance, rtol=tolerance)
                    self.assertEqual(ar["executed_fc_rows_per_layer"], ar["selected_service"]["selected_fc_rows_per_layer"])
                    self.assertEqual(dr["observed"]["fine"]["native_mlp_calls"], 4)
                    self.assertEqual(dr["observed"]["coarse"]["native_mlp_calls"], 4)
                    self.assertTrue(actual.abs().max() > 0)
                    MEASUREMENTS.append({"dtype": str(dtype), "mask": name, "actual": ar, "dual_reference": dr,
                                         "actual_dual_max_abs": float((actual-dual).abs().max()),
                                         "independent_max_abs": float((actual-independent).abs().max())})
            self.assertTrue(torch.equal(saved, u))
            self.assertTrue(all(torch.equal(value, mlp.state_dict()[key]) for key, value in state.items()))
            self.assertTrue(all(not m._forward_pre_hooks for m in mlp.modules()))

    def test_spatial_constant_per_cell_coarse_equals_fine(self):
        for stride in (3, 9):
            mlp = native(dtype=torch.float64)
            u = torch.empty(10, 1, 11, 14, 4, dtype=torch.float64)
            for gy in range(2):
                for gx in range(2):
                    for y in range(9*gy, min(9*(gy+1), 11), stride):
                        for x in range(9*gx, min(9*(gx+1), 14), stride):
                            constant = torch.randint(-8, 9, (10, 1, 1, 1, 4), dtype=torch.int64).double()/4
                            u[:, :, y:min(y+stride, 9*(gy+1), 11), x:min(x+stride, 9*(gx+1), 14)] = constant
            with torch.inference_mode():
                fine, _ = merge.tile_core(mlp, u, torch.ones(2, 2, dtype=torch.bool), stride)
                coarse, _ = merge.tile_core(mlp, u, torch.zeros(2, 2, dtype=torch.bool), stride)
            torch.testing.assert_close(fine, coarse, atol=1e-12, rtol=1e-12)

    def test_partial_cells_use_true_area_and_do_not_mix_time(self):
        local = (torch.arange(10, dtype=torch.float64)[:, None, None, None, None]*1000 +
                 torch.arange(2, dtype=torch.float64)[None, None, :, None, None]*100 +
                 torch.arange(5, dtype=torch.float64)[None, None, None, :, None])
        reps, spans = merge.spatial_mean(local, 3)
        self.assertEqual(reps.shape, (10, 1, 1, 2, 1))
        self.assertEqual(float(reps[0, 0, 0, 0, 0]), 51)
        self.assertEqual(float(reps[0, 0, 0, 1, 0]), 53.5)
        self.assertEqual(float(reps[9, 0, 0, 1, 0]), 9053.5)
        whole, _ = merge.spatial_mean(local, 9)
        self.assertEqual(float(whole[0, 0, 0, 0, 0]), 52)
        self.assertNotEqual(float(whole[0, 0, 0, 0, 0]), float(reps[0].mean()))
        restored = merge.repeat_branch(reps, spans, 2, 5)
        self.assertTrue(torch.equal(restored[:, :, :, :3], reps[:, :, :, :1].expand(-1, -1, 2, 3, -1)))

    def test_threshold_counterexample_and_temporal_dependency(self):
        mlp = threshold_native()
        u = torch.zeros(10, 1, 3, 3, 1, dtype=torch.float64)
        u[:, :, 0, 0] = 1
        mask = torch.zeros(1, 1, dtype=torch.bool)
        with torch.inference_mode():
            fine = mlp(u)
            coarse, _ = merge.tile_core(mlp, u, mask)
        self.assertEqual(float(fine.sum()), 10)
        self.assertEqual(float(coarse.sum()), 0)
        self.assertFalse(torch.equal(fine.mean((2, 3), keepdim=True), coarse[:, :, :1, :1]))
        with torch.no_grad():
            mlp.sn1.spiking_neuron.weight.zero_()
            mlp.sn1.spiking_neuron.weight[0, 9] = 1
        u.zero_(); u[9] = 1
        with torch.inference_mode():
            output, record = merge.tile_core(mlp, u, mask)
        self.assertTrue(torch.equal(output[0], torch.ones_like(output[0])))
        self.assertEqual(float(output[1:].sum()), 0)
        self.assertEqual(record["observed"]["native_input_shapes"]["coarse"], [[10, 1, 1, 1, 1]])

    def test_manual_actual_service_and_overheads(self):
        mlp = native()
        u = torch.randn(10, 1, 11, 14, 4)
        mask = torch.tensor([[True, False], [False, True]])
        for stride, reps, rows, dualrows in ((3, 9, 1000, 1740), (9, 2, 930, 1580)):
            with torch.inference_mode():
                _, rec = merge.tile_core(mlp, u, mask, stride)
                _, dual = merge.tile_core(mlp, u, mask, stride, "dual_reference")
            cost = rec["selected_service"]
            self.assertEqual(cost["fine_spatial_points"], 91)
            self.assertEqual(cost["coarse_source_spatial_points"], 63)
            self.assertEqual(cost["coarse_representative_spatial_points"], reps)
            self.assertEqual(rec["executed_fc_rows_per_layer"], rows)
            self.assertEqual(dual["executed_fc_rows_per_layer"], dualrows)
            self.assertEqual(cost["cancelled_fc_rows_per_layer"], 1540-rows)
            self.assertEqual(cost["selected_added_work"]["pool_input_elements"], 2520)
            self.assertEqual(cost["selected_added_work"]["pool_reduction_additions_logical"], 40*(63-reps))
            self.assertEqual(cost["selected_added_work"]["repeat_dense_write_elements"], 2520)
            self.assertEqual(cost["cancelled_psn_temporal_macs_nominal"], (63-reps)*100*20)
            self.assertEqual(dual["executed_pool_input_elements"], 6160)
            self.assertEqual(dual["executed_repeat_write_elements"], 6160)
            self.assertEqual(dual["executed_pool_divisions_logical"], 800 if stride == 3 else 160)

    def test_autocast_and_pool_overflow_are_rejected(self):
        mlp = native()
        u = torch.ones(10, 1, 9, 9, 4)
        mask = torch.zeros(1, 1, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "Autocast"), torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            merge.tile_core(mlp, u, mask)
        mlp = threshold_native().float()
        huge = torch.full((10, 1, 9, 9, 1), 1e37)
        with self.assertRaisesRegex(ValueError, "pooled representative"), torch.inference_mode():
            merge.tile_core(mlp, huge, mask, 9)
        self.assertTrue(all(not m._forward_pre_hooks for m in mlp.modules()))

    def test_residual_once_mask_snapshot_and_context_restoration(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.cnf = "ADD"; self.use_checkpoint = False; self.mlp = native()
            def forward(self, x):
                return x+self.mlp(x)
        model = torch.nn.Module()
        model.block = Block()
        model.eval().requires_grad_(False)
        original = model.block.mlp
        u = torch.randn(10, 1, 11, 14, 4)
        mask = torch.tensor([[True, False], [False, True]])
        state = copy.deepcopy(original.state_dict())
        with torch.inference_mode():
            expected = u+hand_reference(original, u, mask, 3)
            with merge.replace_mlp(model, "block.mlp", mask, expected_shape=u.shape, expected_u_sha=merge.tensor_sha(u)) as wrapper:
                mask.logical_not_()
                actual = model.block(u)
                self.assertFalse(wrapper.records[0]["external_residual_added_here"])
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        self.assertIs(model.block.mlp, original)
        self.assertTrue(all(torch.equal(value, original.state_dict()[key]) for key, value in state.items()))
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with merge.replace_mlp(model, "block.mlp", mask):
                raise RuntimeError("injected")
        self.assertIs(model.block.mlp, original)
        model.block.cnf = "AND"
        with self.assertRaisesRegex(ValueError, "ADD"):
            with merge.replace_mlp(model, "block.mlp", mask):
                pass

    def test_invalid_contracts_and_hook_cleanup(self):
        u = torch.randn(10, 1, 2, 5, 4)
        mask = torch.ones(1, 1, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "Inference only"):
            merge.tile_core(native(), u, mask)
        for corrupt in ("training", "current_bn", "missing_stats", "bad_variance", "time", "weights", "norm", "bn_step", "fc_step"):
            mlp = native()
            if corrupt == "training": mlp.train()
            if corrupt == "current_bn": mlp.bn1.norm_layer.track_running_stats = False
            if corrupt == "missing_stats": mlp.bn2.norm_layer.running_mean = None
            if corrupt == "bad_variance": mlp.bn2.norm_layer.running_var[0] = -1
            if corrupt == "time": mlp.sn1.spiking_neuron.T = 9
            if corrupt == "weights": mlp.fc1.weight.requires_grad_(True)
            if corrupt == "norm": mlp.norm_layer = "LN"
            if corrupt == "bn_step": mlp.bn1.norm_layer.step_mode = "s"
            if corrupt == "fc_step": mlp.fc1.step_mode = "s"
            with self.assertRaises(ValueError), torch.inference_mode():
                merge.tile_core(mlp, u, mask)
        mlp = native()
        bad_inputs = (u[:9], u.expand(-1, 2, -1, -1, -1), u[:, :, :, :, :3], u.clone().fill_(float("nan")))
        for bad in bad_inputs:
            with self.assertRaises(ValueError), torch.inference_mode():
                merge.tile_core(mlp, bad, mask)
        for bad_mask in (torch.ones(1, 1), torch.ones(2, 2, dtype=torch.bool)):
            with self.assertRaises(ValueError), torch.inference_mode():
                merge.tile_core(mlp, u, bad_mask)
        def injected(module, inputs):
            raise RuntimeError("injected native failure")
        own = mlp.fc2.register_forward_pre_hook(injected)
        try:
            with self.assertRaisesRegex(RuntimeError, "injected"), torch.inference_mode():
                merge.tile_core(mlp, u, mask)
        finally:
            own.remove()
        self.assertTrue(all(not m._forward_pre_hooks for m in mlp.modules()))


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    source = ROOT / "SDformer/third_party/SDformerFlow/models/STSwinNet_SNN"
    inputs = [Path(__file__), Path(merge.__file__), FACTORY] + [source / p for p in
              ("Spiking_swin_transformer3D.py", "Spiking_submodules.py", "Spiking_modules.py")]
    record = {
        "passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "failures": [str(value) for value in result.failures], "errors": [str(value) for value in result.errors],
        "torch": torch.__version__, "device": "cpu", "GPU_initialized": torch.cuda.is_initialized(),
        "checkpoint_loaded": False, "full_model_run": False,
        "scope": "Unmodified native MLP with synthetic weights/inputs only; contract checks, no AEE/cycle/RTL evidence",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "measurements": MEASUREMENTS,
    }
    (HERE / "cpu_results.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    return 0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1


if __name__ == "__main__":
    raise SystemExit(main())
