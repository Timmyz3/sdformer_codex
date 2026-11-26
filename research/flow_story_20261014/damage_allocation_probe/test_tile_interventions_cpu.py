#!/usr/bin/env python3
"""Production intervention path on twelve small native MLPs; no GPU/full model."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch
sys.dont_write_bytecode = True
import torch
import run_tile_interventions as runner
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(runner.BASE/"scope_extension_v1"))
spec = importlib.util.spec_from_file_location("_damage_scope_cpu_fixture", runner.BASE/"scope_extension_v1/test_scope_extension_cpu.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
parent = runner.load_scope_runner()
merge = parent.load_frozen_merge()
MEASUREMENTS = []


def build():
    model, targets, _, _, functional = fixture.build()
    shapes = {name: (10, 1, 20, 21, 4) for name in targets}
    x = torch.randn(10, 1, 20, 21, 4)
    return model, targets, shapes, x, functional


class Tests(unittest.TestCase):
    def test_fixed_12_candidates_146_maximum_and_complete_masks(self):
        self.assertEqual(runner.full_tiles(parent.STAGE_SHAPES[2]), runner.CANDIDATES)
        self.assertEqual(len(runner.ARMS), 13)
        self.assertEqual(8*len(runner.ARMS)+2+8*len(runner.COMBINATION_ARMS), 146)
        for tile in runner.CANDIDATES:
            mask = runner.tile_mask(parent.STAGE_SHAPES[2], (tile,), torch)
            self.assertEqual(tuple(mask.shape), (4, 5))
            self.assertEqual(int(mask.sum()), 1)
            self.assertEqual(int(mask[-1].sum()+mask[:, -1].sum()), 0)
        with self.assertRaisesRegex(ValueError, "complete"):
            runner.tile_mask(parent.STAGE_SHAPES[2], ((3, 0),), torch)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            runner.tile_mask(parent.STAGE_SHAPES[2], ((0, 0), (0, 0)), torch)

    def test_actual_dual_singleton_joint_causal_input_and_service(self):
        model, targets, shapes, x, functional = build()
        before, input_before = parent.source_digest(model), x.clone()
        baseline, br, bb, captured = runner.forward_probe(model, functional, targets, shapes, x, (), merge, torch,
            capture=True, capture_u=True, capture_tiles=((0, 0), (1, 1)), capture_bn=True)
        expected_u = br["layers"][runner.TARGET]["input_sha256"]
        self.assertEqual(merge.tensor_sha(captured["u"]), expected_u)
        self.assertEqual(len(br["batch_norm_inputs"]), 24)
        self.assertTrue(all(v["input_calls"] for v in br["batch_norm_inputs"].values()))
        for selected in ((), ((0, 0),), ((1, 1),), ((0, 0), (1, 1))):
            actual, ar, ab, ac = runner.forward_probe(model, functional, targets, shapes, x, selected, merge, torch,
                capture=True, capture_tiles=selected, expected_u_sha=expected_u)
            dual, dr, db, _ = runner.forward_probe(model, functional, targets, shapes, x, selected, merge, torch,
                mode="dual_reference", capture=True, expected_u_sha=expected_u)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, dual)))
            self.assertEqual(len(ar["layers"]), 8)
            self.assertEqual(ar["summed_fc1_rows"]-br["summed_fc1_rows"], len(selected)*720)
            self.assertEqual(ar["summed_fc1_rows"], ar["summed_fc2_rows"])
            for name in ab:
                self.assertTrue(torch.equal(ab[name], db[name]))
                self.assertEqual(ar["layers"][name]["input_sha256"], dr["layers"][name]["input_sha256"])
            for r, c in selected:
                self.assertTrue(torch.equal(ac["tile_branches"][(r, c)], ab[runner.TARGET][:, :, r*9:(r+1)*9, c*9:(c+1)*9]))
            self.assertTrue(all(model.get_submodule(name) is m for name, m in targets.items()))
            self.assertTrue(all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules()))
            MEASUREMENTS.append({"selected": selected, "FC_rows": ar["summed_fc1_rows"], "dual_rows": dr["summed_fc1_rows"],
                                 "actual_dual_exact": True, "predecision_input_unchanged": True})
        self.assertTrue(torch.equal(x, input_before))
        self.assertEqual(parent.source_digest(model), before)
        with self.assertRaisesRegex(ValueError, "predecision input"):
            runner.forward_probe(model, functional, targets, shapes, x, (), merge, torch, expected_u_sha="0"*64)
        self.assertTrue(all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules()))

    def test_predecision_scores_formula_and_explicit_psn_only_cost(self):
        native = fixture.fixture.native()
        u = torch.ones(10, 1, 18, 18, 4)
        u[:, :, :9, :9] = 2
        u[:, :, 9:, 9:] = 3
        counts = {"sn1": 0, "fc": 0}
        handles = [native.sn1.register_forward_pre_hook(lambda module, inp: counts.__setitem__("sn1", counts["sn1"]+1))]
        handles.extend(getattr(native, name).register_forward_pre_hook(lambda module, inp: counts.__setitem__("fc", counts["fc"]+1))
                       for name in ("fc1", "fc2"))
        try:
            with torch.inference_mode():
                scores, cost = runner.feature_scores(u, native, merge, torch)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(counts, {"sn1": 4, "fc": 0})
        self.assertEqual([x["amplitude_abs_mean"] for x in scores], [2., 1., 1., 3.])
        self.assertTrue(all(x["pooling_residual_mse"] == x["spatial_neighbor_abs_difference"] ==
                            x["temporal_neighbor_abs_difference"] == 0 for x in scores))
        self.assertTrue(all(0 <= x["psn1_activity"] <= 1 and 0 <= x["psn1_spatial_neighbor_hamming"] <= 1 for x in scores))
        self.assertEqual(cost["diagnostic_only_extra_psn1_temporal_MACs_nominal"], 4*81*100*4)
        self.assertFalse(cost["score_reductions_and_feature_reads_counted_in_cycles"])
        self.assertEqual(set(scores[0]), {"tile"}|set(runner.SCORE_NAMES))

    def test_fixed_k2_scoring_ties_and_offline_task_definition(self):
        all_equal = {tile: 1. for tile in runner.CANDIDATES}
        self.assertEqual(runner.top2(all_equal), ((0, 0), (0, 1)))
        with self.assertRaisesRegex(ValueError, "finite"):
            runner.top2({tile: float("nan") for tile in runner.CANDIDATES})
        frame = {"arms": {runner.BASE_ARM: {"metrics": {"regions": {"all_valid": {"AEE": 10.}}}}},
                 "ordinary_scores": [], "offline_fine_coarse_branch_mse": []}
        for i, (r, c) in enumerate(runner.CANDIDATES):
            frame["arms"][f"fine_r{r}_c{c}"] = {"metrics": {"regions": {"all_valid": {"AEE": 10.-i}}}}
            frame["ordinary_scores"].append({"tile": [r, c], "amplitude_abs_mean": -i, "pooling_residual_mse": i})
            frame["offline_fine_coarse_branch_mse"].append({"tile": [r, c], "MSE": i})
        masks = runner.combination_masks(frame)
        self.assertEqual(tuple(masks), runner.COMBINATION_ARMS)
        self.assertEqual(masks["k2_fixed_uniform"], ((0, 3), (2, 0)))
        self.assertEqual(masks["k2_amplitude"], ((0, 0), (0, 1)))
        self.assertEqual(masks["k2_posthoc_singleton_task_gain"], ((2, 2), (2, 3)))
        self.assertEqual(masks["k2_offline_branch_mse"], masks["k2_pooling_residual"])
        self.assertEqual(len(set(masks.values())), 3)

    def test_failure_restores_models_and_registered_hooks(self):
        model, targets, shapes, x, functional = build()
        before = parent.source_digest(model)
        target = targets[runner.TARGET]
        def fail(module, inp):
            raise RuntimeError("injected MLP failure")
        handle = target.fc1.register_forward_pre_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError, "injected MLP"):
                runner.forward_probe(model, functional, targets, shapes, x, ((0, 0),), merge, torch,
                                     capture=True, capture_u=True, capture_tiles=((0, 0),), capture_bn=True)
        finally:
            handle.remove()
        self.assertEqual(parent.source_digest(model), before)
        self.assertTrue(all(model.get_submodule(name) is m for name, m in targets.items()))
        self.assertTrue(all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules()))
        second = list(targets.values())[1]
        with patch.object(second.fc1, "register_forward_pre_hook", side_effect=RuntimeError("injected registration")):
            with self.assertRaisesRegex(RuntimeError, "registration"):
                runner.forward_probe(model, functional, targets, shapes, x, (), merge, torch, capture_bn=True)
        self.assertTrue(all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules()))

    def test_source_import_does_not_initialize_GPU_or_change_visibility(self):
        code = ("import os,sys; os.environ['CUDA_VISIBLE_DEVICES']='marker'; sys.path.insert(0,"+repr(str(HERE))+"); "
                "import run_tile_interventions as r; p=r.load_scope_runner(); p.load_frozen_merge(); "
                "assert os.environ['CUDA_VISIBLE_DEVICES']=='marker'; import torch; assert not torch.cuda.is_initialized()")
        child = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=20)
        self.assertEqual(child.returncode, 0, child.stderr)


def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.manual_seed(20261014)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    sources = [Path(__file__), Path(runner.__file__), runner.SCOPE_RUNNER, runner.BASE/"structured_merge_mlp.py",
               runner.BASE/"scope_extension_v1/test_scope_extension_cpu.py", runner.BASE/"test_structured_merge_cpu.py", fixture.fixture.FACTORY]
    (HERE/"runner_cpu_results.json").write_text(json.dumps({"passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "torch": str(torch.__version__), "GPU_initialized": torch.cuda.is_initialized(), "full_model_run": False,
        "source_sha256": {str(p): runner.sha(p) for p in sources}, "measurements": MEASUREMENTS,
        "scope": "Production singleton/joint whole-tile path on twelve small native MLP fixtures, actual/dual exact, predecision input equality, rows and BN hooks, scoring formulas, tie rules and exceptional cleanup; no full model/GPU/remote run",
        "errors": [str(x) for x in result.errors], "failures": [str(x) for x in result.failures]}, indent=2)+"\n")
    return 0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1


if __name__ == "__main__":
    raise SystemExit(main())
