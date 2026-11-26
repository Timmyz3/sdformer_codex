#!/usr/bin/env python3
"""Own 27-anchor-path checks on tiny native MLPs only; no GPU or real model."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode = True
import torch
import run_anchors as runner
import source_contract
sys.path.insert(0, str(source_contract.source_path("fixture").parent))
fixture = source_contract.load_source("fixture", "_anchors_cpu_toy")
MEASUREMENTS = []


class Tests(unittest.TestCase):
    def test_native_fullfine_layerfine_tails_dual_prefix_and_cleanup(self):
        r10, collector, probe, parent, r9, r8 = runner.load_runtime()
        torch.manual_seed(20261015)
        model, targets, shapes, x, functional = fixture.build()
        before = parent.source_digest(model)
        def forward(arm, **kwargs):
            return runner.forward_arm(r9, r8, parent, model, functional, targets, shapes, x, arm, fixture.merge, torch, **kwargs)
        native = forward("native")
        with torch.inference_mode():
            direct = model(x)["flow"]
        self.assertEqual(len(native[0]), len(direct))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(native[0], direct)))
        self.assertEqual(native[1]["layers"], {})
        cc = forward("CC")
        archived = {"arms": {"coarse_stage02": {"execution": cc[1]}}}
        for arm in ("FF", "fine_s2b2"):
            actual = forward(arm, capture=True)
            dual = forward(arm, mode="dual_reference", capture=True)
            pair = runner.check_pair(actual, dual, torch)
            self.assertEqual(pair["flow_list_length"], 2)
            self.assertEqual(actual[1]["summed_fc1_rows"], 18480 if arm == "FF" else 9100)
            self.assertEqual(dual[1]["summed_fc1_rows"], 20080)
            for name, layer in actual[1]["layers"].items():
                fine = arm == "FF" or name == r10.TARGET
                self.assertEqual(layer["fine_mask"], [[fine, fine], [fine, fine]])
                self.assertEqual(layer["executed_fc_rows_per_layer"], 1540 if fine else 200)
                self.assertEqual(layer["executed_pool_input_elements"], 0 if fine else 6160)
                self.assertEqual(layer["executed_repeat_write_elements"], 0 if fine else 6160)
                self.assertEqual(layer["selected_service"]["external_residual_additions_unchanged"], 6160)
                self.assertIn([10, 1, 2, 5, 4] if fine else [10, 1, 1, 2, 4],
                              layer["observed"]["native_input_shapes"]["fine" if fine else "coarse"])
            if arm == "fine_s2b2":
                runner.check_prefix(actual[1], archived, r10.PREFIX_NAMES)
                corrupt = copy.deepcopy(actual[1])
                corrupt["layers"][r10.PREFIX_NAMES[0]]["output_sha256"] = "0"*64
                with self.assertRaisesRegex(ValueError, "earlier prefix output"):
                    runner.check_prefix(corrupt, archived, r10.PREFIX_NAMES)
            with self.assertRaisesRegex(ValueError, "complete flow list"):
                runner.check_pair(actual, (dual[0]+[dual[0][-1]], dual[1], dual[2]), torch)
            self.assertTrue(all(model.get_submodule(name) is module for name, module in targets.items()))
            self.assertTrue(all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules()))
        self.assertEqual(cc[1]["summed_fc1_rows"], 7760)
        self.assertEqual(parent.source_digest(model), before)
        MEASUREMENTS.append({"fixture_shape_all12": [10, 1, 11, 14, 4], "each_fc_rows":
            {"native": 18480, "FF": 18480, "CC": 7760, "fine_s2b2": 9100, "dual": 20080},
            "pairs": ["FF", "fine_s2b2"], "all_flows_inputs_branches_exact": True,
            "real_tail_fine_shape": [10, 1, 2, 5, 4], "real_tail_coarse_shape": [10, 1, 1, 2, 4]})

    def test_failure_restores_private_wrapper_native_modules_and_hooks(self):
        r10, _, _, parent, r9, r8 = runner.load_runtime()
        model, targets, shapes, x, functional = fixture.build()
        before = parent.source_digest(model); original = parent.wrapped_model
        with patch.object(targets[r10.TARGET].fc2, "forward", side_effect=RuntimeError("injected target failure")):
            with self.assertRaisesRegex(RuntimeError, "injected target"):
                runner.forward_arm(r9, r8, parent, model, functional, targets, shapes, x, "fine_s2b2", fixture.merge, torch)
        self.assertIs(parent.wrapped_model, original)
        self.assertTrue(all(model.get_submodule(name) is module for name, module in targets.items()))
        self.assertTrue(all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules()))
        self.assertEqual(parent.source_digest(model), before)

    def test_production_geometry_and_fixed_27_budget(self):
        shapes = {"sttmultires_unet.encoders.swin3d.layers.{}.swin_blocks.{}.mlp".format(s, b):
                  (10, 1, h, w, c) for s, depth, h, w, c in
                  ((0, 2, 120, 160, 96), (1, 2, 60, 80, 192), (2, 6, 30, 40, 384), (3, 2, 15, 20, 768))
                  for b in range(depth)}
        for arm, count in runner.ROWS.items():
            self.assertEqual(runner.expected_rows(shapes, arm), count)
        for arm in ("FF", "fine_s2b2"):
            self.assertEqual(runner.expected_rows(shapes, arm, "dual_reference"), 609600)
        self.assertEqual(runner.BUDGET, 8*len(runner.ARMS)+1+2)
        weighted = {arm: sum(runner.expected_rows({name: shape}, arm)*8*shape[-1]**2
                            for name, shape in shapes.items()) for arm in runner.ROWS}
        self.assertEqual(weighted["native"], weighted["FF"])
        self.assertEqual(weighted["native"], 169869312000)
        self.assertEqual(weighted["fine_s2b2"], 82221465600)
        MEASUREMENTS.append({"production_geometry_arithmetic_only": True, "each_fc_rows": runner.ROWS,
            "dual_each_fc_rows": 609600, "two_fc_weight_products": weighted, "cycles_claim": False})

    def test_source_identity_receipt_and_safe_import(self):
        for key in runner.pins()["sources"]:
            runner.source_path(key)
        reference = json.loads((runner.R10/"collect_v1/results.json").read_text())
        self.assertEqual(runner.sha(runner.R10/"collect_v1/results.json"), runner.pins()["round10_result_sha256"])
        member = json.loads(runner.source_path("membership").read_text())
        self.assertEqual([f["target_id"] for f in reference["frames"] if f["group"] == "check"], member["groups"]["check"])
        self.assertEqual(len(member["groups"]["check"]), 8)
        _, _, _, parent, _, _ = runner.load_runtime()
        sources = [Path(runner.__file__), Path(__file__)]
        good = {"passed": True, "GPU_initialized": False, "source_sha256": {p.name: runner.sha(p) for p in sources}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"receipt.json"; path.write_text(json.dumps(good))
            parent.checked_cpu_receipt(path, sources)
            good["source_sha256"]["run_anchors.py"] = "0"*64; path.write_text(json.dumps(good))
            with self.assertRaisesRegex(ValueError, "current source"):
                parent.checked_cpu_receipt(path, sources)
        code = ("import os,sys;os.environ['CUDA_VISIBLE_DEVICES']='marker';sys.path.insert(0,"+repr(str(runner.HERE))+");"
                "import run_anchors as r;r.load_runtime();assert os.environ['CUDA_VISIBLE_DEVICES']=='marker';"
                "import torch;assert not torch.cuda.is_initialized()")
        child = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(child.returncode, 0, child.stderr)


def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    sources = [Path(runner.__file__), Path(__file__), Path(fixture.__file__), Path(fixture.fixture.__file__), fixture.fixture.FACTORY]
    receipt = {"schema": "flow_round10_native_anchor_cpu_v1", "passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "torch": torch.__version__, "GPU_initialized": torch.cuda.is_initialized(), "full_model_run": False,
        "source_sha256": {str(p): runner.sha(p) for p in sources}, "source_pins_sha256": runner.sha(runner.HERE/"source_pins.json"),
        "measurements": MEASUREMENTS, "errors": [str(x) for x in result.errors], "failures": [str(x) for x in result.failures],
        "scope": "Implementer CPU toy execution: native/FF/full-block2, both dual pairs/all flow lengths, exact prefix, true fine tails, cleanup and source/identity contracts. Real model not executed; production geometry arithmetic only."}
    (runner.HERE/"runner_cpu_results.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return 0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1


if __name__ == "__main__":
    raise SystemExit(main())
