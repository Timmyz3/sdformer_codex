#!/usr/bin/env python3
"""Small native-MLP CPU fixtures for the new scope-extension execution path."""
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
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
sys.dont_write_bytecode = True
import torch
import run_scope_extension as runner

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
spec = importlib.util.spec_from_file_location("scope_extension_native_fixture", BASE/"test_structured_merge_cpu.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
merge = runner.load_frozen_merge()
MEASUREMENTS = []


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnf, self.use_checkpoint = "ADD", False
        self.mlp = fixture.native()

    def forward(self, x):
        return x+self.mlp(x)


class Stage(torch.nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.swin_blocks = torch.nn.ModuleList([Block() for _ in range(depth)])

    def forward(self, x):
        for block in self.swin_blocks:
            x = block(x)
        return x


class TwelveSmallMLPs(torch.nn.Module):
    """Only twelve synthetic native MLPs; no attention, stem, decoder or full model."""
    def __init__(self):
        super().__init__()
        self.sttmultires_unet = torch.nn.Module()
        self.sttmultires_unet.encoders = torch.nn.Module()
        self.sttmultires_unet.encoders.swin3d = torch.nn.Module()
        self.sttmultires_unet.encoders.swin3d.layers = torch.nn.ModuleList([Stage(depth) for depth in runner.DEPTHS.values()])

    def forward(self, x):
        for layer in self.sttmultires_unet.encoders.swin3d.layers:
            x = layer(x)
        # Both returned tensors are fixtures, not optical-flow predictions.
        return {"flow": [x[:1, 0, :, :, :2].permute(0, 3, 1, 2),
                         x[-1:, 0, :, :, :2].permute(0, 3, 1, 2)]}


def build():
    model = TwelveSmallMLPs().eval().requires_grad_(False)
    targets, declared = runner.target_layers(model)
    shapes = {name: (10, 1, 11, 14, 4) for name in targets}
    x = torch.randn(10, 1, 11, 14, 4)
    functional = types.SimpleNamespace(reset_net=lambda model: None)
    return model, targets, shapes, x, functional


class Tests(unittest.TestCase):
    def test_seven_fixed_arms_and_fifty_nine_forwards(self):
        expected = (("native_dense", "native", ()),
                    ("fine_stage02", "fine", (0, 2)), ("coarse_stage02", "coarse", (0, 2)),
                    ("fine_stage23", "fine", (2, 3)), ("coarse_stage23", "coarse", (2, 3)),
                    ("fine_stage023", "fine", (0, 2, 3)), ("coarse_stage023", "coarse", (0, 2, 3)))
        self.assertEqual(runner.ARMS, expected)
        self.assertEqual(8*len(runner.ARMS)+len(runner.MATCHED), 59)
        arms = {name: (kind, stages) for name, kind, stages in runner.ARMS}
        for name, baseline in runner.MATCHED.items():
            self.assertEqual(arms[name][1], arms[baseline][1])
            self.assertEqual(arms[baseline][0], "fine")
        self.assertEqual([sum(runner.DEPTHS[s] for s in arms[name][1]) for name in runner.MATCHED], [8, 8, 10])

    def test_new_forward_path_combined_actual_dual_rows_and_restoration(self):
        model, targets, shapes, x, functional = build()
        before, original_input = runner.source_digest(model), x.clone()
        for stages in ((0, 2), (2, 3), (0, 2, 3)):
            selected = {name for name in targets if any(f".layers.{stage}." in name for stage in stages)}
            count = len(selected)
            fine, fr, _ = runner.forward_probe(model, functional, targets, shapes, x, "fine", stages, merge, torch)
            actual, ar, ab = runner.forward_probe(model, functional, targets, shapes, x, "coarse", stages, merge, torch, capture=True)
            dual, dr, db = runner.forward_probe(model, functional, targets, shapes, x, "coarse", stages, merge, torch,
                                                mode="dual_reference", capture=True)
            self.assertEqual(set(ar["layers"]), selected)
            self.assertEqual(set(ab), selected)
            self.assertEqual(set(db), selected)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, dual)))
            self.assertEqual(fr["summed_fc1_rows"], 12*1540)
            self.assertEqual(ar["summed_fc1_rows"], count*200+(12-count)*1540)
            self.assertEqual(dr["summed_fc1_rows"], count*1740+(12-count)*1540)
            self.assertEqual(ar["summed_fc1_rows"], ar["summed_fc2_rows"])
            for name in targets:
                if name in selected:
                    self.assertTrue(torch.equal(ab[name], db[name]))
                    self.assertEqual(ar["layers"][name]["input_sha256"], dr["layers"][name]["input_sha256"])
                    self.assertEqual(ar["layers"][name]["executed_fc_rows_per_layer"], 200)
                self.assertIs(model.get_submodule(name), targets[name])
            self.assertTrue(all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules()))
            MEASUREMENTS.append({"stages": list(stages), "selected_layers": sorted(selected),
                                 "actual_flow_exact_to_dual": True, "actual_branches_exact_to_dual": True,
                                 "fine_rows": fr["summed_fc1_rows"], "actual_rows": ar["summed_fc1_rows"],
                                 "dual_rows": dr["summed_fc1_rows"], "all12_actual": ar["all12_observed_fc_rows"]})
        self.assertEqual(runner.source_digest(model), before)
        self.assertTrue(torch.equal(x, original_input))

    def test_native_no_wrap_path_and_exact_target_guard(self):
        model, targets, shapes, x, functional = build()
        output, record, branches = runner.forward_probe(model, functional, targets, shapes, x, "native", (), merge, torch)
        self.assertEqual(record["layers"], {})
        self.assertEqual(branches, {})
        self.assertEqual(record["summed_fc1_rows"], 18480)
        with torch.inference_mode():
            direct = model(x)["flow"]
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(output, direct)))
        model.sttmultires_unet.encoders.swin3d.layers[1].swin_blocks[0].mlp = torch.nn.Identity()
        with self.assertRaisesRegex(ValueError, "twelve"):
            runner.target_layers(model)

    def test_forward_failure_restores_multi_scope_wrappers_and_hooks(self):
        model, targets, shapes, x, functional = build()
        original_state = runner.source_digest(model)
        failing = next(module for name, module in targets.items() if ".layers.2.swin_blocks.1." in name)
        def injected(module, inputs):
            raise RuntimeError("injected producer failure")
        handle = failing.fc2.register_forward_pre_hook(injected)
        try:
            with self.assertRaisesRegex(RuntimeError, "injected producer"):
                runner.forward_probe(model, functional, targets, shapes, x, "coarse", (0, 2, 3), merge, torch, capture=True)
        finally:
            handle.remove()
        self.assertTrue(all(model.get_submodule(name) is module for name, module in targets.items()))
        self.assertTrue(all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules()))
        self.assertEqual(runner.source_digest(model), original_state)

    def test_partial_hook_registration_failure_is_cleaned(self):
        model, targets, shapes, x, functional = build()
        second = list(targets.values())[1]
        with patch.object(second.fc1, "register_forward_pre_hook", side_effect=RuntimeError("injected registration")):
            with self.assertRaisesRegex(RuntimeError, "injected registration"):
                runner.forward_probe(model, functional, targets, shapes, x, "coarse", (0, 2), merge, torch)
        self.assertTrue(all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules()))
        self.assertTrue(all(model.get_submodule(name) is module for name, module in targets.items()))

    def test_identity_only_membership_and_round3_binding(self):
        ids = ["target_"+str(i) for i in range(8)]
        bank = {"schema": "flow_fast_recovery_mask_bank_v1", "status": "MASK_BANK_COMPLETE", "dev_ids": ids,
                "binding": {"checkpoint_sha256": runner.CHECKPOINT_SHA, "dev_ids_sha256": runner.digest(ids)}}
        rows = [{"target_id": target, "protocol_split": "train"} for target in reversed(ids)]
        roles = {target: "gate_development" for target in ids}
        selected = runner.fixed_dev_rows(bank, rows, roles)
        self.assertEqual([row["target_id"] for row in selected], ids)
        receipt = {"schema": "flow_structured_merge_screen_v1", "status": "STRUCTURED_MERGE_DEV8_COMPLETE",
                   "completed_targets": 8, "full_forward_count": 67, "original_model_unchanged": True, "BN_buffers_unchanged": True,
                   "binding": {"module_sha256": runner.MODULE_SHA, "runner_sha256": runner.PARENT_RUNNER_SHA,
                               "checkpoint_sha256": runner.CHECKPOINT_SHA, "identity_bank_sha256": runner.BANK_SHA, "dev_ids": ids}}
        runner.validate_round3(receipt, ids)
        with self.assertRaisesRegex(ValueError, "identity/order"):
            runner.validate_round3(receipt, list(reversed(ids)))
        receipt["binding"]["module_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "source/numerical"):
            runner.validate_round3(receipt, ids)

    def test_cpu_receipt_requires_the_new_runner_and_test_sources(self):
        sources = [Path(runner.__file__), Path(__file__), BASE/"structured_merge_mlp.py"]
        good = {"passed": True, "GPU_initialized": False, "source_sha256": {p.name: runner.sha(p) for p in sources}}
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory)/"receipt.json"
            path.write_text(json.dumps(good))
            runner.checked_cpu_receipt(path, sources)
            good["source_sha256"][Path(runner.__file__).name] = runner.PARENT_RUNNER_SHA
            path.write_text(json.dumps(good))
            with self.assertRaisesRegex(ValueError, "current source"):
                runner.checked_cpu_receipt(path, sources)

    def test_parse_replaces_output_and_import_preserves_device_policy(self):
        original = ["--output", "/old/results.json"]
        def parser(arguments):
            self.assertEqual(arguments[-1], "/new/results.json")
            return arguments
        runner.parse_parent_args(original, Path("/new/results.json"), types.SimpleNamespace(parse_extended=parser))
        self.assertEqual(original[-1], "/old/results.json")
        code = ("import os,sys; os.environ['CUDA_VISIBLE_DEVICES']='marker'; sys.path.insert(0,"+repr(str(HERE))+"); "
                "import run_scope_extension as r; r.load_frozen_merge(); assert os.environ['CUDA_VISIBLE_DEVICES']=='marker'; "
                "assert 'fit_fast_cache' not in sys.modules; import torch; assert not torch.cuda.is_initialized()")
        child = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=15)
        self.assertEqual(child.returncode, 0, child.stderr)


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    sources = [Path(__file__), Path(runner.__file__), BASE/"structured_merge_mlp.py", BASE/"test_structured_merge_cpu.py", fixture.FACTORY]
    (HERE/"runner_cpu_results.json").write_text(json.dumps({
        "passed": result.wasSuccessful(), "tests_run": result.testsRun, "torch": torch.__version__,
        "GPU_initialized": torch.cuda.is_initialized(), "full_model_run": False,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "forked_parent_runner_sha256": runner.PARENT_RUNNER_SHA,
        "scope": "New production forward_probe on twelve small native MLP fixtures; combined scopes, actual/dual, all-layer rows, native restoration, hook failures, source receipts and safe import; no full model/remote/GPU",
        "measurements": MEASUREMENTS,
        "errors": [str(value) for value in result.errors], "failures": [str(value) for value in result.failures],
    }, indent=2)+"\n")
    return 0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1


if __name__ == "__main__":
    raise SystemExit(main())
