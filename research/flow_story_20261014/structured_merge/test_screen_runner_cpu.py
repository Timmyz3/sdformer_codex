#!/usr/bin/env python3
"""CPU tests of screen setup/dispatch only; never load or run the full model."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
sys.dont_write_bytecode = True
import torch
import run_structured_merge_screen as runner
import structured_merge_mlp as merge
import test_structured_merge_cpu as fixture

HERE = Path(__file__).resolve().parent


class Tests(unittest.TestCase):
    def test_parse_replaces_output_before_inherited_parser(self):
        original = ["--foo", "bar", "--output", "/old/results.json"]
        seen = []
        def parse(args):
            seen.append(args)
            self.assertEqual(args[-1], "/new/results.json")
            return types.SimpleNamespace(output=args[-1])
        result = runner.parse_parent_args(original, Path("/new/results.json"), types.SimpleNamespace(parse_extended=parse))
        self.assertEqual(result.output, "/new/results.json")
        self.assertEqual(original[-1], "/old/results.json")
        self.assertEqual(len(seen), 1)

    def test_bank_only_reads_fixed_identities_and_membership(self):
        ids = ["target_"+str(i) for i in range(8)]
        bank = {"schema": "flow_fast_recovery_mask_bank_v1", "status": "MASK_BANK_COMPLETE", "dev_ids": ids,
                "binding": {"checkpoint_sha256": runner.CHECKPOINT_SHA, "dev_ids_sha256": runner.digest(ids)}}
        rows = [{"target_id": target, "protocol_split": "train"} for target in reversed(ids)]
        roles = {target: "gate_development" for target in ids}
        result = runner.fixed_dev_rows(bank, rows, roles)
        self.assertEqual([row["target_id"] for row in result], ids)
        # There are deliberately no frame masks, u hashes, head or fast weights.
        roles[ids[0]] = "gate_fit"
        with self.assertRaisesRegex(ValueError, "development"):
            runner.fixed_dev_rows(bank, rows, roles)

    def test_arm_matrix_keeps_each_scope_matched(self):
        arms = {name: (kind, stages) for name, kind, stages in runner.ARMS}
        self.assertEqual(len(arms), 8)
        for coarse, fine in runner.MATCHED.items():
            self.assertEqual(arms[coarse][1], arms[fine][1])
            self.assertEqual(arms[fine][0], "fine")
        self.assertEqual(arms["coarse_stage0"][1], (0,))
        self.assertEqual(arms["coarse_stage2"][1], (2,))
        self.assertEqual(sum(runner.DEPTHS.values()), 12)

    def test_scope_dispatch_zero_rows_residual_and_restoration(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.cnf = "ADD"; self.use_checkpoint = False; self.mlp = fixture.native()
            def forward(self, x):
                return x+self.mlp(x)
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([Block(), Block(), Block()])
        model.eval().requires_grad_(False)
        targets = {"layers."+str(i)+".mlp": model.layers[i].mlp for i in range(3)}
        # Names contain the production matching convention .layers.<stage>.
        root = torch.nn.Module(); root.enc = model; root.eval().requires_grad_(False)
        targets = {"enc."+name: module for name, module in targets.items()}
        shapes = {name: (10, 1, 11, 14, 4) for name in targets}
        u = torch.randn(10, 1, 11, 14, 4)
        calls = []
        handle = model.layers[0].mlp.fc1.register_forward_pre_hook(lambda module, inputs: calls.append(inputs[0].shape))
        try:
            with torch.inference_mode(), runner.wrapped_model(root, targets, shapes, "zero", (0,), "actual", merge, torch) as wrappers:
                self.assertEqual(set(wrappers), {"enc.layers.0.mlp"})
                self.assertTrue(torch.equal(model.layers[0](u), u))
                self.assertEqual(calls, [])
            self.assertIs(model.layers[0].mlp, targets["enc.layers.0.mlp"])
            with torch.inference_mode(), runner.wrapped_model(root, targets, shapes, "coarse", (2,), "actual", merge, torch) as wrappers:
                result = model.layers[2](u)
                self.assertEqual(set(wrappers), {"enc.layers.2.mlp"})
                self.assertEqual(result.shape, u.shape)
                record = wrappers["enc.layers.2.mlp"].records[0]
                self.assertEqual(record["executed_fc_rows_per_layer"], 200)
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with runner.wrapped_model(root, targets, shapes, "fine", (0, 1, 2), "actual", merge, torch):
                    raise RuntimeError("injected")
            self.assertTrue(all(root.get_submodule(name) is original for name, original in targets.items()))
        finally:
            handle.remove()

    def test_import_preserves_device_environment_and_avoids_fitter(self):
        code = ("import os,sys; os.environ['CUDA_VISIBLE_DEVICES']='marker'; sys.path.insert(0,"+repr(str(HERE))+"); "
                "import run_structured_merge_screen; assert os.environ['CUDA_VISIBLE_DEVICES']=='marker'; "
                "assert 'fit_fast_cache' not in sys.modules; import torch; assert not torch.cuda.is_initialized()")
        child = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=15)
        self.assertEqual(child.returncode, 0, child.stderr)


def main():
    torch.set_num_threads(2)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    sources = [Path(__file__), Path(runner.__file__), Path(merge.__file__)]
    (HERE/"runner_cpu_results.json").write_text(json.dumps({
        "passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "GPU_initialized": torch.cuda.is_initialized(), "full_model_run": False,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "scope": "Synthetic dispatch, fixed identity-only membership, output argument replacement and safe import; remote preflight/GPU pending",
        "errors": [str(value) for value in result.errors], "failures": [str(value) for value in result.failures],
    }, indent=2)+"\n")
    return 0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1


if __name__ == "__main__":
    raise SystemExit(main())
