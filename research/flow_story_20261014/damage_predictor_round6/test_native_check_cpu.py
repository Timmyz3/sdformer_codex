#!/usr/bin/env python3
"""Native anchor uses the existing unwrapped path; no complete model or GPU."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
import torch
import collect_native_check as runner
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
COLLECTOR = runner.load_collector(); PROBE = COLLECTOR.load_probe(); PARENT = PROBE.load_scope_runner()
sys.path.insert(0, str(PROBE.BASE/"scope_extension_v1"))
FIXTURE_PATH = PROBE.BASE/"scope_extension_v1/test_scope_extension_cpu.py"
spec = importlib.util.spec_from_file_location("_native_check_small_fixture", FIXTURE_PATH)
FIXTURE = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIXTURE)
MERGE = PARENT.load_frozen_merge()
MEASUREMENTS = []


class Tests(unittest.TestCase):
    def test_native_unwrapped_forward_matches_native_model_and_counts(self):
        model, targets, shapes, x, functional = FIXTURE.build()
        before = PARENT.source_digest(model)
        with torch.inference_mode():
            reference = model(x)["flow"]
        flows, execution = runner.native_forward(PARENT, model, functional, targets, shapes, x, MERGE, torch)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(reference, flows)))
        self.assertEqual(execution["layers"], {})
        self.assertEqual(len(execution["all12_observed_fc_rows"]), 12)
        self.assertEqual(execution["summed_fc1_rows"], 12*10*11*14)
        self.assertEqual(execution["summed_fc1_rows"], execution["summed_fc2_rows"])
        self.assertEqual(PARENT.source_digest(model), before)
        self.assertTrue(all(model.get_submodule(name) is m for name, m in targets.items()))
        self.assertTrue(all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules()))
        MEASUREMENTS.append({"native_model_and_hooked_path_exact": True, "small_fixture_fc_rows": 18480,
                             "wrapped_MLP_count": 0, "source_state_unchanged": True})

    def test_fixed_declared_native_shape_service(self):
        shapes = [PARENT.STAGE_SHAPES[s] for s, depth in PARENT.DEPTHS.items() for _ in range(depth)]
        self.assertEqual(sum(math.prod(s[:-1]) for s in shapes), 558000)
        self.assertEqual(len(shapes), 12)
        MEASUREMENTS.append({"declared_full_model_FC_rows_per_FC": 558000, "native_check_forwards": 8,
                             "scope": "Static dimension count, not measured cycles or GPU latency"})

    def test_import_no_GPU_visibility_or_model_side_effect(self):
        code = ("import os,sys;os.environ['CUDA_VISIBLE_DEVICES']='marker';sys.path.insert(0,"+repr(str(HERE))+");"
                "import collect_native_check as n;n.load_collector();assert os.environ['CUDA_VISIBLE_DEVICES']=='marker';"
                "import torch;assert not torch.cuda.is_initialized()")
        child = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=20)
        self.assertEqual(child.returncode, 0, child.stderr)


def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(20261014)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    sources = [Path(__file__), Path(runner.__file__), runner.COLLECTOR_PATH, PROBE.SCOPE_RUNNER,
               PROBE.BASE/"structured_merge_mlp.py", FIXTURE_PATH]
    receipt = {"passed": result.wasSuccessful(), "tests_run": result.testsRun, "torch": str(torch.__version__),
               "GPU_initialized": torch.cuda.is_initialized(), "full_model_run": False,
               "source_sha256": {str(p): runner.sha(p) for p in sources}, "measurements": MEASUREMENTS,
               "scope": "Unwrapped frozen native forward on small12MLP fixture, exact direct-model match and full row hooks; safe import; no real check data or full model/GPU run",
               "errors": [str(x) for x in result.errors], "failures": [str(x) for x in result.failures]}
    (HERE/"native_cpu_results.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return 0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1


if __name__ == "__main__":
    raise SystemExit(main())
