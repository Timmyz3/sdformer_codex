#!/usr/bin/env python3
"""Minimal formula, tile-selection, cost and source checks; no GPU/full model."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
import json
from pathlib import Path
import subprocess
import sys
import unittest
sys.dont_write_bytecode = True
import torch
import torch.nn.functional as F
import supplemental_snn_scores as runner
probe, parent, tp = runner.load_dependencies()
HERE = Path(__file__).resolve().parent
MEASUREMENTS = []


def ordinary():
    return [{"tile": list(tile), "psn1_activity": i/12, "psn1_spatial_neighbor_hamming": (11-i)/12}
            for i, tile in enumerate(probe.CANDIDATES)]


class Tests(unittest.TestCase):
    def test_official_score_retains_dense_channel_mixing_and_two_softmaxes(self):
        x = torch.randn(10, 1, 3, 4, 5)
        with torch.inference_mode():
            actual = tp.official_sdt_scores(x)
            mean_channels = x.mean(2)
            temporal = (mean_channels[1:]-mean_channels[:-1]).abs().mean(0).flatten(1).softmax(1).reshape(1, 4, 5)
            mean_time = x.mean(0)
            dense = F.conv2d(mean_time, torch.ones(3, 3, 3, 3)/9, padding=1)
            spatial = (-F.cosine_similarity(mean_time.flatten(2), dense.flatten(2), dim=1, eps=1e-6)).softmax(1).reshape(1, 4, 5)
            depthwise = F.conv2d(mean_time, torch.ones(3, 1, 3, 3)/9, padding=1, groups=3)
            cheap_spatial = (-F.cosine_similarity(mean_time.flatten(2), depthwise.flatten(2), dim=1, eps=1e-6)).softmax(1).reshape(1, 4, 5)
        self.assertTrue(torch.equal(actual["temporal"], temporal))
        self.assertTrue(torch.equal(actual["spatial"], spatial))
        self.assertTrue(torch.equal(actual["total"], temporal+spatial))
        self.assertFalse(torch.equal(spatial, cheap_spatial))
        self.assertEqual(actual["cost"]["official_dense_conv_mac_including_padded_taps"], 1*4*5*3*3*9)
        self.assertEqual(actual["cost"]["softmax_calls"], 2)

    def test_real_full_shape_score_and_same_candidates_without_GT(self):
        u = torch.randn(10, 1, 30, 40, 384)
        saved = u.clone()
        with torch.inference_mode():
            masks, records, maps = runner.masks_and_scores(u, ordinary(), probe, tp, torch)
        manual = {tile: float(maps["total"][0, tile[0]*9:(tile[0]+1)*9, tile[1]*9:(tile[1]+1)*9].mean())
                  for tile in probe.CANDIDATES}
        self.assertEqual(masks[runner.ARMS[0]], probe.top2(manual))
        self.assertEqual(masks[runner.ARMS[1]], ((2, 2), (2, 3)))
        self.assertEqual(masks[runner.ARMS[2]], ((0, 0), (0, 1)))
        self.assertTrue(all(not x["GT_used_for_selection"] for x in records.values()))
        self.assertTrue(all(len(mask) == 2 and set(mask) <= set(probe.CANDIDATES) for mask in masks.values()))
        self.assertTrue(torch.equal(saved, u))
        self.assertAlmostEqual(float(maps["total"].sum()), 2., places=5)
        cost = records[runner.ARMS[0]]["cost"]
        self.assertEqual(cost["official_formula_cost"]["official_dense_conv_mac_including_padded_taps"], 1592524800)
        self.assertEqual(cost["semantic_reduction_and_nonlinear_volumes"]["softmax_exponentials"], 2400)
        self.assertEqual(cost["semantic_reduction_and_nonlinear_volumes"]["softmax_sum_reduction_additions"], 2398)
        self.assertEqual(cost["semantic_reduction_and_nonlinear_volumes"]["complete_tile_mean_additions"], 960)
        MEASUREMENTS.append({"input_shape": list(u.shape), "masks": masks, "cost": cost,
                             "full_map_total_sum": float(maps["total"].sum()), "input_unchanged": True})

    def test_tile_mean_excludes_tail_candidates_and_lexicographic_ties(self):
        total = torch.ones(1, 30, 40)
        total[:, 27:, :] = 1e6
        total[:, :, 36:] = 1e6
        equal = [{"tile": list(tile), "psn1_activity": 1., "psn1_spatial_neighbor_hamming": 1.} for tile in probe.CANDIDATES]
        masks, records = runner.masks_from_score_maps({"total": total}, equal, probe)
        self.assertTrue(all(x == ((0, 0), (0, 1)) for x in masks.values()))
        self.assertTrue(all(score["score"] == 1. for score in records[runner.ARMS[0]]["unit_scores"]))
        with self.assertRaisesRegex(ValueError, "candidates"):
            runner.masks_from_score_maps({"total": total}, equal[:-1], probe)

    def test_parent_alias_uses_mask_identity_and_no_quality_fields(self):
        frame = {"combination_masks": {"z_offline_GT": [[0, 0], [0, 1]], "a_ordinary": [[0, 0], [0, 1]],
                                       "other": [[0, 3], [2, 0]]}}
        self.assertEqual(runner.parent_alias(frame, ((0, 0), (0, 1))), "a_ordinary")
        self.assertEqual(runner.parent_alias(frame, ((0, 3), (2, 0))), "other")
        self.assertIsNone(runner.parent_alias(frame, ((1, 1), (1, 2))))
        self.assertEqual(len(runner.ARMS)*8, 24)

    def test_pins_and_safe_import(self):
        self.assertEqual(runner.sha(HERE/"run_tile_interventions.py"), runner.PROBE_SHA)
        self.assertEqual(runner.sha(HERE/"frozen_tp_score.py"), runner.TP_SHA)
        self.assertEqual(tp.UPSTREAM_REVISION, "bcc915ad62254efa45800245d8abcf40905618fc")
        code = ("import os,sys; os.environ['CUDA_VISIBLE_DEVICES']='marker'; sys.path.insert(0,"+repr(str(HERE))+"); "
                "import supplemental_snn_scores as r; r.load_dependencies(); assert os.environ['CUDA_VISIBLE_DEVICES']=='marker'; "
                "import torch; assert not torch.cuda.is_initialized()")
        child = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=20)
        self.assertEqual(child.returncode, 0, child.stderr)


def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.manual_seed(20261014)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    sources = [Path(__file__), Path(runner.__file__), HERE/"frozen_tp_score.py", HERE/"run_tile_interventions.py"]
    (HERE/"supplemental_cpu_results.json").write_text(json.dumps({"passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "torch": str(torch.__version__), "GPU_initialized": torch.cuda.is_initialized(), "full_model_run": False,
        "source_sha256": {str(p): runner.sha(p) for p in sources}, "measurements": MEASUREMENTS,
        "scope": "Exact TP SDT scoring algebra and true full input shape, dense channel-mixing conv vs depthwise counterexample, full-map softmax costs, complete-tile aggregation/ties, no-GT mask aliasing and safe imports; no GPU/full optical-flow model",
        "errors": [str(x) for x in result.errors], "failures": [str(x) for x in result.failures]}, indent=2)+"\n")
    return 0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1


if __name__ == "__main__":
    raise SystemExit(main())
