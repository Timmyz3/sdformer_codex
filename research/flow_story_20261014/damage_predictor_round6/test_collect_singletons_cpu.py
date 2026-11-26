#!/usr/bin/env python3
"""CPU-only membership, label support and causal descriptor tests for new cache."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import unittest
import torch
import collect_singletons as runner
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent.parent
MOTION = RESEARCH/"flow_story_20261013/motion_groups"
OLD_HEAD = runner.load_pinned(MOTION/"coarse_head.py", runner.COARSE_HEAD_SHA, "_round6_test_old_label")
PROBE = runner.load_probe()
PARENT = PROBE.load_scope_runner()
MERGE = PARENT.load_frozen_merge()
MEASUREMENTS = []


def members():
    manifest = json.loads((HERE/"selection_manifest_v1.json").read_text())
    fit = json.loads((RESEARCH/"flow_story_20261013/recovery/manifests/svs_fit20.json").read_text())
    check = json.loads((RESEARCH/"flow_story_20261013/recovery/manifests/check64.json").read_text())
    dev = json.loads((RESEARCH/"flow_story_20261013/breadth/gate_dev32.json").read_text())
    art = RESEARCH/"flow_story_20261011/preflight/data_protocol/artifacts"
    canonical = json.loads((art/"common_targets.json").read_text())["records"]
    with (art/"train_auxiliary_roles.csv").open() as stream:
        roles = {r["target_id"]: r["auxiliary_role"] for r in csv.DictReader(stream)}
    return manifest, fit, canonical, roles, set(dev["target_ids"]).union(check["target_ids"])


class Tests(unittest.TestCase):
    def test_real_manifest_fit_check_and_group_subset(self):
        data = members()
        all_rows = runner.select_rows(*data)
        self.assertEqual(len(all_rows), 28)
        self.assertEqual([r["target_id"] for g, r in all_rows if g == "fit"], data[1]["target_ids"])
        self.assertEqual(len(runner.select_rows(*data, requested_group="fit")), 20)
        self.assertEqual(len(runner.select_rows(*data, requested_group="check")), 8)
        self.assertEqual(28*len(PROBE.ARMS)+2, 366)
        self.assertEqual(PROBE.full_tiles(PARENT.STAGE_SHAPES[2]), PROBE.CANDIDATES)
        seqs = {g: {r["sequence"] for group, r in all_rows if group == g} for g in ("fit", "check")}
        self.assertFalse(seqs["fit"] & seqs["check"])
        self.assertEqual([len(seqs[x]) for x in ("fit", "check")], [10, 2])
        MEASUREMENTS.append({"members": 28, "fit": 20, "check": 8, "forwards": 366,
                             "sequences": {k: sorted(v) for k, v in seqs.items()}})

    def test_membership_overlap_wrong_roles_and_final_valid_rejected(self):
        data = members()
        changed = copy.deepcopy(data); changed[0]["groups"]["check"][0] = changed[0]["groups"]["fit"][0]
        with self.assertRaisesRegex(ValueError, "overlap"):
            runner.select_rows(*changed)
        changed = copy.deepcopy(data); target = changed[0]["groups"]["check"][0]
        changed[3][target] = "gate_fit"
        with self.assertRaisesRegex(ValueError, "role"):
            runner.select_rows(*changed)
        changed = copy.deepcopy(data); target = changed[0]["groups"]["fit"][0]
        next(r for r in changed[2] if r["target_id"] == target)["protocol_split"] = "valid"
        with self.assertRaisesRegex(ValueError, "train"):
            runner.select_rows(*changed)
        changed = copy.deepcopy(data); changed[0]["groups"]["fit"].reverse()
        with self.assertRaisesRegex(ValueError, "original order"):
            runner.select_rows(*changed)
        changed = copy.deepcopy(data); changed[0]["excluded_ids"].pop()
        with self.assertRaisesRegex(ValueError, "exclusion"):
            runner.select_rows(*changed)

    def test_cell9_exact_matches_old_support_origin_and_empty(self):
        gt = torch.randn(2, 21, 25)
        valid = torch.rand(21, 25) > .4
        valid[:10, :11] = False
        gt[:, ~valid] = float("nan")
        for origin in ((0., 0.), (-1.25, .75)):
            args = dict(feature_hw=(12, 15), feature_stride_pixels=(1.7, 1.8), image_origin_xy=origin)
            actual = runner.cell_targets(gt, valid, cell_size=9, **args)
            reference = OLD_HEAD.make_tile_targets(gt, valid, flow_contract=runner.FLOW_CONTRACT, **args)
            for key in ("flow", "valid", "valid_pixel_count", "support_pixel_count"):
                self.assertTrue(torch.allclose(actual[key].double(), reference[key].double(), atol=0, rtol=0, equal_nan=True), key)
            self.assertEqual(actual["physical_bounds_xyxy"], reference["physical_bounds_xyxy"])
        empty = runner.cell_targets(gt, torch.zeros_like(valid), cell_size=3, feature_hw=(12, 15), feature_stride_pixels=(2, 2))
        self.assertTrue(bool(torch.isnan(empty["flow"]).all()))
        self.assertFalse(bool(empty["valid"].any()))
        bad = gt.clone(); bad[:, valid] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            runner.cell_targets(bad, valid, feature_hw=(12, 15))

    def test_three_cell_full_support_gt_weighted_coverage_and_no_event_filter(self):
        yy, xx = torch.meshgrid(torch.arange(480), torch.arange(640), indexing="ij")
        gt = torch.stack((xx.float()/8, yy.float()/16))
        valid = (xx+yy) % 7 != 0
        valid[:48, :48] = False
        labels = runner.cell_targets(gt, valid, feature_hw=(30, 40))
        audit = runner.verify_cell_partition(labels, gt, valid, OLD_HEAD)
        self.assertEqual(tuple(labels["flow"].shape), (1, 10, 14, 2))
        self.assertEqual(int(labels["support_pixel_count"].sum()), 480*640)
        self.assertEqual(int(labels["valid_pixel_count"].sum()), int(valid.sum()))
        self.assertEqual(int(labels["support_pixel_count"][0, 0, -1]), 48*16)
        self.assertTrue(bool(torch.isnan(labels["flow"][0, 0, 0]).all()))
        # Every voxel is zero: GT support and means must remain unchanged.
        activity = torch.zeros(480, 640, dtype=torch.bool)
        zero = runner.cell_targets(gt, valid & ~activity, feature_hw=(30, 40))
        self.assertTrue(torch.equal(labels["valid_pixel_count"], zero["valid_pixel_count"]))
        self.assertTrue(torch.allclose(labels["flow"], zero["flow"], equal_nan=True))
        MEASUREMENTS.append(audit)

    def test_descriptor_matches_spatial_then_time_merge_order_and_real_tail(self):
        u = torch.randn(10, 1, 30, 40, 384)
        before = u.clone()
        desc = runner.cell_descriptors(u)
        for r in range(4):
            for c in range(5):
                spatial, _ = MERGE.spatial_mean(u[:, :, r*9:r*9+9, c*9:c*9+9], 3)
                self.assertTrue(torch.equal(desc[:, r*3:r*3+3, c*3:c*3+3], spatial.mean(dim=0)))
        self.assertTrue(torch.equal(u, before))
        self.assertEqual(tuple(desc.shape), (1, 10, 14, 384))
        self.assertTrue(torch.equal(desc[:, 0, -1], (u[:, :, :3, 39:].sum((2, 3))/3).mean(0)))
        with self.assertRaisesRegex(ValueError, "complete T10"):
            runner.cell_descriptors(u[:9])

    def test_import_has_no_GPU_or_model_side_effect(self):
        code = ("import os,sys;os.environ['CUDA_VISIBLE_DEVICES']='marker';sys.path.insert(0,"+repr(str(HERE))+");"
                "import collect_singletons as c;c.load_probe();assert os.environ['CUDA_VISIBLE_DEVICES']=='marker';"
                "import torch;assert not torch.cuda.is_initialized()")
        child = subprocess.run([sys.executable, "-B", "-c", code], text=True, capture_output=True, timeout=20)
        self.assertEqual(child.returncode, 0, child.stderr)


def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(20261014)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    sources = [Path(__file__), Path(runner.__file__), runner.PROBE_PATH, PROBE.SCOPE_RUNNER,
               PROBE.BASE/"structured_merge_mlp.py", MOTION/"coarse_head.py", MOTION/"collect_coarse_cache.py"]
    (HERE/"collector_cpu_results.json").write_text(json.dumps({"passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "torch": str(torch.__version__), "GPU_initialized": torch.cuda.is_initialized(), "full_model_run": False,
        "source_sha256": {str(p): runner.sha(p) for p in sources}, "measurements": MEASUREMENTS,
        "scope": "Production membership and label/descriptor functions, exact old9 support crosscheck, all valid including zero-event GT and tail, ordered spatial-then-time pooling; no model/GPU/remote run",
        "errors": [str(x) for x in result.errors], "failures": [str(x) for x in result.failures]}, indent=2)+"\n")
    return 0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1


if __name__ == "__main__":
    raise SystemExit(main())
