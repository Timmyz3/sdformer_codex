#!/usr/bin/env python3
"""R10 target/prefix/service checks using twelve tiny native MLPs, never a model run."""
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
import collect_block2 as runner
import source_contract as contract
import select_members

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(contract.source_path("fixture").parent))
fixture = contract.load_source("fixture", "_round10_native_cpu_fixture")
MEASUREMENTS = []


def build():
    torch.manual_seed(20261015)
    model, targets, _, _, functional = fixture.build()
    shapes = {name: (10, 1, 30, 40, 4) for name in targets}
    return model, targets, shapes, torch.randn(10, 1, 30, 40, 4), functional


def restored(test, model, targets, before, parent):
    test.assertTrue(all(model.get_submodule(name) is module for name, module in targets.items()))
    test.assertTrue(all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules()))
    test.assertEqual(parent.source_digest(model), before)


class Tests(unittest.TestCase):
    def test_all12_block2_masks_prefix_actual_dual_tails_and_services(self):
        collector, probe, parent = runner.load_runtime()
        self.assertEqual(probe.TARGET, contract.TARGET)
        self.assertEqual(collector.load_probe().TARGET, contract.OLD_TARGET)
        model, targets, shapes, x, functional = build()
        before = parent.source_digest(model)
        baseline = probe.forward_probe(model, functional, targets, shapes, x, (), fixture.merge, torch,
                                       capture=True, capture_u=True, capture_tiles=probe.CANDIDATES, capture_bn=True)
        self.assertTrue(torch.equal(baseline[3]["u"], baseline[3]["u"].detach()))
        u_sha = baseline[1]["layers"][contract.TARGET]["input_sha256"]
        self.assertEqual(baseline[1]["summed_fc1_rows"], 59200)
        saved = {(): baseline}
        for tile in probe.CANDIDATES:
            selected = (tile,)
            actual = probe.forward_probe(model, functional, targets, shapes, x, selected, fixture.merge, torch,
                                         capture=True, capture_u=True, expected_u_sha=u_sha, capture_tiles=selected)
            self.assertTrue(torch.equal(actual[3]["u"], baseline[3]["u"]))
            self.assertEqual(actual[1]["summed_fc1_rows"], 59920)
            self.assertEqual(len(actual[1]["all12_observed_fc_rows"]), 12)
            for name, layer in actual[1]["layers"].items():
                self.assertEqual(sum(sum(row) for row in layer["fine_mask"]), int(name == contract.TARGET))
                self.assertFalse(any(layer["fine_mask"][3]))
                self.assertFalse(any(row[4] for row in layer["fine_mask"]))
                self.assertIn([10, 1, 1, 2, 4], layer["observed"]["native_input_shapes"]["coarse"])
                points = 81 if name == contract.TARGET else 0
                representatives = 131 if points else 140
                ledger = layer["selected_service"]
                self.assertEqual(ledger["fine_spatial_points"], points)
                self.assertEqual(ledger["coarse_representative_spatial_points"], representatives)
                self.assertEqual(layer["executed_fc_rows_per_layer"], 10*(points+representatives))
                self.assertEqual(layer["executed_pool_reduction_additions_logical"], 40*(1200-points-representatives))
                self.assertEqual(layer["executed_pool_divisions_logical"], 40*representatives)
                self.assertEqual(layer["executed_repeat_write_elements"], 40*(1200-points))
                self.assertEqual(ledger["external_residual_additions_unchanged"], 48000)
            saved[selected] = actual
            restored(self, model, targets, before, parent)
        for selected in ((), (probe.CANDIDATES[0],)):
            actual = saved[selected]
            dual = probe.forward_probe(model, functional, targets, shapes, x, selected, fixture.merge, torch,
                                       mode="dual_reference", capture=True, expected_u_sha=u_sha)
            self.assertEqual(len(actual[0]), len(dual[0]))
            self.assertEqual(len(actual[0]), 2)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual[0], dual[0])))
            self.assertEqual(set(actual[2]), set(dual[2]))
            self.assertEqual(len(actual[2]), 8)
            self.assertTrue(all(torch.equal(actual[2][name], dual[2][name]) for name in actual[2]))
            self.assertEqual(dual[1]["summed_fc1_rows"], 155200)
            restored(self, model, targets, before, parent)
        MEASUREMENTS.append({"target": probe.TARGET, "candidates_checked": len(probe.CANDIDATES),
            "fixture_shape_all12": [10, 1, 30, 40, 4], "CC_each_fc_rows": 59200,
            "singleton_each_fc_rows": 59920, "dual_each_fc_rows": 155200,
            "actual_dual_pairs": 2, "flow_length_each": 2, "all8_branch_exact": True,
            "prefix_inputs_checked_per_call": 5, "prior_prefix_outputs_checked_per_call": 4,
            "real_3x4_corner_tail_coarse_representative_shape": [10, 1, 1, 2, 4]})

    def test_prefix_and_target_hash_guards_reject_and_cleanup(self):
        collector, probe, parent = runner.load_runtime()
        model, targets, shapes, x, functional = build()
        before = parent.source_digest(model)
        baseline = probe.forward_probe(model, functional, targets, shapes, x, (), fixture.merge, torch)
        with self.assertRaisesRegex(ValueError, "predecision input"):
            probe.forward_probe(model, functional, targets, shapes, x, ((0, 0),), fixture.merge, torch,
                                expected_u_sha="0"*64)
        restored(self, model, targets, before, parent)
        raw = collector.load_probe()
        raw.TARGET = contract.TARGET
        def corrupt(*args, **kwargs):
            out = raw.forward_probe(*args, **kwargs)
            if args[5]:
                out[1]["layers"][contract.PREFIX_NAMES[2]]["input_sha256"] = "0"*64
            return out
        checked = runner.audited_forward(corrupt)
        checked(model, functional, targets, shapes, x, (), fixture.merge, torch)
        with self.assertRaisesRegex(ValueError, "prefix input changed"):
            checked(model, functional, targets, shapes, x, ((0, 0),), fixture.merge, torch)
        restored(self, model, targets, before, parent)
        with patch.object(targets[contract.TARGET].fc2, "forward", side_effect=RuntimeError("injected block2 failure")):
            with self.assertRaisesRegex(RuntimeError, "injected block2"):
                probe.forward_probe(model, functional, targets, shapes, x, ((0, 0),), fixture.merge, torch)
        restored(self, model, targets, before, parent)

    def test_real_geometry_channel_weighted_service_not_fixture_counts(self):
        shapes = {"sttmultires_unet.encoders.swin3d.layers.{}.swin_blocks.{}.mlp".format(s, b):
                  (10, 1, h, w, c) for s, depth, h, w, c in
                  ((0, 2, 120, 160, 96), (1, 2, 60, 80, 192), (2, 6, 30, 40, 384), (3, 2, 15, 20, 768))
                  for b in range(depth)}
        self.assertEqual(runner.expected_rows(shapes, (), "actual"), 153600)
        self.assertEqual(runner.expected_rows(shapes, ((0, 0),), "actual"), 154320)
        self.assertEqual(runner.expected_rows(shapes, (), "dual_reference"), 609600)
        def weighted(selected):
            total = 0
            for name, shape in shapes.items():
                rows = runner.expected_rows({name: shape}, selected, "actual")
                total += rows * 8 * shape[-1]**2
            return total
        self.assertEqual(weighted(()), 69717196800)
        self.assertEqual(weighted(((0, 0),)), 70566543360)
        self.assertEqual(weighted(((0, 0),))-weighted(()), 849346560)
        MEASUREMENTS.append({"production_geometry_only": True, "CC_each_fc_rows": 153600,
            "singleton_each_fc_rows": 154320, "dual_each_fc_rows": 609600,
            "CC_two_fc_weight_products": weighted(()), "singleton_two_fc_weight_products": weighted(((0, 0),)),
            "not_cycles_or_measured_hardware": True})

    def test_membership_exclusions_gap_and_frozen_compatibility(self):
        member = select_members.verify_membership()
        canonical, roles, prior, r9 = select_members.inputs()
        self.assertEqual(member["groups"]["fit"], prior["groups"]["fit"])
        self.assertEqual(len(member["excluded_ids"]), 140)
        self.assertFalse(set(member["groups"]["check"]) & set(member["excluded_ids"]))
        for sequence in select_members.SEQUENCES:
            values = [int(v.rsplit("_", 1)[1]) for v in member["groups"]["check"] if v.startswith(sequence+"_")]
            self.assertEqual(len(values), 4)
            self.assertTrue(all(abs(a-b) >= 3 for i, a in enumerate(values) for b in values[i+1:]))
        collector, _, _ = runner.load_runtime()
        compat = json.loads((HERE/"selection_manifest_v1.json").read_text())
        fit = json.loads(contract.source_path("fit20").read_text())
        chosen = collector.select_rows(compat, fit, canonical, roles,
                                      set(prior["excluded_ids"])-set(prior["groups"]["fit"]))
        self.assertEqual(len(chosen), 28)
        self.assertEqual([row["target_id"] for group, row in chosen if group == "check"], member["groups"]["check"])
        with self.assertRaisesRegex(ValueError, "Duplicate canonical"):
            select_members.select(canonical+[canonical[0]], roles, prior, r9)
        bad = copy.deepcopy(member)
        bad["groups"]["check"][0] = r9["target_ids"][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"bad_member.json"
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "identities/order"):
                select_members.verify_membership(path)

    def test_remote_mapping_and_source_binding(self):
        observed = []
        def fake_sha(path):
            observed.append(str(path))
            key = list(contract.pins())[len(observed)-1]
            return contract.pins()[key]["sha256"]
        with patch.object(contract, "HERE", Path("/data/flow_a/code/research_20261015/block2_allocation_round10")), \
             patch.object(contract, "pins", return_value=json.loads((HERE/"source_pins.json").read_text())["sources"]), \
             patch.object(contract, "sha", side_effect=fake_sha):
            paths = {key: str(contract.source_path(key)) for key in contract.pins()}
        self.assertEqual(paths["canonical"], "/data/flow_a/setup/common_targets.json")
        self.assertEqual(paths["roles"], "/data/flow_a/code/research_20261013/breadth/train_auxiliary_roles.csv")
        self.assertEqual(paths["collector"], "/data/flow_a/code/research_20261014/damage_predictor_round6/collect_singletons.py")
        self.assertEqual(len(contract.verify_all()), 17)

    def test_import_preserves_device_policy_and_new_receipt_binds_sources(self):
        code = ("import os,sys; os.environ['CUDA_VISIBLE_DEVICES']='marker'; sys.path.insert(0,"+repr(str(HERE))+"); "
                "import collect_block2 as r; r.load_runtime(); assert os.environ['CUDA_VISIBLE_DEVICES']=='marker'; "
                "import torch; assert not torch.cuda.is_initialized()")
        child = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(child.returncode, 0, child.stderr)
        _, _, parent = runner.load_runtime()
        sources = [HERE/name for name in ("collect_block2.py", "test_block2_cpu.py", "source_contract.py", "select_members.py")]
        good = {"passed": True, "GPU_initialized": False, "source_sha256": {p.name: contract.sha(p) for p in sources}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"receipt.json"
            path.write_text(json.dumps(good))
            parent.checked_cpu_receipt(path, sources)
            good["source_sha256"]["collect_block2.py"] = "0"*64
            path.write_text(json.dumps(good))
            with self.assertRaisesRegex(ValueError, "current source"):
                parent.checked_cpu_receipt(path, sources)


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    sources = [HERE/name for name in ("collect_block2.py", "test_block2_cpu.py", "source_contract.py", "select_members.py")]
    sources += [Path(fixture.__file__), Path(fixture.fixture.__file__), fixture.fixture.FACTORY]
    receipt = {"schema": "flow_round10_block2_cpu_contract_v1", "passed": result.wasSuccessful(),
        "tests_run": result.testsRun, "torch": torch.__version__, "GPU_initialized": torch.cuda.is_initialized(),
        "full_model_run": False, "source_sha256": {str(p): contract.sha(p) for p in sources},
        "inherited_source_pins_sha256": contract.sha(HERE/"source_pins.json"),
        "membership_sha256": contract.sha(HERE/"membership_v1.json"), "measurements": MEASUREMENTS,
        "scope": "Own block2-target fixture contract; only tiny native C4 MLPs on CPU, not full model, GT, event, quality, GPU or remote validation. Production-shape service is independent arithmetic only.",
        "errors": [str(value) for value in result.errors], "failures": [str(value) for value in result.failures]}
    (HERE/"collector_cpu_results.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return 0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1


if __name__ == "__main__":
    raise SystemExit(main())
